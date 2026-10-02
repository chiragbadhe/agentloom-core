# Tools

A tool is a name, a description, and a typed `execute` function. The argument
schema does double duty: it validates what the model sends, and it becomes the
JSON Schema the provider API receives.

```ts
import { Agent, defineTool } from '@agentloom/core';
import { z } from 'zod';

const getWeather = defineTool({
  name: 'get_weather',
  description:
    'Current weather for a city. Call this before discussing weather.',
  parameters: z.object({
    city: z.string().min(1),
    unit: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  execute: async ({ city, unit }) => {
    const res = await fetch(`https://api.example.com/weather?city=${city}`);
    return res.json();
  },
  serialize: (r) => `${r.city}: ${r.temperature}, ${r.conditions}`,
});

const agent = new Agent({ model: 'openai:gpt-4o-mini', tools: [getWeather] });
```

`TArgs` and `TResult` are inferred from `parameters` and `execute`, so the
registry and the model-facing schema cannot drift apart. `tool` is an alias for
`defineTool`.

## ToolDefinition

| Field              | Notes                                                                         |
| ------------------ | ----------------------------------------------------------------------------- |
| `name`             | Required. Letters, digits, `_`, `-`; max 64 chars, must start with a letter   |
| `description`      | Required. This is what the model reads when deciding to call                  |
| `parameters`       | Argument validator. Omit for no-argument tools                                |
| `jsonSchema`       | Explicit JSON Schema for the API, overriding the conversion from `parameters` |
| `execute`          | Required. `(args, context) => result`                                         |
| `timeoutMs`        | Per-call deadline. Inherits the agent timeout                                 |
| `maxRetries`       | Retry budget for transient failures inside the tool. Default `0`              |
| `hidden`           | Keep in the registry but omit from the model's tool list                      |
| `requiresApproval` | `true`, or a predicate over the validated args                                |
| `serialize`        | Custom rendering of the result. Return `undefined` for the default            |
| `metadata`         | Free-form data for hooks and telemetry                                        |

`defineTool` validates the shape up front and throws `ConfigurationError` for an
empty name, a bad name, a missing description, or a missing `execute`.

## Tool context

The second argument to `execute`:

```ts
interface ToolContext<TState> {
  signal: AbortSignal; // aborts when the run is cancelled or its deadline passes
  runId: string;
  state: TState; // the agent's shared mutable state
  call: ToolCall; // the raw call that triggered this
  iteration: number; // 1-based loop iteration
  logger: Logger; // already tagged with agent/run/tool names
}
```

Pass `signal` to anything cancellable — `fetch`, a database query, a child
process. A tool that ignores it will still be abandoned when the run ends, but
it will keep running in the background.

## Serialization

By default results are JSON-encoded (strings pass through). `serialize` lets you
send something token-cheap and model-friendly:

```ts
serialize: (r) => `${r.city}: ${r.temperature}°C, ${r.sky}`,
serialize: ({ rows }) => rows.slice(0, 20).map((r) => `${r.id} ${r.name}`).join('\n'),
```

Return `undefined` to fall back to the default rendering. Output longer than
`maxToolResultLength` (default 20 000 chars) is truncated with a note, so one
chatty tool cannot blow the context window.

## Registry

`ToolRegistry` is a name-indexed collection with scoping helpers:

```ts
import { ToolRegistry } from '@agentloom/core';

const registry = new ToolRegistry([getWeather, sendEmail]);

registry.names(); // all names, hidden included
registry.size;
registry.has('get_weather');
registry.get('get_weather');
registry.add(anotherTool); // add or replace; replacement keeps position
registry.remove('send_email');

const scoped = registry.only(['get_weather']); // a narrowed view
const safe = registry.exclude(['send_email']); // everything else
registry.rename({ sendEmail: 'send_email' }); // in place, for snake_case models
const copy = registry.clone(); // safe to mutate
```

`only()` throws on an unknown name — a typo in a scoping call should fail loudly
rather than silently drop a capability. `toSpecs()` projects to the
provider-facing shape, omitting hidden tools. `toToolRegistry()` normalizes
whatever you pass as `tools` (array, registry, single tool, `undefined`).

## Execution

```ts
new Agent({
  model: 'openai:gpt-4o-mini',
  tools: [getWeather, sendEmail],
  execution: 'parallel', // default 'sequential'
  toolConcurrency: 4, // default 4
});
```

`sequential` runs one call at a time, in the order the model requested them.
`parallel` runs them concurrently up to `toolConcurrency`, but results are still
returned in the original order so call/result pairing stays valid.

Everything a tool can get wrong — unknown name, invalid arguments, denied by
policy, approval refused, timeout, thrown exception — is converted into a tool
result the model can read and recover from. That is what stops a flaky tool from
killing a long-running agent.

Set `throwOnToolError: true` to make failures throw instead. Useful when you
want the run to end rather than letting the model retry blind.

`ToolExecutor` is exported if you need the same machinery outside the agent
loop: it takes a registry, mode, policy, approval handler, timeouts, retries,
and lifecycle hooks.

## Policies

A policy runs before every call, after arguments are parsed and before approval:

```ts
import { type ToolPolicy } from '@agentloom/core';

const policy: ToolPolicy<State> = (call, context) => {
  if (call.name === 'deleteFile') {
    return { action: 'deny', reason: 'destructive tools are disabled' };
  }
  if (context.state.readOnly && call.name === 'writeFile') {
    return { action: 'deny', reason: 'read-only session' };
  }
  return { action: 'allow' };
};
```

A denial becomes a tool error the model sees, so it can pick another route. A
policy that _throws_ denies the call — the gate fails closed.

## Approval

Mark a tool as requiring approval and supply a handler:

```ts
const sendEmail = defineTool({
  name: 'send_email',
  description: 'Send an email. Requires explicit approval.',
  requiresApproval: true, // or (args) => args.to.endsWith('@partner.com')
  parameters: z.object({
    to: z.string().email(),
    subject: z.string(),
    body: z.string(),
  }),
  execute: async ({ to, subject, body }) => mailer.send({ to, subject, body }),
});

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  tools: [sendEmail],
  approval: async (request, { state }) => {
    // request: { toolName, args, call, runId, iteration, summary }
    return await askOperator(`${request.toolName}: ${request.summary}`);
  },
});
```

No handler, or `false` returned, produces a `ToolApprovalRequiredError` back to
the model. A real implementation would prompt a human or check a permission
store — the handler is async and receives the run's `signal`, so it can be
cancelled too.

Policies and approval compose: the policy runs first, so denying in the policy
skips the approval prompt entirely.

## Built-in tools

### `createCalculatorTool(options?)`

Arithmetic with `+ - * / % ^`, parentheses, constants (`pi`, `e`, `tau`), and
around twenty functions (`sqrt`, `round`, `max`, `log`, …).

```ts
const calculator = createCalculatorTool({ maxLength: 1000, maxDigits: 10 });
```

Implemented as a recursive-descent parser, deliberately not `eval`: tool
arguments are model output, so they are untrusted input. There is no property
access, no assignment, and no path to the host scope. `evaluateExpression` and
`supportedFunctions()` are exported if you want the evaluator directly.

### `createDateTimeTool()`

Actions `now`, `parse`, `format`, `add`, `diff` with IANA timezones and
`iso | date | time | rfc2822 | relative | unix` output formats. `parseDuration`
handles compact forms like `-3d`, `+2h30m`, `90s`.

### `createHttpTool(options?)` / `createFetchUrlTool(options?)`

`get`, `post`, `head` against real URLs. Security defaults: private and loopback
hosts are refused (SSRF), responses are size-capped, and the body is truncated
before it reaches the model.

```ts
const http = createHttpTool({
  allowedHosts: ['api.example.com', '*.internal.example.com'],
  maxBytes: 1_000_000,
  timeoutMs: 15_000,
  maxResponseChars: 4_000,
});
```

Omitting `allowedHosts` allows every public host. Only do that in a sandbox.
`isUrlAllowed()` is exported so you can reuse the check. `createFetchUrlTool`
is the single-URL convenience variant.

### `createFileSystemTool(options)`

`read`, `write`, `append`, `list`, `delete`, `exists`, `stat`, jailed to one
root.

```ts
const fs = createFileSystemTool({ root: process.cwd(), allowWrite: false });
```

`root` is **required** — there is no unrestricted mode, because a
model-controlled path is a sandbox escape. Every path is resolved and checked
against the root _after_ symlink resolution, so `../../etc/passwd` and symlink
escapes are both rejected. Read-only unless `allowWrite: true`; read and write
sizes are capped. `resolveInsideRoot()` is exported for your own checks.

### `createSleepTool(maxMs?)`

Pause for up to 60 seconds by default, for backoff or waiting on external
events. Resolves early when the run is aborted, so cancellation is never blocked
by a long sleep.

## Narrowing tools per run

Build the registry once and pass a scoped view when a run should have fewer
capabilities:

```ts
const registry = new ToolRegistry([searchOrders, issueRefund]);

const agent = new Agent({ model: 'openai:gpt-4o-mini', tools: registry });

// Read-only for this run.
await agent.run('...', { tools: registry.exclude(['issueRefund']) });
```

The agent also exposes `agent.toolSpecs()` for the provider-facing projection of
its current tools.

## More

- [Agent](agent.md) — `toolPolicy`, `approval`, and execution config in context
- [Streaming & events](streaming-and-events.md) — `tool:start`, `tool:end`, `tool:error`
- [Testing](testing.md) — asserting on tool calls without a network
