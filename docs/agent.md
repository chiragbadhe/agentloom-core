# Agent

The agent is one class: a model, a set of tools, memory, and the loop that ties
them together.

```ts
import { Agent } from '@agentloom/core';

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  instructions:
    'You are a support engineer. Escalate after two failed attempts.',
});
```

## The loop

Each iteration:

1. Check the budgets. If one is exhausted, stop with the matching
   `stopReason`.
2. Recall relevant long-term memories and assemble the system prompt
   (instructions, run overrides, memories, plan, output instructions).
3. Build the prompt from memory within `contextWindow`, keeping the most recent
   messages intact.
4. Call the model, retrying transient failures, and stream if the provider can.
5. If an output schema is set and the model produced text, validate it. Invalid
   output is repaired by asking the model again.
6. If the model requested tools, validate arguments, run the policy, request
   approval, execute, and append the results.
7. Otherwise the run is done.

Steps 3–7 repeat until the model answers without tool calls or a limit stops it.

## Methods

| Member                                                                   | Purpose                                            |
| ------------------------------------------------------------------------ | -------------------------------------------------- |
| `run(input, options?)`                                                   | Run to completion, resolving with an `AgentResult` |
| `stream(input, options?)`                                                | Run, yielding events as they happen                |
| `plan(input, options?)`                                                  | Produce a plan without running the loop            |
| `reset()`                                                                | Clear conversation memory and the plan             |
| `fork(overrides?)`                                                       | Independent copy with overridden config            |
| `toolSpecs()`                                                            | Tool specs in the shape providers expect           |
| `on` / `once` / `onMany` / `removeAllListeners`                          | Event subscriptions                                |
| `provider`, `model`, `tools`, `memory`, `limits`, `state`, `planTracker` | Read-only state                                    |

`Agent<TState>` carries a user-defined state object through to every tool. See
[shared state](#shared-state).

## Configuration

`model` (or `provider`) is the only required field.

### Model and provider

| Field             | Default   | Notes                                                                                                                                                                          |
| ----------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`           | —         | `"provider:model"`, e.g. `openai:gpt-4o-mini`. Required unless `provider` is given                                                                                             |
| `provider`        | —         | A `ModelProvider` instance, bypassing the registry                                                                                                                             |
| `providerOptions` | `{}`      | Forwarded to the registry factory when `model` resolves a provider                                                                                                             |
| `modelOptions`    | `{}`      | Sampling and budget knobs applied to every call: `temperature`, `topP`, `topK`, `maxTokens`, `stopSequences`, `seed`, `presencePenalty`, `frequencyPenalty`, `reasoningEffort` |
| `name`            | `'agent'` | Used in logs, events, and error messages                                                                                                                                       |

### Behaviour

| Field          | Default                    | Notes                                                         |
| -------------- | -------------------------- | ------------------------------------------------------------- |
| `instructions` | a generic assistant prompt | A string, or a function `({ input, runId, state }) => string` |
| `limits`       | see [limits](#limits)      | Run budgets                                                   |
| `tokenCounter` | heuristic estimate         | Override with a real tokenizer for exact budgeting            |

### Tools

| Field                 | Default        | Notes                                                     |
| --------------------- | -------------- | --------------------------------------------------------- |
| `tools`               | `[]`           | An array of tools, a `ToolRegistry`, or `undefined`       |
| `execution`           | `'sequential'` | `'sequential'` or `'parallel'`                            |
| `toolConcurrency`     | `4`            | Maximum simultaneous tools when parallel                  |
| `toolPolicy`          | —              | Gate that can deny any call                               |
| `approval`            | —              | Human-in-the-loop handler                                 |
| `throwOnToolError`    | `false`        | `true` throws instead of returning the error to the model |
| `maxToolResultLength` | `20000`        | Truncate longer tool output                               |

### Memory

| Field            | Default                                         | Notes                |
| ---------------- | ----------------------------------------------- | -------------------- |
| `memory`         | `InMemoryConversationMemory`                    | Conversation history |
| `longTermMemory` | —                                               | Cross-run store      |
| `longTerm`       | see [memory](memory.md#long-term-memory-wiring) | Recall/store wiring  |

### Other

| Field     | Default                 | Notes                                       |
| --------- | ----------------------- | ------------------------------------------- |
| `output`  | —                       | Default `OutputConfig` for every run        |
| `planner` | off                     | `PlannerConfig` or `false`                  |
| `retry`   | 3 attempts with backoff | `RetryOptions` or `false`                   |
| `hooks`   | `{}`                    | Lifecycle hooks                             |
| `logger`  | no-op                   | Nothing is written to stdout unless you ask |
| `state`   | `undefined`             | Initial value of the shared state object    |

## Run options

Everything here overrides the agent config for a single run.

| Option               | Effect                                                                  |
| -------------------- | ----------------------------------------------------------------------- |
| `signal`             | Cancel the run. Composed with the agent's own controller                |
| `runId`              | Your own id for correlation in logs and traces                          |
| `system`             | Appended to the agent instructions                                      |
| `instructions`       | Replaces the agent instructions entirely                                |
| `model`              | Override the model, e.g. `anthropic:claude-sonnet-4-5`                  |
| `tools`              | Narrow or extend the tools for this run                                 |
| `limits`             | Per-run limit overrides                                                 |
| `outputSchema`       | Validate the final answer against a schema                              |
| `outputInstructions` | Extra instructions for structured output                                |
| `throwOnError`       | `false` returns a result with `stopReason: 'error'` instead of throwing |
| `metadata`           | Attached to events and forwarded to the provider                        |

```ts
const result = await agent.run(input, {
  model: 'openai:gpt-4o-mini',
  limits: { maxIterations: 3 },
  outputSchema: Invoice,
  metadata: { tenant: 'acme' },
});
```

## Results

```ts
interface AgentResult<TData = unknown> {
  output: string; // final answer text
  data: TData | undefined; // validated structured output, when requested
  messages: readonly ModelMessage[];
  steps: readonly AgentStep[];
  iterations: number;
  usage: Usage;
  stopReason: StopReason;
  error: AgentError | undefined;
  runId: string;
  startedAt: number;
  endedAt: number;
  durationMs: number;
}
```

`AgentStep` is one model call plus the tools it triggered: `iteration`, `request`,
`response`, `toolCalls`, `toolResults`, `usage`, `durationMs`, and `text`. Use it
to explain a run, or to assert on behaviour in a test.

`usage` accumulates input, output, cached, and reasoning tokens across every
call in the run. Providers that omit a total are summed from input plus output.

## Stop reasons

`stopReason` is always present, even on failure.

| Value                 | Meaning                                               |
| --------------------- | ----------------------------------------------------- |
| `completed`           | The model produced a final answer with no tool calls  |
| `max_iterations`      | `maxIterations` reached                               |
| `max_tool_calls`      | The tool-call budget was exhausted                    |
| `max_tokens`          | The token budget was exhausted                        |
| `timeout`             | `timeoutMs` elapsed                                   |
| `aborted`             | The caller aborted                                    |
| `max_output_attempts` | The structured-output repair loop ran out of attempts |
| `error`               | Unrecoverable error; see `result.error`               |
| `cancelled`           | A policy or hook stopped the run deliberately         |

A limit stop is not an exception. `run()` resolves with whatever partial output
the model had produced; check `stopReason` when that matters.

## Limits

```ts
new Agent({
  model: 'openai:gpt-4o-mini',
  limits: {
    maxIterations: 8,
    timeoutMs: 60_000,
    maxTotalTokens: 20_000,
    maxTokensPerCall: 2_000,
    maxToolCalls: 12,
    maxToolCallsPerIteration: 4,
    contextWindow: 32_000,
    maxOutputAttempts: 2,
    modelTimeoutMs: 30_000,
  },
});
```

| Limit                      | Default | Notes                                            |
| -------------------------- | ------- | ------------------------------------------------ |
| `maxIterations`            | `10`    | Model calls per run. The main runaway-loop guard |
| `timeoutMs`                | none    | Wall-clock budget for the whole run              |
| `maxTotalTokens`           | none    | Input plus output across the run                 |
| `maxTokensPerCall`         | none    | Output tokens per model call                     |
| `maxToolCalls`             | none    | Tool calls across the run                        |
| `maxToolCallsPerIteration` | none    | Tool calls within one iteration                  |
| `contextWindow`            | `32000` | Hard ceiling for the assembled prompt            |
| `maxOutputAttempts`        | `2`     | Attempts when structured output fails validation |
| `modelTimeoutMs`           | none    | Per-model-call timeout, bounded by `timeoutMs`   |

Limits are checked before the next iteration rather than after a call returns,
so a run cannot overshoot `maxTotalTokens` by a full model call. When
`maxToolCallsPerIteration` clips a batch, the dropped calls still get tool
results — an error result telling the model to finish with what it has — so
call/result pairing stays valid.

`DEFAULT_LIMITS` and `resolveLimits()` are exported if you want to inspect or
extend the defaults.

## Structured output

```ts
import { z } from 'zod';

const Invoice = z.object({
  vendor: z.string(),
  total: z.number(),
  currency: z.enum(['USD', 'EUR', 'GBP']),
});

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  output: { schema: Invoice, instructions: 'Output only the invoice fields.' },
});

const result = await agent.run(invoiceText);
result.data; // typed
```

Or per run with `outputSchema`. `OutputConfig` also takes `name` (used with
native structured output, default `response`) and `strict` (default `true`).

How it works:

- With no tools registered and a provider that supports strict JSON Schema, the
  schema is sent natively and `strict: true` is requested.
- With tools registered, native forcing is skipped on purpose — several
  providers reject tool use combined with a forced response format, and tool
  reliability matters more. JSON mode is requested where available, and local
  validation plus the repair loop do the rest.
- Invalid output produces an `output:invalid` event, the validation issues are
  fed back as a user message, and the model is asked to correct itself. After
  `maxOutputAttempts`, a `ValidationError` is thrown (or returned, with
  `throwOnError: false`).

Anything with `safeParse` works. A hand-written validator only needs that one
method:

```ts
const Profile = {
  safeParse(input: unknown) {
    const ok = typeof (input as any)?.name === 'string';
    return ok
      ? { success: true as const, data: input as { name: string } }
      : {
          success: false as const,
          error: {
            issues: [{ path: [], message: 'expected { name: string }' }],
          },
        };
  },
};
```

`schemaFromJsonSchema(jsonSchema)` builds a validator from raw JSON Schema.

## Shared state

`TState` is a mutable object shared by every tool in a run. It is how you build
stateful multi-step workflows without a module-level global.

```ts
type State = { readonly tickets: string[] };

const agent = new Agent<State>({
  model: 'openai:gpt-4o-mini',
  state: { tickets: [] },
  tools: [openTicket],
});

const openTicket = defineTool<{ title: string }, { id: string }, State>({
  name: 'open_ticket',
  description: 'Open a support ticket.',
  parameters: z.object({ title: z.string() }),
  execute: async ({ title }, context) => {
    const id = await tickets.create(title);
    context.state.tickets.push(id); // same object as agent.state
    return { id };
  },
});
```

Tools receive `context.state`, along with `signal`, `runId`, `call`,
`iteration`, and a pre-tagged `logger`. State is per agent instance, not per run
— `fork()` inherits the same reference, so give each session its own agent.

## Forking and resetting

```ts
const fast = agent.fork({
  model: 'openai:gpt-4o-mini',
  limits: { maxIterations: 4 },
});
agent.reset(); // clears conversation memory and the plan
```

`fork()` returns a shallow copy with a fresh conversation memory, so the two
agents do not share a transcript. `reset()` leaves long-term memory alone.

## System prompt assembly

The prompt is built in this order, joined by blank lines, empty sections
dropped:

1. `instructions` (or the `RunOptions.instructions` replacement)
2. `RunOptions.system`
3. Recalled long-term memories, each as `- (score) content`
4. The rendered plan, when planning is enabled
5. `output.instructions`
6. `RunOptions.outputInstructions`

`hooks.onSystemPrompt` receives the assembled text and may replace it.

## Cancellation

```ts
const controller = new AbortController();
const run = agent.stream('something slow', { signal: controller.signal });

run.abort(); // or controller.abort()
await run.result; // resolves with stopReason: 'aborted'
```

The run's signal is composed from your signal and the agent's own controller, so
both work. `throwIfAborted` is checked at the top of every iteration, the signal
is passed to every provider call and tool, and streaming is raced against it —
so even a provider that ignores `signal` can still be cancelled.

## Error handling

`run()` throws on fatal errors by default. Pass `throwOnError: false` for a
result instead:

```ts
import {
  AbortError,
  TimeoutError,
  ToolDeniedError,
  ValidationError,
} from '@agentloom/core';

try {
  const result = await agent.run(input);
} catch (error) {
  if (error instanceof TimeoutError) retryLater();
  else if (error instanceof AbortError) /* cancelled */ ;
  else if (error instanceof ValidationError) console.error(error.issues);
  else throw error;
}
```

Tool failures are _not_ thrown by default: they are returned to the model as
tool results with `isError: true`, which is what lets a long-running agent
recover from a flaky call. See [errors](errors.md).

## Offline testing

Pass a `provider` instead of a `model` to run without a network:

```ts
const agent = new Agent({
  provider: scriptedProvider([reply('19 * 23 = 437.')]),
  tools: [createCalculatorTool()],
});
```

See [testing](testing.md) and `examples/08-offline-tour.ts`.
