# Streaming & events

Two ways to observe a run:

- **Events** (`agent.on(...)`) describe what happened. Observational only.
- **Hooks** (`hooks: {...}`) can influence what happens — rewrite the prompt,
  rewrite the request, veto, or abort.

## Streaming

```ts
const run = agent.stream('Write a haiku about retries.');

for await (const event of run) {
  if (event.type === 'text-delta') process.stdout.write(event.text);
}

const result = await run.result;
```

`stream()` and `run()` share one implementation, so streaming is not a separate
code path with different behaviour. The run starts immediately; consume events
whenever you are ready.

`AgentStream` is also a typed emitter, which is handy when streaming into an
existing event bus:

```ts
const run = agent.stream('...');
run.on('tool-result', ({ name, content }) => console.log(name, content));
run.on('error', ({ error, fatal }) => report(error, { fatal }));

const result = await run.result;
```

When you only want the text:

```ts
const run = agent.stream('...');
const answer = run.text(); // start this before awaiting result
const result = await run.result;
console.log(await answer);
```

`text()` consumes the queue, so call it before awaiting `result` — a second
consumer would compete for the same events.

### Stream events

A flat discriminated union; `switch (event.type)` is exhaustive.

| `type`            | Payload                                             | When                                  |
| ----------------- | --------------------------------------------------- | ------------------------------------- |
| `start`           | `runId`, `agentName`, `model`, `input`, `tools`     | The run began                         |
| `text-delta`      | `text`, `iteration`                                 | A text chunk arrived                  |
| `text`            | `text`, `iteration`                                 | The complete text for one model call  |
| `reasoning-delta` | `text`, `iteration`                                 | Extended-thinking output              |
| `tool-call`       | `name`, `args`, `iteration`                         | About to run a tool                   |
| `tool-result`     | `name`, `content`, `isError`, `durationMs`          | A tool finished                       |
| `iteration`       | `iteration`, `toolCallCount`, `usage`, `durationMs` | A loop iteration ended                |
| `plan`            | `plan`, `completed`                                 | The plan was produced or updated      |
| `retry`           | `attempt`, `delayMs`, `error`                       | A retry is scheduled                  |
| `error`           | `error`, `fatal`, `iteration`                       | Any error, including recoverable ones |
| `done`            | `result`                                            | The run finished; the last event      |

`text-delta` is the token stream; `text` is the assembled answer for one model
call. Concatenating deltas of every iteration gives the full output, which is
why `result.output` may differ from a naive delta concatenation when the model
produced a tool call mid-answer.

`text-delta` only appears when the provider supports real streaming.
`BaseProvider` buffers a non-streaming provider into events, so consumers see
uniform events either way.

### Cancellation

```ts
const controller = new AbortController();
const run = agent.stream('...', { signal: controller.signal });

run.abort(); // or controller.abort('user cancelled')
const result = await run.result; // stopReason: 'aborted'
```

`run.aborted` reflects current state. `run.close()` releases resources without
waiting.

## Agent events

Events are richer than stream events and carry run context like `runId` and
`iteration`. All are synchronous, and a throwing listener never interrupts a
run.

```ts
const off = agent.on('model:end', ({ usage, latencyMs }) => {
  metrics.timing('model', latencyMs, { tokens: usage.totalTokens });
});

off(); // unsubscribe
```

| Event             | Fires when                                                                            |
| ----------------- | ------------------------------------------------------------------------------------- |
| `agent:start`     | The run begins. Carries `input`, `provider`, `model`, `tools`, `limits`               |
| `agent:prompt`    | The prompt was assembled. Carries `messages`, `estimatedTokens`, `system`, `recalled` |
| `iteration:start` | A loop iteration begins                                                               |
| `iteration:end`   | An iteration finishes. Carries `toolCallCount`, `usage`, `durationMs`                 |
| `model:start`     | About to call the provider. Carries `request`, `attempt`                              |
| `model:delta`     | A streamed chunk arrived. `reasoning: true` for thinking output                       |
| `model:end`       | The model returned. Carries `result`, `durationMs`                                    |
| `tool:start`      | A tool is about to run                                                                |
| `tool:end`        | A tool succeeded                                                                      |
| `tool:error`      | A tool failed and the error went back to the model                                    |
| `retry`           | A retry is scheduled                                                                  |
| `plan`            | A plan was produced or updated                                                        |
| `memory`          | Long-term recall or store                                                             |
| `output:invalid`  | Structured output failed validation; a repair pass is starting                        |
| `error`           | Any error. `fatal` distinguishes an unrecoverable one                                 |
| `agent:end`       | The run finishes. Carries `stopReason`, `usage`, `durationMs`                         |

Subscriptions: `on`, `once`, `onMany({...})`, `removeAllListeners(event?)`.

```ts
agent.onMany({
  'tool:end': ({ toolName, durationMs }) =>
    metrics.timing(`tool.${toolName}`, durationMs),
  retry: ({ delayMs }) => metrics.inc('llm.retry'),
  error: ({ error, scope, fatal }) =>
    logger.warn({ error: error.message, scope, fatal }),
});
```

A full run produces `agent:start`, then a repeating
`iteration:start` → `agent:prompt` → `model:start` → `model:end` →
(`tool:start` → `tool:end` | `tool:error`)* → `iteration:end`, then
`agent:end`.

## Hooks

Hooks are the "act" side of observability.

```ts
const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  hooks: {
    onAgentStart: ({ input }) => {
      if (input.includes('secret')) throw new Error('blocked by policy hook');
    },
    onSystemPrompt: ({ defaultPrompt, state }) =>
      `${defaultPrompt}\nEnvironment: ${state.environment}`,
    onBeforeModel: ({ request, state }) => ({
      ...request,
      metadata: { ...request.metadata, tenant: state.tenant },
    }),
    onToolError: ({ toolName, error }) =>
      console.warn(`${toolName}: ${error.message}`),
  },
});
```

| Hook                                         | Can                                                                |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `onAgentStart`                               | Abort the run by throwing                                          |
| `onSystemPrompt`                             | Replace the assembled system prompt. Return `undefined` to keep it |
| `onPrompt`                                   | Inspect the final prompt                                           |
| `onBeforeModel`                              | Replace the request. Throwing aborts the run                       |
| `onModelStart`, `onModelDelta`, `onModelEnd` | Observe a model call                                               |
| `onIterationStart`, `onIterationEnd`         | Observe an iteration                                               |
| `onToolStart`, `onToolEnd`, `onToolError`    | Observe tool execution                                             |
| `onRetry`                                    | Observe retries                                                    |
| `onMemory`                                   | Observe long-term recall and store                                 |
| `onOutputInvalid`                            | Observe a validation failure                                       |
| `onBeforeReturn`                             | Rewrite or veto the final answer. Throwing aborts                  |
| `onAgentEnd`                                 | Observe the final result                                           |
| `onError`                                    | Observe any error                                                  |

Vetoing hooks are `onAgentStart`, `onBeforeModel`, and `onBeforeReturn` —
listed in `VETOING_HOOKS`. A throw from one of those ends the run with an
`AgentError`. Every other hook is observational: a throw is swallowed so
logging can never break a run.

`onBeforeReturn` can replace the answer rather than abort it:

```ts
hooks: {
  onBeforeReturn: ({ output, state }) =>
    state.redact ? output.replace(/\b\d{16}\b/g, '****') : undefined,
}
```

## Logging

Nothing is written to stdout unless you ask. Wire up a logger when you want
run-level diagnostics:

```ts
import { createConsoleLogger, MemoryLogSink } from '@agentloom/core';

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  logger: createConsoleLogger({
    level: 'info',
    base: { service: 'support-bot' },
  }),
});

// In tests, swap in a sink you can assert on.
const sink = new MemoryLogSink({ level: 'debug', limit: 100 });
new Agent({ model: 'openai:gpt-4o-mini', logger: sink });
sink.records; // inspectable LogRecord[]
```

`createConsoleLogger` emits one JSON line per record, so logs stay greppable
without a logging dependency. Levels are `debug`, `info`, `warn`, `error`,
`silent`. `logger.child(fields)` derives a logger that stamps every record with
extra fields — the agent tags its own logger with the agent name, and each tool
gets one tagged with the tool, run, and iteration.

`noopLogger()` is the default.

## Putting it together

A CLI that streams, times, and traces:

```ts
import { Agent, createConsoleLogger } from '@agentloom/core';

const agent = new Agent({
  name: 'support',
  model: 'openai:gpt-4o-mini',
  logger: createConsoleLogger({ level: 'info' }),
  limits: { maxIterations: 8, timeoutMs: 60_000, maxTotalTokens: 20_000 },
});

agent.on('tool:end', ({ toolName, durationMs }) =>
  console.error(`[tool] ${toolName} ${durationMs}ms`),
);

const run = agent.stream(question, { signal: AbortSignal.timeout(30_000) });

for await (const event of run) {
  switch (event.type) {
    case 'text-delta':
      process.stdout.write(event.text);
      break;
    case 'tool-call':
      console.error(`\n[tool] ${event.name}(${JSON.stringify(event.args)})`);
      break;
    case 'retry':
      console.error(`\n[retry] waiting ${event.delayMs}ms`);
      break;
    case 'error':
      console.error(`\n[error] ${event.error.message}`);
      break;
  }
}

const result = await run.result;
console.error(`\nstopReason=${result.stopReason} in ${result.durationMs}ms`);
```

## More

- [Agent](agent.md) — limits, cancellation, results
- [Errors](errors.md) — what the `error` events carry
