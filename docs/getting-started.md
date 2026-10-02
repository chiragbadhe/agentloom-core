# Getting started

## Install

```bash
npm install @agentloom/core
```

`zod` is an optional peer dependency (`^3.23` or `^4`). Install it if you want
Zod schemas for tools and structured output; nothing else needs it.

```bash
npm install zod
```

Node 18.17 or newer is required, for global `fetch`.

## API keys

Each built-in provider reads its conventional environment variable, so there is
nothing to wire up:

| Provider    | Variable                             | Default model       |
| ----------- | ------------------------------------ | ------------------- |
| `openai`    | `OPENAI_API_KEY`                     | `gpt-4o-mini`       |
| `anthropic` | `ANTHROPIC_API_KEY`                  | `claude-sonnet-4-5` |
| `google`    | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-2.0-flash`  |
| `ollama`    | none (local server)                  | `llama3.2`          |

```bash
export OPENAI_API_KEY=sk-...
```

A blank value counts as absent, so an empty variable never produces a malformed
`Authorization` header. See [providers](providers.md) for explicit keys,
proxies, and custom hosts.

## Your first agent

```ts
// hello.ts
import { Agent } from '@agentloom/core';

const agent = new Agent({
  name: 'assistant',
  model: 'openai:gpt-4o-mini',
  instructions:
    'You are a concise assistant. Answer in at most three sentences.',
});

const result = await agent.run('Why is TypeScript structural?');

console.log(result.output);
console.log(`stopReason=${result.stopReason} in ${result.durationMs}ms`);
```

```bash
npx tsx hello.ts
```

Two things are worth noticing in that snippet.

`Agent` requires either `model` or `provider` — everything else has a default.
`new Agent({ model: 'openai:gpt-4o-mini' })` is a working agent.

`run()` returns an `AgentResult`, not a string. `result.output` is the answer;
`stopReason`, `usage`, `steps`, and `messages` are there when you need them. A
run that ends because it hit a budget still returns a result — see
[stop reasons](agent.md#stop-reasons).

## Adding a tool

```ts
import { Agent, defineTool } from '@agentloom/core';
import { z } from 'zod';

const getWeather = defineTool({
  name: 'get_weather',
  description: 'Current weather for a city.',
  parameters: z.object({
    city: z.string(),
    unit: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  execute: async ({ city, unit }) => {
    // Your real API call here.
    return {
      city,
      unit,
      temperature: unit === 'celsius' ? 17 : 63,
      sky: 'light rain',
    };
  },
  serialize: (r) => `${r.city}: ${r.temperature}° — ${r.sky}`,
});

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  tools: [getWeather],
});

const { output } = await agent.run('Is it raining in Oslo?');
```

`parameters` does double duty: it validates what the model sends and it is
converted to JSON Schema for the API. The declared return type is what reaches
the model. Details in [tools](tools.md).

## Multi-turn conversations

An `Agent` holds its conversation history, so a chatbot is one agent across many
runs:

```ts
await agent.run('My name is Sam.');
const second = await agent.run('What is my name?');
```

Call `agent.reset()` to clear history between unrelated conversations, and
`agent.fork({ model: 'anthropic:claude-sonnet-4-5' })` for an independent copy
with a different model. Sessions keyed by user or conversation id want one agent
and one memory each — see [memory](memory.md#per-user-sessions).

## Structured output

Pass any validator with a `safeParse` method:

```ts
const Invoice = z.object({
  vendor: z.string(),
  total: z.number(),
  currency: z.enum(['USD', 'EUR', 'GBP']),
});

const result = await agent.run<z.infer<typeof Invoice>>(invoiceText, {
  outputSchema: Invoice,
});

result.data?.vendor; // typed
result.output; // the raw JSON text
```

When validation fails, the issues are fed back to the model and it is asked to
repair the answer. See [structured output](agent.md#structured-output).

## Streaming

```ts
const run = agent.stream('Write a haiku about retries.');

for await (const event of run) {
  if (event.type === 'text-delta') process.stdout.write(event.text);
}

const result = await run.result;
```

`run()` and `stream()` share one implementation, so behaviour does not drift
between them. Full event list in [streaming & events](streaming-and-events.md).

## Cancellation

```ts
const controller = new AbortController();
setTimeout(() => controller.abort(), 5_000);

const result = await agent.run('Do something slow.', {
  signal: controller.signal,
  throwOnError: false, // resolve with stopReason: 'aborted' instead of throwing
});
```

Cancellation is propagated into provider calls and tool execution. A provider
that ignores `signal` is still cancellable, because the loop races it against
the abort signal.

## Guardrails

Anything unattended should set limits:

```ts
const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  limits: {
    maxIterations: 8,
    maxTotalTokens: 20_000,
    maxToolCalls: 12,
    timeoutMs: 60_000,
  },
});
```

Limits are checked _before_ spending, not after, so a run cannot overshoot its
token budget by a whole model call. The full table is in
[limits](agent.md#limits).

## Running the examples

```bash
npx tsx examples/01-basic.ts                          # openai:gpt-4o-mini
MODEL=anthropic:claude-sonnet-4-5 npx tsx examples/01-basic.ts
MODEL=ollama:llama3.2 npx tsx examples/01-basic.ts
```

| File                       | Topic                                      |
| -------------------------- | ------------------------------------------ |
| `01-basic.ts`              | Smallest useful agent                      |
| `02-tools.ts`              | Custom tools, built-ins, policy, approval  |
| `03-streaming.ts`          | Streaming, events, cancellation            |
| `04-structured-output.ts`  | Zod-validated output and the repair loop   |
| `05-memory.ts`             | Windowing, summarization, long-term recall |
| `06-planning-and-hooks.ts` | Planning, hooks, budgets, events, forks    |
| `07-custom-provider.ts`    | Bringing your own provider                 |
| `08-offline-tour.ts`       | Whole loop, no network                     |

## Next

- [Agent](agent.md) — every config field, run option, and result field
- [Tools](tools.md) — policies, approval, execution modes, built-ins
- [Providers](providers.md) — model references, custom providers, proxies
- [Memory](memory.md) — windowing strategies and long-term recall
- [Streaming & events](streaming-and-events.md) — events, hooks, logging
