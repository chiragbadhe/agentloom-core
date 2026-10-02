# @agentloom/core

Production-ready primitives for building AI agents: a provider abstraction
(OpenAI, Anthropic, Gemini, Ollama, or your own), tools with policy and approval
gates, conversation and long-term memory, planning, structured output, and a
hardened agent loop with budgets, retries, and cancellation.

No runtime dependencies. `zod` is an optional peer dependency — any validator
with a `safeParse` method works.

```bash
npm install @agentloom/core zod
```

## Quickstart

```ts
import { Agent, createCalculatorTool } from '@agentloom/core';

const agent = new Agent({
  name: 'assistant',
  model: 'openai:gpt-4o-mini', // API key read from OPENAI_API_KEY
  instructions: 'You are a concise assistant. Use tools instead of guessing.',
  tools: [createCalculatorTool()],
  limits: { maxIterations: 8, timeoutMs: 60_000 },
});

const result = await agent.run('What is 17 * 23?');

console.log(result.output);
console.log(result.stopReason, result.usage.totalTokens, result.durationMs);
```

A model reference is `"provider:model"`. `openai`, `anthropic`, `google`, and
`ollama` are pre-registered; anything else you register yourself.

## Why this and not a framework

The agent loop is one class you can read. Every moving part is an interface you
can swap or ignore:

- **Provider** — one `complete` method and one `capabilities` record. Extend
  `BaseProvider` to inherit retries, timeouts, and abort handling.
- **Tools** — a name, a description, and an argument schema. The same schema
  validates arguments locally and reaches the model as JSON Schema.
- **Memory** — an ordered message list with a token budget. Long-term memory is
  three methods, so it maps onto whatever store you already run.
- **Observability** — 16 typed events plus hooks that can rewrite the prompt,
  the request, or the final answer.

Structured output is validated locally and repaired by a retry loop, so it
works on providers with no native JSON-schema enforcement.

## What it handles for you

| Concern         | Behaviour                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| Runaway loops   | `maxIterations` (default 10) checked before each iteration                                                |
| Cost overruns   | `maxTotalTokens`, `maxTokensPerCall`, `maxToolCalls`                                                      |
| Hangs           | `timeoutMs` for the run, `modelTimeoutMs` per call, `timeoutMs` per tool                                  |
| Flaky providers | Retry with exponential backoff and jitter; honours `Retry-After`                                          |
| Cancellation    | `AbortSignal` composed per run; `stream().abort()`; aborts even a wedged provider                         |
| Bad tool calls  | Unknown names, bad arguments, denials, and timeouts all return readable errors the model can recover from |
| Malformed JSON  | Validation issues are fed back and the model is asked to repair, up to `maxOutputAttempts`                |
| SSRF / escapes  | Built-in HTTP tool refuses private hosts; filesystem tool is jailed to one root                           |

## Documentation

| Guide                                              | Contents                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| [Getting started](docs/getting-started.md)         | Install, keys, first agent, running the examples                         |
| [Agent](docs/agent.md)                             | Config reference, run options, limits, results, stop reasons             |
| [Tools](docs/tools.md)                             | Defining tools, registry, execution modes, policies, approval, built-ins |
| [Providers](docs/providers.md)                     | Built-in providers, model references, custom providers, HTTP layer       |
| [Memory](docs/memory.md)                           | Conversation windowing, summarization, long-term recall                  |
| [Streaming & events](docs/streaming-and-events.md) | Stream API, event catalogue, hooks, logging                              |
| [Planning](docs/planning.md)                       | Plan generation, `update_plan`, custom planners                          |
| [Errors](docs/errors.md)                           | Error hierarchy, codes, retryability, recovery strategies                |
| [Testing](docs/testing.md)                         | Scripted providers, offline testing, injection points                    |

Runnable examples live in [`examples/`](examples), numbered in reading order:

| File                       | Topic                                                    |
| -------------------------- | -------------------------------------------------------- |
| `01-basic.ts`              | Smallest useful agent                                    |
| `02-tools.ts`              | Custom tools, built-ins, policy, approval                |
| `03-streaming.ts`          | Streaming, events, cancellation                          |
| `04-structured-output.ts`  | Zod-validated output and the repair loop                 |
| `05-memory.ts`             | Conversation windowing, summarization, long-term recall  |
| `06-planning-and-hooks.ts` | Planning, hooks, budgets, events, forks                  |
| `07-custom-provider.ts`    | Bringing your own provider                               |
| `08-offline-tour.ts`       | The whole loop with no network — also a testing template |

```bash
npx tsx examples/01-basic.ts
MODEL=anthropic:claude-sonnet-4-5 npx tsx examples/01-basic.ts
```

`08-offline-tour.ts` needs no API key at all.

## Requirements

Node 18.17 or newer (for global `fetch`). ESM and CJS are both published, with
type declarations for each.

## License

MIT
