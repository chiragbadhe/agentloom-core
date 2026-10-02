# Memory

Two separate concerns:

- **Conversation memory** — the ordered message list sent to the model on every
  iteration, windowed to fit a token budget.
- **Long-term memory** — facts worth remembering across runs, recalled at the
  start of a run and stored at the end.

## Conversation memory

The default is `InMemoryConversationMemory`: an array plus token-budgeted
trimming.

```ts
import { Agent, InMemoryConversationMemory } from '@agentloom/core';

const memory = new InMemoryConversationMemory({
  maxMessages: 40,
  maxTokens: 12_000,
  keepRecent: 6,
});

const agent = new Agent({ model: 'openai:gpt-4o-mini', memory });
```

| Option            | Default     | Notes                                                 |
| ----------------- | ----------- | ----------------------------------------------------- |
| `maxMessages`     | `100`       | Soft ceiling on stored messages; oldest evicted first |
| `maxTokens`       | `8000`      | Token budget for `build()`                            |
| `keepRecent`      | `4`         | Recent messages protected from trimming               |
| `strategy`        | `'sliding'` | `'sliding'` or `'summarize'`                          |
| `compact`         | —           | Required by `'summarize'`                             |
| `tokenCounter`    | heuristic   | Override with a real tokenizer                        |
| `initialMessages` | —           | Seed the store with existing history                  |

Inspection and control:

```ts
memory.length; // stored messages
memory.messages(); // oldest first
memory.tokenCount(); // estimated tokens
memory.limits; // { maxTokens, keepRecent, strategy }
memory.state(); // { messages, droppedCount, summaryCount, tokens, trimmed }
memory.clear();
```

The protected tail matters: `build()` never trims inside `keepRecent`, even if
that overshoots the budget. A half-trimmed tool exchange (a call without its
result) breaks most providers, so a slightly over-long prompt is the lesser evil.

## Summarization

`sliding` drops old turns. `summarize` replaces them with a model-written
digest, which preserves detail at the cost of an extra call.

```ts
import { createSummaryCompactor } from '@agentloom/core';

const memory = new InMemoryConversationMemory({
  strategy: 'summarize',
  compact: createSummaryCompactor(async (prompt) => {
    const result = await summarizer.run(prompt);
    return result.output;
  }),
});
```

`createSummaryCompactor` builds the digest prompt and wraps the summary as a
single system message. Returning an empty string is treated as a no-op, so the
plain sliding window is used instead of silently discarding history. A throwing
compactor raises `MemoryError`.

The helpers behind it are exported for custom strategies:
`transcriptOf(messages)` renders a transcript, `summaryMessage(text)` wraps a
digest, and `tokenize`/`truncate` are available from the utils barrel.

## Custom conversation memory

Implement `ConversationMemory` to persist a conversation or change the windowing
strategy. `add`, `messages`, and `clear` plus `length` and `build` are all it
needs:

```ts
import type { ConversationMemory, ModelMessage } from '@agentloom/core';

class RedisMemory implements ConversationMemory {
  length = 0;

  async add(message: ModelMessage) { /* append */ }
  async addAll(messages: readonly ModelMessage[]) { /* append many */ }
  async messages() { /* oldest first */ return []; }
  async clear() { /* drop everything */ }

  async build({ system, maxTokens, tokenCounter, keepRecent }) {
    // Apply your own windowing, then prepend the system prompt.
    return system ? [{ role: 'system', content: system }, ...] : [];
  }
}
```

Methods may be sync or async. `build()` receives the run's `signal` so
compaction can be cancelled.

## Token counting

The default counter is a calibrated heuristic: characters per token for latin
text, denser for CJK, plus per-message template overhead. It is tuned to
_overestimate_ slightly, because overshooting a budget is safe and
undershooting it is not.

For exact counts, pass a real tokenizer:

```ts
import { estimateTokens, type TokenCounter } from '@agentloom/core';

const counter: TokenCounter = (input) => {
  const text = typeof input === 'string' ? input : JSON.stringify(input);
  return myTokenizer.count(text);
};

new Agent({ model: 'openai:gpt-4o-mini', tokenCounter: counter });
```

`estimateTokens`, `estimateValueTokens`, and `estimateMessageTokens` are
exported if you want the built-in behaviour for one input shape.

## Per-user sessions

Memory belongs to the agent instance, so one agent and one memory per session:

```ts
const agents = new Map<string, Agent>();

function agentFor(userId: string): Agent {
  let agent = agents.get(userId);
  if (agent === undefined) {
    agent = new Agent({
      model: 'openai:gpt-4o-mini',
      memory: new InMemoryConversationMemory({ maxTokens: 12_000 }),
      longTermMemory: sharedLongTerm, // often shared across users
      state: { userId },
    });
    agents.set(userId, agent);
  }
  return agent;
}
```

Persist `memory.messages()` and replay them through `initialMessages` to restore
a session after a restart. `agent.fork()` gives an independent copy with fresh
conversation memory; `agent.reset()` clears history in place.

## Long-term memory

`LongTermMemory` is three methods, so it maps onto whatever store you already
run — Postgres + pgvector, Redis, SQLite, a hosted vector DB.

```ts
interface LongTermMemory {
  add(records: readonly NewMemoryRecord[]): void | Promise<void>;
  search(
    query: string,
    options?: MemorySearchOptions,
  ): readonly MemorySearchResult[] | Promise<readonly MemorySearchResult[]>;
  remove(ids: readonly string[]): void | Promise<void>;
  clear(): void | Promise<void>;
}
```

`search` may be synchronous, so an in-memory implementation does not pay for an
unnecessary microtask.

A record is a durable fact:

```ts
{
  id: 'mem_...',
  role: 'user' | 'assistant' | 'system',
  content: 'I prefer TypeScript',
  createdAt: 1712345678901,
  metadata: { kind: 'preference' },  // your own namespace
  importance: 0.8,                   // optional scoring nudge
}
```

| Search option     | Notes                                                    |
| ----------------- | -------------------------------------------------------- |
| `limit`           | Maximum results                                          |
| `minScore`        | Drop results below this relevance score (0–1)            |
| `roles`           | Restrict to certain roles                                |
| `requireAllTerms` | Require every query token to be present                  |
| `searchMetadata`  | Include metadata in scoring (implementations may ignore) |

### Built-in implementations

`InMemoryLongTermMemory` scores with TF-IDF-ish weighting over unique tokens,
nudged by recency and explicit importance. Enough for "remember that the user
prefers X", entirely in process, no dependencies.

```ts
import { InMemoryLongTermMemory } from '@agentloom/core';

const memory = new InMemoryLongTermMemory({
  maxRecords: 1000,
  defaultLimit: 5,
});
```

Oldest records are evicted past `maxRecords`. `all()` returns every record
newest first.

`createFileLongTermMemory` adds JSON-file persistence: it reads once at
construction and writes on mutation.

```ts
import { createFileLongTermMemory } from '@agentloom/core';

const memory = createFileLongTermMemory({
  path: './.agentloom-memories.json',
  maxRecords: 5_000,
  flush: true, // write on every mutation instead of debouncing
});

memory.memory.all(); // the underlying InMemoryLongTermMemory
```

Pass an `adapter` (`{ load, save }`) to store records somewhere else — object
storage, or a fixture in tests. Not safe for concurrent processes; use a real
store for that.

## Wiring long-term memory

```ts
const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  longTermMemory: memory,
  longTerm: {
    recall: true, // inject relevant past facts into the system prompt
    store: true, // persist this run's input and answer
    limit: 5,
    minScore: 0.2,
    roles: ['user', 'assistant'],
    heading: 'Relevant memories from earlier sessions',
  },
});
```

| Field      | Default                                   | Notes                                     |
| ---------- | ----------------------------------------- | ----------------------------------------- |
| `recall`   | `true`                                    | Search before the run                     |
| `store`    | `true`                                    | Save input and final answer after the run |
| `limit`    | `5`                                       | Results requested per recall              |
| `minScore` | `0.01`                                    | Relevance floor                           |
| `roles`    | `['user', 'system']`                      | Roles searched                            |
| `heading`  | `Relevant memories from earlier sessions` | Prompt heading                            |

Recall happens once per run, before planning, and the hits are rendered into the
system prompt as a scored list:

```
Relevant memories from earlier sessions:
- (0.81) I prefer TypeScript
- (0.44) The user works mostly in TypeScript
```

Storing happens when the run finalizes, including limit stops and soft failures,
so nothing is lost when a run is cut short. It is skipped only when the run
throws (the default `throwOnError: true`), since there is no final answer to
record.

Neither recall nor store can sink a run: both are wrapped, a failure is logged
as a warning and reported through the `memory` event, and the run continues
without memories. This matters when the memory backend is the least reliable
part of your stack.

## Eventing

Long-term reads and writes emit `memory` events, so recall latency and hit
counts are observable:

```ts
agent.on('memory', ({ operation, count, detail }) => {
  metrics.observe(`memory.${operation}`, { count, detail });
});
```

Conversation trimming is not evented directly — read it from
`memory.state()` (which reports `droppedCount` and `trimmed`) or from
`agent:prompt`, whose `estimatedTokens` shows what was actually assembled.

## More

- [Agent](agent.md) — where the memory wiring fits in the loop
- [Streaming & events](streaming-and-events.md) — `agent:prompt` and `memory`
- [Testing](testing.md) — a scripted provider plus in-memory stores
