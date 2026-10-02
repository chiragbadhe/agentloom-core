/**
 * 05 — memory: conversation history, compaction, and long-term recall.
 *
 * Run: `npx tsx examples/05-memory.ts`
 */
import {
  Agent,
  createFileLongTermMemory,
  createSummaryCompactor,
  InMemoryConversationMemory,
  InMemoryLongTermMemory,
  type LongTermMemory,
} from '../src/index.js';

/**
 * Conversation memory keeps the transcript and trims it to fit the context
 * window. `maxMessages` and `maxTokens` both apply; the newest turns win.
 */
const memory = new InMemoryConversationMemory({
  maxMessages: 40,
  maxTokens: 12_000,
  // Keep the newest 6 turns verbatim when trimming.
  keepRecent: 6,
  /**
   * `summarize` replaces older turns with a model-written summary instead of
   * dropping them. The default `sliding` strategy just removes them.
   */
  strategy: 'summarize',
  compact: createSummaryCompactor(async (prompt) => {
    const summary = await summarizer.run(prompt);
    return summary.output;
  }),
});

// A second agent used only for summarisation; any provider call would do.
const summarizer = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: 'You compress conversations without losing facts.',
});

/**
 * Long-term memory is your own store. The interface is three methods, so it can
 * be a vector database, Postgres, Redis, or — here — a JSON file.
 */
const longTerm = createFileLongTermMemory({ path: './.agentloom-memories.json' });

const agent = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: 'You are a helpful assistant with a good memory for user preferences.',
  memory,
  longTermMemory: longTerm,
  longTerm: {
    recall: true, // inject relevant past facts into the system prompt
    store: true, // persist this run's input and answer
    limit: 5,
    minScore: 0.2,
    roles: ['user', 'assistant'],
  },
});

// Runs share `memory`, so the agent remembers earlier turns.
await agent.run('My name is Sam and I prefer metric units.');
await agent.run('I also work mostly in TypeScript.');
const answer = await agent.run('What do you know about me?');

console.log(answer.output);
console.log(`\ntranscript: ${memory.messages().length} messages`);

// Records are yours to inspect.
for (const hit of await longTerm.search('preferences', { limit: 5 })) {
  console.log(`  [${hit.score.toFixed(2)}] ${hit.record.content.slice(0, 60)}`);
}

// A memory keyed by user/session keeps conversations separate.
const scoped = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  memory: new InMemoryConversationMemory({ maxMessages: 20 }),
  instructions: 'You are a support agent for a hardware store.',
});

// Long-term memory can also be swapped for your own implementation. Typed as
// the interface, `add`/`search` may be synchronous or asynchronous.
const scratch: LongTermMemory = new InMemoryLongTermMemory();
await scratch.add([
  { role: 'user', content: 'Remember: the office coffee machine is broken.' },
  { role: 'assistant', content: 'Noted, I will avoid scheduling near it.' },
]);
const hits = await scratch.search('coffee machine', { limit: 1 });
console.log(`\nrecalled: ${hits[0]?.record.content}`);

console.log(
  `\nscoped session: ${await scoped.run('Do you know me?').then((r) => r.output)}`,
);
