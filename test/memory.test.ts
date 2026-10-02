import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { MemoryError } from '../src/errors.js';
import {
  assistantMessage,
  createSummaryCompactor,
  defaultTokenCounter,
  InMemoryConversationMemory,
  summaryMessage,
  toolResultMessage,
  transcriptOf,
} from '../src/memory/conversation.js';
import {
  createFileLongTermMemory,
  InMemoryLongTermMemory,
  type MemorySearchOptions,
  type MemorySearchResult,
} from '../src/memory/long-term.js';
import type { ModelMessage } from '../src/providers/types.js';
import type { TokenCounter } from '../src/memory/types.js';

const user = (content: string): ModelMessage => ({ role: 'user', content });
const assistant = (content: string): ModelMessage => ({ role: 'assistant', content });

/** 1 token per message, so budgets are easy to reason about in assertions. */
const perMessage: TokenCounter = (input) =>
  typeof input === 'string' ? 1 : Array.isArray(input) ? input.length : 1;

describe('InMemoryConversationMemory', () => {
  it('stores and returns messages in order', () => {
    const memory = new InMemoryConversationMemory();
    memory.add(user('one'));
    memory.addAll([assistant('two'), user('three')]);

    expect(memory.length).toBe(3);
    expect(memory.messages().map((m) => m.content)).toEqual(['one', 'two', 'three']);
  });

  it('clears', () => {
    const memory = new InMemoryConversationMemory({ initialMessages: [user('seed')] });
    expect(memory.length).toBe(1);
    memory.clear();
    expect(memory.length).toBe(0);
    expect(memory.messages()).toEqual([]);
  });

  it('seeds from initialMessages', () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b')],
    });
    expect(memory.length).toBe(2);
  });

  it('prepends the system prompt in build()', async () => {
    const memory = new InMemoryConversationMemory({ initialMessages: [user('hi')] });
    const built = await memory.build({
      maxTokens: 10_000,
      tokenCounter: defaultTokenCounter,
      keepRecent: 4,
      system: 'You are helpful.',
    });

    expect(built[0]).toEqual({ role: 'system', content: 'You are helpful.' });
    expect(built).toHaveLength(2);
  });

  it('omits an empty system prompt', async () => {
    const memory = new InMemoryConversationMemory({ initialMessages: [user('hi')] });
    const built = await memory.build({
      maxTokens: 10_000,
      tokenCounter: defaultTokenCounter,
      keepRecent: 4,
      system: '',
    });
    expect(built[0]?.role).toBe('user');
  });

  it('returns everything when the budget is generous', async () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b'), user('c')],
    });
    const built = await memory.build({
      maxTokens: 100_000,
      tokenCounter: defaultTokenCounter,
      keepRecent: 4,
    });
    expect(built).toHaveLength(3);
    expect(memory.state().trimmed).toBe(false);
  });

  it('drops the oldest messages when over budget', async () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b'), user('c'), assistant('d')],
      keepRecent: 2,
    });
    // 1 token per message makes the budget easy to reason about.
    const built = await memory.build({
      maxTokens: 2,
      tokenCounter: perMessage,
      keepRecent: 2,
    });

    expect(built.map((m) => m.content)).toEqual(['c', 'd']);
    expect(memory.state().droppedCount).toBe(2);
    expect(memory.state().trimmed).toBe(true);
  });

  it('never trims the protected recent tail, even past the budget', async () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b')],
    });
    const built = await memory.build({
      maxTokens: 1,
      tokenCounter: perMessage,
      keepRecent: 10,
    });

    expect(built.map((m) => m.content)).toEqual(['a', 'b']);
  });

  it('reports token counts', () => {
    const memory = new InMemoryConversationMemory({ initialMessages: [user('hello')] });
    expect(memory.tokenCount()).toBeGreaterThan(0);
    expect(memory.state().tokens).toBe(memory.tokenCount());
  });

  it('caps stored messages, evicting the oldest', () => {
    const memory = new InMemoryConversationMemory({ maxMessages: 4, keepRecent: 2 });
    for (let i = 0; i < 10; i++) memory.add(user(`m${i}`));

    expect(memory.length).toBe(4);
    expect(memory.messages().map((m) => m.content)).toEqual(['m6', 'm7', 'm8', 'm9']);
  });

  it('does not evict when maxMessages is below keepRecent', () => {
    const memory = new InMemoryConversationMemory({ maxMessages: 1, keepRecent: 4 });
    memory.add(user('a'));
    memory.add(user('b'));
    expect(memory.length).toBe(2);
  });

  it('supports a summarize strategy', async () => {
    const summarize = vi.fn(async () => 'earlier: the user asked about cats');
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b'), user('c'), assistant('d')],
      keepRecent: 2,
      strategy: 'summarize',
      compact: createSummaryCompactor(summarize),
    });

    const built = await memory.build({
      maxTokens: 2,
      tokenCounter: perMessage,
      keepRecent: 2,
    });

    expect(summarize).toHaveBeenCalledTimes(1);
    expect(built[0]?.role).toBe('system');
    expect(built[0]?.content).toContain('cats');
    expect(built.slice(1).map((m) => m.content)).toEqual(['c', 'd']);
    expect(memory.state().summaryCount).toBe(1);
  });

  it('falls back to sliding when the compactor declines', async () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b'), user('c'), assistant('d')],
      keepRecent: 2,
      strategy: 'summarize',
      compact: () => undefined,
    });

    const built = await memory.build({
      maxTokens: 2,
      tokenCounter: perMessage,
      keepRecent: 2,
    });
    expect(built.map((m) => m.content)).toEqual(['c', 'd']);
    expect(memory.state().summaryCount).toBe(0);
  });

  it('wraps a compactor failure in a MemoryError', async () => {
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), user('b'), user('c')],
      keepRecent: 1,
      strategy: 'summarize',
      compact: () => {
        throw new Error('summarizer down');
      },
    });

    await expect(
      memory.build({ maxTokens: 1, tokenCounter: perMessage, keepRecent: 1 }),
    ).rejects.toThrow(MemoryError);
  });

  it('honours a custom token counter', async () => {
    const counter = vi.fn(() => 1);
    const memory = new InMemoryConversationMemory({
      initialMessages: [user('a'), assistant('b')],
      keepRecent: 1,
      tokenCounter: counter,
    });
    await memory.build({ maxTokens: 100, tokenCounter: counter, keepRecent: 1 });
    expect(counter).toHaveBeenCalled();
  });

  it('exposes state for diagnostics', () => {
    const memory = new InMemoryConversationMemory({ initialMessages: [user('a')] });
    expect(memory.state().messages).toHaveLength(1);
    expect(memory.state().trimmed).toBe(false);
  });
});

describe('conversation helpers', () => {
  it('summaryMessage wraps text in a system message', () => {
    const message = summaryMessage('  the gist  ');
    expect(message.role).toBe('system');
    expect(message.content).toContain('the gist');
  });

  it('transcriptOf labels every role', () => {
    const text = transcriptOf([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'usr' },
      {
        role: 'assistant',
        content: 'asst',
        toolCalls: [{ id: '1', name: 'calc', arguments: {} }],
      },
      { role: 'tool', content: '42', toolCallId: '1', name: 'calc' },
    ]);

    expect(text).toContain('system: sys');
    expect(text).toContain('user: usr');
    expect(text).toContain('assistant: asst');
    expect(text).toContain('calc');
    expect(text).toContain('42');
  });

  it('assistantMessage builds an assistant turn', () => {
    expect(assistantMessage('hi')).toEqual({ role: 'assistant', content: 'hi' });
    expect(
      assistantMessage('', [{ id: '1', name: 't', arguments: {} }]).toolCalls,
    ).toHaveLength(1);
  });

  it('toolResultMessage builds a tool turn', () => {
    const message = toolResultMessage('1', 'calc', '42');
    expect(message).toMatchObject({
      role: 'tool',
      toolCallId: '1',
      name: 'calc',
      content: '42',
    });
  });

  it('createSummaryCompactor returns undefined for an empty summary', async () => {
    const compact = createSummaryCompactor(async () => '   ');
    await expect(compact([user('a')], { droppedTokens: 1 })).resolves.toBeUndefined();
  });

  it('createSummaryCompactor passes the transcript to the summarizer', async () => {
    const summarize = vi.fn(async (_prompt: string, _signal?: AbortSignal) => 'summary');
    const compact = createSummaryCompactor(summarize);
    await compact([user('a question')], { droppedTokens: 1 });

    const prompt = summarize.mock.calls[0]?.[0] ?? '';
    expect(prompt).toContain('a question');
  });
});

describe('defaultTokenCounter', () => {
  it('counts strings', () => {
    expect(defaultTokenCounter('hello world')).toBeGreaterThan(0);
  });

  it('counts messages and message lists', () => {
    const message: ModelMessage = { role: 'user', content: 'hello world' };
    expect(defaultTokenCounter(message)).toBeGreaterThan(0);
    expect(defaultTokenCounter([message, assistant('more')])).toBeGreaterThan(
      defaultTokenCounter(message),
    );
  });
});

describe('InMemoryLongTermMemory', () => {
  const record = (content: string) => ({ role: 'user' as const, content });
  // `LongTermMemory.search` may be sync or async; this keeps assertions uniform.
  const search = async (
    memory: InMemoryLongTermMemory,
    query: string,
    options?: MemorySearchOptions,
  ): Promise<readonly MemorySearchResult[]> => memory.search(query, options);

  it('stores records and reports its size', () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([record('the cat sat on the mat')]);
    expect(memory.size).toBe(1);
  });

  it('finds records by keyword overlap', async () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([
      record('the user prefers TypeScript over JavaScript'),
      record('the deployment target is a kubernetes cluster'),
    ]);

    const hits = await search(memory, 'TypeScript');
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]?.record.content).toContain('TypeScript');
    expect(hits[0]?.score).toBeGreaterThan(0);
  });

  it('returns nothing for an unrelated query', async () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([record('the user prefers TypeScript')]);
    expect(await search(memory, 'quantum chromodynamics')).toEqual([]);
  });

  it('respects the result limit', async () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([record('cats are great'), record('cats are cute'), record('cats purr')]);

    const hits = await search(memory, 'cats', { limit: 2 });
    expect(hits.length).toBeLessThanOrEqual(2);
  });

  it('filters by role when asked', async () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([record('cats'), { role: 'assistant', content: 'cats' }]);
    expect(await search(memory, 'cats', { roles: ['assistant'] })).toHaveLength(1);
  });

  it('evicts the oldest records past maxRecords', () => {
    const memory = new InMemoryLongTermMemory({ maxRecords: 2 });
    memory.add([record('first'), record('second')]);
    memory.add([record('third')]);

    expect(memory.size).toBe(2);
    const all = [...memory.search('first third second')];
    expect(all.some((hit) => hit.record.content === 'first')).toBe(false);
  });

  it('removes records by id', async () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([{ id: 'keep', role: 'user', content: 'cats' }, record('dogs')]);

    memory.remove(['keep']);
    expect(await search(memory, 'cats')).toEqual([]);
    expect(memory.size).toBe(1);
  });

  it('clears everything', () => {
    const memory = new InMemoryLongTermMemory();
    memory.add([record('a'), record('b')]);
    memory.clear();
    expect(memory.size).toBe(0);
  });

  it('honours a custom default limit', async () => {
    const memory = new InMemoryLongTermMemory({ defaultLimit: 1 });
    memory.add([record('cats a'), record('cats b')]);
    expect(await search(memory, 'cats')).toHaveLength(1);
  });
});

describe('createFileLongTermMemory', () => {
  const dirs: string[] = [];

  afterEach(async () => {
    await Promise.all(
      dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })),
    );
  });

  const tempFile = async (): Promise<string> => {
    const dir = await mkdtemp(join(tmpdir(), 'agentloom-memory-'));
    dirs.push(dir);
    return join(dir, 'records.json');
  };

  it('persists records across instances', async () => {
    const file = await tempFile();

    const first = createFileLongTermMemory({ path: file });
    await first.add([{ role: 'user', content: 'remember the pineapple' }]);

    const second = createFileLongTermMemory({ path: file });
    const hits = await second.search('pineapple');
    expect(hits).toHaveLength(1);
    expect(hits[0]?.record.content).toContain('pineapple');
  });

  it('starts empty when the file is missing', async () => {
    const memory = createFileLongTermMemory({ path: await tempFile() });
    expect(await memory.search('anything')).toEqual([]);
  });

  it('removes and clears', async () => {
    const file = await tempFile();
    const memory = createFileLongTermMemory({ path: file });
    await memory.add([
      { id: 'a', role: 'user', content: 'cats' },
      { id: 'b', role: 'user', content: 'dogs' },
    ]);

    await memory.remove(['a']);
    expect(await memory.search('cats')).toEqual([]);

    await memory.clear();
    expect(await memory.search('dogs')).toEqual([]);
  });

  it('creates missing parent directories', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'agentloom-memory-'));
    dirs.push(dir);

    const memory = createFileLongTermMemory({
      path: join(dir, 'nested', 'deep', 'r.json'),
    });
    await expect(
      memory.add([{ role: 'user', content: 'nested write' }]),
    ).resolves.toBeUndefined();
    expect(await memory.search('nested')).toHaveLength(1);
  });
});
