# Testing

An agent is a class with an injectable provider, so testing one does not require
a network, an API key, or a mock library. Write a provider that returns
canned responses and assert on the result.

## A scripted provider

```ts
import {
  type CompletionResult,
  type ModelProvider,
  type ProviderCapabilities,
} from '@agentloom/core';

function scriptedProvider(replies: readonly CompletionResult[]): ModelProvider {
  let index = 0;
  const next = (): CompletionResult =>
    replies[Math.min(index++, replies.length - 1)]!;

  return {
    id: 'scripted',
    name: 'Scripted',
    defaultModel: 'scripted-1',
    capabilities: {
      tools: true,
      parallelToolCalls: false,
      streaming: true,
      systemMessages: true,
      jsonMode: true,
      strictJsonSchema: true,
      vision: false,
      promptCaching: false,
    } satisfies ProviderCapabilities,
    complete: async () => next(),
    stream: (request) => ({
      async *[Symbol.asyncIterator]() {
        const reply = next();
        yield { type: 'finish', result: { ...reply, model: request.model } };
      },
    }),
  };
}

const reply = (
  content: string,
  toolCalls?: CompletionResult['message']['toolCalls'],
): CompletionResult => ({
  message: { role: 'assistant', content, ...(toolCalls ? { toolCalls } : {}) },
  finishReason: toolCalls ? 'tool_calls' : 'stop',
  usage: { inputTokens: 20, outputTokens: 10, totalTokens: 30 },
  responseId: 'r1',
  providerId: 'scripted',
  model: 'scripted-1',
  latencyMs: 1,
  raw: {},
});
```

The last reply repeats if the agent asks for more, so a test cannot fail with an
exhaustion error instead of the assertion you care about.

## A tool-loop test

```ts
import { describe, expect, it } from 'vitest';
import { Agent, createCalculatorTool } from '@agentloom/core';

it('uses the calculator and reports the result', async () => {
  const agent = new Agent({
    provider: scriptedProvider([
      reply('Let me compute that.', [
        {
          id: 'call_1',
          name: 'calculator',
          arguments: { expression: '19 * 23' },
        },
      ]),
      reply('19 * 23 = 437.'),
    ]),
    instructions: 'Use the calculator for arithmetic.',
    tools: [createCalculatorTool()],
  });

  const result = await agent.run('What is 19 times 23?');

  expect(result.output).toContain('437');
  expect(result.stopReason).toBe('completed');
  expect(result.iterations).toBe(2);
  expect(result.steps[0]!.toolCalls[0]!.name).toBe('calculator');
});
```

`result.steps` gives you every model call and the tools it triggered, which is
usually more useful to assert on than the final text.

`examples/08-offline-tour.ts` is a working template covering tools, streaming,
cancellation, structured output, memory, planning, and limits in one file.

## Cancellation

A provider that never resolves must still be cancellable — that is worth a test
of its own:

```ts
it('stops a provider that ignores its signal', async () => {
  const hang = new Agent({
    provider: {
      ...scriptedProvider([reply('never')]),
      complete: () => new Promise<CompletionResult>(() => undefined),
    },
  });

  const controller = new AbortController();
  const promise = hang.run('wait forever', {
    signal: controller.signal,
    throwOnError: false,
  });

  setTimeout(() => controller.abort(), 10);

  const result = await promise;
  expect(result.stopReason).toBe('aborted');
});
```

## Asserting on tool calls

Capture calls in a tool, or subscribe to events:

```ts
const seen: unknown[] = [];

const agent = new Agent({
  provider: scriptedProvider([
    reply('', [{ id: 'c1', name: 'record', arguments: { kind: 'refund' } }]),
    reply('Done.'),
  ]),
  tools: [record],
  hooks: { onToolEnd: ({ toolName }) => seen.push(toolName) },
});

await agent.run('...');
expect(seen).toEqual(['record']);
```

Events cover more ground than hooks and are typed per event:

```ts
const durations: number[] = [];
agent.on('tool:end', ({ durationMs }) => durations.push(durationMs));
```

## Testing without a validator dependency

Structured output needs only a `safeParse` method, so a test can supply a
literal:

```ts
const Profile = {
  safeParse(input: unknown) {
    const value = input as { name?: unknown; age?: unknown } | null;
    return typeof value?.name === 'string'
      ? ({ success: true, data: value } as const)
      : ({
          success: false,
          error: { issues: [{ path: ['name'], message: 'required' }] },
        } as const);
  },
};

const result = await agent.run('Ada is 36.', { outputSchema: Profile });
expect(result.data).toEqual({ name: 'Ada', age: 36 });
```

## Testing the repair loop

Return invalid JSON first, then valid, and assert the model was asked twice:

```ts
let calls = 0;
const agent = new Agent({
  provider: {
    ...scriptedProvider([reply('{"name": }'), reply('{"name":"Ada"}')]),
    complete: async () => {
      calls++;
      return replies[Math.min(calls - 1, 1)]!;
    },
  },
});

const result = await agent.run('...', { outputSchema: Profile });

expect(calls).toBe(2);
expect(result.data).toEqual({ name: 'Ada' });
```

The invalid attempt also emits `output:invalid`, which you can subscribe to.

## Testing limits

```ts
const loopy = new Agent({
  provider: scriptedProvider([
    reply('again', [
      { id: 'c', name: 'calculator', arguments: { expression: '1+1' } },
    ]),
  ]),
  tools: [createCalculatorTool()],
  limits: { maxIterations: 3 },
});

const result = await loopy.run('go', { throwOnError: false });

expect(result.stopReason).toBe('max_iterations');
expect(result.iterations).toBe(3);
expect(result.error).toBeInstanceOf(ExecutionLimitError);
```

## Testing providers against recorded HTTP

Inject `fetch` to test the wire format without a network:

```ts
import { OpenAIProvider } from '@agentloom/core';

const provider = new OpenAIProvider({
  apiKey: 'test',
  fetch: async (url, init) => {
    expect(url).toContain('/chat/completions');
    expect(JSON.parse(init.body!).tools[0].function.name).toBe('get_weather');
    return fakeResponse({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] });
  },
});

const result = await provider.complete({ model: 'gpt-4o-mini', messages: [...] });
```

`FetchLike` is a narrow subset — `url`, `{ method, headers, body, signal }` — so
a stub only has to satisfy that.

## Logging assertions

```ts
import { MemoryLogSink } from '@agentloom/core';

const sink = new MemoryLogSink({ level: 'debug' });
const agent = new Agent({ provider, logger: sink });

await agent.run('...');

expect(sink.records.some((r) => r.message === 'run stopped')).toBe(false);
```

## Memory

Both stores are in-memory by default and synchronous where possible, so no
setup is needed:

```ts
const memory = new InMemoryConversationMemory({ maxMessages: 10 });
const longTerm = new InMemoryLongTermMemory();

const agent = new Agent({ provider, memory, longTermMemory: longTerm });

await agent.run('My name is Ada.');
const result = await agent.run('What is my name?');

expect(memory.messages().length).toBeGreaterThan(0);
expect(await longTerm.search('Ada')).not.toHaveLength(0);
```

For long-term memory you care about the interface, not the implementation — write
a fake `LongTermMemory` and assert on what the agent searched for and stored.

## Test checklist

- provider accepts `provider` or `model`; prefer `provider` so no registry or key
  is needed
- canned replies repeat rather than throwing when exhausted
- `throwOnError: false` for tests asserting on a stop reason
- assert on `steps` and `usage`, not just `output`
- `vitest` is already configured; `npm test` runs the suite

## More

- [Providers](providers.md) — the provider contract, HTTP injection
- [Agent](agent.md) — `steps`, `stopReason`, limits
- [Errors](errors.md) — typed failures you may want to assert on
