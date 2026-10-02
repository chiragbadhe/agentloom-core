import { describe, expect, it } from 'vitest';

import { ProviderResponseError } from '../../src/errors.js';
import { createOllamaProvider } from '../../src/providers/ollama.js';
import type { CompletionRequest, StreamEvent } from '../../src/providers/types.js';
import { schemaFromJsonSchema } from '../../src/schema.js';
import { stubFetch } from '../helpers/mock-fetch.js';

const request = (overrides: Partial<CompletionRequest> = {}): CompletionRequest => ({
  model: 'llama3.2',
  messages: [{ role: 'user', content: 'hello' }],
  ...overrides,
});

const textResponse = (content: string) => ({
  body: JSON.stringify({
    model: 'llama3.2',
    message: { role: 'assistant', content },
    done: true,
    done_reason: 'stop',
    prompt_eval_count: 6,
    eval_count: 2,
  }),
});

function ndjson(...lines: readonly unknown[]): readonly string[] {
  return lines.map((line) => `${JSON.stringify(line)}\n`);
}

async function collect(stream: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe('OllamaProvider', () => {
  it('reports identity and capabilities', () => {
    const provider = createOllamaProvider();
    expect(provider.id).toBe('ollama');
    expect(provider.name).toBe('Ollama');
    expect(provider.defaultModel).toBe('llama3.2');
    expect(provider.capabilities.vision).toBe(false);
    expect(provider.capabilities.jsonMode).toBe(true);
  });

  it('posts to the local chat endpoint', async () => {
    const fetchImpl = stubFetch(textResponse('hi from llama'));
    const completion = await createOllamaProvider({ fetch: fetchImpl }).complete(
      request(),
    );

    expect(fetchImpl.calls[0]?.url).toBe('http://localhost:11434/api/chat');
    expect(fetchImpl.bodies()[0]).toMatchObject({
      model: 'llama3.2',
      messages: [{ role: 'user', content: 'hello' }],
      stream: false,
    });
    expect(completion.message.content).toBe('hi from llama');
    expect(completion.finishReason).toBe('stop');
    expect(completion.usage).toEqual({ inputTokens: 6, outputTokens: 2, totalTokens: 8 });
  });

  it('honours a custom host and trims the trailing slash', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOllamaProvider({
      host: 'http://ollama.local:1234/',
      fetch: fetchImpl,
    }).complete(request());
    expect(fetchImpl.calls[0]?.url).toBe('http://ollama.local:1234/api/chat');
  });

  it('sends no authorization header', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOllamaProvider({ fetch: fetchImpl }).complete(request());
    expect(fetchImpl.calls[0]?.init.headers['authorization']).toBeUndefined();
  });

  it('maps messages including tool results', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOllamaProvider({ fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'system', content: 'be terse' },
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: '',
            toolCalls: [{ id: 'c1', name: 'add', arguments: { a: 1 } }],
          },
          { role: 'tool', content: '2', toolCallId: 'c1', name: 'add' },
        ],
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      messages: [
        { role: 'system', content: 'be terse' },
        { role: 'user', content: 'hi' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{ function: { name: 'add', arguments: { a: 1 } } }],
        },
        { role: 'tool', content: '2', tool_name: 'add' },
      ],
    });
  });

  it('nests sampling options', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOllamaProvider({ fetch: fetchImpl, keepAlive: '10m' }).complete(
      request({
        temperature: 0.5,
        topP: 0.9,
        topK: 40,
        maxTokens: 128,
        stopSequences: ['X'],
        seed: 5,
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      keep_alive: '10m',
      options: {
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        num_predict: 128,
        stop: ['X'],
        seed: 5,
      },
    });
  });

  it('sends tools and a json format', async () => {
    const fetchImpl = stubFetch([textResponse('a'), textResponse('b')]);
    const provider = createOllamaProvider({ fetch: fetchImpl });

    await provider.complete(
      request({
        tools: [
          {
            name: 'search',
            description: 'Search',
            parameters: schemaFromJsonSchema({
              type: 'object',
              properties: { q: { type: 'string' } },
            }),
          },
        ],
        responseFormat: {
          type: 'json_schema',
          schema: { type: 'object', properties: { a: {} } },
        },
      }),
    );
    await provider.complete(request({ responseFormat: { type: 'json_object' } }));

    expect(fetchImpl.bodies()[0]).toMatchObject({
      tools: [
        {
          type: 'function',
          function: {
            name: 'search',
            description: 'Search',
            parameters: { type: 'object', properties: { q: { type: 'string' } } },
          },
        },
      ],
      format: { type: 'object', properties: { a: {} } },
    });
    expect(fetchImpl.bodies()[1]).toMatchObject({ format: 'json' });
  });

  it('omits format for text mode', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOllamaProvider({ fetch: fetchImpl }).complete(
      request({ responseFormat: { type: 'text' } }),
    );
    expect(fetchImpl.bodies()[0]).not.toHaveProperty('format');
  });

  it('parses tool calls reported as objects or strings', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            { function: { name: 'add', arguments: { a: 1, b: 2 } } },
            { function: { name: 'mul', arguments: '{"a":2,"b":3}' } },
          ],
        },
        done: true,
        done_reason: 'stop',
      }),
    });

    const completion = await createOllamaProvider({ fetch: fetchImpl }).complete(
      request(),
    );

    expect(completion.finishReason).toBe('tool_calls');
    expect(completion.message.toolCalls?.[0]).toMatchObject({
      id: 'call_0',
      name: 'add',
      arguments: { a: 1, b: 2 },
    });
    expect(completion.message.toolCalls?.[1]?.arguments).toEqual({ a: 2, b: 3 });
  });

  it('errors on malformed tool arguments', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        message: { tool_calls: [{ function: { name: 'add', arguments: 'nope' } }] },
        done: true,
      }),
    });
    await expect(
      createOllamaProvider({ fetch: fetchImpl }).complete(request()),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('defaults missing tool arguments to an empty object', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        message: { tool_calls: [{ function: { name: 'ping' } }] },
        done: true,
      }),
    });
    const completion = await createOllamaProvider({ fetch: fetchImpl }).complete(
      request(),
    );
    expect(completion.message.toolCalls?.[0]?.arguments).toEqual({});
  });

  it('lists local models', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        models: [
          {
            name: 'llama3.2:latest',
            size: 2_000_000_000,
            modified_at: '2024-01-01T00:00:00Z',
          },
        ],
      }),
    });

    const models = await createOllamaProvider({ fetch: fetchImpl }).listModels();
    expect(fetchImpl.calls[0]?.url).toBe('http://localhost:11434/api/tags');
    expect(fetchImpl.calls[0]?.init.method).toBe('GET');
    expect(models).toEqual([
      {
        name: 'llama3.2:latest',
        size: 2_000_000_000,
        modified_at: '2024-01-01T00:00:00Z',
      },
    ]);
  });

  it('returns an empty list when the server reports no models', async () => {
    const fetchImpl = stubFetch({ body: '{}' });
    await expect(
      createOllamaProvider({ fetch: fetchImpl }).listModels(),
    ).resolves.toEqual([]);
  });

  it('streams ndjson deltas', async () => {
    const fetchImpl = stubFetch({
      chunks: ndjson(
        { message: { content: 'Hel' } },
        { message: { content: 'lo' } },
        { done: true, done_reason: 'stop', prompt_eval_count: 4, eval_count: 2 },
      ),
    });

    const events = await collect(
      createOllamaProvider({ fetch: fetchImpl }).stream(request()),
    );

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text-delta',
      'text-delta',
      'finish',
    ]);
    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.content).toBe('Hello');
    expect(finished?.type === 'finish' && finished.result.usage.totalTokens).toBe(6);
  });

  it('requests a stream from the API', async () => {
    const fetchImpl = stubFetch({ chunks: ndjson({ done: true }) });
    await collect(createOllamaProvider({ fetch: fetchImpl }).stream(request()));
    expect(fetchImpl.bodies()[0]).toMatchObject({ stream: true });
  });

  it('streams tool calls', async () => {
    const fetchImpl = stubFetch({
      chunks: ndjson(
        { message: { tool_calls: [{ function: { name: 'add', arguments: { a: 1 } } }] } },
        { done: true, done_reason: 'stop' },
      ),
    });

    const events = await collect(
      createOllamaProvider({ fetch: fetchImpl }).stream(request()),
    );
    const toolEvent = events.find((event) => event.type === 'tool-call');
    expect(toolEvent?.type === 'tool-call' && toolEvent.call.arguments).toEqual({ a: 1 });

    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.finishReason).toBe(
      'tool_calls',
    );
  });

  it('throws on an inline error record', async () => {
    const fetchImpl = stubFetch({ chunks: ndjson({ error: 'model not found' }) });
    await expect(
      collect(createOllamaProvider({ fetch: fetchImpl }).stream(request())),
    ).rejects.toThrow(/model not found/);
  });

  it('ignores blank and malformed ndjson lines', async () => {
    const fetchImpl = stubFetch({
      chunks: ['not json\n', '\n', '{"message":{"content":"ok"}}\n', '{"done":true}\n'],
    });
    const events = await collect(
      createOllamaProvider({ fetch: fetchImpl }).stream(request()),
    );
    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.content).toBe('ok');
  });

  it('errors when the response has no readable stream', async () => {
    const fetchImpl = stubFetch({ body: 'x' });
    await expect(
      collect(createOllamaProvider({ fetch: fetchImpl }).stream(request())),
    ).rejects.toThrow(/readable stream/);
  });

  it('surfaces HTTP errors', async () => {
    const fetchImpl = stubFetch({
      status: 404,
      body: 'model not found, try pulling it first',
    });
    await expect(
      createOllamaProvider({ fetch: fetchImpl, retry: false }).complete(request()),
    ).rejects.toThrow(/try pulling it first/);
  });

  it('clones with a different default model', () => {
    const provider = createOllamaProvider({ retry: false });
    expect(provider.withModel('qwen2.5').defaultModel).toBe('qwen2.5');
    expect(provider.defaultModel).toBe('llama3.2');
  });
});
