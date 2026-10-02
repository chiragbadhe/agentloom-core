import { describe, expect, it } from 'vitest';

import { ProviderResponseError } from '../../src/errors.js';
import {
  createOpenAICompatibleProvider,
  createOpenAIProvider,
  OpenAIProvider,
} from '../../src/providers/openai.js';
import type { CompletionRequest, StreamEvent } from '../../src/providers/types.js';
import { schemaFromJsonSchema } from '../../src/schema.js';
import { stubFetch } from '../helpers/mock-fetch.js';

const request = (overrides: Partial<CompletionRequest> = {}): CompletionRequest => ({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'hello' }],
  ...overrides,
});

const textResponse = (content: string) => ({
  body: JSON.stringify({
    id: 'chatcmpl-1',
    model: 'gpt-4o-mini-2024',
    choices: [{ message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 4,
      total_tokens: 14,
      prompt_tokens_details: { cached_tokens: 2 },
      completion_tokens_details: { reasoning_tokens: 1 },
    },
  }),
});

async function collect(stream: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function sseBody(...frames: readonly string[]): readonly string[] {
  return frames.map((frame) => `data: ${frame}\n\n`);
}

describe('OpenAIProvider', () => {
  it('reports identity and capabilities', () => {
    const provider = createOpenAIProvider();
    expect(provider.id).toBe('openai');
    expect(provider.name).toBe('OpenAI');
    expect(provider.defaultModel).toBe('gpt-4o-mini');
    expect(provider.capabilities.tools).toBe(true);
    expect(provider.capabilities.strictJsonSchema).toBe(true);
  });

  it('posts to the chat completions endpoint with auth', async () => {
    const fetchImpl = stubFetch(textResponse('hi there'));
    const provider = createOpenAIProvider({ apiKey: 'sk-test', fetch: fetchImpl });

    const completion = await provider.complete(request());

    const call = fetchImpl.calls[0];
    expect(call?.url).toBe('https://api.openai.com/v1/chat/completions');
    expect(call?.init.headers['authorization']).toBe('Bearer sk-test');
    expect(fetchImpl.bodies()[0]).toMatchObject({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'hello' }],
    });
    expect(completion.message.content).toBe('hi there');
    expect(completion.responseId).toBe('chatcmpl-1');
    expect(completion.model).toBe('gpt-4o-mini-2024');
    expect(completion.finishReason).toBe('stop');
    expect(completion.usage).toEqual({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: 14,
      cachedInputTokens: 2,
      reasoningTokens: 1,
    });
  });

  it('honours baseURL, organization and project headers', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    const provider = createOpenAIProvider({
      apiKey: 'k',
      baseURL: 'https://gateway.test/openai/v1/',
      organization: 'org-1',
      project: 'proj-1',
      fetch: fetchImpl,
    });

    await provider.complete(request());
    expect(fetchImpl.calls[0]?.url).toBe(
      'https://gateway.test/openai/v1/chat/completions',
    );
    expect(fetchImpl.calls[0]?.init.headers['openai-organization']).toBe('org-1');
    expect(fetchImpl.calls[0]?.init.headers['openai-project']).toBe('proj-1');
  });

  it('omits the authorization header without a key', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOpenAIProvider({ apiKey: '', fetch: fetchImpl }).complete(request());
    expect(fetchImpl.calls[0]?.init.headers['authorization']).toBeUndefined();
  });

  it('merges per-call headers', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(request(), {
      headers: { 'x-trace': 'abc' },
    });
    expect(fetchImpl.calls[0]?.init.headers['x-trace']).toBe('abc');
  });

  it('maps messages onto the chat shape', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'system', content: 'be brief' },
          { role: 'user', content: 'hi', name: 'ada' },
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
        { role: 'system', content: 'be brief' },
        { role: 'user', content: 'hi', name: 'ada' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'c1',
              type: 'function',
              function: { name: 'add', arguments: '{"a":1}' },
            },
          ],
        },
        { role: 'tool', tool_call_id: 'c1', content: '2' },
      ],
    });
  });

  it('serialises tools and sampling options', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        tools: [
          {
            name: 'search',
            description: 'Search the web',
            parameters: schemaFromJsonSchema({
              type: 'object',
              properties: { q: { type: 'string' } },
              required: ['q'],
            }),
          },
        ],
        temperature: 0.2,
        topP: 0.9,
        maxTokens: 256,
        stopSequences: ['STOP'],
        seed: 42,
        presencePenalty: 0.1,
        frequencyPenalty: 0.2,
        reasoningEffort: 'low',
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      tools: [
        {
          type: 'function',
          function: {
            name: 'search',
            description: 'Search the web',
            parameters: {
              type: 'object',
              properties: { q: { type: 'string' } },
              required: ['q'],
            },
          },
        },
      ],
      tool_choice: 'auto',
      parallel_tool_calls: true,
      temperature: 0.2,
      top_p: 0.9,
      max_tokens: 256,
      stop: ['STOP'],
      seed: 42,
      presence_penalty: 0.1,
      frequency_penalty: 0.2,
      reasoning_effort: 'low',
    });
  });

  it('sends response_format for json modes', async () => {
    const fetchImpl = stubFetch([textResponse('{}'), textResponse('{}')]);
    const provider = createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl });

    await provider.complete(request({ responseFormat: { type: 'json_object' } }));
    await provider.complete(
      request({
        responseFormat: {
          type: 'json_schema',
          schema: { type: 'object', properties: { a: { type: 'number' } } },
          name: 'result',
          strict: true,
        },
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      response_format: { type: 'json_object' },
    });
    expect(fetchImpl.bodies()[1]).toMatchObject({
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'result',
          schema: { type: 'object', properties: { a: { type: 'number' } } },
          strict: true,
        },
      },
    });
  });

  it('omits response_format for text mode', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({ responseFormat: { type: 'text' } }),
    );
    expect(fetchImpl.bodies()[0]).not.toHaveProperty('response_format');
  });

  it('parses tool calls', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        id: 'chatcmpl-2',
        choices: [
          {
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_a',
                  type: 'function',
                  function: { name: 'add', arguments: '{"a":1,"b":2}' },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
      }),
    });

    const completion = await createOpenAIProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());

    expect(completion.message.content).toBe('');
    expect(completion.message.toolCalls).toEqual([
      {
        id: 'call_a',
        name: 'add',
        arguments: { a: 1, b: 2 },
        rawArguments: '{"a":1,"b":2}',
      },
    ]);
    expect(completion.finishReason).toBe('tool_calls');
  });

  it('parses tool arguments wrapped in markdown fences', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        choices: [
          {
            message: {
              tool_calls: [
                {
                  id: 'c',
                  function: { name: 'add', arguments: '```json\n{"a":1}\n```' },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
      }),
    });

    const completion = await createOpenAIProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.toolCalls?.[0]?.arguments).toEqual({ a: 1 });
  });

  it('errors on malformed tool arguments', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        choices: [
          {
            message: {
              tool_calls: [{ id: 'c', function: { name: 'add', arguments: 'nope' } }],
            },
            finish_reason: 'tool_calls',
          },
        ],
      }),
    });

    await expect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(request()),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('errors when there are no choices', async () => {
    const fetchImpl = stubFetch({ body: JSON.stringify({ choices: [] }) });
    await expect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).complete(request()),
    ).rejects.toThrow(/no choices/);
  });

  it('reports refusal content', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        choices: [
          {
            message: { role: 'assistant', content: null, refusal: 'I cannot' },
            finish_reason: 'stop',
          },
        ],
      }),
    });
    const completion = await createOpenAIProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.content).toBe('');
  });

  it('omits usage when the provider does not report it', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      }),
    });
    const completion = await createOpenAIProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.usage).toEqual({});
  });

  it('streams text deltas and usage', async () => {
    const fetchImpl = stubFetch({
      chunks: sseBody(
        '{"id":"chatcmpl-3","model":"gpt-4o-mini","choices":[{"delta":{"role":"assistant","content":"He"}}]}',
        '{"choices":[{"delta":{"content":"llo"}}]}',
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        '{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
        '[DONE]',
      ),
    });

    const events = await collect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text-delta',
      'text-delta',
      'finish',
    ]);
    const deltas = events
      .filter(
        (event): event is Extract<StreamEvent, { type: 'text-delta' }> =>
          event.type === 'text-delta',
      )
      .map((event) => event.text);
    expect(deltas).toEqual(['He', 'llo']);

    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.content).toBe('Hello');
    expect(finished?.type === 'finish' && finished.result.responseId).toBe('chatcmpl-3');
    expect(finished?.type === 'finish' && finished.result.usage.totalTokens).toBe(5);
  });

  it('requests a stream from the API', async () => {
    const fetchImpl = stubFetch({
      chunks: sseBody('{"choices":[{"delta":{"content":"x"}}]}', '[DONE]'),
    });
    await collect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(fetchImpl.calls[0]?.init.headers['accept']).toBeDefined();
    expect(fetchImpl.bodies()[0]).toMatchObject({
      stream: true,
      stream_options: { include_usage: true },
    });
  });

  it('assembles streamed tool calls by index', async () => {
    const fetchImpl = stubFetch({
      chunks: sseBody(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\\"a\\":"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"mul","arguments":"{\\"a\\":2,\\"b\\":3}"}}]}}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        '[DONE]',
      ),
    });

    const events = await collect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.toolCalls).toEqual([
      { id: 'call_1', name: 'add', arguments: { a: 1 }, rawArguments: '{"a":1}' },
      {
        id: 'call_2',
        name: 'mul',
        arguments: { a: 2, b: 3 },
        rawArguments: '{"a":2,"b":3}',
      },
    ]);
    expect(finished?.type === 'finish' && finished.result.finishReason).toBe(
      'tool_calls',
    );
  });

  it('synthesises ids for anonymous streamed tool calls', async () => {
    const fetchImpl = stubFetch({
      chunks: sseBody(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"noop","arguments":"{}"}}]}}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        '[DONE]',
      ),
    });

    const events = await collect(
      createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    const finished = events.at(-1);
    expect(
      finished?.type === 'finish' && finished.result.message.toolCalls?.[0]?.id,
    ).toMatch(/^resp_.*_call_0$/);
  });

  it('errors on an unparsable stream frame', async () => {
    const fetchImpl = stubFetch({ chunks: sseBody('{not json}', '[DONE]') });
    await expect(
      collect(createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request())),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('errors when the response has no readable stream', async () => {
    const fetchImpl = stubFetch({ body: 'ignored' });
    await expect(
      collect(createOpenAIProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request())),
    ).rejects.toThrow(/readable stream/);
  });

  it('surfaces HTTP errors with the provider body', async () => {
    const fetchImpl = stubFetch({
      status: 401,
      body: JSON.stringify({ error: { message: 'Invalid API key' } }),
    });
    await expect(
      createOpenAIProvider({ apiKey: 'bad', fetch: fetchImpl }).complete(request()),
    ).rejects.toThrow(/Invalid API key/);
  });

  it('retries transient failures', async () => {
    const fetchImpl = stubFetch([{ status: 503, body: '' }, textResponse('recovered')]);
    const completion = await createOpenAIProvider({
      apiKey: 'k',
      fetch: fetchImpl,
      retry: { maxAttempts: 3, initialDelayMs: 1 },
    }).complete(request());

    expect(completion.message.content).toBe('recovered');
    expect(fetchImpl.calls).toHaveLength(2);
  });

  it('clones with a custom default model', () => {
    const provider = createOpenAIProvider({ model: 'gpt-4o', retry: false });
    const clone = provider.withModel('gpt-4o-mini');
    expect(clone).toBeInstanceOf(OpenAIProvider);
    expect(clone.defaultModel).toBe('gpt-4o-mini');
  });
});

describe('createOpenAICompatibleProvider', () => {
  it('overrides id and name for gateways', () => {
    const provider = createOpenAICompatibleProvider({
      id: 'groq',
      name: 'Groq',
      baseURL: 'https://api.groq.com/openai/v1',
      model: 'llama-3.3-70b-versatile',
      fetch: stubFetch(textResponse('ok')),
    });

    expect(provider.id).toBe('groq');
    expect(provider.name).toBe('Groq');
    expect(provider.defaultModel).toBe('llama-3.3-70b-versatile');
  });
});
