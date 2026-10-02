import { describe, expect, it } from 'vitest';

import { ProviderResponseError } from '../../src/errors.js';
import { createAnthropicProvider } from '../../src/providers/anthropic.js';
import type { CompletionRequest, StreamEvent } from '../../src/providers/types.js';
import { schemaFromJsonSchema } from '../../src/schema.js';
import { stubFetch } from '../helpers/mock-fetch.js';

const request = (overrides: Partial<CompletionRequest> = {}): CompletionRequest => ({
  model: 'claude-sonnet-4-5',
  messages: [{ role: 'user', content: 'hello' }],
  ...overrides,
});

const textResponse = (text: string) => ({
  body: JSON.stringify({
    id: 'msg_1',
    model: 'claude-sonnet-4-5-20250929',
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 12, output_tokens: 5, cache_read_input_tokens: 4 },
  }),
});

async function collect(stream: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function sse(...frames: readonly [string, string][]): readonly string[] {
  return frames.map(([event, data]) => `event: ${event}\ndata: ${data}\n\n`);
}

describe('AnthropicProvider', () => {
  it('reports identity and capabilities', () => {
    const provider = createAnthropicProvider();
    expect(provider.id).toBe('anthropic');
    expect(provider.name).toBe('Anthropic');
    expect(provider.defaultModel).toBe('claude-sonnet-4-5');
    expect(provider.capabilities.jsonMode).toBe(false);
    expect(provider.capabilities.strictJsonSchema).toBe(false);
  });

  it('posts to the messages endpoint with the required headers', async () => {
    const fetchImpl = stubFetch(textResponse('hi'));
    const completion = await createAnthropicProvider({
      apiKey: 'sk-ant',
      fetch: fetchImpl,
    }).complete(request());

    const call = fetchImpl.calls[0];
    expect(call?.url).toBe('https://api.anthropic.com/v1/messages');
    expect(call?.init.headers['x-api-key']).toBe('sk-ant');
    expect(call?.init.headers['anthropic-version']).toBe('2023-06-01');

    expect(completion.message.content).toBe('hi');
    expect(completion.responseId).toBe('msg_1');
    expect(completion.model).toBe('claude-sonnet-4-5-20250929');
    expect(completion.finishReason).toBe('stop');
    expect(completion.usage).toEqual({
      inputTokens: 12,
      outputTokens: 5,
      totalTokens: 17,
      cachedInputTokens: 4,
    });
  });

  it('honours a custom api version and base URL', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({
      apiKey: 'k',
      baseURL: 'https://proxy.test/',
      apiVersion: '2024-01-01',
      fetch: fetchImpl,
    }).complete(request());

    expect(fetchImpl.calls[0]?.url).toBe('https://proxy.test/v1/messages');
    expect(fetchImpl.calls[0]?.init.headers['anthropic-version']).toBe('2024-01-01');
  });

  it('lifts system messages to the top-level system field', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'system', content: 'be terse' },
          { role: 'system', content: 'be kind' },
          { role: 'user', content: 'hi' },
        ],
      }),
    );

    const body = fetchImpl.bodies()[0] as Record<string, unknown>;
    expect(body.system).toBe('be terse\n\nbe kind');
    expect(body.messages).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    ]);
  });

  it('always sends max_tokens', async () => {
    const fetchImpl = stubFetch([textResponse('a'), textResponse('b')]);
    const provider = createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl });

    await provider.complete(request());
    await provider.complete(request({ maxTokens: 128 }));

    expect(fetchImpl.bodies()[0]).toMatchObject({ max_tokens: 4096 });
    expect(fetchImpl.bodies()[1]).toMatchObject({ max_tokens: 128 });
  });

  it('maps tool results to tool_result user messages', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'user', content: 'add them' },
          {
            role: 'assistant',
            content: '',
            toolCalls: [{ id: 'toolu_1', name: 'add', arguments: { a: 1, b: 2 } }],
          },
          {
            role: 'tool',
            content: 'boom',
            toolCallId: 'toolu_1',
            name: 'add',
            isError: true,
          },
        ],
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'add them' }] },
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_1', name: 'add', input: { a: 1, b: 2 } },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_1',
              content: 'boom',
              is_error: true,
            },
          ],
        },
      ],
    });
  });

  it('never sends an empty assistant content array', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({ messages: [{ role: 'assistant', content: '' }] }),
    );
    expect(fetchImpl.bodies()[0]).toMatchObject({
      messages: [{ role: 'assistant', content: [{ type: 'text', text: '' }] }],
    });
  });

  it('sends tools as input_schema declarations', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
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
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      tools: [
        {
          name: 'search',
          description: 'Search',
          input_schema: { type: 'object', properties: { q: { type: 'string' } } },
        },
      ],
    });
  });

  it('passes sampling options through', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        temperature: 0.3,
        topP: 0.8,
        topK: 20,
        stopSequences: ['END'],
        metadata: { userId: 'user-1' },
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      temperature: 0.3,
      top_p: 0.8,
      top_k: 20,
      stop_sequences: ['END'],
      metadata: { user_id: 'user-1' },
    });
  });

  it('turns a json response format into a system instruction', async () => {
    const fetchImpl = stubFetch([textResponse('a'), textResponse('b')]);
    const provider = createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl });

    await provider.complete(
      request({
        messages: [{ role: 'system', content: 'be terse' }],
        responseFormat: {
          type: 'json_schema',
          schema: { type: 'object', properties: { a: { type: 'number' } } },
        },
      }),
    );
    await provider.complete(request({ responseFormat: { type: 'json_object' } }));

    const first = fetchImpl.bodies()[0] as { system: string };
    expect(first.system).toContain('be terse');
    expect(first.system).toContain('validates against this JSON Schema');

    const second = fetchImpl.bodies()[1] as { system: string };
    expect(second.system).toContain('single valid JSON object');
  });

  it('parses text and tool_use blocks', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        id: 'msg_2',
        content: [
          { type: 'text', text: 'thinking' },
          { type: 'thinking', thinking: 'ignored' },
          { type: 'tool_use', id: 'toolu_9', name: 'add', input: { a: 1 } },
        ],
        stop_reason: 'tool_use',
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
    });

    const completion = await createAnthropicProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());

    expect(completion.message.content).toBe('thinking');
    expect(completion.message.toolCalls).toEqual([
      { id: 'toolu_9', name: 'add', arguments: { a: 1 }, rawArguments: '{"a":1}' },
    ]);
    expect(completion.finishReason).toBe('tool_calls');
  });

  it('synthesises ids for blocks without one', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({ content: [{ type: 'tool_use', name: 'noop', input: {} }] }),
    });
    const completion = await createAnthropicProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.toolCalls?.[0]?.id).toBe('call_0');
  });

  it('tolerates an empty content list', async () => {
    const fetchImpl = stubFetch({ body: JSON.stringify({ content: [] }) });
    const completion = await createAnthropicProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.content).toBe('');
    expect(completion.usage).toEqual({});
  });

  it('streams text deltas', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        [
          'message_start',
          JSON.stringify({
            type: 'message_start',
            message: { id: 'msg_3', model: 'claude-x', usage: { input_tokens: 7 } },
          }),
        ],
        [
          'content_block_start',
          JSON.stringify({
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'text' },
          }),
        ],
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'Hi' },
          }),
        ],
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: '!' },
          }),
        ],
        [
          'message_delta',
          JSON.stringify({
            type: 'message_delta',
            delta: { stop_reason: 'end_turn' },
            usage: { output_tokens: 3 },
          }),
        ],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });

    const events = await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text-delta',
      'text-delta',
      'finish',
    ]);

    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.content).toBe('Hi!');
    expect(finished?.type === 'finish' && finished.result.responseId).toBe('msg_3');
    expect(finished?.type === 'finish' && finished.result.model).toBe('claude-x');
    expect(finished?.type === 'finish' && finished.result.usage).toMatchObject({
      inputTokens: 7,
      outputTokens: 3,
    });
  });

  it('emits thinking deltas as reasoning', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'thinking_delta', thinking: 'hmm' },
          }),
        ],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });

    const events = await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    expect(events.some((event) => event.type === 'reasoning-delta')).toBe(true);
  });

  it('assembles streamed tool input fragments', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        [
          'content_block_start',
          JSON.stringify({
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'tool_use', id: 'toolu_1', name: 'add' },
          }),
        ],
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: '{"a":' },
          }),
        ],
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: '1}' },
          }),
        ],
        [
          'message_delta',
          JSON.stringify({ type: 'message_delta', delta: { stop_reason: 'tool_use' } }),
        ],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });

    const events = await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.toolCalls).toEqual([
      { id: 'toolu_1', name: 'add', arguments: { a: 1 }, rawArguments: '{"a":1}' },
    ]);
    expect(finished?.type === 'finish' && finished.result.finishReason).toBe(
      'tool_calls',
    );
  });

  it('defaults empty tool input to an empty object', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        [
          'content_block_start',
          JSON.stringify({
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'tool_use', id: 't', name: 'ping' },
          }),
        ],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });
    const events = await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    const finished = events.at(-1);
    expect(
      finished?.type === 'finish' && finished.result.message.toolCalls?.[0]?.arguments,
    ).toEqual({});
  });

  it('throws on a stream error frame', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(['error', JSON.stringify({ error: { message: 'overloaded' } })]),
    });
    await expect(
      collect(
        createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
      ),
    ).rejects.toThrow(/overloaded/);
  });

  it('ignores unparsable stream frames', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        ['ping', 'not json'],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });
    const events = await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    expect(events.at(-1)?.type).toBe('finish');
  });

  it('errors on malformed streamed tool input', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        [
          'content_block_start',
          JSON.stringify({
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'tool_use', id: 't', name: 'add' },
          }),
        ],
        [
          'content_block_delta',
          JSON.stringify({
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: 'nope' },
          }),
        ],
        ['message_stop', JSON.stringify({ type: 'message_stop' })],
      ),
    });

    await expect(
      collect(
        createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
      ),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('errors when the response has no readable stream', async () => {
    const fetchImpl = stubFetch({ body: 'x' });
    await expect(
      collect(
        createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
      ),
    ).rejects.toThrow(/readable stream/);
  });

  it('requests a stream and asks for SSE', async () => {
    const fetchImpl = stubFetch({ chunks: sse(['message_stop', '{}']) });
    await collect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({ stream: true });
    expect(fetchImpl.calls[0]?.init.headers['accept']).toBe('text/event-stream');
  });

  it('surfaces HTTP errors', async () => {
    const fetchImpl = stubFetch({
      status: 429,
      body: JSON.stringify({ error: { message: 'rate limit exceeded' } }),
    });
    await expect(
      createAnthropicProvider({ apiKey: 'k', fetch: fetchImpl, retry: false }).complete(
        request(),
      ),
    ).rejects.toThrow(/rate limit exceeded/);
  });
});
