import { describe, expect, it } from 'vitest';

import { ProviderResponseError } from '../../src/errors.js';
import {
  createGeminiProvider,
  createGoogleProvider,
} from '../../src/providers/google.js';
import type { CompletionRequest, StreamEvent } from '../../src/providers/types.js';
import { schemaFromJsonSchema } from '../../src/schema.js';
import { stubFetch } from '../helpers/mock-fetch.js';

const request = (overrides: Partial<CompletionRequest> = {}): CompletionRequest => ({
  model: 'gemini-2.0-flash',
  messages: [{ role: 'user', content: 'hello' }],
  ...overrides,
});

const textResponse = (text: string) => ({
  body: JSON.stringify({
    responseId: 'resp-1',
    candidates: [{ content: { role: 'model', parts: [{ text }] }, finishReason: 'STOP' }],
    usageMetadata: {
      promptTokenCount: 8,
      candidatesTokenCount: 3,
      totalTokenCount: 11,
      cachedContentTokenCount: 1,
      thoughtsTokenCount: 2,
    },
  }),
});

async function collect(stream: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function sse(...frames: readonly string[]): readonly string[] {
  return frames.map((frame) => `data: ${frame}\n\n`);
}

describe('GoogleProvider', () => {
  it('reports identity and capabilities', () => {
    const provider = createGoogleProvider();
    expect(provider.id).toBe('google');
    expect(provider.name).toBe('Google Gemini');
    expect(provider.defaultModel).toBe('gemini-2.0-flash');
    expect(provider.capabilities.parallelToolCalls).toBe(false);
    expect(provider.capabilities.strictJsonSchema).toBe(true);
  });

  it('is also exported as createGeminiProvider', () => {
    expect(createGeminiProvider).toBe(createGoogleProvider);
  });

  it('posts to the model endpoint with the API key', async () => {
    const fetchImpl = stubFetch(textResponse('hi'));
    const completion = await createGoogleProvider({
      apiKey: 'g-key',
      fetch: fetchImpl,
    }).complete(request());

    expect(fetchImpl.calls[0]?.url).toBe(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=g-key',
    );
    expect(completion.message.content).toBe('hi');
    expect(completion.responseId).toBe('resp-1');
    expect(completion.finishReason).toBe('stop');
    expect(completion.usage).toEqual({
      inputTokens: 8,
      outputTokens: 3,
      totalTokens: 11,
      cachedInputTokens: 1,
      reasoningTokens: 2,
    });
  });

  it('omits the key parameter when unauthenticated', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: '', fetch: fetchImpl }).complete(request());
    expect(fetchImpl.calls[0]?.url).not.toContain('key=');
  });

  it('reads GEMINI_API_KEY as a fallback', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    const previous = process.env['GEMINI_API_KEY'];
    process.env['GEMINI_API_KEY'] = 'from-env';
    try {
      await createGoogleProvider({ fetch: fetchImpl }).complete(request());
      expect(fetchImpl.calls[0]?.url).toContain('key=from-env');
    } finally {
      if (previous === undefined) delete process.env['GEMINI_API_KEY'];
      else process.env['GEMINI_API_KEY'] = previous;
    }
  });

  it('honours a custom base URL', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({
      apiKey: 'k',
      baseURL: 'https://vertex.test/v1beta/',
      fetch: fetchImpl,
    }).complete(request());
    expect(fetchImpl.calls[0]?.url).toContain('https://vertex.test/v1beta/models/');
  });

  it('maps system messages to systemInstruction', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'system', content: 'be terse' },
          { role: 'system', content: 'be kind' },
          { role: 'user', content: 'hi' },
        ],
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      systemInstruction: { parts: [{ text: 'be terse\n\nbe kind' }] },
      contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
    });
  });

  it('maps tool calls and results to Gemini parts', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          { role: 'user', content: 'list files' },
          {
            role: 'assistant',
            content: '',
            toolCalls: [
              { id: 'call_1', name: 'list_directory', arguments: { path: '.' } },
            ],
          },
          {
            role: 'tool',
            content: 'a.txt',
            toolCallId: 'call_1',
            name: 'list_directory',
          },
        ],
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      contents: [
        { role: 'user', parts: [{ text: 'list files' }] },
        {
          role: 'model',
          parts: [
            {
              functionCall: { id: 'call_1', name: 'list_directory', args: { path: '.' } },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_1',
                name: 'list_directory',
                response: { result: 'a.txt' },
              },
            },
          ],
        },
      ],
    });
  });

  it('wraps tool errors for the model', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        messages: [
          {
            role: 'tool',
            content: 'denied',
            toolCallId: 'call_1',
            name: 'delete_file',
            isError: true,
          },
        ],
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      contents: [
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_1',
                name: 'delete_file',
                response: { error: 'denied' },
              },
            },
          ],
        },
      ],
    });
  });

  it('snake_cases tool names on the wire and back', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({ tools: [{ name: 'list_directory', description: 'List' }] }),
    );
    expect(fetchImpl.bodies()[0]).toMatchObject({
      tools: [
        { functionDeclarations: [{ name: 'list_directory', description: 'List' }] },
      ],
      toolConfig: { functionCallingConfig: { mode: 'AUTO' } },
    });

    const callResponse = stubFetch({
      body: JSON.stringify({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [{ functionCall: { id: 'c1', name: 'list_directory', args: {} } }],
            },
            finishReason: 'STOP',
          },
        ],
      }),
    });
    const completion = await createGoogleProvider({
      apiKey: 'k',
      fetch: callResponse,
    }).complete(request());
    expect(completion.message.toolCalls?.[0]?.name).toBe('listDirectory');
    expect(completion.finishReason).toBe('stop');
  });

  it('strips schema keywords Gemini rejects', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(
      request({
        tools: [
          {
            name: 'search',
            description: 'Search',
            parameters: schemaFromJsonSchema({
              $schema: 'https://json-schema.org/draft/2020-12/schema',
              type: 'object',
              additionalProperties: false,
              properties: {
                q: { type: 'string', default: 'x' },
                nested: { $ref: '#/$defs/inner' },
              },
              $defs: { inner: { type: 'string' } },
            }),
          },
        ],
      }),
    );

    const parameters = (
      fetchImpl.bodies()[0] as {
        tools: { functionDeclarations: { parameters: unknown }[] }[];
      }
    ).tools[0]!.functionDeclarations[0]!.parameters as Record<string, unknown>;
    expect(parameters).not.toHaveProperty('$schema');
    expect(parameters).not.toHaveProperty('additionalProperties');
    expect(parameters).not.toHaveProperty('$defs');
    expect(parameters['properties']).toMatchObject({ q: { type: 'string' } });
  });

  it('maps generation options and json modes', async () => {
    const fetchImpl = stubFetch([
      textResponse('a'),
      textResponse('b'),
      textResponse('c'),
    ]);
    const provider = createGoogleProvider({ apiKey: 'k', fetch: fetchImpl });

    await provider.complete(
      request({
        temperature: 0.4,
        topP: 0.7,
        topK: 30,
        maxTokens: 64,
        stopSequences: ['X'],
        seed: 9,
      }),
    );
    await provider.complete(request({ responseFormat: { type: 'json_object' } }));
    await provider.complete(
      request({
        responseFormat: {
          type: 'json_schema',
          schema: { type: 'object', properties: { a: { type: 'number' } } },
        },
      }),
    );

    expect(fetchImpl.bodies()[0]).toMatchObject({
      generationConfig: {
        temperature: 0.4,
        topP: 0.7,
        topK: 30,
        maxOutputTokens: 64,
        stopSequences: ['X'],
        seed: 9,
      },
    });
    expect(fetchImpl.bodies()[1]).toMatchObject({
      generationConfig: { responseMimeType: 'application/json' },
    });
    expect(fetchImpl.bodies()[2]).toMatchObject({
      generationConfig: {
        responseMimeType: 'application/json',
        responseSchema: { type: 'object', properties: { a: { type: 'number' } } },
      },
    });
  });

  it('omits generationConfig when nothing is configured', async () => {
    const fetchImpl = stubFetch(textResponse('ok'));
    await createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(request());
    expect(fetchImpl.bodies()[0]).not.toHaveProperty('generationConfig');
  });

  it('skips thought parts', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [{ text: 'internal', thought: true }, { text: 'answer' }],
            },
            finishReason: 'STOP',
          },
        ],
      }),
    });
    const completion = await createGoogleProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.content).toBe('answer');
  });

  it('synthesises tool call ids', async () => {
    const fetchImpl = stubFetch({
      body: JSON.stringify({
        candidates: [
          { content: { role: 'model', parts: [{ functionCall: { name: 'ping' } }] } },
        ],
      }),
    });
    const completion = await createGoogleProvider({
      apiKey: 'k',
      fetch: fetchImpl,
    }).complete(request());
    expect(completion.message.toolCalls?.[0]).toMatchObject({
      id: 'call_0',
      arguments: {},
    });
  });

  it('errors when there are no candidates', async () => {
    const fetchImpl = stubFetch({ body: JSON.stringify({ candidates: [] }) });
    await expect(
      createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).complete(request()),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('streams text deltas', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        JSON.stringify({ candidates: [{ content: { parts: [{ text: 'Hel' }] } }] }),
        JSON.stringify({
          candidates: [{ content: { parts: [{ text: 'lo' }] }, finishReason: 'STOP' }],
          usageMetadata: {
            promptTokenCount: 1,
            candidatesTokenCount: 2,
            totalTokenCount: 3,
          },
        }),
      ),
    });

    const events = await collect(
      createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text-delta',
      'text-delta',
      'finish',
    ]);
    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.message.content).toBe('Hello');
    expect(finished?.type === 'finish' && finished.result.usage.totalTokens).toBe(3);
    expect(finished?.type === 'finish' && finished.result.finishReason).toBe('stop');
  });

  it('streams tool calls', async () => {
    const fetchImpl = stubFetch({
      chunks: sse(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [
                  {
                    functionCall: {
                      id: 'c1',
                      name: 'list_directory',
                      args: { path: '.' },
                    },
                  },
                ],
              },
            },
          ],
        }),
        JSON.stringify({ candidates: [{ finishReason: 'STOP' }] }),
      ),
    });

    const events = await collect(
      createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );
    const toolEvent = events.find((event) => event.type === 'tool-call');
    expect(toolEvent?.type === 'tool-call' && toolEvent.call.name).toBe('listDirectory');
  });

  it('requests the SSE streaming endpoint', async () => {
    const fetchImpl = stubFetch({ chunks: sse('{}') });
    await collect(
      createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request()),
    );

    expect(fetchImpl.calls[0]?.url).toContain(':streamGenerateContent');
    expect(fetchImpl.calls[0]?.url).toContain('alt=sse');
  });

  it('errors when the response has no readable stream', async () => {
    const fetchImpl = stubFetch({ body: 'x' });
    await expect(
      collect(createGoogleProvider({ apiKey: 'k', fetch: fetchImpl }).stream(request())),
    ).rejects.toThrow(/readable stream/);
  });

  it('surfaces HTTP errors', async () => {
    const fetchImpl = stubFetch({
      status: 403,
      body: JSON.stringify({ error: { message: 'denied' } }),
    });
    await expect(
      createGoogleProvider({ apiKey: 'k', fetch: fetchImpl, retry: false }).complete(
        request(),
      ),
    ).rejects.toThrow(/denied/);
  });
});
