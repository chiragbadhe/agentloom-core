import { describe, expect, it } from 'vitest';

import { ProviderError } from '../../src/errors.js';
import {
  BaseProvider,
  addUsage,
  emptyUsage,
  normalizeFinishReason,
} from '../../src/providers/base.js';
import type {
  CompletionRequest,
  CompletionResult,
  ModelProvider,
  ProviderCapabilities,
  StreamEvent,
} from '../../src/providers/types.js';

const CAPABILITIES: ProviderCapabilities = {
  tools: false,
  parallelToolCalls: false,
  streaming: false,
  systemMessages: true,
  jsonMode: false,
  strictJsonSchema: false,
  vision: false,
  promptCaching: false,
};

function request(overrides: Partial<CompletionRequest> = {}): CompletionRequest {
  return {
    model: 'test-1',
    messages: [{ role: 'user', content: 'hi' }],
    ...overrides,
  };
}

function result(overrides: Partial<CompletionResult> = {}): CompletionResult {
  return {
    message: { role: 'assistant', content: 'hello' },
    finishReason: 'stop',
    usage: {},
    responseId: 'resp_1',
    providerId: 'test',
    model: 'test-1',
    latencyMs: 1,
    raw: {},
    ...overrides,
  };
}

/** Minimal provider that returns a scripted result; no HTTP involved. */
class ScriptedProvider extends BaseProvider {
  readonly id = 'test';
  readonly name = 'Test';
  override defaultModel: string;
  readonly capabilities = CAPABILITIES;
  attempts = 0;

  constructor(
    private readonly script: CompletionResult[] | ((attempt: number) => CompletionResult),
    options: ConstructorParameters<typeof BaseProvider>[0] = {},
    model = 'test-1',
  ) {
    super(options);
    this.defaultModel = model;
  }

  protected async doComplete(): Promise<CompletionResult> {
    this.attempts++;
    const entry =
      typeof this.script === 'function'
        ? this.script(this.attempts)
        : this.script[Math.min(this.attempts - 1, this.script.length - 1)];
    if (entry === undefined) throw new ProviderError('no script entry');
    return entry;
  }
}

async function collect(stream: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe('BaseProvider', () => {
  it('returns the provider result', async () => {
    const provider = new ScriptedProvider([
      result({ message: { role: 'assistant', content: 'x' } }),
    ]);
    const completion = await provider.complete(request());
    expect(completion.message.content).toBe('x');
    expect(completion.providerId).toBe('test');
  });

  it('retries until the attempt succeeds', async () => {
    const provider = new ScriptedProvider(
      (attempt) => {
        if (attempt < 3) throw new ProviderError('flaky', { retryable: true });
        return result();
      },
      { retry: { maxAttempts: 5, initialDelayMs: 1 } },
    );

    await expect(provider.complete(request())).resolves.toMatchObject({
      responseId: 'resp_1',
    });
    expect(provider.attempts).toBe(3);
  });

  it('gives up after maxAttempts', async () => {
    const provider = new ScriptedProvider(
      () => {
        throw new ProviderError('always down', { retryable: true });
      },
      { retry: { maxAttempts: 2, initialDelayMs: 1 } },
    );

    await expect(provider.complete(request())).rejects.toThrow('always down');
    expect(provider.attempts).toBe(2);
  });

  it('does not retry non-retryable errors', async () => {
    const provider = new ScriptedProvider(
      () => {
        throw new ProviderError('bad request', { retryable: false });
      },
      { retry: { maxAttempts: 4, initialDelayMs: 1 } },
    );

    await expect(provider.complete(request())).rejects.toThrow('bad request');
    expect(provider.attempts).toBe(1);
  });

  it('skips retries when retry is false', async () => {
    const provider = new ScriptedProvider(
      () => {
        throw new ProviderError('down', { retryable: true });
      },
      { retry: false },
    );

    await expect(provider.complete(request())).rejects.toThrow('down');
    expect(provider.attempts).toBe(1);
  });

  it('honours the abort signal', async () => {
    const controller = new AbortController();
    const provider = new ScriptedProvider(
      () => {
        controller.abort();
        throw new ProviderError('aborted', { retryable: true });
      },
      { retry: { maxAttempts: 5, initialDelayMs: 50 } },
    );

    await expect(
      provider.complete(request(), { signal: controller.signal }),
    ).rejects.toThrow();
    expect(provider.attempts).toBe(1);
  });

  it('falls back to a buffered stream', async () => {
    const provider = new ScriptedProvider([
      result({
        message: {
          role: 'assistant',
          content: 'streamed',
          toolCalls: [{ id: 'call_1', name: 'noop', arguments: {} }],
        },
        finishReason: 'tool_calls',
      }),
    ]);

    const events = await collect(provider.stream(request()));
    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text-delta',
      'tool-call',
      'finish',
    ]);

    const deltas = events.filter((event) => event.type === 'text-delta');
    expect(deltas[0]).toMatchObject({ text: 'streamed' });

    const finished = events.at(-1);
    expect(finished?.type === 'finish' && finished.result.finishReason).toBe(
      'tool_calls',
    );
  });

  it('skips the text delta for an empty message', async () => {
    const provider = new ScriptedProvider([
      result({ message: { role: 'assistant', content: '' } }),
    ]);
    const events = await collect(provider.stream(request()));
    expect(events.some((event) => event.type === 'text-delta')).toBe(false);
  });

  it('propagates buffered stream failures', async () => {
    const provider = new ScriptedProvider(() => {
      throw new ProviderError('stream failed');
    });
    await expect(collect(provider.stream(request()))).rejects.toThrow('stream failed');
  });

  it('clones with a different default model', async () => {
    const provider = new ScriptedProvider([result()], {}, 'model-a');
    const clone = provider.withModel('model-b');

    expect(clone.defaultModel).toBe('model-b');
    expect(provider.defaultModel).toBe('model-a');
    expect(clone).toBeInstanceOf(ScriptedProvider);
    expect(clone.attempts).toBe(0);
  });

  it('satisfies the ModelProvider contract', async () => {
    const provider: ModelProvider = new ScriptedProvider([result()]);
    expect(provider.id).toBe('test');
    expect(provider.capabilities).toBe(CAPABILITIES);
  });
});

describe('normalizeFinishReason', () => {
  it.each([
    ['stop', 'stop'],
    ['end_turn', 'stop'],
    ['STOP', 'stop'],
    ['length', 'length'],
    ['max_tokens', 'length'],
    ['tool_use', 'tool_calls'],
    ['tool_calls', 'tool_calls'],
    ['content_filter', 'content_filter'],
    ['SAFETY', 'content_filter'],
    ['refusal', 'content_filter'],
    ['error', 'error'],
  ] as const)('maps %s to %s', (input, expected) => {
    expect(normalizeFinishReason(input)).toBe(expected);
  });

  it('falls back to other for unknown values', () => {
    expect(normalizeFinishReason('weird')).toBe('other');
    expect(normalizeFinishReason(undefined)).toBe('other');
  });
});

describe('usage helpers', () => {
  it('emptyUsage has no fields', () => {
    expect(emptyUsage()).toEqual({});
  });

  it('adds defined fields only', () => {
    expect(addUsage({ inputTokens: 1, outputTokens: 2 }, { inputTokens: 3 })).toEqual({
      inputTokens: 4,
      outputTokens: 2,
      totalTokens: undefined,
      cachedInputTokens: undefined,
      reasoningTokens: undefined,
    });
  });

  it('keeps missing values undefined', () => {
    const sum = addUsage({}, {});
    expect(sum.inputTokens).toBeUndefined();
    expect(sum.totalTokens).toBeUndefined();
  });

  it('sums totals when reported', () => {
    expect(addUsage({ totalTokens: 5 }, { totalTokens: 6 }).totalTokens).toBe(11);
  });
});
