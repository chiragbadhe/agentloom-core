import type {
  CompletionRequest,
  CompletionResult,
  ModelProvider,
  ProviderCapabilities,
  ProviderCallOptions,
  StreamEvent,
} from '../../src/providers/types.js';

/** A scripted turn: either a tool call or a final answer. */
export interface ScriptedTurn {
  /** Text the assistant returns. */
  readonly text?: string;
  /** Tool calls the model requests. */
  readonly toolCalls?: readonly {
    readonly id: string;
    readonly name: string;
    readonly arguments: unknown;
  }[];
  /** Reject instead of responding. */
  readonly error?: Error;
  /** Emit a partial response first, then this turn's content (streaming only). */
  readonly chunks?: readonly string[];
  readonly finishReason?: CompletionResult['finishReason'];
}

export interface MockProviderOptions {
  readonly id?: string;
  readonly defaultModel?: string;
  readonly capabilities?: Partial<ProviderCapabilities>;
  /** Per-call stream events, used instead of `chunks` when provided. */
  readonly script?: readonly ScriptedTurn[];
  /** Reject the run once the script is exhausted. Default `true`. */
  readonly throwWhenExhausted?: boolean;
}

export const FULL_CAPABILITIES: ProviderCapabilities = {
  tools: true,
  parallelToolCalls: true,
  streaming: true,
  systemMessages: true,
  jsonMode: true,
  strictJsonSchema: true,
  vision: true,
  promptCaching: true,
};

let callCounter = 0;

/**
 * A deterministic {@link ModelProvider} for tests.
 *
 * It replays a fixed script, records every request it received, and never
 * touches the network.
 */
export class MockProvider implements ModelProvider {
  readonly id: string;
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities;

  /** Every request handed to `complete`/`stream`, in order. */
  readonly requests: CompletionRequest[] = [];
  /** Every signal seen, in order — useful for abort assertions. */
  readonly signals: (AbortSignal | undefined)[] = [];

  private readonly script: readonly ScriptedTurn[];
  private readonly throwWhenExhausted: boolean;
  /** Script cursor. Package-visible so `withModel` clones can inherit it. */
  cursor = 0;

  constructor(options: MockProviderOptions = {}) {
    this.id = options.id ?? 'mock';
    this.name = 'Mock Provider';
    this.defaultModel = options.defaultModel ?? 'mock-1';
    this.capabilities = { ...FULL_CAPABILITIES, ...options.capabilities };
    this.script = options.script ?? [{ text: 'ok' }];
    this.throwWhenExhausted = options.throwWhenExhausted ?? true;
  }

  /** Number of provider calls made so far. */
  get callCount(): number {
    return this.requests.length;
  }

  /** The most recent request, for asserting on tools/formats. */
  get lastRequest(): CompletionRequest | undefined {
    return this.requests.at(-1);
  }

  /** Reset the script cursor and recorded requests. */
  reset(): void {
    this.cursor = 0;
    this.requests.length = 0;
    this.signals.length = 0;
  }

  async complete(
    request: CompletionRequest,
    options?: ProviderCallOptions,
  ): Promise<CompletionResult> {
    this.requests.push(request);
    this.signals.push(options?.signal);

    const turn = this.next();
    if (turn.error !== undefined) throw turn.error;

    return {
      message: {
        role: 'assistant',
        content: turn.text ?? '',
        ...(turn.toolCalls === undefined
          ? {}
          : {
              toolCalls: turn.toolCalls.map((call) => ({
                id: call.id,
                name: call.name,
                arguments: call.arguments,
                rawArguments: JSON.stringify(call.arguments),
              })),
            }),
      },
      finishReason:
        turn.finishReason ?? (turn.toolCalls === undefined ? 'stop' : 'tool_calls'),
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      },
      responseId: `resp_${++callCounter}`,
      providerId: this.id,
      model: request.model,
      latencyMs: 1,
      raw: {},
    };
  }

  async *stream(
    request: CompletionRequest,
    options?: ProviderCallOptions,
  ): AsyncIterable<StreamEvent> {
    this.requests.push(request);
    this.signals.push(options?.signal);

    const turn = this.next();
    const responseId = `resp_${++callCounter}`;

    if (turn.error !== undefined) throw turn.error;

    yield { type: 'start', model: request.model, responseId };

    for (const chunk of turn.chunks ?? (turn.text === undefined ? [] : [turn.text])) {
      yield { type: 'text-delta', text: chunk, responseId };
    }

    for (const call of turn.toolCalls ?? []) {
      yield {
        type: 'tool-call',
        responseId,
        call: {
          id: call.id,
          name: call.name,
          arguments: call.arguments,
          rawArguments: JSON.stringify(call.arguments),
        },
      };
    }

    yield {
      type: 'finish',
      result: {
        message: {
          role: 'assistant',
          content: turn.text ?? '',
          ...(turn.toolCalls === undefined
            ? {}
            : {
                toolCalls: turn.toolCalls.map((call) => ({
                  id: call.id,
                  name: call.name,
                  arguments: call.arguments,
                })),
              }),
        },
        finishReason:
          turn.finishReason ?? (turn.toolCalls === undefined ? 'stop' : 'tool_calls'),
        usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        responseId,
        providerId: this.id,
        model: request.model,
        latencyMs: 1,
        raw: {},
      },
    };
  }

  withModel(model: string): ModelProvider {
    const clone = new MockProvider({
      id: this.id,
      defaultModel: model,
      capabilities: this.capabilities,
      script: this.script,
      throwWhenExhausted: this.throwWhenExhausted,
    });
    // Share the cursor so `withModel` continues the same script.
    clone.cursor = this.cursor;
    return clone;
  }

  private next(): ScriptedTurn {
    const turn = this.script[this.cursor];
    this.cursor += 1;
    if (turn === undefined) {
      if (this.throwWhenExhausted) {
        throw new Error(
          `MockProvider script exhausted after ${this.script.length} turn(s)`,
        );
      }
      return { text: 'done' };
    }
    return turn;
  }
}

/** A provider whose `stream` is unsupported, to exercise the buffered fallback. */
export class NonStreamingProvider extends MockProvider {
  override capabilities: ProviderCapabilities;

  constructor(options: MockProviderOptions = {}) {
    super(options);
    this.capabilities = { ...FULL_CAPABILITIES, streaming: false };
  }

  override stream(): AsyncIterable<StreamEvent> {
    throw new Error('streaming is not supported by this provider');
  }
}

/** Build a mock provider in one call. */
export const mockProvider = (options?: MockProviderOptions): MockProvider =>
  new MockProvider(options);
