import { createId } from '../utils/id.js';
import { noopLogger, type Logger } from '../utils/logger.js';
import { withRetry, type RetryOptions } from '../utils/retry.js';
import { HttpClient, type HttpClientOptions } from './http.js';
import type {
  CompletionRequest,
  CompletionResult,
  FinishReason,
  ModelProvider,
  ProviderCallOptions,
  ProviderCapabilities,
  StreamEvent,
  Usage,
} from './types.js';

export interface BaseProviderOptions extends HttpClientOptions {
  readonly retry?: RetryOptions | false;
  readonly logger?: Logger;
  readonly defaultTimeoutMs?: number;
  /** Extra headers merged into every request (lowercased keys). */
  readonly headers?: Readonly<Record<string, string>>;
}

/**
 * Shared provider plumbing: HTTP access, retry policy, response-id assignment,
 * and a streaming fallback.
 *
 * Extend this and implement {@link BaseProvider.doComplete}; you get retries,
 * timeouts, and abort handling for free. Override {@link BaseProvider.doStream}
 * when the vendor has a real streaming API.
 */
export abstract class BaseProvider implements ModelProvider {
  abstract readonly id: string;
  abstract readonly name: string;
  abstract readonly defaultModel: string;
  abstract readonly capabilities: ProviderCapabilities;

  protected readonly http: HttpClient;
  protected readonly logger: Logger;
  protected readonly retryOptions: RetryOptions | undefined;

  constructor(options: BaseProviderOptions = {}) {
    this.http = new HttpClient({
      fetch: options.fetch,
      defaultTimeoutMs: options.defaultTimeoutMs,
      defaultHeaders: options.defaultHeaders,
    });
    this.logger = options.logger ?? noopLogger();
    this.retryOptions = options.retry === false ? undefined : options.retry;
  }

  /** Vendor-specific single-shot call. Rejects with an `AgentError`. */
  protected abstract doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult>;

  /** Vendor-specific streaming call. Defaults to buffering {@link complete}. */
  protected doStream(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): AsyncIterable<StreamEvent> {
    return bufferedStream(this, request, options);
  }

  async complete(
    request: CompletionRequest,
    options: ProviderCallOptions = {},
  ): Promise<CompletionResult> {
    const log = this.logger.child({ provider: this.id, model: request.model });

    if (this.retryOptions === undefined) {
      return this.doComplete(request, options);
    }

    return withRetry(
      async (attempt) => {
        log.debug('completion attempt', { attempt, messages: request.messages.length });
        return this.doComplete(request, options);
      },
      { signal: options.signal, ...this.retryOptions },
    );
  }

  stream(
    request: CompletionRequest,
    options: ProviderCallOptions = {},
  ): AsyncIterable<StreamEvent> {
    return this.doStream(request, options);
  }

  /**
   * A copy of this provider whose default model is `model`. Implemented via a
   * prototype-preserving clone so subclasses keep their private state.
   */
  withModel(model: string): this {
    const clone = Object.create(Object.getPrototypeOf(this) as object) as this;
    Object.assign(clone, this);
    Object.defineProperty(clone, 'defaultModel', { value: model, enumerable: true });
    return clone;
  }
}

/**
 * Turn a non-streaming provider into a stream by buffering. Emits `start`,
 * one `text-delta`, every `tool-call`, then `finish` — enough for callers that
 * only need uniform event plumbing.
 */
async function* bufferedStream(
  provider: ModelProvider,
  request: CompletionRequest,
  options: ProviderCallOptions,
): AsyncGenerator<StreamEvent> {
  const responseId = createId('resp');
  yield { type: 'start', model: request.model, responseId };

  const result = await provider.complete(request, options);

  if (result.message.content) {
    yield { type: 'text-delta', text: result.message.content, responseId };
  }
  for (const call of result.message.toolCalls ?? []) {
    yield { type: 'tool-call', call, responseId };
  }
  yield { type: 'finish', result: { ...result, responseId } };
}

/**
 * Resolve an API key from an explicit option or the environment.
 *
 * Blank values are treated as absent, so an unset-but-defined variable or an
 * empty string never turns into an `Authorization: Bearer ` header.
 */
export function resolveApiKey(
  explicit: string | undefined,
  envVars: readonly string[],
): string | undefined {
  const candidates = [explicit, ...envVars.map((name) => process.env[name])];
  for (const candidate of candidates) {
    const trimmed = candidate?.trim();
    if (trimmed) return trimmed;
  }
  return undefined;
}

/** Zero-valued usage, used when a provider omits counts. */
export function emptyUsage(): Usage {
  return {};
}

/** Sum usage across several completions. */
export function addUsage(a: Usage, b: Usage): Usage {
  const pick = (x?: number, y?: number): number | undefined =>
    x === undefined && y === undefined ? undefined : (x ?? 0) + (y ?? 0);
  const total =
    a.totalTokens !== undefined || b.totalTokens !== undefined
      ? (a.totalTokens ?? 0) + (b.totalTokens ?? 0)
      : undefined;
  return {
    inputTokens: pick(a.inputTokens, b.inputTokens),
    outputTokens: pick(a.outputTokens, b.outputTokens),
    totalTokens: total,
    cachedInputTokens:
      a.cachedInputTokens === undefined && b.cachedInputTokens === undefined
        ? undefined
        : (a.cachedInputTokens ?? 0) + (b.cachedInputTokens ?? 0),
    reasoningTokens:
      a.reasoningTokens === undefined && b.reasoningTokens === undefined
        ? undefined
        : (a.reasoningTokens ?? 0) + (b.reasoningTokens ?? 0),
  };
}

/** Normalise vendor finish reasons onto the shared union. */
export function normalizeFinishReason(reason: string | undefined): FinishReason {
  switch (reason) {
    case 'stop':
    case 'end_turn':
    case 'STOP':
    case 'stop_sequence':
      return 'stop';
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
    case 'model_length':
      return 'length';
    case 'tool_calls':
    case 'tool_use':
    case 'function_call':
      return 'tool_calls';
    case 'content_filter':
    case 'SAFETY':
    case 'refusal':
    case 'RECITATION':
      return 'content_filter';
    case 'error':
      return 'error';
    default:
      return 'other';
  }
}
