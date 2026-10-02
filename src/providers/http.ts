import {
  AbortError,
  ProviderError,
  ProviderResponseError,
  ProviderTimeoutError,
  isRetryableStatus,
  providerErrorFromResponse,
} from '../errors.js';

// ---------------------------------------------------------------------------
// Minimal fetch surface
// ---------------------------------------------------------------------------

/**
 * The subset of `fetch` we rely on. Injecting this is how you point the kit at
 * a proxy, add tracing, or run tests without a network.
 */
export type FetchLike = (
  input: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
  },
) => Promise<FetchResponseLike>;

export interface FetchResponseLike {
  readonly status: number;
  readonly ok: boolean;
  readonly headers: { forEach(callback: (value: string, key: string) => void): void };
  readonly body?: unknown;
  text(): Promise<string>;
}

export interface HttpRequest {
  readonly url: string;
  readonly method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  readonly headers?: Readonly<Record<string, string>>;
  readonly body?: unknown;
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
  /** Provider id, used to annotate errors. */
  readonly providerId: string;
  /** Retryable contexts (e.g. the model id) recorded on thrown errors. */
  readonly context?: Record<string, unknown>;
}

export interface HttpResponse {
  readonly status: number;
  readonly ok: boolean;
  readonly headers: Readonly<Record<string, string>>;
  /** Buffered body. Only valid once; stream it instead for SSE. */
  text(): Promise<string>;
  /** Incremental body reader, or `null` if unavailable. */
  stream(): AsyncIterable<Uint8Array> | null;
  readonly retryAfterMs: number | undefined;
}

export interface HttpClientOptions {
  readonly fetch?: FetchLike;
  readonly defaultTimeoutMs?: number;
  readonly defaultHeaders?: Readonly<Record<string, string>>;
}

const DEFAULT_TIMEOUT_MS = 120_000;

function resolveGlobalFetch(): FetchLike {
  const candidate = (globalThis as { fetch?: unknown }).fetch;
  if (typeof candidate !== 'function') {
    throw new ProviderError(
      'global fetch is not available. Use Node 18+, or pass a custom `fetch` implementation.',
      { providerId: 'http', retryable: false },
    );
  }
  return candidate.bind(globalThis) as FetchLike;
}

function headersToRecord(headers: {
  forEach(cb: (v: string, k: string) => void): void;
}): Record<string, string> {
  const out: Record<string, string> = {};
  headers.forEach((value, key) => {
    out[key.toLowerCase()] = value;
  });
  return out;
}

/** Parse `Retry-After` (seconds or HTTP date) into milliseconds. */
export function parseRetryAfter(
  value: string | undefined,
  now = Date.now(),
): number | undefined {
  if (value === undefined) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  if (Number.isNaN(date)) return undefined;
  return Math.max(0, date - now);
}

/**
 * Thin HTTP layer shared by every provider: timeouts, abort propagation,
 * header merging, and error normalisation.
 */
export class HttpClient {
  private readonly fetchImpl: FetchLike | undefined;
  private readonly defaultTimeoutMs: number;
  private readonly defaultHeaders: Readonly<Record<string, string>>;

  constructor(options: HttpClientOptions = {}) {
    this.fetchImpl = options.fetch;
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.defaultHeaders = lowerKeys(options.defaultHeaders);
  }

  /**
   * Perform a request. Resolves for any HTTP status; rejects only for
   * transport-level failures and aborts. Use {@link expectOk} to turn a
   * non-2xx into a typed error.
   */
  async request(request: HttpRequest): Promise<HttpResponse> {
    const fetchImpl = this.fetchImpl ?? resolveGlobalFetch();
    const timeoutMs = request.timeoutMs ?? this.defaultTimeoutMs;
    const controller = new AbortController();
    let timedOut = false;

    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            controller.abort();
          }, timeoutMs)
        : undefined;

    const onExternalAbort = () => controller.abort(request.signal?.reason);
    if (request.signal) {
      if (request.signal.aborted) controller.abort(request.signal.reason);
      else request.signal.addEventListener('abort', onExternalAbort, { once: true });
    }

    const body = request.body === undefined ? undefined : JSON.stringify(request.body);
    const headers: Record<string, string> = {
      accept: 'application/json',
      ...this.defaultHeaders,
      ...lowerKeys(request.headers),
    };
    if (body !== undefined && headers['content-type'] === undefined) {
      headers['content-type'] = 'application/json';
    }

    try {
      const response = await fetchImpl(request.url, {
        method: request.method ?? 'POST',
        headers,
        body,
        signal: controller.signal,
      });
      const responseHeaders = headersToRecord(response.headers);
      const streamSource = asAsyncIterable(response.body);

      return {
        status: response.status,
        ok: response.ok,
        headers: responseHeaders,
        text: async () => {
          if (streamSource === null) return response.text();
          const decoder = new TextDecoder();
          let out = '';
          for await (const chunk of streamSource)
            out += decoder.decode(chunk, { stream: true });
          return out + decoder.decode();
        },
        stream: () => streamSource,
        retryAfterMs: parseRetryAfter(responseHeaders['retry-after']),
      };
    } catch (error) {
      if (timedOut) {
        throw new ProviderTimeoutError(
          `Request to ${request.providerId} timed out after ${timeoutMs}ms`,
          { providerId: request.providerId, details: { timeoutMs }, ...request.context },
        );
      }
      if (request.signal?.aborted) throw new AbortError();
      if (error instanceof ProviderError) throw error;
      throw new ProviderError(
        `Network request to ${request.providerId} failed: ${errorMessage(error)}`,
        { providerId: request.providerId, cause: error, ...request.context },
      );
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      request.signal?.removeEventListener('abort', onExternalAbort);
    }
  }

  /** Perform a request and reject with a typed error for non-2xx responses. */
  async expectOk(request: HttpRequest): Promise<HttpResponse> {
    const response = await this.request(request);
    if (!response.ok) {
      const body = await response.text();
      throw providerErrorFromResponse(response.status, body, {
        providerId: request.providerId,
        model:
          typeof request.context?.model === 'string' ? request.context.model : undefined,
        retryAfterMs: response.retryAfterMs,
      });
    }
    return response;
  }

  /** Perform a request and parse the JSON body, erroring on malformed JSON. */
  async json<T>(request: HttpRequest): Promise<T> {
    const response = await this.expectOk(request);
    const text = await response.text();
    return parseJsonBody<T>(text, request);
  }
}

function parseJsonBody<T>(text: string, request: HttpRequest): T {
  if (text.trim() === '') return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    throw new ProviderResponseError(
      `${request.providerId} returned a body that is not valid JSON: ${errorMessage(error)}`,
      {
        providerId: request.providerId,
        responseBody: text.slice(0, 500),
        ...request.context,
      },
    );
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function lowerKeys(
  input: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  const out: Record<string, string> = {};
  if (!input) return out;
  for (const [key, value] of Object.entries(input)) out[key.toLowerCase()] = value;
  return out;
}

function asAsyncIterable(body: unknown): AsyncIterable<Uint8Array> | null {
  if (body === null || body === undefined) return null;
  if (typeof body !== 'object') return null;
  const candidate = body as {
    getReader?: unknown;
    [Symbol.asyncIterator]?: unknown;
  };
  if (typeof candidate[Symbol.asyncIterator] === 'function') {
    return body as AsyncIterable<Uint8Array>;
  }
  if (typeof candidate.getReader === 'function') {
    const readerFactory = candidate.getReader as () => ReaderLike;
    return streamFromReader(readerFactory.call(body));
  }
  return null;
}

/** Structural view of the reader API we use. */
interface ReaderLike {
  read(): Promise<{ done?: boolean; value?: Uint8Array }>;
  releaseLock?(): void;
}

/** Adapt a `ReadableStreamDefaultReader` to an async iterable. */
function streamFromReader(reader: ReaderLike): AsyncIterable<Uint8Array> {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) return;
          if (value !== undefined) yield value;
        }
      } finally {
        reader.releaseLock?.();
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------

export interface SseEvent {
  readonly event: string | undefined;
  readonly data: string;
  readonly id: string | undefined;
}

/**
 * Parse a text/event-stream body into discrete SSE events.
 *
 * Handles chunk boundaries that split mid-frame (the common case with
 * proxies) and the `[DONE]` sentinel used by OpenAI and friends.
 */
export async function* parseSseStream(
  stream: AsyncIterable<Uint8Array>,
): AsyncGenerator<SseEvent> {
  const decoder = new TextDecoder();
  let buffer = '';

  for await (const chunk of stream) {
    buffer += decoder.decode(chunk, { stream: true });
    let separator = buffer.indexOf('\n\n');
    while (separator !== -1) {
      const frame = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const parsed = parseSseFrame(frame);
      if (parsed) yield parsed;
      separator = buffer.indexOf('\n\n');
    }
  }

  buffer += decoder.decode();
  const tail = parseSseFrame(buffer);
  if (tail) yield tail;
}

function parseSseFrame(frame: string): SseEvent | undefined {
  const lines = frame.split('\n');
  let event: string | undefined;
  let id: string | undefined;
  const dataLines: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (line.length === 0 || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1);
    const trimmed = value.startsWith(' ') ? value.slice(1) : value;

    switch (field) {
      case 'event':
        event = trimmed;
        break;
      case 'id':
        id = trimmed;
        break;
      case 'data':
        dataLines.push(trimmed);
        break;
      default:
        break;
    }
  }

  if (dataLines.length === 0 && event === undefined) return undefined;
  return { event, data: dataLines.join('\n'), id };
}

/** True when a request should be retried (network-ish status codes). */
export { isRetryableStatus };
