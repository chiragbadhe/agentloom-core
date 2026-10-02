import type { FetchLike, FetchResponseLike } from '../../src/providers/http.js';

export interface StubResponseInit {
  readonly status?: number;
  readonly headers?: Record<string, string>;
  /** Body text. Mutually exclusive with `chunks`. */
  readonly body?: string;
  /** Emit the body as discrete chunks, exercising streaming paths. */
  readonly chunks?: readonly string[];
}

export interface StubFetch {
  (input: string, init: Parameters<FetchLike>[1]): Promise<FetchResponseLike>;
  readonly calls: { url: string; init: Parameters<FetchLike>[1] }[];
  readonly bodies: () => unknown[];
}

/**
 * A `fetch` stub that records every call and replies with canned responses.
 *
 * Pass an array to script a sequence (one entry per call); the last entry
 * repeats once the script runs out, which keeps retry tests terse.
 */
export function stubFetch(
  responses:
    StubResponseInit | readonly StubResponseInit[] | ((url: string) => StubResponseInit),
): StubFetch {
  const script = Array.isArray(responses)
    ? [...(responses as readonly StubResponseInit[])]
    : undefined;
  const single = script
    ? undefined
    : (responses as StubResponseInit | ((url: string) => StubResponseInit));
  const calls: { url: string; init: Parameters<FetchLike>[1] }[] = [];

  const impl = async (
    url: string,
    init: Parameters<FetchLike>[1],
  ): Promise<FetchResponseLike> => {
    calls.push({ url, init });
    const index = Math.min(calls.length - 1, script ? script.length - 1 : 0);
    const entry =
      script !== undefined
        ? script[index]
        : typeof single === 'function'
          ? single(url)
          : (single as StubResponseInit);
    return toResponse(entry ?? {});
  };

  return Object.assign(impl, {
    calls,
    bodies: () => calls.map((call) => JSON.parse(call.init.body ?? 'null')),
  });
}

function toResponse(init: StubResponseInit): FetchResponseLike {
  const status = init.status ?? 200;
  const headers = new Map(
    Object.entries(init.headers ?? { 'content-type': 'application/json' }).map(
      ([k, v]) => [k.toLowerCase(), v],
    ),
  );
  const body =
    init.chunks === undefined
      ? undefined
      : asyncIterableToStream(
          init.chunks.map((chunk) => new TextEncoder().encode(chunk)),
        );

  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { forEach: (cb) => headers.forEach((value, key) => cb(value, key)) },
    body,
    text: async () => init.body ?? (init.chunks ?? []).join(''),
  };
}

/** Wrap an array of chunks in a minimal `ReadableStream`. */
export function asyncIterableToStream(
  chunks: readonly Uint8Array[],
): ReadableStream<Uint8Array> {
  let index = 0;
  return {
    getReader(): {
      read(): Promise<{ done?: boolean; value?: Uint8Array }>;
      releaseLock(): void;
    } {
      return {
        async read() {
          const value = chunks[index++];
          return value === undefined ? { done: true } : { done: false, value };
        },
        releaseLock() {
          // no-op
        },
      };
    },
  } as unknown as ReadableStream<Uint8Array>;
}

/** An SSE response body from raw text lines. */
export function sseChunks(...lines: string[]): readonly string[] {
  return lines;
}
