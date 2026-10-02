import { describe, expect, it } from 'vitest';

import {
  AbortError,
  ProviderError,
  ProviderResponseError,
  ProviderTimeoutError,
} from '../../src/errors.js';
import {
  HttpClient,
  parseRetryAfter,
  parseSseStream,
  type FetchLike,
} from '../../src/providers/http.js';
import { stubFetch } from '../helpers/mock-fetch.js';

describe('HttpClient', () => {
  it('serialises the body and sets a JSON content type', async () => {
    const fetchImpl = stubFetch({ body: '{"ok":true}' });
    const client = new HttpClient({ fetch: fetchImpl });

    await client.json({ url: 'https://api.test/v1', body: { a: 1 }, providerId: 'test' });

    expect(fetchImpl.calls[0]?.url).toBe('https://api.test/v1');
    expect(fetchImpl.calls[0]?.init.method).toBe('POST');
    expect(fetchImpl.calls[0]?.init.body).toBe('{"a":1}');
    expect(fetchImpl.calls[0]?.init.headers['content-type']).toBe('application/json');
  });

  it('merges default and per-request headers, lowercasing keys', async () => {
    const fetchImpl = stubFetch({ body: '{}' });
    const client = new HttpClient({
      fetch: fetchImpl,
      defaultHeaders: { 'X-Default': 'yes' },
    });

    await client.json({
      url: 'https://api.test/v1',
      providerId: 'test',
      headers: { 'X-Request': 'also' },
    });

    const headers = fetchImpl.calls[0]?.init.headers ?? {};
    expect(headers['x-default']).toBe('yes');
    expect(headers['x-request']).toBe('also');
    expect(headers['accept']).toBe('application/json');
  });

  it('parses the JSON body', async () => {
    const client = new HttpClient({ fetch: stubFetch({ body: '{"value":42}' }) });
    await expect(
      client.json<{ value: number }>({ url: 'https://api.test', providerId: 'test' }),
    ).resolves.toEqual({ value: 42 });
  });

  it('treats an empty body as undefined', async () => {
    const client = new HttpClient({ fetch: stubFetch({ body: '   ' }) });
    await expect(
      client.json({ url: 'https://api.test', providerId: 'test' }),
    ).resolves.toBeUndefined();
  });

  it('rejects malformed JSON with a ProviderResponseError', async () => {
    const client = new HttpClient({ fetch: stubFetch({ body: 'not json' }) });
    await expect(
      client.json({
        url: 'https://api.test',
        providerId: 'test',
        context: { model: 'm' },
      }),
    ).rejects.toBeInstanceOf(ProviderResponseError);
  });

  it('maps non-2xx statuses to typed errors', async () => {
    const client = new HttpClient({
      fetch: stubFetch({ status: 429, body: '{"error":"slow down"}' }),
    });
    const error = await client
      .json({ url: 'https://api.test', providerId: 'openai', context: { model: 'gpt' } })
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ProviderError);
    const providerError = error as ProviderError;
    expect(providerError.providerId).toBe('openai');
    expect(providerError.retryable).toBe(true);
    expect(providerError.statusCode).toBe(429);
    expect(providerError.message).toContain('slow down');
  });

  it('reports 4xx as non-retryable', async () => {
    const client = new HttpClient({
      fetch: stubFetch({ status: 400, body: '{"error":"bad"}' }),
    });
    const error = (await client
      .json({ url: 'https://api.test', providerId: 'openai' })
      .catch((e: unknown) => e)) as ProviderError;
    expect(error.retryable).toBe(false);
    expect(error.statusCode).toBe(400);
  });

  it('resolves without throwing on non-2xx via request()', async () => {
    const client = new HttpClient({ fetch: stubFetch({ status: 503, body: '' }) });
    const response = await client.request({
      url: 'https://api.test',
      providerId: 'test',
    });
    expect(response.status).toBe(503);
    expect(response.ok).toBe(false);
  });

  it('times out into a ProviderTimeoutError', async () => {
    const hanging: FetchLike = () =>
      new Promise((_resolve, reject) => {
        setTimeout(() => reject(new Error('aborted')), 50);
      });

    const client = new HttpClient({ fetch: hanging, defaultTimeoutMs: 10 });
    await expect(
      client.json({ url: 'https://api.test', providerId: 'test' }),
    ).rejects.toBeInstanceOf(ProviderTimeoutError);
  });

  it('converts an external abort into an AbortError', async () => {
    const controller = new AbortController();
    const hanging: FetchLike = (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => reject(new Error('aborted')));
        controller.abort();
      });

    const client = new HttpClient({ fetch: hanging });
    await expect(
      client.json({
        url: 'https://api.test',
        providerId: 'test',
        signal: controller.signal,
      }),
    ).rejects.toBeInstanceOf(AbortError);
  });

  it('wraps transport failures in a ProviderError', async () => {
    const failing: FetchLike = () => Promise.reject(new Error('ECONNREFUSED'));

    const client = new HttpClient({ fetch: failing });
    const error = (await client
      .json({ url: 'https://api.test', providerId: 'test' })
      .catch((e: unknown) => e)) as ProviderError;
    expect(error).toBeInstanceOf(ProviderError);
    expect(error.message).toContain('ECONNREFUSED');
  });

  it('exposes retry-after as milliseconds', async () => {
    const client = new HttpClient({
      fetch: stubFetch({ status: 429, headers: { 'retry-after': '2' }, body: '' }),
    });
    const response = await client.request({
      url: 'https://api.test',
      providerId: 'test',
    });
    expect(response.retryAfterMs).toBe(2000);
  });

  it('buffers a streamed body into text', async () => {
    const client = new HttpClient({
      fetch: stubFetch({ chunks: ['{"va', 'lue":', '7}'] }),
    });
    await expect(
      client.json<{ value: number }>({ url: 'https://api.test', providerId: 'test' }),
    ).resolves.toEqual({ value: 7 });
  });

  it('exposes the raw stream and an async-iterable body', async () => {
    const client = new HttpClient({ fetch: stubFetch({ chunks: ['a', 'b'] }) });
    const response = await client.request({
      url: 'https://api.test',
      providerId: 'test',
    });
    const stream = response.stream();
    expect(stream).not.toBeNull();

    const decoder = new TextDecoder();
    let out = '';
    for await (const chunk of stream ?? []) out += decoder.decode(chunk);
    expect(out).toBe('ab');
  });

  it('reports a null stream when the response has no body', async () => {
    const client = new HttpClient({ fetch: stubFetch({ body: 'x' }) });
    const response = await client.request({
      url: 'https://api.test',
      providerId: 'test',
    });
    expect(response.stream()).toBeNull();
  });

  it('errors when global fetch is missing', async () => {
    const original = globalThis.fetch;
    // @ts-expect-error deliberately removing the global
    delete globalThis.fetch;
    try {
      const client = new HttpClient();
      await expect(
        client.json({ url: 'https://api.test', providerId: 'test' }),
      ).rejects.toThrow(/fetch is not available/);
    } finally {
      globalThis.fetch = original;
    }
  });
});

describe('parseRetryAfter', () => {
  it('parses seconds', () => {
    expect(parseRetryAfter('30')).toBe(30_000);
    expect(parseRetryAfter('0')).toBe(0);
  });

  it('parses HTTP dates', () => {
    const now = Date.parse('2024-01-01T00:00:00Z');
    expect(parseRetryAfter('Mon, 01 Jan 2024 00:00:10 GMT', now)).toBe(10_000);
  });

  it('clamps past dates and ignores garbage', () => {
    const now = Date.parse('2024-01-01T00:00:00Z');
    expect(parseRetryAfter('Mon, 01 Jan 2020 00:00:00 GMT', now)).toBe(0);
    expect(parseRetryAfter('soon', now)).toBeUndefined();
    expect(parseRetryAfter(undefined)).toBeUndefined();
  });
});

describe('parseSseStream', () => {
  async function collect(chunks: readonly string[]): Promise<unknown[]> {
    const encoder = new TextEncoder();
    const iterable = {
      async *[Symbol.asyncIterator]() {
        for (const chunk of chunks) yield encoder.encode(chunk);
      },
    };
    const events = [];
    for await (const event of parseSseStream(iterable)) events.push(event);
    return events;
  }

  it('parses data frames', async () => {
    const events = await collect(['data: {"a":1}\n\ndata: {"a":2}\n\n']);
    expect(events).toEqual([
      { event: undefined, data: '{"a":1}', id: undefined },
      { event: undefined, data: '{"a":2}', id: undefined },
    ]);
  });

  it('reads event names and ids', async () => {
    const events = await collect(['event: ping\nid: 7\ndata: hi\n\n']);
    expect(events[0]).toEqual({ event: 'ping', data: 'hi', id: '7' });
  });

  it('joins multi-line data and ignores comments', async () => {
    const events = await collect([': keepalive\ndata: one\ndata: two\n\n']);
    expect(events).toEqual([{ event: undefined, data: 'one\ntwo', id: undefined }]);
  });

  it('handles frames split across chunk boundaries', async () => {
    const events = await collect(['data: {"a"', ':1}\n', '\ndata: [DONE]\n\n']);
    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({ data: '{"a":1}' });
    expect(events[1]).toMatchObject({ data: '[DONE]' });
  });

  it('emits a trailing frame without a blank line', async () => {
    const events = await collect(['data: tail']);
    expect(events).toEqual([{ event: undefined, data: 'tail', id: undefined }]);
  });

  it('drops blank frames', async () => {
    const events = await collect(['\n\n\ndata: x\n\n']);
    expect(events).toHaveLength(1);
  });
});
