import { describe, expect, it, vi } from 'vitest';

import { ProviderError } from '../src/errors.js';
import {
  computeBackoffDelay,
  resolveRetryPolicy,
  withRetry,
  type ResolvedRetryPolicy,
} from '../src/utils/retry.js';

describe('resolveRetryPolicy', () => {
  it('fills in defaults', () => {
    const policy = resolveRetryPolicy({});
    expect(policy.maxAttempts).toBe(3);
    expect(policy.initialDelayMs).toBe(500);
    expect(policy.maxDelayMs).toBe(30_000);
    expect(policy.backoffFactor).toBe(2);
    expect(policy.jitter).toBe(0.2);
    expect(policy.respectRetryAfter).toBe(true);
  });

  it('normalizes a sub-1 attempt count up to 1', () => {
    expect(resolveRetryPolicy({ maxAttempts: 0 }).maxAttempts).toBe(1);
    expect(resolveRetryPolicy({ maxAttempts: 1 }).maxAttempts).toBe(1);
  });

  it('ignores undefined overrides', () => {
    expect(resolveRetryPolicy({ initialDelayMs: undefined }).initialDelayMs).toBe(500);
  });

  it('defaults shouldRetry to the error flag', () => {
    expect(
      resolveRetryPolicy({}).shouldRetry(new ProviderError('x', { retryable: true }), 1),
    ).toBe(true);
  });
});

describe('computeBackoffDelay', () => {
  const policy = (overrides: Partial<ResolvedRetryPolicy> = {}) =>
    resolveRetryPolicy({ initialDelayMs: 100, jitter: 0, ...overrides });

  it('grows by the backoff factor', () => {
    const p = policy();
    expect(computeBackoffDelay(1, p)).toBe(100);
    expect(computeBackoffDelay(2, p)).toBe(200);
    expect(computeBackoffDelay(3, p)).toBe(400);
  });

  it('honours a custom factor', () => {
    const p = policy({ backoffFactor: 3 });
    expect(computeBackoffDelay(3, p)).toBe(900);
  });

  it('never exceeds maxDelayMs', () => {
    const p = policy({ maxDelayMs: 250 });
    expect(computeBackoffDelay(20, p)).toBe(250);
  });

  it('prefers a server-supplied retryAfterMs', () => {
    expect(computeBackoffDelay(1, policy(), 1_234)).toBe(1_234);
  });

  it('caps retryAfterMs at maxDelayMs', () => {
    expect(computeBackoffDelay(1, policy({ maxDelayMs: 500 }), 10_000)).toBe(500);
  });

  it('adds jitter within ±jitter of the base', () => {
    const p = policy({ initialDelayMs: 1_000, jitter: 0.5, maxDelayMs: 10_000 });
    const samples = Array.from({ length: 40 }, () => computeBackoffDelay(1, p));
    for (const sample of samples) {
      expect(sample).toBeGreaterThanOrEqual(500);
      expect(sample).toBeLessThanOrEqual(1_500);
    }
    expect(new Set(samples).size).toBeGreaterThan(1);
  });

  it('is deterministic without jitter', () => {
    const p = policy();
    expect(computeBackoffDelay(2, p)).toBe(computeBackoffDelay(2, p));
  });
});

describe('withRetry', () => {
  const ok =
    <T>(value: T): (() => Promise<T>) =>
    async () =>
      value;

  it('returns the value without retrying when the first attempt succeeds', async () => {
    const fn = vi.fn(ok('done'));
    await expect(withRetry(fn, { maxAttempts: 3, initialDelayMs: 1 })).resolves.toBe(
      'done',
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('retries a retryable failure and then succeeds', async () => {
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ProviderError('flaky', { retryable: true }))
      .mockRejectedValueOnce(new ProviderError('flaky', { retryable: true }))
      .mockResolvedValue('recovered');

    await expect(
      withRetry(fn, { maxAttempts: 3, initialDelayMs: 1, jitter: 0 }),
    ).resolves.toBe('recovered');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('gives up after the attempt budget and rethrows the last error', async () => {
    const failure = new ProviderError('always down', { retryable: true });
    const fn = vi.fn<() => Promise<string>>().mockRejectedValue(failure);

    await expect(
      withRetry(fn, { maxAttempts: 3, initialDelayMs: 1, jitter: 0 }),
    ).rejects.toThrow('always down');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('does not retry a non-retryable failure', async () => {
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValue(new ProviderError('bad request', { retryable: false }));

    await expect(withRetry(fn, { maxAttempts: 5, initialDelayMs: 1 })).rejects.toThrow(
      'bad request',
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('reports each retry through onRetry', async () => {
    const seen: number[] = [];
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ProviderError('x', { retryable: true }))
      .mockResolvedValue('ok');

    await withRetry(fn, {
      maxAttempts: 3,
      initialDelayMs: 1,
      jitter: 0,
      onRetry: (context) => {
        seen.push(context.attempt);
      },
    });

    expect(seen).toEqual([1]);
  });

  it('passes the attempt number and remaining budget to onRetry', async () => {
    let observed: { attempt: number; remaining: number } | undefined;
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValue(new ProviderError('x', { retryable: true }));

    await withRetry(fn, {
      maxAttempts: 3,
      initialDelayMs: 1,
      jitter: 0,
      onRetry: (context) => {
        observed ??= { attempt: context.attempt, remaining: context.remaining };
      },
    }).catch(() => undefined);

    expect(observed).toEqual({ attempt: 1, remaining: 2 });
  });

  it('stops immediately when the signal aborts between attempts', async () => {
    const controller = new AbortController();
    const fn = vi.fn<() => Promise<string>>().mockImplementation(async () => {
      controller.abort();
      throw new ProviderError('x', { retryable: true });
    });

    await expect(
      withRetry(fn, {
        maxAttempts: 5,
        initialDelayMs: 5_000,
        jitter: 0,
        signal: controller.signal,
      }),
    ).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('honours retryAfterMs from the error instead of the computed backoff', async () => {
    const start = Date.now();
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(
        new ProviderError('rate limited', { retryable: true, retryAfterMs: 40 }),
      )
      .mockResolvedValue('ok');

    await withRetry(fn, { maxAttempts: 2, initialDelayMs: 10, jitter: 0 });
    expect(Date.now() - start).toBeGreaterThanOrEqual(35);
  });

  it('calls an onRetry hook even when a delay is used', async () => {
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ProviderError('x', { retryable: true }))
      .mockResolvedValue('ok');
    const onRetry = vi.fn();

    await withRetry(fn, { maxAttempts: 2, initialDelayMs: 1, jitter: 0, onRetry });
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('withRetry with attempts 1 behaves like a direct call', async () => {
    const fn = vi.fn<() => Promise<string>>().mockResolvedValue('once');
    await expect(withRetry(fn, { maxAttempts: 1 })).resolves.toBe('once');
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
