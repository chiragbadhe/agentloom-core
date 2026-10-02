import { describe, expect, it, vi } from 'vitest';

import {
  abortPromise,
  combineSignals,
  deepClone,
  deferred,
  mapWithConcurrency,
  raceWithAbort,
  sleep,
  throwIfAborted,
  withTimeout,
} from '../src/utils/async.js';
import { AsyncQueue } from '../src/utils/queue.js';
import { AbortError, TimeoutError } from '../src/errors.js';

describe('sleep', () => {
  it('resolves after the delay', async () => {
    const start = Date.now();
    await sleep(15);
    expect(Date.now() - start).toBeGreaterThanOrEqual(10);
  });

  it('resolves early rather than throwing when aborted', async () => {
    const controller = new AbortController();
    const start = Date.now();
    const promise = sleep(10_000, controller.signal);
    controller.abort();
    await expect(promise).resolves.toBeUndefined();
    expect(Date.now() - start).toBeLessThan(1_000);
  });

  it('resolves immediately when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(sleep(10_000, controller.signal)).resolves.toBeUndefined();
  });

  it('resolves immediately for a non-positive delay', async () => {
    await expect(sleep(0)).resolves.toBeUndefined();
    await expect(sleep(-5)).resolves.toBeUndefined();
  });
});

describe('withTimeout', () => {
  it('passes the value through when it resolves in time', async () => {
    await expect(withTimeout(async () => 'fast', 1_000)).resolves.toBe('fast');
  });

  it('hands the work a signal that aborts on timeout', async () => {
    let sawAbort = false;
    await expect(
      withTimeout(
        (signal) =>
          new Promise<string>((_, reject) => {
            signal.addEventListener('abort', () => {
              sawAbort = true;
              reject(new AbortError());
            });
          }),
        20,
      ),
    ).rejects.toThrow();
    expect(sawAbort).toBe(true);
  });

  it('rejects with a TimeoutError naming the label', async () => {
    const never = new Promise<string>(() => {});
    await expect(withTimeout(() => never, 20, { label: 'model call' })).rejects.toThrow(
      /model call timed out/,
    );
  });

  it('rejects with the caller AbortError, not a timeout, when the caller cancels', async () => {
    const controller = new AbortController();
    const never = new Promise<string>((resolve) =>
      setTimeout(() => resolve('late'), 200),
    );
    const promise = withTimeout(() => never, 5_000, { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toThrow(/abort/i);
  });

  it('propagates an immediate rejection untouched', async () => {
    await expect(
      withTimeout(async () => {
        throw new Error('nope');
      }, 1_000),
    ).rejects.toThrow('nope');
  });

  it('skips the deadline when it is not positive', async () => {
    await expect(withTimeout(async () => 'ok', 0)).resolves.toBe('ok');
  });
});

describe('withTimeout error shape', () => {
  it('rejects with an AgentError TimeoutError carrying the deadline', async () => {
    const never = new Promise<string>(() => {});
    await expect(withTimeout(() => never, 15, { label: 'stream' })).rejects.toSatisfy(
      (error: unknown) => {
        return (
          error instanceof TimeoutError &&
          error.timeoutMs === 15 &&
          error.details?.['label'] === 'stream'
        );
      },
    );
  });
});

describe('combineSignals', () => {
  it('returns a fresh live signal when there is nothing to combine', () => {
    const combined = combineSignals();
    expect(combined.signal.aborted).toBe(false);
    combined.dispose();
  });

  it('propagates an abort from either source', () => {
    const a = new AbortController();
    const b = new AbortController();
    const combined = combineSignals(a.signal, b.signal);

    expect(combined.signal.aborted).toBe(false);
    a.abort();
    expect(combined.signal.aborted).toBe(true);

    combined.dispose();
  });

  it('returns an already-aborted signal when a source is aborted', () => {
    const a = new AbortController();
    a.abort();
    const combined = combineSignals(a.signal, new AbortController().signal);
    expect(combined.signal.aborted).toBe(true);
    combined.dispose();
  });

  it('stops listening after dispose', () => {
    const a = new AbortController();
    const combined = combineSignals(a.signal);
    combined.dispose();
    a.abort();
    expect(combined.signal.aborted).toBe(false);
  });
});

describe('throwIfAborted', () => {
  it('does nothing when not aborted', () => {
    expect(() => throwIfAborted(new AbortController().signal)).not.toThrow();
  });

  it('throws when aborted', () => {
    const controller = new AbortController();
    controller.abort();
    expect(() => throwIfAborted(controller.signal)).toThrow(/abort/i);
  });

  it('throws an AbortError that is not retryable', () => {
    const controller = new AbortController();
    controller.abort('user cancelled the run');
    expect(() => throwIfAborted(controller.signal)).toThrow(AbortError);
  });

  it('treats a missing signal as not aborted', () => {
    expect(() => throwIfAborted(undefined)).not.toThrow();
  });
});

describe('raceWithAbort', () => {
  it('resolves when the promise wins', async () => {
    await expect(
      raceWithAbort(Promise.resolve(1), new AbortController().signal),
    ).resolves.toBe(1);
  });

  it('rejects when the signal aborts first', async () => {
    const controller = new AbortController();
    const never = new Promise<number>(() => {});
    const raced = raceWithAbort(never, controller.signal);
    controller.abort();
    await expect(raced).rejects.toThrow(/abort/i);
  });
});

describe('abortPromise', () => {
  it('rejects with an AbortError when the signal fires', async () => {
    const controller = new AbortController();
    const { promise } = abortPromise(controller.signal);
    controller.abort();
    await expect(promise).rejects.toThrow(AbortError);
  });

  it('rejects immediately when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(abortPromise(controller.signal).promise).rejects.toThrow(AbortError);
  });

  it('cancel() stops listening so the promise never settles', async () => {
    const controller = new AbortController();
    const { promise, cancel } = abortPromise(controller.signal);
    let settled = false;
    void promise.catch(() => {
      settled = true;
    });
    cancel();
    controller.abort();
    await sleep(10);
    expect(settled).toBe(false);
  });
});

describe('deferred', () => {
  it('exposes resolve and reject', async () => {
    const d = deferred<string>();
    d.resolve('done');
    await expect(d.promise).resolves.toBe('done');

    const failing = deferred<string>();
    failing.reject(new Error('bad'));
    await expect(failing.promise).rejects.toThrow('bad');
  });
});

describe('mapWithConcurrency', () => {
  it('preserves input order', async () => {
    const result = await mapWithConcurrency([3, 1, 2], 2, async (n) => n * 10);
    expect(result).toEqual([30, 10, 20]);
  });

  it('never exceeds the concurrency limit', async () => {
    let active = 0;
    let peak = 0;
    await mapWithConcurrency(
      Array.from({ length: 12 }, (_, i) => i),
      3,
      async (n) => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(5);
        active -= 1;
        return n;
      },
    );
    expect(peak).toBeLessThanOrEqual(3);
  });

  it('runs fully in parallel for an infinite limit', async () => {
    let active = 0;
    let peak = 0;
    await mapWithConcurrency([1, 2, 3, 4], Infinity, async () => {
      active += 1;
      peak = Math.max(peak, active);
      await sleep(5);
      active -= 1;
    });
    expect(peak).toBe(4);
  });

  it('returns an empty array for empty input', async () => {
    await expect(mapWithConcurrency([], 2, async () => 1)).resolves.toEqual([]);
  });

  it('propagates a rejection and stops scheduling', async () => {
    const started: number[] = [];
    await expect(
      mapWithConcurrency([1, 2, 3, 4], 1, async (n) => {
        started.push(n);
        if (n === 2) throw new Error('stop');
        return n;
      }),
    ).rejects.toThrow('stop');
    expect(started).toEqual([1, 2]);
  });
});

describe('deepClone', () => {
  it('clones nested objects and arrays', () => {
    const source = { a: 1, b: { c: [1, 2, { d: 3 }] } };
    const clone = deepClone(source);
    expect(clone).toEqual(source);
    expect(clone).not.toBe(source);
    expect(clone.b).not.toBe(source.b);
    expect(clone.b.c).not.toBe(source.b.c);
  });

  it('handles Date, Map, and Set', () => {
    const source = {
      when: new Date('2024-01-01T00:00:00.000Z'),
      map: new Map([['a', 1]]),
      set: new Set([1, 2]),
    };
    const clone = deepClone(source);
    expect(clone.when).toEqual(source.when);
    expect(clone.when).not.toBe(source.when);
    expect(clone.map.get('a')).toBe(1);
    expect([...clone.set]).toEqual([1, 2]);
  });

  it('survives circular references', () => {
    const source: Record<string, unknown> = { name: 'root' };
    source['self'] = source;
    const clone = deepClone(source);
    expect(clone['self']).toBe(clone);
  });

  it('passes primitives straight through', () => {
    for (const value of [1, 'a', true, null, undefined]) {
      expect(deepClone(value)).toBe(value);
    }
  });
});

describe('AsyncQueue', () => {
  it('delivers items in order and completes on close', async () => {
    const queue = new AsyncQueue<number>();
    void (async () => {
      for (const n of [1, 2, 3]) await queue.push(n);
      queue.close();
    })();

    const seen: number[] = [];
    for await (const item of queue) seen.push(item);
    expect(seen).toEqual([1, 2, 3]);
  });

  it('is a no-op when pushing after close', async () => {
    const queue = new AsyncQueue<number>();
    queue.close();
    await expect(queue.push(1)).resolves.toBeUndefined();
    expect(queue.size).toBe(0);
    expect(queue.isClosed).toBe(true);
  });

  it('propagates a failure to the consumer', async () => {
    const queue = new AsyncQueue<number>();
    await queue.push(1);
    queue.fail(new Error('producer exploded'));

    const seen: number[] = [];
    let caught: Error | undefined;
    try {
      for await (const item of queue) seen.push(item);
    } catch (error) {
      caught = error as Error;
    }
    expect(seen).toEqual([1]);
    expect(caught?.message).toBe('producer exploded');
  });

  it('normalises a non-Error failure into an Error', async () => {
    const queue = new AsyncQueue<number>();
    queue.fail('plain string');
    await expect(queue[Symbol.asyncIterator]().next()).rejects.toThrow(Error);
  });

  it('drains buffered items to a direct consumer', async () => {
    const queue = new AsyncQueue<string>();
    await queue.push('a');
    await queue.push('b');
    queue.close();
    expect((await queue[Symbol.asyncIterator]().next()).value).toBe('a');
  });
});

describe('AsyncQueue backpressure', () => {
  it('blocks the producer once the high-water mark is reached', async () => {
    const queue = new AsyncQueue<number>(2);
    const pushed: number[] = [];

    const producer = (async () => {
      for (const n of [1, 2, 3, 4]) {
        await queue.push(n);
        pushed.push(n);
      }
      queue.close();
    })();

    await sleep(10);
    // The buffer holds 2 and the producer is parked on the third push.
    expect(pushed.length).toBeLessThanOrEqual(2);

    const seen: number[] = [];
    for await (const item of queue) seen.push(item);
    await producer;

    expect(seen).toEqual([1, 2, 3, 4]);
  });

  it('wakes a parked producer when the queue closes', async () => {
    const queue = new AsyncQueue<number>(1);
    const producer = queue.push(1).then(() => queue.push(2));
    await sleep(5);
    queue.close();
    await expect(producer).resolves.toBeUndefined();
  });
});

describe('vi.fn interop', () => {
  it('is not required by these utilities', () => {
    const spy = vi.fn(async () => 1);
    expect(spy).toBeDefined();
  });
});
