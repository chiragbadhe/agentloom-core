import { AbortError, TimeoutError } from '../errors.js';

/** A cancellable sleep. Resolves early (without throwing) when aborted. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise<void>((resolve) => {
    if (signal?.aborted) return resolve();
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      resolve();
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Resolve/reject when an AbortSignal fires. Used to race against long work.
 * Never rejects on its own — only settles when the signal aborts.
 */
export function abortPromise(signal: AbortSignal): {
  promise: Promise<never>;
  cancel: () => void;
} {
  let onAbort: (() => void) | undefined;
  const promise = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(new AbortError());
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  });
  return {
    promise,
    cancel: () => {
      if (onAbort) signal.removeEventListener('abort', onAbort);
    },
  };
}

/**
 * Combine multiple signals into one. Returns the composite plus a disposer.
 * Node 20+ has `AbortSignal.any`; this is the portable equivalent.
 */
export function combineSignals(...signals: (AbortSignal | undefined)[]): {
  signal: AbortSignal;
  dispose: () => void;
} {
  const controller = new AbortController();
  const present = signals.filter((s): s is AbortSignal => s !== undefined);

  if (present.length === 0) {
    return { signal: controller.signal, dispose: () => undefined };
  }

  const onAbort = (event: Event) => {
    const source = event.target as AbortSignal;
    controller.abort(source.reason);
  };

  const cleanup = () => {
    for (const signal of present) signal.removeEventListener('abort', onAbort);
  };

  for (const signal of present) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener('abort', onAbort, { once: true });
  }

  return { signal: controller.signal, dispose: cleanup };
}

/** Distinguishes a deadline firing from a caller cancellation. */
const TIMED_OUT = Symbol('agentloom.timeout');
const CANCELLED = Symbol('agentloom.cancelled');

/** Resolve `aborting.promise` to a sentinel instead of rejecting. */
function sentinel(
  aborting: { promise: Promise<never>; cancel: () => void },
  value: symbol,
): { settled: Promise<symbol>; cancel: () => void } {
  return {
    settled: aborting.promise.then(
      () => value,
      () => value,
    ),
    cancel: aborting.cancel,
  };
}

/**
 * Run `fn` with a deadline. The work is *not* cancelled on timeout — it is
 * abandoned. Pass the provided `signal` down to anything cancellable so the
 * underlying operation stops too.
 *
 * @throws {TimeoutError} when the deadline elapses first.
 * @throws {AbortError} when the caller's signal fires first.
 */
export async function withTimeout<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
  options: { signal?: AbortSignal; label?: string } = {},
): Promise<T> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return fn(options.signal ?? new AbortController().signal);
  }

  const timeoutController = new AbortController();
  const timer = setTimeout(() => timeoutController.abort(), timeoutMs);

  const { signal, dispose } = combineSignals(timeoutController.signal, options.signal);

  // Sentinels let us tell *which* deadline fired without string-matching error
  // messages, which would misreport a caller's cancellation as a timeout.
  const deadline = sentinel(abortPromise(timeoutController.signal), TIMED_OUT);
  const caller = options.signal
    ? sentinel(abortPromise(options.signal), CANCELLED)
    : undefined;

  const work = fn(signal);
  // The work is abandoned on timeout; swallow its eventual rejection so it never
  // surfaces as an unhandled rejection after this call has already returned.
  void work.catch(() => undefined);

  try {
    const raced = await (caller === undefined
      ? Promise.race([work, deadline.settled])
      : Promise.race([work, deadline.settled, caller.settled]));

    if (raced === TIMED_OUT) throw new TimeoutError(timeoutMs, options.label);
    if (raced === CANCELLED) throw new AbortError();
    return raced as T;
  } finally {
    clearTimeout(timer);
    deadline.cancel();
    caller?.cancel();
    dispose();
  }
}

/**
 * Ensure the signal has not already fired. Call at the top of async work so
 * callers observe abort errors as early as possible.
 */
export function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new AbortError();
}

/** Reject if the signal fires before `promise` settles. */
export function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  const { promise: aborting, cancel } = abortPromise(signal);
  return Promise.race([promise, aborting]).finally(cancel);
}

/**
 * Iterate an async iterable, aborting promptly when `signal` fires.
 *
 * Each `next()` is raced against the signal, so a producer that ignores
 * cancellation (a wedged socket, a provider that forgot to pass the signal on)
 * still unblocks the consumer. Cleanup of the underlying iterator is
 * best-effort and never awaited — awaiting it would re-block on the very
 * producer we are escaping.
 */
export function raceIterableWithAbort<T>(
  iterable: AsyncIterable<T>,
  signal?: AbortSignal,
): AsyncIterable<T> {
  if (signal === undefined) return iterable;
  return {
    async *[Symbol.asyncIterator](): AsyncGenerator<T> {
      const iterator = iterable[Symbol.asyncIterator]();
      try {
        for (;;) {
          const step = await raceWithAbort(iterator.next(), signal);
          if (step.done === true) return;
          yield step.value;
        }
      } finally {
        void iterator.return?.(undefined);
      }
    },
  };
}

/** A promise plus its `resolve`/`reject` handles. */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Bounded-concurrency map that preserves input order. Used for parallel tool
 * execution so a burst of calls cannot exhaust sockets or memory.
 */
export async function mapWithConcurrency<TIn, TOut>(
  items: readonly TIn[],
  limit: number,
  fn: (item: TIn, index: number) => Promise<TOut>,
): Promise<TOut[]> {
  const size = Math.max(1, Math.min(limit, items.length));
  const results = new Array<TOut>(items.length);
  let cursor = 0;

  const workers = Array.from({ length: size }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      const item = items[index];
      if (item === undefined) continue;
      results[index] = await fn(item, index);
    }
  });

  await Promise.all(workers);
  return results;
}

/** Structured-clone-ish deep copy that tolerates functions and cycles. */
export function deepClone<T>(value: T): T {
  return clone(value, new WeakMap()) as T;
}

function clone(value: unknown, seen: WeakMap<object, unknown>): unknown {
  if (typeof value !== 'object' || value === null) return value;
  const asObject = value;
  const existing = seen.get(asObject);
  if (existing !== undefined) return existing;

  if (Array.isArray(value)) {
    const copy: unknown[] = [];
    seen.set(asObject, copy);
    for (const item of value) copy.push(clone(item, seen));
    return copy;
  }

  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof Map) {
    const copy = new Map();
    seen.set(asObject, copy);
    for (const [k, v] of value) copy.set(clone(k, seen), clone(v, seen));
    return copy;
  }
  if (value instanceof Set) {
    const copy = new Set();
    seen.set(asObject, copy);
    for (const v of value) copy.add(clone(v, seen));
    return copy;
  }

  const proto: unknown = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return value;

  const copy: Record<string, unknown> = {};
  seen.set(asObject, copy);
  for (const [key, item] of Object.entries(value)) copy[key] = clone(item, seen);
  return copy;
}
