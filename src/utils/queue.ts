/**
 * A single-consumer async queue bridging a producer (the agent loop) and a
 * consumer (a `for await` loop).
 *
 * Backpressure is applied: once `highWaterMark` items are buffered the
 * producer's `push` resolves only after the consumer drains one, which stops a
 * fast model from buffering an entire response in memory.
 */
export class AsyncQueue<T> implements AsyncIterable<T> {
  private readonly buffered: T[] = [];
  private readonly waiters: {
    resolve: (result: IteratorResult<T>) => void;
    reject: (error: unknown) => void;
  }[] = [];
  private readonly drainWaiters: (() => void)[] = [];
  private failure: Error | undefined;
  private closed = false;

  constructor(private readonly highWaterMark = 64) {}

  get size(): number {
    return this.buffered.length;
  }

  get isClosed(): boolean {
    return this.closed;
  }

  /** Append an item. Resolves once there is room in the buffer. */
  async push(value: T): Promise<void> {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter !== undefined) {
      waiter.resolve({ value, done: false });
      return;
    }
    this.buffered.push(value);
    while (this.buffered.length > this.highWaterMark && !this.closed) {
      await new Promise<void>((resolve) => this.drainWaiters.push(resolve));
    }
  }

  /** Close the queue after the buffered items are consumed. */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    while (this.waiters.length > 0) {
      this.waiters.shift()?.resolve({ value: undefined as never, done: true });
    }
    while (this.drainWaiters.length > 0) this.drainWaiters.shift()?.();
  }

  /** Terminate the queue with an error. */
  fail(error: unknown): void {
    if (this.closed) return;
    this.failure = toError(error);
    this.close();
    while (this.waiters.length > 0) this.waiters.shift()?.reject(error);
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T> {
    for (;;) {
      const item = this.buffered.shift();
      if (item !== undefined) {
        this.drainWaiters.shift()?.();
        yield item;
        continue;
      }
      if (this.closed) {
        if (this.failure !== undefined) throw this.failure;
        return;
      }
      const result: IteratorResult<T> = await new Promise<IteratorResult<T>>(
        (resolve, reject) => {
          this.waiters.push({ resolve, reject });
        },
      );
      if (result.done === true) {
        if (this.failure !== undefined) throw this.failure;
        return;
      }
      yield result.value;
    }
  }
}

/** Normalise a thrown value into a real `Error` so it can be re-thrown. */
function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
