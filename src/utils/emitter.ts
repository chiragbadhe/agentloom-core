/** Callback used to unsubscribe from an event. */
export type Unsubscribe = () => void;

/** Map of event name to listener argument tuple. */
export type EventMap = Record<string, readonly unknown[]>;

/**
 * Internal listener type. `never[]` makes every call site uncallable, which is
 * what keeps stored listeners from being invoked with the wrong arity.
 */
type AnyListener = (...args: never[]) => void;

/**
 * A tiny, dependency-free, strongly typed event emitter.
 *
 * Deliberately not Node's `EventEmitter`: listeners are typed per event name,
 * a throwing listener never breaks the producer, and `emit` is synchronous so
 * ordering is obvious.
 */
export class TypedEventEmitter<M extends EventMap> {
  private readonly listeners = new Map<keyof M, Set<AnyListener>>();
  private readonly onceWrappers = new WeakMap<AnyListener, Unsubscribe>();

  /** Subscribe to an event. Returns an idempotent unsubscribe function. */
  on<K extends keyof M & string>(
    event: K,
    listener: (...args: M[K]) => void,
  ): Unsubscribe {
    return this.register(event, listener, false);
  }

  /** Subscribe for a single delivery. */
  once<K extends keyof M & string>(
    event: K,
    listener: (...args: M[K]) => void,
  ): Unsubscribe {
    return this.register(event, listener, true);
  }

  private register<K extends keyof M & string>(
    event: K,
    listener: (...args: M[K]) => void,
    isOnce: boolean,
  ): Unsubscribe {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    const wrapped = listener as unknown as AnyListener;
    set.add(wrapped);
    if (isOnce) this.onceWrappers.set(wrapped, () => set?.delete(wrapped));

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.onceWrappers.get(wrapped)?.();
      set.delete(wrapped);
    };
  }

  /**
   * Synchronously invoke every listener for `event`.
   *
   * @returns the number of listeners invoked.
   */
  emit<K extends keyof M & string>(event: K, ...args: M[K]): number {
    const set = this.listeners.get(event);
    if (!set || set.size === 0) return 0;
    let count = 0;
    // Copy first: a listener may unsubscribe itself or others mid-dispatch.
    for (const listener of [...set]) {
      // Only `once` listeners are retired here; a plain `on` listener stays
      // subscribed until its disposer (or `off`) runs.
      if (this.onceWrappers.has(listener)) {
        this.onceWrappers.delete(listener);
        set.delete(listener);
      }
      try {
        (listener as unknown as (...a: M[K]) => void)(...args);
        count++;
      } catch {
        // An observer must never be able to break the agent loop.
      }
    }
    return count;
  }

  /** Subscribe to many events at once; returns a single unsubscribe. */
  onMany(
    handlers: Partial<{ [K in keyof M & string]: (...args: M[K]) => void }>,
  ): Unsubscribe {
    const disposers = Object.entries(handlers).map(([event, listener]) =>
      listener ? this.on(event as keyof M & string, listener as never) : undefined,
    );
    return () => {
      for (const dispose of disposers) dispose?.();
    };
  }

  listenerCount<K extends keyof M & string>(event: K): number {
    return this.listeners.get(event)?.size ?? 0;
  }

  /** Every event with at least one listener. */
  eventNames(): (keyof M & string)[] {
    return [...this.listeners.entries()]
      .filter(([, set]) => set.size > 0)
      .map(([event]) => event as keyof M & string);
  }

  /**
   * Remove a previously registered listener.
   *
   * `on()` returns a disposer, which is usually the better way to unsubscribe
   * (it cannot accidentally remove an equal-but-distinct arrow function).
   */
  off<K extends keyof M & string>(event: K, listener: (...args: M[K]) => void): boolean {
    const set = this.listeners.get(event);
    if (!set) return false;
    const before = set.size;
    set.delete(listener as unknown as AnyListener);
    this.onceWrappers.delete(listener as unknown as AnyListener);
    return set.size !== before;
  }

  /** Drop every listener, or every listener for one event. */
  removeAllListeners<K extends keyof M & string>(event?: K): void {
    // `onceWrappers` is a WeakMap, so entries have to be dropped per listener.
    const drop = (set: Set<AnyListener>): void => {
      for (const listener of set) this.onceWrappers.delete(listener);
      set.clear();
    };

    if (event === undefined) {
      for (const set of this.listeners.values()) drop(set);
      return;
    }
    const set = this.listeners.get(event);
    if (set !== undefined) drop(set);
  }

  /**
   * Resolve on the next `event` that satisfies `predicate`.
   *
   * ```ts
   * const done = emitter.waitFor('agent:end', (e) => !e.error);
   * agent.run('go');
   * await done;
   * ```
   */
  waitFor<K extends keyof M & string>(
    event: K,
    predicate: (...args: M[K]) => boolean = () => true,
  ): Promise<M[K][0]> {
    return new Promise<M[K][0]>((resolve) => {
      const off = this.on(event, (...args) => {
        if (!predicate(...args)) return;
        off();
        resolve(args[0]);
      });
    });
  }
}
