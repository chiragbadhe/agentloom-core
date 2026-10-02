import type { AgentError } from '../errors.js';
import type { Usage } from '../providers/types.js';
import type { AsyncQueue } from '../utils/queue.js';
import type { Unsubscribe } from '../utils/emitter.js';
import type { AgentResult } from './result.js';
import type { Plan } from './planner.js';

/**
 * Events yielded by {@link AgentStream}.
 *
 * A flat, discriminated union — everything carries a `type`, so `switch` on
 * `event.type` is exhaustive.
 */
export type AgentStreamEvent =
  | {
      type: 'start';
      runId: string;
      agentName: string;
      model: string;
      input: string;
      tools: readonly string[];
    }
  | { type: 'text-delta'; runId: string; iteration: number; text: string }
  | { type: 'text'; runId: string; iteration: number; text: string }
  | { type: 'reasoning-delta'; runId: string; iteration: number; text: string }
  | { type: 'tool-call'; runId: string; iteration: number; name: string; args: unknown }
  | {
      type: 'tool-result';
      runId: string;
      iteration: number;
      name: string;
      content: string;
      isError: boolean;
      durationMs: number;
    }
  | {
      type: 'iteration';
      runId: string;
      iteration: number;
      toolCallCount: number;
      usage: Usage;
      durationMs: number;
    }
  | { type: 'plan'; runId: string; plan: Plan; completed: number }
  | {
      type: 'retry';
      runId: string;
      iteration: number;
      delayMs: number;
      error: AgentError;
    }
  | { type: 'error'; runId: string; error: AgentError; fatal: boolean; iteration: number }
  | { type: 'done'; runId: string; result: AgentResult<unknown> };

/**
 * Handle for a running agent.
 *
 * Iterate for live events, or just `await run.result`. Cancelling is done
 * through the `AbortSignal` passed to `agent.stream(...)`, or through
 * {@link AgentStream.abort}.
 *
 * ```ts
 * const run = agent.stream('summarize this', { signal });
 * run.on('text', ({ text }) => process.stdout.write(text));
 * const result = await run.result;
 * ```
 */
export class AgentStream<TEvent extends { type: string }, TResult> {
  /** Resolves with the final result. Never rejects on provider errors unless
   * the run was configured to throw. */
  result: Promise<TResult>;

  /** Id of this run. */
  readonly id: string;

  private readonly queue: AsyncQueue<TEvent>;
  private readonly controller: AbortController;
  private readonly dispose: () => void;
  private readonly listeners = new Map<string, Set<(event: unknown) => void>>();

  constructor(
    queue: AsyncQueue<TEvent>,
    controller: AbortController,
    dispose: () => void,
    id: string,
  ) {
    this.queue = queue;
    this.controller = controller;
    this.dispose = dispose;
    this.id = id;
    this.result = Promise.resolve(undefined as unknown as TResult);
  }

  /** Cancel the run. Safe to call more than once. */
  abort(reason?: unknown): void {
    if (!this.controller.signal.aborted) this.controller.abort(reason);
  }

  get aborted(): boolean {
    return this.controller.signal.aborted;
  }

  /** Subscribe to stream events by `type`. */
  on<K extends TEvent['type']>(
    type: K,
    listener: (event: Extract<TEvent, { type: K }>) => void,
  ): Unsubscribe {
    let set = this.listeners.get(type);
    if (set === undefined) {
      set = new Set();
      this.listeners.set(type, set);
    }
    const wrapped = listener as (event: unknown) => void;
    set.add(wrapped);
    return () => set?.delete(wrapped);
  }

  /** Iterate events as they arrive. */
  async *[Symbol.asyncIterator](): AsyncGenerator<TEvent> {
    for await (const event of this.queue) {
      const set = this.listeners.get(event.type);
      if (set !== undefined) {
        for (const listener of set) {
          try {
            listener(event);
          } catch {
            // A listener must not break the stream.
          }
        }
      }
      yield event;
    }
  }

  /**
   * Collect the full assistant answer as a single string, ignoring every other
   * event. The most common way to consume a stream.
   */
  async text(): Promise<string> {
    let out = '';
    for await (const event of this as AsyncIterable<AgentStreamEvent>) {
      if (event.type === 'text') out += event.text;
    }
    return out;
  }

  /** Release resources without waiting for the run. */
  close(): void {
    this.dispose();
  }
}
