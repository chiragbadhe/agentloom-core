import type { Usage } from '../providers/types.js';
import type { ResolvedLimits, StopReason } from './result.js';

/** A mutable view of {@link Usage}, for accumulating counters. */
type MutableUsage = { -readonly [K in keyof Usage]: Usage[K] };
/**
 * Tracks cumulative usage and enforces the run's resource limits.
 *
 * Limits are checked *before* spending, never after, so a run cannot overshoot
 * its token budget by an entire model call.
 */
export class BudgetTracker {
  readonly usage: Usage = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cachedInputTokens: 0,
    reasoningTokens: 0,
  };

  iterations = 0;
  toolCalls = 0;
  modelCalls = 0;

  private readonly limits: ResolvedLimits;
  private readonly startedAt = Date.now();

  constructor(limits: ResolvedLimits) {
    this.limits = limits;
  }

  get elapsedMs(): number {
    return Date.now() - this.startedAt;
  }

  get totalTokens(): number {
    return this.usage.totalTokens ?? 0;
  }

  /** Fold one model response into the running totals. */
  record(usage: Usage): void {
    this.modelCalls++;

    const sum = (a: number | undefined, b: number | undefined): number =>
      (a ?? 0) + (b ?? 0);
    const inputTokens = sum(this.usage.inputTokens, usage.inputTokens);
    const outputTokens = sum(this.usage.outputTokens, usage.outputTokens);
    const next: MutableUsage = {
      inputTokens,
      outputTokens,
      cachedInputTokens: sum(this.usage.cachedInputTokens, usage.cachedInputTokens),
      reasoningTokens: sum(this.usage.reasoningTokens, usage.reasoningTokens),
      totalTokens: 0,
    };

    // Prefer the provider's own total; fall back to input + output when the
    // provider omits it (Ollama and Gemini frequently do).
    next.totalTokens =
      usage.totalTokens !== undefined
        ? sum(this.usage.totalTokens, usage.totalTokens)
        : inputTokens + outputTokens;

    Object.assign(this.usage, next);
  }

  /**
   * Check limits before starting another iteration.
   *
   * @returns the stop reason, or `undefined` when there is budget left.
   */
  check(
    options: { pendingToolCalls?: number; isAbort?: boolean } = {},
  ): StopReason | undefined {
    if (options.isAbort === true) return 'aborted';
    if (this.iterations >= this.limits.maxIterations) return 'max_iterations';

    if (
      this.limits.maxTotalTokens !== undefined &&
      this.totalTokens >= this.limits.maxTotalTokens
    ) {
      return 'max_tokens';
    }

    if (this.limits.timeoutMs !== undefined && this.elapsedMs >= this.limits.timeoutMs) {
      return 'timeout';
    }

    const pending = options.pendingToolCalls ?? 0;
    if (
      this.limits.maxToolCalls !== undefined &&
      this.toolCalls + pending > this.limits.maxToolCalls
    ) {
      return 'max_tool_calls';
    }

    return undefined;
  }

  /** True when adding `count` more tool calls would exceed the budget. */
  exceedsToolCalls(count: number): boolean {
    const cap = this.limits.maxToolCalls;
    if (cap === undefined) return false;
    return this.toolCalls + count > cap;
  }

  /** Cap the number of tools a single iteration may run. */
  clampIterationToolCalls(count: number): number {
    const perIteration = this.limits.maxToolCallsPerIteration;
    const remaining = this.limits.maxToolCalls;
    let allowed = count;
    if (perIteration !== undefined) allowed = Math.min(allowed, perIteration);
    if (remaining !== undefined) {
      allowed = Math.min(allowed, Math.max(0, remaining - this.toolCalls));
    }
    return allowed;
  }

  /** Whether the assembled prompt fits the configured context window. */
  fitsContext(promptTokens: number, reserveForOutput = 0): boolean {
    return promptTokens + reserveForOutput <= this.limits.contextWindow;
  }

  /** Estimated tokens still available. `Infinity` when uncapped. */
  remainingTokens(): number {
    const cap = this.limits.maxTotalTokens;
    if (cap === undefined) return Number.POSITIVE_INFINITY;
    return Math.max(0, cap - this.totalTokens);
  }

  /** A snapshot suitable for logging or including in results. */
  snapshot(): {
    iterations: number;
    modelCalls: number;
    toolCalls: number;
    totalTokens: number;
    elapsedMs: number;
    limits: ResolvedLimits;
  } {
    return {
      iterations: this.iterations,
      modelCalls: this.modelCalls,
      toolCalls: this.toolCalls,
      totalTokens: this.totalTokens,
      elapsedMs: this.elapsedMs,
      limits: this.limits,
    };
  }
}
