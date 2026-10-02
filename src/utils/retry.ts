import { type AgentError, isAbortLikeError, toAgentError } from '../errors.js';
import { sleep } from './async.js';

export interface RetryContext {
  /** 1-based attempt number that just failed. */
  readonly attempt: number;
  /** Attempts remaining after this one (0 on the final attempt). */
  readonly remaining: number;
  readonly error: AgentError;
  /** Delay before the next attempt, in milliseconds. */
  readonly delayMs: number;
}

export interface RetryOptions {
  /** Total attempts including the first. Default `3`. */
  readonly maxAttempts?: number;
  /** First backoff delay in ms. Default `500`. */
  readonly initialDelayMs?: number;
  /** Upper bound on any single delay. Default `30_000`. */
  readonly maxDelayMs?: number;
  /** Exponential growth factor. Default `2`. */
  readonly backoffFactor?: number;
  /** Random fraction added to each delay, 0–1. Default `0.2`. */
  readonly jitter?: number;
  /** Cap on total time spent retrying. Default: unlimited. */
  readonly maxElapsedMs?: number;
  /** Decide whether a failure is worth retrying. Default: `error.retryable`. */
  readonly shouldRetry?: (error: AgentError, attempt: number) => boolean;
  /** Honour a server-provided `retryAfterMs` when present. Default `true`. */
  readonly respectRetryAfter?: boolean;
  /** Observability hook invoked before each retry. */
  readonly onRetry?: (context: RetryContext) => void | Promise<void>;
  /** Abort retries early. Default: never abort. */
  readonly signal?: AbortSignal;
}

export interface ResolvedRetryPolicy {
  readonly maxAttempts: number;
  readonly initialDelayMs: number;
  readonly maxDelayMs: number;
  readonly backoffFactor: number;
  readonly jitter: number;
  readonly maxElapsedMs: number | undefined;
  readonly respectRetryAfter: boolean;
  readonly shouldRetry: (error: AgentError, attempt: number) => boolean;
  readonly onRetry: ((context: RetryContext) => void | Promise<void>) | undefined;
  readonly signal: AbortSignal | undefined;
}

const DEFAULT_RETRY: ResolvedRetryPolicy = {
  maxAttempts: 3,
  initialDelayMs: 500,
  maxDelayMs: 30_000,
  backoffFactor: 2,
  jitter: 0.2,
  maxElapsedMs: undefined,
  respectRetryAfter: true,
  shouldRetry: (error) => error.retryable,
  onRetry: undefined,
  signal: undefined,
};

export function resolveRetryPolicy(options: RetryOptions = {}): ResolvedRetryPolicy {
  return {
    ...DEFAULT_RETRY,
    ...stripUndefined(options),
    maxAttempts: Math.max(
      1,
      Math.floor(options.maxAttempts ?? DEFAULT_RETRY.maxAttempts),
    ),
  };
}

function stripUndefined<T extends object>(input: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) out[key] = value;
  }
  return out as Partial<T>;
}

/**
 * Full-jitter-ish exponential backoff: `initial * factor^(attempt-1)`,
 * clamped to `maxDelayMs`, then spread by ±`jitter`. A server-supplied
 * `retryAfterMs` takes precedence because it is ground truth.
 */
export function computeBackoffDelay(
  attempt: number,
  policy: Pick<
    ResolvedRetryPolicy,
    'initialDelayMs' | 'backoffFactor' | 'maxDelayMs' | 'jitter'
  >,
  retryAfterMs?: number,
): number {
  if (retryAfterMs !== undefined && retryAfterMs >= 0) {
    return Math.min(retryAfterMs, policy.maxDelayMs);
  }
  const exponential =
    policy.initialDelayMs * policy.backoffFactor ** Math.max(0, attempt - 1);
  const capped = Math.min(exponential, policy.maxDelayMs);
  if (policy.jitter <= 0) return Math.round(capped);
  const spread = capped * policy.jitter;
  const offset = (Math.random() * 2 - 1) * spread;
  return Math.max(0, Math.round(capped + offset));
}

/**
 * Execute `fn` with retries. Abort errors are never retried.
 *
 * ```ts
 * const result = await withRetry(() => provider.complete(req), {
 *   maxAttempts: 5,
 *   shouldRetry: (err) => err.code === 'PROVIDER_RATE_LIMIT_ERROR',
 * });
 * ```
 */
export async function withRetry<T>(
  fn: (attempt: number) => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const policy = resolveRetryPolicy(options);
  const startedAt = Date.now();
  let attempt = 0;

  for (;;) {
    attempt++;
    try {
      return await fn(attempt);
    } catch (rawError) {
      const error = toAgentError(rawError, 'Operation failed');

      if (attempt >= policy.maxAttempts) throw error;
      if (isAbortLikeError(error) || policy.signal?.aborted) throw error;
      if (!policy.shouldRetry(error, attempt)) throw error;

      const delayMs = computeBackoffDelay(attempt, policy, error.retryAfterMs);

      if (policy.maxElapsedMs !== undefined) {
        const projected = Date.now() - startedAt + delayMs;
        if (projected > policy.maxElapsedMs) throw error;
      }

      const context: RetryContext = {
        attempt,
        remaining: policy.maxAttempts - attempt,
        error,
        delayMs,
      };

      await policy.onRetry?.(context);
      await sleep(delayMs, policy.signal);

      if (policy.signal?.aborted) throw error;
    }
  }
}
