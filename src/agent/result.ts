import type { AgentError } from '../errors.js';
import type {
  CompletionRequest,
  CompletionResult,
  ModelMessage,
  Usage,
} from '../providers/types.js';
import type { ToolCall } from '../providers/types.js';
import type { ToolMessage } from '../providers/types.js';

/** Why a run ended. Always present on the result, even for failures. */
export type StopReason =
  /** The model produced a final answer with no tool calls. */
  | 'completed'
  /** A configured iteration/loop budget was exhausted. */
  | 'max_iterations'
  /** The tool-call budget was exhausted. */
  | 'max_tool_calls'
  /** The token budget was exhausted. */
  | 'max_tokens'
  /** The wall-clock deadline elapsed. */
  | 'timeout'
  /** The caller aborted via `AbortSignal`. */
  | 'aborted'
  /** A structured-output validation loop ran out of attempts. */
  | 'max_output_attempts'
  /** An unrecoverable error. Inspect `result.error`. */
  | 'error'
  /** A policy or hook stopped the run deliberately. */
  | 'cancelled';

/** One model call plus the tools it triggered. */
export interface AgentStep {
  /** 1-based loop iteration. */
  readonly iteration: number;
  readonly request: CompletionRequest;
  readonly response: CompletionResult;
  /** Tool calls the model requested, in order. */
  readonly toolCalls: readonly ToolCall[];
  /** Results, one per tool call, same order. */
  readonly toolResults: readonly ToolMessage[];
  readonly usage: Usage;
  readonly durationMs: number;
  /** Text emitted by this step (streamed or buffered). */
  readonly text: string;
}

export interface AgentResult<TData = unknown> {
  /**
   * Final answer. When an output schema is configured, prefer `data`; `output`
   * is the raw text the model produced.
   */
  readonly output: string;
  /** Validated structured output, when an `outputSchema` was requested. */
  readonly data: TData | undefined;
  /** Full conversation, including the system prompt and every tool exchange. */
  readonly messages: readonly ModelMessage[];
  readonly steps: readonly AgentStep[];
  readonly iterations: number;
  readonly usage: Usage;
  readonly stopReason: StopReason;
  /** Populated when `stopReason` is `error` (or a soft failure occurred). */
  readonly error: AgentError | undefined;
  readonly runId: string;
  readonly startedAt: number;
  readonly endedAt: number;
  readonly durationMs: number;
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

export interface AgentLimits {
  /**
   * Maximum model calls per run. Default `10`. This is the main runaway-loop
   * guard: every extra tool round trip costs another iteration.
   */
  readonly maxIterations?: number;
  /** Wall-clock budget for the whole run, in ms. Default: none. */
  readonly timeoutMs?: number;
  /** Total token budget (input + output) across the run. Default: none. */
  readonly maxTotalTokens?: number;
  /** Maximum output tokens per model call. */
  readonly maxTokensPerCall?: number;
  /** Maximum tool calls across the run. Default: none. */
  readonly maxToolCalls?: number;
  /** Maximum tool calls within a single iteration. Default: none. */
  readonly maxToolCallsPerIteration?: number;
  /** Hard token ceiling for the assembled prompt. Default `32000`. */
  readonly contextWindow?: number;
  /** Attempts allowed when structured output fails validation. Default `2`. */
  readonly maxOutputAttempts?: number;
  /** Per-model-call timeout. Default: none (bounded by `timeoutMs`). */
  readonly modelTimeoutMs?: number;
}

/** Resolved limits with every field populated. */
export interface ResolvedLimits {
  readonly maxIterations: number;
  readonly timeoutMs: number | undefined;
  readonly maxTotalTokens: number | undefined;
  readonly maxTokensPerCall: number | undefined;
  readonly maxToolCalls: number | undefined;
  readonly maxToolCallsPerIteration: number | undefined;
  readonly contextWindow: number;
  readonly maxOutputAttempts: number;
  readonly modelTimeoutMs: number | undefined;
}

export const DEFAULT_LIMITS: ResolvedLimits = {
  maxIterations: 10,
  timeoutMs: undefined,
  maxTotalTokens: undefined,
  maxTokensPerCall: undefined,
  maxToolCalls: undefined,
  maxToolCallsPerIteration: undefined,
  contextWindow: 32_000,
  maxOutputAttempts: 2,
  modelTimeoutMs: undefined,
};

/** Merge user limits over the defaults. */
export function resolveLimits(limits: AgentLimits = {}): ResolvedLimits {
  return { ...DEFAULT_LIMITS, ...definedOnly(limits) };
}

function definedOnly<T extends object>(input: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) out[key] = value;
  }
  return out as Partial<T>;
}
