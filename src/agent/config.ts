import type { CompletionOptions, ModelProvider } from '../providers/types.js';
import type { Schema } from '../schema.js';
import type { LongTermMemory } from '../memory/long-term.js';
import type { ConversationMemory, TokenCounter } from '../memory/types.js';
import type { AnyTool, ApprovalHandler, ToolPolicy } from '../tools/types.js';
import type { ToolExecutionMode } from '../tools/executor.js';
import type { ToolRegistry } from '../tools/registry.js';
import type { Logger } from '../utils/logger.js';
import type { RetryOptions } from '../utils/retry.js';
import type { AgentHooks } from './hooks.js';
import type { Plan, PlanStep } from './planner.js';
import type { AgentLimits } from './result.js';

/** Anything that can produce a system prompt. */
export type SystemPromptProvider<TState = unknown> = (context: {
  input: string;
  runId: string;
  state: TState;
}) => string | Promise<string>;

/**
 * Configuration for a single run. Everything here is optional and overrides the
 * agent's own configuration for that run only.
 */
export interface RunOptions<TOut = unknown, TState = unknown> {
  /** Cancel the run. Composed with the agent's own controller. */
  readonly signal?: AbortSignal;
  /** Supply your own id for correlation in logs and traces. */
  readonly runId?: string;
  /** Appended to the agent instructions. */
  readonly system?: string;
  /** Replaces the agent instructions entirely. */
  readonly instructions?: string;
  /** Override the model for this run, e.g. `anthropic:claude-sonnet-4-5`. */
  readonly model?: string;
  /** Narrow or extend the tools available for this run. */
  readonly tools?: ToolRegistry<TState> | readonly AnyTool<TState>[] | undefined;
  /** Per-run limit overrides. */
  readonly limits?: AgentLimits;
  /** Validate the final answer against a schema. */
  readonly outputSchema?: Schema<TOut>;
  /** Extra instructions for the structured output. */
  readonly outputInstructions?: string;
  /**
   * Throw on fatal errors (default `true`). Set `false` to receive an
   * `AgentResult` with `stopReason: 'error'` and `error` populated instead.
   */
  readonly throwOnError?: boolean;
  /** Attached to events and forwarded to the provider as metadata. */
  readonly metadata?: Record<string, unknown>;
}

/** Structured-output configuration on the agent itself. */
export interface OutputConfig<TOut = unknown> {
  readonly schema: Schema<TOut>;
  readonly instructions?: string;
  /** Name used with native structured output. Default `response`. */
  readonly name?: string;
  /**
   * Request provider-enforced schema adherence (`strict` JSON schema).
   * Default `true`, and silently downgraded when unsupported.
   */
  readonly strict?: boolean;
}

/**
 * An {@link OutputConfig} for any validator.
 *
 * The agent cannot know the shape up front, so the schema type is erased here
 * rather than defaulted to `never` (which nothing would be assignable to).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyOutputConfig = OutputConfig<any>;

/** Long-term memory wiring. */
export interface LongTermMemoryConfig {
  /** Recall relevant records and inject them into the system prompt. */
  readonly recall?: boolean;
  /** Store the user input and final answer after each run. */
  readonly store?: boolean;
  readonly limit?: number;
  readonly minScore?: number;
  readonly roles?: readonly ('user' | 'assistant' | 'system')[];
  /** Heading used when rendering recalled memories. Default `Relevant memories`. */
  readonly heading?: string;
}

export interface PlannerConfig<TState = unknown> {
  /** Master switch. Equivalent to `planner: false`. */
  readonly enabled?: boolean;
  /** Create a plan before the loop starts. Default `true`. */
  readonly planOnRun?: boolean;
  /** Register an `update_plan` tool so the model can report progress. */
  readonly exposeUpdateTool?: boolean;
  /** Maximum steps in a generated plan. Default `8`. */
  readonly maxSteps?: number;
  /** Custom plan builder; defaults to a structured model call. */
  readonly createPlan?: (
    input: string,
    context: { runId: string; signal: AbortSignal; state: TState },
  ) => Promise<Plan>;
  /** Custom plan renderer; defaults to a markdown checklist. */
  readonly render?: (plan: Plan) => string;
  /** Reuse a plan across runs instead of creating a new one each time. */
  readonly persist?: boolean;
}

/**
 * Everything needed to construct an {@link Agent}.
 *
 * The only required field is `model` (or `provider`). Everything else has a
 * sensible default, so the smallest useful agent is one line:
 *
 * ```ts
 * const agent = new Agent({ model: 'openai:gpt-4o-mini' });
 * ```
 */
export interface AgentConfig<TState = unknown> {
  /** Used in logs, events, and error messages. Default `agent`. */
  readonly name?: string;
  /** Behaviour and constraints for the model. */
  readonly instructions?: string | SystemPromptProvider<TState>;
  /**
   * Model reference as `"provider:model"` — `openai:gpt-4o-mini`,
   * `anthropic:claude-sonnet-4-5`, `google:gemini-2.0-flash`, `ollama:llama3.2`.
   * Ignored when `provider` is given.
   */
  readonly model?: string;
  /** Use a provider instance directly, bypassing the registry. */
  readonly provider?: ModelProvider;
  /** Options forwarded to the registry when `model` resolves a provider. */
  readonly providerOptions?: Record<string, unknown>;
  /** Sampling and budget knobs applied to every model call. */
  readonly modelOptions?: CompletionOptions;
  readonly tools?: ToolRegistry<TState> | readonly AnyTool<TState>[] | undefined;
  readonly memory?: ConversationMemory;
  readonly longTermMemory?: LongTermMemory;
  readonly longTerm?: LongTermMemoryConfig;
  readonly planner?: PlannerConfig<TState> | false;
  /**
   * Default structured output for every run.
   *
   * Erased so any validator fits here; `run<TOut>()` still types `data`
   * precisely when the schema is passed per run.
   */
  readonly output?: AnyOutputConfig;
  readonly limits?: AgentLimits;
  /** Tool execution strategy. Default `sequential`. */
  readonly execution?: ToolExecutionMode;
  /** Max simultaneous tools when `execution` is `parallel`. Default `4`. */
  readonly toolConcurrency?: number;
  readonly toolPolicy?: ToolPolicy<TState>;
  readonly approval?: ApprovalHandler<TState>;
  /** Throw when a tool fails instead of feeding the error back. Default `false`. */
  readonly throwOnToolError?: boolean;
  /** Truncate tool output past this many characters. Default `20000`. */
  readonly maxToolResultLength?: number;
  /** Provider call retry policy. Default: 3 attempts with backoff. */
  readonly retry?: RetryOptions | false;
  readonly hooks?: AgentHooks<TState>;
  readonly logger?: Logger;
  /** Initial value of the shared state object. */
  readonly state?: TState;
  /** Override token estimation (e.g. with a real tokenizer). */
  readonly tokenCounter?: TokenCounter;
}

/** Convenience alias for the plan item shape. */
export type { Plan, PlanStep };
