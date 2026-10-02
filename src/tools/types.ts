import type { Schema } from '../schema.js';
import type { ToolSpec } from '../providers/types.js';
import type { ToolCall } from '../providers/types.js';
import type { Logger } from '../utils/logger.js';

/**
 * Everything a tool's `execute` can see about the current run.
 *
 * `TState` is the agent's user-defined shared state: tools read from it and
 * write to it, which is how you build stateful multi-step workflows without a
 * global variable.
 */
export interface ToolContext<TState = unknown> {
  /** Aborts when the run is cancelled or its deadline passes. */
  readonly signal: AbortSignal;
  /** Id of the run currently in flight. */
  readonly runId: string;
  /** Shared mutable state for this agent instance. */
  readonly state: TState;
  /** The tool call that triggered this invocation. */
  readonly call: ToolCall;
  /** Which loop iteration (1-based) this call belongs to. */
  readonly iteration: number;
  /** Convenience logger, already tagged with agent/run/tool names. */
  readonly logger: Logger;
}

/**
 * The widest tool shape.
 *
 * A `ToolRegistry` is a heterogeneous collection: it holds tools whose argument
 * and result types are unrelated. `unknown` is not a substitute because it
 * erases the concrete types when `defineTool` narrows a `ToolDefinition`, so the
 * erasure is confined to this single named alias.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyTool<TState = unknown> = ToolDefinition<any, any, TState>;

/** The erased agent-state type, for registries shared across differing states. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyState = any;

/** A tool's `execute` implementation. */
export type ToolExecute<TArgs, TResult, TState = unknown> = (
  args: TArgs,
  context: ToolContext<TState>,
) => TResult | Promise<TResult>;

/** Serialized form of a tool result, ready to hand back to the model. */
export interface ToolResultPayload {
  /** String rendered into the conversation. */
  readonly content: string;
  /** Original return value, kept for programmatic consumers. */
  readonly value: unknown;
  /** `true` when the tool failed. */
  readonly isError: boolean;
  readonly durationMs: number;
  /** Number of attempts actually made (accounting for tool-level retries). */
  readonly attempts: number;
}

/**
 * Argument/result types erase to `any` by default so a registry can hold
 * tools with unrelated signatures; {@link AnyTool} names that erasure.
 */
export interface ToolDefinition<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TArgs = any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TResult = any,
  TState = unknown,
> extends ToolSpec {
  readonly name: string;
  readonly description: string;
  /**
   * Argument schema. Also converted to JSON Schema for the provider. Omit for
   * tools that take no arguments.
   */
  readonly parameters?: Schema<TArgs>;
  /**
   * Explicit JSON Schema for the provider API. Overrides the value derived
   * from `parameters` — use when the auto-conversion is not good enough.
   */
  readonly jsonSchema?: Record<string, unknown>;
  // Also a method so its parameters are checked bivariantly: a `TState` of
  // `{ user: string }` then stays assignable to the erased `unknown` used by
  // `ToolRegistry`, instead of failing on contravariance.
  execute(args: TArgs, context: ToolContext<TState>): TResult | Promise<TResult>;
  /** Per-call deadline. Omit to inherit the agent timeout. */
  readonly timeoutMs?: number;
  /** Retry budget for transient failures inside the tool. Default `0`. */
  readonly maxRetries?: number;
  /** Hide from the model's tool list without removing it from the registry. */
  readonly hidden?: boolean;
  /** Requires human approval before executing. See {@link ApprovalRequest}. */
  readonly requiresApproval?: boolean | ((args: TArgs) => boolean);
  /**
   * Convert the return value into text for the model. Return `undefined` to
   * use the default JSON/string rendering.
   */
  // Declared as a method (not a property holding a function) so TypeScript
  // checks its parameter bivariantly. That is what lets a registry hold tools
  // with unrelated `TResult`s — including tools whose `execute` only throws and
  // therefore infers `TResult = never`.
  serialize?(result: TResult): string | undefined;
  /** Free-form metadata for hooks and telemetry. */
  readonly metadata?: Readonly<Record<string, unknown>> | undefined;
}

/** Request handed to an approval callback. */
export interface ApprovalRequest<TArgs = unknown> {
  readonly toolName: string;
  readonly args: TArgs;
  readonly call: ToolCall;
  readonly runId: string;
  readonly iteration: number;
  /** Render a human-readable summary for the approver. */
  readonly summary: string;
}

/** Returns `true` to allow the call. */
export type ApprovalHandler<TState = unknown> = (
  request: ApprovalRequest,
  context: { signal: AbortSignal; state: TState },
) => boolean | Promise<boolean>;

/** Verdict returned by a {@link ToolPolicy}. */
export type ToolDecision =
  { readonly action: 'allow' } | { readonly action: 'deny'; readonly reason: string };

export interface PolicyContext<TState = unknown> {
  readonly runId: string;
  readonly iteration: number;
  readonly signal: AbortSignal;
  readonly state: TState;
}

/**
 * Gate that runs before every tool call. Use for authorization, rate limits,
 * or redacting arguments in dev environments.
 *
 * ```ts
 * const policy: ToolPolicy = ({ call }) =>
 *   call.name === 'deleteFile'
 *     ? { action: 'deny', reason: 'destructive tools are disabled' }
 *     : { action: 'allow' };
 * ```
 */
export type ToolPolicy<TState = unknown> = (
  call: ToolCall,
  context: PolicyContext<TState>,
) => ToolDecision | Promise<ToolDecision>;

export interface ToolRecord {
  readonly call: ToolCall;
  readonly result: ToolResultPayload;
  readonly toolName: string;
}

/** Narrowing helper: did the tool fail? */
export function isErrorResult(result: ToolResultPayload): boolean {
  return result.isError;
}
