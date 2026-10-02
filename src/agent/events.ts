import type { AgentError } from '../errors.js';
import type {
  CompletionRequest,
  CompletionResult,
  ModelMessage,
  Usage,
} from '../providers/types.js';
import type { ToolCall } from '../providers/types.js';
import type { ToolMessage } from '../providers/types.js';
import type { Unsubscribe } from '../utils/emitter.js';
import type { Plan, PlanStep } from './planner.js';
import type { AgentLimits } from './result.js';

/**
 * Every event the agent emits, with its payload.
 *
 * Subscribe with `agent.on('tool:end', handler)`. Events are synchronous and
 * a throwing listener never interrupts a run.
 *
 * ```ts
 * agent.on('model:end', ({ usage, latencyMs }) => metrics.timing('model', latencyMs));
 * ```
 */
export type AgentEventMap = {
  /** A run has started. */
  'agent:start': [event: AgentStartEvent];
  /** A run has finished, successfully or not. */
  'agent:end': [event: AgentEndEvent];
  /** The system prompt was assembled for an iteration. */
  'agent:prompt': [event: PromptEvent];
  /** A loop iteration began. */
  'iteration:start': [event: IterationStartEvent];
  /** A loop iteration finished. */
  'iteration:end': [event: IterationEndEvent];
  /** About to call the model. */
  'model:start': [event: ModelStartEvent];
  /** The model returned. */
  'model:end': [event: ModelEndEvent];
  /** A text chunk arrived while streaming. */
  'model:delta': [event: ModelDeltaEvent];
  /** A tool call is about to run. */
  'tool:start': [event: ToolStartEvent];
  /** A tool call succeeded. */
  'tool:end': [event: ToolEndEvent];
  /** A tool call failed; the error was fed back to the model. */
  'tool:error': [event: ToolErrorEvent];
  /** A retry is scheduled. */
  retry: [event: RetryEvent];
  /** A plan was produced or updated. */
  plan: [event: PlanEvent];
  /** A message was appended to memory. */
  memory: [event: MemoryEvent];
  /** Structured output was rejected; a repair pass is starting. */
  'output:invalid': [event: OutputInvalidEvent];
  /** Any error, including recoverable ones. */
  error: [event: ErrorEvent];
};

export interface BaseRunEvent {
  readonly runId: string;
}

export interface AgentStartEvent extends BaseRunEvent {
  readonly input: string;
  readonly agentName: string;
  readonly provider: string;
  readonly model: string;
  readonly tools: readonly string[];
  readonly limits: AgentLimits;
  readonly startedAt: number;
}

export interface AgentEndEvent extends BaseRunEvent {
  readonly stopReason: string;
  readonly output: string;
  readonly iterations: number;
  readonly usage: Usage;
  readonly durationMs: number;
}

export interface PromptEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly messages: readonly ModelMessage[];
  readonly estimatedTokens: number;
  readonly system: string | undefined;
  readonly recalled: readonly string[];
}

export interface IterationStartEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly messageCount: number;
}

export interface IterationEndEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly toolCallCount: number;
  readonly durationMs: number;
  readonly usage: Usage;
}

export interface ModelStartEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly request: CompletionRequest;
  readonly attempt: number;
}

export interface ModelEndEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly result: CompletionResult;
  readonly durationMs: number;
}

export interface ModelDeltaEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly text: string;
  readonly reasoning?: boolean;
}

export interface ToolStartEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly call: ToolCall;
  readonly args: unknown;
  readonly toolName: string;
}

export interface ToolEndEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly call: ToolCall;
  readonly toolName: string;
  readonly result: ToolMessage;
  readonly durationMs: number;
}

export interface ToolErrorEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly call: ToolCall;
  readonly toolName: string;
  readonly error: AgentError;
  readonly durationMs: number;
}

export interface RetryEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly attempt: number;
  readonly remaining: number;
  readonly delayMs: number;
  readonly error: AgentError;
  readonly scope: 'model' | 'tool';
  readonly toolName?: string;
}

export interface PlanEvent extends BaseRunEvent {
  readonly plan: Plan;
  readonly steps: readonly PlanStep[];
  readonly completed: number;
}

export interface MemoryEvent extends BaseRunEvent {
  readonly operation: 'recall' | 'store' | 'trim' | 'summarize';
  readonly count: number;
  readonly detail?: string;
}

export interface OutputInvalidEvent extends BaseRunEvent {
  readonly iteration: number;
  readonly attempt: number;
  readonly issues: readonly { path: readonly (string | number)[]; message: string }[];
  readonly raw: string;
}

export interface ErrorEvent extends BaseRunEvent {
  readonly error: AgentError;
  readonly iteration: number;
  readonly fatal: boolean;
  readonly scope: 'model' | 'tool' | 'memory' | 'output' | 'agent';
}

/** All event names, useful for exhaustiveness checks in wrappers. */
export type AgentEventName = keyof AgentEventMap;

/** Callback signature returned by `agent.on`. */
export type AgentEventListener<K extends AgentEventName> = (
  event: AgentEventMap[K][0],
) => void;

export type { Unsubscribe };
