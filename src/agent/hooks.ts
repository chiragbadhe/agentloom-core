import type { CompletionRequest } from '../providers/types.js';
import type {
  AgentEndEvent,
  AgentEventMap,
  AgentStartEvent,
  ErrorEvent,
  IterationEndEvent,
  IterationStartEvent,
  MemoryEvent,
  ModelDeltaEvent,
  ModelEndEvent,
  ModelStartEvent,
  OutputInvalidEvent,
  PromptEvent,
  RetryEvent,
  ToolEndEvent,
  ToolErrorEvent,
  ToolStartEvent,
} from './events.js';

/**
 * Lifecycle hooks — the "act" side of observability.
 *
 * Events (`agent.on(...)`) are for observing what happened; hooks are for
 * influencing it. A hook may:
 * - inspect or rewrite the prompt via `onPrompt`
 * - mutate messages before they are sent via `onBeforeModel`
 * - veto or rewrite the final answer via `onBeforeReturn`
 * - abort the run by throwing from `onAgentStart`
 *
 * Every hook is optional and awaited. A hook that throws never corrupts the
 * run: `onAgentStart`, `onBeforeModel`, and `onBeforeReturn` treat a throw as
 * a veto/abort, while observational hooks are isolated.
 */
export interface AgentHooks<TState = unknown> {
  onAgentStart?(event: AgentStartEvent): void | Promise<void>;
  /** Assemble the system prompt. Return `undefined` to keep the configured one. */
  onSystemPrompt?(context: {
    runId: string;
    input: string;
    state: TState;
    defaultPrompt: string;
  }): string | undefined | Promise<string | undefined>;
  /** Inspect or rewrite the final prompt. */
  onPrompt?(event: PromptEvent): void | Promise<void>;
  onBeforeModel?(event: {
    runId: string;
    iteration: number;
    request: CompletionRequest;
    state: TState;
    signal: AbortSignal;
  }): CompletionRequest | void | Promise<CompletionRequest | void>;
  onModelStart?(event: ModelStartEvent): void | Promise<void>;
  onModelDelta?(event: ModelDeltaEvent): void | Promise<void>;
  onModelEnd?(event: ModelEndEvent): void | Promise<void>;
  onIterationStart?(event: IterationStartEvent): void | Promise<void>;
  onIterationEnd?(event: IterationEndEvent): void | Promise<void>;
  onToolStart?(event: ToolStartEvent): void | Promise<void>;
  onToolEnd?(event: ToolEndEvent): void | Promise<void>;
  onToolError?(event: ToolErrorEvent): void | Promise<void>;
  onRetry?(event: RetryEvent): void | Promise<void>;
  onMemory?(event: MemoryEvent): void | Promise<void>;
  onOutputInvalid?(event: OutputInvalidEvent): void | Promise<void>;
  /**
   * Last chance to change (or veto) the result. Return a value to replace the
   * answer, or `undefined` to accept it. Throw to abort the run.
   */
  onBeforeReturn?(context: {
    runId: string;
    output: string;
    stopReason: string;
    data: unknown;
    state: TState;
  }): unknown;
  onAgentEnd?(event: AgentEndEvent): void | Promise<void>;
  onError?(event: ErrorEvent): void | Promise<void>;
}

/** Hook keys whose throw is meaningful rather than merely noisy. */
export const VETOING_HOOKS = [
  'onAgentStart',
  'onBeforeModel',
  'onBeforeReturn',
] as const satisfies readonly (keyof AgentHooks)[];

export type HookName = keyof AgentHooks;

/** Map of event name to hook name, for auto-wiring from a config. */
export type HookRegistry = {
  [K in keyof AgentEventMap]?: HookName;
};
