import type { Schema } from '../schema.js';

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/** A function/tool call requested by the model. */
export interface ToolCall {
  /** Provider-assigned id; echoed back with the result. */
  readonly id: string;
  readonly name: string;
  /** Parsed arguments. Objects for well-behaved models, sometimes strings. */
  readonly arguments: unknown;
  /** The original argument text, kept for debugging malformed calls. */
  readonly rawArguments?: string;
}

export interface SystemMessage {
  readonly role: 'system';
  readonly content: string;
}

export interface UserMessage {
  readonly role: 'user';
  readonly content: string;
  /** Optional participant name for multi-user transcripts. */
  readonly name?: string;
}

export interface AssistantMessage {
  readonly role: 'assistant';
  readonly content: string;
  readonly toolCalls?: readonly ToolCall[];
}

export interface ToolMessage {
  readonly role: 'tool';
  readonly content: string;
  readonly toolCallId: string;
  readonly name: string;
  /**
   * `true` when the tool failed. Providers surface this as an error result so
   * the model can self-correct instead of blindly trusting bad data.
   */
  readonly isError?: boolean;
}

/**
 * Provider-agnostic conversation message.
 *
 * A discriminated union on `role` so switch/if narrowing is exhaustive:
 *
 * ```ts
 * for (const message of messages) {
 *   switch (message.role) {
 *     case 'assistant': message.toolCalls?.length; break;
 *     case 'tool': message.isError; break;
 *   }
 * }
 * ```
 */
export type ModelMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

/** Narrowing helper. */
export function isToolCallMessage(message: ModelMessage): message is ToolMessage {
  return message.role === 'tool';
}

export function isAssistantMessage(message: ModelMessage): message is AssistantMessage {
  return message.role === 'assistant';
}

// ---------------------------------------------------------------------------
// Tools on the wire
// ---------------------------------------------------------------------------

/**
 * The provider-facing description of a tool. Intentionally decoupled from
 * `ToolDefinition` so the provider layer never imports the tools layer.
 */
export interface ToolSpec {
  readonly name: string;
  readonly description: string;
  /** Validates tool arguments, and supplies the JSON Schema for the API. */
  readonly parameters?: Schema<unknown>;
}

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

/**
 * How the model should constrain its output.
 *
 * - `text` — free-form (default)
 * - `json_object` — valid JSON only
 * - `json_schema` — valid JSON matching `schema` (strongest; not every
 *   provider supports it, and providers degrade it to `json_object` when it
 *   is unavailable)
 */
export type ResponseFormat =
  | { readonly type: 'text' }
  | { readonly type: 'json_object' }
  | {
      readonly type: 'json_schema';
      readonly schema: Record<string, unknown>;
      readonly name?: string;
      readonly description?: string;
      readonly strict?: boolean;
    };

/** Token accounting. All fields optional: not every provider reports all of them. */
export interface Usage {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly totalTokens?: number;
  readonly cachedInputTokens?: number;
  readonly reasoningTokens?: number;
}

/** Why the model stopped generating. */
export type FinishReason =
  'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error' | 'other';

/** Per-provider sampling and budget knobs. */
export interface CompletionOptions {
  readonly temperature?: number;
  readonly topP?: number;
  readonly topK?: number;
  readonly maxTokens?: number;
  readonly stopSequences?: readonly string[];
  readonly seed?: number;
  readonly presencePenalty?: number;
  readonly frequencyPenalty?: number;
  /**
   * Reasoning effort for models that support extended thinking.
   * Provider-specific; ignored where unsupported.
   */
  readonly reasoningEffort?: 'minimal' | 'low' | 'medium' | 'high';
}

// ---------------------------------------------------------------------------
// Requests / responses
// ---------------------------------------------------------------------------

export interface CompletionRequest extends CompletionOptions {
  readonly model: string;
  readonly messages: readonly ModelMessage[];
  readonly tools?: readonly ToolSpec[];
  readonly responseFormat?: ResponseFormat;
  /** Free-form passthrough metadata; surfaced to hooks and telemetry. */
  readonly metadata?: Record<string, unknown>;
}

export interface CompletionResult {
  readonly message: AssistantMessage;
  readonly finishReason: FinishReason;
  readonly usage: Usage;
  /** Correlation id assigned by the agent loop. */
  readonly responseId: string;
  readonly providerId: string;
  readonly model: string;
  readonly latencyMs: number;
  /** Raw provider payload. Useful for debugging and provider-specific fields. */
  readonly raw: unknown;
}

/** Incremental stream events. Every stream terminates with `finish` or `error`. */
export type StreamEvent =
  | { readonly type: 'start'; readonly model: string; readonly responseId: string }
  | { readonly type: 'text-delta'; readonly text: string; readonly responseId: string }
  | {
      readonly type: 'reasoning-delta';
      readonly text: string;
      readonly responseId: string;
    }
  | {
      readonly type: 'tool-call';
      readonly call: ToolCall;
      readonly responseId: string;
    }
  | { readonly type: 'finish'; readonly result: CompletionResult }
  | { readonly type: 'error'; readonly error: Error; readonly responseId: string };

export interface StreamOptions {
  readonly signal?: AbortSignal;
  readonly onEvent?: (event: StreamEvent) => void;
}

// ---------------------------------------------------------------------------
// Provider interface
// ---------------------------------------------------------------------------

/** Declares what a provider can actually do, so the agent can adapt. */
export interface ProviderCapabilities {
  readonly tools: boolean;
  readonly parallelToolCalls: boolean;
  readonly streaming: boolean;
  readonly systemMessages: boolean;
  readonly jsonMode: boolean;
  readonly strictJsonSchema: boolean;
  readonly vision: boolean;
  readonly promptCaching: boolean;
}

export interface ProviderCallOptions {
  readonly signal?: AbortSignal;
  /** Per-call deadline. Falls back to the provider default. */
  readonly timeoutMs?: number;
  readonly headers?: Readonly<Record<string, string>>;
}

/**
 * The contract every model backend implements.
 *
 * Implement this directly to add a new vendor, or extend {@link BaseProvider}
 * to inherit sensible defaults (error mapping, retries, stream fallback).
 *
 * ```ts
 * const provider: ModelProvider = {
 *   id: 'my-vendor',
 *   name: 'My Vendor',
 *   defaultModel: 'large',
 *   capabilities: { ... },
 *   complete: async (req, opts) => ({ ... }),
 *   stream: async function* () { ... },
 * };
 * ```
 */
export interface ModelProvider {
  /** Stable identifier used in config, logs, and telemetry. */
  readonly id: string;
  /** Human-readable vendor name. */
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities;
  /** One-shot completion. Must reject with an {@link AgentError} on failure. */
  complete(
    request: CompletionRequest,
    options?: ProviderCallOptions,
  ): Promise<CompletionResult>;
  /** Token-by-token completion. Must throw (not yield `error`) on failure. */
  stream(
    request: CompletionRequest,
    options?: ProviderCallOptions,
  ): AsyncIterable<StreamEvent>;
  /**
   * Optional: return a copy bound to a different default model. Used by
   * {@link createProvider} so `'anthropic:claude-…'` works uniformly.
   */
  withModel?(model: string): ModelProvider;
}
