import type { ModelMessage } from '../providers/types.js';

/**
 * Short-term conversation memory: the ordered message list that gets sent to
 * the model on every iteration.
 *
 * Implement this interface to persist a conversation (Redis, Postgres, a file)
 * or to swap in a different windowing strategy. A minimal implementation needs
 * `add`, `messages`, and `clear`.
 */
export interface ConversationMemory {
  /** Append a message. */
  add(message: ModelMessage): void | Promise<void>;
  /** Append several messages. */
  addAll(messages: readonly ModelMessage[]): void | Promise<void>;
  /** Current messages, oldest first. */
  messages(): readonly ModelMessage[] | Promise<readonly ModelMessage[]>;
  /** Drop all messages. */
  clear(): void | Promise<void>;
  readonly length: number;
  /**
   * Produce the message list for the model, applying the token budget and
   * trimming strategy. `system` is prepended when provided.
   */
  build(options: BuildContext): Promise<readonly ModelMessage[]>;
}

/** What a memory implementation needs to know to assemble a prompt. */
export interface BuildContext {
  /** System prompt for this turn. Omit to use only stored messages. */
  readonly system?: string | undefined;
  /** Hard ceiling on the assembled prompt, in tokens. */
  readonly maxTokens: number;
  readonly tokenCounter: TokenCounter;
  /** Number of most recent messages that must never be trimmed. */
  readonly keepRecent: number;
  /** Aborts compaction (summarization) when the run is cancelled. */
  readonly signal?: AbortSignal | undefined;
}

/** Anything a token counter needs to measure. */
export type TokenCountable = string | ModelMessage | readonly ModelMessage[];

export type TokenCounter = (input: TokenCountable) => number;

/** Context injected by a memory when trimming or summarizing. */
export interface TrimContext {
  readonly signal?: AbortSignal | undefined;
  /** Estimated tokens of the messages about to be dropped. */
  readonly droppedTokens: number;
}

/**
 * Produces a condensed stand-in for messages that fall out of the window.
 * Called only when there is something to drop.
 */
export type MemoryCompactor = (
  messages: readonly ModelMessage[],
  context: TrimContext,
) => Promise<ModelMessage[] | undefined> | ModelMessage[] | undefined;

/** How the window is reduced when it exceeds the token budget. */
export type MemoryStrategy =
  /** Drop the oldest messages. Fast, cheap, loses detail. */
  | 'sliding'
  /**
   * Ask a model to summarize the oldest messages, then keep the summary plus
   * the recent window. Preserves detail, costs an extra call.
   */
  | 'summarize';

export interface ConversationMemoryOptions {
  /** Soft ceiling on stored messages. Default `100`. */
  readonly maxMessages?: number;
  /** Token budget for `build()`. Default `8000`. */
  readonly maxTokens?: number;
  /** Recent messages protected from trimming. Default `4`. */
  readonly keepRecent?: number;
  readonly strategy?: MemoryStrategy;
  /** Required when `strategy` is `summarize`. */
  readonly compact?: MemoryCompactor;
  readonly tokenCounter?: TokenCounter;
  /** Seed the store with existing history. */
  readonly initialMessages?: readonly ModelMessage[];
}
