import { MemoryError } from '../errors.js';
import type {
  AssistantMessage,
  ModelMessage,
  SystemMessage,
  ToolMessage,
} from '../providers/types.js';
import { estimateMessageTokens, estimateTokens } from '../utils/tokens.js';
import { createId } from '../utils/id.js';
import type {
  BuildContext,
  ConversationMemory,
  ConversationMemoryOptions,
  MemoryCompactor,
  MemoryStrategy,
  TokenCounter,
} from './types.js';

const DEFAULT_MAX_MESSAGES = 100;
const DEFAULT_MAX_TOKENS = 8_000;
const DEFAULT_KEEP_RECENT = 4;

/** Default estimator: structural token count for messages, char count for strings. */
export const defaultTokenCounter: TokenCounter = (input) => {
  if (typeof input === 'string') return estimateTokens(input);
  if (Array.isArray(input)) return estimateMessageTokens(input);
  return estimateMessageTokens([input as ModelMessage]);
};

export interface ConversationMemoryState {
  readonly messages: readonly ModelMessage[];
  readonly droppedCount: number;
  readonly summaryCount: number;
  readonly tokens: number;
  readonly trimmed: boolean;
}

/**
 * Default in-memory conversation memory with token-budgeted windowing.
 *
 * `build()` respects `maxTokens`, except when the protected recent tail alone
 * exceeds it — trimming further would corrupt tool call/result pairing, so the
 * tail is kept intact.
 *
 * ```ts
 * const memory = new InMemoryConversationMemory({ maxTokens: 16_000, keepRecent: 6 });
 * memory.add({ role: 'user', content: 'hi' });
 * const prompt = await memory.build({ maxTokens: 16_000, keepRecent: 6 });
 * ```
 */
export class InMemoryConversationMemory implements ConversationMemory {
  private store: ModelMessage[] = [];
  private droppedCount = 0;
  private summaryCount = 0;
  private readonly maxMessages: number;
  private readonly maxTokens: number;
  private readonly keepRecent: number;
  private readonly strategy: MemoryStrategy;
  private readonly compact: MemoryCompactor | undefined;
  private readonly tokenCounter: TokenCounter;

  constructor(options: ConversationMemoryOptions = {}) {
    this.maxMessages = options.maxMessages ?? DEFAULT_MAX_MESSAGES;
    this.maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    this.keepRecent = Math.max(0, options.keepRecent ?? DEFAULT_KEEP_RECENT);
    this.strategy = options.strategy ?? 'sliding';
    this.compact = options.compact;
    this.tokenCounter = options.tokenCounter ?? defaultTokenCounter;
    if (options.initialMessages) this.store = [...options.initialMessages];
    this.enforceMessageCap();
  }

  get length(): number {
    return this.store.length;
  }

  /** Configuration in effect. */
  get limits(): { maxTokens: number; keepRecent: number; strategy: MemoryStrategy } {
    return {
      maxTokens: this.maxTokens,
      keepRecent: this.keepRecent,
      strategy: this.strategy,
    };
  }

  add(message: ModelMessage): void {
    this.store.push(message);
    this.enforceMessageCap();
  }

  addAll(messages: readonly ModelMessage[]): void {
    for (const message of messages) this.add(message);
  }

  messages(): readonly ModelMessage[] {
    return this.store;
  }

  clear(): void {
    this.store = [];
    this.droppedCount = 0;
    this.summaryCount = 0;
  }

  /** Estimated tokens currently held. */
  tokenCount(): number {
    return this.tokenCounter(this.store);
  }

  /** Diagnostics for debugging and for the `memory` lifecycle hook. */
  state(): ConversationMemoryState {
    return {
      messages: this.store,
      droppedCount: this.droppedCount,
      summaryCount: this.summaryCount,
      tokens: this.tokenCount(),
      trimmed: this.droppedCount > 0,
    };
  }

  async build(context: BuildContext): Promise<readonly ModelMessage[]> {
    const keepRecent = Math.max(0, context.keepRecent ?? this.keepRecent);
    const budget = context.maxTokens ?? this.maxTokens;
    const counter = context.tokenCounter ?? this.tokenCounter;

    const system: ModelMessage[] =
      context.system !== undefined && context.system !== ''
        ? [{ role: 'system', content: context.system } satisfies SystemMessage]
        : [];

    // Reserve room for the system prompt before deciding what to drop.
    let window = [...this.store];
    let total = counter(system) + counter(window);

    // Messages inside the protected recent tail are never trimmed, even if
    // that means overshooting the budget — a half-trimmed tool exchange is
    // worse than a slightly over-long prompt.
    const protectFrom = Math.max(0, window.length - keepRecent);
    let cutIndex = 0;

    while (total > budget && cutIndex < protectFrom) {
      total -= counter(window[cutIndex]!);
      cutIndex++;
    }

    if (cutIndex === 0) return [...system, ...window];

    const dropped = window.slice(0, cutIndex);
    window = window.slice(cutIndex);
    this.droppedCount += dropped.length;

    if (this.strategy === 'summarize' && this.compact !== undefined) {
      const compacted = await this.applyCompactor(dropped, context.signal);
      if (compacted !== undefined) {
        window = [...compacted, ...window];
        this.summaryCount++;
      }
    }

    return [...system, ...window];
  }

  private async applyCompactor(
    dropped: readonly ModelMessage[],
    signal: AbortSignal | undefined,
  ): Promise<readonly ModelMessage[] | undefined> {
    if (this.compact === undefined) return undefined;
    try {
      const result = await this.compact(dropped, {
        signal,
        droppedTokens: this.tokenCounter(dropped),
      });
      // An empty compaction would silently discard history, so treat it as a
      // no-op and keep the plain sliding window instead.
      if (result === undefined || result.length === 0) return undefined;
      return result;
    } catch (error) {
      throw new MemoryError(
        `Memory compaction failed: ${error instanceof Error ? error.message : String(error)}`,
        error,
      );
    }
  }

  private enforceMessageCap(): void {
    if (this.store.length <= this.maxMessages) return;
    const overflow = this.store.length - this.maxMessages;
    const keep = Math.max(this.keepRecent, 1);
    const removable = Math.min(overflow, Math.max(0, this.store.length - keep));
    if (removable <= 0) return;
    this.store = this.store.slice(removable);
    this.droppedCount += removable;
  }
}

/**
 * Turns a summarizer prompt into a single system message holding the digest.
 * Pair with {@link createSummaryCompactor} when using `strategy: 'summarize'`.
 */
export function summaryMessage(text: string): SystemMessage {
  return {
    role: 'system',
    content: `Summary of the earlier conversation:\n${text.trim()}`,
  };
}

/** Render messages as a transcript for a summarizer prompt. */
export function transcriptOf(messages: readonly ModelMessage[]): string {
  return messages
    .map((message) => {
      switch (message.role) {
        case 'system':
          return `system: ${message.content}`;
        case 'user':
          return `user: ${message.content}`;
        case 'assistant':
          return `assistant: ${message.content}${renderToolCalls(message)}`;
        case 'tool':
          return `tool(${message.name}): ${message.content}`;
      }
    })
    .join('\n');
}

function renderToolCalls(message: AssistantMessage): string {
  if (message.toolCalls === undefined || message.toolCalls.length === 0) return '';
  const calls = message.toolCalls
    .map((call) => `[called ${call.name}(${safeJson(call.arguments)})]`)
    .join(' ');
  return ` ${calls}`;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

/**
 * Build a {@link MemoryCompactor} from a "summarize with a model" callback.
 *
 * ```ts
 * const memory = new InMemoryConversationMemory({
 *   strategy: 'summarize',
 *   compact: createSummaryCompactor(async (prompt) => {
 *     const res = await provider.complete({ model, messages: [{ role: 'user', content: prompt }] });
 *     return res.message.content;
 *   }),
 * });
 * ```
 */
export function createSummaryCompactor(
  summarize: (prompt: string, signal?: AbortSignal) => Promise<string>,
): MemoryCompactor {
  return async (dropped, context) => {
    const prompt = [
      'Condense the following conversation excerpt into a factual summary.',
      'Preserve decisions, facts, identifiers, and open questions. Drop pleasantries.',
      'Reply with the summary only.',
      '',
      transcriptOf(dropped),
    ].join('\n');

    const text = await summarize(prompt, context.signal);
    if (text.trim() === '') return undefined;
    return [{ ...summaryMessage(text), id: createId('sum') } as SystemMessage];
  };
}

/**
 * Message-level helper for appending an assistant turn with tool calls.
 * Exported because agent runs frequently need it when building custom loops.
 */
export function assistantMessage(
  content: string,
  toolCalls?: AssistantMessage['toolCalls'],
): AssistantMessage {
  return toolCalls === undefined || toolCalls.length === 0
    ? { role: 'assistant', content }
    : { role: 'assistant', content, toolCalls };
}

/** Helper for appending a tool result turn. */
export function toolResultMessage(
  toolCallId: string,
  name: string,
  content: string,
  isError?: boolean,
): ToolMessage {
  return {
    role: 'tool',
    content,
    toolCallId,
    name,
    ...(isError ? { isError: true } : {}),
  };
}
