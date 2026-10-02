/**
 * Token accounting helpers.
 *
 * Accurate counts require a model-specific tokenizer, which would add a heavy
 * dependency. Instead we ship a conservative heuristic that is *calibrated* to
 * overestimate slightly: overshooting a budget is safe, undershooting it is
 * not. Pass your own `tokenCounter` to {@link AgentConfig} for exact counts.
 */

/** Average characters per token, by script. */
const LATIN_CHARS_PER_TOKEN = 4;
const CJK_CHARS_PER_TOKEN = 1.5;

/** Fixed per-message overhead imposed by chat templates. */
const MESSAGE_OVERHEAD_TOKENS = 4;

/**
 * Heuristic token count for a string. Handles CJK text, which is far denser
 * per character than latin text, and never returns less than 1 for
 * non-empty input.
 */
export function estimateTokens(text: string): number {
  if (text.length === 0) return 0;
  let cjk = 0;
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0;
    if (
      (code >= 0x3040 && code <= 0x30ff) ||
      (code >= 0x3400 && code <= 0x4dbf) ||
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0xac00 && code <= 0xd7af) ||
      (code >= 0xf900 && code <= 0xfaff)
    ) {
      cjk++;
    }
  }
  const other = text.length - cjk;
  return Math.max(
    1,
    Math.ceil(cjk / CJK_CHARS_PER_TOKEN + other / LATIN_CHARS_PER_TOKEN),
  );
}

/**
 * Estimate tokens for a JSON-ish value (used for tool arguments). Strings are
 * counted as text; everything else is counted via `JSON.stringify`.
 */
export function estimateValueTokens(value: unknown): number {
  if (value === undefined) return 0;
  if (typeof value === 'string') return estimateTokens(value);
  try {
    return estimateTokens(JSON.stringify(value) ?? '');
  } catch {
    return estimateTokens(Object.prototype.toString.call(value));
  }
}

/** Structural minimum of a single chat message, used for context budgeting. */
export interface TokenCountableMessage {
  readonly role: string;
  readonly content: string;
  readonly toolCalls?: readonly unknown[];
  readonly toolResult?: { readonly content: string; readonly name?: string };
}

/**
 * Estimate the prompt cost of a message list, including per-message template
 * overhead and tool-call/result payloads.
 */
export function estimateMessageTokens(
  messages: readonly TokenCountableMessage[],
): number {
  let total = 0;
  for (const message of messages) {
    total += MESSAGE_OVERHEAD_TOKENS;
    total += estimateTokens(message.content);
    if (message.toolCalls) {
      for (const call of message.toolCalls) total += estimateValueTokens(call) + 4;
    }
    if (message.toolResult) {
      total += estimateTokens(message.toolResult.content) + 4;
    }
  }
  return total;
}
