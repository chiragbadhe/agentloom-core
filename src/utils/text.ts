/** Anything that is not a letter or a digit becomes a separator. */
const SLUG_STRIP = /[^A-Za-z0-9]+/g;

/** `listDirectory` -> `list_directory` */
export function toSnakeCase(input: string): string {
  return input
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(SLUG_STRIP, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase();
}

/** `list_directory` -> `List Directory` */
export function toTitleCase(input: string): string {
  return toSnakeCase(input)
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

/** Lowercase, punctuation-free text for keyword matching. */
export function normalizeText(input: string): string {
  return input.toLowerCase().replace(SLUG_STRIP, ' ').trim();
}

/** Split into unique, lowercase keyword tokens of length >= 2. */
export function tokenize(input: string): string[] {
  const normalized = normalizeText(input);
  if (!normalized) return [];
  const seen = new Set<string>();
  for (const token of normalized.split(' ')) {
    if (token.length >= 2) seen.add(token);
  }
  return [...seen];
}

/** Collapse whitespace and hard-truncate with an explicit marker. */
export function truncate(input: string, maxLength: number, marker = '…'): string {
  if (maxLength <= 0) return '';
  if (input.length <= maxLength) return input;
  if (maxLength <= marker.length) return input.slice(0, maxLength);
  return input.slice(0, maxLength - marker.length) + marker;
}

/**
 * Pull a JSON value out of model output.
 *
 * Handles bare JSON, fenced code blocks (```json ... ```), and JSON with
 * surrounding prose — all common failure modes for structured output.
 * Returns `undefined` when nothing parses.
 */
export function extractJson(input: string): unknown {
  const candidates = jsonCandidates(input);
  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      continue;
    }
  }
  return undefined;
}

function jsonCandidates(input: string): string[] {
  const trimmed = input.trim();
  const out: string[] = [trimmed];

  const fence = /```(?:json|JSON)?\s*([\s\S]*?)```/.exec(trimmed);
  if (fence?.[1]) out.push(fence[1].trim());

  for (const [open, close] of [
    ['{', '}'],
    ['[', ']'],
  ] as const) {
    const start = trimmed.indexOf(open);
    const end = trimmed.lastIndexOf(close);
    if (start !== -1 && end > start) out.push(trimmed.slice(start, end + 1));
  }

  return out.filter((candidate) => candidate.length > 0);
}

/**
 * Render an unknown value as text for a tool result payload.
 *
 * The `typeof` switch is exhaustive on purpose: it keeps objects (JSON) apart
 * from the primitives that would otherwise stringify to `"[object Object]"`.
 */
export function stringifyToolResult(value: unknown): string {
  switch (typeof value) {
    case 'string':
      return value;
    case 'number':
    case 'boolean':
      return String(value);
    case 'undefined':
      return '';
    case 'bigint':
      return `${value.toString()}n`;
    case 'symbol':
      return value.toString();
    case 'function':
      return `[function ${value.name === '' ? 'anonymous' : value.name}]`;
    case 'object': {
      if (value === null) return 'null';
      try {
        return JSON.stringify(value, null, 2) ?? 'null';
      } catch {
        // Circular structures and exotic objects (a throwing getter, a
        // `Proxy` with a hostile `ownKeys`) land here.
        return Object.prototype.toString.call(value);
      }
    }
    default:
      return '';
  }
}

/** Remove nullish/empty entries so prompt builders stay tidy. */
export function compact<T extends Record<string, unknown>>(input: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined && value !== null && value !== '') out[key] = value;
  }
  return out as Partial<T>;
}
