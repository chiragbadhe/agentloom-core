import { ToolExecutionError, ToolValidationError } from '../../errors.js';
import type { Schema } from '../../schema.js';
import { truncate } from '../../utils/text.js';
import { defineTool } from '../registry.js';

export type HttpAction = 'get' | 'post' | 'head';

export interface HttpToolArgs {
  readonly url: string;
  readonly action?: HttpAction;
  readonly headers?: Readonly<Record<string, string>>;
  /** Body for `post`; objects are JSON-encoded. */
  readonly body?: unknown;
  readonly format?: 'text' | 'json' | 'auto';
}

export interface HttpToolResult {
  readonly status: number;
  readonly ok: boolean;
  readonly url: string;
  readonly contentType: string | undefined;
  readonly body: string;
  readonly truncated: boolean;
}

export interface HttpToolOptions {
  /**
   * Hosts the tool may contact. Exact host or `*.suffix` pattern. When
   * omitted, **all** hosts are allowed — only do that in a trusted sandbox.
   */
  readonly allowedHosts?: readonly string[];
  readonly blockedHosts?: readonly string[];
  readonly maxBytes?: number;
  readonly timeoutMs?: number;
  readonly userAgent?: string;
  readonly fetch?: typeof globalThis.fetch;
  readonly maxResponseChars?: number;
}

const PRIVATE_HOST_PATTERN =
  /^(localhost|127\.|0\.0\.0\.0|\[::1\]|169\.254\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/i;

/**
 * Decide whether a URL is allowed, applying host patterns and (optionally)
 * blocking private/loopback ranges to prevent SSRF against internal services.
 */
export function isUrlAllowed(
  rawUrl: string,
  options: Pick<HttpToolOptions, 'allowedHosts' | 'blockedHosts'> & {
    blockPrivateHosts?: boolean;
  },
): { allowed: true } | { allowed: false; reason: string } {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return { allowed: false, reason: `"${rawUrl}" is not a valid URL` };
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return { allowed: false, reason: `protocol "${url.protocol}" is not allowed` };
  }

  const host = url.hostname.toLowerCase();

  if (options.blockedHosts?.some((blocked) => matchesHost(host, blocked))) {
    return { allowed: false, reason: `host "${host}" is blocked` };
  }

  if (options.blockPrivateHosts !== false && PRIVATE_HOST_PATTERN.test(host)) {
    return { allowed: false, reason: `host "${host}" resolves to a private network` };
  }

  if (options.allowedHosts !== undefined && options.allowedHosts.length > 0) {
    const permitted = options.allowedHosts.some((pattern) => matchesHost(host, pattern));
    if (!permitted) {
      return {
        allowed: false,
        reason: `host "${host}" is not in the allowlist (${options.allowedHosts.join(', ')})`,
      };
    }
  }

  return { allowed: true };
}

function matchesHost(host: string, pattern: string): boolean {
  const normalized = pattern.toLowerCase().trim();
  if (normalized.startsWith('*.')) {
    const suffix = normalized.slice(1);
    return host.endsWith(suffix) || host === normalized.slice(2);
  }
  return host === normalized;
}

/**
 * HTTP tool for fetching pages and APIs.
 *
 * Security defaults: private/loopback hosts are refused, responses are size
 * capped, and the body is truncated before it reaches the model. Configure
 * `allowedHosts` for anything production-facing.
 */
export function createHttpTool(options: HttpToolOptions = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new ToolExecutionError('global fetch is unavailable; pass `fetch` explicitly');
  }
  const maxBytes = options.maxBytes ?? 1_000_000;
  const timeoutMs = options.timeoutMs ?? 15_000;
  const maxChars = options.maxResponseChars ?? 20_000;

  const parameterSchema: Schema<HttpToolArgs> = {
    safeParse: (input: unknown) => {
      const issues: { path: (string | number)[]; message: string }[] = [];
      if (typeof input !== 'object' || input === null) {
        return {
          success: false,
          error: { issues: [{ path: [], message: 'expected an object' }] },
        };
      }
      const args = input as Record<string, unknown>;
      if (typeof args.url !== 'string' || args.url.trim() === '') {
        issues.push({ path: ['url'], message: 'required' });
      } else {
        const verdict = isUrlAllowed(args.url, options);
        if (!verdict.allowed) issues.push({ path: ['url'], message: verdict.reason });
      }
      const HTTP_ACTIONS: readonly string[] = ['get', 'post', 'head'];
      if (
        args.action !== undefined &&
        (typeof args.action !== 'string' || !HTTP_ACTIONS.includes(args.action))
      ) {
        issues.push({ path: ['action'], message: 'must be get, post, or head' });
      }
      if (
        args.headers !== undefined &&
        (typeof args.headers !== 'object' || args.headers === null)
      ) {
        issues.push({ path: ['headers'], message: 'must be an object' });
      }
      return issues.length === 0
        ? { success: true, data: input as HttpToolArgs }
        : { success: false, error: { issues } };
    },
    parse: (input: unknown) => {
      const result = parameterSchema.safeParse(input);
      if (!result.success)
        throw new ToolValidationError('Invalid http_request arguments');
      return result.data;
    },
  };

  return defineTool<HttpToolArgs, HttpToolResult>({
    name: 'http_request',
    description:
      'Fetch a URL over HTTP(S) and return the response body. Use this to read web ' +
      'pages, documentation, and public APIs. Private and loopback addresses are ' +
      'blocked. Prefer "auto" format to get JSON parsed automatically.',
    parameters: parameterSchema,
    jsonSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'Absolute http(s) URL' },
        action: { type: 'string', enum: ['get', 'post', 'head'] },
        headers: { type: 'object', additionalProperties: { type: 'string' } },
        body: { description: 'Request body; objects are sent as JSON' },
        format: { type: 'string', enum: ['text', 'json', 'auto'] },
      },
      required: ['url'],
    },
    timeoutMs: timeoutMs + 1_000,
    async execute({ url, action = 'get', headers, body, format = 'auto' }, context) {
      // Re-check here: validation already covers this, but `execute` is also
      // reachable directly, and the allow-list is the SSRF guard.
      const verdict = isUrlAllowed(url, options);
      if (!verdict.allowed) {
        throw new ToolExecutionError(`Refusing to fetch ${url}: ${verdict.reason}`, {
          toolName: 'http_request',
        });
      }

      const controller = new AbortController();
      const onAbort = () => controller.abort();
      context.signal.addEventListener('abort', onAbort, { once: true });
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const isPost = action === 'post';
        const payload =
          body === undefined
            ? undefined
            : typeof body === 'string'
              ? body
              : JSON.stringify(body);

        const response = await fetchImpl(url, {
          method: action.toUpperCase(),
          headers: {
            accept:
              format === 'text'
                ? 'text/plain, text/html;q=0.9'
                : 'application/json, text/plain;q=0.9',
            'user-agent': options.userAgent ?? '@agentloom/core',
            ...(isPost ? { 'content-type': 'application/json' } : {}),
            ...headers,
          },
          ...(payload === undefined ? {} : { body: payload }),
          signal: controller.signal,
        });

        const reader = response.body;
        const text =
          action === 'head'
            ? ''
            : reader === null
              ? await response.text()
              : await readCapped(reader, maxBytes);

        const contentType = response.headers.get('content-type') ?? undefined;
        const formatted =
          format === 'text'
            ? text
            : format === 'json'
              ? safePretty(text)
              : contentType?.includes('json') === true
                ? safePretty(text)
                : text;

        const truncated = formatted.length > maxChars;
        return {
          status: response.status,
          ok: response.ok,
          url,
          contentType,
          body: truncated
            ? `${truncate(formatted, maxChars)}\n[truncated after ${maxChars} characters]`
            : formatted,
          truncated,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new ToolExecutionError(`Request to ${url} failed: ${message}`, {
          toolName: 'http_request',
        });
      } finally {
        clearTimeout(timer);
        context.signal.removeEventListener('abort', onAbort);
      }
    },
  });
}

async function readCapped(
  body: ReadableStream<Uint8Array>,
  maxBytes: number,
): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let out = '';
  let total = 0;

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      out += decoder.decode(value, { stream: true });
      if (total >= maxBytes) break;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
  return out + decoder.decode();
}

function safePretty(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/** Convenience wrapper returning just the text body of a URL. */
export function createFetchUrlTool(options: HttpToolOptions = {}) {
  const http = createHttpTool(options);
  return defineTool<{ url: string; format?: 'text' | 'json' }, string>({
    name: 'fetch_url',
    description: 'Fetch a URL and return its content as text or parsed JSON.',
    parameters: http.parameters as unknown as Schema<{
      url: string;
      format?: 'text' | 'json';
    }>,
    jsonSchema: {
      type: 'object',
      properties: {
        url: { type: 'string' },
        format: { type: 'string', enum: ['text', 'json'] },
      },
      required: ['url'],
    },
    timeoutMs: (http.timeoutMs ?? 16_000) + 1_000,
    async execute(args, context) {
      const result = await http.execute(
        { url: args.url, action: 'get', format: args.format ?? 'auto' },
        context,
      );
      return result.body;
    },
  });
}
