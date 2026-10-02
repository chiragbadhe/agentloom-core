import type { SchemaIssue } from './schema.js';

/** Stable, machine-readable discriminants for every error the kit throws. */
export const ErrorCode = {
  CONFIGURATION: 'CONFIGURATION_ERROR',
  ABORTED: 'ABORTED',
  TIMEOUT: 'TIMEOUT',
  PROVIDER: 'PROVIDER_ERROR',
  PROVIDER_AUTH: 'PROVIDER_AUTH_ERROR',
  PROVIDER_RATE_LIMIT: 'PROVIDER_RATE_LIMIT_ERROR',
  PROVIDER_QUOTA: 'PROVIDER_QUOTA_ERROR',
  PROVIDER_TIMEOUT: 'PROVIDER_TIMEOUT_ERROR',
  PROVIDER_RESPONSE: 'PROVIDER_RESPONSE_ERROR',
  TOOL_NOT_FOUND: 'TOOL_NOT_FOUND_ERROR',
  TOOL_EXECUTION: 'TOOL_EXECUTION_ERROR',
  TOOL_VALIDATION: 'TOOL_VALIDATION_ERROR',
  TOOL_TIMEOUT: 'TOOL_TIMEOUT_ERROR',
  TOOL_DENIED: 'TOOL_DENIED_ERROR',
  TOOL_APPROVAL: 'TOOL_APPROVAL_ERROR',
  VALIDATION: 'VALIDATION_ERROR',
  MEMORY: 'MEMORY_ERROR',
  PLANNING: 'PLANNING_ERROR',
  EXECUTION_LIMIT: 'EXECUTION_LIMIT_ERROR',
} as const;

export type ErrorCodeValue = (typeof ErrorCode)[keyof typeof ErrorCode];

export interface AgentErrorOptions {
  readonly cause?: unknown;
  readonly retryable?: boolean;
  readonly details?: Record<string, unknown>;
  /** Milliseconds the caller should wait before retrying, if known. */
  readonly retryAfterMs?: number;
}

/**
 * Base class for every error raised by `@agentloom/core`.
 *
 * ```ts
 * try { await agent.run('hi'); }
 * catch (err) {
 *   if (err instanceof AgentError && err.code === ErrorCode.TIMEOUT) { ... }
 * }
 * ```
 */
export class AgentError extends Error {
  readonly code: ErrorCodeValue;
  readonly retryable: boolean;
  readonly details: Record<string, unknown> | undefined;
  readonly retryAfterMs: number | undefined;

  constructor(message: string, code: ErrorCodeValue, options: AgentErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = new.target.name;
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.details = options.details;
    this.retryAfterMs = options.retryAfterMs;
    Error.captureStackTrace?.(this, new.target);
  }

  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      retryAfterMs: this.retryAfterMs,
      details: this.details,
    };
  }
}

/** Invalid or contradictory configuration. Always a programming error. */
export class ConfigurationError extends AgentError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, ErrorCode.CONFIGURATION, { details });
  }
}

/** The caller cancelled the operation via `AbortController`. */
export class AbortError extends AgentError {
  constructor(message = 'Operation was aborted') {
    super(message, ErrorCode.ABORTED, { retryable: false });
  }
}

/** A wall-clock deadline elapsed. */
export class TimeoutError extends AgentError {
  readonly timeoutMs: number;

  constructor(timeoutMs: number, label = 'operation') {
    super(`${label} timed out after ${timeoutMs}ms`, ErrorCode.TIMEOUT, {
      retryable: true,
      details: { timeoutMs, label },
    });
    this.timeoutMs = timeoutMs;
  }
}

// ---------------------------------------------------------------------------
// Provider errors
// ---------------------------------------------------------------------------

export interface ProviderErrorOptions extends AgentErrorOptions {
  readonly statusCode?: number;
  readonly providerId?: string;
  readonly model?: string;
  /** Internal override used by subclasses to pin their own code. */
  readonly code?: ErrorCodeValue;
}

export function isAbortLikeError(error: unknown): boolean {
  if (error instanceof AgentError) return error.code === ErrorCode.ABORTED;
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    (error as { name?: unknown }).name === 'AbortError'
  );
}

export function isRetryableError(error: unknown): boolean {
  return error instanceof AgentError && error.retryable;
}

function httpMessage(status: number, body: string): string {
  const snippet = body.length > 400 ? `${body.slice(0, 400)}…` : body;
  return snippet ? `HTTP ${status}: ${snippet}` : `HTTP ${status}`;
}

/** A request to a model provider failed. */
export class ProviderError extends AgentError {
  readonly statusCode: number | undefined;
  readonly providerId: string | undefined;
  readonly model: string | undefined;

  constructor(message: string, options: ProviderErrorOptions = {}) {
    super(message, options.code ?? ErrorCode.PROVIDER, {
      retryable: options.retryable ?? isRetryableStatus(options.statusCode),
      cause: options.cause,
      details: options.details,
      retryAfterMs: options.retryAfterMs,
    });
    this.statusCode = options.statusCode;
    this.providerId = options.providerId;
    this.model = options.model;
  }
}

/** HTTP 401 / 403 — the API key is missing, invalid, or lacks permissions. */
export class ProviderAuthError extends ProviderError {
  constructor(message: string, options: ProviderErrorOptions = {}) {
    super(message, { code: ErrorCode.PROVIDER_AUTH, retryable: false, ...options });
  }
}

/** HTTP 429 — throttled. Retryable, usually after a short pause. */
export class ProviderRateLimitError extends ProviderError {
  constructor(message: string, options: ProviderErrorOptions = {}) {
    super(message, { code: ErrorCode.PROVIDER_RATE_LIMIT, retryable: true, ...options });
  }
}

/** HTTP 402 / 429 with an explicit exhaustion signal — billing or quota. */
export class ProviderQuotaError extends ProviderError {
  constructor(message: string, options: ProviderErrorOptions = {}) {
    super(message, { code: ErrorCode.PROVIDER_QUOTA, retryable: false, ...options });
  }
}

/** The provider accepted the request but did not respond in time. */
export class ProviderTimeoutError extends ProviderError {
  constructor(message: string, options: ProviderErrorOptions = {}) {
    super(message, { code: ErrorCode.PROVIDER_TIMEOUT, retryable: true, ...options });
  }
}

/** The provider replied with a 2xx body we could not understand. */
export class ProviderResponseError extends ProviderError {
  readonly responseBody: string | undefined;

  constructor(
    message: string,
    options: ProviderErrorOptions & { responseBody?: string } = {},
  ) {
    super(message, { code: ErrorCode.PROVIDER_RESPONSE, retryable: false, ...options });
    this.responseBody = options.responseBody;
  }
}

export function isRetryableStatus(status: number | undefined): boolean {
  if (status === undefined) return false;
  return (
    status === 408 || status === 409 || status === 425 || status === 429 || status >= 500
  );
}

/**
 * Map an HTTP response onto the most specific error class available.
 * Used by every built-in provider so behaviour is identical across vendors.
 */
export function providerErrorFromResponse(
  status: number,
  body: string,
  context: { providerId: string; model?: string; retryAfterMs?: number },
): ProviderError {
  const base = {
    statusCode: status,
    providerId: context.providerId,
    model: context.model,
    retryAfterMs: context.retryAfterMs,
  };
  const lower = body.toLowerCase();
  if (status === 401 || status === 403) {
    return new ProviderAuthError(httpMessage(status, body), base);
  }
  if (status === 429) {
    const quota =
      lower.includes('quota') || lower.includes('billing') || lower.includes('credit');
    return quota
      ? new ProviderQuotaError(httpMessage(status, body), base)
      : new ProviderRateLimitError(httpMessage(status, body), base);
  }
  if (status === 402 || (status === 429 && lower.includes('insufficient'))) {
    return new ProviderQuotaError(httpMessage(status, body), base);
  }
  return new ProviderError(httpMessage(status, body), base);
}

// ---------------------------------------------------------------------------
// Tool errors
// ---------------------------------------------------------------------------

export interface ToolErrorOptions extends AgentErrorOptions {
  readonly toolName?: string;
  readonly toolArguments?: unknown;
  readonly toolCallId?: string;
}

export class ToolError extends AgentError {
  readonly toolName: string | undefined;
  readonly toolArguments: unknown;
  readonly toolCallId: string | undefined;

  constructor(message: string, code: ErrorCodeValue, options: ToolErrorOptions = {}) {
    super(message, code, options);
    this.toolName = options.toolName;
    this.toolArguments = options.toolArguments;
    this.toolCallId = options.toolCallId;
  }
}

export class ToolNotFoundError extends ToolError {
  readonly availableTools: readonly string[];

  constructor(name: string, availableTools: readonly string[] = []) {
    super(
      `Tool "${name}" is not registered. Available: ${availableTools.join(', ') || 'none'}`,
      ErrorCode.TOOL_NOT_FOUND,
      {
        toolName: name,
        details: { availableTools },
      },
    );
    this.availableTools = availableTools;
  }
}

/** Tool arguments failed schema validation (or were not valid JSON). */
export class ToolValidationError extends ToolError {
  readonly issues: readonly SchemaIssue[];

  constructor(
    message: string,
    options: ToolErrorOptions & { issues?: readonly SchemaIssue[] } = {},
  ) {
    super(message, ErrorCode.TOOL_VALIDATION, options);
    this.issues = options.issues ?? [];
  }
}

/** The tool's `execute` threw. */
export class ToolExecutionError extends ToolError {
  constructor(message: string, options: ToolErrorOptions = {}) {
    super(message, ErrorCode.TOOL_EXECUTION, { retryable: true, ...options });
  }
}

/** The tool exceeded its own `timeoutMs`. */
export class ToolTimeoutError extends ToolError {
  constructor(message: string, options: ToolErrorOptions = {}) {
    super(message, ErrorCode.TOOL_TIMEOUT, { retryable: true, ...options });
  }
}

/** A {@link ToolPolicy} denied the call. */
export class ToolDeniedError extends ToolError {
  readonly reason: string;

  constructor(name: string, reason: string, options: ToolErrorOptions = {}) {
    super(`Tool "${name}" was denied: ${reason}`, ErrorCode.TOOL_DENIED, {
      ...options,
      toolName: name,
    });
    this.reason = reason;
  }
}

/** Human-in-the-loop approval was required but not granted. */
export class ToolApprovalRequiredError extends ToolError {
  constructor(name: string, options: ToolErrorOptions = {}) {
    super(
      `Tool "${name}" requires approval but none was provided or it was denied`,
      ErrorCode.TOOL_APPROVAL,
      { ...options, toolName: name },
    );
  }
}

// ---------------------------------------------------------------------------
// Other domains
// ---------------------------------------------------------------------------

/** Structured output did not satisfy the requested schema. */
export class ValidationError extends AgentError {
  readonly issues: readonly SchemaIssue[];

  constructor(message: string, issues: readonly SchemaIssue[] = [], cause?: unknown) {
    super(message, ErrorCode.VALIDATION, { cause, details: { issues } });
    this.issues = issues;
  }
}

/** A memory backend failed. */
export class MemoryError extends AgentError {
  constructor(message: string, cause?: unknown) {
    super(message, ErrorCode.MEMORY, { cause, retryable: true });
  }
}

/** Planning / decomposition failed. */
export class PlanningError extends AgentError {
  constructor(message: string, cause?: unknown) {
    super(message, ErrorCode.PLANNING, { cause, retryable: false });
  }
}

/** A configured execution limit (iterations, tokens, tool calls) was hit. */
export class ExecutionLimitError extends AgentError {
  readonly limit: string;
  readonly limitValue: number;

  constructor(limit: string, limitValue: number, message: string) {
    super(message, ErrorCode.EXECUTION_LIMIT, {
      retryable: false,
      details: { limit, limitValue },
    });
    this.limit = limit;
    this.limitValue = limitValue;
  }
}

/** Normalize any thrown value into an {@link AgentError}. */
export function toAgentError(
  error: unknown,
  fallbackMessage = 'Unexpected error',
): AgentError {
  if (error instanceof AgentError) return error;
  if (isAbortLikeError(error)) return new AbortError();
  if (error instanceof Error) {
    return new AgentError(error.message || fallbackMessage, ErrorCode.CONFIGURATION, {
      cause: error,
    });
  }
  return new AgentError(`${fallbackMessage}: ${String(error)}`, ErrorCode.CONFIGURATION, {
    cause: error,
  });
}
