import { describe, expect, it } from 'vitest';

import {
  AbortError,
  AgentError,
  ConfigurationError,
  ErrorCode,
  ExecutionLimitError,
  isAbortLikeError,
  isRetryableError,
  isRetryableStatus,
  MemoryError,
  PlanningError,
  ProviderAuthError,
  ProviderError,
  providerErrorFromResponse,
  ProviderQuotaError,
  ProviderRateLimitError,
  ProviderResponseError,
  ProviderTimeoutError,
  TimeoutError,
  toAgentError,
  ToolApprovalRequiredError,
  ToolDeniedError,
  ToolExecutionError,
  ToolNotFoundError,
  ToolTimeoutError,
  ToolValidationError,
  ValidationError,
} from '../src/errors.js';

describe('AgentError', () => {
  it('carries a code, message, and details', () => {
    const error = new AgentError('boom', ErrorCode.CONFIGURATION, {
      details: { extra: 1 },
    });
    expect(error.name).toBe('AgentError');
    expect(error.code).toBe(ErrorCode.CONFIGURATION);
    expect(error.message).toBe('boom');
    expect(error.details).toEqual({ extra: 1 });
    expect(error).toBeInstanceOf(Error);
  });

  it('defaults to non-retryable and reports an explicit flag', () => {
    expect(new AgentError('x', ErrorCode.CONFIGURATION).retryable).toBe(false);
    expect(new AgentError('x', ErrorCode.PROVIDER, { retryable: true }).retryable).toBe(
      true,
    );
  });

  it('serializes to a plain object for logging', () => {
    const json = new AgentError('lost', ErrorCode.MEMORY, {
      retryable: true,
      retryAfterMs: 500,
    }).toJSON();
    expect(json).toMatchObject({
      name: 'AgentError',
      code: ErrorCode.MEMORY,
      message: 'lost',
      retryable: true,
      retryAfterMs: 500,
    });
  });

  it('keeps the original error as `cause`', () => {
    const cause = new Error('root');
    expect(new AgentError('lost', ErrorCode.MEMORY, { cause }).cause).toBe(cause);
  });
});

describe('error subclasses', () => {
  it.each([
    ['ConfigurationError', ConfigurationError, ErrorCode.CONFIGURATION],
    ['ValidationError', ValidationError, ErrorCode.VALIDATION],
    ['MemoryError', MemoryError, ErrorCode.MEMORY],
    ['PlanningError', PlanningError, ErrorCode.PLANNING],
    ['AbortError', AbortError, ErrorCode.ABORTED],
    ['TimeoutError', TimeoutError, ErrorCode.TIMEOUT],
    ['ExecutionLimitError', ExecutionLimitError, ErrorCode.EXECUTION_LIMIT],
    ['ProviderError', ProviderError, ErrorCode.PROVIDER],
    ['ProviderAuthError', ProviderAuthError, ErrorCode.PROVIDER_AUTH],
    ['ProviderRateLimitError', ProviderRateLimitError, ErrorCode.PROVIDER_RATE_LIMIT],
    ['ProviderQuotaError', ProviderQuotaError, ErrorCode.PROVIDER_QUOTA],
    ['ProviderTimeoutError', ProviderTimeoutError, ErrorCode.PROVIDER_TIMEOUT],
    ['ProviderResponseError', ProviderResponseError, ErrorCode.PROVIDER_RESPONSE],
  ])('%s reports %s and stays an AgentError', (_name, Ctor, code) => {
    const error = new (Ctor as new (message: string) => AgentError)('x');
    expect(error.code).toBe(code);
    expect(error).toBeInstanceOf(AgentError);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe(_name);
  });

  it('names each subclass after its constructor', () => {
    expect(new MemoryError('x').name).toBe('MemoryError');
    expect(new ProviderAuthError('x').name).toBe('ProviderAuthError');
  });
});

describe('ConfigurationError', () => {
  it('is not retryable — a bad config never fixes itself', () => {
    expect(new ConfigurationError('missing apiKey').retryable).toBe(false);
  });
});

describe('TimeoutError', () => {
  it('reports the deadline and is retryable', () => {
    const error = new TimeoutError(1_500, 'model call');
    expect(error.timeoutMs).toBe(1_500);
    expect(error.message).toContain('model call');
    expect(error.message).toContain('1500ms');
    expect(error.retryable).toBe(true);
  });
});

describe('provider errors', () => {
  it('derives retryability from the status code', () => {
    expect(new ProviderError('x', { statusCode: 503 }).retryable).toBe(true);
    expect(new ProviderError('x', { statusCode: 400 }).retryable).toBe(false);
  });

  it('an explicit retryable flag wins over the status code', () => {
    expect(new ProviderError('x', { statusCode: 503, retryable: false }).retryable).toBe(
      false,
    );
  });

  it('records status, provider, and model', () => {
    const error = new ProviderError('x', {
      statusCode: 500,
      providerId: 'openai',
      model: 'gpt-4o-mini',
    });
    expect(error.statusCode).toBe(500);
    expect(error.providerId).toBe('openai');
    expect(error.model).toBe('gpt-4o-mini');
  });

  it('pins codes and retryability per subclass', () => {
    expect(new ProviderAuthError('x').retryable).toBe(false);
    expect(new ProviderRateLimitError('x').retryable).toBe(true);
    expect(new ProviderQuotaError('x').retryable).toBe(false);
    expect(new ProviderTimeoutError('x').retryable).toBe(true);
    expect(new ProviderResponseError('x').retryable).toBe(false);
  });
});

describe('tool errors', () => {
  it('ToolNotFoundError lists the alternatives', () => {
    const error = new ToolNotFoundError('search', ['search', 'fetch']);
    expect(error.availableTools).toEqual(['search', 'fetch']);
    expect(error.message).toContain('search, fetch');
    expect(error.toolName).toBe('search');
  });

  it('ToolValidationError keeps the schema issues', () => {
    const error = new ToolValidationError('bad args', {
      issues: [{ path: ['url'], message: 'required' }],
    });
    expect(error.issues).toEqual([{ path: ['url'], message: 'required' }]);
  });

  it('ToolExecutionError and ToolTimeoutError are retryable', () => {
    expect(new ToolExecutionError('x').retryable).toBe(true);
    expect(new ToolTimeoutError('x').retryable).toBe(true);
  });

  it('ToolDeniedError explains the denial', () => {
    const error = new ToolDeniedError('delete_file', 'read-only session');
    expect(error.reason).toBe('read-only session');
    expect(error.message).toContain('read-only session');
    expect(error.retryable).toBe(false);
  });

  it('ToolApprovalRequiredError names the tool', () => {
    const error = new ToolApprovalRequiredError('send_email');
    expect(error.toolName).toBe('send_email');
    expect(error.code).toBe(ErrorCode.TOOL_APPROVAL);
  });
});

describe('ValidationError', () => {
  it('exposes the issues and reports them in details', () => {
    const issues = [{ path: ['name'], message: 'required' }];
    const error = new ValidationError('bad output', issues);
    expect(error.issues).toEqual(issues);
    expect(error.details).toEqual({ issues });
  });
});

describe('ExecutionLimitError', () => {
  it('names the limit that was hit', () => {
    const error = new ExecutionLimitError('maxIterations', 10, 'stopped at 10');
    expect(error.limit).toBe('maxIterations');
    expect(error.limitValue).toBe(10);
    expect(error.retryable).toBe(false);
  });
});

describe('isAbortLikeError', () => {
  it('detects AbortError, its name, and DOMException aborts', () => {
    expect(isAbortLikeError(new AbortError())).toBe(true);
    expect(isAbortLikeError(new DOMException('aborted', 'AbortError'))).toBe(true);
    expect(isAbortLikeError({ name: 'AbortError' })).toBe(true);
    expect(isAbortLikeError(new Error('nope'))).toBe(false);
    expect(isAbortLikeError(undefined)).toBe(false);
  });

  it('does not treat a TimeoutError as an abort', () => {
    expect(isAbortLikeError(new TimeoutError(1))).toBe(false);
  });
});

describe('isRetryableStatus', () => {
  it.each([408, 409, 425, 429, 500, 502, 503, 504])('retries %i', (status) => {
    expect(isRetryableStatus(status)).toBe(true);
  });

  it.each([400, 401, 403, 404, 422])('does not retry %i', (status) => {
    expect(isRetryableStatus(status)).toBe(false);
  });

  it('treats an unknown status as non-retryable', () => {
    expect(isRetryableStatus(undefined)).toBe(false);
  });
});

describe('isRetryableError', () => {
  it('follows the explicit retryable flag', () => {
    expect(isRetryableError(new ProviderError('flaky', { retryable: true }))).toBe(true);
    expect(isRetryableError(new ProviderError('fatal', { retryable: false }))).toBe(
      false,
    );
  });

  it('maps transient classes to retryable', () => {
    expect(isRetryableError(new ProviderRateLimitError('slow down'))).toBe(true);
    expect(isRetryableError(new ProviderTimeoutError('too slow'))).toBe(true);
    expect(isRetryableError(new ProviderAuthError('bad key'))).toBe(false);
  });

  it('never retries aborts', () => {
    expect(isRetryableError(new AbortError('user cancelled'))).toBe(false);
  });

  it('is false for non-AgentError values', () => {
    expect(isRetryableError(new Error('boom'))).toBe(false);
    expect(isRetryableError(undefined)).toBe(false);
  });
});

describe('providerErrorFromResponse', () => {
  const ctx = { providerId: 'openai', model: 'gpt-4o-mini' };

  it('maps 401/403 to an auth error', () => {
    for (const status of [401, 403]) {
      const error = providerErrorFromResponse(status, 'unauthorized', ctx);
      expect(error).toBeInstanceOf(ProviderAuthError);
      expect(error.retryable).toBe(false);
      expect(error.providerId).toBe('openai');
    }
  });

  it('maps a plain 429 to a retryable rate-limit error', () => {
    const error = providerErrorFromResponse(429, 'slow down', {
      ...ctx,
      retryAfterMs: 3_000,
    });
    expect(error).toBeInstanceOf(ProviderRateLimitError);
    expect(error.retryable).toBe(true);
    expect(error.retryAfterMs).toBe(3_000);
  });

  it('maps 429 with quota wording to a non-retryable quota error', () => {
    const error = providerErrorFromResponse(429, 'You exceeded your quota', ctx);
    expect(error).toBeInstanceOf(ProviderQuotaError);
    expect(error.retryable).toBe(false);
  });

  it('maps 402 to a quota error', () => {
    expect(providerErrorFromResponse(402, 'payment required', ctx)).toBeInstanceOf(
      ProviderQuotaError,
    );
  });

  it('maps 5xx to a retryable provider error', () => {
    const error = providerErrorFromResponse(500, 'boom', ctx);
    expect(error).toBeInstanceOf(ProviderError);
    expect(error.retryable).toBe(true);
    expect(error.statusCode).toBe(500);
  });

  it('maps 4xx to a non-retryable provider error', () => {
    const error = providerErrorFromResponse(400, 'bad request', ctx);
    expect(error.retryable).toBe(false);
  });

  it('includes the response body and truncates long ones', () => {
    expect(providerErrorFromResponse(422, 'invalid', ctx).message).toContain('invalid');
    const long = 'x'.repeat(1_000);
    const error = providerErrorFromResponse(422, long, ctx);
    expect(error.message.length).toBeLessThan(long.length);
    expect(error.message).toContain('…');
  });

  it('handles an empty body', () => {
    expect(providerErrorFromResponse(500, '', ctx).message).toBe('HTTP 500');
  });
});

describe('toAgentError', () => {
  it('passes AgentError subclasses through untouched', () => {
    const original = new ToolNotFoundError('missing');
    expect(toAgentError(original)).toBe(original);
  });

  it('maps an abort-like error to AbortError', () => {
    const wrapped = toAgentError(new DOMException('x', 'AbortError'));
    expect(wrapped).toBeInstanceOf(AbortError);
    expect(wrapped.retryable).toBe(false);
  });

  it('wraps plain errors, preserving message and cause', () => {
    const cause = new Error('kaboom');
    const wrapped = toAgentError(cause);
    expect(wrapped).toBeInstanceOf(AgentError);
    expect(wrapped.message).toBe('kaboom');
    expect(wrapped.cause).toBe(cause);
  });

  it('wraps non-error throwables', () => {
    expect(toAgentError('just a string').message).toContain('just a string');
    expect(toAgentError(42).message).toContain('42');
  });

  it('uses the supplied fallback message for empty errors', () => {
    expect(toAgentError(new Error(''), 'model call failed').message).toBe(
      'model call failed',
    );
  });
});
