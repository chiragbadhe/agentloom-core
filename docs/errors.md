# Errors

Everything the kit throws extends `AgentError` and carries a stable `code`, so
you can branch on failure kind without matching on message text.

```ts
import { AgentError, ErrorCode } from '@agentloom/core';

try {
  await agent.run(input);
} catch (error) {
  if (error instanceof AgentError) {
    console.error(error.code, error.message, error.details);
  }
}
```

`AgentError` fields:

| Field          | Notes                                         |
| -------------- | --------------------------------------------- |
| `code`         | Stable discriminant. See the table below      |
| `retryable`    | Whether retrying could plausibly succeed      |
| `details`      | Structured context, e.g. `{ availableTools }` |
| `retryAfterMs` | Server-advised wait, when known               |
| `cause`        | The underlying error, when there was one      |

`toJSON()` produces a log-friendly object, so a thrown error serializes cleanly
into structured logs.

## Hierarchy

```
AgentError
├── ConfigurationError
├── AbortError
├── TimeoutError
├── ProviderError
│   ├── ProviderAuthError
│   ├── ProviderRateLimitError
│   ├── ProviderQuotaError
│   ├── ProviderTimeoutError
│   └── ProviderResponseError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolValidationError
│   ├── ToolExecutionError
│   ├── ToolTimeoutError
│   ├── ToolDeniedError
│   └── ToolApprovalRequiredError
├── ValidationError
├── MemoryError
├── PlanningError
└── ExecutionLimitError
```

## Codes

| Error                       | `code`                      | Retryable         |
| --------------------------- | --------------------------- | ----------------- |
| `ConfigurationError`        | `CONFIGURATION_ERROR`       | no                |
| `AbortError`                | `ABORTED`                   | no                |
| `TimeoutError`              | `TIMEOUT`                   | yes               |
| `ProviderError`             | `PROVIDER_ERROR`            | depends on status |
| `ProviderAuthError`         | `PROVIDER_AUTH_ERROR`       | no                |
| `ProviderRateLimitError`    | `PROVIDER_RATE_LIMIT_ERROR` | yes               |
| `ProviderQuotaError`        | `PROVIDER_QUOTA_ERROR`      | no                |
| `ProviderTimeoutError`      | `PROVIDER_TIMEOUT_ERROR`    | yes               |
| `ProviderResponseError`     | `PROVIDER_RESPONSE_ERROR`   | no                |
| `ToolNotFoundError`         | `TOOL_NOT_FOUND_ERROR`      | no                |
| `ToolValidationError`       | `TOOL_VALIDATION_ERROR`     | no                |
| `ToolExecutionError`        | `TOOL_EXECUTION_ERROR`      | yes               |
| `ToolTimeoutError`          | `TOOL_TIMEOUT_ERROR`        | yes               |
| `ToolDeniedError`           | `TOOL_DENIED_ERROR`         | no                |
| `ToolApprovalRequiredError` | `TOOL_APPROVAL_ERROR`       | no                |
| `ValidationError`           | `VALIDATION_ERROR`          | no                |
| `MemoryError`               | `MEMORY_ERROR`              | yes               |
| `PlanningError`             | `PLANNING_ERROR`            | no                |
| `ExecutionLimitError`       | `EXECUTION_LIMIT_ERROR`     | no                |

`isRetryableError(error)` and `isRetryableStatus(status)` are exported.
`isRetryableStatus` treats 408, 409, 425, 429, and 5xx as retryable.

## Throwing versus returning

Tool failures are the important exception. By default they do **not** throw:
they become tool results with `isError: true` and a readable message, so the
model can see what went wrong and adapt.

```ts
// Model calls calculator('1/0'), gets an error result, and tries something else.
```

Everything else — provider failures, validation failures, timeouts, aborts —
throws from `run()` by default.

Set `throwOnToolError: true` to change the tool behaviour. Set
`throwOnError: false` on a run to get a result instead of an exception:

```ts
const result = await agent.run(input, { throwOnError: false });

if (result.stopReason === 'error') {
  console.error(result.error?.code, result.error?.message);
} else if (result.stopReason !== 'completed') {
  console.warn(`stopped: ${result.stopReason}`);
}
```

`stopReason: 'aborted'` for a cancellation, `'error'` for anything else fatal.

## Common cases

### Rate limited

Already handled by default: the retry layer backs off and honours
`Retry-After`. To react rather than absorb it:

```ts
if (error instanceof ProviderRateLimitError) {
  await sleep(error.retryAfterMs ?? 2_000);
}
```

`ProviderQuotaError` is deliberately _not_ retryable — billing needs a human.

### Timed out

```ts
if (error instanceof TimeoutError) {
  // Raise the budget, or reduce work per iteration.
}
```

`TimeoutError.timeoutMs` holds the deadline that elapsed.

### Cancelled

```ts
if (error instanceof AbortError) return; // not an error condition
```

### Structured output failed

```ts
if (error instanceof ValidationError) {
  for (const issue of error.issues) {
    console.error(`${issue.path.join('.')}: ${issue.message}`);
  }
}
```

Raise `limits.maxOutputAttempts`, or relax the schema. See
[structured output](agent.md#structured-output).

### A tool was denied or needed approval

These normally reach the model as error results rather than throwing. Set
`throwOnToolError: true` if you want to intercept them:

```ts
if (error instanceof ToolDeniedError) console.warn(error.reason);
if (error instanceof ToolApprovalRequiredError) console.warn('approval needed');
```

### The loop hit a limit

Not an exception. Check `result.stopReason`:

```ts
const result = await agent.run(input);
if (result.stopReason === 'max_iterations') {
  console.warn('gave up after', result.iterations, 'iterations');
}
```

`ExecutionLimitError` carries `limit` and `limitValue`, and lands in
`result.error` when a limit stopped the run.

### Misconfiguration

```ts
if (error instanceof ConfigurationError) {
  console.error(error.details); // e.g. { available: [...] }
}
```

Usually a bug rather than a runtime condition: an unknown provider, a bad tool
name, a contradictory registry scope.

## Recoverable versus fatal

The `error` event carries `fatal` and `scope`, which is what you want for
telemetry rather than for control flow:

```ts
agent.on('error', ({ error, scope, fatal }) => {
  metrics.inc('agent.error', { code: error.code, scope, fatal });
  if (!fatal) logger.debug('recovered', { error: error.message });
});
```

| `scope`  | Source                                    |
| -------- | ----------------------------------------- |
| `model`  | Provider calls                            |
| `tool`   | Anything `TOOL_*`                         |
| `memory` | A memory backend                          |
| `output` | Structured-output validation              |
| `agent`  | Limits, aborts, planning, everything else |

Non-fatal errors are recovered from by design: tool failures, long-term memory
failures, planning failures, and output-repair attempts. A fatal one ends the
run.

## Writing errors that fit in

Raising a typed error from a tool is fine — the executor wraps it:

```ts
import { ToolExecutionError } from '@agentloom/core';

execute: async ({ id }) => {
  const record = await db.find(id);
  if (!record)
    throw new ToolExecutionError(`No record ${id}`, { retryable: false });
  return record;
};
```

Anything thrown that is not already a `ToolError` becomes a
`ToolExecutionError` naming the tool, which is the form the model needs in order
to self-correct. `toAgentError()` normalizes an arbitrary thrown value if you are
building your own provider or tool wrapper.

## More

- [Agent](agent.md) — limits, stop reasons, `throwOnError`
- [Providers](providers.md) — status-to-error mapping and retry behaviour
- [Tools](tools.md) — policies and approval, which produce tool errors
