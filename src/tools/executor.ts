import {
  ToolApprovalRequiredError,
  ToolDeniedError,
  ToolError,
  ToolExecutionError,
  ToolNotFoundError,
  ToolTimeoutError,
  ToolValidationError,
  TimeoutError,
  isAbortLikeError,
} from '../errors.js';
import type { ToolCall, ToolMessage } from '../providers/types.js';
import { formatIssues } from '../schema.js';
import { withTimeout } from '../utils/async.js';
import { noopLogger, type Logger } from '../utils/logger.js';
import { extractJson, stringifyToolResult, truncate } from '../utils/text.js';
import { withRetry, type RetryOptions } from '../utils/retry.js';
import type { ToolRegistry } from './registry.js';
import type {
  ApprovalHandler,
  ApprovalRequest,
  ToolDefinition,
  ToolPolicy,
  ToolResultPayload,
} from './types.js';

export type ToolExecutionMode = 'sequential' | 'parallel';

export interface ToolExecutorOptions<TState = unknown> {
  readonly registry: ToolRegistry<TState>;
  readonly mode?: ToolExecutionMode;
  /** Max simultaneous tools when `mode` is `parallel`. Default `4`. */
  readonly concurrency?: number;
  readonly policy?: ToolPolicy<TState> | undefined;
  readonly approval?: ApprovalHandler<TState> | undefined;
  /** Applied when a tool does not define its own `timeoutMs`. */
  readonly defaultTimeoutMs?: number;
  /** Applied when a tool does not define its own `maxRetries`. */
  readonly defaultMaxRetries?: number;
  /** Truncate tool output beyond this many characters. Default `20_000`. */
  readonly maxResultLength?: number;
  /**
   * Throw instead of returning an error result when a tool fails. Default
   * `false` — surfacing errors to the model lets it self-correct.
   */
  readonly throwOnToolError?: boolean;
  readonly retry?: RetryOptions | false;
  readonly logger?: Logger;
}

export interface ToolExecutorRunOptions<TState = unknown> {
  readonly runId: string;
  readonly iteration: number;
  readonly signal: AbortSignal;
  readonly state: TState;
}

export interface ToolLifecycleInfo<TState = unknown> {
  readonly call: ToolCall;
  readonly tool: ToolDefinition<unknown, unknown, TState>;
  readonly args: unknown;
  readonly context: {
    runId: string;
    iteration: number;
    signal: AbortSignal;
    state: TState;
  };
}

export interface ToolSuccessInfo<TState = unknown> extends ToolLifecycleInfo<TState> {
  readonly result: ToolResultPayload;
}

export interface ToolFailureInfo<TState = unknown> extends ToolLifecycleInfo<TState> {
  readonly error: AgentErrorLike;
}

/** Minimal shape we need from an error, to avoid an import cycle in types. */
interface AgentErrorLike {
  readonly name: string;
  readonly message: string;
  readonly code?: string;
  readonly details?: Record<string, unknown> | undefined;
}

export interface ToolExecutorHooks<TState = unknown> {
  onToolStart?(this: void, info: ToolLifecycleInfo<TState>): void | Promise<void>;
  onToolSuccess?(this: void, info: ToolSuccessInfo<TState>): void | Promise<void>;
  onToolFailure?(this: void, info: ToolFailureInfo<TState>): void | Promise<void>;
}

interface ResolvedCall<TState> {
  readonly call: ToolCall;
  readonly tool: ToolDefinition<unknown, unknown, TState> | undefined;
  readonly args: unknown;
  readonly rawArgs: unknown;
  readonly policyArgs: unknown;
  readonly error: ToolError | undefined;
}

const DEFAULT_MAX_RESULT_LENGTH = 20_000;

/**
 * Turns model tool calls into results.
 *
 * Everything a tool can do wrong — unknown name, bad arguments, denied by
 * policy, timeout, thrown exception — is converted into a normal result the
 * model can read and recover from. That is what keeps a flaky tool from
 * killing a long-running agent.
 */
export class ToolExecutor<TState = unknown> {
  private readonly options: Required<
    Pick<
      ToolExecutorOptions<TState>,
      | 'mode'
      | 'concurrency'
      | 'defaultTimeoutMs'
      | 'defaultMaxRetries'
      | 'maxResultLength'
    >
  > &
    ToolExecutorOptions<TState>;

  constructor(options: ToolExecutorOptions<TState>) {
    this.options = {
      mode: options.mode ?? 'sequential',
      concurrency: options.concurrency ?? 4,
      defaultTimeoutMs: options.defaultTimeoutMs ?? 30_000,
      defaultMaxRetries: options.defaultMaxRetries ?? 0,
      maxResultLength: options.maxResultLength ?? DEFAULT_MAX_RESULT_LENGTH,
      ...options,
    };
  }

  get registry(): ToolRegistry<TState> {
    return this.options.registry;
  }

  /**
   * Execute a batch of calls. Always resolves: failures come back as
   * `isError` results in the same order as the input.
   */
  async execute(
    calls: readonly ToolCall[],
    run: ToolExecutorRunOptions<TState>,
    hooks: ToolExecutorHooks<TState> = {},
  ): Promise<ToolMessage[]> {
    if (calls.length === 0) return [];

    const resolved = await Promise.all(calls.map((call) => this.resolveCall(call, run)));

    if (this.options.mode === 'parallel' && resolved.length > 1) {
      return this.executeParallel(resolved, run, hooks);
    }
    const messages: ToolMessage[] = [];
    for (const item of resolved) {
      messages.push(await this.executeOne(item, run, hooks));
    }
    return messages;
  }

  private async executeParallel(
    resolved: readonly ResolvedCall<TState>[],
    run: ToolExecutorRunOptions<TState>,
    hooks: ToolExecutorHooks<TState>,
  ): Promise<ToolMessage[]> {
    const concurrency = Math.max(1, this.options.concurrency);
    const messages = new Array<ToolMessage>(resolved.length);
    let cursor = 0;

    const worker = async (): Promise<void> => {
      for (;;) {
        const index = cursor++;
        const item = resolved[index];
        if (item === undefined) return;
        messages[index] = await this.executeOne(item, run, hooks);
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(concurrency, resolved.length) }, worker),
    );
    return messages;
  }

  private async executeOne(
    item: ResolvedCall<TState>,
    run: ToolExecutorRunOptions<TState>,
    hooks: ToolExecutorHooks<TState>,
  ): Promise<ToolMessage> {
    const logger = (this.options.logger ?? noopLogger()).child({
      tool: item.call.name,
      runId: run.runId,
      iteration: run.iteration,
    });

    if (item.tool === undefined || item.error !== undefined) {
      const error =
        item.error ?? new ToolNotFoundError(item.call.name, this.registry.names());
      return this.failure(item, error, run, hooks, logger);
    }

    const info: ToolLifecycleInfo<TState> = {
      call: item.call,
      tool: item.tool,
      args: item.args,
      context: run,
    };

    try {
      await hooks.onToolStart?.(info);
    } catch {
      // A hook must not be able to break tool execution.
    }

    const timeoutMs = item.tool.timeoutMs ?? this.options.defaultTimeoutMs;
    const maxRetries = item.tool.maxRetries ?? this.options.defaultMaxRetries;
    const startedAt = Date.now();

    try {
      const raw = await this.invoke(item, run, logger, timeoutMs, maxRetries);
      const content = this.render(item.tool, raw);
      const result: ToolResultPayload = {
        content,
        value: raw,
        isError: false,
        durationMs: Date.now() - startedAt,
        attempts: 1,
      };

      await this.safeHook.call(this, hooks.onToolSuccess, { ...info, result });
      logger.debug('tool succeeded', { durationMs: result.durationMs });

      return {
        role: 'tool',
        content,
        toolCallId: item.call.id,
        name: item.call.name,
      };
    } catch (error) {
      const failure = this.toToolError(error, item);
      return this.failure(item, failure, run, hooks, logger);
    }
  }

  private async invoke(
    item: ResolvedCall<TState>,
    run: ToolExecutorRunOptions<TState>,
    logger: Logger,
    timeoutMs: number,
    maxRetries: number,
  ): Promise<unknown> {
    const tool = item.tool;
    if (tool === undefined) throw new ToolNotFoundError(item.call.name);

    const invoke = async (): Promise<unknown> => {
      const context = {
        signal: run.signal,
        runId: run.runId,
        state: run.state,
        call: item.call,
        iteration: run.iteration,
        logger,
      };
      try {
        return await withTimeout(
          () => Promise.resolve(tool.execute(item.args as never, context as never)),
          timeoutMs,
          { signal: run.signal, label: `tool:${tool.name}` },
        );
      } catch (error) {
        // Normalise so the failure is retryable (`ToolExecutionError` is) and
        // names the tool, which is what the model needs to self-correct.
        if (error instanceof ToolError || isAbortLikeError(error)) throw error;
        throw new ToolExecutionError(describe(error), {
          toolName: tool.name,
          toolCallId: item.call.id,
          cause: error,
        });
      }
    };

    if (maxRetries <= 0) return invoke();

    return withRetry(invoke, {
      maxAttempts: maxRetries + 1,
      initialDelayMs: 250,
      signal: run.signal,
      ...(this.options.retry === false ? {} : (this.options.retry as RetryOptions)),
    });
  }

  /**
   * Resolve name, policy, approval, and arguments before running anything.
   * Async policies and approval handlers are awaited here, so a denied call
   * never reaches `execute`.
   */
  private async resolveCall(
    call: ToolCall,
    run: ToolExecutorRunOptions<TState>,
  ): Promise<ResolvedCall<TState>> {
    const tool = this.registry.get(call.name);

    if (tool === undefined) {
      return {
        call,
        tool: undefined,
        args: {},
        rawArgs: call.arguments,
        policyArgs: call.arguments,
        error: new ToolNotFoundError(call.name, this.registry.names()),
      };
    }

    const rawArgs = parseRawArguments(call.arguments);

    if (this.options.policy !== undefined) {
      try {
        const decision = await this.options.policy(call, {
          runId: run.runId,
          iteration: run.iteration,
          signal: run.signal,
          state: run.state,
        });
        if (decision.action === 'deny') {
          return {
            call,
            tool,
            args: rawArgs,
            rawArgs,
            policyArgs: call.arguments,
            error: new ToolDeniedError(call.name, decision.reason),
          };
        }
      } catch (error) {
        // A throwing policy denies the call — fail closed.
        return {
          call,
          tool,
          args: rawArgs,
          rawArgs,
          policyArgs: call.arguments,
          error: new ToolDeniedError(call.name, `policy threw: ${describe(error)}`),
        };
      }
    }

    if (tool.requiresApproval !== undefined && tool.requiresApproval !== false) {
      const required =
        typeof tool.requiresApproval === 'function'
          ? tool.requiresApproval(rawArgs)
          : tool.requiresApproval;
      if (required) {
        const granted = await this.options.approval?.(
          buildApprovalRequest(call, rawArgs, run),
          { signal: run.signal, state: run.state },
        );
        if (granted !== true) {
          return {
            call,
            tool,
            args: rawArgs,
            rawArgs,
            policyArgs: call.arguments,
            error: new ToolApprovalRequiredError(call.name, {
              toolName: call.name,
              toolCallId: call.id,
              toolArguments: rawArgs,
            }),
          };
        }
      }
    }

    const validated = validateArguments(tool, call, rawArgs);
    return {
      call,
      tool,
      args: validated.ok ? validated.value : rawArgs,
      rawArgs,
      policyArgs: call.arguments,
      error: validated.ok ? undefined : validated.error,
    };
  }

  private render(tool: ToolDefinition<unknown, unknown, TState>, value: unknown): string {
    const custom = tool.serialize?.(value);
    const content = custom ?? stringifyToolResult(value);
    if (content.length <= this.options.maxResultLength) return content;
    const truncated = truncate(content, this.options.maxResultLength);
    return `${truncated}\n\n[truncated: ${content.length} characters total]`;
  }

  private async failure(
    item: ResolvedCall<TState>,
    error: ToolError,
    run: ToolExecutorRunOptions<TState>,
    hooks: ToolExecutorHooks<TState>,
    logger: Logger,
  ): Promise<ToolMessage> {
    const content = `Error: ${error.message}`;

    if (this.options.throwOnToolError === true) throw error;

    await this.safeHook.call(this, hooks.onToolFailure, {
      call: item.call,
      tool: item.tool ?? ({} as ToolDefinition<unknown, unknown, TState>),
      args: item.args,
      context: run,
      error,
    });
    logger.warn('tool failed', { error: error.message, code: error.code });

    return {
      role: 'tool',
      content,
      toolCallId: item.call.id,
      name: item.call.name,
      isError: true,
    };
  }

  private async safeHook(
    hook: ((info: never) => void | Promise<void>) | undefined,
    info: unknown,
  ): Promise<void> {
    try {
      await hook?.(info as never);
    } catch {
      // Observability must never break execution.
    }
  }

  private toToolError(error: unknown, item: ResolvedCall<TState>): ToolError {
    const details = {
      toolName: item.call.name,
      toolCallId: item.call.id,
      toolArguments: item.rawArgs,
    };
    if (error instanceof ToolError) return error;
    if (error instanceof TimeoutError) {
      return new ToolTimeoutError(
        `Tool "${item.call.name}" timed out after ${error.timeoutMs}ms`,
        details,
      );
    }
    if (isAbortLikeError(error)) {
      return new ToolExecutionError(`Tool "${item.call.name}" was aborted`, details);
    }
    return new ToolExecutionError(`Tool "${item.call.name}" failed: ${describe(error)}`, {
      ...details,
      cause: error,
    });
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Some models emit tool arguments as a JSON *string*. Normalise that, plus
 * empty/absent arguments, into a plain object before validation.
 */
function parseRawArguments(args: unknown): unknown {
  if (args === undefined || args === null || args === '') return {};
  if (typeof args === 'string') return extractJson(args) ?? {};
  return args;
}

function validateArguments<TState>(
  tool: ToolDefinition<unknown, unknown, TState>,
  call: ToolCall,
  args: unknown,
): { ok: true; value: unknown } | { ok: false; error: ToolValidationError } {
  if (tool.parameters === undefined) return { ok: true, value: args };
  const result = tool.parameters.safeParse(args);
  if (result.success) return { ok: true, value: result.data };
  return {
    ok: false,
    error: new ToolValidationError(
      `Invalid arguments for tool "${call.name}":\n${formatIssues(result.error.issues)}`,
      {
        toolName: call.name,
        toolCallId: call.id,
        toolArguments: args,
        issues: result.error.issues,
      },
    ),
  };
}

function buildApprovalRequest(
  call: ToolCall,
  args: unknown,
  run: ToolExecutorRunOptions<unknown>,
): ApprovalRequest {
  return {
    toolName: call.name,
    args,
    call,
    runId: run.runId,
    iteration: run.iteration,
    summary: `${call.name}(${truncate(stringifyToolResult(args), 500)})`,
  };
}

function describe(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
