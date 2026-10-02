import {
  AbortError,
  AgentError,
  ConfigurationError,
  ErrorCode,
  ExecutionLimitError,
  ValidationError,
  isAbortLikeError,
  toAgentError,
} from '../errors.js';
import {
  InMemoryConversationMemory,
  defaultTokenCounter,
} from '../memory/conversation.js';
import type {
  LongTermMemory,
  MemoryRecord,
  MemorySearchResult,
} from '../memory/long-term.js';
import type { ConversationMemory, TokenCounter } from '../memory/types.js';
import { toJsonSchema } from '../providers/json-schema.js';
import { createProvider } from '../providers/registry.js';
import type {
  CompletionOptions,
  CompletionRequest,
  CompletionResult,
  ModelMessage,
  ModelProvider,
  ToolCall,
  ToolMessage,
  ToolSpec,
  Usage,
} from '../providers/types.js';
import { formatIssues, type Schema, type SchemaIssue } from '../schema.js';
import { ToolExecutor } from '../tools/executor.js';
import { defineTool, toToolRegistry } from '../tools/registry.js';
import type { ToolRegistry } from '../tools/registry.js';
import {
  combineSignals,
  raceIterableWithAbort,
  raceWithAbort,
  throwIfAborted,
} from '../utils/async.js';
import { TypedEventEmitter, type Unsubscribe } from '../utils/emitter.js';
import { createId } from '../utils/id.js';
import { noopLogger, type Logger } from '../utils/logger.js';
import { AsyncQueue } from '../utils/queue.js';
import { withRetry, type RetryOptions } from '../utils/retry.js';
import { extractJson, stringifyToolResult } from '../utils/text.js';
import { BudgetTracker } from './budget.js';
import type {
  AgentConfig,
  LongTermMemoryConfig,
  OutputConfig,
  PlannerConfig,
  RunOptions,
} from './config.js';
import type {
  AgentEndEvent,
  AgentEventMap,
  AgentEventName,
  AgentStartEvent,
  ErrorEvent,
  IterationEndEvent,
  IterationStartEvent,
  MemoryEvent,
  ModelDeltaEvent,
  ModelEndEvent,
  OutputInvalidEvent,
  PromptEvent,
  RetryEvent,
  ToolEndEvent,
  ToolStartEvent,
} from './events.js';
import type { AgentHooks } from './hooks.js';
import {
  PlanTracker,
  createPlanWithModel,
  renderPlan,
  type Plan,
  type PlanStepStatus,
} from './planner.js';
import type {
  AgentLimits,
  AgentResult,
  AgentStep,
  ResolvedLimits,
  StopReason,
} from './result.js';
import { resolveLimits } from './result.js';
import { AgentStream, type AgentStreamEvent } from './stream.js';

const DEFAULT_INSTRUCTIONS =
  'You are a helpful assistant. Use the tools available to you when they ' +
  'would produce a more accurate or complete answer, and prefer acting over ' +
  'guessing. If a tool fails, read the error and adapt rather than repeating ' +
  'the same call unchanged.';

const DEFAULT_KEEP_RECENT = 4;

interface RunContext<TState> {
  readonly runId: string;
  readonly signal: AbortSignal;
  readonly state: TState;
  readonly controller: AbortController;
}

/**
 * An autonomous agent: a model, a set of tools, memory, and a loop that ties
 * them together.
 *
 * ```ts
 * const agent = new Agent({
 *   name: 'support',
 *   instructions: 'You are a support engineer. Escalate after two failed attempts.',
 *   model: 'openai:gpt-4o-mini',
 *   tools: [createCalculatorTool(), searchOrders],
 *   limits: { maxIterations: 8, timeoutMs: 60_000 },
 * });
 *
 * const { output } = await agent.run('What is 17 * 23?');
 * ```
 *
 * Instances hold conversation state, so a chatbot is one agent across many
 * `run` calls. Use {@link Agent.fork} for an independent copy.
 */
export class Agent<TState = unknown> {
  readonly name: string;
  readonly tools: ToolRegistry<TState>;
  readonly memory: ConversationMemory;
  readonly longTermMemory: LongTermMemory | undefined;
  readonly provider: ModelProvider;
  readonly model: string;
  readonly state: TState;
  readonly limits: ResolvedLimits;
  readonly planTracker = new PlanTracker();

  private readonly config: AgentConfig<TState>;
  private readonly emitter = new TypedEventEmitter<AgentEventMap>();
  private readonly logger: Logger;
  private readonly hooks: AgentHooks<TState>;
  private readonly tokenCounter: TokenCounter;
  private readonly longTerm: Required<LongTermMemoryConfig>;
  private readonly outputConfig: OutputConfig<unknown> | undefined;
  private readonly instructions: string | SystemPromptFn<TState>;
  private readonly modelOptions: CompletionOptions;
  private readonly retryOptions: RetryOptions | undefined;
  private readonly plannerConfig: PlannerConfig<TState>;
  private readonly plannerEnabled: boolean;

  constructor(config: AgentConfig<TState>) {
    if (config.provider === undefined && config.model === undefined) {
      throw new ConfigurationError(
        'Agent requires either `model` (e.g. "openai:gpt-4o-mini") or `provider`.',
      );
    }

    this.config = config;
    this.name = config.name ?? 'agent';
    this.logger = (config.logger ?? noopLogger()).child({ agent: this.name });
    this.hooks = config.hooks ?? {};
    this.tokenCounter = config.tokenCounter ?? defaultTokenCounter;
    this.instructions = config.instructions ?? DEFAULT_INSTRUCTIONS;
    this.state = config.state as TState;
    this.memory = config.memory ?? new InMemoryConversationMemory();
    this.longTermMemory = config.longTermMemory;
    this.modelOptions = config.modelOptions ?? {};
    this.limits = resolveLimits(config.limits);

    const { provider, model } = this.resolveProvider(config);
    this.provider = provider;
    this.model = model;

    this.tools = toToolRegistry<TState>(config.tools);

    this.longTerm = {
      recall: config.longTerm?.recall ?? true,
      store: config.longTerm?.store ?? true,
      limit: config.longTerm?.limit ?? 5,
      minScore: config.longTerm?.minScore ?? 0.01,
      roles: config.longTerm?.roles ?? ['user', 'system'],
      heading: config.longTerm?.heading ?? 'Relevant memories from earlier sessions',
    };

    this.outputConfig = config.output;
    this.retryOptions = config.retry === false ? undefined : config.retry;

    const planner = config.planner === false ? undefined : config.planner;
    this.plannerConfig = planner ?? {};
    this.plannerEnabled = planner !== undefined && planner.enabled !== false;
  }

  /** `true` when this agent plans before executing. */
  get planningEnabled(): boolean {
    return this.plannerEnabled;
  }

  /** The current plan, if planning is enabled. */
  get currentPlan(): Plan | undefined {
    return this.planTracker.plan;
  }

  // -------------------------------------------------------------------------
  // Events
  // -------------------------------------------------------------------------

  /** Subscribe to a lifecycle event. */
  on<K extends AgentEventName>(
    event: K,
    listener: (payload: AgentEventMap[K][0]) => void,
  ): Unsubscribe {
    return this.emitter.on(event, listener);
  }

  /** Subscribe for a single delivery. */
  once<K extends AgentEventName>(
    event: K,
    listener: (payload: AgentEventMap[K][0]) => void,
  ): Unsubscribe {
    return this.emitter.once(event, listener);
  }

  /** Subscribe to several events at once. */
  onMany(
    handlers: Partial<{ [K in AgentEventName]: (payload: AgentEventMap[K][0]) => void }>,
  ): Unsubscribe {
    return this.emitter.onMany(handlers);
  }

  /** Drop listeners for one event, or for all events. */
  removeAllListeners(event?: AgentEventName): void {
    this.emitter.removeAllListeners(event);
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /** Clear conversation memory and the plan. Long-term memory is untouched. */
  reset(): void {
    void this.memory.clear();
    this.planTracker.reset();
  }

  /** Tool specs in the shape providers expect. */
  toolSpecs(): readonly ToolSpec[] {
    return this.tools.toSpecs();
  }

  /** A shallow, independent copy with overridden configuration. */
  fork(overrides: Partial<AgentConfig<TState>> = {}): Agent<TState> {
    return new Agent<TState>({
      ...this.config,
      name: `${this.name}-fork`,
      memory: new InMemoryConversationMemory(),
      tools: this.tools,
      ...overrides,
    });
  }

  // -------------------------------------------------------------------------
  // Entry points
  // -------------------------------------------------------------------------

  /**
   * Run the agent to completion.
   *
   * @throws {AbortError} when cancelled via `options.signal`.
   * @throws {TimeoutError} when `limits.timeoutMs` elapses.
   * @throws {ValidationError} when `outputSchema` cannot be satisfied.
   * @throws {ProviderError} on unrecoverable provider failures.
   */
  async run<TOut = unknown>(
    input: string,
    options: RunOptions<TOut, TState> = {},
  ): Promise<AgentResult<TOut>> {
    const stream = this.createRun(input, options, false);
    return stream.result;
  }

  /**
   * Run the agent, yielding events as they happen.
   *
   * ```ts
   * const controller = new AbortController();
   * const run = agent.stream('write a haiku', { signal: controller.signal });
   * for await (const event of run) {
   *   if (event.type === 'text') process.stdout.write(event.text);
   * }
   * const result = await run.result;
   * ```
   *
   * The run starts immediately; consume events whenever you are ready.
   */
  stream<TOut = unknown>(
    input: string,
    options: RunOptions<TOut, TState> = {},
  ): AgentStream<AgentStreamEvent, AgentResult<TOut>> {
    return this.createRun(input, options, true);
  }

  /** Produce a plan for `input` without running the loop. */
  async plan(input: string, options: { signal?: AbortSignal } = {}): Promise<Plan> {
    const controller = new AbortController();
    const { signal, dispose } = combineSignals(controller.signal, options.signal);
    try {
      return await this.buildPlan(input, createId('plan'), signal);
    } finally {
      dispose();
    }
  }

  // -------------------------------------------------------------------------
  // Run implementation
  // -------------------------------------------------------------------------

  private createRun<TOut>(
    input: string,
    options: RunOptions<TOut, TState>,
    streaming: boolean,
  ): AgentStream<AgentStreamEvent, AgentResult<TOut>> {
    const runId = options.runId ?? createId('run');
    const controller = new AbortController();
    const { signal, dispose } = combineSignals(controller.signal, options.signal);
    const context: RunContext<TState> = {
      runId,
      signal,
      state: this.state,
      controller,
    };

    const queue = new AsyncQueue<AgentStreamEvent>(streaming ? 256 : 1);
    const stream = new AgentStream<AgentStreamEvent, AgentResult<TOut>>(
      queue,
      controller,
      dispose,
      runId,
    );

    // Errors propagate to `stream.result` (and therefore to `run()`); the
    // stream itself just terminates. The extra catch keeps a consumer that
    // only iterates events from triggering an unhandled rejection.
    const execution = this.execute<TOut>(
      input,
      options,
      context,
      streaming ? queue : undefined,
    ).finally(() => {
      queue.close();
      dispose();
    });

    stream.result = execution;
    void execution.catch(() => undefined);
    return stream;
  }

  private async execute<TOut>(
    input: string,
    options: RunOptions<TOut, TState>,
    context: RunContext<TState>,
    queue: AsyncQueue<AgentStreamEvent> | undefined,
  ): Promise<AgentResult<TOut>> {
    const startedAt = Date.now();
    const limits = mergeLimits(this.limits, options.limits);
    const budget = new BudgetTracker(limits);
    const tools =
      options.tools === undefined ? this.tools : toToolRegistry<TState>(options.tools);
    const outputSchema = options.outputSchema ?? this.outputConfig?.schema;
    const tracker = this.planTracker;

    const emit = <K extends AgentEventName>(
      event: K,
      payload: AgentEventMap[K][0],
    ): void => {
      this.emitter.emit(event, ...([payload] as unknown as AgentEventMap[K]));
      const mapped = mapToStreamEvent(event, payload);
      if (mapped !== undefined && queue !== undefined) void queue.push(mapped);
    };

    const hook = async <K extends keyof AgentHooks>(
      name: K,
      payload: unknown,
    ): Promise<unknown> => {
      const fn = this.hooks[name] as ((input: unknown) => unknown) | undefined;
      if (fn === undefined) return undefined;
      // Observational hooks are isolated; vetoing hooks are awaited directly.
      // The call is inside the try because a hook may throw synchronously,
      // before it ever returns a promise to await.
      try {
        return await Promise.resolve(fn(payload));
      } catch {
        return undefined;
      }
    };

    const reportError = async (
      error: AgentError,
      scope: ErrorEvent['scope'],
      fatal: boolean,
    ): Promise<void> => {
      emit('error', {
        runId: context.runId,
        error,
        iteration: budget.iterations,
        fatal,
        scope,
      });
      await hook('onError', {
        runId: context.runId,
        error,
        iteration: budget.iterations,
        fatal,
        scope,
      });
    };

    const startEvent: AgentStartEvent = {
      runId: context.runId,
      input,
      agentName: this.name,
      provider: this.provider.id,
      model: this.model,
      tools: tools.names(),
      limits,
      startedAt,
    };

    try {
      await this.hooks.onAgentStart?.(startEvent);
    } catch (error) {
      const wrapped = toAgentError(error, 'onAgentStart hook failed');
      emit('agent:start', startEvent);
      await reportError(wrapped, 'agent', true);
      throw wrapped;
    }
    emit('agent:start', startEvent);

    const steps: AgentStep[] = [];

    const finalize = async (
      output: string,
      data: unknown,
      stopReason: StopReason,
      error?: AgentError,
    ): Promise<AgentResult<TOut>> => {
      if (stopReason === 'completed' && tracker.plan !== undefined) {
        tracker.completeAll();
      }

      let finalOutput = output;
      try {
        const replacement = await this.hooks.onBeforeReturn?.({
          runId: context.runId,
          output,
          stopReason,
          data,
          state: context.state,
        });
        if (typeof replacement === 'string') finalOutput = replacement;
      } catch (hookError) {
        const wrapped = toAgentError(hookError, 'onBeforeReturn hook failed');
        await reportError(wrapped, 'agent', true);
        throw wrapped;
      }

      if (this.longTermMemory !== undefined && this.longTerm.store) {
        await this.storeMemories(input, finalOutput, context.runId, emit, hook);
      }

      const messages = await this.memory.messages();
      const endedAt = Date.now();
      const result: AgentResult<TOut> = {
        output: finalOutput,
        data: data as TOut | undefined,
        messages,
        steps,
        iterations: budget.iterations,
        usage: budget.usage,
        stopReason,
        error,
        runId: context.runId,
        startedAt,
        endedAt,
        durationMs: endedAt - startedAt,
      };

      emit('agent:end', {
        runId: context.runId,
        stopReason,
        output: finalOutput,
        iterations: budget.iterations,
        usage: budget.usage,
        durationMs: result.durationMs,
      } satisfies AgentEndEvent);
      await hook('onAgentEnd', {
        runId: context.runId,
        stopReason,
        output: finalOutput,
        iterations: budget.iterations,
        usage: budget.usage,
        durationMs: result.durationMs,
      });

      queue?.push({ type: 'done', runId: context.runId, result }).catch(() => undefined);
      return result;
    };

    try {
      await this.memory.add({ role: 'user', content: input });

      const recalled = await this.recallMemories(input, context, emit, hook);

      // Planning happens before the prompt is assembled, so the plan can be
      // rendered into the system prompt for the very first iteration.
      if (this.shouldPlan()) {
        try {
          tracker.set(await this.buildPlan(input, context.runId, context.signal));
        } catch (error) {
          // Planning is an optimization; a failure must not sink the run.
          await reportError(toAgentError(error, 'Planning failed'), 'agent', false);
          this.logger.warn('planning failed, continuing without a plan', { error });
        }
      }

      const systemPrompt = await this.composeSystemPrompt(
        input,
        context,
        options,
        recalled,
      );

      const executor = this.createExecutor(tools);
      if (this.plannerEnabled && this.plannerConfig.exposeUpdateTool !== false) {
        executor.registry.add(this.createPlanTool(tracker, context, emit));
      }

      let stopReason: StopReason = 'completed';
      let lastText = '';
      let outputAttempts = 0;

      for (;;) {
        throwIfAborted(context.signal);

        const gate = budget.check({ isAbort: context.signal.aborted });
        if (gate !== undefined) {
          stopReason = gate;
          break;
        }

        const iteration = ++budget.iterations;
        const iterationStartedAt = Date.now();

        const iterationStart: IterationStartEvent = {
          runId: context.runId,
          iteration,
          messageCount: (await this.memory.messages()).length,
        };
        emit('iteration:start', iterationStart);
        await hook('onIterationStart', iterationStart);

        const promptMessages = await this.memory.build({
          system: systemPrompt,
          maxTokens: limits.contextWindow,
          tokenCounter: this.tokenCounter,
          keepRecent: DEFAULT_KEEP_RECENT,
          signal: context.signal,
        });

        const promptEvent: PromptEvent = {
          runId: context.runId,
          iteration,
          messages: promptMessages,
          estimatedTokens: this.tokenCounter(promptMessages),
          system: systemPrompt,
          recalled: recalled.map((hit) => hit.record.content),
        };
        emit('agent:prompt', promptEvent);
        await hook('onPrompt', promptEvent);

        let request = this.buildRequest(promptMessages, executor.registry, outputSchema);
        const rewritten = await this.hooks.onBeforeModel?.({
          runId: context.runId,
          iteration,
          request,
          state: context.state,
          signal: context.signal,
        });
        if (rewritten !== undefined && rewritten !== null) request = rewritten;

        const response = await this.callModel(
          request,
          iteration,
          context,
          emit,
          hook,
          budget,
          limits,
          queue,
        );
        const text = response.message.content;
        lastText = text;

        emit('model:end', {
          runId: context.runId,
          iteration,
          result: response,
          durationMs: response.latencyMs,
        } satisfies ModelEndEvent);
        await hook('onModelEnd', {
          runId: context.runId,
          iteration,
          result: response,
          durationMs: response.latencyMs,
        });

        if (queue !== undefined && text !== '') {
          await queue.push({ type: 'text', runId: context.runId, iteration, text });
        }

        const toolCalls = response.message.toolCalls ?? [];

        // -- structured output -------------------------------------------
        if (outputSchema !== undefined && toolCalls.length === 0 && text.trim() !== '') {
          const validated = validateStructured(text, outputSchema);

          if (validated.ok) {
            await this.memory.add(response.message);
            steps.push(
              buildStep(iteration, request, response, [], [], iterationStartedAt, text),
            );
            return await finalize(text, validated.data, 'completed');
          }

          outputAttempts++;
          const invalidEvent: OutputInvalidEvent = {
            runId: context.runId,
            iteration,
            attempt: outputAttempts,
            issues: validated.issues,
            raw: text,
          };
          emit('output:invalid', invalidEvent);
          await hook('onOutputInvalid', invalidEvent);

          if (outputAttempts >= limits.maxOutputAttempts) {
            const failure = new ValidationError(
              `Model output did not satisfy the output schema after ${outputAttempts} attempt(s):\n${formatIssues(validated.issues)}`,
              validated.issues,
            );
            await this.memory.add(response.message);
            steps.push(
              buildStep(iteration, request, response, [], [], iterationStartedAt, text),
            );
            await reportError(failure, 'output', true);
            if (options.throwOnError !== false) throw failure;
            return await finalize(text, undefined, 'max_output_attempts', failure);
          }

          await this.memory.add(response.message);
          await this.memory.add({
            role: 'user',
            content: [
              'Your previous response was not valid.',
              'Problems found:',
              formatIssues(validated.issues),
              'Respond again with valid JSON that matches the required schema.',
              'Output only the JSON.',
            ].join('\n'),
          });
          continue;
        }

        await this.memory.add(response.message);

        // -- no tool calls: done -----------------------------------------
        if (toolCalls.length === 0) {
          steps.push(
            buildStep(iteration, request, response, [], [], iterationStartedAt, text),
          );
          await this.endIteration(emit, hook, {
            runId: context.runId,
            iteration,
            toolCallCount: 0,
            durationMs: Date.now() - iterationStartedAt,
            usage: budget.usage,
          });

          if (outputSchema !== undefined) {
            const failure = new ValidationError(
              text.trim() === ''
                ? 'Model returned an empty response but an output schema was required.'
                : 'Model produced tool calls while an output schema was required; no valid JSON was returned.',
            );
            await reportError(failure, 'output', true);
            if (options.throwOnError !== false) throw failure;
            return await finalize(text, undefined, 'max_output_attempts', failure);
          }

          return await finalize(text, undefined, 'completed');
        }

        // -- tool execution ---------------------------------------------
        const allowed = budget.clampIterationToolCalls(toolCalls.length);
        const permitted = toolCalls.slice(0, allowed);
        const overflow = toolCalls.slice(allowed);

        const toolResults = await this.executeTools(
          executor,
          permitted,
          iteration,
          context,
          emit,
          hook,
        );
        await this.memory.addAll(toolResults);
        budget.toolCalls += toolResults.length;

        for (const call of overflow) {
          const message: ToolMessage = {
            role: 'tool',
            content: `Error: the tool-call budget for this run is exhausted, so "${call.name}" was not executed. Finish with what you have.`,
            toolCallId: call.id,
            name: call.name,
            isError: true,
          };
          toolResults.push(message);
          await this.memory.add(message);
        }

        steps.push(
          buildStep(
            iteration,
            request,
            response,
            toolCalls,
            toolResults,
            iterationStartedAt,
            text,
          ),
        );
        await this.endIteration(emit, hook, {
          runId: context.runId,
          iteration,
          toolCallCount: toolResults.length,
          durationMs: Date.now() - iterationStartedAt,
          usage: budget.usage,
        });
      }

      // -- a limit stopped the loop --------------------------------------
      const limitError = new ExecutionLimitError(
        stopReason,
        stopReason === 'max_iterations'
          ? limits.maxIterations
          : (limits.maxToolCalls ?? 0),
        describeStop(stopReason, limits, budget),
      );
      await reportError(limitError, 'agent', false);
      this.logger.info('run stopped', { stopReason, iterations: budget.iterations });

      return await finalize(lastText, undefined, stopReason, limitError);
    } catch (error) {
      const agentError = normalizeRunError(error, context.signal);
      await reportError(
        agentError,
        scopeForError(agentError),
        agentError.code !== ErrorCode.ABORTED,
      );
      this.logger.error('run failed', {
        error: agentError.message,
        code: agentError.code,
        iterations: budget.iterations,
      });

      if (options.throwOnError !== false) throw agentError;
      return await finalize(
        '',
        undefined,
        agentError.code === ErrorCode.ABORTED ? 'aborted' : 'error',
        agentError,
      );
    }
  }

  private async endIteration(
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
    event: IterationEndEvent,
  ): Promise<void> {
    emit('iteration:end', event);
    await hook('onIterationEnd', event);
  }

  // -------------------------------------------------------------------------
  // Tools
  // -------------------------------------------------------------------------

  private createExecutor(tools: ToolRegistry<TState>): ToolExecutor<TState> {
    const config = this.config;
    return new ToolExecutor<TState>({
      registry: tools,
      mode: config.execution ?? 'sequential',
      concurrency: config.toolConcurrency ?? 4,
      policy: config.toolPolicy,
      approval: config.approval,
      maxResultLength: config.maxToolResultLength ?? 20_000,
      throwOnToolError: config.throwOnToolError === true,
      logger: this.logger,
    });
  }

  private async executeTools(
    executor: ToolExecutor<TState>,
    calls: readonly ToolCall[],
    iteration: number,
    context: RunContext<TState>,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
  ): Promise<ToolMessage[]> {
    const startedAt = Date.now();

    return executor.execute(
      calls,
      { runId: context.runId, iteration, signal: context.signal, state: context.state },
      {
        onToolStart: async (info) => {
          const event: ToolStartEvent = {
            runId: context.runId,
            iteration,
            call: info.call,
            args: info.args,
            toolName: info.call.name,
          };
          emit('tool:start', event);
          await hook('onToolStart', event);
        },
        onToolSuccess: async (info) => {
          const event: ToolEndEvent = {
            runId: context.runId,
            iteration,
            call: info.call,
            toolName: info.call.name,
            result: {
              role: 'tool',
              content: info.result.content,
              toolCallId: info.call.id,
              name: info.call.name,
            },
            durationMs: info.result.durationMs,
          };
          emit('tool:end', event);
          await hook('onToolEnd', event);
        },
        onToolFailure: async (info) => {
          const error = toAgentError(info.error, 'Tool failed');
          emit('tool:error', {
            runId: context.runId,
            iteration,
            call: info.call,
            toolName: info.call.name,
            error,
            durationMs: Date.now() - startedAt,
          });
          await hook('onToolError', {
            runId: context.runId,
            iteration,
            call: info.call,
            toolName: info.call.name,
            error,
            durationMs: Date.now() - startedAt,
          });
        },
      },
    );
  }

  private createPlanTool(
    tracker: PlanTracker,
    context: RunContext<TState>,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
  ) {
    return defineTool<{
      steps: { id?: string; title?: string; status?: PlanStepStatus; notes?: string }[];
      goal?: string;
    }>({
      name: 'update_plan',
      description:
        'Report progress on the current plan. Include every step with its latest ' +
        'status (pending, in_progress, completed, or skipped) so the plan stays accurate.',
      jsonSchema: {
        type: 'object',
        properties: {
          steps: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' },
                title: { type: 'string' },
                status: {
                  type: 'string',
                  enum: ['pending', 'in_progress', 'completed', 'skipped'],
                },
                notes: { type: 'string' },
              },
              required: ['status'],
            },
          },
          goal: { type: 'string' },
        },
        required: ['steps'],
      },
      execute: (args) => {
        const plan = tracker.update(args);
        if (plan !== undefined) {
          emit('plan', {
            runId: context.runId,
            plan,
            steps: plan.steps,
            completed: plan.steps.filter((step) => step.status === 'completed').length,
          });
        }
        return {
          updated: true,
          goal: plan?.goal,
          steps: plan?.steps.map((step) => ({
            id: step.id,
            title: step.title,
            status: step.status,
          })),
        };
      },
    });
  }

  // -------------------------------------------------------------------------
  // Model invocation
  // -------------------------------------------------------------------------

  private async callModel(
    request: CompletionRequest,
    iteration: number,
    context: RunContext<TState>,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
    budget: BudgetTracker,
    limits: ResolvedLimits,
    queue: AsyncQueue<AgentStreamEvent> | undefined,
  ): Promise<CompletionResult> {
    const callOptions = {
      signal: context.signal,
      ...(limits.modelTimeoutMs === undefined
        ? {}
        : { timeoutMs: limits.modelTimeoutMs }),
    };

    const call = async (attempt: number): Promise<CompletionResult> => {
      emit('model:start', { runId: context.runId, iteration, request, attempt });
      await hook('onModelStart', { runId: context.runId, iteration, request, attempt });

      const result =
        queue !== undefined && this.provider.capabilities.streaming
          ? await this.completeStreaming(
              request,
              iteration,
              context,
              emit,
              hook,
              callOptions,
            )
          : await raceWithAbort(
              this.provider.complete(request, callOptions),
              context.signal,
            );

      budget.record(result.usage);
      return result;
    };

    if (this.retryOptions === undefined) return call(1);

    return withRetry(call, {
      signal: context.signal,
      onRetry: (retryContext) => {
        const event: RetryEvent = {
          runId: context.runId,
          iteration,
          attempt: retryContext.attempt,
          remaining: retryContext.remaining,
          delayMs: retryContext.delayMs,
          error: retryContext.error,
          scope: 'model',
        };
        emit('retry', event);
        void hook('onRetry', event);
      },
      ...this.retryOptions,
    });
  }

  /**
   * Stream a completion, forwarding deltas as events, and reconstruct a single
   * {@link CompletionResult}. Falls back to buffering if the stream errors
   * before any text arrived.
   */
  private async completeStreaming(
    request: CompletionRequest,
    iteration: number,
    context: RunContext<TState>,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
    callOptions: { signal: AbortSignal; timeoutMs?: number },
  ): Promise<CompletionResult> {
    let result: CompletionResult | undefined;
    const stream = this.provider.stream(request, callOptions);

    // Raced against the abort signal so a provider that ignores `signal`
    // (a buggy HTTP client, a wedged socket) can still be cancelled.
    for await (const event of raceIterableWithAbort(stream, context.signal)) {
      switch (event.type) {
        case 'text-delta': {
          const payload: ModelDeltaEvent = {
            runId: context.runId,
            iteration,
            text: event.text,
          };
          emit('model:delta', payload);
          await hook('onModelDelta', payload);
          break;
        }
        case 'reasoning-delta': {
          const payload: ModelDeltaEvent = {
            runId: context.runId,
            iteration,
            text: event.text,
            reasoning: true,
          };
          emit('model:delta', payload);
          await hook('onModelDelta', payload);
          break;
        }
        case 'finish':
          result = event.result;
          break;
        case 'error':
          throw event.error;
        default:
          break;
      }
    }

    if (result !== undefined) return result;

    // Some providers advertise streaming but emit a single buffered chunk; if
    // nothing came back, do one clean non-streaming call.
    return this.provider.complete(request, callOptions);
  }

  private buildRequest(
    messages: readonly ModelMessage[],
    registry: ToolRegistry<TState>,
    outputSchema: Schema<unknown> | undefined,
  ): CompletionRequest {
    const toolSpecs = registry.toSpecs();
    const tools =
      toolSpecs.length > 0 && this.provider.capabilities.tools
        ? { tools: toolSpecs }
        : {};
    const format = this.resolveResponseFormat(outputSchema, toolSpecs.length);

    return {
      model: this.model,
      messages,
      ...tools,
      ...(format === undefined ? {} : { responseFormat: format }),
      ...definedOnly(this.modelOptions),
    };
  }

  /**
   * Decide how to ask for structured output.
   *
   * Native JSON-schema mode is only requested when no tools are configured:
   * several providers reject tool use combined with a forced response format,
   * and tool reliability matters more than output enforcement. In that case we
   * fall back to instructing the model and validating locally.
   */
  private resolveResponseFormat(
    schema: Schema<unknown> | undefined,
    toolCount: number,
  ): CompletionRequest['responseFormat'] | undefined {
    if (schema === undefined) return undefined;

    if (toolCount === 0 && this.provider.capabilities.strictJsonSchema) {
      return {
        type: 'json_schema',
        name: this.outputConfig?.name ?? 'response',
        schema: toJsonSchema(schema),
        strict: this.outputConfig?.strict ?? true,
      };
    }

    // Tools are present (or the provider lacks strict schemas): ask for JSON
    // where supported and let local validation plus the repair loop do the
    // rest. Forcing a response format alongside tool use is rejected outright
    // by several providers, so tool reliability wins.
    if (this.provider.capabilities.jsonMode) return { type: 'json_object' };
    return undefined;
  }

  // -------------------------------------------------------------------------
  // Prompt assembly
  // -------------------------------------------------------------------------

  private async composeSystemPrompt(
    input: string,
    context: RunContext<TState>,
    options: RunOptions<unknown, TState>,
    recalled: readonly MemorySearchResult[],
  ): Promise<string> {
    const base =
      typeof this.instructions === 'function'
        ? await this.instructions({ input, runId: context.runId, state: context.state })
        : this.instructions;

    const sections = [options.instructions ?? base];

    if (options.system !== undefined && options.system !== '')
      sections.push(options.system);

    if (recalled.length > 0) {
      sections.push(
        [
          `${this.longTerm.heading}:`,
          ...recalled.map((hit) => `- (${hit.score.toFixed(2)}) ${hit.record.content}`),
        ].join('\n'),
      );
    }

    const plan = this.planTracker.plan;
    if (this.plannerEnabled && plan !== undefined) {
      sections.push(this.plannerConfig.render?.(plan) ?? renderPlan(plan));
    }

    if (this.outputConfig?.instructions !== undefined) {
      sections.push(this.outputConfig.instructions);
    }
    if (options.outputInstructions !== undefined) {
      sections.push(options.outputInstructions);
    }

    const assembled = sections.filter((section) => section.trim() !== '').join('\n\n');

    const overridden = await this.hooks.onSystemPrompt?.({
      runId: context.runId,
      input,
      state: context.state,
      defaultPrompt: assembled,
    });

    return overridden ?? assembled;
  }

  private async recallMemories(
    input: string,
    context: RunContext<TState>,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
  ): Promise<readonly MemorySearchResult[]> {
    if (this.longTermMemory === undefined || !this.longTerm.recall) return [];

    const emitMemory = async (count: number, detail?: string): Promise<void> => {
      const event: MemoryEvent = {
        runId: context.runId,
        operation: 'recall',
        count,
        ...(detail === undefined ? {} : { detail }),
      };
      emit('memory', event);
      await hook('onMemory', event);
    };

    try {
      const results = await this.longTermMemory.search(input, {
        limit: this.longTerm.limit,
        minScore: this.longTerm.minScore,
        roles: this.longTerm.roles,
      });
      await emitMemory(results.length);
      return results;
    } catch (error) {
      this.logger.warn('long-term recall failed', { error });
      await emitMemory(0, 'failed');
      return [];
    }
  }

  private async storeMemories(
    input: string,
    output: string,
    runId: string,
    emit: <K extends AgentEventName>(event: K, payload: AgentEventMap[K][0]) => void,
    hook: (name: keyof AgentHooks, payload: unknown) => Promise<unknown>,
  ): Promise<void> {
    if (this.longTermMemory === undefined) return;
    const now = Date.now();
    const records: MemoryRecord[] = [
      { id: createId('mem'), role: 'user', content: input, createdAt: now },
    ];
    if (output.trim() !== '') {
      records.push({
        id: createId('mem'),
        role: 'assistant',
        content: output,
        createdAt: now,
      });
    }
    try {
      await this.longTermMemory.add(records);
      const event: MemoryEvent = { runId, operation: 'store', count: records.length };
      emit('memory', event);
      await hook('onMemory', event);
    } catch (error) {
      this.logger.warn('long-term store failed', { error });
    }
  }

  // -------------------------------------------------------------------------
  // Planning
  // -------------------------------------------------------------------------

  /** Whether this run should ask the model for a plan. */
  private shouldPlan(): boolean {
    if (!this.plannerEnabled) return false;
    if (this.plannerConfig.planOnRun === false) return false;
    // A persisted plan is reused; regenerating it would cost a call for nothing.
    return !(this.plannerConfig.persist === true && this.planTracker.plan !== undefined);
  }

  private async buildPlan(
    input: string,
    runId: string,
    signal: AbortSignal,
  ): Promise<Plan> {
    const custom = this.plannerConfig.createPlan;
    if (custom !== undefined) {
      return custom(input, { runId, signal, state: this.state });
    }

    return createPlanWithModel(
      async (messages, responseFormat) => {
        const result = await this.provider.complete(
          {
            model: this.model,
            messages,
            temperature: 0.2,
            ...(responseFormat === undefined ? {} : { responseFormat }),
          },
          { signal },
        );
        return result.message.content;
      },
      {
        input,
        runId,
        signal,
        maxSteps: this.plannerConfig.maxSteps ?? 8,
        supportsStrictSchema: this.provider.capabilities.strictJsonSchema,
      },
    );
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private resolveProvider(config: AgentConfig<TState>): {
    provider: ModelProvider;
    model: string;
  } {
    if (config.provider !== undefined) {
      return {
        provider: config.provider,
        model: config.model ?? config.provider.defaultModel,
      };
    }
    const ref = config.model as string;
    const provider = createProvider(ref, config.providerOptions ?? {});
    const separator = ref.indexOf(':');
    const requested = separator > 0 ? ref.slice(separator + 1) : ref;
    return { provider, model: requested === '' ? provider.defaultModel : requested };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type SystemPromptFn<TState> = (context: {
  input: string;
  runId: string;
  state: TState;
}) => string | Promise<string>;

function mergeLimits(
  base: ResolvedLimits,
  overrides: AgentLimits | undefined,
): ResolvedLimits {
  if (overrides === undefined) return base;
  return { ...base, ...definedOnly(overrides) };
}

function definedOnly<T extends object>(input: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) out[key] = value;
  }
  return out as Partial<T>;
}

function buildStep(
  iteration: number,
  request: CompletionRequest,
  response: CompletionResult,
  toolCalls: readonly ToolCall[],
  toolResults: readonly ToolMessage[],
  startedAt: number,
  text: string,
): AgentStep {
  return {
    iteration,
    request,
    response,
    toolCalls,
    toolResults,
    usage: response.usage,
    durationMs: Date.now() - startedAt,
    text,
  };
}

function validateStructured<T>(
  text: string,
  schema: Schema<T>,
): { ok: true; data: T } | { ok: false; issues: readonly SchemaIssue[] } {
  const parsed = extractJson(text);
  if (parsed === undefined) {
    return { ok: false, issues: [{ path: [], message: 'response was not valid JSON' }] };
  }
  const result = schema.safeParse(parsed);
  return result.success
    ? { ok: true, data: result.data }
    : { ok: false, issues: result.error.issues };
}

function describeStop(
  stopReason: StopReason,
  limits: ResolvedLimits,
  budget: BudgetTracker,
): string {
  switch (stopReason) {
    case 'max_iterations':
      return `Agent stopped after reaching maxIterations (${limits.maxIterations}).`;
    case 'max_tokens':
      return `Agent stopped after reaching maxTotalTokens (${String(limits.maxTotalTokens)}). Used ${budget.totalTokens}.`;
    case 'max_tool_calls':
      return `Agent stopped after reaching maxToolCalls (${String(limits.maxToolCalls)}).`;
    case 'timeout':
      return `Agent stopped after exceeding timeoutMs (${String(limits.timeoutMs)}ms).`;
    case 'aborted':
      return 'Agent run was aborted.';
    default:
      return `Agent run ended with reason "${stopReason}".`;
  }
}

/** Map an error code onto the lifecycle area it originated from. */
function scopeForError(error: AgentError): ErrorEvent['scope'] {
  if (error.code.startsWith('TOOL_')) return 'tool';
  if (error.code === ErrorCode.MEMORY) return 'memory';
  if (error.code === ErrorCode.VALIDATION) return 'output';
  if (error.code.startsWith('PROVIDER_')) return 'model';
  return 'agent';
}

function normalizeRunError(error: unknown, signal: AbortSignal): AgentError {
  if (error instanceof AgentError) return error;
  if (isAbortLikeError(error)) return new AbortError();
  if (signal.aborted) return new AbortError();
  return toAgentError(error, 'Agent run failed');
}

/**
 * Translate an observability event into the flat stream union, or `undefined`
 * when the event has no stream equivalent.
 */
function mapToStreamEvent(
  event: AgentEventName,
  payload: unknown,
): AgentStreamEvent | undefined {
  const data = payload as Record<string, unknown>;
  switch (event) {
    case 'agent:start':
      return {
        type: 'start',
        runId: String(data['runId']),
        agentName: String(data['agentName']),
        model: String(data['model']),
        input: String(data['input']),
        tools: (data['tools'] as string[]) ?? [],
      };
    case 'model:delta': {
      const reasoning = data['reasoning'] === true;
      const delta = asText(data['text']);
      const base = {
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        text: delta,
      };
      return reasoning
        ? { type: 'reasoning-delta', ...base }
        : { type: 'text-delta', ...base };
    }
    case 'tool:start':
      return {
        type: 'tool-call',
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        name: asText(data['toolName']),
        args: data['args'],
      };
    case 'tool:end':
      return {
        type: 'tool-result',
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        name: asText(data['toolName']),
        content: asText((data['result'] as { content?: unknown } | undefined)?.content),
        isError: false,
        durationMs: Number(data['durationMs'] ?? 0),
      };
    case 'tool:error':
      return {
        type: 'tool-result',
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        name: asText(data['toolName']),
        content: asText(
          (data['error'] as { message?: unknown } | undefined)?.message,
          'tool failed',
        ),
        isError: true,
        durationMs: Number(data['durationMs'] ?? 0),
      };
    case 'iteration:end':
      return {
        type: 'iteration',
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        toolCallCount: Number(data['toolCallCount'] ?? 0),
        usage: (data['usage'] as Usage) ?? {},
        durationMs: Number(data['durationMs'] ?? 0),
      };
    case 'plan':
      return {
        type: 'plan',
        runId: String(data['runId']),
        plan: data['plan'] as Plan,
        completed: Number(data['completed'] ?? 0),
      };
    case 'retry':
      return {
        type: 'retry',
        runId: String(data['runId']),
        iteration: Number(data['iteration'] ?? 0),
        delayMs: Number(data['delayMs'] ?? 0),
        error: data['error'] as AgentError,
      };
    case 'error':
      return {
        type: 'error',
        runId: String(data['runId']),
        error: data['error'] as AgentError,
        fatal: data['fatal'] === true,
        iteration: Number(data['iteration'] ?? 0),
      };
    default:
      return undefined;
  }
}

/**
 * Narrow an `unknown` event field to display text.
 *
 * Stream events are rendered by UI code, so a stray object must never surface
 * as the literal `"[object Object]"`.
 */
function asText(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return fallback;
  return stringifyToolResult(value);
}
