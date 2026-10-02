/**
 * `@agentloom/core` — a dependency-light toolkit for building AI agents.
 *
 * Everything the library offers is exported from this module. Start from
 * {@link Agent}, or assemble the individual pieces (providers, tools, memory)
 * yourself.
 */

export { Agent } from './agent/agent.js';
export { BudgetTracker } from './agent/budget.js';
export { AgentStream } from './agent/stream.js';
export { DEFAULT_LIMITS, resolveLimits } from './agent/result.js';
export { VETOING_HOOKS } from './agent/hooks.js';
export {
  buildPlanPrompt,
  createPlanWithModel,
  normalizePlan,
  PLAN_JSON_SCHEMA,
  PlanTracker,
  renderPlan,
} from './agent/planner.js';

export type {
  AgentConfig,
  LongTermMemoryConfig,
  OutputConfig,
  PlannerConfig,
  RunOptions,
  SystemPromptProvider,
} from './agent/config.js';
export type {
  AgentEndEvent,
  AgentEventListener,
  AgentEventMap,
  AgentEventName,
  AgentStartEvent,
  BaseRunEvent,
  ErrorEvent,
  IterationEndEvent,
  IterationStartEvent,
  MemoryEvent,
  ModelDeltaEvent,
  ModelEndEvent,
  ModelStartEvent,
  OutputInvalidEvent,
  PlanEvent,
  PromptEvent,
  RetryEvent,
  ToolEndEvent,
  ToolErrorEvent,
  ToolStartEvent,
} from './agent/events.js';
export type { AgentHooks, HookName, HookRegistry } from './agent/hooks.js';
export type { PlanContext } from './agent/planner-types.js';
export type { Plan, PlanStep, PlanStepStatus } from './agent/planner.js';
export type {
  AgentLimits,
  AgentResult,
  AgentStep,
  ResolvedLimits,
  StopReason,
} from './agent/result.js';
export type { AgentStreamEvent } from './agent/stream.js';

// --- Schema -----------------------------------------------------------------
export {
  anySchema,
  formatIssues,
  isSchema,
  schemaFromJsonSchema,
  SchemaValidationError,
} from './schema.js';
export type {
  Infer,
  Schema,
  SchemaError,
  SchemaFailure,
  SchemaIssue,
  SchemaResult,
  SchemaSuccess,
} from './schema.js';

// --- Errors -----------------------------------------------------------------
export {
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
  ToolError,
  ToolExecutionError,
  ToolNotFoundError,
  ToolTimeoutError,
  ToolValidationError,
  ValidationError,
} from './errors.js';
export type {
  AgentErrorOptions,
  ErrorCodeValue,
  ProviderErrorOptions,
  ToolErrorOptions,
} from './errors.js';

// --- Providers --------------------------------------------------------------
export { AnthropicProvider } from './providers/anthropic.js';
export {
  addUsage,
  BaseProvider,
  emptyUsage,
  normalizeFinishReason,
} from './providers/base.js';
export { GoogleProvider } from './providers/google.js';
export { HttpClient, parseRetryAfter } from './providers/http.js';
export { OllamaProvider } from './providers/ollama.js';
export { OpenAIProvider } from './providers/openai.js';
export { toJsonSchema, toParametersSchema } from './providers/json-schema.js';
export {
  createProvider,
  createProviderForModel,
  defaultProviderRegistry,
  parseModelRef,
  ProviderRegistry,
  registerProvider,
  unregisterProvider,
} from './providers/registry.js';
export { isAssistantMessage, isToolCallMessage } from './providers/types.js';

export type { BaseProviderOptions } from './providers/base.js';
export type {
  FetchLike,
  FetchResponseLike,
  HttpClientOptions,
  HttpRequest,
  HttpResponse,
  SseEvent,
} from './providers/http.js';
export type { ParsedModelRef, ProviderFactory } from './providers/registry.js';
export type {
  AssistantMessage,
  CompletionOptions,
  CompletionRequest,
  CompletionResult,
  FinishReason,
  ModelMessage,
  ModelProvider,
  ProviderCallOptions,
  ProviderCapabilities,
  ResponseFormat,
  StreamEvent,
  StreamOptions,
  SystemMessage,
  ToolCall,
  ToolMessage,
  ToolSpec,
  Usage,
  UserMessage,
} from './providers/types.js';

// --- Tools ------------------------------------------------------------------
export {
  createCalculatorTool,
  evaluateExpression,
  supportedFunctions,
} from './tools/builtin/calculator.js';
export { createDateTimeTool, parseDuration } from './tools/builtin/datetime.js';
export {
  createFetchUrlTool,
  createHttpTool,
  isUrlAllowed,
} from './tools/builtin/http.js';
export { createFileSystemTool, resolveInsideRoot } from './tools/builtin/filesystem.js';
export { createSleepTool } from './tools/builtin/sleep.js';
export { ToolExecutor } from './tools/executor.js';
export { defineTool, toToolRegistry, tool, ToolRegistry } from './tools/registry.js';
export { isErrorResult } from './tools/types.js';

export type {
  ToolExecutorHooks,
  ToolExecutorOptions,
  ToolExecutorRunOptions,
  ToolExecutionMode,
  ToolFailureInfo,
  ToolLifecycleInfo,
  ToolSuccessInfo,
} from './tools/executor.js';
export type {
  ApprovalHandler,
  ApprovalRequest,
  PolicyContext,
  ToolContext,
  ToolDecision,
  ToolDefinition,
  ToolExecute,
  ToolPolicy,
  ToolRecord,
  ToolResultPayload,
} from './tools/types.js';
export type {
  CalculatorArgs,
  CalculatorResult,
  CalculatorToolOptions,
} from './tools/builtin/calculator.js';
export type {
  DateTimeAction,
  DateTimeArgs,
  DateTimeResult,
} from './tools/builtin/datetime.js';
export type {
  HttpAction,
  HttpToolArgs,
  HttpToolOptions,
  HttpToolResult,
} from './tools/builtin/http.js';
export type {
  FileSystemAction,
  FileSystemLike,
  FileSystemToolArgs,
  FileSystemToolOptions,
  FileSystemToolResult,
} from './tools/builtin/filesystem.js';
export type { SleepToolArgs, SleepToolResult } from './tools/builtin/sleep.js';

// --- Memory -----------------------------------------------------------------
export {
  assistantMessage,
  createSummaryCompactor,
  defaultTokenCounter,
  InMemoryConversationMemory,
  summaryMessage,
  toolResultMessage,
  transcriptOf,
} from './memory/conversation.js';
export { createFileLongTermMemory, InMemoryLongTermMemory } from './memory/long-term.js';

export type { ConversationMemoryState } from './memory/conversation.js';
export type {
  BuildContext,
  ConversationMemory,
  ConversationMemoryOptions,
  MemoryCompactor,
  MemoryStrategy,
  TokenCountable,
  TokenCounter,
  TrimContext,
} from './memory/types.js';
export type {
  FileLongTermMemoryOptions,
  InMemoryLongTermMemoryOptions,
  LongTermMemory,
  MemoryRecord,
  MemorySearchOptions,
  MemorySearchResult,
  NewMemoryRecord,
} from './memory/long-term.js';

// --- Utilities --------------------------------------------------------------
export {
  abortPromise,
  combineSignals,
  deepClone,
  deferred,
  mapWithConcurrency,
  raceIterableWithAbort,
  raceWithAbort,
  sleep,
  throwIfAborted,
  withTimeout,
} from './utils/async.js';
export { TypedEventEmitter } from './utils/emitter.js';
export { createId } from './utils/id.js';
export {
  compact,
  createConsoleLogger,
  LOG_LEVELS,
  MemoryLogSink,
  noopLogger,
  stringifyToolResult,
  toSnakeCase,
  toTitleCase,
  tokenize,
  truncate,
} from './utils/index.js';
export { AsyncQueue } from './utils/queue.js';
export { computeBackoffDelay, resolveRetryPolicy, withRetry } from './utils/retry.js';
export {
  estimateMessageTokens,
  estimateTokens,
  estimateValueTokens,
} from './utils/tokens.js';
export { extractJson, normalizeText } from './utils/text.js';

export type { Deferred } from './utils/async.js';
export type { EventMap, Unsubscribe } from './utils/emitter.js';
export type {
  ConsoleLoggerOptions,
  LogFields,
  LogLevel,
  Logger,
  MemoryLoggerOptions,
} from './utils/logger.js';
export type { ResolvedRetryPolicy, RetryContext, RetryOptions } from './utils/retry.js';
export type { TokenCountableMessage } from './utils/tokens.js';
