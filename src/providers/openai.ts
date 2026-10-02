import { ProviderResponseError } from '../errors.js';
import { extractJson } from '../utils/text.js';
import { createId } from '../utils/id.js';
import { parseSseStream } from './http.js';
import { toParametersSchema } from './json-schema.js';
import {
  BaseProvider,
  type BaseProviderOptions,
  normalizeFinishReason,
  resolveApiKey,
} from './base.js';
import type {
  AssistantMessage,
  CompletionRequest,
  CompletionResult,
  ModelMessage,
  ProviderCallOptions,
  ProviderCapabilities,
  StreamEvent,
  ToolCall,
  Usage,
} from './types.js';

export interface OpenAIProviderOptions extends BaseProviderOptions {
  readonly apiKey?: string;
  /** Override for proxies and gateways. Default `https://api.openai.com/v1`. */
  readonly baseURL?: string;
  readonly organization?: string;
  readonly project?: string;
  /** Default model when the agent does not specify one. */
  readonly model?: string;
  /** Provider id/name, e.g. when wrapping an OpenAI-compatible vendor. */
  readonly id?: string;
  readonly name?: string;
}

const DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const DONE_SENTINEL = '[DONE]';

/**
 * OpenAI Chat Completions.
 *
 * Talks HTTP directly through `fetch`, so there is no `openai` SDK dependency
 * to keep in sync. It is also compatible with the many OpenAI-shaped
 * gateways (Azure, Together, Groq, vLLM, OpenRouter, LiteLLM, …) — point
 * `baseURL` at them.
 */
export class OpenAIProvider extends BaseProvider {
  readonly id: string;
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities = {
    tools: true,
    parallelToolCalls: true,
    streaming: true,
    systemMessages: true,
    jsonMode: true,
    strictJsonSchema: true,
    vision: true,
    promptCaching: true,
  };

  private readonly apiKey: string | undefined;
  private readonly baseURL: string;
  private readonly extraHeaders: Record<string, string>;

  constructor(options: OpenAIProviderOptions = {}) {
    const baseURL = (options.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    super({
      fetch: options.fetch,
      logger: options.logger,
      retry: options.retry,
      defaultTimeoutMs: options.defaultTimeoutMs,
      defaultHeaders: { 'content-type': 'application/json', ...options.headers },
    });
    this.id = options.id ?? 'openai';
    this.name = options.name ?? 'OpenAI';
    this.baseURL = baseURL;
    this.apiKey = resolveApiKey(options.apiKey, ['OPENAI_API_KEY']);
    this.defaultModel = options.model ?? 'gpt-4o-mini';
    this.extraHeaders = {
      ...(options.organization ? { 'openai-organization': options.organization } : {}),
      ...(options.project ? { 'openai-project': options.project } : {}),
    };
  }

  protected override async doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const responseId = createId('resp');
    const response = await this.http.json<OpenAIResponse>({
      url: `${this.baseURL}/chat/completions`,
      method: 'POST',
      headers: this.authHeaders(options),
      body: buildRequestBody(request),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const choice = response.choices?.[0];
    if (!choice) {
      throw new ProviderResponseError('OpenAI returned no choices', {
        providerId: this.id,
        model: request.model,
        responseBody: JSON.stringify(response),
      });
    }

    const message = parseAssistantMessage(choice.message, this.id, request.model);
    const usage = parseUsage(response.usage);

    return {
      message,
      finishReason: normalizeFinishReason(choice.finish_reason),
      usage,
      responseId: response.id ?? responseId,
      providerId: this.id,
      model: response.model ?? request.model,
      latencyMs: startedAt === 0 ? 0 : Date.now() - startedAt,
      raw: response,
    };
  }

  protected override doStream(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): AsyncIterable<StreamEvent> {
    return this.streamRequest(request, options);
  }

  private async *streamRequest(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): AsyncGenerator<StreamEvent> {
    const startedAt = Date.now();
    const responseId = createId('resp');
    const body = {
      ...buildRequestBody(request),
      stream: true,
      stream_options: { include_usage: true },
    };

    const response = await this.http.expectOk({
      url: `${this.baseURL}/chat/completions`,
      method: 'POST',
      headers: this.authHeaders(options),
      body,
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const stream = response.stream();
    if (stream === null) {
      throw new ProviderResponseError(
        'OpenAI response did not expose a readable stream',
        {
          providerId: this.id,
          model: request.model,
        },
      );
    }

    yield { type: 'start', model: request.model, responseId };

    let text = '';
    let finishReason: CompletionResult['finishReason'] = 'other';
    let usage: Usage = {};
    let resolvedModel = request.model;
    let resolvedId = responseId;
    /** Tool calls arrive as deltas keyed by `index`. */
    const toolCallDrafts = new Map<number, { id: string; name: string; args: string }>();

    for await (const frame of parseSseStream(stream)) {
      if (frame.data === DONE_SENTINEL) break;
      const payload = parseStreamFrame(frame.data, this.id, request.model);
      if (payload === undefined) continue;

      resolvedModel = payload.model ?? resolvedModel;
      resolvedId = payload.id ?? resolvedId;
      if (payload.usage) usage = parseUsage(payload.usage);

      const choice = payload.choices?.[0];
      if (!choice) continue;

      if (choice.delta?.content) {
        text += choice.delta.content;
        yield { type: 'text-delta', text: choice.delta.content, responseId };
      }
      for (const partial of choice.delta?.tool_calls ?? []) {
        const index = partial.index ?? toolCallDrafts.size;
        const draft = toolCallDrafts.get(index) ?? { id: '', name: '', args: '' };
        if (partial.id) draft.id = partial.id;
        if (partial.function?.name) draft.name += partial.function.name;
        if (partial.function?.arguments) draft.args += partial.function.arguments;
        toolCallDrafts.set(index, draft);
      }
      if (choice.finish_reason)
        finishReason = normalizeFinishReason(choice.finish_reason);
    }

    const toolCalls: ToolCall[] = [...toolCallDrafts.entries()]
      .sort(([a], [b]) => a - b)
      .map(([index, draft]) => ({
        id: draft.id || `${responseId}_call_${index}`,
        name: draft.name,
        arguments: parseArguments(draft.args, draft.name, this.id, request.model),
        rawArguments: draft.args,
      }));

    const message: AssistantMessage = {
      role: 'assistant',
      content: text,
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    };

    yield {
      type: 'finish',
      result: {
        message,
        finishReason:
          toolCalls.length > 0 && finishReason === 'stop' ? 'tool_calls' : finishReason,
        usage,
        responseId: resolvedId,
        providerId: this.id,
        model: resolvedModel,
        latencyMs: Date.now() - startedAt,
        raw: { streamed: true },
      },
    };
  }

  private authHeaders(options: ProviderCallOptions): Record<string, string> {
    const headers: Record<string, string> = { ...this.extraHeaders, ...options.headers };
    if (this.apiKey !== undefined) headers['authorization'] = `Bearer ${this.apiKey}`;
    return headers;
  }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

interface OpenAIToolCallWire {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

interface OpenAIResponse {
  id?: string;
  model?: string;
  choices?: {
    message?: {
      content?: string | null;
      tool_calls?: OpenAIToolCallWire[];
      refusal?: string;
    };
    finish_reason?: string;
  }[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number };
    completion_tokens_details?: { reasoning_tokens?: number };
  };
}

function buildRequestBody(request: CompletionRequest): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: request.model,
    messages: request.messages.map(toWireMessage),
  };

  if (request.tools && request.tools.length > 0) {
    body.tools = request.tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: toParametersSchema(tool.parameters),
      },
    }));
    body.tool_choice = 'auto';
    body.parallel_tool_calls = true;
  }

  if (request.responseFormat && request.responseFormat.type !== 'text') {
    const format = request.responseFormat;
    body.response_format =
      format.type === 'json_schema'
        ? {
            type: 'json_schema',
            json_schema: {
              name: format.name ?? 'response',
              schema: format.schema,
              strict: format.strict ?? false,
            },
          }
        : { type: 'json_object' };
  }

  applySamplingOptions(body, request);
  return body;
}

function applySamplingOptions(
  body: Record<string, unknown>,
  request: CompletionRequest,
): void {
  if (request.temperature !== undefined) body.temperature = request.temperature;
  if (request.topP !== undefined) body.top_p = request.topP;
  if (request.maxTokens !== undefined) body.max_tokens = request.maxTokens;
  if (request.stopSequences !== undefined) body.stop = request.stopSequences;
  if (request.seed !== undefined) body.seed = request.seed;
  if (request.presencePenalty !== undefined)
    body.presence_penalty = request.presencePenalty;
  if (request.frequencyPenalty !== undefined)
    body.frequency_penalty = request.frequencyPenalty;
  if (request.reasoningEffort !== undefined) {
    body.reasoning_effort = request.reasoningEffort;
  }
}

function toWireMessage(message: ModelMessage): Record<string, unknown> {
  switch (message.role) {
    case 'system':
      return { role: 'system', content: message.content };
    case 'user':
      return {
        role: 'user',
        content: message.content,
        ...(message.name ? { name: message.name } : {}),
      };
    case 'assistant': {
      const wire: Record<string, unknown> = {
        role: 'assistant',
        content: message.content === '' ? null : message.content,
      };
      if (message.toolCalls && message.toolCalls.length > 0) {
        wire.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: {
            name: call.name,
            arguments:
              typeof call.arguments === 'string'
                ? call.arguments
                : JSON.stringify(call.arguments ?? {}),
          },
        }));
      }
      return wire;
    }
    case 'tool':
      return {
        role: 'tool',
        tool_call_id: message.toolCallId,
        content: message.content,
      };
  }
}

function parseAssistantMessage(
  message: { content?: string | null; tool_calls?: OpenAIToolCallWire[] } | undefined,
  providerId: string,
  model: string,
): AssistantMessage {
  const toolCalls: ToolCall[] = (message?.tool_calls ?? []).map((call, index) => ({
    id: call.id ?? `call_${index}`,
    name: call.function?.name ?? '',
    arguments: parseArguments(
      call.function?.arguments ?? '{}',
      call.function?.name ?? 'unknown',
      providerId,
      model,
    ),
    rawArguments: call.function?.arguments,
  }));

  return {
    role: 'assistant',
    content: message?.content ?? '',
    ...(toolCalls.length > 0 ? { toolCalls } : {}),
  };
}

function parseArguments(
  raw: string,
  toolName: string,
  providerId: string,
  model: string | undefined,
): unknown {
  const parsed = extractJson(raw);
  if (parsed === undefined) {
    throw new ProviderResponseError(
      `Tool call arguments for "${toolName}" were not valid JSON`,
      { providerId, model, responseBody: raw.slice(0, 500) },
    );
  }
  return parsed;
}

function parseUsage(usage: OpenAIResponse['usage']): Usage {
  if (!usage) return {};
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    cachedInputTokens: usage.prompt_tokens_details?.cached_tokens,
    reasoningTokens: usage.completion_tokens_details?.reasoning_tokens,
  };
}

function parseStreamFrame(
  data: string,
  providerId: string,
  model: string,
): OpenAIResponse & {
  choices?: {
    delta?: {
      content?: string;
      tool_calls?: {
        index?: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }[];
    };
    finish_reason?: string;
  }[];
} {
  try {
    return JSON.parse(data) as OpenAIResponse;
  } catch (error) {
    throw new ProviderResponseError(
      `Failed to parse streaming frame: ${error instanceof Error ? error.message : String(error)}`,
      { providerId, model, responseBody: data.slice(0, 500) },
    );
  }
}

/** Create an {@link OpenAIProvider}. */
export function createOpenAIProvider(
  options: OpenAIProviderOptions = {},
): OpenAIProvider {
  return new OpenAIProvider(options);
}

/**
 * Create a provider for any OpenAI-compatible endpoint (Azure OpenAI,
 * Together, Groq, OpenRouter, vLLM, LM Studio, LiteLLM, …).
 *
 * ```ts
 * const provider = createOpenAICompatibleProvider({
 *   id: 'groq',
 *   baseURL: 'https://api.groq.com/openai/v1',
 *   model: 'llama-3.3-70b-versatile',
 *   apiKey: process.env.GROQ_API_KEY,
 * });
 * ```
 */
export function createOpenAICompatibleProvider(
  options: OpenAIProviderOptions & { id: string },
): OpenAIProvider {
  return new OpenAIProvider(options);
}
