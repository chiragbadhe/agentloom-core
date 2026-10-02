import { ProviderResponseError } from '../errors.js';
import { createId } from '../utils/id.js';
import { extractJson } from '../utils/text.js';
import {
  BaseProvider,
  type BaseProviderOptions,
  normalizeFinishReason,
  resolveApiKey,
} from './base.js';
import { parseSseStream } from './http.js';
import { toParametersSchema } from './json-schema.js';
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

export interface AnthropicProviderOptions extends BaseProviderOptions {
  readonly apiKey?: string;
  /** Default `https://api.anthropic.com`. */
  readonly baseURL?: string;
  /** API version header. Required by the Messages API. */
  readonly apiVersion?: string;
  readonly model?: string;
  readonly id?: string;
  readonly name?: string;
}

const DEFAULT_BASE_URL = 'https://api.anthropic.com';
const DEFAULT_API_VERSION = '2023-06-01';
const DEFAULT_MAX_TOKENS = 4096;

/**
 * Anthropic Messages API.
 *
 * Notable differences from the OpenAI shape, all handled here:
 * - system prompts are a top-level field, not a message;
 * - `max_tokens` is mandatory;
 * - tool results are `user` messages containing `tool_result` blocks;
 * - tool schemas are passed as `input_schema`.
 */
export class AnthropicProvider extends BaseProvider {
  readonly id: string;
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities = {
    tools: true,
    parallelToolCalls: true,
    streaming: true,
    systemMessages: true,
    jsonMode: false,
    strictJsonSchema: false,
    vision: true,
    promptCaching: true,
  };

  private readonly apiKey: string | undefined;
  private readonly baseURL: string;
  private readonly apiVersion: string;

  constructor(options: AnthropicProviderOptions = {}) {
    const baseURL = (options.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    super({
      fetch: options.fetch,
      logger: options.logger,
      retry: options.retry,
      defaultTimeoutMs: options.defaultTimeoutMs,
      defaultHeaders: { 'content-type': 'application/json', ...options.headers },
    });
    this.id = options.id ?? 'anthropic';
    this.name = options.name ?? 'Anthropic';
    this.baseURL = baseURL;
    this.apiVersion = options.apiVersion ?? DEFAULT_API_VERSION;
    this.apiKey = resolveApiKey(options.apiKey, ['ANTHROPIC_API_KEY']);
    this.defaultModel = options.model ?? 'claude-sonnet-4-5';
  }

  protected override async doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const fallbackId = createId('resp');
    const response = await this.http.json<AnthropicResponse>({
      url: `${this.baseURL}/v1/messages`,
      method: 'POST',
      headers: this.authHeaders(options),
      body: buildBody(request),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const text: string[] = [];
    const toolCalls: ToolCall[] = [];

    for (const block of response.content ?? []) {
      if (block.type === 'text') text.push(block.text ?? '');
      else if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id ?? `call_${toolCalls.length}`,
          name: block.name ?? '',
          arguments: block.input ?? {},
          rawArguments: JSON.stringify(block.input ?? {}),
        });
      }
    }

    const message: AssistantMessage = {
      role: 'assistant',
      content: text.join(''),
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    };

    return {
      message,
      finishReason: normalizeFinishReason(response.stop_reason),
      usage: parseUsage(response.usage),
      responseId: response.id ?? fallbackId,
      providerId: this.id,
      model: response.model ?? request.model,
      latencyMs: Date.now() - startedAt,
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
    const fallbackId = createId('resp');
    const response = await this.http.expectOk({
      url: `${this.baseURL}/v1/messages`,
      method: 'POST',
      headers: { ...this.authHeaders(options), accept: 'text/event-stream' },
      body: { ...buildBody(request), stream: true },
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const stream = response.stream();
    if (stream === null) {
      throw new ProviderResponseError(
        'Anthropic response did not expose a readable stream',
        {
          providerId: this.id,
          model: request.model,
        },
      );
    }

    yield { type: 'start', model: request.model, responseId: fallbackId };

    let text = '';
    let responseId = fallbackId;
    let model = request.model;
    let finishReason: CompletionResult['finishReason'] = 'other';
    let usage: Usage = {};
    /** `input_json_delta` fragments are keyed by block index. */
    const drafts = new Map<number, { id: string; name: string; json: string }>();

    for await (const frame of parseSseStream(stream)) {
      if (frame.event === 'error') {
        const payload = safeParse(frame.data);
        const message =
          (payload as { error?: { message?: string } } | undefined)?.error?.message ??
          frame.data;
        throw new ProviderResponseError(`Anthropic stream error: ${message}`, {
          providerId: this.id,
          model: request.model,
        });
      }

      const payload = safeParse(frame.data) as AnthropicStreamEvent | undefined;
      if (payload === undefined) continue;

      switch (payload.type) {
        case 'message_start': {
          const message = (payload as { message?: { id?: string; model?: string } })
            .message;
          responseId = message?.id ?? responseId;
          model = message?.model ?? model;
          usage = mergeUsage(
            usage,
            parseUsage(
              (payload as { message?: { usage?: unknown } }).message?.usage as never,
            ),
          );
          break;
        }
        case 'content_block_start': {
          const block = (
            payload as { index?: number; content_block?: AnthropicContentBlock }
          ).content_block;
          if (block?.type === 'tool_use') {
            const index = (payload as { index?: number }).index ?? 0;
            drafts.set(index, { id: block.id ?? '', name: block.name ?? '', json: '' });
          }
          break;
        }
        case 'content_block_delta': {
          const delta = (
            payload as {
              index?: number;
              delta?: {
                type?: string;
                text?: string;
                thinking?: string;
                partial_json?: string;
              };
            }
          ).delta;
          if (delta?.type === 'text_delta' && delta.text) {
            text += delta.text;
            yield { type: 'text-delta', text: delta.text, responseId };
          } else if (delta?.type === 'thinking_delta' && delta.thinking) {
            yield { type: 'reasoning-delta', text: delta.thinking, responseId };
          } else if (
            delta?.type === 'input_json_delta' &&
            delta.partial_json !== undefined
          ) {
            const index = (payload as { index?: number }).index ?? 0;
            const draft = drafts.get(index) ?? { id: '', name: '', json: '' };
            draft.json += delta.partial_json;
            drafts.set(index, draft);
          }
          break;
        }
        case 'content_block_stop':
          break;
        case 'message_delta': {
          const delta = payload as {
            delta?: { stop_reason?: string };
            usage?: { output_tokens?: number };
          };
          if (delta.delta?.stop_reason) {
            finishReason = normalizeFinishReason(delta.delta.stop_reason);
          }
          if (delta.usage?.output_tokens !== undefined) {
            usage = { ...usage, outputTokens: delta.usage.output_tokens };
          }
          break;
        }
        case 'message_stop':
          break;
        default:
          break;
      }
    }

    const toolCalls: ToolCall[] = [...drafts.entries()]
      .sort(([a], [b]) => a - b)
      .map(([index, draft]) => ({
        id: draft.id || `${responseId}_call_${index}`,
        name: draft.name,
        arguments: parseToolInput(draft.json, draft.name, this.id, model),
        rawArguments: draft.json,
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
        responseId,
        providerId: this.id,
        model,
        latencyMs: Date.now() - startedAt,
        raw: { streamed: true },
      },
    };
  }

  private authHeaders(options: ProviderCallOptions): Record<string, string> {
    const headers: Record<string, string> = { ...options.headers };
    if (this.apiKey !== undefined) headers['x-api-key'] = this.apiKey;
    headers['anthropic-version'] = this.apiVersion;
    return headers;
  }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

interface AnthropicContentBlock {
  type?: string;
  text?: string;
  id?: string;
  name?: string;
  input?: unknown;
}

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

interface AnthropicResponse {
  id?: string;
  model?: string;
  content?: AnthropicContentBlock[];
  stop_reason?: string;
  usage?: AnthropicUsage;
}

type AnthropicStreamEvent = { type?: string } & Record<string, unknown>;

function buildBody(request: CompletionRequest): Record<string, unknown> {
  const system = request.messages
    .filter(
      (message): message is Extract<ModelMessage, { role: 'system' }> =>
        message.role === 'system',
    )
    .map((message) => message.content)
    .join('\n\n');

  const body: Record<string, unknown> = {
    model: request.model,
    max_tokens: request.maxTokens ?? DEFAULT_MAX_TOKENS,
    messages: request.messages
      .filter((message) => message.role !== 'system')
      .map(toWireMessage),
  };

  if (system !== '') body.system = system;
  if (request.tools && request.tools.length > 0) {
    body.tools = request.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: toParametersSchema(tool.parameters),
    }));
  }
  if (request.temperature !== undefined) body.temperature = request.temperature;
  if (request.topP !== undefined) body.top_p = request.topP;
  if (request.topK !== undefined) body.top_k = request.topK;
  if (request.stopSequences !== undefined) body.stop_sequences = request.stopSequences;
  if (request.metadata !== undefined) {
    const userId = request.metadata['userId'];
    body.metadata = { user_id: typeof userId === 'string' ? userId : '' };
  }

  // Anthropic has no native JSON mode; a schema hint is the best we can do and
  // the agent validates the result anyway.
  if (request.responseFormat && request.responseFormat.type !== 'text') {
    const instruction = responseFormatInstruction(request);
    body.system = [system, instruction].filter(Boolean).join('\n\n');
  }

  return body;
}

function responseFormatInstruction(request: CompletionRequest): string {
  if (request.responseFormat?.type === 'json_schema') {
    return [
      'Respond with a single JSON object that validates against this JSON Schema.',
      'Output raw JSON only — no prose, no markdown fences.',
      JSON.stringify(request.responseFormat.schema),
    ].join('\n');
  }
  return 'Respond with a single valid JSON object. Output raw JSON only — no prose, no markdown fences.';
}

function toWireMessage(message: ModelMessage): Record<string, unknown> {
  switch (message.role) {
    case 'system':
      return { role: 'user', content: message.content };
    case 'user':
      return { role: 'user', content: [{ type: 'text', text: message.content }] };
    case 'assistant': {
      const content: Record<string, unknown>[] = [];
      if (message.content) content.push({ type: 'text', text: message.content });
      for (const call of message.toolCalls ?? []) {
        content.push({
          type: 'tool_use',
          id: call.id,
          name: call.name,
          input: call.arguments ?? {},
        });
      }
      // Anthropic rejects empty content arrays.
      if (content.length === 0) content.push({ type: 'text', text: '' });
      return { role: 'assistant', content };
    }
    case 'tool': {
      const block: Record<string, unknown> = {
        type: 'tool_result',
        tool_use_id: message.toolCallId,
        content: message.content,
      };
      if (message.isError) block.is_error = true;
      return { role: 'user', content: [block] };
    }
  }
}

function parseUsage(usage: AnthropicUsage | undefined): Usage {
  if (!usage) return {};
  const cached =
    usage.cache_read_input_tokens === undefined &&
    usage.cache_creation_input_tokens === undefined
      ? undefined
      : (usage.cache_read_input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0);
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    totalTokens:
      usage.input_tokens === undefined && usage.output_tokens === undefined
        ? undefined
        : (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
    cachedInputTokens: cached,
  };
}

function mergeUsage(a: Usage, b: Usage): Usage {
  return { ...a, ...definedOnly(b) };
}

function definedOnly(usage: Usage): Usage {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(usage)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function parseToolInput(
  json: string,
  toolName: string,
  providerId: string,
  model: string,
): unknown {
  if (json.trim() === '') return {};
  const parsed = extractJson(json);
  if (parsed === undefined) {
    throw new ProviderResponseError(`Tool input for "${toolName}" was not valid JSON`, {
      providerId,
      model,
      responseBody: json.slice(0, 500),
    });
  }
  return parsed;
}

function safeParse(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return undefined;
  }
}

/** Create an {@link AnthropicProvider}. */
export function createAnthropicProvider(
  options: AnthropicProviderOptions = {},
): AnthropicProvider {
  return new AnthropicProvider(options);
}
