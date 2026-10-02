import { ProviderResponseError } from '../errors.js';
import { createId } from '../utils/id.js';
import { extractJson } from '../utils/text.js';
import { BaseProvider, type BaseProviderOptions, normalizeFinishReason } from './base.js';
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

export interface OllamaProviderOptions extends BaseProviderOptions {
  /** Base URL of the Ollama server. Default `http://localhost:11434`. */
  readonly host?: string;
  readonly model?: string;
  readonly id?: string;
  readonly name?: string;
  /**
   * Ollama streams NDJSON rather than SSE, so the reader waits for each
   * complete line. Default `true`.
   */
  readonly keepAlive?: string;
  readonly headers?: Readonly<Record<string, string>>;
}

const DEFAULT_HOST = 'http://localhost:11434';

/**
 * Ollama (local models through the `/api/chat` endpoint).
 *
 * No API key, no network egress, and it speaks newline-delimited JSON instead
 * of SSE — handled by the custom stream reader below.
 */
export class OllamaProvider extends BaseProvider {
  readonly id: string;
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities = {
    tools: true,
    parallelToolCalls: false,
    streaming: true,
    systemMessages: true,
    jsonMode: true,
    strictJsonSchema: false,
    vision: false,
    promptCaching: false,
  };

  private readonly host: string;
  private readonly keepAlive: string | undefined;

  constructor(options: OllamaProviderOptions = {}) {
    const host = (options.host ?? process.env['OLLAMA_HOST'] ?? DEFAULT_HOST).replace(
      /\/+$/,
      '',
    );
    super({
      fetch: options.fetch,
      logger: options.logger,
      retry: options.retry,
      defaultTimeoutMs: options.defaultTimeoutMs,
      defaultHeaders: { 'content-type': 'application/json', ...options.headers },
    });
    this.id = options.id ?? 'ollama';
    this.name = options.name ?? 'Ollama';
    this.host = host;
    this.defaultModel = options.model ?? 'llama3.2';
    this.keepAlive = options.keepAlive;
  }

  /** List models the local server has pulled. */
  async listModels(options: ProviderCallOptions = {}): Promise<OllamaModel[]> {
    const response = await this.http.json<{ models?: OllamaModel[] }>({
      url: `${this.host}/api/tags`,
      method: 'GET',
      headers: { ...options.headers },
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
    });
    return response.models ?? [];
  }

  protected override async doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const fallbackId = createId('resp');
    const response = await this.http.json<OllamaChatResponse>({
      url: `${this.host}/api/chat`,
      method: 'POST',
      headers: { ...options.headers },
      body: this.buildBody(request, false),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const toolCalls: ToolCall[] = (response.message?.tool_calls ?? []).map(
      (call, index) => ({
        id: `call_${index}`,
        name: call.function?.name ?? '',
        arguments: parseToolArguments(call.function?.arguments, this.id, request.model),
        rawArguments:
          typeof call.function?.arguments === 'string'
            ? call.function.arguments
            : JSON.stringify(call.function?.arguments ?? {}),
      }),
    );

    const message: AssistantMessage = {
      role: 'assistant',
      content: response.message?.content ?? '',
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    };

    return {
      message,
      finishReason:
        toolCalls.length > 0 ? 'tool_calls' : normalizeFinishReason(response.done_reason),
      usage: parseUsage(response),
      responseId: fallbackId,
      providerId: this.id,
      model: request.model,
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
    const responseId = createId('resp');
    const response = await this.http.expectOk({
      url: `${this.host}/api/chat`,
      method: 'POST',
      headers: { ...options.headers },
      body: this.buildBody(request, true),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const stream = response.stream();
    if (stream === null) {
      throw new ProviderResponseError(
        'Ollama response did not expose a readable stream',
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
    const toolCalls: ToolCall[] = [];

    for await (const line of parseNdjson(stream)) {
      const payload = safeParse(line) as OllamaChatResponse | undefined;
      if (payload === undefined) continue;
      if (payload.error) {
        throw new ProviderResponseError(`Ollama error: ${payload.error}`, {
          providerId: this.id,
          model: request.model,
        });
      }

      const delta = payload.message?.content;
      if (delta) {
        text += delta;
        yield { type: 'text-delta', text: delta, responseId };
      }
      for (const call of payload.message?.tool_calls ?? []) {
        toolCalls.push({
          id: `call_${toolCalls.length}`,
          name: call.function?.name ?? '',
          arguments: parseToolArguments(call.function?.arguments, this.id, request.model),
        });
        const last = toolCalls[toolCalls.length - 1]!;
        yield { type: 'tool-call', call: last, responseId };
      }
      if (payload.done) {
        finishReason =
          toolCalls.length > 0
            ? 'tool_calls'
            : normalizeFinishReason(payload.done_reason);
        usage = parseUsage(payload);
      }
    }

    const message: AssistantMessage = {
      role: 'assistant',
      content: text,
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    };

    yield {
      type: 'finish',
      result: {
        message,
        finishReason,
        usage,
        responseId,
        providerId: this.id,
        model: request.model,
        latencyMs: Date.now() - startedAt,
        raw: { streamed: true },
      },
    };
  }

  private buildBody(
    request: CompletionRequest,
    stream: boolean,
  ): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: request.model,
      messages: request.messages.map(toWireMessage),
      stream,
    };
    if (this.keepAlive !== undefined) body.keep_alive = this.keepAlive;

    const options: Record<string, unknown> = {};
    if (request.temperature !== undefined) options.temperature = request.temperature;
    if (request.topP !== undefined) options.top_p = request.topP;
    if (request.topK !== undefined) options.top_k = request.topK;
    if (request.maxTokens !== undefined) options.num_predict = request.maxTokens;
    if (request.stopSequences !== undefined) options.stop = request.stopSequences;
    if (request.seed !== undefined) options.seed = request.seed;
    if (Object.keys(options).length > 0) body.options = options;

    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools.map((tool) => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: toParametersSchema(tool.parameters),
        },
      }));
    }

    if (request.responseFormat?.type === 'json_schema') {
      body.format = request.responseFormat.schema;
    } else if (request.responseFormat?.type === 'json_object') {
      body.format = 'json';
    }

    return body;
  }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

export interface OllamaModel {
  name: string;
  model?: string;
  size?: number;
  digest?: string;
  modified_at?: string;
}

interface OllamaChatResponse {
  model?: string;
  created_at?: string;
  done?: boolean;
  done_reason?: string;
  error?: string;
  message?: {
    role?: string;
    content?: string;
    tool_calls?: {
      function?: { name?: string; arguments?: unknown };
    }[];
  };
  prompt_eval_count?: number;
  eval_count?: number;
}

function toWireMessage(message: ModelMessage): Record<string, unknown> {
  switch (message.role) {
    case 'system':
    case 'user':
      return { role: message.role, content: message.content };
    case 'assistant': {
      const wire: Record<string, unknown> = {
        role: 'assistant',
        content: message.content,
      };
      if (message.toolCalls && message.toolCalls.length > 0) {
        wire.tool_calls = message.toolCalls.map((call) => ({
          function: { name: call.name, arguments: call.arguments ?? {} },
        }));
      }
      return wire;
    }
    case 'tool':
      return { role: 'tool', content: message.content, tool_name: message.name };
  }
}

function parseUsage(response: OllamaChatResponse): Usage {
  const inputTokens = response.prompt_eval_count;
  const outputTokens = response.eval_count;
  if (inputTokens === undefined && outputTokens === undefined) return {};
  return {
    inputTokens,
    outputTokens,
    totalTokens:
      inputTokens === undefined && outputTokens === undefined
        ? undefined
        : (inputTokens ?? 0) + (outputTokens ?? 0),
  };
}

function parseToolArguments(args: unknown, providerId: string, model: string): unknown {
  if (args === undefined || args === null) return {};
  if (typeof args !== 'string') return args;
  const parsed = extractJson(args);
  if (parsed === undefined) {
    throw new ProviderResponseError(`Ollama tool arguments were not valid JSON`, {
      providerId,
      model,
      responseBody: args.slice(0, 500),
    });
  }
  return parsed;
}

function safeParse(line: string): unknown {
  try {
    return JSON.parse(line);
  } catch {
    return undefined;
  }
}

/** Yield each newline-delimited JSON record from a byte stream. */
async function* parseNdjson(stream: AsyncIterable<Uint8Array>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = '';
  for await (const chunk of stream) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline = buffer.indexOf('\n');
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line.length > 0) yield line;
      newline = buffer.indexOf('\n');
    }
  }
  buffer += decoder.decode();
  const tail = buffer.trim();
  if (tail.length > 0) yield tail;
}

/** Create an {@link OllamaProvider}. */
export function createOllamaProvider(
  options: OllamaProviderOptions = {},
): OllamaProvider {
  return new OllamaProvider(options);
}
