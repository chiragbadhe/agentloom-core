import { ProviderResponseError } from '../errors.js';
import { createId } from '../utils/id.js';
import { toSnakeCase } from '../utils/text.js';
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

export interface GoogleProviderOptions extends BaseProviderOptions {
  /** API key. Falls back to `GOOGLE_API_KEY` or `GEMINI_API_KEY`. */
  readonly apiKey?: string;
  /** Default `https://generativelanguage.googleapis.com/v1beta`. */
  readonly baseURL?: string;
  readonly model?: string;
  readonly id?: string;
  readonly name?: string;
}

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

/**
 * Google Gemini (`generateContent`).
 *
 * Differences from the OpenAI shape handled here:
 * - `contents` use `user`/`model` roles, and the system prompt is a separate
 *   `systemInstruction`;
 * - consecutive same-role turns are merged, as Gemini rejects `user,user`;
 * - tool calls are `functionCall` parts and results are `functionResponse`
 *   parts attached to a `user` turn;
 * - structured output is `responseMimeType` + `responseSchema`.
 */
export class GoogleProvider extends BaseProvider {
  readonly id: string;
  readonly name: string;
  readonly defaultModel: string;
  readonly capabilities: ProviderCapabilities = {
    tools: true,
    parallelToolCalls: false,
    streaming: true,
    systemMessages: true,
    jsonMode: true,
    strictJsonSchema: true,
    vision: true,
    promptCaching: false,
  };

  private readonly apiKey: string | undefined;
  private readonly baseURL: string;

  constructor(options: GoogleProviderOptions = {}) {
    const baseURL = (options.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    super({
      fetch: options.fetch,
      logger: options.logger,
      retry: options.retry,
      defaultTimeoutMs: options.defaultTimeoutMs,
      defaultHeaders: { 'content-type': 'application/json', ...options.headers },
    });
    this.id = options.id ?? 'google';
    this.name = options.name ?? 'Google Gemini';
    this.baseURL = baseURL;
    this.apiKey = resolveApiKey(options.apiKey, ['GOOGLE_API_KEY', 'GEMINI_API_KEY']);
    this.defaultModel = options.model ?? 'gemini-2.0-flash';
  }

  protected override async doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const fallbackId = createId('resp');
    const response = await this.http.json<GeminiResponse>({
      url: this.endpoint(request.model, 'generateContent'),
      method: 'POST',
      headers: { ...options.headers },
      body: buildBody(request),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const candidate = response.candidates?.[0];
    const { text, toolCalls } = parseCandidate(candidate, this.id, request.model);

    const message: AssistantMessage = {
      role: 'assistant',
      content: text,
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
    };

    return {
      message,
      finishReason: normalizeFinishReason(candidate?.finishReason),
      usage: parseUsage(response.usageMetadata),
      responseId: response.responseId ?? fallbackId,
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
    const url = `${this.endpoint(request.model, 'streamGenerateContent')}&alt=sse`;
    const response = await this.http.expectOk({
      url,
      method: 'POST',
      headers: { ...options.headers },
      body: buildBody(request),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    const stream = response.stream();
    if (stream === null) {
      throw new ProviderResponseError(
        'Gemini response did not expose a readable stream',
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
    const toolCalls = new Map<string, ToolCall>();
    let index = 0;

    for await (const frame of parseSseStream(stream)) {
      const payload = safeParse(frame.data) as GeminiResponse | undefined;
      if (payload === undefined) continue;
      usage = parseUsage(payload.usageMetadata);
      // Usage-only frames (and keep-alives) legitimately carry no candidates.
      const candidate = payload.candidates?.[0];
      if (candidate === undefined) continue;
      const parsed = parseCandidate(candidate, this.id, request.model);
      if (parsed.text) {
        text += parsed.text;
        yield { type: 'text-delta', text: parsed.text, responseId };
      }
      for (const call of parsed.toolCalls) {
        const key = `${call.name}:${index}`;
        toolCalls.set(key, call);
        index++;
        yield { type: 'tool-call', call, responseId };
      }
      if (candidate.finishReason) {
        finishReason = normalizeFinishReason(candidate.finishReason);
      }
    }

    const message: AssistantMessage = {
      role: 'assistant',
      content: text,
      ...(toolCalls.size > 0 ? { toolCalls: [...toolCalls.values()] } : {}),
    };

    yield {
      type: 'finish',
      result: {
        message,
        finishReason:
          toolCalls.size > 0 && finishReason === 'stop' ? 'tool_calls' : finishReason,
        usage,
        responseId,
        providerId: this.id,
        model: request.model,
        latencyMs: Date.now() - startedAt,
        raw: { streamed: true },
      },
    };
  }

  private endpoint(model: string, method: string): string {
    const key =
      this.apiKey === undefined ? '' : `?key=${encodeURIComponent(this.apiKey)}`;
    return `${this.baseURL}/models/${encodeURIComponent(model)}:${method}${key}`;
  }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

interface GeminiPart {
  text?: string;
  functionCall?: { id?: string; name?: string; args?: unknown };
  functionResponse?: { id?: string; name?: string; response?: unknown };
  thought?: boolean;
}

interface GeminiCandidate {
  content?: { role?: string; parts?: GeminiPart[] };
  finishReason?: string;
  index?: number;
}

interface GeminiResponse {
  candidates?: GeminiCandidate[];
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
    cachedContentTokenCount?: number;
    thoughtsTokenCount?: number;
  };
  responseId?: string;
}

function buildBody(request: CompletionRequest): Record<string, unknown> {
  const systemParts = request.messages
    .filter((message) => message.role === 'system')
    .map((message) => (message as { content: string }).content);

  const body: Record<string, unknown> = {
    contents: mergeTurns(request.messages.filter((message) => message.role !== 'system')),
  };
  if (systemParts.length > 0)
    body.systemInstruction = { parts: [{ text: systemParts.join('\n\n') }] };

  if (request.tools && request.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: request.tools.map((tool) => ({
          name: toSnakeCase(tool.name),
          description: tool.description,
          parameters: stripUnsupportedKeywords(toParametersSchema(tool.parameters)),
        })),
      },
    ];
    body.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  }

  const generationConfig: Record<string, unknown> = {};
  if (request.temperature !== undefined)
    generationConfig.temperature = request.temperature;
  if (request.topP !== undefined) generationConfig.topP = request.topP;
  if (request.topK !== undefined) generationConfig.topK = request.topK;
  if (request.maxTokens !== undefined)
    generationConfig.maxOutputTokens = request.maxTokens;
  if (request.stopSequences !== undefined)
    generationConfig.stopSequences = request.stopSequences;
  if (request.seed !== undefined) generationConfig.seed = request.seed;
  if (request.responseFormat && request.responseFormat.type !== 'text') {
    generationConfig.responseMimeType = 'application/json';
    if (request.responseFormat.type === 'json_schema') {
      generationConfig.responseSchema = stripUnsupportedKeywords(
        request.responseFormat.schema,
      );
    }
  }
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;

  return body;
}

/**
 * Gemini rejects a JSON Schema containing unsupported keywords such as
 * `$schema`, `default`, or `additionalProperties: false` in some versions.
 * Stripping them keeps tool calling working instead of failing at request time.
 */
function stripUnsupportedKeywords(
  schema: Record<string, unknown>,
): Record<string, unknown> {
  const drop = new Set([
    '$schema',
    'default',
    'additionalProperties',
    '$id',
    'definitions',
    '$defs',
  ]);
  const walk = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(walk);
    if (typeof node !== 'object' || node === null) return node;
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      if (drop.has(key)) continue;
      out[key] = walk(value);
    }
    return out;
  };
  return walk(schema) as Record<string, unknown>;
}

/**
 * Gemini requires strictly alternating user/model turns, so consecutive
 * messages with the same role are merged into one.
 */
function mergeTurns(messages: readonly ModelMessage[]): Record<string, unknown>[] {
  const turns: Record<string, unknown>[] = [];

  for (const message of messages) {
    const role = message.role === 'assistant' ? 'model' : 'user';
    const parts = toParts(message);
    const last = turns[turns.length - 1];

    if (last !== undefined && last['role'] === role) {
      (last['parts'] as GeminiPart[]).push(...parts);
      continue;
    }
    turns.push({ role, parts });
  }

  return turns;
}

function toParts(message: ModelMessage): GeminiPart[] {
  switch (message.role) {
    case 'system':
    case 'user':
      return [{ text: message.content }];
    case 'assistant': {
      const parts: GeminiPart[] = [];
      if (message.content) parts.push({ text: message.content });
      for (const call of message.toolCalls ?? []) {
        parts.push({
          functionCall: {
            id: call.id,
            name: toSnakeCase(call.name),
            args: call.arguments ?? {},
          },
        });
      }
      return parts;
    }
    case 'tool':
      return [
        {
          functionResponse: {
            id: message.toolCallId,
            name: toSnakeCase(message.name),
            response: message.isError
              ? { error: message.content }
              : { result: safeParse(message.content) ?? message.content },
          },
        },
      ];
  }
}

function parseCandidate(
  candidate: GeminiCandidate | undefined,
  providerId: string,
  model: string,
): { text: string; toolCalls: ToolCall[] } {
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];

  for (const part of candidate?.content?.parts ?? []) {
    if (part.functionCall) {
      toolCalls.push({
        id: part.functionCall.id ?? `call_${toolCalls.length}`,
        name: fromSnakeCase(part.functionCall.name ?? ''),
        arguments: part.functionCall.args ?? {},
        rawArguments: JSON.stringify(part.functionCall.args ?? {}),
      });
      continue;
    }
    if (typeof part.text === 'string' && !part.thought) textParts.push(part.text);
  }

  if (candidate === undefined) {
    throw new ProviderResponseError('Gemini returned no candidates', {
      providerId,
      model,
    });
  }

  return { text: textParts.join(''), toolCalls };
}

/** `list_directory` -> `listDirectory` */
function fromSnakeCase(value: string): string {
  return value.replace(/_([a-z0-9])/g, (_, char: string) => char.toUpperCase());
}

function parseUsage(usage: GeminiResponse['usageMetadata']): Usage {
  if (!usage) return {};
  return {
    inputTokens: usage.promptTokenCount,
    outputTokens: usage.candidatesTokenCount,
    totalTokens: usage.totalTokenCount,
    cachedInputTokens: usage.cachedContentTokenCount,
    reasoningTokens: usage.thoughtsTokenCount,
  };
}

function safeParse(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return undefined;
  }
}

/** Create a {@link GoogleProvider}. */
export function createGoogleProvider(
  options: GoogleProviderOptions = {},
): GoogleProvider {
  return new GoogleProvider(options);
}

/** Alias for {@link createGoogleProvider}. */
export const createGeminiProvider = createGoogleProvider;
