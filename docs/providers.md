# Providers

A provider turns a `CompletionRequest` into a `CompletionResult`. There are four
built in and the interface is small enough that adding a fifth is a short file.

All providers talk HTTP directly through `fetch`. There is no vendor SDK
dependency to keep in sync with the wire format.

## Model references

```ts
model: 'openai:gpt-4o-mini';
model: 'anthropic:claude-sonnet-4-5';
model: 'google:gemini-2.0-flash';
model: 'ollama:llama3.2';
```

`"provider:model"`. A bare registered id or alias (`'anthropic'`, `'gemini'`)
means "that provider's default model". A bare string that is _not_ a registered
provider is treated as a model on OpenAI, so `'gpt-4o-mini'` works as written.
`parseModelRef()` and `createProviderForModel()` expose this logic.

| Provider    | Aliases                              | Key                                  | Default model       |
| ----------- | ------------------------------------ | ------------------------------------ | ------------------- |
| `openai`    | `gpt`, `openai-compatible`           | `OPENAI_API_KEY`                     | `gpt-4o-mini`       |
| `anthropic` | `claude`, `anthropic-messages`       | `ANTHROPIC_API_KEY`                  | `claude-sonnet-4-5` |
| `google`    | `gemini`, `google-genai`, `googleai` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-2.0-flash`  |
| `ollama`    | `local`, `ollama-local`              | none                                 | `llama3.2`          |

Override defaults on an agent with `providerOptions`:

```ts
new Agent({
  model: 'openai:gpt-4o-mini',
  providerOptions: {
    organization: 'org-123',
    baseURL: 'https://proxy.internal/v1',
  },
});
```

## Built-in providers

### OpenAI

Chat Completions, compatible with the many OpenAI-shaped gateways (Azure,
Together, Groq, OpenRouter, vLLM, LiteLLM, LM Studio).

```ts
import { OpenAIProvider } from '@agentloom/core';

new Agent({
  model: 'openai:gpt-4o-mini',
  providerOptions: {
    baseURL: 'https://api.groq.com/openai/v1',
    apiKey: process.env.GROQ_API_KEY,
    model: 'llama-3.3-70b-versatile',
  },
});
```

| Option                    | Default                     |
| ------------------------- | --------------------------- |
| `apiKey`                  | `OPENAI_API_KEY`            |
| `baseURL`                 | `https://api.openai.com/v1` |
| `organization`, `project` | unset                       |
| `model`                   | `gpt-4o-mini`               |
| `id`, `name`              | `openai`, `OpenAI`          |

Set `id` and `name` when wrapping a compatible vendor, so events and errors
attribute correctly.

### Anthropic

Messages API. The differences from the OpenAI shape — system prompt as a
top-level field, mandatory `max_tokens`, tool results as `user` messages
carrying `tool_result` blocks, schemas passed as `input_schema` — are all
handled inside the provider.

```ts
new Agent({ model: 'anthropic:claude-sonnet-4-5' });
```

| Option       | Default                     |
| ------------ | --------------------------- |
| `apiKey`     | `ANTHROPIC_API_KEY`         |
| `baseURL`    | `https://api.anthropic.com` |
| `apiVersion` | `2023-06-01`                |

`max_tokens` is mandatory on this API, so the provider sends `4096` unless
`modelOptions.maxTokens` (or `limits.maxTokensPerCall`) says otherwise.

### Google

Gemini `generateContent`. Consecutive same-role turns are merged because Gemini
rejects `user,user`; tool calls are `functionCall` parts and results are
`functionResponse` parts; structured output uses `responseMimeType` +
`responseSchema`.

```ts
new Agent({ model: 'google:gemini-2.0-flash' });
```

| Option    | Default                                            |
| --------- | -------------------------------------------------- |
| `apiKey`  | `GOOGLE_API_KEY`, then `GEMINI_API_KEY`            |
| `baseURL` | `https://generativelanguage.googleapis.com/v1beta` |
| `model`   | `gemini-2.0-flash`                                 |

### Ollama

Local models through `/api/chat`. No API key, no egress. Ollama streams NDJSON
rather than SSE, which the provider handles with its own line reader.

```ts
import { OllamaProvider } from '@agentloom/core';

const ollama = new OllamaProvider({ host: 'http://box.local:11434' });
const models = await ollama.listModels(); // what the server has pulled

new Agent({ model: 'ollama:llama3.2' });
```

`host` also reads `OLLAMA_HOST`. `listModels()` is the only provider-specific
helper; everything else goes through the shared interface.

## Capabilities

Every provider declares what it can actually do, and the agent adapts:

```ts
interface ProviderCapabilities {
  tools: boolean;
  parallelToolCalls: boolean;
  streaming: boolean;
  systemMessages: boolean;
  jsonMode: boolean;
  strictJsonSchema: boolean;
  vision: boolean;
  promptCaching: boolean;
}
```

This is why structured output degrades instead of failing: a provider without
`strictJsonSchema` falls back to JSON mode plus local validation and the repair
loop. A provider without `tools` simply has no tools attached to the request.

| Provider  | tools | parallel | streaming | jsonMode | strictJsonSchema |
| --------- | ----- | -------- | --------- | -------- | ---------------- |
| OpenAI    | yes   | yes      | yes       | yes      | yes              |
| Anthropic | yes   | yes      | yes       | no       | no               |
| Google    | yes   | no       | yes       | yes      | yes              |
| Ollama    | yes   | no       | yes       | yes      | no               |

Getting these right on a custom provider matters: an optimistic
`strictJsonSchema: true` on a provider that ignores it produces validation
failures you did not need.

## Custom providers

Extend `BaseProvider` and implement `doComplete`. You inherit HTTP with
timeouts and abort handling, retries, response-id assignment, and a streaming
fallback that buffers a non-streaming provider into events.

```ts
import {
  BaseProvider,
  type CompletionRequest,
  type CompletionResult,
  type ProviderCallOptions,
  type StreamEvent,
} from '@agentloom/core';

class TinyLlamaProvider extends BaseProvider {
  readonly id = 'tiny-llama';
  readonly name = 'Tiny Llama';
  readonly defaultModel = 'tiny-1b';
  readonly capabilities = {
    tools: true,
    parallelToolCalls: false,
    streaming: true,
    systemMessages: true,
    jsonMode: false,
    strictJsonSchema: false,
    vision: false,
    promptCaching: false,
  };

  protected async doComplete(
    request: CompletionRequest,
    options: ProviderCallOptions,
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const result = await this.http.json<MyVendorResponse>({
      url: `${this.host}/generate`,
      method: 'POST',
      headers: { authorization: `Bearer ${this.apiKey}` },
      body: toWire(request),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      providerId: this.id,
      context: { model: request.model },
    });

    return {
      message: { role: 'assistant', content: result.text },
      finishReason: 'stop',
      usage: { inputTokens: result.in, outputTokens: result.out },
      responseId: result.id,
      providerId: this.id,
      model: request.model,
      latencyMs: Date.now() - startedAt,
      raw: result,
    };
  }
}
```

Override `doStream` when the vendor has real streaming. Throwing on failure is
correct; yielding an `error` event is not — the agent expects the iterable to
reject.

`withModel()` returns a prototype-preserving clone bound to a different default
model, which is how `'tiny-llama:tiny-1b'` resolves without mutating the
registered instance.

### The plain interface

When you already have a client from another library, implement `ModelProvider`
directly and adapt it:

```ts
import { type ModelProvider } from '@agentloom/core';

const adapter: ModelProvider = {
  id: 'acme-internal',
  name: 'Acme Internal',
  defaultModel: 'acme-7b',
  capabilities,
  async complete(request) {
    const res = await acme.chat(toWire(request));
    return {
      message: { role: 'assistant', content: res.text },
      finishReason: 'stop',
      usage: { inputTokens: res.in, outputTokens: res.out },
      responseId: res.id,
      providerId: 'acme-internal',
      model: request.model,
      latencyMs: res.elapsed,
      raw: res,
    };
  },
  stream: (request) => ({
    async *[Symbol.asyncIterator]() {
      for await (const chunk of acme.stream(request)) {
        yield { type: 'text-delta', text: chunk, responseId: 'r1' };
      }
      yield { type: 'finish', result: await adapter.complete(request) };
    },
  }),
};

new Agent({ provider: adapter, model: 'acme-7b' });
```

The contract: `complete` rejects with an `AgentError` on failure; `stream` throws
rather than yielding an error event.

### Registering

```ts
import {
  createProvider,
  defaultProviderRegistry,
  registerProvider,
  unregisterProvider,
} from '@agentloom/core';

registerProvider(
  'tiny-llama',
  (options = {}) => new TinyLlamaProvider(options),
  {
    defaults: { host: 'http://localhost:8080' },
    aliases: ['tiny'],
  },
);

defaultProviderRegistry().ids(); // every id and alias, sorted
unregisterProvider('tiny-llama');

const provider = createProvider('tiny-llama:tiny-1b');
new Agent({ model: 'tiny-llama:tiny-1b' }); // now resolvable from config
```

`registerProvider` replaces an existing factory for the same id, which is what
tests and hot reloads need. `ProviderRegistry` can also be constructed directly
with `includeBuiltins: false` for an isolated registry.

## The HTTP layer

`HttpClient` is shared by every built-in provider. It resolves any HTTP status
(and leaves it to `expectOk` to raise a typed error), merges default headers,
propagates aborts, and normalizes timeouts and transport failures.

```ts
const provider = new OpenAIProvider({
  fetch: myInstrumentedFetch, // inject a proxy, tracing, or a test double
  defaultTimeoutMs: 30_000,
  defaultHeaders: { 'x-team': 'platform' },
});
```

| Method     | Behaviour                                                                  |
| ---------- | -------------------------------------------------------------------------- |
| `request`  | Resolves for any status. Rejects only on transport failures and aborts     |
| `expectOk` | As above, but rejects with a typed `ProviderError` for non-2xx             |
| `json`     | `expectOk` plus JSON parsing, with a `ProviderResponseError` on a bad body |

`parseRetryAfter()` converts a `Retry-After` header (seconds or HTTP date) into
milliseconds, and the retry layer honours it in preference to computed backoff.

## Retry

Provider calls retry by default: three attempts, exponential backoff from
500 ms, capped at 30 s, with jitter, and `Retry-After` respected.

```ts
new Agent({
  model: 'openai:gpt-4o-mini',
  retry: {
    maxAttempts: 5,
    initialDelayMs: 250,
    shouldRetry: (error) => error.code === ErrorCode.PROVIDER_RATE_LIMIT,
    onRetry: ({ attempt, delayMs, error }) =>
      metrics.inc('llm.retry', { attempt, error: error.code }),
  },
});
```

Set `retry: false` on the agent to disable it. `withRetry`, `resolveRetryPolicy`,
and `computeBackoffDelay` are exported for use elsewhere.

Aborts are never retried. The default predicate is `error.retryable`, which
`ProviderError` derives from the HTTP status: 408, 409, 425, 429, and 5xx are
retryable; 401, 403, 402, and malformed bodies are not.

## Errors

`providerErrorFromResponse` maps a status and body to the most specific class
available, identically across vendors:

| Status                | Error                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------- |
| 401, 403              | `ProviderAuthError`                                                                           |
| 429                   | `ProviderRateLimitError`, or `ProviderQuotaError` when the body mentions quota/billing/credit |
| 402                   | `ProviderQuotaError`                                                                          |
| anything else non-2xx | `ProviderError` with `statusCode` set                                                         |

Timeouts become `ProviderTimeoutError`; unparseable 2xx bodies become
`ProviderResponseError` with `responseBody` attached. See [errors](errors.md).

## More

- [Agent](agent.md) — `model`, `provider`, `providerOptions`, `modelOptions`
- [Streaming & events](streaming-and-events.md) — how streaming reaches your UI
- [Testing](testing.md) — scripted providers and `fetch` injection
