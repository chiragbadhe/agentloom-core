/**
 * 07 — bring your own provider.
 *
 * `ModelProvider` is three members, so a new vendor is a small file. Extend
 * `BaseProvider` instead and you inherit retries, timeouts, abort handling, and
 * the buffered-stream fallback for free.
 *
 * Run: `npx tsx examples/07-custom-provider.ts`
 */
import {
  BaseProvider,
  createProvider,
  defaultProviderRegistry,
  registerProvider,
  type CompletionRequest,
  type CompletionResult,
  type ModelProvider,
  type ProviderCallOptions,
  type ProviderCapabilities,
  type StreamEvent,
} from '../src/index.js';

const CAPABILITIES: ProviderCapabilities = {
  tools: true,
  parallelToolCalls: false,
  streaming: true,
  systemMessages: true,
  jsonMode: false,
  strictJsonSchema: false,
  vision: false,
  promptCaching: false,
};

/**
 * A provider that "executes" tool calls locally and answers from a lookup
 * table. It streams word by word so the streaming path is real, not simulated.
 */
class TinyLlamaProvider extends BaseProvider {
  readonly id = 'tiny-llama';
  readonly name = 'Tiny Llama (example)';
  readonly defaultModel = 'tiny-1b';
  readonly capabilities = CAPABILITIES;

  protected async doComplete(
    request: CompletionRequest,
    _options: ProviderCallOptions = {},
  ): Promise<CompletionResult> {
    const startedAt = Date.now();
    const lastUser = [...request.messages].reverse().find((m) => m.role === 'user');
    const input = lastUser?.role === 'user' ? lastUser.content : '';

    return {
      message: { role: 'assistant', content: `You said: ${input}` },
      finishReason: 'stop',
      usage: {
        inputTokens: input.length,
        outputTokens: 8,
        totalTokens: input.length + 8,
      },
      responseId: 'tiny-1',
      providerId: this.id,
      model: request.model,
      latencyMs: Date.now() - startedAt,
      raw: { tools: request.tools?.map((t) => t.name) ?? [] },
    };
  }

  protected override doStream(
    request: CompletionRequest,
    options: ProviderCallOptions = {},
  ): AsyncIterable<StreamEvent> {
    const stream = async function* (
      this: TinyLlamaProvider,
    ): AsyncGenerator<StreamEvent> {
      const responseId = 'tiny-stream';
      yield { type: 'start', model: request.model, responseId };

      const result = await this.doComplete(request, options);
      for (const word of result.message.content.split(' ')) {
        yield { type: 'text-delta', text: word, responseId };
      }
      yield { type: 'finish', result: { ...result, responseId } };
    };
    return stream.call(this);
  }
}

/**
 * Alternatively implement `ModelProvider` directly — useful when you already
 * have a client from another library and just need to adapt it.
 */
const adapter: ModelProvider = {
  id: 'acme-internal',
  name: 'Acme Internal',
  defaultModel: 'acme-7b',
  capabilities: CAPABILITIES,
  complete: async (request) => ({
    message: { role: 'assistant', content: 'ok' },
    finishReason: 'stop',
    usage: {},
    responseId: 'acme-1',
    providerId: 'acme-internal',
    model: request.model,
    latencyMs: 1,
    raw: {},
  }),
  stream: (request) => ({
    async *[Symbol.asyncIterator]() {
      yield { type: 'finish', result: await adapter.complete(request) };
    },
  }),
};

/** Register a factory so `model: 'tiny-llama:tiny-1b'` resolves. */
registerProvider('tiny-llama', (options = {}) => new TinyLlamaProvider(options));

console.log('registered providers:', defaultProviderRegistry().ids().join(', '));

const provider = createProvider('tiny-llama:tiny-1b');
console.log(`${provider.name} -> ${provider.defaultModel}`);

// The registry hands back a clone bound to the requested model, so provider
// instances stay reusable.
console.log('resolved model:', provider.defaultModel);

const events: string[] = [];
for await (const event of provider.stream({
  model: 'tiny-1b',
  messages: [{ role: 'user', content: 'hello world' }],
})) {
  events.push(event.type);
}
console.log('stream events:', events.join(' -> '));

const adapterResult = await adapter.complete({
  model: 'acme-7b',
  messages: [{ role: 'user', content: 'ping' }],
});
console.log('adapter said:', adapterResult.message.content);
