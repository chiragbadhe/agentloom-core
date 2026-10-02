import { ConfigurationError } from '../errors.js';
import { createAnthropicProvider } from './anthropic.js';
import { createGoogleProvider } from './google.js';
import { createOllamaProvider } from './ollama.js';
import { createOpenAIProvider } from './openai.js';
import type { ModelProvider } from './types.js';

/** Factory signature for a registered provider. */
export type ProviderFactory = (options?: Record<string, unknown>) => ModelProvider;

interface Registration {
  readonly factory: ProviderFactory;
  readonly defaults?: Record<string, unknown>;
  readonly aliases: readonly string[];
}

/**
 * Registry of provider factories, keyed by id.
 *
 * Built-in providers are pre-registered. Add your own to make it usable from a
 * plain config string:
 *
 * ```ts
 * registerProvider('my-vendor', (options) => new MyProvider(options));
 * const agent = new Agent({ model: 'my-vendor:large' });
 * ```
 */
export class ProviderRegistry {
  private readonly providers = new Map<string, Registration>();

  constructor(includeBuiltins = true) {
    if (includeBuiltins) {
      for (const [id, factory, defaults, aliases] of builtInProviders) {
        this.providers.set(id, { factory, defaults, aliases });
      }
    }
  }

  /** Register (or replace) a provider factory. */
  register(
    id: string,
    factory: ProviderFactory,
    options: { defaults?: Record<string, unknown>; aliases?: readonly string[] } = {},
  ): this {
    this.providers.set(id, {
      factory,
      defaults: options.defaults,
      aliases: options.aliases ?? [],
    });
    return this;
  }

  unregister(id: string): boolean {
    return this.providers.delete(id);
  }

  has(idOrAlias: string): boolean {
    return this.resolveId(idOrAlias) !== undefined;
  }

  /** All registered ids plus aliases, sorted. */
  ids(): string[] {
    const out: string[] = [];
    for (const [id, registration] of this.providers) {
      out.push(id);
      out.push(...registration.aliases);
    }
    return out.sort();
  }

  private resolveId(idOrAlias: string): string | undefined {
    if (this.providers.has(idOrAlias)) return idOrAlias;
    for (const [id, registration] of this.providers) {
      if (registration.aliases.includes(idOrAlias)) return id;
    }
    return undefined;
  }

  /** Instantiate a provider by id or alias. */
  create(idOrAlias: string, options: Record<string, unknown> = {}): ModelProvider {
    const id = this.resolveId(idOrAlias);
    const registration = id === undefined ? undefined : this.providers.get(id);
    if (registration === undefined) {
      throw new ConfigurationError(
        `Unknown provider "${idOrAlias}". Registered providers: ${this.ids().join(', ')}`,
        { requested: idOrAlias, available: this.ids() },
      );
    }
    return registration.factory({ ...registration.defaults, ...options });
  }
}

const builtInProviders: readonly [
  string,
  ProviderFactory,
  Record<string, unknown> | undefined,
  readonly string[],
][] = [
  [
    'openai',
    (options) => createOpenAIProvider(options),
    undefined,
    ['gpt', 'openai-compatible'],
  ],
  [
    'anthropic',
    (options) => createAnthropicProvider(options),
    undefined,
    ['claude', 'anthropic-messages'],
  ],
  [
    'google',
    (options) => createGoogleProvider(options),
    undefined,
    ['gemini', 'google-genai', 'googleai'],
  ],
  [
    'ollama',
    (options) => createOllamaProvider(options),
    undefined,
    ['local', 'ollama-local'],
  ],
];

const globalRegistry = new ProviderRegistry();

/**
 * Register a provider on the default registry. Registering the same id twice
 * replaces the previous factory, which is what tests and hot reloads need.
 */
export function registerProvider(
  id: string,
  factory: ProviderFactory,
  options?: { defaults?: Record<string, unknown>; aliases?: readonly string[] },
): void {
  globalRegistry.register(id, factory, options);
}

export function unregisterProvider(id: string): boolean {
  return globalRegistry.unregister(id);
}

/** The process-wide registry used by {@link createProvider}. */
export function defaultProviderRegistry(): ProviderRegistry {
  return globalRegistry;
}

export interface ParsedModelRef {
  readonly providerId: string;
  readonly model: string | undefined;
}

/**
 * Parse `"provider:model"`. A bare string is treated as a model on the default
 * provider (`openai`), matching what most people mean by `'gpt-4o-mini'`.
 */
export function parseModelRef(ref: string, defaultProviderId = 'openai'): ParsedModelRef {
  const separator = ref.indexOf(':');
  if (separator <= 0) return { providerId: defaultProviderId, model: ref };
  return {
    providerId: ref.slice(0, separator),
    model: ref.slice(separator + 1) || undefined,
  };
}

/**
 * Resolve a model string into a ready-to-use provider.
 *
 * ```ts
 * createProvider('anthropic:claude-sonnet-4-5');
 * createProvider('ollama:llama3.2', { host: 'http://box.local:11434' });
 * ```
 */
export function createProvider(
  ref: string,
  options: Record<string, unknown> = {},
): ModelProvider {
  const { providerId, model } = resolveRef(ref);
  const provider = globalRegistry.create(providerId, options);
  if (model === undefined) return provider;
  if (typeof provider.withModel !== 'function') return provider;
  return provider.withModel(model);
}

/** Instantiate a provider for `ref` with per-call options applied. */
export function createProviderForModel(
  ref: string,
  options: Record<string, unknown> = {},
): { provider: ModelProvider; model: string } {
  const { providerId, model } = resolveRef(ref);
  const provider = globalRegistry.create(providerId, options);
  return { provider, model: model ?? provider.defaultModel };
}

/**
 * Like {@link parseModelRef}, but a bare *registered* provider id or alias
 * (`'anthropic'`, `'gemini'`) means "that provider's default model" rather than
 * "a model named `anthropic` on OpenAI".
 */
function resolveRef(ref: string): ParsedModelRef {
  if (!ref.includes(':') && globalRegistry.has(ref)) {
    return { providerId: ref, model: undefined };
  }
  return parseModelRef(ref);
}
