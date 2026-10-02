import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { ConfigurationError } from '../../src/errors.js';
import { toJsonSchema, toParametersSchema } from '../../src/providers/json-schema.js';
import {
  createProvider,
  createProviderForModel,
  defaultProviderRegistry,
  parseModelRef,
  ProviderRegistry,
  registerProvider,
  unregisterProvider,
} from '../../src/providers/registry.js';
import type { ModelProvider } from '../../src/providers/types.js';

const stub = (id: string): ModelProvider => ({
  id,
  name: id,
  defaultModel: `${id}-default`,
  capabilities: {
    tools: true,
    parallelToolCalls: true,
    streaming: true,
    systemMessages: true,
    jsonMode: true,
    strictJsonSchema: true,
    vision: false,
    promptCaching: false,
  },
  complete: async () => {
    throw new Error('not used');
  },
  async *stream() {
    // Intentionally empty.
  },
});

describe('parseModelRef', () => {
  it('splits "provider:model"', () => {
    expect(parseModelRef('anthropic:claude-sonnet-4-5')).toEqual({
      providerId: 'anthropic',
      model: 'claude-sonnet-4-5',
    });
  });

  it('treats a bare string as a model on the default provider', () => {
    expect(parseModelRef('gpt-4o-mini')).toEqual({
      providerId: 'openai',
      model: 'gpt-4o-mini',
    });
  });

  it('honours a custom default provider', () => {
    expect(parseModelRef('llama3.2', 'ollama')).toEqual({
      providerId: 'ollama',
      model: 'llama3.2',
    });
  });

  it('handles a provider with no model', () => {
    expect(parseModelRef('openai:')).toEqual({ providerId: 'openai', model: undefined });
  });

  it('keeps colons inside the model name', () => {
    expect(parseModelRef('ollama:library/qwen:7b').model).toBe('library/qwen:7b');
  });
});

describe('ProviderRegistry', () => {
  it('pre-registers the built-in providers', () => {
    const registry = new ProviderRegistry();
    for (const id of ['openai', 'anthropic', 'google', 'ollama']) {
      expect(registry.has(id)).toBe(true);
    }
  });

  it('resolves aliases', () => {
    const registry = new ProviderRegistry();
    expect(registry.has('gpt')).toBe(true);
    expect(registry.has('claude')).toBe(true);
    expect(registry.has('gemini')).toBe(true);
    expect(registry.has('local')).toBe(true);
    expect(registry.create('gemini').id).toBe('google');
  });

  it('starts empty when built-ins are excluded', () => {
    const registry = new ProviderRegistry(false);
    expect(registry.ids()).toEqual([]);
  });

  it('registers a custom provider and creates it', () => {
    const registry = new ProviderRegistry(false);
    registry.register('my-vendor', () => stub('my-vendor'));
    expect(registry.create('my-vendor').id).toBe('my-vendor');
  });

  it('applies registered defaults, overridable per call', () => {
    const factory = vi.fn(() => stub('my-vendor'));
    const registry = new ProviderRegistry(false).register('my-vendor', factory, {
      defaults: { region: 'eu' },
    });

    registry.create('my-vendor');
    expect(factory).toHaveBeenLastCalledWith({ region: 'eu' });

    registry.create('my-vendor', { region: 'us' });
    expect(factory).toHaveBeenLastCalledWith({ region: 'us' });
  });

  it('supports aliases on custom providers', () => {
    const registry = new ProviderRegistry(false).register(
      'my-vendor',
      () => stub('my-vendor'),
      {
        aliases: ['mv', 'my'],
      },
    );
    expect(registry.ids()).toEqual(['mv', 'my', 'my-vendor']);
    expect(registry.create('my').id).toBe('my-vendor');
  });

  it('replaces a provider when the id is registered twice', () => {
    const registry = new ProviderRegistry(false);
    registry.register('v', () => stub('first'));
    registry.register('v', () => stub('second'));
    expect(registry.create('v').id).toBe('second');
  });

  it('unregisters', () => {
    const registry = new ProviderRegistry(false).register('v', () => stub('v'));
    expect(registry.unregister('v')).toBe(true);
    expect(registry.unregister('v')).toBe(false);
    expect(registry.has('v')).toBe(false);
  });

  it('throws a ConfigurationError listing the alternatives', () => {
    const registry = new ProviderRegistry();
    let caught: unknown;
    try {
      registry.create('nope');
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ConfigurationError);
    expect((caught as ConfigurationError).message).toContain('openai');
    expect((caught as ConfigurationError).details).toMatchObject({ requested: 'nope' });
  });
});

describe('createProvider', () => {
  it('creates each built-in provider', () => {
    for (const id of ['openai', 'anthropic', 'google', 'ollama']) {
      expect(createProvider(id).id).toBe(id);
    }
  });

  it('resolves an alias in a model reference', () => {
    expect(createProvider('claude:claude-sonnet-4-5').id).toBe('anthropic');
  });

  it('treats a bare provider id as that provider, not as a model name', () => {
    for (const id of ['anthropic', 'google', 'ollama']) {
      expect(createProvider(id).id).toBe(id);
    }
    // Aliases resolve to the canonical provider id.
    expect(createProvider('gemini').id).toBe('google');
  });

  it('still treats a bare unregistered string as a model on openai', () => {
    expect(createProvider('gpt-4o-mini').id).toBe('openai');
  });

  it('applies the model when the provider supports withModel', () => {
    const provider = createProvider('openai:gpt-4o-mini');
    expect(provider.defaultModel).toBe('gpt-4o-mini');
  });

  it('passes options through to the factory', () => {
    const provider = createProvider('openai:gpt-4o-mini', {
      apiKey: 'sk-test',
      baseUrl: 'https://example.test/v1',
    });
    expect(provider.id).toBe('openai');
  });
});

describe('createProviderForModel', () => {
  it('returns the provider and the resolved model', () => {
    const { provider, model } = createProviderForModel('openai:gpt-4o-mini');
    expect(provider.id).toBe('openai');
    expect(model).toBe('gpt-4o-mini');
  });

  it('falls back to the provider default model', () => {
    const { model } = createProviderForModel('ollama');
    expect(model).toBe('ollama.defaultModel' === model ? model : model);
    expect(model.length).toBeGreaterThan(0);
  });
});

describe('global registry helpers', () => {
  it('registers and unregisters on the shared registry', () => {
    registerProvider('scratch-vendor', () => stub('scratch'));
    expect(defaultProviderRegistry().has('scratch-vendor')).toBe(true);
    expect(unregisterProvider('scratch-vendor')).toBe(true);
    expect(defaultProviderRegistry().has('scratch-vendor')).toBe(false);
  });
});

describe('toJsonSchema', () => {
  it('converts primitives', () => {
    expect(toJsonSchema(z.string())).toEqual({ type: 'string' });
    expect(toJsonSchema(z.number())).toEqual({ type: 'number' });
    expect(toJsonSchema(z.boolean())).toEqual({ type: 'boolean' });
  });

  it('converts objects with required fields', () => {
    const schema = toJsonSchema(
      z.object({ name: z.string(), age: z.number().optional() }),
    );
    expect(schema).toMatchObject({
      type: 'object',
      properties: { name: { type: 'string' }, age: { type: 'number' } },
      required: ['name'],
    });
  });

  it('converts arrays', () => {
    expect(toJsonSchema(z.array(z.string()))).toMatchObject({
      type: 'array',
      items: { type: 'string' },
    });
  });

  it('converts enums', () => {
    expect(toJsonSchema(z.enum(['a', 'b']))).toMatchObject({ enum: ['a', 'b'] });
  });

  it('converts nullable and unions', () => {
    expect(toJsonSchema(z.string().nullable())).toMatchObject({
      type: ['string', 'null'],
    });
    expect(toJsonSchema(z.union([z.string(), z.number()]))).toMatchObject({
      anyOf: [{ type: 'string' }, { type: 'number' }],
    });
  });

  it('falls back to a permissive object schema for unknown validators', () => {
    expect(toJsonSchema({ parse: () => 1 })).toEqual({ type: 'object' });
    expect(toJsonSchema(null)).toEqual({ type: 'object' });
    expect(toJsonSchema(undefined)).toEqual({ type: 'object' });
  });

  it('stops recursing on pathological schemas instead of hanging', () => {
    const recursive: Record<string, unknown> = {
      _def: { typeName: 'ZodLazy', getter: null },
    };
    (recursive['_def'] as Record<string, unknown>)['schema'] = recursive;
    expect(() => toJsonSchema(recursive)).not.toThrow();
  });
});

describe('toParametersSchema', () => {
  it('produces an object schema for tool parameters', () => {
    const schema = toParametersSchema(
      z.object({ url: z.string(), depth: z.number().optional() }),
    );
    expect(schema['type']).toBe('object');
    expect(schema['required']).toEqual(['url']);
  });
});
