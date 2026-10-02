import { ConfigurationError } from '../errors.js';
import type { ToolSpec } from '../providers/types.js';
import { anySchema, type Schema } from '../schema.js';
import { noopLogger, type Logger } from '../utils/logger.js';
import { toJsonSchema } from '../providers/json-schema.js';
import { schemaFromJsonSchema } from '../schema.js';
import type { AnyState, AnyTool, ToolDefinition } from './types.js';

const NAME_PATTERN = /^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/;

/**
 * Register a tool. The main entry point for tool authors — it validates the
 * shape up front and infers the JSON Schema sent to the model.
 *
 * ```ts
 * const getWeather = defineTool({
 *   name: 'get_weather',
 *   description: 'Current weather for a city.',
 *   parameters: z.object({ city: z.string(), units: z.enum(['c','f']).default('c') }),
 *   execute: async ({ city, units }) => fetchWeather(city, units),
 * });
 * ```
 *
 * `TArgs` and `TResult` are inferred from `parameters` and `execute`, so the
 * registry and the model-facing schema can never drift apart.
 */
export function defineTool<
  TArgs = Record<string, never>,
  TResult = unknown,
  TState = unknown,
>(
  definition: ToolDefinition<TArgs, TResult, TState>,
): ToolDefinition<TArgs, TResult, TState> {
  if (definition.name.trim() === '') {
    throw new ConfigurationError('Tool name cannot be empty');
  }
  if (!NAME_PATTERN.test(definition.name)) {
    throw new ConfigurationError(
      `Invalid tool name "${definition.name}". Use letters, digits, "_" or "-" (max 64 chars, must start with a letter).`,
      { name: definition.name },
    );
  }
  if (definition.description.trim() === '') {
    throw new ConfigurationError(`Tool "${definition.name}" must have a description`);
  }
  if (typeof definition.execute !== 'function') {
    throw new ConfigurationError(
      `Tool "${definition.name}" must define an execute function`,
    );
  }
  return definition;
}

/** Alias for {@link defineTool}, for people who prefer `tool()`. */
export const tool = defineTool;

/**
 * A name-indexed collection of tools with aliasing and scoping.
 *
 * ```ts
 * const registry = new ToolRegistry([getWeather, sendEmail]);
 * registry.only('get_weather');            // scoped view
 * registry.rename({ sendEmail: 'send_email' });
 * ```
 */
export class ToolRegistry<TState = unknown> {
  private readonly tools = new Map<string, AnyTool<TState>>();
  private readonly logger: Logger;

  constructor(
    tools: Iterable<AnyTool<TState>> | ToolRegistry<AnyState> = [],
    options: { logger?: Logger } = {},
  ) {
    this.logger = options.logger ?? noopLogger();
    const source: Iterable<AnyTool<TState>> =
      tools instanceof ToolRegistry ? tools.all() : tools;
    for (const item of source) this.add(item);
  }

  /** Add or replace a tool. Replacing keeps the original insertion position. */
  add<TArgs, TResult>(definition: ToolDefinition<TArgs, TResult, TState>): this {
    const checked = defineTool(definition as AnyTool<TState>);
    if (this.tools.has(checked.name)) {
      this.logger.debug('tool replaced', { tool: checked.name });
    }
    this.tools.set(checked.name, checked);
    return this;
  }

  /** Add many tools. */
  addAll(definitions: Iterable<AnyTool<TState>>): this {
    for (const definition of definitions) this.add(definition);
    return this;
  }

  /** Remove a tool. Returns `true` when something was removed. */
  remove(name: string): boolean {
    return this.tools.delete(name);
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  get(name: string): AnyTool<TState> | undefined {
    return this.tools.get(name);
  }

  /** Names of every registered tool, including hidden ones. */
  names(): string[] {
    return [...this.tools.keys()];
  }

  /** Every tool, including hidden ones. */
  all(): AnyTool<TState>[] {
    return [...this.tools.values()];
  }

  get size(): number {
    return this.tools.size;
  }

  /**
   * A new registry containing only `names`. Unknown names throw — a typo in a
   * scoping call should fail loudly, not silently drop a capability.
   */
  only(names: readonly string[]): ToolRegistry<TState> {
    const next = new ToolRegistry<TState>([], { logger: this.logger });
    for (const name of names) {
      const found = this.tools.get(name);
      if (found === undefined) {
        throw new ConfigurationError(
          `Cannot scope to unknown tool "${name}". Available: ${this.names().join(', ')}`,
          { name, available: this.names() },
        );
      }
      next.add(found);
    }
    return next;
  }

  exclude(names: readonly string[]): ToolRegistry<TState> {
    const excluded = new Set(names);
    const next = new ToolRegistry<TState>([], { logger: this.logger });
    for (const [name, definition] of this.tools) {
      if (!excluded.has(name)) next.add(definition);
    }
    return next;
  }

  /** Shallow copy — safe to mutate without affecting the original. */
  clone(): ToolRegistry<TState> {
    return new ToolRegistry(this, { logger: this.logger });
  }

  /** Rename tools in place (useful when models expect `snake_case`). */
  rename(map: Readonly<Record<string, string>>): this {
    const entries = [...this.tools.entries()];
    this.tools.clear();
    for (const [name, definition] of entries) {
      const nextName = map[name] ?? name;
      this.tools.set(nextName, { ...definition, name: nextName });
    }
    return this;
  }

  /**
   * Project tools down to the provider-facing shape, hiding hidden tools and
   * attaching a JSON Schema for the API.
   */
  toSpecs(options: { includeHidden?: boolean } = {}): ToolSpec[] {
    const specs: ToolSpec[] = [];
    for (const definition of this.tools.values()) {
      if (definition.hidden === true && options.includeHidden !== true) continue;
      specs.push({
        name: definition.name,
        description: definition.description,
        parameters: this.validateSchema(definition),
      });
    }
    return specs;
  }

  /** The JSON Schema the provider will see for `name`. */
  jsonSchemaFor(name: string): Record<string, unknown> | undefined {
    const definition = this.tools.get(name);
    if (definition === undefined) return undefined;
    return definition.jsonSchema ?? toJsonSchema(definition.parameters);
  }

  /**
   * The validator a tool advertises to the provider.
   *
   * A tool may describe its arguments with `parameters` (a validator, converted
   * to JSON Schema on the way out) or with an explicit `jsonSchema`. When only
   * the latter is present it is wrapped so it reaches the API verbatim.
   */
  private validateSchema(
    definition: ToolDefinition<unknown, unknown, unknown>,
  ): Schema<unknown> {
    if (definition.parameters !== undefined) return definition.parameters;
    if (definition.jsonSchema === undefined) return anySchema();
    return schemaFromJsonSchema<unknown>(definition.jsonSchema);
  }

  [Symbol.iterator](): Iterator<AnyTool<TState>> {
    return this.tools.values();
  }
}

/**
 * Normalize the many shapes `Agent` accepts for `tools` into one registry.
 *
 * ```ts
 * toToolRegistry([weather, email]);            // array
 * toToolRegistry(existingRegistry);            // registry
 * toToolRegistry(weather);                     // single tool
 * ```
 */
export function toToolRegistry<TState = unknown>(
  input: ToolRegistry<AnyState> | Iterable<AnyTool<TState>> | AnyTool<TState> | undefined,
): ToolRegistry<TState> {
  if (input === undefined) return new ToolRegistry<TState>();
  if (input instanceof ToolRegistry) return input as ToolRegistry<TState>;
  if (typeof (input as Iterable<ToolDefinition>)[Symbol.iterator] === 'function') {
    return new ToolRegistry<TState>(input as Iterable<AnyTool<TState>>);
  }
  return new ToolRegistry<TState>([input as AnyTool<TState>]);
}
