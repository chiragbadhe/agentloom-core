import type { Schema } from '../schema.js';

/**
 * Best-effort conversion of a validation schema into JSON Schema for the
 * provider's native tool/response-format APIs.
 *
 * This is intentionally dependency-free: instead of pulling in a
 * `zod-to-json-schema` dependency we duck-type the two zod internal shapes
 * (`_def.typeName` on v3, `_def.type` on v4). Any other validator yields a
 * permissive `{ type: 'object' }`.
 *
 * When the result is not good enough, override it explicitly:
 *
 * ```ts
 * defineTool({ name, parameters: MySchema, jsonSchema: myExactSchema, ... })
 * ```
 */
export function toJsonSchema(schema: unknown): Record<string, unknown> {
  // An explicit schema always wins: re-converting it would be lossy.
  const explicit = (schema as { jsonSchema?: unknown } | undefined)?.jsonSchema;
  if (explicit !== null && typeof explicit === 'object') {
    return explicit as Record<string, unknown>;
  }
  const converted = convert(schema, 0);
  return converted ?? { type: 'object' };
}

/** Guard against pathological schemas (recursive expansion). */
const MAX_DEPTH = 12;

function convert(schema: unknown, depth: number): Record<string, unknown> | undefined {
  if (schema === null || typeof schema !== 'object' || depth > MAX_DEPTH)
    return undefined;

  const def = (schema as { _def?: Record<string, unknown> })._def;
  if (def === undefined || def === null || typeof def !== 'object') {
    return { type: 'object' };
  }

  const kind = readKind(def);
  switch (kind) {
    case 'string':
      return { type: 'string' };
    case 'number':
      return { type: 'number' };
    case 'bigint':
    case 'date':
      return { type: 'string' };
    case 'boolean':
      return { type: 'boolean' };
    case 'null':
      return { type: 'null' };
    case 'undefined':
    case 'void':
    case 'never':
      return { not: {} };
    case 'any':
    case 'unknown':
      return {};
    case 'nan':
      return { type: 'number' };
    case 'literal': {
      const value = def.value;
      return typeof value === 'object' && value !== null && !(value instanceof Date)
        ? {}
        : { const: value };
    }
    case 'enum': {
      const values = readEnumValues(def);
      return values.length > 0 ? { type: 'string', enum: values } : {};
    }
    case 'nativeenum': {
      const values = readEnumValues(def);
      return values.length > 0 ? { enum: values } : {};
    }
    case 'array': {
      const items = convert(firstDefined(def, 'element', 'type', 'valueType'), depth + 1);
      const node: Record<string, unknown> = { type: 'array' };
      if (items) node.items = items;
      if (typeof def.minLength === 'number') node.minItems = def.minLength;
      if (typeof def.maxLength === 'number') node.maxItems = def.maxLength;
      return node;
    }
    case 'tuple': {
      const rest = def.rest;
      const inner = Array.isArray(def.items)
        ? def.items.map((item) => convert(item, depth + 1)).filter(isDefined)
        : [];
      if (rest && !isEmptyRecord(rest)) {
        const items = convert(rest, depth + 1);
        return {
          type: 'array',
          prefixItems: inner,
          ...(items ? { items } : {}),
        };
      }
      return inner.length > 0 ? { type: 'array', prefixItems: inner } : { type: 'array' };
    }
    case 'set': {
      const items = convert(firstDefined(def, 'valueType', 'element'), depth + 1);
      const node: Record<string, unknown> = { type: 'array', uniqueItems: true };
      if (items) node.items = items;
      return node;
    }
    case 'object': {
      const shape = readShape(def);
      if (shape === undefined) return { type: 'object' };
      const properties: Record<string, unknown> = {};
      const required: string[] = [];
      for (const [key, value] of Object.entries(shape)) {
        const node = convert(value, depth + 1);
        properties[key] = node ?? {};
        if (!isOptionalSchema(value)) required.push(key);
      }
      const result: Record<string, unknown> = { type: 'object', properties };
      if (required.length > 0) result.required = required;
      const extra = def.unknownKeys;
      if (extra === 'passthrough') result.additionalProperties = true;
      else if (extra === 'strict') result.additionalProperties = false;
      return result;
    }
    case 'record': {
      const values = convert(firstDefined(def, 'valueType', 'value'), depth + 1);
      const node: Record<string, unknown> = { type: 'object' };
      if (values) node.additionalProperties = values;
      return node;
    }
    case 'map': {
      const values = convert(def.valueType, depth + 1);
      return values
        ? { type: 'object', additionalProperties: values }
        : { type: 'object' };
    }
    case 'union': {
      const options = Array.isArray(def.options) ? def.options : [];
      const nodes = options.map((option) => convert(option, depth + 1)).filter(isDefined);
      if (nodes.length === 0) return {};
      return nodes.length === 1 ? nodes[0]! : { anyOf: nodes };
    }
    case 'discriminatedunion': {
      return convert(def.options, depth + 1) ?? {};
    }
    case 'intersection': {
      const left = convert(def.left, depth + 1);
      const right = convert(def.right, depth + 1);
      if (left && right) return { allOf: [left, right] };
      return left ?? right ?? {};
    }
    case 'nullable': {
      const inner = convert(def.innerType, depth + 1);
      if (inner === undefined) return {};
      if (inner.type === undefined) return inner;
      const type = inner.type;
      const types: unknown[] = Array.isArray(type) ? [...(type as unknown[])] : [type];
      return { ...inner, type: [...types, 'null'] };
    }
    case 'optional':
    case 'default':
    case 'readonly':
    case 'catch':
    case 'brand':
    case 'pipe': {
      const inner = firstDefined(def, 'innerType', 'schema', 'type', 'in');
      return convert(inner, depth + 1) ?? {};
    }
    case 'lazy': {
      return convert(readGetter(def, 'getter'), depth + 1) ?? {};
    }
    case 'promise':
      return convert(def.type, depth + 1) ?? {};
    case 'effects':
    case 'transform':
      return convert(firstDefined(def, 'schema', 'in'), depth + 1) ?? {};
    default:
      return { type: 'object' };
  }
}

function readKind(def: Record<string, unknown>): string {
  const typeName = def.typeName;
  if (typeof typeName === 'string' && typeName.startsWith('Zod')) {
    return typeName.slice(3).toLowerCase();
  }
  const type = def.type;
  if (typeof type === 'string') return type.toLowerCase();
  return '';
}

/** v3 stores `shape` as a getter; v4 stores a plain object. */
function readShape(def: Record<string, unknown>): Record<string, unknown> | undefined {
  const shape = readGetter(def, 'shape');
  return typeof shape === 'object' && shape !== null
    ? (shape as Record<string, unknown>)
    : undefined;
}

function readGetter(def: Record<string, unknown>, key: string): unknown {
  const value = def[key];
  return typeof value === 'function' ? (value as () => unknown)() : value;
}

function readEnumValues(def: Record<string, unknown>): unknown[] {
  const direct = def.values;
  if (Array.isArray(direct)) return direct;
  const entries = def.entries;
  if (Array.isArray(entries)) return entries;
  if (typeof entries === 'object' && entries !== null) return Object.values(entries);
  return [];
}

function firstDefined(def: Record<string, unknown>, ...keys: readonly string[]): unknown {
  for (const key of keys) {
    const value = def[key];
    if (value !== undefined) return value;
  }
  return undefined;
}

function isOptionalSchema(schema: unknown): boolean {
  const def = (schema as { _def?: Record<string, unknown> } | undefined)?._def;
  if (def === undefined) return false;
  const kind = readKind(def);
  if (kind === 'optional') return true;
  // v4 makes the field optional when `_def.isOptional` is set.
  return def.isOptional === true || def.optional === true;
}

function isEmptyRecord(value: unknown): boolean {
  return typeof value === 'object' && value !== null && Object.keys(value).length === 0;
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

/**
 * Convert a schema straight into an OpenAI/Anthropic-style function
 * parameter definition.
 */
export function toParametersSchema(
  schema: Schema<unknown> | undefined,
): Record<string, unknown> {
  if (schema === undefined) return { type: 'object', properties: {} };
  return toJsonSchema(schema);
}
