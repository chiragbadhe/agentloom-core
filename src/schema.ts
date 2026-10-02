/**
 * Structural schema interface.
 *
 * `@agentloom/core` deliberately does **not** import from `zod` at type level,
 * even though `zod` is a peer dependency. Anything that implements
 * {@link Schema} works — including `zod` v3 and v4 schemas, `valibot`, and
 * hand-written validators — which keeps the runtime free of a hard coupling
 * and avoids duplicate-install bugs.
 *
 * @example
 * ```ts
 * import { z } from 'zod';
 * const User = z.object({ name: z.string() }); // satisfies Schema<{name: string}>
 * ```
 */
export interface SchemaIssue {
  /** Dotted path to the offending value, e.g. `['address', 'zip']`. */
  readonly path: readonly (string | number)[];
  /** Human readable description of the problem. */
  readonly message: string;
  /** Machine readable issue code (`too_small`, `invalid_type`, ...). */
  readonly code?: string;
}

export interface SchemaSuccess<T> {
  readonly success: true;
  readonly data: T;
}

export interface SchemaFailure {
  readonly success: false;
  readonly error: SchemaError;
}

export type SchemaResult<T> = SchemaSuccess<T> | SchemaFailure;

/** Normalized error shape produced by {@link Schema.safeParse}. */
export interface SchemaError {
  readonly issues: readonly SchemaIssue[];
}

/**
 * The minimal contract `@agentloom/core` requires to validate structured data.
 *
 * Zod schemas satisfy this interface structurally:
 * - `z.ZodType` (v3) / `z.ZodType` (v4) expose `safeParse`.
 */
export interface Schema<T> {
  /** Parse without throwing. Returns a discriminated result. */
  safeParse(input: unknown): SchemaResult<T>;
  /**
   * Parse and throw on failure. Optional — nothing in the kit requires it, so
   * hand-written validators only need `safeParse`.
   */
  parse?(input: unknown): T;
  /**
   * Exact JSON Schema for this validator. Providers prefer it over a converted
   * approximation, so it is set by {@link schemaFromJsonSchema}.
   */
  readonly jsonSchema?: Record<string, unknown>;
}

/** Extract the output type produced by a {@link Schema}. */
export type Infer<S> = S extends Schema<infer T> ? T : never;

const ANY: Schema<unknown> = {
  parse: (input: unknown) => input,
  safeParse: (input: unknown) => ({ success: true, data: input }),
};

/**
 * The permissive schema used when a tool or agent does not declare one.
 * Everything validates; nothing is coerced.
 */
export function anySchema(): Schema<unknown> {
  return ANY;
}

export function isSchema(value: unknown): value is Schema<unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as Schema<unknown>).safeParse === 'function'
  );
}

/** Format issues into a compact, model-friendly string. */
export function formatIssues(issues: readonly SchemaIssue[], limit = 8): string {
  const shown = issues.slice(0, limit);
  const lines = shown.map((issue) => {
    const path = issue.path.length > 0 ? issue.path.join('.') : '(root)';
    return `- ${path}: ${issue.message}`;
  });
  if (issues.length > shown.length) {
    lines.push(`- ...and ${issues.length - shown.length} more`);
  }
  return lines.join('\n');
}

/**
 * Build a schema from a JSON Schema object. Used for native structured-output
 * mode, where the model enforces the shape and we validate the response.
 *
 * ```ts
 * const user = schemaFromJsonSchema<{ name: string }>({
 *   type: 'object',
 *   properties: { name: { type: 'string' } },
 *   required: ['name'],
 * });
 * ```
 */
export function schemaFromJsonSchema<T>(jsonSchema: Record<string, unknown>): Schema<T> {
  const validate = (input: unknown): SchemaResult<T> => {
    const issues = validateJsonSchema(input, jsonSchema);
    return issues.length === 0
      ? { success: true, data: input as T }
      : { success: false, error: { issues } };
  };
  return {
    // Kept so the exact schema reaches provider APIs instead of a lossy
    // re-conversion of the validator.
    jsonSchema,
    safeParse: validate,
    parse: (input: unknown) => {
      const result = validate(input);
      if (!result.success) {
        throw new SchemaValidationError(result.error.issues);
      }
      return result.data;
    },
  };
}

/** Thrown by `schema.parse()` when validation fails. */
export class SchemaValidationError extends Error {
  readonly issues: readonly SchemaIssue[];

  constructor(issues: readonly SchemaIssue[]) {
    super(`Validation failed:\n${formatIssues(issues)}`);
    this.name = 'SchemaValidationError';
    this.issues = issues;
  }
}

interface JsonSchemaNode {
  type?: string | string[];
  required?: readonly string[];
  properties?: Record<string, JsonSchemaNode>;
  items?: JsonSchemaNode;
  enum?: readonly unknown[];
  additionalProperties?: boolean | JsonSchemaNode;
}

/**
 * A deliberately small JSON Schema validator covering the subset used by
 * structured outputs (type, properties, required, items, enum). Anything it
 * cannot prove invalid is accepted — the model/provider is the primary
 * enforcer, this is the safety net.
 */
function validateJsonSchema(
  input: unknown,
  schema: JsonSchemaNode,
  path: (string | number)[] = [],
): SchemaIssue[] {
  const issues: SchemaIssue[] = [];

  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => matchesType(input, t))) {
      issues.push({
        path,
        message: `expected ${types.join(' or ')} but received ${describe(input)}`,
        code: 'invalid_type',
      });
      return issues;
    }
  }

  if (schema.enum !== undefined && !schema.enum.includes(input)) {
    issues.push({
      path,
      message: `expected one of ${JSON.stringify(schema.enum)}`,
      code: 'invalid_enum_value',
    });
  }

  if (isRecord(input) && schema.properties !== undefined) {
    for (const key of schema.required ?? []) {
      if (!(key in input)) {
        issues.push({ path: [...path, key], message: 'required', code: 'required' });
      }
    }
    for (const [key, value] of Object.entries(input)) {
      const child = schema.properties[key];
      if (child !== undefined) {
        issues.push(...validateJsonSchema(value, child, [...path, key]));
      }
    }
  }

  if (Array.isArray(input) && schema.items !== undefined) {
    input.forEach((item, index) => {
      issues.push(
        ...validateJsonSchema(item, schema.items as JsonSchemaNode, [...path, index]),
      );
    });
  }

  return issues;
}

function matchesType(input: unknown, type: string): boolean {
  switch (type) {
    case 'string':
      return typeof input === 'string';
    case 'number':
      return typeof input === 'number' && Number.isFinite(input);
    case 'integer':
      return typeof input === 'number' && Number.isInteger(input);
    case 'boolean':
      return typeof input === 'boolean';
    case 'null':
      return input === null;
    case 'array':
      return Array.isArray(input);
    case 'object':
      return isRecord(input);
    default:
      return true;
  }
}

function describe(input: unknown): string {
  if (input === null) return 'null';
  if (Array.isArray(input)) return 'array';
  if (typeof input === 'number') return Number.isFinite(input) ? 'number' : 'NaN';
  return typeof input;
}

function isRecord(input: unknown): input is Record<string, unknown> {
  return typeof input === 'object' && input !== null && !Array.isArray(input);
}
