import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import {
  anySchema,
  formatIssues,
  isSchema,
  schemaFromJsonSchema,
  SchemaValidationError,
  type Schema,
} from '../src/schema.js';

describe('anySchema', () => {
  it('accepts everything unchanged', () => {
    const schema = anySchema();
    for (const value of [undefined, null, 0, '', { a: 1 }, [1, 2]]) {
      expect(schema.safeParse(value)).toEqual({ success: true, data: value });
    }
  });

  it('returns the same singleton instance every time', () => {
    expect(anySchema()).toBe(anySchema());
  });
});

describe('isSchema', () => {
  it('detects zod v3 schemas', () => {
    expect(isSchema(z.string())).toBe(true);
    expect(isSchema(z.object({ a: z.string() }))).toBe(true);
  });

  it('detects hand-written validators', () => {
    const custom: Schema<number> = {
      safeParse: (input) =>
        typeof input === 'number'
          ? { success: true, data: input }
          : {
              success: false,
              error: { issues: [{ path: [], message: 'not a number' }] },
            },
      parse: (input) => {
        const result = custom.safeParse(input);
        if (!result.success) throw new SchemaValidationError(result.error.issues);
        return result.data;
      },
    };
    expect(isSchema(custom)).toBe(true);
    expect(custom.safeParse(1)).toEqual({ success: true, data: 1 });
  });

  it('rejects non-schemas', () => {
    for (const value of [null, undefined, 42, 'zod', {}, { parse: () => 1 }]) {
      expect(isSchema(value)).toBe(false);
    }
  });
});

describe('formatIssues', () => {
  it('renders dotted paths', () => {
    const text = formatIssues([
      { path: ['address', 'zip'], message: 'required' },
      { path: [], message: 'root problem' },
    ]);
    expect(text).toContain('address.zip: required');
    expect(text).toContain('(root): root problem');
  });

  it('caps the number of issues and reports the remainder', () => {
    const issues = Array.from({ length: 12 }, (_, i) => ({
      path: [`f${i}`],
      message: 'bad',
    }));
    const text = formatIssues(issues, 5);
    expect(text.split('\n')).toHaveLength(6);
    expect(text).toContain('and 7 more');
  });
});

describe('schemaFromJsonSchema', () => {
  const bookSchema = {
    type: 'object',
    required: ['title'],
    properties: {
      title: { type: 'string' },
      pages: { type: 'number' },
      tags: { type: 'array', items: { type: 'string' } },
    },
  } as const;

  it('validates required properties', () => {
    const schema = schemaFromJsonSchema<{ title: string }>(bookSchema);
    expect(schema.safeParse({ title: 'Dune' })).toEqual({
      success: true,
      data: { title: 'Dune' },
    });

    const bad = schema.safeParse({ pages: 1 });
    expect(bad.success).toBe(false);
    if (!bad.success) {
      expect(bad.error.issues[0]?.path).toEqual(['title']);
    }
  });

  it('reports the wrong type for a present property', () => {
    const schema = schemaFromJsonSchema(bookSchema);
    const result = schema.safeParse({ title: 7 });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]?.path).toEqual(['title']);
      expect(result.error.issues[0]?.code).toBe('invalid_type');
    }
  });

  it('validates array item types', () => {
    const schema = schemaFromJsonSchema(bookSchema);
    const result = schema.safeParse({ title: 'Dune', tags: ['sci-fi', 42] });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]?.path).toEqual(['tags', 1]);
    }
  });

  it('throws SchemaValidationError from parse()', () => {
    const schema = schemaFromJsonSchema(bookSchema);
    expect(() => schema.parse?.({})).toThrow(SchemaValidationError);
    expect(schema.parse?.({ title: 'Dune' })).toEqual({ title: 'Dune' });
  });
});

describe('zod interop', () => {
  it('validates structured agent output through the Schema interface', () => {
    const User = z.object({ name: z.string(), age: z.number().int() });
    const schema: Schema<{ name: string; age: number }> = User;

    expect(schema.safeParse({ name: 'Ada', age: 36 })).toEqual({
      success: true,
      data: { name: 'Ada', age: 36 },
    });

    const result = schema.safeParse({ name: 'Ada', age: 3.5 });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]?.path).toEqual(['age']);
    }
  });
});
