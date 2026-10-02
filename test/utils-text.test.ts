import { describe, expect, it } from 'vitest';

import { TypedEventEmitter } from '../src/utils/emitter.js';
import {
  estimateMessageTokens,
  estimateTokens,
  estimateValueTokens,
} from '../src/utils/tokens.js';
import {
  compact,
  extractJson,
  normalizeText,
  stringifyToolResult,
  toSnakeCase,
  toTitleCase,
  tokenize,
  truncate,
} from '../src/utils/text.js';

describe('toSnakeCase', () => {
  it.each([
    ['getWeather', 'get_weather'],
    ['GetWeather', 'get_weather'],
    ['get weather', 'get_weather'],
    ['get-weather', 'get_weather'],
    ['list_directory', 'list_directory'],
    ['--weird--name--', 'weird_name'],
    ['', ''],
  ])('converts %o to %o', (input, expected) => {
    expect(toSnakeCase(input)).toBe(expected);
  });

  it('splits acronym runs without doubling underscores', () => {
    expect(toSnakeCase('getHTTPResponse')).toBe('get_httpresponse');
    expect(toSnakeCase('parseHTML5Document')).toBe('parse_html5_document');
  });
});

describe('toTitleCase', () => {
  it('title-cases a snake or kebab identifier', () => {
    expect(toTitleCase('get_weather')).toBe('Get Weather');
    expect(toTitleCase('send-email')).toBe('Send Email');
  });
});

describe('tokenize', () => {
  it('splits identifiers into words', () => {
    expect(tokenize('get_user_profile')).toEqual(['get', 'user', 'profile']);
  });

  it('lowercases and drops punctuation', () => {
    expect(tokenize('Hello, World!')).toEqual(['hello', 'world']);
  });

  it('returns an empty array for empty input', () => {
    expect(tokenize('')).toEqual([]);
  });

  it('is stable for mixed whitespace and casing', () => {
    expect(tokenize('  Foo   BAR  ')).toEqual(['foo', 'bar']);
  });
});

describe('truncate', () => {
  it('leaves short strings alone', () => {
    expect(truncate('short', 100)).toBe('short');
  });

  it('appends an ellipsis when it cuts', () => {
    const result = truncate('a'.repeat(200), 50);
    expect(result.length).toBeLessThanOrEqual(50);
    expect(result.endsWith('…')).toBe(true);
  });

  it('handles a length of 0', () => {
    expect(truncate('anything', 0)).toBe('');
  });

  it('never exceeds the requested length', () => {
    for (const max of [1, 5, 10, 50]) {
      expect(truncate('x'.repeat(500), max).length).toBeLessThanOrEqual(max);
    }
  });
});

describe('extractJson', () => {
  it('parses a bare JSON object', () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 });
  });

  it('strips markdown fences', () => {
    const text = ['```json', '{"a":1}', '```'].join('\n');
    expect(extractJson(text)).toEqual({ a: 1 });
  });

  it('strips unlabelled fences', () => {
    expect(extractJson(['```', '{"a":1}', '```'].join('\n'))).toEqual({ a: 1 });
  });

  it('finds JSON embedded in prose', () => {
    expect(extractJson('Here you go: {"a":1} — hope that helps')).toEqual({ a: 1 });
  });

  it('finds a JSON array', () => {
    expect(extractJson('results: [1,2,3]')).toEqual([1, 2, 3]);
  });

  it('preserves nested braces and strings containing braces', () => {
    expect(extractJson('x {"a":{"b":"}"}} y')).toEqual({ a: { b: '}' } });
  });

  it('returns undefined when there is no JSON at all', () => {
    expect(extractJson('no json here')).toBeUndefined();
    expect(extractJson('')).toBeUndefined();
  });
});

describe('stringifyToolResult', () => {
  it('passes strings through', () => {
    expect(stringifyToolResult('already text')).toBe('already text');
  });

  it('renders undefined as an empty string', () => {
    expect(stringifyToolResult(undefined)).toBe('');
  });

  it('renders null and primitives', () => {
    expect(stringifyToolResult(null)).toBe('null');
    expect(stringifyToolResult(42)).toBe('42');
    expect(stringifyToolResult(true)).toBe('true');
  });

  it('pretty-prints objects', () => {
    expect(stringifyToolResult({ a: 1 })).toBe('{\n  "a": 1\n}');
  });

  it('never leaks [object Object]', () => {
    for (const value of [{}, new Map(), new Set(), () => {}, Object.create(null)]) {
      expect(stringifyToolResult(value)).not.toContain('[object Object]');
    }
  });

  it('survives circular structures', () => {
    const circular: Record<string, unknown> = { a: 1 };
    circular['self'] = circular;
    expect(() => stringifyToolResult(circular)).not.toThrow();
    expect(stringifyToolResult(circular)).toContain('Object');
  });

  it('handles bigint and symbol', () => {
    expect(stringifyToolResult(10n)).toBe('10n');
    expect(stringifyToolResult(Symbol('tag'))).toBe('Symbol(tag)');
  });
});

describe('compact', () => {
  it('drops nullish and empty values', () => {
    expect(compact({ a: 1, b: undefined, c: null, d: '', e: 0, f: false })).toEqual({
      a: 1,
      e: 0,
      f: false,
    });
  });

  it('returns an empty object when everything is dropped', () => {
    expect(compact({ a: undefined })).toEqual({});
  });
});

describe('normalizeText', () => {
  it('collapses whitespace and trims', () => {
    expect(normalizeText('  a   b  ')).toBe('a b');
  });
});

describe('estimateTokens', () => {
  it('grows with length', () => {
    expect(estimateTokens('hello world')).toBeGreaterThan(0);
    expect(estimateTokens('a'.repeat(4_000))).toBeGreaterThan(
      estimateTokens('a'.repeat(40)),
    );
  });

  it('returns 0 for an empty string', () => {
    expect(estimateTokens('')).toBe(0);
  });

  it('is deterministic', () => {
    expect(estimateTokens('repeatable')).toBe(estimateTokens('repeatable'));
  });
});

describe('estimateValueTokens', () => {
  it('counts strings as text', () => {
    expect(estimateValueTokens('hello')).toBe(estimateTokens('hello'));
  });

  it('counts undefined as free', () => {
    expect(estimateValueTokens(undefined)).toBe(0);
  });

  it('counts objects via their JSON form', () => {
    expect(estimateValueTokens({ a: 1 })).toBeGreaterThan(0);
  });

  it('handles circular values without throwing', () => {
    const circular: Record<string, unknown> = {};
    circular['self'] = circular;
    expect(() => estimateValueTokens(circular)).not.toThrow();
  });
});

describe('estimateMessageTokens', () => {
  it('counts a whole message list', () => {
    const one = estimateMessageTokens([{ role: 'user', content: 'hi' }]);
    const two = estimateMessageTokens([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hi' },
    ]);
    expect(two).toBeGreaterThan(one);
  });

  it('adds per-message overhead even for empty content', () => {
    expect(estimateMessageTokens([{ role: 'user', content: '' }])).toBeGreaterThan(0);
  });

  it('counts assistant tool calls', () => {
    const withoutCalls = estimateMessageTokens([{ role: 'assistant', content: 'ok' }]);
    const withCalls = estimateMessageTokens([
      {
        role: 'assistant',
        content: 'ok',
        toolCalls: [{ id: 'a', name: 'search', arguments: { query: 'weather' } }],
      },
    ]);
    expect(withCalls).toBeGreaterThan(withoutCalls);
  });

  it('counts tool result content', () => {
    const bare = estimateMessageTokens([{ role: 'tool', content: '' }]);
    const filled = estimateMessageTokens([
      {
        role: 'tool',
        content: '',
        toolResult: { content: 'x'.repeat(200), name: 'search' },
      },
    ]);
    expect(filled).toBeGreaterThan(bare);
  });

  it('returns 0 for an empty transcript', () => {
    expect(estimateMessageTokens([])).toBe(0);
  });
});

describe('TypedEventEmitter', () => {
  type Map1 = {
    ping: [value: number];
    pong: [value: string, extra?: boolean];
  };

  it('delivers arguments to every listener', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: number[] = [];
    emitter.on('ping', (value) => seen.push(value));
    emitter.on('ping', (value) => seen.push(value * 2));

    emitter.emit('ping', 3);
    expect(seen).toEqual([3, 6]);
  });

  it('keeps an `on` listener subscribed across emits', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: number[] = [];
    emitter.on('ping', (value) => seen.push(value));

    emitter.emit('ping', 1);
    emitter.emit('ping', 2);
    expect(seen).toEqual([1, 2]);
    expect(emitter.listenerCount('ping')).toBe(1);
  });

  it('once() fires a single time', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: number[] = [];
    emitter.once('ping', (value) => seen.push(value));

    emitter.emit('ping', 1);
    emitter.emit('ping', 2);
    expect(seen).toEqual([1]);
  });

  it('onMany() subscribes several events and unsubscribes them together', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: string[] = [];
    const off = emitter.onMany({
      ping: (value) => seen.push(`ping${value}`),
      pong: (value) => seen.push(`pong${value}`),
    });

    emitter.emit('ping', 1);
    emitter.emit('pong', 'x');
    expect(seen).toEqual(['ping1', 'pongx']);

    off();
    emitter.emit('ping', 2);
    expect(seen).toEqual(['ping1', 'pongx']);
  });

  it('returns an unsubscribe function from on()', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: number[] = [];
    const off = emitter.on('ping', (value) => seen.push(value));

    off();
    emitter.emit('ping', 1);
    expect(seen).toEqual([]);
  });

  it('off() removes a specific listener and reports whether it did', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: string[] = [];
    const listener = (value: string) => seen.push(value);

    emitter.on('pong', listener);
    expect(emitter.off('pong', listener)).toBe(true);
    expect(emitter.off('pong', listener)).toBe(false);

    emitter.emit('pong', 'x');
    expect(seen).toEqual([]);
  });

  it('removeAllListeners() with no argument clears everything', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: string[] = [];
    emitter.on('ping', (value) => seen.push(`p${value}`));
    emitter.on('pong', (value) => seen.push(`q${value}`));

    emitter.removeAllListeners();
    emitter.emit('ping', 1);
    emitter.emit('pong', 'x');
    expect(seen).toEqual([]);
  });

  it('removeAllListeners(event) clears only that event', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: string[] = [];
    emitter.on('ping', (value) => seen.push(`p${value}`));
    emitter.on('pong', (value) => seen.push(`q${value}`));

    emitter.removeAllListeners('ping');
    emitter.emit('ping', 1);
    emitter.emit('pong', 'x');
    expect(seen).toEqual(['qx']);
  });

  it('an error in one listener does not stop the others', () => {
    const emitter = new TypedEventEmitter<Map1>();
    const seen: number[] = [];
    emitter.on('ping', () => {
      throw new Error('listener blew up');
    });
    emitter.on('ping', (value) => seen.push(value));

    emitter.emit('ping', 7);
    expect(seen).toEqual([7]);
  });

  it('waitFor resolves on the next matching emit', async () => {
    const emitter = new TypedEventEmitter<Map1>();
    const waiting = emitter.waitFor('pong', () => true);
    emitter.emit('pong', 'ready');
    await expect(waiting).resolves.toBe('ready');
  });

  it('counts listeners and lists active events', () => {
    const emitter = new TypedEventEmitter<Map1>();
    emitter.on('ping', () => undefined);
    emitter.on('ping', () => undefined);
    expect(emitter.listenerCount('ping')).toBe(2);
    expect(emitter.listenerCount('pong')).toBe(0);
    expect(emitter.eventNames()).toEqual(['ping']);
  });

  it('emit returns the number of listeners invoked', () => {
    const emitter = new TypedEventEmitter<Map1>();
    expect(emitter.emit('ping', 1)).toBe(0);
    emitter.on('ping', () => undefined);
    expect(emitter.emit('ping', 1)).toBe(1);
  });

  it('waitFor ignores events that fail the predicate', async () => {
    const emitter = new TypedEventEmitter<Map1>();
    const waiting = emitter.waitFor('pong', (value) => value === 'ready');
    emitter.emit('pong', 'not yet');
    emitter.emit('pong', 'ready');
    await expect(waiting).resolves.toBe('ready');
  });
});
