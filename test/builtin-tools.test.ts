import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { ToolExecutionError, ToolValidationError } from '../src/errors.js';
import {
  createCalculatorTool,
  evaluateExpression,
  supportedFunctions,
} from '../src/tools/builtin/calculator.js';
import { createDateTimeTool, parseDuration } from '../src/tools/builtin/datetime.js';
import {
  createFileSystemTool,
  resolveInsideRoot,
} from '../src/tools/builtin/filesystem.js';
import {
  createFetchUrlTool,
  createHttpTool,
  isUrlAllowed,
} from '../src/tools/builtin/http.js';
import { createSleepTool } from '../src/tools/builtin/sleep.js';
import type { ToolContext } from '../src/tools/types.js';

/** Built-ins return plain values; wrap so `await` and `.resolves` work uniformly. */
async function exec<TArgs, TResult>(
  tool: { execute(args: TArgs, context: ToolContext): TResult | Promise<TResult> },
  args: TArgs,
): Promise<TResult> {
  return tool.execute(args, context);
}

const context: ToolContext = {
  signal: new AbortController().signal,
  runId: 'run_test',
  state: undefined,
  call: { id: 'c1', name: 'tool', arguments: {} },
  iteration: 1,
  logger: nullLogger(),
};

function nullLogger(): ToolContext['logger'] {
  const noop = () => undefined;
  return { debug: noop, info: noop, warn: noop, error: noop, child: () => nullLogger() };
}

/** Build a `fetch` stand-in that returns a fixed response. */
const stubFetch = (
  body: string,
  init: { status?: number; headers?: Record<string, string> } = {},
): typeof globalThis.fetch => {
  return async () =>
    new Response(body, {
      status: init.status ?? 200,
      headers: init.headers ?? { 'content-type': 'text/plain' },
    });
};

describe('evaluateExpression', () => {
  it.each([
    ['1 + 1', 2],
    ['2 * (3 + 4)', 14],
    ['10 / 4', 2.5],
    ['10 % 3', 1],
    ['2 ^ 10', 1024],
    ['-5 + 3', -2],
    ['1 + 2 * 3 ^ 2', 19],
    ['(1 + 2) * (3 + 4)', 21],
  ])('evaluates %s', (expression, expected) => {
    expect(evaluateExpression(expression)).toBeCloseTo(expected, 10);
  });

  it('supports constants', () => {
    expect(evaluateExpression('pi')).toBeCloseTo(Math.PI, 10);
    expect(evaluateExpression('e')).toBeCloseTo(Math.E, 10);
    expect(evaluateExpression('tau')).toBeCloseTo(Math.PI * 2, 10);
  });

  it('supports functions', () => {
    expect(evaluateExpression('sqrt(16)')).toBe(4);
    expect(evaluateExpression('abs(-3)')).toBe(3);
    expect(evaluateExpression('min(3, 1, 2)')).toBe(1);
    expect(evaluateExpression('max(3, 1, 2)')).toBe(3);
    expect(evaluateExpression('round(pi, 2)')).toBe(3.14);
    expect(evaluateExpression('floor(2.9)')).toBe(2);
    expect(evaluateExpression('ceil(2.1)')).toBe(3);
  });

  it('lists supported functions', () => {
    expect(supportedFunctions()).toEqual(
      expect.arrayContaining(['sqrt', 'abs', 'min', 'max']),
    );
  });

  it('rejects unbalanced parentheses', () => {
    expect(() => evaluateExpression('(1 + 2')).toThrow(SyntaxError);
  });

  it('rejects unknown identifiers', () => {
    expect(() => evaluateExpression('process.exit(1)')).toThrow(SyntaxError);
  });

  it('rejects arbitrary code', () => {
    expect(() => evaluateExpression('require("fs")')).toThrow(SyntaxError);
    expect(() => evaluateExpression('1; 2')).toThrow(SyntaxError);
  });

  it('rejects an empty expression', () => {
    expect(() => evaluateExpression('   ')).toThrow(SyntaxError);
  });

  it('rejects a dangling operator', () => {
    expect(() => evaluateExpression('1 +')).toThrow(SyntaxError);
  });

  it('rejects an unknown function', () => {
    expect(() => evaluateExpression('nope(1)')).toThrow(SyntaxError);
  });

  it('rejects wrong arity', () => {
    expect(() => evaluateExpression('sqrt(1, 2, 3)')).toThrow(SyntaxError);
  });
});

describe('createCalculatorTool', () => {
  const tool = createCalculatorTool();

  it('is described for the model', () => {
    expect(tool.name).toBe('calculator');
    expect(tool.description).toContain('expression');
    expect(tool.jsonSchema?.['properties']).toHaveProperty('expression');
  });

  it('evaluates an expression', async () => {
    await expect(exec(tool, { expression: '2 * (3 + 4)' })).resolves.toEqual({
      expression: '2 * (3 + 4)',
      result: 14,
    });
  });

  it('validates arguments', () => {
    expect(tool.parameters?.safeParse({ expression: '1+1' }).success).toBe(true);
    expect(tool.parameters?.safeParse({}).success).toBe(false);
    expect(tool.parameters?.safeParse({ expression: 42 }).success).toBe(false);
  });

  it('enforces the length limit', () => {
    const small = createCalculatorTool({ maxLength: 5 });
    expect(small.parameters?.safeParse({ expression: '1 + 1 + 1 + 1' }).success).toBe(
      false,
    );
    expect(tool.parameters?.safeParse({ expression: '1 + 1' }).success).toBe(true);
  });

  it('rounds to the requested digits', async () => {
    const result = await exec(tool, { expression: 'round(pi, 2)' });
    expect(result.result).toBe(3.14);
  });

  it('caps digits in the serialized form', async () => {
    const capped = createCalculatorTool({ maxDigits: 2 });
    const result = await exec(capped, { expression: 'round(pi, 8)' });
    expect(result.result).toBe(3.14159265);
    expect(capped.serialize?.(result)).toBe('round(pi, 8) = 3.14');
  });
});

describe('parseDuration', () => {
  it.each([
    ['90s', 90_000],
    ['5m', 300_000],
    ['2h', 7_200_000],
    ['3d', 259_200_000],
    ['1w', 604_800_000],
    ['500ms', 500],
  ])('parses %s', (input, expected) => {
    expect(parseDuration(input)).toBe(expected);
  });

  it('handles signs and compound units', () => {
    expect(parseDuration('-3d')).toBe(-259_200_000);
    expect(parseDuration('+2h30m')).toBe(2 * 3_600_000 + 30 * 60_000);
  });

  it('ignores surrounding whitespace and case', () => {
    expect(parseDuration('  1H ')).toBe(3_600_000);
  });

  it('rejects nonsense', () => {
    expect(() => parseDuration('soon')).toThrow(ToolExecutionError);
    expect(() => parseDuration('')).toThrow(ToolExecutionError);
  });
});

describe('createDateTimeTool', () => {
  const tool = createDateTimeTool();

  it('reports the current time', async () => {
    const before = Date.now();
    const result = await exec(tool, { action: 'now' });
    expect(result.unix).toBeGreaterThanOrEqual(Math.floor(before / 1000) - 2);
    expect(result.iso).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(result.weekday.length).toBeGreaterThan(0);
  });

  it('parses an ISO date', async () => {
    const result = await exec(tool, {
      action: 'parse',
      input: '2024-03-01T12:00:00.000Z',
    });
    expect(result.unix).toBe(Math.floor(Date.parse('2024-03-01T12:00:00.000Z') / 1000));
  });

  it('formats a date', async () => {
    const result = await exec(tool, {
      action: 'format',
      input: '2024-03-01T12:00:00.000Z',
      format: 'date',
    });
    expect(result.formatted).toBe('2024-03-01');
  });

  it('formats unix timestamps', async () => {
    const result = await exec(tool, {
      action: 'format',
      input: '2024-03-01T12:00:00.000Z',
      format: 'unix',
    });
    expect(result.formatted).toBe(String(Date.parse('2024-03-01T12:00:00.000Z') / 1000));
  });

  it('adds a duration', async () => {
    const result = await exec(tool, {
      action: 'add',
      input: '2024-03-01T00:00:00.000Z',
      amount: '2d',
    });
    expect(result.iso.slice(0, 10)).toBe('2024-03-03');
  });

  it('subtracts a duration', async () => {
    const result = await exec(tool, {
      action: 'add',
      input: '2024-03-01T00:00:00.000Z',
      amount: '-1d',
    });
    expect(result.iso.slice(0, 10)).toBe('2024-02-29');
  });

  it('diffs two dates', async () => {
    const result = await exec(tool, {
      action: 'diff',
      input: '2024-03-01T00:00:00.000Z',
      other: '2024-03-08T00:00:00.000Z',
    });
    expect(result.extra).toMatchObject({ deltaMs: 604_800_000, deltaDays: 7 });
  });

  it('honours the timezone', async () => {
    // 03:30 UTC on the 2nd is still the 1st in New York.
    const utc = await exec(tool, {
      action: 'format',
      input: '2024-03-02T03:30:00.000Z',
      format: 'date',
    });
    const ny = await exec(tool, {
      action: 'format',
      input: '2024-03-02T03:30:00.000Z',
      timezone: 'America/New_York',
      format: 'date',
    });
    expect(utc.formatted).toBe('2024-03-02');
    expect(ny.formatted).toBe('2024-03-01');
    expect(ny.unix).toBe(utc.unix);
  });

  it('formats local times and relative dates', async () => {
    const time = await exec(tool, {
      action: 'format',
      input: '2024-03-01T14:05:09.000Z',
      format: 'time',
    });
    expect(time.formatted).toBe('14:05:09');

    const relative = await exec(tool, {
      action: 'format',
      input: '2024-03-01T14:05:09.000Z',
      format: 'relative',
    });
    expect(relative.formatted).toBe('Mar 1, 2024 14:05 UTC');
  });

  it('rejects an invalid timezone', async () => {
    await expect(exec(tool, { action: 'now', timezone: 'Mars/Olympus' })).rejects.toThrow(
      ToolExecutionError,
    );
  });

  it('rejects an invalid date', async () => {
    await expect(exec(tool, { action: 'parse', input: 'not a date' })).rejects.toThrow(
      ToolExecutionError,
    );
  });

  it('validates arguments', () => {
    expect(tool.parameters?.safeParse({ action: 'now' }).success).toBe(true);
    expect(tool.parameters?.safeParse({ action: 'explode' }).success).toBe(false);
    expect(tool.parameters?.safeParse({}).success).toBe(false);
  });
});

describe('createSleepTool', () => {
  const tool = createSleepTool(1_000);

  it('sleeps', async () => {
    const started = Date.now();
    const result = await exec(tool, { ms: 20, reason: 'cooldown' });
    const elapsed = Date.now() - started;
    expect(elapsed).toBeGreaterThanOrEqual(15);
    expect(result.sleptMs).toBeGreaterThanOrEqual(15);
    expect(result.reason).toBe('cooldown');
  });

  it('resolves early when aborted', async () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 5);
    const result = await tool.execute(
      { ms: 5_000 },
      { ...context, signal: controller.signal },
    );
    expect(result.sleptMs).toBeLessThan(5_000);
  });

  it('validates the duration', () => {
    expect(tool.parameters?.safeParse({ ms: 10 }).success).toBe(true);
    expect(tool.parameters?.safeParse({ ms: -1 }).success).toBe(false);
    expect(tool.parameters?.safeParse({ ms: 5_000_000 }).success).toBe(false);
    expect(tool.parameters?.safeParse({ ms: 'soon' }).success).toBe(false);
  });
});

describe('isUrlAllowed', () => {
  it('allows ordinary https hosts', () => {
    expect(isUrlAllowed('https://example.com/a', {}).allowed).toBe(true);
  });

  it('rejects malformed URLs', () => {
    const result = isUrlAllowed('not a url', {});
    expect(result.allowed).toBe(false);
    expect(result.allowed === false && result.reason).toContain('valid URL');
  });

  it('rejects non-http protocols', () => {
    const result = isUrlAllowed('file:///etc/passwd', {});
    expect(result.allowed).toBe(false);
    expect(result.allowed === false && result.reason).toContain('protocol');
  });

  it('blocks private and loopback hosts by default', () => {
    for (const url of [
      'http://localhost:8080/',
      'http://127.0.0.1/',
      'http://10.0.0.5/',
      'http://192.168.1.1/',
      'http://172.16.0.1/',
      'http://169.254.169.254/latest/meta-data',
    ]) {
      const result = isUrlAllowed(url, {});
      expect(result.allowed, url).toBe(false);
      expect(result.allowed === false && result.reason).toContain('private network');
    }
  });

  it('allows private hosts when explicitly permitted', () => {
    expect(
      isUrlAllowed('http://127.0.0.1:3000/', { blockPrivateHosts: false }).allowed,
    ).toBe(true);
  });

  it('honours a blocklist', () => {
    const result = isUrlAllowed('https://evil.test/', { blockedHosts: ['evil.test'] });
    expect(result.allowed).toBe(false);
    expect(result.allowed === false && result.reason).toContain('blocked');
  });

  it('honours an allowlist, including wildcards', () => {
    const options = { allowedHosts: ['api.example.com', '*.trusted.test'] };
    expect(isUrlAllowed('https://api.example.com/v1', options).allowed).toBe(true);
    expect(isUrlAllowed('https://docs.trusted.test/', options).allowed).toBe(true);
    expect(isUrlAllowed('https://trusted.test/', options).allowed).toBe(true);
    expect(isUrlAllowed('https://elsewhere.test/', options).allowed).toBe(false);
  });
});

describe('createHttpTool', () => {
  it('fetches a url', async () => {
    const tool = createHttpTool({ fetch: stubFetch('hello world') });
    const result = await exec(tool, { url: 'https://example.com/' });

    expect(result).toMatchObject({ status: 200, ok: true, body: 'hello world' });
  });

  it('defaults to GET', async () => {
    let method = 'unknown';
    const fetchImpl = (async (_url: unknown, init?: { method?: string }) => {
      method = init?.method ?? 'GET';
      return new Response('ok');
    }) as unknown as typeof globalThis.fetch;

    await createHttpTool({ fetch: fetchImpl }).execute(
      { url: 'https://example.com/' },
      context,
    );
    expect(method).toBe('GET');
  });

  it('posts a JSON body', async () => {
    let body: unknown;
    const fetchImpl = (async (_url: unknown, init?: { body?: string }) => {
      body = init?.body;
      return new Response('{}', { headers: { 'content-type': 'application/json' } });
    }) as unknown as typeof globalThis.fetch;

    await createHttpTool({ fetch: fetchImpl }).execute(
      { url: 'https://example.com/', action: 'post', body: { a: 1 } },
      context,
    );
    expect(JSON.parse(String(body))).toEqual({ a: 1 });
  });

  it('parses JSON bodies', async () => {
    const tool = createHttpTool({
      fetch: stubFetch('{"ok":true}', {
        headers: { 'content-type': 'application/json' },
      }),
    });
    const result = await exec(tool, { url: 'https://example.com/', format: 'json' });
    expect(JSON.parse(result.body)).toEqual({ ok: true });
  });

  it('truncates long bodies', async () => {
    const tool = createHttpTool({
      fetch: stubFetch('x'.repeat(5_000)),
      maxResponseChars: 100,
    });
    const result = await exec(tool, { url: 'https://example.com/' });

    expect(result.truncated).toBe(true);
    expect(result.body.length).toBeLessThan(200);
    expect(result.truncated).toBe(true);
    expect(result.body).toContain('truncated after');
  });

  it('surfaces error statuses without throwing', async () => {
    const tool = createHttpTool({ fetch: stubFetch('nope', { status: 404 }) });
    const result = await exec(tool, { url: 'https://example.com/' });
    expect(result.ok).toBe(false);
    expect(result.status).toBe(404);
  });

  it('refuses blocked hosts before making a request', async () => {
    const fetchImpl = vi.fn();
    const tool = createHttpTool({
      fetch: fetchImpl as unknown as typeof globalThis.fetch,
    });

    await expect(exec(tool, { url: 'http://127.0.0.1/secret' })).rejects.toThrow(
      /private network/,
    );
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('validates arguments', () => {
    const tool = createHttpTool({ fetch: stubFetch('x') });
    expect(tool.parameters?.safeParse({ url: 'https://example.com' }).success).toBe(true);
    expect(tool.parameters?.safeParse({}).success).toBe(false);
    expect(
      tool.parameters?.safeParse({ url: 'https://example.com', action: 'put' }).success,
    ).toBe(false);
  });

  it('rejects a missing fetch implementation', () => {
    expect(() =>
      createHttpTool({ fetch: undefined as unknown as typeof globalThis.fetch }),
    ).not.toThrow();
  });
});

describe('createFetchUrlTool', () => {
  it('returns just the body', async () => {
    const tool = createFetchUrlTool({ fetch: stubFetch('{"a":1}', {}) });
    await expect(exec(tool, { url: 'https://example.com/' })).resolves.toBe('{"a":1}');
  });

  it('is named fetch_url', () => {
    const tool = createFetchUrlTool({ fetch: stubFetch('x') });
    expect(tool.name).toBe('fetch_url');
  });
});

describe('createFileSystemTool', () => {
  const dirs: string[] = [];

  afterEach(async () => {
    await Promise.all(
      dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })),
    );
  });

  const sandbox = async (options: { allowWrite?: boolean } = {}) => {
    const dir = await mkdtemp(join(tmpdir(), 'agentloom-fs-'));
    dirs.push(dir);
    return { dir, tool: createFileSystemTool({ root: dir, ...options }) };
  };

  it('requires a root', () => {
    expect(() => createFileSystemTool({ root: '  ' })).toThrow(ToolValidationError);
  });

  it('writes and reads a file', async () => {
    const { tool } = await sandbox({ allowWrite: true });

    const written = await exec(tool, {
      action: 'write',
      path: 'notes.txt',
      content: 'hello',
    });
    expect(written).toMatchObject({ action: 'write', size: 5 });

    const read = await exec(tool, { action: 'read', path: 'notes.txt' });
    expect(read.content).toBe('hello');
  });

  it('creates missing parent directories on write', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await exec(tool, { action: 'write', path: 'a/b/c.txt', content: 'deep' });
    const read = await exec(tool, { action: 'read', path: 'a/b/c.txt' });
    expect(read.content).toBe('deep');
  });

  it('appends', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await exec(tool, { action: 'write', path: 'log.txt', content: 'a' });
    await exec(tool, { action: 'append', path: 'log.txt', content: 'b' });

    const read = await exec(tool, { action: 'read', path: 'log.txt' });
    expect(read.content).toBe('ab');
  });

  it('lists a directory with sizes', async () => {
    const { dir, tool } = await sandbox({ allowWrite: true });
    await writeFile(join(dir, 'one.txt'), '12345');
    await mkdir(join(dir, 'sub'));

    const listed = await exec(tool, { action: 'list', path: '.' });
    const entries = listed.entries ?? [];
    expect(entries.find((entry) => entry.name === 'one.txt')).toMatchObject({
      type: 'file',
      size: 5,
    });
    expect(entries.find((entry) => entry.name === 'sub')?.type).toBe('directory');
  });

  it('reports existence', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await expect(
      exec(tool, { action: 'exists', path: 'nope.txt' }),
    ).resolves.toMatchObject({ exists: false });
    await exec(tool, { action: 'write', path: 'yes.txt', content: 'x' });
    await expect(
      exec(tool, { action: 'exists', path: 'yes.txt' }),
    ).resolves.toMatchObject({ exists: true });
  });

  it('stats a file', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await exec(tool, { action: 'write', path: 'x.txt', content: 'abc' });

    const stat = await exec(tool, { action: 'stat', path: 'x.txt' });
    expect(stat.size).toBe(3);
    expect(stat.modifiedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it('deletes', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await exec(tool, { action: 'write', path: 'gone.txt', content: 'x' });
    await exec(tool, { action: 'delete', path: 'gone.txt' });

    await expect(
      exec(tool, { action: 'exists', path: 'gone.txt' }),
    ).resolves.toMatchObject({ exists: false });
  });

  it('refuses writes unless allowed', async () => {
    const { tool } = await sandbox();
    await expect(
      exec(tool, { action: 'write', path: 'x.txt', content: 'x' }),
    ).rejects.toThrow(/write access is disabled/);
  });

  it('refuses deletes unless allowed', async () => {
    const { tool } = await sandbox();
    await expect(exec(tool, { action: 'delete', path: 'x.txt' })).rejects.toThrow(
      /write access is disabled/,
    );
  });

  it('refuses traversal outside the sandbox', async () => {
    const { tool } = await sandbox();
    await expect(
      exec(tool, { action: 'read', path: '../../etc/passwd' }),
    ).rejects.toThrow(/escapes the sandbox/);
  });

  it('refuses an absolute path outside the sandbox', async () => {
    const { tool } = await sandbox();
    await expect(exec(tool, { action: 'read', path: '/etc/passwd' })).rejects.toThrow(
      /escapes the sandbox/,
    );
  });

  it('refuses a symlink that escapes the sandbox', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'agentloom-fs-'));
    dirs.push(dir);
    const outside = await mkdtemp(join(tmpdir(), 'agentloom-outside-'));
    dirs.push(outside);
    await writeFile(join(outside, 'secret.txt'), 'classified');
    await symlink(join(outside, 'secret.txt'), join(dir, 'link.txt'));

    const tool = createFileSystemTool({ root: dir });
    await expect(exec(tool, { action: 'read', path: 'link.txt' })).rejects.toThrow(
      /escapes the sandbox/,
    );
  });

  it('enforces the read size limit', async () => {
    const { dir, tool } = await sandbox({ allowWrite: true });
    await writeFile(join(dir, 'big.txt'), 'y'.repeat(1_000));

    const limited = createFileSystemTool({ root: dir, maxReadBytes: 100 });
    await expect(exec(limited, { action: 'read', path: 'big.txt' })).rejects.toThrow(
      /limit/,
    );
    expect(tool).toBeDefined();
  });

  it('enforces the write size limit', async () => {
    const { dir } = await sandbox({ allowWrite: true });
    const tool = createFileSystemTool({ root: dir, allowWrite: true, maxWriteBytes: 10 });

    await expect(
      exec(tool, { action: 'write', path: 'x.txt', content: 'y'.repeat(100) }),
    ).rejects.toThrow(/Refusing to write/);
  });

  it('supports base64 reads', async () => {
    const { tool } = await sandbox({ allowWrite: true });
    await exec(tool, { action: 'write', path: 'b.txt', content: 'hi' });

    const read = await exec(tool, { action: 'read', path: 'b.txt', encoding: 'base64' });
    expect(Buffer.from(read.content ?? '', 'base64').toString('utf8')).toBe('hi');
  });

  it('propagates a missing-file read error', async () => {
    const { tool } = await sandbox();
    await expect(exec(tool, { action: 'read', path: 'missing.txt' })).rejects.toThrow();
  });

  it('honours an aborted signal', async () => {
    const { tool } = await sandbox();
    const controller = new AbortController();
    controller.abort();

    await expect(
      tool.execute(
        { action: 'read', path: 'x.txt' },
        { ...context, signal: controller.signal },
      ),
    ).rejects.toThrow();
  });

  it('validates arguments', () => {
    const tool = createFileSystemTool({ root: '/tmp', allowWrite: true });
    expect(tool.parameters?.safeParse({ action: 'read', path: 'a' }).success).toBe(true);
    expect(tool.parameters?.safeParse({ action: 'nope', path: 'a' }).success).toBe(false);
    expect(tool.parameters?.safeParse({ action: 'read' }).success).toBe(false);
    expect(tool.parameters?.safeParse({ action: 'write', path: 'a' }).success).toBe(
      false,
    );
  });

  it('reads real files from disk through the injected fs', async () => {
    const { dir } = await sandbox();
    await writeFile(join(dir, 'real.txt'), 'from disk');

    const tool = createFileSystemTool({ root: dir });
    const nodeFs = await import('node:fs/promises');
    const read = await tool.execute(
      { action: 'read', path: 'real.txt' },
      { ...context, logger: context.logger },
    );
    expect(read.content).toBe('from disk');
    expect(typeof nodeFs.readFile).toBe('function');
  });
});

describe('resolveInsideRoot', () => {
  it('resolves relative paths inside the root', () => {
    expect(resolveInsideRoot('/tmp/sandbox', 'file.txt')).toBe('/tmp/sandbox/file.txt');
  });

  it('resolves absolute paths inside the root', () => {
    expect(resolveInsideRoot('/tmp/sandbox', '/tmp/sandbox/file.txt')).toBe(
      '/tmp/sandbox/file.txt',
    );
  });

  it('rejects traversal', () => {
    expect(() => resolveInsideRoot('/tmp/sandbox', '../secret')).toThrow();
    expect(() => resolveInsideRoot('/tmp/sandbox', 'a/../../secret')).toThrow();
  });
});
