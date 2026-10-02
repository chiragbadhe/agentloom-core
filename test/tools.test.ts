import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import {
  ConfigurationError,
  ToolApprovalRequiredError,
  ToolDeniedError,
  ToolNotFoundError,
  ToolTimeoutError,
  ToolValidationError,
} from '../src/errors.js';
import type { ToolCall } from '../src/providers/types.js';
import { ToolExecutor } from '../src/tools/executor.js';
import { defineTool, toToolRegistry, tool, ToolRegistry } from '../src/tools/registry.js';
import type {
  AnyTool,
  PolicyContext,
  ToolContext,
  ToolDefinition,
} from '../src/tools/types.js';
import { sleep } from '../src/utils/async.js';

const echo = defineTool({
  name: 'echo',
  description: 'Echo the input back.',
  parameters: z.object({ text: z.string() }),
  execute: (args) => ({ echoed: args.text }),
});

const call = (name: string, args: unknown, id = `call_${name}`): ToolCall => ({
  id,
  name,
  arguments: args,
});

const run = <TState = unknown>(state: TState = undefined as TState) => ({
  runId: 'run_1',
  iteration: 1,
  signal: new AbortController().signal,
  state,
});

const contextFor = <TState>(args: unknown, state: TState): ToolContext<TState> =>
  ({
    runId: 'run_1',
    iteration: 1,
    signal: new AbortController().signal,
    state,
    call: call('x', args),
    logger: { debug() {}, info() {}, warn() {}, error() {} },
  }) as unknown as ToolContext<TState>;

describe('defineTool', () => {
  it('returns the definition unchanged when valid', () => {
    expect(defineTool(echo)).toBe(echo);
  });

  it('rejects an empty name', () => {
    expect(() => defineTool({ name: '  ', description: 'x', execute: () => 1 })).toThrow(
      ConfigurationError,
    );
  });

  it('rejects an invalid name', () => {
    for (const name of ['1bad', 'has space', 'bad!', 'a'.repeat(65)]) {
      expect(() => defineTool({ name, description: 'x', execute: () => 1 })).toThrow(
        ConfigurationError,
      );
    }
  });

  it('requires a description', () => {
    expect(() => defineTool({ name: 'ok', description: ' ', execute: () => 1 })).toThrow(
      /must have a description/,
    );
  });

  it('requires an execute function', () => {
    expect(() =>
      defineTool({ name: 'ok', description: 'x' } as unknown as ToolDefinition),
    ).toThrow(/must define an execute function/);
  });

  it('exposes `tool` as an alias', () => {
    expect(tool).toBe(defineTool);
  });
});

describe('ToolRegistry', () => {
  it('adds, finds, and counts tools', () => {
    const registry = new ToolRegistry([echo]);
    expect(registry.has('echo')).toBe(true);
    expect(registry.get('echo')?.name).toBe('echo');
    expect(registry.names()).toEqual(['echo']);
    expect(registry.size).toBe(1);
  });

  it('replaces a tool while keeping its position', () => {
    const registry = new ToolRegistry([echo]);
    registry.add(defineTool({ name: 'echo', description: 'v2', execute: () => 1 }));
    registry.add(defineTool({ name: 'other', description: 'x', execute: () => 1 }));

    expect(registry.names()).toEqual(['echo', 'other']);
    expect(registry.get('echo')?.description).toBe('v2');
    expect(registry.size).toBe(2);
  });

  it('removes a tool', () => {
    const registry = new ToolRegistry([echo]);
    expect(registry.remove('echo')).toBe(true);
    expect(registry.remove('echo')).toBe(false);
    expect(registry.size).toBe(0);
  });

  it('scopes with only() and fails loudly on a typo', () => {
    const second = defineTool({ name: 'second', description: 'x', execute: () => 1 });
    const registry = new ToolRegistry([echo, second]);

    expect(registry.only(['echo']).names()).toEqual(['echo']);
    expect(registry.exclude(['echo']).names()).toEqual(['second']);
    expect(() => registry.only(['nope'])).toThrow(ConfigurationError);
  });

  it('clones without aliasing the original', () => {
    const registry = new ToolRegistry([echo]);
    const copy = registry.clone();
    copy.add(defineTool({ name: 'extra', description: 'x', execute: () => 1 }));

    expect(registry.names()).toEqual(['echo']);
    expect(copy.size).toBe(2);
  });

  it('renames in place', () => {
    const registry = new ToolRegistry([echo]);
    registry.rename({ echo: 'echo_text' });
    expect(registry.names()).toEqual(['echo_text']);
    expect(registry.get('echo_text')?.description).toBe(echo.description);
  });

  it('hides hidden tools from toSpecs but keeps them addressable', () => {
    const hidden = defineTool({
      name: 'secret',
      description: 'x',
      hidden: true,
      execute: () => 1,
    });
    const registry = new ToolRegistry([echo, hidden]);

    expect(registry.toSpecs().map((s) => s.name)).toEqual(['echo']);
    expect(registry.toSpecs({ includeHidden: true }).map((s) => s.name)).toEqual([
      'echo',
      'secret',
    ]);
    expect(registry.get('secret')).toBeDefined();
  });

  it('derives the JSON Schema from parameters', () => {
    const registry = new ToolRegistry([echo]);
    expect(registry.jsonSchemaFor('echo')).toMatchObject({
      type: 'object',
      required: ['text'],
    });
    expect(registry.jsonSchemaFor('missing')).toBeUndefined();
  });

  it('prefers an explicit jsonSchema over the derived one', () => {
    const registry = new ToolRegistry([
      defineTool({
        name: 'custom',
        description: 'x',
        parameters: z.object({ a: z.string() }),
        jsonSchema: {
          type: 'object',
          properties: { a: { type: 'string', format: 'uri' } },
        },
        execute: () => 1,
      }),
    ]);
    expect(registry.jsonSchemaFor('custom')).toMatchObject({
      properties: { a: { type: 'string', format: 'uri' } },
    });
  });

  it('is iterable', () => {
    const registry = new ToolRegistry([echo]);
    expect([...registry].map((t) => t.name)).toEqual(['echo']);
  });

  it('constructs from another registry', () => {
    const registry = new ToolRegistry(new ToolRegistry([echo]));
    expect(registry.names()).toEqual(['echo']);
  });
});

describe('toToolRegistry', () => {
  it('accepts a single tool, an array, a registry, or undefined', () => {
    expect(toToolRegistry(echo).names()).toEqual(['echo']);
    expect(toToolRegistry([echo]).names()).toEqual(['echo']);
    expect(toToolRegistry(new ToolRegistry([echo])).names()).toEqual(['echo']);
    expect(toToolRegistry(undefined).names()).toEqual([]);
  });

  it('is iterable, so a Set works too', () => {
    expect(toToolRegistry(new Set([echo])).names()).toEqual(['echo']);
  });
});

describe('ToolExecutor', () => {
  const executor = (tools: readonly AnyTool[], options = {}) =>
    new ToolExecutor({ registry: toToolRegistry(tools), ...options });

  it('returns tool messages in call order', async () => {
    const messages = await executor([echo]).execute(
      [call('echo', { text: 'first' }), call('echo', { text: 'second' }, 'call_2')],
      run(),
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]?.toolCallId).toBe('call_echo');
    expect(messages[0]?.content).toContain('first');
    expect(messages[1]?.toolCallId).toBe('call_2');
    expect(messages[1]?.isError).toBeUndefined();
  });

  it('reports an unknown tool as an error result, not a throw', async () => {
    const messages = await executor([echo]).execute([call('nope', {})], run());
    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toContain('nope');
  });

  it('rejects arguments that fail the schema', async () => {
    const messages = await executor([echo]).execute([call('echo', { text: 42 })], run());
    expect(messages[0]?.isError).toBe(true);
  });

  it('parses string arguments that the model sent as JSON text', async () => {
    const messages = await executor([echo]).execute(
      [call('echo', JSON.stringify({ text: 'from json' }))],
      run(),
    );
    expect(messages[0]?.isError).toBeUndefined();
    expect(messages[0]?.content).toContain('from json');
  });

  it('captures a thrown tool error so the model can retry', async () => {
    const boom = defineTool({
      name: 'boom',
      description: 'always fails',
      execute: () => {
        throw new Error('kaboom');
      },
    });

    const messages = await executor([boom]).execute([call('boom', {})], run());
    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toContain('kaboom');
  });

  it('throws instead when throwOnToolError is set', async () => {
    const boom = defineTool({
      name: 'boom',
      description: 'always fails',
      execute: () => {
        throw new Error('kaboom');
      },
    });

    await expect(
      executor([boom], { throwOnToolError: true }).execute([call('boom', {})], run()),
    ).rejects.toThrow(/kaboom/);
  });

  it('honours a policy denial', async () => {
    const messages = await executor([echo], {
      policy: () => ({ action: 'deny', reason: 'not allowed in this environment' }),
    }).execute([call('echo', { text: 'x' })], run());

    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toContain('not allowed in this environment');
  });

  it('never runs a denied tool', async () => {
    const spy = vi.fn(() => ({ ran: true }));
    const guarded = defineTool<Record<string, never>, { ran: boolean }>({
      name: 'guarded',
      description: 'x',
      execute: spy,
    });

    await executor([guarded], {
      policy: () => ({ action: 'deny', reason: 'nope' }),
    }).execute([call('guarded', {})], run());
    expect(spy).not.toHaveBeenCalled();
  });

  it('fails closed when a policy throws', async () => {
    const messages = await executor([echo], {
      policy: () => {
        throw new Error('policy exploded');
      },
    }).execute([call('echo', { text: 'x' })], run());

    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toContain('policy exploded');
  });

  it('awaits an async policy', async () => {
    const messages = await executor([echo], {
      policy: async () => ({ action: 'allow' as const }),
    }).execute([call('echo', { text: 'x' })], run());
    expect(messages[0]?.isError).toBeUndefined();
  });

  it('passes the call and run context to the policy', async () => {
    const policy = vi.fn((_call: ToolCall, _context: PolicyContext) => ({
      action: 'allow' as const,
    }));
    await executor([echo], { policy }).execute(
      [call('echo', { text: 'x' })],
      run({ user: 'ada' }),
    );

    expect(policy).toHaveBeenCalledTimes(1);
    expect(policy.mock.calls[0]?.[0]).toMatchObject({
      id: 'call_echo',
      name: 'echo',
      arguments: { text: 'x' },
    });
    expect(policy.mock.calls[0]?.[1]).toMatchObject({
      runId: 'run_1',
      iteration: 1,
      state: { user: 'ada' },
    });
  });

  it('runs tools sequentially by default', async () => {
    const order: string[] = [];
    const make = (name: string) =>
      defineTool({
        name,
        description: 'x',
        execute: async () => {
          order.push(`start:${name}`);
          await sleep(5);
          order.push(`end:${name}`);
          return name;
        },
      });

    await executor([make('a'), make('b')]).execute([call('a', {}), call('b', {})], run());

    expect(order).toEqual(['start:a', 'end:a', 'start:b', 'end:b']);
  });

  it('runs tools in parallel when asked', async () => {
    let active = 0;
    let peak = 0;
    const slow = defineTool({
      name: 'slow',
      description: 'x',
      execute: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(10);
        active -= 1;
        return 'done';
      },
    });

    await executor([slow], { mode: 'parallel', concurrency: 3 }).execute(
      [call('slow', {}, 'c1'), call('slow', {}, 'c2'), call('slow', {}, 'c3')],
      run(),
    );

    expect(peak).toBeGreaterThan(1);
  });

  it('respects the concurrency limit in parallel mode', async () => {
    let active = 0;
    let peak = 0;
    const slow = defineTool({
      name: 'slow',
      description: 'x',
      execute: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await sleep(5);
        active -= 1;
        return 'done';
      },
    });

    await executor([slow], { mode: 'parallel', concurrency: 2 }).execute(
      Array.from({ length: 6 }, (_, i) => call('slow', {}, `c${i}`)),
      run(),
    );

    expect(peak).toBeLessThanOrEqual(2);
  });

  it('gives each tool the run context', async () => {
    const seen: ToolContext<{ user: string }>[] = [];
    const stateful = defineTool<{ text: string }, unknown, { user: string }>({
      name: 'stateful',
      description: 'x',
      parameters: z.object({ text: z.string() }),
      execute: (args, context) => {
        seen.push(context);
        return { args, user: context.state.user };
      },
    });

    const messages = await executor([stateful]).execute(
      [call('stateful', { text: 'hi' })],
      run({ user: 'ada' }),
    );

    expect(seen[0]?.state.user).toBe('ada');
    expect(seen[0]?.iteration).toBe(1);
    expect(seen[0]?.call.name).toBe('stateful');
    expect(messages[0]?.content).toContain('ada');
  });

  it('truncates oversized results', async () => {
    const huge = defineTool({
      name: 'huge',
      description: 'x',
      execute: () => 'x'.repeat(50_000),
    });

    const messages = await executor([huge], { maxResultLength: 100 }).execute(
      [call('huge', {})],
      run(),
    );

    const content = messages[0]?.content ?? '';
    expect(content.length).toBeLessThan(50_000);
    expect(content).toContain('[truncated: 50000 characters total]');
  });

  it('uses a tool serialize() when provided', async () => {
    const formatted = defineTool({
      name: 'formatted',
      description: 'x',
      execute: () => ({ value: 42, secret: 'hide me' }),
      serialize: (result) => `value=${result.value}`,
    });

    const messages = await executor([formatted]).execute([call('formatted', {})], run());
    expect(messages[0]?.content).toBe('value=42');
  });

  it('times out a slow tool and reports it', async () => {
    const slow = defineTool({
      name: 'slow',
      description: 'x',
      timeoutMs: 20,
      execute: () => new Promise<string>(() => {}),
    });

    const messages = await executor([slow]).execute([call('slow', {})], run());
    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toMatch(/timed out|timed out/i);
  });

  it('retries a tool up to maxRetries', async () => {
    let attempts = 0;
    const flaky = defineTool({
      name: 'flaky',
      description: 'x',
      maxRetries: 2,
      execute: () => {
        attempts += 1;
        if (attempts < 3) throw new Error('transient');
        return 'recovered';
      },
    });

    const messages = await executor([flaky], {
      retry: { maxAttempts: 3, initialDelayMs: 1, jitter: 0 },
    }).execute([call('flaky', {})], run());

    expect(attempts).toBe(3);
    expect(messages[0]?.isError).toBeUndefined();
  });

  it('requires approval when the tool asks for it', async () => {
    const guarded = defineTool({
      name: 'guarded',
      description: 'x',
      requiresApproval: true,
      execute: () => 'ran',
    });

    const messages = await executor([guarded]).execute([call('guarded', {})], run());
    expect(messages[0]?.isError).toBe(true);
    expect(messages[0]?.content).toMatch(/approval/i);
  });

  it('runs a tool once approval is granted', async () => {
    const guarded = defineTool({
      name: 'guarded',
      description: 'x',
      requiresApproval: true,
      execute: () => 'ran',
    });

    const messages = await executor([guarded], {
      approval: () => Promise.resolve(true),
    }).execute([call('guarded', {})], run());
    expect(messages[0]?.isError).toBeUndefined();
    expect(messages[0]?.content).toContain('ran');
  });

  it('reports a denial from the approval handler', async () => {
    const guarded = defineTool({
      name: 'guarded',
      description: 'x',
      requiresApproval: true,
      execute: () => 'ran',
    });

    const messages = await executor([guarded], {
      approval: () => Promise.resolve(false),
    }).execute([call('guarded', {})], run());
    expect(messages[0]?.isError).toBe(true);
  });

  it('returns an empty array for no calls', async () => {
    await expect(executor([echo]).execute([], run())).resolves.toEqual([]);
  });

  it('propagates aborts instead of swallowing them', async () => {
    const controller = new AbortController();
    controller.abort();
    const slow = defineTool({
      name: 'slow',
      description: 'x',
      execute: () => {
        controller.signal.throwIfAborted();
        return 'never';
      },
    });

    const messages = await executor([slow]).execute([call('slow', {})], {
      runId: 'r',
      iteration: 1,
      signal: controller.signal,
      state: undefined,
    });
    expect(messages[0]?.isError).toBe(true);
  });
});

describe('error classes raised by the executor', () => {
  it('exposes ToolNotFoundError for a missing tool', () => {
    const error = new ToolNotFoundError('x', ['a', 'b']);
    expect(error.availableTools).toEqual(['a', 'b']);
  });

  it('exposes ToolTimeoutError', () => {
    expect(new ToolTimeoutError('slow').retryable).toBe(true);
  });

  it('exposes ToolValidationError with issues', () => {
    expect(new ToolValidationError('bad', { issues: [] }).issues).toEqual([]);
  });

  it('exposes ToolDeniedError with a reason', () => {
    expect(new ToolDeniedError('t', 'policy').reason).toBe('policy');
  });

  it('exposes ToolApprovalRequiredError', () => {
    expect(new ToolApprovalRequiredError('t').toolName).toBe('t');
  });
});

describe('ToolExecutor context helper', () => {
  it('builds a minimal context', () => {
    const context = contextFor({ a: 1 }, { user: 'ada' });
    expect(context.state).toEqual({ user: 'ada' });
    expect(context.iteration).toBe(1);
  });
});
