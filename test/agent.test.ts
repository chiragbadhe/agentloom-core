import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { Agent } from '../src/agent/agent.js';
import type { AgentConfig, PlannerConfig } from '../src/agent/config.js';
import {
  AbortError,
  ConfigurationError,
  ExecutionLimitError,
  ProviderError,
  ValidationError,
} from '../src/errors.js';
import { InMemoryConversationMemory } from '../src/memory/conversation.js';
import { InMemoryLongTermMemory } from '../src/memory/long-term.js';
import { defineTool } from '../src/tools/registry.js';
import type { ApprovalHandler, ToolExecute } from '../src/tools/types.js';
import type { MockProvider, ScriptedTurn } from './helpers/mock-provider.js';
import { NonStreamingProvider, mockProvider } from './helpers/mock-provider.js';

/** Echoes its input; handy as a scriptable, side-effect-free tool. */
const echoTool = defineTool<{ text: string }>({
  name: 'echo',
  description: 'Echo the given text.',
  jsonSchema: {
    type: 'object',
    properties: { text: { type: 'string' } },
    required: ['text'],
  },
  execute: (args) => ({ echoed: args.text }),
});

const addTool = defineTool<{ a: number; b: number }>({
  name: 'add',
  description: 'Add two numbers.',
  jsonSchema: {
    type: 'object',
    properties: { a: { type: 'number' }, b: { type: 'number' } },
    required: ['a', 'b'],
  },
  execute: (args) => ({ sum: args.a + args.b }),
});

/** A script that keeps calling `toolName` forever. */
const toolLoop = (count: number, toolName = 'echo'): ScriptedTurn[] =>
  Array.from({ length: count }, (_, index) => ({
    toolCalls: [{ id: `c${index}`, name: toolName, arguments: { text: 'again' } }],
  }));

type Config<TState = unknown> = Omit<
  AgentConfig<TState>,
  'model' | 'provider' | 'planner'
>;

/**
 * An agent backed by a scripted provider. Planning is off unless a test asks
 * for it, so the script is spent on the loop the test is about.
 */
function makeAgent<TState = unknown>(
  script: readonly ScriptedTurn[],
  config: Config<TState> = {},
): { agent: Agent<TState>; provider: MockProvider } {
  const provider = mockProvider({ script });
  const agent = new Agent<TState>({
    provider,
    model: 'mock-1',
    planner: false,
    ...config,
  });
  return { agent, provider };
}

describe('Agent construction', () => {
  it('requires a model or provider', () => {
    expect(() => new Agent({})).toThrow(ConfigurationError);
  });

  it('defaults the name and creates memory', () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    expect(agent.name).toBe('agent');
    expect(agent.memory.length).toBe(0);
    expect(agent.toolSpecs()).toEqual([]);
  });

  it('honours a custom name', () => {
    const { agent } = makeAgent([{ text: 'ok' }], { name: 'researcher' });
    expect(agent.name).toBe('researcher');
  });

  it('accepts a provider instance', () => {
    const provider = mockProvider();
    const agent = new Agent({ provider, model: 'custom-1' });
    expect(agent.model).toBe('custom-1');
    expect(agent.provider).toBe(provider);
  });

  it('falls back to the provider default model', () => {
    const agent = new Agent({ provider: mockProvider({ defaultModel: 'mock-9' }) });
    expect(agent.model).toBe('mock-9');
  });

  it('normalizes limits and exposes them', () => {
    const { agent } = makeAgent([{ text: 'ok' }], { limits: { maxIterations: 3 } });
    expect(agent.limits.maxIterations).toBe(3);
    expect(agent.limits.contextWindow).toBe(32_000);
  });

  it('registers tools from an array', () => {
    const { agent } = makeAgent([{ text: 'ok' }], { tools: [echoTool, addTool] });
    expect(agent.toolSpecs().map((spec) => spec.name)).toEqual(['echo', 'add']);
  });

  it('leaves planning off unless it is configured', () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    expect(agent.planningEnabled).toBe(false);
  });

  it('enables planning when a planner config is given', () => {
    const agent = new Agent({ provider: mockProvider(), planner: { maxSteps: 3 } });
    expect(agent.planningEnabled).toBe(true);
    expect(agent.currentPlan).toBeUndefined();
  });

  it('disables planning with planner: { enabled: false }', () => {
    const agent = new Agent({ provider: mockProvider(), planner: { enabled: false } });
    expect(agent.planningEnabled).toBe(false);
  });

  it('holds the configured state object', () => {
    const state = { user: 'ada' };
    const agent = new Agent<typeof state>({ provider: mockProvider(), state });
    expect(agent.state).toBe(state);
  });

  it('resolves providers from a model reference', () => {
    const agent = new Agent({ model: 'openai:gpt-4o-mini' });
    expect(agent.provider.id).toBe('openai');
    expect(agent.model).toBe('gpt-4o-mini');
  });
});

describe('Agent.run', () => {
  it('returns the final answer', async () => {
    const { agent } = makeAgent([{ text: 'hello there' }]);
    const result = await agent.run('hi');

    expect(result.output).toBe('hello there');
    expect(result.stopReason).toBe('completed');
    expect(result.data).toBeUndefined();
    expect(result.error).toBeUndefined();
    expect(result.iterations).toBe(1);
    expect(result.runId).toMatch(/^run_/);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('reports usage and steps', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    const result = await agent.run('hi');

    expect(result.usage).toMatchObject({
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
    });
    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]?.iteration).toBe(1);
    expect(result.steps[0]?.request.model).toBe('mock-1');
    expect(result.steps[0]?.toolCalls).toEqual([]);
    expect(result.steps[0]?.text).toBe('ok');
  });

  it('records the conversation in memory', async () => {
    const { agent } = makeAgent([{ text: 'hi' }]);
    const result = await agent.run('hello');

    expect(result.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
    expect(agent.memory.length).toBe(2);
  });

  it('keeps history across runs', async () => {
    const { agent, provider } = makeAgent([{ text: 'first' }, { text: 'second' }]);
    await agent.run('one');
    await agent.run('two');

    expect(agent.memory.length).toBe(4);
    expect(provider.lastRequest?.messages).toHaveLength(4);
  });

  it('sends the configured instructions as the system prompt', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      instructions: 'Be terse.',
    });
    await agent.run('hi');

    expect(provider.lastRequest?.messages[0]).toMatchObject({
      role: 'system',
      content: 'Be terse.',
    });
  });

  it('supports an instructions function', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      instructions: ({ input }) => `Task: ${input}`,
    });
    await agent.run('what is 2+2?');

    expect(provider.lastRequest?.messages[0]?.content).toBe('Task: what is 2+2?');
  });

  it('replaces instructions per run', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], { instructions: 'base' });
    await agent.run('hi', { instructions: 'override' });
    expect(provider.lastRequest?.messages[0]?.content).toBe('override');
  });

  it('appends per-run system text', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], { instructions: 'base' });
    await agent.run('hi', { system: 'extra context' });

    const system = provider.lastRequest?.messages[0]?.content ?? '';
    expect(system).toContain('base');
    expect(system).toContain('extra context');
  });

  it('uses a caller-supplied run id', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    await expect(agent.run('hi', { runId: 'run_custom' })).resolves.toMatchObject({
      runId: 'run_custom',
    });
  });

  it('forwards model options to the provider', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      modelOptions: { temperature: 0.1 },
    });
    await agent.run('hi');
    expect(provider.lastRequest?.temperature).toBe(0.1);
  });

  it('accepts a custom conversation memory', async () => {
    const memory = new InMemoryConversationMemory();
    const { agent } = makeAgent([{ text: 'ok' }], { memory });

    await agent.run('hi');
    expect(memory.length).toBe(2);
  });

  it('clears memory and plan on reset', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    await agent.run('hi');
    agent.reset();

    expect(agent.memory.length).toBe(0);
    expect(agent.currentPlan).toBeUndefined();
  });
});

describe('Agent.run with tools', () => {
  it('executes tool calls and feeds results back', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'call_1', name: 'echo', arguments: { text: 'ping' } }] },
        { text: 'the tool said ping' },
      ],
      { tools: [echoTool] },
    );

    const result = await agent.run('echo ping');

    expect(result.output).toBe('the tool said ping');
    expect(result.iterations).toBe(2);
    expect(result.steps[0]?.toolCalls).toHaveLength(1);
    expect(result.steps[0]?.toolResults[0]?.content).toContain('ping');
    expect(result.messages.find((message) => message.role === 'tool')).toMatchObject({
      name: 'echo',
      toolCallId: 'call_1',
    });
  });

  it('sends tool specs to the provider', async () => {
    const { agent, provider } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'add', arguments: { a: 1, b: 2 } }] },
        { text: 'three' },
      ],
      { tools: [addTool] },
    );
    await agent.run('add 1 and 2');

    expect(provider.requests[0]?.tools?.map((spec) => spec.name)).toEqual(['add']);
    const parameters = provider.requests[0]?.tools?.[0]?.parameters;
    expect(parameters?.jsonSchema).toMatchObject({
      type: 'object',
      properties: { a: { type: 'number' }, b: { type: 'number' } },
    });
  });

  it('omits tools when the provider has no tool support', async () => {
    const provider = mockProvider({
      script: [{ text: 'ok' }],
      capabilities: { tools: false },
    });
    const agent = new Agent({
      provider,
      model: 'mock-1',
      planner: false,
      tools: [echoTool],
    });

    await agent.run('hi');
    expect(provider.lastRequest?.tools).toBeUndefined();
  });

  it('runs several tool calls in one iteration', async () => {
    const { agent } = makeAgent(
      [
        {
          toolCalls: [
            { id: 'c1', name: 'add', arguments: { a: 1, b: 1 } },
            { id: 'c2', name: 'add', arguments: { a: 2, b: 2 } },
          ],
        },
        { text: 'done' },
      ],
      { tools: [addTool] },
    );

    const result = await agent.run('two adds');
    expect(result.steps[0]?.toolResults).toHaveLength(2);
    expect(result.iterations).toBe(2);
  });

  it('runs tools in parallel when configured', async () => {
    const slow = defineTool<{ ms: number }>({
      name: 'slow',
      description: 'Sleep then report.',
      jsonSchema: { type: 'object', properties: { ms: { type: 'number' } } },
      execute: async (args) => {
        await new Promise((resolve) => setTimeout(resolve, args.ms));
        return { slept: args.ms };
      },
    });
    const { agent } = makeAgent(
      [
        {
          toolCalls: [
            { id: 'c1', name: 'slow', arguments: { ms: 60 } },
            { id: 'c2', name: 'slow', arguments: { ms: 60 } },
          ],
        },
        { text: 'done' },
      ],
      { tools: [slow], execution: 'parallel' },
    );

    const started = Date.now();
    await agent.run('go');
    expect(Date.now() - started).toBeLessThan(300);
  });

  it('feeds tool failures back to the model', async () => {
    const failing = defineTool({
      name: 'boom',
      description: 'Always fails.',
      jsonSchema: { type: 'object', properties: {} },
      execute: () => {
        throw new Error('kaboom');
      },
    });
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'boom', arguments: {} }] },
        { text: 'I could not do that' },
      ],
      { tools: [failing] },
    );

    const errors: unknown[] = [];
    agent.on('tool:error', (event) => errors.push(event.error.message));

    const result = await agent.run('try boom');

    expect(result.output).toBe('I could not do that');
    expect(result.steps[0]?.toolResults[0]?.isError).toBe(true);
    expect(errors).toHaveLength(1);
  });

  it('throws on tool failure when configured to', async () => {
    const failing = defineTool({
      name: 'boom',
      description: 'Always fails.',
      jsonSchema: { type: 'object', properties: {} },
      execute: () => {
        throw new Error('kaboom');
      },
    });
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'boom', arguments: {} }] },
        { text: 'unreachable' },
      ],
      { tools: [failing], throwOnToolError: true },
    );

    await expect(agent.run('try boom')).rejects.toThrow();
  });

  it('reports an unknown tool as a tool error', async () => {
    const { agent } = makeAgent(
      [{ toolCalls: [{ id: 'c1', name: 'nope', arguments: {} }] }, { text: 'recovered' }],
      { tools: [echoTool] },
    );

    const result = await agent.run('go');
    expect(result.steps[0]?.toolResults[0]?.isError).toBe(true);
    expect(result.output).toBe('recovered');
  });

  it('honours a tool policy that denies a call', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'understood' },
      ],
      {
        tools: [echoTool],
        toolPolicy: () => ({ action: 'deny', reason: 'not allowed' }),
      },
    );

    const result = await agent.run('go');
    expect(result.steps[0]?.toolResults[0]?.content).toContain('not allowed');
  });

  it('asks for approval and skips the call when declined', async () => {
    const guarded = defineTool<{ text: string }>({
      ...echoTool,
      requiresApproval: true,
    });
    const approval = vi.fn<ApprovalHandler>(async () => false);
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'skipped' },
      ],
      { tools: [guarded], approval },
    );

    const result = await agent.run('go');
    expect(approval).toHaveBeenCalledTimes(1);
    expect(approval.mock.calls[0]?.[0]).toMatchObject({ toolName: 'echo' });
    expect(result.steps[0]?.toolResults[0]?.isError).toBe(true);
    expect(result.output).toBe('skipped');
  });

  it('only asks for approval when the guard requires it', async () => {
    const guarded = defineTool<{ text: string }>({
      ...echoTool,
      requiresApproval: (args) => args.text !== 'skip me',
    });
    const approval = vi.fn<ApprovalHandler>(async () => true);
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'go' } }] },
        { toolCalls: [{ id: 'c2', name: 'echo', arguments: { text: 'skip me' } }] },
        { text: 'done' },
      ],
      { tools: [guarded], approval },
    );

    const result = await agent.run('go');
    expect(approval).toHaveBeenCalledTimes(1);
    expect(approval.mock.calls[0]?.[0].args).toEqual({ text: 'go' });
    expect(result.steps[0]?.toolResults[0]?.isError).toBeUndefined();
    expect(result.steps[1]?.toolResults[0]?.isError).toBeUndefined();
  });

  it('skips a guarded call the approver rejects', async () => {
    const guarded = defineTool<{ text: string }>({
      ...echoTool,
      requiresApproval: (args) => args.text === 'danger',
    });
    const approval = vi.fn<ApprovalHandler>(async () => false);
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'danger' } }] },
        { text: 'fine' },
      ],
      { tools: [guarded], approval },
    );

    const result = await agent.run('go');
    expect(result.steps[0]?.toolResults[0]?.isError).toBe(true);
    expect(result.output).toBe('fine');
  });

  it('does not ask for approval for unguarded tools', async () => {
    const approval = vi.fn<ApprovalHandler>(async () => true);
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'done' },
      ],
      { tools: [echoTool], approval },
    );

    await agent.run('go');
    expect(approval).not.toHaveBeenCalled();
  });

  it('overrides tools per run', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], { tools: [echoTool] });
    await agent.run('hi', { tools: [addTool] });
    expect(provider.lastRequest?.tools?.map((spec) => spec.name)).toEqual(['add']);
  });

  it('passes run context and state to tools', async () => {
    const spy = vi.fn<
      ToolExecute<{ text: string }, { echoed: string }, { user: string }>
    >((args) => ({ echoed: args.text }));
    const stateful = defineTool<{ text: string }, { echoed: string }, { user: string }>({
      name: 'stateful',
      description: 'Reads state.',
      jsonSchema: { type: 'object', properties: { text: { type: 'string' } } },
      execute: spy,
    });
    const provider = mockProvider({
      script: [
        { toolCalls: [{ id: 'c1', name: 'stateful', arguments: { text: 'hi' } }] },
        { text: 'done' },
      ],
    });
    const agent = new Agent<{ user: string }>({
      provider,
      model: 'mock-1',
      planner: false,
      tools: [stateful],
      state: { user: 'ada' },
    });

    await agent.run('go');
    const context = spy.mock.calls[0]?.[1];
    expect(context?.state.user).toBe('ada');
    expect(context?.runId).toMatch(/^run_/);
  });

  it('truncates oversized tool output', async () => {
    const loud = defineTool({
      name: 'loud',
      description: 'Returns a lot of text.',
      jsonSchema: { type: 'object', properties: {} },
      execute: () => 'x'.repeat(500),
    });
    const { agent } = makeAgent(
      [{ toolCalls: [{ id: 'c1', name: 'loud', arguments: {} }] }, { text: 'ok' }],
      { tools: [loud], maxToolResultLength: 50 },
    );

    const result = await agent.run('go');
    const content = result.steps[0]?.toolResults[0]?.content ?? '';
    expect(content.length).toBeLessThan(120);
    expect(content).toMatch(/truncat/i);
  });
});

describe('Agent limits', () => {
  it('stops after maxIterations', async () => {
    const { agent } = makeAgent(toolLoop(10), {
      tools: [echoTool],
      limits: { maxIterations: 2 },
    });

    const result = await agent.run('loop forever');

    expect(result.stopReason).toBe('max_iterations');
    expect(result.iterations).toBe(2);
    expect(result.error).toBeInstanceOf(ExecutionLimitError);
    expect(result.error?.code).toBe('EXECUTION_LIMIT_ERROR');
  });

  it('accepts per-run limit overrides', async () => {
    const { agent } = makeAgent(toolLoop(10), {
      tools: [echoTool],
      limits: { maxIterations: 10 },
    });

    const result = await agent.run('loop', { limits: { maxIterations: 1 } });
    expect(result.stopReason).toBe('max_iterations');
  });

  it('stops when the token budget is exhausted', async () => {
    const { agent } = makeAgent(toolLoop(10), {
      tools: [echoTool],
      limits: { maxTotalTokens: 20 },
    });

    const result = await agent.run('loop');
    expect(result.stopReason).toBe('max_tokens');
  });

  it('refuses tool calls past the run budget but still lets the model answer', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'a' } }] },
        { toolCalls: [{ id: 'c2', name: 'echo', arguments: { text: 'b' } }] },
        { text: 'here is what I found' },
      ],
      { tools: [echoTool], limits: { maxToolCalls: 1, maxIterations: 5 } },
    );

    const result = await agent.run('loop');
    expect(result.steps[0]?.toolResults[0]?.isError).toBeUndefined();
    expect(result.steps[1]?.toolResults[0]?.isError).toBe(true);
    expect(result.output).toBe('here is what I found');
  });

  it('skips tool calls beyond maxToolCallsPerIteration and tells the model', async () => {
    const { agent } = makeAgent(
      [
        {
          toolCalls: [
            { id: 'c1', name: 'echo', arguments: { text: 'a' } },
            { id: 'c2', name: 'echo', arguments: { text: 'b' } },
          ],
        },
        { text: 'ok' },
      ],
      { tools: [echoTool], limits: { maxToolCallsPerIteration: 1 } },
    );

    const result = await agent.run('go');
    const results = result.steps[0]?.toolResults ?? [];
    expect(results).toHaveLength(2);
    expect(results[1]?.isError).toBe(true);
    expect(results[1]?.content).toContain('budget');
  });

  it('stops when the wall-clock timeout elapses', async () => {
    const slow = defineTool({
      name: 'slow',
      description: 'Sleeps.',
      jsonSchema: { type: 'object', properties: {} },
      execute: () => new Promise((resolve) => setTimeout(() => resolve('slept'), 30)),
    });
    const { agent } = makeAgent(toolLoop(10, 'slow'), {
      tools: [slow],
      limits: { timeoutMs: 40, maxIterations: 20 },
    });

    const result = await agent.run('loop');
    expect(result.stopReason).toBe('timeout');
  });

  it('includes the budget in the limit error message', async () => {
    const { agent } = makeAgent(toolLoop(10), {
      tools: [echoTool],
      limits: { maxIterations: 1 },
    });

    const result = await agent.run('loop');
    expect(result.error?.message).toContain('1');
  });

  it('keeps the last text when a limit stops the loop', async () => {
    const { agent } = makeAgent(
      [
        {
          text: 'partial answer',
          toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }],
        },
        { toolCalls: [{ id: 'c2', name: 'echo', arguments: { text: 'x' } }] },
      ],
      { tools: [echoTool], limits: { maxIterations: 1 } },
    );

    const result = await agent.run('loop');
    expect(result.output).toBe('partial answer');
  });
});

describe('Agent cancellation', () => {
  it('throws AbortError when the signal is already aborted', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    const controller = new AbortController();
    controller.abort();

    await expect(agent.run('hi', { signal: controller.signal })).rejects.toThrow(
      AbortError,
    );
  });

  it('forwards the composed signal to the provider', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }]);
    await agent.run('hi');
    expect(provider.signals[0]).toBeInstanceOf(AbortSignal);
    expect(provider.signals[0]?.aborted).toBe(false);
  });

  it('aborts mid-run when the signal fires later', async () => {
    const slow = defineTool({
      name: 'slow',
      description: 'Sleeps.',
      jsonSchema: { type: 'object', properties: {} },
      execute: () => new Promise((resolve) => setTimeout(() => resolve('slept'), 15)),
    });
    const { agent } = makeAgent(toolLoop(10, 'slow'), {
      tools: [slow],
      limits: { maxIterations: 10 },
    });

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 5);
    await expect(agent.run('loop', { signal: controller.signal })).rejects.toThrow(
      AbortError,
    );
  });

  it('does not abort an unrelated run', async () => {
    const { agent } = makeAgent([{ text: 'fine' }]);
    const controller = new AbortController();
    controller.abort();

    await expect(agent.run('one')).resolves.toMatchObject({ output: 'fine' });
  });
});

describe('Agent errors', () => {
  it('propagates provider failures', async () => {
    const { agent } = makeAgent([{ error: new ProviderError('upstream is down') }], {
      retry: false,
    });
    await expect(agent.run('hi')).rejects.toThrow(/upstream is down/);
  });

  it('wraps an unknown provider failure in an AgentError', async () => {
    const { agent } = makeAgent([{ error: new Error('socket hang up') }], {
      retry: false,
    });
    await expect(agent.run('hi')).rejects.toThrow(/socket hang up/);
  });

  it('returns the error instead of throwing when throwOnError is false', async () => {
    const { agent } = makeAgent([{ error: new ProviderError('upstream is down') }], {
      retry: false,
    });

    const result = await agent.run('hi', { throwOnError: false });
    expect(result.stopReason).toBe('error');
    expect(result.error).toBeInstanceOf(ProviderError);
    expect(result.error?.message).toContain('upstream is down');
  });

  it('emits an error event', async () => {
    const { agent } = makeAgent([{ error: new ProviderError('nope') }], { retry: false });

    const seen: unknown[] = [];
    agent.on('error', (event) => seen.push(event));
    await agent.run('hi', { throwOnError: false });

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ fatal: true, scope: 'model' });
  });

  it('retries transient model failures', async () => {
    const { agent, provider } = makeAgent(
      [{ error: new ProviderError('blip', { statusCode: 503 }) }, { text: 'recovered' }],
      { retry: { maxAttempts: 2, initialDelayMs: 1 } },
    );

    const retries: unknown[] = [];
    agent.on('retry', (event) => retries.push(event));

    const result = await agent.run('hi');
    expect(result.output).toBe('recovered');
    expect(provider.callCount).toBe(2);
    expect(retries).toHaveLength(1);
  });

  it('does not retry when retry is false', async () => {
    const { agent, provider } = makeAgent(
      [{ error: new ProviderError('blip') }, { text: 'never' }],
      {
        retry: false,
      },
    );

    await expect(agent.run('hi')).rejects.toThrow();
    expect(provider.callCount).toBe(1);
  });

  it('gives up after the retry budget', async () => {
    const { agent, provider } = makeAgent(
      [
        { error: new ProviderError('blip', { statusCode: 503 }) },
        { error: new ProviderError('blip', { statusCode: 503 }) },
        { text: 'never' },
      ],
      { retry: { maxAttempts: 2, initialDelayMs: 1 } },
    );

    await expect(agent.run('hi')).rejects.toThrow();
    expect(provider.callCount).toBe(2);
  });

  it('wraps a non-Error throw into an AgentError', async () => {
    // A provider that rejects with a bare value, which the agent must still
    // turn into something with a code and a message.
    const { agent } = makeAgent([{ text: 'ok' }], {
      retry: false,
      hooks: {
        onAgentStart: () => {
          // eslint-disable-next-line @typescript-eslint/only-throw-error
          throw 'a string failure';
        },
      },
    });

    await expect(agent.run('hi')).rejects.toThrow('a string failure');
  });

  it('reports non-retryable errors immediately', async () => {
    const { agent, provider } = makeAgent(
      [
        { error: new ProviderError('bad request', { retryable: false }) },
        { text: 'never' },
      ],
      { retry: { maxAttempts: 3, initialDelayMs: 1 } },
    );

    await agent.run('hi').catch(() => undefined);
    expect(provider.callCount).toBe(1);
  });
});

describe('Agent events', () => {
  it('emits the lifecycle events in order', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'all done' },
      ],
      { tools: [echoTool] },
    );

    const seen: string[] = [];
    agent.onMany({
      'agent:start': () => seen.push('agent:start'),
      'iteration:start': () => seen.push('iteration:start'),
      'model:start': () => seen.push('model:start'),
      'model:end': () => seen.push('model:end'),
      'tool:start': () => seen.push('tool:start'),
      'tool:end': () => seen.push('tool:end'),
      'iteration:end': () => seen.push('iteration:end'),
      'agent:end': () => seen.push('agent:end'),
    });

    await agent.run('go');

    expect(seen).toEqual([
      'agent:start',
      'iteration:start',
      'model:start',
      'model:end',
      'tool:start',
      'tool:end',
      'iteration:end',
      'iteration:start',
      'model:start',
      'model:end',
      'iteration:end',
      'agent:end',
    ]);
  });

  it('carries start, end, and prompt payloads', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);

    const starts: unknown[] = [];
    const ends: unknown[] = [];
    const prompts: unknown[] = [];
    agent.onMany({
      'agent:start': (event) => starts.push(event),
      'agent:end': (event) => ends.push(event),
      'agent:prompt': (event) => prompts.push(event),
    });

    await agent.run('hi', { runId: 'run_1' });

    expect(starts[0]).toMatchObject({
      runId: 'run_1',
      input: 'hi',
      agentName: 'agent',
      provider: 'mock',
      model: 'mock-1',
      tools: [],
    });
    expect(ends[0]).toMatchObject({
      stopReason: 'completed',
      output: 'ok',
      iterations: 1,
    });
    expect(prompts[0]).toMatchObject({
      iteration: 1,
      estimatedTokens: expect.any(Number),
      messages: expect.any(Array),
    });
  });

  it('supports unsubscribe and once', async () => {
    const { agent } = makeAgent([
      { text: 'a' },
      { text: 'b' },
      { text: 'c' },
      { text: 'd' },
    ]);

    let starts = 0;
    const off = agent.on('agent:start', () => {
      starts += 1;
    });
    await agent.run('one');
    off();
    await agent.run('two');
    expect(starts).toBe(1);

    let once = 0;
    agent.once('agent:start', () => {
      once += 1;
    });
    await agent.run('three');
    await agent.run('four');
    expect(once).toBe(1);
  });

  it('removes listeners for one event', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    let count = 0;
    agent.on('agent:end', () => {
      count += 1;
    });
    agent.removeAllListeners('agent:end');

    await agent.run('hi');
    expect(count).toBe(0);
  });

  it('removes every listener', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    let count = 0;
    agent.on('agent:start', () => {
      count += 1;
    });
    agent.on('agent:end', () => {
      count += 1;
    });
    agent.removeAllListeners();

    await agent.run('hi');
    expect(count).toBe(0);
  });

  it('keeps running when a listener throws', async () => {
    const { agent } = makeAgent([{ text: 'ok' }]);
    agent.on('agent:end', () => {
      throw new Error('listener exploded');
    });

    await expect(agent.run('hi')).resolves.toMatchObject({ output: 'ok' });
  });
});

describe('Agent hooks', () => {
  it('lets onBeforeModel rewrite the request', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      hooks: {
        onBeforeModel: ({ request }) => ({
          ...request,
          messages: [{ role: 'user', content: 'rewritten' }],
        }),
      },
    });

    await agent.run('hi');
    expect(provider.lastRequest?.messages).toEqual([
      { role: 'user', content: 'rewritten' },
    ]);
  });

  it('lets onSystemPrompt replace the prompt', async () => {
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      hooks: { onSystemPrompt: () => 'from the hook' },
    });

    await agent.run('hi');
    expect(provider.lastRequest?.messages[0]?.content).toBe('from the hook');
  });

  it('receives the default prompt in onSystemPrompt', async () => {
    const seen: string[] = [];
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      instructions: 'default',
      hooks: {
        onSystemPrompt: ({ defaultPrompt }) => {
          seen.push(defaultPrompt);
          return undefined;
        },
      },
    });

    await agent.run('hi');
    expect(seen).toEqual(['default']);
    expect(provider.lastRequest?.messages[0]?.content).toBe('default');
  });

  it('lets onBeforeReturn rewrite the answer', async () => {
    const { agent } = makeAgent([{ text: 'raw' }], {
      hooks: { onBeforeReturn: ({ output }) => `${output} (edited)` },
    });

    await expect(agent.run('hi')).resolves.toMatchObject({ output: 'raw (edited)' });
  });

  it('lets onBeforeReturn veto by throwing', async () => {
    const { agent } = makeAgent([{ text: 'raw' }], {
      hooks: {
        onBeforeReturn: () => {
          throw new Error('not allowed to answer');
        },
      },
    });

    await expect(agent.run('hi')).rejects.toThrow('not allowed to answer');
  });

  it('aborts the run from onAgentStart', async () => {
    const { agent, provider } = makeAgent([{ text: 'never' }], {
      hooks: {
        onAgentStart: () => {
          throw new Error('blocked by policy');
        },
      },
    });

    await expect(agent.run('hi')).rejects.toThrow('blocked by policy');
    expect(provider.callCount).toBe(0);
  });

  it('calls the observational hooks', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'done' },
      ],
      {
        tools: [echoTool],
        hooks: {
          onPrompt: () => void calls.push('prompt'),
          onIterationStart: () => void calls.push('iteration'),
          onModelStart: () => void calls.push('model'),
          onModelEnd: () => void calls.push('modelEnd'),
          onToolStart: () => void calls.push('tool'),
          onToolEnd: () => void calls.push('toolEnd'),
          onIterationEnd: () => void calls.push('iterationEnd'),
          onAgentEnd: () => void calls.push('end'),
        },
      },
    );
    const calls: string[] = [];

    await agent.run('go');
    expect(new Set(calls)).toEqual(
      new Set([
        'prompt',
        'iteration',
        'model',
        'modelEnd',
        'tool',
        'toolEnd',
        'iterationEnd',
        'end',
      ]),
    );
  });

  it('isolates throwing observational hooks', async () => {
    const { agent } = makeAgent([{ text: 'ok' }], {
      hooks: {
        onModelEnd: () => {
          throw new Error('metrics service down');
        },
      },
    });

    await expect(agent.run('hi')).resolves.toMatchObject({ output: 'ok' });
  });

  it('receives state in hooks', async () => {
    const provider = mockProvider({ script: [{ text: 'ok' }] });
    const agent = new Agent<{ user: string }>({
      provider,
      model: 'mock-1',
      planner: false,
      state: { user: 'ada' },
      hooks: {
        onBeforeReturn: ({ state }) => state.user,
      },
    });

    await expect(agent.run('hi')).resolves.toMatchObject({ output: 'ada' });
  });
});

describe('Agent structured output', () => {
  const User = z.object({ name: z.string(), age: z.number().int() });

  it('validates JSON output into data', async () => {
    const { agent } = makeAgent([{ text: '{"name":"ada","age":36}' }]);

    const result = await agent.run<{ name: string; age: number }>('who?', {
      outputSchema: User,
    });

    expect(result.data).toEqual({ name: 'ada', age: 36 });
    expect(result.stopReason).toBe('completed');
    expect(result.output).toContain('ada');
  });

  it('accepts JSON wrapped in prose', async () => {
    const { agent } = makeAgent([{ text: 'Here you go: {"name":"ada","age":36} done' }]);
    const result = await agent.run('who?', { outputSchema: User });
    expect(result.data).toEqual({ name: 'ada', age: 36 });
  });

  it('requests native json_schema when no tools are configured', async () => {
    const { agent, provider } = makeAgent([{ text: '{"name":"ada","age":36}' }]);
    await agent.run('who?', { outputSchema: User });

    expect(provider.lastRequest?.responseFormat).toMatchObject({
      type: 'json_schema',
      strict: true,
    });
  });

  it('falls back to json_object when tools are present', async () => {
    const { agent, provider } = makeAgent([{ text: '{"name":"ada","age":36}' }], {
      tools: [echoTool],
    });

    await agent.run('who?', { outputSchema: User });
    expect(provider.lastRequest?.responseFormat).toEqual({ type: 'json_object' });
    expect(provider.lastRequest?.tools).toHaveLength(1);
  });

  it('drops to json_object when the provider lacks strict schemas', async () => {
    const provider = mockProvider({
      script: [{ text: '{"name":"ada","age":36}' }],
      capabilities: { strictJsonSchema: false },
    });
    const agent = new Agent({ provider, model: 'mock-1', planner: false });

    await agent.run('who?', { outputSchema: User });
    expect(provider.lastRequest?.responseFormat).toEqual({ type: 'json_object' });
  });

  it('omits the response format when the provider has no json mode', async () => {
    const provider = mockProvider({
      script: [{ text: '{"name":"ada","age":36}' }],
      capabilities: { jsonMode: false, strictJsonSchema: false },
    });
    const agent = new Agent({ provider, model: 'mock-1', planner: false });

    await agent.run('who?', { outputSchema: User });
    expect(provider.lastRequest?.responseFormat).toBeUndefined();
  });

  it('repairs invalid output by asking again', async () => {
    const { agent, provider } = makeAgent([
      { text: '{"name":"ada"}' },
      { text: '{"name":"ada","age":36}' },
    ]);

    const invalid: unknown[] = [];
    agent.on('output:invalid', (event) => invalid.push(event));

    const result = await agent.run('who?', {
      outputSchema: User,
      limits: { maxOutputAttempts: 3 },
    });

    expect(result.data).toEqual({ name: 'ada', age: 36 });
    expect(invalid).toHaveLength(1);
    expect(provider.callCount).toBe(2);
  });

  it('sends the validation problems back to the model', async () => {
    const { agent, provider } = makeAgent([
      { text: 'nope' },
      { text: '{"name":"ada","age":36}' },
    ]);

    await agent.run('who?', { outputSchema: User, limits: { maxOutputAttempts: 3 } });

    expect(JSON.stringify(provider.requests[1]?.messages)).toContain('not valid');
  });

  it('throws a ValidationError after exhausting attempts', async () => {
    const { agent } = makeAgent([{ text: 'nope' }, { text: 'still nope' }]);

    await expect(
      agent.run('who?', { outputSchema: User, limits: { maxOutputAttempts: 2 } }),
    ).rejects.toThrow(ValidationError);
  });

  it('reports the failure instead of throwing when asked', async () => {
    const { agent } = makeAgent([{ text: 'nope' }, { text: 'still nope' }]);

    const result = await agent.run('who?', {
      outputSchema: User,
      limits: { maxOutputAttempts: 2 },
      throwOnError: false,
    });

    expect(result.stopReason).toBe('max_output_attempts');
    expect(result.error).toBeInstanceOf(ValidationError);
    expect(result.data).toBeUndefined();
  });

  it('lets tools run first, then validates the final answer', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'ada' } }] },
        { text: '{"name":"ada","age":36}' },
      ],
      { tools: [echoTool] },
    );

    const result = await agent.run('who?', { outputSchema: User });
    expect(result.iterations).toBe(2);
    expect(result.steps[0]?.toolResults).toHaveLength(1);
    expect(result.data).toEqual({ name: 'ada', age: 36 });
  });

  it('stops on the iteration limit when the model only calls tools', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { toolCalls: [{ id: 'c2', name: 'echo', arguments: { text: 'y' } }] },
      ],
      { tools: [echoTool], limits: { maxIterations: 2 } },
    );

    const result = await agent.run('who?', { outputSchema: User });
    expect(result.stopReason).toBe('max_iterations');
    expect(result.data).toBeUndefined();
  });

  it('fails when the model returns nothing at all', async () => {
    const { agent } = makeAgent([{ text: '' }]);
    await expect(agent.run('who?', { outputSchema: User })).rejects.toThrow(
      /empty response/,
    );
  });

  it('supports a default output schema on the agent', async () => {
    const { agent, provider } = makeAgent([{ text: '{"name":"ada","age":36}' }], {
      output: { schema: User, instructions: 'Always answer as JSON.', name: 'user' },
    });

    const result = await agent.run('who?');
    expect(result.data).toEqual({ name: 'ada', age: 36 });
    expect(provider.lastRequest?.responseFormat).toMatchObject({ name: 'user' });
    expect(JSON.stringify(provider.lastRequest?.messages[0])).toContain(
      'Always answer as JSON.',
    );
  });
});

describe('Agent planning', () => {
  const planTurn = {
    text: JSON.stringify({
      goal: 'ship it',
      steps: [{ title: 'first' }, { title: 'second' }],
    }),
  };

  type PlanningConfig = Omit<Config, 'planner'> & { planner: PlannerConfig };

  const makePlanningAgent = (
    script: readonly ScriptedTurn[],
    config: Partial<PlanningConfig> = {},
  ): { agent: Agent; provider: MockProvider } => {
    const provider = mockProvider({ script });
    const agent = new Agent({
      provider,
      model: 'mock-1',
      planner: { planOnRun: true, exposeUpdateTool: true },
      ...config,
    });
    return { agent, provider };
  };

  it('plans before the loop and renders the plan into the prompt', async () => {
    const { agent, provider } = makePlanningAgent([planTurn, { text: 'done' }]);
    const result = await agent.run('ship the feature');

    expect(agent.currentPlan?.steps).toHaveLength(2);
    const system = JSON.stringify(provider.requests[1]?.messages[0]);
    expect(system).toContain('ship it');
    expect(system).toContain('first');
    expect(result.output).toBe('done');
  });

  it('registers the update_plan tool', async () => {
    const { agent, provider } = makePlanningAgent([planTurn, { text: 'done' }]);
    await agent.run('ship it');
    expect(provider.requests[1]?.tools?.map((spec) => spec.name)).toContain(
      'update_plan',
    );
  });

  it('excludes the planner tool when disabled', async () => {
    const { agent, provider } = makePlanningAgent([planTurn, { text: 'done' }], {
      planner: { planOnRun: true, exposeUpdateTool: false },
    });
    await agent.run('ship it');
    expect(provider.requests[1]?.tools ?? []).toHaveLength(0);
  });

  it('updates the plan through the tool and emits a plan event', async () => {
    const { agent } = makePlanningAgent([
      planTurn,
      {
        toolCalls: [
          {
            id: 'c1',
            name: 'update_plan',
            arguments: { steps: [{ id: 'step_1', status: 'completed' }] },
          },
        ],
      },
      { text: 'finished' },
    ]);

    const plans: unknown[] = [];
    agent.on('plan', (event) => plans.push(event));

    const result = await agent.run('ship it');

    expect(plans).toHaveLength(1);
    expect(agent.currentPlan?.steps[0]?.status).toBe('completed');
    expect(agent.currentPlan?.steps[1]?.status).toBe('completed');
    expect(result.output).toBe('finished');
  });

  it('skips planning when planOnRun is false', async () => {
    const { agent, provider } = makePlanningAgent([{ text: 'done' }], {
      planner: { planOnRun: false },
    });
    await agent.run('ship it');

    expect(agent.currentPlan).toBeUndefined();
    expect(provider.callCount).toBe(1);
  });

  it('supports a custom plan builder', async () => {
    const createPlan = vi.fn<NonNullable<PlannerConfig['createPlan']>>(async () => ({
      goal: 'custom goal',
      steps: [{ id: 'step_1', title: 'only step', status: 'pending' as const }],
      createdAt: Date.now(),
    }));
    const { agent, provider } = makePlanningAgent([{ text: 'done' }], {
      planner: { createPlan, render: () => 'CUSTOM PLAN' },
    });

    const result = await agent.run('go');
    expect(createPlan).toHaveBeenCalledTimes(1);
    expect(createPlan.mock.calls[0]?.[0]).toBe('go');
    expect(JSON.stringify(provider.requests[0]?.messages[0])).toContain('CUSTOM PLAN');
    expect(result.output).toBe('done');
  });

  it('keeps the run alive when planning fails', async () => {
    const { agent } = makePlanningAgent([{ text: 'done' }], {
      planner: {
        createPlan: async () => {
          throw new Error('planner exploded');
        },
      },
    });

    const result = await agent.run('go');
    expect(result.output).toBe('done');
    expect(agent.currentPlan).toBeUndefined();
  });

  it('plans without running the loop', async () => {
    const { agent, provider } = makePlanningAgent([planTurn]);
    const plan = await agent.plan('think about it');

    expect(plan.goal).toBe('ship it');
    expect(provider.callCount).toBe(1);
  });

  it('completes every step when the run completes', async () => {
    const { agent } = makePlanningAgent([planTurn, { text: 'done' }]);
    await agent.run('ship it');
    expect(agent.currentPlan?.steps.every((step) => step.status === 'completed')).toBe(
      true,
    );
  });

  it('leaves the plan alone when the run hits a limit', async () => {
    const { agent } = makePlanningAgent([planTurn, ...toolLoop(5)], {
      limits: { maxIterations: 1 },
    });
    const result = await agent.run('ship it');

    expect(result.stopReason).toBe('max_iterations');
    expect(agent.currentPlan?.steps.every((step) => step.status === 'pending')).toBe(
      true,
    );
  });

  it('reuses the plan across runs when persist is set', async () => {
    const { agent } = makePlanningAgent(
      [planTurn, { text: 'first' }, { text: 'second' }],
      {
        planner: { persist: true },
      },
    );

    await agent.run('ship it');
    await agent.run('ship it again');

    expect(agent.currentPlan?.goal).toBe('ship it');
  });
});

describe('Agent long-term memory', () => {
  it('recalls relevant records into the system prompt', async () => {
    const longTerm = new InMemoryLongTermMemory();
    longTerm.add([{ role: 'user', content: 'the user is a marine biologist' }]);
    const { agent, provider } = makeAgent([{ text: 'noted' }], {
      longTermMemory: longTerm,
    });

    await agent.run('who is the user?');
    expect(JSON.stringify(provider.lastRequest?.messages[0])).toContain(
      'marine biologist',
    );
  });

  it('stores the input and answer after a run', async () => {
    const longTerm = new InMemoryLongTermMemory();
    const { agent } = makeAgent([{ text: 'the answer is 42' }], {
      longTermMemory: longTerm,
    });

    await agent.run('what is the answer?');
    expect(longTerm.size).toBe(2);
  });

  it('can be disabled', async () => {
    const longTerm = new InMemoryLongTermMemory();
    const { agent, provider } = makeAgent([{ text: 'ok' }], {
      longTermMemory: longTerm,
      longTerm: { recall: false, store: false },
    });

    await agent.run('hello');
    expect(longTerm.size).toBe(0);
    expect(JSON.stringify(provider.lastRequest?.messages[0])).not.toContain(
      'Relevant memories',
    );
  });

  it('emits memory events', async () => {
    const longTerm = new InMemoryLongTermMemory();
    const { agent } = makeAgent([{ text: 'ok' }], { longTermMemory: longTerm });

    const operations: string[] = [];
    agent.on('memory', (event) => operations.push(event.operation));
    await agent.run('hello');

    expect(operations).toContain('recall');
    expect(operations).toContain('store');
  });

  it('survives a long-term store failure', async () => {
    const longTerm = new InMemoryLongTermMemory();
    longTerm.add = () => {
      throw new Error('disk full');
    };
    const { agent } = makeAgent([{ text: 'ok' }], { longTermMemory: longTerm });

    await expect(agent.run('hello')).resolves.toMatchObject({ output: 'ok' });
  });
});

describe('Agent streaming', () => {
  it('buffers a non-streamed run', async () => {
    const { agent, provider } = makeAgent([{ text: 'streamed answer' }]);
    const deltas: string[] = [];
    agent.on('model:delta', (event) => deltas.push(event.text));

    const result = await agent.run('go');
    expect(result.output).toBe('streamed answer');
    expect(deltas).toEqual([]);
    expect(provider.callCount).toBe(1);
  });

  it('emits text deltas on a streamed run', async () => {
    const { agent } = makeAgent([
      { text: 'streamed answer', chunks: ['streamed ', 'answer'] },
    ]);

    const deltas: string[] = [];
    agent.on('model:delta', (event) => deltas.push(event.text));

    await agent.stream('go').result;
    expect(deltas).toEqual(['streamed ', 'answer']);
  });

  it('buffers when the provider cannot stream', async () => {
    const provider = new NonStreamingProvider({ script: [{ text: 'buffered' }] });
    const agent = new Agent({ provider, model: 'mock-1', planner: false });
    await expect(agent.run('go')).resolves.toMatchObject({ output: 'buffered' });
  });

  it('streams typed events to the consumer', async () => {
    const { agent } = makeAgent([{ text: 'hi', chunks: ['h', 'i'] }]);
    const run = agent.stream('go');

    const events: string[] = [];
    for await (const event of run) events.push(event.type);

    await expect(run.result).resolves.toMatchObject({ output: 'hi' });
    expect(events).toContain('text');
    expect(events.at(-1)).toBe('done');
  });

  it('emits tool events while streaming', async () => {
    const { agent } = makeAgent(
      [
        { toolCalls: [{ id: 'c1', name: 'echo', arguments: { text: 'x' } }] },
        { text: 'done' },
      ],
      { tools: [echoTool] },
    );

    const run = agent.stream('go');
    const types = new Set<string>();
    for await (const event of run) types.add(event.type);

    expect(types).toContain('tool-call');
    expect(types).toContain('tool-result');
  });

  it('collects text with .text()', async () => {
    const { agent } = makeAgent([{ text: 'collected', chunks: ['col', 'lected'] }]);
    await expect(agent.stream('go').text()).resolves.toBe('collected');
  });

  it('supports per-type listeners on the stream handle', async () => {
    const { agent } = makeAgent([{ text: 'hi', chunks: ['hi'] }]);
    const run = agent.stream('go');

    const seen: string[] = [];
    run.on('text', (event) => seen.push(event.text));
    await expect(run.text()).resolves.toBe('hi');
    expect(seen).toEqual(['hi']);
  });

  it('supports abort() on the handle', async () => {
    const { agent } = makeAgent(toolLoop(10), {
      tools: [echoTool],
      limits: { maxIterations: 10 },
    });

    const run = agent.stream('loop');
    const iterator = run[Symbol.asyncIterator]();
    await iterator.next();
    run.abort();

    expect(run.aborted).toBe(true);
    await expect(run.result).rejects.toThrow(AbortError);
  });

  it('exposes a run id on the handle', async () => {
    const { agent } = makeAgent([{ text: 'hi' }]);
    const run = agent.stream('go', { runId: 'run_stream' });
    await run.result;
    expect(run.id).toBe('run_stream');
  });
});

describe('Agent.fork', () => {
  it('creates an independent copy with fresh memory', async () => {
    const { agent } = makeAgent([{ text: 'one' }, { text: 'two' }], {
      instructions: 'base',
    });
    await agent.run('first');

    const fork = agent.fork({ instructions: 'forked' });
    expect(fork.name).toBe('agent-fork');
    expect(fork.memory.length).toBe(0);
    expect(fork.toolSpecs()).toEqual(agent.toolSpecs());
    expect(fork.state).toBe(agent.state);
  });

  it('shares the provider but not the memory', async () => {
    const { agent } = makeAgent([{ text: 'one' }, { text: 'two' }]);
    const fork = agent.fork();

    await expect(agent.run('a')).resolves.toMatchObject({ output: 'one' });
    expect(fork.memory.length).toBe(0);
    await expect(fork.run('b')).resolves.toMatchObject({ output: 'two' });
    expect(agent.memory.length).toBe(2);
    expect(fork.memory.length).toBe(2);
  });
});
