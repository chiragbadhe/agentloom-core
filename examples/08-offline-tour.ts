/**
 * 08 — the whole loop, offline.
 *
 * This example wires a scripted provider into a real `Agent`, so tools, memory,
 * limits, structured output, planning, streaming, and hooks all execute without
 * touching the network. It is also a good template for testing your own agents:
 * swap the provider for a script and assert on the result.
 *
 * Run: `npx tsx examples/08-offline-tour.ts`
 */
import {
  Agent,
  createCalculatorTool,
  InMemoryConversationMemory,
  type CompletionResult,
  type ModelProvider,
  type ProviderCapabilities,
  type StreamEvent,
} from '../src/index.js';

/**
 * A provider that replays a fixed list of replies. `complete()` and `stream()`
 * both use the same script, so the agent behaves identically either way.
 */
function scriptedProvider(
  replies: readonly CompletionResult[],
  providerId = 'scripted',
): ModelProvider {
  let index = 0;
  const next = (): CompletionResult => {
    const reply = replies[Math.min(index, replies.length - 1)];
    index++;
    if (reply === undefined) throw new Error('the scripted provider ran out of replies');
    return reply;
  };

  return {
    id: providerId,
    name: `Scripted (${providerId})`,
    defaultModel: 'scripted-1',
    capabilities: {
      tools: true,
      parallelToolCalls: false,
      streaming: true,
      systemMessages: true,
      jsonMode: true,
      strictJsonSchema: true,
      vision: false,
      promptCaching: false,
    } satisfies ProviderCapabilities,

    complete: async () => next(),

    stream: (request) => ({
      async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent> {
        const reply = next();
        const responseId = 'scripted-stream';
        yield { type: 'start', model: request.model, responseId };
        for (const piece of reply.message.content.match(/.{1,12}/gs) ?? []) {
          yield { type: 'text-delta', text: piece, responseId };
        }
        for (const call of reply.message.toolCalls ?? []) {
          yield { type: 'tool-call', call, responseId };
        }
        yield { type: 'finish', result: { ...reply, responseId } };
      },
    }),
  };
}

const reply = (
  content: string,
  toolCalls?: CompletionResult['message']['toolCalls'],
): CompletionResult => ({
  message: { role: 'assistant', content, ...(toolCalls ? { toolCalls } : {}) },
  finishReason: toolCalls === undefined ? 'stop' : 'tool_calls',
  usage: { inputTokens: 20, outputTokens: 10, totalTokens: 30 },
  responseId: 'scripted-1',
  providerId: 'scripted',
  model: 'scripted-1',
  latencyMs: 1,
  raw: {},
});

// --- 1. A tool loop ---------------------------------------------------------

const toolAgent = new Agent({
  name: 'calculator-bot',
  provider: scriptedProvider([
    reply('Let me compute that.', [
      { id: 'call_1', name: 'calculator', arguments: { expression: '19 * 23' } },
    ]),
    reply('19 * 23 = 437.'),
  ]),
  instructions: 'Use the calculator for arithmetic.',
  tools: [createCalculatorTool()],
  hooks: {
    onToolEnd: ({ toolName, result }) =>
      console.log(`  tool ${toolName} -> ${result.content}`),
  },
});

const toolResult = await toolAgent.run('What is 19 times 23?');
console.log(`1. tools: ${toolResult.output} (iterations=${toolResult.iterations})`);

// --- 2. Streaming -----------------------------------------------------------

const streamAgent = new Agent({
  provider: scriptedProvider([reply('Streaming works, token by token.')]),
  instructions: 'Be brief.',
});

const run = streamAgent.stream('Say something interesting.');
let streamed = '';
for await (const event of run) {
  if (event.type === 'text-delta') streamed += event.text;
}
const streamResult = await run.result;
console.log(`2. stream: "${streamed}" (stopReason=${streamResult.stopReason})`);

// --- 3. Cancellation -------------------------------------------------------

const hang = new Agent({
  provider: {
    ...scriptedProvider([reply('never finishes')]),
    complete: () => new Promise<CompletionResult>(() => undefined),
    // Never yields and never resolves: the agent must still be cancellable.
    stream: () => ({
      [Symbol.asyncIterator]: () => ({
        next: () => new Promise<IteratorResult<StreamEvent>>(() => undefined),
      }),
    }),
  },
});

const controller = new AbortController();
const cancelled = hang.run('wait forever', {
  signal: controller.signal,
  throwOnError: false,
});
setTimeout(() => controller.abort(), 20);
const cancelledResult = await cancelled;
console.log(`3. cancel: stopReason=${cancelledResult.stopReason}`);

// --- 4. Structured output --------------------------------------------------

// Any validator with `safeParse` qualifies — no zod needed.
const Profile = {
  safeParse(input: unknown) {
    const value = input as { name?: unknown; age?: unknown } | null;
    const ok =
      typeof value?.name === 'string' && typeof value?.age === 'number' && value.age > 0;
    return ok
      ? ({
          success: true,
          data: { name: value.name as string, age: value.age as number },
        } as const)
      : ({
          success: false,
          error: {
            issues: [{ path: [], message: 'expected { name: string, age: number }' }],
          },
        } as const);
  },
};

const structured = new Agent({
  provider: scriptedProvider([reply('{"name":"Ada","age":36}')]),
  instructions: 'Extract profile data.',
});
const structuredResult = await structured.run('Ada is 36.', { outputSchema: Profile });
console.log(`4. schema: ${JSON.stringify(structuredResult.data)}`);

// --- 5. Memory -------------------------------------------------------------

const memory = new InMemoryConversationMemory({ maxMessages: 10 });
const memoryAgent = new Agent({
  provider: scriptedProvider([reply('Noted.'), reply('You told me your name is Ada.')]),
  instructions: 'Remember things.',
  memory,
});

await memoryAgent.run('My name is Ada.');
const remembered = await memoryAgent.run('What is my name?');
console.log(
  `5. memory: ${remembered.output} (${memory.messages().length} messages stored)`,
);

// --- 6. Planning -----------------------------------------------------------

const planner = new Agent({
  provider: scriptedProvider([
    // The plan is generated with the same provider, so it must come first.
    reply(
      JSON.stringify({
        goal: 'Deploy the release',
        steps: [
          { id: '1', title: 'Run the smoke tests', status: 'pending' },
          { id: '2', title: 'Promote to production', status: 'pending' },
        ],
      }),
    ),
    reply('All steps complete.'),
  ]),
  instructions: 'Follow the plan.',
  planner: { maxSteps: 4 },
});

const planned = await planner.plan('Deploy the release');
console.log(`6. plan: ${planned.goal} (${planned.steps.length} steps)`);

// --- 7. Limits -------------------------------------------------------------

const loopy = new Agent({
  provider: scriptedProvider([
    reply('again', [{ id: 'c', name: 'calculator', arguments: { expression: '1+1' } }]),
  ]),
  instructions: 'Loop.',
  tools: [createCalculatorTool()],
  limits: { maxIterations: 3 },
});
const limited = await loopy.run('go', { throwOnError: false });
console.log(
  `7. limits: stopReason=${limited.stopReason} after ${limited.iterations} iterations`,
);
