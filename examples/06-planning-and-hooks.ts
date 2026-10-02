/**
 * 06 — planning, hooks, budget, and observability.
 *
 * Run: `npx tsx examples/06-planning-and-hooks.ts`
 */
import { Agent, createConsoleLogger, type Plan } from '../src/index.js';

const agent = new Agent({
  name: 'ops',
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: [
    'You are an operations assistant.',
    'Use the plan as a checklist, and call update_plan as you finish steps.',
  ].join(' '),
  logger: createConsoleLogger({ level: 'warn' }),

  // Planning is opt-in: pass `false` (or omit the key) to turn it off.
  planner: {
    exposeUpdateTool: true,
    maxSteps: 6,
    // Reuse the plan across runs instead of replanning every time.
    persist: false,
  },

  // Everything the agent can do to itself, in one place.
  hooks: {
    /** Compose the final system prompt. Return `undefined` to keep the default. */
    onSystemPrompt: ({ defaultPrompt, state }) =>
      `${defaultPrompt}\nEnvironment: ${state.environment}`,

    /**
     * Rewrite the request just before it goes to the provider. Returning a
     * request replaces it; throwing aborts the run.
     */
    onBeforeModel: ({ request, state }) => {
      state.modelCalls++;
      return {
        ...request,
        metadata: { ...request.metadata, tenant: state.tenant },
      };
    },

    onRetry: ({ attempt, error, delayMs }) => {
      console.warn(`retry ${attempt}: ${error.message} (waiting ${delayMs}ms)`);
    },

    onOutputInvalid: ({ issues }) => {
      console.warn(`output invalid: ${issues.map((issue) => issue.message).join('; ')}`);
    },
  },

  state: { tenant: 'acme', environment: 'staging', modelCalls: 0 },

  limits: {
    maxIterations: 12,
    maxToolCalls: 20,
    maxTotalTokens: 60_000,
    timeoutMs: 120_000,
  },
});

// Events are useful for tracing, metrics, and dashboards.
agent.on('agent:start', ({ runId, model }) => console.log(`[${runId}] ${model}`));
agent.on('plan', ({ steps }) => console.log(`plan with ${steps.length} steps`));
agent.on('tool:end', ({ toolName, durationMs }) =>
  console.log(`  ${toolName} ${durationMs}ms`),
);
agent.on('retry', ({ delayMs }) => console.log(`  retrying in ${delayMs}ms`));

/**
 * Hooks can veto. `onAgentStart`, `onSystemPrompt`, `onBeforeModel`, and
 * `onBeforeReturn` propagate a thrown error out of the run; the rest are
 * observational and their failures are swallowed so logging never breaks a run.
 */
const guarded = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: 'You answer questions.',
  hooks: {
    onAgentStart: ({ input }) => {
      if (input.includes('secret')) throw new Error('blocked by policy hook');
    },
  },
});

await guarded.run('what is 2 + 2?').then(
  (result) => console.log(`\n${result.output}`),
  (error: unknown) => console.log(`\nvetoed: ${(error as Error).message}`),
);

/** Generate a plan without running the loop — useful for previews and approvals. */
const plan: Plan = await agent.plan(
  'Roll back the release that caused elevated 5xx rates.',
);
console.log('\npreview plan:');
plan.steps.forEach((step, index) => console.log(`  ${index + 1}. ${step.title}`));

/** Soft failures: no exception, just a result you inspect. */
const soft = await agent.run('Ship the rollback now.', { throwOnError: false });
if (soft.stopReason === 'error') console.error(soft.error?.message);
else console.log(`\n${soft.output}`);

/** Fork an agent with a different model, keeping the rest of the config. */
const fast = agent.fork({
  name: 'ops-fast',
  model: 'openai:gpt-4o-mini',
  limits: { maxIterations: 4 },
});
console.log(`forked agent: ${fast.name}`);
