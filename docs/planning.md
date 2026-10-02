# Planning

Planning is opt-in. Enable it and the agent breaks the objective into steps
before the loop starts, renders the plan into the system prompt, and gives the
model a tool to report progress.

```ts
const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  instructions: 'You are an operations assistant. Use the plan as a checklist.',
  planner: { maxSteps: 6 },
});

const result = await agent.run(
  'Roll back the release that caused elevated 5xx rates.',
);
```

Omit `planner` (or pass `false`) and no planning happens — no extra model call.

## How it works

1. Before the loop starts, a plan is generated with a single structured model
   call. Native JSON Schema is requested when the provider supports it;
   otherwise the model is prompted to emit JSON.
2. The plan is validated. Malformed output raises `PlanningError`.
3. The rendered plan becomes part of the system prompt, so it is visible from
   the very first iteration.
4. An `update_plan` tool is registered so the model can mark steps as it works.
5. Each update emits a `plan` event.
6. On a successful run, remaining steps are marked complete.

Planning failure never sinks a run: the error is reported as non-fatal, logged
as a warning, and the loop continues without a plan.

## Plan shape

```ts
interface Plan {
  goal: string;
  steps: readonly PlanStep[];
  createdAt: number;
}

interface PlanStep {
  id: string; // step_1, step_2, ...
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  notes?: string;
}
```

The default renderer produces a markdown checklist:

```
Goal: Roll back the release that caused elevated 5xx rates.

Plan:
[ ] step_1: Identify the offending release
[~] step_2: Verify the rollback on staging
[-] step_3: Skip the canary — staging was enough

Work through the steps in order. Mark progress with the update_plan tool.
```

Markers are `[ ]` pending, `[~]` in progress, `[x]` completed, `[-]` skipped.

## Configuration

| Field              | Default      | Notes                                         |
| ------------------ | ------------ | --------------------------------------------- |
| `enabled`          | `true`       | Master switch; equivalent to `planner: false` |
| `planOnRun`        | `true`       | Generate a plan before the loop               |
| `exposeUpdateTool` | `true`       | Register the `update_plan` tool               |
| `maxSteps`         | `8`          | Maximum steps in a generated plan             |
| `createPlan`       | model call   | Custom plan builder                           |
| `render`           | `renderPlan` | Custom prompt renderer                        |
| `persist`          | `false`      | Reuse the plan across runs                    |

`agent.planningEnabled` and `agent.currentPlan` reflect the current state.

## Previewing a plan

`agent.plan()` generates a plan without running the loop — useful for a UI
confirmation step, a cost estimate, or a dry run:

```ts
const plan = await agent.plan(
  'Roll back the release that caused elevated 5xx rates.',
);

for (const [index, step] of plan.steps.entries()) {
  console.log(`${index + 1}. ${step.title}`);
}
```

## Watching progress

```ts
agent.on('plan', ({ plan, steps, completed }) => {
  console.log(`${completed}/${steps.length}: ${plan.goal}`);
  for (const step of steps) console.log(`  ${step.status}: ${step.title}`);
});
```

`PlanTracker` is exposed as `agent.planTracker` for finer control:

```ts
agent.planTracker.plan; // current plan, or undefined
agent.planTracker.completedCount;
agent.planTracker.onChange((plan) => render(plan)); // unsubscribe by calling the returned fn
agent.planTracker.reset();
```

Updates are matched by id first, then by title. Unknown steps are ignored: the
model is not trusted to corrupt plan bookkeeping.

## Reusing a plan

By default each run gets a fresh plan. With `persist: true`, an existing plan is
reused instead of paying for another model call:

```ts
new Agent({ model: 'openai:gpt-4o-mini', planner: { persist: true } });
```

## Custom planner

Take planning out of the model entirely — deterministic decomposition, a
templated checklist, a queue of known steps:

```ts
import type { Plan } from '@agentloom/core';

const agent = new Agent({
  model: 'openai:gpt-4o-mini',
  planner: {
    createPlan: async (input, { runId, signal }): Promise<Plan> => ({
      goal: input,
      steps: [
        {
          id: 'step_1',
          title: 'Gather context',
          description: '...',
          status: 'pending',
        },
        { id: 'step_2', title: 'Draft the response', status: 'pending' },
      ],
      createdAt: Date.now(),
    }),
  },
});
```

For a model-backed planner with your own prompt, use `createPlanWithModel` and
keep the validation:

```ts
import { createPlanWithModel, normalizePlan } from '@agentloom/core';

createPlan: async (input, { runId, signal }) =>
  createPlanWithModel(
    async (messages, responseFormat) => {
      const res = await provider.complete(
        { model: 'gpt-4o-mini', messages, responseFormat },
        { signal },
      );
      return res.message.content;
    },
    { input, runId, signal, maxSteps: 6, supportsStrictSchema: true },
  ),
```

## Custom rendering

```ts
planner: {
  render: (plan) => `Objective: ${plan.goal}\n` +
    plan.steps.map((s, i) => `${i + 1}. [${s.status}] ${s.title}`).join('\n'),
}
```

## Cost

Planning is one extra model call per run when `planOnRun` is on. It pays off for
multi-step objectives where the model otherwise wanders; for a single tool call
it is overhead. Turn it off per agent, or `planOnRun: false` to build the plan
yourself and still expose `update_plan`.

## More

- [Agent](agent.md) — where planning sits in the loop, and prompt assembly order
- [Streaming & events](streaming-and-events.md) — the `plan` event
- `examples/06-planning-and-hooks.ts` — planning, hooks, and budgets together
