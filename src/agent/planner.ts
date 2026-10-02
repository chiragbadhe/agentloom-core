import { PlanningError } from '../errors.js';
import type { CompletionRequest, ModelMessage } from '../providers/types.js';
import { schemaFromJsonSchema, type Schema } from '../schema.js';
import { extractJson } from '../utils/text.js';
import type { PlanContext } from './planner-types.js';

export type PlanStepStatus = 'pending' | 'in_progress' | 'completed' | 'skipped';

export interface PlanStep {
  readonly id: string;
  readonly title: string;
  readonly description?: string;
  readonly status: PlanStepStatus;
  readonly notes?: string;
}

export interface Plan {
  readonly goal: string;
  readonly steps: readonly PlanStep[];
  readonly createdAt: number;
}

/** JSON Schema for plans, used for native structured output and validation. */
export const PLAN_JSON_SCHEMA: Record<string, unknown> = {
  type: 'object',
  properties: {
    goal: { type: 'string', description: 'One-sentence statement of the objective.' },
    steps: {
      type: 'array',
      minItems: 1,
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Short imperative step.' },
          description: { type: 'string', description: 'What to do, concretely.' },
        },
        required: ['title'],
      },
    },
  },
  required: ['goal', 'steps'],
};

const PLAN_VALIDATOR: Schema<{
  goal: string;
  steps: { title: string; description?: string }[];
}> = schemaFromJsonSchema(PLAN_JSON_SCHEMA);

/** Prompt used when the provider cannot enforce the plan schema natively. */
export function buildPlanPrompt(input: string, maxSteps: number): string {
  return [
    'Break the following objective into a short, concrete plan.',
    `Use at most ${maxSteps} steps. Each step must be independently verifiable.`,
    'Respond with a single JSON object and nothing else, shaped like:',
    JSON.stringify({
      goal: '<one sentence>',
      steps: [{ title: '<step>', description: '<detail>' }],
    }),
    '',
    `Objective: ${input}`,
  ].join('\n');
}

/**
 * Create a plan with a model call.
 *
 * Prefers native structured output (JSON schema) when the provider supports
 * it, and falls back to prompt-plus-parse elsewhere. In both cases the result
 * is validated before it is used, so a malformed plan surfaces as a
 * {@link PlanningError} instead of confusing the run.
 */
export async function createPlanWithModel(
  request: (
    messages: readonly ModelMessage[],
    responseFormat?: CompletionRequest['responseFormat'],
  ) => Promise<string>,
  context: PlanContext & { maxSteps: number; supportsStrictSchema: boolean },
): Promise<Plan> {
  const prompt = buildPlanPrompt(context.input, context.maxSteps);

  const messages: ModelMessage[] = [
    {
      role: 'system',
      content:
        'You are a planning assistant. You output only valid JSON. ' +
        'No prose, no markdown fences.',
    },
    { role: 'user', content: prompt },
  ];

  const responseFormat: CompletionRequest['responseFormat'] = context.supportsStrictSchema
    ? { type: 'json_schema', name: 'plan', schema: PLAN_JSON_SCHEMA, strict: true }
    : { type: 'json_object' };

  const raw = await request(messages, responseFormat);

  return normalizePlan(raw, context.input);
}

/**
 * Validate and normalize raw plan JSON. Exported so custom planners can reuse
 * the same validation guarantees.
 */
export function normalizePlan(raw: string, goal: string): Plan {
  const parsed = extractJson(raw);
  if (parsed === undefined) {
    throw new PlanningError(
      `Planner did not return valid JSON: ${truncateForError(raw)}`,
    );
  }

  const validated = PLAN_VALIDATOR.safeParse(parsed);
  if (!validated.success) {
    throw new PlanningError(
      `Planner output did not match the plan schema:\n${validated.error.issues
        .map((issue) => `- ${issue.path.join('.') || '(root)'}: ${issue.message}`)
        .join('\n')}`,
    );
  }

  const steps: PlanStep[] = validated.data.steps.map((step, index) => ({
    id: `step_${index + 1}`,
    title: step.title,
    ...(step.description === undefined ? {} : { description: step.description }),
    status: 'pending',
  }));

  return {
    goal: validated.data.goal || goal,
    steps,
    createdAt: Date.now(),
  };
}

/** Render a plan as a markdown checklist for the system prompt. */
export function renderPlan(plan: Plan): string {
  const lines = [`Goal: ${plan.goal}`, '', 'Plan:'];
  for (const step of plan.steps) {
    const marker =
      step.status === 'completed'
        ? '[x]'
        : step.status === 'in_progress'
          ? '[~]'
          : step.status === 'skipped'
            ? '[-]'
            : '[ ]';
    const detail = step.description === undefined ? '' : ` — ${step.description}`;
    lines.push(`${marker} ${step.id}: ${step.title}${detail}`);
  }
  lines.push(
    '',
    'Work through the steps in order. Mark progress with the update_plan tool.',
  );
  return lines.join('\n');
}

/**
 * Mutable plan handle held by the agent for the duration of a run.
 * Steps are updated by the `update_plan` tool and observed via `onPlanChange`.
 */
export class PlanTracker {
  private current: Plan | undefined;
  private readonly listeners = new Set<(plan: Plan) => void>();

  constructor(private readonly persist: boolean = false) {}

  get plan(): Plan | undefined {
    return this.current;
  }

  get completedCount(): number {
    return this.current?.steps.filter((step) => step.status === 'completed').length ?? 0;
  }

  set(plan: Plan | undefined): void {
    if (plan === undefined) return;
    if (this.persist && this.current !== undefined) return;
    this.current = plan;
    this.emit();
  }

  /**
   * Apply a model-reported update.
   *
   * Matches steps by id first, then by title, and ignores unknown steps — the
   * model is not trusted to corrupt plan bookkeeping.
   */
  update(update: {
    goal?: string;
    steps?: readonly {
      id?: string;
      title?: string;
      status?: PlanStepStatus;
      notes?: string;
    }[];
  }): Plan | undefined {
    const plan = this.current;
    if (plan === undefined) return undefined;

    const matches = (step: PlanStep) =>
      update.steps?.find((candidate) => {
        const id = candidate.id?.toLowerCase();
        const title = candidate.title?.toLowerCase();
        return (
          (id !== undefined && id === step.id.toLowerCase()) ||
          (title !== undefined && title === step.title.toLowerCase())
        );
      });

    const steps = plan.steps.map((step) => {
      const match = matches(step);
      if (match === undefined) return step;
      return {
        ...step,
        ...(match.status === undefined ? {} : { status: match.status }),
        ...(match.notes === undefined ? {} : { notes: match.notes }),
      };
    });

    this.current = {
      ...plan,
      ...(update.goal === undefined ? {} : { goal: update.goal }),
      steps,
    };
    this.emit();
    return this.current;
  }

  /** Mark every step complete. Used when the run finishes. */
  completeAll(): void {
    const plan = this.current;
    if (plan === undefined) return;
    this.current = {
      ...plan,
      steps: plan.steps.map((step) =>
        step.status === 'pending' || step.status === 'in_progress'
          ? { ...step, status: 'completed' }
          : step,
      ),
    };
    this.emit();
  }

  reset(): void {
    this.current = undefined;
  }

  onChange(listener: (plan: Plan) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    if (this.current === undefined) return;
    for (const listener of this.listeners) {
      try {
        listener(this.current);
      } catch {
        // A plan observer must never break the run.
      }
    }
  }
}

function truncateForError(value: string): string {
  return value.length > 300 ? `${value.slice(0, 300)}…` : value;
}
