import { describe, expect, it, vi } from 'vitest';

import { PlanningError } from '../src/errors.js';
import {
  buildPlanPrompt,
  createPlanWithModel,
  normalizePlan,
  PlanTracker,
  PLAN_JSON_SCHEMA,
  renderPlan,
  type Plan,
} from '../src/agent/planner.js';
import type { CompletionRequest, ModelMessage } from '../src/providers/types.js';

const rawPlan = (goal = 'ship the feature'): string =>
  JSON.stringify({
    goal,
    steps: [{ title: 'design' }, { title: 'build', description: 'code it' }],
  });

const plan = (): Plan => normalizePlan(rawPlan(), 'fallback goal');

describe('normalizePlan', () => {
  it('assigns ids and pending status', () => {
    const result = plan();
    expect(result.goal).toBe('ship the feature');
    expect(result.steps.map((step) => step.id)).toEqual(['step_1', 'step_2']);
    expect(result.steps.every((step) => step.status === 'pending')).toBe(true);
    expect(result.steps[1]?.description).toBe('code it');
    expect(result.createdAt).toBeGreaterThan(0);
  });

  it('falls back to the caller goal when the model omits it', () => {
    const result = normalizePlan(
      JSON.stringify({ goal: '', steps: [{ title: 'a' }] }),
      'my goal',
    );
    expect(result.goal).toBe('my goal');
  });

  it('rejects non-JSON output', () => {
    expect(() => normalizePlan('sorry, I cannot help', 'g')).toThrow(PlanningError);
  });

  it('rejects output that does not match the schema', () => {
    expect(() => normalizePlan(JSON.stringify({ steps: [] }), 'g')).toThrow(
      PlanningError,
    );
    expect(() => normalizePlan(JSON.stringify({ goal: 'g' }), 'g')).toThrow(
      PlanningError,
    );
  });

  it('parses plans wrapped in markdown fences or prose', () => {
    expect(normalizePlan(`\`\`\`json\n${rawPlan()}\n\`\`\``, 'g').steps).toHaveLength(2);
    expect(
      normalizePlan(`Here you go: ${rawPlan()} hope that helps`, 'g').steps,
    ).toHaveLength(2);
  });

  it('exposes a JSON schema usable for native structured output', () => {
    expect(PLAN_JSON_SCHEMA).toMatchObject({ type: 'object' });
    const properties = PLAN_JSON_SCHEMA['properties'] as Record<string, unknown>;
    expect(Object.keys(properties)).toEqual(['goal', 'steps']);
  });
});

describe('buildPlanPrompt', () => {
  it('includes the objective and the step budget', () => {
    const prompt = buildPlanPrompt('write a haiku about sqlite', 4);
    expect(prompt).toContain('write a haiku about sqlite');
    expect(prompt).toContain('at most 4 steps');
  });
});

describe('renderPlan', () => {
  it('renders a markdown checklist', () => {
    const rendered = renderPlan(plan());
    expect(rendered).toContain('Goal: ship the feature');
    expect(rendered).toContain('[ ] step_1: design');
    expect(rendered).toContain('[ ] step_2: build — code it');
  });

  it('marks each status distinctly', () => {
    const tracked = plan();
    const tracker = new PlanTracker();
    tracker.set(tracked);
    tracker.update({ steps: [{ id: 'step_1', status: 'completed' }] });
    tracker.update({ steps: [{ id: 'step_2', status: 'in_progress' }] });

    const rendered = renderPlan(tracker.plan!);
    expect(rendered).toContain('[x] step_1');
    expect(rendered).toContain('[~] step_2');
  });

  it('renders skipped steps', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.update({ steps: [{ id: 'step_1', status: 'skipped' }] });
    expect(renderPlan(tracker.plan!)).toContain('[-] step_1');
  });
});

describe('createPlanWithModel', () => {
  const context = {
    input: 'do the thing',
    runId: 'run_1',
    signal: new AbortController().signal,
    maxSteps: 5,
    supportsStrictSchema: false,
  };

  it('requests a plan and validates the response', async () => {
    const request = vi.fn(
      async (
        _messages: readonly ModelMessage[],
        _format?: CompletionRequest['responseFormat'],
      ) => rawPlan(),
    );
    const result = await createPlanWithModel(request, context);

    expect(result.steps).toHaveLength(2);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it('asks for plain JSON when strict schema is unsupported', async () => {
    const request = vi.fn(
      async (
        _messages: readonly ModelMessage[],
        _format?: CompletionRequest['responseFormat'],
      ) => rawPlan(),
    );
    await createPlanWithModel(request, context);

    const messages = request.mock.calls[0]?.[0] ?? [];
    const responseFormat = request.mock.calls[0]?.[1];
    expect(messages[0]).toMatchObject({ role: 'system' });
    expect(messages[1]).toMatchObject({ role: 'user' });
    expect(responseFormat).toEqual({ type: 'json_object' });
  });

  it('asks for native schema when the provider supports it', async () => {
    const request = vi.fn(
      async (
        _messages: readonly ModelMessage[],
        _format?: CompletionRequest['responseFormat'],
      ) => rawPlan(),
    );
    await createPlanWithModel(request, { ...context, supportsStrictSchema: true });

    expect(request.mock.calls[0]?.[1]).toMatchObject({
      type: 'json_schema',
      strict: true,
    });
  });

  it('propagates planner output errors', async () => {
    const request = vi.fn(
      async (
        _messages: readonly ModelMessage[],
        _format?: CompletionRequest['responseFormat'],
      ) => 'not json at all',
    );
    await expect(createPlanWithModel(request, context)).rejects.toThrow(PlanningError);
  });
});

describe('PlanTracker', () => {
  it('starts empty', () => {
    const tracker = new PlanTracker();
    expect(tracker.plan).toBeUndefined();
    expect(tracker.completedCount).toBe(0);
  });

  it('ignores an undefined plan', () => {
    const tracker = new PlanTracker();
    tracker.set(undefined);
    expect(tracker.plan).toBeUndefined();
  });

  it('stores a plan and reports completion', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    expect(tracker.plan?.steps).toHaveLength(2);
    expect(tracker.completedCount).toBe(0);
  });

  it('does not overwrite an existing plan when persist is set', () => {
    const tracker = new PlanTracker(true);
    const first = plan();
    tracker.set(first);
    tracker.set(
      normalizePlan(JSON.stringify({ goal: 'other', steps: [{ title: 'x' }] }), 'g'),
    );

    expect(tracker.plan?.goal).toBe(first.goal);
  });

  it('overwrites when persist is off', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.set(
      normalizePlan(JSON.stringify({ goal: 'other', steps: [{ title: 'x' }] }), 'g'),
    );
    expect(tracker.plan?.goal).toBe('other');
  });

  it('updates status by id', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    const updated = tracker.update({ steps: [{ id: 'step_1', status: 'completed' }] });

    expect(updated?.steps[0]?.status).toBe('completed');
    expect(tracker.completedCount).toBe(1);
  });

  it('updates status by title, case-insensitively', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.update({ steps: [{ title: 'DESIGN', status: 'completed' }] });
    expect(tracker.plan?.steps[0]?.status).toBe('completed');
  });

  it('attaches notes', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.update({
      steps: [{ id: 'step_2', status: 'in_progress', notes: 'halfway' }],
    });
    expect(tracker.plan?.steps[1]?.notes).toBe('halfway');
  });

  it('ignores unknown steps and statuses', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    const updated = tracker.update({
      steps: [{ id: 'step_99', status: 'completed' }],
    });
    expect(updated?.steps.every((step) => step.status === 'pending')).toBe(true);
  });

  it('updates the goal', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.update({ goal: 'new goal' });
    expect(tracker.plan?.goal).toBe('new goal');
  });

  it('returns undefined when updating with no plan', () => {
    expect(new PlanTracker().update({ steps: [] })).toBeUndefined();
  });

  it('completes every unfinished step', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.update({ steps: [{ id: 'step_1', status: 'skipped' }] });
    tracker.completeAll();

    expect(tracker.completedCount).toBe(1);
    expect(tracker.plan?.steps[1]?.status).toBe('completed');
    expect(tracker.plan?.steps[0]?.status).toBe('skipped');
  });

  it('completeAll is a no-op without a plan', () => {
    const tracker = new PlanTracker();
    expect(() => tracker.completeAll()).not.toThrow();
  });

  it('resets', () => {
    const tracker = new PlanTracker();
    tracker.set(plan());
    tracker.reset();
    expect(tracker.plan).toBeUndefined();
  });

  it('notifies listeners until unsubscribed', () => {
    const tracker = new PlanTracker();
    const listener = vi.fn();
    const unsubscribe = tracker.onChange(listener);

    tracker.set(plan());
    tracker.update({ steps: [{ id: 'step_1', status: 'completed' }] });
    expect(listener).toHaveBeenCalledTimes(2);
    expect(listener).toHaveBeenLastCalledWith(
      expect.objectContaining({ steps: expect.any(Array) }),
    );

    unsubscribe();
    tracker.update({ steps: [{ id: 'step_2', status: 'completed' }] });
    expect(listener).toHaveBeenCalledTimes(2);
  });
});
