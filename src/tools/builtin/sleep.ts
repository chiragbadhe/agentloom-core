import { ToolValidationError } from '../../errors.js';
import type { Schema } from '../../schema.js';
import { sleep } from '../../utils/async.js';
import { defineTool } from '../registry.js';

export interface SleepToolArgs {
  /** Milliseconds to wait. Capped at 60s to keep runs predictable. */
  readonly ms: number;
  readonly reason?: string;
}

export interface SleepToolResult {
  readonly sleptMs: number;
  readonly reason: string | undefined;
}

const MAX_SLEEP_MS = 60_000;

/**
 * Pause execution. Useful for rate-limit backoff and for agents that wait on
 * human input between steps.
 *
 * Resolves early (without throwing) when the run is aborted, so cancellation
 * is never blocked by a long sleep.
 */
export function createSleepTool(maxMs = MAX_SLEEP_MS) {
  const parameterSchema: Schema<SleepToolArgs> = {
    safeParse: (input: unknown) => {
      if (typeof input !== 'object' || input === null) {
        return {
          success: false,
          error: { issues: [{ path: [], message: 'expected an object' }] },
        };
      }
      const args = input as Record<string, unknown>;
      const issues: { path: (string | number)[]; message: string }[] = [];
      if (typeof args.ms !== 'number' || !Number.isFinite(args.ms)) {
        issues.push({ path: ['ms'], message: 'required, must be a finite number' });
      } else if (args.ms < 0) {
        issues.push({ path: ['ms'], message: 'must be >= 0' });
      } else if (args.ms > maxMs) {
        issues.push({ path: ['ms'], message: `must be <= ${maxMs}` });
      }
      if (args.reason !== undefined && typeof args.reason !== 'string') {
        issues.push({ path: ['reason'], message: 'must be a string' });
      }
      return issues.length === 0
        ? { success: true, data: input as SleepToolArgs }
        : { success: false, error: { issues } };
    },
    parse: (input: unknown) => {
      const result = parameterSchema.safeParse(input);
      if (!result.success) throw new ToolValidationError('Invalid sleep arguments');
      return result.data;
    },
  };

  return defineTool<SleepToolArgs, SleepToolResult>({
    name: 'sleep',
    description: `Pause for a number of milliseconds (max ${maxMs}). Use for backoff or to wait on external events.`,
    parameters: parameterSchema,
    jsonSchema: {
      type: 'object',
      properties: {
        ms: { type: 'number', minimum: 0, maximum: maxMs },
        reason: { type: 'string' },
      },
      required: ['ms'],
    },
    async execute({ ms, reason }, context) {
      const startedAt = Date.now();
      await sleep(ms, context.signal);
      return { sleptMs: Date.now() - startedAt, reason };
    },
  });
}
