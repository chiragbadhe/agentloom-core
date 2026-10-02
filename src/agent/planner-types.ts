/** Shared context passed to plan builders. Kept in its own module so
 * `planner.ts` has no import cycle with `config.ts`. */
export interface PlanContext {
  readonly input: string;
  readonly runId: string;
  readonly signal: AbortSignal;
}
