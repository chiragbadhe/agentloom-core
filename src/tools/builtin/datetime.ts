import { ToolExecutionError } from '../../errors.js';
import type { Schema } from '../../schema.js';
import { defineTool } from '../registry.js';

export type DateTimeAction = 'now' | 'parse' | 'format' | 'add' | 'diff';

export interface DateTimeArgs {
  readonly action: DateTimeAction;
  /** IANA timezone, e.g. `Europe/Berlin`. Defaults to UTC. */
  readonly timezone?: string;
  /** Required for `parse` and `format`. */
  readonly input?: string;
  /** Required for `add` and `diff`. A duration like `-3d` or `+2h30m`. */
  readonly amount?: string;
  /** Second operand for `diff`. */
  readonly other?: string;
  /** Output format for `format`. Defaults to ISO 8601. */
  readonly format?: 'iso' | 'date' | 'time' | 'rfc2822' | 'relative' | 'unix';
}

export interface DateTimeResult {
  readonly iso: string;
  readonly formatted: string;
  readonly timezone: string;
  readonly unix: number;
  readonly weekday: string;
  readonly extra?: Record<string, unknown>;
}

const DURATION_UNITS: Readonly<Record<string, number>> = {
  ms: 1,
  s: 1000,
  m: 60_000,
  h: 3_600_000,
  d: 86_400_000,
  w: 604_800_000,
  y: 31_536_000_000,
};

function schema(): Schema<DateTimeArgs> {
  const actions: readonly string[] = ['now', 'parse', 'format', 'add', 'diff'];
  return {
    safeParse: (input: unknown) => {
      const issues: { path: (string | number)[]; message: string }[] = [];
      if (typeof input !== 'object' || input === null) {
        return {
          success: false,
          error: { issues: [{ path: [], message: 'expected an object' }] },
        };
      }
      const args = input as Record<string, unknown>;
      if (typeof args.action !== 'string') {
        issues.push({ path: ['action'], message: 'required' });
      } else if (!actions.includes(args.action)) {
        issues.push({
          path: ['action'],
          message: `must be one of ${actions.join(', ')}`,
        });
      }
      const needsInput = args.action === 'parse' || args.action === 'format';
      const needsAmount = args.action === 'add' || args.action === 'diff';
      if (needsInput && typeof args.input !== 'string') {
        issues.push({
          path: ['input'],
          message: `required for action "${String(args.action)}"`,
        });
      }
      if (needsAmount && typeof args.amount !== 'string') {
        issues.push({
          path: ['amount'],
          message: `required for action "${String(args.action)}"`,
        });
      }
      if (args.action === 'diff' && typeof args.other !== 'string') {
        issues.push({ path: ['other'], message: 'required for action "diff"' });
      }
      if (args.timezone !== undefined && typeof args.timezone !== 'string') {
        issues.push({ path: ['timezone'], message: 'must be a string' });
      }
      return issues.length === 0
        ? { success: true, data: input as DateTimeArgs }
        : { success: false, error: { issues } };
    },
    parse: (input: unknown) => {
      const result = schema().safeParse(input);
      if (!result.success) throw new ToolExecutionError('Invalid date_time arguments');
      return result.data;
    },
  };
}

/** Parse a compact duration such as `-3d`, `+2h30m`, `90s`. */
export function parseDuration(input: string): number {
  const trimmed = input.trim().toLowerCase();
  const sign = trimmed.startsWith('-') ? -1 : trimmed.startsWith('+') ? 1 : 1;
  const body = trimmed.replace(/^[+-]/, '');
  if (body === '') throw new ToolExecutionError('Duration is empty');

  let total = 0;
  let matched = false;
  const pattern = /(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w|y)/g;

  for (const match of body.matchAll(pattern)) {
    const amount = Number(match[1]);
    const unit = match[2]!;
    const multiplier = DURATION_UNITS[unit];
    if (multiplier === undefined) continue;
    total += amount * multiplier;
    matched = true;
  }

  if (!matched) {
    throw new ToolExecutionError(
      `Could not parse duration "${input}". Use forms like "3d", "-2h", "1h30m", "90s".`,
    );
  }
  return sign * total;
}

function formatInTimezone(
  date: Date,
  timezone: string,
  format: NonNullable<DateTimeArgs['format']>,
): string {
  if (format === 'unix') return String(Math.floor(date.getTime() / 1000));
  if (format === 'iso') return date.toISOString();
  if (format === 'rfc2822') return date.toUTCString();

  const options: Intl.DateTimeFormatOptions =
    format === 'date'
      ? { year: 'numeric', month: '2-digit', day: '2-digit' }
      : format === 'time'
        ? {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hourCycle: 'h23',
          }
        : {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hourCycle: 'h23',
            timeZoneName: 'short',
          };

  const parts = new Intl.DateTimeFormat('en-US', { timeZone: timezone, ...options })
    .formatToParts(date)
    .filter((part) => part.type !== 'literal');

  const values: Record<string, string> = {};
  for (const part of parts) values[part.type] = part.value;

  // Deterministic, locale-independent output: a model should never have to
  // guess whether "03/01/2024" means March 1st or January 3rd.
  if (format === 'date') return `${values['year']}-${values['month']}-${values['day']}`;
  if (format === 'time') {
    return `${values['hour']}:${values['minute']}:${values['second']}`;
  }
  return (
    `${values['month']} ${values['day']}, ${values['year']} ` +
    `${values['hour']}:${values['minute']} ${values['timeZoneName'] ?? ''}`
  ).trim();
}

function parseDate(input: string): Date {
  const date = new Date(input);
  if (Number.isNaN(date.getTime())) {
    throw new ToolExecutionError(`Could not parse date "${input}"`, {
      toolName: 'date_time',
    });
  }
  return date;
}

function assertTimezone(timezone: string): void {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: timezone });
  } catch {
    throw new ToolExecutionError(`Unknown timezone "${timezone}"`, {
      toolName: 'date_time',
    });
  }
}

/**
 * Date and time utility: current time, parsing, formatting, arithmetic, and
 * diffs. Timezone-aware and dependency-free.
 */
export function createDateTimeTool() {
  return defineTool<DateTimeArgs, DateTimeResult>({
    name: 'date_time',
    description:
      'Work with dates and times. Actions: "now" (current time), "parse" (ISO string to ' +
      'timestamp), "format" (render a date), "add" (shift by a duration like "-3d"), ' +
      '"diff" (difference between two dates). Always specify an IANA timezone ' +
      'such as "UTC" or "America/New_York" when the user mentions one.',
    parameters: schema(),
    jsonSchema: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['now', 'parse', 'format', 'add', 'diff'] },
        timezone: { type: 'string', description: 'IANA timezone, e.g. "Europe/Paris"' },
        input: { type: 'string', description: 'ISO 8601 date or datetime' },
        other: { type: 'string', description: 'Second date for "diff"' },
        amount: { type: 'string', description: 'Duration such as "-3d", "2h30m"' },
        format: {
          type: 'string',
          enum: ['iso', 'date', 'time', 'rfc2822', 'relative', 'unix'],
        },
      },
      required: ['action'],
    },
    execute({ action, timezone = 'UTC', input, other, amount, format = 'iso' }) {
      assertTimezone(timezone);
      const base = new Date();

      switch (action) {
        case 'now':
          return describe(base, timezone, format);
        case 'parse':
        case 'format':
          return describe(parseDate(input ?? ''), timezone, format);
        case 'add': {
          // `input` shifts an existing date; without it we shift "now".
          const from = input === undefined ? base : parseDate(input);
          const delta = parseDuration(amount ?? '');
          const shifted = new Date(from.getTime() + delta);
          return describe(shifted, timezone, format, {
            delta,
            appliedTo: from.toISOString(),
          });
        }
        case 'diff': {
          if (input === undefined || other === undefined) {
            throw new ToolExecutionError('diff requires "input" and "other"', {
              toolName: 'date_time',
            });
          }
          const a = parseDate(input);
          const b = parseDate(other);
          const deltaMs = b.getTime() - a.getTime();
          return describe(b, timezone, format, {
            deltaMs,
            deltaSeconds: deltaMs / 1000,
            deltaMinutes: deltaMs / 60_000,
            deltaHours: deltaMs / 3_600_000,
            deltaDays: deltaMs / 86_400_000,
          });
        }
        default:
          throw new ToolExecutionError(`Unsupported action "${String(action)}"`, {
            toolName: 'date_time',
          });
      }
    },
  });
}

function describe(
  date: Date,
  timezone: string,
  format: NonNullable<DateTimeArgs['format']>,
  extra?: Record<string, unknown>,
): DateTimeResult {
  return {
    iso: date.toISOString(),
    formatted: formatInTimezone(date, timezone, format),
    timezone,
    unix: Math.floor(date.getTime() / 1000),
    weekday: new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      weekday: 'long',
    }).format(date),
    ...(extra ? { extra } : {}),
  };
}
