import { createId } from './id.js';

/** Log levels, ordered from most to least verbose. */
export const LOG_LEVELS = ['debug', 'info', 'warn', 'error', 'silent'] as const;

export type LogLevel = (typeof LOG_LEVELS)[number];

export interface LogFields {
  readonly [key: string]: unknown;
}

export interface Logger {
  debug(message: string, fields?: LogFields): void;
  info(message: string, fields?: LogFields): void;
  warn(message: string, fields?: LogFields): void;
  error(message: string, fields?: LogFields): void;
  /** Derive a logger that stamps every record with `fields`. */
  child(fields: LogFields): Logger;
}

const LEVEL_WEIGHT: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
  silent: 100,
};

export interface ConsoleLoggerOptions {
  readonly level?: LogLevel;
  readonly base?: LogFields;
  readonly sink?: Pick<Console, 'log' | 'warn' | 'error'>;
}

/**
 * Structured logger over `console`. Every record is emitted as one JSON line,
 * which keeps logs greppable and machine-parseable without a dependency.
 */
export function createConsoleLogger(options: ConsoleLoggerOptions = {}): Logger {
  const level = options.level ?? 'info';
  const base = options.base ?? {};
  const sink = options.sink ?? console;

  const write = (
    recordLevel: Exclude<LogLevel, 'silent'>,
    message: string,
    fields?: LogFields,
  ) => {
    if (LEVEL_WEIGHT[recordLevel] < LEVEL_WEIGHT[level]) return;
    const payload = JSON.stringify({
      level: recordLevel,
      msg: message,
      ...base,
      ...fields,
    });
    if (recordLevel === 'error') sink.error(payload);
    else if (recordLevel === 'warn') sink.warn(payload);
    else sink.log(payload);
  };

  return {
    debug: (message, fields) => write('debug', message, fields),
    info: (message, fields) => write('info', message, fields),
    warn: (message, fields) => write('warn', message, fields),
    error: (message, fields) => write('error', message, fields),
    child: (fields) => createConsoleLogger({ ...options, base: { ...base, ...fields } }),
  };
}

/** Discards everything. The default, so the library never writes to stdout. */
export function noopLogger(): Logger {
  return {
    debug: () => undefined,
    info: () => undefined,
    warn: () => undefined,
    error: () => undefined,
    child: () => noopLogger(),
  };
}

export interface MemoryLoggerOptions {
  readonly level?: LogLevel;
  readonly limit?: number;
}

export interface LogRecord {
  readonly level: Exclude<LogLevel, 'silent'>;
  readonly message: string;
  readonly fields: LogFields;
  readonly timestamp: string;
}

/**
 * In-memory logger for tests and debugging. Records are inspectable via
 * {@link MemoryLogSink.records} and cap the stored history to avoid leaks.
 */
export class MemoryLogSink implements Logger {
  readonly records: LogRecord[] = [];
  private readonly limit: number;
  private readonly minWeight: number;
  private readonly context: LogFields;

  constructor(options: MemoryLoggerOptions & { context?: LogFields } = {}) {
    this.limit = options.limit ?? 500;
    this.minWeight = LEVEL_WEIGHT[options.level ?? 'debug'];
    this.context = options.context ?? {};
  }

  private write = (
    level: Exclude<LogLevel, 'silent'>,
    message: string,
    fields?: LogFields,
  ): void => {
    if (LEVEL_WEIGHT[level] < this.minWeight) return;
    this.records.push({
      level,
      message,
      fields: { ...this.context, ...fields },
      timestamp: new Date().toISOString(),
    });
    if (this.records.length > this.limit) this.records.shift();
  };

  debug = (message: string, fields?: LogFields) => this.write('debug', message, fields);
  info = (message: string, fields?: LogFields) => this.write('info', message, fields);
  warn = (message: string, fields?: LogFields) => this.write('warn', message, fields);
  error = (message: string, fields?: LogFields) => this.write('error', message, fields);

  child(fields: LogFields): Logger {
    return new MemoryLogSink({
      level: 'debug',
      limit: this.limit,
      context: { ...this.context, ...fields },
    });
  }

  /** All records whose message contains `needle`. */
  find(needle: string): LogRecord[] {
    return this.records.filter((r) => r.message.includes(needle));
  }

  clear(): void {
    this.records.length = 0;
  }
}

/**
 * Wrap any logger so every record gets a freshly minted id (a run id or trace
 * id). Handy for correlating provider calls with agent iterations.
 */
export function withIdLogger(logger: Logger, id: string = createId('run')): Logger {
  return logger.child({ runId: id });
}
