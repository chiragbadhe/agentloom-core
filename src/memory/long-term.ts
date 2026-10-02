import { MemoryError } from '../errors.js';
import { createId } from '../utils/id.js';
import { tokenize } from '../utils/text.js';

/**
 * A durable fact worth remembering across runs.
 *
 * ```ts
 * { role: 'user', content: 'I prefer TypeScript', metadata: { kind: 'preference' } }
 * ```
 */
export interface MemoryRecord {
  readonly id: string;
  readonly role: 'user' | 'assistant' | 'system';
  readonly content: string;
  readonly createdAt: number;
  /** Free-form: `preference`, `fact`, `episode`, ... */
  readonly metadata?: Readonly<Record<string, unknown>>;
  /** Optional pre-computed tokens, reused to speed up search. */
  readonly importance?: number;
}

export type NewMemoryRecord = Omit<MemoryRecord, 'id' | 'createdAt'> &
  Partial<Pick<MemoryRecord, 'id' | 'createdAt'>>;

export interface MemorySearchOptions {
  readonly limit?: number;
  /** Drop results below this relevance score (0–1). */
  readonly minScore?: number;
  readonly roles?: readonly MemoryRecord['role'][];
  /** Require every token to be present. */
  readonly requireAllTerms?: boolean;
  /** Include the record's own metadata in scoring. Default `false`. */
  readonly searchMetadata?: boolean;
}

export interface MemorySearchResult {
  readonly record: MemoryRecord;
  readonly score: number;
}

/**
 * Pluggable long-term memory.
 *
 * The contract is intentionally tiny so it maps onto whatever store you have —
 * Postgres + pgvector, Redis, SQLite, a hosted vector DB:
 *
 * ```ts
 * const memory: LongTermMemory = {
 *   async add(records) { /* insert *\/ },
 *   async search(query, options) { /* vector search *\/ },
 *   async remove(ids) { /* delete *\/ },
 *   async clear() { /* truncate *\/ },
 * };
 * ```
 */
export interface LongTermMemory {
  add(records: readonly NewMemoryRecord[]): Promise<void> | void;
  /**
   * Retrieve records relevant to `query`.
   *
   * May return synchronously or as a promise, so a synchronous in-memory
   * implementation does not pay for an unnecessary microtask.
   */
  search(
    query: string,
    options?: MemorySearchOptions,
  ): readonly MemorySearchResult[] | Promise<readonly MemorySearchResult[]>;
  remove(ids: readonly string[]): Promise<void> | void;
  clear(): Promise<void> | void;
}

export interface InMemoryLongTermMemoryOptions {
  /** Maximum records retained; oldest evicted first. Default `1000`. */
  readonly maxRecords?: number;
  /** Records returned by default from `search`. Default `5`. */
  readonly defaultLimit?: number;
}

/**
 * Keyword-relevance long-term memory with no external dependencies.
 *
 * Scores with TF-IDF-ish weighting over unique tokens, which is enough for
 * "remember that the user prefers X" style recall and runs entirely in
 * process. Swap in a vector store when you need semantic recall.
 */
export class InMemoryLongTermMemory implements LongTermMemory {
  private records: MemoryRecord[] = [];
  private readonly index = new Map<string, Set<number>>();
  private readonly maxRecords: number;
  private readonly defaultLimit: number;

  constructor(options: InMemoryLongTermMemoryOptions = {}) {
    this.maxRecords = options.maxRecords ?? 1000;
    this.defaultLimit = options.defaultLimit ?? 5;
  }

  get size(): number {
    return this.records.length;
  }

  add(records: readonly NewMemoryRecord[]): void {
    for (const input of records) {
      const record: MemoryRecord = {
        id: input.id ?? createId('mem'),
        role: input.role,
        content: input.content,
        createdAt: input.createdAt ?? Date.now(),
        ...(input.metadata ? { metadata: input.metadata } : {}),
        ...(input.importance !== undefined ? { importance: input.importance } : {}),
      };
      this.indexRecord(record, this.records.length);
      this.records.push(record);
    }
    this.evict();
  }

  search(query: string, options: MemorySearchOptions = {}): MemorySearchResult[] {
    const limit = options.limit ?? this.defaultLimit;
    const minScore = options.minScore ?? 0;
    const terms = tokenize(query);
    if (terms.length === 0) return [];

    const allowedRoles = options.roles;
    const scores = new Map<number, number>();

    for (const term of terms) {
      const postings = this.index.get(term);
      if (postings === undefined) continue;
      // Longer, rarer terms carry more signal than common short ones.
      const termWeight = 1 + Math.log(1 + term.length);
      for (const position of postings) {
        const record = this.records[position];
        if (record === undefined) continue;
        if (allowedRoles && !allowedRoles.includes(record.role)) continue;
        scores.set(position, (scores.get(position) ?? 0) + termWeight);
      }
    }

    const matchedTerms = new Map<number, number>();
    for (const term of terms) {
      const postings = this.index.get(term);
      if (postings === undefined) continue;
      for (const position of postings) {
        matchedTerms.set(position, (matchedTerms.get(position) ?? 0) + 1);
      }
    }

    const maxPossible = terms.length * (1 + Math.log(1 + longestLength(terms)));
    const results: MemorySearchResult[] = [];

    for (const [position, rawScore] of scores) {
      const record = this.records[position];
      if (record === undefined) continue;
      if (
        options.requireAllTerms === true &&
        (matchedTerms.get(position) ?? 0) < terms.length
      ) {
        continue;
      }
      // Recency and explicit importance both nudge otherwise-equal matches.
      const ageDays = (Date.now() - record.createdAt) / 86_400_000;
      const recency = 1 / (1 + ageDays / 30);
      const bonus = record.importance ?? 0;
      const score = Math.min(
        1,
        (rawScore / maxPossible) * 0.8 + recency * 0.1 + bonus * 0.1,
      );
      if (score >= minScore) results.push({ record, score });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  remove(ids: readonly string[]): void {
    const removing = new Set(ids);
    const kept: MemoryRecord[] = [];
    for (const record of this.records) {
      if (removing.has(record.id)) continue;
      kept.push(record);
    }
    this.reindex(kept);
  }

  clear(): void {
    this.records = [];
    this.index.clear();
  }

  /** Every stored record, newest first. */
  all(): readonly MemoryRecord[] {
    return [...this.records].reverse();
  }

  private indexRecord(record: MemoryRecord, position: number): void {
    for (const term of tokenize(record.content)) {
      let postings = this.index.get(term);
      if (postings === undefined) {
        postings = new Set();
        this.index.set(term, postings);
      }
      postings.add(position);
    }
  }

  private evict(): void {
    const overflow = this.records.length - this.maxRecords;
    if (overflow <= 0) return;
    const kept = this.records.slice(overflow);
    this.reindex(kept);
  }

  private reindex(records: readonly MemoryRecord[]): void {
    this.records = [...records];
    this.index.clear();
    this.records.forEach((record, position) => this.indexRecord(record, position));
  }
}

function longestLength(terms: readonly string[]): number {
  return terms.reduce((max, term) => Math.max(max, term.length), 0);
}

/** Options for {@link createFileLongTermMemory}. */
export interface FileLongTermMemoryOptions {
  /** Path to a JSON file. Parent directories are created on first write. */
  readonly path: string;
  readonly maxRecords?: number;
  readonly defaultLimit?: number;
  /**
   * Load and persist through these hooks instead of the filesystem — handy for
   * object storage, or for tests.
   */
  readonly adapter?: {
    load(): Promise<readonly MemoryRecord[]> | readonly MemoryRecord[];
    save(records: readonly MemoryRecord[]): Promise<void> | void;
  };
  /** Debounce writes. Default `false` (write on every mutation). */
  readonly flush?: boolean;
}

/**
 * JSON-file-backed long-term memory.
 *
 * Reads once at construction and writes on mutation. Not safe for concurrent
 * processes — use a real store for that.
 */
export function createFileLongTermMemory(
  options: FileLongTermMemoryOptions,
): LongTermMemory & { readonly memory: InMemoryLongTermMemory } {
  const inner = new InMemoryLongTermMemory({
    maxRecords: options.maxRecords,
    defaultLimit: options.defaultLimit,
  });

  const adapter: NonNullable<FileLongTermMemoryOptions['adapter']> =
    options.adapter ?? createJsonFileAdapter(options.path);

  let loaded = false;
  let pending: Promise<void> = Promise.resolve();

  const ensureLoaded = async (): Promise<void> => {
    if (loaded) return;
    loaded = true;
    try {
      const records = await adapter.load();
      if (records.length > 0) inner.add(records);
    } catch (error) {
      throw new MemoryError(
        `Could not load long-term memory: ${error instanceof Error ? error.message : String(error)}`,
        error,
      );
    }
  };

  const snapshot = (): MemoryRecord[] => [...inner.all()].reverse();

  const persist = (): Promise<void> => {
    if (options.flush === true) {
      return Promise.resolve(adapter.save(snapshot()));
    }
    pending = pending
      .then(() => adapter.save(snapshot()))
      .catch((error) => {
        throw new MemoryError(
          `Could not persist long-term memory: ${error instanceof Error ? error.message : String(error)}`,
          error,
        );
      });
    return pending;
  };

  return {
    memory: inner,
    async add(records) {
      await ensureLoaded();
      inner.add(records);
      await persist();
    },

    async search(query, searchOptions) {
      await ensureLoaded();
      return inner.search(query, searchOptions);
    },
    async remove(ids) {
      await ensureLoaded();
      inner.remove(ids);
      await persist();
    },
    async clear() {
      inner.clear();
      loaded = true;
      await persist();
    },
  };
}

function createJsonFileAdapter(
  path: string,
): NonNullable<FileLongTermMemoryOptions['adapter']> {
  return {
    async load() {
      const fs = await import('node:fs/promises');
      try {
        const raw = await fs.readFile(path, 'utf8');
        const parsed: unknown = JSON.parse(raw);
        return Array.isArray(parsed) ? (parsed as MemoryRecord[]) : [];
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === 'ENOENT') return [];
        throw error;
      }
    },
    async save(records) {
      const fs = await import('node:fs/promises');
      const nodePath = await import('node:path');
      await fs.mkdir(nodePath.dirname(path), { recursive: true }).catch(() => undefined);
      await fs.writeFile(path, JSON.stringify(records, null, 2), 'utf8');
    },
  };
}
