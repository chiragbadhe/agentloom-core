import { ToolExecutionError, ToolValidationError } from '../../errors.js';
import type { Schema } from '../../schema.js';
import { throwIfAborted } from '../../utils/async.js';
import { truncate } from '../../utils/text.js';
import { defineTool } from '../registry.js';
import type { ToolContext } from '../types.js';

export type FileSystemAction =
  'read' | 'write' | 'append' | 'list' | 'delete' | 'exists' | 'stat';

export interface FileSystemToolArgs {
  readonly action: FileSystemAction;
  /** Path relative to the sandbox root, or absolute inside it. */
  readonly path: string;
  /** Content for `write` and `append`. */
  readonly content?: string;
  readonly encoding?: 'utf8' | 'base64';
}

export interface FileSystemToolResult {
  readonly action: FileSystemAction;
  readonly path: string;
  readonly content?: string;
  readonly entries?: { name: string; type: 'file' | 'directory'; size: number }[];
  readonly exists?: boolean;
  readonly size?: number;
  readonly modifiedAt?: string;
}

export interface FileSystemToolOptions {
  /**
   * The only directory the tool may touch. **Required** — there is no
   * "unrestricted" mode, because a model-controlled path is a sandbox escape.
   */
  readonly root: string;
  /** Allow `write`, `append`, and `delete`. Default `false` (read-only). */
  readonly allowWrite?: boolean;
  readonly maxReadBytes?: number;
  readonly maxWriteBytes?: number;
  readonly defaultEncoding?: 'utf8' | 'base64';
  /** Injectable for tests. */
  readonly fs?: FileSystemLike;
}

/** The slice of `node:fs/promises` we depend on, so tests can substitute it. */
export interface FileSystemLike {
  readFile(path: string, encoding: 'utf8'): Promise<string>;
  writeFile(path: string, data: string, encoding: 'utf8'): Promise<void>;
  appendFile(path: string, data: string, encoding: 'utf8'): Promise<void>;
  readdir(
    path: string,
    options: { withFileTypes: true },
  ): Promise<{ name: string; isDirectory(): boolean; isFile(): boolean }[]>;
  stat(path: string): Promise<{ size: number; isDirectory(): boolean; mtime: Date }>;
  rm(path: string, options: { recursive?: boolean; force?: boolean }): Promise<void>;
  mkdir(path: string, options: { recursive: boolean }): Promise<void>;
  realpath(path: string): Promise<string>;
}

/** Actions that need `allowWrite`. */
const MUTATING_ACTIONS: readonly string[] = ['write', 'append', 'delete'];
const READ_ACTIONS: readonly string[] = ['read', 'list', 'exists', 'stat'];

async function loadFs(): Promise<FileSystemLike> {
  // Dynamic so bundlers targeting the browser do not hard-require node:fs.
  const nodeFs = await import('node:fs/promises');
  const candidate = nodeFs as unknown as Partial<FileSystemLike> & {
    default?: Partial<FileSystemLike>;
    promises?: Partial<FileSystemLike>;
  };
  // CJS interop exposes the API either directly, on `default`, or on `promises`.
  const resolved = candidate.default ?? candidate;
  const fs = candidate.promises ?? resolved;
  if (typeof fs.mkdir !== 'function') {
    throw new ToolExecutionError('node:fs/promises is unavailable in this runtime');
  }
  return fs as FileSystemLike;
}

/**
 * Sandbox a filesystem tool to a single directory.
 *
 * Every path is resolved and checked against `root`, including after symlink
 * resolution, so `../../etc/passwd` and symlink escapes are both rejected.
 */
export function createFileSystemTool(options: FileSystemToolOptions) {
  if (options.root.trim() === '') {
    throw new ToolValidationError(
      'createFileSystemTool requires a non-empty root directory',
    );
  }
  const root = options.root;
  const allowWrite = options.allowWrite === true;
  const maxReadBytes = options.maxReadBytes ?? 500_000;
  const maxWriteBytes = options.maxWriteBytes ?? 1_000_000;
  const defaultEncoding = options.defaultEncoding ?? 'utf8';

  const parameterSchema: Schema<FileSystemToolArgs> = {
    safeParse: (input: unknown) => {
      const issues: { path: (string | number)[]; message: string }[] = [];
      if (typeof input !== 'object' || input === null) {
        return {
          success: false,
          error: { issues: [{ path: [], message: 'expected an object' }] },
        };
      }
      const args = input as Record<string, unknown>;
      const actions: readonly string[] = MUTATING_ACTIONS.concat(READ_ACTIONS);
      if (typeof args.action !== 'string' || !actions.includes(args.action)) {
        issues.push({
          path: ['action'],
          message: `must be one of ${actions.join(', ')}`,
        });
      }
      if (typeof args.path !== 'string' || args.path.trim() === '') {
        issues.push({ path: ['path'], message: 'required' });
      }
      const writes = args.action === 'write' || args.action === 'append';
      if (writes && typeof args.content !== 'string') {
        issues.push({
          path: ['content'],
          message: `required for action "${String(args.action)}"`,
        });
      }
      if (writes && !allowWrite) {
        issues.push({
          path: ['action'],
          message: 'write access is disabled for this tool',
        });
      }
      if (args.action === 'delete' && !allowWrite) {
        issues.push({
          path: ['action'],
          message: 'write access is disabled for this tool',
        });
      }
      return issues.length === 0
        ? { success: true, data: input as FileSystemToolArgs }
        : { success: false, error: { issues } };
    },
    parse: (input: unknown) => {
      const result = parameterSchema.safeParse(input);
      if (!result.success) throw new ToolValidationError('Invalid file_system arguments');
      return result.data;
    },
  };

  return defineTool<FileSystemToolArgs, FileSystemToolResult, unknown>({
    name: 'file_system',
    description:
      'Read, write, and list files inside a sandboxed directory. Actions: "read", ' +
      '"write", "append", "list", "delete", "exists", "stat". Paths must stay ' +
      'inside the sandbox. Write access may be disabled.',
    parameters: parameterSchema,
    jsonSchema: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['read', 'write', 'append', 'list', 'delete', 'exists', 'stat'],
        },
        path: { type: 'string', description: 'Path relative to the sandbox root' },
        content: { type: 'string', description: 'Content for write/append' },
      },
      required: ['action', 'path'],
    },
    async execute(args, context: ToolContext) {
      const fs = options.fs ?? (await loadFs());
      // Re-check permissions here: the schema enforces them too, but `execute`
      // is reachable directly and this tool touches the disk.
      if (!allowWrite && MUTATING_ACTIONS.includes(args.action)) {
        throw new ToolValidationError('write access is disabled for this tool', {
          toolName: 'file_system',
        });
      }
      const target = resolveInsideRoot(root, args.path);
      await assertInsideRoot(fs, root, target, args.path);
      const encoding = args.encoding ?? defaultEncoding;
      throwIfAborted(context.signal);

      switch (args.action) {
        case 'read': {
          const raw = await fs.readFile(target, 'utf8');
          if (raw.length > maxReadBytes) {
            throw new ToolValidationError(
              `File is ${raw.length} characters, above the ${maxReadBytes} limit. Use "list" or "stat" first.`,
            );
          }
          const content =
            encoding === 'base64' ? Buffer.from(raw, 'utf8').toString('base64') : raw;
          return {
            action: 'read',
            path: target,
            content: truncate(content, maxReadBytes),
          };
        }
        case 'write': {
          const content = decodeContent(args.content ?? '', encoding);
          if (content.length > maxWriteBytes) {
            throw new ToolValidationError(
              `Refusing to write ${content.length} characters (limit ${maxWriteBytes})`,
            );
          }
          await ensureDirectory(fs, dirname(target));
          await fs.writeFile(target, content, 'utf8');
          return { action: 'write', path: target, size: Buffer.byteLength(content) };
        }
        case 'append': {
          const content = decodeContent(args.content ?? '', encoding);
          await ensureDirectory(fs, dirname(target));
          await fs.appendFile(target, content, 'utf8');
          return { action: 'append', path: target, size: Buffer.byteLength(content) };
        }
        case 'list': {
          const entries = await fs.readdir(target, { withFileTypes: true });
          const withSizes = await Promise.all(
            entries.map(
              async (
                entry,
              ): Promise<NonNullable<FileSystemToolResult['entries']>[number]> => {
                const isDirectory = entry.isDirectory();
                const size = isDirectory
                  ? 0
                  : await fs
                      .stat(join(target, entry.name))
                      .then((stat) => stat.size)
                      .catch(() => 0);
                return {
                  name: entry.name,
                  type: isDirectory ? 'directory' : 'file',
                  size,
                };
              },
            ),
          );
          return { action: 'list', path: target, entries: withSizes };
        }
        case 'delete': {
          await fs.rm(target, { recursive: true, force: true });
          return { action: 'delete', path: target };
        }
        case 'exists': {
          const exists = await fs
            .stat(target)
            .then(() => true)
            .catch(() => false);
          return { action: 'exists', path: target, exists };
        }
        case 'stat': {
          const stat = await fs.stat(target);
          if (stat.size > maxReadBytes) {
            throw new ToolValidationError(
              `File is ${stat.size} bytes, above the ${maxReadBytes} byte limit`,
            );
          }
          return {
            action: 'stat',
            path: target,
            size: stat.size,
            exists: true,
            ...(stat.isDirectory() ? {} : { modifiedAt: stat.mtime.toISOString() }),
          };
        }
        default:
          throw new ToolValidationError(`Unsupported action`, {
            toolName: 'file_system',
          });
      }
    },
  });
}

/**
 * Resolve `candidate` and assert it is inside `root`.
 *
 * Rejects `..` traversal, absolute paths outside the root, and (on platforms
 * that support it) symlinks pointing out of the sandbox.
 */
export function resolveInsideRoot(root: string, candidate: string): string {
  const normalizedRoot = root.replace(/\/+$/, '');
  const resolvedRoot = pathResolve(normalizedRoot);
  const target = pathResolve(
    candidate.startsWith('/') || /^[A-Za-z]:/.test(candidate)
      ? candidate
      : `${normalizedRoot}/${candidate}`,
  );

  if (target !== resolvedRoot && !target.startsWith(`${resolvedRoot}/`)) {
    throw new ToolValidationError(
      `Path "${candidate}" escapes the sandbox root (${normalizedRoot})`,
    );
  }
  return target;
}

function pathResolve(input: string): string {
  const isAbsolute = input.startsWith('/');
  const segments: string[] = [];
  for (const segment of input.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment === '..') {
      segments.pop();
      continue;
    }
    segments.push(segment);
  }
  return (isAbsolute ? '/' : '') + segments.join('/');
}

function dirname(input: string): string {
  const index = input.lastIndexOf('/');
  return index <= 0 ? '/' : input.slice(0, index);
}

function join(base: string, name: string): string {
  return base.endsWith('/') ? `${base}${name}` : `${base}/${name}`;
}

function decodeContent(content: string, encoding: 'utf8' | 'base64'): string {
  if (encoding !== 'base64') return content;
  const decoded = Buffer.from(content, 'base64').toString('utf8');
  if (decoded.includes('\uFFFD')) {
    throw new ToolValidationError('content is not valid base64');
  }
  return decoded;
}

/**
 * Guard against symlink escapes: the lexical check in `resolveInsideRoot`
 * cannot see through links, so resolve the deepest existing ancestor and make
 * sure it is still inside the root.
 */
async function assertInsideRoot(
  fs: FileSystemLike,
  root: string,
  target: string,
  display: string,
): Promise<void> {
  const resolvedRoot = await fs.realpath(root).catch(() => root);
  const existing = await nearestExisting(fs, target);
  if (existing === undefined) return; // nothing on disk yet: lexical check is enough

  const real = await fs.realpath(existing);
  if (real !== resolvedRoot && !real.startsWith(`${resolvedRoot}/`)) {
    throw new ToolValidationError(
      `Path "${display}" escapes the sandbox root through a symbolic link`,
    );
  }
}

async function nearestExisting(
  fs: FileSystemLike,
  target: string,
): Promise<string | undefined> {
  let current = target;
  for (;;) {
    try {
      await fs.stat(current);
      return current;
    } catch {
      const parent = dirname(current);
      if (parent === current) return undefined;
      current = parent;
    }
  }
}

async function ensureDirectory(fs: FileSystemLike, path: string): Promise<void> {
  try {
    await fs.mkdir(path, { recursive: true });
  } catch (error) {
    throw new ToolExecutionError(
      `Could not create directory ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
