import { ToolExecutionError } from '../../errors.js';
import type { Schema } from '../../schema.js';
import { defineTool } from '../registry.js';

/**
 * A recursive-descent expression evaluator.
 *
 * Deliberately not `eval` or `new Function`: tool arguments come from a model,
 * so they are untrusted input. This parser only understands arithmetic — there
 * is no property access, no assignment, and no way to reach the host scope.
 */

type TokenType = 'number' | 'ident' | 'operator' | 'lparen' | 'rparen' | 'comma' | 'eof';

interface Token {
  readonly type: TokenType;
  readonly value: string;
  readonly position: number;
}

const OPERATORS = ['+', '-', '*', '/', '%', '^', '**', '(', ')', ','];

const CONSTANTS: Readonly<Record<string, number>> = {
  pi: Math.PI,
  π: Math.PI,
  e: Math.E,
  tau: Math.PI * 2,
};

const FUNCTIONS: Readonly<Record<string, (...args: number[]) => number>> = {
  abs: Math.abs,
  ceil: Math.ceil,
  cos: Math.cos,
  exp: Math.exp,
  floor: Math.floor,
  ln: Math.log,
  log: (value, base) =>
    base === undefined ? Math.log10(value) : Math.log(value) / Math.log(base),
  log2: Math.log2,
  log10: Math.log10,
  max: (...args) => Math.max(...args),
  min: (...args) => Math.min(...args),
  pow: (base, exponent) => base ** exponent,
  round: (value, digits = 0) => {
    const factor = 10 ** digits;
    return Math.round(value * factor) / factor;
  },
  sign: Math.sign,
  sin: Math.sin,
  sqrt: Math.sqrt,
  tan: Math.tan,
  trunc: Math.trunc,
};

/**
 * Arity bounds per function, so `sqrt(1, 2)` fails loudly instead of silently
 * ignoring the extra argument the way `Math.sqrt` does.
 */
const ARITY: Readonly<Record<string, { min: number; max: number }>> = {
  abs: { min: 1, max: 1 },
  ceil: { min: 1, max: 1 },
  cos: { min: 1, max: 1 },
  exp: { min: 1, max: 1 },
  floor: { min: 1, max: 1 },
  ln: { min: 1, max: 1 },
  log: { min: 1, max: 2 },
  log2: { min: 1, max: 1 },
  log10: { min: 1, max: 1 },
  max: { min: 1, max: Number.POSITIVE_INFINITY },
  min: { min: 1, max: Number.POSITIVE_INFINITY },
  pow: { min: 2, max: 2 },
  round: { min: 1, max: 2 },
  sign: { min: 1, max: 1 },
  sin: { min: 1, max: 1 },
  sqrt: { min: 1, max: 1 },
  tan: { min: 1, max: 1 },
  trunc: { min: 1, max: 1 },
};

class ExpressionParser {
  private index = 0;

  constructor(private readonly tokens: readonly Token[]) {}

  parse(): number {
    const value = this.parseExpression();
    this.expectType('eof');
    return value;
  }

  /** additive -> multiplicative -> unary -> power -> primary */
  private parseExpression(): number {
    let left = this.parseTerm();
    for (;;) {
      if (this.check('+')) {
        this.advance();
        left += this.parseTerm();
      } else if (this.check('-')) {
        this.advance();
        left -= this.parseTerm();
      } else {
        return left;
      }
    }
  }

  private parseTerm(): number {
    let left = this.parseUnary();
    for (;;) {
      if (this.check('*') || this.check('/') || this.check('%')) {
        const operator = this.advance().value;
        const right = this.parseUnary();
        if (operator === '*') left *= right;
        else if (operator === '/') {
          if (right === 0) throw new RangeError('Division by zero');
          left /= right;
        } else {
          if (right === 0) throw new RangeError('Modulo by zero');
          left %= right;
        }
      } else {
        return left;
      }
    }
  }

  private parseUnary(): number {
    if (this.check('-')) {
      this.advance();
      return -this.parseUnary();
    }
    if (this.check('+')) {
      this.advance();
      return this.parseUnary();
    }
    return this.parsePower();
  }

  /** Right-associative, so `2^3^2 === 512`. */
  private parsePower(): number {
    const base = this.parsePrimary();
    if (this.check('^') || this.check('**')) {
      this.advance();
      const exponent = this.parseUnary();
      return base ** exponent;
    }
    return base;
  }

  private parsePrimary(): number {
    const token = this.peek();
    if (token === undefined) throw new SyntaxError('Unexpected end of expression');

    if (token.type === 'number') {
      this.advance();
      return Number(token.value);
    }

    if (token.type === 'lparen') {
      this.advance();
      const value = this.parseExpression();
      this.expectType('rparen');
      return value;
    }

    if (token.type === 'ident') {
      this.advance();
      const name = token.value.toLowerCase();

      if (this.checkType('lparen')) {
        this.advance();
        const args: number[] = [];
        if (!this.checkType('rparen')) {
          args.push(this.parseExpression());
          while (this.checkType('comma')) {
            this.advance();
            args.push(this.parseExpression());
          }
        }
        this.expectType('rparen');
        const fn = FUNCTIONS[name];
        if (fn === undefined) throw new SyntaxError(`Unknown function "${token.value}"`);
        const arity = ARITY[name]!;
        if (args.length < arity.min || args.length > arity.max) {
          const expected =
            arity.min === arity.max ? `${arity.min}` : `${arity.min} to ${arity.max}`;
          throw new SyntaxError(
            `${token.value}() expects ${expected} argument(s) but received ${args.length}`,
          );
        }
        return this.safe(fn, args, token.value);
      }

      if (Object.hasOwn(CONSTANTS, name)) return CONSTANTS[name]!;
      throw new SyntaxError(`Unknown identifier "${token.value}"`);
    }

    throw new SyntaxError(`Unexpected token "${token.value}"`);
  }

  private safe(fn: (...args: number[]) => number, args: number[], label: string): number {
    const result = fn(...args);
    if (Number.isNaN(result))
      throw new RangeError(`${label} is not defined for these inputs`);
    return result;
  }

  private peek(): Token | undefined {
    return this.tokens[this.index];
  }

  /** Value comparison, for operators. */
  private check(value: string): boolean {
    return this.tokens[this.index]?.value === value;
  }

  /** Type comparison, for punctuation whose value is only a character. */
  private checkType(type: TokenType): boolean {
    return this.tokens[this.index]?.type === type;
  }

  private advance(): Token {
    const token = this.tokens[this.index];
    if (token === undefined) throw new SyntaxError('Unexpected end of expression');
    this.index++;
    return token;
  }

  private expectType(type: TokenType): void {
    const token = this.advance();
    if (token.type !== type) {
      throw new SyntaxError(`Expected ${type} but found "${token.value}"`);
    }
  }
}

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < input.length) {
    const char = input[i]!;

    if (char === ' ' || char === '\t' || char === '\n' || char === '\r') {
      i++;
      continue;
    }

    if (char >= '0' && char <= '9') {
      const start = i;
      while (i < input.length && /[0-9._]/.test(input[i]!)) i++;
      if (input[i] === 'e' || input[i] === 'E') {
        const exponentStart = i;
        i++;
        if (input[i] === '+' || input[i] === '-') i++;
        const digit = input[i];
        if (digit !== undefined && /[0-9]/.test(digit)) {
          while (i < input.length && /[0-9]/.test(input[i]!)) i++;
        } else {
          i = exponentStart;
        }
      }
      const text = input.slice(start, i);
      if (!Number.isFinite(Number(text))) {
        throw new SyntaxError(`Invalid number "${text}"`);
      }
      tokens.push({ type: 'number', value: text, position: start });
      continue;
    }

    if (/[a-zA-Zπ_]/.test(char)) {
      const start = i;
      while (i < input.length && /[a-zA-Z0-9_π]/.test(input[i]!)) i++;
      tokens.push({ type: 'ident', value: input.slice(start, i), position: start });
      continue;
    }

    const twoChar = input.slice(i, i + 2);
    if (twoChar === '**') {
      tokens.push({ type: 'operator', value: '**', position: i });
      i += 2;
      continue;
    }

    if (OPERATORS.includes(char)) {
      tokens.push({
        type:
          char === '('
            ? 'lparen'
            : char === ')'
              ? 'rparen'
              : char === ','
                ? 'comma'
                : 'operator',
        value: char,
        position: i,
      });
      i++;
      continue;
    }

    throw new SyntaxError(`Unexpected character "${char}" at position ${i}`);
  }

  tokens.push({ type: 'eof', value: '<end>', position: input.length });
  return tokens;
}

/**
 * Evaluate an arithmetic expression.
 *
 * ```ts
 * evaluateExpression('2 * (3 + 4) - sqrt(16)'); // 10
 * ```
 *
 * @throws {SyntaxError} on malformed input.
 * @throws {ArithmeticError} on division by zero, NaN, or infinity results.
 */
export function evaluateExpression(expression: string): number {
  const trimmed = expression.trim();
  if (trimmed === '') throw new SyntaxError('Expression is empty');
  if (trimmed.length > 1000) throw new SyntaxError('Expression is too long');
  return new ExpressionParser(tokenize(trimmed)).parse();
}

/** The set of identifiers this evaluator understands. */
export function supportedFunctions(): string[] {
  return Object.keys(FUNCTIONS).sort();
}

/** Minimal string-bounded schema, so built-ins need no zod dependency. */
function boundedStringSchema(maxLength: number): Schema<{ expression: string }> {
  const ok = (input: unknown) =>
    typeof input === 'object' &&
    input !== null &&
    typeof (input as { expression?: unknown }).expression === 'string' &&
    (input as { expression: string }).expression.length <= maxLength;

  const issues = (input: unknown) => {
    const value = (input as { expression?: unknown } | null)?.expression;
    if (value === undefined) return [{ path: ['expression'], message: 'required' }];
    if (typeof value !== 'string') {
      return [{ path: ['expression'], message: 'must be a string' }];
    }
    return [{ path: ['expression'], message: `must be at most ${maxLength} characters` }];
  };

  return {
    safeParse: (input: unknown) =>
      ok(input)
        ? { success: true, data: input as { expression: string } }
        : { success: false, error: { issues: issues(input) } },
    parse: (input: unknown) => {
      if (!ok(input)) throw new ToolExecutionError('Invalid calculator arguments');
      return input as { expression: string };
    },
  };
}

export interface CalculatorToolOptions {
  /** Cap on expression length. Default `1000`. */
  readonly maxLength?: number;
  /** Cap on `round()` digits. Default `10`. */
  readonly maxDigits?: number;
}

export interface CalculatorArgs {
  readonly expression: string;
}

export interface CalculatorResult {
  readonly expression: string;
  readonly result: number;
}

/**
 * Arithmetic calculator tool. Safe by construction: no `eval`, no host access,
 * and a hard length limit.
 *
 * ```ts
 * const calculator = createCalculatorTool();
 * await calculator.execute({ expression: 'sqrt(2) + 2^10' }, ctx);
 * // { expression: 'sqrt(2) + 2^10', result: 1025.414… }
 * ```
 */
export function createCalculatorTool(options: CalculatorToolOptions = {}) {
  const maxLength = options.maxLength ?? 1000;
  const maxDigits = options.maxDigits ?? 10;

  return defineTool<CalculatorArgs, CalculatorResult>({
    name: 'calculator',
    description:
      'Evaluate an arithmetic expression. Supports + - * / % ^, parentheses, ' +
      `constants (pi, e, tau) and functions (${supportedFunctions().join(', ')}). ` +
      'Use this instead of doing mental arithmetic.',
    parameters: boundedStringSchema(maxLength),
    jsonSchema: {
      type: 'object',
      properties: { expression: { type: 'string', description: 'e.g. "2 * (3 + 4)"' } },
      required: ['expression'],
    },
    serialize: (result) =>
      `${result.expression} = ${roundForDisplay(result.result, maxDigits)}`,
    execute({ expression }) {
      try {
        const result = evaluateExpression(expression);
        if (!Number.isFinite(result)) {
          throw new ToolExecutionError(`Result is not a finite number: ${expression}`);
        }
        return { expression, result };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new ToolExecutionError(`Could not evaluate "${expression}": ${message}`, {
          toolName: 'calculator',
        });
      }
    },
  });
}

function roundForDisplay(value: number, maxDigits: number): number {
  if (Number.isInteger(value)) return value;
  return Number(value.toFixed(Math.min(maxDigits, 15)));
}
