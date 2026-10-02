/**
 * 02 — tools: custom, built-in, approval, and policy.
 *
 * Run: `npx tsx examples/02-tools.ts`
 */
import { z } from 'zod';

import {
  Agent,
  createCalculatorTool,
  createDateTimeTool,
  createFileSystemTool,
  createHttpTool,
  defineTool,
  ToolDeniedError,
  type ToolPolicy,
} from '../src/index.js';

/**
 * A custom tool. Zod validates the arguments *and* provides the JSON Schema
 * sent to the model, so the two never drift apart.
 *
 * `context` gives you the run id, an abort signal, the shared agent state, and a
 * logger. The declared return type is what the model sees.
 */
const getWeather = defineTool({
  name: 'get_weather',
  description: 'Current weather for a city. Always call this before discussing weather.',
  parameters: z.object({
    city: z.string().min(1),
    unit: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  // Stand-in for a real API call.
  execute: async ({ city, unit }): Promise<Weather> => ({
    city,
    unit,
    temperature: unit === 'celsius' ? 17 : 63,
    conditions: 'light rain',
  }),
  // Optional: what actually reaches the model (token-cheap, model-friendly).
  serialize: (result) =>
    `${result.city}: ${result.temperature}${result.unit === 'celsius' ? 'C' : 'F'}, ${result.conditions}`,
});

interface Weather {
  city: string;
  unit: 'celsius' | 'fahrenheit';
  temperature: number;
  conditions: string;
}

// Shared agent state, available to every tool through `context.state`.
type State = { readonly queryLog: string[] };

/**
 * Tools that touch the outside world should declare `requiresApproval` and be
 * screened by a policy. Both run before `execute`.
 */
const sendEmail = defineTool<
  { to: string; subject: string; body: string },
  { sent: boolean },
  State
>({
  name: 'send_email',
  description: 'Send an email. Requires explicit approval.',
  requiresApproval: true,
  parameters: z.object({ to: z.string().email(), subject: z.string(), body: z.string() }),
  execute: ({ to }, context) => {
    context.state.queryLog.push(`email:${to}`);
    return { sent: true };
  },
});

/**
 * Policies run first and can deny a call outright. Return
 * `{ action: 'allow' }` (or `undefined`) to continue.
 */
const policy: ToolPolicy<State> = (call) => {
  const to = (call.arguments as { to?: string } | undefined)?.to ?? '';
  if (call.name === 'send_email' && to.endsWith('@spam.test')) {
    return { action: 'deny', reason: 'refusing to email known spam domains' };
  }
  return { action: 'allow' };
};

const agent = new Agent<State>({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions:
    'You are a research assistant with access to a calculator, the current time, ' +
    'a sandboxed workspace, HTTP, weather, and email. Use tools rather than guessing.',
  tools: [
    getWeather,
    sendEmail,
    createCalculatorTool(),
    createDateTimeTool(),
    createHttpTool({ maxResponseChars: 4_000 }),
    // Sandboxed to one directory: `../` and symlink escapes are rejected.
    createFileSystemTool({ root: process.cwd(), allowWrite: false }),
  ],
  state: { queryLog: [] },
  toolPolicy: policy,
  approval: async (request) => {
    console.warn(`\n[approval] ${request.toolName}(${JSON.stringify(request.args)})`);
    // A real app would prompt a human, or check a permission store.
    return true;
  },
  execution: 'parallel',
  toolConcurrency: 4,
  maxToolResultLength: 8_000,
  hooks: {
    onToolStart: ({ toolName }) => console.log(`→ ${toolName}`),
    onToolEnd: ({ toolName, durationMs }) =>
      console.log(`← ${toolName} (${durationMs}ms)`),
    onToolError: ({ toolName, error }) => console.warn(`✗ ${toolName}: ${error.message}`),
  },
});

try {
  const result = await agent.run(
    'What is 18% of 250, what time is it in Europe/Berlin, and is it raining in Oslo?',
  );
  console.log(`\n${result.output}`);
} catch (error) {
  // Denials surface as regular tool errors so the model can adapt, but they are
  // typed, so you can also react to them here.
  if (error instanceof ToolDeniedError) console.error(`denied: ${error.message}`);
  else throw error;
}

console.log(`\ntool calls: ${agent.state.queryLog.length}`);
