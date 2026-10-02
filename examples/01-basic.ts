/**
 * 01 — the smallest useful agent.
 *
 * Run: `npx tsx examples/01-basic.ts` (or set MODEL to pick a different provider)
 *
 * A model reference is `"provider:model"`. The API key is read from the
 * provider's conventional environment variable, so nothing else is needed.
 */
import { Agent } from '../src/index.js';

const model = process.env['MODEL'] ?? 'openai:gpt-4o-mini';

const agent = new Agent({
  name: 'assistant',
  model,
  instructions:
    'You are a concise assistant. Answer in at most three sentences unless asked for detail.',
  // Guardrails for anything unattended. `run()` also throws on fatal errors.
  limits: { maxIterations: 5, timeoutMs: 60_000, maxTotalTokens: 8_000 },
});

const result = await agent.run('Explain what a tool-calling loop is, in two sentences.');

console.log(result.output);
console.log(
  `\nstopReason=${result.stopReason} iterations=${result.iterations} ` +
    `tokens=${result.usage.totalTokens ?? 'n/a'} in ${result.durationMs}ms`,
);
