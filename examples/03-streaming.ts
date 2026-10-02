/**
 * 03 — streaming, events, and cancellation.
 *
 * Run: `npx tsx examples/03-streaming.ts`
 */
import { Agent } from '../src/index.js';

const agent = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: 'You are a patient teacher. Keep answers under 200 words.',
});

const controller = new AbortController();
// Abort mid-answer: the run resolves with `stopReason: 'aborted'` when you pass
// `throwOnError: false`, or rejects with `AbortError` by default.
const stopAt = Number(process.env['STOP_AFTER_MS'] ?? Number.POSITIVE_INFINITY);
const timer = setTimeout(() => {
  console.error('\n[aborting]');
  controller.abort();
}, stopAt);

const run = agent.stream('Write a short poem about retries in distributed systems.', {
  signal: controller.signal,
});

for await (const event of run) {
  switch (event.type) {
    case 'text-delta':
      process.stdout.write(event.text);
      break;
    case 'reasoning-delta':
      process.stdout.write(`[thinking] ${event.text}`);
      break;
    case 'tool-call':
      console.log(`\n[tool] ${event.name}(${JSON.stringify(event.args)})`);
      break;
    case 'tool-result':
      console.log(`[tool] ${event.name} -> ${event.content.slice(0, 80)}`);
      break;
    case 'retry':
      console.log(`\n[retry] attempt failed, waiting ${event.delayMs}ms`);
      break;
    case 'iteration':
      console.log(
        `\n[iteration ${event.iteration}] tokens=${event.usage.totalTokens ?? 0} ${event.durationMs}ms`,
      );
      break;
    case 'error':
      console.error(`\n[error] ${event.error.message} (fatal=${event.fatal})`);
      break;
    default:
      break;
  }
}

clearTimeout(timer);

const result = await run.result;
console.log(`\n\nstopReason=${result.stopReason} durationMs=${result.durationMs}`);

/**
 * The same object is also a typed event emitter, and `text()` collects the whole
 * answer for you — useful when you do not care about the individual events.
 */
// `text()` consumes the queue, so start it before awaiting the result.
const second = agent.stream('Name three prime numbers.');
const answer = second.text();
const secondResult = await second.result;

console.log(`\ncollected: ${await answer}`);
console.log(`iterations: ${secondResult.iterations}`);
