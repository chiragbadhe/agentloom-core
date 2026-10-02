/**
 * 04 — structured output validated with Zod.
 *
 * Run: `npx tsx examples/04-structured-output.ts`
 */
import { z } from 'zod';

import { Agent, ValidationError } from '../src/index.js';

// Any validator works: it needs a `safeParse` returning
// `{ success: true, data } | { success: false, error: { issues } }`.
// A hand-written JSON Schema works too, via `schemaFromJsonSchema`.
const Invoice = z.object({
  vendor: z.string(),
  invoiceNumber: z.string().regex(/^[A-Z]{2,}-\d{4}$/, 'expected FORMAT-1234'),
  issuedOn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'expected YYYY-MM-DD'),
  total: z.number().positive(),
  currency: z.enum(['USD', 'EUR', 'GBP']),
  lineItems: z
    .array(
      z.object({
        description: z.string(),
        quantity: z.number().int().positive(),
        unitPrice: z.number().nonnegative(),
      }),
    )
    .min(1),
  needsReview: z.boolean(),
});

const agent = new Agent({
  model: process.env['MODEL'] ?? 'openai:gpt-4o-mini',
  instructions: 'You extract invoice data from unstructured text.',
  limits: {
    // Attempts allowed when the model's JSON does not validate. The agent feeds
    // the validation errors back and asks for a correction.
    maxOutputAttempts: 3,
  },
});

const result = await agent.run<z.infer<typeof Invoice>>(
  [
    'Acme Corp billed us on 2024-03-01 for 12 widgets at 4.50 EUR each and 3 gizmos at 2.00 EUR each.',
    'Invoice number is IN-7781. Everything reconciles, so no human review is needed.',
  ].join('\n'),
  { outputSchema: Invoice },
);

// `result.data` is fully typed; `result.output` is the raw JSON text.
if (result.data === undefined) {
  console.error(`no structured output (stopReason=${result.stopReason})`);
} else {
  console.log(`${result.data.vendor} ${result.data.invoiceNumber}: ${result.data.total}`);
  console.log(`currency=${result.data.currency} review=${result.data.needsReview}`);
}

// Failures are typed, and carry the issues that caused them.
try {
  await agent.run('Invoice from Globex, number ???, dated sometime in March.', {
    outputSchema: Invoice,
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(`\nvalidation failed (${error.issues.length} issues):`);
    for (const issue of error.issues)
      console.error(`  ${issue.path.join('.')}: ${issue.message}`);
  } else {
    throw error;
  }
}
