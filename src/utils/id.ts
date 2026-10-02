let counter = 0;

/**
 * Short, sortable, collision-resistant identifier.
 *
 * Format: `<prefix>_<base36 ms><base36 counter><base36 random>`, which keeps
 * trace ids roughly ordered by creation time — handy when eyeballing logs.
 */
export function createId(prefix = 'ag'): string {
  counter = (counter + 1) % 1_679_616;
  const time = Date.now().toString(36);
  const seq = counter.toString(36).padStart(4, '0');
  const random = Math.floor(Math.random() * 1_679_616)
    .toString(36)
    .padStart(4, '0');
  return `${prefix}_${time}${seq}${random}`;
}
