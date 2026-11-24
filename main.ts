// site-generator-speed-01-v6: harness driver.
//
// The grader runs `node /bot/main.ts` (Node 22). The driver reads the whole
// case set from /input/cases.json and writes /output/answers.json, one
// answer per case in the cases' order. All of the work is done by the
// native engine in ./sitegen (a statically linked C11 binary): the driver
// hands it the input and output paths in a single spawn, so Node itself
// never parses or stringifies any case JSON. The binary's CPU time is the
// entry's CPU time; the driver adds one process spawn.
//
// Directories come from MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR when set
// (local runs outside a container), else /input and /output.
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const inDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const engine = fileURLToPath(new URL('./sitegen', import.meta.url));

// A fresh clone can lose the executable bit on the binary; restore it so
// the spawn cannot fail with EACCES. A no-op when the bit is already set.
try {
  chmodSync(engine, 0o755);
} catch {
  // The bit was already set, or the file is read-only in the sandbox.
}

mkdirSync(outDir, { recursive: true });

const done = spawnSync(
  engine,
  [join(inDir, 'cases.json'), join(outDir, 'answers.json')],
  { stdio: 'inherit' },
);
if (done.error) throw done.error;
if (done.status !== 0) process.exit(done.status ?? 1);

// A zero exit with no answers file would silently score zero cases, so
// confirm the engine actually wrote one before reporting success.
const answersPath = join(outDir, 'answers.json');
if (!existsSync(answersPath) || statSync(answersPath).size === 0) {
  console.error('sitegen exited 0 but wrote no answers.json');
  process.exit(1);
}
