// solve.js: the entry's run(input), the single-case side of the driver
// contract. It answers one site alone through the same native engine the
// driver uses for the whole set: the input is written to a temporary file,
// the engine builds it, and the output tree is read back. Each call is
// independent — nothing is kept from one case to the next.
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const engine = fileURLToPath(new URL('./sitegen', import.meta.url));

export function run(input) {
  const dir = mkdtempSync(join(tmpdir(), 'sitegen-v6-'));
  try {
    const inFile = join(dir, 'case.json');
    const outFile = join(dir, 'answer.json');
    // The engine takes a cases file (an array of {id, input}) and writes
    // an answers array, so a lone case goes in wrapped with its id.
    writeFileSync(inFile, JSON.stringify([{ id: 'case', input }]));
    const done = spawnSync(engine, [inFile, outFile], { stdio: 'inherit' });
    if (done.error) throw done.error;
    if (done.status !== 0) throw new Error(`sitegen exited ${done.status}`);
    return JSON.parse(readFileSync(outFile, 'utf8'))[0];
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
