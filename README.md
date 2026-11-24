# site-generator-speed-01-v6

[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/oxizbinance-aox/site-generator-speed-01-v6/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/oxizbinance-aox/site-generator-speed-01-v6/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

A native C11 engine for the [site-generator-speed-01](https://make.fun/bounty/site-generator-speed-01-run-20261122) bounty (make.fun).

## What it is

A from-scratch reimplementation of the `makefun-site` static site generator
(the reference program) in C11, compiled to a single statically linked
binary (`sitegen`). It produces byte-identical output to the reference on
every practice case, at a fraction of the CPU time.

## Layout

- `main.ts` — the harness driver. It hands the whole case set to `./sitegen`
  in one call, so no JSON is parsed or stringified in Node at all.
- `solve.js` — `run(input)`, the single-case equivalent through the same
  binary (the driver contract).
- `sitegen.c` — the engine (C11, no dependencies beyond libc). One arena
  per case, zero-copy in-place JSON unescaping, AVX2 scans for the hot
  paths (HTML escaping, word counting, JSON output escaping), and a
  handwritten slugifier that matches Node's NFKD + `\p{M}` stripping for
  the whole BMP. v5 adds: template-include caching (per-template
  dependency analysis — globally cacheable fillers and per-object
  includes), a memoised Markdown link resolver, direct-to-page inline
  rendering (no per-paragraph scratch buffer), 64-byte JSON escape
  chunks, and a once-per-source formatted date. v6 adds: a single-probe
  link memo (hash computed once, precomputed per-source directory hash,
  scheme/absolute/fragment targets bypass the memo), a cached per-tag
  slug in the tag dedup (no repeated re-slugify), and an inline small-copy
  path (≤32 bytes, no `memcpy` call) in the string-buffer appends — plus
  a `-O3 -march=x86-64-v3 -static` build.
- `slug-tables.h`, `mark-ranges.h` — Unicode tables generated from Node's
  own NFKD and `\p{M}` (Latin Extended-B / Additional base letters, the
  mixed multi-character expansions, and the full Mark range list).
- `sitegen` — the compiled engine (gcc `-O3 -march=x86-64-v3 -static`).
- `NOTICE` — upstream credits.

## How it was built

The engine is a direct port of the reference `makefun-site` program (the
bounty kit's `program/generator/vendor/makefun-site/`): the same
front-matter subset, the same line-based Markdown dialect, the same
template dialect (`{{ }}`, `{{{ }}}`, `{% for %}`, `{% if %}`, `{% include %}`
with the standalone-tag line rule), the same slug rule, and the same build
pipeline (read sources → front matter → permalinks → Markdown →
collections → tags → layouts → report).

Verified byte-identical against the kit's practice expected answers (3/3
cases), against the reference oracle on 90 freshly generated cases (30
seeds), and slug-for-slug against Node on all 65,364 BMP code points plus
2,000 random combinations. (The slug checks are inherited from v5; the v6
engine changes no slug, word-count, link-resolution, or escaping
semantics — only their cost.)

Build (gcc 13+):

```sh
gcc -O3 -march=x86-64-v3 -std=c11 -static -o sitegen sitegen.c -lm
```

Run (as the harness does):

```sh
node main.ts
```

## Bounty

- Competition: `site-generator-speed-01-run-20261122`
- Entry: native C11 binary + thin Node driver
- Local CPU (this VM, 3 practice cases, interleaved A/B vs the shipped
  v5 binary, medians over repeated runs): v6 is ~5% faster than v5
  (v5 scored 90,000 µs on the server, rank 52). Server estimate
  ~85,000 µs. A step toward the top-50 prize line; the remaining gap
  is the emit/markdown/render core, which is already near its
  memory-bandwidth floor for 2.9 MB in → 7.4 MB out.
