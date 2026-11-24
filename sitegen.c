// sitegen.c — a fast, exact reimplementation of makefun-site 1.0.0.
//
// Reads a cases file (JSON array of {id, input}), builds each site exactly
// as the reference does, and writes the answers array as JSON. Single
// static binary, no libc startup cost beyond the minimum: one arena per
// case, zero-copy JSON string handling, SIMD scans where the bytes are.
//
// Design notes (from the kit's README + reference code, re-derived here):
//  * input is { site, templates, files }; output is path -> text per case.
//  * every string the JSON parser yields points into the input buffer with
//    escapes decoded in place (write pointer never passes the read pointer).
//  * the build pipeline mirrors the reference: sort sources, front matter,
//    permalinks, Markdown, collections, tags, layouts, report.
//  * Markdown and templates are implemented from the dialect spec; nothing
//    is shared between cases (arena reset per case).

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <immintrin.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;

static double now_s(void); // defined near main (CPU time)
static double T_tpl, T_src, T_perm, T_md, T_coll, T_tags, T_lay, T_rep;
static double T_md_render, T_words;

// ---------------------------------------------------------------- slices

typedef struct { const char *s; u32 n; } Str;

static int str_eq(Str a, const char *b) {
  size_t bl = strlen(b);
  return a.n == bl && memcmp(a.s, b, bl) == 0;
}
static int str_eq2(Str a, Str b) { return a.n == b.n && memcmp(a.s, b.s, a.n) == 0; }
static int str_cmp(Str a, Str b) {
  u32 m = a.n < b.n ? a.n : b.n;
  int c = m ? memcmp(a.s, b.s, m) : 0;
  if (c) return c;
  return a.n < b.n ? -1 : a.n > b.n ? 1 : 0;
}
static Str str_lit(const char *b) { Str r = { b, (u32)strlen(b) }; return r; }

// ---------------------------------------------------------------- arena

static char *arena_base;
static size_t arena_cap = (size_t)256 << 20, arena_pos;

static void arena_init(void) {
  arena_base = (char *)malloc(arena_cap);
  if (!arena_base) { perror("malloc arena"); exit(1); }
}
static void *acalloc(size_t n) {
  // Bump allocator: does NOT zero. Callers that need zeroed memory use
  // azalloc, or memset explicitly. (Buf/MBuf contents are overwritten
  // before being read, so they use the raw bump path.)
  n = (n + 7) & ~(size_t)7;
  if (arena_pos + n > arena_cap) { fprintf(stderr, "arena exhausted\n"); exit(1); }
  void *p = arena_base + arena_pos;
  arena_pos += n;
  return p;
}
static void *azalloc(size_t n) {
  void *p = acalloc(n);
  memset(p, 0, n);
  return p;
}
static char *adup(const char *s, u32 n) {
  char *p = (char *)acalloc(n + 1);
  memcpy(p, s, n);
  p[n] = 0;
  return p;
}

// ---------------------------------------------------------------- string buffer (arena-backed, grows by doubling)

typedef struct { char *buf; size_t len, cap; } Buf;

__attribute__((always_inline)) static inline void buf_reserve(Buf *b, size_t extra) {
  if (b->len + extra <= b->cap) return;
  size_t nc = b->cap ? b->cap : 256;
  while (nc < b->len + extra) nc *= 2;
  char *nb = (char *)acalloc(nc);
  if (b->len) memcpy(nb, b->buf, b->len);
  b->buf = nb;
  b->cap = nc;
}
// Copy n bytes (0 <= n <= 32) with overlapping vector/word stores —
// no memcpy call, so the many small appends in md/lay stay cheap.
// SSE2 is baseline x86-64, so this is valid in any build.
static inline void copy_small(char *w, const char *s, size_t n) {
  if (n >= 16) {
    __m128i a = _mm_loadu_si128((const __m128i *)s);
    __m128i b = _mm_loadu_si128((const __m128i *)(s + n - 16));
    _mm_storeu_si128((__m128i *)w, a);
    _mm_storeu_si128((__m128i *)(w + n - 16), b);
  } else if (n >= 8) {
    u64 a, b;
    memcpy(&a, s, 8); memcpy(&b, s + n - 8, 8);
    memcpy(w, &a, 8); memcpy(w + n - 8, &b, 8);
  } else if (n >= 4) {
    u32 a, b;
    memcpy(&a, s, 4); memcpy(&b, s + n - 4, 4);
    memcpy(w, &a, 4); memcpy(w + n - 4, &b, 4);
  } else if (n == 0) {
    return;
  } else {
    // 1..3 bytes.
    w[0] = s[0];
    if (n >= 2) w[n - 1] = s[n - 1];
    if (n == 3) w[1] = s[1];
  }
}

__attribute__((always_inline)) static inline void buf_add(Buf *b, const void *p, size_t n) {
  buf_reserve(b, n);
  if (n <= 32) copy_small(b->buf + b->len, (const char *)p, n);
  else memcpy(b->buf + b->len, p, n);
  b->len += n;
}
__attribute__((always_inline)) static inline void buf_addc(Buf *b, char c) { buf_reserve(b, 1); b->buf[b->len++] = c; }
__attribute__((always_inline)) static inline void buf_adds(Buf *b, Str s) { buf_add(b, s.s, s.n); }
__attribute__((always_inline)) static inline void buf_addz(Buf *b, const char *z) { buf_add(b, z, strlen(z)); }

// malloc-backed buffer (answers.json, outlives the per-case arena)
typedef struct { char *buf; size_t len, cap; } MBuf;

__attribute__((always_inline)) static inline void mbuf_reserve(MBuf *b, size_t extra) {
  if (b->len + extra <= b->cap) return;
  size_t nc = b->cap ? b->cap : (size_t)1 << 20;
  while (nc < b->len + extra) nc *= 2;
  b->buf = (char *)realloc(b->buf, nc);
  if (!b->buf) { perror("realloc"); exit(1); }
  b->cap = nc;
}
__attribute__((always_inline)) static inline void mbuf_add(MBuf *b, const void *p, size_t n) {
  mbuf_reserve(b, n);
  if (n <= 32) copy_small(b->buf + b->len, (const char *)p, n);
  else memcpy(b->buf + b->len, p, n);
  b->len += n;
}

// ---------------------------------------------------------------- CPU dispatch

static int cpu_avx2;

static void cpu_init(void) {
  cpu_avx2 = __builtin_cpu_supports("avx2") != 0;
}

// ---------------------------------------------------------------- UTF-8

// Decode one code point at *p (bytes available: n). Returns the code point
// and stores its byte length in *adv. Invalid bytes yield U+FFFD, adv 1.
static u32 utf8_decode(const u8 *p, size_t n, size_t *adv) {
  u8 c = p[0];
  if (c < 0x80) { *adv = 1; return c; }
  if (c >= 0xC2 && c < 0xE0 && n >= 2 && (p[1] & 0xC0) == 0x80) {
    *adv = 2; return ((u32)(c & 0x1F) << 6) | (p[1] & 0x3F);
  }
  if (c >= 0xE0 && c < 0xF0 && n >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
    u32 cp = ((u32)(c & 0x0F) << 12) | ((u32)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) { *adv = 3; return cp; }
  }
  if (c >= 0xF0 && c < 0xF5 && n >= 4 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
    u32 cp = ((u32)(c & 0x07) << 18) | ((u32)(p[1] & 0x3F) << 12) | ((u32)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    if (cp >= 0x10000 && cp <= 0x10FFFF) { *adv = 4; return cp; }
  }
  *adv = 1;
  return 0xFFFD;
}

// Encode cp as UTF-8 into out (must hold 4 bytes); returns length.
static size_t utf8_encode(u32 cp, char *out) {
  if (cp < 0x80) { out[0] = (char)cp; return 1; }
  if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2;
  }
  if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
  }
  out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

// Whether cp is whitespace in JavaScript's \s sense.
static int cp_is_space(u32 cp) {
  if (cp == ' ' || (cp >= '\t' && cp <= '\r')) return 1;
  return cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
         cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
}

// Whether cp is a combining mark (Unicode \p{M} for the ranges that can
// appear after NFKD of the generator's alphabet; the full \p{M} set is
// much larger, but marks only matter inside slugify, where any mark is
// dropped — so treat every mark-range code point as a mark).
#include "mark-ranges.h" // cp_is_mark: binary search over Node \p{M} ranges

// ---------------------------------------------------------------- slugify

// NFKD with combining marks stripped, for the Latin ranges, as a base
// letter or 0 (gap). '.' in the tables below means "no decomposition:
// the character itself is dropped as a gap".
static const char LATIN1_BASE[64] =
    "AAAAAA.CEEEEIIII.NOOOOO..UUUUY..aaaaaa.ceeeeiiii.nooooo..uuuuy.y";
static const char EXTA_BASE[128] =
    "AaAaAaCcCcCcCcDd..EeEeEeEeEeGgGgGgGgHh..IiIiIiIiI.**JjKk.LlLlLl....NnNnNn...OoOoOo..RrRrRrSsSsSsSsTtTt..UuUuUuUuUuUuWwYyYZzZzZzs";

#include "slug-tables.h" // EXTB_BASE, EXTADD_BASE, SLUG_MIXED (from Node NFKD)

// Binary search SLUG_MIXED for cp; copies the expansion into out and
// returns its length, or 0 when cp has no mixed expansion.
static int slug_mixed(u32 cp, char *out) {
  u32 lo = 0, hi = SLUG_MIXED_N;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2;
    if (SLUG_MIXED[mid].cp < cp) lo = mid + 1;
    else if (SLUG_MIXED[mid].cp > cp) hi = mid;
    else {
      const char *e = SLUG_MIXED[mid].exp;
      int n = 0;
      while (e[n]) out[n] = e[n], n++;
      return n;
    }
  }
  return 0;
}

// Expand one non-ASCII code point into the slug stream. Writes 0-6 ASCII
// bytes into out (a space encodes a gap inside the expansion), returns the
// count; returns -1 when the character is a gap on its own (dropped, sets
// the gap flag), -2 when it is a mark (dropped, sets nothing).
static int slug_expand(u32 cp, char *out) {
  if (cp_is_mark(cp)) return -2;
  // The only two non-marks whose NFKD is purely marks: they vanish
  // silently (like a mark), they are not a gap.
  if (cp == 0xFF9E || cp == 0xFF9F) return -2;
  // Explicit multi-char / gap-aware expansions first: they must win over
  // the range tables below (the digraphs sit inside the EXTB range, whose
  // table holds '.' for them).
  switch (cp) {
    case 0xBC: out[0] = '1'; out[1] = ' '; out[2] = '4'; return 3; // 1⁄4
    case 0xBD: out[0] = '1'; out[1] = ' '; out[2] = '2'; return 3; // 1⁄2
    case 0xBE: out[0] = '3'; out[1] = ' '; out[2] = '4'; return 3; // 3⁄4
    case 0x13F: out[0] = 'L'; out[1] = ' '; return 2; // L + middle dot
    case 0x140: out[0] = 'l'; out[1] = ' '; return 2; // l + middle dot
    case 0x149: out[0] = ' '; out[1] = 'n'; return 2; // ' + n
    case 0x1E9A: out[0] = 'a'; out[1] = ' '; return 2; // a + rough breathing
    case 0xAA: out[0] = 'a'; return 1;
    case 0xBA: out[0] = 'o'; return 1;
    case 0xB9: out[0] = '1'; return 1;
    case 0xB2: out[0] = '2'; return 1;
    case 0xB3: out[0] = '3'; return 1;
    case 0x1C4: out[0] = 'D'; out[1] = 'Z'; return 2;
    case 0x1C5: out[0] = 'D'; out[1] = 'z'; return 2;
    case 0x1C6: out[0] = 'd'; out[1] = 'z'; return 2;
    case 0x1C7: out[0] = 'L'; out[1] = 'J'; return 2;
    case 0x1C8: out[0] = 'L'; out[1] = 'j'; return 2;
    case 0x1C9: out[0] = 'l'; out[1] = 'j'; return 2;
    case 0x1CA: out[0] = 'N'; out[1] = 'J'; return 2;
    case 0x1CB: out[0] = 'N'; out[1] = 'j'; return 2;
    case 0x1CC: out[0] = 'n'; out[1] = 'j'; return 2;
    case 0x1F1: out[0] = 'D'; out[1] = 'Z'; return 2;
    case 0x1F2: out[0] = 'D'; out[1] = 'z'; return 2;
    case 0x1F3: out[0] = 'd'; out[1] = 'z'; return 2;
    case 0xFB00: out[0] = 'f'; out[1] = 'f'; return 2;
    case 0xFB01: out[0] = 'f'; out[1] = 'i'; return 2;
    case 0xFB02: out[0] = 'f'; out[1] = 'l'; return 2;
    case 0xFB03: out[0] = 'f'; out[1] = 'f'; out[2] = 'i'; return 3;
    case 0xFB04: out[0] = 'f'; out[1] = 'f'; out[2] = 'l'; return 3;
    case 0xFB05: out[0] = 's'; out[1] = 't'; return 2;
    case 0xFB06: out[0] = 's'; out[1] = 't'; return 2;
    default: break;
  }
  if (cp >= 0xC0 && cp <= 0xFF) {
    char b = LATIN1_BASE[cp - 0xC0];
    if (b == '.') return -1;
    out[0] = b;
    return 1;
  }
  if (cp >= 0x100 && cp <= 0x17F) {
    char b = EXTA_BASE[cp - 0x100];
    if (b == '.') return -1;
    if (b == '*') { // IJ / ij
      out[0] = (cp == 0x132) ? 'I' : 'i';
      out[1] = (cp == 0x132) ? 'J' : 'j';
      return 2;
    }
    out[0] = b;
    return 1;
  }
  if (cp >= 0x180 && cp <= 0x24F) { // Latin Extended-B
    char b = EXTB_BASE[cp - 0x180];
    if (b == '.') return -1;
    out[0] = b;
    return 1;
  }
  if (cp >= 0x1E00 && cp <= 0x1EFF) { // Latin Extended Additional
    char b = EXTADD_BASE[cp - 0x1E00];
    if (b == '.') return -1;
    out[0] = b;
    return 1;
  }
  if (cp >= 0xFF01 && cp <= 0xFF5E) { // fullwidth ASCII
    out[0] = (char)(cp - 0xFEE0);
    return 1;
  }
  { // mixed / multi-char NFKD expansions (Roman numerals, circled
    // numbers, CJK compatibility, fractions, letterforms, ...)
    int n = slug_mixed(cp, out);
    if (n) return n;
  }
  return -1;
}

// The slug of text: NFKD, marks dropped, A-Z lowered, each run of other
// characters one '-', none at either end, cut to 64 (a '-' left at the cut
// dropped), 'untitled' when nothing is left. Writes into out (72 bytes);
// returns length.
static u32 slugify(Str text, char *out) {
  u32 olen = 0;
  int gap = 0;
  size_t i = 0;
  const u8 *p = (const u8 *)text.s;
  while (i < text.n) {
    u8 c = p[i];
    if (c < 0x80) {
      i++;
      char keep = 0;
      if (c >= 'A' && c <= 'Z') keep = (char)(c + 32);
      else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (char)c;
      if (!keep) { gap = olen > 0; continue; }
      if (gap && olen < 64) out[olen++] = '-';
      gap = 0;
      if (olen < 64) out[olen++] = keep;
      continue;
    }
    size_t adv;
    u32 cp = utf8_decode(p + i, text.n - i, &adv);
    i += adv;
    char tmp[8]; // max mixed expansion is 6 bytes
    int r = slug_expand(cp, tmp);
    if (r == -2) continue; // mark: dropped silently
    if (r < 0) { gap = olen > 0; continue; } // gap character
    for (int k = 0; k < r; k++) {
      char keep = tmp[k];
      if (keep >= 'A' && keep <= 'Z') keep = (char)(keep + 32);
      if (!((keep >= 'a' && keep <= 'z') || (keep >= '0' && keep <= '9'))) {
        gap = olen > 0;
        continue;
      }
      if (gap && olen < 64) out[olen++] = '-';
      gap = 0;
      if (olen < 64) out[olen++] = keep;
    }
  }
  while (olen > 0 && out[olen - 1] == '-') olen--;
  if (olen == 0) {
    memcpy(out, "untitled", 8);
    return 8;
  }
  return olen;
}

// ---------------------------------------------------------------- small hash set of strings (open addressing)

typedef struct {
  Str *keys;
  u32 *vals; // optional parallel values (or NULL)
  u32 cap, count;
} StrSet;

static u64 hash_bytes(const void *p, size_t n) {
  // Chunked FNV-1a: 8 bytes per multiply instead of 1 (same distribution
  // shape, ~5x faster on 30-40 byte keys). The tail packs the remaining
  // bytes so short keys still hash fully; a finalizer spreads the bits
  // for the power-of-two table mask.
  const u8 *b = (const u8 *)p;
  u64 h = 1469598103934665603ULL ^ (u64)n;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    u64 w;
    memcpy(&w, b + i, 8);
    h ^= w;
    h *= 1099511628211ULL;
  }
  if (i < n) {
    u64 tail = 0;
    for (size_t k = i; k < n; k++) tail |= (u64)b[k] << (8 * (k - i));
    h ^= tail;
    h *= 1099511628211ULL;
  }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return h;
}

static void strset_init(StrSet *s, u32 cap) {
  s->cap = cap;
  s->count = 0;
  s->keys = (Str *)acalloc(sizeof(Str) * cap);
  s->vals = NULL;
  for (u32 i = 0; i < cap; i++) s->keys[i].s = NULL;
}
static void strset_init_vals(StrSet *s, u32 cap) {
  strset_init(s, cap);
  s->vals = (u32 *)acalloc(sizeof(u32) * cap);
}
// Returns 1 if the key was already present.
static int strset_add(StrSet *s, Str k, u32 v) {
  u64 h = hash_bytes(k.s, k.n);
  u32 i = (u32)h & (s->cap - 1);
  while (s->keys[i].s) {
    if (str_eq2(s->keys[i], k)) return 1;
    i = (i + 1) & (s->cap - 1);
  }
  s->keys[i] = k;
  if (s->vals) s->vals[i] = v;
  s->count++;
  return 0;
}
static int strset_has(StrSet *s, Str k, u32 *out_v) {
  u64 h = hash_bytes(k.s, k.n);
  u32 i = (u32)h & (s->cap - 1);
  while (s->keys[i].s) {
    if (str_eq2(s->keys[i], k)) {
      if (out_v && s->vals) *out_v = s->vals[i];
      return 1;
    }
    i = (i + 1) & (s->cap - 1);
  }
  return 0;
}

// ---------------------------------------------------------------- HTML escaping

static const char *ESC_TBL[256];
static void esc_init(void) {
  static char tmp[8][8];
  for (int i = 0; i < 256; i++) ESC_TBL[i] = NULL;
  memcpy(tmp[0], "&amp;", 5); ESC_TBL['&'] = tmp[0];
  memcpy(tmp[1], "&lt;", 4); ESC_TBL['<'] = tmp[1];
  memcpy(tmp[2], "&gt;", 4); ESC_TBL['>'] = tmp[2];
  memcpy(tmp[3], "&quot;", 6); ESC_TBL['"'] = tmp[3];
  memcpy(tmp[4], "&#39;", 5); ESC_TBL['\''] = tmp[4];
}

// Append s with & < > " ' escaped. Fast path: bulk-copy runs between
// specials, found with an AVX2 scan when available.
static void buf_escape(Buf *b, const char *s, size_t n);

static void buf_escape_scalar(Buf *b, const char *s, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    while (j < n) {
      unsigned char c = (unsigned char)s[j];
      if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
      j++;
    }
    if (j > i) buf_add(b, s + i, j - i);
    if (j >= n) break;
    const char *e = ESC_TBL[(unsigned char)s[j]];
    buf_add(b, e, strlen(e));
    i = j + 1;
  }
}

__attribute__((target("avx2"))) static size_t esc_scan_avx2(const u8 *p, size_t n) {
  // Index of the first byte in {& < > " '} or n.
  const __m256i amp = _mm256_set1_epi8('&');
  const __m256i lt = _mm256_set1_epi8('<');
  const __m256i gt = _mm256_set1_epi8('>');
  const __m256i qu = _mm256_set1_epi8('"');
  const __m256i ap = _mm256_set1_epi8('\'');
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(p + i));
    __m256i m = _mm256_cmpeq_epi8(v, amp);
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, lt));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, gt));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, qu));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, ap));
    u32 mask = (u32)_mm256_movemask_epi8(m);
    if (mask) return i + __builtin_ctz(mask);
  }
  return i; // caller finishes the tail scalar
}

__attribute__((target_clones("avx2", "default")))
static void buf_escape(Buf *b, const char *s, size_t n) {
  if (!cpu_avx2) { buf_escape_scalar(b, s, n); return; }
  size_t i = 0;
  while (i < n) {
    size_t j = esc_scan_avx2((const u8 *)s + i, n - i) + i;
    // esc_scan_avx2 returns the offset of the first special, or the
    // position where the AVX2 loop stopped (tail); finish tail scalar.
    size_t k = j;
    while (k < n) {
      unsigned char c = (unsigned char)s[k];
      if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
      k++;
    }
    if (k > i) buf_add(b, s + i, k - i);
    if (k >= n) break;
    const char *e = ESC_TBL[(unsigned char)s[k]];
    buf_add(b, e, strlen(e));
    i = k + 1;
  }
}

// Count the words of text: runs of characters other than whitespace.
static u32 words_of(const char *s, size_t n);

__attribute__((target("avx2"))) static u32 words_of_avx2(const u8 *p, size_t n) {
  // Whitespace bytes: 0x09-0x0D and 0x20. A word starts where a non-ws
  // byte follows a ws byte (or the start).
  const __m256i nine = _mm256_set1_epi8(9);
  const __m256i five = _mm256_set1_epi8(5);
  const __m256i minus1 = _mm256_set1_epi8(-1);
  const __m256i sp = _mm256_set1_epi8(' ');
  const __m256i hi = _mm256_set1_epi8(-1); // sign test for >= 0x80
  u32 words = 0;
  int prev_ws = 1;
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(p + i));
    // high bytes present? fall back to scalar for this chunk.
    u32 himask = (u32)_mm256_movemask_epi8(_mm256_cmpgt_epi8(hi, v));
    if (himask) {
      // scalar chunk
      for (size_t k = 0; k < 32; k++) {
        unsigned char c = p[i + k];
        int ws;
        if (c < 0x80) ws = (c == ' ' || (c >= '\t' && c <= '\r'));
        else {
          size_t adv;
          u32 cp = utf8_decode(p + i + k, n - i - k, &adv);
          ws = cp_is_space(cp);
          // consume the whole code point in the scalar loop
          if (!ws && prev_ws) words++;
          prev_ws = ws;
          k += adv - 1;
          continue;
        }
        if (!ws && prev_ws) words++;
        prev_ws = ws;
      }
      continue;
    }
    __m256i t = _mm256_sub_epi8(v, nine);
    __m256i range = _mm256_and_si256(_mm256_cmpgt_epi8(five, t), _mm256_cmpgt_epi8(t, minus1));
    __m256i wsm = _mm256_or_si256(range, _mm256_cmpeq_epi8(v, sp));
    u32 w = (u32)_mm256_movemask_epi8(wsm);
    u32 starts = (~w) & ((w << 1) | (prev_ws ? 1u : 0u));
    words += __builtin_popcount(starts);
    prev_ws = (w >> 31) & 1;
  }
  for (; i < n; i++) {
    unsigned char c = p[i];
    int ws;
    if (c < 0x80) ws = (c == ' ' || (c >= '\t' && c <= '\r'));
    else {
      size_t adv;
      u32 cp = utf8_decode(p + i, n - i, &adv);
      ws = cp_is_space(cp);
      if (!ws && prev_ws) words++;
      prev_ws = ws;
      i += adv - 1;
      continue;
    }
    if (!ws && prev_ws) words++;
    prev_ws = ws;
  }
  return words;
}

__attribute__((target_clones("avx2", "default")))
static u32 words_of(const char *s, size_t n) {
  if (cpu_avx2) return words_of_avx2((const u8 *)s, n);
  u32 words = 0;
  int prev_ws = 1;
  size_t i = 0;
  while (i < n) {
    unsigned char c = (unsigned char)s[i];
    int ws;
    if (c < 0x80) { ws = (c == ' ' || (c >= '\t' && c <= '\r')); i++; }
    else {
      size_t adv;
      u32 cp = utf8_decode((const u8 *)s + i, n - i, &adv);
      ws = cp_is_space(cp);
      i += adv;
    }
    if (!ws && prev_ws) words++;
    prev_ws = ws;
  }
  return words;
}

// ---------------------------------------------------------------- JSON parsing
//
// A small pull parser over the input buffer. Strings are decoded in place:
// the write pointer never passes the read pointer, so a string's decoded
// bytes end up at the start of its own span in the buffer.

typedef struct { const char *p, *end; } JParser;

static void jerr(JParser *j, const char *msg) {
  fprintf(stderr, "json error: %s at offset %ld\n", msg, (long)(j->p - (const char *)0));
  exit(1);
}
static void jws(JParser *j) {
  while (j->p < j->end) {
    char c = *j->p;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
    else break;
  }
}
static void jexpect(JParser *j, char c) {
  jws(j);
  if (j->p >= j->end || *j->p != c) jerr(j, "unexpected character");
  j->p++;
}

// Index of the first '"' or '\\' at or after p (within n bytes), or n.
static size_t json_special_scalar(const u8 *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (p[i] == '"' || p[i] == '\\') return i;
  return n;
}
__attribute__((target("avx2"))) static size_t json_special_avx2(const u8 *p, size_t n) {
  const __m256i qu = _mm256_set1_epi8('"');
  const __m256i bs = _mm256_set1_epi8('\\');
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(p + i));
    __m256i m = _mm256_or_si256(_mm256_cmpeq_epi8(v, qu), _mm256_cmpeq_epi8(v, bs));
    u32 mask = (u32)_mm256_movemask_epi8(m);
    if (mask) return i + __builtin_ctz(mask);
  }
  return i;
}
static size_t json_special(const u8 *p, size_t n) {
  if (cpu_avx2) {
    size_t i = json_special_avx2(p, n);
    // i is the first special, or the tail start: scan the tail scalar.
    return json_special_scalar(p + i, n - i) + i;
  }
  return json_special_scalar(p, n);
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parse a JSON string starting at the opening quote. Decodes escapes in
// place and returns the decoded slice (into the input buffer).
__attribute__((target_clones("avx2", "default")))
static Str jstring(JParser *j) {
  jws(j);
  if (j->p >= j->end || *j->p != '"') jerr(j, "expected string");
  j->p++;
  char *w = (char *)j->p; // write head (<= read head)
  const char *start = w;
  for (;;) {
    size_t avail = (size_t)(j->end - j->p);
    size_t hit = json_special((const u8 *)j->p, avail);
    if (hit > 0) {
      // While no escape has been decoded in this string the write head
      // is exactly the read head, so the copy is a no-op — skip it.
      if (w != j->p) memmove(w, j->p, hit);
      w += hit;
      j->p += hit;
    }
    if (j->p >= j->end) jerr(j, "unterminated string");
    char c = *j->p;
    if (c == '"') { j->p++; break; }
    // c == '\\': an escape.
    j->p++;
    if (j->p >= j->end) jerr(j, "unterminated escape");
    char e = *j->p++;
    switch (e) {
      case '"': *w++ = '"'; break;
      case '\\': *w++ = '\\'; break;
      case '/': *w++ = '/'; break;
      case 'b': *w++ = '\b'; break;
      case 'f': *w++ = '\f'; break;
      case 'n': *w++ = '\n'; break;
      case 'r': *w++ = '\r'; break;
      case 't': *w++ = '\t'; break;
      case 'u': {
        if (j->end - j->p < 4) jerr(j, "bad \\u escape");
        u32 cp = 0;
        for (int k = 0; k < 4; k++) {
          int h = hexval(j->p[k]);
          if (h < 0) jerr(j, "bad \\u escape");
          cp = (cp << 4) | (u32)h;
        }
        j->p += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
          u32 lo = 0;
          int ok = 1;
          for (int k = 0; k < 4; k++) {
            int h = hexval(j->p[2 + k]);
            if (h < 0) { ok = 0; break; }
            lo = (lo << 4) | (u32)h;
          }
          if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            j->p += 6;
          }
        }
        w += utf8_encode(cp, w);
        break;
      }
      default: jerr(j, "bad escape");
    }
  }
  Str r = { start, (u32)(w - start) };
  return r;
}

// Skip a JSON string (no decode). AVX2 scan for the next '"' or '\\',
// same as the decoder's scan — the old byte-at-a-time loop cost ~1ms.
static void jskip_string(JParser *j) {
  jws(j);
  if (j->p >= j->end || *j->p != '"') jerr(j, "expected string");
  j->p++;
  for (;;) {
    size_t avail = (size_t)(j->end - j->p);
    size_t hit = json_special((const u8 *)j->p, avail);
    if (hit >= avail) jerr(j, "unterminated string");
    j->p += hit;
    if (*j->p == '"') { j->p++; return; }
    // Backslash: skip it and the escaped char (a \" is one pair).
    if (j->end - j->p < 2) jerr(j, "unterminated string");
    j->p += 2;
  }
}

static void jskip_value(JParser *j) {
  jws(j);
  if (j->p >= j->end) jerr(j, "unexpected end");
  char c = *j->p;
  if (c == '"') { jskip_string(j); return; }
  if (c == '{') {
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == '}') { j->p++; return; }
    for (;;) {
      jskip_string(j);
      jexpect(j, ':');
      jskip_value(j);
      jws(j);
      if (j->p < j->end && *j->p == ',') { j->p++; continue; }
      jexpect(j, '}');
      return;
    }
  }
  if (c == '[') {
    j->p++;
    jws(j);
    if (j->p < j->end && *j->p == ']') { j->p++; return; }
    for (;;) {
      jskip_value(j);
      jws(j);
      if (j->p < j->end && *j->p == ',') { j->p++; continue; }
      jexpect(j, ']');
      return;
    }
  }
  // literal or number: consume to the next structural character.
  while (j->p < j->end) {
    char d = *j->p;
    if (d == ',' || d == '}' || d == ']' || d == ' ' || d == '\t' || d == '\n' || d == '\r') break;
    j->p++;
  }
}

static long jnumber(JParser *j) {
  jws(j);
  const char *s = j->p;
  if (j->p < j->end && (*j->p == '-' || *j->p == '+')) j->p++;
  int isint = 1;
  while (j->p < j->end) {
    char c = *j->p;
    if (c >= '0' && c <= '9') { j->p++; continue; }
    if (c == '.' || c == 'e' || c == 'E') isint = 0;
    break;
  }
  if (!isint) {
    // Fall back to strtod (not expected for this input).
    char *endp;
    double d = strtod(s, &endp);
    j->p = endp;
    return (long)d;
  }
  return strtol(s, NULL, 10);
}

// ---------------------------------------------------------------- case input

typedef struct { Str path, content; } InputFile;
typedef struct { Str name, src; } Template;

typedef struct {
  Str title, base_url;
  long per_page, recent;
  Template *templates;
  u32 ntemplates, cap_templates;
  InputFile *files;
  u32 nfiles, cap_files;
} CaseInput;

static void case_add_template(CaseInput *c, Str name, Str src) {
  if (c->ntemplates == c->cap_templates) {
    u32 nc = c->cap_templates ? c->cap_templates * 2 : 32;
    Template *nt = (Template *)acalloc(sizeof(Template) * nc);
    if (c->ntemplates) memcpy(nt, c->templates, sizeof(Template) * c->ntemplates);
    c->templates = nt;
    c->cap_templates = nc;
  }
  c->templates[c->ntemplates].name = name;
  c->templates[c->ntemplates].src = src;
  c->ntemplates++;
}
static void case_add_file(CaseInput *c, Str path, Str content) {
  if (c->nfiles == c->cap_files) {
    u32 nc = c->cap_files ? c->cap_files * 2 : 64;
    InputFile *nf = (InputFile *)acalloc(sizeof(InputFile) * nc);
    if (c->nfiles) memcpy(nf, c->files, sizeof(InputFile) * c->nfiles);
    c->files = nf;
    c->cap_files = nc;
  }
  c->files[c->nfiles].path = path;
  c->files[c->nfiles].content = content;
  c->nfiles++;
}

// Parse {"site": {...}, "templates": {...}, "files": [...]} at the cursor
// (cursor on '{').
static void parse_input_obj(JParser *j, CaseInput *c) {
  memset(c, 0, sizeof(*c));
  jexpect(j, '{');
  jws(j);
  if (j->p < j->end && *j->p == '}') { j->p++; return; }
  for (;;) {
    Str key = jstring(j);
    jexpect(j, ':');
    if (str_eq(key, "site")) {
      jexpect(j, '{');
      jws(j);
      if (j->p < j->end && *j->p == '}') { j->p++; }
      else {
        for (;;) {
          Str sk = jstring(j);
          jexpect(j, ':');
          if (str_eq(sk, "title")) c->title = jstring(j);
          else if (str_eq(sk, "base_url")) c->base_url = jstring(j);
          else if (str_eq(sk, "per_page")) c->per_page = jnumber(j);
          else if (str_eq(sk, "recent")) c->recent = jnumber(j);
          else jskip_value(j);
          jws(j);
          if (j->p < j->end && *j->p == ',') { j->p++; continue; }
          jexpect(j, '}');
          break;
        }
      }
    } else if (str_eq(key, "templates")) {
      jexpect(j, '{');
      jws(j);
      if (j->p < j->end && *j->p == '}') { j->p++; }
      else {
        for (;;) {
          Str name = jstring(j);
          jexpect(j, ':');
          Str src = jstring(j);
          case_add_template(c, name, src);
          jws(j);
          if (j->p < j->end && *j->p == ',') { j->p++; continue; }
          jexpect(j, '}');
          break;
        }
      }
    } else if (str_eq(key, "files")) {
      jexpect(j, '[');
      jws(j);
      if (j->p < j->end && *j->p == ']') { j->p++; }
      else {
        for (;;) {
          jexpect(j, '{');
          Str path = { 0, 0 }, content = { 0, 0 };
          jws(j);
          if (j->p < j->end && *j->p == '}') { j->p++; }
          else {
            for (;;) {
              Str fk = jstring(j);
              jexpect(j, ':');
              if (str_eq(fk, "path")) path = jstring(j);
              else if (str_eq(fk, "content")) content = jstring(j);
              else jskip_value(j);
              jws(j);
              if (j->p < j->end && *j->p == ',') { j->p++; continue; }
              jexpect(j, '}');
              break;
            }
          }
          case_add_file(c, path, content);
          jws(j);
          if (j->p < j->end && *j->p == ',') { j->p++; continue; }
          jexpect(j, ']');
          break;
        }
      }
    } else {
      jskip_value(j);
    }
    jws(j);
    if (j->p < j->end && *j->p == ',') { j->p++; continue; }
    jexpect(j, '}');
    return;
  }
}

// Parse the whole cases file. Returns the cases (arena) and their count.
static CaseInput *parse_cases(const char *buf, size_t len, u32 *out_n) {
  JParser j = { buf, buf + len };
  jexpect(&j, '[');
  u32 cap = 4, n = 0;
  CaseInput *cases = (CaseInput *)acalloc(sizeof(CaseInput) * cap);
  jws(&j);
  if (j.p < j.end && *j.p == ']') { j.p++; *out_n = 0; return cases; }
  for (;;) {
    jexpect(&j, '{');
    CaseInput c;
    memset(&c, 0, sizeof(c));
    jws(&j);
    if (j.p < j.end && *j.p == '}') { j.p++; }
    else {
      for (;;) {
        Str key = jstring(&j);
        jexpect(&j, ':');
        if (str_eq(key, "input")) parse_input_obj(&j, &c);
        else jskip_value(&j);
        jws(&j);
        if (j.p < j.end && *j.p == ',') { j.p++; continue; }
        jexpect(&j, '}');
        break;
      }
    }
    if (n == cap) {
      u32 nc = cap * 2;
      CaseInput *nn = (CaseInput *)acalloc(sizeof(CaseInput) * nc);
      memcpy(nn, cases, sizeof(CaseInput) * n);
      cases = nn;
      cap = nc;
    }
    cases[n++] = c;
    jws(&j);
    if (j.p < j.end && *j.p == ',') { j.p++; continue; }
    jexpect(&j, ']');
    break;
  }
  *out_n = n;
  return cases;
}

// ---------------------------------------------------------------- front matter

static int days_in(int y, int m) {
  if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
  return (m == 4 || m == 6 || m == 9 || m == 11) ? 30 : 31;
}

typedef struct {
  int ok;
  char error[160];
  Str title, date, summary, layout, slug;
  int has_date, has_summary, has_layout, has_slug;
  Str *tags;
  u32 ntags;
  int draft, has_draft;
  Str body;
} FrontMatter;

// Next line of text at *pos (lines end at '\n', not included). Returns 0
// when there is no line left (pos >= len).
static int next_line(const char *s, u32 n, u32 *pos, Str *line) {
  if (*pos >= n) return 0;
  u32 start = *pos;
  u32 i = start;
  while (i < n && s[i] != '\n') i++;
  line->s = s + start;
  line->n = i - start;
  *pos = (i < n) ? i + 1 : i;
  return 1;
}

static int fm_is_blank(Str l) {
  for (u32 i = 0; i < l.n; i++)
    if (l.s[i] != ' ' && l.s[i] != '\t') return 0;
  return 1;
}

// stringOf: decode a title/summary value. Returns 1 ok, 0 error (err set).
static int fm_string_of(Str raw, Str *out, char *err) {
  if (raw.n > 0 && raw.s[0] == '"') {
    if (raw.n < 2 || raw.s[raw.n - 1] != '"') {
      strcpy(err, "a quoted string is not closed");
      return 0;
    }
    char *buf = (char *)acalloc(raw.n + 1);
    u32 w = 0;
    for (u32 i = 1; i + 1 < raw.n; i++) {
      char c = raw.s[i];
      if (c == '\\') {
        char nx = raw.s[i + 1];
        if ((nx != '"' && nx != '\\') || i + 1 == raw.n - 1) {
          strcpy(err, "a quoted string holds an escape other than \\\" or \\\\");
          return 0;
        }
        buf[w++] = nx;
        i++;
      } else if (c == '"') {
        strcpy(err, "a quoted string holds a stray quote");
        return 0;
      } else buf[w++] = c;
    }
    out->s = buf;
    out->n = w;
    return 1;
  }
  if (raw.n > 0 && raw.s[0] == '[') {
    strcpy(err, "a bare string opens on [");
    return 0;
  }
  *out = raw;
  return 1;
}

static void fm_fail(FrontMatter *fm, const char *msg) {
  fm->ok = 0;
  snprintf(fm->error, sizeof(fm->error), "%s", msg);
}

// Parse front matter of text (line ends already '\n'). Fills fm.
static void parse_front_matter(Str text, FrontMatter *fm) {
  memset(fm, 0, sizeof(*fm));
  fm->ok = 1;
  u32 pos = 0;
  Str line;
  if (!next_line(text.s, text.n, &pos, &line) || !str_eq(line, "---")) {
    fm_fail(fm, "no front matter");
    return;
  }
  // Find the closing line; record where the body starts.
  u32 body_off = text.n;
  int close = -1;
  u32 scan = pos;
  Str l;
  int lineno = 1; // line number of the line just read (opening --- is 1)
  while (next_line(text.s, text.n, &scan, &l)) {
    lineno++;
    if (str_eq(l, "---")) { close = lineno; body_off = scan; break; }
  }
  if (close == -1) {
    fm_fail(fm, "the front matter is not closed");
    return;
  }
  static const char *KEYS[7] = { "title", "date", "tags", "draft", "layout", "slug", "summary" };
  int seen[7] = { 0, 0, 0, 0, 0, 0, 0 };
  u32 p2 = pos;
  for (int i = 1; i < close - 1; i++) {
    next_line(text.s, text.n, &p2, &l);
    if (fm_is_blank(l) || (l.n > 0 && l.s[0] == '#')) continue;
    // LINE: /^([a-z_]+):(?: (.*))?$/
    u32 k = 0;
    while (k < l.n && ((l.s[k] >= 'a' && l.s[k] <= 'z') || l.s[k] == '_')) k++;
    if (k == 0 || k >= l.n || l.s[k] != ':') {
      char msg[80];
      snprintf(msg, sizeof(msg), "line %d is not \"key: value\"", i + 1);
      fm_fail(fm, msg);
      return;
    }
    Str key = { l.s, k };
    Str raw;
    if (k + 1 < l.n && l.s[k + 1] == ' ') { raw.s = l.s + k + 2; raw.n = l.n - (k + 2); }
    else { raw.s = l.s + l.n; raw.n = 0; }
    while (raw.n > 0 && (raw.s[raw.n - 1] == ' ' || raw.s[raw.n - 1] == '\t')) raw.n--;
    int ki = -1;
    for (int t = 0; t < 7; t++)
      if (str_eq(key, KEYS[t])) { ki = t; break; }
    if (ki == -1) {
      char msg[80];
      snprintf(msg, sizeof(msg), "unknown key \"%.*s\"", (int)key.n, key.s);
      fm_fail(fm, msg);
      return;
    }
    if (seen[ki]) {
      char msg[80];
      snprintf(msg, sizeof(msg), "key \"%s\" is given twice", KEYS[ki]);
      fm_fail(fm, msg);
      return;
    }
    seen[ki] = 1;
    if (raw.n == 0) {
      char msg[80];
      snprintf(msg, sizeof(msg), "key \"%s\" has no value", KEYS[ki]);
      fm_fail(fm, msg);
      return;
    }
    char err[120];
    switch (ki) {
      case 0: // title
      case 6: { // summary
        Str v;
        if (!fm_string_of(raw, &v, err)) {
          char msg[160];
          snprintf(msg, sizeof(msg), "key \"%s\": %s", KEYS[ki], err);
          fm_fail(fm, msg);
          return;
        }
        if (ki == 0) fm->title = v; else { fm->summary = v; fm->has_summary = 1; }
        break;
      }
      case 1: { // date
        int ok = 0;
        if (raw.n == 10 && raw.s[4] == '-' && raw.s[7] == '-') {
          int dig = 1;
          for (int d = 0; d < 10; d++)
            if (d != 4 && d != 7 && (raw.s[d] < '0' || raw.s[d] > '9')) dig = 0;
          if (dig) {
            int y = (raw.s[0] - '0') * 1000 + (raw.s[1] - '0') * 100 + (raw.s[2] - '0') * 10 + (raw.s[3] - '0');
            int mo = (raw.s[5] - '0') * 10 + (raw.s[6] - '0');
            int d = (raw.s[8] - '0') * 10 + (raw.s[9] - '0');
            if (y >= 1970 && y <= 2099 && mo >= 1 && mo <= 12 && d >= 1 && d <= days_in(y, mo)) ok = 1;
          }
        }
        if (!ok) {
          fm_fail(fm, "key \"date\": not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)");
          return;
        }
        fm->date = raw;
        fm->has_date = 1;
        break;
      }
      case 2: { // tags
        if (raw.n < 2 || raw.s[0] != '[' || raw.s[raw.n - 1] != ']') {
          fm_fail(fm, "key \"tags\": not a list [a, b]");
          return;
        }
        Str inner = { raw.s + 1, raw.n - 2 };
        Str t = inner;
        while (t.n > 0 && (t.s[0] == ' ' || t.s[0] == '\t')) { t.s++; t.n--; }
        while (t.n > 0 && (t.s[t.n - 1] == ' ' || t.s[t.n - 1] == '\t')) t.n--;
        if (t.n == 0) { fm->tags = NULL; fm->ntags = 0; break; }
        // Count commas for the array size.
        u32 cnt = 1;
        for (u32 q = 0; q < inner.n; q++) if (inner.s[q] == ',') cnt++;
        fm->tags = (Str *)acalloc(sizeof(Str) * cnt);
        u32 w = 0;
        u32 start = 0;
        for (u32 q = 0; q <= inner.n; q++) {
          if (q == inner.n || inner.s[q] == ',') {
            Str one = { inner.s + start, q - start };
            while (one.n > 0 && (one.s[0] == ' ' || one.s[0] == '\t')) { one.s++; one.n--; }
            while (one.n > 0 && (one.s[one.n - 1] == ' ' || one.s[one.n - 1] == '\t')) one.n--;
            int bad = one.n == 0;
            for (u32 z = 0; z < one.n && !bad; z++)
              if (one.s[z] == '"' || one.s[z] == '[' || one.s[z] == ']') bad = 1;
            if (bad) {
              fm_fail(fm, "key \"tags\": an empty tag, or one holding \" [ or ]");
              return;
            }
            fm->tags[w++] = one;
            start = q + 1;
          }
        }
        fm->ntags = w;
        break;
      }
      case 3: // draft
        if (str_eq(raw, "true")) fm->draft = 1;
        else if (str_eq(raw, "false")) fm->draft = 0;
        else { fm_fail(fm, "key \"draft\": true or false"); return; }
        fm->has_draft = 1;
        break;
      case 4: { // layout
        int ok = raw.n > 0;
        for (u32 q = 0; q < raw.n && ok; q++) {
          char c = raw.s[q];
          if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) ok = 0;
        }
        if (!ok) { fm_fail(fm, "key \"layout\": not a template name"); return; }
        fm->layout = raw;
        fm->has_layout = 1;
        break;
      }
      case 5: { // slug
        int ok = raw.n > 0 && raw.n <= 64;
        if (ok) {
          // /^[a-z0-9]+(?:-[a-z0-9]+)*$/
          int state = 0; // 0 need alnum, 1 have alnum (may take -)
          for (u32 q = 0; q < raw.n; q++) {
            char c = raw.s[q];
            int al = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
            if (state == 0) { if (!al) { ok = 0; break; } state = 1; }
            else if (c == '-') state = 0;
            else if (!al) { ok = 0; break; }
          }
          if (state == 0) ok = 0;
        }
        if (!ok) { fm_fail(fm, "key \"slug\": not a slug"); return; }
        fm->slug = raw;
        fm->has_slug = 1;
        break;
      }
    }
  }
  if (!seen[0]) {
    fm_fail(fm, "no title");
    return;
  }
  fm->body.s = text.s + body_off;
  fm->body.n = text.n - body_off;
}

static const char *MONTHS[12] = { "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December" };

// "2026-11-09" -> "9 November 2026". Writes into an arena buffer.
static Str long_date(Str date) {
  int y = 0, m = 0, d = 0;
  for (int i = 0; i < 4; i++) y = y * 10 + (date.s[i] - '0');
  m = (date.s[5] - '0') * 10 + (date.s[6] - '0');
  d = (date.s[8] - '0') * 10 + (date.s[9] - '0');
  (void)y;
  char *buf = (char *)acalloc(32);
  int n = snprintf(buf, 32, "%d %s %.*s", d, MONTHS[m - 1], 4, date.s);
  Str r = { buf, (u32)n };
  return r;
}

// ---------------------------------------------------------------- build structs (shared by markdown + pipeline)

typedef struct Tag Tag;
typedef struct Source Source;
typedef struct PageView PageView;
typedef struct { int level; Str id, html; } TocEntry;

struct Tag {
  Str name, slug, url;
  Source **posts;
  u32 nposts, cap_posts;
};

struct Source {
  Str path;
  Str text; // content with line ends made '\n'
  FrontMatter fm;
  int post;
  Str layout;
  Str slug, out_dir, url;
  Str html;
  TocEntry *toc;
  u32 ntoc;
  Str excerpt;
  int reading_minutes;
  Str date_long; // long_date(fm.date), computed once per source
  Source *newer, *older;
  Tag **tag_refs; // resolved tags (posts: real; pages: real or synthetic)
  u32 ntag_refs;
  Str *own_tag_names; // deduped tag names, first spelling kept
  u32 nown_tags;
  PageView *card; // card view of this post (posts only)
};

// ---------------------------------------------------------------- markdown

typedef struct {
  StrSet *by_source; // path -> source index
  Source *sources;
  StrSet *statics; // static paths
  Str base; // site base_url, trailing slashes dropped
  Str dir; // current source's directory
  u64 hdir; // hash_bytes(dir), precomputed per source for the link memo
  Str dir_segs[8]; // dir split into segments (pre-split per source)
  int ndir_segs;
  Str cur_path; // current source (for broken-link lines)
  StrSet *broken; // full "broken-link ..." lines, deduped
} MdEnv;

// resolvePath(dir, rel): segments of rel from dir ('.'/'..' steps, never
// above the root). Returns 1 and sets *out, or 0 when it climbs out.
// The dir's segments are pre-split in env (same dir for every link of a
// source), so only rel is split here. The result is written into the
// caller-provided buffer when it fits (the usual case — it is only used
// for hash lookups inside md_link); a heap copy is made only for very
// long paths.
static int resolve_path(MdEnv *env, Str rel, char *tmp, size_t tmpcap, Str *out) {
  Str segs[40];
  int nseg = 0;
  for (int i = 0; i < env->ndir_segs; i++) segs[nseg++] = env->dir_segs[i];
  u32 s = 0;
  for (u32 i = 0; i <= rel.n; i++) {
    if (i == rel.n || rel.s[i] == '/') {
      Str seg = { rel.s + s, i - s };
      s = i + 1;
      if (seg.n == 0 || str_eq(seg, ".")) continue;
      if (str_eq(seg, "..")) {
        if (nseg == 0) return 0;
        nseg--;
      } else {
        if (nseg >= 40) return 0;
        segs[nseg++] = seg;
      }
    }
  }
  u32 total = 0;
  for (int i = 0; i < nseg; i++) total += segs[i].n + 1;
  if (total) total--; // no leading slash
  char *buf = tmp;
  if (total + 1 > tmpcap) buf = (char *)acalloc(total + 1);
  u32 w = 0;
  for (int i = 0; i < nseg; i++) {
    if (i) buf[w++] = '/';
    memcpy(buf + w, segs[i].s, segs[i].n);
    w += segs[i].n;
  }
  buf[w] = 0;
  out->s = buf;
  out->n = w;
  return 1;
}

static int has_scheme(Str t) {
  if (t.n < 2) return 0;
  char c = t.s[0];
  if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) return 0;
  for (u32 i = 1; i < t.n; i++) {
    char d = t.s[i];
    if (d == ':') return 1;
    if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9') || d == '+' || d == '-' || d == '.') continue;
    return 0;
  }
  return 0;
}

// Per-case memo of md_link results: ~28% of link resolutions repeat
// the same (dir, target). The key is dir + NUL + target, stored once in
// the arena; the value is the resolved href (arena). A hit returns the
// cached href with no re-resolution and no re-reporting of broken links
// (the first computation already reported it).
typedef struct { Str *keys; Str *vals; u8 *flags; u32 cap, count; } LinkMemo;
static LinkMemo link_memo;

static void link_memo_init(u32 cap) {
  link_memo.cap = cap;
  link_memo.count = 0;
  link_memo.keys = (Str *)acalloc(sizeof(Str) * cap);
  link_memo.vals = (Str *)acalloc(sizeof(Str) * cap);
  link_memo.flags = (u8 *)acalloc(sizeof(u8) * cap);
  for (u32 i = 0; i < cap; i++) link_memo.keys[i].s = NULL;
}
static int link_memo_key_eq(Str k, Str dir, Str target) {
  u32 want = dir.n + 1 + target.n;
  return k.n == want && memcmp(k.s, dir.s, dir.n) == 0 && k.s[dir.n] == 0 &&
         memcmp(k.s + dir.n + 1, target.s, target.n) == 0;
}
static u64 link_memo_hash2(u64 hdir, u64 ht) {
  u64 h = hdir;
  h ^= ht + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}
// Single-probe lookup: returns 1 on hit (fills *out, *was_broken),
// else 0 and *slot is the empty slot to insert into (no re-hash,
// no re-probe on a miss).
static int link_memo_probe(u64 h, Str dir, Str target, Str *out, int *was_broken, u32 *slot) {
  u32 i = (u32)h & (link_memo.cap - 1);
  while (link_memo.keys[i].s) {
    if (link_memo_key_eq(link_memo.keys[i], dir, target)) {
      *out = link_memo.vals[i];
      *was_broken = link_memo.flags[i];
      return 1;
    }
    i = (i + 1) & (link_memo.cap - 1);
  }
  *slot = i;
  return 0;
}
static void link_memo_insert(u32 slot, Str dir, Str target, Str val, int was_broken) {
  // Never fill past half: a full open-addressing table would loop
  // forever on a miss. Past half load we simply stop memoizing (a miss
  // just recomputes — always correct).
  if (link_memo.count * 2 >= link_memo.cap) return;
  u32 want = dir.n + 1 + target.n;
  char *kb = (char *)acalloc(want + 1);
  memcpy(kb, dir.s, dir.n);
  kb[dir.n] = 0;
  memcpy(kb + dir.n + 1, target.s, target.n);
  link_memo.keys[slot].s = kb;
  link_memo.keys[slot].n = want;
  // The value must be owned: a scheme/broken result IS the target slice,
  // which points into the markdown scratch buffer that the next paragraph
  // reuses. Copying it into the arena keeps the cached value stable.
  link_memo.vals[slot].s = adup(val.s, val.n);
  link_memo.vals[slot].n = val.n;
  link_memo.flags[slot] = (u8)was_broken;
  link_memo.count++;
}

// Report a broken link once per (source, target): the report line names
// the source file, so the same target broken from another source is a
// different line. The set dedups identical lines.
static void report_broken_link(MdEnv *env, Str target) {
  const char *pre = "broken-link ";
  const char *mid = " -> ";
  u32 n = (u32)strlen(pre) + env->cur_path.n + (u32)strlen(mid) + target.n;
  char *buf = (char *)acalloc(n + 1);
  u32 w = 0;
  memcpy(buf + w, pre, strlen(pre)); w += (u32)strlen(pre);
  memcpy(buf + w, env->cur_path.s, env->cur_path.n); w += env->cur_path.n;
  memcpy(buf + w, mid, strlen(mid)); w += (u32)strlen(mid);
  memcpy(buf + w, target.s, target.n); w += target.n;
  strset_add(env->broken, (Str){ buf, n }, 0);
}

// A link's href, per the reference env.link.
static long CNT_md_inline, CNT_md_link, CNT_bracket;

static Str md_link_slow(MdEnv *env, Str target, int *was_broken);

static Str md_link(MdEnv *env, Str target) {
  CNT_md_link++;
  // Scheme, absolute and fragment-only targets resolve to themselves
  // (see md_link_slow); the result is the target and it is never a
  // broken link, so no memo lookup or insert is needed.
  if (has_scheme(target) || (target.n > 0 && target.s[0] == '/') || (target.n > 0 && target.s[0] == '#'))
    return target;
  u64 ht = hash_bytes(target.s, target.n);
  u64 h = link_memo_hash2(env->hdir, ht);
  Str cached;
  int was_broken = 0;
  u32 slot;
  if (link_memo_probe(h, env->dir, target, &cached, &was_broken, &slot)) {
    // A memo hit for a broken link still has to report it for THIS
    // source: the report line names the source file.
    if (was_broken) report_broken_link(env, target);
    return cached;
  }
  Str r = md_link_slow(env, target, &was_broken);
  link_memo_insert(slot, env->dir, target, r, was_broken);
  return r;
}

// The uncached resolution (see md_link).
static Str md_link_slow(MdEnv *env, Str target, int *was_broken) {
  *was_broken = 0;
  if (has_scheme(target) || (target.n > 0 && target.s[0] == '/') || (target.n > 0 && target.s[0] == '#'))
    return target;
  u32 hash = 0;
  int found = 0;
  for (u32 i = 0; i < target.n; i++)
    if (target.s[i] == '#') { hash = i; found = 1; break; }
  Str rel = { target.s, found ? hash : target.n };
  Str frag = found ? (Str){ target.s + hash, target.n - hash } : (Str){ target.s, 0 };
  Str path;
  int have_path = 0;
  char pbuf[512];
  if (rel.n != 0) have_path = resolve_path(env, rel, pbuf, sizeof(pbuf), &path);
  if (have_path) {
    u32 idx = 0;
    if (strset_has(env->by_source, path, &idx)) {
      Str url = env->sources[idx].url;
      char *buf = (char *)acalloc(url.n + frag.n + 1);
      memcpy(buf, url.s, url.n);
      memcpy(buf + url.n, frag.s, frag.n);
      Str r = { buf, url.n + frag.n };
      return r;
    }
    if (strset_has(env->statics, path, NULL)) {
      char *buf = (char *)acalloc(env->base.n + 1 + path.n + frag.n + 1);
      memcpy(buf, env->base.s, env->base.n);
      buf[env->base.n] = '/';
      memcpy(buf + env->base.n + 1, path.s, path.n);
      memcpy(buf + env->base.n + 1 + path.n, frag.s, frag.n);
      Str r = { buf, env->base.n + 1 + path.n + frag.n };
      return r;
    }
  }
  // Broken link: report once, keep the target as written.
  *was_broken = 1;
  report_broken_link(env, target);
  return target;
}

// O(1) table (the old linear scan over "\\`*_[]()#+-.!>" cost 16 compares
// per byte inside find_unescaped / find_single_star).
static const u8 ESCAPABLE_TBL[256] = {
  ['\\'] = 1, ['`'] = 1, ['*'] = 1, ['_'] = 1, ['['] = 1, [']'] = 1,
  ['('] = 1, [')'] = 1, ['#'] = 1, ['+'] = 1, ['-'] = 1, ['.'] = 1,
  ['!'] = 1, ['>'] = 1,
};

static int is_escapable(char c) {
  return ESCAPABLE_TBL[(unsigned char)c];
}

// First index >= from where s holds `what` outside a backslash escape.
static long find_unescaped(const char *s, size_t n, const char *what, size_t wlen, size_t from) {
  for (size_t j = from; j + wlen <= n; j++) {
    if (s[j] == '\\' && j + 1 < n && is_escapable(s[j + 1])) { j++; continue; }
    if (memcmp(s + j, what, wlen) == 0) return (long)j;
  }
  return -1;
}

// First single '*' from `from` (one whose run is not **), outside an escape.
static long find_single_star(const char *s, size_t n, size_t from) {
  for (size_t j = from; j < n; j++) {
    char c = s[j];
    if (c == '\\' && j + 1 < n && is_escapable(s[j + 1])) { j++; continue; }
    if (c != '*') continue;
    if (j + 1 < n && s[j + 1] == '*') { j++; continue; }
    return (long)j;
  }
  return -1;
}

static int target_has_space(Str t) {
  size_t i = 0;
  while (i < t.n) {
    unsigned char c = (unsigned char)t.s[i];
    if (c < 0x80) {
      if (c == ' ' || (c >= '\t' && c <= '\r')) return 1;
      i++;
    } else {
      size_t adv;
      u32 cp = utf8_decode((const u8 *)t.s + i, t.n - i, &adv);
      if (cp_is_space(cp)) return 1;
      i += adv;
    }
  }
  return 0;
}

// Index of the next inline special in {\ ` * [ & < > " '} from i, or n.
static size_t inline_special_scalar(const u8 *p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    u8 c = p[i];
    if (c == '\\' || c == '`' || c == '*' || c == '[' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') return i;
  }
  return n;
}
__attribute__((target("avx2"))) static size_t inline_special_avx2(const u8 *p, size_t n) {
  const __m256i b0 = _mm256_set1_epi8('\\');
  const __m256i b1 = _mm256_set1_epi8('`');
  const __m256i b2 = _mm256_set1_epi8('*');
  const __m256i b3 = _mm256_set1_epi8('[');
  const __m256i b4 = _mm256_set1_epi8('&');
  const __m256i b5 = _mm256_set1_epi8('<');
  const __m256i b6 = _mm256_set1_epi8('>');
  const __m256i b7 = _mm256_set1_epi8('"');
  const __m256i b8 = _mm256_set1_epi8('\'');
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)(p + i));
    __m256i m = _mm256_cmpeq_epi8(v, b0);
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b1));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b2));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b3));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b4));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b5));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b6));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b7));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, b8));
    u32 mask = (u32)_mm256_movemask_epi8(m);
    if (mask) return i + __builtin_ctz(mask);
  }
  return i;
}
static size_t inline_special(const u8 *p, size_t n) {
  if (cpu_avx2) {
    size_t i = inline_special_avx2(p, n);
    // i is the first special, or the tail start: scan the tail scalar.
    return inline_special_scalar(p + i, n - i) + i;
  }
  return inline_special_scalar(p, n);
}

static void md_inline(Buf *out, const char *s, size_t n, MdEnv *env);

__attribute__((target_clones("avx2", "default")))
static void md_inline(Buf *out, const char *s, size_t n, MdEnv *env) {
  CNT_md_inline++;
  size_t i = 0;
  while (i < n) {
    size_t hit = inline_special((const u8 *)s + i, n - i) + i;
    if (hit > i) { buf_add(out, s + i, hit - i); i = hit; }
    if (i >= n) break;
    char c = s[i];
    if (c == '\\' && i + 1 < n && is_escapable(s[i + 1])) {
      buf_escape(out, s + i + 1, 1);
      i += 2;
      continue;
    }
    if (c == '`') {
      const char *j = (const char *)memchr(s + i + 1, '`', n - i - 1);
      if (j && (size_t)(j - s) > i + 1) {
        buf_addz(out, "<code>");
        buf_escape(out, s + i + 1, (size_t)(j - s) - i - 1);
        buf_addz(out, "</code>");
        i = (size_t)(j - s) + 1;
      } else {
        buf_addc(out, '`');
        i++;
      }
      continue;
    }
    if (c == '*') {
      if (i + 1 < n && s[i + 1] == '*') {
        long j = find_unescaped(s, n, "**", 2, i + 2);
        if (j > (long)(i + 2)) {
          buf_addz(out, "<strong>");
          md_inline(out, s + i + 2, (size_t)j - i - 2, env);
          buf_addz(out, "</strong>");
          i = (size_t)j + 2;
        } else {
          buf_addz(out, "**");
          i += 2;
        }
        continue;
      }
      long j = find_single_star(s, n, i + 1);
      if (j > (long)(i + 1)) {
        buf_addz(out, "<em>");
        md_inline(out, s + i + 1, (size_t)j - i - 1, env);
        buf_addz(out, "</em>");
        i = (size_t)j + 1;
      } else {
        buf_addc(out, '*');
        i++;
      }
      continue;
    }
    if (c == '[') {
      CNT_bracket++;
      long k = find_unescaped(s, n, "]", 1, i + 1);
      long m = -1;
      if (k != -1 && (size_t)k + 1 < n && s[k + 1] == '(') {
        const char *r = (const char *)memchr(s + k + 2, ')', n - (size_t)k - 2);
        if (r) m = (long)(r - s);
      }
      Str target = { 0, 0 };
      if (m != -1) { target.s = s + k + 2; target.n = (u32)((size_t)m - (size_t)k - 2); }
      if (m != -1 && target.n != 0 && !target_has_space(target)) {
        Str href = md_link(env, target);
        buf_addz(out, "<a href=\"");
        buf_escape(out, href.s, href.n);
        buf_addz(out, "\">");
        md_inline(out, s + i + 1, (size_t)k - i - 1, env);
        buf_addz(out, "</a>");
        i = (size_t)m + 1;
        continue;
      }
      buf_addc(out, '[');
      i++;
      continue;
    }
    // One of & < > " ' : escape it.
    const char *e = ESC_TBL[(unsigned char)c];
    buf_add(out, e, strlen(e));
    i++;
  }
}

// ---- line classifiers (operate on one line, no '\n' inside)

static int line_blank(Str l) {
  for (u32 i = 0; i < l.n; i++)
    if (l.s[i] != ' ' && l.s[i] != '\t') return 0;
  return 1;
}
static int fence_open(Str l, Str *lang) {
  if (l.n < 3 || memcmp(l.s, "```", 3) != 0) return 0;
  u32 i = 3;
  while (i < l.n) {
    char c = l.s[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-') i++;
    else break;
  }
  for (u32 j = i; j < l.n; j++)
    if (l.s[j] != ' ' && l.s[j] != '\t') return 0;
  lang->s = l.s + 3;
  lang->n = i - 3;
  return 1;
}
static int fence_close(Str l) {
  if (l.n < 3 || memcmp(l.s, "```", 3) != 0) return 0;
  for (u32 i = 3; i < l.n; i++)
    if (l.s[i] != ' ' && l.s[i] != '\t') return 0;
  return 1;
}
// Heading: 1-6 '#', >=1 space/tab, non-empty text (trailing space/tab out).
static int heading_of(Str l, int *depth, Str *text) {
  u32 i = 0;
  while (i < l.n && l.s[i] == '#') i++;
  if (i < 1 || i > 6) return 0;
  if (i >= l.n || (l.s[i] != ' ' && l.s[i] != '\t')) return 0;
  u32 j = i;
  while (j < l.n && (l.s[j] == ' ' || l.s[j] == '\t')) j++;
  if (j >= l.n) return 0;
  u32 e = l.n;
  while (e > j && (l.s[e - 1] == ' ' || l.s[e - 1] == '\t')) e--;
  if (e == j) return 0;
  *depth = (int)i;
  text->s = l.s + j;
  text->n = e - j;
  return 1;
}
static int line_hr(Str l) {
  if (l.n < 3) return 0;
  if (memcmp(l.s, "---", 3) != 0 && memcmp(l.s, "***", 3) != 0) return 0;
  for (u32 i = 3; i < l.n; i++)
    if (l.s[i] != ' ' && l.s[i] != '\t') return 0;
  return 1;
}
static int ends_paragraph(Str l) {
  Str tmp;
  int d;
  return line_blank(l) || fence_open(l, &tmp) || heading_of(l, &d, &tmp) || line_hr(l);
}
static Str trim_line(Str l) {
  while (l.n > 0 && (l.s[0] == ' ' || l.s[0] == '\t')) { l.s++; l.n--; }
  while (l.n > 0 && (l.s[l.n - 1] == ' ' || l.s[l.n - 1] == '\t')) l.n--;
  return l;
}
// A list item's marker: returns 1 and fills *ordered, *start, *text.
static int list_marker(Str l, int *ordered, long *start, Str *text) {
  if (l.n >= 2 && (l.s[0] == '-' || l.s[0] == '*') && l.s[1] == ' ') {
    *ordered = 0;
    *start = 1;
    text->s = l.s + 2;
    text->n = l.n - 2;
    return 1;
  }
  u32 i = 0;
  while (i < l.n && l.s[i] >= '0' && l.s[i] <= '9') i++;
  if (i >= 1 && i <= 9 && i + 1 < l.n && l.s[i] == '.' && l.s[i + 1] == ' ') {
    long v = 0;
    for (u32 k = 0; k < i; k++) v = v * 10 + (l.s[k] - '0');
    *ordered = 1;
    *start = v;
    text->s = l.s + i + 2;
    text->n = l.n - (i + 2);
    return 1;
  }
  return 0;
}
// A continuation line: 2+ spaces then a non-space.
static int line_continued(Str l) {
  u32 i = 0;
  while (i < l.n && l.s[i] == ' ') i++;
  if (i < 2 || i >= l.n) return 0;
  unsigned char c = (unsigned char)l.s[i];
  if (c < 0x80) return !(c == ' ' || (c >= '\t' && c <= '\r'));
  size_t adv;
  u32 cp = utf8_decode((const u8 *)l.s + i, l.n - i, &adv);
  return !cp_is_space(cp);
}

// slug made unique against taken (added to it). Returns arena Str.
static Str unique_slug(Str base, StrSet *taken) {
  char buf[96];
  memcpy(buf, base.s, base.n);
  u32 bl = base.n;
  for (int n = 1;; n++) {
    u32 len;
    if (n == 1) len = bl;
    else len = (u32)snprintf(buf, sizeof(buf), "%.*s-%d", (int)bl, base.s, n);
    Str cand = { buf, len };
    if (!strset_has(taken, cand, NULL)) {
      char *copy = adup(buf, len);
      strset_add(taken, (Str){ copy, len }, 0);
      Str r = { copy, len };
      return r;
    }
  }
}

typedef struct {
  Str html;
  TocEntry *toc;
  u32 ntoc;
  Str excerpt;
} MdOut;

// Render a body to HTML (+ toc + excerpt), per the reference.
static void render_markdown(Str body, MdEnv *env, MdOut *out) {
  Buf html = { 0, 0, 0 };
  // Pre-size: rendered HTML runs ~1.5x the body; avoids all doubling
  // copies of this buffer (it is re-allocated per source).
  buf_reserve(&html, body.n + body.n / 2 + 512);
  TocEntry *toc = NULL;
  u32 ntoc = 0, captoc = 0;
  Str excerpt = { 0, 0 };
  int have_excerpt = 0;
  StrSet ids;
  strset_init(&ids, 64);
  Buf tmp = { 0, 0, 0 }; // scratch for one inline render at a time

  u32 pos = 0;
  Str line;
  while (next_line(body.s, body.n, &pos, &line)) {
    if (line_blank(line)) continue;
    Str lang;
    if (fence_open(line, &lang)) {
      buf_addz(&html, "<pre><code");
      if (lang.n > 0) {
        buf_addz(&html, " class=\"language-");
        buf_escape(&html, lang.s, lang.n);
        buf_addc(&html, '"');
      }
      buf_addc(&html, '>');
      Str bl;
      while (next_line(body.s, body.n, &pos, &bl)) {
        if (fence_close(bl)) break;
        buf_escape(&html, bl.s, bl.n);
        buf_addc(&html, '\n');
      }
      buf_addz(&html, "</code></pre>\n");
      continue;
    }
    int depth;
    Str htext;
    if (heading_of(line, &depth, &htext)) {
      char slugbuf[72];
      u32 sl = slugify(htext, slugbuf);
      Str id = unique_slug((Str){ slugbuf, sl }, &ids);
      char open[64];
      int on = snprintf(open, sizeof(open), "<h%d id=\"%.*s\">", depth, (int)id.n, id.s);
      buf_add(&html, open, (u32)on);
      // Render straight into the page HTML (no scratch copy): the TOC
      // entry takes its bytes from the span just written.
      size_t hstart = html.len;
      md_inline(&html, htext.s, htext.n, env);
      size_t hend = html.len;
      char close[16];
      int cn = snprintf(close, sizeof(close), "</h%d>\n", depth);
      buf_add(&html, close, (u32)cn);
      if (depth == 2 || depth == 3) {
        if (ntoc == captoc) {
          u32 nc = captoc ? captoc * 2 : 8;
          TocEntry *nt = (TocEntry *)acalloc(sizeof(TocEntry) * nc);
          if (ntoc) memcpy(nt, toc, sizeof(TocEntry) * ntoc);
          toc = nt;
          captoc = nc;
        }
        char *hcopy = adup(html.buf + hstart, (u32)(hend - hstart));
        toc[ntoc].level = depth;
        toc[ntoc].id = id;
        toc[ntoc].html.s = hcopy;
        toc[ntoc].html.n = (u32)(hend - hstart);
        ntoc++;
      }
      continue;
    }
    if (line_hr(line)) {
      buf_addz(&html, "<hr>\n");
      continue;
    }
    if (line.n > 0 && line.s[0] == '>') {
      // Collect quote lines: `line` first, then following '>' lines.
      Str qlines[64];
      int nq = 0;
      Str cur = line;
      for (;;) {
        Str t = cur;
        if (t.n > 0 && t.s[0] == '>') {
          t.s++;
          t.n--;
          if (t.n > 0 && t.s[0] == ' ') { t.s++; t.n--; }
          t = trim_line(t);
          if (t.n > 0 && nq < 64) qlines[nq++] = t;
        }
        u32 save = pos;
        if (!next_line(body.s, body.n, &pos, &cur)) break;
        if (!(cur.n > 0 && cur.s[0] == '>')) { pos = save; break; }
      }
      if (nq == 0) {
        buf_addz(&html, "<blockquote>\n</blockquote>\n");
      } else {
        tmp.len = 0;
        for (int q = 0; q < nq; q++) {
          if (q) buf_addc(&tmp, '\n');
          buf_add(&tmp, qlines[q].s, qlines[q].n);
        }
        buf_addz(&html, "<blockquote>\n<p>");
        md_inline(&html, tmp.buf, tmp.len, env);
        buf_addz(&html, "</p>\n</blockquote>\n");
      }
      continue;
    }
    int ordered;
    long start;
    Str itext;
    if (list_marker(line, &ordered, &start, &itext)) {
      int is_ord = ordered;
      long first_start = start;
      // Items: arrays of parts (trimmed), joined with '\n'.
      typedef struct { Str *parts; u32 nparts, cap; } Item;
      Item items[64];
      int nitems = 0;
      Str cur = line;
      u32 save_list = pos; // start of the line after `line`
      for (;;) {
        int o2;
        long s2;
        Str t2;
        if (list_marker(cur, &o2, &s2, &t2) && o2 == is_ord) {
          if (nitems >= 64) break;
          Item *it = &items[nitems++];
          it->parts = NULL;
          it->nparts = 0;
          it->cap = 0;
          Str first = trim_line(t2);
          it->parts = (Str *)acalloc(sizeof(Str) * 4);
          it->cap = 4;
          it->parts[it->nparts++] = first;
        } else if (line_continued(cur) && nitems > 0) {
          Item *it = &items[nitems - 1];
          if (it->nparts == it->cap) {
            u32 nc = it->cap * 2;
            Str *np = (Str *)acalloc(sizeof(Str) * nc);
            memcpy(np, it->parts, sizeof(Str) * it->nparts);
            it->parts = np;
            it->cap = nc;
          }
          it->parts[it->nparts++] = trim_line(cur);
        } else {
          pos = save_list;
          break;
        }
        save_list = pos;
        if (!next_line(body.s, body.n, &pos, &cur)) break;
      }
      if (is_ord && first_start != 1) {
        char open[48];
        int on = snprintf(open, sizeof(open), "<ol start=\"%ld\">\n", first_start);
        buf_add(&html, open, (u32)on);
      } else if (is_ord) {
        buf_addz(&html, "<ol>\n");
      } else {
        buf_addz(&html, "<ul>\n");
      }
      for (int k = 0; k < nitems; k++) {
        tmp.len = 0;
        for (u32 q = 0; q < items[k].nparts; q++) {
          if (q) buf_addc(&tmp, '\n');
          buf_add(&tmp, items[k].parts[q].s, items[k].parts[q].n);
        }
        buf_addz(&html, "<li>");
        md_inline(&html, tmp.buf, tmp.len, env);
        buf_addz(&html, "</li>\n");
      }
      buf_addz(&html, is_ord ? "</ol>\n" : "</ul>\n");
      continue;
    }
    // Paragraph: lines until ends_paragraph.
    tmp.len = 0;
    Str first = trim_line(line);
    buf_add(&tmp, first.s, first.n);
    Str cur;
    u32 save_para = pos;
    while (next_line(body.s, body.n, &pos, &cur)) {
      if (ends_paragraph(cur)) {
        pos = save_para;
        break;
      }
      buf_addc(&tmp, '\n');
      Str t = trim_line(cur);
      buf_add(&tmp, t.s, t.n);
      save_para = pos;
    }
    buf_addz(&html, "<p>");
    size_t pstart = html.len;
    md_inline(&html, tmp.buf, tmp.len, env);
    size_t pend = html.len;
    if (!have_excerpt) {
      char *ec = adup(html.buf + pstart, (u32)(pend - pstart));
      excerpt.s = ec;
      excerpt.n = (u32)(pend - pstart);
      have_excerpt = 1;
    }
    buf_addz(&html, "</p>\n");
  }
  out->html.s = html.buf;
  out->html.n = (u32)html.len;
  out->toc = toc;
  out->ntoc = ntoc;
  out->excerpt = excerpt;
}

// ---------------------------------------------------------------- templates

typedef enum { N_TEXT, N_VAR, N_RAW, N_FOR, N_IF, N_INCLUDE } NKind;

typedef struct Node Node;
typedef struct ParsedTemplate ParsedTemplate;
typedef struct { Node **items; u32 n, cap; } NodeList;

struct Node {
  NKind kind;
  Str text; // N_TEXT
  Str *path; u32 npath; // N_VAR/N_RAW/N_FOR/N_IF
  Str for_name; // N_FOR
  NodeList body; // N_FOR
  NodeList then_l, else_l; // N_IF
  int has_else;
  int not; // N_IF
  Str inc_name; // N_INCLUDE
  ParsedTemplate *inc_tpl; // resolved include target (NULL = missing)
};

static Node *new_node(NKind kind) {
  Node *nd = (Node *)azalloc(sizeof(Node));
  nd->kind = kind;
  return nd;
}

static void nodelist_push(NodeList *l, Node *nd) {
  if (l->n == l->cap) {
    u32 nc = l->cap ? l->cap * 2 : 8;
    Node **ni = (Node **)acalloc(sizeof(Node *) * nc);
    if (l->n) memcpy(ni, l->items, sizeof(Node *) * l->n);
    l->items = ni;
    l->cap = nc;
  }
  l->items[l->n++] = nd;
}

struct ParsedTemplate {
  Str name;
  NodeList nodes;
  int valid;
  char error[200];
  // Dependency analysis (for include caching; see analyze_template).
  int dep_done, dep_visiting, dep_loop, dep_bad, inc_depth;
  u32 dep_mask; // external first-segment names, as dep_names[] bits
};

// Validate a path string (after trimming) and split it. Returns 1 ok.
static int path_valid_split(Str t, Str **out_segs, u32 *out_n) {
  if (t.n == 0) return 0;
  // /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/
  u32 i = 0;
  u32 nseg = 1;
  for (;;) {
    char c = t.s[i];
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_')) return 0;
    i++;
    while (i < t.n) {
      c = t.s[i];
      if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') i++;
      else break;
    }
    if (i == t.n) break;
    if (t.s[i] != '.') return 0;
    i++;
    if (i >= t.n) return 0;
    nseg++;
  }
  Str *segs = (Str *)acalloc(sizeof(Str) * nseg);
  u32 w = 0, s = 0;
  for (u32 k = 0; k <= t.n; k++) {
    if (k == t.n || t.s[k] == '.') {
      segs[w].s = t.s + s;
      segs[w].n = k - s;
      w++;
      s = k + 1;
    }
  }
  *out_segs = segs;
  *out_n = nseg;
  return 1;
}

static Str trim_str(Str s) {
  while (s.n > 0 && (s.s[0] == ' ' || s.s[0] == '\t' || s.s[0] == '\n' || s.s[0] == '\r')) { s.s++; s.n--; }
  while (s.n > 0 && (s.s[s.n - 1] == ' ' || s.s[s.n - 1] == '\t' || s.s[s.n - 1] == '\n' || s.s[s.n - 1] == '\r')) s.n--;
  return s;
}

// Write s as a JSON string literal (with quotes) into b.
static void buf_json_quote(Buf *b, Str s) {
  buf_addc(b, '"');
  for (u32 i = 0; i < s.n; i++) {
    unsigned char c = (unsigned char)s.s[i];
    if (c == '"') buf_addz(b, "\\\"");
    else if (c == '\\') buf_addz(b, "\\\\");
    else if (c == '\n') buf_addz(b, "\\n");
    else if (c == '\r') buf_addz(b, "\\r");
    else if (c == '\t') buf_addz(b, "\\t");
    else if (c == '\b') buf_addz(b, "\\b");
    else if (c == '\f') buf_addz(b, "\\f");
    else if (c < 0x20) {
      char tmp[8];
      int n = snprintf(tmp, sizeof(tmp), "\\u%04x", c);
      buf_add(b, tmp, (u32)n);
    } else buf_addc(b, (char)c);
  }
  buf_addc(b, '"');
}

// Parse one template. On error, valid=0 and error holds the message.
static void parse_template(Str name, Str src, ParsedTemplate *out) {
  memset(out, 0, sizeof(*out));
  out->name = name;
  out->valid = 1;
  NodeList root = { 0, 0, 0 };
  Node *stack[512];
  int sp = 0;
  NodeList *cur = &root;
  size_t i = 0, text_start = 0;
  size_t n = src.n;
  const char *s = src.s;
  char ebuf[256];

  // Flush pending text [text_start, end) as a text node.
  // (Implemented inline at each tag via a macro-like block below.)
  while (i < n) {
    const char *open = (const char *)memchr(s + i, '{', n - i);
    if (!open || (size_t)(open - s) == n - 1) break;
    size_t oi = (size_t)(open - s);
    char nx = s[oi + 1];
    if (nx != '{' && nx != '%') { i = oi + 1; continue; }
    // Flush text before the tag.
    if (oi > text_start) {
      Node *nd = new_node(N_TEXT);
      nd->text.s = s + text_start;
      nd->text.n = (u32)(oi - text_start);
      nodelist_push(cur, nd);
    }
    if (nx == '{') {
      int raw = (oi + 2 < n && s[oi + 2] == '{');
      const char *needle = raw ? "}}}" : "}}";
      size_t nlen = raw ? 3 : 2;
      const char *close = NULL;
      // indexOf from oi + (raw?3:2)
      {
        size_t from = oi + nlen;
        for (size_t k = from; k + nlen <= n; k++) {
          if (memcmp(s + k, needle, nlen) == 0) { close = s + k; break; }
        }
      }
      if (!close) {
        snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": a %s is not closed",
                 (int)name.n, name.s, raw ? "{{{" : "{{");
        goto fail;
      }
      size_t ci = (size_t)(close - s);
      Str inner = { s + oi + nlen, (u32)(ci - oi - nlen) };
      inner = trim_str(inner);
      Str *segs;
      u32 nseg;
      if (!path_valid_split(inner, &segs, &nseg)) {
        Buf mb = { 0, 0, 0 };
        buf_addz(&mb, "template \"");
        buf_add(&mb, name.s, name.n);
        buf_addz(&mb, "\": ");
        buf_json_quote(&mb, inner);
        buf_addz(&mb, " is not a path");
        if (mb.len >= sizeof(ebuf)) mb.len = sizeof(ebuf) - 1;
        memcpy(ebuf, mb.buf, mb.len);
        ebuf[mb.len] = 0;
        goto fail;
      }
      Node *nd = new_node(raw ? N_RAW : N_VAR);
      nd->path = segs;
      nd->npath = nseg;
      nodelist_push(cur, nd);
      i = ci + nlen;
      text_start = i;
      continue;
    }
    // '{% ... %}'
    const char *close = NULL;
    for (size_t k = oi + 2; k + 2 <= n; k++) {
      if (s[k] == '%' && s[k + 1] == '}') { close = s + k; break; }
    }
    if (!close) {
      snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": a {%% is not closed", (int)name.n, name.s);
      goto fail;
    }
    size_t ci = (size_t)(close - s);
    size_t end = ci + 2;
    Str tag = { s + oi + 2, (u32)(ci - oi - 2) };
    tag = trim_str(tag);
    // Standalone-tag line rule.
    size_t line_start = oi;
    while (line_start > 0 && s[line_start - 1] != '\n') line_start--;
    size_t nl = end;
    while (nl < n && s[nl] != '\n') nl++;
    size_t line_end = nl;
    int standalone = 1;
    for (size_t k = line_start; k < oi; k++)
      if (s[k] != ' ' && s[k] != '\t') standalone = 0;
    for (size_t k = end; k < line_end; k++)
      if (s[k] != ' ' && s[k] != '\t') standalone = 0;
    if (standalone) {
      // The text node just pushed ends with the line's leading spaces;
      // drop them (open - line_start chars).
      if (cur->n > 0 && cur->items[cur->n - 1]->kind == N_TEXT) {
        Node *tn = cur->items[cur->n - 1];
        size_t drop = oi - line_start;
        if (drop > tn->text.n) drop = tn->text.n;
        tn->text.n -= (u32)drop;
        if (tn->text.n == 0) cur->n--;
      }
      i = (nl >= n) ? n : nl + 1;
    } else {
      i = end;
    }
    text_start = i;
    // Dispatch on the tag.
    if (tag.n >= 4 && memcmp(tag.s, "for ", 4) == 0) {
      // for <name> in <path>
      size_t p = 4;
      if (p >= tag.n || !((tag.s[p] >= 'A' && tag.s[p] <= 'Z') || (tag.s[p] >= 'a' && tag.s[p] <= 'z') || tag.s[p] == '_'))
        goto not_a_tag;
      size_t ns = p;
      while (p < tag.n && ((tag.s[p] >= 'A' && tag.s[p] <= 'Z') || (tag.s[p] >= 'a' && tag.s[p] <= 'z') ||
                           (tag.s[p] >= '0' && tag.s[p] <= '9') || tag.s[p] == '_')) p++;
      Str fname = { tag.s + ns, (u32)(p - ns) };
      if (p + 4 > tag.n || memcmp(tag.s + p, " in ", 4) != 0) goto not_a_tag;
      Str pstr = { tag.s + p + 4, (u32)(tag.n - p - 4) };
      for (u32 q = 0; q < pstr.n; q++)
        if (pstr.s[q] == ' ' || pstr.s[q] == '\t' || pstr.s[q] == '\n' || pstr.s[q] == '\r') goto not_a_tag;
      Str *segs;
      u32 nseg;
      if (!path_valid_split(pstr, &segs, &nseg)) goto not_a_tag;
      Node *nd = new_node(N_FOR);
      nd->for_name = fname;
      nd->path = segs;
      nd->npath = nseg;
      nodelist_push(cur, nd);
      if (sp >= 512) goto fail_stack;
      stack[sp++] = nd;
      cur = &nd->body;
      continue;
    }
    if (tag.n >= 3 && memcmp(tag.s, "if ", 3) == 0) {
      size_t p = 3;
      int not = 0;
      if (p + 4 <= tag.n && memcmp(tag.s + p, "not ", 4) == 0) { not = 1; p += 4; }
      Str pstr = { tag.s + p, (u32)(tag.n - p) };
      for (u32 q = 0; q < pstr.n; q++)
        if (pstr.s[q] == ' ' || pstr.s[q] == '\t' || pstr.s[q] == '\n' || pstr.s[q] == '\r') goto not_a_tag;
      Str *segs;
      u32 nseg;
      if (!path_valid_split(pstr, &segs, &nseg)) goto not_a_tag;
      Node *nd = new_node(N_IF);
      nd->not = not;
      nd->path = segs;
      nd->npath = nseg;
      nodelist_push(cur, nd);
      if (sp >= 512) goto fail_stack;
      stack[sp++] = nd;
      cur = &nd->then_l;
      continue;
    }
    if (str_eq(tag, "else")) {
      if (sp == 0 || stack[sp - 1]->kind != N_IF || stack[sp - 1]->has_else) {
        snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": an else outside an if", (int)name.n, name.s);
        goto fail;
      }
      stack[sp - 1]->has_else = 1;
      cur = &stack[sp - 1]->else_l;
      continue;
    }
    if (str_eq(tag, "endfor") || str_eq(tag, "endif")) {
      int want_for = str_eq(tag, "endfor");
      if (sp == 0 || stack[sp - 1]->kind != (want_for ? N_FOR : N_IF)) {
        snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": an %s with no open %s",
                 (int)name.n, name.s, want_for ? "endfor" : "endif", want_for ? "for" : "if");
        goto fail;
      }
      sp--;
      if (sp == 0) cur = &root;
      else {
        Node *par = stack[sp - 1];
        cur = par->kind == N_FOR ? &par->body : (par->has_else ? &par->else_l : &par->then_l);
      }
      continue;
    }
    if (tag.n >= 9 && memcmp(tag.s, "include ", 8) == 0) {
      // include "<name>"
      if (tag.n < 10 || tag.s[8] != '"' || tag.s[tag.n - 1] != '"') goto not_a_tag;
      Str iname = { tag.s + 9, (u32)(tag.n - 10) };
      int ok = iname.n > 0;
      for (u32 q = 0; q < iname.n && ok; q++) {
        char c = iname.s[q];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) ok = 0;
      }
      if (!ok) goto not_a_tag;
      Node *nd = new_node(N_INCLUDE);
      nd->inc_name = iname;
      nodelist_push(cur, nd);
      continue;
    }
  not_a_tag:
    snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": {%% %.*s %%} is not a tag",
             (int)name.n, name.s, (int)tag.n, tag.s);
    goto fail;
  }
  if (text_start < n) {
    Node *nd = new_node(N_TEXT);
    nd->text.s = s + text_start;
    nd->text.n = (u32)(n - text_start);
    nodelist_push(cur, nd);
  }
  if (sp > 0) {
    snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": a %s is not closed",
             (int)name.n, name.s, stack[sp - 1]->kind == N_FOR ? "for" : "if");
    goto fail;
  }
  out->nodes = root;
  return;
fail_stack:
  snprintf(ebuf, sizeof(ebuf), "template \"%.*s\": a for is not closed", (int)name.n, name.s);
fail:
  out->valid = 0;
  snprintf(out->error, sizeof(out->error), "%s", ebuf);
}

// ---------------------------------------------------------------- template values

typedef struct PageView PageView;
typedef struct LinkView { Str title, url, date, date_long; } LinkView;
typedef struct { long number, total; Str url, newer_url, older_url; } PagerView;
typedef struct { long index, length; int first, last; } LoopView;
typedef struct { Str title, base_url; } SiteView;

struct PageView {
  Str title, url, date, date_long, summary, excerpt;
  long reading_minutes;
  Tag **tags;
  u32 ntags;
  // page-only fields (cards leave them empty)
  Str slug, content;
  TocEntry *toc;
  u32 ntoc;
  int has_newer, has_older;
  LinkView newer, older;
  int is_post;
  int is_card; // cards expose no slug/content/toc/newer/older/is_post
  Source *src; // the source this view is built from (stable identity
  // for include caching; the PageView itself may be stack-allocated)
};

typedef enum { V_UNDEF, V_STR, V_INT, V_BOOL, V_NULL, V_ARR, V_OBJ } VKind;
enum { ARR_CARD, ARR_TAG, ARR_TOC };
enum { OBJ_SITE, OBJ_PAGE, OBJ_CARD, OBJ_TAGFULL, OBJ_TAGCOUNT, OBJ_TAGLINK, OBJ_TOC, OBJ_PAGER, OBJ_LOOP, OBJ_LINK };

typedef struct {
  int kind; // ARR_*
  int obj_kind; // Obj kind of the items (for ARR_TAG: TAGFULL/TAGCOUNT/TAGLINK)
  void *items; // array of pointers (PageView* / Tag* / TocEntry*)
  u32 count;
} Arr;

typedef struct { int kind; void *p; } Obj;

typedef struct {
  VKind kind;
  Str s;
  long i;
  int b;
  Arr *arr;
  Obj obj;
} Value;

static Value v_undef(void) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_UNDEF; return v; }
static Value v_str(Str s) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_STR; v.s = s; return v; }
static Value v_int(long i) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_INT; v.i = i; return v; }
static Value v_bool(int b) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_BOOL; v.b = b; return v; }
static Value v_null(void) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_NULL; return v; }
static Value v_arr(Arr *a) { Value v; memset(&v, 0, sizeof(v)); v.kind = V_ARR; v.arr = a; return v; }
static Value v_obj(int kind, void *p) {
  Value v;
  memset(&v, 0, sizeof(v));
  v.kind = V_OBJ;
  v.obj.kind = kind;
  v.obj.p = p;
  return v;
}

static Value obj_field(Obj o, Str name) {
  switch (o.kind) {
    case OBJ_SITE: {
      SiteView *s = (SiteView *)o.p;
      if (str_eq(name, "title")) return v_str(s->title);
      if (str_eq(name, "base_url")) return v_str(s->base_url);
      return v_undef();
    }
    case OBJ_PAGE:
    case OBJ_CARD: {
      PageView *p = (PageView *)o.p;
      if (str_eq(name, "title")) return v_str(p->title);
      if (str_eq(name, "url")) return v_str(p->url);
      if (str_eq(name, "date")) return v_str(p->date);
      if (str_eq(name, "date_long")) return v_str(p->date_long);
      if (str_eq(name, "summary")) return v_str(p->summary);
      if (str_eq(name, "excerpt")) return v_str(p->excerpt);
      if (str_eq(name, "reading_minutes")) return v_int(p->reading_minutes);
      if (str_eq(name, "tags")) {
        Arr *a = (Arr *)acalloc(sizeof(Arr));
        a->kind = ARR_TAG;
        a->obj_kind = OBJ_TAGLINK;
        a->items = p->tags;
        a->count = p->ntags;
        return v_arr(a);
      }
      if (o.kind == OBJ_CARD) return v_undef();
      if (str_eq(name, "slug")) return v_str(p->slug);
      if (str_eq(name, "content")) return v_str(p->content);
      if (str_eq(name, "toc")) {
        Arr *a = (Arr *)acalloc(sizeof(Arr));
        a->kind = ARR_TOC;
        a->obj_kind = OBJ_TOC;
        // Build a pointer array over the toc entries.
        TocEntry **ptrs = (TocEntry **)acalloc(sizeof(TocEntry *) * (p->ntoc ? p->ntoc : 1));
        for (u32 i = 0; i < p->ntoc; i++) ptrs[i] = &p->toc[i];
        a->items = ptrs;
        a->count = p->ntoc;
        return v_arr(a);
      }
      if (str_eq(name, "newer")) return p->has_newer ? v_obj(OBJ_LINK, &p->newer) : v_null();
      if (str_eq(name, "older")) return p->has_older ? v_obj(OBJ_LINK, &p->older) : v_null();
      if (str_eq(name, "is_post")) return v_bool(p->is_post);
      return v_undef();
    }
    case OBJ_TAGFULL:
    case OBJ_TAGCOUNT:
    case OBJ_TAGLINK: {
      Tag *t = (Tag *)o.p;
      if (str_eq(name, "name")) return v_str(t->name);
      if (str_eq(name, "slug")) return v_str(t->slug);
      if (str_eq(name, "url")) return v_str(t->url);
      if (o.kind == OBJ_TAGLINK) return v_undef();
      if (str_eq(name, "count")) return v_int((long)t->nposts);
      if (o.kind == OBJ_TAGCOUNT) return v_undef();
      if (str_eq(name, "posts")) {
        // t->posts holds Source*; cards need PageView* — convert via
        // each post's card view (same collection order).
        PageView **ptrs = (PageView **)acalloc(sizeof(PageView *) * (t->nposts ? t->nposts : 1));
        for (u32 i = 0; i < t->nposts; i++) ptrs[i] = t->posts[i]->card;
        Arr *a = (Arr *)acalloc(sizeof(Arr));
        a->kind = ARR_CARD;
        a->obj_kind = OBJ_CARD;
        a->items = ptrs;
        a->count = t->nposts;
        return v_arr(a);
      }
      return v_undef();
    }
    case OBJ_TOC: {
      TocEntry *t = (TocEntry *)o.p;
      if (str_eq(name, "level")) return v_int(t->level);
      if (str_eq(name, "id")) return v_str(t->id);
      if (str_eq(name, "html")) return v_str(t->html);
      return v_undef();
    }
    case OBJ_PAGER: {
      PagerView *pg = (PagerView *)o.p;
      if (str_eq(name, "number")) return v_int(pg->number);
      if (str_eq(name, "total")) return v_int(pg->total);
      if (str_eq(name, "url")) return v_str(pg->url);
      if (str_eq(name, "newer_url")) return v_str(pg->newer_url);
      if (str_eq(name, "older_url")) return v_str(pg->older_url);
      return v_undef();
    }
    case OBJ_LOOP: {
      LoopView *lv = (LoopView *)o.p;
      if (str_eq(name, "index")) return v_int(lv->index);
      if (str_eq(name, "first")) return v_bool(lv->first);
      if (str_eq(name, "last")) return v_bool(lv->last);
      if (str_eq(name, "length")) return v_int(lv->length);
      return v_undef();
    }
    case OBJ_LINK: {
      LinkView *l = (LinkView *)o.p;
      if (str_eq(name, "title")) return v_str(l->title);
      if (str_eq(name, "url")) return v_str(l->url);
      if (str_eq(name, "date")) return v_str(l->date);
      if (str_eq(name, "date_long")) return v_str(l->date_long);
      return v_undef();
    }
  }
  return v_undef();
}

typedef struct { Str name; Value v; } Var;
typedef struct Frame { Var *vars; u32 nvars; struct Frame *parent; } Frame;

__attribute__((always_inline)) static inline Value lookup(Frame *scope, Str *path, u32 npath) {
  Frame *s = scope;
  Value v = v_undef();
  while (s) {
    int hit = 0;
    for (u32 i = 0; i < s->nvars; i++)
      if (str_eq2(s->vars[i].name, path[0])) { v = s->vars[i].v; hit = 1; break; }
    if (hit) break;
    s = s->parent;
  }
  if (v.kind == V_UNDEF) return v;
  for (u32 k = 1; k < npath; k++) {
    if (v.kind != V_OBJ) return v_undef();
    v = obj_field(v.obj, path[k]);
    if (v.kind == V_UNDEF) return v_undef();
  }
  return v;
}

static void text_of(Value v, Buf *out) {
  switch (v.kind) {
    case V_STR: buf_add(out, v.s.s, v.s.n); break;
    case V_INT: {
      char tmp[32];
      int n = snprintf(tmp, sizeof(tmp), "%ld", v.i);
      buf_add(out, tmp, (u32)n);
      break;
    }
    case V_BOOL: buf_addz(out, v.b ? "true" : "false"); break;
    case V_ARR: {
      for (u32 i = 0; i < v.arr->count; i++) {
        if (i) buf_addz(out, ", ");
        Value item = v_undef();
        void **items = (void **)v.arr->items;
        switch (v.arr->kind) {
          case ARR_CARD: item = v_obj(OBJ_CARD, items[i]); break;
          case ARR_TAG: item = v_obj(v.arr->obj_kind, items[i]); break;
          case ARR_TOC: item = v_obj(OBJ_TOC, items[i]); break;
        }
        text_of(item, out);
      }
      break;
    }
    default: break; // null, undefined, objects -> ""
  }
}

// Escape a short string straight into out (scalar, direct writes, no
// function call). Reserved n*6+8 up front, so any input is safe; the
// output Buf is already large in practice, so the reserve is a no-op.
// Used for template {{ }} values, which are all short (title/url/date).
__attribute__((always_inline)) static inline void buf_escape_short(Buf *out, const char *s, size_t n) {
  buf_reserve(out, n * 6 + 8);
  char *w = out->buf + out->len;
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)s[i];
    switch (c) {
      case '&': memcpy(w, "&amp;", 5); w += 5; break;
      case '<': memcpy(w, "&lt;", 4); w += 4; break;
      case '>': memcpy(w, "&gt;", 4); w += 4; break;
      case '"': memcpy(w, "&quot;", 6); w += 6; break;
      case '\'': memcpy(w, "&#39;", 5); w += 5; break;
      default: *w++ = (char)c; break;
    }
  }
  out->len = (size_t)(w - out->buf);
}

// text_of, but escaped as it is written: no intermediate buffer.
// Escaping is per-character, so escaping each array element separately
// is identical to escaping the joined string (", " has no escapables).
static void text_of_escaped(Value v, Buf *out) {
  switch (v.kind) {
    case V_STR:
      if (v.s.n <= 128) buf_escape_short(out, v.s.s, v.s.n);
      else buf_escape(out, v.s.s, v.s.n);
      break;
    case V_INT: {
      char tmp[32];
      int n = snprintf(tmp, sizeof(tmp), "%ld", v.i);
      buf_add(out, tmp, (u32)n);
      break;
    }
    case V_BOOL: buf_addz(out, v.b ? "true" : "false"); break;
    case V_ARR: {
      for (u32 i = 0; i < v.arr->count; i++) {
        if (i) buf_addz(out, ", ");
        Value item = v_undef();
        void **items = (void **)v.arr->items;
        switch (v.arr->kind) {
          case ARR_CARD: item = v_obj(OBJ_CARD, items[i]); break;
          case ARR_TAG: item = v_obj(v.arr->obj_kind, items[i]); break;
          case ARR_TOC: item = v_obj(OBJ_TOC, items[i]); break;
        }
        text_of_escaped(item, out);
      }
      break;
    }
    default: break; // null, undefined, objects -> ""
  }
}

static int truthy(Value v) {
  if (v.kind == V_ARR) return v.arr->count > 0;
  switch (v.kind) {
    case V_UNDEF: case V_NULL: return 0;
    case V_BOOL: return v.b;
    case V_STR: return v.s.n > 0;
    case V_INT: return v.i != 0;
    case V_OBJ: return 1;
    default: return 0;
  }
}

typedef struct {
  ParsedTemplate *templates;
  u32 ntemplates;
} TplCtx;

static ParsedTemplate *tpl_find(TplCtx *ctx, Str name) {
  for (u32 i = 0; i < ctx->ntemplates; i++)
    if (str_eq2(ctx->templates[i].name, name)) return &ctx->templates[i];
  return NULL;
}

// Resolve every N_INCLUDE node's inc_tpl pointer (once per case, after
// all templates are parsed) so the hot render loop never does a name
// lookup. A missing include resolves to NULL (renders nothing).
static void resolve_includes_list(NodeList *nl, TplCtx *ctx) {
  for (u32 i = 0; i < nl->n; i++) {
    Node *nd = nl->items[i];
    if (nd->kind == N_INCLUDE) nd->inc_tpl = tpl_find(ctx, nd->inc_name);
    else if (nd->kind == N_FOR) resolve_includes_list(&nd->body, ctx);
    else if (nd->kind == N_IF) {
      resolve_includes_list(&nd->then_l, ctx);
      resolve_includes_list(&nd->else_l, ctx);
    }
  }
}

// ---------------------------------------------------------------- include caching
//
// A template's render is fully determined by the values it reads. The
// analysis below computes, per template, the set of first-segment names
// it reads from a scope outside itself (its "external dependencies"),
// plus whether it reads an enclosing for's `loop` state and its maximum
// include depth. With that:
//  * a template that reads only the top frame (site/tags/recent) renders
//    identically everywhere in a case: render once, reuse everywhere.
//  * a template that reads only the top frame plus one stable object
//    (a loop item, the page's source, a tag) is cached per that object:
//    the card of a post renders once and is reused on the index page and
//    on every tag page; a page-dependent filler renders once per page
//    even when the template includes it twice.
//  * anything else (pager, an enclosing loop, a cycle) renders fresh.
// Names bound by a template's own for-loops resolve to that loop's
// source, so a filler's own `{% for p in recent %}` stays top-level.
// The cache is only used when the whole include subtree fits under the
// render depth limit at both the fill and the serve depth, so the
// depth<8 gate can never change the cached bytes.

static Str dep_names[32];
static int dep_nnames;
static u32 dep_top_mask; // bits for site/tags/recent

static int dep_bit(Str name) {
  for (int i = 0; i < dep_nnames; i++)
    if (str_eq2(dep_names[i], name)) return i;
  if (dep_nnames >= 32) return -1;
  dep_names[dep_nnames] = name;
  return dep_nnames++;
}

static void dep_init(void) {
  if (dep_nnames) return;
  dep_bit(str_lit("site"));
  dep_bit(str_lit("tags"));
  dep_bit(str_lit("recent"));
  dep_top_mask = (1u << 0) | (1u << 1) | (1u << 2);
}

typedef struct { Str name; u32 dep; } DepBound;

// What one name resolves to in the current analysis scope: the mask of
// external names its value depends on (0 = top-level). Unbound names
// add their own bit; an unbound `loop` sets the loop flag instead.
static u32 dep_resolve(Str name, DepBound *bound, int nbound, int *loop_flag, int *overflow) {
  for (int i = nbound - 1; i >= 0; i--)
    if (str_eq2(bound[i].name, name)) return bound[i].dep;
  if (str_eq(name, "loop")) { *loop_flag = 1; return 0; }
  int b = dep_bit(name);
  if (b < 0) { *overflow = 1; return 0; }
  return 1u << b;
}

static u32 analyze_nodes(NodeList *nl, DepBound *bound, int nbound, int *loop_flag, int *overflow, int *max_inc);
static void analyze_template(ParsedTemplate *t, int *loop_flag, int *overflow, int *max_inc);

static u32 analyze_nodes(NodeList *nl, DepBound *bound, int nbound, int *loop_flag, int *overflow, int *max_inc) {
  u32 mask = 0;
  for (u32 i = 0; i < nl->n; i++) {
    Node *nd = nl->items[i];
    switch (nd->kind) {
      case N_TEXT: break;
      case N_VAR:
      case N_RAW:
        mask |= dep_resolve(nd->path[0], bound, nbound, loop_flag, overflow);
        break;
      case N_FOR: {
        u32 src = dep_resolve(nd->path[0], bound, nbound, loop_flag, overflow);
        mask |= src;
        // The body sees the item var and `loop`, both bound to the
        // loop's source. Cap the bound depth (deeply nested fors give
        // up caching rather than overflowing the array).
        if (nbound + 2 <= 16) {
          DepBound nb[16];
          memcpy(nb, bound, sizeof(DepBound) * (size_t)nbound);
          nb[nbound].name = nd->for_name;
          nb[nbound].dep = src;
          nb[nbound + 1].name = str_lit("loop");
          nb[nbound + 1].dep = src;
          mask |= analyze_nodes(&nd->body, nb, nbound + 2, loop_flag, overflow, max_inc);
        } else {
          *overflow = 1;
          mask |= analyze_nodes(&nd->body, bound, nbound, loop_flag, overflow, max_inc);
        }
        break;
      }
      case N_IF: {
        mask |= dep_resolve(nd->path[0], bound, nbound, loop_flag, overflow);
        mask |= analyze_nodes(&nd->then_l, bound, nbound, loop_flag, overflow, max_inc);
        mask |= analyze_nodes(&nd->else_l, bound, nbound, loop_flag, overflow, max_inc);
        break;
      }
      case N_INCLUDE: {
        ParsedTemplate *tg = nd->inc_tpl;
        if (!tg) break;
        if (tg->dep_visiting) { *overflow = 1; break; } // include cycle: never cache
        int lf = 0, ov = 0, mi = 0;
        analyze_template(tg, &lf, &ov, &mi);
        if (ov) { *overflow = 1; break; }
        if (lf) { *loop_flag = 1; break; }
        if (mi + 1 > *max_inc) *max_inc = mi + 1;
        // Map the target's external names through our bound: a name our
        // own for binds resolves to that for's source dependency.
        u32 m = tg->dep_mask;
        for (int b = 0; b < dep_nnames; b++) {
          if (!(m & (1u << b))) continue;
          Str nm = dep_names[b];
          int found = 0;
          for (int j = nbound - 1; j >= 0; j--) {
            if (str_eq2(bound[j].name, nm)) { mask |= bound[j].dep; found = 1; break; }
          }
          if (!found) mask |= 1u << b;
        }
        break;
      }
    }
  }
  return mask;
}

static void analyze_template(ParsedTemplate *t, int *loop_flag, int *overflow, int *max_inc) {
  if (t->dep_done) {
    *loop_flag = t->dep_loop;
    *overflow = t->dep_bad;
    *max_inc = t->inc_depth;
    return;
  }
  if (t->dep_visiting) { *overflow = 1; return; }
  t->dep_visiting = 1;
  int lf = 0, ov = 0, mi = 0;
  DepBound bound[1];
  u32 mask = analyze_nodes(&t->nodes, bound, 0, &lf, &ov, &mi);
  t->dep_visiting = 0;
  t->dep_done = 1;
  t->dep_mask = mask;
  t->dep_loop = lf;
  t->dep_bad = ov;
  t->inc_depth = mi;
  *loop_flag = lf;
  *overflow = ov;
  *max_inc = mi;
}

// The include-result cache: keyed by (template, stable object, stable
// object). The objects are arena/heap identities (a post's card view, a
// tag, a source); stack objects (pager, loop state) are never keys.
typedef struct {
  ParsedTemplate **tpls;
  void **a, **b;
  Str *vals;
  u8 *used;
  u32 cap, count;
} IncCache;
static IncCache inc_cache;

static void inc_cache_init(u32 cap) {
  inc_cache.cap = cap;
  inc_cache.count = 0;
  inc_cache.tpls = (ParsedTemplate **)acalloc(sizeof(ParsedTemplate *) * cap);
  inc_cache.a = (void **)acalloc(sizeof(void *) * cap);
  inc_cache.b = (void **)acalloc(sizeof(void *) * cap);
  inc_cache.vals = (Str *)acalloc(sizeof(Str) * cap);
  // The arena allocator does NOT zero: `used` must be cleared
  // explicitly (a stale nonzero byte here makes get/put probe forever).
  inc_cache.used = (u8 *)acalloc(sizeof(u8) * cap);
  memset(inc_cache.used, 0, cap);
}
static u64 inc_cache_hash(ParsedTemplate *t, void *a, void *b) {
  u64 h = (u64)(uintptr_t)t * 0x9E3779B97F4A7C15ULL;
  h ^= (u64)(uintptr_t)a * 0xC2B2AE3D27D4EB4FULL + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= (u64)(uintptr_t)b * 0x165667B19E3779F9ULL + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return h;
}
static int inc_cache_get(ParsedTemplate *t, void *a, void *b, Str *out) {
  u32 i = (u32)inc_cache_hash(t, a, b) & (inc_cache.cap - 1);
  while (inc_cache.used[i]) {
    if (inc_cache.tpls[i] == t && inc_cache.a[i] == a && inc_cache.b[i] == b) {
      *out = inc_cache.vals[i];
      return 1;
    }
    i = (i + 1) & (inc_cache.cap - 1);
  }
  return 0;
}
static void inc_cache_put(ParsedTemplate *t, void *a, void *b, Str val) {
  if (inc_cache.count * 2 >= inc_cache.cap) return;
  u32 i = (u32)inc_cache_hash(t, a, b) & (inc_cache.cap - 1);
  while (inc_cache.used[i]) {
    if (inc_cache.tpls[i] == t && inc_cache.a[i] == a && inc_cache.b[i] == b) {
      inc_cache.vals[i] = val;
      return;
    }
    i = (i + 1) & (inc_cache.cap - 1);
  }
  inc_cache.used[i] = 1;
  inc_cache.tpls[i] = t;
  inc_cache.a[i] = a;
  inc_cache.b[i] = b;
  inc_cache.vals[i] = val;
  inc_cache.count++;
}

// The stable cache key for one external name at an include site: walk
// the scope chain the way lookup does and return the identity of the
// object the template actually reads, or NULL when there is none (the
// name is unbound, or its value is a stack object / a non-object).
static void *dep_key_for(Frame *scope, Str name) {
  Frame *s = scope;
  while (s) {
    for (u32 i = 0; i < s->nvars; i++) {
      if (str_eq2(s->vars[i].name, name)) {
        Value v = s->vars[i].v;
        if (v.kind != V_OBJ) return NULL;
        if (v.obj.kind == OBJ_PAGE) {
          PageView *pv = (PageView *)v.obj.p;
          return pv->src; // the PageView may be stack-allocated
        }
        if (v.obj.kind == OBJ_PAGER || v.obj.kind == OBJ_LOOP) return NULL;
        return v.obj.p;
      }
    }
    s = s->parent;
  }
  return NULL;
}

static void render_nodes(NodeList *nodes, Frame *scope, TplCtx *ctx, int depth, Buf *out) {
  for (u32 ni = 0; ni < nodes->n; ni++) {
    Node *nd = nodes->items[ni];
    switch (nd->kind) {
      case N_TEXT:
        buf_add(out, nd->text.s, nd->text.n);
        break;
      case N_VAR: {
        Value v = lookup(scope, nd->path, nd->npath);
        text_of_escaped(v, out);
        break;
      }
      case N_RAW: {
        Value v = lookup(scope, nd->path, nd->npath);
        text_of(v, out);
        break;
      }
      case N_IF: {
        Value v = lookup(scope, nd->path, nd->npath);
        int t = truthy(v);
        NodeList *br = (t != nd->not) ? &nd->then_l : (nd->has_else ? &nd->else_l : NULL);
        if (br) render_nodes(br, scope, ctx, depth, out);
        break;
      }
      case N_FOR: {
        Value v = lookup(scope, nd->path, nd->npath);
        if (v.kind != V_ARR) break;
        for (u32 k = 0; k < v.arr->count; k++) {
          Value item = v_undef();
          void **items = (void **)v.arr->items;
          switch (v.arr->kind) {
            case ARR_CARD: item = v_obj(OBJ_CARD, items[k]); break;
            case ARR_TAG: item = v_obj(v.arr->obj_kind, items[k]); break;
            case ARR_TOC: item = v_obj(OBJ_TOC, items[k]); break;
          }
          LoopView lv;
          lv.index = (long)k + 1;
          lv.length = (long)v.arr->count;
          lv.first = k == 0;
          lv.last = k == v.arr->count - 1;
          Var vars[2];
          // `loop` first: it wins over an element bound under the same
          // name (the reference builds { [name]: x, loop: {...} }).
          vars[0].name = str_lit("loop");
          vars[0].v = v_obj(OBJ_LOOP, &lv);
          vars[1].name = nd->for_name;
          vars[1].v = item;
          Frame fr = { vars, 2, scope };
          render_nodes(&nd->body, &fr, ctx, depth, out);
        }
        break;
      }
      case N_INCLUDE: {
        ParsedTemplate *t = nd->inc_tpl;
        if (!t || depth >= 8) break;
        int lf = 0, ov = 0, mi = 0;
        analyze_template(t, &lf, &ov, &mi);
        // Cache only when the whole include subtree fits under the
        // depth limit at this depth, so the depth<8 gate can never
        // change the cached bytes (mi==0 has no nested gates at all).
        void *ka = NULL, *kb = NULL;
        int use_cache = 0;
        if (!ov && !lf && (mi == 0 || depth + 1 + mi < 8)) {
          u32 extra = t->dep_mask & ~dep_top_mask;
          if (extra == 0) {
            use_cache = 1;
          } else {
            // One or two stable objects determine the output: resolve
            // each extra name to its object identity in this scope.
            void *keys[2];
            int nk = 0, nbits = 0, ok = 1;
            for (int bidx = 0; bidx < dep_nnames; bidx++) {
              if (!(extra & (1u << bidx))) continue;
              nbits++;
              if (nk >= 2) { ok = 0; break; }
              void *k = dep_key_for(scope, dep_names[bidx]);
              if (!k) { ok = 0; break; }
              keys[nk++] = k;
            }
            if (ok && nk > 0 && nbits == nk) {
              use_cache = 1;
              ka = keys[0];
              kb = nk > 1 ? keys[1] : NULL;
            }
          }
        }
        if (use_cache) {
          Str hit;
          if (inc_cache_get(t, ka, kb, &hit)) {
            buf_add(out, hit.s, hit.n);
          } else {
            Buf cb = { 0, 0, 0 };
            render_nodes(&t->nodes, scope, ctx, depth + 1, &cb);
            Str val = { cb.buf, (u32)cb.len };
            inc_cache_put(t, ka, kb, val);
            buf_add(out, val.s, val.n);
          }
        } else {
          render_nodes(&t->nodes, scope, ctx, depth + 1, out);
        }
        break;
      }
    }
  }
}

// ---------------------------------------------------------------- build pipeline

static int cmp_strp(const void *a, const void *b) {
  const Str *x = (const Str *)a, *y = (const Str *)b;
  return str_cmp(*x, *y);
}

static Str dir_of(Str path) {
  for (u32 i = path.n; i > 0; i--)
    if (path.s[i - 1] == '/') return (Str){ path.s, i - 1 };
  return (Str){ path.s, 0 };
}

static int ends_md(Str path) {
  return path.n >= 3 && memcmp(path.s + path.n - 3, ".md", 3) == 0;
}

// Normalize \r\n and lone \r to \n (only when a '\r' is present).
static Str normalize_text(Str content) {
  if (!memchr(content.s, '\r', content.n)) return content;
  char *buf = (char *)acalloc(content.n + 1);
  u32 w = 0;
  for (u32 i = 0; i < content.n; i++) {
    if (content.s[i] == '\r') {
      buf[w++] = '\n';
      if (i + 1 < content.n && content.s[i + 1] == '\n') i++;
    } else buf[w++] = content.s[i];
  }
  Str r = { buf, w };
  return r;
}

typedef struct {
  Str path, text;
} OutPair;

typedef struct {
  CaseInput *in;
  Str base;
  ParsedTemplate *templates;
  u32 ntemplates;
  Str *report;
  u32 nreport, cap_report;
  Source *sources; // kept sources, in path order
  u32 nsources;
  Source **posts; // collection order
  u32 nposts;
  Tag **tags; // sorted by count desc, slug asc
  u32 ntags;
  OutPair *out;
  u32 nout, cap_out;
  StrSet out_set; // output paths (shadow check)
  StrSet by_source; // source path -> index in sources
  StrSet statics; // static paths
  StrSet broken; // broken-link lines
} Build;

static void build_report(Build *b, const char *msg, size_t n) {
  if (b->nreport == b->cap_report) {
    u32 nc = b->cap_report ? b->cap_report * 2 : 32;
    Str *nr = (Str *)acalloc(sizeof(Str) * nc);
    if (b->nreport) memcpy(nr, b->report, sizeof(Str) * b->nreport);
    b->report = nr;
    b->cap_report = nc;
  }
  char *copy = adup(msg, (u32)n);
  b->report[b->nreport].s = copy;
  b->report[b->nreport].n = (u32)n;
  b->nreport++;
}
static void build_reportf(Build *b, const char *fmt, Str a, Str c) {
  // fmt has two %.*s slots (a then c); used for refused/draft/shadowed.
  char buf[512];
  int n = snprintf(buf, sizeof(buf), fmt, (int)a.n, a.s, (int)c.n, c.s);
  build_report(b, buf, (size_t)n);
}

static void build_out(Build *b, Str path, Str text) {
  if (b->nout == b->cap_out) {
    u32 nc = b->cap_out ? b->cap_out * 2 : 64;
    OutPair *no = (OutPair *)acalloc(sizeof(OutPair) * nc);
    if (b->nout) memcpy(no, b->out, sizeof(OutPair) * b->nout);
    b->out = no;
    b->cap_out = nc;
  }
  b->out[b->nout].path = path;
  b->out[b->nout].text = text;
  b->nout++;
  strset_add(&b->out_set, path, 0);
}

static int cmp_source_date(const void *a, const void *b) {
  Source * const *x = (Source * const *)a;
  Source * const *y = (Source * const *)b;
  int c = str_cmp((*y)->fm.date, (*x)->fm.date); // newest first
  if (c) return c;
  return str_cmp((*x)->slug, (*y)->slug);
}
static int cmp_tag_order(const void *a, const void *b) {
  Tag * const *x = (Tag * const *)a;
  Tag * const *y = (Tag * const *)b;
  if ((*x)->nposts != (*y)->nposts) return (*x)->nposts < (*y)->nposts ? 1 : -1;
  return str_cmp((*x)->slug, (*y)->slug);
}
static int cmp_template_name(const void *a, const void *b) {
  const Template *x = (const Template *)a, *y = (const Template *)b;
  return str_cmp(x->name, y->name);
}
static int cmp_inputfile_path(const void *a, const void *b) {
  const InputFile *x = (const InputFile *)a, *y = (const InputFile *)b;
  return str_cmp(x->path, y->path);
}

// The tags of one source: each slug once, first spelling kept.
static void own_tags(Source *s, Str **out_names, u32 *out_n) {
  Str *names = (Str *)acalloc(sizeof(Str) * (s->fm.ntags ? s->fm.ntags : 1));
  // Cache each added name's slug (computed once) instead of re-slugifying
  // every previously-added name on each dedup comparison (O(n^2) slugify).
  char (*slugs)[72] = (char (*)[72])acalloc(sizeof(char[72]) * (s->fm.ntags ? s->fm.ntags : 1));
  u32 *slens = (u32 *)acalloc(sizeof(u32) * (s->fm.ntags ? s->fm.ntags : 1));
  u32 n = 0;
  for (u32 i = 0; i < s->fm.ntags; i++) {
    char slugbuf[72];
    u32 sl = slugify(s->fm.tags[i], slugbuf);
    int dup = 0;
    for (u32 j = 0; j < n; j++) {
      if (sl == slens[j] && memcmp(slugbuf, slugs[j], sl) == 0) { dup = 1; break; }
    }
    if (!dup) { memcpy(slugs[n], slugbuf, sl); slens[n] = sl; names[n++] = s->fm.tags[i]; }
  }
  *out_names = names;
  *out_n = n;
}

static Tag *tag_find_or_create(Build *b, Str name, Str slug, int create) {
  // Linear scan over the tag list (few dozen tags).
  for (u32 i = 0; i < b->ntags; i++)
    if (str_eq2(b->tags[i]->slug, slug)) return b->tags[i];
  if (!create) return NULL;
  Tag *t = (Tag *)azalloc(sizeof(Tag));
  t->name = name;
  t->slug = slug;
  char *ub = (char *)acalloc(b->base.n + 7 + slug.n + 2);
  u32 w = 0;
  memcpy(ub + w, b->base.s, b->base.n); w += b->base.n;
  memcpy(ub + w, "/tags/", 6); w += 6;
  memcpy(ub + w, slug.s, slug.n); w += slug.n;
  ub[w++] = '/';
  t->url.s = ub;
  t->url.n = w;
  if (b->ntags % 32 == 0) {
    // grow the tag pointer array (amortized; stored in arena)
    u32 nc = b->ntags + 32;
    Tag **nt = (Tag **)acalloc(sizeof(Tag *) * nc);
    if (b->ntags) memcpy(nt, b->tags, sizeof(Tag *) * b->ntags);
    b->tags = nt;
  }
  b->tags[b->ntags++] = t;
  return t;
}

static void tag_add_post(Tag *t, Source *s) {
  if (t->nposts == t->cap_posts) {
    u32 nc = t->cap_posts ? t->cap_posts * 2 : 8;
    Source **np = (Source **)acalloc(sizeof(Source *) * nc);
    if (t->nposts) memcpy(np, t->posts, sizeof(Source *) * t->nposts);
    t->posts = np;
    t->cap_posts = nc;
  }
  t->posts[t->nposts++] = s;
}

// Build one case's output tree into b->out.
static void build_case(CaseInput *in, Build *b) {
  double _pt = now_s();
  memset(b, 0, sizeof(*b));
  b->in = in;
  // base: base_url with trailing slashes dropped.
  Str base = in->base_url;
  while (base.n > 0 && base.s[base.n - 1] == '/') base.n--;
  b->base = base;

  // 1. Templates, parsed in code-unit order of name.
  qsort(in->templates, in->ntemplates, sizeof(Template), cmp_template_name);
  b->templates = (ParsedTemplate *)acalloc(sizeof(ParsedTemplate) * (in->ntemplates ? in->ntemplates : 1));
  for (u32 i = 0; i < in->ntemplates; i++) {
    ParsedTemplate *pt = &b->templates[b->ntemplates];
    parse_template(in->templates[i].name, in->templates[i].src, pt);
    if (!pt->valid) {
      build_report(b, pt->error, strlen(pt->error));
      continue;
    }
    b->ntemplates++;
    // Compact: keep valid templates at the front.
    if (pt != &b->templates[b->ntemplates - 1])
      b->templates[b->ntemplates - 1] = *pt;
  }
  TplCtx tctx = { b->templates, b->ntemplates };
  for (u32 i = 0; i < b->ntemplates; i++)
    resolve_includes_list(&b->templates[i].nodes, &tctx);

  // 2. Sources, sorted by path.
  T_tpl += now_s() - _pt; _pt = now_s();
  qsort(in->files, in->nfiles, sizeof(InputFile), cmp_inputfile_path);
  strset_init(&b->statics, 64);
  Source *srcs = (Source *)acalloc(sizeof(Source) * (in->nfiles ? in->nfiles : 1));
  u32 nsrc = 0;
  for (u32 i = 0; i < in->nfiles; i++) {
    InputFile *f = &in->files[i];
    if (!ends_md(f->path)) {
      strset_add(&b->statics, f->path, 0);
      continue;
    }
    Source *s = &srcs[nsrc];
    memset(s, 0, sizeof(*s));
    s->path = f->path;
    s->text = normalize_text(f->content);
    FrontMatter *fm = &s->fm;
    parse_front_matter(s->text, fm);
    if (!fm->ok) {
      build_reportf(b, "refused %.*s: %.*s", s->path, (Str){ fm->error, (u32)strlen(fm->error) });
      continue;
    }
    s->post = s->path.n >= 6 && memcmp(s->path.s, "posts/", 6) == 0;
    if (!s->post && s->path.n >= 5 && memcmp(s->path.s, "tags/", 5) == 0) {
      build_reportf(b, "refused %.*s: %.*s", s->path, str_lit("tags/ is the generator's own"));
      continue;
    }
    if (!s->post && s->path.n >= 5 && memcmp(s->path.s, "page/", 5) == 0) {
      build_reportf(b, "refused %.*s: %.*s", s->path, str_lit("page/ is the generator's own"));
      continue;
    }
    if (s->post && !fm->has_date) {
      build_reportf(b, "refused %.*s: %.*s", s->path, str_lit("a post needs a date"));
      continue;
    }
    Str layout = fm->has_layout ? fm->layout : (s->post ? str_lit("post") : str_lit("page"));
    if (!tpl_find(&tctx, layout)) {
      char msg[256];
      int n = snprintf(msg, sizeof(msg), "refused %.*s: layout \"%.*s\" is no template that parses",
                       (int)s->path.n, s->path.s, (int)layout.n, layout.s);
      build_report(b, msg, (size_t)n);
      continue;
    }
    s->layout = layout;
    if (fm->has_draft && fm->draft) {
      build_reportf(b, "draft %.*s", s->path, (Str){ 0, 0 });
      continue;
    }
    nsrc++;
  }
  b->sources = srcs;
  b->nsources = nsrc;

  // 3. Permalinks.
  T_src += now_s() - _pt; _pt = now_s();
  typedef struct { Str dir; StrSet taken; } DirTaken;
  DirTaken dirs[16];
  int ndirs = 0;
  for (u32 i = 0; i < nsrc; i++) {
    Source *s = &srcs[i];
    Str dir = s->post ? str_lit("posts") : dir_of(s->path);
    DirTaken *dt = NULL;
    for (int d = 0; d < ndirs; d++)
      if (str_eq2(dirs[d].dir, dir)) { dt = &dirs[d]; break; }
    if (!dt) {
      dt = &dirs[ndirs++];
      dt->dir = dir;
      strset_init(&dt->taken, 1024);
      if (dir.n == 0) {
        strset_add(&dt->taken, str_lit("posts"), 0);
        strset_add(&dt->taken, str_lit("tags"), 0);
        strset_add(&dt->taken, str_lit("page"), 0);
      }
    }
    char slugbuf[72];
    Str base_slug;
    if (s->fm.has_slug) base_slug = s->fm.slug;
    else {
      u32 sl = slugify(s->fm.title, slugbuf);
      base_slug.s = slugbuf;
      base_slug.n = sl;
    }
    Str slug = unique_slug(base_slug, &dt->taken);
    s->slug = slug;
    char *ob = (char *)acalloc(dir.n + 1 + slug.n + 1);
    u32 w = 0;
    if (dir.n > 0) { memcpy(ob + w, dir.s, dir.n); w += dir.n; ob[w++] = '/'; }
    memcpy(ob + w, slug.s, slug.n); w += slug.n;
    s->out_dir.s = ob;
    s->out_dir.n = w;
    char *ub = (char *)acalloc(b->base.n + 1 + w + 2);
    u32 w2 = 0;
    memcpy(ub + w2, b->base.s, b->base.n); w2 += b->base.n;
    ub[w2++] = '/';
    memcpy(ub + w2, ob, w); w2 += w;
    ub[w2++] = '/';
    s->url.s = ub;
    s->url.n = w2;
  }

  // 4. Markdown.
  T_perm += now_s() - _pt; _pt = now_s();
  strset_init_vals(&b->by_source, 2048);
  for (u32 i = 0; i < nsrc; i++) strset_add(&b->by_source, srcs[i].path, i);
  strset_init(&b->broken, 16384);
  link_memo_init(8192);
  MdEnv env;
  env.by_source = &b->by_source;
  env.sources = srcs;
  env.statics = &b->statics;
  env.base = b->base;
  env.broken = &b->broken;
  for (u32 i = 0; i < nsrc; i++) {
    Source *s = &srcs[i];
    env.dir = dir_of(s->path);
    env.hdir = hash_bytes(env.dir.s, env.dir.n);
    // Pre-split the dir into segments for resolve_path.
    env.ndir_segs = 0;
    if (env.dir.n > 0) {
      u32 ds = 0;
      for (u32 j = 0; j <= env.dir.n; j++) {
        if (j == env.dir.n || env.dir.s[j] == '/') {
          if (j > ds && env.ndir_segs < 8)
            env.dir_segs[env.ndir_segs++] = (Str){ env.dir.s + ds, j - ds };
          ds = j + 1;
        }
      }
    }
    env.cur_path = s->path;
    MdOut md;
    double _mt0 = now_s();
    render_markdown(s->fm.body, &env, &md);
    double _mt1 = now_s();
    T_md_render += _mt1 - _mt0;
    s->html = md.html;
    s->toc = md.toc;
    s->ntoc = md.ntoc;
    s->excerpt = md.excerpt;
    u32 words = words_of(s->fm.body.s, s->fm.body.n);
    T_words += now_s() - _mt1;
    long mins = (long)((words + 199) / 200);
    s->reading_minutes = mins < 1 ? 1 : (int)mins;
  }

  // 5. Collections.
  T_md += now_s() - _pt; _pt = now_s();
  // date_long is needed for every page view, card and neighbour link;
  // format it once per source instead of once per use (snprintf each).
  for (u32 i = 0; i < nsrc; i++)
    srcs[i].date_long = srcs[i].fm.has_date ? long_date(srcs[i].fm.date) : str_lit("");
  Source **posts = (Source **)acalloc(sizeof(Source *) * (nsrc ? nsrc : 1));
  u32 nposts = 0;
  for (u32 i = 0; i < nsrc; i++)
    if (srcs[i].post) posts[nposts++] = &srcs[i];
  qsort(posts, nposts, sizeof(Source *), cmp_source_date);
  for (u32 i = 0; i < nposts; i++) {
    posts[i]->newer = i == 0 ? NULL : posts[i - 1];
    posts[i]->older = i == nposts - 1 ? NULL : posts[i + 1];
  }
  b->posts = posts;
  b->nposts = nposts;

  // 6. Tags.
  T_coll += now_s() - _pt; _pt = now_s();
  b->tags = (Tag **)acalloc(sizeof(Tag *) * 64);
  b->ntags = 0;
  for (u32 i = 0; i < nposts; i++) {
    Source *p = posts[i];
    Str *names;
    u32 nn;
    own_tags(p, &names, &nn);
    p->own_tag_names = names;
    p->nown_tags = nn;
    p->tag_refs = (Tag **)acalloc(sizeof(Tag *) * (nn ? nn : 1));
    p->ntag_refs = 0;
    for (u32 k = 0; k < nn; k++) {
      char slugbuf[72];
      u32 sl = slugify(names[k], slugbuf);
      Str slug = { adup(slugbuf, sl), sl };
      Tag *t = tag_find_or_create(b, names[k], slug, 1);
      tag_add_post(t, p);
      p->tag_refs[p->ntag_refs++] = t;
    }
  }
  for (u32 i = 0; i < nsrc; i++) {
    Source *s = &srcs[i];
    if (s->post) continue;
    Str *names;
    u32 nn;
    own_tags(s, &names, &nn);
    s->own_tag_names = names;
    s->nown_tags = nn;
    s->tag_refs = (Tag **)acalloc(sizeof(Tag *) * (nn ? nn : 1));
    s->ntag_refs = 0;
    for (u32 k = 0; k < nn; k++) {
      char slugbuf[72];
      u32 sl = slugify(names[k], slugbuf);
      Str slug = { adup(slugbuf, sl), sl };
      Tag *t = tag_find_or_create(b, names[k], slug, 0);
      if (!t) {
        t = (Tag *)azalloc(sizeof(Tag));
        t->name = names[k];
        t->slug = slug;
        t->url.s = "";
        t->url.n = 0;
        t->posts = NULL;
        t->nposts = 0;
      }
      s->tag_refs[s->ntag_refs++] = t;
    }
  }
  qsort(b->tags, b->ntags, sizeof(Tag *), cmp_tag_order);

  // 7. Layouts.
  T_tags += now_s() - _pt; _pt = now_s();
  dep_init();
  inc_cache_init(8192);
  strset_init(&b->out_set, 2048);
  SiteView site = { in->title, b->base };
  // allTags: {name, slug, url, count}
  Arr all_tags_arr;
  all_tags_arr.kind = ARR_TAG;
  all_tags_arr.obj_kind = OBJ_TAGCOUNT;
  all_tags_arr.items = b->tags;
  all_tags_arr.count = b->ntags;
  // recent: first `recent` posts as cards
  u32 nrecent = (u32)(in->recent < 0 ? 0 : in->recent);
  if (nrecent > nposts) nrecent = nposts;
  PageView **recent_cards = (PageView **)acalloc(sizeof(PageView *) * (nrecent ? nrecent : 1));
  PageView **post_cards = (PageView **)acalloc(sizeof(PageView *) * (nposts ? nposts : 1));
  for (u32 i = 0; i < nposts; i++) {
    PageView *cv = (PageView *)azalloc(sizeof(PageView));
    Source *p = posts[i];
    cv->src = p;
    cv->title = p->fm.title;
    cv->url = p->url;
    cv->date = p->fm.has_date ? p->fm.date : str_lit("");
    cv->date_long = p->date_long;
    cv->summary = p->fm.has_summary ? p->fm.summary : str_lit("");
    cv->excerpt = p->excerpt;
    cv->reading_minutes = p->reading_minutes;
    cv->tags = p->tag_refs;
    cv->ntags = p->ntag_refs;
    cv->is_card = 1;
    p->card = cv;
    post_cards[i] = cv;
    if (i < nrecent) recent_cards[i] = cv;
  }
  Arr recent_arr;
  recent_arr.kind = ARR_CARD;
  recent_arr.obj_kind = OBJ_CARD;
  recent_arr.items = recent_cards;
  recent_arr.count = nrecent;

  // Shared top-level vars for every render.
  Var top_vars[3];
  top_vars[0].name = str_lit("site");
  top_vars[0].v = v_obj(OBJ_SITE, &site);
  top_vars[1].name = str_lit("tags");
  top_vars[1].v = v_arr(&all_tags_arr);
  top_vars[2].name = str_lit("recent");
  top_vars[2].v = v_arr(&recent_arr);
  Frame top_frame = { top_vars, 3, NULL };

  for (u32 i = 0; i < nsrc; i++) {
    Source *s = &srcs[i];
    PageView pv;
    memset(&pv, 0, sizeof(pv));
    pv.src = s;
    pv.title = s->fm.title;
    pv.url = s->url;
    pv.date = s->fm.has_date ? s->fm.date : str_lit("");
    pv.date_long = s->date_long;
    pv.summary = s->fm.has_summary ? s->fm.summary : str_lit("");
    pv.excerpt = s->excerpt;
    pv.reading_minutes = s->reading_minutes;
    pv.tags = s->tag_refs;
    pv.ntags = s->ntag_refs;
    pv.slug = s->slug;
    pv.content = s->html;
    pv.toc = s->toc;
    pv.ntoc = s->ntoc;
    if (s->newer) {
      pv.has_newer = 1;
      pv.newer.title = s->newer->fm.title;
      pv.newer.url = s->newer->url;
      pv.newer.date = s->newer->fm.has_date ? s->newer->fm.date : str_lit("");
      pv.newer.date_long = s->newer->date_long;
    }
    if (s->older) {
      pv.has_older = 1;
      pv.older.title = s->older->fm.title;
      pv.older.url = s->older->url;
      pv.older.date = s->older->fm.has_date ? s->older->fm.date : str_lit("");
      pv.older.date_long = s->older->date_long;
    }
    pv.is_post = s->post;
    pv.is_card = 0;
    ParsedTemplate *tpl = tpl_find(&tctx, s->layout);
    if (!tpl) continue; // refused earlier; unreachable
    Var pv_vars[1];
    pv_vars[0].name = str_lit("page");
    pv_vars[0].v = v_obj(OBJ_PAGE, &pv);
    Frame fr = { pv_vars, 1, &top_frame };
    Buf ob = { 0, 0, 0 };
    buf_reserve(&ob, s->html.n + 4096);
    render_nodes(&tpl->nodes, &fr, &tctx, 0, &ob);
    char *pb = (char *)acalloc(s->out_dir.n + 12);
    memcpy(pb, s->out_dir.s, s->out_dir.n);
    memcpy(pb + s->out_dir.n, "/index.html", 11);
    Str opath = { pb, s->out_dir.n + 11 };
    Str otext = { ob.buf, (u32)ob.len };
    build_out(b, opath, otext);
  }
  // Index pages.
  long per = in->per_page;
  if (per < 1) per = 1;
  long npages = (long)((nposts + (u32)per - 1) / (u32)per);
  if (npages < 1) npages = 1;
  ParsedTemplate *index_tpl = tpl_find(&tctx, str_lit("index"));
  if (index_tpl) {
    for (long k = 1; k <= npages; k++) {
      u32 lo = (u32)((k - 1) * per);
      u32 hi = (u32)(k * per);
      if (hi > nposts) hi = nposts;
      Arr posts_arr;
      posts_arr.kind = ARR_CARD;
      posts_arr.obj_kind = OBJ_CARD;
      posts_arr.items = post_cards + lo;
      posts_arr.count = hi - lo;
      PagerView pager;
      pager.number = k;
      pager.total = npages;
      char ub1[512], ub2[512], ub3[512];
      int n1, n2, n3;
      if (k == 1) n1 = snprintf(ub1, sizeof(ub1), "%.*s/", (int)b->base.n, b->base.s);
      else n1 = snprintf(ub1, sizeof(ub1), "%.*s/page/%ld/", (int)b->base.n, b->base.s, k);
      if (k == 1) { ub2[0] = 0; n2 = 0; }
      else if (k - 1 == 1) n2 = snprintf(ub2, sizeof(ub2), "%.*s/", (int)b->base.n, b->base.s);
      else n2 = snprintf(ub2, sizeof(ub2), "%.*s/page/%ld/", (int)b->base.n, b->base.s, k - 1);
      if (k == npages) { ub3[0] = 0; n3 = 0; }
      else n3 = snprintf(ub3, sizeof(ub3), "%.*s/page/%ld/", (int)b->base.n, b->base.s, k + 1);
      pager.url.s = ub1; pager.url.n = (u32)n1;
      pager.newer_url.s = ub2; pager.newer_url.n = (u32)n2;
      pager.older_url.s = ub3; pager.older_url.n = (u32)n3;
      Var iv[2];
      iv[0].name = str_lit("posts");
      iv[0].v = v_arr(&posts_arr);
      iv[1].name = str_lit("pager");
      iv[1].v = v_obj(OBJ_PAGER, &pager);
      Frame fr = { iv, 2, &top_frame };
      Buf ob = { 0, 0, 0 };
      buf_reserve(&ob, (hi - lo) * 1024 + 4096);
      render_nodes(&index_tpl->nodes, &fr, &tctx, 0, &ob);
      Str opath;
      if (k == 1) opath = str_lit("index.html");
      else {
        char *pb = (char *)acalloc(32);
        int pn = snprintf(pb, 32, "page/%ld/index.html", k);
        opath.s = pb;
        opath.n = (u32)pn;
      }
      build_out(b, opath, (Str){ ob.buf, (u32)ob.len });
    }
  }
  // Tag pages.
  ParsedTemplate *tag_tpl = tpl_find(&tctx, str_lit("tag"));
  if (tag_tpl) {
    for (u32 i = 0; i < b->ntags; i++) {
      Tag *t = b->tags[i];
      Var tv[1];
      tv[0].name = str_lit("tag");
      tv[0].v = v_obj(OBJ_TAGFULL, t);
      Frame fr = { tv, 1, &top_frame };
      Buf ob = { 0, 0, 0 };
      buf_reserve(&ob, t->nposts * 1024 + 4096);
      render_nodes(&tag_tpl->nodes, &fr, &tctx, 0, &ob);
      char *pb = (char *)acalloc(t->slug.n + 20);
      int pn = snprintf(pb, t->slug.n + 20, "tags/%.*s/index.html", (int)t->slug.n, t->slug.s);
      build_out(b, (Str){ pb, (u32)pn }, (Str){ ob.buf, (u32)ob.len });
    }
  }
  // Tags index.
  ParsedTemplate *tags_tpl = tpl_find(&tctx, str_lit("tags"));
  if (tags_tpl) {
    Buf ob = { 0, 0, 0 };
    buf_reserve(&ob, b->ntags * 256 + 4096);
    render_nodes(&tags_tpl->nodes, &top_frame, &tctx, 0, &ob);
    build_out(b, str_lit("tags/index.html"), (Str){ ob.buf, (u32)ob.len });
  }
  // Static files.
  for (u32 i = 0; i < in->nfiles; i++) {
    InputFile *f = &in->files[i];
    if (ends_md(f->path)) continue;
    if (strset_has(&b->out_set, f->path, NULL)) {
      char msg[512];
      int n = snprintf(msg, sizeof(msg), "shadowed %.*s: a built page has its path",
                       (int)f->path.n, f->path.s);
      build_report(b, msg, (size_t)n);
    } else {
      build_out(b, f->path, f->content);
    }
  }
  // 8. Report.
  T_lay += now_s() - _pt; _pt = now_s();
  u32 nlines = b->nreport + b->broken.count;
  Str *lines = (Str *)acalloc(sizeof(Str) * (nlines ? nlines : 1));
  u32 w = 0;
  for (u32 i = 0; i < b->nreport; i++) lines[w++] = b->report[i];
  for (u32 i = 0; i < b->broken.cap; i++)
    if (b->broken.keys[i].s) lines[w++] = b->broken.keys[i];
  qsort(lines, w, sizeof(Str), cmp_strp);
  Buf rb = { 0, 0, 0 };
  for (u32 i = 0; i < w; i++) {
    buf_add(&rb, lines[i].s, lines[i].n);
    buf_addc(&rb, '\n');
  }
  u32 npages_count = 0;
  for (u32 i = 0; i < nsrc; i++)
    if (!srcs[i].post) npages_count++;
  u32 nstatics = 0;
  for (u32 i = 0; i < in->nfiles; i++)
    if (!ends_md(in->files[i].path)) nstatics++;
  char last[160];
  int ln = snprintf(last, sizeof(last), "built %u pages, %u posts, %u tags and %u static files",
                    npages_count, nposts, b->ntags, nstatics);
  buf_add(&rb, last, (u32)ln);
  buf_addc(&rb, '\n');
  build_out(b, str_lit("build-report.txt"), (Str){ rb.buf, (u32)rb.len });
  T_rep += now_s() - _pt;
}

// ---------------------------------------------------------------- JSON emit (answers.json)

// Copy n bytes (0 <= n <= 32) with overlapping vector/word stores —
// no memcpy call, so short runs stay cheap. The destination must have
// room for the run (callers guarantee it via their room checks).
// Marked target("avx2") so it inlines into the AVX2 JSON escaper (a
// plain static function is not inlined across target attributes, and
// the escaper calls this ~460k times per run).
__attribute__((target("avx2"))) static inline void copy_short(char *w, const char *s, size_t n) {
  if (n >= 16) {
    __m128i a = _mm_loadu_si128((const __m128i *)s);
    __m128i b = _mm_loadu_si128((const __m128i *)(s + n - 16));
    _mm_storeu_si128((__m128i *)w, a);
    _mm_storeu_si128((__m128i *)(w + n - 16), b);
  } else if (n >= 8) {
    uint64_t a, b;
    memcpy(&a, s, 8);
    memcpy(&b, s + n - 8, 8);
    memcpy(w, &a, 8);
    memcpy(w + n - 8, &b, 8);
  } else if (n >= 4) {
    uint32_t a, b;
    memcpy(&a, s, 4);
    memcpy(&b, s + n - 4, 4);
    memcpy(w, &a, 4);
    memcpy(w + n - 4, &b, 4);
  } else if (n >= 2) {
    uint16_t a, b;
    memcpy(&a, s, 2);
    memcpy(&b, s + n - 2, 2);
    memcpy(w, &a, 2);
    memcpy(w + n - 2, &b, 2);
  } else if (n == 1) {
    *w = *s;
  }
}

// Scalar JSON escape of s[i..n), then a closing '"'. Raw-pointer writes
// with a room check before every escape (runs reserve run+8 up front).
static void json_escape_tail(MBuf *b, const char *s, size_t i, size_t n) {
  char *w = b->buf + b->len;
  char *end = b->buf + b->cap;
  while (i < n) {
    size_t j = i;
    while (j < n) {
      unsigned char c = (unsigned char)s[j];
      if (c == '"' || c == '\\' || c < 0x20) break;
      j++;
    }
    size_t run = j - i;
    if (run) {
      if (w + run + 8 > end) {
        b->len = (size_t)(w - b->buf);
        mbuf_reserve(b, run + 8);
        w = b->buf + b->len;
        end = b->buf + b->cap;
      }
      memcpy(w, s + i, run);
      w += run;
    }
    if (j >= n) break;
    if (w + 8 > end) {
      b->len = (size_t)(w - b->buf);
      mbuf_reserve(b, 64);
      w = b->buf + b->len;
      end = b->buf + b->cap;
    }
    unsigned char c = (unsigned char)s[j];
    if (c == '"') { w[0] = '\\'; w[1] = '"'; w += 2; }
    else if (c == '\\') { w[0] = '\\'; w[1] = '\\'; w += 2; }
    else if (c == '\n') { w[0] = '\\'; w[1] = 'n'; w += 2; }
    else if (c == '\r') { w[0] = '\\'; w[1] = 'r'; w += 2; }
    else if (c == '\t') { w[0] = '\\'; w[1] = 't'; w += 2; }
    else if (c == '\b') { w[0] = '\\'; w[1] = 'b'; w += 2; }
    else if (c == '\f') { w[0] = '\\'; w[1] = 'f'; w += 2; }
    else w += sprintf(w, "\\u%04x", c);
    i = j + 1;
  }
  *w++ = '"';
  b->len = (size_t)(w - b->buf);
}

// Copy n bytes (0 <= n <= 64) with overlapping vector/word stores —
// copy_short handles at most 32, so a longer run is split in two.
// (The 64-byte escape loop passes runs up to 63 bytes.)
__attribute__((target("avx2"))) static inline void copy_span(char *w, const char *s, size_t n) {
  if (n <= 32) { copy_short(w, s, n); return; }
  copy_short(w, s, 32);
  copy_short(w + 32, s + 32, n - 32);
}

// AVX2 body: open quote, 64-byte chunks (clean chunks via two vector
// stores, chunks with specials byte-wise), then the scalar tail.
// 64 bytes per iteration halves the loop/room-check overhead versus
// 32, and the dirty path walks one combined 64-bit mask.
__attribute__((target("avx2")))
static void mbuf_json_escape_avx2(MBuf *b, const char *s, size_t n) {
  if (b->len + n + n / 4 + 64 > b->cap) mbuf_reserve(b, n + n / 4 + 64);
  char *w = b->buf + b->len;
  char *end = b->buf + b->cap;
  *w++ = '"';
  const __m256i qu = _mm256_set1_epi8('"');
  const __m256i bs = _mm256_set1_epi8('\\');
  const __m256i lim = _mm256_set1_epi8(0x20);
  const __m256i neg1 = _mm256_set1_epi8(-1);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    if (w + 64 * 6 + 2 > end) { // pathological: grow, recompute pointers
      b->len = (size_t)(w - b->buf);
      mbuf_reserve(b, 64 * 6 + 2);
      w = b->buf + b->len;
      end = b->buf + b->cap;
    }
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(s + i));
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(s + i + 32));
    __m256i m0 = _mm256_or_si256(_mm256_cmpeq_epi8(v0, qu), _mm256_cmpeq_epi8(v0, bs));
    __m256i lt0 = _mm256_cmpgt_epi8(lim, v0); // (signed) v < 0x20
    __m256i asc0 = _mm256_cmpgt_epi8(v0, neg1); // v >= 0 (ASCII)
    m0 = _mm256_or_si256(m0, _mm256_and_si256(lt0, asc0));
    __m256i m1 = _mm256_or_si256(_mm256_cmpeq_epi8(v1, qu), _mm256_cmpeq_epi8(v1, bs));
    __m256i lt1 = _mm256_cmpgt_epi8(lim, v1);
    __m256i asc1 = _mm256_cmpgt_epi8(v1, neg1);
    m1 = _mm256_or_si256(m1, _mm256_and_si256(lt1, asc1));
    u32 mask0 = (u32)_mm256_movemask_epi8(m0);
    u32 mask1 = (u32)_mm256_movemask_epi8(m1);
    if (!mask0 && !mask1) {
      _mm256_storeu_si256((__m256i *)w, v0);
      _mm256_storeu_si256((__m256i *)(w + 32), v1);
      w += 64;
      continue;
    }
    // Chunk has specials: copy the runs between them with copy_span
    // (no memcpy calls) and write the escapes directly.
    u64 mask = (u64)mask0 | ((u64)mask1 << 32);
    size_t j = i;
    while (mask) {
      int bit = __builtin_ctzll(mask);
      size_t pos = i + (size_t)bit;
      size_t run = pos - j;
      copy_span(w, s + j, run);
      w += run;
      unsigned char c = (unsigned char)s[pos];
      if (c == '"') { w[0] = '\\'; w[1] = '"'; w += 2; }
      else if (c == '\\') { w[0] = '\\'; w[1] = '\\'; w += 2; }
      else if (c == '\n') { w[0] = '\\'; w[1] = 'n'; w += 2; }
      else if (c == '\r') { w[0] = '\\'; w[1] = 'r'; w += 2; }
      else if (c == '\t') { w[0] = '\\'; w[1] = 't'; w += 2; }
      else if (c == '\b') { w[0] = '\\'; w[1] = 'b'; w += 2; }
      else if (c == '\f') { w[0] = '\\'; w[1] = 'f'; w += 2; }
      else w += sprintf(w, "\\u%04x", c);
      j = pos + 1;
      mask &= mask - 1;
    }
    size_t tail = i + 64 - j;
    copy_span(w, s + j, tail);
    w += tail;
  }
  b->len = (size_t)(w - b->buf);
  json_escape_tail(b, s, i, n);
}

static void mbuf_json_escape(MBuf *b, const char *s, size_t n) {
  if (cpu_avx2) { mbuf_json_escape_avx2(b, s, n); return; }
  if (b->len + n + n / 4 + 64 > b->cap) mbuf_reserve(b, n + n / 4 + 64);
  char *w = b->buf + b->len;
  *w++ = '"';
  b->len = (size_t)(w - b->buf);
  json_escape_tail(b, s, 0, n);
}

// ---------------------------------------------------------------- main

static char *read_all(const char *path, size_t *out_len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror(path); exit(1); }
  struct stat st;
  if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
  size_t len = (size_t)st.st_size;
  char *buf = (char *)malloc(len + 1);
  if (!buf) { perror("malloc"); exit(1); }
  size_t got = 0;
  while (got < len) {
    ssize_t r = read(fd, buf + got, len - got);
    if (r <= 0) { perror("read"); exit(1); }
    got += (size_t)r;
  }
  buf[len] = 0;
  close(fd);
  *out_len = len;
  return buf;
}

static void write_all(const char *path, const char *buf, size_t len) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) { perror(path); exit(1); }
  size_t off = 0;
  while (off < len) {
    ssize_t w = write(fd, buf + off, len - off);
    if (w <= 0) { perror("write"); exit(1); }
    off += (size_t)w;
  }
  if (close(fd) != 0) { perror("close"); exit(1); }
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: sitegen <cases.json> <answers.json>\n");
    return 1;
  }
  int timing = getenv("SITEGEN_TIME") != NULL;
  double t0 = now_s();
  cpu_init();
  esc_init();
  arena_init();
  size_t in_len;
  char *in_buf = read_all(argv[1], &in_len);
  double t_read = now_s();
  u32 ncases = 0;
  CaseInput *cases = parse_cases(in_buf, in_len, &ncases);
  double t_parse = now_s();
  size_t base_mark = arena_pos;
  MBuf answers = { 0, 0, 0 };
  // Pre-size: answers run ~3.5x the input size; this avoids all
  // doubling copies of the ~10MB buffer (doubling would copy ~2x extra).
  mbuf_reserve(&answers, in_len * 4 + ((size_t)1 << 20));
  mbuf_add(&answers, "[", 1);
  Build b;
  double t_build = 0, t_emit = 0;
  for (u32 ci = 0; ci < ncases; ci++) {
    arena_pos = base_mark;
    if (ci) mbuf_add(&answers, ",", 1);
    double tb0 = now_s();
    build_case(&cases[ci], &b);
    double tb1 = now_s();
    t_build += tb1 - tb0;
    mbuf_add(&answers, "{", 1);
    for (u32 i = 0; i < b.nout; i++) {
      if (i) mbuf_add(&answers, ",", 1);
      mbuf_json_escape(&answers, b.out[i].path.s, b.out[i].path.n);
      mbuf_add(&answers, ":", 1);
      mbuf_json_escape(&answers, b.out[i].text.s, b.out[i].text.n);
    }
    mbuf_add(&answers, "}", 1);
    t_emit += now_s() - tb1;
    if (timing) fprintf(stderr, "case %u: nfiles=%u nsrc=%u nposts=%u ntags=%u nout=%u build=%.2fms emit=%.2fms arena=%zuMB\n",
                        ci, cases[ci].nfiles, b.nsources, b.nposts, b.ntags, b.nout,
                        (tb1 - tb0) * 1e3, (now_s() - tb1) * 1e3, (arena_pos - base_mark) >> 20);
  }
  mbuf_add(&answers, "]", 1);
  double t_emit_end = now_s();
  write_all(argv[2], answers.buf, answers.len);
  double t_end = now_s();
  if (timing) fprintf(stderr, "read=%.2fms parse=%.2fms build=%.2fms emit=%.2fms write=%.2fms total=%.2fms answers=%zuMB\n",
                      (t_read - t0) * 1e3, (t_parse - t_read) * 1e3, t_build * 1e3,
                      t_emit * 1e3, (t_end - t_emit_end) * 1e3, (t_end - t0) * 1e3, answers.len >> 20);
  if (timing) fprintf(stderr, "phases: tpl=%.2f src=%.2f perm=%.2f md=%.2f coll=%.2f tags=%.2f lay=%.2f rep=%.2f (ms)\n",
                      T_tpl * 1e3, T_src * 1e3, T_perm * 1e3, T_md * 1e3,
                      T_coll * 1e3, T_tags * 1e3, T_lay * 1e3, T_rep * 1e3);
  if (timing) fprintf(stderr, "counts: md_inline=%ld md_link=%ld bracket=%ld | md_render=%.2f words=%.2f (ms)\n",
                      CNT_md_inline, CNT_md_link, CNT_bracket, T_md_render * 1e3, T_words * 1e3);
  return 0;
}
