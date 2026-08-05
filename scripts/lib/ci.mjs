// Shared helpers for the CI engine smokes: HF weight download with a disk cache
// (.ci-cache/, restored by actions/cache) and a 16-bit mono WAV reader.
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const CACHE = fileURLToPath(new URL("../../.ci-cache/", import.meta.url));

/** Download (or reuse cached) HF file → Uint8Array. Retries transient failures
 * (HF 429/5xx from shared runner IPs) and verifies Content-Length when given, so
 * a truncated/HTML body can't poison the cache. */
export async function hfGet(repo, path) {
  const dest = join(CACHE, repo, path);
  if (!existsSync(dest) || statSync(dest).size === 0) {
    mkdirSync(dirname(dest), { recursive: true });
    const url = `https://huggingface.co/${repo}/resolve/main/${path}`;
    console.log(`  fetching ${repo}/${path} ...`);
    let lastErr;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (attempt) await new Promise((r) => setTimeout(r, 2000 * 2 ** attempt));
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HF fetch ${url} → ${res.status}`);
        const expect = Number(res.headers.get("content-length") || 0);
        await pipeline(Readable.fromWeb(res.body), createWriteStream(dest + ".part"));
        const got = statSync(dest + ".part").size;
        if (expect && got !== expect) throw new Error(`short read ${got}/${expect} for ${path}`);
        renameSync(dest + ".part", dest);
        lastErr = null;
        break;
      } catch (e) {
        lastErr = e;
        rmSync(dest + ".part", { force: true });
        console.warn(`  retry ${attempt + 1}/3: ${e.message}`);
      }
    }
    if (lastErr) throw lastErr;
  }
  return new Uint8Array(readFileSync(dest));
}

export const hfF32 = async (repo, path) => {
  const u = await hfGet(repo, path);
  return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4);
};
export const hfJson = async (repo, path) => JSON.parse(new TextDecoder().decode(await hfGet(repo, path)));
export const hfText = async (repo, path) => new TextDecoder().decode(await hfGet(repo, path));

/** Minimal 16-bit mono WAV reader (chunk-scanned). */
export function readWav(p) {
  const b = readFileSync(p);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12,
    dO = -1,
    dL = 0;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const s = dv.getUint32(o + 4, true);
    if (id === "data") {
      dO = o + 8;
      dL = s;
      break;
    }
    o += 8 + s + (s & 1);
  }
  const n = dL / 2,
    out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i * 2, true) / 32768;
  return out;
}

export function assert(cond, msg) {
  if (!cond) {
    console.error(`ASSERT FAILED: ${msg}`);
    process.exit(1);
  }
  console.log(`  ok: ${msg}`);
}
