// Shared helpers for the CI engine smokes: HF weight download with a disk cache
// (.ci-cache/, restored by actions/cache) and a 16-bit mono WAV reader.
import { createWriteStream, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { dirname, join } from "node:path";

const CACHE = new URL("../../.ci-cache/", import.meta.url).pathname;

/** Download (or reuse cached) HF file → Uint8Array. */
export async function hfGet(repo, path) {
  const dest = join(CACHE, repo, path);
  if (!existsSync(dest) || statSync(dest).size === 0) {
    mkdirSync(dirname(dest), { recursive: true });
    const url = `https://huggingface.co/${repo}/resolve/main/${path}`;
    console.log(`  fetching ${repo}/${path} ...`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HF fetch ${url} → ${res.status}`);
    await pipeline(Readable.fromWeb(res.body), createWriteStream(dest + ".part"));
    const { renameSync } = await import("node:fs");
    renameSync(dest + ".part", dest);
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
