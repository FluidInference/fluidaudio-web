// Fetch model files from Hugging Face and persist them via the Cache API so the
// (often hundreds of MB) weights are downloaded once per browser. Reports byte
// progress for the UI.

import type { ProgressCb } from "./types";

const CACHE_NAME = "fluidaudio-web-models-v1";
const HF_BASE = "https://huggingface.co";

/** Resolve `repo` + `path` to the HF `resolve/main` URL. */
export function hfUrl(repo: string, path: string, revision = "main"): string {
  return `${HF_BASE}/${repo}/resolve/${revision}/${path}`;
}

/** Fetch a single URL as bytes, using the Cache API and streaming progress. */
export async function fetchCached(
  url: string,
  onProgress?: ProgressCb,
  label = url
): Promise<Uint8Array> {
  // Cache API is best-effort: opening/matching/putting can throw (quota, or the
  // per-entry size limit — a 600 MB weight file exceeds it in Chrome), and that
  // must never fail the load.
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null;
  }

  // referrerPolicy no-referrer: HF hotlink-protects some hosts (e.g. *.workers.dev)
  // by returning 404 when the Referer is theirs → surfaces as a CORS error. The
  // page-level <meta name="referrer"> covers third-party libs; this covers ours.
  const res = await fetch(url, { referrerPolicy: "no-referrer" });
  if (!res.ok || !res.body) throw new Error(`fetch ${url} → ${res.status}`);

  const total = Number(res.headers.get("content-length") || 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.({ file: label, loaded, total, fraction: total ? loaded / total : 0 });
  }

  const bytes = concat(chunks, loaded);
  // Store a fresh Response so subsequent loads are instant — best-effort: Chrome's
  // Cache API rejects entries beyond a few hundred MB ("Failed to execute 'put'"),
  // so on failure we just skip caching (re-download next time) rather than fail.
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "content-length": String(loaded) } }));
    } catch {
      /* too large to cache — fine, keep going uncached */
    }
  }
  return bytes;
}

/** Fetch many files, aggregating progress across the set. */
export async function fetchAll(
  files: { repo: string; path: string; revision?: string }[],
  onProgress?: ProgressCb
): Promise<Map<string, Uint8Array>> {
  const out = new Map<string, Uint8Array>();
  let doneBytes = 0;
  // Rough overall fraction: weight each file equally in count, refine by bytes.
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    const url = hfUrl(f.repo, f.path, f.revision);
    const bytes = await fetchCached(url, (p) => {
      onProgress?.({
        file: f.path,
        loaded: doneBytes + p.loaded,
        total: 0,
        fraction: (i + p.fraction) / files.length,
      });
    }, f.path);
    doneBytes += bytes.byteLength;
    out.set(f.path, bytes);
  }
  return out;
}

export async function clearModelCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}

function concat(chunks: Uint8Array[], total: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}
