// Text normalization (FluidInference/text-processing-rs, vendored wasm build).
// Two directions, both pure WASM (no network beyond the 1MB module):
//   TN  (written → spoken):  tnNormalizeSentenceLang("It costs $4.50", "en")
//                            → "It costs four dollars fifty cents"   (TTS input)
//   ITN (spoken → written):  normalizeSentence("i paid twenty one dollars")
//                            → "i paid $21"                          (ASR output)
// Only the entry points verified production-quality are exposed: en TN and
// en ITN (de/fr sentence TN drops surrounding context upstream, zh TN mangles
// CJK input — retested before widening).

let mod: any = null;
let loading: Promise<any> | null = null;

async function init(): Promise<any> {
  const m = await import("../vendor/text-processing/text_processing_rs.js");
  if (typeof window === "undefined" && typeof importScripts === "undefined") {
    // node (gates): feed bytes directly — the glue's fetch(file URL) path doesn't work there.
    // @ts-expect-error node types absent in the browser tsconfig; branch never runs in-browser
    const { readFileSync } = await import("node:fs");
    // @ts-expect-error see above
    const { fileURLToPath } = await import("node:url");
    await m.default({ module_or_path: readFileSync(fileURLToPath(new URL("../vendor/text-processing/text_processing_rs_bg.wasm", import.meta.url))) });
  } else {
    const url = (await import("../vendor/text-processing/text_processing_rs_bg.wasm?url")).default;
    await m.default({ module_or_path: url });
  }
  return m;
}

/** Load (once) and return the wasm module; null if it fails (callers degrade gracefully). */
export async function loadTextNorm(): Promise<any | null> {
  if (mod) return mod;
  loading = loading ?? init().then((m) => (mod = m));
  try {
    return await loading;
  } catch (e) {
    console.warn("[textnorm] wasm unavailable, text passes through unnormalized:", e);
    loading = null;
    return null;
  }
}

/** Written → spoken (TTS input), English. Pass-through if wasm unavailable. */
export function tnEnglish(m: any | null, text: string): string {
  return m ? m.tnNormalizeSentenceLang(text, "en") : text;
}

/** Spoken → written (ASR output), English rules; near no-op on other languages. */
export function itn(m: any | null, text: string): string {
  return m ? m.normalizeSentence(text) : text;
}
