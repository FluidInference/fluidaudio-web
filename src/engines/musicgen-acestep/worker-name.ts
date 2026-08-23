// Shared name for the ACE inference worker (used by the page wiring in
// src/music.ts and by AceStepMusicClient).
//
// TEST-ONLY override: `?aceTestMaxGpuBufferBytes=N` on the page URL threads a
// WebGPU buffer-limit cap to the worker through its name, so desktop Chrome
// can emulate iOS adapters (N=1073741824). See worker.ts for the facade.
export function aceInferenceWorkerName(): string {
  const raw = new URLSearchParams(location.search).get("aceTestMaxGpuBufferBytes");
  if (raw === null) return "ace-step-inference";
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) return "ace-step-inference";
  return `ace-step-inference?testMaxGpuBufferBytes=${value}`;
}
