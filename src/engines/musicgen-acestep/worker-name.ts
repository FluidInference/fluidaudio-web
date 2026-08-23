// Shared name for the ACE inference worker (used by the page wiring in
// src/music.ts and by AceStepMusicClient).
//
// TEST-ONLY overrides threaded to the worker through its name:
// - `?aceTestMaxGpuBufferBytes=N` caps the WebGPU buffer limits, so desktop
//   Chrome can emulate iOS adapters (N=1073741824).
// - `?aceTestDisableSubgroups=1` hides the `subgroups` feature (and the
//   fixed-subgroup-size adapter info), so desktop Chrome can emulate
//   Safari/Firefox adapters and exercise the portable kernel path.
// See worker.ts for the facade.
export function aceInferenceWorkerName(): string {
  const search = new URLSearchParams(location.search);
  const flags = new URLSearchParams();
  const rawCap = search.get("aceTestMaxGpuBufferBytes");
  if (rawCap !== null) {
    const value = Number(rawCap);
    if (Number.isSafeInteger(value) && value > 0) {
      flags.set("testMaxGpuBufferBytes", String(value));
    }
  }
  if (search.get("aceTestDisableSubgroups") === "1") {
    flags.set("testDisableSubgroups", "1");
  }
  const query = flags.toString();
  return query === "" ? "ace-step-inference" : `ace-step-inference?${query}`;
}
