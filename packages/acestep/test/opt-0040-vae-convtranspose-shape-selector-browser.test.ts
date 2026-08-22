import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { parseOpt0040ThermalGate } from
  "./browser/opt-0040-vae-convtranspose-shape-selector-contract.js";

const WORKER_SOURCE = source(
  "./browser/opt-0040-vae-convtranspose-shape-selector-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0040-vae-convtranspose-shape-selector.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0040-vae-convtranspose-shape-selector.html",
);
const THERMAL_SOURCE = source(
  "./browser/opt-0040-vae-convtranspose-shape-selector-contract.ts",
);
const BACKEND_SOURCE = source("../src/webgpu/vae-fp16-backend.ts");

describe("OPT-0040 authenticated C512 browser gate", () => {
  it("reuses the authenticated revision-6 fixture with independent backend owners", () => {
    expect(WORKER_SOURCE).toContain(
      '"eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899"',
    );
    expect(WORKER_SOURCE).toContain(
      '"893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47"',
    );
    expect(WORKER_SOURCE).toContain(
      'expectedManifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256',
    );
    expect(WORKER_SOURCE).toContain("converterRevision");
    expect(WORKER_SOURCE).toContain("independentOwnedWeightPhases: true");
    expect(WORKER_SOURCE).toContain(
      '"opt-0028-mixed-fp16-fixed32-exact-packed-v1"',
    );
    expect(WORKER_SOURCE).toContain(
      '"opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"',
    );
  });

  it("pins identical batch64 scheduling and the complete C512 topology", () => {
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER",
    );
    expect(WORKER_SOURCE).toContain("const DECODER_COMMAND_BUFFER_COUNT = 123");
    expect(WORKER_SOURCE).toContain("const TOTAL_COMMAND_BUFFER_COUNT = 124");
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT",
    );
    expect(WORKER_SOURCE).toContain(
      "[ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID]: 644",
    );
    expect(WORKER_SOURCE).toContain(
      "[ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID]: 368",
    );
    expect(WORKER_SOURCE).toContain(
      "[ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID]: 276",
    );
    for (const expected of [
      '["block-0-conv-t1", 92,',
      '["block-1-conv-t1", 138,',
      '["block-2-conv-t1", 138,',
      '["block-3-conv-t1", 138,',
      '["block-4-conv-t1", 138,',
    ]) expect(WORKER_SOURCE).toContain(expected);
    expect(WORKER_SOURCE).toContain("identicalNonConvTransposeOwners: true");
    expect(BACKEND_SOURCE).toContain("captureDispatchTopology()");
    expect(BACKEND_SOURCE).toContain("operationQuantumCounts");
  });

  it("finishes exactness and deterministic reruns before enabling timing", () => {
    expect(WORKER_SOURCE).toContain(
      'export const OPT_0040_WARMUP_ORDER = Object.freeze([\n  "control",\n  "selector",\n  "selector",\n  "control",',
    );
    expect(WORKER_SOURCE).toContain("controlDeterminism");
    expect(WORKER_SOURCE).toContain("selectorDeterminism");
    expect(WORKER_SOURCE).toContain("crossArm");
    expect(WORKER_SOURCE).toContain("new Uint32Array(");
    expect(WORKER_SOURCE).toContain("new Uint16Array(");
    expect(WORKER_SOURCE).toContain("u32MismatchCount === 0");
    expect(WORKER_SOURCE).toContain("u16MismatchCount === 0");
    expect(WORKER_SOURCE).toContain(
      '"complete-final-fp32-byte-pattern-raw-u32-and-u16-views"',
    );
    expect(WORKER_SOURCE).toContain(
      "comparedU32WordsPerExecution: OPT_0040_C512_OUTPUT_ELEMENTS",
    );
    expect(WORKER_SOURCE).toContain(
      "comparedU16WordsPerExecution: OPT_0040_C512_OUTPUT_U16_WORDS",
    );
    expect(PAGE_SOURCE).toContain('message.type === "ready-for-thermal-gate"');
  });

  it("uses balanced pairs and gates family plus complete decoder wall", () => {
    expect(WORKER_SOURCE).toContain(
      'export const OPT_0040_TIMED_ORDER = Object.freeze([\n  "control",\n  "selector",\n  "selector",\n  "control",',
    );
    expect(WORKER_SOURCE).toContain(
      "OPT_0040_REQUIRED_CONV_TRANSPOSE_SPEEDUP = 1.10",
    );
    expect(WORKER_SOURCE).toContain(
      'profile.families["conv-transpose1d"]',
    );
    expect(WORKER_SOURCE).toContain("convTransposeFamilySpeedup");
    expect(WORKER_SOURCE).toContain("completeDecoderNoRegressionPassed");
    expect(WORKER_SOURCE).toContain("outerWindowSpeedup");
    expect(WORKER_SOURCE).toContain("forward.passed && reverse.passed");
    expect(WORKER_SOURCE).toContain("outerWallReportedButNotGating: true");
  });

  it("runs cancellation only after timing and cleans both owners idempotently", () => {
    expect(WORKER_SOURCE).toContain(
      'postProgress("running post-timing batch64 cancellation probe")',
    );
    expect(WORKER_SOURCE).toContain("active.abortAfterFirstProgress.abort(");
    expect(WORKER_SOURCE).toContain("observed.progressEventCount === 1");
    expect(WORKER_SOURCE).toContain(
      "progress.completedDecoderQuanta === CANCEL_AFTER_QUANTA",
    );
    expect(WORKER_SOURCE).toContain("backendRemainedLiveAfterPerCallAbort: true");
    expect(WORKER_SOURCE).toContain("idempotentDestroyPromises");
    expect(WORKER_SOURCE).toContain("bothBackendOwnersDestroyed");
    expect(WORKER_SOURCE).toContain("deviceContextDestroyed: true");
  });

  it("accepts one truthful level-0 observation after 30 seconds", () => {
    const valid = thermalParameters({
      started: 10_000,
      checked: 40_001,
      observations: 1,
      level: 0,
      gap: "",
    });
    expect(parseOpt0040ThermalGate(valid, 9_999, 40_001)).toEqual({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      protocol: "wait-30s-then-one-level0-check",
      startedAtEpochMilliseconds: 10_000,
      checkedAtEpochMilliseconds: 40_001,
      durationMilliseconds: 30_001,
      observationCount: 1,
      observedLevel: 0,
      maximumObservationGapMilliseconds: 30_001,
    });
    for (const invalid of [
      thermalParameters({
        started: 10_000,
        checked: 40_001,
        observations: 31,
        level: 0,
        gap: "30_001",
      }),
      thermalParameters({
        started: 10_000,
        checked: 40_001,
        observations: 1,
        level: 1,
        gap: "30_001",
      }),
      thermalParameters({
        started: 10_000,
        checked: 39_999,
        observations: 1,
        level: 0,
        gap: "29_999",
      }),
    ]) {
      expect(() => parseOpt0040ThermalGate(invalid, 9_999, 40_001))
        .toThrow(/one truthful level-0/);
    }
    expect(THERMAL_SOURCE).toContain("observationCount !== 1");
    expect(THERMAL_SOURCE).not.toContain("pollMilliseconds");
    expect(HTML_SOURCE).toContain("leave the machine idle for 30 seconds");
    expect(HTML_SOURCE).toContain('id="thermal-gate" disabled');
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(PAGE_SOURCE).toContain("thermalStarted.value = String(Date.now())");
  });

  it("requires explicit run identity and produces a downloadable receipt", () => {
    expect(PAGE_SOURCE).toContain("parseOpt0018RunIdentity(");
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0040_RESULT__ = receipt");
    expect(PAGE_SOURCE).toContain("URL.createObjectURL(new Blob(");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0040-vae-convtranspose-shape-selector.ts"',
    );
    expect(WORKER_SOURCE).toContain("productionDefaultChanged: false");
    expect(WORKER_SOURCE).toContain("under60SecondClaim: false");
    expect(WORKER_SOURCE).not.toContain("timestamp-query");
  });
});

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

function thermalParameters(values: Readonly<{
  started: number;
  checked: number;
  observations: number;
  level: number;
  gap: string;
}>): URLSearchParams {
  return new URLSearchParams({
    thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
    thermalStartedAtEpochMilliseconds: String(values.started),
    thermalCheckedAtEpochMilliseconds: String(values.checked),
    thermalObservations: String(values.observations),
    thermalObservedLevel: String(values.level),
    thermalMaximumObservationGapMilliseconds: values.gap,
  });
}
