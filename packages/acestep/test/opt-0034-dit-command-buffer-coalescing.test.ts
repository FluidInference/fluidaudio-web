import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { parseOpt0034ThermalGate } from
  "./browser/opt-0034-dit-command-buffer-coalescing-contract.js";

const GRAPH_SOURCE = source("../src/webgpu/dit-graph.ts");
const BACKEND_SOURCE = source("../src/webgpu/dit-backend.ts");
const PIPELINE_SOURCE = source("../src/runtime/webgpu-pipeline.ts");
const WORKER_SOURCE = source(
  "./browser/opt-0034-dit-command-buffer-coalescing-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0034-dit-command-buffer-coalescing.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0034-dit-command-buffer-coalescing.html",
);
const THERMAL_CONTRACT_SOURCE = source(
  "./browser/opt-0034-dit-command-buffer-coalescing-contract.ts",
);
const RECORD_SOURCE = source(
  "../optimization/experiments/OPT-0034-dit-command-buffer-coalescing.md",
);
const LEDGER_SOURCE = source("../optimization/LEDGER.md");

describe("OPT-0034 DiT command-buffer coalescing contract", () => {
  it("keeps the production default at batch1 and restricts the experiment arms", () => {
    expect(GRAPH_SOURCE).toContain(
      "export const ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER = 1 as const",
    );
    expect(GRAPH_SOURCE).toContain(
      "export type AceDitPhysicalQuantaPerCommandBuffer = 1 | 8 | 16",
    );
    expect(GRAPH_SOURCE).toContain(
      "ACE DiT physical quanta per command buffer must be 1, 8, or 16",
    );
    expect(GRAPH_SOURCE).toContain("planAceDitPhysicalQuantumBatches(");
    expect(GRAPH_SOURCE).toContain("commandBufferCount: batches.length");
    expect(GRAPH_SOURCE).toContain("ACE DiT scheduler lost physical FIFO position");
    expect(GRAPH_SOURCE).toContain("command.sequence.encodeQuantum(");
    expect(GRAPH_SOURCE).toContain("options.signal.throwIfAborted()");
  });

  it("captures exact batch telemetry without enabling OPT-0018 simultaneously", () => {
    expect(BACKEND_SOURCE).toContain(
      'schema: "ace-dit-opt0034-command-buffer-coalescing-v1"',
    );
    expect(BACKEND_SOURCE).toContain("physicalGraphQuantumCount: 2_553");
    expect(BACKEND_SOURCE).toContain("maximumBatchSubmitThroughDrainMs");
    expect(BACKEND_SOURCE).toContain("descriptorTableSha256");
    expect(BACKEND_SOURCE).toContain(
      "ACE DiT command-attribution captures are mutually exclusive",
    );
    expect(PIPELINE_SOURCE).toContain("opt0034DitRun?: AceOpt0034DitRunOptions");
    expect(PIPELINE_SOURCE).toContain(
      '"ACE_DIT_OPT0034_SCHEDULING_PROFILE"',
    );
    expect(PIPELINE_SOURCE).toContain(
      'schema: "ace-dit-opt0034-m2250-checkpoint-v1"',
    );
  });

  it("precomputes batch membership and keeps every timed drain callback bounded", () => {
    expect(GRAPH_SOURCE).toContain(
      "createAceDitPhysicalCommandBatchDescriptorTables(",
    );
    expect(GRAPH_SOURCE).toContain(
      "this.physicalCommandBatchDescriptors =",
    );
    expect(GRAPH_SOURCE).toContain(
      "const descriptor = batchDescriptors?.[",
    );
    expect(BACKEND_SOURCE).toContain(
      "const timings = new Float64Array(batchPlans.length)",
    );
    expect(BACKEND_SOURCE).toContain(
      "timings[descriptor.batchIndex] = submitThroughDrainMs",
    );
    const captureStart = BACKEND_SOURCE.indexOf(
      "const timings = new Float64Array(batchPlans.length)",
    );
    const callbackStart = BACKEND_SOURCE.indexOf(
      "record(\n      descriptor: AceDitPhysicalCommandBatchDescriptor",
      captureStart,
    );
    const callbackEnd = BACKEND_SOURCE.indexOf(
      "finish(\n      timing: Opt0034CaptureFinishTiming",
      callbackStart,
    );
    expect(captureStart).toBeGreaterThan(0);
    expect(callbackStart).toBeGreaterThan(captureStart);
    expect(callbackEnd).toBeGreaterThan(callbackStart);
    const timedCallbackSource = BACKEND_SOURCE.slice(
      callbackStart,
      callbackEnd,
    );
    expect(timedCallbackSource).not.toContain(".slice(");
    expect(timedCallbackSource).not.toContain(".reduce(");
    expect(timedCallbackSource).not.toContain("Object.freeze({");
  });

  it("runs one authenticated batch1/batch8/batch16 comparison and gates exactness", () => {
    expect(WORKER_SOURCE).toContain(
      "const ARMS = Object.freeze([1, 8, 16] as const)",
    );
    expect(WORKER_SOURCE).toContain("PHYSICAL_GRAPH_QUANTA = 2_553");
    expect(WORKER_SOURCE).toContain("FINAL_LATENT_ELEMENTS = 288_000");
    expect(WORKER_SOURCE).toContain("REQUIRED_COMPLETE_STAGE_SPEEDUP = 1.10");
    expect(WORKER_SOURCE).toContain("opt0034DitRun: Object.freeze(");
    expect(WORKER_SOURCE).toContain("controller.abort(PRIVATE_DIT_STOP)");
    expect(WORKER_SOURCE).toContain("compareFinalLatents(arms)");
    expect(WORKER_SOURCE).toContain(
      'comparisonKind: "complete-detached-final-latent-raw-u32"',
    );
    expect(WORKER_SOURCE).toContain("mismatchCount === 0");
    expect(WORKER_SOURCE).toContain("vaeWeightAcquireStarted: false");
    expect(WORKER_SOURCE).toContain("productionDefaultChanged: false");
    expect(WORKER_SOURCE).toContain("under60SecondClaim: false");
    expect(WORKER_SOURCE).not.toContain("timestamp-query");
  });

  it("requires an external nominal gate only after preparation", () => {
    expect(PAGE_SOURCE).toContain("parseOpt0034ThermalGate(");
    expect(PAGE_SOURCE).toContain("readyAtEpochMilliseconds");
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0034_RESULT__ = receipt");
    expect(THERMAL_CONTRACT_SOURCE).toContain(
      'protocol: "wait-30s-then-one-level0-check"',
    );
    expect(THERMAL_CONTRACT_SOURCE).toContain("observationCount !== 1");
    expect(THERMAL_CONTRACT_SOURCE).toContain("observedLevel !== 0");
    expect(THERMAL_CONTRACT_SOURCE).not.toContain("pollMilliseconds");
    expect(HTML_SOURCE).toContain('id="thermal-gate" disabled');
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(HTML_SOURCE).toContain("nominal 30-second wait");
    expect(HTML_SOURCE).toContain("1, 8, then 16");
  });

  it("accepts exactly one level-0 check after 30 seconds and rejects fiction", () => {
    const valid = thermalParameters({
      started: 10_000,
      checked: 40_001,
      observations: 1,
      level: 0,
      gap: "",
    });
    expect(parseOpt0034ThermalGate(valid, 9_000, 40_001)).toEqual({
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
    expect(() => parseOpt0034ThermalGate(
      thermalParameters({
        started: 10_000,
        checked: 40_001,
        observations: 31,
        level: 0,
        gap: "30_001",
      }),
      9_000,
      40_001,
    )).toThrow(/one truthful level-0/);
    expect(() => parseOpt0034ThermalGate(
      thermalParameters({
        started: 10_000,
        checked: 40_001,
        observations: 1,
        level: 1,
        gap: "30_001",
      }),
      9_000,
      40_001,
    )).toThrow(/one truthful level-0/);
    expect(() => parseOpt0034ThermalGate(
      thermalParameters({
        started: 10_000,
        checked: 39_999,
        observations: 1,
        level: 0,
        gap: "29_999",
      }),
      9_000,
      39_999,
    )).toThrow(/one truthful level-0/);
  });

  it("was registered before evidence and preserves the exact decision gate", () => {
    expect(RECORD_SOURCE).toContain("# OPT-0034");
    expect(RECORD_SOURCE).toContain("Evidence: `inconclusive`");
    expect(RECORD_SOURCE).toContain("Compare the shipped batch of one");
    expect(RECORD_SOURCE).toContain("raw-U32 identity of the final latent");
    expect(RECORD_SOURCE).toContain("at least `1.10x`");
    expect(LEDGER_SOURCE).toContain("| OPT-0034 | DiT submission batching |");
    expect(LEDGER_SOURCE).toContain(
      "batch1/8/16 reduced graph command buffers/drains `2,553 -> 320 -> 160`",
    );
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
