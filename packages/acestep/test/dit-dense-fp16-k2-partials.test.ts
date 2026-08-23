import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID,
  ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT,
  AceOpt0074DenseK2PartialsKernel,
  aceOpt0074DenseK2PartialsWgsl,
  planAceOpt0032DenseK4Partials,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0074-dit-dense-fp16-k2-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0074-dit-dense-fp16-k2-partials.html",
  import.meta.url,
), "utf8");

describe("OPT-0074 dense FP16 K2 partials", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("reuses the authenticated K4 physical plan for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0032DenseK4Partials({
      rows: 2_250,
      inner,
      columns,
    });
    expect(ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT).toBe(
      ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
    );
    expect(plan).toMatchObject({
      tileRows: 32,
      tileColumns: 128,
      tileInner: 4,
      workgroupSize: 128,
      subgroupSize: 32,
      rowTiles: 71,
      columnTiles: columns / 128,
      innerK4Groups: inner / 4,
    });
  });

  it("widens two consecutive FP16 K2 dots in increasing pair order", () => {
    const wgsl = aceOpt0074DenseK2PartialsWgsl({
      rows: 2_250,
      inner: 2_048,
      columns: 2_048,
    });
    expect(ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID).toMatch(
      /k2-partials-fixed32-wg128-m32-n128-v1$/,
    );
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("weight: array<vec4<f16>>");
    expect(wgsl.match(/subgroupBroadcast\(lane_a, [0-7]u\)/g)).toHaveLength(8);
    expect(wgsl.match(/dot\(a0\.xy, b[0-3]\.xy\)/g)).toHaveLength(4);
    expect(wgsl.match(/dot\(a0\.zw, b[0-3]\.zw\)/g)).toHaveLength(4);
    expect(wgsl).toContain("let partial0_01 = vec4<f16>(");
    expect(wgsl).toContain("acc0 = acc0 + vec4<f32>(partial0_01)");
    expect(wgsl).toContain("let partial0_23 = vec4<f16>(");
    expect(wgsl).toContain("acc0 = acc0 + vec4<f32>(partial0_23)");
    expect(wgsl.indexOf("partial0_01")).toBeLessThan(
      wgsl.indexOf("partial0_23"),
    );
    expect(wgsl).not.toContain("var<workgroup>");
    expect(wgsl).not.toContain("workgroupBarrier");
    expect(wgsl).not.toMatch(/var acc\d+\s*=\s*vec4<f16>/);
  });

  it("fails closed without fixed32 shader-f16 subgroup support", () => {
    const base = {
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
      },
    };
    const missingF16 = {
      ...base,
      features: new Set(["subgroups"]),
    } as unknown as GPUDevice;
    expect(() => AceOpt0074DenseK2PartialsKernel.create(missingF16, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    })).toThrow(/shader-f16 and fixed 32-lane subgroups/);

    const variableSubgroup = {
      ...base,
      features: new Set(["shader-f16", "subgroups"]),
    } as unknown as GPUDevice;
    expect(() => AceOpt0074DenseK2PartialsKernel.create(variableSubgroup, {
      subgroupMinSize: 4,
      subgroupMaxSize: 64,
    })).toThrow(/fixed 32-lane/);
  });

  it("remains benchmark-only and absent from production selectors", () => {
    expect(KERNEL_SOURCE).toContain("Benchmark-only K2 arithmetic owner");
    for (const productionSource of [
      "../src/webgpu/ace-dit.ts",
      "../src/webgpu/dit-backend.ts",
      "../src/webgpu/dit-graph.ts",
      "../src/runtime/protocol.ts",
      "../demo/worker.ts",
    ]) {
      const source = readFileSync(new URL(productionSource, import.meta.url), "utf8");
      expect(source).not.toContain("OPT_0074");
      expect(source).not.toContain("opt-0074");
    }
  });

  it("screens exact, K2, and K4 over full and adversarial fixtures", () => {
    expect(HARNESS_SOURCE).toContain('type Arm = "exact" | "k2" | "k4"');
    expect(HARNESS_SOURCE).toContain("FULL_OUTPUT_COUNT = 25_344_000");
    expect(HARNESS_SOURCE).toContain("fullK2DeterministicReruns: true");
    expect(HARNESS_SOURCE).toContain("k2NotWorseThanK4");
    expect(HARNESS_SOURCE).toContain('caseSpec("signed-zero"');
    expect(HARNESS_SOURCE).toContain('caseSpec("k4-cancellation"');
    expect(HARNESS_SOURCE).toContain('caseSpec("finite-range"');
    expect(HARNESS_SOURCE).toContain('caseSpec("long-k6144"');
    expect(HARNESS_SOURCE).toContain("completedBeforeReady: true");
    expect(HARNESS_SOURCE).toContain("__ACE_OPT0074_RESULT__");
  });

  it("uses a balanced six-round timestamp timing gate", () => {
    expect(HARNESS_SOURCE).toContain(
      'requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"]',
    );
    expect(HARNESS_SOURCE).toContain('type: "timestamp"');
    expect(HARNESS_SOURCE).toContain("timestampWrites: {");
    expect(HARNESS_SOURCE).toContain("const TIMING_ROUNDS = Object.freeze([");
    expect(HARNESS_SOURCE.match(/armOrder: Object\.freeze\(\[/g)).toHaveLength(6);
    for (const order of [
      '["exact", "k2", "k4"]',
      '["k2", "k4", "exact"]',
      '["k4", "exact", "k2"]',
      '["k4", "k2", "exact"]',
      '["k2", "exact", "k4"]',
      '["exact", "k4", "k2"]',
    ]) expect(HARNESS_SOURCE).toContain(`armOrder: Object.freeze(${order}`);
    expect(HARNESS_SOURCE.match(
      /shapeOrder: Object\.freeze\(\[0, 1, 2, 3\]\)/g,
    )).toHaveLength(3);
    expect(HARNESS_SOURCE.match(
      /shapeOrder: Object\.freeze\(\[3, 2, 1, 0\]\)/g,
    )).toHaveLength(3);
    expect(HARNESS_SOURCE).toContain(
      "samplesPerArmPerShape: TIMING_ROUNDS.length",
    );
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_WEIGHTED_WALL_SPEEDUP = 1.15",
    );
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_WEIGHTED_WALL_SAVING_MILLISECONDS = 25",
    );
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_WEIGHTED_GPU_SPEEDUP = 1.15",
    );
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_WEIGHTED_GPU_SAVING_MILLISECONDS = 25",
    );
    expect(HARNESS_SOURCE).toContain("everyShapeConsistentAndNonOverlapped");
    expect(HARNESS_SOURCE).toContain("weightedGpuSpeedupPassed");
    expect(HARNESS_SOURCE).toContain("everyWeightedRoundWallWin");
    expect(HARNESS_SOURCE).toContain("everyWeightedRoundGpuWin");
    expect(HARNESS_SOURCE).toContain("weightedWallRangesSeparated");
    expect(HARNESS_SOURCE).toContain("weightedGpuRangesSeparated");
    expect(HARNESS_SOURCE).toContain("primitiveFollowUpAuthorized: false");
    expect(HARNESS_SOURCE).toContain("externalThermalGateAuditedByPage: false");
    expect(HARNESS_SOURCE).toContain("productionIntegrationAuthorized: false");
    expect(HARNESS_SOURCE).toContain("measurementCompletedAtEpochMilliseconds");
    expect(HARNESS_SOURCE).toContain("cleanupCompletedAtEpochMilliseconds");
    expect(HARNESS_SOURCE).toContain("timingEvidence.rawSamples.slice()");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain("start the thermal-pressure logger");
    expect(HARNESS_HTML).toContain("Keep thermal polling through");
    expect(HARNESS_HTML).toContain(
      './opt-0074-dit-dense-fp16-k2-partials.ts',
    );
  });
});
