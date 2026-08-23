import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID,
  ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID,
  ACE_OPT_0038_DENSE_WEIGHT_LAYOUT,
  aceOpt0038DenseBoundedPartialsWgsl,
  planAceOpt0038DenseBoundedPartials,
} from "../src/webgpu/kernels/dit-dense-fp16-k8-k16-partials.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k8-k16-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0038-dit-dense-fp16-k8-k16-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0038-dit-dense-fp16-k8-k16-partials.html",
  import.meta.url,
), "utf8");

describe("OPT-0038 dense FP16 K8/K16 bounded partials", () => {
  it.each([
    ["k8" as const, 8, 2, 256],
    ["k16" as const, 16, 4, 128],
  ])(
    "plans %s over the unchanged K4-native layout",
    (variant, partialInner, k4GroupsPerPartial, innerPartialBlocks) => {
      const plan = planAceOpt0038DenseBoundedPartials({
        rows: 2_250,
        inner: 2_048,
        columns: 2_048,
      }, variant);
      expect(plan).toMatchObject({
        variant,
        tileRows: 32,
        tileColumns: 128,
        tileInner: 4,
        workgroupSize: 128,
        subgroupSize: 32,
        rowTiles: 71,
        columnTiles: 16,
        innerK4Groups: 512,
        partialInner,
        k4GroupsPerPartial,
        innerPartialBlocks,
        workgroupCount: 1_136,
      });
      expect(plan.packedWeightStorageShape).toEqual([16, 512, 4, 32, 4]);
      expect(ACE_OPT_0038_DENSE_WEIGHT_LAYOUT).toBe(
        ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
      );
    },
  );

  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("accepts every registered M2250 K%i/N%i geometry", (inner, columns) => {
    for (const variant of ["k8", "k16"] as const) {
      const plan = planAceOpt0038DenseBoundedPartials(
        { rows: 2_250, inner, columns },
        variant,
      );
      expect(plan.workgroupsX).toBe(columns / 128);
      expect(plan.workgroupsY).toBe(71);
    }
  });

  it("emits a K8-local FP16 chain and one FP32 update per K8 block", () => {
    const wgsl = aceOpt0038DenseBoundedPartialsWgsl({
      rows: 2_250,
      inner: 2_048,
      columns: 2_048,
    }, "k8");
    expect(ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID).toMatch(
      /k8-partials-fixed32-wg128-m32-n128-v1$/,
    );
    expect(wgsl).toContain("const INNER_PARTIAL_BLOCKS = 256u;");
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("var partial0: vec4<f16>;");
    expect(wgsl.match(/partial0 = vec4<f16>\(/g)).toHaveLength(1);
    expect(wgsl.match(/partial0 = partial0 \+ vec4<f16>\(/g)).toHaveLength(1);
    expect(wgsl).toContain("acc0 = acc0 + vec4<f32>(partial0);");
    expect(wgsl).toContain("var acc0 = vec4<f32>(0.0);");
    expect(wgsl.match(/subgroupBroadcast\(lane_a_[01], 0u\)/g)).toHaveLength(2);
    expect(wgsl).not.toContain("var<workgroup>");
    expect(wgsl).not.toContain("workgroupBarrier");
  });

  it("emits a K16-local FP16 chain and one FP32 update per K16 block", () => {
    const wgsl = aceOpt0038DenseBoundedPartialsWgsl({
      rows: 2_250,
      inner: 6_144,
      columns: 2_048,
    }, "k16");
    expect(ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID).toMatch(
      /k16-partials-fixed32-wg128-m32-n128-v1$/,
    );
    expect(wgsl).toContain("const INNER_PARTIAL_BLOCKS = 384u;");
    expect(wgsl.match(/partial7 = vec4<f16>\(/g)).toHaveLength(1);
    expect(wgsl.match(/partial7 = partial7 \+ vec4<f16>\(/g)).toHaveLength(3);
    expect(wgsl.match(/subgroupBroadcast\(lane_a_[0-3], 7u\)/g)).toHaveLength(4);
    expect(wgsl.match(/acc7 = acc7 \+ vec4<f32>\(partial7\);/g)).toHaveLength(1);
    expect(wgsl).not.toMatch(/var acc\d+\s*=\s*vec4<f16>/);
  });

  it("rejects unregistered geometry and an unknown runtime variant", () => {
    expect(() => planAceOpt0038DenseBoundedPartials({
      rows: 2_250,
      inner: 1_024,
      columns: 2_048,
    }, "k8")).toThrow(/non-production/);
    expect(() => planAceOpt0038DenseBoundedPartials({
      rows: 2_250,
      inner: 2_048,
      columns: 2_048,
    }, "k32" as never)).toThrow(/unknown/);
  });

  it("gates all four arms on full/adversarial correctness before one timing button", () => {
    for (const shape of [
      'fullSpec("h-h", 2_048, 2_048, 4, 0)',
      'fullSpec("h-1024", 2_048, 1_024, 2, 1)',
      'fullSpec("h-6144", 2_048, 6_144, 2, 2)',
      'fullSpec("6144-h", 6_144, 2_048, 1, 3)',
    ]) expect(HARNESS_SOURCE).toContain(shape);
    for (const fixture of [
      '"signed-zero"',
      '"cancellation"',
      '"range"',
      '"long-k"',
    ]) expect(HARNESS_SOURCE).toContain(fixture);
    expect(HARNESS_SOURCE).toContain('type Arm = "control" | "k4" | "k8" | "k16"');
    expect(HARNESS_SOURCE).toContain("FULL_OUTPUT_COUNT = 25_344_000");
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(HARNESS_SOURCE).toContain("deterministicRawU32");
    expect(HARNESS_SOURCE).toContain("finiteToZeroCount");
    expect(HARNESS_SOURCE).toContain(
      "FULL_FINITE_TO_ZERO_RATE_MAXIMUM = 1 / 1_000_000",
    );
    expect(HARNESS_SOURCE).toContain("finiteToZeroEventsPerMillion");
    expect(HARNESS_SOURCE).toContain(
      "ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR = 4",
    );
    expect(HARNESS_SOURCE).toContain("candidateNonFiniteCount");
    expect(HARNESS_SOURCE).toContain("relativeRmsError");
    expect(HARNESS_SOURCE).toContain("nrmse");
    expect(HARNESS_SOURCE).toContain("snrDecibels");
    expect(HARNESS_SOURCE).toContain("pearsonCorrelation");
    expect(HARNESS_SOURCE).toContain("completedBeforeReady: true");
    expect(HARNESS_SOURCE).toContain(
      "READY — four-arm full outputs and adversarial screens completed",
    );
    expect(HARNESS_SOURCE).toContain("requiredSpeedupOverK4");
    expect(HARNESS_SOURCE).toContain("REQUIRED_SPEEDUP_OVER_K4 = 1.15");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0038_RESULT__ = receipt");
    expect(HARNESS_SOURCE).not.toContain("workgroupBarrier");
    expect(KERNEL_SOURCE).not.toContain("OPT-0020");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain(
      "leave the machine nominal and idle for 30 seconds",
    );
    expect(HARNESS_HTML).toContain(
      "opt-0038-dit-dense-fp16-k8-k16-partials.ts",
    );
  });
});
