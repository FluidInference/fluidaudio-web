import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0009_ACCUMULATION_ORDERS,
  ACE_OPT_0009_ACCUMULATION_SCOPE,
  ACE_OPT_0009_ACCUMULATION_SHAPES,
  ACE_OPT_0009_CORRECTNESS_FIXTURES,
  ACE_OPT_0009_EXACT_SHADER_SHA256,
  ACE_OPT_0009_FAIRNESS_DISCLOSURE,
  OPT_0009_ACE_SUBGROUP_GEMM_SOURCE_SHA256,
  aceOpt0009AccumulationWgsl,
  aceOpt0009AdaptedWeightScalarIndex,
  compareAceOpt0009GpuOutputs,
  aceOpt0009CpuOutputValue,
  planAceOpt0009AdaptedGemm,
  summarizeAceOpt0009Samples,
  type AceOpt0009AccumulationArm,
} from "../benchmark/opt-0009-ace-gemm-accumulation-ab.js";

const ARMS = [
  "packed-bf16-fp32-oracle",
  "fp16-fp32-accum",
  "fp16-native-accum",
] as const satisfies readonly AceOpt0009AccumulationArm[];

describe("OPT-0009 ACE three-arm accumulation browser contract", () => {
  it("pins the accepted production packed-BF16 oracle source", async () => {
    const source = await readFile(new URL(
      "../src/webgpu/kernels/subgroup-gemm.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(source).digest("hex")).toBe(
      OPT_0009_ACE_SUBGROUP_GEMM_SOURCE_SHA256,
    );
  });

  it("plans all exact M2250 shapes with guarded M32/N256/K32 geometry", () => {
    expect(ACE_OPT_0009_ACCUMULATION_SHAPES.map((shape) => shape.id)).toEqual([
      "ace-m2250-k2048-n2048",
      "ace-m2250-k2048-n1024",
      "ace-m2250-k2048-n6144",
      "ace-m2250-k6144-n2048",
    ]);
    for (const shape of ACE_OPT_0009_ACCUMULATION_SHAPES) {
      const plan = planAceOpt0009AdaptedGemm(shape);
      expect(plan).toMatchObject({
        tileRows: 32,
        tileColumns: 256,
        tileInner: 32,
        workgroupSize: 128,
        subgroupSize: 32,
        rowTiles: 71,
        scheduledRows: 2_272,
        sourceClassification:
          "explicitly-adapted-parakeet-fixed32-direct-n256",
      });
      expect(plan.workgroups).toEqual([shape.columns / 256, 71, 1]);
      expect(plan.validFlops).toBe(shape.rows * shape.inner * shape.columns * 2);
      expect(plan.scheduledFlops).toBe(2_272 * shape.inner * shape.columns * 2);
    }
  });

  it("maps direct N256 weights bijectively across N256/K32 boundaries", () => {
    const shape = { inner: 64, columns: 512 };
    const count = shape.inner * shape.columns;
    const mapped = Array.from({ length: count }, (_, logical) =>
      aceOpt0009AdaptedWeightScalarIndex(shape, logical)
    );
    expect(new Set(mapped).size).toBe(count);
    expect(Math.min(...mapped)).toBe(0);
    expect(Math.max(...mapped)).toBe(count - 1);
    expect(aceOpt0009AdaptedWeightScalarIndex(shape, 31)).toBe(31 * 256);
    expect(aceOpt0009AdaptedWeightScalarIndex(shape, 32)).toBe(256 * 32);
    expect(aceOpt0009AdaptedWeightScalarIndex(shape, 256 * 64)).toBe(256 * 64);
  });

  it("labels adaptations, guards the M tail, and preserves production oracle math", () => {
    const shape = { rows: 2_250, inner: 2_048, columns: 2_048 };
    const oracle = aceOpt0009AccumulationWgsl(shape, "packed-bf16-fp32-oracle");
    const fp32 = aceOpt0009AccumulationWgsl(shape, "fp16-fp32-accum");
    const native = aceOpt0009AccumulationWgsl(shape, "fp16-native-accum");
    expect(oracle).toContain("var<storage, read> activation: array<f32>");
    expect(oracle).toContain("decode_bf16_low");
    expect(oracle).toContain("output_range.first_workgroup");
    for (const adapted of [fp32, native]) {
      expect(adapted).toContain("explicit adaptation of pinned Parakeet");
      expect(adapted).toContain("This is not unchanged Parakeet source");
      expect(adapted).toContain("enable f16;");
      expect(adapted).toContain("@compute @workgroup_size(128, 1, 1)");
      expect(adapted).toContain("group.y >= 71u");
      expect(adapted).toContain("lane_row < ROWS");
      expect(adapted).toContain("if (row < ROWS)");
      expect(adapted).toContain("output: array<vec4<f32>>");
      expect(adapted).not.toContain("var<workgroup>");
    }
    expect(fp32).toContain("var acc0_0 = vec4<f32>(0.0)");
    expect(fp32).toContain("vec4<f32>(f32(a0)) * vec4<f32>(b0)");
    expect(fp32).not.toMatch(/\bfma\s*\(/);
    expect(native).toContain("var acc0_0 = vec4<f16>(0.0h)");
    expect(native).toContain("fma(vec4<f16>(a0), b0, acc0_0)");
  });

  it("balances every arm and every order position exactly twice", () => {
    expect(ACE_OPT_0009_ACCUMULATION_ORDERS).toHaveLength(6);
    for (const arm of ARMS) {
      expect(ACE_OPT_0009_ACCUMULATION_ORDERS.flat().filter((id) => id === arm))
        .toHaveLength(6);
      for (let position = 0; position < 3; position += 1) {
        expect(ACE_OPT_0009_ACCUMULATION_ORDERS.filter(
          (order) => order[position] === arm,
        )).toHaveLength(2);
      }
    }
  });

  it("covers the adversarial matrix and makes no numerical threshold claim", () => {
    const coverage = new Set(ACE_OPT_0009_CORRECTNESS_FIXTURES.flatMap(
      (fixture) => fixture.coverage,
    ));
    for (const required of [
      "positive-zero",
      "negative-zero",
      "cancellation-sensitive-magnitudes",
      "fp16-max-finite",
      "fp16-min-normal",
      "fp16-min-subnormal",
      "positive-overflow",
      "negative-overflow",
      "finite-nonfinite-classification",
      "long-k",
      "k6144",
      "k2048",
      "m-tail",
      "benign-production-shaped-probe",
      "complete-output",
    ]) expect(coverage).toContain(required);
    expect(ACE_OPT_0009_ACCUMULATION_SCOPE.acceptanceThresholdApplied).toBe(false);
    expect(ACE_OPT_0009_ACCUMULATION_SCOPE.closesExperiment).toBe(false);
    expect(ACE_OPT_0009_FAIRNESS_DISCLOSURE.unavoidableDistinctions).toHaveLength(3);
  });

  it("keeps CPU diagnostics sensitive to native FP16 overflow", () => {
    const fixture = ACE_OPT_0009_CORRECTNESS_FIXTURES.find(
      (candidate) => candidate.fixtureKind === "fp16-range-overflow",
    )!;
    expect(aceOpt0009CpuOutputValue(fixture, "fp16-fp32-accum", 0))
      .toSatisfy(Number.isFinite);
    expect(aceOpt0009CpuOutputValue(fixture, "fp16-native-accum", 0))
      .toBe(Infinity);
    expect(summarizeAceOpt0009Samples([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
      range: 8,
    });
  });

  it("reports signed-zero differences separately from numeric classes", () => {
    expect(compareAceOpt0009GpuOutputs(
      new Float32Array([0, -0, 1]),
      new Float32Array([-0, -0, 1]),
    )).toMatchObject({
      classificationMismatchCount: 0,
      signedZeroMismatchCount: 1,
      bitExactCount: 2,
    });
  });

  it("pins every exact generated shader", () => {
    const actual: Record<string, string> = {};
    for (const shape of ACE_OPT_0009_ACCUMULATION_SHAPES) {
      for (const arm of ARMS) {
        const key = `${shape.id}:${arm}`;
        const source = aceOpt0009AccumulationWgsl(shape, arm);
        actual[key] = createHash("sha256").update(source).digest("hex");
      }
    }
    expect(actual).toEqual(ACE_OPT_0009_EXACT_SHADER_SHA256);
  });
});
