import {
  OPT_0001_GEMM_SHAPES,
  opt0001ActivationValue,
  opt0001WeightValue,
} from "./browser/opt-0001-gemm-profiler.js";
import {
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
  planAceTiledGemm,
} from "../src/webgpu/kernels/gemm.js";
import { describe, expect, it } from "vitest";

describe("OPT-0001 exact-shape GEMM profiler contract", () => {
  it("pins the four production DiT GEMM shapes at M=2250", () => {
    expect(OPT_0001_GEMM_SHAPES).toEqual([
      { id: "h-to-h", shape: { rows: 2_250, inner: 2_048, columns: 2_048 } },
      { id: "h-to-1024", shape: { rows: 2_250, inner: 2_048, columns: 1_024 } },
      { id: "h-to-6144", shape: { rows: 2_250, inner: 2_048, columns: 6_144 } },
      { id: "6144-to-h", shape: { rows: 2_250, inner: 6_144, columns: 2_048 } },
    ]);
    expect(Object.isFrozen(OPT_0001_GEMM_SHAPES)).toBe(true);
    for (const fixture of OPT_0001_GEMM_SHAPES) {
      expect(Object.isFrozen(fixture)).toBe(true);
      expect(Object.isFrozen(fixture.shape)).toBe(true);
    }
  });

  it("retains complete, ordered, bounded cooperative range plans", () => {
    expect(OPT_0001_GEMM_SHAPES.map(({ shape }) =>
      planAceTiledGemm(shape).outputRangeCount
    )).toEqual([5, 3, 14, 14]);
    for (const { shape } of OPT_0001_GEMM_SHAPES) {
      const plan = planAceTiledGemm(shape);
      expect(plan.outputRangeCount).toBeGreaterThan(1);
      expect(plan.outputRanges).toHaveLength(plan.outputRangeCount);
      expect(plan.outputRanges.reduce(
        (total, range) => total + range.outputCount,
        0,
      )).toBe(shape.rows * shape.columns);
      expect(plan.outputRanges.map((range) => range.firstOutput)).toEqual(
        [...plan.outputRanges]
          .map((range) => range.firstOutput)
          .sort((left, right) => left - right),
      );
      expect(plan.outputRanges.reduce(
        (total, range) => total + range.multiplyAdds,
        0,
      )).toBeGreaterThanOrEqual(shape.rows * shape.inner * shape.columns);
      for (const range of plan.outputRanges) {
        expect(range.outputCount).toBeLessThanOrEqual(
          ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
        );
        expect(range.multiplyAdds).toBeLessThanOrEqual(
          ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
        );
      }
    }
  });

  it("uses finite, varied, exactly BF16-representable weight fixtures", () => {
    const activation = Array.from({ length: 1_024 }, (_, index) =>
      opt0001ActivationValue(index)
    );
    const weight = Array.from({ length: 1_024 }, (_, index) =>
      opt0001WeightValue(index)
    );
    expect(activation.every(Number.isFinite)).toBe(true);
    expect(weight.every(Number.isFinite)).toBe(true);
    expect(new Set(activation).size).toBeGreaterThan(20);
    expect(new Set(weight).size).toBeGreaterThan(20);
    for (const value of weight) expect(roundToBf16(value)).toBe(value);
  });
});

function roundToBf16(value: number): number {
  const scratch = new Float32Array([value]);
  const bits = new Uint32Array(scratch.buffer);
  const source = bits[0]!;
  const rounded = ((source + 0x7fff + ((source >>> 16) & 1)) >>> 16) & 0xffff;
  bits[0] = rounded << 16;
  return scratch[0]!;
}
