import {
  OPT_0003_GEMM_SHAPES,
  OPT_0003_PAIRED_ORDERS,
  opt0003ActivationValue,
  opt0003ExpectedOutputValue,
  opt0003OutputSentinelIndices,
  opt0003WeightValue,
  parseOpt0003ThermalGateMetadata,
  summarizeOpt0003Samples,
} from "./browser/opt-0003-subgroup-gemm-ab.js";
import { planAceTiledGemm } from "../src/webgpu/kernels/gemm.js";
import { planAceOpt0003SubgroupGemm } from
  "../benchmark/opt-0003-subgroup-gemm.js";
import { describe, expect, it } from "vitest";

describe("OPT-0003 subgroup GEMM paired A/B browser contract", () => {
  it("pins the four weighted M=2250 production shapes", () => {
    expect(OPT_0003_GEMM_SHAPES).toEqual([
      { id: "h-to-h", shape: { rows: 2_250, inner: 2_048, columns: 2_048 } },
      { id: "h-to-1024", shape: { rows: 2_250, inner: 2_048, columns: 1_024 } },
      { id: "h-to-6144", shape: { rows: 2_250, inner: 2_048, columns: 6_144 } },
      { id: "6144-to-h", shape: { rows: 2_250, inner: 6_144, columns: 2_048 } },
    ]);
    expect(Object.isFrozen(OPT_0003_GEMM_SHAPES)).toBe(true);
    for (const fixture of OPT_0003_GEMM_SHAPES) {
      expect(Object.isFrozen(fixture)).toBe(true);
      expect(Object.isFrozen(fixture.shape)).toBe(true);
    }
  });

  it("uses four balanced portable/subgroup AB/BA samples", () => {
    expect(OPT_0003_PAIRED_ORDERS).toEqual([
      ["portable", "subgroup"],
      ["subgroup", "portable"],
      ["subgroup", "portable"],
      ["portable", "subgroup"],
    ]);
    expect(OPT_0003_PAIRED_ORDERS.flat().filter((id) => id === "portable"))
      .toHaveLength(4);
    expect(OPT_0003_PAIRED_ORDERS.flat().filter((id) => id === "subgroup"))
      .toHaveLength(4);
  });

  it("retains raw samples with median and range", () => {
    expect(summarizeOpt0003Samples([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
      range: 8,
    });
    expect(() => summarizeOpt0003Samples([])).toThrow(/must not be empty/);
    expect(() => summarizeOpt0003Samples([1, Number.NaN])).toThrow(
      /finite non-negative/,
    );
  });

  it("fails closed unless external thermal metadata proves the gate", () => {
    const valid = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalDurationSeconds: "30.008",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1021",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0003ThermalGateMetadata(valid)).toEqual({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      durationSeconds: 30.008,
      observationCount: 31,
      pollMilliseconds: 1_000,
      maximumPollGapMilliseconds: 1_021,
      nonNominalObservationCount: 0,
    });
    const short = new URLSearchParams(valid);
    short.set("thermalDurationSeconds", "29.9");
    expect(() => parseOpt0003ThermalGateMetadata(short)).toThrow(
      /30 continuous nominal/,
    );
    const hot = new URLSearchParams(valid);
    hot.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0003ThermalGateMetadata(hot)).toThrow(/non-nominal/);
  });

  it("uses finite varied fixtures with exactly BF16-representable weights", () => {
    const activations = Array.from({ length: 1_024 }, (_, index) =>
      opt0003ActivationValue(index)
    );
    const weights = Array.from({ length: 1_024 }, (_, index) =>
      opt0003WeightValue(index)
    );
    expect(activations.every(Number.isFinite)).toBe(true);
    expect(weights.every(Number.isFinite)).toBe(true);
    expect(new Set(activations).size).toBeGreaterThan(20);
    expect(new Set(weights).size).toBeGreaterThan(20);
    for (const value of weights) expect(roundToBf16(value)).toBe(value);
  });

  it("pins independent CPU sentinels at output tails and range boundaries", () => {
    for (const { shape } of OPT_0003_GEMM_SHAPES) {
      const sentinels = opt0003OutputSentinelIndices(shape);
      const plan = planAceTiledGemm(shape);
      const subgroupPlan = planAceOpt0003SubgroupGemm(shape);
      expect(sentinels).toContain(0);
      expect(sentinels).toContain(shape.rows * shape.columns - 1);
      for (const range of plan.outputRanges) {
        expect(sentinels).toContain(range.firstOutput);
        expect(sentinels).toContain(
          Math.min(
            shape.rows * shape.columns - 1,
            range.firstOutput + range.outputCount - 1,
          ),
        );
      }
      for (const range of subgroupPlan.outputRanges) {
        expect(sentinels).toContain(range.firstOutput);
        expect(sentinels).toContain(
          Math.min(
            shape.rows * shape.columns - 1,
            range.firstOutput + range.outputCount - 1,
          ),
        );
      }
      for (const index of sentinels) {
        expect(Number.isFinite(opt0003ExpectedOutputValue(shape, index)))
          .toBe(true);
      }
    }
  });

  it("pins contraction and reassociation discriminants used by GPU preflight", () => {
    const activation = [-68.06752014160156, 12.192401885986328] as const;
    const weight = [1, 5.5625] as const;
    let separatelyRounded = 0;
    let contracted = 0;
    for (let index = 0; index < activation.length; index += 1) {
      separatelyRounded = Math.fround(
        separatelyRounded +
          Math.fround(activation[index]! * weight[index]!),
      );
      contracted = Math.fround(
        contracted + activation[index]! * weight[index]!,
      );
    }
    expect(float32Bits(separatelyRounded)).toBe(0xbe7d_3800);
    expect(float32Bits(contracted)).toBe(0xbe7d_3830);

    const cancellation = [16_777_216, 1, -16_777_216, 0.5] as const;
    let sourceOrder = 0;
    for (const value of cancellation) {
      sourceOrder = Math.fround(sourceOrder + value);
    }
    const reassociated = Math.fround(
      Math.fround(cancellation[0] + cancellation[2]) +
        Math.fround(cancellation[1] + cancellation[3]),
    );
    expect(sourceOrder).toBe(0.5);
    expect(reassociated).toBe(1.5);
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

function float32Bits(value: number): number {
  const scratch = new Float32Array([value]);
  return new Uint32Array(scratch.buffer)[0]!;
}
