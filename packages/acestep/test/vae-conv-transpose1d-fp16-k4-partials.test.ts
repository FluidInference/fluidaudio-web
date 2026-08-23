import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
  aceOpt0048VaeConvTranspose1dK4PackedWeightIndex,
  aceOpt0048VaeConvTranspose1dK4Wgsl,
  packAceOpt0048VaeConvTranspose1dK4WeightU16,
  planAceOpt0048VaeConvTranspose1dK4,
  planAceOpt0048VaeConvTranspose1dK4Range,
  planAceOpt0048VaeConvTranspose1dK4Weight,
  unpackAceOpt0048VaeConvTranspose1dK4WeightU16,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";
import {
  buildOpt0048Cases,
  parseOpt0048ThermalGate,
} from "./browser/opt-0048-vae-convtranspose-k4-partials.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0048-vae-convtranspose-k4-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0048-vae-convtranspose-k4-partials.html",
  import.meta.url,
), "utf8");

const SHAPES = Object.freeze([
  shape(300, 2_048, 1_024, 10),
  shape(3_000, 1_024, 512, 6),
  shape(18_000, 512, 256, 4),
  shape(72_000, 256, 128, 4),
  shape(288_000, 128, 128, 2),
]);

describe("OPT-0048 ConvTranspose bounded FP16 K4 partials", () => {
  it("inherits OPT-0040's frozen per-shape reuse ownership", () => {
    for (const [index, candidate] of SHAPES.entries()) {
      const plan = planAceOpt0048VaeConvTranspose1dK4(
        `block-${index}-conv-t1`,
        candidate,
      );
      expect(plan).toMatchObject({
        operationLabel: `block-${index}-conv-t1`,
        reuseAxis: index < 3 ? "channel" : "row",
        kernelId: index < 3
          ? ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
          : ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
        weightLayout: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
        rowsPerSubgroup: index < 3 ? 4 : 8,
        channelsPerLane: index < 3 ? 8 : 4,
        accumulatorCountPerLane: 32,
        workgroupStorageBytes: 0,
        workgroupBarrierCount: 0,
        reductionSemantics:
          "increasing-tap-cin4-fp16-dot4-partials-fp32-running-state",
      });
      expect(plan.inputChannelK4Groups).toBe(candidate.inputChannels / 4);
      expect(plan.packedWeightStorageShape).toEqual([
        candidate.stride,
        2,
        candidate.inputChannels / 4,
        candidate.outputChannels / (index < 3 ? 256 : 128),
        32,
        index < 3 ? 8 : 4,
        4,
      ]);
      const range = planAceOpt0048VaeConvTranspose1dK4Range(plan, {
        base: 0,
        count: plan.outputElements,
      });
      expect(range).toMatchObject({
        count: plan.outputElements,
        workgroupsY: candidate.outputChannels / (index < 3 ? 256 : 128),
        workgroupsZ: candidate.stride,
      });
      expect(
        range.workgroupsX * plan.rowsPerWorkgroup * candidate.stride,
      ).toBeGreaterThanOrEqual(plan.outputFrames);
      expect(
        (range.workgroupsX - 1) * plan.rowsPerWorkgroup * candidate.stride,
      ).toBeLessThan(plan.outputFrames);
    }
  });

  it.each([
    ["channel" as const, 256, [2, 2, 2, 1, 32, 8, 4]],
    ["row" as const, 128, [2, 2, 2, 1, 32, 4, 4]],
  ])(
    "proves exhaustive %s forward/inverse layout identity",
    (reuseAxis, outputChannels, expectedShape) => {
      const plan = planAceOpt0048VaeConvTranspose1dK4Weight({
        kernelSize: 4,
        stride: 2,
        dilation: 1,
        outputPadding: 0,
        inputChannels: 8,
        outputChannels,
      }, reuseAxis);
      expect(plan.packedWeightStorageShape).toEqual(expectedShape);
      const logical = Uint16Array.from(
        { length: plan.logicalWeightElements },
        (_, index) => (index * 4051 + 17) & 0xffff,
      );
      const packed = packAceOpt0048VaeConvTranspose1dK4WeightU16(
        logical,
        plan,
      );
      const visited = new Uint8Array(logical.length);
      for (let phase = 0; phase < plan.stride; phase += 1) {
        for (let tap = 0; tap < 2; tap += 1) {
          for (let input = 0; input < plan.inputChannels; input += 1) {
            for (let output = 0; output < plan.outputChannels; output += 1) {
              const physical = aceOpt0048VaeConvTranspose1dK4PackedWeightIndex(
                phase,
                tap,
                input,
                output,
                plan,
              );
              expect(visited[physical]).toBe(0);
              visited[physical] = 1;
              const logicalIndex = (((phase * 2 + tap) * plan.inputChannels +
                input) * plan.outputChannels + output);
              expect(packed[physical]).toBe(logical[logicalIndex]);
            }
          }
        }
      }
      expect(visited.every((entry) => entry === 1)).toBe(true);
      expect(unpackAceOpt0048VaeConvTranspose1dK4WeightU16(packed, plan))
        .toEqual(logical);
    },
  );

  it("makes each lane's output-contiguous K4 vectors adjacent", () => {
    const channel = planAceOpt0048VaeConvTranspose1dK4Weight({
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      outputPadding: 0,
      inputChannels: 8,
      outputChannels: 256,
    }, "channel");
    expect(index(0, 0, 0, 0, channel)).toBe(0);
    expect(index(0, 0, 3, 0, channel)).toBe(3);
    expect(index(0, 0, 0, 1, channel)).toBe(4);
    expect(index(0, 0, 0, 8, channel)).toBe(32);
    expect(index(0, 0, 4, 0, channel)).toBe(1_024);
    expect(index(0, 1, 0, 0, channel)).toBe(2_048);
    expect(index(1, 0, 0, 0, channel)).toBe(4_096);

    const row = planAceOpt0048VaeConvTranspose1dK4Weight({
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      outputPadding: 0,
      inputChannels: 8,
      outputChannels: 128,
    }, "row");
    expect(index(0, 0, 0, 1, row)).toBe(4);
    expect(index(0, 0, 0, 4, row)).toBe(16);
    expect(index(0, 0, 4, 0, row)).toBe(512);
  });

  it("emits only bounded native dot4 partials into FP32 running state", () => {
    for (const [index, candidate] of [SHAPES[0]!, SHAPES[4]!].entries()) {
      const operation = index === 0 ? "block-0-conv-t1" : "block-4-conv-t1";
      const source = aceOpt0048VaeConvTranspose1dK4Wgsl(
        operation,
        candidate,
      );
      expect(source).toContain("input: array<vec4<f16>>");
      expect(source).toContain("packed_weight: array<vec4<f16>>");
      expect(source).toContain(
        "for (var input_channel4 = 0u;\n      input_channel4 < INPUT_CHANNEL_K4_GROUPS;",
      );
      expect(source.match(/dot\(input_operand\d+, weight\d+\)/g))
        .toHaveLength(32);
      expect(source.match(/let partial\d+_[01] = vec4<f16>\(/g))
        .toHaveLength(8);
      expect(source.match(/\+ vec4<f32>\(partial\d+_[01]\)/g))
        .toHaveLength(8);
      expect(source.match(/= f16\(sum/g)).toHaveLength(32);
      expect(source).not.toContain("var<workgroup>");
      expect(source).not.toContain("workgroupBarrier");
      expect(source).not.toMatch(/var sum\d+(?:_[01])?\s*=\s*vec4<f16>/);
    }
    expect(KERNEL_SOURCE).not.toContain("K8");
    expect(KERNEL_SOURCE).not.toContain("K16");
  });

  it("fails closed on unregistered shapes and malformed pack coordinates", () => {
    expect(() => planAceOpt0048VaeConvTranspose1dK4(
      "block-0-conv-t1",
      SHAPES[1]!,
    )).toThrow(/authenticated shape/);
    expect(() => planAceOpt0048VaeConvTranspose1dK4Weight({
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      outputPadding: 0,
      inputChannels: 6,
      outputChannels: 128,
    }, "row")).toThrow(/Cin divisible by 4/);
    const plan = planAceOpt0048VaeConvTranspose1dK4Weight({
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      outputPadding: 0,
      inputChannels: 8,
      outputChannels: 128,
    }, "row");
    expect(() => packAceOpt0048VaeConvTranspose1dK4WeightU16(
      new Uint16Array(1),
      plan,
    )).toThrow(/expected 4096/);
    expect(() => index(2, 0, 0, 0, plan)).toThrow(/phase .*out of bounds/);
  });

  it("freezes the five-shape browser screen and six balanced permutations", () => {
    expect(buildOpt0048Cases().map(({ label, reuseAxis }) => [
      label,
      reuseAxis,
    ])).toEqual([
      ["block-0-conv-t1", "channel"],
      ["block-1-conv-t1", "channel"],
      ["block-2-conv-t1", "channel"],
      ["block-3-conv-t1", "row"],
      ["block-4-conv-t1", "row"],
    ]);
    expect(HARNESS_SOURCE).toContain("TIMING_ORDERS.length");
    expect(HARNESS_SOURCE).toContain("REQUIRED_SUMMED_SPEEDUP = 1.50");
    expect(HARNESS_SOURCE).toContain("noSlowerEveryShape");
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL = 0x7e55");
    expect(HARNESS_SOURCE).toContain("STORAGE_CANARY = 0xa55a");
    expect(HARNESS_SOURCE).toContain("deterministicRawU16: true");
    expect(HARNESS_SOURCE).toContain("completeWrites: true");
    expect(HARNESS_SOURCE).toContain("allCanariesIntact: true");
    expect(HARNESS_SOURCE).toContain("allFinite: true");
    expect(HARNESS_SOURCE).toContain("packingOutsideTimedRegion: true");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0048_RESULT__ = receipt");
    expect(HARNESS_SOURCE).not.toContain("vae-fp16-decoder");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(HARNESS_HTML).toContain('name="checkCount"');
  });

  it("accepts exactly one fresh level-0 check after at least 30 seconds", () => {
    const valid = thermal({
      waitStartedAtEpochMilliseconds: 10_000,
      checkedAtEpochMilliseconds: 40_055,
      checkCount: 1,
      thermalLevel: 0,
    });
    expect(parseOpt0048ThermalGate(valid, 9_000, 40_120)).toMatchObject({
      waitDurationMilliseconds: 30_055,
      checkCount: 1,
      thermalLevel: 0,
      launchDelayMilliseconds: 65,
    });
    for (const invalid of [
      { waitStartedAtEpochMilliseconds: 10_000, checkedAtEpochMilliseconds: 39_999, checkCount: 1, thermalLevel: 0 },
      { waitStartedAtEpochMilliseconds: 10_000, checkedAtEpochMilliseconds: 40_055, checkCount: 2, thermalLevel: 0 },
      { waitStartedAtEpochMilliseconds: 10_000, checkedAtEpochMilliseconds: 40_055, checkCount: 1, thermalLevel: 1 },
    ]) {
      expect(() => parseOpt0048ThermalGate(
        thermal(invalid),
        9_000,
        40_120,
      )).toThrow(/exactly one level-0 notifyutil check/);
    }
  });
});

function shape(
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  stride: number,
): AceVaeConvTranspose1dShape {
  return Object.freeze({
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 2 * stride,
    stride,
    dilation: 1,
    padding: stride / 2,
    outputPadding: 0,
  });
}

function index(
  phase: number,
  tap: number,
  inputChannel: number,
  outputChannel: number,
  plan: Parameters<
    typeof aceOpt0048VaeConvTranspose1dK4PackedWeightIndex
  >[4],
): number {
  return aceOpt0048VaeConvTranspose1dK4PackedWeightIndex(
    phase,
    tap,
    inputChannel,
    outputChannel,
    plan,
  );
}

function thermal(values: Readonly<{
  waitStartedAtEpochMilliseconds: number;
  checkedAtEpochMilliseconds: number;
  checkCount: number;
  thermalLevel: number;
}>): URLSearchParams {
  return new URLSearchParams({
    thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
    waitStartedAtEpochMilliseconds: String(
      values.waitStartedAtEpochMilliseconds,
    ),
    checkedAtEpochMilliseconds: String(values.checkedAtEpochMilliseconds),
    checkCount: String(values.checkCount),
    thermalLevel: String(values.thermalLevel),
  });
}
