import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  aceFp16VaeCongruentConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
  aceOpt0022VaeConvTranspose1dSubgroupWgsl,
  planAceOpt0022VaeConvTranspose1d,
  planAceOpt0022VaeConvTranspose1dRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";
import {
  OPT_0022_C300_EXPECTED_TOPOLOGY,
  OPT_0022_WEIGHTED_SPEEDUP_THRESHOLD,
  buildOpt0022C300Topology,
  buildOpt0022TimingOrders,
  compareOpt0022InverseWeightLayouts,
  opt0022InputFixtureBitsAt,
  opt0022NativeWeightFixtureBitsAt,
  opt0022NativeWeightIndex,
  opt0022PolyphaseWeightIndex,
  parseOpt0022ThermalGate,
  summarizeOpt0022Timing,
  type Opt0022ProbeTimingInput,
} from "./browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.html",
  import.meta.url,
));
const CURRENT_CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts",
  import.meta.url,
));
const CANDIDATE_CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const CURRENT_CORE_SOURCE = readFileSync(CURRENT_CORE_PATH, "utf8");
const CANDIDATE_CORE_SOURCE = readFileSync(CANDIDATE_CORE_PATH, "utf8");
const EXPECTED_HARNESS_SHA256 =
  "4db0643edc6449087d6b28b5ee6dba3a3b0b9caabc2b4f8eb8d7a44bb27cadb8";
const EXPECTED_HTML_SHA256 =
  "b97ca849b7a2613f3e345cef49bfd50456a25931aab32cf8454adcd5a383d1dc";
const EXPECTED_GENERATED_SHADER_SHA256 = Object.freeze([
  Object.freeze({
    label: "block-0-conv-t1",
    A: "891cf509d3a1753676ed4e352ff72b24d2dc3a8cab923af04109b979737455bf",
    B: "812c99bbbdb26a27cf06ca3f90247322a0b91de7254688c5d5fb4b155a8b963a",
  }),
  Object.freeze({
    label: "block-1-conv-t1",
    A: "5484dfeadd387c80aeffb5f696924642663e417b002dd8bf328bec247e34fcc6",
    B: "5997931ef29ad39982c98e45ef8620d57b3c5db25b1589e639ae4a0a6b1dfdc0",
  }),
  Object.freeze({
    label: "block-2-conv-t1",
    A: "a8d547c77e94c426d69b0d590f5d8fc35f8613e56da50df189d4556433f7bb7c",
    B: "529f81b8cc9a2a4a09849e024c2c543fba39dc514864e4c0294fdfeb17fc45ab",
  }),
  Object.freeze({
    label: "block-3-conv-t1",
    A: "91908682c7c1e65bea607463b66616329b8f5ac72c20c4cf093ffec5b5a6a549",
    B: "4f8e5b4d4e203d394f1fa39b2a3127d8fe3f05fb889c92c9d873fb5c7b83d732",
  }),
  Object.freeze({
    label: "block-4-conv-t1",
    A: "f560adea6e1369a438c30e5272861a194c486a70799c15b73eb26ca31ddcb2c7",
    B: "92921ab4159f36788fbf076fb1a2c1f10ad52adddb24f8a5689a9e045101f5f8",
  }),
]);

describe("OPT-0022 target-browser subgroup/polyphase primitive gate", () => {
  it("freezes the five exact C300 operations, 378 ranges, and 15 probes", () => {
    const topology = buildOpt0022C300Topology();
    expect(topology).toHaveLength(5);
    expect(topology.reduce((sum, operation) =>
      sum + operation.ranges.length, 0)).toBe(378);
    expect(topology.reduce((sum, operation) =>
      sum + operation.selectedRanges.length, 0)).toBe(15);
    expect(topology.map((operation) => ({
      label: operation.label,
      shape: operation.shape,
      outputFrames: operation.outputFrames,
      rangeCount: operation.ranges.length,
      fullRangeRows: operation.ranges[0]!.outputRowCount,
      tailRangeRows: operation.ranges.at(-1)!.outputRowCount,
    }))).toEqual(OPT_0022_C300_EXPECTED_TOPOLOGY);
    expect(topology.map((operation) =>
      operation.selectedRanges.map((range) => ({
        stratum: range.stratum,
        rangeIndex: range.rangeIndex,
        base: range.base,
        count: range.count,
        weight: range.weight,
      }))
    )).toEqual([
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 57_344, weight: 1 },
        { stratum: "interior", rangeIndex: 27, base: 1_548_288, count: 57_344, weight: 52 },
        { stratum: "tail", rangeIndex: 53, base: 3_039_232, count: 32_768, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 114_688, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 4_587_520, count: 114_688, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 9_175_040, count: 40_960, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 229_376, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 9_175_040, count: 229_376, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 18_350_080, count: 81_920, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 458_752, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 18_350_080, count: 458_752, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 36_700_160, count: 163_840, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 917_504, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 36_700_160, count: 917_504, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 73_400_320, count: 327_680, weight: 1 },
      ],
    ]);
  });

  it("proves every selected probe covers all phases and both one-tap edges", () => {
    for (const operation of buildOpt0022C300Topology()) {
      const completeMask = (1 << operation.shape.stride) - 1;
      expect(operation.selectedRanges.every((range) =>
        range.phaseMask === completeMask &&
        range.oneTapOutputRowCount + range.twoTapOutputRowCount ===
          range.outputRowCount
      )).toBe(true);
      expect(operation.selectedRanges[0]).toMatchObject({
        stratum: "first",
        containsFirstOutputRow: true,
      });
      expect(operation.selectedRanges[0]!.oneTapOutputRowCount).toBeGreaterThan(0);
      expect(operation.selectedRanges[2]).toMatchObject({
        stratum: "tail",
        containsLastOutputRow: true,
      });
      expect(operation.selectedRanges[2]!.oneTapOutputRowCount).toBeGreaterThan(0);
    }
  });

  it("makes adjacent, first, and last input-time rows discriminating", () => {
    for (const operation of buildOpt0022C300Topology()) {
      const { inputChannels, inputFrames } = operation.shape;
      expect(inputChannels % 17).not.toBe(0);
      const row = (inputTime: number) => Array.from(
        { length: inputChannels },
        (_, inputChannel) => opt0022InputFixtureBitsAt(
          operation.ordinal,
          inputTime,
          inputChannel,
          inputChannels,
        ),
      );
      const middle = Math.floor(inputFrames / 2);
      expect(row(0)).not.toEqual(row(1));
      expect(row(middle - 1)).not.toEqual(row(middle));
      expect(row(inputFrames - 2)).not.toEqual(row(inputFrames - 1));
      expect(row(0)).not.toEqual(row(inputFrames - 1));
    }
    expect(HARNESS_SOURCE).toContain("0x3555");
    expect(HARNESS_SOURCE).toContain(
      "operation.shape.inputChannels % INPUT_PATTERN.length === 0",
    );
  });

  it("makes Cout+1, Cout+7, interior, and boundary weight slices distinct", () => {
    for (const operation of buildOpt0022C300Topology()) {
      const { shape } = operation;
      const outputChannelStride = shape.kernelSize * shape.inputChannels;
      expect(gcdForContract(outputChannelStride, 19)).toBe(1);
      expect(gcdForContract(shape.inputChannels, 19)).toBe(1);
      expect(gcdForContract(shape.stride * shape.inputChannels, 19)).toBe(1);
      const period = Array.from({ length: 19 }, (_, inputChannel) =>
        opt0022NativeWeightFixtureBitsAt(
          operation.ordinal,
          0,
          0,
          inputChannel,
          shape,
        )
      );
      expect(new Set(period).size).toBe(19);
      const slice = (outputChannel: number) => Array.from(
        { length: outputChannelStride },
        (_, index) => opt0022NativeWeightFixtureBitsAt(
          operation.ordinal,
          outputChannel,
          Math.floor(index / shape.inputChannels),
          index % shape.inputChannels,
          shape,
        ),
      );
      const middle = Math.floor(shape.outputChannels / 2);
      expect(slice(0)).not.toEqual(slice(1));
      expect(slice(0)).not.toEqual(slice(7));
      expect(slice(middle - 1)).not.toEqual(slice(middle));
      expect(slice(shape.outputChannels - 2)).not.toEqual(
        slice(shape.outputChannels - 1),
      );
      expect(slice(0)).not.toEqual(slice(shape.outputChannels - 1));
    }
    expect(HARNESS_SOURCE).toContain(
      "gcd(kernelSize * inputChannels, WEIGHT_PATTERN.length) !== 1",
    );
  });

  it("freezes the 16x workgroup reduction and full 49,610,752-word layout", () => {
    let weightWords = 0;
    let currentWorkgroups = 0;
    let candidateWorkgroups = 0;
    let currentBarrierEvents = 0;
    for (const operation of buildOpt0022C300Topology()) {
      const A = planAceFp16VaeConvTranspose1d(operation.shape);
      const B = planAceOpt0022VaeConvTranspose1d(operation.shape);
      expect(B).toMatchObject({
        kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
        weightLayout: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
        subgroupSize: 32,
        workgroupSize: 128,
        subgroupsPerWorkgroup: 4,
        rowsPerSubgroup: 16,
        channelsPerSubgroup: 32,
        channelsPerWorkgroup: 128,
        taps: 2,
        inputChannelChunk: 8,
        workgroupStorageBytes: 0,
        workgroupBarrierCount: 0,
      });
      expect(B.weightElements).toBe(A.weightElements);
      expect(B.polyphaseWeightStorageBytes).toBe(A.weightStorageBytes);
      weightWords += A.weightElements;
      for (const range of operation.ranges) {
        const aRange = planAceFp16VaeConvTranspose1dCongruentRange(A, range);
        const bRange = planAceOpt0022VaeConvTranspose1dRange(B, range);
        const aWorkgroups = aRange.workgroupsX * aRange.workgroupsY *
          aRange.workgroupsZ;
        const bWorkgroups = bRange.workgroupsX * bRange.workgroupsY *
          bRange.workgroupsZ;
        expect(aWorkgroups).toBe(bWorkgroups * 16);
        currentWorkgroups += aWorkgroups;
        candidateWorkgroups += bWorkgroups;
        currentBarrierEvents += aWorkgroups * 2 * A.inputChannelChunkCount * 2;
      }
    }
    expect(weightWords).toBe(49_610_752);
    expect(currentWorkgroups).toBe(1_169_664);
    expect(candidateWorkgroups).toBe(73_104);
    expect(currentBarrierEvents).toBe(28_594_176);
  });

  it("round-trips a discriminating native/polyphase fixture exactly", () => {
    const shape: AceVaeConvTranspose1dShape = Object.freeze({
      batch: 1,
      inputFrames: 3,
      inputChannels: 3,
      outputChannels: 5,
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      padding: 1,
      outputPadding: 0,
    });
    const words = shape.outputChannels * shape.kernelSize * shape.inputChannels;
    const native = new Uint16Array(words);
    const polyphase = new Uint16Array(words);
    for (let output = 0; output < shape.outputChannels; output += 1) {
      for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
        for (let input = 0; input < shape.inputChannels; input += 1) {
          const nativeIndex = opt0022NativeWeightIndex(
            shape,
            output,
            kernel,
            input,
          );
          const polyphaseIndex = opt0022PolyphaseWeightIndex(
            shape,
            kernel % shape.stride,
            Math.floor(kernel / shape.stride),
            input,
            output,
          );
          native[nativeIndex] = nativeIndex + 1;
          polyphase[polyphaseIndex] = native[nativeIndex]!;
        }
      }
    }
    expect(compareOpt0022InverseWeightLayouts(native, polyphase, shape))
      .toMatchObject({
        comparedU16Count: words,
        mismatchCount: 0,
        firstMismatch: null,
        rawU16Exact: true,
      });
    polyphase[opt0022PolyphaseWeightIndex(shape, 1, 1, 2, 4)]! ^= 1;
    expect(compareOpt0022InverseWeightLayouts(native, polyphase, shape))
      .toMatchObject({ mismatchCount: 1, rawU16Exact: false });
  });

  it("builds six balanced AB/BA rounds and rotates all 15 probes", () => {
    const orders = buildOpt0022TimingOrders();
    expect(orders).toHaveLength(90);
    for (let round = 0; round < 6; round += 1) {
      const entries = orders.filter((entry) => entry.roundIndex === round);
      expect(entries).toHaveLength(15);
      expect(new Set(entries.map((entry) => entry.probeOrdinal)).size).toBe(15);
      expect(entries.every((entry) => entry.order.join("") ===
        (round % 2 === 0 ? "AB" : "BA"))).toBe(true);
      expect(entries[0]!.probeOrdinal).toBe(round);
    }
    for (let probe = 0; probe < 15; probe += 1) {
      expect(orders.filter((entry) => entry.probeOrdinal === probe)).toHaveLength(6);
    }
  });

  it("enforces every-operation, both-position, and exact speedup gates", () => {
    expect(OPT_0022_WEIGHTED_SPEEDUP_THRESHOLD).toBe(1.3398349037268882);
    const positive = summarizeOpt0022Timing(timingInputs(20, 10));
    expect(positive).toMatchObject({
      exactRangeCount: 378,
      everyOperationBFaster: true,
      bothTimingPositionsBFaster: true,
      passed: true,
      decision: "positive-package-layer-gate-authorized",
      weighted: { A: 7_560, B: 3_780, speedup: 2 },
    });
    expect(positive["operations"]).toHaveLength(5);
    expect(positive["positions"]).toEqual([
      { position: 0, A: 7_560, B: 3_780, bFaster: true, speedup: 2 },
      { position: 1, A: 7_560, B: 3_780, bFaster: true, speedup: 2 },
    ]);
    const belowThreshold = summarizeOpt0022Timing(timingInputs(20, 16));
    expect(belowThreshold).toMatchObject({
      everyOperationBFaster: true,
      bothTimingPositionsBFaster: true,
      passed: false,
      decision: "negative-stop-primitive-gate",
    });
    const positionFailure = summarizeOpt0022Timing(timingInputs(
      20,
      10,
      (round) => round % 2 === 0 ? 10 : 30,
    ));
    expect(positionFailure).toMatchObject({
      bothTimingPositionsBFaster: false,
      passed: false,
    });
  });

  it("accepts exactly one fresh 30-second nominal thermal pre-gate", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0022ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0022ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
  });

  it("authenticates both core sources and all ten generated shaders", () => {
    expect(hashText(CURRENT_CORE_SOURCE)).toBe(
      "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2",
    );
    expect(hashText(CANDIDATE_CORE_SOURCE)).toBe(
      "b3a02e29419021d78f669b7ed0333b80c8e5739ed46d9d9645f814d971a9edfa",
    );
    expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID).toBe(
      "ace-vae-fp16-congruent-two-tap-conv-transpose1d-v1",
    );
    expect(ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID).toBe(
      "ace-vae-fp16-fixed32-subgroup-polyphase-conv-transpose1d-v1",
    );
    expect(buildOpt0022C300Topology().map((operation) => Object.freeze({
      label: operation.label,
      A: hashText(aceFp16VaeCongruentConvTranspose1dWgsl(operation.shape)),
      B: hashText(aceOpt0022VaeConvTranspose1dSubgroupWgsl(operation.shape)),
    }))).toEqual(EXPECTED_GENERATED_SHADER_SHA256);
  });

  it("pins bounded correctness, source immutability, lifecycle, and stop rules", () => {
    expect(HARNESS_SOURCE).toContain("TOTAL_WEIGHT_U16_COUNT = 49_610_752");
    expect(HARNESS_SOURCE).toContain(
      "CORRECTNESS_COMPARISON_U16_COUNT = 8_404_992",
    );
    expect(HARNESS_SOURCE).toContain("balancedCorrectnessOrders");
    expect(HARNESS_SOURCE).toContain("allPhasesCoveredInEveryProbe");
    expect(HARNESS_SOURCE).toContain("bothOneTapBoundariesCoveredPerOperation");
    expect(HARNESS_SOURCE).toContain("qNaNPrefillCompleteWrites");
    expect(HARNESS_SOURCE).toContain("guardsAndAdjacentCanariesUntouched");
    expect(HARNESS_SOURCE).toContain("verifyAllSourcesImmutable");
    expect(HARNESS_SOURCE).toContain("native-weight");
    expect(HARNESS_SOURCE).toContain("polyphase-weight");
    expect(HARNESS_SOURCE).toContain("mapsBalanced");
    expect(HARNESS_SOURCE).toContain("zeroLiveResources");
    expect(HARNESS_SOURCE).toContain("deviceDestroyed");
    expect(HARNESS_SOURCE).toContain("negative-stop-primitive-gate");
    expect(HARNESS_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HARNESS_SOURCE).not.toContain("PENDING_");
    expect(HTML_SOURCE).toContain("49,610,752 weight words");
    expect(HTML_SOURCE).toContain("8,404,992 raw-U16");
    expect(HTML_SOURCE).toContain("continuous 30-second nominal");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.ts"',
    );
  });

  it("freezes the three-file gate surface without runtime integration", () => {
    expect(hashText(HARNESS_SOURCE)).toBe(EXPECTED_HARNESS_SHA256);
    expect(hashText(HTML_SOURCE)).toBe(EXPECTED_HTML_SHA256);
    expect(HARNESS_SOURCE).not.toContain("worker-runtime");
    expect(HARNESS_SOURCE).not.toContain("model/convert");
    expect(HARNESS_SOURCE).not.toContain("createVaeBackend");
    expect(HARNESS_SOURCE).not.toContain("productionProfile");
    expect(HARNESS_SOURCE).toContain("productionIntegrationAuthorized: false");
    expect(HARNESS_SOURCE).toContain("m2250ProductRunAuthorized: false");
  });
});

function timingInputs(
  aMilliseconds: number,
  bMilliseconds: number,
  bByRound: (round: number) => number = () => bMilliseconds,
): readonly Opt0022ProbeTimingInput[] {
  return buildOpt0022C300Topology().flatMap((operation) =>
    operation.selectedRanges.map((range) => Object.freeze({
      operationLabel: operation.label,
      stratum: range.stratum,
      weight: range.weight,
      rounds: Object.freeze(Array.from({ length: 6 }, (_, roundIndex) =>
        Object.freeze({
          roundIndex,
          order: roundIndex % 2 === 0 ? "AB" as const : "BA" as const,
          aMilliseconds,
          bMilliseconds: bByRound(roundIndex),
        })
      )),
    }))
  );
}

function hashText(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function gcdForContract(left: number, right: number): number {
  let a = left;
  let b = right;
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}
