import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  OPT_0057_NUMERICAL_ENVELOPE,
  OPT_0057_REQUIRED_K7_SPEEDUP,
  compareOpt0057Raw,
  compareOpt0057Waveforms,
  evaluateOpt0057BalancedTiming,
  parseOpt0057ThermalGate,
  type Opt0057TimingSample,
} from "./browser/opt-0057-vae-k7-k4-row-reuse-c512-contract.js";
import { ACE_OPT_0057_VAE_K7_ROUTES } from
  "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";
import {
  float16BitsToNumber,
  numberToFloat16Bits,
} from "./browser/opt-0011-vae-conv1d-fp16-ab.js";

const WORKER_SOURCE = source(
  "./browser/opt-0057-vae-k7-k4-row-reuse-c512-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0057-vae-k7-k4-row-reuse-c512.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0057-vae-k7-k4-row-reuse-c512.html",
);
const CONTRACT_SOURCE = source(
  "./browser/opt-0057-vae-k7-k4-row-reuse-c512-contract.ts",
);
const EXACT_TRANSPOSE_SOURCE = source(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.ts",
);
const K4_TRANSPOSE_SOURCE = source(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.ts",
);

describe("OPT-0057 authenticated revision-7 C512 browser gate", () => {
  it("authenticates both exact package identities and never co-resides GPU owners", () => {
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-experimental/manifest.json"',
    );
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-revision7-experimental/manifest.json"',
    );
    expect(WORKER_SOURCE).toContain(
      "expectedManifestSha256",
    );
    expect(WORKER_SOURCE).toContain(
      "authenticatedVaeConverterRevision: 7 as const",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256",
    );
    expect(WORKER_SOURCE).toContain(
      "tracker.active !== null || tracker.liveOwners !== 0",
    );
    expect(WORKER_SOURCE).toContain("maximumAllowedLiveOwners: 1");
    expect(WORKER_SOURCE).toContain(
      "phaseBackendAndBuffersDestroyedBeforeNextOwner: true",
    );
    expect(WORKER_SOURCE).toContain("idempotentDestroyPromises");
    expect(WORKER_SOURCE).toContain("candidateAndRevision6NeverCoResident: true");
  });

  it("reconciles every C512 K7 and ConvTranspose quantum by physical owner", () => {
    expect(WORKER_SOURCE).toContain("const K7_TOTAL_QUANTA = 4_090");
    expect(WORKER_SOURCE).toContain("const K7_SELECTED_QUANTA = 3_360");
    expect(WORKER_SOURCE).toContain("const K7_NATIVE_QUANTA = 730");
    expect(WORKER_SOURCE).toContain(
      "const CONV_TRANSPOSE_TOTAL_QUANTA = 644",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID",
    );
    for (const expected of [
      '["block-0-conv-t1", 92,',
      '["block-1-conv-t1", 138,',
      '["block-2-conv-t1", 138,',
      '["block-3-conv-t1", 138,',
      '["block-4-conv-t1", 138,',
    ]) expect(WORKER_SOURCE).toContain(expected);
    expect(WORKER_SOURCE).toContain("selectedK7Operations !== 12");
    expect(WORKER_SOURCE).toContain("nativeK7Operations !== 5");
    expect(WORKER_SOURCE).toContain("sumQuanta(k1) !== 819");
    expect(WORKER_SOURCE).toContain("sumQuanta(snake) !== 1_611");
    expect(WORKER_SOURCE).toContain("sumQuanta(add) !== 690");
    expect(WORKER_SOURCE).toContain("unchangedOtherOwners: true");

    const plan = planAceVaeDecoder(512);
    const cooperative = planAceVaeDecoderQuanta(plan);
    const counts = plan.operations.map((operation, operationIndex) => ({
      operation,
      quantumCount: cooperative.quanta.filter((quantum) =>
        quantum.operationIndex === operationIndex
      ).length,
    }));
    const routes = new Map(ACE_OPT_0057_VAE_K7_ROUTES.map((route) => [
      route.operationLabel,
      route,
    ]));
    const k7 = counts.filter(({ operation }) => routes.has(operation.label));
    expect(k7).toHaveLength(17);
    expect(k7.reduce((sum, entry) => sum + entry.quantumCount, 0)).toBe(4_090);
    expect(k7.filter(({ operation }) =>
      routes.get(operation.label)!.owner === "row-reuse-k4"
    ).reduce((sum, entry) => sum + entry.quantumCount, 0)).toBe(3_360);
    expect(k7.filter(({ operation }) =>
      routes.get(operation.label)!.owner === "native-scalar-fp32"
    ).reduce((sum, entry) => sum + entry.quantumCount, 0)).toBe(730);
    expect(counts.filter(({ operation }) =>
      operation.kind === "conv-transpose1d"
    ).map(({ operation, quantumCount }) => [operation.label, quantumCount]))
      .toEqual([
        ["block-0-conv-t1", 92],
        ["block-1-conv-t1", 138],
        ["block-2-conv-t1", 138],
        ["block-3-conv-t1", 138],
        ["block-4-conv-t1", 138],
      ]);
  });

  it("uses a selected-label native-layout OPT-0024 K4 oracle and separate scalar envelope", () => {
    expect(WORKER_SOURCE).toContain(
      "createNativeLayoutK4OracleBackend",
    );
    expect(WORKER_SOURCE).toContain(
      "replaceSelectedK7WithNativeLayoutOpt0024",
    );
    expect(WORKER_SOURCE).toContain("selectAceOpt0057VaeK7(");
    expect(WORKER_SOURCE).toContain(
      "/-operation-\\d+-(.+)-quantum-\\d+$/u.exec(label)",
    );
    expect(WORKER_SOURCE).not.toContain(
      "label.includes(`-${route.operationLabel}-`)",
    );
    expect(WORKER_SOURCE).toContain(
      'selection.route.owner === "row-reuse-k4"',
    );
    expect(WORKER_SOURCE).toContain(
      "rawU16CandidateToNativeLayoutOpt0024K4Oracle",
    );
    expect(WORKER_SOURCE).toContain(
      'numericalEnvelopeAuthority: "OPT-0044-unchanged-from-OPT-0024"',
    );
    expect(WORKER_SOURCE).toContain(
      '"893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47"',
    );
    expect(OPT_0057_NUMERICAL_ENVELOPE).toEqual({
      nrmseMaximum: 0.003,
      snrMinimumDb: 50,
      pearsonMinimum: 0.9999,
      relativeRmsDriftMaximum: 0.005,
      relativeEnergyDriftMaximum: 0.005,
      relativePeakDriftMaximum: 0.01,
      relativeDcOffsetDriftMaximum: 0.001,
      relativeMaximumAbsoluteErrorMaximum: 0.02,
    });
    const reference = new Float32Array([0.25, -0.5, 0.75, -1]);
    expect(compareOpt0057Waveforms(reference, reference).passed).toBe(true);
    expect(compareOpt0057Waveforms(
      reference,
      new Float32Array([1, 1, 1, 1]),
    ).passed).toBe(false);
  });

  it("proves the frozen final-output raw oracle is confounded by ConvTranspose arithmetic", () => {
    expect(WORKER_SOURCE).toContain(
      "createNativeLayoutK4OracleBackend",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "compareOpt0057Raw(k4OracleRuns[0]!.output, candidateRuns[0]!.output)",
    );
    expect(EXACT_TRANSPOSE_SOURCE).toContain(
      "for (var inner = 0u; inner < INPUT_CHANNELS; inner += 1u)",
    );
    expect(EXACT_TRANSPOSE_SOURCE).toContain(
      "sum${row} = sum${row} + vec4<f32>(a${row}) * weight_value",
    );
    expect(K4_TRANSPOSE_SOURCE).toContain(
      "let partial${row}_0 = vec4<f16>",
    );
    expect(K4_TRANSPOSE_SOURCE).toContain(
      "vec4<f32>(partial${row}_0)",
    );

    // Two Cin4 groups are enough to disprove arithmetic identity. The exact
    // owner accumulates every FP16 product in FP32; OPT-0048 rounds each K4
    // partial to FP16 before widening it into the FP32 running state.
    const input = [
      0x357a, 0xaddc, 0x3568, 0x3c5c,
      0xb786, 0xb6a8, 0x389d, 0x267a,
    ];
    const weight = [
      0x3c6f, 0xb9cf, 0xb8ae, 0x24de,
      0xb582, 0x3d36, 0x2ae6, 0xba5b,
    ];
    expect(exactScalarFp32ThenFp16(input, weight)).toBe(0xae69);
    expect(k4Fp16PartialThenFp32ThenFp16(input, weight)).toBe(0xae68);
  });

  it("checks complete raw-U16 identity and deterministic reloaded owners", () => {
    const a = new Float32Array([1, -0, 3.5]);
    const b = new Float32Array(a);
    expect(compareOpt0057Raw(a, b)).toMatchObject({
      comparedU32WordCount: 3,
      comparedU16WordCount: 6,
      u32MismatchCount: 0,
      u16MismatchCount: 0,
      rawU32Exact: true,
      rawU16Exact: true,
    });
    b[1] = 0;
    expect(compareOpt0057Raw(a, b)).toMatchObject({
      u32MismatchCount: 1,
      u16MismatchCount: 1,
      rawU32Exact: false,
      rawU16Exact: false,
    });
    expect(WORKER_SOURCE).toContain(
      'export const OPT_0057_CORRECTNESS_ORDER = Object.freeze([\n' +
        '  "rev6-scalar",\n  "rev6-scalar",\n' +
        '  "rev6-k4-oracle",\n  "rev6-k4-oracle",\n' +
        '  "rev7-candidate",\n  "rev7-candidate",',
    );
    expect(WORKER_SOURCE).toContain("scalarDeterminism");
    expect(WORKER_SOURCE).toContain("k4OracleDeterminism");
    expect(WORKER_SOURCE).toContain("candidateDeterminism");
    expect(WORKER_SOURCE).toContain("firstFrameFinite");
    expect(WORKER_SOURCE).toContain("lastFrameFinite");
  });

  it("gates balanced K7 median and both decoder directions while reporting outer wall", () => {
    expect(OPT_0057_REQUIRED_K7_SPEEDUP).toBe(1.5);
    const passing = timingSamples({
      scalarK7: [150, 165],
      candidateK7: [90, 100],
      scalarDecoder: [250, 260],
      candidateDecoder: [220, 230],
    });
    expect(evaluateOpt0057BalancedTiming(passing)).toMatchObject({
      aggregate: {
        medianK7Speedup: 1.6578947368421053,
        requiredMedianK7Speedup: 1.5,
        medianK7Passed: true,
      },
      outerWindowWallReportedButNotGating: true,
      passed: true,
    });
    const slowReverse = timingSamples({
      scalarK7: [150, 165],
      candidateK7: [90, 170],
      scalarDecoder: [250, 260],
      candidateDecoder: [220, 270],
    });
    expect(evaluateOpt0057BalancedTiming(slowReverse).passed).toBe(false);
    expect(WORKER_SOURCE).toContain(
      'export const OPT_0057_TIMED_ORDER = Object.freeze([\n' +
        '  "rev6-scalar",\n  "rev7-candidate",\n' +
        '  "rev7-candidate",\n  "rev6-scalar",',
    );
    expect(WORKER_SOURCE).toContain(
      "requiresBothDirectionsToImproveHomogeneousK7Wall: true",
    );
    expect(WORKER_SOURCE).toContain(
      "requiresNoCompleteDecoderRegressionInBothPairedOrders: true",
    );
    expect(WORKER_SOURCE).toContain(
      "convTransposeAndOuterWallsReportedButNotGating: true",
    );
  });

  it("accepts exactly one truthful external level-0 check after 30 seconds", () => {
    const valid = thermalParameters({
      started: 10_000,
      checked: 40_001,
      observations: 1,
      level: 0,
      gap: "",
    });
    expect(parseOpt0057ThermalGate(valid, 9_999, 40_001)).toEqual({
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
        observations: 2,
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
      expect(() => parseOpt0057ThermalGate(invalid, 9_999, 40_001))
        .toThrow(/one truthful level-0/);
    }
    expect(CONTRACT_SOURCE).toContain("observationCount !== 1");
    expect(CONTRACT_SOURCE).not.toContain("pollMilliseconds");
    expect(HTML_SOURCE).toContain("leave the machine idle for at least 30 seconds");
  });

  it("is button-gated, identity-bound, downloadable, and changes no default", () => {
    expect(PAGE_SOURCE).toContain('prepareButton.addEventListener("click"');
    expect(PAGE_SOURCE.indexOf("new Worker(")).toBeGreaterThan(
      PAGE_SOURCE.indexOf('prepareButton.addEventListener("click"'),
    );
    expect(PAGE_SOURCE).toContain("parseOpt0018RunIdentity(");
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0057_RESULT__ = receipt");
    expect(PAGE_SOURCE).toContain("URL.createObjectURL(new Blob(");
    expect(PAGE_SOURCE).toContain("thermalStarted.value = String(Date.now())");
    expect(HTML_SOURCE).toContain('id="thermal-gate" disabled');
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(HTML_SOURCE).toContain(
      'src="./opt-0057-vae-k7-k4-row-reuse-c512.ts"',
    );
    expect(WORKER_SOURCE).toContain("productionDefaultChanged: false");
    expect(WORKER_SOURCE).toContain("productSelectionAuthorized: false");
    expect(WORKER_SOURCE).toContain("listeningApprovalStillRequired: true");
    expect(WORKER_SOURCE).toContain("under60SecondClaim: false");
    expect(WORKER_SOURCE).not.toContain("timestamp-query");
  });
});

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

function timingSamples(values: Readonly<{
  scalarK7: readonly [number, number];
  candidateK7: readonly [number, number];
  scalarDecoder: readonly [number, number];
  candidateDecoder: readonly [number, number];
}>): readonly Opt0057TimingSample[] {
  return [
    sample("rev6-scalar", values.scalarK7[0], values.scalarDecoder[0]),
    sample("rev7-candidate", values.candidateK7[0], values.candidateDecoder[0]),
    sample("rev7-candidate", values.candidateK7[1], values.candidateDecoder[1]),
    sample("rev6-scalar", values.scalarK7[1], values.scalarDecoder[1]),
  ];
}

function sample(
  arm: Opt0057TimingSample["arm"],
  k7: number,
  decoder: number,
): Opt0057TimingSample {
  return {
    arm,
    k7FamilySubmitThroughDrainMs: k7,
    convTransposeFamilySubmitThroughDrainMs: 50,
    decoderSubmitThroughDrainMs: decoder,
    outerWindowWallMs: decoder + 20,
  };
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

function exactScalarFp32ThenFp16(
  inputBits: readonly number[],
  weightBits: readonly number[],
): number {
  let sum = Math.fround(0);
  for (let index = 0; index < inputBits.length; index += 1) {
    const product = Math.fround(
      float16BitsToNumber(inputBits[index]!) *
        float16BitsToNumber(weightBits[index]!),
    );
    sum = Math.fround(sum + product);
  }
  return numberToFloat16Bits(sum);
}

function k4Fp16PartialThenFp32ThenFp16(
  inputBits: readonly number[],
  weightBits: readonly number[],
): number {
  if (inputBits.length !== weightBits.length || inputBits.length % 4 !== 0) {
    throw new Error("K4 counterexample requires complete paired Cin4 groups");
  }
  let sum = Math.fround(0);
  for (let base = 0; base < inputBits.length; base += 4) {
    let partial = Math.fround(0);
    for (let lane = 0; lane < 4; lane += 1) {
      const index = base + lane;
      const product = Math.fround(
        float16BitsToNumber(inputBits[index]!) *
          float16BitsToNumber(weightBits[index]!),
      );
      partial = Math.fround(partial + product);
    }
    const roundedPartial = float16BitsToNumber(
      numberToFloat16Bits(partial),
    );
    sum = Math.fround(sum + roundedPartial);
  }
  return numberToFloat16Bits(sum);
}
