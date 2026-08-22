import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0019_DENSE_WORKGROUP_STORAGE_BYTES,
  planAceOpt0019DenseCooperativePanels,
} from "../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.js";
import {
  ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
  ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE,
  ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE,
  ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
  aceOpt0021DenseCooperativeVec4PanelsWgsl,
  planAceOpt0021DenseCooperativeVec4Panels,
} from "../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.js";
import {
  buildOpt0021FixtureDeclaration,
  buildOpt0021ShapeSpecs,
  buildOpt0021TimingOrders,
  compareOpt0021ExactWords,
  fillOpt0021ActivationFixture,
  fillOpt0021PackedWeightFixture,
  opt0021ActivationBitsAt,
  opt0021PackedWeightBitsAt,
  parseOpt0021ThermalGate,
  summarizeOpt0021Timing,
  type Opt0021Arm,
  type Opt0021TimingInput,
} from "./browser/opt-0021-dit-dense-vec4-panels.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0021-dit-dense-vec4-panels.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0021-dit-dense-vec4-panels.html",
  import.meta.url,
));
const CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const ARMS = ["A", "B", "C"] as const;
const EXPECTED_HARNESS_SHA256 =
  "e0a019be08bc6222594de5e5083cc8cef1395d8395ed003661bc1e71c8cea055";
const EXPECTED_HTML_SHA256 =
  "84fb23733bc30a6d37d0e40418f9fee22a37c0a87c13be3e15e81c75f1690ffa";
const EXPECTED_FIXTURE_HASHES = Object.freeze([
  Object.freeze({
    id: "h-h",
    activationSha256:
      "e66bc914da370971b0a717a3db8e9fa5b26820fe2d0bd5fa156b650f68fc5b99",
    packedWeightSha256:
      "6e985fca3119b135d740c2f8814fe3ddbf538bdaabde96cfca1dd363b62d45eb",
  }),
  Object.freeze({
    id: "h-1024",
    activationSha256:
      "47b9e4ecef742a678bb263443e783b5ea753dc513af2b986c15d97bbafb315cf",
    packedWeightSha256:
      "3e4db4f1da0770dc2b2ffd6f2b8643e1778b2993495d4a2085bcc0bea7b383f3",
  }),
  Object.freeze({
    id: "h-6144",
    activationSha256:
      "1469272232904084304f9834d5b9f1cf152ba940c7e25e82e0baac0708838c62",
    packedWeightSha256:
      "898266fc61785f391230299844c4e50e6394daa631783e3a40f95ba436cb088c",
  }),
  Object.freeze({
    id: "6144-h",
    activationSha256:
      "39604dec14faf7d5f8a2c2400f48ce162cbce268de9b176ac7acf2e84612b1aa",
    packedWeightSha256:
      "e20d4d09edb58957967021c4f4957653be93ab3bcca5828d7a6c32403f2c24da",
  }),
]);
const EXPECTED_CANDIDATE_SHADER_HASHES = Object.freeze({
  "h-h": "9b79d3301c6b7f63d09d6b85cbccc941f45004272cb13e963310bb1418408b3f",
  "h-1024": "9cf6a82955323a01e7aa512d869f3f7cd28454cccdcad4bb2554a11e4435b42f",
  "h-6144": "8c6ccb19cab202338d66dcb140533b7859b1c0cbe94e74297a57df6f646da5d1",
  "6144-h": "654e97942df80d7ffcf24b650e2589fc4697bd2251a99ae2a36c2f85789d023b",
});

describe("OPT-0021 target-browser dense vec4-panel gate", () => {
  it("pins the exact three arms, four M2250 shapes, and B/C accounting", () => {
    expect(buildOpt0021ShapeSpecs().map(({ id, shape,
      productionMultiplicity, feedForwardMultiplicity }) => ({
      id,
      shape,
      productionMultiplicity,
      feedForwardMultiplicity,
    }))).toEqual([
      {
        id: "h-h",
        shape: { rows: 2_250, inner: 2_048, columns: 2_048 },
        productionMultiplicity: 4,
        feedForwardMultiplicity: 0,
      },
      {
        id: "h-1024",
        shape: { rows: 2_250, inner: 2_048, columns: 1_024 },
        productionMultiplicity: 2,
        feedForwardMultiplicity: 0,
      },
      {
        id: "h-6144",
        shape: { rows: 2_250, inner: 2_048, columns: 6_144 },
        productionMultiplicity: 2,
        feedForwardMultiplicity: 2,
      },
      {
        id: "6144-h",
        shape: { rows: 2_250, inner: 6_144, columns: 2_048 },
        productionMultiplicity: 1,
        feedForwardMultiplicity: 1,
      },
    ]);
    let currentWorkgroups = 0;
    let cooperativeWorkgroups = 0;
    let panelIterations = 0;
    let scheduledMacs = 0;
    let validMacs = 0;
    let activationBytes = 0;
    let weightBytes = 0;
    let barriers = 0;
    for (const spec of buildOpt0021ShapeSpecs()) {
      const A = planAceOpt0009DenseGemm(spec.shape);
      const B = planAceOpt0019DenseCooperativePanels(spec.shape);
      const C = planAceOpt0021DenseCooperativeVec4Panels(spec.shape);
      expect(A).toMatchObject({
        tileRows: 32,
        tileColumns: 256,
        tileInner: 32,
        workgroupSize: 128,
      });
      expect(B).toMatchObject({
        tileRows: 64,
        tileColumns: 128,
        tileInner: 16,
        workgroupSize: 256,
        workgroupStorageBytes: 6_400,
      });
      expect(C).toMatchObject({
        kernelSetId:
          "opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1",
        tileRows: 64,
        tileColumns: 128,
        tileInner: 16,
        workgroupSize: 256,
        workgroupSizeX: 16,
        workgroupSizeY: 16,
        inputPanelStride: 17,
        inputPanelElements: 272,
        weightPanelStride: 33,
        weightPanelElements: 528,
        workgroupStorageBytes: 6_400,
      });
      expect(C.workgroupCount).toBe(B.workgroupCount);
      expect(C.scheduledMultiplyAdds).toBe(B.scheduledMultiplyAdds);
      expect(C.validMultiplyAdds).toBe(B.validMultiplyAdds);
      expect(C.estimatedGlobalActivationBytes)
        .toBe(B.estimatedGlobalActivationBytes);
      expect(C.estimatedGlobalWeightBytes).toBe(B.estimatedGlobalWeightBytes);
      expect(C.estimatedGlobalOperandBytes).toBe(B.estimatedGlobalOperandBytes);
      expect(C.estimatedGlobalOutputBytes).toBe(B.estimatedGlobalOutputBytes);
      expect(C.barrierEvents).toBe(B.barrierEvents);
      expect(C.packedWeightStorageShape).toEqual(B.packedWeightStorageShape);
      currentWorkgroups += A.workgroupCount * spec.productionMultiplicity;
      cooperativeWorkgroups += B.workgroupCount * spec.productionMultiplicity;
      panelIterations += B.workgroupCount * B.innerTiles *
        spec.productionMultiplicity;
      scheduledMacs += B.scheduledMultiplyAdds * spec.productionMultiplicity;
      validMacs += B.validMultiplyAdds * spec.productionMultiplicity;
      activationBytes += B.estimatedGlobalActivationBytes *
        spec.productionMultiplicity;
      weightBytes += B.estimatedGlobalWeightBytes *
        spec.productionMultiplicity;
      barriers += B.barrierEvents * spec.productionMultiplicity;
    }
    expect(ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID)
      .toBe("opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1");
    expect(ACE_OPT_0019_DENSE_WORKGROUP_STORAGE_BYTES).toBe(6_400);
    expect(ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE).toBe(17);
    expect(ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE).toBe(33);
    expect(ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES).toBe(6_400);
    expect(buildOpt0021ShapeSpecs().reduce((sum, spec) =>
      sum + spec.shape.rows * spec.shape.columns * 3, 0
    )).toBe(76_032_000);
    expect({
      currentWorkgroups,
      cooperativeWorkgroups,
      panelIterations,
      scheduledMacs,
      validMacs,
      activationBytes,
      weightBytes,
      totalOperandBytes: activationBytes + weightBytes,
      barriers,
    }).toEqual({
      currentWorkgroups: 6_816,
      cooperativeWorkgroups: 6_912,
      panelIterations: 1_032_192,
      scheduledMacs: 135_291_469_824,
      validMacs: 132_120_576_000,
      activationBytes: 4_128_768_000,
      weightBytes: 4_227_858_432,
      totalOperandBytes: 8_356_626_432,
      barriers: 2_064_384,
    });
  });

  it("freezes all fixture bytes and their eight per-shape SHA-256 identities", {
    timeout: 120_000,
  }, () => {
    const received: Array<Record<string, string>> = [];
    for (const [shapeIndex, spec] of buildOpt0021ShapeSpecs().entries()) {
      const activation = new Float32Array(spec.shape.rows * spec.shape.inner);
      fillOpt0021ActivationFixture(activation, shapeIndex);
      for (const index of [
        0,
        1,
        spec.shape.inner - 1,
        activation.length - 1,
      ]) {
        const row = Math.floor(index / spec.shape.inner);
        const inner = index % spec.shape.inner;
        expect(activation[index]).toBe(halfToNumber(
          opt0021ActivationBitsAt(shapeIndex, row, inner),
        ));
      }
      const activationSha256 = hashBytes(new Uint8Array(activation.buffer));
      const weight = new Uint16Array(spec.shape.inner * spec.shape.columns);
      fillOpt0021PackedWeightFixture(weight, shapeIndex);
      const packedWeightSha256 = hashBytes(new Uint8Array(weight.buffer));
      received.push({ id: spec.id, activationSha256, packedWeightSha256 });
    }
    expect(received).toEqual(EXPECTED_FIXTURE_HASHES);
    expect(buildOpt0021ShapeSpecs().map((spec) => ({
      id: spec.id,
      activationSha256: spec.fixture.activationSha256,
      packedWeightSha256: spec.fixture.packedWeightSha256,
    }))).toEqual(EXPECTED_FIXTURE_HASHES);
  });

  it("proves the copied generator declaration, signs, cancellation, and K6144", () => {
    const declaration = buildOpt0021FixtureDeclaration() as {
      version: string;
      finiteNonPowerFp16MagnitudeBits: readonly number[];
      activationSeeds: readonly number[];
      weightSeeds: readonly number[];
      constants: Readonly<Record<string, unknown>>;
    };
    expect(hashBytes(Buffer.from(canonicalJson(declaration))))
      .toBe("954aff0a07dcc2946ac8191c054e2ecf63473d05ed9ebf81dc7db6d535f80f0c");
    expect(declaration.version).toBe("opt-0020-finite-fp16-cancellation-v1");
    expect(declaration.activationSeeds).toEqual([
      0x3141_5926, 0x2718_2818, 0x6a09_e667, 0xbb67_ae85,
    ]);
    expect(declaration.weightSeeds).toEqual([
      0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c,
    ]);
    expect(declaration.finiteNonPowerFp16MagnitudeBits).toEqual([
      0x2411, 0x28b5, 0x2d53, 0x31e7, 0x356b, 0x39ad,
    ]);
    expect(declaration.constants).toEqual({
      subjectMix: 0x9e37_79b1,
      groupMix: 0x85eb_ca6b,
      offsetMix: 0xc2b2_ae35,
      pairedCancellationGroups:
        "g and g^1 except every eighth group's breaker",
      withinK4ProductSignPattern: "+--+",
      groupSignFlip: "activation sign xor (floor(k/4) & 1)",
    });
    const exponentBands = new Set<number>();
    for (const bits of declaration.finiteNonPowerFp16MagnitudeBits) {
      const exponent = bits >>> 10 & 0x1f;
      expect(exponent).toBeGreaterThan(0);
      expect(exponent).toBeLessThan(0x1f);
      expect(bits & 0x03ff).not.toBe(0);
      exponentBands.add(exponent);
    }
    expect(exponentBands.size).toBe(6);
    expect(new Set(Array.from({ length: 64 }, (_, row) =>
      opt0021ActivationBitsAt(0, row, 8) >>> 15
    ))).toEqual(new Set([0, 1]));
    expect(new Set(Array.from({ length: 64 }, (_, column) =>
      opt0021PackedWeightBitsAt(0, 8, column) >>> 15
    ))).toEqual(new Set([0, 1]));

    const productSigns = (shapeIndex: number, group: number) => Array.from(
      { length: 4 },
      (_, offset) => {
        const inner = group * 4 + offset;
        return (opt0021ActivationBitsAt(shapeIndex, 7, inner) ^
          opt0021PackedWeightBitsAt(shapeIndex, inner, 11)) >>> 15;
      },
    );
    for (const [shapeIndex, group] of [[0, 2], [3, 1_532]] as const) {
      const even = productSigns(shapeIndex, group);
      const odd = productSigns(shapeIndex, group + 1);
      expect(even[0]).toBe(even[3]);
      expect(even[1]).toBe(even[2]);
      expect(even[0]).not.toBe(even[1]);
      expect(odd).toEqual(even.map((sign) => sign ^ 1));
      for (let offset = 0; offset < 4; offset += 1) {
        const k0 = group * 4 + offset;
        const k1 = (group + 1) * 4 + offset;
        expect(opt0021ActivationBitsAt(shapeIndex, 7, k0) & 0x7fff)
          .toBe(opt0021ActivationBitsAt(shapeIndex, 7, k1) & 0x7fff);
        expect(opt0021PackedWeightBitsAt(shapeIndex, k0, 11) & 0x7fff)
          .toBe(opt0021PackedWeightBitsAt(shapeIndex, k1, 11) & 0x7fff);
      }
    }
    const long = buildOpt0021ShapeSpecs()[3]!;
    expect(long.shape.inner).toBe(6_144);
    for (const bits of [
      opt0021ActivationBitsAt(3, 2_249, 6_143),
      opt0021PackedWeightBitsAt(3, 6_143, 2_047),
    ]) {
      expect(bits >>> 10 & 0x1f).toBeGreaterThan(0);
      expect(bits >>> 10 & 0x1f).toBeLessThan(0x1f);
      expect(bits & 0x03ff).not.toBe(0);
    }
  });

  it("pins the candidate source and all four generated shader identities", () => {
    expect(hashBytes(readFileSync(CORE_PATH)))
      .toBe("2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1");
    const received = Object.fromEntries(buildOpt0021ShapeSpecs().map((spec) => [
      spec.id,
      hashBytes(Buffer.from(
        aceOpt0021DenseCooperativeVec4PanelsWgsl(spec.shape),
      )),
    ]));
    expect(received).toEqual(EXPECTED_CANDIDATE_SHADER_HASHES);
  });

  it("compares every raw U32 word and reports all mismatches", () => {
    const expected = new Uint32Array([
      0x0000_0000, 0x8000_0000, 0x3f80_0000, 0x7fc0_2155,
    ]);
    expect(compareOpt0021ExactWords(expected, new Uint32Array(expected)))
      .toEqual({
        comparedU32Count: 4,
        mismatchCount: 0,
        firstMismatch: null,
        rawU32Exact: true,
      });
    const actual = new Uint32Array([
      0x8000_0000, 0x8000_0000, 0x3f80_0001, 0x7fc0_2155,
    ]);
    expect(compareOpt0021ExactWords(expected, actual)).toEqual({
      comparedU32Count: 4,
      mismatchCount: 2,
      firstMismatch: {
        index: 0,
        expectedWord: 0,
        actualWord: 0x8000_0000,
      },
      rawU32Exact: false,
    });
    expect(() => compareOpt0021ExactWords(expected, new Uint32Array(3)))
      .toThrow(/length changed/);
  });

  it("uses every arm permutation once and rotates shape order by round mod four", () => {
    const orders = buildOpt0021TimingOrders();
    expect(orders).toHaveLength(24);
    expect(Array.from({ length: 6 }, (_, roundIndex) =>
      orders.filter((entry) => entry.roundIndex === roundIndex)
        .map((entry) => entry.shapeIndex)
    )).toEqual([
      [0, 1, 2, 3],
      [1, 2, 3, 0],
      [2, 3, 0, 1],
      [3, 0, 1, 2],
      [0, 1, 2, 3],
      [1, 2, 3, 0],
    ]);
    const permutations = [
      ["A", "B", "C"],
      ["A", "C", "B"],
      ["B", "A", "C"],
      ["B", "C", "A"],
      ["C", "A", "B"],
      ["C", "B", "A"],
    ];
    const positionCounts = Object.fromEntries(ARMS.map((arm) =>
      [arm, [0, 0, 0]]
    )) as Record<Opt0021Arm, number[]>;
    for (const entry of orders) {
      expect(entry.order).toEqual(permutations[entry.roundIndex]);
      entry.order.forEach((arm, position) => {
        positionCounts[arm]![position]! += 1;
      });
    }
    expect(positionCounts).toEqual({
      A: [8, 8, 8],
      B: [8, 8, 8],
      C: [8, 8, 8],
    });
  });

  it("uses median six and conjunctively enforces every frozen speed gate", () => {
    const passing = timingInputs([
      [20, 15, 10],
      [20, 15, 10],
      [50, 30, 20],
      [50, 30, 20],
    ]);
    expect(summarizeOpt0021Timing(passing)).toMatchObject({
      completeDense: {
        milliseconds: { A: 270, B: 180, C: 120 },
        bToCSpeedup: 1.5,
        bToCThreshold: 1.075,
        aToCSavingMilliseconds: 150,
        aToCSavingThresholdMilliseconds: 52.0834,
      },
      feedForward: {
        milliseconds: { A: 150, B: 90, C: 60 },
      },
      everyShapeCFasterThanB: true,
      passed: true,
      decision: "positive-package-layer-gate-authorized",
    });
    expect((summarizeOpt0021Timing(timingInputs([
      [20, 10, 10],
      [20, 15, 10],
      [50, 30, 20],
      [50, 30, 20],
    ]))["passed"])).toBe(false);
    expect((summarizeOpt0021Timing(timingInputs([
      [20, 10.749, 10],
      [20, 10.749, 10],
      [20, 10.749, 10],
      [80, 10.749, 10],
    ]))["passed"])).toBe(false);
    const savingMiss = summarizeOpt0021Timing(timingInputs([
      [10, 15, 10],
      [10, 15, 10],
      [10, 15, 10],
      [62.083_399, 15, 10],
    ]));
    expect(savingMiss).toMatchObject({
      everyShapeCFasterThanB: true,
      passed: false,
      decision: "negative-stop-primitive-gate",
    });
    const medianProof = summarizeOpt0021Timing([
      {
        id: "h-h",
        samples: { A: [100, 1, 4, 8, 7, 2], B: sixAround(8), C: sixAround(4) },
      },
      ...timingInputs([[0, 0, 0], [20, 15, 10], [20, 15, 10], [20, 15, 10]])
        .slice(1),
    ]);
    expect((medianProof["strata"] as readonly Readonly<{
      medians: Readonly<Record<Opt0021Arm, number>>;
    }>[])[0]!.medians.A).toBe(5.5);
  });

  it("accepts only a fresh finite positive 30-second nominal thermal gate", () => {
    const valid = () => new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0021ThermalGate(valid(), 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      maximumPollGapMilliseconds: 1_010,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    for (const mutate of [
      (parameters: URLSearchParams) => parameters.set("thermalSource", "wrong"),
      (parameters: URLSearchParams) => parameters.set("thermalObservations", "30"),
      (parameters: URLSearchParams) => parameters.set("thermalObservations", "31.5"),
      (parameters: URLSearchParams) => parameters.set("thermalPollMilliseconds", "999"),
      (parameters: URLSearchParams) => parameters.set("thermalMaximumPollGapMilliseconds", "0"),
      (parameters: URLSearchParams) => parameters.set("thermalMaximumPollGapMilliseconds", "-1"),
      (parameters: URLSearchParams) => parameters.set("thermalMaximumPollGapMilliseconds", "1251"),
      (parameters: URLSearchParams) => parameters.set("thermalMaximumPollGapMilliseconds", "NaN"),
      (parameters: URLSearchParams) => parameters.set("thermalNonNominalObservations", "1"),
      (parameters: URLSearchParams) => parameters.set("thermalCompletedAtEpochMilliseconds", "31999"),
    ]) {
      const parameters = valid();
      mutate(parameters);
      expect(() => parseOpt0021ThermalGate(parameters, 1_999, 32_020))
        .toThrow();
    }
    expect(() => parseOpt0021ThermalGate(valid(), 2_001, 32_020)).toThrow();
    expect(() => parseOpt0021ThermalGate(valid(), 1_999, 37_011)).toThrow();
    expect(() => parseOpt0021ThermalGate(valid(), Number.NaN, 32_020)).toThrow();
    expect(() => parseOpt0021ThermalGate(valid(), 1_999, Number.POSITIVE_INFINITY))
      .toThrow();
  });

  it("freezes source authority, complete exactness, bounded timing, and cleanup", () => {
    expect(hashBytes(Buffer.from(HARNESS_SOURCE))).toBe(EXPECTED_HARNESS_SHA256);
    expect(hashBytes(Buffer.from(HTML_SOURCE))).toBe(EXPECTED_HTML_SHA256);
    for (const pin of [
      "7dcbe50c7f04d1e07f6b30657da96372ca8574d1",
      "2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1",
      "954aff0a07dcc2946ac8191c054e2ecf63473d05ed9ebf81dc7db6d535f80f0c",
      "sha256-canonical-json-recursive-sorted-object-keys-v1",
      "executionOrder: Object.freeze([\"A\", \"B\", \"C\", \"C\"]",
      "FULL_EXACT_COMPARED_U32_COUNT = 3 * FULL_OUTPUT_U32_COUNT",
      "comparisonsPerOutputWord: 3",
      "aVersusBRawU32Exact = correctnessCases.every",
      "aVersusCFirstRawU32Exact = correctnessCases.every",
      "cFirstVersusRerunRawU32Exact = correctnessCases.every",
      "allFourExecutionsShareShapeHash = correctnessCases.every",
      "aggregateOutputHashManifestSha256",
      "persistentWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT",
      "OUTPUT_PREFILL_QNAN_U32",
      "prefixCanaryIntact",
      "suffixCanaryIntact",
      "tailRowWritten",
      "continuousExternalThermalTraceRequiredThroughCleanup: true",
      "unchangedThermalRetryPerformed: false",
      "timingSkippedBecauseCorrectnessOrLifecycleGateFailed: true",
      'disposition: "negative-stop-exactness-gate"',
      "packageNativeLayerGateAuthorized",
      "productionIntegrationAuthorized: false",
      "c98TrajectoryRunAuthorized: false",
      "m2250ProductRunAuthorized: false",
      "zeroLiveResources",
      "deviceDestroyed: true",
      "__ACE_OPT0021_RESULT__",
    ]) expect(HARNESS_SOURCE).toContain(pin);
    expect(HARNESS_SOURCE).toContain("?raw");
    expect(HARNESS_SOURCE).not.toContain(
      "./opt-0020-dit-dense-cooperative-dot4",
    );
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HARNESS_SOURCE).not.toContain("PENDING_");
    expect(HARNESS_SOURCE).not.toContain("__OPT0021_");
    expect(HARNESS_SOURCE).not.toMatch(/\b(?:ulp|nrmse|pearson|tolerance)\b/i);
    expect(HARNESS_SOURCE).not.toContain("outputWords:");
    const comparisonFunction = HARNESS_SOURCE.slice(
      HARNESS_SOURCE.indexOf("function compareExactWords("),
      HARNESS_SOURCE.indexOf("function shapeSpec("),
    );
    expect(comparisonFunction).toContain("compareOpt0021ExactWords");
    expect(comparisonFunction).not.toContain("throw new Error");
    expect(HARNESS_SOURCE).toContain("buildCorrectnessStopReceipt(value)");
    const timingFunction = HARNESS_SOURCE.slice(
      HARNESS_SOURCE.indexOf("async function executeTimedDispatch("),
      HARNESS_SOURCE.indexOf("async function createActivationBuffer("),
    );
    expect(timingFunction.match(/dispatch\.encode\(pass\)/g)).toHaveLength(1);
    expect(timingFunction.match(/device\.queue\.submit/g)).toHaveLength(1);
    expect(timingFunction.match(/onSubmittedWorkDone/g)).toHaveLength(1);
    expect(HTML_SOURCE).toContain("continuous 30-second nominal");
    expect(HTML_SOURCE).toContain("Six rounds use every A/B/C permutation");
    expect(HTML_SOURCE).toContain("A, B, and both C executions must be raw-U32 identical");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0021-dit-dense-vec4-panels.ts"',
    );
  });
});

function timingInputs(
  medians: readonly (readonly [number, number, number])[],
): Opt0021TimingInput[] {
  return buildOpt0021ShapeSpecs().map((spec, index) => {
    const values = medians[index]!;
    return {
      id: spec.id,
      samples: {
        A: sixAround(values[0]),
        B: sixAround(values[1]),
        C: sixAround(values[2]),
      },
    };
  });
}

function sixAround(value: number): readonly number[] {
  if (value <= 2) return [value, value, value, value, value, value];
  return [value - 2, value - 1, value, value, value + 1, value + 2];
}

function hashBytes(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJsonValue(value));
}

function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonValue);
  if (value !== null && typeof value === "object") {
    const record = value as Readonly<Record<string, unknown>>;
    return Object.fromEntries(Object.keys(record).sort().map((key) =>
      [key, sortJsonValue(record[key])]
    ));
  }
  return value;
}

function halfToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 0x1f;
  const mantissa = bits & 0x03ff;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
}
