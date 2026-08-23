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
  ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES,
  planAceOpt0020DenseCooperativeDot4,
} from "../src/webgpu/kernels/dit-dense-fp16-cooperative-dot4.js";
import {
  buildOpt0020FixtureDeclaration,
  buildOpt0020ShapeSpecs,
  buildOpt0020TimingOrders,
  compareOpt0020Numerics,
  fillOpt0020ActivationFixture,
  fillOpt0020PackedWeightFixture,
  opt0020ActivationBitsAt,
  opt0020PackedWeightBitsAt,
  parseOpt0020ThermalGate,
  summarizeOpt0020Timing,
  type Opt0020Arm,
  type Opt0020TimingInput,
} from "./browser/opt-0020-dit-dense-cooperative-dot4.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0020-dit-dense-cooperative-dot4.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0020-dit-dense-cooperative-dot4.html",
  import.meta.url,
));
const CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-dot4.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const ARMS = ["A", "B", "C"] as const;
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

describe("OPT-0020 target-browser cooperative dense dot4 gate", () => {
  it("pins the exact three arms, four M2250 shapes, and B/C traffic identity", () => {
    expect(buildOpt0020ShapeSpecs().map(({ id, shape,
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
    let scheduledMacs = 0;
    let validMacs = 0;
    let activationBytes = 0;
    let weightBytes = 0;
    let barriers = 0;
    for (const spec of buildOpt0020ShapeSpecs()) {
      const A = planAceOpt0009DenseGemm(spec.shape);
      const B = planAceOpt0019DenseCooperativePanels(spec.shape);
      const C = planAceOpt0020DenseCooperativeDot4(spec.shape);
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
      });
      expect(C).toMatchObject({
        tileRows: 64,
        tileColumns: 128,
        tileInner: 16,
        workgroupSize: 256,
        workgroupStorageBytes: 6_528,
      });
      expect(C.workgroupCount).toBe(B.workgroupCount);
      expect(C.scheduledMultiplyAdds).toBe(B.scheduledMultiplyAdds);
      expect(C.validMultiplyAdds).toBe(B.validMultiplyAdds);
      expect(C.estimatedGlobalActivationBytes)
        .toBe(B.estimatedGlobalActivationBytes);
      expect(C.estimatedGlobalWeightBytes).toBe(B.estimatedGlobalWeightBytes);
      expect(C.barrierEvents).toBe(B.barrierEvents);
      currentWorkgroups += A.workgroupCount * spec.productionMultiplicity;
      cooperativeWorkgroups += B.workgroupCount * spec.productionMultiplicity;
      scheduledMacs += B.scheduledMultiplyAdds * spec.productionMultiplicity;
      validMacs += B.validMultiplyAdds * spec.productionMultiplicity;
      activationBytes += B.estimatedGlobalActivationBytes *
        spec.productionMultiplicity;
      weightBytes += B.estimatedGlobalWeightBytes *
        spec.productionMultiplicity;
      barriers += B.barrierEvents * spec.productionMultiplicity;
    }
    expect(ACE_OPT_0019_DENSE_WORKGROUP_STORAGE_BYTES).toBe(6_400);
    expect(ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES).toBe(6_528);
    expect({
      currentWorkgroups,
      cooperativeWorkgroups,
      scheduledMacs,
      validMacs,
      activationBytes,
      weightBytes,
      totalOperandBytes: activationBytes + weightBytes,
      barriers,
    }).toEqual({
      currentWorkgroups: 6_816,
      cooperativeWorkgroups: 6_912,
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
    for (const [shapeIndex, spec] of buildOpt0020ShapeSpecs().entries()) {
      const activation = new Float32Array(spec.shape.rows * spec.shape.inner);
      fillOpt0020ActivationFixture(activation, shapeIndex);
      for (const index of [
        0,
        1,
        spec.shape.inner - 1,
        activation.length - 1,
      ]) {
        const row = Math.floor(index / spec.shape.inner);
        const inner = index % spec.shape.inner;
        expect(activation[index]).toBe(halfToNumber(
          opt0020ActivationBitsAt(shapeIndex, row, inner),
        ));
      }
      const activationSha256 = hashBytes(new Uint8Array(activation.buffer));
      const weight = new Uint16Array(spec.shape.inner * spec.shape.columns);
      fillOpt0020PackedWeightFixture(weight, shapeIndex);
      const packedWeightSha256 = hashBytes(new Uint8Array(weight.buffer));
      received.push({ id: spec.id, activationSha256, packedWeightSha256 });
    }
    expect(received).toEqual(EXPECTED_FIXTURE_HASHES);
    expect(buildOpt0020ShapeSpecs().map((spec) => ({
      id: spec.id,
      activationSha256: spec.fixture.activationSha256,
      packedWeightSha256: spec.fixture.packedWeightSha256,
    }))).toEqual(EXPECTED_FIXTURE_HASHES);
  });

  it("proves finite non-power exponent bands, seeded signs, K4 and long-K cancellation", () => {
    const declaration = buildOpt0020FixtureDeclaration() as {
      version: string;
      finiteNonPowerFp16MagnitudeBits: readonly number[];
      activationSeeds: readonly number[];
      weightSeeds: readonly number[];
      constants: Readonly<Record<string, unknown>>;
    };
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
      const mantissa = bits & 0x03ff;
      expect(exponent).toBeGreaterThan(0);
      expect(exponent).toBeLessThan(0x1f);
      expect(mantissa).not.toBe(0);
      exponentBands.add(exponent);
    }
    expect(exponentBands.size).toBe(6);

    const activationSigns = new Set(Array.from({ length: 64 }, (_, row) =>
      opt0020ActivationBitsAt(0, row, 8) >>> 15
    ));
    const weightSigns = new Set(Array.from({ length: 64 }, (_, column) =>
      opt0020PackedWeightBitsAt(0, 8, column) >>> 15
    ));
    expect(activationSigns).toEqual(new Set([0, 1]));
    expect(weightSigns).toEqual(new Set([0, 1]));

    const productSigns = (group: number) => Array.from(
      { length: 4 },
      (_, offset) => {
        const inner = group * 4 + offset;
        return (opt0020ActivationBitsAt(0, 7, inner) ^
          opt0020PackedWeightBitsAt(0, inner, 11)) >>> 15;
      },
    );
    const group2 = productSigns(2);
    const group3 = productSigns(3);
    expect(group2[0]).toBe(group2[3]);
    expect(group2[1]).toBe(group2[2]);
    expect(group2[0]).not.toBe(group2[1]);
    expect(group3).toEqual(group2.map((sign) => sign ^ 1));
    expect(opt0020ActivationBitsAt(0, 7, 8) >>> 15)
      .not.toBe(opt0020ActivationBitsAt(0, 7, 12) >>> 15);
    for (let offset = 0; offset < 4; offset += 1) {
      const k2 = 2 * 4 + offset;
      const k3 = 3 * 4 + offset;
      expect(opt0020ActivationBitsAt(0, 7, k2) & 0x7fff)
        .toBe(opt0020ActivationBitsAt(0, 7, k3) & 0x7fff);
      expect(opt0020PackedWeightBitsAt(0, k2, 11) & 0x7fff)
        .toBe(opt0020PackedWeightBitsAt(0, k3, 11) & 0x7fff);
    }
    for (const bitAt of [
      (offset: number) => opt0020ActivationBitsAt(0, 7, 8 + offset),
      (offset: number) => opt0020PackedWeightBitsAt(0, 8 + offset, 11),
    ]) {
      expect(bitAt(0) & 0x7fff).toBe(bitAt(3) & 0x7fff);
      expect(bitAt(1) & 0x7fff).toBe(bitAt(2) & 0x7fff);
    }
    const long = buildOpt0020ShapeSpecs()[3]!;
    expect(long.shape.inner).toBe(6_144);
    const finalActivation = opt0020ActivationBitsAt(3, 2_249, 6_143);
    const finalWeight = opt0020PackedWeightBitsAt(3, 6_143, 2_047);
    for (const bits of [finalActivation, finalWeight]) {
      expect(bits >>> 10 & 0x1f).toBeGreaterThan(0);
      expect(bits >>> 10 & 0x1f).toBeLessThan(0x1f);
      expect(bits & 0x03ff).not.toBe(0);
    }
  });

  it("uses every arm permutation once and rotates shape order by round mod four", () => {
    const orders = buildOpt0020TimingOrders();
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
    )) as Record<Opt0020Arm, number[]>;
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

  it("uses median six and conjunctively enforces all frozen speed gates", () => {
    const passing = timingInputs([
      [15.5, 13, 8],
      [15.5, 13, 8],
      [46.5, 39, 24],
      [54.25, 45.5, 28],
    ]);
    expect(summarizeOpt0020Timing(passing)).toMatchObject({
      completeDense: {
        milliseconds: { A: 240.25, B: 201.5, C: 124 },
        bToCSpeedup: 201.5 / 124,
        bToCThreshold: 1.25,
        aToCSpeedup: 240.25 / 124,
        aToCThreshold: 1.55,
      },
      feedForward: {
        milliseconds: { A: 147.25, B: 123.5, C: 76 },
        bToCSpeedup: 123.5 / 76,
        bToCThreshold: 1.25,
      },
      everyShapeCFasterThanB: true,
      passed: true,
      decision: "positive-package-layer-gate-authorized",
    });
    const oneShapeRegression = timingInputs([
      [15.5, 13, 13.001],
      [15.5, 13, 4],
      [46.5, 39, 12],
      [54.25, 45.5, 14],
    ]);
    expect(summarizeOpt0020Timing(oneShapeRegression)).toMatchObject({
      everyShapeCFasterThanB: false,
      passed: false,
    });
    const feedForwardMiss = timingInputs([
      [30, 20, 8],
      [30, 20, 8],
      [30, 20, 17],
      [30, 20, 17],
    ]);
    expect(summarizeOpt0020Timing(feedForwardMiss)).toMatchObject({
      feedForward: { bToCSpeedup: 60 / 51 },
      passed: false,
    });
  });

  it("computes compact exhaustive sign-aware ULP bins and frozen numerical gates", () => {
    const control = new Float32Array([1, -1, 0, -0, 0.5]);
    const exact = new Uint32Array(control.buffer.slice(0));
    const exactSummary = compareOpt0020Numerics(
      new Uint32Array(control.buffer),
      exact,
      5,
    );
    expect(exactSummary).toMatchObject({
      count: 5,
      differingCount: 0,
      nrmse: 0,
      snrDecibels: "positive-infinity",
      pearsonCorrelation: 1,
      maximumAbsoluteError: 0,
      relativeMaximumError: 0,
      signedZeroDifferenceCount: 0,
      passed: true,
    });
    expect(sumUlpBins(exactSummary)).toBe(5);
    expect((exactSummary["signAwareF32UlpDistribution"] as readonly Readonly<{
      minimum: number;
      maximum: number;
    }>[]).map(({ minimum, maximum }) => [minimum, maximum])).toEqual([
      [0, 0], [1, 1], [2, 2], [3, 3], [4, 7], [8, 15],
      [16, 31], [32, 63], [64, 127], [128, 255], [256, 511],
      [512, 1_023], [1_024, 4_095], [4_096, 65_535],
      [65_536, 0xffff_ffff],
    ]);

    const signedZeroCandidate = new Uint32Array(exact);
    signedZeroCandidate[2] = 0x8000_0000;
    const signedZeroSummary = compareOpt0020Numerics(
      new Uint32Array(control.buffer),
      signedZeroCandidate,
      5,
    );
    expect(signedZeroSummary).toMatchObject({
      count: 5,
      differingCount: 1,
      signedZeroDifferenceCount: 1,
      maximumAbsoluteError: 0,
      passed: true,
    });
    expect(sumUlpBins(signedZeroSummary)).toBe(5);

    const bad = new Float32Array(control);
    bad[0] = 1.01;
    const badSummary = compareOpt0020Numerics(
      new Uint32Array(control.buffer),
      new Uint32Array(bad.buffer),
      5,
    );
    expect(badSummary).toMatchObject({ passed: false });
    expect(sumUlpBins(badSummary)).toBe(5);
  });

  it("accepts exactly one fresh 30-second nominal thermal gate", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0020ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0020ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
    parameters.set("thermalNonNominalObservations", "0");
    parameters.set("thermalMaximumPollGapMilliseconds", "-1");
    expect(() => parseOpt0020ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
  });

  it("freezes full-output/lifecycle evidence, bounded receipt, one encode, and no retry", () => {
    expect(hashBytes(readFileSync(CORE_PATH)))
      .toBe("466cf7b4c8f860ff55a89b03a5bb2ead99c0d849420918e11d6888e11482d28e");
    for (const pin of [
      "fce77739841572942eca4e96cc6a9f48eb02a971",
      "executionOrder: Object.freeze([\"A\", \"B\", \"C\", \"C\"]",
      "aVersusBRawU32Exact: true",
      "cFirstVersusRerunRawU32Exact: true",
      "candidateVersusControlStatCount: FULL_CANDIDATE_CONTROL_COUNT",
      "aggregateOutputHashManifestSha256",
      "persistentWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT",
      "OUTPUT_PREFILL_QNAN_U32",
      "prefixCanaryIntact",
      "suffixCanaryIntact",
      "tailRowWritten",
      "signAwareF32UlpDistribution",
      "continuousExternalThermalTraceRequiredThroughCleanup: true",
      "unchangedThermalRetryPerformed: false",
      "packageNativeLayerGateAuthorized",
      "productionIntegrationAuthorized: false",
      "c98TrajectoryRunAuthorized: false",
      "m2250ProductRunAuthorized: false",
      "__ACE_OPT0020_RESULT__",
    ]) expect(HARNESS_SOURCE).toContain(pin);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HARNESS_SOURCE).not.toContain("PENDING_");
    const timingFunction = HARNESS_SOURCE.slice(
      HARNESS_SOURCE.indexOf("async function executeTimedDispatch("),
      HARNESS_SOURCE.indexOf("async function createActivationBuffer("),
    );
    expect(timingFunction.match(/dispatch\.encode\(pass\)/g)).toHaveLength(1);
    expect(timingFunction.match(/device\.queue\.submit/g)).toHaveLength(1);
    expect(timingFunction.match(/onSubmittedWorkDone/g)).toHaveLength(1);
    expect(HARNESS_SOURCE).not.toContain("outputWords:");
    expect(HTML_SOURCE).toContain("continuous 30-second nominal");
    expect(HTML_SOURCE).toContain("Six rounds use every A/B/C permutation");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0020-dit-dense-cooperative-dot4.ts"',
    );
  });
});

function timingInputs(
  medians: readonly (readonly [number, number, number])[],
): Opt0020TimingInput[] {
  return buildOpt0020ShapeSpecs().map((spec, index) => {
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
  return [value - 2, value - 1, value, value, value + 1, value + 2];
}

function hashBytes(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function sumUlpBins(summary: Readonly<Record<string, unknown>>): number {
  const bins = summary["signAwareF32UlpDistribution"] as readonly Readonly<{
    count: number;
  }>[];
  return bins.reduce((sum, bin) => sum + bin.count, 0);
}

function halfToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 0x1f;
  const mantissa = bits & 0x03ff;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
}
