import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RECORD = source(
  "../optimization/experiments/OPT-0081-dit-f16-dense-input-multicast.md",
);
const LEDGER = source("../optimization/LEDGER.md");
const PRIMITIVE_RECEIPT_SOURCE = source(
  "../optimization/results/OPT-0081/primitive.json",
);
const PRIMITIVE_RECEIPT = JSON.parse(PRIMITIVE_RECEIPT_SOURCE);
const ACTUAL_MLP_RECEIPT_SOURCE = source(
  "../optimization/results/OPT-0081/actual-mlp.json",
);
const ACTUAL_MLP_RECEIPT = JSON.parse(ACTUAL_MLP_RECEIPT_SOURCE);
const OPT0009_SOURCE = source("../src/webgpu/kernels/dit-dense-fp16.ts");
const OPT0078_SOURCE = source(
  "../src/webgpu/kernels/dit-dense-fp16-weight-multicast.ts",
);

const ROLES = Object.freeze([
  Object.freeze({ name: "selfModulated", elements: 4_608_000, consumers: 3 }),
  Object.freeze({
    name: "selfMergedAttention",
    elements: 4_608_000,
    consumers: 1,
  }),
  Object.freeze({ name: "crossNormalized", elements: 4_608_000, consumers: 1 }),
  Object.freeze({
    name: "crossMergedAttention",
    elements: 4_608_000,
    consumers: 1,
  }),
  Object.freeze({ name: "mlpModulated", elements: 4_608_000, consumers: 2 }),
  Object.freeze({
    name: "gatedActivation",
    elements: 13_824_000,
    consumers: 1,
  }),
]);

describe("OPT-0081 registration contract", () => {
  it("allocates exactly OPT-0081 from the pushed baseline", () => {
    expect(RECORD).toContain("# OPT-0081");
    expect(RECORD).toContain("Evidence: `inconclusive`");
    expect(RECORD).toContain("Disposition: `benchmark-only`");
    expect(RECORD).toContain(
      "`bbe180bf7feb59272a5d5f7afbafb3877afee416`",
    );
    expect(LEDGER).toContain("Next available ID: `OPT-0089`");
    const rows = LEDGER.split("\n").filter((line) =>
      line.startsWith("| OPT-0081 |")
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]).toContain("| inconclusive | benchmark-only |");
    expect(rows[0]).toContain(
      "[record](experiments/OPT-0081-dit-f16-dense-input-multicast.md)",
    );
  });

  it("binds the sole authoritative primitive receipt and B-only decision", () => {
    expect(sha256(PRIMITIVE_RECEIPT_SOURCE)).toBe(
      "8cb2c7c30dec7d179729d5644608fb1f0b9ad5b0478ba92125476936644c775c",
    );
    expect(PRIMITIVE_RECEIPT).toMatchObject({
      schema: "ace-opt-0081-f16-dense-input-multicast-primitive-v1",
      experiment: "OPT-0081",
      status: "completed",
      passed: true,
      inPagePassed: true,
      decision: {
        selectedArm: "B",
        bStandalonePassed: true,
        cPassed: false,
        diagnosticProfileFollowUpAuthorized: true,
        productionIntegrationAuthorized: false,
        packageChangeAuthorized: false,
        trajectoryOrListeningClaim: false,
        unchangedTimingRetryPerformed: false,
      },
      identity: {
        registrationCommit: "70a5e4a29c5455ec00a4b757dcdf5cdcc70a5e91",
        allocationBaselineCommit:
          "bbe180bf7feb59272a5d5f7afbafb3877afee416",
        implementationCheckpointCommit:
          "312d67024978a64b77d2563dd9386b4328f17d33",
        generatedShaderAggregateSha256:
          "c5c9a02d77ca6191fd78620ce7d2bde7fc20b9ef16ddac921a9d503653e530b1",
      },
    });
    expect(PRIMITIVE_RECEIPT.timing.comparisons.bOverA).toMatchObject({
      gpu: {
        meanBaselineMilliseconds: 212.000768,
        meanCandidateMilliseconds: 173.768704,
        meanSpeedup: 1.2200169715255513,
        meanSavingMilliseconds: 38.23206399999998,
        pairWins: 8,
      },
      wall: {
        meanBaselineMilliseconds: 227.35000026226044,
        meanCandidateMilliseconds: 186.54999974370003,
        meanSpeedup: 1.218708124227367,
        meanSavingMilliseconds: 40.80000051856041,
        pairWins: 8,
      },
      meanWallGpuSavingRatio: 1.0671670909151134,
      projectedMeanGpuSavingMilliseconds: 7340.556287999996,
      projectedMeanWallSavingMilliseconds: 7833.600099563599,
    });
    expect(PRIMITIVE_RECEIPT.timing.comparisons.cOverB).toMatchObject({
      gpu: { meanSavingMilliseconds: 1.9988480000000095, pairWins: 4 },
      wall: { meanSavingMilliseconds: 2.949999749660492, pairWins: 5 },
      meanWallGpuSavingRatio: 1.475849964409739,
    });
    expect(PRIMITIVE_RECEIPT.timing.strata).toHaveLength(4);
    for (const stratum of PRIMITIVE_RECEIPT.timing.strata) {
      expect(stratum.bOverAPassed).toBe(true);
      for (const arm of ["A", "B", "C"]) {
        expect(stratum.arms[arm].samples).toHaveLength(8);
      }
    }
    expect(PRIMITIVE_RECEIPT.correctness).toMatchObject({
      comparedU32CountPerComparison: 25_344_000,
      comparisonCount: 24,
      comparisonsPerOutputWord: 6,
      everyOutputRawU32Exact: true,
      producerAudit: { comparedU16Count: 36_864_000, passed: true },
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
      passed: true,
    });
    expect(PRIMITIVE_RECEIPT.thermal.completion).toMatchObject({
      sha256: "0c77b6874e1cfd25ef0cffc4d510fbaff724df17101a2c0e2d67a560f0acbd45",
      observationCount: 101,
      maximumPollGapMilliseconds: 1004,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      coversCleanup: true,
    });
    expect(PRIMITIVE_RECEIPT.cleanup.firstCall).toMatchObject({
      createdBufferCount: 66,
      destroyedBufferCount: 66,
      liveBufferCount: 0,
      liveBytes: 0,
      mapCount: 164,
      unmapCount: 164,
      idempotent: true,
    });
    expect(RECORD).toContain("Separately, an untimed launch-gate preflight");
    expect(RECORD).toContain("It is not a\nperformance run");
    expect(LEDGER).toContain(
      "[primitive result](results/OPT-0081/primitive.json)",
    );
  });

  it("binds the authoritative actual-MLP receipt and representative-layer-only authority", () => {
    expect(sha256(ACTUAL_MLP_RECEIPT_SOURCE)).toBe(
      "92c27035c18ecddd32ebc6a15e8e732f14f36437ee471cbbc0f698d3ab107bfd",
    );
    expect(ACTUAL_MLP_RECEIPT).toMatchObject({
      schema: "ace-opt-0081-f16-actual-mlp-chain-v1",
      experiment: "OPT-0081",
      status: "completed",
      passed: true,
      inPagePassed: true,
      identity: {
        registrationCommit: "606d1e29f56867bfda637c117b58778c634c4ee9",
        bOnlyCorrectionCommit: "0f13bcc486569819df7587349b8b1e049b924ccd",
        harnessImplementationCheckpointCommit:
          "436355ff16fb971d11a959e99e1550abc6186480",
        generatedShaderAggregateSha256:
          "080fff1d8c115c8d748d93e4f62d4285456fd1e9e7fc7a327799398a5e6e97c3",
      },
      decision: {
        selectedArm: "B",
        bStandalonePassed: true,
        cDiagnosticPassed: false,
        cPrimitiveQualified: false,
        cSelectableUnderOpt0081: false,
        representativeLayerFollowUpAuthorized: true,
        productionIntegrationAuthorized: false,
        packageChangeAuthorized: false,
        trajectoryOrListeningClaim: false,
        unchangedTimingRetryPerformed: false,
      },
      protocol: {
        actualM2250ProducersInsideEveryTimedPanel: true,
        fixedPanelOrder: ["gate", "up", "down", "chain"],
        samplesPerArmPerPanel: 8,
        onePassOneCommandBufferOneSubmitOneDrainPerPanelSample: true,
        oneTimestampQueryPairPerPanelSample: true,
        unchangedTimingRetryPerformed: false,
      },
    });

    const expectedPanels = {
      gate: {
        gpu: [47.423488, 36.962304, 36.306944, 32.636928, 8],
        wall: [
          48.912499994039536,
          38.58749997615814,
          37.550000071525574,
          34.85000002384186,
          8,
        ],
      },
      up: {
        gpu: [40.108032, 37.855232, 36.929536, 33.619968, 7],
        wall: [
          41.625000059604645,
          39.099999994039536,
          38.40000009536743,
          35.549999952316284,
          7,
        ],
      },
      down: {
        gpu: [42.582016, 41.107456, 41.12384, 36.765696000000005, 7],
        wall: [
          43.4375,
          42.06249997019768,
          41.89999997615814,
          37.65000009536743,
          7,
        ],
      },
      chain: {
        gpu: [
          123.05203199999998,
          108.068864,
          112.88576,
          101.35142400000001,
          7,
        ],
        wall: [
          124.15000003576279,
          108.7374999821186,
          113.95000004768372,
          102.04999995231628,
          7,
        ],
      },
    };
    expect(ACTUAL_MLP_RECEIPT.timing.panels).toHaveLength(4);
    for (const [panelId, expected] of Object.entries(expectedPanels)) {
      const panel = ACTUAL_MLP_RECEIPT.timing.panels.find(
        (candidate: { id: string }) => candidate.id === panelId,
      );
      expect(panel?.bOverAPassed).toBe(true);
      for (const arm of ["A", "B", "C"]) {
        expect(panel?.arms[arm].samples).toHaveLength(8);
      }
      const comparison = ACTUAL_MLP_RECEIPT.timing.comparisons[panelId].bOverA;
      expect([
        comparison.gpu.meanBaselineMilliseconds,
        comparison.gpu.meanCandidateMilliseconds,
        comparison.gpu.medianBaselineMilliseconds,
        comparison.gpu.medianCandidateMilliseconds,
        comparison.gpu.pairWins,
      ]).toEqual(expected.gpu);
      expect([
        comparison.wall.meanBaselineMilliseconds,
        comparison.wall.meanCandidateMilliseconds,
        comparison.wall.medianBaselineMilliseconds,
        comparison.wall.medianCandidateMilliseconds,
        comparison.wall.pairWins,
      ]).toEqual(expected.wall);
    }

    expect(ACTUAL_MLP_RECEIPT.timing.comparisons.chain.bOverA).toMatchObject({
      gpu: { meanSavingMilliseconds: 14.983167999999978, pairWins: 7 },
      wall: { meanSavingMilliseconds: 15.41250005364418, pairWins: 7 },
      meanWallGpuSavingRatio: 1.028654290844513,
      projectedMeanGpuSavingMilliseconds: 2876.7682559999957,
      projectedMeanWallSavingMilliseconds: 2959.2000102996826,
    });
    expect(ACTUAL_MLP_RECEIPT.timing.comparisons.chain.cOverB).toMatchObject({
      gpu: { meanSavingMilliseconds: 4.857855999999998, pairWins: 6 },
      wall: { meanSavingMilliseconds: 4.72500005364418, pairWins: 6 },
    });
    expect(ACTUAL_MLP_RECEIPT.correctness).toMatchObject({
      rawCheckpointCount: 5,
      denseU32WordsPerComparison: 32_256_000,
      denseU32ComparisonCount: 18,
      candidateBoundaryU16WordsRead: 73_728_000,
      everyBoundaryRawU16Exact: true,
      everyDenseRawU32Exact: true,
      everyArmRerunExact: true,
      directBCRawExact: true,
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
      passed: true,
    });
    expect(ACTUAL_MLP_RECEIPT.thermal.completion).toMatchObject({
      sha256: "27ab8f4c8dd2e338fca7f89672acb3bfec70aab6e852900604684083070f19ea",
      byteLength: 10_476,
      observationCount: 98,
      maximumPollGapMilliseconds: 1013,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      coversCleanup: true,
    });
    expect(ACTUAL_MLP_RECEIPT.cleanup.firstCall).toMatchObject({
      createdBufferCount: 16,
      destroyedBufferCount: 16,
      liveBufferCount: 0,
      liveBytes: 0,
      maximumLiveBytes: 388_861_984,
      mapCount: 132,
      unmapCount: 132,
      armBPostDestroyRejected: true,
      armCPostDestroyRejected: true,
      producerPostDestroyRejected: true,
      idempotent: true,
    });
    for (const frozen of [
      "arm-B representative-layer diagnostic",
      "Production integration: not warranted or authorized",
      "608bdbca56a428fa243842368631754a62ee67dc",
      "Authorized next step: none; OPT-0081 is closed on this browser/GPU",
    ]) expect(`${RECORD}\n${LEDGER}`).toContain(frozen);
    expect(LEDGER).toContain(
      "[actual-MLP result](results/OPT-0081/actual-mlp.json)",
    );
  });

  it("freezes only the six producer roles and nine repeated dense consumers", () => {
    expect(ROLES).toHaveLength(6);
    expect(ROLES.reduce((sum, role) => sum + role.consumers, 0)).toBe(9);
    const elements = ROLES.reduce((sum, role) => sum + role.elements, 0);
    expect(elements).toBe(36_864_000);
    expect(elements * 4).toBe(147_456_000);
    expect(elements * 2).toBe(73_728_000);
    for (const role of ROLES) expect(RECORD).toContain(`\`${role.name}\``);
    for (const frozen of [
      "`K2048/N2048` | 4",
      "`K2048/N1024` | 2",
      "`K2048/N6144` | 2",
      "`K6144/N2048` | 1",
      "`674,815,488 → 601,087,488`",
      "`73,728,000`-byte reduction",
    ]) expect(RECORD).toContain(frozen);
  });

  it("pins the exact request, store, and MLP materiality accounting", () => {
    const currentActivationRequests = 2_084_569_088;
    const typedActivationRequests = currentActivationRequests / 2;
    const currentWeights = 33_353_105_408;
    const multicastWeights = 8_338_276_352;
    const storeSaving = 73_728_000;
    const perLayerEvaluationSaving =
      currentActivationRequests - typedActivationRequests + storeSaving;
    expect({
      currentTotal: currentActivationRequests + currentWeights,
      typedCurrentTotal: typedActivationRequests + currentWeights,
      typedMulticastTotal: typedActivationRequests + multicastWeights,
      unchangedF32MulticastTotal: currentActivationRequests + multicastWeights,
      perLayerEvaluationSaving,
      graphSaving: perLayerEvaluationSaving * 24 * 8,
    }).toEqual({
      currentTotal: 35_437_674_496,
      typedCurrentTotal: 34_395_389_952,
      typedMulticastTotal: 9_380_560_896,
      unchangedF32MulticastTotal: 10_422_845_440,
      perLayerEvaluationSaving: 1_116_012_544,
      graphSaving: 214_274_408_448,
    });
    for (const value of [
      "2,084,569,088 B",
      "1,042,284,544 B",
      "33,353,105,408 B",
      "8,338,276,352 B",
      "35,437,674,496 B",
      "34,395,389,952 B",
      "9,380,560,896 B",
      "10,422,845,440 B",
      "1,116,012,544",
      "214,274,408,448",
      "706,904,064",
      "`63.34%`",
    ]) expect(RECORD).toContain(value);
  });

  it("freezes scalar typed-F16 A/B/C owners without mutating OPT-0009 or OPT-0078", () => {
    expect(RECORD).toContain("**A — current control.**");
    expect(RECORD).toContain("**B — typed-input causal arm.**");
    expect(RECORD).toContain("**C — typed-input plus multicast candidate.**");
    expect(RECORD).toContain("scalar typed\n`array<f16>` binding");
    expect(RECORD).toContain("There is no conversion\ndispatch");
    expect(RECORD).toContain("no unchanged F32 replay of OPT-0078");
    for (const path of [
      "src/webgpu/kernels/dit-f16-dense-input-producers.ts",
      "src/webgpu/kernels/dit-dense-f16-input.ts",
      "src/webgpu/kernels/dit-dense-f16-input-weight-multicast.ts",
      "test/browser/opt-0081-dit-f16-dense-input-multicast.ts",
    ]) expect(RECORD).toContain(`\`${path}\``);
    expect(sha256(OPT0009_SOURCE)).toBe(
      "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3",
    );
    expect(sha256(OPT0078_SOURCE)).toBe(
      "1a8907e9c24d12ddd61e58d55e467051ea5def92db975be47d808f2a31318f1d",
    );
    expect(RECORD).toContain(
      "> Do not repeat this unchanged benchmark or relax its threshold.",
    );
    expect(RECORD).toContain(
      "Arm C changes every dense activation binding from FP32\nstorage",
    );
  });

  it("pins the balanced actual-GPU gate and ordered exact escalation", () => {
    for (const frozen of [
      "`ABC, CBA, BCA, ACB, CAB, BAC, ABC, CBA`",
      "relative order of every pair `4/4`",
      "at least seven of eight",
      "at least `1.12x`",
      "at least `20.8334 ms`",
      "at least `4,000 ms`",
      "at least `10.4167 ms`",
      "at least `2,000 ms`",
      "within `0.75x..1.25x`",
      "is at least\n`1.05x`",
      "at least `15.625 ms`",
      "project at least `3,000 ms`",
      "Producer execution\n  is inside every panel's timed interval",
      "There is one MLP timing run and no unchanged retry",
      "Production selection remains B-only",
    ]) expect(RECORD).toContain(frozen);
    const stages = [
      "**Actual MLP chain.**",
      "**Representative layers.**",
      "**One complete evaluation.**",
      "**Full trajectory.**",
      "**Product and selection.**",
    ].map((stage) => RECORD.indexOf(stage));
    expect(stages.every((index) => index >= 0)).toBe(true);
    expect(stages).toEqual([...stages].sort((left, right) => left - right));
    expect(RECORD).toContain(
      "No new subjective listening comparison is required only if raw-U16 boundary",
    );
    expect(RECORD).toContain(
      "The first raw mismatch stops this\nexperiment",
    );
    expect(RECORD).toContain("independently cooled A/B and B/A");
    expect(RECORD).not.toContain("independently cooled A/C and C/A");
    expect(RECORD).toContain("do not waive a threshold after observation");
  });

  it("freezes the exact arm-B two-layer graph-prefix diagnostic before performance source", () => {
    for (const frozen of [
      "### Frozen arm-B representative-layer diagnostic",
      "`opt-0081-six-dense-input-f16-storage-v1`",
      "M2250/C98 at evaluation 0",
      "`ace-dit-eval-0-layer-0` with sliding self attention",
      "`ace-dit-eval-0-layer-1` with full self attention",
      "canonical conditioner may execute only\nduring excluded setup",
      "no conditioner work is part of the selected command slice",
      "physical graph\ncommands 25 through 52 inclusive",
      "11\nphysical commands for the sliding layer, and 15 for the full/quad layer",
      "exactly 28 command buffers and 28 completion fences",
      "seven completion\nepochs/true drains, six real idle turns",
      "command 37, the final layer-0 command",
      "commands 38 through 40",
      "existing exported OPT-0080 depth-two\nscheduler directly",
      "ordinary\nproduct/backend selector must omit or reject",
      "Descriptor and tap-capture commands are correctness-only",
      "outside the\n28 timed commands",
      "`73,728,000` U16 words",
      "`110,592,000` U32 words",
      "`9,216,000` U32 words",
      "no\nmeasurement-only submit, drain, map, readback, descriptor/tap capture",
      "`AB, BA, BA, AB, AB, BA, BA, AB`",
      "at least three of\n  four forward and three of four reverse pairs",
      "at least `31.25 ms`",
      "`31.25 * (24 / 2) * 8 = 3,000 ms`",
      "complete cleanup within `1,000 ms`",
      "does\nnot select production",
    ]) expect(RECORD).toContain(frozen);

    expect(31.25 * (24 / 2) * 8).toBe(3_000);
    expect(RECORD).toContain(
      "Run only the frozen arm-B representative-layer\n   diagnostic above",
    );
    expect(RECORD).toContain(
      "only the independently cooled complete-evaluation A/B and B/A gate",
    );
  });
});

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
