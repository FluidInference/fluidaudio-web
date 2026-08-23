import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0075_WIDTH128_RMSNORM_KERNEL_ID,
  ACE_OPT_0075_WIDTH128_RMSNORM_LANES,
  AceOpt0075Width128RmsNormKernel,
  aceCorrectnessRmsNormWgsl,
  aceOpt0075Width128RmsNormWgsl,
  planAceOpt0075Width128RmsNorm,
} from "../src/webgpu/kernels/rmsnorm.js";

const RMSNORM_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/rmsnorm.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0075-dit-width128-rmsnorm-wg128.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0075-dit-width128-rmsnorm-wg128.html",
  import.meta.url,
), "utf8");

describe("OPT-0075 width-128 RMSNorm WG128", () => {
  it("plans only width-128 production rows", () => {
    expect(ACE_OPT_0075_WIDTH128_RMSNORM_LANES).toBe(128);
    expect(ACE_OPT_0075_WIDTH128_RMSNORM_KERNEL_ID).toMatch(
      /reference-bf16-rmsnorm-width128-wg128-v1$/,
    );
    expect(planAceOpt0075Width128RmsNorm({
      rows: 36_000,
      width: 128,
      epsilon: 1e-6,
    })).toMatchObject({
      rows: 36_000,
      width: 128,
      elements: 4_608_000,
      workgroupsX: 36_000,
      workgroupsY: 1,
    });
    expect(() => planAceOpt0075Width128RmsNorm({
      rows: 2_250,
      width: 2_048,
      epsilon: 1e-6,
    })).toThrow(/only width 128/);
  });

  it("retains the inert +0 before the exact stride-64 tree", () => {
    const current = aceCorrectnessRmsNormWgsl("reference-bf16", {
      rows: 36_000,
      width: 128,
      epsilon: 1e-6,
    });
    const candidate = aceOpt0075Width128RmsNormWgsl({
      rows: 36_000,
      width: 128,
      epsilon: 1e-6,
    });
    expect(current).toContain("@compute @workgroup_size(256, 1, 1)");
    expect(current).toContain("partial_squares: array<f32, 256>");
    expect(current).toContain("var stride = 128u");
    expect(candidate).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(candidate).toContain("partial_squares: array<f32, 128>");
    expect(candidate).toContain("square_sum = square_sum + 0.0;");
    expect(candidate).toContain("var stride = 64u");
    expect(candidate).toContain(`
    workgroupBarrier();
    stride = stride >> 1u;
  }
  workgroupBarrier();
  let inverse_rms = inverseSqrt`);
    expect(candidate).toContain(`
  let pair = weight[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);`);
    expect(candidate).toContain(
      "output[index] = value * inverse_rms * load_weight(column);",
    );
    expect(candidate).not.toContain("enable f16");
    expect(candidate.indexOf("square_sum = square_sum + 0.0;")).toBeLessThan(
      candidate.indexOf("partial_squares[lane] = square_sum;"),
    );
  });

  it("models the same FP32 sum tree on finite edge values", () => {
    const values = Array.from({ length: 128 }, (_, index) => [
      0,
      -0,
      f32FromBits(0x0000_0001),
      f32FromBits(0x007f_ffff),
      f32FromBits(0x0080_0000),
      2 ** -12,
      -(2 ** -7),
      0.5,
      -1,
      3.75,
      2 ** 20,
    ][index % 11]!);
    expect(f32Bits(currentWidth128SquareSum(values))).toBe(
      f32Bits(candidateWidth128SquareSum(values)),
    );
  });

  it("fails closed below WG128 or 512 shared bytes", () => {
    const insufficientLanes = {
      limits: {
        maxComputeInvocationsPerWorkgroup: 64,
        maxComputeWorkgroupSizeX: 64,
        maxComputeWorkgroupStorageSize: 16_384,
      },
    } as unknown as GPUDevice;
    expect(() => AceOpt0075Width128RmsNormKernel.create(insufficientLanes))
      .toThrow(/128 lanes and 512 bytes/);

    const insufficientShared = {
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupStorageSize: 256,
      },
    } as unknown as GPUDevice;
    expect(() => AceOpt0075Width128RmsNormKernel.create(insufficientShared))
      .toThrow(/128 lanes and 512 bytes/);
  });

  it("remains benchmark-only and absent from every shared production caller", () => {
    expect(RMSNORM_SOURCE).toContain("Benchmark-only width-128 owner");
    for (const productionSource of sourceFiles(new URL("../src/", import.meta.url))) {
      if (productionSource.pathname.endsWith("/webgpu/kernels/rmsnorm.ts")) continue;
      const source = readFileSync(productionSource, "utf8");
      expect(source).not.toContain("OPT_0075");
      expect(source).not.toContain("opt-0075");
      expect(source).not.toContain("AceOpt0075Width128RmsNormKernel");
    }
  });

  it("gates timing on complete exact production and bounded edge fixtures", () => {
    expect(HARNESS_SOURCE).toContain('shapeSpec("q-36000", 36_000, 2, 0)');
    expect(HARNESS_SOURCE).toContain('shapeSpec("k-18000", 18_000, 1, 1)');
    expect(HARNESS_SOURCE).toContain(
      'shapeSpec("cross-cache-784", 784, 0, 2)',
    );
    for (const fixture of [
      "signed-zero",
      "normal-subnormal-boundary",
      "alternating-magnitude",
      "maximum-finite-bf16-scale",
    ]) expect(HARNESS_SOURCE).toContain(`caseSpec("${fixture}"`);
    expect(HARNESS_SOURCE).toContain("rawU32IdentityRequired: true");
    expect(HARNESS_SOURCE).toContain(
      "deterministicCandidateRerunsRequired: true",
    );
    expect(HARNESS_SOURCE).toContain("completedBeforeReady: true");
    expect(HARNESS_SOURCE).toContain("currentCandidate.differingU32Count === 0");
    expect(HARNESS_SOURCE).toContain("candidateRerun.differingU32Count === 0");
    expect(HARNESS_SOURCE).toContain("current.nonFiniteCount > 0");
    expect(HARNESS_SOURCE).toContain("prefixCanaryIntact");
    expect(HARNESS_SOURCE).toContain("suffixCanaryIntact");
  });

  it("pins the balanced repeated-dispatch timestamp and anti-micro gate", () => {
    expect(HARNESS_SOURCE).toContain('requiredFeatures: ["timestamp-query"]');
    expect(HARNESS_SOURCE).toContain('type: "timestamp"');
    expect(HARNESS_SOURCE).toContain("timestampWrites: {");
    expect(HARNESS_SOURCE).toContain("DISPATCH_REPETITIONS = 8");
    expect(HARNESS_SOURCE.match(/^  timingRound\(/gm)).toHaveLength(8);
    expect(HARNESS_SOURCE).toContain("2 * q.perDispatchGpuMilliseconds");
    expect(HARNESS_SOURCE).toContain("2 * q.perDispatchWallMilliseconds");
    expect(HARNESS_SOURCE).toContain("REQUIRED_SPEEDUP = 1.25");
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_LAYER_MIX_SAVING_MILLISECONDS = 10.5",
    );
    expect(HARNESS_SOURCE).toContain("everyPrimaryShapePairedGpuWin");
    expect(HARNESS_SOURCE).toContain("everyPrimaryShapePairedWallWin");
    expect(HARNESS_SOURCE).toContain("everyWeightedRoundGpuWin");
    expect(HARNESS_SOURCE).toContain("everyWeightedRoundWallWin");
    expect(HARNESS_SOURCE).toContain(
      "weightedMeanWallGpuSavingAgreementPassed",
    );
    expect(HARNESS_SOURCE).toContain(
      "weightedMedianWallGpuSavingAgreementPassed",
    );
    expect(HARNESS_SOURCE).toContain("harnessSourceSha256");
    expect(HARNESS_SOURCE).toContain("harnessHtmlSha256");
    expect(HARNESS_SOURCE).toContain("externalThermalGateAuditedByPage: false");
    expect(HARNESS_SOURCE).toContain("productionIntegrationAuthorized: false");
    expect(HARNESS_SOURCE).toContain("cleanupCompletedAtEpochMilliseconds");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain("keep polling through the completed cleanup");
  });

  it("persists an authenticated nominal inconclusive receipt", () => {
    const resultBytes = readFileSync(new URL(
      "../optimization/results/OPT-0075/result.json",
      import.meta.url,
    ));
    const receiptBytes = readFileSync(new URL(
      "../optimization/results/OPT-0075/browser-receipt.json",
      import.meta.url,
    ));
    const thermalGateBytes = readFileSync(new URL(
      "../optimization/results/OPT-0075/thermal-gate.json",
      import.meta.url,
    ));
    const thermalTraceBytes = readFileSync(new URL(
      "../optimization/results/OPT-0075/thermal-trace.json",
      import.meta.url,
    ));
    const result = JSON.parse(resultBytes.toString("utf8"));
    const receipt = JSON.parse(receiptBytes.toString("utf8"));
    const gate = JSON.parse(thermalGateBytes.toString("utf8"));
    const trace = JSON.parse(thermalTraceBytes.toString("utf8"));

    expect(result).toMatchObject({
      status: "inconclusive",
      passed: false,
      correctness: {
        passed: true,
        productionU32ComparisonsPerCandidate: 7_012_352,
        adversarialU32ComparisonsPerCandidate: 2_048,
        rawU32DifferenceCount: 0,
        candidateRerunDifferenceCount: 0,
      },
      thermal: { passed: true, nonNominalObservationCount: 0 },
      timing: { roundCount: 8, rawSampleCount: 48 },
    });
    expect(result.identity.browserReceiptBytes).toBe(receiptBytes.byteLength);
    expect(result.identity.browserReceiptSha256).toBe(sha256(receiptBytes));
    expect(result.identity.thermalGateSha256).toBe(sha256(thermalGateBytes));
    expect(result.identity.thermalTraceSha256).toBe(sha256(thermalTraceBytes));
    expect(receipt.identity.kernelSourceSha256).toBe(sha256(RMSNORM_SOURCE));
    expect(receipt.identity.harnessSourceSha256).toBe(sha256(HARNESS_SOURCE));
    expect(receipt.identity.harnessHtmlSha256).toBe(sha256(HARNESS_HTML));
    expect(receipt.correctness.passed).toBe(true);
    expect(receipt.timing.rawSamples).toHaveLength(48);
    expect(receipt.cleanup.zeroLiveBuffers).toBe(true);
    expect(receipt.timing.gates.weightedMeanGpuSavingPassed).toBe(false);
    expect(receipt.timing.gates.everyWeightedRoundGpuWin).toBe(false);
    expect(receipt.timing.projectedEightEvaluationSavingsMilliseconds.meanGpu)
      .toBeCloseTo(
        receipt.timing.savingsMillisecondsPerLayerMix.meanGpu * 24 * 8,
        10,
      );

    expect(gate.observations).toHaveLength(gate.observationCount);
    expect(gate.observations.every((item: { level: number }) => item.level === 0))
      .toBe(true);
    expect(gate.completedAtEpochMilliseconds - gate.startedAtEpochMilliseconds)
      .toBeGreaterThanOrEqual(30_000);
    expect(trace.observations).toHaveLength(trace.observationCount);
    expect(trace.nonNominalObservationCount).toBe(0);
    expect(trace.maximumPollGapMilliseconds).toBeLessThanOrEqual(1_250);
    expect(trace.completedAtEpochMilliseconds).toBeGreaterThan(
      receipt.timing.cleanupCompletedAtEpochMilliseconds,
    );
  });
});

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function sourceFiles(directory: URL): URL[] {
  const files: URL[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const child = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directory);
    if (entry.isDirectory()) files.push(...sourceFiles(child));
    else if (entry.isFile() && entry.name.endsWith(".ts")) files.push(child);
  }
  return files;
}

function currentWidth128SquareSum(values: readonly number[]): number {
  const partial = new Float32Array(256);
  for (let lane = 0; lane < 128; lane += 1) {
    partial[lane] = addF32(0, multiplyF32(values[lane]!, values[lane]!));
  }
  for (let stride = 128; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane += 1) {
      partial[lane] = addF32(partial[lane]!, partial[lane + stride]!);
    }
  }
  return partial[0]!;
}

function candidateWidth128SquareSum(values: readonly number[]): number {
  const partial = new Float32Array(128);
  for (let lane = 0; lane < 128; lane += 1) {
    const square = addF32(0, multiplyF32(values[lane]!, values[lane]!));
    partial[lane] = addF32(square, 0);
  }
  for (let stride = 64; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane += 1) {
      partial[lane] = addF32(partial[lane]!, partial[lane + stride]!);
    }
  }
  return partial[0]!;
}

function addF32(left: number, right: number): number {
  return Math.fround(Math.fround(left) + Math.fround(right));
}

function multiplyF32(left: number, right: number): number {
  return Math.fround(Math.fround(left) * Math.fround(right));
}

function f32Bits(value: number): number {
  const values = new Float32Array([value]);
  return new Uint32Array(values.buffer)[0]!;
}

function f32FromBits(bits: number): number {
  const words = new Uint32Array([bits]);
  return new Float32Array(words.buffer)[0]!;
}
