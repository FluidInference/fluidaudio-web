import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  OPT_0066_NUMERICAL_ENVELOPE,
  OPT_0066_REQUIRED_CONV_TRANSPOSE_SPEEDUP,
  OPT_0066_REQUIRED_K7_SPEEDUP,
  compareOpt0066Raw,
  compareOpt0066Waveforms,
  evaluateOpt0066BalancedTiming,
  parseOpt0066ThermalGate,
  type Opt0066TimingSample,
} from "./browser/opt-0066-vae-dual-k4-quality-c512-contract.js";
import {
  composeOpt0066CompleteTransposeOracleOwner,
  type Opt0066NativeTransposeOwner,
} from "./browser/opt-0066-vae-dual-k4-oracle.js";
import {
  compareOpt0066U16,
  readOpt0066ExactFileSliceU16,
} from "./browser/opt-0066-vae-dual-k4-package-proof.js";
import { ACE_OPT_0057_VAE_K7_ROUTES } from
  "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";

const WORKER_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-quality-c512-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-quality-c512.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-quality-c512.html",
);
const CONTRACT_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-quality-c512-contract.ts",
);
const PACKAGE_PROOF_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-package-proof.ts",
);
const ORACLE_SOURCE = source(
  "./browser/opt-0066-vae-dual-k4-oracle.ts",
);

describe("OPT-0066 authenticated dual-K4 C512 browser gate", () => {
  it("authenticates both packages and finishes bounded layout proof before WebGPU", () => {
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-experimental/manifest.json"',
    );
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-revision7-experimental/manifest.json"',
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
      "ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id",
    );
    expect(WORKER_SOURCE.indexOf("prepareOpt0066PackageLayoutProof("))
      .toBeLessThan(WORKER_SOURCE.indexOf("requestAceWebGpuDevice({"));
    expect(PACKAGE_PROOF_SOURCE).toContain(
      "packAceOpt0051VaeK7WeightU16",
    );
    expect(PACKAGE_PROOF_SOURCE).toContain(
      "packAceOpt0048VaeConvTranspose1dK4WeightU16",
    );
    expect(PACKAGE_PROOF_SOURCE).toContain(
      "ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT",
    );
    expect(PACKAGE_PROOF_SOURCE).toContain("15_335_424");
    expect(WORKER_SOURCE).toContain(
      "boundedPreDeviceNativeToPackedProof: packageLayoutProof.proof",
    );
  });

  it("reads exact tensor spans with canaries and reports raw mismatches", async () => {
    const canaryBytes = new Uint8Array([
      0xaa, 0xbb,
      0x34, 0x12, 0x78, 0x56,
      0xcc, 0xdd,
    ]);
    const words = await readOpt0066ExactFileSliceU16(
      new Blob([canaryBytes]),
      2,
      4,
      "canary",
    );
    expect([...words]).toEqual([0x1234, 0x5678]);
    await expect(readOpt0066ExactFileSliceU16(
      new Blob([canaryBytes]),
      1,
      3,
    )).rejects.toThrow(/exact even File span/);
    expect(compareOpt0066U16(
      new Uint16Array([1, 2, 3]),
      new Uint16Array([1, 9, 8]),
    )).toEqual({
      comparedWordCount: 3,
      mismatchCount: 2,
      firstMismatch: { index: 1, expected: 2, actual: 9 },
    });
  });

  it("composes native block 0 with real K4 blocks 1-4 and destroys once", async () => {
    const nativeCreate = vi.fn(async (..._args: unknown[]) => Object.freeze({
      kernelId: "native",
    }));
    const nativeDestroy = vi.fn();
    const k4Create = vi.fn(async (..._args: unknown[]) =>
      Object.freeze({ kernelId: "k4" })
    );
    const k4Destroy = vi.fn();
    const derivedDestroy = vi.fn();
    const native = {
      createDispatch: nativeCreate,
      destroy: nativeDestroy,
    } as unknown as Opt0066NativeTransposeOwner;
    const k4 = {
      createDispatch: k4Create,
      destroy: k4Destroy,
    } as unknown as Parameters<
      typeof composeOpt0066CompleteTransposeOracleOwner
    >[1];
    const fakeBinding = (id: number): GPUBufferBinding => ({
      buffer: Object.freeze({ id }) as unknown as GPUBuffer,
      offset: 0,
      size: 2,
    });
    const derived = [1, 2, 3, 4].map((block) => Object.freeze({
      operationLabel: `block-${block}-conv-t1`,
      binding: fakeBinding(100 + block),
    }));
    const owner = composeOpt0066CompleteTransposeOracleOwner(
      native,
      k4,
      derived,
      derivedDestroy,
    );
    const bindings = {
      input: fakeBinding(1),
      polyphaseWeight: fakeBinding(2),
      bias: fakeBinding(3),
      output: fakeBinding(4),
    } as Parameters<Opt0066NativeTransposeOwner["createDispatch"]>[3];
    await owner.createDispatch(
      "block0",
      "block-0-conv-t1",
      {} as never,
      bindings,
      {} as never,
    );
    await owner.createDispatch(
      "block3",
      "block-3-conv-t1",
      {} as never,
      bindings,
      {} as never,
    );
    expect(nativeCreate).toHaveBeenCalledOnce();
    expect(k4Create).toHaveBeenCalledOnce();
    expect(k4Create.mock.calls[0]![3]).toMatchObject({
      weight: derived[2]!.binding,
    });
    owner.destroy();
    owner.destroy();
    expect(nativeDestroy).toHaveBeenCalledOnce();
    expect(k4Destroy).toHaveBeenCalledOnce();
    expect(derivedDestroy).toHaveBeenCalledOnce();
    await expect(owner.createDispatch(
      "destroyed",
      "block-1-conv-t1",
      {} as never,
      bindings,
      {} as never,
    )).rejects.toThrow(/was destroyed/);
    expect(ORACLE_SOURCE).toContain(
      "AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel.create",
    );
    expect(ORACLE_SOURCE).toContain("mappedAtCreation: true");
  });

  it("reconciles every C512 K7 and ConvTranspose quantum by arithmetic owner", () => {
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
    expect(WORKER_SOURCE).toContain(
      'const expectedTranspose = arm !== "rev6-scalar"',
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID",
    );
    expect(WORKER_SOURCE).toContain("sumQuanta(k1) !== 819");
    expect(WORKER_SOURCE).toContain("sumQuanta(snake) !== 1_611");
    expect(WORKER_SOURCE).toContain("sumQuanta(add) !== 690");
  });

  it("requires complete same-arithmetic raw identity and a separate scalar envelope", () => {
    expect(WORKER_SOURCE).toContain(
      "createCompleteSameArithmeticOracleBackend",
    );
    expect(WORKER_SOURCE).toContain(
      "replaceSelectedK7WithNativeLayoutOpt0024",
    );
    expect(WORKER_SOURCE).toContain(
      "installOpt0066CompleteTransposeOracle(",
    );
    expect(WORKER_SOURCE).toContain(
      "rawCandidateToCompleteSameArithmeticOracle",
    );
    expect(WORKER_SOURCE).toContain(
      'numericalEnvelopeAuthority: "OPT-0044-unchanged-from-OPT-0024"',
    );
    expect(WORKER_SOURCE).toContain(
      '"893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47"',
    );
    expect(WORKER_SOURCE).not.toContain("rawU16CandidateToNativeLayoutOpt0024");
    expect(OPT_0066_NUMERICAL_ENVELOPE).toEqual({
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
    expect(compareOpt0066Waveforms(reference, reference).passed).toBe(true);
    expect(compareOpt0066Waveforms(
      reference,
      new Float32Array([1, 1, 1, 1]),
    ).passed).toBe(false);
  });

  it("runs two deterministic complete owners per correctness arm", () => {
    const a = new Float32Array([1, -0, 3.5]);
    const b = new Float32Array(a);
    expect(compareOpt0066Raw(a, b)).toMatchObject({
      comparedU32WordCount: 3,
      comparedU16WordCount: 6,
      u32MismatchCount: 0,
      u16MismatchCount: 0,
      rawU32Exact: true,
      rawU16Exact: true,
    });
    b[1] = 0;
    expect(compareOpt0066Raw(a, b)).toMatchObject({
      u32MismatchCount: 1,
      u16MismatchCount: 1,
      rawU32Exact: false,
      rawU16Exact: false,
    });
    expect(WORKER_SOURCE).toContain(
      'export const OPT_0066_CORRECTNESS_ORDER = Object.freeze([\n' +
        '  "rev6-scalar",\n  "rev6-scalar",\n' +
        '  "rev6-same-arithmetic-oracle",\n' +
        '  "rev6-same-arithmetic-oracle",\n' +
        '  "rev7-candidate",\n  "rev7-candidate",',
    );
    expect(WORKER_SOURCE).toContain("scalarDeterminism");
    expect(WORKER_SOURCE).toContain("sameArithmeticOracleDeterminism");
    expect(WORKER_SOURCE).toContain("candidateDeterminism");
    expect(WORKER_SOURCE).toContain("firstFrameFinite");
    expect(WORKER_SOURCE).toContain("lastFrameFinite");
    expect(WORKER_SOURCE).toContain(
      "tracker.active !== null || tracker.liveOwners !== 0",
    );
    expect(WORKER_SOURCE).toContain("maximumAllowedLiveOwners: 1");
    expect(WORKER_SOURCE).toContain("idempotentDestroyPromises");
  });

  it("gates both paired directions and both family medians plus complete walls", () => {
    expect(OPT_0066_REQUIRED_K7_SPEEDUP).toBe(1.5);
    expect(OPT_0066_REQUIRED_CONV_TRANSPOSE_SPEEDUP).toBe(1.3);
    const passing = timingSamples({
      scalarK7: [180, 195],
      candidateK7: [100, 110],
      scalarTranspose: [65, 70],
      candidateTranspose: [45, 48],
      scalarDecoder: [300, 315],
      candidateDecoder: [270, 285],
      scalarOuter: [340, 355],
      candidateOuter: [310, 330],
    });
    expect(evaluateOpt0066BalancedTiming(passing)).toMatchObject({
      forward: {
        k7Improved: true,
        convTransposeImproved: true,
        decoderNoRegression: true,
        outerWindowNoRegression: true,
      },
      reverse: {
        k7Improved: true,
        convTransposeImproved: true,
        decoderNoRegression: true,
        outerWindowNoRegression: true,
      },
      aggregate: {
        requiredMedianK7Speedup: 1.5,
        medianK7Passed: true,
        requiredMedianConvTransposeSpeedup: 1.3,
        medianConvTransposePassed: true,
      },
      outerWindowWallGatingNoRegression: true,
      passed: true,
    });
    const outerRegression = [...passing];
    outerRegression[2] = Object.freeze({
      ...outerRegression[2]!,
      outerWindowWallMs: 400,
    });
    expect(evaluateOpt0066BalancedTiming(outerRegression).passed).toBe(false);
    const transposeBelowMedian = timingSamples({
      scalarK7: [180, 195],
      candidateK7: [100, 110],
      scalarTranspose: [65, 70],
      candidateTranspose: [55, 56],
      scalarDecoder: [300, 315],
      candidateDecoder: [270, 285],
      scalarOuter: [340, 355],
      candidateOuter: [310, 330],
    });
    expect(evaluateOpt0066BalancedTiming(transposeBelowMedian).passed).toBe(
      false,
    );
    expect(WORKER_SOURCE).toContain(
      "requiresBothDirectionsToImproveHomogeneousConvTransposeWall: true",
    );
    expect(WORKER_SOURCE).toContain(
      "requiresNoOuterWindowRegressionInBothPairedOrders: true",
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
    expect(parseOpt0066ThermalGate(valid, 9_999, 40_001)).toEqual({
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
      expect(() => parseOpt0066ThermalGate(invalid, 9_999, 40_001))
        .toThrow(/one truthful level-0/);
    }
    expect(CONTRACT_SOURCE).toContain("observationCount !== 1");
    expect(CONTRACT_SOURCE).not.toContain("pollMilliseconds");
    expect(HTML_SOURCE).toContain(
      "leave the machine idle for at least 30 seconds",
    );
  });

  it("is button-gated, downloadable, fail-closed, and changes no default", () => {
    expect(PAGE_SOURCE).toContain('prepareButton.addEventListener("click"');
    expect(PAGE_SOURCE.indexOf("new Worker(")).toBeGreaterThan(
      PAGE_SOURCE.indexOf('prepareButton.addEventListener("click"'),
    );
    expect(PAGE_SOURCE).toContain("parseOpt0018RunIdentity(");
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0066_RESULT__ = receipt");
    expect(PAGE_SOURCE).toContain("URL.createObjectURL(new Blob(");
    expect(PAGE_SOURCE).toContain("thermalStarted.value = String(Date.now())");
    expect(HTML_SOURCE).toContain('id="thermal-gate" disabled');
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(HTML_SOURCE).toContain(
      'src="./opt-0066-vae-dual-k4-quality-c512.ts"',
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
  scalarTranspose: readonly [number, number];
  candidateTranspose: readonly [number, number];
  scalarDecoder: readonly [number, number];
  candidateDecoder: readonly [number, number];
  scalarOuter: readonly [number, number];
  candidateOuter: readonly [number, number];
}>): readonly Opt0066TimingSample[] {
  return [
    sample(
      "rev6-scalar",
      values.scalarK7[0],
      values.scalarTranspose[0],
      values.scalarDecoder[0],
      values.scalarOuter[0],
    ),
    sample(
      "rev7-candidate",
      values.candidateK7[0],
      values.candidateTranspose[0],
      values.candidateDecoder[0],
      values.candidateOuter[0],
    ),
    sample(
      "rev7-candidate",
      values.candidateK7[1],
      values.candidateTranspose[1],
      values.candidateDecoder[1],
      values.candidateOuter[1],
    ),
    sample(
      "rev6-scalar",
      values.scalarK7[1],
      values.scalarTranspose[1],
      values.scalarDecoder[1],
      values.scalarOuter[1],
    ),
  ];
}

function sample(
  arm: Opt0066TimingSample["arm"],
  k7: number,
  transpose: number,
  decoder: number,
  outer: number,
): Opt0066TimingSample {
  return {
    arm,
    k7FamilySubmitThroughDrainMs: k7,
    convTransposeFamilySubmitThroughDrainMs: transpose,
    decoderSubmitThroughDrainMs: decoder,
    outerWindowWallMs: outer,
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
