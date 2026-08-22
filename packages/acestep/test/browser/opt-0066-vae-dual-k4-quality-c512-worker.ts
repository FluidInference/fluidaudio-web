/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />
/// <reference lib="webworker" />

import { createAceOpt0011LatentFixture } from
  "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import type { AceVaeChunkGpuBackendProgress } from
  "../../src/webgpu/vae-backend.js";
import {
  ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  AceOpt0011Fp16VaeChunkGpuBackend,
  type AceOpt0011Fp16VaeDispatchTopologyReceipt,
  type AceOpt0011Fp16VaePreparedBuffers,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
  AceOpt0011Fp16VaeDecoderRuntime,
  type AceOpt0011Fp16VaeWindowBindings,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  resolveAceOpt0011Fp16VaePackageBindings,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
} from "../../src/webgpu/kernels/vae-pointwise-fp16.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  type AceFp16VaeConv1dSubgroupKernel,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import { ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0057_VAE_K7_ROUTES,
  selectAceOpt0057VaeK7,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import { ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID } from
  "../../src/webgpu/kernels/vae-snake-fp16.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
} from "../../src/webgpu/vae-chunks.js";
import { planAceVaeDecoder } from "../../src/webgpu/vae-decoder.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  compareOpt0066Raw,
  compareOpt0066Waveforms,
  evaluateOpt0066BalancedTiming,
  type Opt0066RawComparison,
  type Opt0066ThermalGate,
  type Opt0066TimedArm,
  type Opt0066TimingSample,
  type Opt0066WaveformMetrics,
} from "./opt-0066-vae-dual-k4-quality-c512-contract.js";
import { installOpt0066CompleteTransposeOracle } from
  "./opt-0066-vae-dual-k4-oracle.js";
import {
  prepareOpt0066PackageLayoutProof,
  type Opt0066DerivedTransposeWeight,
  type Opt0066PreparedPackageLayoutProof,
} from "./opt-0066-vae-dual-k4-package-proof.js";

export const OPT_0066_BROWSER_SCHEMA =
  "ace-opt-0066-vae-revision7-dual-k4-quality-c512-abba-v1" as const;
export const OPT_0066_C512_FRAMES = 512 as const;
export const OPT_0066_C512_INPUT_ELEMENTS = 32_768 as const;
export const OPT_0066_C512_OUTPUT_ELEMENTS = 1_966_080 as const;
export const OPT_0066_C512_OUTPUT_U16_WORDS = 3_932_160 as const;
export const OPT_0066_C512_FIXTURE_SHA256 =
  "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899" as const;
export const OPT_0066_REV6_SCALAR_OUTPUT_SHA256 =
  "893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47" as const;
export const OPT_0066_CORRECTNESS_ORDER = Object.freeze([
  "rev6-scalar",
  "rev6-scalar",
  "rev6-same-arithmetic-oracle",
  "rev6-same-arithmetic-oracle",
  "rev7-candidate",
  "rev7-candidate",
] as const);
export const OPT_0066_TIMED_ORDER = Object.freeze([
  "rev6-scalar",
  "rev7-candidate",
  "rev7-candidate",
  "rev6-scalar",
] as const);

const EXPERIMENT_ID = "OPT-0066" as const;
const REVISION6_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json";
const REVISION7_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json";
const FLOAT16_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const REQUIRED_SUBGROUP_SIZE = 32;
const DECODER_COMMAND_BUFFER_COUNT = 123;
const TOTAL_COMMAND_BUFFER_COUNT = 124;
const REQUESTED_COOPERATIVE_IDLE_MS = 123;
const K7_TOTAL_QUANTA = 4_090;
const K7_SELECTED_QUANTA = 3_360;
const K7_NATIVE_QUANTA = 730;
const CONV_TRANSPOSE_TOTAL_QUANTA = 644;

export type Opt0066OwnedArm =
  | "rev6-scalar"
  | "rev6-same-arithmetic-oracle"
  | "rev7-candidate";

type WorkerCommand =
  | Readonly<{
      readonly type: "initialize";
      readonly identity: Opt0018RunIdentity;
    }>
  | Readonly<{
      readonly type: "run";
      readonly thermalGate: Opt0066ThermalGate;
    }>
  | Readonly<{ readonly type: "dispose" }>;

interface WorkerEvent {
  readonly type:
    | "progress"
    | "ready-for-thermal-gate"
    | "comparison-complete"
    | "failed";
  readonly message?: string;
  readonly readyAtEpochMilliseconds?: number;
  readonly preparation?: Readonly<Record<string, unknown>>;
  readonly result?: Readonly<Record<string, unknown>>;
  readonly error?: Readonly<Record<string, unknown>>;
}

interface PreparedPackage {
  readonly revision: 6 | 7;
  readonly manifestPath: string;
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly acquired: ReadonlyMap<string, File>;
  readonly residentBytes: number;
}

interface ObserverSummary {
  readonly progressEventCount: number;
  readonly finalProgress: AceVaeChunkGpuBackendProgress;
  readonly familyProfile: AceOpt0011Fp16VaeWindowFamilyProfile;
}

interface ExecutionResult {
  readonly arm: Opt0066OwnedArm;
  readonly output: Float32Array<ArrayBuffer>;
  readonly topology: AceOpt0011Fp16VaeDispatchTopologyReceipt;
  readonly topologySummary: Readonly<Record<string, unknown>>;
  readonly ownerLifecycle: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<{
    readonly arm: Opt0066OwnedArm;
    readonly measured: boolean;
    readonly scan: Readonly<Record<string, number>>;
    readonly timing: Readonly<{
      readonly ownerSetupWallMs: number;
      readonly outerWindowWallMs: number;
      readonly k7FamilySubmitThroughDrainMs: number;
      readonly convTransposeFamilySubmitThroughDrainMs: number;
      readonly decoderSubmitThroughDrainMs: number;
      readonly decoderCommandBufferCount: number;
      readonly totalCommandBufferCount: number;
      readonly queueDrainCount: number;
      readonly requestedCooperativeIdleMs: number;
    }>;
    readonly familyProfile: AceOpt0011Fp16VaeWindowFamilyProfile;
  }>;
}

interface OwnerTracker {
  active: Opt0066OwnedArm | null;
  liveOwners: number;
  peakLiveOwners: number;
  created: Record<Opt0066OwnedArm, number>;
  destroyed: Record<Opt0066OwnedArm, number>;
  everyBackendDestroyIdempotent: boolean;
}

interface PreparedGate {
  readonly identity: Opt0018RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly packages: Readonly<{
    revision6: PreparedPackage;
    revision7: PreparedPackage;
  }>;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly window: AceVaeDecodeWindow;
  readonly fixture: Float32Array<ArrayBuffer>;
  readonly packageLayoutProof: Opt0066PreparedPackageLayoutProof;
  readonly references: Readonly<{
    scalar: Float32Array<ArrayBuffer>;
    candidate: Float32Array<ArrayBuffer>;
  }>;
  readonly topologyByArm: Readonly<Record<
    Opt0066TimedArm,
    Readonly<Record<string, unknown>>
  >>;
  readonly familyTopology: Readonly<Record<string, unknown>>;
  readonly ownerTracker: OwnerTracker;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly readyAtEpochMilliseconds: number;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  destroy(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

class RunObserver {
  private active: {
    readonly label: string;
    progressEventCount: number;
    finalProgress?: AceVaeChunkGpuBackendProgress;
    familyProfileCount: number;
    familyProfile?: AceOpt0011Fp16VaeWindowFamilyProfile;
  } | undefined;

  begin(label: string): void {
    if (this.active !== undefined) {
      throw new Error("OPT-0066 observer already owns a run");
    }
    this.active = {
      label,
      progressEventCount: 0,
      familyProfileCount: 0,
    };
  }

  readonly onProgress = (progress: AceVaeChunkGpuBackendProgress): void => {
    if (this.active === undefined) return;
    this.active.progressEventCount += 1;
    this.active.finalProgress = Object.freeze({ ...progress });
  };

  readonly onFamilyProfile = (
    profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  ): void => {
    if (this.active === undefined) return;
    this.active.familyProfileCount += 1;
    this.active.familyProfile = profile;
  };

  finish(label: string): ObserverSummary {
    const active = this.active;
    this.active = undefined;
    if (
      active === undefined || active.label !== label ||
      active.finalProgress === undefined ||
      active.familyProfileCount !== 1 || active.familyProfile === undefined
    ) {
      throw new Error(`OPT-0066 ${label} observation is incomplete`);
    }
    return Object.freeze({
      progressEventCount: active.progressEventCount,
      finalProgress: active.finalProgress,
      familyProfile: active.familyProfile,
    });
  }

  cancel(label: string): void {
    if (this.active?.label === label) this.active = undefined;
  }
}

let prepared: PreparedGate | undefined;
let operation = Promise.resolve();

globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
  operation = operation.then(async () => {
    const command = event.data;
    if (command.type === "initialize") {
      if (prepared !== undefined) throw new Error("OPT-0066 is already prepared");
      prepared = await prepareGate(validateOpt0018RunIdentity(command.identity));
      postWorker({
        type: "ready-for-thermal-gate",
        readyAtEpochMilliseconds: prepared.readyAtEpochMilliseconds,
        preparation: prepared.preparationReceipt,
      });
      return;
    }
    if (command.type === "dispose") {
      const retained = prepared;
      prepared = undefined;
      await retained?.destroy(new DOMException(
        "OPT-0066 page disposed",
        "AbortError",
      ));
      return;
    }
    if (prepared === undefined) {
      throw new Error("OPT-0066 timing requested before READY");
    }
    validateThermalGate(command.thermalGate, prepared.readyAtEpochMilliseconds);
    const retained = prepared;
    let timed: Readonly<Record<string, unknown>>;
    let cleanup: Readonly<Record<string, unknown>>;
    try {
      timed = await runTimedGate(retained, command.thermalGate);
    } finally {
      cleanup = await retained.destroy();
      if (prepared === retained) prepared = undefined;
    }
    const clean = cleanup["passed"] === true;
    const timedStatus = timed["status"];
    postWorker({
      type: "comparison-complete",
      result: Object.freeze({
        ...timed,
        status: clean && (timedStatus === "passed" || timedStatus === "negative")
          ? timedStatus
          : "failed",
        cleanup,
      }),
    });
  }).catch(async (error: unknown) => {
    const retained = prepared;
    prepared = undefined;
    let cleanup: Readonly<Record<string, unknown>> | undefined;
    try {
      cleanup = await retained?.destroy(error);
    } catch (cleanupError) {
      postWorker({
        type: "failed",
        error: serializeOpt0018Failure(error, cleanupError),
      });
      return;
    }
    postWorker({
      type: "failed",
      error: Object.freeze({
        ...serializeOpt0018Failure(error),
        ...(cleanup === undefined ? {} : { cleanup }),
      }),
    });
  });
});

async function prepareGate(identity: Opt0018RunIdentity): Promise<PreparedGate> {
  const started = performance.now();
  postProgress("authenticating C512 fixture and both VAE package identities");
  const fixtureBytes = createAceOpt0011LatentFixture(OPT_0066_C512_FRAMES);
  if (
    fixtureBytes.byteLength !== OPT_0066_C512_INPUT_ELEMENTS * FLOAT32_BYTES ||
    await sha256Hex(fixtureBytes) !== OPT_0066_C512_FIXTURE_SHA256
  ) {
    throw new Error("OPT-0066 C512 fixture identity changed");
  }
  const fixture = new Float32Array(OPT_0066_C512_INPUT_ELEMENTS);
  fixture.set(new Float32Array(
    fixtureBytes.buffer,
    fixtureBytes.byteOffset,
    OPT_0066_C512_INPUT_ELEMENTS,
  ));
  const [revision6, revision7] = await Promise.all([
    authenticatePackage(6),
    authenticatePackage(7),
  ]);
  const cache = await AceOpfsModelCache.open();
  const revision6WithFiles = await acquirePackageFiles(revision6, cache);
  const revision7WithFiles = await acquirePackageFiles(revision7, cache);
  const packages = Object.freeze({
    revision6: revision6WithFiles,
    revision7: revision7WithFiles,
  });
  postProgress(
    "proving every selected native-to-packed tensor span before WebGPU",
  );
  const packageLayoutProof = await prepareOpt0066PackageLayoutProof(
    Object.freeze({
      manifest: packages.revision6.loaded.manifest,
      files: packages.revision6.acquired,
    }),
    Object.freeze({
      manifest: packages.revision7.loaded.manifest,
      files: packages.revision7.acquired,
    }),
  );
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  const ownerTracker = createOwnerTracker();
  let context: AceWebGpuDeviceContext | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });
    requireFixed32Subgroups(context);
    const plan = planAceVaeChunkedDecode(OPT_0066_C512_FRAMES, {
      chunkFrames: OPT_0066_C512_FRAMES,
      overlapFrames: 64,
    });
    const window = requireOneC512Window(plan);
    const executions: ExecutionResult[] = [];
    for (const [index, arm] of OPT_0066_CORRECTNESS_ORDER.entries()) {
      postProgress(
        `untimed sequential correctness ${index + 1}/` +
          `${OPT_0066_CORRECTNESS_ORDER.length}: ${arm}`,
      );
      executions.push(await executeOwnedArm({
        context,
        packages,
        plan,
        window,
        fixture,
        sameArithmeticOracleTransposeWeights:
          packageLayoutProof.derivedTransposeWeights,
        ownerTracker,
        arm,
        measured: false,
        label: `correctness-${index}-${arm}`,
      }));
      await browserYield();
    }
    const scalarRuns = executions.filter((execution) =>
      execution.arm === "rev6-scalar"
    );
    const sameArithmeticOracleRuns = executions.filter((execution) =>
      execution.arm === "rev6-same-arithmetic-oracle"
    );
    const candidateRuns = executions.filter((execution) =>
      execution.arm === "rev7-candidate"
    );
    if (
      scalarRuns.length !== 2 || sameArithmeticOracleRuns.length !== 2 ||
      candidateRuns.length !== 2
    ) {
      throw new Error("OPT-0066 correctness ownership order changed");
    }
    const outputHashes = await Promise.all(executions.map((execution) =>
      sha256Float32(execution.output)
    ));
    const scalarDeterminism = compareOpt0066Raw(
      scalarRuns[0]!.output,
      scalarRuns[1]!.output,
    );
    const sameArithmeticOracleDeterminism = compareOpt0066Raw(
      sameArithmeticOracleRuns[0]!.output,
      sameArithmeticOracleRuns[1]!.output,
    );
    const candidateDeterminism = compareOpt0066Raw(
      candidateRuns[0]!.output,
      candidateRuns[1]!.output,
    );
    const candidateToSameArithmeticOracle = Object.freeze([
      compareOpt0066Raw(
        sameArithmeticOracleRuns[0]!.output,
        candidateRuns[0]!.output,
      ),
      compareOpt0066Raw(
        sameArithmeticOracleRuns[1]!.output,
        candidateRuns[1]!.output,
      ),
    ]);
    const numericalEnvelope = Object.freeze([
      compareAllChannels(scalarRuns[0]!.output, candidateRuns[0]!.output),
      compareAllChannels(scalarRuns[1]!.output, candidateRuns[1]!.output),
    ]);
    const familyTopology = summarizeFamilyTopology(
      executions[0]!.receipt.familyProfile,
    );
    const stableFamilyTopology = executions.every((execution) =>
      JSON.stringify(summarizeFamilyTopology(
        execution.receipt.familyProfile,
      )) === JSON.stringify(familyTopology)
    );
    const stableArmTopologies = executions.every((execution) => {
      const first = executions.find((candidate) =>
        candidate.arm === execution.arm
      );
      return first !== undefined &&
        JSON.stringify(first.topologySummary) ===
          JSON.stringify(execution.topologySummary);
    });
    const scalarHashesAccepted = scalarRuns.every((run) =>
      outputHashes[executions.indexOf(run)] ===
        OPT_0066_REV6_SCALAR_OUTPUT_SHA256
    );
    const rawOraclePassed = candidateToSameArithmeticOracle.every(
      rawComparisonPassed,
    );
    const numericalPassed = numericalEnvelope.every(allMetricsPassed);
    const deterministic = rawComparisonPassed(scalarDeterminism) &&
      rawComparisonPassed(sameArithmeticOracleDeterminism) &&
      rawComparisonPassed(candidateDeterminism);
    const ownership = snapshotOwnerTracker(ownerTracker);
    if (
      !scalarHashesAccepted || !rawOraclePassed || !numericalPassed ||
      !deterministic || !stableFamilyTopology || !stableArmTopologies ||
      ownership["noLiveOwners"] !== true ||
      ownership["peakLiveOwners"] !== 1 || runtimeEvents.length !== 0
    ) {
      throw new Error(
        "OPT-0066 sequential C512 correctness gate failed: " +
          JSON.stringify({
            scalarHashesAccepted,
            outputHashes: executions.map((execution, index) => ({
              arm: execution.arm,
              sha256: outputHashes[index],
            })),
            rawOraclePassed,
            candidateToSameArithmeticOracle,
            numericalPassed,
            numericalEnvelope,
            deterministic,
            scalarDeterminism,
            sameArithmeticOracleDeterminism,
            candidateDeterminism,
            stableFamilyTopology,
            stableArmTopologies,
            ownership,
            runtimeEvents,
          }),
      );
    }
    const readyAtEpochMilliseconds = Date.now();
    const topologyByArm = Object.freeze({
      "rev6-scalar": scalarRuns[0]!.topologySummary,
      "rev7-candidate": candidateRuns[0]!.topologySummary,
    });
    const preparationReceipt = Object.freeze({
      schema: OPT_0066_BROWSER_SCHEMA,
      status: "ready",
      experimentId: EXPERIMENT_ID,
      identity,
      fixture: Object.freeze({
        frames: OPT_0066_C512_FRAMES,
        elements: OPT_0066_C512_INPUT_ELEMENTS,
        byteLength: fixtureBytes.byteLength,
        sha256: OPT_0066_C512_FIXTURE_SHA256,
      }),
      packages: Object.freeze({
        revision6: packageReceipt(packages.revision6),
        revision7: packageReceipt(packages.revision7),
        boundedPreDeviceNativeToPackedProof: packageLayoutProof.proof,
        retainedRevision6DerivedConvTransposeBytes:
          packageLayoutProof.derivedTransposeWeights.reduce(
            (sum, entry) => sum + entry.words.byteLength,
            0,
          ),
        manifestsAndOpfsFileHandlesMayCoexist: true,
        gpuWeightPhasesOrBackendsMayCoexist: false,
      }),
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        adapterInfo: context.capabilities.adapterInfo,
        deviceFeatures: context.capabilities.deviceFeatures,
        deviceLimits: context.capabilities.deviceLimits,
      }),
      topology: Object.freeze({
        scalar: scalarRuns[0]!.topologySummary,
        completeSameArithmeticOracle:
          sameArithmeticOracleRuns[0]!.topologySummary,
        revision7Candidate: candidateRuns[0]!.topologySummary,
        reconciledK7QuantumCount: K7_TOTAL_QUANTA,
        selectedK7QuantumCount: K7_SELECTED_QUANTA,
        unchangedNativeK7QuantumCount: K7_NATIVE_QUANTA,
        reconciledConvTransposeQuantumCount: CONV_TRANSPOSE_TOTAL_QUANTA,
        unchangedOtherOwners: true,
        stableAcrossReruns: stableArmTopologies,
      }),
      scheduling: Object.freeze({
        quantaPerCommandBuffer:
          ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
        decoderQuantumCount:
          ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT,
        decoderCommandBufferCount: DECODER_COMMAND_BUFFER_COUNT,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: TOTAL_COMMAND_BUFFER_COUNT,
        queueDrainCount: TOTAL_COMMAND_BUFFER_COUNT,
        requestedCooperativeIdleMs: REQUESTED_COOPERATIVE_IDLE_MS,
        familyTopology,
      }),
      correctness: Object.freeze({
        order: OPT_0066_CORRECTNESS_ORDER,
        completeRunsPerArm: 2,
        outputHashes: Object.freeze(executions.map((execution, index) =>
          Object.freeze({ arm: execution.arm, sha256: outputHashes[index] })
        )),
        acceptedRevision6ScalarOutputSha256:
          OPT_0066_REV6_SCALAR_OUTPUT_SHA256,
        scalarDeterminism,
        sameArithmeticOracleDeterminism,
        candidateDeterminism,
        rawCandidateToCompleteSameArithmeticOracle:
          candidateToSameArithmeticOracle,
        rawOraclePassed,
        numericalEnvelopeAgainstRevision6Scalar: numericalEnvelope,
        numericalEnvelopeAuthority: "OPT-0044-unchanged-from-OPT-0024",
        numericalPassed,
        performanceClaim: null,
        passed: true,
      }),
      ownership,
      lifecycle: Object.freeze({
        dedicatedWorker: true,
        oneGpuPackageOwnerAtATime: true,
        candidateAndRevision6NeverCoResident: true,
        everyBackendDestroyedBeforeNextOwner: true,
        cleanupOnPreparationFailure: true,
        cleanupOnTimedFailure: true,
      }),
      runtimeEvents: Object.freeze([...runtimeEvents]),
      preparationWallMs: performance.now() - started,
      readyAtEpochMilliseconds,
      readyForThermalGate: true,
      productionDefaultChanged: false,
    });
    return createPreparedGate({
      identity,
      context,
      packages,
      plan,
      window,
      fixture,
      packageLayoutProof,
      references: Object.freeze({
        scalar: scalarRuns[0]!.output,
        candidate: candidateRuns[0]!.output,
      }),
      topologyByArm,
      familyTopology,
      ownerTracker,
      runtimeEvents,
      readyAtEpochMilliseconds,
      preparationReceipt,
    });
  } catch (error) {
    context?.destroy();
    throw error;
  }
}

function createPreparedGate(input: Omit<PreparedGate, "destroy">): PreparedGate {
  let destroyPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  return Object.freeze({
    ...input,
    destroy(reason: unknown = new DOMException(
      "OPT-0066 comparison complete",
      "AbortError",
    )): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        const started = performance.now();
        const before = snapshotOwnerTracker(input.ownerTracker);
        input.context.destroy();
        const after = snapshotOwnerTracker(input.ownerTracker);
        const passed = before["noLiveOwners"] === true &&
          after["noLiveOwners"] === true &&
          after["createdOwnerCount"] === after["destroyedOwnerCount"] &&
          after["everyBackendDestroyIdempotent"] === true;
        return Object.freeze({
          passed,
          reason: serializeOpt0018Failure(reason),
          ownerStateBeforeDeviceDestroy: before,
          ownerStateAfterDeviceDestroy: after,
          allWeightPhasesBackendsActivationControlAndReadbackDestroyed: passed,
          deviceContextDestroyed: true,
          wallMs: performance.now() - started,
          completedAtEpochMilliseconds: Date.now(),
        });
      })();
      return destroyPromise;
    },
  });
}

async function runTimedGate(
  preparedGate: PreparedGate,
  thermalGate: Opt0066ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  const executions: ExecutionResult[] = [];
  for (const [index, arm] of OPT_0066_TIMED_ORDER.entries()) {
    postProgress(`timed sequential C512 ${index + 1}/4: ${arm}`);
    const execution = await executeOwnedArm({
      context: preparedGate.context,
      packages: preparedGate.packages,
      plan: preparedGate.plan,
      window: preparedGate.window,
      fixture: preparedGate.fixture,
      sameArithmeticOracleTransposeWeights:
        preparedGate.packageLayoutProof.derivedTransposeWeights,
      ownerTracker: preparedGate.ownerTracker,
      arm,
      measured: true,
      label: `timed-${index}-${arm}`,
    });
    if (
      JSON.stringify(execution.topologySummary) !== JSON.stringify(
        preparedGate.topologyByArm[arm],
      ) ||
      JSON.stringify(summarizeFamilyTopology(
        execution.receipt.familyProfile,
      )) !== JSON.stringify(preparedGate.familyTopology)
    ) {
      throw new Error("OPT-0066 timed topology changed after READY");
    }
    executions.push(execution);
    await browserYield();
  }
  const outputReceipts = await Promise.all(executions.map(async (
    execution,
    index,
  ) => {
    const reference = execution.arm === "rev6-scalar"
      ? preparedGate.references.scalar
      : preparedGate.references.candidate;
    return Object.freeze({
      index,
      outputSha256: await sha256Float32(execution.output),
      comparisonToUntimedReference: compareOpt0066Raw(
        reference,
        execution.output,
      ),
      topology: execution.topologySummary,
      ownerLifecycle: execution.ownerLifecycle,
      ...execution.receipt,
    });
  }));
  const deterministic = outputReceipts.every((receipt) =>
    rawComparisonPassed(receipt.comparisonToUntimedReference)
  );
  const timingSamples: Opt0066TimingSample[] = executions.map((execution) =>
    Object.freeze({
      arm: execution.arm as Opt0066TimedArm,
      k7FamilySubmitThroughDrainMs:
        execution.receipt.timing.k7FamilySubmitThroughDrainMs,
      convTransposeFamilySubmitThroughDrainMs:
        execution.receipt.timing.convTransposeFamilySubmitThroughDrainMs,
      decoderSubmitThroughDrainMs:
        execution.receipt.timing.decoderSubmitThroughDrainMs,
      outerWindowWallMs: execution.receipt.timing.outerWindowWallMs,
    })
  );
  const performance = evaluateOpt0066BalancedTiming(timingSamples);
  const numerical = Object.freeze([
    compareAllChannels(executions[0]!.output, executions[1]!.output),
    compareAllChannels(executions[3]!.output, executions[2]!.output),
  ]);
  const numericalPassed = numerical.every(allMetricsPassed);
  const ownership = snapshotOwnerTracker(preparedGate.ownerTracker);
  const correctnessAndLifecyclePassed = deterministic && numericalPassed &&
    ownership["noLiveOwners"] === true &&
    ownership["createdOwnerCount"] === ownership["destroyedOwnerCount"] &&
    preparedGate.runtimeEvents.length === 0;
  const passed = correctnessAndLifecyclePassed && performance.passed;
  return Object.freeze({
    schema: OPT_0066_BROWSER_SCHEMA,
    status: passed
      ? "passed"
      : correctnessAndLifecyclePassed
        ? "negative"
        : "failed",
    decision: passed
      ? "positive-authenticated-dual-k4-c512-quality-gate-passed"
      : correctnessAndLifecyclePassed
        ? "negative-below-k7-convtranspose-or-complete-wall-gate"
        : "failed-correctness-or-lifecycle-gate",
    experimentId: EXPERIMENT_ID,
    identity: preparedGate.identity,
    environment: preparedGate.preparationReceipt["environment"],
    fixture: preparedGate.preparationReceipt["fixture"],
    packages: preparedGate.preparationReceipt["packages"],
    topology: preparedGate.preparationReceipt["topology"],
    protocol: Object.freeze({
      correctnessOrder: OPT_0066_CORRECTNESS_ORDER,
      timedOrder: OPT_0066_TIMED_ORDER,
      completeUntimedRunsPerArm: 2,
      completeTimedRunsPerArm: 2,
      quantaPerCommandBuffer:
        ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyMillisecondsBetweenCommandBuffers: 1,
      sequentialPackageOwnership: true,
      maximumSimultaneousGpuPackageOwners: 1,
      familyTimingUsesHomogeneousCommandBuffers: true,
      mixedFamilyCommandBuffersReportedSeparately: true,
      outerWindowWallIncludesLatentUploadSubmissionDrainReadbackAndMap: true,
      packageAcquisitionUploadAndGraphBuildReportedSeparatelyNotTimedAsDecoder:
        true,
      dedicatedWorker: true,
      stockChromeWebGpuOnly: true,
      experimentalBrowserFlags: false,
      timestampQueries: false,
      webNn: false,
      thermalGate,
    }),
    executions: Object.freeze(outputReceipts),
    deterministic: Object.freeze({ passed: deterministic }),
    numericalEnvelopeAgainstRevision6Scalar: numerical,
    numericalEnvelopeAuthority: "OPT-0044-unchanged-from-OPT-0024",
    numericalPassed,
    performance,
    performanceGate: Object.freeze({
      requiresBothDirectionsToImproveHomogeneousK7Wall: true,
      requiredMedianK7Speedup: 1.50,
      requiresBothDirectionsToImproveHomogeneousConvTransposeWall: true,
      requiredMedianConvTransposeSpeedup: 1.30,
      requiresNoCompleteDecoderRegressionInBothPairedOrders: true,
      requiresNoOuterWindowRegressionInBothPairedOrders: true,
      passed: performance.passed,
    }),
    ownership,
    runtimeEvents: Object.freeze([...preparedGate.runtimeEvents]),
    productionDefaultChanged: false,
    productSelectionAuthorized: false,
    listeningApprovalStillRequired: true,
    under60SecondClaim: false,
  });
}

async function executeOwnedArm(input: Readonly<{
  context: AceWebGpuDeviceContext;
  packages: PreparedGate["packages"];
  plan: AceVaeChunkedDecodePlan;
  window: AceVaeDecodeWindow;
  fixture: Float32Array<ArrayBuffer>;
  sameArithmeticOracleTransposeWeights:
    readonly Opt0066DerivedTransposeWeight[];
  ownerTracker: OwnerTracker;
  arm: Opt0066OwnedArm;
  measured: boolean;
  label: string;
}>): Promise<ExecutionResult> {
  const observer = new RunObserver();
  const setupStarted = performance.now();
  const packageForArm = input.arm === "rev7-candidate"
    ? input.packages.revision7
    : input.packages.revision6;
  const owned = await withOwnedBackend(
    input.context,
    packageForArm,
    input.plan,
    input.fixture,
    input.sameArithmeticOracleTransposeWeights,
    input.ownerTracker,
    input.arm,
    observer,
    async (backend) => {
      const ownerSetupWallMs = performance.now() - setupStarted;
      const topology = backend.captureDispatchTopology();
      const topologySummary = validateTopology(input.arm, topology, backend);
      const executed = await executeBackend(
        backend,
        observer,
        input.window,
        input.arm,
        input.measured,
        input.label,
        ownerSetupWallMs,
      );
      return Object.freeze({ topology, topologySummary, executed });
    },
  );
  return Object.freeze({
    arm: input.arm,
    output: owned.value.executed.output,
    topology: owned.value.topology,
    topologySummary: owned.value.topologySummary,
    ownerLifecycle: owned.lifecycle,
    receipt: owned.value.executed.receipt,
  });
}

async function withOwnedBackend<Value>(
  context: AceWebGpuDeviceContext,
  pkg: PreparedPackage,
  plan: AceVaeChunkedDecodePlan,
  fixture: Float32Array<ArrayBuffer>,
  sameArithmeticOracleTransposeWeights:
    readonly Opt0066DerivedTransposeWeight[],
  tracker: OwnerTracker,
  arm: Opt0066OwnedArm,
  observer: RunObserver,
  use: (backend: AceOpt0011Fp16VaeChunkGpuBackend) => Promise<Value>,
): Promise<Readonly<{
  value: Value;
  lifecycle: Readonly<Record<string, unknown>>;
}>> {
  if (tracker.active !== null || tracker.liveOwners !== 0) {
    throw new Error(
      `OPT-0066 refused co-resident ${arm}; ${String(tracker.active)} is live`,
    );
  }
  tracker.active = arm;
  tracker.liveOwners = 1;
  tracker.peakLiveOwners = Math.max(tracker.peakLiveOwners, 1);
  tracker.created[arm] += 1;
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  let value: Value | undefined;
  let useCompleted = false;
  let idempotentDestroyPromises = true;
  const ownerStarted = performance.now();
  try {
    phase = await loadVaePhase(context.device, pkg, arm);
    const transferredPhase = phase;
    phase = undefined;
    backend = arm === "rev6-same-arithmetic-oracle"
      ? await createCompleteSameArithmeticOracleBackend(
          context.device,
          pkg,
          plan,
          fixture,
          transferredPhase,
          sameArithmeticOracleTransposeWeights,
          observer,
        )
      : await createStandardBackend(
          context.device,
          pkg,
          plan,
          fixture,
          transferredPhase,
          arm,
          observer,
        );
    value = await use(backend);
    useCompleted = true;
  } finally {
    try {
      if (backend !== undefined) {
        const reason = new DOMException(
          `OPT-0066 ${arm} sequential owner complete`,
          "AbortError",
        );
        const first = backend.destroy(reason);
        const second = backend.destroy(reason);
        idempotentDestroyPromises = first === second;
        tracker.everyBackendDestroyIdempotent &&= idempotentDestroyPromises;
        await first;
      } else {
        phase?.destroy();
      }
    } finally {
      tracker.destroyed[arm] += 1;
      tracker.liveOwners = 0;
      tracker.active = null;
    }
  }
  if (!useCompleted) {
    throw new Error(`OPT-0066 ${arm} owner did not complete`);
  }
  return Object.freeze({
    value: value!,
    lifecycle: Object.freeze({
      arm,
      packageRevision: pkg.revision,
      packageManifestSha256: pkg.loaded.manifestSha256,
      phaseBackendAndBuffersDestroyedBeforeNextOwner: true,
      idempotentDestroyPromises,
      liveOwnersAfterDestroy: tracker.liveOwners,
      ownerLifetimeWallMs: performance.now() - ownerStarted,
    }),
  });
}

async function createStandardBackend(
  device: GPUDevice,
  pkg: PreparedPackage,
  plan: AceVaeChunkedDecodePlan,
  fixture: Float32Array<ArrayBuffer>,
  phase: AceGpuTensorPhase,
  arm: Exclude<Opt0066OwnedArm, "rev6-same-arithmetic-oracle">,
  observer: RunObserver,
): Promise<AceOpt0011Fp16VaeChunkGpuBackend> {
  const common = {
    device,
    plan,
    finalLatents: fixture,
    authenticatedPackage: pkg.loaded,
    ownedVaeWeights: phase,
    maximumWindowFrames: OPT_0066_C512_FRAMES,
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
    quantaPerCommandBuffer:
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
    onProgress: observer.onProgress,
    onFamilyProfile: observer.onFamilyProfile,
  };
  return await AceOpt0011Fp16VaeChunkGpuBackend.create(
    arm === "rev6-scalar"
      ? {
          ...common,
          runtimeProfileId:
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
        }
      : {
          ...common,
          runtimeProfileId:
            ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
        },
  );
}

/**
 * Benchmark-private full-decoder oracle: revision-6 native K7 weights use
 * OPT-0024 K4 at the twelve selected labels, while revision-6 native
 * ConvTranspose weights are independently packed and run through OPT-0048 K4
 * at blocks 1-4. All other owners remain literal OPT-0040. This is never a
 * runtime fallback or a scalar-production identity claim.
 */
async function createCompleteSameArithmeticOracleBackend(
  device: GPUDevice,
  pkg: PreparedPackage,
  plan: AceVaeChunkedDecodePlan,
  fixture: Float32Array<ArrayBuffer>,
  phase: AceGpuTensorPhase,
  transposeWeights: readonly Opt0066DerivedTransposeWeight[],
  observer: RunObserver,
): Promise<AceOpt0011Fp16VaeChunkGpuBackend> {
  let buffers: AceOpt0011Fp16VaePreparedBuffers | undefined;
  let runtime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
  let ownershipTransferred = false;
  try {
    const packageBindings = resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      pkg.loaded,
      phase,
    );
    buffers = createOracleBuffers(device);
    runtime = AceOpt0011Fp16VaeDecoderRuntime.create(device, {
      runtimeProfileId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    replaceSelectedK7WithNativeLayoutOpt0024(device, runtime);
    const transposeOracle = installOpt0066CompleteTransposeOracle(
      device,
      runtime,
      transposeWeights,
    );
    if (
      transposeOracle.derivedBufferCount !== 4 ||
      transposeOracle.derivedWeightBytes !== 15_335_424
    ) {
      throw new Error("OPT-0066 complete transpose oracle inventory changed");
    }
    const dispatchSet = await runtime.createChunkDispatchSet(
      "opt-0066-c512-complete-same-arithmetic-oracle",
      plan.latentFrames,
      OPT_0066_C512_FRAMES,
      oracleBindings(buffers, packageBindings),
    );
    ownershipTransferred = true;
    return AceOpt0011Fp16VaeChunkGpuBackend.fromPreparedResources({
      device,
      plan,
      finalLatents: fixture,
      ownedVaeWeights: phase,
      buffers,
      runtime,
      dispatchSet,
      runtimeProfile: ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
      quantaPerCommandBuffer:
        ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      onProgress: observer.onProgress,
      onFamilyProfile: observer.onFamilyProfile,
    });
  } catch (error) {
    if (!ownershipTransferred) {
      runtime?.destroy();
      destroyOracleBuffers(buffers);
      phase.destroy();
    }
    throw error;
  }
}

function replaceSelectedK7WithNativeLayoutOpt0024(
  device: GPUDevice,
  runtime: AceOpt0011Fp16VaeDecoderRuntime,
): void {
  type Replaceable = { subgroupConv1d: AceFp16VaeConv1dSubgroupKernel };
  const replaceable = runtime as unknown as Replaceable;
  const native = replaceable.subgroupConv1d;
  const k4 = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(device, {
    subgroupMinSize: REQUIRED_SUBGROUP_SIZE,
    subgroupMaxSize: REQUIRED_SUBGROUP_SIZE,
  });
  type CreateDispatch = AceFp16VaeConv1dSubgroupKernel["createDispatch"];
  let destroyed = false;
  replaceable.subgroupConv1d = Object.freeze({
    async createDispatch(
      ...args: Parameters<CreateDispatch>
    ): Promise<Awaited<ReturnType<CreateDispatch>>> {
      const operationLabel = operationLabelFromDispatchLabel(args[0]);
      const selection = selectAceOpt0057VaeK7(
        operationLabel,
        args[1],
        args[2].bias !== undefined,
        args[3],
      );
      const dispatch = selection.route.owner === "row-reuse-k4"
        ? await k4.createDispatch(...args)
        : await native.createDispatch(...args);
      return dispatch as Awaited<ReturnType<CreateDispatch>>;
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      native.destroy();
      k4.destroy();
    },
  }) as AceFp16VaeConv1dSubgroupKernel;
}

function operationLabelFromDispatchLabel(label: string): string {
  const match = /-operation-\d+-(.+)-quantum-\d+$/u.exec(label);
  const operationLabel = match?.[1];
  const route = ACE_OPT_0057_VAE_K7_ROUTES.find((candidate) =>
    candidate.operationLabel === operationLabel
  );
  if (route === undefined) {
    throw new Error(`OPT-0066 could not authenticate K7 dispatch label ${label}`);
  }
  return route.operationLabel;
}

function createOracleBuffers(
  device: GPUDevice,
): AceOpt0011Fp16VaePreparedBuffers {
  const graph = planAceVaeDecoder(OPT_0066_C512_FRAMES);
  const create = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const created: GPUBuffer[] = [];
  try {
    const retain = (buffer: GPUBuffer): GPUBuffer => {
      created.push(buffer);
      return buffer;
    };
    const stagingInput = retain(create(
      "opt-0066-k4-oracle-staging-input",
      graph.inputElements * FLOAT32_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ));
    const decoderInput = retain(create(
      "opt-0066-k4-oracle-decoder-input",
      graph.inputElements * FLOAT16_BYTES,
      GPUBufferUsage.STORAGE,
    ));
    const workspaces = Object.freeze([0, 1, 2].map((index) => retain(create(
      `opt-0066-k4-oracle-workspace-${index}`,
      ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
      GPUBufferUsage.STORAGE,
    )))) as readonly [GPUBuffer, GPUBuffer, GPUBuffer];
    const output = retain(create(
      "opt-0066-k4-oracle-output",
      graph.outputElements * FLOAT32_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    ));
    const readback = retain(create(
      "opt-0066-k4-oracle-readback",
      graph.outputElements * FLOAT32_BYTES,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    ));
    return Object.freeze({
      stagingInput,
      decoderInput,
      workspaces,
      output,
      readback,
    });
  } catch (error) {
    for (const buffer of created) buffer.destroy();
    throw error;
  }
}

function destroyOracleBuffers(
  buffers: AceOpt0011Fp16VaePreparedBuffers | undefined,
): void {
  if (buffers === undefined) return;
  buffers.stagingInput.destroy();
  buffers.decoderInput.destroy();
  for (const workspace of buffers.workspaces) workspace.destroy();
  buffers.output.destroy();
  buffers.readback.destroy();
}

function oracleBindings(
  buffers: AceOpt0011Fp16VaePreparedBuffers,
  packageBindings: ReturnType<typeof resolveAceOpt0011Fp16VaePackageBindings>,
): AceOpt0011Fp16VaeWindowBindings {
  const binding = (buffer: GPUBuffer): GPUBufferBinding => Object.freeze({
    buffer,
    offset: 0,
    size: Number(buffer.size),
  });
  return Object.freeze({
    stagingInput: binding(buffers.stagingInput),
    decoderInput: binding(buffers.decoderInput),
    workspaces: Object.freeze(buffers.workspaces.map(binding)) as readonly [
      GPUBufferBinding,
      GPUBufferBinding,
      GPUBufferBinding,
    ],
    output: binding(buffers.output),
    package: packageBindings,
  });
}

async function executeBackend(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  observer: RunObserver,
  window: AceVaeDecodeWindow,
  arm: Opt0066OwnedArm,
  measured: boolean,
  label: string,
  ownerSetupWallMs: number,
): Promise<Readonly<{
  output: Float32Array<ArrayBuffer>;
  receipt: ExecutionResult["receipt"];
}>> {
  observer.begin(label);
  try {
    const started = performance.now();
    const output = await backend.decodeWindow(window);
    const outerWindowWallMs = performance.now() - started;
    const observed = observer.finish(label);
    const progress = observed.finalProgress;
    const profile = observed.familyProfile;
    if (
      output.length !== OPT_0066_C512_OUTPUT_ELEMENTS ||
      profile.windowIndex !== window.index ||
      profile.inputFrames !== OPT_0066_C512_FRAMES ||
      profile.quantaPerCommandBuffer !==
        ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
      profile.decoderQuantumCount !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      profile.decoderBatchCount !== DECODER_COMMAND_BUFFER_COUNT ||
      observed.progressEventCount !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.windowIndex !== window.index ||
      progress.completedDecoderQuanta !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      progress.totalDecoderQuanta !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      progress.completedCommandBuffers !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.totalCommandBuffers !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.queueDrains !== TOTAL_COMMAND_BUFFER_COUNT ||
      progress.cooperativeIdleMs !== REQUESTED_COOPERATIVE_IDLE_MS ||
      progress.stage !== "readback"
    ) {
      throw new Error(`OPT-0066 ${label} scheduling topology changed`);
    }
    const scan = scanOutput(output);
    if (
      scan["elementCount"] !== OPT_0066_C512_OUTPUT_ELEMENTS ||
      scan["nonFiniteCount"] !== 0 || scan["nonzeroCount"] === 0 ||
      scan["stereoDifferenceFrameCount"] === 0 ||
      scan["firstFrameFinite"] !== 1 || scan["lastFrameFinite"] !== 1
    ) {
      throw new Error(`OPT-0066 ${label} output is incomplete`);
    }
    const k7 = profile.families["k7-conv1d"];
    const convTranspose = profile.families["conv-transpose1d"];
    if (
      k7.batchCount < 1 || k7.quantumCount < 1 ||
      !(k7.submitThroughDrainMs > 0) ||
      convTranspose.batchCount < 1 || convTranspose.quantumCount < 1 ||
      !(convTranspose.submitThroughDrainMs > 0) ||
      !(profile.decoderSubmitThroughDrainMs > 0) ||
      !(outerWindowWallMs > 0) || !(ownerSetupWallMs > 0)
    ) {
      throw new Error(`OPT-0066 ${label} family timing is incomplete`);
    }
    return Object.freeze({
      output: output as Float32Array<ArrayBuffer>,
      receipt: Object.freeze({
        arm,
        measured,
        scan,
        timing: Object.freeze({
          ownerSetupWallMs,
          outerWindowWallMs,
          k7FamilySubmitThroughDrainMs: k7.submitThroughDrainMs,
          convTransposeFamilySubmitThroughDrainMs:
            convTranspose.submitThroughDrainMs,
          decoderSubmitThroughDrainMs: profile.decoderSubmitThroughDrainMs,
          decoderCommandBufferCount: profile.decoderBatchCount,
          totalCommandBufferCount: progress.totalCommandBuffers,
          queueDrainCount: progress.queueDrains,
          requestedCooperativeIdleMs: progress.cooperativeIdleMs,
        }),
        familyProfile: profile,
      }),
    });
  } catch (error) {
    observer.cancel(label);
    throw error;
  }
}

const EXPECTED_K7_QUANTA_BY_LABEL = Object.freeze({
  conv1: 2,
  "block-0-res-1-conv1": 160,
  "block-0-res-2-conv1": 160,
  "block-0-res-3-conv1": 160,
  "block-1-res-1-conv1": 240,
  "block-1-res-2-conv1": 240,
  "block-1-res-3-conv1": 240,
  "block-2-res-1-conv1": 240,
  "block-2-res-2-conv1": 240,
  "block-2-res-3-conv1": 240,
  "block-3-res-1-conv1": 240,
  "block-3-res-2-conv1": 240,
  "block-3-res-3-conv1": 240,
  "block-4-res-1-conv1": 480,
  "block-4-res-2-conv1": 480,
  "block-4-res-3-conv1": 480,
  conv2: 8,
} as const);

function validateTopology(
  arm: Opt0066OwnedArm,
  topology: AceOpt0011Fp16VaeDispatchTopologyReceipt,
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
): Readonly<Record<string, unknown>> {
  const window = topology.windows[0];
  const revision6Package = arm !== "rev7-candidate";
  const expectedProfile = revision6Package
    ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE
    : ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
  const expectedKernelTopology = revision6Package
    ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY
    : ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY;
  const expectedK7SelectedKernel = arm === "rev6-same-arithmetic-oracle"
    ? ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID
    : arm === "rev7-candidate"
      ? ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
      : null;
  const expectedKernelCounts: Record<string, number> = {
    [ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID]: 1,
    [ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID]: arm === "rev6-scalar"
      ? K7_TOTAL_QUANTA
      : K7_NATIVE_QUANTA,
    [ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID]: 819,
    [ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID]: 1_611,
    [ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID]: 690,
  };
  if (expectedK7SelectedKernel !== null) {
    expectedKernelCounts[expectedK7SelectedKernel] = K7_SELECTED_QUANTA;
  }
  if (arm !== "rev6-scalar") {
    expectedKernelCounts[ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID] = 92;
    expectedKernelCounts[
      ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
    ] = 276;
    expectedKernelCounts[
      ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID
    ] = 276;
  } else {
    expectedKernelCounts[ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID] =
      368;
    expectedKernelCounts[ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID] =
      276;
  }
  if (
    backend.runtimeProfileId !== expectedProfile.id ||
    backend.kernelSetId !== expectedProfile.kernelSetId ||
    !sameKernelTopology(backend.kernelTopology, expectedKernelTopology) ||
    backend.memory.quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
    topology.uniqueWindowFrames.length !== 1 ||
    topology.uniqueWindowFrames[0] !== OPT_0066_C512_FRAMES ||
    window === undefined || window.inputFrames !== OPT_0066_C512_FRAMES ||
    window.operationCount !== 88 ||
    window.graphQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT ||
    window.sequenceQuantumCount !==
      ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
    !sameCounts(window.kernelQuantumCounts, expectedKernelCounts)
  ) {
    throw new Error(`OPT-0066 ${arm} complete C512 kernel topology changed`);
  }
  const routeByLabel = new Map(ACE_OPT_0057_VAE_K7_ROUTES.map((route) => [
    route.operationLabel,
    route,
  ]));
  const k7Operations = window.operationQuantumCounts.filter((entry) =>
    routeByLabel.has(entry.operationLabel)
  );
  let selectedK7Quanta = 0;
  let nativeK7Quanta = 0;
  let selectedK7Operations = 0;
  let nativeK7Operations = 0;
  for (const entry of k7Operations) {
    const route = routeByLabel.get(entry.operationLabel)!;
    const expectedCount = EXPECTED_K7_QUANTA_BY_LABEL[
      entry.operationLabel as keyof typeof EXPECTED_K7_QUANTA_BY_LABEL
    ];
    const selected = route.owner === "row-reuse-k4";
    const expectedKernel = arm === "rev6-scalar" || !selected
      ? ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
      : expectedK7SelectedKernel;
    if (
      expectedCount === undefined || entry.operationKind !== "conv1d" ||
      entry.quantumCount !== expectedCount || entry.kernelId !== expectedKernel
    ) {
      throw new Error(`OPT-0066 ${arm} K7 route changed at ${entry.operationLabel}`);
    }
    if (selected) {
      selectedK7Operations += 1;
      selectedK7Quanta += entry.quantumCount;
    } else {
      nativeK7Operations += 1;
      nativeK7Quanta += entry.quantumCount;
    }
  }
  const expectedTranspose = arm !== "rev6-scalar"
    ? [
        ["block-0-conv-t1", 92, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
        ["block-1-conv-t1", 138, ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID],
        ["block-2-conv-t1", 138, ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID],
        ["block-3-conv-t1", 138, ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID],
        ["block-4-conv-t1", 138, ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID],
      ] as const
    : [
        ["block-0-conv-t1", 92, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
        ["block-1-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
        ["block-2-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID],
        ["block-3-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID],
        ["block-4-conv-t1", 138, ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID],
      ] as const;
  const transposeOperations = window.operationQuantumCounts.filter((entry) =>
    entry.operationKind === "conv-transpose1d"
  );
  if (
    k7Operations.length !== 17 || selectedK7Operations !== 12 ||
    nativeK7Operations !== 5 || selectedK7Quanta !== K7_SELECTED_QUANTA ||
    nativeK7Quanta !== K7_NATIVE_QUANTA ||
    transposeOperations.length !== 5 ||
    expectedTranspose.some(([label, count, kernelId], index) =>
      transposeOperations[index]?.operationLabel !== label ||
      transposeOperations[index]?.quantumCount !== count ||
      transposeOperations[index]?.kernelId !== kernelId
    )
  ) {
    throw new Error(`OPT-0066 ${arm} operation-level routing changed`);
  }
  const k1 = window.operationQuantumCounts.filter((entry) =>
    entry.operationKind === "conv1d" && !routeByLabel.has(entry.operationLabel)
  );
  const snake = window.operationQuantumCounts.filter((entry) =>
    entry.operationKind === "snake"
  );
  const add = window.operationQuantumCounts.filter((entry) =>
    entry.operationKind === "add"
  );
  if (
    sumQuanta(k1) !== 819 || k1.some((entry) =>
      entry.kernelId !== ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID
    ) ||
    sumQuanta(snake) !== 1_611 || snake.some((entry) =>
      entry.kernelId !== ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID
    ) ||
    sumQuanta(add) !== 690 || add.some((entry) =>
      entry.kernelId !== ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID
    )
  ) {
    throw new Error(`OPT-0066 ${arm} changed an unrelated operation owner`);
  }
  return Object.freeze({
    arm,
    runtimeProfileId: backend.runtimeProfileId,
    kernelSetId: backend.kernelSetId,
    runtimeProfileRole: arm === "rev6-same-arithmetic-oracle"
      ? "revision6-package-and-literal-opt0040-unselected-owner-base"
      : "authenticated-runtime-profile",
    effectiveK7Arithmetic: arm === "rev6-scalar"
      ? "scalar-fp32"
      : "selected-label-increasing-k-cin4-fp16-dot4-then-fp32-running-state",
    operationCount: window.operationCount,
    graphQuantumCount: window.graphQuantumCount,
    sequenceQuantumCount: window.sequenceQuantumCount,
    kernelQuantumCounts: window.kernelQuantumCounts,
    k7: Object.freeze({
      totalQuantumCount: selectedK7Quanta + nativeK7Quanta,
      selectedOperationCount: selectedK7Operations,
      selectedQuantumCount: selectedK7Quanta,
      unchangedNativeOperationCount: nativeK7Operations,
      unchangedNativeQuantumCount: nativeK7Quanta,
      selectedKernelId: expectedK7SelectedKernel,
      nativeKernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
      routes: Object.freeze(k7Operations.map((entry) => Object.freeze({
        operationLabel: entry.operationLabel,
        quantumCount: entry.quantumCount,
        kernelId: entry.kernelId,
      }))),
    }),
    convTranspose: Object.freeze({
      effectiveArithmetic: arm === "rev6-scalar"
        ? "scalar-fp32"
        : "block0-scalar;blocks1-4-opt0048-fp16-dot4-then-fp32-running-state",
      totalQuantumCount: sumQuanta(transposeOperations),
      routes: Object.freeze(expectedTranspose.map(
        ([operationLabel, quantumCount, kernelId]) => Object.freeze({
          operationLabel,
          quantumCount,
          kernelId,
        }),
      )),
    }),
    unrelatedOwners: Object.freeze({
      ingress: Object.freeze({
        kernelId: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
        quantumCount: 1,
      }),
      k1: Object.freeze({
        kernelId: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
        quantumCount: 819,
      }),
      snake: Object.freeze({
        kernelId: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
        quantumCount: 1_611,
      }),
      add: Object.freeze({
        kernelId: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
        quantumCount: 690,
      }),
      unchanged: true,
    }),
    benchmarkPrivateCompleteSameArithmeticOverride:
      arm === "rev6-same-arithmetic-oracle",
  });
}

function sameKernelTopology(
  actual: AceOpt0011Fp16VaeChunkGpuBackend["kernelTopology"],
  expected: AceOpt0011Fp16VaeChunkGpuBackend["kernelTopology"],
): boolean {
  return actual.id === expected.id && actual.backend === expected.backend &&
    actual.ingress === expected.ingress &&
    actual.conv1dK1 === expected.conv1dK1 &&
    actual.conv1dK7 === expected.conv1dK7 &&
    actual.convTranspose1d === expected.convTranspose1d &&
    actual.snake === expected.snake && actual.add === expected.add;
}

function sameCounts(
  actual: Readonly<Record<string, number>>,
  expected: Readonly<Record<string, number>>,
): boolean {
  const names = Object.keys(expected);
  return Object.keys(actual).length === names.length &&
    names.every((name) => actual[name] === expected[name]);
}

function sumQuanta(
  entries: readonly Readonly<{ readonly quantumCount: number }>[],
): number {
  return entries.reduce((sum, entry) => sum + entry.quantumCount, 0);
}

async function authenticatePackage(
  revision: 6 | 7,
): Promise<Omit<PreparedPackage, "acquired">> {
  const manifestPath = revision === 6
    ? REVISION6_MANIFEST_PATH
    : REVISION7_MANIFEST_PATH;
  const expectedManifestSha256 = revision === 6
    ? ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256
    : ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256;
  const expectedManifestBytes = revision === 6
    ? ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES
    : ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES;
  const expectedConverterRevision = revision === 6
    ? ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION
    : ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION;
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(manifestPath, globalThis.location.href).href,
    expectedManifestSha256,
    expectedProfile: "fp16-vae-experimental",
    ...(revision === 7
      ? { authenticatedVaeConverterRevision: 7 as const }
      : {}),
  });
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) =>
    shardNames.has(file.name)
  );
  const residentBytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  if (
    loaded.manifestSha256 !== expectedManifestSha256 ||
    loaded.manifestByteLength !== expectedManifestBytes ||
    loaded.manifest.provenance.converterRevision !==
      expectedConverterRevision ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    shardNames.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) {
    throw new Error(
      `OPT-0066 authenticated revision-${revision} package identity changed`,
    );
  }
  return Object.freeze({
    revision,
    manifestPath,
    loaded,
    files: Object.freeze(files),
    residentBytes,
  });
}

async function acquirePackageFiles(
  pkg: Omit<PreparedPackage, "acquired">,
  cache: AceOpfsModelCache,
): Promise<PreparedPackage> {
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({
      ...pkg.loaded.manifest,
      files: pkg.files,
    }),
    manifestUrl: pkg.loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => postProgress(
      `acquiring revision-${pkg.revision} VAE ` +
        `${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== pkg.files.length ||
    acquired.plan.runtimeBytes !== pkg.residentBytes
  ) {
    throw new Error(
      `OPT-0066 revision-${pkg.revision} package acquisition changed`,
    );
  }
  return Object.freeze({ ...pkg, acquired: acquired.files });
}

async function loadVaePhase(
  device: GPUDevice,
  pkg: PreparedPackage,
  arm: Opt0066OwnedArm,
): Promise<AceGpuTensorPhase> {
  postProgress(`uploading sole sequential ${arm} VAE weight phase`);
  const phase = await AceGpuTensorPhase.load(
    device,
    pkg.loaded.manifest,
    pkg.acquired,
    ["vae"],
    {
      onProgress: (progress) => postProgress(
        `uploading ${arm} VAE ${progress.phaseFileIndex + 1}/` +
          `${progress.phaseFileCount}: ${progress.loadedPhaseBytes}/` +
          `${progress.totalPhaseBytes} bytes`,
      ),
    },
  );
  if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
    phase.destroy();
    throw new Error(`OPT-0066 ${arm} resident VAE bytes changed`);
  }
  return phase;
}

function packageReceipt(pkg: PreparedPackage): Readonly<Record<string, unknown>> {
  return Object.freeze({
    revision: pkg.revision,
    manifestPath: pkg.manifestPath,
    manifestUrl: pkg.loaded.manifestUrl,
    manifestSha256: pkg.loaded.manifestSha256,
    manifestByteLength: pkg.loaded.manifestByteLength,
    converterRevision: pkg.loaded.manifest.provenance.converterRevision,
    tensorRecordCount: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    weightFileCount: pkg.files.length,
    residentBytes: pkg.residentBytes,
    authenticatedBeforeGpuUpload: true,
  });
}

function createOwnerTracker(): OwnerTracker {
  return {
    active: null,
    liveOwners: 0,
    peakLiveOwners: 0,
    created: {
      "rev6-scalar": 0,
      "rev6-same-arithmetic-oracle": 0,
      "rev7-candidate": 0,
    },
    destroyed: {
      "rev6-scalar": 0,
      "rev6-same-arithmetic-oracle": 0,
      "rev7-candidate": 0,
    },
    everyBackendDestroyIdempotent: true,
  };
}

function snapshotOwnerTracker(
  tracker: OwnerTracker,
): Readonly<Record<string, unknown>> {
  const created = Object.freeze({ ...tracker.created });
  const destroyed = Object.freeze({ ...tracker.destroyed });
  const createdOwnerCount = Object.values(created).reduce(
    (sum, count) => sum + count,
    0,
  );
  const destroyedOwnerCount = Object.values(destroyed).reduce(
    (sum, count) => sum + count,
    0,
  );
  return Object.freeze({
    active: tracker.active,
    liveOwners: tracker.liveOwners,
    peakLiveOwners: tracker.peakLiveOwners,
    maximumAllowedLiveOwners: 1,
    created,
    destroyed,
    createdOwnerCount,
    destroyedOwnerCount,
    noLiveOwners: tracker.active === null && tracker.liveOwners === 0,
    allCreatedOwnersDestroyed: createdOwnerCount === destroyedOwnerCount,
    everyBackendDestroyIdempotent: tracker.everyBackendDestroyIdempotent,
  });
}

function requireOneC512Window(plan: AceVaeChunkedDecodePlan): AceVaeDecodeWindow {
  const window = plan.windows[0];
  if (
    plan.latentFrames !== OPT_0066_C512_FRAMES ||
    plan.maximumWindowFrames !== OPT_0066_C512_FRAMES ||
    plan.windows.length !== 1 || window === undefined || window.index !== 0 ||
    window.latentWindowFrames !== OPT_0066_C512_FRAMES ||
    window.discardPrefixLatentFrames !== 0 ||
    window.discardSuffixLatentFrames !== 0
  ) {
    throw new Error("OPT-0066 plan is not one exact C512 window");
  }
  return window;
}

function compareAllChannels(
  control: Float32Array,
  candidate: Float32Array,
): Readonly<Record<"joint" | "left" | "right", Opt0066WaveformMetrics>> {
  return Object.freeze({
    joint: compareOpt0066Waveforms(control, candidate),
    left: compareOpt0066Waveforms(control, candidate, 2, 0),
    right: compareOpt0066Waveforms(control, candidate, 2, 1),
  });
}

function allMetricsPassed(
  metrics: Readonly<Record<"joint" | "left" | "right",
    Opt0066WaveformMetrics>>,
): boolean {
  return metrics.joint.passed && metrics.left.passed && metrics.right.passed;
}

function rawComparisonPassed(comparison: Opt0066RawComparison): boolean {
  return comparison.comparedU32WordCount === OPT_0066_C512_OUTPUT_ELEMENTS &&
    comparison.comparedU16WordCount === OPT_0066_C512_OUTPUT_U16_WORDS &&
    comparison.rawU32Exact && comparison.rawU16Exact;
}

function summarizeFamilyTopology(
  profile: AceOpt0011Fp16VaeWindowFamilyProfile,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    quantaPerCommandBuffer: profile.quantaPerCommandBuffer,
    decoderBatchCount: profile.decoderBatchCount,
    decoderQuantumCount: profile.decoderQuantumCount,
    homogeneousBatchCount: profile.homogeneousBatchCount,
    homogeneousQuantumCount: profile.homogeneousQuantumCount,
    mixedBatchCount: profile.mixedBatchCount,
    mixedQuantumCount: profile.mixedQuantumCount,
    families: Object.freeze(Object.fromEntries(Object.entries(
      profile.families,
    ).map(([family, timing]) => [
      family,
      Object.freeze({
        batchCount: timing.batchCount,
        quantumCount: timing.quantumCount,
      }),
    ]))),
  });
}

function scanOutput(output: Float32Array): Readonly<Record<string, number>> {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let stereoDifferenceFrameCount = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let peak = 0;
  for (let index = 0; index < output.length; index += 1) {
    const value = output[index]!;
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (index % 2 === 0 && output[index + 1] !== value) {
      stereoDifferenceFrameCount += 1;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    peak = Math.max(peak, Math.abs(value));
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    nonFiniteCount,
    nonzeroCount,
    stereoDifferenceFrameCount,
    minimum,
    maximum,
    peak,
    firstFrameFinite: Number(
      Number.isFinite(output[0]) && Number.isFinite(output[1]),
    ),
    lastFrameFinite: Number(
      Number.isFinite(output.at(-2)) && Number.isFinite(output.at(-1)),
    ),
  });
}

function requireFixed32Subgroups(context: AceWebGpuDeviceContext): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== REQUIRED_SUBGROUP_SIZE ||
    info.subgroupMaxSize !== REQUIRED_SUBGROUP_SIZE
  ) {
    throw new Error("OPT-0066 requires stock fixed 32-lane WebGPU subgroups");
  }
}

function validateThermalGate(
  gate: Opt0066ThermalGate,
  readyAtEpochMilliseconds: number,
): void {
  if (
    gate.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    gate.command !== "notifyutil -g com.apple.system.thermalpressurelevel" ||
    gate.protocol !== "wait-30s-then-one-level0-check" ||
    gate.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    gate.checkedAtEpochMilliseconds < gate.startedAtEpochMilliseconds ||
    gate.durationMilliseconds < 30_000 || gate.observationCount !== 1 ||
    gate.observedLevel !== 0 ||
    gate.maximumObservationGapMilliseconds !== gate.durationMilliseconds
  ) {
    throw new Error("OPT-0066 worker rejected the one-check thermal gate");
  }
}

async function sha256Float32(values: Float32Array): Promise<string> {
  return await sha256Hex(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest(
    "SHA-256",
    bytes as Uint8Array<ArrayBuffer>,
  ));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function postProgress(message: string): void {
  postWorker({ type: "progress", message });
}

function postWorker(event: WorkerEvent): void {
  (globalThis as unknown as DedicatedWorkerGlobalScope).postMessage(event);
}
