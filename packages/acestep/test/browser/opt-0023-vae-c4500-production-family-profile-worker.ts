/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  createAceOpt0011LatentFixture,
} from "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import {
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import { createAceOpt0011VaeAcquisitionManifest } from
  "../../src/runtime/webgpu-pipeline.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  AceOpt0011Fp16VaeChunkGpuBackend,
  type AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
  type AceOpt0011Fp16VaeProfileFamily,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  planAceVaeChunkedDecode,
  streamAceVaeRawChunks,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
  type AceVaeRawStreamStats,
} from "../../src/webgpu/vae-chunks.js";
import {
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
  planAceOpt0011Fp16VaeChunkDispatches,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
  requireAceOpt0011Fp16VaePackageIdentity,
} from "../../src/webgpu/vae-fp16-profile.js";
import { AceVaeRawF32FileSink } from "../../src/webgpu/vae-wav.js";
import fixtureSource from
  "../../benchmark/opt-0011-vae-fp16-storage-window.ts?raw";
import acquireSource from "../../src/model/acquire.ts?raw";
import cacheSource from "../../src/model/cache.ts?raw";
import gpuTensorsSource from "../../src/model/gpu-tensors.ts?raw";
import gpuUploadSource from "../../src/model/gpu-upload.ts?raw";
import manifestSource from "../../src/model/manifest.ts?raw";
import packageSource from "../../src/model/package.ts?raw";
import sha256Source from "../../src/model/sha256.ts?raw";
import strictJsonSource from "../../src/model/strict-json.ts?raw";
import pipelineSource from "../../src/runtime/webgpu-pipeline.ts?raw";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import capabilitiesSource from "../../src/webgpu/capabilities.ts?raw";
import deviceSource from "../../src/webgpu/device.ts?raw";
import scopedBuffersSource from
  "../../src/webgpu/scoped-buffer-allocation.ts?raw";
import vaeBackendSource from "../../src/webgpu/vae-backend.ts?raw";
import chunksSource from "../../src/webgpu/vae-chunks.ts?raw";
import decoderSource from "../../src/webgpu/vae-decoder.ts?raw";
import fp16BackendSource from "../../src/webgpu/vae-fp16-backend.ts?raw";
import fp16DecoderSource from "../../src/webgpu/vae-fp16-decoder.ts?raw";
import fp16PackageSource from "../../src/webgpu/vae-fp16-package.ts?raw";
import fp16ProfileSource from "../../src/webgpu/vae-fp16-profile.ts?raw";
import wavSource from "../../src/webgpu/vae-wav.ts?raw";
import correctnessUtilsSource from
  "../../src/webgpu/kernels/correctness-utils.ts?raw";
import vaePrimitivesSource from
  "../../src/webgpu/kernels/vae-primitives.ts?raw";
import conv1dSource from "../../src/webgpu/kernels/vae-conv1d-fp16.ts?raw";
import conv1dSubgroupSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts?raw";
import convTransposeSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import pointwiseSource from
  "../../src/webgpu/kernels/vae-pointwise-fp16.ts?raw";
import snakeSource from "../../src/webgpu/kernels/vae-snake-fp16.ts?raw";
import workerSource from
  "./opt-0023-vae-c4500-production-family-profile-worker.ts?raw";
import pageSource from
  "./opt-0023-vae-c4500-production-family-profile.ts?raw";
import htmlSource from
  "./opt-0023-vae-c4500-production-family-profile.html?raw";
import contractSource from
  "../opt-0023-vae-c4500-production-family-profile-contract.test.ts?raw";
import {
  OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS,
  OPT_0023_THERMAL_COMMAND,
  parseOpt0023RunIdentity,
  serializeOpt0023Failure,
  type Opt0023RunIdentity,
  type Opt0023ThermalGate,
} from "./opt-0023-vae-c4500-production-family-profile.js";

export const OPT_0023_LATENT_FRAMES = 4_500;
export const OPT_0023_LATENT_CHANNELS = 64;
export const OPT_0023_LATENT_ELEMENTS = 288_000;
export const OPT_0023_LATENT_BYTES = 1_152_000;
export const OPT_0023_LATENT_SHA256 =
  "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971";
export const OPT_0023_OUTPUT_AUDIO_FRAMES = 8_640_000;
export const OPT_0023_OUTPUT_ELEMENTS = 17_280_000;
export const OPT_0023_OUTPUT_BYTES = 69_120_000;
export const OPT_0023_DETACHED_WINDOW_BYTES = 90_746_880;
export const OPT_0023_ACCOUNTED_GPU_BYTES = 944_808_752;
export const OPT_0023_TRACKED_BUFFER_COUNT = 17;
export const OPT_0023_TIMED_MAP_COUNT = 12;
export const OPT_0023_TOTAL_MAP_COUNT = 13;
export const OPT_0023_DECODER_COMMAND_BUFFERS = 11_338;
export const OPT_0023_TOTAL_COMMAND_BUFFERS = 11_350;
export const OPT_0023_INTERNAL_IDLE_MILLISECONDS = 11_338;
export const OPT_0023_BETWEEN_WINDOW_IDLE_MILLISECONDS = 11;
export const OPT_0023_RUNTIME_PROFILE =
  "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" as const;
export const OPT_0023_KERNEL_SET =
  "opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1" as const;
const MANIFEST_PATH = "/model/files-fp16-vae-experimental/manifest.json";
const RAW_HASH_MAXIMUM_CHUNK_BYTES = 1024 * 1024;
const REQUIRED_WORKSPACE_BYTES = 251_658_240;
const OPT_0023_PROGRESS_COMMAND_BUFFER_TOTALS = Object.freeze([
  863, 983, 983, 983, 983, 983, 983, 983, 983, 983, 983, 657,
] as const);
const OPT_0023_PROGRESS_DECODER_QUANTA = Object.freeze([
  6_895, 7_855, 7_855, 7_855, 7_855, 7_855,
  7_855, 7_855, 7_855, 7_855, 7_855, 5_242,
] as const);
const ACE_SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0";
const ACE_MAIN_MODEL_REVISION = "19671f406d603126926c1b7e2adc169acbcade22";
const ACE_PLANNER_MODEL_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80";
const OPT_0015_INTEGRATED_COMMIT = "36608b857827b2b1d31ac91bf5cca9639fb0b9ed";
const OPT_0023_REGISTRATION_COMMIT =
  "9a3e37d48c75139f98bfb9958f35061247b56da6";
const OPT_0023_REGISTRATION_RECORD_SHA256 =
  "c87b472ed544ba3a0177c41ba7e66bb33cb4c9ececb88be64da0e4d2845a5ee1";

const REGISTERED_SOURCE_SHA256 = Object.freeze({
  "src/runtime/webgpu-pipeline.ts":
    "255e06c39470f87a50d7e3f3e4a02d0b457fd81ff86eb5b95591ba86f6e6b677",
  "src/webgpu/vae-chunks.ts":
    "0059b79fcce3ad5679d3c05f5f3da1954db80194775540521d68002253f5baaf",
  "src/webgpu/vae-fp16-backend.ts":
    "fcab85b55888e491458d72e915a5b8b838d948f305ed9c1ec935db980a850dd9",
  "src/webgpu/vae-fp16-decoder.ts":
    "c229da8de47f5ab1445cd1bbdc1356e6d07f5b35adfcda94e3d702a7ccab75b1",
  "src/webgpu/vae-fp16-profile.ts":
    "946dfc164759037fade6270c1bc67ed928dc6cbdf2fa63ffa785a3622be36f23",
});

const SOURCE_TEXT = Object.freeze({
  "benchmark/opt-0011-vae-fp16-storage-window.ts": fixtureSource,
  "src/model/acquire.ts": acquireSource,
  "src/model/cache.ts": cacheSource,
  "src/model/gpu-tensors.ts": gpuTensorsSource,
  "src/model/gpu-upload.ts": gpuUploadSource,
  "src/model/manifest.ts": manifestSource,
  "src/model/package.ts": packageSource,
  "src/model/sha256.ts": sha256Source,
  "src/model/strict-json.ts": strictJsonSource,
  "src/runtime/scheduler.ts": schedulerSource,
  "src/runtime/webgpu-pipeline.ts": pipelineSource,
  "src/webgpu/capabilities.ts": capabilitiesSource,
  "src/webgpu/device.ts": deviceSource,
  "src/webgpu/scoped-buffer-allocation.ts": scopedBuffersSource,
  "src/webgpu/vae-backend.ts": vaeBackendSource,
  "src/webgpu/vae-chunks.ts": chunksSource,
  "src/webgpu/vae-decoder.ts": decoderSource,
  "src/webgpu/vae-fp16-backend.ts": fp16BackendSource,
  "src/webgpu/vae-fp16-decoder.ts": fp16DecoderSource,
  "src/webgpu/vae-fp16-package.ts": fp16PackageSource,
  "src/webgpu/vae-fp16-profile.ts": fp16ProfileSource,
  "src/webgpu/vae-wav.ts": wavSource,
  "src/webgpu/kernels/correctness-utils.ts": correctnessUtilsSource,
  "src/webgpu/kernels/vae-primitives.ts": vaePrimitivesSource,
  "src/webgpu/kernels/vae-conv1d-fp16.ts": conv1dSource,
  "src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts": conv1dSubgroupSource,
  "src/webgpu/kernels/vae-conv-transpose1d-fp16.ts": convTransposeSource,
  "src/webgpu/kernels/vae-pointwise-fp16.ts": pointwiseSource,
  "src/webgpu/kernels/vae-snake-fp16.ts": snakeSource,
  "test/browser/opt-0023-vae-c4500-production-family-profile-worker.ts":
    workerSource,
  "test/browser/opt-0023-vae-c4500-production-family-profile.ts": pageSource,
  "test/browser/opt-0023-vae-c4500-production-family-profile.html": htmlSource,
  "test/opt-0023-vae-c4500-production-family-profile-contract.test.ts":
    contractSource,
});

type IncomingMessage =
  | Readonly<{ type: "initialize"; identity: unknown }>
  | Readonly<{ type: "run"; thermalGate: Opt0023ThermalGate }>
  | Readonly<{ type: "dispose" }>;

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly acquisition: Readonly<Record<string, unknown>>;
}

interface PreparedSession {
  readonly identity: Opt0023RunIdentity;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly package: PreparedPackage;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly topology: ReturnType<typeof validateOpt0023Topology>;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly audit: DeviceResourceAudit;
  readonly profiles: ProfileCollector;
  readonly progress: ProgressCollector;
  readonly backend: AceOpt0011Fp16VaeChunkGpuBackend;
  readonly memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly preparation: Readonly<Record<string, unknown>> & {
    readonly warmupStartedAtEpochMilliseconds: number;
    readonly warmupCompletedAtEpochMilliseconds: number;
  };
  backendReleased: boolean;
  raw?: Opt0023RawFile;
  cleanupPromise?: Promise<Readonly<Record<string, unknown>>>;
}

interface WindowWall {
  readonly windowIndex: number;
  readonly inputFrames: number;
  readonly startedAtMilliseconds: number;
  readonly completedAtMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly outputElements: number;
}

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let session: PreparedSession | undefined;

function installWorker(): void {
  self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
    if (event.data.type === "initialize" && lifecycle === "idle") {
      lifecycle = "preparing";
      void initializeSession(event.data.identity).then(
        (prepared) => {
          if (lifecycle !== "preparing") return;
          session = prepared;
          lifecycle = "ready";
          self.postMessage({
            type: "ready-for-thermal-gate",
            preparation: prepared.preparation,
          });
        },
        (error: unknown) => fail(error),
      );
      return;
    }
    if (event.data.type === "run" && lifecycle === "ready") {
      lifecycle = "running";
      const active = session!;
      session = undefined;
      void runProfile(active, event.data.thermalGate).then(
        (result) => {
          lifecycle = "settled";
          self.postMessage({ type: "profile-complete", result });
        },
        (error: unknown) => fail(error, active),
      );
      return;
    }
    if (event.data.type === "dispose" && lifecycle === "ready") {
      lifecycle = "running";
      const active = session!;
      session = undefined;
      void cleanupPrepared(active).then(
        () => {
          lifecycle = "settled";
          self.postMessage({ type: "disposed" });
        },
        (error: unknown) => fail(error),
      );
    }
  });
}

if (
  typeof WorkerGlobalScope !== "undefined" &&
  self instanceof WorkerGlobalScope
) installWorker();

async function initializeSession(identityValue: unknown): Promise<PreparedSession> {
  const preparationStarted = performance.now();
  const initializationStartedAtEpochMilliseconds = Date.now();
  const identity = validateRunIdentity(identityValue);
  postProgress("authenticating controlling source bytes and latent fixture");
  const sourceAuthority = createSourceAuthority();
  let latentFixture: Float32Array<ArrayBuffer> | undefined =
    createLatentFixture().latent;
  const plan = validateOpt0023Plan(planAceVaeChunkedDecode(
    OPT_0023_LATENT_FRAMES,
    { chunkFrames: 512, overlapFrames: 64 },
  ));
  const topology = validateOpt0023Topology(plan);

  const acquisitionStarted = performance.now();
  const preparedPackage = await preparePackage();
  const packageAcquisitionWallMilliseconds =
    performance.now() - acquisitionStarted;
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  postProgress("requesting shader-f16 + fixed32 subgroup WebGPU device");
  const deviceStarted = performance.now();
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxBufferSize: REQUIRED_WORKSPACE_BYTES,
      maxStorageBufferBindingSize: REQUIRED_WORKSPACE_BYTES,
    },
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  const deviceRequestWallMilliseconds = performance.now() - deviceStarted;
  const profiles = new ProfileCollector();
  const progress = new ProgressCollector();
  let audit: DeviceResourceAudit | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  try {
    requireDevice(context);
    audit = new DeviceResourceAudit(context.device);
    postProgress("uploading the seven authenticated VAE weight files");
    audit.setScope("weights");
    const uploadStarted = performance.now();
    phase = await AceGpuTensorPhase.load(
      audit.device,
      preparedPackage.loaded.manifest,
      preparedPackage.acquiredFiles,
      ["vae"],
    );
    const phaseUploadWallMilliseconds = performance.now() - uploadStarted;
    if (
      phase.phases.length !== 1 || phase.phases[0] !== "vae" ||
      phase.packageManifest !== preparedPackage.loaded.manifest ||
      phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES
    ) throw new Error("OPT-0023 VAE phase upload accounting changed");

    postProgress("compiling and allocating unchanged OPT-0015 VAE backend");
    audit.setScope("backend");
    const backendCreateStarted = performance.now();
    const transferredPhase = phase;
    phase = undefined;
    const transferredLatent = latentFixture;
    if (transferredLatent === undefined) {
      throw new Error("OPT-0023 latent fixture was released before backend create");
    }
    backend = await AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: audit.device,
      plan,
      finalLatents: transferredLatent,
      authenticatedPackage: preparedPackage.loaded,
      ownedVaeWeights: transferredPhase,
      maximumWindowFrames: 512,
      runtimeProfileId: OPT_0023_RUNTIME_PROFILE,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      onProgress: (event) => progress.accept(event),
      onFamilyProfile: (profile) => profiles.accept(profile),
    });
    latentFixture = undefined;
    const combinedBackendCreateWallMilliseconds =
      performance.now() - backendCreateStarted;
    validateBackendIdentity(backend);
    validateMemory(backend.memory);
    audit.requireLive(OPT_0023_TRACKED_BUFFER_COUNT, OPT_0023_ACCOUNTED_GPU_BYTES);

    postProgress("running the one untimed exact C512 warmup");
    const warmupWindow = plan.windows[1];
    if (warmupWindow?.latentWindowFrames !== 512) {
      throw new Error("OPT-0023 exact C512 warmup window is absent");
    }
    const warmup = await runWarmup(
      backend,
      warmupWindow,
      profiles,
      progress,
    );
    profiles.armTimed();
    progress.armTimed();
    audit.armTimed();
    if (runtimeEvents.length !== 0) {
      throw new Error("WebGPU emitted a runtime event during OPT-0023 preparation");
    }
    const memoryAtTimedStart = audit.snapshot();
    const stableBackend = backend;
    backend = undefined;
    return {
      identity,
      sourceAuthority,
      package: preparedPackage,
      plan,
      topology,
      context,
      runtimeEvents,
      audit,
      profiles,
      progress,
      backend: stableBackend,
      memory: stableBackend.memory,
      preparation: Object.freeze({
        initializationStartedAtEpochMilliseconds,
        initializationCompletedAtEpochMilliseconds: Date.now(),
        initializationWallMilliseconds: performance.now() - preparationStarted,
        packageAcquisitionWallMilliseconds,
        deviceRequestWallMilliseconds,
        phaseUploadWallMilliseconds,
        combinedBackendCreateWallMilliseconds,
        warmupStartedAtEpochMilliseconds:
          warmup.startedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds:
          warmup.completedAtEpochMilliseconds,
        warmupWallMilliseconds: warmup.wallMilliseconds,
        warmupOutput: warmup.output,
        warmupProgress: warmup.progress,
        memoryAtTimedStart,
        exactlyOneUntimedC512Warmup: true,
        prefillDispatchOrQueueWriteAdded: false,
      }),
      backendReleased: false,
    };
  } catch (error) {
    const cleanupErrors: unknown[] = [];
    try {
      await settleOpt0023InitializationOwners({
        ...(phase === undefined ? {} : { phase }),
        ...(backend === undefined ? {} : { backend }),
        context,
      });
    } catch (cleanupError) { cleanupErrors.push(cleanupError); }
    const afterCleanup = audit?.snapshot();
    if (
      afterCleanup !== undefined &&
      !resourceCleanupIsBalanced(afterCleanup)
    ) {
      cleanupErrors.push(new Error(
        "OPT-0023 initialization cleanup did not balance GPU resources",
      ));
    }
    if (cleanupErrors.length !== 0) {
      throw new Opt0023FailureWithCleanup(error, cleanupErrors);
    }
    throw error;
  }
}

async function runProfile(
  prepared: PreparedSession,
  thermalGate: Opt0023ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  try {
    validateWorkerThermalGate(thermalGate, prepared.preparation);
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("WebGPU emitted a runtime event before the timed run");
    }
    prepared.raw = await Opt0023RawFile.create(prepared.plan);
    const preDispatchLaunchDelayMilliseconds =
      Date.now() - thermalGate.gateCompletedAtEpochMilliseconds;
    if (
      preDispatchLaunchDelayMilliseconds < 0 ||
      preDispatchLaunchDelayMilliseconds >
        OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
    ) {
      throw new Error("OPT-0023 raw setup missed the pre-dispatch launch bound");
    }
    const windowWalls: WindowWall[] = [];
    const observedBackend = Object.freeze({
      decodeWindow: async (window: AceVaeDecodeWindow): Promise<Float32Array> => {
        const startedAtMilliseconds = performance.now();
        const output = await prepared.backend.decodeWindow(window);
        const completedAtMilliseconds = performance.now();
        windowWalls.push(Object.freeze({
          windowIndex: window.index,
          inputFrames: window.latentWindowFrames,
          startedAtMilliseconds,
          completedAtMilliseconds,
          wallMilliseconds: completedAtMilliseconds - startedAtMilliseconds,
          outputElements: output.length,
        }));
        return output;
      },
    });

    const timedStarted = performance.now();
    const stats = await streamAceVaeRawChunks(
      prepared.plan,
      observedBackend,
      prepared.raw.sink,
    );
    prepared.raw.sink.finish();
    const timedCompleted = performance.now();
    const timedStartedAtEpochMilliseconds = performance.timeOrigin + timedStarted;
    const timedCompletedAtEpochMilliseconds = performance.timeOrigin + timedCompleted;
    const fullStreamWallMilliseconds = timedCompleted - timedStarted;
    const gateToTimedStartMilliseconds =
      timedStartedAtEpochMilliseconds - thermalGate.gateCompletedAtEpochMilliseconds;
    if (
      gateToTimedStartMilliseconds < 0 ||
      gateToTimedStartMilliseconds > OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
    ) throw new Error("OPT-0023 timed stream missed the post-gate launch bound");

    const profiles = prepared.profiles.finishTimed(prepared.plan.windows);
    const rawSizeBeforeClose = prepared.raw.getSize();
    validateTimedResult(
      prepared,
      stats,
      windowWalls,
      profiles,
      rawSizeBeforeClose,
    );
    const timedResources = prepared.audit.timedSnapshot();
    const postTimingValidationWallMilliseconds =
      performance.now() - timedCompleted;
    const backendDestroyStartedAtEpochMilliseconds = Date.now();
    const backendDestroyStarted = performance.now();
    await prepared.backend.destroy();
    await prepared.backend.destroy();
    const backendDestroyWallMilliseconds =
      performance.now() - backendDestroyStarted;
    const backendDestroyCompletedAtEpochMilliseconds = Date.now();
    prepared.backendReleased = true;
    const resourcesAfterBackendDestroy = prepared.audit.snapshot();
    if (
      resourcesAfterBackendDestroy.liveBufferCount !== 0 ||
      resourcesAfterBackendDestroy.destroyedBufferCount !==
        OPT_0023_TRACKED_BUFFER_COUNT ||
      resourcesAfterBackendDestroy.mapCount !== OPT_0023_TOTAL_MAP_COUNT ||
      resourcesAfterBackendDestroy.unmapCount !== OPT_0023_TOTAL_MAP_COUNT ||
      resourcesAfterBackendDestroy.mappedBufferCount !== 0 ||
      resourcesAfterBackendDestroy.totalDestroyCallCount !==
        OPT_0023_TRACKED_BUFFER_COUNT ||
      resourcesAfterBackendDestroy.everyBufferDestroyedExactlyOnce !== true ||
      resourcesAfterBackendDestroy.mapOverlapDetected !== false
    ) throw new Error("OPT-0023 backend cleanup left tracked GPU buffers live");

    const rawHashStarted = performance.now();
    prepared.raw.close();
    const rawHash = await prepared.raw.hashBounded();
    const rawHashWallMilliseconds = performance.now() - rawHashStarted;
    if (rawHash.byteLength !== OPT_0023_OUTPUT_BYTES) {
      throw new Error("OPT-0023 raw-file hash did not cover the exact output");
    }
    const progress = prepared.progress.finishTimed();
    const timing = Object.freeze({
      ...createTimingReceipt(
        fullStreamWallMilliseconds,
        windowWalls,
        profiles,
      ),
      excludedWalls: Object.freeze({
        postTimingValidationWallMilliseconds,
        backendDestroyWallMilliseconds,
        rawHashWallMilliseconds,
        finalPageJoinAndSerializationExcluded: true,
        finalPageSerializationWallMilliseconds: null,
        finalSerializationWallNotClaimedBecauseReceiptIsSelfReferential: true,
      }),
    });
    const attribution = aggregateProfiles(profiles);
    const cleanup = await cleanupPrepared(prepared);
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0023 observed a WebGPU runtime event");
    }
    const cleanupCompletedAtEpochMilliseconds =
      cleanup.completedAtEpochMilliseconds as number;
    return Object.freeze({
      schema: "ace-opt-0023-vae-c4500-production-family-profile-v2",
      experimentId: "OPT-0023",
      status: "passed",
      identity: prepared.identity,
      frozenAuthority: Object.freeze({
        observedManifestProvenance:
          prepared.package.loaded.manifest.provenance,
        opt0015IntegratedCommit: OPT_0015_INTEGRATED_COMMIT,
        opt0023RegistrationCommit: OPT_0023_REGISTRATION_COMMIT,
        opt0023RegistrationRecordSha256:
          OPT_0023_REGISTRATION_RECORD_SHA256,
        sourceAuthority: prepared.sourceAuthority,
      }),
      deterministicLatent: Object.freeze({
        generator: "xorshift32-13-17-5-high24-symmetric-f32-v1",
        seed: "0x00110512",
        latentFrames: OPT_0023_LATENT_FRAMES,
        channels: OPT_0023_LATENT_CHANNELS,
        elementCount: OPT_0023_LATENT_ELEMENTS,
        byteLength: OPT_0023_LATENT_BYTES,
        sha256: OPT_0023_LATENT_SHA256,
      }),
      package: packageReceipt(prepared.package),
      runtime: Object.freeze({
        runtimeProfileId: prepared.backend.runtimeProfileId,
        kernelSetId: prepared.backend.kernelSetId,
        kernelTopology: prepared.backend.kernelTopology,
        precisionMapSha256:
          ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
        quantaPerCommandBuffer:
          ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
        capabilities: prepared.context.capabilities,
      }),
      plan: prepared.topology,
      preparation: prepared.preparation,
      timing,
      attribution,
      perWindowProfiles: profiles,
      progress,
      output: Object.freeze({
        rawFormat: "stereo-interleaved-f32-le",
        sampleRateHz: prepared.plan.sampleRateHz,
        audioFrames: prepared.plan.outputAudioFrames,
        interleavedElements: stats.outputInterleavedElements,
        byteLength: rawHash.byteLength,
        sha256: rawHash.sha256,
        rawPeak: stats.peak,
        finiteSamples: stats.finiteSamples,
        windowsDecoded: stats.windowsDecoded,
        hashReadCount: rawHash.readCount,
        maximumHashChunkBytes: rawHash.maximumChunkBytes,
        rawHashWallMilliseconds,
        hashExcludedFromTiming: true,
        normalizationPerformed: false,
        wavEncodingPerformed: false,
      }),
      memory: Object.freeze({
        plan: prepared.memory,
        atTimedStart: prepared.preparation.memoryAtTimedStart,
        timed: timedResources,
        afterBackendDestroy: resourcesAfterBackendDestroy,
        boundedCpuBytes: prepared.memory.boundedCpuBytes,
        rawOutputFileBacked: true,
        fullSongWaveformMaterialized: false,
      }),
      protocol: Object.freeze({
        thermalGate,
        thermalClassification: "pending-external-artifact-join",
        sameTraceMustCoverWarmupGateRunValidationAndCleanup: true,
        unchangedThermalRetryPerformed: false,
        fullProductRunPerformed: false,
        extraCancellationRunPerformed: false,
      }),
      lifecycle: Object.freeze({
        timedStartedAtEpochMilliseconds,
        timedCompletedAtEpochMilliseconds,
        gateToTimedStartMilliseconds,
        backendDestroyStartedAtEpochMilliseconds,
        backendDestroyCompletedAtEpochMilliseconds,
        backendDestroyWallMilliseconds,
        backendDestroyIdempotenceChecked: true,
        rawHashCompletedBeforeRawFileAndDeviceCleanup: true,
        tailCleanup: cleanup,
        cleanupCompletedAtEpochMilliseconds,
      }),
    });
  } catch (error) {
    let cleanupError: unknown;
    try { await cleanupPrepared(prepared); } catch (value) {
      cleanupError = value;
    }
    if (cleanupError !== undefined) {
      throw new Opt0023FailureWithCleanup(error, [cleanupError]);
    }
    throw error;
  }
}

async function preparePackage(): Promise<PreparedPackage> {
  const manifestStarted = performance.now();
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(MANIFEST_PATH, self.location.href).href,
    expectedManifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
  });
  const manifestWallMilliseconds = performance.now() - manifestStarted;
  requireAceOpt0011Fp16VaePackageIdentity(loaded);
  if (
    loaded.manifest.provenance.referenceCommit !== ACE_SOURCE_REVISION ||
    loaded.manifest.provenance.aceSnapshot !== ACE_MAIN_MODEL_REVISION ||
    loaded.manifest.provenance.plannerSnapshot !== ACE_PLANNER_MODEL_REVISION
  ) throw new Error("OPT-0023 authenticated package provenance changed");
  const acquisitionManifest = createAceOpt0011VaeAcquisitionManifest(
    loaded.manifest,
  );
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = acquisitionManifest.files;
  const residentBytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  const parameterElements = tensors.reduce(
    (sum, tensor) => sum + tensor.byteLength / 2,
    0,
  );
  if (
    loaded.manifestSha256 !== ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES ||
    loaded.manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    new Set(tensors.map((tensor) => tensor.logicalTensor)).size !==
      ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT ||
    parameterElements !== ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS ||
    parameterElements !== 84_395_776 ||
    files.length !== 7 || shardNames.size !== 7 ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    files.some((file) => file.kind !== "weights") ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) throw new Error("OPT-0023 authenticated VAE package inventory changed");
  const cache = await AceOpfsModelCache.open();
  const acquiredStarted = performance.now();
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      if (progress.fileReceivedBytes === progress.fileBytes) {
        postProgress(
          `authenticated VAE file ${progress.fileIndex + 1}/${progress.fileCount}`,
        );
      }
    },
  });
  const acquiredWallMilliseconds = performance.now() - acquiredStarted;
  if (
    acquired.files.size !== files.length ||
    acquired.plan.runtimeBytes !== residentBytes
  ) throw new Error("OPT-0023 VAE package acquisition accounting changed");
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    acquiredFiles: acquired.files,
    acquisition: Object.freeze({
      manifestWallMilliseconds,
      acquiredWallMilliseconds,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
      runtimeBytes: acquired.plan.runtimeBytes,
    }),
  });
}

function createLatentFixture(): Readonly<{
  latent: Float32Array<ArrayBuffer>;
}> {
  const bytes = createAceOpt0011LatentFixture(OPT_0023_LATENT_FRAMES);
  const sha256 = hashBytes(bytes);
  if (
    bytes.byteLength !== OPT_0023_LATENT_BYTES ||
    sha256 !== OPT_0023_LATENT_SHA256 ||
    bytes.byteOffset !== 0 ||
    bytes.buffer.byteLength !== bytes.byteLength
  ) throw new Error("OPT-0023 deterministic latent fixture identity changed");
  const latent = new Float32Array(bytes.buffer as ArrayBuffer);
  let finiteCount = 0;
  let nonzeroCount = 0;
  for (const value of latent) {
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
  }
  if (
    latent.length !== OPT_0023_LATENT_ELEMENTS ||
    finiteCount !== latent.length || nonzeroCount === 0
  ) throw new Error("OPT-0023 deterministic latent contents changed");
  return Object.freeze({ latent });
}

function validateOpt0023Plan(plan: AceVaeChunkedDecodePlan): AceVaeChunkedDecodePlan {
  const shapes = plan.windows.map((window) => window.latentWindowFrames);
  const detachedBytes = plan.windows.reduce(
    (sum, window) => sum +
      window.decodedAudioFrames * plan.audioChannels * Float32Array.BYTES_PER_ELEMENT,
    0,
  );
  if (
    plan.latentFrames !== OPT_0023_LATENT_FRAMES ||
    plan.chunkFrames !== 512 || plan.overlapFrames !== 64 ||
    plan.strideFrames !== 384 || plan.hopLength !== 1_920 ||
    plan.sampleRateHz !== 48_000 || plan.audioChannels !== 2 ||
    plan.outputAudioFrames !== OPT_0023_OUTPUT_AUDIO_FRAMES ||
    plan.outputInterleavedElements !== OPT_0023_OUTPUT_ELEMENTS ||
    plan.outputFloat32Bytes !== OPT_0023_OUTPUT_BYTES ||
    plan.maximumWindowFrames !== 512 || plan.windows.length !== 12 ||
    JSON.stringify(shapes) !== JSON.stringify([448, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 340]) ||
    detachedBytes !== OPT_0023_DETACHED_WINDOW_BYTES
  ) throw new Error("OPT-0023 C4500 production chunk plan changed");
  for (const [index, window] of plan.windows.entries()) {
    if (
      window.index !== index ||
      window.outputStartAudioFrame !== window.coreStartLatentFrame * 1_920 ||
      window.outputAudioFrames !==
        (window.coreEndLatentFrame - window.coreStartLatentFrame) * 1_920
    ) throw new Error(`OPT-0023 window ${index} geometry changed`);
  }
  return plan;
}

export function validateOpt0023Topology(
  plan: AceVaeChunkedDecodePlan,
): Readonly<Record<string, unknown>> {
  const topology = planAceOpt0011Fp16VaeChunkDispatches(
    OPT_0023_LATENT_FRAMES,
    512,
    256,
  );
  const rows = topology.topologies.map((entry) => Object.freeze({
    inputFrames: entry.inputFrames,
    operationCount: entry.operationCount,
    graphQuantumCount: entry.graphQuantumCount,
    sequenceQuantumCount: entry.sequenceQuantumCount,
    decoderCommandBufferCount: entry.decoderCommandBufferCountAtBatch8,
    totalCommandBufferCount: entry.commandBufferCountAtBatch8,
    controlBytes: entry.dynamicControls.byteLength,
  }));
  if (
    topology.chunkPlan.latentFrames !== plan.latentFrames ||
    JSON.stringify(topology.uniqueWindowFrames) !== "[340,448,512]" ||
    JSON.stringify(rows) !== JSON.stringify([
      { inputFrames: 340, operationCount: 88, graphQuantumCount: 5241,
        sequenceQuantumCount: 5242, decoderCommandBufferCount: 656,
        totalCommandBufferCount: 657, controlBytes: 1341712 },
      { inputFrames: 448, operationCount: 88, graphQuantumCount: 6894,
        sequenceQuantumCount: 6895, decoderCommandBufferCount: 862,
        totalCommandBufferCount: 863, controlBytes: 1764880 },
      { inputFrames: 512, operationCount: 88, graphQuantumCount: 7854,
        sequenceQuantumCount: 7855, decoderCommandBufferCount: 982,
        totalCommandBufferCount: 983, controlBytes: 2010640 },
    ]) ||
    topology.uniqueDynamicControlBytes !== 5_117_232 ||
    topology.aggregateGraphQuantumCount !== 90_675 ||
    topology.aggregateSequenceQuantumCount !== 90_687 ||
    topology.aggregateCommandBufferCountAtBatch8 !==
      OPT_0023_TOTAL_COMMAND_BUFFERS ||
    topology.maximumFp16WorkspaceBytes !== REQUIRED_WORKSPACE_BYTES
  ) throw new Error("OPT-0023 exact C4500 dispatch topology changed");
  return Object.freeze({
    latentFrames: OPT_0023_LATENT_FRAMES,
    chunkFrames: 512,
    overlapFrames: 64,
    windowShapes: Object.freeze(plan.windows.map((window) =>
      window.latentWindowFrames
    )),
    uniqueShapes: Object.freeze(rows),
    aggregateGraphQuantumCount: topology.aggregateGraphQuantumCount,
    aggregateSequenceQuantumCount: topology.aggregateSequenceQuantumCount,
    decoderCommandBufferCount: OPT_0023_DECODER_COMMAND_BUFFERS,
    readbackCommandBufferCount: 12,
    totalCommandBufferCount: topology.aggregateCommandBufferCountAtBatch8,
    requestedInternalIdleMilliseconds: OPT_0023_INTERNAL_IDLE_MILLISECONDS,
    requestedBetweenWindowIdleMilliseconds:
      OPT_0023_BETWEEN_WINDOW_IDLE_MILLISECONDS,
  });
}

function validateBackendIdentity(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
): void {
  if (
    backend.runtimeProfileId !== OPT_0023_RUNTIME_PROFILE ||
    backend.kernelSetId !== OPT_0023_KERNEL_SET ||
    JSON.stringify(backend.kernelTopology) !== JSON.stringify(
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
    ) ||
    ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256 !==
      "4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85"
  ) throw new Error("OPT-0023 production backend identity changed");
}

function validateMemory(memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan): void {
  if (
    memory.residentWeightBytes !== 168_791_552 ||
    memory.stagingInputBufferBytes !== 131_072 ||
    memory.decoderInputBufferBytes !== 65_536 ||
    memory.workspaceBufferBytes !== REQUIRED_WORKSPACE_BYTES ||
    memory.workspaceBufferCount !== 3 ||
    memory.outputBufferBytes !== 7_864_320 ||
    memory.readbackBufferBytes !== 7_864_320 ||
    memory.controlBufferBytes !== 5_117_232 ||
    memory.accountedGpuBytes !== OPT_0023_ACCOUNTED_GPU_BYTES ||
    memory.latentSnapshotBytes !== OPT_0023_LATENT_BYTES ||
    memory.maximumReturnedWindowBytes !== 7_864_320 ||
    memory.boundedCpuBytes !== 9_016_320 ||
    memory.maximumWindowFrames !== 512 ||
    memory.quantaPerCommandBuffer !== 8
  ) throw new Error("OPT-0023 backend memory plan changed");
}

async function runWarmup(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  window: AceVaeDecodeWindow,
  profiles: ProfileCollector,
  progress: ProgressCollector,
): Promise<Readonly<{
  startedAtEpochMilliseconds: number;
  completedAtEpochMilliseconds: number;
  wallMilliseconds: number;
  output: Readonly<Record<string, unknown>>;
  progress: Readonly<Record<string, unknown>>;
}>> {
  const profileBeforeWarmup = profiles.count;
  progress.begin("warmup");
  const startedAtEpochMilliseconds = Date.now();
  const started = performance.now();
  const output = await backend.decodeWindow(window);
  const wallMilliseconds = performance.now() - started;
  const completedAtEpochMilliseconds = Date.now();
  const outputSummary = summarizeWarmupOutput(output, window);
  profiles.validateSingleAfter(profileBeforeWarmup, window);
  const progressSummary = progress.endWarmup(window);
  return Object.freeze({
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    wallMilliseconds,
    output: outputSummary,
    progress: progressSummary,
  });
}

function summarizeWarmupOutput(
  output: Float32Array,
  window: AceVaeDecodeWindow,
): Readonly<Record<string, unknown>> {
  let finiteCount = 0;
  let nonzeroCount = 0;
  for (const value of output) {
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
  }
  if (
    window.latentWindowFrames !== 512 || output.length !== 1_966_080 ||
    output.byteLength !== 7_864_320 || finiteCount !== output.length ||
    nonzeroCount === 0
  ) throw new Error("OPT-0023 C512 warmup output validation failed");
  return Object.freeze({
    inputFrames: 512,
    elementCount: output.length,
    byteLength: output.byteLength,
    finiteCount,
    nonzeroCount,
  });
}

function validateTimedResult(
  prepared: PreparedSession,
  stats: AceVaeRawStreamStats,
  windowWalls: readonly WindowWall[],
  profiles: readonly AceOpt0011Fp16VaeWindowFamilyProfile[],
  rawSize: number,
): void {
  if (
    stats.windowsDecoded !== 12 ||
    stats.outputInterleavedElements !== OPT_0023_OUTPUT_ELEMENTS ||
    stats.finiteSamples !== OPT_0023_OUTPUT_ELEMENTS ||
    stats.cooperativeIdleMs !== OPT_0023_BETWEEN_WINDOW_IDLE_MILLISECONDS ||
    !Number.isFinite(stats.peak) || stats.peak <= 0 ||
    windowWalls.length !== 12 ||
    windowWalls.some((wall, index) =>
      wall.windowIndex !== index || wall.inputFrames !==
        prepared.plan.windows[index]?.latentWindowFrames ||
      wall.outputElements !==
        prepared.plan.windows[index]!.decodedAudioFrames *
          prepared.plan.audioChannels ||
      !Number.isFinite(wall.wallMilliseconds) || wall.wallMilliseconds < 0
    ) ||
    profiles.length !== 12 ||
    rawSize !== OPT_0023_OUTPUT_BYTES
  ) throw new Error("OPT-0023 complete stream validation failed");
  prepared.progress.requireTimedComplete();
  prepared.audit.requireTimedUnchanged();
}

function validateWorkerThermalGate(
  gate: Opt0023ThermalGate,
  preparation: PreparedSession["preparation"],
): void {
  const nowEpochMilliseconds = Date.now();
  const finiteNumbers = [
    gate.traceStartedAtEpochMilliseconds,
    gate.gateStartedAtEpochMilliseconds,
    gate.gateCompletedAtEpochMilliseconds,
    gate.durationMilliseconds,
    gate.pollMilliseconds,
    gate.maximumPollGapMilliseconds,
    preparation.warmupStartedAtEpochMilliseconds,
    preparation.warmupCompletedAtEpochMilliseconds,
  ];
  const nonnegativeIntegers = [
    gate.traceStartObservationIndex,
    gate.gateStartObservationIndex,
    gate.gateCompletedObservationIndex,
    gate.traceObservationCountThroughGate,
    gate.observationCount,
    gate.nonNominalObservationCount,
    gate.missingObservationCount,
  ];
  if (
    gate.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    gate.command !== OPT_0023_THERMAL_COMMAND ||
    finiteNumbers.some((value) => !Number.isFinite(value)) ||
    nonnegativeIntegers.some((value) =>
      !Number.isSafeInteger(value) || value < 0
    ) ||
    preparation.warmupCompletedAtEpochMilliseconds <
      preparation.warmupStartedAtEpochMilliseconds ||
    gate.traceStartedAtEpochMilliseconds >
      preparation.warmupStartedAtEpochMilliseconds ||
    gate.gateStartedAtEpochMilliseconds <
      preparation.warmupCompletedAtEpochMilliseconds ||
    gate.gateCompletedAtEpochMilliseconds <
      gate.gateStartedAtEpochMilliseconds ||
    gate.durationMilliseconds !==
      gate.gateCompletedAtEpochMilliseconds -
        gate.gateStartedAtEpochMilliseconds ||
    gate.gateStartObservationIndex <= gate.traceStartObservationIndex ||
    gate.gateCompletedObservationIndex < gate.gateStartObservationIndex ||
    gate.observationCount !==
      gate.gateCompletedObservationIndex - gate.gateStartObservationIndex + 1 ||
    gate.traceObservationCountThroughGate !==
      gate.gateCompletedObservationIndex - gate.traceStartObservationIndex + 1 ||
    gate.durationMilliseconds < 30_000 ||
    gate.observationCount < Math.floor(gate.durationMilliseconds / 1_000) + 1 ||
    gate.nonNominalObservationCount !== 0 ||
    gate.missingObservationCount !== 0 ||
    gate.maximumPollGapMilliseconds < 0 ||
    gate.maximumPollGapMilliseconds > 1_250 ||
    gate.pollMilliseconds !== 1_000 ||
    nowEpochMilliseconds - gate.gateCompletedAtEpochMilliseconds < 0 ||
    nowEpochMilliseconds - gate.gateCompletedAtEpochMilliseconds >
      OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
  ) throw new Error("OPT-0023 worker rejected the external thermal gate");
}

class ProfileCollector {
  private readonly values: AceOpt0011Fp16VaeWindowFamilyProfile[] = [];
  private callbackFailure: unknown;
  private timedStart = -1;

  get count(): number { return this.values.length; }

  accept(profile: AceOpt0011Fp16VaeWindowFamilyProfile): void {
    try { this.values.push(profile); } catch (error) {
      this.callbackFailure ??= error;
    }
  }

  validateSingleAfter(before: number, window: AceVaeDecodeWindow): void {
    if (this.callbackFailure !== undefined) throw this.callbackFailure;
    if (this.values.length !== before + 1) {
      throw new Error(`OPT-0023 window ${window.index} lost its family profile`);
    }
    validateFamilyProfile(this.values[before]!, window);
  }

  armTimed(): void {
    if (this.values.length !== 1 || this.timedStart !== -1) {
      throw new Error("OPT-0023 family collector warmup boundary changed");
    }
    this.values.length = 0;
    this.timedStart = 0;
  }

  requireTimedCount(count: number): void {
    if (
      this.callbackFailure !== undefined || this.timedStart !== 0 ||
      this.values.length - this.timedStart !== count
    ) throw new Error("OPT-0023 timed family profile count changed");
  }

  finishTimed(
    windows: readonly AceVaeDecodeWindow[],
  ): readonly AceOpt0011Fp16VaeWindowFamilyProfile[] {
    this.requireTimedCount(12);
    const timed = this.values.slice(this.timedStart);
    if (windows.length !== timed.length) {
      throw new Error("OPT-0023 timed profile/window cardinality changed");
    }
    for (const [index, profile] of timed.entries()) {
      validateFamilyProfile(profile, windows[index]!);
    }
    return Object.freeze(timed);
  }
}

function validateFamilyProfile(
  profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  window: AceVaeDecodeWindow,
): void {
  const expected = expectedProfileCounts(window.latentWindowFrames);
  const actualFamilies = [
    familyCounts(profile, "k7-conv1d"),
    familyCounts(profile, "k1-conv1d"),
    familyCounts(profile, "conv-transpose1d"),
    familyCounts(profile, "snake"),
    familyCounts(profile, "add"),
  ];
  const familyWallSum = actualFamilies.reduce(
    (sum, value) => sum + value[2],
    0,
  );
  if (
    profile.windowIndex !== window.index ||
    profile.inputFrames !== window.latentWindowFrames ||
    profile.quantaPerCommandBuffer !== 8 ||
    profile.decoderBatchCount !== expected.decoderBatches ||
    profile.decoderQuantumCount !== expected.decoderQuanta ||
    profile.homogeneousBatchCount !== expected.homogeneousBatches ||
    profile.homogeneousQuantumCount !== expected.homogeneousQuanta ||
    profile.mixedBatchCount !== expected.mixedBatches ||
    profile.mixedQuantumCount !== expected.mixedQuanta ||
    JSON.stringify(actualFamilies.map((value) => value.slice(0, 2))) !==
      JSON.stringify(expected.families) ||
    [
      profile.decoderSubmitThroughDrainMs,
      profile.homogeneousSubmitThroughDrainMs,
      profile.mixedSubmitThroughDrainMs,
      ...actualFamilies.map((value) => value[2]),
    ].some((value) => !Number.isFinite(value) || value < 0) ||
    !nearlyEqual(
      familyWallSum,
      profile.homogeneousSubmitThroughDrainMs,
    ) ||
    !nearlyEqual(
      profile.homogeneousSubmitThroughDrainMs +
        profile.mixedSubmitThroughDrainMs,
      profile.decoderSubmitThroughDrainMs,
    )
  ) throw new Error(`OPT-0023 C${window.latentWindowFrames} family profile changed`);
}

function familyCounts(
  profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  family: AceOpt0011Fp16VaeProfileFamily,
): readonly [number, number, number] {
  const value = profile.families[family];
  return [value.batchCount, value.quantumCount, value.submitThroughDrainMs];
}

function expectedProfileCounts(inputFrames: number): Readonly<{
  decoderBatches: number;
  decoderQuanta: number;
  homogeneousBatches: number;
  homogeneousQuanta: number;
  mixedBatches: number;
  mixedQuanta: number;
  families: readonly (readonly [number, number])[];
}> {
  switch (inputFrames) {
    case 340: return Object.freeze({
      decoderBatches: 656, decoderQuanta: 5_242,
      homogeneousBatches: 582, homogeneousQuanta: 4_650,
      mixedBatches: 74, mixedQuanta: 592,
      families: Object.freeze([[327, 2610], [54, 432], [50, 400],
        [106, 848], [45, 360]] as const),
    });
    case 448: return Object.freeze({
      decoderBatches: 862, decoderQuanta: 6_895,
      homogeneousBatches: 793, homogeneousQuanta: 6_343,
      mixedBatches: 69, mixedQuanta: 552,
      families: Object.freeze([[436, 3487], [75, 600], [66, 528],
        [153, 1224], [63, 504]] as const),
    });
    case 512: return Object.freeze({
      decoderBatches: 982, decoderQuanta: 7_855,
      homogeneousBatches: 911, homogeneousQuanta: 7_287,
      mixedBatches: 71, mixedQuanta: 568,
      families: Object.freeze([[500, 3999], [89, 712], [78, 624],
        [170, 1360], [74, 592]] as const),
    });
    default: throw new Error(`OPT-0023 unexpected family shape C${inputFrames}`);
  }
}

class ProgressCollector {
  private mode: "warmup" | "timed" = "warmup";
  private total = 0;
  private decoder = 0;
  private readback = 0;
  private readonly byWindow = Array.from({ length: 12 }, () => 0);
  private readonly lastCompleted = Array.from({ length: 12 }, () => 0);
  private readonly lastTotal = Array.from({ length: 12 }, () => 0);
  private warmupSummary: Readonly<Record<string, unknown>> | undefined;
  private callbackFailure: string | undefined;
  private timedWindow = 0;
  private warmupCompleted = 0;

  accept(event: Readonly<{
    windowIndex: number;
    completedCommandBuffers: number;
    totalCommandBuffers: number;
    queueDrains: number;
    cooperativeIdleMs: number;
    completedDecoderQuanta: number;
    totalDecoderQuanta: number;
    stage: "decoder" | "readback";
  }>): void {
    const expectedWindow = this.mode === "warmup" ? 1 : this.timedWindow;
    const expectedTotal = this.mode === "warmup"
      ? 983
      : OPT_0023_PROGRESS_COMMAND_BUFFER_TOTALS[expectedWindow];
    const expectedQuanta = this.mode === "warmup"
      ? 7_855
      : OPT_0023_PROGRESS_DECODER_QUANTA[expectedWindow];
    const priorInWindow = this.mode === "warmup"
      ? this.warmupCompleted
      : (this.byWindow[expectedWindow] ?? -1);
    const expectedCompleted = priorInWindow + 1;
    const expectedStage = expectedCompleted === expectedTotal
      ? "readback"
      : "decoder";
    if (
      expectedTotal === undefined || expectedQuanta === undefined ||
      event.windowIndex !== expectedWindow ||
      event.completedCommandBuffers !== expectedCompleted ||
      event.totalCommandBuffers !== expectedTotal ||
      event.queueDrains !== expectedCompleted ||
      event.cooperativeIdleMs !== Math.min(
        expectedCompleted,
        expectedTotal - 1,
      ) ||
      event.totalDecoderQuanta !== expectedQuanta ||
      event.completedDecoderQuanta !== Math.min(
        expectedCompleted * 8,
        expectedQuanta,
      ) ||
      event.stage !== expectedStage
    ) this.callbackFailure ??= "OPT-0023 progress order/count changed";
    this.total += 1;
    if (event.stage === "decoder") this.decoder += 1;
    else this.readback += 1;
    if (this.mode === "warmup") {
      this.warmupCompleted += 1;
    } else if (this.byWindow[event.windowIndex] !== undefined) {
      this.byWindow[event.windowIndex]! += 1;
      this.lastCompleted[event.windowIndex] = event.completedCommandBuffers;
      this.lastTotal[event.windowIndex] = event.totalCommandBuffers;
      if (
        event.windowIndex === this.timedWindow &&
        event.completedCommandBuffers === expectedTotal
      ) this.timedWindow += 1;
    }
  }

  begin(expected: "warmup"): void {
    if (expected !== this.mode || this.total !== 0) {
      throw new Error("OPT-0023 warmup progress boundary changed");
    }
  }

  endWarmup(window: AceVaeDecodeWindow): Readonly<Record<string, unknown>> {
    if (
      window.latentWindowFrames !== 512 || this.total !== 983 ||
      this.decoder !== 982 || this.readback !== 1 ||
      this.warmupCompleted !== 983 || this.callbackFailure !== undefined
    ) throw new Error("OPT-0023 C512 warmup progress changed");
    this.warmupSummary = Object.freeze({
      eventCount: this.total,
      decoderCommandBufferCount: this.decoder,
      readbackCommandBufferCount: this.readback,
    });
    return this.warmupSummary;
  }

  armTimed(): void {
    if (this.warmupSummary === undefined) {
      throw new Error("OPT-0023 progress collector was not warmed");
    }
    this.mode = "timed";
    this.total = 0;
    this.decoder = 0;
    this.readback = 0;
  }

  requireTimedComplete(): void {
    const expected = [863, 983, 983, 983, 983, 983, 983, 983, 983, 983, 983, 657];
    if (
      this.mode !== "timed" || this.total !== OPT_0023_TOTAL_COMMAND_BUFFERS ||
      this.decoder !== OPT_0023_DECODER_COMMAND_BUFFERS || this.readback !== 12 ||
      JSON.stringify(this.byWindow) !== JSON.stringify(expected) ||
      this.lastCompleted.some((value, index) => value !== expected[index]) ||
      this.lastTotal.some((value, index) => value !== expected[index]) ||
      this.timedWindow !== 12 || this.callbackFailure !== undefined
    ) throw new Error("OPT-0023 timed progress accounting changed");
  }

  finishTimed(): Readonly<Record<string, unknown>> {
    this.requireTimedComplete();
    return Object.freeze({
      eventCount: this.total,
      decoderCommandBufferCount: this.decoder,
      readbackCommandBufferCount: this.readback,
      perWindowEventCounts: Object.freeze([...this.byWindow]),
      perCommandRecordsRetained: false,
      allProgressObservedInStrictOrder: true,
    });
  }
}

interface BufferRecord {
  readonly label: string;
  readonly size: number;
  readonly scope: "weights" | "backend";
  destroyed: boolean;
  destroyCalls: number;
  mapped: boolean;
  mapPending: boolean;
}

class DeviceResourceAudit {
  readonly device: GPUDevice;
  private scope: BufferRecord["scope"] = "weights";
  private readonly records = new Map<GPUBuffer, BufferRecord>();
  private maximumLiveCount = 0;
  private maximumLiveBytes = 0;
  private mapCount = 0;
  private unmapCount = 0;
  private timedMapStart = -1;
  private timedUnmapStart = -1;
  private timedMaximumLiveCount = 0;
  private timedMaximumLiveBytes = 0;
  private maximumMappedBufferCount = 0;
  private mapOverlapDetected = false;

  constructor(target: GPUDevice) {
    this.device = new Proxy(target, {
      get: (device, property) => {
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer =>
            this.track(device.createBuffer(descriptor), descriptor);
        }
        const value = Reflect.get(device, property, device) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(device)
          : value;
      },
    }) as GPUDevice;
  }

  setScope(scope: BufferRecord["scope"]): void { this.scope = scope; }

  armTimed(): void {
    this.requireLive(OPT_0023_TRACKED_BUFFER_COUNT, OPT_0023_ACCOUNTED_GPU_BYTES);
    if (this.mapCount !== 1 || this.unmapCount !== 1) {
      throw new Error("OPT-0023 warmup map/unmap accounting changed");
    }
    this.timedMapStart = this.mapCount;
    this.timedUnmapStart = this.unmapCount;
    this.timedMaximumLiveCount = this.liveCount();
    this.timedMaximumLiveBytes = this.liveBytes();
  }

  requireTimedUnchanged(): void {
    if (
      this.liveCount() !== OPT_0023_TRACKED_BUFFER_COUNT ||
      this.liveBytes() !== OPT_0023_ACCOUNTED_GPU_BYTES ||
      this.timedMaximumLiveCount !== OPT_0023_TRACKED_BUFFER_COUNT ||
      this.timedMaximumLiveBytes !== OPT_0023_ACCOUNTED_GPU_BYTES ||
      this.mapCount - this.timedMapStart !== OPT_0023_TIMED_MAP_COUNT ||
      this.unmapCount - this.timedUnmapStart !== OPT_0023_TIMED_MAP_COUNT ||
      [...this.records.values()].some((record) =>
        record.mapped || record.mapPending
      ) ||
      this.maximumMappedBufferCount !== 1 || this.mapOverlapDetected
    ) throw new Error("OPT-0023 timed GPU resource accounting changed");
  }

  requireLive(count: number, bytes: number): void {
    if (this.liveCount() !== count || this.liveBytes() !== bytes) {
      throw new Error("OPT-0023 tracked steady GPU allocation changed");
    }
  }

  snapshot(): Readonly<Record<string, unknown>> {
    const records = [...this.records.values()];
    return Object.freeze({
      createdBufferCount: records.length,
      createdBufferBytes: records.reduce((sum, record) => sum + record.size, 0),
      destroyedBufferCount: records.filter((record) => record.destroyed).length,
      liveBufferCount: this.liveCount(),
      liveBufferBytes: this.liveBytes(),
      maximumLiveBufferCount: this.maximumLiveCount,
      maximumLiveBufferBytes: this.maximumLiveBytes,
      mapCount: this.mapCount,
      unmapCount: this.unmapCount,
      mappedBufferCount: records.filter((record) => record.mapped).length,
      maximumMappedBufferCount: this.maximumMappedBufferCount,
      mapOverlapDetected: this.mapOverlapDetected,
      totalDestroyCallCount: records.reduce(
        (sum, record) => sum + record.destroyCalls,
        0,
      ),
      everyBufferDestroyedExactlyOnce: records.every((record) =>
        record.destroyed && record.destroyCalls === 1
      ),
      weights: scopeSummary(records, "weights"),
      backend: scopeSummary(records, "backend"),
    });
  }

  timedSnapshot(): Readonly<Record<string, unknown>> {
    this.requireTimedUnchanged();
    return Object.freeze({
      liveBufferCount: this.liveCount(),
      liveBufferBytes: this.liveBytes(),
      maximumLiveBufferCount: this.timedMaximumLiveCount,
      maximumLiveBufferBytes: this.timedMaximumLiveBytes,
      mapCount: this.mapCount - this.timedMapStart,
      unmapCount: this.unmapCount - this.timedUnmapStart,
      mappedBufferCount: [...this.records.values()]
        .filter((record) => record.mapped).length,
      maximumMappedBufferCount: this.maximumMappedBufferCount,
      mapOverlapDetected: this.mapOverlapDetected,
      newBufferCount: 0,
      newBufferBytes: 0,
    });
  }

  private track(buffer: GPUBuffer, descriptor: GPUBufferDescriptor): GPUBuffer {
    const record: BufferRecord = {
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      scope: this.scope,
      destroyed: false,
      destroyCalls: 0,
      mapped: false,
      mapPending: false,
    };
    this.records.set(buffer, record);
    this.updateMaximum();
    const destroy = buffer.destroy.bind(buffer);
    const mapAsync = buffer.mapAsync.bind(buffer);
    const unmap = buffer.unmap.bind(buffer);
    Object.defineProperties(buffer, {
      destroy: {
        configurable: true,
        value: () => {
          record.destroyCalls += 1;
          if (record.destroyed) return;
          record.destroyed = true;
          destroy();
        },
      },
      mapAsync: {
        configurable: true,
        value: async (...arguments_: Parameters<GPUBuffer["mapAsync"]>) => {
          this.mapCount += 1;
          if (record.mapped || record.mapPending) this.mapOverlapDetected = true;
          record.mapPending = true;
          try {
            await mapAsync(...arguments_);
            record.mapped = true;
            this.maximumMappedBufferCount = Math.max(
              this.maximumMappedBufferCount,
              [...this.records.values()].filter((value) => value.mapped).length,
            );
          } finally {
            record.mapPending = false;
          }
        },
      },
      unmap: {
        configurable: true,
        value: () => {
          this.unmapCount += 1;
          record.mapped = false;
          unmap();
        },
      },
    });
    return buffer;
  }

  private updateMaximum(): void {
    this.maximumLiveCount = Math.max(this.maximumLiveCount, this.liveCount());
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes());
    if (this.timedMapStart >= 0) {
      this.timedMaximumLiveCount = Math.max(
        this.timedMaximumLiveCount,
        this.liveCount(),
      );
      this.timedMaximumLiveBytes = Math.max(
        this.timedMaximumLiveBytes,
        this.liveBytes(),
      );
    }
  }

  private liveCount(): number {
    return [...this.records.values()].filter((record) => !record.destroyed).length;
  }

  private liveBytes(): number {
    return [...this.records.values()].filter((record) => !record.destroyed)
      .reduce((sum, record) => sum + record.size, 0);
  }
}

function scopeSummary(
  records: readonly BufferRecord[],
  scope: BufferRecord["scope"],
): Readonly<Record<string, number>> {
  const selected = records.filter((record) => record.scope === scope);
  return Object.freeze({
    createdBufferCount: selected.length,
    createdBufferBytes: selected.reduce((sum, record) => sum + record.size, 0),
    destroyedBufferCount: selected.filter((record) => record.destroyed).length,
    liveBufferCount: selected.filter((record) => !record.destroyed).length,
  });
}

export interface Opt0023RawSetupDependencies {
  readonly getRoot: () => Promise<FileSystemDirectoryHandle>;
  readonly randomUuid: () => string;
}

export class Opt0023RawFile {
  readonly sink: AceVaeRawF32FileSink;
  private closed = false;
  private removed = false;

  private constructor(
    private readonly root: FileSystemDirectoryHandle,
    private readonly directoryName: string,
    private readonly fileHandle: FileSystemFileHandle,
    private readonly access: FileSystemSyncAccessHandle,
    plan: AceVaeChunkedDecodePlan,
  ) {
    this.sink = new AceVaeRawF32FileSink(access, plan);
  }

  static async create(
    plan: AceVaeChunkedDecodePlan,
    dependencies: Opt0023RawSetupDependencies = {
      getRoot: async () => await navigator.storage.getDirectory(),
      randomUuid: () => crypto.randomUUID(),
    },
  ): Promise<Opt0023RawFile> {
    const root = await dependencies.getRoot();
    const directoryName = `ace-opt-0023-${dependencies.randomUuid()}`;
    let directoryCreated = false;
    let access: FileSystemSyncAccessHandle | undefined;
    try {
      const directory = await root.getDirectoryHandle(directoryName, {
        create: true,
      });
      directoryCreated = true;
      const fileHandle = await directory.getFileHandle("raw.f32.partial", {
        create: true,
      });
      access = await fileHandle.createSyncAccessHandle();
      return new Opt0023RawFile(
        root,
        directoryName,
        fileHandle,
        access,
        plan,
      );
    } catch (primary) {
      const cleanupErrors: unknown[] = [];
      if (access !== undefined) {
        try { access.close(); } catch (error) { cleanupErrors.push(error); }
      }
      if (directoryCreated) {
        try {
          await root.removeEntry(directoryName, { recursive: true });
        } catch (error) { cleanupErrors.push(error); }
      }
      if (cleanupErrors.length !== 0) {
        throw new Opt0023FailureWithCleanup(primary, cleanupErrors);
      }
      throw primary;
    }
  }

  getSize(): number { return this.access.getSize(); }

  close(): void {
    if (this.closed) return;
    this.access.close();
    this.closed = true;
  }

  async hashBounded(): Promise<Readonly<{
    byteLength: number;
    sha256: string;
    readCount: number;
    maximumChunkBytes: number;
  }>> {
    if (!this.closed) {
      throw new Error("OPT-0023 raw sync handle must close before hashing");
    }
    const file = await this.fileHandle.getFile();
    if (file.size !== OPT_0023_OUTPUT_BYTES) {
      throw new Error("OPT-0023 raw OPFS File snapshot has the wrong size");
    }
    const hash = new AceIncrementalSha256();
    let byteLength = 0;
    let readCount = 0;
    let maximumChunkBytes = 0;
    for (
      let fileOffset = 0;
      fileOffset < file.size;
      fileOffset += RAW_HASH_MAXIMUM_CHUNK_BYTES
    ) {
      const slice = file.slice(
        fileOffset,
        Math.min(fileOffset + RAW_HASH_MAXIMUM_CHUNK_BYTES, file.size),
      );
      const chunk = new Uint8Array(await slice.arrayBuffer());
      if (
        chunk.byteLength === 0 ||
        chunk.byteLength !== slice.size ||
        chunk.byteLength > RAW_HASH_MAXIMUM_CHUNK_BYTES
      ) {
        throw new Error("OPT-0023 raw hash returned an invalid bounded slice");
      }
      hash.update(chunk);
      byteLength += chunk.byteLength;
      readCount += 1;
      maximumChunkBytes = Math.max(maximumChunkBytes, chunk.byteLength);
    }
    if (byteLength !== OPT_0023_OUTPUT_BYTES) {
      throw new Error("OPT-0023 raw hash read the wrong byte count");
    }
    return Object.freeze({
      byteLength,
      sha256: hash.digestHex(),
      readCount,
      maximumChunkBytes,
    });
  }

  async remove(): Promise<void> {
    if (this.removed) return;
    const errors: unknown[] = [];
    try { this.close(); } catch (error) { errors.push(error); }
    try {
      await this.root.removeEntry(this.directoryName, { recursive: true });
      this.removed = true;
    } catch (error) { errors.push(error); }
    if (errors.length !== 0) {
      throw new AggregateError(errors, "OPT-0023 raw temporary-file cleanup failed");
    }
  }
}

export interface Opt0023CleanupOwners {
  readonly backend?: Readonly<{ destroy(): Promise<void> }>;
  readonly raw?: Readonly<{ close(): void; remove(): Promise<void> }>;
  readonly context?: Readonly<{
    destroy(): void;
    readonly lost: Promise<Readonly<{ reason: string }>>;
  }>;
}

export interface Opt0023InitializationOwners extends Opt0023CleanupOwners {
  readonly phase?: Readonly<{ destroy(): void }>;
}

export async function settleOpt0023InitializationOwners(
  owners: Opt0023InitializationOwners,
): Promise<Readonly<Record<string, unknown>>> {
  const errors: unknown[] = [];
  if (owners.phase !== undefined) {
    try { owners.phase.destroy(); } catch (error) { errors.push(error); }
  }
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  try {
    cleanup = await settleOpt0023CleanupOwners(owners);
  } catch (error) { errors.push(error); }
  if (errors.length !== 0) {
    throw new AggregateError(errors, "OPT-0023 initialization cleanup failed");
  }
  return Object.freeze({
    ...cleanup,
    phaseDestroyed: owners.phase !== undefined,
  });
}

export async function settleOpt0023CleanupOwners(
  owners: Opt0023CleanupOwners,
): Promise<Readonly<Record<string, unknown>>> {
  const startedAtEpochMilliseconds = Date.now();
  const started = performance.now();
  const errors: unknown[] = [];
  let lostReason: string | undefined;
  if (owners.backend !== undefined) {
    try {
      await owners.backend.destroy();
      await owners.backend.destroy();
    } catch (error) { errors.push(error); }
  }
  if (owners.raw !== undefined) {
    try { owners.raw.close(); } catch (error) { errors.push(error); }
    try { await owners.raw.remove(); } catch (error) { errors.push(error); }
  }
  if (owners.context !== undefined) {
    try { owners.context.destroy(); } catch (error) { errors.push(error); }
    try { lostReason = (await owners.context.lost).reason; } catch (error) {
      errors.push(error);
    }
  }
  if (errors.length !== 0) throw new AggregateError(errors, "OPT-0023 cleanup failed");
  if (owners.context !== undefined && lostReason !== "destroyed") {
    throw new Error(`OPT-0023 device loss reason was ${String(lostReason)}`);
  }
  return Object.freeze({
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds: Date.now(),
    wallMilliseconds: performance.now() - started,
    backendDestroyCalledTwice: owners.backend !== undefined,
    rawTemporaryEntryRemoved: owners.raw !== undefined,
    deviceDestroyed: owners.context !== undefined,
    deviceLostReason: lostReason ?? null,
  });
}

async function cleanupPrepared(
  prepared: PreparedSession,
): Promise<Readonly<Record<string, unknown>>> {
  prepared.cleanupPromise ??= (async () => {
    const summary = await settleOpt0023CleanupOwners({
      ...(prepared.backendReleased ? {} : { backend: prepared.backend }),
      ...(prepared.raw === undefined ? {} : { raw: prepared.raw }),
      context: prepared.context,
    });
    const resources = prepared.audit.snapshot();
    if (!resourceCleanupIsBalanced(resources)) {
      throw new Error("OPT-0023 cleanup did not balance tracked GPU resources");
    }
    return Object.freeze({
      ...summary,
      resources,
      runtimeEventCount: prepared.runtimeEvents.length,
    });
  })();
  return await prepared.cleanupPromise;
}

function resourceCleanupIsBalanced(
  resources: Readonly<Record<string, unknown>>,
): boolean {
  return resources.liveBufferCount === 0 &&
    resources.liveBufferBytes === 0 &&
    resources.createdBufferCount === resources.destroyedBufferCount &&
    resources.mapCount === resources.unmapCount &&
    resources.mappedBufferCount === 0 &&
    resources.mapOverlapDetected === false &&
    resources.totalDestroyCallCount === resources.createdBufferCount &&
    resources.everyBufferDestroyedExactlyOnce === true;
}

export function aggregateProfiles(
  profiles: readonly AceOpt0011Fp16VaeWindowFamilyProfile[],
): Readonly<Record<string, unknown>> {
  if (profiles.length !== 12) {
    throw new Error("OPT-0023 aggregate requires twelve window profiles");
  }
  const families = Object.fromEntries([
    "k7-conv1d", "k1-conv1d", "conv-transpose1d", "snake", "add",
  ].map((family) => {
    const typedFamily = family as AceOpt0011Fp16VaeProfileFamily;
    return [family, Object.freeze({
      batchCount: profiles.reduce(
        (sum, profile) => sum + profile.families[typedFamily].batchCount,
        0,
      ),
      quantumCount: profiles.reduce(
        (sum, profile) => sum + profile.families[typedFamily].quantumCount,
        0,
      ),
      submitThroughDrainMilliseconds: profiles.reduce(
        (sum, profile) =>
          sum + profile.families[typedFamily].submitThroughDrainMs,
        0,
      ),
    })];
  })) as Record<string, Readonly<Record<string, number>>>;
  const decoderSubmitThroughDrainMilliseconds = profiles.reduce(
    (sum, profile) => sum + profile.decoderSubmitThroughDrainMs,
    0,
  );
  const mixedBatchCount = profiles.reduce(
    (sum, profile) => sum + profile.mixedBatchCount,
    0,
  );
  const mixedQuantumCount = profiles.reduce(
    (sum, profile) => sum + profile.mixedQuantumCount,
    0,
  );
  const mixedSubmitThroughDrainMilliseconds = profiles.reduce(
    (sum, profile) => sum + profile.mixedSubmitThroughDrainMs,
    0,
  );
  const expectedFamily = [[5763, 46087], [1019, 8152], [896, 7168],
    [1959, 15672], [848, 6784]];
  const homogeneousSubmitThroughDrainMilliseconds = Object.values(families)
    .reduce(
      (sum, value) => sum + value["submitThroughDrainMilliseconds"]!,
      0,
    );
  if (
    JSON.stringify(Object.values(families).map((value) => [
      value.batchCount, value.quantumCount,
    ])) !== JSON.stringify(expectedFamily) ||
    mixedBatchCount !== 853 || mixedQuantumCount !== 6_824 ||
    !nearlyEqual(
      homogeneousSubmitThroughDrainMilliseconds +
        mixedSubmitThroughDrainMilliseconds,
      decoderSubmitThroughDrainMilliseconds,
    )
  ) throw new Error("OPT-0023 aggregate family accounting changed");
  return Object.freeze({
    decoderBatchCount: OPT_0023_DECODER_COMMAND_BUFFERS,
    decoderQuantumCount: 90_687,
    decoderSubmitThroughDrainMilliseconds,
    homogeneousBatchCount: 10_485,
    homogeneousQuantumCount: 83_863,
    homogeneousSubmitThroughDrainMilliseconds,
    mixedBatchCount,
    mixedQuantumCount,
    mixedSubmitThroughDrainMilliseconds,
    families: Object.freeze(families),
    mixedWallSplitOrEstimated: false,
    readbackIncludedInFamilyWall: false,
  });
}

function createTimingReceipt(
  fullStreamWallMilliseconds: number,
  windowWalls: readonly WindowWall[],
  profiles: readonly AceOpt0011Fp16VaeWindowFamilyProfile[],
): Readonly<Record<string, unknown>> {
  const summedDecodeWallMilliseconds = windowWalls.reduce(
    (sum, wall) => sum + wall.wallMilliseconds,
    0,
  );
  const decoderSubmitThroughDrainMilliseconds = profiles.reduce(
    (sum, profile) => sum + profile.decoderSubmitThroughDrainMs,
    0,
  );
  const withinDecodeNonfamilyResidualMilliseconds =
    summedDecodeWallMilliseconds - decoderSubmitThroughDrainMilliseconds;
  const outsideDecodeStreamResidualMilliseconds =
    fullStreamWallMilliseconds - summedDecodeWallMilliseconds;
  if (
    [fullStreamWallMilliseconds, summedDecodeWallMilliseconds,
      decoderSubmitThroughDrainMilliseconds,
      withinDecodeNonfamilyResidualMilliseconds,
      outsideDecodeStreamResidualMilliseconds]
      .some((value) => !Number.isFinite(value) || value < 0)
  ) throw new Error("OPT-0023 timing residual accounting is invalid");
  return Object.freeze({
    fullStreamWallMilliseconds,
    summedDecodeWallMilliseconds,
    decoderSubmitThroughDrainMilliseconds,
    withinDecodeNonfamilyResidualMilliseconds,
    outsideDecodeStreamResidualMilliseconds,
    windowWalls: Object.freeze(windowWalls),
    authoritativeInterval:
      "immediately-before-streamAceVaeRawChunks through rawSink.finish/flush",
    familyInterval: "decoder-command-buffer-submit-through-matching-drain",
    noReadbackOrResidualProration: true,
  });
}

function packageReceipt(prepared: PreparedPackage): Readonly<Record<string, unknown>> {
  return Object.freeze({
    manifestPath: MANIFEST_PATH,
    manifestSha256: prepared.loaded.manifestSha256,
    manifestByteLength: prepared.loaded.manifestByteLength,
    profile: prepared.loaded.manifest.profile,
    converterRevision: prepared.loaded.manifest.provenance.converterRevision,
    provenance: prepared.loaded.manifest.provenance,
    tensorRecordCount: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    logicalTensorCount: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    parameterElements: 84_395_776,
    residentBytes: ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    files: Object.freeze(prepared.files.map((file) => Object.freeze({
      name: file.name,
      byteLength: file.byteLength,
      sha256: file.sha256,
    }))),
    acquisition: prepared.acquisition,
  });
}

function requireDevice(context: AceWebGpuDeviceContext): void {
  const capabilities = context.capabilities;
  if (
    capabilities.adapterInfo.subgroupMinSize !== 32 ||
    capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    context.device.limits.maxBufferSize < REQUIRED_WORKSPACE_BYTES ||
    context.device.limits.maxStorageBufferBindingSize < REQUIRED_WORKSPACE_BYTES
  ) throw new Error("OPT-0023 requires shader-f16 and exact fixed32 subgroups");
}

function createSourceAuthority(): Readonly<Record<string, unknown>> {
  const files = Object.entries(SOURCE_TEXT).map(([path, source]) =>
    Object.freeze({
      path,
      byteLength: new TextEncoder().encode(source).byteLength,
      sha256: hashBytes(new TextEncoder().encode(source)),
    })
  ).sort((left, right) => left.path.localeCompare(right.path));
  const byPath = Object.fromEntries(files.map((file) => [file.path, file.sha256]));
  for (const [path, expected] of Object.entries(REGISTERED_SOURCE_SHA256)) {
    if (byPath[path] !== expected) {
      throw new Error(`OPT-0023 registered source identity changed: ${path}`);
    }
  }
  const canonical = JSON.stringify(files);
  return Object.freeze({
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.byteLength, 0),
    files: Object.freeze(files),
    aggregateSha256: hashBytes(new TextEncoder().encode(canonical)),
  });
}

function hashBytes(bytes: Uint8Array): string {
  return new AceIncrementalSha256().update(bytes).digestHex();
}

function nearlyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-6;
}

function validateRunIdentity(value: unknown): Opt0023RunIdentity {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("OPT-0023 run identity is missing");
  }
  return parseOpt0023RunIdentity(new URLSearchParams(
    Object.entries(value).map(([name, field]) => [name, String(field)]),
  ));
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

async function fail(error: unknown, active?: PreparedSession): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  let cleanupError: unknown;
  const target = active ?? session;
  session = undefined;
  if (target !== undefined) {
    try { await cleanupPrepared(target); } catch (value) { cleanupError = value; }
  }
  if (error instanceof Opt0023FailureWithCleanup) {
    cleanupError = cleanupError ?? error.cleanupErrors;
    error = error.primary;
  }
  self.postMessage({
    type: "failed",
    error: serializeOpt0023Failure(error, cleanupError),
  });
}

class Opt0023FailureWithCleanup extends Error {
  constructor(
    readonly primary: unknown,
    readonly cleanupErrors: readonly unknown[],
  ) {
    super("OPT-0023 primary operation and cleanup both failed");
    this.name = "Opt0023FailureWithCleanup";
  }
}
