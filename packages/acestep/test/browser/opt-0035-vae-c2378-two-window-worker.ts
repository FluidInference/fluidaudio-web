/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import { createAceOpt0011LatentFixture } from
  "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import {
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  AceOpt0011Fp16VaeChunkGpuBackend,
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
  planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory,
  type AceOpt0011Fp16VaeBackendMaximumWindowFrames,
  type AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  deriveAceVaePostprocessPlan,
  planAceVaeChunkedDecode,
  streamAceVaeRawChunks,
  type AceVaeChunkedDecodePlan,
  type AceVaeRawStreamStats,
} from "../../src/webgpu/vae-chunks.js";
import {
  AceVaeRawF32FileSink,
  writeNormalizedAceVaeFloat32WavCooperatively,
} from "../../src/webgpu/vae-wav.js";

export const OPT_0035_SCHEMA =
  "ace-opt-0035-vae-c2378-two-window-abba-v1" as const;
export const OPT_0035_LATENT_FRAMES = 4_500 as const;
export const OPT_0035_LATENT_CHANNELS = 64 as const;
export const OPT_0035_LATENT_ELEMENTS = 288_000 as const;
export const OPT_0035_LATENT_BYTES = 1_152_000 as const;
export const OPT_0035_LATENT_SHA256 =
  "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971" as const;
export const OPT_0035_OUTPUT_ELEMENTS = 17_280_000 as const;
export const OPT_0035_OUTPUT_BYTES = 69_120_000 as const;
export const OPT_0035_CONTROL_CHUNK_FRAMES = 512 as const;
export const OPT_0035_CANDIDATE_CHUNK_FRAMES =
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES;
export const OPT_0035_OVERLAP_FRAMES = 64 as const;
export const OPT_0035_SEAM_RADIUS_LATENT_FRAMES = 64 as const;
export const OPT_0035_TIMED_ORDER = Object.freeze([
  "c512",
  "c2378",
  "c2378",
  "c512",
] as const);
export const OPT_0035_SPEEDUP_GATE = 1.15 as const;

const EXPERIMENT_ID = "OPT-0035" as const;
const MANIFEST_PATH = "/model/files-fp16-vae-experimental/manifest.json";
const RAW_BLOCK_BYTES = 1024 * 1024;
const NRMSE_LIMIT = 0.003;
const SNR_MINIMUM_DB = 50;
const PEARSON_MINIMUM = 0.9999;

export type Opt0035Arm = "c512" | "c2378";
export type Opt0035QuantaPerCommandBuffer = 8 | 64;

export interface Opt0035CoverageReceipt {
  readonly control: AceVaeChunkedDecodePlan;
  readonly candidate: AceVaeChunkedDecodePlan;
  readonly controlDecodedLatentFrames: 5_908;
  readonly candidateDecodedLatentFrames: 4_628;
  readonly decodedLatentFrameReduction: 1_280;
  readonly decodedLatentFrameReductionRatio: number;
  readonly controlSeams: readonly number[];
  readonly candidateSeams: readonly [2_250];
}

export interface Opt0035WaveformMetric {
  readonly count: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly pearson: number;
  readonly maximumAbsoluteError: number;
  readonly controlPeak: number;
  readonly candidatePeak: number;
  readonly finite: boolean;
  readonly passed: boolean;
}

export interface Opt0035WaveformMetrics {
  readonly joint: Opt0035WaveformMetric;
  readonly left: Opt0035WaveformMetric;
  readonly right: Opt0035WaveformMetric;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly residentBytes: number;
}

interface ArmExecution {
  readonly arm: Opt0035Arm;
  readonly wallMs: number;
  readonly stats: AceVaeRawStreamStats;
  readonly memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly raw: Opt0035RawArtifact;
  readonly sha256: string;
  readonly resources: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly context: AceWebGpuDeviceContext;
  readonly pkg: PreparedPackage;
  readonly files: ReadonlyMap<string, File>;
  readonly fixture: Float32Array<ArrayBuffer>;
  readonly quantaPerCommandBuffer: Opt0035QuantaPerCommandBuffer;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly activeGuard: ActiveArmGuard;
  readonly resourceAudit: Opt0035DeviceResourceAudit;
  readonly listeningArtifact: Opt0035ListeningArtifact;
  readonly correctness: Readonly<Record<string, unknown>>;
  destroy(): Promise<Readonly<Record<string, unknown>>>;
}

export type Opt0035WorkerCommand =
  | Readonly<{
      readonly type: "prepare";
      readonly quantaPerCommandBuffer: Opt0035QuantaPerCommandBuffer;
    }>
  | Readonly<{ readonly type: "run" | "dispose" }>;

interface Opt0035WorkerEvent {
  readonly type: "progress" | "prepared" | "result" | "disposed" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
  readonly audio?: Blob;
}

/** Pure gate used by static tests and the worker before any device request. */
export function planOpt0035Coverage(): Opt0035CoverageReceipt {
  const control = planAceVaeChunkedDecode(OPT_0035_LATENT_FRAMES, {
    chunkFrames: OPT_0035_CONTROL_CHUNK_FRAMES,
    overlapFrames: OPT_0035_OVERLAP_FRAMES,
  });
  const candidate = planAceVaeChunkedDecode(OPT_0035_LATENT_FRAMES, {
    chunkFrames: OPT_0035_CANDIDATE_CHUNK_FRAMES,
    overlapFrames: OPT_0035_OVERLAP_FRAMES,
  });
  const controlDecodedLatentFrames = control.windows.reduce(
    (sum, window) => sum + window.latentWindowFrames,
    0,
  );
  const candidateDecodedLatentFrames = candidate.windows.reduce(
    (sum, window) => sum + window.latentWindowFrames,
    0,
  );
  const controlSeams = Object.freeze(
    control.windows.slice(1).map((window) => window.coreStartLatentFrame),
  );
  const candidateSeams = Object.freeze(
    candidate.windows.slice(1).map((window) => window.coreStartLatentFrame),
  );
  if (
    controlDecodedLatentFrames !== 5_908 ||
    candidateDecodedLatentFrames !== 4_628 ||
    control.windows.length !== 12 ||
    candidate.windows.length !== 2 ||
    candidate.maximumWindowFrames !== 2_314 ||
    JSON.stringify(candidate.windows.map((window) => ({
      window: [window.windowStartLatentFrame, window.windowEndLatentFrame],
      core: [window.coreStartLatentFrame, window.coreEndLatentFrame],
      prefix: window.discardPrefixLatentFrames,
      suffix: window.discardSuffixLatentFrames,
    }))) !== JSON.stringify([
      { window: [0, 2_314], core: [0, 2_250], prefix: 0, suffix: 64 },
      { window: [2_186, 4_500], core: [2_250, 4_500], prefix: 64, suffix: 0 },
    ]) ||
    JSON.stringify(candidateSeams) !== JSON.stringify([2_250]) ||
    [...control.windows, ...candidate.windows].some((window, index, all) =>
      window.outputAudioFrames <= 0 ||
      (index > 0 && index < control.windows.length &&
        all[index - 1]!.coreEndLatentFrame !== window.coreStartLatentFrame)
    )
  ) {
    throw new Error("OPT-0035 exact C4500 coverage changed");
  }
  return Object.freeze({
    control,
    candidate,
    controlDecodedLatentFrames: 5_908,
    candidateDecodedLatentFrames: 4_628,
    decodedLatentFrameReduction: 1_280,
    decodedLatentFrameReductionRatio: 1_280 / 5_908,
    controlSeams,
    candidateSeams: candidateSeams as readonly [2_250],
  });
}

/** Small-array oracle for the same metric accumulator used by OPFS comparison. */
export function compareOpt0035Waveforms(
  control: Float32Array,
  candidate: Float32Array,
): Opt0035WaveformMetrics {
  if (control.length !== candidate.length || control.length % 2 !== 0) {
    throw new RangeError("OPT-0035 comparison requires equal stereo arrays");
  }
  const accumulator = new StereoMetricAccumulator();
  accumulator.add(control, candidate);
  return accumulator.finish();
}

export function resolveOpt0035QuantaPerCommandBuffer(
  value: number,
): Opt0035QuantaPerCommandBuffer {
  if (value !== 8 && value !== 64) {
    throw new RangeError("OPT-0035 command-buffer batch must be 8 or 64");
  }
  return value;
}

class MetricAccumulator {
  private count = 0;
  private sumControl = 0;
  private sumCandidate = 0;
  private sumControlSquared = 0;
  private sumCandidateSquared = 0;
  private sumProduct = 0;
  private sumSquaredError = 0;
  private maximumAbsoluteError = 0;
  private controlPeak = 0;
  private candidatePeak = 0;
  private finite = true;

  add(control: number, candidate: number): void {
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) {
      this.finite = false;
      return;
    }
    const error = candidate - control;
    this.count += 1;
    this.sumControl += control;
    this.sumCandidate += candidate;
    this.sumControlSquared += control * control;
    this.sumCandidateSquared += candidate * candidate;
    this.sumProduct += control * candidate;
    this.sumSquaredError += error * error;
    this.maximumAbsoluteError = Math.max(
      this.maximumAbsoluteError,
      Math.abs(error),
    );
    this.controlPeak = Math.max(this.controlPeak, Math.abs(control));
    this.candidatePeak = Math.max(this.candidatePeak, Math.abs(candidate));
  }

  finish(): Opt0035WaveformMetric {
    if (this.count === 0) throw new Error("OPT-0035 metric is empty");
    const controlMean = this.sumControl / this.count;
    const candidateMean = this.sumCandidate / this.count;
    const controlMeanSquared = this.sumControlSquared / this.count;
    const candidateMeanSquared = this.sumCandidateSquared / this.count;
    const rmse = Math.sqrt(this.sumSquaredError / this.count);
    const controlRms = Math.sqrt(controlMeanSquared);
    const nrmse = rmse / Math.max(controlRms, 1e-30);
    const snrDb = rmse === 0
      ? Number.POSITIVE_INFINITY
      : 20 * Math.log10(Math.max(controlRms, 1e-30) / rmse);
    const covariance = this.sumProduct / this.count -
      controlMean * candidateMean;
    const denominator = Math.sqrt(
      Math.max(0, controlMeanSquared - controlMean * controlMean) *
        Math.max(0, candidateMeanSquared - candidateMean * candidateMean),
    );
    const pearson = denominator === 0
      ? (rmse === 0 ? 1 : 0)
      : covariance / denominator;
    return Object.freeze({
      count: this.count,
      nrmse,
      snrDb,
      pearson,
      maximumAbsoluteError: this.maximumAbsoluteError,
      controlPeak: this.controlPeak,
      candidatePeak: this.candidatePeak,
      finite: this.finite,
      passed: this.finite && nrmse <= NRMSE_LIMIT &&
        snrDb >= SNR_MINIMUM_DB && pearson >= PEARSON_MINIMUM,
    });
  }
}

class StereoMetricAccumulator {
  private readonly joint = new MetricAccumulator();
  private readonly left = new MetricAccumulator();
  private readonly right = new MetricAccumulator();

  add(control: Float32Array, candidate: Float32Array): void {
    if (control.length !== candidate.length || control.length % 2 !== 0) {
      throw new RangeError("OPT-0035 metric block lost stereo alignment");
    }
    for (let index = 0; index < control.length; index += 2) {
      const controlLeft = control[index]!;
      const controlRight = control[index + 1]!;
      const candidateLeft = candidate[index]!;
      const candidateRight = candidate[index + 1]!;
      this.left.add(controlLeft, candidateLeft);
      this.right.add(controlRight, candidateRight);
      this.joint.add(controlLeft, candidateLeft);
      this.joint.add(controlRight, candidateRight);
    }
  }

  finish(): Opt0035WaveformMetrics {
    return Object.freeze({
      joint: this.joint.finish(),
      left: this.left.finish(),
      right: this.right.finish(),
    });
  }
}

class ActiveArmGuard {
  private active = 0;
  private maximum = 0;

  acquire(): () => void {
    this.active += 1;
    this.maximum = Math.max(this.maximum, this.active);
    if (this.active !== 1) {
      throw new Error("OPT-0035 attempted simultaneous VAE arm ownership");
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active -= 1;
    };
  }

  snapshot(): Readonly<Record<string, number | boolean>> {
    return Object.freeze({
      activeArmOwners: this.active,
      maximumSimultaneousArmOwners: this.maximum,
      sequentialOwnershipPassed: this.maximum === 1 && this.active === 0,
    });
  }
}

interface Opt0035BufferRecord {
  readonly arm: Opt0035Arm;
  readonly label: string;
  readonly size: number;
  destroyed: boolean;
  destroyCalls: number;
}

/** Tracks real WebGPU allocation high-water and proves serial arm cleanup. */
class Opt0035DeviceResourceAudit {
  readonly device: GPUDevice;
  private readonly records: Opt0035BufferRecord[] = [];
  private activeArm: Opt0035Arm | undefined;
  private activeStart = 0;
  private activeMaximumLiveCount = 0;
  private activeMaximumLiveBytes = 0;
  private maximumLiveCount = 0;
  private maximumLiveBytes = 0;

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

  beginArm(arm: Opt0035Arm): void {
    if (this.activeArm !== undefined || this.liveCount() !== 0) {
      throw new Error("OPT-0035 resource-audit arms overlap");
    }
    this.activeArm = arm;
    this.activeStart = this.records.length;
    this.activeMaximumLiveCount = 0;
    this.activeMaximumLiveBytes = 0;
  }

  finishArm(): Readonly<Record<string, unknown>> {
    const arm = this.activeArm;
    if (arm === undefined) {
      throw new Error("OPT-0035 resource audit has no active arm");
    }
    const selected = this.records.slice(this.activeStart);
    this.activeArm = undefined;
    const createdBufferBytes = selected.reduce(
      (sum, record) => sum + record.size,
      0,
    );
    const destroyedBufferCount = selected.filter((record) =>
      record.destroyed
    ).length;
    const everyBufferDestroyedExactlyOnce = selected.every((record) =>
      record.destroyed && record.destroyCalls === 1
    );
    const passed = selected.length > 0 && this.liveCount() === 0 &&
      destroyedBufferCount === selected.length &&
      everyBufferDestroyedExactlyOnce;
    const receipt = Object.freeze({
      arm,
      createdBufferCount: selected.length,
      createdBufferBytes,
      destroyedBufferCount,
      totalDestroyCallCount: selected.reduce(
        (sum, record) => sum + record.destroyCalls,
        0,
      ),
      liveBufferCountAfterArm: this.liveCount(),
      liveBufferBytesAfterArm: this.liveBytes(),
      maximumLiveBufferCount: this.activeMaximumLiveCount,
      maximumLiveBufferBytes: this.activeMaximumLiveBytes,
      everyBufferDestroyedExactlyOnce,
      passed,
    });
    if (!passed) {
      throw new Error(
        `OPT-0035 ${arm} did not destroy every tracked GPU buffer exactly once`,
      );
    }
    return receipt;
  }

  snapshot(): Readonly<Record<string, unknown>> {
    const everyBufferDestroyedExactlyOnce = this.records.every((record) =>
      record.destroyed && record.destroyCalls === 1
    );
    return Object.freeze({
      activeArm: this.activeArm ?? null,
      createdBufferCount: this.records.length,
      createdBufferBytes: this.records.reduce(
        (sum, record) => sum + record.size,
        0,
      ),
      destroyedBufferCount: this.records.filter((record) =>
        record.destroyed
      ).length,
      totalDestroyCallCount: this.records.reduce(
        (sum, record) => sum + record.destroyCalls,
        0,
      ),
      liveBufferCount: this.liveCount(),
      liveBufferBytes: this.liveBytes(),
      maximumLiveBufferCount: this.maximumLiveCount,
      maximumLiveBufferBytes: this.maximumLiveBytes,
      everyBufferDestroyedExactlyOnce,
      passed: this.activeArm === undefined && this.liveCount() === 0 &&
        everyBufferDestroyedExactlyOnce,
    });
  }

  private track(
    buffer: GPUBuffer,
    descriptor: GPUBufferDescriptor,
  ): GPUBuffer {
    const arm = this.activeArm;
    if (arm === undefined) {
      buffer.destroy();
      throw new Error("OPT-0035 observed a GPU allocation outside an arm");
    }
    const record: Opt0035BufferRecord = {
      arm,
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      destroyed: false,
      destroyCalls: 0,
    };
    this.records.push(record);
    const destroy = buffer.destroy.bind(buffer);
    Object.defineProperty(buffer, "destroy", {
      configurable: true,
      value: () => {
        record.destroyCalls += 1;
        if (record.destroyed) return;
        record.destroyed = true;
        destroy();
      },
    });
    this.updateMaximum();
    return buffer;
  }

  private updateMaximum(): void {
    const count = this.liveCount();
    const bytes = this.liveBytes();
    this.activeMaximumLiveCount = Math.max(
      this.activeMaximumLiveCount,
      count,
    );
    this.activeMaximumLiveBytes = Math.max(
      this.activeMaximumLiveBytes,
      bytes,
    );
    this.maximumLiveCount = Math.max(this.maximumLiveCount, count);
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, bytes);
  }

  private liveCount(): number {
    return this.records.filter((record) => !record.destroyed).length;
  }

  private liveBytes(): number {
    return this.records.filter((record) => !record.destroyed).reduce(
      (sum, record) => sum + record.size,
      0,
    );
  }
}

class Opt0035ListeningArtifact {
  private removed = false;

  constructor(
    readonly audio: Blob,
    private readonly root: FileSystemDirectoryHandle,
    private readonly directoryName: string,
  ) {}

  async remove(): Promise<void> {
    if (this.removed) return;
    this.removed = true;
    await this.root.removeEntry(this.directoryName, { recursive: true });
  }
}

class Opt0035RawArtifact {
  readonly sink: AceVaeRawF32FileSink;
  private closed = false;
  private removed = false;

  private constructor(
    private readonly root: FileSystemDirectoryHandle,
    private readonly directoryName: string,
    private readonly directory: FileSystemDirectoryHandle,
    private readonly rawAccess: FileSystemSyncAccessHandle,
    plan: AceVaeChunkedDecodePlan,
  ) {
    this.sink = new AceVaeRawF32FileSink(rawAccess, plan);
  }

  static async create(plan: AceVaeChunkedDecodePlan): Promise<Opt0035RawArtifact> {
    const root = await navigator.storage.getDirectory();
    const directoryName = `ace-opt-0035-${crypto.randomUUID()}`;
    const directory = await root.getDirectoryHandle(directoryName, {
      create: true,
    });
    let access: FileSystemSyncAccessHandle | undefined;
    try {
      const rawHandle = await directory.getFileHandle("raw.f32", {
        create: true,
      });
      access = await rawHandle.createSyncAccessHandle();
      return new Opt0035RawArtifact(
        root,
        directoryName,
        directory,
        access,
        plan,
      );
    } catch (error) {
      access?.close();
      await root.removeEntry(directoryName, { recursive: true });
      throw error;
    }
  }

  finish(): void {
    this.requireOpen();
    this.sink.finish();
  }

  readBlock(at: number, byteLength: number): Uint8Array<ArrayBuffer> {
    this.requireOpen();
    const bytes = new Uint8Array(byteLength);
    const read = this.rawAccess.read(bytes, { at });
    if (read !== byteLength) {
      throw new Error(`OPT-0035 raw read returned ${read}/${byteLength} bytes`);
    }
    return bytes;
  }

  async hashBounded(expectedBytes: number): Promise<string> {
    this.requireOpen();
    if (this.rawAccess.getSize() !== expectedBytes) {
      throw new Error("OPT-0035 raw output has the wrong byte length");
    }
    const hash = new AceIncrementalSha256();
    for (let at = 0; at < expectedBytes; at += RAW_BLOCK_BYTES) {
      hash.update(this.readBlock(at, Math.min(RAW_BLOCK_BYTES, expectedBytes - at)));
      await browserYield();
    }
    return hash.digestHex();
  }

  async retainNormalizedWav(
    plan: AceVaeChunkedDecodePlan,
    stats: AceVaeRawStreamStats,
  ): Promise<Opt0035ListeningArtifact> {
    this.requireOpen();
    const wavHandle = await this.directory.getFileHandle("candidate.wav", {
      create: true,
    });
    const wavAccess = await wavHandle.createSyncAccessHandle();
    try {
      await writeNormalizedAceVaeFloat32WavCooperatively(
        this.rawAccess,
        wavAccess,
        plan,
        deriveAceVaePostprocessPlan(stats.peak),
      );
    } finally {
      wavAccess.close();
    }
    this.rawAccess.close();
    this.closed = true;
    await this.directory.removeEntry("raw.f32");
    const file = await wavHandle.getFile();
    return new Opt0035ListeningArtifact(
      file.slice(0, file.size, "audio/wav"),
      this.root,
      this.directoryName,
    );
  }

  async remove(): Promise<void> {
    if (this.removed) return;
    this.removed = true;
    if (!this.closed) {
      this.rawAccess.close();
      this.closed = true;
    }
    await this.root.removeEntry(this.directoryName, { recursive: true });
  }

  private requireOpen(): void {
    if (this.closed || this.removed) {
      throw new Error("OPT-0035 raw artifact is closed");
    }
  }
}

function installWorker(): void {
  let prepared: PreparedGate | undefined;
  let operation = Promise.resolve();
  globalThis.addEventListener(
    "message",
    (event: MessageEvent<Opt0035WorkerCommand>) => {
      operation = operation.then(async () => {
        if (event.data.type === "prepare") {
          if (prepared !== undefined) throw new Error("OPT-0035 already prepared");
          prepared = await prepareGate(event.data.quantaPerCommandBuffer);
          postWorker({
            type: "prepared",
            message: "READY — sequential C512/C2378 correctness passed; timing has not run",
            receipt: prepared.correctness,
            audio: prepared.listeningArtifact.audio,
          });
          return;
        }
        if (event.data.type === "dispose") {
          const retained = prepared;
          prepared = undefined;
          await retained?.destroy();
          postWorker({ type: "disposed" });
          return;
        }
        if (prepared === undefined) {
          throw new Error("OPT-0035 timing requested before READY");
        }
        const retained = prepared;
        prepared = undefined;
        let receipt: Readonly<Record<string, unknown>> = Object.freeze({});
        try {
          receipt = await runTimedGate(retained);
        } finally {
          const cleanup = await retained.destroy();
          receipt = Object.freeze({
            ...(receipt ?? {}),
            cleanup,
          });
        }
        postWorker({ type: "result", receipt });
      }).catch(async (error: unknown) => {
        const retained = prepared;
        prepared = undefined;
        const cleanup = await retained?.destroy();
        postWorker({
          type: "error",
          message: errorText(error),
          receipt: Object.freeze({
            schema: OPT_0035_SCHEMA,
            experimentId: EXPERIMENT_ID,
            status: "failed",
            error: errorText(error),
            ...(cleanup === undefined ? {} : { cleanup }),
          }),
        });
      });
    },
  );
}

async function prepareGate(
  selectedBatch: number,
): Promise<PreparedGate> {
  const quantaPerCommandBuffer = resolveOpt0035QuantaPerCommandBuffer(
    selectedBatch,
  );
  const coverage = planOpt0035Coverage();
  const controlMemory = planAceOpt0011Fp16VaeChunkGpuBackendMemory(
    coverage.control,
    256,
    quantaPerCommandBuffer,
  );
  const candidateMemory = planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory(
    coverage.candidate,
    256,
    quantaPerCommandBuffer,
  );
  if (
    candidateMemory.workspaceBufferBytes !==
      ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES ||
    candidateMemory.maximumWindowFrames !==
      OPT_0035_CANDIDATE_CHUNK_FRAMES
  ) throw new Error("OPT-0035 C2378 memory contract changed");

  postProgress("authenticating deterministic C4500 latent and revision-6 package");
  const fixtureBytes = createAceOpt0011LatentFixture(OPT_0035_LATENT_FRAMES);
  if (
    fixtureBytes.byteLength !== OPT_0035_LATENT_BYTES ||
    await sha256Bytes(fixtureBytes) !== OPT_0035_LATENT_SHA256
  ) throw new Error("OPT-0035 C4500 latent identity changed");
  const fixture = Float32Array.from(new Float32Array(
    fixtureBytes.buffer,
    fixtureBytes.byteOffset,
    OPT_0035_LATENT_ELEMENTS,
  )) as Float32Array<ArrayBuffer>;
  const pkg = await authenticatePackage();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  const activeGuard = new ActiveArmGuard();
  let context: AceWebGpuDeviceContext | undefined;
  let control: ArmExecution | undefined;
  let candidate: ArmExecution | undefined;
  let repeat: ArmExecution | undefined;
  let listeningArtifact: Opt0035ListeningArtifact | undefined;
  let resourceAudit: Opt0035DeviceResourceAudit | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      requiredLimits: {
        maxBufferSize: candidateMemory.workspaceBufferBytes,
        maxStorageBufferBindingSize: candidateMemory.workspaceBufferBytes,
      },
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });
    requireDevice(context, candidateMemory.workspaceBufferBytes);
    resourceAudit = new Opt0035DeviceResourceAudit(context.device);
    const files = await acquirePackageFiles(pkg);

    postProgress("untimed correctness 1/3: sequential exact C512 authority");
    control = await executeArm({
      arm: "c512",
      context,
      pkg,
      files,
      fixture,
      plan: coverage.control,
      maximumWindowFrames: OPT_0035_CONTROL_CHUNK_FRAMES,
      quantaPerCommandBuffer,
      activeGuard,
      resourceAudit,
    });
    postProgress("untimed correctness 2/3: sequential exact C2378 candidate");
    candidate = await executeArm({
      arm: "c2378",
      context,
      pkg,
      files,
      fixture,
      plan: coverage.candidate,
      maximumWindowFrames: OPT_0035_CANDIDATE_CHUNK_FRAMES,
      quantaPerCommandBuffer,
      activeGuard,
      resourceAudit,
    });
    postProgress("untimed correctness 3/3: exact C2378 determinism repeat");
    repeat = await executeArm({
      arm: "c2378",
      context,
      pkg,
      files,
      fixture,
      plan: coverage.candidate,
      maximumWindowFrames: OPT_0035_CANDIDATE_CHUNK_FRAMES,
      quantaPerCommandBuffer,
      activeGuard,
      resourceAudit,
    });

    postProgress("bounded OPFS waveform and seam comparison");
    const comparison = await compareRawArtifacts(
      control.raw,
      candidate.raw,
      coverage,
    );
    const deterministic = candidate.sha256 === repeat.sha256 &&
      candidate.stats.peak === repeat.stats.peak &&
      candidate.stats.finiteSamples === repeat.stats.finiteSamples;
    const waveformPassed = allMetricsPassed(comparison.full);
    const seamsFinite = Object.values(comparison.seams).every((metrics) =>
      metrics.joint.finite && metrics.left.finite && metrics.right.finite
    );
    if (
      !deterministic || !waveformPassed || !seamsFinite ||
      runtimeEvents.length !== 0 ||
      activeGuard.snapshot()["sequentialOwnershipPassed"] !== true
    ) throw new Error("OPT-0035 untimed correctness gate failed");

    const correctnessHashes = Object.freeze({
      control: control.sha256,
      candidate: candidate.sha256,
      repeat: repeat.sha256,
    });
    const correctnessResources = Object.freeze({
      control: control.resources,
      candidate: candidate.resources,
      repeat: repeat.resources,
      aggregate: resourceAudit.snapshot(),
    });
    listeningArtifact = await candidate.raw.retainNormalizedWav(
      coverage.candidate,
      candidate.stats,
    );
    await Promise.all([control.raw.remove(), repeat.raw.remove()]);
    control = undefined;
    candidate = undefined;
    repeat = undefined;
    const correctness = Object.freeze({
      schema: OPT_0035_SCHEMA,
      experimentId: EXPERIMENT_ID,
      status: "ready",
      classification:
        "revision6-exact-packed-nativeK7-C512vsC2378-sequential",
      fixture: Object.freeze({
        latentFrames: OPT_0035_LATENT_FRAMES,
        channels: OPT_0035_LATENT_CHANNELS,
        elementCount: OPT_0035_LATENT_ELEMENTS,
        byteLength: OPT_0035_LATENT_BYTES,
        sha256: OPT_0035_LATENT_SHA256,
      }),
      package: Object.freeze({
        manifestSha256: pkg.loaded.manifestSha256,
        manifestByteLength: pkg.loaded.manifestByteLength,
        converterRevision: pkg.loaded.manifest.provenance.converterRevision,
        residentBytes: pkg.residentBytes,
      }),
      runtime: Object.freeze({
        runtimeProfileId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
        kernelSetId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId,
        nativeExactK7: true,
        approximateK7Used: false,
        quantaPerCommandBuffer,
        adapterInfo: context.capabilities.adapterInfo,
        deviceLimits: context.capabilities.deviceLimits,
        requestedLimits: context.capabilities.requestedLimits,
      }),
      coverage: coverageReceipt(coverage),
      memory: Object.freeze({ control: controlMemory, candidate: candidateMemory }),
      correctness: Object.freeze({
        controlSha256: correctnessHashes.control,
        candidateSha256: correctnessHashes.candidate,
        repeatSha256: correctnessHashes.repeat,
        deterministic,
        metrics: comparison.full,
        seamNeighborhoods: comparison.seams,
        seamRadiusLatentFrames: OPT_0035_SEAM_RADIUS_LATENT_FRAMES,
        waveformPassed,
        seamsFinite,
        listeningArtifact: "normalized-float32-wav",
        resources: correctnessResources,
      }),
      ownership: activeGuard.snapshot(),
      runtimeEvents: Object.freeze([...runtimeEvents]),
      GPUWorkRanOnlyAfterExplicitPrepareButton: true,
      timingHasRun: false,
      externalNominalThermalGateRequiredBeforeTiming: true,
    });
    return createPreparedGate({
      context,
      pkg,
      files,
      fixture,
      quantaPerCommandBuffer,
      runtimeEvents,
      activeGuard,
      resourceAudit,
      listeningArtifact,
      correctness,
    });
  } catch (error) {
    await Promise.allSettled([
      control?.raw.remove() ?? Promise.resolve(),
      candidate?.raw.remove() ?? Promise.resolve(),
      repeat?.raw.remove() ?? Promise.resolve(),
      listeningArtifact?.remove() ?? Promise.resolve(),
    ]);
    context?.destroy();
    throw error;
  }
}

function createPreparedGate(
  input: Omit<PreparedGate, "destroy">,
): PreparedGate {
  let destroyPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  return Object.freeze({
    ...input,
    destroy(): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        input.context.destroy();
        await input.listeningArtifact.remove();
        return Object.freeze({
          deviceContextDestroyed: true,
          allBackendAndWeightOwnersDestroyedBeforeResult: true,
          allActivationReadbackAndTemporaryRawFilesRemoved: true,
          listeningArtifactRemoved: true,
          ownership: input.activeGuard.snapshot(),
          resources: input.resourceAudit.snapshot(),
          passed:
            input.activeGuard.snapshot()["sequentialOwnershipPassed"] === true &&
            input.resourceAudit.snapshot()["passed"] === true,
        });
      })();
      return destroyPromise;
    },
  });
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const coverage = planOpt0035Coverage();
  const executions: Readonly<Record<string, unknown>>[] = [];
  for (let index = 0; index < OPT_0035_TIMED_ORDER.length; index += 1) {
    const arm = OPT_0035_TIMED_ORDER[index]!;
    postProgress(`timed ${index + 1}/4: ${arm}`);
    const plan = arm === "c512" ? coverage.control : coverage.candidate;
    const execution = await executeArm({
      arm,
      context: prepared.context,
      pkg: prepared.pkg,
      files: prepared.files,
      fixture: prepared.fixture,
      plan,
      maximumWindowFrames: arm === "c512"
        ? OPT_0035_CONTROL_CHUNK_FRAMES
        : OPT_0035_CANDIDATE_CHUNK_FRAMES,
      quantaPerCommandBuffer: prepared.quantaPerCommandBuffer,
      activeGuard: prepared.activeGuard,
      resourceAudit: prepared.resourceAudit,
    });
    executions.push(Object.freeze({
      arm,
      wallMs: execution.wallMs,
      stats: execution.stats,
      memory: execution.memory,
      resources: execution.resources,
    }));
    await execution.raw.remove();
  }
  const controlMedianMs = median(executions
    .filter((entry) => entry["arm"] === "c512")
    .map((entry) => entry["wallMs"] as number));
  const candidateMedianMs = median(executions
    .filter((entry) => entry["arm"] === "c2378")
    .map((entry) => entry["wallMs"] as number));
  const speedup = controlMedianMs / candidateMedianMs;
  const passed = speedup >= OPT_0035_SPEEDUP_GATE &&
    prepared.runtimeEvents.length === 0 &&
    prepared.activeGuard.snapshot()["sequentialOwnershipPassed"] === true;
  return Object.freeze({
    schema: OPT_0035_SCHEMA,
    experimentId: EXPERIMENT_ID,
    status: passed ? "passed" : "failed",
    classification:
      "revision6-exact-packed-nativeK7-C512vsC2378-sequential-ABBA",
    order: OPT_0035_TIMED_ORDER,
    executions: Object.freeze(executions),
    aggregate: Object.freeze({
      controlMedianMs,
      candidateMedianMs,
      speedup,
      savingMs: controlMedianMs - candidateMedianMs,
      requiredSpeedup: OPT_0035_SPEEDUP_GATE,
    }),
    correctnessAuthority: prepared.correctness,
    ownership: prepared.activeGuard.snapshot(),
    runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
    resources: prepared.resourceAudit.snapshot(),
    performanceGatePassed: passed,
    listeningApprovalRequiredForProduction: true,
    productionDefaultChanged: false,
  });
}

async function executeArm(input: Readonly<{
  arm: Opt0035Arm;
  context: AceWebGpuDeviceContext;
  pkg: PreparedPackage;
  files: ReadonlyMap<string, File>;
  fixture: Float32Array<ArrayBuffer>;
  plan: AceVaeChunkedDecodePlan;
  maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames;
  quantaPerCommandBuffer: Opt0035QuantaPerCommandBuffer;
  activeGuard: ActiveArmGuard;
  resourceAudit: Opt0035DeviceResourceAudit;
}>): Promise<ArmExecution> {
  const release = input.activeGuard.acquire();
  let resourceAuditStarted = false;
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  let raw: Opt0035RawArtifact | undefined;
  try {
    input.resourceAudit.beginArm(input.arm);
    resourceAuditStarted = true;
    phase = await loadVaePhase(
      input.resourceAudit.device,
      input.pkg,
      input.files,
      input.arm,
    );
    const ownedPhase = phase;
    phase = undefined;
    backend = await AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: input.resourceAudit.device,
      plan: input.plan,
      finalLatents: input.fixture,
      authenticatedPackage: input.pkg.loaded,
      ownedVaeWeights: ownedPhase,
      maximumWindowFrames: input.maximumWindowFrames,
      runtimeProfileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      quantaPerCommandBuffer: input.quantaPerCommandBuffer,
    });
    if (
      backend.runtimeProfileId !==
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id ||
      backend.memory.maximumWindowFrames !== input.maximumWindowFrames ||
      backend.memory.quantaPerCommandBuffer !== input.quantaPerCommandBuffer
    ) throw new Error(`OPT-0035 ${input.arm} backend contract changed`);
    raw = await Opt0035RawArtifact.create(input.plan);
    const started = performance.now();
    const stats = await streamAceVaeRawChunks(input.plan, backend, raw.sink);
    raw.finish();
    const wallMs = performance.now() - started;
    if (
      stats.outputInterleavedElements !== OPT_0035_OUTPUT_ELEMENTS ||
      stats.finiteSamples !== OPT_0035_OUTPUT_ELEMENTS ||
      stats.windowsDecoded !== input.plan.windows.length ||
      !Number.isFinite(stats.peak) || stats.peak <= 0
    ) throw new Error(`OPT-0035 ${input.arm} produced invalid raw output`);
    const sha256 = await raw.hashBounded(OPT_0035_OUTPUT_BYTES);
    const memory = backend.memory;
    await backend.destroy();
    backend = undefined;
    await input.resourceAudit.device.queue.onSubmittedWorkDone();
    await browserYield();
    const resources = input.resourceAudit.finishArm();
    release();
    return Object.freeze({
      arm: input.arm,
      wallMs,
      stats,
      memory,
      raw,
      sha256,
      resources,
    });
  } catch (error) {
    await backend?.destroy(error);
    phase?.destroy();
    await raw?.remove();
    let resourceError: unknown;
    if (resourceAuditStarted) {
      try {
        input.resourceAudit.finishArm();
      } catch (caught) {
        resourceError = caught;
      }
    }
    release();
    if (resourceError !== undefined) {
      throw new AggregateError(
        [error, resourceError],
        `OPT-0035 ${input.arm} execution and GPU cleanup both failed`,
      );
    }
    throw error;
  }
}

async function compareRawArtifacts(
  control: Opt0035RawArtifact,
  candidate: Opt0035RawArtifact,
  coverage: Opt0035CoverageReceipt,
): Promise<Readonly<{
  full: Opt0035WaveformMetrics;
  seams: Readonly<Record<string, Opt0035WaveformMetrics>>;
}>> {
  const full = new StereoMetricAccumulator();
  const seamFrames = Object.freeze([
    ...coverage.controlSeams.map((frame) => [`control-${frame}`, frame] as const),
    ...coverage.candidateSeams.map((frame) => [`candidate-${frame}`, frame] as const),
  ]);
  const seamAccumulators = new Map(
    seamFrames.map(([label]) => [label, new StereoMetricAccumulator()]),
  );
  for (let at = 0; at < OPT_0035_OUTPUT_BYTES; at += RAW_BLOCK_BYTES) {
    const byteLength = Math.min(RAW_BLOCK_BYTES, OPT_0035_OUTPUT_BYTES - at);
    const controlValues = new Float32Array(control.readBlock(at, byteLength).buffer);
    const candidateValues = new Float32Array(candidate.readBlock(at, byteLength).buffer);
    full.add(controlValues, candidateValues);
    const blockStartElement = at / Float32Array.BYTES_PER_ELEMENT;
    const blockEndElement = blockStartElement + controlValues.length;
    for (const [label, latentFrame] of seamFrames) {
      const startElement = Math.max(0,
        (latentFrame - OPT_0035_SEAM_RADIUS_LATENT_FRAMES) * 1_920 * 2);
      const endElement = Math.min(OPT_0035_OUTPUT_ELEMENTS,
        (latentFrame + OPT_0035_SEAM_RADIUS_LATENT_FRAMES) * 1_920 * 2);
      const overlapStart = Math.max(blockStartElement, startElement);
      const overlapEnd = Math.min(blockEndElement, endElement);
      if (overlapStart >= overlapEnd) continue;
      const localStart = overlapStart - blockStartElement;
      const localEnd = overlapEnd - blockStartElement;
      seamAccumulators.get(label)!.add(
        controlValues.subarray(localStart, localEnd),
        candidateValues.subarray(localStart, localEnd),
      );
    }
    await browserYield();
  }
  return Object.freeze({
    full: full.finish(),
    seams: Object.freeze(Object.fromEntries(
      [...seamAccumulators].map(([label, accumulator]) => [
        label,
        accumulator.finish(),
      ]),
    )),
  });
}

async function authenticatePackage(): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(MANIFEST_PATH, globalThis.location.href).href,
    expectedManifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
  });
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) => shardNames.has(file.name));
  const residentBytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  if (
    loaded.manifestSha256 !== ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES ||
    loaded.manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) throw new Error("OPT-0035 authenticated revision-6 package changed");
  return Object.freeze({ loaded, files: Object.freeze(files), residentBytes });
}

async function acquirePackageFiles(
  pkg: PreparedPackage,
): Promise<ReadonlyMap<string, File>> {
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({ ...pkg.loaded.manifest, files: pkg.files }),
    manifestUrl: pkg.loaded.manifestUrl,
    cache: await AceOpfsModelCache.open(),
    onFileProgress: (progress) => postProgress(
      `acquiring VAE ${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes}`,
    ),
  });
  if (
    acquired.files.size !== pkg.files.length ||
    acquired.plan.runtimeBytes !== pkg.residentBytes
  ) throw new Error("OPT-0035 package acquisition accounting changed");
  return acquired.files;
}

async function loadVaePhase(
  device: GPUDevice,
  pkg: PreparedPackage,
  files: ReadonlyMap<string, File>,
  arm: Opt0035Arm,
): Promise<AceGpuTensorPhase> {
  postProgress(`uploading sequential ${arm} VAE weights`);
  const phase = await AceGpuTensorPhase.load(
    device,
    pkg.loaded.manifest,
    files,
    ["vae"],
  );
  if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
    phase.destroy();
    throw new Error(`OPT-0035 ${arm} resident weight bytes changed`);
  }
  return phase;
}

function requireDevice(
  context: AceWebGpuDeviceContext,
  requiredWorkspaceBytes: number,
): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32 ||
    context.device.limits.maxBufferSize < requiredWorkspaceBytes ||
    context.device.limits.maxStorageBufferBindingSize < requiredWorkspaceBytes
  ) throw new Error("OPT-0035 requires stock fixed32 WebGPU and C2378 limits");
}

function coverageReceipt(
  coverage: Opt0035CoverageReceipt,
): Readonly<Record<string, unknown>> {
  const summarize = (plan: AceVaeChunkedDecodePlan) => Object.freeze({
    chunkFrames: plan.chunkFrames,
    overlapFrames: plan.overlapFrames,
    strideFrames: plan.strideFrames,
    windowCount: plan.windows.length,
    maximumActualWindowFrames: plan.maximumWindowFrames,
    windows: Object.freeze(plan.windows.map((window) => Object.freeze({
      index: window.index,
      latentWindow: Object.freeze([
        window.windowStartLatentFrame,
        window.windowEndLatentFrame,
      ]),
      core: Object.freeze([
        window.coreStartLatentFrame,
        window.coreEndLatentFrame,
      ]),
      discardPrefixLatentFrames: window.discardPrefixLatentFrames,
      discardSuffixLatentFrames: window.discardSuffixLatentFrames,
    }))),
  });
  return Object.freeze({
    control: summarize(coverage.control),
    candidate: summarize(coverage.candidate),
    controlDecodedLatentFrames: coverage.controlDecodedLatentFrames,
    candidateDecodedLatentFrames: coverage.candidateDecodedLatentFrames,
    decodedLatentFrameReduction: coverage.decodedLatentFrameReduction,
    decodedLatentFrameReductionRatio:
      coverage.decodedLatentFrameReductionRatio,
    controlSeams: coverage.controlSeams,
    candidateSeams: coverage.candidateSeams,
    exactOutputCoverageOnce: true,
  });
}

function allMetricsPassed(metrics: Opt0035WaveformMetrics): boolean {
  return metrics.joint.passed && metrics.left.passed && metrics.right.passed;
}

function median(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError("OPT-0035 median requires finite values");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
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

function postWorker(event: Opt0035WorkerEvent): void {
  (globalThis as unknown as DedicatedWorkerGlobalScope).postMessage(event);
}

function errorText(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}\n${error.stack ?? ""}`
    : String(error);
}

if (
  typeof WorkerGlobalScope !== "undefined" &&
  globalThis instanceof WorkerGlobalScope
) installWorker();
