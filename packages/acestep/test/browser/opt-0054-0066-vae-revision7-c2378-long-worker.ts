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
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import type { AceVaeChunkGpuBackendProgress } from
  "../../src/webgpu/vae-backend.js";
import {
  AceOpt0011Fp16VaeChunkGpuBackend,
  type AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
  type AceOpt0011Fp16VaeDispatchTopologyReceipt,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
} from "../../src/webgpu/vae-fp16-decoder.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  streamAceVaeRawChunks,
  type AceVaeChunkedDecodePlan,
  type AceVaeRawStreamStats,
  type AceVaeRawStreamTrace,
} from "../../src/webgpu/vae-chunks.js";
import { AceVaeRawF32FileSink } from "../../src/webgpu/vae-wav.js";
import {
  OPT_0054_0066_ARM_ORDER,
  OPT_0054_0066_AUDIO_CHANNELS,
  OPT_0054_0066_CONTROL_RAW_SHA256,
  OPT_0054_0066_EXPERIMENT_ASSOCIATIONS,
  OPT_0054_0066_HOP_LENGTH,
  OPT_0054_0066_LATENT_BYTES,
  OPT_0054_0066_LATENT_CHANNELS,
  OPT_0054_0066_LATENT_ELEMENTS,
  OPT_0054_0066_LATENT_FRAMES,
  OPT_0054_0066_LATENT_SHA256,
  OPT_0054_0066_LONG_PROTOCOL_ID,
  OPT_0054_0066_LONG_SCHEMA,
  OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
  OPT_0054_0066_OUTPUT_BYTES,
  OPT_0054_0066_OUTPUT_ELEMENTS,
  OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
  OPT_0054_0066_RAW_BLOCK_BYTES,
  OPT_0054_0066_SEAM_RADIUS_LATENT_FRAMES,
  Opt00540066StereoMetricAccumulator,
  allOpt00540066MetricsPassed,
  planOpt00540066LongGate,
  reconcileOpt00540066CapturedWindowTopology,
  type Opt00540066Arm,
  type Opt00540066ArmTopologyPlan,
  type Opt00540066LongGatePlan,
  type Opt00540066StereoWaveformMetrics,
} from "./opt-0054-0066-vae-revision7-c2378-long-contract.js";

const REVISION6_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json";
const REVISION7_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json";
const REQUIRED_SUBGROUP_SIZE = 32;

export type Opt00540066WorkerCommand =
  | Readonly<{ readonly type: "run" }>
  | Readonly<{ readonly type: "cancel" | "dispose" }>;

interface WorkerEvent {
  readonly type: "progress" | "result" | "cancelled" | "disposed" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
}

interface PreparedPackage {
  readonly revision: 6 | 7;
  readonly manifestPath: string;
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly acquired: ReadonlyMap<string, File>;
  readonly residentBytes: number;
}

interface OwnerTracker {
  active: Opt00540066Arm | null;
  liveOwners: number;
  peakLiveOwners: number;
  created: Record<Opt00540066Arm, number>;
  destroyed: Record<Opt00540066Arm, number>;
  everyBackendDestroyIdempotent: boolean;
}

interface ExecutionResult {
  readonly arm: Exclude<Opt00540066Arm, "rev7-cancellation-probe">;
  readonly raw: Opt00540066RawArtifact;
  readonly sha256: string;
  readonly scan: RawScan;
  readonly stats: AceVaeRawStreamStats;
  readonly trace: AceVaeRawStreamTrace;
  readonly memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly topology: AceOpt0011Fp16VaeDispatchTopologyReceipt;
  readonly backendValidation: Readonly<Record<string, unknown>>;
  readonly progress: Readonly<Record<string, unknown>>;
  readonly ownerLifecycle: Readonly<Record<string, unknown>>;
  readonly resources: Readonly<Record<string, unknown>>;
  readonly seamContinuity: readonly Readonly<Record<string, unknown>>[];
}

interface RawScan {
  readonly byteLength: number;
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly zeroCount: number;
  readonly stereoDifferenceFrameCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly peak: number;
  readonly firstLeft: number;
  readonly firstRight: number;
  readonly lastLeft: number;
  readonly lastRight: number;
  readonly sampleClass: "finite-bipolar-nonzero-stereo";
  readonly passed: true;
}

interface RawComparison {
  readonly comparedU32WordCount: number;
  readonly u32MismatchCount: number;
  readonly firstU32MismatchIndex: number | null;
  readonly rawU32Exact: boolean;
  readonly metrics: Opt00540066StereoWaveformMetrics;
  readonly seamNeighborhoods: Readonly<Record<
    string,
    Opt00540066StereoWaveformMetrics
  >>;
  readonly seamsFinite: boolean;
}

interface BufferRecord {
  readonly arm: Opt00540066Arm;
  readonly label: string;
  readonly size: number;
  destroyed: boolean;
  destroyCalls: number;
  mapCalls: number;
  unmapCalls: number;
  mappedOrPending: boolean;
}

/** Audit actual per-arm GPU high-water while forbidding co-resident owners. */
class DeviceResourceAudit {
  readonly device: GPUDevice;
  private readonly records: BufferRecord[] = [];
  private activeArm: Opt00540066Arm | undefined;
  private activeStart = 0;
  private activeMaximumLiveBytes = 0;
  private activeMaximumLiveCount = 0;
  private maximumLiveBytes = 0;
  private maximumLiveCount = 0;
  private maximumMappedCount = 0;

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

  beginArm(arm: Opt00540066Arm): void {
    if (this.activeArm !== undefined || this.liveRecords().length !== 0) {
      throw new Error("OPT-0054/0066 refused overlapping GPU arm ownership");
    }
    this.activeArm = arm;
    this.activeStart = this.records.length;
    this.activeMaximumLiveBytes = 0;
    this.activeMaximumLiveCount = 0;
  }

  finishArm(
    expectedMemory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
  ): Readonly<Record<string, unknown>> {
    const arm = this.activeArm;
    if (arm === undefined) {
      throw new Error("OPT-0054/0066 resource audit has no active arm");
    }
    const selected = this.records.slice(this.activeStart);
    this.activeArm = undefined;
    const destroyed = selected.filter((record) => record.destroyed);
    const everyDestroyedExactlyOnce = selected.every((record) =>
      record.destroyed && record.destroyCalls === 1
    );
    const everyMapUnmapped = selected.every((record) =>
      !record.mappedOrPending && record.mapCalls === record.unmapCalls
    );
    const passed = selected.length > 0 && this.liveRecords().length === 0 &&
      destroyed.length === selected.length && everyDestroyedExactlyOnce &&
      everyMapUnmapped &&
      this.activeMaximumLiveBytes === expectedMemory.accountedGpuBytes &&
      this.activeMaximumLiveBytes < OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES;
    const receipt = Object.freeze({
      arm,
      createdBufferCount: selected.length,
      createdBufferBytes: sum(selected.map((record) => record.size)),
      destroyedBufferCount: destroyed.length,
      totalDestroyCallCount: sum(selected.map((record) => record.destroyCalls)),
      mapCallCount: sum(selected.map((record) => record.mapCalls)),
      unmapCallCount: sum(selected.map((record) => record.unmapCalls)),
      liveBufferCountAfterArm: this.liveRecords().length,
      maximumLiveBufferCount: this.activeMaximumLiveCount,
      maximumLiveBufferBytes: this.activeMaximumLiveBytes,
      plannedAccountedGpuBytes: expectedMemory.accountedGpuBytes,
      belowFourGigabytes:
        this.activeMaximumLiveBytes < OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
      everyBufferDestroyedExactlyOnce: everyDestroyedExactlyOnce,
      everyMapUnmapped,
      passed,
    });
    if (!passed) {
      throw new Error(
        `OPT-0054/0066 ${arm} GPU resource accounting failed: ` +
          JSON.stringify(receipt),
      );
    }
    return receipt;
  }

  snapshot(): Readonly<Record<string, unknown>> {
    const live = this.liveRecords();
    const everyDestroyedExactlyOnce = this.records.every((record) =>
      record.destroyed && record.destroyCalls === 1
    );
    const everyMapUnmapped = this.records.every((record) =>
      !record.mappedOrPending && record.mapCalls === record.unmapCalls
    );
    return Object.freeze({
      activeArm: this.activeArm ?? null,
      createdBufferCount: this.records.length,
      destroyedBufferCount: this.records.length - live.length,
      liveBufferCount: live.length,
      liveBufferBytes: sum(live.map((record) => record.size)),
      maximumLiveBufferCount: this.maximumLiveCount,
      maximumLiveBufferBytes: this.maximumLiveBytes,
      maximumMappedBufferCount: this.maximumMappedCount,
      everyBufferDestroyedExactlyOnce: everyDestroyedExactlyOnce,
      everyMapUnmapped,
      passed: this.activeArm === undefined && live.length === 0 &&
        everyDestroyedExactlyOnce && everyMapUnmapped &&
        this.maximumLiveBytes < OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
    });
  }

  private track(
    buffer: GPUBuffer,
    descriptor: GPUBufferDescriptor,
  ): GPUBuffer {
    const arm = this.activeArm;
    if (arm === undefined) {
      buffer.destroy();
      throw new Error("OPT-0054/0066 observed GPU allocation outside an arm");
    }
    const record: BufferRecord = {
      arm,
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      destroyed: false,
      destroyCalls: 0,
      mapCalls: 0,
      unmapCalls: 0,
      mappedOrPending: descriptor.mappedAtCreation === true,
    };
    this.records.push(record);
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
          record.mappedOrPending = false;
          destroy();
        },
      },
      mapAsync: {
        configurable: true,
        value: async (...args: Parameters<GPUBuffer["mapAsync"]>) => {
          record.mapCalls += 1;
          record.mappedOrPending = true;
          this.observe();
          try {
            await mapAsync(...args);
          } catch (error) {
            record.mappedOrPending = false;
            throw error;
          }
        },
      },
      unmap: {
        configurable: true,
        value: () => {
          record.unmapCalls += 1;
          record.mappedOrPending = false;
          unmap();
        },
      },
    });
    this.observe();
    return buffer;
  }

  private observe(): void {
    const live = this.liveRecords();
    const liveBytes = sum(live.map((record) => record.size));
    const mapped = this.records.filter((record) => record.mappedOrPending).length;
    this.activeMaximumLiveCount = Math.max(this.activeMaximumLiveCount, live.length);
    this.activeMaximumLiveBytes = Math.max(this.activeMaximumLiveBytes, liveBytes);
    this.maximumLiveCount = Math.max(this.maximumLiveCount, live.length);
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, liveBytes);
    this.maximumMappedCount = Math.max(this.maximumMappedCount, mapped);
  }

  private liveRecords(): BufferRecord[] {
    return this.records.filter((record) => !record.destroyed);
  }
}

class ProgressObserver {
  private count = 0;
  private readonly finalByWindow = new Map<number, AceVaeChunkGpuBackendProgress>();

  readonly onProgress = (progress: AceVaeChunkGpuBackendProgress): void => {
    this.count += 1;
    if (progress.stage === "readback") {
      if (this.finalByWindow.has(progress.windowIndex)) {
        throw new Error("OPT-0054/0066 duplicated a window readback progress event");
      }
      this.finalByWindow.set(progress.windowIndex, Object.freeze({ ...progress }));
    }
  };

  finish(
    plan: AceVaeChunkedDecodePlan,
    topology: AceOpt0011Fp16VaeDispatchTopologyReceipt,
    expected: Opt00540066ArmTopologyPlan,
  ): Readonly<Record<string, unknown>> {
    const byFrames = new Map(topology.windows.map((window) => [
      window.inputFrames,
      window,
    ]));
    const windows = plan.windows.map((window) => {
      const dispatch = byFrames.get(window.latentWindowFrames);
      const final = this.finalByWindow.get(window.index);
      if (dispatch === undefined || final === undefined) {
        throw new Error("OPT-0054/0066 progress lost an exact window shape");
      }
      const decoderCommandBuffers = Math.ceil(
        dispatch.sequenceQuantumCount / OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
      );
      const totalCommandBuffers = decoderCommandBuffers + 1;
      if (
        final.completedDecoderQuanta !== dispatch.sequenceQuantumCount ||
        final.totalDecoderQuanta !== dispatch.sequenceQuantumCount ||
        final.completedCommandBuffers !== totalCommandBuffers ||
        final.totalCommandBuffers !== totalCommandBuffers ||
        final.queueDrains !== totalCommandBuffers ||
        final.cooperativeIdleMs !== totalCommandBuffers - 1 ||
        final.stage !== "readback"
      ) {
        throw new Error(
          `OPT-0054/0066 window ${window.index} scheduling changed`,
        );
      }
      return Object.freeze({
        windowIndex: window.index,
        inputFrames: window.latentWindowFrames,
        graphQuantumCount: dispatch.graphQuantumCount,
        sequenceQuantumCount: dispatch.sequenceQuantumCount,
        decoderCommandBufferCount: decoderCommandBuffers,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: totalCommandBuffers,
        queueDrainCount: final.queueDrains,
        requestedCooperativeIdleMs: final.cooperativeIdleMs,
      });
    });
    if (
      this.count !== expected.totalCommandBufferCount ||
      windows.length !== expected.windowCount
    ) {
      throw new Error("OPT-0054/0066 aggregate progress topology changed");
    }
    return Object.freeze({
      progressEventCount: this.count,
      finalReadbackEventCount: this.finalByWindow.size,
      windows: Object.freeze(windows),
      aggregateDecoderAndReadbackCommandBufferCount:
        expected.totalCommandBufferCount,
      aggregateDecoderRequestedCooperativeIdleMs:
        expected.decoderRequestedCooperativeIdleMs,
    });
  }
}

class Opt00540066RawArtifact {
  readonly sink: AceVaeRawF32FileSink;
  private closed = false;
  private removed = false;

  private constructor(
    readonly arm: Opt00540066Arm,
    private readonly root: FileSystemDirectoryHandle,
    private readonly directoryName: string,
    private readonly access: FileSystemSyncAccessHandle,
    plan: AceVaeChunkedDecodePlan,
  ) {
    this.sink = new AceVaeRawF32FileSink(access, plan);
  }

  static async create(
    arm: Opt00540066Arm,
    plan: AceVaeChunkedDecodePlan,
  ): Promise<Opt00540066RawArtifact> {
    const root = await navigator.storage.getDirectory();
    const directoryName =
      `ace-opt-0054-0066-c4500-${arm}-${crypto.randomUUID()}`;
    const directory = await root.getDirectoryHandle(directoryName, {
      create: true,
    });
    let access: FileSystemSyncAccessHandle | undefined;
    try {
      const handle = await directory.getFileHandle("raw.f32", { create: true });
      access = await handle.createSyncAccessHandle();
      return new Opt00540066RawArtifact(
        arm,
        root,
        directoryName,
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
    if (
      at % Float32Array.BYTES_PER_ELEMENT !== 0 ||
      byteLength % Float32Array.BYTES_PER_ELEMENT !== 0
    ) {
      throw new RangeError("OPT-0054/0066 raw read lost FP32 alignment");
    }
    const bytes = new Uint8Array(byteLength);
    const read = this.access.read(bytes, { at });
    if (read !== byteLength) {
      throw new Error(
        `OPT-0054/0066 raw read returned ${read}/${byteLength} bytes`,
      );
    }
    return bytes;
  }

  async inspect(
    signal: AbortSignal,
  ): Promise<Readonly<{ readonly sha256: string; readonly scan: RawScan }>> {
    this.requireOpen();
    if (this.access.getSize() !== OPT_0054_0066_OUTPUT_BYTES) {
      throw new Error("OPT-0054/0066 raw waveform byte length changed");
    }
    const hash = new AceIncrementalSha256();
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let nonzeroCount = 0;
    let positiveCount = 0;
    let negativeCount = 0;
    let zeroCount = 0;
    let stereoDifferenceFrameCount = 0;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let peak = 0;
    let firstLeft = Number.NaN;
    let firstRight = Number.NaN;
    let lastLeft = Number.NaN;
    let lastRight = Number.NaN;
    for (
      let at = 0;
      at < OPT_0054_0066_OUTPUT_BYTES;
      at += OPT_0054_0066_RAW_BLOCK_BYTES
    ) {
      signal.throwIfAborted();
      const byteLength = Math.min(
        OPT_0054_0066_RAW_BLOCK_BYTES,
        OPT_0054_0066_OUTPUT_BYTES - at,
      );
      const bytes = this.readBlock(at, byteLength);
      hash.update(bytes);
      const values = new Float32Array(bytes.buffer);
      if (at === 0) {
        firstLeft = values[0]!;
        firstRight = values[1]!;
      }
      lastLeft = values.at(-2)!;
      lastRight = values.at(-1)!;
      for (let index = 0; index < values.length; index += 2) {
        const left = values[index]!;
        const right = values[index + 1]!;
        if (left !== right) stereoDifferenceFrameCount += 1;
        for (let channel = 0; channel < 2; channel += 1) {
          const value = values[index + channel]!;
          if (!Number.isFinite(value)) {
            nonFiniteCount += 1;
            continue;
          }
          finiteCount += 1;
          if (value === 0) zeroCount += 1;
          else {
            nonzeroCount += 1;
            if (value > 0) positiveCount += 1;
            else negativeCount += 1;
          }
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
          peak = Math.max(peak, Math.abs(value));
        }
      }
      await browserYield();
    }
    const passed = finiteCount === OPT_0054_0066_OUTPUT_ELEMENTS &&
      nonFiniteCount === 0 && nonzeroCount > 0 && positiveCount > 0 &&
      negativeCount > 0 && stereoDifferenceFrameCount > 0 &&
      [firstLeft, firstRight, lastLeft, lastRight, minimum, maximum, peak]
        .every(Number.isFinite) && peak > 0;
    if (!passed) {
      throw new Error("OPT-0054/0066 raw waveform class scan failed");
    }
    return Object.freeze({
      sha256: hash.digestHex(),
      scan: Object.freeze({
        byteLength: OPT_0054_0066_OUTPUT_BYTES,
        elementCount: OPT_0054_0066_OUTPUT_ELEMENTS,
        finiteCount,
        nonFiniteCount,
        nonzeroCount,
        positiveCount,
        negativeCount,
        zeroCount,
        stereoDifferenceFrameCount,
        minimum,
        maximum,
        peak,
        firstLeft,
        firstRight,
        lastLeft,
        lastRight,
        sampleClass: "finite-bipolar-nonzero-stereo" as const,
        passed: true as const,
      }),
    });
  }

  inspectSeams(
    seamLatentFrames: readonly number[],
  ): readonly Readonly<Record<string, unknown>>[] {
    return Object.freeze(seamLatentFrames.map((latentFrame) => {
      const audioFrame = latentFrame * OPT_0054_0066_HOP_LENGTH;
      const firstElement = (audioFrame - 2) * OPT_0054_0066_AUDIO_CHANNELS;
      const bytes = this.readBlock(
        firstElement * Float32Array.BYTES_PER_ELEMENT,
        4 * OPT_0054_0066_AUDIO_CHANNELS * Float32Array.BYTES_PER_ELEMENT,
      );
      const values = new Float32Array(bytes.buffer);
      const channels = [0, 1].map((channel) => {
        const before2 = values[channel]!;
        const before = values[2 + channel]!;
        const at = values[4 + channel]!;
        const after = values[6 + channel]!;
        const finite = [before2, before, at, after].every(Number.isFinite);
        return Object.freeze({
          channel,
          valueBefore2: before2,
          valueBefore: before,
          valueAt: at,
          valueAfter: after,
          valueJump: Math.abs(at - before),
          firstDifferenceJump: Math.abs((after - at) - (before - before2)),
          finite,
        });
      });
      if (channels.some((channel) => !channel.finite)) {
        throw new Error("OPT-0054/0066 seam continuity is non-finite");
      }
      return Object.freeze({
        latentFrame,
        audioFrame,
        channels: Object.freeze(channels),
        finite: true,
      });
    }));
  }

  async remove(): Promise<void> {
    if (this.removed) return;
    if (!this.closed) {
      this.access.close();
      this.closed = true;
    }
    await this.root.removeEntry(this.directoryName, { recursive: true });
    this.removed = true;
  }

  snapshot(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      arm: this.arm,
      directoryNamePrefix: "ace-opt-0054-0066-c4500-",
      closed: this.closed,
      removed: this.removed,
    });
  }

  private requireOpen(): void {
    if (this.closed || this.removed) {
      throw new Error("OPT-0054/0066 raw artifact is closed");
    }
  }
}

class GateExecutionError extends Error {
  constructor(
    readonly original: unknown,
    readonly cleanup: Readonly<Record<string, unknown>>,
  ) {
    super(errorText(original));
    this.name = "GateExecutionError";
  }
}

let active: Readonly<{
  controller: AbortController;
  promise: Promise<void>;
}> | undefined;
let disposeAfterActive = false;

function installWorker(): void {
  globalThis.addEventListener(
    "message",
    (event: MessageEvent<Opt00540066WorkerCommand>) => {
      const command = event.data;
      if (command.type === "cancel" || command.type === "dispose") {
        if (command.type === "dispose") disposeAfterActive = true;
        active?.controller.abort(new DOMException(
          command.type === "dispose"
            ? "OPT-0054/0066 worker disposed"
            : "OPT-0054/0066 run cancelled by page",
          "AbortError",
        ));
        if (command.type === "dispose" && active === undefined) {
          postWorker({ type: "disposed" });
          disposeAfterActive = false;
        }
        return;
      }
      if (active !== undefined) {
        postWorker({
          type: "error",
          message: "InvalidStateError: OPT-0054/0066 gate is already running",
        });
        return;
      }
      const controller = new AbortController();
      const promise = runAndReport(controller.signal);
      active = Object.freeze({ controller, promise });
      void promise.finally(() => {
        if (active?.promise === promise) active = undefined;
        if (disposeAfterActive) {
          postWorker({ type: "disposed" });
          disposeAfterActive = false;
        }
      });
    },
  );
}

async function runAndReport(signal: AbortSignal): Promise<void> {
  try {
    const receipt = await runGateWithCleanup(signal);
    postWorker({ type: "result", receipt });
  } catch (error) {
    const cleanup = error instanceof GateExecutionError
      ? error.cleanup
      : undefined;
    const original = error instanceof GateExecutionError
      ? error.original
      : error;
    const cancelled = isAbortError(original);
    postWorker({
      type: cancelled ? "cancelled" : "error",
      message: errorText(original),
      receipt: Object.freeze({
        schema: OPT_0054_0066_LONG_SCHEMA,
        protocolId: OPT_0054_0066_LONG_PROTOCOL_ID,
        experimentAssociations: OPT_0054_0066_EXPERIMENT_ASSOCIATIONS,
        status: cancelled ? "cancelled" : "failed",
        error: serializeError(original),
        ...(cleanup === undefined ? {} : { cleanup }),
        productionDefaultChanged: false,
      }),
    });
  }
}

async function runGateWithCleanup(
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  const plan = planOpt00540066LongGate();
  const artifacts = new Set<Opt00540066RawArtifact>();
  const ownerTracker = createOwnerTracker();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let audit: DeviceResourceAudit | undefined;
  let body: Readonly<Record<string, unknown>> | undefined;
  let failure: unknown;
  try {
    signal.throwIfAborted();
    postProgress("authenticating deterministic C4500 latent");
    const fixtureBytes = createAceOpt0011LatentFixture(
      OPT_0054_0066_LATENT_FRAMES,
    );
    if (
      fixtureBytes.byteLength !== OPT_0054_0066_LATENT_BYTES ||
      await sha256Bytes(fixtureBytes) !== OPT_0054_0066_LATENT_SHA256
    ) {
      throw new Error("OPT-0054/0066 C4500 latent identity changed");
    }
    const fixture = new Float32Array(OPT_0054_0066_LATENT_ELEMENTS);
    fixture.set(new Float32Array(
      fixtureBytes.buffer,
      fixtureBytes.byteOffset,
      OPT_0054_0066_LATENT_ELEMENTS,
    ));

    postProgress("authenticating revision-6 and revision-7 VAE packages");
    const cache = await AceOpfsModelCache.open();
    const revision6 = await acquirePackage(
      await authenticatePackage(6),
      cache,
      signal,
    );
    const revision7 = await acquirePackage(
      await authenticatePackage(7),
      cache,
      signal,
    );
    signal.throwIfAborted();

    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      requiredLimits: {
        maxBufferSize: plan.candidateMemory.workspaceBufferBytes,
        maxStorageBufferBindingSize: plan.candidateMemory.workspaceBufferBytes,
      },
      signal,
      onRuntimeEvent: (runtimeEvent) => runtimeEvents.push(runtimeEvent),
    });
    requireDevice(context, plan.candidateMemory.workspaceBufferBytes);
    audit = new DeviceResourceAudit(context.device);

    postProgress("running deterministic mid-graph cancellation probe");
    const cancellation = await runCancellationProbe({
      context,
      device: audit.device,
      pkg: revision7,
      fixture,
      plan: plan.candidate,
      memory: plan.candidateMemory,
      ownerTracker,
      audit,
      artifacts,
      signal,
    });

    const executions: ExecutionResult[] = [];
    postProgress("1/3 sequential full waveform: revision-6 OPT-0028 C512");
    executions.push(await executeSuccessfulArm({
      arm: "rev6-opt0028-c512-control",
      context,
      device: audit.device,
      pkg: revision6,
      fixture,
      plan: plan.control,
      gatePlan: plan,
      memory: plan.controlMemory,
      topologyPlan: plan.controlTopology,
      ownerTracker,
      audit,
      artifacts,
      signal,
    }));
    postProgress("2/3 sequential full waveform: revision-7 OPT-0066 C2378");
    executions.push(await executeSuccessfulArm({
      arm: "rev7-opt0066-c2378-candidate",
      context,
      device: audit.device,
      pkg: revision7,
      fixture,
      plan: plan.candidate,
      gatePlan: plan,
      memory: plan.candidateMemory,
      topologyPlan: plan.candidateTopology,
      ownerTracker,
      audit,
      artifacts,
      signal,
    }));
    postProgress("3/3 sequential full waveform: revision-7 candidate repeat");
    executions.push(await executeSuccessfulArm({
      arm: "rev7-opt0066-c2378-repeat",
      context,
      device: audit.device,
      pkg: revision7,
      fixture,
      plan: plan.candidate,
      gatePlan: plan,
      memory: plan.candidateMemory,
      topologyPlan: plan.candidateTopology,
      ownerTracker,
      audit,
      artifacts,
      signal,
    }));
    const [control, candidate, repeat] = executions;
    if (control === undefined || candidate === undefined || repeat === undefined) {
      throw new Error("OPT-0054/0066 complete arm inventory changed");
    }
    signal.throwIfAborted();
    postProgress("bounded OPFS control/candidate waveform and seam comparison");
    const controlToCandidate = await compareRawArtifacts(
      control.raw,
      candidate.raw,
      plan,
      signal,
    );
    postProgress("bounded OPFS candidate repeat raw-U32 determinism comparison");
    const candidateToRepeat = await compareRawArtifacts(
      candidate.raw,
      repeat.raw,
      plan,
      signal,
    );
    const scanClassStable = candidate.scan.sampleClass ===
        repeat.scan.sampleClass &&
      JSON.stringify(candidate.scan) === JSON.stringify(repeat.scan);
    const statsStable = JSON.stringify(candidate.stats) ===
      JSON.stringify(repeat.stats);
    const topologyStable = JSON.stringify(candidate.topology) ===
      JSON.stringify(repeat.topology);
    const deterministic = candidate.sha256 === repeat.sha256 &&
      candidateToRepeat.rawU32Exact &&
      candidateToRepeat.u32MismatchCount === 0 && scanClassStable &&
      statsStable && topologyStable;
    const waveformPassed = allOpt00540066MetricsPassed(
      controlToCandidate.metrics,
    );
    const seamsPassed = controlToCandidate.seamsFinite &&
      candidateToRepeat.seamsFinite &&
      [...control.seamContinuity, ...candidate.seamContinuity,
        ...repeat.seamContinuity].every((entry) => entry["finite"] === true);
    const ownership = snapshotOwnerTracker(ownerTracker);
    const auditBeforeCleanup = audit.snapshot();
    if (
      control.sha256 !== OPT_0054_0066_CONTROL_RAW_SHA256 ||
      !deterministic || !waveformPassed || !seamsPassed ||
      ownership["peakLiveOwners"] !== 1 ||
      ownership["noLiveOwners"] !== true ||
      auditBeforeCleanup["passed"] !== true || runtimeEvents.length !== 0
    ) {
      throw new Error("OPT-0054/0066 long-waveform correctness gate failed");
    }
    await Promise.all(executions.map((execution) => execution.raw.remove()));
    const rawArtifactsAfterRemoval = Object.freeze(
      [...artifacts].map((artifact) => artifact.snapshot()),
    );
    if (rawArtifactsAfterRemoval.some((entry) => entry["removed"] !== true)) {
      throw new Error("OPT-0054/0066 did not remove every raw OPFS artifact");
    }
    body = Object.freeze({
      schema: OPT_0054_0066_LONG_SCHEMA,
      protocolId: OPT_0054_0066_LONG_PROTOCOL_ID,
      experimentAssociations: OPT_0054_0066_EXPERIMENT_ASSOCIATIONS,
      status: "passed",
      decision: "ready-revision7-c2378-long-waveform-gate-passed",
      fixture: Object.freeze({
        latentFrames: OPT_0054_0066_LATENT_FRAMES,
        channels: OPT_0054_0066_LATENT_CHANNELS,
        elementCount: OPT_0054_0066_LATENT_ELEMENTS,
        byteLength: OPT_0054_0066_LATENT_BYTES,
        sha256: OPT_0054_0066_LATENT_SHA256,
      }),
      packages: Object.freeze({
        revision6: packageReceipt(revision6),
        revision7: packageReceipt(revision7),
        manifestsAndImmutableOpfsFilesMayCoexist: true,
        gpuWeightPhasesAndBackendsMayCoexist: false,
      }),
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        adapterInfo: context.capabilities.adapterInfo,
        deviceFeatures: context.capabilities.deviceFeatures,
        deviceLimits: context.capabilities.deviceLimits,
        requestedLimits: context.capabilities.requestedLimits,
      }),
      protocol: Object.freeze({
        order: OPT_0054_0066_ARM_ORDER,
        dedicatedWorker: true,
        stockChromeWebGpuOnly: true,
        experimentalBrowserFlags: false,
        timestampQueries: false,
        webNn: false,
        quantaPerCommandBuffer: OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
        oneOutstandingCommandBuffer: true,
        queueDrainAfterEveryCommandBuffer: true,
        realQueueEmptyMillisecondsBetweenCommandBuffers: 1,
        sequentialGpuPackageOwnership: true,
        maximumSimultaneousGpuPackageOwners: 1,
        fullRawWaveformStoredOnlyInOpfs: true,
        rawComparisonBlockBytes: OPT_0054_0066_RAW_BLOCK_BYTES,
        noFullWaveformArrayBuffer: true,
        seamNeighborhoodMetricsDescriptiveOnly: true,
        seamGate: "finite-neighborhoods-and-explicit-continuity-samples",
        performanceTimingPerformed: false,
        executionOrderIsNotPerformanceEvidence: true,
      }),
      coverage: coverageReceipt(plan),
      memory: Object.freeze({
        control: plan.controlMemory,
        candidate: plan.candidateMemory,
        candidateBelowFourGigabytes:
          plan.candidateMemory.accountedGpuBytes <
            OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
      }),
      cancellation,
      executions: Object.freeze(executions.map(executionReceipt)),
      comparisons: Object.freeze({
        controlToCandidate,
        candidateToRepeat,
        acceptedRevision6ControlRawSha256:
          OPT_0054_0066_CONTROL_RAW_SHA256,
        candidateDeterministic: deterministic,
        scanClassStable,
        statsStable,
        topologyStable,
        waveformPassed,
        seamsPassed,
        numericalEnvelopeAuthority: "OPT-0066-frozen-OPT-0044-envelope",
      }),
      ownership,
      resourceAuditBeforeDeviceDestroy: auditBeforeCleanup,
      rawArtifactsAfterRemoval,
      runtimeEvents: Object.freeze([...runtimeEvents]),
      productionDefaultChanged: false,
      productionProfileChanged: false,
      productSelectionAuthorized: false,
      listeningApprovalStillRequired: true,
      under60SecondClaim: false,
      performanceClaim: null,
    });
  } catch (error) {
    failure = error;
  }

  const cleanup = await cleanupGate(
    context,
    audit,
    ownerTracker,
    artifacts,
    failure,
  );
  if (failure !== undefined) throw new GateExecutionError(failure, cleanup);
  if (body === undefined || cleanup["passed"] !== true) {
    throw new GateExecutionError(
      new Error("OPT-0054/0066 final cleanup gate failed"),
      cleanup,
    );
  }
  return Object.freeze({ ...body, cleanup });
}

async function executeSuccessfulArm(input: Readonly<{
  arm: Exclude<Opt00540066Arm, "rev7-cancellation-probe">;
  context: AceWebGpuDeviceContext;
  device: GPUDevice;
  pkg: PreparedPackage;
  fixture: Float32Array<ArrayBuffer>;
  plan: AceVaeChunkedDecodePlan;
  gatePlan: Opt00540066LongGatePlan;
  memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  topologyPlan: Opt00540066ArmTopologyPlan;
  ownerTracker: OwnerTracker;
  audit: DeviceResourceAudit;
  artifacts: Set<Opt00540066RawArtifact>;
  signal: AbortSignal;
}>): Promise<ExecutionResult> {
  const observer = new ProgressObserver();
  let raw: Opt00540066RawArtifact | undefined;
  const owned = await withOwnedBackend({
    context: input.context,
    device: input.device,
    pkg: input.pkg,
    fixture: input.fixture,
    plan: input.plan,
    memory: input.memory,
    ownerTracker: input.ownerTracker,
    audit: input.audit,
    arm: input.arm,
    signal: input.signal,
    onProgress: observer.onProgress,
    use: async (backend) => {
      const topology = backend.captureDispatchTopology();
      const backendValidation = validateBackend(
        input.arm,
        backend,
        topology,
        input.memory,
        input.topologyPlan,
      );
      raw = await Opt00540066RawArtifact.create(input.arm, input.plan);
      input.artifacts.add(raw);
      let trace: AceVaeRawStreamTrace | undefined;
      const stats = await streamAceVaeRawChunks(
        input.plan,
        backend,
        raw.sink,
        {
          signal: input.signal,
          onTrace: (value) => {
            if (trace !== undefined) {
              throw new Error("OPT-0054/0066 emitted duplicate raw trace");
            }
            trace = value;
          },
        },
      );
      raw.finish();
      if (
        trace === undefined ||
        stats.outputInterleavedElements !== OPT_0054_0066_OUTPUT_ELEMENTS ||
        stats.finiteSamples !== OPT_0054_0066_OUTPUT_ELEMENTS ||
        stats.windowsDecoded !== input.plan.windows.length ||
        stats.cooperativeIdleMs !== input.plan.windows.length - 1 ||
        !Number.isFinite(stats.peak) || stats.peak <= 0
      ) {
        throw new Error(`OPT-0054/0066 ${input.arm} raw stream changed`);
      }
      const progress = observer.finish(
        input.plan,
        topology,
        input.topologyPlan,
      );
      return Object.freeze({
        topology,
        backendValidation,
        stats,
        trace,
        progress,
        memory: backend.memory,
      });
    },
  });
  if (raw === undefined) {
    throw new Error(`OPT-0054/0066 ${input.arm} omitted its OPFS artifact`);
  }
  const inspected = await raw.inspect(input.signal);
  const seams = input.arm === "rev6-opt0028-c512-control"
    ? input.gatePlan.controlSeams
    : input.gatePlan.candidateSeams;
  const seamContinuity = raw.inspectSeams(seams);
  return Object.freeze({
    arm: input.arm,
    raw,
    sha256: inspected.sha256,
    scan: inspected.scan,
    stats: owned.value.stats,
    trace: owned.value.trace,
    memory: owned.value.memory,
    topology: owned.value.topology,
    backendValidation: owned.value.backendValidation,
    progress: owned.value.progress,
    ownerLifecycle: owned.lifecycle,
    resources: owned.resources,
    seamContinuity,
  });
}

async function runCancellationProbe(input: Readonly<{
  context: AceWebGpuDeviceContext;
  device: GPUDevice;
  pkg: PreparedPackage;
  fixture: Float32Array<ArrayBuffer>;
  plan: AceVaeChunkedDecodePlan;
  memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  ownerTracker: OwnerTracker;
  audit: DeviceResourceAudit;
  artifacts: Set<Opt00540066RawArtifact>;
  signal: AbortSignal;
}>): Promise<Readonly<Record<string, unknown>>> {
  const local = new AbortController();
  const combined = AbortSignal.any([input.signal, local.signal]);
  let progressEventCount = 0;
  let callbackCountAfterAbort = 0;
  let abortRequestedAtProgressEvent: number | null = null;
  let raw: Opt00540066RawArtifact | undefined;
  const owned = await withOwnedBackend({
    ...input,
    arm: "rev7-cancellation-probe",
    signal: combined,
    onProgress: () => {
      progressEventCount += 1;
      if (local.signal.aborted) {
        callbackCountAfterAbort += 1;
        return;
      }
      abortRequestedAtProgressEvent = progressEventCount;
      local.abort(new DOMException(
        "OPT-0054/0066 deterministic cancellation probe",
        "AbortError",
      ));
    },
    use: async (backend) => {
      raw = await Opt00540066RawArtifact.create(
        "rev7-cancellation-probe",
        input.plan,
      );
      input.artifacts.add(raw);
      let rejection: unknown;
      try {
        await streamAceVaeRawChunks(input.plan, backend, raw.sink, {
          signal: combined,
        });
      } catch (error) {
        rejection = error;
      }
      input.signal.throwIfAborted();
      if (!isAbortError(rejection) || !local.signal.aborted) {
        throw new Error("OPT-0054/0066 cancellation probe did not abort");
      }
      await raw.remove();
      return Object.freeze({ rejection: serializeError(rejection) });
    },
  });
  await browserYield();
  if (
    abortRequestedAtProgressEvent !== 1 || progressEventCount !== 1 ||
    callbackCountAfterAbort !== 0 || raw?.snapshot()["removed"] !== true
  ) {
    throw new Error("OPT-0054/0066 cancellation boundary changed");
  }
  return Object.freeze({
    arm: "rev7-cancellation-probe",
    abortRequestedAtProgressEvent,
    finalProgressEventCount: progressEventCount,
    callbackCountAfterAbort,
    rejection: owned.value.rejection,
    partialRawArtifact: raw.snapshot(),
    ownerLifecycle: owned.lifecycle,
    resources: owned.resources,
    noLaterSubmitDrainMapCallbackOrRawPublication: true,
    passed: true,
  });
}

async function withOwnedBackend<Value>(input: Readonly<{
  context: AceWebGpuDeviceContext;
  device: GPUDevice;
  pkg: PreparedPackage;
  fixture: Float32Array<ArrayBuffer>;
  plan: AceVaeChunkedDecodePlan;
  memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  ownerTracker: OwnerTracker;
  audit: DeviceResourceAudit;
  arm: Opt00540066Arm;
  signal: AbortSignal;
  onProgress: (progress: AceVaeChunkGpuBackendProgress) => void;
  use: (backend: AceOpt0011Fp16VaeChunkGpuBackend) => Promise<Value>;
}>): Promise<Readonly<{
  readonly value: Value;
  readonly lifecycle: Readonly<Record<string, unknown>>;
  readonly resources: Readonly<Record<string, unknown>>;
}>> {
  const tracker = input.ownerTracker;
  if (tracker.active !== null || tracker.liveOwners !== 0) {
    throw new Error("OPT-0054/0066 refused co-resident VAE package owners");
  }
  input.signal.throwIfAborted();
  input.audit.beginArm(input.arm);
  tracker.active = input.arm;
  tracker.liveOwners = 1;
  tracker.peakLiveOwners = Math.max(tracker.peakLiveOwners, 1);
  tracker.created[input.arm] += 1;
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  let value: Value | undefined;
  let completed = false;
  let destroyPromisesIdentical = true;
  let operationError: unknown;
  try {
    phase = await AceGpuTensorPhase.load(
      input.device,
      input.pkg.loaded.manifest,
      input.pkg.acquired,
      ["vae"],
      {
        signal: input.signal,
        onProgress: (progress) => postProgress(
          `uploading ${input.arm} VAE ${progress.phaseFileIndex + 1}/` +
            `${progress.phaseFileCount}: ${progress.loadedPhaseBytes}/` +
            `${progress.totalPhaseBytes}`,
        ),
      },
    );
    if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
      throw new Error(`OPT-0054/0066 ${input.arm} resident bytes changed`);
    }
    const transferred = phase;
    phase = undefined;
    const common = {
      device: input.device,
      plan: input.plan,
      finalLatents: input.fixture,
      authenticatedPackage: input.pkg.loaded,
      ownedVaeWeights: transferred,
      maximumWindowFrames: input.memory.maximumWindowFrames,
      subgroupMinSize: 32 as const,
      subgroupMaxSize: 32 as const,
      quantaPerCommandBuffer: OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
      signal: input.signal,
      onProgress: input.onProgress,
    };
    backend = await AceOpt0011Fp16VaeChunkGpuBackend.create(
      input.arm === "rev6-opt0028-c512-control"
        ? {
            ...common,
            runtimeProfileId:
              ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
          }
        : {
            ...common,
            runtimeProfileId:
              ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
          },
    );
    value = await input.use(backend);
    completed = true;
  } catch (error) {
    operationError = error;
  }
  let cleanupError: unknown;
  try {
    if (backend !== undefined) {
      const reason = new DOMException(
        `OPT-0054/0066 ${input.arm} sequential owner complete`,
        "AbortError",
      );
      const first = backend.destroy(reason);
      const second = backend.destroy(reason);
      destroyPromisesIdentical = first === second;
      tracker.everyBackendDestroyIdempotent &&= destroyPromisesIdentical;
      await first;
    } else {
      phase?.destroy();
    }
    await input.context.device.queue.onSubmittedWorkDone();
  } catch (error) {
    cleanupError = error;
  } finally {
    tracker.destroyed[input.arm] += 1;
    tracker.liveOwners = 0;
    tracker.active = null;
  }
  let resources: Readonly<Record<string, unknown>> | undefined;
  try {
    resources = input.audit.finishArm(input.memory);
  } catch (error) {
    cleanupError ??= error;
  }
  if (operationError !== undefined || cleanupError !== undefined || !completed) {
    throw new AggregateError(
      [operationError, cleanupError].filter((value) => value !== undefined),
      `OPT-0054/0066 ${input.arm} execution or cleanup failed`,
    );
  }
  return Object.freeze({
    value: value!,
    lifecycle: Object.freeze({
      arm: input.arm,
      packageRevision: input.pkg.revision,
      packageManifestSha256: input.pkg.loaded.manifestSha256,
      phaseBackendAndBuffersDestroyedBeforeNextOwner: true,
      idempotentDestroyPromises: destroyPromisesIdentical,
      liveOwnersAfterDestroy: tracker.liveOwners,
    }),
    resources: resources!,
  });
}

async function compareRawArtifacts(
  control: Opt00540066RawArtifact,
  candidate: Opt00540066RawArtifact,
  plan: Opt00540066LongGatePlan,
  signal: AbortSignal,
): Promise<RawComparison> {
  const full = new Opt00540066StereoMetricAccumulator();
  const seamAccumulators = new Map(plan.comparisonSeams.map(({ label }) => [
    label,
    new Opt00540066StereoMetricAccumulator(),
  ]));
  let comparedU32WordCount = 0;
  let u32MismatchCount = 0;
  let firstU32MismatchIndex: number | null = null;
  for (
    let at = 0;
    at < OPT_0054_0066_OUTPUT_BYTES;
    at += OPT_0054_0066_RAW_BLOCK_BYTES
  ) {
    signal.throwIfAborted();
    const byteLength = Math.min(
      OPT_0054_0066_RAW_BLOCK_BYTES,
      OPT_0054_0066_OUTPUT_BYTES - at,
    );
    const controlBytes = control.readBlock(at, byteLength);
    const candidateBytes = candidate.readBlock(at, byteLength);
    const controlValues = new Float32Array(controlBytes.buffer);
    const candidateValues = new Float32Array(candidateBytes.buffer);
    full.add(controlValues, candidateValues);
    const controlWords = new Uint32Array(controlBytes.buffer);
    const candidateWords = new Uint32Array(candidateBytes.buffer);
    for (let index = 0; index < controlWords.length; index += 1) {
      if (controlWords[index] === candidateWords[index]) continue;
      u32MismatchCount += 1;
      firstU32MismatchIndex ??= comparedU32WordCount + index;
    }
    const blockStartElement = at / Float32Array.BYTES_PER_ELEMENT;
    const blockEndElement = blockStartElement + controlValues.length;
    for (const { label, latentFrame } of plan.comparisonSeams) {
      const startElement = Math.max(
        0,
        (latentFrame - OPT_0054_0066_SEAM_RADIUS_LATENT_FRAMES) *
          OPT_0054_0066_HOP_LENGTH * OPT_0054_0066_AUDIO_CHANNELS,
      );
      const endElement = Math.min(
        OPT_0054_0066_OUTPUT_ELEMENTS,
        (latentFrame + OPT_0054_0066_SEAM_RADIUS_LATENT_FRAMES) *
          OPT_0054_0066_HOP_LENGTH * OPT_0054_0066_AUDIO_CHANNELS,
      );
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
    comparedU32WordCount += controlWords.length;
    await browserYield();
  }
  const seamNeighborhoods = Object.freeze(Object.fromEntries(
    [...seamAccumulators].map(([label, accumulator]) => [
      label,
      accumulator.finish(),
    ]),
  ));
  if (comparedU32WordCount !== OPT_0054_0066_OUTPUT_ELEMENTS) {
    throw new Error("OPT-0054/0066 bounded comparison lost output coverage");
  }
  return Object.freeze({
    comparedU32WordCount,
    u32MismatchCount,
    firstU32MismatchIndex,
    rawU32Exact: u32MismatchCount === 0,
    metrics: full.finish(),
    seamNeighborhoods,
    seamsFinite: Object.values(seamNeighborhoods).every((metrics) =>
      metrics.joint.finite && metrics.left.finite && metrics.right.finite
    ),
  });
}

function validateBackend(
  arm: Exclude<Opt00540066Arm, "rev7-cancellation-probe">,
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  topology: AceOpt0011Fp16VaeDispatchTopologyReceipt,
  memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
  expected: Opt00540066ArmTopologyPlan,
): Readonly<Record<string, unknown>> {
  const control = arm === "rev6-opt0028-c512-control";
  const expectedProfile = control
    ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE
    : ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
  const expectedKernelTopology = control
    ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY
    : ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY;
  const topologyWindows = Object.freeze(topology.windows.map(
    reconcileOpt00540066CapturedWindowTopology,
  ));
  const predicates = Object.freeze({
    backendRuntimeProfileMatches:
      backend.runtimeProfileId === expectedProfile.id,
    backendKernelSetMatches:
      backend.kernelSetId === expectedProfile.kernelSetId,
    backendKernelTopologyMatches:
      JSON.stringify(backend.kernelTopology) ===
        JSON.stringify(expectedKernelTopology),
    backendMemoryMatches:
      JSON.stringify(backend.memory) === JSON.stringify(memory),
    uniqueWindowFramesMatch:
      JSON.stringify(topology.uniqueWindowFrames) ===
        JSON.stringify(expected.uniqueWindowFrames),
    capturedRuntimeProfileMatches:
      topology.runtimeProfileId === expectedProfile.id,
    capturedKernelSetMatches:
      topology.kernelSetId === expectedProfile.kernelSetId,
    everyWindowReconciled: topologyWindows.every((window) => window.passed),
  });
  const passed = Object.values(predicates).every(Boolean);
  const receipt = Object.freeze({
    arm,
    expected: Object.freeze({
      runtimeProfileId: expectedProfile.id,
      kernelSetId: expectedProfile.kernelSetId,
      kernelTopology: expectedKernelTopology,
      memory,
      uniqueWindowFrames: expected.uniqueWindowFrames,
    }),
    actual: Object.freeze({
      backendRuntimeProfileId: backend.runtimeProfileId,
      backendKernelSetId: backend.kernelSetId,
      backendKernelTopology: backend.kernelTopology,
      backendMemory: backend.memory,
      capturedRuntimeProfileId: topology.runtimeProfileId,
      capturedKernelSetId: topology.kernelSetId,
      uniqueWindowFrames: topology.uniqueWindowFrames,
      windows: topologyWindows,
    }),
    predicates,
    passed,
  });
  if (!passed) {
    throw new Error(
      `OPT-0054/0066 ${arm} topology or memory changed: ` +
        JSON.stringify(receipt),
    );
  }
  return receipt;
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
  const tensors = Object.values(loaded.manifest.tensors).filter((tensor) =>
    tensor.phase === "vae"
  );
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) => shardNames.has(file.name));
  const residentBytes = sum(files.map((file) => file.byteLength));
  if (
    loaded.manifestSha256 !== expectedManifestSha256 ||
    loaded.manifestByteLength !== expectedManifestBytes ||
    loaded.manifest.provenance.converterRevision !== expectedConverterRevision ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    shardNames.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) {
    throw new Error(
      `OPT-0054/0066 authenticated revision-${revision} package changed`,
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

async function acquirePackage(
  pkg: Omit<PreparedPackage, "acquired">,
  cache: AceOpfsModelCache,
  signal: AbortSignal,
): Promise<PreparedPackage> {
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({ ...pkg.loaded.manifest, files: pkg.files }),
    manifestUrl: pkg.loaded.manifestUrl,
    cache,
    signal,
    onFileProgress: (progress) => postProgress(
      `acquiring revision-${pkg.revision} VAE ` +
        `${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes}`,
    ),
  });
  if (
    acquired.files.size !== pkg.files.length ||
    acquired.plan.runtimeBytes !== pkg.residentBytes
  ) {
    throw new Error(
      `OPT-0054/0066 revision-${pkg.revision} acquisition changed`,
    );
  }
  return Object.freeze({ ...pkg, acquired: acquired.files });
}

async function cleanupGate(
  context: AceWebGpuDeviceContext | undefined,
  audit: DeviceResourceAudit | undefined,
  ownerTracker: OwnerTracker,
  artifacts: ReadonlySet<Opt00540066RawArtifact>,
  reason: unknown,
): Promise<Readonly<Record<string, unknown>>> {
  const errors: unknown[] = [];
  for (const artifact of artifacts) {
    try {
      await artifact.remove();
    } catch (error) {
      errors.push(error);
    }
  }
  try {
    await context?.device.queue.onSubmittedWorkDone();
  } catch (error) {
    errors.push(error);
  }
  context?.destroy();
  const ownership = snapshotOwnerTracker(ownerTracker);
  const resources = audit?.snapshot() ?? Object.freeze({ passed: true });
  const rawArtifacts = Object.freeze(
    [...artifacts].map((artifact) => artifact.snapshot()),
  );
  const passed = errors.length === 0 && ownership["noLiveOwners"] === true &&
    ownership["allCreatedOwnersDestroyed"] === true &&
    ownership["everyBackendDestroyIdempotent"] === true &&
    resources["passed"] === true &&
    rawArtifacts.every((artifact) => artifact["removed"] === true);
  return Object.freeze({
    passed,
    reason: reason === undefined ? null : serializeError(reason),
    errors: Object.freeze(errors.map(serializeError)),
    ownership,
    resources,
    rawArtifacts,
    everyRawOpfsDirectoryRemoved: rawArtifacts.every((artifact) =>
      artifact["removed"] === true
    ),
    deviceContextDestroyed: context !== undefined,
  });
}

function coverageReceipt(plan: Opt00540066LongGatePlan): Readonly<Record<string, unknown>> {
  const summarize = (chunk: AceVaeChunkedDecodePlan) => Object.freeze({
    chunkFrames: chunk.chunkFrames,
    overlapFrames: chunk.overlapFrames,
    strideFrames: chunk.strideFrames,
    windowCount: chunk.windows.length,
    maximumActualWindowFrames: chunk.maximumWindowFrames,
    windows: Object.freeze(chunk.windows.map((window) => Object.freeze({
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
    control: summarize(plan.control),
    candidate: summarize(plan.candidate),
    controlTopology: plan.controlTopology,
    candidateTopology: plan.candidateTopology,
    controlSeams: plan.controlSeams,
    candidateSeams: plan.candidateSeams,
    seamRadiusLatentFrames: OPT_0054_0066_SEAM_RADIUS_LATENT_FRAMES,
    exactOutputCoverageOnce: true,
  });
}

function executionReceipt(execution: ExecutionResult): Readonly<Record<string, unknown>> {
  return Object.freeze({
    arm: execution.arm,
    rawSha256: execution.sha256,
    scan: execution.scan,
    streamStats: execution.stats,
    boundedRawStreamTrace: execution.trace,
    memory: execution.memory,
    topology: execution.topology,
    backendValidation: execution.backendValidation,
    progress: execution.progress,
    seamContinuity: execution.seamContinuity,
    ownerLifecycle: execution.ownerLifecycle,
    resources: execution.resources,
  });
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
    files: Object.freeze(pkg.files.map((file) => Object.freeze({
      name: file.name,
      byteLength: file.byteLength,
      sha256: file.sha256,
    }))),
    authenticatedBeforeGpuUpload: true,
  });
}

function createOwnerTracker(): OwnerTracker {
  return {
    active: null,
    liveOwners: 0,
    peakLiveOwners: 0,
    created: {
      "rev7-cancellation-probe": 0,
      "rev6-opt0028-c512-control": 0,
      "rev7-opt0066-c2378-candidate": 0,
      "rev7-opt0066-c2378-repeat": 0,
    },
    destroyed: {
      "rev7-cancellation-probe": 0,
      "rev6-opt0028-c512-control": 0,
      "rev7-opt0066-c2378-candidate": 0,
      "rev7-opt0066-c2378-repeat": 0,
    },
    everyBackendDestroyIdempotent: true,
  };
}

function snapshotOwnerTracker(tracker: OwnerTracker): Readonly<Record<string, unknown>> {
  const created = Object.freeze({ ...tracker.created });
  const destroyed = Object.freeze({ ...tracker.destroyed });
  const createdOwnerCount = sum(Object.values(created));
  const destroyedOwnerCount = sum(Object.values(destroyed));
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

function requireDevice(
  context: AceWebGpuDeviceContext,
  requiredWorkspaceBytes: number,
): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== REQUIRED_SUBGROUP_SIZE ||
    info.subgroupMaxSize !== REQUIRED_SUBGROUP_SIZE ||
    context.device.limits.maxBufferSize < requiredWorkspaceBytes ||
    context.device.limits.maxStorageBufferBindingSize < requiredWorkspaceBytes
  ) {
    throw new Error(
      "OPT-0054/0066 requires stock fixed32 WebGPU and C2378 limits",
    );
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
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

function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "AbortError") return true;
  if (error instanceof AggregateError) {
    return error.errors.some(isAbortError);
  }
  return false;
}

function serializeError(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof AggregateError) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      errors: Object.freeze(error.errors.map(serializeError)),
    });
  }
  if (error instanceof Error || error instanceof DOMException) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
    });
  }
  return Object.freeze({ name: "UnknownError", message: String(error) });
}

function errorText(error: unknown): string {
  const serialized = serializeError(error);
  return `${String(serialized["name"])}: ${String(serialized["message"])}`;
}

function postProgress(message: string): void {
  postWorker({ type: "progress", message });
}

function postWorker(event: WorkerEvent): void {
  (globalThis as unknown as DedicatedWorkerGlobalScope).postMessage(event);
}

if (
  typeof WorkerGlobalScope !== "undefined" &&
  globalThis instanceof WorkerGlobalScope
) installWorker();
