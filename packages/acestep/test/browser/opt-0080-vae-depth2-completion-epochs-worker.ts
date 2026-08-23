/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import { createAceOpt0011LatentFixture } from
  "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
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
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import type { AceVaeChunkGpuBackendProgress } from
  "../../src/webgpu/vae-backend.js";
import {
  AceOpt0011Fp16VaeChunkGpuBackend,
  planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory,
  type AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
  type AceOpt0011Fp16VaeDispatchTopologyReceipt,
  type AceOpt0080VaeSchedulingEvidence,
  type AceOpt0080VaeSchedulingProfile,
} from "../../src/webgpu/vae-fp16-backend.js";
import { ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES } from
  "../../src/webgpu/vae-fp16-decoder.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../../src/webgpu/vae-fp16-package.js";
import { ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE } from
  "../../src/webgpu/vae-fp16-profile.js";
import { planAceVaeDecoder } from "../../src/webgpu/vae-decoder.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
} from "../../src/webgpu/vae-chunks.js";
import {
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_VAE_ARM_ORDER,
  OPT_0080_VAE_DECODER_QUANTA,
  OPT_0080_VAE_FIXTURE_SHA256,
  OPT_0080_VAE_LATENT_FRAMES,
  OPT_0080_VAE_MAXIMUM_WINDOW_FRAMES,
  OPT_0080_VAE_OUTPUT_ELEMENTS,
  OPT_0080_VAE_OVERLAP_FRAMES,
  OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER,
  OPT_0080_VAE_WAVEFORM_SHA256,
  requireOpt0080VaeThermalGate,
  requireOpt0080VaeThermalTrace,
  requireOpt0080VaeHeartbeat,
  requireOpt0080VaeTopology,
  type Opt0080VaeArmId,
  type Opt0080VaeCancellationEvidence,
  type Opt0080VaeHeartbeatCapture,
  type Opt0080VaeThermalGate,
  type Opt0080VaeThermalTrace,
  type Opt0080VaeTimingSample,
} from "./opt-0080-vae-depth2-completion-epochs-contract.js";
import {
  buildOpt0080VaeResult,
  type Opt0080VaeCorrectnessReceipt,
  type Opt0080VaeTimedArmResult,
} from "./opt-0080-vae-depth2-completion-epochs-result.js";

const REVISION7_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json";
const CPU_GUARD_WORDS = 64;
const CPU_COMPARE_BLOCK_WORDS = 262_144;
const CPU_GUARD_PATTERN = 0xa55a_0080;
const GPU_GUARD_BYTES = 256;
const GPU_GUARD_PATTERN = 0xa55a_0080;
const GPU_GUARD_EXPECTED_SHA256 =
  "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d";
const GPU_GUARDED_BUFFER_LABELS = Object.freeze([
  "ace-opt-0011-fp16-vae-staging-input",
  "ace-opt-0011-fp16-vae-decoder-input",
  "ace-opt-0011-fp16-vae-workspace-0",
  "ace-opt-0011-fp16-vae-workspace-1",
  "ace-opt-0011-fp16-vae-workspace-2",
  "ace-opt-0011-fp16-vae-output",
] as const);
const GPU_GUARD_ACTIVE_C2314_PLAN = planAceVaeDecoder(
  OPT_0080_VAE_LATENT_FRAMES,
);
const GPU_GUARD_ACTIVE_END_BYTES = Object.freeze({
  "ace-opt-0011-fp16-vae-staging-input":
    GPU_GUARD_ACTIVE_C2314_PLAN.inputElements * Float32Array.BYTES_PER_ELEMENT,
  "ace-opt-0011-fp16-vae-decoder-input":
    GPU_GUARD_ACTIVE_C2314_PLAN.inputElements * Uint16Array.BYTES_PER_ELEMENT,
  "ace-opt-0011-fp16-vae-workspace-0":
    GPU_GUARD_ACTIVE_C2314_PLAN.maximumActivationElements *
      Uint16Array.BYTES_PER_ELEMENT,
  "ace-opt-0011-fp16-vae-workspace-1":
    GPU_GUARD_ACTIVE_C2314_PLAN.maximumActivationElements *
      Uint16Array.BYTES_PER_ELEMENT,
  "ace-opt-0011-fp16-vae-workspace-2":
    GPU_GUARD_ACTIVE_C2314_PLAN.maximumActivationElements *
      Uint16Array.BYTES_PER_ELEMENT,
  "ace-opt-0011-fp16-vae-output":
    GPU_GUARD_ACTIVE_C2314_PLAN.outputElements * Float32Array.BYTES_PER_ELEMENT,
} satisfies Readonly<Record<
  typeof GPU_GUARDED_BUFFER_LABELS[number],
  number
>>);

type WorkerCommand =
  | Readonly<{ readonly type: "prepare"; readonly identity: Opt0018RunIdentity }>
  | Readonly<{
      readonly type: "run-arm";
      readonly armId: Opt0080VaeArmId;
      readonly thermalGate: Opt0080VaeThermalGate;
    }>
  | Readonly<{
      readonly type: "complete-thermal";
      readonly armId: Opt0080VaeArmId;
      readonly thermalTrace: Opt0080VaeThermalTrace;
      readonly heartbeat: Opt0080VaeHeartbeatCapture;
    }>
  | Readonly<{ readonly type: "dispose" }>;

interface WorkerEvent {
  readonly type:
    | "progress"
    | "ready-for-arm"
    | "arm-complete"
    | "gate-rejected"
    | "trace-rejected"
    | "gate-complete"
    | "failed"
    | "disposed";
  readonly message?: string;
  readonly armId?: Opt0080VaeArmId;
  readonly schedulingProfile?: AceOpt0080VaeSchedulingProfile;
  readonly order?: number;
  readonly readyAtEpochMilliseconds?: number;
  readonly settledAtEpochMilliseconds?: number;
  readonly sample?: Omit<Opt0080VaeTimingSample, "heartbeat">;
  readonly preflights?: Readonly<Record<string, unknown>>;
  readonly result?: Readonly<Record<string, unknown>>;
  readonly error?: Readonly<Record<string, unknown>>;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly acquired: ReadonlyMap<string, File>;
  readonly residentBytes: number;
}

interface DecodeExecution {
  readonly output: Float32Array<ArrayBuffer>;
  readonly outputSha256: string;
  readonly topology: AceOpt0080VaeSchedulingEvidence;
  readonly schedulingWallMs: number;
  readonly preSchedulingWallMs: number;
  readonly mapAndDetachWallMs: number;
  readonly decodeWindowWallMs: number;
  readonly epochWallSumMs: number;
  readonly scan: Readonly<Record<string, unknown>>;
  readonly guards: Readonly<Record<string, unknown>>;
}

interface PendingArm {
  readonly armId: Opt0080VaeArmId;
  readonly order: 0 | 1 | 2 | 3;
  readonly readyAtEpochMilliseconds: number;
  readonly launchedAtEpochMilliseconds: number;
  readonly settledAtEpochMilliseconds: number;
  readonly sampleBase: Omit<Opt0080VaeTimingSample, "heartbeat">;
  readonly gate: Opt0080VaeThermalGate;
  readonly guardEvidence: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly identity: Opt0018RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly audit: DeviceResourceAudit;
  readonly device: GPUDevice;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly pkg: PreparedPackage;
  readonly backend: AceOpt0011Fp16VaeChunkGpuBackend;
  readonly dispatchTopology: AceOpt0011Fp16VaeDispatchTopologyReceipt;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly fixture: Float32Array<ArrayBuffer>;
  readonly oracle: Float32Array<ArrayBuffer>;
  readonly correctness: Opt0080VaeCorrectnessReceipt;
  readonly cancellation: Opt0080VaeCancellationEvidence;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  readonly lifetime: AbortController;
  readonly timedArms: Opt0080VaeTimedArmResult[];
  pendingArm: PendingArm | undefined;
  readyAtEpochMilliseconds: number;
  nextArmIndex: number;
  cleanup: Readonly<Record<string, unknown>> | undefined;
  destroy(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

interface BufferRecord {
  readonly label: string;
  readonly size: number;
  readonly logicalSize: number;
  readonly buffer: GPUBuffer;
  readonly gpuGuarded: boolean;
  destroyed: boolean;
  destroyCalls: number;
  mapCalls: number;
  unmapCalls: number;
  mappedOrPending: boolean;
}

class DeviceResourceAudit {
  readonly device: GPUDevice;
  private readonly records: BufferRecord[] = [];
  private guardReadback: GPUBuffer | undefined;
  private maximumLiveBytes = 0;
  private maximumLiveCount = 0;
  private maximumMappedCount = 0;

  constructor(target: GPUDevice) {
    this.device = new Proxy(target, {
      get: (device, property) => {
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer => {
            const gpuGuarded = GPU_GUARDED_BUFFER_LABELS.includes(
              descriptor.label as typeof GPU_GUARDED_BUFFER_LABELS[number],
            );
            const physicalDescriptor = gpuGuarded
              ? {
                  ...descriptor,
                  size: Number(descriptor.size) + GPU_GUARD_BYTES,
                  usage: descriptor.usage | GPUBufferUsage.COPY_SRC |
                    GPUBufferUsage.COPY_DST,
                }
              : descriptor;
            return this.track(
              device.createBuffer(physicalDescriptor),
              descriptor,
              physicalDescriptor,
              gpuGuarded,
            );
          };
        }
        const value = Reflect.get(device, property, device) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(device)
          : value;
      },
    }) as GPUDevice;
  }

  snapshot(): Readonly<Record<string, unknown>> {
    const live = this.records.filter((record) => !record.destroyed);
    const mapped = this.records.filter((record) => record.mappedOrPending);
    return Object.freeze({
      createdBufferCount: this.records.length,
      createdBufferBytes: sum(this.records.map((record) => record.size)),
      destroyedBufferCount: this.records.length - live.length,
      liveBufferCount: live.length,
      liveBufferBytes: sum(live.map((record) => record.size)),
      maximumLiveBufferCount: this.maximumLiveCount,
      maximumLiveBufferBytes: this.maximumLiveBytes,
      mapCallCount: sum(this.records.map((record) => record.mapCalls)),
      unmapCallCount: sum(this.records.map((record) => record.unmapCalls)),
      mappedOrPendingBufferCount: mapped.length,
      maximumMappedBufferCount: this.maximumMappedCount,
      totalDestroyCallCount: sum(this.records.map((record) => record.destroyCalls)),
      everyBufferDestroyedExactlyOnce: this.records.length > 0 &&
        this.records.every((record) =>
          record.destroyed && record.destroyCalls === 1
        ),
      records: Object.freeze(this.records.map((record) => Object.freeze({
        label: record.label,
        size: record.size,
        logicalSize: record.logicalSize,
        gpuGuarded: record.gpuGuarded,
        destroyed: record.destroyed,
        destroyCalls: record.destroyCalls,
        mapCalls: record.mapCalls,
        unmapCalls: record.unmapCalls,
      }))),
    });
  }

  async initializeGpuGuards(signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    const regions = this.guardRegions();
    if (regions.length !== 2 * GPU_GUARDED_BUFFER_LABELS.length) {
      throw new Error(
        `OPT-0080 VAE expected ${2 * GPU_GUARDED_BUFFER_LABELS.length} ` +
          `guard regions, got ${regions.length}`,
      );
    }
    const canary = new Uint32Array(
      GPU_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
    );
    canary.fill(GPU_GUARD_PATTERN);
    for (const region of regions) {
      this.device.queue.writeBuffer(
        region.record.buffer,
        region.offset,
        canary,
      );
    }
    await this.device.queue.onSubmittedWorkDone();
    signal.throwIfAborted();
  }

  async readGpuGuards(signal: AbortSignal): Promise<Readonly<Record<string, unknown>>> {
    signal.throwIfAborted();
    const regions = this.guardRegions();
    if (regions.length !== 2 * GPU_GUARDED_BUFFER_LABELS.length) {
      throw new Error("OPT-0080 VAE guarded region inventory changed");
    }
    const checkedBytes = regions.length * GPU_GUARD_BYTES;
    this.guardReadback ??= this.device.createBuffer({
      label: "opt-0080-vae-guard-readback",
      size: checkedBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "opt-0080-vae-guard-readback",
    });
    for (const [index, region] of regions.entries()) {
      encoder.copyBufferToBuffer(
        region.record.buffer,
        region.offset,
        this.guardReadback,
        index * GPU_GUARD_BYTES,
        GPU_GUARD_BYTES,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    signal.throwIfAborted();
    await this.guardReadback.mapAsync(GPUMapMode.READ, 0, checkedBytes);
    let bytes: Uint8Array<ArrayBuffer>;
    try {
      signal.throwIfAborted();
      bytes = new Uint8Array(
        this.guardReadback.getMappedRange(0, checkedBytes).slice(0),
      ) as Uint8Array<ArrayBuffer>;
    } finally {
      if (this.guardReadback.mapState !== "unmapped") {
        this.guardReadback.unmap();
      }
    }
    const words = new Uint32Array(bytes.buffer);
    const firstMismatch = words.findIndex((word) => word !== GPU_GUARD_PATTERN);
    const sha256 = await sha256Hex(bytes);
    const spans = Object.freeze(regions.map((region, index) => {
      const firstWord = index * GPU_GUARD_BYTES /
        Uint32Array.BYTES_PER_ELEMENT;
      const count = GPU_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
      return Object.freeze({
        label: region.record.label,
        region: region.kind,
        offsetBytes: region.offset,
        activeC2314EndBytes: region.activeEnd,
        logicalC2378AllocationBytes: region.record.logicalSize,
        guardBytes: GPU_GUARD_BYTES,
        activeEndGuardFitsBeforeAllocationEnd:
          region.activeEnd + GPU_GUARD_BYTES <= region.record.logicalSize,
        passed: words.subarray(firstWord, firstWord + count).every((word) =>
          word === GPU_GUARD_PATTERN
        ),
      });
    }));
    const passed = firstMismatch < 0 &&
      sha256 === GPU_GUARD_EXPECTED_SHA256 &&
      spans.every((span) => span.passed);
    const receipt = Object.freeze({
      scheme: "c2314-active-end-and-c2378-allocation-end-guards-v1",
      guardedBufferCount: GPU_GUARDED_BUFFER_LABELS.length,
      guardedRegionCount: regions.length,
      activeEndGuardRegionCount: GPU_GUARDED_BUFFER_LABELS.length,
      allocationEndGuardRegionCount: GPU_GUARDED_BUFFER_LABELS.length,
      guardBytesPerRegion: GPU_GUARD_BYTES,
      checkedBytes,
      patternU32: GPU_GUARD_PATTERN,
      sha256,
      expectedSha256: GPU_GUARD_EXPECTED_SHA256,
      firstMismatchWord: firstMismatch < 0 ? null : firstMismatch,
      spans,
      diagnosticCommandBufferCount: 1,
      diagnosticCompletionFenceCount: 1,
      initializationQueueFenceCount: 1,
      excludedFromSchedulerTopologyAndTiming: true,
      passed,
    });
    if (!passed) {
      throw new Error(
        `OPT-0080 VAE physical GPU active/allocation-end guard changed: ` +
          JSON.stringify(receipt),
      );
    }
    return receipt;
  }

  destroyGpuGuardResources(): void {
    this.guardReadback?.destroy();
    this.guardReadback = undefined;
  }

  private track(
    buffer: GPUBuffer,
    logicalDescriptor: GPUBufferDescriptor,
    physicalDescriptor: GPUBufferDescriptor,
    gpuGuarded: boolean,
  ): GPUBuffer {
    const record: BufferRecord = {
      label: logicalDescriptor.label ?? "",
      size: Number(physicalDescriptor.size),
      logicalSize: Number(logicalDescriptor.size),
      buffer,
      gpuGuarded,
      destroyed: false,
      destroyCalls: 0,
      mapCalls: 0,
      unmapCalls: 0,
      mappedOrPending: logicalDescriptor.mappedAtCreation === true,
    };
    this.records.push(record);
    const destroy = buffer.destroy.bind(buffer);
    const mapAsync = buffer.mapAsync.bind(buffer);
    const unmap = buffer.unmap.bind(buffer);
    Object.defineProperties(buffer, {
      ...(gpuGuarded
        ? {
            size: {
              configurable: true,
              value: Number(logicalDescriptor.size),
            },
          }
        : {}),
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

  private guardedRecords(): readonly BufferRecord[] {
    return GPU_GUARDED_BUFFER_LABELS.map((label) => {
      const matches = this.records.filter((record) =>
        record.gpuGuarded && !record.destroyed && record.label === label
      );
      if (matches.length !== 1) {
        throw new Error(
          `OPT-0080 VAE expected one live guarded ${label}, got ` +
            `${matches.length}`,
        );
      }
      return matches[0]!;
    });
  }

  private guardRegions(): readonly Readonly<{
    readonly record: BufferRecord;
    readonly kind: "active-c2314-end" | "allocation-c2378-end";
    readonly offset: number;
    readonly activeEnd: number;
  }>[] {
    return Object.freeze(this.guardedRecords().flatMap((record) => {
      const label = record.label as typeof GPU_GUARDED_BUFFER_LABELS[number];
      const activeEnd = GPU_GUARD_ACTIVE_END_BYTES[label];
      if (
        !Number.isSafeInteger(activeEnd) || activeEnd <= 0 ||
        activeEnd + GPU_GUARD_BYTES > record.logicalSize
      ) {
        throw new Error(
          `OPT-0080 VAE ${label} active C2314 guard does not fit inside ` +
            `the logical C2378 allocation: ${activeEnd}+${GPU_GUARD_BYTES}>` +
            `${record.logicalSize}`,
        );
      }
      return [
        Object.freeze({
          record,
          kind: "active-c2314-end" as const,
          offset: activeEnd,
          activeEnd,
        }),
        Object.freeze({
          record,
          kind: "allocation-c2378-end" as const,
          offset: record.logicalSize,
          activeEnd,
        }),
      ];
    }));
  }

  private observe(): void {
    const live = this.records.filter((record) => !record.destroyed);
    const mapped = this.records.filter((record) => record.mappedOrPending);
    this.maximumLiveBytes = Math.max(
      this.maximumLiveBytes,
      sum(live.map((record) => record.size)),
    );
    this.maximumLiveCount = Math.max(this.maximumLiveCount, live.length);
    this.maximumMappedCount = Math.max(this.maximumMappedCount, mapped.length);
  }
}

interface ActiveCancellation {
  controller: AbortController;
  reason: DOMException;
  progressAtAbort: number;
  progressAfterAbort: number;
  terminalFence: Promise<void> | undefined;
  terminalFenceResolved: boolean;
  device: GPUDevice;
}

let activeCancellation: ActiveCancellation | undefined;

function installWorker(): void {
  let prepared: PreparedGate | undefined;
  let operation = Promise.resolve();
  globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
    if (event.data.type === "dispose") {
      prepared?.lifetime.abort(new DOMException(
        "OPT-0080 VAE screen disposed",
        "AbortError",
      ));
    }
    operation = operation.then(async () => {
      const message = event.data;
      if (message.type === "prepare") {
        if (prepared !== undefined) throw new Error("OPT-0080 VAE already prepared");
        prepared = await prepareGate(validateOpt0018RunIdentity(message.identity));
        postReady(prepared);
        return;
      }
      if (message.type === "dispose") {
        const retained = prepared;
        prepared = undefined;
        await retained?.destroy(new DOMException(
          "OPT-0080 VAE screen disposed",
          "AbortError",
        ));
        postWorker({ type: "disposed" });
        return;
      }
      if (prepared === undefined) {
        throw new Error("OPT-0080 VAE command arrived before preparation");
      }
      if (message.type === "run-arm") {
        await runArm(prepared, message.armId, message.thermalGate);
        return;
      }
      await completeThermal(
        prepared,
        message.armId,
        message.thermalTrace,
        message.heartbeat,
      );
      if (prepared.nextArmIndex === OPT_0080_VAE_ARM_ORDER.length) {
        const result = buildOpt0080VaeResult({
          runIdentity: prepared.identity,
          package: packageReceipt(prepared.pkg),
          device: deviceReceipt(prepared.context),
          dispatchTopology: prepared.dispatchTopology,
          memory: prepared.memory,
          correctness: prepared.correctness,
          preflightEvidence: prepared.preparationReceipt,
          cancellation: prepared.cancellation,
          arms: prepared.timedArms,
          lifecycle: prepared.cleanup!,
          runtimeEvents: prepared.runtimeEvents,
          rejectedSetupAttempts: Object.freeze([]),
        });
        postWorker({ type: "gate-complete", result });
      } else {
        postReady(prepared);
      }
    }).catch(async (error: unknown) => {
      const retained = prepared;
      prepared = undefined;
      let cleanup: Readonly<Record<string, unknown>> | undefined;
      let cleanupError: unknown;
      try {
        cleanup = await retained?.destroy(error);
      } catch (caught) {
        cleanupError = caught;
      }
      postWorker({
        type: "failed",
        error: Object.freeze({
          ...serializeOpt0018Failure(error, cleanupError),
          ...(cleanup === undefined ? {} : { cleanup }),
        }),
      });
    });
  });
}

async function prepareGate(identity: Opt0018RunIdentity): Promise<PreparedGate> {
  const lifetime = new AbortController();
  postProgress("authenticating the exact C2314 latent fixture");
  const fixtureBytes = createAceOpt0011LatentFixture(
    OPT_0080_VAE_LATENT_FRAMES,
  );
  if (
    fixtureBytes.byteLength !==
      OPT_0080_VAE_LATENT_FRAMES * 64 * Float32Array.BYTES_PER_ELEMENT ||
    await sha256Hex(fixtureBytes) !== OPT_0080_VAE_FIXTURE_SHA256
  ) throw new Error("OPT-0080 VAE C2314 fixture identity changed");
  const fixture = new Float32Array(
    fixtureBytes.buffer.slice(
      fixtureBytes.byteOffset,
      fixtureBytes.byteOffset + fixtureBytes.byteLength,
    ),
  ) as Float32Array<ArrayBuffer>;
  const plan = planAceVaeChunkedDecode(OPT_0080_VAE_LATENT_FRAMES, {
    chunkFrames: OPT_0080_VAE_MAXIMUM_WINDOW_FRAMES,
    overlapFrames: OPT_0080_VAE_OVERLAP_FRAMES,
  });
  if (
    plan.windows.length !== 1 || !plan.direct ||
    plan.outputInterleavedElements !== OPT_0080_VAE_OUTPUT_ELEMENTS
  ) throw new Error("OPT-0080 VAE C2314 direct plan changed");
  const memory = planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory(
    plan,
    256,
    OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER,
  );

  postProgress("authenticating and acquiring the physical OPT-0066 revision-7 VAE");
  const pkg = await authenticateAndAcquirePackage(lifetime.signal);
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let audit: DeviceResourceAudit | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      requiredLimits: {
        maxBufferSize: memory.workspaceBufferBytes + GPU_GUARD_BYTES,
        maxStorageBufferBindingSize: memory.workspaceBufferBytes,
      },
      signal: lifetime.signal,
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });
    requireDevice(context, memory);
    audit = new DeviceResourceAudit(context.device);
    postProgress("uploading the sole persistent revision-7 VAE owner");
    phase = await AceGpuTensorPhase.load(
      audit.device,
      pkg.loaded.manifest,
      pkg.acquired,
      ["vae"],
      {
        signal: lifetime.signal,
        onProgress: (progress) => postProgress(
          `uploading VAE ${progress.phaseFileIndex + 1}/` +
            `${progress.phaseFileCount}: ${progress.loadedPhaseBytes}/` +
            `${progress.totalPhaseBytes} bytes`,
        ),
      },
    );
    if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
      throw new Error("OPT-0080 VAE resident weight bytes changed");
    }
    const transferred = phase;
    phase = undefined;
    backend = await AceOpt0011Fp16VaeChunkGpuBackend.create({
      device: audit.device,
      plan,
      finalLatents: fixture,
      authenticatedPackage: pkg.loaded,
      ownedVaeWeights: transferred,
      maximumWindowFrames:
        ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
      quantaPerCommandBuffer: OPT_0080_VAE_QUANTA_PER_COMMAND_BUFFER,
      runtimeProfileId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      submissionPolicy: "depth1-epoch1",
      onProgress: onBackendProgress,
    });
    if (JSON.stringify(backend.memory) !== JSON.stringify(memory)) {
      throw new Error("OPT-0080 VAE backend memory differs from frozen plan");
    }
    const dispatchTopology = backend.captureDispatchTopology();
    requireDispatchTopology(dispatchTopology);

    postProgress("untimed control C2314 full-waveform oracle");
    const control = await executeDecode(
      backend,
      plan,
      "depth1-epoch1",
      lifetime.signal,
      audit,
    );
    requireOpt0080VaeTopology(control.topology, "depth1-epoch1");
    postProgress("untimed candidate C2314 raw-U32 exact comparison");
    const candidate = await executeDecode(
      backend,
      plan,
      "depth2-phase-epoch4",
      lifetime.signal,
      audit,
    );
    requireOpt0080VaeTopology(candidate.topology, "depth2-phase-epoch4");
    const candidateComparison = compareExactBounded(
      control.output,
      candidate.output,
    );
    if (
      control.outputSha256 !== candidate.outputSha256 ||
      candidateComparison.mismatchCount !== 0 ||
      candidateComparison.canariesPassed !== true
    ) throw new Error("OPT-0080 VAE untimed A/B waveform changed");

    postProgress("untimed candidate cancellation and post-cancel exact reuse probe");
    const cancellationBase = await runCancellation(
      backend,
      plan,
      audit.device,
      audit,
      lifetime.signal,
    );
    const postCancelProbe = await executeDecode(
      backend,
      plan,
      "depth1-epoch1",
      lifetime.signal,
      audit,
    );
    const probeComparison = compareExactBounded(
      control.output,
      postCancelProbe.output,
    );
    const postCancellationExactProbePassed =
      postCancelProbe.outputSha256 === control.outputSha256 &&
      probeComparison.mismatchCount === 0 &&
      probeComparison.canariesPassed;
    if (!postCancellationExactProbePassed) {
      throw new Error("OPT-0080 VAE post-cancellation owner reuse changed");
    }
    const cancellation = Object.freeze({
      ...cancellationBase,
      postCancellationExactProbePassed: true as const,
    });
    const correctness = Object.freeze({
      controlOutputSha256: control.outputSha256,
      candidateOutputSha256: candidate.outputSha256,
      postCancellationProbeSha256: postCancelProbe.outputSha256,
      comparedU32WordCount: OPT_0080_VAE_OUTPUT_ELEMENTS,
      controlCandidateU32MismatchCount: 0 as const,
      controlProbeU32MismatchCount: 0 as const,
      controlNonFiniteCount: 0 as const,
      controlNonzeroCount: Number(control.scan["nonzeroCount"]),
      boundedComparisonCanariesPassed: true as const,
      gpuGuardCanariesPassed: true as const,
      gpuGuardExpectedSha256: GPU_GUARD_EXPECTED_SHA256,
      gpuGuardedBufferCount: 6 as const,
      gpuGuardedRegionCount: 12 as const,
      gpuGuardCheckedBytesPerExecution: 3_072 as const,
      topologyExact: true as const,
      excludedFromTiming: true as const,
    }) satisfies Opt0080VaeCorrectnessReceipt;
    const preparationReceipt = Object.freeze({
      fixture: Object.freeze({
        latentFrames: OPT_0080_VAE_LATENT_FRAMES,
        sha256: OPT_0080_VAE_FIXTURE_SHA256,
        byteLength: fixtureBytes.byteLength,
      }),
      package: packageReceipt(pkg),
      device: deviceReceipt(context),
      memory,
      dispatchTopology,
      correctness,
      correctnessExecutions: Object.freeze({
        control: executionReceipt(control),
        candidate: executionReceipt(candidate),
        postCancellationProbe: executionReceipt(postCancelProbe),
      }),
      comparisons: Object.freeze({
        controlCandidate: candidateComparison,
        controlPostCancellationProbe: probeComparison,
      }),
      cancellation,
      persistentOwner: Object.freeze({
        deviceCount: 1,
        backendCount: 1,
        packageOwnerCount: 1,
        reusedAcrossCorrectnessCancellationAndTiming: true,
      }),
      readyAtEpochMilliseconds: Date.now(),
    });
    const retainedBackend = backend;
    backend = undefined;
    return createPreparedGate({
      identity,
      context,
      audit,
      device: audit.device,
      plan,
      memory,
      pkg,
      backend: retainedBackend,
      dispatchTopology,
      runtimeEvents,
      fixture,
      oracle: control.output,
      correctness,
      cancellation,
      preparationReceipt,
      lifetime,
      timedArms: [],
      pendingArm: undefined,
      readyAtEpochMilliseconds: 0,
      nextArmIndex: 0,
      cleanup: undefined,
    });
  } catch (error) {
    if (backend !== undefined) await backend.destroy(error);
    else phase?.destroy();
    audit?.destroyGpuGuardResources();
    context?.destroy();
    throw error;
  }
}

function createPreparedGate(
  input: Omit<PreparedGate, "destroy">,
): PreparedGate {
  let destroyPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  const gate: PreparedGate = {
    ...input,
    destroy(reason: unknown = new DOMException(
      "OPT-0080 VAE screen complete",
      "AbortError",
    )): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        const started = performance.now();
        await input.device.queue.onSubmittedWorkDone();
        const first = input.backend.destroy(reason);
        const second = input.backend.destroy(reason);
        const idempotentPromise = first === second;
        await first;
        await input.device.queue.onSubmittedWorkDone();
        input.audit.destroyGpuGuardResources();
        const resources = input.audit.snapshot();
        const passed = idempotentPromise &&
          resources["liveBufferCount"] === 0 &&
          resources["liveBufferBytes"] === 0 &&
          resources["mappedOrPendingBufferCount"] === 0 &&
          resources["everyBufferDestroyedExactlyOnce"] === true;
        input.context.destroy();
        const receipt = Object.freeze({
          passed,
          reason: serializeOpt0018Failure(reason),
          persistentDeviceBackendAndPackageOwnerCount: 1,
          fifoBackendUseWasSequential: true,
          queueDrainedBeforeBackendDestroy: true,
          backendDestroyPromiseIdempotent: idempotentPromise,
          resourcesBeforeDeviceDestroy: resources,
          zeroLiveBuffersBeforeDeviceDestroy:
            resources["liveBufferCount"] === 0,
          zeroMappedBuffersBeforeDeviceDestroy:
            resources["mappedOrPendingBufferCount"] === 0,
          deviceContextDestroyed: true,
          wallMs: performance.now() - started,
          completedAtEpochMilliseconds: Date.now(),
        });
        if (!passed) {
          throw new Error(
            `OPT-0080 VAE lifecycle audit failed: ${JSON.stringify(receipt)}`,
          );
        }
        return receipt;
      })();
      return destroyPromise;
    },
  };
  return gate;
}

function postReady(prepared: PreparedGate): void {
  if (prepared.pendingArm !== undefined) {
    throw new Error("OPT-0080 VAE refused overlapping arm readiness");
  }
  const expected = OPT_0080_VAE_ARM_ORDER[prepared.nextArmIndex];
  if (expected === undefined) {
    throw new Error("OPT-0080 VAE has no remaining timed arm");
  }
  const readyAtEpochMilliseconds = Date.now();
  prepared.readyAtEpochMilliseconds = readyAtEpochMilliseconds;
  postWorker({
    type: "ready-for-arm",
    armId: expected.armId,
    schedulingProfile: expected.schedulingProfile,
    order: expected.order,
    readyAtEpochMilliseconds,
    ...(expected.order === 0
      ? { preflights: prepared.preparationReceipt }
      : {}),
  });
}

async function runArm(
  prepared: PreparedGate,
  armId: Opt0080VaeArmId,
  suppliedGate: Opt0080VaeThermalGate,
): Promise<void> {
  if (prepared.pendingArm !== undefined) {
    throw new Error("OPT-0080 VAE refused overlapping timed arms");
  }
  const expected = OPT_0080_VAE_ARM_ORDER[prepared.nextArmIndex];
  if (expected === undefined || expected.armId !== armId) {
    throw new Error("OPT-0080 VAE ABBA order changed");
  }
  const launchedAtEpochMilliseconds = Date.now();
  let thermalGate: Opt0080VaeThermalGate;
  try {
    thermalGate = requireOpt0080VaeThermalGate(
      suppliedGate,
      prepared.readyAtEpochMilliseconds,
      launchedAtEpochMilliseconds,
    );
  } catch (error) {
    postWorker({
      type: "gate-rejected",
      armId,
      error: serializeOpt0018Failure(error),
    });
    return;
  }
  postProgress(`${armId}: decoding one timed C2314 window`);
  const execution = await executeDecode(
    prepared.backend,
    prepared.plan,
    expected.schedulingProfile,
    prepared.lifetime.signal,
    prepared.audit,
  );
  const comparison = compareExactBounded(prepared.oracle, execution.output);
  if (
    execution.outputSha256 !== prepared.correctness.controlOutputSha256 ||
    comparison.mismatchCount !== 0 || !comparison.canariesPassed
  ) throw new Error(`OPT-0080 VAE ${armId} timed waveform changed`);
  requireOpt0080VaeTopology(execution.topology, expected.schedulingProfile);
  let cleanupCompletedAt = Date.now();
  if (expected.order === 3) {
    prepared.cleanup = await prepared.destroy(new DOMException(
      "OPT-0080 VAE timed ABBA complete",
      "AbortError",
    ));
    cleanupCompletedAt = Number(
      prepared.cleanup["completedAtEpochMilliseconds"],
    );
  }
  const sampleBase = Object.freeze({
    armId,
    order: expected.order,
    schedulingProfile: expected.schedulingProfile,
    schedulingWallMs: execution.schedulingWallMs,
    preSchedulingWallMs: execution.preSchedulingWallMs,
    mapAndDetachWallMs: execution.mapAndDetachWallMs,
    decodeWindowWallMs: execution.decodeWindowWallMs,
    epochWallSumMs: execution.epochWallSumMs,
    outputSha256: execution.outputSha256,
    rawU32MismatchCount: 0 as const,
    topology: execution.topology,
  });
  prepared.pendingArm = Object.freeze({
    armId,
    order: expected.order,
    readyAtEpochMilliseconds: prepared.readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
    settledAtEpochMilliseconds: cleanupCompletedAt,
    sampleBase,
    gate: thermalGate,
    guardEvidence: execution.guards,
  });
  postWorker({
    type: "arm-complete",
    armId,
    settledAtEpochMilliseconds: cleanupCompletedAt,
    sample: sampleBase,
    result: Object.freeze({
      execution: executionReceipt(execution),
      comparison,
      finalCleanup: expected.order === 3 ? prepared.cleanup : null,
    }),
  });
}

async function completeThermal(
  prepared: PreparedGate,
  armId: Opt0080VaeArmId,
  suppliedTrace: Opt0080VaeThermalTrace,
  heartbeat: Opt0080VaeHeartbeatCapture,
): Promise<void> {
  const pending = prepared.pendingArm;
  if (pending === undefined || pending.armId !== armId) {
    throw new Error("OPT-0080 VAE thermal trace arm order changed");
  }
  let thermalTrace: Opt0080VaeThermalTrace;
  try {
    requireOpt0080VaeHeartbeat(heartbeat);
    if (
      heartbeat.completedAtEpochMilliseconds <
        pending.settledAtEpochMilliseconds
    ) throw new Error("OPT-0080 VAE heartbeat stopped before arm settlement");
    thermalTrace = requireOpt0080VaeThermalTrace(
      suppliedTrace,
      pending.gate,
      pending.settledAtEpochMilliseconds,
      Date.now(),
    );
  } catch (error) {
    postWorker({
      type: "trace-rejected",
      armId,
      error: serializeOpt0018Failure(error),
    });
    return;
  }
  prepared.timedArms.push(Object.freeze({
    armId,
    order: pending.order,
    readyAtEpochMilliseconds: pending.readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds: pending.launchedAtEpochMilliseconds,
    settledAtEpochMilliseconds: pending.settledAtEpochMilliseconds,
    sample: Object.freeze({ ...pending.sampleBase, heartbeat }),
    gate: pending.gate,
    trace: thermalTrace,
    guardEvidence: pending.guardEvidence,
  }));
  prepared.pendingArm = undefined;
  prepared.nextArmIndex += 1;
}

async function executeDecode(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  plan: AceVaeChunkedDecodePlan,
  schedulingProfile: AceOpt0080VaeSchedulingProfile,
  signal: AbortSignal,
  audit: DeviceResourceAudit,
): Promise<DecodeExecution> {
  const window = plan.windows[0]!;
  await audit.initializeGpuGuards(signal);
  let topology: AceOpt0080VaeSchedulingEvidence | undefined;
  let schedulingCompletedAt = 0;
  const startedAt = performance.now();
  const output = await backend.decodeWindow(window, signal, {
    schedulingProfile,
    onSchedulingEvidence: (evidence) => {
      if (topology !== undefined) {
        throw new Error("OPT-0080 VAE emitted duplicate scheduling evidence");
      }
      topology = evidence;
      schedulingCompletedAt = performance.now();
    },
  }) as Float32Array<ArrayBuffer>;
  const completedAt = performance.now();
  if (topology === undefined || schedulingCompletedAt < startedAt) {
    throw new Error("OPT-0080 VAE scheduling evidence was not emitted");
  }
  requireOpt0080VaeTopology(topology, schedulingProfile);
  const outputSha256 = await sha256Float32(output);
  if (outputSha256 !== OPT_0080_VAE_WAVEFORM_SHA256) {
    throw new Error(
      `OPT-0080 VAE output differs from committed C2314 waveform authority: ` +
        `${outputSha256}`,
    );
  }
  const guards = await audit.readGpuGuards(signal);
  const scan = scanOutput(output);
  if (
    output.length !== OPT_0080_VAE_OUTPUT_ELEMENTS ||
    scan["nonFiniteCount"] !== 0 || Number(scan["nonzeroCount"]) <= 0
  ) throw new Error("OPT-0080 VAE output completeness changed");
  const schedulingWallMs = schedulingCompletedAt - startedAt;
  const preSchedulingWallMs = schedulingWallMs - topology.schedulingWallMs;
  if (
    !Number.isFinite(preSchedulingWallMs) || preSchedulingWallMs < 0 ||
    topology.schedulingWallMs <= 0
  ) throw new Error("OPT-0080 VAE scheduling wall clocks disagree");
  return Object.freeze({
    output,
    outputSha256,
    topology,
    schedulingWallMs: topology.schedulingWallMs,
    preSchedulingWallMs,
    mapAndDetachWallMs: completedAt - schedulingCompletedAt,
    decodeWindowWallMs: completedAt - startedAt,
    epochWallSumMs: sum(topology.completionEpochs.map((epoch) =>
      epoch.submitThroughTrueDrainMs
    )),
    scan,
    guards,
  });
}

async function runCancellation(
  backend: AceOpt0011Fp16VaeChunkGpuBackend,
  plan: AceVaeChunkedDecodePlan,
  device: GPUDevice,
  audit: DeviceResourceAudit,
  lifetimeSignal: AbortSignal,
): Promise<Omit<Opt0080VaeCancellationEvidence,
  "postCancellationExactProbePassed">> {
  lifetimeSignal.throwIfAborted();
  const controller = new AbortController();
  const reason = new DOMException(
    "OPT-0080 VAE deliberate candidate cancellation",
    "AbortError",
  );
  const state = {
    controller,
    reason,
    progressAtAbort: 0,
    progressAfterAbort: 0,
    terminalFence: undefined as Promise<void> | undefined,
    terminalFenceResolved: false,
    device,
  };
  await audit.initializeGpuGuards(lifetimeSignal);
  activeCancellation = state;
  let schedulingEvidenceCallbackCount = 0;
  let outputPublished = false;
  let rejection: unknown;
  let terminalFenceResolvedAtRejection = false;
  const unhandled: unknown[] = [];
  const onUnhandled = (event: PromiseRejectionEvent): void => {
    unhandled.push(event.reason);
    event.preventDefault();
  };
  globalThis.addEventListener("unhandledrejection", onUnhandled);
  let abortAt = 0;
  try {
    await backend.decodeWindow(plan.windows[0]!, controller.signal, {
      schedulingProfile: "depth2-phase-epoch4",
      onSchedulingEvidence: () => {
        schedulingEvidenceCallbackCount += 1;
      },
    });
    outputPublished = true;
  } catch (error) {
    rejection = error;
    abortAt = cancellationAbortPerformanceMs;
    terminalFenceResolvedAtRejection = state.terminalFenceResolved;
  } finally {
    activeCancellation = undefined;
  }
  const rejectedAt = performance.now();
  await state.terminalFence;
  await device.queue.onSubmittedWorkDone();
  const guards = await audit.readGpuGuards(lifetimeSignal);
  await browserYield();
  globalThis.removeEventListener("unhandledrejection", onUnhandled);
  const evidence = Object.freeze({
    scope: "actual-c2314-vae-window" as const,
    schedulingProfile: "depth2-phase-epoch4" as const,
    abortedFromFirstProgressCallback: true as const,
    progressEventCountAtAbort: state.progressAtAbort as 1,
    progressEventCountAfterAbort: state.progressAfterAbort,
    schedulingEvidenceCallbackCount,
    rejectedWithExactAbortReason: rejection === reason,
    queueDrainedBeforeRejection: terminalFenceResolvedAtRejection,
    outputPublished,
    unhandledRejectionCount: unhandled.length as 0,
    abortThroughRejectionMs: rejectedAt - abortAt,
    gpuGuards: guards,
  });
  if (
    evidence.progressEventCountAtAbort !== 1 ||
    evidence.progressEventCountAfterAbort !== 0 ||
    evidence.schedulingEvidenceCallbackCount !== 0 ||
    evidence.rejectedWithExactAbortReason !== true ||
    evidence.queueDrainedBeforeRejection !== true ||
    evidence.outputPublished !== false || evidence.unhandledRejectionCount !== 0 ||
    !Number.isFinite(evidence.abortThroughRejectionMs) ||
    evidence.abortThroughRejectionMs < 0 ||
    evidence.abortThroughRejectionMs > 1_000
  ) throw new Error(
    `OPT-0080 VAE cancellation boundary changed: ${JSON.stringify(evidence)}`,
  );
  return Object.freeze({
    ...evidence,
    progressEventCountAfterAbort: 0 as const,
    schedulingEvidenceCallbackCount: 0 as const,
    rejectedWithExactAbortReason: true as const,
    queueDrainedBeforeRejection: true as const,
    outputPublished: false as const,
    unhandledRejectionCount: 0 as const,
  });
}

let cancellationAbortPerformanceMs = 0;

function onBackendProgress(_progress: AceVaeChunkGpuBackendProgress): void {
  const active = activeCancellation;
  if (active === undefined) return;
  if (!active.controller.signal.aborted) {
    active.progressAtAbort += 1;
    cancellationAbortPerformanceMs = performance.now();
    active.controller.abort(active.reason);
    active.terminalFence = active.device.queue.onSubmittedWorkDone().then(() => {
      active.terminalFenceResolved = true;
    });
  } else {
    active.progressAfterAbort += 1;
  }
}

function requireDispatchTopology(
  topology: AceOpt0011Fp16VaeDispatchTopologyReceipt,
): void {
  const window = topology.windows[0];
  if (
    topology.runtimeProfileId !==
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id ||
    topology.kernelSetId !==
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId ||
    topology.uniqueWindowFrames.length !== 1 ||
    topology.uniqueWindowFrames[0] !== OPT_0080_VAE_LATENT_FRAMES ||
    topology.windows.length !== 1 || window === undefined ||
    window.inputFrames !== OPT_0080_VAE_LATENT_FRAMES ||
    window.operationCount !== 88 ||
    window.graphQuantumCount !== OPT_0080_VAE_DECODER_QUANTA - 1 ||
    window.sequenceQuantumCount !== OPT_0080_VAE_DECODER_QUANTA ||
    window.operationQuantumCounts.length !== 88
  ) throw new Error("OPT-0080 VAE authenticated dispatch topology changed");
}

function scanOutput(output: Float32Array): Readonly<Record<string, unknown>> {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let positiveCount = 0;
  let negativeCount = 0;
  let peak = 0;
  for (const value of output) {
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (value > 0) positiveCount += 1;
    if (value < 0) negativeCount += 1;
    peak = Math.max(peak, Math.abs(value));
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    nonFiniteCount,
    nonzeroCount,
    positiveCount,
    negativeCount,
    peak,
    completeFiniteBipolarNonzero: output.length ===
        OPT_0080_VAE_OUTPUT_ELEMENTS && nonFiniteCount === 0 &&
      nonzeroCount > 0 && positiveCount > 0 && negativeCount > 0,
  });
}

function compareExactBounded(
  oracle: Float32Array,
  candidate: Float32Array,
): Readonly<{
  readonly comparedU32WordCount: number;
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
  readonly blockWords: number;
  readonly blockCount: number;
  readonly guardWordsPerSide: number;
  readonly guardPatternU32: number;
  readonly canariesPassed: boolean;
  readonly maximumComparisonScratchBytes: number;
}> {
  if (
    oracle.length !== OPT_0080_VAE_OUTPUT_ELEMENTS ||
    candidate.length !== OPT_0080_VAE_OUTPUT_ELEMENTS
  ) throw new Error("OPT-0080 VAE bounded comparison length changed");
  const left = new Uint32Array(
    oracle.buffer,
    oracle.byteOffset,
    oracle.length,
  );
  const right = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  const scratch = new Uint32Array(
    CPU_COMPARE_BLOCK_WORDS + 2 * CPU_GUARD_WORDS,
  );
  scratch.fill(CPU_GUARD_PATTERN);
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  let blockCount = 0;
  let canariesPassed = true;
  for (let at = 0; at < left.length; at += CPU_COMPARE_BLOCK_WORDS) {
    const count = Math.min(CPU_COMPARE_BLOCK_WORDS, left.length - at);
    scratch.fill(CPU_GUARD_PATTERN);
    scratch.set(right.subarray(at, at + count), CPU_GUARD_WORDS);
    for (let index = 0; index < count; index += 1) {
      if (left[at + index] === scratch[CPU_GUARD_WORDS + index]) continue;
      mismatchCount += 1;
      firstMismatchIndex ??= at + index;
    }
    canariesPassed &&= scratch.subarray(0, CPU_GUARD_WORDS).every((word) =>
      word === CPU_GUARD_PATTERN
    );
    canariesPassed &&= scratch.subarray(
      CPU_GUARD_WORDS + CPU_COMPARE_BLOCK_WORDS,
    ).every((word) => word === CPU_GUARD_PATTERN);
    blockCount += 1;
  }
  return Object.freeze({
    comparedU32WordCount: left.length,
    mismatchCount,
    firstMismatchIndex,
    blockWords: CPU_COMPARE_BLOCK_WORDS,
    blockCount,
    guardWordsPerSide: CPU_GUARD_WORDS,
    guardPatternU32: CPU_GUARD_PATTERN,
    canariesPassed,
    maximumComparisonScratchBytes: scratch.byteLength,
  });
}

function executionReceipt(execution: DecodeExecution): Readonly<Record<string, unknown>> {
  return Object.freeze({
    outputSha256: execution.outputSha256,
    outputElements: execution.output.length,
    outputBytes: execution.output.byteLength,
    schedulingWallMs: execution.schedulingWallMs,
    preSchedulingWallMs: execution.preSchedulingWallMs,
    mapAndDetachWallMs: execution.mapAndDetachWallMs,
    decodeWindowWallMs: execution.decodeWindowWallMs,
    epochWallSumMs: execution.epochWallSumMs,
    topology: execution.topology,
    scan: execution.scan,
    guards: execution.guards,
  });
}

async function authenticateAndAcquirePackage(
  signal: AbortSignal,
): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(
      REVISION7_MANIFEST_PATH,
      globalThis.location.href,
    ).href,
    expectedManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
    authenticatedVaeConverterRevision: 7,
    signal,
  });
  const tensors = Object.values(loaded.manifest.tensors).filter((tensor) =>
    tensor.phase === "vae"
  );
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) => shardNames.has(file.name));
  const residentBytes = sum(files.map((file) => file.byteLength));
  if (
    loaded.manifestSha256 !== ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES ||
    loaded.manifest.provenance.converterRevision !==
      ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    shardNames.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shardNames.has(name))
  ) throw new Error("OPT-0080 VAE authenticated revision-7 package changed");
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({ ...loaded.manifest, files }),
    manifestUrl: loaded.manifestUrl,
    cache: await AceOpfsModelCache.open(),
    signal,
    onFileProgress: (progress) => postProgress(
      `acquiring VAE ${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== files.length ||
    acquired.plan.runtimeBytes !== residentBytes
  ) throw new Error("OPT-0080 VAE package acquisition accounting changed");
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    acquired: acquired.files,
    residentBytes,
  });
}

function packageReceipt(pkg: PreparedPackage): Readonly<Record<string, unknown>> {
  return Object.freeze({
    revision: 7,
    physicalOptimizationIdentity: "OPT-0066",
    manifestPath: REVISION7_MANIFEST_PATH,
    manifestUrl: pkg.loaded.manifestUrl,
    manifestSha256: pkg.loaded.manifestSha256,
    manifestByteLength: pkg.loaded.manifestByteLength,
    converterRevision: pkg.loaded.manifest.provenance.converterRevision,
    tensorRecordCount: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    weightFileCount: pkg.files.length,
    weightFiles: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
    residentBytes: pkg.residentBytes,
    authenticatedAndAcquiredBeforeDeviceRequest: true,
  });
}

function deviceReceipt(context: AceWebGpuDeviceContext): Readonly<Record<string, unknown>> {
  return Object.freeze({
    adapterInfo: context.capabilities.adapterInfo,
    executionProfile: context.capabilities.executionProfile,
    schedulingProfile: context.capabilities.schedulingProfile,
    adapterFeatures: context.capabilities.adapterFeatures,
    deviceFeatures: context.capabilities.deviceFeatures,
    adapterLimits: context.capabilities.adapterLimits,
    deviceLimits: context.capabilities.deviceLimits,
    requestedLimits: context.capabilities.requestedLimits,
    shaderF16: context.device.features.has("shader-f16"),
    subgroups: context.device.features.has("subgroups"),
    subgroupMinSize: context.capabilities.adapterInfo.subgroupMinSize,
    subgroupMaxSize: context.capabilities.adapterInfo.subgroupMaxSize,
  });
}

function requireDevice(
  context: AceWebGpuDeviceContext,
  memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
): void {
  const info = context.capabilities.adapterInfo;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32 ||
    context.device.limits.maxBufferSize <
      memory.workspaceBufferBytes + GPU_GUARD_BYTES ||
    context.device.limits.maxStorageBufferBindingSize <
      memory.workspaceBufferBytes ||
    context.device.limits.maxStorageBuffersPerShaderStage < 8 ||
    context.device.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    context.device.limits.maxComputeWorkgroupSizeX < 256
  ) throw new Error("OPT-0080 VAE requires fixed32 C2314 WebGPU limits");
}

async function sha256Float32(values: Float32Array): Promise<string> {
  return await sha256Hex(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = bytes.buffer instanceof ArrayBuffer &&
      bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes as Uint8Array<ArrayBuffer>
    : Uint8Array.from(bytes) as Uint8Array<ArrayBuffer>;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", copy));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
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

if (
  typeof WorkerGlobalScope !== "undefined" &&
  globalThis instanceof WorkerGlobalScope
) installWorker();
