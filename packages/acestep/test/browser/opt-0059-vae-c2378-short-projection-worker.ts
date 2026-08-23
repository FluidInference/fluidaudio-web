/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />
/// <reference lib="webworker" />

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
  AceFifoGraphOwner,
  submitAceCommandBufferFactoriesCooperatively,
} from "../../src/runtime/scheduler.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  AceOpt0011Fp16VaeDecoderRuntime,
  type AceOpt0011Fp16VaeWindowBindings,
  type AceOpt0011Fp16VaeWindowDispatch,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  resolveAceOpt0054Fp16VaePackageBindings,
} from "../../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  planAceVaeDecoder,
  type AceVaeDecoderOperation,
} from "../../src/webgpu/vae-decoder.js";
import {
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import { compareOpt0066Raw } from
  "./opt-0066-vae-dual-k4-quality-c512-contract.js";
import {
  OPT_0059_EXPERIMENT_ID,
  OPT_0059_FIXTURE_SHA256,
  OPT_0059_MAIN_ORDER,
  OPT_0059_EDGE_ORDER,
  OPT_0059_MAXIMUM_LIVE_GPU_BYTES,
  OPT_0059_PROFILE_FAMILIES,
  OPT_0059_QUANTA_PER_COMMAND_BUFFER,
  OPT_0059_REVISION7_MANIFEST_PATH,
  OPT_0059_SCHEMA,
  OPT_0059_SHAPES,
  evaluateOpt0059Timing,
  parseOpt0059ThermalGate,
  planOpt0059Gate,
  planOpt0059Shape,
  type Opt0059ProfileFamily,
  type Opt0059Shape,
  type Opt0059ShapePlan,
  type Opt0059ThermalGate,
  type Opt0059TimingSample,
} from "./opt-0059-vae-c2378-short-projection-contract.js";

const GUARD_BYTES = 256;
const STORAGE_CANARY_U32 = 0xa55a_c33c;
const STORAGE_CANARY_SHA256 =
  "3dee710588de5d1031ddeb00a0150cbe5ed5f8851eb9233bb0b04edfbc09d138";
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_0059;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

type WorkerCommand =
  | Readonly<{ readonly type: "initialize"; readonly identity: Opt0018RunIdentity }>
  | Readonly<{ readonly type: "run"; readonly thermalGate: Opt0059ThermalGate }>
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
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly acquired: ReadonlyMap<string, File>;
  readonly residentBytes: number;
}

interface GuardedBinding {
  readonly label: string;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly logicalBytes: number;
}

interface SharedAllocation {
  readonly stagingInput: GuardedBinding;
  readonly decoderInput: GuardedBinding;
  readonly workspaces: readonly [GuardedBinding, GuardedBinding, GuardedBinding];
  readonly output: GuardedBinding;
  readonly guarded: readonly GuardedBinding[];
  readonly readback: GPUBuffer;
  readonly canaryReadback: GPUBuffer;
  readonly bindings: AceOpt0011Fp16VaeWindowBindings;
  destroy(): void;
}

type ProfileBucket = Opt0059ProfileFamily | "mixed";

interface ProfileQuantum {
  readonly family: Opt0059ProfileFamily | "ingress";
  readonly operationLabel: string;
  encode(pass: GPUComputePassEncoder): void;
}

interface PreparedShape {
  readonly inputFrames: Opt0059Shape;
  readonly plan: Opt0059ShapePlan;
  readonly base: AceOpt0011Fp16VaeWindowDispatch;
  readonly quanta: readonly ProfileQuantum[];
  readonly topology: Readonly<Record<string, unknown>>;
}

interface ExecutionResult {
  readonly output: Float32Array<ArrayBuffer>;
  readonly outputSha256: string;
  readonly sample: Opt0059TimingSample;
  readonly scan: Readonly<Record<string, unknown>>;
  readonly canaries: Readonly<Record<string, unknown>>;
  readonly scheduling: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly identity: Opt0018RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly device: GPUDevice;
  readonly lifetimeAbort: AbortController;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly audit: DeviceResourceAudit;
  readonly graphOwner: AceFifoGraphOwner;
  readonly pkg: PreparedPackage;
  readonly phase: AceGpuTensorPhase;
  readonly allocation: SharedAllocation;
  readonly runtime: AceOpt0011Fp16VaeDecoderRuntime;
  readonly fixtures: Readonly<Record<`${Opt0059Shape}`, Float32Array<ArrayBuffer>>>;
  readonly shapes: Readonly<Record<`${Opt0059Shape}`, PreparedShape>>;
  readonly sameProfileOracleSha256: Readonly<Record<`${Opt0059Shape}`, string>>;
  readonly prefill: Readonly<{ readonly pipeline: GPUComputePipeline; readonly bindGroup: GPUBindGroup; readonly wordCount: number }>;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  destroy(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

type BufferScope = "weights" | "activation" | "controls";

interface BufferRecord {
  readonly label: string;
  readonly scope: BufferScope;
  readonly size: number;
  destroyed: boolean;
  destroyCalls: number;
  mapCount: number;
  unmapCount: number;
  mapped: boolean;
  mapPending: boolean;
}

/** Audit the real buffers created by the single persistent benchmark owner. */
class DeviceResourceAudit {
  readonly device: GPUDevice;
  private scope: BufferScope = "weights";
  private readonly records = new Map<GPUBuffer, BufferRecord>();
  private maximumLiveBytes = 0;
  private maximumLiveCount = 0;
  private maximumMappedCount = 0;
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

  setScope(scope: BufferScope): void { this.scope = scope; }

  /** Benchmark-only lookup of one authenticated runtime-owned control. */
  requireLiveBuffer(label: string): GPUBuffer {
    const matches = [...this.records].filter(([, record]) =>
      !record.destroyed && record.label === label
    );
    if (matches.length !== 1) {
      throw new Error(
        `OPT-0059 expected one live ${label} buffer, got ${matches.length}`,
      );
    }
    return matches[0]![0];
  }

  snapshot(): Readonly<Record<string, unknown>> {
    const records = [...this.records.values()];
    const live = records.filter((record) => !record.destroyed);
    const mapped = records.filter((record) => record.mapped || record.mapPending);
    return Object.freeze({
      createdBufferCount: records.length,
      createdBufferBytes: sum(records.map((record) => record.size)),
      destroyedBufferCount: records.length - live.length,
      liveBufferCount: live.length,
      liveBufferBytes: sum(live.map((record) => record.size)),
      maximumLiveBufferCount: this.maximumLiveCount,
      maximumLiveBufferBytes: this.maximumLiveBytes,
      maximumLiveGpuBytesLimit: OPT_0059_MAXIMUM_LIVE_GPU_BYTES,
      belowFourGigabytes: this.maximumLiveBytes < OPT_0059_MAXIMUM_LIVE_GPU_BYTES,
      mapCount: sum(records.map((record) => record.mapCount)),
      unmapCount: sum(records.map((record) => record.unmapCount)),
      mappedOrPendingBufferCount: mapped.length,
      maximumMappedBufferCount: this.maximumMappedCount,
      mapOverlapDetected: this.mapOverlapDetected,
      totalDestroyCallCount: sum(records.map((record) => record.destroyCalls)),
      everyBufferDestroyedExactlyOnce: records.length > 0 && records.every(
        (record) => record.destroyed && record.destroyCalls === 1,
      ),
      scopes: Object.freeze(Object.fromEntries(
        (["weights", "activation", "controls"] as const).map((scope) => {
          const selected = records.filter((record) => record.scope === scope);
          return [scope, Object.freeze({
            count: selected.length,
            bytes: sum(selected.map((record) => record.size)),
            liveCount: selected.filter((record) => !record.destroyed).length,
            liveBytes: sum(selected.filter((record) => !record.destroyed)
              .map((record) => record.size)),
          })];
        }),
      )),
      records: Object.freeze(records.map((record) => Object.freeze({
        label: record.label,
        scope: record.scope,
        size: record.size,
        destroyed: record.destroyed,
        destroyCalls: record.destroyCalls,
        mapCount: record.mapCount,
        unmapCount: record.unmapCount,
      }))),
    });
  }

  private track(buffer: GPUBuffer, descriptor: GPUBufferDescriptor): GPUBuffer {
    const record: BufferRecord = {
      label: descriptor.label ?? "",
      scope: this.scope,
      size: Number(descriptor.size),
      destroyed: false,
      destroyCalls: 0,
      mapCount: 0,
      unmapCount: 0,
      mapped: descriptor.mappedAtCreation === true,
      mapPending: false,
    };
    this.records.set(buffer, record);
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
          record.mapped = false;
          record.mapPending = false;
          destroy();
        },
      },
      mapAsync: {
        configurable: true,
        value: async (...args: Parameters<GPUBuffer["mapAsync"]>) => {
          record.mapCount += 1;
          record.mapPending = true;
          this.observeMapped();
          try {
            await mapAsync(...args);
            record.mapPending = false;
            record.mapped = true;
            this.observeMapped();
          } catch (error) {
            record.mapPending = false;
            throw error;
          }
        },
      },
      unmap: {
        configurable: true,
        value: () => {
          record.unmapCount += 1;
          record.mapped = false;
          record.mapPending = false;
          unmap();
        },
      },
    });
    this.observeLive();
    this.observeMapped();
    return buffer;
  }

  private observeLive(): void {
    const live = [...this.records.values()].filter((record) => !record.destroyed);
    this.maximumLiveCount = Math.max(this.maximumLiveCount, live.length);
    this.maximumLiveBytes = Math.max(
      this.maximumLiveBytes,
      sum(live.map((record) => record.size)),
    );
  }

  private observeMapped(): void {
    const count = [...this.records.values()].filter((record) =>
      record.mapped || record.mapPending
    ).length;
    this.maximumMappedCount = Math.max(this.maximumMappedCount, count);
    if (count > 1) this.mapOverlapDetected = true;
  }
}

function installWorker(): void {
  let prepared: PreparedGate | undefined;
  let activeAbort: AbortController | undefined;
  let operation = Promise.resolve();
  globalThis.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
    if (event.data.type === "dispose") {
      activeAbort?.abort(new DOMException("OPT-0059 disposed", "AbortError"));
    }
    operation = operation.then(async () => {
      if (event.data.type === "initialize") {
        if (prepared !== undefined) throw new Error("OPT-0059 is already initialized");
        activeAbort = new AbortController();
        prepared = await prepareGate(
          validateOpt0018RunIdentity(event.data.identity),
          activeAbort,
        );
        postWorker({
          type: "ready-for-thermal-gate",
          message: "READY — all four exact-shape correctness gates passed; timing has not run",
          readyAtEpochMilliseconds: prepared.readyAtEpochMilliseconds,
          preparation: prepared.preparationReceipt,
        });
        return;
      }
      if (event.data.type === "dispose") {
        const retained = prepared;
        prepared = undefined;
        await retained?.destroy(new DOMException("OPT-0059 disposed", "AbortError"));
        activeAbort = undefined;
        return;
      }
      if (prepared === undefined) throw new Error("OPT-0059 timing requested before READY");
      const retained = prepared;
      prepared = undefined;
      let result: Readonly<Record<string, unknown>> | undefined;
      let failure: unknown;
      try {
        result = await runTimedGate(retained, event.data.thermalGate);
      } catch (error) {
        failure = error;
      }
      let cleanup: Readonly<Record<string, unknown>> | undefined;
      try {
        cleanup = await retained.destroy(failure);
      } catch (error) {
        failure = failure === undefined
          ? error
          : new AggregateError([failure, error], "OPT-0059 run and cleanup failed");
      }
      if (failure !== undefined) {
        postWorker({
          type: "failed",
          error: Object.freeze({
            ...serializeOpt0018Failure(failure),
            preparation: retained.preparationReceipt,
            ...(cleanup === undefined ? {} : { cleanup }),
          }),
        });
        activeAbort = undefined;
        return;
      }
      postWorker({
        type: "comparison-complete",
        result: Object.freeze({ ...result!, cleanup }),
      });
      activeAbort = undefined;
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
      activeAbort = undefined;
    });
  });
}

async function prepareGate(
  identity: Opt0018RunIdentity,
  lifetimeAbort: AbortController,
): Promise<PreparedGate> {
  const preparationStarted = performance.now();
  const gatePlan = planOpt0059Gate();
  if (
    gatePlan.maximumAllocation.maximumActualWindowFrames !== 2_314 ||
    !gatePlan.maximumAllocation.belowFourGigabytes ||
    gatePlan.sixC128K1MappingChanges.length !== 6
  ) throw new Error("OPT-0059 frozen resource or mapping plan changed");
  const storageLimitPlan = Object.freeze({
    guardBytesPerSide: GUARD_BYTES,
    maximumLogicalStorageBindingBytes:
      gatePlan.shapes["2314"].workspaceBytes,
    maximumPhysicalBufferBytes:
      gatePlan.maximumAllocation.guardedWorkspaceBytesEach,
  });
  if (
    storageLimitPlan.maximumPhysicalBufferBytes !==
      storageLimitPlan.maximumLogicalStorageBindingBytes +
        2 * storageLimitPlan.guardBytesPerSide
  ) throw new Error("OPT-0059 guarded storage limit plan changed");

  postProgress("authenticating four deterministic fixtures before WebGPU");
  const fixtures = await authenticateFixtures();
  postProgress("authenticating and acquiring the sole revision-7 VAE package");
  const pkg = await authenticateAndAcquirePackage(lifetimeAbort.signal);
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  const graphOwner = new AceFifoGraphOwner();
  let context: AceWebGpuDeviceContext | undefined;
  let audit: DeviceResourceAudit | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let allocation: SharedAllocation | undefined;
  let runtime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
  try {
    context = await requestAceWebGpuDevice({
      modelProfile: "raw-fp16",
      schedulingProfile: "cooperative",
      requiredFeatures: ["subgroups"],
      requiredLimits: {
        // maxBufferSize governs the complete physical allocation, including
        // both canaries. maxStorageBufferBindingSize governs only the logical
        // subrange exposed to shaders between those canaries.
        maxBufferSize: storageLimitPlan.maximumPhysicalBufferBytes,
        maxStorageBufferBindingSize:
          storageLimitPlan.maximumLogicalStorageBindingBytes,
      },
      onRuntimeEvent: (event) => runtimeEvents.push(event),
      signal: lifetimeAbort.signal,
    });
    requireDevice(context, storageLimitPlan);
    audit = new DeviceResourceAudit(context.device);
    audit.setScope("weights");
    phase = await AceGpuTensorPhase.load(
      audit.device,
      pkg.loaded.manifest,
      pkg.acquired,
      ["vae"],
      {
        signal: lifetimeAbort.signal,
        onProgress: (progress) => postProgress(
          `uploading sole revision-7 VAE owner ` +
            `${progress.phaseFileIndex + 1}/${progress.phaseFileCount}: ` +
            `${progress.loadedPhaseBytes}/${progress.totalPhaseBytes} bytes`,
        ),
      },
    );
    if (phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
      throw new Error("OPT-0059 resident revision-7 VAE bytes changed");
    }
    const packageBindings = resolveAceOpt0054Fp16VaePackageBindings(
      // Package identity is deliberately authenticated against the canonical
      // frame-neutral B256 graph. Exact C340/C448/C512/C2314 geometry belongs
      // only to createChunkDispatchSet below.
      planAceVaeDecoder(256),
      pkg.loaded,
      phase,
    );
    audit.setScope("activation");
    allocation = createSharedAllocation(audit.device, gatePlan, packageBindings);
    audit.setScope("controls");
    runtime = AceOpt0011Fp16VaeDecoderRuntime.create(audit.device, {
      runtimeProfileId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const preparedShapes: Array<readonly [`${Opt0059Shape}`, PreparedShape]> = [];
    for (const inputFrames of OPT_0059_SHAPES) {
      lifetimeAbort.signal.throwIfAborted();
      postProgress(`assembling exact C${inputFrames} dual-K4 topology`);
      const set = await runtime.createChunkDispatchSet(
        `opt-0059-c${inputFrames}`,
        inputFrames,
        ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
        allocation.bindings,
      );
      const base = set.dispatches.find((dispatch) =>
        dispatch.plan.inputFrames === inputFrames
      );
      if (
        base === undefined || set.dispatches.length !== 1 ||
        set.windows.length !== 1 || set.windows[0]?.dispatch !== base
      ) throw new Error(`OPT-0059 C${inputFrames} exact dispatch set changed`);
      const rangeControlBuffer = audit.requireLiveBuffer(
        `opt-0059-c${inputFrames}-window-${inputFrames}-` +
          "fp16-vae-range-controls",
      );
      const profiled = createProfiledQuanta(
        base,
        rangeControlBuffer,
      );
      preparedShapes.push([
        String(inputFrames) as `${Opt0059Shape}`,
        Object.freeze({
          inputFrames,
          plan: gatePlan.shapes[String(inputFrames) as `${Opt0059Shape}`],
          base,
          quanta: profiled.quanta,
          topology: profiled.topology,
        }),
      ]);
    }
    const shapes = Object.freeze(Object.fromEntries(preparedShapes)) as
      Readonly<Record<`${Opt0059Shape}`, PreparedShape>>;
    audit.setScope("activation");
    const prefill = await createOutputPrefill(
      audit.device,
      allocation.output.binding,
      gatePlan.shapes["2314"].outputBytes / FLOAT32_BYTES,
    );
    const liveAudit = audit.snapshot();
    if (
      liveAudit["liveBufferBytes"] !==
        gatePlan.maximumAllocation.plannedLiveGpuBytes ||
      liveAudit["belowFourGigabytes"] !== true
    ) {
      throw new Error(
        `OPT-0059 live GPU accounting differs from plan: ` +
          JSON.stringify(liveAudit),
      );
    }

    const correctness: Readonly<Record<string, unknown>>[] = [];
    const oracleHashes: Array<readonly [`${Opt0059Shape}`, string]> = [];
    for (const inputFrames of OPT_0059_SHAPES) {
      postProgress(
        `untimed C${inputFrames} revision-7 oracle/profile/repeat exact correctness`,
      );
      const fixture = fixtures[String(inputFrames) as `${Opt0059Shape}`];
      const preparedShape = shapes[String(inputFrames) as `${Opt0059Shape}`];
      const baseQuanta = Object.freeze(preparedShape.base.quanta.map((quantum) =>
        Object.freeze({
          family: familyForBaseQuantum(preparedShape.base, quantum.operationIndex),
          operationLabel: quantum.operationLabel,
          encode: (pass: GPUComputePassEncoder) => quantum.encode(pass),
        })
      ));
      const oracle = await executeShape({
        device: audit.device,
        graphOwner,
        allocation,
        prefill,
        shape: preparedShape,
        fixture,
        quanta: baseQuanta,
        measured: false,
        label: `correctness-c${inputFrames}-base`,
        signal: lifetimeAbort.signal,
      });
      const candidate = await executeShape({
        device: audit.device,
        graphOwner,
        allocation,
        prefill,
        shape: preparedShape,
        fixture,
        quanta: preparedShape.quanta,
        measured: false,
        label: `correctness-c${inputFrames}-profile`,
        signal: lifetimeAbort.signal,
      });
      const repeat = await executeShape({
        device: audit.device,
        graphOwner,
        allocation,
        prefill,
        shape: preparedShape,
        fixture,
        quanta: preparedShape.quanta,
        measured: false,
        label: `correctness-c${inputFrames}-profile-repeat`,
        signal: lifetimeAbort.signal,
      });
      const oracleToCandidate = compareOpt0066Raw(
        oracle.output,
        candidate.output,
      );
      const candidateDeterminism = compareOpt0066Raw(
        candidate.output,
        repeat.output,
      );
      const passed = rawExact(oracleToCandidate) &&
        rawExact(candidateDeterminism) &&
        oracle.outputSha256 === candidate.outputSha256 &&
        candidate.outputSha256 === repeat.outputSha256;
      if (!passed) {
        throw new Error(
          `OPT-0059 C${inputFrames} exact revision-7 correctness failed: ` +
            JSON.stringify({ oracleToCandidate, candidateDeterminism }),
        );
      }
      correctness.push(Object.freeze({
        inputFrames,
        order: Object.freeze([
          "dual-k4-oracle",
          "dual-k4-profile",
          "dual-k4-profile-repeat",
        ]),
        hashes: Object.freeze({
          oracle: oracle.outputSha256,
          profile: candidate.outputSha256,
          profileRepeat: repeat.outputSha256,
        }),
        rawOracleToProfile: oracleToCandidate,
        rawProfileDeterminism: candidateDeterminism,
        scans: Object.freeze({
          oracle: oracle.scan,
          profile: candidate.scan,
          profileRepeat: repeat.scan,
        }),
        canaries: Object.freeze({
          oracle: oracle.canaries,
          profile: candidate.canaries,
          profileRepeat: repeat.canaries,
        }),
        standaloneK1AddAndSuccessorSnakeDispatchesRetained: true,
        passed,
      }));
      oracleHashes.push([
        String(inputFrames) as `${Opt0059Shape}`,
        oracle.outputSha256,
      ]);
      await browserYield();
    }
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0059 emitted a WebGPU runtime event during preparation");
    }
    const readyAtEpochMilliseconds = Date.now();
    const sameProfileOracleSha256 = Object.freeze(
      Object.fromEntries(oracleHashes),
    ) as Readonly<Record<`${Opt0059Shape}`, string>>;
    const preparationReceipt = Object.freeze({
      schema: OPT_0059_SCHEMA,
      experimentId: OPT_0059_EXPERIMENT_ID,
      status: "ready",
      classification:
        "authenticated-revision7-dual-k4-c2378-geometry-only",
      identity,
      package: packageReceipt(pkg),
      fixtures: Object.freeze(OPT_0059_SHAPES.map((inputFrames) => Object.freeze({
        inputFrames,
        channels: 64,
        elementCount: fixtures[String(inputFrames) as `${Opt0059Shape}`].length,
        byteLength: fixtures[String(inputFrames) as `${Opt0059Shape}`].byteLength,
        sha256: OPT_0059_FIXTURE_SHA256[String(inputFrames) as `${Opt0059Shape}`],
        authenticatedBeforeDeviceRequest: true,
      }))),
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        adapterInfo: context.capabilities.adapterInfo,
        deviceFeatures: context.capabilities.deviceFeatures,
        deviceLimits: context.capabilities.deviceLimits,
        requestedLimits: context.capabilities.requestedLimits,
      }),
      ownership: Object.freeze({
        dedicatedWorker: true,
        fifoGraphOwnerCount: 1,
        gpuPackageOwnerCount: 1,
        weightPhaseCount: 1,
        c2314CapableActivationAllocationCount: 1,
        dispatchShapes: OPT_0059_SHAPES,
        sharedWeightsWorkspacesAndReadbackAcrossShapes: true,
        noJavaScriptPackageMirror: true,
      }),
      cancellationAndProgress: Object.freeze({
        oneLifetimeAbortController: true,
        disposeAbortsActiveWorkBeforeSerializedCleanup: true,
        schedulerChecksAbortAfterEveryQueueDrain: true,
        submittedWorkDrainsBeforeFifoOwnerRelease: true,
        progressReportedEvery16CommandBuffersAndAtCompletion: true,
      }),
      topology: Object.freeze({
        packageBindingAuthenticationFrames: 256,
        packageBindingsAreFrameNeutral: true,
        runtimeProfileId: runtime.runtimeProfileId,
        kernelSetId: runtime.kernelSetId,
        kernelTopology: runtime.kernelTopology,
        precisionMapSha256:
          ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
            .precisionMapSha256,
        shapes: Object.freeze(Object.fromEntries(OPT_0059_SHAPES.map(
          (inputFrames) => [
            String(inputFrames),
            shapes[String(inputFrames) as `${Opt0059Shape}`].topology,
          ],
        ))),
        opt0063FusionDisposition: Object.freeze({
          integrated: false,
          reason:
            "Add output aliases the all-channel K1 input and remains live as " +
            "the next residual skip; redirecting it leaves stale state while " +
            "writing it in-place creates a cross-workgroup read/write race",
          standaloneK1AddAndSuccessorSnakeDispatchesRetained: true,
          addSnakeOnlyFusionRequiresANewExperimentId: true,
        }),
      }),
      resourcePlan: gatePlan.maximumAllocation,
      storageLimitPlan,
      measuredLiveResourcesAtReady: liveAudit,
      projectionPlan: gatePlan,
      sameProfileOracleSha256,
      correctness: Object.freeze(correctness),
      correctnessPassed: true,
      runtimeEvents: Object.freeze([...runtimeEvents]),
      canaryPatternU32: STORAGE_CANARY_U32,
      expectedCanaryReadbackSha256: STORAGE_CANARY_SHA256,
      outputPrefillQNaNU32: OUTPUT_PREFILL_QNAN_U32,
      preparationWallMs: performance.now() - preparationStarted,
      readyAtEpochMilliseconds,
      readyForThermalGate: true,
      timingHasRun: false,
      gpuWorkRanOnlyAfterExplicitInitializeButton: true,
      productionDefaultChanged: false,
    });
    return createPreparedGate({
      identity,
      context,
      device: audit.device,
      lifetimeAbort,
      runtimeEvents,
      audit,
      graphOwner,
      pkg,
      phase,
      allocation,
      runtime,
      fixtures,
      shapes,
      sameProfileOracleSha256,
      prefill,
      preparationReceipt,
      readyAtEpochMilliseconds,
    });
  } catch (error) {
    try { await context?.device.queue.onSubmittedWorkDone(); } catch { /* device loss */ }
    runtime?.destroy();
    allocation?.destroy();
    phase?.destroy();
    await graphOwner.dispose();
    context?.destroy();
    throw error;
  }
}

async function authenticateFixtures(): Promise<
  Readonly<Record<`${Opt0059Shape}`, Float32Array<ArrayBuffer>>>
> {
  const entries: Array<readonly [`${Opt0059Shape}`, Float32Array<ArrayBuffer>]> = [];
  for (const inputFrames of OPT_0059_SHAPES) {
    const bytes = createAceOpt0011LatentFixture(inputFrames);
    const key = String(inputFrames) as `${Opt0059Shape}`;
    if (
      bytes.byteLength !== inputFrames * 64 * FLOAT32_BYTES ||
      await sha256Hex(bytes) !== OPT_0059_FIXTURE_SHA256[key]
    ) throw new Error(`OPT-0059 C${inputFrames} fixture identity changed`);
    const fixture = new Float32Array(inputFrames * 64);
    fixture.set(new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength / FLOAT32_BYTES,
    ));
    entries.push([key, fixture as Float32Array<ArrayBuffer>]);
  }
  return Object.freeze(Object.fromEntries(entries)) as
    Readonly<Record<`${Opt0059Shape}`, Float32Array<ArrayBuffer>>>;
}

async function authenticateAndAcquirePackage(
  signal: AbortSignal,
): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(
      OPT_0059_REVISION7_MANIFEST_PATH,
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
  ) throw new Error("OPT-0059 authenticated revision-7 package changed");
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({ ...loaded.manifest, files }),
    manifestUrl: loaded.manifestUrl,
    cache: await AceOpfsModelCache.open(),
    signal,
    onFileProgress: (progress) => postProgress(
      `acquiring sole revision-7 VAE package ` +
        `${progress.fileIndex + 1}/${progress.fileCount}: ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== files.length ||
    acquired.plan.runtimeBytes !== residentBytes
  ) throw new Error("OPT-0059 package acquisition accounting changed");
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
    manifestPath: OPT_0059_REVISION7_MANIFEST_PATH,
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

function createSharedAllocation(
  device: GPUDevice,
  gatePlan: ReturnType<typeof planOpt0059Gate>,
  packageBindings: ReturnType<typeof resolveAceOpt0054Fp16VaePackageBindings>,
): SharedAllocation {
  const maximum = gatePlan.shapes["2314"];
  const created: GPUBuffer[] = [];
  const createGuarded = (
    label: string,
    logicalBytes: number,
    usage: GPUBufferUsageFlags,
  ): GuardedBinding => {
    const buffer = device.createBuffer({
      label,
      size: logicalBytes + 2 * GUARD_BYTES,
      usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    created.push(buffer);
    return Object.freeze({
      label,
      buffer,
      logicalBytes,
      binding: Object.freeze({ buffer, offset: GUARD_BYTES, size: logicalBytes }),
    });
  };
  try {
    const stagingInput = createGuarded(
      "opt-0059-c2314-staging-input",
      maximum.stagingInputBytes,
      GPUBufferUsage.STORAGE,
    );
    const decoderInput = createGuarded(
      "opt-0059-c2314-decoder-input",
      maximum.decoderInputBytes,
      GPUBufferUsage.STORAGE,
    );
    const workspaces = Object.freeze([0, 1, 2].map((index) => createGuarded(
      `opt-0059-c2314-workspace-${index}`,
      maximum.workspaceBytes,
      GPUBufferUsage.STORAGE,
    ))) as readonly [GuardedBinding, GuardedBinding, GuardedBinding];
    const output = createGuarded(
      "opt-0059-c2314-output",
      maximum.outputBytes,
      GPUBufferUsage.STORAGE,
    );
    const guarded = Object.freeze([
      stagingInput,
      decoderInput,
      ...workspaces,
      output,
    ]);
    const readback = device.createBuffer({
      label: "opt-0059-c2314-output-readback",
      size: maximum.outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    created.push(readback);
    const canaryReadback = device.createBuffer({
      label: "opt-0059-storage-canary-readback",
      size: gatePlan.maximumAllocation.canaryReadbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    created.push(canaryReadback);
    const bindings = Object.freeze({
      stagingInput: stagingInput.binding,
      decoderInput: decoderInput.binding,
      workspaces: Object.freeze(workspaces.map((entry) => entry.binding)) as
        readonly [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding],
      output: output.binding,
      package: packageBindings,
    });
    initializeCanaries(device, guarded);
    let destroyed = false;
    return Object.freeze({
      stagingInput,
      decoderInput,
      workspaces,
      output,
      guarded,
      readback,
      canaryReadback,
      bindings,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of created) buffer.destroy();
      },
    });
  } catch (error) {
    for (const buffer of created) buffer.destroy();
    throw error;
  }
}

function initializeCanaries(
  device: GPUDevice,
  guarded: readonly GuardedBinding[],
): void {
  const canary = new Uint32Array(GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT);
  canary.fill(STORAGE_CANARY_U32);
  for (const entry of guarded) {
    device.queue.writeBuffer(entry.buffer, 0, canary);
    device.queue.writeBuffer(
      entry.buffer,
      GUARD_BYTES + entry.logicalBytes,
      canary,
    );
  }
}

function createProfiledQuanta(
  base: AceOpt0011Fp16VaeWindowDispatch,
  rangeControlBuffer: GPUBuffer,
): Readonly<{
  quanta: readonly ProfileQuantum[];
  topology: Readonly<Record<string, unknown>>;
}> {
  const shapePlan = planOpt0059Shape(base.plan.inputFrames as Opt0059Shape);
  const quanta = Object.freeze(base.quanta.map((quantum) => Object.freeze({
    family: familyForBaseQuantum(base, quantum.operationIndex),
    operationLabel: quantum.operationLabel,
    encode: (pass: GPUComputePassEncoder) => quantum.encode(pass),
  })));
  const familyCounts: Record<Opt0059ProfileFamily, number> = {
    "k7-conv1d": 0,
    "k1-conv1d": 0,
    "conv-transpose1d": 0,
    snake: 0,
    add: 0,
  };
  for (const quantum of base.graphQuanta) {
    const operationIndex = quantum.operationIndex!;
    const operation = base.plan.operations[operationIndex]!;
    const family = familyForOperation(operation);
    familyCounts[family] += 1;
  }
  if (
    base.graphQuantumCount !== shapePlan.graphQuantumCount ||
    quanta.length !== shapePlan.sequenceQuantumCount ||
    JSON.stringify(familyCounts) !== JSON.stringify(shapePlan.familyQuantumCounts)
  ) throw new Error(`OPT-0059 C${base.plan.inputFrames} pure topology changed`);
  return Object.freeze({
    quanta,
    topology: Object.freeze({
      inputFrames: base.plan.inputFrames,
      operationCount: base.operationCount,
      graphQuantumCount: base.graphQuantumCount,
      sequenceQuantumCount: base.quanta.length,
      familyQuantumCounts: Object.freeze(familyCounts),
      decoderCommandBufferCount: Math.ceil(
        quanta.length / OPT_0059_QUANTA_PER_COMMAND_BUFFER,
      ),
      readbackCommandBufferCount: 1,
      totalCommandBufferCount:
        Math.ceil(quanta.length / OPT_0059_QUANTA_PER_COMMAND_BUFFER) + 1,
      dynamicControlRecordCount: base.dynamicControls.recordCount,
      dynamicControlBytes: base.dynamicControls.byteLength,
      retainedRangeControlBufferBytes: rangeControlBuffer.size,
      twoDimensionalK1OperationLabels:
        shapePlan.twoDimensionalK1OperationLabels,
      k1Routes: shapePlan.k1Routes,
      opt0063FusionIntegrated: false,
      standaloneK1AddAndSuccessorSnakeDispatchesRetained: true,
    }),
  });
}

async function createOutputPrefill(
  device: GPUDevice,
  output: GPUBufferBinding,
  wordCount: number,
): Promise<Readonly<{
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly wordCount: number;
}>> {
  if (!Number.isSafeInteger(wordCount) || wordCount < 1) {
    throw new RangeError("OPT-0059 output prefill word count changed");
  }
  const module = device.createShaderModule({
    label: "opt-0059-output-qnan-prefill",
    code: /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
const WORD_COUNT: u32 = ${wordCount}u;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < WORD_COUNT) { output[id.x] = ${OUTPUT_PREFILL_QNAN_U32}u; }
}
`,
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0059 qNaN prefill WGSL failed: ` +
        errors.map(({ lineNum, linePos, message }) =>
          `${lineNum}:${linePos} ${message}`
        ).join("; "),
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    label: "opt-0059-output-qnan-prefill",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    label: "opt-0059-output-qnan-prefill-bindings",
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: output }],
  });
  return Object.freeze({ pipeline, bindGroup, wordCount });
}

async function executeShape(input: Readonly<{
  device: GPUDevice;
  graphOwner: AceFifoGraphOwner;
  allocation: SharedAllocation;
  prefill: PreparedGate["prefill"];
  shape: PreparedShape;
  fixture: Float32Array<ArrayBuffer>;
  quanta: readonly ProfileQuantum[];
  measured: boolean;
  label: string;
  signal: AbortSignal;
}>): Promise<ExecutionResult> {
  input.signal.throwIfAborted();
  const shapePlan = input.shape.plan;
  if (
    input.fixture.byteLength !== shapePlan.stagingInputBytes ||
    input.quanta.length < 1
  ) throw new Error(`${input.label} execution inputs changed`);
  initializeCanaries(input.device, input.allocation.guarded);
  input.device.queue.writeBuffer(
    input.allocation.stagingInput.buffer,
    GUARD_BYTES,
    input.fixture,
  );
  const prefillEncoder = input.device.createCommandEncoder({
    label: `${input.label}-prefill`,
  });
  const prefillPass = prefillEncoder.beginComputePass({
    label: `${input.label}-prefill-pass`,
  });
  prefillPass.setPipeline(input.prefill.pipeline);
  prefillPass.setBindGroup(0, input.prefill.bindGroup);
  prefillPass.dispatchWorkgroups(Math.ceil(input.prefill.wordCount / 256));
  prefillPass.end();
  input.device.queue.submit([prefillEncoder.finish()]);
  await input.device.queue.onSubmittedWorkDone();

  const decoderCommandBufferCount = Math.ceil(
    input.quanta.length / OPT_0059_QUANTA_PER_COMMAND_BUFFER,
  );
  const totalCommandBufferCount = decoderCommandBufferCount + 1;
  const buckets = emptyBuckets();
  let decoderSubmitThroughDrainMs = 0;
  let readbackSubmitThroughDrainMs = 0;
  let decoderWallMs = 0;
  let schedulingResult: Awaited<ReturnType<
    typeof submitAceCommandBufferFactoriesCooperatively
  >> | undefined;
  const lease = await input.graphOwner.acquire(input.signal);
  let outputWords: Uint32Array<ArrayBuffer> | undefined;
  let readbackMapWallMs = 0;
  let outerWindowWallMs = 0;
  let canaries: Readonly<Record<string, unknown>> | undefined;
  const batchProfiles = Array.from(
    { length: decoderCommandBufferCount },
    (_, index) => classifyBatch(input.quanta.slice(
      index * OPT_0059_QUANTA_PER_COMMAND_BUFFER,
      Math.min(
        (index + 1) * OPT_0059_QUANTA_PER_COMMAND_BUFFER,
        input.quanta.length,
      ),
    )),
  );
  try {
    const outerStarted = performance.now();
    schedulingResult = await submitAceCommandBufferFactoriesCooperatively({
      queue: input.device.queue,
      commandBufferCount: totalCommandBufferCount,
      signal: input.signal,
      createCommandBuffer: (index) => {
        const encoder = input.device.createCommandEncoder({
          label: `${input.label}-command-${index}`,
        });
        if (index === decoderCommandBufferCount) {
          encoder.copyBufferToBuffer(
            input.allocation.output.buffer,
            GUARD_BYTES,
            input.allocation.readback,
            0,
            input.allocation.output.logicalBytes,
          );
          return encoder.finish();
        }
        const start = index * OPT_0059_QUANTA_PER_COMMAND_BUFFER;
        const end = Math.min(
          start + OPT_0059_QUANTA_PER_COMMAND_BUFFER,
          input.quanta.length,
        );
        const pass = encoder.beginComputePass({
          label: `${input.label}-decoder-${start}-${end}`,
        });
        for (let quantum = start; quantum < end; quantum += 1) {
          input.quanta[quantum]!.encode(pass);
        }
        pass.end();
        return encoder.finish();
      },
      onCommandBufferDrained: ({ commandBufferIndex, submitThroughDrainMs }) => {
        if (commandBufferIndex === decoderCommandBufferCount) {
          readbackSubmitThroughDrainMs += submitThroughDrainMs;
          return;
        }
        decoderSubmitThroughDrainMs += submitThroughDrainMs;
        const profile = batchProfiles[commandBufferIndex]!;
        const bucket = buckets[profile.bucket];
        bucket.batchCount += 1;
        bucket.quantumCount += profile.quantumCount;
        bucket.submitThroughDrainMs += submitThroughDrainMs;
        if (commandBufferIndex === decoderCommandBufferCount - 1) {
          decoderWallMs = performance.now() - outerStarted;
        }
      },
      onProgress: (progress) => {
        if (
          progress.completedCommandBuffers === progress.totalCommandBuffers ||
          progress.completedCommandBuffers % 16 === 0
        ) {
          postProgress(
            `${input.label}: ${progress.completedCommandBuffers}/` +
              `${progress.totalCommandBuffers} command buffers, ` +
              `${progress.cooperativeIdleMs} ms requested idle`,
          );
        }
      },
    });
    const mapStarted = performance.now();
    await input.allocation.readback.mapAsync(
      GPUMapMode.READ,
      0,
      input.allocation.output.logicalBytes,
    );
    outputWords = new Uint32Array(
      input.allocation.readback.getMappedRange(
        0,
        input.allocation.output.logicalBytes,
      ).slice(0),
    ) as Uint32Array<ArrayBuffer>;
    input.allocation.readback.unmap();
    readbackMapWallMs = performance.now() - mapStarted;
    outerWindowWallMs = performance.now() - outerStarted;
    canaries = await readCanaries(input.device, input.allocation);
  } finally {
    if (input.allocation.readback.mapState !== "unmapped") {
      input.allocation.readback.unmap();
    }
    lease.release();
  }
  if (
    outputWords === undefined || canaries === undefined ||
    schedulingResult === undefined ||
    schedulingResult.commandBuffersSubmitted !== totalCommandBufferCount ||
    schedulingResult.queueDrains !== totalCommandBufferCount ||
    schedulingResult.cooperativeIdleMs !== totalCommandBufferCount - 1 ||
    decoderWallMs <= 0 || decoderSubmitThroughDrainMs <= 0 ||
    readbackSubmitThroughDrainMs <= 0 || readbackMapWallMs <= 0 ||
    outerWindowWallMs <= 0
  ) throw new Error(`${input.label} timing/scheduling receipt is incomplete`);
  const activeWords = shapePlan.outputBytes / Uint32Array.BYTES_PER_ELEMENT;
  let nonFiniteCount = 0;
  let activePrefillQNaNCount = 0;
  let firstTailMismatchWord: number | null = null;
  const floats = new Float32Array(
    outputWords.buffer,
    outputWords.byteOffset,
    activeWords,
  );
  for (let index = 0; index < activeWords; index += 1) {
    if (!Number.isFinite(floats[index]!)) nonFiniteCount += 1;
    if (outputWords[index] === OUTPUT_PREFILL_QNAN_U32) {
      activePrefillQNaNCount += 1;
    }
  }
  for (let index = activeWords; index < outputWords.length; index += 1) {
    if (outputWords[index] !== OUTPUT_PREFILL_QNAN_U32) {
      firstTailMismatchWord = index;
      break;
    }
  }
  const scan = Object.freeze({
    activeFloat32Count: activeWords,
    activeByteLength: shapePlan.outputBytes,
    nonFiniteCount,
    activePrefillQNaNCount,
    tailWordCount: outputWords.length - activeWords,
    firstTailMismatchWord,
    activeCompleteFiniteWritePassed:
      nonFiniteCount === 0 && activePrefillQNaNCount === 0,
    inactiveTailUntouchedPassed: firstTailMismatchWord === null,
  });
  if (
    nonFiniteCount !== 0 || activePrefillQNaNCount !== 0 ||
    firstTailMismatchWord !== null || canaries["passed"] !== true
  ) throw new Error(`${input.label} complete-write/finite/canary gate failed`);
  const activeBytes = new Uint8Array(
    outputWords.buffer,
    outputWords.byteOffset,
    shapePlan.outputBytes,
  );
  const activeOutput = new Float32Array(activeWords);
  activeOutput.set(floats);
  return Object.freeze({
    output: activeOutput as Float32Array<ArrayBuffer>,
    outputSha256: await sha256Hex(activeBytes),
    sample: Object.freeze({
      inputFrames: input.shape.inputFrames,
      decoderSubmitThroughDrainMs,
      decoderWallMs,
      readbackSubmitThroughDrainMs,
      readbackMapWallMs,
      outerWindowWallMs,
      families: Object.freeze(Object.fromEntries(
        OPT_0059_PROFILE_FAMILIES.map((family) => [
          family,
          Object.freeze({ ...buckets[family] }),
        ]),
      )) as Opt0059TimingSample["families"],
      mixed: Object.freeze({ ...buckets.mixed }),
    }),
    scan,
    canaries,
    scheduling: Object.freeze({
      measured: input.measured,
      label: input.label,
      fifoOwnerSequence: lease.sequence,
      quantaPerCommandBuffer: OPT_0059_QUANTA_PER_COMMAND_BUFFER,
      decoderQuantumCount: input.quanta.length,
      decoderCommandBufferCount,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount,
      queueDrainCount: schedulingResult.queueDrains,
      requestedCooperativeIdleMs: schedulingResult.cooperativeIdleMs,
      oneOutstandingCommandBuffer: true,
    }),
  });
}

async function readCanaries(
  device: GPUDevice,
  allocation: SharedAllocation,
): Promise<Readonly<Record<string, unknown>>> {
  const encoder = device.createCommandEncoder({
    label: "opt-0059-canary-readback",
  });
  let destinationOffset = 0;
  for (const guarded of allocation.guarded) {
    encoder.copyBufferToBuffer(
      guarded.buffer,
      0,
      allocation.canaryReadback,
      destinationOffset,
      GUARD_BYTES,
    );
    destinationOffset += GUARD_BYTES;
    encoder.copyBufferToBuffer(
      guarded.buffer,
      GUARD_BYTES + guarded.logicalBytes,
      allocation.canaryReadback,
      destinationOffset,
      GUARD_BYTES,
    );
    destinationOffset += GUARD_BYTES;
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await allocation.canaryReadback.mapAsync(GPUMapMode.READ, 0, destinationOffset);
  const bytes = new Uint8Array(
    allocation.canaryReadback.getMappedRange(0, destinationOffset).slice(0),
  ) as Uint8Array<ArrayBuffer>;
  allocation.canaryReadback.unmap();
  const words = new Uint32Array(bytes.buffer);
  const firstMismatchWord = words.findIndex((word) => word !== STORAGE_CANARY_U32);
  const spans = Object.freeze(allocation.guarded.map((guarded, index) => {
    const firstWord = index * 2 * GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
    const spanWords = GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
    const prefixPassed = words.subarray(firstWord, firstWord + spanWords)
      .every((word) => word === STORAGE_CANARY_U32);
    const suffixPassed = words.subarray(
      firstWord + spanWords,
      firstWord + 2 * spanWords,
    ).every((word) => word === STORAGE_CANARY_U32);
    return Object.freeze({ label: guarded.label, prefixPassed, suffixPassed });
  }));
  const sha256 = await sha256Hex(bytes);
  return Object.freeze({
    patternU32: STORAGE_CANARY_U32,
    guardedBufferCount: allocation.guarded.length,
    checkedBytes: destinationOffset,
    sha256,
    expectedSha256: STORAGE_CANARY_SHA256,
    firstMismatchWord: firstMismatchWord < 0 ? null : firstMismatchWord,
    spans,
    passed: sha256 === STORAGE_CANARY_SHA256 && firstMismatchWord < 0 &&
      spans.every((span) => span.prefixPassed && span.suffixPassed),
  });
}

async function runTimedGate(
  prepared: PreparedGate,
  suppliedThermalGate: Opt0059ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermalGate = revalidateThermalGate(
    suppliedThermalGate,
    prepared.readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const main: Opt0059TimingSample[] = [];
  const edge: Opt0059TimingSample[] = [];
  const executions: Readonly<Record<string, unknown>>[] = [];
  const runOrder = Object.freeze([
    ...OPT_0059_MAIN_ORDER.map((shape) => ["main", shape] as const),
    ...OPT_0059_EDGE_ORDER.map((shape) => ["edge", shape] as const),
  ]);
  for (const [index, [group, inputFrames]] of runOrder.entries()) {
    postProgress(
      `timed ${index + 1}/${runOrder.length}: ${group} C${inputFrames}`,
    );
    const key = String(inputFrames) as `${Opt0059Shape}`;
    const result = await executeShape({
      device: prepared.device,
      graphOwner: prepared.graphOwner,
      allocation: prepared.allocation,
      prefill: prepared.prefill,
      shape: prepared.shapes[key],
      fixture: prepared.fixtures[key],
      quanta: prepared.shapes[key].quanta,
      measured: true,
      label: `timed-${group}-${index}-c${inputFrames}`,
      signal: prepared.lifetimeAbort.signal,
    });
    const sameProfileOracleSha256 = prepared.sameProfileOracleSha256[key];
    if (result.outputSha256 !== sameProfileOracleSha256) {
      throw new Error(
        `OPT-0059 timed C${inputFrames} output differs from its prepared ` +
          "same-profile oracle",
      );
    }
    (group === "main" ? main : edge).push(result.sample);
    executions.push(Object.freeze({
      group,
      orderIndex: index,
      inputFrames,
      outputSha256: result.outputSha256,
      sameProfileOracleSha256,
      matchesPreparedSameProfileOracle: true,
      timing: result.sample,
      scheduling: result.scheduling,
      scan: result.scan,
      canaries: result.canaries,
    }));
    await browserYield();
  }
  if (prepared.runtimeEvents.length !== 0) {
    throw new Error("OPT-0059 emitted a WebGPU runtime event during timing");
  }
  const evaluation = evaluateOpt0059Timing(main, edge);
  return Object.freeze({
    schema: OPT_0059_SCHEMA,
    experimentId: OPT_0059_EXPERIMENT_ID,
    status: evaluation.passed ? "passed" : "negative",
    decision: evaluation.passed
      ? "positive-c2378-short-projected-wall-gate"
      : "negative-retain-c512-production-windowing",
    identity: prepared.identity,
    thermalGate,
    // Deliberately self-contained: READY provenance is retained verbatim.
    preparation: prepared.preparationReceipt,
    timing: Object.freeze({
      mainOrder: OPT_0059_MAIN_ORDER,
      edgeOrder: OPT_0059_EDGE_ORDER,
      executions: Object.freeze(executions),
      main: Object.freeze(main),
      edge: Object.freeze(edge),
      evaluation,
    }),
    runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
    performanceGatePassed: evaluation.passed,
    correctnessAuthorityRetainedInFinalReceipt: true,
    packageFixtureTopologyCanaryAndResourcePlanRetainedInFinalReceipt: true,
    productionDefaultChanged: false,
  });
}

function createPreparedGate(
  input: Omit<PreparedGate, "destroy">,
): PreparedGate {
  let destroyPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  return Object.freeze({
    ...input,
    destroy(reason: unknown = new DOMException(
      "OPT-0059 comparison complete",
      "AbortError",
    )): Promise<Readonly<Record<string, unknown>>> {
      if (destroyPromise !== undefined) return destroyPromise;
      destroyPromise = (async () => {
        const started = performance.now();
        input.lifetimeAbort.abort(reason);
        await input.device.queue.onSubmittedWorkDone();
        input.runtime.destroy();
        input.allocation.destroy();
        input.phase.destroy();
        await input.graphOwner.dispose();
        const beforeDeviceDestroy = input.audit.snapshot();
        const passed =
          beforeDeviceDestroy["liveBufferCount"] === 0 &&
          beforeDeviceDestroy["liveBufferBytes"] === 0 &&
          beforeDeviceDestroy["mappedOrPendingBufferCount"] === 0 &&
          beforeDeviceDestroy["everyBufferDestroyedExactlyOnce"] === true &&
          beforeDeviceDestroy["belowFourGigabytes"] === true &&
          beforeDeviceDestroy["maximumLiveBufferBytes"] ===
            planOpt0059Gate().maximumAllocation.plannedLiveGpuBytes;
        input.context.destroy();
        if (!passed) {
          throw new Error(
            `OPT-0059 lifecycle audit failed: ` +
              JSON.stringify(beforeDeviceDestroy),
          );
        }
        return Object.freeze({
          passed,
          reason: serializeOpt0018Failure(reason),
          resourcesBeforeDeviceDestroy: beforeDeviceDestroy,
          queueDrainedBeforeDestroy: true,
          decoderRuntimeAndFourControlBuffersDestroyed: true,
          oneActivationAllocationDestroyed: true,
          soleGpuWeightPhaseDestroyed: true,
          fifoGraphOwnerDisposed: true,
          deviceContextDestroyed: true,
          allBuffersDestroyedExactlyOnce: true,
          maximumLiveGpuBytesBelowFourGigabytes: true,
          wallMs: performance.now() - started,
          completedAtEpochMilliseconds: Date.now(),
        });
      })();
      return destroyPromise;
    },
  });
}

function revalidateThermalGate(
  gate: Opt0059ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0059ThermalGate {
  const parameters = new URLSearchParams({
    thermalSource: gate.source,
    thermalStartedAtEpochMilliseconds: String(gate.startedAtEpochMilliseconds),
    thermalCheckedAtEpochMilliseconds: String(gate.checkedAtEpochMilliseconds),
    thermalObservations: String(gate.observationCount),
    thermalObservedLevel: String(gate.observedLevel),
  });
  const verified = parseOpt0059ThermalGate(
    parameters,
    readyAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
  if (JSON.stringify(verified) !== JSON.stringify(gate)) {
    throw new Error("OPT-0059 worker thermal receipt differs from page proof");
  }
  return verified;
}

function requireDevice(
  context: AceWebGpuDeviceContext,
  storageLimitPlan: Readonly<{
    readonly guardBytesPerSide: number;
    readonly maximumLogicalStorageBindingBytes: number;
    readonly maximumPhysicalBufferBytes: number;
  }>,
): void {
  const info = context.capabilities.adapterInfo;
  const storageAlignment = context.device.limits.minStorageBufferOffsetAlignment;
  if (
    !context.device.features.has("shader-f16") ||
    !context.device.features.has("subgroups") ||
    info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32 ||
    storageLimitPlan.maximumPhysicalBufferBytes !==
      storageLimitPlan.maximumLogicalStorageBindingBytes +
        2 * storageLimitPlan.guardBytesPerSide ||
    !Number.isSafeInteger(storageAlignment) || storageAlignment < 1 ||
    storageLimitPlan.guardBytesPerSide % storageAlignment !== 0 ||
    context.device.limits.maxBufferSize <
      storageLimitPlan.maximumPhysicalBufferBytes ||
    context.device.limits.maxStorageBufferBindingSize <
      storageLimitPlan.maximumLogicalStorageBindingBytes ||
    context.device.limits.maxStorageBuffersPerShaderStage < 8 ||
    context.device.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    context.device.limits.maxComputeWorkgroupSizeX < 256
  ) throw new Error("OPT-0059 requires fixed32 WebGPU and C2314 storage limits");
}

function familyForBaseQuantum(
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  operationIndex: number | null,
): ProfileQuantum["family"] {
  return operationIndex === null
    ? "ingress"
    : familyForOperation(dispatch.plan.operations[operationIndex]!);
}

function familyForOperation(
  operation: AceVaeDecoderOperation,
): Opt0059ProfileFamily {
  if (operation.kind === "conv-transpose1d") return "conv-transpose1d";
  if (operation.kind === "snake") return "snake";
  if (operation.kind === "conv1d" && operation.shape.kernelSize === 1) {
    return "k1-conv1d";
  }
  if (operation.kind === "conv1d") return "k7-conv1d";
  return "add";
}

function emptyBuckets(): Record<ProfileBucket, {
  batchCount: number;
  quantumCount: number;
  submitThroughDrainMs: number;
}> {
  return {
    "k7-conv1d": { batchCount: 0, quantumCount: 0, submitThroughDrainMs: 0 },
    "k1-conv1d": {
      batchCount: 0,
      quantumCount: 0,
      submitThroughDrainMs: 0,
    },
    "conv-transpose1d": {
      batchCount: 0,
      quantumCount: 0,
      submitThroughDrainMs: 0,
    },
    snake: { batchCount: 0, quantumCount: 0, submitThroughDrainMs: 0 },
    add: { batchCount: 0, quantumCount: 0, submitThroughDrainMs: 0 },
    mixed: { batchCount: 0, quantumCount: 0, submitThroughDrainMs: 0 },
  };
}

function classifyBatch(quanta: readonly ProfileQuantum[]): Readonly<{
  bucket: ProfileBucket;
  quantumCount: number;
}> {
  const families = new Set(quanta.map((quantum) => quantum.family));
  const only = families.size === 1 ? [...families][0] : undefined;
  return Object.freeze({
    bucket: only === undefined || only === "ingress" ? "mixed" : only,
    quantumCount: quanta.length,
  });
}

function rawExact(comparison: ReturnType<typeof compareOpt0066Raw>): boolean {
  return comparison.rawU32Exact && comparison.rawU16Exact &&
    comparison.u32MismatchCount === 0 && comparison.u16MismatchCount === 0;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest(
    "SHA-256",
    bytes as Uint8Array<ArrayBuffer>,
  ));
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
