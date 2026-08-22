import type { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import {
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
} from "../model/manifest.js";
import {
  AceFifoGraphOwner,
  submitAceCommandBufferFactoriesCooperatively,
  submitAceCommandBufferFactoriesDepth2Epoch4,
  type AceDepth2Epoch4CommandBufferCompletionTiming,
  type AceDepth2Epoch4CompletionEpochTiming,
} from "../runtime/scheduler.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY,
  ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES,
  ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
  AceOpt0011Fp16VaeDecoderRuntime,
  planAceOpt0011Fp16VaeChunkDispatches,
  type AceOpt0011Fp16VaeDecoderKernelTopology,
  type AceOpt0011Fp16VaeChunkDispatchSet,
  type AceOpt0011Fp16VaeWindowDispatch,
  type AceOpt0011Fp16VaeWindowBindings,
} from "./vae-fp16-decoder.js";
import {
  resolveAceOpt0011Fp16VaePackageBindings,
  resolveAceOpt0054Fp16VaePackageBindings,
  type AceOpt0011VaePackageBindings,
} from "./vae-fp16-package.js";
import {
  selectAceVaeRuntimeProfile,
  type AceVaeAuthenticatedPackageIdentity,
  type AceVaeRuntimeProfile,
} from "./vae-fp16-profile.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkBackend,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
} from "./vae-chunks.js";
import {
  planAceVaeDecoder,
  type AceVaeDecoderOperation,
} from "./vae-decoder.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";
import type { AceVaeChunkGpuBackendProgress } from "./vae-backend.js";

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAXIMUM_WINDOW_FRAMES = 512;
export const ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER = 64;
export const ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY =
  "opt-0080-c2314-depth2-phase-epoch4" as const;
export const ACE_OPT_0080_VAE_PRODUCTION_WINDOW_FRAMES = 2_314 as const;
export const ACE_OPT_0080_VAE_PRODUCTION_DECODER_QUANTA = 35_498 as const;
export const ACE_OPT_0080_VAE_PRODUCTION_DECODER_COMMAND_BUFFERS = 555 as const;
export const ACE_OPT_0080_VAE_PRODUCTION_TOTAL_COMMAND_BUFFERS = 556 as const;

export type AceOpt0011Fp16VaeBackendMaximumWindowFrames =
  | typeof MAXIMUM_WINDOW_FRAMES
  | typeof ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES;

export const ACE_OPT_0011_FP16_VAE_PROFILE_FAMILIES = Object.freeze([
  "k7-conv1d",
  "k1-conv1d",
  "conv-transpose1d",
  "snake",
  "add",
] as const);

export type AceOpt0011Fp16VaeProfileFamily =
  (typeof ACE_OPT_0011_FP16_VAE_PROFILE_FAMILIES)[number];

export interface AceOpt0011Fp16VaeFamilyTimingTotal {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly submitThroughDrainMs: number;
}

export interface AceOpt0011Fp16VaeWindowFamilyProfile {
  readonly windowIndex: number;
  readonly inputFrames: number;
  readonly quantaPerCommandBuffer: number;
  readonly decoderBatchCount: number;
  readonly decoderQuantumCount: number;
  readonly decoderSubmitThroughDrainMs: number;
  readonly homogeneousBatchCount: number;
  readonly homogeneousQuantumCount: number;
  readonly homogeneousSubmitThroughDrainMs: number;
  readonly mixedBatchCount: number;
  readonly mixedQuantumCount: number;
  readonly mixedSubmitThroughDrainMs: number;
  readonly families: Readonly<Record<
    AceOpt0011Fp16VaeProfileFamily,
    AceOpt0011Fp16VaeFamilyTimingTotal
  >>;
}

export type AceOpt0080VaeSchedulingProfile =
  | "depth1-epoch1"
  | "depth2-phase-epoch4";

export type AceOpt0080VaeProductionSchedulingPolicy =
  typeof ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY;

export interface AceOpt0080VaeCommandCompletionEvidence {
  readonly commandBufferIndex: number;
  readonly commandKind: "decoder" | "readback";
  /** Cumulative under depth two and therefore deliberately non-additive. */
  readonly submitThroughCompletionFenceMs: number;
  readonly trueQueueDrain: boolean;
  readonly completionEpochIndex: number;
}

export interface AceOpt0080VaeCompletionEpochEvidence {
  readonly completionEpochIndex: number;
  readonly phaseIndex: 0;
  readonly firstCommandBufferIndex: number;
  readonly lastCommandBufferIndex: number;
  readonly commandBufferCount: number;
  /** Disjoint within the single window; these values may be summed. */
  readonly submitThroughTrueDrainMs: number;
}

export interface AceOpt0080VaeSchedulingEvidence {
  readonly schema: "ace-opt-0080-vae-window-scheduling-v1";
  readonly windowIndex: number;
  readonly schedulingProfile: AceOpt0080VaeSchedulingProfile;
  readonly decoderQuantumCount: number;
  readonly quantaPerCommandBuffer: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  /** First lazy command creation through the final true drain; excludes map. */
  readonly schedulingWallMs: number;
  readonly commandBuffersSubmitted: number;
  readonly completionFenceRequestedCount: number;
  readonly completionFenceSettledCount: number;
  readonly completionFenceRejectedCount: number;
  readonly trueQueueDrainCount: number;
  readonly completionEpochCount: number;
  readonly cooperativeIdleTurns: number;
  readonly requestedCooperativeIdleMs: number;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
  readonly commandCompletions:
    readonly AceOpt0080VaeCommandCompletionEvidence[];
  readonly completionEpochs: readonly AceOpt0080VaeCompletionEpochEvidence[];
}

/** @internal Per-window OPT-0080 benchmark seam; ordinary production omits it. */
export interface AceOpt0080VaeRunOptions {
  readonly schedulingProfile: AceOpt0080VaeSchedulingProfile;
  readonly onSchedulingEvidence: (
    evidence: AceOpt0080VaeSchedulingEvidence,
  ) => void;
}

export interface AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan {
  readonly residentWeightBytes: number;
  readonly stagingInputBufferBytes: number;
  readonly decoderInputBufferBytes: number;
  readonly workspaceBufferBytes: number;
  readonly workspaceBufferCount: 3;
  readonly outputBufferBytes: number;
  readonly readbackBufferBytes: number;
  /** One immutable dynamic-control allocation per retained exact shape. */
  readonly controlBufferBytes: number;
  readonly accountedGpuBytes: number;
  readonly latentSnapshotBytes: number;
  readonly maximumReturnedWindowBytes: number;
  readonly boundedCpuBytes: number;
  readonly maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames;
  readonly quantaPerCommandBuffer: number;
}

interface AceOpt0011Fp16VaeChunkGpuBackendBaseOptions {
  readonly device: GPUDevice;
  readonly plan: AceVaeChunkedDecodePlan;
  /** FP32 NLC `[latentFrames,64]`; snapshotted before factory resolution. */
  readonly finalLatents: Float32Array;
  readonly authenticatedPackage: AceVaeAuthenticatedPackageIdentity;
  /** Ownership transfers at call entry, including on factory failure. */
  readonly ownedVaeWeights: AceGpuTensorPhase;
  /** Explicit authenticated C512 or OPT-0070 C2378 window ceiling. */
  readonly maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames;
  /** @internal OPT-0027 benchmark seam; production defaults to the accepted 8. */
  readonly quantaPerCommandBuffer?: number;
  /** Omission preserves the production depth-one scheduler. */
  readonly submissionPolicy?: AceOpt0080VaeSchedulingProfile;
  /**
   * Exact production selector. Unlike the benchmark-wide `submissionPolicy`,
   * this promotes only an authenticated C2314 window and leaves every other
   * window on depth one.
   */
  readonly productionSchedulingPolicy?:
    AceOpt0080VaeProductionSchedulingPolicy;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceVaeChunkGpuBackendProgress) => void;
  /** One bounded aggregate per successfully decoded window. */
  readonly onFamilyProfile?: (
    profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  ) => void;
  /** @internal Deterministic test seam; production uses the real 1 ms timer. */
  readonly yieldQueueIdle?: () => Promise<void>;
}

export type AceOpt0011Fp16VaeChunkGpuBackendOptions =
  AceOpt0011Fp16VaeChunkGpuBackendBaseOptions & (
    | Readonly<{
        readonly runtimeProfileId?:
          "opt-0011-mixed-fp16-portable-v1";
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0028-mixed-fp16-portable-exact-packed-v1";
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0028-mixed-fp16-fixed32-exact-packed-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0054-mixed-fp16-fixed32-revision7-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
    | Readonly<{
        readonly runtimeProfileId:
          "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1";
        readonly subgroupMinSize: 32;
        readonly subgroupMaxSize: 32;
      }>
  );

type RequiredBackendRuntimeSelection =
  | Readonly<{
      readonly runtimeProfileId:
        | "opt-0011-mixed-fp16-portable-v1"
        | "opt-0028-mixed-fp16-portable-exact-packed-v1";
    }>
  | Readonly<{
      readonly runtimeProfileId:
        | "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
        | "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
        | "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
        | "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
        | "opt-0054-mixed-fp16-fixed32-revision7-v1"
        | "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>;

/** @internal Prepared-resource seam for deterministic coordinator tests. */
export interface AceOpt0011Fp16VaePreparedBuffers {
  readonly stagingInput: GPUBuffer;
  readonly decoderInput: GPUBuffer;
  readonly workspaces: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  readonly output: GPUBuffer;
  readonly readback: GPUBuffer;
}

interface DecoderRuntimeOwner {
  destroy(): void;
}

interface BackendRuntimeResources {
  readonly device: GPUDevice;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly ownedVaeWeights: AceGpuTensorPhase;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceVaeChunkGpuBackendProgress) => void;
  readonly onFamilyProfile?: (
    profile: AceOpt0011Fp16VaeWindowFamilyProfile,
  ) => void;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly quantaPerCommandBuffer?: number;
  readonly submissionPolicy?: AceOpt0080VaeSchedulingProfile;
  readonly productionSchedulingPolicy?:
    AceOpt0080VaeProductionSchedulingPolicy;
}

/** @internal Prepared-resource seam; ownership transfers at call entry. */
export interface AceOpt0011Fp16VaePreparedResources
  extends BackendRuntimeResources {
  /** @internal Omission retains the original C512 prepared-resource seam. */
  readonly maximumWindowFrames?: AceOpt0011Fp16VaeBackendMaximumWindowFrames;
  readonly runtimeProfile: AceVaeRuntimeProfile;
  readonly finalLatents: Float32Array;
  readonly buffers: AceOpt0011Fp16VaePreparedBuffers;
  readonly runtime: DecoderRuntimeOwner;
  readonly dispatchSet: AceOpt0011Fp16VaeChunkDispatchSet;
}

export interface AceOpt0011Fp16VaeDispatchTopologyReceipt {
  readonly runtimeProfileId:
    AceOpt0011Fp16VaeChunkDispatchSet["runtimeProfileId"];
  readonly kernelSetId: AceOpt0011Fp16VaeChunkDispatchSet["kernelSetId"];
  readonly uniqueWindowFrames: readonly number[];
  readonly windows: readonly Readonly<{
    readonly inputFrames: number;
    readonly operationCount: number;
    readonly graphQuantumCount: number;
    readonly sequenceQuantumCount: number;
    readonly kernelQuantumCounts: Readonly<Record<string, number>>;
    readonly operationQuantumCounts: readonly Readonly<{
      readonly operationIndex: number;
      readonly operationLabel: string;
      readonly operationKind: AceVaeDecoderOperation["kind"];
      readonly kernelId: AceOpt0011Fp16VaeWindowDispatch["graphQuanta"][number]["kernelId"];
      readonly quantumCount: number;
    }>[];
  }>[];
}

export class AceOpt0011Fp16VaeChunkGpuBackend
  implements AceVaeChunkBackend {
  readonly memory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly runtimeProfileId:
    AceOpt0011Fp16VaeChunkDispatchSet["runtimeProfileId"];
  readonly kernelSetId: AceOpt0011Fp16VaeChunkDispatchSet["kernelSetId"];
  readonly kernelTopology:
    AceOpt0011Fp16VaeChunkDispatchSet["kernelTopology"];

  private readonly graphOwner = new AceFifoGraphOwner();
  private readonly lifetime = new AbortController();
  private readonly sourceSignal: AbortSignal | undefined;
  private readonly sourceAbortListener: (() => void) | undefined;
  private destroyPromise: Promise<void> | undefined;
  private state: "live" | "destroying" | "destroyed" = "live";

  private constructor(
    private readonly resources: BackendRuntimeResources,
    private readonly finalLatents: Float32Array<ArrayBuffer>,
    private readonly buffers: AceOpt0011Fp16VaePreparedBuffers,
    private readonly runtime: DecoderRuntimeOwner,
    private readonly dispatchSet: AceOpt0011Fp16VaeChunkDispatchSet,
    maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames,
  ) {
    this.runtimeProfileId = dispatchSet.runtimeProfileId;
    this.kernelSetId = dispatchSet.kernelSetId;
    this.kernelTopology = dispatchSet.kernelTopology;
    this.memory = planFp16VaeChunkGpuBackendMemory(
      resources.plan,
      resources.device.limits.minUniformBufferOffsetAlignment,
      resolveQuantaPerCommandBuffer(resources.quantaPerCommandBuffer),
      maximumWindowFrames,
    );
    this.sourceSignal = resources.signal;
    this.sourceAbortListener = resources.signal === undefined
      ? undefined
      : () => {
          const reason = resources.signal!.reason;
          if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
          void this.beginDestroy(reason).catch(() => undefined);
        };
    this.sourceSignal?.addEventListener(
      "abort",
      this.sourceAbortListener!,
      { once: true },
    );
    const weakBackend = new WeakRef(this);
    void resources.device.lost.then((info) => {
      const backend = weakBackend.deref();
      if (backend === undefined || backend.state !== "live") return;
      const error = new Error(
        `OPT-0011 FP16 VAE device lost (${info.reason}): ` +
          `${info.message || "no device message"}`,
      );
      backend.lifetime.abort(error);
      void backend.beginDestroy(error).catch(() => undefined);
    });
  }

  static async create(
    options: AceOpt0011Fp16VaeChunkGpuBackendOptions,
  ): Promise<AceOpt0011Fp16VaeChunkGpuBackend> {
    const weights = options.ownedVaeWeights;
    let buffers: AceOpt0011Fp16VaePreparedBuffers | undefined;
    let runtime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
    let backend: AceOpt0011Fp16VaeChunkGpuBackend | undefined;
    try {
      options.signal?.throwIfAborted();
      const submissionPolicy = resolveOpt0080VaeSchedulingProfile(
        options.submissionPolicy,
      );
      requireCompatibleVaeSchedulingProfile(
        submissionPolicy,
        options.onFamilyProfile,
      );
      requireBackendPlan(options.plan, options.maximumWindowFrames);
      const runtimeSelection = requireBackendRuntimeSelection(options);
      requireOpt0080VaeProductionSchedulingPolicy({
        productionSchedulingPolicy: options.productionSchedulingPolicy,
        submissionPolicy: options.submissionPolicy,
        runtimeProfileId: runtimeSelection.runtimeProfileId,
        packageConverterRevision:
          options.authenticatedPackage.manifest.provenance.converterRevision,
        maximumWindowFrames: options.maximumWindowFrames,
        quantaPerCommandBuffer: options.quantaPerCommandBuffer,
        onFamilyProfile: options.onFamilyProfile,
        yieldQueueIdle: options.yieldQueueIdle,
      });
      const runtimeProfile = "subgroupMinSize" in runtimeSelection
        ? selectAceVaeRuntimeProfile({
            requestedProfile: runtimeSelection.runtimeProfileId,
            package: options.authenticatedPackage,
            deviceFeatures: options.device.features,
            subgroupMinSize: runtimeSelection.subgroupMinSize,
            subgroupMaxSize: runtimeSelection.subgroupMaxSize,
            deviceLimits: options.device.limits,
            decoderPlan: planAceVaeDecoder(256),
          })
        : selectAceVaeRuntimeProfile({
            requestedProfile: runtimeSelection.runtimeProfileId,
            package: options.authenticatedPackage,
            deviceFeatures: options.device.features,
            deviceLimits: options.device.limits,
            decoderPlan: planAceVaeDecoder(256),
          });
      requireMaximumWindowDeviceLimits(
        options.device,
        options.maximumWindowFrames,
      );
      const finalLatents = snapshotLatents(options.plan, options.finalLatents);
      const packageBindings = runtimeSelection.runtimeProfileId ===
          "opt-0054-mixed-fp16-fixed32-revision7-v1" ||
          runtimeSelection.runtimeProfileId ===
            "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
        ? resolveAceOpt0054Fp16VaePackageBindings(
            planAceVaeDecoder(256),
            options.authenticatedPackage,
            weights,
          )
        : resolveAceOpt0011Fp16VaePackageBindings(
            planAceVaeDecoder(256),
            options.authenticatedPackage,
            weights,
          );
      buffers = await createBuffers(
        options.device,
        options.maximumWindowFrames,
      );
      options.signal?.throwIfAborted();
      runtime = AceOpt0011Fp16VaeDecoderRuntime.create(
        options.device,
        runtimeSelection,
      );
      const dispatchSet = await runtime.createChunkDispatchSet(
        `ace-opt-0011-fp16-vae-${options.plan.latentFrames}`,
        options.plan.latentFrames,
        options.maximumWindowFrames,
        runtimeBindings(buffers, packageBindings),
      );
      requireDispatchSet(options.plan, dispatchSet, runtimeProfile);
      options.signal?.throwIfAborted();
      const retainedResources: BackendRuntimeResources = Object.freeze({
        device: options.device,
        plan: options.plan,
        ownedVaeWeights: weights,
        ...(options.submissionPolicy === undefined
          ? {}
          : { submissionPolicy }),
        ...(options.productionSchedulingPolicy === undefined
          ? {}
          : {
              productionSchedulingPolicy:
                options.productionSchedulingPolicy,
            }),
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        ...(options.onProgress === undefined
          ? {}
          : { onProgress: options.onProgress }),
        ...(options.onFamilyProfile === undefined
          ? {}
          : { onFamilyProfile: options.onFamilyProfile }),
        ...(options.yieldQueueIdle === undefined
          ? {}
          : { yieldQueueIdle: options.yieldQueueIdle }),
        ...(options.quantaPerCommandBuffer === undefined
          ? {}
          : {
              quantaPerCommandBuffer: resolveQuantaPerCommandBuffer(
                options.quantaPerCommandBuffer,
              ),
            }),
      });
      backend = new AceOpt0011Fp16VaeChunkGpuBackend(
        retainedResources,
        finalLatents,
        buffers,
        runtime,
        dispatchSet,
        options.maximumWindowFrames,
      );
      return backend;
    } catch (error) {
      if (backend !== undefined) {
        await backend.destroy(error);
      } else {
        runtime?.destroy();
        destroyBuffers(buffers);
        weights.destroy();
      }
      throw error;
    }
  }

  /** @internal Build the coordinator around authenticated-equivalent fakes. */
  static fromPreparedResources(
    prepared: AceOpt0011Fp16VaePreparedResources,
  ): AceOpt0011Fp16VaeChunkGpuBackend {
    try {
      prepared.signal?.throwIfAborted();
      const maximumWindowFrames = prepared.maximumWindowFrames ??
        MAXIMUM_WINDOW_FRAMES;
      const submissionPolicy = resolveOpt0080VaeSchedulingProfile(
        prepared.submissionPolicy,
      );
      requireCompatibleVaeSchedulingProfile(
        submissionPolicy,
        prepared.onFamilyProfile,
      );
      requireOpt0080VaeProductionSchedulingPolicy({
        productionSchedulingPolicy: prepared.productionSchedulingPolicy,
        submissionPolicy: prepared.submissionPolicy,
        runtimeProfileId: prepared.runtimeProfile.id,
        packageConverterRevision:
          prepared.runtimeProfile.packageConverterRevision,
        maximumWindowFrames,
        quantaPerCommandBuffer: prepared.quantaPerCommandBuffer,
        onFamilyProfile: prepared.onFamilyProfile,
        yieldQueueIdle: prepared.yieldQueueIdle,
      });
      requireBackendPlan(prepared.plan, maximumWindowFrames);
      requireDispatchSet(
        prepared.plan,
        prepared.dispatchSet,
        prepared.runtimeProfile,
      );
      requirePreparedBuffers(prepared.buffers);
      const snapshot = snapshotLatents(prepared.plan, prepared.finalLatents);
      const retained: BackendRuntimeResources = Object.freeze({
        device: prepared.device,
        plan: prepared.plan,
        ownedVaeWeights: prepared.ownedVaeWeights,
        ...(prepared.submissionPolicy === undefined
          ? {}
          : { submissionPolicy }),
        ...(prepared.productionSchedulingPolicy === undefined
          ? {}
          : {
              productionSchedulingPolicy:
                prepared.productionSchedulingPolicy,
            }),
        ...(prepared.signal === undefined
          ? {}
          : { signal: prepared.signal }),
        ...(prepared.onProgress === undefined
          ? {}
          : { onProgress: prepared.onProgress }),
        ...(prepared.onFamilyProfile === undefined
          ? {}
          : { onFamilyProfile: prepared.onFamilyProfile }),
        ...(prepared.yieldQueueIdle === undefined
          ? {}
          : { yieldQueueIdle: prepared.yieldQueueIdle }),
        ...(prepared.quantaPerCommandBuffer === undefined
          ? {}
          : {
              quantaPerCommandBuffer: resolveQuantaPerCommandBuffer(
                prepared.quantaPerCommandBuffer,
              ),
            }),
      });
      return new AceOpt0011Fp16VaeChunkGpuBackend(
        retained,
        snapshot,
        prepared.buffers,
        prepared.runtime,
        prepared.dispatchSet,
        maximumWindowFrames,
      );
    } catch (error) {
      prepared.runtime.destroy();
      destroyBuffers(prepared.buffers);
      prepared.ownedVaeWeights.destroy();
      throw error;
    }
  }

  async decodeWindow(
    window: AceVaeDecodeWindow,
    signal?: AbortSignal,
    opt0080Run?: AceOpt0080VaeRunOptions,
  ): Promise<Float32Array> {
    this.requireLive();
    requirePlanWindow(this.resources.plan, window);
    const opt0080 = resolveOpt0080VaeRunOptions(opt0080Run);
    if (
      opt0080 !== undefined &&
      this.resources.productionSchedulingPolicy !== undefined
    ) {
      throw new Error(
        "OPT-0080 VAE production scheduling cannot use a benchmark callback",
      );
    }
    const schedulingProfile = opt0080?.schedulingProfile ??
      this.resources.submissionPolicy ??
      resolveOpt0080VaeProductionWindowSchedulingProfile(
        this.resources.productionSchedulingPolicy,
        window.latentWindowFrames,
      );
    requireCompatibleVaeSchedulingProfile(
      schedulingProfile,
      this.resources.onFamilyProfile,
    );
    const activeSignal = combineSignals([
      this.lifetime.signal,
      this.resources.signal,
      signal,
    ]);
    activeSignal.throwIfAborted();
    const lease = await this.graphOwner.acquire(activeSignal);
    try {
      this.requireLive();
      activeSignal.throwIfAborted();
      const dispatch = this.dispatchSet.windows[window.index]?.dispatch;
      if (
        dispatch === undefined ||
        dispatch.plan.inputFrames !== window.latentWindowFrames
      ) {
        throw new Error(
          `OPT-0011 FP16 VAE window ${window.index} lost its exact dispatch`,
        );
      }
      const quantaPerCommandBuffer = resolveQuantaPerCommandBuffer(
        this.resources.quantaPerCommandBuffer,
      );
      const decoderCommandBuffers = Math.ceil(
        dispatch.quanta.length / quantaPerCommandBuffer,
      );
      const totalCommandBuffers = decoderCommandBuffers + 1;
      if (
        this.resources.productionSchedulingPolicy !== undefined &&
        schedulingProfile === "depth2-phase-epoch4" &&
        (dispatch.quanta.length !==
            ACE_OPT_0080_VAE_PRODUCTION_DECODER_QUANTA ||
          quantaPerCommandBuffer !==
            ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
          decoderCommandBuffers !==
            ACE_OPT_0080_VAE_PRODUCTION_DECODER_COMMAND_BUFFERS ||
          totalCommandBuffers !==
            ACE_OPT_0080_VAE_PRODUCTION_TOTAL_COMMAND_BUFFERS)
      ) {
        throw new Error(
          "OPT-0080 VAE C2314 production command topology changed",
        );
      }
      this.uploadLatentWindow(window, dispatch);
      activeSignal.throwIfAborted();

      const outputBytes = dispatch.activeOutputBytes;
      const familyProfiler = this.resources.onFamilyProfile === undefined
        ? undefined
        : createWindowFamilyProfiler(
            window,
            dispatch,
            quantaPerCommandBuffer,
          );
      const createCommandBuffer = (index: number): GPUCommandBuffer => {
        activeSignal.throwIfAborted();
        return index < decoderCommandBuffers
          ? encodeQuantumBatch(
              this.resources.device,
              dispatch,
              index,
              window.index,
              activeSignal,
              quantaPerCommandBuffer,
            )
          : encodeReadback(
              this.resources.device,
              this.buffers.output,
              this.buffers.readback,
              outputBytes,
              window.index,
            );
      };
      const reportProgress = (
        completedCommandBuffers: number,
        queueDrains: number,
        cooperativeIdleMs: number,
      ): void => {
        const readback = completedCommandBuffers > decoderCommandBuffers;
        this.resources.onProgress?.({
          windowIndex: window.index,
          completedDecoderQuanta: readback
            ? dispatch.quanta.length
            : Math.min(
                completedCommandBuffers * quantaPerCommandBuffer,
                dispatch.quanta.length,
              ),
          totalDecoderQuanta: dispatch.quanta.length,
          completedCommandBuffers,
          totalCommandBuffers,
          queueDrains,
          cooperativeIdleMs,
          stage: readback ? "readback" : "decoder",
        });
      };
      const commandCompletions: AceOpt0080VaeCommandCompletionEvidence[] = [];
      const completionEpochs: AceOpt0080VaeCompletionEpochEvidence[] = [];
      let schedulingEvidenceBase: Readonly<{
        commandBuffersSubmitted: number;
        completionFenceRequestedCount: number;
        completionFenceSettledCount: number;
        completionFenceRejectedCount: number;
        trueQueueDrainCount: number;
        completionEpochCount: number;
        cooperativeIdleTurns: number;
        requestedCooperativeIdleMs: number;
        maximumOutstandingCommandBuffers: 1 | 2;
      }>;

      const schedulingStartedAt = opt0080 === undefined
        ? 0
        : performance.now();
      let schedulingCompletedAt: number | undefined;
      if (schedulingProfile === "depth2-phase-epoch4") {
        const scheduled = await submitAceCommandBufferFactoriesDepth2Epoch4({
          queue: this.resources.device.queue,
          device: this.resources.device,
          commandBufferCount: totalCommandBuffers,
          // Decoder and ordinary readback deliberately share one phase. For
          // C2314 this is ceil((555 + 1) / 4) = 139 true drains, not 140.
          phaseCommandBufferCounts: Object.freeze([totalCommandBuffers]),
          createCommandBuffer,
          signal: activeSignal,
          ...(this.resources.yieldQueueIdle === undefined
            ? {}
            : { yieldQueueIdle: this.resources.yieldQueueIdle }),
          onCommandBufferCompleted: (timing, progress) => {
            if (opt0080 !== undefined) {
              commandCompletions.push(opt0080VaeCompletionEvidence(
                timing,
                decoderCommandBuffers,
              ));
            }
            reportProgress(
              progress.completedCommandBuffers,
              progress.trueQueueDrainCount,
              progress.requestedCooperativeIdleMs,
            );
          },
          ...(opt0080 === undefined
            ? {}
            : {
                onCompletionEpochDrained: (timing) => {
                  completionEpochs.push(opt0080VaeEpochEvidence(timing));
                },
              }),
        });
        schedulingCompletedAt = opt0080 === undefined
          ? undefined
          : performance.now();
        const maximumOutstandingCommandBuffers =
          scheduled.maximumOutstandingCommandBuffers;
        if (maximumOutstandingCommandBuffers !== 2) {
          throw new Error(
            "OPT-0080 VAE depth-two scheduler did not reach two outstanding " +
              "singleton command buffers",
          );
        }
        schedulingEvidenceBase = Object.freeze({
          commandBuffersSubmitted: scheduled.commandBuffersSubmitted,
          completionFenceRequestedCount:
            scheduled.completionFenceRequestedCount,
          completionFenceSettledCount: scheduled.completionFenceSettledCount,
          completionFenceRejectedCount:
            scheduled.completionFenceRejectedCount,
          trueQueueDrainCount: scheduled.trueQueueDrainCount,
          completionEpochCount: scheduled.completionEpochCount,
          cooperativeIdleTurns: scheduled.cooperativeIdleTurns,
          requestedCooperativeIdleMs: scheduled.requestedCooperativeIdleMs,
          maximumOutstandingCommandBuffers,
        });
      } else {
        const scheduled = await submitAceCommandBufferFactoriesCooperatively({
          queue: this.resources.device.queue,
          commandBufferCount: totalCommandBuffers,
          createCommandBuffer,
          signal: activeSignal,
          ...(this.resources.yieldQueueIdle === undefined
            ? {}
            : { yieldQueueIdle: this.resources.yieldQueueIdle }),
          ...(familyProfiler === undefined && opt0080 === undefined
            ? {}
            : {
                onCommandBufferDrained: (timing) => {
                  familyProfiler?.record(
                    timing.commandBufferIndex,
                    timing.submitThroughDrainMs,
                  );
                  if (opt0080 === undefined) return;
                  commandCompletions.push(Object.freeze({
                    commandBufferIndex: timing.commandBufferIndex,
                    commandKind:
                      timing.commandBufferIndex < decoderCommandBuffers
                        ? "decoder" as const
                        : "readback" as const,
                    submitThroughCompletionFenceMs:
                      timing.submitThroughDrainMs,
                    trueQueueDrain: true,
                    completionEpochIndex: timing.commandBufferIndex,
                  }));
                  completionEpochs.push(Object.freeze({
                    completionEpochIndex: timing.commandBufferIndex,
                    phaseIndex: 0 as const,
                    firstCommandBufferIndex: timing.commandBufferIndex,
                    lastCommandBufferIndex: timing.commandBufferIndex,
                    commandBufferCount: 1,
                    submitThroughTrueDrainMs: timing.submitThroughDrainMs,
                  }));
                },
              }),
          onProgress: (progress) => {
            reportProgress(
              progress.completedCommandBuffers,
              progress.queueDrains,
              progress.cooperativeIdleMs,
            );
          },
        });
        schedulingCompletedAt = opt0080 === undefined
          ? undefined
          : performance.now();
        schedulingEvidenceBase = Object.freeze({
          commandBuffersSubmitted: scheduled.commandBuffersSubmitted,
          completionFenceRequestedCount: totalCommandBuffers,
          completionFenceSettledCount: totalCommandBuffers,
          completionFenceRejectedCount: 0,
          trueQueueDrainCount: scheduled.queueDrains,
          completionEpochCount: scheduled.queueDrains,
          cooperativeIdleTurns: totalCommandBuffers - 1,
          requestedCooperativeIdleMs: scheduled.cooperativeIdleMs,
          maximumOutstandingCommandBuffers: 1 as const,
        });
      }

      if (opt0080 !== undefined) {
        activeSignal.throwIfAborted();
        if (schedulingCompletedAt === undefined) {
          throw new Error("OPT-0080 VAE scheduling wall was not captured");
        }
        const schedulingWallMs = nonnegativeVaeTimingElapsed(
          schedulingCompletedAt,
          schedulingStartedAt,
        );
        opt0080.onSchedulingEvidence(Object.freeze({
          schema: "ace-opt-0080-vae-window-scheduling-v1" as const,
          windowIndex: window.index,
          schedulingProfile,
          decoderQuantumCount: dispatch.quanta.length,
          quantaPerCommandBuffer,
          decoderCommandBufferCount: decoderCommandBuffers,
          readbackCommandBufferCount: 1 as const,
          totalCommandBufferCount: totalCommandBuffers,
          schedulingWallMs,
          ...schedulingEvidenceBase,
          commandCompletions: Object.freeze([...commandCompletions]),
          completionEpochs: Object.freeze([...completionEpochs]),
        }));
        activeSignal.throwIfAborted();
      }
      const output = await mapDetachedOutput(
        this.buffers.readback,
        outputBytes,
        activeSignal,
      );
      if (familyProfiler !== undefined) {
        try {
          this.resources.onFamilyProfile?.(familyProfiler.finish());
        } catch {
          // Profiling is observational and cannot invalidate decoded audio.
        }
      }
      return output;
    } finally {
      lease.release();
    }
  }

  /** @internal Immutable benchmark receipt; never called from timed execution. */
  captureDispatchTopology(): AceOpt0011Fp16VaeDispatchTopologyReceipt {
    this.requireLive();
    return Object.freeze({
      runtimeProfileId: this.runtimeProfileId,
      kernelSetId: this.kernelSetId,
      uniqueWindowFrames: Object.freeze([
        ...this.dispatchSet.topology.uniqueWindowFrames,
      ]),
      windows: Object.freeze(this.dispatchSet.dispatches.map((dispatch) => {
        const kernelQuantumCounts: Record<string, number> = {};
        for (const quantum of dispatch.quanta) {
          kernelQuantumCounts[quantum.kernelId] =
            (kernelQuantumCounts[quantum.kernelId] ?? 0) + 1;
        }
        const operationQuantumCounts = dispatch.plan.operations.map(
          (operation, operationIndex) => {
            const quanta = dispatch.graphQuanta.filter((quantum) =>
              quantum.operationIndex === operationIndex
            );
            const kernelIds = new Set(quanta.map((quantum) => quantum.kernelId));
            if (quanta.length === 0 || kernelIds.size !== 1) {
              throw new Error(
                `OPT-0011 FP16 VAE operation ${operation.label} lost one kernel owner`,
              );
            }
            return Object.freeze({
              operationIndex,
              operationLabel: operation.label,
              operationKind: operation.kind,
              kernelId: quanta[0]!.kernelId,
              quantumCount: quanta.length,
            });
          },
        );
        return Object.freeze({
          inputFrames: dispatch.plan.inputFrames,
          operationCount: dispatch.operationCount,
          graphQuantumCount: dispatch.graphQuantumCount,
          sequenceQuantumCount: dispatch.quanta.length,
          kernelQuantumCounts: Object.freeze(kernelQuantumCounts),
          operationQuantumCounts: Object.freeze(operationQuantumCounts),
        });
      })),
    });
  }

  destroy(reason: unknown = destroyedError()): Promise<void> {
    return this.beginDestroy(reason);
  }

  private uploadLatentWindow(
    window: AceVaeDecodeWindow,
    dispatch: AceOpt0011Fp16VaeWindowDispatch,
  ): void {
    const channels = this.resources.plan.decoderWorkspacePlan.config
      .decoderInputChannels;
    const start = window.windowStartLatentFrame * channels;
    const end = window.windowEndLatentFrame * channels;
    const source = this.finalLatents.subarray(start, end);
    if (
      source.byteLength !== dispatch.activeStagingInputBytes ||
      dispatch.plan.inputFrames !== window.latentWindowFrames
    ) {
      throw new Error(
        `OPT-0011 FP16 VAE window ${window.index} lost its exact latent slice`,
      );
    }
    this.resources.device.queue.writeBuffer(
      this.buffers.stagingInput,
      0,
      source,
    );
  }

  private beginDestroy(reason: unknown): Promise<void> {
    if (this.destroyPromise !== undefined) return this.destroyPromise;
    this.state = "destroying";
    if (
      this.sourceSignal !== undefined &&
      this.sourceAbortListener !== undefined
    ) {
      this.sourceSignal.removeEventListener("abort", this.sourceAbortListener);
    }
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    this.destroyPromise = (async () => {
      try {
        await this.graphOwner.dispose();
      } finally {
        try {
          this.runtime.destroy();
          destroyBuffers(this.buffers);
          this.resources.ownedVaeWeights.destroy();
        } finally {
          this.state = "destroyed";
        }
      }
    })();
    return this.destroyPromise;
  }

  private requireLive(): void {
    if (this.state !== "live") {
      throw new DOMException(
        `OPT-0011 FP16 VAE backend is ${this.state}`,
        "InvalidStateError",
      );
    }
  }
}

export function planAceOpt0011Fp16VaeChunkGpuBackendMemory(
  plan: AceVaeChunkedDecodePlan,
  recordAlignment = 256,
  quantaPerCommandBuffer =
    ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
): AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan {
  return planFp16VaeChunkGpuBackendMemory(
    plan,
    recordAlignment,
    quantaPerCommandBuffer,
    MAXIMUM_WINDOW_FRAMES,
  );
}

/** Exact memory contract for the C2378 maximum window. */
export function planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory(
  plan: AceVaeChunkedDecodePlan,
  recordAlignment = 256,
  quantaPerCommandBuffer =
    ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
): AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan {
  return planFp16VaeChunkGpuBackendMemory(
    plan,
    recordAlignment,
    quantaPerCommandBuffer,
    ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  );
}

function planFp16VaeChunkGpuBackendMemory(
  plan: AceVaeChunkedDecodePlan,
  recordAlignment: number,
  quantaPerCommandBuffer: number,
  maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames,
): AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan {
  const resolvedQuantaPerCommandBuffer = resolveQuantaPerCommandBuffer(
    quantaPerCommandBuffer,
  );
  requireBackendPlan(plan, maximumWindowFrames);
  const maximum = planAceVaeDecoder(maximumWindowFrames);
  const residentWeightBytes = 168_791_552 as const;
  const stagingInputBufferBytes = checkedProduct(
    [maximum.inputElements, FLOAT32_BYTES],
    "FP16 VAE staging-input bytes",
  );
  const decoderInputBufferBytes = checkedProduct(
    [maximum.inputElements, FLOAT16_BYTES],
    "FP16 VAE decoder-input bytes",
  );
  const workspaceBufferBytes = checkedProduct(
    [maximum.maximumActivationElements, FLOAT16_BYTES],
    "FP16 VAE workspace bytes",
  );
  const outputBufferBytes = checkedProduct(
    [maximum.outputElements, FLOAT32_BYTES],
    "FP16 VAE output bytes",
  );
  const readbackBufferBytes = outputBufferBytes;
  const topology = planAceOpt0011Fp16VaeChunkDispatches(
    plan.latentFrames,
    maximumWindowFrames,
    recordAlignment,
  );
  const controlBufferBytes = topology.uniqueDynamicControlBytes;
  const accountedGpuBytes = residentWeightBytes + stagingInputBufferBytes +
    decoderInputBufferBytes + workspaceBufferBytes * 3 + outputBufferBytes +
    readbackBufferBytes + controlBufferBytes;
  const latentSnapshotBytes = checkedProduct(
    [plan.latentFrames, 64, FLOAT32_BYTES],
    "OPT-0011 FP16 VAE latent snapshot bytes",
  );
  return Object.freeze({
    residentWeightBytes,
    stagingInputBufferBytes,
    decoderInputBufferBytes,
    workspaceBufferBytes,
    workspaceBufferCount: 3,
    outputBufferBytes,
    readbackBufferBytes,
    controlBufferBytes,
    accountedGpuBytes,
    latentSnapshotBytes,
    maximumReturnedWindowBytes: outputBufferBytes,
    boundedCpuBytes: latentSnapshotBytes + outputBufferBytes,
    maximumWindowFrames,
    quantaPerCommandBuffer: resolvedQuantaPerCommandBuffer,
  });
}

async function createBuffers(
  device: GPUDevice,
  maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames,
): Promise<AceOpt0011Fp16VaePreparedBuffers> {
  const maximum = planAceVaeDecoder(maximumWindowFrames);
  const buffers = await createAceScopedBuffers(device, [
    {
      label: "ace-opt-0011-fp16-vae-staging-input",
      size: maximum.inputElements * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    {
      label: "ace-opt-0011-fp16-vae-decoder-input",
      size: maximum.inputElements * FLOAT16_BYTES,
      usage: GPUBufferUsage.STORAGE,
    },
    ...Array.from({ length: 3 }, (_, index) => ({
      label: `ace-opt-0011-fp16-vae-workspace-${index}`,
      size: maximum.maximumActivationElements * FLOAT16_BYTES,
      usage: GPUBufferUsage.STORAGE,
    })),
    {
      label: "ace-opt-0011-fp16-vae-output",
      size: maximum.outputElements * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    },
    {
      label: "ace-opt-0011-fp16-vae-readback",
      size: maximum.outputElements * FLOAT32_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    },
  ], maximumWindowFrames === MAXIMUM_WINDOW_FRAMES
    ? "OPT-0011 FP16 VAE production allocation"
    : "OPT-0070 FP16 VAE C2378 production allocation");
  return Object.freeze({
    stagingInput: buffers[0]!,
    decoderInput: buffers[1]!,
    workspaces: Object.freeze([
      buffers[2]!,
      buffers[3]!,
      buffers[4]!,
    ]) as readonly [GPUBuffer, GPUBuffer, GPUBuffer],
    output: buffers[5]!,
    readback: buffers[6]!,
  });
}

function runtimeBindings(
  buffers: AceOpt0011Fp16VaePreparedBuffers,
  packageBindings: AceOpt0011VaePackageBindings,
): AceOpt0011Fp16VaeWindowBindings {
  return Object.freeze({
    stagingInput: binding(buffers.stagingInput),
    decoderInput: binding(buffers.decoderInput),
    workspaces: Object.freeze([
      binding(buffers.workspaces[0]),
      binding(buffers.workspaces[1]),
      binding(buffers.workspaces[2]),
    ]) as readonly [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding],
    output: binding(buffers.output),
    package: packageBindings,
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size: Number(buffer.size) });
}

interface AceOpt0011Fp16VaeProfileBatch {
  readonly family: AceOpt0011Fp16VaeProfileFamily | null;
  readonly quantumCount: number;
}

interface AceOpt0011Fp16VaeWindowFamilyProfiler {
  record(commandBufferIndex: number, submitThroughDrainMs: number): void;
  finish(): AceOpt0011Fp16VaeWindowFamilyProfile;
}

function resolveOpt0080VaeSchedulingProfile(
  value: AceOpt0080VaeSchedulingProfile | undefined,
): AceOpt0080VaeSchedulingProfile {
  if (value === undefined || value === "depth1-epoch1") {
    return "depth1-epoch1";
  }
  if (value === "depth2-phase-epoch4") return value;
  throw new Error("OPT-0080 VAE submission policy changed");
}

/**
 * Resolve the narrow production policy without turning it into a backend-wide
 * depth-two switch. Exported only so product orchestration can apply the same
 * exact per-window rule to forced A/B arms.
 */
export function resolveOpt0080VaeProductionWindowSchedulingProfile(
  policy: AceOpt0080VaeProductionSchedulingPolicy | undefined,
  latentWindowFrames: number,
): AceOpt0080VaeSchedulingProfile {
  if (policy === undefined) return "depth1-epoch1";
  if (policy !== ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY) {
    throw new Error("OPT-0080 VAE production scheduling policy changed");
  }
  return latentWindowFrames === ACE_OPT_0080_VAE_PRODUCTION_WINDOW_FRAMES
    ? "depth2-phase-epoch4"
    : "depth1-epoch1";
}

interface Opt0080VaeProductionSchedulingSelection {
  readonly productionSchedulingPolicy?: unknown;
  readonly submissionPolicy?: unknown;
  readonly runtimeProfileId: unknown;
  readonly packageConverterRevision: unknown;
  readonly maximumWindowFrames: unknown;
  readonly quantaPerCommandBuffer?: unknown;
  readonly onFamilyProfile?: unknown;
  readonly yieldQueueIdle?: unknown;
}

function requireOpt0080VaeProductionSchedulingPolicy(
  selection: Opt0080VaeProductionSchedulingSelection,
): void {
  if (selection.productionSchedulingPolicy === undefined) return;
  if (
    selection.productionSchedulingPolicy !==
      ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY ||
    selection.submissionPolicy !== undefined ||
    selection.runtimeProfileId !==
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1" ||
    selection.packageConverterRevision !==
      ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    selection.maximumWindowFrames !==
      ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES ||
    selection.quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER ||
    selection.onFamilyProfile !== undefined ||
    selection.yieldQueueIdle !== undefined
  ) {
    throw new Error(
      "OPT-0080 VAE production scheduling requires exact revision-7 " +
        "OPT-0066 C2378/qpc64 resources without diagnostic seams",
    );
  }
}

function requireCompatibleVaeSchedulingProfile(
  schedulingProfile: AceOpt0080VaeSchedulingProfile,
  onFamilyProfile:
    | ((profile: AceOpt0011Fp16VaeWindowFamilyProfile) => void)
    | undefined,
): void {
  if (
    schedulingProfile === "depth2-phase-epoch4" &&
    onFamilyProfile !== undefined
  ) {
    throw new Error(
      "OPT-0080 VAE depth-two scheduling cannot attribute overlapping " +
        "completion-fence intervals by kernel family",
    );
  }
}

function resolveOpt0080VaeRunOptions(
  value: AceOpt0080VaeRunOptions | undefined,
): AceOpt0080VaeRunOptions | undefined {
  if (value === undefined) return undefined;
  const untrusted = value as Readonly<{
    schedulingProfile?: unknown;
    onSchedulingEvidence?: unknown;
  }>;
  if (
    (untrusted.schedulingProfile !== "depth1-epoch1" &&
      untrusted.schedulingProfile !== "depth2-phase-epoch4") ||
    typeof untrusted.onSchedulingEvidence !== "function"
  ) {
    throw new Error("OPT-0080 VAE run options changed");
  }
  return Object.freeze({
    schedulingProfile: untrusted.schedulingProfile,
    onSchedulingEvidence: untrusted.onSchedulingEvidence as
      AceOpt0080VaeRunOptions["onSchedulingEvidence"],
  });
}

function opt0080VaeCompletionEvidence(
  timing: AceDepth2Epoch4CommandBufferCompletionTiming,
  decoderCommandBufferCount: number,
): AceOpt0080VaeCommandCompletionEvidence {
  return Object.freeze({
    commandBufferIndex: timing.commandBufferIndex,
    commandKind: timing.commandBufferIndex < decoderCommandBufferCount
      ? "decoder" as const
      : "readback" as const,
    submitThroughCompletionFenceMs:
      timing.submitThroughCompletionFenceMs,
    trueQueueDrain: timing.trueQueueDrain,
    completionEpochIndex: timing.completionEpochIndex,
  });
}

function opt0080VaeEpochEvidence(
  timing: AceDepth2Epoch4CompletionEpochTiming,
): AceOpt0080VaeCompletionEpochEvidence {
  if (timing.phaseIndex !== 0) {
    throw new Error("OPT-0080 VAE window scheduling split into extra phases");
  }
  return Object.freeze({
    completionEpochIndex: timing.completionEpochIndex,
    phaseIndex: 0 as const,
    firstCommandBufferIndex: timing.firstCommandBufferIndex,
    lastCommandBufferIndex: timing.lastCommandBufferIndex,
    commandBufferCount: timing.commandBufferCount,
    submitThroughTrueDrainMs: timing.submitThroughTrueDrainMs,
  });
}

function nonnegativeVaeTimingElapsed(
  finishedAt: number,
  startedAt: number,
): number {
  const elapsed = finishedAt - startedAt;
  if (!Number.isFinite(elapsed) || elapsed < 0) {
    throw new RangeError(
      "OPT-0080 VAE scheduling clock must be finite and monotonic",
    );
  }
  return elapsed;
}

function createWindowFamilyProfiler(
  window: AceVaeDecodeWindow,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  quantaPerCommandBuffer: number,
): AceOpt0011Fp16VaeWindowFamilyProfiler {
  const batches = Array.from(
    { length: Math.ceil(dispatch.quanta.length / quantaPerCommandBuffer) },
    (_, batchIndex): AceOpt0011Fp16VaeProfileBatch => {
      const first = batchIndex * quantaPerCommandBuffer;
      const end = Math.min(
        first + quantaPerCommandBuffer,
        dispatch.quanta.length,
      );
      const families = new Set<AceOpt0011Fp16VaeProfileFamily>();
      let mixed = false;
      for (let quantumIndex = first; quantumIndex < end; quantumIndex += 1) {
        const family = profileFamily(dispatch, quantumIndex);
        if (family === null) {
          mixed = true;
        } else {
          families.add(family);
        }
      }
      return Object.freeze({
        family: !mixed && families.size === 1
          ? families.values().next().value!
          : null,
        quantumCount: end - first,
      });
    },
  );
  const families: Record<
    AceOpt0011Fp16VaeProfileFamily,
    { batchCount: number; quantumCount: number; submitThroughDrainMs: number }
  > = {
    "k7-conv1d": emptyMutableFamilyTiming(),
    "k1-conv1d": emptyMutableFamilyTiming(),
    "conv-transpose1d": emptyMutableFamilyTiming(),
    snake: emptyMutableFamilyTiming(),
    add: emptyMutableFamilyTiming(),
  };
  let recordedDecoderBatches = 0;
  let decoderSubmitThroughDrainMs = 0;
  let homogeneousBatchCount = 0;
  let homogeneousQuantumCount = 0;
  let homogeneousSubmitThroughDrainMs = 0;
  let mixedBatchCount = 0;
  let mixedQuantumCount = 0;
  let mixedSubmitThroughDrainMs = 0;

  return Object.freeze({
    record(
      commandBufferIndex: number,
      submitThroughDrainMs: number,
    ): void {
      // The final command buffer is readback and intentionally excluded.
      if (commandBufferIndex >= batches.length) return;
      if (
        commandBufferIndex !== recordedDecoderBatches ||
        !Number.isFinite(submitThroughDrainMs) ||
        submitThroughDrainMs < 0
      ) {
        throw new Error("OPT-0011 FP16 VAE family timing order changed");
      }
      const batch = batches[commandBufferIndex]!;
      recordedDecoderBatches += 1;
      decoderSubmitThroughDrainMs += submitThroughDrainMs;
      if (batch.family === null) {
        mixedBatchCount += 1;
        mixedQuantumCount += batch.quantumCount;
        mixedSubmitThroughDrainMs += submitThroughDrainMs;
        return;
      }
      const total = families[batch.family];
      total.batchCount += 1;
      total.quantumCount += batch.quantumCount;
      total.submitThroughDrainMs += submitThroughDrainMs;
      homogeneousBatchCount += 1;
      homogeneousQuantumCount += batch.quantumCount;
      homogeneousSubmitThroughDrainMs += submitThroughDrainMs;
    },
    finish(): AceOpt0011Fp16VaeWindowFamilyProfile {
      if (recordedDecoderBatches !== batches.length) {
        throw new Error("OPT-0011 FP16 VAE family timing is incomplete");
      }
      return Object.freeze({
        windowIndex: window.index,
        inputFrames: dispatch.plan.inputFrames,
        quantaPerCommandBuffer,
        decoderBatchCount: batches.length,
        decoderQuantumCount: dispatch.quanta.length,
        decoderSubmitThroughDrainMs,
        homogeneousBatchCount,
        homogeneousQuantumCount,
        homogeneousSubmitThroughDrainMs,
        mixedBatchCount,
        mixedQuantumCount,
        mixedSubmitThroughDrainMs,
        families: Object.freeze(Object.fromEntries(
          ACE_OPT_0011_FP16_VAE_PROFILE_FAMILIES.map((family) => [
            family,
            Object.freeze({ ...families[family] }),
          ]),
        )) as Readonly<Record<
          AceOpt0011Fp16VaeProfileFamily,
          AceOpt0011Fp16VaeFamilyTimingTotal
        >>,
      });
    },
  });
}

function emptyMutableFamilyTiming(): {
  batchCount: number;
  quantumCount: number;
  submitThroughDrainMs: number;
} {
  return { batchCount: 0, quantumCount: 0, submitThroughDrainMs: 0 };
}

function profileFamily(
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  quantumIndex: number,
): AceOpt0011Fp16VaeProfileFamily | null {
  const quantum = dispatch.quanta[quantumIndex];
  if (quantum === undefined) return null;
  const operationIndex = quantum.operationIndex;
  if (operationIndex === undefined || operationIndex === null) return null;
  const operation = dispatch.plan.operations[operationIndex];
  if (
    operation === undefined ||
    operation.kind !== quantum.operationKind
  ) return null;
  return operationProfileFamily(operation);
}

function operationProfileFamily(
  operation: AceVaeDecoderOperation,
): AceOpt0011Fp16VaeProfileFamily | null {
  switch (operation.kind) {
    case "conv1d":
      return operation.shape.kernelSize === 7
        ? "k7-conv1d"
        : operation.shape.kernelSize === 1
          ? "k1-conv1d"
          : null;
    case "conv-transpose1d":
      return "conv-transpose1d";
    case "snake":
      return "snake";
    case "add":
      return "add";
  }
}

function encodeQuantumBatch(
  device: GPUDevice,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  batchIndex: number,
  windowIndex: number,
  signal: AbortSignal,
  quantaPerCommandBuffer: number,
): GPUCommandBuffer {
  const first = batchIndex * quantaPerCommandBuffer;
  const end = Math.min(
    first + quantaPerCommandBuffer,
    dispatch.quanta.length,
  );
  if (first >= end) {
    throw new RangeError("OPT-0011 FP16 VAE command batch is empty");
  }
  const encoder = device.createCommandEncoder({
    label: `ace-opt-0011-fp16-vae-window-${windowIndex}-batch-${batchIndex}`,
  });
  const pass = encoder.beginComputePass({
    label: `ace-opt-0011-fp16-vae-window-${windowIndex}-batch-${batchIndex}-pass`,
  });
  for (let index = first; index < end; index += 1) {
    signal.throwIfAborted();
    dispatch.quanta[index]!.encode(pass);
  }
  pass.end();
  signal.throwIfAborted();
  return encoder.finish();
}

function encodeReadback(
  device: GPUDevice,
  output: GPUBuffer,
  readback: GPUBuffer,
  bytes: number,
  windowIndex: number,
): GPUCommandBuffer {
  if (
    !Number.isSafeInteger(bytes) ||
    bytes <= 0 ||
    bytes > output.size ||
    bytes > readback.size
  ) {
    throw new RangeError(
      `OPT-0011 FP16 VAE window ${windowIndex} exceeds readback capacity`,
    );
  }
  const encoder = device.createCommandEncoder({
    label: `ace-opt-0011-fp16-vae-window-${windowIndex}-readback`,
  });
  encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
  return encoder.finish();
}

function resolveQuantaPerCommandBuffer(value: number | undefined): number {
  const resolved = value ??
    ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
  if (
    resolved !== ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER &&
    resolved !== ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER
  ) {
    throw new RangeError("VAE quanta per command buffer must be 8 or 64");
  }
  return resolved;
}

async function mapDetachedOutput(
  readback: GPUBuffer,
  bytes: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  signal.throwIfAborted();
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    signal.throwIfAborted();
    return Float32Array.from(
      new Float32Array(readback.getMappedRange(0, bytes)),
    );
  } finally {
    readback.unmap();
  }
}

function requireBackendPlan(
  plan: AceVaeChunkedDecodePlan,
  maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames,
): void {
  if (
    maximumWindowFrames !== MAXIMUM_WINDOW_FRAMES &&
    maximumWindowFrames !==
      ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES
  ) {
    throw new Error(
      "FP16 VAE backend requires authenticated maxWindowFrames=512 or 2378",
    );
  }
  const expected = planAceVaeChunkedDecode(plan.latentFrames, {
    chunkFrames: maximumWindowFrames,
    overlapFrames: 64,
  });
  if (JSON.stringify(plan) !== JSON.stringify(expected)) {
    throw new Error(
      `FP16 VAE plan is not the exact C-${maximumWindowFrames}/64 chunk geometry`,
    );
  }
}

function requireBackendRuntimeSelection(
  options: AceOpt0011Fp16VaeChunkGpuBackendOptions,
): RequiredBackendRuntimeSelection {
  const untrusted = options as Readonly<{
    runtimeProfileId?: unknown;
    subgroupMinSize?: unknown;
    subgroupMaxSize?: unknown;
  }>;
  if (
    untrusted.runtimeProfileId === undefined ||
    untrusted.runtimeProfileId === "opt-0011-mixed-fp16-portable-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0028-mixed-fp16-portable-exact-packed-v1"
  ) {
    return Object.freeze({
      runtimeProfileId: untrusted.runtimeProfileId ??
        "opt-0011-mixed-fp16-portable-v1",
    });
  }
  if (
    untrusted.runtimeProfileId ===
      "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0028-mixed-fp16-fixed32-exact-packed-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0054-mixed-fp16-fixed32-revision7-v1" ||
    untrusted.runtimeProfileId ===
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
  ) {
    if (
      untrusted.subgroupMinSize !== 32 ||
      untrusted.subgroupMaxSize !== 32
    ) {
      throw new Error(
        "OPT-0011 FP16 VAE fixed32 hybrid requires authenticated 32/32 subgroups",
      );
    }
    return Object.freeze({
      runtimeProfileId: untrusted.runtimeProfileId,
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
  }
  throw new Error(
    "OPT-0011 FP16 VAE received unknown runtime profile " +
      String(untrusted.runtimeProfileId),
  );
}

function requireDispatchSet(
  plan: AceVaeChunkedDecodePlan,
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  runtimeProfile: AceVaeRuntimeProfile,
): void {
  const expectedTopology = requireKernelTopology(runtimeProfile);
  if (
    set.runtimeProfileId !== runtimeProfile.id ||
    set.kernelSetId !== runtimeProfile.kernelSetId ||
    !sameKernelTopology(set.kernelTopology, expectedTopology) ||
    set.dispatches.some((dispatch) =>
      dispatch.runtimeProfileId !== set.runtimeProfileId ||
      dispatch.kernelSetId !== set.kernelSetId ||
      !sameKernelTopology(dispatch.kernelTopology, expectedTopology) ||
      dispatch.precisionMap.profileId !== set.runtimeProfileId ||
      dispatch.precisionMap.kernelSetId !== set.kernelSetId
    ) ||
    JSON.stringify(set.topology.chunkPlan) !== JSON.stringify(plan) ||
    set.windows.length !== plan.windows.length ||
    set.windows.some((entry, index) =>
      JSON.stringify(entry.window) !== JSON.stringify(plan.windows[index]) ||
      entry.dispatch.plan.inputFrames !== entry.window.latentWindowFrames ||
      !set.dispatches.includes(entry.dispatch)
    )
  ) {
    throw new Error(
      "OPT-0011 FP16 VAE retained dispatch set diverged from the chunk plan",
    );
  }
}

function requireKernelTopology(
  runtimeProfile: AceVaeRuntimeProfile,
): AceOpt0011Fp16VaeDecoderKernelTopology {
  if (
    runtimeProfile.id === "opt-0011-mixed-fp16-portable-v1" &&
    runtimeProfile.kernelBackend === "portable-workgroup" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0028-mixed-fp16-portable-exact-packed-v1" &&
    runtimeProfile.kernelBackend === "portable-workgroup-exact-packed" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" &&
    runtimeProfile.kernelBackend === "fixed32-subgroup-k7-hybrid" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" &&
    runtimeProfile.kernelBackend ===
      "fixed32-subgroup-k7-congruent-transpose-hybrid" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id === "opt-0028-mixed-fp16-fixed32-exact-packed-v1" &&
    runtimeProfile.kernelBackend === "fixed32-subgroup-exact-packed" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" &&
    runtimeProfile.kernelBackend ===
      "fixed32-subgroup-exact-packed-shape-selected" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id === "opt-0054-mixed-fp16-fixed32-revision7-v1" &&
    runtimeProfile.kernelBackend ===
      "fixed32-subgroup-revision7-k4-shape-selected" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY;
  }
  if (
    runtimeProfile.id ===
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1" &&
    runtimeProfile.kernelBackend === "fixed32-subgroup-dual-k4-quality" &&
    runtimeProfile.kernelSetId ===
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY.id
  ) {
    return ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY;
  }
  throw new Error(
    "OPT-0011 FP16 VAE retained an unsupported runtime profile topology",
  );
}

function sameKernelTopology(
  actual: AceOpt0011Fp16VaeDecoderKernelTopology,
  expected: AceOpt0011Fp16VaeDecoderKernelTopology,
): boolean {
  return actual.id === expected.id &&
    actual.backend === expected.backend &&
    actual.ingress === expected.ingress &&
    actual.conv1dK1 === expected.conv1dK1 &&
    actual.conv1dK7 === expected.conv1dK7 &&
    actual.convTranspose1d === expected.convTranspose1d &&
    actual.snake === expected.snake &&
    actual.add === expected.add;
}

function requireMaximumWindowDeviceLimits(
  device: GPUDevice,
  maximumWindowFrames: AceOpt0011Fp16VaeBackendMaximumWindowFrames,
): void {
  const requiredWorkspaceBytes = maximumWindowFrames === MAXIMUM_WINDOW_FRAMES
    ? ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES
    : ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES;
  for (const name of ["maxBufferSize", "maxStorageBufferBindingSize"] as const) {
    const value = Number(device.limits[name]);
    if (
      !Number.isSafeInteger(value) ||
      value < requiredWorkspaceBytes
    ) {
      throw new RangeError(
        `FP16 VAE C${maximumWindowFrames} requires ${name} >= ` +
          `${requiredWorkspaceBytes}; received ${value}`,
      );
    }
  }
}

function snapshotLatents(
  plan: AceVaeChunkedDecodePlan,
  latents: Float32Array,
): Float32Array<ArrayBuffer> {
  const expected = plan.latentFrames * 64;
  if (!(latents instanceof Float32Array) || latents.length !== expected) {
    throw new RangeError(
      `OPT-0011 FP16 VAE final latent has ${latents.length} elements; ` +
        `expected ${expected}`,
    );
  }
  const result = new Float32Array(latents.length);
  for (let index = 0; index < latents.length; index += 1) {
    const value = latents[index]!;
    if (!Number.isFinite(value)) {
      throw new Error(`OPT-0011 FP16 VAE latent is non-finite at ${index}`);
    }
    result[index] = value;
  }
  return result;
}

function requirePlanWindow(
  plan: AceVaeChunkedDecodePlan,
  window: AceVaeDecodeWindow,
): void {
  const expected = plan.windows[window.index];
  if (
    expected === undefined ||
    (Object.keys(expected) as Array<keyof AceVaeDecodeWindow>).some((key) =>
      expected[key] !== window[key]
    )
  ) {
    throw new Error(
      `OPT-0011 FP16 VAE window ${window.index} is not in the fixed plan`,
    );
  }
}

function combineSignals(
  signals: readonly (AbortSignal | undefined)[],
): AbortSignal {
  const present = signals.filter((signal): signal is AbortSignal =>
    signal !== undefined
  );
  return present.length === 1 ? present[0]! : AbortSignal.any(present);
}

function requirePreparedBuffers(
  buffers: AceOpt0011Fp16VaePreparedBuffers,
): void {
  const required = [
    [buffers.stagingInput, 131_072],
    [buffers.decoderInput, 65_536],
    ...buffers.workspaces.map((buffer) => [buffer, 251_658_240] as const),
    [buffers.output, 7_864_320],
    [buffers.readback, 7_864_320],
  ] as const;
  if (required.some(([buffer, bytes]) => Number(buffer.size) < bytes)) {
    throw new RangeError(
      "OPT-0011 FP16 VAE prepared buffers are below the C512 envelope",
    );
  }
}

function destroyBuffers(
  buffers: AceOpt0011Fp16VaePreparedBuffers | undefined,
): void {
  if (buffers === undefined) return;
  buffers.stagingInput.destroy();
  buffers.decoderInput.destroy();
  for (const workspace of buffers.workspaces) workspace.destroy();
  buffers.output.destroy();
  buffers.readback.destroy();
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${label} contains an invalid value`);
    }
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}

function destroyedError(): DOMException {
  return new DOMException(
    "OPT-0011 FP16 VAE chunk GPU backend was destroyed",
    "AbortError",
  );
}
