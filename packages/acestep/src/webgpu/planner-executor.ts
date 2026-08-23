import type { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import type {
  AcePlannerDecodeBatch,
  AcePlannerGraphExecutor,
  AcePlannerLogitRange,
  AcePlannerPrefillBatch,
} from "../runtime/planner.js";
import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  AceFifoGraphOwner,
} from "../runtime/scheduler.js";
import { AceGpuArena, type AceArenaBufferPlan } from "./arena.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  aceActivationBytes,
  checkedAceProduct,
} from "./kernels/correctness-utils.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  AceCorrectnessPlannerModelRuntime,
  createAcePlannerModelControlData,
  planAcePlannerHeadSlices,
  planAcePlannerModel,
  resolveAcePlannerModelWeights,
  validateAcePlannerModelBatch,
  type AcePlannerModelBatch,
  type AcePlannerModelBindings,
  type AcePlannerModelDispatch,
  type AcePlannerModelPlan,
  type AcePlannerModelQuantum,
  type AcePlannerModelWeights,
} from "./planner-model.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  type AceQwen3BlockScratch,
} from "./qwen3.js";
import type {
  AceOpt0087PlannerDenseArm,
  AceOpt0087PlannerDenseRole,
  AceOpt0087PlannerDenseSelectionReason,
} from "./planner-dense-owner.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";

const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const F32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const STORAGE_ALIGNMENT = 256;

export interface AcePlannerGpuExecutorProgress {
  readonly phaseKind: "prefill" | "decode";
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly stage: "model" | "readback";
  readonly quantum: AcePlannerModelQuantum | null;
  /** Peak explicit GPU bytes across every fresh phase created so far. */
  readonly peakAccountedGpuBytes: number;
  /** Lifetime totals across every prefill/decode invocation. */
  readonly cumulativeQueueDrains: number;
  readonly cumulativeCooperativeIdleMs: number;
}

/** @internal Benchmark selector. Ordinary production deliberately omits it. */
export type AcePlannerOpt0085SchedulingProfile =
  | "depth1-epoch1"
  | "opt-0085-depth2-epoch4";

/** @internal Exact per-invocation topology emitted only to OPT-0085 harnesses. */
export interface AcePlannerOpt0085SchedulingDiagnostics {
  readonly schema: "ace-opt-0085-planner-scheduling-v1";
  readonly schedulingProfile: AcePlannerOpt0085SchedulingProfile;
  readonly phaseKind: "prefill" | "decode";
  readonly totalCommandBuffers: number;
  readonly commandBuffersSubmitted: number;
  readonly completionFenceRequestedCount: number;
  readonly completionFenceSettledCount: number;
  readonly completionFenceRejectedCount: number;
  readonly trueQueueDrainCount: number;
  readonly completionEpochCount: number;
  readonly cooperativeIdleTurns: number;
  readonly requestedCooperativeIdleMs: number;
  readonly maximumOutstandingCommandBuffers: number;
}

/** @internal One unchanged model quantum's fenced wall interval for OPT-0087. */
export interface AcePlannerOpt0087QuantumWallTiming {
  readonly index: number;
  readonly kind: AcePlannerModelQuantum["kind"];
  readonly layer: number | null;
  readonly primitiveCount: number;
  readonly submitThroughDrainWallMilliseconds: number;
}

/** @internal Serializable dense selection evidence for the OPT-0087 browser gate. */
export interface AcePlannerOpt0087DenseSelectionDiagnostics {
  readonly role: AceOpt0087PlannerDenseRole;
  readonly requestedArm: AceOpt0087PlannerDenseArm;
  readonly selectedArm: AceOpt0087PlannerDenseArm;
  readonly reason: AceOpt0087PlannerDenseSelectionReason;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
}

/** @internal Exact per-invocation package receipt emitted only to OPT-0087. */
export interface AcePlannerOpt0087InvocationDiagnostics {
  readonly schema: "ace-opt-0087-planner-package-invocation-v1";
  readonly arm: AceOpt0087PlannerDenseArm;
  readonly phaseKind: "decode";
  readonly modelQuantumCount: 33;
  readonly totalCommandBuffers: 34;
  readonly commandBuffersSubmitted: 34;
  readonly trueQueueDrainCount: 34;
  readonly cooperativeIdleTurns: 34;
  readonly requestedCooperativeIdleMs: 34;
  readonly maximumOutstandingCommandBuffers: 1;
  readonly readbackMapCount: 2;
  readonly readbackShardCount: number;
  readonly readbackByteLength: number;
  readonly cacheAppendReadbackByteLength: number;
  readonly cacheAppendLogicalByteLength: number;
  readonly cacheAppendCopyCount: number;
  readonly cacheAppendKeyValueWordCount: number;
  readonly cacheAppendValidityWordCount: number;
  /** Exact `[layer,K/V,row,KV-head,dimension]` bits followed by validity words. */
  readonly cacheAppendWords: Uint32Array;
  readonly logitRows: 1 | 2;
  readonly logitTokenCount: number;
  readonly accountedGpuBytes: number;
  readonly arenaBufferCount: number;
  readonly layerQuantumCount: 28;
  readonly tiedHeadQuantumCount: 2;
  readonly quantumTimings: readonly AcePlannerOpt0087QuantumWallTiming[];
  readonly transformerLayerWallMilliseconds: number;
  readonly tiedHeadWallMilliseconds: number;
  readonly readbackWallMilliseconds: number;
  readonly modelThroughReadbackWallMilliseconds: number;
  readonly writeStatusWords: readonly number[];
  readonly denseSelections:
    readonly AcePlannerOpt0087DenseSelectionDiagnostics[];
  readonly headQuantumSliceFirstRows: readonly (readonly number[])[];
}

export interface AcePlannerLogitReadbackShard {
  readonly sourceShardIndex: number;
  readonly firstRow: number;
  readonly destinationFirstRow: number;
  readonly rowCount: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface AcePlannerLogitReadbackLayout {
  readonly rows: 1 | 2;
  readonly firstTokenId: number;
  readonly tokenCount: number;
  readonly shards: readonly AcePlannerLogitReadbackShard[];
  readonly writeStatusByteOffset: number;
  readonly writeStatusByteLength: number;
  readonly byteLength: number;
}

/** Complete explicit memory owned by one fresh planner phase. */
export interface AcePlannerGpuPhaseMemoryPlan {
  readonly model: AcePlannerModelPlan;
  readonly residentWeightBytes: number;
  readonly allocatedCacheCapacity: number;
  readonly keyValueCacheBufferCount: number;
  readonly keyValueCacheBytes: number;
  readonly transientActivationBytes: number;
  readonly logitStorageBytes: number;
  readonly controlStorageBytes: number;
  readonly arenaBytes: number;
  readonly readbackBytes: number;
  readonly accountedGpuBytes: number;
  readonly arenaBufferCount: number;
  readonly readbackLayout: AcePlannerLogitReadbackLayout;
}

export interface AcePlannerGpuExecutorOptions {
  readonly device: GPUDevice;
  readonly modelProfile: AceModelProfileId;
  /** Ownership transfers at call entry, including on factory failure. */
  readonly ownedPlannerWeights: AceGpuTensorPhase;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AcePlannerGpuExecutorProgress) => void;
  /** @internal OPT-0085 browser-evidence hook; not part of product progress. */
  readonly onOpt0085Scheduling?: (
    diagnostics: AcePlannerOpt0085SchedulingDiagnostics,
  ) => void;
}

/** @internal Benchmark-only paired owner; ordinary product construction omits it. */
export interface AcePlannerOpt0087GpuExecutorOptions
  extends Omit<AcePlannerGpuExecutorOptions, "onOpt0085Scheduling"> {
  readonly onOpt0087Invocation: (
    diagnostics: AcePlannerOpt0087InvocationDiagnostics,
  ) => void;
}

type AcePlannerModelRuntime = Pick<
  AceCorrectnessPlannerModelRuntime,
  "createPlannerDispatch" | "destroy"
> & Partial<Pick<
  AceCorrectnessPlannerModelRuntime,
  "createPlannerDispatchForOpt0087"
>>;

export interface AcePlannerPreparedPhaseGpuResources {
  readonly batch: 1 | 2;
  readonly prefillTokens: number;
  readonly cacheCapacity: number;
  readonly bindings: AcePlannerModelBindings;
  readonly readback: GPUBuffer;
  /** @internal Allocated and mapped only by the paired OPT-0087 executor. */
  readonly opt0087CacheAppendReadback?: AcePlannerOpt0087CacheAppendReadback;
  readonly memory: AcePlannerGpuPhaseMemoryPlan;
  destroy(): void;
}

/** @internal Exact append-only cache evidence layout for one paired phase. */
interface AcePlannerOpt0087CacheAppendReadbackLayout {
  readonly rows: 1 | 2;
  readonly keyValueWordCount: number;
  readonly validityWordCount: number;
  readonly logicalByteLength: number;
  readonly byteLength: number;
  readonly copyCount: number;
}

/** @internal Paired-owner-only MAP_READ storage; never present in production. */
interface AcePlannerOpt0087CacheAppendReadback {
  readonly buffer: GPUBuffer;
  readonly layout: AcePlannerOpt0087CacheAppendReadbackLayout;
}

/** @internal Deterministic ownership seam used by coordinator tests. */
export interface AcePlannerPreparedGpuExecutorResources {
  readonly device: GPUDevice;
  readonly modelProfile: AceModelProfileId;
  readonly runtime: AcePlannerModelRuntime;
  readonly weights: AcePlannerModelWeights;
  readonly residentWeightBytes: number;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AcePlannerGpuExecutorProgress) => void;
  readonly yieldQueueIdle?: () => Promise<void>;
  /** @internal OPT-0085 deterministic topology capture. */
  readonly onOpt0085Scheduling?: (
    diagnostics: AcePlannerOpt0085SchedulingDiagnostics,
  ) => void;
  /** @internal Marks the sole paired pipeline owner for OPT-0087. */
  readonly opt0087PairedOwner?: true;
  /** @internal Successful-invocation evidence hook for OPT-0087. */
  readonly onOpt0087Invocation?: (
    diagnostics: AcePlannerOpt0087InvocationDiagnostics,
  ) => void;
  createPhase(
    batch: AcePlannerPrefillBatch,
  ): Promise<AcePlannerPreparedPhaseGpuResources>;
  /** Destroys the model runtime and the exclusively owned planner phase. */
  destroy(): void;
}

interface ActivePlannerPhase {
  readonly resources: AcePlannerPreparedPhaseGpuResources;
  readonly dispatches: Map<string, AcePlannerModelDispatch>;
  lastBatch: AcePlannerModelBatch | undefined;
  cachedTokens: number;
}

export class AcePlannerGpuDeviceLostError extends Error {
  override readonly name = "AcePlannerGpuDeviceLostError";

  constructor(info: Pick<GPUDeviceLostInfo, "reason" | "message">) {
    super(
      `ACE planner WebGPU device lost (${info.reason}): ` +
        (info.message || "no device message"),
    );
  }
}

/**
 * Stateful, exclusively owned implementation of `AcePlannerGraphExecutor`.
 *
 * A prefill transaction replaces all phase-sized graph storage. Decode calls
 * reuse that phase's exact K/V allocation and controls, but are FIFO-serialized
 * through the final readback. Ordinary production constructs and fully drains
 * one command buffer before the real queue-empty 1 ms interval and its
 * successor. The internal OPT-0085 arm may keep two unchanged singleton
 * buffers submitted within four-completion epochs.
 */
export class AcePlannerGpuExecutor implements AcePlannerGraphExecutor {
  private readonly graphOwner = new AceFifoGraphOwner();
  private readonly scheduler = new AceCooperativeGpuScheduler();
  private readonly lifetime = new AbortController();
  private readonly resourceAbortSignal: AbortSignal | undefined;
  private readonly resourceAbortListener: (() => void) | undefined;
  private activePhase: ActivePlannerPhase | undefined;
  private peakAccountedGpuBytes = 0;
  private cumulativeQueueDrains = 0;
  private cumulativeCooperativeIdleMs = 0;
  private destroyPromise: Promise<void> | undefined;
  private state: "live" | "destroying" | "destroyed" = "live";

  private constructor(
    private readonly resources: AcePlannerPreparedGpuExecutorResources,
  ) {
    validatePreparedExecutorResources(resources);
    resources.signal?.throwIfAborted();
    this.resourceAbortSignal = resources.signal;
    this.resourceAbortListener = resources.signal === undefined
      ? undefined
      : () => {
          const reason = resources.signal!.reason;
          if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
          void this.beginDestroy(reason).catch(() => undefined);
        };
    this.resourceAbortSignal?.addEventListener(
      "abort",
      this.resourceAbortListener!,
      { once: true },
    );

    const weakExecutor = new WeakRef(this);
    void resources.device.lost.then((info) => {
      const executor = weakExecutor.deref();
      if (executor === undefined || executor.state !== "live") return;
      const error = new AcePlannerGpuDeviceLostError(info);
      executor.lifetime.abort(error);
      void executor.beginDestroy(error).catch(() => undefined);
    });
  }

  static create(options: AcePlannerGpuExecutorOptions): AcePlannerGpuExecutor {
    return AcePlannerGpuExecutor.createOwned(options, false);
  }

  /** @internal One authenticated planner allocation with both frozen OPT-0087 pipelines. */
  static createForOpt0087(
    options: AcePlannerOpt0087GpuExecutorOptions,
  ): AcePlannerGpuExecutor {
    return AcePlannerGpuExecutor.createOwned(options, true);
  }

  private static createOwned(
    options: AcePlannerGpuExecutorOptions | AcePlannerOpt0087GpuExecutorOptions,
    pairedOpt0087: boolean,
  ): AcePlannerGpuExecutor {
    const owned = options.ownedPlannerWeights;
    let runtime: AceCorrectnessPlannerModelRuntime | undefined;
    let published = false;
    try {
      options.signal?.throwIfAborted();
      if (owned.phases.length !== 1 || owned.phases[0] !== "planner") {
        throw new Error(
          "ACE planner executor requires an exclusively resident planner phase; " +
            `got ${owned.phases.join("+")}`,
        );
      }
      const weights = resolveAcePlannerModelWeights(owned, options.modelProfile);
      if (pairedOpt0087 && options.modelProfile !== "reference-bf16") {
        throw new Error("ACE planner OPT-0087 requires reference-BF16");
      }
      runtime = pairedOpt0087
        ? AceCorrectnessPlannerModelRuntime.createForOpt0087(
            options.device,
            options.modelProfile,
          )
        : AceCorrectnessPlannerModelRuntime.create(
            options.device,
            options.modelProfile,
          );
      let rootDestroyed = false;
      const stableRuntime = runtime;
      const resources: AcePlannerPreparedGpuExecutorResources = {
        device: options.device,
        modelProfile: options.modelProfile,
        runtime: stableRuntime,
        weights,
        residentWeightBytes: owned.residentBytes,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        ...(options.onProgress === undefined
          ? {}
          : { onProgress: options.onProgress }),
        ...(!("onOpt0085Scheduling" in options) ||
          options.onOpt0085Scheduling === undefined
          ? {}
          : { onOpt0085Scheduling: options.onOpt0085Scheduling }),
        ...(pairedOpt0087
          ? {
              opt0087PairedOwner: true as const,
              onOpt0087Invocation:
                (options as AcePlannerOpt0087GpuExecutorOptions)
                  .onOpt0087Invocation,
            }
          : {}),
        createPhase: async (batch) => await createConcretePlannerPhase(
          options.device,
          options.modelProfile,
          weights,
          owned.residentBytes,
          batch,
          pairedOpt0087,
        ),
        destroy(): void {
          if (rootDestroyed) return;
          rootDestroyed = true;
          stableRuntime.destroy();
          owned.destroy();
        },
      };
      const executor = new AcePlannerGpuExecutor(resources);
      published = true;
      return executor;
    } finally {
      if (!published) {
        runtime?.destroy();
        owned.destroy();
      }
    }
  }

  /** @internal Construct the real coordinator over deterministic fake buffers. */
  static fromPreparedResources(
    resources: AcePlannerPreparedGpuExecutorResources,
  ): AcePlannerGpuExecutor {
    try {
      return new AcePlannerGpuExecutor(resources);
    } catch (error) {
      resources.destroy();
      throw error;
    }
  }

  prefill(
    batch: AcePlannerPrefillBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    return this.invoke(batch, logitRange);
  }

  decode(
    batch: AcePlannerDecodeBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    return this.invoke(batch, logitRange);
  }

  /** @internal OPT-0085 explicit arm over one authenticated weight owner. */
  prefillForOpt0085(
    schedulingProfile: AcePlannerOpt0085SchedulingProfile,
    batch: AcePlannerPrefillBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.requireOpt0085Experiment(schedulingProfile);
    return this.invoke(batch, logitRange, schedulingProfile);
  }

  /** @internal OPT-0085 explicit arm over one authenticated weight owner. */
  decodeForOpt0085(
    schedulingProfile: AcePlannerOpt0085SchedulingProfile,
    batch: AcePlannerDecodeBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.requireOpt0085Experiment(schedulingProfile);
    return this.invoke(batch, logitRange, schedulingProfile);
  }

  /** @internal Frozen full-head A/B arm over the paired OPT-0087 owner. */
  decodeForOpt0087(
    arm: AceOpt0087PlannerDenseArm,
    batch: AcePlannerDecodeBatch,
  ): Promise<readonly ArrayLike<number>[]> {
    this.requireOpt0087Experiment(arm);
    return this.invoke(batch, undefined, undefined, arm);
  }

  destroy(reason: unknown = destroyedError()): Promise<void> {
    return this.beginDestroy(reason);
  }

  /** @internal OPT-0082 replays only the tied head over the last hidden rows. */
  async replayTiedHeadForOpt0082(
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly Float32Array[]> {
    this.requireLive();
    this.lifetime.signal.throwIfAborted();
    const lease = await this.graphOwner.acquire(this.lifetime.signal);
    try {
      const phase = this.activePhase;
      const batch = phase?.lastBatch;
      if (phase === undefined || batch === undefined) {
        throw new Error("OPT-0082 tied-head replay requires a completed planner call");
      }
      this.requireOpt0082Profile(logitRange, batch.rows);
      const dispatch = await this.dispatchFor(phase, batch, logitRange);
      validateDispatchOrder(dispatch, batch, logitRange);
      const headQuanta = dispatch.quanta.filter(
        (quantum) => quantum.kind === "tied-lm-head",
      );
      if (headQuanta.length === 0) {
        throw new Error("OPT-0082 tied-head replay has no head quantum");
      }
      const readbackLayout = createPlannerReadbackLayout(
        this.resources.modelProfile,
        batch.rows as 1 | 2,
        logitRange,
      );
      return await this.runAndReadback(
        phase.resources,
        dispatch,
        false,
        readbackLayout,
        headQuanta,
      );
    } finally {
      lease.release();
    }
  }

  private async invoke(
    batch: AcePlannerModelBatch,
    logitRange?: AcePlannerLogitRange,
    opt0085SchedulingProfile?: AcePlannerOpt0085SchedulingProfile,
    opt0087Arm?: AceOpt0087PlannerDenseArm,
  ): Promise<readonly Float32Array[]> {
    this.requireLive();
    this.requireOpt0082Profile(logitRange, batch.rows);
    validateAcePlannerModelBatch(batch);
    this.lifetime.signal.throwIfAborted();
    const lease = await this.graphOwner.acquire(this.lifetime.signal);
    let phase: ActivePlannerPhase | undefined;
    try {
      this.requireLive();
      this.lifetime.signal.throwIfAborted();
      if (batch.kind === "prefill") {
        this.releaseActivePhase();
        const created = await this.resources.createPhase(batch);
        try {
          validatePreparedPhaseResources(
            created,
            batch,
            this.resources.modelProfile,
            this.resources.opt0087PairedOwner === true,
          );
        } catch (error) {
          created.destroy();
          throw error;
        }
        phase = {
          resources: created,
          dispatches: new Map(),
          lastBatch: undefined,
          cachedTokens: 0,
        };
        this.peakAccountedGpuBytes = Math.max(
          this.peakAccountedGpuBytes,
          accountedPlannerPhaseGpuBytes(created),
        );
        this.activePhase = phase;
      } else {
        phase = this.requireDecodePhase(batch);
      }

      const controls = createAcePlannerModelControlData(batch);
      uploadPlannerInvocation(
        this.resources.device.queue,
        phase.resources.bindings,
        batch,
        controls,
      );
      this.lifetime.signal.throwIfAborted();
      const dispatch = await this.dispatchFor(
        phase,
        batch,
        logitRange,
        opt0087Arm,
      );
      validateDispatchOrder(dispatch, batch, logitRange);
      this.lifetime.signal.throwIfAborted();
      const readbackLayout = createPlannerReadbackLayout(
        this.resources.modelProfile,
        batch.rows as 1 | 2,
        logitRange,
      );
      const opt0087CacheAppendPosition = opt0087Arm === undefined
        ? undefined
        : batch.kind === "decode"
          ? batch.cachedTokensBeforeAppend
          : (() => {
              throw new Error("OPT-0087 cache evidence requires decode");
            })();
      const logits = await this.runAndReadback(
        phase.resources,
        dispatch,
        controls.clearCacheBeforeDispatch,
        readbackLayout,
        dispatch.quanta,
        opt0085SchedulingProfile,
        opt0087Arm,
        opt0087CacheAppendPosition,
      );
      phase.cachedTokens = batch.kind === "prefill"
        ? batch.tokens
        : batch.cachedTokensBeforeAppend + 1;
      phase.lastBatch = batch;
      return logits;
    } catch (error) {
      // Prefill never publishes a partial fresh cache. A decode error may have
      // happened after an append, so that cache is also unusable. Every GPU
      // submission above is drained before it rejects.
      if (phase !== undefined && phase === this.activePhase) {
        this.releasePhase(phase);
      }
      throw error;
    } finally {
      lease.release();
    }
  }

  private requireDecodePhase(batch: AcePlannerDecodeBatch): ActivePlannerPhase {
    const phase = this.activePhase;
    if (phase === undefined) {
      throw new Error("ACE planner decode requires a successful fresh prefill");
    }
    if (
      phase.resources.batch !== batch.rows ||
      phase.resources.cacheCapacity !== batch.cacheCapacity
    ) {
      throw new Error("ACE planner decode does not match its resident phase geometry");
    }
    if (phase.cachedTokens !== batch.cachedTokensBeforeAppend) {
      throw new Error(
        `ACE planner decode starts at ${batch.cachedTokensBeforeAppend}; ` +
          `resident cache ends at ${phase.cachedTokens}`,
      );
    }
    return phase;
  }

  private requireOpt0082Profile(
    logitRange: AcePlannerLogitRange | undefined,
    rows: number,
  ): void {
    if (
      logitRange !== undefined &&
      this.resources.modelProfile !== "reference-bf16"
    ) {
      throw new Error("OPT-0082 compact logits require reference-BF16");
    }
    if (logitRange !== undefined && rows !== 2) {
      throw new Error("OPT-0082 compact logits require two planner rows");
    }
  }

  private async dispatchFor(
    phase: ActivePlannerPhase,
    batch: AcePlannerModelBatch,
    logitRange?: AcePlannerLogitRange,
    opt0087Arm?: AceOpt0087PlannerDenseArm,
  ): Promise<AcePlannerModelDispatch> {
    const rangeLabel = logitRange === undefined
      ? "full"
      : `${logitRange.firstTokenId}-${logitRange.tokenCount}`;
    const key = `${opt0087Arm ?? "ordinary"}:${batch.kind}:${rangeLabel}`;
    const cached = phase.dispatches.get(key);
    if (cached !== undefined) return cached;
    const label = batch.kind === "prefill"
      ? `ace-planner-prefill-${batch.rows}x${batch.tokens}-capacity-${batch.cacheCapacity}-${rangeLabel}`
      : `ace-planner-decode-${batch.rows}x1-capacity-${batch.cacheCapacity}-${rangeLabel}`;
    const dispatch = opt0087Arm === undefined
      ? await this.resources.runtime.createPlannerDispatch(
          label,
          batch,
          phase.resources.bindings,
          logitRange,
        )
      : await this.resources.runtime.createPlannerDispatchForOpt0087!(
          label,
          opt0087Arm,
          batch,
          phase.resources.bindings,
          logitRange,
        );
    phase.dispatches.set(key, dispatch);
    return dispatch;
  }

  private async runAndReadback(
    phase: AcePlannerPreparedPhaseGpuResources,
    dispatch: AcePlannerModelDispatch,
    clearCache: boolean,
    readbackLayout: AcePlannerLogitReadbackLayout,
    quanta: readonly AcePlannerModelQuantum[] = dispatch.quanta,
    opt0085SchedulingProfile?: AcePlannerOpt0085SchedulingProfile,
    opt0087Arm?: AceOpt0087PlannerDenseArm,
    opt0087CacheAppendPosition?: number,
  ): Promise<readonly Float32Array[]> {
    if (
      (opt0087Arm === undefined) !==
        (opt0087CacheAppendPosition === undefined) ||
      (opt0087Arm !== undefined &&
        phase.opt0087CacheAppendReadback === undefined)
    ) {
      throw new Error("OPT-0087 cache append evidence resources are incomplete");
    }
    if (
      opt0085SchedulingProfile ===
        "opt-0085-depth2-epoch4"
    ) {
      return await this.runAndReadbackDepth2Epoch4(
        phase,
        dispatch,
        clearCache,
        readbackLayout,
        quanta,
      );
    }
    const signal = this.lifetime.signal;
    const totalCommandBuffers = quanta.length + 1;
    let completedCommandBuffers = 0;
    let queueDrains = 0;
    let cooperativeIdleMs = 0;
    const opt0087Started = opt0087Arm === undefined
      ? undefined
      : performance.now();
    const opt0087QuantumTimings: AcePlannerOpt0087QuantumWallTiming[] = [];

    for (let index = 0; index < quanta.length; index += 1) {
      signal.throwIfAborted();
      const quantum = quanta[index]!;
      const quantumStarted = opt0087Arm === undefined
        ? undefined
        : performance.now();
      const commandBuffer = encodePlannerQuantum(
        this.resources.device,
        phase,
        quantum,
        clearCache && index === 0,
      );
      const scheduled = await this.scheduler.run({
        queue: this.resources.device.queue,
        commandBuffers: [commandBuffer],
        signal,
        ownerSignal: signal,
      });
      completedCommandBuffers += scheduled.commandBuffersSubmitted;
      queueDrains += scheduled.queueDrains;
      if (quantumStarted !== undefined) {
        opt0087QuantumTimings.push(Object.freeze({
          index,
          kind: quantum.kind,
          layer: quantum.layer,
          primitiveCount: quantum.primitiveCount,
          submitThroughDrainWallMilliseconds:
            performance.now() - quantumStarted,
        }));
      }

      // A readback command still remains after the last model quantum, so all
      // model commands are non-final and receive the real queue-empty gap.
      const idle = (this.resources.yieldQueueIdle ?? yieldQueueIdle)();
      cooperativeIdleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
      try {
        this.cumulativeQueueDrains += scheduled.queueDrains;
        this.cumulativeCooperativeIdleMs +=
          ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
        this.resources.onProgress?.({
          phaseKind: dispatch.plan.kind,
          completedCommandBuffers,
          totalCommandBuffers,
          queueDrains,
          cooperativeIdleMs,
          stage: "model",
          quantum,
          peakAccountedGpuBytes: this.peakAccountedGpuBytes,
          cumulativeQueueDrains: this.cumulativeQueueDrains,
          cumulativeCooperativeIdleMs: this.cumulativeCooperativeIdleMs,
        });
      } catch (error) {
        await idle;
        throw error;
      }
      await idle;
      signal.throwIfAborted();
    }

    const readbackStarted = opt0087Arm === undefined
      ? undefined
      : performance.now();
    const copy = encodePlannerReadback(
      this.resources.device,
      phase,
      readbackLayout,
      opt0087CacheAppendPosition,
    );
    const scheduled = await this.scheduler.run({
      queue: this.resources.device.queue,
      commandBuffers: [copy],
      signal,
      ownerSignal: signal,
    });
    completedCommandBuffers += scheduled.commandBuffersSubmitted;
    queueDrains += scheduled.queueDrains;
    this.cumulativeQueueDrains += scheduled.queueDrains;
    const readbackIdle = (this.resources.yieldQueueIdle ?? yieldQueueIdle)();
    cooperativeIdleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
    this.cumulativeCooperativeIdleMs +=
      ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
    let logits: readonly Float32Array[];
    let opt0087WriteStatusWords: readonly number[] | undefined;
    let opt0087CacheAppendWords: Uint32Array | undefined;
    try {
      this.resources.onProgress?.({
        phaseKind: dispatch.plan.kind,
        completedCommandBuffers,
        totalCommandBuffers,
        queueDrains,
        cooperativeIdleMs,
        stage: "readback",
        quantum: null,
        peakAccountedGpuBytes: this.peakAccountedGpuBytes,
        cumulativeQueueDrains: this.cumulativeQueueDrains,
        cumulativeCooperativeIdleMs: this.cumulativeCooperativeIdleMs,
      });
      if (opt0085SchedulingProfile !== undefined) {
        this.resources.onOpt0085Scheduling!(Object.freeze({
          schema: "ace-opt-0085-planner-scheduling-v1",
          schedulingProfile: "depth1-epoch1",
          phaseKind: dispatch.plan.kind,
          totalCommandBuffers,
          commandBuffersSubmitted: totalCommandBuffers,
          completionFenceRequestedCount: totalCommandBuffers,
          completionFenceSettledCount: totalCommandBuffers,
          completionFenceRejectedCount: 0,
          trueQueueDrainCount: totalCommandBuffers,
          completionEpochCount: totalCommandBuffers,
          cooperativeIdleTurns: totalCommandBuffers,
          requestedCooperativeIdleMs: cooperativeIdleMs,
          maximumOutstandingCommandBuffers: 1,
        }));
      }
      if (opt0087Arm === undefined) {
        logits = await mapPlannerLogits(
          phase.readback,
          readbackLayout,
          this.resources.modelProfile,
          signal,
        );
      } else {
        const mapped = await mapPlannerLogitsForOpt0087(
          phase.readback,
          readbackLayout,
          this.resources.modelProfile,
          signal,
        );
        logits = mapped.logits;
        opt0087WriteStatusWords = mapped.writeStatusWords;
        opt0087CacheAppendWords = await mapPlannerCacheAppendForOpt0087(
          phase.opt0087CacheAppendReadback!,
          signal,
        );
      }
    } finally {
      await readbackIdle;
    }
    signal.throwIfAborted();
    if (opt0087Arm !== undefined) {
      this.publishOpt0087Diagnostics(
        opt0087Arm,
        dispatch,
        opt0087QuantumTimings,
        totalCommandBuffers,
        completedCommandBuffers,
        queueDrains,
        cooperativeIdleMs,
        readbackStarted!,
        opt0087Started!,
        opt0087WriteStatusWords!,
        opt0087CacheAppendWords!,
        readbackLayout,
        phase.memory,
      );
    }
    return logits!;
  }

  private async runAndReadbackDepth2Epoch4(
    phase: AcePlannerPreparedPhaseGpuResources,
    dispatch: AcePlannerModelDispatch,
    clearCache: boolean,
    readbackLayout: AcePlannerLogitReadbackLayout,
    quanta: readonly AcePlannerModelQuantum[],
  ): Promise<readonly Float32Array[]> {
    const signal = this.lifetime.signal;
    const totalCommandBuffers = quanta.length + 1;
    const scheduled = await this.scheduler.runLazyDepth2Epoch4({
      queue: this.resources.device.queue,
      commandBufferCount: totalCommandBuffers,
      phaseCommandBufferCounts: [totalCommandBuffers],
      signal,
      ownerSignal: signal,
      device: this.resources.device,
      createCommandBuffer: (index) => {
        if (index === quanta.length) {
          return encodePlannerReadback(
            this.resources.device,
            phase,
            readbackLayout,
          );
        }
        const quantum = quanta[index]!;
        return encodePlannerQuantum(
          this.resources.device,
          phase,
          quantum,
          clearCache && index === 0,
        );
      },
      yieldQueueIdle: async () => {
        this.cumulativeCooperativeIdleMs +=
          ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
        await (this.resources.yieldQueueIdle ?? yieldQueueIdle)();
      },
      onCommandBufferCompleted: (timing, progress) => {
        if (timing.trueQueueDrain) this.cumulativeQueueDrains += 1;
        const readback = timing.commandBufferIndex === quanta.length;
        this.resources.onProgress?.({
          phaseKind: dispatch.plan.kind,
          completedCommandBuffers: progress.completedCommandBuffers,
          totalCommandBuffers,
          queueDrains: progress.trueQueueDrainCount,
          cooperativeIdleMs: progress.requestedCooperativeIdleMs,
          stage: readback ? "readback" : "model",
          quantum: readback ? null : quanta[timing.commandBufferIndex]!,
          peakAccountedGpuBytes: this.peakAccountedGpuBytes,
          cumulativeQueueDrains: this.cumulativeQueueDrains,
          cumulativeCooperativeIdleMs: this.cumulativeCooperativeIdleMs,
        });
      },
    });
    signal.throwIfAborted();
    if (
      scheduled.commandBuffersSubmitted !== totalCommandBuffers ||
      scheduled.completionFenceRequestedCount !== totalCommandBuffers ||
      scheduled.completionFenceSettledCount !== totalCommandBuffers ||
      scheduled.completionFenceRejectedCount !== 0 ||
      scheduled.trueQueueDrainCount !== Math.ceil(totalCommandBuffers / 4) ||
      scheduled.completionEpochCount !== Math.ceil(totalCommandBuffers / 4) ||
      scheduled.cooperativeIdleTurns !== Math.ceil(totalCommandBuffers / 4) - 1 ||
      scheduled.requestedCooperativeIdleMs !==
        scheduled.cooperativeIdleTurns * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS ||
      scheduled.maximumOutstandingCommandBuffers !==
        (totalCommandBuffers === 1 ? 1 : 2)
    ) {
      throw new Error("OPT-0085 planner scheduling topology did not reconcile");
    }
    this.resources.onOpt0085Scheduling!(Object.freeze({
      schema: "ace-opt-0085-planner-scheduling-v1",
      schedulingProfile: "opt-0085-depth2-epoch4",
      phaseKind: dispatch.plan.kind,
      totalCommandBuffers,
      commandBuffersSubmitted: scheduled.commandBuffersSubmitted,
      completionFenceRequestedCount: scheduled.completionFenceRequestedCount,
      completionFenceSettledCount: scheduled.completionFenceSettledCount,
      completionFenceRejectedCount: scheduled.completionFenceRejectedCount,
      trueQueueDrainCount: scheduled.trueQueueDrainCount,
      completionEpochCount: scheduled.completionEpochCount,
      cooperativeIdleTurns: scheduled.cooperativeIdleTurns,
      requestedCooperativeIdleMs: scheduled.requestedCooperativeIdleMs,
      maximumOutstandingCommandBuffers:
        scheduled.maximumOutstandingCommandBuffers,
    }));

    // The final command is the readback copy. The depth-two scheduler resolves
    // only after its terminal cumulative fence is fulfilled, so mapping cannot
    // race the tied head or the copy despite the permitted one-buffer overlap.
    return await mapPlannerLogits(
      phase.readback,
      readbackLayout,
      this.resources.modelProfile,
      signal,
    );
  }

  /** @internal Stripped with the benchmark-only OPT-0087 surface. */
  private publishOpt0087Diagnostics(
    arm: AceOpt0087PlannerDenseArm,
    dispatch: AcePlannerModelDispatch,
    quantumTimings: readonly AcePlannerOpt0087QuantumWallTiming[],
    totalCommandBuffers: number,
    commandBuffersSubmitted: number,
    trueQueueDrainCount: number,
    requestedCooperativeIdleMs: number,
    readbackStarted: number,
    invocationStarted: number,
    writeStatusWords: readonly number[],
    cacheAppendWords: Uint32Array,
    readbackLayout: AcePlannerLogitReadbackLayout,
    memory: AcePlannerGpuPhaseMemoryPlan,
  ): void {
    const completed = performance.now();
    const layers = quantumTimings.filter(({ kind }) => kind === "layer");
    const heads = quantumTimings.filter(({ kind }) => kind === "tied-lm-head");
    const selections = dispatch.opt0087DenseSelections;
    const headGroups = dispatch.opt0087HeadQuantumSliceFirstRows;
    const cacheAppendLayout = createOpt0087CacheAppendReadbackLayout(
      readbackLayout.rows,
    );
    if (
      dispatch.plan.kind !== "decode" ||
      quantumTimings.length !== 33 ||
      totalCommandBuffers !== 34 ||
      commandBuffersSubmitted !== 34 ||
      trueQueueDrainCount !== 34 ||
      requestedCooperativeIdleMs !== 34 ||
      layers.length !== 28 ||
      heads.length !== 2 ||
      selections === undefined ||
      headGroups === undefined ||
      headGroups.length !== 2 ||
      cacheAppendWords.length * U32_BYTES !==
        cacheAppendLayout.logicalByteLength
    ) throw new Error("OPT-0087 planner package topology did not reconcile");
    const sumWall = (
      values: readonly AcePlannerOpt0087QuantumWallTiming[],
    ): number => values.reduce(
      (sum, timing) => sum + timing.submitThroughDrainWallMilliseconds,
      0,
    );
    this.resources.onOpt0087Invocation!(Object.freeze({
      schema: "ace-opt-0087-planner-package-invocation-v1",
      arm,
      phaseKind: "decode",
      modelQuantumCount: 33,
      totalCommandBuffers: 34,
      commandBuffersSubmitted: 34,
      trueQueueDrainCount: 34,
      cooperativeIdleTurns: 34,
      requestedCooperativeIdleMs: 34,
      maximumOutstandingCommandBuffers: 1,
      readbackMapCount: 2,
      readbackShardCount: readbackLayout.shards.length,
      readbackByteLength: readbackLayout.byteLength,
      cacheAppendReadbackByteLength: cacheAppendLayout.byteLength,
      cacheAppendLogicalByteLength: cacheAppendLayout.logicalByteLength,
      cacheAppendCopyCount: cacheAppendLayout.copyCount,
      cacheAppendKeyValueWordCount: cacheAppendLayout.keyValueWordCount,
      cacheAppendValidityWordCount: cacheAppendLayout.validityWordCount,
      cacheAppendWords,
      logitRows: readbackLayout.rows,
      logitTokenCount: readbackLayout.tokenCount,
      accountedGpuBytes: checkedSum([
        memory.accountedGpuBytes,
        cacheAppendLayout.byteLength,
      ], "OPT-0087 accounted GPU bytes"),
      arenaBufferCount: memory.arenaBufferCount,
      layerQuantumCount: 28,
      tiedHeadQuantumCount: 2,
      quantumTimings: Object.freeze([...quantumTimings]),
      transformerLayerWallMilliseconds: sumWall(layers),
      tiedHeadWallMilliseconds: sumWall(heads),
      readbackWallMilliseconds: completed - readbackStarted,
      modelThroughReadbackWallMilliseconds: completed - invocationStarted,
      writeStatusWords: Object.freeze([...writeStatusWords]),
      denseSelections: Object.freeze(selections.map((selection) =>
        Object.freeze({
          role: selection.role,
          requestedArm: selection.requestedArm,
          selectedArm: selection.selectedArm,
          reason: selection.reason,
          rows: selection.shape.rows,
          inner: selection.shape.inner,
          columns: selection.shape.columns,
        })
      )),
      headQuantumSliceFirstRows: Object.freeze(headGroups.map((group) =>
        Object.freeze([...group])
      )),
    }));
  }

  private beginDestroy(reason: unknown): Promise<void> {
    if (this.destroyPromise !== undefined) return this.destroyPromise;
    this.state = "destroying";
    if (
      this.resourceAbortSignal !== undefined &&
      this.resourceAbortListener !== undefined
    ) {
      this.resourceAbortSignal.removeEventListener(
        "abort",
        this.resourceAbortListener,
      );
    }
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    this.destroyPromise = (async () => {
      try {
        await this.graphOwner.dispose();
        await this.scheduler.dispose();
      } finally {
        try {
          this.releaseActivePhase();
          this.resources.destroy();
        } finally {
          this.state = "destroyed";
        }
      }
    })();
    return this.destroyPromise;
  }

  private releaseActivePhase(): void {
    const phase = this.activePhase;
    if (phase === undefined) return;
    this.activePhase = undefined;
    phase.resources.destroy();
  }

  private releasePhase(phase: ActivePlannerPhase): void {
    if (this.activePhase === phase) this.activePhase = undefined;
    phase.resources.destroy();
  }

  private requireLive(): void {
    if (this.state !== "live") {
      throw new DOMException(
        `ACE planner GPU executor is ${this.state}`,
        "InvalidStateError",
      );
    }
  }

  private requireOpt0085Experiment(
    schedulingProfile: AcePlannerOpt0085SchedulingProfile,
  ): void {
    if (
      schedulingProfile !== "depth1-epoch1" &&
      schedulingProfile !== "opt-0085-depth2-epoch4"
    ) {
      throw new TypeError("ACE planner OPT-0085 scheduling profile is invalid");
    }
    if (
      this.resources.modelProfile !== "reference-bf16" ||
      this.resources.onOpt0085Scheduling === undefined
    ) {
      throw new Error(
        "ACE planner OPT-0085 explicit arms require reference-BF16 diagnostics",
      );
    }
  }

  /** @internal Stripped with the benchmark-only OPT-0087 surface. */
  private requireOpt0087Experiment(arm: AceOpt0087PlannerDenseArm): void {
    if (arm !== "generic-a" && arm !== "direct-b") {
      throw new TypeError("ACE planner OPT-0087 dense arm is invalid");
    }
    if (
      this.resources.modelProfile !== "reference-bf16" ||
      this.resources.opt0087PairedOwner !== true ||
      this.resources.onOpt0087Invocation === undefined ||
      typeof this.resources.runtime.createPlannerDispatchForOpt0087 !==
        "function"
    ) {
      throw new Error(
        "ACE planner OPT-0087 explicit arms require paired reference-BF16 diagnostics",
      );
    }
  }
}

export function planAcePlannerGpuPhaseMemory(
  modelProfile: AceModelProfileId,
  prefill: AcePlannerPrefillBatch,
  residentWeightBytes: number,
): AcePlannerGpuPhaseMemoryPlan {
  requireModelProfile(modelProfile);
  validateAcePlannerModelBatch(prefill);
  requireNonNegativeSafeInteger(
    residentWeightBytes,
    "ACE planner resident weight bytes",
  );
  const model = planAcePlannerModel(modelProfile, prefill);
  const layout = createPlannerArenaLayout(modelProfile, prefill, model);
  const readbackLayout = createPlannerReadbackLayout(
    modelProfile,
    prefill.rows as 1 | 2,
  );
  const arenaBytes = checkedSum(
    layout.plans.map(({ byteLength }) => byteLength),
    "ACE planner arena bytes",
  );
  const accountedGpuBytes = checkedSum(
    [residentWeightBytes, arenaBytes, readbackLayout.byteLength],
    "ACE planner accounted GPU bytes",
  );
  return Object.freeze({
    model,
    residentWeightBytes,
    allocatedCacheCapacity: prefill.cacheCapacity,
    keyValueCacheBufferCount: ACE_PLANNER_QWEN3_CONFIG.layerCount * 2,
    keyValueCacheBytes: model.kvCacheBytes,
    transientActivationBytes: model.transientActivationBytes,
    logitStorageBytes: model.logitsBytes,
    controlStorageBytes: model.cacheValidityBytes + model.invocationControlBytes,
    arenaBytes,
    readbackBytes: readbackLayout.byteLength,
    accountedGpuBytes,
    arenaBufferCount: layout.plans.length,
    readbackLayout,
  });
}

/** Detach and reconstruct the requested sharded LM-head interval in token order. */
export function reconstructAcePlannerLogits(
  mappedBytes: ArrayBuffer,
  layout: AcePlannerLogitReadbackLayout,
  modelProfile: AceModelProfileId,
): readonly Float32Array[] {
  requireModelProfile(modelProfile);
  if (!(mappedBytes instanceof ArrayBuffer) || mappedBytes.byteLength < layout.byteLength) {
    throw new RangeError("ACE planner mapped readback is smaller than its layout");
  }
  const status = new Uint32Array(
    mappedBytes,
    layout.writeStatusByteOffset,
    layout.rows,
  );
  // The composer binds one CPU-validated row-start/capacity tuple to all 28
  // layers. Every layer therefore admits or rejects the identical physical
  // range; the final overwrite cannot hide an earlier range rejection. Device
  // loss and queue-drain failure are handled independently of this status bit.
  for (let row = 0; row < status.length; row += 1) {
    if (status[row] !== 1) {
      throw new Error(`ACE planner cache append failed for physical row ${row}`);
    }
  }
  const logits = Array.from(
    { length: layout.rows },
    () => new Float32Array(layout.tokenCount),
  );
  for (const shard of layout.shards) {
    const elements = layout.rows * shard.rowCount;
    if (modelProfile === "reference-bf16") {
      const values = new Float32Array(
        mappedBytes,
        shard.byteOffset,
        elements,
      );
      copyLogitShard(logits, values, shard);
    } else if (modelProfile === "raw-fp16") {
      const bits = new Uint16Array(mappedBytes, shard.byteOffset, elements);
      for (let row = 0; row < layout.rows; row += 1) {
        const destination = logits[row]!;
        const sourceStart = row * shard.rowCount;
        for (let column = 0; column < shard.rowCount; column += 1) {
          destination[shard.destinationFirstRow + column] = fp16BitsToNumber(
            bits[sourceStart + column]!,
          );
        }
      }
    } else {
      throw new TypeError(`Unknown ACE planner model profile ${String(modelProfile)}`);
    }
  }
  return Object.freeze(logits);
}

function createOpt0087CacheAppendReadbackLayout(
  rows: 1 | 2,
): AcePlannerOpt0087CacheAppendReadbackLayout {
  const keyValueWordCount = checkedAceProduct([
    ACE_PLANNER_QWEN3_CONFIG.layerCount,
    2,
    rows,
    ACE_PLANNER_QWEN3_CONFIG.keyValueHeads,
    ACE_PLANNER_QWEN3_CONFIG.headDimension,
  ], "OPT-0087 cache append words");
  const validityWordCount = rows;
  const logicalByteLength = checkedAceProduct([
    keyValueWordCount + validityWordCount,
    U32_BYTES,
  ], "OPT-0087 cache append bytes");
  const copyCount =
    ACE_PLANNER_QWEN3_CONFIG.layerCount * 2 * rows *
      ACE_PLANNER_QWEN3_CONFIG.keyValueHeads + rows;
  return Object.freeze({
    rows,
    keyValueWordCount,
    validityWordCount,
    logicalByteLength,
    byteLength: align(logicalByteLength, STORAGE_ALIGNMENT),
    copyCount,
  });
}

async function createConcretePlannerPhase(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  weights: AcePlannerModelWeights,
  residentWeightBytes: number,
  prefill: AcePlannerPrefillBatch,
  pairedOpt0087: boolean,
): Promise<AcePlannerPreparedPhaseGpuResources> {
  const memory = planAcePlannerGpuPhaseMemory(
    modelProfile,
    prefill,
    residentWeightBytes,
  );
  const layout = createPlannerArenaLayout(modelProfile, prefill, memory.model);
  let arena: AceGpuArena | undefined;
  let readback: GPUBuffer | undefined;
  let cacheAppendReadback: GPUBuffer | undefined;
  let published = false;
  try {
    arena = await AceGpuArena.create(device, layout.plans);
    const opt0087Layout = pairedOpt0087
      ? createOpt0087CacheAppendReadbackLayout(prefill.rows as 1 | 2)
      : undefined;
    const buffers = await createAceScopedBuffers(
      device,
      [
        {
          label: "ace-planner-logit-readback",
          size: memory.readbackBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        },
        ...(opt0087Layout === undefined
          ? []
          : [{
              label: "ace-opt-0087-planner-cache-append-readback",
              size: opt0087Layout.byteLength,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            }]),
      ],
      "ACE planner logit readback allocation",
    );
    readback = buffers[0];
    cacheAppendReadback = buffers[1];
    const bindings = bindPlannerArena(arena, layout, weights);
    const stableArena = arena;
    const stableReadback = readback!;
    const stableCacheAppendReadback = cacheAppendReadback;
    let destroyed = false;
    const resources: AcePlannerPreparedPhaseGpuResources = Object.freeze({
      batch: prefill.rows as 1 | 2,
      prefillTokens: prefill.tokens,
      cacheCapacity: prefill.cacheCapacity,
      bindings,
      readback: stableReadback,
      ...(opt0087Layout === undefined || stableCacheAppendReadback === undefined
        ? {}
        : {
            opt0087CacheAppendReadback: Object.freeze({
              buffer: stableCacheAppendReadback,
              layout: opt0087Layout,
            }),
          }),
      memory,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        stableCacheAppendReadback?.destroy();
        stableReadback.destroy();
        stableArena.destroy();
      },
    });
    published = true;
    return resources;
  } finally {
    if (!published) {
      readback?.destroy();
      cacheAppendReadback?.destroy();
      arena?.destroy();
    }
  }
}

interface PlannerArenaLayout {
  readonly plans: readonly AceArenaBufferPlan[];
  readonly bytes: ReadonlyMap<string, number>;
}

function createPlannerArenaLayout(
  modelProfile: AceModelProfileId,
  prefill: AcePlannerPrefillBatch,
  plan: AcePlannerModelPlan,
): PlannerArenaLayout {
  const entries: Array<readonly [string, number]> = [];
  const activation = (label: string, elements: number): void => {
    entries.push([label, aceActivationBytes(modelProfile, elements)]);
  };
  const u32 = (label: string, elements: number): void => {
    entries.push([label, checkedAceProduct([elements, U32_BYTES], label)]);
  };
  const f32 = (label: string, elements: number): void => {
    entries.push([label, checkedAceProduct([elements, F32_BYTES], label)]);
  };

  u32("token-ids", plan.rows);
  u32("valid-lengths", prefill.rows * 2);
  u32("query-positions", plan.rows);
  u32("source-validity", plan.rows);
  u32("row-start-positions", prefill.rows);
  f32(
    "rope-cosine",
    plan.rows * ACE_PLANNER_QWEN3_CONFIG.headDimension,
  );
  f32(
    "rope-sine",
    plan.rows * ACE_PLANNER_QWEN3_CONFIG.headDimension,
  );
  u32("last-physical-row-indices", prefill.rows);
  u32("cache-validity", prefill.rows * prefill.cacheCapacity);
  u32("write-status", prefill.rows);

  activation("embedded", plan.hiddenElements);
  activation("layer-output-0", plan.hiddenElements);
  activation("layer-output-1", plan.hiddenElements);
  activation("normalized-sequence", plan.hiddenElements);
  activation(
    "last-hidden-rows",
    prefill.rows * ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
  );
  for (const label of [
    "normalized-input",
    "projected-attention",
    "after-attention",
    "normalized-after-attention",
    "projected-mlp",
  ]) activation(`scratch-${label}`, plan.hiddenElements);
  for (const label of [
    "query-flat",
    "query-heads",
    "normalized-query-heads",
    "rotated-query-heads",
    "attention-heads",
    "merged-attention",
  ]) activation(`scratch-${label}`, plan.queryElements);
  for (const label of [
    "key-flat",
    "value-flat",
    "key-heads",
    "value-heads",
    "normalized-key-heads",
    "rotated-key-heads",
  ]) activation(`scratch-${label}`, plan.keyValueElements);
  for (const label of ["gate", "up", "gated-activation"]) {
    activation(`scratch-${label}`, plan.intermediateElements);
  }

  const oneCacheBytes = plan.kvCacheBytesPerLayer / 2;
  for (let layer = 0; layer < ACE_PLANNER_QWEN3_CONFIG.layerCount; layer += 1) {
    entries.push([`layer-${layer}-cache-key`, oneCacheBytes]);
    entries.push([`layer-${layer}-cache-value`, oneCacheBytes]);
  }
  for (let index = 0; index < ACE_PLANNER_EMBEDDING_ROW_PARTS.length; index += 1) {
    const rows = ACE_PLANNER_EMBEDDING_ROW_PARTS[index]!.rowCount;
    activation(`logits-${index}`, prefill.rows * rows);
  }

  const bytes = new Map<string, number>();
  const plans = entries.map(([label, byteLength]) => {
    if (bytes.has(label)) throw new Error(`Duplicate ACE planner arena label ${label}`);
    const bindingBytes = align(byteLength, 4);
    bytes.set(label, bindingBytes);
    return Object.freeze({ label, byteLength: align(bindingBytes, STORAGE_ALIGNMENT) });
  });
  return Object.freeze({ plans: Object.freeze(plans), bytes });
}

function bindPlannerArena(
  arena: AceGpuArena,
  layout: PlannerArenaLayout,
  weights: AcePlannerModelWeights,
): AcePlannerModelBindings {
  const indices = new Map(layout.plans.map((plan, index) => [plan.label, index]));
  const binding = (label: string): GPUBufferBinding => {
    const index = indices.get(label);
    const bytes = layout.bytes.get(label);
    if (index === undefined || bytes === undefined) {
      throw new Error(`Missing ACE planner arena allocation ${label}`);
    }
    return arena.binding(arena.slice(label, index, 0, bytes));
  };
  const block: AceQwen3BlockScratch = {
    normalizedInput: binding("scratch-normalized-input"),
    queryFlat: binding("scratch-query-flat"),
    keyFlat: binding("scratch-key-flat"),
    valueFlat: binding("scratch-value-flat"),
    queryHeads: binding("scratch-query-heads"),
    keyHeads: binding("scratch-key-heads"),
    valueHeads: binding("scratch-value-heads"),
    normalizedQueryHeads: binding("scratch-normalized-query-heads"),
    normalizedKeyHeads: binding("scratch-normalized-key-heads"),
    rotatedQueryHeads: binding("scratch-rotated-query-heads"),
    rotatedKeyHeads: binding("scratch-rotated-key-heads"),
    attentionHeads: binding("scratch-attention-heads"),
    mergedAttention: binding("scratch-merged-attention"),
    projectedAttention: binding("scratch-projected-attention"),
    afterAttention: binding("scratch-after-attention"),
    normalizedAfterAttention: binding("scratch-normalized-after-attention"),
    gate: binding("scratch-gate"),
    up: binding("scratch-up"),
    gatedActivation: binding("scratch-gated-activation"),
    projectedMlp: binding("scratch-projected-mlp"),
  };
  return Object.freeze({
    tokenIds: binding("token-ids"),
    weights,
    controls: Object.freeze({
      validLengths: binding("valid-lengths"),
      queryPositions: binding("query-positions"),
      sourceValidity: binding("source-validity"),
      rowStartPositions: binding("row-start-positions"),
      cosine: binding("rope-cosine"),
      sine: binding("rope-sine"),
      lastPhysicalRowIndices: binding("last-physical-row-indices"),
      cacheValidity: binding("cache-validity"),
      writeStatus: binding("write-status"),
    }),
    cache: Object.freeze({
      layers: Object.freeze(Array.from(
        { length: ACE_PLANNER_QWEN3_CONFIG.layerCount },
        (_, layer) => Object.freeze({
          key: binding(`layer-${layer}-cache-key`),
          value: binding(`layer-${layer}-cache-value`),
        }),
      )),
    }),
    scratch: Object.freeze({
      embedded: binding("embedded"),
      block,
      layerOutputs: Object.freeze([
        binding("layer-output-0"),
        binding("layer-output-1"),
      ]) as readonly [GPUBufferBinding, GPUBufferBinding],
      normalizedSequence: binding("normalized-sequence"),
      lastHiddenRows: binding("last-hidden-rows"),
    }),
    logits: Object.freeze(ACE_PLANNER_EMBEDDING_ROW_PARTS.map(
      (_, index) => binding(`logits-${index}`),
    )),
  });
}

export function createPlannerReadbackLayout(
  modelProfile: AceModelProfileId,
  rows: 1 | 2,
  logitRange?: AcePlannerLogitRange,
): AcePlannerLogitReadbackLayout {
  const firstTokenId = logitRange?.firstTokenId ?? 0;
  const tokenCount = logitRange?.tokenCount ??
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize;
  let cursor = 0;
  const shards = planAcePlannerHeadSlices(logitRange).map((slice) => {
    cursor = align(cursor, STORAGE_ALIGNMENT);
    const byteLength = aceActivationBytes(modelProfile, rows * slice.rowCount);
    const shard = Object.freeze({
      sourceShardIndex: slice.shardIndex,
      firstRow: slice.globalFirstRow,
      destinationFirstRow: slice.globalFirstRow - firstTokenId,
      rowCount: slice.rowCount,
      byteOffset: cursor,
      byteLength,
    });
    cursor += align(byteLength, 4);
    return shard;
  });
  cursor = align(cursor, STORAGE_ALIGNMENT);
  const writeStatusByteOffset = cursor;
  const writeStatusByteLength = rows * U32_BYTES;
  cursor += writeStatusByteLength;
  return Object.freeze({
    rows,
    firstTokenId,
    tokenCount,
    shards: Object.freeze(shards),
    writeStatusByteOffset,
    writeStatusByteLength,
    byteLength: align(cursor, STORAGE_ALIGNMENT),
  });
}

function uploadPlannerInvocation(
  queue: GPUQueue,
  bindings: AcePlannerModelBindings,
  batch: AcePlannerModelBatch,
  controls: ReturnType<typeof createAcePlannerModelControlData>,
): void {
  writeBinding(queue, bindings.tokenIds, batch.inputIds);
  writeBinding(queue, bindings.controls.validLengths, controls.validLengths);
  writeBinding(queue, bindings.controls.queryPositions, controls.queryPositions);
  writeBinding(queue, bindings.controls.sourceValidity, controls.sourceValidity);
  writeBinding(
    queue,
    bindings.controls.rowStartPositions,
    controls.rowStartPositions,
  );
  writeBinding(queue, bindings.controls.cosine, controls.cosine);
  writeBinding(queue, bindings.controls.sine, controls.sine);
  writeBinding(
    queue,
    bindings.controls.lastPhysicalRowIndices,
    controls.lastPhysicalRowIndices,
  );
}

function writeBinding(
  queue: GPUQueue,
  binding: GPUBufferBinding,
  source: Uint32Array | Float32Array,
): void {
  const available = binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  if (source.byteLength > available) {
    throw new RangeError("ACE planner control upload exceeds its phase allocation");
  }
  if (!(source.buffer instanceof ArrayBuffer)) {
    throw new TypeError("ACE planner controls must use detached-safe ArrayBuffer storage");
  }
  queue.writeBuffer(
    binding.buffer,
    binding.offset ?? 0,
    source.buffer,
    source.byteOffset,
    source.byteLength,
  );
}

function encodePlannerQuantum(
  device: GPUDevice,
  phase: AcePlannerPreparedPhaseGpuResources,
  quantum: AcePlannerModelQuantum,
  clearCache: boolean,
): GPUCommandBuffer {
  const encoder = device.createCommandEncoder({ label: `${quantum.id}-command` });
  if (clearCache) {
    clearBinding(encoder, phase.bindings.controls.cacheValidity);
    clearBinding(encoder, phase.bindings.controls.writeStatus);
  }
  const pass = encoder.beginComputePass({ label: quantum.id });
  quantum.encode(pass);
  pass.end();
  return encoder.finish();
}

function encodePlannerReadback(
  device: GPUDevice,
  phase: AcePlannerPreparedPhaseGpuResources,
  layout: AcePlannerLogitReadbackLayout,
  opt0087CacheAppendPosition?: number,
): GPUCommandBuffer {
  const encoder = device.createCommandEncoder({ label: "ace-planner-logit-readback" });
  for (let index = 0; index < layout.shards.length; index += 1) {
    const shard = layout.shards[index]!;
    const source = phase.bindings.logits[shard.sourceShardIndex]!;
    encoder.copyBufferToBuffer(
      source.buffer,
      source.offset ?? 0,
      phase.readback,
      shard.byteOffset,
      align(shard.byteLength, 4),
    );
  }
  const status = phase.bindings.controls.writeStatus;
  encoder.copyBufferToBuffer(
    status.buffer,
    status.offset ?? 0,
    phase.readback,
    layout.writeStatusByteOffset,
    layout.writeStatusByteLength,
  );
  if (opt0087CacheAppendPosition !== undefined) {
    encodeOpt0087CacheAppendCopies(
      encoder,
      phase,
      opt0087CacheAppendPosition,
    );
  }
  return encoder.finish();
}

function encodeOpt0087CacheAppendCopies(
  encoder: GPUCommandEncoder,
  phase: AcePlannerPreparedPhaseGpuResources,
  cachePosition: number,
): void {
  const evidence = phase.opt0087CacheAppendReadback;
  if (evidence === undefined) {
    throw new Error("OPT-0087 cache append readback is unavailable");
  }
  if (
    !Number.isSafeInteger(cachePosition) ||
    cachePosition < 0 ||
    cachePosition >= phase.cacheCapacity
  ) throw new RangeError("OPT-0087 cache append position is invalid");
  const rowBytes = ACE_PLANNER_QWEN3_CONFIG.headDimension * F32_BYTES;
  let destinationByteOffset = 0;
  let copyCount = 0;
  for (const layer of phase.bindings.cache.layers) {
    for (const binding of [layer.key, layer.value]) {
      for (let row = 0; row < phase.batch; row += 1) {
        for (
          let head = 0;
          head < ACE_PLANNER_QWEN3_CONFIG.keyValueHeads;
          head += 1
        ) {
          const sourceWordOffset =
            ((row * ACE_PLANNER_QWEN3_CONFIG.keyValueHeads + head) *
              phase.cacheCapacity + cachePosition) *
                ACE_PLANNER_QWEN3_CONFIG.headDimension;
          encoder.copyBufferToBuffer(
            binding.buffer,
            (binding.offset ?? 0) + sourceWordOffset * F32_BYTES,
            evidence.buffer,
            destinationByteOffset,
            rowBytes,
          );
          destinationByteOffset += rowBytes;
          copyCount += 1;
        }
      }
    }
  }
  if (
    destinationByteOffset !==
      evidence.layout.keyValueWordCount * U32_BYTES
  ) throw new Error("OPT-0087 K/V cache copy layout did not reconcile");
  const validity = phase.bindings.controls.cacheValidity;
  for (let row = 0; row < phase.batch; row += 1) {
    encoder.copyBufferToBuffer(
      validity.buffer,
      (validity.offset ?? 0) +
        (row * phase.cacheCapacity + cachePosition) * U32_BYTES,
      evidence.buffer,
      destinationByteOffset,
      U32_BYTES,
    );
    destinationByteOffset += U32_BYTES;
    copyCount += 1;
  }
  if (
    destinationByteOffset !== evidence.layout.logicalByteLength ||
    copyCount !== evidence.layout.copyCount
  ) throw new Error("OPT-0087 cache append copy layout did not reconcile");
}

interface MappedPlannerLogits {
  readonly logits: readonly Float32Array[];
  readonly writeStatusWords: readonly number[];
}

async function mapPlannerLogits(
  readback: GPUBuffer,
  layout: AcePlannerLogitReadbackLayout,
  modelProfile: AceModelProfileId,
  signal: AbortSignal,
): Promise<readonly Float32Array[]> {
  const detached = await mapPlannerReadback(readback, layout, signal);
  return reconstructAcePlannerLogits(detached, layout, modelProfile);
}

/** @internal Status capture exists only for the OPT-0087 evidence path. */
async function mapPlannerLogitsForOpt0087(
  readback: GPUBuffer,
  layout: AcePlannerLogitReadbackLayout,
  modelProfile: AceModelProfileId,
  signal: AbortSignal,
): Promise<MappedPlannerLogits> {
  const detached = await mapPlannerReadback(readback, layout, signal);
  const writeStatusWords = Object.freeze([...new Uint32Array(
    detached,
    layout.writeStatusByteOffset,
    layout.rows,
  )]);
  return Object.freeze({
    logits: reconstructAcePlannerLogits(detached, layout, modelProfile),
    writeStatusWords,
  });
}

async function mapPlannerReadback(
  readback: GPUBuffer,
  layout: AcePlannerLogitReadbackLayout,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  return await mapPlannerBuffer(readback, layout.byteLength, signal);
}

/** @internal Maps only the paired OPT-0087 append-evidence allocation. */
async function mapPlannerCacheAppendForOpt0087(
  readback: AcePlannerOpt0087CacheAppendReadback,
  signal: AbortSignal,
): Promise<Uint32Array> {
  const detached = await mapPlannerBuffer(
    readback.buffer,
    readback.layout.byteLength,
    signal,
  );
  return new Uint32Array(
    detached,
    0,
    readback.layout.logicalByteLength / U32_BYTES,
  );
}

async function mapPlannerBuffer(
  readback: GPUBuffer,
  byteLength: number,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  signal.throwIfAborted();
  await readback.mapAsync(GPUMapMode.READ, 0, byteLength);
  let detached: ArrayBuffer;
  try {
    detached = readback.getMappedRange(0, byteLength).slice(0);
  } finally {
    readback.unmap();
  }
  signal.throwIfAborted();
  return detached;
}

function validateDispatchOrder(
  dispatch: AcePlannerModelDispatch,
  batch: AcePlannerModelBatch,
  logitRange?: AcePlannerLogitRange,
): void {
  if (
    dispatch.plan.kind !== batch.kind ||
    dispatch.plan.batch !== batch.rows ||
    dispatch.plan.tokens !== batch.tokens ||
    dispatch.plan.cacheCapacity !== batch.cacheCapacity
  ) {
    throw new Error("ACE planner runtime returned a dispatch for different geometry");
  }
  if (
    (logitRange === undefined && dispatch.logitRange !== null) ||
    (logitRange !== undefined &&
      (dispatch.logitRange === null ||
        dispatch.logitRange.firstTokenId !== logitRange.firstTokenId ||
        dispatch.logitRange.tokenCount !== logitRange.tokenCount))
  ) {
    throw new Error("ACE planner runtime returned a different logit range");
  }
  if (
    dispatch.quanta[0]?.kind !== "embedding" ||
    dispatch.quanta.at(-1)?.kind !== "tied-lm-head"
  ) {
    throw new Error("ACE planner cooperative quantum order changed");
  }
  let cursor = 1;
  for (let layer = 0; layer < ACE_PLANNER_QWEN3_CONFIG.layerCount; layer += 1) {
    let parts = 0;
    while (
      dispatch.quanta[cursor]?.kind === "layer" &&
      dispatch.quanta[cursor]?.layer === layer
    ) {
      cursor += 1;
      parts += 1;
    }
    if (parts === 0) {
      throw new Error(`ACE planner layer quantum ${layer} changed order`);
    }
  }
  if (dispatch.quanta[cursor]?.kind !== "final-norm") {
    throw new Error("ACE planner final norm quantum changed order");
  }
  cursor += 1;
  if (dispatch.quanta[cursor]?.kind !== "last-row-gather") {
    throw new Error("ACE planner last-row quantum changed order");
  }
  cursor += 1;
  if (
    cursor >= dispatch.quanta.length ||
    dispatch.quanta.slice(cursor).some((quantum) => quantum.kind !== "tied-lm-head")
  ) {
    throw new Error("ACE planner tied-head quantum order changed");
  }
}

function validatePreparedExecutorResources(
  resources: AcePlannerPreparedGpuExecutorResources,
): void {
  if (
    resources.modelProfile !== "reference-bf16" &&
    resources.modelProfile !== "raw-fp16"
  ) {
    throw new TypeError("ACE prepared planner executor has an unknown profile");
  }
  requireNonNegativeSafeInteger(
    resources.residentWeightBytes,
    "ACE prepared planner resident bytes",
  );
  if (
    typeof resources.runtime?.createPlannerDispatch !== "function" ||
    typeof resources.runtime.destroy !== "function" ||
    typeof resources.createPhase !== "function" ||
    typeof resources.destroy !== "function"
  ) {
    throw new TypeError("ACE prepared planner executor resources are incomplete");
  }
  if (
    resources.onOpt0085Scheduling !== undefined &&
    typeof resources.onOpt0085Scheduling !== "function"
  ) {
    throw new TypeError("ACE planner OPT-0085 diagnostics hook is invalid");
  }
  if (
    resources.onOpt0085Scheduling !== undefined &&
    resources.modelProfile !== "reference-bf16"
  ) {
    throw new Error("ACE planner OPT-0085 requires reference-BF16");
  }
  if (
    resources.opt0087PairedOwner !== undefined &&
    resources.opt0087PairedOwner !== true
  ) {
    throw new TypeError("ACE planner OPT-0087 paired-owner marker is invalid");
  }
  if (
    (resources.opt0087PairedOwner !== true &&
      resources.onOpt0087Invocation !== undefined) ||
    (resources.opt0087PairedOwner === true &&
      (resources.onOpt0087Invocation === undefined ||
        typeof resources.runtime.createPlannerDispatchForOpt0087 !==
          "function"))
  ) {
    throw new TypeError("ACE planner OPT-0087 paired resources are incomplete");
  }
  if (
    resources.opt0087PairedOwner === true &&
    resources.modelProfile !== "reference-bf16"
  ) {
    throw new Error("ACE planner OPT-0087 requires reference-BF16");
  }
}

function validatePreparedPhaseResources(
  resources: AcePlannerPreparedPhaseGpuResources,
  batch: AcePlannerPrefillBatch,
  modelProfile: AceModelProfileId,
  expectOpt0087CacheEvidence: boolean,
): void {
  if (
    resources.batch !== batch.rows ||
    resources.prefillTokens !== batch.tokens ||
    resources.cacheCapacity !== batch.cacheCapacity ||
    resources.memory.model.kind !== "prefill" ||
    resources.memory.model.batch !== batch.rows ||
    resources.memory.model.tokens !== batch.tokens ||
    resources.memory.allocatedCacheCapacity !== batch.cacheCapacity ||
    resources.memory.readbackLayout.rows !== batch.rows
  ) {
    throw new Error("ACE prepared planner phase does not match its prefill");
  }
  if (resources.readback.size < resources.memory.readbackBytes) {
    throw new RangeError("ACE planner readback buffer is smaller than its layout");
  }
  const cacheEvidence = resources.opt0087CacheAppendReadback;
  if ((cacheEvidence !== undefined) !== expectOpt0087CacheEvidence) {
    throw new Error("ACE planner OPT-0087 cache evidence ownership changed");
  }
  if (cacheEvidence !== undefined) {
    const expected = createOpt0087CacheAppendReadbackLayout(
      batch.rows as 1 | 2,
    );
    if (
      cacheEvidence.buffer.size < expected.byteLength ||
      cacheEvidence.layout.rows !== expected.rows ||
      cacheEvidence.layout.keyValueWordCount !== expected.keyValueWordCount ||
      cacheEvidence.layout.validityWordCount !== expected.validityWordCount ||
      cacheEvidence.layout.logicalByteLength !== expected.logicalByteLength ||
      cacheEvidence.layout.byteLength !== expected.byteLength ||
      cacheEvidence.layout.copyCount !== expected.copyCount
    ) throw new Error("ACE planner OPT-0087 cache evidence layout changed");
  }
  if (typeof resources.destroy !== "function") {
    throw new TypeError("ACE prepared planner phase requires explicit cleanup");
  }
  if (resources.bindings.cache.layers.length !== ACE_PLANNER_QWEN3_CONFIG.layerCount) {
    throw new Error("ACE prepared planner phase is missing disjoint layer caches");
  }
  if (resources.memory.model.totalActivationBytes !==
    planAcePlannerModel(modelProfile, batch).totalActivationBytes) {
    throw new Error("ACE prepared planner phase activation profile changed");
  }
}

function accountedPlannerPhaseGpuBytes(
  resources: AcePlannerPreparedPhaseGpuResources,
): number {
  return checkedSum([
    resources.memory.accountedGpuBytes,
    resources.opt0087CacheAppendReadback?.layout.byteLength ?? 0,
  ], "ACE planner phase accounted GPU bytes");
}

function clearBinding(
  encoder: GPUCommandEncoder,
  binding: GPUBufferBinding,
): void {
  encoder.clearBuffer(
    binding.buffer,
    binding.offset ?? 0,
    binding.size ?? binding.buffer.size - (binding.offset ?? 0),
  );
}

function copyLogitShard(
  output: readonly Float32Array[],
  source: Float32Array,
  shard: AcePlannerLogitReadbackShard,
): void {
  for (let row = 0; row < output.length; row += 1) {
    output[row]!.set(
      source.subarray(row * shard.rowCount, (row + 1) * shard.rowCount),
      shard.destinationFirstRow,
    );
  }
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) return u32BitsToF32(sign);
    exponent = 1;
    while ((mantissa & 0x400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x3ff;
  } else if (exponent === 0x1f) {
    return u32BitsToF32(sign | 0x7f80_0000 | (mantissa << 13));
  }
  return u32BitsToF32(
    sign | ((exponent - 15 + 127) << 23) | (mantissa << 13),
  );
}

const FLOAT_BITS_BUFFER = new ArrayBuffer(4);
const FLOAT_BITS_F32 = new Float32Array(FLOAT_BITS_BUFFER);
const FLOAT_BITS_U32 = new Uint32Array(FLOAT_BITS_BUFFER);

function u32BitsToF32(bits: number): number {
  FLOAT_BITS_U32[0] = bits;
  return FLOAT_BITS_F32[0]!;
}

function align(value: number, alignment: number): number {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    !Number.isSafeInteger(alignment) ||
    alignment <= 0
  ) {
    throw new RangeError("Invalid ACE planner alignment");
  }
  const aligned = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(aligned)) {
    throw new RangeError("ACE planner aligned byte count is not safe");
  }
  return aligned;
}

function checkedSum(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${label} contains an invalid value`);
    }
    total += value;
    if (!Number.isSafeInteger(total)) {
      throw new RangeError(`${label} is not a safe integer`);
    }
  }
  return total;
}

function requireNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
}

function requireModelProfile(modelProfile: AceModelProfileId): void {
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE planner model profile ${String(modelProfile)}`);
  }
}

function yieldQueueIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

function destroyedError(): DOMException {
  return new DOMException("ACE planner GPU executor was destroyed", "AbortError");
}
