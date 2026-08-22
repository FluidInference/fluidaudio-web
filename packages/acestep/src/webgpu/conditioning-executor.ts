import type { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import type { AcePackageManifest } from "../model/manifest.js";
import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  AceFifoGraphOwner,
} from "../runtime/scheduler.js";
import {
  createAceEncoderControlData,
  createAceEncoderFullControlData,
  createAceEncoderRopeTables,
  planAceEncoderBlock,
  type AceEncoderBlockScratch,
} from "./ace-encoder.js";
import { AceGpuArena, type AceArenaBufferPlan } from "./arena.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  aceActivationBytes,
  checkedAceProduct,
} from "./kernels/correctness-utils.js";
import { ACE_FSQ_CODEBOOK_SIZE } from "./kernels/fsq-decode.js";
import {
  ACE_AUDIO_LATENT_CHANNELS,
  ACE_CONDITION_ENCODER_CONFIG,
  ACE_CONDITION_HIDDEN_SIZE,
  ACE_NO_REFERENCE_TIMBRE_FRAMES,
  ACE_SEMANTIC_POOL_WIDTH,
  AceCorrectnessSemanticConditionerRuntime,
  createAceDirectV1ChunkMask,
  createAceNoReferenceTimbreControls,
  planAceConditioner,
  planAceSemanticDecode,
  type AceConditionerDispatchSource,
  type AceDirectConditionerBindings,
  type AceDirectConditionerPlan,
  type AceSemanticDecodeBindings,
  type AceSemanticDecodePlan,
} from "./semantic-conditioner.js";
import {
  resolveAceDirectConditionerPackage,
  resolveAceSemanticPackageWeights,
  validateAceConditionerPackageInventory,
} from "./semantic-conditioner-package.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";
import {
  ACE_TEXT_QWEN3_CONFIG,
  type AceQwen3BlockScratch,
} from "./qwen3.js";
import {
  AceCorrectnessTextEncoderRuntime,
  createAceTextEncoderControlData,
  planAceTextEncoder,
  resolveAceTextEncoderWeights,
  validateAceTextEncoderManifestInventory,
  type AceQwen3ModelPlan,
  type AceTextEncoderBindings,
} from "./text-encoder.js";

const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const F32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const STORAGE_ALIGNMENT = 256;

type Destroyable = Readonly<{ destroy(): void }>;

export type AceConditioningMode =
  | Readonly<{ readonly kind: "direct" }>
  | Readonly<{
      readonly kind: "planner";
      /** Parsed audio codes in `[0,64000)`, not vocabulary token IDs. */
      readonly semanticCodeIds: Uint32Array;
    }>;

export interface AceConditioningGpuRequest {
  readonly textTokenIds: Uint32Array;
  readonly lyricTokenIds: Uint32Array;
  readonly textMask: Uint32Array;
  readonly lyricMask: Uint32Array;
  readonly latentFrames: number;
  readonly mode: AceConditioningMode;
}

export interface AceConditioningGpuRequestSnapshot {
  readonly textTokenIds: Uint32Array<ArrayBuffer>;
  readonly lyricTokenIds: Uint32Array<ArrayBuffer>;
  readonly textMask: Uint32Array<ArrayBuffer>;
  readonly lyricMask: Uint32Array<ArrayBuffer>;
  readonly latentFrames: number;
  readonly mode: AceConditioningMode;
}

export interface AceConditioningGpuProgress {
  readonly phase: "text" | "semantic" | "conditioner";
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly quantumId: string;
}

export interface AceConditioningGpuMemoryAccounting {
  readonly textWeightBytes: number;
  readonly textRetainedBytes: number;
  readonly textWorkingBytes: number;
  readonly semanticWeightBytes: number;
  readonly semanticRetainedBytes: number;
  readonly semanticWorkingBytes: number;
  readonly conditionerWeightBytes: number;
  readonly conditionerWorkingBytes: number;
  readonly resultBytes: number;
  readonly resultReadbackBytes: number;
  readonly returnedCpuBytes: number;
  readonly peakAccountedGpuBytes: number;
}

export interface AceConditioningGpuResult {
  readonly mode: AceConditioningMode["kind"];
  readonly batch: 1;
  readonly conditionTokens: number;
  readonly latentFrames: number;
  readonly hiddenSize: typeof ACE_CONDITION_HIDDEN_SIZE;
  readonly contextChannels: 128;
  /** Detached logical FP32 `[1,conditionTokens,2048]` for DiT input upload. */
  readonly conditionHiddenStates: Float32Array<ArrayBuffer>;
  /** Detached diagnostic U32 `[1,conditionTokens]`; current DiT ignores it. */
  readonly conditionMask: Uint32Array<ArrayBuffer>;
  /** Detached logical FP32 `[1,latentFrames,128]` for DiT input upload. */
  readonly contextLatents: Float32Array<ArrayBuffer>;
  readonly memory: AceConditioningGpuMemoryAccounting;
}

export interface AceConditioningGpuExecutorOptions {
  readonly device: GPUDevice;
  readonly manifest: AcePackageManifest;
  readonly modelProfile: AceModelProfileId;
  /** Ownership transfers at call entry, including on factory failure. */
  readonly ownedTextWeights: AceGpuTensorPhase;
  /** Ownership transfers when the promise resolves. Required by planner mode. */
  readonly loadSemanticWeights?: (
    signal: AbortSignal,
  ) => Promise<AceGpuTensorPhase>;
  /** Must load exactly the `conditioner+constants` lifetime. */
  readonly loadConditionerWeights: (
    signal: AbortSignal,
  ) => Promise<AceGpuTensorPhase>;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceConditioningGpuProgress) => void;
  /** @internal Deterministic queue-empty seam for lifecycle tests. */
  readonly yieldQueueIdle?: () => Promise<void>;
}

export class AceConditioningGpuDeviceLostError extends Error {
  override readonly name = "AceConditioningGpuDeviceLostError";

  constructor(info: Pick<GPUDeviceLostInfo, "reason" | "message">) {
    super(
      `ACE conditioning WebGPU device lost (${info.reason}): ` +
        (info.message || "no device message"),
    );
  }
}

/**
 * Single-use production owner for text, optional semantic decode, and the
 * shared direct/planner conditioner. A FIFO lease covers all phase changes.
 * Every quantum is submitted alone, fully drained, then followed by a real
 * queue-empty 1 ms interval when another command buffer remains.
 */
export class AceConditioningGpuExecutor {
  private readonly graphOwner = new AceFifoGraphOwner();
  private readonly scheduler = new AceCooperativeGpuScheduler();
  private readonly lifetime = new AbortController();
  private readonly owned = new Set<Destroyable>();
  private readonly externalAbortListener: (() => void) | undefined;
  private state: "live" | "running" | "completed" | "destroying" | "destroyed" =
    "live";
  private destroyPromise: Promise<void> | undefined;

  private constructor(
    private readonly options: AceConditioningGpuExecutorOptions,
  ) {
    this.owned.add(options.ownedTextWeights);
    this.externalAbortListener = options.signal === undefined
      ? undefined
      : () => {
          if (!this.lifetime.signal.aborted) {
            this.lifetime.abort(options.signal!.reason);
          }
          void this.beginDestroy().catch(() => undefined);
        };
    options.signal?.addEventListener("abort", this.externalAbortListener!, {
      once: true,
    });
    const weak = new WeakRef(this);
    void options.device.lost.then((info) => {
      const owner = weak.deref();
      if (owner === undefined || owner.state === "completed" || owner.state === "destroyed") {
        return;
      }
      const error = new AceConditioningGpuDeviceLostError(info);
      if (!owner.lifetime.signal.aborted) owner.lifetime.abort(error);
      void owner.beginDestroy().catch(() => undefined);
    });
  }

  static create(options: AceConditioningGpuExecutorOptions): AceConditioningGpuExecutor {
    let published = false;
    try {
      options.signal?.throwIfAborted();
      requireManifestProfile(options.manifest, options.modelProfile);
      validateAceTextEncoderManifestInventory(options.manifest);
      validateAceConditionerPackageInventory(options.manifest);
      validateAceConditioningPhaseSet(options.ownedTextWeights, ["text"], "text");
      validateAceConditioningPhaseManifest(
        options.ownedTextWeights,
        options.manifest,
        "text",
      );
      // Resolve now so a wrong phase/package never reaches graph construction.
      resolveAceTextEncoderWeights(options.ownedTextWeights, options.modelProfile);
      const executor = new AceConditioningGpuExecutor(options);
      published = true;
      return executor;
    } finally {
      if (!published) options.ownedTextWeights.destroy();
    }
  }

  /** @internal Deterministic lifetime seam; production uses `create`. */
  static fromPreparedOptionsForTest(
    options: AceConditioningGpuExecutorOptions,
  ): AceConditioningGpuExecutor {
    try {
      options.signal?.throwIfAborted();
      return new AceConditioningGpuExecutor(options);
    } catch (error) {
      options.ownedTextWeights.destroy();
      throw error;
    }
  }

  async run(request: AceConditioningGpuRequest): Promise<AceConditioningGpuResult> {
    this.requireLive();
    validateRequest(request);
    const snapshot = snapshotAceConditioningGpuRequest(request);
    this.state = "running";
    const lease = await this.graphOwner.acquire(this.lifetime.signal);
    let finalArena: NamedArena | undefined;
    try {
      const text = await this.runTextPhase(snapshot);
      const semantic = snapshot.mode.kind === "planner"
        ? await this.runSemanticPhase(snapshot.mode.semanticCodeIds)
        : undefined;
      const conditioned = await this.runConditionerPhase(snapshot, text, semantic);
      finalArena = conditioned.finalArena;
      this.releaseOwned(text.retained);
      if (semantic !== undefined) this.releaseOwned(semantic.retained);
      const detached = await this.readbackResult(conditioned.plan, finalArena);
      this.releaseOwned(finalArena);
      finalArena = undefined;
      const result: AceConditioningGpuResult = Object.freeze({
        mode: snapshot.mode.kind,
        batch: 1,
        conditionTokens: conditioned.plan.conditionTokens,
        latentFrames: snapshot.latentFrames,
        hiddenSize: ACE_CONDITION_HIDDEN_SIZE,
        contextChannels: 128,
        conditionHiddenStates: detached.condition,
        conditionMask: detached.mask,
        contextLatents: detached.context,
        memory: planAceConditioningGpuMemory({
          modelProfile: this.options.modelProfile,
          textTokens: snapshot.textTokenIds.length,
          lyricTokens: snapshot.lyricTokenIds.length,
          latentFrames: snapshot.latentFrames,
          semanticCodeTokens: snapshot.mode.kind === "planner"
            ? snapshot.mode.semanticCodeIds.length
            : 0,
          textWeightBytes: text.weightBytes,
          semanticWeightBytes: semantic?.weightBytes ?? 0,
          conditionerWeightBytes: conditioned.weightBytes,
        }),
      });
      this.state = "completed";
      this.detachAbortListener();
      return result;
    } catch (error) {
      this.destroyOwned();
      if (finalArena !== undefined && this.owned.has(finalArena)) finalArena.destroy();
      this.state = "destroyed";
      this.detachAbortListener();
      throw error;
    } finally {
      lease.release();
      if (this.state === "completed" || this.state === "destroyed") {
        await this.scheduler.dispose();
        await this.graphOwner.dispose();
      }
    }
  }

  private async readbackResult(
    plan: AceDirectConditionerPlan,
    finalArena: NamedArena,
  ): Promise<Readonly<{
    condition: Float32Array<ArrayBuffer>;
    mask: Uint32Array<ArrayBuffer>;
    context: Float32Array<ArrayBuffer>;
    readbackBytes: number;
  }>> {
    const layout = createResultReadbackLayout(this.options.modelProfile, plan);
    const [readback] = await createAceScopedBuffers(
      this.options.device,
      [{
        label: "ace-conditioner-result-readback",
        size: layout.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }],
      "ACE conditioner result readback allocation",
    );
    const owner = this.own(bufferOwner(readback!));
    try {
      await this.runQuanta("conditioner", [{
        id: "ace-conditioner-result-readback",
        encodeCommand: (encoder: GPUCommandEncoder): void => {
          copyBindingToBufferAt(
            encoder,
            finalArena.binding("condition-hidden-states"),
            readback!,
            layout.conditionOffset,
            layout.conditionStoredBytes,
          );
          copyBindingToBufferAt(
            encoder,
            finalArena.binding("condition-mask"),
            readback!,
            layout.maskOffset,
            layout.maskBytes,
          );
          copyBindingToBufferAt(
            encoder,
            finalArena.binding("context-latents"),
            readback!,
            layout.contextOffset,
            layout.contextStoredBytes,
          );
        },
      }], true);
      this.lifetime.signal.throwIfAborted();
      await readback!.mapAsync(GPUMapMode.READ, 0, layout.byteLength);
      try {
        this.lifetime.signal.throwIfAborted();
        const mapped = readback!.getMappedRange(0, layout.byteLength);
        return Object.freeze({
          condition: decodeAceConditioningActivation(
            this.options.modelProfile,
            mapped,
            layout.conditionOffset,
            plan.conditionElements,
          ),
          mask: copyU32(mapped, layout.maskOffset, plan.batch * plan.conditionTokens),
          context: decodeAceConditioningActivation(
            this.options.modelProfile,
            mapped,
            layout.contextOffset,
            plan.contextElements,
          ),
          readbackBytes: layout.byteLength,
        });
      } finally {
        readback!.unmap();
      }
    } finally {
      this.releaseOwned(owner);
    }
  }

  destroy(reason: unknown = new DOMException(
    "ACE conditioning executor was destroyed",
    "AbortError",
  )): Promise<void> {
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    return this.beginDestroy();
  }

  private async runTextPhase(
    request: AceConditioningGpuRequestSnapshot,
  ): Promise<TextPhaseOutput> {
    const phase = this.options.ownedTextWeights;
    const weights = resolveAceTextEncoderWeights(phase, this.options.modelProfile);
    const textPlan = planAceTextEncoder(this.options.modelProfile, {
      batch: 1,
      tokens: request.textTokenIds.length,
    });
    const retained = await this.ownAsync(createNamedArena(
      this.options.device,
      "ace-text-retained",
      [
        activationEntry(this.options.modelProfile, "text-hidden", textPlan.outputElements),
        activationEntry(
          this.options.modelProfile,
          "lyric-hidden",
          request.lyricTokenIds.length * ACE_TEXT_QWEN3_CONFIG.hiddenSize,
        ),
        u32Entry("text-mask", request.textMask.length),
        u32Entry("lyric-mask", request.lyricMask.length),
      ],
    ));
    const work = await this.ownAsync(createTextWorkArena(
      this.options.device,
      this.options.modelProfile,
      textPlan,
      request.lyricTokenIds.length,
    ));
    const runtime = this.own(AceCorrectnessTextEncoderRuntime.create(
      this.options.device,
      this.options.modelProfile,
    ));
    const controls = createAceTextEncoderControlData(request.textTokenIds.length);
    writeBinding(this.options.device.queue, work.binding("text-token-ids"), request.textTokenIds);
    writeBinding(this.options.device.queue, work.binding("lyric-token-ids"), request.lyricTokenIds);
    writeBinding(this.options.device.queue, work.binding("valid-lengths"), controls.validLengths);
    writeBinding(this.options.device.queue, work.binding("query-positions"), controls.queryPositions);
    writeBinding(this.options.device.queue, work.binding("key-validity"), controls.keyValidity);
    writeBinding(this.options.device.queue, work.binding("cosine"), controls.cosine);
    writeBinding(this.options.device.queue, work.binding("sine"), controls.sine);
    writeBinding(this.options.device.queue, retained.binding("text-mask"), request.textMask);
    writeBinding(this.options.device.queue, retained.binding("lyric-mask"), request.lyricMask);

    const bindings: AceTextEncoderBindings = {
      tokenIds: work.binding("text-token-ids"),
      output: retained.binding("text-hidden"),
      weights,
      controls: {
        validLengths: work.binding("valid-lengths"),
        queryPositions: work.binding("query-positions"),
        keyValidity: work.binding("key-validity"),
        cosine: work.binding("cosine"),
        sine: work.binding("sine"),
      },
      scratch: {
        embedded: work.binding("embedded"),
        block: qwenScratch(work),
        layerOutputs: [
          work.binding("layer-output-0"),
          work.binding("layer-output-1"),
        ],
      },
    };
    const [textDispatch, lyricDispatch] = await Promise.all([
      runtime.createTextEncoderDispatch("ace-production-text", {
        batch: 1,
        tokens: request.textTokenIds.length,
      }, bindings),
      runtime.createLyricEmbeddingDispatch(
        "ace-production-lyric-embedding",
        request.lyricTokenIds.length,
        {
          tokenIds: work.binding("lyric-token-ids"),
          embedding: weights.embedding,
          output: retained.binding("lyric-hidden"),
        },
      ),
    ]);
    await this.runQuanta("text", [
      ...textDispatch.quanta,
      {
        id: lyricDispatch.label,
        encode: (pass: GPUComputePassEncoder) => lyricDispatch.encode(pass),
      },
    ], false);
    const weightBytes = phase.residentBytes;
    const workingBytes = work.byteLength;
    this.releaseOwned(runtime);
    this.releaseOwned(work);
    this.releaseOwned(phase);
    return Object.freeze({ retained, weightBytes, workingBytes });
  }

  private async runSemanticPhase(codeIds: Uint32Array): Promise<SemanticPhaseOutput> {
    const loader = this.options.loadSemanticWeights;
    if (loader === undefined) {
      throw new Error("ACE planner conditioning requires a semantic phase loader");
    }
    this.lifetime.signal.throwIfAborted();
    const phase = this.own(await loader(this.lifetime.signal));
    validateAceConditioningPhaseSet(phase, ["semantic"], "semantic");
    validateAceConditioningPhaseManifest(phase, this.options.manifest, "semantic");
    const weights = resolveAceSemanticPackageWeights(
      phase,
      this.options.modelProfile,
    );
    const plan = planAceSemanticDecode({ batch: 1, codeTokens: codeIds.length });
    const retained = await this.ownAsync(createNamedArena(
      this.options.device,
      "ace-semantic-retained",
      [activationEntry(this.options.modelProfile, "semantic-hints", plan.outputElements)],
    ));
    const work = await this.ownAsync(createSemanticWorkArena(
      this.options.device,
      this.options.modelProfile,
      plan,
    ));
    const [statusReadback] = await createAceScopedBuffers(
      this.options.device,
      [{
        label: "ace-semantic-validation-readback",
        size: STORAGE_ALIGNMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }],
      "ACE semantic validation readback allocation",
    );
    const readbackOwner = this.own(bufferOwner(statusReadback!));
    const runtime = this.own(AceCorrectnessSemanticConditionerRuntime.create(
      this.options.device,
      this.options.modelProfile,
    ));
    const fullControls = createAceEncoderFullControlData(
      plan.codeCount,
      ACE_SEMANTIC_POOL_WIDTH,
    );
    const rope = createAceEncoderRopeTables(
      ACE_SEMANTIC_POOL_WIDTH,
      ACE_CONDITION_ENCODER_CONFIG,
    );
    writeBinding(this.options.device.queue, work.binding("code-ids"), codeIds);
    writeBinding(this.options.device.queue, work.binding("valid-lengths"), fullControls.validLengths);
    writeBinding(this.options.device.queue, work.binding("cosine"), rope.cosine);
    writeBinding(this.options.device.queue, work.binding("sine"), rope.sine);
    const dispatch = await runtime.createSemanticDecodeDispatch(
      "ace-production-semantic",
      { batch: 1, codeTokens: codeIds.length },
      semanticBindings(work, retained, weights),
    );
    await this.runQuanta("semantic", [
      ...dispatch.quanta.map((quantum, index): QuantumEncoder => Object.freeze({
        id: quantum.id,
        ...(index === 0
          ? {
              beforePass: (encoder: GPUCommandEncoder): void => clearBinding(
                encoder,
                work.binding("validation-status"),
              ),
            }
          : {}),
        encode: (pass: GPUComputePassEncoder): void => quantum.encode(pass),
      })),
      {
        id: "ace-semantic-validation-readback",
        encodeCommand: (encoder: GPUCommandEncoder): void => {
          copyBindingToBuffer(
            encoder,
            work.binding("validation-status"),
            statusReadback!,
            U32_BYTES,
          );
        },
      },
    ], false);
    this.lifetime.signal.throwIfAborted();
    await statusReadback!.mapAsync(GPUMapMode.READ, 0, U32_BYTES);
    try {
      const status = new Uint32Array(statusReadback!.getMappedRange(0, U32_BYTES))[0];
      if (status !== 0) throw new Error("ACE semantic FSQ rejected an invalid code ID");
    } finally {
      statusReadback!.unmap();
    }
    const weightBytes = phase.residentBytes;
    const workingBytes = work.byteLength + STORAGE_ALIGNMENT;
    this.releaseOwned(readbackOwner);
    this.releaseOwned(runtime);
    this.releaseOwned(work);
    this.releaseOwned(phase);
    return Object.freeze({
      retained,
      plan,
      weightBytes,
      workingBytes,
    });
  }

  private async runConditionerPhase(
    request: AceConditioningGpuRequestSnapshot,
    text: TextPhaseOutput,
    semantic: SemanticPhaseOutput | undefined,
  ): Promise<ConditionerPhaseOutput> {
    this.lifetime.signal.throwIfAborted();
    const phase = this.own(await this.options.loadConditionerWeights(
      this.lifetime.signal,
    ));
    validateAceConditioningPhaseSet(
      phase,
      ["conditioner", "constants"],
      "conditioner",
    );
    validateAceConditioningPhaseManifest(
      phase,
      this.options.manifest,
      "conditioner",
    );
    const resolved = resolveAceDirectConditionerPackage(
      phase,
      this.options.modelProfile,
    );
    const sourcePlan = semantic === undefined
      ? { kind: "direct-silence" } as const
      : {
          kind: "planner-semantic-cover" as const,
          semanticCodeCount: semantic.plan.codeCount,
          semanticFrames: semantic.plan.outputFrames,
        };
    const plan = planAceConditioner({
      batch: 1,
      textTokens: request.textTokenIds.length,
      lyricTokens: request.lyricTokenIds.length,
      latentFrames: request.latentFrames,
    }, sourcePlan);
    const finalArena = await this.ownAsync(createNamedArena(
      this.options.device,
      "ace-conditioner-result",
      [
        activationEntry(
          this.options.modelProfile,
          "condition-hidden-states",
          plan.conditionElements,
        ),
        u32Entry("condition-mask", plan.conditionTokens),
        activationEntry(
          this.options.modelProfile,
          "context-latents",
          plan.contextElements,
        ),
      ],
    ));
    const work = await this.ownAsync(createConditionerWorkArena(
      this.options.device,
      this.options.modelProfile,
      plan,
    ));
    const runtime = this.own(AceCorrectnessSemanticConditionerRuntime.create(
      this.options.device,
      this.options.modelProfile,
    ));
    const lyricControls = createAceEncoderControlData(
      Array.from(request.lyricMask),
      1,
      request.lyricTokenIds.length,
    );
    const lyricRope = createAceEncoderRopeTables(
      request.lyricTokenIds.length,
      ACE_CONDITION_ENCODER_CONFIG,
    );
    const timbreControls = createAceEncoderFullControlData(
      1,
      ACE_NO_REFERENCE_TIMBRE_FRAMES,
    );
    const timbreRope = createAceEncoderRopeTables(
      ACE_NO_REFERENCE_TIMBRE_FRAMES,
      ACE_CONDITION_ENCODER_CONFIG,
    );
    const noReference = createAceNoReferenceTimbreControls(1);
    writeBinding(this.options.device.queue, work.binding("lyric-valid-lengths"), lyricControls.validLengths);
    writeBinding(this.options.device.queue, work.binding("lyric-cosine"), lyricRope.cosine);
    writeBinding(this.options.device.queue, work.binding("lyric-sine"), lyricRope.sine);
    writeBinding(this.options.device.queue, work.binding("timbre-valid-lengths"), timbreControls.validLengths);
    writeBinding(this.options.device.queue, work.binding("timbre-cosine"), timbreRope.cosine);
    writeBinding(this.options.device.queue, work.binding("timbre-sine"), timbreRope.sine);
    writeBinding(this.options.device.queue, work.binding("timbre-first-row"), noReference.firstRowIndices);
    writeBinding(this.options.device.queue, work.binding("timbre-mask"), noReference.mask);
    writeBinding(
      this.options.device.queue,
      work.binding("chunk-mask"),
      createAceDirectV1ChunkMask(1, request.latentFrames),
    );
    const dispatchSource: AceConditionerDispatchSource = semantic === undefined
      ? { kind: "direct-silence" }
      : {
          kind: "planner-semantic-cover",
          semanticCodeCount: semantic.plan.codeCount,
          semanticFrames: semantic.plan.outputFrames,
          semanticHints: semantic.retained.binding("semantic-hints"),
        };
    const bindings = conditionerBindings(
      text.retained,
      work,
      finalArena,
      resolved.silenceSource,
      resolved.weights,
    );
    const dispatch = await runtime.createConditionerDispatch(
      "ace-production-conditioner",
      {
        batch: 1,
        textTokens: request.textTokenIds.length,
        lyricTokens: request.lyricTokenIds.length,
        latentFrames: request.latentFrames,
      },
      dispatchSource,
      bindings,
    );
    await this.runQuanta("conditioner", dispatch.quanta, false);
    const weightBytes = phase.residentBytes;
    const workingBytes = work.byteLength;
    this.releaseOwned(runtime);
    this.releaseOwned(work);
    this.releaseOwned(phase);
    return Object.freeze({ finalArena, plan, weightBytes, workingBytes });
  }

  private async runQuanta(
    phase: AceConditioningGpuProgress["phase"],
    quanta: readonly QuantumEncoder[],
    finalGraphCommand: boolean,
  ): Promise<void> {
    if (quanta.length === 0) throw new RangeError(`ACE ${phase} has no GPU quanta`);
    let drains = 0;
    let idleMs = 0;
    for (let index = 0; index < quanta.length; index += 1) {
      this.lifetime.signal.throwIfAborted();
      const quantum = quanta[index]!;
      const encoder = this.options.device.createCommandEncoder({
        label: `${quantum.id}-command`,
      });
      quantum.beforePass?.(encoder);
      if (quantum.encodeCommand !== undefined) {
        quantum.encodeCommand(encoder);
      } else {
        const pass = encoder.beginComputePass({ label: quantum.id });
        quantum.encode!(pass);
        pass.end();
      }
      const commandBuffer = encoder.finish();
      const scheduled = await this.scheduler.run({
        queue: this.options.device.queue,
        commandBuffers: [commandBuffer],
        signal: this.lifetime.signal,
        ownerSignal: this.lifetime.signal,
      });
      drains += scheduled.queueDrains;
      const final = !aceConditioningNeedsIdleAfterQuantum(
        finalGraphCommand,
        index,
        quanta.length,
      );
      if (!final) {
        const idle = (this.options.yieldQueueIdle ?? yieldQueueIdle)();
        idleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
        try {
          this.options.onProgress?.({
            phase,
            completedCommandBuffers: index + 1,
            totalCommandBuffers: quanta.length,
            queueDrains: drains,
            cooperativeIdleMs: idleMs,
            quantumId: quantum.id,
          });
        } catch (error) {
          await idle;
          throw error;
        }
        await idle;
      } else {
        this.options.onProgress?.({
          phase,
          completedCommandBuffers: index + 1,
          totalCommandBuffers: quanta.length,
          queueDrains: drains,
          cooperativeIdleMs: idleMs,
          quantumId: quantum.id,
        });
      }
    }
  }

  private own<T extends Destroyable>(resource: T): T {
    this.owned.add(resource);
    return resource;
  }

  private async ownAsync<T extends Destroyable>(promise: Promise<T>): Promise<T> {
    const resource = await promise;
    this.owned.add(resource);
    return resource;
  }

  private releaseOwned(resource: Destroyable): void {
    if (!this.owned.delete(resource)) return;
    resource.destroy();
  }

  private destroyOwned(): void {
    for (const resource of [...this.owned].reverse()) resource.destroy();
    this.owned.clear();
  }

  private beginDestroy(): Promise<void> {
    if (this.destroyPromise !== undefined) return this.destroyPromise;
    if (this.state === "completed" || this.state === "destroyed") {
      this.destroyPromise = Promise.resolve();
      return this.destroyPromise;
    }
    this.state = "destroying";
    this.destroyPromise = (async () => {
      await this.graphOwner.dispose();
      await this.scheduler.dispose();
      this.destroyOwned();
      this.state = "destroyed";
      this.detachAbortListener();
    })();
    return this.destroyPromise;
  }

  private requireLive(): void {
    if (this.state !== "live") {
      throw new Error(`ACE conditioning executor is ${this.state}`);
    }
    this.options.signal?.throwIfAborted();
    this.lifetime.signal.throwIfAborted();
  }

  private detachAbortListener(): void {
    if (this.externalAbortListener !== undefined) {
      this.options.signal?.removeEventListener("abort", this.externalAbortListener);
    }
  }
}

interface TextPhaseOutput {
  readonly retained: NamedArena;
  readonly weightBytes: number;
  readonly workingBytes: number;
}

interface SemanticPhaseOutput extends TextPhaseOutput {
  readonly plan: AceSemanticDecodePlan;
}

interface ConditionerPhaseOutput {
  readonly finalArena: NamedArena;
  readonly plan: AceDirectConditionerPlan;
  readonly weightBytes: number;
  readonly workingBytes: number;
}

interface QuantumEncoder {
  readonly id: string;
  readonly beforePass?: (encoder: GPUCommandEncoder) => void;
  readonly encode?: (pass: GPUComputePassEncoder) => void;
  readonly encodeCommand?: (encoder: GPUCommandEncoder) => void;
}

interface ArenaEntry {
  readonly label: string;
  readonly bytes: number;
}

export interface AceConditioningGpuMemoryPlanInput {
  readonly modelProfile: AceModelProfileId;
  readonly textTokens: number;
  readonly lyricTokens: number;
  readonly latentFrames: number;
  readonly semanticCodeTokens: number;
  readonly textWeightBytes: number;
  readonly semanticWeightBytes: number;
  readonly conditionerWeightBytes: number;
}

/** Exact allocation accounting for the three disjoint GPU lifetimes. */
export function planAceConditioningGpuMemory(
  input: AceConditioningGpuMemoryPlanInput,
): AceConditioningGpuMemoryAccounting {
  const textPlan = planAceTextEncoder(input.modelProfile, {
    batch: 1,
    tokens: input.textTokens,
  });
  if (!Number.isSafeInteger(input.lyricTokens) || input.lyricTokens <= 0) {
    throw new RangeError("ACE conditioning memory lyricTokens must be positive");
  }
  if (!Number.isSafeInteger(input.latentFrames) || input.latentFrames <= 0) {
    throw new RangeError("ACE conditioning memory latentFrames must be positive");
  }
  for (const [name, bytes] of Object.entries({
    textWeightBytes: input.textWeightBytes,
    semanticWeightBytes: input.semanticWeightBytes,
    conditionerWeightBytes: input.conditionerWeightBytes,
  })) {
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new RangeError(`ACE conditioning memory ${name} is invalid`);
    }
  }
  if (!Number.isSafeInteger(input.semanticCodeTokens) || input.semanticCodeTokens < 0) {
    throw new RangeError("ACE conditioning memory semanticCodeTokens is invalid");
  }
  const textRetainedBytes = accountedArenaBytes([
    activationEntry(input.modelProfile, "text-hidden", textPlan.outputElements),
    activationEntry(
      input.modelProfile,
      "lyric-hidden",
      input.lyricTokens * ACE_TEXT_QWEN3_CONFIG.hiddenSize,
    ),
    u32Entry("text-mask", input.textTokens),
    u32Entry("lyric-mask", input.lyricTokens),
  ]);
  const textWorkingBytes = accountedTextWorkBytes(
    input.modelProfile,
    textPlan,
    input.lyricTokens,
  );
  const semanticPlan = input.semanticCodeTokens === 0
    ? undefined
    : planAceSemanticDecode({ batch: 1, codeTokens: input.semanticCodeTokens });
  const semanticRetainedBytes = semanticPlan === undefined
    ? 0
    : accountedArenaBytes([
        activationEntry(input.modelProfile, "semantic-hints", semanticPlan.outputElements),
      ]);
  const semanticWorkingBytes = semanticPlan === undefined
    ? 0
    : accountedSemanticWorkBytes(input.modelProfile, semanticPlan) + STORAGE_ALIGNMENT;
  const conditionerPlan = planAceConditioner({
    batch: 1,
    textTokens: input.textTokens,
    lyricTokens: input.lyricTokens,
    latentFrames: input.latentFrames,
  }, semanticPlan === undefined
    ? { kind: "direct-silence" }
    : {
        kind: "planner-semantic-cover",
        semanticCodeCount: semanticPlan.codeCount,
        semanticFrames: semanticPlan.outputFrames,
      });
  const conditionerWorkingBytes = accountedConditionerWorkBytes(
    input.modelProfile,
    conditionerPlan,
  );
  const resultEntries = [
    activationEntry(
      input.modelProfile,
      "condition-hidden-states",
      conditionerPlan.conditionElements,
    ),
    u32Entry("condition-mask", conditionerPlan.conditionTokens),
    activationEntry(
      input.modelProfile,
      "context-latents",
      conditionerPlan.contextElements,
    ),
  ];
  const resultBytes = accountedArenaBytes(resultEntries);
  const resultReadbackBytes = createResultReadbackLayout(
    input.modelProfile,
    conditionerPlan,
  ).byteLength;
  const returnedCpuBytes = checkedSum([
    conditionerPlan.conditionElements * F32_BYTES,
    conditionerPlan.conditionTokens * U32_BYTES,
    conditionerPlan.contextElements * F32_BYTES,
  ], "ACE conditioning returned CPU bytes");
  return Object.freeze({
    textWeightBytes: input.textWeightBytes,
    textRetainedBytes,
    textWorkingBytes,
    semanticWeightBytes: input.semanticWeightBytes,
    semanticRetainedBytes,
    semanticWorkingBytes,
    conditionerWeightBytes: input.conditionerWeightBytes,
    conditionerWorkingBytes,
    resultBytes,
    resultReadbackBytes,
    returnedCpuBytes,
    peakAccountedGpuBytes: Math.max(
      input.textWeightBytes + textRetainedBytes + textWorkingBytes,
      textRetainedBytes + input.semanticWeightBytes +
        semanticRetainedBytes + semanticWorkingBytes,
      textRetainedBytes + semanticRetainedBytes +
        input.conditionerWeightBytes + conditionerWorkingBytes + resultBytes,
      resultBytes + resultReadbackBytes,
    ),
  });
}

class NamedArena implements Destroyable {
  readonly byteLength: number;
  private readonly indices: ReadonlyMap<string, number>;
  private readonly bytes: ReadonlyMap<string, number>;

  constructor(
    private readonly arena: AceGpuArena,
    entries: readonly ArenaEntry[],
  ) {
    this.byteLength = arena.byteLength;
    this.indices = new Map(entries.map((entry, index) => [entry.label, index]));
    this.bytes = new Map(entries.map((entry) => [entry.label, align(entry.bytes, 4)]));
  }

  binding(label: string): GPUBufferBinding {
    const index = this.indices.get(label);
    const bytes = this.bytes.get(label);
    if (index === undefined || bytes === undefined) {
      throw new Error(`ACE arena binding ${label} is absent`);
    }
    return this.arena.binding(this.arena.slice(label, index, 0, bytes));
  }

  destroy(): void {
    this.arena.destroy();
  }
}

async function createNamedArena(
  device: GPUDevice,
  prefix: string,
  entries: readonly ArenaEntry[],
): Promise<NamedArena> {
  const labels = new Set<string>();
  const plans: AceArenaBufferPlan[] = entries.map((entry) => {
    if (labels.has(entry.label)) throw new Error(`Duplicate ACE arena entry ${entry.label}`);
    labels.add(entry.label);
    if (!Number.isSafeInteger(entry.bytes) || entry.bytes <= 0) {
      throw new RangeError(`Invalid ACE arena entry ${entry.label}`);
    }
    return Object.freeze({
      label: `${prefix}-${entry.label}`,
      byteLength: align(entry.bytes, STORAGE_ALIGNMENT),
    });
  });
  const arena = await AceGpuArena.create(device, plans);
  return new NamedArena(arena, entries);
}

function createTextWorkArena(
  device: GPUDevice,
  profile: AceModelProfileId,
  plan: AceQwen3ModelPlan,
  lyricTokens: number,
): Promise<NamedArena> {
  return createNamedArena(
    device,
    "ace-text-work",
    textWorkEntries(profile, plan, lyricTokens),
  );
}

function textWorkEntries(
  profile: AceModelProfileId,
  plan: AceQwen3ModelPlan,
  lyricTokens: number,
): ArenaEntry[] {
  const entries: ArenaEntry[] = [
    u32Entry("text-token-ids", plan.tokens),
    u32Entry("lyric-token-ids", lyricTokens),
    u32Entry("valid-lengths", 2),
    u32Entry("query-positions", plan.tokens),
    u32Entry("key-validity", plan.tokens),
    f32Entry("cosine", plan.tokens * ACE_TEXT_QWEN3_CONFIG.headDimension),
    f32Entry("sine", plan.tokens * ACE_TEXT_QWEN3_CONFIG.headDimension),
    activationEntry(profile, "embedded", plan.hiddenElements),
    activationEntry(profile, "layer-output-0", plan.hiddenElements),
    activationEntry(profile, "layer-output-1", plan.hiddenElements),
  ];
  addQwenScratch(entries, profile, plan);
  return entries;
}

function createSemanticWorkArena(
  device: GPUDevice,
  profile: AceModelProfileId,
  plan: AceSemanticDecodePlan,
): Promise<NamedArena> {
  return createNamedArena(
    device,
    "ace-semantic-work",
    semanticWorkEntries(profile, plan),
  );
}

function semanticWorkEntries(
  profile: AceModelProfileId,
  plan: AceSemanticDecodePlan,
): ArenaEntry[] {
  const block = planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
    batch: plan.codeCount,
    tokens: ACE_SEMANTIC_POOL_WIDTH,
    attentionMode: "full",
  });
  const entries: ArenaEntry[] = [
    u32Entry("code-ids", plan.codeCount),
    u32Entry("validation-status", 1),
    u32Entry("valid-lengths", plan.codeCount * 2),
    f32Entry("cosine", ACE_SEMANTIC_POOL_WIDTH * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    f32Entry("sine", ACE_SEMANTIC_POOL_WIDTH * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    activationEntry(profile, "fsq-scalars", plan.codeCount * 6),
    activationEntry(profile, "quantized", plan.codeCount * ACE_CONDITION_HIDDEN_SIZE),
    activationEntry(profile, "embedded-codes", plan.codeCount * ACE_CONDITION_HIDDEN_SIZE),
    activationEntry(profile, "patch-input", block.hiddenElements),
    activationEntry(profile, "layer-output-0", block.hiddenElements),
    activationEntry(profile, "layer-output-1", block.hiddenElements),
    activationEntry(profile, "normalized", block.hiddenElements),
  ];
  addEncoderScratch(entries, "block", profile, block);
  return entries;
}

function createConditionerWorkArena(
  device: GPUDevice,
  profile: AceModelProfileId,
  plan: AceDirectConditionerPlan,
): Promise<NamedArena> {
  return createNamedArena(
    device,
    "ace-conditioner-work",
    conditionerWorkEntries(profile, plan),
  );
}

function conditionerWorkEntries(
  profile: AceModelProfileId,
  plan: AceDirectConditionerPlan,
): ArenaEntry[] {
  const lyricBlock = planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
    batch: plan.batch,
    tokens: plan.lyricTokens,
    attentionMode: "full",
  });
  const timbreBlock = planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
    batch: plan.batch,
    tokens: ACE_NO_REFERENCE_TIMBRE_FRAMES,
    attentionMode: "full",
  });
  const entries: ArenaEntry[] = [
    u32Entry("lyric-valid-lengths", plan.batch * 2),
    f32Entry("lyric-cosine", plan.lyricTokens * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    f32Entry("lyric-sine", plan.lyricTokens * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    u32Entry("timbre-valid-lengths", plan.batch * 2),
    f32Entry("timbre-cosine", ACE_NO_REFERENCE_TIMBRE_FRAMES * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    f32Entry("timbre-sine", ACE_NO_REFERENCE_TIMBRE_FRAMES * ACE_CONDITION_ENCODER_CONFIG.headDimension),
    u32Entry("timbre-first-row", plan.batch),
    u32Entry("timbre-mask", plan.batch),
    u32Entry("chunk-mask", plan.batch * plan.latentFrames),
    activationEntry(profile, "text-projected", plan.textRows * ACE_CONDITION_HIDDEN_SIZE),
    activationEntry(profile, "lyric-projected", lyricBlock.hiddenElements),
    activationEntry(profile, "lyric-output-a", lyricBlock.hiddenElements),
    activationEntry(profile, "lyric-output-b", lyricBlock.hiddenElements),
    activationEntry(profile, "lyric-encoded", lyricBlock.hiddenElements),
    activationEntry(
      profile,
      "timbre-source",
      plan.timbreRows * ACE_AUDIO_LATENT_CHANNELS,
    ),
    activationEntry(profile, "timbre-projected", timbreBlock.hiddenElements),
    activationEntry(profile, "timbre-output-a", timbreBlock.hiddenElements),
    activationEntry(profile, "timbre-output-b", timbreBlock.hiddenElements),
    activationEntry(profile, "timbre-normalized", timbreBlock.hiddenElements),
    activationEntry(profile, "timbre-token", plan.batch * ACE_CONDITION_HIDDEN_SIZE),
    u32Entry("first-pack-indices", plan.batch * plan.firstPackedTokens),
    activationEntry(
      profile,
      "first-packed",
      plan.batch * plan.firstPackedTokens * ACE_CONDITION_HIDDEN_SIZE,
    ),
    u32Entry("first-packed-mask", plan.batch * plan.firstPackedTokens),
    u32Entry("second-pack-indices", plan.batch * plan.conditionTokens),
    activationEntry(
      profile,
      "source-latents",
      plan.batch * plan.latentFrames * ACE_AUDIO_LATENT_CHANNELS,
    ),
  ];
  addEncoderScratch(entries, "lyric-block", profile, lyricBlock);
  addEncoderScratch(entries, "timbre-block", profile, timbreBlock);
  return entries;
}

function accountedTextWorkBytes(
  profile: AceModelProfileId,
  plan: AceQwen3ModelPlan,
  lyricTokens: number,
): number {
  return accountedArenaBytes(textWorkEntries(profile, plan, lyricTokens));
}

function accountedSemanticWorkBytes(
  profile: AceModelProfileId,
  plan: AceSemanticDecodePlan,
): number {
  return accountedArenaBytes(semanticWorkEntries(profile, plan));
}

function accountedConditionerWorkBytes(
  profile: AceModelProfileId,
  plan: AceDirectConditionerPlan,
): number {
  return accountedArenaBytes(conditionerWorkEntries(profile, plan));
}

function accountedArenaBytes(entries: readonly ArenaEntry[]): number {
  return checkedSum(
    entries.map((entry) => align(entry.bytes, STORAGE_ALIGNMENT)),
    "ACE conditioning arena bytes",
  );
}

function qwenScratch(arena: NamedArena): AceQwen3BlockScratch {
  return scratchRecord(arena, "scratch") as unknown as AceQwen3BlockScratch;
}

function encoderScratch(arena: NamedArena, prefix: string): AceEncoderBlockScratch {
  return scratchRecord(arena, prefix) as unknown as AceEncoderBlockScratch;
}

function scratchRecord(
  arena: NamedArena,
  prefix: string,
): Readonly<Record<string, GPUBufferBinding>> {
  return Object.freeze(Object.fromEntries(SCRATCH_FIELDS.map((name) => [
    name,
    arena.binding(`${prefix}-${camelToKebab(name)}`),
  ])));
}

function semanticBindings(
  work: NamedArena,
  retained: NamedArena,
  weights: AceSemanticDecodeBindings["weights"],
): AceSemanticDecodeBindings {
  return Object.freeze({
    codeIds: work.binding("code-ids"),
    output: retained.binding("semantic-hints"),
    weights,
    controls: Object.freeze({
      validationStatus: work.binding("validation-status"),
      validLengths: work.binding("valid-lengths"),
      cosine: work.binding("cosine"),
      sine: work.binding("sine"),
    }),
    scratch: Object.freeze({
      fsqScalars: work.binding("fsq-scalars"),
      quantized: work.binding("quantized"),
      embeddedCodes: work.binding("embedded-codes"),
      patchInput: work.binding("patch-input"),
      block: encoderScratch(work, "block"),
      layerOutputs: Object.freeze([
        work.binding("layer-output-0"),
        work.binding("layer-output-1"),
      ]),
      normalized: work.binding("normalized"),
    }),
  });
}

function conditionerBindings(
  text: NamedArena,
  work: NamedArena,
  output: NamedArena,
  silenceSource: GPUBufferBinding,
  weights: AceDirectConditionerBindings["weights"],
): AceDirectConditionerBindings {
  const lyricA = work.binding("lyric-output-a");
  const lyricB = work.binding("lyric-output-b");
  const timbreA = work.binding("timbre-output-a");
  const timbreB = work.binding("timbre-output-b");
  return Object.freeze({
    textHiddenStates: text.binding("text-hidden"),
    lyricHiddenStates: text.binding("lyric-hidden"),
    textMask: text.binding("text-mask"),
    lyricMask: text.binding("lyric-mask"),
    silenceSource,
    chunkMask: work.binding("chunk-mask"),
    output: Object.freeze({
      conditionHiddenStates: output.binding("condition-hidden-states"),
      conditionMask: output.binding("condition-mask"),
      contextLatents: output.binding("context-latents"),
    }),
    weights,
    controls: Object.freeze({
      lyricValidLengths: work.binding("lyric-valid-lengths"),
      lyricCosine: work.binding("lyric-cosine"),
      lyricSine: work.binding("lyric-sine"),
      timbreValidLengths: work.binding("timbre-valid-lengths"),
      timbreCosine: work.binding("timbre-cosine"),
      timbreSine: work.binding("timbre-sine"),
      timbreFirstRowIndices: work.binding("timbre-first-row"),
      timbreMask: work.binding("timbre-mask"),
    }),
    scratch: Object.freeze({
      textProjected: work.binding("text-projected"),
      lyricProjected: work.binding("lyric-projected"),
      lyricBlock: encoderScratch(work, "lyric-block"),
      lyricLayerOutputs: Object.freeze([
        lyricA, lyricB, lyricA, lyricB, lyricA, lyricB, lyricA, lyricB,
      ]),
      lyricEncoded: work.binding("lyric-encoded"),
      timbreSource: work.binding("timbre-source"),
      timbreProjected: work.binding("timbre-projected"),
      timbreBlock: encoderScratch(work, "timbre-block"),
      timbreLayerOutputs: Object.freeze([timbreA, timbreB, timbreA, timbreB]),
      timbreNormalized: work.binding("timbre-normalized"),
      timbreToken: work.binding("timbre-token"),
      firstPackIndices: work.binding("first-pack-indices"),
      firstPacked: work.binding("first-packed"),
      firstPackedMask: work.binding("first-packed-mask"),
      secondPackIndices: work.binding("second-pack-indices"),
      sourceLatents: work.binding("source-latents"),
    }),
  });
}

function addQwenScratch(
  entries: ArenaEntry[],
  profile: AceModelProfileId,
  plan: AceQwen3ModelPlan,
): void {
  addScratchBySizes(entries, "scratch", profile, {
    hiddenElements: plan.hiddenElements,
    queryElements: plan.queryElements,
    keyValueElements: plan.keyValueElements,
    intermediateElements: plan.intermediateElements,
  });
}

function addEncoderScratch(
  entries: ArenaEntry[],
  prefix: string,
  profile: AceModelProfileId,
  plan: Readonly<{
    hiddenElements: number;
    queryElements: number;
    keyValueElements: number;
    intermediateElements: number;
  }>,
): void {
  addScratchBySizes(entries, prefix, profile, plan);
}

function addScratchBySizes(
  entries: ArenaEntry[],
  prefix: string,
  profile: AceModelProfileId,
  plan: Readonly<{
    hiddenElements: number;
    queryElements: number;
    keyValueElements: number;
    intermediateElements: number;
  }>,
): void {
  for (const name of HIDDEN_SCRATCH_FIELDS) {
    entries.push(activationEntry(profile, `${prefix}-${camelToKebab(name)}`, plan.hiddenElements));
  }
  for (const name of QUERY_SCRATCH_FIELDS) {
    entries.push(activationEntry(profile, `${prefix}-${camelToKebab(name)}`, plan.queryElements));
  }
  for (const name of KV_SCRATCH_FIELDS) {
    entries.push(activationEntry(profile, `${prefix}-${camelToKebab(name)}`, plan.keyValueElements));
  }
  for (const name of INTERMEDIATE_SCRATCH_FIELDS) {
    entries.push(activationEntry(
      profile,
      `${prefix}-${camelToKebab(name)}`,
      plan.intermediateElements,
    ));
  }
}

const HIDDEN_SCRATCH_FIELDS = Object.freeze([
  "normalizedInput",
  "projectedAttention",
  "afterAttention",
  "normalizedAfterAttention",
  "projectedMlp",
] as const);
const QUERY_SCRATCH_FIELDS = Object.freeze([
  "queryFlat",
  "queryHeads",
  "normalizedQueryHeads",
  "rotatedQueryHeads",
  "attentionHeads",
  "mergedAttention",
] as const);
const KV_SCRATCH_FIELDS = Object.freeze([
  "keyFlat",
  "valueFlat",
  "keyHeads",
  "valueHeads",
  "normalizedKeyHeads",
  "rotatedKeyHeads",
] as const);
const INTERMEDIATE_SCRATCH_FIELDS = Object.freeze([
  "gate",
  "up",
  "gatedActivation",
] as const);
const SCRATCH_FIELDS = Object.freeze([
  ...HIDDEN_SCRATCH_FIELDS,
  ...QUERY_SCRATCH_FIELDS,
  ...KV_SCRATCH_FIELDS,
  ...INTERMEDIATE_SCRATCH_FIELDS,
] as const);

function activationEntry(
  profile: AceModelProfileId,
  label: string,
  elements: number,
): ArenaEntry {
  return Object.freeze({ label, bytes: aceActivationBytes(profile, elements) });
}

function u32Entry(label: string, elements: number): ArenaEntry {
  return Object.freeze({ label, bytes: checkedAceProduct([elements, U32_BYTES], label) });
}

function f32Entry(label: string, elements: number): ArenaEntry {
  return Object.freeze({ label, bytes: checkedAceProduct([elements, F32_BYTES], label) });
}

function validateRequest(request: AceConditioningGpuRequest): void {
  requireNonEmptyU32(request.textTokenIds, "text token IDs");
  requireNonEmptyU32(request.lyricTokenIds, "lyric token IDs");
  requireBinaryMask(request.textMask, request.textTokenIds.length, "text mask");
  requireBinaryMask(request.lyricMask, request.lyricTokenIds.length, "lyric mask");
  if (!Number.isSafeInteger(request.latentFrames) || request.latentFrames <= 0) {
    throw new RangeError("ACE conditioning latentFrames must be positive");
  }
  for (let index = 0; index < request.textTokenIds.length; index += 1) {
    if (request.textTokenIds[index]! >= ACE_TEXT_QWEN3_CONFIG.vocabularySize) {
      throw new RangeError(`ACE text token ${index} is outside the text vocabulary`);
    }
  }
  for (let index = 0; index < request.lyricTokenIds.length; index += 1) {
    if (request.lyricTokenIds[index]! >= ACE_TEXT_QWEN3_CONFIG.vocabularySize) {
      throw new RangeError(`ACE lyric token ${index} is outside the text vocabulary`);
    }
  }
  if (request.mode.kind === "planner") {
    requireNonEmptyU32(request.mode.semanticCodeIds, "semantic code IDs");
    for (let index = 0; index < request.mode.semanticCodeIds.length; index += 1) {
      if (request.mode.semanticCodeIds[index]! >= ACE_FSQ_CODEBOOK_SIZE) {
        throw new RangeError(`ACE semantic code ${index} is outside the FSQ codebook`);
      }
    }
  } else if (request.mode.kind !== "direct") {
    throw new TypeError(
      `Unknown ACE conditioning mode ${String((request.mode as { kind?: unknown }).kind)}`,
    );
  }
}

/** Snapshot every caller-mutable array before the first asynchronous boundary. */
export function snapshotAceConditioningGpuRequest(
  request: AceConditioningGpuRequest,
): AceConditioningGpuRequestSnapshot {
  return Object.freeze({
    textTokenIds: copyDetachedU32(request.textTokenIds),
    lyricTokenIds: copyDetachedU32(request.lyricTokenIds),
    textMask: copyDetachedU32(request.textMask),
    lyricMask: copyDetachedU32(request.lyricMask),
    latentFrames: request.latentFrames,
    mode: request.mode.kind === "direct"
      ? Object.freeze({ kind: "direct" as const })
      : Object.freeze({
          kind: "planner" as const,
          semanticCodeIds: copyDetachedU32(request.mode.semanticCodeIds),
        }),
  });
}

/** Every phase-local last quantum is non-final until the result readback. */
export function aceConditioningNeedsIdleAfterQuantum(
  finalGraphCommand: boolean,
  index: number,
  quantumCount: number,
): boolean {
  if (
    !Number.isSafeInteger(index) ||
    !Number.isSafeInteger(quantumCount) ||
    index < 0 ||
    quantumCount <= 0 ||
    index >= quantumCount
  ) {
    throw new RangeError("ACE conditioning quantum position is invalid");
  }
  return !finalGraphCommand || index !== quantumCount - 1;
}

function copyDetachedU32(source: Uint32Array): Uint32Array<ArrayBuffer> {
  const output = new Uint32Array(source.length) as Uint32Array<ArrayBuffer>;
  output.set(source);
  return output;
}

function requireNonEmptyU32(value: Uint32Array, label: string): void {
  if (!(value instanceof Uint32Array) || value.length === 0) {
    throw new TypeError(`ACE ${label} must be a non-empty Uint32Array`);
  }
}

function requireBinaryMask(mask: Uint32Array, length: number, label: string): void {
  if (!(mask instanceof Uint32Array) || mask.length !== length) {
    throw new RangeError(`ACE ${label} has the wrong length`);
  }
  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] !== 0 && mask[index] !== 1) {
      throw new RangeError(`ACE ${label}[${index}] must be zero or one`);
    }
  }
}

function requireManifestProfile(
  manifest: AcePackageManifest,
  modelProfile: AceModelProfileId,
): void {
  const expected = manifest.profile === "reference" ? "reference-bf16" : "raw-fp16";
  if (modelProfile !== expected) {
    throw new Error(
      `ACE conditioning profile ${modelProfile} differs from package ${manifest.profile}`,
    );
  }
}

export function validateAceConditioningPhaseSet(
  phase: Pick<AceGpuTensorPhase, "phases">,
  expected: readonly string[],
  label: string,
): void {
  if (
    phase.phases.length !== expected.length ||
    expected.some((value) => !phase.phases.includes(value as never))
  ) {
    throw new Error(
      `ACE ${label} owner requires phases ${expected.join("+")}; got ${phase.phases.join("+")}`,
    );
  }
}

export function validateAceConditioningPhaseManifest(
  phase: Pick<AceGpuTensorPhase, "packageManifest">,
  expected: AcePackageManifest,
  label: string,
): void {
  if (phase.packageManifest !== expected) {
    throw new Error(
      `ACE ${label} phase was not authenticated by this executor's manifest identity`,
    );
  }
}

function writeBinding(
  queue: GPUQueue,
  binding: GPUBufferBinding,
  source: Uint32Array | Float32Array,
): void {
  const available = binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  if (source.byteLength > available) {
    throw new RangeError("ACE conditioning upload exceeds its allocation");
  }
  if (!(source.buffer instanceof ArrayBuffer)) {
    throw new TypeError("ACE conditioning upload requires ArrayBuffer storage");
  }
  queue.writeBuffer(
    binding.buffer,
    binding.offset ?? 0,
    source.buffer,
    source.byteOffset,
    source.byteLength,
  );
}

function clearBinding(encoder: GPUCommandEncoder, binding: GPUBufferBinding): void {
  encoder.clearBuffer(binding.buffer, binding.offset ?? 0, binding.size);
}

function copyBindingToBuffer(
  encoder: GPUCommandEncoder,
  source: GPUBufferBinding,
  destination: GPUBuffer,
  bytes: number,
): void {
  encoder.copyBufferToBuffer(
    source.buffer,
    source.offset ?? 0,
    destination,
    0,
    bytes,
  );
}

function copyBindingToBufferAt(
  encoder: GPUCommandEncoder,
  source: GPUBufferBinding,
  destination: GPUBuffer,
  destinationOffset: number,
  bytes: number,
): void {
  encoder.copyBufferToBuffer(
    source.buffer,
    source.offset ?? 0,
    destination,
    destinationOffset,
    bytes,
  );
}

interface ResultReadbackLayout {
  readonly conditionOffset: number;
  readonly conditionStoredBytes: number;
  readonly maskOffset: number;
  readonly maskBytes: number;
  readonly contextOffset: number;
  readonly contextStoredBytes: number;
  readonly byteLength: number;
}

function createResultReadbackLayout(
  profile: AceModelProfileId,
  plan: AceDirectConditionerPlan,
): ResultReadbackLayout {
  let cursor = 0;
  const conditionOffset = cursor;
  const conditionStoredBytes = aceActivationBytes(profile, plan.conditionElements);
  cursor += conditionStoredBytes;
  cursor = align(cursor, STORAGE_ALIGNMENT);
  const maskOffset = cursor;
  const maskBytes = plan.batch * plan.conditionTokens * U32_BYTES;
  cursor += maskBytes;
  cursor = align(cursor, STORAGE_ALIGNMENT);
  const contextOffset = cursor;
  const contextStoredBytes = aceActivationBytes(profile, plan.contextElements);
  cursor += contextStoredBytes;
  return Object.freeze({
    conditionOffset,
    conditionStoredBytes,
    maskOffset,
    maskBytes,
    contextOffset,
    contextStoredBytes,
    byteLength: align(cursor, STORAGE_ALIGNMENT),
  });
}

export function decodeAceConditioningActivation(
  profile: AceModelProfileId,
  mapped: ArrayBuffer,
  byteOffset: number,
  elements: number,
): Float32Array<ArrayBuffer> {
  const output = new Float32Array(elements) as Float32Array<ArrayBuffer>;
  if (profile === "reference-bf16") {
    output.set(new Float32Array(mapped, byteOffset, elements));
    return output;
  }
  if (profile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE conditioning profile ${String(profile)}`);
  }
  const source = new Uint16Array(mapped, byteOffset, elements);
  const scratch = new ArrayBuffer(4);
  const scratchU32 = new Uint32Array(scratch);
  const scratchF32 = new Float32Array(scratch);
  for (let index = 0; index < elements; index += 1) {
    scratchU32[0] = fp16ToFloat32Bits(source[index]!);
    output[index] = scratchF32[0]!;
  }
  return output;
}

function copyU32(
  mapped: ArrayBuffer,
  byteOffset: number,
  elements: number,
): Uint32Array<ArrayBuffer> {
  const output = new Uint32Array(elements) as Uint32Array<ArrayBuffer>;
  output.set(new Uint32Array(mapped, byteOffset, elements));
  return output;
}

function fp16ToFloat32Bits(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  let word: number;
  if (exponent === 0) {
    if (fraction === 0) {
      word = sign;
    } else {
      let mantissa = fraction;
      let shift = 0;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        shift += 1;
      }
      mantissa &= 0x03ff;
      word = sign | ((127 - 14 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    word = sign | 0x7f800000 | (fraction << 13);
  } else {
    word = sign | ((exponent + 112) << 23) | (fraction << 13);
  }
  return word >>> 0;
}

function bufferOwner(buffer: GPUBuffer): Destroyable {
  return Object.freeze({ destroy: () => buffer.destroy() });
}

function yieldQueueIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

function camelToKebab(value: string): string {
  return value.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

function align(value: number, alignment: number): number {
  const result = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(result)) throw new RangeError("ACE conditioning alignment overflow");
  return result;
}

function checkedSum(values: readonly number[], label: string): number {
  let result = 0;
  for (const value of values) {
    result += value;
    if (!Number.isSafeInteger(result) || result < 0) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}
