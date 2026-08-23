import { ACE_GRAPH_CONTRACT } from "../model/graph-contract.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessAttentionKernel,
  type AceAttentionDispatch,
} from "./kernels/attention.js";
import {
  aceCompositeCooperativeQuanta,
  type AceGpuEncodeQuantum,
  type AceGemmBufferBindings,
  type AceGemmDispatch,
  type AceGemmShape,
} from "./kernels/gemm.js";
import {
  AceCorrectnessKvCacheWriteKernel,
  type AceKvCacheWriteDispatch,
} from "./kernels/kv-cache.js";
import {
  AceCorrectnessRmsNormKernel,
  type AceRmsNormDispatch,
} from "./kernels/rmsnorm.js";
import {
  AceCorrectnessBatchedRopeKernel,
  type AceBatchedRopeDispatch,
} from "./kernels/batched-rope.js";
import {
  AceCorrectnessTransformerPlumbingKernel,
  type AceTransformerDispatch,
} from "./kernels/transformer-plumbing.js";
import {
  aceActivationBytes,
  acePackedWeightBytes,
} from "./kernels/correctness-utils.js";
import {
  AceOpt0087PlannerDenseOwner,
  type AceOpt0087PlannerDenseArm,
  type AceOpt0087PlannerDenseRole,
  type AceOpt0087PlannerDenseSelection,
} from "./planner-dense-owner.js";

const MAX_U32 = 0xffff_ffff;

/**
 * Provenance for the pinned Qwen3 rotary-frequency oracle below.
 *
 * The words were emitted by the exact Transformers expression
 *
 *   1 / (theta ** (torch.arange(0, dim, 2).float() / dim))
 *
 * from the ACE source revision and dependency versions named here. The source
 * file hash authenticates the implementation that was inspected; the vector
 * hash authenticates the 64 little-endian FP32 words committed below.
 */
export const ACE_QWEN3_ROPE_REFERENCE_PROVENANCE = Object.freeze({
  aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  transformersVersion: "4.57.6",
  transformersWheelSha256:
    "4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550",
  transformersQwen3SourceSha256:
    "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
  torchVersion: "2.10.0",
  inverseFrequencyLittleEndianSha256:
    "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e",
} as const);

/** Exact FP32 words for `dim=128, theta=1_000_000` in frequency order. */
export const ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS: readonly number[] =
  Object.freeze([
    0x3f800000, 0x3f4e4bad, 0x3f263de0, 0x3f05f6ef,
    0x3ed7e89b, 0x3eadfcff, 0x3e8c3504, 0x3e61f835,
    0x3e361887, 0x3e12bd91, 0x3dec7fd6, 0x3dbe94c6,
    0x3d99940d, 0x3d778513, 0x3d47763f, 0x3d20bc1d,
    0x3d0186e3, 0x3cd0c1a8, 0x3ca8398b, 0x3c879008,
    0x3c5a7bf2, 0x3c301052, 0x3c0de12d, 0x3be4aa46,
    0x3bb8449c, 0x3b947dae, 0x3b6f520e, 0x3b40dac5,
    0x3b1b690d, 0x3afa78f0, 0x3ac9d75c, 0x3aa2a6f7,
    0x3a83126f, 0x3a533f28, 0x3a2a3b44, 0x3a092e02,
    0x39dd1725, 0x39b229fb, 0x398f9272, 0x39676492,
    0x393a7753, 0x39164324, 0x38f22ce2, 0x38c327b5,
    0x389d43a4, 0x387d75d4, 0x384c3fbe, 0x382497ab,
    0x3804a2b2, 0x37d5c441, 0x37ac431d, 0x378ad0ed,
    0x375fba4f, 0x37344a0e, 0x371148e3, 0x36ea2732,
    0x36bcb0c1, 0x36980e02, 0x36751070, 0x36457bab,
    0x361f23e4, 0x36003dec, 0x35ceaf79, 0x35a68e4c,
  ]);

// These two authenticated toy tables are retained only for the independently
// oracled miniature browser graph and contract tests. Arbitrary geometries do
// not fall back to JavaScript's differently rounded `Math.pow` path.
const ACE_QWEN3_TEST_INV_FREQUENCY_WORDS = Object.freeze({
  "2:100": Object.freeze([0x3f800000]),
  "4:100": Object.freeze([0x3f800000, 0x3dcccccd]),
} as const);

export interface AceQwen3Config {
  readonly id: string;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly layerCount: number;
  readonly queryHeads: number;
  readonly keyValueHeads: number;
  readonly headDimension: number;
  readonly vocabularySize: number;
  readonly maximumPositionEmbeddings: number;
  readonly ropeTheta: number;
  readonly rmsNormEpsilon: number;
  readonly attentionBias: false;
  readonly hiddenActivation: "silu";
  readonly tieWordEmbeddings: true;
}

export const ACE_TEXT_QWEN3_CONFIG: Readonly<AceQwen3Config> = Object.freeze({
  id: "ace-text-qwen3-embedding-0.6b",
  hiddenSize: ACE_GRAPH_CONTRACT.textEncoder.hiddenSize,
  intermediateSize: ACE_GRAPH_CONTRACT.textEncoder.intermediateSize,
  layerCount: ACE_GRAPH_CONTRACT.textEncoder.layerCount,
  queryHeads: ACE_GRAPH_CONTRACT.textEncoder.attentionHeads,
  keyValueHeads: ACE_GRAPH_CONTRACT.textEncoder.keyValueHeads,
  headDimension: ACE_GRAPH_CONTRACT.textEncoder.headDimension,
  vocabularySize: ACE_GRAPH_CONTRACT.textEncoder.vocabularySize,
  maximumPositionEmbeddings:
    ACE_GRAPH_CONTRACT.textEncoder.maximumPositionEmbeddings,
  ropeTheta: ACE_GRAPH_CONTRACT.textEncoder.ropeTheta,
  rmsNormEpsilon: 1e-6,
  attentionBias: false,
  hiddenActivation: "silu",
  tieWordEmbeddings: true,
});

export const ACE_PLANNER_QWEN3_CONFIG: Readonly<AceQwen3Config> = Object.freeze({
  id: "ace-planner-qwen3-0.6b",
  hiddenSize: ACE_GRAPH_CONTRACT.planner.hiddenSize,
  intermediateSize: ACE_GRAPH_CONTRACT.planner.intermediateSize,
  layerCount: ACE_GRAPH_CONTRACT.planner.layerCount,
  queryHeads: ACE_GRAPH_CONTRACT.planner.attentionHeads,
  keyValueHeads: ACE_GRAPH_CONTRACT.planner.keyValueHeads,
  headDimension: ACE_GRAPH_CONTRACT.planner.headDimension,
  vocabularySize: ACE_GRAPH_CONTRACT.planner.vocabularySize,
  maximumPositionEmbeddings:
    ACE_GRAPH_CONTRACT.planner.maximumPositionEmbeddings,
  ropeTheta: ACE_GRAPH_CONTRACT.planner.ropeTheta,
  rmsNormEpsilon: 1e-6,
  attentionBias: false,
  hiddenActivation: "silu",
  tieWordEmbeddings: true,
});

export type AceQwen3AttentionStorage =
  | Readonly<{ readonly kind: "uncached" }>
  | Readonly<{
      readonly kind: "cached";
      readonly cacheCapacity: number;
    }>;

export interface AceQwen3BlockShape {
  readonly batch: number;
  readonly tokens: number;
  readonly attention: AceQwen3AttentionStorage;
}

export interface AceQwen3BlockPlan extends AceQwen3BlockShape {
  readonly rows: number;
  readonly queryWidth: number;
  readonly keyValueWidth: number;
  readonly hiddenElements: number;
  readonly queryElements: number;
  readonly keyValueElements: number;
  readonly intermediateElements: number;
  readonly attentionKeyValueTokens: number;
}

export interface AceQwen3BlockWeights {
  readonly inputLayerNorm: GPUBufferBinding;
  readonly queryProjection: GPUBufferBinding;
  readonly keyProjection: GPUBufferBinding;
  readonly valueProjection: GPUBufferBinding;
  readonly queryNorm: GPUBufferBinding;
  readonly keyNorm: GPUBufferBinding;
  readonly outputProjection: GPUBufferBinding;
  readonly postAttentionLayerNorm: GPUBufferBinding;
  readonly gateProjection: GPUBufferBinding;
  readonly upProjection: GPUBufferBinding;
  readonly downProjection: GPUBufferBinding;
}

/**
 * Explicit correctness scratch. Distinct fields make every model-math edge
 * inspectable in a golden capture; Stage 1 callers must not alias them.
 */
export interface AceQwen3BlockScratch {
  readonly normalizedInput: GPUBufferBinding;
  readonly queryFlat: GPUBufferBinding;
  readonly keyFlat: GPUBufferBinding;
  readonly valueFlat: GPUBufferBinding;
  readonly queryHeads: GPUBufferBinding;
  readonly keyHeads: GPUBufferBinding;
  readonly valueHeads: GPUBufferBinding;
  readonly normalizedQueryHeads: GPUBufferBinding;
  readonly normalizedKeyHeads: GPUBufferBinding;
  readonly rotatedQueryHeads: GPUBufferBinding;
  readonly rotatedKeyHeads: GPUBufferBinding;
  readonly attentionHeads: GPUBufferBinding;
  readonly mergedAttention: GPUBufferBinding;
  readonly projectedAttention: GPUBufferBinding;
  readonly afterAttention: GPUBufferBinding;
  readonly normalizedAfterAttention: GPUBufferBinding;
  readonly gate: GPUBufferBinding;
  readonly up: GPUBufferBinding;
  readonly gatedActivation: GPUBufferBinding;
  readonly projectedMlp: GPUBufferBinding;
}

export interface AceQwen3CausalBindings {
  /** U32 `[batch,2]`: local query extent and physical cache extent. */
  readonly validLengths: GPUBufferBinding;
  /** U32 `[batch,tokens]`: physical cache position of every local query. */
  readonly queryPositions: GPUBufferBinding;
  /** FP32 `[batch,tokens,headDimension]`; generated from explicit rotary IDs. */
  readonly cosine: GPUBufferBinding;
  readonly sine: GPUBufferBinding;
}

export interface AceQwen3UncachedBindings extends AceQwen3CausalBindings {
  readonly kind: "uncached";
  /** U32 `[batch,tokens]`; exactly one admits a key. */
  readonly keyValidity: GPUBufferBinding;
}

export interface AceQwen3CachedBindings extends AceQwen3CausalBindings {
  readonly kind: "cached";
  /** U32 `[batch,tokens]`; exactly one admits an appended key. */
  readonly sourceValidity: GPUBufferBinding;
  readonly cacheKey: GPUBufferBinding;
  readonly cacheValue: GPUBufferBinding;
  readonly cacheValidity: GPUBufferBinding;
  /** U32 `[batch]`; must match the CPU-validated append plan. */
  readonly rowStartPositions: GPUBufferBinding;
  /** U32 `[batch]`; one means the complete row append was admitted. */
  readonly writeStatus: GPUBufferBinding;
}

export type AceQwen3AttentionBindings =
  | AceQwen3UncachedBindings
  | AceQwen3CachedBindings;

export interface AceQwen3BlockBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly weights: AceQwen3BlockWeights;
  readonly scratch: AceQwen3BlockScratch;
  readonly attention: AceQwen3AttentionBindings;
}

type AceQwen3PrimitiveDispatch =
  | AceGemmDispatch
  | AceRmsNormDispatch
  | AceBatchedRopeDispatch
  | AceAttentionDispatch
  | AceKvCacheWriteDispatch
  | AceTransformerDispatch<unknown>;

export interface AceQwen3BlockDispatch {
  readonly label: string;
  readonly plan: AceQwen3BlockPlan;
  readonly primitiveCount: number;
  readonly cooperativeQuanta: readonly AceGpuEncodeQuantum[];
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceQwen3CausalControlInput {
  readonly batch: number;
  readonly tokens: number;
  readonly cacheCapacity: number;
  readonly rowStartPositions: readonly number[];
  readonly validKeyLengths: readonly number[];
  readonly sourceValidity: readonly number[];
}

export interface AceQwen3CausalControlData {
  readonly rowStartPositions: Uint32Array;
  readonly validLengths: Uint32Array;
  readonly queryPositions: Uint32Array;
  readonly sourceValidity: Uint32Array;
}

export interface AceQwen3RopeTables {
  readonly cosine: Float32Array;
  readonly sine: Float32Array;
}

export interface AceQwen3RopeTableShape {
  readonly batch: number;
  readonly tokens: number;
  readonly headDimension: number;
  readonly ropeTheta: number;
  readonly maximumPositionEmbeddings: number;
}

export interface AceQwen3TiedWeightShard {
  readonly firstRow: number;
  readonly rowCount: number;
  /** The exact embedding-table row shard, reused as the LM-head weight. */
  readonly weight: GPUBufferBinding;
  /** Shard-local logits for one normalized hidden row. */
  readonly logits: GPUBufferBinding;
}

export interface AceQwen3TiedOutputDispatch {
  readonly label: string;
  readonly vocabularySize: number;
  readonly shards: readonly Readonly<{
    readonly firstRow: number;
    readonly rowCount: number;
  }>[];
  readonly cooperativeQuanta: readonly AceGpuEncodeQuantum[];
  encode(pass: GPUComputePassEncoder): void;
}

export function aceQwen3BlockCooperativeQuanta(
  dispatch: AceQwen3BlockDispatch,
): readonly AceGpuEncodeQuantum[] {
  return dispatch.cooperativeQuanta;
}

/** @internal One of the seven immutable dense edges in a Qwen decoder block. */
export interface AceOpt0087Qwen3DenseRequest {
  readonly label: string;
  readonly role: Exclude<AceOpt0087PlannerDenseRole, "tied-lm-head">;
  readonly shape: AceGemmShape;
  readonly bindings: AceGemmBufferBindings;
}

/** @internal Per-block OPT-0087 invocation state; never part of product API. */
export interface AceOpt0087Qwen3PlannerDenseOptions {
  readonly arm: AceOpt0087PlannerDenseArm;
  readonly kind: "prefill" | "decode";
  readonly onSelection: (selection: AceOpt0087PlannerDenseSelection) => void;
}

/**
 * @internal
 * Single source of truth for all seven Qwen dense roles. The runtime consumes
 * these exact descriptors for both A and B, so the candidate cannot silently
 * omit a projection or substitute a copied weight binding.
 */
export function aceOpt0087Qwen3DenseRequests(
  label: string,
  config: AceQwen3Config,
  plan: AceQwen3BlockPlan,
  bindings: AceQwen3BlockBindings,
): readonly AceOpt0087Qwen3DenseRequest[] {
  const s = bindings.scratch;
  const w = bindings.weights;
  return Object.freeze([
    Object.freeze({
      label: `${label}-query-projection`,
      role: "query-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.queryWidth,
      }),
      bindings: Object.freeze({
        activation: s.normalizedInput,
        weight: w.queryProjection,
        output: s.queryFlat,
      }),
    }),
    Object.freeze({
      label: `${label}-key-projection`,
      role: "key-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }),
      bindings: Object.freeze({
        activation: s.normalizedInput,
        weight: w.keyProjection,
        output: s.keyFlat,
      }),
    }),
    Object.freeze({
      label: `${label}-value-projection`,
      role: "value-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }),
      bindings: Object.freeze({
        activation: s.normalizedInput,
        weight: w.valueProjection,
        output: s.valueFlat,
      }),
    }),
    Object.freeze({
      label: `${label}-attention-output-projection`,
      role: "attention-output-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: plan.queryWidth,
        columns: config.hiddenSize,
      }),
      bindings: Object.freeze({
        activation: s.mergedAttention,
        weight: w.outputProjection,
        output: s.projectedAttention,
      }),
    }),
    Object.freeze({
      label: `${label}-gate-projection`,
      role: "gate-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }),
      bindings: Object.freeze({
        activation: s.normalizedAfterAttention,
        weight: w.gateProjection,
        output: s.gate,
      }),
    }),
    Object.freeze({
      label: `${label}-up-projection`,
      role: "up-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }),
      bindings: Object.freeze({
        activation: s.normalizedAfterAttention,
        weight: w.upProjection,
        output: s.up,
      }),
    }),
    Object.freeze({
      label: `${label}-down-projection`,
      role: "down-projection" as const,
      shape: Object.freeze({
        rows: plan.rows,
        inner: config.intermediateSize,
        columns: config.hiddenSize,
      }),
      bindings: Object.freeze({
        activation: s.gatedActivation,
        weight: w.downProjection,
        output: s.projectedMlp,
      }),
    }),
  ]);
}

/**
 * Correctness-first Qwen3 decoder-block composition used by both pinned 0.6B
 * graphs. It emits only primitive dispatches and never owns model buffers.
 */
export class AceCorrectnessQwen3Runtime {
  readonly modelProfile: AceModelProfileId;

  private readonly gemm: AceOpt0087PlannerDenseOwner;
  private readonly rmsNorm: AceCorrectnessRmsNormKernel;
  private readonly rope: AceCorrectnessBatchedRopeKernel;
  private readonly attention: AceCorrectnessAttentionKernel;
  private readonly cacheWrite: AceCorrectnessKvCacheWriteKernel;
  private readonly plumbing: AceCorrectnessTransformerPlumbingKernel;
  private destroyed = false;

  private constructor(
    modelProfile: AceModelProfileId,
    gemm: AceOpt0087PlannerDenseOwner,
    rmsNorm: AceCorrectnessRmsNormKernel,
    rope: AceCorrectnessBatchedRopeKernel,
    attention: AceCorrectnessAttentionKernel,
    cacheWrite: AceCorrectnessKvCacheWriteKernel,
    plumbing: AceCorrectnessTransformerPlumbingKernel,
  ) {
    this.modelProfile = modelProfile;
    this.gemm = gemm;
    this.rmsNorm = rmsNorm;
    this.rope = rope;
    this.attention = attention;
    this.cacheWrite = cacheWrite;
    this.plumbing = plumbing;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessQwen3Runtime {
    return AceCorrectnessQwen3Runtime.createOwned(device, modelProfile, false);
  }

  /** @internal Paired pipeline owner for the frozen OPT-0087 browser gate. */
  static createForOpt0087Planner(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessQwen3Runtime {
    return AceCorrectnessQwen3Runtime.createOwned(device, modelProfile, true);
  }

  private static createOwned(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    pairedOpt0087: boolean,
  ): AceCorrectnessQwen3Runtime {
    let gemm: AceOpt0087PlannerDenseOwner | undefined;
    let rmsNorm: AceCorrectnessRmsNormKernel | undefined;
    let rope: AceCorrectnessBatchedRopeKernel | undefined;
    let attention: AceCorrectnessAttentionKernel | undefined;
    let cacheWrite: AceCorrectnessKvCacheWriteKernel | undefined;
    let plumbing: AceCorrectnessTransformerPlumbingKernel | undefined;
    try {
      gemm = pairedOpt0087
        ? AceOpt0087PlannerDenseOwner.createPairedForOpt0087(
            device,
            modelProfile,
          )
        : AceOpt0087PlannerDenseOwner.createGeneric(device, modelProfile);
      rmsNorm = AceCorrectnessRmsNormKernel.create(device, modelProfile);
      rope = AceCorrectnessBatchedRopeKernel.create(device, modelProfile);
      attention = AceCorrectnessAttentionKernel.create(device, modelProfile);
      cacheWrite = AceCorrectnessKvCacheWriteKernel.create(device, modelProfile);
      plumbing = AceCorrectnessTransformerPlumbingKernel.create(device, modelProfile);
      return new AceCorrectnessQwen3Runtime(
        modelProfile,
        gemm,
        rmsNorm,
        rope,
        attention,
        cacheWrite,
        plumbing,
      );
    } catch (error) {
      gemm?.destroy();
      rmsNorm?.destroy();
      rope?.destroy();
      attention?.destroy();
      cacheWrite?.destroy();
      plumbing?.destroy();
      throw error;
    }
  }

  async createBlockDispatch(
    label: string,
    config: AceQwen3Config,
    shape: AceQwen3BlockShape,
    bindings: AceQwen3BlockBindings,
  ): Promise<AceQwen3BlockDispatch> {
    return this.createBlockDispatchOwned(label, config, shape, bindings);
  }

  /** @internal Explicit per-invocation arm over one planner weight owner. */
  async createPlannerBlockDispatchForOpt0087(
    label: string,
    config: AceQwen3Config,
    shape: AceQwen3BlockShape,
    bindings: AceQwen3BlockBindings,
    options: AceOpt0087Qwen3PlannerDenseOptions,
  ): Promise<AceQwen3BlockDispatch> {
    return this.createBlockDispatchOwned(label, config, shape, bindings, options);
  }

  private async createBlockDispatchOwned(
    label: string,
    config: AceQwen3Config,
    shape: AceQwen3BlockShape,
    bindings: AceQwen3BlockBindings,
    opt0087?: AceOpt0087Qwen3PlannerDenseOptions,
  ): Promise<AceQwen3BlockDispatch> {
    this.requireLive();
    const plan = planAceQwen3Block(config, shape);
    if (bindings.attention.kind !== plan.attention.kind) {
      throw new TypeError(`${label} attention bindings do not match the block plan`);
    }
    requireDisjointQwen3GraphBindings(label, bindings);

    const hiddenShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      width: config.hiddenSize,
    } as const;
    const intermediateShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      width: config.intermediateSize,
    } as const;
    const queryHeadShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      heads: config.queryHeads,
      headDimension: config.headDimension,
    } as const;
    const keyValueHeadShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      heads: config.keyValueHeads,
      headDimension: config.headDimension,
    } as const;
    const s = bindings.scratch;
    const w = bindings.weights;
    const a = bindings.attention;
    const denseRequests = aceOpt0087Qwen3DenseRequests(
      label,
      config,
      plan,
      bindings,
    );
    const denseDispatch = (index: number): Promise<AceGemmDispatch> => {
      const request = denseRequests[index];
      if (request === undefined) {
        throw new RangeError(`${label} dense request ${index} is missing`);
      }
      if (opt0087 === undefined) {
        return this.gemm.createGenericDispatch(
          request.label,
          request.shape,
          request.bindings,
        );
      }
      return this.gemm.createDispatchForOpt0087({
        ...request,
        invocation: {
          owner: config.id === ACE_PLANNER_QWEN3_CONFIG.id
            ? "planner"
            : "non-planner",
          kind: opt0087.kind,
          batch: plan.batch,
          tokens: plan.tokens,
          requestedArm: opt0087.arm,
        },
      }, opt0087.onSelection).then(({ dispatch }) => dispatch);
    };

    const commonDispatches = await Promise.all([
      this.rmsNorm.createDispatch(`${label}-input-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: bindings.input,
        weight: w.inputLayerNorm,
        output: s.normalizedInput,
      }),
      denseDispatch(0),
      denseDispatch(1),
      denseDispatch(2),
      this.plumbing.createHeadTransformDispatch(
        `${label}-split-query-heads`,
        "split-heads",
        queryHeadShape,
        { input: s.queryFlat, output: s.queryHeads },
      ),
      this.plumbing.createHeadTransformDispatch(
        `${label}-split-key-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: s.keyFlat, output: s.keyHeads },
      ),
      this.plumbing.createHeadTransformDispatch(
        `${label}-split-value-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: s.valueFlat, output: s.valueHeads },
      ),
      this.rmsNorm.createDispatch(`${label}-query-norm`, {
        rows: plan.batch * config.queryHeads * plan.tokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.queryHeads,
        weight: w.queryNorm,
        output: s.normalizedQueryHeads,
      }),
      this.rmsNorm.createDispatch(`${label}-key-norm`, {
        rows: plan.batch * config.keyValueHeads * plan.tokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.keyHeads,
        weight: w.keyNorm,
        output: s.normalizedKeyHeads,
      }),
      this.rope.createDispatch(`${label}-query-rope`, {
        batch: plan.batch,
        heads: config.queryHeads,
        tokens: plan.tokens,
        headDimension: config.headDimension,
      }, {
        input: s.normalizedQueryHeads,
        cosine: a.cosine,
        sine: a.sine,
        output: s.rotatedQueryHeads,
      }),
      this.rope.createDispatch(`${label}-key-rope`, {
        batch: plan.batch,
        heads: config.keyValueHeads,
        tokens: plan.tokens,
        headDimension: config.headDimension,
      }, {
        input: s.normalizedKeyHeads,
        cosine: a.cosine,
        sine: a.sine,
        output: s.rotatedKeyHeads,
      }),
      this.plumbing.createHeadTransformDispatch(
        `${label}-merge-attention-heads`,
        "merge-heads",
        queryHeadShape,
        { input: s.attentionHeads, output: s.mergedAttention },
      ),
      denseDispatch(3),
      this.plumbing.createResidualAddDispatch(`${label}-attention-residual`, hiddenShape, {
        left: bindings.input,
        right: s.projectedAttention,
        output: s.afterAttention,
      }),
      this.rmsNorm.createDispatch(`${label}-post-attention-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.afterAttention,
        weight: w.postAttentionLayerNorm,
        output: s.normalizedAfterAttention,
      }),
      denseDispatch(4),
      denseDispatch(5),
      this.plumbing.createSwiGluDispatch(`${label}-swiglu`, intermediateShape, {
        gate: s.gate,
        up: s.up,
        output: s.gatedActivation,
      }),
      denseDispatch(6),
      this.plumbing.createResidualAddDispatch(`${label}-mlp-residual`, hiddenShape, {
        left: s.afterAttention,
        right: s.projectedMlp,
        output: bindings.output,
      }),
    ]);

    const prefix = commonDispatches.slice(0, 11);
    const suffix = commonDispatches.slice(11);
    const attentionDispatch = await this.attention.createDispatch(
      `${label}-causal-attention`,
      {
        batch: plan.batch,
        queryHeads: config.queryHeads,
        keyValueHeads: config.keyValueHeads,
        queryTokens: plan.tokens,
        keyValueTokens: plan.attentionKeyValueTokens,
        headDimension: config.headDimension,
        mode: "causal",
        keyValidity: "causal-per-key",
      },
      {
        query: s.rotatedQueryHeads,
        key: a.kind === "cached" ? a.cacheKey : s.rotatedKeyHeads,
        value: a.kind === "cached" ? a.cacheValue : s.valueHeads,
        validLengths: a.validLengths,
        output: s.attentionHeads,
        keyValidity: a.kind === "cached" ? a.cacheValidity : a.keyValidity,
        queryPositions: a.queryPositions,
      },
    );
    const middle: AceQwen3PrimitiveDispatch[] = [];
    if (a.kind === "cached") {
      middle.push(await this.cacheWrite.createDispatch(
        `${label}-cache-append`,
        {
          batch: plan.batch,
          keyValueHeads: config.keyValueHeads,
          appendTokens: plan.tokens,
          cacheCapacity: plan.attentionKeyValueTokens,
          headDimension: config.headDimension,
        },
        {
          sourceKey: s.rotatedKeyHeads,
          sourceValue: s.valueHeads,
          sourceValidity: a.sourceValidity,
          cacheKey: a.cacheKey,
          cacheValue: a.cacheValue,
          cacheValidity: a.cacheValidity,
          rowStartPositions: a.rowStartPositions,
          writeStatus: a.writeStatus,
        },
      ));
    }
    middle.push(attentionDispatch);
    this.requireLive(" while compiling");
    const dispatches: readonly AceQwen3PrimitiveDispatch[] = Object.freeze([
      ...prefix,
      ...middle,
      ...suffix,
    ]);
    const cooperativeQuanta = aceCompositeCooperativeQuanta(dispatches);
    return Object.freeze({
      label,
      plan,
      primitiveCount: dispatches.length,
      cooperativeQuanta,
      encode(pass: GPUComputePassEncoder): void {
        for (const dispatch of dispatches) dispatch.encode(pass);
      },
    });
  }

  async createFinalNormDispatch(
    label: string,
    config: AceQwen3Config,
    rows: number,
    bindings: Readonly<{
      readonly input: GPUBufferBinding;
      readonly weight: GPUBufferBinding;
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceRmsNormDispatch> {
    this.requireLive();
    validateAceQwen3Config(config);
    requirePositiveInteger(rows, `${label} rows`);
    return this.rmsNorm.createDispatch(label, {
      rows,
      width: config.hiddenSize,
      epsilon: config.rmsNormEpsilon,
    }, bindings);
  }

  /**
   * One-row planner LM head. Requiring embedding shards here makes the pinned
   * `tie_word_embeddings=true` contract structural rather than conventional.
   */
  async createTiedOutputDispatch(
    label: string,
    config: AceQwen3Config,
    normalizedHiddenRow: GPUBufferBinding,
    shards: readonly AceQwen3TiedWeightShard[],
  ): Promise<AceQwen3TiedOutputDispatch> {
    this.requireLive();
    validateAceQwen3Config(config);
    if (shards.length === 0) {
      throw new RangeError(`${label} requires at least one tied vocabulary shard`);
    }
    let nextRow = 0;
    const frozenShards = shards.map((shard, index) => {
      requirePositiveInteger(shard.rowCount, `${label} shard ${index} rowCount`);
      if (shard.firstRow !== nextRow) {
        throw new RangeError(`${label} shard ${index} must start at row ${nextRow}`);
      }
      nextRow += shard.rowCount;
      if (!Number.isSafeInteger(nextRow) || nextRow > MAX_U32) {
        throw new RangeError(`${label} vocabulary coverage exceeds U32`);
      }
      return Object.freeze({ firstRow: shard.firstRow, rowCount: shard.rowCount });
    });
    if (nextRow !== config.vocabularySize) {
      throw new RangeError(
        `${label} tied shards cover ${nextRow} rows, expected ${config.vocabularySize}`,
      );
    }
    validateAceQwen3TiedOutputBindings(
      label,
      this.modelProfile,
      config,
      normalizedHiddenRow,
      shards,
    );
    const dispatches = await Promise.all(shards.map((shard, index) =>
      this.gemm.createGenericDispatch(`${label}-rows-${shard.firstRow}`, {
        rows: 1,
        inner: config.hiddenSize,
        columns: shard.rowCount,
      }, {
        activation: normalizedHiddenRow,
        weight: shard.weight,
        output: shard.logits,
      }).catch((error: unknown) => {
        throw new Error(`${label} tied shard ${index} failed`, { cause: error });
      })
    ));
    this.requireLive(" while compiling");
    const cooperativeQuanta = aceCompositeCooperativeQuanta(dispatches);
    return Object.freeze({
      label,
      vocabularySize: config.vocabularySize,
      shards: Object.freeze(frozenShards),
      cooperativeQuanta,
      encode(pass: GPUComputePassEncoder): void {
        for (const dispatch of dispatches) dispatch.encode(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.gemm.destroy();
    this.rmsNorm.destroy();
    this.rope.destroy();
    this.attention.destroy();
    this.cacheWrite.destroy();
    this.plumbing.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) throw new Error(`ACE Qwen3 runtime was destroyed${suffix}`);
  }
}

export function planAceQwen3Block(
  config: AceQwen3Config,
  shape: AceQwen3BlockShape,
): AceQwen3BlockPlan {
  validateAceQwen3Config(config);
  requirePositiveInteger(shape.batch, "ACE Qwen3 batch");
  requirePositiveInteger(shape.tokens, "ACE Qwen3 tokens");
  if (shape.tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE Qwen3 tokens exceed maximumPositionEmbeddings");
  }
  if (shape.attention.kind !== "uncached" && shape.attention.kind !== "cached") {
    throw new TypeError(
      `Unknown ACE Qwen3 attention storage ${String((shape.attention as { kind?: unknown }).kind)}`,
    );
  }
  let attentionKeyValueTokens = shape.tokens;
  if (shape.attention.kind === "cached") {
    requirePositiveInteger(
      shape.attention.cacheCapacity,
      "ACE Qwen3 cacheCapacity",
    );
    if (shape.tokens > shape.attention.cacheCapacity) {
      throw new RangeError("ACE Qwen3 append tokens exceed cache capacity");
    }
    if (shape.attention.cacheCapacity > config.maximumPositionEmbeddings) {
      throw new RangeError("ACE Qwen3 cache exceeds maximumPositionEmbeddings");
    }
    attentionKeyValueTokens = shape.attention.cacheCapacity;
  }
  const rows = checkedProduct([shape.batch, shape.tokens], "ACE Qwen3 rows");
  const queryWidth = checkedProduct(
    [config.queryHeads, config.headDimension],
    "ACE Qwen3 query width",
  );
  const keyValueWidth = checkedProduct(
    [config.keyValueHeads, config.headDimension],
    "ACE Qwen3 KV width",
  );
  const hiddenElements = checkedProduct(
    [rows, config.hiddenSize],
    "ACE Qwen3 hidden elements",
  );
  const queryElements = checkedProduct(
    [rows, queryWidth],
    "ACE Qwen3 query elements",
  );
  const keyValueElements = checkedProduct(
    [rows, keyValueWidth],
    "ACE Qwen3 KV elements",
  );
  const intermediateElements = checkedProduct(
    [rows, config.intermediateSize],
    "ACE Qwen3 intermediate elements",
  );
  for (const [name, value] of Object.entries({
    hiddenElements,
    queryElements,
    keyValueElements,
    intermediateElements,
  })) {
    if (value > MAX_U32) {
      throw new RangeError(`ACE Qwen3 ${name} exceeds U32 shader indexing`);
    }
  }
  return Object.freeze({
    ...shape,
    attention: Object.freeze({ ...shape.attention }),
    rows,
    queryWidth,
    keyValueWidth,
    hiddenElements,
    queryElements,
    keyValueElements,
    intermediateElements,
    attentionKeyValueTokens,
  });
}

export function validateAceQwen3Config(config: AceQwen3Config): void {
  if (typeof config.id !== "string" || config.id.length === 0) {
    throw new TypeError("ACE Qwen3 config id must be non-empty");
  }
  for (const [name, value] of Object.entries({
    hiddenSize: config.hiddenSize,
    intermediateSize: config.intermediateSize,
    layerCount: config.layerCount,
    queryHeads: config.queryHeads,
    keyValueHeads: config.keyValueHeads,
    headDimension: config.headDimension,
    vocabularySize: config.vocabularySize,
    maximumPositionEmbeddings: config.maximumPositionEmbeddings,
  })) {
    requirePositiveInteger(value, `ACE Qwen3 ${name}`);
    if (value > MAX_U32) throw new RangeError(`ACE Qwen3 ${name} exceeds U32`);
  }
  if (config.queryHeads % config.keyValueHeads !== 0) {
    throw new RangeError("ACE Qwen3 query heads must be divisible by KV heads");
  }
  if (config.headDimension % 2 !== 0 || config.headDimension > 128) {
    throw new RangeError("ACE Qwen3 headDimension must be even and at most 128");
  }
  if (
    typeof config.ropeTheta !== "number" ||
    !Number.isFinite(config.ropeTheta) ||
    config.ropeTheta <= 0
  ) {
    throw new RangeError("ACE Qwen3 ropeTheta must be positive and finite");
  }
  if (
    typeof config.rmsNormEpsilon !== "number" ||
    !Number.isFinite(config.rmsNormEpsilon) ||
    config.rmsNormEpsilon <= 0
  ) {
    throw new RangeError("ACE Qwen3 rmsNormEpsilon must be positive and finite");
  }
  if (config.attentionBias !== false) {
    throw new TypeError("ACE Qwen3 correctness runtime does not permit attention bias");
  }
  if (config.hiddenActivation !== "silu") {
    throw new TypeError("ACE Qwen3 correctness runtime requires SiLU/SwiGLU");
  }
  if (config.tieWordEmbeddings !== true) {
    throw new TypeError("ACE Qwen3 pinned graphs require tied word embeddings");
  }
}

/**
 * Validates the exact byte ranges touched by the tied embedding/LM-head
 * dispatches. Read-only embedding shards may share a backing allocation, but
 * every logits range must be disjoint from the hidden row, every weight range,
 * and every other logits range for the complete multi-dispatch operation.
 */
export function validateAceQwen3TiedOutputBindings(
  label: string,
  modelProfile: AceModelProfileId,
  config: AceQwen3Config,
  normalizedHiddenRow: GPUBufferBinding,
  shards: readonly AceQwen3TiedWeightShard[],
): void {
  validateAceQwen3Config(config);
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(
      `Unknown ACE Qwen3 tied-output model profile ${String(modelProfile)}`,
    );
  }
  if (shards.length === 0) {
    throw new RangeError(`${label} requires at least one tied vocabulary shard`);
  }

  const hidden = usedBindingRange(
    normalizedHiddenRow,
    aceActivationBytes(modelProfile, config.hiddenSize),
    `${label} normalized hidden row`,
  );
  let nextRow = 0;
  const weights: AceNamedUsedBindingRange[] = [];
  const logits: AceNamedUsedBindingRange[] = [];
  for (let index = 0; index < shards.length; index += 1) {
    const shard = shards[index]!;
    requirePositiveInteger(shard.rowCount, `${label} shard ${index} rowCount`);
    if (shard.firstRow !== nextRow) {
      throw new RangeError(`${label} shard ${index} must start at row ${nextRow}`);
    }
    nextRow += shard.rowCount;
    if (!Number.isSafeInteger(nextRow) || nextRow > MAX_U32) {
      throw new RangeError(`${label} vocabulary coverage exceeds U32`);
    }
    const weightElements = checkedProduct(
      [shard.rowCount, config.hiddenSize],
      `${label} shard ${index} weight elements`,
    );
    weights.push({
      name: `weight shard ${index}`,
      binding: shard.weight,
      range: usedBindingRange(
        shard.weight,
        acePackedWeightBytes(modelProfile, weightElements),
        `${label} weight shard ${index}`,
      ),
    });
    logits.push({
      name: `logits shard ${index}`,
      binding: shard.logits,
      range: usedBindingRange(
        shard.logits,
        aceActivationBytes(modelProfile, shard.rowCount),
        `${label} logits shard ${index}`,
      ),
    });
  }
  if (nextRow !== config.vocabularySize) {
    throw new RangeError(
      `${label} tied shards cover ${nextRow} rows, expected ${config.vocabularySize}`,
    );
  }

  const hiddenBinding: AceNamedUsedBindingRange = {
    name: "normalized hidden row",
    binding: normalizedHiddenRow,
    range: hidden,
  };
  for (let index = 0; index < logits.length; index += 1) {
    const output = logits[index]!;
    requireNonOverlappingUsedBindings(label, output, hiddenBinding);
    for (const weight of weights) {
      requireNonOverlappingUsedBindings(label, output, weight);
    }
    for (let other = index + 1; other < logits.length; other += 1) {
      requireNonOverlappingUsedBindings(label, output, logits[other]!);
    }
  }
}

/**
 * Builds the only accepted control tensors for a causal cache append. Causal
 * positions refer to physical cache slots; rotary IDs are intentionally a
 * separate input to `createAceQwen3RopeTables`.
 */
export function createAceQwen3CausalControlData(
  input: AceQwen3CausalControlInput,
): AceQwen3CausalControlData {
  requirePositiveInteger(input.batch, "ACE Qwen3 control batch");
  requirePositiveInteger(input.tokens, "ACE Qwen3 control tokens");
  requirePositiveInteger(input.cacheCapacity, "ACE Qwen3 control cacheCapacity");
  if (input.tokens > input.cacheCapacity) {
    throw new RangeError("ACE Qwen3 control tokens exceed cache capacity");
  }
  requireArrayLength(
    input.rowStartPositions,
    input.batch,
    "ACE Qwen3 rowStartPositions",
  );
  requireArrayLength(
    input.validKeyLengths,
    input.batch,
    "ACE Qwen3 validKeyLengths",
  );
  requireArrayLength(
    input.sourceValidity,
    checkedProduct([input.batch, input.tokens], "ACE Qwen3 source validity"),
    "ACE Qwen3 sourceValidity",
  );
  const rowStarts = new Uint32Array(input.batch);
  const validLengths = new Uint32Array(input.batch * 2);
  const queryPositions = new Uint32Array(input.batch * input.tokens);
  const sourceValidity = new Uint32Array(input.sourceValidity.length);
  for (let batch = 0; batch < input.batch; batch += 1) {
    const rowStart = input.rowStartPositions[batch]!;
    const validKeyLength = input.validKeyLengths[batch]!;
    requireNonNegativeU32(rowStart, `ACE Qwen3 row ${batch} start`);
    requirePositiveInteger(validKeyLength, `ACE Qwen3 row ${batch} key length`);
    if (rowStart + input.tokens > input.cacheCapacity) {
      throw new RangeError(`ACE Qwen3 row ${batch} append exceeds cache capacity`);
    }
    if (
      validKeyLength < rowStart + input.tokens ||
      validKeyLength > input.cacheCapacity
    ) {
      throw new RangeError(
        `ACE Qwen3 row ${batch} key length does not contain the complete append`,
      );
    }
    rowStarts[batch] = rowStart;
    // Every local query is executed. Padding is fail-closed as a key and its
    // output is ignored by the graph; valid suffix queries remain reachable.
    validLengths[batch * 2] = input.tokens;
    validLengths[batch * 2 + 1] = validKeyLength;
    for (let token = 0; token < input.tokens; token += 1) {
      queryPositions[batch * input.tokens + token] = rowStart + token;
    }
  }
  for (let index = 0; index < input.sourceValidity.length; index += 1) {
    const value = input.sourceValidity[index]!;
    if (value !== 0 && value !== 1) {
      throw new RangeError(`ACE Qwen3 sourceValidity[${index}] must be zero or one`);
    }
    sourceValidity[index] = value;
  }
  return Object.freeze({
    rowStartPositions: rowStarts,
    validLengths,
    queryPositions,
    sourceValidity,
  });
}

/**
 * Qwen split-half RoPE tables using authenticated upstream FP32 frequencies.
 *
 * JavaScript rounds `Math.pow` only after a binary64 power and reciprocal;
 * PyTorch performs both operations in FP32. That difference reaches visible
 * phase error at planner context lengths, so unsupported frequency geometries
 * fail closed instead of claiming exactness through generic double math.
 * Host trigonometric implementations may still differ from PyTorch by one FP32
 * ULP; committed Transformers-derived vectors bound that remaining operation.
 */
export function createAceQwen3RopeTables(
  positionIds: readonly number[],
  shape: AceQwen3RopeTableShape,
): AceQwen3RopeTables {
  requirePositiveInteger(shape.batch, "ACE Qwen3 RoPE batch");
  requirePositiveInteger(shape.tokens, "ACE Qwen3 RoPE tokens");
  requireArrayLength(
    positionIds,
    checkedProduct([shape.batch, shape.tokens], "ACE Qwen3 RoPE positions"),
    "ACE Qwen3 RoPE positionIds",
  );
  if (
    !Number.isSafeInteger(shape.headDimension) ||
    shape.headDimension <= 0 ||
    shape.headDimension % 2 !== 0
  ) {
    throw new RangeError("ACE Qwen3 RoPE headDimension must be positive and even");
  }
  if (
    typeof shape.ropeTheta !== "number" ||
    !Number.isFinite(shape.ropeTheta) ||
    shape.ropeTheta <= 0
  ) {
    throw new RangeError("ACE Qwen3 RoPE theta must be positive and finite");
  }
  requirePositiveInteger(
    shape.maximumPositionEmbeddings,
    "ACE Qwen3 maximumPositionEmbeddings",
  );
  const half = shape.headDimension / 2;
  const inverseFrequencies = aceQwen3InverseFrequencies(
    shape.headDimension,
    shape.ropeTheta,
  );
  const cosine = new Float32Array(positionIds.length * shape.headDimension);
  const sine = new Float32Array(cosine.length);
  for (let token = 0; token < positionIds.length; token += 1) {
    const position = positionIds[token]!;
    requireNonNegativeU32(position, `ACE Qwen3 positionIds[${token}]`);
    if (position >= shape.maximumPositionEmbeddings) {
      throw new RangeError(`ACE Qwen3 positionIds[${token}] exceeds the model limit`);
    }
    for (let dimension = 0; dimension < half; dimension += 1) {
      const phase = Math.fround(position * inverseFrequencies[dimension]!);
      const cos = Math.fround(Math.cos(phase));
      const sin = Math.fround(Math.sin(phase));
      const first = token * shape.headDimension + dimension;
      cosine[first] = cos;
      cosine[first + half] = cos;
      sine[first] = sin;
      sine[first + half] = sin;
    }
  }
  return Object.freeze({ cosine, sine });
}

function aceQwen3InverseFrequencies(
  headDimension: number,
  ropeTheta: number,
): Float32Array {
  let words: readonly number[] | undefined;
  if (headDimension === 128 && ropeTheta === 1_000_000) {
    words = ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS;
  } else {
    const key = `${headDimension}:${ropeTheta}` as keyof
      typeof ACE_QWEN3_TEST_INV_FREQUENCY_WORDS;
    words = ACE_QWEN3_TEST_INV_FREQUENCY_WORDS[key];
  }
  if (words === undefined || words.length !== headDimension / 2) {
    throw new RangeError(
      `ACE Qwen3 RoPE has no authenticated FP32 inverse-frequency vector for ` +
        `headDimension=${headDimension}, ropeTheta=${ropeTheta}`,
    );
  }
  const storage = new ArrayBuffer(words.length * Uint32Array.BYTES_PER_ELEMENT);
  new Uint32Array(storage).set(words);
  return new Float32Array(storage);
}

function requireDisjointQwen3GraphBindings(
  label: string,
  bindings: AceQwen3BlockBindings,
): void {
  const scratch = Object.entries(bindings.scratch).map(([name, binding]) => ({
    name: `scratch ${name}`,
    binding,
  }));
  const writable: { readonly name: string; readonly binding: GPUBufferBinding }[] = [
    ...scratch,
    { name: "output", binding: bindings.output },
  ];
  const readonly: { readonly name: string; readonly binding: GPUBufferBinding }[] = [
    { name: "input", binding: bindings.input },
    ...Object.entries(bindings.weights).map(([name, binding]) => ({
      name: `weight ${name}`,
      binding,
    })),
    { name: "valid lengths", binding: bindings.attention.validLengths },
    { name: "query positions", binding: bindings.attention.queryPositions },
    { name: "cosine", binding: bindings.attention.cosine },
    { name: "sine", binding: bindings.attention.sine },
  ];
  if (bindings.attention.kind === "cached") {
    writable.push(
      { name: "cache key", binding: bindings.attention.cacheKey },
      { name: "cache value", binding: bindings.attention.cacheValue },
      { name: "cache validity", binding: bindings.attention.cacheValidity },
      { name: "write status", binding: bindings.attention.writeStatus },
    );
    readonly.push(
      { name: "source validity", binding: bindings.attention.sourceValidity },
      { name: "row start positions", binding: bindings.attention.rowStartPositions },
    );
  } else {
    readonly.push({ name: "key validity", binding: bindings.attention.keyValidity });
  }
  for (let leftIndex = 0; leftIndex < writable.length; leftIndex += 1) {
    const left = writable[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < writable.length; rightIndex += 1) {
      requireNonOverlappingBindings(label, left, writable[rightIndex]!);
    }
    for (const read of readonly) {
      requireNonOverlappingBindings(label, left, read);
    }
  }
}

function requireNonOverlappingBindings(
  label: string,
  left: Readonly<{ readonly name: string; readonly binding: GPUBufferBinding }>,
  right: Readonly<{ readonly name: string; readonly binding: GPUBufferBinding }>,
): void {
  const leftRange = bindingRange(left.binding, `${label} ${left.name}`);
  const rightRange = bindingRange(right.binding, `${label} ${right.name}`);
  if (
    left.binding.buffer === right.binding.buffer &&
    leftRange.start < rightRange.end &&
    rightRange.start < leftRange.end
  ) {
    throw new RangeError(`${label} ${left.name} overlaps ${right.name}`);
  }
}

interface AceNamedUsedBindingRange {
  readonly name: string;
  readonly binding: GPUBufferBinding;
  readonly range: Readonly<{ readonly start: number; readonly end: number }>;
}

function requireNonOverlappingUsedBindings(
  label: string,
  left: AceNamedUsedBindingRange,
  right: AceNamedUsedBindingRange,
): void {
  if (
    left.binding.buffer === right.binding.buffer &&
    left.range.start < right.range.end &&
    right.range.start < left.range.end
  ) {
    throw new RangeError(`${label} ${left.name} overlaps ${right.name}`);
  }
}

function usedBindingRange(
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): { readonly start: number; readonly end: number } {
  const exposed = bindingRange(binding, label);
  if (exposed.end - exposed.start < requiredBytes) {
    throw new RangeError(`${label} binding does not expose ${requiredBytes} bytes`);
  }
  return { start: exposed.start, end: exposed.start + requiredBytes };
}

function bindingRange(
  binding: GPUBufferBinding,
  label: string,
): { readonly start: number; readonly end: number } {
  const start = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - start;
  if (
    !Number.isSafeInteger(start) ||
    start < 0 ||
    !Number.isSafeInteger(size) ||
    size <= 0 ||
    start + size > binding.buffer.size
  ) {
    throw new RangeError(`${label} is not a valid buffer range`);
  }
  return { start, end: start + size };
}

function requireArrayLength(
  array: readonly number[],
  expected: number,
  label: string,
): void {
  if (array.length !== expected) {
    throw new RangeError(`${label} has ${array.length} entries; ${expected} required`);
  }
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function requireNonNegativeU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_U32) {
    throw new RangeError(`${label} must be a non-negative U32 integer`);
  }
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} is not a safe integer`);
    }
  }
  return result;
}
