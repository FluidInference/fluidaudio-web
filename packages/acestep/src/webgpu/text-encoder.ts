import type {
  AceGpuLogicalTensor,
} from "../model/gpu-tensors.js";
import type {
  AcePackageManifest,
  AcePackageTensorRecord,
} from "../model/manifest.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessEmbeddingKernel,
  type AceEmbeddingShardBinding,
} from "./kernels/embedding.js";
import {
  aceActivationBytes,
  requirePositiveSafeInteger,
} from "./kernels/correctness-utils.js";
import {
  acePrimitiveCooperativeQuanta,
} from "./kernels/gemm.js";
import {
  ACE_TEXT_QWEN3_CONFIG,
  AceCorrectnessQwen3Runtime,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  validateAceQwen3Config,
  type AceQwen3BlockDispatch,
  type AceQwen3BlockScratch,
  type AceQwen3BlockWeights,
  type AceQwen3Config,
  type AceQwen3RopeTables,
} from "./qwen3.js";

export const ACE_TEXT_ENCODER_MAX_TEXT_TOKENS = 256;
export const ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS = 2_048;

export const ACE_TEXT_ENCODER_INPUT_CONTRACT = Object.freeze({
  batch: 1,
  tokenizerPadding: "longest-single-item",
  textTruncationTokens: ACE_TEXT_ENCODER_MAX_TEXT_TOKENS,
  lyricTruncationTokens: ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS,
  textAttention: "causal-all-keys-valid",
  downstreamTextMask: "tokenizer-mask-bypasses-qwen-and-is-retained-for-conditioner",
  positionIds: "zero-based-physical-token-index",
  upstreamKeyValueCache: "created-by-config-default-but-output-dead",
  browserKeyValueCache: "elided-with-identical-last-hidden-state",
  output: "last-hidden-state-after-final-rmsnorm",
  lyricOutput: "embedding-table-lookup-only",
  downstreamLyricMask: "tokenizer-mask-bypasses-embedding-and-is-retained-for-conditioner",
} as const);

/**
 * Audited upstream contract for the production text path.
 *
 * ACE calls `AutoModel` on the authenticated Qwen directory and invokes
 * `Qwen3Model(input_ids=..., lyric_attention_mask=None)`. Transformers 4.57.6
 * does not declare `lyric_attention_mask`; it reaches `**kwargs` and leaves
 * `attention_mask` unset. Consequently the encoder is the ordinary causal
 * 28-layer Qwen3 model with positions `0..T-1` and a final RMSNorm. The input
 * decorator defaults `use_cache` from the config, so upstream creates an empty
 * DynamicCache and returns it; ACE observes only `last_hidden_state`. This
 * composer elides that output-dead copy while preserving identical attention
 * keys/values. Lyrics deliberately use only `embed_tokens`.
 */
export const ACE_TEXT_ENCODER_REFERENCE_PROVENANCE = Object.freeze({
  aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  modelSnapshotRevision: "19671f406d603126926c1b7e2adc169acbcade22",
  conditioningEmbedSourceSha256:
    "45e80702219496edcef5712fa8403583c61f0df128833b9657d0d129d035a3f2",
  conditioningTextSourceSha256:
    "0313ef5fc5ade1baa5a806f3748e5c853961768d057045e040fcb28a33a153b6",
  componentLoaderSourceSha256:
    "621fc7d24ee847835de2eb66f35451120cb92310cf5e112e4edbdf955535d6de",
  qwenConfigSha256:
    "bb23c1607cfe059a58d8f0196cf1cebb52082b1056b8e358a579da80a5759420",
  transformersVersion: "4.57.6",
  transformersWheelSha256:
    "4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550",
  transformersQwen3SourceSha256:
    "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
  transformersInputDecoratorSourceSha256:
    "94728825190dff491f2e0b93beb59c8246adb0b6049e3203c511405f43bb7e30",
} as const);

export interface AceTextEncoderLayerTensorNames {
  readonly inputLayerNorm: string;
  readonly queryProjection: string;
  readonly keyProjection: string;
  readonly valueProjection: string;
  readonly queryNorm: string;
  readonly keyNorm: string;
  readonly outputProjection: string;
  readonly postAttentionLayerNorm: string;
  readonly gateProjection: string;
  readonly upProjection: string;
  readonly downProjection: string;
}

export const ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES = Object.freeze({
  embedding: "text.embed_tokens.weight",
  finalNorm: "text.norm.weight",
} as const);

export function aceTextEncoderLayerTensorNames(
  layer: number,
): AceTextEncoderLayerTensorNames {
  if (!Number.isSafeInteger(layer) || layer < 0 || layer >= ACE_TEXT_QWEN3_CONFIG.layerCount) {
    throw new RangeError(
      `ACE text encoder layer must be in [0, ${ACE_TEXT_QWEN3_CONFIG.layerCount - 1}]`,
    );
  }
  const prefix = `text.layers.${layer}`;
  return Object.freeze({
    inputLayerNorm: `${prefix}.input_layernorm.weight`,
    queryProjection: `${prefix}.self_attn.q_proj.weight`,
    keyProjection: `${prefix}.self_attn.k_proj.weight`,
    valueProjection: `${prefix}.self_attn.v_proj.weight`,
    queryNorm: `${prefix}.self_attn.q_norm.weight`,
    keyNorm: `${prefix}.self_attn.k_norm.weight`,
    outputProjection: `${prefix}.self_attn.o_proj.weight`,
    postAttentionLayerNorm: `${prefix}.post_attention_layernorm.weight`,
    gateProjection: `${prefix}.mlp.gate_proj.weight`,
    upProjection: `${prefix}.mlp.up_proj.weight`,
    downProjection: `${prefix}.mlp.down_proj.weight`,
  });
}

export const ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES: readonly AceTextEncoderLayerTensorNames[] =
  Object.freeze(Array.from(
    { length: ACE_TEXT_QWEN3_CONFIG.layerCount },
    (_, layer) => aceTextEncoderLayerTensorNames(layer),
  ));

export const ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES: readonly string[] =
  Object.freeze([
    ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.embedding,
    ...ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES.flatMap((layer) => Object.values(layer)),
    ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.finalNorm,
  ]);

const ACE_TEXT_EMBEDDING_ROW_PARTS = Object.freeze([
  Object.freeze({ firstRow: 0, rowCount: 49_152 }),
  Object.freeze({ firstRow: 49_152, rowCount: 49_152 }),
  Object.freeze({ firstRow: 98_304, rowCount: 49_152 }),
  Object.freeze({ firstRow: 147_456, rowCount: 4_213 }),
]);

export const ACE_TEXT_ENCODER_PHYSICAL_TENSOR_NAMES: readonly string[] =
  Object.freeze([
    ...ACE_TEXT_EMBEDDING_ROW_PARTS.map(({ firstRow, rowCount }) =>
      `${ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.embedding}.rows-${padRow(firstRow)}-${padRow(firstRow + rowCount)}`
    ),
    ...ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES.flatMap((layer) => Object.values(layer)),
    ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.finalNorm,
  ]);

export interface AceTextEncoderTensorResolver {
  logicalTensor(logicalTensor: string): AceGpuLogicalTensor;
  binding(logicalTensor: string): GPUBufferBinding;
}

export interface AceTextEncoderWeights {
  readonly embedding: readonly AceEmbeddingShardBinding[];
  readonly layers: readonly AceQwen3BlockWeights[];
  readonly finalNorm: GPUBufferBinding;
}

export interface AceTextEncoderControlData extends AceQwen3RopeTables {
  /** U32 `[2]`: every local query and key is valid. */
  readonly validLengths: Uint32Array;
  /** U32 `[tokens]`: physical causal positions `0..T-1`. */
  readonly queryPositions: Uint32Array;
  /** U32 `[tokens]`: all ones for the batch-one `padding=longest` product path. */
  readonly keyValidity: Uint32Array;
}

export interface AceTextEncoderShape {
  readonly batch: 1;
  readonly tokens: number;
}

export interface AceQwen3ModelPlan {
  readonly batch: 1;
  readonly tokens: number;
  readonly rows: number;
  readonly layerCount: number;
  readonly hiddenElements: number;
  readonly queryElements: number;
  readonly keyValueElements: number;
  readonly intermediateElements: number;
  readonly outputElements: number;
  readonly scratchActivationElements: number;
  readonly scratchActivationBytes: number;
  readonly residentActivationElements: number;
  readonly residentActivationBytes: number;
}

export interface AceTextEncoderBindings {
  readonly tokenIds: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly weights: AceTextEncoderWeights;
  readonly controls: Readonly<{
    readonly validLengths: GPUBufferBinding;
    readonly queryPositions: GPUBufferBinding;
    readonly keyValidity: GPUBufferBinding;
    readonly cosine: GPUBufferBinding;
    readonly sine: GPUBufferBinding;
  }>;
  readonly scratch: Readonly<{
    readonly embedded: GPUBufferBinding;
    /** Shared across sequential layer quanta; every field must be disjoint. */
    readonly block: AceQwen3BlockScratch;
    /** Two disjoint activation ranges used as layer-to-layer ping-pong storage. */
    readonly layerOutputs: readonly [GPUBufferBinding, GPUBufferBinding];
  }>;
}

export interface AceTextEncoderQuantum {
  readonly id: string;
  readonly primitiveCount: number;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceTextEncoderDispatch {
  readonly label: string;
  readonly plan: AceQwen3ModelPlan;
  readonly primitiveCount: number;
  /** Embedding, one entry per decoder layer, then final norm. */
  readonly quanta: readonly AceTextEncoderQuantum[];
  /** Correctness harness convenience; production submits `quanta` cooperatively. */
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceLyricEmbeddingDispatch {
  readonly label: string;
  readonly tokenCount: number;
  readonly primitiveCount: number;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Correctness-first Qwen3Model composer. It owns pipelines only: the FIFO graph
 * owner supplies authenticated weights, controls, activations, and scratch.
 */
export class AceCorrectnessTextEncoderRuntime {
  private readonly embedding: AceCorrectnessEmbeddingKernel;
  private readonly qwen: AceCorrectnessQwen3Runtime;
  private destroyed = false;

  private constructor(
    readonly modelProfile: AceModelProfileId,
    embedding: AceCorrectnessEmbeddingKernel,
    qwen: AceCorrectnessQwen3Runtime,
  ) {
    this.embedding = embedding;
    this.qwen = qwen;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessTextEncoderRuntime {
    let embedding: AceCorrectnessEmbeddingKernel | undefined;
    let qwen: AceCorrectnessQwen3Runtime | undefined;
    try {
      embedding = AceCorrectnessEmbeddingKernel.create(device, modelProfile);
      qwen = AceCorrectnessQwen3Runtime.create(device, modelProfile);
      return new AceCorrectnessTextEncoderRuntime(modelProfile, embedding, qwen);
    } catch (error) {
      embedding?.destroy();
      qwen?.destroy();
      throw error;
    }
  }

  async createTextEncoderDispatch(
    label: string,
    shape: AceTextEncoderShape,
    bindings: AceTextEncoderBindings,
  ): Promise<AceTextEncoderDispatch> {
    if (shape.tokens > ACE_TEXT_ENCODER_MAX_TEXT_TOKENS) {
      throw new RangeError(
        `ACE product text encoder tokens exceed ${ACE_TEXT_ENCODER_MAX_TEXT_TOKENS}`,
      );
    }
    return this.createQwen3ModelDispatch(
      label,
      ACE_TEXT_QWEN3_CONFIG,
      shape,
      bindings,
    );
  }

  /**
   * Real uncached Qwen3Model composition shared by the pinned graph and the
   * authenticated miniature Metal oracle. Product code uses the exact wrapper
   * above, which additionally enforces the 256-token ACE preprocessing cap.
   */
  async createQwen3ModelDispatch(
    label: string,
    config: AceQwen3Config,
    shape: AceTextEncoderShape,
    bindings: AceTextEncoderBindings,
  ): Promise<AceTextEncoderDispatch> {
    this.requireLive();
    const plan = planAceQwen3Model(this.modelProfile, config, shape);
    if (bindings.weights.layers.length !== config.layerCount) {
      throw new RangeError(
        `${label} has ${bindings.weights.layers.length} layer bindings; ` +
          `${config.layerCount} required`,
      );
    }
    if (bindings.scratch.layerOutputs.length !== 2) {
      throw new RangeError(`${label} requires two layer-output ping-pong bindings`);
    }

    const embeddingDispatch = await this.embedding.createDispatch(
      `${label}-embed-tokens`,
      {
        tokenCount: plan.tokens,
        width: config.hiddenSize,
        vocabularySize: config.vocabularySize,
      },
      {
        tokenIds: bindings.tokenIds,
        shards: bindings.weights.embedding,
        output: bindings.scratch.embedded,
      },
    );
    const layers: AceQwen3BlockDispatch[] = [];
    for (let layer = 0; layer < config.layerCount; layer += 1) {
      const input = layer === 0
        ? bindings.scratch.embedded
        : bindings.scratch.layerOutputs[(layer - 1) & 1]!;
      const output = bindings.scratch.layerOutputs[layer & 1]!;
      layers.push(await this.qwen.createBlockDispatch(
        `${label}-layer-${layer}`,
        config,
        { batch: 1, tokens: plan.tokens, attention: { kind: "uncached" } },
        {
          input,
          output,
          weights: bindings.weights.layers[layer]!,
          scratch: bindings.scratch.block,
          attention: {
            kind: "uncached",
            validLengths: bindings.controls.validLengths,
            queryPositions: bindings.controls.queryPositions,
            keyValidity: bindings.controls.keyValidity,
            cosine: bindings.controls.cosine,
            sine: bindings.controls.sine,
          },
        },
      ));
    }
    const finalNorm = await this.qwen.createFinalNormDispatch(
      `${label}-final-norm`,
      config,
      plan.rows,
      {
        input: bindings.scratch.layerOutputs[(config.layerCount - 1) & 1]!,
        weight: bindings.weights.finalNorm,
        output: bindings.output,
      },
    );
    this.requireLive(" while compiling");

    const quanta: AceTextEncoderQuantum[] = [
      ...acePrimitiveCooperativeQuanta(embeddingDispatch),
      ...layers.flatMap((layer) => layer.cooperativeQuanta),
      ...acePrimitiveCooperativeQuanta(finalNorm),
    ];
    const primitiveCount = 2 + layers.reduce(
      (total, layer) => total + layer.primitiveCount,
      0,
    );
    return Object.freeze({
      label,
      plan,
      primitiveCount,
      quanta: Object.freeze(quanta),
      encode(pass: GPUComputePassEncoder): void {
        for (const entry of quanta) entry.encode(pass);
      },
    });
  }

  /** Exact upstream `infer_lyric_embeddings`: embedding-table lookup only. */
  async createLyricEmbeddingDispatch(
    label: string,
    tokenCount: number,
    bindings: Readonly<{
      readonly tokenIds: GPUBufferBinding;
      readonly embedding: readonly AceEmbeddingShardBinding[];
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceLyricEmbeddingDispatch> {
    this.requireLive();
    requirePositiveSafeInteger(tokenCount, `${label} tokenCount`);
    if (tokenCount > ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS) {
      throw new RangeError(
        `ACE lyric embedding tokens exceed ${ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS}`,
      );
    }
    return this.createQwen3EmbeddingDispatch(
      label,
      ACE_TEXT_QWEN3_CONFIG,
      tokenCount,
      bindings,
    );
  }

  /** Real embedding-table gather used by the authenticated miniature oracle. */
  async createQwen3EmbeddingDispatch(
    label: string,
    config: AceQwen3Config,
    tokenCount: number,
    bindings: Readonly<{
      readonly tokenIds: GPUBufferBinding;
      readonly embedding: readonly AceEmbeddingShardBinding[];
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceLyricEmbeddingDispatch> {
    this.requireLive();
    validateAceQwen3Config(config);
    requirePositiveSafeInteger(tokenCount, `${label} tokenCount`);
    const dispatch = await this.embedding.createDispatch(label, {
      tokenCount,
      width: config.hiddenSize,
      vocabularySize: config.vocabularySize,
    }, {
      tokenIds: bindings.tokenIds,
      shards: bindings.embedding,
      output: bindings.output,
    });
    this.requireLive(" while compiling");
    return Object.freeze({
      label,
      tokenCount,
      primitiveCount: bindings.embedding.length,
      encode(pass: GPUComputePassEncoder): void {
        dispatch.encode(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.embedding.destroy();
    this.qwen.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE text-encoder runtime was destroyed${suffix}`);
    }
  }
}

export function planAceTextEncoder(
  modelProfile: AceModelProfileId,
  shape: AceTextEncoderShape,
): AceQwen3ModelPlan {
  if (shape.tokens > ACE_TEXT_ENCODER_MAX_TEXT_TOKENS) {
    throw new RangeError(
      `ACE product text encoder tokens exceed ${ACE_TEXT_ENCODER_MAX_TEXT_TOKENS}`,
    );
  }
  return planAceQwen3Model(modelProfile, ACE_TEXT_QWEN3_CONFIG, shape);
}

export function planAceQwen3Model(
  modelProfile: AceModelProfileId,
  config: AceQwen3Config,
  shape: AceTextEncoderShape,
): AceQwen3ModelPlan {
  validateAceQwen3Config(config);
  if (shape.batch !== 1) {
    throw new RangeError("ACE text encoder production composer requires batch=1");
  }
  const block = planAceQwen3Block(config, {
    batch: 1,
    tokens: shape.tokens,
    attention: { kind: "uncached" },
  });
  const blockScratchElements = checkedSum([
    // Hidden-width scratch: normalized input, projected attention, both
    // residual intermediates, and projected MLP.
    5 * block.hiddenElements,
    // Q-width scratch: flat/split/normalized/rotated Q, attention, merged.
    6 * block.queryElements,
    // KV-width scratch: flat K/V, split K/V, normalized/rotated K.
    6 * block.keyValueElements,
    // Gate, up, and activated SwiGLU tensors.
    3 * block.intermediateElements,
  ], "ACE Qwen3 block scratch elements");
  const scratchActivationElements = checkedSum([
    blockScratchElements,
    // Embedding output and two layer ping-pong outputs.
    3 * block.hiddenElements,
  ], "ACE Qwen3 model scratch elements");
  const residentActivationElements = checkedSum([
    scratchActivationElements,
    block.hiddenElements,
  ], "ACE Qwen3 resident activation elements");
  return Object.freeze({
    batch: 1,
    tokens: shape.tokens,
    rows: block.rows,
    layerCount: config.layerCount,
    hiddenElements: block.hiddenElements,
    queryElements: block.queryElements,
    keyValueElements: block.keyValueElements,
    intermediateElements: block.intermediateElements,
    outputElements: block.hiddenElements,
    scratchActivationElements,
    scratchActivationBytes: aceActivationBytes(
      modelProfile,
      scratchActivationElements,
    ),
    residentActivationElements,
    residentActivationBytes: aceActivationBytes(
      modelProfile,
      residentActivationElements,
    ),
  });
}

/**
 * Exact batch-one controls produced by Transformers 4.57.6 when ACE passes the
 * unsupported `lyric_attention_mask=None` keyword to `Qwen3Model`.
 */
export function createAceTextEncoderControlData(
  tokens: number,
  config: AceQwen3Config = ACE_TEXT_QWEN3_CONFIG,
): AceTextEncoderControlData {
  validateAceQwen3Config(config);
  requirePositiveSafeInteger(tokens, "ACE text encoder tokens");
  if (tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE text encoder tokens exceed maximumPositionEmbeddings");
  }
  const queryPositions = new Uint32Array(tokens);
  const keyValidity = new Uint32Array(tokens);
  const positionIds: number[] = [];
  for (let token = 0; token < tokens; token += 1) {
    queryPositions[token] = token;
    keyValidity[token] = 1;
    positionIds.push(token);
  }
  const rope = createAceQwen3RopeTables(positionIds, {
    batch: 1,
    tokens,
    headDimension: config.headDimension,
    ropeTheta: config.ropeTheta,
    maximumPositionEmbeddings: config.maximumPositionEmbeddings,
  });
  return Object.freeze({
    validLengths: new Uint32Array([tokens, tokens]),
    queryPositions,
    keyValidity,
    cosine: rope.cosine,
    sine: rope.sine,
  });
}

/** Resolve and shape-check all production text weights without coalescing rows. */
export function resolveAceTextEncoderWeights(
  resolver: AceTextEncoderTensorResolver,
  modelProfile: AceModelProfileId,
): AceTextEncoderWeights {
  const embedding = resolver.logicalTensor(
    ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.embedding,
  );
  requireLogicalShape(
    embedding,
    [ACE_TEXT_QWEN3_CONFIG.vocabularySize, ACE_TEXT_QWEN3_CONFIG.hiddenSize],
  );
  if (embedding.parts.length !== ACE_TEXT_EMBEDDING_ROW_PARTS.length) {
    throw new Error(
      `ACE text embedding has ${embedding.parts.length} parts; ` +
        `${ACE_TEXT_EMBEDDING_ROW_PARTS.length} required`,
    );
  }
  const embeddingBindings = embedding.parts.map((part, index) => {
    const expected = ACE_TEXT_EMBEDDING_ROW_PARTS[index]!;
    requireProfileTensor(part.tensor, modelProfile, true);
    if (
      part.tensor.partStart !== expected.firstRow ||
      part.tensor.partEnd - part.tensor.partStart !== expected.rowCount
    ) {
      throw new Error(`ACE text embedding row shard ${index} changed`);
    }
    return Object.freeze({
      firstRow: part.tensor.partStart,
      rowCount: part.tensor.partEnd - part.tensor.partStart,
      weight: part.binding,
    });
  });
  const layers = ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES.map((names) =>
    Object.freeze({
      inputLayerNorm: resolveUnsharded(
        resolver,
        names.inputLayerNorm,
        [ACE_TEXT_QWEN3_CONFIG.hiddenSize],
        modelProfile,
      ),
      queryProjection: resolveUnsharded(
        resolver,
        names.queryProjection,
        [
          ACE_TEXT_QWEN3_CONFIG.queryHeads * ACE_TEXT_QWEN3_CONFIG.headDimension,
          ACE_TEXT_QWEN3_CONFIG.hiddenSize,
        ],
        modelProfile,
      ),
      keyProjection: resolveUnsharded(
        resolver,
        names.keyProjection,
        [
          ACE_TEXT_QWEN3_CONFIG.keyValueHeads * ACE_TEXT_QWEN3_CONFIG.headDimension,
          ACE_TEXT_QWEN3_CONFIG.hiddenSize,
        ],
        modelProfile,
      ),
      valueProjection: resolveUnsharded(
        resolver,
        names.valueProjection,
        [
          ACE_TEXT_QWEN3_CONFIG.keyValueHeads * ACE_TEXT_QWEN3_CONFIG.headDimension,
          ACE_TEXT_QWEN3_CONFIG.hiddenSize,
        ],
        modelProfile,
      ),
      queryNorm: resolveUnsharded(
        resolver,
        names.queryNorm,
        [ACE_TEXT_QWEN3_CONFIG.headDimension],
        modelProfile,
      ),
      keyNorm: resolveUnsharded(
        resolver,
        names.keyNorm,
        [ACE_TEXT_QWEN3_CONFIG.headDimension],
        modelProfile,
      ),
      outputProjection: resolveUnsharded(
        resolver,
        names.outputProjection,
        [
          ACE_TEXT_QWEN3_CONFIG.hiddenSize,
          ACE_TEXT_QWEN3_CONFIG.queryHeads * ACE_TEXT_QWEN3_CONFIG.headDimension,
        ],
        modelProfile,
      ),
      postAttentionLayerNorm: resolveUnsharded(
        resolver,
        names.postAttentionLayerNorm,
        [ACE_TEXT_QWEN3_CONFIG.hiddenSize],
        modelProfile,
      ),
      gateProjection: resolveUnsharded(
        resolver,
        names.gateProjection,
        [ACE_TEXT_QWEN3_CONFIG.intermediateSize, ACE_TEXT_QWEN3_CONFIG.hiddenSize],
        modelProfile,
      ),
      upProjection: resolveUnsharded(
        resolver,
        names.upProjection,
        [ACE_TEXT_QWEN3_CONFIG.intermediateSize, ACE_TEXT_QWEN3_CONFIG.hiddenSize],
        modelProfile,
      ),
      downProjection: resolveUnsharded(
        resolver,
        names.downProjection,
        [ACE_TEXT_QWEN3_CONFIG.hiddenSize, ACE_TEXT_QWEN3_CONFIG.intermediateSize],
        modelProfile,
      ),
    })
  );
  return Object.freeze({
    embedding: Object.freeze(embeddingBindings),
    layers: Object.freeze(layers),
    finalNorm: resolveUnsharded(
      resolver,
      ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.finalNorm,
      [ACE_TEXT_QWEN3_CONFIG.hiddenSize],
      modelProfile,
    ),
  });
}

/** Fail closed if the generated package contains a missing or hidden text tensor. */
export function validateAceTextEncoderManifestInventory(
  manifest: AcePackageManifest,
): void {
  const profile: AceModelProfileId = manifest.profile === "reference"
    ? "reference-bf16"
    : "raw-fp16";
  const physical = Object.entries(manifest.tensors)
    .filter(([name, tensor]) =>
      tensor.phase === "text" ||
      name.startsWith("text.") ||
      tensor.logicalTensor.startsWith("text.")
    )
    .map(([name]) => name)
    .sort();
  const expectedPhysical = [...ACE_TEXT_ENCODER_PHYSICAL_TENSOR_NAMES].sort();
  requireExactInventory(physical, expectedPhysical, `${manifest.profile} physical text`);

  const logical = [...new Set(Object.values(manifest.tensors)
    .filter((tensor) =>
      tensor.phase === "text" || tensor.logicalTensor.startsWith("text.")
    )
    .map((tensor) => tensor.logicalTensor))].sort();
  const expectedLogical = [...ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES].sort();
  requireExactInventory(logical, expectedLogical, `${manifest.profile} logical text`);

  for (const name of ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES) {
    const parts = Object.values(manifest.tensors)
      .filter((tensor) => tensor.logicalTensor === name)
      .sort((left, right) => left.partStart - right.partStart);
    if (parts.length === 0) throw new Error(`ACE text tensor ${name} is absent`);
    for (const part of parts) requireProfileTensor(
      part,
      profile,
      name === ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES.embedding,
    );
  }
}

function resolveUnsharded(
  resolver: AceTextEncoderTensorResolver,
  name: string,
  expectedShape: readonly number[],
  profile: AceModelProfileId,
): GPUBufferBinding {
  const logical = resolver.logicalTensor(name);
  requireLogicalShape(logical, expectedShape);
  if (logical.parts.length !== 1) {
    throw new Error(`ACE text tensor ${name} unexpectedly has ${logical.parts.length} parts`);
  }
  requireProfileTensor(logical.parts[0]!.tensor, profile, false);
  return resolver.binding(name);
}

function requireLogicalShape(
  logical: AceGpuLogicalTensor,
  expected: readonly number[],
): void {
  if (
    logical.logicalShape.length !== expected.length ||
    logical.logicalShape.some((value, index) => value !== expected[index])
  ) {
    throw new Error(
      `ACE text tensor ${logical.logicalTensor} has shape ` +
        `[${logical.logicalShape.join(",")}], expected [${expected.join(",")}]`,
    );
  }
}

function requireProfileTensor(
  tensor: AcePackageTensorRecord,
  profile: AceModelProfileId,
  rowSharded: boolean,
): void {
  const expectedDtype = profile === "reference-bf16"
    ? "uint32-bf16-pairs"
    : "float16";
  const expectedLayout = rowSharded
    ? profile === "reference-bf16"
      ? "row-shard-axis0-bf16-pairs-lsb-u32"
      : "row-shard-axis0"
    : profile === "reference-bf16"
      ? "source-row-major-bf16-pairs-lsb-u32"
      : "source-row-major";
  if (
    tensor.phase !== "text" ||
    tensor.lifetime !== "text" ||
    tensor.dtype !== expectedDtype ||
    tensor.layout !== expectedLayout
  ) {
    throw new Error(`ACE text tensor ${tensor.logicalTensor} violates ${profile} storage`);
  }
}

function requireExactInventory(
  actual: readonly string[],
  expected: readonly string[],
  label: string,
): void {
  if (
    actual.length !== expected.length ||
    actual.some((value, index) => value !== expected[index])
  ) {
    const expectedSet = new Set(expected);
    const actualSet = new Set(actual);
    const missing = expected.filter((name) => !actualSet.has(name));
    const hidden = actual.filter((name) => !expectedSet.has(name));
    throw new Error(
      `ACE ${label} inventory changed; missing=[${missing.join(",")}], ` +
        `hidden=[${hidden.join(",")}]`,
    );
  }
}

function checkedSum(values: readonly number[], label: string): number {
  let result = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${label} contains an invalid value`);
    }
    result += value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}

function padRow(row: number): string {
  return row.toString().padStart(6, "0");
}
