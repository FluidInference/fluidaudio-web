import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessAttentionKernel,
  type AceAttentionDispatch,
} from "./kernels/attention.js";
import {
  aceCompositeCooperativeQuanta,
  AceCorrectnessGemmKernel,
  type AceGpuEncodeQuantum,
  type AceGemmDispatch,
} from "./kernels/gemm.js";
import {
  AceCorrectnessRmsNormKernel,
  type AceRmsNormDispatch,
} from "./kernels/rmsnorm.js";
import {
  AceCorrectnessRopeKernel,
  type AceRopeDispatch,
} from "./kernels/rope.js";
import {
  AceCorrectnessTransformerPlumbingKernel,
  type AceTransformerDispatch,
} from "./kernels/transformer-plumbing.js";
import {
  createAceQwen3RopeTables,
  type AceQwen3RopeTables,
} from "./qwen3.js";

const MAX_U32 = 0xffff_ffff;

export type AceEncoderAttentionMode = "full" | "sliding";

/**
 * Shared architecture of the pinned ACE lyric, timbre, and audio-token
 * detokenizer encoders. These are bidirectional ACE blocks, not causal Qwen
 * decoder blocks, despite using the same Qwen3 RMSNorm/RoPE/MLP equations.
 */
export interface AceEncoderConfig {
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly queryHeads: number;
  readonly keyValueHeads: number;
  readonly headDimension: number;
  readonly maximumPositionEmbeddings: number;
  readonly ropeTheta: number;
  readonly rmsNormEpsilon: number;
  readonly slidingRadius: number;
  readonly attentionBias: false;
  readonly hiddenActivation: "silu";
}

export interface AceEncoderBlockShape {
  readonly batch: number;
  readonly tokens: number;
  readonly attentionMode: AceEncoderAttentionMode;
}

export interface AceEncoderBlockPlan extends AceEncoderBlockShape {
  readonly rows: number;
  readonly queryWidth: number;
  readonly keyValueWidth: number;
  readonly hiddenElements: number;
  readonly queryElements: number;
  readonly keyValueElements: number;
  readonly intermediateElements: number;
}

export interface AceEncoderBlockWeights {
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

/** Explicit Stage 1 scratch; no field may overlap another field. */
export interface AceEncoderBlockScratch {
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

export interface AceEncoderBlockBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly weights: AceEncoderBlockWeights;
  readonly scratch: AceEncoderBlockScratch;
  /** U32 `[batch,2]`: full query length, then prefix-valid key length. */
  readonly validLengths: GPUBufferBinding;
  /** FP32 `[tokens,headDimension]`, using Qwen split-half RoPE. */
  readonly cosine: GPUBufferBinding;
  readonly sine: GPUBufferBinding;
}

type AceEncoderPrimitiveDispatch =
  | AceGemmDispatch
  | AceRmsNormDispatch
  | AceRopeDispatch
  | AceAttentionDispatch
  | AceTransformerDispatch<unknown>;

export interface AceEncoderBlockDispatch {
  readonly label: string;
  readonly plan: AceEncoderBlockPlan;
  readonly primitiveCount: number;
  readonly cooperativeQuanta: readonly AceGpuEncodeQuantum[];
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceEncoderControlData {
  /** U32 `[batch,2]`: all queries execute; only the valid key prefix attends. */
  readonly validLengths: Uint32Array;
  readonly validKeyLengths: Uint32Array;
}

/**
 * Correctness-first composition of one pinned ACE bidirectional encoder layer.
 * It owns pipelines only; model weights, activations, controls, and scratch are
 * supplied by the FIFO graph owner.
 */
export class AceCorrectnessEncoderRuntime {
  readonly modelProfile: AceModelProfileId;

  private readonly gemm: AceCorrectnessGemmKernel;
  private readonly rmsNorm: AceCorrectnessRmsNormKernel;
  private readonly rope: AceCorrectnessRopeKernel;
  private readonly attention: AceCorrectnessAttentionKernel;
  private readonly plumbing: AceCorrectnessTransformerPlumbingKernel;
  private destroyed = false;

  private constructor(
    modelProfile: AceModelProfileId,
    gemm: AceCorrectnessGemmKernel,
    rmsNorm: AceCorrectnessRmsNormKernel,
    rope: AceCorrectnessRopeKernel,
    attention: AceCorrectnessAttentionKernel,
    plumbing: AceCorrectnessTransformerPlumbingKernel,
  ) {
    this.modelProfile = modelProfile;
    this.gemm = gemm;
    this.rmsNorm = rmsNorm;
    this.rope = rope;
    this.attention = attention;
    this.plumbing = plumbing;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessEncoderRuntime {
    let gemm: AceCorrectnessGemmKernel | undefined;
    let rmsNorm: AceCorrectnessRmsNormKernel | undefined;
    let rope: AceCorrectnessRopeKernel | undefined;
    let attention: AceCorrectnessAttentionKernel | undefined;
    let plumbing: AceCorrectnessTransformerPlumbingKernel | undefined;
    try {
      gemm = AceCorrectnessGemmKernel.create(device, modelProfile);
      rmsNorm = AceCorrectnessRmsNormKernel.create(device, modelProfile);
      rope = AceCorrectnessRopeKernel.create(device, modelProfile);
      attention = AceCorrectnessAttentionKernel.create(device, modelProfile);
      plumbing = AceCorrectnessTransformerPlumbingKernel.create(
        device,
        modelProfile,
      );
      return new AceCorrectnessEncoderRuntime(
        modelProfile,
        gemm,
        rmsNorm,
        rope,
        attention,
        plumbing,
      );
    } catch (error) {
      gemm?.destroy();
      rmsNorm?.destroy();
      rope?.destroy();
      attention?.destroy();
      plumbing?.destroy();
      throw error;
    }
  }

  async createBlockDispatch(
    label: string,
    config: AceEncoderConfig,
    shape: AceEncoderBlockShape,
    bindings: AceEncoderBlockBindings,
  ): Promise<AceEncoderBlockDispatch> {
    this.requireLive();
    const plan = planAceEncoderBlock(config, shape);
    validateAceEncoderBlockBindingAliases(label, bindings);

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

    const dispatches = await Promise.all<AceEncoderPrimitiveDispatch>([
      this.rmsNorm.createDispatch(`${label}-input-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: bindings.input,
        weight: w.inputLayerNorm,
        output: s.normalizedInput,
      }),
      this.gemm.createDispatch(`${label}-query-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.queryWidth,
      }, {
        activation: s.normalizedInput,
        weight: w.queryProjection,
        output: s.queryFlat,
      }),
      this.gemm.createDispatch(`${label}-key-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: s.normalizedInput,
        weight: w.keyProjection,
        output: s.keyFlat,
      }),
      this.gemm.createDispatch(`${label}-value-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: s.normalizedInput,
        weight: w.valueProjection,
        output: s.valueFlat,
      }),
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
      this.rope.createDispatch(`${label}-query-rope`, queryHeadShape, {
        input: s.normalizedQueryHeads,
        cosine: bindings.cosine,
        sine: bindings.sine,
        output: s.rotatedQueryHeads,
      }),
      this.rope.createDispatch(`${label}-key-rope`, keyValueHeadShape, {
        input: s.normalizedKeyHeads,
        cosine: bindings.cosine,
        sine: bindings.sine,
        output: s.rotatedKeyHeads,
      }),
      this.attention.createDispatch(`${label}-${plan.attentionMode}-attention`, {
        batch: plan.batch,
        queryHeads: config.queryHeads,
        keyValueHeads: config.keyValueHeads,
        queryTokens: plan.tokens,
        keyValueTokens: plan.tokens,
        headDimension: config.headDimension,
        mode: plan.attentionMode,
        ...(plan.attentionMode === "sliding"
          ? { slidingRadius: config.slidingRadius }
          : {}),
      }, {
        query: s.rotatedQueryHeads,
        key: s.rotatedKeyHeads,
        value: s.valueHeads,
        validLengths: bindings.validLengths,
        output: s.attentionHeads,
      }),
      this.plumbing.createHeadTransformDispatch(
        `${label}-merge-attention-heads`,
        "merge-heads",
        queryHeadShape,
        { input: s.attentionHeads, output: s.mergedAttention },
      ),
      this.gemm.createDispatch(`${label}-attention-output-projection`, {
        rows: plan.rows,
        inner: plan.queryWidth,
        columns: config.hiddenSize,
      }, {
        activation: s.mergedAttention,
        weight: w.outputProjection,
        output: s.projectedAttention,
      }),
      this.plumbing.createResidualAddDispatch(
        `${label}-attention-residual`,
        hiddenShape,
        {
          left: bindings.input,
          right: s.projectedAttention,
          output: s.afterAttention,
        },
      ),
      this.rmsNorm.createDispatch(`${label}-post-attention-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.afterAttention,
        weight: w.postAttentionLayerNorm,
        output: s.normalizedAfterAttention,
      }),
      this.gemm.createDispatch(`${label}-gate-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }, {
        activation: s.normalizedAfterAttention,
        weight: w.gateProjection,
        output: s.gate,
      }),
      this.gemm.createDispatch(`${label}-up-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }, {
        activation: s.normalizedAfterAttention,
        weight: w.upProjection,
        output: s.up,
      }),
      this.plumbing.createSwiGluDispatch(`${label}-swiglu`, intermediateShape, {
        gate: s.gate,
        up: s.up,
        output: s.gatedActivation,
      }),
      this.gemm.createDispatch(`${label}-down-projection`, {
        rows: plan.rows,
        inner: config.intermediateSize,
        columns: config.hiddenSize,
      }, {
        activation: s.gatedActivation,
        weight: w.downProjection,
        output: s.projectedMlp,
      }),
      this.plumbing.createResidualAddDispatch(`${label}-mlp-residual`, hiddenShape, {
        left: s.afterAttention,
        right: s.projectedMlp,
        output: bindings.output,
      }),
    ]);
    this.requireLive(" while compiling");
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
    config: AceEncoderConfig,
    rows: number,
    bindings: Readonly<{
      readonly input: GPUBufferBinding;
      readonly weight: GPUBufferBinding;
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceRmsNormDispatch> {
    this.requireLive();
    validateAceEncoderConfig(config);
    requirePositiveInteger(rows, `${label} rows`);
    return this.rmsNorm.createDispatch(label, {
      rows,
      width: config.hiddenSize,
      epsilon: config.rmsNormEpsilon,
    }, bindings);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.gemm.destroy();
    this.rmsNorm.destroy();
    this.rope.destroy();
    this.attention.destroy();
    this.plumbing.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE encoder runtime was destroyed${suffix}`);
    }
  }
}

export function planAceEncoderBlock(
  config: AceEncoderConfig,
  shape: AceEncoderBlockShape,
): AceEncoderBlockPlan {
  validateAceEncoderConfig(config);
  requirePositiveInteger(shape.batch, "ACE encoder batch");
  requirePositiveInteger(shape.tokens, "ACE encoder tokens");
  if (shape.tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE encoder tokens exceed maximumPositionEmbeddings");
  }
  if (shape.attentionMode !== "full" && shape.attentionMode !== "sliding") {
    throw new TypeError(
      `Unknown ACE encoder attention mode ${String(shape.attentionMode)}`,
    );
  }
  const rows = checkedProduct([shape.batch, shape.tokens], "ACE encoder rows");
  const queryWidth = checkedProduct(
    [config.queryHeads, config.headDimension],
    "ACE encoder query width",
  );
  const keyValueWidth = checkedProduct(
    [config.keyValueHeads, config.headDimension],
    "ACE encoder KV width",
  );
  const hiddenElements = checkedProduct(
    [rows, config.hiddenSize],
    "ACE encoder hidden elements",
  );
  const queryElements = checkedProduct(
    [rows, queryWidth],
    "ACE encoder query elements",
  );
  const keyValueElements = checkedProduct(
    [rows, keyValueWidth],
    "ACE encoder KV elements",
  );
  const intermediateElements = checkedProduct(
    [rows, config.intermediateSize],
    "ACE encoder intermediate elements",
  );
  for (const [name, value] of Object.entries({
    hiddenElements,
    queryElements,
    keyValueElements,
    intermediateElements,
  })) {
    if (value > MAX_U32) {
      throw new RangeError(`ACE encoder ${name} exceeds U32 shader indexing`);
    }
  }
  return Object.freeze({
    ...shape,
    rows,
    queryWidth,
    keyValueWidth,
    hiddenElements,
    queryElements,
    keyValueElements,
    intermediateElements,
  });
}

export function validateAceEncoderConfig(config: AceEncoderConfig): void {
  for (const [name, value] of Object.entries({
    hiddenSize: config.hiddenSize,
    intermediateSize: config.intermediateSize,
    queryHeads: config.queryHeads,
    keyValueHeads: config.keyValueHeads,
    headDimension: config.headDimension,
    maximumPositionEmbeddings: config.maximumPositionEmbeddings,
    slidingRadius: config.slidingRadius,
  })) {
    requirePositiveInteger(value, `ACE encoder ${name}`);
    if (value > MAX_U32) throw new RangeError(`ACE encoder ${name} exceeds U32`);
  }
  if (config.queryHeads % config.keyValueHeads !== 0) {
    throw new RangeError("ACE encoder query heads must be divisible by KV heads");
  }
  if (config.headDimension % 2 !== 0 || config.headDimension > 128) {
    throw new RangeError("ACE encoder headDimension must be even and at most 128");
  }
  if (config.queryHeads * config.headDimension !== config.hiddenSize) {
    throw new RangeError("ACE encoder query width must equal hiddenSize");
  }
  if (!Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0) {
    throw new RangeError("ACE encoder ropeTheta must be positive and finite");
  }
  if (!Number.isFinite(config.rmsNormEpsilon) || config.rmsNormEpsilon <= 0) {
    throw new RangeError("ACE encoder rmsNormEpsilon must be positive and finite");
  }
  if (config.attentionBias !== false) {
    throw new TypeError("ACE encoder correctness runtime does not permit attention bias");
  }
  if (config.hiddenActivation !== "silu") {
    throw new TypeError("ACE encoder correctness runtime requires SiLU/SwiGLU");
  }
}

/**
 * Converts right-padded binary masks to the attention kernel's compact prefix
 * controls. Every query still executes, matching upstream's key-only padding
 * mask; only padded keys are excluded.
 */
export function createAceEncoderControlData(
  mask: readonly number[],
  batch: number,
  tokens: number,
): AceEncoderControlData {
  requirePositiveInteger(batch, "ACE encoder control batch");
  requirePositiveInteger(tokens, "ACE encoder control tokens");
  const expected = checkedProduct([batch, tokens], "ACE encoder control mask");
  if (mask.length !== expected) {
    throw new RangeError(
      `ACE encoder mask has ${mask.length} entries; ${expected} required`,
    );
  }
  const validLengths = new Uint32Array(batch * 2);
  const validKeyLengths = new Uint32Array(batch);
  for (let row = 0; row < batch; row += 1) {
    let valid = 0;
    let paddingSeen = false;
    for (let token = 0; token < tokens; token += 1) {
      const value = mask[row * tokens + token];
      if (value !== 0 && value !== 1) {
        throw new RangeError(
          `ACE encoder mask[${row * tokens + token}] must be zero or one`,
        );
      }
      if (value === 1) {
        if (paddingSeen) {
          throw new RangeError("ACE encoder masks must be right-padded prefixes");
        }
        valid += 1;
      } else {
        paddingSeen = true;
      }
    }
    if (valid === 0) {
      throw new RangeError(`ACE encoder mask row ${row} has no valid token`);
    }
    // Upstream applies the 2D mask to keys only. Padded queries still flow
    // through every layer before the final sequence pack discards them.
    validLengths[row * 2] = tokens;
    validLengths[row * 2 + 1] = valid;
    validKeyLengths[row] = valid;
  }
  return Object.freeze({ validLengths, validKeyLengths });
}

export function createAceEncoderFullControlData(
  batch: number,
  tokens: number,
): AceEncoderControlData {
  requirePositiveInteger(batch, "ACE encoder full-control batch");
  requirePositiveInteger(tokens, "ACE encoder full-control tokens");
  const mask = new Uint8Array(batch * tokens);
  mask.fill(1);
  return createAceEncoderControlData(Array.from(mask), batch, tokens);
}

export function createAceEncoderRopeTables(
  tokens: number,
  config: AceEncoderConfig,
): AceQwen3RopeTables {
  validateAceEncoderConfig(config);
  requirePositiveInteger(tokens, "ACE encoder RoPE tokens");
  if (tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE encoder RoPE tokens exceed maximumPositionEmbeddings");
  }
  return createAceQwen3RopeTables(
    Array.from({ length: tokens }, (_, index) => index),
    { ...config, batch: 1, tokens },
  );
}

export function aceEncoderLayerAttentionMode(
  layerIndex: number,
): AceEncoderAttentionMode {
  if (!Number.isSafeInteger(layerIndex) || layerIndex < 0) {
    throw new RangeError("ACE encoder layer index must be a non-negative integer");
  }
  return layerIndex % 2 === 0 ? "sliding" : "full";
}

/**
 * Rejects graph aliases that primitive-local checks cannot see. Every block
 * write is disjoint from every other live write and from all model/control
 * reads. The graph owner may reuse these ranges only between whole blocks.
 */
export function validateAceEncoderBlockBindingAliases(
  label: string,
  bindings: AceEncoderBlockBindings,
): void {
  const writable = [
    { name: "output", binding: bindings.output },
    ...Object.entries(bindings.scratch).map(([name, binding]) => ({
      name: `scratch ${name}`,
      binding,
    })),
  ];
  const readonly = [
    { name: "input", binding: bindings.input },
    ...Object.entries(bindings.weights).map(([name, binding]) => ({
      name: `weight ${name}`,
      binding,
    })),
    { name: "valid lengths", binding: bindings.validLengths },
    { name: "cosine", binding: bindings.cosine },
    { name: "sine", binding: bindings.sine },
  ];
  requireDisjointWritableBindings(label, writable, readonly);
}

function requireDisjointWritableBindings(
  label: string,
  writable: readonly Readonly<{
    name: string;
    binding: GPUBufferBinding;
  }>[],
  readonlyBindings: readonly Readonly<{
    name: string;
    binding: GPUBufferBinding;
  }>[],
): void {
  for (let leftIndex = 0; leftIndex < writable.length; leftIndex += 1) {
    const left = writable[leftIndex]!;
    const leftRange = bindingRange(left.binding, `${label} ${left.name}`);
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < writable.length;
      rightIndex += 1
    ) {
      const right = writable[rightIndex]!;
      if (left.binding.buffer !== right.binding.buffer) continue;
      const rightRange = bindingRange(
        right.binding,
        `${label} ${right.name}`,
      );
      if (leftRange.start < rightRange.end && rightRange.start < leftRange.end) {
        throw new RangeError(
          `${label} ${left.name} overlaps writable ${right.name}`,
        );
      }
    }
    for (const right of readonlyBindings) {
      if (left.binding.buffer !== right.binding.buffer) continue;
      const rightRange = bindingRange(
        right.binding,
        `${label} ${right.name}`,
      );
      if (leftRange.start < rightRange.end && rightRange.start < leftRange.end) {
        throw new RangeError(
          `${label} ${left.name} overlaps read-only ${right.name}`,
        );
      }
    }
  }
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

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
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
