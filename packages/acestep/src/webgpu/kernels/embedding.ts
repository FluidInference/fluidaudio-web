import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  acePackedWeightBytes,
  checkedAceProduct,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
  requireAceU32,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

export interface AceEmbeddingShape {
  readonly tokenCount: number;
  readonly width: number;
  readonly vocabularySize: number;
}

export interface AceEmbeddingShardShape {
  readonly firstRow: number;
  readonly rowCount: number;
}

export interface AceEmbeddingShardBinding extends AceEmbeddingShardShape {
  /** Row-major shard-local embedding weights. */
  readonly weight: GPUBufferBinding;
}

export interface AceEmbeddingPlan extends AceEmbeddingShape {
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly shards: readonly AceEmbeddingShardShape[];
}

export interface AceEmbeddingBindings {
  /** Unsigned token IDs. The caller must keep every ID below vocabularySize. */
  readonly tokenIds: GPUBufferBinding;
  readonly shards: readonly AceEmbeddingShardBinding[];
  readonly output: GPUBufferBinding;
}

export interface AceEmbeddingDispatch {
  readonly label: string;
  readonly plan: AceEmbeddingPlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledEmbeddingShard {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
}

/**
 * Shard-aware Qwen embedding gather without ever assembling the vocabulary.
 *
 * Each physical row shard is dispatched in vocabulary order. Exactly one
 * shard writes each in-range token. Packed BF16 is decoded directly to the
 * FP32 reference activation; raw FP16 remains FP16.
 */
export class AceCorrectnessEmbeddingKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessEmbeddingKernel {
    requireAceModelProfile(device, modelProfile, "embedding");
    return new AceCorrectnessEmbeddingKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceEmbeddingShape,
    bindings: AceEmbeddingBindings,
  ): Promise<AceEmbeddingDispatch> {
    this.requireLive();
    const plan = planAceEmbedding(shape, bindings.shards);
    requireAceBindingBytes(
      bindings.tokenIds,
      plan.tokenCount * Uint32Array.BYTES_PER_ELEMENT,
      `${label} token IDs`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.tokenIds, ...bindings.shards.map((shard) => shard.weight)],
      label,
    );

    const compiled = await Promise.all(
      bindings.shards.map(async (shard): Promise<CompiledEmbeddingShard> => {
        const weightElements = checkedAceProduct(
          [shard.rowCount, plan.width],
          "ACE embedding shard",
        );
        requireAceU32(weightElements, "ACE embedding shard elements");
        requireAceBindingBytes(
          shard.weight,
          acePackedWeightBytes(this.modelProfile, weightElements),
          `${label} shard ${shard.firstRow}`,
        );
        const key = `${shapeKey(plan)}:${shard.firstRow}:${shard.rowCount}`;
        const pipeline = await this.pipelineFor(
          key,
          `${label}-rows-${shard.firstRow}-${shard.firstRow + shard.rowCount}`,
          aceCorrectnessEmbeddingWgsl(this.modelProfile, plan, shard),
        );
        const bindGroup = this.device.createBindGroup({
          label: `${label}-rows-${shard.firstRow}-bindings`,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: bindings.tokenIds },
            { binding: 1, resource: shard.weight },
            { binding: 2, resource: bindings.output },
          ],
        });
        return Object.freeze({ pipeline, bindGroup });
      }),
    );
    this.requireLive(" while compiling");

    return Object.freeze({
      label,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        for (const shard of compiled) {
          pass.setPipeline(shard.pipeline);
          pass.setBindGroup(0, shard.bindGroup);
          pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY, 1);
        }
      },
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(
    key: string,
    label: string,
    source: string,
  ): Promise<GPUComputePipeline> {
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const module = this.device.createShaderModule({ label, code: source });
    const created = this.device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE embedding kernel was destroyed${suffix}`);
    }
  }
}

export function planAceEmbedding(
  shape: AceEmbeddingShape,
  shards: readonly AceEmbeddingShardShape[],
): AceEmbeddingPlan {
  requirePositiveSafeInteger(shape.tokenCount, "ACE embedding tokenCount");
  requirePositiveSafeInteger(shape.width, "ACE embedding width");
  requirePositiveSafeInteger(
    shape.vocabularySize,
    "ACE embedding vocabularySize",
  );
  requireAceU32(shape.vocabularySize, "ACE embedding vocabulary");
  if (shards.length === 0) {
    throw new RangeError("ACE embedding requires at least one row shard");
  }
  let expectedFirstRow = 0;
  const frozenShards = shards.map((shard, index) => {
    if (!Number.isSafeInteger(shard.firstRow) || shard.firstRow < 0) {
      throw new RangeError(`ACE embedding shard ${index} firstRow is invalid`);
    }
    requirePositiveSafeInteger(
      shard.rowCount,
      `ACE embedding shard ${index} rowCount`,
    );
    if (shard.firstRow !== expectedFirstRow) {
      throw new RangeError(
        `ACE embedding shard ${index} must start at row ${expectedFirstRow}`,
      );
    }
    expectedFirstRow += shard.rowCount;
    requireAceU32(expectedFirstRow, "ACE embedding shard row coverage");
    return Object.freeze({
      firstRow: shard.firstRow,
      rowCount: shard.rowCount,
    });
  });
  if (expectedFirstRow !== shape.vocabularySize) {
    throw new RangeError(
      `ACE embedding shards cover ${expectedFirstRow} rows, expected ${shape.vocabularySize}`,
    );
  }
  const outputElements = checkedAceProduct(
    [shape.tokenCount, shape.width],
    "ACE embedding output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE embedding");
  return Object.freeze({
    ...shape,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
    shards: Object.freeze(frozenShards),
  });
}

export function aceCorrectnessEmbeddingWgsl(
  modelProfile: AceModelProfileId,
  shape: AceEmbeddingShape,
  shard: AceEmbeddingShardShape,
): string {
  const plan = planAceEmbedding(shape, [
    { firstRow: 0, rowCount: shape.vocabularySize },
  ]);
  if (
    !Number.isSafeInteger(shard.firstRow) ||
    shard.firstRow < 0 ||
    !Number.isSafeInteger(shard.rowCount) ||
    shard.rowCount <= 0 ||
    shard.firstRow + shard.rowCount > shape.vocabularySize
  ) {
    throw new RangeError("ACE embedding shader shard is outside the vocabulary");
  }
  requireAceU32(
    checkedAceProduct(
      [shard.rowCount, shape.width],
      "ACE embedding shader shard",
    ),
    "ACE embedding shader shard elements",
  );
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(
      `Unknown ACE embedding model profile ${String(modelProfile)}`,
    );
  }
  const loader = f16
    ? "let value = weight[local_element];"
    : /* wgsl */ `let pair = weight[local_element >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (local_element & 1u) == 0u);
  let value = bitcast<f32>(bits16 << 16u);`;
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const WIDTH: u32 = ${plan.width}u;
const VOCABULARY_SIZE: u32 = ${plan.vocabularySize}u;
const FIRST_ROW: u32 = ${shard.firstRow}u;
const ROW_COUNT: u32 = ${shard.rowCount}u;

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<${f16 ? "f16" : "u32"}>;
@group(0) @binding(2) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let token_index = index / WIDTH;
  let dimension = index % WIDTH;
  let token_id = token_ids[token_index];
  if (token_id >= VOCABULARY_SIZE) {
    output[index] = ${f16 ? "f16(0.0)" : "0.0"};
    return;
  }
  if (token_id < FIRST_ROW || token_id >= FIRST_ROW + ROW_COUNT) { return; }
  let local_element = (token_id - FIRST_ROW) * WIDTH + dimension;
  ${loader}
  output[index] = value;
}
`;
}

function shapeKey(shape: AceEmbeddingShape): string {
  return `${shape.tokenCount}x${shape.width}x${shape.vocabularySize}`;
}
