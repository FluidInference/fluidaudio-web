import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  checkedAceProduct,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

export const ACE_SILENCE_SOURCE_CHANNELS = 64;
export const ACE_SILENCE_SOURCE_FRAMES = 15_000;
export const ACE_CONTEXT_CHANNELS = 128;

export interface AceSilenceExpandShape {
  readonly batch: number;
  readonly frames: number;
}

export interface AceSilenceExpandPlan extends AceSilenceExpandShape {
  readonly channels: typeof ACE_SILENCE_SOURCE_CHANNELS;
  readonly sourceFrames: typeof ACE_SILENCE_SOURCE_FRAMES;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceSilenceExpandBindings {
  /** Pinned FP32 source layout `[1,64,15000]` (NCT). */
  readonly source: GPUBufferBinding;
  /** Profile activation layout `[batch,frames,64]` (NTC). */
  readonly output: GPUBufferBinding;
}

export interface AceDirectContextShape {
  readonly batch: number;
  readonly frames: number;
}

export interface AceDirectContextPlan extends AceDirectContextShape {
  readonly sourceChannels: typeof ACE_SILENCE_SOURCE_CHANNELS;
  readonly outputChannels: typeof ACE_CONTEXT_CHANNELS;
  readonly sourceElements: number;
  readonly maskElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceDirectContextBindings {
  /** Profile activation layout `[batch,frames,64]`. */
  readonly sourceLatents: GPUBufferBinding;
  /** U32 binary mask `[batch,frames]`. */
  readonly chunkMask: GPUBufferBinding;
  /** Profile activation layout `[batch,frames,128]`. */
  readonly output: GPUBufferBinding;
}

export interface AceSemanticSourceShape {
  readonly batch: number;
  readonly semanticFrames: number;
  readonly outputFrames: number;
}

export interface AceSemanticSourcePlan extends AceSemanticSourceShape {
  readonly channels: typeof ACE_SILENCE_SOURCE_CHANNELS;
  readonly semanticElements: number;
  readonly outputElements: number;
  readonly copiedFrames: number;
  readonly paddedFrames: number;
  readonly truncatedFrames: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceSemanticSourceBindings {
  /** Profile activation layout `[batch,semanticFrames,64]`. */
  readonly semanticHints: GPUBufferBinding;
  /** Pinned FP32 source layout `[1,64,15000]` (NCT). */
  readonly silenceSource: GPUBufferBinding;
  /** Profile activation layout `[batch,outputFrames,64]`. */
  readonly output: GPUBufferBinding;
}

export interface AceConditionLayoutDispatch<Plan> {
  readonly label: string;
  readonly operation: "silence-expand" | "semantic-source" | "direct-context";
  readonly plan: Plan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Exact condition-layout kernels. The silence constant intentionally remains
 * in its authenticated source NCT layout on disk; this kernel slices,
 * transposes, and batch-expands it directly on the GPU.
 */
export class AceCorrectnessConditionLayoutKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessConditionLayoutKernel {
    requireAceModelProfile(device, modelProfile, "condition layout");
    return new AceCorrectnessConditionLayoutKernel(device, modelProfile);
  }

  async createSilenceExpandDispatch(
    label: string,
    shape: AceSilenceExpandShape,
    bindings: AceSilenceExpandBindings,
  ): Promise<AceConditionLayoutDispatch<AceSilenceExpandPlan>> {
    this.requireLive();
    const plan = planAceSilenceExpand(shape);
    requireAceBindingBytes(
      bindings.source,
      ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES *
        Float32Array.BYTES_PER_ELEMENT,
      `${label} source`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(bindings.output, [bindings.source], label);
    return this.createDispatch(
      label,
      "silence-expand",
      plan,
      [bindings.source, bindings.output],
      aceCorrectnessSilenceExpandWgsl(this.modelProfile, plan),
    );
  }

  async createDirectContextDispatch(
    label: string,
    shape: AceDirectContextShape,
    bindings: AceDirectContextBindings,
  ): Promise<AceConditionLayoutDispatch<AceDirectContextPlan>> {
    this.requireLive();
    const plan = planAceDirectContext(shape);
    requireAceBindingBytes(
      bindings.sourceLatents,
      aceActivationBytes(this.modelProfile, plan.sourceElements),
      `${label} source latents`,
    );
    requireAceBindingBytes(
      bindings.chunkMask,
      plan.maskElements * Uint32Array.BYTES_PER_ELEMENT,
      `${label} chunk mask`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.sourceLatents, bindings.chunkMask],
      label,
    );
    return this.createDispatch(
      label,
      "direct-context",
      plan,
      [bindings.sourceLatents, bindings.chunkMask, bindings.output],
      aceCorrectnessDirectContextWgsl(this.modelProfile, plan),
    );
  }

  /**
   * Exact upstream `precomputed_lm_hints_25Hz` source preparation. Semantic
   * frames are cropped from the right or followed by silence starting at
   * silence frame zero (not the absolute destination frame).
   */
  async createSemanticSourceDispatch(
    label: string,
    shape: AceSemanticSourceShape,
    bindings: AceSemanticSourceBindings,
  ): Promise<AceConditionLayoutDispatch<AceSemanticSourcePlan>> {
    this.requireLive();
    const plan = planAceSemanticSource(shape);
    requireAceBindingBytes(
      bindings.semanticHints,
      aceActivationBytes(this.modelProfile, plan.semanticElements),
      `${label} semantic hints`,
    );
    requireAceBindingBytes(
      bindings.silenceSource,
      ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES *
        Float32Array.BYTES_PER_ELEMENT,
      `${label} silence source`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.semanticHints, bindings.silenceSource],
      label,
    );
    return this.createDispatch(
      label,
      "semantic-source",
      plan,
      [bindings.semanticHints, bindings.silenceSource, bindings.output],
      aceCorrectnessSemanticSourceWgsl(this.modelProfile, plan),
    );
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private async createDispatch<Plan extends {
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>(
    label: string,
    operation: "silence-expand" | "semantic-source" | "direct-context",
    plan: Plan,
    bindings: readonly GPUBufferBinding[],
    source: string,
  ): Promise<AceConditionLayoutDispatch<Plan>> {
    const key = `${operation}:${this.modelProfile}:${JSON.stringify(plan)}`;
    const pipeline = await this.pipelineFor(key, label, source);
    this.requireLive(" while compiling");
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((resource, binding) => ({ binding, resource })),
    });
    return Object.freeze({
      label,
      operation,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY, 1);
      },
    });
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
      throw new Error(`ACE condition-layout kernel was destroyed${suffix}`);
    }
  }
}

export function planAceSilenceExpand(
  shape: AceSilenceExpandShape,
): AceSilenceExpandPlan {
  requirePositiveSafeInteger(shape.batch, "ACE silence-expand batch");
  requirePositiveSafeInteger(shape.frames, "ACE silence-expand frames");
  const outputElements = checkedAceProduct(
    [shape.batch, shape.frames, ACE_SILENCE_SOURCE_CHANNELS],
    "ACE silence-expand output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE silence-expand");
  return Object.freeze({
    ...shape,
    channels: ACE_SILENCE_SOURCE_CHANNELS,
    sourceFrames: ACE_SILENCE_SOURCE_FRAMES,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function planAceDirectContext(
  shape: AceDirectContextShape,
): AceDirectContextPlan {
  requirePositiveSafeInteger(shape.batch, "ACE direct-context batch");
  requirePositiveSafeInteger(shape.frames, "ACE direct-context frames");
  const rows = checkedAceProduct(
    [shape.batch, shape.frames],
    "ACE direct-context rows",
  );
  const sourceElements = checkedAceProduct(
    [rows, ACE_SILENCE_SOURCE_CHANNELS],
    "ACE direct-context source",
  );
  const outputElements = checkedAceProduct(
    [rows, ACE_CONTEXT_CHANNELS],
    "ACE direct-context output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE direct-context");
  return Object.freeze({
    ...shape,
    sourceChannels: ACE_SILENCE_SOURCE_CHANNELS,
    outputChannels: ACE_CONTEXT_CHANNELS,
    sourceElements,
    maskElements: rows,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function planAceSemanticSource(
  shape: AceSemanticSourceShape,
): AceSemanticSourcePlan {
  requirePositiveSafeInteger(shape.batch, "ACE semantic-source batch");
  requirePositiveSafeInteger(
    shape.semanticFrames,
    "ACE semantic-source semanticFrames",
  );
  requirePositiveSafeInteger(
    shape.outputFrames,
    "ACE semantic-source outputFrames",
  );
  const semanticElements = checkedAceProduct(
    [shape.batch, shape.semanticFrames, ACE_SILENCE_SOURCE_CHANNELS],
    "ACE semantic-source hints",
  );
  const outputElements = checkedAceProduct(
    [shape.batch, shape.outputFrames, ACE_SILENCE_SOURCE_CHANNELS],
    "ACE semantic-source output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE semantic-source");
  return Object.freeze({
    ...shape,
    channels: ACE_SILENCE_SOURCE_CHANNELS,
    semanticElements,
    outputElements,
    copiedFrames: Math.min(shape.semanticFrames, shape.outputFrames),
    paddedFrames: Math.max(0, shape.outputFrames - shape.semanticFrames),
    truncatedFrames: Math.max(0, shape.semanticFrames - shape.outputFrames),
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function aceCorrectnessSilenceExpandWgsl(
  modelProfile: AceModelProfileId,
  shape: AceSilenceExpandShape,
): string {
  const plan = planAceSilenceExpand(shape);
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(`Unknown ACE condition-layout profile ${String(modelProfile)}`);
  }
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const FRAMES: u32 = ${plan.frames}u;
const SOURCE_FRAMES: u32 = ${ACE_SILENCE_SOURCE_FRAMES}u;
const CHANNELS: u32 = ${ACE_SILENCE_SOURCE_CHANNELS}u;

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let channel = index % CHANNELS;
  let frame = (index / CHANNELS) % FRAMES;
  // The pinned handler tiles silence_latent when the requested duration is
  // longer than its stored tensor, then truncates to the requested length.
  let source_index = channel * SOURCE_FRAMES + (frame % SOURCE_FRAMES);
  output[index] = ${f16 ? "f16(source[source_index])" : "source[source_index]"};
}
`;
}

export function aceCorrectnessDirectContextWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDirectContextShape,
): string {
  const plan = planAceDirectContext(shape);
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(`Unknown ACE condition-layout profile ${String(modelProfile)}`);
  }
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const SOURCE_CHANNELS: u32 = ${ACE_SILENCE_SOURCE_CHANNELS}u;
const OUTPUT_CHANNELS: u32 = ${ACE_CONTEXT_CHANNELS}u;

@group(0) @binding(0) var<storage, read> source_latents: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> chunk_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let channel = index % OUTPUT_CHANNELS;
  let row = index / OUTPUT_CHANNELS;
  if (channel < SOURCE_CHANNELS) {
    output[index] = source_latents[row * SOURCE_CHANNELS + channel];
  } else {
    output[index] = ${f16 ? "f16(f32(chunk_mask[row]))" : "f32(chunk_mask[row])"};
  }
}
`;
}

export function aceCorrectnessSemanticSourceWgsl(
  modelProfile: AceModelProfileId,
  shape: AceSemanticSourceShape,
): string {
  const plan = planAceSemanticSource(shape);
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(`Unknown ACE condition-layout profile ${String(modelProfile)}`);
  }
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const SEMANTIC_FRAMES: u32 = ${plan.semanticFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const SOURCE_FRAMES: u32 = ${ACE_SILENCE_SOURCE_FRAMES}u;
const CHANNELS: u32 = ${ACE_SILENCE_SOURCE_CHANNELS}u;

@group(0) @binding(0) var<storage, read> semantic_hints: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> silence_source: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let channel = index % CHANNELS;
  let row = index / CHANNELS;
  let frame = row % OUTPUT_FRAMES;
  let batch = row / OUTPUT_FRAMES;
  if (frame < SEMANTIC_FRAMES) {
    let semantic_index = (batch * SEMANTIC_FRAMES + frame) * CHANNELS + channel;
    output[index] = semantic_hints[semantic_index];
  } else {
    let padding_frame = (frame - SEMANTIC_FRAMES) % SOURCE_FRAMES;
    let silence_index = channel * SOURCE_FRAMES + padding_frame;
    output[index] = ${f16 ? "f16(silence_source[silence_index])" : "silence_source[silence_index]"};
  }
}
`;
}

export function expandAceSilenceLatentCpu(
  sourceNct: Float32Array,
  batch: number,
  frames: number,
): Float32Array {
  const plan = planAceSilenceExpand({ batch, frames });
  const required = ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES;
  if (sourceNct.length !== required) {
    throw new RangeError(
      `ACE silence source has ${sourceNct.length} elements; ${required} required`,
    );
  }
  const output = new Float32Array(plan.outputElements);
  for (let row = 0; row < batch * frames; row += 1) {
    const frame = row % frames;
    for (let channel = 0; channel < ACE_SILENCE_SOURCE_CHANNELS; channel += 1) {
      output[row * ACE_SILENCE_SOURCE_CHANNELS + channel] =
        sourceNct[
          channel * ACE_SILENCE_SOURCE_FRAMES +
            (frame % ACE_SILENCE_SOURCE_FRAMES)
        ]!;
    }
  }
  return output;
}

export function createAceDirectContextCpu(
  sourceLatents: Float32Array,
  chunkMask: readonly number[],
  batch: number,
  frames: number,
): Float32Array {
  const plan = planAceDirectContext({ batch, frames });
  if (sourceLatents.length !== plan.sourceElements) {
    throw new RangeError("ACE direct-context CPU source shape mismatch");
  }
  if (chunkMask.length !== plan.maskElements) {
    throw new RangeError("ACE direct-context CPU mask shape mismatch");
  }
  const output = new Float32Array(plan.outputElements);
  for (let row = 0; row < plan.maskElements; row += 1) {
    const mask = chunkMask[row];
    if (mask !== 0 && mask !== 1) {
      throw new RangeError(`ACE direct chunk mask[${row}] must be zero or one`);
    }
    const sourceBase = row * ACE_SILENCE_SOURCE_CHANNELS;
    const outputBase = row * ACE_CONTEXT_CHANNELS;
    output.set(
      sourceLatents.subarray(
        sourceBase,
        sourceBase + ACE_SILENCE_SOURCE_CHANNELS,
      ),
      outputBase,
    );
    output.fill(mask, outputBase + ACE_SILENCE_SOURCE_CHANNELS, outputBase + ACE_CONTEXT_CHANNELS);
  }
  return output;
}

export function composeAceSemanticSourceCpu(
  semanticHints: Float32Array,
  silenceSourceNct: Float32Array,
  batch: number,
  semanticFrames: number,
  outputFrames: number,
): Float32Array {
  const plan = planAceSemanticSource({ batch, semanticFrames, outputFrames });
  if (semanticHints.length !== plan.semanticElements) {
    throw new RangeError("ACE semantic-source CPU hint shape mismatch");
  }
  const requiredSilence = ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES;
  if (silenceSourceNct.length !== requiredSilence) {
    throw new RangeError("ACE semantic-source CPU silence shape mismatch");
  }
  const output = new Float32Array(plan.outputElements);
  for (let item = 0; item < batch; item += 1) {
    for (let frame = 0; frame < outputFrames; frame += 1) {
      for (let channel = 0; channel < ACE_SILENCE_SOURCE_CHANNELS; channel += 1) {
        const destination =
          (item * outputFrames + frame) * ACE_SILENCE_SOURCE_CHANNELS + channel;
        output[destination] = frame < semanticFrames
          ? semanticHints[
              (item * semanticFrames + frame) * ACE_SILENCE_SOURCE_CHANNELS + channel
            ]!
          : silenceSourceNct[
              channel * ACE_SILENCE_SOURCE_FRAMES +
                ((frame - semanticFrames) % ACE_SILENCE_SOURCE_FRAMES)
            ]!;
      }
    }
  }
  return output;
}
