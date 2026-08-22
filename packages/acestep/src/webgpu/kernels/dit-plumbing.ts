import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  acePackedWeightBytes,
  checkedAceProduct,
  checkedAceSum,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

/**
 * Exact little-endian FP32 words produced by the pinned upstream expression:
 * `torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)`.
 * Generated with Torch 2.10.0 on macOS arm64. The 512-byte payload SHA-256 is
 * `97280b1699e126e32cf1e88b1f02603981a549c529734af7989ecbfa3294492e`.
 * Keeping the words authenticated avoids implementation-defined WGSL `exp`
 * drift before the model's cosine/sine phase evaluation.
 */
export const ACE_DIT_TIMESTEP_FREQUENCY_BITS_FP32: readonly number[] =
  Object.freeze([
    0x3f800000, 0x3f6e39f8, 0x3f5dafd6, 0x3f4e4bac,
    0x3f3ff911, 0x3f32a506, 0x3f263de0, 0x3f1ab32b,
    0x3f0ff59a, 0x3f05f6ee, 0x3ef953ce, 0x3ee80460,
    0x3ed7e89a, 0x3ec8eb24, 0x3ebaf81a, 0x3eadfcff,
    0x3ea1e89b, 0x3e96aaea, 0x3e8c3503, 0x3e827908,
    0x3e72d422, 0x3e61f836, 0x3e5247ed, 0x3e43ae7c,
    0x3e361887, 0x3e297409, 0x3e1db040, 0x3e12bd90,
    0x3e088d77, 0x3dfe24df, 0x3dec7fd5, 0x3ddc1464,
    0x3dcccccc, 0x3dbe94c8, 0x3db15978, 0x3da50957,
    0x3d99940d, 0x3d8eea6c, 0x3d84fe4c, 0x3d778511,
    0x3d6655c1, 0x3d5657e3, 0x3d477640, 0x3d399d18,
    0x3d2cba16, 0x3d20bc1b, 0x3d159348, 0x3d0b30cb,
    0x3d0186e2, 0x3cf11177, 0x3ce054d1, 0x3cd0c1a8,
    0x3cc2434e, 0x3cb4c692, 0x3ca83989, 0x3c9c8b96,
    0x3c91ad3a, 0x3c879009, 0x3c7c4d30, 0x3c6ac8e6,
    0x3c5a7bf1, 0x3c4b50b5, 0x3c3d330e, 0x3c301050,
    0x3c23d70a, 0x3c187706, 0x3c0de12e, 0x3c040777,
    0x3bf5b9ad, 0x3be4aa45, 0x3bd4ca16, 0x3bc6040a,
    0x3bb8449a, 0x3bab7982, 0x3b9f91cc, 0x3b947daf,
    0x3b8a2e75, 0x3b80967c, 0x3b6f520d, 0x3b5eb47a,
    0x3b4f3e33, 0x3b40dac2, 0x3b33770d, 0x3b270153,
    0x3b1b690e, 0x3b109ed8, 0x3b06946d, 0x3afa78f0,
    0x3ae91528, 0x3ad8e674, 0x3ac9d759, 0x3abbd3ea,
    0x3aaec98e, 0x3aa2a6f7, 0x3a975c0b, 0x3a8cd9d9,
    0x3a83126e, 0x3a73f1a2, 0x3a6301e3, 0x3a533f24,
    0x3a44948a, 0x3a36ee9e, 0x3a2a3b44, 0x3a1e69a2,
    0x3a136a14, 0x3a092e01, 0x39ff4fac, 0x39ed95e4,
    0x39dd1722, 0x39cdbd93, 0x39bf74d6, 0x39b229fb,
    0x39a5cb60, 0x399a48a0, 0x398f9275, 0x39859aa4,
    0x3978a80d, 0x3967648c, 0x395753e2, 0x394860c0,
    0x393a7753, 0x392d852a, 0x39217919, 0x39164326,
    0x390bd46d, 0x39021f28, 0x38f22cdd, 0x38e15c8e,
  ]);

export const ACE_DIT_TIMESTEP_FREQUENCY_SHA256 =
  "97280b1699e126e32cf1e88b1f02603981a549c529734af7989ecbfa3294492e";

const AUTHENTICATED_TIMESTEP_DIMENSION = 256;
const AUTHENTICATED_TIMESTEP_SCALE = 1_000;
const AUTHENTICATED_TIMESTEP_MAXIMUM_PERIOD = 10_000;

export interface AceDitConcatenateShape {
  readonly batch: number;
  readonly time: number;
  readonly leftWidth: number;
  readonly rightWidth: number;
}

export interface AceDitPatchProjectionShape {
  readonly batch: number;
  readonly time: number;
  readonly inputChannels: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
}

export interface AceDitUnpatchProjectionShape {
  readonly batch: number;
  readonly time: number;
  readonly outputChannels: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
}

export interface AceDitModulationShape {
  readonly batch: number;
  readonly groups: number;
  readonly width: number;
  /** Layer projections have one vector per group; output AdaLN broadcasts temb. */
  readonly projectionLayout: "per-group" | "per-batch";
}

export interface AceDitTimestepEmbeddingShape {
  readonly batch: number;
  readonly dimension: number;
  readonly scale: number;
  readonly maximumPeriod: number;
}

export interface AceDitLinearUpdateShape {
  readonly batch: number;
  readonly time: number;
  readonly channels: number;
  /** `output = latent - velocity * coefficient`. */
  readonly coefficient: number;
}

export interface AceDitPlumbingPlan {
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceDitPatchProjectionPlan extends AceDitPlumbingPlan {
  readonly batch: number;
  readonly time: number;
  readonly paddedTime: number;
  readonly tokens: number;
  readonly inputChannels: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
}

export interface AceDitUnpatchProjectionPlan extends AceDitPlumbingPlan {
  readonly batch: number;
  readonly time: number;
  readonly tokens: number;
  readonly outputChannels: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
}

export interface AceDitModulationPlan extends AceDitPlumbingPlan {
  readonly batch: number;
  readonly groups: number;
  readonly width: number;
  readonly groupElements: number;
}

export interface AceDitDispatch<Plan> {
  readonly label: string;
  readonly plan: Plan;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceDitConcatenateBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceDitProjectionBindings {
  readonly input: GPUBufferBinding;
  readonly weight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceDitModulationBindings {
  /** Activation `[batch,groups,width]`. */
  readonly projection: GPUBufferBinding;
  /** Packed model weight `[1,groups,width]`. */
  readonly table: GPUBufferBinding;
  /** Group-major activation `[groups,batch,width]`. */
  readonly output: GPUBufferBinding;
}

export interface AceDitTimestepEmbeddingBindings {
  /** FP32 `[batch]`, irrespective of model profile. */
  readonly timestep: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceDitLinearUpdateBindings {
  readonly latent: GPUBufferBinding;
  readonly velocity: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

/**
 * Direct kernels for the DiT-only operations that are not ordinary linear,
 * attention, normalization, RoPE, or transformer elementwise primitives.
 * Every operation is kept separate so Stage 1 taps retain the upstream graph.
 */
export class AceCorrectnessDitPlumbingKernel {
  private readonly pipelines = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessDitPlumbingKernel {
    requireAceModelProfile(device, modelProfile, "DiT plumbing");
    return new AceCorrectnessDitPlumbingKernel(device, modelProfile);
  }

  async createConcatenateDispatch(
    label: string,
    shape: AceDitConcatenateShape,
    bindings: AceDitConcatenateBindings,
  ): Promise<AceDitDispatch<AceDitPlumbingPlan>> {
    this.requireLive();
    const plan = planAceDitConcatenate(shape);
    const rows = checkedAceProduct([shape.batch, shape.time], `${label} rows`);
    const leftElements = checkedAceProduct([rows, shape.leftWidth], `${label} left`);
    const rightElements = checkedAceProduct([rows, shape.rightWidth], `${label} right`);
    requireAceBindingBytes(
      bindings.left,
      aceActivationBytes(this.modelProfile, leftElements),
      `${label} left`,
    );
    requireAceBindingBytes(
      bindings.right,
      aceActivationBytes(this.modelProfile, rightElements),
      `${label} right`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.elements),
      `${label} output`,
    );
    requireAceDisjointOutput(bindings.output, [bindings.left, bindings.right], label);
    return await this.createDispatch(
      label,
      `concat:${shape.batch}x${shape.time}x${shape.leftWidth}x${shape.rightWidth}`,
      plan,
      aceCorrectnessDitConcatenateWgsl(this.modelProfile, shape),
      [bindings.left, bindings.right, bindings.output],
    );
  }

  async createPatchProjectionDispatch(
    label: string,
    shape: AceDitPatchProjectionShape,
    bindings: AceDitProjectionBindings,
  ): Promise<AceDitDispatch<AceDitPatchProjectionPlan>> {
    this.requireLive();
    const plan = planAceDitPatchProjection(shape);
    const inputElements = checkedAceProduct(
      [plan.batch, plan.time, plan.inputChannels],
      `${label} input`,
    );
    const weightElements = checkedAceProduct(
      [plan.hiddenSize, plan.inputChannels, plan.patchSize],
      `${label} weight`,
    );
    requireProjectionBindings(
      this.modelProfile,
      label,
      bindings,
      inputElements,
      weightElements,
      plan.hiddenSize,
      plan.elements,
    );
    return await this.createDispatch(
      label,
      `patch:${shape.batch}x${shape.time}x${shape.inputChannels}x${shape.hiddenSize}x${shape.patchSize}`,
      plan,
      aceCorrectnessDitPatchProjectionWgsl(this.modelProfile, shape),
      [bindings.input, bindings.weight, bindings.bias, bindings.output],
    );
  }

  async createUnpatchProjectionDispatch(
    label: string,
    shape: AceDitUnpatchProjectionShape,
    bindings: AceDitProjectionBindings,
  ): Promise<AceDitDispatch<AceDitUnpatchProjectionPlan>> {
    this.requireLive();
    const plan = planAceDitUnpatchProjection(shape);
    const inputElements = checkedAceProduct(
      [plan.batch, plan.tokens, plan.hiddenSize],
      `${label} input`,
    );
    const weightElements = checkedAceProduct(
      [plan.hiddenSize, plan.outputChannels, plan.patchSize],
      `${label} weight`,
    );
    requireProjectionBindings(
      this.modelProfile,
      label,
      bindings,
      inputElements,
      weightElements,
      plan.outputChannels,
      plan.elements,
    );
    return await this.createDispatch(
      label,
      `unpatch:${shape.batch}x${shape.time}x${shape.outputChannels}x${shape.hiddenSize}x${shape.patchSize}`,
      plan,
      aceCorrectnessDitUnpatchProjectionWgsl(this.modelProfile, shape),
      [bindings.input, bindings.weight, bindings.bias, bindings.output],
    );
  }

  async createModulationDispatch(
    label: string,
    shape: AceDitModulationShape,
    bindings: AceDitModulationBindings,
  ): Promise<AceDitDispatch<AceDitModulationPlan>> {
    this.requireLive();
    const plan = planAceDitModulation(shape);
    requireAceBindingBytes(
      bindings.projection,
      aceActivationBytes(
        this.modelProfile,
        shape.projectionLayout === "per-group"
          ? plan.elements
          : plan.groupElements,
      ),
      `${label} projection`,
    );
    requireAceBindingBytes(
      bindings.table,
      acePackedWeightBytes(this.modelProfile, plan.groups * plan.width),
      `${label} table`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.elements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.projection, bindings.table],
      label,
    );
    return await this.createDispatch(
      label,
      `modulation:${shape.batch}x${shape.groups}x${shape.width}:${shape.projectionLayout}`,
      plan,
      aceCorrectnessDitModulationWgsl(this.modelProfile, shape),
      [bindings.projection, bindings.table, bindings.output],
    );
  }

  async createTimestepEmbeddingDispatch(
    label: string,
    shape: AceDitTimestepEmbeddingShape,
    bindings: AceDitTimestepEmbeddingBindings,
  ): Promise<AceDitDispatch<AceDitPlumbingPlan>> {
    this.requireLive();
    const plan = planAceDitTimestepEmbedding(shape);
    requireAceBindingBytes(
      bindings.timestep,
      shape.batch * Float32Array.BYTES_PER_ELEMENT,
      `${label} timestep`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.elements),
      `${label} output`,
    );
    requireAceDisjointOutput(bindings.output, [bindings.timestep], label);
    return await this.createDispatch(
      label,
      `timestep:${shape.batch}x${shape.dimension}:${f32Key(shape.scale)}:${f32Key(shape.maximumPeriod)}`,
      plan,
      aceCorrectnessDitTimestepEmbeddingWgsl(this.modelProfile, shape),
      [bindings.timestep, bindings.output],
    );
  }

  async createLinearUpdateDispatch(
    label: string,
    shape: AceDitLinearUpdateShape,
    bindings: AceDitLinearUpdateBindings,
  ): Promise<AceDitDispatch<AceDitPlumbingPlan>> {
    this.requireLive();
    const plan = planAceDitLinearUpdate(shape);
    const bytes = aceActivationBytes(this.modelProfile, plan.elements);
    requireAceBindingBytes(bindings.latent, bytes, `${label} latent`);
    requireAceBindingBytes(bindings.velocity, bytes, `${label} velocity`);
    requireAceBindingBytes(bindings.output, bytes, `${label} output`);
    requireAceDisjointOutput(
      bindings.output,
      [bindings.latent, bindings.velocity],
      label,
    );
    return await this.createDispatch(
      label,
      `update:${shape.batch}x${shape.time}x${shape.channels}:${f32Key(shape.coefficient)}`,
      plan,
      aceCorrectnessDitLinearUpdateWgsl(this.modelProfile, shape),
      [bindings.latent, bindings.velocity, bindings.output],
    );
  }

  destroy(): void {
    this.destroyed = true;
    this.pipelines.clear();
  }

  private async createDispatch<Plan extends AceDitPlumbingPlan>(
    label: string,
    key: string,
    plan: Plan,
    source: string,
    bindings: readonly GPUBufferBinding[],
  ): Promise<AceDitDispatch<Plan>> {
    const pipeline = await this.pipelineFor(key, label, source);
    this.requireLive(" while compiling");
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((resource, binding) => ({ binding, resource })),
    });
    return Object.freeze({
      label,
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
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const module = this.device.createShaderModule({ label, code: source });
    const created = this.device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE DiT plumbing kernel was destroyed${suffix}`);
    }
  }
}

export function planAceDitConcatenate(
  shape: AceDitConcatenateShape,
): AceDitPlumbingPlan {
  validatePositiveShape(shape, "ACE DiT concatenate");
  const width = checkedAceSum(
    shape.leftWidth,
    shape.rightWidth,
    "ACE DiT concatenate width",
  );
  const elements = checkedAceProduct(
    [shape.batch, shape.time, width],
    "ACE DiT concatenate",
  );
  return Object.freeze({ ...planAceLinearDispatch(elements, "ACE DiT concatenate") });
}

export function planAceDitPatchProjection(
  shape: AceDitPatchProjectionShape,
): AceDitPatchProjectionPlan {
  validatePositiveShape(shape, "ACE DiT patch projection");
  const tokens = Math.ceil(shape.time / shape.patchSize);
  const paddedTime = checkedAceProduct(
    [tokens, shape.patchSize],
    "ACE DiT padded time",
  );
  const elements = checkedAceProduct(
    [shape.batch, tokens, shape.hiddenSize],
    "ACE DiT patch output",
  );
  return Object.freeze({
    ...shape,
    paddedTime,
    tokens,
    ...planAceLinearDispatch(elements, "ACE DiT patch projection"),
  });
}

export function planAceDitUnpatchProjection(
  shape: AceDitUnpatchProjectionShape,
): AceDitUnpatchProjectionPlan {
  validatePositiveShape(shape, "ACE DiT unpatch projection");
  const tokens = Math.ceil(shape.time / shape.patchSize);
  const elements = checkedAceProduct(
    [shape.batch, shape.time, shape.outputChannels],
    "ACE DiT unpatch output",
  );
  return Object.freeze({
    ...shape,
    tokens,
    ...planAceLinearDispatch(elements, "ACE DiT unpatch projection"),
  });
}

export function planAceDitModulation(
  shape: AceDitModulationShape,
): AceDitModulationPlan {
  validatePositiveShape(
    { batch: shape.batch, groups: shape.groups, width: shape.width },
    "ACE DiT modulation",
  );
  if (
    shape.projectionLayout !== "per-group" &&
    shape.projectionLayout !== "per-batch"
  ) {
    throw new TypeError(
      `Unknown ACE DiT modulation projection layout ${String(shape.projectionLayout)}`,
    );
  }
  const groupElements = checkedAceProduct(
    [shape.batch, shape.width],
    "ACE DiT modulation group",
  );
  const elements = checkedAceProduct(
    [shape.groups, groupElements],
    "ACE DiT modulation",
  );
  return Object.freeze({
    ...shape,
    groupElements,
    ...planAceLinearDispatch(elements, "ACE DiT modulation"),
  });
}

export function planAceDitTimestepEmbedding(
  shape: AceDitTimestepEmbeddingShape,
): AceDitPlumbingPlan {
  requirePositiveSafeInteger(shape.batch, "ACE DiT timestep batch");
  requirePositiveSafeInteger(shape.dimension, "ACE DiT timestep dimension");
  if (!Number.isFinite(shape.scale) || shape.scale <= 0) {
    throw new RangeError("ACE DiT timestep scale must be positive and finite");
  }
  if (!Number.isFinite(shape.maximumPeriod) || shape.maximumPeriod <= 1) {
    throw new RangeError("ACE DiT timestep maximumPeriod must be finite and greater than one");
  }
  if (
    shape.dimension !== AUTHENTICATED_TIMESTEP_DIMENSION ||
    shape.scale !== AUTHENTICATED_TIMESTEP_SCALE ||
    shape.maximumPeriod !== AUTHENTICATED_TIMESTEP_MAXIMUM_PERIOD
  ) {
    throw new RangeError(
      "ACE DiT timestep embedding has no authenticated Torch 2.10 frequency vector for this geometry",
    );
  }
  const elements = checkedAceProduct(
    [shape.batch, shape.dimension],
    "ACE DiT timestep embedding",
  );
  return Object.freeze({
    ...planAceLinearDispatch(elements, "ACE DiT timestep embedding"),
  });
}

export function planAceDitLinearUpdate(
  shape: AceDitLinearUpdateShape,
): AceDitPlumbingPlan {
  validatePositiveShape(
    { batch: shape.batch, time: shape.time, channels: shape.channels },
    "ACE DiT linear update",
  );
  if (!Number.isFinite(shape.coefficient) || shape.coefficient < 0 || shape.coefficient > 1) {
    throw new RangeError("ACE DiT update coefficient must be finite in [0, 1]");
  }
  const elements = checkedAceProduct(
    [shape.batch, shape.time, shape.channels],
    "ACE DiT linear update",
  );
  return Object.freeze({
    ...planAceLinearDispatch(elements, "ACE DiT linear update"),
  });
}

export function aceCorrectnessDitConcatenateWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitConcatenateShape,
): string {
  const plan = planAceDitConcatenate(shape);
  const type = activationType(modelProfile, "concatenate");
  const outputWidth = shape.leftWidth + shape.rightWidth;
  return /* wgsl */ `${profilePreamble(modelProfile)}
const LEFT_WIDTH: u32 = ${shape.leftWidth}u;
const RIGHT_WIDTH: u32 = ${shape.rightWidth}u;
const OUTPUT_WIDTH: u32 = ${outputWidth}u;
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> left: array<${type}>;
@group(0) @binding(1) var<storage, read> right: array<${type}>;
@group(0) @binding(2) var<storage, read_write> output: array<${type}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let row = index / OUTPUT_WIDTH;
  let column = index % OUTPUT_WIDTH;
  if (column < LEFT_WIDTH) {
    output[index] = left[row * LEFT_WIDTH + column];
  } else {
    output[index] = right[row * RIGHT_WIDTH + (column - LEFT_WIDTH)];
  }
}
`;
}

export function aceCorrectnessDitPatchProjectionWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitPatchProjectionShape,
): string {
  const plan = planAceDitPatchProjection(shape);
  const types = projectionTypes(modelProfile, "patch projection");
  return /* wgsl */ `${types.preamble}
const TIME: u32 = ${plan.time}u;
const TOKENS: u32 = ${plan.tokens}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const HIDDEN_SIZE: u32 = ${plan.hiddenSize}u;
const PATCH_SIZE: u32 = ${plan.patchSize}u;
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> input: array<${types.activation}>;
@group(0) @binding(1) var<storage, read> weight: array<${types.weight}>;
@group(0) @binding(2) var<storage, read> bias: array<${types.weight}>;
@group(0) @binding(3) var<storage, read_write> output: array<${types.activation}>;
${types.weightLoader}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let hidden = index % HIDDEN_SIZE;
  let flat_token = index / HIDDEN_SIZE;
  let token = flat_token % TOKENS;
  let batch = flat_token / TOKENS;
  var sum: ${types.accumulation} = ${types.loadBias("hidden")};
  for (var channel = 0u; channel < INPUT_CHANNELS; channel += 1u) {
    for (var kernel_index = 0u; kernel_index < PATCH_SIZE; kernel_index += 1u) {
      let time = token * PATCH_SIZE + kernel_index;
      if (time < TIME) {
        let input_index = (batch * TIME + time) * INPUT_CHANNELS + channel;
        let weight_index = (hidden * INPUT_CHANNELS + channel) * PATCH_SIZE + kernel_index;
        sum = sum + ${types.loadActivation("input_index")} * ${types.loadWeight("weight_index")};
      }
    }
  }
  output[index] = ${types.store("sum")};
}
`;
}

export function aceCorrectnessDitUnpatchProjectionWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitUnpatchProjectionShape,
): string {
  const plan = planAceDitUnpatchProjection(shape);
  const types = projectionTypes(modelProfile, "unpatch projection");
  return /* wgsl */ `${types.preamble}
const TIME: u32 = ${plan.time}u;
const TOKENS: u32 = ${plan.tokens}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const HIDDEN_SIZE: u32 = ${plan.hiddenSize}u;
const PATCH_SIZE: u32 = ${plan.patchSize}u;
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> input: array<${types.activation}>;
@group(0) @binding(1) var<storage, read> weight: array<${types.weight}>;
@group(0) @binding(2) var<storage, read> bias: array<${types.weight}>;
@group(0) @binding(3) var<storage, read_write> output: array<${types.activation}>;
${types.weightLoader}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let channel = index % OUTPUT_CHANNELS;
  let flat_time = index / OUTPUT_CHANNELS;
  let time = flat_time % TIME;
  let batch = flat_time / TIME;
  let token = time / PATCH_SIZE;
  let kernel_index = time % PATCH_SIZE;
  var sum: ${types.accumulation} = ${types.loadBias("channel")};
  for (var hidden = 0u; hidden < HIDDEN_SIZE; hidden += 1u) {
    let input_index = (batch * TOKENS + token) * HIDDEN_SIZE + hidden;
    let weight_index = (hidden * OUTPUT_CHANNELS + channel) * PATCH_SIZE + kernel_index;
    sum = sum + ${types.loadActivation("input_index")} * ${types.loadWeight("weight_index")};
  }
  output[index] = ${types.store("sum")};
}
`;
}

export function aceCorrectnessDitModulationWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitModulationShape,
): string {
  const plan = planAceDitModulation(shape);
  const types = projectionTypes(modelProfile, "modulation");
  const tableLoader = modelProfile === "reference-bf16"
    ? `
fn load_table(index: u32) -> f32 {
  let pair = table[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}`
    : "";
  const valueExpression = modelProfile === "reference-bf16"
    ? "projection[projection_index] + load_table(table_index)"
    : "projection[projection_index] + table[table_index]";
  const projectionIndex = shape.projectionLayout === "per-group"
    ? "(batch * GROUPS + group) * WIDTH + column"
    : "batch * WIDTH + column";
  return /* wgsl */ `${types.preamble}
const GROUPS: u32 = ${plan.groups}u;
const WIDTH: u32 = ${plan.width}u;
const GROUP_ELEMENTS: u32 = ${plan.groupElements}u;
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> projection: array<${types.activation}>;
@group(0) @binding(1) var<storage, read> table: array<${types.weight}>;
@group(0) @binding(2) var<storage, read_write> output: array<${types.activation}>;
${tableLoader}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let output_index = linear_index(workgroup_id, lane);
  if (output_index >= ELEMENTS) { return; }
  let group = output_index / GROUP_ELEMENTS;
  let batch_column = output_index % GROUP_ELEMENTS;
  let batch = batch_column / WIDTH;
  let column = batch_column % WIDTH;
  let projection_index = ${projectionIndex};
  let table_index = group * WIDTH + column;
  let value = ${valueExpression};
  output[output_index] = ${types.store("value")};
}
`;
}

export function aceCorrectnessDitTimestepEmbeddingWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitTimestepEmbeddingShape,
): string {
  const plan = planAceDitTimestepEmbedding(shape);
  const type = activationType(modelProfile, "timestep embedding");
  const half = Math.floor(shape.dimension / 2);
  const frequencyWords = ACE_DIT_TIMESTEP_FREQUENCY_BITS_FP32
    .map((bits) => `0x${bits.toString(16).padStart(8, "0")}u`)
    .join(", ");
  return /* wgsl */ `${profilePreamble(modelProfile)}
const DIMENSION: u32 = ${shape.dimension}u;
const HALF: u32 = ${half}u;
const SCALE: f32 = ${Math.fround(shape.scale)};
const TIMESTEP_FREQUENCY_BITS: array<u32, ${half}> = array<u32, ${half}>(${frequencyWords});
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> timestep: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<${type}>;

fn round_bfloat16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let rounded = bits + 0x00007fffu + ((bits >> 16u) & 1u);
  return bitcast<f32>(rounded & 0xffff0000u);
}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let batch = index / DIMENSION;
  let dimension = index % DIMENSION;
  var value = 0.0;
  if (dimension < HALF * 2u) {
    let frequency_index = dimension % HALF;
    let frequency = bitcast<f32>(TIMESTEP_FREQUENCY_BITS[frequency_index]);
    let scaled_timestep = round_bfloat16(timestep[batch] * SCALE);
    let phase = scaled_timestep * frequency;
    value = select(sin(phase), cos(phase), dimension < HALF);
  }
  output[index] = ${modelProfile === "raw-fp16" ? "f16(value)" : "value"};
}
`;
}

export function aceCorrectnessDitLinearUpdateWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDitLinearUpdateShape,
): string {
  const plan = planAceDitLinearUpdate(shape);
  const type = activationType(modelProfile, "linear update");
  const coefficient = Math.fround(shape.coefficient);
  return /* wgsl */ `${profilePreamble(modelProfile)}
const COEFFICIENT: f32 = ${coefficient};
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
@group(0) @binding(0) var<storage, read> latent: array<${type}>;
@group(0) @binding(1) var<storage, read> velocity: array<${type}>;
@group(0) @binding(2) var<storage, read_write> output: array<${type}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  ${modelProfile === "raw-fp16"
    ? "let value = latent[index] - velocity[index] * f16(COEFFICIENT);"
    : "let value = latent[index] - velocity[index] * COEFFICIENT;"}
  output[index] = value;
}
`;
}

function requireProjectionBindings(
  profile: AceModelProfileId,
  label: string,
  bindings: AceDitProjectionBindings,
  inputElements: number,
  weightElements: number,
  biasElements: number,
  outputElements: number,
): void {
  requireAceBindingBytes(
    bindings.input,
    aceActivationBytes(profile, inputElements),
    `${label} input`,
  );
  requireAceBindingBytes(
    bindings.weight,
    acePackedWeightBytes(profile, weightElements),
    `${label} weight`,
  );
  requireAceBindingBytes(
    bindings.bias,
    acePackedWeightBytes(profile, biasElements),
    `${label} bias`,
  );
  requireAceBindingBytes(
    bindings.output,
    aceActivationBytes(profile, outputElements),
    `${label} output`,
  );
  requireAceDisjointOutput(
    bindings.output,
    [bindings.input, bindings.weight, bindings.bias],
    label,
  );
}

function validatePositiveShape(
  shape: object,
  label: string,
): void {
  for (const [name, value] of Object.entries(shape)) {
    if (typeof value !== "number") {
      throw new TypeError(`${label} ${name} must be numeric`);
    }
    requirePositiveSafeInteger(value, `${label} ${name}`);
  }
}

function profilePreamble(modelProfile: AceModelProfileId): string {
  if (modelProfile === "reference-bf16") return "";
  if (modelProfile === "raw-fp16") return "enable f16;";
  throw new TypeError(
    `Unknown ACE DiT plumbing model profile ${String(modelProfile)}`,
  );
}

function activationType(
  modelProfile: AceModelProfileId,
  operation: string,
): "f32" | "f16" {
  if (modelProfile === "reference-bf16") return "f32";
  if (modelProfile === "raw-fp16") return "f16";
  throw new TypeError(
    `Unknown ACE DiT ${operation} model profile ${String(modelProfile)}`,
  );
}

function projectionTypes(modelProfile: AceModelProfileId, operation: string) {
  if (modelProfile === "reference-bf16") {
    return {
      preamble: "",
      activation: "f32",
      weight: "u32",
      accumulation: "f32",
      weightLoader: `
fn load_weight(index: u32) -> f32 {
  let pair = weight[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}
fn load_bias(index: u32) -> f32 {
  let pair = bias[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}`,
      loadWeight: (index: string) => `load_weight(${index})`,
      loadBias: (index: string) => `load_bias(${index})`,
      loadActivation: (index: string) => `input[${index}]`,
      store: (value: string) => value,
    } as const;
  }
  if (modelProfile === "raw-fp16") {
    return {
      preamble: "enable f16;",
      activation: "f16",
      weight: "f16",
      accumulation: "f16",
      weightLoader: "",
      loadWeight: (index: string) => `weight[${index}]`,
      loadBias: (index: string) => `bias[${index}]`,
      loadActivation: (index: string) => `input[${index}]`,
      store: (value: string) => value,
    } as const;
  }
  throw new TypeError(
    `Unknown ACE DiT ${operation} model profile ${String(modelProfile)}`,
  );
}

function f32Key(value: number): string {
  const array = new Float32Array([value]);
  return new Uint32Array(array.buffer)[0]!.toString(16);
}
