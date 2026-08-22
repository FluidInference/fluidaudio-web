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
} from "./correctness-utils.js";

/** Pinned ACE-Step 1.5 ResidualFSQ scalar levels, in radix order. */
export const ACE_FSQ_LEVELS = Object.freeze([8, 8, 8, 5, 5, 5] as const);

/** Mixed-radix bases produced by the pinned FSQ implementation. */
export const ACE_FSQ_BASES = Object.freeze([
  1,
  8,
  64,
  512,
  2_560,
  12_800,
] as const);

export const ACE_FSQ_CODE_DIMENSION = ACE_FSQ_LEVELS.length;
export const ACE_FSQ_CODEBOOK_SIZE = 64_000;

/**
 * Scalar codebook values after the pinned model's implicit FP32 codebook is
 * moved to BF16. Both runtime profiles originate from these values.
 */
export const ACE_FSQ_LEVEL_8_BFLOAT16_VALUES = Object.freeze([
  -1,
  -0.71484375,
  -0.427734375,
  -0.142578125,
  0.142578125,
  0.427734375,
  0.71484375,
  1,
] as const);

export const ACE_FSQ_LEVEL_5_BFLOAT16_VALUES = Object.freeze([
  -1,
  -0.5,
  0,
  0.5,
  1,
] as const);

export interface AceFsqDecodeShape {
  readonly codeCount: number;
}

export interface AceFsqDecodePlan extends AceFsqDecodeShape {
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFsqDecodeBindings {
  /** Parsed audio-code values in `[0, 64000)`, not planner token IDs. */
  readonly codeIds: GPUBufferBinding;
  /** `[codeCount, 6]` reference-FP32 or raw-FP16 activations. */
  readonly output: GPUBufferBinding;
  /** One zero-initialized u32. The shader sets it to one for an invalid code. */
  readonly validationStatus: GPUBufferBinding;
}

export interface AceFsqDecodeDispatch {
  readonly label: string;
  readonly plan: AceFsqDecodePlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Exact mixed-radix inverse for the pinned single-layer ResidualFSQ codebook.
 * The implicit codebook is constructed in FP32 but is a registered buffer, so
 * upstream model placement rounds it to BF16 before this decode is consumed.
 *
 * This kernel intentionally stops before `quantizer.project_out`. The retained
 * 6→2048 linear projection is dispatched through the ordinary GEMM path, so
 * its package weights and bias stay auditable rather than being embedded in a
 * specialized shader.
 */
export class AceCorrectnessFsqDecodeKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessFsqDecodeKernel {
    requireAceModelProfile(device, modelProfile, "FSQ decode");
    return new AceCorrectnessFsqDecodeKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceFsqDecodeShape,
    bindings: AceFsqDecodeBindings,
  ): Promise<AceFsqDecodeDispatch> {
    this.requireLive();
    const plan = planAceFsqDecode(shape);
    requireAceBindingBytes(
      bindings.codeIds,
      plan.codeCount * Uint32Array.BYTES_PER_ELEMENT,
      `${label} code IDs`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceBindingBytes(
      bindings.validationStatus,
      Uint32Array.BYTES_PER_ELEMENT,
      `${label} validation status`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.codeIds, bindings.validationStatus],
      label,
    );
    requireAceDisjointOutput(
      bindings.validationStatus,
      [bindings.codeIds, bindings.output],
      `${label} validation status`,
    );

    const pipeline = await this.pipelineFor(plan);
    this.requireLive(" while compiling");
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.codeIds },
        { binding: 1, resource: bindings.output },
        { binding: 2, resource: bindings.validationStatus },
      ],
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

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(plan: AceFsqDecodePlan): Promise<GPUComputePipeline> {
    const key = `${this.modelProfile}:${plan.codeCount}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-fsq-${key}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessFsqDecodeWgsl(this.modelProfile, plan),
    });
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
      throw new Error(`ACE FSQ decode kernel was destroyed${suffix}`);
    }
  }
}

export function planAceFsqDecode(shape: AceFsqDecodeShape): AceFsqDecodePlan {
  const outputElements = checkedAceProduct(
    [shape.codeCount, ACE_FSQ_CODE_DIMENSION],
    "ACE FSQ output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE FSQ decode");
  return Object.freeze({
    codeCount: shape.codeCount,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function decodeAceFsqCodes(codeIds: readonly number[]): Float32Array {
  if (codeIds.length === 0) {
    throw new RangeError("ACE FSQ decode requires at least one code");
  }
  const output = new Float32Array(codeIds.length * ACE_FSQ_CODE_DIMENSION);
  for (let codeIndex = 0; codeIndex < codeIds.length; codeIndex += 1) {
    const code = codeIds[codeIndex];
    if (
      code === undefined ||
      !Number.isSafeInteger(code) ||
      code < 0 ||
      code >= ACE_FSQ_CODEBOOK_SIZE
    ) {
      throw new RangeError(
        `ACE FSQ code ${codeIndex} must be an integer in [0, ${ACE_FSQ_CODEBOOK_SIZE})`,
      );
    }
    for (let dimension = 0; dimension < ACE_FSQ_CODE_DIMENSION; dimension += 1) {
      const level = ACE_FSQ_LEVELS[dimension]!;
      const digit = Math.floor(code / ACE_FSQ_BASES[dimension]!) % level;
      output[codeIndex * ACE_FSQ_CODE_DIMENSION + dimension] = level === 8
        ? ACE_FSQ_LEVEL_8_BFLOAT16_VALUES[digit]!
        : ACE_FSQ_LEVEL_5_BFLOAT16_VALUES[digit]!;
    }
  }
  return output;
}

export function aceCorrectnessFsqDecodeWgsl(
  modelProfile: AceModelProfileId,
  shape: AceFsqDecodeShape,
): string {
  const plan = planAceFsqDecode(shape);
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(
      `Unknown ACE FSQ decode model profile ${String(modelProfile)}`,
    );
  }
  const scalar = f16 ? "f16" : "f32";
  const scalarLiteral = (value: number): string => {
    const decimal = Number.isInteger(value) ? `${value}.0` : String(value);
    return f16 ? `f16(${decimal})` : decimal;
  };
  const level8Values = ACE_FSQ_LEVEL_8_BFLOAT16_VALUES
    .map(scalarLiteral)
    .join(", ");
  const level5Values = ACE_FSQ_LEVEL_5_BFLOAT16_VALUES
    .map(scalarLiteral)
    .join(", ");
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const CODE_DIMENSION: u32 = ${ACE_FSQ_CODE_DIMENSION}u;
const CODEBOOK_SIZE: u32 = ${ACE_FSQ_CODEBOOK_SIZE}u;

const LEVELS = array<u32, ${ACE_FSQ_CODE_DIMENSION}>(8u, 8u, 8u, 5u, 5u, 5u);
const BASES = array<u32, ${ACE_FSQ_CODE_DIMENSION}>(1u, 8u, 64u, 512u, 2560u, 12800u);
const LEVEL_8_VALUES = array<${scalar}, 8>(${level8Values});
const LEVEL_5_VALUES = array<${scalar}, 5>(${level5Values});

@group(0) @binding(0) var<storage, read> code_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<${scalar}>;
@group(0) @binding(2) var<storage, read_write> validation_status: array<atomic<u32>>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let code_index = index / CODE_DIMENSION;
  let dimension = index % CODE_DIMENSION;
  let code = code_ids[code_index];
  if (code >= CODEBOOK_SIZE) {
    output[index] = ${f16 ? "f16(0.0)" : "0.0"};
    atomicStore(&validation_status[0], 1u);
    return;
  }
  let level = LEVELS[dimension];
  let digit = (code / BASES[dimension]) % level;
  if (level == 8u) {
    output[index] = LEVEL_8_VALUES[digit];
  } else {
    output[index] = LEVEL_5_VALUES[digit];
  }
}
`;
}
