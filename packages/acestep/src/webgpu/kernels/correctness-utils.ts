import type { AceModelProfileId } from "../capabilities.js";

export const ACE_CORRECTNESS_WORKGROUP_SIZE = 256;
export const ACE_MAX_DISPATCH_DIMENSION = 65_535;

export interface AceLinearDispatchPlan {
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export function requireAceModelProfile(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  operation: string,
): void {
  if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
    throw new Error(`ACE raw-FP16 ${operation} requires WebGPU shader-f16`);
  }
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(
      `Unknown ACE ${operation} model profile ${String(modelProfile)}`,
    );
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup <
      ACE_CORRECTNESS_WORKGROUP_SIZE ||
    device.limits.maxComputeWorkgroupSizeX < ACE_CORRECTNESS_WORKGROUP_SIZE
  ) {
    throw new Error(
      `ACE ${operation} requires ${ACE_CORRECTNESS_WORKGROUP_SIZE} compute lanes`,
    );
  }
}

export function planAceLinearDispatch(
  elements: number,
  operation: string,
): AceLinearDispatchPlan {
  requirePositiveSafeInteger(elements, `${operation} elements`);
  if (elements > 0xffff_ffff) {
    throw new RangeError(`${operation} exceeds WGSL's u32 indexing domain`);
  }
  const totalWorkgroups = Math.ceil(elements / ACE_CORRECTNESS_WORKGROUP_SIZE);
  const workgroupsX = Math.min(totalWorkgroups, ACE_MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(totalWorkgroups / workgroupsX);
  if (workgroupsY > ACE_MAX_DISPATCH_DIMENSION) {
    throw new RangeError(`${operation} exceeds the portable 2D dispatch domain`);
  }
  return Object.freeze({ elements, workgroupsX, workgroupsY });
}

export function checkedAceProduct(
  values: readonly number[],
  label: string,
): number {
  let result = 1;
  for (const value of values) {
    requirePositiveSafeInteger(value, label);
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} element count is not a safe integer`);
    }
  }
  return result;
}

export function checkedAceSum(
  left: number,
  right: number,
  label: string,
): number {
  requireNonNegativeSafeInteger(left, label);
  requireNonNegativeSafeInteger(right, label);
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`${label} is not a safe integer`);
  }
  return result;
}

export function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

export function requireNonNegativeSafeInteger(
  value: number,
  label: string,
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
}

export function requireAceU32(value: number, label: string): void {
  requireNonNegativeSafeInteger(value, label);
  if (value > 0xffff_ffff) {
    throw new RangeError(`${label} exceeds WGSL's u32 domain`);
  }
}

export function aceActivationBytes(
  modelProfile: AceModelProfileId,
  elements: number,
): number {
  return checkedAceByteProduct(
    elements,
    modelProfile === "reference-bf16" ? 4 : 2,
    "ACE activation",
  );
}

export function acePackedWeightBytes(
  modelProfile: AceModelProfileId,
  elements: number,
): number {
  if (modelProfile === "reference-bf16") {
    return checkedAceByteProduct(Math.ceil(elements / 2), 4, "ACE packed BF16 weight");
  }
  return checkedAceByteProduct(elements, 2, "ACE FP16 weight");
}

export function requireAceBindingBytes(
  binding: GPUBufferBinding,
  required: number,
  label: string,
): void {
  requireNonNegativeSafeInteger(required, `${label} required bytes`);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - offset;
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(available) ||
    available < required ||
    offset + available > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${required} bytes`);
  }
}

export function requireAceDisjointOutput(
  output: GPUBufferBinding,
  inputs: readonly GPUBufferBinding[],
  label: string,
): void {
  const outputStart = output.offset ?? 0;
  const outputEnd = outputStart + (output.size ?? output.buffer.size - outputStart);
  for (const input of inputs) {
    if (input.buffer !== output.buffer) continue;
    const inputStart = input.offset ?? 0;
    const inputEnd = inputStart + (input.size ?? input.buffer.size - inputStart);
    if (inputStart < outputEnd && outputStart < inputEnd) {
      throw new RangeError(`${label} output must not overlap an input binding`);
    }
  }
}

export function aceLinearInvocationWgsl(elements: number, workgroupsX: number): string {
  return /* wgsl */ `
const ELEMENTS: u32 = ${elements}u;
const DISPATCH_X: u32 = ${workgroupsX}u;

fn linear_index(workgroup_id: vec3<u32>, lane: u32) -> u32 {
  let workgroup = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  return workgroup * ${ACE_CORRECTNESS_WORKGROUP_SIZE}u + lane;
}
`;
}

function checkedAceByteProduct(
  elements: number,
  bytesPerElement: number,
  label: string,
): number {
  requirePositiveSafeInteger(elements, `${label} elements`);
  const bytes = elements * bytesPerElement;
  if (!Number.isSafeInteger(bytes)) {
    throw new RangeError(`${label} byte count is not a safe integer`);
  }
  return bytes;
}
