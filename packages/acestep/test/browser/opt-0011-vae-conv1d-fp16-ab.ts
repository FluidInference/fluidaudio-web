/// <reference types="@webgpu/types" />

import opt0011VaeConv1dFp16CoreSource from
  "../../benchmark/opt-0011-vae-conv1d-fp16.ts?raw";
import type { AceVaeConv1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";
import {
  AceOpt0011VaeConv1dFp16PortableWorkgroupKernel,
  AceOpt0011VaeConv1dFp16ScalarOracleKernel,
  aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
  aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
  planAceOpt0011VaeConv1dFp16,
  type AceOpt0011VaeConv1dFp16Bindings,
  type AceOpt0011VaeConv1dFp16Dispatch,
  type AceOpt0011VaeConv1dFp16OutputStorage,
  type AceOpt0011VaeConv1dFp16Plan,
} from "../../benchmark/opt-0011-vae-conv1d-fp16.js";

export type Opt0011VaeConv1dFp16CaseId =
  | "d1-b2-f35-c65-c13-bias"
  | "d3-b3-f51-c64-c11-no-bias"
  | "d9-b2-f67-c63-c9-bias"
  | "signed-zero"
  | "arithmetic-subnormal-cancellation-range-edge"
  | "long-cin1024"
  | "final-b1-f4097-c128-c2-f32"
  | "fifo-cancellation-two-range"
  | "production-block0-d1-c1024";

export type Opt0011VaeConv1dUploadRole = "input" | "weight" | "bias";
export type Opt0011VaeConv1dCpuClassification =
  | "both-allowed-forms"
  | "separate-rounded-only"
  | "one-round-contracted-only"
  | "unexpected";

export interface Opt0011VaeConv1dFixture {
  readonly id: Opt0011VaeConv1dFp16CaseId;
  readonly shape: AceVaeConv1dShape;
  readonly hasBias: boolean;
  readonly outputStorage: AceOpt0011VaeConv1dFp16OutputStorage;
  readonly cpuOracleScope: "complete-output" | "range-boundary-sparse";
  readonly coverage: readonly string[];
}

export interface Opt0011VaeConv1dCpuForms {
  readonly separateRoundedFp32: number;
  readonly oneRoundContractedFp32: number;
  readonly separateRoundedFp32Bits: number;
  readonly oneRoundContractedFp32Bits: number;
  readonly separateStoreBits: number;
  readonly contractedStoreBits: number;
}

export interface Opt0011VaeConv1dRawBitComparison {
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
}

const OUTPUT_GUARD_BYTES = 256;
const OUTPUT_GUARD_WORD = 0xa55a_a55a;
const SCALAR_F16_SENTINEL = 0x7e11;
const WORKGROUP_F16_SENTINEL = 0x7e22;
const SCALAR_F32_SENTINEL = 0x7fc1_1111;
const WORKGROUP_F32_SENTINEL = 0x7fc2_2222;
const QUEUE_EMPTY_IDLE_MILLISECONDS = 1;
const FIFO_EXACT_ZERO_CANONICALIZATION_INDEX = 176_232;
const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

const SMALL_INPUT_BITS = Object.freeze([
  0x0000, 0x8000, 0x2400, 0xa400, 0x2c00, 0xac00, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3800, 0xb800, 0x3555, 0xb555, 0x1800, 0x9800,
]);
const SMALL_WEIGHT_BITS = Object.freeze([
  0x0000, 0x8000, 0x1400, 0x9400, 0x1800, 0x9800, 0x1c00, 0x9c00,
  0x2000, 0xa000, 0x2200, 0xa200, 0x2400, 0xa400,
]);
const SMALL_BIAS_BITS = Object.freeze([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400, 0x2800, 0xa800,
]);

export const OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES = Object.freeze([
  fixture(
    "d1-b2-f35-c65-c13-bias",
    shape(2, 35, 65, 13, 1),
    true,
    "float16",
    "complete-output",
    ["dilation-1", "batch-2", "cin-64-tail", "cout-tail", "bias"],
  ),
  fixture(
    "d3-b3-f51-c64-c11-no-bias",
    shape(3, 51, 64, 11, 3),
    false,
    "float16",
    "complete-output",
    ["dilation-3", "batch-3", "cin-exact-chunk", "cout-tail", "no-bias"],
  ),
  fixture(
    "d9-b2-f67-c63-c9-bias",
    shape(2, 67, 63, 9, 9),
    true,
    "float16",
    "complete-output",
    ["dilation-9", "batch-2", "cin-under-chunk", "frame-tail", "bias"],
  ),
  fixture(
    "signed-zero",
    shape(1, 7, 1, 2, 1),
    true,
    "float16",
    "complete-output",
    ["positive-zero", "negative-zero", "exact-f16-store"],
  ),
  fixture(
    "arithmetic-subnormal-cancellation-range-edge",
    shape(1, 13, 65, 5, 1),
    false,
    "float16",
    "complete-output",
    [
      "source-k-then-cin-order",
      "min-f16-subnormal",
      "min-f16-normal",
      "max-f16-subnormal",
      "max-f16-finite",
      "cancellation",
      "rounding-classifier",
    ],
  ),
  fixture(
    "long-cin1024",
    shape(1, 11, 1_024, 3, 3),
    true,
    "float16",
    "complete-output",
    ["cin-1024", "sixteen-input-channel-chunks", "long-reduction"],
  ),
  fixture(
    "final-b1-f4097-c128-c2-f32",
    shape(1, 4_097, 128, 2, 1),
    false,
    "float32",
    "complete-output",
    ["final-raw-waveform-boundary", "f32-output", "frame-4097", "no-bias"],
  ),
  fixture(
    "fifo-cancellation-two-range",
    shape(1, 1_048_577, 1, 1, 1),
    false,
    "float16",
    "range-boundary-sparse",
    ["two-real-ranges", "post-drain-cancellation", "readback-prevention"],
  ),
  fixture(
    "production-block0-d1-c1024",
    shape(1, 2_560, 1_024, 1_024, 1),
    true,
    "float16",
    "range-boundary-sparse",
    [
      "full-production-shape",
      "eighty-ranges",
      "complete-u16-readback",
      "scalar-workgroup-bit-identity",
    ],
  ),
] satisfies readonly Opt0011VaeConv1dFixture[]);

/** SHA-256 over each complete, 4-byte-aligned binding upload byte stream. */
export const OPT_0011_VAE_CONV1D_UPLOAD_SHA256 = Object.freeze({
  "d1-b2-f35-c65-c13-bias": Object.freeze({
    input: "88c5102d41df48731f547933816c6fa2e700503eb9112531ddf80b16dad36c32",
    weight: "b1d6e09473bf5fce84ee7db3ddc875ba5a5a124ab1520552a8376717fc989cc4",
    bias: "026145a6086b025c5b8a3f1e1099954b01a16bac055a3e8a63593ca0b48db0de",
  }),
  "d3-b3-f51-c64-c11-no-bias": Object.freeze({
    input: "7eef6a82af0e24fc36b30554ead40eb4533bee407f7522e8f949532e019d1e6e",
    weight: "a51f9ef14f6a1fec8920e399338ebd26b7c8af2d79c0d9642e0343d86f8fafbe",
  }),
  "d9-b2-f67-c63-c9-bias": Object.freeze({
    input: "04856fe2e4b36e8e15ae255a18c8fecc3dbedcfa5ebea6f9e5287d01f116f0c6",
    weight: "aed721aeb73a4606ae91fb1caf1b4525a89e56993dd5eea3920e1b1b5ca73273",
    bias: "dcacaab9cd32fbc986bacf78a1cc82562c1b6376933dd3d2d060c9b5d8f11763",
  }),
  "signed-zero": Object.freeze({
    input: "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
    weight: "28877ff8db4183eb0e2f0ae3f3bb071b4c8aa5a8c83d75dd5843893b860fc5ec",
    bias: "94751be059d25844fadecb89656e16fe3ec44058475017859ec762f26173f014",
  }),
  "arithmetic-subnormal-cancellation-range-edge": Object.freeze({
    input: "2a7d1871829eb049f1ddb06e1d38d66e7a628283d1eb23fb65281c52d9265beb",
    weight: "48476a6f3d4c7cf1640851aa5ddf8f58c50a0643fb3fe2f40be118204bec77d7",
  }),
  "long-cin1024": Object.freeze({
    input: "18489fba25a6e66213247d0493070e5b3357850d5187fed2361f55cfd2bb5b2b",
    weight: "20a059a0d54a9a1d616e241b6f0e3af4b8ffa1fe5831e46ac080309769da45e2",
    bias: "19339763db311d265d796d9595f714a38829f81875a6bb0855f071882cdb8d15",
  }),
  "final-b1-f4097-c128-c2-f32": Object.freeze({
    input: "9e19c6e3a5b7c601295044e66da45474748fb0338bff87263c4afc6fc9aad267",
    weight: "bbca47a3ded0c427ba823451e394057fcc80517d6f5139f61d33e9eb2cc9e41d",
  }),
  "fifo-cancellation-two-range": Object.freeze({
    input: "944b404a030b3ec13d7879d9f4cc1987efc427085d61e0c4f0feb579b059feea",
    weight: "9af0593d93e1d60d31d203dd5ed3ba92ca63a8a67ee01a81eb60147870024461",
  }),
  "production-block0-d1-c1024": Object.freeze({
    input: "aacb71043e63b70304ed1edbdc72b92ac3a2033124ad03781616d0847e84bbe1",
    weight: "23fc2f0822fcb98e6ef804159ad68d9f9d81011664795298102c951edf6a3c3c",
    bias: "b3b67685c64dbe731a9c199ed2b0e0c5e2af3ce24121dd6efa11492c112001c5",
  }),
} satisfies Readonly<Record<
  Opt0011VaeConv1dFp16CaseId,
  Readonly<Partial<Record<Opt0011VaeConv1dUploadRole, string>>>
>>);

export const OPT_0011_VAE_CONV1D_CORE_COMMIT =
  "82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd" as const;
export const OPT_0011_VAE_CONV1D_CORE_SOURCE_SHA256 =
  "bdb1ce2732d8617f61132401ab01155163a4f4197e7c7b01eb550b8408553ceb" as const;

/** SHA-256 over the exact generated WGSL passed to each shader module. */
export const OPT_0011_VAE_CONV1D_GENERATED_SHADER_SHA256 = Object.freeze({
  "d1-b2-f35-c65-c13-bias": Object.freeze({
    scalar: "649c791a7fca0b5bdd199593f42aacbff81c19706d19b1628278bfae31d966f8",
    workgroup: "f5c43b25d0f665448ee51bfc1097d1b4db930a3f84445e777e774685262a74f7",
  }),
  "d3-b3-f51-c64-c11-no-bias": Object.freeze({
    scalar: "de0a860421d579c9ed50634fecdd80850560d15910b94e421eb41f9e21bf4307",
    workgroup: "0b004e039af37b0d27b40505b4c690f82d22e8b04a4bcdbda7dea0d1cf33f4a7",
  }),
  "d9-b2-f67-c63-c9-bias": Object.freeze({
    scalar: "e9e2c3ff91035723336896ebc91a9d43234dcd7f079f1d17d7056c0d9b2b4088",
    workgroup: "44158d239ba04781e1bdb34756e419d64c86b102f38e9a8761c0da8f611f6457",
  }),
  "signed-zero": Object.freeze({
    scalar: "5e1d05fc90916e04dd0b005a8dbd2ce262cadfdaa6344b842ec45aef3ea8bf8e",
    workgroup: "dbb5f9c58e8bab0a0ab299a05986a6a4437dff6a19b92c510666784d7d86751d",
  }),
  "arithmetic-subnormal-cancellation-range-edge": Object.freeze({
    scalar: "d4dec8389c026875a4f20c80d54e556d08f8d13c3944b7b169fd59364c7b2a9b",
    workgroup: "48032dae76bc16357015ce7f3553632a01b4bc3ed6197bb4bc1808bff5759509",
  }),
  "long-cin1024": Object.freeze({
    scalar: "ea7e38ee59a5fe6b47b1cb2cd4e82c14fd096d9bdd502ddcf03f3dff3c234a16",
    workgroup: "f3e4bf9b05016c8166c00054aa5d670893728ed69a091b5f6a1d9282aaa1bfd0",
  }),
  "final-b1-f4097-c128-c2-f32": Object.freeze({
    scalar: "fae1722f85f474195ce150e348a106dae19fa72bd3fb7b0c5dd494dfa1f1ec20",
    workgroup: "40a67099bdc6d2d0618afc70976457ee68e4fc3fa3927ae8d678e251ea088261",
  }),
  "fifo-cancellation-two-range": Object.freeze({
    scalar: "526ac79ee8de8def9326ad12b337f7280083c068fb651aadb1ca0a16216164ef",
    workgroup: "6472d5735f2cf74497269b247e27c2dc2e7e9acc5ef084f6788cd30d0982174e",
  }),
  "production-block0-d1-c1024": Object.freeze({
    scalar: "0e28ef309e8e1ec4beb476bffd57b5ecc6d5591a8d56fff9d74841595983ce8f",
    workgroup: "8a6a217d956ba42f4b73952d8d0b9017afa2a286eeb5da08d14d16dc61bc16c2",
  }),
} satisfies Readonly<Record<
  Opt0011VaeConv1dFp16CaseId,
  Readonly<{ readonly scalar: string; readonly workgroup: string }>
>>);

export function opt0011VaeConv1dFixture(
  id: Opt0011VaeConv1dFp16CaseId,
): Opt0011VaeConv1dFixture {
  const found = OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES.find(
    (candidate) => candidate.id === id,
  );
  if (found === undefined) throw new Error(`Unknown OPT-0011 fixture ${id}`);
  return found;
}

export function opt0011VaeConv1dUploadRoles(
  fixtureValue: Opt0011VaeConv1dFixture,
): readonly Opt0011VaeConv1dUploadRole[] {
  return fixtureValue.hasBias
    ? Object.freeze(["input", "weight", "bias"] as const)
    : Object.freeze(["input", "weight"] as const);
}

/**
 * Generates the exact little-endian bytes hashed by the static contract and
 * uploaded by the browser. Padding required only by WebGPU stays zero.
 */
export function opt0011VaeConv1dUploadBytes(
  id: Opt0011VaeConv1dFp16CaseId,
  role: Opt0011VaeConv1dUploadRole,
): Uint8Array {
  const fixtureValue = opt0011VaeConv1dFixture(id);
  const plan = planAceOpt0011VaeConv1dFp16(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  const elements = role === "input"
    ? plan.inputElements
    : role === "weight"
    ? plan.weightElements
    : fixtureValue.hasBias
    ? plan.outputChannels
    : 0;
  if (role === "bias" && !fixtureValue.hasBias) {
    throw new Error(`${id} has no bias upload`);
  }
  const bindingBytes = role === "input"
    ? plan.inputBindingBytes
    : role === "weight"
    ? plan.weightBindingBytes
    : plan.biasBindingBytes;
  const bytes = new Uint8Array(bindingBytes);
  for (let index = 0; index < elements; index += 1) {
    const bits = role === "input"
      ? inputBitsUnchecked(fixtureValue, index)
      : role === "weight"
      ? weightBitsUnchecked(fixtureValue, index)
      : biasBitsUnchecked(fixtureValue, index);
    bytes[index * 2] = bits & 0xff;
    bytes[index * 2 + 1] = bits >>> 8;
  }
  return bytes;
}

export function opt0011VaeConv1dInputBits(
  id: Opt0011VaeConv1dFp16CaseId,
  index: number,
): number {
  const fixtureValue = opt0011VaeConv1dFixture(id);
  const plan = planAceOpt0011VaeConv1dFp16(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  requireIndex(index, plan.inputElements, `${id} input`);
  return inputBitsUnchecked(fixtureValue, index);
}

function inputBitsUnchecked(
  fixtureValue: Opt0011VaeConv1dFixture,
  index: number,
): number {
  if (fixtureValue.id === "signed-zero") return 0x0000;
  if (fixtureValue.id === "arithmetic-subnormal-cancellation-range-edge") {
    const channel = index % fixtureValue.shape.inputChannels;
    const row = Math.floor(index / fixtureValue.shape.inputChannels);
    if (row !== 6) return 0x0000;
    return [
      0x7bff, 0x1400, 0xfbff, 0x0001, 0x0400, 0x03ff,
      0x3555, 0xb99a, 0x3c01,
    ][channel] ?? 0x0000;
  }
  return SMALL_INPUT_BITS[
    mix32(index ^ fixtureSalt(fixtureValue.id) ^ 0x1357_9bdf) %
      SMALL_INPUT_BITS.length
  ]!;
}

export function opt0011VaeConv1dWeightBits(
  id: Opt0011VaeConv1dFp16CaseId,
  index: number,
): number {
  const fixtureValue = opt0011VaeConv1dFixture(id);
  const plan = planAceOpt0011VaeConv1dFp16(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  requireIndex(index, plan.weightElements, `${id} weight`);
  return weightBitsUnchecked(fixtureValue, index);
}

function weightBitsUnchecked(
  fixtureValue: Opt0011VaeConv1dFixture,
  index: number,
): number {
  const inputChannels = fixtureValue.shape.inputChannels;
  const kernel = Math.floor(index / inputChannels) % 7;
  const outputChannel = Math.floor(index / (inputChannels * 7));
  const inputChannel = index % inputChannels;
  if (fixtureValue.id === "signed-zero") {
    return outputChannel === 0 ? 0x8000 : 0x0000;
  }
  if (fixtureValue.id === "arithmetic-subnormal-cancellation-range-edge") {
    if (kernel !== 3) return 0x0000;
    if (outputChannel === 0 && inputChannel <= 2) return 0x3c00;
    if (outputChannel === 1 && inputChannel === 3) return 0x3c00;
    if (outputChannel === 2) {
      if (inputChannel === 0) return 0x0400;
      if (inputChannel === 4) return 0x3c00;
      if (inputChannel === 5) return 0xbc00;
    }
    if (outputChannel === 3 && inputChannel >= 6 && inputChannel <= 8) {
      return [0x3555, 0x399a, 0xb555][inputChannel - 6]!;
    }
    if (outputChannel === 4) {
      if (inputChannel === 0) return 0x8400;
      if (inputChannel === 3) return 0x7bff;
      if (inputChannel === 4) return 0x3c00;
    }
    return 0x0000;
  }
  return SMALL_WEIGHT_BITS[
    mix32(index ^ fixtureSalt(fixtureValue.id) ^ 0x2468_ace0) %
      SMALL_WEIGHT_BITS.length
  ]!;
}

export function opt0011VaeConv1dBiasBits(
  id: Opt0011VaeConv1dFp16CaseId,
  outputChannel: number,
): number {
  const fixtureValue = opt0011VaeConv1dFixture(id);
  if (!fixtureValue.hasBias) throw new Error(`${id} has no bias upload`);
  requireIndex(outputChannel, fixtureValue.shape.outputChannels, `${id} bias`);
  return biasBitsUnchecked(fixtureValue, outputChannel);
}

function biasBitsUnchecked(
  fixtureValue: Opt0011VaeConv1dFixture,
  outputChannel: number,
): number {
  if (fixtureValue.id === "signed-zero") {
    return outputChannel === 0 ? 0x8000 : 0x0000;
  }
  return SMALL_BIAS_BITS[
    mix32(outputChannel ^ fixtureSalt(fixtureValue.id) ^ 0x0bad_f00d) %
      SMALL_BIAS_BITS.length
  ]!;
}

export function opt0011VaeConv1dCpuForms(
  id: Opt0011VaeConv1dFp16CaseId,
  outputIndex: number,
): Opt0011VaeConv1dCpuForms {
  const fixtureValue = opt0011VaeConv1dFixture(id);
  const plan = planAceOpt0011VaeConv1dFp16(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  requireIndex(outputIndex, plan.outputElements, `${id} output`);
  const outputChannel = outputIndex % plan.outputChannels;
  const outputRow = Math.floor(outputIndex / plan.outputChannels);
  const outputTime = outputRow % plan.outputFrames;
  const batch = Math.floor(outputRow / plan.outputFrames);
  const initial = fixtureValue.hasBias
    ? float16BitsToNumber(biasBitsUnchecked(fixtureValue, outputChannel))
    : 0;
  let separate = initial;
  let contracted = initial;
  for (let kernel = 0; kernel < 7; kernel += 1) {
    const paddedTime = outputTime + kernel * fixtureValue.shape.dilation;
    if (paddedTime < fixtureValue.shape.padding) continue;
    const inputTime = paddedTime - fixtureValue.shape.padding;
    if (inputTime >= fixtureValue.shape.inputFrames) continue;
    const inputBase = (batch * fixtureValue.shape.inputFrames + inputTime) *
      fixtureValue.shape.inputChannels;
    const weightBase = (outputChannel * 7 + kernel) *
      fixtureValue.shape.inputChannels;
    for (
      let inputChannel = 0;
      inputChannel < fixtureValue.shape.inputChannels;
      inputChannel += 1
    ) {
      const inputValue = float16BitsToNumber(
        inputBitsUnchecked(fixtureValue, inputBase + inputChannel),
      );
      const weightValue = float16BitsToNumber(
        weightBitsUnchecked(fixtureValue, weightBase + inputChannel),
      );
      const roundedProduct = Math.fround(inputValue * weightValue);
      separate = Math.fround(separate + roundedProduct);
      // FP16 operands make the product exactly representable in FP32; this
      // double expression supplies the declared one-final-rounding FMA form.
      contracted = Math.fround(contracted + inputValue * weightValue);
    }
  }
  const separateRoundedFp32 = separate;
  const oneRoundContractedFp32 = contracted;
  const separateRoundedFp32Bits = float32Bits(separateRoundedFp32);
  const oneRoundContractedFp32Bits = float32Bits(oneRoundContractedFp32);
  return Object.freeze({
    separateRoundedFp32,
    oneRoundContractedFp32,
    separateRoundedFp32Bits,
    oneRoundContractedFp32Bits,
    separateStoreBits: fixtureValue.outputStorage === "float16"
      ? numberToFloat16Bits(separateRoundedFp32)
      : separateRoundedFp32Bits,
    contractedStoreBits: fixtureValue.outputStorage === "float16"
      ? numberToFloat16Bits(oneRoundContractedFp32)
      : oneRoundContractedFp32Bits,
  });
}

export function classifyOpt0011VaeConv1dOutputBits(
  actualBits: number,
  forms: Opt0011VaeConv1dCpuForms,
): Opt0011VaeConv1dCpuClassification {
  const separate = actualBits === forms.separateStoreBits;
  const contracted = actualBits === forms.contractedStoreBits;
  if (separate && contracted) return "both-allowed-forms";
  if (separate) return "separate-rounded-only";
  if (contracted) return "one-round-contracted-only";
  return "unexpected";
}

export function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway || (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return nextExponent >= 0x1f ? sign | 0x7c00 : sign | (nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15);
}

interface ExecutionCounts {
  encodedCommandBuffers: number;
  submissions: number;
  drains: number;
  dispatches: number;
  queueEmptyIdleTurns: number;
}

interface OutputTarget {
  readonly id: "scalar" | "portable-workgroup";
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly sentinelBits: number;
  readonly sentinelWord: number;
}

interface OutputReadback {
  readonly bits: Uint16Array | Uint32Array;
  readonly sha256: string;
  readonly scan: Readonly<Record<string, unknown>>;
}

interface PreparedFixture {
  readonly fixture: Opt0011VaeConv1dFixture;
  readonly plan: AceOpt0011VaeConv1dFp16Plan;
  readonly scalar: AceOpt0011VaeConv1dFp16Dispatch;
  readonly workgroup: AceOpt0011VaeConv1dFp16Dispatch;
  readonly scalarOutput: OutputTarget;
  readonly workgroupOutput: OutputTarget;
  readonly uploadIdentity: Readonly<Record<string, unknown>>;
  readonly sourceIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  created = 0;
  destroyed = 0;
  maximumLive = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.maximumLive = Math.max(this.maximumLive, this.live.size);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      created: this.created,
      destroyed: this.destroyed,
      live: this.live.size,
      maximumLive: this.maximumLive,
    });
  }
}

if (typeof document !== "undefined") installBrowserUi();

function installBrowserUi(): void {
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing OPT-0011 start button");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("authenticating deterministic fixture uploads");
    void runBrowser().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0011-vae-conv1d-fp16-correctness-v1",
        status: "failed",
        experimentId: "OPT-0011",
        error: errorReceipt(error),
      }),
    );
  }, { once: true });
}

async function runBrowser(): Promise<unknown> {
  const actualCoreSourceSha256 = await sha256Hex(
    new TextEncoder().encode(opt0011VaeConv1dFp16CoreSource),
  );
  if (actualCoreSourceSha256 !== OPT_0011_VAE_CONV1D_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0011 rejected unauthenticated Conv1D core source");
  }
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const limits = requiredDeviceLimits();
  const rawDevice = await adapter.requestDevice({
    label: "ace-opt-0011-vae-conv1d-fp16-correctness-device",
    requiredFeatures: ["shader-f16"],
    requiredLimits: limits,
  });
  const tracker = new BufferTracker();
  const device = rawDevice;
  const uncapturedErrors: Readonly<Record<string, unknown>>[] = [];
  const runtimeErrors: Readonly<Record<string, unknown>>[] = [];
  let unexpectedDeviceLoss: Readonly<Record<string, unknown>> | null = null;
  let destroyingDevice = false;
  const onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(errorReceipt(event.error));
  };
  const onWindowError = (event: ErrorEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "error",
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    }));
  };
  const onUnhandledRejection = (event: PromiseRejectionEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "unhandledrejection",
      reason: errorReceipt(event.reason),
    }));
  };
  rawDevice.addEventListener("uncapturederror", onUncaptured);
  window.addEventListener("error", onWindowError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  void rawDevice.lost.then((info) => {
    if (!destroyingDevice) {
      unexpectedDeviceLoss = Object.freeze({
        reason: info.reason,
        message: info.message,
      });
    }
  });
  const heartbeat = startHeartbeat();
  const cases: unknown[] = [];
  let cancellation: unknown = null;
  let responsiveness: unknown;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let postCleanupValidationFailure: Error | undefined;
  try {
    for (
      let index = 0;
      index < OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES.length;
      index += 1
    ) {
      const fixtureValue = OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES[index]!;
      updateProgress(
        `correctness ${index + 1}/${OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES.length}: ${fixtureValue.id}`,
      );
      cases.push(await runFixture(device, tracker, fixtureValue));
      await yieldToBrowser();
    }
    updateProgress("post-drain cancellation and readback-prevention proof");
    cancellation = await runCancellationProof(device, tracker);
    await rawDevice.queue.onSubmittedWorkDone();
    await yieldToBrowser();
  } finally {
    const cleanupStartedAtEpochMilliseconds = Date.now();
    tracker.destroyAll();
    tracker.destroyAll();
    const harnessBuffersDestroyedAtEpochMilliseconds = Date.now();
    const receipt = tracker.receipt();
    const accountingExact =
      receipt.live === 0 && receipt.created === receipt.destroyed;
    destroyingDevice = true;
    rawDevice.destroy();
    const deviceDestroyCalledAtEpochMilliseconds = Date.now();
    const intentionalDeviceLoss = await rawDevice.lost;
    const deviceLossSettledAtEpochMilliseconds = Date.now();
    // Retain every observer through two tasks after the intentional loss
    // boundary so cleanup-time uncaptured/runtime events cannot escape.
    await yieldToBrowser();
    await yieldToBrowser();
    const postCleanupEventTurnsCompletedAtEpochMilliseconds = Date.now();
    const finalUncapturedErrors = Object.freeze([...uncapturedErrors]);
    const finalRuntimeErrors = Object.freeze([...runtimeErrors]);
    const finalUnexpectedDeviceLoss = unexpectedDeviceLoss;
    const finalEventSnapshotAtEpochMilliseconds = Date.now();
    const postCleanupValidationClean =
      accountingExact && finalUncapturedErrors.length === 0 &&
      finalRuntimeErrors.length === 0 && finalUnexpectedDeviceLoss === null &&
      intentionalDeviceLoss.reason === "destroyed";
    const postCleanupValidationAtEpochMilliseconds = Date.now();
    responsiveness = heartbeat.stop();
    const heartbeatStoppedAtEpochMilliseconds = Date.now();
    cleanup = Object.freeze({
      ...receipt,
      trackedScope: "harness-owned-buffers-only",
      kernelOwnedControlBuffers: "owned-by-idempotent-kernel-destroy-contract",
      destroyAllCalledTwice: true,
      idempotent: accountingExact,
      deviceDestroyed: true,
      intentionalDeviceLoss: Object.freeze({
        reason: intentionalDeviceLoss.reason,
        message: intentionalDeviceLoss.message,
        exactDestroyedReason: intentionalDeviceLoss.reason === "destroyed",
      }),
      uncapturedErrors: finalUncapturedErrors,
      runtimeErrors: finalRuntimeErrors,
      unexpectedDeviceLoss: finalUnexpectedDeviceLoss,
      heartbeatCoveredCleanup: true,
      heartbeat: responsiveness,
      epochs: Object.freeze({
        cleanupStartedAtEpochMilliseconds,
        harnessBuffersDestroyedAtEpochMilliseconds,
        deviceDestroyCalledAtEpochMilliseconds,
        deviceLossSettledAtEpochMilliseconds,
        postCleanupEventTurnsCompletedAtEpochMilliseconds,
        finalEventSnapshotAtEpochMilliseconds,
        postCleanupValidationAtEpochMilliseconds,
        heartbeatStoppedAtEpochMilliseconds,
      }),
    });
    rawDevice.removeEventListener("uncapturederror", onUncaptured);
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    if (!postCleanupValidationClean) {
      postCleanupValidationFailure = new Error(
        "OPT-0011 post-cleanup resource, runtime, or device-event validation failed",
      );
    }
  }
  if (postCleanupValidationFailure !== undefined) {
    throw postCleanupValidationFailure;
  }
  return Object.freeze({
    schema: "ace-opt-0011-vae-conv1d-fp16-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification: "correctness-only-no-timing-or-thermal-claim",
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter, rawDevice),
    }),
    protocol: Object.freeze({
      requiredFeature: "shader-f16",
      kernels: Object.freeze([
        "opt-0011-vae-conv1d-fp16-scalar-oracle-v1",
        "opt-0011-vae-conv1d-fp16-portable-workgroup-v1",
      ]),
      outputComparison: "full-u16-for-f16-and-full-u32-for-f32",
      oneOutstandingCommandBuffer: true,
      everyRangeSubmittedInPlanOrder: true,
      drainAfterEveryCommandBuffer: true,
      realQueueEmptyTurnBetweenRanges: true,
      compilationUploadAndWallTimeReported: false,
      performanceClaim: null,
      thermalClaim: null,
    }),
    sourceAuthority: Object.freeze({
      coreCommit: OPT_0011_VAE_CONV1D_CORE_COMMIT,
      coreSourceSha256: actualCoreSourceSha256,
      generatedShaderHashesFrozenAndVerifiedBeforeExecution: true,
    }),
    cases: Object.freeze(cases),
    cancellation,
    responsiveness,
    cleanup,
  });
}

async function runFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtureValue: Opt0011VaeConv1dFixture,
): Promise<unknown> {
  const prepared = await prepareFixture(device, tracker, fixtureValue);
  try {
    const scalarFirst = await executeAndRead(
      device,
      tracker,
      prepared.scalar,
      prepared.scalarOutput,
      true,
      "first",
    );
    const workgroupFirst = await executeAndRead(
      device,
      tracker,
      prepared.workgroup,
      prepared.workgroupOutput,
      true,
      "first",
    );
    const scalarRerun = await executeAndRead(
      device,
      tracker,
      prepared.scalar,
      prepared.scalarOutput,
      false,
      "rerun",
    );
    const workgroupRerun = await executeAndRead(
      device,
      tracker,
      prepared.workgroup,
      prepared.workgroupOutput,
      false,
      "rerun",
    );
    if (
      scalarFirst.readback.sha256 !== scalarRerun.readback.sha256 ||
      workgroupFirst.readback.sha256 !== workgroupRerun.readback.sha256
    ) {
      throw new Error(`${fixtureValue.id} output hash changed on deterministic rerun`);
    }
    const bitComparison = compareOpt0011VaeConv1dRawBits(
      scalarFirst.readback.bits,
      workgroupFirst.readback.bits,
    );
    if (bitComparison.mismatchCount !== 0) {
      const firstMismatchIndex = bitComparison.firstMismatchIndex;
      if (firstMismatchIndex === null) {
        throw new Error(`${fixtureValue.id} mismatch diagnostic lost its first index`);
      }
      const rangeIndex = prepared.plan.outputRanges.findIndex((range) =>
        firstMismatchIndex >= range.firstOutput &&
        firstMismatchIndex < range.firstOutput + range.outputCount
      );
      const range = prepared.plan.outputRanges[rangeIndex];
      if (range === undefined) {
        throw new Error(
          `${fixtureValue.id} mismatch index ${firstMismatchIndex} is outside every planned range`,
        );
      }
      const cpuForms = opt0011VaeConv1dCpuForms(
        fixtureValue.id,
        firstMismatchIndex,
      );
      throw new Error(
        `${fixtureValue.id} has ${bitComparison.mismatchCount} scalar/workgroup bit mismatches; ` +
          `firstMismatchIndex=${firstMismatchIndex}, ` +
          `scalarBits=0x${hex(scalarFirst.readback.bits[firstMismatchIndex]!, fixtureValue.outputStorage)}, ` +
          `workgroupBits=0x${hex(workgroupFirst.readback.bits[firstMismatchIndex]!, fixtureValue.outputStorage)}, ` +
          `rangeIndex=${rangeIndex}, rangeFirstOutput=${range.firstOutput}, ` +
          `rangeOutputCount=${range.outputCount}, ` +
          `rangeFirstOutputRow=${range.firstOutputRow}, ` +
          `rangeOutputRowCount=${range.outputRowCount}, ` +
          `cpuSeparateStoreBits=0x${hex(cpuForms.separateStoreBits, fixtureValue.outputStorage)}, ` +
          `cpuContractedStoreBits=0x${hex(cpuForms.contractedStoreBits, fixtureValue.outputStorage)}`,
      );
    }
    const cpuIndices = cpuOracleIndices(prepared);
    const cpuOracle = compareBothToCpu(
      fixtureValue,
      scalarFirst.readback.bits,
      workgroupFirst.readback.bits,
      cpuIndices,
    );
    return Object.freeze({
      id: fixtureValue.id,
      shape: fixtureValue.shape,
      hasBias: fixtureValue.hasBias,
      outputStorage: fixtureValue.outputStorage,
      coverage: fixtureValue.coverage,
      plan: Object.freeze({
        outputFrames: prepared.plan.outputFrames,
        outputElements: prepared.plan.outputElements,
        outputRangeCount: prepared.plan.outputRangeCount,
        outputRowsPerRange: Object.freeze(
          prepared.plan.outputRanges.map((range) => range.outputRowCount),
        ),
        workgroupStorageBytes: prepared.plan.workgroupStorageBytes,
      }),
      uploadIdentity: prepared.uploadIdentity,
      sourceIdentity: prepared.sourceIdentity,
      outputPrefill: Object.freeze({
        scalar: hex(prepared.scalarOutput.sentinelBits, fixtureValue.outputStorage),
        workgroup: hex(
          prepared.workgroupOutput.sentinelBits,
          fixtureValue.outputStorage,
        ),
        independent: true,
        quietNaN: true,
      }),
      scalar: executionReceipt(scalarFirst, scalarRerun),
      workgroup: executionReceipt(workgroupFirst, workgroupRerun),
      fullOutputBitUnit: fixtureValue.outputStorage === "float16" ? "u16" : "u32",
      fullOutputBitMismatchCount: bitComparison.mismatchCount,
      fullOutputBitIdentical: true,
      deterministicRerunHashes: true,
      cpuOracle,
      performanceClaim: null,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

async function prepareFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtureValue: Opt0011VaeConv1dFixture,
): Promise<PreparedFixture> {
  const plan = planAceOpt0011VaeConv1dFp16(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  const owned: GPUBuffer[] = [];
  const scalarKernel = AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(device);
  const workgroupKernel =
    AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(device);
  let destroyed = false;
  try {
    const scalarSource = aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
      fixtureValue.shape,
      fixtureValue.hasBias,
      fixtureValue.outputStorage,
    );
    const workgroupSource = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
      fixtureValue.shape,
      fixtureValue.hasBias,
      fixtureValue.outputStorage,
    );
    const scalarSourceSha256 = await sha256Hex(
      new TextEncoder().encode(scalarSource),
    );
    const workgroupSourceSha256 = await sha256Hex(
      new TextEncoder().encode(workgroupSource),
    );
    const expectedSource =
      OPT_0011_VAE_CONV1D_GENERATED_SHADER_SHA256[fixtureValue.id];
    if (
      scalarSourceSha256 !== expectedSource.scalar ||
      workgroupSourceSha256 !== expectedSource.workgroup
    ) {
      throw new Error(`${fixtureValue.id} generated shader SHA-256 changed`);
    }
    const uploads = new Map<Opt0011VaeConv1dUploadRole, Uint8Array>();
    const uploadHashes: Record<string, string> = {};
    for (const role of opt0011VaeConv1dUploadRoles(fixtureValue)) {
      const bytes = opt0011VaeConv1dUploadBytes(fixtureValue.id, role);
      const actual = await sha256Hex(bytes);
      const expected = (
        OPT_0011_VAE_CONV1D_UPLOAD_SHA256 as Readonly<Record<
          Opt0011VaeConv1dFp16CaseId,
          Readonly<Partial<Record<Opt0011VaeConv1dUploadRole, string>>>
        >>
      )[fixtureValue.id][role];
      if (expected === undefined || actual !== expected) {
        throw new Error(`${fixtureValue.id} ${role} upload SHA-256 changed`);
      }
      uploads.set(role, bytes);
      uploadHashes[role] = actual;
    }
    const input = createUploadBuffer(
      device,
      tracker,
      `${fixtureValue.id}-input`,
      uploads.get("input")!,
    );
    const weight = createUploadBuffer(
      device,
      tracker,
      `${fixtureValue.id}-weight`,
      uploads.get("weight")!,
    );
    owned.push(input, weight);
    const biasBytes = uploads.get("bias");
    const bias = biasBytes === undefined
      ? undefined
      : createUploadBuffer(
          device,
          tracker,
          `${fixtureValue.id}-bias`,
          biasBytes,
        );
    if (bias !== undefined) owned.push(bias);
    const scalarOutput = createOutputTarget(
      device,
      tracker,
      `${fixtureValue.id}-scalar-output`,
      plan,
      "scalar",
    );
    const workgroupOutput = createOutputTarget(
      device,
      tracker,
      `${fixtureValue.id}-workgroup-output`,
      plan,
      "portable-workgroup",
    );
    owned.push(scalarOutput.buffer, workgroupOutput.buffer);
    const common = {
      input: binding(input, plan.inputBindingBytes),
      weight: binding(weight, plan.weightBindingBytes),
      ...(bias === undefined
        ? {}
        : { bias: binding(bias, plan.biasBindingBytes) }),
    };
    const scalarBindings: AceOpt0011VaeConv1dFp16Bindings = {
      ...common,
      output: scalarOutput.binding,
    };
    const workgroupBindings: AceOpt0011VaeConv1dFp16Bindings = {
      ...common,
      output: workgroupOutput.binding,
    };
    const scalar = await scalarKernel.createDispatch(
      `${fixtureValue.id}-scalar`,
      fixtureValue.shape,
      scalarBindings,
      fixtureValue.outputStorage,
    );
    const workgroup = await workgroupKernel.createDispatch(
      `${fixtureValue.id}-workgroup`,
      fixtureValue.shape,
      workgroupBindings,
      fixtureValue.outputStorage,
    );
    if (scalar.rangeCount !== plan.outputRangeCount || workgroup.rangeCount !== plan.outputRangeCount) {
      throw new Error(`${fixtureValue.id} compiled range topology changed`);
    }
    return Object.freeze({
      fixture: fixtureValue,
      plan,
      scalar,
      workgroup,
      scalarOutput,
      workgroupOutput,
      uploadIdentity: Object.freeze({
        recipe: "explicit-little-endian-f16-binding-bytes-v1",
        hashes: Object.freeze(uploadHashes),
        bindingBytes: Object.freeze(Object.fromEntries(
          [...uploads].map(([role, bytes]) => [role, bytes.byteLength]),
        )),
      }),
      sourceIdentity: Object.freeze({
        coreCommit: OPT_0011_VAE_CONV1D_CORE_COMMIT,
        coreSourceSha256: OPT_0011_VAE_CONV1D_CORE_SOURCE_SHA256,
        scalarSha256: scalarSourceSha256,
        workgroupSha256: workgroupSourceSha256,
        scalarBytes: new TextEncoder().encode(scalarSource).byteLength,
        workgroupBytes: new TextEncoder().encode(workgroupSource).byteLength,
      }),
      destroy(): void {
        scalarKernel.destroy();
        workgroupKernel.destroy();
        if (destroyed) return;
        destroyed = true;
        for (const buffer of owned) tracker.destroy(buffer);
      },
    });
  } catch (error) {
    scalarKernel.destroy();
    workgroupKernel.destroy();
    for (const buffer of owned) tracker.destroy(buffer);
    throw error;
  }
}

async function executeAndRead(
  device: GPUDevice,
  tracker: BufferTracker,
  dispatch: AceOpt0011VaeConv1dFp16Dispatch,
  output: OutputTarget,
  retainBits: boolean,
  runLabel: string,
): Promise<{
  readonly execution: Readonly<ExecutionCounts & { readonly fifoRangeIndices: readonly number[] }>;
  readonly readback: OutputReadback;
}> {
  prefillOutput(device, output);
  await device.queue.onSubmittedWorkDone();
  const execution = await executeAllRanges(device, dispatch, runLabel);
  const readback = await readOutput(device, tracker, dispatch.plan, output, retainBits);
  return Object.freeze({ execution, readback });
}

async function executeAllRanges(
  device: GPUDevice,
  dispatch: AceOpt0011VaeConv1dFp16Dispatch,
  runLabel: string,
): Promise<Readonly<ExecutionCounts & { readonly fifoRangeIndices: readonly number[] }>> {
  const counts: ExecutionCounts = {
    encodedCommandBuffers: 0,
    submissions: 0,
    drains: 0,
    dispatches: 0,
    queueEmptyIdleTurns: 0,
  };
  const fifoRangeIndices: number[] = [];
  for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
    const encoder = device.createCommandEncoder({
      label: `${dispatch.label}-${runLabel}-range-${rangeIndex}-encoder`,
    });
    const pass = encoder.beginComputePass({
      label: `${dispatch.label}-${runLabel}-range-${rangeIndex}-pass`,
    });
    dispatch.encodeRange(pass, rangeIndex);
    pass.end();
    counts.encodedCommandBuffers += 1;
    counts.dispatches += 1;
    fifoRangeIndices.push(rangeIndex);
    device.queue.submit([encoder.finish()]);
    counts.submissions += 1;
    await device.queue.onSubmittedWorkDone();
    counts.drains += 1;
    if (rangeIndex + 1 < dispatch.rangeCount) {
      await queueEmptyIdleTurn();
      counts.queueEmptyIdleTurns += 1;
    }
  }
  if (
    counts.encodedCommandBuffers !== dispatch.rangeCount ||
    counts.submissions !== dispatch.rangeCount ||
    counts.drains !== dispatch.rangeCount ||
    counts.dispatches !== dispatch.rangeCount ||
    counts.queueEmptyIdleTurns !== Math.max(0, dispatch.rangeCount - 1) ||
    fifoRangeIndices.some((value, index) => value !== index)
  ) {
    throw new Error(`${dispatch.label} FIFO execution accounting failed`);
  }
  return Object.freeze({ ...counts, fifoRangeIndices: Object.freeze(fifoRangeIndices) });
}

async function runCancellationProof(
  device: GPUDevice,
  tracker: BufferTracker,
): Promise<unknown> {
  const fixtureValue = opt0011VaeConv1dFixture("fifo-cancellation-two-range");
  const prepared = await prepareFixture(device, tracker, fixtureValue);
  try {
    if (prepared.workgroup.rangeCount !== 2) {
      throw new Error("OPT-0011 cancellation fixture must have exactly two ranges");
    }
    prefillOutput(device, prepared.workgroupOutput);
    await device.queue.onSubmittedWorkDone();
    const controller = new AbortController();
    let encodeCount = 0;
    let submitCount = 0;
    let drainCount = 0;
    let readbackCount = 0;
    let skippedRangeCount = 0;
    let idleTurnDelivered = false;
    for (let rangeIndex = 0; rangeIndex < prepared.workgroup.rangeCount; rangeIndex += 1) {
      if (controller.signal.aborted) {
        skippedRangeCount += 1;
        continue;
      }
      const encoder = device.createCommandEncoder({
        label: `opt-0011-cancel-range-${rangeIndex}-encoder`,
      });
      const pass = encoder.beginComputePass();
      prepared.workgroup.encodeRange(pass, rangeIndex);
      pass.end();
      encodeCount += 1;
      device.queue.submit([encoder.finish()]);
      submitCount += 1;
      await device.queue.onSubmittedWorkDone();
      drainCount += 1;
      await queueEmptyIdleTurn();
      idleTurnDelivered = true;
      controller.abort("cancel-after-first-drained-range-and-real-idle");
    }
    if (!controller.signal.aborted) {
      readbackCount += 1;
      await readOutput(
        device,
        tracker,
        prepared.plan,
        prepared.workgroupOutput,
        false,
      );
    }
    if (
      !controller.signal.aborted || !idleTurnDelivered || encodeCount !== 1 ||
      submitCount !== 1 || drainCount !== 1 || readbackCount !== 0 ||
      skippedRangeCount !== 1
    ) {
      throw new Error("OPT-0011 cancellation did not stop later GPU work and readback");
    }
    return Object.freeze({
      fixtureId: fixtureValue.id,
      plannedRangeCount: prepared.workgroup.rangeCount,
      cancellationPoint: "after-first-drained-range-and-real-queue-empty-idle",
      encodeCount,
      submitCount,
      drainCount,
      readbackCount,
      skippedRangeCount,
      signalAborted: true,
      realQueueEmptyIdleTurnDelivered: true,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
      readbackPrevented: true,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

function compareBothToCpu(
  fixtureValue: Opt0011VaeConv1dFixture,
  scalar: Uint16Array | Uint32Array,
  workgroup: Uint16Array | Uint32Array,
  indices: readonly number[],
): unknown {
  const scalarMetrics = newCpuMetrics(indices.length);
  const workgroupMetrics = newCpuMetrics(indices.length);
  for (const index of indices) {
    const forms = opt0011VaeConv1dCpuForms(fixtureValue.id, index);
    addCpuMetric(scalarMetrics, scalar[index]!, forms, fixtureValue.outputStorage);
    addCpuMetric(workgroupMetrics, workgroup[index]!, forms, fixtureValue.outputStorage);
  }
  const scalarReceipt = finishCpuMetrics(scalarMetrics);
  const workgroupReceipt = finishCpuMetrics(workgroupMetrics);
  for (const [arm, receipt] of [
    ["scalar", scalarReceipt],
    ["workgroup", workgroupReceipt],
  ] as const) {
    if (
      receipt.unexpectedClassificationCount !== 0 ||
      receipt.nonFiniteCount !== 0 || receipt.saturationCount !== 0
    ) {
      throw new Error(`${fixtureValue.id} ${arm} failed CPU oracle classification`);
    }
  }
  return Object.freeze({
    arithmeticOrder: "kernel-outer-then-increasing-input-channel",
    operands: "exact-f16-bits-expanded-to-f32",
    allowedForms: Object.freeze([
      "separate-rounded-product-plus-add",
      "one-round-contracted-expression",
    ]),
    outputBoundary: fixtureValue.outputStorage === "float16"
      ? "exact-rne-f16-store-including-signed-zero"
      : "exact-f32-store-including-signed-zero",
    scope: fixtureValue.cpuOracleScope,
    comparedOutputCount: indices.length,
    comparedIndicesSha256Role: fixtureValue.cpuOracleScope === "complete-output"
      ? "contiguous-complete-output"
      : "first-middle-last-and-every-range-boundary",
    scalar: scalarReceipt,
    workgroup: workgroupReceipt,
  });
}

interface MutableCpuMetrics {
  readonly comparedOutputCount: number;
  bothAllowedCount: number;
  separateOnlyCount: number;
  contractedOnlyCount: number;
  unexpectedClassificationCount: number;
  nonFiniteCount: number;
  saturationCount: number;
  signedZeroCount: number;
  maximumAbsoluteErrorToSeparateFp32: number;
  maximumRelativeErrorToSeparateFp32: number;
  squaredErrorToSeparateFp32: number;
}

function newCpuMetrics(comparedOutputCount: number): MutableCpuMetrics {
  return {
    comparedOutputCount,
    bothAllowedCount: 0,
    separateOnlyCount: 0,
    contractedOnlyCount: 0,
    unexpectedClassificationCount: 0,
    nonFiniteCount: 0,
    saturationCount: 0,
    signedZeroCount: 0,
    maximumAbsoluteErrorToSeparateFp32: 0,
    maximumRelativeErrorToSeparateFp32: 0,
    squaredErrorToSeparateFp32: 0,
  };
}

function addCpuMetric(
  metrics: MutableCpuMetrics,
  actualBits: number,
  forms: Opt0011VaeConv1dCpuForms,
  storage: AceOpt0011VaeConv1dFp16OutputStorage,
): void {
  const classification = classifyOpt0011VaeConv1dOutputBits(actualBits, forms);
  switch (classification) {
    case "both-allowed-forms": metrics.bothAllowedCount += 1; break;
    case "separate-rounded-only": metrics.separateOnlyCount += 1; break;
    case "one-round-contracted-only": metrics.contractedOnlyCount += 1; break;
    case "unexpected": metrics.unexpectedClassificationCount += 1; break;
  }
  const actual = storage === "float16"
    ? float16BitsToNumber(actualBits)
    : float32FromBits(actualBits);
  if (!Number.isFinite(actual)) metrics.nonFiniteCount += 1;
  if (
    storage === "float16"
      ? (actualBits & 0x7fff) === 0x7bff
      : (actualBits & 0x7fff_ffff) === 0x7f7f_ffff
  ) metrics.saturationCount += 1;
  if (actual === 0) metrics.signedZeroCount += 1;
  if (Number.isFinite(actual) && Number.isFinite(forms.separateRoundedFp32)) {
    const absolute = Math.abs(actual - forms.separateRoundedFp32);
    const relative = absolute / Math.max(Math.abs(forms.separateRoundedFp32), 1e-30);
    metrics.maximumAbsoluteErrorToSeparateFp32 = Math.max(
      metrics.maximumAbsoluteErrorToSeparateFp32,
      absolute,
    );
    metrics.maximumRelativeErrorToSeparateFp32 = Math.max(
      metrics.maximumRelativeErrorToSeparateFp32,
      relative,
    );
    metrics.squaredErrorToSeparateFp32 += absolute * absolute;
  }
}

function finishCpuMetrics(metrics: MutableCpuMetrics): Readonly<MutableCpuMetrics & {
  readonly rootMeanSquareErrorToSeparateFp32: number;
}> {
  return Object.freeze({
    ...metrics,
    rootMeanSquareErrorToSeparateFp32: Math.sqrt(
      metrics.squaredErrorToSeparateFp32 /
        Math.max(1, metrics.comparedOutputCount),
    ),
  });
}

function cpuOracleIndices(prepared: PreparedFixture): readonly number[] {
  if (prepared.fixture.cpuOracleScope === "complete-output") {
    return Object.freeze(Array.from(
      { length: prepared.plan.outputElements },
      (_, index) => index,
    ));
  }
  const indices = new Set<number>([
    0,
    Math.floor(prepared.plan.outputElements / 2),
    prepared.plan.outputElements - 1,
  ]);
  for (const range of prepared.plan.outputRanges) {
    indices.add(range.firstOutput);
    indices.add(range.firstOutput + range.outputCount - 1);
  }
  if (prepared.fixture.id === "fifo-cancellation-two-range") {
    indices.add(FIFO_EXACT_ZERO_CANONICALIZATION_INDEX);
  }
  return Object.freeze([...indices].sort((left, right) => left - right));
}

function executionReceipt(
  first: Awaited<ReturnType<typeof executeAndRead>>,
  rerun: Awaited<ReturnType<typeof executeAndRead>>,
): unknown {
  return Object.freeze({
    firstExecution: first.execution,
    rerunExecution: rerun.execution,
    firstScan: first.readback.scan,
    rerunScan: rerun.readback.scan,
    firstSha256: first.readback.sha256,
    rerunSha256: rerun.readback.sha256,
    deterministic: first.readback.sha256 === rerun.readback.sha256,
  });
}

function createUploadBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: Uint8Array,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    new Uint8Array(buffer.getMappedRange()).set(bytes);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createOutputTarget(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceOpt0011VaeConv1dFp16Plan,
  id: OutputTarget["id"],
): OutputTarget {
  const size = OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES;
  const buffer = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const sentinelBits = plan.outputStorage === "float16"
    ? id === "scalar" ? SCALAR_F16_SENTINEL : WORKGROUP_F16_SENTINEL
    : id === "scalar" ? SCALAR_F32_SENTINEL : WORKGROUP_F32_SENTINEL;
  const sentinelWord = plan.outputStorage === "float16"
    ? sentinelBits | (sentinelBits << 16)
    : sentinelBits;
  try {
    const words = new Uint32Array(buffer.getMappedRange());
    words.fill(OUTPUT_GUARD_WORD);
    words.fill(
      sentinelWord,
      OUTPUT_GUARD_BYTES / 4,
      (OUTPUT_GUARD_BYTES + plan.outputBindingBytes) / 4,
    );
    buffer.unmap();
    return Object.freeze({
      id,
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OUTPUT_GUARD_BYTES,
        size: plan.outputBindingBytes,
      }),
      sentinelBits,
      sentinelWord,
    });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function prefillOutput(device: GPUDevice, target: OutputTarget): void {
  const words = new Uint32Array((target.binding.size ?? 0) / 4);
  words.fill(target.sentinelWord);
  device.queue.writeBuffer(
    target.buffer,
    target.binding.offset ?? 0,
    words,
  );
}

async function readOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  plan: AceOpt0011VaeConv1dFp16Plan,
  target: OutputTarget,
  retainBits: boolean,
): Promise<OutputReadback> {
  const bytes = OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES;
  const readback = tracker.create(device, {
    label: `${target.id}-readback`,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({ label: `${target.id}-readback-encoder` });
    encoder.copyBufferToBuffer(target.buffer, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const range = readback.getMappedRange();
    const prefix = new Uint32Array(range, 0, OUTPUT_GUARD_BYTES / 4);
    const suffix = new Uint32Array(
      range,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
      OUTPUT_GUARD_BYTES / 4,
    );
    const guardUntouched = [...prefix, ...suffix].every(
      (word) => word === OUTPUT_GUARD_WORD,
    );
    const elementBytes = plan.outputStorage === "float16" ? 2 : 4;
    const raw = new Uint8Array(
      range,
      OUTPUT_GUARD_BYTES,
      plan.outputStorageBytes,
    ).slice();
    const bits = plan.outputStorage === "float16"
      ? new Uint16Array(raw.buffer, raw.byteOffset, plan.outputElements)
      : new Uint32Array(raw.buffer, raw.byteOffset, plan.outputElements);
    const bindingScalars = plan.outputBindingBytes / elementBytes;
    const padded = plan.outputStorage === "float16"
      ? new Uint16Array(range, OUTPUT_GUARD_BYTES, bindingScalars)
      : new Uint32Array(range, OUTPUT_GUARD_BYTES, bindingScalars);
    let paddingUntouched = true;
    for (let index = plan.outputElements; index < padded.length; index += 1) {
      if (padded[index] !== target.sentinelBits) paddingUntouched = false;
    }
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let sentinelCount = 0;
    let saturationCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let subnormalCount = 0;
    for (let index = 0; index < bits.length; index += 1) {
      const rawBits = bits[index]!;
      const value = plan.outputStorage === "float16"
        ? float16BitsToNumber(rawBits)
        : float32FromBits(rawBits);
      if (Number.isFinite(value)) finiteCount += 1;
      else nonFiniteCount += 1;
      if (rawBits === target.sentinelBits) sentinelCount += 1;
      if (
        plan.outputStorage === "float16"
          ? (rawBits & 0x7fff) === 0x7bff
          : (rawBits & 0x7fff_ffff) === 0x7f7f_ffff
      ) saturationCount += 1;
      if (value === 0) {
        const negative = plan.outputStorage === "float16"
          ? (rawBits & 0x8000) !== 0
          : (rawBits & 0x8000_0000) !== 0;
        if (negative) negativeZeroCount += 1;
        else positiveZeroCount += 1;
      }
      if (
        plan.outputStorage === "float16" &&
        (rawBits & 0x7c00) === 0 && (rawBits & 0x03ff) !== 0
      ) subnormalCount += 1;
    }
    if (
      !guardUntouched || !paddingUntouched || finiteCount !== plan.outputElements ||
      nonFiniteCount !== 0 || sentinelCount !== 0 || saturationCount !== 0
    ) {
      throw new Error(`${target.id} output failed complete finite guarded write checks`);
    }
    const retained = retainBits
      ? plan.outputStorage === "float16"
        ? new Uint16Array(bits)
        : new Uint32Array(bits)
      : plan.outputStorage === "float16"
      ? new Uint16Array(0)
      : new Uint32Array(0);
    return Object.freeze({
      bits: retained,
      sha256: await sha256Hex(raw),
      scan: Object.freeze({
        elementCount: plan.outputElements,
        finiteCount,
        nonFiniteCount,
        sentinelCount,
        saturationCount,
        positiveZeroCount,
        negativeZeroCount,
        subnormalCount,
        externalPrefixGuardUntouched: guardUntouched,
        externalSuffixGuardUntouched: guardUntouched,
        bindingPaddingSentinelUntouched: paddingUntouched,
        completeOutputReadback: true,
      }),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

export function compareOpt0011VaeConv1dRawBits(
  left: Uint16Array | Uint32Array,
  right: Uint16Array | Uint32Array,
): Opt0011VaeConv1dRawBitComparison {
  if (left.constructor !== right.constructor || left.length !== right.length) {
    throw new Error("OPT-0011 output bit domains differ");
  }
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      mismatchCount += 1;
      firstMismatchIndex ??= index;
    }
  }
  return Object.freeze({ mismatchCount, firstMismatchIndex });
}

function requiredDeviceLimits(): Record<string, number> {
  let maximumBinding = 0;
  let maximumBuffer = 0;
  let maximumStorage = 0;
  for (const fixtureValue of OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES) {
    const plan = planAceOpt0011VaeConv1dFp16(
      fixtureValue.shape,
      fixtureValue.outputStorage,
    );
    maximumBinding = Math.max(
      maximumBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      maximumBinding,
      OUTPUT_GUARD_BYTES * 2 + plan.outputBindingBytes,
    );
    maximumStorage = Math.max(maximumStorage, plan.workgroupStorageBytes);
  }
  return {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumBinding,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 16,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupStorageSize: maximumStorage,
  };
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("OPT-0011 requires adapter feature shader-f16");
  }
  const required = requiredDeviceLimits();
  for (const [name, value] of Object.entries(required)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < value) {
      throw new Error(`OPT-0011 adapter limit ${name} is ${actual}, requires ${value}`);
    }
  }
}

function adapterReceipt(adapter: GPUAdapter, device: GPUDevice): unknown {
  return Object.freeze({
    features: Object.freeze([...adapter.features].sort()),
    info: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    requiredLimits: Object.freeze(requiredDeviceLimits()),
    actualAdapterLimits: gpuLimitReceipt(adapter.limits),
    actualRequestedDeviceLimits: gpuLimitReceipt(device.limits),
  });
}

function gpuLimitReceipt(limits: GPUSupportedLimits): unknown {
  return Object.freeze({
    maxBufferSize: Number(limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize),
    maxComputeInvocationsPerWorkgroup:
      limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension:
      limits.maxComputeWorkgroupsPerDimension,
    minStorageBufferOffsetAlignment: limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
  });
}

function startHeartbeat(): { stop(): Readonly<Record<string, unknown>> } {
  const startedAtEpochMilliseconds = Date.now();
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let animationFrame = 0;
  const animate = (now: number): void => {
    if (stopped) return;
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrameCount += 1;
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, now - lastTimer);
    lastTimer = now;
    timerTickCount += 1;
  }, 16);
  let receipt: Readonly<Record<string, unknown>> | undefined;
  return Object.freeze({
    stop(): Readonly<Record<string, unknown>> {
      if (receipt !== undefined) return receipt;
      stopped = true;
      cancelAnimationFrame(animationFrame);
      window.clearInterval(timer);
      receipt = Object.freeze({
        startedAtEpochMilliseconds,
        stoppedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
        observed: animationFrameCount > 0 && timerTickCount > 0,
      });
      return receipt;
    },
  });
}

async function queueEmptyIdleTurn(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, QUEUE_EMPTY_IDLE_MILLISECONDS);
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function shape(
  batch: number,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  dilation: 1 | 3 | 9,
): AceVaeConv1dShape {
  return Object.freeze({
    batch,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: dilation * 3,
  });
}

function fixture(
  id: Opt0011VaeConv1dFp16CaseId,
  fixtureShape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
  cpuOracleScope: Opt0011VaeConv1dFixture["cpuOracleScope"],
  coverage: readonly string[],
): Opt0011VaeConv1dFixture {
  return Object.freeze({
    id,
    shape: fixtureShape,
    hasBias,
    outputStorage,
    cpuOracleScope,
    coverage: Object.freeze([...coverage]),
  });
}

function fixtureSalt(id: Opt0011VaeConv1dFp16CaseId): number {
  const ordinal = (() => {
    switch (id) {
      case "d1-b2-f35-c65-c13-bias": return 1;
      case "d3-b3-f51-c64-c11-no-bias": return 2;
      case "d9-b2-f67-c63-c9-bias": return 3;
      case "signed-zero": return 4;
      case "arithmetic-subnormal-cancellation-range-edge": return 5;
      case "long-cin1024": return 6;
      case "final-b1-f4097-c128-c2-f32": return 7;
      case "fifo-cancellation-two-range": return 8;
      case "production-block0-d1-c1024": return 9;
    }
  })();
  return Math.imul(ordinal, 0x9e37_79b1) >>> 0;
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x7feb_352d) >>> 0;
  mixed ^= mixed >>> 15;
  mixed = Math.imul(mixed, 0x846c_a68b) >>> 0;
  return (mixed ^ (mixed >>> 16)) >>> 0;
}

function requireIndex(index: number, length: number, label: string): void {
  if (!Number.isSafeInteger(index) || index < 0 || index >= length) {
    throw new RangeError(`${label} index is outside [0, ${length})`);
  }
}

function float32Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  return UINT32_SCRATCH[0]!;
}

function float32FromBits(bits: number): number {
  UINT32_SCRATCH[0] = bits >>> 0;
  return FLOAT32_SCRATCH[0]!;
}

function hex(
  value: number,
  storage: AceOpt0011VaeConv1dFp16OutputStorage,
): string {
  return value.toString(16).padStart(storage === "float16" ? 4 : 8, "0");
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      ...(error.stack === undefined ? {} : { stack: error.stack }),
    });
  }
  return Object.freeze({ name: "Error", message: String(error) });
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = status;
  const output = document.querySelector<HTMLElement>("#result");
  if (output !== null) output.textContent = JSON.stringify(result, null, 2);
}
