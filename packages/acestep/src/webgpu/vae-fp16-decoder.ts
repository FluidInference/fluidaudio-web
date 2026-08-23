import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./kernels/correctness-utils.js";
import {
  ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
  AceFp16VaeConv1dKernel,
} from "./kernels/vae-conv1d-fp16.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
} from "./kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
  AceOpt0025VaeK1SubgroupGemmKernel,
} from "./kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
  AceOpt0028VaeK1PortablePackedKernel,
} from "./kernels/vae-k1-fp16-portable-packed.js";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
  AceFp16VaeConvTranspose1dKernel,
  type AceFp16VaeConvTranspose1dKernelId,
} from "./kernels/vae-conv-transpose1d-fp16.js";
import {
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
  AceOpt0026VaeConvTranspose1dKernel,
} from "./kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
  AceOpt0028VaeConvTranspose1dPortablePackedKernel,
} from "./kernels/vae-conv-transpose1d-fp16-portable-packed.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
} from "./kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0040VaeConvTranspose1dShapeSelectorKernel,
} from "./kernels/vae-conv-transpose1d-fp16-shape-selector.js";
import {
  ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel,
  AceOpt0088VaeConvTranspose1dK4PortableShapeSelectorKernel,
} from "./kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
} from "./kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0057VaeK7ShapeSelectorKernel,
  AceOpt0088VaeK7PortableShapeSelectorKernel,
} from "./kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID,
} from "./kernels/vae-conv1d-fp16-k4-row-reuse-portable.js";
import {
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
} from "./kernels/vae-conv-transpose1d-fp16-k4-portable.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from "./kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
  AceFp16VaePointwiseKernel,
} from "./kernels/vae-pointwise-fp16.js";
import {
  ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
  AceFp16VaeSnakeKernel,
} from "./kernels/vae-snake-fp16.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  type AceOpt0011VaeOperationBindings,
  type AceOpt0011VaePackageBindings,
} from "./vae-fp16-package.js";
import {
  ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_K1_FP16_TILE_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
} from "../model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../model/package.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256,
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
  ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
  ACE_OPT_0011_VAE_WINDOW_FRAMES,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  hashAceVaePrecisionMap,
  requireAceOpt0011VaeDecoderGeometry,
  type AceVaePrecisionMapEntry,
  type AceVaePrecisionMap,
  type AceVaeRuntimeProfileId,
} from "./vae-fp16-profile.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderCooperativePlan,
  type AceVaeDecoderGraphPlan,
  type AceVaeDecoderOperation,
  type AceVaeDecoderQuantumPlan,
  type AceVaeDecoderSlot,
} from "./vae-decoder.js";
import {
  ACE_VAE_DEFAULT_OVERLAP_FRAMES,
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
} from "./vae-chunks.js";

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_WGSL_DYNAMIC_OFFSET = 0xffff_ffff;

export const ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT = 88;
export const ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT = 3_942;
export const ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT = 3_943;
export const ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES = 16;
export const ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER = 8;
export const ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT = 7_854;
export const ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT = 7_855;
export const ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES = 251_658_240;
export const ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES = 2_010_640;
export const ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8 = 983;
/** @internal OPT-0035 benchmark-only maximum-window profile. */
export const ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES = 2_378;
export const ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES = 1_168_834_560;
/** Capped C2378-family geometry for one-GiB maxBufferSize adapters (iOS). */
export const ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES = 2_176;
export const ACE_CAPPED_VAE_FP16_C2176_WORKSPACE_BYTES = 1_069_547_520;

export type AceOpt0011Fp16VaeMaximumWindowFrames =
  | 256
  | 512
  | typeof ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES
  | typeof ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES;

export type AceOpt0011Fp16VaeDecoderKernelId =
  | typeof ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
  | typeof ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID
  | typeof ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  | typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID
  | typeof ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID
  | typeof ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID
  | typeof ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID
  | typeof ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
  | typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
  | typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID
  | typeof ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID
  | typeof ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID
  | typeof ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID;

export type AceOpt0011Fp16VaeDecoderKernelSetId =
  | typeof ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID
  | typeof ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID
  | typeof ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID
  | typeof ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID
  | typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID
  | typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID
  | typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID
  | typeof ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID
  | typeof ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID;

export interface AceOpt0011Fp16VaeDecoderKernelTopology {
  readonly id: AceOpt0011Fp16VaeDecoderKernelSetId;
  readonly backend:
    | "portable-workgroup"
    | "portable-workgroup-exact-packed"
    | "fixed32-subgroup-k7-hybrid"
    | "fixed32-subgroup-k7-congruent-transpose-hybrid"
    | "fixed32-subgroup-exact-packed"
    | "fixed32-subgroup-exact-packed-shape-selected"
    | "fixed32-subgroup-revision7-k4-shape-selected"
    | "fixed32-subgroup-dual-k4-quality"
    | "portable-workgroup-dual-k4";
  readonly ingress: typeof ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID;
  readonly conv1dK1:
    | typeof ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID
    | typeof ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID
    | typeof ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID;
  readonly conv1dK7:
    | typeof ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID
    | typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
    | typeof ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID
    | typeof ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID;
  readonly convTranspose1d:
    | AceFp16VaeConvTranspose1dKernelId
    | typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID
    | typeof ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID
    | typeof ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID
    | typeof ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID
    | typeof ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID;
  readonly snake: typeof ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID;
  readonly add: typeof ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID;
}

export const ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
    backend: "portable-workgroup",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    convTranspose1d: ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
    backend: "fixed32-subgroup-k7-hybrid",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    convTranspose1d: ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
    backend: "portable-workgroup-exact-packed",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    convTranspose1d:
      ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
    backend: "fixed32-subgroup-k7-congruent-transpose-hybrid",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    convTranspose1d: ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
    backend: "fixed32-subgroup-exact-packed",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    convTranspose1d: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
    backend: "fixed32-subgroup-exact-packed-shape-selected",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
    conv1dK7: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    convTranspose1d:
      ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID,
    backend: "fixed32-subgroup-revision7-k4-shape-selected",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
    conv1dK7: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    convTranspose1d:
      ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
    backend: "fixed32-subgroup-dual-k4-quality",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
    conv1dK7: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    convTranspose1d:
      ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export const ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY:
  Readonly<AceOpt0011Fp16VaeDecoderKernelTopology> = Object.freeze({
    id: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
    backend: "portable-workgroup-dual-k4",
    ingress: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    conv1dK1: ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
    conv1dK7: ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
    convTranspose1d:
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
    snake: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    add: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  });

export type AceOpt0011Fp16VaeDecoderRuntimeOptions =
  | Readonly<{
      readonly runtimeProfileId?: "opt-0011-mixed-fp16-portable-v1";
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0028-mixed-fp16-portable-exact-packed-v1";
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0028-mixed-fp16-fixed32-exact-packed-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0054-mixed-fp16-fixed32-revision7-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1";
      readonly subgroupMinSize: 32;
      readonly subgroupMaxSize: 32;
    }>
  | Readonly<{
      readonly runtimeProfileId:
        "opt-0088-mixed-fp16-portable-dual-k4-v1";
    }>;

export interface AceOpt0011Fp16VaeWindowBindings {
  /** FP32 NLC latent window `[1,inputFrames,64]`. */
  readonly stagingInput: GPUBufferBinding;
  /** Dedicated FP16 NLC decoder ingress `[1,inputFrames,64]`. */
  readonly decoderInput: GPUBufferBinding;
  /** Three disjoint FP16 activation workspaces. */
  readonly workspaces: readonly [
    GPUBufferBinding,
    GPUBufferBinding,
    GPUBufferBinding,
  ];
  /** FP32 interleaved raw waveform `[1,inputFrames*1920,2]`. */
  readonly output: GPUBufferBinding;
  /** Exact authenticated, unsharded revision-5 VAE package bindings. */
  readonly package: AceOpt0011VaePackageBindings;
}

/** The original exact B-256 binding seam. */
export interface AceOpt0011Fp16VaeDecoderBindings
  extends AceOpt0011Fp16VaeWindowBindings {}

export interface AceOpt0011Fp16VaeDecoderControlRecord {
  readonly recordIndex: number;
  readonly sequenceIndex: number;
  readonly graphQuantumIndex: number | null;
  readonly operationIndex: number | null;
  readonly operationLabel: string;
  readonly operationKind:
    | "ingress-cast"
    | AceVaeDecoderOperation["kind"];
  readonly outputBase: number;
  readonly outputCount: number;
  readonly byteOffset: number;
}

export interface AceOpt0011Fp16VaeWindowDynamicControlPlan {
  readonly recordBytes:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES;
  readonly recordAlignment: number;
  readonly recordCount: number;
  readonly byteLength: number;
  readonly records: readonly AceOpt0011Fp16VaeDecoderControlRecord[];
}

export interface AceOpt0011Fp16VaeDecoderDynamicControlPlan
  extends AceOpt0011Fp16VaeWindowDynamicControlPlan {
  readonly recordCount:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT;
}

export interface AceOpt0011Fp16VaeDecoderQuantum {
  readonly sequenceIndex: number;
  readonly graphQuantumIndex: number | null;
  readonly operationIndex: number | null;
  readonly operationLabel: string;
  readonly operationKind:
    | "ingress-cast"
    | AceVaeDecoderOperation["kind"];
  readonly logicalOutputBase: number;
  readonly logicalOutputCount: number;
  readonly estimatedMaximumMultiplyAccumulates: number;
  readonly primitiveCount: 1;
  readonly kernelId: AceOpt0011Fp16VaeDecoderKernelId;
  readonly control: AceOpt0011Fp16VaeDecoderControlRecord;
  readonly precision: AceVaePrecisionMapEntry;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0011Fp16VaeWindowDispatch {
  readonly label: string;
  readonly runtimeProfileId: Extract<
    AceVaeRuntimeProfileId,
    | "opt-0011-mixed-fp16-portable-v1"
    | "opt-0028-mixed-fp16-portable-exact-packed-v1"
    | "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
    | "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
    | "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
    | "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
    | "opt-0054-mixed-fp16-fixed32-revision7-v1"
    | "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
    | "opt-0088-mixed-fp16-portable-dual-k4-v1"
  >;
  readonly kernelSetId: AceOpt0011Fp16VaeDecoderKernelSetId;
  readonly kernelTopology: AceOpt0011Fp16VaeDecoderKernelTopology;
  readonly plan: AceVaeDecoderGraphPlan;
  readonly cooperativePlan: AceVaeDecoderCooperativePlan;
  /** Canonical hash-bearing arithmetic contract; entries are frame-neutral. */
  readonly precisionMap: AceVaePrecisionMap;
  readonly dynamicControls: AceOpt0011Fp16VaeWindowDynamicControlPlan;
  readonly ingressQuantum: AceOpt0011Fp16VaeDecoderQuantum;
  readonly graphQuanta: readonly AceOpt0011Fp16VaeDecoderQuantum[];
  /** Ingress followed by every graph quantum in strict FIFO order. */
  readonly quanta: readonly AceOpt0011Fp16VaeDecoderQuantum[];
  readonly operationCount:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT;
  readonly graphQuantumCount: number;
  readonly primitiveCount: number;
  /** Exact active prefix sizes inside the shared maximum-sized bindings. */
  readonly activeStagingInputBytes: number;
  readonly activeDecoderInputBytes: number;
  readonly activeOutputBytes: number;
  readonly decoderCommandBufferCountAtBatch8: number;
  /** Bounded decoder batches followed by one output-copy command buffer. */
  readonly commandBufferCountAtBatch8: number;
}

export interface AceOpt0011Fp16VaeDecoderDispatch
  extends AceOpt0011Fp16VaeWindowDispatch {
  readonly dynamicControls: AceOpt0011Fp16VaeDecoderDynamicControlPlan;
  readonly graphQuantumCount:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT;
  readonly primitiveCount:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT;
}

export interface AceOpt0011Fp16VaeWindowQuantumFamilyCounts {
  readonly conv1d: number;
  readonly "conv-transpose1d": number;
  readonly snake: number;
  readonly add: number;
}

export interface AceOpt0011Fp16VaeWindowTopology {
  readonly inputFrames: number;
  readonly plan: AceVaeDecoderGraphPlan;
  readonly cooperativePlan: AceVaeDecoderCooperativePlan;
  readonly dynamicControls: AceOpt0011Fp16VaeWindowDynamicControlPlan;
  readonly operationCount:
    typeof ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT;
  readonly graphQuantumCount: number;
  readonly sequenceQuantumCount: number;
  readonly quantumFamilyCounts:
    AceOpt0011Fp16VaeWindowQuantumFamilyCounts;
  readonly activeStagingInputBytes: number;
  readonly activeDecoderInputBytes: number;
  readonly activeOutputBytes: number;
  readonly fp16WorkspaceBytes: number;
  readonly decoderCommandBufferCountAtBatch8: number;
  readonly commandBufferCountAtBatch8: number;
}

export interface AceOpt0011Fp16VaeChunkDispatchPlan {
  readonly maximumWindowFramesProfile:
    AceOpt0011Fp16VaeMaximumWindowFrames;
  readonly chunkPlan: AceVaeChunkedDecodePlan;
  readonly uniqueWindowFrames: readonly number[];
  readonly topologies: readonly AceOpt0011Fp16VaeWindowTopology[];
  readonly windowTopologyIndices: readonly number[];
  readonly maximumFp16WorkspaceBytes: number;
  readonly uniqueDynamicControlBytes: number;
  readonly aggregateGraphQuantumCount: number;
  readonly aggregateSequenceQuantumCount: number;
  readonly aggregateCommandBufferCountAtBatch8: number;
}

export interface AceOpt0011Fp16VaeChunkWindowDispatch {
  readonly window: AceVaeDecodeWindow;
  readonly dispatch: AceOpt0011Fp16VaeWindowDispatch;
}

export interface AceOpt0011Fp16VaeChunkDispatchSet {
  readonly label: string;
  readonly runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"];
  readonly kernelSetId: AceOpt0011Fp16VaeDecoderKernelSetId;
  readonly kernelTopology: AceOpt0011Fp16VaeDecoderKernelTopology;
  readonly topology: AceOpt0011Fp16VaeChunkDispatchPlan;
  /** Sorted one-to-one with `topology.uniqueWindowFrames`. */
  readonly dispatches: readonly AceOpt0011Fp16VaeWindowDispatch[];
  /** One entry per decode window; repeated shapes reuse dispatch identity. */
  readonly windows: readonly AceOpt0011Fp16VaeChunkWindowDispatch[];
}

export class AceOpt0011Fp16VaeDecoderContractError extends Error {
  readonly code = "INVALID_OPT_0011_FP16_VAE_DECODER";

  constructor(message: string) {
    super(message);
    this.name = "AceOpt0011Fp16VaeDecoderContractError";
  }
}

const CANONICAL_PLAN = planAceVaeDecoder(ACE_OPT_0011_VAE_WINDOW_FRAMES);
const CANONICAL_COOPERATIVE_PLAN = planAceVaeDecoderQuanta(CANONICAL_PLAN);
const REVISION7_TRANSPOSE_K4_REUSE_BY_TENSOR: ReadonlyMap<
  string,
  "channel" | "row"
> = new Map(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(
  ({ tensor, reuseAxis }) => [tensor, reuseAxis] as const,
));

interface RequiredRuntimeSelection {
  readonly runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"];
  readonly kernelTopology: AceOpt0011Fp16VaeDecoderKernelTopology;
  readonly precisionMap: AceVaePrecisionMap;
  readonly subgroupCapability?: Readonly<{
    readonly subgroupMinSize: 32;
    readonly subgroupMaxSize: 32;
  }>;
}

function requireRuntimeSelection(
  device: GPUDevice,
  options: AceOpt0011Fp16VaeDecoderRuntimeOptions,
): RequiredRuntimeSelection {
  const runtimeProfileId = options.runtimeProfileId ??
    "opt-0011-mixed-fp16-portable-v1";
  if (runtimeProfileId === "opt-0011-mixed-fp16-portable-v1") {
    return Object.freeze({
      runtimeProfileId: "opt-0011-mixed-fp16-portable-v1",
      kernelTopology: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0011_VAE_FP16_PRECISION_MAP,
    });
  }
  if (runtimeProfileId === "opt-0028-mixed-fp16-portable-exact-packed-v1") {
    if (
      hashAceVaePrecisionMap(
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
      ) !== ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256
    ) {
      throw contractError(
        "OPT-0028 portable exact-packed precision-map identity changed",
      );
    }
    return Object.freeze({
      runtimeProfileId,
      kernelTopology:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
    });
  }
  if (runtimeProfileId === "opt-0088-mixed-fp16-portable-dual-k4-v1") {
    if (
      hashAceVaePrecisionMap(
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
      ) !== ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256
    ) {
      throw contractError(
        "OPT-0088 portable dual-K4 precision-map identity changed",
      );
    }
    return Object.freeze({
      runtimeProfileId,
      kernelTopology: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY,
      precisionMap: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
    });
  }
  if (
    runtimeProfileId !== "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" &&
    runtimeProfileId !==
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" &&
    runtimeProfileId !== "opt-0028-mixed-fp16-fixed32-exact-packed-v1" &&
    runtimeProfileId !==
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" &&
    runtimeProfileId !== "opt-0054-mixed-fp16-fixed32-revision7-v1" &&
    runtimeProfileId !==
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
  ) {
    throw contractError(
      `unknown decoder runtime profile ${String(runtimeProfileId)}`,
    );
  }
  const hybrid = options as Extract<
    AceOpt0011Fp16VaeDecoderRuntimeOptions,
    { subgroupMinSize: 32; subgroupMaxSize: 32 }
  >;
  if (
    !device.features.has("subgroups") ||
    hybrid.subgroupMinSize !== 32 ||
    hybrid.subgroupMaxSize !== 32
  ) {
    throw contractError(
      "fixed32 K7 hybrid requires enabled, authenticated 32/32 subgroups",
    );
  }
  const exactPacked = runtimeProfileId ===
    "opt-0028-mixed-fp16-fixed32-exact-packed-v1";
  const shapeSelected = runtimeProfileId ===
    "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1";
  const opt66 = runtimeProfileId ===
    "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1";
  const revision7 = runtimeProfileId ===
      "opt-0054-mixed-fp16-fixed32-revision7-v1" || opt66;
  const congruent = runtimeProfileId ===
    "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1";
  const precisionMap = opt66
    ? ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP
    : revision7
    ? ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP
    : shapeSelected
    ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP
    : exactPacked
    ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP
    : congruent
      ? ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP
      : ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP;
  const precisionMapSha256 = opt66
    ? ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256
    : revision7
    ? ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256
    : shapeSelected
    ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256
    : exactPacked
    ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256
    : congruent
      ? ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256
      : ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256;
  if (hashAceVaePrecisionMap(precisionMap) !== precisionMapSha256) {
    throw contractError("fixed32 K7 hybrid precision-map identity changed");
  }
  const kernelTopology = opt66
    ? ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY
    : revision7
    ? ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY
    : shapeSelected
    ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_TOPOLOGY
    : exactPacked
    ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_TOPOLOGY
    : congruent
      ? ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_TOPOLOGY
      : ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_TOPOLOGY;
  return Object.freeze({
    runtimeProfileId,
    kernelTopology,
    precisionMap,
    subgroupCapability: Object.freeze({
      subgroupMinSize: 32 as const,
      subgroupMaxSize: 32 as const,
    }),
  });
}

/**
 * The package-native OPT-0011 FP16 decoder. The original B-256 entry point is
 * preserved, while `createChunkDispatchSet` admits exact B/C edge shapes.
 * Callers select a portable or explicit fixed32-K7 hybrid kernel set,
 * provide the authenticated revision-5 package, and own maximum-sized
 * activation storage.
 */
export class AceOpt0011Fp16VaeDecoderRuntime {
  private readonly conv1d: AceFp16VaeConv1dKernel;
  private readonly subgroupConv1d: AceFp16VaeConv1dSubgroupKernel | undefined;
  private readonly revision7K7:
    AceOpt0057VaeK7ShapeSelectorKernel | undefined;
  private readonly portableRevision7K7:
    AceOpt0088VaeK7PortableShapeSelectorKernel | undefined;
  private readonly k1SubgroupGemm:
    AceOpt0025VaeK1SubgroupGemmKernel | undefined;
  private readonly k1PortablePacked:
    AceOpt0028VaeK1PortablePackedKernel | undefined;
  private readonly convTranspose1d:
    AceFp16VaeConvTranspose1dKernel<AceFp16VaeConvTranspose1dKernelId> |
    undefined;
  private readonly exactPackedConvTranspose1d:
    AceOpt0026VaeConvTranspose1dKernel | undefined;
  private readonly portablePackedConvTranspose1d:
    AceOpt0028VaeConvTranspose1dPortablePackedKernel | undefined;
  private readonly shapeSelectedPackedConvTranspose1d:
    AceOpt0040VaeConvTranspose1dShapeSelectorKernel | undefined;
  private readonly revision7ConvTranspose1d:
    AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel | undefined;
  private readonly portableRevision7ConvTranspose1d:
    AceOpt0088VaeConvTranspose1dK4PortableShapeSelectorKernel | undefined;
  private readonly pointwise: AceFp16VaePointwiseKernel;
  private readonly snake: AceFp16VaeSnakeKernel;
  private readonly controlBuffers = new Set<GPUBuffer>();
  private destroyed = false;

  readonly runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"];
  readonly kernelSetId: AceOpt0011Fp16VaeDecoderKernelSetId;
  readonly kernelTopology: AceOpt0011Fp16VaeDecoderKernelTopology;
  readonly precisionMap: AceVaePrecisionMap;

  private constructor(
    private readonly device: GPUDevice,
    selection: RequiredRuntimeSelection,
  ) {
    this.runtimeProfileId = selection.runtimeProfileId;
    this.kernelTopology = selection.kernelTopology;
    this.kernelSetId = selection.kernelTopology.id;
    this.precisionMap = selection.precisionMap;
    this.conv1d = AceFp16VaeConv1dKernel.create(device);
    this.subgroupConv1d = selection.kernelTopology.conv1dK7 ===
        ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID
      ? AceFp16VaeConv1dSubgroupKernel.create(
          device,
          selection.subgroupCapability!,
        )
      : undefined;
    this.revision7K7 = selection.kernelTopology.conv1dK7 ===
        ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID
      ? AceOpt0057VaeK7ShapeSelectorKernel.create(
          device,
          selection.subgroupCapability!,
        )
      : undefined;
    this.portableRevision7K7 = selection.kernelTopology.conv1dK7 ===
        ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID
      ? AceOpt0088VaeK7PortableShapeSelectorKernel.create(device)
      : undefined;
    this.k1SubgroupGemm = selection.kernelTopology.conv1dK1 ===
        ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID
      ? AceOpt0025VaeK1SubgroupGemmKernel.create(
          device,
          selection.subgroupCapability!,
        )
      : undefined;
    this.k1PortablePacked = selection.kernelTopology.conv1dK1 ===
        ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID
      ? AceOpt0028VaeK1PortablePackedKernel.create(device)
      : undefined;
    this.convTranspose1d =
        selection.kernelTopology.convTranspose1d ===
            ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID ||
          selection.kernelTopology.convTranspose1d ===
            ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID ||
          selection.kernelTopology.convTranspose1d ===
            ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID ||
          selection.kernelTopology.convTranspose1d ===
            ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID ||
          selection.kernelTopology.convTranspose1d ===
            ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID
      ? undefined
      : selection.kernelTopology.convTranspose1d ===
        ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
      ? AceFp16VaeConvTranspose1dKernel.createCongruent(device)
      : AceFp16VaeConvTranspose1dKernel.create(device);
    this.exactPackedConvTranspose1d =
        selection.kernelTopology.convTranspose1d ===
          ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID
      ? AceOpt0026VaeConvTranspose1dKernel.create(
          device,
          selection.subgroupCapability!,
      )
      : undefined;
    this.portablePackedConvTranspose1d =
        selection.kernelTopology.convTranspose1d ===
          ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID
      ? AceOpt0028VaeConvTranspose1dPortablePackedKernel.create(device)
      : undefined;
    this.shapeSelectedPackedConvTranspose1d =
        selection.kernelTopology.convTranspose1d ===
          ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID
      ? AceOpt0040VaeConvTranspose1dShapeSelectorKernel.create(
          device,
          selection.subgroupCapability!,
        )
      : undefined;
    this.revision7ConvTranspose1d =
        selection.kernelTopology.convTranspose1d ===
          ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID
      ? AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel.create(
          device,
          selection.subgroupCapability!,
        )
      : undefined;
    this.portableRevision7ConvTranspose1d =
        selection.kernelTopology.convTranspose1d ===
          ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID
      ? AceOpt0088VaeConvTranspose1dK4PortableShapeSelectorKernel.create(
          device,
        )
      : undefined;
    this.pointwise = AceFp16VaePointwiseKernel.create(device);
    this.snake = AceFp16VaeSnakeKernel.create(device);
  }

  static create(
    device: GPUDevice,
    options: AceOpt0011Fp16VaeDecoderRuntimeOptions = {},
  ): AceOpt0011Fp16VaeDecoderRuntime {
    requireDeviceContract(device);
    requireCanonicalContract();
    const selection = requireRuntimeSelection(device, options);
    return new AceOpt0011Fp16VaeDecoderRuntime(device, selection);
  }

  async createDecoderDispatch(
    label: string,
    bindings: AceOpt0011Fp16VaeDecoderBindings,
  ): Promise<AceOpt0011Fp16VaeDecoderDispatch> {
    this.requireLive();
    requireDispatchLabel(label);
    const topology = planWindowTopology(
      CANONICAL_PLAN,
      CANONICAL_COOPERATIVE_PLAN,
      this.device.limits.minUniformBufferOffsetAlignment,
    );
    requireCanonicalTopology(topology);
    requirePlanDeviceContract(this.device, topology);
    requirePackageBindings(CANONICAL_PLAN, bindings.package, this.runtimeProfileId);
    requireRuntimeBindings(this.device, label, CANONICAL_PLAN, bindings);
    const built = await this.createWindowDispatch(label, topology, bindings);
    let retained = false;
    try {
      this.requireLive();
      this.controlBuffers.add(built.controlBuffer);
      retained = true;
      return built.dispatch as AceOpt0011Fp16VaeDecoderDispatch;
    } finally {
      if (!retained) built.controlBuffer.destroy();
    }
  }

  /**
   * Build every exact window shape for one pinned 256/64 or 512/64 chunk plan.
   * Edge windows are compiled at their real frame count and repeated shapes
   * reuse the same dispatch and immutable control buffer.
   */
  async createChunkDispatchSet(
    label: string,
    latentFrames: number,
    maximumWindowFramesProfile: AceOpt0011Fp16VaeMaximumWindowFrames,
    bindings: AceOpt0011Fp16VaeWindowBindings,
  ): Promise<AceOpt0011Fp16VaeChunkDispatchSet> {
    this.requireLive();
    requireDispatchLabel(label);
    const topology = planAceOpt0011Fp16VaeChunkDispatches(
      latentFrames,
      maximumWindowFramesProfile,
      this.device.limits.minUniformBufferOffsetAlignment,
    );
    const maximumTopology = topology.topologies.find((candidate) =>
      candidate.inputFrames === topology.chunkPlan.maximumWindowFrames
    );
    if (maximumTopology === undefined) {
      throw contractError("chunk topology lost its maximum window shape");
    }

    // Finish every plan/package/resource/device check before the first control
    // allocation so a rejected C profile cannot leave a partial B dispatch.
    for (const candidate of topology.topologies) {
      requirePlanDeviceContract(this.device, candidate);
    }
    requirePackageBindings(
      maximumTopology.plan,
      bindings.package,
      this.runtimeProfileId,
    );
    requireRuntimeBindings(
      this.device,
      label,
      maximumTopology.plan,
      bindings,
    );

    const pending: Array<Readonly<{
      dispatch: AceOpt0011Fp16VaeWindowDispatch;
      controlBuffer: GPUBuffer;
    }>> = [];
    let committed = false;
    try {
      for (const candidate of topology.topologies) {
        this.requireLive();
        pending.push(await this.createWindowDispatch(
          `${label}-window-${candidate.inputFrames}`,
          candidate,
          bindings,
        ));
      }
      this.requireLive();
      const dispatches = Object.freeze(pending.map(({ dispatch }) => dispatch));
      const byFrames = new Map(dispatches.map((dispatch) => [
        dispatch.plan.inputFrames,
        dispatch,
      ]));
      const windows = Object.freeze(topology.chunkPlan.windows.map((window) => {
        const dispatch = byFrames.get(window.latentWindowFrames);
        if (dispatch === undefined) {
          throw contractError(
            `window ${window.index} has no exact ${window.latentWindowFrames}-frame dispatch`,
          );
        }
        return Object.freeze({ window, dispatch });
      }));
      const result = Object.freeze({
        label,
        runtimeProfileId: this.runtimeProfileId,
        kernelSetId: this.kernelSetId,
        kernelTopology: this.kernelTopology,
        topology,
        dispatches,
        windows,
      });
      for (const { controlBuffer } of pending) {
        this.controlBuffers.add(controlBuffer);
      }
      committed = true;
      return result;
    } finally {
      if (!committed) {
        for (const { controlBuffer } of pending) controlBuffer.destroy();
      }
    }
  }

  private async createWindowDispatch(
    label: string,
    topology: AceOpt0011Fp16VaeWindowTopology,
    bindings: AceOpt0011Fp16VaeWindowBindings,
  ): Promise<Readonly<{
    dispatch: AceOpt0011Fp16VaeWindowDispatch;
    controlBuffer: GPUBuffer;
  }>> {
    const plan = topology.plan;
    const cooperativePlan = topology.cooperativePlan;
    const dynamicControls = topology.dynamicControls;
    const rangeControls = await createDynamicControlBuffer(
      this.device,
      dynamicControls,
      `${label}-fp16-vae-range-controls`,
    );
    let complete = false;
    try {
      this.requireLive();
      const ingressRecord = dynamicControls.records[0]!;
      const ingressPrecision = this.precisionMap.entries[0]!;
      const ingressDispatch = await this.pointwise.createIngressDispatch(
        `${label}-sequence-0-${ingressPrecision.label}`,
        {
          batch: plan.batch,
          frames: plan.inputFrames,
          channels: plan.config.decoderInputChannels,
        },
        {
          input: bindings.stagingInput,
          output: bindings.decoderInput,
        },
        rangeBinding(rangeControls, ingressRecord),
      );
      const ingressQuantum = freezeQuantum({
        sequenceIndex: 0,
        graphQuantumIndex: null,
        operationIndex: null,
        operationLabel: ingressPrecision.label,
        operationKind: "ingress-cast",
        logicalOutputBase: ingressRecord.outputBase,
        logicalOutputCount: ingressRecord.outputCount,
        estimatedMaximumMultiplyAccumulates: 0,
        kernelId: ingressDispatch.kernelId,
        control: ingressRecord,
        precision: ingressPrecision,
        encode: (pass) => ingressDispatch.encode(pass),
      });

      const graphQuanta: AceOpt0011Fp16VaeDecoderQuantum[] = [];
      for (const graphQuantum of cooperativePlan.quanta) {
        const operation = plan.operations[graphQuantum.operationIndex]!;
        const operationBindings = bindings.package.operations[
          graphQuantum.operationIndex
        ]!;
        const sequenceIndex = graphQuantum.index + 1;
        const control = dynamicControls.records[sequenceIndex]!;
        const precision = this.precisionMap.entries[
          graphQuantum.operationIndex + 1
        ]!;
        const dispatch = await this.createOperationDispatch(
          `${label}-sequence-${sequenceIndex}-${graphQuantum.id}`,
          operation,
          operationBindings,
          graphQuantum,
          control,
          precision,
          rangeControls,
          bindings,
        );
        graphQuanta.push(freezeQuantum({
          sequenceIndex,
          graphQuantumIndex: graphQuantum.index,
          operationIndex: graphQuantum.operationIndex,
          operationLabel: graphQuantum.operationLabel,
          operationKind: graphQuantum.operationKind,
          logicalOutputBase: graphQuantum.logicalOutputBase,
          logicalOutputCount: graphQuantum.logicalOutputCount,
          estimatedMaximumMultiplyAccumulates:
            graphQuantum.estimatedMaximumMultiplyAccumulates,
          kernelId: dispatch.kernelId,
          control,
          precision,
          encode: (pass) => dispatch.encode(pass),
        }));
      }
      this.requireLive();
      if (graphQuanta.length !== topology.graphQuantumCount) {
        throw contractError("graph dispatch construction was incomplete");
      }
      const frozenGraphQuanta = Object.freeze(graphQuanta);
      const quanta = Object.freeze([ingressQuantum, ...frozenGraphQuanta]);
      if (quanta.length !== topology.sequenceQuantumCount) {
        throw contractError("window dispatch sequence is incomplete");
      }
      const dispatch = Object.freeze({
        label,
        runtimeProfileId: this.runtimeProfileId,
        kernelSetId: this.kernelSetId,
        kernelTopology: this.kernelTopology,
        plan,
        cooperativePlan,
        precisionMap: this.precisionMap,
        dynamicControls,
        ingressQuantum,
        graphQuanta: frozenGraphQuanta,
        quanta,
        operationCount: ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT,
        graphQuantumCount: topology.graphQuantumCount,
        primitiveCount: topology.sequenceQuantumCount,
        activeStagingInputBytes: topology.activeStagingInputBytes,
        activeDecoderInputBytes: topology.activeDecoderInputBytes,
        activeOutputBytes: topology.activeOutputBytes,
        decoderCommandBufferCountAtBatch8:
          topology.decoderCommandBufferCountAtBatch8,
        commandBufferCountAtBatch8: topology.commandBufferCountAtBatch8,
      });
      complete = true;
      return Object.freeze({ dispatch, controlBuffer: rangeControls });
    } finally {
      if (!complete) rangeControls.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.controlBuffers) buffer.destroy();
    this.controlBuffers.clear();
    this.conv1d.destroy();
    this.subgroupConv1d?.destroy();
    this.revision7K7?.destroy();
    this.portableRevision7K7?.destroy();
    this.k1SubgroupGemm?.destroy();
    this.k1PortablePacked?.destroy();
    this.convTranspose1d?.destroy();
    this.exactPackedConvTranspose1d?.destroy();
    this.portablePackedConvTranspose1d?.destroy();
    this.shapeSelectedPackedConvTranspose1d?.destroy();
    this.revision7ConvTranspose1d?.destroy();
    this.portableRevision7ConvTranspose1d?.destroy();
    this.pointwise.destroy();
    this.snake.destroy();
  }

  private async createOperationDispatch(
    label: string,
    operation: AceVaeDecoderOperation,
    operationBindings: AceOpt0011VaeOperationBindings,
    graphQuantum: AceVaeDecoderQuantumPlan,
    control: AceOpt0011Fp16VaeDecoderControlRecord,
    precision: AceVaePrecisionMapEntry,
    rangeControls: GPUBuffer,
    bindings: AceOpt0011Fp16VaeWindowBindings,
  ): Promise<Readonly<{
    kernelId: AceOpt0011Fp16VaeDecoderKernelId;
    encode(pass: GPUComputePassEncoder): void;
  }>> {
    const primitive = graphQuantum.primitives[0]!;
    if (
      control.outputBase !== primitive.outputBase ||
      control.outputCount !== primitive.outputCount
    ) {
      throw contractError(`${graphQuantum.id} lost its dynamic-control range`);
    }
    const range = rangeBinding(rangeControls, control);
    const input = resolveSlot(bindings, operation.input);
    const output = resolveSlot(bindings, operation.output);
    switch (operation.kind) {
      case "conv1d": {
        if (operationBindings.kind !== "conv1d") {
          throw contractError(`${operation.label} lost its Conv1D bindings`);
        }
        const outputStorage = precision.output.storage;
        if (outputStorage !== "float16" && outputStorage !== "float32") {
          throw contractError(`${operation.label} has unknown output storage`);
        }
        if (
          operation.shape.kernelSize === 1 &&
          this.k1PortablePacked !== undefined
        ) {
          if (operationBindings.bias === undefined || outputStorage !== "float16") {
            throw contractError(
              `${operation.label} is outside the portable packed K1 family`,
            );
          }
          return this.k1PortablePacked.createRangeDispatch(
            label,
            operation.shape,
            {
              input,
              packedWeight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (
          operation.shape.kernelSize === 1 &&
          this.k1SubgroupGemm !== undefined
        ) {
          if (operationBindings.bias === undefined || outputStorage !== "float16") {
            throw contractError(
              `${operation.label} is outside the exact biased-FP16 K1 family`,
            );
          }
          return this.k1SubgroupGemm.createRangeDispatch(
            label,
            operation.shape,
            {
              input,
              packedWeight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (
          operation.shape.kernelSize === 7 &&
          this.revision7K7 !== undefined
        ) {
          return this.revision7K7.createDispatch(
            label,
            operation.label,
            operation.shape,
            {
              input,
              weight: operationBindings.weight.binding,
              ...(operationBindings.bias === undefined
                ? {}
                : { bias: operationBindings.bias.binding }),
              output,
            },
            outputStorage,
            range,
          );
        }
        if (
          operation.shape.kernelSize === 7 &&
          this.portableRevision7K7 !== undefined
        ) {
          return this.portableRevision7K7.createDispatch(
            label,
            operation.label,
            operation.shape,
            {
              input,
              weight: operationBindings.weight.binding,
              ...(operationBindings.bias === undefined
                ? {}
                : { bias: operationBindings.bias.binding }),
              output,
            },
            outputStorage,
            range,
          );
        }
        const conv1d = operation.shape.kernelSize === 7 &&
            this.subgroupConv1d !== undefined
          ? this.subgroupConv1d
          : this.conv1d;
        return conv1d.createDispatch(
          label,
          operation.shape,
          {
            input,
            weight: operationBindings.weight.binding,
            ...(operationBindings.bias === undefined
              ? {}
              : { bias: operationBindings.bias.binding }),
            output,
          },
          outputStorage,
          range,
        );
      }
      case "conv-transpose1d":
        if (operationBindings.kind !== "conv-transpose1d") {
          throw contractError(
            `${operation.label} lost its ConvTranspose1D bindings`,
          );
        }
        if (this.revision7ConvTranspose1d !== undefined) {
          return this.revision7ConvTranspose1d.createDispatch(
            label,
            operation.label,
            operation.shape,
            {
              input,
              weight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (this.portableRevision7ConvTranspose1d !== undefined) {
          return this.portableRevision7ConvTranspose1d.createDispatch(
            label,
            operation.label,
            operation.shape,
            {
              input,
              weight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (this.portablePackedConvTranspose1d !== undefined) {
          return this.portablePackedConvTranspose1d.createDispatch(
            label,
            operation.shape,
            {
              input,
              polyphaseWeight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (this.shapeSelectedPackedConvTranspose1d !== undefined) {
          return this.shapeSelectedPackedConvTranspose1d.createDispatch(
            label,
            operation.label,
            operation.shape,
            {
              input,
              polyphaseWeight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (this.exactPackedConvTranspose1d !== undefined) {
          return this.exactPackedConvTranspose1d.createDispatch(
            label,
            operation.shape,
            {
              input,
              polyphaseWeight: operationBindings.weight.binding,
              bias: operationBindings.bias.binding,
              output,
            },
            range,
          );
        }
        if (this.convTranspose1d === undefined) {
          throw contractError(`${operation.label} has no transpose kernel`);
        }
        return this.convTranspose1d.createDispatch(
          label,
          operation.shape,
          {
            input,
            weight: operationBindings.weight.binding,
            bias: operationBindings.bias.binding,
            output,
          },
          range,
        );
      case "snake":
        if (operationBindings.kind !== "snake") {
          throw contractError(`${operation.label} lost its Snake bindings`);
        }
        return this.snake.createDispatch(
          label,
          operation.shape,
          {
            input,
            alpha: operationBindings.alpha.binding,
            beta: operationBindings.beta.binding,
            output,
          },
          range,
        );
      case "add":
        if (operationBindings.kind !== "add") {
          throw contractError(`${operation.label} lost its Add bindings`);
        }
        return this.pointwise.createAddDispatch(
          label,
          operation.shape,
          {
            left: input,
            right: resolveSlot(bindings, operation.right),
            output,
          },
          range,
        );
    }
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0011 FP16 VAE decoder runtime was destroyed");
    }
  }
}

/** Preserve the original exact B-256 control-planning API and byte identity. */
export function planAceOpt0011Fp16VaeDecoderDynamicControls(
  recordAlignment: number,
): AceOpt0011Fp16VaeDecoderDynamicControlPlan {
  requireCanonicalContract();
  const controls = planDynamicControls(
    CANONICAL_PLAN,
    CANONICAL_COOPERATIVE_PLAN,
    recordAlignment,
  );
  if (
    controls.recordCount !==
      ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT
  ) {
    throw contractError("canonical B-256 control count changed");
  }
  return controls as AceOpt0011Fp16VaeDecoderDynamicControlPlan;
}

/** Build the exact one-write control layout for an unpadded window shape. */
export function planAceOpt0011Fp16VaeWindowDynamicControls(
  inputFrames: number,
  recordAlignment: number,
): AceOpt0011Fp16VaeWindowDynamicControlPlan {
  requireCanonicalContract();
  const plan = planAceVaeDecoder(inputFrames);
  const cooperative = planAceVaeDecoderQuanta(plan);
  requireGeometryNeutralPlan(plan, cooperative);
  return planDynamicControls(plan, cooperative, recordAlignment);
}

/**
 * Derive the complete exact-shape topology for an approved maximum-window
 * profile. C2378 is retained only for the isolated OPT-0035 benchmark gate.
 * This is pure planning: no GPU resource or pipeline allocation occurs here.
 */
export function planAceOpt0011Fp16VaeChunkDispatches(
  latentFrames: number,
  maximumWindowFramesProfile: AceOpt0011Fp16VaeMaximumWindowFrames,
  recordAlignment: number,
): AceOpt0011Fp16VaeChunkDispatchPlan {
  requireCanonicalContract();
  if (
    maximumWindowFramesProfile !== 256 &&
    maximumWindowFramesProfile !== 512 &&
    maximumWindowFramesProfile !==
      ACE_CAPPED_VAE_FP16_C2176_MAXIMUM_WINDOW_FRAMES &&
    maximumWindowFramesProfile !==
      ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES
  ) {
    throw contractError(
      "chunk runtime supports only 256-, 512-, capped 2176-, or OPT-0035 " +
        "2378-frame profiles",
    );
  }
  const chunkPlan = planAceVaeChunkedDecode(latentFrames, {
    chunkFrames: maximumWindowFramesProfile,
    overlapFrames: ACE_VAE_DEFAULT_OVERLAP_FRAMES,
  });
  const uniqueWindowFrames = Object.freeze(
    [...new Set(chunkPlan.windows.map((window) => window.latentWindowFrames))]
      .sort((left, right) => left - right),
  );
  if (
    uniqueWindowFrames.length === 0 ||
    uniqueWindowFrames.at(-1) !== chunkPlan.maximumWindowFrames ||
    uniqueWindowFrames.some((frames) =>
      !Number.isSafeInteger(frames) ||
      frames <= 0 ||
      frames > maximumWindowFramesProfile
    )
  ) {
    throw contractError("chunk planner produced an invalid exact-shape inventory");
  }
  const topologies = Object.freeze(uniqueWindowFrames.map((inputFrames) => {
    const plan = planAceVaeDecoder(inputFrames);
    const cooperative = planAceVaeDecoderQuanta(plan);
    return planWindowTopology(plan, cooperative, recordAlignment);
  }));
  const maximumTopology = topologies.at(-1);
  if (
    maximumTopology === undefined ||
    JSON.stringify(maximumTopology.plan) !==
      JSON.stringify(chunkPlan.decoderWorkspacePlan)
  ) {
    throw contractError(
      "exact-shape inventory diverged from the chunk planner workspace graph",
    );
  }
  const topologyIndexByFrames = new Map(
    uniqueWindowFrames.map((frames, index) => [frames, index]),
  );
  const windowTopologyIndices = Object.freeze(chunkPlan.windows.map((window) => {
    const index = topologyIndexByFrames.get(window.latentWindowFrames);
    if (index === undefined) {
      throw contractError(`window ${window.index} lost its exact topology`);
    }
    return index;
  }));
  if (
    new Set(windowTopologyIndices).size !== topologies.length ||
    windowTopologyIndices.some((topologyIndex, windowIndex) =>
      topologies[topologyIndex]?.inputFrames !==
        chunkPlan.windows[windowIndex]?.latentWindowFrames
    )
  ) {
    throw contractError("chunk windows do not cover every exact topology");
  }
  let uniqueDynamicControlBytes = 0;
  let aggregateGraphQuantumCount = 0;
  let aggregateSequenceQuantumCount = 0;
  let aggregateCommandBufferCountAtBatch8 = 0;
  for (const topology of topologies) {
    uniqueDynamicControlBytes = checkedSum(
      uniqueDynamicControlBytes,
      topology.dynamicControls.byteLength,
      "unique dynamic-control bytes",
    );
  }
  for (const topologyIndex of windowTopologyIndices) {
    const topology = topologies[topologyIndex]!;
    aggregateGraphQuantumCount = checkedSum(
      aggregateGraphQuantumCount,
      topology.graphQuantumCount,
      "aggregate graph quanta",
    );
    aggregateSequenceQuantumCount = checkedSum(
      aggregateSequenceQuantumCount,
      topology.sequenceQuantumCount,
      "aggregate sequence quanta",
    );
    aggregateCommandBufferCountAtBatch8 = checkedSum(
      aggregateCommandBufferCountAtBatch8,
      topology.commandBufferCountAtBatch8,
      "aggregate batch-8 command buffers",
    );
  }
  return Object.freeze({
    maximumWindowFramesProfile,
    chunkPlan,
    uniqueWindowFrames,
    topologies,
    windowTopologyIndices,
    maximumFp16WorkspaceBytes: Math.max(
      ...topologies.map((topology) => topology.fp16WorkspaceBytes),
    ),
    uniqueDynamicControlBytes,
    aggregateGraphQuantumCount,
    aggregateSequenceQuantumCount,
    aggregateCommandBufferCountAtBatch8,
  });
}

function planWindowTopology(
  plan: AceVaeDecoderGraphPlan,
  cooperativePlan: AceVaeDecoderCooperativePlan,
  recordAlignment: number,
): AceOpt0011Fp16VaeWindowTopology {
  requireGeometryNeutralPlan(plan, cooperativePlan);
  const dynamicControls = planDynamicControls(
    plan,
    cooperativePlan,
    recordAlignment,
  );
  const sequenceQuantumCount = checkedSum(
    cooperativePlan.quantumCount,
    1,
    "window sequence quanta",
  );
  if (dynamicControls.recordCount !== sequenceQuantumCount) {
    throw contractError("window controls do not cover ingress plus graph");
  }
  const decoderCommandBufferCountAtBatch8 = Math.ceil(
    sequenceQuantumCount /
      ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  );
  const topology: AceOpt0011Fp16VaeWindowTopology = Object.freeze({
    inputFrames: plan.inputFrames,
    plan,
    cooperativePlan,
    dynamicControls,
    operationCount: ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT,
    graphQuantumCount: cooperativePlan.quantumCount,
    sequenceQuantumCount,
    quantumFamilyCounts: freezeKindCounts(countKinds(cooperativePlan.quanta)),
    activeStagingInputBytes: checkedProduct(
      plan.inputElements,
      FLOAT32_BYTES,
      "active staging-input bytes",
    ),
    activeDecoderInputBytes: checkedProduct(
      plan.inputElements,
      FLOAT16_BYTES,
      "active decoder-input bytes",
    ),
    activeOutputBytes: checkedProduct(
      plan.outputElements,
      FLOAT32_BYTES,
      "active output bytes",
    ),
    fp16WorkspaceBytes: checkedProduct(
      plan.maximumActivationElements,
      FLOAT16_BYTES,
      "FP16 workspace bytes",
    ),
    decoderCommandBufferCountAtBatch8,
    commandBufferCountAtBatch8: checkedSum(
      decoderCommandBufferCountAtBatch8,
      1,
      "window command buffers",
    ),
  });
  requirePinnedWindowTopology(topology);
  return topology;
}

function planDynamicControls(
  plan: AceVaeDecoderGraphPlan,
  cooperativePlan: AceVaeDecoderCooperativePlan,
  recordAlignment: number,
): AceOpt0011Fp16VaeWindowDynamicControlPlan {
  if (!isValidGpuAlignment(recordAlignment)) {
    throw contractError("dynamic controls require a valid uniform alignment");
  }
  const ingress = freezeControlRecord({
    recordIndex: 0,
    sequenceIndex: 0,
    graphQuantumIndex: null,
    operationIndex: null,
    operationLabel: "f32-latent-to-f16-decoder-input",
    operationKind: "ingress-cast",
    outputBase: 0,
    outputCount: plan.inputElements,
    byteOffset: 0,
  });
  const records: AceOpt0011Fp16VaeDecoderControlRecord[] = [ingress];
  for (const quantum of cooperativePlan.quanta) {
    const primitive = quantum.primitives[0]!;
    const recordIndex = quantum.index + 1;
    const byteOffset = checkedProduct(
      recordIndex,
      recordAlignment,
      "dynamic-control byte offset",
    );
    if (byteOffset > MAX_WGSL_DYNAMIC_OFFSET) {
      throw contractError("dynamic-control offset exceeds WGSL's u32 domain");
    }
    records.push(freezeControlRecord({
      recordIndex,
      sequenceIndex: recordIndex,
      graphQuantumIndex: quantum.index,
      operationIndex: quantum.operationIndex,
      operationLabel: quantum.operationLabel,
      operationKind: quantum.operationKind,
      outputBase: primitive.outputBase,
      outputCount: primitive.outputCount,
      byteOffset,
    }));
  }
  if (
    records.length !== cooperativePlan.quantumCount + 1 ||
    records.some((record, index) =>
      record.recordIndex !== index ||
      record.sequenceIndex !== index ||
      record.byteOffset !== index * recordAlignment
    )
  ) {
    throw contractError("dynamic-control records are not complete and dense");
  }
  const byteLength = checkedSum(
    records.at(-1)!.byteOffset,
    ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES,
    "dynamic-control byte length",
  );
  return Object.freeze({
    recordBytes: ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES,
    recordAlignment,
    recordCount: records.length,
    byteLength,
    records: Object.freeze(records),
  });
}

function requireGeometryNeutralPlan(
  plan: AceVaeDecoderGraphPlan,
  cooperativePlan: AceVaeDecoderCooperativePlan,
): void {
  if (
    !Number.isSafeInteger(plan.inputFrames) ||
    plan.inputFrames <= 0 ||
    plan.inputFrames > ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES ||
    plan.batch !== 1 ||
    JSON.stringify(plan.config) !== JSON.stringify(CANONICAL_PLAN.config) ||
    plan.operations.length !== ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT ||
    plan.primitiveCount !== ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT ||
    plan.requiredTensorNames.length !==
      ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT ||
    plan.requiredTensorNames.some((name, index) =>
      name !== CANONICAL_PLAN.requiredTensorNames[index]
    ) ||
    plan.parameterElements !== CANONICAL_PLAN.parameterElements ||
    plan.parameterBytes !== CANONICAL_PLAN.parameterBytes ||
    plan.outputFrames * CANONICAL_PLAN.inputFrames !==
      CANONICAL_PLAN.outputFrames * plan.inputFrames
  ) {
    throw contractError(
      `${plan.inputFrames}-frame graph lost the geometry-neutral 88-op/145-tensor skeleton`,
    );
  }
  for (let index = 0; index < plan.operations.length; index += 1) {
    requireGeometryNeutralOperation(
      plan.operations[index]!,
      CANONICAL_PLAN.operations[index]!,
      plan.inputFrames,
      index,
    );
    requirePrecisionEntry(
      plan.operations[index]!,
      index,
      ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries[index + 1]!,
    );
  }
  if (
    cooperativePlan.quantumCount !== cooperativePlan.quanta.length ||
    cooperativePlan.primitiveDispatchCount !== cooperativePlan.quantumCount ||
    cooperativePlan.quanta.some((quantum, index) => {
      const operation = plan.operations[quantum.operationIndex];
      const primitive = quantum.primitives[0];
      if (
        quantum.index !== index ||
        operation === undefined ||
        quantum.operationLabel !== operation.label ||
        quantum.operationKind !== operation.kind ||
        quantum.primitives.length !== 1 ||
        primitive === undefined ||
        primitive.controlRecordIndex !== quantum.index
      ) return true;
      return operation.kind === "conv-transpose1d"
        ? primitive.physicalPartIndex !== 0 ||
          primitive.firstOutputChannel !== 0 ||
          primitive.outputChannels !== operation.shape.outputChannels
        : primitive.physicalPartIndex !== undefined;
    })
  ) {
    throw contractError(
      `${plan.inputFrames}-frame graph lost the one-primitive FIFO topology`,
    );
  }
}

function requireGeometryNeutralOperation(
  operation: AceVaeDecoderOperation,
  canonical: AceVaeDecoderOperation,
  inputFrames: number,
  operationIndex: number,
): void {
  if (
    operation.kind !== canonical.kind ||
    operation.label !== canonical.label ||
    operation.input !== canonical.input ||
    operation.output !== canonical.output
  ) {
    throw contractError(`operation ${operationIndex} changed its graph skeleton`);
  }
  switch (operation.kind) {
    case "conv1d":
      if (
        canonical.kind !== "conv1d" ||
        operation.weight !== canonical.weight ||
        operation.bias !== canonical.bias ||
        operation.shape.batch !== canonical.shape.batch ||
        operation.shape.inputChannels !== canonical.shape.inputChannels ||
        operation.shape.outputChannels !== canonical.shape.outputChannels ||
        operation.shape.kernelSize !== canonical.shape.kernelSize ||
        operation.shape.stride !== canonical.shape.stride ||
        operation.shape.dilation !== canonical.shape.dilation ||
        operation.shape.padding !== canonical.shape.padding ||
        operation.shape.inputFrames * CANONICAL_PLAN.inputFrames !==
          canonical.shape.inputFrames * inputFrames
      ) {
        throw contractError(
          `operation ${operationIndex} changed its geometry-neutral Conv1D contract`,
        );
      }
      return;
    case "conv-transpose1d":
      if (
        canonical.kind !== "conv-transpose1d" ||
        operation.weight !== canonical.weight ||
        operation.bias !== canonical.bias ||
        operation.shape.batch !== canonical.shape.batch ||
        operation.shape.inputChannels !== canonical.shape.inputChannels ||
        operation.shape.outputChannels !== canonical.shape.outputChannels ||
        operation.shape.kernelSize !== canonical.shape.kernelSize ||
        operation.shape.stride !== canonical.shape.stride ||
        operation.shape.dilation !== canonical.shape.dilation ||
        operation.shape.padding !== canonical.shape.padding ||
        operation.shape.outputPadding !== canonical.shape.outputPadding ||
        operation.shape.inputFrames * CANONICAL_PLAN.inputFrames !==
          canonical.shape.inputFrames * inputFrames
      ) {
        throw contractError(
          `operation ${operationIndex} changed its geometry-neutral transpose contract`,
        );
      }
      return;
    case "snake":
      if (
        canonical.kind !== "snake" ||
        operation.alpha !== canonical.alpha ||
        operation.beta !== canonical.beta ||
        operation.shape.batch !== canonical.shape.batch ||
        operation.shape.channels !== canonical.shape.channels ||
        operation.shape.frames * CANONICAL_PLAN.inputFrames !==
          canonical.shape.frames * inputFrames
      ) {
        throw contractError(
          `operation ${operationIndex} changed its geometry-neutral Snake contract`,
        );
      }
      return;
    case "add":
      if (
        canonical.kind !== "add" ||
        operation.right !== canonical.right ||
        operation.shape.batch !== canonical.shape.batch ||
        operation.shape.channels !== canonical.shape.channels ||
        operation.shape.frames * CANONICAL_PLAN.inputFrames !==
          canonical.shape.frames * inputFrames
      ) {
        throw contractError(
          `operation ${operationIndex} changed its geometry-neutral Add contract`,
        );
      }
  }
}

function requirePinnedWindowTopology(
  topology: AceOpt0011Fp16VaeWindowTopology,
): void {
  const counts = topology.quantumFamilyCounts;
  if (topology.inputFrames === 256) {
    if (
      topology.graphQuantumCount !==
        ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT ||
      topology.sequenceQuantumCount !==
        ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT ||
      counts.conv1d !== 2_459 ||
      counts["conv-transpose1d"] !== 322 ||
      counts.snake !== 813 ||
      counts.add !== 348 ||
      topology.fp16WorkspaceBytes !== 125_829_120 ||
      topology.commandBufferCountAtBatch8 !== 494 ||
      (topology.dynamicControls.recordAlignment === 256 &&
        topology.dynamicControls.byteLength !== 1_009_168)
    ) {
      throw contractError("canonical B-256 window topology changed");
    }
  }
  if (topology.inputFrames === 512) {
    if (
      topology.graphQuantumCount !==
        ACE_OPT_0011_VAE_FP16_C512_GRAPH_QUANTUM_COUNT ||
      topology.sequenceQuantumCount !==
        ACE_OPT_0011_VAE_FP16_C512_SEQUENCE_QUANTUM_COUNT ||
      counts.conv1d !== 4_909 ||
      counts["conv-transpose1d"] !== 644 ||
      counts.snake !== 1_611 ||
      counts.add !== 690 ||
      topology.fp16WorkspaceBytes !==
        ACE_OPT_0011_VAE_FP16_C512_WORKSPACE_BYTES ||
      topology.commandBufferCountAtBatch8 !==
        ACE_OPT_0011_VAE_FP16_C512_COMMAND_BUFFER_COUNT_AT_BATCH8 ||
      (topology.dynamicControls.recordAlignment === 256 &&
        topology.dynamicControls.byteLength !==
          ACE_OPT_0011_VAE_FP16_C512_CONTROL_BYTES)
    ) {
      throw contractError("canonical C-512 window topology changed");
    }
  }
}

function requireCanonicalTopology(
  topology: AceOpt0011Fp16VaeWindowTopology,
): void {
  if (
    topology.inputFrames !== ACE_OPT_0011_VAE_WINDOW_FRAMES ||
    topology.plan !== CANONICAL_PLAN ||
    topology.cooperativePlan !== CANONICAL_COOPERATIVE_PLAN ||
    topology.dynamicControls.recordCount !==
      ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT
  ) {
    throw contractError("original B-256 dispatch seam changed");
  }
}

function requireCanonicalContract(): void {
  requireAceOpt0011VaeDecoderGeometry(CANONICAL_PLAN);
  requireGeometryNeutralPlan(CANONICAL_PLAN, CANONICAL_COOPERATIVE_PLAN);
  if (
    CANONICAL_PLAN.operations.length !==
      ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT ||
    CANONICAL_PLAN.primitiveCount !==
      ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT ||
    CANONICAL_COOPERATIVE_PLAN.quantumCount !==
      ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT ||
    CANONICAL_COOPERATIVE_PLAN.primitiveDispatchCount !==
      ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT ||
    CANONICAL_COOPERATIVE_PLAN.quanta.some((quantum) => {
      const primitive = quantum.primitives[0];
      const operation = CANONICAL_PLAN.operations[quantum.operationIndex];
      if (
        primitive === undefined ||
        quantum.primitives.length !== 1 ||
        primitive.controlRecordIndex !== quantum.index ||
        operation === undefined
      ) return true;
      return operation.kind === "conv-transpose1d"
        ? primitive.physicalPartIndex !== 0 ||
          primitive.firstOutputChannel !== 0 ||
          primitive.outputChannels !== operation.shape.outputChannels
        : primitive.physicalPartIndex !== undefined;
    })
  ) {
    throw contractError("canonical B-256 graph topology changed");
  }
  const kindCounts = countKinds(CANONICAL_COOPERATIVE_PLAN.quanta);
  if (
    kindCounts.conv1d !== 2_459 ||
    kindCounts["conv-transpose1d"] !== 322 ||
    kindCounts.snake !== 813 ||
    kindCounts.add !== 348
  ) {
    throw contractError("canonical B-256 quantum-family counts changed");
  }
  const precision = ACE_OPT_0011_VAE_FP16_PRECISION_MAP;
  if (
    precision.entries.length !==
      ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT + 1 ||
    hashAceVaePrecisionMap(precision) !==
      ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256
  ) {
    throw contractError("the 89-entry precision-map identity changed");
  }
  const ingress = precision.entries[0]!;
  if (
    ingress.sequenceIndex !== 0 ||
    ingress.graphOperationIndex !== null ||
    ingress.label !== "f32-latent-to-f16-decoder-input" ||
    ingress.kind !== "ingress-cast" ||
    ingress.output.storage !== "float16"
  ) {
    throw contractError("the decoder ingress precision contract changed");
  }
  for (const [operationIndex, operation] of CANONICAL_PLAN.operations.entries()) {
    requirePrecisionEntry(operation, operationIndex, precision.entries[
      operationIndex + 1
    ]!);
  }
}

function requirePrecisionEntry(
  operation: AceVaeDecoderOperation,
  operationIndex: number,
  precision: AceVaePrecisionMapEntry,
): void {
  const expectedFamily = operation.kind === "conv1d"
    ? operation.shape.kernelSize === 1
      ? "conv1d-k1"
      : operation.shape.kernelSize === 7
        ? "conv1d-k7"
        : undefined
    : operation.kind;
  const isFinal = operationIndex ===
    ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT - 1;
  if (
    precision.sequenceIndex !== operationIndex + 1 ||
    precision.graphOperationIndex !== operationIndex ||
    precision.label !== operation.label ||
    precision.kind !== operation.kind ||
    precision.kernelFamily !== expectedFamily ||
    precision.inputs.some((input) => input.storage !== "float16") ||
    precision.parameters.some((parameter) =>
      parameter.storage !== "float16"
    ) ||
    precision.registerArithmetic !== "float32" ||
    precision.output.slot !== operation.output ||
    precision.output.storage !== (isFinal ? "float32" : "float16") ||
    precision.output.rounding !== (isFinal
      ? "none"
      : "ieee-binary16-round-to-nearest-ties-to-even")
  ) {
    throw contractError(`operation ${operationIndex} precision contract changed`);
  }
  if (
    isFinal &&
    (
      operation.kind !== "conv1d" ||
      operation.shape.kernelSize !== 7 ||
      operation.bias !== undefined ||
      operation.output !== "output"
    )
  ) {
    throw contractError("final FP32 boundary is not no-bias K7 Conv1D");
  }
  if (!isFinal && operation.output === "output") {
    throw contractError("an intermediate operation targets the FP32 output");
  }
}

function requirePackageBindings(
  plan: AceVaeDecoderGraphPlan,
  packageBindings: AceOpt0011VaePackageBindings,
  runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"],
): void {
  const exactPacked = isExactPackedRuntimeProfile(runtimeProfileId);
  const revision7 = isRevision7RuntimeProfile(runtimeProfileId);
  const expectedManifestSha256 = revision7
    ? ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256
    : exactPacked
    ? ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256
    : ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256;
  const expectedManifestBytes = revision7
    ? ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES
    : exactPacked
    ? ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES
    : ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES;
  if (
    !Object.isFrozen(packageBindings) ||
    !Object.isFrozen(packageBindings.operations) ||
    !Object.isFrozen(packageBindings.tensors) ||
    packageBindings.manifestSha256 !== expectedManifestSha256 ||
    packageBindings.manifestByteLength !== expectedManifestBytes ||
    packageBindings.residentWeightBytes !==
      ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    packageBindings.weightFiles.length !==
      ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    packageBindings.weightFiles.some((name, index) =>
      name !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES[index]
    ) ||
    packageBindings.operations.length !== plan.operations.length ||
    Object.keys(packageBindings.tensors).length !==
      plan.requiredTensorNames.length ||
    Object.keys(packageBindings.tensors).length !==
      ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT
  ) {
    throw contractError("package bindings do not preserve the OPT-0011 identity");
  }
  for (const name of plan.requiredTensorNames) {
    const tensor = packageBindings.tensors[name];
    const expectedStorageShape = tensor === undefined
      ? []
      : expectedPackageStorageShape(tensor, runtimeProfileId);
    if (
      tensor === undefined ||
      !Object.isFrozen(tensor) ||
      tensor.logicalTensor !== name ||
      tensor.physicalTensor !== name ||
      tensor.record.logicalTensor !== name ||
      tensor.record.dtype !== "float16" ||
      tensor.record.phase !== "vae" ||
      tensor.record.lifetime !== "vae" ||
      tensor.record.partAxis !== 0 ||
      tensor.record.partStart !== 0 ||
      tensor.record.partEnd !== tensor.logicalShape[0] ||
      !sameShape(tensor.record.logicalShape, tensor.logicalShape) ||
      !sameShape(tensor.record.storageShape, expectedStorageShape) ||
      tensor.record.byteLength !== checkedShapeElements(
        tensor.logicalShape,
        name,
      ) * FLOAT16_BYTES ||
      tensor.binding.offset !== tensor.record.byteOffset ||
      tensor.binding.size !== tensor.record.byteLength
    ) {
      throw contractError(`package tensor ${name} lost its authenticated span`);
    }
  }
  for (const [operationIndex, operation] of plan.operations.entries()) {
    const resolved = packageBindings.operations[operationIndex];
    if (
      resolved === undefined ||
      !Object.isFrozen(resolved) ||
      resolved.operationIndex !== operationIndex ||
      resolved.label !== operation.label ||
      resolved.kind !== operation.kind
    ) {
      throw contractError(
        `package operation ${operationIndex} does not match ${operation.label}/${operation.kind}`,
      );
    }
    requireOperationTensorRoles(packageBindings, operation, resolved);
  }
}

function expectedPackageStorageShape(
  tensor: AceOpt0011VaePackageBindings["tensors"][string],
  runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"],
): readonly number[] {
  if (!isExactPackedRuntimeProfile(runtimeProfileId)) {
    return tensor.logicalShape;
  }
  const [outputChannels, kernelSize, inputChannels] = tensor.logicalShape;
  if (tensor.record.layout === ACE_VAE_K1_FP16_TILE_LAYOUT) {
    if (
      tensor.logicalShape.length !== 3 || kernelSize !== 1 ||
      outputChannels === undefined || inputChannels === undefined
    ) {
      throw contractError(`${tensor.logicalTensor} has invalid packed K1 geometry`);
    }
    return [outputChannels / 128, inputChannels / 32, 32, 128];
  }
  if (
    tensor.record.layout === ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT
  ) {
    if (
      tensor.logicalShape.length !== 3 || outputChannels === undefined ||
      kernelSize === undefined || inputChannels === undefined
    ) {
      throw contractError(
        `${tensor.logicalTensor} has invalid packed transpose geometry`,
      );
    }
    return [kernelSize / 2, 2, inputChannels, outputChannels];
  }
  if (tensor.record.layout === ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT) {
    if (
      !isRevision7RuntimeProfile(runtimeProfileId) ||
      tensor.logicalShape.length !== 3 || kernelSize !== 7 ||
      outputChannels === undefined || outputChannels % 64 !== 0 ||
      inputChannels === undefined || inputChannels % 4 !== 0 ||
      outputChannels !== inputChannels
    ) {
      throw contractError(
        `${tensor.logicalTensor} has invalid revision-7 row-reuse geometry`,
      );
    }
    return [7, inputChannels / 4, outputChannels / 64, 32, 2, 4];
  }
  if (tensor.record.layout === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT) {
    const reuseAxis = REVISION7_TRANSPOSE_K4_REUSE_BY_TENSOR.get(
      tensor.logicalTensor,
    );
    if (
      !isRevision7RuntimeProfile(runtimeProfileId) ||
      tensor.logicalShape.length !== 3 || outputChannels === undefined ||
      kernelSize === undefined || kernelSize % 2 !== 0 ||
      inputChannels === undefined || inputChannels % 4 !== 0 ||
      reuseAxis === undefined
    ) {
      throw contractError(
        `${tensor.logicalTensor} has invalid revision-7 transpose K4 geometry`,
      );
    }
    const outputsPerLane = reuseAxis === "channel" ? 8 : 4;
    const outputTile = outputsPerLane * 32;
    if (outputChannels % outputTile !== 0) {
      throw contractError(
        `${tensor.logicalTensor} has invalid revision-7 transpose K4 tile`,
      );
    }
    return [
      kernelSize / 2,
      2,
      inputChannels / 4,
      outputChannels / outputTile,
      32,
      outputsPerLane,
      4,
    ];
  }
  if (tensor.logicalShape.length === 3 && kernelSize === 1) {
    throw contractError(`${tensor.logicalTensor} is not tile-major packed K1`);
  }
  if (
    tensor.logicalShape.length === 3 && kernelSize !== 1 &&
    tensor.record.layout !== ACE_VAE_CONV1D_FP16_LAYOUT
  ) {
    throw contractError(`${tensor.logicalTensor} changed its K7/native layout`);
  }
  return tensor.logicalShape;
}

function isExactPackedRuntimeProfile(
  runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"],
): boolean {
  return runtimeProfileId ===
      "opt-0028-mixed-fp16-portable-exact-packed-v1" ||
    runtimeProfileId === "opt-0028-mixed-fp16-fixed32-exact-packed-v1" ||
    runtimeProfileId ===
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" ||
    isRevision7RuntimeProfile(runtimeProfileId);
}

function isRevision7RuntimeProfile(
  runtimeProfileId: AceOpt0011Fp16VaeWindowDispatch["runtimeProfileId"],
): boolean {
  return runtimeProfileId === "opt-0054-mixed-fp16-fixed32-revision7-v1" ||
    runtimeProfileId === "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1" ||
    runtimeProfileId === "opt-0088-mixed-fp16-portable-dual-k4-v1";
}

function requireOperationTensorRoles(
  packageBindings: AceOpt0011VaePackageBindings,
  operation: AceVaeDecoderOperation,
  resolved: AceOpt0011VaeOperationBindings,
): void {
  switch (operation.kind) {
    case "conv1d":
      if (
        resolved.kind !== "conv1d" ||
        resolved.weight !== packageBindings.tensors[operation.weight] ||
        !sameShape(resolved.weight.logicalShape, [
          operation.shape.outputChannels,
          operation.shape.kernelSize,
          operation.shape.inputChannels,
        ]) ||
        (operation.bias === undefined
          ? resolved.bias !== undefined
          : resolved.bias === undefined ||
            resolved.bias !== packageBindings.tensors[operation.bias] ||
            !sameShape(resolved.bias.logicalShape, [
              operation.shape.outputChannels,
            ]))
      ) {
        throw contractError(`${operation.label} has forged Conv1D tensor roles`);
      }
      return;
    case "conv-transpose1d":
      if (
        resolved.kind !== "conv-transpose1d" ||
        resolved.weight !== packageBindings.tensors[operation.weight] ||
        resolved.bias !== packageBindings.tensors[operation.bias] ||
        !sameShape(resolved.weight.logicalShape, [
          operation.shape.outputChannels,
          operation.shape.kernelSize,
          operation.shape.inputChannels,
        ]) ||
        !sameShape(resolved.bias.logicalShape, [
          operation.shape.outputChannels,
        ]) ||
        resolved.weight.record.partStart !== 0 ||
        resolved.weight.record.partEnd !== operation.shape.outputChannels
      ) {
        throw contractError(
          `${operation.label} is not one full unsharded transpose binding`,
        );
      }
      return;
    case "snake":
      if (
        resolved.kind !== "snake" ||
        resolved.alpha !== packageBindings.tensors[operation.alpha] ||
        resolved.beta !== packageBindings.tensors[operation.beta] ||
        !sameShape(resolved.alpha.logicalShape, [operation.shape.channels]) ||
        !sameShape(resolved.beta.logicalShape, [operation.shape.channels])
      ) {
        throw contractError(`${operation.label} has forged Snake tensor roles`);
      }
      return;
    case "add":
      if (resolved.kind !== "add") {
        throw contractError(`${operation.label} has forged Add bindings`);
      }
  }
}

function requireRuntimeBindings(
  device: GPUDevice,
  label: string,
  plan: AceVaeDecoderGraphPlan,
  bindings: AceOpt0011Fp16VaeWindowBindings,
): void {
  const workspaceBytes = plan.maximumActivationElements * FLOAT16_BYTES;
  const activationBindings: readonly [string, GPUBufferBinding, number][] = [
    ["staging input", bindings.stagingInput, plan.inputElements * FLOAT32_BYTES],
    ["decoder input", bindings.decoderInput, plan.inputElements * FLOAT16_BYTES],
    ["workspace 0", bindings.workspaces[0], workspaceBytes],
    ["workspace 1", bindings.workspaces[1], workspaceBytes],
    ["workspace 2", bindings.workspaces[2], workspaceBytes],
    ["output", bindings.output, plan.outputElements * FLOAT32_BYTES],
  ];
  const tensorBindings = Object.values(bindings.package.tensors).map(
    (tensor) => tensor.binding,
  );
  for (const [name, binding, requiredBytes] of activationBindings) {
    requireStorageBinding(
      device,
      binding,
      requiredBytes,
      `${label} ${name}`,
    );
  }
  for (const tensor of Object.values(bindings.package.tensors)) {
    requireStorageBinding(
      device,
      tensor.binding,
      tensor.record.byteLength,
      `${label} package tensor ${tensor.logicalTensor}`,
    );
  }
  const allActivations = activationBindings.map(([, binding]) => binding);
  for (let index = 0; index < allActivations.length; index += 1) {
    requireAceDisjointOutput(
      allActivations[index]!,
      [
        ...allActivations.filter((_, other) => other !== index),
        ...tensorBindings,
      ],
      `${label} activation ${index}`,
    );
  }
}

function requireStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): void {
  requireAceBindingBytes(binding, requiredBytes, label);
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  const maximumBuffer = Number(device.limits.maxBufferSize);
  const maximumBinding = Number(device.limits.maxStorageBufferBindingSize);
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const bindingBytes = Number(binding.size ?? bufferBytes - offset);
  if (
    !isValidGpuAlignment(alignment) ||
    !Number.isSafeInteger(maximumBuffer) ||
    !Number.isSafeInteger(maximumBinding) ||
    !Number.isSafeInteger(bufferBytes) ||
    !Number.isSafeInteger(bindingBytes) ||
    bufferBytes < 1 ||
    bufferBytes > maximumBuffer ||
    requiredBytes > maximumBinding ||
    bindingBytes > maximumBinding ||
    offset % alignment !== 0 ||
    requiredBytes % 4 !== 0
  ) {
    throw contractError(`${label} violates device storage limits or alignment`);
  }
}

function requirePlanDeviceContract(
  device: GPUDevice,
  topology: AceOpt0011Fp16VaeWindowTopology,
): void {
  const plan = topology.plan;
  const maximumStorageBytes = Math.max(
    topology.activeStagingInputBytes,
    topology.activeDecoderInputBytes,
    topology.fp16WorkspaceBytes,
    topology.activeOutputBytes,
  );
  const maximumBufferBytes = Math.max(
    maximumStorageBytes,
    topology.dynamicControls.byteLength,
  );
  const maxBufferSize = Number(device.limits.maxBufferSize);
  const maxStorageBufferBindingSize = Number(
    device.limits.maxStorageBufferBindingSize,
  );
  if (
    !Number.isSafeInteger(maxBufferSize) ||
    maxBufferSize < maximumBufferBytes
  ) {
    throw contractError(
      `${plan.inputFrames}-frame window requires maxBufferSize >= ${maximumBufferBytes}`,
    );
  }
  if (
    !Number.isSafeInteger(maxStorageBufferBindingSize) ||
    maxStorageBufferBindingSize < maximumStorageBytes
  ) {
    throw contractError(
      `${plan.inputFrames}-frame window requires maxStorageBufferBindingSize >= ${maximumStorageBytes}`,
    );
  }
}

function requireDeviceContract(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw contractError("runtime requires WebGPU shader-f16");
  }
  const requirements = [
    [
      "maxBufferSize",
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredLimits.maxBufferSize,
    ],
    [
      "maxStorageBufferBindingSize",
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredLimits
        .maxStorageBufferBindingSize,
    ],
    [
      "maxComputeWorkgroupStorageSize",
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredLimits
        .maxComputeWorkgroupStorageSize,
    ],
    [
      "maxComputeInvocationsPerWorkgroup",
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredLimits
        .maxComputeInvocationsPerWorkgroup,
    ],
    ["maxComputeWorkgroupSizeX", 256],
    ["maxComputeWorkgroupSizeY", 8],
    ["maxComputeWorkgroupsPerDimension", 1],
    ["maxBindGroups", 1],
    ["maxBindingsPerBindGroup", 5],
    ["maxStorageBuffersPerShaderStage", 4],
    ["maxUniformBuffersPerShaderStage", 1],
    ["maxDynamicUniformBuffersPerPipelineLayout", 1],
    [
      "maxUniformBufferBindingSize",
      ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES,
    ],
  ] as const;
  for (const [name, minimum] of requirements) {
    const value = Number(device.limits[name]);
    if (!Number.isSafeInteger(value) || value < minimum) {
      throw contractError(`runtime requires ${name} >= ${minimum}`);
    }
  }
  if (
    !isValidGpuAlignment(device.limits.minStorageBufferOffsetAlignment) ||
    !isValidGpuAlignment(device.limits.minUniformBufferOffsetAlignment)
  ) {
    throw contractError("runtime requires valid storage and uniform alignments");
  }
}

async function createDynamicControlBuffer(
  device: GPUDevice,
  plan: AceOpt0011Fp16VaeWindowDynamicControlPlan,
  label: string,
): Promise<GPUBuffer> {
  const [buffer] = await createAceScopedBuffers(
    device,
    [{
      label,
      size: plan.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }],
    "OPT-0011 FP16 VAE dynamic-control allocation",
  );
  const payload = new Uint32Array(plan.byteLength / 4);
  for (const record of plan.records) {
    const wordOffset = record.byteOffset / 4;
    payload[wordOffset] = record.outputBase;
    payload[wordOffset + 1] = record.outputCount;
  }
  try {
    device.queue.writeBuffer(buffer!, 0, payload);
  } catch (error) {
    buffer!.destroy();
    throw error;
  }
  return buffer!;
}

function rangeBinding(
  buffer: GPUBuffer,
  record: AceOpt0011Fp16VaeDecoderControlRecord,
): Readonly<{
  base: number;
  count: number;
  control: GPUBufferBinding;
}> {
  return Object.freeze({
    base: record.outputBase,
    count: record.outputCount,
    control: Object.freeze({
      buffer,
      offset: record.byteOffset,
      size: ACE_OPT_0011_VAE_FP16_DECODER_CONTROL_RECORD_BYTES,
    }),
  });
}

function resolveSlot(
  bindings: AceOpt0011Fp16VaeWindowBindings,
  slot: AceVaeDecoderSlot,
): GPUBufferBinding {
  switch (slot) {
    case "input":
      return bindings.decoderInput;
    case "workspace-0":
      return bindings.workspaces[0];
    case "workspace-1":
      return bindings.workspaces[1];
    case "workspace-2":
      return bindings.workspaces[2];
    case "output":
      return bindings.output;
  }
}

function freezeQuantum(
  quantum: Omit<AceOpt0011Fp16VaeDecoderQuantum, "primitiveCount">,
): AceOpt0011Fp16VaeDecoderQuantum {
  return Object.freeze({ ...quantum, primitiveCount: 1 as const });
}

function freezeControlRecord(
  record: AceOpt0011Fp16VaeDecoderControlRecord,
): AceOpt0011Fp16VaeDecoderControlRecord {
  return Object.freeze(record);
}

function countKinds(
  quanta: readonly AceVaeDecoderQuantumPlan[],
): Record<AceVaeDecoderOperation["kind"], number> {
  const counts = {
    conv1d: 0,
    "conv-transpose1d": 0,
    snake: 0,
    add: 0,
  };
  for (const quantum of quanta) counts[quantum.operationKind] += 1;
  return counts;
}

function freezeKindCounts(
  counts: Record<AceVaeDecoderOperation["kind"], number>,
): AceOpt0011Fp16VaeWindowQuantumFamilyCounts {
  return Object.freeze({ ...counts });
}

function requireDispatchLabel(label: string): void {
  if (typeof label !== "string" || label.length === 0) {
    throw contractError("dispatch label must be non-empty");
  }
}

function sameShape(
  actual: readonly number[],
  expected: readonly number[],
): boolean {
  return actual.length === expected.length &&
    actual.every((extent, index) => extent === expected[index]);
}

function checkedShapeElements(shape: readonly number[], label: string): number {
  let elements = 1;
  for (const extent of shape) {
    if (!Number.isSafeInteger(extent) || extent <= 0) {
      throw contractError(`package tensor ${label} has an unsafe shape`);
    }
    elements = checkedProduct(elements, extent, `${label} elements`);
  }
  return elements;
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product) || product < 0) {
    throw contractError(`${label} is not a safe integer`);
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum) || sum < 0) {
    throw contractError(`${label} is not a safe integer`);
  }
  return sum;
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= 4 &&
    Number.isInteger(Math.log2(value));
}

function contractError(
  message: string,
): AceOpt0011Fp16VaeDecoderContractError {
  return new AceOpt0011Fp16VaeDecoderContractError(`OPT-0011 ${message}`);
}
