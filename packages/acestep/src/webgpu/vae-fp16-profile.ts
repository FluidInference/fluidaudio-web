import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_PACKAGE_CONVERTER_REVISION,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
} from "../model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  type AceLoadedPackageManifest,
} from "../model/package.js";
import { aceSha256Hex } from "../model/sha256.js";
import {
  ACE_OOBLECK_DECODER_CONFIG,
  planAceVaeDecoder,
  type AceVaeDecoderGraphPlan,
  type AceVaeDecoderOperation,
  type AceVaeDecoderSlot,
} from "./vae-decoder.js";

export const ACE_VAE_RUNTIME_PROFILE_IDS = Object.freeze([
  "fp32-oracle",
  "opt-0011-mixed-fp16-portable-v1",
  "opt-0028-mixed-fp16-portable-exact-packed-v1",
  "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
  "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
  "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
  "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
  "opt-0054-mixed-fp16-fixed32-revision7-v1",
  "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
] as const);

export type AceVaeRuntimeProfileId =
  (typeof ACE_VAE_RUNTIME_PROFILE_IDS)[number];

export const ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID =
  "opt-0011-vae-fp16-portable-kernel-set-v1" as const;
export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID =
  "opt-0028-vae-fp16-portable-exact-packed-kernel-set-v1" as const;
export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID =
  "opt-0011-vae-fp16-fixed32-k7-hybrid-kernel-set-v1" as const;
export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID =
  "opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1" as const;
export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID =
  "opt-0028-vae-fp16-fixed32-exact-packed-kernel-set-v1" as const;
export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID =
  "opt-0040-vae-fp16-fixed32-exact-packed-shape-selected-kernel-set-v1" as const;
export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID =
  "opt-0054-vae-fp16-fixed32-revision7-k4-shape-selected-kernel-set-v1" as const;
export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID =
  "opt-0066-vae-fp16-fixed32-dual-k4-quality-kernel-set-v1" as const;
export const ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE = 32 as const;

export const ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256 =
  "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3" as const;
export const ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES = 714_687;
export const ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949" as const;
export const ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES = 715_301;
export const ACE_OPT_0011_VAE_WINDOW_FRAMES = 256;

export const ACE_VAE_FP32_ORACLE_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const ACE_VAE_FP32_ORACLE_MANIFEST_BYTES = 759_692;

export type AceVaeStoragePrecision = "float16" | "float32";

export interface AceVaeRuntimeStorageContract {
  readonly parameters: AceVaeStoragePrecision;
  readonly decoderInput: AceVaeStoragePrecision;
  readonly internalWorkspaces: AceVaeStoragePrecision;
  readonly accumulation: "float32";
  readonly nonlinear: "float32";
  readonly finalOutput: "float32";
  readonly internalStoreRounding:
    | "none"
    | "ieee-binary16-round-to-nearest-ties-to-even";
}

export interface AceVaeRuntimeDeviceLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
}

export interface AceVaeRuntimeProfile {
  readonly id: AceVaeRuntimeProfileId;
  readonly packageProfile: "reference" | "fp16-vae-experimental";
  readonly packageConverterRevision: number;
  readonly manifestSha256: string;
  readonly manifestByteLength: number;
  readonly windowFrames: typeof ACE_OPT_0011_VAE_WINDOW_FRAMES;
  readonly batch: 1;
  readonly kernelBackend:
    | "portable-workgroup"
    | "portable-workgroup-exact-packed"
    | "fixed32-subgroup-k7-hybrid"
    | "fixed32-subgroup-k7-congruent-transpose-hybrid"
    | "fixed32-subgroup-exact-packed"
    | "fixed32-subgroup-exact-packed-shape-selected"
    | "fixed32-subgroup-revision7-k4-shape-selected"
    | "fixed32-subgroup-dual-k4-quality";
  readonly kernelSetId:
    | "ace-vae-fp32-correctness-kernel-set-v1"
    | typeof ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID
    | typeof ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID
    | typeof ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID
    | typeof ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID
    | typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID
    | typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID
    | typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID
    | typeof ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID;
  readonly requiredFeatures: readonly ("shader-f16" | "subgroups")[];
  readonly requiredSubgroupSize:
    | null
    | typeof ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE;
  readonly requiredLimits: AceVaeRuntimeDeviceLimits;
  readonly storage: AceVaeRuntimeStorageContract;
  readonly precisionMapSha256: string | null;
}

export type AceVaeAuthenticatedPackageIdentity = Pick<
  AceLoadedPackageManifest,
  "manifest" | "manifestSha256" | "manifestByteLength"
>;

export type AceVaePrecisionMapOperationKind =
  | "ingress-cast"
  | AceVaeDecoderOperation["kind"];

export interface AceVaePrecisionMapInput {
  readonly role: "input" | "right";
  readonly slot: AceVaeDecoderSlot | "ingress-staging";
  readonly storage: AceVaeStoragePrecision;
}

export interface AceVaePrecisionMapParameter {
  readonly role: "weight" | "bias" | "alpha" | "beta";
  readonly logicalTensor: string;
  readonly storage: AceVaeStoragePrecision;
}

export interface AceVaePrecisionMapOutput {
  readonly slot: AceVaeDecoderSlot;
  readonly storage: AceVaeStoragePrecision;
  readonly rounding:
    | "none"
    | "ieee-binary16-round-to-nearest-ties-to-even";
}

export interface AceVaePrecisionMapEntry {
  readonly sequenceIndex: number;
  readonly graphOperationIndex: number | null;
  readonly label: string;
  readonly kind: AceVaePrecisionMapOperationKind;
  readonly kernelFamily:
    | "f32-to-f16-cast"
    | "conv1d-k1"
    | "conv1d-k7"
    | "conv-transpose1d"
    | "snake"
    | "add";
  readonly inputs: readonly AceVaePrecisionMapInput[];
  readonly parameters: readonly AceVaePrecisionMapParameter[];
  readonly registerArithmetic:
    | "float32"
    | "float16-dot4-partials-then-float32-running-state";
  readonly accumulation: "none" | "float32";
  readonly nonlinearArithmetic: "none" | "float32";
  readonly reductionOrder:
    | "none"
    | "kernel-ascending-then-input-channel-ascending"
    | "kernel-ascending-then-input-channel-groups-of-four-ascending"
    | "tap-ascending-then-input-channel-groups-of-four-ascending";
  readonly contraction:
    | "none"
    | "wgsl-multiply-add-contraction-permitted"
    | "wgsl-f16-dot4-partials-then-f32-add";
  readonly transcendentals: readonly (
    | "exp-alpha"
    | "exp-beta"
    | "sin-alpha-times-input"
  )[];
  readonly output: AceVaePrecisionMapOutput;
}

export interface AceVaePrecisionMap {
  readonly schema: "ace-opt-0011-vae-precision-map-v1";
  readonly profileId:
    | "opt-0011-mixed-fp16-portable-v1"
    | "opt-0028-mixed-fp16-portable-exact-packed-v1"
    | "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
    | "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
    | "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
    | "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
    | "opt-0054-mixed-fp16-fixed32-revision7-v1"
    | "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1";
  readonly kernelSetId:
    | typeof ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID
    | typeof ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID
    | typeof ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID
    | typeof ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID
    | typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID
    | typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID
    | typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID
    | typeof ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID;
  readonly decoderConfigId: typeof ACE_OOBLECK_DECODER_CONFIG.id;
  readonly batch: 1;
  readonly inputFrames: typeof ACE_OPT_0011_VAE_WINDOW_FRAMES;
  readonly graphOperationCount: 88;
  readonly entries: readonly AceVaePrecisionMapEntry[];
}

const ACE_OPT_0011_CANONICAL_DECODER_PLAN = planAceVaeDecoder(
  ACE_OPT_0011_VAE_WINDOW_FRAMES,
);
const ACE_OPT_0054_K7_K4_OPERATION_LABELS: ReadonlySet<string> =
  new Set(ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(
    ({ operationLabel }) => operationLabel,
  ));
const ACE_OPT_0054_TRANSPOSE_K4_OPERATION_LABELS: ReadonlySet<string> =
  new Set(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(
    ({ operationLabel }) => operationLabel,
  ));

export const ACE_OPT_0011_VAE_FP16_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0011-mixed-fp16-portable-v1",
    ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
  );

export const ACE_OPT_0011_VAE_FP16_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0011_VAE_FP16_PRECISION_MAP);

// Frozen independently by the focused contract test. Changing any operation,
// storage boundary, reduction order, or contraction allowance changes this ID.
export const ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256 =
  "465d40d9150912abc7ddfaa3ccf2b80c7639fd304722a495cdd37c16119d64ee" as const;

export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0028-mixed-fp16-portable-exact-packed-v1",
    ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
  );

export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP);

export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256 =
  "6e65d765f1ee5dbab79afb040689cc41cbd7dbd5f1a84ab8807c3f28b48fefe6" as const;

export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
    ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
  );

export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP);

export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256 =
  "d247c0035b6ac33e8bbd33bdae5f4bf64c583e23665734a8a2ea3066fcd999ae" as const;

export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
    ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
  );

export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(
    ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
  );

export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256 =
  "4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85" as const;

export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
    ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
  );

export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP);

export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256 =
  "5ad25bc038b67771535f08b37579ff47c6741352e26906381f046549bf067973" as const;

export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
    ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
  );

export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP);

export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256 =
  "d3d001bde01243708d43b923fe23ec25731a4d202cf6c7c2aa7ff214bfcebe06" as const;

export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0054-mixed-fp16-fixed32-revision7-v1",
    ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID,
    true,
  );

export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP);

// Frozen by the focused OPT-0054 runtime-profile contract test.
export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256 =
  "0875d2609f22e128ec7f768baa65c7e29109607d52c66d012a205a83dea23a40" as const;

export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP =
  createAceOpt0011VaeFp16PrecisionMap(
    ACE_OPT_0011_CANONICAL_DECODER_PLAN,
    "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
    ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
    true,
  );

export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_CANONICAL_JSON =
  JSON.stringify(ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP);

// Frozen by the focused OPT-0066 dual-K4 diagnostic-profile contract test.
export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256 =
  "4815ec86311e401a9bf8cec3f4474d479ef3ac28c925f4824fed4c9ae3b8bc60" as const;

export function hashAceVaePrecisionMap(map: AceVaePrecisionMap): string {
  return aceSha256Hex(new TextEncoder().encode(JSON.stringify(map)));
}

export const ACE_VAE_FP32_ORACLE_PROFILE: Readonly<AceVaeRuntimeProfile> =
  Object.freeze({
    id: "fp32-oracle",
    packageProfile: "reference",
    packageConverterRevision: ACE_PACKAGE_CONVERTER_REVISION,
    manifestSha256: ACE_VAE_FP32_ORACLE_MANIFEST_SHA256,
    manifestByteLength: ACE_VAE_FP32_ORACLE_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "portable-workgroup",
    kernelSetId: "ace-vae-fp32-correctness-kernel-set-v1",
    requiredFeatures: Object.freeze([] as const),
    requiredSubgroupSize: null,
    requiredLimits: Object.freeze({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float32",
      decoderInput: "float32",
      internalWorkspaces: "float32",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "none",
    }),
    precisionMapSha256: null,
  });

export const ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE:
  Readonly<AceVaeRuntimeProfile> = Object.freeze({
    id: "opt-0011-mixed-fp16-portable-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: 5,
    manifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "portable-workgroup",
    kernelSetId: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16"] as const),
    requiredSubgroupSize: null,
    requiredLimits: Object.freeze({
      // Exact B-256 largest resource is one 62,914,560-element FP16 workspace.
      maxBufferSize: 125_829_120,
      maxStorageBufferBindingSize: 125_829_120,
      // Audited portable-kernel envelope. Every later kernel admitted to this
      // kernel-set ID must fit these already-selected device capabilities.
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding:
        "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256: ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
  });

export const ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE =
  Object.freeze({
    id: "opt-0028-mixed-fp16-portable-exact-packed-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "portable-workgroup-exact-packed",
    kernelSetId: ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16"] as const),
    requiredSubgroupSize: null,
    requiredLimits: Object.freeze({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

export const ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE = Object.freeze({
    id: "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: 5,
    manifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "fixed32-subgroup-k7-hybrid",
    kernelSetId:
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
    requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
    requiredLimits: Object.freeze({
      maxBufferSize: 125_829_120,
      maxStorageBufferBindingSize: 125_829_120,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding:
        "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

export const ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE =
  Object.freeze({
    id: "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: 5,
    manifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "fixed32-subgroup-k7-congruent-transpose-hybrid",
    kernelSetId:
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
    requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
    requiredLimits: Object.freeze({
      maxBufferSize: 125_829_120,
      maxStorageBufferBindingSize: 125_829_120,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding:
        "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

export const ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE =
  Object.freeze({
    id: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "fixed32-subgroup-exact-packed",
    kernelSetId: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
    requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
    requiredLimits: Object.freeze({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

export const ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE =
  Object.freeze({
    id: "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision: ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "fixed32-subgroup-exact-packed-shape-selected",
    kernelSetId: ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
    requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
    requiredLimits: Object.freeze({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

/** Explicit revision-6 scalar-FP32-K7 sequential oracle; never a fallback. */
export const ACE_OPT_0054_REVISION6_SCALAR_FP32_SEQUENTIAL_ORACLE_PROFILE =
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE;

export const ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE = Object.freeze({
  id: "opt-0054-mixed-fp16-fixed32-revision7-v1",
  packageProfile: "fp16-vae-experimental",
  packageConverterRevision:
    ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
  batch: 1,
  kernelBackend: "fixed32-subgroup-revision7-k4-shape-selected",
  kernelSetId: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_SET_ID,
  requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
  requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
  requiredLimits: Object.freeze({
    maxBufferSize: 251_658_240,
    maxStorageBufferBindingSize: 251_658_240,
    maxComputeWorkgroupStorageSize: 16 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
  }),
  storage: Object.freeze({
    parameters: "float16",
    decoderInput: "float16",
    internalWorkspaces: "float16",
    accumulation: "float32",
    nonlinear: "float32",
    finalOutput: "float32",
    internalStoreRounding: "ieee-binary16-round-to-nearest-ties-to-even",
  }),
  precisionMapSha256:
    ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256,
} as const satisfies AceVaeRuntimeProfile);

/** Benchmark-only corrected dual-K4 quality identity; never a product default. */
export const ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE =
  Object.freeze({
    id: "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
    packageProfile: "fp16-vae-experimental",
    packageConverterRevision:
      ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
    windowFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    batch: 1,
    kernelBackend: "fixed32-subgroup-dual-k4-quality",
    kernelSetId: ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
    requiredFeatures: Object.freeze(["shader-f16", "subgroups"] as const),
    requiredSubgroupSize: ACE_OPT_0011_VAE_FP16_FIXED_SUBGROUP_SIZE,
    requiredLimits: Object.freeze({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
    }),
    storage: Object.freeze({
      parameters: "float16",
      decoderInput: "float16",
      internalWorkspaces: "float16",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "ieee-binary16-round-to-nearest-ties-to-even",
    }),
    precisionMapSha256:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256,
  } as const satisfies AceVaeRuntimeProfile);

export const ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE =
  "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1" as const;

/**
 * Public product identity for the unchanged, measured OPT-0066 physical
 * decoder. The public identity is intentionally not an
 * `AceVaeRuntimeProfileId`: only the physical profile may cross the backend
 * boundary, so an OPT-0072 request can never fall through to another kernel.
 */
export const ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT =
  Object.freeze({
    id: ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    physicalRuntimeProfileId:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    manifestSha256:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.manifestSha256,
    manifestByteLength:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.manifestByteLength,
    kernelSetId:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
    precisionMapSha256:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256,
  } as const);

/** Authenticate the sole public OPT-0072 to physical-OPT-0066 mapping. */
export function requireAceOpt0072VaeProductionRuntimeProfile(
  runtimeProfile: unknown,
): typeof ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT {
  const contract =
    ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT;
  const physical = ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
  if (
    runtimeProfile !== contract.id ||
    contract.physicalRuntimeProfileId !== physical.id ||
    contract.manifestSha256 !== physical.manifestSha256 ||
    contract.manifestByteLength !== physical.manifestByteLength ||
    contract.kernelSetId !== physical.kernelSetId ||
    contract.precisionMapSha256 !== physical.precisionMapSha256 ||
    hashAceVaePrecisionMap(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
    ) !== physical.precisionMapSha256
  ) {
    throw new AceVaeRuntimeProfileError(
      "OPT-0072 VAE production runtime profile is not authenticated",
    );
  }
  return contract;
}

export interface AceVaeRuntimeProfileSelectionRequest {
  readonly requestedProfile: AceVaeRuntimeProfileId;
  readonly package: AceVaeAuthenticatedPackageIdentity;
  readonly deviceFeatures: Iterable<string>;
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly deviceLimits: AceVaeRuntimeDeviceLimits;
  readonly decoderPlan: AceVaeDecoderGraphPlan;
}

export class AceVaeRuntimeProfileError extends Error {
  readonly code = "INVALID_VAE_RUNTIME_PROFILE";

  constructor(message: string) {
    super(message);
    this.name = "AceVaeRuntimeProfileError";
  }
}

/** Select exactly the requested VAE profile. This function has no fallback. */
export function selectAceVaeRuntimeProfile(
  request: AceVaeRuntimeProfileSelectionRequest,
): AceVaeRuntimeProfile {
  const profile = request.requestedProfile === "fp32-oracle"
    ? ACE_VAE_FP32_ORACLE_PROFILE
    : request.requestedProfile === "opt-0011-mixed-fp16-portable-v1"
      ? ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE
      : request.requestedProfile ===
          "opt-0028-mixed-fp16-portable-exact-packed-v1"
        ? ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE
      : request.requestedProfile ===
          "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1"
        ? ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE
        : request.requestedProfile ===
            "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"
          ? ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE
          : request.requestedProfile ===
              "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
            ? ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE
          : request.requestedProfile ===
              "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
            ? ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE
          : request.requestedProfile ===
              "opt-0054-mixed-fp16-fixed32-revision7-v1"
            ? ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE
          : request.requestedProfile ===
              "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1"
            ? ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
          : undefined;
  if (profile === undefined) {
    throw new AceVaeRuntimeProfileError(
      `Unknown ACE VAE runtime profile ${String(request.requestedProfile)}`,
    );
  }
  requireAceOpt0011VaeDecoderGeometry(request.decoderPlan);
  requireAceVaePackageIdentity(profile, request.package);
  const features = new Set(request.deviceFeatures);
  for (const feature of profile.requiredFeatures) {
    if (!features.has(feature)) {
      throw new AceVaeRuntimeProfileError(
        `ACE VAE profile ${profile.id} requires ${feature}`,
      );
    }
  }
  if (
    profile.requiredSubgroupSize !== null &&
    (request.subgroupMinSize !== profile.requiredSubgroupSize ||
      request.subgroupMaxSize !== profile.requiredSubgroupSize)
  ) {
    throw new AceVaeRuntimeProfileError(
      `ACE VAE profile ${profile.id} requires reported fixed ` +
        `${profile.requiredSubgroupSize}-lane subgroups`,
    );
  }
  requireAceVaeRuntimeLimits(
    profile.id,
    request.deviceLimits,
    profile.requiredLimits,
  );
  if (
    (profile.id === "opt-0011-mixed-fp16-portable-v1" &&
      hashAceVaePrecisionMap(ACE_OPT_0011_VAE_FP16_PRECISION_MAP) !==
        ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256) ||
    (profile.id === "opt-0028-mixed-fp16-portable-exact-packed-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
      ) !==
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256) ||
    (profile.id === "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
      ) !==
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256) ||
    (profile.id ===
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
      ) !==
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256) ||
    (profile.id === "opt-0028-mixed-fp16-fixed32-exact-packed-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
      ) !== ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256)
    ||
    (profile.id ===
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
      ) !== ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256)
    ||
    (profile.id === "opt-0054-mixed-fp16-fixed32-revision7-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
      ) !== ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256)
    ||
    (profile.id === "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1" &&
      hashAceVaePrecisionMap(
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
      ) !==
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256)
  ) {
    throw new AceVaeRuntimeProfileError(
      "OPT-0011 FP16 VAE precision-map identity changed",
    );
  }
  return profile;
}

function requireAceVaeRuntimeLimits(
  profileId: AceVaeRuntimeProfileId,
  available: AceVaeRuntimeDeviceLimits,
  required: AceVaeRuntimeDeviceLimits,
): void {
  for (const name of [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
  ] as const) {
    const value = available[name];
    if (!Number.isSafeInteger(value) || value < required[name]) {
      throw new AceVaeRuntimeProfileError(
        `ACE VAE profile ${profileId} requires ${name} >= ${required[name]}; received ${String(value)}`,
      );
    }
  }
}

export function requireAceOpt0011Fp16VaePackageIdentity(
  loaded: AceVaeAuthenticatedPackageIdentity,
): void {
  requireAceVaePackageIdentity(
    ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
    loaded,
  );
}

export function requireAceOpt0028Fp16VaePackageIdentity(
  loaded: AceVaeAuthenticatedPackageIdentity,
): void {
  requireAceVaePackageIdentity(
    ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
    loaded,
  );
}

export function requireAceOpt0054Fp16VaePackageIdentity(
  loaded: AceVaeAuthenticatedPackageIdentity,
): void {
  requireAceVaePackageIdentity(
    ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
    loaded,
  );
}

export function requireAceOpt0066Fp16VaePackageIdentity(
  loaded: AceVaeAuthenticatedPackageIdentity,
): void {
  requireAceVaePackageIdentity(
    ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
    loaded,
  );
}

export function requireAceOpt0011VaeDecoderGeometry(
  plan: AceVaeDecoderGraphPlan,
): void {
  if (
    JSON.stringify(plan) !== JSON.stringify(ACE_OPT_0011_CANONICAL_DECODER_PLAN)
  ) {
    throw new AceVaeRuntimeProfileError(
      "OPT-0011 VAE profile requires the exact batch-1 256-frame decoder graph",
    );
  }
}

function requireAceVaePackageIdentity(
  profile: AceVaeRuntimeProfile,
  loaded: AceVaeAuthenticatedPackageIdentity,
): void {
  const manifest = loaded.manifest;
  if (
    manifest.profile !== profile.packageProfile ||
    manifest.provenance.converterRevision !==
      profile.packageConverterRevision ||
    loaded.manifestSha256 !== profile.manifestSha256 ||
    loaded.manifestByteLength !== profile.manifestByteLength
  ) {
    throw new AceVaeRuntimeProfileError(
      `ACE VAE profile ${profile.id} requires its exact authenticated package identity`,
    );
  }
}

function createAceOpt0011VaeFp16PrecisionMap(
  plan: AceVaeDecoderGraphPlan,
  profileId: AceVaePrecisionMap["profileId"],
  kernelSetId: AceVaePrecisionMap["kernelSetId"],
  revision7K4 = false,
): AceVaePrecisionMap {
  if (plan.operations.length !== 88) {
    throw new Error(
      `OPT-0011 FP16 VAE precision map expected 88 operations, received ${plan.operations.length}`,
    );
  }
  const entries: AceVaePrecisionMapEntry[] = [Object.freeze({
    sequenceIndex: 0,
    graphOperationIndex: null,
    label: "f32-latent-to-f16-decoder-input",
    kind: "ingress-cast",
    kernelFamily: "f32-to-f16-cast",
    inputs: Object.freeze([Object.freeze({
      role: "input",
      slot: "ingress-staging",
      storage: "float32",
    })]),
    parameters: Object.freeze([]),
    registerArithmetic: "float32",
    accumulation: "none",
    nonlinearArithmetic: "none",
    reductionOrder: "none",
    contraction: "none",
    transcendentals: Object.freeze([]),
    output: Object.freeze({
      slot: "input",
      storage: "float16",
      rounding: "ieee-binary16-round-to-nearest-ties-to-even",
    }),
  })];
  for (let index = 0; index < plan.operations.length; index += 1) {
    entries.push(precisionEntryForOperation(
      plan.operations[index]!,
      index,
      revision7K4,
    ));
  }
  return Object.freeze({
    schema: "ace-opt-0011-vae-precision-map-v1",
    profileId,
    kernelSetId,
    decoderConfigId: ACE_OOBLECK_DECODER_CONFIG.id,
    batch: 1,
    inputFrames: ACE_OPT_0011_VAE_WINDOW_FRAMES,
    graphOperationCount: 88,
    entries: Object.freeze(entries),
  });
}

function precisionEntryForOperation(
  operation: AceVaeDecoderOperation,
  graphOperationIndex: number,
  revision7K4: boolean,
): AceVaePrecisionMapEntry {
  const base = {
    sequenceIndex: graphOperationIndex + 1,
    graphOperationIndex,
    label: operation.label,
    kind: operation.kind,
    inputs: Object.freeze([
      Object.freeze({
        role: "input" as const,
        slot: operation.input,
        storage: "float16" as const,
      }),
      ...(operation.kind === "add"
        ? [Object.freeze({
            role: "right" as const,
            slot: operation.right,
            storage: "float16" as const,
          })]
        : []),
    ]),
    registerArithmetic: "float32" as const,
    output: Object.freeze({
      slot: operation.output,
      storage: operation.output === "output"
        ? "float32" as const
        : "float16" as const,
      rounding: operation.output === "output"
        ? "none" as const
        : "ieee-binary16-round-to-nearest-ties-to-even" as const,
    }),
  };
  switch (operation.kind) {
    case "conv1d": {
      if (operation.shape.kernelSize !== 1 && operation.shape.kernelSize !== 7) {
        throw new Error(
          `OPT-0011 precision map has undeclared Conv1D K${operation.shape.kernelSize}`,
        );
      }
      const k4 = revision7K4 && operation.shape.kernelSize === 7 &&
        ACE_OPT_0054_K7_K4_OPERATION_LABELS.has(operation.label);
      return Object.freeze({
        ...base,
        kernelFamily: operation.shape.kernelSize === 1
          ? "conv1d-k1"
          : "conv1d-k7",
        parameters: Object.freeze([
          precisionParameter("weight", operation.weight),
          ...(operation.bias === undefined
            ? []
            : [precisionParameter("bias", operation.bias)]),
        ]),
        registerArithmetic: k4
          ? "float16-dot4-partials-then-float32-running-state"
          : "float32",
        accumulation: "float32",
        nonlinearArithmetic: "none",
        reductionOrder: k4
          ? "kernel-ascending-then-input-channel-groups-of-four-ascending"
          : "kernel-ascending-then-input-channel-ascending",
        contraction: k4
          ? "wgsl-f16-dot4-partials-then-f32-add"
          : "wgsl-multiply-add-contraction-permitted",
        transcendentals: Object.freeze([]),
      });
    }
    case "conv-transpose1d": {
      const k4 = revision7K4 &&
        ACE_OPT_0054_TRANSPOSE_K4_OPERATION_LABELS.has(operation.label);
      return Object.freeze({
        ...base,
        kernelFamily: "conv-transpose1d",
        parameters: Object.freeze([
          precisionParameter("weight", operation.weight),
          precisionParameter("bias", operation.bias),
        ]),
        registerArithmetic: k4
          ? "float16-dot4-partials-then-float32-running-state"
          : "float32",
        accumulation: "float32",
        nonlinearArithmetic: "none",
        reductionOrder: k4
          ? "tap-ascending-then-input-channel-groups-of-four-ascending"
          : "kernel-ascending-then-input-channel-ascending",
        contraction: k4
          ? "wgsl-f16-dot4-partials-then-f32-add"
          : "wgsl-multiply-add-contraction-permitted",
        transcendentals: Object.freeze([]),
      });
    }
    case "snake":
      return Object.freeze({
        ...base,
        kernelFamily: "snake",
        parameters: Object.freeze([
          precisionParameter("alpha", operation.alpha),
          precisionParameter("beta", operation.beta),
        ]),
        accumulation: "none",
        nonlinearArithmetic: "float32",
        reductionOrder: "none",
        contraction: "wgsl-multiply-add-contraction-permitted",
        transcendentals: Object.freeze([
          "exp-alpha" as const,
          "exp-beta" as const,
          "sin-alpha-times-input" as const,
        ]),
      });
    case "add":
      return Object.freeze({
        ...base,
        kernelFamily: "add",
        parameters: Object.freeze([]),
        accumulation: "none",
        nonlinearArithmetic: "none",
        reductionOrder: "none",
        contraction: "none",
        transcendentals: Object.freeze([]),
      });
  }
}

function precisionParameter(
  role: AceVaePrecisionMapParameter["role"],
  logicalTensor: string,
): AceVaePrecisionMapParameter {
  return Object.freeze({ role, logicalTensor, storage: "float16" });
}
