export const ACE_PACKAGE_FORMAT = "ace-step-webgpu-v1" as const;
export const ACE_PACKAGE_ALIGNMENT_BYTES = 256;
export const ACE_PORTABLE_STORAGE_BINDING_BYTES = 128 * 1024 * 1024;
export const ACE_MAX_WEIGHT_SHARD_BYTES = 120 * 1024 * 1024;
export const ACE_MODEL_SNAPSHOT_REVISION =
  "19671f406d603126926c1b7e2adc169acbcade22" as const;
export const ACE_PLANNER_SNAPSHOT_REVISION =
  "148d8ea0225bdab342ee1ae3a354275ccd60ca80" as const;
export const ACE_REFERENCE_SOURCE_REVISION =
  "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0" as const;
export const ACE_PACKAGE_CONVERTER_REVISION = 4;
export const ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION = 6;
export const ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION = 7;
export const ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION = 8;
export const ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS = 1_510_404_096;
export const ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES = 3_020_808_192;
export const ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT = 456;
export const ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS = 84_395_776;
export const ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES = 168_791_552;
export const ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT = 145;
// Revision 4 has 146 VAE records because one FP32 transpose weight is split.
// FP16 makes it one 83,886,080-byte record; the silence latent stays separate.
export const ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT = 145;
export const ACE_DIT_GEMM_TILE_LAYOUT =
  "dit-gemm-n128-k32-tile-major-v1" as const;
export const ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION =
  "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1" as const;
export const ACE_DIT_GEMM_FP16_TRANSFORMATION =
  "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1" as const;
export const ACE_DIT_DENSE_FP16_TILE_LAYOUT =
  "dit-gemm-n256-k32-tile-major-v1" as const;
export const ACE_DIT_DENSE_FP16_TRANSFORMATION =
  "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1" as const;
export const ACE_DIT_DENSE_K4_FP16_LAYOUT =
  "dit-gemm-n128-k4-output4-lane32-k4-v1" as const;
export const ACE_DIT_DENSE_K4_FP16_TRANSFORMATION =
  "bf16-to-ieee-fp16-dit-gemm-n128-k4-output4-lane32-k4-v1" as const;
export const ACE_VAE_CONV1D_LAYOUT =
  "conv1d-output-kernel-input-f32-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_LAYOUT =
  "conv-transpose1d-output-kernel-input-f32-v1" as const;
export const ACE_VAE_CHANNEL_VECTOR_LAYOUT = "channel-vector-f32-v1" as const;
export const ACE_VAE_CONV1D_FP16_LAYOUT =
  "conv1d-output-kernel-input-f16-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_FP16_LAYOUT =
  "conv-transpose1d-output-kernel-input-f16-v1" as const;
export const ACE_VAE_K1_FP16_TILE_LAYOUT =
  "conv1d-k1-cout128-cin32-tile-major-f16-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT =
  "conv-transpose1d-phase-tap-input-output-f16-v1" as const;
export const ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT =
  "k7-cin4-cout-band64-lane32-output2-cin-element4" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT =
  "ace-opt-0048-phase-tap-cin4-cout-tile-lane-output-k4-f16-v1" as const;
export const ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT =
  "channel-vector-f16-v1" as const;
export const ACE_VAE_CONV1D_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-oik-to-oki-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-iok-to-oki-v1" as const;
export const ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION =
  "bf16-to-fp32-flatten-1-c-1-to-c-v1" as const;
export const ACE_VAE_BIAS_FP16_TRANSFORMATION =
  "bf16-to-fp32-to-ieee-fp16-v1" as const;
export const ACE_VAE_CONV1D_FP16_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1" as const;
export const ACE_VAE_K1_FP16_TILE_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1" as const;
export const ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-oik-to-k7-cin4-cout-band64-lane32-output2-cin-element4-ieee-fp16-v1" as const;
export const ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION =
  "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-cin4-cout-tile-lane-output-k4-ieee-fp16-v1" as const;
export const ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION =
  "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1" as const;

export const ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS = Object.freeze([
  { operationLabel: "block-0-res-1-conv1", tensor: "vae.decoder.block.0.res_unit1.conv1.weight", channels: 1_024, dilation: 1 },
  { operationLabel: "block-0-res-2-conv1", tensor: "vae.decoder.block.0.res_unit2.conv1.weight", channels: 1_024, dilation: 3 },
  { operationLabel: "block-0-res-3-conv1", tensor: "vae.decoder.block.0.res_unit3.conv1.weight", channels: 1_024, dilation: 9 },
  { operationLabel: "block-1-res-1-conv1", tensor: "vae.decoder.block.1.res_unit1.conv1.weight", channels: 512, dilation: 1 },
  { operationLabel: "block-1-res-2-conv1", tensor: "vae.decoder.block.1.res_unit2.conv1.weight", channels: 512, dilation: 3 },
  { operationLabel: "block-1-res-3-conv1", tensor: "vae.decoder.block.1.res_unit3.conv1.weight", channels: 512, dilation: 9 },
  { operationLabel: "block-3-res-1-conv1", tensor: "vae.decoder.block.3.res_unit1.conv1.weight", channels: 128, dilation: 1 },
  { operationLabel: "block-3-res-2-conv1", tensor: "vae.decoder.block.3.res_unit2.conv1.weight", channels: 128, dilation: 3 },
  { operationLabel: "block-3-res-3-conv1", tensor: "vae.decoder.block.3.res_unit3.conv1.weight", channels: 128, dilation: 9 },
  { operationLabel: "block-4-res-1-conv1", tensor: "vae.decoder.block.4.res_unit1.conv1.weight", channels: 128, dilation: 1 },
  { operationLabel: "block-4-res-2-conv1", tensor: "vae.decoder.block.4.res_unit2.conv1.weight", channels: 128, dilation: 3 },
  { operationLabel: "block-4-res-3-conv1", tensor: "vae.decoder.block.4.res_unit3.conv1.weight", channels: 128, dilation: 9 },
] as const);

export const ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS = Object.freeze([
  { operationLabel: "block-1-conv-t1", tensor: "vae.decoder.block.1.conv_t1.weight", inputChannels: 1_024, outputChannels: 512, stride: 6, reuseAxis: "channel" },
  { operationLabel: "block-2-conv-t1", tensor: "vae.decoder.block.2.conv_t1.weight", inputChannels: 512, outputChannels: 256, stride: 4, reuseAxis: "channel" },
  { operationLabel: "block-3-conv-t1", tensor: "vae.decoder.block.3.conv_t1.weight", inputChannels: 256, outputChannels: 128, stride: 4, reuseAxis: "row" },
  { operationLabel: "block-4-conv-t1", tensor: "vae.decoder.block.4.conv_t1.weight", inputChannels: 128, outputChannels: 128, stride: 2, reuseAxis: "row" },
] as const);
export const ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR =
  "vae.decoder.block.0.conv_t1.weight" as const;

export type AcePackageProfile =
  | "reference"
  | "fp16"
  | "fp16-vae-experimental"
  | "fp16-dit-dense-experimental";
export type AcePackageFileKind =
  | "conversion-plan"
  | "upstream-asset"
  | "license"
  | "weights"
  | "constant";
export type AceTensorDtype =
  | "float16"
  | "float32"
  | "uint32"
  | "uint32-bf16-pairs";
export type AceTensorPhase =
  | "planner"
  | "text"
  | "conditioner"
  | "semantic"
  | "dit"
  | "vae"
  | "constants";
export type AceTensorLifetime = AceTensorPhase | "initial-latent";
export type AceTensorLayout =
  | "source-row-major"
  | "source-row-major-bf16-pairs-lsb-u32"
  | "row-shard-axis0"
  | "row-shard-axis0-bf16-pairs-lsb-u32"
  | "contiguous-nct-f32"
  | typeof ACE_DIT_GEMM_TILE_LAYOUT
  | typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT
  | typeof ACE_DIT_DENSE_K4_FP16_LAYOUT
  | typeof ACE_VAE_CONV1D_LAYOUT
  | typeof ACE_VAE_CONV_TRANSPOSE1D_LAYOUT
  | typeof ACE_VAE_CHANNEL_VECTOR_LAYOUT
  | typeof ACE_VAE_CONV1D_FP16_LAYOUT
  | typeof ACE_VAE_CONV_TRANSPOSE1D_FP16_LAYOUT
  | typeof ACE_VAE_K1_FP16_TILE_LAYOUT
  | typeof ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT
  | typeof ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT
  | typeof ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT
  | typeof ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT;
export type AceTensorTransformation =
  | "preserve-bf16-bits-pack-u32-pairs"
  | "bf16-to-ieee-fp16"
  | "bf16-to-fp32"
  | typeof ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
  | typeof ACE_DIT_GEMM_FP16_TRANSFORMATION
  | typeof ACE_DIT_DENSE_FP16_TRANSFORMATION
  | typeof ACE_DIT_DENSE_K4_FP16_TRANSFORMATION
  | typeof ACE_VAE_CONV1D_TRANSFORMATION
  | typeof ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION
  | typeof ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION
  | typeof ACE_VAE_BIAS_FP16_TRANSFORMATION
  | typeof ACE_VAE_CONV1D_FP16_TRANSFORMATION
  | typeof ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION
  | typeof ACE_VAE_K1_FP16_TILE_TRANSFORMATION
  | typeof ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
  | typeof ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION
  | typeof ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION
  | typeof ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION
  | "validated-pytorch-zip-storage-extraction";

export interface AcePackageSourceRecord {
  readonly key: string;
  readonly component: string;
  readonly repository: string;
  readonly revision: string;
  readonly path: string;
  readonly byteLength: number;
  readonly sha256: string;
  readonly tensorCount?: number;
  readonly parameterCount?: number;
  readonly headerLength?: number;
  readonly headerSha256?: string;
  readonly inventorySha256?: string;
}

export interface AcePackageFileRecord {
  readonly name: string;
  readonly byteLength: number;
  readonly sha256: string;
  readonly kind: AcePackageFileKind;
}

export interface AcePackageTensorRecord {
  readonly shard: string;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly dtype: AceTensorDtype;
  readonly logicalShape: readonly number[];
  readonly storageShape: readonly number[];
  readonly layout: AceTensorLayout;
  readonly phase: AceTensorPhase;
  readonly lifetime: AceTensorLifetime;
  readonly source: string;
  readonly transformation: AceTensorTransformation;
  readonly logicalTensor: string;
  readonly partAxis: 0;
  readonly partStart: number;
  readonly partEnd: number;
}

export interface AcePackageAccounting {
  readonly sourceTensors: number;
  readonly directlyIncluded: number;
  readonly consumedByTransform: number;
  readonly excluded: number;
  readonly outputTensorsBeforeRowSharding: number;
  readonly constantTensors: number;
  readonly outputTensorsAfterRowSharding: number;
}

export interface AcePackageLicenseRecord {
  readonly component: string;
  readonly spdx: string;
  readonly notice: string;
  readonly source: string;
}

export interface AcePackageProvenance {
  readonly converterRevision: number;
  readonly aceSnapshot: string;
  readonly plannerSnapshot: string;
  readonly referenceRepository: string;
  readonly referenceCommit: string;
  readonly referenceLicenseGitBlob: string;
  readonly referenceLicenseSha256: string;
  readonly determinism: string;
}

export interface AcePackageManifest {
  readonly format: typeof ACE_PACKAGE_FORMAT;
  readonly profile: AcePackageProfile;
  readonly alignment: typeof ACE_PACKAGE_ALIGNMENT_BYTES;
  readonly portableStorageBindingBytes: typeof ACE_PORTABLE_STORAGE_BINDING_BYTES;
  readonly source: readonly AcePackageSourceRecord[];
  readonly files: readonly AcePackageFileRecord[];
  readonly tensors: Readonly<Record<string, AcePackageTensorRecord>>;
  readonly accounting: AcePackageAccounting;
  readonly licenses: readonly AcePackageLicenseRecord[];
  readonly provenance: AcePackageProvenance;
}

export class AcePackageManifestError extends Error {
  readonly code = "INVALID_MODEL_MANIFEST";

  constructor(message: string) {
    super(message);
    this.name = "AcePackageManifestError";
  }
}

const ROOT_KEYS = [
  "format",
  "profile",
  "alignment",
  "portableStorageBindingBytes",
  "source",
  "files",
  "tensors",
  "accounting",
  "licenses",
  "provenance",
] as const;
const SOURCE_BASE_KEYS = [
  "key",
  "component",
  "repository",
  "revision",
  "path",
  "byteLength",
  "sha256",
] as const;
const SOURCE_TENSOR_KEYS = [
  ...SOURCE_BASE_KEYS,
  "tensorCount",
  "parameterCount",
  "headerLength",
  "headerSha256",
  "inventorySha256",
] as const;
const FILE_KEYS = ["name", "byteLength", "sha256", "kind"] as const;
const TENSOR_KEYS = [
  "shard",
  "byteOffset",
  "byteLength",
  "dtype",
  "logicalShape",
  "storageShape",
  "layout",
  "phase",
  "lifetime",
  "source",
  "transformation",
  "logicalTensor",
  "partAxis",
  "partStart",
  "partEnd",
] as const;
const ACCOUNTING_KEYS = [
  "sourceTensors",
  "directlyIncluded",
  "consumedByTransform",
  "excluded",
  "outputTensorsBeforeRowSharding",
  "constantTensors",
  "outputTensorsAfterRowSharding",
] as const;
const LICENSE_KEYS = ["component", "spdx", "notice", "source"] as const;
const PROVENANCE_KEYS = [
  "converterRevision",
  "aceSnapshot",
  "plannerSnapshot",
  "referenceRepository",
  "referenceCommit",
  "referenceLicenseGitBlob",
  "referenceLicenseSha256",
  "determinism",
] as const;
const FILE_KINDS = new Set<AcePackageFileKind>([
  "conversion-plan",
  "upstream-asset",
  "license",
  "weights",
  "constant",
]);
const DTYPE_BYTES: Readonly<Record<AceTensorDtype, number>> = {
  float16: 2,
  float32: 4,
  uint32: 4,
  "uint32-bf16-pairs": 4,
};
const TENSOR_PHASES = new Set<AceTensorPhase>([
  "planner",
  "text",
  "conditioner",
  "semantic",
  "dit",
  "vae",
  "constants",
]);
const TENSOR_LIFETIMES = new Set<AceTensorLifetime>([
  ...TENSOR_PHASES,
  "initial-latent",
]);
const TENSOR_TRANSFORMATIONS = new Set([
  "preserve-bf16-bits-pack-u32-pairs",
  "bf16-to-ieee-fp16",
  "bf16-to-fp32",
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_FP16_TRANSFORMATION,
  ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
  ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION,
  ACE_VAE_BIAS_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_FP16_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
  ACE_VAE_K1_FP16_TILE_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
  ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
  ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
  "validated-pytorch-zip-storage-extraction",
]);
const NATIVE_LAYOUT_BY_TRANSFORMATION: Readonly<
  Partial<Record<AceTensorTransformation, AceTensorLayout>>
> = {
  [ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION]: ACE_DIT_GEMM_TILE_LAYOUT,
  [ACE_DIT_GEMM_FP16_TRANSFORMATION]: ACE_DIT_GEMM_TILE_LAYOUT,
  [ACE_DIT_DENSE_FP16_TRANSFORMATION]: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
  [ACE_DIT_DENSE_K4_FP16_TRANSFORMATION]: ACE_DIT_DENSE_K4_FP16_LAYOUT,
  [ACE_VAE_CONV1D_TRANSFORMATION]: ACE_VAE_CONV1D_LAYOUT,
  [ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION]: ACE_VAE_CONV_TRANSPOSE1D_LAYOUT,
  [ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION]: ACE_VAE_CHANNEL_VECTOR_LAYOUT,
  [ACE_VAE_CONV1D_FP16_TRANSFORMATION]: ACE_VAE_CONV1D_FP16_LAYOUT,
  [ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION]:
    ACE_VAE_CONV_TRANSPOSE1D_FP16_LAYOUT,
  [ACE_VAE_K1_FP16_TILE_TRANSFORMATION]: ACE_VAE_K1_FP16_TILE_LAYOUT,
  [ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION]:
    ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  [ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION]:
    ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  [ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION]:
    ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  [ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION]:
    ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
};

export interface AcePackageManifestParseOptions {
  /** Accepted only after the caller authenticates the exact OPT-0054 digest. */
  readonly authenticatedVaeConverterRevision?: 7;
  /**
   * Accepted only after the caller authenticates the exact OPT-0009 manifest
   * digest. Ordinary parsing remains pinned to the current revision 8 layout.
   */
  readonly authenticatedDitDenseConverterRevision?: 7;
}
const VAE_CONV1D_SOURCE =
  /^vae-weights:decoder\.(?:conv[12]|block\.\d+\.res_unit[123]\.conv[12])\.weight_v$/;
const VAE_CONV_TRANSPOSE1D_SOURCE =
  /^vae-weights:decoder\.block\.\d+\.conv_t1\.weight_v$/;
const VAE_CHANNEL_VECTOR_SOURCE =
  /^vae-weights:decoder\.(?:snake1|block\.\d+\.(?:snake1|res_unit[123]\.snake[12]))\.(?:alpha|beta)$/;
const VAE_BIAS_SOURCE =
  /^vae-weights:decoder\.(?:conv1|block\.[0-4]\.(?:conv_t1|res_unit[123]\.conv[12]))\.bias$/;

function buildRevision7VaeRuntimeShapes(): ReadonlyMap<string, readonly number[]> {
  const shapes = new Map<string, readonly number[]>();
  const blockChannels = [1_024, 512, 256, 128, 128] as const;
  const blockInputs = [2_048, 1_024, 512, 256, 128] as const;
  const transposeKernels = [20, 12, 8, 8, 4] as const;
  shapes.set("vae-weights:decoder.conv1.weight_v", [2_048, 7, 64]);
  shapes.set("vae-weights:decoder.conv2.weight_v", [2, 7, 128]);
  shapes.set("vae-weights:decoder.conv1.bias", [2_048]);
  for (let block = 0; block < blockChannels.length; block += 1) {
    const channels = blockChannels[block]!;
    shapes.set(
      `vae-weights:decoder.block.${block}.conv_t1.weight_v`,
      [channels, transposeKernels[block]!, blockInputs[block]!],
    );
    shapes.set(`vae-weights:decoder.block.${block}.conv_t1.bias`, [channels]);
    for (let residual = 1; residual <= 3; residual += 1) {
      shapes.set(
        `vae-weights:decoder.block.${block}.res_unit${residual}.conv1.weight_v`,
        [channels, 7, channels],
      );
      shapes.set(
        `vae-weights:decoder.block.${block}.res_unit${residual}.conv2.weight_v`,
        [channels, 1, channels],
      );
      for (const convolution of [1, 2] as const) {
        shapes.set(
          `vae-weights:decoder.block.${block}.res_unit${residual}.conv${convolution}.bias`,
          [channels],
        );
      }
      for (const snake of [1, 2] as const) {
        for (const parameter of ["alpha", "beta"] as const) {
          shapes.set(
            `vae-weights:decoder.block.${block}.res_unit${residual}.snake${snake}.${parameter}`,
            [channels],
          );
        }
      }
    }
    for (const parameter of ["alpha", "beta"] as const) {
      shapes.set(
        `vae-weights:decoder.block.${block}.snake1.${parameter}`,
        [blockInputs[block]!],
      );
    }
  }
  for (const parameter of ["alpha", "beta"] as const) {
    shapes.set(`vae-weights:decoder.snake1.${parameter}`, [128]);
  }
  if (shapes.size !== ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT) {
    throw new Error(`revision-7 VAE contract has ${shapes.size} tensors`);
  }
  return shapes;
}

const VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE = buildRevision7VaeRuntimeShapes();
const VAE_REVISION7_ROW_REUSE_TENSORS: ReadonlySet<string> = new Set<string>(
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map((contract) => contract.tensor),
);
const VAE_REVISION7_TRANSPOSE_K4_TENSORS: ReadonlyMap<
  string,
  "channel" | "row"
> = new Map<string, "channel" | "row">(
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map((contract) => [
    contract.tensor,
    contract.reuseAxis,
  ] as const),
);
const DIT_GEMM_SOURCE =
  /^ace-turbo-weights:decoder\.(?:condition_embedder\.weight|time_embed(?:_r)?\.(?:linear_[12]|time_proj)\.weight|layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|cross_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight)$/;
const DIT_REPEATED_DENSE_SOURCE =
  /^ace-turbo-weights:decoder\.layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|cross_attn\.(?:q_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$/;
const SILENCE_LATENT_TENSOR_NAME = "constants.silence_latent";
const SILENCE_LATENT_SHARD = "constants/silence-latent-f32.bin";
const SILENCE_LATENT_SOURCE =
  "ace-silence-latent:silence_latent/data/0";
const SILENCE_LATENT_SHAPE = [1, 64, 15_000] as const;
const SILENCE_LATENT_BYTES = 3_840_000;
const REQUIRED_LICENSE_FILES: Readonly<
  Record<string, { readonly byteLength: number; readonly sha256: string }>
> = {
  "licenses/ACE-Step-LICENSE": {
    byteLength: 1_064,
    sha256: "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357",
  },
  "licenses/Apache-2.0-LICENSE": {
    byteLength: 11_358,
    sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
  },
  "licenses/Qwen-NOTICE.txt": {
    byteLength: 439,
    sha256: "c57cecae352eb5793befd1f28f44f351e148c9a28044d855b8c361c562195f0b",
  },
};

/** Validate the converter's canonical v1 manifest and reject every extension. */
export function parseAcePackageManifest(
  value: unknown,
  expectedProfile?: AcePackageProfile,
  options: AcePackageManifestParseOptions = {},
): AcePackageManifest {
  const root = requireRecord(value, "manifest");
  requireExactKeys(root, ROOT_KEYS, "manifest");
  if (root.format !== ACE_PACKAGE_FORMAT) fail("unknown package format");
  if (
    root.profile !== "reference" &&
    root.profile !== "fp16" &&
    root.profile !== "fp16-vae-experimental" &&
    root.profile !== "fp16-dit-dense-experimental"
  ) {
    fail("unknown package profile");
  }
  if (expectedProfile !== undefined && root.profile !== expectedProfile) {
    fail(`package profile ${String(root.profile)} does not match ${expectedProfile}`);
  }
  if (
    options.authenticatedVaeConverterRevision !== undefined &&
    (options.authenticatedVaeConverterRevision !== 7 ||
      expectedProfile !== "fp16-vae-experimental" ||
      root.profile !== "fp16-vae-experimental")
  ) {
    fail("authenticated VAE converter revision has the wrong profile");
  }
  if (
    options.authenticatedDitDenseConverterRevision !== undefined &&
    (options.authenticatedDitDenseConverterRevision !== 7 ||
      expectedProfile !== "fp16-dit-dense-experimental" ||
      root.profile !== "fp16-dit-dense-experimental")
  ) {
    fail("authenticated legacy converter revision has the wrong profile");
  }
  if (root.alignment !== ACE_PACKAGE_ALIGNMENT_BYTES) {
    fail("unsupported package alignment");
  }
  if (root.portableStorageBindingBytes !== ACE_PORTABLE_STORAGE_BINDING_BYTES) {
    fail("unsupported storage-binding contract");
  }

  const sources = parseSources(root.source);
  const files = parseFiles(root.files);
  const accounting = parseAccounting(root.accounting);
  const licenses = parseLicenses(root.licenses);
  const provenance = parseProvenance(
    root.provenance,
    root.profile,
    options.authenticatedVaeConverterRevision,
    options.authenticatedDitDenseConverterRevision,
  );
  const tensors = parseTensors(
    root.tensors,
    root.profile,
    sources,
    files,
    accounting,
    provenance.converterRevision,
  );
  if (root.profile === "fp16-vae-experimental") {
    validateExperimentalVaePayload(tensors, provenance.converterRevision);
  }
  if (root.profile === "fp16-dit-dense-experimental") {
    validateExperimentalDitDensePayload(tensors, provenance.converterRevision);
  }

  const sourceTensorCount = sources.reduce(
    (total, source) => total + (source.tensorCount ?? 0),
    0,
  );
  if (sourceTensorCount !== accounting.sourceTensors) {
    fail("source tensor counts do not match accounting");
  }
  if (
    accounting.directlyIncluded +
      accounting.consumedByTransform +
      accounting.excluded !==
      accounting.sourceTensors ||
    accounting.outputTensorsBeforeRowSharding !== accounting.directlyIncluded ||
    accounting.outputTensorsAfterRowSharding !== Object.keys(tensors).length ||
    accounting.outputTensorsAfterRowSharding <
      accounting.outputTensorsBeforeRowSharding + accounting.constantTensors
  ) {
    fail("manifest accounting is internally inconsistent");
  }
  if (!new Set(licenses.map((license) => license.spdx)).has("MIT")) {
    fail("ACE-Step MIT license metadata is missing");
  }
  if (!new Set(licenses.map((license) => license.spdx)).has("Apache-2.0")) {
    fail("Qwen Apache-2.0 license metadata is missing");
  }

  return {
    format: ACE_PACKAGE_FORMAT,
    profile: root.profile,
    alignment: ACE_PACKAGE_ALIGNMENT_BYTES,
    portableStorageBindingBytes: ACE_PORTABLE_STORAGE_BINDING_BYTES,
    source: sources,
    files,
    tensors,
    accounting,
    licenses,
    provenance,
  };
}

export interface AceLogicalTensorPart {
  readonly tensorName: string;
  readonly tensor: AcePackageTensorRecord;
}

export interface AceResolvedLogicalTensor {
  readonly logicalTensor: string;
  readonly logicalShape: readonly number[];
  readonly dtype: AceTensorDtype;
  readonly transformation: AceTensorTransformation;
  readonly parts: readonly AceLogicalTensorPart[];
}

export interface AceResolvedLogicalTensorRows extends AceLogicalTensorPart {
  /** Requested half-open logical row interval carried by this part. */
  readonly logicalRowStart: number;
  readonly logicalRowEnd: number;
  /** Half-open row interval relative to the beginning of this part. */
  readonly partRowStart: number;
  readonly partRowEnd: number;
  /** Logical scalar offset/count within the part, before packed-BF16 word addressing. */
  readonly storageElementOffset: number;
  readonly storageElementCount: number;
}

/** Resolve a complete logical tensor into canonical runtime-axis-0 pieces. */
export function resolveAceLogicalTensor(
  manifest: AcePackageManifest,
  logicalTensor: string,
): AceResolvedLogicalTensor {
  if (logicalTensor.length === 0) fail("logical tensor name must be nonempty");
  const parts = Object.entries(manifest.tensors)
    .filter(([, tensor]) => tensor.logicalTensor === logicalTensor)
    .sort((left, right) => left[1].partStart - right[1].partStart)
    .map(([tensorName, tensor]) => ({ tensorName, tensor }));
  const first = parts[0]?.tensor;
  if (first === undefined) fail(`logical tensor ${logicalTensor} is absent`);
  return {
    logicalTensor,
    logicalShape: first.logicalShape,
    dtype: first.dtype,
    transformation: first.transformation,
    parts,
  };
}

/** Resolve an embedding or other tensor's logical row interval across shards. */
export function resolveAceLogicalTensorRows(
  manifest: AcePackageManifest,
  logicalTensor: string,
  rowStart: number,
  rowEnd: number,
): readonly AceResolvedLogicalTensorRows[] {
  const resolved = resolveAceLogicalTensor(manifest, logicalTensor);
  const rows = resolved.logicalShape[0]!;
  if (
    !Number.isSafeInteger(rowStart) ||
    !Number.isSafeInteger(rowEnd) ||
    rowStart < 0 ||
    rowStart >= rowEnd ||
    rowEnd > rows
  ) {
    fail(`logical tensor ${logicalTensor} has an invalid requested row interval`);
  }
  const rowElements = checkedProduct(
    resolved.logicalShape.slice(1),
    `logical tensor ${logicalTensor} row shape`,
  );
  const result: AceResolvedLogicalTensorRows[] = [];
  for (const part of resolved.parts) {
    const logicalRowStart = Math.max(rowStart, part.tensor.partStart);
    const logicalRowEnd = Math.min(rowEnd, part.tensor.partEnd);
    if (logicalRowStart >= logicalRowEnd) continue;
    const partRowStart = logicalRowStart - part.tensor.partStart;
    const partRowEnd = logicalRowEnd - part.tensor.partStart;
    result.push({
      ...part,
      logicalRowStart,
      logicalRowEnd,
      partRowStart,
      partRowEnd,
      storageElementOffset: partRowStart * rowElements,
      storageElementCount: (partRowEnd - partRowStart) * rowElements,
    });
  }
  return result;
}

function parseSources(value: unknown): AcePackageSourceRecord[] {
  if (!Array.isArray(value) || value.length === 0) fail("source inventory is empty");
  const keys = new Set<string>();
  return value.map((raw, index) => {
    const path = `source[${index}]`;
    const source = requireRecord(raw, path);
    const tensorSource = Object.hasOwn(source, "tensorCount");
    requireExactKeys(source, tensorSource ? SOURCE_TENSOR_KEYS : SOURCE_BASE_KEYS, path);
    const key = requireNonEmptyString(source.key, `${path}.key`);
    if (keys.has(key)) fail(`duplicate source key ${key}`);
    keys.add(key);
    const repository = requireNonEmptyString(source.repository, `${path}.repository`);
    const revision = requireShaLike(source.revision, 40, `${path}.revision`);
    if (
      (repository === "ACE-Step/Ace-Step1.5" &&
        revision !== ACE_MODEL_SNAPSHOT_REVISION) ||
      (repository === "ACE-Step/acestep-5Hz-lm-0.6B" &&
        revision !== ACE_PLANNER_SNAPSHOT_REVISION) ||
      (repository !== "ACE-Step/Ace-Step1.5" &&
        repository !== "ACE-Step/acestep-5Hz-lm-0.6B")
    ) {
      fail(`${path} is not from a pinned source snapshot`);
    }
    const base = {
      key,
      component: requireNonEmptyString(source.component, `${path}.component`),
      repository,
      revision,
      path: requireSafeRelativePath(source.path, `${path}.path`),
      byteLength: requireNonNegativeInteger(source.byteLength, `${path}.byteLength`),
      sha256: requireSha256(source.sha256, `${path}.sha256`),
    };
    if (!tensorSource) return base;
    return {
      ...base,
      tensorCount: requirePositiveInteger(source.tensorCount, `${path}.tensorCount`),
      parameterCount: requirePositiveInteger(
        source.parameterCount,
        `${path}.parameterCount`,
      ),
      headerLength: requirePositiveInteger(source.headerLength, `${path}.headerLength`),
      headerSha256: requireSha256(source.headerSha256, `${path}.headerSha256`),
      inventorySha256: requireSha256(
        source.inventorySha256,
        `${path}.inventorySha256`,
      ),
    };
  });
}

function parseFiles(value: unknown): AcePackageFileRecord[] {
  if (!Array.isArray(value) || value.length === 0) fail("file inventory is empty");
  const names = new Set<string>();
  const digestLengths = new Map<string, number>();
  const records = value.map((raw, index) => {
    const path = `files[${index}]`;
    const file = requireRecord(raw, path);
    requireExactKeys(file, FILE_KEYS, path);
    const name = requireSafeRelativePath(file.name, `${path}.name`);
    if (name === "manifest.json") fail("manifest must not list or hash itself");
    if (names.has(name)) fail(`duplicate package filename ${name}`);
    names.add(name);
    const kind = file.kind;
    if (typeof kind !== "string" || !FILE_KINDS.has(kind as AcePackageFileKind)) {
      fail(`${path}.kind is invalid`);
    }
    const byteLength = requireNonNegativeInteger(file.byteLength, `${path}.byteLength`);
    if (kind === "weights" && byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
      fail(`${name} exceeds the weight-shard limit`);
    }
    if (kind === "constant" && byteLength > ACE_PORTABLE_STORAGE_BINDING_BYTES) {
      fail(`${name} exceeds the portable binding limit`);
    }
    const sha256 = requireSha256(file.sha256, `${path}.sha256`);
    const digestLength = digestLengths.get(sha256);
    if (digestLength !== undefined && digestLength !== byteLength) {
      fail(`${path} conflicts with another file using the same content identity`);
    }
    digestLengths.set(sha256, byteLength);
    return {
      name,
      byteLength,
      sha256,
      kind: kind as AcePackageFileKind,
    };
  });
  for (let index = 1; index < records.length; index += 1) {
    if (records[index - 1]!.name >= records[index]!.name) {
      fail("file inventory is not in canonical name order");
    }
  }
  if (records.filter((file) => file.kind === "conversion-plan").length !== 1) {
    fail("package must contain exactly one conversion plan");
  }
  const byName = new Map(records.map((record) => [record.name, record]));
  for (const [name, expected] of Object.entries(REQUIRED_LICENSE_FILES)) {
    const actual = byName.get(name);
    if (
      actual?.kind !== "license" ||
      actual.byteLength !== expected.byteLength ||
      actual.sha256 !== expected.sha256
    ) {
      fail(`required license payload ${name} is missing or changed`);
    }
  }
  return records;
}

function parseAccounting(value: unknown): AcePackageAccounting {
  const raw = requireRecord(value, "accounting");
  requireExactKeys(raw, ACCOUNTING_KEYS, "accounting");
  return Object.fromEntries(
    ACCOUNTING_KEYS.map((key) => [
      key,
      requireNonNegativeInteger(raw[key], `accounting.${key}`),
    ]),
  ) as unknown as AcePackageAccounting;
}

function parseLicenses(value: unknown): AcePackageLicenseRecord[] {
  if (!Array.isArray(value) || value.length !== 2) {
    fail("license metadata must contain the canonical ACE-Step and Qwen records");
  }
  const components = new Set<string>();
  const spdxIdentifiers = new Set<string>();
  const records = value.map((raw, index) => {
    const path = `licenses[${index}]`;
    const license = requireRecord(raw, path);
    requireExactKeys(license, LICENSE_KEYS, path);
    const component = requireNonEmptyString(license.component, `${path}.component`);
    const spdx = requireNonEmptyString(license.spdx, `${path}.spdx`);
    if (components.has(component) || spdxIdentifiers.has(spdx)) {
      fail("license metadata contains a duplicate component or SPDX identifier");
    }
    components.add(component);
    spdxIdentifiers.add(spdx);
    return {
      component,
      spdx,
      notice: requireNonEmptyString(license.notice, `${path}.notice`),
      source: requireNonEmptyString(license.source, `${path}.source`),
    };
  });
  if (!spdxIdentifiers.has("MIT") || !spdxIdentifiers.has("Apache-2.0")) {
    fail("license metadata does not identify the canonical upstream licenses");
  }
  return records;
}

function parseProvenance(
  value: unknown,
  profile: AcePackageProfile,
  authenticatedVaeConverterRevision?: 7,
  authenticatedDitDenseConverterRevision?: 7,
): AcePackageProvenance {
  const raw = requireRecord(value, "provenance");
  requireExactKeys(raw, PROVENANCE_KEYS, "provenance");
  const converterRevision = profile === "fp16-vae-experimental"
    ? authenticatedVaeConverterRevision ??
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION
    : profile === "fp16-dit-dense-experimental"
      ? authenticatedDitDenseConverterRevision ??
        ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION
      : ACE_PACKAGE_CONVERTER_REVISION;
  if (raw.converterRevision !== converterRevision) {
    fail("unsupported converter revision");
  }
  if (raw.aceSnapshot !== ACE_MODEL_SNAPSHOT_REVISION) {
    fail("provenance ACE snapshot is not pinned");
  }
  if (raw.plannerSnapshot !== ACE_PLANNER_SNAPSHOT_REVISION) {
    fail("provenance planner snapshot is not pinned");
  }
  if (raw.referenceCommit !== ACE_REFERENCE_SOURCE_REVISION) {
    fail("provenance reference source commit is not pinned");
  }
  if (raw.referenceRepository !== "https://github.com/ace-step/ACE-Step-1.5.git") {
    fail("provenance reference repository is not pinned");
  }
  if (
    raw.determinism !==
    "sorted source/output inventories, fixed transforms, canonical JSON"
  ) {
    fail("provenance determinism contract is not canonical");
  }
  return {
    converterRevision,
    aceSnapshot: ACE_MODEL_SNAPSHOT_REVISION,
    plannerSnapshot: ACE_PLANNER_SNAPSHOT_REVISION,
    referenceRepository: raw.referenceRepository,
    referenceCommit: ACE_REFERENCE_SOURCE_REVISION,
    referenceLicenseGitBlob: requireExactIdentity(
      raw.referenceLicenseGitBlob,
      "600451d484a555c1273baa2602f32a37fdd0d0ab",
      "provenance.referenceLicenseGitBlob",
    ),
    referenceLicenseSha256: requireExactIdentity(
      raw.referenceLicenseSha256,
      "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357",
      "provenance.referenceLicenseSha256",
    ),
    determinism: raw.determinism,
  };
}

function parseTensors(
  value: unknown,
  profile: AcePackageProfile,
  sources: readonly AcePackageSourceRecord[],
  files: readonly AcePackageFileRecord[],
  accounting: AcePackageAccounting,
  converterRevision: number,
): Record<string, AcePackageTensorRecord> {
  const rawTensors = requireRecord(value, "tensors");
  const sourcesByKey = new Map(sources.map((source) => [source.key, source]));
  const filesByName = new Map(files.map((file) => [file.name, file]));
  const tensors: Record<string, AcePackageTensorRecord> = Object.create(null) as Record<
    string,
    AcePackageTensorRecord
  >;
  const spansByShard = new Map<string, Array<readonly [string, number, number]>>();
  const partsByLogical = new Map<
    string,
    Array<readonly [string, AcePackageTensorRecord]>
  >();
  const referencedTensorFiles = new Set<string>();
  const names = Object.keys(rawTensors);
  if (names.length === 0) fail("tensor inventory is empty");
  for (let index = 1; index < names.length; index += 1) {
    if (names[index - 1]! >= names[index]!) {
      fail("tensor inventory is not in canonical name order");
    }
  }

  for (const name of names) {
    if (name.length === 0) fail("tensor name is empty");
    const path = `tensors.${name}`;
    const raw = requireRecord(rawTensors[name], path);
    requireExactKeys(raw, TENSOR_KEYS, path);
    const shard = requireSafeRelativePath(raw.shard, `${path}.shard`);
    const file = filesByName.get(shard);
    if (file?.kind !== "weights" && file?.kind !== "constant") {
      fail(`${path}.shard is not a tensor payload`);
    }
    referencedTensorFiles.add(shard);
    const byteOffset = requireNonNegativeInteger(raw.byteOffset, `${path}.byteOffset`);
    const byteLength = requireNonNegativeInteger(raw.byteLength, `${path}.byteLength`);
    if (byteOffset % ACE_PACKAGE_ALIGNMENT_BYTES !== 0) {
      fail(`${path}.byteOffset is not ${ACE_PACKAGE_ALIGNMENT_BYTES}-byte aligned`);
    }
    if (!Number.isSafeInteger(byteOffset + byteLength) || byteOffset + byteLength > file.byteLength) {
      fail(`${path} exceeds its shard`);
    }
    const dtype = raw.dtype;
    if (typeof dtype !== "string" || !Object.hasOwn(DTYPE_BYTES, dtype)) {
      fail(`${path}.dtype is invalid`);
    }
    const logicalShape = requireShape(raw.logicalShape, `${path}.logicalShape`);
    const storageShape = requireShape(raw.storageShape, `${path}.storageShape`);
    const phase = raw.phase;
    const lifetime = raw.lifetime;
    if (typeof phase !== "string" || !TENSOR_PHASES.has(phase as AceTensorPhase)) {
      fail(`${path}.phase is invalid`);
    }
    if (
      typeof lifetime !== "string" ||
      !TENSOR_LIFETIMES.has(lifetime as AceTensorLifetime)
    ) {
      fail(`${path}.lifetime is invalid`);
    }
    const transformation = requireNonEmptyString(
      raw.transformation,
      `${path}.transformation`,
    ) as AceTensorTransformation;
    const authenticatedRev7Dense =
      profile === "fp16-dit-dense-experimental" &&
      converterRevision === 7 &&
      transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION;
    if (!TENSOR_TRANSFORMATIONS.has(transformation) && !authenticatedRev7Dense) {
      fail(`${path}.transformation is invalid`);
    }
    validateStoragePolicy(
      profile,
      dtype as AceTensorDtype,
      transformation,
      path,
      converterRevision,
    );
    const source = requireNonEmptyString(raw.source, `${path}.source`);
    const sourceSeparator = source.indexOf(":");
    const sourceRecord = sourcesByKey.get(source.slice(0, sourceSeparator));
    if (
      sourceSeparator <= 0 ||
      sourceSeparator === source.length - 1 ||
      sourceRecord === undefined
    ) {
      fail(`${path}.source does not name a manifest source`);
    }
    const logicalTensor = requireNonEmptyString(raw.logicalTensor, `${path}.logicalTensor`);
    validateNativeTensorContract(
      transformation,
      source,
      logicalShape,
      phase as AceTensorPhase,
      path,
    );
    if (raw.partAxis !== 0) fail(`${path}.partAxis must be zero`);
    const partStart = requireNonNegativeInteger(raw.partStart, `${path}.partStart`);
    const partEnd = requirePositiveInteger(raw.partEnd, `${path}.partEnd`);
    if (partStart >= partEnd || partEnd > logicalShape[0]!) {
      fail(`${path} has an invalid logical part extent`);
    }
    if (
      phase === "constants"
        ? lifetime !== "initial-latent" ||
          transformation !== "validated-pytorch-zip-storage-extraction" ||
          sourceRecord.key !== "ace-silence-latent"
        : lifetime !== phase || sourceRecord.tensorCount === undefined
    ) {
      fail(`${path} violates its phase, lifetime, or source-kind contract`);
    }
    const partShape = [partEnd - partStart, ...logicalShape.slice(1)];
    const partElements = checkedProduct(partShape, `${path}.logicalShape`);
    const rowSharded = partStart !== 0 || partEnd !== logicalShape[0];
    if (
      (transformation === ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION ||
        transformation === ACE_DIT_GEMM_FP16_TRANSFORMATION ||
        transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION ||
        transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION ||
        transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION ||
        transformation ===
          ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION ||
        transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION ||
        transformation === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION) &&
      rowSharded
    ) {
      fail(`${path} must contain its complete tile-major DiT GEMM matrix`);
    }
    const layoutBase = rowSharded ? "row-shard-axis0" : "source-row-major";
    let expectedStorageShape: number[];
    if (transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION) {
      const [columns, inner] = logicalShape;
      if (columns! % 128 !== 0 || inner! % 4 !== 0) {
        fail(`${path} has an invalid packed DiT K4 shape`);
      }
      expectedStorageShape = [columns! / 128, inner! / 4, 4, 32, 4];
    } else if (transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION) {
      const [outputChannels, kernel, inputChannels] = logicalShape;
      if (
        kernel !== 1 ||
        outputChannels !== inputChannels ||
        outputChannels! % 128 !== 0 ||
        inputChannels! % 32 !== 0
      ) {
        fail(`${path} has an invalid packed K1 shape`);
      }
      expectedStorageShape = [
        outputChannels! / 128,
        inputChannels! / 32,
        32,
        128,
      ];
    } else if (transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION) {
      const [outputChannels, kernel, inputChannels] = logicalShape;
      if (
        kernel !== 7 ||
        outputChannels! % 128 !== 0 ||
        inputChannels! % 4 !== 0
      ) {
        fail(`${path} has an invalid row-reuse K7 shape`);
      }
      expectedStorageShape = [
        kernel!,
        inputChannels! / 4,
        outputChannels! / 64,
        32,
        2,
        4,
      ];
    } else if (
      transformation ===
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
    ) {
      const [outputChannels, kernel, inputChannels] = logicalShape;
      if (kernel! % 2 !== 0) {
        fail(`${path} has an invalid polyphase transpose shape`);
      }
      expectedStorageShape = [kernel! / 2, 2, inputChannels!, outputChannels!];
    } else if (
      transformation === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION
    ) {
      const [outputChannels, kernel, inputChannels] = logicalShape;
      const reuseAxis = VAE_REVISION7_TRANSPOSE_K4_TENSORS.get(name);
      const outputsPerLane = reuseAxis === "channel"
        ? 8
        : reuseAxis === "row"
          ? 4
          : 0;
      const outputTile = 32 * outputsPerLane;
      if (
        kernel! % 2 !== 0 ||
        inputChannels! % 4 !== 0 ||
        outputsPerLane === 0 ||
        outputChannels! % outputTile !== 0
      ) {
        fail(`${path} has an invalid transpose K4 shape`);
      }
      expectedStorageShape = [
        kernel! / 2,
        2,
        inputChannels! / 4,
        outputChannels! / outputTile,
        32,
        outputsPerLane,
        4,
      ];
    } else {
      expectedStorageShape = dtype === "uint32-bf16-pairs"
        ? [Math.ceil(partElements / 2)]
        : partShape;
    }
    const expectedByteLength = checkedProduct(
      expectedStorageShape,
      `${path}.storageShape`,
    ) * DTYPE_BYTES[dtype as AceTensorDtype];
    const expectedLayout =
      phase === "constants"
        ? "contiguous-nct-f32"
        : (NATIVE_LAYOUT_BY_TRANSFORMATION[transformation] ??
          (dtype === "uint32-bf16-pairs"
            ? `${layoutBase}-bf16-pairs-lsb-u32`
            : layoutBase));
    if (
      !arraysEqual(storageShape, expectedStorageShape) ||
      byteLength !== expectedByteLength ||
      raw.layout !== expectedLayout
    ) {
      fail(`${path} storage shape, bytes, or layout is inconsistent`);
    }
    const tensor: AcePackageTensorRecord = {
      shard,
      byteOffset,
      byteLength,
      dtype: dtype as AceTensorDtype,
      logicalShape,
      storageShape,
      layout: expectedLayout as AceTensorLayout,
      phase: phase as AceTensorPhase,
      lifetime: lifetime as AceTensorLifetime,
      source,
      transformation,
      logicalTensor,
      partAxis: 0,
      partStart,
      partEnd,
    };
    tensors[name] = tensor;
    const spans = spansByShard.get(shard) ?? [];
    spans.push([name, byteOffset, byteOffset + byteLength]);
    spansByShard.set(shard, spans);
    const parts = partsByLogical.get(logicalTensor) ?? [];
    parts.push([name, tensor]);
    partsByLogical.set(logicalTensor, parts);
  }

  for (const [shard, spans] of spansByShard) {
    spans.sort((left, right) => left[1] - right[1]);
    let cursor = 0;
    for (const [name, start, end] of spans) {
      if (start < cursor) fail(`tensor ${name} overlaps another span in ${shard}`);
      cursor = end;
    }
  }
  let constantLogicalTensors = 0;
  for (const [logicalName, parts] of partsByLogical) {
    parts.sort((left, right) => left[1].partStart - right[1].partStart);
    const first = parts[0]![1];
    let cursor = 0;
    for (const [partName, part] of parts) {
      if (
        part.partStart !== cursor ||
        !arraysEqual(part.logicalShape, first.logicalShape) ||
        part.dtype !== first.dtype ||
        part.source !== first.source ||
        part.transformation !== first.transformation ||
        part.phase !== first.phase ||
        part.lifetime !== first.lifetime
      ) {
        fail(`logical tensor ${logicalName} has an inconsistent part ${partName}`);
      }
      cursor = part.partEnd;
    }
    if (cursor !== first.logicalShape[0]) {
      fail(`logical tensor ${logicalName} is incomplete`);
    }
    if (first.phase === "constants") constantLogicalTensors += 1;
  }
  if (
    constantLogicalTensors !== accounting.constantTensors ||
    partsByLogical.size !==
      accounting.outputTensorsBeforeRowSharding + accounting.constantTensors
  ) {
    fail("logical tensor counts do not match accounting");
  }
  for (const file of files) {
    if (
      (file.kind === "weights" || file.kind === "constant") &&
      !referencedTensorFiles.has(file.name)
    ) {
      fail(`tensor payload ${file.name} is not referenced by any tensor`);
    }
  }
  return tensors;
}

function validateExperimentalVaePayload(
  tensors: Readonly<Record<string, AcePackageTensorRecord>>,
  converterRevision: number,
): void {
  const constants = Object.entries(tensors).filter(
    ([, tensor]) => tensor.phase === "constants",
  );
  if (
    constants.length !== 1 ||
    constants[0]![0] !== SILENCE_LATENT_TENSOR_NAME
  ) {
    fail(
      "experimental FP16 VAE package must contain exactly one canonical " +
      "constants.silence_latent record",
    );
  }
  const silence = constants[0]![1];
  if (
    silence.shard !== SILENCE_LATENT_SHARD ||
    silence.byteOffset !== 0 ||
    silence.byteLength !== SILENCE_LATENT_BYTES ||
    silence.dtype !== "float32" ||
    !arraysEqual(silence.logicalShape, SILENCE_LATENT_SHAPE) ||
    !arraysEqual(silence.storageShape, SILENCE_LATENT_SHAPE) ||
    silence.layout !== "contiguous-nct-f32" ||
    silence.lifetime !== "initial-latent" ||
    silence.source !== SILENCE_LATENT_SOURCE ||
    silence.transformation !== "validated-pytorch-zip-storage-extraction" ||
    silence.logicalTensor !== SILENCE_LATENT_TENSOR_NAME ||
    silence.partAxis !== 0 ||
    silence.partStart !== 0 ||
    silence.partEnd !== 1
  ) {
    fail(
      "experimental FP16 VAE package has a non-canonical " +
      "constants.silence_latent record",
    );
  }
  const vaeTensors = Object.entries(tensors).filter(
    ([, tensor]) => tensor.phase === "vae",
  );
  const logicalNames = new Set(vaeTensors.map(([, tensor]) => tensor.logicalTensor));
  if (
    logicalNames.size !== ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT ||
    vaeTensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT
  ) {
    fail(
      "experimental FP16 VAE must contain exactly " +
      `${ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT} logical tensors and ` +
      `${ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT} unsharded records`,
    );
  }
  const revision7 = converterRevision ===
    ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION;
  if (revision7) {
    const sources = new Set(vaeTensors.map(([, tensor]) => tensor.source));
    if (
      sources.size !== vaeTensors.length ||
      sources.size !== VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.size ||
      [...sources].some(
        (source) => !VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.has(source),
      )
    ) {
      fail(
        "experimental FP16 VAE package does not contain the exact " +
        "revision-7 decoder source inventory",
      );
    }
  }
  let parameterElements = 0;
  let parameterBytes = 0;
  for (const [name, tensor] of vaeTensors) {
    let validLayout: boolean;
    if (revision7) {
      const expectedShape = VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.get(
        tensor.source,
      )!;
      const sourceTensor = tensor.source.slice("vae-weights:".length);
      const expectedName = sourceTensor.endsWith(".weight_v")
        ? `vae.${sourceTensor.slice(0, -".weight_v".length)}.weight`
        : `vae.${sourceTensor}`;
      let expectedTransformation: AceTensorTransformation;
      let expectedLayout: AceTensorLayout;
      if (VAE_REVISION7_ROW_REUSE_TENSORS.has(name)) {
        expectedTransformation = ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION;
        expectedLayout = ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT;
      } else if (VAE_REVISION7_TRANSPOSE_K4_TENSORS.has(name)) {
        expectedTransformation =
          ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION;
        expectedLayout = ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT;
      } else if (name === ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR) {
        expectedTransformation =
          ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION;
        expectedLayout = ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT;
      } else if (VAE_CONV1D_SOURCE.test(tensor.source)) {
        expectedTransformation = tensor.logicalShape[1] === 1
          ? ACE_VAE_K1_FP16_TILE_TRANSFORMATION
          : ACE_VAE_CONV1D_FP16_TRANSFORMATION;
        expectedLayout = tensor.logicalShape[1] === 1
          ? ACE_VAE_K1_FP16_TILE_LAYOUT
          : ACE_VAE_CONV1D_FP16_LAYOUT;
      } else if (VAE_BIAS_SOURCE.test(tensor.source)) {
        expectedTransformation = ACE_VAE_BIAS_FP16_TRANSFORMATION;
        expectedLayout = "source-row-major";
      } else {
        expectedTransformation = ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION;
        expectedLayout = ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT;
      }
      validLayout =
        name === expectedName &&
        arraysEqual(tensor.logicalShape, expectedShape) &&
        tensor.transformation === expectedTransformation &&
        tensor.layout === expectedLayout;
    } else {
      const expectedWeightTransformation = VAE_CONV1D_SOURCE.test(tensor.source)
        ? tensor.logicalShape[1] === 1
          ? ACE_VAE_K1_FP16_TILE_TRANSFORMATION
          : ACE_VAE_CONV1D_FP16_TRANSFORMATION
        : VAE_CONV_TRANSPOSE1D_SOURCE.test(tensor.source)
          ? ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
          : undefined;
      validLayout =
        (tensor.transformation === ACE_VAE_BIAS_FP16_TRANSFORMATION ||
          tensor.transformation === ACE_VAE_CONV1D_FP16_TRANSFORMATION ||
          tensor.transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION ||
          tensor.transformation ===
            ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION ||
          tensor.transformation ===
            ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION) &&
        (expectedWeightTransformation === undefined ||
          tensor.transformation === expectedWeightTransformation);
    }
    if (
      tensor.dtype !== "float16" ||
      tensor.partStart !== 0 ||
      tensor.partEnd !== tensor.logicalShape[0] ||
      tensor.logicalTensor !== name ||
      !validLayout
    ) {
      fail(
        `experimental FP16 VAE tensor ${name} violates its exact ` +
        `${revision7 ? "revision-7" : "revision-6"} layout contract`,
      );
    }
    parameterElements += checkedProduct(
      tensor.logicalShape,
      `experimental FP16 VAE tensor ${name}`,
    );
    parameterBytes += tensor.byteLength;
  }
  if (
    parameterElements !== ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS ||
    parameterBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    parameterBytes !== parameterElements * DTYPE_BYTES.float16
  ) {
    fail(
      "experimental FP16 VAE payload is not exactly " +
      `${ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS} elements/` +
      `${ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES} bytes`,
    );
  }
}

function validateExperimentalDitDensePayload(
  tensors: Readonly<Record<string, AcePackageTensorRecord>>,
  converterRevision: number,
): void {
  const constants = Object.entries(tensors).filter(
    ([, tensor]) => tensor.phase === "constants",
  );
  if (
    constants.length !== 1 ||
    constants[0]![0] !== SILENCE_LATENT_TENSOR_NAME
  ) {
    fail(
      "experimental FP16 DiT dense package must retain the canonical " +
      "constants.silence_latent record",
    );
  }
  const denseSuffixes = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
  ] as const;
  const allLayerSuffixes = [
    "scale_shift_table",
    "self_attn_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "cross_attn_norm.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.k_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.q_norm.weight",
    "cross_attn.k_norm.weight",
    "cross_attn.o_proj.weight",
    "mlp_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
  ] as const;
  const expectedNames = new Set(
    Array.from({ length: 24 }, (_, layer) =>
      allLayerSuffixes.map((suffix) => `ace.decoder.layers.${layer}.${suffix}`)
    ).flat(),
  );
  const denseNames = new Set(
    Array.from({ length: 24 }, (_, layer) =>
      denseSuffixes.map((suffix) => `ace.decoder.layers.${layer}.${suffix}`)
    ).flat(),
  );
  const crossCacheNames = new Set(
    Array.from({ length: 24 }, (_, layer) => [
      `ace.decoder.layers.${layer}.cross_attn.k_proj.weight`,
      `ace.decoder.layers.${layer}.cross_attn.v_proj.weight`,
    ]).flat(),
  );
  const ditTensors = Object.entries(tensors).filter(
    ([, tensor]) => tensor.phase === "dit",
  );
  if (
    expectedNames.size !== ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT ||
    ditTensors.length !== expectedNames.size ||
    ditTensors.some(([name]) => !expectedNames.has(name)) ||
    Object.keys(tensors).length !== expectedNames.size + 1
  ) {
    fail(
      "experimental mixed DiT package is not exactly the 24 repeated " +
      "layers plus the canonical constant",
    );
  }
  let parameterElements = 0;
  let parameterBytes = 0;
  const authenticatedRev7 = converterRevision === 7;
  for (const [name, tensor] of ditTensors) {
    const commonValid = tensor.logicalTensor === name &&
      tensor.partStart === 0 &&
      tensor.partEnd === tensor.logicalShape[0];
    const valid = denseNames.has(name)
      ? commonValid &&
        tensor.dtype === "float16" &&
        tensor.logicalShape.length === 2 &&
        (authenticatedRev7
          ? tensor.layout === ACE_DIT_DENSE_FP16_TILE_LAYOUT &&
            tensor.transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION &&
            tensor.logicalShape[0]! % 256 === 0 &&
            tensor.logicalShape[1]! % 32 === 0 &&
            arraysEqual(tensor.storageShape, tensor.logicalShape)
          : converterRevision ===
              ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION &&
            tensor.layout === ACE_DIT_DENSE_K4_FP16_LAYOUT &&
            tensor.transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION &&
            tensor.logicalShape[0]! % 128 === 0 &&
            tensor.logicalShape[1]! % 4 === 0 &&
            arraysEqual(tensor.storageShape, [
              tensor.logicalShape[0]! / 128,
              tensor.logicalShape[1]! / 4,
              4,
              32,
              4,
            ]))
      : crossCacheNames.has(name)
        ? commonValid &&
          tensor.dtype === "uint32-bf16-pairs" &&
          tensor.layout === ACE_DIT_GEMM_TILE_LAYOUT &&
          tensor.transformation === ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
        : commonValid &&
          tensor.dtype === "uint32-bf16-pairs" &&
          tensor.layout === "source-row-major-bf16-pairs-lsb-u32" &&
          tensor.transformation === "preserve-bf16-bits-pack-u32-pairs";
    if (!valid) {
      fail(
        `experimental mixed DiT tensor ${name} violates its exact ` +
        "storage contract",
      );
    }
    parameterElements += checkedProduct(
      tensor.logicalShape,
      `experimental FP16 DiT dense tensor ${name}`,
    );
    parameterBytes += tensor.byteLength;
  }
  if (
    parameterElements !== ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS ||
    parameterBytes !== ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES
  ) {
    fail(
      "experimental mixed DiT layer payload does not match its exact " +
      "element and byte totals",
    );
  }
}

function validateStoragePolicy(
  profile: AcePackageProfile,
  dtype: AceTensorDtype,
  transformation: AceTensorTransformation,
  path: string,
  converterRevision: number,
): void {
  const valid =
    (transformation === "preserve-bf16-bits-pack-u32-pairs" &&
      (profile === "reference" ||
        profile === "fp16-dit-dense-experimental") &&
      dtype === "uint32-bf16-pairs") ||
    (transformation === ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION &&
      (profile === "reference" ||
        profile === "fp16-dit-dense-experimental") &&
      dtype === "uint32-bf16-pairs") ||
    (transformation === "bf16-to-ieee-fp16" &&
      (profile === "fp16" || profile === "fp16-vae-experimental") &&
      dtype === "float16") ||
    (transformation === ACE_DIT_GEMM_FP16_TRANSFORMATION &&
      (profile === "fp16" || profile === "fp16-vae-experimental") &&
      dtype === "float16") ||
    (transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION &&
      profile === "fp16-dit-dense-experimental" &&
      converterRevision ===
        ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION &&
      dtype === "float16") ||
    (transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION &&
      profile === "fp16-dit-dense-experimental" &&
      converterRevision === 7 &&
      dtype === "float16") ||
    ((transformation === "bf16-to-fp32" ||
      transformation === ACE_VAE_CONV1D_TRANSFORMATION ||
      transformation === ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION ||
      transformation === ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION) &&
      profile !== "fp16-vae-experimental" &&
      dtype === "float32") ||
    (transformation === "validated-pytorch-zip-storage-extraction" &&
      dtype === "float32") ||
    ((transformation === ACE_VAE_BIAS_FP16_TRANSFORMATION ||
      transformation === ACE_VAE_CONV1D_FP16_TRANSFORMATION ||
      transformation === ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION ||
      transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION ||
      transformation ===
        ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION ||
      transformation === ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION) &&
      profile === "fp16-vae-experimental" &&
      dtype === "float16") ||
    ((transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION ||
      transformation === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION) &&
      profile === "fp16-vae-experimental" &&
      converterRevision ===
        ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION &&
      dtype === "float16");
  if (!valid) fail(`${path} violates the ${profile} storage policy`);
}

function validateNativeTensorContract(
  transformation: AceTensorTransformation,
  source: string,
  logicalShape: readonly number[],
  phase: AceTensorPhase,
  path: string,
): void {
  const ditGemmTransformation =
    transformation === ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION ||
    transformation === ACE_DIT_GEMM_FP16_TRANSFORMATION;
  const ditDenseFp16Transformation =
    transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION ||
    transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION;
  const ditGemmSource = DIT_GEMM_SOURCE.test(source);
  const ditDenseFp16Source = DIT_REPEATED_DENSE_SOURCE.test(source);
  const conv1dTransformation =
    transformation === ACE_VAE_CONV1D_TRANSFORMATION ||
    transformation === ACE_VAE_CONV1D_FP16_TRANSFORMATION ||
    transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION ||
    transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION;
  const conv1dSource = VAE_CONV1D_SOURCE.test(source);
  const convTransposeTransformation =
    transformation === ACE_VAE_CONV_TRANSPOSE1D_TRANSFORMATION ||
    transformation === ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION ||
    transformation ===
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION ||
    transformation === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION;
  const convTransposeSource = VAE_CONV_TRANSPOSE1D_SOURCE.test(source);
  const channelVectorTransformation =
    transformation === ACE_VAE_CHANNEL_VECTOR_TRANSFORMATION ||
    transformation === ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION;
  const channelVectorSource = VAE_CHANNEL_VECTOR_SOURCE.test(source);
  const biasTransformation =
    transformation === "bf16-to-fp32" ||
    transformation === ACE_VAE_BIAS_FP16_TRANSFORMATION;
  const biasSource = VAE_BIAS_SOURCE.test(source);
  if (ditGemmTransformation || ditDenseFp16Transformation || ditGemmSource) {
    if (
      (!ditGemmTransformation && !ditDenseFp16Transformation) ||
      !ditGemmSource ||
      (ditDenseFp16Transformation && !ditDenseFp16Source) ||
      phase !== "dit" ||
      logicalShape.length !== 2 ||
      logicalShape[0]! %
        (transformation === ACE_DIT_DENSE_FP16_TRANSFORMATION ? 256 : 128) !==
        0 ||
      logicalShape[1]! %
        (transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION ? 4 : 32) !==
        0
    ) {
      fail(`${path} violates the DiT GEMM tile-major contract`);
    }
  } else if (conv1dTransformation || conv1dSource) {
    if (
      !conv1dTransformation ||
      !conv1dSource ||
      phase !== "vae" ||
      logicalShape.length !== 3
    ) {
      fail(`${path} violates the Conv1d native-layout contract`);
    }
  } else if (convTransposeTransformation || convTransposeSource) {
    if (
      !convTransposeTransformation ||
      !convTransposeSource ||
      phase !== "vae" ||
      logicalShape.length !== 3
    ) {
      fail(`${path} violates the ConvTranspose1d native-layout contract`);
    }
  } else if (channelVectorTransformation || channelVectorSource) {
    if (
      !channelVectorTransformation ||
      !channelVectorSource ||
      phase !== "vae" ||
      logicalShape.length !== 1
    ) {
      fail(`${path} violates the Snake channel-vector contract`);
    }
  } else if (biasTransformation || biasSource) {
    if (
      !biasTransformation ||
      !biasSource ||
      phase !== "vae" ||
      logicalShape.length !== 1
    ) {
      fail(`${path} violates the VAE bias contract`);
    }
  }
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireExactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  path: string,
): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (!arraysEqual(actual, expected)) fail(`${path} has unknown or missing fields`);
}

function requireNonEmptyString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) fail(`${path} must be nonempty`);
  return value;
}

function requireNonNegativeInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    fail(`${path} must be a non-negative safe integer`);
  }
  return value as number;
}

function requirePositiveInteger(value: unknown, path: string): number {
  const result = requireNonNegativeInteger(value, path);
  if (result === 0) fail(`${path} must be positive`);
  return result;
}

function requireSha256(value: unknown, path: string): string {
  return requireShaLike(value, 64, path);
}

function requireShaLike(value: unknown, length: number, path: string): string {
  if (
    typeof value !== "string" ||
    value.length !== length ||
    !/^[0-9a-f]+$/.test(value)
  ) {
    fail(`${path} must be a lowercase ${length}-character hexadecimal identity`);
  }
  return value;
}

function requireExactIdentity(
  value: unknown,
  expected: string,
  path: string,
): string {
  const result = requireShaLike(value, expected.length, path);
  if (result !== expected) fail(`${path} is not the pinned identity`);
  return result;
}

function requireSafeRelativePath(value: unknown, path: string): string {
  const result = requireNonEmptyString(value, path);
  if (
    result.startsWith("/") ||
    result.includes("\\") ||
    !/^[A-Za-z0-9._/-]+$/.test(result) ||
    result.split("/").some((part) => part.length === 0 || part === "." || part === "..")
  ) {
    fail(`${path} is not a safe canonical relative path`);
  }
  return result;
}

function requireShape(value: unknown, path: string): number[] {
  if (!Array.isArray(value) || value.length === 0) fail(`${path} must be a nonempty shape`);
  return value.map((dimension, index) =>
    requireNonNegativeInteger(dimension, `${path}[${index}]`),
  );
}

function checkedProduct(values: readonly number[], path: string): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) fail(`${path} product is unsafe`);
  }
  return product;
}

function arraysEqual(
  left: readonly unknown[],
  right: readonly unknown[],
): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function fail(message: string): never {
  throw new AcePackageManifestError(`Invalid ACE model manifest: ${message}`);
}
