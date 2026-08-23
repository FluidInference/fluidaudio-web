import type {
  AceGpuLogicalTensor,
  AceGpuTensorPhase,
} from "../model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  ACE_VAE_BIAS_FP16_TRANSFORMATION,
  ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
  ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV1D_FP16_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
  ACE_VAE_K1_FP16_TILE_LAYOUT,
  ACE_VAE_K1_FP16_TILE_TRANSFORMATION,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  type AcePackageManifest,
  type AcePackageTensorRecord,
  type AceTensorLayout,
  type AceTensorTransformation,
} from "../model/manifest.js";
import type {
  AceVaeDecoderGraphPlan,
  AceVaeDecoderOperation,
} from "./vae-decoder.js";
import {
  requireAceOpt0028Fp16VaePackageIdentity,
  requireAceOpt0054Fp16VaePackageIdentity,
  requireAceOpt0011VaeDecoderGeometry,
  type AceVaeAuthenticatedPackageIdentity,
} from "./vae-fp16-profile.js";

export const ACE_OPT_0011_VAE_FP16_WEIGHT_FILES = Object.freeze([
  "weights/vae/layer-00-00.bin",
  "weights/vae/layer-00-01.bin",
  "weights/vae/layer-01-00.bin",
  "weights/vae/layer-02-00.bin",
  "weights/vae/layer-03-00.bin",
  "weights/vae/layer-04-00.bin",
  "weights/vae/shared-00.bin",
] as const);

const REVISION7_ROW_REUSE_TENSORS: ReadonlySet<string> = new Set(
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(({ tensor }) => tensor),
);
const REVISION7_TRANSPOSE_K4_BY_TENSOR: ReadonlyMap<
  string,
  "channel" | "row"
> = new Map(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(
  ({ tensor, reuseAxis }) => [tensor, reuseAxis] as const,
));

export interface AceOpt0011VaeResolvedTensor {
  readonly logicalTensor: string;
  readonly logicalShape: readonly number[];
  readonly physicalTensor: string;
  readonly record: AcePackageTensorRecord;
  readonly binding: GPUBufferBinding;
}

interface AceOpt0011VaeOperationBindingsBase {
  readonly operationIndex: number;
  readonly label: string;
}

export interface AceOpt0011VaeConv1dOperationBindings
  extends AceOpt0011VaeOperationBindingsBase {
  readonly kind: "conv1d";
  readonly weight: AceOpt0011VaeResolvedTensor;
  readonly bias?: AceOpt0011VaeResolvedTensor;
}

export interface AceOpt0011VaeConvTranspose1dOperationBindings
  extends AceOpt0011VaeOperationBindingsBase {
  readonly kind: "conv-transpose1d";
  readonly weight: AceOpt0011VaeResolvedTensor;
  readonly bias: AceOpt0011VaeResolvedTensor;
}

export interface AceOpt0011VaeSnakeOperationBindings
  extends AceOpt0011VaeOperationBindingsBase {
  readonly kind: "snake";
  readonly alpha: AceOpt0011VaeResolvedTensor;
  readonly beta: AceOpt0011VaeResolvedTensor;
}

export interface AceOpt0011VaeAddOperationBindings
  extends AceOpt0011VaeOperationBindingsBase {
  readonly kind: "add";
}

export type AceOpt0011VaeOperationBindings =
  | AceOpt0011VaeConv1dOperationBindings
  | AceOpt0011VaeConvTranspose1dOperationBindings
  | AceOpt0011VaeSnakeOperationBindings
  | AceOpt0011VaeAddOperationBindings;

export interface AceOpt0011VaePackageBindings {
  readonly manifestSha256: string;
  readonly manifestByteLength: number;
  readonly residentWeightBytes: typeof ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES;
  readonly weightFiles: readonly string[];
  readonly tensors: Readonly<Record<string, AceOpt0011VaeResolvedTensor>>;
  readonly operations: readonly AceOpt0011VaeOperationBindings[];
}

export class AceOpt0011VaePackageContractError extends Error {
  readonly code = "INVALID_OPT_0011_VAE_PACKAGE_BINDINGS";

  constructor(message: string) {
    super(message);
    this.name = "AceOpt0011VaePackageContractError";
  }
}

/**
 * Resolve the exact revision-6 experimental VAE package into all 88 graph
 * operations while retaining each authenticated tensor record. No generic
 * package validator is weakened and no alternate dtype or layout is accepted.
 */
export function resolveAceOpt0011Fp16VaePackageBindings(
  plan: AceVaeDecoderGraphPlan,
  loaded: AceVaeAuthenticatedPackageIdentity,
  phase: AceGpuTensorPhase,
): AceOpt0011VaePackageBindings {
  return resolveFp16VaePackageBindings(plan, loaded, phase, 6);
}

/**
 * Resolve only the exact authenticated revision-7 OPT-0054 mixed-layout
 * package. This is a distinct trust root, never a fallback from revision 6.
 */
export function resolveAceOpt0054Fp16VaePackageBindings(
  plan: AceVaeDecoderGraphPlan,
  loaded: AceVaeAuthenticatedPackageIdentity,
  phase: AceGpuTensorPhase,
): AceOpt0011VaePackageBindings {
  return resolveFp16VaePackageBindings(plan, loaded, phase, 7);
}

function resolveFp16VaePackageBindings(
  plan: AceVaeDecoderGraphPlan,
  loaded: AceVaeAuthenticatedPackageIdentity,
  phase: AceGpuTensorPhase,
  revision: 6 | 7,
): AceOpt0011VaePackageBindings {
  requireAceOpt0011VaeDecoderGeometry(plan);
  if (revision === 7) {
    requireAceOpt0054Fp16VaePackageIdentity(loaded);
  } else {
    requireAceOpt0028Fp16VaePackageIdentity(loaded);
  }
  const manifest = loaded.manifest;
  requirePhaseIdentity(phase, manifest);
  requireExperimentalVaeInventory(manifest, phase.residentBytes, revision);

  const requiredNames = new Set(plan.requiredTensorNames);
  const manifestNames = Object.values(manifest.tensors)
    .filter((record) => record.phase === "vae")
    .map((record) => record.logicalTensor);
  if (
    manifestNames.length !== requiredNames.size ||
    manifestNames.some((name) => !requiredNames.has(name))
  ) {
    throw contractError(
      "the experimental VAE tensor inventory does not equal the decoder graph",
    );
  }

  const tensors: Record<string, AceOpt0011VaeResolvedTensor> = {};
  const resolve = (
    logicalTensor: string,
    expectedShape: readonly number[],
    contract: AceOpt0011TensorStorageContract,
  ): AceOpt0011VaeResolvedTensor => {
    const existing = tensors[logicalTensor];
    if (existing !== undefined) return existing;
    const logical = phase.logicalTensor(logicalTensor);
    const resolved = requireLogicalTensor(
      logical,
      logicalTensor,
      expectedShape,
      contract,
      manifest,
    );
    tensors[logicalTensor] = resolved;
    return resolved;
  };

  const operations = plan.operations.map((operation, operationIndex) =>
    resolveOperationBindings(operation, operationIndex, resolve, revision)
  );
  if (Object.keys(tensors).length !== plan.requiredTensorNames.length) {
    throw contractError(
      `resolved ${Object.keys(tensors).length} tensors, expected ${plan.requiredTensorNames.length}`,
    );
  }
  return Object.freeze({
    manifestSha256: loaded.manifestSha256,
    manifestByteLength: loaded.manifestByteLength,
    residentWeightBytes: ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    weightFiles: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
    tensors: Object.freeze(tensors),
    operations: Object.freeze(operations),
  });
}

interface AceOpt0011TensorStorageContract {
  readonly layout: AceTensorLayout;
  readonly transformation: AceTensorTransformation;
  readonly source: string;
}

type ResolveTensor = (
  logicalTensor: string,
  expectedShape: readonly number[],
  contract: AceOpt0011TensorStorageContract,
) => AceOpt0011VaeResolvedTensor;

function resolveOperationBindings(
  operation: AceVaeDecoderOperation,
  operationIndex: number,
  resolve: ResolveTensor,
  revision: 6 | 7,
): AceOpt0011VaeOperationBindings {
  const base = { operationIndex, label: operation.label };
  switch (operation.kind) {
    case "conv1d": {
      const weight = resolve(
        operation.weight,
        [
          operation.shape.outputChannels,
          operation.shape.kernelSize,
          operation.shape.inputChannels,
        ],
        storageContract(
          operation.weight,
          operation.shape.kernelSize === 1
            ? "conv1d-k1-weight"
            : "conv1d-weight",
          revision,
        ),
      );
      const bias = operation.bias === undefined
        ? undefined
        : resolve(
            operation.bias,
            [operation.shape.outputChannels],
            storageContract(operation.bias, "bias", revision),
          );
      return Object.freeze({
        ...base,
        kind: "conv1d",
        weight,
        ...(bias === undefined ? {} : { bias }),
      });
    }
    case "conv-transpose1d":
      return Object.freeze({
        ...base,
        kind: "conv-transpose1d",
        weight: resolve(
          operation.weight,
          [
            operation.shape.outputChannels,
            operation.shape.kernelSize,
            operation.shape.inputChannels,
          ],
          storageContract(
            operation.weight,
            "conv-transpose1d-weight",
            revision,
          ),
        ),
        bias: resolve(
          operation.bias,
          [operation.shape.outputChannels],
          storageContract(operation.bias, "bias", revision),
        ),
      });
    case "snake":
      return Object.freeze({
        ...base,
        kind: "snake",
        alpha: resolve(
          operation.alpha,
          [operation.shape.channels],
          storageContract(operation.alpha, "channel-vector", revision),
        ),
        beta: resolve(
          operation.beta,
          [operation.shape.channels],
          storageContract(operation.beta, "channel-vector", revision),
        ),
      });
    case "add":
      return Object.freeze({ ...base, kind: "add" });
  }
}

function storageContract(
  logicalTensor: string,
  role:
    | "conv1d-weight"
    | "conv1d-k1-weight"
    | "conv-transpose1d-weight"
    | "channel-vector"
    | "bias",
  revision: 6 | 7,
): AceOpt0011TensorStorageContract {
  const sourceBase = logicalTensor.slice("vae.".length);
  switch (role) {
    case "conv1d-weight": {
      const rowReuse = revision === 7 &&
        REVISION7_ROW_REUSE_TENSORS.has(logicalTensor);
      return Object.freeze({
        layout: rowReuse
          ? ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT
          : ACE_VAE_CONV1D_FP16_LAYOUT,
        transformation: rowReuse
          ? ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION
          : ACE_VAE_CONV1D_FP16_TRANSFORMATION,
        source: `vae-weights:${sourceBase.replace(/\.weight$/, ".weight_v")}`,
      });
    }
    case "conv1d-k1-weight":
      return Object.freeze({
        layout: ACE_VAE_K1_FP16_TILE_LAYOUT,
        transformation: ACE_VAE_K1_FP16_TILE_TRANSFORMATION,
        source: `vae-weights:${sourceBase.replace(/\.weight$/, ".weight_v")}`,
      });
    case "conv-transpose1d-weight": {
      const k4 = revision === 7 && logicalTensor !==
        ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR;
      if (k4 && !REVISION7_TRANSPOSE_K4_BY_TENSOR.has(logicalTensor)) {
        throw contractError(
          `revision-7 transpose tensor ${logicalTensor} has no declared owner`,
        );
      }
      return Object.freeze({
        layout: k4
          ? ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT
          : ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
        transformation: k4
          ? ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION
          : ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
        source: `vae-weights:${sourceBase.replace(/\.weight$/, ".weight_v")}`,
      });
    }
    case "channel-vector":
      return Object.freeze({
        layout: ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
        transformation: ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
        source: `vae-weights:${sourceBase}`,
      });
    case "bias":
      return Object.freeze({
        layout: "source-row-major",
        transformation: ACE_VAE_BIAS_FP16_TRANSFORMATION,
        source: `vae-weights:${sourceBase}`,
      });
  }
}

function requireLogicalTensor(
  logical: AceGpuLogicalTensor,
  expectedName: string,
  expectedShape: readonly number[],
  contract: AceOpt0011TensorStorageContract,
  manifest: AcePackageManifest,
): AceOpt0011VaeResolvedTensor {
  if (
    logical.logicalTensor !== expectedName ||
    !sameShape(logical.logicalShape, expectedShape) ||
    logical.parts.length !== 1
  ) {
    throw contractError(
      `tensor ${expectedName} is not one complete logical tensor with shape [${expectedShape.join(",")}]`,
    );
  }
  const part = logical.parts[0]!;
  const record = part.tensor;
  const manifestRecord = manifest.tensors[expectedName];
  const elements = checkedProduct(expectedShape, expectedName);
  const expectedStorageShape = storageShapeFor(
    expectedShape,
    contract.transformation,
    expectedName,
  );
  if (
    manifestRecord === undefined ||
    part.tensorName !== expectedName ||
    record !== manifestRecord ||
    record.logicalTensor !== expectedName ||
    record.dtype !== "float16" ||
    record.phase !== "vae" ||
    record.lifetime !== "vae" ||
    record.layout !== contract.layout ||
    record.transformation !== contract.transformation ||
    record.source !== contract.source ||
    !sameShape(record.logicalShape, expectedShape) ||
    !sameShape(record.storageShape, expectedStorageShape) ||
    record.byteLength !== elements * 2 ||
    record.partAxis !== 0 ||
    record.partStart !== 0 ||
    record.partEnd !== expectedShape[0]
  ) {
    throw contractError(`tensor ${expectedName} violates its FP16 package contract`);
  }
  const offset = part.binding.offset;
  const size = part.binding.size;
  if (
    offset !== record.byteOffset ||
    size !== record.byteLength ||
    offset + size > part.binding.buffer.size
  ) {
    throw contractError(
      `tensor ${expectedName} binding does not preserve its authenticated byte span`,
    );
  }
  return Object.freeze({
    logicalTensor: expectedName,
    logicalShape: Object.freeze([...expectedShape]),
    physicalTensor: part.tensorName,
    record,
    binding: part.binding,
  });
}

function requirePhaseIdentity(
  phase: AceGpuTensorPhase,
  manifest: AcePackageManifest,
): void {
  if (
    phase.packageManifest !== manifest ||
    phase.phases.length !== 1 ||
    phase.phases[0] !== "vae"
  ) {
    throw contractError(
      "the resident phase is not the exclusive VAE phase of this authenticated manifest",
    );
  }
}

function requireExperimentalVaeInventory(
  manifest: AcePackageManifest,
  residentBytes: number,
  revision: 6 | 7,
): void {
  const entries = Object.entries(manifest.tensors)
    .filter(([, record]) => record.phase === "vae");
  const logicalNames = new Set(entries.map(([, record]) => record.logicalTensor));
  const shards = new Set(entries.map(([, record]) => record.shard));
  const payloadBytes = entries.reduce(
    (total, [, record]) => total + record.byteLength,
    0,
  );
  if (
    entries.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    logicalNames.size !== ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT ||
    payloadBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    shards.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !shards.has(name))
  ) {
    throw contractError(
      "the experimental VAE inventory is not 145 unsharded tensors/168791552 bytes/seven files",
    );
  }
  const files = new Map(manifest.files.map((file) => [file.name, file]));
  let fileBytes = 0;
  for (const shard of ACE_OPT_0011_VAE_FP16_WEIGHT_FILES) {
    const file = files.get(shard);
    if (file === undefined || file.kind !== "weights") {
      throw contractError(`the experimental VAE weight file ${shard} is absent`);
    }
    fileBytes += file.byteLength;
  }
  if (fileBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
    throw contractError(
      "the seven experimental VAE weight files have the wrong resident byte total",
    );
  }
  for (const [physicalName, record] of entries) {
    const expectedStorageShape = storageShapeFor(
      record.logicalShape,
      record.transformation,
      physicalName,
    );
    if (
      physicalName !== record.logicalTensor ||
      record.dtype !== "float16" ||
      record.partAxis !== 0 ||
      record.partStart !== 0 ||
      record.partEnd !== record.logicalShape[0] ||
      !sameShape(record.storageShape, expectedStorageShape)
    ) {
      throw contractError(
        `experimental VAE tensor ${physicalName} is not one complete revision-${revision} FP16 record`,
      );
    }
  }
}

function storageShapeFor(
  logicalShape: readonly number[],
  transformation: AceTensorTransformation,
  name: string,
): readonly number[] {
  if (transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION) {
    const [outputChannels, kernel, inputChannels] = logicalShape;
    if (
      logicalShape.length !== 3 ||
      kernel !== 1 ||
      outputChannels !== inputChannels ||
      outputChannels! % 128 !== 0 ||
      inputChannels! % 32 !== 0
    ) {
      throw contractError(`tensor ${name} has an invalid packed K1 shape`);
    }
    return Object.freeze([
      outputChannels! / 128,
      inputChannels! / 32,
      32,
      128,
    ]);
  }
  if (
    transformation ===
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
  ) {
    const [outputChannels, kernel, inputChannels] = logicalShape;
    if (
      logicalShape.length !== 3 ||
      kernel! % 2 !== 0 ||
      outputChannels === undefined ||
      inputChannels === undefined
    ) {
      throw contractError(
        `tensor ${name} has an invalid packed ConvTranspose1d shape`,
      );
    }
    return Object.freeze([kernel! / 2, 2, inputChannels, outputChannels]);
  }
  if (transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION) {
    const [outputChannels, kernel, inputChannels] = logicalShape;
    if (
      logicalShape.length !== 3 || kernel !== 7 ||
      outputChannels === undefined || outputChannels % 64 !== 0 ||
      inputChannels === undefined || inputChannels % 4 !== 0 ||
      outputChannels !== inputChannels
    ) {
      throw contractError(`tensor ${name} has an invalid row-reuse K7 shape`);
    }
    return Object.freeze([
      7,
      inputChannels / 4,
      outputChannels / 64,
      32,
      2,
      4,
    ]);
  }
  if (transformation === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION) {
    const [outputChannels, kernel, inputChannels] = logicalShape;
    const reuseAxis = REVISION7_TRANSPOSE_K4_BY_TENSOR.get(name);
    if (
      logicalShape.length !== 3 || outputChannels === undefined ||
      kernel === undefined || kernel % 2 !== 0 ||
      inputChannels === undefined || inputChannels % 4 !== 0 ||
      reuseAxis === undefined
    ) {
      throw contractError(`tensor ${name} has an invalid transpose K4 shape`);
    }
    const outputsPerLane = reuseAxis === "channel" ? 8 : 4;
    const outputTile = outputsPerLane * 32;
    if (outputChannels % outputTile !== 0) {
      throw contractError(`tensor ${name} has an invalid transpose K4 tile`);
    }
    return Object.freeze([
      kernel / 2,
      2,
      inputChannels / 4,
      outputChannels / outputTile,
      32,
      outputsPerLane,
      4,
    ]);
  }
  return logicalShape;
}

function sameShape(
  actual: readonly number[],
  expected: readonly number[],
): boolean {
  return actual.length === expected.length &&
    actual.every((value, index) => value === expected[index]);
}

function checkedProduct(shape: readonly number[], name: string): number {
  let product = 1;
  for (const extent of shape) {
    product *= extent;
    if (!Number.isSafeInteger(product) || extent <= 0) {
      throw contractError(`tensor ${name} has an unsafe shape`);
    }
  }
  return product;
}

function contractError(message: string): AceOpt0011VaePackageContractError {
  return new AceOpt0011VaePackageContractError(`OPT-0011 ${message}`);
}
