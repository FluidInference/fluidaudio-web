/** Browser package metadata emitted by model/convert.py. */
export interface DiCoSeTensorManifest {
  readonly name: string;
  readonly shape: readonly number[];
  readonly offset: number;
  readonly byteLength: number;
  /** All model tensors are native IEEE 754 binary16 values. */
  readonly dtype: "f16";
  /** Linear weights keep logical [in_features, out_features] shape. */
  readonly layout:
    | "row-major"
    | "linear-in-out"
    | "linear-tile-n128-k32"
    | "linear-tile-n256-k32"
    | "conv-oihw";
}

export interface DiCoSeModelConfig {
  readonly sampleRate: 44100;
  readonly nFft: 2048;
  readonly hopLength: 441;
  readonly winLength: 2048;
  readonly stereo: true;
  readonly stems: readonly ["drums", "bass", "other", "vocals"];
  readonly dim: 384;
  readonly depth: 8;
  readonly heads: 8;
  readonly dimHead: 64;
  readonly freqsPerBands: readonly number[];
}

export interface DiCoSeModelManifest {
  readonly schema: "dicose-wgsl-package-v1";
  readonly source: Readonly<{
    readonly upstreamRevision: string;
    readonly deterministicCheckpointSha256: string;
    readonly cdCheckpointSha256: string;
  }>;
  readonly config: DiCoSeModelConfig;
  readonly weights: Readonly<{
    readonly file: string;
    readonly byteLength: number;
    readonly sha256: string;
  }>;
  readonly tensors: readonly DiCoSeTensorManifest[];
}

export function parseModelManifest(value: unknown): DiCoSeModelManifest {
  if (value === null || typeof value !== "object") {
    throw new TypeError("DiCoSe model manifest must be an object");
  }
  const manifest = value as Partial<DiCoSeModelManifest>;
  if (manifest.schema !== "dicose-wgsl-package-v1") {
    throw new TypeError("Unsupported DiCoSe model package schema");
  }
  if (manifest.config === undefined || manifest.weights === undefined) {
    throw new TypeError("DiCoSe model manifest omits config or weights");
  }
  if (
    typeof manifest.weights.file !== "string" || manifest.weights.file.length === 0 ||
    !Number.isSafeInteger(manifest.weights.byteLength) || manifest.weights.byteLength <= 0 ||
    !/^[0-9a-f]{64}$/.test(manifest.weights.sha256)
  ) {
    throw new TypeError("DiCoSe model manifest has invalid weight metadata");
  }
  if (!Array.isArray(manifest.tensors) || manifest.tensors.length === 0) {
    throw new TypeError("DiCoSe model manifest has no tensors");
  }
  const config = manifest.config;
  if (
    config.dim !== 384 || config.depth !== 8 || config.nFft !== 2048 ||
    config.hopLength !== 441 || !Array.isArray(config.freqsPerBands)
  ) {
    throw new TypeError("DiCoSe model config is not the published BS-RoFormer profile");
  }
  const seen = new Set<string>();
  for (const tensor of manifest.tensors) validateTensor(tensor, seen);
  return manifest as DiCoSeModelManifest;
}

function validateTensor(
  tensor: unknown,
  seen: Set<string>,
): asserts tensor is DiCoSeTensorManifest {
  if (tensor === null || typeof tensor !== "object") {
    throw new TypeError("DiCoSe tensor metadata must be an object");
  }
  const value = tensor as Partial<DiCoSeTensorManifest>;
  if (typeof value.name !== "string" || value.name.length === 0) {
    throw new TypeError("DiCoSe tensor is missing a name");
  }
  if (seen.has(value.name)) throw new TypeError(`Duplicate tensor ${value.name}`);
  seen.add(value.name);
  if (
    value.dtype !== "f16" ||
    (value.layout !== "row-major" && value.layout !== "linear-in-out" &&
      value.layout !== "linear-tile-n128-k32" &&
      value.layout !== "linear-tile-n256-k32" &&
      value.layout !== "conv-oihw") ||
    !Array.isArray(value.shape) || value.shape.length === 0 ||
    value.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    typeof value.offset !== "number" || !Number.isSafeInteger(value.offset) || value.offset < 0 || value.offset % 256 !== 0 ||
    typeof value.byteLength !== "number" || !Number.isSafeInteger(value.byteLength) || value.byteLength <= 0 || value.byteLength % 2 !== 0
  ) {
    throw new TypeError(`Invalid DiCoSe tensor metadata for ${value.name}`);
  }
  const elements = value.shape.reduce((product, dimension) => product * dimension, 1);
  if (elements * 2 !== value.byteLength) {
    throw new TypeError(`Byte length mismatch for DiCoSe tensor ${value.name}`);
  }
}
