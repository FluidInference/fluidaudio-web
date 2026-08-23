import type { AceGpuLogicalTensor } from "../model/gpu-tensors.js";
import type {
  AcePackageManifest,
  AcePackageTensorRecord,
  AceTensorPhase,
} from "../model/manifest.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  ACE_AUDIO_LATENT_CHANNELS,
  ACE_CONDITION_HEAD_DIMENSION,
  ACE_CONDITION_HIDDEN_SIZE,
  ACE_CONDITION_INTERMEDIATE_SIZE,
  ACE_CONDITION_KEY_VALUE_HEADS,
  ACE_CONDITION_QUERY_HEADS,
  ACE_DETOKENIZER_LAYER_COUNT,
  ACE_DIRECT_CONDITIONER_TENSOR_NAMES,
  ACE_LYRIC_ENCODER_LAYER_COUNT,
  ACE_SEMANTIC_POOL_WIDTH,
  ACE_SEMANTIC_TENSOR_NAMES,
  ACE_TEXT_EMBEDDING_WIDTH,
  ACE_TIMBRE_ENCODER_LAYER_COUNT,
  aceEncoderLayerTensorNames,
  type AceDirectConditionerBindings,
  type AceEncoderLayerTensorNames,
  type AceSemanticDecodeBindings,
} from "./semantic-conditioner.js";

export const ACE_DETOKENIZER_LAYER_PREFIX = "ace.detokenizer";
export const ACE_LYRIC_ENCODER_LAYER_PREFIX = "ace.encoder.lyric_encoder";
export const ACE_TIMBRE_ENCODER_LAYER_PREFIX = "ace.encoder.timbre_encoder";
export const ACE_UNUSED_TIMBRE_SPECIAL_TOKEN =
  "ace.encoder.timbre_encoder.special_token";

export interface AceConditionerTensorResolver {
  logicalTensor(logicalTensor: string): AceGpuLogicalTensor;
  binding(logicalTensor: string): GPUBufferBinding;
}

export type AceSemanticPackageWeights = AceSemanticDecodeBindings["weights"];
export type AceDirectConditionerPackageWeights =
  AceDirectConditionerBindings["weights"];

export interface AceDirectConditionerPackageResolution {
  readonly weights: AceDirectConditionerPackageWeights;
  readonly silenceSource: GPUBufferBinding;
  /**
   * Pinned Turbo constructs this parameter but comments out its concatenation
   * in `AceStepTimbreEncoder.forward`. Keeping the authenticated binding here
   * makes the one intentionally output-dead conditioner tensor explicit.
   */
  readonly auditedUnusedTimbreSpecialToken: GPUBufferBinding;
}

export const ACE_SEMANTIC_LOGICAL_TENSOR_NAMES: readonly string[] =
  Object.freeze([
    ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOut,
    ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOutBias,
    ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjection,
    ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjectionBias,
    ACE_SEMANTIC_TENSOR_NAMES.specialTokens,
    ...encoderLayerNames(
      ACE_DETOKENIZER_LAYER_PREFIX,
      ACE_DETOKENIZER_LAYER_COUNT,
    ).flatMap((layer) => Object.values(layer)),
    ACE_SEMANTIC_TENSOR_NAMES.finalNorm,
    ACE_SEMANTIC_TENSOR_NAMES.acousticProjection,
    ACE_SEMANTIC_TENSOR_NAMES.acousticProjectionBias,
  ]);

export const ACE_DIRECT_CONDITIONER_LOGICAL_TENSOR_NAMES: readonly string[] =
  Object.freeze([
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.textProjection,
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjection,
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjectionBias,
    ...encoderLayerNames(
      ACE_LYRIC_ENCODER_LAYER_PREFIX,
      ACE_LYRIC_ENCODER_LAYER_COUNT,
    ).flatMap((layer) => Object.values(layer)),
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricFinalNorm,
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjection,
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjectionBias,
    ...encoderLayerNames(
      ACE_TIMBRE_ENCODER_LAYER_PREFIX,
      ACE_TIMBRE_ENCODER_LAYER_COUNT,
    ).flatMap((layer) => Object.values(layer)),
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreFinalNorm,
    ACE_UNUSED_TIMBRE_SPECIAL_TOKEN,
  ]);

const SEMANTIC_SHAPES = buildSemanticShapes();
const DIRECT_SHAPES = buildDirectShapes();

/** Exact logical shape used by the semantic/direct graph package contract. */
export function aceConditionerExpectedLogicalShape(
  logicalTensor: string,
): readonly number[] {
  if (logicalTensor === ACE_DIRECT_CONDITIONER_TENSOR_NAMES.silenceSource) {
    return Object.freeze([1, ACE_AUDIO_LATENT_CHANNELS, 15_000]);
  }
  const shape = SEMANTIC_SHAPES.get(logicalTensor) ??
    DIRECT_SHAPES.get(logicalTensor);
  if (shape === undefined) {
    throw new Error(`Unknown ACE conditioner tensor ${logicalTensor}`);
  }
  return shape;
}

/** Resolve all 30 semantic-phase tensors into the learned detokenizer graph. */
export function resolveAceSemanticPackageWeights(
  resolver: AceConditionerTensorResolver,
  modelProfile: AceModelProfileId,
): AceSemanticPackageWeights {
  const resolve = semanticResolver(resolver, modelProfile);
  return Object.freeze({
    fsqProjectOut: resolve(ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOut),
    fsqProjectOutBias: resolve(ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOutBias),
    detokenizerInputProjection: resolve(
      ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjection,
    ),
    detokenizerInputProjectionBias: resolve(
      ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjectionBias,
    ),
    specialTokens: resolve(ACE_SEMANTIC_TENSOR_NAMES.specialTokens),
    layers: Object.freeze(encoderLayerNames(
      ACE_DETOKENIZER_LAYER_PREFIX,
      ACE_DETOKENIZER_LAYER_COUNT,
    ).map((names) => resolveEncoderLayer(names, resolve))),
    finalNorm: resolve(ACE_SEMANTIC_TENSOR_NAMES.finalNorm),
    acousticProjection: resolve(ACE_SEMANTIC_TENSOR_NAMES.acousticProjection),
    acousticProjectionBias: resolve(
      ACE_SEMANTIC_TENSOR_NAMES.acousticProjectionBias,
    ),
  });
}

/** Resolve the direct-v1 conditioner and its authenticated silence constant. */
export function resolveAceDirectConditionerPackage(
  resolver: AceConditionerTensorResolver,
  modelProfile: AceModelProfileId,
): AceDirectConditionerPackageResolution {
  const resolve = directResolver(resolver, modelProfile);
  const weights: AceDirectConditionerPackageWeights = Object.freeze({
    textProjection: resolve(ACE_DIRECT_CONDITIONER_TENSOR_NAMES.textProjection),
    lyricInputProjection: resolve(
      ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjection,
    ),
    lyricInputProjectionBias: resolve(
      ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjectionBias,
    ),
    lyricLayers: Object.freeze(encoderLayerNames(
      ACE_LYRIC_ENCODER_LAYER_PREFIX,
      ACE_LYRIC_ENCODER_LAYER_COUNT,
    ).map((names) => resolveEncoderLayer(names, resolve))),
    lyricFinalNorm: resolve(ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricFinalNorm),
    timbreInputProjection: resolve(
      ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjection,
    ),
    timbreInputProjectionBias: resolve(
      ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjectionBias,
    ),
    timbreLayers: Object.freeze(encoderLayerNames(
      ACE_TIMBRE_ENCODER_LAYER_PREFIX,
      ACE_TIMBRE_ENCODER_LAYER_COUNT,
    ).map((names) => resolveEncoderLayer(names, resolve))),
    timbreFinalNorm: resolve(
      ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreFinalNorm,
    ),
  });
  return Object.freeze({
    weights,
    silenceSource: resolveSilenceSource(resolver),
    auditedUnusedTimbreSpecialToken: resolve(ACE_UNUSED_TIMBRE_SPECIAL_TOKEN),
  });
}

/** Fail closed on hidden, missing, reshaped, or retyped package tensors. */
export function validateAceConditionerPackageInventory(
  manifest: AcePackageManifest,
): void {
  const profile: AceModelProfileId = manifest.profile === "reference"
    ? "reference-bf16"
    : "raw-fp16";
  validatePhaseInventory(
    manifest,
    "semantic",
    ACE_SEMANTIC_LOGICAL_TENSOR_NAMES,
    SEMANTIC_SHAPES,
    profile,
  );
  validatePhaseInventory(
    manifest,
    "conditioner",
    ACE_DIRECT_CONDITIONER_LOGICAL_TENSOR_NAMES,
    DIRECT_SHAPES,
    profile,
  );
  requireManifestTensor(
    manifest.tensors[ACE_DIRECT_CONDITIONER_TENSOR_NAMES.silenceSource],
    ACE_DIRECT_CONDITIONER_TENSOR_NAMES.silenceSource,
    [1, ACE_AUDIO_LATENT_CHANNELS, 15_000],
    "constants",
    "float32",
    "contiguous-nct-f32",
  );
}

function semanticResolver(
  resolver: AceConditionerTensorResolver,
  modelProfile: AceModelProfileId,
): (name: string) => GPUBufferBinding {
  return (name) => resolveWeight(
    resolver,
    name,
    requireShape(SEMANTIC_SHAPES, name),
    "semantic",
    modelProfile,
  );
}

function directResolver(
  resolver: AceConditionerTensorResolver,
  modelProfile: AceModelProfileId,
): (name: string) => GPUBufferBinding {
  return (name) => resolveWeight(
    resolver,
    name,
    requireShape(DIRECT_SHAPES, name),
    "conditioner",
    modelProfile,
  );
}

function resolveSilenceSource(
  resolver: AceConditionerTensorResolver,
): GPUBufferBinding {
  const name = ACE_DIRECT_CONDITIONER_TENSOR_NAMES.silenceSource;
  const logical = resolver.logicalTensor(name);
  requireLogical(
    logical,
    [1, ACE_AUDIO_LATENT_CHANNELS, 15_000],
    "constants",
    "float32",
    "contiguous-nct-f32",
  );
  return resolver.binding(name);
}

function resolveWeight(
  resolver: AceConditionerTensorResolver,
  name: string,
  shape: readonly number[],
  phase: "semantic" | "conditioner",
  modelProfile: AceModelProfileId,
): GPUBufferBinding {
  const dtype = modelProfile === "reference-bf16"
    ? "uint32-bf16-pairs"
    : modelProfile === "raw-fp16"
      ? "float16"
      : null;
  const layout = modelProfile === "reference-bf16"
    ? "source-row-major-bf16-pairs-lsb-u32"
    : modelProfile === "raw-fp16"
      ? "source-row-major"
      : null;
  if (dtype === null || layout === null) {
    throw new TypeError(`Unknown ACE conditioner model profile ${String(modelProfile)}`);
  }
  requireLogical(resolver.logicalTensor(name), shape, phase, dtype, layout);
  return resolver.binding(name);
}

function resolveEncoderLayer(
  names: AceEncoderLayerTensorNames,
  resolve: (name: string) => GPUBufferBinding,
) {
  return Object.freeze({
    inputLayerNorm: resolve(names.inputLayerNorm),
    queryProjection: resolve(names.queryProjection),
    keyProjection: resolve(names.keyProjection),
    valueProjection: resolve(names.valueProjection),
    queryNorm: resolve(names.queryNorm),
    keyNorm: resolve(names.keyNorm),
    outputProjection: resolve(names.outputProjection),
    postAttentionLayerNorm: resolve(names.postAttentionLayerNorm),
    gateProjection: resolve(names.gateProjection),
    upProjection: resolve(names.upProjection),
    downProjection: resolve(names.downProjection),
  });
}

function encoderLayerNames(
  prefix: string,
  count: number,
): readonly AceEncoderLayerTensorNames[] {
  return Object.freeze(Array.from(
    { length: count },
    (_, layer) => aceEncoderLayerTensorNames(prefix, layer),
  ));
}

function addEncoderLayerShapes(
  map: Map<string, readonly number[]>,
  prefix: string,
  count: number,
): void {
  const hidden = ACE_CONDITION_HIDDEN_SIZE;
  const query = ACE_CONDITION_QUERY_HEADS * ACE_CONDITION_HEAD_DIMENSION;
  const keyValue = ACE_CONDITION_KEY_VALUE_HEADS * ACE_CONDITION_HEAD_DIMENSION;
  for (const layer of encoderLayerNames(prefix, count)) {
    addShape(map, layer.inputLayerNorm, [hidden]);
    addShape(map, layer.queryProjection, [query, hidden]);
    addShape(map, layer.keyProjection, [keyValue, hidden]);
    addShape(map, layer.valueProjection, [keyValue, hidden]);
    addShape(map, layer.queryNorm, [ACE_CONDITION_HEAD_DIMENSION]);
    addShape(map, layer.keyNorm, [ACE_CONDITION_HEAD_DIMENSION]);
    addShape(map, layer.outputProjection, [hidden, query]);
    addShape(map, layer.postAttentionLayerNorm, [hidden]);
    addShape(map, layer.gateProjection, [ACE_CONDITION_INTERMEDIATE_SIZE, hidden]);
    addShape(map, layer.upProjection, [ACE_CONDITION_INTERMEDIATE_SIZE, hidden]);
    addShape(map, layer.downProjection, [hidden, ACE_CONDITION_INTERMEDIATE_SIZE]);
  }
}

function buildSemanticShapes(): ReadonlyMap<string, readonly number[]> {
  const map = new Map<string, readonly number[]>();
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOut, [ACE_CONDITION_HIDDEN_SIZE, 6]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOutBias, [ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjection, [ACE_CONDITION_HIDDEN_SIZE, ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.detokenizerInputProjectionBias, [ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.specialTokens, [1, ACE_SEMANTIC_POOL_WIDTH, ACE_CONDITION_HIDDEN_SIZE]);
  addEncoderLayerShapes(map, ACE_DETOKENIZER_LAYER_PREFIX, ACE_DETOKENIZER_LAYER_COUNT);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.finalNorm, [ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.acousticProjection, [ACE_AUDIO_LATENT_CHANNELS, ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_SEMANTIC_TENSOR_NAMES.acousticProjectionBias, [ACE_AUDIO_LATENT_CHANNELS]);
  requireMapSize(map, ACE_SEMANTIC_LOGICAL_TENSOR_NAMES.length, "semantic");
  return map;
}

function buildDirectShapes(): ReadonlyMap<string, readonly number[]> {
  const map = new Map<string, readonly number[]>();
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.textProjection, [ACE_CONDITION_HIDDEN_SIZE, ACE_TEXT_EMBEDDING_WIDTH]);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjection, [ACE_CONDITION_HIDDEN_SIZE, ACE_TEXT_EMBEDDING_WIDTH]);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricInputProjectionBias, [ACE_CONDITION_HIDDEN_SIZE]);
  addEncoderLayerShapes(map, ACE_LYRIC_ENCODER_LAYER_PREFIX, ACE_LYRIC_ENCODER_LAYER_COUNT);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.lyricFinalNorm, [ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjection, [ACE_CONDITION_HIDDEN_SIZE, ACE_AUDIO_LATENT_CHANNELS]);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreInputProjectionBias, [ACE_CONDITION_HIDDEN_SIZE]);
  addEncoderLayerShapes(map, ACE_TIMBRE_ENCODER_LAYER_PREFIX, ACE_TIMBRE_ENCODER_LAYER_COUNT);
  addShape(map, ACE_DIRECT_CONDITIONER_TENSOR_NAMES.timbreFinalNorm, [ACE_CONDITION_HIDDEN_SIZE]);
  addShape(map, ACE_UNUSED_TIMBRE_SPECIAL_TOKEN, [1, 1, ACE_CONDITION_HIDDEN_SIZE]);
  requireMapSize(map, ACE_DIRECT_CONDITIONER_LOGICAL_TENSOR_NAMES.length, "direct conditioner");
  return map;
}

function addShape(
  map: Map<string, readonly number[]>,
  name: string,
  shape: readonly number[],
): void {
  if (map.has(name)) throw new Error(`Duplicate ACE conditioner tensor ${name}`);
  map.set(name, Object.freeze([...shape]));
}

function requireMapSize(
  map: ReadonlyMap<string, readonly number[]>,
  expected: number,
  label: string,
): void {
  if (map.size !== expected) {
    throw new Error(`ACE ${label} shape map has ${map.size} entries; ${expected} required`);
  }
}

function requireShape(
  map: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] {
  const shape = map.get(name);
  if (shape === undefined) throw new Error(`Unknown ACE conditioner tensor ${name}`);
  return shape;
}

function requireLogical(
  logical: AceGpuLogicalTensor,
  shape: readonly number[],
  phase: AceTensorPhase,
  dtype: string,
  layout: string,
): void {
  if (
    logical.logicalShape.length !== shape.length ||
    logical.logicalShape.some((value, index) => value !== shape[index])
  ) {
    throw new Error(
      `ACE tensor ${logical.logicalTensor} has shape [${logical.logicalShape}], ` +
        `expected [${shape}]`,
    );
  }
  if (logical.parts.length !== 1) {
    throw new Error(`ACE tensor ${logical.logicalTensor} must be unsharded`);
  }
  requireTensorRecord(
    logical.parts[0]!.tensor,
    logical.logicalTensor,
    shape,
    phase,
    dtype,
    layout,
  );
}

function validatePhaseInventory(
  manifest: AcePackageManifest,
  phase: "semantic" | "conditioner",
  expectedNames: readonly string[],
  shapes: ReadonlyMap<string, readonly number[]>,
  profile: AceModelProfileId,
): void {
  const actual = Object.entries(manifest.tensors)
    .filter(([, tensor]) => tensor.phase === phase)
    .map(([name]) => name)
    .sort();
  const expected = [...expectedNames].sort();
  if (
    actual.length !== expected.length ||
    actual.some((name, index) => name !== expected[index])
  ) {
    throw new Error(`ACE ${phase} manifest inventory differs from the graph`);
  }
  const dtype = profile === "reference-bf16" ? "uint32-bf16-pairs" : "float16";
  const layout = profile === "reference-bf16"
    ? "source-row-major-bf16-pairs-lsb-u32"
    : "source-row-major";
  for (const name of expectedNames) {
    requireManifestTensor(
      manifest.tensors[name],
      name,
      requireShape(shapes, name),
      phase,
      dtype,
      layout,
    );
  }
}

function requireManifestTensor(
  tensor: AcePackageTensorRecord | undefined,
  name: string,
  shape: readonly number[],
  phase: AceTensorPhase,
  dtype: string,
  layout: string,
): void {
  if (tensor === undefined) throw new Error(`ACE package is missing ${name}`);
  requireTensorRecord(tensor, name, shape, phase, dtype, layout);
}

function requireTensorRecord(
  tensor: AcePackageTensorRecord,
  name: string,
  shape: readonly number[],
  phase: AceTensorPhase,
  dtype: string,
  layout: string,
): void {
  if (
    tensor.logicalTensor !== name ||
    tensor.logicalShape.length !== shape.length ||
    tensor.logicalShape.some((value, index) => value !== shape[index]) ||
    tensor.phase !== phase ||
    tensor.dtype !== dtype ||
    tensor.layout !== layout ||
    tensor.partAxis !== 0 ||
    tensor.partStart !== 0 ||
    tensor.partEnd !== shape[0]
  ) {
    throw new Error(`ACE package tensor ${name} violates its graph contract`);
  }
}
