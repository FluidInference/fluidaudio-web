/** Canonical browser-package tensor names consumed by the ACE Turbo DiT. */

import type { AceGpuLogicalTensor } from "../model/gpu-tensors.js";
import {
  ACE_DIT_DENSE_K4_FP16_LAYOUT,
  ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
  ACE_DIT_DENSE_FP16_TILE_LAYOUT,
  ACE_DIT_DENSE_FP16_TRANSFORMATION,
  ACE_DIT_GEMM_FP16_TRANSFORMATION,
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_TILE_LAYOUT,
  type AceTensorDtype,
  type AceTensorLayout,
  type AceTensorTransformation,
} from "../model/manifest.js";
import {
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  type AceDitDenseRuntimeProfile,
} from "./dit-fp16-package.js";
import type { AceModelProfileId } from "./capabilities.js";
import type {
  AceDitCrossCacheWeights,
  AceDitLayerWeights,
  AceDitOutputWeights,
  AceDitTimestepBranchWeights,
  AceDitTimestepWeights,
} from "./ace-dit.js";
import { ACE_TURBO_DIT_CONFIG } from "./ace-dit.js";

export const ACE_DIT_LAYER_COUNT = 24;

export interface AceDitProjectionTensorNames {
  readonly weight: string;
  readonly bias: string;
}

export interface AceDitTimestepBranchTensorNames {
  readonly linear1Weight: string;
  readonly linear1Bias: string;
  readonly linear2Weight: string;
  readonly linear2Bias: string;
  readonly projectionWeight: string;
  readonly projectionBias: string;
}

export interface AceDitOutputTensorNames {
  readonly norm: string;
  readonly scaleShiftTable: string;
  readonly projection: string;
  readonly bias: string;
}

export interface AceDitSharedTensorNames {
  readonly conditionProjection: AceDitProjectionTensorNames;
  readonly inputProjection: AceDitProjectionTensorNames;
  readonly timestep: AceDitTimestepBranchTensorNames;
  readonly relativeTimestep: AceDitTimestepBranchTensorNames;
  readonly output: AceDitOutputTensorNames;
}

export interface AceDitLayerTensorNames {
  readonly scaleShiftTable: string;
  readonly selfAttentionNorm: string;
  readonly selfQueryProjection: string;
  readonly selfKeyProjection: string;
  readonly selfValueProjection: string;
  readonly selfQueryNorm: string;
  readonly selfKeyNorm: string;
  readonly selfOutputProjection: string;
  readonly crossAttentionNorm: string;
  readonly crossQueryProjection: string;
  readonly crossKeyProjection: string;
  readonly crossValueProjection: string;
  readonly crossQueryNorm: string;
  readonly crossKeyNorm: string;
  readonly crossOutputProjection: string;
  readonly mlpNorm: string;
  readonly gateProjection: string;
  readonly upProjection: string;
  readonly downProjection: string;
}

export interface AceDitCanonicalTensorNameMap {
  readonly shared: AceDitSharedTensorNames;
  readonly layers: readonly AceDitLayerTensorNames[];
}

export interface AceDitTensorResolver {
  logicalTensor(logicalTensor: string): AceGpuLogicalTensor;
  binding(logicalTensor: string): GPUBufferBinding;
}

export interface AceDitResolvedPackageWeights {
  readonly conditionProjection: Readonly<{
    readonly weight: GPUBufferBinding;
    readonly bias: GPUBufferBinding;
  }>;
  readonly inputProjection: Readonly<{
    readonly weight: GPUBufferBinding;
    readonly bias: GPUBufferBinding;
  }>;
  readonly timestep: AceDitTimestepWeights;
  readonly output: AceDitOutputWeights;
  readonly layers: readonly AceDitLayerWeights[];
  readonly crossCaches: readonly AceDitCrossCacheWeights[];
}

export interface AceDitRepeatedDenseLayerWeights {
  readonly selfQueryProjection: GPUBufferBinding;
  readonly selfKeyProjection: GPUBufferBinding;
  readonly selfValueProjection: GPUBufferBinding;
  readonly selfOutputProjection: GPUBufferBinding;
  readonly crossQueryProjection: GPUBufferBinding;
  readonly crossOutputProjection: GPUBufferBinding;
  readonly gateProjection: GPUBufferBinding;
  readonly upProjection: GPUBufferBinding;
  readonly downProjection: GPUBufferBinding;
}

export interface AceDitResolvedRepeatedDenseWeights {
  readonly layers: readonly AceDitRepeatedDenseLayerWeights[];
}

export const ACE_DIT_SHARED_TENSOR_NAMES: Readonly<AceDitSharedTensorNames> =
  Object.freeze({
    conditionProjection: Object.freeze({
      weight: "ace.decoder.condition_embedder.weight",
      bias: "ace.decoder.condition_embedder.bias",
    }),
    inputProjection: Object.freeze({
      weight: "ace.decoder.proj_in.1.weight",
      bias: "ace.decoder.proj_in.1.bias",
    }),
    timestep: timestepBranchTensorNames("ace.decoder.time_embed"),
    relativeTimestep: timestepBranchTensorNames("ace.decoder.time_embed_r"),
    output: Object.freeze({
      norm: "ace.decoder.norm_out.weight",
      scaleShiftTable: "ace.decoder.scale_shift_table",
      projection: "ace.decoder.proj_out.1.weight",
      bias: "ace.decoder.proj_out.1.bias",
    }),
  });

/**
 * Build the exact 19 canonical package tensors required by one DiT layer.
 * The bounded index prevents an otherwise plausible name from escaping the
 * pinned 24-layer package contract.
 */
export function aceDitLayerTensorNames(layer: number): AceDitLayerTensorNames {
  if (!Number.isSafeInteger(layer) || layer < 0 || layer >= ACE_DIT_LAYER_COUNT) {
    throw new RangeError(`ACE DiT tensor layer must be in [0, ${ACE_DIT_LAYER_COUNT - 1}]`);
  }
  const base = `ace.decoder.layers.${layer}`;
  return Object.freeze({
    scaleShiftTable: `${base}.scale_shift_table`,
    selfAttentionNorm: `${base}.self_attn_norm.weight`,
    selfQueryProjection: `${base}.self_attn.q_proj.weight`,
    selfKeyProjection: `${base}.self_attn.k_proj.weight`,
    selfValueProjection: `${base}.self_attn.v_proj.weight`,
    selfQueryNorm: `${base}.self_attn.q_norm.weight`,
    selfKeyNorm: `${base}.self_attn.k_norm.weight`,
    selfOutputProjection: `${base}.self_attn.o_proj.weight`,
    crossAttentionNorm: `${base}.cross_attn_norm.weight`,
    crossQueryProjection: `${base}.cross_attn.q_proj.weight`,
    crossKeyProjection: `${base}.cross_attn.k_proj.weight`,
    crossValueProjection: `${base}.cross_attn.v_proj.weight`,
    crossQueryNorm: `${base}.cross_attn.q_norm.weight`,
    crossKeyNorm: `${base}.cross_attn.k_norm.weight`,
    crossOutputProjection: `${base}.cross_attn.o_proj.weight`,
    mlpNorm: `${base}.mlp_norm.weight`,
    gateProjection: `${base}.mlp.gate_proj.weight`,
    upProjection: `${base}.mlp.up_proj.weight`,
    downProjection: `${base}.mlp.down_proj.weight`,
  });
}

export function createAceDitCanonicalTensorNameMap(): AceDitCanonicalTensorNameMap {
  return Object.freeze({
    shared: ACE_DIT_SHARED_TENSOR_NAMES,
    layers: Object.freeze(
      Array.from({ length: ACE_DIT_LAYER_COUNT }, (_, layer) =>
        aceDitLayerTensorNames(layer)
      ),
    ),
  });
}

/** Flat, deterministic package-resolution order: 20 shared, then 24 × 19. */
export function aceDitCanonicalTensorNames(): readonly string[] {
  const shared = ACE_DIT_SHARED_TENSOR_NAMES;
  return Object.freeze([
    shared.conditionProjection.weight,
    shared.conditionProjection.bias,
    shared.inputProjection.weight,
    shared.inputProjection.bias,
    ...Object.values(shared.timestep),
    ...Object.values(shared.relativeTimestep),
    ...Object.values(shared.output),
    ...Array.from({ length: ACE_DIT_LAYER_COUNT }, (_, layer) =>
      Object.values(aceDitLayerTensorNames(layer))
    ).flat(),
  ]);
}

const ACE_DIT_GEMM_WEIGHT_TENSOR_NAMES = buildAceDitGemmWeightTensorNames();
const ACE_DIT_REPEATED_DENSE_WEIGHT_TENSOR_NAMES =
  buildAceDitRepeatedDenseWeightTensorNames();

/** @internal Exact canonical package-name predicate for rank-two DiT GEMMs. */
export function isAceDitGemmWeightTensorName(name: string): boolean {
  return ACE_DIT_GEMM_WEIGHT_TENSOR_NAMES.has(name);
}

/** Exact nine denoise-time dense matrices repeated by each of 24 layers. */
export function isAceDitRepeatedDenseWeightTensorName(name: string): boolean {
  return ACE_DIT_REPEATED_DENSE_WEIGHT_TENSOR_NAMES.has(name);
}

const ACE_DIT_EXPECTED_LOGICAL_SHAPES = buildAceDitExpectedLogicalShapes();

/** Exact source-logical shape used to authenticate every graph binding. */
export function aceDitExpectedLogicalShape(name: string): readonly number[] {
  const shape = ACE_DIT_EXPECTED_LOGICAL_SHAPES.get(name);
  if (shape === undefined) {
    throw new Error(`Unknown ACE DiT logical tensor ${name}`);
  }
  return shape;
}

/** Resolve all 476 authenticated, unsharded DiT tensors into graph roles. */
export function resolveAceDitPackageWeights(
  resolver: AceDitTensorResolver,
  modelProfile: AceModelProfileId,
  mixedLayerResolver?: AceDitTensorResolver,
  denseRuntimeProfile?: AceDitDenseRuntimeProfile,
): AceDitResolvedPackageWeights {
  if (
    (mixedLayerResolver === undefined) !== (denseRuntimeProfile === undefined)
  ) {
    throw new Error(
      "ACE mixed DiT resolver and dense runtime profile must be supplied together",
    );
  }
  const names = createAceDitCanonicalTensorNameMap();
  const binding = (name: string): GPUBufferBinding => {
    const expectedShape = aceDitExpectedLogicalShape(name);
    const selected = mixedLayerResolver !== undefined &&
        name.startsWith("ace.decoder.layers.")
      ? mixedLayerResolver
      : resolver;
    const logical = selected.logicalTensor(name);
    if (
      mixedLayerResolver !== undefined &&
      isAceDitRepeatedDenseWeightTensorName(name)
    ) {
      requireAceDitRepeatedDenseLogicalTensor(
        logical,
        name,
        expectedShape,
        denseRuntimeProfile!,
      );
    } else {
      requireAceDitLogicalTensor(logical, name, expectedShape, modelProfile);
    }
    return selected.binding(name);
  };
  const branch = (
    value: AceDitTimestepBranchTensorNames,
  ): AceDitTimestepBranchWeights => Object.freeze({
    linear1Weight: binding(value.linear1Weight),
    linear1Bias: binding(value.linear1Bias),
    linear2Weight: binding(value.linear2Weight),
    linear2Bias: binding(value.linear2Bias),
    projectionWeight: binding(value.projectionWeight),
    projectionBias: binding(value.projectionBias),
  });
  const layers: AceDitLayerWeights[] = [];
  const crossCaches: AceDitCrossCacheWeights[] = [];
  for (const layer of names.layers) {
    layers.push(Object.freeze({
      scaleShiftTable: binding(layer.scaleShiftTable),
      selfAttentionNorm: binding(layer.selfAttentionNorm),
      selfQueryProjection: binding(layer.selfQueryProjection),
      selfKeyProjection: binding(layer.selfKeyProjection),
      selfValueProjection: binding(layer.selfValueProjection),
      selfQueryNorm: binding(layer.selfQueryNorm),
      selfKeyNorm: binding(layer.selfKeyNorm),
      selfOutputProjection: binding(layer.selfOutputProjection),
      crossAttentionNorm: binding(layer.crossAttentionNorm),
      crossQueryProjection: binding(layer.crossQueryProjection),
      crossQueryNorm: binding(layer.crossQueryNorm),
      crossOutputProjection: binding(layer.crossOutputProjection),
      mlpNorm: binding(layer.mlpNorm),
      gateProjection: binding(layer.gateProjection),
      upProjection: binding(layer.upProjection),
      downProjection: binding(layer.downProjection),
    }));
    crossCaches.push(Object.freeze({
      keyProjection: binding(layer.crossKeyProjection),
      valueProjection: binding(layer.crossValueProjection),
      keyNorm: binding(layer.crossKeyNorm),
    }));
  }
  return Object.freeze({
    conditionProjection: Object.freeze({
      weight: binding(names.shared.conditionProjection.weight),
      bias: binding(names.shared.conditionProjection.bias),
    }),
    inputProjection: Object.freeze({
      weight: binding(names.shared.inputProjection.weight),
      bias: binding(names.shared.inputProjection.bias),
    }),
    timestep: Object.freeze({
      timestep: branch(names.shared.timestep),
      relative: branch(names.shared.relativeTimestep),
    }),
    output: Object.freeze({
      norm: binding(names.shared.output.norm),
      scaleShiftTable: binding(names.shared.output.scaleShiftTable),
      projection: binding(names.shared.output.projection),
      bias: binding(names.shared.output.bias),
    }),
    layers: Object.freeze(layers),
    crossCaches: Object.freeze(crossCaches),
  });
}

/** Resolve the exact 24 x 9 matrices for one authenticated dense profile. */
export function resolveAceDitRepeatedDensePackageWeights(
  resolver: AceDitTensorResolver,
  denseRuntimeProfile: AceDitDenseRuntimeProfile,
): AceDitResolvedRepeatedDenseWeights {
  const binding = (name: string): GPUBufferBinding => {
    const expectedShape = aceDitExpectedLogicalShape(name);
    const logical = resolver.logicalTensor(name);
    requireAceDitRepeatedDenseLogicalTensor(
      logical,
      name,
      expectedShape,
      denseRuntimeProfile,
    );
    return resolver.binding(name);
  };
  return Object.freeze({
    layers: Object.freeze(
      Array.from({ length: ACE_DIT_LAYER_COUNT }, (_, layerIndex) => {
        const layer = aceDitLayerTensorNames(layerIndex);
        return Object.freeze({
          selfQueryProjection: binding(layer.selfQueryProjection),
          selfKeyProjection: binding(layer.selfKeyProjection),
          selfValueProjection: binding(layer.selfValueProjection),
          selfOutputProjection: binding(layer.selfOutputProjection),
          crossQueryProjection: binding(layer.crossQueryProjection),
          crossOutputProjection: binding(layer.crossOutputProjection),
          gateProjection: binding(layer.gateProjection),
          upProjection: binding(layer.upProjection),
          downProjection: binding(layer.downProjection),
        });
      }),
    ),
  });
}

/** Keep every reference binding except the nine denoise-time dense matrices. */
export function replaceAceDitRepeatedDenseWeights(
  reference: AceDitResolvedPackageWeights,
  dense: AceDitResolvedRepeatedDenseWeights,
): AceDitResolvedPackageWeights {
  if (
    reference.layers.length !== ACE_DIT_LAYER_COUNT ||
    dense.layers.length !== ACE_DIT_LAYER_COUNT
  ) {
    throw new Error("ACE mixed DiT layer inventories changed");
  }
  return Object.freeze({
    ...reference,
    layers: Object.freeze(reference.layers.map((layer, index) =>
      Object.freeze({ ...layer, ...dense.layers[index]! })
    )),
  });
}

function timestepBranchTensorNames(
  base: string,
): Readonly<AceDitTimestepBranchTensorNames> {
  return Object.freeze({
    linear1Weight: `${base}.linear_1.weight`,
    linear1Bias: `${base}.linear_1.bias`,
    linear2Weight: `${base}.linear_2.weight`,
    linear2Bias: `${base}.linear_2.bias`,
    projectionWeight: `${base}.time_proj.weight`,
    projectionBias: `${base}.time_proj.bias`,
  });
}

function buildAceDitGemmWeightTensorNames(): ReadonlySet<string> {
  const shared = ACE_DIT_SHARED_TENSOR_NAMES;
  const names = [
    shared.conditionProjection.weight,
    shared.timestep.linear1Weight,
    shared.timestep.linear2Weight,
    shared.timestep.projectionWeight,
    shared.relativeTimestep.linear1Weight,
    shared.relativeTimestep.linear2Weight,
    shared.relativeTimestep.projectionWeight,
  ];
  for (let layerIndex = 0; layerIndex < ACE_DIT_LAYER_COUNT; layerIndex += 1) {
    const layer = aceDitLayerTensorNames(layerIndex);
    names.push(
      layer.selfQueryProjection,
      layer.selfKeyProjection,
      layer.selfValueProjection,
      layer.selfOutputProjection,
      layer.crossQueryProjection,
      layer.crossKeyProjection,
      layer.crossValueProjection,
      layer.crossOutputProjection,
      layer.gateProjection,
      layer.upProjection,
      layer.downProjection,
    );
  }
  const result = new Set(names);
  const expectedCount = 7 + ACE_DIT_LAYER_COUNT * 11;
  if (result.size !== expectedCount || result.size !== names.length) {
    throw new Error(
      `ACE DiT GEMM package contract has ${result.size}/${names.length} unique names, ` +
        `expected ${expectedCount}`,
    );
  }
  return result;
}

function buildAceDitRepeatedDenseWeightTensorNames(): ReadonlySet<string> {
  const names = new Set<string>();
  for (let layerIndex = 0; layerIndex < ACE_DIT_LAYER_COUNT; layerIndex += 1) {
    const layer = aceDitLayerTensorNames(layerIndex);
    names.add(layer.selfQueryProjection);
    names.add(layer.selfKeyProjection);
    names.add(layer.selfValueProjection);
    names.add(layer.selfOutputProjection);
    names.add(layer.crossQueryProjection);
    names.add(layer.crossOutputProjection);
    names.add(layer.gateProjection);
    names.add(layer.upProjection);
    names.add(layer.downProjection);
  }
  if (names.size !== ACE_DIT_LAYER_COUNT * 9) {
    throw new Error("ACE repeated DiT dense tensor inventory changed");
  }
  return names;
}

function buildAceDitExpectedLogicalShapes(): ReadonlyMap<string, readonly number[]> {
  const config = ACE_TURBO_DIT_CONFIG;
  const hidden = config.hiddenSize;
  const intermediate = config.intermediateSize;
  const queryWidth = config.queryHeads * config.headDimension;
  const keyValueWidth = config.keyValueHeads * config.headDimension;
  const map = new Map<string, readonly number[]>();
  const add = (name: string, shape: readonly number[]): void => {
    if (map.has(name)) throw new Error(`Duplicate ACE DiT shape contract ${name}`);
    map.set(name, Object.freeze([...shape]));
  };
  const addTimestep = (names: AceDitTimestepBranchTensorNames): void => {
    add(names.linear1Weight, [hidden, config.timestepInputSize]);
    add(names.linear1Bias, [hidden]);
    add(names.linear2Weight, [hidden, hidden]);
    add(names.linear2Bias, [hidden]);
    add(names.projectionWeight, [6 * hidden, hidden]);
    add(names.projectionBias, [6 * hidden]);
  };

  const shared = ACE_DIT_SHARED_TENSOR_NAMES;
  add(shared.conditionProjection.weight, [hidden, config.conditionInputSize]);
  add(shared.conditionProjection.bias, [hidden]);
  add(shared.inputProjection.weight, [hidden, config.inputChannels, config.patchSize]);
  add(shared.inputProjection.bias, [hidden]);
  addTimestep(shared.timestep);
  addTimestep(shared.relativeTimestep);
  add(shared.output.norm, [hidden]);
  add(shared.output.scaleShiftTable, [1, 2, hidden]);
  add(shared.output.projection, [hidden, config.audioChannels, config.patchSize]);
  add(shared.output.bias, [config.audioChannels]);

  for (let index = 0; index < ACE_DIT_LAYER_COUNT; index += 1) {
    const layer = aceDitLayerTensorNames(index);
    add(layer.scaleShiftTable, [1, 6, hidden]);
    add(layer.selfAttentionNorm, [hidden]);
    add(layer.selfQueryProjection, [queryWidth, hidden]);
    add(layer.selfKeyProjection, [keyValueWidth, hidden]);
    add(layer.selfValueProjection, [keyValueWidth, hidden]);
    add(layer.selfQueryNorm, [config.headDimension]);
    add(layer.selfKeyNorm, [config.headDimension]);
    add(layer.selfOutputProjection, [hidden, queryWidth]);
    add(layer.crossAttentionNorm, [hidden]);
    add(layer.crossQueryProjection, [queryWidth, hidden]);
    add(layer.crossKeyProjection, [keyValueWidth, hidden]);
    add(layer.crossValueProjection, [keyValueWidth, hidden]);
    add(layer.crossQueryNorm, [config.headDimension]);
    add(layer.crossKeyNorm, [config.headDimension]);
    add(layer.crossOutputProjection, [hidden, queryWidth]);
    add(layer.mlpNorm, [hidden]);
    add(layer.gateProjection, [intermediate, hidden]);
    add(layer.upProjection, [intermediate, hidden]);
    add(layer.downProjection, [hidden, intermediate]);
  }
  if (map.size !== 20 + ACE_DIT_LAYER_COUNT * 19) {
    throw new Error(`ACE DiT shape contract has ${map.size} entries`);
  }
  return map;
}

function requireAceDitLogicalTensor(
  logical: AceGpuLogicalTensor,
  expectedName: string,
  expectedShape: readonly number[],
  modelProfile: AceModelProfileId,
): void {
  if (
    logical.logicalTensor !== expectedName ||
    logical.logicalShape.length !== expectedShape.length ||
    logical.logicalShape.some((value, index) => value !== expectedShape[index])
  ) {
    throw new Error(
      `ACE DiT tensor ${logical.logicalTensor} has shape ` +
        `[${logical.logicalShape.join(",")}], expected [${expectedShape.join(",")}]`,
    );
  }
  if (logical.parts.length !== 1) {
    throw new Error(
      `ACE DiT tensor ${logical.logicalTensor} unexpectedly has ` +
        `${logical.parts.length} physical parts`,
    );
  }
  const part = logical.parts[0]!;
  const tensor = part.tensor;
  const contract = aceDitStorageContract(expectedName, expectedShape, modelProfile);
  if (
    part.tensorName !== expectedName ||
    tensor.logicalTensor !== expectedName ||
    tensor.logicalShape.length !== expectedShape.length ||
    tensor.logicalShape.some((value, index) => value !== expectedShape[index]) ||
    tensor.storageShape.length !== contract.storageShape.length ||
    tensor.storageShape.some(
      (value, index) => value !== contract.storageShape[index],
    ) ||
    tensor.byteLength !== contract.byteLength ||
    tensor.phase !== "dit" ||
    tensor.lifetime !== "dit" ||
    tensor.source !== `ace-turbo-weights:${expectedName.slice("ace.".length)}` ||
    tensor.dtype !== contract.dtype ||
    tensor.layout !== contract.layout ||
    tensor.transformation !== contract.transformation ||
    tensor.partAxis !== 0 ||
    tensor.partStart !== 0 ||
    tensor.partEnd !== expectedShape[0]
  ) {
    throw new Error(`ACE DiT tensor ${logical.logicalTensor} violates its package contract`);
  }
}

function requireAceDitRepeatedDenseLogicalTensor(
  logical: AceGpuLogicalTensor,
  expectedName: string,
  expectedShape: readonly number[],
  denseRuntimeProfile: AceDitDenseRuntimeProfile,
): void {
  if (
    logical.logicalTensor !== expectedName ||
    logical.logicalShape.length !== expectedShape.length ||
    logical.logicalShape.some((value, index) => value !== expectedShape[index]) ||
    logical.parts.length !== 1
  ) {
    throw new Error(`Optimized DiT dense tensor ${expectedName} changed shape`);
  }
  const part = logical.parts[0]!;
  const tensor = part.tensor;
  const elements = expectedShape.reduce((product, extent) => product * extent, 1);
  const [columns, inner] = expectedShape;
  if (columns === undefined || inner === undefined) {
    throw new Error(`Optimized DiT dense tensor ${expectedName} is not rank two`);
  }
  const k4 =
    denseRuntimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE ||
    denseRuntimeProfile === ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
  if (
    !k4 && denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE
  ) {
    throw new TypeError(
      `Unknown ACE DiT dense runtime profile ${String(denseRuntimeProfile)}`,
    );
  }
  const expectedStorageShape = k4
    ? [columns / 128, inner / 4, 4, 32, 4]
    : expectedShape;
  const expectedLayout = k4
    ? ACE_DIT_DENSE_K4_FP16_LAYOUT
    : ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  const expectedTransformation = k4
    ? ACE_DIT_DENSE_K4_FP16_TRANSFORMATION
    : ACE_DIT_DENSE_FP16_TRANSFORMATION;
  if (
    part.tensorName !== expectedName ||
    tensor.logicalTensor !== expectedName ||
    tensor.logicalShape.length !== expectedShape.length ||
    tensor.logicalShape.some((value, index) => value !== expectedShape[index]) ||
    tensor.storageShape.length !== expectedStorageShape.length ||
    tensor.storageShape.some(
      (value, index) => value !== expectedStorageShape[index],
    ) ||
    tensor.byteLength !== elements * 2 ||
    tensor.dtype !== "float16" ||
    tensor.layout !== expectedLayout ||
    tensor.transformation !== expectedTransformation ||
    tensor.phase !== "dit" ||
    tensor.lifetime !== "dit" ||
    tensor.source !== `ace-turbo-weights:${expectedName.slice("ace.".length)}` ||
    tensor.partAxis !== 0 ||
    tensor.partStart !== 0 ||
    tensor.partEnd !== expectedShape[0]
  ) {
    throw new Error(
      `Optimized DiT dense tensor ${expectedName} violates its ` +
        `${denseRuntimeProfile} package contract`,
    );
  }
}

function aceDitStorageContract(
  name: string,
  logicalShape: readonly number[],
  modelProfile: AceModelProfileId,
): Readonly<{
  dtype: AceTensorDtype;
  layout: AceTensorLayout;
  transformation: AceTensorTransformation;
  storageShape: readonly number[];
  byteLength: number;
}> {
  const tiled = isAceDitGemmWeightTensorName(name);
  const elements = logicalShape.reduce((product, extent) => product * extent, 1);
  if (!Number.isSafeInteger(elements)) {
    throw new RangeError(`ACE DiT tensor ${name} has an unsafe element count`);
  }
  if (modelProfile === "reference-bf16") {
    return Object.freeze({
      dtype: "uint32-bf16-pairs",
      layout: tiled
        ? ACE_DIT_GEMM_TILE_LAYOUT
        : "source-row-major-bf16-pairs-lsb-u32",
      transformation: tiled
        ? ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
        : "preserve-bf16-bits-pack-u32-pairs",
      storageShape: Object.freeze([Math.ceil(elements / 2)]),
      byteLength: Math.ceil(elements / 2) * 4,
    });
  }
  if (modelProfile === "raw-fp16") {
    return Object.freeze({
      dtype: "float16",
      layout: tiled ? ACE_DIT_GEMM_TILE_LAYOUT : "source-row-major",
      transformation: tiled
        ? ACE_DIT_GEMM_FP16_TRANSFORMATION
        : "bf16-to-ieee-fp16",
      storageShape: Object.freeze([...logicalShape]),
      byteLength: elements * 2,
    });
  }
  throw new TypeError(`Unknown ACE DiT model profile ${String(modelProfile)}`);
}
