import type { AceDynamicConditionalWeightingConfiguration } from "../api.js";
import {
  ACE_GRAPH_CONTRACT,
  ACE_DIT_LAYER_TYPES,
} from "../model/graph-contract.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessAttentionKernel,
  type AceAttentionDispatch,
  type AceAttentionRuntimeConfiguration,
} from "./kernels/attention.js";
import {
  AceOpt0062QuadQueryAttentionKernel,
  type AceOpt0062AttentionRouteProfile,
} from "./kernels/attention-quad-query-production.js";
import {
  AceOpt0070ProductionAttentionKernel,
  type AceOpt0070ProductionAttentionRouteProfile,
} from "./kernels/attention-opt0070-production.js";
import {
  AceCorrectnessDcwKernel,
  deriveAceDcwStepScales,
  type AceDcwDispatch,
} from "./kernels/dcw.js";
import {
  AceCorrectnessDitPlumbingKernel,
  type AceDitDispatch,
  type AceDitModulationPlan,
  type AceDitPatchProjectionPlan,
  type AceDitPlumbingPlan,
  type AceDitUnpatchProjectionPlan,
} from "./kernels/dit-plumbing.js";
import {
  aceCompositeCooperativeSequence,
  type AceGpuEncodeSequence,
  AceCorrectnessGemmKernel,
  type AceGemmDispatch,
  type AceGemmKernel,
  type AceGemmWeightLayout,
} from "./kernels/gemm.js";
import {
  AceSubgroupGemmKernel,
  type AceFixed32SubgroupCapability,
} from "./kernels/subgroup-gemm.js";
import { AceOpt0009DenseGemmKernel } from "./kernels/dit-dense-fp16.js";
import { AceOpt0081DenseF16InputKernel } from
  "./kernels/dit-dense-f16-input.js";
import { AceOpt0037DenseK4ProductionKernel } from
  "./kernels/dit-dense-fp16-k4-production.js";
import {
  AceOpt0056SelectiveDenseKernel,
  type AceOpt0056DenseRouteProfile,
} from "./kernels/dit-dense-fp16-k4-selective-exact.js";
import {
  AceCorrectnessRmsNormKernel,
  type AceRmsNormDispatch,
} from "./kernels/rmsnorm.js";
import {
  AceCorrectnessRopeKernel,
  type AceRopeDispatch,
} from "./kernels/rope.js";
import {
  AceCorrectnessTransformerPlumbingKernel,
  type AceTransformerDispatch,
} from "./kernels/transformer-plumbing.js";
import {
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
  AceOpt0081F16DenseInputProducerKernel,
  type AceOpt0081F16DenseInputRole,
} from "./kernels/dit-f16-dense-input-producers.js";
import { aceActivationBytes } from "./kernels/correctness-utils.js";
import {
  createAceQwen3RopeTables,
  type AceQwen3RopeTables,
} from "./qwen3.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  requireResolvedAceDitSamplerScheduleProfile,
  type AceDitSamplerScheduleProfile,
} from "./dit-sampler-profile.js";

const MAX_U32 = 0xffff_ffff;
const TIMESTEP_INPUT_SIZE = 256;
const TIMESTEP_SCALE = 1_000;
const TIMESTEP_MAXIMUM_PERIOD = 10_000;
const LAYER_MODULATION_GROUPS = 6;
const OUTPUT_MODULATION_GROUPS = 2;

export interface AceDitConfig {
  readonly id: string;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly layerCount: number;
  readonly queryHeads: number;
  readonly keyValueHeads: number;
  readonly headDimension: number;
  readonly maximumPositionEmbeddings: number;
  readonly ropeTheta: number;
  readonly rmsNormEpsilon: number;
  readonly slidingRadius: number;
  readonly patchSize: number;
  readonly inputChannels: number;
  readonly audioChannels: number;
  readonly contextChannels: number;
  readonly conditionInputSize: number;
  readonly attentionBias: false;
  readonly hiddenActivation: "silu";
  readonly timestepInputSize: number;
  readonly timestepScale: number;
  readonly timestepMaximumPeriod: number;
  /** The pinned implementation overwrites both supplied masks with `None`. */
  readonly maskPolicy: "discard-all-supplied-masks";
}

export const ACE_TURBO_DIT_CONFIG: Readonly<AceDitConfig> = Object.freeze({
  id: "ace-step-1.5-turbo-dit-v1",
  hiddenSize: ACE_GRAPH_CONTRACT.dit.hiddenSize,
  intermediateSize: ACE_GRAPH_CONTRACT.dit.intermediateSize,
  layerCount: ACE_GRAPH_CONTRACT.dit.layerCount,
  queryHeads: ACE_GRAPH_CONTRACT.dit.attentionHeads,
  keyValueHeads: ACE_GRAPH_CONTRACT.dit.keyValueHeads,
  headDimension: ACE_GRAPH_CONTRACT.dit.headDimension,
  maximumPositionEmbeddings:
    ACE_GRAPH_CONTRACT.dit.maximumPositionEmbeddings,
  ropeTheta: ACE_GRAPH_CONTRACT.dit.ropeTheta,
  rmsNormEpsilon: ACE_GRAPH_CONTRACT.dit.rmsNormEpsilon,
  slidingRadius: ACE_GRAPH_CONTRACT.dit.slidingWindowRadius,
  patchSize: ACE_GRAPH_CONTRACT.dit.patchSize,
  inputChannels: ACE_GRAPH_CONTRACT.dit.inputChannels,
  audioChannels: ACE_GRAPH_CONTRACT.dit.audioChannels,
  contextChannels: ACE_GRAPH_CONTRACT.dit.contextChannels,
  conditionInputSize: ACE_GRAPH_CONTRACT.conditioner.hiddenSize,
  attentionBias: false,
  hiddenActivation: "silu",
  timestepInputSize: TIMESTEP_INPUT_SIZE,
  timestepScale: TIMESTEP_SCALE,
  timestepMaximumPeriod: TIMESTEP_MAXIMUM_PERIOD,
  maskPolicy: "discard-all-supplied-masks",
});

export type AceDitGemmRuntimeConfiguration =
  | Readonly<{
      backend: "portable";
      weightLayout: AceGemmWeightLayout;
    }>
  | Readonly<{
      backend: "subgroups";
      capability: AceFixed32SubgroupCapability;
    }>;

export type AceDitDenseGemmRuntimeConfiguration =
  | Readonly<{ backend: "reference" }>
  | Readonly<{
      backend: "opt-0009-fp16-fp32";
      capability: AceFixed32SubgroupCapability;
    }>
  | Readonly<{
      backend: "opt-0037-k4-fp16-partials";
      capability: AceFixed32SubgroupCapability;
    }>
  | Readonly<{
      backend: "opt-0056-selective-k4-exact-down";
      capability: AceFixed32SubgroupCapability;
    }>;

/**
 * Diagnostic-only repeated-layer activation storage selection. Omission is
 * the current all-FP32 graph; this literal is not a model or package profile.
 */
export type AceDitDenseInputStorageProfile =
  typeof ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE;

export function requireAceDitDenseInputStorageProfile(
  profile?: AceDitDenseInputStorageProfile,
): AceDitDenseInputStorageProfile | undefined {
  if (
    profile === undefined ||
    profile === ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE
  ) return profile;
  throw new Error("ACE DiT dense-input storage profile is not authenticated");
}

export interface AceDitEvaluationShape {
  readonly batch: number;
  readonly latentFrames: number;
  readonly conditionTokens: number;
}

export interface AceDitEvaluationPlan extends AceDitEvaluationShape {
  readonly paddedLatentFrames: number;
  readonly latentPaddingFrames: number;
  readonly tokens: number;
  readonly rows: number;
  readonly hiddenElements: number;
  readonly intermediateElements: number;
  readonly queryElements: number;
  readonly selfKeyValueElements: number;
  readonly crossKeyValueElementsPerLayer: number;
  readonly crossKeyValueElementsAllLayers: number;
  readonly contextElements: number;
  readonly latentElements: number;
}

export interface AceDitSamplerStep {
  readonly index: number;
  /** Human/API-facing shift-3 value committed in the fixture. */
  readonly declaredTimestep: number;
  /** Value actually stored by upstream in its BF16 `t_schedule`. */
  readonly timestep: number;
  readonly declaredNextTimestep: number | null;
  readonly nextTimestep: number | null;
  readonly update: "euler" | "predicted-clean";
  /** Euler `dt`, or current `t` for the final direct-x0 update. */
  readonly updateCoefficient: number;
  readonly lowBandScale: number;
  readonly highBandScale: number;
}

export interface AceDitAttentionControlData {
  readonly selfValidLengths: Uint32Array;
  readonly crossValidLengths: Uint32Array;
}

export type AceDitAttentionMode = "sliding" | "full";

export interface AceDitLayerShape {
  readonly batch: number;
  readonly tokens: number;
  readonly conditionTokens: number;
  readonly attentionMode: AceDitAttentionMode;
}

export interface AceDitLayerPlan extends AceDitLayerShape {
  readonly rows: number;
  readonly queryWidth: number;
  readonly keyValueWidth: number;
  readonly hiddenElements: number;
  readonly queryElements: number;
  readonly selfKeyValueElements: number;
  readonly crossKeyValueElements: number;
  readonly intermediateElements: number;
  readonly modulationElements: number;
}

export interface AceDitLayerWeights {
  readonly scaleShiftTable: GPUBufferBinding;
  readonly selfAttentionNorm: GPUBufferBinding;
  readonly selfQueryProjection: GPUBufferBinding;
  readonly selfKeyProjection: GPUBufferBinding;
  readonly selfValueProjection: GPUBufferBinding;
  readonly selfQueryNorm: GPUBufferBinding;
  readonly selfKeyNorm: GPUBufferBinding;
  readonly selfOutputProjection: GPUBufferBinding;
  readonly crossAttentionNorm: GPUBufferBinding;
  readonly crossQueryProjection: GPUBufferBinding;
  readonly crossQueryNorm: GPUBufferBinding;
  readonly crossOutputProjection: GPUBufferBinding;
  readonly mlpNorm: GPUBufferBinding;
  readonly gateProjection: GPUBufferBinding;
  readonly upProjection: GPUBufferBinding;
  readonly downProjection: GPUBufferBinding;
}

export interface AceDitCrossCacheWeights {
  readonly keyProjection: GPUBufferBinding;
  readonly valueProjection: GPUBufferBinding;
  readonly keyNorm: GPUBufferBinding;
}

export interface AceDitCrossCacheScratch {
  readonly keyFlat: GPUBufferBinding;
  readonly valueFlat: GPUBufferBinding;
  readonly keyHeads: GPUBufferBinding;
}

export interface AceDitCrossCacheBindings {
  readonly projectedCondition: GPUBufferBinding;
  readonly weights: AceDitCrossCacheWeights;
  readonly scratch: AceDitCrossCacheScratch;
  /** `[batch,keyValueHeads,conditionTokens,headDimension]`. */
  readonly key: GPUBufferBinding;
  readonly value: GPUBufferBinding;
}

/** Every field is a distinct Stage 1 tap/lifetime edge. */
export interface AceDitLayerScratch {
  /** Group-major `[6,batch,hiddenSize]`. */
  readonly modulation: GPUBufferBinding;
  readonly selfNormalized: GPUBufferBinding;
  readonly selfModulated: GPUBufferBinding;
  readonly selfQueryFlat: GPUBufferBinding;
  readonly selfKeyFlat: GPUBufferBinding;
  readonly selfValueFlat: GPUBufferBinding;
  readonly selfQueryHeads: GPUBufferBinding;
  readonly selfKeyHeads: GPUBufferBinding;
  readonly selfValueHeads: GPUBufferBinding;
  readonly selfNormalizedQueryHeads: GPUBufferBinding;
  readonly selfNormalizedKeyHeads: GPUBufferBinding;
  readonly selfRotatedQueryHeads: GPUBufferBinding;
  readonly selfRotatedKeyHeads: GPUBufferBinding;
  readonly selfAttentionHeads: GPUBufferBinding;
  readonly selfMergedAttention: GPUBufferBinding;
  readonly selfProjectedAttention: GPUBufferBinding;
  readonly afterSelfAttention: GPUBufferBinding;
  readonly crossNormalized: GPUBufferBinding;
  readonly crossQueryFlat: GPUBufferBinding;
  readonly crossQueryHeads: GPUBufferBinding;
  readonly crossNormalizedQueryHeads: GPUBufferBinding;
  readonly crossAttentionHeads: GPUBufferBinding;
  readonly crossMergedAttention: GPUBufferBinding;
  readonly crossProjectedAttention: GPUBufferBinding;
  readonly afterCrossAttention: GPUBufferBinding;
  readonly mlpNormalized: GPUBufferBinding;
  readonly mlpModulated: GPUBufferBinding;
  readonly gate: GPUBufferBinding;
  readonly up: GPUBufferBinding;
  readonly gatedActivation: GPUBufferBinding;
  readonly projectedMlp: GPUBufferBinding;
}

export function isAceOpt0081F16DenseInputScratchRole(
  role: keyof AceDitLayerScratch,
): role is AceOpt0081F16DenseInputRole {
  return (ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES as readonly string[])
    .includes(role);
}

export function aceDitLayerScratchBytes(
  modelProfile: AceModelProfileId,
  role: keyof AceDitLayerScratch,
  elements: number,
  denseInputStorageProfile?: AceDitDenseInputStorageProfile,
): number {
  const profile = requireAceDitDenseInputStorageProfile(
    denseInputStorageProfile,
  );
  if (profile !== undefined && modelProfile !== "reference-bf16") {
    throw new Error(
      "OPT-0081 dense-input storage requires the reference-bf16 graph profile",
    );
  }
  return aceActivationBytes(
    profile !== undefined && isAceOpt0081F16DenseInputScratchRole(role)
      ? "raw-fp16"
      : modelProfile,
    elements,
  );
}

export interface AceDitLayerBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly weights: AceDitLayerWeights;
  readonly scratch: AceDitLayerScratch;
  /** Shared `[batch,6,hiddenSize]` output from the timestep MLPs. */
  readonly timestepProjection: GPUBufferBinding;
  readonly crossKey: GPUBufferBinding;
  readonly crossValue: GPUBufferBinding;
  readonly selfValidLengths: GPUBufferBinding;
  readonly crossValidLengths: GPUBufferBinding;
  /** FP32 `[tokens,headDimension]`. */
  readonly cosine: GPUBufferBinding;
  readonly sine: GPUBufferBinding;
  /** OPT-0062 correctness-only query8/quad raw-word oracle. */
  readonly opt0062AttentionIdentity?: Readonly<{
    readonly counters: GPUBufferBinding;
    readonly routeIndex: number;
  }>;
}

export interface AceDitTimestepBranchWeights {
  readonly linear1Weight: GPUBufferBinding;
  readonly linear1Bias: GPUBufferBinding;
  readonly linear2Weight: GPUBufferBinding;
  readonly linear2Bias: GPUBufferBinding;
  readonly projectionWeight: GPUBufferBinding;
  readonly projectionBias: GPUBufferBinding;
}

export interface AceDitTimestepWeights {
  readonly timestep: AceDitTimestepBranchWeights;
  readonly relative: AceDitTimestepBranchWeights;
}

export interface AceDitTimestepScratch {
  readonly timestepFrequency: GPUBufferBinding;
  readonly relativeFrequency: GPUBufferBinding;
  readonly timestepLinear1: GPUBufferBinding;
  readonly relativeLinear1: GPUBufferBinding;
  readonly timestepActivation1: GPUBufferBinding;
  readonly relativeActivation1: GPUBufferBinding;
  readonly timestepEmbedding: GPUBufferBinding;
  readonly relativeEmbedding: GPUBufferBinding;
  readonly timestepActivation2: GPUBufferBinding;
  readonly relativeActivation2: GPUBufferBinding;
  readonly timestepProjection: GPUBufferBinding;
  readonly relativeProjection: GPUBufferBinding;
}

export interface AceDitTimestepBindings {
  /** FP32 `[batch]`; upload the `AceDitSamplerStep.timestep` value. */
  readonly timestep: GPUBufferBinding;
  /** FP32 `[batch]`; v1 uploads exact zero because `timestep_r == timestep`. */
  readonly relativeTimestep: GPUBufferBinding;
  readonly weights: AceDitTimestepWeights;
  readonly scratch: AceDitTimestepScratch;
  readonly embedding: GPUBufferBinding;
  readonly projection: GPUBufferBinding;
}

export interface AceDitOutputWeights {
  readonly norm: GPUBufferBinding;
  readonly scaleShiftTable: GPUBufferBinding;
  /** Source ConvTranspose1d layout `[hiddenSize,audioChannels,patchSize]`. */
  readonly projection: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
}

export interface AceDitOutputScratch {
  readonly normalized: GPUBufferBinding;
  /** Group-major `[2,batch,hiddenSize]`. */
  readonly modulation: GPUBufferBinding;
  readonly modulated: GPUBufferBinding;
}

export interface AceDitOutputBindings {
  readonly hidden: GPUBufferBinding;
  readonly timestepEmbedding: GPUBufferBinding;
  readonly weights: AceDitOutputWeights;
  readonly scratch: AceDitOutputScratch;
  readonly velocity: GPUBufferBinding;
}

export interface AceDitSamplerBindings {
  readonly latent: GPUBufferBinding;
  readonly velocity: GPUBufferBinding;
  readonly stepped: GPUBufferBinding;
  readonly predictedClean: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

type AceDitPrimitiveDispatch =
  | AceGemmDispatch
  | AceRmsNormDispatch
  | AceRopeDispatch
  | AceAttentionDispatch
  | AceDcwDispatch
  | AceDitDispatch<unknown>
  | AceTransformerDispatch<unknown>;

export interface AceDitCompositeDispatch<Plan> {
  readonly label: string;
  readonly plan: Plan;
  readonly primitiveCount: number;
  readonly cooperativeSequence: AceGpuEncodeSequence;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Correctness-first DiT graph composition. It owns pipelines only; the FIFO
 * graph owner supplies all model buffers, caches, activations, and scratch.
 */
export function createAceDitGemmKernel(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  configuration: AceDitGemmRuntimeConfiguration,
): AceGemmKernel {
  if (configuration.backend === "portable") {
    return AceCorrectnessGemmKernel.create(
      device,
      modelProfile,
      configuration.weightLayout,
    );
  }
  if (configuration.backend !== "subgroups") {
    throw new TypeError(
      `Unknown ACE DiT GEMM backend ${String((configuration as { backend?: unknown }).backend)}`,
    );
  }
  if (modelProfile !== "reference-bf16") {
    throw new Error(
      "ACE DiT subgroup GEMM is authenticated only for reference-bf16",
    );
  }
  return AceSubgroupGemmKernel.create(device, configuration.capability);
}

export function createAceDitDenseGemmKernel(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  configuration: Exclude<
    AceDitDenseGemmRuntimeConfiguration,
    { backend: "reference" }
  >,
): AceGemmKernel {
  if (modelProfile !== "reference-bf16") {
    throw new Error(
      "Optimized dense GEMM requires the reference-bf16 graph profile",
    );
  }
  if (configuration.backend === "opt-0009-fp16-fp32") {
    return AceOpt0009DenseGemmKernel.create(device, configuration.capability);
  }
  if (configuration.backend === "opt-0037-k4-fp16-partials") {
    return AceOpt0037DenseK4ProductionKernel.create(
      device,
      configuration.capability,
    );
  }
  if (configuration.backend === "opt-0056-selective-k4-exact-down") {
    return AceOpt0056SelectiveDenseKernel.create(
      device,
      configuration.capability,
    );
  }
  throw new TypeError(
    `Unknown optimized ACE DiT dense backend ${String(
      (configuration as { backend?: unknown }).backend,
    )}`,
  );
}

function requireOpt0081DiagnosticRuntimeConfiguration(
  modelProfile: AceModelProfileId,
  gemmConfiguration: AceDitGemmRuntimeConfiguration,
  denseGemmConfiguration: AceDitDenseGemmRuntimeConfiguration,
  attentionConfiguration: AceAttentionRuntimeConfiguration,
  denseInputStorageProfile?: AceDitDenseInputStorageProfile,
): AceDitDenseInputStorageProfile | undefined {
  const profile = requireAceDitDenseInputStorageProfile(
    denseInputStorageProfile,
  );
  if (profile === undefined) return undefined;
  if (
    modelProfile !== "reference-bf16" ||
    gemmConfiguration.backend !== "subgroups" ||
    gemmConfiguration.capability.subgroupMinSize !== 32 ||
    gemmConfiguration.capability.subgroupMaxSize !== 32 ||
    denseGemmConfiguration.backend !== "opt-0009-fp16-fp32" ||
    denseGemmConfiguration.capability.subgroupMinSize !== 32 ||
    denseGemmConfiguration.capability.subgroupMaxSize !== 32 ||
    attentionConfiguration.backend !==
      "opt-0070-fixed32-quad-query32-full-self-production" ||
    attentionConfiguration.capability.subgroupMinSize !== 32 ||
    attentionConfiguration.capability.subgroupMaxSize !== 32 ||
    attentionConfiguration.expectedQueryTokens !== 2_250 ||
    attentionConfiguration.expectedConditionTokens !== 98
  ) {
    throw new Error(
      "OPT-0081 dense-input storage requires the exact M2250/C98 " +
        "OPT-0009/OPT-0070 fixed32 diagnostic tuple",
    );
  }
  return profile;
}

function createOpt0081DenseInputKernel(
  device: GPUDevice,
  configuration: AceDitDenseGemmRuntimeConfiguration,
): AceOpt0081DenseF16InputKernel {
  if (configuration.backend !== "opt-0009-fp16-fp32") {
    throw new Error(
      "OPT-0081 dense-input storage and dense owner configuration diverged",
    );
  }
  return AceOpt0081DenseF16InputKernel.create(
    device,
    configuration.capability,
  );
}

export class AceCorrectnessDitRuntime {
  readonly modelProfile: AceModelProfileId;
  readonly attentionBackend: AceAttentionRuntimeConfiguration["backend"];
  readonly denseInputStorageProfile: AceDitDenseInputStorageProfile | undefined;

  private readonly gemm: AceGemmKernel;
  private readonly denseGemm: AceGemmKernel;
  private readonly rmsNorm: AceCorrectnessRmsNormKernel;
  private readonly rope: AceCorrectnessRopeKernel;
  private readonly attention:
    | AceCorrectnessAttentionKernel
    | AceOpt0062QuadQueryAttentionKernel
    | AceOpt0070ProductionAttentionKernel;
  private readonly transformer: AceCorrectnessTransformerPlumbingKernel;
  private readonly denseInputProducer:
    | AceOpt0081F16DenseInputProducerKernel
    | undefined;
  private readonly dit: AceCorrectnessDitPlumbingKernel;
  private readonly dcw: AceCorrectnessDcwKernel;
  private destroyed = false;

  private constructor(
    modelProfile: AceModelProfileId,
    gemm: AceGemmKernel,
    denseGemm: AceGemmKernel,
    rmsNorm: AceCorrectnessRmsNormKernel,
    rope: AceCorrectnessRopeKernel,
    attention:
      | AceCorrectnessAttentionKernel
      | AceOpt0062QuadQueryAttentionKernel
      | AceOpt0070ProductionAttentionKernel,
    transformer: AceCorrectnessTransformerPlumbingKernel,
    denseInputProducer: AceOpt0081F16DenseInputProducerKernel | undefined,
    dit: AceCorrectnessDitPlumbingKernel,
    dcw: AceCorrectnessDcwKernel,
    denseInputStorageProfile: AceDitDenseInputStorageProfile | undefined,
  ) {
    this.modelProfile = modelProfile;
    this.attentionBackend = attention.configuration.backend;
    this.gemm = gemm;
    this.denseGemm = denseGemm;
    this.rmsNorm = rmsNorm;
    this.rope = rope;
    this.attention = attention;
    this.transformer = transformer;
    this.denseInputProducer = denseInputProducer;
    this.dit = dit;
    this.dcw = dcw;
    this.denseInputStorageProfile = denseInputStorageProfile;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    gemmConfiguration: AceDitGemmRuntimeConfiguration,
    /** @internal Direct primitive harnesses explicitly remain reference-only. */
    denseGemmConfiguration: AceDitDenseGemmRuntimeConfiguration = {
      backend: "reference",
    },
    /** @internal Direct primitive harnesses explicitly remain portable. */
    attentionConfiguration: AceAttentionRuntimeConfiguration = {
      backend: "portable",
    },
    /** @internal OPT-0081 representative-layer diagnostic only. */
    denseInputStorageProfile?: AceDitDenseInputStorageProfile,
  ): AceCorrectnessDitRuntime {
    const resolvedDenseInputStorageProfile =
      requireOpt0081DiagnosticRuntimeConfiguration(
        modelProfile,
        gemmConfiguration,
        denseGemmConfiguration,
        attentionConfiguration,
        denseInputStorageProfile,
      );
    let gemm: AceGemmKernel | undefined;
    let denseGemm: AceGemmKernel | undefined;
    let rmsNorm: AceCorrectnessRmsNormKernel | undefined;
    let rope: AceCorrectnessRopeKernel | undefined;
    let attention:
      | AceCorrectnessAttentionKernel
      | AceOpt0062QuadQueryAttentionKernel
      | AceOpt0070ProductionAttentionKernel
      | undefined;
    let transformer: AceCorrectnessTransformerPlumbingKernel | undefined;
    let denseInputProducer:
      | AceOpt0081F16DenseInputProducerKernel
      | undefined;
    let dit: AceCorrectnessDitPlumbingKernel | undefined;
    let dcw: AceCorrectnessDcwKernel | undefined;
    try {
      gemm = createAceDitGemmKernel(device, modelProfile, gemmConfiguration);
      denseGemm = resolvedDenseInputStorageProfile !== undefined
        ? createOpt0081DenseInputKernel(device, denseGemmConfiguration)
        : denseGemmConfiguration.backend === "reference"
        ? gemm
        : createAceDitDenseGemmKernel(
            device,
            modelProfile,
            denseGemmConfiguration,
          );
      rmsNorm = AceCorrectnessRmsNormKernel.create(device, modelProfile);
      rope = AceCorrectnessRopeKernel.create(device, modelProfile);
      attention = attentionConfiguration.backend ===
          "opt-0062-fixed32-quad-query32-full-self"
        ? AceOpt0062QuadQueryAttentionKernel.create(
            device,
            modelProfile,
            attentionConfiguration,
          )
        : attentionConfiguration.backend ===
            "opt-0070-fixed32-quad-query32-full-self-production"
          ? AceOpt0070ProductionAttentionKernel.create(
              device,
              modelProfile,
              attentionConfiguration,
            )
          : AceCorrectnessAttentionKernel.create(
              device,
              modelProfile,
              attentionConfiguration,
            );
      transformer = AceCorrectnessTransformerPlumbingKernel.create(
        device,
        modelProfile,
      );
      denseInputProducer = resolvedDenseInputStorageProfile === undefined
        ? undefined
        : AceOpt0081F16DenseInputProducerKernel.create(device);
      dit = AceCorrectnessDitPlumbingKernel.create(device, modelProfile);
      dcw = AceCorrectnessDcwKernel.create(device, modelProfile);
      return new AceCorrectnessDitRuntime(
        modelProfile,
        gemm,
        denseGemm,
        rmsNorm,
        rope,
        attention,
        transformer,
        denseInputProducer,
        dit,
        dcw,
        resolvedDenseInputStorageProfile,
      );
    } catch (error) {
      if (denseGemm !== gemm) denseGemm?.destroy();
      gemm?.destroy();
      rmsNorm?.destroy();
      rope?.destroy();
      attention?.destroy();
      denseInputProducer?.destroy();
      transformer?.destroy();
      dit?.destroy();
      dcw?.destroy();
      throw error;
    }
  }

  async createConditionProjectionDispatch(
    label: string,
    config: AceDitConfig,
    batch: number,
    conditionTokens: number,
    bindings: Readonly<{
      readonly input: GPUBufferBinding;
      readonly weight: GPUBufferBinding;
      readonly bias: GPUBufferBinding;
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceGemmDispatch> {
    this.requireLive();
    validateAceDitConfig(config);
    requireDistinctWritableBindings(label, [
      { name: "projected condition", binding: bindings.output },
    ], [
      { name: "condition input", binding: bindings.input },
      { name: "condition projection weight", binding: bindings.weight },
      { name: "condition projection bias", binding: bindings.bias },
    ]);
    const rows = checkedProduct(
      [batch, conditionTokens],
      `${label} condition rows`,
    );
    return await this.gemm.createDispatch(label, {
      rows,
      inner: config.conditionInputSize,
      columns: config.hiddenSize,
    }, {
      activation: bindings.input,
      weight: bindings.weight,
      bias: bindings.bias,
      output: bindings.output,
    });
  }

  async createCrossCacheDispatch(
    label: string,
    config: AceDitConfig,
    shape: AceDitLayerShape,
    bindings: AceDitCrossCacheBindings,
  ): Promise<AceDitCompositeDispatch<AceDitLayerPlan>> {
    this.requireLive();
    const plan = planAceDitLayer(config, shape);
    requireDistinctWritableBindings(label, [
      ...namedBindings("scratch", bindings.scratch),
      { name: "cross key", binding: bindings.key },
      { name: "cross value", binding: bindings.value },
    ], [
      { name: "projected condition", binding: bindings.projectedCondition },
      ...namedBindings("weight", bindings.weights),
    ]);
    const keyValueHeadShape = {
      batch: plan.batch,
      tokens: plan.conditionTokens,
      heads: config.keyValueHeads,
      headDimension: config.headDimension,
    } as const;
    const dispatches: readonly AceDitPrimitiveDispatch[] = await Promise.all([
      this.gemm.createDispatch(`${label}-key-projection`, {
        rows: plan.batch * plan.conditionTokens,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: bindings.projectedCondition,
        weight: bindings.weights.keyProjection,
        output: bindings.scratch.keyFlat,
      }),
      this.gemm.createDispatch(`${label}-value-projection`, {
        rows: plan.batch * plan.conditionTokens,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: bindings.projectedCondition,
        weight: bindings.weights.valueProjection,
        output: bindings.scratch.valueFlat,
      }),
      this.transformer.createHeadTransformDispatch(
        `${label}-split-key-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: bindings.scratch.keyFlat, output: bindings.scratch.keyHeads },
      ),
      this.transformer.createHeadTransformDispatch(
        `${label}-split-value-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: bindings.scratch.valueFlat, output: bindings.value },
      ),
      this.rmsNorm.createDispatch(`${label}-key-norm`, {
        rows: plan.batch * config.keyValueHeads * plan.conditionTokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: bindings.scratch.keyHeads,
        weight: bindings.weights.keyNorm,
        output: bindings.key,
      }),
    ]);
    this.requireLive(" while compiling");
    return compositeDispatch(label, plan, dispatches);
  }

  async createLayerDispatch(
    label: string,
    config: AceDitConfig,
    shape: AceDitLayerShape,
    bindings: AceDitLayerBindings,
  ): Promise<AceDitCompositeDispatch<AceDitLayerPlan>> {
    this.requireLive();
    const plan = planAceDitLayer(config, shape);
    const denseInputProducer = this.denseInputProducer;
    if (
      this.denseInputStorageProfile !== undefined &&
      (denseInputProducer === undefined ||
        config !== ACE_TURBO_DIT_CONFIG ||
        plan.batch !== 1 ||
        plan.tokens !== 2_250 ||
        plan.conditionTokens !== 98 ||
        bindings.opt0062AttentionIdentity !== undefined)
    ) {
      throw new Error(
        "OPT-0081 dense-input storage is closed to exact M2250/C98 " +
          "production-attention layers without the OPT-0062 oracle",
      );
    }
    if (
      bindings.opt0062AttentionIdentity !== undefined &&
      plan.attentionMode !== "full"
    ) {
      throw new Error("OPT-0062 identity oracle is valid only on full-self layers");
    }
    requireDistinctWritableBindings(label, [
      ...namedBindings("scratch", bindings.scratch),
      { name: "output", binding: bindings.output },
      ...(bindings.opt0062AttentionIdentity === undefined
        ? []
        : [{
            name: "OPT-0062 identity counters",
            binding: bindings.opt0062AttentionIdentity.counters,
          }]),
    ], [
      { name: "input", binding: bindings.input },
      ...namedBindings("weight", bindings.weights),
      { name: "timestep projection", binding: bindings.timestepProjection },
      { name: "cross key", binding: bindings.crossKey },
      { name: "cross value", binding: bindings.crossValue },
      { name: "self valid lengths", binding: bindings.selfValidLengths },
      { name: "cross valid lengths", binding: bindings.crossValidLengths },
      { name: "cosine", binding: bindings.cosine },
      { name: "sine", binding: bindings.sine },
    ]);

    const hiddenShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      width: config.hiddenSize,
    } as const;
    const intermediateShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      width: config.intermediateSize,
    } as const;
    const queryHeadShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      heads: config.queryHeads,
      headDimension: config.headDimension,
    } as const;
    const keyValueHeadShape = {
      batch: plan.batch,
      tokens: plan.tokens,
      heads: config.keyValueHeads,
      headDimension: config.headDimension,
    } as const;
    const s = bindings.scratch;
    const w = bindings.weights;
    const modulationBytes = aceActivationBytes(
      this.modelProfile,
      plan.batch * config.hiddenSize,
    );
    const modulation = Array.from({ length: LAYER_MODULATION_GROUPS }, (_, group) =>
      sliceBinding(s.modulation, group * modulationBytes, modulationBytes, `${label} modulation ${group}`)
    );

    const dispatches: AceDitPrimitiveDispatch[] = [];
    dispatches.push(await this.dit.createModulationDispatch(
      `${label}-modulation`,
      {
        batch: plan.batch,
        groups: LAYER_MODULATION_GROUPS,
        width: config.hiddenSize,
        projectionLayout: "per-group",
      },
      {
        projection: bindings.timestepProjection,
        table: w.scaleShiftTable,
        output: s.modulation,
      },
    ));
    dispatches.push(await this.rmsNorm.createDispatch(`${label}-self-norm`, {
      rows: plan.rows,
      width: config.hiddenSize,
      epsilon: config.rmsNormEpsilon,
    }, {
      input: bindings.input,
      weight: w.selfAttentionNorm,
      output: s.selfNormalized,
    }));
    dispatches.push(await (denseInputProducer === undefined
      ? this.transformer.createAdaLnDispatch(
          `${label}-self-adaln`,
          hiddenShape,
          {
            normalized: s.selfNormalized,
            scale: modulation[1]!,
            shift: modulation[0]!,
            output: s.selfModulated,
          },
        )
      : denseInputProducer.createSelfModulatedDispatch(
          `${label}-self-adaln`,
          "selfModulated",
          "adaln",
          hiddenShape,
          {
            normalized: s.selfNormalized,
            scale: modulation[1]!,
            shift: modulation[0]!,
            output: s.selfModulated,
          },
        )));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.denseGemm.createDispatch(`${label}-self-query-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.queryWidth,
      }, {
        activation: s.selfModulated,
        weight: w.selfQueryProjection,
        output: s.selfQueryFlat,
      }),
      this.denseGemm.createDispatch(`${label}-self-key-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: s.selfModulated,
        weight: w.selfKeyProjection,
        output: s.selfKeyFlat,
      }),
      this.denseGemm.createDispatch(`${label}-self-value-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.keyValueWidth,
      }, {
        activation: s.selfModulated,
        weight: w.selfValueProjection,
        output: s.selfValueFlat,
      }),
    ]));
    dispatches.push(
      await this.transformer.createHeadTransformDispatch(
        `${label}-self-split-query-heads`,
        "split-heads",
        queryHeadShape,
        { input: s.selfQueryFlat, output: s.selfQueryHeads },
      ),
      await this.transformer.createHeadTransformDispatch(
        `${label}-self-split-key-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: s.selfKeyFlat, output: s.selfKeyHeads },
      ),
      await this.transformer.createHeadTransformDispatch(
        `${label}-self-split-value-heads`,
        "split-heads",
        keyValueHeadShape,
        { input: s.selfValueFlat, output: s.selfValueHeads },
      ),
    );
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.rmsNorm.createDispatch(`${label}-self-query-norm`, {
        rows: plan.batch * config.queryHeads * plan.tokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.selfQueryHeads,
        weight: w.selfQueryNorm,
        output: s.selfNormalizedQueryHeads,
      }),
      this.rmsNorm.createDispatch(`${label}-self-key-norm`, {
        rows: plan.batch * config.keyValueHeads * plan.tokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.selfKeyHeads,
        weight: w.selfKeyNorm,
        output: s.selfNormalizedKeyHeads,
      }),
    ]));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.rope.createDispatch(`${label}-self-query-rope`, queryHeadShape, {
        input: s.selfNormalizedQueryHeads,
        cosine: bindings.cosine,
        sine: bindings.sine,
        output: s.selfRotatedQueryHeads,
      }),
      this.rope.createDispatch(`${label}-self-key-rope`, keyValueHeadShape, {
        input: s.selfNormalizedKeyHeads,
        cosine: bindings.cosine,
        sine: bindings.sine,
        output: s.selfRotatedKeyHeads,
      }),
    ]));
    dispatches.push(await this.attention.createDispatch(
      `${label}-self-${plan.attentionMode}-attention`,
      {
        batch: plan.batch,
        queryHeads: config.queryHeads,
        keyValueHeads: config.keyValueHeads,
        queryTokens: plan.tokens,
        keyValueTokens: plan.tokens,
        headDimension: config.headDimension,
        mode: plan.attentionMode,
        ...(plan.attentionMode === "sliding"
          ? { slidingRadius: config.slidingRadius }
          : {}),
      },
      {
        query: s.selfRotatedQueryHeads,
        key: s.selfRotatedKeyHeads,
        value: s.selfValueHeads,
        validLengths: bindings.selfValidLengths,
        output: s.selfAttentionHeads,
        ...(bindings.opt0062AttentionIdentity === undefined
          ? {}
          : {
              opt0062Identity: {
                oracleOutput: s.selfMergedAttention,
                counters: bindings.opt0062AttentionIdentity.counters,
                routeIndex: bindings.opt0062AttentionIdentity.routeIndex,
              },
            }),
      },
    ));
    dispatches.push(
      await (denseInputProducer === undefined
        ? this.transformer.createHeadTransformDispatch(
            `${label}-self-merge-heads`,
            "merge-heads",
            queryHeadShape,
            { input: s.selfAttentionHeads, output: s.selfMergedAttention },
          )
        : denseInputProducer.createSelfMergedAttentionDispatch(
            `${label}-self-merge-heads`,
            "selfMergedAttention",
            "merge-heads",
            queryHeadShape,
            { input: s.selfAttentionHeads, output: s.selfMergedAttention },
          )),
      await this.denseGemm.createDispatch(`${label}-self-output-projection`, {
        rows: plan.rows,
        inner: plan.queryWidth,
        columns: config.hiddenSize,
      }, {
        activation: s.selfMergedAttention,
        weight: w.selfOutputProjection,
        output: s.selfProjectedAttention,
      }),
      await this.transformer.createGatedResidualDispatch(
        `${label}-self-gated-residual`,
        hiddenShape,
        {
          residual: bindings.input,
          branch: s.selfProjectedAttention,
          gate: modulation[2]!,
          output: s.afterSelfAttention,
        },
      ),
      await (denseInputProducer === undefined
        ? this.rmsNorm.createDispatch(`${label}-cross-norm`, {
            rows: plan.rows,
            width: config.hiddenSize,
            epsilon: config.rmsNormEpsilon,
          }, {
            input: s.afterSelfAttention,
            weight: w.crossAttentionNorm,
            output: s.crossNormalized,
          })
        : denseInputProducer.createCrossNormalizedDispatch(
            `${label}-cross-norm`,
            "crossNormalized",
            "cross-rmsnorm",
            {
              rows: plan.rows,
              width: config.hiddenSize,
              epsilon: config.rmsNormEpsilon,
            },
            {
              input: s.afterSelfAttention,
              weight: w.crossAttentionNorm,
              output: s.crossNormalized,
            },
          )),
      await this.denseGemm.createDispatch(`${label}-cross-query-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: plan.queryWidth,
      }, {
        activation: s.crossNormalized,
        weight: w.crossQueryProjection,
        output: s.crossQueryFlat,
      }),
      await this.transformer.createHeadTransformDispatch(
        `${label}-cross-split-query-heads`,
        "split-heads",
        queryHeadShape,
        { input: s.crossQueryFlat, output: s.crossQueryHeads },
      ),
      await this.rmsNorm.createDispatch(`${label}-cross-query-norm`, {
        rows: plan.batch * config.queryHeads * plan.tokens,
        width: config.headDimension,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.crossQueryHeads,
        weight: w.crossQueryNorm,
        output: s.crossNormalizedQueryHeads,
      }),
      await this.attention.createDispatch(`${label}-cross-attention`, {
        batch: plan.batch,
        queryHeads: config.queryHeads,
        keyValueHeads: config.keyValueHeads,
        queryTokens: plan.tokens,
        keyValueTokens: plan.conditionTokens,
        headDimension: config.headDimension,
        mode: "full",
      }, {
        query: s.crossNormalizedQueryHeads,
        key: bindings.crossKey,
        value: bindings.crossValue,
        validLengths: bindings.crossValidLengths,
        output: s.crossAttentionHeads,
      }),
      await (denseInputProducer === undefined
        ? this.transformer.createHeadTransformDispatch(
            `${label}-cross-merge-heads`,
            "merge-heads",
            queryHeadShape,
            { input: s.crossAttentionHeads, output: s.crossMergedAttention },
          )
        : denseInputProducer.createCrossMergedAttentionDispatch(
            `${label}-cross-merge-heads`,
            "crossMergedAttention",
            "merge-heads",
            queryHeadShape,
            { input: s.crossAttentionHeads, output: s.crossMergedAttention },
          )),
      await this.denseGemm.createDispatch(`${label}-cross-output-projection`, {
        rows: plan.rows,
        inner: plan.queryWidth,
        columns: config.hiddenSize,
      }, {
        activation: s.crossMergedAttention,
        weight: w.crossOutputProjection,
        output: s.crossProjectedAttention,
      }),
      await this.transformer.createResidualAddDispatch(
        `${label}-cross-residual`,
        hiddenShape,
        {
          left: s.afterSelfAttention,
          right: s.crossProjectedAttention,
          output: s.afterCrossAttention,
        },
      ),
      await this.rmsNorm.createDispatch(`${label}-mlp-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: s.afterCrossAttention,
        weight: w.mlpNorm,
        output: s.mlpNormalized,
      }),
      await (denseInputProducer === undefined
        ? this.transformer.createAdaLnDispatch(
            `${label}-mlp-adaln`,
            hiddenShape,
            {
              normalized: s.mlpNormalized,
              scale: modulation[4]!,
              shift: modulation[3]!,
              output: s.mlpModulated,
            },
          )
        : denseInputProducer.createMlpModulatedDispatch(
            `${label}-mlp-adaln`,
            "mlpModulated",
            "adaln",
            hiddenShape,
            {
              normalized: s.mlpNormalized,
              scale: modulation[4]!,
              shift: modulation[3]!,
              output: s.mlpModulated,
            },
          )),
    );
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.denseGemm.createDispatch(`${label}-mlp-gate-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }, {
        activation: s.mlpModulated,
        weight: w.gateProjection,
        output: s.gate,
      }),
      this.denseGemm.createDispatch(`${label}-mlp-up-projection`, {
        rows: plan.rows,
        inner: config.hiddenSize,
        columns: config.intermediateSize,
      }, {
        activation: s.mlpModulated,
        weight: w.upProjection,
        output: s.up,
      }),
    ]));
    dispatches.push(
      await (denseInputProducer === undefined
        ? this.transformer.createSwiGluDispatch(
            `${label}-mlp-swiglu`,
            intermediateShape,
            { gate: s.gate, up: s.up, output: s.gatedActivation },
          )
        : denseInputProducer.createGatedActivationDispatch(
            `${label}-mlp-swiglu`,
            "gatedActivation",
            "swiglu",
            intermediateShape,
            { gate: s.gate, up: s.up, output: s.gatedActivation },
          )),
      await this.denseGemm.createDispatch(`${label}-mlp-down-projection`, {
        rows: plan.rows,
        inner: config.intermediateSize,
        columns: config.hiddenSize,
      }, {
        activation: s.gatedActivation,
        weight: w.downProjection,
        output: s.projectedMlp,
      }),
      await this.transformer.createGatedResidualDispatch(
        `${label}-mlp-gated-residual`,
        hiddenShape,
        {
          residual: s.afterCrossAttention,
          branch: s.projectedMlp,
          gate: modulation[5]!,
          output: bindings.output,
        },
      ),
    );
    this.requireLive(" while compiling");
    return compositeDispatch(label, plan, Object.freeze(dispatches));
  }

  async createTimestepDispatch(
    label: string,
    config: AceDitConfig,
    batch: number,
    bindings: AceDitTimestepBindings,
  ): Promise<AceDitCompositeDispatch<Readonly<{ batch: number }>>> {
    this.requireLive();
    validateAceDitConfig(config);
    requirePositiveInteger(batch, `${label} batch`);
    requireDistinctWritableBindings(label, [
      ...namedBindings("scratch", bindings.scratch),
      { name: "embedding", binding: bindings.embedding },
      { name: "projection", binding: bindings.projection },
    ], [
      { name: "timestep", binding: bindings.timestep },
      { name: "relative timestep", binding: bindings.relativeTimestep },
      ...namedBindings("timestep weight", bindings.weights.timestep),
      ...namedBindings("relative weight", bindings.weights.relative),
    ]);
    const hiddenShape = { batch, tokens: 1, width: config.hiddenSize } as const;
    const projectionShape = {
      batch,
      tokens: LAYER_MODULATION_GROUPS,
      width: config.hiddenSize,
    } as const;
    const s = bindings.scratch;
    const t = bindings.weights.timestep;
    const r = bindings.weights.relative;
    const dispatches: AceDitPrimitiveDispatch[] = [
      await this.dit.createTimestepEmbeddingDispatch(
        `${label}-timestep-frequency`,
        {
          batch,
          dimension: config.timestepInputSize,
          scale: config.timestepScale,
          maximumPeriod: config.timestepMaximumPeriod,
        },
        { timestep: bindings.timestep, output: s.timestepFrequency },
      ),
      await this.dit.createTimestepEmbeddingDispatch(
        `${label}-relative-frequency`,
        {
          batch,
          dimension: config.timestepInputSize,
          scale: config.timestepScale,
          maximumPeriod: config.timestepMaximumPeriod,
        },
        { timestep: bindings.relativeTimestep, output: s.relativeFrequency },
      ),
    ];
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.gemm.createDispatch(`${label}-timestep-linear-1`, {
        rows: batch,
        inner: config.timestepInputSize,
        columns: config.hiddenSize,
      }, {
        activation: s.timestepFrequency,
        weight: t.linear1Weight,
        bias: t.linear1Bias,
        output: s.timestepLinear1,
      }),
      this.gemm.createDispatch(`${label}-relative-linear-1`, {
        rows: batch,
        inner: config.timestepInputSize,
        columns: config.hiddenSize,
      }, {
        activation: s.relativeFrequency,
        weight: r.linear1Weight,
        bias: r.linear1Bias,
        output: s.relativeLinear1,
      }),
    ]));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.transformer.createSiluDispatch(
        `${label}-timestep-silu-1`,
        hiddenShape,
        { input: s.timestepLinear1, output: s.timestepActivation1 },
      ),
      this.transformer.createSiluDispatch(
        `${label}-relative-silu-1`,
        hiddenShape,
        { input: s.relativeLinear1, output: s.relativeActivation1 },
      ),
    ]));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.gemm.createDispatch(`${label}-timestep-linear-2`, {
        rows: batch,
        inner: config.hiddenSize,
        columns: config.hiddenSize,
      }, {
        activation: s.timestepActivation1,
        weight: t.linear2Weight,
        bias: t.linear2Bias,
        output: s.timestepEmbedding,
      }),
      this.gemm.createDispatch(`${label}-relative-linear-2`, {
        rows: batch,
        inner: config.hiddenSize,
        columns: config.hiddenSize,
      }, {
        activation: s.relativeActivation1,
        weight: r.linear2Weight,
        bias: r.linear2Bias,
        output: s.relativeEmbedding,
      }),
    ]));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.transformer.createSiluDispatch(
        `${label}-timestep-silu-2`,
        hiddenShape,
        { input: s.timestepEmbedding, output: s.timestepActivation2 },
      ),
      this.transformer.createSiluDispatch(
        `${label}-relative-silu-2`,
        hiddenShape,
        { input: s.relativeEmbedding, output: s.relativeActivation2 },
      ),
    ]));
    dispatches.push(...await Promise.all<AceDitPrimitiveDispatch>([
      this.gemm.createDispatch(`${label}-timestep-projection`, {
        rows: batch,
        inner: config.hiddenSize,
        columns: LAYER_MODULATION_GROUPS * config.hiddenSize,
      }, {
        activation: s.timestepActivation2,
        weight: t.projectionWeight,
        bias: t.projectionBias,
        output: s.timestepProjection,
      }),
      this.gemm.createDispatch(`${label}-relative-projection`, {
        rows: batch,
        inner: config.hiddenSize,
        columns: LAYER_MODULATION_GROUPS * config.hiddenSize,
      }, {
        activation: s.relativeActivation2,
        weight: r.projectionWeight,
        bias: r.projectionBias,
        output: s.relativeProjection,
      }),
      this.transformer.createResidualAddDispatch(
        `${label}-embedding-add`,
        hiddenShape,
        {
          left: s.timestepEmbedding,
          right: s.relativeEmbedding,
          output: bindings.embedding,
        },
      ),
      this.transformer.createResidualAddDispatch(
        `${label}-projection-add`,
        projectionShape,
        {
          left: s.timestepProjection,
          right: s.relativeProjection,
          output: bindings.projection,
        },
      ),
    ]));
    this.requireLive(" while compiling");
    return compositeDispatch(label, Object.freeze({ batch }), Object.freeze(dispatches));
  }

  async createInputProjectionDispatch(
    label: string,
    config: AceDitConfig,
    shape: AceDitEvaluationShape,
    bindings: Readonly<{
      readonly context: GPUBufferBinding;
      readonly latent: GPUBufferBinding;
      readonly concatenated: GPUBufferBinding;
      readonly weight: GPUBufferBinding;
      readonly bias: GPUBufferBinding;
      readonly output: GPUBufferBinding;
    }>,
  ): Promise<AceDitCompositeDispatch<AceDitEvaluationPlan>> {
    this.requireLive();
    const plan = planAceDitEvaluation(config, shape);
    validateAceDitInputProjectionBindingAliases(label, bindings);
    const dispatches: readonly AceDitPrimitiveDispatch[] = [
      await this.dit.createConcatenateDispatch(`${label}-concatenate`, {
        batch: plan.batch,
        time: plan.latentFrames,
        leftWidth: config.contextChannels,
        rightWidth: config.audioChannels,
      }, {
        left: bindings.context,
        right: bindings.latent,
        output: bindings.concatenated,
      }),
      await this.dit.createPatchProjectionDispatch(`${label}-patch`, {
        batch: plan.batch,
        time: plan.latentFrames,
        inputChannels: config.inputChannels,
        hiddenSize: config.hiddenSize,
        patchSize: config.patchSize,
      }, {
        input: bindings.concatenated,
        weight: bindings.weight,
        bias: bindings.bias,
        output: bindings.output,
      }),
    ];
    this.requireLive(" while compiling");
    return compositeDispatch(label, plan, dispatches);
  }

  async createOutputProjectionDispatch(
    label: string,
    config: AceDitConfig,
    shape: AceDitEvaluationShape,
    bindings: AceDitOutputBindings,
  ): Promise<AceDitCompositeDispatch<AceDitEvaluationPlan>> {
    this.requireLive();
    const plan = planAceDitEvaluation(config, shape);
    requireDistinctWritableBindings(label, [
      ...namedBindings("scratch", bindings.scratch),
      { name: "velocity", binding: bindings.velocity },
    ], [
      { name: "hidden", binding: bindings.hidden },
      { name: "timestep embedding", binding: bindings.timestepEmbedding },
      ...namedBindings("weight", bindings.weights),
    ]);
    const modulationBytes = aceActivationBytes(
      this.modelProfile,
      plan.batch * config.hiddenSize,
    );
    const modulation = Array.from({ length: OUTPUT_MODULATION_GROUPS }, (_, group) =>
      sliceBinding(
        bindings.scratch.modulation,
        group * modulationBytes,
        modulationBytes,
        `${label} output modulation ${group}`,
      )
    );
    const dispatches: readonly AceDitPrimitiveDispatch[] = [
      await this.rmsNorm.createDispatch(`${label}-norm`, {
        rows: plan.rows,
        width: config.hiddenSize,
        epsilon: config.rmsNormEpsilon,
      }, {
        input: bindings.hidden,
        weight: bindings.weights.norm,
        output: bindings.scratch.normalized,
      }),
      await this.dit.createModulationDispatch(`${label}-modulation`, {
        batch: plan.batch,
        groups: OUTPUT_MODULATION_GROUPS,
        width: config.hiddenSize,
        projectionLayout: "per-batch",
      }, {
        projection: bindings.timestepEmbedding,
        table: bindings.weights.scaleShiftTable,
        output: bindings.scratch.modulation,
      }),
      await this.transformer.createAdaLnDispatch(`${label}-adaln`, {
        batch: plan.batch,
        tokens: plan.tokens,
        width: config.hiddenSize,
      }, {
        normalized: bindings.scratch.normalized,
        scale: modulation[1]!,
        shift: modulation[0]!,
        output: bindings.scratch.modulated,
      }),
      await this.dit.createUnpatchProjectionDispatch(`${label}-unpatch`, {
        batch: plan.batch,
        time: plan.latentFrames,
        outputChannels: config.audioChannels,
        hiddenSize: config.hiddenSize,
        patchSize: config.patchSize,
      }, {
        input: bindings.scratch.modulated,
        weight: bindings.weights.projection,
        bias: bindings.weights.bias,
        output: bindings.velocity,
      }),
    ];
    this.requireLive(" while compiling");
    return compositeDispatch(label, plan, dispatches);
  }

  async createSamplerStepDispatch(
    label: string,
    config: AceDitConfig,
    shape: AceDitEvaluationShape,
    step: AceDitSamplerStep,
    samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile>,
    dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
    bindings: AceDitSamplerBindings,
  ): Promise<AceDitCompositeDispatch<AceDitSamplerStep>> {
    this.requireLive();
    const plan = planAceDitEvaluation(config, shape);
    const canonicalStep = createAceDitSamplerSchedule(
      samplerScheduleProfile,
      dcwConfiguration,
    )[step.index];
    if (canonicalStep === undefined || !samplerStepsEqual(step, canonicalStep)) {
      throw new RangeError(`${label} sampler step is not the pinned Turbo schedule`);
    }
    requireDistinctWritableBindings(label, [
      { name: "stepped", binding: bindings.stepped },
      { name: "predicted clean", binding: bindings.predictedClean },
      { name: "output", binding: bindings.output },
    ], [
      { name: "latent", binding: bindings.latent },
      { name: "velocity", binding: bindings.velocity },
    ]);
    const latentShape = {
      batch: plan.batch,
      time: plan.latentFrames,
      channels: config.audioChannels,
    } as const;
    const dispatches: readonly AceDitPrimitiveDispatch[] = [
      await this.dit.createLinearUpdateDispatch(`${label}-sampler-update`, {
        ...latentShape,
        coefficient: step.updateCoefficient,
      }, {
        latent: bindings.latent,
        velocity: bindings.velocity,
        output: bindings.stepped,
      }),
      await this.dit.createLinearUpdateDispatch(`${label}-predicted-clean`, {
        ...latentShape,
        coefficient: step.timestep,
      }, {
        latent: bindings.latent,
        velocity: bindings.velocity,
        output: bindings.predictedClean,
      }),
      await this.dcw.createDispatch(
        `${label}-dcw`,
        latentShape,
        dcwConfiguration,
        step.timestep,
        {
          steppedLatent: bindings.stepped,
          predictedCleanLatent: bindings.predictedClean,
          output: bindings.output,
        },
      ),
    ];
    this.requireLive(" while compiling");
    return compositeDispatch(label, step, dispatches);
  }

  /** Benchmark-only proof that all 216 static repeated-dense routes compiled. */
  finalizeDenseRoutes(): AceOpt0056DenseRouteProfile | undefined {
    this.requireLive(" while finalizing dense routes");
    return this.denseGemm instanceof AceOpt0056SelectiveDenseKernel
      ? this.denseGemm.finalizeRoutes()
      : undefined;
  }

  /** OPT-0062 proof that all and only 96 production full-self routes compiled. */
  finalizeAttentionRoutes():
    | AceOpt0062AttentionRouteProfile
    | AceOpt0070ProductionAttentionRouteProfile
    | undefined {
    this.requireLive(" while finalizing attention routes");
    return this.attention instanceof AceOpt0062QuadQueryAttentionKernel ||
        this.attention instanceof AceOpt0070ProductionAttentionKernel
      ? this.attention.finalizeRoutes()
      : undefined;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.denseGemm !== this.gemm) this.denseGemm.destroy();
    this.gemm.destroy();
    this.rmsNorm.destroy();
    this.rope.destroy();
    this.attention.destroy();
    this.denseInputProducer?.destroy();
    this.transformer.destroy();
    this.dit.destroy();
    this.dcw.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE DiT runtime was destroyed${suffix}`);
    }
  }
}

export function planAceDitEvaluation(
  config: AceDitConfig,
  shape: AceDitEvaluationShape,
): AceDitEvaluationPlan {
  validateAceDitConfig(config);
  requirePositiveInteger(shape.batch, "ACE DiT batch");
  requirePositiveInteger(shape.latentFrames, "ACE DiT latent frames");
  requirePositiveInteger(shape.conditionTokens, "ACE DiT condition tokens");
  const tokens = Math.ceil(shape.latentFrames / config.patchSize);
  const paddedLatentFrames = checkedProduct(
    [tokens, config.patchSize],
    "ACE DiT padded latent frames",
  );
  const latentPaddingFrames = paddedLatentFrames - shape.latentFrames;
  if (tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE DiT tokens exceed maximumPositionEmbeddings");
  }
  const rows = checkedProduct([shape.batch, tokens], "ACE DiT rows");
  const queryWidth = config.queryHeads * config.headDimension;
  const keyValueWidth = config.keyValueHeads * config.headDimension;
  const hiddenElements = checkedProduct(
    [rows, config.hiddenSize],
    "ACE DiT hidden elements",
  );
  const intermediateElements = checkedProduct(
    [rows, config.intermediateSize],
    "ACE DiT intermediate elements",
  );
  const queryElements = checkedProduct(
    [rows, queryWidth],
    "ACE DiT query elements",
  );
  const selfKeyValueElements = checkedProduct(
    [rows, keyValueWidth],
    "ACE DiT self KV elements",
  );
  const crossKeyValueElementsPerLayer = checkedProduct(
    [shape.batch, shape.conditionTokens, keyValueWidth],
    "ACE DiT cross KV elements",
  );
  const crossKeyValueElementsAllLayers = checkedProduct(
    [crossKeyValueElementsPerLayer, config.layerCount, 2],
    "ACE DiT all cross K/V elements",
  );
  const contextElements = checkedProduct(
    [shape.batch, shape.latentFrames, config.contextChannels],
    "ACE DiT context elements",
  );
  const latentElements = checkedProduct(
    [shape.batch, shape.latentFrames, config.audioChannels],
    "ACE DiT latent elements",
  );
  for (const [name, value] of Object.entries({
    hiddenElements,
    intermediateElements,
    queryElements,
    selfKeyValueElements,
    crossKeyValueElementsPerLayer,
    crossKeyValueElementsAllLayers,
    contextElements,
    latentElements,
  })) {
    if (value > MAX_U32) {
      throw new RangeError(`ACE DiT ${name} exceeds U32 shader indexing`);
    }
  }
  return Object.freeze({
    ...shape,
    paddedLatentFrames,
    latentPaddingFrames,
    tokens,
    rows,
    hiddenElements,
    intermediateElements,
    queryElements,
    selfKeyValueElements,
    crossKeyValueElementsPerLayer,
    crossKeyValueElementsAllLayers,
    contextElements,
    latentElements,
  });
}

export function planAceDitLayer(
  config: AceDitConfig,
  shape: AceDitLayerShape,
): AceDitLayerPlan {
  validateAceDitConfig(config);
  requirePositiveInteger(shape.batch, "ACE DiT layer batch");
  requirePositiveInteger(shape.tokens, "ACE DiT layer tokens");
  requirePositiveInteger(shape.conditionTokens, "ACE DiT layer condition tokens");
  if (shape.tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE DiT layer tokens exceed maximumPositionEmbeddings");
  }
  if (shape.attentionMode !== "full" && shape.attentionMode !== "sliding") {
    throw new TypeError(
      `Unknown ACE DiT attention mode ${String(shape.attentionMode)}`,
    );
  }
  const rows = checkedProduct([shape.batch, shape.tokens], "ACE DiT layer rows");
  const queryWidth = checkedProduct(
    [config.queryHeads, config.headDimension],
    "ACE DiT query width",
  );
  const keyValueWidth = checkedProduct(
    [config.keyValueHeads, config.headDimension],
    "ACE DiT KV width",
  );
  const hiddenElements = checkedProduct(
    [rows, config.hiddenSize],
    "ACE DiT layer hidden elements",
  );
  const queryElements = checkedProduct(
    [rows, queryWidth],
    "ACE DiT layer query elements",
  );
  const selfKeyValueElements = checkedProduct(
    [rows, keyValueWidth],
    "ACE DiT layer self KV elements",
  );
  const crossKeyValueElements = checkedProduct(
    [shape.batch, shape.conditionTokens, keyValueWidth],
    "ACE DiT layer cross KV elements",
  );
  const intermediateElements = checkedProduct(
    [rows, config.intermediateSize],
    "ACE DiT layer intermediate elements",
  );
  const modulationElements = checkedProduct(
    [LAYER_MODULATION_GROUPS, shape.batch, config.hiddenSize],
    "ACE DiT layer modulation elements",
  );
  for (const [name, value] of Object.entries({
    hiddenElements,
    queryElements,
    selfKeyValueElements,
    crossKeyValueElements,
    intermediateElements,
    modulationElements,
  })) {
    if (value > MAX_U32) {
      throw new RangeError(`ACE DiT layer ${name} exceeds U32 shader indexing`);
    }
  }
  return Object.freeze({
    ...shape,
    rows,
    queryWidth,
    keyValueWidth,
    hiddenElements,
    queryElements,
    selfKeyValueElements,
    crossKeyValueElements,
    intermediateElements,
    modulationElements,
  });
}

export function validateAceDitConfig(config: AceDitConfig): void {
  if (typeof config.id !== "string" || config.id.length === 0) {
    throw new TypeError("ACE DiT config id must be non-empty");
  }
  for (const [name, value] of Object.entries({
    hiddenSize: config.hiddenSize,
    intermediateSize: config.intermediateSize,
    layerCount: config.layerCount,
    queryHeads: config.queryHeads,
    keyValueHeads: config.keyValueHeads,
    headDimension: config.headDimension,
    maximumPositionEmbeddings: config.maximumPositionEmbeddings,
    slidingRadius: config.slidingRadius,
    patchSize: config.patchSize,
    inputChannels: config.inputChannels,
    audioChannels: config.audioChannels,
    contextChannels: config.contextChannels,
    conditionInputSize: config.conditionInputSize,
    timestepInputSize: config.timestepInputSize,
  })) {
    requirePositiveInteger(value, `ACE DiT ${name}`);
    if (value > MAX_U32) throw new RangeError(`ACE DiT ${name} exceeds U32`);
  }
  if (config.queryHeads % config.keyValueHeads !== 0) {
    throw new RangeError("ACE DiT query heads must be divisible by KV heads");
  }
  if (
    config.queryHeads * config.headDimension !== config.hiddenSize ||
    config.headDimension % 2 !== 0 ||
    config.headDimension > 128
  ) {
    throw new RangeError("ACE DiT query/head geometry is inconsistent");
  }
  if (config.contextChannels + config.audioChannels !== config.inputChannels) {
    throw new RangeError("ACE DiT input channel geometry is inconsistent");
  }
  if (!Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0) {
    throw new RangeError("ACE DiT ropeTheta must be positive and finite");
  }
  if (!Number.isFinite(config.rmsNormEpsilon) || config.rmsNormEpsilon <= 0) {
    throw new RangeError("ACE DiT rmsNormEpsilon must be positive and finite");
  }
  if (!Number.isFinite(config.timestepScale) || config.timestepScale <= 0) {
    throw new RangeError("ACE DiT timestep scale must be positive and finite");
  }
  if (
    !Number.isFinite(config.timestepMaximumPeriod) ||
    config.timestepMaximumPeriod <= 1
  ) {
    throw new RangeError("ACE DiT timestep maximum period must exceed one");
  }
  if (config.attentionBias !== false) {
    throw new TypeError("ACE DiT correctness runtime does not permit attention bias");
  }
  if (config.hiddenActivation !== "silu") {
    throw new TypeError("ACE DiT correctness runtime requires SiLU/SwiGLU");
  }
  if (config.maskPolicy !== "discard-all-supplied-masks") {
    throw new TypeError("ACE DiT must preserve the pinned mask-discard behavior");
  }
}

export function aceDitLayerAttentionMode(layerIndex: number): AceDitAttentionMode {
  if (
    !Number.isSafeInteger(layerIndex) ||
    layerIndex < 0 ||
    layerIndex >= ACE_DIT_LAYER_TYPES.length
  ) {
    throw new RangeError("ACE DiT layer index is outside the pinned 24 layers");
  }
  return ACE_DIT_LAYER_TYPES[layerIndex] === "sliding_attention"
    ? "sliding"
    : "full";
}

/** All rows are admitted because the pinned DiT discards both supplied masks. */
export function createAceDitAttentionControlData(
  shape: AceDitEvaluationShape,
  config: AceDitConfig = ACE_TURBO_DIT_CONFIG,
): AceDitAttentionControlData {
  const plan = planAceDitEvaluation(config, shape);
  const selfValidLengths = new Uint32Array(plan.batch * 2);
  const crossValidLengths = new Uint32Array(plan.batch * 2);
  for (let batch = 0; batch < plan.batch; batch += 1) {
    selfValidLengths[batch * 2] = plan.tokens;
    selfValidLengths[batch * 2 + 1] = plan.tokens;
    crossValidLengths[batch * 2] = plan.tokens;
    crossValidLengths[batch * 2 + 1] = plan.conditionTokens;
  }
  return Object.freeze({ selfValidLengths, crossValidLengths });
}

export function createAceDitRopeTables(
  tokens: number,
  config: AceDitConfig = ACE_TURBO_DIT_CONFIG,
): AceQwen3RopeTables {
  validateAceDitConfig(config);
  requirePositiveInteger(tokens, "ACE DiT RoPE tokens");
  if (tokens > config.maximumPositionEmbeddings) {
    throw new RangeError("ACE DiT RoPE tokens exceed maximumPositionEmbeddings");
  }
  return createAceQwen3RopeTables(
    Array.from({ length: tokens }, (_, index) => index),
    {
      batch: 1,
      tokens,
      headDimension: config.headDimension,
      ropeTheta: config.ropeTheta,
      maximumPositionEmbeddings: config.maximumPositionEmbeddings,
    },
  );
}

/**
 * Reproduce the native Turbo loop's BF16 schedule materialization. The
 * declared values remain attached so reports can explain the rounding.
 */
export function createAceDitTurboSamplerSchedule(
  dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
): readonly AceDitSamplerStep[] {
  return createAceDitSamplerSchedule(
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
    dcwConfiguration,
  );
}

/** Materialize one authenticated schedule with the unchanged DCW equations. */
export function createAceDitSamplerSchedule(
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile>,
  dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
): readonly AceDitSamplerStep[] {
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const declared = profile.declaredTimesteps;
  const actual = profile.effectiveBfloat16Timesteps;
  const updates = profile.effectiveBfloat16UpdateCoefficients;
  if (
    actual.length !== declared.length ||
    updates.length !== actual.length ||
    actual.length !== profile.evaluationCount ||
    declared.some(
      (timestep, index) =>
        roundAceBfloat16ToNumber(timestep) !== actual[index],
    )
  ) {
    throw new Error("ACE declared and effective BF16 sampler schedules diverged");
  }
  return Object.freeze(actual.map((timestep, index) => {
    const nextTimestep = actual[index + 1] ?? null;
    const declaredNextTimestep = declared[index + 1] ?? null;
    const update = nextTimestep === null ? "predicted-clean" : "euler";
    const updateCoefficient = roundAceBfloat16ToNumber(
      nextTimestep === null ? timestep : timestep - nextTimestep,
    );
    if (updateCoefficient !== updates[index]) {
      throw new Error(
        `ACE sampler schedule ${profile.id} update ${index} diverged`,
      );
    }
    const scales = deriveAceDcwStepScales(dcwConfiguration, timestep);
    return Object.freeze({
      index,
      declaredTimestep: declared[index]!,
      timestep,
      declaredNextTimestep,
      nextTimestep,
      update,
      updateCoefficient,
      lowBandScale: scales.lowBandScale,
      highBandScale: scales.highBandScale,
    });
  }));
}

export function roundAceBfloat16ToNumber(value: number): number {
  if (!Number.isFinite(value)) {
    throw new RangeError("ACE BF16 rounding requires a finite value");
  }
  const f32 = new Float32Array([value]);
  const words = new Uint32Array(f32.buffer);
  const bits = words[0]!;
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff_0000;
  words[0] = rounded >>> 0;
  return f32[0]!;
}

/** Fail before concat can overwrite a later patch-projection input or weight. */
export function validateAceDitInputProjectionBindingAliases(
  label: string,
  bindings: Readonly<{
    readonly context: GPUBufferBinding;
    readonly latent: GPUBufferBinding;
    readonly concatenated: GPUBufferBinding;
    readonly weight: GPUBufferBinding;
    readonly bias: GPUBufferBinding;
    readonly output: GPUBufferBinding;
  }>,
): void {
  requireDistinctWritableBindings(label, [
    { name: "concatenated", binding: bindings.concatenated },
    { name: "output", binding: bindings.output },
  ], [
    { name: "context", binding: bindings.context },
    { name: "latent", binding: bindings.latent },
    { name: "weight", binding: bindings.weight },
    { name: "bias", binding: bindings.bias },
  ]);
}

function samplerStepsEqual(left: AceDitSamplerStep, right: AceDitSamplerStep): boolean {
  return Object.keys(right).every((key) =>
    left[key as keyof AceDitSamplerStep] === right[key as keyof AceDitSamplerStep]
  );
}

function compositeDispatch<Plan>(
  label: string,
  plan: Plan,
  dispatches: readonly AceDitPrimitiveDispatch[],
): AceDitCompositeDispatch<Plan> {
  const cooperativeSequence = aceCompositeCooperativeSequence(dispatches);
  return Object.freeze({
    label,
    plan,
    primitiveCount: dispatches.length,
    cooperativeSequence,
    encode(pass: GPUComputePassEncoder): void {
      for (const dispatch of dispatches) dispatch.encode(pass);
    },
  });
}

interface NamedBinding {
  readonly name: string;
  readonly binding: GPUBufferBinding;
}

function namedBindings(prefix: string, value: object): NamedBinding[] {
  return Object.entries(value).map(([name, binding]) => ({
    name: `${prefix} ${name}`,
    binding: binding as GPUBufferBinding,
  }));
}

function requireDistinctWritableBindings(
  label: string,
  writable: readonly NamedBinding[],
  readonlyBindings: readonly NamedBinding[],
): void {
  for (let index = 0; index < writable.length; index += 1) {
    const left = writable[index]!;
    for (let other = index + 1; other < writable.length; other += 1) {
      requireNonOverlappingBindings(label, left, writable[other]!);
    }
    for (const read of readonlyBindings) {
      requireNonOverlappingBindings(label, left, read);
    }
  }
}

function requireNonOverlappingBindings(
  label: string,
  left: NamedBinding,
  right: NamedBinding,
): void {
  if (left.binding.buffer !== right.binding.buffer) return;
  const leftRange = bindingRange(left.binding, `${label} ${left.name}`);
  const rightRange = bindingRange(right.binding, `${label} ${right.name}`);
  if (leftRange.start < rightRange.end && rightRange.start < leftRange.end) {
    throw new RangeError(`${label} ${left.name} overlaps ${right.name}`);
  }
}

function sliceBinding(
  binding: GPUBufferBinding,
  relativeOffset: number,
  size: number,
  label: string,
): GPUBufferBinding {
  const range = bindingRange(binding, label);
  if (
    !Number.isSafeInteger(relativeOffset) ||
    relativeOffset < 0 ||
    !Number.isSafeInteger(size) ||
    size <= 0 ||
    relativeOffset + size > range.end - range.start
  ) {
    throw new RangeError(`${label} slice is outside its parent binding`);
  }
  return {
    buffer: binding.buffer,
    offset: range.start + relativeOffset,
    size,
  };
}

function bindingRange(
  binding: GPUBufferBinding,
  label: string,
): { readonly start: number; readonly end: number } {
  const start = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - start;
  if (
    !Number.isSafeInteger(start) ||
    start < 0 ||
    !Number.isSafeInteger(size) ||
    size <= 0 ||
    start + size > binding.buffer.size
  ) {
    throw new RangeError(`${label} is not a valid buffer range`);
  }
  return { start, end: start + size };
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1;
  for (const value of values) {
    requirePositiveInteger(value, label);
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} is not a safe integer`);
    }
  }
  return result;
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

// Re-export plan types for callers that allocate exact graph buffers.
export type {
  AceDitModulationPlan,
  AceDitPatchProjectionPlan,
  AceDitPlumbingPlan,
  AceDitUnpatchProjectionPlan,
};
