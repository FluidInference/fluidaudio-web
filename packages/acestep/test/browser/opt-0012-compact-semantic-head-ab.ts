/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import opt0012CoreSource from "../../benchmark/opt-0012-compact-semantic-head.ts?raw";
import apiSource from "../../src/api.ts?raw";
import graphContractSource from "../../src/model/graph-contract.ts?raw";
import strictJsonSource from "../../src/model/strict-json.ts?raw";
import manifestSource from "../../src/model/manifest.ts?raw";
import plannerSource from "../../src/runtime/planner.ts?raw";
import plannerSamplingSource from "../../src/runtime/planner-sampling.ts?raw";
import seedSource from "../../src/runtime/seed.ts?raw";
import qwenBpeSource from "../../src/tokenizer/qwen-bpe.ts?raw";
import tokenizerIndexSource from "../../src/tokenizer/index.ts?raw";
import tokenizerLoaderSource from "../../src/tokenizer/loader.ts?raw";
import tokenizerChatSource from "../../src/tokenizer/chat.ts?raw";
import tokenizerConditioningTextSource from "../../src/tokenizer/conditioning-text.ts?raw";
import plannerModelSource from "../../src/webgpu/planner-model.ts?raw";
import capabilitiesSource from "../../src/webgpu/capabilities.ts?raw";
import attentionSource from "../../src/webgpu/kernels/attention.ts?raw";
import batchedRopeSource from "../../src/webgpu/kernels/batched-rope.ts?raw";
import embeddingSource from "../../src/webgpu/kernels/embedding.ts?raw";
import gemmSource from "../../src/webgpu/kernels/gemm.ts?raw";
import kvCacheSource from "../../src/webgpu/kernels/kv-cache.ts?raw";
import rmsnormSource from "../../src/webgpu/kernels/rmsnorm.ts?raw";
import tensorCopySource from "../../src/webgpu/kernels/tensor-copy.ts?raw";
import transformerPlumbingSource from "../../src/webgpu/kernels/transformer-plumbing.ts?raw";
import correctnessUtilsSource from "../../src/webgpu/kernels/correctness-utils.ts?raw";
import qwen3Source from "../../src/webgpu/qwen3.ts?raw";
import acquireSource from "../../src/model/acquire.ts?raw";
import cacheSource from "../../src/model/cache.ts?raw";
import gpuTensorsSource from "../../src/model/gpu-tensors.ts?raw";
import gpuUploadSource from "../../src/model/gpu-upload.ts?raw";
import packageSource from "../../src/model/package.ts?raw";
import sha256Source from "../../src/model/sha256.ts?raw";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import deviceSource from "../../src/webgpu/device.ts?raw";
import plannerExecutorSource from "../../src/webgpu/planner-executor.ts?raw";
import arenaSource from "../../src/webgpu/arena.ts?raw";
import scopedBufferSource from "../../src/webgpu/scoped-buffer-allocation.ts?raw";

import {
  ACE_OPT_0012_CORE_SOURCE_IDENTITIES,
  ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
  ACE_OPT_0012_NEGATIVE_INFINITY_U32,
  ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
  ACE_OPT_0012_REGULAR_ALLOWED_TOKENS,
  ACE_OPT_0012_SAMPLING_PARAMETERS,
  AceOpt0012CompactSamplingCursor,
  aceOpt0012Float16BitsToFloat32U32,
  createAceOpt0012CompactSemanticHeadPlan,
  createAceOpt0012PlanRequest,
  decodeAceOpt0012CompactFp16Readback,
  reconstructAceOpt0012FullPlannerLogits,
  traceAceOpt0012CompactBrowserV1Sampling,
  type AceOpt0012CompactSemanticHeadPlan,
  type AceOpt0012CompactSamplingTrace,
  type AceOpt0012DecodedCompactLogits,
  type AceOpt0012PlanRequest,
  type AceOpt0012SemanticState,
  type AceOpt0012SourceShardBinding,
} from "../../benchmark/opt-0012-compact-semantic-head.js";
import { DEFAULT_ACE_PLANNER_CONFIGURATION } from "../../src/api.js";
import { ACE_REQUIRED_WEBGPU_LIMITS } from "../../src/webgpu/capabilities.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import { ACE_PACKAGE_CONVERTER_REVISION, type AcePackageFileRecord, type AcePackageManifest, type AcePackageTensorRecord } from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  createAcePlannerCodePrompts,
  type AcePlannerDecodeBatch,
  type AcePlannerPrefillBatch,
} from "../../src/runtime/planner.js";
import {
  AcePlannerSamplingCursor,
  applyAcePlannerRepetitionPenalty,
  applyAcePlannerTopK,
  combineAcePlannerCfgLogits,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerBrowserTopPKeep,
  createAcePlannerFilteredLogits,
  maskAcePlannerLogits,
  sampleAcePlannerToken,
  type AcePlannerSamplingParameters,
} from "../../src/runtime/planner-sampling.js";
import { aceCategoricalTokenFromWord, aceRandomWord, canonicalizeSeed } from "../../src/runtime/seed.js";
import { ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS } from "../../src/runtime/scheduler.js";
import { ACE_QWEN_IM_END_TOKEN_ID, ACE_QWEN_PAD_TOKEN_ID, loadPinnedAceTokenizer, type AceQwenBpeTokenizer } from "../../src/tokenizer/index.js";
import { requestAceWebGpuDevice, type AceGpuRuntimeEvent, type AceWebGpuDeviceContext } from "../../src/webgpu/device.js";
import { AceCorrectnessGemmKernel, aceCorrectnessGemmWgsl, type AceGemmDispatch } from "../../src/webgpu/kernels/gemm.js";
import { ACE_PLANNER_EMBEDDING_ROW_PARTS, type AcePlannerModelDispatch } from "../../src/webgpu/planner-model.js";
import {
  AcePlannerGpuExecutor,
  reconstructAcePlannerLogits,
  type AcePlannerGpuExecutorProgress,
  type AcePlannerLogitReadbackLayout,
  type AcePlannerPreparedGpuExecutorResources,
  type AcePlannerPreparedPhaseGpuResources,
} from "../../src/webgpu/planner-executor.js";
import { ACE_PLANNER_QWEN3_CONFIG, createAceQwen3CausalControlData } from "../../src/webgpu/qwen3.js";

export const OPT_0012_ALLOCATION_COMMIT = "f5e8e5db0b88a9a44dc96b73319183114daf136a" as const;
export const OPT_0012_CORE_COMMIT = "f73380bceebdd5568d93908a67ff33cea2b7d8f0" as const;
export const OPT_0012_CORE_SOURCE_SHA256 = "f8f2eb7cd72d8cdf2414be1d13a501553a44d30752bde02754d4ae43e88e1681" as const;
export const OPT_0012_FP16_MANIFEST_SHA256 = "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b" as const;
export const OPT_0012_FP16_MANIFEST_PATH = "/model/files-fp16/manifest.json";
export const OPT_0012_ACCEPTED_SEMANTIC_CODE_SHA256 = "42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be" as const;
export const OPT_0012_ACCEPTED_COT_RECEIPT_SHA256 = "554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678" as const;
export const OPT_0012_THERMAL_SOURCE = "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0012_THERMAL_POLL_MILLISECONDS = 1_000;
export const OPT_0012_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
export const OPT_0012_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0012_PLANNER_TENSOR_COUNT = 314;
export const OPT_0012_PLANNER_WEIGHT_FILE_COUNT = 33;
export const OPT_0012_PLANNER_RESIDENT_BYTES = 1_325_768_704;
export const OPT_0012_TOKENIZER_FILE_COUNT = 3;
export const OPT_0012_ACQUIRED_FILE_COUNT = 36;
export const OPT_0012_FULL_HEAD_LOGIT_BYTES = 868_816;
export const OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET = 868_864;
export const OPT_0012_FULL_READBACK_STATUS_BYTES = 8;
export const OPT_0012_FULL_READBACK_USED_BYTES = 868_872;
export const OPT_0012_FULL_READBACK_ALLOCATION_BYTES = 869_120;
export const OPT_0012_MODEL_QUANTUM_COUNT = 33;
export const OPT_0012_PRE_HEAD_QUANTUM_COUNT = 31;
export const OPT_0012_FULL_COMMAND_BUFFER_COUNT = 34;
export const OPT_0012_CANDIDATE_COMMAND_BUFFER_COUNT = 33;
export const OPT_0012_PREFILL_MODEL_QUANTUM_COUNT = 145;
export const OPT_0012_PREFILL_PRE_HEAD_QUANTUM_COUNT = 143;
export const OPT_0012_FULL_PREFILL_COMMAND_BUFFER_COUNT = 146;
export const OPT_0012_CANDIDATE_PREFILL_COMMAND_BUFFER_COUNT = 145;
export const OPT_0012_FULL_PHYSICAL_DISPATCH_COUNT = 628;
export const OPT_0012_REGULAR_PHYSICAL_DISPATCH_COUNT = 625;
export const OPT_0012_EOS_PHYSICAL_DISPATCH_COUNT = 624;
export const OPT_0012_FULL_COPY_COUNT = 6;
export const OPT_0012_REGULAR_COPY_COUNT = 3;
export const OPT_0012_EOS_COPY_COUNT = 2;
export const OPT_0012_FP16_DOMAIN_WORD_COUNT = 0x1_0000;
export const OPT_0012_FP16_NON_NAN_WORD_COUNT = 63_490;
export const OPT_0012_FP16_NAN_WORD_COUNT = 2_046;
export const OPT_0012_FP16_DOMAIN_OUTPUT_SHA256 = "b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf" as const;
export const OPT_0012_FP16_NON_NAN_OUTPUT_SHA256 = "680bbc22915f61aa1bbfc7265bc3882a6aa42d299bfd2c571807196e5544de2e" as const;
export const OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256 = "32f37d24c421f50695da47516d20cffa27c96fb2be53d1ac6f88cfbc1cec9039" as const;
export const OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT = 54;

const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const HEARTBEAT_INTERVAL_MILLISECONDS = 10;
const CANDIDATE_OUTPUT_GUARD_BYTES = 256;
const SENTINEL_A = 0xa5;
const SENTINEL_B = 0x5a;
const ACCEPTED_SEED = canonicalizeSeed("000000000badc0de");
const FIRST_SEMANTIC_DRAW_INDEX = 109n;

const ACCEPTED_RESOLVED_CAPTION =
  "A clean electric guitar plays a gentle, melodic chord progression with a " +
  "slightly funky, neo-soul feel. The tone is warm and direct, with a light " +
  "touch of reverb adding a touch of space to the arpeggiated chords. This " +
  "short, looping instrumental piece feels like a thoughtful intro or a " +
  "mellow interlude.";
const ACCEPTED_LYRICS =
  "[Verse]\nOpen the curtains, let the whole day in\n" +
  "Dust in the sunlight starts to spin\n\n[Chorus]\n" +
  "We found the rhythm under our feet\n" +
  "Turn up the room and follow the beat";
const ACCEPTED_COT_TEXT = `<think>
bpm: 100
caption: A clean electric guitar plays a gentle, melodic chord progression with a
  slightly funky, neo-soul feel. The tone is warm and direct, with a light touch of
  reverb adding a touch of space to the arpeggiated chords. This short, looping instrumental
  piece feels like a thoughtful intro or a mellow interlude.
duration: 30
keyscale: B minor
language: unknown
timesignature: 2
</think>`;

export type Opt0012Arm = "A" | "B" | "C";

export function opt0012ExpectedCommandBufferTopology(
  arm: Opt0012Arm,
  phaseKind: "prefill" | "decode",
): Readonly<{
  readonly preHeadCommandBufferCount: number;
  readonly modelCommandBufferCount: number;
  readonly totalCommandBufferCount: number;
}> {
  if (phaseKind === "prefill") {
    return Object.freeze({
      preHeadCommandBufferCount: OPT_0012_PREFILL_PRE_HEAD_QUANTUM_COUNT,
      modelCommandBufferCount: arm === "A" ? OPT_0012_PREFILL_MODEL_QUANTUM_COUNT : OPT_0012_CANDIDATE_PREFILL_COMMAND_BUFFER_COUNT - 1,
      totalCommandBufferCount: arm === "A" ? OPT_0012_FULL_PREFILL_COMMAND_BUFFER_COUNT : OPT_0012_CANDIDATE_PREFILL_COMMAND_BUFFER_COUNT,
    });
  }
  return Object.freeze({
    preHeadCommandBufferCount: OPT_0012_PRE_HEAD_QUANTUM_COUNT,
    modelCommandBufferCount: arm === "A" ? OPT_0012_MODEL_QUANTUM_COUNT : OPT_0012_CANDIDATE_COMMAND_BUFFER_COUNT - 1,
    totalCommandBufferCount: arm === "A" ? OPT_0012_FULL_COMMAND_BUFFER_COUNT : OPT_0012_CANDIDATE_COMMAND_BUFFER_COUNT,
  });
}
export type Opt0012CasePosition = "short" | "mid" | "long";

export interface Opt0012HeadTailQuantumDescriptor {
  readonly id: string;
  readonly logicalId?: string;
  readonly kind: string;
  readonly layer: number | null;
  readonly primitiveCount: number;
}

export interface Opt0012ObservedHeadDispatchDescriptor {
  readonly pipelineLabel: string;
  readonly bindGroupLabel: string;
  readonly workgroups: readonly [number, number, number];
}

export interface Opt0012ObservedCopyDescriptor {
  readonly sourceBufferLabel: string;
  readonly sourceOffset: number;
  readonly destinationBufferLabel: string;
  readonly destinationOffset: number;
  readonly copiedBytes: number;
}

export function authenticateOpt0012ObservedCopies(arm: Opt0012Arm, state: AceOpt0012SemanticState, copies: readonly Opt0012ObservedCopyDescriptor[]): number {
  const fullLogitBytes = [196_608, 196_608, 196_608, 196_608, 82_384];
  const fullDestinations = [0, 196_608, 393_216, 589_824, 786_432];
  const expected =
    arm === "A"
      ? [
          ...fullLogitBytes.map((copiedBytes, index) => ({
            sourceBufferLabel: `logits-${index}`,
            sourceOffset: 0,
            destinationBufferLabel: "ace-planner-logit-readback",
            destinationOffset: fullDestinations[index]!,
            copiedBytes,
          })),
          {
            sourceBufferLabel: "write-status",
            sourceOffset: 0,
            destinationBufferLabel: "ace-planner-logit-readback",
            destinationOffset: OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET,
            copiedBytes: OPT_0012_FULL_READBACK_STATUS_BYTES,
          },
        ]
      : state === "regular-code"
        ? [
            {
              sourceBufferLabel: "opt-0012-regular-code-shard-3-output",
              sourceOffset: CANDIDATE_OUTPUT_GUARD_BYTES,
              destinationBufferLabel: "opt-0012-compact-head-readback",
              destinationOffset: 0,
              copiedBytes: 179_756,
            },
            {
              sourceBufferLabel: "opt-0012-regular-code-shard-4-output",
              sourceOffset: CANDIDATE_OUTPUT_GUARD_BYTES,
              destinationBufferLabel: "opt-0012-compact-head-readback",
              destinationOffset: 179_756,
              copiedBytes: 76_244,
            },
            {
              sourceBufferLabel: "write-status",
              sourceOffset: 0,
              destinationBufferLabel: "opt-0012-compact-head-readback",
              destinationOffset: 256_000,
              copiedBytes: 8,
            },
          ]
        : [
            {
              sourceBufferLabel: "opt-0012-forced-eos-shard-3-output",
              sourceOffset: CANDIDATE_OUTPUT_GUARD_BYTES,
              destinationBufferLabel: "opt-0012-compact-head-readback",
              destinationOffset: 0,
              copiedBytes: 4,
            },
            {
              sourceBufferLabel: "write-status",
              sourceOffset: 0,
              destinationBufferLabel: "opt-0012-compact-head-readback",
              destinationOffset: 4,
              copiedBytes: 8,
            },
          ];
  if (copies.length !== expected.length) {
    throw new Error("OPT-0012 observed readback copy count changed");
  }
  let copiedBytes = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const actual = copies[index]!;
    const wanted = expected[index]!;
    if (
      actual.sourceBufferLabel !== wanted.sourceBufferLabel ||
      actual.sourceOffset !== wanted.sourceOffset ||
      actual.destinationBufferLabel !== wanted.destinationBufferLabel ||
      actual.destinationOffset !== wanted.destinationOffset ||
      actual.copiedBytes !== wanted.copiedBytes
    ) {
      throw new Error(`OPT-0012 observed readback copy ${index} changed`);
    }
    copiedBytes += actual.copiedBytes;
  }
  return copiedBytes;
}

export function authenticateOpt0012ObservedHeadDispatches(
  arm: Opt0012Arm,
  state: AceOpt0012SemanticState,
  dispatches: readonly Opt0012ObservedHeadDispatchDescriptor[],
): void {
  const expected =
    arm === "A"
      ? ACE_PLANNER_EMBEDDING_ROW_PARTS.map((shard, shardIndex) => ({
          shardIndex,
          firstRow: shard.firstRow,
          columns: shard.rowCount,
          labelFragment: `-lm-head-rows-${shard.firstRow}-source-row-major-range-0-bindings`,
        }))
      : state === "regular-code"
        ? [
            { shardIndex: 3, firstRow: 151_669, columns: 44_939, labelFragment: "opt-0012-regular-code-shard-3-source-row-major-range-0-bindings" },
            { shardIndex: 4, firstRow: 196_608, columns: 19_061, labelFragment: "opt-0012-regular-code-shard-4-source-row-major-range-0-bindings" },
          ]
        : [{ shardIndex: 3, firstRow: ACE_QWEN_IM_END_TOKEN_ID, columns: 1, labelFragment: "opt-0012-forced-eos-shard-3-source-row-major-range-0-bindings" }];
  if (dispatches.length !== expected.length) {
    throw new Error("OPT-0012 observed tied-head shard count changed");
  }
  for (let index = 0; index < expected.length; index += 1) {
    const actual = dispatches[index]!;
    const shard = expected[index]!;
    const expectedPipeline = `ace-correctness-gemm-raw-fp16-source-row-major-2x1024x${shard.columns}-no-bias`;
    const expectedWorkgroups = Math.ceil(shard.columns / 128);
    if (
      !actual.bindGroupLabel.endsWith(shard.labelFragment) ||
      actual.pipelineLabel !== expectedPipeline ||
      actual.workgroups[0] !== expectedWorkgroups ||
      actual.workgroups[1] !== 1 ||
      actual.workgroups[2] !== 1
    ) {
      throw new Error(`OPT-0012 observed tied-head shard ${shard.shardIndex} identity/work changed`);
    }
  }
}

export interface Opt0012TrackedBufferCleanupRecord {
  readonly label: string;
  readonly destroyCallCount: number;
  readonly mapCallCount: number;
  readonly unmapCallCount: number;
  readonly destroyed: boolean;
  readonly mapped: boolean;
}

export function validateOpt0012TrackedBufferCleanup(
  records: readonly Opt0012TrackedBufferCleanupRecord[],
  tracking: Readonly<{
    destructionTrackingSupported: boolean;
    mapTrackingSupported: boolean;
  }>,
): Readonly<Record<string, unknown>> {
  if (!tracking.destructionTrackingSupported || !tracking.mapTrackingSupported) {
    throw new Error("OPT-0012 GPUBuffer lifecycle instrumentation is unsupported");
  }
  const offending = records.find(
    (record) => !record.destroyed || record.destroyCallCount !== 1 || record.mapped || record.mapCallCount !== record.unmapCallCount,
  );
  if (offending !== undefined) {
    throw new Error(
      `OPT-0012 tracked GPUBuffer cleanup is incomplete for ${offending.label || "<unlabelled>"}: ` +
        `destroyed=${offending.destroyed}, destroyCalls=${offending.destroyCallCount}, ` +
        `mapped=${offending.mapped}, maps=${offending.mapCallCount}, ` +
        `unmaps=${offending.unmapCallCount}`,
    );
  }
  return Object.freeze({
    trackedBufferCount: records.length,
    zeroLiveOwnedResources: true,
    everyTrackedBufferDestroyedExactlyOnce: true,
    everyMappedBufferUnmappedBeforeDestroy: true,
    everyMapBalancedByUnmap: true,
    firstOffendingLabel: null,
  });
}

/**
 * Authenticate the production composer split before suppressing either full
 * tied-head quantum. Decode retains 31 pre-head quanta; the authenticated
 * 253-token trajectory prefill has 143 because each layer expands to five.
 */
export function authenticateOpt0012ProductionHeadTail(
  dispatchLabel: string,
  quanta: readonly Opt0012HeadTailQuantumDescriptor[],
  phaseKind: "prefill" | "decode",
): number {
  const logicalId = `${dispatchLabel}-tied-lm-head`;
  const first = quanta.at(-2);
  const second = quanta.at(-1);
  const expected = opt0012ExpectedCommandBufferTopology("A", phaseKind);
  if (
    quanta.length !== expected.modelCommandBufferCount ||
    !dispatchLabel.includes(`-${phaseKind}-`) ||
    quanta.slice(0, -2).some((quantum) => quantum.kind === "tied-lm-head") ||
    first?.id !== `${logicalId}-part-0` ||
    first.logicalId !== logicalId ||
    first.kind !== "tied-lm-head" ||
    first.layer !== null ||
    first.primitiveCount !== 2 ||
    second?.id !== `${logicalId}-part-1` ||
    second.logicalId !== logicalId ||
    second.kind !== "tied-lm-head" ||
    second.layer !== null ||
    second.primitiveCount !== 3
  ) {
    throw new Error("OPT-0012 production two-quantum tied-head tail changed");
  }
  return expected.preHeadCommandBufferCount;
}

/** Six rounds; every arm occupies each position exactly twice. */
export const OPT_0012_BALANCED_ORDERS = Object.freeze([
  Object.freeze(["A", "B", "C"]),
  Object.freeze(["A", "C", "B"]),
  Object.freeze(["B", "A", "C"]),
  Object.freeze(["B", "C", "A"]),
  Object.freeze(["C", "A", "B"]),
  Object.freeze(["C", "B", "A"]),
] satisfies readonly (readonly Opt0012Arm[])[]);

type Opt0012ConversionArm = "legacy-allocating" | "allocation-free";
type Opt0012ReplayArm = "B" | "C";

/** Exactly one complete-domain pass in each order before package acquisition. */
export const OPT_0012_PREPACKAGE_CONVERSION_ORDERS = Object.freeze([
  Object.freeze(["legacy-allocating", "allocation-free"]),
  Object.freeze(["allocation-free", "legacy-allocating"]),
] satisfies readonly (readonly Opt0012ConversionArm[])[]);

/** Six paired rounds; each binary arm occupies each position three times. */
export const OPT_0012_CONVERSION_BALANCED_ORDERS = Object.freeze([
  Object.freeze(["legacy-allocating", "allocation-free"]),
  Object.freeze(["allocation-free", "legacy-allocating"]),
  Object.freeze(["allocation-free", "legacy-allocating"]),
  Object.freeze(["legacy-allocating", "allocation-free"]),
  Object.freeze(["legacy-allocating", "allocation-free"]),
  Object.freeze(["allocation-free", "legacy-allocating"]),
] satisfies readonly (readonly Opt0012ConversionArm[])[]);

/** Six common-byte replay rounds; B and C each occupy both positions thrice. */
export const OPT_0012_REPLAY_BALANCED_ORDERS = Object.freeze([
  Object.freeze(["B", "C"]),
  Object.freeze(["C", "B"]),
  Object.freeze(["C", "B"]),
  Object.freeze(["B", "C"]),
  Object.freeze(["B", "C"]),
  Object.freeze(["C", "B"]),
] satisfies readonly (readonly Opt0012ReplayArm[])[]);

interface Opt0012Fp16ConversionOutput {
  readonly floats: Float32Array;
  readonly words: Uint32Array;
}

interface Opt0012Fp16ConversionPass {
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly legacyWords: Uint32Array;
  readonly candidateWords: Uint32Array;
}

/** Two fixed target-browser passes under the stable synthetic-NaN envelope. */
export function runOpt0012Fp16ConversionCorrectnessGate(): Readonly<Record<string, unknown>> {
  const orders = OPT_0012_PREPACKAGE_CONVERSION_ORDERS;
  const passes = orders.map((order, passIndex) => runOpt0012Fp16ConversionPass(order, passIndex));
  const firstCandidate = passes[0]!.candidateWords;
  const secondCandidate = passes[1]!.candidateWords;
  let candidateCrossPassMismatchCount = 0;
  let firstCandidateCrossPassMismatchFp16U16: number | null = null;
  for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1) {
    if (firstCandidate[bits] !== secondCandidate[bits]) {
      candidateCrossPassMismatchCount += 1;
      firstCandidateCrossPassMismatchFp16U16 ??= bits;
    }
  }
  const candidateCrossPass = Object.freeze({
    comparedFp16U16Count: OPT_0012_FP16_DOMAIN_WORD_COUNT,
    mismatchCount: candidateCrossPassMismatchCount,
    firstMismatchFp16U16: firstCandidateCrossPassMismatchFp16U16,
    firstOutputU32LeSha256: sha256U32Words(firstCandidate),
    secondOutputU32LeSha256: sha256U32Words(secondCandidate),
    wordForWordIdentical: candidateCrossPassMismatchCount === 0,
  });
  const publicPasses = Object.freeze(passes.map((pass) => pass.receipt));
  const everyPassAccepted = publicPasses.every((pass) => pass["acceptedStableNanEnvelope"] === true);
  if (
    !everyPassAccepted ||
    candidateCrossPassMismatchCount !== 0 ||
    candidateCrossPass.firstOutputU32LeSha256 !== OPT_0012_FP16_DOMAIN_OUTPUT_SHA256 ||
    candidateCrossPass.secondOutputU32LeSha256 !== OPT_0012_FP16_DOMAIN_OUTPUT_SHA256
  ) {
    throw new Error(
      "OPT-0012 stable FP16 NaN-envelope gate failed: " +
        JSON.stringify({
          orders,
          passes: publicPasses,
          candidateCrossPass,
        }),
    );
  }
  return Object.freeze({
    authority: "target-browser-stable-FP16-NaN-envelope",
    legacyImplementation: "frozen-per-scalar-typed-array-allocation",
    candidateImplementation: "deterministic-quiet-all-allocation-free-U32",
    protocol: Object.freeze({
      orders,
      passCount: 2,
      exactlyOnePassPerOrder: true,
      noAdaptiveRepetition: true,
      completedBeforePackageAcquisition: true,
    }),
    expected: Object.freeze({
      completeDomainWordCount: OPT_0012_FP16_DOMAIN_WORD_COUNT,
      nonNaNWordCount: OPT_0012_FP16_NON_NAN_WORD_COUNT,
      nanWordCount: OPT_0012_FP16_NAN_WORD_COUNT,
      candidateU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
      nonNaNU32LeSha256: OPT_0012_FP16_NON_NAN_OUTPUT_SHA256,
      canonicalNaNU32LeSha256: OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256,
    }),
    passes: publicPasses,
    candidateCrossPass,
    acceptedStableNanEnvelope: true,
  });
}

function runOpt0012Fp16ConversionPass(order: readonly Opt0012ConversionArm[], passIndex: number): Opt0012Fp16ConversionPass {
  const outputs = new Map<Opt0012ConversionArm, Opt0012Fp16ConversionOutput>();
  for (const arm of ["legacy-allocating", "allocation-free"] as const) {
    const floats = new Float32Array(OPT_0012_FP16_DOMAIN_WORD_COUNT);
    outputs.set(
      arm,
      Object.freeze({
        floats,
        words: new Uint32Array(floats.buffer),
      }),
    );
  }
  for (const arm of order) {
    const output = outputs.get(arm)!;
    writeOpt0012Fp16ConversionDomain(arm, output.floats, output.words);
  }
  const legacyWords = outputs.get("legacy-allocating")!.words;
  const candidateWords = outputs.get("allocation-free")!.words;
  const legacy = analyzeOpt0012Fp16ConversionEnvelope("legacy-allocating", legacyWords);
  const candidate = analyzeOpt0012Fp16ConversionEnvelope("allocation-free", candidateWords);
  let rawDifferenceCount = 0;
  let firstRawDifference: Readonly<Record<string, number>> | null = null;
  for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1) {
    if (legacyWords[bits] !== candidateWords[bits]) {
      rawDifferenceCount += 1;
      firstRawDifference ??= Object.freeze({
        fp16U16: bits,
        legacyFp32U32: legacyWords[bits]!,
        candidateFp32U32: candidateWords[bits]!,
      });
    }
  }
  const acceptedStableNanEnvelope = legacy.acceptedStableNanEnvelope === true && candidate.acceptedStableNanEnvelope === true;
  return Object.freeze({
    receipt: Object.freeze({
      passIndex,
      order: Object.freeze([...order]),
      completeDomainPerArm: true,
      legacy,
      candidate,
      rawComparison: Object.freeze({
        rawDifferenceCount,
        firstRawDifference,
        legacyRawHashIsDiagnosticOnly: true,
      }),
      acceptedStableNanEnvelope,
    }),
    legacyWords,
    candidateWords,
  });
}

function analyzeOpt0012Fp16ConversionEnvelope(arm: Opt0012ConversionArm, words: Uint32Array): Readonly<Record<string, unknown>> {
  if (words.length !== OPT_0012_FP16_DOMAIN_WORD_COUNT) {
    throw new Error("OPT-0012 FP16 envelope received an incomplete domain");
  }
  const canonicalWords = new Uint32Array(words.length);
  const nonNaNWords = new Uint32Array(OPT_0012_FP16_NON_NAN_WORD_COUNT);
  const canonicalNaNWords = new Uint32Array(OPT_0012_FP16_NAN_WORD_COUNT);
  const inputClasses = {
    zero: 0,
    subnormal: 0,
    normal: 0,
    infinity: 0,
    quietNaN: 0,
    signalingNaN: 0,
  };
  const nanOutputClasses = {
    positiveSignalingInput: { signaling: 0, quiet: 0 },
    positiveQuietInput: { signaling: 0, quiet: 0 },
    negativeSignalingInput: { signaling: 0, quiet: 0 },
    negativeQuietInput: { signaling: 0, quiet: 0 },
  };
  let nonNaNCursor = 0;
  let nanCursor = 0;
  let nonNaNMismatchCount = 0;
  let nanClassificationMismatchCount = 0;
  let signMismatchCount = 0;
  let payloadExcludingQuietMismatchCount = 0;
  let disallowedWordMismatchCount = 0;
  let firstDisallowedFp16U16: number | null = null;
  for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1) {
    const actual = words[bits]!;
    const rawExpected = expandOpt0012Fp16ToRawFp32U32(bits);
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x03ff;
    const nanInput = exponent === 0x1f && mantissa !== 0;
    if (!nanInput) {
      if (exponent === 0) {
        inputClasses[mantissa === 0 ? "zero" : "subnormal"] += 1;
      } else if (exponent === 0x1f) {
        inputClasses.infinity += 1;
      } else {
        inputClasses.normal += 1;
      }
      nonNaNWords[nonNaNCursor] = actual;
      nonNaNCursor += 1;
      if (actual !== rawExpected) {
        nonNaNMismatchCount += 1;
        disallowedWordMismatchCount += 1;
        firstDisallowedFp16U16 ??= bits;
      }
      canonicalWords[bits] = actual;
      continue;
    }
    const inputQuiet = (mantissa & 0x0200) !== 0;
    inputClasses[inputQuiet ? "quietNaN" : "signalingNaN"] += 1;
    const canonicalExpected = (rawExpected | 0x0040_0000) >>> 0;
    const actualIsNaN = (actual & 0x7f80_0000) === 0x7f80_0000 && (actual & 0x007f_ffff) !== 0;
    const actualQuiet = (actual & 0x0040_0000) !== 0;
    const positive = (bits & 0x8000) === 0;
    const classCounts = positive
      ? inputQuiet
        ? nanOutputClasses.positiveQuietInput
        : nanOutputClasses.positiveSignalingInput
      : inputQuiet
        ? nanOutputClasses.negativeQuietInput
        : nanOutputClasses.negativeSignalingInput;
    classCounts[actualQuiet ? "quiet" : "signaling"] += 1;
    if (!actualIsNaN) nanClassificationMismatchCount += 1;
    if (((actual ^ rawExpected) & 0x8000_0000) !== 0) signMismatchCount += 1;
    if (((actual ^ rawExpected) & 0x003f_ffff) !== 0) {
      payloadExcludingQuietMismatchCount += 1;
    }
    if (actual !== rawExpected && actual !== canonicalExpected) {
      disallowedWordMismatchCount += 1;
      firstDisallowedFp16U16 ??= bits;
    }
    canonicalWords[bits] = actual | 0x0040_0000;
    canonicalNaNWords[nanCursor] = actual | 0x0040_0000;
    nanCursor += 1;
  }
  const rawOutputU32LeSha256 = sha256U32Words(words);
  const canonicalOutputU32LeSha256 = sha256U32Words(canonicalWords);
  const nonNaNOutputU32LeSha256 = sha256U32Words(nonNaNWords);
  const canonicalNaNOutputU32LeSha256 = sha256U32Words(canonicalNaNWords);
  const signalingOutputCount =
    nanOutputClasses.positiveSignalingInput.signaling +
    nanOutputClasses.positiveQuietInput.signaling +
    nanOutputClasses.negativeSignalingInput.signaling +
    nanOutputClasses.negativeQuietInput.signaling;
  const quietOutputCount = OPT_0012_FP16_NAN_WORD_COUNT - signalingOutputCount;
  const expectedInputClasses = JSON.stringify({
    zero: 2,
    subnormal: 2_046,
    normal: 61_440,
    infinity: 2,
    quietNaN: 1_024,
    signalingNaN: 1_022,
  });
  const commonAccepted =
    nonNaNCursor === OPT_0012_FP16_NON_NAN_WORD_COUNT &&
    nanCursor === OPT_0012_FP16_NAN_WORD_COUNT &&
    JSON.stringify(inputClasses) === expectedInputClasses &&
    nonNaNMismatchCount === 0 &&
    nanClassificationMismatchCount === 0 &&
    signMismatchCount === 0 &&
    payloadExcludingQuietMismatchCount === 0 &&
    disallowedWordMismatchCount === 0 &&
    nonNaNOutputU32LeSha256 === OPT_0012_FP16_NON_NAN_OUTPUT_SHA256 &&
    canonicalOutputU32LeSha256 === OPT_0012_FP16_DOMAIN_OUTPUT_SHA256 &&
    canonicalNaNOutputU32LeSha256 === OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256;
  const acceptedStableNanEnvelope =
    commonAccepted &&
    (arm === "legacy-allocating" ||
      (rawOutputU32LeSha256 === OPT_0012_FP16_DOMAIN_OUTPUT_SHA256 && signalingOutputCount === 0 && quietOutputCount === OPT_0012_FP16_NAN_WORD_COUNT));
  return Object.freeze({
    arm,
    rawOutputU32LeSha256,
    canonicalOutputU32LeSha256,
    nonNaNOutputU32LeSha256,
    canonicalNaNOutputU32LeSha256,
    inputClasses: Object.freeze(inputClasses),
    nanOutputClassesByInputClassAndSign: Object.freeze({
      positiveSignalingInput: Object.freeze(nanOutputClasses.positiveSignalingInput),
      positiveQuietInput: Object.freeze(nanOutputClasses.positiveQuietInput),
      negativeSignalingInput: Object.freeze(nanOutputClasses.negativeSignalingInput),
      negativeQuietInput: Object.freeze(nanOutputClasses.negativeQuietInput),
    }),
    signalingOutputCount,
    quietOutputCount,
    nonNaNMismatchCount,
    nanClassificationMismatchCount,
    signMismatchCount,
    payloadExcludingQuietMismatchCount,
    disallowedWordMismatchCount,
    firstDisallowedFp16U16,
    legacyRawHashIsDiagnosticOnly: arm === "legacy-allocating",
    everyNonNaNExact: nonNaNMismatchCount === 0,
    everyNaNAllowedRawOrQuietBitOnly:
      nanClassificationMismatchCount === 0 && signMismatchCount === 0 && payloadExcludingQuietMismatchCount === 0 && disallowedWordMismatchCount === 0,
    candidateDeterministicQuietAll: arm === "allocation-free" && signalingOutputCount === 0 && quietOutputCount === OPT_0012_FP16_NAN_WORD_COUNT,
    acceptedStableNanEnvelope,
  });
}

function expandOpt0012Fp16ToRawFp32U32(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign >>> 0;
    exponent = 1;
    while ((mantissa & 0x0400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x03ff;
    return (sign | ((exponent + 112) << 23) | (mantissa << 13)) >>> 0;
  }
  if (exponent === 0x1f) {
    return (sign | 0x7f80_0000 | (mantissa << 13)) >>> 0;
  }
  return (sign | ((exponent + 112) << 23) | (mantissa << 13)) >>> 0;
}

function writeOpt0012Fp16ConversionDomain(arm: Opt0012ConversionArm, floats: Float32Array, words: Uint32Array): void {
  if (floats.length !== OPT_0012_FP16_DOMAIN_WORD_COUNT || words.length !== OPT_0012_FP16_DOMAIN_WORD_COUNT || floats.buffer !== words.buffer) {
    throw new Error("OPT-0012 FP16 conversion domain output changed");
  }
  if (arm === "legacy-allocating") {
    for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1) {
      floats[bits] = legacyAllocatingAceOpt0012Float16BitsToNumber(bits);
    }
    return;
  }
  for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1) {
    words[bits] = aceOpt0012Float16BitsToFloat32U32(bits);
  }
}

/** Exact pre-correction allocator retained only as a harness-local oracle. */
function legacyAllocatingAceOpt0012Float16BitsToNumber(bits: number): number {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) {
    throw new RangeError("OPT-0012 FP16 bits must be an unsigned 16-bit integer");
  }
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let output: number;
  if (exponent === 0) {
    if (mantissa === 0) {
      output = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      output = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    output = sign | 0x7f80_0000 | (mantissa << 13);
  } else {
    output = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  const words = new Uint32Array([output >>> 0]);
  return new Float32Array(words.buffer)[0]!;
}

export const OPT_0012_CASE_SPECS = Object.freeze([
  Object.freeze({
    id: "semantic-m2-short",
    position: "short" as const,
    cachedTokensBeforeAppend: 268,
    cacheCapacity: 768,
    drawIndex: 125,
  }),
  Object.freeze({
    id: "semantic-m2-mid",
    position: "mid" as const,
    cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280,
    drawIndex: 185,
  }),
  Object.freeze({
    id: "semantic-m2-long",
    position: "long" as const,
    cachedTokensBeforeAppend: 401,
    cacheCapacity: 2_048,
    drawIndex: 258,
  }),
]);

export const OPT_0012_EOS_CASE_SPEC = Object.freeze({
  id: "semantic-m2-terminal-eos",
  position: "long" as const,
  cachedTokensBeforeAppend: 402,
  cacheCapacity: 2_048,
  drawIndex: 258,
});

export const OPT_0012_OPT10_SAMPLE_AUTHORITIES = Object.freeze({
  "semantic-m2-short": Object.freeze({
    tokenId: 192_370,
    word: 2_004_582_350,
    positiveCandidateCount: 16,
    drawIndex: "125",
    drawEnd: "126",
  }),
  "semantic-m2-mid": Object.freeze({
    tokenId: 156_326,
    word: 503_673_048,
    positiveCandidateCount: 1,
    drawIndex: "185",
    drawEnd: "186",
  }),
  "semantic-m2-long": Object.freeze({
    tokenId: 155_832,
    word: 3_288_166_745,
    positiveCandidateCount: 1,
    drawIndex: "258",
    drawEnd: "259",
  }),
} as const);

export const OPT_0012_ACCEPTED_SEMANTIC_CODE_IDS = Object.freeze([
  44_244, 2_430, 7_541, 38_339, 43_500, 14_023, 63_719, 16_071, 63_855, 58_755, 37_828, 20_336, 52_689, 20_856, 53_201, 11_240, 17_873, 15_217, 22_037, 12_976,
  20_848, 47_248, 54_461, 28_656, 28_812, 12_669, 10_110, 63_939, 48_579, 19_199, 63_718, 28_871, 51_055, 45_955, 27_133, 22_832, 53_202, 20_920, 53_200, 9_720,
  22_992, 17_785, 15_536, 12_797, 12_715, 40_782, 28_800, 41_083, 28_411, 11_130, 21_929, 29_754, 10_008, 33_298, 35_986, 25_049, 7_613, 57_560, 45_964, 14_910,
  34_822, 18_567, 53_535, 28_710, 7_230, 25_184, 7_480, 25_456, 25_572, 9_702, 61_421, 62_781, 12_723, 63_314, 13_779, 57_352, 2_705, 51_234, 61_459, 50_923,
  28_968, 15_224, 19_416, 26_752, 26_688, 26_752, 39_560, 42_256, 57_104, 57_176, 56_728, 56_800, 5_544, 8_112, 23_480, 24_568, 28_820, 12_669, 10_101, 63_426,
  58_828, 16_583, 18_435, 16_709, 53_300, 20_421, 62_950, 36_528, 53_202, 10_236, 53_196, 37_950, 57_820, 33_075, 31_276, 10_168, 10_171, 40_847, 26_434,
  28_159, 41_684, 12_669, 10_101, 63_426, 58_860, 16_583, 38_467, 903, 53_357, 31_175, 62_934, 35_992, 22_938, 23_158, 1_621, 23_534, 2_501, 24_576, 18_421,
  12_279, 25_460, 51_159, 25_986, 50_117, 33_807, 35_847, 35_847, 35_847, 35_847, 35_847,
]);

export const OPT_0012_CANDIDATE_SHADER_SHA256 = Object.freeze({
  "regular-shard-3": "b780701ae4d0015ad0b78be146d4fedd0e161b1cc44f95527aae30a1945a0d41",
  "regular-shard-4": "a5c23e10b3b5bf69c4632f61fc92ada7e7852fa822bf5faf40115581ed40b4ce",
  "forced-eos-shard-3": "c824f31396311a52ff48005b3124bc8d05564060b584bc113758c7f77071437d",
} as const);

export const OPT_0012_SOURCE_IDENTITIES = Object.freeze({
  ...ACE_OPT_0012_CORE_SOURCE_IDENTITIES,
  "benchmark/opt-0012-compact-semantic-head.ts": OPT_0012_CORE_SOURCE_SHA256,
  "src/api.ts": "d5768d9059a235ea483268a73a4a37437d4203b5bd6971a0fba1b8876abe72c7",
  "src/model/graph-contract.ts": "ad6de30e8c089f1b0d1dd86951518150904bf1761a7285b2f5f1a2e4304aa116",
  "src/model/strict-json.ts": "1f431ca4db4cc919308769a85acdf1ce8d1febde4af9fb72dd87540b64c51fba",
  "src/model/acquire.ts": "4dae4fee7497ce77640819936388e44c2eca14757e1646f5ad85493542169c78",
  "src/model/cache.ts": "4b7599bfacc99fe2451fa7cb902d1ad7046100aeb3e82ef3995034b8e3b41a25",
  "src/model/gpu-tensors.ts": "7bf1b358e1b3fd15873457974cd26c2c04f96d988a1c0cc099346a8b50c77797",
  "src/model/gpu-upload.ts": "061ebb1e6f078b6b56a21f67afadcaf60d2d051790f667d481299c4e10abb319",
  "src/model/package.ts": "78a616a9ca6561ae8f6248ac222608bacf5c5412ca30e063fc4562857f04c6ef",
  "src/model/sha256.ts": "9c14ea19291acb490cf3d5a884881700d873400ca2f2bb4aa4bf7efa8b7138d1",
  "src/runtime/scheduler.ts": "a3c319b7bf7f8d4141c1a0814072d1b9f2a9b50a54b54ef82290a40ed49698c7",
  "src/tokenizer/index.ts": "18a9815b18f79248933087bf8c04b6f3ae653fae80fbb0e58004ce9ae38a7419",
  "src/tokenizer/loader.ts": "70ddb6303b5751496c9c4ce3f77e30d37cbdf36e6bd0ccada565942c23d45930",
  "src/tokenizer/chat.ts": "b0a3244f8291e56c7ae381b0ea293b274bed1c25de74fff4b9446b584acbc852",
  "src/tokenizer/conditioning-text.ts": "f589799f856ef962226c16585af4ec90434867c5857fe44b1601087d64bc9095",
  "src/webgpu/arena.ts": "11ae7e88b9b6f2ca1529e265c6e20e2576684c99396d25ee2d76d361cd0cd093",
  "src/webgpu/capabilities.ts": "b9387c2fd094c5ad77c243abb34ff441bfc4bf3e47fe7fb218d61b6c703cec31",
  "src/webgpu/device.ts": "16536f25472890650a6f58603eab359c779c0dcba4b1d7e7cd82876a464e3972",
  "src/webgpu/planner-executor.ts": "f802db8c35701278619c0ad5c13e7543a224662d561d1db1d6e3a71271032786",
  "src/webgpu/kernels/attention.ts": "824d86bca3cdd4be4de425a482eb516ac8fa318148d1324a750f7b7c8e8035b2",
  "src/webgpu/kernels/batched-rope.ts": "05301c6d020ab720c004b6a85fad76aaee725ac96df3fba6ff8c1ac57d81377b",
  "src/webgpu/kernels/embedding.ts": "049750864bd389978c6b90575b7f91e18fb046a884d5092db66b06527fe161d4",
  "src/webgpu/kernels/kv-cache.ts": "a08e5c94396771a22ebb92f90124fec71721792c31c25751dcdc57bfdabd3a79",
  "src/webgpu/kernels/rmsnorm.ts": "f49ace6e346c9e13ea48be9632e505a42f678b8e9e7427cd2a1f1e9ded525b5d",
  "src/webgpu/kernels/tensor-copy.ts": "b00963d03be8c4d29e6c2a0f504efa513b841e8641e193a8e43a43df5b7ac193",
  "src/webgpu/kernels/transformer-plumbing.ts": "baeb6a3931b1593d6b3a9396e27ea527a4f2083cb91571410d7ebac12053e92f",
  "src/webgpu/scoped-buffer-allocation.ts": "599cab5d15b441c516c75d2b4c632e35116e2b4c3c2b115415874c8119c8b7c0",
});

const OPT_0012_SOURCE_TEXT = Object.freeze({
  "benchmark/opt-0012-compact-semantic-head.ts": opt0012CoreSource,
  "src/api.ts": apiSource,
  "src/model/graph-contract.ts": graphContractSource,
  "src/model/strict-json.ts": strictJsonSource,
  "src/model/manifest.ts": manifestSource,
  "src/runtime/planner.ts": plannerSource,
  "src/runtime/planner-sampling.ts": plannerSamplingSource,
  "src/runtime/seed.ts": seedSource,
  "src/tokenizer/qwen-bpe.ts": qwenBpeSource,
  "src/tokenizer/index.ts": tokenizerIndexSource,
  "src/tokenizer/loader.ts": tokenizerLoaderSource,
  "src/tokenizer/chat.ts": tokenizerChatSource,
  "src/tokenizer/conditioning-text.ts": tokenizerConditioningTextSource,
  "src/webgpu/planner-model.ts": plannerModelSource,
  "src/webgpu/capabilities.ts": capabilitiesSource,
  "src/webgpu/kernels/attention.ts": attentionSource,
  "src/webgpu/kernels/batched-rope.ts": batchedRopeSource,
  "src/webgpu/kernels/embedding.ts": embeddingSource,
  "src/webgpu/kernels/gemm.ts": gemmSource,
  "src/webgpu/kernels/kv-cache.ts": kvCacheSource,
  "src/webgpu/kernels/rmsnorm.ts": rmsnormSource,
  "src/webgpu/kernels/tensor-copy.ts": tensorCopySource,
  "src/webgpu/kernels/transformer-plumbing.ts": transformerPlumbingSource,
  "src/webgpu/kernels/correctness-utils.ts": correctnessUtilsSource,
  "src/webgpu/qwen3.ts": qwen3Source,
  "src/model/acquire.ts": acquireSource,
  "src/model/cache.ts": cacheSource,
  "src/model/gpu-tensors.ts": gpuTensorsSource,
  "src/model/gpu-upload.ts": gpuUploadSource,
  "src/model/package.ts": packageSource,
  "src/model/sha256.ts": sha256Source,
  "src/runtime/scheduler.ts": schedulerSource,
  "src/webgpu/arena.ts": arenaSource,
  "src/webgpu/device.ts": deviceSource,
  "src/webgpu/planner-executor.ts": plannerExecutorSource,
  "src/webgpu/scoped-buffer-allocation.ts": scopedBufferSource,
});

export interface Opt0012RunIdentity {
  readonly harnessCommit: string;
  readonly coreCommit: typeof OPT_0012_CORE_COMMIT;
  readonly allocationCommit: typeof OPT_0012_ALLOCATION_COMMIT;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0012ThermalGateMetadata {
  readonly source: typeof OPT_0012_THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0012_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

export function parseOpt0012RunIdentity(parameters: URLSearchParams): Opt0012RunIdentity {
  const harnessCommit = requiredIdentityString(parameters, "harnessCommit");
  const coreCommit = requiredIdentityString(parameters, "coreCommit");
  const allocationCommit = requiredIdentityString(parameters, "allocationCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0012 harnessCommit must be a 40-character hex commit");
  }
  if (coreCommit !== OPT_0012_CORE_COMMIT) {
    throw new Error("OPT-0012 coreCommit does not match the frozen benchmark core");
  }
  if (allocationCommit !== OPT_0012_ALLOCATION_COMMIT) {
    throw new Error("OPT-0012 allocationCommit does not match the experiment");
  }
  return Object.freeze({
    harnessCommit,
    coreCommit: OPT_0012_CORE_COMMIT,
    allocationCommit: OPT_0012_ALLOCATION_COMMIT,
    machineModel: requiredIdentityString(parameters, "machineModel"),
    osVersion: requiredIdentityString(parameters, "osVersion"),
    osBuild: requiredIdentityString(parameters, "osBuild"),
    browserVersion: requiredIdentityString(parameters, "browserVersion"),
    gpuCoreCount: requiredIdentityInteger(parameters, "gpuCoreCount"),
    memoryBytes: requiredIdentityInteger(parameters, "memoryBytes"),
  });
}

export function parseOpt0012ThermalGateMetadata(
  parameters: URLSearchParams,
  warmupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0012ThermalGateMetadata {
  const source = parameters.get("thermalSource");
  const startedAtEpochMilliseconds = requiredFiniteNumber(parameters, "thermalStartedAtEpochMilliseconds");
  const completedAtEpochMilliseconds = requiredFiniteNumber(parameters, "thermalCompletedAtEpochMilliseconds");
  const observationCount = requiredFiniteNumber(parameters, "thermalObservations");
  const pollMilliseconds = requiredFiniteNumber(parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredFiniteNumber(parameters, "thermalMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredFiniteNumber(parameters, "thermalNonNominalObservations");
  if (source !== OPT_0012_THERMAL_SOURCE) {
    throw new Error("OPT-0012 requires the accepted notifyutil thermal source");
  }
  if (
    startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error("OPT-0012 thermal gate must begin after correctness/warmup and be current");
  }
  const durationMilliseconds = completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  if (
    durationMilliseconds < OPT_0012_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(observationCount) ||
    observationCount < Math.floor(durationMilliseconds / OPT_0012_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0012 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== OPT_0012_THERMAL_POLL_MILLISECONDS) {
    throw new Error("OPT-0012 thermal polling must use 1,000 ms intervals");
  }
  if (maximumPollGapMilliseconds < 0 || maximumPollGapMilliseconds > OPT_0012_THERMAL_POLL_MILLISECONDS + OPT_0012_THERMAL_POLL_TOLERANCE_MILLISECONDS) {
    throw new Error("OPT-0012 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0012 thermal gate observed non-nominal pressure");
  }
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0012_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

interface HeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0012RunIdentity;
}

interface RunTimedMessage {
  readonly type: "run-timed";
  readonly thermal: Opt0012ThermalGateMetadata;
}

interface RunTrajectoryMessage {
  readonly type: "run-trajectory";
}

type IncomingMessage = InitializeMessage | RunTimedMessage | RunTrajectoryMessage;

type WorkerMessage =
  | Readonly<{
      readonly type: "ready-for-thermal-gate";
      readonly warmupCompletedAtEpochMilliseconds: number;
      readonly preparation: unknown;
    }>
  | Readonly<{ readonly type: "progress"; readonly message: string }>
  | Readonly<{
      readonly type: "passed";
      readonly result: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{ readonly type: "failed"; readonly error: unknown }>;

if (typeof document !== "undefined") initializePage();
if (typeof document === "undefined" && typeof self !== "undefined" && "WorkerGlobalScope" in globalThis && self instanceof WorkerGlobalScope)
  initializeWorker();

function initializePage(): void {
  const prepare = requireElement<HTMLButtonElement>("#prepare");
  const runTimed = requireElement<HTMLButtonElement>("#run-timed");
  const runTrajectory = requireElement<HTMLButtonElement>("#run-trajectory");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let identity: Opt0012RunIdentity;
  try {
    identity = parseOpt0012RunIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishPageFailure(error);
    return;
  }
  const worker = new Worker(new URL("./opt-0012-compact-semantic-head-ab.ts", import.meta.url), { type: "module", name: "ace-opt-0012-worker" });
  let heartbeat: ReturnType<typeof startPageHeartbeat> | undefined;
  let warmupCompletedAtEpochMilliseconds: number | undefined;
  let enableTimer: number | undefined;
  let settled = false;

  populateThermalInputs(new URL(location.href).searchParams);
  prepare.addEventListener(
    "click",
    () => {
      prepare.disabled = true;
      document.body.dataset.status = "preparing";
      updatePageProgress("authenticating package/source identities and running A/B/C correctness");
      heartbeat = startPageHeartbeat();
      worker.postMessage({ type: "initialize", identity });
    },
    { once: true },
  );

  runTimed.addEventListener(
    "click",
    () => {
      try {
        if (warmupCompletedAtEpochMilliseconds === undefined) {
          throw new Error("OPT-0012 correctness/warmup has not completed");
        }
        setBlankThermalCompletionToNow();
        const thermal = parseOpt0012ThermalGateMetadata(thermalParameters(), warmupCompletedAtEpochMilliseconds, Date.now());
        runTimed.disabled = true;
        runTrajectory.disabled = true;
        thermalGate.disabled = true;
        document.body.dataset.status = "running";
        updatePageProgress("running balanced A/B/C short/mid/long timing");
        worker.postMessage({ type: "run-timed", thermal });
      } catch (error) {
        settled = true;
        worker.terminate();
        finishPageFailure(error, heartbeat?.stop());
      }
    },
    { once: true },
  );

  runTrajectory.addEventListener(
    "click",
    () => {
      if (warmupCompletedAtEpochMilliseconds === undefined) {
        finishPageFailure(new Error("OPT-0012 correctness/warmup has not completed"));
        return;
      }
      runTimed.disabled = true;
      runTrajectory.disabled = true;
      thermalGate.disabled = true;
      document.body.dataset.status = "running";
      updatePageProgress("running the raw-FP16 six-run 150-code plus forced-EOS equality gate");
      worker.postMessage({ type: "run-trajectory" });
    },
    { once: true },
  );

  worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    if (settled) return;
    const message = event.data;
    if (message.type === "progress") {
      updatePageProgress(message.message);
      return;
    }
    if (message.type === "ready-for-thermal-gate") {
      warmupCompletedAtEpochMilliseconds = message.warmupCompletedAtEpochMilliseconds;
      thermalGate.disabled = false;
      runTrajectory.disabled = false;
      setBlankThermalStartToNow();
      const eligibleAt = Date.now() + OPT_0012_MINIMUM_NOMINAL_MILLISECONDS;
      updatePageProgress("correctness and symmetric warmups passed; begin/reset the external nominal trace");
      enableTimer = window.setInterval(() => {
        const remaining = Math.max(0, eligibleAt - Date.now());
        if (remaining > 0) {
          updatePageProgress(`external nominal pre-gate: ${(remaining / 1_000).toFixed(1)} s remaining`);
          return;
        }
        if (enableTimer !== undefined) clearInterval(enableTimer);
        runTimed.disabled = false;
        updatePageProgress("enter the external trace metadata, run timing, or run trajectory-only");
      }, 250);
      requireElement<HTMLElement>("#preparation").textContent = JSON.stringify(message.preparation, null, 2);
      return;
    }
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    const pageHeartbeat = heartbeat?.stop();
    worker.terminate();
    if (message.type === "passed") {
      document.body.dataset.status = "passed";
      updatePageProgress("passed; keep the external logger running through its final poll");
      publishPageResult(message.result, pageHeartbeat);
    } else {
      finishPageFailure(message.error, pageHeartbeat);
    }
  });

  worker.addEventListener("error", (event) => {
    if (settled) return;
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    finishPageFailure(event.error ?? event.message, heartbeat?.stop());
    worker.terminate();
  });
}

const OPT_0012_RAW_RESULT_GLOBAL = "__ACE_OPT_0012_RAW_RESULT_JSON__";
export const OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS = 100_000;

export function parseOpt0012RawResultChunkOffset(value: string): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error("OPT-0012 raw-result chunk offset is not canonical decimal");
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error("OPT-0012 raw-result chunk offset is not a safe integer");
  }
  return offset;
}

export function sliceOpt0012RawResultChunk(
  rawResultJson: string,
  offset: number,
): Readonly<{
  readonly chunk: string;
  readonly start: number;
  readonly end: number;
  readonly nextOffset: number;
  readonly totalCodeUnits: number;
  readonly complete: boolean;
}> {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > rawResultJson.length) {
    throw new Error("OPT-0012 raw-result chunk offset is invalid");
  }
  if (
    offset > 0 &&
    offset < rawResultJson.length &&
    rawResultJson.charCodeAt(offset - 1) >= 0xd800 &&
    rawResultJson.charCodeAt(offset - 1) <= 0xdbff &&
    rawResultJson.charCodeAt(offset) >= 0xdc00 &&
    rawResultJson.charCodeAt(offset) <= 0xdfff
  ) {
    throw new Error("OPT-0012 raw-result chunk offset splits a surrogate pair");
  }
  let end = Math.min(offset + OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS, rawResultJson.length);
  if (
    end < rawResultJson.length &&
    end > offset &&
    rawResultJson.charCodeAt(end - 1) >= 0xd800 &&
    rawResultJson.charCodeAt(end - 1) <= 0xdbff &&
    rawResultJson.charCodeAt(end) >= 0xdc00 &&
    rawResultJson.charCodeAt(end) <= 0xdfff
  ) {
    end -= 1;
  }
  return Object.freeze({
    chunk: rawResultJson.slice(offset, end),
    start: offset,
    end,
    nextOffset: end,
    totalCodeUnits: rawResultJson.length,
    complete: end === rawResultJson.length,
  });
}

function publishPageResult(result: Readonly<Record<string, unknown>>, pageHeartbeat: HeartbeatSnapshot | undefined): void {
  const rawResultJson = JSON.stringify({
    ...result,
    pageHeartbeat,
  });
  if (
    !Reflect.defineProperty(globalThis, OPT_0012_RAW_RESULT_GLOBAL, {
      value: rawResultJson,
      configurable: false,
      enumerable: false,
      writable: false,
    })
  ) {
    throw new Error("OPT-0012 could not publish the raw result receipt");
  }
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    {
      experiment: result["experiment"],
      mode: result["mode"],
      status: "passed",
      rawResultJsonCodeUnitLength: rawResultJson.length,
      rawResultRetrieval: "bounded-dom-chunks",
      rawResultMainWorldGlobal: OPT_0012_RAW_RESULT_GLOBAL,
      rawResultChunkCodeUnitLimit: OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS,
      fullReceiptIntentionallyKeptOutOfDom: true,
    },
    null,
    2,
  );
  enableRawResultChunkRetrieval();
}

function enableRawResultChunkRetrieval(): void {
  const retrieval = requireElement<HTMLFieldSetElement>("#raw-result-retrieval");
  const offsetInput = requireElement<HTMLInputElement>('input[name="rawResultOffset"]');
  const publish = requireElement<HTMLButtonElement>("#publish-raw-result-chunk");
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  offsetInput.value = "0";
  output.textContent = "";
  output.dataset.state = "empty";
  retrieval.hidden = false;
  retrieval.disabled = false;
  publish.disabled = false;
  let publicationSequence = 0;
  publish.addEventListener("click", () => {
    output.textContent = "";
    output.dataset.state = "publishing";
    output.dataset.publicationSequence = String(++publicationSequence);
    delete output.dataset.startOffset;
    delete output.dataset.endOffsetExclusive;
    delete output.dataset.chunkCodeUnitLength;
    delete output.dataset.totalCodeUnitLength;
    delete output.dataset.done;
    try {
      const rawResultJson = Reflect.get(globalThis, OPT_0012_RAW_RESULT_GLOBAL);
      if (typeof rawResultJson !== "string") {
        throw new Error("OPT-0012 raw-result main-world receipt is unavailable");
      }
      const slice = sliceOpt0012RawResultChunk(rawResultJson, parseOpt0012RawResultChunkOffset(offsetInput.value));
      output.textContent = slice.chunk;
      output.dataset.startOffset = String(slice.start);
      output.dataset.endOffsetExclusive = String(slice.end);
      output.dataset.chunkCodeUnitLength = String(slice.chunk.length);
      output.dataset.totalCodeUnitLength = String(slice.totalCodeUnits);
      output.dataset.done = String(slice.complete);
      output.dataset.state = "published";
      offsetInput.value = String(slice.nextOffset);
    } catch (error) {
      output.dataset.state = "failed";
      output.textContent = JSON.stringify(serializeError(error));
      publish.disabled = true;
    }
  });
}

function startPageHeartbeat(): { stop(): HeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let stopped = false;
  let animationHandle = 0;
  const animation = (now: number): void => {
    maximumAnimationFrameGapMilliseconds = Math.max(maximumAnimationFrameGapMilliseconds, now - lastAnimationFrame);
    lastAnimationFrame = now;
    animationFrameCount += 1;
    if (!stopped) animationHandle = requestAnimationFrame(animation);
  };
  animationHandle = requestAnimationFrame(animation);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, now - lastTimer);
    lastTimer = now;
    timerTickCount += 1;
  }, HEARTBEAT_INTERVAL_MILLISECONDS);
  return {
    stop(): HeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        const terminalNow = performance.now();
        maximumAnimationFrameGapMilliseconds = Math.max(maximumAnimationFrameGapMilliseconds, terminalNow - lastAnimationFrame);
        maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, terminalNow - lastTimer);
        cancelAnimationFrame(animationHandle);
        clearInterval(timerHandle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

function populateThermalInputs(parameters: URLSearchParams): void {
  for (const name of [
    "thermalStartedAtEpochMilliseconds",
    "thermalCompletedAtEpochMilliseconds",
    "thermalObservations",
    "thermalMaximumPollGapMilliseconds",
    "thermalNonNominalObservations",
  ]) {
    const value = parameters.get(name);
    if (value !== null) thermalInput(name).value = value;
  }
}

function setBlankThermalStartToNow(): void {
  const input = thermalInput("thermalStartedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function setBlankThermalCompletionToNow(): void {
  const input = thermalInput("thermalCompletedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function thermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>("#thermal-gate input[name]")) parameters.set(input.name, input.value);
  return parameters;
}

function thermalInput(name: string): HTMLInputElement {
  return requireElement<HTMLInputElement>(`input[name=\"${name}\"]`);
}

function requiredIdentityString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "" || value !== value.trim()) {
    throw new Error(`OPT-0012 requires run identity ${name}`);
  }
  return value;
}

function requiredIdentityInteger(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredIdentityString(parameters, name));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`OPT-0012 run identity ${name} must be positive`);
  }
  return value;
}

function requiredFiniteNumber(parameters: URLSearchParams, name: string): number {
  const raw = parameters.get(name);
  const value = raw === null || raw.trim() === "" ? Number.NaN : Number(raw);
  if (!Number.isFinite(value)) throw new Error(`OPT-0012 requires ${name}`);
  return value;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0012 element ${selector}`);
  return element;
}

function updatePageProgress(message: string): void {
  requireElement<HTMLElement>("#progress").textContent = message;
}

function finishPageFailure(error: unknown, heartbeat?: HeartbeatSnapshot): void {
  document.body.dataset.status = "failed";
  updatePageProgress("failed");
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    {
      error: serializeError(error),
      ...(heartbeat === undefined ? {} : { pageHeartbeat: heartbeat }),
    },
    null,
    2,
  );
}

// Worker implementation is below the pure page/contract surface so importing
// this module under Vitest never requests a device or opens OPFS.

type WorkerLifecycle = "idle" | "preparing" | "ready" | "running" | "settled";

let workerLifecycle: WorkerLifecycle = "idle";
let workerSession: PreparedSession | undefined;
let workerHeartbeat: ReturnType<typeof startWorkerHeartbeat> | undefined;

function initializeWorker(): void {
  self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
    const message = event.data;
    if (message.type === "initialize") {
      if (workerLifecycle !== "idle") return;
      workerLifecycle = "preparing";
      workerHeartbeat = startWorkerHeartbeat();
      void initializeSession(message.identity).then(
        (prepared) => {
          if (workerLifecycle !== "preparing") return;
          workerSession = prepared;
          workerLifecycle = "ready";
          self.postMessage({
            type: "ready-for-thermal-gate",
            warmupCompletedAtEpochMilliseconds: prepared.warmupCompletedAtEpochMilliseconds,
            preparation: publicPreparationSummary(prepared),
          } satisfies WorkerMessage);
        },
        (error: unknown) => void failWorkerAndCleanup(error),
      );
      return;
    }
    if (message.type === "run-timed" && workerLifecycle === "ready") {
      workerLifecycle = "running";
      const prepared = workerSession!;
      workerSession = undefined;
      void runTimedAndCleanup(prepared, message.thermal).then(
        (result) => {
          workerLifecycle = "settled";
          self.postMessage({ type: "passed", result } satisfies WorkerMessage);
        },
        (error: unknown) => void failWorkerAndCleanup(error, prepared),
      );
      return;
    }
    if (message.type === "run-trajectory" && workerLifecycle === "ready") {
      workerLifecycle = "running";
      const prepared = workerSession!;
      workerSession = undefined;
      void runTrajectoryAndCleanup(prepared).then(
        (result) => {
          workerLifecycle = "settled";
          self.postMessage({ type: "passed", result } satisfies WorkerMessage);
        },
        (error: unknown) => void failWorkerAndCleanup(error, prepared),
      );
    }
  });
}

interface WorkerHeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly timerTickCount: number;
  readonly maximumTimerGapMilliseconds: number;
}

function startWorkerHeartbeat(): { stop(): WorkerHeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  let timerTickCount = 0;
  let maximumTimerGapMilliseconds = 0;
  let last = performance.now();
  let stopped = false;
  const handle = setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, now - last);
    last = now;
    timerTickCount += 1;
  }, HEARTBEAT_INTERVAL_MILLISECONDS);
  return {
    stop(): WorkerHeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, performance.now() - last);
        clearInterval(handle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        timerTickCount,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

async function failWorkerAndCleanup(error: unknown, prepared = workerSession): Promise<void> {
  workerSession = undefined;
  workerLifecycle = "settled";
  let cleanupError: unknown;
  try {
    if (prepared !== undefined) await destroyPreparedSession(prepared);
  } catch (caught) {
    cleanupError = caught;
  }
  const heartbeat = workerHeartbeat?.stop();
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      primary: serializeError(error),
      ...(cleanupError === undefined ? {} : { cleanup: serializeError(cleanupError) }),
      workerHeartbeat: heartbeat,
    }),
  } satisfies WorkerMessage);
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message } satisfies WorkerMessage);
}

function serializeError(error: unknown): unknown {
  return error instanceof Error ? Object.freeze({ name: error.name, message: error.message, stack: error.stack }) : error;
}

// Concrete session, GPU instrumentation, and correctness/timing routines are
// defined in the next section of this benchmark-local module.

interface PreparedSession {
  readonly identity: Opt0012RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly observer: Opt0012DeviceObserver;
  readonly abortController: AbortController;
  readonly executor: AcePlannerGpuExecutor;
  readonly candidateRunner: CandidateHeadRunner;
  readonly preparedPackage: PreparedPackage;
  readonly tokenizer: AceQwenBpeTokenizer;
  readonly fixtures: readonly PlannerCaseFixture[];
  readonly eosFixture: PlannerCaseFixture;
  readonly regularPlan: AceOpt0012CompactSemanticHeadPlan;
  readonly eosPlan: AceOpt0012CompactSemanticHeadPlan;
  readonly fp16ConversionGate: Readonly<Record<string, unknown>>;
  readonly correctness: readonly Readonly<Record<string, unknown>>[];
  readonly adversarialSampling: Readonly<Record<string, unknown>>;
  readonly sourceAuthentication: Readonly<Record<string, string>>;
  readonly capabilityAuthentication: Readonly<Record<string, unknown>>;
  readonly packageAcquisitionWallMilliseconds: number;
  readonly phaseUploadWallMilliseconds: number;
  readonly executorCompileAndCorrectnessWallMilliseconds: number;
  readonly warmupCompletedAtEpochMilliseconds: number;
  destroyed: boolean;
  cleanupReceipt: Readonly<Record<string, unknown>> | null;
  cleanupPromise: Promise<Readonly<Record<string, unknown>>> | null;
}

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly tiedWeightShards: readonly AceOpt0012SourceShardBinding[];
  readonly summary: Readonly<Record<string, unknown>>;
}

type CaseSpec = (typeof OPT_0012_CASE_SPECS)[number] | typeof OPT_0012_EOS_CASE_SPEC;

interface PlannerCaseFixture {
  readonly spec: CaseSpec;
  readonly prefill: AcePlannerPrefillBatch;
  readonly decode: AcePlannerDecodeBatch;
  readonly seenTokenIds: readonly number[];
  readonly decodeTokenId: number;
  readonly promptRows: readonly (readonly number[])[];
}

interface PackageArmExecution {
  readonly arm: Opt0012Arm;
  readonly state: AceOpt0012SemanticState;
  readonly logits: readonly Float32Array[];
  readonly compact: AceOpt0012DecodedCompactLogits | null;
  readonly compactMappedBytes: ArrayBuffer | null;
  readonly fullMappedBytes: ArrayBuffer | null;
  readonly sample: SampleReceipt;
  readonly trace: Readonly<Record<string, unknown>>;
  readonly fp16NaNCensus: Readonly<Record<string, unknown>> | null;
  readonly guard: Readonly<Record<string, unknown>> | null;
  readonly commonCandidateReplay: Readonly<Record<string, unknown>> | null;
  readonly intervals: Readonly<Record<string, number>>;
}

interface SampleReceipt {
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
  readonly drawIndex: string;
  readonly drawEnd: string;
}

interface SameByteReplayExecution {
  readonly arm: Opt0012ReplayArm;
  readonly sample: SampleReceipt;
  readonly totalWallMilliseconds: number;
  readonly fp16DecodeMilliseconds: number;
  readonly reconstructionMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly samplingStageIntervals: SamplingStageIntervals;
  readonly namedNonOverlappingIntervals: Readonly<Record<string, unknown>>;
  readonly decodedRowSha256: readonly [string, string];
  readonly reconstructedFullRowSha256: readonly [string, string] | null;
  readonly writeStatus: readonly [number, number];
  readonly exactSampleAuthority: true;
}

interface SameByteReplayMeasuredExecution {
  readonly timing: Omit<SameByteReplayExecution, "decodedRowSha256" | "reconstructedFullRowSha256" | "exactSampleAuthority">;
  readonly decoded: AceOpt0012DecodedCompactLogits;
  readonly reconstructedFullRows: readonly [Float32Array, Float32Array] | null;
}

interface SameByteReplayPair {
  readonly pairIndex: number;
  readonly order: readonly Opt0012ReplayArm[];
  readonly mappedBytesSha256Before: string;
  readonly mappedBytesSha256After: string;
  readonly B: SameByteReplayExecution;
  readonly C: SameByteReplayExecution;
  readonly sameImmutableBytes: true;
  readonly noYieldInsidePair: true;
}

interface ConversionMicrobenchmarkExecution {
  readonly arm: Opt0012ConversionArm;
  readonly wallMilliseconds: number;
  readonly envelope: Readonly<Record<string, unknown>>;
  readonly acceptedStableNanEnvelope: true;
}

interface ConversionMicrobenchmarkPair {
  readonly pairIndex: number;
  readonly order: readonly Opt0012ConversionArm[];
  readonly legacy: ConversionMicrobenchmarkExecution;
  readonly candidate: ConversionMicrobenchmarkExecution;
  readonly noYieldInsidePair: true;
}

interface TrajectoryArmExecution {
  readonly arm: Opt0012Arm;
  readonly repeatIndex: number;
  readonly codeCount: number;
  readonly tokens: readonly SampleReceipt[];
  readonly terminal: SampleReceipt;
  readonly terminalTokenId: number;
  readonly finalDrawEnd: string;
  readonly semanticCodeSha256: string;
  readonly serializedAudioCodeTextSha256: string;
  readonly topology: Readonly<Record<string, unknown>>;
  readonly totalWallMilliseconds: number;
  readonly performanceComparisonAuthority: string;
  readonly semanticHashAuthority: string;
}

interface FullSamplingTrace {
  readonly cfg: Float32Array;
  readonly penalized: Float32Array;
  readonly topK: Float32Array;
  readonly topPKeep: Uint8Array;
  readonly topP: Float32Array;
  readonly scaled: Float32Array;
  readonly weights: Float32Array;
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
}

async function initializeSession(identity: unknown): Promise<PreparedSession> {
  const runIdentity = validateWorkerRunIdentity(identity);
  postProgress("authenticating every pinned planner/head/sampler source");
  const sourceAuthentication = authenticateOpt0012Sources();
  postProgress("correctness: exhaustive target-browser FP16 conversion gate");
  const fp16ConversionGate = runOpt0012Fp16ConversionCorrectnessGate();
  postProgress("authenticating converter-revision-4 raw-FP16 package inventory");
  const acquisitionStarted = performance.now();
  const preparedPackage = await preparePackage();
  const packageAcquisitionWallMilliseconds = performance.now() - acquisitionStarted;
  const regularPlan = createAuthenticatedPlan("regular-code", preparedPackage.tiedWeightShards);
  const eosPlan = createAuthenticatedPlan("forced-eos", preparedPackage.tiedWeightShards);
  validateFrozenPlans(regularPlan, eosPlan);
  // This is the browser-level M1 fail-closed proof. A candidate plan never
  // exists for M1, and the execution wrapper also requires rows===2.
  const m1Request = createAceOpt0012PlanRequest("regular-code");
  let m1Rejected = false;
  try {
    createAceOpt0012CompactSemanticHeadPlan({ ...m1Request, phase: "cot-m1" });
  } catch {
    m1Rejected = true;
  }
  if (!m1Rejected) throw new Error("OPT-0012 candidate admitted M1");

  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  postProgress("requesting the shipped raw-FP16 cooperative WebGPU device");
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  const capabilityAuthentication = authenticateOpt0012Capabilities(context);
  const observer = new Opt0012DeviceObserver(context.device);
  const abortController = new AbortController();
  let phase: AceGpuTensorPhase | undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  let candidateRunner: CandidateHeadRunner | undefined;
  try {
    postProgress("uploading only the authenticated planner phase");
    const uploadStarted = performance.now();
    let lastStatusAt = 0;
    phase = await AceGpuTensorPhase.load(observer.device, preparedPackage.manifest, preparedPackage.acquiredFiles, ["planner"], {
      signal: abortController.signal,
      onProgress: (progress) => {
        const now = performance.now();
        if (now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS || progress.loadedPhaseBytes === progress.totalPhaseBytes) {
          lastStatusAt = now;
          postProgress(
            `uploading planner shard ${progress.phaseFileIndex + 1}/` +
              `${progress.phaseFileCount} (${formatBytes(progress.loadedPhaseBytes)}/` +
              `${formatBytes(progress.totalPhaseBytes)})`,
          );
        }
      },
    });
    if (
      phase.phases.length !== 1 ||
      phase.phases[0] !== "planner" ||
      phase.residentBytes !== OPT_0012_PLANNER_RESIDENT_BYTES ||
      phase.packageManifest !== preparedPackage.manifest
    ) {
      throw new Error("OPT-0012 resident planner phase identity changed");
    }
    authenticateResidentTiedBindings(phase, preparedPackage.tiedWeightShards);
    const phaseUploadWallMilliseconds = performance.now() - uploadStarted;

    const loadedTokenizer = await loadPinnedAceTokenizer("planner", {
      tokenizerJson: requirePackageFile(preparedPackage.acquiredFiles, "assets/planner/tokenizer.json"),
      tokenizerConfigJson: requirePackageFile(preparedPackage.acquiredFiles, "assets/planner/tokenizer_config.json"),
      chatTemplate: requirePackageFile(preparedPackage.acquiredFiles, "assets/planner/chat_template.jinja"),
    });
    const tokenizer = loadedTokenizer.tokenizer;
    const fixtures = createCaseFixtures(tokenizer);
    const eosFixture = createEosCaseFixture(fixtures[2]!.promptRows);
    const compileAndCorrectnessStarted = performance.now();
    const progressRecorder = new Opt0012ProgressRecorder();
    executor = AcePlannerGpuExecutor.create({
      device: observer.device,
      modelProfile: "raw-fp16",
      ownedPlannerWeights: phase,
      signal: abortController.signal,
      onProgress: (progress) => progressRecorder.acceptProduction(progress),
    });
    installOpt0012ExecutorIdleHook(executor, progressRecorder);
    phase = undefined;
    candidateRunner = new CandidateHeadRunner(observer, executor, progressRecorder, abortController.signal, regularPlan, eosPlan);
    const adversarialSampling = runAdversarialSamplingGate(regularPlan, eosPlan);
    const correctness: Array<Readonly<Record<string, unknown>>> = [];
    for (let index = 0; index < fixtures.length; index += 1) {
      const fixture = fixtures[index]!;
      postProgress(`correctness ${index + 1}/${fixtures.length}: ${fixture.spec.id} A/B/C`);
      correctness.push(await runPackageCorrectnessCase(observer, executor, candidateRunner, fixture, regularPlan));
    }
    postProgress("correctness: forced-EOS A/B/C at the long cache point");
    correctness.push(await runPackageCorrectnessCase(observer, executor, candidateRunner, eosFixture, eosPlan));
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0012 correctness observed a WebGPU runtime event");
    }
    const executorCompileAndCorrectnessWallMilliseconds = performance.now() - compileAndCorrectnessStarted;
    return {
      identity: runIdentity,
      context,
      runtimeEvents,
      observer,
      abortController,
      executor,
      candidateRunner,
      preparedPackage,
      tokenizer,
      fixtures,
      eosFixture,
      regularPlan,
      eosPlan,
      fp16ConversionGate,
      correctness: Object.freeze(correctness),
      adversarialSampling,
      sourceAuthentication,
      capabilityAuthentication,
      packageAcquisitionWallMilliseconds,
      phaseUploadWallMilliseconds,
      executorCompileAndCorrectnessWallMilliseconds,
      warmupCompletedAtEpochMilliseconds: Date.now(),
      destroyed: false,
      cleanupReceipt: null,
      cleanupPromise: null,
    };
  } catch (error) {
    candidateRunner?.destroy();
    if (executor !== undefined) await executor.destroy(error);
    else phase?.destroy();
    context.destroy();
    await context.lost.catch(() => undefined);
    await yieldToBrowser();
    throw error;
  }
}

function validateWorkerRunIdentity(identity: unknown): Opt0012RunIdentity {
  if (typeof identity !== "object" || identity === null) {
    throw new Error("OPT-0012 worker requires a run identity");
  }
  const candidate = identity as Readonly<Record<string, unknown>>;
  const parameters = new URLSearchParams();
  for (const name of [
    "harnessCommit",
    "coreCommit",
    "allocationCommit",
    "machineModel",
    "osVersion",
    "osBuild",
    "browserVersion",
    "gpuCoreCount",
    "memoryBytes",
  ]) {
    const value = candidate[name];
    if (typeof value !== "string" && typeof value !== "number") {
      throw new Error(`OPT-0012 worker rejected run identity ${name}`);
    }
    parameters.set(name, String(value));
  }
  return parseOpt0012RunIdentity(parameters);
}

function authenticateOpt0012Capabilities(context: AceWebGpuDeviceContext): Readonly<Record<string, unknown>> {
  const capabilities = context.capabilities;
  const profile = capabilities.executionProfile;
  const requiredLimits = Object.entries(ACE_REQUIRED_WEBGPU_LIMITS).sort(([left], [right]) => left.localeCompare(right));
  const expectedRequestedLimits = requiredLimits.filter(([name]) => name !== "minStorageBufferOffsetAlignment" && name !== "minUniformBufferOffsetAlignment");
  const observedRequestedLimits = Object.entries(capabilities.requestedLimits).sort(([left], [right]) => left.localeCompare(right));
  if (
    profile.id !== "raw-fp16-portable" ||
    profile.modelProfile !== "raw-fp16" ||
    profile.weightStorage !== "float16" ||
    profile.matrixArithmetic !== "float16" ||
    profile.sensitiveReductions !== "float32" ||
    profile.vaeArithmetic !== "float32" ||
    profile.kernelBackend !== "portable" ||
    profile.requiredFeatures.length !== 1 ||
    profile.requiredFeatures[0] !== "shader-f16" ||
    capabilities.schedulingProfile !== "cooperative" ||
    !capabilities.adapterFeatures.includes("shader-f16") ||
    !capabilities.deviceFeatures.includes("shader-f16") ||
    !capabilities.stockFeatures["shader-f16"].adapterSupported ||
    !capabilities.stockFeatures["shader-f16"].deviceEnabled ||
    !capabilities.stockFeatures["shader-f16"].required ||
    !capabilities.stockFeatures["shader-f16"].requested ||
    JSON.stringify(observedRequestedLimits) !== JSON.stringify(expectedRequestedLimits)
  ) {
    throw new Error("OPT-0012 WebGPU capability identity changed");
  }
  for (const [name, requested] of requiredLimits) {
    const deviceActual = capabilities.deviceLimits[name as keyof typeof capabilities.deviceLimits];
    const adapterActual = capabilities.adapterLimits[name as keyof typeof capabilities.adapterLimits];
    const minimumAlignment = name === "minStorageBufferOffsetAlignment" || name === "minUniformBufferOffsetAlignment";
    if (
      typeof requested !== "number" ||
      (minimumAlignment ? deviceActual > requested || adapterActual > requested : deviceActual < requested || adapterActual < requested)
    ) {
      throw new Error(`OPT-0012 adapter/device limit ${name} changed`);
    }
  }
  return Object.freeze({
    ...capabilities,
    authenticatedExecutionProfile: "raw-fp16-portable",
    authenticatedSchedulingProfile: "cooperative",
    shaderF16AdapterSupportedAndDeviceEnabled: true,
    exactRequestedLimits: true,
    actualDeviceLimitsSatisfyEveryRequestedLimit: true,
  });
}

function authenticateOpt0012Sources(): Readonly<Record<string, string>> {
  const identityNames = Object.keys(OPT_0012_SOURCE_IDENTITIES).sort();
  const sourceNames = Object.keys(OPT_0012_SOURCE_TEXT).sort();
  if (identityNames.join("\n") !== sourceNames.join("\n")) {
    throw new Error("OPT-0012 source identity inventory is incomplete");
  }
  const result: Record<string, string> = {};
  for (const name of identityNames) {
    const source = OPT_0012_SOURCE_TEXT[name as keyof typeof OPT_0012_SOURCE_TEXT];
    const expected = OPT_0012_SOURCE_IDENTITIES[name as keyof typeof OPT_0012_SOURCE_IDENTITIES];
    const actual = aceSha256Hex(new TextEncoder().encode(source));
    if (actual !== expected) {
      throw new Error(`OPT-0012 source authentication failed for ${name}`);
    }
    result[name] = actual;
  }
  return Object.freeze(result);
}

async function preparePackage(): Promise<PreparedPackage> {
  const manifestUrl = new URL(OPT_0012_FP16_MANIFEST_PATH, self.location.href).href;
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: OPT_0012_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16",
  });
  if (
    loaded.manifestSha256 !== OPT_0012_FP16_MANIFEST_SHA256 ||
    loaded.manifest.provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION ||
    ACE_PACKAGE_CONVERTER_REVISION !== 4
  ) {
    throw new Error("OPT-0012 raw-FP16 manifest trust root changed");
  }
  const inventory = validatePlannerInventory(loaded.manifest);
  const acquisitionManifest = Object.freeze({
    ...loaded.manifest,
    files: inventory.files,
  });
  const cache = await AceOpfsModelCache.open();
  let lastStatusAt = 0;
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      const now = performance.now();
      if (
        now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
        (progress.fileIndex + 1 === progress.fileCount && progress.fileReceivedBytes === progress.fileBytes)
      ) {
        lastStatusAt = now;
        postProgress(
          `acquiring authenticated planner file ${progress.fileIndex + 1}/` +
            `${progress.fileCount} (${formatBytes(progress.completedBytes)}/` +
            `${formatBytes(progress.totalBytes)}, ${progress.source})`,
        );
      }
    },
  });
  if (acquired.files.size !== OPT_0012_ACQUIRED_FILE_COUNT || acquired.plan.files.length !== OPT_0012_ACQUIRED_FILE_COUNT) {
    throw new Error("OPT-0012 bounded planner acquisition changed");
  }
  for (const file of inventory.files) {
    const source = acquired.files.get(file.name);
    if (!(source instanceof File) || source.size !== file.byteLength) {
      throw new Error(`OPT-0012 acquired file identity changed: ${file.name}`);
    }
  }
  return Object.freeze({
    manifest: loaded.manifest,
    manifestUrl: loaded.manifestUrl,
    acquiredFiles: acquired.files,
    tiedWeightShards: inventory.tiedWeightShards,
    summary: Object.freeze({
      manifestSha256: loaded.manifestSha256,
      manifestByteLength: loaded.manifestByteLength,
      converterRevision: loaded.manifest.provenance.converterRevision,
      plannerTensorCount: inventory.tensorCount,
      plannerWeightFileCount: inventory.weightFileCount,
      tokenizerFileCount: OPT_0012_TOKENIZER_FILE_COUNT,
      plannerResidentBytes: inventory.residentBytes,
      acquiredFileCount: acquired.files.size,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
      everyAcquiredFileAuthenticatedByManifestSha256: true,
      files: Object.freeze(
        inventory.files.map((file) =>
          Object.freeze({
            name: file.name,
            byteLength: file.byteLength,
            sha256: file.sha256,
          }),
        ),
      ),
      tiedWeightShards: inventory.tiedWeightShards,
    }),
  });
}

function validatePlannerInventory(manifest: AcePackageManifest): Readonly<{
  files: readonly AcePackageFileRecord[];
  tensorCount: number;
  weightFileCount: number;
  residentBytes: number;
  tiedWeightShards: readonly AceOpt0012SourceShardBinding[];
}> {
  const tensors = Object.values(manifest.tensors).filter((tensor) => tensor.phase === "planner");
  const weightNames = new Set(tensors.map((tensor) => tensor.shard));
  const tokenizerNames = new Set(["assets/planner/tokenizer.json", "assets/planner/tokenizer_config.json", "assets/planner/chat_template.jinja"]);
  const files = manifest.files.filter((file) => weightNames.has(file.name) || tokenizerNames.has(file.name));
  const weightFiles = files.filter((file) => weightNames.has(file.name));
  const fileByName = new Map(files.map((file) => [file.name, file]));
  const residentBytes = sumSafe(
    weightFiles.map((file) => file.byteLength),
    "OPT-0012 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0012 planner tensor bytes",
  );
  if (
    tensors.length !== OPT_0012_PLANNER_TENSOR_COUNT ||
    weightNames.size !== OPT_0012_PLANNER_WEIGHT_FILE_COUNT ||
    weightFiles.length !== OPT_0012_PLANNER_WEIGHT_FILE_COUNT ||
    files.length !== OPT_0012_ACQUIRED_FILE_COUNT ||
    residentBytes !== OPT_0012_PLANNER_RESIDENT_BYTES ||
    tensorBytes !== OPT_0012_PLANNER_RESIDENT_BYTES
  ) {
    throw new Error("OPT-0012 planner inventory changed from revision 4");
  }
  const expected = createAceOpt0012PlanRequest("regular-code").tiedWeightShards;
  const tiedWeightShards = expected.map((shard) => {
    const tensor = manifest.tensors[shard.tensorName];
    if (tensor === undefined) {
      throw new Error(`OPT-0012 manifest omitted ${shard.tensorName}`);
    }
    authenticateTiedTensorRecord(tensor, shard);
    const file = fileByName.get(tensor.shard);
    if (file === undefined || file.kind !== "weights") {
      throw new Error(`OPT-0012 tied tensor file is absent: ${tensor.shard}`);
    }
    return Object.freeze({
      ...shard,
      bindingByteOffset: tensor.byteOffset,
      bindingByteLength: tensor.byteLength,
      bufferByteLength: file.byteLength,
    });
  });
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    weightFileCount: weightFiles.length,
    residentBytes,
    tiedWeightShards: Object.freeze(tiedWeightShards),
  });
}

function authenticateTiedTensorRecord(tensor: AcePackageTensorRecord, expected: AceOpt0012SourceShardBinding): void {
  if (
    tensor.phase !== "planner" ||
    tensor.lifetime !== "planner" ||
    tensor.dtype !== "float16" ||
    tensor.layout !== "row-shard-axis0" ||
    tensor.transformation !== "bf16-to-ieee-fp16" ||
    tensor.source !== "planner-weights:model.embed_tokens.weight" ||
    tensor.logicalTensor !== "planner.model.embed_tokens.weight" ||
    tensor.partAxis !== 0 ||
    tensor.partStart !== expected.firstRow ||
    tensor.partEnd !== expected.firstRow + expected.rowCount ||
    tensor.byteOffset !== expected.bindingByteOffset ||
    tensor.byteLength !== expected.bindingByteLength ||
    tensor.logicalShape.length !== 2 ||
    tensor.logicalShape[0] !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize ||
    tensor.logicalShape[1] !== ACE_PLANNER_QWEN3_CONFIG.hiddenSize ||
    tensor.storageShape.length !== 2 ||
    tensor.storageShape[0] !== expected.rowCount ||
    tensor.storageShape[1] !== ACE_PLANNER_QWEN3_CONFIG.hiddenSize
  ) {
    throw new Error(`OPT-0012 tied tensor metadata changed at row ${expected.firstRow}`);
  }
}

function createAuthenticatedPlan(state: AceOpt0012SemanticState, shards: readonly AceOpt0012SourceShardBinding[]): AceOpt0012CompactSemanticHeadPlan {
  const base = createAceOpt0012PlanRequest(state);
  const request: AceOpt0012PlanRequest = Object.freeze({
    ...base,
    tiedWeightShards: shards,
  });
  return createAceOpt0012CompactSemanticHeadPlan(request);
}

function validateFrozenPlans(regular: AceOpt0012CompactSemanticHeadPlan, eos: AceOpt0012CompactSemanticHeadPlan): void {
  if (
    ACE_PLANNER_SEMANTIC_CODE_COUNT !== 64_000 ||
    DEFAULT_ACE_PLANNER_CONFIGURATION.temperature !== ACE_OPT_0012_SAMPLING_PARAMETERS.temperature ||
    DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale !== ACE_OPT_0012_SAMPLING_PARAMETERS.guidanceScale ||
    DEFAULT_ACE_PLANNER_CONFIGURATION.topK !== ACE_OPT_0012_SAMPLING_PARAMETERS.topK ||
    DEFAULT_ACE_PLANNER_CONFIGURATION.topP !== ACE_OPT_0012_SAMPLING_PARAMETERS.topP ||
    regular.intersections.length !== 2 ||
    regular.intersections[0]?.shardIndex !== 3 ||
    regular.intersections[0].localFirstRow !== 4_213 ||
    regular.intersections[0].rowCount !== 44_939 ||
    regular.intersections[1]?.shardIndex !== 4 ||
    regular.intersections[1].localFirstRow !== 0 ||
    regular.intersections[1].rowCount !== 19_061 ||
    regular.readback.rawLogitBytes !== 256_000 ||
    regular.readback.allocationBytes !== 256_256 ||
    regular.readback.copies.length !== OPT_0012_REGULAR_COPY_COUNT ||
    regular.workgroupCount !== 501 ||
    eos.intersections.length !== 1 ||
    eos.intersections[0]?.shardIndex !== 3 ||
    eos.intersections[0].localFirstRow !== 4_189 ||
    eos.intersections[0].rowCount !== 1 ||
    eos.readback.rawLogitBytes !== 4 ||
    eos.readback.allocationBytes !== 256 ||
    eos.readback.copies.length !== OPT_0012_EOS_COPY_COUNT ||
    eos.workgroupCount !== 1
  ) {
    throw new Error("OPT-0012 frozen regular/EOS plan changed");
  }
  const generated = {
    "regular-shard-3": aceSha256Hex(
      new TextEncoder().encode(
        aceCorrectnessGemmWgsl(
          "raw-fp16",
          {
            rows: 2,
            inner: 1_024,
            columns: 44_939,
          },
          false,
          "source-row-major",
        ),
      ),
    ),
    "regular-shard-4": aceSha256Hex(
      new TextEncoder().encode(
        aceCorrectnessGemmWgsl(
          "raw-fp16",
          {
            rows: 2,
            inner: 1_024,
            columns: 19_061,
          },
          false,
          "source-row-major",
        ),
      ),
    ),
    "forced-eos-shard-3": aceSha256Hex(
      new TextEncoder().encode(
        aceCorrectnessGemmWgsl(
          "raw-fp16",
          {
            rows: 2,
            inner: 1_024,
            columns: 1,
          },
          false,
          "source-row-major",
        ),
      ),
    ),
  };
  for (const [name, sha256] of Object.entries(generated)) {
    if (sha256 !== OPT_0012_CANDIDATE_SHADER_SHA256[name as keyof typeof OPT_0012_CANDIDATE_SHADER_SHA256])
      throw new Error(`OPT-0012 candidate shader changed: ${name}`);
  }
}

function authenticateResidentTiedBindings(phase: AceGpuTensorPhase, expected: readonly AceOpt0012SourceShardBinding[]): void {
  const logical = phase.logicalTensor("planner.model.embed_tokens.weight");
  if (logical.parts.length !== expected.length) {
    throw new Error("OPT-0012 resident tied-head shard count changed");
  }
  for (let index = 0; index < logical.parts.length; index += 1) {
    const part = logical.parts[index]!;
    const shard = expected[index]!;
    if (
      part.tensorName !== shard.tensorName ||
      part.binding.offset !== shard.bindingByteOffset ||
      part.binding.size !== shard.bindingByteLength ||
      part.binding.buffer.size !== shard.bufferByteLength
    ) {
      throw new Error(`OPT-0012 resident tied-head binding ${index} changed`);
    }
  }
}

function createCaseFixtures(tokenizer: AceQwenBpeTokenizer): readonly PlannerCaseFixture[] {
  const prompts = createAcePlannerCodePrompts(ACCEPTED_RESOLVED_CAPTION, ACCEPTED_LYRICS, ACCEPTED_COT_TEXT);
  const promptRows = Object.freeze([Object.freeze(tokenizer.encode(prompts.conditional)), Object.freeze(tokenizer.encode(prompts.unconditional))]);
  const continuation = OPT_0012_ACCEPTED_SEMANTIC_CODE_IDS.map((code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code);
  if (
    promptRows[0]!.length !== 253 ||
    promptRows[1]!.length !== 33 ||
    continuation.length !== 150 ||
    sha256U32Le(OPT_0012_ACCEPTED_SEMANTIC_CODE_IDS) !== OPT_0012_ACCEPTED_SEMANTIC_CODE_SHA256
  ) {
    throw new Error("OPT-0012 historical packed-BF16 teacher fixture changed");
  }
  return Object.freeze(
    OPT_0012_CASE_SPECS.map((spec) => {
      const prefill = createPaddedSemanticPrefill(promptRows, continuation, spec.cachedTokensBeforeAppend, spec.cacheCapacity);
      const promptWidth = Math.max(...promptRows.map((row) => row.length));
      const continuationIndex = spec.cachedTokensBeforeAppend - promptWidth;
      const decodeTokenId = continuation[continuationIndex]!;
      const decode = createSemanticDecodeBatch(decodeTokenId, spec.cachedTokensBeforeAppend, spec.cacheCapacity);
      return Object.freeze({
        spec,
        prefill,
        decode,
        decodeTokenId,
        seenTokenIds: Object.freeze([...prefill.inputIds.slice(0, spec.cachedTokensBeforeAppend), decodeTokenId]),
        promptRows,
      });
    }),
  );
}

function createEosCaseFixture(promptRows: readonly (readonly number[])[]): PlannerCaseFixture {
  const continuation = OPT_0012_ACCEPTED_SEMANTIC_CODE_IDS.map((code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code);
  const spec = OPT_0012_EOS_CASE_SPEC;
  const prefill = createPaddedSemanticPrefill(promptRows, continuation, spec.cachedTokensBeforeAppend, spec.cacheCapacity);
  const promptWidth = Math.max(...promptRows.map((row) => row.length));
  const continuationIndex = spec.cachedTokensBeforeAppend - promptWidth;
  if (continuationIndex !== 149 || continuation[continuationIndex] === undefined) {
    throw new Error("OPT-0012 forced-EOS teacher fixture lost its 150th code");
  }
  const decodeTokenId = continuation[continuationIndex]!;
  const decode = createSemanticDecodeBatch(decodeTokenId, spec.cachedTokensBeforeAppend, spec.cacheCapacity);
  return Object.freeze({
    spec,
    prefill,
    decode,
    decodeTokenId,
    seenTokenIds: Object.freeze([...prefill.inputIds.slice(0, spec.cachedTokensBeforeAppend), decodeTokenId]),
    promptRows,
  });
}

function createPaddedSemanticPrefill(
  promptRows: readonly (readonly number[])[],
  continuation: readonly number[],
  tokens: number,
  cacheCapacity: number,
): AcePlannerPrefillBatch {
  if (promptRows.length !== 2 || tokens >= cacheCapacity) {
    throw new Error("OPT-0012 semantic prefill geometry is invalid");
  }
  const promptWidth = Math.max(...promptRows.map((row) => row.length));
  if (promptWidth > tokens || tokens - promptWidth > continuation.length) {
    throw new Error("OPT-0012 semantic prefill leaves its teacher-forced fixture");
  }
  const inputIds = new Uint32Array(2 * tokens);
  const keyValidity = new Uint32Array(2 * tokens);
  for (let row = 0; row < 2; row += 1) {
    const prompt = promptRows[row]!;
    const rowOffset = row * tokens;
    const padding = promptWidth - prompt.length;
    inputIds.fill(ACE_QWEN_PAD_TOKEN_ID, rowOffset, rowOffset + padding);
    inputIds.set(prompt, rowOffset + padding);
    keyValidity.fill(1, rowOffset + padding, rowOffset + promptWidth);
    for (let position = promptWidth; position < tokens; position += 1) {
      inputIds[rowOffset + position] = continuation[position - promptWidth]!;
      keyValidity[rowOffset + position] = 1;
    }
  }
  const causal = createAceQwen3CausalControlData({
    batch: 2,
    tokens,
    cacheCapacity,
    rowStartPositions: [0, 0],
    validKeyLengths: [tokens, tokens],
    sourceValidity: [...keyValidity],
  });
  return Object.freeze({
    kind: "prefill" as const,
    rows: 2 as const,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: 1 as const,
  });
}

function createSemanticDecodeBatch(tokenId: number, cachedTokensBeforeAppend: number, cacheCapacity: number): AcePlannerDecodeBatch {
  const causal = createAceQwen3CausalControlData({
    batch: 2,
    tokens: 1,
    cacheCapacity,
    rowStartPositions: [cachedTokensBeforeAppend, cachedTokensBeforeAppend],
    validKeyLengths: [cachedTokensBeforeAppend + 1, cachedTokensBeforeAppend + 1],
    sourceValidity: [1, 1],
  });
  return Object.freeze({
    kind: "decode" as const,
    rows: 2 as const,
    tokens: 1 as const,
    cacheCapacity,
    cachedTokensBeforeAppend,
    inputIds: new Uint32Array([tokenId, tokenId]),
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: 1 as const,
  });
}

function authenticateDecodeControls(decode: AcePlannerDecodeBatch): Readonly<Record<string, unknown>> {
  const cached = decode.cachedTokensBeforeAppend;
  const expectedValidLengths = [1, cached + 1, 1, cached + 1];
  const expectedRows = [cached, cached];
  if (
    decode.rows !== 2 ||
    decode.tokens !== 1 ||
    decode.inputIds.length !== 2 ||
    decode.inputIds[0] !== decode.inputIds[1] ||
    JSON.stringify([...decode.causal.validLengths]) !== JSON.stringify(expectedValidLengths) ||
    JSON.stringify([...decode.causal.rowStartPositions]) !== JSON.stringify(expectedRows) ||
    JSON.stringify([...decode.causal.queryPositions]) !== JSON.stringify(expectedRows) ||
    JSON.stringify([...decode.causal.sourceValidity]) !== JSON.stringify([1, 1])
  ) {
    throw new Error("OPT-0012 decode cache-append controls changed");
  }
  return Object.freeze({
    cachedTokensBeforeAppend: cached,
    cacheCapacity: decode.cacheCapacity,
    appendedTokenId: decode.inputIds[0],
    rowStartPositions: Object.freeze(expectedRows),
    validLengths: Object.freeze(expectedValidLengths),
    queryPositions: Object.freeze(expectedRows),
    sourceValidity: Object.freeze([1, 1]),
    exact: true,
  });
}

function runAdversarialSamplingGate(regular: AceOpt0012CompactSemanticHeadPlan, eos: AceOpt0012CompactSemanticHeadPlan): Readonly<Record<string, unknown>> {
  const cases = [
    adversarialRows(
      regular,
      "boundary-ties",
      (conditional, unconditional) => {
        conditional.fill(-20);
        unconditional.fill(-20);
        conditional[44_938] = 5;
        conditional[44_939] = 5;
        unconditional[44_938] = 1;
        unconditional[44_939] = 1;
      },
      {
        temperature: 1,
        guidanceScale: 2,
        topK: 1,
        topP: 0.49,
        repetitionPenalty: 1,
      },
      [],
      0xffff_ffff,
    ),
    adversarialRows(
      regular,
      "top-p-boundary-tie",
      (conditional, unconditional) => {
        conditional.fill(-120);
        unconditional.fill(-120);
        conditional[44_938] = 5;
        conditional[44_939] = 5;
        unconditional[44_938] = 1;
        unconditional[44_939] = 1;
      },
      {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 0.49,
        repetitionPenalty: 1,
      },
      [],
      0xffff_ffff,
    ),
    adversarialRows(
      regular,
      "first-dominant",
      (conditional, unconditional) => {
        conditional.fill(-120);
        unconditional.fill(-120);
        conditional[0] = 12;
        conditional[conditional.length - 1] = 11.5;
      },
      {
        temperature: 2,
        guidanceScale: 2,
        topK: regular.vocabularySize,
        topP: 0.000_001,
        repetitionPenalty: 1,
      },
      [],
      0x8000_0000,
    ),
    adversarialRows(
      regular,
      "last-dominant",
      (conditional, unconditional) => {
        conditional.fill(-120);
        unconditional.fill(-120);
        conditional[0] = 11.5;
        conditional[conditional.length - 1] = 12;
      },
      {
        temperature: 2,
        guidanceScale: 2,
        topK: regular.vocabularySize,
        topP: 0.000_001,
        repetitionPenalty: 1,
      },
      [],
      0xffff_ffff,
    ),
    adversarialRows(
      regular,
      "repetition-top-p-boundary",
      (conditional, unconditional) => {
        for (let index = 0; index < conditional.length; index += 1) {
          conditional[index] = Math.fround((((index * 73) % 257) - 128) / 16);
          unconditional[index] = Math.fround((((index * 31) % 193) - 96) / 24);
        }
      },
      {
        temperature: 0.25,
        guidanceScale: 1.5,
        topK: 31,
        topP: 0.37,
        repetitionPenalty: 1.3,
      },
      [regular.firstCandidateTokenId + 17, regular.firstCandidateTokenId + 31_999],
      0,
    ),
    adversarialRows(
      regular,
      "subnormal-zero-tail",
      (conditional, unconditional) => {
        conditional.fill(-110);
        unconditional.fill(-110);
        conditional[0] = 0;
        unconditional[0] = 0;
        conditional[conditional.length - 1] = -103.9;
        unconditional[unconditional.length - 1] = -103.9;
      },
      {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
      [],
      0xffff_ffff,
    ),
    adversarialRows(
      eos,
      "one-positive-forced-eos",
      (conditional, unconditional) => {
        conditional[0] = 3;
        unconditional[0] = -2;
      },
      {
        temperature: 0.85,
        guidanceScale: 2,
        topK: 32,
        topP: 0.9,
        repetitionPenalty: 1,
      },
      [ACE_QWEN_IM_END_TOKEN_ID],
      0x1234_5678,
    ),
  ];
  const topPBoundaryTie = cases.find((entry) => entry.id === "top-p-boundary-tie");
  if (
    topPBoundaryTie === undefined ||
    (topPBoundaryTie.topKGlobalTokenIds as readonly number[]).length !== regular.candidateCount ||
    !(topPBoundaryTie.topKGlobalTokenIds as readonly number[]).includes(196_607) ||
    !(topPBoundaryTie.topKGlobalTokenIds as readonly number[]).includes(196_608) ||
    JSON.stringify(topPBoundaryTie.topPGlobalTokenIds) !== JSON.stringify([196_607])
  ) {
    throw new Error("OPT-0012 adversarial top-p tie no longer crosses shard boundary");
  }
  return Object.freeze({
    cases: Object.freeze(cases),
    everyObservableBoundaryComparedAsU32: true,
    globalTieOrder: "descending-logit-then-ascending-global-token-id",
    topPBoundaryTieGlobalIds: Object.freeze({
      tiedBeforeCutoff: Object.freeze([196_607, 196_608]),
      keptAfterCutoff: Object.freeze([196_607]),
    }),
  });
}

function adversarialRows(
  plan: AceOpt0012CompactSemanticHeadPlan,
  id: string,
  mutate: (conditional: Float32Array, unconditional: Float32Array) => void,
  parameters: AcePlannerSamplingParameters,
  seenTokenIds: readonly number[],
  word: number,
): Readonly<Record<string, unknown>> {
  const conditional = new Float32Array(plan.candidateCount);
  const unconditional = new Float32Array(plan.candidateCount);
  mutate(conditional, unconditional);
  const comparison = compareFullAndCompactSampling(plan, conditional, unconditional, seenTokenIds, parameters, word);
  return Object.freeze({ id, ...comparison });
}

function compareFullAndCompactSampling(
  plan: AceOpt0012CompactSemanticHeadPlan,
  conditional: Float32Array,
  unconditional: Float32Array,
  seenTokenIds: readonly number[],
  parameters: AcePlannerSamplingParameters,
  word: number,
): Readonly<Record<string, unknown>> {
  const compact = traceAceOpt0012CompactBrowserV1Sampling({
    plan,
    conditionalLogits: conditional,
    unconditionalLogits: unconditional,
    seenTokenIds,
    parameters,
    word,
  });
  const full = fullVectorTrace(plan, conditional, unconditional, seenTokenIds, parameters, word);
  const u32Equivalence = requireCompactTraceEqualsFull(plan, compact, full);
  return Object.freeze({
    tokenId: compact.tokenId,
    word: compact.word,
    positiveCandidateCount: compact.positiveCandidateCount,
    topKGlobalTokenIds: compact.topKGlobalTokenIds,
    topPGlobalTokenIds: compact.topPGlobalTokenIds,
    cfgSha256: sha256FloatWords(compact.cfgLogits),
    weightsSha256: sha256FloatWords(compact.weights),
    u32Equivalence,
    exact: true,
  });
}

function fullVectorTrace(
  plan: AceOpt0012CompactSemanticHeadPlan,
  conditionalCompact: Float32Array,
  unconditionalCompact: Float32Array,
  seenTokenIds: readonly number[],
  parameters: AcePlannerSamplingParameters,
  word: number,
): FullSamplingTrace {
  const [conditional, unconditional] = reconstructAceOpt0012FullPlannerLogits(
    {
      conditionalLogits: conditionalCompact,
      unconditionalLogits: unconditionalCompact,
      writeStatus: new Uint32Array([1, 1]),
    },
    plan,
  );
  const compactCfg = combineAcePlannerCfgLogits(conditionalCompact, unconditionalCompact, parameters.guidanceScale);
  const cfg = new Float32Array(plan.vocabularySize);
  cfg.fill(Number.NEGATIVE_INFINITY);
  cfg.set(compactCfg, plan.firstCandidateTokenId);
  const penalized = applyAcePlannerRepetitionPenalty(cfg, seenTokenIds, parameters.repetitionPenalty);
  const topK = applyAcePlannerTopK(penalized, parameters.topK);
  const topPKeep = createAcePlannerBrowserTopPKeep(topK, parameters.topP);
  const topP = topK.slice();
  for (let tokenId = 0; tokenId < topP.length; tokenId += 1) {
    if (topPKeep[tokenId] === 0) topP[tokenId] = Number.NEGATIVE_INFINITY;
  }
  const scaled = scaleTemperature(topP, parameters.temperature);
  const weights = createAcePlannerBrowserSamplingWeights(topP, parameters.temperature);
  const allowedTokens = plan.state === "regular-code" ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS;
  const filtered = createAcePlannerFilteredLogits({
    conditionalLogits: conditional,
    unconditionalLogits: unconditional,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens,
    parameters,
  });
  requireU32ArraysEqual(filtered, topP, "full filtered trace");
  const sampled = sampleAcePlannerToken({
    conditionalLogits: conditional,
    unconditionalLogits: unconditional,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens,
    parameters,
    word,
  });
  return Object.freeze({
    cfg,
    penalized,
    topK,
    topPKeep,
    topP,
    scaled,
    weights,
    tokenId: sampled.tokenId,
    word: sampled.word,
    positiveCandidateCount: sampled.positiveCandidateCount,
  });
}

function requireCompactTraceEqualsFull(
  plan: AceOpt0012CompactSemanticHeadPlan,
  compact: AceOpt0012CompactSamplingTrace,
  full: FullSamplingTrace,
): Readonly<Record<string, unknown>> {
  const first = plan.firstCandidateTokenId;
  const end = first + plan.candidateCount;
  const cfg = requireFloat32BoundaryReceipt(compact.cfgLogits, full.cfg.subarray(first, end), "CFG");
  const finalMask = requireFloat32BoundaryReceipt(compact.finalMaskedLogits, full.cfg.subarray(first, end), "final mask");
  const repetition = requireFloat32BoundaryReceipt(compact.repetitionPenalizedLogits, full.penalized.subarray(first, end), "repetition");
  const topKLogits = requireFloat32BoundaryReceipt(compact.topKLogits, full.topK.subarray(first, end), "top-k");
  const topKGlobalTokenIds = requireU32IdBoundaryReceipt(compact.topKGlobalTokenIds, finiteTokenIds(full.topK), "top-k global IDs");
  const topPKeep = requireByteBoundaryReceipt(compact.topPKeep, full.topPKeep.slice(first, end), "top-p keep");
  const topPGlobalTokenIds = requireU32IdBoundaryReceipt(compact.topPGlobalTokenIds, finiteTokenIds(full.topP), "top-p global IDs");
  const topPLogits = requireFloat32BoundaryReceipt(compact.topPLogits, full.topP.subarray(first, end), "top-p logits");
  const temperature = requireFloat32BoundaryReceipt(compact.temperatureScaledLogits, full.scaled.subarray(first, end), "temperature");
  const weights = requireFloat32BoundaryReceipt(compact.weights, full.weights.subarray(first, end), "softmax weights");
  if (
    compact.tokenId !== full.tokenId ||
    compact.word !== full.word ||
    compact.positiveCandidateCount !== full.positiveCandidateCount ||
    compact.tokenId !== plan.firstCandidateTokenId + compact.selectedCandidateIndex
  ) {
    throw new Error("OPT-0012 compact/full categorical receipt changed");
  }
  return Object.freeze({
    cfg,
    finalMask,
    repetition,
    topKLogits,
    topKGlobalTokenIds,
    topPKeep,
    topPGlobalTokenIds,
    topPLogits,
    temperature,
    weights,
    categorical: Object.freeze({
      tokenId: compact.tokenId,
      word: compact.word,
      positiveCandidateCount: compact.positiveCandidateCount,
      selectedCandidateIndex: compact.selectedCandidateIndex,
      globalMappingExact: true,
    }),
    everyBoundaryExact: true,
  });
}

function requireFloat32BoundaryReceipt(candidate: Float32Array, reference: Float32Array, label: string): Readonly<Record<string, unknown>> {
  requireU32ArraysEqual(candidate, reference, label);
  const candidateSha256 = sha256FloatWords(candidate);
  const referenceSha256 = sha256FloatWords(reference);
  if (candidateSha256 !== referenceSha256) {
    throw new Error(`OPT-0012 ${label} U32 hash differs`);
  }
  return Object.freeze({
    comparedU32Count: candidate.length,
    mismatchCount: 0,
    firstMismatchIndex: null,
    worstMismatchIndex: null,
    referenceSha256,
    candidateSha256,
    exact: true,
  });
}

function requireU32IdBoundaryReceipt(candidate: readonly number[], reference: readonly number[], label: string): Readonly<Record<string, unknown>> {
  requireNumberArraysEqual(candidate, reference, label);
  const candidateSha256 = sha256U32Le(candidate);
  const referenceSha256 = sha256U32Le(reference);
  if (candidateSha256 !== referenceSha256) {
    throw new Error(`OPT-0012 ${label} hash differs`);
  }
  return Object.freeze({
    comparedU32Count: candidate.length,
    mismatchCount: 0,
    firstMismatchIndex: null,
    worstMismatchIndex: null,
    referenceSha256,
    candidateSha256,
    exact: true,
  });
}

function requireByteBoundaryReceipt(candidate: Uint8Array, reference: Uint8Array, label: string): Readonly<Record<string, unknown>> {
  requireByteArraysEqual(candidate, reference, label);
  const candidateSha256 = aceSha256Hex(candidate);
  const referenceSha256 = aceSha256Hex(reference);
  if (candidateSha256 !== referenceSha256) {
    throw new Error(`OPT-0012 ${label} byte hash differs`);
  }
  return Object.freeze({
    comparedByteCount: candidate.length,
    mismatchCount: 0,
    firstMismatchIndex: null,
    worstMismatchIndex: null,
    referenceSha256,
    candidateSha256,
    exact: true,
  });
}

async function runPackageCorrectnessCase(
  observer: Opt0012DeviceObserver,
  executor: AcePlannerGpuExecutor,
  runner: CandidateHeadRunner,
  fixture: PlannerCaseFixture,
  plan: AceOpt0012CompactSemanticHeadPlan,
): Promise<Readonly<Record<string, unknown>>> {
  const drawIndex = plan.state === "regular-code" ? fixture.spec.drawIndex : fixture.spec.drawIndex + 1;
  const aFirst = await executePackageArm(observer, executor, runner, fixture, plan, "A", drawIndex, SENTINEL_A, false, "correctness-first");
  const aRepeat = await executePackageArm(observer, executor, runner, fixture, plan, "A", drawIndex, SENTINEL_B, false, "deterministic-rerun");
  const b = await executePackageArm(observer, executor, runner, fixture, plan, "B", drawIndex, SENTINEL_A, true, "correctness-first");
  const bRepeat = await executePackageArm(observer, executor, runner, fixture, plan, "B", drawIndex, SENTINEL_B, true, "deterministic-rerun");
  const c = await executePackageArm(observer, executor, runner, fixture, plan, "C", drawIndex, SENTINEL_A, true, "correctness-first");
  const cRepeat = await executePackageArm(observer, executor, runner, fixture, plan, "C", drawIndex, SENTINEL_B, true, "deterministic-rerun");
  if (
    aFirst.fullMappedBytes === null ||
    aRepeat.fullMappedBytes === null ||
    b.compactMappedBytes === null ||
    bRepeat.compactMappedBytes === null ||
    c.compactMappedBytes === null ||
    cRepeat.compactMappedBytes === null ||
    b.compact === null ||
    bRepeat.compact === null ||
    c.compact === null ||
    cRepeat.compact === null
  ) {
    throw new Error("OPT-0012 correctness omitted a raw readback receipt");
  }
  requireByteArraysEqual(
    new Uint8Array(aFirst.fullMappedBytes, 0, OPT_0012_FULL_HEAD_LOGIT_BYTES),
    new Uint8Array(aRepeat.fullMappedBytes, 0, OPT_0012_FULL_HEAD_LOGIT_BYTES),
    `${fixture.spec.id} A deterministic raw logits`,
  );
  requireByteArraysEqual(
    new Uint8Array(aFirst.fullMappedBytes, OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET, OPT_0012_FULL_READBACK_STATUS_BYTES),
    new Uint8Array(aRepeat.fullMappedBytes, OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET, OPT_0012_FULL_READBACK_STATUS_BYTES),
    `${fixture.spec.id} A deterministic write status`,
  );
  requireByteArraysEqual(new Uint8Array(b.compactMappedBytes), new Uint8Array(bRepeat.compactMappedBytes), `${fixture.spec.id} B deterministic readback`);
  requireByteArraysEqual(new Uint8Array(c.compactMappedBytes), new Uint8Array(cRepeat.compactMappedBytes), `${fixture.spec.id} C deterministic readback`);
  requireByteArraysEqual(new Uint8Array(b.compactMappedBytes), new Uint8Array(c.compactMappedBytes), `${fixture.spec.id} B/C common GPU readback`);
  const retainedB = compareRetainedFp16Bits(plan, aFirst.fullMappedBytes, b.compactMappedBytes);
  const retainedC = compareRetainedFp16Bits(plan, aFirst.fullMappedBytes, c.compactMappedBytes);
  const reconstructedB = compareReconstructedFullVectors(plan, aFirst.logits, b.compact);
  const reconstructedC = compareReconstructedFullVectors(plan, aFirst.logits, c.compact);
  requireSameSample(aFirst.sample, aRepeat.sample, "A rerun");
  requireSameSample(aFirst.sample, b.sample, "A/B");
  requireSameSample(b.sample, bRepeat.sample, "B rerun");
  requireSameSample(b.sample, c.sample, "B/C");
  requireSameSample(c.sample, cRepeat.sample, "C rerun");
  const fullHostDecodeCalibration = measureFullHostFp16Decode(aFirst.fullMappedBytes, aFirst.logits);
  const expectedSample =
    plan.state === "forced-eos"
      ? Object.freeze({
          tokenId: ACE_QWEN_IM_END_TOKEN_ID,
          word: aFirst.sample.word,
          positiveCandidateCount: 1,
          drawIndex: "259",
          drawEnd: "260",
        })
      : OPT_0012_OPT10_SAMPLE_AUTHORITIES[fixture.spec.id as keyof typeof OPT_0012_OPT10_SAMPLE_AUTHORITIES];
  if (expectedSample === undefined || !sameSample(aFirst.sample, expectedSample)) {
    throw new Error(`OPT-0012 ${fixture.spec.id} differs from its raw-FP16 arm-A authority`);
  }
  return Object.freeze({
    case: fixture.spec,
    state: plan.state,
    decodeTokenId: fixture.decodeTokenId,
    expectedRawFp16Sample: expectedSample,
    A: publicArmExecution(aFirst),
    ADeterministicRerun: publicArmExecution(aRepeat),
    B: publicArmExecution(b),
    BDeterministicRerun: publicArmExecution(bRepeat),
    C: publicArmExecution(c),
    CDeterministicRerun: publicArmExecution(cRepeat),
    retainedFp16AversusB: retainedB,
    retainedFp16AversusC: retainedC,
    reconstructedFullU32AversusB: reconstructedB,
    reconstructedFullU32AversusC: reconstructedC,
    untimedFullHostFp16DecodeCalibration: fullHostDecodeCalibration,
    sameCandidateMappedBytesForBAndC: true,
    exactSampleAndCursorIdentity: true,
    historicalBf16TrajectoryRole: "teacher-forced-cache-fixture-only",
  });
}

async function executePackageArm(
  observer: Opt0012DeviceObserver,
  executor: AcePlannerGpuExecutor,
  runner: CandidateHeadRunner,
  fixture: PlannerCaseFixture,
  plan: AceOpt0012CompactSemanticHeadPlan,
  arm: Opt0012Arm,
  drawIndex: number,
  sentinel: number,
  guardProof: boolean,
  purpose: string,
  roundIndex = -1,
  order = "correctness",
  orderPosition = -1,
): Promise<PackageArmExecution> {
  const cacheAppendControls = authenticateDecodeControls(fixture.decode);
  await executor.prefill(fixture.prefill);
  const statusPoison = await runner.poisonActiveWriteStatusOutsidePrimaryWall();
  const candidatePreparation = arm === "A" ? null : await runner.prepareCandidateResourcesOutsidePrimaryWall(plan, sentinel);
  const captureFullReadback = arm === "A" && purpose !== "timing";
  observer.beginTrace(`${fixture.spec.id}-${plan.state}-${arm}-${purpose}`, captureFullReadback);
  const wallStarted = performance.now();
  const startedAtEpochMilliseconds = Date.now();
  let returned: readonly ArrayLike<number>[];
  let compact: AceOpt0012DecodedCompactLogits | null = null;
  let compactMappedBytes: ArrayBuffer | null = null;
  let guard: Readonly<Record<string, unknown>> | null = null;
  let commonCandidateReplay: Readonly<Record<string, unknown>> | null = null;
  let reconstructionMilliseconds = 0;
  let samplingMilliseconds = 0;
  let candidateHostDecodeMilliseconds: number | null = null;
  try {
    const decodeStarted = performance.now();
    returned = await runner.invoke(() => executor.decode(fixture.decode), {
      arm,
      phaseKind: "decode",
      plan,
      sentinel,
      guardProof,
      cancellationBoundary: null,
      boundaryAbortController: null,
      allowCorrectnessOnlyInsideTracePreparation: false,
    });
    const decodeEnded = performance.now();
    const explicitProgress = runner.takeProgressReceipt(arm === "A" ? OPT_0012_FULL_COMMAND_BUFFER_COUNT : OPT_0012_CANDIDATE_COMMAND_BUFFER_COUNT, true);
    const candidateReceipt = arm === "A" ? null : runner.takeReceipt();
    if (candidateReceipt !== null) {
      if (candidateReceipt.preparation !== candidatePreparation) {
        throw new Error("OPT-0012 candidate preparation receipt identity changed");
      }
      compact = candidateReceipt.decoded;
      compactMappedBytes = candidateReceipt.mappedBytes;
      candidateHostDecodeMilliseconds = candidateReceipt.hostDecodeMilliseconds;
    }
    const logits = requireFloat32Rows(returned, arm === "A" ? plan.vocabularySize : plan.candidateCount);
    let sampling: TimedSamplingExecution;
    if (arm === "C") {
      sampling = sampleCompactRows(plan, compact!, fixture.seenTokenIds, drawIndex);
    } else {
      let fullRows = logits;
      if (arm === "B") {
        const reconstructionStarted = performance.now();
        fullRows = reconstructAceOpt0012FullPlannerLogits(compact!, plan);
        reconstructionMilliseconds = performance.now() - reconstructionStarted;
      }
      sampling = sampleFullRows(plan, fullRows, fixture.seenTokenIds, drawIndex);
    }
    const sample = sampling.sample;
    samplingMilliseconds = sampling.intervals.samplingMilliseconds;
    if (purpose !== "timing") {
      const unchangedAuthority =
        arm === "C"
          ? sampleCompactRowsAuthority(plan, compact!, fixture.seenTokenIds, drawIndex)
          : sampleFullRowsAuthority(plan, arm === "B" ? reconstructAceOpt0012FullPlannerLogits(compact!, plan) : logits, fixture.seenTokenIds, drawIndex);
      requireSameSample(sample, unchangedAuthority, `${arm} decomposed/unchanged browser-v1 sampler`);
    }
    const wallEnded = performance.now();
    const observedTrace = observer.endTrace({
      arm,
      phaseKind: "decode",
      state: plan.state,
      expectedPhysicalDispatchCount:
        arm === "A"
          ? OPT_0012_FULL_PHYSICAL_DISPATCH_COUNT
          : plan.state === "regular-code"
            ? OPT_0012_REGULAR_PHYSICAL_DISPATCH_COUNT
            : OPT_0012_EOS_PHYSICAL_DISPATCH_COUNT,
      expectedCopyCount: arm === "A" ? OPT_0012_FULL_COPY_COUNT : plan.readback.copies.length,
      expectedCommandBufferCount: arm === "A" ? OPT_0012_FULL_COMMAND_BUFFER_COUNT : OPT_0012_CANDIDATE_COMMAND_BUFFER_COUNT,
      roundIndex,
      order,
      orderPosition,
      startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Date.now(),
      wallMilliseconds: wallEnded - wallStarted,
      decodeMilliseconds: decodeEnded - decodeStarted,
      reconstructionMilliseconds,
      samplingMilliseconds,
      callbackMilliseconds: sampling.intervals.callbackMilliseconds,
      samplingStageIntervals: sampling.intervals,
      decodeStartedAt: decodeStarted,
      decodeEndedAt: decodeEnded,
      candidateHostDecodeMilliseconds,
      readbackIdleInterval: requireCompleteReadbackIdle(explicitProgress),
      expectedEncodedFreshStatusClears: Object.freeze([]),
    });
    const trace = Object.freeze({
      ...observedTrace,
      explicitProgress,
      statusFreshness: Object.freeze({
        preInvocationZeroed: true,
        poisonAndFence: statusPoison,
        requiredSuccessWords: Object.freeze([1, 1]),
        excludedFromPrimaryWall: true,
      }),
      cacheAppendControls,
      candidatePreparation,
      candidateAllocationDispatchPreparationExcludedFromPrimaryWall: arm === "A" || candidatePreparation !== null,
    });
    const fullMappedBytes = captureFullReadback ? observer.takeFullReadbackBytes() : null;
    const fp16NaNCensus =
      purpose === "timing"
        ? null
        : arm === "A"
          ? fullMappedBytes === null
            ? requireZeroOpt0012DecodedFullFp16NaNs(logits, `${fixture.spec.id} ${arm} ${purpose}`, plan.state)
            : requireZeroOpt0012RawFullFp16NaNs(fullMappedBytes, `${fixture.spec.id} ${arm} ${purpose}`, plan.state)
          : requireZeroOpt0012RawCompactFp16NaNs(plan, compactMappedBytes!, `${fixture.spec.id} ${arm} ${purpose}`);
    // Common-byte B/C attribution is deliberately outside the primary token
    // wall and GPU trace. It replays both CPU representations over the exact
    // same retained mapped bytes and categorical word.
    if (compact !== null && purpose !== "timing") {
      const replayStarted = performance.now();
      const comparison = compareFullAndCompactSampling(
        plan,
        compact.conditionalLogits,
        compact.unconditionalLogits,
        fixture.seenTokenIds,
        ACE_OPT_0012_SAMPLING_PARAMETERS,
        sample.word,
      );
      commonCandidateReplay = Object.freeze({
        ...comparison,
        replayMilliseconds: performance.now() - replayStarted,
        sameMappedBytesAuthority: true,
        excludedFromPrimaryTokenWall: true,
      });
    }
    if (guardProof) guard = await runner.verifyAndReleasePendingGuards();
    else runner.releasePendingGuardsWithoutReadback();
    return Object.freeze({
      arm,
      state: plan.state,
      logits,
      compact,
      compactMappedBytes,
      fullMappedBytes,
      sample,
      trace,
      fp16NaNCensus,
      guard,
      commonCandidateReplay,
      intervals: Object.freeze({
        totalWallMilliseconds: wallEnded - wallStarted,
        decodeMilliseconds: decodeEnded - decodeStarted,
        reconstructionMilliseconds,
        samplingMilliseconds,
        preCfgConstraintMilliseconds: sampling.intervals.preCfgConstraintMilliseconds,
        cfgMilliseconds: sampling.intervals.cfgMilliseconds,
        postCfgConstraintAndRepetitionMilliseconds: sampling.intervals.postCfgConstraintAndRepetitionMilliseconds,
        constraintMilliseconds: sampling.intervals.constraintMilliseconds,
        topKMilliseconds: sampling.intervals.topKMilliseconds,
        topPMilliseconds: sampling.intervals.topPMilliseconds,
        temperatureAndSoftmaxMilliseconds: sampling.intervals.temperatureAndSoftmaxMilliseconds,
        categoricalWordAndGlobalMappingMilliseconds: sampling.intervals.categoricalWordAndGlobalMappingMilliseconds,
        callbackMilliseconds: sampling.intervals.callbackMilliseconds,
      }),
    });
  } catch (error) {
    observer.abandonTrace();
    runner.releasePendingGuardsWithoutReadback();
    throw error;
  }
}

interface SamplingStageIntervals {
  readonly samplingMilliseconds: number;
  readonly preCfgConstraintMilliseconds: number;
  readonly cfgMilliseconds: number;
  readonly postCfgConstraintAndRepetitionMilliseconds: number;
  readonly constraintMilliseconds: number;
  readonly topKMilliseconds: number;
  readonly topPMilliseconds: number;
  readonly temperatureAndSoftmaxMilliseconds: number;
  readonly categoricalWordAndGlobalMappingMilliseconds: number;
  readonly callbackMilliseconds: number;
  readonly callbackInvocationCount: number;
}

interface TimedSamplingExecution {
  readonly sample: SampleReceipt;
  readonly intervals: SamplingStageIntervals;
}

const NO_SAMPLING_STAGE_INTERVALS: SamplingStageIntervals = Object.freeze({
  samplingMilliseconds: 0,
  preCfgConstraintMilliseconds: 0,
  cfgMilliseconds: 0,
  postCfgConstraintAndRepetitionMilliseconds: 0,
  constraintMilliseconds: 0,
  topKMilliseconds: 0,
  topPMilliseconds: 0,
  temperatureAndSoftmaxMilliseconds: 0,
  categoricalWordAndGlobalMappingMilliseconds: 0,
  callbackMilliseconds: 0,
  callbackInvocationCount: 0,
});

function sampleFullRows(
  plan: AceOpt0012CompactSemanticHeadPlan,
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  drawIndex: number | bigint,
): TimedSamplingExecution {
  if (rows.length !== 2 || rows[0]!.length !== plan.vocabularySize) {
    throw new Error("OPT-0012 full sampler received compact rows");
  }
  const samplingStarted = performance.now();
  let stageStarted = performance.now();
  const preCfgAllowedTokens = ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS;
  const allowedTokens = plan.state === "regular-code" ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS;
  const conditionalAllowed = maskAcePlannerLogits(rows[0]!, preCfgAllowedTokens);
  const unconditionalAllowed = maskAcePlannerLogits(rows[1]!, preCfgAllowedTokens);
  const preCfgConstraintMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  let filtered = combineOpt0012AllowedCfgForTiming(conditionalAllowed, unconditionalAllowed, ACE_OPT_0012_SAMPLING_PARAMETERS.guidanceScale);
  const cfgMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  filtered = maskAcePlannerLogits(filtered, allowedTokens);
  filtered = applyAcePlannerRepetitionPenalty(filtered, seenTokenIds, ACE_OPT_0012_SAMPLING_PARAMETERS.repetitionPenalty);
  const postCfgConstraintAndRepetitionMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  filtered = applyAcePlannerTopK(filtered, ACE_OPT_0012_SAMPLING_PARAMETERS.topK);
  const topKMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  filtered = applyOpt0012BrowserTopPForTiming(filtered, ACE_OPT_0012_SAMPLING_PARAMETERS.topP);
  const topPMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const weights = createAcePlannerBrowserSamplingWeights(filtered, ACE_OPT_0012_SAMPLING_PARAMETERS.temperature);
  const temperatureAndSoftmaxMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const draw = requireSamplingDrawIndex(drawIndex);
  const word = aceRandomWord(ACCEPTED_SEED, "planner-sampling", draw);
  const tokenId = aceCategoricalTokenFromWord(weights, word);
  const positiveCandidateCount = countPositiveWeights(weights);
  const categoricalWordAndGlobalMappingMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const sample = publishOpt0012SampleCallback({
    tokenId,
    word,
    positiveCandidateCount,
    drawIndex: draw.toString(),
    drawEnd: (draw + 1n).toString(),
  });
  const callbackMilliseconds = performance.now() - stageStarted;
  const samplingMilliseconds = performance.now() - samplingStarted;
  return Object.freeze({
    sample,
    intervals: createSamplingStageIntervals({
      samplingMilliseconds,
      preCfgConstraintMilliseconds,
      cfgMilliseconds,
      postCfgConstraintAndRepetitionMilliseconds,
      topKMilliseconds,
      topPMilliseconds,
      temperatureAndSoftmaxMilliseconds,
      categoricalWordAndGlobalMappingMilliseconds,
      callbackMilliseconds,
    }),
  });
}

function sampleFullRowsAuthority(
  plan: AceOpt0012CompactSemanticHeadPlan,
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  drawIndex: number | bigint,
): SampleReceipt {
  const cursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const sampled = cursor.sample({
    conditionalLogits: rows[0]!,
    unconditionalLogits: rows[1]!,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens: plan.state === "regular-code" ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
  });
  return Object.freeze({
    tokenId: sampled.tokenId,
    word: sampled.word,
    positiveCandidateCount: sampled.positiveCandidateCount,
    drawIndex: sampled.drawIndex.toString(),
    drawEnd: cursor.consumed.toString(),
  });
}

function sampleCompactRows(
  plan: AceOpt0012CompactSemanticHeadPlan,
  decoded: AceOpt0012DecodedCompactLogits,
  seenTokenIds: readonly number[],
  drawIndex: number | bigint,
): TimedSamplingExecution {
  const samplingStarted = performance.now();
  let stageStarted = performance.now();
  const conditionalAllowed = maskAcePlannerLogits(decoded.conditionalLogits, { kind: "all" });
  const unconditionalAllowed = maskAcePlannerLogits(decoded.unconditionalLogits, { kind: "all" });
  const preCfgConstraintMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  let filtered = combineAcePlannerCfgLogits(conditionalAllowed, unconditionalAllowed, ACE_OPT_0012_SAMPLING_PARAMETERS.guidanceScale);
  const cfgMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  filtered = maskAcePlannerLogits(filtered, { kind: "all" });
  filtered = applyAcePlannerRepetitionPenalty(
    filtered,
    mapOpt0012SeenTokensToCandidateDomain(seenTokenIds, plan),
    ACE_OPT_0012_SAMPLING_PARAMETERS.repetitionPenalty,
  );
  const postCfgConstraintAndRepetitionMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const compactTopK = ACE_OPT_0012_SAMPLING_PARAMETERS.topK >= plan.candidateCount ? 0 : ACE_OPT_0012_SAMPLING_PARAMETERS.topK;
  filtered = applyAcePlannerTopK(filtered, compactTopK);
  const topKMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  filtered = applyOpt0012BrowserTopPForTiming(filtered, ACE_OPT_0012_SAMPLING_PARAMETERS.topP);
  const topPMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const weights = createAcePlannerBrowserSamplingWeights(filtered, ACE_OPT_0012_SAMPLING_PARAMETERS.temperature);
  const temperatureAndSoftmaxMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const draw = requireSamplingDrawIndex(drawIndex);
  const word = aceRandomWord(ACCEPTED_SEED, "planner-sampling", draw);
  const selectedCandidateIndex = aceCategoricalTokenFromWord(weights, word);
  const tokenId = plan.firstCandidateTokenId + selectedCandidateIndex;
  const positiveCandidateCount = countPositiveWeights(weights);
  const categoricalWordAndGlobalMappingMilliseconds = performance.now() - stageStarted;

  stageStarted = performance.now();
  const sample = publishOpt0012SampleCallback({
    tokenId,
    word,
    positiveCandidateCount,
    drawIndex: draw.toString(),
    drawEnd: (draw + 1n).toString(),
  });
  const callbackMilliseconds = performance.now() - stageStarted;
  const samplingMilliseconds = performance.now() - samplingStarted;
  return Object.freeze({
    sample,
    intervals: createSamplingStageIntervals({
      samplingMilliseconds,
      preCfgConstraintMilliseconds,
      cfgMilliseconds,
      postCfgConstraintAndRepetitionMilliseconds,
      topKMilliseconds,
      topPMilliseconds,
      temperatureAndSoftmaxMilliseconds,
      categoricalWordAndGlobalMappingMilliseconds,
      callbackMilliseconds,
    }),
  });
}

function sampleCompactRowsAuthority(
  plan: AceOpt0012CompactSemanticHeadPlan,
  decoded: AceOpt0012DecodedCompactLogits,
  seenTokenIds: readonly number[],
  drawIndex: number | bigint,
): SampleReceipt {
  const cursor = new AceOpt0012CompactSamplingCursor(ACCEPTED_SEED, drawIndex);
  const sampled = cursor.sample({
    plan,
    conditionalLogits: decoded.conditionalLogits,
    unconditionalLogits: decoded.unconditionalLogits,
    seenTokenIds,
  });
  return Object.freeze({
    tokenId: sampled.tokenId,
    word: sampled.word,
    positiveCandidateCount: sampled.positiveCandidateCount,
    drawIndex: sampled.drawIndex.toString(),
    drawEnd: sampled.drawEnd.toString(),
  });
}

function combineOpt0012AllowedCfgForTiming(conditional: Float32Array, unconditional: Float32Array, guidanceScale: number): Float32Array {
  if (conditional.length !== unconditional.length) {
    throw new Error("OPT-0012 timed CFG row lengths changed");
  }
  const scale = Math.fround(guidanceScale);
  const output = new Float32Array(conditional.length);
  output.fill(Number.NEGATIVE_INFINITY);
  let finiteCandidateCount = 0;
  for (let tokenId = 0; tokenId < output.length; tokenId += 1) {
    const conditionalValue = conditional[tokenId]!;
    const unconditionalValue = unconditional[tokenId]!;
    if (conditionalValue === Number.NEGATIVE_INFINITY && unconditionalValue === Number.NEGATIVE_INFINITY) continue;
    if (!Number.isFinite(conditionalValue) || !Number.isFinite(unconditionalValue)) {
      throw new Error("OPT-0012 timed CFG allowed subspaces differ");
    }
    output[tokenId] = Math.fround(unconditionalValue + Math.fround(scale * Math.fround(conditionalValue - unconditionalValue)));
    finiteCandidateCount += 1;
  }
  if (finiteCandidateCount === 0) {
    throw new Error("OPT-0012 timed CFG produced no finite candidates");
  }
  return output;
}

function applyOpt0012BrowserTopPForTiming(logits: Float32Array, topP: number): Float32Array {
  const output = logits.slice();
  const keep = createAcePlannerBrowserTopPKeep(output, topP);
  if (keep.length !== output.length) {
    throw new Error("OPT-0012 timed top-p keep shape changed");
  }
  for (let index = 0; index < output.length; index += 1) {
    if (keep[index] === 0) output[index] = Number.NEGATIVE_INFINITY;
  }
  return output;
}

function mapOpt0012SeenTokensToCandidateDomain(seenTokenIds: readonly number[], plan: AceOpt0012CompactSemanticHeadPlan): number[] {
  const local: number[] = [];
  for (const tokenId of seenTokenIds) {
    if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId >= plan.vocabularySize) {
      throw new Error(`OPT-0012 timed sampler saw invalid token ${String(tokenId)}`);
    }
    const candidateIndex = tokenId - plan.firstCandidateTokenId;
    if (candidateIndex >= 0 && candidateIndex < plan.candidateCount) {
      local.push(candidateIndex);
    }
  }
  return local;
}

function requireSamplingDrawIndex(drawIndex: number | bigint): bigint {
  const draw = typeof drawIndex === "bigint" ? drawIndex : BigInt(drawIndex);
  if (draw < 0n || (typeof drawIndex === "number" && !Number.isSafeInteger(drawIndex))) {
    throw new Error("OPT-0012 timed sampler draw index changed");
  }
  return draw;
}

function countPositiveWeights(weights: Float32Array): number {
  let count = 0;
  for (const weight of weights) if (weight > 0) count += 1;
  if (count === 0) throw new Error("OPT-0012 timed sampler has no probability mass");
  return count;
}

function publishOpt0012SampleCallback(input: SampleReceipt): SampleReceipt {
  return Object.freeze({ ...input });
}

function createSamplingStageIntervals(input: Omit<SamplingStageIntervals, "constraintMilliseconds" | "callbackInvocationCount">): SamplingStageIntervals {
  const values = Object.values(input);
  if (values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("OPT-0012 timed sampling interval is invalid");
  }
  return Object.freeze({
    ...input,
    constraintMilliseconds: input.preCfgConstraintMilliseconds + input.postCfgConstraintAndRepetitionMilliseconds,
    callbackInvocationCount: 1,
  });
}

function compareRetainedFp16Bits(
  plan: AceOpt0012CompactSemanticHeadPlan,
  fullMappedBytes: ArrayBuffer,
  compactMappedBytes: ArrayBuffer,
): Readonly<Record<string, unknown>> {
  const full = extractFullRetainedFp16Bits(plan, fullMappedBytes);
  const compact = extractCompactLogicalFp16Bits(plan, compactMappedBytes);
  let mismatchCount = 0;
  let firstMismatchRow: number | null = null;
  let firstMismatchCandidate: number | null = null;
  for (let index = 0; index < full.length; index += 1) {
    if (full[index] !== compact[index]) {
      mismatchCount += 1;
      if (firstMismatchRow === null) {
        firstMismatchRow = Math.floor(index / plan.candidateCount);
        firstMismatchCandidate = index % plan.candidateCount;
      }
    }
  }
  if (mismatchCount !== 0) {
    throw new Error("OPT-0012 retained candidate FP16 bits differ from arm A");
  }
  return Object.freeze({
    comparedU16Count: full.length,
    mismatchCount,
    firstMismatchRow,
    firstMismatchCandidate,
    firstMismatchGlobalTokenId: firstMismatchCandidate === null ? null : plan.firstCandidateTokenId + firstMismatchCandidate,
    worstMismatchRow: null,
    worstMismatchCandidate: null,
    worstMismatchGlobalTokenId: null,
    referenceSha256: sha256U16Le(full),
    candidateSha256: sha256U16Le(compact),
    bitExact: true,
  });
}

function compareReconstructedFullVectors(
  plan: AceOpt0012CompactSemanticHeadPlan,
  armA: readonly Float32Array[],
  compact: AceOpt0012DecodedCompactLogits,
): Readonly<Record<string, unknown>> {
  const reconstructed = reconstructAceOpt0012FullPlannerLogits(compact, plan);
  let comparedU32Count = 0;
  let mismatchCount = 0;
  let firstMismatchRow: number | null = null;
  let firstMismatchTokenId: number | null = null;
  const referenceWords = new Uint32Array(2 * plan.vocabularySize);
  const candidateWords = new Uint32Array(2 * plan.vocabularySize);
  for (let row = 0; row < 2; row += 1) {
    for (let tokenId = 0; tokenId < plan.vocabularySize; tokenId += 1) {
      const admitted = tokenId >= plan.firstCandidateTokenId && tokenId <= plan.lastCandidateTokenId;
      const expected = admitted ? floatWord(armA[row]![tokenId]!) : ACE_OPT_0012_NEGATIVE_INFINITY_U32;
      const actual = floatWord(reconstructed[row]![tokenId]!);
      referenceWords[comparedU32Count] = expected;
      candidateWords[comparedU32Count] = actual;
      comparedU32Count += 1;
      if (actual !== expected) {
        mismatchCount += 1;
        if (firstMismatchRow === null) {
          firstMismatchRow = row;
          firstMismatchTokenId = tokenId;
        }
      }
    }
  }
  if (mismatchCount !== 0) {
    throw new Error("OPT-0012 reconstructed vector differs from masked arm A");
  }
  return Object.freeze({
    comparedU32Count,
    mismatchCount,
    firstMismatchRow,
    firstMismatchTokenId,
    worstMismatchRow: null,
    worstMismatchTokenId: null,
    maskedArmAReferenceSha256: sha256U32Words(referenceWords),
    reconstructedCandidateSha256: sha256U32Words(candidateWords),
    everyOmittedEntryNegativeInfinityU32: true,
    bitExact: true,
  });
}

function extractFullRetainedFp16Bits(plan: AceOpt0012CompactSemanticHeadPlan, mapped: ArrayBuffer): Uint16Array {
  if (mapped.byteLength !== OPT_0012_FULL_READBACK_ALLOCATION_BYTES) {
    throw new Error("OPT-0012 full mapped readback allocation changed");
  }
  const shardOffsets: number[] = [];
  let cursor = 0;
  for (const shard of ACE_PLANNER_EMBEDDING_ROW_PARTS) {
    shardOffsets.push(cursor);
    cursor += align4(2 * shard.rowCount * Uint16Array.BYTES_PER_ELEMENT);
  }
  if (cursor !== OPT_0012_FULL_HEAD_LOGIT_BYTES) {
    throw new Error("OPT-0012 full readback shard accounting changed");
  }
  const output = new Uint16Array(2 * plan.candidateCount);
  for (const intersection of plan.intersections) {
    const shard = ACE_PLANNER_EMBEDDING_ROW_PARTS[intersection.shardIndex]!;
    const base = shardOffsets[intersection.shardIndex]!;
    for (let row = 0; row < 2; row += 1) {
      const source = new Uint16Array(mapped, base + (row * shard.rowCount + intersection.localFirstRow) * 2, intersection.rowCount);
      output.set(source, row * plan.candidateCount + intersection.globalFirstRow - plan.firstCandidateTokenId);
    }
  }
  return output;
}

function measureFullHostFp16Decode(mapped: ArrayBuffer, expected: readonly Float32Array[]): Readonly<Record<string, unknown>> {
  let cursor = 0;
  const shards = ACE_PLANNER_EMBEDDING_ROW_PARTS.map((shard, sourceShardIndex) => {
    cursor = align256(cursor);
    const byteLength = 2 * shard.rowCount * Uint16Array.BYTES_PER_ELEMENT;
    const receipt = Object.freeze({
      sourceShardIndex,
      firstRow: shard.firstRow,
      destinationFirstRow: shard.firstRow,
      rowCount: shard.rowCount,
      byteOffset: cursor,
      byteLength,
    });
    cursor += align4(byteLength);
    return receipt;
  });
  if (cursor !== OPT_0012_FULL_HEAD_LOGIT_BYTES) {
    throw new Error("OPT-0012 full host-decode shard layout changed");
  }
  const layout: AcePlannerLogitReadbackLayout = Object.freeze({
    rows: 2,
    firstTokenId: 0,
    tokenCount: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    shards: Object.freeze(shards),
    writeStatusByteOffset: OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET,
    writeStatusByteLength: OPT_0012_FULL_READBACK_STATUS_BYTES,
    byteLength: OPT_0012_FULL_READBACK_ALLOCATION_BYTES,
  });
  const started = performance.now();
  const decoded = reconstructAcePlannerLogits(mapped, layout, "raw-fp16");
  const milliseconds = performance.now() - started;
  const rows = requireFloat32Rows(decoded, ACE_PLANNER_QWEN3_CONFIG.vocabularySize);
  requireU32ArraysEqual(rows[0]!, expected[0]!, "untimed full host decode row 0");
  requireU32ArraysEqual(rows[1]!, expected[1]!, "untimed full host decode row 1");
  return Object.freeze({
    milliseconds,
    rowCount: 2,
    vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    comparedU32Count: 2 * ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    bitExactToPrimaryArmA: true,
    excludedFromPrimaryTokenTiming: true,
  });
}

function extractCompactLogicalFp16Bits(plan: AceOpt0012CompactSemanticHeadPlan, mapped: ArrayBuffer): Uint16Array {
  if (mapped.byteLength !== plan.readback.allocationBytes) {
    throw new Error("OPT-0012 compact mapped allocation changed");
  }
  const output = new Uint16Array(2 * plan.candidateCount);
  for (const span of plan.readback.logicalSpans) {
    output.set(new Uint16Array(mapped, span.sourceByteOffset, span.candidateCount), span.physicalRow * plan.candidateCount + span.destinationCandidateOffset);
  }
  return output;
}

function requireZeroOpt0012RawFullFp16NaNs(mapped: ArrayBuffer, label: string, state: AceOpt0012SemanticState): Readonly<Record<string, unknown>> {
  if (mapped.byteLength !== OPT_0012_FULL_READBACK_ALLOCATION_BYTES) {
    throw new Error("OPT-0012 full FP16 NaN census allocation changed");
  }
  return requireZeroOpt0012RawFp16NaNs(
    new Uint16Array(mapped, 0, OPT_0012_FULL_HEAD_LOGIT_BYTES / Uint16Array.BYTES_PER_ELEMENT),
    label,
    state,
    "raw-full-vocabulary-readback-U16",
  );
}

function requireZeroOpt0012RawCompactFp16NaNs(plan: AceOpt0012CompactSemanticHeadPlan, mapped: ArrayBuffer, label: string): Readonly<Record<string, unknown>> {
  return requireZeroOpt0012RawFp16NaNs(extractCompactLogicalFp16Bits(plan, mapped), label, plan.state, "raw-compact-logical-span-readback-U16");
}

function requireZeroOpt0012RawFp16NaNs(
  words: Uint16Array,
  label: string,
  state: AceOpt0012SemanticState,
  authority: string,
): Readonly<Record<string, unknown>> {
  let nanCount = 0;
  let positiveSignalingCount = 0;
  let positiveQuietCount = 0;
  let negativeSignalingCount = 0;
  let negativeQuietCount = 0;
  let firstNanWordIndex: number | null = null;
  let firstNanFp16U16: number | null = null;
  for (let index = 0; index < words.length; index += 1) {
    const bits = words[index]!;
    const mantissa = bits & 0x03ff;
    if (((bits >>> 10) & 0x1f) !== 0x1f || mantissa === 0) continue;
    nanCount += 1;
    firstNanWordIndex ??= index;
    firstNanFp16U16 ??= bits;
    const quiet = (mantissa & 0x0200) !== 0;
    const positive = (bits & 0x8000) === 0;
    if (positive && quiet) positiveQuietCount += 1;
    else if (positive) positiveSignalingCount += 1;
    else if (quiet) negativeQuietCount += 1;
    else negativeSignalingCount += 1;
  }
  const receipt = Object.freeze({
    label,
    state,
    authority,
    inspectedBinary16WordCount: words.length,
    rawBinary16U16LeSha256: sha256U16Le(words),
    binary16NaNCount: nanCount,
    firstNanWordIndex,
    firstNanFp16U16,
    nanClasses: Object.freeze({
      positiveSignaling: positiveSignalingCount,
      positiveQuiet: positiveQuietCount,
      negativeSignaling: negativeSignalingCount,
      negativeQuiet: negativeQuietCount,
    }),
    actualPackageReadback: true,
    zeroBinary16NaNs: nanCount === 0,
    excludedFromPrimaryTimingWall: true,
  });
  if (nanCount !== 0) {
    throw new Error("OPT-0012 actual package readback contains FP16 NaN: " + JSON.stringify(receipt));
  }
  return receipt;
}

function requireZeroOpt0012DecodedFullFp16NaNs(
  rows: readonly Float32Array[],
  label: string,
  state: AceOpt0012SemanticState,
): Readonly<Record<string, unknown>> {
  if (rows.length !== 2 || rows.some((row) => row.length !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize)) {
    throw new Error("OPT-0012 decoded full FP16 NaN census shape changed");
  }
  let nanCount = 0;
  let firstNanRow: number | null = null;
  let firstNanTokenId: number | null = null;
  for (let row = 0; row < rows.length; row += 1) {
    for (let tokenId = 0; tokenId < rows[row]!.length; tokenId += 1) {
      if (!Number.isNaN(rows[row]![tokenId])) continue;
      nanCount += 1;
      firstNanRow ??= row;
      firstNanTokenId ??= tokenId;
    }
  }
  const receipt = Object.freeze({
    label,
    state,
    authority: "lossless-full-FP16-readback-decode-NaN-classification",
    inspectedBinary16WordCount: 2 * ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    binary16NaNCount: nanCount,
    firstNanRow,
    firstNanTokenId,
    rawBinary16WordsRetained: false,
    everyBinary16NaNWouldDecodeToNumberNaN: true,
    actualPackageReadback: true,
    zeroBinary16NaNs: nanCount === 0,
    excludedFromPrimaryTimingWall: true,
  });
  if (nanCount !== 0) {
    throw new Error("OPT-0012 decoded full package readback contains FP16 NaN: " + JSON.stringify(receipt));
  }
  return receipt;
}

function publicArmExecution(execution: PackageArmExecution): Readonly<Record<string, unknown>> {
  return Object.freeze({
    arm: execution.arm,
    state: execution.state,
    sample: execution.sample,
    trace: execution.trace,
    fp16NaNCensus:
      execution.fp16NaNCensus ??
      Object.freeze({
        status: "deferred-until-after-entire-primary-timing-window",
        authority: "primaryTimingFp16NaNCensuses",
        join: Object.freeze({
          traceLabel: execution.trace["traceLabel"],
          arm: execution.arm,
          roundIndex: execution.trace["roundIndex"],
          order: execution.trace["order"],
          orderPosition: execution.trace["orderPosition"],
        }),
      }),
    guard: execution.guard,
    commonCandidateReplay: execution.commonCandidateReplay,
    intervals: execution.intervals,
    fullReadbackSha256: execution.fullMappedBytes === null ? null : sha256FullSemanticReadback(execution.fullMappedBytes),
    compactReadbackSha256: execution.compactMappedBytes === null ? null : aceSha256Hex(new Uint8Array(execution.compactMappedBytes)),
  });
}

async function runSameImmutableByteReplayCase(
  plan: AceOpt0012CompactSemanticHeadPlan,
  fixture: PlannerCaseFixture,
  b: PackageArmExecution,
  c: PackageArmExecution,
  authority: SampleReceipt,
): Promise<Readonly<Record<string, unknown>>> {
  if (b.compactMappedBytes === null || c.compactMappedBytes === null) {
    throw new Error("OPT-0012 same-byte replay omitted candidate bytes");
  }
  requireByteArraysEqual(new Uint8Array(b.compactMappedBytes), new Uint8Array(c.compactMappedBytes), `${fixture.spec.id} B/C same-byte source`);
  const mappedBytes = b.compactMappedBytes;
  const sourceSha256 = aceSha256Hex(new Uint8Array(mappedBytes));
  const fp16NaNCensus = requireZeroOpt0012RawCompactFp16NaNs(plan, mappedBytes, `${fixture.spec.id} same-immutable-byte replay`);
  const pairs: SameByteReplayPair[] = [];
  for (let pairIndex = 0; pairIndex < OPT_0012_REPLAY_BALANCED_ORDERS.length; pairIndex += 1) {
    const order = OPT_0012_REPLAY_BALANCED_ORDERS[pairIndex]!;
    const before = aceSha256Hex(new Uint8Array(mappedBytes));
    let bMeasured: SameByteReplayMeasuredExecution | undefined;
    let cMeasured: SameByteReplayMeasuredExecution | undefined;
    // Deliberately no await/yield between the two arms in a pair.
    for (const arm of order) {
      const measured = runSameByteReplayExecution(arm, mappedBytes, plan, fixture);
      if (arm === "B") bMeasured = measured;
      else cMeasured = measured;
    }
    if (bMeasured === undefined || cMeasured === undefined) {
      throw new Error("OPT-0012 same-byte replay pair omitted an arm");
    }
    requireSameSample(bMeasured.timing.sample, authority, `${fixture.spec.id} replay pair ${pairIndex} B authority`);
    requireSameSample(cMeasured.timing.sample, authority, `${fixture.spec.id} replay pair ${pairIndex} C authority`);
    requireSameSample(bMeasured.timing.sample, cMeasured.timing.sample, `${fixture.spec.id} replay pair ${pairIndex} B/C`);
    // Hashing and receipt expansion start only after both timed arms finish.
    const bExecution = finalizeSameByteReplayExecution(bMeasured);
    const cExecution = finalizeSameByteReplayExecution(cMeasured);
    const after = aceSha256Hex(new Uint8Array(mappedBytes));
    if (before !== sourceSha256 || after !== sourceSha256) {
      throw new Error("OPT-0012 same-byte replay mutated its mapped input");
    }
    pairs.push(
      Object.freeze({
        pairIndex,
        order,
        mappedBytesSha256Before: before,
        mappedBytesSha256After: after,
        B: bExecution,
        C: cExecution,
        sameImmutableBytes: true,
        noYieldInsidePair: true,
      }),
    );
    await yieldToBrowser();
  }
  const decoded = decodeAceOpt0012CompactFp16Readback(mappedBytes, plan);
  const u32Equivalence = compareFullAndCompactSampling(
    plan,
    decoded.conditionalLogits,
    decoded.unconditionalLogits,
    fixture.seenTokenIds,
    ACE_OPT_0012_SAMPLING_PARAMETERS,
    authority.word,
  );
  if (
    u32Equivalence["tokenId"] !== authority.tokenId ||
    u32Equivalence["word"] !== authority.word ||
    aceSha256Hex(new Uint8Array(mappedBytes)) !== sourceSha256
  ) {
    throw new Error("OPT-0012 same-byte replay equivalence authority changed");
  }
  return Object.freeze({
    case: fixture.spec,
    protocol: Object.freeze({
      orders: OPT_0012_REPLAY_BALANCED_ORDERS,
      pairCount: OPT_0012_REPLAY_BALANCED_ORDERS.length,
      commonGpuWorkExcluded: true,
      sameArrayBufferObjectForEveryArmAndPair: true,
      noYieldInsidePair: true,
      yieldOnlyAfterCompletedPair: true,
      hashingExcludedFromArmTiming: true,
      noHashingOrEquivalenceWorkBetweenTimedPairArms: true,
      timingAuthority: "synchronous-performance-now-mapped-bytes-to-sample-wall",
      commonInputFp16NaNCensusExcludedFromEveryArmTiming: true,
    }),
    commonInput: Object.freeze({
      mappedByteLength: mappedBytes.byteLength,
      mappedBytesSha256: sourceSha256,
      planState: plan.state,
      samplerWord: authority.word,
      drawIndex: authority.drawIndex,
      drawEnd: authority.drawEnd,
      fp16NaNCensus,
    }),
    pairs: Object.freeze(pairs),
    B: summarizeSameByteReplayArm(pairs, "B"),
    C: summarizeSameByteReplayArm(pairs, "C"),
    samePairComparison: summarizeSameByteReplayPairs(pairs),
    u32Equivalence: Object.freeze({
      ...u32Equivalence,
      decodedConditionalSha256: sha256FloatWords(decoded.conditionalLogits),
      decodedUnconditionalSha256: sha256FloatWords(decoded.unconditionalLogits),
      appliesToEveryPairBecauseInputPlanSeenTokensAndWordAreIdentical: true,
      excludedFromEveryReplayArmTiming: true,
    }),
    exactSampleAndCursorIdentity: true,
    excludedFromEveryPrimaryTimingObservation: true,
  });
}

function runSameByteReplayExecution(
  arm: Opt0012ReplayArm,
  mappedBytes: ArrayBuffer,
  plan: AceOpt0012CompactSemanticHeadPlan,
  fixture: PlannerCaseFixture,
): SameByteReplayMeasuredExecution {
  const totalStarted = performance.now();
  const decodeStarted = totalStarted;
  const decoded = decodeAceOpt0012CompactFp16Readback(mappedBytes, plan);
  const decodeEnded = performance.now();
  let fullRows: readonly [Float32Array, Float32Array] | null = null;
  let reconstructionMilliseconds = 0;
  if (arm === "B") {
    const reconstructionStarted = performance.now();
    fullRows = reconstructAceOpt0012FullPlannerLogits(decoded, plan);
    reconstructionMilliseconds = performance.now() - reconstructionStarted;
  }
  const samplingWallStarted = performance.now();
  const sampling =
    arm === "B"
      ? sampleFullRows(plan, fullRows!, fixture.seenTokenIds, fixture.spec.drawIndex)
      : sampleCompactRows(plan, decoded, fixture.seenTokenIds, fixture.spec.drawIndex);
  const samplingWallEnded = performance.now();
  const totalEnded = samplingWallEnded;
  const fp16DecodeMilliseconds = decodeEnded - decodeStarted;
  const samplingWallMilliseconds = samplingWallEnded - samplingWallStarted;
  return Object.freeze({
    timing: Object.freeze({
      arm,
      sample: sampling.sample,
      totalWallMilliseconds: totalEnded - totalStarted,
      fp16DecodeMilliseconds,
      reconstructionMilliseconds,
      samplingWallMilliseconds,
      samplingStageIntervals: sampling.intervals,
      namedNonOverlappingIntervals: Object.freeze({
        fp16HostDecodeMilliseconds: fp16DecodeMilliseconds,
        fullVectorReconstructionMilliseconds: reconstructionMilliseconds,
        primarySamplingStages: Object.freeze({
          preCfgConstraintMilliseconds: sampling.intervals.preCfgConstraintMilliseconds,
          cfgMilliseconds: sampling.intervals.cfgMilliseconds,
          postCfgConstraintAndRepetitionMilliseconds: sampling.intervals.postCfgConstraintAndRepetitionMilliseconds,
          topKMilliseconds: sampling.intervals.topKMilliseconds,
          topPMilliseconds: sampling.intervals.topPMilliseconds,
          temperatureAndSoftmaxMilliseconds: sampling.intervals.temperatureAndSoftmaxMilliseconds,
          categoricalWordAndGlobalMappingMilliseconds: sampling.intervals.categoricalWordAndGlobalMappingMilliseconds,
          callbackMilliseconds: sampling.intervals.callbackMilliseconds,
        }),
        intervalsOccurInListedOrder: true,
        noOverlappingClockSubtractionUsed: true,
      }),
      writeStatus: Object.freeze([decoded.writeStatus[0]!, decoded.writeStatus[1]!] as const),
    }),
    decoded,
    reconstructedFullRows: fullRows,
  });
}

function finalizeSameByteReplayExecution(measured: SameByteReplayMeasuredExecution): SameByteReplayExecution {
  const fullRows = measured.reconstructedFullRows;
  return Object.freeze({
    ...measured.timing,
    decodedRowSha256: Object.freeze([sha256FloatWords(measured.decoded.conditionalLogits), sha256FloatWords(measured.decoded.unconditionalLogits)] as const),
    reconstructedFullRowSha256: fullRows === null ? null : Object.freeze([sha256FloatWords(fullRows[0]), sha256FloatWords(fullRows[1])] as const),
    exactSampleAuthority: true,
  });
}

function summarizeSameByteReplayArm(pairs: readonly SameByteReplayPair[], arm: Opt0012ReplayArm): Readonly<Record<string, unknown>> {
  const samples = pairs.map((pair) => pair[arm]);
  const positions = [0, 1].map((position) => pairs.filter((pair) => pair.order[position] === arm).length);
  if (
    samples.length !== OPT_0012_REPLAY_BALANCED_ORDERS.length ||
    positions.some((count) => count !== 3) ||
    samples.some((sample) => !sameSample(sample.sample, samples[0]!.sample))
  ) {
    throw new Error(`OPT-0012 same-byte replay ${arm} balance changed`);
  }
  const samplingInterval = (name: keyof SamplingStageIntervals) => summarizeNumbers(samples.map((sample) => sample.samplingStageIntervals[name]));
  return Object.freeze({
    sampleCount: samples.length,
    everyOrderPositionCount: Object.freeze(positions),
    totalWallMilliseconds: summarizeNumbers(samples.map((sample) => sample.totalWallMilliseconds)),
    fp16DecodeMilliseconds: summarizeNumbers(samples.map((sample) => sample.fp16DecodeMilliseconds)),
    reconstructionMilliseconds: summarizeNumbers(samples.map((sample) => sample.reconstructionMilliseconds)),
    samplingWallMilliseconds: summarizeNumbers(samples.map((sample) => sample.samplingWallMilliseconds)),
    samplingStageIntervals: Object.freeze({
      samplingMilliseconds: samplingInterval("samplingMilliseconds"),
      preCfgConstraintMilliseconds: samplingInterval("preCfgConstraintMilliseconds"),
      cfgMilliseconds: samplingInterval("cfgMilliseconds"),
      postCfgConstraintAndRepetitionMilliseconds: samplingInterval("postCfgConstraintAndRepetitionMilliseconds"),
      topKMilliseconds: samplingInterval("topKMilliseconds"),
      topPMilliseconds: samplingInterval("topPMilliseconds"),
      temperatureAndSoftmaxMilliseconds: samplingInterval("temperatureAndSoftmaxMilliseconds"),
      categoricalWordAndGlobalMappingMilliseconds: samplingInterval("categoricalWordAndGlobalMappingMilliseconds"),
      callbackMilliseconds: samplingInterval("callbackMilliseconds"),
    }),
    exactSampleAndCursorIdentity: true,
  });
}

function summarizeSameByteReplayPairs(pairs: readonly SameByteReplayPair[]): Readonly<Record<string, unknown>> {
  const rounds = pairs.map((pair) => {
    const deltaCMinusBMilliseconds = pair.C.totalWallMilliseconds - pair.B.totalWallMilliseconds;
    return Object.freeze({
      pairIndex: pair.pairIndex,
      order: pair.order.join(""),
      bOrderPosition: pair.order.indexOf("B"),
      cOrderPosition: pair.order.indexOf("C"),
      bMilliseconds: pair.B.totalWallMilliseconds,
      cMilliseconds: pair.C.totalWallMilliseconds,
      deltaCMinusBMilliseconds,
      winner: deltaCMinusBMilliseconds === 0 ? "tie" : deltaCMinusBMilliseconds < 0 ? "C" : "B",
    });
  });
  return Object.freeze({
    leftArm: "B-full",
    rightArm: "C-compact",
    pairCount: rounds.length,
    leftWins: rounds.filter((round) => round.winner === "B").length,
    rightWins: rounds.filter((round) => round.winner === "C").length,
    ties: rounds.filter((round) => round.winner === "tie").length,
    rawDeltaConvention: "C-minus-B-milliseconds",
    thresholdApplied: false,
    rounds: Object.freeze(rounds),
  });
}

async function runOpt0012Fp16ConversionMicrobenchmark(): Promise<Readonly<Record<string, unknown>>> {
  const domain = new Uint16Array(OPT_0012_FP16_DOMAIN_WORD_COUNT);
  for (let bits = 0; bits < domain.length; bits += 1) domain[bits] = bits;
  const domainSha256 = sha256U16Le(domain);
  if (domainSha256 !== "68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b") {
    throw new Error("OPT-0012 complete FP16 conversion domain hash changed");
  }
  const outputs = new Map<
    Opt0012ConversionArm,
    Readonly<{
      floats: Float32Array;
      words: Uint32Array;
    }>
  >();
  for (const arm of ["legacy-allocating", "allocation-free"] as const) {
    const floats = new Float32Array(OPT_0012_FP16_DOMAIN_WORD_COUNT);
    outputs.set(
      arm,
      Object.freeze({
        floats,
        words: new Uint32Array(floats.buffer),
      }),
    );
  }
  const pairs: ConversionMicrobenchmarkPair[] = [];
  for (let pairIndex = 0; pairIndex < OPT_0012_CONVERSION_BALANCED_ORDERS.length; pairIndex += 1) {
    const order = OPT_0012_CONVERSION_BALANCED_ORDERS[pairIndex]!;
    let legacyWallMilliseconds: number | undefined;
    let candidateWallMilliseconds: number | undefined;
    // Deliberately no await/yield between the two arms in a pair.
    for (const arm of order) {
      const output = outputs.get(arm)!;
      const started = performance.now();
      writeOpt0012Fp16ConversionDomain(arm, output.floats, output.words);
      const wallMilliseconds = performance.now() - started;
      if (arm === "legacy-allocating") {
        legacyWallMilliseconds = wallMilliseconds;
      } else {
        candidateWallMilliseconds = wallMilliseconds;
      }
    }
    if (legacyWallMilliseconds === undefined || candidateWallMilliseconds === undefined) {
      throw new Error("OPT-0012 conversion timing pair omitted an arm");
    }
    // Hashing and receipt expansion start only after both timed arms finish.
    const legacyEnvelope = analyzeOpt0012Fp16ConversionEnvelope("legacy-allocating", outputs.get("legacy-allocating")!.words);
    const candidateEnvelope = analyzeOpt0012Fp16ConversionEnvelope("allocation-free", outputs.get("allocation-free")!.words);
    if (legacyEnvelope["acceptedStableNanEnvelope"] !== true || candidateEnvelope["acceptedStableNanEnvelope"] !== true) {
      throw new Error("OPT-0012 timed conversion NaN envelope changed: " + JSON.stringify({ legacyEnvelope, candidateEnvelope }));
    }
    const legacy: ConversionMicrobenchmarkExecution = Object.freeze({
      arm: "legacy-allocating",
      wallMilliseconds: legacyWallMilliseconds,
      envelope: legacyEnvelope,
      acceptedStableNanEnvelope: true,
    });
    const candidate: ConversionMicrobenchmarkExecution = Object.freeze({
      arm: "allocation-free",
      wallMilliseconds: candidateWallMilliseconds,
      envelope: candidateEnvelope,
      acceptedStableNanEnvelope: true,
    });
    pairs.push(
      Object.freeze({
        pairIndex,
        order,
        legacy,
        candidate,
        noYieldInsidePair: true,
      }),
    );
    await yieldToBrowser();
  }
  const summarizeArm = (arm: Opt0012ConversionArm) => {
    const samples = pairs.map((pair) => (arm === "legacy-allocating" ? pair.legacy : pair.candidate));
    const positions = [0, 1].map((position) => pairs.filter((pair) => pair.order[position] === arm).length);
    if (positions.some((count) => count !== 3)) {
      throw new Error(`OPT-0012 ${arm} conversion timing balance changed`);
    }
    return Object.freeze({
      sampleCount: samples.length,
      everyOrderPositionCount: Object.freeze(positions),
      wallMilliseconds: summarizeNumbers(samples.map((sample) => sample.wallMilliseconds)),
      everyRawOutputU32LeSha256: Object.freeze(samples.map((sample) => sample.envelope["rawOutputU32LeSha256"] as string)),
      everyCanonicalOutputU32LeSha256: Object.freeze(samples.map((sample) => sample.envelope["canonicalOutputU32LeSha256"] as string)),
      everyAcceptedStableNanEnvelope: samples.every((sample) => sample.acceptedStableNanEnvelope),
    });
  };
  const rounds = pairs.map((pair) => {
    const deltaCandidateMinusLegacyMilliseconds = pair.candidate.wallMilliseconds - pair.legacy.wallMilliseconds;
    return Object.freeze({
      pairIndex: pair.pairIndex,
      order: pair.order,
      legacyOrderPosition: pair.order.indexOf("legacy-allocating"),
      candidateOrderPosition: pair.order.indexOf("allocation-free"),
      legacyMilliseconds: pair.legacy.wallMilliseconds,
      candidateMilliseconds: pair.candidate.wallMilliseconds,
      deltaCandidateMinusLegacyMilliseconds,
      winner: deltaCandidateMinusLegacyMilliseconds === 0 ? "tie" : deltaCandidateMinusLegacyMilliseconds < 0 ? "allocation-free" : "legacy-allocating",
    });
  });
  return Object.freeze({
    authority: "fixed-schedule-target-browser-allocation-JIT-diagnostic",
    domain: Object.freeze({
      firstFp16U16: 0,
      lastFp16U16: 0xffff,
      wordCount: domain.length,
      inputU16LeSha256: domainSha256,
      expectedOutputU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
      expectedNonNaNU32LeSha256: OPT_0012_FP16_NON_NAN_OUTPUT_SHA256,
      expectedCanonicalNaNU32LeSha256: OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256,
    }),
    protocol: Object.freeze({
      orders: OPT_0012_CONVERSION_BALANCED_ORDERS,
      pairCount: pairs.length,
      outputsPreallocatedOutsideTiming: true,
      outputHashingExcludedFromArmTiming: true,
      noHashingOrReceiptExpansionBetweenTimedPairArms: true,
      noYieldInsidePair: true,
      yieldOnlyAfterCompletedPair: true,
      outsideEveryPrimaryTokenWall: true,
      executedAfterSameByteReplayTiming: true,
      envelopeValidationAndReceiptExpansionOnlyAfterBothTimedArms: true,
      legacyRawNanHashesAreDiagnosticsNotPortablePayloadEvidence: true,
      doesNotClaimStableJitTier: true,
    }),
    pairs: Object.freeze(pairs),
    legacy: summarizeArm("legacy-allocating"),
    candidate: summarizeArm("allocation-free"),
    samePairComparison: Object.freeze({
      pairCount: rounds.length,
      legacyWins: rounds.filter((round) => round.winner === "legacy-allocating").length,
      candidateWins: rounds.filter((round) => round.winner === "allocation-free").length,
      ties: rounds.filter((round) => round.winner === "tie").length,
      rawDeltaConvention: "candidate-minus-legacy-milliseconds",
      thresholdApplied: false,
      rounds: Object.freeze(rounds),
    }),
  });
}

export function validateOpt0012PrimaryTimingFp16NaNCensusCoverage(receipts: readonly Readonly<Record<string, unknown>>[]): Readonly<Record<string, unknown>> {
  if (receipts.length !== OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT) {
    throw new Error("OPT-0012 primary timing FP16 NaN census must contain exactly 54 receipts");
  }
  const compositeJoinKeys = new Set<string>();
  const traceLabels = new Set<string>();
  const armCounts: Record<Opt0012Arm, number> = { A: 0, B: 0, C: 0 };
  for (const receipt of receipts) {
    const arm = receipt["arm"];
    const traceLabel = receipt["traceLabel"];
    const roundIndex = receipt["roundIndex"];
    const order = receipt["order"];
    const orderPosition = receipt["orderPosition"];
    const census = receipt["census"];
    if (
      (arm !== "A" && arm !== "B" && arm !== "C") ||
      typeof traceLabel !== "string" ||
      traceLabel.length === 0 ||
      !Number.isInteger(roundIndex) ||
      typeof roundIndex !== "number" ||
      roundIndex < 0 ||
      roundIndex >= OPT_0012_BALANCED_ORDERS.length ||
      typeof order !== "string" ||
      !Number.isInteger(orderPosition) ||
      typeof orderPosition !== "number" ||
      census === null ||
      typeof census !== "object" ||
      (census as Readonly<Record<string, unknown>>)["zeroBinary16NaNs"] !== true
    ) {
      throw new Error("OPT-0012 primary timing FP16 NaN census receipt changed");
    }
    const expectedOrder = OPT_0012_BALANCED_ORDERS[roundIndex]!;
    if (order !== expectedOrder.join("") || orderPosition !== expectedOrder.indexOf(arm)) {
      throw new Error("OPT-0012 primary timing FP16 NaN census schedule changed");
    }
    const compositeJoinKey = JSON.stringify([traceLabel, roundIndex, order, orderPosition]);
    if (compositeJoinKeys.has(compositeJoinKey)) {
      throw new Error("OPT-0012 primary timing FP16 NaN census has a duplicate composite join key");
    }
    compositeJoinKeys.add(compositeJoinKey);
    traceLabels.add(traceLabel);
    armCounts[arm] += 1;
  }
  const expectedPerArm = OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT / 3;
  if (
    compositeJoinKeys.size !== OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT ||
    armCounts.A !== expectedPerArm ||
    armCounts.B !== expectedPerArm ||
    armCounts.C !== expectedPerArm
  ) {
    throw new Error("OPT-0012 primary timing FP16 NaN census coverage changed");
  }
  return Object.freeze({
    receiptCount: receipts.length,
    expectedReceiptCount: OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT,
    A: armCounts.A,
    B: armCounts.B,
    C: armCounts.C,
    uniqueTraceLabelCount: traceLabels.size,
    uniqueCompositeJoinKeyCount: compositeJoinKeys.size,
    compositeJoinFields: Object.freeze(["traceLabel", "roundIndex", "order", "orderPosition"]),
    oneToOneCompositeTraceJoin: true,
    everyActualPackageReadbackHasZeroBinary16NaNs: true,
  });
}

async function runTimedAndCleanup(prepared: PreparedSession, thermal: Opt0012ThermalGateMetadata): Promise<Readonly<Record<string, unknown>>> {
  validateWorkerThermalGate(thermal, prepared.warmupCompletedAtEpochMilliseconds);
  const timedStartedAtEpochMilliseconds = Date.now();
  const cases: Array<Readonly<Record<string, unknown>>> = [];
  const timedReadbackExecutions: PackageArmExecution[] = [];
  const deferredCandidateReplays: Array<
    Readonly<{
      fixture: PlannerCaseFixture;
      b: PackageArmExecution;
      c: PackageArmExecution;
      authority: SampleReceipt;
    }>
  > = [];
  try {
    for (const fixture of prepared.fixtures) {
      const armSamples = new Map<Opt0012Arm, PackageArmExecution[]>([
        ["A", []],
        ["B", []],
        ["C", []],
      ]);
      for (let roundIndex = 0; roundIndex < OPT_0012_BALANCED_ORDERS.length; roundIndex += 1) {
        const order = OPT_0012_BALANCED_ORDERS[roundIndex]!;
        for (let orderPosition = 0; orderPosition < order.length; orderPosition += 1) {
          const arm = order[orderPosition]!;
          postProgress(`timing ${fixture.spec.id} round ${roundIndex + 1}/6 ` + `${order.join("")} arm ${arm}`);
          const sample = await executePackageArm(
            prepared.observer,
            prepared.executor,
            prepared.candidateRunner,
            fixture,
            prepared.regularPlan,
            arm,
            fixture.spec.drawIndex,
            (roundIndex + orderPosition) % 2 === 0 ? SENTINEL_A : SENTINEL_B,
            false,
            "timing",
            roundIndex,
            order.join(""),
            orderPosition,
          );
          armSamples.get(arm)!.push(sample);
          timedReadbackExecutions.push(sample);
          await yieldToBrowser();
        }
      }
      const authority = armSamples.get("A")![0]!.sample;
      for (const arm of ["A", "B", "C"] as const) {
        for (const sample of armSamples.get(arm)!) {
          requireSameSample(authority, sample.sample, `${fixture.spec.id} timed A/B/C authority`);
        }
      }
      deferredCandidateReplays.push(
        Object.freeze({
          fixture,
          b: armSamples.get("B")![0]!,
          c: armSamples.get("C")![0]!,
          authority,
        }),
      );
      cases.push(
        Object.freeze({
          case: fixture.spec,
          orders: OPT_0012_BALANCED_ORDERS,
          A: summarizeTimedArm(armSamples.get("A")!),
          B: summarizeTimedArm(armSamples.get("B")!),
          C: summarizeTimedArm(armSamples.get("C")!),
          sameRoundComparisons: Object.freeze({
            AversusB: summarizeSameRoundComparison(armSamples.get("A")!, armSamples.get("B")!, "A", "B"),
            BversusC: summarizeSameRoundComparison(armSamples.get("B")!, armSamples.get("C")!, "B", "C"),
          }),
          attributableComparisons: Object.freeze({
            AversusB: "restricted-head/readback plus reconstruction",
            BversusC: "same-byte common-GPU CPU replay",
            AversusC: "combined context only",
            speedThresholdApplied: false,
          }),
        }),
      );
    }
    const primaryTokenTimedCompletedAtEpochMilliseconds = Date.now();
    postProgress("validating zero FP16 NaNs after the complete primary timing window");
    const primaryTimingFp16NaNCensuses = Object.freeze(
      timedReadbackExecutions.map((execution) => {
        const census =
          execution.arm === "A"
            ? requireZeroOpt0012DecodedFullFp16NaNs(execution.logits, String(execution.trace["traceLabel"]), execution.state)
            : requireZeroOpt0012RawCompactFp16NaNs(prepared.regularPlan, execution.compactMappedBytes!, String(execution.trace["traceLabel"]));
        return Object.freeze({
          arm: execution.arm,
          state: execution.state,
          traceLabel: execution.trace["traceLabel"],
          roundIndex: execution.trace["roundIndex"],
          order: execution.trace["order"],
          orderPosition: execution.trace["orderPosition"],
          census,
        });
      }),
    );
    const primaryTimingFp16NaNCensusCoverage = validateOpt0012PrimaryTimingFp16NaNCensusCoverage(primaryTimingFp16NaNCensuses);
    const postTimingCandidateReplays: Array<Readonly<Record<string, unknown>>> = [];
    for (let replayIndex = 0; replayIndex < deferredCandidateReplays.length; replayIndex += 1) {
      const { fixture, b, c, authority } = deferredCandidateReplays[replayIndex]!;
      postProgress(`timing same-immutable-byte B/C replay ${replayIndex + 1}/` + `${deferredCandidateReplays.length} ${fixture.spec.id}`);
      postTimingCandidateReplays.push(await runSameImmutableByteReplayCase(prepared.regularPlan, fixture, b, c, authority));
    }
    postProgress("timing balanced exhaustive FP16 host conversions");
    const fp16ConversionMicrobenchmark = await runOpt0012Fp16ConversionMicrobenchmark();
    const correctivePostPrimaryTimingCompletedAtEpochMilliseconds = Date.now();
    postProgress("running post-timing drained candidate cancellation proofs");
    const cancellation = await runCancellationProofs(prepared);
    const cleanup = await destroyPreparedSession(prepared);
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    const heartbeat = workerHeartbeat?.stop();
    return Object.freeze({
      experiment: "OPT-0012",
      mode: "balanced-timing",
      identity: prepared.identity,
      coreSourceSha256: OPT_0012_CORE_SOURCE_SHA256,
      candidateShaderSha256: OPT_0012_CANDIDATE_SHADER_SHA256,
      package: prepared.preparedPackage.summary,
      sourceAuthentication: prepared.sourceAuthentication,
      capabilities: prepared.capabilityAuthentication,
      correctness: prepared.correctness,
      adversarialSampling: prepared.adversarialSampling,
      thermal: Object.freeze({
        ...thermal,
        preGateOnly: true,
        status: "pending-external-artifact-join",
        browserReceiptClaimsPlanValidThermalCoverage: false,
        continuousLoggerRequiredThroughEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      }),
      timedStartedAtEpochMilliseconds,
      timedCompletedAtEpochMilliseconds: primaryTokenTimedCompletedAtEpochMilliseconds,
      primaryTokenTimedCompletedAtEpochMilliseconds,
      correctivePostPrimaryTimingCompletedAtEpochMilliseconds,
      cases: Object.freeze(cases),
      primaryTimingFp16NaNCensusCoverage,
      primaryTimingFp16NaNCensuses,
      postTimingCandidateReplays: Object.freeze(postTimingCandidateReplays),
      fp16ConversionGate: prepared.fp16ConversionGate,
      fp16ConversionMicrobenchmark,
      correctiveTimingProtocol: Object.freeze({
        originalSixOrderAbcCompletedFirstAndUnchanged: true,
        primaryReadbackNanCensusExecutedOnlyAfterEntirePrimaryWindow: true,
        sameImmutableByteReplayExecutedAfterPrimaryTokenTiming: true,
        fp16ConversionMicrobenchmarkExecutedAfterSameByteReplay: true,
        cancellationAndCleanupExecutedAfterAllTiming: true,
      }),
      cancellation,
      cleanup,
      runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
      workerHeartbeat: heartbeat,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
    });
  } catch (error) {
    await destroyPreparedSession(prepared);
    throw error;
  }
}

function summarizeTimedArm(samples: readonly PackageArmExecution[]): Readonly<Record<string, unknown>> {
  const rounds = samples.map((sample) => sample.trace.roundIndex as number).sort((left, right) => left - right);
  const positionCounts = [0, 1, 2].map((position) => samples.filter((sample) => sample.trace.orderPosition === position).length);
  if (
    samples.length !== 6 ||
    rounds.some((round, index) => round !== index) ||
    positionCounts.some((count) => count !== 2) ||
    samples.some((sample) => {
      const round = sample.trace.roundIndex as number;
      const expectedOrder = OPT_0012_BALANCED_ORDERS[round];
      return expectedOrder === undefined || sample.trace.order !== expectedOrder.join("") || sample.trace.orderPosition !== expectedOrder.indexOf(sample.arm);
    }) ||
    samples.some((sample) => !sameSample(samples[0]!.sample, sample.sample)) ||
    samples.some((sample) => sample.fp16NaNCensus !== null)
  ) {
    throw new Error("OPT-0012 timed arm does not contain six samples");
  }
  const totals = samples.map((sample) => sample.intervals["totalWallMilliseconds"]!);
  return Object.freeze({
    rawSamples: Object.freeze(samples.map(publicArmExecution)),
    totalWallMilliseconds: summarizeNumbers(totals),
    exactSampleIdentity: true,
    sampleCount: samples.length,
    distinctRounds: Object.freeze(rounds),
    everyOrderPositionCount: Object.freeze(positionCounts),
    fp16NaNCensusDeferredUntilAfterEntirePrimaryTimingWindow: true,
  });
}

function summarizeSameRoundComparison(
  left: readonly PackageArmExecution[],
  right: readonly PackageArmExecution[],
  leftArm: Opt0012Arm,
  rightArm: Opt0012Arm,
): Readonly<Record<string, unknown>> {
  const byRound = (samples: readonly PackageArmExecution[]) => new Map(samples.map((sample) => [sample.trace.roundIndex as number, sample]));
  const leftByRound = byRound(left);
  const rightByRound = byRound(right);
  const rounds = OPT_0012_BALANCED_ORDERS.map((order, roundIndex) => {
    const leftSample = leftByRound.get(roundIndex);
    const rightSample = rightByRound.get(roundIndex);
    if (
      leftSample === undefined ||
      rightSample === undefined ||
      leftSample.arm !== leftArm ||
      rightSample.arm !== rightArm ||
      leftSample.trace.order !== order.join("") ||
      rightSample.trace.order !== order.join("")
    ) {
      throw new Error(`OPT-0012 same-round ${leftArm}/${rightArm} pairing changed`);
    }
    const leftMilliseconds = leftSample.intervals["totalWallMilliseconds"]!;
    const rightMilliseconds = rightSample.intervals["totalWallMilliseconds"]!;
    const deltaRightMinusLeftMilliseconds = rightMilliseconds - leftMilliseconds;
    return Object.freeze({
      roundIndex,
      order: order.join(""),
      leftArm,
      rightArm,
      leftOrderPosition: leftSample.trace.orderPosition,
      rightOrderPosition: rightSample.trace.orderPosition,
      leftMilliseconds,
      rightMilliseconds,
      deltaRightMinusLeftMilliseconds,
      winner: deltaRightMinusLeftMilliseconds === 0 ? "tie" : deltaRightMinusLeftMilliseconds < 0 ? rightArm : leftArm,
    });
  });
  const leftWins = rounds.filter((round) => round.winner === leftArm).length;
  const rightWins = rounds.filter((round) => round.winner === rightArm).length;
  const ties = rounds.filter((round) => round.winner === "tie").length;
  if (leftWins + rightWins + ties !== OPT_0012_BALANCED_ORDERS.length) {
    throw new Error("OPT-0012 same-round win accounting changed");
  }
  return Object.freeze({
    leftArm,
    rightArm,
    roundCount: rounds.length,
    leftWins,
    rightWins,
    ties,
    rawDeltaConvention: "right-minus-left-milliseconds",
    thresholdApplied: false,
    rounds: Object.freeze(rounds),
  });
}

function validateWorkerThermalGate(thermal: Opt0012ThermalGateMetadata, warmupCompletedAtEpochMilliseconds: number): void {
  if (
    thermal.source !== OPT_0012_THERMAL_SOURCE ||
    thermal.startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds < thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds !== thermal.completedAtEpochMilliseconds - thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds < OPT_0012_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount < Math.floor(thermal.durationMilliseconds / 1_000) + 1 ||
    thermal.pollMilliseconds !== OPT_0012_THERMAL_POLL_MILLISECONDS ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds > 1_250 ||
    thermal.nonNominalObservationCount !== 0
  ) {
    throw new Error("OPT-0012 worker rejected the external thermal gate");
  }
}

interface RawDispatchRecord extends Opt0012ObservedHeadDispatchDescriptor {
  readonly index: number;
  readonly passLabel: string;
  readonly pipelineLabel: string;
  readonly bindGroupLabel: string;
  readonly workgroups: readonly [number, number, number];
}

interface RawCopyRecord extends Opt0012ObservedCopyDescriptor {
  readonly index: number;
  readonly sourceBufferLabel: string;
  readonly sourceOffset: number;
  readonly destinationBufferLabel: string;
  readonly destinationOffset: number;
  readonly copiedBytes: number;
}

interface Opt0012ExpectedClearDescriptor {
  readonly bufferLabel: string;
  readonly offset: number;
  readonly size: number;
}

interface MutableCommandRecord {
  readonly label: string;
  readonly passLabels: string[];
  readonly dispatches: RawDispatchRecord[];
  readonly copies: RawCopyRecord[];
  readonly clears: Array<
    Readonly<{
      bufferLabel: string;
      offset: number;
      size: number;
    }>
  >;
  readonly encodeStartedAt: number;
  encodeEndedAt?: number;
  submitStartedAt?: number;
  submitReturnedAt?: number;
  drainStartedAt?: number;
  drainEndedAt?: number;
  idleEndedAt?: number;
}

interface MutableMapRecord {
  readonly bufferLabel: string;
  readonly mapStartedAt: number;
  mapEndedAt?: number;
  unmapAt?: number;
}

interface TrackedBufferRecord extends Opt0012TrackedBufferCleanupRecord {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyCallCount: number;
  mapCallCount: number;
  unmapCallCount: number;
  destroyed: boolean;
  mapped: boolean;
  readonly createdAtResourceEpoch: number;
  destroyedAtResourceEpoch: number | null;
}

interface TraceCompletionMetadata {
  readonly arm: Opt0012Arm;
  readonly phaseKind: "prefill" | "decode";
  readonly state: AceOpt0012SemanticState;
  readonly expectedPhysicalDispatchCount: number;
  readonly expectedCopyCount: number;
  readonly expectedCommandBufferCount: number;
  readonly roundIndex: number;
  readonly order: string;
  readonly orderPosition: number;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly decodeMilliseconds: number;
  readonly reconstructionMilliseconds: number;
  readonly samplingMilliseconds: number;
  readonly callbackMilliseconds: number;
  readonly samplingStageIntervals: SamplingStageIntervals;
  readonly decodeStartedAt: number;
  readonly decodeEndedAt: number;
  readonly candidateHostDecodeMilliseconds: number | null;
  readonly readbackIdleInterval: Opt0012IdleIntervalReceipt;
  readonly expectedEncodedFreshStatusClears: readonly Opt0012ExpectedClearDescriptor[];
}

class Opt0012DeviceObserver {
  readonly device: GPUDevice;

  private readonly queue: GPUQueue;
  private readonly commandByBuffer = new WeakMap<GPUCommandBuffer, MutableCommandRecord>();
  private readonly bufferRecords = new Map<GPUBuffer, TrackedBufferRecord>();
  private readonly bufferLabels = new WeakMap<GPUBuffer, string>();
  private active = false;
  private traceLabel = "";
  private captureFullReadback = false;
  private commands: MutableCommandRecord[] = [];
  private pending: MutableCommandRecord[] = [];
  private submissionCount = 0;
  private drainCount = 0;
  private maximumOutstandingCommandBuffers = 0;
  private fullReadbackBytes: ArrayBuffer | null = null;
  private maps: MutableMapRecord[] = [];
  private destructionTrackingSupported = true;
  private mapTrackingSupported = true;
  private resourceEpoch = 0;
  private liveTrackedBufferCount = 0;
  private liveTrackedBufferBytes = 0;
  private maximumLiveTrackedBufferCount = 0;
  private maximumLiveTrackedBufferBytes = 0;
  private highWaterResourceEpoch = 0;
  private highWaterTriggerLabel = "";
  private highWaterLiveBufferCount = 0;
  private countHighWaterResourceEpoch = 0;
  private countHighWaterTriggerLabel = "";
  private countHighWaterLiveBufferBytes = 0;
  private lifetimeSubmitCallCount = 0;
  private lifetimeSubmittedCommandBufferCount = 0;
  private lifetimeDrainCallCount = 0;
  private lifetimeMapAsyncCallCount = 0;

  constructor(private readonly target: GPUDevice) {
    this.queue = this.createQueueProxy(target.queue);
    this.device = new Proxy(target, {
      get: (device, property) => {
        if (property === "queue") return this.queue;
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer => this.createTrackedBuffer(descriptor);
        }
        if (property === "createCommandEncoder") {
          return (descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder => this.createObservedCommandEncoder(descriptor);
        }
        const value = Reflect.get(device, property, device) as unknown;
        return typeof value === "function" ? (value as (...args: unknown[]) => unknown).bind(device) : value;
      },
    }) as GPUDevice;
  }

  beginTrace(label: string, captureFullReadback: boolean): void {
    if (this.active || this.pending.length !== 0) {
      throw new DOMException("OPT-0012 command trace overlaps", "InvalidStateError");
    }
    this.active = true;
    this.traceLabel = label;
    this.captureFullReadback = captureFullReadback;
    this.commands = [];
    this.pending = [];
    this.submissionCount = 0;
    this.drainCount = 0;
    this.maximumOutstandingCommandBuffers = 0;
    this.fullReadbackBytes = null;
    this.maps = [];
  }

  endTrace(metadata: TraceCompletionMetadata): Readonly<Record<string, unknown>> {
    if (!this.active) {
      throw new DOMException("OPT-0012 trace is not active", "InvalidStateError");
    }
    if (this.pending.length !== 0) {
      throw new Error("OPT-0012 trace ended before queue completion");
    }
    const endedAt = performance.now();
    const last = this.commands.at(-1);
    if (last !== undefined && last.idleEndedAt === undefined) last.idleEndedAt = endedAt;
    const expectedTopology = opt0012ExpectedCommandBufferTopology(metadata.arm, metadata.phaseKind);
    if (metadata.expectedCommandBufferCount !== expectedTopology.totalCommandBufferCount) {
      throw new Error("OPT-0012 trace command-buffer expectation changed");
    }
    const physicalDispatchCount = this.commands.reduce((sum, command) => sum + command.dispatches.length, 0);
    const copies = this.commands.flatMap((command) => command.copies);
    const clears = this.commands.flatMap((command) => command.clears);
    if (
      clears.length !== metadata.expectedEncodedFreshStatusClears.length ||
      clears.some((clear, index) => {
        const expected = metadata.expectedEncodedFreshStatusClears[index];
        return expected === undefined || clear.bufferLabel !== expected.bufferLabel || clear.offset !== expected.offset || clear.size !== expected.size;
      }) ||
      (clears.length !== 0 && this.commands[0]!.clears.length !== clears.length)
    ) {
      throw new Error("OPT-0012 encoded fresh prefill status clear changed");
    }
    const authenticatedCopiedBytes = authenticateOpt0012ObservedCopies(metadata.arm, metadata.state, copies);
    if (
      this.commands.length !== metadata.expectedCommandBufferCount ||
      this.submissionCount !== metadata.expectedCommandBufferCount ||
      this.drainCount !== metadata.expectedCommandBufferCount ||
      this.maximumOutstandingCommandBuffers !== 1 ||
      physicalDispatchCount !== metadata.expectedPhysicalDispatchCount ||
      copies.length !== metadata.expectedCopyCount
    ) {
      throw new Error(`OPT-0012 ${metadata.arm} topology changed: ` + `${this.commands.length}/${physicalDispatchCount}/${copies.length}`);
    }
    const readback = this.commands.at(-1);
    const heads = metadata.arm === "A" ? this.commands.slice(-3, -1) : this.commands.slice(-2, -1);
    const preHead = this.commands.slice(0, expectedTopology.preHeadCommandBufferCount);
    const preHeadPhysicalDispatchCount = preHead.reduce((sum, command) => sum + command.dispatches.length, 0);
    const expectedHeadPrimitiveCounts = metadata.arm === "A" ? [2, 3] : [metadata.state === "regular-code" ? 2 : 1];
    if (
      readback === undefined ||
      preHead.length !== expectedTopology.preHeadCommandBufferCount ||
      preHeadPhysicalDispatchCount !== 623 ||
      preHead.some((command) => command.label.includes("tied-lm-head")) ||
      heads.length !== (metadata.arm === "A" ? 2 : 1) ||
      heads.some((head) => !head.label.includes("tied-lm-head")) ||
      heads.some((head, index) => head.dispatches.length !== expectedHeadPrimitiveCounts[index]) ||
      (metadata.arm === "A" && (!heads[0]!.label.endsWith("tied-lm-head-part-0-command") || !heads[1]!.label.endsWith("tied-lm-head-part-1-command"))) ||
      (metadata.arm !== "A" && !heads[0]!.label.endsWith("opt-0012-tied-lm-head-command")) ||
      !readback.label.includes("readback")
    ) {
      throw new Error("OPT-0012 head/readback command order changed");
    }
    const observedHeadDispatches = heads.flatMap((head) => head.dispatches);
    authenticateOpt0012ObservedHeadDispatches(metadata.arm, metadata.state, observedHeadDispatches);
    const maximumSingleDrainMilliseconds = Math.max(
      ...this.commands.map((command) => requiredTimestamp(command.drainEndedAt, "drain end") - requiredTimestamp(command.drainStartedAt, "drain start")),
    );
    if (this.maps.length !== 1) {
      throw new Error(`OPT-0012 trace observed ${this.maps.length} readback maps`);
    }
    const map = this.maps[0]!;
    const mapEndedAt = requiredTimestamp(map.mapEndedAt, "map end");
    const unmapAt = requiredTimestamp(map.unmapAt, "unmap");
    if (
      !map.bufferLabel.includes("readback") ||
      map.mapStartedAt < requiredTimestamp(readback.drainEndedAt, "readback drain end") ||
      mapEndedAt < map.mapStartedAt ||
      unmapAt < mapEndedAt ||
      metadata.decodeEndedAt < unmapAt
    ) {
      throw new Error("OPT-0012 readback map/host-decode ordering changed");
    }
    const postUnmapDecodeMilliseconds = metadata.decodeEndedAt - unmapAt;
    const exactReadbackIdle = metadata.readbackIdleInterval;
    const candidateHostDecodeMilliseconds = metadata.candidateHostDecodeMilliseconds;
    if (
      postUnmapDecodeMilliseconds < 0 ||
      exactReadbackIdle.stage !== "readback" ||
      exactReadbackIdle.requestedMilliseconds !== ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS ||
      !exactReadbackIdle.completed ||
      exactReadbackIdle.startedAt < requiredTimestamp(readback.drainEndedAt, "readback drain end") ||
      exactReadbackIdle.startedAt > map.mapStartedAt ||
      exactReadbackIdle.endedAt < map.mapStartedAt ||
      exactReadbackIdle.endedAt > metadata.decodeEndedAt ||
      exactReadbackIdle.source !== (metadata.arm === "A" ? "production-executor-yieldQueueIdle" : "benchmark-candidate-equivalent-yieldQueueIdle") ||
      (metadata.arm !== "A" && (candidateHostDecodeMilliseconds === null || candidateHostDecodeMilliseconds < 0))
    ) {
      throw new Error("OPT-0012 host FP16 decode interval is absent");
    }
    const preHeadIntervals = summarizeCommandIntervals(preHead);
    const tiedHeadIntervals = summarizeCommandIntervals(heads);
    const readbackIntervals = summarizeCommandIntervals([readback]);
    const samplingStageComponentMilliseconds =
      metadata.samplingStageIntervals.preCfgConstraintMilliseconds +
      metadata.samplingStageIntervals.cfgMilliseconds +
      metadata.samplingStageIntervals.postCfgConstraintAndRepetitionMilliseconds +
      metadata.samplingStageIntervals.topKMilliseconds +
      metadata.samplingStageIntervals.topPMilliseconds +
      metadata.samplingStageIntervals.temperatureAndSoftmaxMilliseconds +
      metadata.samplingStageIntervals.categoricalWordAndGlobalMappingMilliseconds +
      metadata.samplingStageIntervals.callbackMilliseconds;
    if (
      metadata.samplingStageIntervals.callbackInvocationCount !== (metadata.order === "trajectory" ? 0 : 1) ||
      samplingStageComponentMilliseconds > metadata.samplingMilliseconds
    ) {
      throw new Error("OPT-0012 primary sampling stage intervals overlap or changed");
    }
    const result = Object.freeze({
      traceLabel: this.traceLabel,
      ...metadata,
      commandBufferCount: this.commands.length,
      submissionCount: this.submissionCount,
      queueDrainCount: this.drainCount,
      realCooperativeIdleCount: this.commands.length,
      requestedIdleMilliseconds: this.commands.length,
      realCooperativeIdleTimingAuthority: "explicit-progress-idleIntervals-attached-by-caller",
      maximumOutstandingCommandBuffers: this.maximumOutstandingCommandBuffers,
      physicalDispatchCount,
      preHeadCommandBufferCount: preHead.length,
      preHeadPhysicalDispatchCount,
      headPhysicalDispatchCounts: Object.freeze([...expectedHeadPrimitiveCounts]),
      observedHeadDispatches: Object.freeze([...observedHeadDispatches]),
      copyCount: copies.length,
      copiedBytes: copies.reduce((sum, copy) => sum + copy.copiedBytes, 0),
      authenticatedCopiedBytes,
      encodedFreshStatusClears: Object.freeze(clears),
      encodedFreshStatusClearAuthenticated: metadata.expectedEncodedFreshStatusClears.length === 2,
      maximumSingleDrainMilliseconds,
      nonOverlappingIntervals: Object.freeze({
        preHead: preHeadIntervals,
        tiedHead: tiedHeadIntervals,
        readbackCopySubmitDrainAndObservedPreMapGap: readbackIntervals,
        mapWaitMilliseconds: mapEndedAt - map.mapStartedAt,
        mappedRangeCopyMilliseconds: unmapAt - mapEndedAt,
        candidateHostFp16DecodeMilliseconds: candidateHostDecodeMilliseconds,
        armAPostUnmapDecodePlusIdleResidualMilliseconds: metadata.arm === "A" ? postUnmapDecodeMilliseconds : null,
        armAPureHostFp16DecodeAuthority: metadata.arm === "A" ? "untimed-correctness-calibration" : null,
        fullVectorReconstructionMilliseconds: metadata.reconstructionMilliseconds,
        primarySampling: Object.freeze({
          ...metadata.samplingStageIntervals,
          measuredStageComponentMilliseconds: samplingStageComponentMilliseconds,
          wallMilliseconds: metadata.samplingMilliseconds,
          cfgAndConstraintSegmentsExecutedInProductionOrder: true,
          decomposedWithAcceptedBrowserV1ProductionPrimitives: true,
          unchangedCursorAuthorityComparedDuringCorrectnessGate: true,
          nonOverlapping: true,
        }),
        callbackMilliseconds: metadata.callbackMilliseconds,
        readbackIdlePolicyRunsConcurrentlyWithMapAndHostDecode: true,
        exactReadbackIdleReportedSeparatelyBecauseItOverlapsMapDecode: true,
        noOverlappingClockSubtractionUsed: true,
      }),
      tiedHeadCommands: Object.freeze(heads.map(publicCommandRecord)),
      readback: publicCommandRecord(readback),
      commands: Object.freeze(this.commands.map(publicCommandRecord)),
      copies: Object.freeze(copies),
      progressAuthority: Object.freeze({
        source: "transparent-device-proxy-observed-submit-drain-topology",
        role: "topology-corroboration-not-runtime-progress-substitution",
        explicitRuntimeProgressAttachedByCaller: true,
        completedCommandBuffers: this.commands.length,
        totalCommandBuffers: metadata.expectedCommandBufferCount,
        queueDrains: this.drainCount,
        monotonicallyCompletedOneCommandBufferPerDrain: true,
        stages: Object.freeze(this.commands.map((_, index) => (index === this.commands.length - 1 ? "readback" : "model"))),
      }),
      exactOverlappingReadbackIdle: Object.freeze({
        ...exactReadbackIdle,
        mapOverlapMilliseconds: Math.max(0, Math.min(exactReadbackIdle.endedAt, unmapAt) - Math.max(exactReadbackIdle.startedAt, map.mapStartedAt)),
        postUnmapDecodeResidualOverlapMilliseconds: Math.max(
          0,
          Math.min(exactReadbackIdle.endedAt, metadata.decodeEndedAt) - Math.max(exactReadbackIdle.startedAt, unmapAt),
        ),
        completedBeforeDecodeReturn: true,
        reportedAsOverlappingDiagnosticWithoutSubtraction: true,
      }),
    });
    this.finishTrace();
    return result;
  }

  endCancellationTrace(
    expected: Readonly<{
      commandBufferCount: number;
      physicalDispatchCount: number;
    }>,
  ): Readonly<Record<string, unknown>> {
    if (!this.active || this.pending.length !== 0) {
      throw new Error("OPT-0012 cancellation trace is not fully drained");
    }
    const last = this.commands.at(-1);
    if (last !== undefined && last.idleEndedAt === undefined) {
      last.idleEndedAt = performance.now();
    }
    const physicalDispatchCount = this.commands.reduce((sum, command) => sum + command.dispatches.length, 0);
    const copyCount = this.commands.reduce((sum, command) => sum + command.copies.length, 0);
    if (
      this.commands.length !== expected.commandBufferCount ||
      this.submissionCount !== expected.commandBufferCount ||
      this.drainCount !== expected.commandBufferCount ||
      physicalDispatchCount !== expected.physicalDispatchCount ||
      copyCount !== 0 ||
      this.maps.length !== 0
    ) {
      throw new Error("OPT-0012 cancellation crossed its drained boundary");
    }
    const result = Object.freeze({
      traceLabel: this.traceLabel,
      commandBufferCount: this.commands.length,
      submissionCount: this.submissionCount,
      queueDrainCount: this.drainCount,
      physicalDispatchCount,
      copyCount,
      mapCount: 0,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
      laterReadbackAndMapPrevented: true,
      commands: Object.freeze(this.commands.map(publicCommandRecord)),
    });
    this.finishTrace();
    return result;
  }

  abandonTrace(): void {
    this.finishTrace();
  }

  takeFullReadbackBytes(): ArrayBuffer {
    const bytes = this.fullReadbackBytes;
    this.fullReadbackBytes = null;
    if (bytes === null) {
      throw new Error("OPT-0012 arm A did not expose its mapped readback bytes");
    }
    return bytes;
  }

  resourceSnapshot(): Readonly<Record<string, unknown>> &
    Readonly<{
      liveTrackedBufferCount: number;
    }> {
    const records = [...this.bufferRecords.values()];
    const derivedLiveCount = records.filter((record) => !record.destroyed).length;
    const derivedLiveBytes = records.reduce((sum, record) => sum + (record.destroyed ? 0 : record.size), 0);
    if (derivedLiveCount !== this.liveTrackedBufferCount || derivedLiveBytes !== this.liveTrackedBufferBytes) {
      throw new Error("OPT-0012 simultaneous GPUBuffer residency accounting drifted");
    }
    return Object.freeze({
      createdBufferCount: records.length,
      destroyedBufferCount: records.filter((record) => record.destroyed).length,
      liveTrackedBufferCount: this.liveTrackedBufferCount,
      liveTrackedBufferBytes: this.liveTrackedBufferBytes,
      maximumSimultaneouslyLiveTrackedBufferCount: this.maximumLiveTrackedBufferCount,
      maximumSimultaneouslyLiveTrackedBufferBytes: this.maximumLiveTrackedBufferBytes,
      simultaneousHighWaterByBytes: Object.freeze({
        resourceEpoch: this.highWaterResourceEpoch,
        triggerLabel: this.highWaterTriggerLabel,
        liveBufferCountAtByteHighWater: this.highWaterLiveBufferCount,
        liveBufferBytes: this.maximumLiveTrackedBufferBytes,
      }),
      simultaneousHighWaterByCount: Object.freeze({
        resourceEpoch: this.countHighWaterResourceEpoch,
        triggerLabel: this.countHighWaterTriggerLabel,
        liveBufferCount: this.maximumLiveTrackedBufferCount,
        liveBufferBytesAtCountHighWater: this.countHighWaterLiveBufferBytes,
      }),
      totalDestroyCallCount: records.reduce((sum, record) => sum + record.destroyCallCount, 0),
      totalMapCallCount: records.reduce((sum, record) => sum + record.mapCallCount, 0),
      totalUnmapCallCount: records.reduce((sum, record) => sum + record.unmapCallCount, 0),
      destructionTrackingSupported: this.destructionTrackingSupported,
      mapTrackingSupported: this.mapTrackingSupported,
      records: Object.freeze(records.map((record) => Object.freeze({ ...record }))),
    });
  }

  finalResourceSummary(): Readonly<Record<string, unknown>> {
    const summary = this.resourceSnapshot();
    const lifecycle = validateOpt0012TrackedBufferCleanup([...this.bufferRecords.values()], {
      destructionTrackingSupported: this.destructionTrackingSupported,
      mapTrackingSupported: this.mapTrackingSupported,
    });
    return Object.freeze({
      ...summary,
      ...lifecycle,
    });
  }

  activitySnapshot(): Readonly<Record<string, number>> {
    return Object.freeze({
      lifetimeSubmitCallCount: this.lifetimeSubmitCallCount,
      lifetimeSubmittedCommandBufferCount: this.lifetimeSubmittedCommandBufferCount,
      lifetimeDrainCallCount: this.lifetimeDrainCallCount,
      lifetimeMapAsyncCallCount: this.lifetimeMapAsyncCallCount,
      resourceEpoch: this.resourceEpoch,
      liveTrackedBufferCount: this.liveTrackedBufferCount,
      liveTrackedBufferBytes: this.liveTrackedBufferBytes,
    });
  }

  private finishTrace(): void {
    this.active = false;
    this.traceLabel = "";
    this.captureFullReadback = false;
    this.commands = [];
    this.pending = [];
    this.submissionCount = 0;
    this.drainCount = 0;
    this.maximumOutstandingCommandBuffers = 0;
    this.maps = [];
  }

  private createTrackedBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.target.createBuffer(descriptor);
    const createdAtResourceEpoch = ++this.resourceEpoch;
    const record: TrackedBufferRecord = {
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      usage: Number(descriptor.usage),
      destroyCallCount: 0,
      mapCallCount: descriptor.mappedAtCreation === true ? 1 : 0,
      unmapCallCount: 0,
      destroyed: false,
      mapped: descriptor.mappedAtCreation === true,
      createdAtResourceEpoch,
      destroyedAtResourceEpoch: null,
    };
    this.bufferRecords.set(buffer, record);
    this.bufferLabels.set(buffer, record.label);
    this.liveTrackedBufferCount += 1;
    this.liveTrackedBufferBytes += record.size;
    if (
      this.liveTrackedBufferBytes > this.maximumLiveTrackedBufferBytes ||
      (this.liveTrackedBufferBytes === this.maximumLiveTrackedBufferBytes && this.liveTrackedBufferCount > this.maximumLiveTrackedBufferCount)
    ) {
      this.maximumLiveTrackedBufferBytes = this.liveTrackedBufferBytes;
      this.highWaterResourceEpoch = createdAtResourceEpoch;
      this.highWaterTriggerLabel = record.label;
      this.highWaterLiveBufferCount = this.liveTrackedBufferCount;
    }
    if (this.liveTrackedBufferCount > this.maximumLiveTrackedBufferCount) {
      this.maximumLiveTrackedBufferCount = this.liveTrackedBufferCount;
      this.countHighWaterResourceEpoch = createdAtResourceEpoch;
      this.countHighWaterTriggerLabel = record.label;
      this.countHighWaterLiveBufferBytes = this.liveTrackedBufferBytes;
    }
    const destroy = buffer.destroy.bind(buffer);
    const mapAsync = buffer.mapAsync.bind(buffer);
    const getMappedRange = buffer.getMappedRange.bind(buffer);
    const unmap = buffer.unmap.bind(buffer);
    try {
      Object.defineProperties(buffer, {
        destroy: {
          configurable: true,
          value: (): void => {
            record.destroyCallCount += 1;
            if (record.mapped) {
              throw new Error(`OPT-0012 destroyed mapped buffer ${record.label}`);
            }
            destroy();
            if (!record.destroyed) {
              record.destroyed = true;
              record.destroyedAtResourceEpoch = ++this.resourceEpoch;
              this.liveTrackedBufferCount -= 1;
              this.liveTrackedBufferBytes -= record.size;
              if (this.liveTrackedBufferCount < 0 || this.liveTrackedBufferBytes < 0) {
                throw new Error("OPT-0012 GPUBuffer residency underflowed");
              }
            }
          },
        },
        mapAsync: {
          configurable: true,
          value: async (...arguments_: Parameters<GPUBuffer["mapAsync"]>): Promise<void> => {
            record.mapCallCount += 1;
            this.lifetimeMapAsyncCallCount += 1;
            const mapRecord: MutableMapRecord | null = this.active ? { bufferLabel: record.label, mapStartedAt: performance.now() } : null;
            if (mapRecord !== null) {
              const last = this.commands.at(-1);
              if (last !== undefined && last.idleEndedAt === undefined) {
                last.idleEndedAt = mapRecord.mapStartedAt;
              }
              this.maps.push(mapRecord);
            }
            await mapAsync(...arguments_);
            record.mapped = true;
            if (mapRecord !== null) mapRecord.mapEndedAt = performance.now();
          },
        },
        getMappedRange: {
          configurable: true,
          value: (...arguments_: Parameters<GPUBuffer["getMappedRange"]>): ArrayBuffer => {
            const range = getMappedRange(...arguments_);
            if (this.active && this.captureFullReadback && record.label === "ace-planner-logit-readback") {
              this.fullReadbackBytes = range.slice(0);
            }
            return range;
          },
        },
        unmap: {
          configurable: true,
          value: (): void => {
            record.unmapCallCount += 1;
            record.mapped = false;
            if (this.active) {
              const mapRecord = [...this.maps].reverse().find((candidate) => candidate.bufferLabel === record.label && candidate.unmapAt === undefined);
              if (mapRecord !== undefined) mapRecord.unmapAt = performance.now();
            }
            unmap();
          },
        },
      });
    } catch {
      this.destructionTrackingSupported = false;
      this.mapTrackingSupported = false;
    }
    return buffer;
  }

  private createObservedCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
    const encoder = this.target.createCommandEncoder(descriptor);
    if (!this.active) return encoder;
    const now = performance.now();
    const prior = this.commands.at(-1);
    if (prior !== undefined && prior.idleEndedAt === undefined) {
      prior.idleEndedAt = now;
    }
    const record: MutableCommandRecord = {
      label: descriptor?.label ?? "",
      passLabels: [],
      dispatches: [],
      copies: [],
      clears: [],
      encodeStartedAt: now,
    };
    this.commands.push(record);
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (passDescriptor?: GPUComputePassDescriptor): GPUComputePassEncoder =>
            this.createObservedPass(target.beginComputePass(passDescriptor), record, passDescriptor?.label ?? "");
        }
        if (property === "copyBufferToBuffer") {
          return (source: GPUBuffer, sourceOffset: GPUSize64, destination: GPUBuffer, destinationOffset: GPUSize64, size: GPUSize64): void => {
            const copy = Object.freeze({
              index: record.copies.length,
              sourceBufferLabel: this.bufferLabels.get(source) ?? source.label,
              sourceOffset: Number(sourceOffset),
              destinationBufferLabel: this.bufferLabels.get(destination) ?? destination.label,
              destinationOffset: Number(destinationOffset),
              copiedBytes: Number(size),
            });
            record.copies.push(copy);
            target.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
          };
        }
        if (property === "clearBuffer") {
          return (buffer: GPUBuffer, offset = 0, size?: GPUSize64): void => {
            const resolvedSize = size === undefined ? buffer.size - offset : Number(size);
            record.clears.push(
              Object.freeze({
                bufferLabel: this.bufferLabels.get(buffer) ?? buffer.label,
                offset: Number(offset),
                size: resolvedSize,
              }),
            );
            target.clearBuffer(buffer, offset, size);
          };
        }
        if (property === "finish") {
          return (finishDescriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer => {
            record.encodeEndedAt = performance.now();
            const command = target.finish(finishDescriptor);
            this.commandByBuffer.set(command, record);
            return command;
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function" ? (value as (...args: unknown[]) => unknown).bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  private createObservedPass(pass: GPUComputePassEncoder, record: MutableCommandRecord, passLabel: string): GPUComputePassEncoder {
    record.passLabels.push(passLabel);
    let pipelineLabel = "";
    let bindGroupLabel = "";
    return new Proxy(pass, {
      get: (target, property) => {
        if (property === "setPipeline") {
          return (pipeline: GPUComputePipeline): void => {
            pipelineLabel = pipeline.label;
            target.setPipeline(pipeline);
          };
        }
        if (property === "setBindGroup") {
          return (index: GPUIndex32, bindGroup: GPUBindGroup | null, dynamicOffsets?: Iterable<GPUBufferDynamicOffset>): void => {
            if (index === 0 && bindGroup !== null) bindGroupLabel = bindGroup.label;
            if (dynamicOffsets === undefined) target.setBindGroup(index, bindGroup);
            else target.setBindGroup(index, bindGroup, dynamicOffsets);
          };
        }
        if (property === "dispatchWorkgroups") {
          return (x: GPUSize32, y: GPUSize32 = 1, z: GPUSize32 = 1): void => {
            if (pipelineLabel === "" || bindGroupLabel === "") {
              throw new Error("OPT-0012 dispatch omitted pipeline/bind-group identity");
            }
            record.dispatches.push(
              Object.freeze({
                index: record.dispatches.length,
                passLabel,
                pipelineLabel,
                bindGroupLabel,
                workgroups: Object.freeze([Number(x), Number(y), Number(z)] as const),
              }),
            );
            target.dispatchWorkgroups(x, y, z);
          };
        }
        if (property === "dispatchWorkgroupsIndirect") {
          return (): never => {
            throw new Error("OPT-0012 traced an unexpected indirect dispatch");
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function" ? (value as (...args: unknown[]) => unknown).bind(target) : value;
      },
    }) as GPUComputePassEncoder;
  }

  private createQueueProxy(target: GPUQueue): GPUQueue {
    return new Proxy(target, {
      get: (queue, property) => {
        if (property === "submit") {
          return (commands: Iterable<GPUCommandBuffer>): void => {
            const list = [...commands];
            this.lifetimeSubmitCallCount += 1;
            this.lifetimeSubmittedCommandBufferCount += list.length;
            if (this.active) {
              if (list.length !== 1 || this.pending.length !== 0) {
                throw new Error("OPT-0012 violated one-outstanding-command-buffer policy");
              }
              const record = this.commandByBuffer.get(list[0]!);
              if (record === undefined) {
                throw new Error("OPT-0012 submitted an unobserved command buffer");
              }
              record.submitStartedAt = performance.now();
              this.pending.push(record);
              this.submissionCount += 1;
              this.maximumOutstandingCommandBuffers = Math.max(this.maximumOutstandingCommandBuffers, this.pending.length);
            }
            queue.submit(list);
            if (this.active) this.pending[0]!.submitReturnedAt = performance.now();
          };
        }
        if (property === "onSubmittedWorkDone") {
          return async (): Promise<void> => {
            this.lifetimeDrainCallCount += 1;
            const pending = this.active ? [...this.pending] : [];
            const startedAt = performance.now();
            for (const record of pending) record.drainStartedAt = startedAt;
            await queue.onSubmittedWorkDone();
            if (this.active) {
              const endedAt = performance.now();
              for (const record of pending) record.drainEndedAt = endedAt;
              this.drainCount += pending.length;
              this.pending.splice(0, pending.length);
            }
          };
        }
        if (property === "writeBuffer") {
          return (...arguments_: Parameters<GPUQueue["writeBuffer"]>): void => {
            if (this.active && this.pending.length !== 0) {
              throw new Error("OPT-0012 queue.writeBuffer raced recorded GPU work");
            }
            queue.writeBuffer(...arguments_);
          };
        }
        const value = Reflect.get(queue, property, queue) as unknown;
        return typeof value === "function" ? (value as (...args: unknown[]) => unknown).bind(queue) : value;
      },
    }) as GPUQueue;
  }
}

function publicCommandRecord(record: MutableCommandRecord): Readonly<Record<string, unknown>> {
  const encodeEndedAt = requiredTimestamp(record.encodeEndedAt, "encode end");
  const submitStartedAt = requiredTimestamp(record.submitStartedAt, "submit start");
  const submitReturnedAt = requiredTimestamp(record.submitReturnedAt, "submit return");
  const drainStartedAt = requiredTimestamp(record.drainStartedAt, "drain start");
  const drainEndedAt = requiredTimestamp(record.drainEndedAt, "drain end");
  const idleEndedAt = requiredTimestamp(record.idleEndedAt, "post-drain gap end");
  return Object.freeze({
    label: record.label,
    passLabels: Object.freeze([...record.passLabels]),
    dispatches: Object.freeze([...record.dispatches]),
    copies: Object.freeze([...record.copies]),
    clears: Object.freeze([...record.clears]),
    encodeStartedAt: record.encodeStartedAt,
    encodeEndedAt,
    submitStartedAt,
    submitReturnedAt,
    drainStartedAt,
    drainEndedAt,
    observedPostDrainGapStartedAt: drainEndedAt,
    observedPostDrainGapEndedAt: idleEndedAt,
    encodeMilliseconds: encodeEndedAt - record.encodeStartedAt,
    submitMilliseconds: submitReturnedAt - submitStartedAt,
    drainMilliseconds: drainEndedAt - drainStartedAt,
    observedPostDrainGapMilliseconds: idleEndedAt - drainEndedAt,
  });
}

function summarizeCommandIntervals(commands: readonly MutableCommandRecord[]): Readonly<Record<string, number>> {
  let encodeMilliseconds = 0;
  let submitMilliseconds = 0;
  let drainMilliseconds = 0;
  let observedPostDrainGapMilliseconds = 0;
  for (const command of commands) {
    const encodeEndedAt = requiredTimestamp(command.encodeEndedAt, "encode end");
    const submitStartedAt = requiredTimestamp(command.submitStartedAt, "submit start");
    const submitReturnedAt = requiredTimestamp(command.submitReturnedAt, "submit return");
    const drainStartedAt = requiredTimestamp(command.drainStartedAt, "drain start");
    const drainEndedAt = requiredTimestamp(command.drainEndedAt, "drain end");
    const idleEndedAt = requiredTimestamp(command.idleEndedAt, "post-drain gap end");
    encodeMilliseconds += encodeEndedAt - command.encodeStartedAt;
    submitMilliseconds += submitReturnedAt - submitStartedAt;
    drainMilliseconds += drainEndedAt - drainStartedAt;
    observedPostDrainGapMilliseconds += idleEndedAt - drainEndedAt;
  }
  return Object.freeze({
    commandBufferCount: commands.length,
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    observedPostDrainGapMilliseconds,
  });
}

type CancellationBoundary = "after-pre-head" | "after-head" | null;

interface CandidateInvocation {
  readonly arm: Opt0012Arm;
  readonly phaseKind: "prefill" | "decode";
  readonly plan: AceOpt0012CompactSemanticHeadPlan;
  readonly sentinel: number;
  readonly guardProof: boolean;
  readonly cancellationBoundary: CancellationBoundary;
  readonly boundaryAbortController: AbortController | null;
  readonly allowCorrectnessOnlyInsideTracePreparation: boolean;
}

interface CandidateOutputHandle {
  readonly buffer: GPUBuffer;
  readonly payloadOffset: number;
  readonly payloadBytes: number;
  readonly totalBytes: number;
  readonly sentinel: number;
  readonly label: string;
}

interface CandidateReceipt {
  readonly decoded: AceOpt0012DecodedCompactLogits;
  readonly mappedBytes: ArrayBuffer;
  readonly outputs: readonly CandidateOutputHandle[];
  readonly readback: GPUBuffer;
  readonly hostDecodeMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface PreparedCandidateInvocationResources {
  readonly phase: AcePlannerPreparedPhaseGpuResources;
  readonly plan: AceOpt0012CompactSemanticHeadPlan;
  readonly sentinel: number;
  readonly outputs: readonly CandidateOutputHandle[];
  readonly readback: GPUBuffer;
  readonly dispatches: readonly AceGemmDispatch[];
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface PrivateExecutorRunAndReadback {
  runAndReadback(phase: AcePlannerPreparedPhaseGpuResources, dispatch: AcePlannerModelDispatch, clearCache: boolean): Promise<readonly Float32Array[]>;
}

interface Opt0012CandidateProgressQuantum {
  readonly id: string;
  readonly logicalId?: string;
  readonly kind: string;
  readonly layer: number | null;
  readonly primitiveCount: number;
}

interface MutableOpt0012IdleInterval {
  readonly source: "production-executor-yieldQueueIdle" | "benchmark-candidate-equivalent-yieldQueueIdle";
  readonly startedAt: number;
  readonly requestedMilliseconds: number;
  stage: "model" | "readback" | null;
  endedAt?: number;
}

interface Opt0012IdleIntervalReceipt {
  readonly source: MutableOpt0012IdleInterval["source"];
  readonly stage: "model" | "readback";
  readonly startedAt: number;
  readonly endedAt: number;
  readonly requestedMilliseconds: number;
  readonly elapsedMilliseconds: number;
  readonly completed: true;
}

interface Opt0012ProgressReceipt extends Readonly<Record<string, unknown>> {
  readonly complete: boolean;
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly idleIntervals: readonly Opt0012IdleIntervalReceipt[];
  readonly readbackIdle: Opt0012IdleIntervalReceipt | null;
}

class Opt0012ProgressRecorder {
  private active: Readonly<{
    arm: Opt0012Arm;
    phaseKind: "prefill" | "decode";
    totalCommandBuffers: number;
  }> | null = null;
  private events: Array<Readonly<Record<string, unknown>>> = [];
  private idleIntervals: MutableOpt0012IdleInterval[] = [];
  private idleCompletions: Promise<void>[] = [];
  private lifetimeCompletedCommandBuffers = 0;
  private lifetimeQueueDrains = 0;
  private lifetimeCooperativeIdleMilliseconds = 0;
  private lastPeakAccountedGpuBytes = 0;

  begin(invocation: CandidateInvocation): void {
    if (this.active !== null || this.events.length !== 0 || this.idleIntervals.length !== 0 || this.idleCompletions.length !== 0) {
      throw new DOMException("OPT-0012 progress trace overlaps", "InvalidStateError");
    }
    const topology = opt0012ExpectedCommandBufferTopology(invocation.arm, invocation.phaseKind);
    this.active = Object.freeze({
      arm: invocation.arm,
      phaseKind: invocation.phaseKind,
      totalCommandBuffers: topology.totalCommandBufferCount,
    });
  }

  yieldProductionIdle(): Promise<void> {
    return this.yieldIdle("production-executor-yieldQueueIdle");
  }

  yieldCandidateIdle(): Promise<void> {
    return this.yieldIdle("benchmark-candidate-equivalent-yieldQueueIdle");
  }

  acceptProduction(progress: AcePlannerGpuExecutorProgress): void {
    const active = this.active;
    if (active === null) return;
    if (active.arm !== "A") {
      throw new Error("OPT-0012 candidate unexpectedly emitted production progress");
    }
    this.lastPeakAccountedGpuBytes = progress.peakAccountedGpuBytes;
    this.accept({
      phaseKind: progress.phaseKind,
      completedCommandBuffers: progress.completedCommandBuffers,
      totalCommandBuffers: progress.totalCommandBuffers,
      queueDrains: progress.queueDrains,
      cooperativeIdleMs: progress.cooperativeIdleMs,
      stage: progress.stage,
      quantum: progress.quantum,
      source: "production-executor-onProgress",
      productionCumulativeQueueDrains: progress.cumulativeQueueDrains,
      productionCumulativeCooperativeIdleMs: progress.cumulativeCooperativeIdleMs,
    });
  }

  acceptCandidate(stage: "model" | "readback", quantum: Opt0012CandidateProgressQuantum | null): void {
    const active = this.active;
    if (active === null || active.arm === "A") {
      throw new Error("OPT-0012 candidate progress has no candidate invocation");
    }
    const completed = this.events.length + 1;
    this.accept({
      phaseKind: active.phaseKind,
      completedCommandBuffers: completed,
      totalCommandBuffers: active.totalCommandBuffers,
      queueDrains: completed,
      cooperativeIdleMs: completed,
      stage,
      quantum,
      source: "benchmark-candidate-equivalent-progress",
      productionCumulativeQueueDrains: null,
      productionCumulativeCooperativeIdleMs: null,
    });
  }

  finish(complete: boolean): Opt0012ProgressReceipt {
    const active = this.active;
    if (active === null) {
      throw new DOMException("OPT-0012 progress trace is absent", "InvalidStateError");
    }
    const events = Object.freeze([...this.events]);
    const expectedCompleted = complete ? active.totalCommandBuffers : events.length;
    const idleIntervals = this.idleIntervals.map((interval, index): Opt0012IdleIntervalReceipt => {
      if (interval.endedAt === undefined) {
        throw new Error(`OPT-0012 cooperative idle ${index} did not complete`);
      }
      if (interval.stage === null) {
        throw new Error(`OPT-0012 cooperative idle ${index} has no accepted progress stage`);
      }
      return Object.freeze({
        source: interval.source,
        stage: interval.stage,
        startedAt: interval.startedAt,
        endedAt: interval.endedAt,
        requestedMilliseconds: interval.requestedMilliseconds,
        elapsedMilliseconds: interval.endedAt - interval.startedAt,
        completed: true as const,
      });
    });
    const expectedIdleSource = active.arm === "A" ? "production-executor-yieldQueueIdle" : "benchmark-candidate-equivalent-yieldQueueIdle";
    if (
      events.length !== expectedCompleted ||
      idleIntervals.length !== events.length ||
      (complete && events.at(-1)?.stage !== "readback") ||
      idleIntervals.some(
        (interval, index) =>
          interval.source !== expectedIdleSource ||
          interval.requestedMilliseconds !== ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS ||
          interval.elapsedMilliseconds < 0 ||
          interval.stage !== events[index]!.stage,
      ) ||
      events.some(
        (event, index) =>
          event.completedCommandBuffers !== index + 1 ||
          event.totalCommandBuffers !== active.totalCommandBuffers ||
          event.queueDrains !== index + 1 ||
          event.cooperativeIdleMs !== index + 1 ||
          (index < events.length - 1 && event.stage !== "model"),
      )
    ) {
      throw new Error("OPT-0012 explicit progress sequence changed");
    }
    this.active = null;
    this.events = [];
    this.idleIntervals = [];
    this.idleCompletions = [];
    return Object.freeze({
      arm: active.arm,
      phaseKind: active.phaseKind,
      complete,
      completedCommandBuffers: events.length,
      totalCommandBuffers: active.totalCommandBuffers,
      queueDrains: events.length,
      cooperativeIdleMilliseconds: events.length,
      lifetimeCompletedCommandBuffers: this.lifetimeCompletedCommandBuffers,
      lifetimeQueueDrains: this.lifetimeQueueDrains,
      lifetimeCooperativeIdleMilliseconds: this.lifetimeCooperativeIdleMilliseconds,
      lastPeakAccountedGpuBytes: this.lastPeakAccountedGpuBytes,
      monotonicExact: true,
      events,
      idleIntervals: Object.freeze(idleIntervals),
      readbackIdle: complete ? idleIntervals.at(-1)! : null,
      everyRealIdleTimerCompleted: true,
    });
  }

  async settleIdleAfterFailure(): Promise<void> {
    const completions = [...this.idleCompletions];
    await Promise.all(completions);
    if (completions.length !== this.idleIntervals.length || this.idleIntervals.some((interval) => interval.endedAt === undefined)) {
      throw new Error("OPT-0012 failed to settle real cooperative idles");
    }
  }

  private yieldIdle(source: MutableOpt0012IdleInterval["source"]): Promise<void> {
    const active = this.active;
    if (active === null) {
      return new Promise((resolve) => {
        setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
      });
    }
    const expectedSource = active.arm === "A" ? "production-executor-yieldQueueIdle" : "benchmark-candidate-equivalent-yieldQueueIdle";
    if (source !== expectedSource || this.idleIntervals.length !== this.events.length) {
      throw new Error("OPT-0012 cooperative idle sequence changed");
    }
    const interval: MutableOpt0012IdleInterval = {
      source,
      startedAt: performance.now(),
      requestedMilliseconds: ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      stage: null,
    };
    this.idleIntervals.push(interval);
    const completion = new Promise<void>((resolve) => {
      setTimeout(() => {
        interval.endedAt = performance.now();
        resolve();
      }, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
    });
    this.idleCompletions.push(completion);
    return completion;
  }

  private accept(
    input: Readonly<{
      phaseKind: "prefill" | "decode";
      completedCommandBuffers: number;
      totalCommandBuffers: number;
      queueDrains: number;
      cooperativeIdleMs: number;
      stage: "model" | "readback";
      quantum: Opt0012CandidateProgressQuantum | null;
      source: string;
      productionCumulativeQueueDrains: number | null;
      productionCumulativeCooperativeIdleMs: number | null;
    }>,
  ): void {
    const active = this.active;
    const expectedCompleted = this.events.length + 1;
    if (
      active === null ||
      input.phaseKind !== active.phaseKind ||
      input.totalCommandBuffers !== active.totalCommandBuffers ||
      input.completedCommandBuffers !== expectedCompleted ||
      input.queueDrains !== expectedCompleted ||
      input.cooperativeIdleMs !== expectedCompleted ||
      (input.stage === "model" && input.quantum === null) ||
      (input.stage === "readback" && input.quantum !== null)
    ) {
      throw new Error("OPT-0012 runtime progress changed");
    }
    const idle = this.idleIntervals[input.completedCommandBuffers - 1];
    if (idle === undefined || idle.stage !== null) {
      throw new Error("OPT-0012 runtime progress has no matching real idle timer");
    }
    idle.stage = input.stage;
    this.lifetimeCompletedCommandBuffers += 1;
    this.lifetimeQueueDrains += 1;
    this.lifetimeCooperativeIdleMilliseconds += 1;
    this.events.push(
      Object.freeze({
        ...input,
        quantum:
          input.quantum === null
            ? null
            : Object.freeze({
                id: input.quantum.id,
                logicalId: input.quantum.logicalId ?? null,
                kind: input.quantum.kind,
                layer: input.quantum.layer,
                primitiveCount: input.quantum.primitiveCount,
              }),
        harnessCumulativeCompletedCommandBuffers: this.lifetimeCompletedCommandBuffers,
        harnessCumulativeQueueDrains: this.lifetimeQueueDrains,
        harnessCumulativeCooperativeIdleMilliseconds: this.lifetimeCooperativeIdleMilliseconds,
        peakAccountedGpuBytes: this.lastPeakAccountedGpuBytes,
      }),
    );
  }
}

function requireCompleteReadbackIdle(progress: Opt0012ProgressReceipt): Opt0012IdleIntervalReceipt {
  const idle = progress.readbackIdle;
  if (idle === null || idle.stage !== "readback" || !idle.completed || idle !== progress.idleIntervals.at(-1)) {
    throw new Error("OPT-0012 exact readback idle receipt is absent");
  }
  return idle;
}

function installOpt0012ExecutorIdleHook(executor: AcePlannerGpuExecutor, progress: Opt0012ProgressRecorder): void {
  const resources = Reflect.get(executor as unknown as object, "resources") as AcePlannerPreparedGpuExecutorResources | undefined;
  if (resources === undefined || typeof resources !== "object" || resources.yieldQueueIdle !== undefined) {
    throw new Error("OPT-0012 authenticated executor idle seam changed");
  }
  const hook = (): Promise<void> => progress.yieldProductionIdle();
  Object.defineProperty(resources, "yieldQueueIdle", {
    configurable: true,
    enumerable: true,
    value: hook,
  });
  if (resources.yieldQueueIdle !== hook) {
    throw new Error("OPT-0012 executor idle hook did not install");
  }
}

class CandidateHeadRunner {
  private readonly kernel: AceCorrectnessGemmKernel;
  private readonly privateExecutor: PrivateExecutorRunAndReadback;
  private readonly originalRunAndReadback: PrivateExecutorRunAndReadback["runAndReadback"];
  private activeInvocation: CandidateInvocation | null = null;
  private receipt: CandidateReceipt | null = null;
  private pendingGuardOutputs: readonly CandidateOutputHandle[] = [];
  private pendingCompactReadback: GPUBuffer | null = null;
  private preparedCandidateResources: PreparedCandidateInvocationResources | null = null;
  private progressReceipt: Opt0012ProgressReceipt | null = null;
  private destroyed = false;

  constructor(
    private readonly observer: Opt0012DeviceObserver,
    executor: AcePlannerGpuExecutor,
    private readonly progress: Opt0012ProgressRecorder,
    private readonly signal: AbortSignal,
    private readonly regularPlan: AceOpt0012CompactSemanticHeadPlan,
    private readonly eosPlan: AceOpt0012CompactSemanticHeadPlan,
  ) {
    this.kernel = AceCorrectnessGemmKernel.create(observer.device, "raw-fp16", "source-row-major");
    this.privateExecutor = executor as unknown as PrivateExecutorRunAndReadback;
    const original = Reflect.get(executor as unknown as object, "runAndReadback") as unknown;
    if (typeof original !== "function") {
      this.kernel.destroy();
      throw new Error("OPT-0012 authenticated executor seam is absent");
    }
    this.originalRunAndReadback = (original as PrivateExecutorRunAndReadback["runAndReadback"]).bind(executor);
    Object.defineProperty(executor, "runAndReadback", {
      configurable: true,
      value: async (phase: AcePlannerPreparedPhaseGpuResources, dispatch: AcePlannerModelDispatch, clearCache: boolean): Promise<readonly Float32Array[]> => {
        const invocation = this.activeInvocation;
        if (invocation === null) {
          return await this.originalRunAndReadback(phase, dispatch, clearCache);
        }
        if (invocation.arm === "A") {
          return await this.originalRunAndReadback(phase, dispatch, clearCache);
        }
        return await this.runCandidate(phase, dispatch, clearCache, invocation);
      },
    });
  }

  async invoke(operation: () => Promise<readonly ArrayLike<number>[]>, invocation: CandidateInvocation): Promise<readonly ArrayLike<number>[]> {
    if (this.destroyed || this.activeInvocation !== null || this.receipt !== null || this.progressReceipt !== null) {
      throw new DOMException("OPT-0012 candidate invocation overlaps", "InvalidStateError");
    }
    if (invocation.plan !== this.regularPlan && invocation.plan !== this.eosPlan) {
      throw new Error("OPT-0012 candidate invocation requires an authenticated plan");
    }
    if ((invocation.cancellationBoundary === null) !== (invocation.boundaryAbortController === null)) {
      throw new Error("OPT-0012 cancellation boundary/controller pairing changed");
    }
    if (
      (invocation.arm === "A" && this.preparedCandidateResources !== null) ||
      (invocation.arm !== "A" && this.preparedCandidateResources === null && !invocation.allowCorrectnessOnlyInsideTracePreparation)
    ) {
      throw new Error("OPT-0012 candidate resources were not prepared before timing");
    }
    this.activeInvocation = invocation;
    this.progress.begin(invocation);
    try {
      const result = await operation();
      this.progressReceipt = this.progress.finish(true);
      return result;
    } catch (error) {
      if (this.preparedCandidateResources !== null) {
        this.destroyPreparedCandidateResources(this.preparedCandidateResources);
        this.preparedCandidateResources = null;
      }
      try {
        await this.progress.settleIdleAfterFailure();
        this.progressReceipt = this.progress.finish(false);
      } catch (progressError) {
        const primary = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
        const progressFailure = progressError instanceof Error ? `${progressError.name}: ${progressError.message}` : String(progressError);
        throw new Error(`OPT-0012 progress failure settlement failed; primary=${primary}; ` + `settlement=${progressFailure}`, { cause: error });
      }
      throw error;
    } finally {
      this.activeInvocation = null;
    }
  }

  takeProgressReceipt(expectedCompletedCommandBuffers: number, complete: boolean): Opt0012ProgressReceipt {
    const receipt = this.progressReceipt;
    this.progressReceipt = null;
    if (receipt === null || receipt.complete !== complete || receipt.completedCommandBuffers !== expectedCompletedCommandBuffers) {
      throw new Error("OPT-0012 explicit progress receipt changed");
    }
    return receipt;
  }

  async prepareCandidateResourcesOutsidePrimaryWall(plan: AceOpt0012CompactSemanticHeadPlan, sentinel: number): Promise<Readonly<Record<string, unknown>>> {
    if (this.destroyed || this.activeInvocation !== null || this.preparedCandidateResources !== null || this.receipt !== null) {
      throw new DOMException("OPT-0012 candidate resource preparation overlaps", "InvalidStateError");
    }
    const phase = this.requireActivePhaseResources();
    const prepared = await this.createPreparedCandidateResources(phase, plan, sentinel, true);
    this.preparedCandidateResources = prepared;
    return prepared.receipt;
  }

  async poisonActiveWriteStatusOutsidePrimaryWall(): Promise<
    Readonly<{
      poisonedWords: readonly [0, 0];
      fenced: true;
      milliseconds: number;
    }>
  > {
    if (this.destroyed || this.activeInvocation !== null) {
      throw new DOMException("OPT-0012 cannot poison active status", "InvalidStateError");
    }
    const active = Reflect.get(this.privateExecutor as unknown as object, "activePhase") as unknown;
    const resources = typeof active === "object" && active !== null ? (Reflect.get(active, "resources") as unknown) : undefined;
    const bindings = typeof resources === "object" && resources !== null ? (Reflect.get(resources, "bindings") as unknown) : undefined;
    const controls = typeof bindings === "object" && bindings !== null ? (Reflect.get(bindings, "controls") as unknown) : undefined;
    const writeStatus = typeof controls === "object" && controls !== null ? (Reflect.get(controls, "writeStatus") as GPUBufferBinding | undefined) : undefined;
    if (writeStatus === undefined) {
      throw new Error("OPT-0012 active phase write-status seam changed");
    }
    const started = performance.now();
    zeroStatusBinding(this.observer.device.queue, writeStatus);
    await this.observer.device.queue.onSubmittedWorkDone();
    return Object.freeze({
      poisonedWords: Object.freeze([0, 0] as const),
      fenced: true,
      milliseconds: performance.now() - started,
    });
  }

  describeActiveControlClears(): readonly Opt0012ExpectedClearDescriptor[] {
    const controls = this.requireActivePhaseResources().bindings.controls;
    const describe = (binding: GPUBufferBinding): Opt0012ExpectedClearDescriptor =>
      Object.freeze({
        bufferLabel: binding.buffer.label,
        offset: binding.offset ?? 0,
        size: binding.size ?? binding.buffer.size - (binding.offset ?? 0),
      });
    const result = Object.freeze([describe(controls.cacheValidity), describe(controls.writeStatus)]);
    if (result[1]!.size !== 8 || result[0]!.size <= result[1]!.size || result[0]!.bufferLabel === result[1]!.bufferLabel) {
      throw new Error("OPT-0012 active control clear layout changed");
    }
    return result;
  }

  activeCachePublicationSnapshot(): Readonly<Record<string, unknown>> {
    const active = Reflect.get(this.privateExecutor as unknown as object, "activePhase") as unknown;
    if (active === undefined) {
      return Object.freeze({ published: false });
    }
    if (typeof active !== "object" || active === null) {
      throw new Error("OPT-0012 active cache publication seam changed");
    }
    const resources = Reflect.get(active, "resources") as AcePlannerPreparedPhaseGpuResources | undefined;
    const cachedTokens = Reflect.get(active, "cachedTokens") as unknown;
    if (resources === undefined || !Number.isSafeInteger(cachedTokens) || typeof cachedTokens !== "number") {
      throw new Error("OPT-0012 active cache publication receipt changed");
    }
    return Object.freeze({
      published: true,
      cachedTokens,
      prefillTokens: resources.prefillTokens,
      cacheCapacity: resources.cacheCapacity,
      batch: resources.batch,
    });
  }

  takeReceipt(): CandidateReceipt {
    const receipt = this.receipt;
    this.receipt = null;
    if (receipt === null) throw new Error("OPT-0012 candidate receipt is absent");
    this.pendingGuardOutputs = receipt.outputs;
    this.pendingCompactReadback = receipt.readback;
    return receipt;
  }

  async verifyAndReleasePendingGuards(): Promise<Readonly<Record<string, unknown>>> {
    const outputs = this.pendingGuardOutputs;
    const compactReadback = this.pendingCompactReadback;
    this.pendingGuardOutputs = [];
    this.pendingCompactReadback = null;
    if (outputs.length === 0 || compactReadback === null) {
      throw new Error("OPT-0012 guard proof has no candidate outputs");
    }
    const totalBytes = sumSafe(
      outputs.map((output) => output.totalBytes),
      "OPT-0012 guard bytes",
    );
    const readback = this.observer.device.createBuffer({
      label: "opt-0012-candidate-output-guard-readback",
      size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.observer.device.createCommandEncoder({
        label: "opt-0012-candidate-output-guard-readback-command",
      });
      let destinationOffset = 0;
      for (const output of outputs) {
        encoder.copyBufferToBuffer(output.buffer, 0, readback, destinationOffset, output.totalBytes);
        destinationOffset += output.totalBytes;
      }
      this.observer.device.queue.submit([encoder.finish()]);
      await this.observer.device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(readback.getMappedRange()).slice();
      readback.unmap();
      let cursor = 0;
      for (const output of outputs) {
        requireSentinel(bytes.subarray(cursor, cursor + output.payloadOffset), output.sentinel, `${output.label} prefix guard`);
        const tailStart = cursor + output.payloadOffset + output.payloadBytes;
        requireSentinel(bytes.subarray(tailStart, cursor + output.totalBytes), output.sentinel, `${output.label} tail guard`);
        cursor += output.totalBytes;
      }
      return Object.freeze({
        outputCount: outputs.length,
        guardBytesCompared: outputs.length * CANDIDATE_OUTPUT_GUARD_BYTES * 2,
        separateUntimedGuardCopyCommandBufferCount: 1,
        prefixAndTailUnchanged: true,
      });
    } finally {
      readback.destroy();
      compactReadback.destroy();
      for (const output of outputs) output.buffer.destroy();
    }
  }

  releasePendingGuardsWithoutReadback(): void {
    for (const output of this.pendingGuardOutputs) output.buffer.destroy();
    this.pendingGuardOutputs = [];
    this.pendingCompactReadback?.destroy();
    this.pendingCompactReadback = null;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.progressReceipt = null;
    this.releasePendingGuardsWithoutReadback();
    if (this.preparedCandidateResources !== null) {
      this.destroyPreparedCandidateResources(this.preparedCandidateResources);
      this.preparedCandidateResources = null;
    }
    this.kernel.destroy();
    Object.defineProperty(this.privateExecutor, "runAndReadback", {
      configurable: true,
      value: this.originalRunAndReadback,
    });
  }

  private requireActivePhaseResources(): AcePlannerPreparedPhaseGpuResources {
    const active = Reflect.get(this.privateExecutor as unknown as object, "activePhase") as unknown;
    const resources = typeof active === "object" && active !== null ? (Reflect.get(active, "resources") as unknown) : undefined;
    if (typeof resources !== "object" || resources === null) {
      throw new Error("OPT-0012 authenticated active phase seam changed");
    }
    return resources as AcePlannerPreparedPhaseGpuResources;
  }

  private async createPreparedCandidateResources(
    phase: AcePlannerPreparedPhaseGpuResources,
    plan: AceOpt0012CompactSemanticHeadPlan,
    sentinel: number,
    preparedOutsidePrimaryWall: boolean,
  ): Promise<PreparedCandidateInvocationResources> {
    if (plan !== this.regularPlan && plan !== this.eosPlan) {
      throw new Error("OPT-0012 candidate preparation plan changed");
    }
    this.validateSourceBindings(phase, plan);
    const started = performance.now();
    const outputs = plan.headSlices.map((slice) => this.createOutput(`opt-0012-${plan.state}-shard-${slice.shardIndex}-output`, slice.rawLogitBytes, sentinel));
    const readback = createSentinelBuffer(
      this.observer.device,
      "opt-0012-compact-head-readback",
      plan.readback.allocationBytes,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      0x3c,
    );
    try {
      const dispatches = await Promise.all(
        plan.headSlices.map(async (slice, index): Promise<AceGemmDispatch> => {
          const owner = phase.bindings.weights.embedding[slice.shardIndex]!;
          return await this.kernel.createDispatch(
            `opt-0012-${plan.state}-shard-${slice.shardIndex}`,
            { rows: 2, inner: plan.hiddenSize, columns: slice.rowCount },
            {
              activation: phase.bindings.scratch.lastHiddenRows,
              weight: {
                buffer: owner.weight.buffer,
                offset: slice.sourceBindingByteOffset,
                size: slice.sourceBindingByteLength,
              },
              output: {
                buffer: outputs[index]!.buffer,
                offset: outputs[index]!.payloadOffset,
                size: outputs[index]!.payloadBytes,
              },
            },
          );
        }),
      );
      const preparationMilliseconds = performance.now() - started;
      const receipt = Object.freeze({
        state: plan.state,
        preparedOutsidePrimaryWall,
        preparationMilliseconds,
        outputBufferCount: outputs.length,
        outputPayloadBytes: sumSafe(
          outputs.map((output) => output.payloadBytes),
          "OPT-0012 candidate output payload bytes",
        ),
        outputAllocationBytes: sumSafe(
          outputs.map((output) => output.totalBytes),
          "OPT-0012 candidate output allocation bytes",
        ),
        compactReadbackAllocationBytes: plan.readback.allocationBytes,
        gemmDispatchCount: dispatches.length,
        everyOutputAndReadbackSentinelInitializedByMappedCreation: true,
        noAllocationMapFillOrDispatchConstructionInPrimaryWall: preparedOutsidePrimaryWall,
      });
      return Object.freeze({
        phase,
        plan,
        sentinel,
        outputs: Object.freeze(outputs),
        readback,
        dispatches: Object.freeze(dispatches),
        receipt,
      });
    } catch (error) {
      readback.destroy();
      for (const output of outputs) output.buffer.destroy();
      throw error;
    }
  }

  private destroyPreparedCandidateResources(prepared: PreparedCandidateInvocationResources): void {
    prepared.readback.destroy();
    for (const output of prepared.outputs) output.buffer.destroy();
  }

  private async runCandidate(
    phase: AcePlannerPreparedPhaseGpuResources,
    dispatch: AcePlannerModelDispatch,
    clearCache: boolean,
    invocation: CandidateInvocation,
  ): Promise<readonly Float32Array[]> {
    const plan = invocation.plan;
    if (dispatch.plan.batch !== 2 || phase.batch !== 2) {
      throw new Error("OPT-0012 candidate is not an authenticated M2 dispatch");
    }
    this.validateSourceBindings(phase, plan);
    let prepared = this.preparedCandidateResources;
    this.preparedCandidateResources = null;
    if (prepared === null) {
      if (!invocation.allowCorrectnessOnlyInsideTracePreparation) {
        throw new Error("OPT-0012 candidate entered timing without prepared resources");
      }
      prepared = await this.createPreparedCandidateResources(phase, plan, invocation.sentinel, false);
    }
    if (prepared.phase !== phase || prepared.plan !== plan || prepared.sentinel !== invocation.sentinel) {
      this.destroyPreparedCandidateResources(prepared);
      throw new Error("OPT-0012 prepared candidate resource identity changed");
    }
    const outputs = prepared.outputs;
    const readback = prepared.readback;
    const dispatches = prepared.dispatches;
    try {
      const preHeadCount = authenticateOpt0012ProductionHeadTail(dispatch.label, dispatch.quanta, invocation.phaseKind);
      const removedFullHead = dispatch.quanta.slice(preHeadCount);
      const preHead = dispatch.quanta.slice(0, preHeadCount);
      for (let index = 0; index < preHead.length; index += 1) {
        this.signal.throwIfAborted();
        const quantum = preHead[index]!;
        const encoder = this.observer.device.createCommandEncoder({
          label: `${quantum.id}-command`,
        });
        if (clearCache && index === 0) {
          clearBinding(encoder, phase.bindings.controls.cacheValidity);
          clearBinding(encoder, phase.bindings.controls.writeStatus);
        }
        const pass = encoder.beginComputePass({ label: quantum.id });
        quantum.encode(pass);
        pass.end();
        await submitAndDrain(this.observer.device, encoder.finish());
        const idle = this.progress.yieldCandidateIdle();
        try {
          this.progress.acceptCandidate("model", quantum);
        } catch (error) {
          await idle;
          throw error;
        }
        await idle;
        this.signal.throwIfAborted();
      }
      if (invocation.cancellationBoundary === "after-pre-head") {
        invocation.boundaryAbortController!.abort(new DOMException("OPT-0012 cancellation after fully drained pre-head boundary", "AbortError"));
        invocation.boundaryAbortController!.signal.throwIfAborted();
      }

      const headLogicalId = removedFullHead[0]!.logicalId!;
      const headEncoder = this.observer.device.createCommandEncoder({
        label: `${dispatch.label}-opt-0012-tied-lm-head-command`,
      });
      const headPass = headEncoder.beginComputePass({
        label: `${headLogicalId}-opt-0012-${plan.state}`,
      });
      for (const candidateDispatch of dispatches) candidateDispatch.encode(headPass);
      headPass.end();
      this.signal.throwIfAborted();
      await submitAndDrain(this.observer.device, headEncoder.finish());
      const headIdle = this.progress.yieldCandidateIdle();
      try {
        this.progress.acceptCandidate("model", {
          id: `${dispatch.label}-opt-0012-tied-lm-head`,
          logicalId: headLogicalId,
          kind: "tied-lm-head",
          layer: null,
          primitiveCount: dispatches.length,
        });
      } catch (error) {
        await headIdle;
        throw error;
      }
      await headIdle;
      this.signal.throwIfAborted();
      if (invocation.cancellationBoundary === "after-head") {
        invocation.boundaryAbortController!.abort(new DOMException("OPT-0012 cancellation after fully drained candidate head boundary", "AbortError"));
        invocation.boundaryAbortController!.signal.throwIfAborted();
      }

      const readbackEncoder = this.observer.device.createCommandEncoder({
        label: "opt-0012-compact-head-readback-command",
      });
      for (const copy of plan.readback.copies) {
        if (copy.kind === "logits") {
          const outputIndex = plan.headSlices.findIndex((slice) => slice.shardIndex === copy.shardIndex);
          const output = outputs[outputIndex];
          if (output === undefined) {
            throw new Error("OPT-0012 compact copy references no head output");
          }
          readbackEncoder.copyBufferToBuffer(
            output.buffer,
            output.payloadOffset + copy.sourceByteOffset,
            readback,
            copy.destinationByteOffset,
            copy.byteLength,
          );
        } else {
          const status = phase.bindings.controls.writeStatus;
          readbackEncoder.copyBufferToBuffer(
            status.buffer,
            (status.offset ?? 0) + copy.sourceByteOffset,
            readback,
            copy.destinationByteOffset,
            copy.byteLength,
          );
        }
      }
      this.signal.throwIfAborted();
      await submitAndDrain(this.observer.device, readbackEncoder.finish());
      const readbackIdle = this.progress.yieldCandidateIdle();
      try {
        this.progress.acceptCandidate("readback", null);
      } catch (error) {
        await readbackIdle;
        throw error;
      }
      let mappedBytes: ArrayBuffer;
      let decoded: AceOpt0012DecodedCompactLogits;
      let hostDecodeMilliseconds: number;
      try {
        await readback.mapAsync(GPUMapMode.READ, 0, plan.readback.allocationBytes);
        mappedBytes = readback.getMappedRange(0, plan.readback.allocationBytes).slice(0);
        readback.unmap();
        requireSentinel(new Uint8Array(mappedBytes, plan.readback.usedBytes, plan.readback.alignmentPaddingBytes), 0x3c, "compact readback alignment padding");
        const hostDecodeStarted = performance.now();
        decoded = decodeAceOpt0012CompactFp16Readback(mappedBytes, plan);
        hostDecodeMilliseconds = performance.now() - hostDecodeStarted;
      } finally {
        await readbackIdle;
      }
      this.signal.throwIfAborted();
      this.receipt = Object.freeze({
        decoded,
        mappedBytes,
        outputs: Object.freeze(outputs),
        readback,
        hostDecodeMilliseconds,
        preparation: prepared.receipt,
      });
      return Object.freeze([decoded.conditionalLogits, decoded.unconditionalLogits]);
    } catch (error) {
      this.destroyPreparedCandidateResources(prepared);
      throw error;
    }
  }

  private createOutput(label: string, payloadBytes: number, sentinel: number): CandidateOutputHandle {
    const totalBytes = CANDIDATE_OUTPUT_GUARD_BYTES + payloadBytes + CANDIDATE_OUTPUT_GUARD_BYTES;
    const buffer = createSentinelBuffer(this.observer.device, label, totalBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, sentinel);
    return Object.freeze({
      buffer,
      payloadOffset: CANDIDATE_OUTPUT_GUARD_BYTES,
      payloadBytes,
      totalBytes,
      sentinel,
      label,
    });
  }

  private validateSourceBindings(phase: AcePlannerPreparedPhaseGpuResources, plan: AceOpt0012CompactSemanticHeadPlan): void {
    if (phase.bindings.weights.embedding.length !== 5) {
      throw new Error("OPT-0012 phase lost a tied embedding binding");
    }
    for (const slice of plan.headSlices) {
      const owner = phase.bindings.weights.embedding[slice.shardIndex];
      const expected =
        plan.state === "regular-code" ? this.regularPlan.headSlices.find((entry) => entry.shardIndex === slice.shardIndex) : this.eosPlan.headSlices[0];
      if (
        owner === undefined ||
        expected !== slice ||
        owner.firstRow !== ACE_PLANNER_EMBEDDING_ROW_PARTS[slice.shardIndex]!.firstRow ||
        owner.rowCount !== ACE_PLANNER_EMBEDDING_ROW_PARTS[slice.shardIndex]!.rowCount ||
        owner.weight.offset !== slice.sourceOwnerByteOffset ||
        owner.weight.size !== slice.sourceOwnerByteLength ||
        slice.sourceBindingByteEnd > owner.weight.buffer.size
      ) {
        throw new Error(`OPT-0012 source sub-binding ${slice.shardIndex} changed`);
      }
    }
  }
}

async function runCancellationProofs(prepared: PreparedSession): Promise<Readonly<Record<string, unknown>>> {
  const fixture = prepared.fixtures[0]!;
  const preHead = await runCancellationAtBoundary(prepared, fixture, "after-pre-head");
  const postHead = await runCancellationAtBoundary(prepared, fixture, "after-head");
  return Object.freeze({
    afterFullyDrainedPreHead: preHead,
    afterFullyDrainedHeadBeforeReadback: postHead,
    eachBoundaryUsedRealAbortSignalTransition: true,
    eachRejectedInvocationReleasedItsCachePhase: true,
    noLaterCopyMapReconstructionSamplingCallbackOrFinalization: true,
  });
}

async function runCancellationAtBoundary(
  prepared: PreparedSession,
  fixture: PlannerCaseFixture,
  boundary: Exclude<CancellationBoundary, null>,
): Promise<Readonly<Record<string, unknown>>> {
  await prepared.executor.prefill(fixture.prefill);
  const statusPoison = await prepared.candidateRunner.poisonActiveWriteStatusOutsidePrimaryWall();
  const candidatePreparation = await prepared.candidateRunner.prepareCandidateResourcesOutsidePrimaryWall(prepared.regularPlan, SENTINEL_A);
  const boundaryAbortController = new AbortController();
  prepared.observer.beginTrace(`${fixture.spec.id}-cancellation-${boundary}`, false);
  const cursor = new AceOpt0012CompactSamplingCursor(ACCEPTED_SEED, fixture.spec.drawIndex);
  const cursorBefore = cursor.consumed;
  let callbackCount = 0;
  let finalizationCount = 0;
  let rejection: unknown;
  try {
    const returned = await prepared.candidateRunner.invoke(() => prepared.executor.decode(fixture.decode), {
      arm: "C",
      phaseKind: "decode",
      plan: prepared.regularPlan,
      sentinel: SENTINEL_A,
      guardProof: false,
      cancellationBoundary: boundary,
      boundaryAbortController,
      allowCorrectnessOnlyInsideTracePreparation: false,
    });
    callbackCount += 1;
    finalizationCount += 1;
    void returned;
  } catch (error) {
    rejection = error;
  }
  if (!(rejection instanceof DOMException) || rejection.name !== "AbortError") {
    prepared.observer.abandonTrace();
    throw new Error(`OPT-0012 ${boundary} cancellation did not reject`);
  }
  if (
    !boundaryAbortController.signal.aborted ||
    !(boundaryAbortController.signal.reason instanceof DOMException) ||
    boundaryAbortController.signal.reason.name !== "AbortError"
  ) {
    throw new Error(`OPT-0012 ${boundary} cancellation did not transition signal`);
  }
  const expectedCompletedCommandBuffers = boundary === "after-pre-head" ? 31 : 32;
  const explicitProgress = prepared.candidateRunner.takeProgressReceipt(expectedCompletedCommandBuffers, false);
  const trace = prepared.observer.endCancellationTrace(
    boundary === "after-pre-head" ? { commandBufferCount: 31, physicalDispatchCount: 623 } : { commandBufferCount: 32, physicalDispatchCount: 625 },
  );
  const cachePublication = await requireCancelledDecodePhaseReleased(prepared.executor, fixture.decode);
  if (cursor.consumed !== cursorBefore || callbackCount !== 0 || finalizationCount !== 0) {
    throw new Error(`OPT-0012 ${boundary} cancellation published later CPU state`);
  }
  return Object.freeze({
    boundary,
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
    abortSignal: Object.freeze({
      aborted: boundaryAbortController.signal.aborted,
      reasonName: boundaryAbortController.signal.reason.name,
      reasonMessage: boundaryAbortController.signal.reason.message,
    }),
    statusPoison,
    candidatePreparation,
    explicitProgress,
    trace,
    cachePublication,
    samplingCursor: Object.freeze({
      before: cursorBefore.toString(),
      after: cursor.consumed.toString(),
      sampleCallCount: 0,
    }),
    callbackCount,
    finalizationCount,
  });
}

async function requireCancelledDecodePhaseReleased(executor: AcePlannerGpuExecutor, decode: AcePlannerDecodeBatch): Promise<Readonly<Record<string, unknown>>> {
  let rejection: unknown;
  try {
    await executor.decode(decode);
  } catch (error) {
    rejection = error;
  }
  if (!(rejection instanceof Error) || !rejection.message.includes("requires a successful fresh prefill")) {
    throw new Error("OPT-0012 cancelled decode cache phase remained published");
  }
  return Object.freeze({
    subsequentDecodeRejected: true,
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
    rejectedInvocationCacheNotPublished: true,
  });
}

async function runTrajectoryAndCleanup(prepared: PreparedSession): Promise<Readonly<Record<string, unknown>>> {
  const startedAtEpochMilliseconds = Date.now();
  try {
    const trajectories: TrajectoryArmExecution[] = [];
    let rawFp16Authority: TrajectoryArmExecution | null = null;
    for (const arm of ["A", "B", "C"] as const) {
      for (let repeatIndex = 0; repeatIndex < 2; repeatIndex += 1) {
        postProgress(`raw-FP16 trajectory arm ${arm} repeat ${repeatIndex + 1}/2: ` + "150 regular codes plus forced EOS");
        const trajectory = await runTrajectoryArm(prepared, arm, repeatIndex, rawFp16Authority);
        trajectories.push(trajectory);
        rawFp16Authority ??= trajectory;
      }
    }
    const authority = rawFp16Authority!;
    for (const candidate of trajectories.slice(1)) {
      if (
        JSON.stringify(candidate.tokens) !== JSON.stringify(authority.tokens) ||
        candidate.semanticCodeSha256 !== authority.semanticCodeSha256 ||
        candidate.serializedAudioCodeTextSha256 !== authority.serializedAudioCodeTextSha256 ||
        JSON.stringify(candidate.terminal) !== JSON.stringify(authority.terminal)
      ) {
        throw new Error("OPT-0012 complete raw-FP16 semantic trajectory diverged by arm/repeat");
      }
    }
    postProgress("trajectory: probing cancellation during a resident semantic cache");
    const trajectoryCancellation = await runTrajectoryCancellation(prepared, authority);
    const cleanup = await destroyPreparedSession(prepared);
    const completedAtEpochMilliseconds = Date.now();
    const heartbeat = workerHeartbeat?.stop();
    return Object.freeze({
      experiment: "OPT-0012",
      mode: "raw-fp16-30-second-trajectory-correctness",
      identity: prepared.identity,
      coreSourceSha256: OPT_0012_CORE_SOURCE_SHA256,
      candidateShaderSha256: OPT_0012_CANDIDATE_SHADER_SHA256,
      package: prepared.preparedPackage.summary,
      sourceAuthentication: prepared.sourceAuthentication,
      capabilities: prepared.capabilityAuthentication,
      historicalBf16TeacherFixture: Object.freeze({
        cotReceiptSha256: OPT_0012_ACCEPTED_COT_RECEIPT_SHA256,
        semanticCodeSha256: OPT_0012_ACCEPTED_SEMANTIC_CODE_SHA256,
        outputAuthority: false,
      }),
      m1Execution: "not-executed-by-this-benchmark-mode",
      initialCursorAuthority: "frozen-historical-packed-bf16-M1-fixture-not-an-observed-path",
      firstSemanticDrawIndex: FIRST_SEMANTIC_DRAW_INDEX.toString(),
      rawFp16ArmAAuthority: Object.freeze({
        semanticCodeSha256: authority.semanticCodeSha256,
        serializedAudioCodeTextSha256: authority.serializedAudioCodeTextSha256,
        terminal: authority.terminal,
      }),
      trajectories: Object.freeze(trajectories),
      executionsPerArm: 2,
      allSixPerDrawReceiptsExact: true,
      everyArmSelfRepeatExact: true,
      trajectoryCancellation,
      fixedOrderPerformanceInterpretationForbidden: true,
      timingComparisonThermallyBalanced: false,
      cleanup,
      runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
      workerHeartbeat: heartbeat,
      startedAtEpochMilliseconds,
      completedAtEpochMilliseconds,
    });
  } catch (error) {
    await destroyPreparedSession(prepared);
    throw error;
  }
}

async function runTrajectoryArm(
  prepared: PreparedSession,
  arm: Opt0012Arm,
  repeatIndex: number,
  authority: TrajectoryArmExecution | null,
): Promise<TrajectoryArmExecution> {
  const prompts = createAcePlannerCodePrompts(ACCEPTED_RESOLVED_CAPTION, ACCEPTED_LYRICS, ACCEPTED_COT_TEXT);
  const promptRows = Object.freeze([
    Object.freeze(prepared.tokenizer.encode(prompts.conditional)),
    Object.freeze(prepared.tokenizer.encode(prompts.unconditional)),
  ]);
  const promptWidth = Math.max(...promptRows.map((row) => row.length));
  const capacity = 768;
  const prefill = createPaddedSemanticPrefill(promptRows, [], promptWidth, capacity);
  const fullCursor = arm === "C" ? null : new AcePlannerSamplingCursor(ACCEPTED_SEED, FIRST_SEMANTIC_DRAW_INDEX);
  const compactCursor = arm === "C" ? new AceOpt0012CompactSamplingCursor(ACCEPTED_SEED, FIRST_SEMANTIC_DRAW_INDEX) : null;
  const seenTokenIds: number[] = [...promptRows[0]!];
  const receipts: SampleReceipt[] = [];
  const topology: Array<Readonly<Record<string, unknown>>> = [];
  const wallStarted = performance.now();

  let rows = await executeTrajectoryInvocation(prepared, arm, prepared.regularPlan, () => prepared.executor.prefill(prefill), "prefill-first-code", topology);
  for (let codeIndex = 0; codeIndex < 150; codeIndex += 1) {
    const sample = sampleTrajectoryRows(arm, prepared.regularPlan, rows, seenTokenIds, fullCursor, compactCursor);
    if (sample.tokenId < ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID || sample.tokenId >= ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + ACE_PLANNER_SEMANTIC_CODE_COUNT) {
      throw new Error(`OPT-0012 trajectory ${arm} emitted a non-code token`);
    }
    const expected = authority?.tokens[codeIndex];
    if (expected !== undefined && !sameSample(sample, expected)) {
      throw new Error(
        `OPT-0012 raw-FP16 trajectory first divergence at arm ${arm} ` +
          `repeat ${repeatIndex} draw ${codeIndex}: ` +
          `${JSON.stringify(sample)} != ${JSON.stringify(expected)}`,
      );
    }
    receipts.push(sample);
    seenTokenIds.push(sample.tokenId);
    if (codeIndex < 149) {
      const cachedTokens = promptWidth + codeIndex;
      const decode = createSemanticDecodeBatch(sample.tokenId, cachedTokens, capacity);
      rows = await executeTrajectoryInvocation(
        prepared,
        arm,
        prepared.regularPlan,
        () => prepared.executor.decode(decode),
        `regular-${codeIndex + 1}`,
        topology,
      );
    }
  }
  const terminalDecode = createSemanticDecodeBatch(receipts.at(-1)!.tokenId, promptWidth + 149, capacity);
  const terminalRows = await executeTrajectoryInvocation(
    prepared,
    arm,
    prepared.eosPlan,
    () => prepared.executor.decode(terminalDecode),
    "forced-eos",
    topology,
  );
  const terminal = sampleTrajectoryRows(arm, prepared.eosPlan, terminalRows, seenTokenIds, fullCursor, compactCursor);
  if (terminal.tokenId !== ACE_QWEN_IM_END_TOKEN_ID || terminal.positiveCandidateCount !== 1 || terminal.drawIndex !== "259" || terminal.drawEnd !== "260") {
    throw new Error(`OPT-0012 trajectory ${arm} terminal receipt changed`);
  }
  if (authority !== null && !sameSample(terminal, authority.terminal)) {
    throw new Error(
      `OPT-0012 raw-FP16 trajectory terminal divergence at arm ${arm} ` +
        `repeat ${repeatIndex}: ${JSON.stringify(terminal)} != ` +
        JSON.stringify(authority.terminal),
    );
  }
  const codeIds = receipts.map((receipt) => receipt.tokenId - ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID);
  const semanticCodeSha256 = sha256U32Le(codeIds);
  const serializedAudioCodeText = codeIds.map((code) => `<|audio_code_${code}|>`).join("");
  return Object.freeze({
    arm,
    repeatIndex,
    codeCount: receipts.length,
    tokens: Object.freeze(receipts),
    terminal,
    terminalTokenId: terminal.tokenId,
    finalDrawEnd: terminal.drawEnd,
    semanticCodeSha256,
    serializedAudioCodeTextSha256: aceSha256Hex(new TextEncoder().encode(serializedAudioCodeText)),
    topology: summarizeTrajectoryTopology(arm, topology),
    totalWallMilliseconds: performance.now() - wallStarted,
    performanceComparisonAuthority: "none-fixed-order-correctness-only",
    semanticHashAuthority: "observed-raw-fp16-arm-A-and-cross-arm-repeat-identity",
  });
}

async function executeTrajectoryInvocation(
  prepared: PreparedSession,
  arm: Opt0012Arm,
  plan: AceOpt0012CompactSemanticHeadPlan,
  operation: () => Promise<readonly ArrayLike<number>[]>,
  label: string,
  topology: Array<Readonly<Record<string, unknown>>>,
): Promise<readonly Float32Array[]> {
  const statusPoison = label === "prefill-first-code" ? null : await prepared.candidateRunner.poisonActiveWriteStatusOutsidePrimaryWall();
  const candidatePreparation =
    arm === "A" || label === "prefill-first-code" ? null : await prepared.candidateRunner.prepareCandidateResourcesOutsidePrimaryWall(plan, SENTINEL_A);
  prepared.observer.beginTrace(`trajectory-${arm}-${label}`, false);
  const startedAtEpochMilliseconds = Date.now();
  const wallStarted = performance.now();
  try {
    const decodeStarted = performance.now();
    const returned = await prepared.candidateRunner.invoke(operation, {
      arm,
      phaseKind: label === "prefill-first-code" ? "prefill" : "decode",
      plan,
      sentinel: SENTINEL_A,
      guardProof: false,
      cancellationBoundary: null,
      boundaryAbortController: null,
      allowCorrectnessOnlyInsideTracePreparation: arm !== "A" && label === "prefill-first-code",
    });
    const decodeEnded = performance.now();
    const phaseKind = label === "prefill-first-code" ? "prefill" : "decode";
    const explicitProgress = prepared.candidateRunner.takeProgressReceipt(opt0012ExpectedCommandBufferTopology(arm, phaseKind).totalCommandBufferCount, true);
    const receipt = arm === "A" ? null : prepared.candidateRunner.takeReceipt();
    if (receipt !== null && label !== "prefill-first-code" && receipt.preparation !== candidatePreparation) {
      throw new Error("OPT-0012 trajectory preparation receipt identity changed");
    }
    const rows = arm === "A" ? requireFloat32Rows(returned, plan.vocabularySize) : [receipt!.decoded.conditionalLogits, receipt!.decoded.unconditionalLogits];
    const expectedEncodedFreshStatusClears = label === "prefill-first-code" ? prepared.candidateRunner.describeActiveControlClears() : Object.freeze([]);
    const ended = performance.now();
    const observedTrace = prepared.observer.endTrace({
      arm,
      phaseKind,
      state: plan.state,
      expectedPhysicalDispatchCount:
        arm === "A"
          ? OPT_0012_FULL_PHYSICAL_DISPATCH_COUNT
          : plan.state === "regular-code"
            ? OPT_0012_REGULAR_PHYSICAL_DISPATCH_COUNT
            : OPT_0012_EOS_PHYSICAL_DISPATCH_COUNT,
      expectedCopyCount: arm === "A" ? OPT_0012_FULL_COPY_COUNT : plan.readback.copies.length,
      expectedCommandBufferCount: opt0012ExpectedCommandBufferTopology(arm, phaseKind).totalCommandBufferCount,
      roundIndex: -1,
      order: "trajectory",
      orderPosition: -1,
      startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Date.now(),
      wallMilliseconds: ended - wallStarted,
      decodeMilliseconds: ended - wallStarted,
      reconstructionMilliseconds: 0,
      samplingMilliseconds: 0,
      callbackMilliseconds: 0,
      samplingStageIntervals: NO_SAMPLING_STAGE_INTERVALS,
      decodeStartedAt: decodeStarted,
      decodeEndedAt: decodeEnded,
      candidateHostDecodeMilliseconds: receipt?.hostDecodeMilliseconds ?? null,
      readbackIdleInterval: requireCompleteReadbackIdle(explicitProgress),
      expectedEncodedFreshStatusClears,
    });
    const fp16NaNCensus =
      arm === "A"
        ? requireZeroOpt0012DecodedFullFp16NaNs(rows, `trajectory ${arm} ${label}`, plan.state)
        : requireZeroOpt0012RawCompactFp16NaNs(plan, receipt!.mappedBytes, `trajectory ${arm} ${label}`);
    topology.push(
      Object.freeze({
        ...observedTrace,
        explicitProgress,
        statusFreshness: Object.freeze({
          encodedClearOnFreshPrefill: label === "prefill-first-code",
          preInvocationZeroed: label !== "prefill-first-code",
          poisonAndFence: statusPoison,
          requiredSuccessWords: Object.freeze([1, 1]),
        }),
        candidatePreparation: receipt?.preparation ?? null,
        fp16NaNCensus,
        correctnessOnlyFirstPrefillPreparedInsideTrace: arm !== "A" && label === "prefill-first-code",
      }),
    );
    prepared.candidateRunner.releasePendingGuardsWithoutReadback();
    return Object.freeze(rows);
  } catch (error) {
    prepared.observer.abandonTrace();
    prepared.candidateRunner.releasePendingGuardsWithoutReadback();
    throw error;
  }
}

function sampleTrajectoryRows(
  arm: Opt0012Arm,
  plan: AceOpt0012CompactSemanticHeadPlan,
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  fullCursor: AcePlannerSamplingCursor | null,
  compactCursor: AceOpt0012CompactSamplingCursor | null,
): SampleReceipt {
  if (arm === "C") {
    if (compactCursor === null) throw new Error("OPT-0012 compact cursor is absent");
    const sampled = compactCursor.sample({
      plan,
      conditionalLogits: rows[0]!,
      unconditionalLogits: rows[1]!,
      seenTokenIds,
    });
    return Object.freeze({
      tokenId: sampled.tokenId,
      word: sampled.word,
      positiveCandidateCount: sampled.positiveCandidateCount,
      drawIndex: sampled.drawIndex.toString(),
      drawEnd: sampled.drawEnd.toString(),
    });
  }
  if (fullCursor === null) throw new Error("OPT-0012 full cursor is absent");
  let fullRows = rows;
  if (arm === "B") {
    fullRows = reconstructAceOpt0012FullPlannerLogits(
      {
        conditionalLogits: rows[0]!,
        unconditionalLogits: rows[1]!,
        writeStatus: new Uint32Array([1, 1]),
      },
      plan,
    );
  }
  const sampled = fullCursor.sample({
    conditionalLogits: fullRows[0]!,
    unconditionalLogits: fullRows[1]!,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens: plan.state === "regular-code" ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
  });
  return Object.freeze({
    tokenId: sampled.tokenId,
    word: sampled.word,
    positiveCandidateCount: sampled.positiveCandidateCount,
    drawIndex: sampled.drawIndex.toString(),
    drawEnd: fullCursor.consumed.toString(),
  });
}

function summarizeTrajectoryTopology(arm: Opt0012Arm, traces: readonly Readonly<Record<string, unknown>>[]): Readonly<Record<string, unknown>> {
  if (traces.length !== 151 || traces.slice(0, 150).some((trace) => trace.state !== "regular-code") || traces[150]?.state !== "forced-eos") {
    throw new Error("OPT-0012 trajectory topology omitted an invocation");
  }
  const commandBufferCount = sumSafe(
    traces.map((trace) => trace.commandBufferCount as number),
    "trajectory command buffers",
  );
  const queueDrainCount = sumSafe(
    traces.map((trace) => trace.queueDrainCount as number),
    "trajectory queue drains",
  );
  const cooperativeIdleCount = sumSafe(
    traces.map((trace) => trace.realCooperativeIdleCount as number),
    "trajectory cooperative idles",
  );
  const physicalDispatchCount = sumSafe(
    traces.map((trace) => trace.physicalDispatchCount as number),
    "trajectory dispatches",
  );
  const copyCount = sumSafe(
    traces.map((trace) => trace.copyCount as number),
    "trajectory copies",
  );
  const encodedFreshPrefillStatusClearCount = traces.filter(
    (trace) => (trace.statusFreshness as Readonly<Record<string, unknown>> | undefined)?.encodedClearOnFreshPrefill === true,
  ).length;
  const explicitPreInvocationStatusResetCount = traces.filter(
    (trace) => (trace.statusFreshness as Readonly<Record<string, unknown>> | undefined)?.preInvocationZeroed === true,
  ).length;
  const fp16NaNCensuses = traces.map((trace) => trace.fp16NaNCensus as Readonly<Record<string, unknown>>);
  const prefillTopology = opt0012ExpectedCommandBufferTopology(arm, "prefill");
  const decodeTopology = opt0012ExpectedCommandBufferTopology(arm, "decode");
  const expected =
    arm === "A"
      ? { commandBufferCount: 5_246, physicalDispatchCount: 94_828, copyCount: 906 }
      : { commandBufferCount: 5_095, physicalDispatchCount: 94_374, copyCount: 452 };
  if (
    traces[0]?.commandBufferCount !== prefillTopology.totalCommandBufferCount ||
    traces.slice(1).some((trace) => trace.commandBufferCount !== decodeTopology.totalCommandBufferCount) ||
    commandBufferCount !== expected.commandBufferCount ||
    queueDrainCount !== expected.commandBufferCount ||
    cooperativeIdleCount !== expected.commandBufferCount ||
    physicalDispatchCount !== expected.physicalDispatchCount ||
    copyCount !== expected.copyCount ||
    encodedFreshPrefillStatusClearCount !== 1 ||
    explicitPreInvocationStatusResetCount !== 150 ||
    fp16NaNCensuses.some((census) => census.zeroBinary16NaNs !== true)
  ) {
    throw new Error(`OPT-0012 trajectory ${arm} aggregate topology changed`);
  }
  return Object.freeze({
    arm,
    invocationCount: traces.length,
    regularInvocationCount: 150,
    forcedEosInvocationCount: 1,
    commandBufferCount,
    queueDrainCount,
    cooperativeIdleCount,
    physicalDispatchCount,
    copyCount,
    encodedFreshPrefillStatusClearCount,
    explicitPreInvocationStatusResetCount,
    everyInvocationRequiredFreshStatusWords: true,
    everyActualPackageReadbackHasZeroBinary16NaNs: true,
    fp16NaNCensuses: Object.freeze(fp16NaNCensuses),
    maximumSingleDrainMilliseconds: Math.max(...traces.map((trace) => trace.maximumSingleDrainMilliseconds as number)),
    first: traces[0],
    last: traces.at(-1),
  });
}

async function runTrajectoryCancellation(prepared: PreparedSession, authority: TrajectoryArmExecution): Promise<Readonly<Record<string, unknown>>> {
  const prompts = createAcePlannerCodePrompts(ACCEPTED_RESOLVED_CAPTION, ACCEPTED_LYRICS, ACCEPTED_COT_TEXT);
  const promptRows = Object.freeze([
    Object.freeze(prepared.tokenizer.encode(prompts.conditional)),
    Object.freeze(prepared.tokenizer.encode(prompts.unconditional)),
  ]);
  const promptWidth = Math.max(...promptRows.map((row) => row.length));
  const capacity = 768;
  const prefixCodeCount = 8;
  const acceptedPrefix = authority.tokens.slice(0, prefixCodeCount + 1);
  if (
    acceptedPrefix.length !== prefixCodeCount + 1 ||
    acceptedPrefix.some((receipt, index) => receipt.drawIndex !== (FIRST_SEMANTIC_DRAW_INDEX + BigInt(index)).toString())
  ) {
    throw new Error("OPT-0012 resident cancellation raw authority prefix changed");
  }
  const prefillTokens = acceptedPrefix.slice(0, prefixCodeCount).map((receipt) => receipt.tokenId);
  const prefill = createPaddedSemanticPrefill(promptRows, prefillTokens, promptWidth + prefixCodeCount, capacity);
  const prefillTopology: Array<Readonly<Record<string, unknown>>> = [];
  await executeTrajectoryInvocation(prepared, "C", prepared.regularPlan, () => prepared.executor.prefill(prefill), "prefill-first-code", prefillTopology);
  const cacheBefore = prepared.candidateRunner.activeCachePublicationSnapshot();
  if (cacheBefore.published !== true || cacheBefore.cachedTokens !== promptWidth + prefixCodeCount) {
    throw new Error("OPT-0012 resident cancellation cache prefix was not published");
  }
  const nextAccepted = acceptedPrefix[prefixCodeCount]!;
  const cursor = new AceOpt0012CompactSamplingCursor(ACCEPTED_SEED, BigInt(nextAccepted.drawEnd));
  const cursorBefore = cursor.consumed;
  let publishedCallbackCount = acceptedPrefix.length;
  const callbackCountBefore = publishedCallbackCount;
  let finalizationCount = 0;
  const decode = createSemanticDecodeBatch(nextAccepted.tokenId, promptWidth + prefixCodeCount, capacity);
  const statusPoison = await prepared.candidateRunner.poisonActiveWriteStatusOutsidePrimaryWall();
  const candidatePreparation = await prepared.candidateRunner.prepareCandidateResourcesOutsidePrimaryWall(prepared.regularPlan, SENTINEL_A);
  const boundaryAbortController = new AbortController();
  prepared.observer.beginTrace("raw-resident-trajectory-cancellation-after-head", false);
  let rejection: unknown;
  try {
    const returned = await prepared.candidateRunner.invoke(() => prepared.executor.decode(decode), {
      arm: "C",
      phaseKind: "decode",
      plan: prepared.regularPlan,
      sentinel: SENTINEL_A,
      guardProof: false,
      cancellationBoundary: "after-head",
      boundaryAbortController,
      allowCorrectnessOnlyInsideTracePreparation: false,
    });
    publishedCallbackCount += 1;
    finalizationCount += 1;
    void returned;
  } catch (error) {
    rejection = error;
  }
  if (!(rejection instanceof DOMException) || rejection.name !== "AbortError" || !boundaryAbortController.signal.aborted) {
    prepared.observer.abandonTrace();
    throw new Error("OPT-0012 resident trajectory cancellation did not abort");
  }
  const explicitProgress = prepared.candidateRunner.takeProgressReceipt(32, false);
  const trace = prepared.observer.endCancellationTrace({
    commandBufferCount: 32,
    physicalDispatchCount: 625,
  });
  const cacheAfter = prepared.candidateRunner.activeCachePublicationSnapshot();
  const cachePublication = await requireCancelledDecodePhaseReleased(prepared.executor, decode);
  const activityBeforeTurns = prepared.observer.activitySnapshot();
  await yieldToBrowser();
  await yieldToBrowser();
  const activityAfterTurns = prepared.observer.activitySnapshot();
  if (
    JSON.stringify(activityBeforeTurns) !== JSON.stringify(activityAfterTurns) ||
    cacheAfter.published !== false ||
    cursor.consumed !== cursorBefore ||
    publishedCallbackCount !== callbackCountBefore ||
    finalizationCount !== 0
  ) {
    throw new Error("OPT-0012 resident cancellation published later work or state");
  }
  return Object.freeze({
    authority: "observed-raw-fp16-arm-A-prefix",
    prefixCodeCount,
    acceptedTokenCountBeforeCancelledAppend: acceptedPrefix.length,
    acceptedPrefix: Object.freeze(acceptedPrefix),
    prefillTopology: Object.freeze(prefillTopology),
    cacheBefore,
    cancelledAppendToken: nextAccepted,
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
    abortSignal: Object.freeze({
      aborted: boundaryAbortController.signal.aborted,
      reasonName: (boundaryAbortController.signal.reason as DOMException).name,
    }),
    statusPoison,
    candidatePreparation,
    explicitProgress,
    trace,
    cacheAfter,
    cachePublication,
    samplingCursor: Object.freeze({
      before: cursorBefore.toString(),
      after: cursor.consumed.toString(),
      sampleCallCountDuringCancelledInvocation: 0,
    }),
    publishedCallbackCountBefore: callbackCountBefore,
    publishedCallbackCountAfter: publishedCallbackCount,
    finalizationCount,
    activityBeforeTurns,
    activityAfterTurns,
    postCancellationMacrotaskTurnCount: 2,
    noLaterSubmitDrainMapAllocationCursorCallbackOrFinalization: true,
  });
}

async function destroyPreparedSession(prepared: PreparedSession): Promise<Readonly<Record<string, unknown>>> {
  if (prepared.cleanupReceipt !== null) return prepared.cleanupReceipt;
  if (prepared.cleanupPromise !== null) return await prepared.cleanupPromise;
  prepared.destroyed = true;
  const cleanupStartedAtEpochMilliseconds = Date.now();
  const cleanup = (async (): Promise<Readonly<Record<string, unknown>>> => {
    let cleanupError: unknown;
    let postDestroyPrefill: Readonly<Record<string, unknown>> | null = null;
    let postDestroyDecode: Readonly<Record<string, unknown>> | null = null;
    let beforeDeviceDestroy: Readonly<Record<string, unknown>> | null = null;
    const rememberError = (error: unknown): void => {
      cleanupError ??= error;
    };
    try {
      prepared.candidateRunner.destroy();
      await prepared.executor.destroy();
      await prepared.executor.destroy();
      postDestroyPrefill = await requireExecutorDestroyedRejection(() => prepared.executor.prefill(prepared.fixtures[0]!.prefill), "prefill");
      postDestroyDecode = await requireExecutorDestroyedRejection(() => prepared.executor.decode(prepared.fixtures[0]!.decode), "decode");
      beforeDeviceDestroy = prepared.observer.resourceSnapshot();
      if (beforeDeviceDestroy.liveTrackedBufferCount !== 0) {
        throw new Error("OPT-0012 cleanup left tracked GPU buffers live");
      }
      if (prepared.runtimeEvents.length !== 0) {
        throw new Error("OPT-0012 observed a pre-destroy runtime/device-loss event");
      }
      // AceWebGpuDeviceContext removes its uncaptured-error listener as part of
      // destroy().  Give already queued errors two task turns to arrive while
      // that listener is still installed, then authenticate the empty stream.
      await yieldToBrowser();
      await yieldToBrowser();
      if (prepared.runtimeEvents.length !== 0) {
        throw new Error("OPT-0012 observed a queued runtime event before context destroy");
      }
    } catch (error) {
      rememberError(error);
    }
    const deviceDestroyCalledAtEpochMilliseconds = Date.now();
    let intentionalLoss: Awaited<PreparedSession["context"]["lost"]> | null = null;
    let deviceLossObservedAtEpochMilliseconds: number | null = null;
    try {
      prepared.context.destroy();
      const observedLoss = await prepared.context.lost;
      intentionalLoss = observedLoss;
      deviceLossObservedAtEpochMilliseconds = Date.now();
      if (observedLoss.reason !== "destroyed") {
        throw new Error(`OPT-0012 cleanup observed device loss ${observedLoss.reason}, not destroyed`);
      }
    } catch (error) {
      rememberError(error);
    }
    // Keep runtime-event and heartbeat observation alive beyond the intentional
    // loss notification so a queued uncaptured error cannot arrive after pass.
    await yieldToBrowser();
    await yieldToBrowser();
    let resources: Readonly<Record<string, unknown>> | null = null;
    try {
      if (prepared.runtimeEvents.length !== 0) {
        throw new Error("OPT-0012 observed a runtime event through post-loss cleanup");
      }
      resources = prepared.observer.finalResourceSummary();
    } catch (error) {
      rememberError(error);
    }
    if (
      cleanupError !== undefined ||
      intentionalLoss === null ||
      deviceLossObservedAtEpochMilliseconds === null ||
      postDestroyPrefill === null ||
      postDestroyDecode === null ||
      beforeDeviceDestroy === null ||
      resources === null
    ) {
      throw cleanupError ?? new Error("OPT-0012 cleanup receipt is incomplete");
    }
    return Object.freeze({
      cleanupStartedAtEpochMilliseconds,
      deviceDestroyCalledAtEpochMilliseconds,
      deviceLossObservedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds: Date.now(),
      executorDestroyCallCount: 2,
      executorDestroyIdempotent: true,
      postDestroyPrefill,
      postDestroyDecode,
      beforeDeviceDestroy,
      intentionalDeviceLoss: Object.freeze({
        type: intentionalLoss.type,
        reason: intentionalLoss.reason,
        message: intentionalLoss.message,
        suppressedFromRuntimeEventsByContext: true,
      }),
      preDestroyRuntimeObservationMacrotaskTurnCount: 2,
      preDestroyRuntimeEventCount: 0,
      postLossMacrotaskTurnCount: 2,
      contextRuntimeEventCountAfterPostLossTurns: prepared.runtimeEvents.length,
      postLossRuntimeEventScope: "context-event-stream-after-pre-destroy-queue-observation-not-independent-raw-device-listener",
      resources,
    });
  })();
  prepared.cleanupPromise = cleanup;
  const receipt = await cleanup;
  prepared.cleanupReceipt = receipt;
  return receipt;
}

async function requireExecutorDestroyedRejection(
  operation: () => Promise<readonly ArrayLike<number>[]>,
  operationName: "prefill" | "decode",
): Promise<Readonly<Record<string, unknown>>> {
  let rejection: unknown;
  try {
    await operation();
  } catch (error) {
    rejection = error;
  }
  if (!(rejection instanceof DOMException) || rejection.name !== "InvalidStateError" || !rejection.message.includes("executor is destroyed")) {
    throw new Error(`OPT-0012 executor ${operationName} did not reject after destroy`);
  }
  return Object.freeze({
    operation: operationName,
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
  });
}

function publicPreparationSummary(prepared: PreparedSession): Readonly<Record<string, unknown>> {
  return Object.freeze({
    identity: prepared.identity,
    packageAcquisitionWallMilliseconds: prepared.packageAcquisitionWallMilliseconds,
    phaseUploadWallMilliseconds: prepared.phaseUploadWallMilliseconds,
    executorCompileAndCorrectnessWallMilliseconds: prepared.executorCompileAndCorrectnessWallMilliseconds,
    package: prepared.preparedPackage.summary,
    sourceAuthentication: prepared.sourceAuthentication,
    capabilities: prepared.capabilityAuthentication,
    candidateShaderSha256: OPT_0012_CANDIDATE_SHADER_SHA256,
    plans: Object.freeze({
      regular: publicPlan(prepared.regularPlan),
      forcedEos: publicPlan(prepared.eosPlan),
    }),
    cases: Object.freeze(
      prepared.fixtures.map((fixture) =>
        Object.freeze({
          spec: fixture.spec,
          decodeTokenId: fixture.decodeTokenId,
          promptLengths: Object.freeze(fixture.promptRows.map((row) => row.length)),
        }),
      ),
    ),
    forcedEosCase: Object.freeze({
      spec: prepared.eosFixture.spec,
      decodeTokenId: prepared.eosFixture.decodeTokenId,
      historicalBf16TrajectoryRole: "teacher-forced-cache-fixture-only",
      acceptedRegularCodeCountBeforeForcedEos: 150,
    }),
    fp16ConversionGate: prepared.fp16ConversionGate,
    correctness: prepared.correctness,
    adversarialSampling: prepared.adversarialSampling,
    m1FailClosed: true,
    symmetricUntimedWarmups: Object.freeze({
      A: prepared.correctness.length * 2,
      B: prepared.correctness.length * 2,
      C: prepared.correctness.length * 2,
      fp16ConversionLegacy: 2,
      fp16ConversionAllocationFree: 2,
      fp16ConversionOrders: OPT_0012_PREPACKAGE_CONVERSION_ORDERS,
      derivation: "four correctness states times first-run-plus-rerun; exhaustive " + "fixed forward-order and reverse-order conversion gate passes",
    }),
    warmupCompletedAtEpochMilliseconds: prepared.warmupCompletedAtEpochMilliseconds,
  });
}

function publicPlan(plan: AceOpt0012CompactSemanticHeadPlan): Readonly<Record<string, unknown>> {
  return Object.freeze({
    state: plan.state,
    firstCandidateTokenId: plan.firstCandidateTokenId,
    candidateCount: plan.candidateCount,
    intersections: plan.intersections,
    headSlices: plan.headSlices,
    logicalWeightTrafficBytes: plan.logicalWeightTrafficBytes,
    logicalMultiplyAdds: plan.logicalMultiplyAdds,
    scheduledMultiplyAdds: plan.scheduledMultiplyAdds,
    workgroupCount: plan.workgroupCount,
    readback: plan.readback,
  });
}

function createSentinelBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags, sentinel: number): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).fill(sentinel);
  buffer.unmap();
  return buffer;
}

function clearBinding(encoder: GPUCommandEncoder, binding: GPUBufferBinding): void {
  encoder.clearBuffer(binding.buffer, binding.offset ?? 0, binding.size ?? binding.buffer.size - (binding.offset ?? 0));
}

function zeroStatusBinding(queue: GPUQueue, binding: GPUBufferBinding): void {
  const byteOffset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - byteOffset;
  if (available !== OPT_0012_FULL_READBACK_STATUS_BYTES) {
    throw new Error("OPT-0012 write-status binding extent changed");
  }
  const zero = new Uint32Array(2);
  queue.writeBuffer(binding.buffer, byteOffset, zero.buffer, zero.byteOffset, zero.byteLength);
}

async function submitAndDrain(device: GPUDevice, command: GPUCommandBuffer): Promise<void> {
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
}

function requireSentinel(bytes: Uint8Array, sentinel: number, label: string): void {
  for (let index = 0; index < bytes.length; index += 1) {
    if (bytes[index] !== sentinel) {
      throw new Error(`OPT-0012 ${label} changed at byte ${index}`);
    }
  }
}

function requireFloat32Rows(rows: readonly ArrayLike<number>[], expectedLength: number): readonly Float32Array[] {
  if (rows.length !== 2) throw new Error("OPT-0012 expected two CFG rows");
  return Object.freeze(
    rows.map((row, index) => {
      if (!(row instanceof Float32Array) || row.length !== expectedLength) {
        throw new Error(`OPT-0012 row ${index} has the wrong logit extent`);
      }
      return row;
    }),
  );
}

function requiredTimestamp(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`OPT-0012 trace omitted ${label}`);
  }
  return value;
}

function requirePackageFile(files: ReadonlyMap<string, File>, name: string): File {
  const file = files.get(name);
  if (!(file instanceof File)) throw new Error(`OPT-0012 package omitted ${name}`);
  return file;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function sumSafe(values: readonly number[], label: string): number {
  let result = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`OPT-0012 ${label} contains an invalid value`);
    }
    result += value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`OPT-0012 ${label} exceeds safe integers`);
    }
  }
  return result;
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function align256(value: number): number {
  return Math.ceil(value / 256) * 256;
}

function scaleTemperature(logits: Float32Array, temperature: number): Float32Array {
  const result = logits.slice();
  const rounded = Math.fround(temperature);
  for (let index = 0; index < result.length; index += 1) {
    if (result[index] !== Number.NEGATIVE_INFINITY) {
      result[index] = Math.fround(result[index]! / rounded);
    }
  }
  return result;
}

function finiteTokenIds(values: ArrayLike<number>): number[] {
  const result: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (Number.isFinite(values[index])) result.push(index);
  }
  return result;
}

function floatWord(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

function requireU32ArraysEqual(left: ArrayLike<number>, right: ArrayLike<number>, label: string): void {
  if (left.length !== right.length) {
    throw new Error(`OPT-0012 ${label} lengths differ`);
  }
  for (let index = 0; index < left.length; index += 1) {
    if (floatWord(left[index]!) !== floatWord(right[index]!)) {
      throw new Error(`OPT-0012 ${label} differs at ${index}`);
    }
  }
}

function requireNumberArraysEqual(left: readonly number[], right: readonly number[], label: string): void {
  if (left.length !== right.length || left.some((value, index) => value !== right[index])) {
    throw new Error(`OPT-0012 ${label} differs`);
  }
}

function requireByteArraysEqual(left: Uint8Array, right: Uint8Array, label: string): void {
  if (left.length !== right.length) {
    throw new Error(`OPT-0012 ${label} byte lengths differ`);
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      throw new Error(`OPT-0012 ${label} differs at byte ${index}`);
    }
  }
}

function sameSample(left: SampleReceipt, right: SampleReceipt): boolean {
  return (
    left.tokenId === right.tokenId &&
    left.word === right.word &&
    left.positiveCandidateCount === right.positiveCandidateCount &&
    left.drawIndex === right.drawIndex &&
    left.drawEnd === right.drawEnd
  );
}

function requireSameSample(left: SampleReceipt, right: SampleReceipt, label: string): void {
  if (!sameSample(left, right)) throw new Error(`OPT-0012 ${label} sample differs`);
}

function sha256FloatWords(values: Float32Array): string {
  return aceSha256Hex(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
}

function sha256U16Le(values: Uint16Array): string {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint16(index * 2, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function sha256FullSemanticReadback(mapped: ArrayBuffer): string {
  if (mapped.byteLength !== OPT_0012_FULL_READBACK_ALLOCATION_BYTES) {
    throw new Error("OPT-0012 full readback allocation changed before hashing");
  }
  const semantic = new Uint8Array(OPT_0012_FULL_HEAD_LOGIT_BYTES + OPT_0012_FULL_READBACK_STATUS_BYTES);
  semantic.set(new Uint8Array(mapped, 0, OPT_0012_FULL_HEAD_LOGIT_BYTES));
  semantic.set(new Uint8Array(mapped, OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET, OPT_0012_FULL_READBACK_STATUS_BYTES), OPT_0012_FULL_HEAD_LOGIT_BYTES);
  return aceSha256Hex(semantic);
}

function sha256U32Words(values: Uint32Array): string {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * 4, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function sha256U32Le(values: readonly number[]): string {
  const words = new Uint32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
      throw new RangeError("OPT-0012 semantic code is not U32");
    }
    words[index] = value;
  }
  return sha256U32Words(words);
}

function summarizeNumbers(values: readonly number[]): Readonly<Record<string, number>> {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("OPT-0012 cannot summarize empty/non-finite samples");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!;
  return Object.freeze({
    minimum: sorted[0]!,
    median,
    maximum: sorted.at(-1)!,
    range: sorted.at(-1)! - sorted[0]!,
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
