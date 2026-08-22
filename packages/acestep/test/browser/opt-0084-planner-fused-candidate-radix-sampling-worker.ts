/// <reference lib="webworker" />

import plannerSamplingSource from
  "../../src/runtime/planner-sampling.ts?raw";
import { DEFAULT_ACE_PLANNER_CONFIGURATION } from "../../src/api.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_MODEL_SNAPSHOT_REVISION,
  ACE_PACKAGE_CONVERTER_REVISION,
  ACE_PLANNER_SNAPSHOT_REVISION,
  ACE_REFERENCE_SOURCE_REVISION,
  type AcePackageFileRecord,
  type AcePackageManifest,
} from "../../src/model/manifest.js";
import {
  ACE_REFERENCE_MANIFEST_SHA256,
  loadAcePackageManifest,
} from "../../src/model/package.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  createAcePlannerCodePrompts,
  createAcePlannerCotPrompt,
  type AcePlannerDecodeBatch,
  type AcePlannerPrefillBatch,
} from "../../src/runtime/planner.js";
import { AcePlannerMetadataConstraintController } from
  "../../src/runtime/planner-metadata-fsm.js";
import {
  ACE_BROWSER_SOFTMAX_V1,
  AceOpt0084PlannerSamplingWorkspace,
  AcePlannerSamplingCursor,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerFilteredLogits,
  sampleAcePlannerToken,
  type AceOpt0084PlannerSamplingWorkspaceStats,
  type AcePlannerAllowedTokens,
  type AcePlannerCursorSample,
  type AcePlannerTokenSample,
  type AcePlannerTokenSampleInput,
} from "../../src/runtime/planner-sampling.js";
import { aceRandomWord, canonicalizeSeed } from "../../src/runtime/seed.js";
import {
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_IM_END_TOKEN_ID,
  ACE_QWEN_PAD_TOKEN_ID,
  loadPinnedAceTokenizer,
  type AceQwenBpeTokenizer,
} from "../../src/tokenizer/index.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import { AcePlannerGpuExecutor } from "../../src/webgpu/planner-executor.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
} from "../../src/webgpu/qwen3.js";
import {
  OPT_0084_CANDIDATE_SEAM_COMMIT,
  OPT_0084_EXPERIMENT_ID,
  OPT_0084_PAIR_ORDERS,
  OPT_0084_RECEIPT_SCHEMA,
  OPT_0084_STATE_IDS,
  OPT_0084_THERMAL_COMMAND,
  OPT_0084_THERMAL_POLL_MILLISECONDS,
  OPT_0084_THERMAL_SOURCE,
  OPT_0084_TIMING_ROUND_COUNT,
  evaluateOpt0084Timing,
  validateOpt0084RunIdentity,
  type Opt0084RunIdentity,
  type Opt0084ThermalLaunch,
  type Opt0084TimingSample,
} from "./opt-0084-planner-fused-candidate-radix-sampling-contract.js";

const MANIFEST_PATH = "/model/files-reference/manifest.json" as const;
const EXPECTED_PLANNER_SAMPLING_SOURCE_SHA256 =
  "67055acfbb96e10682092e5d0ccfa9a5d822fd708a091fe46f7f47458226d0f3";
const PLANNER_TENSOR_COUNT = 314;
const PLANNER_WEIGHT_FILE_COUNT = 33;
const PLANNER_RESIDENT_BYTES = 1_325_768_704;
const TOKENIZER_FILE_NAMES = Object.freeze([
  "assets/planner/tokenizer.json",
  "assets/planner/tokenizer_config.json",
  "assets/planner/chat_template.jinja",
] as const);
const ACQUIRED_FILE_COUNT = PLANNER_WEIGHT_FILE_COUNT +
  TOKENIZER_FILE_NAMES.length;
const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const WARMUP_INVOCATIONS_PER_ARM_AND_STATE = 2;
const BACKTICK_TOKEN_ID = 63;
const ACCEPTED_SEED = canonicalizeSeed("000000000badc0de");
const ACCEPTED_PROMPT =
  "Bright neo-soul with elastic bass, clipped rhythm guitar, crisp pocket " +
  "drums, warm keys, and a confident mezzo-soprano vocal.";
const ACCEPTED_LYRICS =
  "[Verse]\nOpen the curtains, let the whole day in\n" +
  "Dust in the sunlight starts to spin\n\n[Chorus]\n" +
  "We found the rhythm under our feet\n" +
  "Turn up the room and follow the beat";
const ACCEPTED_RESOLVED_CAPTION =
  "A clean electric guitar plays a gentle, melodic chord progression with a " +
  "slightly funky, neo-soul feel. The tone is warm and direct, with a light " +
  "touch of reverb adding a touch of space to the arpeggiated chords. This " +
  "short, looping instrumental piece feels like a thoughtful intro or a " +
  "mellow interlude.";
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
const ACCEPTED_COT_TRAJECTORY_SHA256 =
  "476515e1db6ebc30e1622eb30ac02a8ef4289d89ca12e34c64b5f911bc960da2";
const ACCEPTED_SEMANTIC_CODE_SHA256 =
  "42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be";

const ACCEPTED_SEMANTIC_CODE_IDS = Object.freeze([
  44_244, 2_430, 7_541, 38_339, 43_500, 14_023, 63_719, 16_071,
  63_855, 58_755, 37_828, 20_336, 52_689, 20_856, 53_201, 11_240,
  17_873, 15_217, 22_037, 12_976, 20_848, 47_248, 54_461, 28_656,
  28_812, 12_669, 10_110, 63_939, 48_579, 19_199, 63_718, 28_871,
  51_055, 45_955, 27_133, 22_832, 53_202, 20_920, 53_200, 9_720,
  22_992, 17_785, 15_536, 12_797, 12_715, 40_782, 28_800, 41_083,
  28_411, 11_130, 21_929, 29_754, 10_008, 33_298, 35_986, 25_049,
  7_613, 57_560, 45_964, 14_910, 34_822, 18_567, 53_535, 28_710,
  7_230, 25_184, 7_480, 25_456, 25_572, 9_702, 61_421, 62_781,
  12_723, 63_314, 13_779, 57_352, 2_705, 51_234, 61_459, 50_923,
  28_968, 15_224, 19_416, 26_752, 26_688, 26_752, 39_560, 42_256,
  57_104, 57_176, 56_728, 56_800, 5_544, 8_112, 23_480, 24_568,
  28_820, 12_669, 10_101, 63_426, 58_828, 16_583, 18_435, 16_709,
  53_300, 20_421, 62_950, 36_528, 53_202, 10_236, 53_196, 37_950,
  57_820, 33_075, 31_276, 10_168, 10_171, 40_847, 26_434, 28_159,
  41_684, 12_669, 10_101, 63_426, 58_860, 16_583, 38_467, 903,
  53_357, 31_175, 62_934, 35_992, 22_938, 23_158, 1_621, 23_534,
  2_501, 24_576, 18_421, 12_279, 25_460, 51_159, 25_986, 50_117,
  33_807, 35_847, 35_847, 35_847, 35_847, 35_847,
]);

const FIXTURE_SPECS = Object.freeze([
  Object.freeze({ id: "semantic-early" as const, mode: "semantic" as const,
    constraint: "semantic-64000" as const, cachedTokensBeforeAppend: 268,
    cacheCapacity: 768, drawIndex: 125 }),
  Object.freeze({ id: "semantic-middle" as const, mode: "semantic" as const,
    constraint: "semantic-64000" as const, cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280, drawIndex: 185 }),
  Object.freeze({ id: "semantic-late" as const, mode: "semantic" as const,
    constraint: "semantic-64000" as const, cachedTokensBeforeAppend: 401,
    cacheCapacity: 2_048, drawIndex: 258 }),
  Object.freeze({ id: "cot-singleton" as const, mode: "cot" as const,
    constraint: "singleton" as const, cachedTokensBeforeAppend: 120,
    cacheCapacity: 512, drawIndex: 16 }),
  Object.freeze({ id: "cot-small" as const, mode: "cot" as const,
    constraint: "small" as const, cachedTokensBeforeAppend: 140,
    cacheCapacity: 768, drawIndex: 36 }),
  Object.freeze({ id: "cot-caption" as const, mode: "cot" as const,
    constraint: "caption" as const, cachedTokensBeforeAppend: 160,
    cacheCapacity: 1_024, drawIndex: 56 }),
  Object.freeze({ id: "cot-all" as const, mode: "cot" as const,
    constraint: "all" as const, cachedTokensBeforeAppend: 212,
    cacheCapacity: 2_048, drawIndex: 108 }),
]);

type FixtureSpec = (typeof FIXTURE_SPECS)[number];
const PREFILL_CORRECTNESS_SPEC = Object.freeze({
  id: "semantic-prefill" as const,
  mode: "semantic" as const,
  constraint: "semantic-64000" as const,
  cachedTokensBeforeAppend: 268,
  cacheCapacity: 768,
  drawIndex: 124,
});
type SamplingFixtureSpec = FixtureSpec | typeof PREFILL_CORRECTNESS_SPEC;

interface IncomingPrepareMessage {
  readonly type: "prepare";
  readonly identity: unknown;
}
interface IncomingRunMessage {
  readonly type: "run";
  readonly thermalLaunch: Opt0084ThermalLaunch;
}
interface IncomingCancelMessage { readonly type: "cancel" }
type IncomingMessage = IncomingPrepareMessage | IncomingRunMessage |
  IncomingCancelMessage;

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface FixturePlan {
  readonly spec: FixtureSpec;
  readonly prefill: AcePlannerPrefillBatch;
  readonly decode: AcePlannerDecodeBatch;
  readonly prefillSeenTokenIds: readonly number[];
  readonly seenTokenIds: readonly number[];
  readonly allowedTokens: AcePlannerAllowedTokens;
  readonly preCfgAllowedTokens?: AcePlannerAllowedTokens;
}

interface RetainedFixture extends FixturePlan {
  readonly logits: readonly Float32Array[];
  readonly initialLogitSha256: string;
  readonly workspace: AceOpt0084PlannerSamplingWorkspace;
  readonly statsAfterWarmup: AceOpt0084PlannerSamplingWorkspaceStats;
}

interface RetainedCorrectnessFixture {
  readonly spec: typeof PREFILL_CORRECTNESS_SPEC;
  readonly logits: readonly Float32Array[];
  readonly initialLogitSha256: string;
  readonly workspace: AceOpt0084PlannerSamplingWorkspace;
  readonly statsAfterWarmup: AceOpt0084PlannerSamplingWorkspaceStats;
  readonly seenTokenIds: readonly number[];
  readonly allowedTokens: AcePlannerAllowedTokens;
  readonly preCfgAllowedTokens: AcePlannerAllowedTokens;
}

type SamplingFixtureView = RetainedFixture | RetainedCorrectnessFixture;

interface PreparedSession {
  readonly runIdentity: Opt0084RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly executor: AcePlannerGpuExecutor;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly fixtures: readonly RetainedFixture[];
  readonly correctnessOnlyFixtures: readonly RetainedCorrectnessFixture[];
  readonly packageReceipt: Readonly<Record<string, unknown>>;
  readonly tokenizerReceipt: Readonly<Record<string, unknown>>;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  readonly correctnessReceipt: Readonly<Record<string, unknown>>;
  readonly warmupReceipt: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  cleanup(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let activeAbortController: AbortController | undefined;
let activePrepared: PreparedSession | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    activeAbortController?.abort(new DOMException(
      "OPT-0084 cancellation requested",
      "AbortError",
    ));
    if (lifecycle === "ready" && activePrepared !== undefined) {
      void failAndCleanup(
        "cancelled-ready",
        new DOMException("OPT-0084 cancelled while ready", "AbortError"),
      );
    }
    return;
  }
  if (message.type === "prepare" && lifecycle === "idle") {
    lifecycle = "preparing";
    let identity: Opt0084RunIdentity;
    try {
      identity = validateOpt0084RunIdentity(message.identity);
    } catch (error) {
      void failAndCleanup("run-identity", error);
      return;
    }
    void prepareHarness(identity).then((prepared) => {
      if (lifecycle !== "preparing") return;
      activePrepared = prepared;
      lifecycle = "ready";
      self.postMessage({
        type: "ready",
        readyAtEpochMilliseconds: prepared.readyAtEpochMilliseconds,
        preparation: prepared.preparationReceipt,
      });
    }, (error) => failAndCleanup("prepare", error));
    return;
  }
  if (
    message.type === "run" && lifecycle === "ready" &&
    activePrepared !== undefined
  ) {
    lifecycle = "running";
    const prepared = activePrepared;
    void runTiming(prepared, message.thermalLaunch).then((evidence) => {
      lifecycle = "settled";
      activePrepared = undefined;
      activeAbortController = undefined;
      self.postMessage({ type: "measurement-complete", evidence });
    }, (error) => failAndCleanup("timing", error));
  }
});

async function failAndCleanup(phase: string, error: unknown): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  const prepared = activePrepared;
  activePrepared = undefined;
  activeAbortController = undefined;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let cleanupError: unknown;
  try {
    cleanup = await prepared?.cleanup(error);
  } catch (caught) {
    cleanupError = caught;
  }
  self.postMessage({
    type: "failed",
    phase,
    error: errorValue(error),
    ...(cleanup === undefined ? {} : { cleanup }),
    ...(cleanupError === undefined ? {} : { cleanupError: errorValue(cleanupError) }),
  });
}

async function prepareHarness(
  runIdentity: Opt0084RunIdentity,
): Promise<PreparedSession> {
  const abortController = new AbortController();
  activeAbortController = abortController;
  const signal = abortController.signal;
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  const preparationStartedAtEpochMilliseconds = Date.now();
  try {
    validateFrozenFixtureConstants();
    const sourceSha256 = aceSha256Hex(new TextEncoder().encode(
      plannerSamplingSource,
    ));
    if (sourceSha256 !== EXPECTED_PLANNER_SAMPLING_SOURCE_SHA256) {
      throw new Error("OPT-0084 candidate sampler source changed from 245b5fe");
    }

    postProgress("authenticating and acquiring the reference-BF16 planner");
    const acquisitionStarted = performance.now();
    const preparedPackage = await preparePackage(signal);
    const acquisitionWallMilliseconds = performance.now() - acquisitionStarted;
    signal.throwIfAborted();

    postProgress("requesting the production reference-BF16 WebGPU profile");
    context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      schedulingProfile: "cooperative",
      signal,
      onRuntimeEvent: (event) => runtimeEvents.push(event),
    });

    postProgress("uploading the authenticated planner phase");
    const uploadStarted = performance.now();
    let lastUploadStatusAt = 0;
    phase = await AceGpuTensorPhase.load(
      context.device,
      preparedPackage.manifest,
      preparedPackage.acquiredFiles,
      ["planner"],
      {
        signal,
        onProgress: (progress) => {
          const now = performance.now();
          if (
            now - lastUploadStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
            progress.loadedPhaseBytes === progress.totalPhaseBytes
          ) {
            lastUploadStatusAt = now;
            postProgress(
              `uploading planner shard ${progress.phaseFileIndex + 1}/` +
                `${progress.phaseFileCount} ` +
                `(${formatBytes(progress.loadedPhaseBytes)}/` +
                `${formatBytes(progress.totalPhaseBytes)})`,
            );
          }
        },
      },
    );
    if (
      phase.phases.length !== 1 || phase.phases[0] !== "planner" ||
      phase.residentBytes !== PLANNER_RESIDENT_BYTES
    ) {
      throw new Error("OPT-0084 loaded planner identity changed");
    }
    const uploadWallMilliseconds = performance.now() - uploadStarted;

    postProgress("authenticating the pinned planner tokenizer");
    const tokenizerLoaded = await loadPinnedAceTokenizer("planner", {
      tokenizerJson: requirePackageFile(
        preparedPackage.acquiredFiles,
        TOKENIZER_FILE_NAMES[0],
      ),
      tokenizerConfigJson: requirePackageFile(
        preparedPackage.acquiredFiles,
        TOKENIZER_FILE_NAMES[1],
      ),
      chatTemplate: requirePackageFile(
        preparedPackage.acquiredFiles,
        TOKENIZER_FILE_NAMES[2],
      ),
    });
    const plans = createFixturePlans(tokenizerLoaded.tokenizer);
    executor = AcePlannerGpuExecutor.create({
      device: context.device,
      modelProfile: "reference-bf16",
      ownedPlannerWeights: phase,
      signal,
    });
    phase = undefined;

    const retained: RetainedFixture[] = [];
    const correctnessOnly: RetainedCorrectnessFixture[] = [];
    const fixturePreparation: Readonly<Record<string, unknown>>[] = [];
    for (let index = 0; index < plans.length; index += 1) {
      signal.throwIfAborted();
      const plan = plans[index]!;
      postProgress(
        `retaining actual BF16 logits ${index + 1}/${plans.length}: ` +
          plan.spec.id,
      );
      const prefillStarted = performance.now();
      const prefillReturned = await executor.prefill(plan.prefill);
      const prefillWallMilliseconds = performance.now() - prefillStarted;
      if (plan.spec.id === "semantic-early") {
        const prefillLogits = requireAndRetainRows(
          prefillReturned,
          2,
          PREFILL_CORRECTNESS_SPEC.id,
        );
        const prefillHash = hashRows(prefillLogits);
        const prefillWorkspace = new AceOpt0084PlannerSamplingWorkspace();
        correctnessOnly.push(Object.freeze({
          spec: PREFILL_CORRECTNESS_SPEC,
          logits: prefillLogits,
          initialLogitSha256: prefillHash,
          workspace: prefillWorkspace,
          statsAfterWarmup: prefillWorkspace.stats,
          seenTokenIds: plan.prefillSeenTokenIds,
          allowedTokens: plan.allowedTokens,
          preCfgAllowedTokens: plan.preCfgAllowedTokens!,
        }));
        fixturePreparation.push(Object.freeze({
          id: PREFILL_CORRECTNESS_SPEC.id,
          mode: PREFILL_CORRECTNESS_SPEC.mode,
          constraintFamily: PREFILL_CORRECTNESS_SPEC.constraint,
          executionKind: "actual-semantic-prefill-return-correctness-only",
          cachedTokensBeforeAppend:
            PREFILL_CORRECTNESS_SPEC.cachedTokensBeforeAppend,
          cacheCapacity: PREFILL_CORRECTNESS_SPEC.cacheCapacity,
          drawIndex: PREFILL_CORRECTNESS_SPEC.drawIndex,
          rows: 2,
          rowElements: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
          rawLogitSha256: prefillHash,
          prefillWallMilliseconds,
          includedInTiming: false,
        }));
      }
      signal.throwIfAborted();
      const decodeStarted = performance.now();
      const returned = await executor.decode(plan.decode);
      const decodeWallMilliseconds = performance.now() - decodeStarted;
      const expectedRows = plan.spec.mode === "semantic" ? 2 : 1;
      const logits = requireAndRetainRows(returned, expectedRows, plan.spec.id);
      const initialLogitSha256 = hashRows(logits);
      const workspace = new AceOpt0084PlannerSamplingWorkspace();
      retained.push({
        ...plan,
        logits,
        initialLogitSha256,
        workspace,
        statsAfterWarmup: workspace.stats,
      });
      fixturePreparation.push(Object.freeze({
        id: plan.spec.id,
        mode: plan.spec.mode,
        constraintFamily: plan.spec.constraint,
        cachedTokensBeforeAppend: plan.spec.cachedTokensBeforeAppend,
        cacheCapacity: plan.spec.cacheCapacity,
        drawIndex: plan.spec.drawIndex,
        rows: expectedRows,
        rowElements: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
        rawLogitSha256: initialLogitSha256,
        prefillWallMilliseconds,
        decodeWallMilliseconds,
        includedInTiming: true,
      }));
    }

    postProgress("checking filtered logits, weights, samples, words, and cursors");
    if (correctnessOnly.length !== 1) {
      throw new Error("OPT-0084 omitted its semantic prefill correctness fixture");
    }
    const correctnessStates = [...correctnessOnly, ...retained].map((fixture) =>
      checkFixtureCorrectness(fixture));
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0084 observed a WebGPU runtime event in preparation");
    }
    const correctnessReceipt = Object.freeze({
      passed: correctnessStates.every((state) => state["passed"] === true),
      states: Object.freeze(correctnessStates),
      fullInputValidationRetained: true,
      filteredLogitComparison: "raw-u32-over-complete-logical-vocabulary",
      weightComparison: "raw-u32-over-complete-logical-vocabulary",
      cursorComparison: "token-word-positive-count-draw-index-consumed",
      forcedTerminalState: "cot-singleton",
      semanticProductionBf16Positions: Object.freeze([
        "prefill", "early", "middle", "late",
      ]),
    });
    if (correctnessReceipt.passed !== true) {
      throw new Error("OPT-0084 exact sampler correctness failed");
    }

    postProgress("warming both sampler arms outside the timing interval");
    const warmedFixtures: RetainedFixture[] = [];
    const warmedCorrectnessOnly: RetainedCorrectnessFixture[] = [];
    const warmupStates: Readonly<Record<string, unknown>>[] = [];
    for (const fixture of [...correctnessOnly, ...retained]) {
      for (
        let invocation = 0;
        invocation < WARMUP_INVOCATIONS_PER_ARM_AND_STATE;
        invocation += 1
      ) {
        const word = timingWord(fixture.spec, -1 - invocation);
        const input = createSampleInput(fixture, word);
        const a = sampleAcePlannerToken(input);
        const b = fixture.workspace.sample(input);
        requireSameSample(a, b, `${fixture.spec.id} warmup ${invocation}`);
      }
      const statsAfterWarmup = fixture.workspace.stats;
      if (fixture.spec.id === PREFILL_CORRECTNESS_SPEC.id) {
        warmedCorrectnessOnly.push(Object.freeze({
          ...fixture,
          statsAfterWarmup,
        }) as RetainedCorrectnessFixture);
      } else {
        warmedFixtures.push(Object.freeze({
          ...fixture,
          statsAfterWarmup,
        }) as RetainedFixture);
      }
      warmupStates.push(Object.freeze({
        id: fixture.spec.id,
        invocationsPerArm: WARMUP_INVOCATIONS_PER_ARM_AND_STATE,
        workspace: workspaceReceipt(statsAfterWarmup),
      }));
    }
    const warmupReceipt = Object.freeze({
      bothArmsInitializedBeforeTiming: true,
      states: Object.freeze(warmupStates),
    });
    const readyAtEpochMilliseconds = Date.now();
    let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
    const ownedContext = context;
    const ownedExecutor = executor;
    const cleanup = (reason?: unknown): Promise<Readonly<Record<string, unknown>>> => {
      if (cleanupPromise !== undefined) {
        return cleanupPromise.then((receipt) => Object.freeze({
          ...receipt,
          repeatedCall: true,
        }));
      }
      cleanupPromise = (async () => {
        const cleanupStartedAtEpochMilliseconds = Date.now();
        let destroyError: unknown;
        try {
          await ownedExecutor.destroy(reason);
          await ownedExecutor.destroy(reason);
        } catch (error) {
          destroyError = error;
        } finally {
          ownedContext.destroy();
          ownedContext.destroy();
        }
        const cleanupCompletedAtEpochMilliseconds = Date.now();
        if (destroyError !== undefined) throw destroyError;
        return Object.freeze({
          cleanupStartedAtEpochMilliseconds,
          cleanupCompletedAtEpochMilliseconds,
          executorDestroyCalledTwice: true,
          deviceDestroyCalledTwice: true,
          idempotentOwnerCleanup: true,
          runtimeEventCount: runtimeEvents.length,
          repeatedCall: false,
        });
      })();
      return cleanupPromise;
    };
    const preparationReceipt = Object.freeze({
      experiment: OPT_0084_EXPERIMENT_ID,
      runIdentity,
      candidateSeamCommit: OPT_0084_CANDIDATE_SEAM_COMMIT,
      plannerSamplingSourceSha256: sourceSha256,
      preparationStartedAtEpochMilliseconds,
      readyAtEpochMilliseconds,
      package: preparedPackage.receipt,
      tokenizer: tokenizerLoaded.assetIdentity,
      execution: Object.freeze({
        modelProfile: "reference-bf16",
        executionProfile: context.capabilities.executionProfile.id,
        schedulingProfile: "cooperative",
        adapterInfo: context.capabilities.adapterInfo,
        onePlannerExecutor: true,
        actualDecodeRowsRetainedBeforeTiming: true,
        actualPrefillRowsRetainedForCorrectness: true,
        correctnessFixtureCount:
          warmedFixtures.length + warmedCorrectnessOnly.length,
        timedFixtureCount: warmedFixtures.length,
        acquisitionWallMilliseconds,
        uploadWallMilliseconds,
      }),
      fixturePreparation: Object.freeze(fixturePreparation),
      correctness: correctnessReceipt,
      warmup: warmupReceipt,
    });
    return Object.freeze({
      runIdentity,
      context,
      executor,
      runtimeEvents,
      fixtures: Object.freeze(warmedFixtures),
      correctnessOnlyFixtures: Object.freeze(warmedCorrectnessOnly),
      packageReceipt: preparedPackage.receipt,
      tokenizerReceipt: tokenizerLoaded.assetIdentity,
      preparationReceipt,
      correctnessReceipt,
      warmupReceipt,
      readyAtEpochMilliseconds,
      cleanup,
    });
  } catch (error) {
    const cleanupFailures: unknown[] = [];
    try {
      if (executor !== undefined) await executor.destroy(error);
      else phase?.destroy();
    } catch (cleanupError) {
      cleanupFailures.push(cleanupError);
    } finally {
      try {
        context?.destroy();
      } catch (contextDestroyError) {
        cleanupFailures.push(contextDestroyError);
      }
    }
    if (cleanupFailures.length > 0) {
      throw new Error(
        `OPT-0084 preparation failed (${errorText(error)}); cleanup also ` +
          `failed: ${cleanupFailures.map(errorText).join("; ")}`,
        { cause: error },
      );
    }
    throw error;
  }
}

async function runTiming(
  prepared: PreparedSession,
  thermalLaunch: Opt0084ThermalLaunch,
): Promise<Readonly<Record<string, unknown>>> {
  validateThermalLaunch(thermalLaunch, prepared.readyAtEpochMilliseconds);
  const signal = activeAbortController!.signal;
  signal.throwIfAborted();
  const measurementStartedAtEpochMilliseconds = Date.now();
  if (measurementStartedAtEpochMilliseconds <
      thermalLaunch.gateCompletedAtEpochMilliseconds) {
    throw new Error("OPT-0084 timing began before its nominal thermal gate");
  }
  const samples: Opt0084TimingSample[] = [];
  for (let roundIndex = 0; roundIndex < OPT_0084_TIMING_ROUND_COUNT;
    roundIndex += 1) {
    const order = OPT_0084_PAIR_ORDERS[roundIndex]!;
    const stateOrder = rotateStates(prepared.fixtures, roundIndex);
    for (let armPosition = 0; armPosition < order.length; armPosition += 1) {
      const arm = order[armPosition]!;
      for (let statePosition = 0; statePosition < stateOrder.length;
        statePosition += 1) {
        signal.throwIfAborted();
        const fixture = stateOrder[statePosition]!;
        const word = timingWord(fixture.spec, roundIndex);
        const input = createSampleInput(fixture, word);
        const started = performance.now();
        const sampled = arm === "A"
          ? sampleAcePlannerToken(input)
          : fixture.workspace.sample(input);
        const ended = performance.now();
        samples.push(Object.freeze({
          roundIndex,
          stateId: fixture.spec.id,
          arm,
          armPosition: armPosition as 0 | 1,
          statePosition,
          wallMilliseconds: ended - started,
          tokenId: sampled.tokenId,
          word: sampled.word,
          positiveCandidateCount: sampled.positiveCandidateCount,
        }));
      }
    }
    postProgress(`timed pair ${roundIndex + 1}/${OPT_0084_TIMING_ROUND_COUNT}`);
    await yieldToWorker();
  }
  const measurementCompletedAtEpochMilliseconds = Date.now();
  const timing = evaluateOpt0084Timing(samples);
  const postTimingFixtureChecks: Readonly<Record<string, unknown>>[] = [];
  for (const fixture of [
    ...prepared.correctnessOnlyFixtures,
    ...prepared.fixtures,
  ]) {
    const finalHash = hashRows(fixture.logits);
    const finalStats = fixture.workspace.stats;
    if (finalHash !== fixture.initialLogitSha256) {
      throw new Error(`${fixture.spec.id} retained actual logits mutated`);
    }
    if (!sameWorkspaceStats(finalStats, fixture.statsAfterWarmup)) {
      throw new Error(`${fixture.spec.id} OPT-0084 workspace grew during timing`);
    }
    postTimingFixtureChecks.push(Object.freeze({
      id: fixture.spec.id,
      initialRawLogitSha256: fixture.initialLogitSha256,
      finalRawLogitSha256: finalHash,
      immutable: true,
      workspaceBeforeTiming: workspaceReceipt(fixture.statsAfterWarmup),
      workspaceAfterTiming: workspaceReceipt(finalStats),
      allocationReusePassed: true,
    }));
  }
  const cancellation = runCancellationCheckpointProof(prepared.fixtures[0]!);
  if (prepared.runtimeEvents.length !== 0) {
    throw new Error("OPT-0084 observed a WebGPU runtime event before cleanup");
  }
  const cleanupFirst = await prepared.cleanup();
  const cleanupSecond = await prepared.cleanup();
  const cleanupCompletedAtEpochMilliseconds = requireReceiptNumber(
    cleanupFirst,
    "cleanupCompletedAtEpochMilliseconds",
  );
  const cleanupPassed = cleanupFirst["idempotentOwnerCleanup"] === true &&
    cleanupFirst["runtimeEventCount"] === 0 &&
    cleanupSecond["repeatedCall"] === true;
  const inPagePassed = prepared.correctnessReceipt["passed"] === true &&
    timing.passed && cancellation["passed"] === true && cleanupPassed;
  return Object.freeze({
    schema: OPT_0084_RECEIPT_SCHEMA,
    experiment: OPT_0084_EXPERIMENT_ID,
    status: "awaiting-external-thermal-completion",
    passed: false,
    inPagePassed,
    thermalLaunch,
    cleanupCompletedAtEpochMilliseconds,
    identity: Object.freeze({
      ...prepared.runIdentity,
      candidateSeamCommit: OPT_0084_CANDIDATE_SEAM_COMMIT,
      plannerSamplingSourceSha256: EXPECTED_PLANNER_SAMPLING_SOURCE_SHA256,
      manifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
      userAgent: self.navigator.userAgent,
      hardwareConcurrency: self.navigator.hardwareConcurrency,
      performanceTimeOriginEpochMilliseconds: performance.timeOrigin,
    }),
    package: prepared.packageReceipt,
    tokenizer: prepared.tokenizerReceipt,
    protocol: Object.freeze({
      benchmarkOnly: true,
      productionChanged: false,
      productionIntegrationAuthorized: false,
      modelProfile: "reference-bf16",
      actualImmutableProductionBf16Logits: true,
      semanticCorrectnessPositions: Object.freeze([
        "prefill", "early", "middle", "late",
      ]),
      semanticTimedPositions: Object.freeze(["early", "middle", "late"]),
      cotConstraintFamilies: Object.freeze([
        "singleton", "small", "caption", "all",
      ]),
      samplerOnlyTiming: true,
      modelExecutionExcludedFromTiming: true,
      bothArmsInitializedAndWarmedOutsideTiming: true,
      balancedInterleavedPairCount: OPT_0084_TIMING_ROUND_COUNT,
      timingAuthority: "worker-performance-now-synchronous-complete-sampler-wall",
      armA: "current-ace-browser-softmax-v1-full-vector-production-sampler",
      armB: "opt0084-reused-candidate-domain-radix-workspace",
      defaultProjection: "mean-three-semantic-position-medians-times-900-draws",
      allocationCountsAndBytesDiagnosticOnly: true,
      timingRetryPerformed: false,
    }),
    preparation: prepared.preparationReceipt,
    correctness: prepared.correctnessReceipt,
    warmup: prepared.warmupReceipt,
    timing: Object.freeze({
      ...timing,
      measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(samples),
    }),
    postTimingFixtureChecks: Object.freeze(postTimingFixtureChecks),
    cancellation,
    runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
    cleanup: Object.freeze({
      firstCall: cleanupFirst,
      secondCall: cleanupSecond,
      passed: cleanupPassed,
    }),
    decision: Object.freeze({
      disposition: inPagePassed
        ? "positive-browser-sampler-gate-integration-gates-still-required"
        : "negative-stop-browser-sampler-gate",
      browserSamplerGatePassed: inPagePassed,
      completeTrajectoryGateRequired: inPagePassed,
      plannerEnabledProductGateRequired: inPagePassed,
      productionIntegrationAuthorized: false,
      unchangedTimingRetryAuthorized: false,
    }),
  });
}

function checkFixtureCorrectness(
  fixture: SamplingFixtureView,
): Readonly<Record<string, unknown>> {
  const word = aceRandomWord(
    ACCEPTED_SEED,
    "planner-sampling",
    BigInt(fixture.spec.drawIndex),
  );
  const input = createSampleInput(fixture, word);
  const aFiltered = createAcePlannerFilteredLogits(withoutWord(input));
  const aWeights = createAcePlannerBrowserSamplingWeights(
    aFiltered,
    input.parameters.temperature,
  );
  const aSample = sampleAcePlannerToken(input);
  const bSample = fixture.workspace.sample(input);
  requireSameSample(aSample, bSample, `${fixture.spec.id} direct sample`);
  const candidateIds = fixture.workspace.copyLastCandidateTokenIds();
  const bFiltered = fixture.workspace.copyLastFilteredLogits();
  const bWeights = fixture.workspace.copyLastWeights();
  if (
    candidateIds.length !== bFiltered.length ||
    candidateIds.length !== bWeights.length
  ) {
    throw new Error(`${fixture.spec.id} candidate diagnostics differ in length`);
  }
  const aFilteredWords = words(aFiltered);
  const aWeightWords = words(aWeights);
  const bFilteredWords = words(bFiltered);
  const bWeightWords = words(bWeights);
  const negativeInfinityWord = rawWord(Number.NEGATIVE_INFINITY);
  let candidateOrdinal = 0;
  let comparedCandidateWords = 0;
  let checkedOmittedTokens = 0;
  for (let tokenId = 0; tokenId < ACE_PLANNER_QWEN3_CONFIG.vocabularySize;
    tokenId += 1) {
    const candidateTokenId = candidateIds[candidateOrdinal];
    if (candidateTokenId === tokenId) {
      if (
        bFilteredWords[candidateOrdinal] !== aFilteredWords[tokenId] ||
        bWeightWords[candidateOrdinal] !== aWeightWords[tokenId]
      ) {
        throw new Error(
          `${fixture.spec.id} raw filtered/weight mismatch at token ${tokenId}`,
        );
      }
      candidateOrdinal += 1;
      comparedCandidateWords += 2;
    } else {
      if (
        aFilteredWords[tokenId] !== negativeInfinityWord ||
        aWeightWords[tokenId] !== 0
      ) {
        throw new Error(
          `${fixture.spec.id} omitted token ${tokenId} remained active`,
        );
      }
      checkedOmittedTokens += 1;
    }
  }
  if (candidateOrdinal !== candidateIds.length) {
    throw new Error(`${fixture.spec.id} candidate IDs are not ascending/complete`);
  }

  const aCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED,
    fixture.spec.drawIndex);
  const bCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED,
    fixture.spec.drawIndex);
  const cursorInput = withoutWord(input);
  const aCursorSample = aCursor.sample(cursorInput);
  const bCursorSample = bCursor.sampleOpt0084(cursorInput);
  requireSameCursorSample(aCursorSample, bCursorSample,
    `${fixture.spec.id} cursor`);
  if (
    aCursor.consumed !== bCursor.consumed ||
    aCursor.consumed !== BigInt(fixture.spec.drawIndex + 1) ||
    aCursorSample.word !== word || bCursorSample.word !== word
  ) {
    throw new Error(`${fixture.spec.id} word or cursor commit differs`);
  }
  const hashAfterCorrectness = hashRows(fixture.logits);
  if (hashAfterCorrectness !== fixture.initialLogitSha256) {
    throw new Error(`${fixture.spec.id} input logits mutated during correctness`);
  }
  return Object.freeze({
    id: fixture.spec.id,
    passed: true,
    constraintFamily: fixture.spec.constraint,
    rawLogitSha256: fixture.initialLogitSha256,
    candidateCount: candidateIds.length,
    logicalVocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    comparedCandidateWords,
    checkedOmittedTokens,
    filteredLogitMismatchCount: 0,
    weightMismatchCount: 0,
    armAFullFilteredLogitSha256: hashFloat32(aFiltered),
    armAFullWeightSha256: hashFloat32(aWeights),
    armBCandidateTokenIdU32LeSha256: hashU32(Array.from(candidateIds)),
    armBCandidateFilteredLogitSha256: hashFloat32(bFiltered),
    armBCandidateWeightSha256: hashFloat32(bWeights),
    sample: sampleReceipt(aSample),
    cursor: cursorSampleReceipt(aCursorSample, aCursor),
    workspaceAfterCorrectness: workspaceReceipt(fixture.workspace.stats),
  });
}

function runCancellationCheckpointProof(
  fixture: RetainedFixture,
): Readonly<Record<string, unknown>> {
  const controller = new AbortController();
  controller.abort(new DOMException("OPT-0084 cancellation probe", "AbortError"));
  const aCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, 700);
  const bCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, 700);
  const aBefore = aCursor.consumed;
  const bBefore = bCursor.consumed;
  let aRejected = false;
  let bRejected = false;
  try {
    controller.signal.throwIfAborted();
    aCursor.sample(withoutWord(createSampleInput(fixture, 0)));
  } catch (error) {
    aRejected = error instanceof DOMException && error.name === "AbortError";
  }
  try {
    controller.signal.throwIfAborted();
    bCursor.sampleOpt0084(withoutWord(createSampleInput(fixture, 0)));
  } catch (error) {
    bRejected = error instanceof DOMException && error.name === "AbortError";
  }
  const passed = aRejected && bRejected && aCursor.consumed === aBefore &&
    bCursor.consumed === bBefore;
  if (!passed) throw new Error("OPT-0084 cancellation checkpoint committed a draw");
  return Object.freeze({
    passed,
    checkpointBeforeEachBoundedTimedSample: true,
    armARejectedBeforeSample: aRejected,
    armBRejectedBeforeSample: bRejected,
    armACursorBefore: aBefore.toString(),
    armACursorAfter: aCursor.consumed.toString(),
    armBCursorBefore: bBefore.toString(),
    armBCursorAfter: bCursor.consumed.toString(),
  });
}

function createSampleInput(
  fixture: Pick<SamplingFixtureView, "logits" | "seenTokenIds" |
    "allowedTokens" | "preCfgAllowedTokens" | "spec">,
  word: number,
): AcePlannerTokenSampleInput {
  const semantic = fixture.spec.mode === "semantic";
  return Object.freeze({
    conditionalLogits: fixture.logits[0]!,
    ...(semantic ? { unconditionalLogits: fixture.logits[1]! } : {}),
    seenTokenIds: fixture.seenTokenIds,
    ...(fixture.preCfgAllowedTokens === undefined
      ? {}
      : { preCfgAllowedTokens: fixture.preCfgAllowedTokens }),
    allowedTokens: fixture.allowedTokens,
    parameters: Object.freeze({
      temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
      guidanceScale: semantic
        ? DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale
        : 1,
      topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
      topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
      repetitionPenalty: 1,
    }),
    word,
    softmax: ACE_BROWSER_SOFTMAX_V1,
  });
}

function withoutWord(
  input: AcePlannerTokenSampleInput,
): Omit<AcePlannerTokenSampleInput, "word"> {
  return Object.freeze({
    conditionalLogits: input.conditionalLogits,
    ...(input.unconditionalLogits === undefined
      ? {}
      : { unconditionalLogits: input.unconditionalLogits }),
    seenTokenIds: input.seenTokenIds,
    ...(input.preCfgAllowedTokens === undefined
      ? {}
      : { preCfgAllowedTokens: input.preCfgAllowedTokens }),
    allowedTokens: input.allowedTokens,
    parameters: input.parameters,
    ...(input.softmax === undefined ? {} : { softmax: input.softmax }),
  });
}

function createFixturePlans(tokenizer: AceQwenBpeTokenizer): readonly FixturePlan[] {
  const cotPrompt = createAcePlannerCotPrompt(ACCEPTED_PROMPT, ACCEPTED_LYRICS);
  const codePrompts = createAcePlannerCodePrompts(
    ACCEPTED_RESOLVED_CAPTION,
    ACCEPTED_LYRICS,
    ACCEPTED_COT_TEXT,
  );
  const cotBase = tokenizer.encode(cotPrompt);
  const cotContinuation = createReceiptTextDerivedCotTeacherTokens(
    tokenizer,
    cotBase,
  );
  const semanticBases = Object.freeze([
    tokenizer.encode(codePrompts.conditional),
    tokenizer.encode(codePrompts.unconditional),
  ]);
  const semanticContinuation = ACCEPTED_SEMANTIC_CODE_IDS.map(
    (code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code,
  );
  if (
    cotBase.length !== 105 || cotContinuation.length !== 109 ||
    semanticBases[0]!.length !== 253 || semanticBases[1]!.length !== 33 ||
    semanticContinuation.length !== 150 ||
    hashU32(cotContinuation) !== ACCEPTED_COT_TRAJECTORY_SHA256 ||
    hashU32(ACCEPTED_SEMANTIC_CODE_IDS) !== ACCEPTED_SEMANTIC_CODE_SHA256
  ) {
    throw new Error("OPT-0084 accepted planner fixture identity changed");
  }
  const captionAllowedTokens = createCaptionAllowedTokens();
  const semanticAllowedTokens: AcePlannerAllowedTokens = Object.freeze({
    kind: "range",
    firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
  });
  const semanticPreCfgAllowedTokens: AcePlannerAllowedTokens = Object.freeze({
    kind: "range",
    firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
    additionalTokenIds: Object.freeze([ACE_QWEN_IM_END_TOKEN_ID]),
  });
  return Object.freeze(FIXTURE_SPECS.map((spec) => {
    const baseRows = spec.mode === "semantic"
      ? semanticBases
      : Object.freeze([cotBase]);
    const continuation = spec.mode === "semantic"
      ? semanticContinuation
      : cotContinuation;
    const baseWidth = Math.max(...baseRows.map((row) => row.length));
    const continuationIndex = spec.cachedTokensBeforeAppend - baseWidth;
    const decodeTokenId = continuation[continuationIndex];
    if (decodeTokenId === undefined) {
      throw new Error(`${spec.id} leaves the accepted planner trajectory`);
    }
    const prefill = createPaddedPrefill(
      baseRows,
      continuation,
      spec.cachedTokensBeforeAppend,
      spec.cacheCapacity,
    );
    const decode = createDecodeBatch(
      baseRows.length as 1 | 2,
      spec.cacheCapacity,
      spec.cachedTokensBeforeAppend,
      decodeTokenId,
    );
    const allowedTokens = spec.constraint === "semantic-64000"
      ? semanticAllowedTokens
      : spec.constraint === "singleton"
      ? Object.freeze({ kind: "ids" as const,
          tokenIds: Object.freeze([ACE_QWEN_IM_END_TOKEN_ID]) })
      : spec.constraint === "small"
      ? createRepresentativeSmallConstraint(tokenizer)
      : spec.constraint === "caption"
      ? captionAllowedTokens
      : Object.freeze({ kind: "all" as const });
    return Object.freeze({
      spec,
      prefill,
      decode,
      prefillSeenTokenIds: Object.freeze([
        ...baseRows[0]!,
        ...continuation.slice(0, continuationIndex),
      ]),
      seenTokenIds: Object.freeze([
        ...baseRows[0]!,
        ...continuation.slice(0, continuationIndex + 1),
      ]),
      allowedTokens,
      ...(spec.mode === "semantic"
        ? { preCfgAllowedTokens: semanticPreCfgAllowedTokens }
        : {}),
    });
  }));
}

function createRepresentativeSmallConstraint(
  tokenizer: AceQwenBpeTokenizer,
): AcePlannerAllowedTokens {
  const fragments = [" 30", " 60", " 90", " 100", " 120", " 180", "\n"];
  const tokenIds = [...new Set(fragments.flatMap((text) =>
    tokenizer.encode(text)))].sort((left, right) => left - right);
  if (tokenIds.length < 2 || tokenIds.length > 32) {
    throw new Error("OPT-0084 representative small constraint changed");
  }
  return Object.freeze({ kind: "ids", tokenIds: Object.freeze(tokenIds) });
}

function createCaptionAllowedTokens(): AcePlannerAllowedTokens {
  const tokenIds: number[] = [];
  const audioEnd = ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID +
    ACE_PLANNER_SEMANTIC_CODE_COUNT;
  for (let tokenId = 0; tokenId < ACE_PLANNER_QWEN3_CONFIG.vocabularySize;
    tokenId += 1) {
    if (tokenId === BACKTICK_TOKEN_ID) continue;
    if (tokenId >= ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID &&
      tokenId < audioEnd) continue;
    tokenIds.push(tokenId);
  }
  if (tokenIds.length !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize -
      ACE_PLANNER_SEMANTIC_CODE_COUNT - 1) {
    throw new Error("OPT-0084 caption-domain cardinality changed");
  }
  return Object.freeze({ kind: "ids", tokenIds: Object.freeze(tokenIds) });
}

function createReceiptTextDerivedCotTeacherTokens(
  tokenizer: AceQwenBpeTokenizer,
  promptTokenIds: readonly number[],
): readonly number[] {
  const closingTag = "</think>";
  if (!ACCEPTED_COT_TEXT.endsWith(closingTag)) {
    throw new Error("OPT-0084 accepted CoT closing tag changed");
  }
  let remaining = ACCEPTED_COT_TEXT.slice(0, -closingTag.length);
  const controller = new AcePlannerMetadataConstraintController({ tokenizer });
  const accepted: number[] = [];
  const logits = new Float32Array(ACE_PLANNER_QWEN3_CONFIG.vocabularySize);
  logits.fill(-1);
  while (remaining.length > 0) {
    const candidates: Array<Readonly<{ id: number; text: string }>> = [];
    for (let length = 1; length <= Math.min(96, remaining.length);
      length += 1) {
      const text = remaining.slice(0, length);
      const tokenIds = tokenizer.encode(text);
      if (tokenIds.length === 1 && tokenizer.decode(tokenIds) === text) {
        candidates.push(Object.freeze({ id: tokenIds[0]!, text }));
      }
    }
    for (const candidate of candidates) logits[candidate.id] = candidate.text.length;
    const step = accepted.length;
    const allowed = controller.allowedTokens({
      step,
      promptTokenIds,
      emittedTokenIds: accepted,
      logits,
    });
    const chosen = candidates
      .filter((candidate) => allowedTokenIncludes(allowed, candidate.id))
      .sort((left, right) => right.text.length - left.text.length)[0];
    if (chosen === undefined) {
      throw new Error(`OPT-0084 accepted CoT cannot advance at step ${step}`);
    }
    accepted.push(chosen.id);
    controller.acceptToken({
      step,
      tokenId: chosen.id,
      tokenText: chosen.text,
      emittedTokenIds: accepted,
    });
    for (const candidate of candidates) logits[candidate.id] = -1;
    remaining = remaining.slice(chosen.text.length);
  }
  const terminal = controller.allowedTokens({
    step: accepted.length,
    promptTokenIds,
    emittedTokenIds: accepted,
    logits,
  });
  if (
    terminal.kind !== "ids" || terminal.tokenIds.length !== 1 ||
    terminal.tokenIds[0] !== ACE_QWEN_IM_END_TOKEN_ID
  ) {
    throw new Error("OPT-0084 accepted CoT did not reach terminal EOS");
  }
  accepted.push(ACE_QWEN_IM_END_TOKEN_ID);
  return Object.freeze(accepted);
}

function allowedTokenIncludes(
  allowed: AcePlannerAllowedTokens,
  tokenId: number,
): boolean {
  if (allowed.kind === "all") return true;
  if (allowed.kind === "ids") return allowed.tokenIds.includes(tokenId);
  return tokenId >= allowed.firstTokenId &&
      tokenId < allowed.firstTokenId + allowed.tokenCount ||
    (allowed.additionalTokenIds ?? []).includes(tokenId);
}

function createPaddedPrefill(
  baseRows: readonly (readonly number[])[],
  continuation: readonly number[],
  tokens: number,
  cacheCapacity: number,
): AcePlannerPrefillBatch {
  const rows = baseRows.length;
  if ((rows !== 1 && rows !== 2) || tokens >= cacheCapacity) {
    throw new Error("OPT-0084 planner prefill geometry is invalid");
  }
  const baseWidth = Math.max(...baseRows.map((row) => row.length));
  if (baseWidth >= tokens || tokens - baseWidth >= continuation.length) {
    throw new Error("OPT-0084 planner prefill leaves its accepted trajectory");
  }
  const inputIds = new Uint32Array(rows * tokens);
  const keyValidity = new Uint32Array(rows * tokens);
  for (let row = 0; row < rows; row += 1) {
    const base = baseRows[row]!;
    const rowOffset = row * tokens;
    const leftPadding = baseWidth - base.length;
    inputIds.fill(ACE_QWEN_PAD_TOKEN_ID, rowOffset, rowOffset + leftPadding);
    inputIds.set(base, rowOffset + leftPadding);
    keyValidity.fill(1, rowOffset + leftPadding, rowOffset + baseWidth);
    for (let position = baseWidth; position < tokens; position += 1) {
      inputIds[rowOffset + position] = continuation[position - baseWidth]!;
      keyValidity[rowOffset + position] = 1;
    }
  }
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens,
    cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(0),
    validKeyLengths: Array<number>(rows).fill(tokens),
    sourceValidity: [...keyValidity],
  });
  return Object.freeze({
    kind: "prefill",
    rows: rows as 1 | 2,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  });
}

function createDecodeBatch(
  rows: 1 | 2,
  cacheCapacity: number,
  cachedTokensBeforeAppend: number,
  decodeTokenId: number,
): AcePlannerDecodeBatch {
  const inputIds = new Uint32Array(rows);
  inputIds.fill(decodeTokenId);
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens: 1,
    cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(cachedTokensBeforeAppend),
    validKeyLengths: Array<number>(rows).fill(cachedTokensBeforeAppend + 1),
    sourceValidity: Array<number>(rows).fill(1),
  });
  return Object.freeze({
    kind: "decode",
    rows,
    tokens: 1,
    cacheCapacity,
    cachedTokensBeforeAppend,
    inputIds,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  });
}

async function preparePackage(signal: AbortSignal): Promise<PreparedPackage> {
  const manifestUrl = new URL(MANIFEST_PATH, self.location.href).href;
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    expectedProfile: "reference",
    signal,
  });
  const provenance = loaded.manifest.provenance;
  if (
    loaded.manifestSha256 !== ACE_REFERENCE_MANIFEST_SHA256 ||
    provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION ||
    provenance.aceSnapshot !== ACE_MODEL_SNAPSHOT_REVISION ||
    provenance.plannerSnapshot !== ACE_PLANNER_SNAPSHOT_REVISION ||
    provenance.referenceCommit !== ACE_REFERENCE_SOURCE_REVISION
  ) {
    throw new Error("OPT-0084 current reference manifest identity changed");
  }
  const inventory = validatePlannerInventory(loaded.manifest);
  const acquisitionManifest: AcePackageManifest = Object.freeze({
    ...loaded.manifest,
    files: inventory.files,
  });
  const cache = await AceOpfsModelCache.open();
  let lastStatusAt = 0;
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    signal,
    onFileProgress: (progress) => {
      const now = performance.now();
      if (
        now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
        progress.fileIndex + 1 === progress.fileCount &&
          progress.fileReceivedBytes === progress.fileBytes
      ) {
        lastStatusAt = now;
        postProgress(
          `acquiring planner file ${progress.fileIndex + 1}/` +
            `${progress.fileCount} (${formatBytes(progress.completedBytes)}/` +
            `${formatBytes(progress.totalBytes)}, ${progress.source})`,
        );
      }
    },
  });
  if (
    acquired.files.size !== ACQUIRED_FILE_COUNT ||
    acquired.plan.files.length !== ACQUIRED_FILE_COUNT
  ) {
    throw new Error("OPT-0084 bounded planner acquisition changed");
  }
  return Object.freeze({
    manifest: loaded.manifest,
    acquiredFiles: acquired.files,
    receipt: Object.freeze({
      manifestPath: MANIFEST_PATH,
      manifestUrl: loaded.manifestUrl,
      manifestSha256: loaded.manifestSha256,
      manifestByteLength: loaded.manifestByteLength,
      manifestId: loaded.manifestId,
      profile: loaded.manifest.profile,
      converterRevision: provenance.converterRevision,
      aceSnapshot: provenance.aceSnapshot,
      plannerSnapshot: provenance.plannerSnapshot,
      referenceCommit: provenance.referenceCommit,
      plannerTensorCount: inventory.tensorCount,
      plannerWeightFileCount: inventory.weightFileCount,
      tokenizerFileCount: TOKENIZER_FILE_NAMES.length,
      plannerResidentBytes: inventory.residentBytes,
      acquiredFileCount: acquired.files.size,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
    }),
  });
}

function validatePlannerInventory(manifest: AcePackageManifest): Readonly<{
  readonly files: readonly AcePackageFileRecord[];
  readonly tensorCount: number;
  readonly weightFileCount: number;
  readonly residentBytes: number;
}> {
  const tensors = Object.values(manifest.tensors).filter(
    (tensor) => tensor.phase === "planner",
  );
  const weightNames = new Set(tensors.map((tensor) => tensor.shard));
  const tokenizerNames = new Set<string>(TOKENIZER_FILE_NAMES);
  const files = manifest.files.filter((file) =>
    weightNames.has(file.name) || tokenizerNames.has(file.name));
  const weightFiles = files.filter((file) => weightNames.has(file.name));
  const residentBytes = sumSafe(
    weightFiles.map((file) => file.byteLength),
    "OPT-0084 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0084 planner tensor bytes",
  );
  if (
    tensors.length !== PLANNER_TENSOR_COUNT ||
    weightNames.size !== PLANNER_WEIGHT_FILE_COUNT ||
    weightFiles.length !== PLANNER_WEIGHT_FILE_COUNT ||
    files.length !== ACQUIRED_FILE_COUNT ||
    residentBytes !== PLANNER_RESIDENT_BYTES ||
    tensorBytes !== PLANNER_RESIDENT_BYTES ||
    TOKENIZER_FILE_NAMES.some((name) => !files.some((file) => file.name === name))
  ) {
    throw new Error("OPT-0084 reference planner inventory changed");
  }
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    weightFileCount: weightFiles.length,
    residentBytes,
  });
}

function requireAndRetainRows(
  rows: readonly ArrayLike<number>[],
  expectedRows: number,
  label: string,
): readonly Float32Array[] {
  if (rows.length !== expectedRows) {
    throw new Error(`${label} returned ${rows.length} rows`);
  }
  return Object.freeze(rows.map((row, index) => {
    if (!(row instanceof Float32Array) ||
      row.length !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize) {
      throw new Error(`${label} row ${index} has the wrong storage or extent`);
    }
    for (let tokenId = 0; tokenId < row.length; tokenId += 1) {
      const value = row[tokenId]!;
      if (Number.isNaN(value) || value === Number.POSITIVE_INFINITY) {
        throw new Error(`${label} row ${index} has invalid logit ${tokenId}`);
      }
    }
    return row.slice();
  }));
}

function timingWord(spec: SamplingFixtureSpec, roundIndex: number): number {
  const draw = BigInt(spec.drawIndex) + BigInt(roundIndex + 101) * 1_024n;
  return aceRandomWord(ACCEPTED_SEED, "planner-sampling", draw);
}

function rotateStates(
  fixtures: readonly RetainedFixture[],
  roundIndex: number,
): readonly RetainedFixture[] {
  const offset = roundIndex % fixtures.length;
  return Object.freeze([
    ...fixtures.slice(offset),
    ...fixtures.slice(0, offset),
  ]);
}

function validateThermalLaunch(
  thermal: Opt0084ThermalLaunch,
  readyAtEpochMilliseconds: number,
): void {
  const duration = thermal.gateCompletedAtEpochMilliseconds -
    thermal.gateStartedAtEpochMilliseconds;
  if (
    thermal.source !== OPT_0084_THERMAL_SOURCE ||
    thermal.command !== OPT_0084_THERMAL_COMMAND ||
    thermal.pollMilliseconds !== OPT_0084_THERMAL_POLL_MILLISECONDS ||
    thermal.traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    thermal.gateStartedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    duration < 30_000 ||
    thermal.observationCount < Math.floor(duration /
      OPT_0084_THERMAL_POLL_MILLISECONDS) + 1 ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds > 1_250 ||
    thermal.nonNominalObservationCount !== 0 ||
    thermal.missingObservationCount !== 0 ||
    thermal.readyToGateDelayMilliseconds !==
      thermal.gateStartedAtEpochMilliseconds - readyAtEpochMilliseconds ||
    thermal.launchDelayMilliseconds < 0 ||
    thermal.launchDelayMilliseconds > 5_000
  ) {
    throw new Error("OPT-0084 worker rejected the nominal thermal launch gate");
  }
}

function workspaceReceipt(
  stats: AceOpt0084PlannerSamplingWorkspaceStats,
): Readonly<Record<string, unknown>> {
  const retainedBytes = stats.candidateCapacity * 16 + stats.maskCapacity +
    512 * Uint32Array.BYTES_PER_ELEMENT;
  return Object.freeze({
    candidateCapacity: stats.candidateCapacity,
    maskCapacity: stats.maskCapacity,
    storageAllocationCount: stats.storageAllocationCount,
    estimatedRetainedBytes: retainedBytes,
    retainedBytesAreDiagnostic: true,
  });
}

function sameWorkspaceStats(
  left: AceOpt0084PlannerSamplingWorkspaceStats,
  right: AceOpt0084PlannerSamplingWorkspaceStats,
): boolean {
  return left.candidateCapacity === right.candidateCapacity &&
    left.maskCapacity === right.maskCapacity &&
    left.storageAllocationCount === right.storageAllocationCount;
}

function requireSameSample(
  expected: AcePlannerTokenSample,
  actual: AcePlannerTokenSample,
  label: string,
): void {
  if (
    expected.tokenId !== actual.tokenId || expected.word !== actual.word ||
    expected.positiveCandidateCount !== actual.positiveCandidateCount
  ) {
    throw new Error(`${label} differs: ${JSON.stringify(actual)} != ` +
      JSON.stringify(expected));
  }
}

function requireSameCursorSample(
  expected: AcePlannerCursorSample,
  actual: AcePlannerCursorSample,
  label: string,
): void {
  requireSameSample(expected, actual, label);
  if (expected.drawIndex !== actual.drawIndex) {
    throw new Error(`${label} draw index differs`);
  }
}

function sampleReceipt(sample: AcePlannerTokenSample): Readonly<Record<string, unknown>> {
  return Object.freeze({
    tokenId: sample.tokenId,
    word: sample.word,
    positiveCandidateCount: sample.positiveCandidateCount,
  });
}

function cursorSampleReceipt(
  sample: AcePlannerCursorSample,
  cursor: AcePlannerSamplingCursor,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    ...sampleReceipt(sample),
    drawIndex: sample.drawIndex.toString(),
    drawEnd: cursor.consumed.toString(),
  });
}

function hashRows(rows: readonly Float32Array[]): string {
  const bytes = new Uint8Array(rows.reduce((total, row) => total + row.byteLength, 0));
  let offset = 0;
  for (const row of rows) {
    bytes.set(new Uint8Array(row.buffer, row.byteOffset, row.byteLength), offset);
    offset += row.byteLength;
  }
  return aceSha256Hex(bytes);
}

function hashFloat32(values: Float32Array): string {
  return aceSha256Hex(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

function hashU32(values: readonly number[]): string {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function words(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}

function rawWord(value: number): number {
  const array = Float32Array.of(value);
  return new Uint32Array(array.buffer)[0]!;
}

function validateFrozenFixtureConstants(): void {
  if (
    FIXTURE_SPECS.length !== OPT_0084_STATE_IDS.length ||
    FIXTURE_SPECS.some((spec, index) => spec.id !== OPT_0084_STATE_IDS[index]) ||
    ACCEPTED_SEMANTIC_CODE_IDS.length !== 150 ||
    ACCEPTED_SEMANTIC_CODE_IDS.some((code) =>
      !Number.isSafeInteger(code) || code < 0 ||
      code >= ACE_PLANNER_SEMANTIC_CODE_COUNT) ||
    OPT_0084_PAIR_ORDERS.length !== OPT_0084_TIMING_ROUND_COUNT ||
    OPT_0084_PAIR_ORDERS.filter((order) => order[0] === "A").length !== 8 ||
    OPT_0084_PAIR_ORDERS.filter((order) => order[0] === "B").length !== 8
  ) {
    throw new Error("OPT-0084 frozen fixture or pair schedule changed");
  }
}

function requireReceiptNumber(
  receipt: Readonly<Record<string, unknown>>,
  name: string,
): number {
  const value = Number(receipt[name]);
  if (!Number.isFinite(value)) throw new Error(`OPT-0084 receipt omitted ${name}`);
  return value;
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0084 package omitted ${name}`);
  return file;
}

function sumSafe(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total)) throw new RangeError(`${label} is unsafe`);
  }
  return total;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

async function yieldToWorker(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      ...(error.cause === undefined ? {} : { cause: String(error.cause) }),
    });
  }
  return Object.freeze({ name: typeof error, message: String(error), stack: null });
}

function errorText(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export {};
