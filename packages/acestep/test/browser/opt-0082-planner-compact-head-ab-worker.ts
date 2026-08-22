/// <reference lib="webworker" />

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
  type AcePlannerDecodeBatch,
  type AcePlannerLogitRange,
  type AcePlannerPrefillBatch,
} from "../../src/runtime/planner.js";
import {
  ACE_BROWSER_SOFTMAX_V1,
  AcePlannerSamplingCursor,
  createAcePlannerCompactFilteredLogits,
  createAcePlannerFilteredLogits,
  type AcePlannerCursorSample,
  type AcePlannerSamplingParameters,
} from "../../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../../src/runtime/seed.js";
import {
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

const MANIFEST_PATH = "/model/files-reference/manifest.json";
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
const BASE_PAIR_ORDERS = Object.freeze([
  Object.freeze(["full", "compact"] as const),
  Object.freeze(["compact", "full"] as const),
  Object.freeze(["compact", "full"] as const),
  Object.freeze(["full", "compact"] as const),
  Object.freeze(["full", "compact"] as const),
  Object.freeze(["compact", "full"] as const),
]);
const TIMING_PAIR_COUNT = BASE_PAIR_ORDERS.length * 2;
const ACCEPTED_SEMANTIC_CODE_SHA256 =
  "42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be";
const ACCEPTED_SEED = canonicalizeSeed("000000000badc0de");
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
const ACCEPTED_SEMANTIC_TOKEN_IDS = Object.freeze(
  ACCEPTED_SEMANTIC_CODE_IDS.map(
    (code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code,
  ),
);

const CASE_SPECS = Object.freeze([
  Object.freeze({
    id: "semantic-short",
    cachedTokensBeforeAppend: 268,
    cacheCapacity: 768,
    drawIndex: 125,
  }),
  Object.freeze({
    id: "semantic-middle",
    cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280,
    drawIndex: 185,
  }),
  Object.freeze({
    id: "semantic-long",
    cachedTokensBeforeAppend: 401,
    cacheCapacity: 2_048,
    drawIndex: 258,
  }),
]);

const REGULAR_RANGE: AcePlannerLogitRange = Object.freeze({
  firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
});
const EOS_RANGE: AcePlannerLogitRange = Object.freeze({
  firstTokenId: ACE_QWEN_IM_END_TOKEN_ID,
  tokenCount: 1,
});
const PAIR_ORDERS = Object.freeze([
  ...BASE_PAIR_ORDERS,
  ...BASE_PAIR_ORDERS,
]);
const SAMPLING_PARAMETERS: AcePlannerSamplingParameters = Object.freeze({
  temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
  guidanceScale: DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale,
  topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
  topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
  repetitionPenalty: 1,
});

type Arm = "full" | "compact";
type CaseSpec = (typeof CASE_SPECS)[number];

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface SemanticFixture {
  readonly spec: CaseSpec;
  readonly prefill: AcePlannerPrefillBatch;
  readonly decode: AcePlannerDecodeBatch;
  readonly seenTokenIds: readonly number[];
  readonly nextTeacherTokenIndex: number;
}

interface SampleReceipt {
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
  readonly drawIndex: string;
  readonly drawEnd: string;
}

interface ArmExecution {
  readonly arm: Arm;
  readonly cachedTokensBeforeAppend: number;
  readonly inputTokenId: number;
  readonly oppositeArm: Arm;
  readonly sample: SampleReceipt;
  readonly rawExact: ExactComparisonReceipt;
  readonly filteredExact: ExactComparisonReceipt;
  readonly modelWallMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly completeWallMilliseconds: number;
  readonly oppositeHeadReplayWallMilliseconds: number;
  readonly oppositeSamplingWallMilliseconds: number;
}

interface CaseExecution {
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly compactWins: number;
  readonly fullCompleteWallMilliseconds: readonly number[];
  readonly compactCompleteWallMilliseconds: readonly number[];
  readonly compactMedianBelowFull: boolean;
}

interface ExactComparisonReceipt {
  readonly rowCount: number;
  readonly comparedElements: number;
  readonly mismatchCount: 0;
  readonly nanCount: 0;
}

interface IncomingMessage {
  readonly type: "run" | "cancel";
}

let lifecycle: "idle" | "running" | "settled" = "idle";
let activeAbortController: AbortController | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "cancel") {
    activeAbortController?.abort(new DOMException("OPT-0082 cancelled", "AbortError"));
    return;
  }
  if (lifecycle !== "idle") return;
  lifecycle = "running";
  void runHarness().then(
    (result) => {
      lifecycle = "settled";
      self.postMessage({ type: "passed", result });
    },
    (error) => {
      lifecycle = "settled";
      self.postMessage({ type: "failed", error: errorValue(error) });
    },
  );
});

async function runHarness(): Promise<Readonly<Record<string, unknown>>> {
  const abortController = new AbortController();
  activeAbortController = abortController;
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  const startedAtEpochMilliseconds = Date.now();
  try {
    validatePinnedFixture();
    postProgress("authenticating and acquiring current reference-BF16 planner files");
    const acquisitionStarted = performance.now();
    const preparedPackage = await preparePackage(abortController.signal);
    const acquisitionWallMilliseconds = performance.now() - acquisitionStarted;

    postProgress("requesting the production reference-BF16 WebGPU profile");
    context = await requestAceWebGpuDevice({
      modelProfile: "reference-bf16",
      schedulingProfile: "cooperative",
      signal: abortController.signal,
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
        signal: abortController.signal,
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
      phase.phases.length !== 1 ||
      phase.phases[0] !== "planner" ||
      phase.residentBytes !== PLANNER_RESIDENT_BYTES
    ) {
      throw new Error("OPT-0082 loaded planner identity changed");
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
    const fixtures = createSemanticFixtures(tokenizerLoaded.tokenizer);
    executor = AcePlannerGpuExecutor.create({
      device: context.device,
      modelProfile: "reference-bf16",
      ownedPlannerWeights: phase,
      signal: abortController.signal,
    });
    phase = undefined;

    const cases: Readonly<Record<string, unknown>>[] = [];
    const allFullCompleteWallMilliseconds: number[] = [];
    const allCompactCompleteWallMilliseconds: number[] = [];
    let aggregateCompactWins = 0;
    let everyCachePositionCompactMedianBelowFull = true;
    for (let index = 0; index < fixtures.length; index += 1) {
      const fixture = fixtures[index]!;
      postProgress(
        `${fixture.spec.id}: ordinary full-vocabulary prefill and decode`,
      );
      const executed = await runCase(
        executor,
        fixture,
        index,
        abortController.signal,
      );
      cases.push(executed.receipt);
      aggregateCompactWins += executed.compactWins;
      allFullCompleteWallMilliseconds.push(
        ...executed.fullCompleteWallMilliseconds,
      );
      allCompactCompleteWallMilliseconds.push(
        ...executed.compactCompleteWallMilliseconds,
      );
      everyCachePositionCompactMedianBelowFull &&=
        executed.compactMedianBelowFull;
    }
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0082 observed a WebGPU runtime error before cleanup");
    }

    const executionProfile = context.capabilities.executionProfile.id;
    const adapterInfo = context.capabilities.adapterInfo;
    const aggregateFullMedian = median(allFullCompleteWallMilliseconds);
    const aggregateCompactMedian = median(allCompactCompleteWallMilliseconds);
    postProgress("destroying the planner owner and WebGPU device");
    await executor.destroy();
    executor = undefined;
    context.destroy();
    context = undefined;
    activeAbortController = undefined;
    return Object.freeze({
      kind: "ace-opt-0082-reference-bf16-compact-semantic-head-ab",
      status: "passed",
      timingEvidenceScope: "developer-ab-without-attached-thermal-trace",
      startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Date.now(),
      package: preparedPackage.receipt,
      tokenizer: tokenizerLoaded.assetIdentity,
      execution: Object.freeze({
        modelProfile: "reference-bf16",
        executionProfile,
        schedulingProfile: "cooperative",
        adapterInfo,
        onePlannerExecutor: true,
        ordinaryDecodeBeforeReplay: true,
        timedArmsUseSequentialActualDecodeCalls: true,
        timedCompleteWallIncludesSelectedSampler: true,
        oppositeHeadReplayExcludedFromTimedCompleteWall: true,
        replaySignature:
          "replayTiedHeadForOpt0082(logitRange?: { firstTokenId; tokenCount })",
        setupExcludedFromTiming: true,
        timingPairOrders: PAIR_ORDERS.map((order) => order.join("-")),
        acquisitionWallMilliseconds,
        uploadWallMilliseconds,
      }),
      fixture: Object.freeze({
        semanticCodeCount: ACCEPTED_SEMANTIC_CODE_IDS.length,
        semanticCodeU32LeSha256: ACCEPTED_SEMANTIC_CODE_SHA256,
        seed: ACCEPTED_SEED,
        cases: CASE_SPECS,
      }),
      aggregateTiming: Object.freeze({
        pairCount: allFullCompleteWallMilliseconds.length,
        compactWins: aggregateCompactWins,
        fullCompleteWallMilliseconds: summarize(
          allFullCompleteWallMilliseconds,
          CASE_SPECS.length * TIMING_PAIR_COUNT,
        ),
        compactCompleteWallMilliseconds: summarize(
          allCompactCompleteWallMilliseconds,
          CASE_SPECS.length * TIMING_PAIR_COUNT,
        ),
        medianCompleteSpeedup: aggregateFullMedian / aggregateCompactMedian,
        medianCompleteSavingMilliseconds:
          aggregateFullMedian - aggregateCompactMedian,
        projected900TokenSavingSeconds:
          (aggregateFullMedian - aggregateCompactMedian) * 900 / 1_000,
        everyCachePositionCompactMedianBelowFull,
      }),
      cases: Object.freeze(cases),
      runtimeEvents: Object.freeze([...runtimeEvents]),
      cleanup: Object.freeze({
        executorDestroyed: true,
        deviceDestroyed: true,
      }),
    });
  } catch (error) {
    activeAbortController = undefined;
    let cleanupError: unknown;
    try {
      if (executor !== undefined) await executor.destroy(error);
      else phase?.destroy();
    } catch (caught) {
      cleanupError = caught;
    } finally {
      context?.destroy();
    }
    if (cleanupError !== undefined) {
      throw new Error(
        `OPT-0082 failed (${errorText(error)}); cleanup also failed: ` +
          errorText(cleanupError),
        { cause: error },
      );
    }
    throw error;
  }
}

async function runCase(
  executor: AcePlannerGpuExecutor,
  fixture: SemanticFixture,
  caseIndex: number,
  signal: AbortSignal,
): Promise<CaseExecution> {
  signal.throwIfAborted();
  const prefillStarted = performance.now();
  await executor.prefill(fixture.prefill);
  const prefillWallMilliseconds = performance.now() - prefillStarted;
  signal.throwIfAborted();
  const decodeStarted = performance.now();
  const ordinaryDecodeRows = requireFloat32Rows(
    await executor.decode(fixture.decode),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} ordinary decode`,
  );
  const ordinaryDecodeWallMilliseconds = performance.now() - decodeStarted;

  postProgress(`${fixture.spec.id}: exact full/compact replay and sampler checks`);
  const fullReplayRows = requireFloat32Rows(
    await executor.replayTiedHeadForOpt0082(),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} full replay`,
  );
  const ordinaryReplayExact = requireExactSlice(
    ordinaryDecodeRows,
    fullReplayRows,
    0,
    `${fixture.spec.id} ordinary decode/full replay`,
  );
  const compactRows = requireFloat32Rows(
    await executor.replayTiedHeadForOpt0082(REGULAR_RANGE),
    REGULAR_RANGE.tokenCount,
    `${fixture.spec.id} compact replay`,
  );
  const compactExact = requireExactSlice(
    fullReplayRows,
    compactRows,
    REGULAR_RANGE.firstTokenId,
    `${fixture.spec.id} regular compact replay`,
  );
  const regularSampling = compareSamplerPaths(
    fullReplayRows,
    compactRows,
    REGULAR_RANGE,
    fixture.seenTokenIds,
    fixture.spec.drawIndex,
    false,
    `${fixture.spec.id} regular sampler`,
  );
  const eosRows = requireFloat32Rows(
    await executor.replayTiedHeadForOpt0082(EOS_RANGE),
    EOS_RANGE.tokenCount,
    `${fixture.spec.id} EOS replay`,
  );
  const eosExact = requireExactSlice(
    fullReplayRows,
    eosRows,
    EOS_RANGE.firstTokenId,
    `${fixture.spec.id} EOS compact replay`,
  );
  const eosSampling = compareSamplerPaths(
    fullReplayRows,
    eosRows,
    EOS_RANGE,
    fixture.seenTokenIds,
    fixture.spec.drawIndex,
    true,
    `${fixture.spec.id} EOS sampler`,
  );
  if (
    eosSampling.full.tokenId !== ACE_QWEN_IM_END_TOKEN_ID ||
    eosSampling.compact.tokenId !== ACE_QWEN_IM_END_TOKEN_ID
  ) {
    throw new Error(`${fixture.spec.id} forced EOS sampler did not select EOS`);
  }

  const pairs: Readonly<Record<string, unknown>>[] = [];
  const fullComplete: number[] = [];
  const compactComplete: number[] = [];
  const fullModel: number[] = [];
  const compactModel: number[] = [];
  const fullSampling: number[] = [];
  const compactSampling: number[] = [];
  const fullOppositeHeadReplay: number[] = [];
  const compactOppositeHeadReplay: number[] = [];
  const fullOppositeSampling: number[] = [];
  const compactOppositeSampling: number[] = [];
  const seenTokenIds = [...fixture.seenTokenIds];
  let cachedTokensBeforeAppend = fixture.spec.cachedTokensBeforeAppend + 1;
  let nextTeacherTokenIndex = fixture.nextTeacherTokenIndex;
  let nextDrawIndex = fixture.spec.drawIndex + 1;
  let compactWins = 0;
  for (let pairIndex = 0; pairIndex < PAIR_ORDERS.length; pairIndex += 1) {
    signal.throwIfAborted();
    const order = PAIR_ORDERS[pairIndex]!;
    const runs = new Map<Arm, ArmExecution>();
    for (const arm of order) {
      const inputTokenId = ACCEPTED_SEMANTIC_TOKEN_IDS[
        nextTeacherTokenIndex % ACCEPTED_SEMANTIC_TOKEN_IDS.length
      ]!;
      const batch = createDecodeBatch(
        fixture.spec.cacheCapacity,
        cachedTokensBeforeAppend,
        inputTokenId,
      );
      const seenForStep = Object.freeze([...seenTokenIds, inputTokenId]);
      const executed = await executeTimedArm(
        executor,
        arm,
        fixture.spec.id,
        batch,
        seenForStep,
        nextDrawIndex,
      );
      runs.set(arm, executed);
      seenTokenIds.push(inputTokenId);
      cachedTokensBeforeAppend += 1;
      nextTeacherTokenIndex += 1;
      nextDrawIndex += 1;
    }
    const full = requireMapValue(runs, "full");
    const compact = requireMapValue(runs, "compact");
    fullComplete.push(full.completeWallMilliseconds);
    compactComplete.push(compact.completeWallMilliseconds);
    fullModel.push(full.modelWallMilliseconds);
    compactModel.push(compact.modelWallMilliseconds);
    fullSampling.push(full.samplingWallMilliseconds);
    compactSampling.push(compact.samplingWallMilliseconds);
    fullOppositeHeadReplay.push(full.oppositeHeadReplayWallMilliseconds);
    compactOppositeHeadReplay.push(compact.oppositeHeadReplayWallMilliseconds);
    fullOppositeSampling.push(full.oppositeSamplingWallMilliseconds);
    compactOppositeSampling.push(compact.oppositeSamplingWallMilliseconds);
    if (compact.completeWallMilliseconds < full.completeWallMilliseconds) {
      compactWins += 1;
    }
    pairs.push(Object.freeze({
      pairIndex,
      order: order.join("-"),
      full: publicArmExecution(full),
      compact: publicArmExecution(compact),
      compactCompleteSavingMilliseconds:
        full.completeWallMilliseconds - compact.completeWallMilliseconds,
    }));
    postProgress(
      `${fixture.spec.id}: timed pair ${pairIndex + 1}/${PAIR_ORDERS.length}`,
    );
  }

  const fullCompleteMedian = median(fullComplete);
  const compactCompleteMedian = median(compactComplete);
  const receipt = Object.freeze({
    id: fixture.spec.id,
    cachedTokensBeforeAppend: fixture.spec.cachedTokensBeforeAppend,
    finalCachedTokens: cachedTokensBeforeAppend,
    cacheCapacity: fixture.spec.cacheCapacity,
    drawIndex: fixture.spec.drawIndex,
    setup: Object.freeze({
      prefillWallMilliseconds,
      ordinaryDecodeWallMilliseconds,
      ordinaryDecodeFullRowSha256: rowDigests(ordinaryDecodeRows),
    }),
    correctness: Object.freeze({
      ordinaryDecodeVsFullReplay: ordinaryReplayExact,
      regularCompactVsFullSlice: compactExact,
      regularFilteredVsFullSlice: regularSampling.filteredExact,
      regularFullSample: regularSampling.full,
      regularCompactSample: regularSampling.compact,
      eosCompactVsFullSlice: eosExact,
      eosFilteredVsFullSlice: eosSampling.filteredExact,
      eosFullSample: eosSampling.full,
      eosCompactSample: eosSampling.compact,
      fullReplayRowSha256: rowDigests(fullReplayRows),
      regularCompactRowSha256: rowDigests(compactRows),
      eosCompactRowSha256: rowDigests(eosRows),
    }),
    timing: Object.freeze({
      sampleCountPerArm: TIMING_PAIR_COUNT,
      timingAuthority:
        "performance-now-around-sequential-executor-decode-plus-selected-sampler",
      modelWallDefinition:
        "executor-decode-including-transformer-selected-head-readback",
      adjacentPositionPairing: true,
      oppositeHeadReplayExcludedFromCompleteTokenWall: true,
      compactWins,
      full: Object.freeze({
        completeWallMilliseconds: summarize(fullComplete, TIMING_PAIR_COUNT),
        modelWallMilliseconds: summarize(fullModel, TIMING_PAIR_COUNT),
        samplingWallMilliseconds: summarize(fullSampling, TIMING_PAIR_COUNT),
        oppositeHeadReplayWallMilliseconds: summarize(
          fullOppositeHeadReplay,
          TIMING_PAIR_COUNT,
        ),
        oppositeSamplingWallMilliseconds: summarize(
          fullOppositeSampling,
          TIMING_PAIR_COUNT,
        ),
      }),
      compact: Object.freeze({
        completeWallMilliseconds: summarize(compactComplete, TIMING_PAIR_COUNT),
        modelWallMilliseconds: summarize(compactModel, TIMING_PAIR_COUNT),
        samplingWallMilliseconds: summarize(compactSampling, TIMING_PAIR_COUNT),
        oppositeHeadReplayWallMilliseconds: summarize(
          compactOppositeHeadReplay,
          TIMING_PAIR_COUNT,
        ),
        oppositeSamplingWallMilliseconds: summarize(
          compactOppositeSampling,
          TIMING_PAIR_COUNT,
        ),
      }),
      medianCompleteSpeedup: fullCompleteMedian / compactCompleteMedian,
      medianCompleteSavingMilliseconds:
        fullCompleteMedian - compactCompleteMedian,
      pairs: Object.freeze(pairs),
    }),
    caseIndex,
  });
  return Object.freeze({
    receipt,
    compactWins,
    fullCompleteWallMilliseconds: Object.freeze(fullComplete),
    compactCompleteWallMilliseconds: Object.freeze(compactComplete),
    compactMedianBelowFull: compactCompleteMedian < fullCompleteMedian,
  });
}

async function executeTimedArm(
  executor: AcePlannerGpuExecutor,
  arm: Arm,
  caseId: string,
  batch: AcePlannerDecodeBatch,
  seenTokenIds: readonly number[],
  drawIndex: number,
): Promise<ArmExecution> {
  const completeStarted = performance.now();
  const returnedRows = arm === "compact"
    ? await executor.decode(batch, REGULAR_RANGE)
    : await executor.decode(batch);
  const primaryRows = requireFloat32Rows(
    returnedRows,
    arm === "compact"
      ? REGULAR_RANGE.tokenCount
      : ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${caseId} timed ${arm} decode`,
  );
  const modelEnded = performance.now();
  const sample = arm === "full"
    ? sampleFull(primaryRows, seenTokenIds, drawIndex, false)
    : sampleCompact(
        primaryRows,
        REGULAR_RANGE,
        seenTokenIds,
        drawIndex,
      );
  const samplingEnded = performance.now();
  const oppositeArm: Arm = arm === "full" ? "compact" : "full";
  const oppositeReplayStarted = performance.now();
  const replayedRows = oppositeArm === "compact"
    ? await executor.replayTiedHeadForOpt0082(REGULAR_RANGE)
    : await executor.replayTiedHeadForOpt0082();
  const oppositeRows = requireFloat32Rows(
    replayedRows,
    oppositeArm === "compact"
      ? REGULAR_RANGE.tokenCount
      : ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${caseId} opposite ${oppositeArm} replay`,
  );
  const oppositeReplayEnded = performance.now();
  const oppositeSample = oppositeArm === "full"
    ? sampleFull(oppositeRows, seenTokenIds, drawIndex, false)
    : sampleCompact(oppositeRows, REGULAR_RANGE, seenTokenIds, drawIndex);
  const oppositeSamplingEnded = performance.now();
  requireSameSample(sample, oppositeSample, `${caseId} timed ${arm} sampler`);
  const fullRows = arm === "full" ? primaryRows : oppositeRows;
  const compactRows = arm === "compact" ? primaryRows : oppositeRows;
  const rawExact = requireExactSlice(
    fullRows,
    compactRows,
    REGULAR_RANGE.firstTokenId,
    `${caseId} timed ${arm} opposite replay`,
  );
  const filteredExact = compareFilteredPaths(
    fullRows,
    compactRows,
    REGULAR_RANGE,
    seenTokenIds,
    false,
    `${caseId} timed ${arm} sampler`,
  );
  return Object.freeze({
    arm,
    cachedTokensBeforeAppend: batch.cachedTokensBeforeAppend,
    inputTokenId: batch.inputIds[0]!,
    oppositeArm,
    sample,
    rawExact,
    filteredExact,
    modelWallMilliseconds: modelEnded - completeStarted,
    samplingWallMilliseconds: samplingEnded - modelEnded,
    completeWallMilliseconds: samplingEnded - completeStarted,
    oppositeHeadReplayWallMilliseconds:
      oppositeReplayEnded - oppositeReplayStarted,
    oppositeSamplingWallMilliseconds:
      oppositeSamplingEnded - oppositeReplayEnded,
  });
}

function compareSamplerPaths(
  fullRows: readonly Float32Array[],
  compactRows: readonly Float32Array[],
  range: AcePlannerLogitRange,
  seenTokenIds: readonly number[],
  drawIndex: number,
  forcingEos: boolean,
  label: string,
): Readonly<{
  readonly full: SampleReceipt;
  readonly compact: SampleReceipt;
  readonly filteredExact: ExactComparisonReceipt;
}> {
  const full = sampleFull(fullRows, seenTokenIds, drawIndex, forcingEos);
  const compact = sampleCompact(compactRows, range, seenTokenIds, drawIndex);
  requireSameSample(full, compact, label);
  return Object.freeze({
    full,
    compact,
    filteredExact: compareFilteredPaths(
      fullRows,
      compactRows,
      range,
      seenTokenIds,
      forcingEos,
      label,
    ),
  });
}

function compareFilteredPaths(
  fullRows: readonly Float32Array[],
  compactRows: readonly Float32Array[],
  range: AcePlannerLogitRange,
  seenTokenIds: readonly number[],
  forcingEos: boolean,
  label: string,
): ExactComparisonReceipt {
  const fullFiltered = [createAcePlannerFilteredLogits({
    conditionalLogits: fullRows[0]!,
    unconditionalLogits: fullRows[1]!,
    seenTokenIds,
    preCfgAllowedTokens: {
      kind: "range",
      firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
      additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
    },
    allowedTokens: forcingEos
      ? { kind: "ids", tokenIds: [ACE_QWEN_IM_END_TOKEN_ID] }
      : {
          kind: "range",
          firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
          tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
        },
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
  })];
  // Filtering is one CFG result rather than one output per model row. Keep a
  // one-row shape so the same raw-U32 slice comparator owns both checks.
  const compactFiltered = [createAcePlannerCompactFilteredLogits({
    firstTokenId: range.firstTokenId,
    vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    conditionalLogits: compactRows[0]!,
    unconditionalLogits: compactRows[1]!,
    seenTokenIds,
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
  })];
  return requireExactSlice(
    [fullFiltered[0]!],
    compactFiltered,
    range.firstTokenId,
    `${label} filtered logits`,
  );
}

function sampleFull(
  logits: readonly Float32Array[],
  seenTokenIds: readonly number[],
  drawIndex: number,
  forcingEos: boolean,
): SampleReceipt {
  const cursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const sample = cursor.sample({
    conditionalLogits: logits[0]!,
    unconditionalLogits: logits[1]!,
    seenTokenIds,
    preCfgAllowedTokens: {
      kind: "range",
      firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
      additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
    },
    allowedTokens: forcingEos
      ? { kind: "ids", tokenIds: [ACE_QWEN_IM_END_TOKEN_ID] }
      : {
          kind: "range",
          firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
          tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
        },
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
  });
  return sampleReceipt(sample, cursor);
}

function sampleCompact(
  logits: readonly Float32Array[],
  range: AcePlannerLogitRange,
  seenTokenIds: readonly number[],
  drawIndex: number,
): SampleReceipt {
  const cursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const sample = cursor.sampleCompact({
    firstTokenId: range.firstTokenId,
    vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    conditionalLogits: logits[0]!,
    unconditionalLogits: logits[1]!,
    seenTokenIds,
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
  });
  return sampleReceipt(sample, cursor);
}

function sampleReceipt(
  sample: AcePlannerCursorSample,
  cursor: AcePlannerSamplingCursor,
): SampleReceipt {
  return Object.freeze({
    tokenId: sample.tokenId,
    word: sample.word,
    positiveCandidateCount: sample.positiveCandidateCount,
    drawIndex: sample.drawIndex.toString(),
    drawEnd: cursor.consumed.toString(),
  });
}

function createSemanticFixtures(
  tokenizer: AceQwenBpeTokenizer,
): readonly SemanticFixture[] {
  const prompts = createAcePlannerCodePrompts(
    ACCEPTED_RESOLVED_CAPTION,
    ACCEPTED_LYRICS,
    ACCEPTED_COT_TEXT,
  );
  const baseRows = Object.freeze([
    tokenizer.encode(prompts.conditional),
    tokenizer.encode(prompts.unconditional),
  ]);
  if (baseRows[0]!.length !== 253 || baseRows[1]!.length !== 33) {
    throw new Error("OPT-0082 semantic prompt tokenization changed");
  }
  return Object.freeze(CASE_SPECS.map((spec) => {
    const prefill = createPaddedPrefill(
      baseRows,
      ACCEPTED_SEMANTIC_TOKEN_IDS,
      spec.cachedTokensBeforeAppend,
      spec.cacheCapacity,
    );
    const baseWidth = Math.max(...baseRows.map((row) => row.length));
    const continuationIndex = spec.cachedTokensBeforeAppend - baseWidth;
    const decodeTokenId = ACCEPTED_SEMANTIC_TOKEN_IDS[continuationIndex];
    if (decodeTokenId === undefined) {
      throw new Error(`OPT-0082 ${spec.id} leaves the accepted trajectory`);
    }
    const decode = createDecodeBatch(
      spec.cacheCapacity,
      spec.cachedTokensBeforeAppend,
      decodeTokenId,
    );
    return Object.freeze({
      spec,
      prefill,
      decode,
      seenTokenIds: Object.freeze([
        ...baseRows[0]!,
        ...ACCEPTED_SEMANTIC_TOKEN_IDS.slice(0, continuationIndex + 1),
      ]),
      nextTeacherTokenIndex: continuationIndex + 1,
    });
  }));
}

function createPaddedPrefill(
  baseRows: readonly (readonly number[])[],
  continuation: readonly number[],
  tokens: number,
  cacheCapacity: number,
): AcePlannerPrefillBatch {
  const rows = 2 as const;
  const baseWidth = Math.max(...baseRows.map((row) => row.length));
  if (
    baseRows.length !== rows ||
    baseWidth >= tokens ||
    tokens >= cacheCapacity ||
    tokens - baseWidth >= continuation.length
  ) {
    throw new Error("OPT-0082 semantic prefill geometry is invalid");
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
    rowStartPositions: [0, 0],
    validKeyLengths: [tokens, tokens],
    sourceValidity: [...keyValidity],
  });
  return Object.freeze({
    kind: "prefill",
    rows,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0,
    unconditionalRow: 1,
  });
}

function createDecodeBatch(
  cacheCapacity: number,
  cachedTokensBeforeAppend: number,
  decodeTokenId: number,
): AcePlannerDecodeBatch {
  const rows = 2 as const;
  const inputIds = new Uint32Array([decodeTokenId, decodeTokenId]);
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens: 1,
    cacheCapacity,
    rowStartPositions: [
      cachedTokensBeforeAppend,
      cachedTokensBeforeAppend,
    ],
    validKeyLengths: [
      cachedTokensBeforeAppend + 1,
      cachedTokensBeforeAppend + 1,
    ],
    sourceValidity: [1, 1],
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
    unconditionalRow: 1,
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
    throw new Error("OPT-0082 current reference manifest identity changed");
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
    throw new Error("OPT-0082 bounded planner acquisition changed");
  }
  return Object.freeze({
    manifest: loaded.manifest,
    manifestUrl: loaded.manifestUrl,
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
    "OPT-0082 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0082 planner tensor bytes",
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
    throw new Error("OPT-0082 reference planner inventory changed");
  }
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    weightFileCount: weightFiles.length,
    residentBytes,
  });
}

function requireFloat32Rows(
  rows: readonly ArrayLike<number>[],
  expectedColumns: number,
  label: string,
): readonly Float32Array[] {
  if (rows.length !== 2) throw new Error(`${label} returned ${rows.length} rows`);
  return Object.freeze(rows.map((row, index) => {
    if (!(row instanceof Float32Array) || row.length !== expectedColumns) {
      throw new Error(`${label} row ${index} has the wrong storage or extent`);
    }
    return row;
  }));
}

function requireExactSlice(
  fullRows: readonly Float32Array[],
  compactRows: readonly Float32Array[],
  firstTokenId: number,
  label: string,
): ExactComparisonReceipt {
  if (fullRows.length !== compactRows.length) {
    throw new Error(`${label} row count differs`);
  }
  let comparedElements = 0;
  let nanCount = 0;
  for (let row = 0; row < fullRows.length; row += 1) {
    const full = fullRows[row]!;
    const compact = compactRows[row]!;
    if (firstTokenId + compact.length > full.length) {
      throw new Error(`${label} compact extent exceeds the full row`);
    }
    const fullBits = new Uint32Array(
      full.buffer,
      full.byteOffset,
      full.length,
    );
    const compactBits = new Uint32Array(
      compact.buffer,
      compact.byteOffset,
      compact.length,
    );
    for (let column = 0; column < compact.length; column += 1) {
      if (Number.isNaN(compact[column])) nanCount += 1;
      const expected = fullBits[firstTokenId + column]!;
      const actual = compactBits[column]!;
      if (actual !== expected) {
        throw new Error(
          `${label} raw-F32 mismatch at row ${row}, local column ${column}, ` +
            `global token ${firstTokenId + column}: ` +
            `0x${actual.toString(16).padStart(8, "0")} != ` +
            `0x${expected.toString(16).padStart(8, "0")}`,
        );
      }
      comparedElements += 1;
    }
  }
  if (nanCount !== 0) throw new Error(`${label} contains ${nanCount} NaNs`);
  return Object.freeze({
    rowCount: compactRows.length,
    comparedElements,
    mismatchCount: 0,
    nanCount: 0,
  });
}

function requireSameSample(
  expected: SampleReceipt,
  actual: SampleReceipt,
  label: string,
): void {
  if (
    expected.tokenId !== actual.tokenId ||
    expected.word !== actual.word ||
    expected.positiveCandidateCount !== actual.positiveCandidateCount ||
    expected.drawIndex !== actual.drawIndex ||
    expected.drawEnd !== actual.drawEnd
  ) {
    throw new Error(
      `${label} differs: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`,
    );
  }
}

function publicArmExecution(
  execution: ArmExecution,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    arm: execution.arm,
    cachedTokensBeforeAppend: execution.cachedTokensBeforeAppend,
    inputTokenId: execution.inputTokenId,
    oppositeArm: execution.oppositeArm,
    modelWallMilliseconds: execution.modelWallMilliseconds,
    samplingWallMilliseconds: execution.samplingWallMilliseconds,
    completeWallMilliseconds: execution.completeWallMilliseconds,
    oppositeHeadReplayWallMilliseconds:
      execution.oppositeHeadReplayWallMilliseconds,
    oppositeSamplingWallMilliseconds:
      execution.oppositeSamplingWallMilliseconds,
    sample: execution.sample,
    rawExact: execution.rawExact,
    filteredExact: execution.filteredExact,
  });
}

function rowDigests(rows: readonly Float32Array[]): readonly string[] {
  return Object.freeze(rows.map((row) => aceSha256Hex(new Uint8Array(
    row.buffer,
    row.byteOffset,
    row.byteLength,
  ))));
}

function validatePinnedFixture(): void {
  if (
    ACCEPTED_SEMANTIC_CODE_IDS.length !== 150 ||
    ACCEPTED_SEMANTIC_CODE_IDS.some(
      (code) => !Number.isSafeInteger(code) ||
        code < 0 || code >= ACE_PLANNER_SEMANTIC_CODE_COUNT,
    ) ||
    sha256U32Le(ACCEPTED_SEMANTIC_CODE_IDS) !==
      ACCEPTED_SEMANTIC_CODE_SHA256
  ) {
    throw new Error("OPT-0082 pinned semantic fixture identity changed");
  }
  if (
    PAIR_ORDERS.length !== TIMING_PAIR_COUNT ||
    PAIR_ORDERS.filter((order) => order[0] === "full").length !== 6 ||
    PAIR_ORDERS.filter((order) => order[0] === "compact").length !== 6
  ) {
    throw new Error("OPT-0082 A/B timing order is not balanced");
  }
}

function sha256U32Le(values: readonly number[]): string {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function summarize(
  values: readonly number[],
  expectedCount: number,
): Readonly<Record<string, number>> {
  if (
    values.length !== expectedCount ||
    values.some((value) => !Number.isFinite(value) || value < 0)
  ) {
    throw new Error("OPT-0082 timing sample set is invalid");
  }
  return Object.freeze({
    minimum: Math.min(...values),
    median: median(values),
    maximum: Math.max(...values),
  });
}

function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("OPT-0082 cannot summarize no values");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function requireMapValue<T>(map: ReadonlyMap<Arm, T>, arm: Arm): T {
  const value = map.get(arm);
  if (value === undefined) throw new Error(`OPT-0082 omitted ${arm} arm`);
  return value;
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0082 package omitted ${name}`);
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

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      ...(error.cause === undefined ? {} : { cause: errorText(error.cause) }),
    });
  }
  return Object.freeze({ name: "Error", message: String(error), stack: null });
}

function errorText(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
