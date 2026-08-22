/// <reference lib="webworker" />

import plannerSamplingSource from "../../src/runtime/planner-sampling.ts?raw";
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
  AceOpt0084PlannerSamplingWorkspace,
  AcePlannerSamplingCursor,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerCompactFilteredLogits,
  createAcePlannerFilteredLogits,
  sampleAcePlannerCompactTokenOpt0084,
  sampleAcePlannerToken,
  sampleAcePlannerTokenOpt0084,
  type AceOpt0084PlannerSamplingWorkspaceStats,
  type AcePlannerCursorSample,
  type AcePlannerSamplingParameters,
  type AcePlannerTokenSample,
} from "../../src/runtime/planner-sampling.js";
import { aceRandomWord, canonicalizeSeed } from "../../src/runtime/seed.js";
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
import {
  OPT_0084_CANDIDATE_SEAM_COMMIT,
  OPT_0084_COMPOUND_ARM_ORDERS,
  OPT_0084_COMPOUND_POSITION_IDS,
  OPT_0084_COMPOUND_RECEIPT_SCHEMA,
  OPT_0084_COMPOUND_TIMING_ROUND_COUNT,
  OPT_0084_EXPERIMENT_ID,
  OPT_0084_THERMAL_COMMAND,
  OPT_0084_THERMAL_POLL_MILLISECONDS,
  OPT_0084_THERMAL_SOURCE,
  createOpt0084CompoundAbortingLogitRow,
  evaluateOpt0084CompoundTiming,
  validateOpt0084RunIdentity,
  type Opt0084CompoundArm,
  type Opt0084CompoundTimingSample,
  type Opt0084RunIdentity,
  type Opt0084ThermalLaunch,
} from "./opt-0084-planner-compact-head-fused-sampling-contract.js";

const MANIFEST_PATH = "/model/files-reference/manifest.json";
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
    id: "semantic-early" as const,
    cachedTokensBeforeAppend: 268,
    cacheCapacity: 768,
    drawIndex: 125,
  }),
  Object.freeze({
    id: "semantic-middle" as const,
    cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280,
    drawIndex: 185,
  }),
  Object.freeze({
    id: "semantic-late" as const,
    cachedTokensBeforeAppend: 401,
    cacheCapacity: 2_048,
    drawIndex: 258,
  }),
]);

const REGULAR_RANGE: AcePlannerLogitRange = Object.freeze({
  firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
});
const SAMPLING_PARAMETERS: AcePlannerSamplingParameters = Object.freeze({
  temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
  guidanceScale: DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale,
  topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
  topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
  repetitionPenalty: 1,
});

type Arm = Opt0084CompoundArm;
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

interface ExactComparisonReceipt {
  readonly rowCount: number;
  readonly comparedElements: number;
  readonly mismatchCount: 0;
  readonly nanCount: 0;
}

interface FixtureRuntime extends SemanticFixture {
  readonly workspace: AceOpt0084PlannerSamplingWorkspace;
  readonly workspaceStatsAfterWarmup: AceOpt0084PlannerSamplingWorkspaceStats;
  readonly preparationCorrectness: Readonly<Record<string, unknown>>;
}

interface PreparedSession {
  readonly runIdentity: Opt0084RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly executor: AcePlannerGpuExecutor;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly fixtures: readonly FixtureRuntime[];
  readonly firstPositionRecurringWarmup:
    Readonly<Record<string, unknown>>;
  readonly packageReceipt: Readonly<Record<string, unknown>>;
  readonly tokenizerReceipt: Readonly<Record<string, unknown>>;
  readonly preparationReceipt: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  cleanup(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

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

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let activeAbortController: AbortController | undefined;
let activePrepared: PreparedSession | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    activeAbortController?.abort(new DOMException(
      "OPT-0084 compound cancellation requested",
      "AbortError",
    ));
    if (lifecycle === "ready" && activePrepared !== undefined) {
      void failAndCleanup(
        "cancelled-ready",
        new DOMException("OPT-0084 compound cancelled while ready", "AbortError"),
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
    }, (error) => void failAndCleanup("preparation", error));
    return;
  }
  if (message.type === "run" && lifecycle === "ready" &&
    activePrepared !== undefined) {
    lifecycle = "running";
    const prepared = activePrepared;
    void runTiming(prepared, message.thermalLaunch).then(
      (evidence) => {
        lifecycle = "settled";
        activePrepared = undefined;
        activeAbortController = undefined;
        self.postMessage({ type: "measurement-complete", evidence });
      },
      (error) => void failAndCleanup("measurement", error),
    );
    return;
  }
  if (message.type === "run") {
    void failAndCleanup("protocol", new Error(
      `OPT-0084 compound run received while ${lifecycle}`,
    ));
  }
});

async function failAndCleanup(phaseName: string, error: unknown): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  const prepared = activePrepared;
  activePrepared = undefined;
  activeAbortController = undefined;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let cleanupError: unknown;
  if (prepared !== undefined) {
    try {
      cleanup = await prepared.cleanup(error);
    } catch (caught) {
      cleanupError = caught;
    }
  }
  self.postMessage({
    type: "failed",
    phase: phaseName,
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
    validatePinnedFixture();
    const samplingSourceSha256 = aceSha256Hex(new TextEncoder().encode(
      plannerSamplingSource,
    ));
    if (samplingSourceSha256 !== EXPECTED_PLANNER_SAMPLING_SOURCE_SHA256) {
      throw new Error("OPT-0084 compound sampler source changed from its seam");
    }
    postProgress("authenticating and acquiring current reference-BF16 planner files");
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
    if (phase.phases.length !== 1 || phase.phases[0] !== "planner" ||
      phase.residentBytes !== PLANNER_RESIDENT_BYTES) {
      throw new Error("OPT-0084 compound loaded planner identity changed");
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
    const fixturePlans = createSemanticFixtures(tokenizerLoaded.tokenizer);
    executor = AcePlannerGpuExecutor.create({
      device: context.device,
      modelProfile: "reference-bf16",
      ownedPlannerWeights: phase,
      signal,
    });
    phase = undefined;

    const fixtures: FixtureRuntime[] = [];
    for (let index = 0; index < fixturePlans.length; index += 1) {
      signal.throwIfAborted();
      const fixture = fixturePlans[index]!;
      postProgress(
        `${fixture.spec.id}: actual full/compact same-state correctness and warmup`,
      );
      const workspace = new AceOpt0084PlannerSamplingWorkspace();
      const sameStateProof = await runSameStateDispatchAndCacheProof(
        executor,
        fixture,
        workspace,
      );
      // A second invocation freezes steady-state candidate capacity before the
      // recurring-token benchmark. Arm A intentionally retains its current
      // per-call production allocations.
      checkAllThreePaths(
        sameStateProof.fullRows,
        sameStateProof.compactRows,
        fixture.seenTokenIds,
        fixture.spec.drawIndex + 1,
        aceRandomWord(
          ACCEPTED_SEED,
          "planner-sampling",
          BigInt(fixture.spec.drawIndex + 1),
        ),
        workspace,
        `${fixture.spec.id} preparation warmup`,
      );
      fixtures.push(Object.freeze({
        ...fixture,
        workspace,
        workspaceStatsAfterWarmup: workspace.stats,
        preparationCorrectness: sameStateProof.receipt,
      }));
    }
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0084 compound observed a WebGPU runtime event");
    }
    postProgress(
      "prewarming the first recurring phase before READY for the worker-clock launch bound",
    );
    const firstFixture = fixtures[0];
    if (firstFixture === undefined) {
      throw new Error("OPT-0084 compound omitted its first semantic fixture");
    }
    const firstPositionRecurringWarmup = await prepareRecurringPhase(
      executor,
      firstFixture,
      signal,
    );
    if (runtimeEvents.length !== 0) {
      throw new Error(
        "OPT-0084 compound observed a WebGPU event in first-position prewarm",
      );
    }
    const readyAtEpochMilliseconds = Date.now();
    let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
    const ownedExecutor = executor;
    const ownedContext = context;
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
      gate: "compact-head-fused-sampling-compound",
      runIdentity,
      candidateSeamCommit: OPT_0084_CANDIDATE_SEAM_COMMIT,
      plannerSamplingSourceSha256: samplingSourceSha256,
      preparationStartedAtEpochMilliseconds,
      readyAtEpochMilliseconds,
      package: preparedPackage.receipt,
      tokenizer: tokenizerLoaded.assetIdentity,
      fixture: compoundFixtureReceipt(),
      execution: Object.freeze({
        modelProfile: "reference-bf16",
        executionProfile: context.capabilities.executionProfile.id,
        schedulingProfile: "cooperative",
        adapterInfo: context.capabilities.adapterInfo,
        oneAuthenticatedPlannerOwner: true,
        duplicateEquivalentPhaseSameStateFullCompactDispatchProof: true,
        exactNextTokenFullLogitCacheAppendWitness: true,
        executorWriteStatusValidatedOnEveryMappedReadback: true,
        acquisitionWallMilliseconds,
        uploadWallMilliseconds,
      }),
      correctness: Object.freeze({
        passed: true,
        positions: Object.freeze(fixtures.map((fixture) =>
          fixture.preparationCorrectness)),
        retainedLogitsRawU32Exact: true,
        filteringRawU32Exact: true,
        tokenWordCursorExact: true,
        cacheAppendGeometryValidatedByExecutor: true,
        writeStatusValidatedByExecutor: true,
        samePreAppendCacheAndDispatchProof: true,
      }),
      warmup: Object.freeze({
        bothDecodeDispatchTopologiesConstructedDuringPreparation: true,
        candidateWorkspacesInitializedBeforeReady: true,
        note: "the first phase is warmed before READY; later positions repeat one untimed decode/replay warmup because dispatch caches are phase-local",
        firstPositionRecurringWarmupCompletedBeforeReady: true,
        firstPositionRecurringWarmup,
      }),
    });
    return Object.freeze({
      runIdentity,
      context,
      executor,
      runtimeEvents,
      fixtures: Object.freeze(fixtures),
      firstPositionRecurringWarmup,
      packageReceipt: preparedPackage.receipt,
      tokenizerReceipt: tokenizerLoaded.assetIdentity,
      preparationReceipt,
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
        `OPT-0084 compound preparation failed (${errorText(error)}); cleanup ` +
          `also failed: ${cleanupFailures.map(errorText).join("; ")}`,
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
  const samples: Opt0084CompoundTimingSample[] = [];
  const cases: Readonly<Record<string, unknown>>[] = [];
  const workerRunAcceptedAtEpochMilliseconds = Date.now();
  if (workerRunAcceptedAtEpochMilliseconds <
    thermalLaunch.gateCompletedAtEpochMilliseconds) {
    throw new Error("OPT-0084 compound timing began before the nominal gate");
  }
  let measurementStartedAtEpochMilliseconds: number | undefined;
  for (let positionOrder = 0; positionOrder < prepared.fixtures.length;
    positionOrder += 1) {
    signal.throwIfAborted();
    const fixture = prepared.fixtures[positionOrder]!;
    const executed = await runCase(
      prepared.executor,
      fixture,
      positionOrder,
      signal,
      positionOrder === 0
        ? prepared.firstPositionRecurringWarmup
        : undefined,
      positionOrder === 0
        ? () => {
            const started = Date.now();
            const workerLaunchGapMilliseconds = started -
              thermalLaunch.gateCompletedAtEpochMilliseconds;
            if (workerLaunchGapMilliseconds < 0 ||
              workerLaunchGapMilliseconds > 5_000) {
              throw new Error(
                "OPT-0084 compound worker-clock timing launch gap exceeded 5 seconds",
              );
            }
            measurementStartedAtEpochMilliseconds = started;
          }
        : undefined,
    );
    samples.push(...executed.samples);
    cases.push(executed.receipt);
  }
  if (measurementStartedAtEpochMilliseconds === undefined) {
    throw new Error("OPT-0084 compound omitted its actual timing start boundary");
  }
  const measurementCompletedAtEpochMilliseconds = Date.now();
  const decision = evaluateOpt0084CompoundTiming(samples);
  const cancellation = runCancellationCheckpointProof(prepared.fixtures[0]!);
  if (prepared.runtimeEvents.length !== 0) {
    throw new Error("OPT-0084 compound observed a WebGPU runtime event");
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
  const inPagePassed = decision.passed && cancellation["passed"] === true &&
    cleanupPassed;
  return Object.freeze({
    schema: OPT_0084_COMPOUND_RECEIPT_SCHEMA,
    experiment: OPT_0084_EXPERIMENT_ID,
    gate: "compact-head-fused-sampling-compound",
    status: "awaiting-external-thermal-completion",
    passed: false,
    inPagePassed,
    thermalLaunch,
    measurementStartedAtEpochMilliseconds,
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
      oneAuthenticatedPlannerOwner: true,
      actualSequentialM2Tokens: true,
      completeTokenArms: Object.freeze({
        A: "full-217204-row-head-plus-current-browser-v1-sampler",
        B: "same-full-head-plus-OPT-0084-fused-candidate-radix-sampler",
        C: "exact-64000-row-compact-head-plus-OPT-0084-compact-sampler",
      }),
      timingAuthority:
        "performance-now-around-actual-sequential-decode-plus-selected-sampler",
      recurringTokenWallExcludesOneTimeDispatchConstruction: true,
      untimedDecodeAndReplayBeforeEachPositionTiming: true,
      sameStateOppositeHeadReplayExcludedFromCompleteTokenWall: true,
      reusedExactMechanism:
        "replayTiedHeadForOpt0082(logitRange?) benchmark-internal exact head replay",
      armOrders: OPT_0084_COMPOUND_ARM_ORDERS.map((order) => order.join("-")),
      positionIds: OPT_0084_COMPOUND_POSITION_IDS,
      fixture: compoundFixtureReceipt(),
      workerRunAcceptedAtEpochMilliseconds,
      measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
    }),
    correctness: Object.freeze({
      preparation: prepared.preparationReceipt["correctness"],
      everyTimedPrimaryHasSameStateFullCompactReplay: true,
      everyRetainedLogitRawU32Exact: decision.everySampleSameStateExact,
      everyFilteredStateRawU32Exact: decision.everySampleSameStateExact,
      everyTokenWordAndCursorExact: decision.everySampleSameStateExact,
      everyCacheAppendGeometryValidated: true,
      everyMappedWriteStatusValidated: true,
      cancellation,
      lifecycle: Object.freeze({
        cleanupFirst,
        cleanupSecond,
        cleanupPassed,
        runtimeEventCount: prepared.runtimeEvents.length,
      }),
      passed: decision.everySampleSameStateExact &&
        cancellation["passed"] === true && cleanupPassed,
    }),
    timing: Object.freeze({
      samples: Object.freeze(samples),
      decision,
      cases: Object.freeze(cases),
    }),
    decision: Object.freeze({
      compoundGatePassed: inPagePassed,
      productionIntegrationAuthorized: false,
      trajectoryAndProductGatesStillRequired: inPagePassed,
      unchangedTimingRetryAuthorized: false,
    }),
  });
}

async function runSameStateDispatchAndCacheProof(
  executor: AcePlannerGpuExecutor,
  fixture: SemanticFixture,
  workspace: AceOpt0084PlannerSamplingWorkspace,
): Promise<Readonly<{
  readonly fullRows: readonly Float32Array[];
  readonly compactRows: readonly Float32Array[];
  readonly receipt: Readonly<Record<string, unknown>>;
}>> {
  const nextTokenId = ACCEPTED_SEMANTIC_TOKEN_IDS[
    fixture.nextTeacherTokenIndex % ACCEPTED_SEMANTIC_TOKEN_IDS.length
  ]!;
  const nextBatch = createDecodeBatch(
    fixture.spec.cacheCapacity,
    fixture.spec.cachedTokensBeforeAppend + 1,
    nextTokenId,
  );

  // Branch A starts from a fresh deterministic prefill, executes the ordinary
  // full-head dispatch, then records a full next-token witness of the cache
  // append made by that dispatch.
  const prefillA = requireFloat32Rows(
    await executor.prefill(fixture.prefill),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} same-state branch A prefill`,
  );
  const fullRows = requireFloat32Rows(
    await executor.decode(fixture.decode),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} same-state branch A full decode`,
  );
  const nextAfterFull = requireFloat32Rows(
    await executor.decode(nextBatch),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} branch A cache witness`,
  );

  // Branch C discards the mutated A phase, recreates the exact pre-append
  // phase from the same immutable batch, and executes the compact dispatch.
  // Its next full decode is therefore an independent cache-append witness.
  const prefillC = requireFloat32Rows(
    await executor.prefill(fixture.prefill),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} same-state branch C prefill`,
  );
  const compactRows = requireFloat32Rows(
    await executor.decode(fixture.decode, REGULAR_RANGE),
    REGULAR_RANGE.tokenCount,
    `${fixture.spec.id} same-state branch C compact decode`,
  );
  const nextAfterCompact = requireFloat32Rows(
    await executor.decode(nextBatch),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} branch C cache witness`,
  );

  const prefillExact = requireExactSlice(
    prefillA,
    prefillC,
    0,
    `${fixture.spec.id} duplicate equivalent prefill`,
  );
  const retainedExact = requireExactSlice(
    fullRows,
    compactRows,
    REGULAR_RANGE.firstTokenId,
    `${fixture.spec.id} same-pre-append full/compact dispatch`,
  );
  const cacheWitnessExact = requireExactSlice(
    nextAfterFull,
    nextAfterCompact,
    0,
    `${fixture.spec.id} exact next-token cache witness`,
  );
  const word = aceRandomWord(
    ACCEPTED_SEED,
    "planner-sampling",
    BigInt(fixture.spec.drawIndex),
  );
  const sampling = checkAllThreePaths(
    fullRows,
    compactRows,
    fixture.seenTokenIds,
    fixture.spec.drawIndex,
    word,
    workspace,
    `${fixture.spec.id} duplicate-phase same-state proof`,
  );
  return Object.freeze({
    fullRows,
    compactRows,
    receipt: Object.freeze({
      id: fixture.spec.id,
      passed: true,
      mechanism:
        "duplicate-equivalent-prefill-ordinary-vs-compact-dispatch-plus-next-full-logit-cache-witness",
      branchA: "fresh-prefill -> ordinary-full decode -> next-full decode",
      branchC: "fresh-identical-prefill -> compact decode -> next-full decode",
      samePreAppendCacheEstablishedByExactPrefill: prefillExact,
      sameStateRetainedLogitsRawU32Exact: retainedExact,
      cacheAppendExactNextTokenFullLogitWitness: cacheWitnessExact,
      allMappedWriteStatusWordsValidatedByReconstruction: true,
      prefillARawU32Sha256: hashRows(prefillA),
      prefillCRawU32Sha256: hashRows(prefillC),
      nextAfterFullRawU32Sha256: hashRows(nextAfterFull),
      nextAfterCompactRawU32Sha256: hashRows(nextAfterCompact),
      sampling,
    }),
  });
}

async function prepareRecurringPhase(
  executor: AcePlannerGpuExecutor,
  fixture: FixtureRuntime,
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  signal.throwIfAborted();
  const prefillStarted = performance.now();
  await executor.prefill(fixture.prefill);
  const prefillWallMilliseconds = performance.now() - prefillStarted;
  signal.throwIfAborted();
  const warmupStarted = performance.now();
  const warmupFullRows = requireFloat32Rows(
    await executor.decode(fixture.decode),
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${fixture.spec.id} recurring warmup full decode`,
  );
  const warmupCompactRows = requireFloat32Rows(
    await executor.replayTiedHeadForOpt0082(REGULAR_RANGE),
    REGULAR_RANGE.tokenCount,
    `${fixture.spec.id} recurring warmup compact replay`,
  );
  const correctness = checkAllThreePaths(
    warmupFullRows,
    warmupCompactRows,
    fixture.seenTokenIds,
    fixture.spec.drawIndex,
    aceRandomWord(
      ACCEPTED_SEED,
      "planner-sampling",
      BigInt(fixture.spec.drawIndex),
    ),
    fixture.workspace,
    `${fixture.spec.id} recurring warmup`,
  );
  const warmupWallMilliseconds = performance.now() - warmupStarted;
  return Object.freeze({
    fixtureId: fixture.spec.id,
    prefillWallMilliseconds,
    wallMilliseconds: warmupWallMilliseconds,
    fullDecodeConstructedAndExecuted: true,
    compactDecodeDispatchConstructedBySameStateHeadReplay: true,
    excludedFromEveryTimingSample: true,
    correctness,
  });
}

async function runCase(
  executor: AcePlannerGpuExecutor,
  fixture: FixtureRuntime,
  positionOrder: number,
  signal: AbortSignal,
  prewarmedPhase: Readonly<Record<string, unknown>> | undefined,
  onFirstTimedSample?: () => void,
): Promise<Readonly<{
  readonly samples: readonly Opt0084CompoundTimingSample[];
  readonly receipt: Readonly<Record<string, unknown>>;
}>> {
  signal.throwIfAborted();
  const recurringWarmup = prewarmedPhase ?? await prepareRecurringPhase(
    executor,
    fixture,
    signal,
  );
  if (recurringWarmup["fixtureId"] !== fixture.spec.id) {
    throw new Error(`${fixture.spec.id} recurring phase warmup identity differs`);
  }
  const statsBeforeTiming = fixture.workspace.stats;
  if (!sameWorkspaceStats(
    statsBeforeTiming,
    fixture.workspaceStatsAfterWarmup,
  )) {
    throw new Error(`${fixture.spec.id} workspace grew during recurring warmup`);
  }
  onFirstTimedSample?.();

  const samples: Opt0084CompoundTimingSample[] = [];
  const sameStateChecks: Readonly<Record<string, unknown>>[] = [];
  const seenTokenIds = [...fixture.seenTokenIds];
  let cachedTokensBeforeAppend = fixture.spec.cachedTokensBeforeAppend + 1;
  let nextTeacherTokenIndex = fixture.nextTeacherTokenIndex;
  let drawIndex = fixture.spec.drawIndex + 1;
  for (let roundIndex = 0;
    roundIndex < OPT_0084_COMPOUND_TIMING_ROUND_COUNT;
    roundIndex += 1) {
    const order = OPT_0084_COMPOUND_ARM_ORDERS[roundIndex]!;
    for (let armPosition = 0; armPosition < order.length; armPosition += 1) {
      signal.throwIfAborted();
      const arm = order[armPosition]!;
      const inputTokenId = ACCEPTED_SEMANTIC_TOKEN_IDS[
        nextTeacherTokenIndex % ACCEPTED_SEMANTIC_TOKEN_IDS.length
      ]!;
      const batch = createDecodeBatch(
        fixture.spec.cacheCapacity,
        cachedTokensBeforeAppend,
        inputTokenId,
      );
      const seenForStep = Object.freeze([...seenTokenIds, inputTokenId]);
      const execution = await executeTimedArm(
        executor,
        fixture.workspace,
        arm,
        fixture.spec.id,
        batch,
        seenForStep,
        drawIndex,
      );
      samples.push(Object.freeze({
        roundIndex,
        positionId: fixture.spec.id,
        arm,
        armPosition: armPosition as 0 | 1 | 2,
        positionOrder,
        cachedTokensBeforeAppend,
        completeTokenWallMilliseconds: execution.completeWallMilliseconds,
        modelWallMilliseconds: execution.modelWallMilliseconds,
        samplingWallMilliseconds: execution.samplingWallMilliseconds,
        tokenId: execution.sample.tokenId,
        word: execution.sample.word,
        drawIndex,
        cursorEnd: drawIndex + 1,
        sameStateReplayExact: true,
        cacheWriteStatusValidated: true,
      }));
      sameStateChecks.push(execution.correctness);
      seenTokenIds.push(inputTokenId);
      cachedTokensBeforeAppend += 1;
      nextTeacherTokenIndex += 1;
      drawIndex += 1;
    }
    postProgress(
      `${fixture.spec.id}: recurring triple ${roundIndex + 1}/` +
        `${OPT_0084_COMPOUND_TIMING_ROUND_COUNT}`,
    );
    await yieldToWorker();
  }
  const statsAfterTiming = fixture.workspace.stats;
  if (!sameWorkspaceStats(statsBeforeTiming, statsAfterTiming)) {
    throw new Error(`${fixture.spec.id} workspace allocated during timing`);
  }
  return Object.freeze({
    samples: Object.freeze(samples),
    receipt: Object.freeze({
      id: fixture.spec.id,
      positionOrder,
      initialCachedTokensBeforeAppend: fixture.spec.cachedTokensBeforeAppend,
      recurringTimingFirstCachedTokensBeforeAppend:
        fixture.spec.cachedTokensBeforeAppend + 1,
      finalCachedTokens: cachedTokensBeforeAppend,
      cacheCapacity: fixture.spec.cacheCapacity,
      untimedRecurringWarmup: recurringWarmup,
      timedInvocationCount: samples.length,
      sameStateChecks: Object.freeze(sameStateChecks),
      workspaceBeforeTiming: workspaceReceipt(statsBeforeTiming),
      workspaceAfterTiming: workspaceReceipt(statsAfterTiming),
      noCandidateStorageAllocationDuringTiming: true,
    }),
  });
}

async function executeTimedArm(
  executor: AcePlannerGpuExecutor,
  workspace: AceOpt0084PlannerSamplingWorkspace,
  arm: Arm,
  positionId: string,
  batch: AcePlannerDecodeBatch,
  seenTokenIds: readonly number[],
  drawIndex: number,
): Promise<Readonly<{
  readonly sample: SampleReceipt;
  readonly modelWallMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly completeWallMilliseconds: number;
  readonly correctness: Readonly<Record<string, unknown>>;
}>> {
  const completeStarted = performance.now();
  const returned = arm === "C"
    ? await executor.decode(batch, REGULAR_RANGE)
    : await executor.decode(batch);
  const rows = requireFloat32Rows(
    returned,
    arm === "C" ? REGULAR_RANGE.tokenCount :
      ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    `${positionId} timed ${arm} decode`,
  );
  const modelEnded = performance.now();
  const word = aceRandomWord(
    ACCEPTED_SEED,
    "planner-sampling",
    BigInt(drawIndex),
  );
  const primary = arm === "A"
    ? sampleAcePlannerToken(fullSampleInput(rows, seenTokenIds, word))
    : arm === "B"
    ? sampleAcePlannerTokenOpt0084(
        fullSampleInput(rows, seenTokenIds, word),
        workspace,
      )
    : sampleAcePlannerCompactTokenOpt0084(
        compactSampleInput(rows, seenTokenIds, word),
        workspace,
      );
  const samplingEnded = performance.now();
  const timedPrimaryWorkspace = arm === "A"
    ? Object.freeze({
        arm,
        applicable: false,
        reason: "Arm A has no OPT-0084 workspace",
      })
    : captureTimedPrimaryWorkspace(
        workspace,
        arm,
        rows,
        seenTokenIds,
        word,
        `${positionId} timed ${arm} primary-before-replay`,
      );

  // Replay only the opposite tied head after the timed boundary. This obtains
  // a full and compact view of the exact same final hidden rows without a
  // second transformer invocation or a second cache append.
  const opposite = arm === "C"
    ? requireFloat32Rows(
        await executor.replayTiedHeadForOpt0082(),
        ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
        `${positionId} timed ${arm} full replay`,
      )
    : requireFloat32Rows(
        await executor.replayTiedHeadForOpt0082(REGULAR_RANGE),
        REGULAR_RANGE.tokenCount,
        `${positionId} timed ${arm} compact replay`,
      );
  const fullRows = arm === "C" ? opposite : rows;
  const compactRows = arm === "C" ? rows : opposite;
  const correctness = checkAllThreePaths(
    fullRows,
    compactRows,
    seenTokenIds,
    drawIndex,
    word,
    workspace,
    `${positionId} timed ${arm} same-state replay`,
  );
  const expected = correctness["sample"] as Readonly<Record<string, unknown>>;
  if (
    primary.tokenId !== expected["tokenId"] ||
    primary.word !== expected["word"] ||
    primary.positiveCandidateCount !== expected["positiveCandidateCount"]
  ) {
    throw new Error(`${positionId} timed ${arm} primary sample changed on replay`);
  }
  return Object.freeze({
    sample: Object.freeze({
      tokenId: primary.tokenId,
      word: primary.word,
      positiveCandidateCount: primary.positiveCandidateCount,
      drawIndex: drawIndex.toString(),
      drawEnd: (drawIndex + 1).toString(),
    }),
    modelWallMilliseconds: modelEnded - completeStarted,
    samplingWallMilliseconds: samplingEnded - modelEnded,
    completeWallMilliseconds: samplingEnded - completeStarted,
    correctness: Object.freeze({
      ...correctness,
      timedPrimaryWorkspace,
      timedPrimaryWorkspaceCapturedBeforeReplay: true,
    }),
  });
}

function captureTimedPrimaryWorkspace(
  workspace: AceOpt0084PlannerSamplingWorkspace,
  arm: "B" | "C",
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  word: number,
  label: string,
): Readonly<Record<string, unknown>> {
  if (arm === "B") {
    const input = fullSampleInput(rows, seenTokenIds, word);
    const filtered = createAcePlannerFilteredLogits(input);
    const weights = createAcePlannerBrowserSamplingWeights(
      filtered,
      SAMPLING_PARAMETERS.temperature,
    );
    return Object.freeze({
      arm,
      applicable: true,
      capturedBeforeAnyReplayOrDiagnosticRerun: true,
      comparison: requireWorkspaceExact(
        workspace,
        filtered,
        weights,
        0,
        label,
      ),
    });
  }
  const input = compactSampleInput(rows, seenTokenIds, word);
  const filtered = createAcePlannerCompactFilteredLogits(input);
  const weights = createAcePlannerBrowserSamplingWeights(
    filtered,
    SAMPLING_PARAMETERS.temperature,
  );
  return Object.freeze({
    arm,
    applicable: true,
    capturedBeforeAnyReplayOrDiagnosticRerun: true,
    comparison: requireWorkspaceExact(
      workspace,
      filtered,
      weights,
      REGULAR_RANGE.firstTokenId,
      label,
    ),
  });
}

function checkAllThreePaths(
  fullRows: readonly Float32Array[],
  compactRows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  drawIndex: number,
  word: number,
  workspace: AceOpt0084PlannerSamplingWorkspace,
  label: string,
): Readonly<Record<string, unknown>> {
  const rawExact = requireExactSlice(
    fullRows,
    compactRows,
    REGULAR_RANGE.firstTokenId,
    `${label} retained logits`,
  );
  const filteredExact = compareFilteredPaths(
    fullRows,
    compactRows,
    REGULAR_RANGE,
    seenTokenIds,
    false,
    label,
  );
  const fullInput = fullSampleInput(fullRows, seenTokenIds, word);
  const compactInput = compactSampleInput(compactRows, seenTokenIds, word);
  const a = sampleAcePlannerToken(fullInput);
  const expectedFiltered = createAcePlannerFilteredLogits(fullInput);
  const expectedWeights = createAcePlannerBrowserSamplingWeights(
    expectedFiltered,
    SAMPLING_PARAMETERS.temperature,
  );
  const b = sampleAcePlannerTokenOpt0084(fullInput, workspace);
  const bWorkspace = requireWorkspaceExact(
    workspace,
    expectedFiltered,
    expectedWeights,
    0,
    `${label} B`,
  );
  const c = sampleAcePlannerCompactTokenOpt0084(compactInput, workspace);
  const cWorkspace = requireWorkspaceExact(
    workspace,
    expectedFiltered,
    expectedWeights,
    0,
    `${label} C`,
  );
  requireSameDirectSample(a, b, `${label} A/B`);
  requireSameDirectSample(a, c, `${label} A/C`);

  const cursorInput = withoutWord(fullInput);
  const compactCursorInput = withoutCompactWord(compactInput);
  const aCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const bCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const cCursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, drawIndex);
  const aCursorSample = aCursor.sample(cursorInput);
  const bCursorSample = bCursor.sampleOpt0084(cursorInput);
  const cCursorSample = cCursor.sampleCompactOpt0084(compactCursorInput);
  requireSameCursorSample(aCursorSample, bCursorSample, `${label} cursor A/B`);
  requireSameCursorSample(aCursorSample, cCursorSample, `${label} cursor A/C`);
  if (
    aCursor.consumed !== BigInt(drawIndex + 1) ||
    bCursor.consumed !== aCursor.consumed ||
    cCursor.consumed !== aCursor.consumed ||
    aCursorSample.word !== word
  ) {
    throw new Error(`${label} cursor/word boundary differs`);
  }
  return Object.freeze({
    label,
    rawExact,
    filteredExact,
    fusedFullFilteredAndWeightExact: bWorkspace,
    fusedCompactFilteredAndWeightExact: cWorkspace,
    sample: Object.freeze({
      tokenId: a.tokenId,
      word: a.word,
      positiveCandidateCount: a.positiveCandidateCount,
      drawIndex,
      cursorEnd: drawIndex + 1,
    }),
    cursorExact: true,
    cacheAppendGeometryValidatedByExecutor: true,
    mappedWriteStatusValidatedByExecutor: true,
  });
}

function fullSampleInput(
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  word: number,
) {
  return Object.freeze({
    conditionalLogits: rows[0]!,
    unconditionalLogits: rows[1]!,
    seenTokenIds,
    preCfgAllowedTokens: Object.freeze({
      kind: "range" as const,
      firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
      additionalTokenIds: Object.freeze([ACE_QWEN_IM_END_TOKEN_ID]),
    }),
    allowedTokens: Object.freeze({
      kind: "range" as const,
      firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
    }),
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
    word,
  });
}

function compactSampleInput(
  rows: readonly Float32Array[],
  seenTokenIds: readonly number[],
  word: number,
) {
  return Object.freeze({
    firstTokenId: REGULAR_RANGE.firstTokenId,
    vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    conditionalLogits: rows[0]!,
    unconditionalLogits: rows[1]!,
    seenTokenIds,
    parameters: SAMPLING_PARAMETERS,
    softmax: ACE_BROWSER_SOFTMAX_V1,
    word,
  });
}

function withoutWord(input: ReturnType<typeof fullSampleInput>) {
  const { word: _word, ...rest } = input;
  return Object.freeze(rest);
}

function withoutCompactWord(input: ReturnType<typeof compactSampleInput>) {
  const { word: _word, ...rest } = input;
  return Object.freeze(rest);
}

function requireWorkspaceExact(
  workspace: AceOpt0084PlannerSamplingWorkspace,
  expectedFiltered: Float32Array,
  expectedWeights: Float32Array,
  expectedFirstTokenId: number,
  label: string,
): Readonly<Record<string, unknown>> {
  const ids = workspace.copyLastCandidateTokenIds();
  const filtered = workspace.copyLastFilteredLogits();
  const weights = workspace.copyLastWeights();
  if (
    ids.length !== REGULAR_RANGE.tokenCount ||
    filtered.length !== ids.length || weights.length !== ids.length
  ) {
    throw new Error(`${label} workspace extent differs`);
  }
  const expectedFilteredWords = new Uint32Array(
    expectedFiltered.buffer,
    expectedFiltered.byteOffset,
    expectedFiltered.length,
  );
  const expectedWeightWords = new Uint32Array(
    expectedWeights.buffer,
    expectedWeights.byteOffset,
    expectedWeights.length,
  );
  const filteredWords = new Uint32Array(
    filtered.buffer,
    filtered.byteOffset,
    filtered.length,
  );
  const weightWords = new Uint32Array(
    weights.buffer,
    weights.byteOffset,
    weights.length,
  );
  for (let local = 0; local < ids.length; local += 1) {
    const global = REGULAR_RANGE.firstTokenId + local;
    const expectedIndex = global - expectedFirstTokenId;
    if (
      ids[local] !== global ||
      expectedIndex < 0 || expectedIndex >= expectedFilteredWords.length ||
      filteredWords[local] !== expectedFilteredWords[expectedIndex] ||
      weightWords[local] !== expectedWeightWords[expectedIndex]
    ) {
      throw new Error(`${label} workspace raw word differs at ${local}`);
    }
  }
  return Object.freeze({
    candidateCount: ids.length,
    candidateIdsAscendingAndComplete: true,
    filteredRawU32MismatchCount: 0,
    weightRawU32MismatchCount: 0,
    candidateIdU32LeSha256: aceSha256Hex(new Uint8Array(
      Uint32Array.from(ids).buffer,
    )),
    filteredRawU32Sha256: aceSha256Hex(new Uint8Array(
      filtered.buffer,
      filtered.byteOffset,
      filtered.byteLength,
    )),
    weightRawU32Sha256: aceSha256Hex(new Uint8Array(
      weights.buffer,
      weights.byteOffset,
      weights.byteLength,
    )),
  });
}

function requireSameDirectSample(
  expected: AcePlannerTokenSample,
  actual: AcePlannerTokenSample,
  label: string,
): void {
  if (
    expected.tokenId !== actual.tokenId || expected.word !== actual.word ||
    expected.positiveCandidateCount !== actual.positiveCandidateCount
  ) {
    throw new Error(`${label} direct sample differs`);
  }
}

function requireSameCursorSample(
  expected: AcePlannerCursorSample,
  actual: AcePlannerCursorSample,
  label: string,
): void {
  if (
    expected.tokenId !== actual.tokenId || expected.word !== actual.word ||
    expected.positiveCandidateCount !== actual.positiveCandidateCount ||
    expected.drawIndex !== actual.drawIndex
  ) {
    throw new Error(`${label} cursor sample differs`);
  }
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
    throw new Error("OPT-0084 compound semantic prompt tokenization changed");
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
      throw new Error(`OPT-0084 compound ${spec.id} leaves the accepted trajectory`);
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
    throw new Error("OPT-0084 compound semantic prefill geometry is invalid");
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
    throw new Error("OPT-0084 compound current reference manifest identity changed");
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
    throw new Error("OPT-0084 compound bounded planner acquisition changed");
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
    "OPT-0084 compound planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0084 compound planner tensor bytes",
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
    throw new Error("OPT-0084 compound reference planner inventory changed");
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
    throw new Error("OPT-0084 compound pinned semantic fixture identity changed");
  }
  if (
    OPT_0084_COMPOUND_ARM_ORDERS.length !==
      OPT_0084_COMPOUND_TIMING_ROUND_COUNT ||
    OPT_0084_COMPOUND_ARM_ORDERS.filter((order) =>
      order.indexOf("A") < order.indexOf("C")).length !== 8 ||
    OPT_0084_COMPOUND_ARM_ORDERS.filter((order) =>
      order.indexOf("C") < order.indexOf("A")).length !== 8 ||
    OPT_0084_COMPOUND_POSITION_IDS.join(",") !==
      CASE_SPECS.map((spec) => spec.id).join(",")
  ) {
    throw new Error("OPT-0084 compound A/C timing order is not balanced");
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

function hashRows(rows: readonly Float32Array[]): readonly string[] {
  return Object.freeze(rows.map((row) => aceSha256Hex(new Uint8Array(
    row.buffer,
    row.byteOffset,
    row.byteLength,
  ))));
}

function compoundFixtureReceipt(): Readonly<Record<string, unknown>> {
  return Object.freeze({
    seed: ACCEPTED_SEED,
    semanticCodeCount: ACCEPTED_SEMANTIC_CODE_IDS.length,
    semanticCodeU32LeSha256: ACCEPTED_SEMANTIC_CODE_SHA256,
    cases: CASE_SPECS,
    teacherStream:
      "cyclic-pinned-150-code-stream-for-bounded-sequential-kernel-timing",
    trajectoryOracleClaimed: false,
  });
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0084 compound package omitted ${name}`);
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
    throw new Error("OPT-0084 compound thermal launch clone failed validation");
  }
}

function runCancellationCheckpointProof(
  fixture: FixtureRuntime,
): Readonly<Record<string, unknown>> {
  const cursors = {
    A: new AcePlannerSamplingCursor(ACCEPTED_SEED, 700),
    B: new AcePlannerSamplingCursor(ACCEPTED_SEED, 700),
    C: new AcePlannerSamplingCursor(ACCEPTED_SEED, 700),
  };
  const before = Object.freeze({
    A: cursors.A.consumed,
    B: cursors.B.consumed,
    C: cursors.C.consumed,
  });
  const rejected: Record<Arm, boolean> = { A: false, B: false, C: false };
  for (const arm of ["A", "B", "C"] as const) {
    const controller = new AbortController();
    const length = arm === "C"
      ? REGULAR_RANGE.tokenCount
      : ACE_PLANNER_QWEN3_CONFIG.vocabularySize;
    const abortingRow = createOpt0084CompoundAbortingLogitRow(
      length,
      controller,
      `OPT-0084 compound ${arm} cancellation probe`,
    ) as Float32Array;
    const rows = Object.freeze([abortingRow, abortingRow]);
    try {
      if (arm === "A") {
        cursors.A.sample(withoutWord(fullSampleInput(
          rows,
          fixture.seenTokenIds,
          0,
        )));
      } else if (arm === "B") {
        cursors.B.sampleOpt0084(withoutWord(fullSampleInput(
          rows,
          fixture.seenTokenIds,
          0,
        )));
      } else {
        cursors.C.sampleCompactOpt0084(withoutCompactWord(compactSampleInput(
          rows,
          fixture.seenTokenIds,
          0,
        )));
      }
    } catch (error) {
      rejected[arm] = controller.signal.aborted &&
        error instanceof DOMException && error.name === "AbortError";
    }
  }
  const passed = rejected.A && rejected.B && rejected.C &&
    cursors.A.consumed === before.A && cursors.B.consumed === before.B &&
    cursors.C.consumed === before.C;
  if (!passed) throw new Error("OPT-0084 compound cancellation committed a draw");
  return Object.freeze({
    passed,
    fixtureId: fixture.spec.id,
    actualSamplerApisEntered: Object.freeze({
      A: "AcePlannerSamplingCursor.sample",
      B: "AcePlannerSamplingCursor.sampleOpt0084",
      C: "AcePlannerSamplingCursor.sampleCompactOpt0084",
    }),
    abortRaisedInsideActualLogitAccessBeforeCursorCommit: true,
    activeExecutorCancellationProofClaimed: false,
    liveExecutorCancellationRemainsAvailableThroughItsOwnedAbortSignal: true,
    rejected: Object.freeze(rejected),
    cursorBefore: Object.freeze({
      A: before.A.toString(), B: before.B.toString(), C: before.C.toString(),
    }),
    cursorAfter: Object.freeze({
      A: cursors.A.consumed.toString(),
      B: cursors.B.consumed.toString(),
      C: cursors.C.consumed.toString(),
    }),
  });
}

function workspaceReceipt(
  stats: AceOpt0084PlannerSamplingWorkspaceStats,
): Readonly<Record<string, number>> {
  return Object.freeze({
    candidateCapacity: stats.candidateCapacity,
    maskCapacity: stats.maskCapacity,
    storageAllocationCount: stats.storageAllocationCount,
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

function requireReceiptNumber(
  receipt: Readonly<Record<string, unknown>>,
  key: string,
): number {
  const value = receipt[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`OPT-0084 compound receipt ${key} is invalid`);
  }
  return value;
}

function yieldToWorker(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
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
