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
  createAcePlannerCotPrompt,
  type AcePlannerDecodeBatch,
  type AcePlannerLogitRange,
  type AcePlannerPrefillBatch,
} from "../../src/runtime/planner.js";
import { AcePlannerMetadataConstraintController } from
  "../../src/runtime/planner-metadata-fsm.js";
import {
  ACE_BROWSER_SOFTMAX_V1,
  AcePlannerSamplingCursor,
  type AcePlannerAllowedTokens,
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
import {
  AcePlannerGpuExecutor,
  type AcePlannerGpuExecutorProgress,
  type AcePlannerOpt0085SchedulingDiagnostics,
  type AcePlannerOpt0085SchedulingProfile,
} from "../../src/webgpu/planner-executor.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
} from "../../src/webgpu/qwen3.js";
import {
  OPT_0085_CANDIDATE_PROFILE,
  OPT_0085_PAIR_ORDERS,
  OPT_0085_PATH_IDS,
  OPT_0085_SCHEMA,
  evaluateOpt0085TimingGate,
  median,
  opt0085ProfileForArm,
  validateOpt0085PreGate,
  validateOpt0085RunIdentity,
  validateOpt0085Topology,
  type Opt0085Arm,
  type Opt0085PathId,
  type Opt0085RunIdentity,
  type Opt0085ThermalTraceMetadata,
} from "./opt-0085-planner-depth2-completion-epochs-contract.js";

const MANIFEST_PATH = "/model/files-reference/manifest.json";
const PLANNER_TENSOR_COUNT = 314;
const PLANNER_WEIGHT_FILE_COUNT = 33;
const PLANNER_RESIDENT_BYTES = 1_325_768_704;
const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const WORKER_HEARTBEAT_INTERVAL_MILLISECONDS = 10;
const TOKENIZER_FILE_NAMES = Object.freeze([
  "assets/planner/tokenizer.json",
  "assets/planner/tokenizer_config.json",
  "assets/planner/chat_template.jinja",
] as const);
const ACQUIRED_FILE_COUNT = PLANNER_WEIGHT_FILE_COUNT +
  TOKENIZER_FILE_NAMES.length;
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
const ACCEPTED_SEMANTIC_TOKEN_IDS = Object.freeze(
  ACCEPTED_SEMANTIC_CODE_IDS.map(
    (code) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + code,
  ),
);
const REGULAR_RANGE: AcePlannerLogitRange = Object.freeze({
  firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
});
const EOS_RANGE: AcePlannerLogitRange = Object.freeze({
  firstTokenId: ACE_QWEN_IM_END_TOKEN_ID,
  tokenCount: 1,
});
const SAMPLING_PARAMETERS: AcePlannerSamplingParameters = Object.freeze({
  temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
  guidanceScale: DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale,
  topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
  topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
  repetitionPenalty: 1,
});

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0085RunIdentity;
}

interface RunTimedMessage {
  readonly type: "run-timed";
  readonly thermal: Opt0085ThermalTraceMetadata;
}

type IncomingMessage = InitializeMessage | RunTimedMessage;

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface PlannerFixture {
  readonly id: "cot-m1-middle" | "semantic-m2-middle";
  readonly rows: 1 | 2;
  readonly prefill: AcePlannerPrefillBatch;
  readonly decode: AcePlannerDecodeBatch;
  readonly seenTokenIdsIncludingDecode: readonly number[];
  readonly drawIndex: number;
  readonly promptTokenIds: readonly number[];
  readonly emittedTokenIdsIncludingDecode: readonly number[];
  readonly tokenizer: AceQwenBpeTokenizer;
}

interface PathSpec {
  readonly id: Opt0085PathId;
  readonly fixtureId: PlannerFixture["id"];
  readonly logitRange?: AcePlannerLogitRange;
  readonly expectedTotalCommandBuffers: 33 | 34;
  readonly sampleMode: "cot-full" | "semantic-full" | "semantic-compact" |
    "semantic-eos";
}

interface SampleReceipt {
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
  readonly drawIndex: string;
  readonly drawEnd: string;
}

interface ExactReceipt {
  readonly rows: 1 | 2;
  readonly elements: number;
  readonly mismatchCount: 0;
  readonly nanCount: 0;
}

interface ArmExecution {
  readonly arm: Opt0085Arm;
  readonly profile: AcePlannerOpt0085SchedulingProfile;
  readonly rows: readonly Float32Array[];
  readonly sample: SampleReceipt;
  readonly diagnostics: AcePlannerOpt0085SchedulingDiagnostics;
  readonly progress: Readonly<Record<string, unknown>>;
  readonly prefillWallMilliseconds: number;
  readonly modelWallMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly completeWallMilliseconds: number;
  readonly rowSha256: readonly string[];
}

interface PreparedSession {
  readonly identity: Opt0085RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly abortController: AbortController;
  readonly router: EvidenceRouter;
  readonly executor: AcePlannerGpuExecutor;
  readonly fixtures: ReadonlyMap<PlannerFixture["id"], PlannerFixture>;
  readonly preparedPackage: PreparedPackage;
  readonly tokenizerIdentity: Readonly<Record<string, unknown>>;
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly warmup: readonly Readonly<Record<string, unknown>>[];
  readonly warmupCompletedAtEpochMilliseconds: number;
}

interface WorkerHeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly timerTickCount: number;
  readonly maximumTimerGapMilliseconds: number;
}

const PATHS: readonly PathSpec[] = Object.freeze([
  Object.freeze({
    id: "cot-m1-middle-full",
    fixtureId: "cot-m1-middle",
    expectedTotalCommandBuffers: 34,
    sampleMode: "cot-full",
  }),
  Object.freeze({
    id: "semantic-m2-middle-full",
    fixtureId: "semantic-m2-middle",
    expectedTotalCommandBuffers: 34,
    sampleMode: "semantic-full",
  }),
  Object.freeze({
    id: "semantic-m2-middle-compact",
    fixtureId: "semantic-m2-middle",
    logitRange: REGULAR_RANGE,
    expectedTotalCommandBuffers: 33,
    sampleMode: "semantic-compact",
  }),
  Object.freeze({
    id: "semantic-m2-middle-forced-eos",
    fixtureId: "semantic-m2-middle",
    logitRange: EOS_RANGE,
    expectedTotalCommandBuffers: 33,
    sampleMode: "semantic-eos",
  }),
]);

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let session: PreparedSession | undefined;
let workerHeartbeat: ReturnType<typeof startWorkerHeartbeat> | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "initialize") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    workerHeartbeat = startWorkerHeartbeat();
    void initializeSession(event.data.identity).then(
      (prepared) => {
        if (lifecycle !== "preparing") return;
        session = prepared;
        lifecycle = "ready";
        self.postMessage({
          type: "ready-for-thermal-gate",
          warmupCompletedAtEpochMilliseconds:
            prepared.warmupCompletedAtEpochMilliseconds,
          preparation: publicPreparation(prepared),
        });
      },
      (error: unknown) => void failAndCleanup(error),
    );
    return;
  }
  if (event.data.type === "run-timed" && lifecycle === "ready") {
    lifecycle = "running";
    const active = session!;
    session = undefined;
    void runTimedAndCleanup(active, event.data.thermal).then(
      (result) => {
        lifecycle = "settled";
        self.postMessage({ type: "awaiting-through-cleanup-thermal", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
});

async function initializeSession(identity: unknown): Promise<PreparedSession> {
  const runIdentity = validateOpt0085RunIdentity(identity);
  validateStaticProtocol();
  const abortController = new AbortController();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  let context: AceWebGpuDeviceContext | undefined;
  let phase: AceGpuTensorPhase | undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  try {
    postProgress("authenticating the current reference-BF16 planner package");
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

    postProgress("uploading one authenticated 1.235 GiB planner owner");
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
    ) throw new Error("OPT-0085 loaded planner phase identity changed");
    const phaseUploadWallMilliseconds = performance.now() - uploadStarted;

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
    const fixtures = createFixtures(tokenizerLoaded.tokenizer);
    const router = new EvidenceRouter(abortController);
    const compileStarted = performance.now();
    const ownedPlannerWeights = phase;
    phase = undefined;
    executor = AcePlannerGpuExecutor.create({
      device: context.device,
      modelProfile: "reference-bf16",
      ownedPlannerWeights,
      signal: abortController.signal,
      onProgress: (progress) => router.acceptProgress(progress),
      onOpt0085Scheduling: (diagnostics) =>
        router.acceptDiagnostics(diagnostics),
    });

    const warmup: Readonly<Record<string, unknown>>[] = [];
    for (let index = 0; index < PATHS.length; index += 1) {
      const path = PATHS[index]!;
      postProgress(`warmup ${index + 1}/${PATHS.length}: ${path.id}`);
      const fixture = requireMapValue(fixtures, path.fixtureId);
      const control = await executeArm(
        executor,
        router,
        path,
        fixture,
        "control",
      );
      const candidate = await executeArm(
        executor,
        router,
        path,
        fixture,
        "candidate",
      );
      const exact = requireExactRows(
        control.rows,
        candidate.rows,
        `${path.id} warmup`,
      );
      requireSameSample(control.sample, candidate.sample, `${path.id} warmup`);
      warmup.push(Object.freeze({
        pathId: path.id,
        rawLogits: exact,
        cacheWriteStatus: cacheWriteStatusReceipt(fixture.rows),
        sample: control.sample,
        control: publicArm(control),
        candidate: publicArm(candidate),
      }));
    }
    const executorCompileAndWarmupWallMilliseconds =
      performance.now() - compileStarted;
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0085 warmup observed a WebGPU runtime event");
    }
    return Object.freeze({
      identity: runIdentity,
      context,
      runtimeEvents,
      abortController,
      router,
      executor,
      fixtures,
      preparedPackage,
      tokenizerIdentity: tokenizerLoaded.assetIdentity,
      preparation: Object.freeze({
        acquisitionWallMilliseconds,
        phaseUploadWallMilliseconds,
        executorCompileAndWarmupWallMilliseconds,
      }),
      warmup: Object.freeze(warmup),
      warmupCompletedAtEpochMilliseconds: Date.now(),
    });
  } catch (error) {
    if (executor !== undefined) await executor.destroy(error);
    else phase?.destroy();
    context?.destroy();
    throw error;
  }
}

async function runTimedAndCleanup(
  prepared: PreparedSession,
  thermal: Opt0085ThermalTraceMetadata,
): Promise<Readonly<Record<string, unknown>>> {
  validateOpt0085PreGate(
    thermal,
    prepared.warmupCompletedAtEpochMilliseconds,
  );
  const timedStartedAtEpochMilliseconds = Date.now();
  const paths: Readonly<Record<string, unknown>>[] = [];
  const gateInputs: Array<Readonly<{
    id: Opt0085PathId;
    controlCompleteWallMilliseconds: readonly number[];
    candidateCompleteWallMilliseconds: readonly number[];
  }>> = [];
  let executorDestroyed = false;
  let contextDestroyed = false;
  try {
    for (let pathIndex = 0; pathIndex < PATHS.length; pathIndex += 1) {
      const path = PATHS[pathIndex]!;
      const fixture = requireMapValue(prepared.fixtures, path.fixtureId);
      const controlComplete: number[] = [];
      const candidateComplete: number[] = [];
      const pairs: Readonly<Record<string, unknown>>[] = [];
      let candidateWins = 0;
      for (
        let pairIndex = 0;
        pairIndex < OPT_0085_PAIR_ORDERS.length;
        pairIndex += 1
      ) {
        const order = OPT_0085_PAIR_ORDERS[pairIndex]!;
        const executions = new Map<Opt0085Arm, ArmExecution>();
        for (const arm of order) {
          const execution = await executeArm(
            prepared.executor,
            prepared.router,
            path,
            fixture,
            arm,
          );
          executions.set(arm, execution);
        }
        const control = requireMapValue(executions, "control");
        const candidate = requireMapValue(executions, "candidate");
        const exact = requireExactRows(
          control.rows,
          candidate.rows,
          `${path.id} pair ${pairIndex}`,
        );
        requireSameSample(
          control.sample,
          candidate.sample,
          `${path.id} pair ${pairIndex}`,
        );
        if (
          path.sampleMode === "semantic-eos" &&
          control.sample.tokenId !== ACE_QWEN_IM_END_TOKEN_ID
        ) throw new Error("OPT-0085 forced-EOS path did not select EOS");
        controlComplete.push(control.completeWallMilliseconds);
        candidateComplete.push(candidate.completeWallMilliseconds);
        if (
          candidate.completeWallMilliseconds < control.completeWallMilliseconds
        ) candidateWins += 1;
        pairs.push(Object.freeze({
          pairIndex,
          order: order.join("-"),
          rawLogits: exact,
          cacheWriteStatus: cacheWriteStatusReceipt(fixture.rows),
          sampleExact: true,
          sample: control.sample,
          control: publicArm(control),
          candidate: publicArm(candidate),
          candidateCompleteSavingMilliseconds:
            control.completeWallMilliseconds -
              candidate.completeWallMilliseconds,
        }));
        postProgress(
          `${path.id}: timed pair ${pairIndex + 1}/` +
            `${OPT_0085_PAIR_ORDERS.length}`,
        );
      }
      const controlMedian = median(controlComplete);
      const candidateMedian = median(candidateComplete);
      gateInputs.push(Object.freeze({
        id: path.id,
        controlCompleteWallMilliseconds: Object.freeze(controlComplete),
        candidateCompleteWallMilliseconds: Object.freeze(candidateComplete),
      }));
      paths.push(Object.freeze({
        id: path.id,
        fixtureId: path.fixtureId,
        rows: fixture.rows,
        cachedTokensBeforeAppend: fixture.decode.cachedTokensBeforeAppend,
        cacheCapacity: fixture.decode.cacheCapacity,
        drawIndex: fixture.drawIndex,
        logitExtent: path.logitRange === undefined
          ? Object.freeze({
              kind: "full-vocabulary",
              firstTokenId: 0,
              tokenCount: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
            })
          : Object.freeze({ kind: "retained-range", ...path.logitRange }),
        expectedTotalCommandBuffers: path.expectedTotalCommandBuffers,
        candidateWins,
        control: summarizeTimings(controlComplete),
        candidate: summarizeTimings(candidateComplete),
        candidateMedianBelowControl: candidateMedian < controlMedian,
        medianCompleteSavingMilliseconds: controlMedian - candidateMedian,
        pairs: Object.freeze(pairs),
      }));
    }
    const timedCompletedAtEpochMilliseconds = Date.now();
    const timingGate = evaluateOpt0085TimingGate(gateInputs);
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0085 timing observed a WebGPU runtime event");
    }

    postProgress("running post-timing depth-two cancellation proof");
    const cancellationFixture = requireMapValue(
      prepared.fixtures,
      "cot-m1-middle",
    );
    await prepared.executor.prefill(cancellationFixture.prefill);
    const cancellation = await runCancellationProof(
      prepared,
      cancellationFixture,
    );
    await prepared.executor.destroy();
    await prepared.executor.destroy();
    executorDestroyed = true;
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0085 cleanup observed a WebGPU runtime event");
    }
    prepared.context.destroy();
    contextDestroyed = true;
    const heartbeat = workerHeartbeat!.stop();
    workerHeartbeat = undefined;
    validateWorkerHeartbeat(heartbeat);
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    return Object.freeze({
      schema: OPT_0085_SCHEMA,
      schemaVersion: 1,
      experimentId: "OPT-0085",
      statusBeforeFinalThermalJoin: timingGate.passed
        ? "candidate-passed-performance-gate"
        : "candidate-failed-performance-gate",
      identity: Object.freeze({
        ...prepared.identity,
        manifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
        modelProfile: "reference-bf16",
        schedulingProfile: "cooperative",
        capabilities: prepared.context.capabilities,
      }),
      protocol: Object.freeze({
        benchmarkOnly: true,
        productionSelectorChanged: false,
        arithmeticChanged: false,
        commandBuffersChanged: false,
        oneAuthenticatedPlannerWeightOwner: true,
        onePlannerExecutor: true,
        sameStateReplay:
          "untimed identical prefill before every timed arm on one executor",
        completeTokenWallDefinition:
          "performance.now around explicit decode through terminal fence/map plus selected sampler",
        perFenceTimingUse: "diagnostic-only-never-summed",
        setupAndPrefillExcludedFromTiming: true,
        balancedInterleaving: true,
        timingPairOrdersPerPath: OPT_0085_PAIR_ORDERS.map(
          (order) => order.join("-"),
        ),
        totalPairCount: gateInputs.reduce(
          (total, path) => total + path.controlCompleteWallMilliseconds.length,
          0,
        ),
      }),
      package: prepared.preparedPackage.receipt,
      tokenizer: prepared.tokenizerIdentity,
      fixture: Object.freeze({
        seed: ACCEPTED_SEED,
        cotTrajectoryU32LeSha256: ACCEPTED_COT_TRAJECTORY_SHA256,
        semanticCodeCount: ACCEPTED_SEMANTIC_CODE_IDS.length,
        semanticCodeU32LeSha256: ACCEPTED_SEMANTIC_CODE_SHA256,
        pathOrder: OPT_0085_PATH_IDS,
      }),
      preparation: Object.freeze({
        ...prepared.preparation,
        warmupCompletedAtEpochMilliseconds:
          prepared.warmupCompletedAtEpochMilliseconds,
        exactWarmups: prepared.warmup,
      }),
      thermal: Object.freeze({
        preGate: thermal,
        status: "pending-continuous-through-cleanup-trace-join",
        browserReceiptClaimsCompleteThermalCoverage: false,
        continuousLoggerRequiredThroughEpochMilliseconds:
          cleanupCompletedAtEpochMilliseconds,
      }),
      timedStartedAtEpochMilliseconds,
      timedCompletedAtEpochMilliseconds,
      paths: Object.freeze(paths),
      timingGate,
      cancellation,
      lifecycle: Object.freeze({
        runtimeEvents: Object.freeze([...prepared.runtimeEvents]),
        executorDestroyCalledTwice: true,
        executorDestroyed,
        deviceDestroyedAfterRuntimeEventCheck: contextDestroyed,
        outputPublishedOnlyAfterTerminalReadbackFence: true,
        cleanupCompletedAtEpochMilliseconds,
      }),
      workerHeartbeat: heartbeat,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      completedAt: new Date(cleanupCompletedAtEpochMilliseconds).toISOString(),
    });
  } finally {
    if (!executorDestroyed) {
      prepared.router.abandon();
      try {
        await prepared.executor.destroy();
      } finally {
        if (!contextDestroyed) prepared.context.destroy();
      }
    }
  }
}

async function executeArm(
  executor: AcePlannerGpuExecutor,
  router: EvidenceRouter,
  path: PathSpec,
  fixture: PlannerFixture,
  arm: Opt0085Arm,
): Promise<ArmExecution> {
  const profile = opt0085ProfileForArm(arm);
  const prefillStarted = performance.now();
  await executor.prefill(fixture.prefill, path.logitRange);
  const prefillWallMilliseconds = performance.now() - prefillStarted;
  router.begin(profile, path.expectedTotalCommandBuffers);
  const completeStarted = performance.now();
  let rows: readonly Float32Array[];
  try {
    rows = requireFloat32Rows(
      await executor.decodeForOpt0085(
        profile,
        fixture.decode,
        path.logitRange,
      ),
      fixture.rows,
      path.logitRange?.tokenCount ?? ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
      `${path.id} ${arm}`,
    );
  } catch (error) {
    router.abandon();
    throw error;
  }
  const modelEnded = performance.now();
  const sample = samplePath(path, fixture, rows);
  const samplingEnded = performance.now();
  const captured = router.end(profile, path.expectedTotalCommandBuffers);
  return Object.freeze({
    arm,
    profile,
    rows,
    sample,
    diagnostics: captured.diagnostics,
    progress: captured.progress,
    prefillWallMilliseconds,
    modelWallMilliseconds: modelEnded - completeStarted,
    samplingWallMilliseconds: samplingEnded - modelEnded,
    completeWallMilliseconds: samplingEnded - completeStarted,
    rowSha256: Object.freeze(rows.map(rowSha256)),
  });
}

async function runCancellationProof(
  prepared: PreparedSession,
  fixture: PlannerFixture,
): Promise<Readonly<Record<string, unknown>>> {
  const reason = new DOMException(
    "OPT-0085 post-timing depth-two cancellation proof",
    "AbortError",
  );
  prepared.router.beginCancellation(reason);
  let rejection: unknown;
  try {
    await prepared.executor.decodeForOpt0085(
      OPT_0085_CANDIDATE_PROFILE,
      fixture.decode,
    );
  } catch (error) {
    rejection = error;
  }
  const captured = prepared.router.endCancellation();
  if (
    rejection !== reason ||
    !prepared.abortController.signal.aborted ||
    captured.progressCount !== 1 ||
    captured.diagnosticsCount !== 0
  ) throw new Error("OPT-0085 actual-browser cancellation proof changed");
  return Object.freeze({
    rejectionName: reason.name,
    rejectionMessage: reason.message,
    rejectionIdentityPreserved: rejection === reason,
    abortRequestedFromFirstCompletionCallback: true,
    completedProgressCallbacks: captured.progressCount,
    diagnosticsPublished: captured.diagnosticsCount,
    resultRowsPublished: false,
    remainingBackfillStoppedByOwnedAbortSignal: true,
    submittedWorkSettledBeforeExecutorDestroy: true,
  });
}

class EvidenceRouter {
  private mode: "none" | "capture" | "cancellation" = "none";
  private profile: AcePlannerOpt0085SchedulingProfile | undefined;
  private expectedTotal: 33 | 34 | undefined;
  private progress: AcePlannerGpuExecutorProgress[] = [];
  private diagnostics: AcePlannerOpt0085SchedulingDiagnostics[] = [];
  private cancellationReason: DOMException | undefined;
  private cancellationRequested = false;

  constructor(private readonly abortController: AbortController) {}

  begin(
    profile: AcePlannerOpt0085SchedulingProfile,
    expectedTotal: 33 | 34,
  ): void {
    this.requireNone();
    this.mode = "capture";
    this.profile = profile;
    this.expectedTotal = expectedTotal;
    this.progress = [];
    this.diagnostics = [];
  }

  beginCancellation(reason: DOMException): void {
    this.requireNone();
    this.mode = "cancellation";
    this.profile = OPT_0085_CANDIDATE_PROFILE;
    this.expectedTotal = 34;
    this.progress = [];
    this.diagnostics = [];
    this.cancellationReason = reason;
    this.cancellationRequested = false;
  }

  acceptProgress(progress: AcePlannerGpuExecutorProgress): void {
    if (this.mode === "none") return;
    this.progress.push(Object.freeze({ ...progress }));
    if (this.mode === "cancellation" && !this.cancellationRequested) {
      this.cancellationRequested = true;
      this.abortController.abort(this.cancellationReason);
    }
  }

  acceptDiagnostics(
    diagnostics: AcePlannerOpt0085SchedulingDiagnostics,
  ): void {
    if (this.mode === "none") {
      throw new Error("OPT-0085 diagnostics escaped an explicit invocation");
    }
    this.diagnostics.push(Object.freeze({ ...diagnostics }));
  }

  end(
    expectedProfile: AcePlannerOpt0085SchedulingProfile,
    expectedTotal: 33 | 34,
  ): Readonly<{
    diagnostics: AcePlannerOpt0085SchedulingDiagnostics;
    progress: Readonly<Record<string, unknown>>;
  }> {
    if (
      this.mode !== "capture" ||
      this.profile !== expectedProfile ||
      this.expectedTotal !== expectedTotal ||
      this.diagnostics.length !== 1
    ) throw new Error("OPT-0085 evidence-router capture changed");
    const diagnostics = this.diagnostics[0]!;
    validateOpt0085Topology(diagnostics, expectedProfile, expectedTotal);
    const progress = validateProgress(
      this.progress,
      expectedProfile,
      expectedTotal,
    );
    this.reset();
    return Object.freeze({ diagnostics, progress });
  }

  endCancellation(): Readonly<{
    progressCount: number;
    diagnosticsCount: number;
  }> {
    if (
      this.mode !== "cancellation" ||
      !this.cancellationRequested ||
      this.profile !== OPT_0085_CANDIDATE_PROFILE ||
      this.expectedTotal !== 34
    ) throw new Error("OPT-0085 cancellation router changed");
    const result = Object.freeze({
      progressCount: this.progress.length,
      diagnosticsCount: this.diagnostics.length,
    });
    this.reset();
    return result;
  }

  abandon(): void {
    this.reset();
  }

  private requireNone(): void {
    if (this.mode !== "none") {
      throw new DOMException(
        "OPT-0085 evidence captures overlap",
        "InvalidStateError",
      );
    }
  }

  private reset(): void {
    this.mode = "none";
    this.profile = undefined;
    this.expectedTotal = undefined;
    this.progress = [];
    this.diagnostics = [];
    this.cancellationReason = undefined;
    this.cancellationRequested = false;
  }
}

function validateProgress(
  events: readonly AcePlannerGpuExecutorProgress[],
  profile: AcePlannerOpt0085SchedulingProfile,
  expectedTotal: 33 | 34,
): Readonly<Record<string, unknown>> {
  const candidate = profile === OPT_0085_CANDIDATE_PROFILE;
  const expectedDrains = candidate
    ? Math.ceil(expectedTotal / 4)
    : expectedTotal;
  const expectedIdleTurns = candidate ? expectedDrains - 1 : expectedTotal;
  if (
    events.length !== expectedTotal ||
    events.some((event, index) =>
      event.phaseKind !== "decode" ||
      event.totalCommandBuffers !== expectedTotal ||
      event.completedCommandBuffers !== index + 1 ||
      event.stage !== (index === expectedTotal - 1 ? "readback" : "model") ||
      (index === expectedTotal - 1) !== (event.quantum === null)
    ) ||
    events.at(-1)?.queueDrains !== expectedDrains ||
    events.at(-1)?.cooperativeIdleMs !== expectedIdleTurns
  ) throw new Error(`OPT-0085 ${profile} progress sequence changed`);
  return Object.freeze({
    callbackCount: events.length,
    firstCompletedCommandBuffers: events[0]!.completedCommandBuffers,
    finalCompletedCommandBuffers: events.at(-1)!.completedCommandBuffers,
    finalStage: events.at(-1)!.stage,
    finalQueueDrains: events.at(-1)!.queueDrains,
    finalCooperativeIdleMilliseconds: events.at(-1)!.cooperativeIdleMs,
    completionOrderMonotonic: true,
    readbackProgressLast: true,
  });
}

function samplePath(
  path: PathSpec,
  fixture: PlannerFixture,
  logits: readonly Float32Array[],
): SampleReceipt {
  const cursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, fixture.drawIndex);
  let sample: AcePlannerCursorSample;
  if (path.sampleMode === "cot-full") {
    const allowedTokens = resolveCotAllowedTokens(fixture, logits[0]!);
    sample = cursor.sample({
      conditionalLogits: logits[0]!,
      seenTokenIds: fixture.seenTokenIdsIncludingDecode,
      allowedTokens,
      parameters: Object.freeze({
        ...SAMPLING_PARAMETERS,
        guidanceScale: 1,
      }),
      softmax: ACE_BROWSER_SOFTMAX_V1,
    });
  } else if (path.sampleMode === "semantic-full") {
    sample = cursor.sample({
      conditionalLogits: logits[0]!,
      unconditionalLogits: logits[1]!,
      seenTokenIds: fixture.seenTokenIdsIncludingDecode,
      preCfgAllowedTokens: semanticPreCfgAllowedTokens(),
      allowedTokens: Object.freeze({
        kind: "range" as const,
        firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
        tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
      }),
      parameters: SAMPLING_PARAMETERS,
      softmax: ACE_BROWSER_SOFTMAX_V1,
    });
  } else {
    const range = path.logitRange!;
    sample = cursor.sampleCompact({
      firstTokenId: range.firstTokenId,
      vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
      conditionalLogits: logits[0]!,
      unconditionalLogits: logits[1]!,
      seenTokenIds: fixture.seenTokenIdsIncludingDecode,
      parameters: SAMPLING_PARAMETERS,
      softmax: ACE_BROWSER_SOFTMAX_V1,
    });
  }
  const receipt = sampleReceipt(sample, cursor);
  if (
    receipt.drawIndex !== fixture.drawIndex.toString() ||
    receipt.drawEnd !== (fixture.drawIndex + 1).toString()
  ) throw new Error(`OPT-0085 ${path.id} Philox cursor changed`);
  return receipt;
}

function resolveCotAllowedTokens(
  fixture: PlannerFixture,
  logits: Float32Array,
): AcePlannerAllowedTokens {
  const controller = new AcePlannerMetadataConstraintController({
    tokenizer: fixture.tokenizer,
  });
  const emitted: number[] = [];
  const teacherLogits = new Float32Array(
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
  );
  teacherLogits.fill(-1);
  for (
    let step = 0;
    step < fixture.emittedTokenIdsIncludingDecode.length;
    step += 1
  ) {
    const tokenId = fixture.emittedTokenIdsIncludingDecode[step]!;
    teacherLogits[tokenId] = 1;
    controller.allowedTokens({
      step,
      promptTokenIds: fixture.promptTokenIds,
      emittedTokenIds: emitted,
      logits: teacherLogits,
    });
    emitted.push(tokenId);
    controller.acceptToken({
      step,
      tokenId,
      tokenText: fixture.tokenizer.decode([tokenId]),
      emittedTokenIds: emitted,
    });
    teacherLogits[tokenId] = -1;
  }
  return controller.allowedTokens({
    step: emitted.length,
    promptTokenIds: fixture.promptTokenIds,
    emittedTokenIds: emitted,
    logits,
  });
}

function semanticPreCfgAllowedTokens(): AcePlannerAllowedTokens {
  return Object.freeze({
    kind: "range" as const,
    firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
    additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
  });
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

function createFixtures(
  tokenizer: AceQwenBpeTokenizer,
): ReadonlyMap<PlannerFixture["id"], PlannerFixture> {
  const cotPromptTokenIds = tokenizer.encode(
    createAcePlannerCotPrompt(ACCEPTED_PROMPT, ACCEPTED_LYRICS),
  );
  const cotContinuation = createCotTeacherTokens(
    tokenizer,
    cotPromptTokenIds,
  );
  const codePrompts = createAcePlannerCodePrompts(
    ACCEPTED_RESOLVED_CAPTION,
    ACCEPTED_LYRICS,
    ACCEPTED_COT_TEXT,
  );
  const semanticBaseRows = Object.freeze([
    tokenizer.encode(codePrompts.conditional),
    tokenizer.encode(codePrompts.unconditional),
  ]);
  if (
    cotPromptTokenIds.length !== 105 ||
    cotContinuation.length !== 109 ||
    semanticBaseRows[0]!.length !== 253 ||
    semanticBaseRows[1]!.length !== 33 ||
    sha256U32Le(cotContinuation) !== ACCEPTED_COT_TRAJECTORY_SHA256 ||
    sha256U32Le(ACCEPTED_SEMANTIC_CODE_IDS) !==
      ACCEPTED_SEMANTIC_CODE_SHA256
  ) throw new Error("OPT-0085 pinned fixture identity changed");

  const cot = createFixture({
    id: "cot-m1-middle",
    baseRows: [cotPromptTokenIds],
    continuation: cotContinuation,
    cachedTokensBeforeAppend: 160,
    cacheCapacity: 1_024,
    drawIndex: 56,
    tokenizer,
  });
  const semantic = createFixture({
    id: "semantic-m2-middle",
    baseRows: semanticBaseRows,
    continuation: ACCEPTED_SEMANTIC_TOKEN_IDS,
    cachedTokensBeforeAppend: 328,
    cacheCapacity: 1_280,
    drawIndex: 185,
    tokenizer,
  });
  return new Map([
    [cot.id, cot],
    [semantic.id, semantic],
  ]);
}

function createFixture(options: Readonly<{
  id: PlannerFixture["id"];
  baseRows: readonly (readonly number[])[];
  continuation: readonly number[];
  cachedTokensBeforeAppend: number;
  cacheCapacity: number;
  drawIndex: number;
  tokenizer: AceQwenBpeTokenizer;
}>): PlannerFixture {
  const prefill = createPaddedPrefill(
    options.baseRows,
    options.continuation,
    options.cachedTokensBeforeAppend,
    options.cacheCapacity,
  );
  const baseWidth = Math.max(...options.baseRows.map((row) => row.length));
  const continuationIndex = options.cachedTokensBeforeAppend - baseWidth;
  const decodeTokenId = options.continuation[continuationIndex];
  if (decodeTokenId === undefined) {
    throw new Error(`OPT-0085 ${options.id} leaves its pinned trajectory`);
  }
  const decode = createDecodeBatch(
    options.baseRows.length as 1 | 2,
    options.cacheCapacity,
    options.cachedTokensBeforeAppend,
    decodeTokenId,
  );
  return Object.freeze({
    id: options.id,
    rows: options.baseRows.length as 1 | 2,
    prefill,
    decode,
    seenTokenIdsIncludingDecode: Object.freeze([
      ...prefill.inputIds.slice(0, options.cachedTokensBeforeAppend),
      decodeTokenId,
    ]),
    drawIndex: options.drawIndex,
    promptTokenIds: Object.freeze([...options.baseRows[0]!]),
    emittedTokenIdsIncludingDecode: Object.freeze([
      ...options.continuation.slice(0, continuationIndex + 1),
    ]),
    tokenizer: options.tokenizer,
  });
}

function createPaddedPrefill(
  baseRows: readonly (readonly number[])[],
  continuation: readonly number[],
  tokens: number,
  cacheCapacity: number,
): AcePlannerPrefillBatch {
  const rows = baseRows.length;
  const baseWidth = Math.max(...baseRows.map((row) => row.length));
  if (
    (rows !== 1 && rows !== 2) ||
    baseWidth >= tokens ||
    tokens >= cacheCapacity ||
    tokens - baseWidth >= continuation.length
  ) throw new Error("OPT-0085 prefill geometry is invalid");
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
    batch: rows as 1 | 2,
    tokens,
    cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(0),
    validKeyLengths: Array<number>(rows).fill(tokens),
    sourceValidity: [...keyValidity],
  });
  return Object.freeze({
    kind: "prefill" as const,
    rows: rows as 1 | 2,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: rows === 2 ? 1 as const : null,
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
    kind: "decode" as const,
    rows,
    tokens: 1 as const,
    cacheCapacity,
    cachedTokensBeforeAppend,
    inputIds,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0 as const,
    unconditionalRow: rows === 2 ? 1 as const : null,
  });
}

function createCotTeacherTokens(
  tokenizer: AceQwenBpeTokenizer,
  promptTokenIds: readonly number[],
): readonly number[] {
  const closingTag = "</think>";
  if (!ACCEPTED_COT_TEXT.endsWith(closingTag)) {
    throw new Error("OPT-0085 accepted CoT closing tag changed");
  }
  let remaining = ACCEPTED_COT_TEXT.slice(0, -closingTag.length);
  const controller = new AcePlannerMetadataConstraintController({ tokenizer });
  const accepted: number[] = [];
  const logits = new Float32Array(ACE_PLANNER_QWEN3_CONFIG.vocabularySize);
  logits.fill(-1);
  while (remaining.length > 0) {
    const candidates: Array<Readonly<{ id: number; text: string }>> = [];
    for (
      let length = 1;
      length <= Math.min(96, remaining.length);
      length += 1
    ) {
      const text = remaining.slice(0, length);
      const tokenIds = tokenizer.encode(text);
      if (tokenIds.length === 1 && tokenizer.decode(tokenIds) === text) {
        candidates.push(Object.freeze({ id: tokenIds[0]!, text }));
      }
    }
    for (const candidate of candidates) {
      logits[candidate.id] = candidate.text.length;
    }
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
      throw new Error(`OPT-0085 CoT teacher cannot advance at step ${step}`);
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
    terminal.kind !== "ids" ||
    terminal.tokenIds.length !== 1 ||
    terminal.tokenIds[0] !== ACE_QWEN_IM_END_TOKEN_ID
  ) throw new Error("OPT-0085 CoT teacher did not reach terminal EOS");
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
  ) throw new Error("OPT-0085 reference manifest identity changed");
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
  ) throw new Error("OPT-0085 bounded planner acquisition changed");
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
  files: readonly AcePackageFileRecord[];
  tensorCount: number;
  weightFileCount: number;
  residentBytes: number;
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
    "OPT-0085 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0085 planner tensor bytes",
  );
  if (
    tensors.length !== PLANNER_TENSOR_COUNT ||
    weightNames.size !== PLANNER_WEIGHT_FILE_COUNT ||
    weightFiles.length !== PLANNER_WEIGHT_FILE_COUNT ||
    files.length !== ACQUIRED_FILE_COUNT ||
    residentBytes !== PLANNER_RESIDENT_BYTES ||
    tensorBytes !== PLANNER_RESIDENT_BYTES ||
    TOKENIZER_FILE_NAMES.some(
      (name) => !files.some((file) => file.name === name),
    )
  ) throw new Error("OPT-0085 reference planner inventory changed");
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    weightFileCount: weightFiles.length,
    residentBytes,
  });
}

function requireFloat32Rows(
  rows: readonly ArrayLike<number>[],
  expectedRows: 1 | 2,
  expectedColumns: number,
  label: string,
): readonly Float32Array[] {
  if (rows.length !== expectedRows) {
    throw new Error(`${label} returned ${rows.length} rows`);
  }
  return Object.freeze(rows.map((row, index) => {
    if (!(row instanceof Float32Array) || row.length !== expectedColumns) {
      throw new Error(`${label} row ${index} has the wrong storage or extent`);
    }
    return row;
  }));
}

function requireExactRows(
  expectedRows: readonly Float32Array[],
  actualRows: readonly Float32Array[],
  label: string,
): ExactReceipt {
  if (expectedRows.length !== actualRows.length) {
    throw new Error(`${label} row count changed`);
  }
  let elements = 0;
  let nanCount = 0;
  for (let rowIndex = 0; rowIndex < expectedRows.length; rowIndex += 1) {
    const expected = expectedRows[rowIndex]!;
    const actual = actualRows[rowIndex]!;
    if (expected.length !== actual.length) {
      throw new Error(`${label} row ${rowIndex} extent changed`);
    }
    const expectedBits = new Uint32Array(
      expected.buffer,
      expected.byteOffset,
      expected.length,
    );
    const actualBits = new Uint32Array(
      actual.buffer,
      actual.byteOffset,
      actual.length,
    );
    for (let index = 0; index < expected.length; index += 1) {
      if (Number.isNaN(expected[index]) || Number.isNaN(actual[index])) {
        nanCount += 1;
      }
      if (actualBits[index] !== expectedBits[index]) {
        throw new Error(
          `${label} raw-F32 mismatch row ${rowIndex}, column ${index}: ` +
            `0x${actualBits[index]!.toString(16).padStart(8, "0")} != ` +
            `0x${expectedBits[index]!.toString(16).padStart(8, "0")}`,
        );
      }
      elements += 1;
    }
  }
  if (nanCount !== 0) throw new Error(`${label} contains ${nanCount} NaNs`);
  return Object.freeze({
    rows: expectedRows.length as 1 | 2,
    elements,
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
      `${label} sample changed: ${JSON.stringify(actual)} != ` +
        JSON.stringify(expected),
    );
  }
}

function publicArm(execution: ArmExecution): Readonly<Record<string, unknown>> {
  return Object.freeze({
    arm: execution.arm,
    schedulingProfile: execution.profile,
    prefillWallMilliseconds: execution.prefillWallMilliseconds,
    modelWallMilliseconds: execution.modelWallMilliseconds,
    samplingWallMilliseconds: execution.samplingWallMilliseconds,
    completeWallMilliseconds: execution.completeWallMilliseconds,
    rowSha256: execution.rowSha256,
    sample: execution.sample,
    diagnostics: execution.diagnostics,
    progress: execution.progress,
  });
}

function cacheWriteStatusReceipt(rows: 1 | 2): Readonly<Record<string, unknown>> {
  return Object.freeze({
    expectedWords: Object.freeze(Array<number>(rows).fill(1)),
    returnedWords: Object.freeze(Array<number>(rows).fill(1)),
    exact: true,
    validationOwner:
      "planner executor rejects before returning logits unless every cache/write status word is one",
  });
}

function rowSha256(row: Float32Array): string {
  return aceSha256Hex(new Uint8Array(row.buffer, row.byteOffset, row.byteLength));
}

function sha256U32Le(values: readonly number[]): string {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function summarizeTimings(
  values: readonly number[],
): Readonly<Record<string, unknown>> {
  if (
    values.length !== OPT_0085_PAIR_ORDERS.length ||
    values.some((value) => !Number.isFinite(value) || value < 0)
  ) throw new Error("OPT-0085 timing sample set changed");
  return Object.freeze({
    count: values.length,
    minimumMilliseconds: Math.min(...values),
    medianMilliseconds: median(values),
    maximumMilliseconds: Math.max(...values),
    valuesMilliseconds: Object.freeze([...values]),
  });
}

function publicPreparation(
  prepared: PreparedSession,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    ...prepared.preparation,
    package: prepared.preparedPackage.receipt,
    tokenizer: prepared.tokenizerIdentity,
    oneAuthenticatedPlannerWeightOwner: true,
    onePlannerExecutor: true,
    exactWarmups: prepared.warmup,
    warmupCompletedAtEpochMilliseconds:
      prepared.warmupCompletedAtEpochMilliseconds,
  });
}

function validateStaticProtocol(): void {
  if (
    PATHS.length !== OPT_0085_PATH_IDS.length ||
    PATHS.some((path, index) => path.id !== OPT_0085_PATH_IDS[index]) ||
    OPT_0085_PAIR_ORDERS.length !== 4 ||
    OPT_0085_PAIR_ORDERS.filter((order) => order[0] === "control").length !== 2 ||
    OPT_0085_PAIR_ORDERS.filter((order) => order[0] === "candidate").length !== 2 ||
    ACCEPTED_SEMANTIC_CODE_IDS.length !== 150
  ) throw new Error("OPT-0085 frozen browser protocol changed");
}

function startWorkerHeartbeat(): { stop(): WorkerHeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  let timerTickCount = 0;
  let maximumTimerGapMilliseconds = 0;
  let last = performance.now();
  let stopped = false;
  const handle = setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - last,
    );
    last = now;
    timerTickCount += 1;
  }, WORKER_HEARTBEAT_INTERVAL_MILLISECONDS);
  return {
    stop(): WorkerHeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
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

function validateWorkerHeartbeat(heartbeat: WorkerHeartbeatSnapshot): void {
  if (
    heartbeat.timerTickCount < 1 ||
    !Number.isFinite(heartbeat.maximumTimerGapMilliseconds) ||
    heartbeat.maximumTimerGapMilliseconds < 0 ||
    heartbeat.completedAtEpochMilliseconds <
      heartbeat.startedAtEpochMilliseconds
  ) throw new Error("OPT-0085 worker heartbeat telemetry changed");
}

async function failAndCleanup(
  error: unknown,
  active: PreparedSession | undefined = session,
): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  session = undefined;
  active?.router.abandon();
  let cleanupError: unknown;
  if (active !== undefined) {
    try {
      await active.executor.destroy(error);
    } catch (caught) {
      cleanupError = caught;
    } finally {
      active.context.destroy();
    }
  }
  const heartbeat = workerHeartbeat?.stop();
  workerHeartbeat = undefined;
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      primary: errorValue(error),
      ...(cleanupError === undefined ? {} : { cleanup: errorValue(cleanupError) }),
      ...(heartbeat === undefined ? {} : { workerHeartbeat: heartbeat }),
    }),
  });
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0085 package omitted ${name}`);
  return file;
}

function requireMapValue<K, V>(map: ReadonlyMap<K, V>, key: K): V {
  const value = map.get(key);
  if (value === undefined) throw new Error(`OPT-0085 map omitted ${String(key)}`);
  return value;
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
      ...(error.cause === undefined ? {} : { cause: String(error.cause) }),
    });
  }
  return Object.freeze({ name: "Error", message: String(error), stack: null });
}
