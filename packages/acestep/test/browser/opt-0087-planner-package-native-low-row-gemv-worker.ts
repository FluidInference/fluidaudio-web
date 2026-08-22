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
  type AcePlannerOpt0087InvocationDiagnostics,
} from "../../src/webgpu/planner-executor.js";
import type {
  AceOpt0087PlannerDenseArm,
} from "../../src/webgpu/planner-dense-owner.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
} from "../../src/webgpu/qwen3.js";
import {
  OPT_0087_PAIR_ORDERS,
  OPT_0087_PATH_IDS,
  OPT_0087_SCHEMA,
  OPT_0087_COLD_GENERIC_ARM_MAP_COUNT,
  OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT,
  OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES,
  OPT_0087_COMPILE_CACHE_CONTROL_BUFFER_BYTES,
  OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT,
  Opt0087ResourcePairTopologyError,
  Opt0087ResourceTopologyError,
  evaluateOpt0087TimingGate,
  median,
  opt0087DenseArmForArm,
  validateOpt0087ExplicitArmResources,
  validateOpt0087ResourcePair,
  validateOpt0087ThermalLaunch,
  validateOpt0087RunIdentity,
  validateOpt0087Topology,
  type Opt0087Arm,
  type Opt0087ExplicitArmResourceDelta,
  type Opt0087ExplicitArmResourceExpectation,
  type Opt0087PathId,
  type Opt0087RunIdentity,
  type Opt0087ThermalLaunch,
} from "./opt-0087-planner-package-native-low-row-gemv-contract.js";

const MANIFEST_PATH = "/model/files-reference/manifest.json";
const PLANNER_TENSOR_COUNT = 314;
const PLANNER_WEIGHT_FILE_COUNT = 33;
const PLANNER_RESIDENT_BYTES = 1_325_768_704;
const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const WORKER_HEARTBEAT_INTERVAL_MILLISECONDS = 10;
const LOGICAL_LAYER_WEIGHT_BYTES_PER_ROW = 880_932_864;
const LOGICAL_TIED_HEAD_WEIGHT_BYTES_PER_ROW = 444_833_792;
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
const SAMPLING_PARAMETERS: AcePlannerSamplingParameters = Object.freeze({
  temperature: DEFAULT_ACE_PLANNER_CONFIGURATION.temperature,
  guidanceScale: DEFAULT_ACE_PLANNER_CONFIGURATION.guidanceScale,
  topK: DEFAULT_ACE_PLANNER_CONFIGURATION.topK,
  topP: DEFAULT_ACE_PLANNER_CONFIGURATION.topP,
  repetitionPenalty: 1,
});

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0087RunIdentity;
}

interface RunTimedMessage {
  readonly type: "run-timed";
  readonly thermalLaunch: Opt0087ThermalLaunch;
}

interface CancelMessage { readonly type: "cancel" }

type IncomingMessage = InitializeMessage | RunTimedMessage | CancelMessage;

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
  readonly id: Opt0087PathId;
  readonly fixtureId: PlannerFixture["id"];
  readonly sampleMode: "cot-full" | "semantic-full";
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
  readonly nonFiniteCount: 0;
}

interface ArmExecution {
  readonly arm: Opt0087Arm;
  readonly denseArm: AceOpt0087PlannerDenseArm;
  readonly rows: readonly Float32Array[];
  readonly sample: SampleReceipt;
  readonly diagnostics: AcePlannerOpt0087InvocationDiagnostics;
  readonly progress: Readonly<Record<string, unknown>>;
  readonly prefillWallMilliseconds: number;
  readonly modelThroughReadbackWallMilliseconds: number;
  readonly dispatchInclusiveDecodeWallMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly recurringCompleteTokenWallMilliseconds: number;
  readonly dispatchInclusiveCompleteTokenWallMilliseconds: number;
  readonly rowSha256: readonly string[];
  readonly resourceExpectation: Opt0087ExplicitArmResourceExpectation;
  readonly resources: Opt0087ExplicitArmResourceDelta;
}

interface PreparedSession {
  readonly identity: Opt0087RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly abortController: AbortController;
  readonly router: EvidenceRouter;
  readonly resourceTracker: Opt0087GpuResourceTracker;
  readonly executor: AcePlannerGpuExecutor;
  readonly fixtures: ReadonlyMap<PlannerFixture["id"], PlannerFixture>;
  readonly preparedPackage: PreparedPackage;
  readonly tokenizerIdentity: Readonly<Record<string, unknown>>;
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly warmup: readonly Readonly<Record<string, unknown>>[];
  readonly warmupCompletedAtEpochMilliseconds: number;
  cleanup(reason?: unknown): Promise<Readonly<Record<string, unknown>>>;
}

interface WorkerHeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly timerTickCount: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface Opt0087GpuResourceSnapshot {
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly liveBufferCount: number;
  readonly createdByteLength: number;
  readonly destroyedByteLength: number;
  readonly liveByteLength: number;
  readonly maximumLiveBufferCount: number;
  readonly maximumLiveByteLength: number;
  readonly successfulMapCount: number;
  readonly failedMapCount: number;
  readonly unmapCount: number;
  readonly activeMapCount: number;
  readonly destroyedWhileMappedCount: number;
  readonly repeatedDestroyCallCount: number;
}

interface Opt0087TrackedGpuBuffer {
  readonly byteLength: number;
  mapped: boolean;
  destroyed: boolean;
}

const PATHS: readonly PathSpec[] = Object.freeze([
  Object.freeze({
    id: "cot-m1-middle-full",
    fixtureId: "cot-m1-middle",
    sampleMode: "cot-full",
  }),
  Object.freeze({
    id: "semantic-m2-middle-full",
    fixtureId: "semantic-m2-middle",
    sampleMode: "semantic-full",
  }),
]);

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let session: PreparedSession | undefined;
let activeAbortController: AbortController | undefined;
let workerHeartbeat: ReturnType<typeof startWorkerHeartbeat> | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "cancel") {
    const reason = new DOMException(
      "OPT-0087 cancellation requested",
      "AbortError",
    );
    activeAbortController?.abort(reason);
    if (lifecycle === "ready" && session !== undefined) {
      void failAndCleanup(reason, session);
    }
    return;
  }
  if (event.data.type === "initialize") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    workerHeartbeat = startWorkerHeartbeat();
    const abortController = new AbortController();
    activeAbortController = abortController;
    void initializeSession(event.data.identity, abortController).then(
      (prepared) => {
        if (lifecycle !== "preparing") return;
        if (abortController.signal.aborted) {
          void failAndCleanup(abortController.signal.reason, prepared);
          return;
        }
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
    void runTimedAndCleanup(active, event.data.thermalLaunch).then(
      (result) => {
        lifecycle = "settled";
        session = undefined;
        activeAbortController = undefined;
        self.postMessage({ type: "awaiting-through-cleanup-thermal", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
});

async function initializeSession(
  identity: unknown,
  abortController: AbortController,
): Promise<PreparedSession> {
  const runIdentity = validateOpt0087RunIdentity(identity);
  validateStaticProtocol();
  const resourceTracker = new Opt0087GpuResourceTracker();
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
    const trackedDevice = resourceTracker.wrapDevice(context.device);

    postProgress("uploading one authenticated 1.235 GiB planner owner");
    const uploadStarted = performance.now();
    let lastUploadStatusAt = 0;
    phase = await AceGpuTensorPhase.load(
      trackedDevice,
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
    ) throw new Error("OPT-0087 loaded planner phase identity changed");
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
    executor = AcePlannerGpuExecutor.createForOpt0087({
      device: trackedDevice,
      modelProfile: "reference-bf16",
      ownedPlannerWeights,
      signal: abortController.signal,
      onProgress: (progress) => router.acceptProgress(progress),
      onOpt0087Invocation: (diagnostics) =>
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
        resourceTracker,
        abortController.signal,
        path,
        fixture,
        "control",
        "cold-generic-a-compile-cache",
      );
      const candidate = await executeArm(
        executor,
        router,
        resourceTracker,
        abortController.signal,
        path,
        fixture,
        "candidate",
        "no-new-compile-cache-buffer",
      );
      const exact = requireExactRows(
        control.rows,
        candidate.rows,
        `${path.id} warmup`,
      );
      requireSameSample(control.sample, candidate.sample, `${path.id} warmup`);
      requireSameTopology(control, candidate, `${path.id} warmup`);
      warmup.push(Object.freeze({
        pathId: path.id,
        rawLogits: exact,
        cacheWriteStatus: cacheWriteStatusReceipt(
          control.diagnostics,
          candidate.diagnostics,
          fixture.rows,
        ),
        cacheAppend: requireExactCacheAppend(
          control.diagnostics,
          candidate.diagnostics,
          `${path.id} warmup`,
        ),
        sample: control.sample,
        control: publicArm(control),
        candidate: publicArm(candidate),
      }));
    }
    const executorCompileAndWarmupWallMilliseconds =
      performance.now() - compileStarted;
    if (runtimeEvents.length !== 0) {
      throw new Error("OPT-0087 warmup observed a WebGPU runtime event");
    }
    let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
    const ownedExecutor = executor;
    const ownedContext = context;
    const cleanup = (
      reason?: unknown,
    ): Promise<Readonly<Record<string, unknown>>> => {
      if (cleanupPromise !== undefined) {
        return cleanupPromise.then((receipt) => Object.freeze({
          ...receipt,
          repeatedCall: true,
        }));
      }
      cleanupPromise = (async () => {
        const cleanupStartedAtEpochMilliseconds = Date.now();
        const failures: unknown[] = [];
        for (let call = 0; call < 2; call += 1) {
          try {
            await ownedExecutor.destroy(reason);
          } catch (error) {
            failures.push(error);
          }
        }
        let gpuBuffers: Opt0087GpuResourceSnapshot | undefined;
        try {
          gpuBuffers = resourceTracker.requireBalancedAfterCleanup();
        } catch (error) {
          failures.push(error);
        } finally {
          for (let call = 0; call < 2; call += 1) {
            try {
              ownedContext.destroy();
            } catch (error) {
              failures.push(error);
            }
          }
        }
        const cleanupCompletedAtEpochMilliseconds = Date.now();
        if (failures.length > 0 || gpuBuffers === undefined) {
          throw new Error(
            `OPT-0087 owner cleanup failed: ${failures.map(errorText).join("; ")}`,
            { cause: failures[0] },
          );
        }
        return Object.freeze({
          cleanupStartedAtEpochMilliseconds,
          cleanupCompletedAtEpochMilliseconds,
          executorDestroyCalledTwice: true,
          deviceDestroyCalledTwice: true,
          idempotentOwnerCleanup: true,
          runtimeEventCount: runtimeEvents.length,
          gpuBuffers,
          repeatedCall: false,
        });
      })();
      return cleanupPromise;
    };
    return Object.freeze({
      identity: runIdentity,
      context,
      runtimeEvents,
      abortController,
      router,
      resourceTracker,
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
      cleanup,
    });
  } catch (error) {
    const cleanupFailures: unknown[] = [];
    try {
      if (executor !== undefined) await executor.destroy(error);
      else phase?.destroy();
      resourceTracker.requireBalancedAfterCleanup();
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
        `OPT-0087 preparation failed (${errorText(error)}); cleanup also ` +
          `failed: ${cleanupFailures.map(errorText).join("; ")}`,
        { cause: error },
      );
    }
    throw error;
  }
}

async function runTimedAndCleanup(
  prepared: PreparedSession,
  thermalLaunch: Opt0087ThermalLaunch,
): Promise<Readonly<Record<string, unknown>>> {
  validateOpt0087ThermalLaunch(
    thermalLaunch,
    prepared.warmupCompletedAtEpochMilliseconds,
  );
  prepared.abortController.signal.throwIfAborted();
  const timedStartedAtEpochMilliseconds = Date.now();
  if (timedStartedAtEpochMilliseconds <
      thermalLaunch.gateCompletedAtEpochMilliseconds) {
    throw new Error("OPT-0087 timing began before its nominal thermal gate");
  }
  const paths: Readonly<Record<string, unknown>>[] = [];
  const gateInputs: Array<Readonly<{
    id: Opt0087PathId;
    control: Readonly<{
      transformerLayerWallMilliseconds: readonly number[];
      tiedHeadWallMilliseconds: readonly number[];
      modelThroughReadbackWallMilliseconds: readonly number[];
      completeTokenWallMilliseconds: readonly number[];
    }>;
    candidate: Readonly<{
      transformerLayerWallMilliseconds: readonly number[];
      tiedHeadWallMilliseconds: readonly number[];
      modelThroughReadbackWallMilliseconds: readonly number[];
      completeTokenWallMilliseconds: readonly number[];
    }>;
  }>> = [];
  for (let pathIndex = 0; pathIndex < PATHS.length; pathIndex += 1) {
      const path = PATHS[pathIndex]!;
      const fixture = requireMapValue(prepared.fixtures, path.fixtureId);
      const timing = {
        control: emptyArmTimings(),
        candidate: emptyArmTimings(),
      };
      const pairs: Readonly<Record<string, unknown>>[] = [];
      let candidateWins = 0;
      for (
        let pairIndex = 0;
        pairIndex < OPT_0087_PAIR_ORDERS.length;
        pairIndex += 1
      ) {
        const order = OPT_0087_PAIR_ORDERS[pairIndex]!;
        const executions = new Map<Opt0087Arm, ArmExecution>();
        for (const arm of order) {
          prepared.abortController.signal.throwIfAborted();
          const execution = await executeArm(
            prepared.executor,
            prepared.router,
            prepared.resourceTracker,
            prepared.abortController.signal,
            path,
            fixture,
            arm,
            "no-new-compile-cache-buffer",
          );
          executions.set(arm, execution);
          await yieldToWorker();
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
        requireSameTopology(
          control,
          candidate,
          `${path.id} pair ${pairIndex}`,
        );
        appendArmTiming(timing.control, control);
        appendArmTiming(timing.candidate, candidate);
        if (
          candidate.modelThroughReadbackWallMilliseconds <
            control.modelThroughReadbackWallMilliseconds
        ) candidateWins += 1;
        pairs.push(Object.freeze({
          pairIndex,
          order: order.join("-"),
          rawLogits: exact,
          cacheWriteStatus: cacheWriteStatusReceipt(
            control.diagnostics,
            candidate.diagnostics,
            fixture.rows,
          ),
          cacheAppend: requireExactCacheAppend(
            control.diagnostics,
            candidate.diagnostics,
            `${path.id} pair ${pairIndex}`,
          ),
          sampleExact: true,
          sample: control.sample,
          control: publicArm(control),
          candidate: publicArm(candidate),
          candidateModelThroughReadbackSavingMilliseconds:
            control.modelThroughReadbackWallMilliseconds -
              candidate.modelThroughReadbackWallMilliseconds,
          candidateCompleteTokenSavingMilliseconds:
            control.recurringCompleteTokenWallMilliseconds -
              candidate.recurringCompleteTokenWallMilliseconds,
        }));
        postProgress(
          `${path.id}: timed pair ${pairIndex + 1}/` +
            `${OPT_0087_PAIR_ORDERS.length}`,
        );
        await yieldToWorker();
      }
      gateInputs.push(Object.freeze({
        id: path.id,
        control: freezeArmTimings(timing.control),
        candidate: freezeArmTimings(timing.candidate),
      }));
      paths.push(Object.freeze({
        id: path.id,
        fixtureId: path.fixtureId,
        rows: fixture.rows,
        cachedTokensBeforeAppend: fixture.decode.cachedTokensBeforeAppend,
        cacheCapacity: fixture.decode.cacheCapacity,
        drawIndex: fixture.drawIndex,
        logitExtent: Object.freeze({
          kind: "full-vocabulary",
          firstTokenId: 0,
          tokenCount: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
        }),
        expectedModelQuantumCount: 33,
        expectedTotalCommandBuffers: 34,
        candidateWins,
        control: summarizeArmTimings(timing.control, fixture.rows),
        candidate: summarizeArmTimings(timing.candidate, fixture.rows),
        medianModelThroughReadbackSavingMilliseconds:
          median(timing.control.modelThroughReadbackWallMilliseconds) -
            median(timing.candidate.modelThroughReadbackWallMilliseconds),
        medianCompleteTokenSavingMilliseconds:
          median(timing.control.completeTokenWallMilliseconds) -
            median(timing.candidate.completeTokenWallMilliseconds),
        pairs: Object.freeze(pairs),
      }));
    }
    const timedCompletedAtEpochMilliseconds = Date.now();
    const timingGate = evaluateOpt0087TimingGate(gateInputs);
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0087 timing observed a WebGPU runtime event");
    }

    postProgress("running post-timing package-boundary cancellation proof");
    const cancellationFixture = requireMapValue(
      prepared.fixtures,
      "cot-m1-middle",
    );
    await prepared.executor.prefill(cancellationFixture.prefill);
    const cancellation = await runCancellationProof(
      prepared,
      cancellationFixture,
    );
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("OPT-0087 cleanup observed a WebGPU runtime event");
    }
    const cleanupFirst = await prepared.cleanup();
    const cleanupSecond = await prepared.cleanup();
    const cleanupCompletedValue =
      cleanupFirst["cleanupCompletedAtEpochMilliseconds"];
    if (
      typeof cleanupCompletedValue !== "number" ||
      !Number.isSafeInteger(cleanupCompletedValue) ||
      cleanupFirst["idempotentOwnerCleanup"] !== true ||
      cleanupFirst["runtimeEventCount"] !== 0 ||
      cleanupSecond["repeatedCall"] !== true
    ) throw new Error("OPT-0087 paired owner cleanup receipt changed");
    const cleanupCompletedAtEpochMilliseconds = cleanupCompletedValue;
    const heartbeat = workerHeartbeat!.stop();
    workerHeartbeat = undefined;
    validateWorkerHeartbeat(heartbeat);
    return Object.freeze({
      schema: OPT_0087_SCHEMA,
      schemaVersion: 1,
      experimentId: "OPT-0087",
      statusBeforeFinalThermalJoin: timingGate.passed
        ? "candidate-passed-performance-gate"
        : "candidate-failed-performance-gate",
      identity: Object.freeze({
        ...prepared.identity,
        manifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
        modelProfile: "reference-bf16",
        schedulingProfile: "cooperative",
        plannerQueueDepth: 1,
        capabilities: prepared.context.capabilities,
      }),
      protocol: Object.freeze({
        benchmarkOnly: true,
        productionSelectorChanged: false,
        arithmeticChanged: false,
        commandBufferBoundariesChanged: false,
        terminalReadbackEvidenceCopiesAdded: true,
        terminalReadbackCommandBuffers: 1,
        opt0087CacheAppendEvidence: Object.freeze({
          allocationOwner: "paired OPT-0087 phase only",
          copyPlacement: "existing terminal readback command",
          additionalCommandBuffers: 0,
          additionalQueueDrains: 0,
          mapCountPerExplicitArm: 2,
          authoritativeWallIncludesCacheCopiesAndEvidenceMap: true,
        }),
        compileCacheDispatchConstructionResources: Object.freeze({
          allocationOwner:
            "planner-runtime shape compile cache retained across phase replay through owner cleanup",
          warmupInvocationOrder: Object.freeze([
            "M1-generic-A-cold",
            "M1-direct-B-no-new",
            "M2-generic-A-cold",
            "M2-direct-B-no-new",
          ]),
          firstColdGenericAByShape: Object.freeze({
            shapeOrder: Object.freeze(["M1", "M2"]),
            bufferCountPerShape:
              OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT,
            bytesPerBuffer: OPT_0087_COMPILE_CACHE_CONTROL_BUFFER_BYTES,
            totalBytesPerShape:
              OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES,
            shapeCount: 2,
            totalPersistentBufferCountAcrossWarmup:
              2 * OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT,
            totalPersistentBytesAcrossWarmup:
              2 * OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES,
            mappedAtCreationBufferCountPerShape:
              OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT,
            totalSuccessfulMapAndUnmapCountPerArm:
              OPT_0087_COLD_GENERIC_ARM_MAP_COUNT,
          }),
          directBAndEveryWarmedArm: Object.freeze({
            createdBufferCount: 0,
            createdByteLength: 0,
            evidenceReadbackMapCount: 2,
            totalSuccessfulMapAndUnmapCountPerArm:
              OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT,
          }),
          evidenceReadbackMapCount: 2,
          coldGenericAllocationsCompleteBeforeAuthoritativeModelWall: true,
          coldGenericAllocationsIncludedInOuterDispatchInclusiveWall: true,
          warmupResourceAsymmetryExpected: true,
          everyTimedPairRequiresExactNoNewAllocationResources: true,
          releasedOnlyAtPlannerRuntimeOwnerCleanup: true,
        }),
        schedulingChanged: false,
        samplerChanged: false,
        headExtentChanged: false,
        opt0082CompactHeadDisabled: true,
        opt0084SamplingDisabled: true,
        opt0085DepthTwoDisabled: true,
        opt0086DownstreamSchedulingDisabled: true,
        oneAuthenticatedPlannerWeightOwner: true,
        onePlannerExecutor: true,
        sameStateReplay:
          "untimed identical prefill before every timed arm on one executor",
        resourceAccounting:
          "per-arm deltas begin after excluded state replay; first cold generic-A M1 and M2 warmups each create five persistent 256-byte runtime compile-cache controls plus two evidence maps, direct-B and every warmed/timed arm create none and map only two evidence buffers, and owner-wide cleanup balances every retained allocation with zero live buffers",
        completeTokenWallDefinition:
          "executor model-through-readback after dispatch construction plus the disjoint selected-sampler interval",
        cotSamplerStatePreparation:
          "metadata FSM construction and trajectory replay are outside timing; only current-step constraint selection and production sampler are inside",
        dispatchConstructionTimingUse:
          "outer dispatch-inclusive decode/complete walls are disclosed separately and never enter the gate or projection",
        layerAndHeadWallDefinition:
          "disjoint performance.now intervals around unchanged encode, singleton submit, and terminal fence drain; cooperative idle excluded",
        perQuantumTimingUse:
          "disjoint package attribution only; never added to model-through-readback wall",
        setupAndPrefillExcludedFromTiming: true,
        balancedInterleaving: true,
        timingPairOrdersPerPath: OPT_0087_PAIR_ORDERS.map(
          (order) => order.join("-"),
        ),
        totalPairCount: gateInputs.reduce(
          (total, path) => total +
            path.control.modelThroughReadbackWallMilliseconds.length,
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
        pathOrder: OPT_0087_PATH_IDS,
      }),
      preparation: Object.freeze({
        ...prepared.preparation,
        warmupCompletedAtEpochMilliseconds:
          prepared.warmupCompletedAtEpochMilliseconds,
        exactWarmups: prepared.warmup,
      }),
      thermal: Object.freeze({
        launch: thermalLaunch,
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
        firstCleanup: cleanupFirst,
        secondCleanup: cleanupSecond,
        executorDestroyCalledTwice: true,
        executorDestroyed: true,
        deviceDestroyedAfterRuntimeEventCheck: true,
        actualGpuBufferCreateDestroyBalanced: true,
        noLiveGpuBuffersAfterCleanup: true,
        actualMapUnmapBalanced: true,
        outputPublishedOnlyAfterTerminalReadbackFence: true,
        cleanupCompletedAtEpochMilliseconds,
      }),
      workerHeartbeat: heartbeat,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      completedAt: new Date(cleanupCompletedAtEpochMilliseconds).toISOString(),
    });
}

async function executeArm(
  executor: AcePlannerGpuExecutor,
  router: EvidenceRouter,
  resourceTracker: Opt0087GpuResourceTracker,
  signal: AbortSignal,
  path: PathSpec,
  fixture: PlannerFixture,
  arm: Opt0087Arm,
  resourceExpectation: Opt0087ExplicitArmResourceExpectation,
): Promise<ArmExecution> {
  signal.throwIfAborted();
  const denseArm = opt0087DenseArmForArm(arm);
  const prefillStarted = performance.now();
  // Recreate identical cache state before every timed arm. This generic
  // control prefill is intentionally outside all authoritative gate walls.
  await executor.prefill(fixture.prefill);
  signal.throwIfAborted();
  const prefillWallMilliseconds = performance.now() - prefillStarted;
  const cotController = path.sampleMode === "cot-full"
    ? prepareCotConstraintController(fixture)
    : undefined;
  const resourcesBeforeArm = resourceTracker.snapshot();
  router.begin(denseArm, fixture.rows);
  const completeStarted = performance.now();
  let rows: readonly Float32Array[];
  try {
    rows = requireFloat32Rows(
      await executor.decodeForOpt0087(
        denseArm,
        fixture.decode,
      ),
      fixture.rows,
      ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
      `${path.id} ${arm}`,
    );
  } catch (error) {
    router.abandon();
    throw error;
  }
  const modelEnded = performance.now();
  const sample = samplePath(path, fixture, rows, cotController);
  const samplingEnded = performance.now();
  const captured = router.end(denseArm, fixture.rows);
  const resourcesAfterArm = resourceTracker.snapshot();
  return Object.freeze({
    arm,
    denseArm,
    rows,
    sample,
    diagnostics: captured.diagnostics,
    progress: captured.progress,
    prefillWallMilliseconds,
    modelThroughReadbackWallMilliseconds:
      captured.diagnostics.modelThroughReadbackWallMilliseconds,
    dispatchInclusiveDecodeWallMilliseconds: modelEnded - completeStarted,
    samplingWallMilliseconds: samplingEnded - modelEnded,
    recurringCompleteTokenWallMilliseconds:
      captured.diagnostics.modelThroughReadbackWallMilliseconds +
        (samplingEnded - modelEnded),
    dispatchInclusiveCompleteTokenWallMilliseconds:
      samplingEnded - completeStarted,
    rowSha256: Object.freeze(rows.map(rowSha256)),
    resourceExpectation,
    resources: opt0087ResourceDelta(
      resourcesBeforeArm,
      resourcesAfterArm,
      resourceExpectation,
    ),
  });
}

async function runCancellationProof(
  prepared: PreparedSession,
  fixture: PlannerFixture,
): Promise<Readonly<Record<string, unknown>>> {
  const reason = new DOMException(
    "OPT-0087 post-timing package boundary cancellation proof",
    "AbortError",
  );
  prepared.router.beginCancellation(reason);
  const started = performance.now();
  let rejection: unknown;
  try {
    await prepared.executor.decodeForOpt0087(
      "direct-b",
      fixture.decode,
    );
  } catch (error) {
    rejection = error;
  }
  const wallMilliseconds = performance.now() - started;
  const captured = prepared.router.endCancellation();
  if (
    rejection !== reason ||
    !prepared.abortController.signal.aborted ||
    captured.progressCount !== 1 ||
    captured.diagnosticsCount !== 0 ||
    wallMilliseconds > 10_000
  ) throw new Error("OPT-0087 actual-browser cancellation proof changed");
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
    wallMilliseconds,
    maximumAllowedWallMilliseconds: 10_000,
    bounded: true,
  });
}

/** Browser-harness-only accounting over every buffer made by the paired owner. */
class Opt0087GpuResourceTracker {
  private readonly buffers = new Map<GPUBuffer, Opt0087TrackedGpuBuffer>();
  private createdBufferCount = 0;
  private destroyedBufferCount = 0;
  private createdByteLength = 0;
  private destroyedByteLength = 0;
  private liveByteLength = 0;
  private maximumLiveBufferCount = 0;
  private maximumLiveByteLength = 0;
  private successfulMapCount = 0;
  private failedMapCount = 0;
  private unmapCount = 0;
  private activeMapCount = 0;
  private destroyedWhileMappedCount = 0;
  private repeatedDestroyCallCount = 0;
  private wrappedDevice: GPUDevice | undefined;

  wrapDevice(device: GPUDevice): GPUDevice {
    if (this.wrappedDevice !== undefined) return this.wrappedDevice;
    const tracker = this;
    this.wrappedDevice = new Proxy(device, {
      get(target, property) {
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer =>
            tracker.trackBuffer(target.createBuffer(descriptor), descriptor);
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    }) as GPUDevice;
    return this.wrappedDevice;
  }

  snapshot(): Opt0087GpuResourceSnapshot {
    return Object.freeze({
      createdBufferCount: this.createdBufferCount,
      destroyedBufferCount: this.destroyedBufferCount,
      liveBufferCount: this.buffers.size,
      createdByteLength: this.createdByteLength,
      destroyedByteLength: this.destroyedByteLength,
      liveByteLength: this.liveByteLength,
      maximumLiveBufferCount: this.maximumLiveBufferCount,
      maximumLiveByteLength: this.maximumLiveByteLength,
      successfulMapCount: this.successfulMapCount,
      failedMapCount: this.failedMapCount,
      unmapCount: this.unmapCount,
      activeMapCount: this.activeMapCount,
      destroyedWhileMappedCount: this.destroyedWhileMappedCount,
      repeatedDestroyCallCount: this.repeatedDestroyCallCount,
    });
  }

  requireBalancedAfterCleanup(): Opt0087GpuResourceSnapshot {
    const receipt = this.snapshot();
    if (
      receipt.createdBufferCount !== receipt.destroyedBufferCount ||
      receipt.createdByteLength !== receipt.destroyedByteLength ||
      receipt.liveBufferCount !== 0 ||
      receipt.liveByteLength !== 0 ||
      receipt.successfulMapCount !== receipt.unmapCount ||
      receipt.failedMapCount !== 0 ||
      receipt.activeMapCount !== 0 ||
      receipt.destroyedWhileMappedCount !== 0 ||
      receipt.repeatedDestroyCallCount !== 0
    ) throw new Error("OPT-0087 paired GPU buffer ownership did not balance");
    return receipt;
  }

  private trackBuffer(
    buffer: GPUBuffer,
    descriptor: GPUBufferDescriptor,
  ): GPUBuffer {
    const byteLength = Number(descriptor.size);
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      buffer.destroy();
      throw new Error("OPT-0087 tracked GPU buffer has invalid size");
    }
    const state: Opt0087TrackedGpuBuffer = {
      byteLength,
      mapped: descriptor.mappedAtCreation === true,
      destroyed: false,
    };
    const originalDestroy = buffer.destroy.bind(buffer);
    const originalMapAsync = buffer.mapAsync.bind(buffer);
    const originalUnmap = buffer.unmap.bind(buffer);
    try {
      Object.defineProperties(buffer, {
        destroy: {
          configurable: true,
          value: () => {
            if (state.destroyed) {
              this.repeatedDestroyCallCount += 1;
              return;
            }
            originalDestroy();
            state.destroyed = true;
            this.destroyedBufferCount += 1;
            this.destroyedByteLength += state.byteLength;
            this.liveByteLength -= state.byteLength;
            this.buffers.delete(buffer);
            if (state.mapped) {
              state.mapped = false;
              this.activeMapCount -= 1;
              this.destroyedWhileMappedCount += 1;
            }
          },
        },
        mapAsync: {
          configurable: true,
          value: async (...args: Parameters<GPUBuffer["mapAsync"]>) => {
            try {
              await originalMapAsync(...args);
            } catch (error) {
              this.failedMapCount += 1;
              throw error;
            }
            state.mapped = true;
            this.successfulMapCount += 1;
            this.activeMapCount += 1;
          },
        },
        unmap: {
          configurable: true,
          value: () => {
            originalUnmap();
            if (state.mapped) {
              state.mapped = false;
              this.unmapCount += 1;
              this.activeMapCount -= 1;
            }
          },
        },
      });
    } catch (error) {
      originalDestroy();
      throw new Error(
        "OPT-0087 cannot instrument paired GPU buffer ownership",
        { cause: error },
      );
    }
    this.buffers.set(buffer, state);
    this.createdBufferCount += 1;
    this.createdByteLength += byteLength;
    this.liveByteLength += byteLength;
    if (state.mapped) {
      this.successfulMapCount += 1;
      this.activeMapCount += 1;
    }
    this.maximumLiveBufferCount = Math.max(
      this.maximumLiveBufferCount,
      this.buffers.size,
    );
    this.maximumLiveByteLength = Math.max(
      this.maximumLiveByteLength,
      this.liveByteLength,
    );
    return buffer;
  }
}

function opt0087ResourceDelta(
  before: Opt0087GpuResourceSnapshot,
  after: Opt0087GpuResourceSnapshot,
  expectation: Opt0087ExplicitArmResourceExpectation,
): Opt0087ExplicitArmResourceDelta {
  const delta = (field: keyof Opt0087GpuResourceSnapshot): number =>
    after[field] - before[field];
  const receipt: Opt0087ExplicitArmResourceDelta = Object.freeze({
    createdBufferCount: delta("createdBufferCount"),
    destroyedBufferCount: delta("destroyedBufferCount"),
    createdByteLength: delta("createdByteLength"),
    destroyedByteLength: delta("destroyedByteLength"),
    successfulMapCount: delta("successfulMapCount"),
    failedMapCount: delta("failedMapCount"),
    unmapCount: delta("unmapCount"),
    destroyedWhileMappedCount: delta("destroyedWhileMappedCount"),
    repeatedDestroyCallCount: delta("repeatedDestroyCallCount"),
    liveBufferCountBefore: before.liveBufferCount,
    liveBufferCountAfter: after.liveBufferCount,
    liveByteLengthBefore: before.liveByteLength,
    liveByteLengthAfter: after.liveByteLength,
    activeMapCountBefore: before.activeMapCount,
    activeMapCountAfter: after.activeMapCount,
  });
  validateOpt0087ExplicitArmResources(receipt, expectation);
  return receipt;
}

class EvidenceRouter {
  private mode: "none" | "capture" | "cancellation" = "none";
  private arm: AceOpt0087PlannerDenseArm | undefined;
  private rows: 1 | 2 | undefined;
  private progress: AcePlannerGpuExecutorProgress[] = [];
  private diagnostics: AcePlannerOpt0087InvocationDiagnostics[] = [];
  private cancellationReason: DOMException | undefined;
  private cancellationRequested = false;

  constructor(private readonly abortController: AbortController) {}

  begin(
    arm: AceOpt0087PlannerDenseArm,
    rows: 1 | 2,
  ): void {
    this.requireNone();
    this.mode = "capture";
    this.arm = arm;
    this.rows = rows;
    this.progress = [];
    this.diagnostics = [];
  }

  beginCancellation(reason: DOMException): void {
    this.requireNone();
    this.mode = "cancellation";
    this.arm = "direct-b";
    this.rows = 1;
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
    diagnostics: AcePlannerOpt0087InvocationDiagnostics,
  ): void {
    if (this.mode === "none") {
      throw new Error("OPT-0087 diagnostics escaped an explicit invocation");
    }
    this.diagnostics.push(Object.freeze({ ...diagnostics }));
  }

  end(
    expectedArm: AceOpt0087PlannerDenseArm,
    expectedRows: 1 | 2,
  ): Readonly<{
    diagnostics: AcePlannerOpt0087InvocationDiagnostics;
    progress: Readonly<Record<string, unknown>>;
  }> {
    if (
      this.mode !== "capture" ||
      this.arm !== expectedArm ||
      this.rows !== expectedRows ||
      this.diagnostics.length !== 1
    ) throw new Error("OPT-0087 evidence-router capture changed");
    const diagnostics = this.diagnostics[0]!;
    validateOpt0087Topology(diagnostics, expectedArm, expectedRows);
    const progress = validateProgress(
      this.progress,
      expectedArm,
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
      this.arm !== "direct-b" ||
      this.rows !== 1
    ) throw new Error("OPT-0087 cancellation router changed");
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
        "OPT-0087 evidence captures overlap",
        "InvalidStateError",
      );
    }
  }

  private reset(): void {
    this.mode = "none";
    this.arm = undefined;
    this.rows = undefined;
    this.progress = [];
    this.diagnostics = [];
    this.cancellationReason = undefined;
    this.cancellationRequested = false;
  }
}

function validateProgress(
  events: readonly AcePlannerGpuExecutorProgress[],
  arm: AceOpt0087PlannerDenseArm,
): Readonly<Record<string, unknown>> {
  const expectedTotal = 34;
  const expectedDrains = 34;
  const expectedIdleTurns = 34;
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
  ) throw new Error(`OPT-0087 ${arm} progress sequence changed`);
  return Object.freeze({
    callbackCount: events.length,
    firstCompletedCommandBuffers: events[0]!.completedCommandBuffers,
    finalCompletedCommandBuffers: events.at(-1)!.completedCommandBuffers,
    finalStage: events.at(-1)!.stage,
    finalQueueDrains: events.at(-1)!.queueDrains,
    finalCooperativeIdleMilliseconds: events.at(-1)!.cooperativeIdleMs,
    peakAccountedGpuBytes: events.at(-1)!.peakAccountedGpuBytes,
    completionOrderMonotonic: true,
    readbackProgressLast: true,
  });
}

function samplePath(
  path: PathSpec,
  fixture: PlannerFixture,
  logits: readonly Float32Array[],
  cotController?: AcePlannerMetadataConstraintController,
): SampleReceipt {
  const cursor = new AcePlannerSamplingCursor(ACCEPTED_SEED, fixture.drawIndex);
  let sample: AcePlannerCursorSample;
  if (path.sampleMode === "cot-full") {
    if (cotController === undefined) {
      throw new Error("OPT-0087 CoT sampling state was not prepared");
    }
    const allowedTokens = cotController.allowedTokens({
      step: fixture.emittedTokenIdsIncludingDecode.length,
      promptTokenIds: fixture.promptTokenIds,
      emittedTokenIds: fixture.emittedTokenIdsIncludingDecode,
      logits: logits[0]!,
    });
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
  } else {
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
  }
  const receipt = sampleReceipt(sample, cursor);
  if (
    receipt.drawIndex !== fixture.drawIndex.toString() ||
    receipt.drawEnd !== (fixture.drawIndex + 1).toString()
  ) throw new Error(`OPT-0087 ${path.id} Philox cursor changed`);
  return receipt;
}

function prepareCotConstraintController(
  fixture: PlannerFixture,
): AcePlannerMetadataConstraintController {
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
  return controller;
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
  ) throw new Error("OPT-0087 pinned fixture identity changed");

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
    throw new Error(`OPT-0087 ${options.id} leaves its pinned trajectory`);
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
  ) throw new Error("OPT-0087 prefill geometry is invalid");
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
    throw new Error("OPT-0087 accepted CoT closing tag changed");
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
      throw new Error(`OPT-0087 CoT teacher cannot advance at step ${step}`);
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
  ) throw new Error("OPT-0087 CoT teacher did not reach terminal EOS");
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
  ) throw new Error("OPT-0087 reference manifest identity changed");
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
  ) throw new Error("OPT-0087 bounded planner acquisition changed");
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
    "OPT-0087 planner resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0087 planner tensor bytes",
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
  ) throw new Error("OPT-0087 reference planner inventory changed");
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
  let nonFiniteCount = 0;
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
      if (!Number.isFinite(expected[index]) || !Number.isFinite(actual[index])) {
        nonFiniteCount += 1;
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
  if (nonFiniteCount !== 0) {
    throw new Error(`${label} contains ${nonFiniteCount} non-finite logits`);
  }
  return Object.freeze({
    rows: expectedRows.length as 1 | 2,
    elements,
    mismatchCount: 0,
    nonFiniteCount: 0,
  });
}

function requireExactCacheAppend(
  control: AcePlannerOpt0087InvocationDiagnostics,
  candidate: AcePlannerOpt0087InvocationDiagnostics,
  label: string,
): Readonly<Record<string, unknown>> {
  const expected = control.cacheAppendWords;
  const actual = candidate.cacheAppendWords;
  if (
    expected.length !== control.cacheAppendKeyValueWordCount +
      control.cacheAppendValidityWordCount ||
    actual.length !== candidate.cacheAppendKeyValueWordCount +
      candidate.cacheAppendValidityWordCount ||
    control.cacheAppendKeyValueWordCount !==
      candidate.cacheAppendKeyValueWordCount ||
    control.cacheAppendValidityWordCount !==
      candidate.cacheAppendValidityWordCount ||
    expected.length !== actual.length
  ) throw new Error(`${label} cache append extent changed`);
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      throw new Error(
        `${label} cache append raw-U32 mismatch at ${index}: ` +
          `0x${actual[index]!.toString(16).padStart(8, "0")} != ` +
          `0x${expected[index]!.toString(16).padStart(8, "0")}`,
      );
    }
  }
  const keyValueWords = control.cacheAppendKeyValueWordCount;
  const controlValidity = expected.subarray(keyValueWords);
  const candidateValidity = actual.subarray(keyValueWords);
  if (
    [...controlValidity].some((word) => word !== 1) ||
    [...candidateValidity].some((word) => word !== 1)
  ) throw new Error(`${label} appended cache-validity word changed`);
  return Object.freeze({
    order: "layer,K-or-V,physical-row,KV-head,dimension; then validity-row",
    keyValueWordCount: keyValueWords,
    validityWordCount: control.cacheAppendValidityWordCount,
    logicalByteLength: control.cacheAppendLogicalByteLength,
    mismatchCount: 0,
    exact: true,
    controlCompleteU32LeSha256: hashU32Storage(expected),
    candidateCompleteU32LeSha256: hashU32Storage(actual),
    controlKeyValueU32LeSha256:
      hashU32Storage(expected.subarray(0, keyValueWords)),
    candidateKeyValueU32LeSha256:
      hashU32Storage(actual.subarray(0, keyValueWords)),
    controlValidityU32LeSha256: hashU32Storage(controlValidity),
    candidateValidityU32LeSha256: hashU32Storage(candidateValidity),
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

function requireSameTopology(
  control: ArmExecution,
  candidate: ArmExecution,
  label: string,
): void {
  const stableDiagnostics = (execution: ArmExecution) => {
    const diagnostics = execution.diagnostics;
    return {
      modelQuantumCount: diagnostics.modelQuantumCount,
      totalCommandBuffers: diagnostics.totalCommandBuffers,
      commandBuffersSubmitted: diagnostics.commandBuffersSubmitted,
      trueQueueDrainCount: diagnostics.trueQueueDrainCount,
      cooperativeIdleTurns: diagnostics.cooperativeIdleTurns,
      requestedCooperativeIdleMs: diagnostics.requestedCooperativeIdleMs,
      maximumOutstandingCommandBuffers:
        diagnostics.maximumOutstandingCommandBuffers,
      readbackMapCount: diagnostics.readbackMapCount,
      readbackShardCount: diagnostics.readbackShardCount,
      readbackByteLength: diagnostics.readbackByteLength,
      cacheAppendReadbackByteLength:
        diagnostics.cacheAppendReadbackByteLength,
      cacheAppendLogicalByteLength:
        diagnostics.cacheAppendLogicalByteLength,
      cacheAppendCopyCount: diagnostics.cacheAppendCopyCount,
      cacheAppendKeyValueWordCount:
        diagnostics.cacheAppendKeyValueWordCount,
      cacheAppendValidityWordCount:
        diagnostics.cacheAppendValidityWordCount,
      logitRows: diagnostics.logitRows,
      logitTokenCount: diagnostics.logitTokenCount,
      accountedGpuBytes: diagnostics.accountedGpuBytes,
      arenaBufferCount: diagnostics.arenaBufferCount,
      quantumTopology: diagnostics.quantumTimings.map((timing) => ({
        index: timing.index,
        kind: timing.kind,
        layer: timing.layer,
        primitiveCount: timing.primitiveCount,
      })),
      denseTopology: diagnostics.denseSelections.map((selection) => ({
        role: selection.role,
        rows: selection.rows,
        inner: selection.inner,
        columns: selection.columns,
      })),
      headQuantumSliceFirstRows: diagnostics.headQuantumSliceFirstRows,
      peakAccountedGpuBytes: execution.progress.peakAccountedGpuBytes,
    };
  };
  if (
    JSON.stringify(stableDiagnostics(control)) !==
      JSON.stringify(stableDiagnostics(candidate))
  ) throw new Error(`${label} command/readback topology changed`);
  const warmedPair =
    control.resourceExpectation === "no-new-compile-cache-buffer" &&
    candidate.resourceExpectation === "no-new-compile-cache-buffer";
  const frozenWarmupAsymmetry =
    control.arm === "control" &&
    control.resourceExpectation === "cold-generic-a-compile-cache" &&
    candidate.arm === "candidate" &&
    candidate.resourceExpectation === "no-new-compile-cache-buffer";
  if (!warmedPair && !frozenWarmupAsymmetry) {
    throw new Error(`${label} resource expectation order changed`);
  }
  validateOpt0087ResourcePair(
    control.resources,
    candidate.resources,
    warmedPair ? "warmed-timed-pair" : "cold-warmup-a-to-direct-b",
  );
}

function publicArm(execution: ArmExecution): Readonly<Record<string, unknown>> {
  return Object.freeze({
    arm: execution.arm,
    denseArm: execution.denseArm,
    prefillWallMilliseconds: execution.prefillWallMilliseconds,
    modelThroughReadbackWallMilliseconds:
      execution.modelThroughReadbackWallMilliseconds,
    dispatchInclusiveDecodeWallMilliseconds:
      execution.dispatchInclusiveDecodeWallMilliseconds,
    samplingWallMilliseconds: execution.samplingWallMilliseconds,
    recurringCompleteTokenWallMilliseconds:
      execution.recurringCompleteTokenWallMilliseconds,
    dispatchInclusiveCompleteTokenWallMilliseconds:
      execution.dispatchInclusiveCompleteTokenWallMilliseconds,
    rowSha256: execution.rowSha256,
    sample: execution.sample,
    resourceExpectation: execution.resourceExpectation,
    diagnostics: publicOpt0087Diagnostics(execution.diagnostics),
    progress: execution.progress,
    resources: execution.resources,
  });
}

function publicOpt0087Diagnostics(
  diagnostics: AcePlannerOpt0087InvocationDiagnostics,
): Readonly<Record<string, unknown>> {
  const { cacheAppendWords, ...serializable } = diagnostics;
  return Object.freeze({
    ...serializable,
    cacheAppendU32LeSha256: hashU32Storage(cacheAppendWords),
    cacheAppendWordsRetainedInReceipt: false,
  });
}

function cacheWriteStatusReceipt(
  control: AcePlannerOpt0087InvocationDiagnostics,
  candidate: AcePlannerOpt0087InvocationDiagnostics,
  rows: 1 | 2,
): Readonly<Record<string, unknown>> {
  const expected = Array<number>(rows).fill(1);
  if (
    control.writeStatusWords.length !== rows ||
    candidate.writeStatusWords.length !== rows ||
    control.writeStatusWords.some((word, index) => word !== expected[index]) ||
    candidate.writeStatusWords.some((word, index) => word !== expected[index])
  ) throw new Error("OPT-0087 cache/write status changed");
  return Object.freeze({
    expectedWords: Object.freeze(expected),
    controlWords: Object.freeze([...control.writeStatusWords]),
    candidateWords: Object.freeze([...candidate.writeStatusWords]),
    exact: true,
    validationOwner:
      "planner executor rejects before returning logits unless every cache/write status word is one",
  });
}

function rowSha256(row: Float32Array): string {
  return aceSha256Hex(new Uint8Array(row.buffer, row.byteOffset, row.byteLength));
}

function hashU32Storage(words: Uint32Array): string {
  return aceSha256Hex(new Uint8Array(
    words.buffer,
    words.byteOffset,
    words.byteLength,
  ));
}

function sha256U32Le(values: readonly number[]): string {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

interface MutableArmTimings {
  readonly transformerLayerWallMilliseconds: number[];
  readonly tiedHeadWallMilliseconds: number[];
  readonly readbackWallMilliseconds: number[];
  readonly modelThroughReadbackWallMilliseconds: number[];
  readonly samplingWallMilliseconds: number[];
  readonly completeTokenWallMilliseconds: number[];
}

function emptyArmTimings(): MutableArmTimings {
  return {
    transformerLayerWallMilliseconds: [],
    tiedHeadWallMilliseconds: [],
    readbackWallMilliseconds: [],
    modelThroughReadbackWallMilliseconds: [],
    samplingWallMilliseconds: [],
    completeTokenWallMilliseconds: [],
  };
}

function appendArmTiming(
  target: MutableArmTimings,
  execution: ArmExecution,
): void {
  target.transformerLayerWallMilliseconds.push(
    execution.diagnostics.transformerLayerWallMilliseconds,
  );
  target.tiedHeadWallMilliseconds.push(
    execution.diagnostics.tiedHeadWallMilliseconds,
  );
  target.readbackWallMilliseconds.push(
    execution.diagnostics.readbackWallMilliseconds,
  );
  target.modelThroughReadbackWallMilliseconds.push(
    execution.modelThroughReadbackWallMilliseconds,
  );
  target.samplingWallMilliseconds.push(execution.samplingWallMilliseconds);
  target.completeTokenWallMilliseconds.push(
    execution.recurringCompleteTokenWallMilliseconds,
  );
}

function freezeArmTimings(timing: MutableArmTimings) {
  return Object.freeze({
    transformerLayerWallMilliseconds: Object.freeze([
      ...timing.transformerLayerWallMilliseconds,
    ]),
    tiedHeadWallMilliseconds: Object.freeze([
      ...timing.tiedHeadWallMilliseconds,
    ]),
    modelThroughReadbackWallMilliseconds: Object.freeze([
      ...timing.modelThroughReadbackWallMilliseconds,
    ]),
    completeTokenWallMilliseconds: Object.freeze([
      ...timing.completeTokenWallMilliseconds,
    ]),
  });
}

function summarizeArmTimings(
  timing: MutableArmTimings,
  rows: 1 | 2,
): Readonly<Record<string, unknown>> {
  const layer = summarizeTimings(timing.transformerLayerWallMilliseconds);
  const head = summarizeTimings(timing.tiedHeadWallMilliseconds);
  const layerMedian = median(timing.transformerLayerWallMilliseconds);
  const headMedian = median(timing.tiedHeadWallMilliseconds);
  return Object.freeze({
    transformerLayers: layer,
    tiedHead: head,
    readback: summarizeTimings(timing.readbackWallMilliseconds),
    modelThroughReadback:
      summarizeTimings(timing.modelThroughReadbackWallMilliseconds),
    sampling: summarizeTimings(timing.samplingWallMilliseconds),
    completeToken: summarizeTimings(timing.completeTokenWallMilliseconds),
    logicalWeightBandwidth: Object.freeze({
      layerBytesPerToken: LOGICAL_LAYER_WEIGHT_BYTES_PER_ROW * rows,
      tiedHeadBytesPerToken: LOGICAL_TIED_HEAD_WEIGHT_BYTES_PER_ROW * rows,
      layerMedianGigabytesPerSecond:
        LOGICAL_LAYER_WEIGHT_BYTES_PER_ROW * rows / (layerMedian * 1_000_000),
      tiedHeadMedianGigabytesPerSecond:
        LOGICAL_TIED_HEAD_WEIGHT_BYTES_PER_ROW * rows / (headMedian * 1_000_000),
    }),
  });
}

function summarizeTimings(
  values: readonly number[],
): Readonly<Record<string, unknown>> {
  if (
    values.length !== OPT_0087_PAIR_ORDERS.length ||
    values.some((value) => !Number.isFinite(value) || value <= 0)
  ) throw new Error("OPT-0087 timing sample set changed");
  return Object.freeze({
    count: values.length,
    minimumMilliseconds: Math.min(...values),
    medianMilliseconds: median(values),
    meanMilliseconds: mean(values),
    standardDeviationMilliseconds: standardDeviation(values),
    maximumMilliseconds: Math.max(...values),
    valuesMilliseconds: Object.freeze([...values]),
  });
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: readonly number[]): number {
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
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
    PATHS.length !== OPT_0087_PATH_IDS.length ||
    PATHS.some((path, index) => path.id !== OPT_0087_PATH_IDS[index]) ||
    OPT_0087_PAIR_ORDERS.length !== 8 ||
    OPT_0087_PAIR_ORDERS.filter((order) => order[0] === "control").length !== 4 ||
    OPT_0087_PAIR_ORDERS.filter((order) => order[0] === "candidate").length !== 4 ||
    ACCEPTED_SEMANTIC_CODE_IDS.length !== 150
  ) throw new Error("OPT-0087 frozen browser protocol changed");
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
  ) throw new Error("OPT-0087 worker heartbeat telemetry changed");
}

async function failAndCleanup(
  error: unknown,
  active: PreparedSession | undefined = session,
): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  session = undefined;
  activeAbortController = undefined;
  active?.router.abandon();
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let cleanupError: unknown;
  if (active !== undefined) {
    try {
      cleanup = await active.cleanup(error);
    } catch (caught) {
      cleanupError = caught;
    }
  }
  const heartbeat = workerHeartbeat?.stop();
  workerHeartbeat = undefined;
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      primary: errorValue(error),
      ...(cleanup === undefined ? {} : { cleanup }),
      ...(cleanupError === undefined
        ? {}
        : { cleanupError: errorValue(cleanupError) }),
      ...(heartbeat === undefined ? {} : { workerHeartbeat: heartbeat }),
    }),
  });
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`OPT-0087 package omitted ${name}`);
  return file;
}

function requireMapValue<K, V>(map: ReadonlyMap<K, V>, key: K): V {
  const value = map.get(key);
  if (value === undefined) throw new Error(`OPT-0087 map omitted ${String(key)}`);
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

async function yieldToWorker(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      ...(error instanceof Opt0087ResourceTopologyError ||
        error instanceof Opt0087ResourcePairTopologyError
        ? { diagnostic: error.diagnostic }
        : {}),
      ...(error.cause === undefined ? {} : { cause: String(error.cause) }),
    });
  }
  return Object.freeze({ name: "Error", message: String(error), stack: null });
}

function errorText(error: unknown): string {
  return error instanceof Error
    ? `${error.name}: ${error.message}`
    : String(error);
}
