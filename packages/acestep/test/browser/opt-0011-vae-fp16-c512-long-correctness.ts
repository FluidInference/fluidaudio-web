/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />
/// <reference lib="webworker" />

import fixtureSource from
  "../../benchmark/opt-0011-vae-fp16-storage-window.ts?raw";
import {
  createAceOpt0011LatentFixture,
  planAceOpt0011TemporalSupport,
} from "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import acquireSource from "../../src/model/acquire.ts?raw";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import gpuTensorsSource from "../../src/model/gpu-tensors.ts?raw";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import packageSource from "../../src/model/package.ts?raw";
import {
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import { AceCooperativeGpuScheduler } from "../../src/runtime/scheduler.js";
import chunksSource from "../../src/webgpu/vae-chunks.ts?raw";
import type { AceVaeDecodeWindow } from "../../src/webgpu/vae-chunks.js";
import decoderSource from "../../src/webgpu/vae-decoder.ts?raw";
import { planAceVaeDecoder } from "../../src/webgpu/vae-decoder.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
} from "../../src/webgpu/device.js";
import conv1dSource from "../../src/webgpu/kernels/vae-conv1d-fp16.ts?raw";
import convTransposeSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import pointwiseSource from
  "../../src/webgpu/kernels/vae-pointwise-fp16.ts?raw";
import snakeSource from "../../src/webgpu/kernels/vae-snake-fp16.ts?raw";
import decoderRuntimeSource from "../../src/webgpu/vae-fp16-decoder.ts?raw";
import {
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  AceOpt0011Fp16VaeDecoderRuntime,
  type AceOpt0011Fp16VaeChunkDispatchSet,
  type AceOpt0011Fp16VaeWindowDispatch,
} from "../../src/webgpu/vae-fp16-decoder.js";
import fp16PackageSource from "../../src/webgpu/vae-fp16-package.ts?raw";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  resolveAceOpt0011Fp16VaePackageBindings,
  type AceOpt0011VaePackageBindings,
} from "../../src/webgpu/vae-fp16-package.js";
import fp16ProfileSource from "../../src/webgpu/vae-fp16-profile.ts?raw";
import {
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
} from "../../src/webgpu/vae-fp16-profile.js";

export const OPT_0011_C512_LONG_RUNTIME_COMMIT =
  "d5178ed84e3144e609c461af44e0c71d75d565ba" as const;
export const OPT_0011_C512_LONG_RUNTIME_SOURCE_SHA256 =
  "dd83ec341e7d27d2c4f3cc1f673c1d7d7a05818212de12f76146ae546bb508d7" as const;
export const OPT_0011_C512_LONG_B256_ARTIFACT_SHA256 =
  "827d6d46feeac13ad45e78487d4e19e9fca4c10baacf139499206a5c497f1f54" as const;
export const OPT_0011_C512_LONG_B256_ARTIFACT_BYTES = 28_763;
export const OPT_0011_C512_LONG_B256_OUTPUT_SHA256 =
  "782ac4036233045b7facbc583369f31c5c74dd83ff4d3197daaeccc829886327" as const;
export const OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256 =
  "9b950a596069ee381e178d1ccc4cc5d5c4bc77d2ba37c6727bd9fdb3216ed9f9" as const;
export const OPT_0011_C512_LONG_B256_BINDING_SHA256 =
  "5664970bd6dc29c4168dbf75cef7e17c4340a2057d4ee34591b607e8ae62e89a" as const;
export const OPT_0011_C512_LONG_FIXTURE_SHA256 =
  "e8919adc02d83f2efcd60bcb6dec4f104628d2ed66742d0eddbffc6b0a481a14" as const;
export const OPT_0011_C512_DIRECT_FIXTURE_SHA256 =
  "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899" as const;
export const OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS = 32_768;

const CANDIDATE_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json" as const;
const B256_ARTIFACT_PATH =
  "/optimization/artifacts/OPT-0011/raw/fp16-b256-window-correctness.json" as const;
const B256_ARTIFACT_HARNESS_COMMIT =
  "4c3ff7db2f96d1a62b62de4269d7c68c06b93a64" as const;
const B256_ARTIFACT_RUNTIME_COMMIT =
  "c5db676227e074c08a3499794c2bc2e99cf5a856" as const;
const LONG_FRAMES = 1_024;
const DIRECT_B_FRAMES = 256;
const DIRECT_C_FRAMES = 512;
const LATENT_CHANNELS = 64;
const HOP_LENGTH = 1_920;
const AUDIO_CHANNELS = 2;
const LONG_OUTPUT_FRAMES = LONG_FRAMES * HOP_LENGTH;
const LONG_OUTPUT_ELEMENTS = 3_932_160;
const LONG_OUTPUT_BYTES = LONG_OUTPUT_ELEMENTS * 4;
const GUARD_BYTES = 256;
const GUARD_WORD = 0xa55a_5aa5;
const OUTPUT_QNAN_WORD = 0x7fc5_0011;
const STAGING_QNAN_WORD = 0x7fc5_1011;
const DECODER_QNAN_U16 = 0x7e11;
const READBACK_COPY_COUNT = 13;
const LOCAL_SEAM_RADIUS_AUDIO_FRAMES = 1_920;
const RAW_RESULT_GLOBAL = "__ACE_OPT_0011_C512_LONG_RAW_RESULT_JSON__";

const B_WINDOW_INPUT_SHA256 = Object.freeze([
  "bfbc7d911561c7247b9e0e253f4ecb698b946e7b886045aaf9831ff0e6436e3c",
  "0d6670e9a8f6bc7a62eb953336958f199433277c13c0e93bcbfd39ab58c3ac4f",
  "8dad56dbed867a8db5b75cd857368d85459d0cea1e0e61ff4fd855821c71636a",
  "1c42e13a7eacb72f4c3ec1dfced63e94dfc6129c82e196f60a38dc059bebcff4",
  "db8e175ef56b14fbabe5d3b3a5b0a2a03931b3d7e8220a6385181d5c3647295f",
  "0cd1d5ec9509c9c8b89e03b5444e8b61f569109db685320f8bb2fbdcd552fcab",
  "53f5ec718d5feeb9fb0e26e8ac2cf9306c0be17e4dfe8938a9764255b4ab6b07",
  "ae46a59e458151872c3311a2b0c80a1e773d8a066806a2176f460823ce3ded0d",
] as const);
const C_WINDOW_INPUT_SHA256 = Object.freeze([
  "a1b3d1bebcfbdce1c4665a76534fd67a888a78072adb66e84d29fa7d05e26653",
  "af19dd79b7ee416eedb73202fe725b95843e30090dd70d4a81051a1244b58afb",
  "9c069016690717780af9448d038b2785116a888d7670c435b5c3c9fc3ec497bd",
] as const);

const SOURCE_AUTHORITIES = Object.freeze([
  ["fixture", fixtureSource,
    "0a35af918bc1bd9a4013fc963ce0c436fbe8e775e6f3bc70e914759c35fb6a11"],
  ["model-acquire", acquireSource,
    "25abe6a73a868a6ded404b2a59b2c60394bba953ea384e22ed309718dff9a214"],
  ["model-gpu-tensors", gpuTensorsSource,
    "19e946038a2f99dc46f8e85fa9e08c6469499b10c2e8805f003278bf6cb00be1"],
  ["model-package", packageSource,
    "6b997d86c408c175eb5c1065a642f7b41c0319a8fa739ae703a4b0258ca05ec5"],
  ["scheduler", schedulerSource,
    "a6825fb677883df136f480baa0613437316d867f2ea8bf70bb6f60ca25bd5e16"],
  ["vae-chunks", chunksSource,
    "23bbc8e6e7e8b1978075ee64bdd72ee3338058aa0df83d0293a577e0c6dffc22"],
  ["decoder-graph", decoderSource,
    "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e"],
  ["fp16-decoder-runtime", decoderRuntimeSource,
    OPT_0011_C512_LONG_RUNTIME_SOURCE_SHA256],
  ["fp16-package", fp16PackageSource,
    "3a43403ebe9d4ba1ee444601296ffd06f44198c396a49a244f08cf590e699198"],
  ["fp16-profile", fp16ProfileSource,
    "43ba5dcac52589bd47f77bff65fa8293d8c22ed5fec8eaac78935e5de9fc51be"],
  ["fp16-conv1d", conv1dSource,
    "fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310"],
  ["fp16-conv-transpose1d", convTransposeSource,
    "ecad5f7e981c7310d73565cb15a95123d32725ed6bb41342f484235db3caadd5"],
  ["fp16-pointwise", pointwiseSource,
    "c801eb209132ed2705a3b7e7b742afd2a6b17855d257938b5df515b6285f3eab"],
  ["fp16-snake", snakeSource,
    "0e0cc8d1974e6f36942a98777e43c6b48b27c00a8cb0d912ff1f510be426601f"],
] as const);

export interface Opt0011C512LongRunIdentity {
  readonly harnessCommit: string;
  readonly runtimeCommit: typeof OPT_0011_C512_LONG_RUNTIME_COMMIT;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

interface WorkerRunMessage {
  readonly type: "run";
  readonly identity: Opt0011C512LongRunIdentity;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly tensorCount: number;
  readonly residentBytes: number;
}

type AuditScope = "setup" | "package" | "B" | "C";

interface BufferRecord {
  readonly scope: AuditScope;
  readonly label: string;
  readonly size: number;
  destroyCalls: number;
  destroyed: boolean;
}

interface MutableCommandRecord {
  readonly label: string;
  computePassCount: number;
  dispatchCount: number;
  copyCount: number;
  clearCount: number;
  finished: boolean;
  submitted: boolean;
  drained: boolean;
}

interface MutableExecutionTrace {
  readonly id: string;
  readonly commands: MutableCommandRecord[];
  submissionCount: number;
  drainCount: number;
  writeBufferCount: number;
}

interface GuardedBinding {
  readonly label: string;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly payloadBytes: number;
}

interface ArmResources {
  readonly maximumFrames: 256 | 512;
  readonly stagingInput: GuardedBinding;
  readonly decoderInput: GuardedBinding;
  readonly workspaces: readonly [GuardedBinding, GuardedBinding, GuardedBinding];
  readonly output: GuardedBinding;
  readonly outputReadback: GPUBuffer;
  readonly guardReadback: GPUBuffer;
  readonly guarded: readonly GuardedBinding[];
  destroy(): void;
}

interface OutputScan {
  readonly elementCount: number;
  readonly byteLength: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly qNaNSentinelCount: number;
  readonly stereoDifferenceFrameCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly peak: number;
  readonly clampBoundaryCount: number;
  readonly sha256: string;
}

interface WindowExecution {
  readonly activeOutput: Float32Array<ArrayBuffer>;
  readonly scan: OutputScan;
  readonly inactiveTail: Readonly<Record<string, unknown>>;
  readonly trace: Readonly<Record<string, unknown>>;
}

interface LongArmResult {
  readonly output: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface HeartbeatController {
  stop(): Readonly<Record<string, unknown>>;
}

interface ActualGateAccounting {
  readonly fullWindowExecutionCount: number;
  readonly partialWindowExecutionCount: number;
  readonly dispatchCount: number;
  readonly commandBufferCount: number;
  readonly readbackCommandBufferCount: number;
  readonly readbackCopyCount: number;
  readonly completedRealIdleCount: number;
  readonly rawU32ComparisonCount: number;
  readonly createdBufferCount: number;
  readonly maximumLiveBufferCount: number;
  readonly lifetimeCreatedBufferBytes: number;
  readonly executionTraceCount: number;
  readonly queueDrainCount: number;
  readonly completedInternalRealIdleCount: number;
  readonly completedBetweenWindowRealIdleCount: number;
  readonly everyCompletedExecutionAggregatedExactlyOnce: true;
}

if (typeof document !== "undefined") installPage();
else if (
  typeof self !== "undefined" &&
  new URL(self.location.href).searchParams.get("dedicatedWorker") === "1"
) installWorker();

export function parseOpt0011C512LongRunIdentity(
  parameters: URLSearchParams,
): Opt0011C512LongRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  const runtimeCommit = requiredIdentity(parameters, "runtimeCommit");
  if (!/^[0-9a-f]{40}$/u.test(harnessCommit)) {
    throw new Error("OPT-0011 C512/long gate requires a full harness commit");
  }
  if (runtimeCommit !== OPT_0011_C512_LONG_RUNTIME_COMMIT) {
    throw new Error("OPT-0011 C512/long runtime commit changed");
  }
  const positiveInteger = (name: string): number => {
    const value = Number(requiredIdentity(parameters, name));
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`OPT-0011 C512/long gate requires positive ${name}`);
    }
    return value;
  };
  return Object.freeze({
    harnessCommit,
    runtimeCommit,
    machineModel: requiredIdentity(parameters, "machineModel"),
    osVersion: requiredIdentity(parameters, "osVersion"),
    osBuild: requiredIdentity(parameters, "osBuild"),
    browserVersion: requiredIdentity(parameters, "browserVersion"),
    gpuCoreCount: positiveInteger("gpuCoreCount"),
    memoryBytes: positiveInteger("memoryBytes"),
  });
}

function installPage(): void {
  installRawResultChunkRetrieval();
  const start = requireElement<HTMLButtonElement>("#start");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    let identity: Opt0011C512LongRunIdentity;
    try {
      identity = parseOpt0011C512LongRunIdentity(
        new URLSearchParams(window.location.search),
      );
    } catch (error) {
      finishPage("failed", failureReceipt(error));
      return;
    }
    const pageHeartbeat = startPageHeartbeat();
    const workerUrl = new URL(
      "./opt-0011-vae-fp16-c512-long-correctness.ts",
      import.meta.url,
    );
    workerUrl.searchParams.set("dedicatedWorker", "1");
    const worker = new Worker(workerUrl, { type: "module" });
    let settled = false;
    worker.addEventListener("message", (event: MessageEvent) => {
      const data = event.data as Record<string, unknown>;
      if (data["type"] === "progress") {
        updateProgress(String(data["message"]));
        return;
      }
      if (data["type"] !== "passed" && data["type"] !== "failed") return;
      settled = true;
      const status = data["type"] as "passed" | "failed";
      const result = data["result"] as Readonly<Record<string, unknown>>;
      worker.terminate();
      const pageLiveness = pageHeartbeat.stop();
      if (status === "passed" && pageLiveness["observed"] !== true) {
        finishPage("failed", Object.freeze({
          ...failureReceipt(new Error("OPT-0011 page heartbeat was not live")),
          workerResult: result,
          pageHeartbeat: pageLiveness,
        }));
        return;
      }
      finishPage(status, Object.freeze({ ...result, pageHeartbeat: pageLiveness }));
    });
    worker.addEventListener("error", (event) => {
      if (settled) return;
      settled = true;
      worker.terminate();
      finishPage("failed", Object.freeze({
        ...failureReceipt(event.error ?? event.message),
        pageHeartbeat: pageHeartbeat.stop(),
      }));
    });
    worker.postMessage({ type: "run", identity } satisfies WorkerRunMessage);
  }, { once: true });
}

function installWorker(): void {
  let started = false;
  self.addEventListener("message", (event: MessageEvent<WorkerRunMessage>) => {
    if (started || event.data.type !== "run") return;
    started = true;
    const heartbeat = startWorkerHeartbeat();
    void runWorker(event.data.identity, heartbeat).then(
      (result) => self.postMessage({ type: "passed", result }),
      (error: unknown) => self.postMessage({
        type: "failed",
        result: Object.freeze({
          ...failureReceipt(error),
          workerHeartbeat: safeStopHeartbeat(heartbeat),
        }),
      }),
    );
  });
}

async function runWorker(
  identity: Opt0011C512LongRunIdentity,
  heartbeat: HeartbeatController,
): Promise<Readonly<Record<string, unknown>>> {
  validateWorkerIdentity(identity);
  postProgress("authenticating sources, B256 evidence, and frozen latents");
  const sourceAuthority = await authenticateSources();
  const b256Authority = await authenticateB256Artifact();
  const longBytes = createAceOpt0011LatentFixture(LONG_FRAMES);
  if (
    longBytes.byteLength !== LONG_FRAMES * LATENT_CHANNELS * 4 ||
    await sha256Hex(longBytes) !== OPT_0011_C512_LONG_FIXTURE_SHA256
  ) throw new Error("OPT-0011 long latent fixture identity changed");
  const directBBytes = longBytes.subarray(0, DIRECT_B_FRAMES * LATENT_CHANNELS * 4);
  if (
    await sha256Hex(directBBytes) !==
      "55333d3ae4a0aca83dc1509b837c577f54646924e658e01e53889dc8a5a44875"
  ) throw new Error("OPT-0011 B256 prefix fixture identity changed");
  const directCBytes = longBytes.subarray(0, DIRECT_C_FRAMES * LATENT_CHANNELS * 4);
  if (await sha256Hex(directCBytes) !== OPT_0011_C512_DIRECT_FIXTURE_SHA256) {
    throw new Error("OPT-0011 C512 prefix fixture identity changed");
  }
  const longLatent = float32Copy(longBytes);
  const directB = longLatent.subarray(0, DIRECT_B_FRAMES * LATENT_CHANNELS);
  const directC = longLatent.subarray(0, DIRECT_C_FRAMES * LATENT_CHANNELS);
  const temporalSupport = planAceOpt0011TemporalSupport();
  if (
    temporalSupport.hopLength !== HOP_LENGTH ||
    temporalSupport.maximumPastLatentFrames !== 9 ||
    temporalSupport.maximumFutureLatentFrames !== 9 ||
    temporalSupport.maximumRadiusLatentFrames !== 9
  ) throw new Error("OPT-0011 temporal-support authority changed");

  postProgress("authenticating the one revision-5 candidate package");
  const prepared = await authenticateCandidatePackage();
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  const audit = new DeviceAudit(context.device);
  let phase: AceGpuTensorPhase | undefined;
  let bResult: LongArmResult | undefined;
  let bReceipt: Readonly<Record<string, unknown>> | undefined;
  let cResult: LongArmResult | undefined;
  let cReceipt: Readonly<Record<string, unknown>> | undefined;
  let bridgeReceipt: Readonly<Record<string, unknown>> | undefined;
  let cancellationReceipt: Readonly<Record<string, unknown>> | undefined;
  let comparison: Readonly<Record<string, unknown>> | undefined;
  let seamReceipt: Readonly<Record<string, unknown>> | undefined;
  let intentionalLoss: Awaited<typeof context.lost> | undefined;
  let postCleanupError: Error | undefined;
  const rememberCleanupError = (error: Error): void => {
    postCleanupError ??= error;
  };
  try {
    audit.setScope("package");
    const files = await acquirePackageFiles(prepared);
    phase = await AceGpuTensorPhase.load(
      audit.device,
      prepared.loaded.manifest,
      files,
      ["vae"],
    );
    if (
      phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
      phase.phases.length !== 1 || phase.phases[0] !== "vae" ||
      audit.liveCount("package") !== 7
    ) throw new Error("OPT-0011 candidate resident package accounting changed");
    const packageBindings = resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(DIRECT_B_FRAMES),
      prepared.loaded,
      phase,
    );

    audit.setScope("B");
    const b = await runBPhase(
      audit,
      packageBindings,
      longLatent,
      directB,
      b256Authority,
    );
    bResult = b.long;
    bReceipt = b.receipt;
    bridgeReceipt = b.bridge;
    cancellationReceipt = Object.freeze({ betweenWindows: b.cancellation });
    if (audit.liveCount("B") !== 0 || audit.liveCount("package") !== 7) {
      throw new Error("OPT-0011 B resources did not drain before C");
    }

    audit.setScope("C");
    const c = await runCPhase(
      audit,
      packageBindings,
      longLatent,
      directC,
    );
    cResult = c.long;
    cReceipt = c.receipt;
    cancellationReceipt = Object.freeze({
      ...cancellationReceipt,
      betweenBatches: c.cancellation,
    });
    if (audit.liveCount("C") !== 0 || audit.liveCount("package") !== 7) {
      throw new Error("OPT-0011 C resources did not drain before package cleanup");
    }

    comparison = audit.compareU32(bResult.output, cResult.output);
    if (comparison.mismatchCount !== 0) {
      throw new Error("OPT-0011 B/C long waveform U32 identity failed");
    }
    const [bSha256, cSha256] = await Promise.all([
      sha256Hex(bytesOf(bResult.output)),
      sha256Hex(bytesOf(cResult.output)),
    ]);
    if (bSha256 !== cSha256) {
      throw new Error("OPT-0011 B/C long waveform hash identity failed");
    }
    comparison = Object.freeze({
      ...comparison,
      bSha256,
      cSha256,
      hashesEqual: true,
    });
    seamReceipt = compareSeams(bResult.output, cResult.output);
    phase.destroy();
    phase = undefined;
  } finally {
    phase?.destroy();
    audit.destroyAll();
    audit.destroyAll();
    const beforeDeviceDestroy = audit.resourceSummary();
    if (
      beforeDeviceDestroy.liveBufferCount !== 0 ||
      beforeDeviceDestroy.createdBufferCount !==
        beforeDeviceDestroy.destroyedBufferCount ||
      beforeDeviceDestroy.createdBufferCount !== 28 ||
      beforeDeviceDestroy.maximumLiveBufferCount !== 18 ||
      beforeDeviceDestroy.createdBufferBytes !== 1_331_961_680
    ) rememberCleanupError(new Error("OPT-0011 lifecycle/resource totals changed"));
    await yieldToBrowser();
    await yieldToBrowser();
    if (runtimeEvents.length !== 0) {
      rememberCleanupError(new Error("OPT-0011 observed a queued runtime event"));
    }
    try {
      context.destroy();
      const loss = await context.lost;
      intentionalLoss = loss;
      if (loss.reason !== "destroyed") {
        rememberCleanupError(new Error("OPT-0011 intentional device loss changed"));
      }
    } catch (error) {
      rememberCleanupError(asError(error));
    }
    await yieldToBrowser();
    await yieldToBrowser();
    if (runtimeEvents.length !== 0) {
      rememberCleanupError(new Error("OPT-0011 runtime event appeared after loss"));
    }
  }
  const workerHeartbeat = heartbeat.stop();
  if (workerHeartbeat["observed"] !== true) {
    throw new Error("OPT-0011 worker heartbeat was not live");
  }
  if (postCleanupError !== undefined) throw postCleanupError;
  if (
    bResult === undefined || bReceipt === undefined ||
    cResult === undefined || cReceipt === undefined ||
    bridgeReceipt === undefined || cancellationReceipt === undefined ||
    comparison === undefined || seamReceipt === undefined ||
    intentionalLoss === undefined
  ) throw new Error("OPT-0011 C512/long result was incomplete");
  const resources = audit.resourceSummary();
  const actualGateAccounting = audit.aggregateActualExecutionAccounting();
  validateActualGateAccounting(actualGateAccounting);
  return Object.freeze({
    schema: "ace-opt-0011-fp16-vae-c512-long-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification:
      "complete-C512-and-B-C1024-correctness-only-no-performance-or-thermal-timing",
    identity,
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      executionProfile: context.capabilities.executionProfile,
      adapterInfo: context.capabilities.adapterInfo,
      adapterFeatures: context.capabilities.adapterFeatures,
      deviceFeatures: context.capabilities.deviceFeatures,
      adapterLimits: context.capabilities.adapterLimits,
      deviceLimits: context.capabilities.deviceLimits,
      runtimeEvents,
    }),
    protocol: Object.freeze({
      dedicatedWorkerInference: true,
      heavyweightSequence: ["one-package", "B-resources", "destroy-B", "C-resources", "destroy-C", "destroy-package"],
      packageLoadCount: 1,
      simultaneousHeavyweightPackageCount: 1,
      simultaneousArmResourceSetCount: 1,
      decoderQuantaPerCommandBuffer:
        ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyIdleBetweenCommandBuffersAndWindows: true,
      exactEdgeShapesNoLatentPadding: true,
      maximumOutputBindingWithPoisonedInactiveTail: true,
      performanceClaim: null,
      thermalClaim: null,
      responsivenessClaim: null,
      listeningClaim: null,
      qualityClaim: null,
      selectorClaim: null,
      productionIntegrationClaim: null,
    }),
    sourceAuthority,
    priorB256Authority: b256Authority,
    fixture: Object.freeze({
      generator: "xorshift32-13-17-5-high24-symmetric-f32-v1",
      seed: "0x00110512",
      frames: LONG_FRAMES,
      channels: LATENT_CHANNELS,
      byteLength: longBytes.byteLength,
      sha256: OPT_0011_C512_LONG_FIXTURE_SHA256,
      directC512PrefixSha256: OPT_0011_C512_DIRECT_FIXTURE_SHA256,
      bWindowInputSha256: B_WINDOW_INPUT_SHA256,
      cWindowInputSha256: C_WINDOW_INPUT_SHA256,
    }),
    temporalSupport: Object.freeze({
      ...temporalSupport,
      overlapLatentFrames: 64,
      minimumSupportMarginLatentFrames: 55,
      completeBCU32IdentityRequired: true,
    }),
    package: packageReceipt(prepared),
    bridge: bridgeReceipt,
    B: bReceipt,
    C: cReceipt,
    comparison,
    seams: seamReceipt,
    cancellation: cancellationReceipt,
    aggregateGate: opt0011C512LongExpectedTopology().fullGate,
    actualGateAccounting,
    cleanup: Object.freeze({
      ...resources,
      scopeBytes: audit.scopeBytesReceipt(),
      allBResourcesDestroyedBeforeC: true,
      packageResidentAcrossSequentialArms: true,
      destroyAllCalledTwice: true,
      idempotent: true,
      deviceDestroyed: true,
      intentionalDeviceLoss: Object.freeze({
        type: intentionalLoss.type,
        reason: intentionalLoss.reason,
        message: intentionalLoss.message,
      }),
      runtimeEventCount: runtimeEvents.length,
    }),
    workerHeartbeat,
  });
}

async function authenticateSources(): Promise<Readonly<Record<string, unknown>>> {
  const sources: unknown[] = [];
  for (const [name, source, expectedSha256] of SOURCE_AUTHORITIES) {
    const bytes = new TextEncoder().encode(source);
    const sha256 = await sha256Hex(bytes);
    if (sha256 !== expectedSha256) {
      throw new Error(`OPT-0011 source ${name} changed`);
    }
    sources.push(Object.freeze({ name, byteLength: bytes.byteLength, sha256 }));
  }
  return Object.freeze({
    runtimeCommit: OPT_0011_C512_LONG_RUNTIME_COMMIT,
    sources: Object.freeze(sources),
    everySourceAuthenticatedBeforeGpuExecution: true,
  });
}

async function authenticateB256Artifact(): Promise<Readonly<Record<string, unknown>>> {
  const response = await fetch(new URL(B256_ARTIFACT_PATH, self.location.href), {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("OPT-0011 B256 authority fetch failed");
  const bytes = new Uint8Array(await response.arrayBuffer());
  const sha256 = await sha256Hex(bytes);
  if (
    bytes.byteLength !== OPT_0011_C512_LONG_B256_ARTIFACT_BYTES ||
    sha256 !== OPT_0011_C512_LONG_B256_ARTIFACT_SHA256
  ) throw new Error("OPT-0011 B256 authority artifact identity changed");
  const artifact = JSON.parse(new TextDecoder().decode(bytes)) as
    Record<string, unknown>;
  const identity = artifact["identity"] as Record<string, unknown>;
  const candidate = artifact["candidate"] as Record<string, unknown>;
  const first = candidate["first"] as Record<string, unknown>;
  const graph = candidate["graphIdentity"] as Record<string, unknown>;
  const comparison = artifact["comparison"] as Record<string, unknown>;
  const cleanup = artifact["cleanup"] as Record<string, unknown>;
  if (
    artifact["status"] !== "passed" ||
    identity["harnessCommit"] !== B256_ARTIFACT_HARNESS_COMMIT ||
    identity["runtimeCommit"] !== B256_ARTIFACT_RUNTIME_COMMIT ||
    first["sha256"] !== OPT_0011_C512_LONG_B256_OUTPUT_SHA256 ||
    graph["topologySha256"] !== OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256 ||
    graph["operationBindingSha256"] !==
      OPT_0011_C512_LONG_B256_BINDING_SHA256 ||
    comparison["comparedElementCount"] !== 983_040 ||
    cleanup["createdBufferCount"] !== cleanup["destroyedBufferCount"]
  ) throw new Error("OPT-0011 B256 authority facts changed");
  return Object.freeze({
    path: B256_ARTIFACT_PATH,
    byteLength: bytes.byteLength,
    sha256,
    harnessCommit: identity["harnessCommit"],
    runtimeCommit: identity["runtimeCommit"],
    outputSha256: first["sha256"],
    topologySha256: graph["topologySha256"],
    operationBindingSha256: graph["operationBindingSha256"],
    completeABWaveformBoundsPassed: true,
    lifecyclePassed: true,
  });
}

async function authenticateCandidatePackage(): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(CANDIDATE_MANIFEST_PATH, self.location.href).href,
    expectedManifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
    expectedProfile: "fp16-vae-experimental",
  });
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const names = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) => names.has(file.name));
  const residentBytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  if (
    loaded.manifestSha256 !== ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES ||
    loaded.manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    tensors.length !== ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT ||
    files.length !== 7 || names.size !== 7 ||
    residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    files.some((file) => file.kind !== "weights") ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !names.has(name))
  ) throw new Error("OPT-0011 candidate package inventory changed");
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    tensorCount: tensors.length,
    residentBytes,
  });
}

async function acquirePackageFiles(
  prepared: PreparedPackage,
): Promise<ReadonlyMap<string, File>> {
  const cache = await AceOpfsModelCache.open();
  const acquired = await acquireAceModelFiles({
    manifest: Object.freeze({ ...prepared.loaded.manifest, files: prepared.files }),
    manifestUrl: prepared.loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => postProgress(
      `candidate VAE ${progress.fileIndex + 1}/${progress.fileCount} ` +
        `${progress.completedBytes}/${progress.totalBytes} bytes`,
    ),
  });
  if (
    acquired.files.size !== prepared.files.length ||
    acquired.plan.runtimeBytes !== prepared.residentBytes
  ) throw new Error("OPT-0011 candidate acquisition accounting changed");
  return acquired.files;
}

async function runBPhase(
  audit: DeviceAudit,
  packageBindings: AceOpt0011VaePackageBindings,
  longLatent: Float32Array<ArrayBuffer>,
  directB: Float32Array<ArrayBuffer>,
  authority: Readonly<Record<string, unknown>>,
): Promise<Readonly<{
  long: LongArmResult;
  bridge: Readonly<Record<string, unknown>>;
  cancellation: Readonly<Record<string, unknown>>;
  receipt: Readonly<Record<string, unknown>>;
}>> {
  postProgress("building exact B192/B256 dispatches from one resident package");
  const resources = createArmResources(audit.device, "B", 256);
  const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(audit.device);
  const scheduler = new AceCooperativeGpuScheduler();
  try {
    const set = await runtime.createChunkDispatchSet(
      "opt-0011-b1024",
      LONG_FRAMES,
      256,
      runtimeBindings(resources, packageBindings),
    );
    validateChunkSet(set, "B");
    if (
      audit.liveCount("B") !== 10 || audit.liveCount("package") !== 7 ||
      audit.totalLiveCount() !== 17 || audit.liveBytes("B") !== 387_224_864
    ) throw new Error("OPT-0011 B live resource peak changed");
    const b256 = requireDispatch(set, DIRECT_B_FRAMES);
    const bridgeIdentity = await candidateGraphIdentity(b256, packageBindings);
    if (
      bridgeIdentity["topologySha256"] !==
        OPT_0011_C512_LONG_B256_TOPOLOGY_SHA256 ||
      bridgeIdentity["operationBindingSha256"] !==
        OPT_0011_C512_LONG_B256_BINDING_SHA256 ||
      authority["outputSha256"] !== OPT_0011_C512_LONG_B256_OUTPUT_SHA256
    ) throw new Error("OPT-0011 generic B256 evidence bridge changed");
    postProgress("bridging the generic B256 dispatch to the passed B256 output");
    const bridgeExecution = await executeWindow(
      audit,
      scheduler,
      b256,
      resources,
      directB,
      "B-bridge",
    );
    if (bridgeExecution.scan.sha256 !== OPT_0011_C512_LONG_B256_OUTPUT_SHA256) {
      throw new Error("OPT-0011 generic B256 output bridge failed");
    }
    const bridge = Object.freeze({
      classification: "one-candidate-execution-bridge-no-A-load-no-rerun",
      priorArtifactSha256: OPT_0011_C512_LONG_B256_ARTIFACT_SHA256,
      output: bridgeExecution.scan,
      execution: bridgeExecution.trace,
      graphIdentity: bridgeIdentity,
      passed: true,
    });

    postProgress("executing complete B1024 first plan and deterministic rerun");
    const long = await runLongArm(
      audit,
      scheduler,
      set,
      resources,
      longLatent,
      "B",
      B_WINDOW_INPUT_SHA256,
    );
    postProgress("proving cancellation after one complete drained B192 window");
    const cancellation = await runBetweenWindowCancellation(
      audit,
      scheduler,
      set,
      resources,
      longLatent,
    );
    const shaderIdentity = await audit.shaderIdentity("B");
    return Object.freeze({
      long,
      bridge,
      cancellation,
      receipt: Object.freeze({
        profile: "opt-0011-mixed-fp16-portable-v1",
        arm: "B-256/64",
        topology: compactChunkTopology(set),
        resourceAccount: Object.freeze({
          logicalGpuBytes: 556_010_272,
          guardedAllocatedGpuBytes: 556_016_416,
          armOwnedBytesExcludingSharedPackage: 387_224_864,
          peakLiveBufferCountIncludingPackage: 17,
        }),
        long: long.receipt,
        shaderIdentity,
      }),
    });
  } finally {
    await scheduler.dispose();
    runtime.destroy();
    resources.destroy();
  }
}

async function runCPhase(
  audit: DeviceAudit,
  packageBindings: AceOpt0011VaePackageBindings,
  longLatent: Float32Array<ArrayBuffer>,
  directC: Float32Array<ArrayBuffer>,
): Promise<Readonly<{
  long: LongArmResult;
  cancellation: Readonly<Record<string, unknown>>;
  receipt: Readonly<Record<string, unknown>>;
}>> {
  postProgress("building exact C320/C448/C512 dispatches");
  const resources = createArmResources(audit.device, "C", 512);
  const runtime = AceOpt0011Fp16VaeDecoderRuntime.create(audit.device);
  const scheduler = new AceCooperativeGpuScheduler();
  try {
    const set = await runtime.createChunkDispatchSet(
      "opt-0011-c1024",
      LONG_FRAMES,
      512,
      runtimeBindings(resources, packageBindings),
    );
    validateChunkSet(set, "C");
    if (
      audit.liveCount("C") !== 11 || audit.liveCount("package") !== 7 ||
      audit.totalLiveCount() !== 18 || audit.liveBytes("C") !== 775_945_264
    ) throw new Error("OPT-0011 C live resource peak changed");
    const c512 = requireDispatch(set, DIRECT_C_FRAMES);
    postProgress("executing complete C512 first window and deterministic rerun");
    const first = await executeWindow(
      audit,
      scheduler,
      c512,
      resources,
      directC,
      "C512-first",
    );
    const rerun = await executeWindow(
      audit,
      scheduler,
      c512,
      resources,
      directC,
      "C512-rerun",
    );
    const deterministic = audit.compareU32(
      first.activeOutput,
      rerun.activeOutput,
    );
    if (deterministic.mismatchCount !== 0) {
      throw new Error("OPT-0011 complete C512 deterministic rerun failed");
    }
    postProgress("proving cancellation after the first drained C512 batch");
    const cancellation = await runBetweenBatchCancellation(
      audit,
      scheduler,
      c512,
      resources,
      directC,
    );
    postProgress("executing complete C1024 first plan and deterministic rerun");
    const long = await runLongArm(
      audit,
      scheduler,
      set,
      resources,
      longLatent,
      "C",
      C_WINDOW_INPUT_SHA256,
    );
    const shaderIdentity = await audit.shaderIdentity("C");
    return Object.freeze({
      long,
      cancellation,
      receipt: Object.freeze({
        profile: "opt-0011-mixed-fp16-portable-v1",
        arm: "C-512/64",
        topology: compactChunkTopology(set),
        resourceAccount: Object.freeze({
          logicalGpuBytes: 944_730_672,
          guardedAllocatedGpuBytes: 944_736_816,
          armOwnedBytesExcludingSharedPackage: 775_945_264,
          peakLiveBufferCountIncludingPackage: 18,
        }),
        directC512: Object.freeze({
          fixtureSha256: OPT_0011_C512_DIRECT_FIXTURE_SHA256,
          first: first.scan,
          rerun: rerun.scan,
          deterministicU32Comparison: deterministic,
          firstExecution: first.trace,
          rerunExecution: rerun.trace,
          completeActiveOutputAndPhysicalTailChecked: true,
        }),
        long: long.receipt,
        shaderIdentity,
      }),
    });
  } finally {
    await scheduler.dispose();
    runtime.destroy();
    resources.destroy();
  }
}

async function runLongArm(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  resources: ArmResources,
  longLatent: Float32Array<ArrayBuffer>,
  arm: "B" | "C",
  expectedInputHashes: readonly string[],
): Promise<LongArmResult> {
  const firstOutput = new Float32Array(LONG_OUTPUT_ELEMENTS);
  const rerunOutput = new Float32Array(LONG_OUTPUT_ELEMENTS);
  const coverage = new Uint8Array(LONG_OUTPUT_FRAMES);
  const firstWindows: WindowExecution[] = [];
  const inputHashes: string[] = [];
  const firstTraces: unknown[] = [];
  let betweenWindowIdleCount = 0;
  for (const [index, entry] of set.windows.entries()) {
    const latent = latentWindow(longLatent, entry.window);
    const inputSha256 = await sha256Hex(bytesOf(latent));
    if (inputSha256 !== expectedInputHashes[index]) {
      throw new Error(`OPT-0011 ${arm} window ${index} input identity changed`);
    }
    inputHashes.push(inputSha256);
    const execution = await executeWindow(
      audit,
      scheduler,
      entry.dispatch,
      resources,
      latent,
      `${arm}-long-first-window-${index}`,
    );
    firstWindows.push(execution);
    writeCore(firstOutput, execution.activeOutput, entry.window, coverage);
    firstTraces.push(execution.trace);
    if (index + 1 < set.windows.length) {
      await audit.deliverRealQueueEmptyIdle("between-window");
      betweenWindowIdleCount += 1;
    }
  }
  requireExactCoverage(coverage, `${arm} first`);
  const windowReceipts: unknown[] = [];
  const rerunTraces: unknown[] = [];
  for (const [index, entry] of set.windows.entries()) {
    const latent = latentWindow(longLatent, entry.window);
    const execution = await executeWindow(
      audit,
      scheduler,
      entry.dispatch,
      resources,
      latent,
      `${arm}-long-rerun-window-${index}`,
    );
    const repeat = audit.compareU32(
      firstWindows[index]!.activeOutput,
      execution.activeOutput,
    );
    if (repeat.mismatchCount !== 0) {
      throw new Error(`OPT-0011 ${arm} window ${index} rerun diverged`);
    }
    writeCore(rerunOutput, execution.activeOutput, entry.window);
    rerunTraces.push(execution.trace);
    windowReceipts.push(Object.freeze({
      ...entry.window,
      inputSha256: inputHashes[index],
      dispatchInputFrames: entry.dispatch.plan.inputFrames,
      activeStagingInputBytes: entry.dispatch.activeStagingInputBytes,
      activeDecoderInputBytes: entry.dispatch.activeDecoderInputBytes,
      activeOutputBytes: entry.dispatch.activeOutputBytes,
      first: firstWindows[index]!.scan,
      rerun: execution.scan,
      inactiveTailFirst: firstWindows[index]!.inactiveTail,
      inactiveTailRerun: execution.inactiveTail,
      deterministicFullDecodedWindowU32Comparison: repeat,
    }));
    if (index + 1 < set.windows.length) {
      await audit.deliverRealQueueEmptyIdle("between-window");
      betweenWindowIdleCount += 1;
    }
  }
  const assembledRepeat = audit.compareU32(firstOutput, rerunOutput);
  if (assembledRepeat.mismatchCount !== 0) {
    throw new Error(`OPT-0011 ${arm} assembled rerun diverged`);
  }
  const firstScan = await scanOutput(firstOutput, LONG_OUTPUT_ELEMENTS);
  const rerunScan = await scanOutput(rerunOutput, LONG_OUTPUT_ELEMENTS);
  const expectedBetweenIdles = 2 * (set.windows.length - 1);
  if (betweenWindowIdleCount !== expectedBetweenIdles) {
    throw new Error(`OPT-0011 ${arm} between-window idle accounting changed`);
  }
  return Object.freeze({
    output: detachFloat32(firstOutput),
    receipt: Object.freeze({
      first: firstScan,
      rerun: rerunScan,
      deterministicAssembledU32Comparison: assembledRepeat,
      completeDecodedWindowU32ComparisonCount: set.windows.reduce(
        (sum, entry) => sum + entry.dispatch.plan.outputElements,
        0,
      ),
      outputCoverage: Object.freeze({
        outputAudioFrames: LONG_OUTPUT_FRAMES,
        outputInterleavedElements: LONG_OUTPUT_ELEMENTS,
        outputFloat32Bytes: LONG_OUTPUT_BYTES,
        everyAudioFrameCoveredExactlyOnce: true,
        coverageGapCount: 0,
        coverageDuplicationCount: 0,
      }),
      windows: Object.freeze(windowReceipts),
      firstExecutionTraces: Object.freeze(firstTraces),
      rerunExecutionTraces: Object.freeze(rerunTraces),
      completedBetweenWindowIdleCount: betweenWindowIdleCount,
    }),
  });
}

async function executeWindow(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  resources: ArmResources,
  latent: Float32Array<ArrayBuffer>,
  id: string,
): Promise<WindowExecution> {
  await initializeWindow(audit.device.queue, dispatch, resources, latent);
  audit.beginExecution(id);
  const scheduling = await scheduler.runLazy({
    queue: audit.device.queue,
    commandBufferCount: dispatch.commandBufferCountAtBatch8,
    createCommandBuffer: (index) =>
      index < dispatch.decoderCommandBufferCountAtBatch8
        ? encodeWindowBatch(audit.device, dispatch, index, id)
        : encodeWindowReadback(audit.device, resources, id),
    signal: new AbortController().signal,
    yieldQueueIdle: () => audit.deliverRealQueueEmptyIdle("internal"),
  });
  const trace = audit.endExecution();
  validateExecutionTrace(trace, dispatch, id);
  if (
    scheduling.commandBuffersSubmitted !== dispatch.commandBufferCountAtBatch8 ||
    scheduling.queueDrains !== dispatch.commandBufferCountAtBatch8 ||
    scheduling.cooperativeIdleMs !== dispatch.commandBufferCountAtBatch8 - 1
  ) throw new Error(`${id} scheduler accounting changed`);
  const maximumOutput = await mapMaximumOutput(resources.outputReadback);
  await validateGuards(resources.guardReadback, resources.guarded.length, id);
  const activeElements = dispatch.activeOutputBytes / 4;
  const activeOutput = detachFloat32(maximumOutput.subarray(0, activeElements));
  const inactiveTail = validateInactiveOutputTail(
    maximumOutput,
    activeElements,
    resources.output.payloadBytes / 4,
    id,
  );
  const scan = await scanOutput(activeOutput, activeElements);
  validateCompleteOutput(scan, activeElements, id);
  return Object.freeze({
    activeOutput,
    scan,
    inactiveTail,
    trace: Object.freeze({
      ...compactTrace(trace, dispatch),
      scheduling,
      batchSize: ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
      completedRealIdleTurns: dispatch.commandBufferCountAtBatch8 - 1,
    }),
  });
}

async function initializeWindow(
  queue: GPUQueue,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  resources: ArmResources,
  latent: Float32Array<ArrayBuffer>,
): Promise<void> {
  if (
    latent.byteLength !== dispatch.activeStagingInputBytes ||
    dispatch.activeStagingInputBytes > resources.stagingInput.payloadBytes ||
    dispatch.activeDecoderInputBytes > resources.decoderInput.payloadBytes ||
    dispatch.activeOutputBytes > resources.output.payloadBytes
  ) throw new Error("OPT-0011 active window byte contract changed");
  const guard = new Uint32Array(GUARD_BYTES / 4);
  guard.fill(GUARD_WORD);
  for (const item of resources.guarded) {
    queue.writeBuffer(item.buffer, 0, guard);
    queue.writeBuffer(item.buffer, GUARD_BYTES + item.payloadBytes, guard);
  }
  const stagingPoison = new Uint32Array(resources.stagingInput.payloadBytes / 4);
  stagingPoison.fill(STAGING_QNAN_WORD);
  queue.writeBuffer(resources.stagingInput.buffer, GUARD_BYTES, stagingPoison);
  queue.writeBuffer(resources.stagingInput.buffer, GUARD_BYTES, latent);
  const decoderPoison = new Uint16Array(resources.decoderInput.payloadBytes / 2);
  decoderPoison.fill(DECODER_QNAN_U16);
  queue.writeBuffer(resources.decoderInput.buffer, GUARD_BYTES, decoderPoison);
  const outputPoison = new Uint32Array(resources.output.payloadBytes / 4);
  outputPoison.fill(OUTPUT_QNAN_WORD);
  queue.writeBuffer(resources.output.buffer, GUARD_BYTES, outputPoison);
  await queue.onSubmittedWorkDone();
}

function encodeWindowBatch(
  device: GPUDevice,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  batchIndex: number,
  id: string,
): GPUCommandBuffer {
  const first = batchIndex *
    ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
  const end = Math.min(
    first + ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
    dispatch.quanta.length,
  );
  const encoder = device.createCommandEncoder({ label: `${id}-batch-${batchIndex}` });
  const pass = encoder.beginComputePass({ label: `${id}-batch-${batchIndex}-pass` });
  for (let index = first; index < end; index += 1) {
    dispatch.quanta[index]!.encode(pass);
  }
  pass.end();
  return encoder.finish();
}

function encodeWindowReadback(
  device: GPUDevice,
  resources: ArmResources,
  id: string,
): GPUCommandBuffer {
  const encoder = device.createCommandEncoder({ label: `${id}-readback` });
  encoder.copyBufferToBuffer(
    resources.output.buffer,
    GUARD_BYTES,
    resources.outputReadback,
    0,
    resources.output.payloadBytes,
  );
  let destination = 0;
  for (const item of resources.guarded) {
    encoder.copyBufferToBuffer(
      item.buffer,
      0,
      resources.guardReadback,
      destination,
      GUARD_BYTES,
    );
    destination += GUARD_BYTES;
    encoder.copyBufferToBuffer(
      item.buffer,
      GUARD_BYTES + item.payloadBytes,
      resources.guardReadback,
      destination,
      GUARD_BYTES,
    );
    destination += GUARD_BYTES;
  }
  return encoder.finish();
}

async function runBetweenBatchCancellation(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  resources: ArmResources,
  latent: Float32Array<ArrayBuffer>,
): Promise<Readonly<Record<string, unknown>>> {
  await initializeWindow(audit.device.queue, dispatch, resources, latent);
  const controller = new AbortController();
  const reason = new DOMException(
    "cancel-after-first-drained-c512-batch-and-real-idle",
    "AbortError",
  );
  let rejection: unknown;
  audit.beginExecution("C512-batch-cancellation");
  try {
    await scheduler.runLazy({
      queue: audit.device.queue,
      commandBufferCount: dispatch.commandBufferCountAtBatch8,
      createCommandBuffer: (index) => {
        if (index >= dispatch.decoderCommandBufferCountAtBatch8) {
          throw new Error("OPT-0011 batch cancellation encoded readback");
        }
        return encodeWindowBatch(
          audit.device,
          dispatch,
          index,
          "C512-batch-cancellation",
        );
      },
      signal: controller.signal,
      yieldQueueIdle: () => audit.deliverRealQueueEmptyIdle("internal"),
      onProgress: (progress) => {
        if (progress.completedCommandBuffers === 1) controller.abort(reason);
      },
    });
  } catch (error) {
    rejection = error;
  }
  const trace = audit.endExecution();
  const command = trace.commands[0];
  if (
    rejection !== reason || trace.commands.length !== 1 ||
    trace.submissionCount !== 1 || trace.drainCount !== 1 ||
    trace.writeBufferCount !== 0 || command === undefined ||
    command.label !== "C512-batch-cancellation-batch-0" ||
    command.computePassCount !== 1 || command.dispatchCount !== 8 ||
    command.copyCount !== 0 || command.clearCount !== 0 ||
    !command.finished || !command.submitted || !command.drained
  ) throw new Error("OPT-0011 between-batch cancellation accounting changed");
  return Object.freeze({
    rejectionName: "AbortError",
    rejectionMessage: reason.message,
    cancellationPoint: "after-first-drained-C512-batch-and-real-idle",
    completedDecoderQuanta: 8,
    encodedCommandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    completedRealIdleCount: 1,
    laterEncodingPrevented: true,
    laterSubmissionPrevented: true,
    readbackPrevented: true,
    sinkWritePrevented: true,
    normalizationPrevented: true,
    outputFinalizationPrevented: true,
    metricsPublicationPrevented: true,
    drainBeforeRelease: true,
  });
}

async function runBetweenWindowCancellation(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  resources: ArmResources,
  longLatent: Float32Array<ArrayBuffer>,
): Promise<Readonly<Record<string, unknown>>> {
  const controller = new AbortController();
  const reason = new DOMException(
    "cancel-after-first-drained-b192-window-sink-and-real-idle",
    "AbortError",
  );
  let rejection: unknown;
  let uploadCount = 0;
  let completedWindowCount = 0;
  let sinkWriteCount = 0;
  let sinkElementCount = 0;
  try {
    for (const [index, entry] of set.windows.entries()) {
      controller.signal.throwIfAborted();
      uploadCount += 1;
      const execution = await executeWindow(
        audit,
        scheduler,
        entry.dispatch,
        resources,
        latentWindow(longLatent, entry.window),
        `B-window-cancellation-window-${index}`,
      );
      completedWindowCount += 1;
      const core = coreView(execution.activeOutput, entry.window);
      sinkElementCount += core.length;
      sinkWriteCount += 1;
      if (index === 0) {
        await audit.deliverRealQueueEmptyIdle("between-window");
        controller.abort(reason);
      }
      controller.signal.throwIfAborted();
    }
  } catch (error) {
    rejection = error;
  }
  if (
    rejection !== reason || uploadCount !== 1 || completedWindowCount !== 1 ||
    sinkWriteCount !== 1 || sinkElementCount !== 491_520
  ) throw new Error("OPT-0011 between-window cancellation accounting changed");
  return Object.freeze({
    rejectionName: "AbortError",
    rejectionMessage: reason.message,
    cancellationPoint:
      "after-complete-drained-B192-window-sink-write-and-real-between-window-idle",
    completedWindowCount,
    completedWindowFrames: 192,
    completedDecoderQuanta: 2_968,
    encodedCommandBufferCount: 372,
    submissionCount: 372,
    queueDrainCount: 372,
    completedInternalIdleCount: 371,
    completedBetweenWindowIdleCount: 1,
    sinkWriteCount,
    sinkElementCount,
    laterWindowUploadPrevented: true,
    laterEncodingPrevented: true,
    laterSubmissionPrevented: true,
    laterReadbackPrevented: true,
    laterSinkWritePrevented: true,
    normalizationPrevented: true,
    outputFinalizationPrevented: true,
    metricsPublicationPrevented: true,
    drainBeforeRelease: true,
  });
}

function createArmResources(
  device: GPUDevice,
  arm: "B" | "C",
  maximumFrames: 256 | 512,
): ArmResources {
  const graph = planAceVaeDecoder(maximumFrames);
  const stagingInput = createGuardedBinding(
    device,
    `opt-0011-${arm}-staging-input`,
    graph.inputElements * 4,
  );
  const decoderInput = createGuardedBinding(
    device,
    `opt-0011-${arm}-decoder-input`,
    graph.inputElements * 2,
  );
  const workspaceBytes = graph.maximumActivationElements * 2;
  const workspaces = [0, 1, 2].map((index) => createGuardedBinding(
    device,
    `opt-0011-${arm}-workspace-${index}`,
    workspaceBytes,
  )) as [GuardedBinding, GuardedBinding, GuardedBinding];
  const output = createGuardedBinding(
    device,
    `opt-0011-${arm}-output`,
    graph.outputElements * 4,
  );
  const guarded = Object.freeze([
    stagingInput,
    decoderInput,
    ...workspaces,
    output,
  ]);
  const outputReadback = device.createBuffer({
    label: `opt-0011-${arm}-output-readback`,
    size: output.payloadBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const guardReadback = device.createBuffer({
    label: `opt-0011-${arm}-guard-readback`,
    size: guarded.length * 2 * GUARD_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let destroyed = false;
  return Object.freeze({
    maximumFrames,
    stagingInput,
    decoderInput,
    workspaces,
    output,
    outputReadback,
    guardReadback,
    guarded,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      for (const item of guarded) item.buffer.destroy();
      outputReadback.destroy();
      guardReadback.destroy();
    },
  });
}

function createGuardedBinding(
  device: GPUDevice,
  label: string,
  payloadBytes: number,
): GuardedBinding {
  if (!Number.isSafeInteger(payloadBytes) || payloadBytes <= 0 || payloadBytes % 4 !== 0) {
    throw new Error(`${label} payload is invalid`);
  }
  const buffer = device.createBuffer({
    label,
    size: GUARD_BYTES + payloadBytes + GUARD_BYTES,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  return Object.freeze({
    label,
    buffer,
    binding: Object.freeze({ buffer, offset: GUARD_BYTES, size: payloadBytes }),
    payloadBytes,
  });
}

function runtimeBindings(
  resources: ArmResources,
  packageBindings: AceOpt0011VaePackageBindings,
) {
  return Object.freeze({
    stagingInput: resources.stagingInput.binding,
    decoderInput: resources.decoderInput.binding,
    workspaces: resources.workspaces.map((item) => item.binding) as
      [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding],
    output: resources.output.binding,
    package: packageBindings,
  });
}

function requireDispatch(
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  frames: number,
): AceOpt0011Fp16VaeWindowDispatch {
  const matches = set.dispatches.filter((dispatch) =>
    dispatch.plan.inputFrames === frames
  );
  if (matches.length !== 1) {
    throw new Error(`OPT-0011 expected one ${frames}-frame dispatch`);
  }
  return matches[0]!;
}

function validateChunkSet(
  set: AceOpt0011Fp16VaeChunkDispatchSet,
  arm: "B" | "C",
): void {
  const expected = opt0011C512LongExpectedTopology()[arm];
  const topology = set.topology;
  const kinds = topology.topologies.reduce(
    (totals, item) => ({
      conv1d: totals.conv1d + item.quantumFamilyCounts.conv1d,
      transpose: totals.transpose +
        item.quantumFamilyCounts["conv-transpose1d"],
      snake: totals.snake + item.quantumFamilyCounts.snake,
      add: totals.add + item.quantumFamilyCounts.add,
    }),
    { conv1d: 0, transpose: 0, snake: 0, add: 0 },
  );
  const windowFrames = set.windows.map((entry) => entry.window.latentWindowFrames);
  const mappedIdentity = set.windows.every((entry) =>
    entry.dispatch.plan.inputFrames === entry.window.latentWindowFrames &&
    set.dispatches.includes(entry.dispatch)
  );
  if (
    JSON.stringify(topology.uniqueWindowFrames) !==
      JSON.stringify(expected.uniqueWindowFrames) ||
    JSON.stringify(windowFrames) !== JSON.stringify(expected.windowFrames) ||
    topology.chunkPlan.windows.length !== expected.windowCount ||
    topology.aggregateGraphQuantumCount !== expected.graphQuantumCount ||
    topology.aggregateSequenceQuantumCount !== expected.sequenceQuantumCount ||
    topology.aggregateCommandBufferCountAtBatch8 !== expected.commandBufferCount ||
    topology.uniqueDynamicControlBytes !== expected.controlBytes ||
    topology.maximumFp16WorkspaceBytes !== expected.maximumWorkspaceBytes ||
    kinds.conv1d !== expected.uniqueShapeFamilyCounts.conv1d ||
    kinds.transpose !== expected.uniqueShapeFamilyCounts.transpose ||
    kinds.snake !== expected.uniqueShapeFamilyCounts.snake ||
    kinds.add !== expected.uniqueShapeFamilyCounts.add ||
    !mappedIdentity ||
    set.dispatches.some((dispatch) =>
      dispatch.plan.inputFrames * LATENT_CHANNELS * 4 !==
        dispatch.activeStagingInputBytes ||
      dispatch.plan.inputFrames * LATENT_CHANNELS * 2 !==
        dispatch.activeDecoderInputBytes ||
      dispatch.plan.inputFrames * HOP_LENGTH * AUDIO_CHANNELS * 4 !==
        dispatch.activeOutputBytes
    )
  ) throw new Error(`OPT-0011 ${arm} chunk dispatch topology changed`);
}

export function opt0011C512LongExpectedTopology() {
  return Object.freeze({
    C512: Object.freeze({
      operationCount: 88,
      graphQuantumCount: 7_854,
      sequenceQuantumCount: 7_855,
      familyCounts: Object.freeze({
        conv1d: 4_909,
        transpose: 644,
        snake: 1_611,
        add: 690,
      }),
      controlBytes: 2_010_640,
      workspaceBytes: 251_658_240,
      logicalGpuBytes: 941_702_160,
      computeCommandBufferCount: 982,
      commandBufferCount: 983,
    }),
    B: Object.freeze({
      uniqueWindowFrames: Object.freeze([192, 256]),
      windowFrames: Object.freeze([192, 256, 256, 256, 256, 256, 256, 192]),
      windowCount: 8,
      graphQuantumCount: 29_586,
      sequenceQuantumCount: 29_594,
      commandBufferCount: 3_708,
      controlBytes: 1_768_736,
      maximumWorkspaceBytes: 125_829_120,
      uniqueShapeFamilyCounts: Object.freeze({
        conv1d: 4_302,
        transpose: 565,
        snake: 1_430,
        add: 612,
      }),
      aggregateWindowFamilyCounts: Object.freeze({
        conv1d: 18_440,
        transpose: 2_418,
        snake: 6_112,
        add: 2_616,
      }),
      decodedLatentFrames: 1_920,
      scheduledDecodedFloat32Bytes: 29_491_200,
      retainedOutputFloat32Bytes: LONG_OUTPUT_BYTES,
      logicalGpuBytes: 556_010_272,
    }),
    C: Object.freeze({
      uniqueWindowFrames: Object.freeze([320, 448, 512]),
      windowFrames: Object.freeze([448, 512, 320]),
      windowCount: 3,
      graphQuantumCount: 19_684,
      sequenceQuantumCount: 19_687,
      commandBufferCount: 2_465,
      controlBytes: 5_039_152,
      maximumWorkspaceBytes: 251_658_240,
      uniqueShapeFamilyCounts: Object.freeze({
        conv1d: 12_275,
        transpose: 1_606,
        snake: 4_063,
        add: 1_740,
      }),
      aggregateWindowFamilyCounts: Object.freeze({
        conv1d: 12_275,
        transpose: 1_606,
        snake: 4_063,
        add: 1_740,
      }),
      decodedLatentFrames: 1_280,
      scheduledDecodedFloat32Bytes: 19_660_800,
      retainedOutputFloat32Bytes: LONG_OUTPUT_BYTES,
      logicalGpuBytes: 944_730_672,
    }),
    fullGate: Object.freeze({
      fullWindowExecutionCount: 26,
      partialWindowExecutionCount: 1,
      dispatchCount: 121_191,
      commandBufferCount: 15_179,
      readbackCommandBufferCount: 26,
      readbackCopyCount: 338,
      completedRealIdleCount: 15_172,
      rawU32ComparisonCount: 26_050_560,
      createdBufferCount: 28,
      maximumLiveBufferCount: 18,
      lifetimeCreatedBufferBytes: 1_331_961_680,
    }),
  });
}

function validateActualGateAccounting(actual: ActualGateAccounting): void {
  const expected = opt0011C512LongExpectedTopology().fullGate;
  for (const [name, value] of Object.entries(expected)) {
    if (Reflect.get(actual, name) !== value) {
      throw new Error(
        `OPT-0011 actual gate accounting ${name} changed: ` +
          `${String(Reflect.get(actual, name))}/${String(value)}`,
      );
    }
  }
  if (
    actual.executionTraceCount !==
      actual.fullWindowExecutionCount + actual.partialWindowExecutionCount ||
    actual.queueDrainCount !== actual.commandBufferCount ||
    actual.completedInternalRealIdleCount +
        actual.completedBetweenWindowRealIdleCount !==
      actual.completedRealIdleCount ||
    actual.readbackCopyCount !==
      actual.readbackCommandBufferCount * READBACK_COPY_COUNT ||
    actual.everyCompletedExecutionAggregatedExactlyOnce !== true
  ) throw new Error("OPT-0011 actual gate accounting derivation changed");
}

function compactChunkTopology(
  set: AceOpt0011Fp16VaeChunkDispatchSet,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    maximumWindowFramesProfile: set.topology.maximumWindowFramesProfile,
    logicalLatentFrames: set.topology.chunkPlan.latentFrames,
    chunkFrames: set.topology.chunkPlan.chunkFrames,
    overlapFrames: set.topology.chunkPlan.overlapFrames,
    strideFrames: set.topology.chunkPlan.strideFrames,
    uniqueWindowFrames: set.topology.uniqueWindowFrames,
    windowTopologyIndices: set.topology.windowTopologyIndices,
    windows: Object.freeze(set.windows.map(({ window, dispatch }) =>
      Object.freeze({
        ...window,
        dispatchInputFrames: dispatch.plan.inputFrames,
      })
    )),
    topologies: Object.freeze(set.topology.topologies.map((item) =>
      Object.freeze({
        inputFrames: item.inputFrames,
        operationCount: item.operationCount,
        graphQuantumCount: item.graphQuantumCount,
        sequenceQuantumCount: item.sequenceQuantumCount,
        quantumFamilyCounts: item.quantumFamilyCounts,
        activeStagingInputBytes: item.activeStagingInputBytes,
        activeDecoderInputBytes: item.activeDecoderInputBytes,
        activeOutputBytes: item.activeOutputBytes,
        fp16WorkspaceBytes: item.fp16WorkspaceBytes,
        dynamicControlBytes: item.dynamicControls.byteLength,
        decoderCommandBufferCountAtBatch8:
          item.decoderCommandBufferCountAtBatch8,
        commandBufferCountAtBatch8: item.commandBufferCountAtBatch8,
      })
    )),
    maximumFp16WorkspaceBytes: set.topology.maximumFp16WorkspaceBytes,
    uniqueDynamicControlBytes: set.topology.uniqueDynamicControlBytes,
    aggregateGraphQuantumCount: set.topology.aggregateGraphQuantumCount,
    aggregateSequenceQuantumCount: set.topology.aggregateSequenceQuantumCount,
    aggregateCommandBufferCountAtBatch8:
      set.topology.aggregateCommandBufferCountAtBatch8,
    exactWindowDispatchIdentityMapping: true,
  });
}

async function candidateGraphIdentity(
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  packageBindings: AceOpt0011VaePackageBindings,
): Promise<Readonly<Record<string, unknown>>> {
  const encoder = new TextEncoder();
  const graph = dispatch.quanta.map((quantum) => ({
    sequenceIndex: quantum.sequenceIndex,
    graphQuantumIndex: quantum.graphQuantumIndex,
    operationIndex: quantum.operationIndex,
    operationLabel: quantum.operationLabel,
    operationKind: quantum.operationKind,
    base: quantum.logicalOutputBase,
    count: quantum.logicalOutputCount,
    kernelId: quantum.kernelId,
    controlOffset: quantum.control.byteOffset,
  }));
  const tensor = (value: { logicalTensor: string; record: {
    shard: string;
    byteOffset: number;
    byteLength: number;
    dtype: string;
  } }) => ({
    name: value.logicalTensor,
    shard: value.record.shard,
    byteOffset: value.record.byteOffset,
    byteLength: value.record.byteLength,
    dtype: value.record.dtype,
  });
  const operations = packageBindings.operations.map((operation) => {
    switch (operation.kind) {
      case "conv1d":
        return {
          index: operation.operationIndex,
          label: operation.label,
          kind: operation.kind,
          weight: tensor(operation.weight),
          bias: operation.bias === undefined ? null : tensor(operation.bias),
        };
      case "conv-transpose1d":
        return {
          index: operation.operationIndex,
          label: operation.label,
          kind: operation.kind,
          weight: tensor(operation.weight),
          bias: tensor(operation.bias),
        };
      case "snake":
        return {
          index: operation.operationIndex,
          label: operation.label,
          kind: operation.kind,
          alpha: tensor(operation.alpha),
          beta: tensor(operation.beta),
        };
      case "add":
        return {
          index: operation.operationIndex,
          label: operation.label,
          kind: operation.kind,
        };
    }
  });
  return Object.freeze({
    sequenceQuantumCount: graph.length,
    topologySha256: await sha256Hex(encoder.encode(JSON.stringify(graph))),
    operationCount: operations.length,
    operationBindingSha256:
      await sha256Hex(encoder.encode(JSON.stringify(operations))),
    tensorCount: Object.keys(packageBindings.tensors).length,
    precisionMapSha256: ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
    completeCompactTrace: true,
  });
}

function latentWindow(
  longLatent: Float32Array<ArrayBuffer>,
  window: AceVaeDecodeWindow,
): Float32Array<ArrayBuffer> {
  const start = window.windowStartLatentFrame * LATENT_CHANNELS;
  const end = window.windowEndLatentFrame * LATENT_CHANNELS;
  const result = longLatent.subarray(start, end);
  if (result.length !== window.latentWindowFrames * LATENT_CHANNELS) {
    throw new Error("OPT-0011 latent slice extent changed");
  }
  return result;
}

function coreView(
  decoded: Float32Array,
  window: AceVaeDecodeWindow,
): Float32Array {
  const start = window.discardPrefixAudioFrames * AUDIO_CHANNELS;
  const end = decoded.length -
    window.discardSuffixAudioFrames * AUDIO_CHANNELS;
  const core = decoded.subarray(start, end);
  if (core.length !== window.outputAudioFrames * AUDIO_CHANNELS) {
    throw new Error("OPT-0011 core trim extent changed");
  }
  return core;
}

function writeCore(
  output: Float32Array,
  decoded: Float32Array,
  window: AceVaeDecodeWindow,
  coverage?: Uint8Array,
): void {
  const core = coreView(decoded, window);
  output.set(core, window.outputStartAudioFrame * AUDIO_CHANNELS);
  if (coverage !== undefined) {
    const end = window.outputStartAudioFrame + window.outputAudioFrames;
    for (let frame = window.outputStartAudioFrame; frame < end; frame += 1) {
      coverage[frame] = coverage[frame]! + 1;
    }
  }
}

function requireExactCoverage(coverage: Uint8Array, label: string): void {
  let gapCount = 0;
  let duplicationCount = 0;
  for (const value of coverage) {
    if (value === 0) gapCount += 1;
    else if (value !== 1) duplicationCount += 1;
  }
  if (gapCount !== 0 || duplicationCount !== 0) {
    throw new Error(
      `OPT-0011 ${label} coverage changed: ${gapCount}/${duplicationCount}`,
    );
  }
}

async function mapMaximumOutput(
  buffer: GPUBuffer,
): Promise<Float32Array<ArrayBuffer>> {
  const bytes = Number(buffer.size);
  await buffer.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    return new Float32Array(buffer.getMappedRange(0, bytes).slice(0));
  } finally {
    buffer.unmap();
  }
}

async function validateGuards(
  buffer: GPUBuffer,
  bindingCount: number,
  label: string,
): Promise<void> {
  const bytes = bindingCount * 2 * GUARD_BYTES;
  await buffer.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    const words = new Uint32Array(buffer.getMappedRange(0, bytes));
    for (let index = 0; index < words.length; index += 1) {
      if (words[index] !== GUARD_WORD) {
        throw new Error(`${label} guard changed at word ${index}`);
      }
    }
  } finally {
    buffer.unmap();
  }
}

function validateInactiveOutputTail(
  maximumOutput: Float32Array,
  activeElements: number,
  maximumElements: number,
  label: string,
): Readonly<Record<string, unknown>> {
  if (
    maximumOutput.length !== maximumElements || activeElements > maximumElements
  ) throw new Error(`${label} maximum output extent changed`);
  const words = new Uint32Array(
    maximumOutput.buffer,
    maximumOutput.byteOffset,
    maximumOutput.length,
  );
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = activeElements; index < words.length; index += 1) {
    if (words[index] === OUTPUT_QNAN_WORD) continue;
    mismatchCount += 1;
    firstMismatchIndex ??= index;
  }
  if (mismatchCount !== 0) {
    throw new Error(`${label} wrote its poisoned inactive output tail`);
  }
  return Object.freeze({
    activeElementCount: activeElements,
    inactiveTailElementCount: maximumElements - activeElements,
    comparedInactiveTailU32Count: maximumElements - activeElements,
    qNaNSentinelWord: `0x${OUTPUT_QNAN_WORD.toString(16)}`,
    mismatchCount,
    firstMismatchIndex,
    exact: true,
  });
}

async function scanOutput(
  output: Float32Array,
  expectedElements: number,
): Promise<OutputScan> {
  if (output.length !== expectedElements || output.byteLength !== expectedElements * 4) {
    throw new Error("OPT-0011 output extent changed");
  }
  const bits = new Uint32Array(output.buffer, output.byteOffset, output.length);
  let finiteCount = 0;
  let nonzeroCount = 0;
  let qNaNSentinelCount = 0;
  let stereoDifferenceFrameCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let peak = 0;
  let clampBoundaryCount = 0;
  for (let index = 0; index < output.length; index += 1) {
    const value = output[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (bits[index] === OUTPUT_QNAN_WORD) qNaNSentinelCount += 1;
    if ((index & 1) === 0 && bits[index] !== bits[index + 1]) {
      stereoDifferenceFrameCount += 1;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    peak = Math.max(peak, Math.abs(value));
    if (Math.abs(value) === 65_504) clampBoundaryCount += 1;
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    finiteCount,
    nonzeroCount,
    qNaNSentinelCount,
    stereoDifferenceFrameCount,
    minimum,
    maximum,
    peak,
    clampBoundaryCount,
    sha256: await sha256Hex(bytesOf(output)),
  });
}

function validateCompleteOutput(
  scan: OutputScan,
  expectedElements: number,
  label: string,
): void {
  if (
    scan.elementCount !== expectedElements ||
    scan.byteLength !== expectedElements * 4 ||
    scan.finiteCount !== expectedElements || scan.nonzeroCount === 0 ||
    scan.qNaNSentinelCount !== 0 || scan.stereoDifferenceFrameCount === 0 ||
    scan.clampBoundaryCount !== 0 || !Number.isFinite(scan.minimum) ||
    !Number.isFinite(scan.maximum) || scan.peak === 0
  ) throw new Error(`${label} output completeness/finiteness changed`);
}

export function compareOpt0011C512LongU32(
  left: Float32Array,
  right: Float32Array,
) {
  return compareU32(left, right);
}

function compareU32(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length) throw new Error("U32 extents differ");
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  let worstMismatchIndex: number | null = null;
  let worstXor = 0;
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] === b[index]) continue;
    mismatchCount += 1;
    firstMismatchIndex ??= index;
    const xor = (a[index]! ^ b[index]!) >>> 0;
    if (xor > worstXor) {
      worstXor = xor;
      worstMismatchIndex = index;
    }
  }
  return Object.freeze({
    comparedWordCount: a.length,
    mismatchCount,
    firstMismatchIndex,
    worstMismatchIndex,
    bitExact: mismatchCount === 0,
  });
}

function compareSeams(
  b: Float32Array,
  c: Float32Array,
): Readonly<Record<string, unknown>> {
  const exact = compareU32(b, c);
  if (
    b.length !== LONG_OUTPUT_ELEMENTS || c.length !== LONG_OUTPUT_ELEMENTS ||
    exact.mismatchCount !== 0
  ) throw new Error("OPT-0011 seam comparison requires exact full B/C output");
  const bLatentSeams = Object.freeze([128, 256, 384, 512, 640, 768, 896]);
  const cLatentSeams = Object.freeze([384, 768]);
  return Object.freeze({
    diagnosticOnlyExactBCIdentityIsTheGate: true,
    localRadiusAudioFrames: LOCAL_SEAM_RADIUS_AUDIO_FRAMES,
    percentileDefinition:
      "nearest-rank-p99.9-of-absolute-local-sample-first-differences",
    B: Object.freeze(bLatentSeams.map((latentFrame) =>
      seamMetric(b, c, latentFrame)
    )),
    C: Object.freeze(cLatentSeams.map((latentFrame) =>
      seamMetric(c, b, latentFrame)
    )),
    matchedInterior: compareMatchedInteriors(b, c, bLatentSeams),
    cSeamLatentFramesAreSubsetOfBSeams: true,
    cIntroducedBitDifferenceCount: 0,
    nonFiniteSeamSampleCount: 0,
    coverageGapCount: 0,
    coverageDuplicationCount: 0,
  });
}

function compareMatchedInteriors(
  b: Float32Array,
  c: Float32Array,
  seamLatentFrames: readonly number[],
): Readonly<Record<string, unknown>> {
  const boundaries = [0, ...seamLatentFrames, LONG_FRAMES];
  const regions = boundaries.slice(0, -1).map((left, index) => {
    const right = boundaries[index + 1]!;
    const startAudioFrame = left * HOP_LENGTH +
      LOCAL_SEAM_RADIUS_AUDIO_FRAMES;
    const endAudioFrame = right * HOP_LENGTH -
      LOCAL_SEAM_RADIUS_AUDIO_FRAMES;
    const start = startAudioFrame * AUDIO_CHANNELS;
    const end = endAudioFrame * AUDIO_CHANNELS;
    const bRegion = b.subarray(start, end);
    const cRegion = c.subarray(start, end);
    const exact = compareU32(bRegion, cRegion);
    let bSquares = 0;
    let cSquares = 0;
    let errorSquares = 0;
    for (let element = 0; element < bRegion.length; element += 1) {
      const bValue = bRegion[element]!;
      const cValue = cRegion[element]!;
      bSquares += bValue * bValue;
      cSquares += cValue * cValue;
      const error = cValue - bValue;
      errorSquares += error * error;
    }
    if (exact.mismatchCount !== 0 || bRegion.length === 0) {
      throw new Error(`OPT-0011 matched interior ${index} diverged`);
    }
    return Object.freeze({
      leftBoundaryLatentFrame: left,
      rightBoundaryLatentFrame: right,
      excludedRadiusAudioFramesAtEachBoundary:
        LOCAL_SEAM_RADIUS_AUDIO_FRAMES,
      startAudioFrame,
      endAudioFrameExclusive: endAudioFrame,
      interleavedElementCount: bRegion.length,
      exactU32Comparison: exact,
      bRms: Math.sqrt(bSquares / bRegion.length),
      cRms: Math.sqrt(cSquares / cRegion.length),
      bcErrorRms: Math.sqrt(errorSquares / bRegion.length),
    });
  });
  return Object.freeze({
    classification:
      "diagnostic-regions-between-all-B-seams-excluding-seam-neighborhoods",
    regionCount: regions.length,
    comparedInterleavedElementCount: regions.reduce(
      (sum, region) => sum + region.interleavedElementCount,
      0,
    ),
    everyRegionBitExact: true,
    regions: Object.freeze(regions),
  });
}

function seamMetric(
  primary: Float32Array,
  comparator: Float32Array,
  latentFrame: number,
): Readonly<Record<string, unknown>> {
  const audioFrame = latentFrame * HOP_LENGTH;
  const channels = [0, 1].map((channel) => {
    const before2 = primary[(audioFrame - 2) * 2 + channel]!;
    const before = primary[(audioFrame - 1) * 2 + channel]!;
    const at = primary[audioFrame * 2 + channel]!;
    const after = primary[(audioFrame + 1) * 2 + channel]!;
    const valueJump = Math.abs(at - before);
    const firstDifferenceJump = Math.abs((after - at) - (before - before2));
    const start = Math.max(1, audioFrame - LOCAL_SEAM_RADIUS_AUDIO_FRAMES);
    const end = Math.min(
      LONG_OUTPUT_FRAMES - 1,
      audioFrame + LOCAL_SEAM_RADIUS_AUDIO_FRAMES,
    );
    const deltas: number[] = [];
    let sumSquares = 0;
    let errorSquares = 0;
    let mismatchCount = 0;
    let sampleCount = 0;
    for (let frame = start; frame <= end; frame += 1) {
      const index = frame * 2 + channel;
      const value = primary[index]!;
      const other = comparator[index]!;
      if (!Number.isFinite(value) || !Number.isFinite(other)) {
        throw new Error("OPT-0011 seam metric encountered non-finite data");
      }
      deltas.push(Math.abs(value - primary[(frame - 1) * 2 + channel]!));
      sumSquares += value * value;
      const error = other - value;
      errorSquares += error * error;
      if (f32Bits(value) !== f32Bits(other)) mismatchCount += 1;
      sampleCount += 1;
    }
    deltas.sort((left, right) => left - right);
    const p999 = deltas[Math.max(0, Math.ceil(deltas.length * 0.999) - 1)]!;
    const rank = deltas.filter((value) => value <= valueJump).length /
      deltas.length;
    return Object.freeze({
      channel,
      valueBefore: before,
      valueAt: at,
      valueAfter: after,
      valueJump,
      firstDifferenceJump,
      localRms: Math.sqrt(sumSquares / sampleCount),
      localBcErrorRms: Math.sqrt(errorSquares / sampleCount),
      localBcU32MismatchCount: mismatchCount,
      localP999AbsoluteFirstDifference: p999,
      valueJumpToLocalP999Ratio: p999 === 0 ? null : valueJump / p999,
      valueJumpLocalPercentileRank: rank,
    });
  });
  return Object.freeze({
    latentFrame,
    audioFrame,
    interleavedElementIndex: audioFrame * AUDIO_CHANNELS,
    channels: Object.freeze(channels),
  });
}

function f32Bits(value: number): number {
  const values = new Float32Array([value]);
  return new Uint32Array(values.buffer)[0]!;
}

function packageReceipt(prepared: PreparedPackage): Readonly<Record<string, unknown>> {
  return Object.freeze({
    manifestSha256: prepared.loaded.manifestSha256,
    manifestByteLength: prepared.loaded.manifestByteLength,
    profile: prepared.loaded.manifest.profile,
    converterRevision: prepared.loaded.manifest.provenance.converterRevision,
    vaeTensorCount: prepared.tensorCount,
    vaeFileCount: prepared.files.length,
    residentBytes: prepared.residentBytes,
    files: Object.freeze(prepared.files.map((file) => Object.freeze({
      name: file.name,
      byteLength: file.byteLength,
      sha256: file.sha256,
    }))),
  });
}

function validateExecutionTrace(
  trace: MutableExecutionTrace,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
  label: string,
): void {
  const computeCount = dispatch.decoderCommandBufferCountAtBatch8;
  const totalCount = dispatch.commandBufferCountAtBatch8;
  const finalBatchDispatches = dispatch.quanta.length -
    (computeCount - 1) * ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
  if (
    trace.commands.length !== totalCount ||
    trace.submissionCount !== totalCount || trace.drainCount !== totalCount ||
    trace.writeBufferCount !== 0 ||
    trace.commands.some((command) =>
      !command.finished || !command.submitted || !command.drained
    )
  ) throw new Error(`${label} physical execution accounting changed`);
  for (let index = 0; index < computeCount; index += 1) {
    const command = trace.commands[index]!;
    const expectedDispatches = index + 1 === computeCount
      ? finalBatchDispatches
      : ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER;
    if (
      command.label !== `${label}-batch-${index}` ||
      command.computePassCount !== 1 ||
      command.dispatchCount !== expectedDispatches ||
      command.copyCount !== 0 || command.clearCount !== 0
    ) throw new Error(`${label} compute batch ${index} changed`);
  }
  const readback = trace.commands[computeCount]!;
  if (
    readback.label !== `${label}-readback` ||
    readback.computePassCount !== 0 || readback.dispatchCount !== 0 ||
    readback.copyCount !== READBACK_COPY_COUNT || readback.clearCount !== 0
  ) throw new Error(`${label} readback topology changed`);
}

function compactTrace(
  trace: MutableExecutionTrace,
  dispatch: AceOpt0011Fp16VaeWindowDispatch,
): Readonly<Record<string, unknown>> {
  const compute = trace.commands.slice(0, dispatch.decoderCommandBufferCountAtBatch8);
  const readback = trace.commands.at(-1)!;
  return Object.freeze({
    executionId: trace.id,
    inputFrames: dispatch.plan.inputFrames,
    sequenceQuantumCount: dispatch.quanta.length,
    commandBufferCount: trace.commands.length,
    computeCommandBufferCount: compute.length,
    readbackCommandBufferCount: 1,
    submissionCount: trace.submissionCount,
    queueDrainCount: trace.drainCount,
    computePassCount: compute.reduce((sum, command) =>
      sum + command.computePassCount, 0),
    dispatchCount: compute.reduce((sum, command) => sum + command.dispatchCount, 0),
    computeBatchDispatchHistogram: histogram(
      compute.map((command) => command.dispatchCount),
    ),
    readback: Object.freeze({
      label: readback.label,
      copyCount: readback.copyCount,
    }),
    everyCommandFinishedSubmittedAndDrained: true,
  });
}

function histogram(values: readonly number[]): Readonly<Record<string, number>> {
  const result: Record<string, number> = {};
  for (const value of values) {
    result[String(value)] = (result[String(value)] ?? 0) + 1;
  }
  return Object.freeze(result);
}

class DeviceAudit {
  readonly device: GPUDevice;
  private scope: AuditScope = "setup";
  private readonly records = new Map<GPUBuffer, BufferRecord>();
  private readonly shaderSources = new Map<AuditScope, string[]>();
  private readonly commandBuffers = new WeakMap<
    GPUCommandBuffer,
    MutableCommandRecord
  >();
  private active: MutableExecutionTrace | undefined;
  private maximumLive = 0;
  private readonly completedExecutions: MutableExecutionTrace[] = [];
  private completedInternalRealIdleCount = 0;
  private completedBetweenWindowRealIdleCount = 0;
  private rawU32ComparisonCount = 0;

  constructor(rawDevice: GPUDevice) {
    const queue = this.wrapQueue(rawDevice.queue);
    this.device = new Proxy(rawDevice, {
      get: (target, property) => {
        if (property === "queue") return queue;
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor) =>
            this.trackBuffer(target.createBuffer(descriptor), descriptor);
        }
        if (property === "createShaderModule") {
          return (descriptor: GPUShaderModuleDescriptor) => {
            const list = this.shaderSources.get(this.scope) ?? [];
            list.push(descriptor.code);
            this.shaderSources.set(this.scope, list);
            return target.createShaderModule(descriptor);
          };
        }
        if (property === "createCommandEncoder") {
          return (descriptor?: GPUCommandEncoderDescriptor) =>
            this.wrapEncoder(target.createCommandEncoder(descriptor), descriptor);
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    });
  }

  setScope(scope: AuditScope): void {
    if (this.active !== undefined) throw new Error("scope changed during trace");
    this.scope = scope;
  }

  beginExecution(id: string): void {
    if (this.active !== undefined) throw new Error("execution traces overlap");
    this.active = {
      id,
      commands: [],
      submissionCount: 0,
      drainCount: 0,
      writeBufferCount: 0,
    };
  }

  endExecution(): MutableExecutionTrace {
    const trace = this.active;
    if (trace === undefined) throw new Error("execution trace is absent");
    this.active = undefined;
    this.completedExecutions.push(trace);
    return trace;
  }

  async deliverRealQueueEmptyIdle(
    kind: "internal" | "between-window",
  ): Promise<void> {
    await new Promise<void>((resolve) => setTimeout(resolve, 1));
    if (kind === "internal") this.completedInternalRealIdleCount += 1;
    else this.completedBetweenWindowRealIdleCount += 1;
  }

  compareU32(left: Float32Array, right: Float32Array) {
    const result = compareU32(left, right);
    this.rawU32ComparisonCount += result.comparedWordCount;
    return result;
  }

  aggregateActualExecutionAccounting(): ActualGateAccounting {
    if (this.active !== undefined) {
      throw new Error("OPT-0011 cannot aggregate an active execution trace");
    }
    const commands = this.completedExecutions.flatMap((trace) => trace.commands);
    const readbacks = commands.filter((command) => command.copyCount !== 0);
    if (
      readbacks.some((command) =>
        command.copyCount !== READBACK_COPY_COUNT ||
        command.computePassCount !== 0 || command.dispatchCount !== 0
      ) ||
      this.completedExecutions.some((trace) =>
        trace.commands.filter((command) => command.copyCount !== 0).length > 1
      )
    ) throw new Error("OPT-0011 aggregate readback trace changed");
    const fullWindowExecutionCount = this.completedExecutions.filter((trace) =>
      trace.commands.some((command) => command.copyCount !== 0)
    ).length;
    const values = [...this.records.values()];
    return Object.freeze({
      fullWindowExecutionCount,
      partialWindowExecutionCount:
        this.completedExecutions.length - fullWindowExecutionCount,
      dispatchCount: commands.reduce(
        (sum, command) => sum + command.dispatchCount,
        0,
      ),
      commandBufferCount: commands.length,
      readbackCommandBufferCount: readbacks.length,
      readbackCopyCount: readbacks.reduce(
        (sum, command) => sum + command.copyCount,
        0,
      ),
      completedRealIdleCount:
        this.completedInternalRealIdleCount +
        this.completedBetweenWindowRealIdleCount,
      rawU32ComparisonCount: this.rawU32ComparisonCount,
      createdBufferCount: values.length,
      maximumLiveBufferCount: this.maximumLive,
      lifetimeCreatedBufferBytes: values.reduce(
        (sum, record) => sum + record.size,
        0,
      ),
      executionTraceCount: this.completedExecutions.length,
      queueDrainCount: this.completedExecutions.reduce(
        (sum, trace) => sum + trace.drainCount,
        0,
      ),
      completedInternalRealIdleCount: this.completedInternalRealIdleCount,
      completedBetweenWindowRealIdleCount:
        this.completedBetweenWindowRealIdleCount,
      everyCompletedExecutionAggregatedExactlyOnce: true,
    });
  }

  liveCount(scope: AuditScope): number {
    return [...this.records.values()].filter((record) =>
      record.scope === scope && !record.destroyed
    ).length;
  }

  totalLiveCount(): number {
    return [...this.records.values()].filter((record) => !record.destroyed).length;
  }

  liveBytes(scope: AuditScope): number {
    return [...this.records.values()].filter((record) =>
      record.scope === scope && !record.destroyed
    ).reduce((sum, record) => sum + record.size, 0);
  }

  async shaderIdentity(scope: AuditScope): Promise<Readonly<Record<string, unknown>>> {
    const sources = this.shaderSources.get(scope) ?? [];
    const records = await Promise.all([...new Set(sources)].map(async (source) => {
      const bytes = new TextEncoder().encode(source);
      return Object.freeze({
        byteLength: bytes.byteLength,
        sha256: await sha256Hex(bytes),
      });
    }));
    records.sort((left, right) => left.sha256.localeCompare(right.sha256));
    return Object.freeze({
      shaderModuleCreateCount: sources.length,
      uniqueShaderCount: records.length,
      uniqueShaders: Object.freeze(records),
      aggregateSha256: await sha256Hex(
        new TextEncoder().encode(JSON.stringify(records)),
      ),
      everyExecutedShaderCaptured: true,
    });
  }

  destroyAll(): void {
    for (const [buffer, record] of this.records) {
      if (!record.destroyed) buffer.destroy();
    }
  }

  scopeBytesReceipt(): Readonly<Record<string, unknown>> {
    return Object.freeze(Object.fromEntries(
      (["setup", "package", "B", "C"] as const).map((scope) => {
        const records = [...this.records.values()].filter((record) =>
          record.scope === scope
        );
        return [scope, Object.freeze({
          createdBufferCount: records.length,
          createdBufferBytes: records.reduce((sum, record) => sum + record.size, 0),
          destroyedBufferCount: records.filter((record) => record.destroyed).length,
          liveBufferCount: records.filter((record) => !record.destroyed).length,
        })];
      }),
    ));
  }

  resourceSummary(): Readonly<Record<string, number | boolean>> {
    const values = [...this.records.values()];
    return Object.freeze({
      destructionTrackingSupported: true,
      createdBufferCount: values.length,
      destroyedBufferCount: values.filter((record) => record.destroyed).length,
      liveBufferCount: values.filter((record) => !record.destroyed).length,
      totalDestroyCallCount: values.reduce(
        (sum, record) => sum + record.destroyCalls,
        0,
      ),
      maximumLiveBufferCount: this.maximumLive,
      createdBufferBytes: values.reduce((sum, record) => sum + record.size, 0),
    });
  }

  private trackBuffer(
    buffer: GPUBuffer,
    descriptor: GPUBufferDescriptor,
  ): GPUBuffer {
    const record: BufferRecord = {
      scope: this.scope,
      label: descriptor.label ?? "",
      size: Number(descriptor.size),
      destroyCalls: 0,
      destroyed: false,
    };
    this.records.set(buffer, record);
    this.maximumLive = Math.max(this.maximumLive, this.totalLiveCount());
    const destroy = buffer.destroy.bind(buffer);
    Object.defineProperty(buffer, "destroy", {
      configurable: true,
      value: () => {
        if (record.destroyed) return;
        record.destroyCalls += 1;
        record.destroyed = true;
        destroy();
      },
    });
    return buffer;
  }

  private wrapQueue(queue: GPUQueue): GPUQueue {
    return new Proxy(queue, {
      get: (target, property) => {
        if (property === "submit") {
          return (buffers: Iterable<GPUCommandBuffer>) => {
            const list = [...buffers];
            if (this.active !== undefined) {
              if (list.length !== 1) throw new Error("OPT-0011 submitted !=1 CB");
              const record = this.commandBuffers.get(list[0]!);
              if (record === undefined || record.submitted) {
                throw new Error("OPT-0011 submitted an unknown command buffer");
              }
              record.submitted = true;
              this.active.submissionCount += 1;
            }
            target.submit(list);
          };
        }
        if (property === "onSubmittedWorkDone") {
          return async () => {
            await target.onSubmittedWorkDone();
            if (this.active !== undefined) {
              const record = [...this.active.commands]
                .reverse().find((candidate) =>
                  candidate.submitted && !candidate.drained
                );
              if (record === undefined) {
                throw new Error("OPT-0011 drain had no pending command buffer");
              }
              record.drained = true;
              this.active.drainCount += 1;
            }
          };
        }
        if (property === "writeBuffer") {
          return (...args: Parameters<GPUQueue["writeBuffer"]>) => {
            if (this.active !== undefined) this.active.writeBufferCount += 1;
            return target.writeBuffer(...args);
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    });
  }

  private wrapEncoder(
    encoder: GPUCommandEncoder,
    descriptor?: GPUCommandEncoderDescriptor,
  ): GPUCommandEncoder {
    if (this.active === undefined) return encoder;
    const record: MutableCommandRecord = {
      label: descriptor?.label ?? "",
      computePassCount: 0,
      dispatchCount: 0,
      copyCount: 0,
      clearCount: 0,
      finished: false,
      submitted: false,
      drained: false,
    };
    this.active.commands.push(record);
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (passDescriptor?: GPUComputePassDescriptor) => {
            record.computePassCount += 1;
            const pass = target.beginComputePass(passDescriptor);
            return new Proxy(pass, {
              get: (passTarget, passProperty) => {
                if (passProperty === "dispatchWorkgroups") {
                  return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroups"]>) => {
                    record.dispatchCount += 1;
                    return passTarget.dispatchWorkgroups(...args);
                  };
                }
                const value = Reflect.get(
                  passTarget,
                  passProperty,
                  passTarget,
                ) as unknown;
                return typeof value === "function"
                  ? (value as (...args: unknown[]) => unknown).bind(passTarget)
                  : value;
              },
            });
          };
        }
        if (property === "copyBufferToBuffer") {
          return (...args: Parameters<GPUCommandEncoder["copyBufferToBuffer"]>) => {
            record.copyCount += 1;
            return target.copyBufferToBuffer(...args);
          };
        }
        if (property === "clearBuffer") {
          return (...args: Parameters<GPUCommandEncoder["clearBuffer"]>) => {
            record.clearCount += 1;
            return target.clearBuffer(...args);
          };
        }
        if (property === "finish") {
          return (finishDescriptor?: GPUCommandBufferDescriptor) => {
            if (record.finished) throw new Error("OPT-0011 encoder finished twice");
            record.finished = true;
            const command = target.finish(finishDescriptor);
            this.commandBuffers.set(command, record);
            return command;
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    });
  }
}

function validateWorkerIdentity(identity: Opt0011C512LongRunIdentity): void {
  const roundTrip = new URLSearchParams({
    harnessCommit: identity.harnessCommit,
    runtimeCommit: identity.runtimeCommit,
    machineModel: identity.machineModel,
    osVersion: identity.osVersion,
    osBuild: identity.osBuild,
    browserVersion: identity.browserVersion,
    gpuCoreCount: String(identity.gpuCoreCount),
    memoryBytes: String(identity.memoryBytes),
  });
  if (
    JSON.stringify(parseOpt0011C512LongRunIdentity(roundTrip)) !==
      JSON.stringify(identity)
  ) throw new Error("OPT-0011 C512/long worker identity changed during clone");
}

function startWorkerHeartbeat(): HeartbeatController {
  let tickCount = 0;
  let last = performance.now();
  let maximumGap = 0;
  const timer = setInterval(() => {
    const now = performance.now();
    maximumGap = Math.max(maximumGap, now - last);
    last = now;
    tickCount += 1;
  }, 10);
  let receipt: Readonly<Record<string, unknown>> | undefined;
  return Object.freeze({
    stop() {
      if (receipt !== undefined) return receipt;
      clearInterval(timer);
      receipt = Object.freeze({
        tickCount,
        maximumTimerGapMilliseconds: maximumGap,
        observed: tickCount > 0,
        livenessOnlyNoResponsivenessThresholdOrClaim: true,
      });
      return receipt;
    },
  });
}

function startPageHeartbeat(): HeartbeatController {
  let stopped = false;
  let frameCount = 0;
  let timerCount = 0;
  let lastFrame = performance.now();
  let lastTimer = performance.now();
  let maximumFrameGap = 0;
  let maximumTimerGap = 0;
  let frameHandle = 0;
  const animate = (now: number): void => {
    if (stopped) return;
    maximumFrameGap = Math.max(maximumFrameGap, now - lastFrame);
    lastFrame = now;
    frameCount += 1;
    frameHandle = requestAnimationFrame(animate);
  };
  frameHandle = requestAnimationFrame(animate);
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGap = Math.max(maximumTimerGap, now - lastTimer);
    lastTimer = now;
    timerCount += 1;
  }, 16);
  let receipt: Readonly<Record<string, unknown>> | undefined;
  return Object.freeze({
    stop() {
      if (receipt !== undefined) return receipt;
      stopped = true;
      cancelAnimationFrame(frameHandle);
      clearInterval(timer);
      receipt = Object.freeze({
        frameCount,
        timerCount,
        maximumAnimationFrameGapMilliseconds: maximumFrameGap,
        maximumTimerGapMilliseconds: maximumTimerGap,
        observed: frameCount > 0 && timerCount > 0,
        livenessOnlyNoResponsivenessThresholdOrClaim: true,
      });
      return receipt;
    },
  });
}

function safeStopHeartbeat(heartbeat: HeartbeatController): unknown {
  try {
    return heartbeat.stop();
  } catch (error) {
    return errorReceipt(error);
  }
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function float32Copy(bytes: Uint8Array): Float32Array<ArrayBuffer> {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error("OPT-0011 latent byte extent is not Float32-aligned");
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function detachFloat32(values: Float32Array): Float32Array<ArrayBuffer> {
  const copy = new Float32Array(values.length);
  copy.set(values);
  return copy;
}

function bytesOf(values: Float32Array): Uint8Array {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requiredIdentity(parameters: URLSearchParams, name: string): string {
  const values = parameters.getAll(name);
  if (values.length !== 1 || values[0] === "" || values[0]!.trim() === "") {
    throw new Error(`OPT-0011 C512/long gate requires one ${name}`);
  }
  return values[0]!;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error || error instanceof DOMException) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error instanceof Error ? error.stack ?? null : null,
    });
  }
  return Object.freeze({ name: "UnknownError", message: String(error) });
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0011-fp16-vae-c512-long-correctness-v1",
    status: "failed",
    experimentId: "OPT-0011",
    error: errorReceipt(error),
  });
}

export function parseOpt0011C512LongRawResultChunkOffset(value: string): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error("OPT-0011 C512/long raw-result offset is not canonical");
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error("OPT-0011 C512/long raw-result offset is unsafe");
  }
  return offset;
}

export function sliceOpt0011C512LongRawResultChunk(
  raw: string,
  offset: number,
): Readonly<{
  readonly chunk: string;
  readonly start: number;
  readonly end: number;
  readonly nextOffset: number;
  readonly totalCodeUnits: number;
  readonly complete: boolean;
}> {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > raw.length) {
    throw new Error("OPT-0011 C512/long raw-result offset is invalid");
  }
  let end = Math.min(
    offset + OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS,
    raw.length,
  );
  if (
    end < raw.length && end > offset &&
    isHighSurrogate(raw.charCodeAt(end - 1)) &&
    isLowSurrogate(raw.charCodeAt(end))
  ) end -= 1;
  return Object.freeze({
    chunk: raw.slice(offset, end),
    start: offset,
    end,
    nextOffset: end,
    totalCodeUnits: raw.length,
    complete: end === raw.length,
  });
}

function installRawResultChunkRetrieval(): void {
  const input = requireElement<HTMLInputElement>('input[name="rawResultOffset"]');
  const button = requireElement<HTMLButtonElement>("#publish-raw-result-chunk");
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  let sequence = 0;
  button.addEventListener("click", () => {
    output.textContent = "";
    output.dataset.state = "publishing";
    output.dataset.publicationSequence = String(++sequence);
    try {
      const raw = Reflect.get(globalThis, RAW_RESULT_GLOBAL);
      if (typeof raw !== "string") throw new Error("raw result unavailable");
      const slice = sliceOpt0011C512LongRawResultChunk(
        raw,
        parseOpt0011C512LongRawResultChunkOffset(input.value),
      );
      output.textContent = slice.chunk;
      output.dataset.startOffset = String(slice.start);
      output.dataset.endOffsetExclusive = String(slice.end);
      output.dataset.chunkCodeUnitLength = String(slice.chunk.length);
      output.dataset.totalCodeUnitLength = String(slice.totalCodeUnits);
      output.dataset.done = String(slice.complete);
      output.dataset.state = "published";
      input.value = String(slice.nextOffset);
    } catch (error) {
      output.dataset.state = "failed";
      output.textContent = JSON.stringify(errorReceipt(error));
    }
  });
}

function finishPage(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  updateProgress(status);
  const raw = JSON.stringify(result);
  if (!Reflect.defineProperty(globalThis, RAW_RESULT_GLOBAL, {
    value: raw,
    configurable: false,
    enumerable: false,
    writable: false,
  })) throw new Error("OPT-0011 C512/long could not publish receipt");
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    schema: result["schema"] ?? null,
    status,
    experimentId: "OPT-0011",
    classification: result["classification"] ?? null,
    rawResultJsonCodeUnitLength: raw.length,
    rawResultRetrieval: "bounded-restartable-dom-chunks-from-page-start",
    rawResultChunkCodeUnitLimit:
      OPT_0011_C512_LONG_RAW_RESULT_CHUNK_CODE_UNITS,
    fullReceiptIntentionallyKeptOutOfDom: true,
  }, null, 2);
  const input = requireElement<HTMLInputElement>('input[name="rawResultOffset"]');
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  input.value = "0";
  output.textContent = "";
  output.dataset.state = "ready";
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0011 element ${selector}`);
  return element;
}

function isHighSurrogate(value: number): boolean {
  return value >= 0xd800 && value <= 0xdbff;
}

function isLowSurrogate(value: number): boolean {
  return value >= 0xdc00 && value <= 0xdfff;
}
