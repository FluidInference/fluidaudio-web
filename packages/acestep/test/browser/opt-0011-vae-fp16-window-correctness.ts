/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />
/// <reference lib="webworker" />

import fixtureSource from
  "../../benchmark/opt-0011-vae-fp16-storage-window.ts?raw";
import { createAceOpt0011LatentFixture } from
  "../../benchmark/opt-0011-vae-fp16-storage-window.js";
import acquireSource from "../../src/model/acquire.ts?raw";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import gpuTensorsSource from "../../src/model/gpu-tensors.ts?raw";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  ACE_PACKAGE_CONVERTER_REVISION,
  resolveAceLogicalTensor,
  type AcePackageFileRecord,
} from "../../src/model/manifest.js";
import packageSource from "../../src/model/package.ts?raw";
import {
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../../src/model/package.js";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import {
  AceCooperativeGpuScheduler,
} from "../../src/runtime/scheduler.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
} from "../../src/webgpu/device.js";
import conv1dSource from "../../src/webgpu/kernels/vae-conv1d-fp16.ts?raw";
import {
  aceFp16VaeConv1dWgsl,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import convTransposeSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import { aceFp16VaeConvTranspose1dWgsl } from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import pointwiseSource from
  "../../src/webgpu/kernels/vae-pointwise-fp16.ts?raw";
import {
  aceFp16VaeAddWgsl,
  aceFp16VaeIngressWgsl,
} from "../../src/webgpu/kernels/vae-pointwise-fp16.js";
import snakeSource from "../../src/webgpu/kernels/vae-snake-fp16.ts?raw";
import { aceFp16VaeSnakeWgsl } from
  "../../src/webgpu/kernels/vae-snake-fp16.js";
import oraclePrimitivesSource from
  "../../src/webgpu/kernels/vae-primitives.ts?raw";
import {
  aceCorrectnessVaeAddWgsl,
  aceCorrectnessVaeConv1dWgsl,
  aceCorrectnessVaeConvTranspose1dPartWgsl,
  aceCorrectnessVaeConvTranspose1dWgsl,
  aceCorrectnessVaeSnakeWgsl,
} from "../../src/webgpu/kernels/vae-primitives.js";
import oracleBackendSource from "../../src/webgpu/vae-backend.ts?raw";
import {
  ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
  AceVaeChunkGpuBackend,
} from "../../src/webgpu/vae-backend.js";
import { planAceVaeChunkedDecode } from "../../src/webgpu/vae-chunks.js";
import decoderSource from "../../src/webgpu/vae-decoder.ts?raw";
import {
  planAceVaeDecoder,
  type AceVaeDecoderGraphPlan,
} from "../../src/webgpu/vae-decoder.js";
import decoderRuntimeSource from "../../src/webgpu/vae-fp16-decoder.ts?raw";
import {
  ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT,
  ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT,
  ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT,
  AceOpt0011Fp16VaeDecoderRuntime,
  type AceOpt0011Fp16VaeDecoderDispatch,
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
  ACE_VAE_FP32_ORACLE_MANIFEST_BYTES,
  ACE_VAE_FP32_ORACLE_MANIFEST_SHA256,
} from "../../src/webgpu/vae-fp16-profile.js";

export const OPT_0011_FP16_WINDOW_RUNTIME_COMMIT =
  "d5178ed84e3144e609c461af44e0c71d75d565ba" as const;
export const OPT_0011_FP16_WINDOW_RUNTIME_SOURCE_SHA256 =
  "15e3a98ceec3b6d169d7ca78b921d2abde34c43ee0d3abf81b4314be0b483964" as const;
export const OPT_0011_FP16_WINDOW_FIXTURE_SHA256 =
  "55333d3ae4a0aca83dc1509b837c577f54646924e658e01e53889dc8a5a44875" as const;
export const OPT_0011_FP16_WINDOW_REFERENCE_MANIFEST_PATH =
  "/model/files-reference/manifest.json" as const;
export const OPT_0011_FP16_WINDOW_CANDIDATE_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json" as const;
export const OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS = 32_768;

const FRAMES = 256;
const CHANNELS = 64;
const LATENT_ELEMENTS = 16_384;
const LATENT_BYTES = 65_536;
const OUTPUT_ELEMENTS = 983_040;
const OUTPUT_BYTES = 3_932_160;
const OUTPUT_FRAMES = 491_520;
const GUARD_BYTES = 256;
const GUARD_WORD = 0xa55a_5aa5;
const OUTPUT_QNAN_WORD = 0x7fc5_0011;
const ORACLE_VAE_TENSORS = 146;
const ORACLE_VAE_FILES = 8;
const ORACLE_VAE_BYTES = 337_583_104;
const B_OPERATION_COUNT = 88;
const B_SEQUENCE_QUANTA = 3_943;
const B_GRAPH_QUANTA = 3_942;
const A_PRIMITIVE_DISPATCHES = 3_988;
const B_COMPUTE_COMMAND_BUFFERS = 493;
const TOTAL_COMMAND_BUFFERS = 494;
const COMPLETED_IDLE_TURNS = 493;
const A_READBACK_COPIES = 1;
const B_READBACK_COPIES = 13;
const RAW_RESULT_GLOBAL = "__ACE_OPT_0011_FP16_WINDOW_RAW_RESULT_JSON__";

export const OPT_0011_FP16_WINDOW_BOUNDS = Object.freeze({
  maximumNormalizedRmsError: 0.003,
  minimumSnrDecibels: 50,
  minimumCorrelation: 0.9999,
  maximumRelativeRmsDrift: 0.005,
  maximumRelativeEnergyDrift: 0.005,
  maximumRelativePeakDrift: 0.01,
  maximumDcDriftScale: 0.001,
  maximumNormalizedAbsoluteError: 0.02,
});

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
  ["oracle-backend", oracleBackendSource,
    "bc649cead040c20eefc0fbf3384953c9e5e271597584bfc4904d5a0370fffa71"],
  ["decoder-graph", decoderSource,
    "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e"],
  ["oracle-primitives", oraclePrimitivesSource,
    "d6dc03f49f07ac10be40f391b29f52720d914622cfe04688f5fc9d14b59b5e4d"],
  ["fp16-decoder-runtime", decoderRuntimeSource,
    OPT_0011_FP16_WINDOW_RUNTIME_SOURCE_SHA256],
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

export interface Opt0011Fp16WindowRunIdentity {
  readonly harnessCommit: string;
  readonly runtimeCommit: typeof OPT_0011_FP16_WINDOW_RUNTIME_COMMIT;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

interface WorkerRunMessage {
  readonly type: "run";
  readonly identity: Opt0011Fp16WindowRunIdentity;
}

interface PreparedPackage {
  readonly loaded: AceLoadedPackageManifest;
  readonly files: readonly AcePackageFileRecord[];
  readonly tensorCount: number;
  readonly residentBytes: number;
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

export type Opt0011ExecutionTraceArm = "oracle" | "candidate";

export interface Opt0011ExecutionTraceTopology {
  readonly id: string;
  readonly commandLabels: readonly string[];
  readonly computePassCounts: readonly number[];
  readonly dispatchCounts: readonly number[];
  readonly copyCounts: readonly number[];
  readonly clearCounts: readonly number[];
  readonly submissionCount: number;
  readonly drainCount: number;
  readonly writeBufferCount: number;
  readonly incompleteCommandCount: number;
}

interface BufferRecord {
  readonly scope: "A" | "B" | "setup";
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyCalls: number;
  destroyed: boolean;
}

interface GuardedBinding {
  readonly label: string;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly payloadBytes: number;
}

interface BResources {
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

interface FullExecution {
  readonly output: Float32Array<ArrayBuffer>;
  readonly scan: OutputScan;
  readonly trace: Readonly<Record<string, unknown>>;
}

interface HeartbeatController {
  stop(): Readonly<Record<string, unknown>>;
}

if (typeof document !== "undefined") installPage();
else if (
  typeof self !== "undefined" &&
  new URL(self.location.href).searchParams.get("dedicatedWorker") === "1"
) installWorker();

export function parseOpt0011Fp16WindowRunIdentity(
  parameters: URLSearchParams,
): Opt0011Fp16WindowRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  const runtimeCommit = requiredIdentity(parameters, "runtimeCommit");
  if (!/^[0-9a-f]{40}$/u.test(harnessCommit)) {
    throw new Error("OPT-0011 FP16 window requires a full harness commit");
  }
  if (runtimeCommit !== OPT_0011_FP16_WINDOW_RUNTIME_COMMIT) {
    throw new Error("OPT-0011 FP16 window runtime commit changed");
  }
  const positiveInteger = (name: string): number => {
    const value = Number(requiredIdentity(parameters, name));
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`OPT-0011 FP16 window requires positive ${name}`);
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
    let identity: Opt0011Fp16WindowRunIdentity;
    try {
      identity = parseOpt0011Fp16WindowRunIdentity(
        new URLSearchParams(window.location.search),
      );
    } catch (error) {
      finishPage("failed", failureReceipt(error));
      return;
    }
    const pageHeartbeat = startPageHeartbeat();
    const workerUrl = new URL(
      "./opt-0011-vae-fp16-window-correctness.ts",
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
      const workerResult = data["result"] as Readonly<Record<string, unknown>>;
      worker.terminate();
      const pageLiveness = pageHeartbeat.stop();
      if (status === "passed" && pageLiveness["observed"] !== true) {
        finishPage("failed", Object.freeze({
          ...failureReceipt(new Error("OPT-0011 page heartbeat was not live")),
          workerResult,
          pageHeartbeat: pageLiveness,
        }));
        return;
      }
      finishPage(status, Object.freeze({
        ...workerResult,
        pageHeartbeat: pageLiveness,
      }));
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
          schema: "ace-opt-0011-fp16-vae-b256-window-correctness-v1",
          status: "failed",
          experimentId: "OPT-0011",
          error: errorReceipt(error),
          workerHeartbeat: safeStopHeartbeat(heartbeat),
        }),
      }),
    );
  });
}

async function runWorker(
  identity: Opt0011Fp16WindowRunIdentity,
  heartbeat: HeartbeatController,
): Promise<Readonly<Record<string, unknown>>> {
  validateWorkerIdentity(identity);
  postProgress("authenticating frozen sources and OPT-0011 latent fixture");
  const sourceAuthority = await authenticateSources();
  const latentBytes = createAceOpt0011LatentFixture(FRAMES);
  if (
    latentBytes.byteLength !== LATENT_BYTES ||
    await sha256Hex(latentBytes) !== OPT_0011_FP16_WINDOW_FIXTURE_SHA256
  ) throw new Error("OPT-0011 B-256 latent fixture identity changed");
  const latentBuffer = new ArrayBuffer(latentBytes.byteLength);
  new Uint8Array(latentBuffer).set(latentBytes);
  const latent = new Float32Array(latentBuffer);
  if (latent.length !== LATENT_ELEMENTS) {
    throw new Error("OPT-0011 B-256 latent extent changed");
  }

  postProgress("authenticating revision-4 A and revision-5 B manifests");
  const [oraclePackage, candidatePackage] = await Promise.all([
    authenticatePackage(
      OPT_0011_FP16_WINDOW_REFERENCE_MANIFEST_PATH,
      ACE_VAE_FP32_ORACLE_MANIFEST_SHA256,
      "reference",
      ACE_VAE_FP32_ORACLE_MANIFEST_BYTES,
      ORACLE_VAE_TENSORS,
      ORACLE_VAE_FILES,
      ORACLE_VAE_BYTES,
    ),
    authenticatePackage(
      OPT_0011_FP16_WINDOW_CANDIDATE_MANIFEST_PATH,
      ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
      "fp16-vae-experimental",
      ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
      ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
      ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length,
      ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    ),
  ]);

  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  postProgress("requesting one shader-f16 device for sequential A then B");
  const context = await requestAceWebGpuDevice({
    modelProfile: "raw-fp16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  const audit = new DeviceAudit(context.device);
  let oracleOutput: Float32Array<ArrayBuffer> | undefined;
  let oracleReceipt: Readonly<Record<string, unknown>> | undefined;
  let candidateReceipt: Readonly<Record<string, unknown>> | undefined;
  let comparison: Readonly<Record<string, unknown>> | undefined;
  let cancellation: Readonly<Record<string, unknown>> | undefined;
  let postCleanupError: Error | undefined;
  let intentionalLoss: Awaited<typeof context.lost> | undefined;
  const rememberPostCleanupError = (error: Error): void => {
    postCleanupError ??= error;
  };
  try {
    const oracle = await runOracleArm(
      audit,
      oraclePackage,
      latent,
      context.device.queue,
    );
    oracleOutput = oracle.output;
    oracleReceipt = oracle.receipt;
    if (audit.liveCount("A") !== 0) {
      throw new Error("OPT-0011 retained A heavyweight GPU resources before B");
    }

    const candidate = await runCandidateArm(
      audit,
      candidatePackage,
      latent,
      oracleOutput,
      context.device.queue,
    );
    candidateReceipt = candidate.receipt;
    comparison = candidate.comparison;
    cancellation = candidate.cancellation;
    if (audit.liveCount("B") !== 0) {
      throw new Error("OPT-0011 B cleanup left GPU buffers live");
    }
  } finally {
    audit.destroyAll();
    audit.destroyAll();
    const beforeDeviceDestroy = audit.resourceSummary();
    if (beforeDeviceDestroy.liveBufferCount !== 0) {
      rememberPostCleanupError(
        new Error("OPT-0011 cleanup retained tracked buffers"),
      );
    }
    // AceWebGpuDeviceContext removes its uncaptured-error listener in
    // destroy(). Keep it installed for two task turns so already queued Chrome
    // validation errors cannot escape the authenticated runtime-event stream.
    await yieldToBrowser();
    await yieldToBrowser();
    if (runtimeEvents.length !== 0) {
      rememberPostCleanupError(
        new Error("OPT-0011 observed a queued runtime event before destroy"),
      );
    }
    try {
      context.destroy();
      const loss = await context.lost;
      intentionalLoss = loss;
      if (loss.reason !== "destroyed") {
        rememberPostCleanupError(
          new Error("OPT-0011 intentional device loss changed"),
        );
      }
    } catch (error) {
      rememberPostCleanupError(
        error instanceof Error
          ? error
          : new Error(`OPT-0011 device cleanup failed: ${String(error)}`),
      );
    }
    // Keep the heartbeat and runtime-event observation alive beyond the loss
    // notification before allowing a passing receipt to be published.
    await yieldToBrowser();
    await yieldToBrowser();
    if (runtimeEvents.length !== 0) {
      rememberPostCleanupError(
        new Error("OPT-0011 observed a runtime event through post-loss cleanup"),
      );
    }
  }
  const workerHeartbeat = heartbeat.stop();
  if (workerHeartbeat["observed"] !== true) {
    throw new Error("OPT-0011 worker heartbeat was not live");
  }
  if (postCleanupError !== undefined) throw postCleanupError;
  if (
    oracleOutput === undefined || oracleReceipt === undefined ||
    candidateReceipt === undefined || comparison === undefined ||
    cancellation === undefined || intentionalLoss === undefined ||
    runtimeEvents.length !== 0
  ) throw new Error("OPT-0011 B-256 result was incomplete");
  const resources = audit.resourceSummary();
  if (
    resources.liveBufferCount !== 0 ||
    resources.createdBufferCount !== resources.destroyedBufferCount ||
    resources.totalDestroyCallCount !== resources.createdBufferCount
  ) throw new Error("OPT-0011 tracked resource accounting diverged");

  return Object.freeze({
    schema: "ace-opt-0011-fp16-vae-b256-window-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification:
      "complete-B256-window-correctness-only-no-performance-or-thermal-timing",
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
      sequentialHeavyweightArms: ["A", "destroy-A", "B"],
      simultaneousHeavyweightPackageCount: 1,
      latentFrames: FRAMES,
      decoderQuantaPerCommandBuffer:
        ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      realQueueEmptyIdleBetweenCommandBuffers: true,
      performanceClaim: null,
      thermalClaim: null,
      qualityClaim: null,
      listeningClaim: null,
      selectorClaim: null,
      productionIntegrationClaim: null,
      fp16512Claim: null,
    }),
    sourceAuthority,
    latent: Object.freeze({
      generator: "xorshift32-13-17-5-high24-symmetric-f32-v1",
      seed: "0x00110512",
      frames: FRAMES,
      channels: CHANNELS,
      elementCount: LATENT_ELEMENTS,
      byteLength: LATENT_BYTES,
      sha256: OPT_0011_FP16_WINDOW_FIXTURE_SHA256,
    }),
    packageIdentity: Object.freeze({
      oracle: packageReceipt(oraclePackage),
      candidate: packageReceipt(candidatePackage),
    }),
    oracle: oracleReceipt,
    candidate: candidateReceipt,
    comparison,
    cancellation,
    cleanup: Object.freeze({
      ...resources,
      allOracleResourcesDestroyedBeforeCandidateAcquisition: true,
      destroyAllCalledTwice: true,
      idempotent: true,
      deviceDestroyed: true,
      intentionalDeviceLoss: Object.freeze({
        type: intentionalLoss.type,
        reason: intentionalLoss.reason,
        message: intentionalLoss.message,
        suppressedFromRuntimeEventsByContext: true,
      }),
      preDestroyRuntimeObservationMacrotaskTurnCount: 2,
      preDestroyRuntimeEventCount: 0,
      postLossMacrotaskTurnCount: 2,
      contextRuntimeEventCountAfterPostLossTurns: runtimeEvents.length,
      postLossRuntimeEventScope:
        "context-event-stream-after-pre-destroy-queue-observation-not-independent-raw-device-listener",
      workerHeartbeatObservationStoppedAfterPostLossTurns: true,
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
    runtimeCommit: OPT_0011_FP16_WINDOW_RUNTIME_COMMIT,
    sources: Object.freeze(sources),
    everySourceAuthenticatedBeforeGpuExecution: true,
  });
}

async function authenticatePackage(
  path: string,
  sha256: string,
  profile: "reference" | "fp16-vae-experimental",
  manifestBytes: number,
  tensorCount: number,
  fileCount: number,
  residentBytes: number,
): Promise<PreparedPackage> {
  const loaded = await loadAcePackageManifest({
    manifestUrl: new URL(path, self.location.href).href,
    expectedManifestSha256: sha256,
    expectedProfile: profile,
  });
  if (
    loaded.manifestSha256 !== sha256 ||
    loaded.manifestByteLength !== manifestBytes
  ) throw new Error(`OPT-0011 ${profile} manifest identity changed`);
  const tensors = Object.values(loaded.manifest.tensors)
    .filter((tensor) => tensor.phase === "vae");
  const names = new Set(tensors.map((tensor) => tensor.shard));
  const files = loaded.manifest.files.filter((file) => names.has(file.name));
  const bytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  if (
    tensors.length !== tensorCount || names.size !== fileCount ||
    files.length !== fileCount || bytes !== residentBytes ||
    files.some((file) => file.kind !== "weights")
  ) throw new Error(`OPT-0011 ${profile} VAE inventory changed`);
  if (
    profile === "reference" &&
    loaded.manifest.provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION
  ) throw new Error("OPT-0011 A converter revision changed");
  if (
    profile === "fp16-vae-experimental" &&
    loaded.manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION
  ) throw new Error("OPT-0011 B converter revision changed");
  if (
    profile === "fp16-vae-experimental" &&
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name) => !names.has(name))
  ) throw new Error("OPT-0011 B exact seven-file inventory changed");
  return Object.freeze({
    loaded,
    files: Object.freeze(files),
    tensorCount: tensors.length,
    residentBytes: bytes,
  });
}

async function acquirePackageFiles(
  prepared: PreparedPackage,
  label: string,
): Promise<ReadonlyMap<string, File>> {
  const cache = await AceOpfsModelCache.open();
  const manifest = Object.freeze({
    ...prepared.loaded.manifest,
    files: prepared.files,
  });
  const acquired = await acquireAceModelFiles({
    manifest,
    manifestUrl: prepared.loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      postProgress(
        `${label} file ${progress.fileIndex + 1}/${progress.fileCount} ` +
          `${progress.completedBytes}/${progress.totalBytes} bytes`,
      );
    },
  });
  if (
    acquired.files.size !== prepared.files.length ||
    acquired.plan.runtimeBytes !== prepared.residentBytes
  ) throw new Error(`OPT-0011 ${label} acquisition accounting changed`);
  return acquired.files;
}

async function runOracleArm(
  audit: DeviceAudit,
  prepared: PreparedPackage,
  latent: Float32Array<ArrayBuffer>,
  rawQueue: GPUQueue,
): Promise<{
  readonly output: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<Record<string, unknown>>;
}> {
  audit.setScope("A");
  postProgress("acquiring and uploading revision-4 FP32 oracle VAE only");
  const files = await acquirePackageFiles(prepared, "A oracle VAE");
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceVaeChunkGpuBackend | undefined;
  try {
    phase = await AceGpuTensorPhase.load(
      audit.device,
      prepared.loaded.manifest,
      files,
      ["vae"],
    );
    if (
      phase.residentBytes !== ORACLE_VAE_BYTES ||
      phase.phases.length !== 1 || phase.phases[0] !== "vae"
    ) throw new Error("OPT-0011 A resident phase changed");
    const plan = planAceVaeChunkedDecode(FRAMES);
    requireSingleWindowPlan(plan);
    backend = await AceVaeChunkGpuBackend.create({
      device: audit.device,
      plan,
      finalLatents: latent,
      ownedVaeWeights: phase,
    });
    phase = undefined;
    const outputBuffer = audit.requireBuffer("A", "ace-vae-chunk-output");
    postProgress("executing FP32 oracle first complete B-256 window");
    await prefillFp32Output(rawQueue, outputBuffer, 0);
    audit.beginExecution("A-first");
    const firstRaw = await backend.decodeWindow(plan.windows[0]!);
    const firstTrace = audit.endExecution();
    const first = detachFloat32(firstRaw);
    const firstScan = await scanOutput(first);
    validateCompleteOutput(firstScan, "A first");

    postProgress("executing FP32 oracle deterministic rerun");
    await prefillFp32Output(rawQueue, outputBuffer, 0);
    audit.beginExecution("A-rerun");
    const rerunRaw = await backend.decodeWindow(plan.windows[0]!);
    const rerunTrace = audit.endExecution();
    const rerun = detachFloat32(rerunRaw);
    const rerunScan = await scanOutput(rerun);
    validateCompleteOutput(rerunScan, "A rerun");
    const repeat = compareU32(first, rerun);
    if (repeat.mismatchCount !== 0) {
      throw new Error("OPT-0011 newly derived A oracle was not deterministic");
    }
    validateExecutionTrace(firstTrace, "oracle", "A first");
    validateExecutionTrace(rerunTrace, "oracle", "A rerun");
    const shaderIdentity = await audit.shaderIdentity("A");
    await backend.destroy();
    backend = undefined;
    if (audit.liveCount("A") !== 0) {
      throw new Error("OPT-0011 A destruction left heavyweight buffers live");
    }
    return Object.freeze({
      output: first,
      receipt: Object.freeze({
        profile: "accepted-packed-bf16-fp32-oracle",
        manifestSha256: ACE_VAE_FP32_ORACLE_MANIFEST_SHA256,
        first: firstScan,
        rerun: rerunScan,
        deterministicU32Comparison: repeat,
        firstExecution: compactTrace(firstTrace, "oracle"),
        rerunExecution: compactTrace(rerunTrace, "oracle"),
        shaderIdentity,
        destroyedBeforeCandidateAcquisition: true,
      }),
    });
  } finally {
    await backend?.destroy();
    phase?.destroy();
  }
}

async function runCandidateArm(
  audit: DeviceAudit,
  prepared: PreparedPackage,
  latent: Float32Array<ArrayBuffer>,
  oracle: Float32Array<ArrayBuffer>,
  rawQueue: GPUQueue,
): Promise<{
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly comparison: Readonly<Record<string, unknown>>;
  readonly cancellation: Readonly<Record<string, unknown>>;
}> {
  audit.setScope("B");
  postProgress("acquiring and uploading revision-5 FP16 VAE only");
  const files = await acquirePackageFiles(prepared, "B candidate VAE");
  let phase: AceGpuTensorPhase | undefined;
  let runtime: AceOpt0011Fp16VaeDecoderRuntime | undefined;
  let scheduler: AceCooperativeGpuScheduler | undefined;
  let resources: BResources | undefined;
  try {
    phase = await AceGpuTensorPhase.load(
      audit.device,
      prepared.loaded.manifest,
      files,
      ["vae"],
    );
    if (
      phase.residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
      phase.phases.length !== 1 || phase.phases[0] !== "vae"
    ) throw new Error("OPT-0011 B resident phase changed");
    const graph = planAceVaeDecoder(FRAMES);
    const packageBindings = resolveAceOpt0011Fp16VaePackageBindings(
      graph,
      prepared.loaded,
      phase,
    );
    resources = createBResources(audit.device, graph);
    runtime = AceOpt0011Fp16VaeDecoderRuntime.create(audit.device);
    const dispatch = await runtime.createDecoderDispatch(
      "opt-0011-complete-b256",
      {
        stagingInput: resources.stagingInput.binding,
        decoderInput: resources.decoderInput.binding,
        workspaces: resources.workspaces.map((item) => item.binding) as
          [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding],
        output: resources.output.binding,
        package: packageBindings,
      },
    );
    validateCandidateDispatch(dispatch);
    scheduler = new AceCooperativeGpuScheduler();
    const shaderIdentity = await audit.shaderIdentity("B");
    if (
      shaderIdentity["shaderModuleCreateCount"] !== 39 ||
      shaderIdentity["uniqueShaderCount"] !== 34
    ) {
      throw new Error("OPT-0011 B portable shader/module count changed");
    }
    const graphIdentity = await candidateGraphIdentity(
      dispatch,
      packageBindings,
    );

    postProgress("executing complete FP16 B-256 first window");
    await initializeFirstCandidateExecution(rawQueue, resources, latent);
    const first = await executeCandidateWindow(
      audit,
      scheduler,
      dispatch,
      resources,
      "B-first",
    );
    postProgress("executing complete FP16 B-256 deterministic rerun");
    await initializeCandidateRerun(rawQueue, resources, latent);
    const rerun = await executeCandidateWindow(
      audit,
      scheduler,
      dispatch,
      resources,
      "B-rerun",
    );
    const repeat = compareU32(first.output, rerun.output);
    if (repeat.mismatchCount !== 0) {
      throw new Error("OPT-0011 B-256 deterministic rerun diverged");
    }

    postProgress("scanning complete A/B raw waveforms against frozen bounds");
    const comparison = compareWaveforms(oracle, first.output);
    requireFrozenBounds(comparison);

    postProgress("proving cancellation after one drained B-256 batch");
    await initializeCandidateRerun(rawQueue, resources, latent);
    const cancellation = await runCandidateCancellation(
      audit,
      scheduler,
      dispatch,
    );

    await scheduler.dispose();
    scheduler = undefined;
    runtime.destroy();
    runtime = undefined;
    phase.destroy();
    phase = undefined;
    resources.destroy();
    resources = undefined;
    if (audit.liveCount("B") !== 0) {
      throw new Error("OPT-0011 B owner cleanup left buffers live");
    }
    return Object.freeze({
      receipt: Object.freeze({
        profile: "opt-0011-mixed-fp16-portable-v1",
        manifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
        precisionMapSha256: ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
        first: first.scan,
        rerun: rerun.scan,
        deterministicU32Comparison: repeat,
        firstExecution: first.trace,
        rerunExecution: rerun.trace,
        shaderIdentity,
        graphIdentity,
        guards: Object.freeze({
          guardedBindingCount: 6,
          guardBytesPerSide: GUARD_BYTES,
          everyHeadAndTailExactOnBothExecutions: true,
        }),
      }),
      comparison,
      cancellation,
    });
  } finally {
    await scheduler?.dispose();
    runtime?.destroy();
    phase?.destroy();
    resources?.destroy();
  }
}

function createBResources(
  device: GPUDevice,
  graph: AceVaeDecoderGraphPlan,
): BResources {
  const stagingInput = createGuardedBinding(
    device,
    "opt-0011-b-staging-input",
    graph.inputElements * 4,
  );
  const decoderInput = createGuardedBinding(
    device,
    "opt-0011-b-decoder-input",
    graph.inputElements * 2,
  );
  const workspaceBytes = graph.maximumActivationElements * 2;
  const workspaces = [0, 1, 2].map((index) => createGuardedBinding(
    device,
    `opt-0011-b-workspace-${index}`,
    workspaceBytes,
  )) as [GuardedBinding, GuardedBinding, GuardedBinding];
  const output = createGuardedBinding(
    device,
    "opt-0011-b-output",
    graph.outputElements * 4,
  );
  const guarded = Object.freeze([
    stagingInput,
    decoderInput,
    ...workspaces,
    output,
  ]);
  const outputReadback = device.createBuffer({
    label: "opt-0011-b-output-readback",
    size: OUTPUT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const guardReadback = device.createBuffer({
    label: "opt-0011-b-guard-readback",
    size: guarded.length * 2 * GUARD_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let destroyed = false;
  return Object.freeze({
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
  if (payloadBytes % 4 !== 0) throw new Error(`${label} payload is unaligned`);
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

async function initializeFirstCandidateExecution(
  queue: GPUQueue,
  resources: BResources,
  latent: Float32Array<ArrayBuffer>,
): Promise<void> {
  const encoder = queueDevice(queue).createCommandEncoder({
    label: "opt-0011-b-first-internal-clear",
  });
  encoder.clearBuffer(
    resources.decoderInput.buffer,
    GUARD_BYTES,
    resources.decoderInput.payloadBytes,
  );
  for (const workspace of resources.workspaces) {
    encoder.clearBuffer(workspace.buffer, GUARD_BYTES, workspace.payloadBytes);
  }
  queue.submit([encoder.finish()]);
  await queue.onSubmittedWorkDone();
  await initializeCandidateRerun(queue, resources, latent);
}

// WebGPU exposes no queue->device link. The first clear uses this temporarily
// installed map, populated by DeviceAudit's queue proxy.
const QUEUE_DEVICES = new WeakMap<GPUQueue, GPUDevice>();
function queueDevice(queue: GPUQueue): GPUDevice {
  const device = QUEUE_DEVICES.get(queue);
  if (device === undefined) throw new Error("OPT-0011 queue device is absent");
  return device;
}

async function initializeCandidateRerun(
  queue: GPUQueue,
  resources: BResources,
  latent: Float32Array<ArrayBuffer>,
): Promise<void> {
  const guard = new Uint32Array(GUARD_BYTES / 4);
  guard.fill(GUARD_WORD);
  for (const item of resources.guarded) {
    queue.writeBuffer(item.buffer, 0, guard);
    queue.writeBuffer(item.buffer, GUARD_BYTES + item.payloadBytes, guard);
  }
  queue.writeBuffer(resources.stagingInput.buffer, GUARD_BYTES, latent);
  const sentinel = new Uint32Array(OUTPUT_ELEMENTS);
  sentinel.fill(OUTPUT_QNAN_WORD);
  queue.writeBuffer(resources.output.buffer, GUARD_BYTES, sentinel);
  await queue.onSubmittedWorkDone();
}

async function executeCandidateWindow(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  dispatch: AceOpt0011Fp16VaeDecoderDispatch,
  resources: BResources,
  id: string,
): Promise<FullExecution> {
  const controller = new AbortController();
  audit.beginExecution(id);
  const scheduling = await scheduler.runLazy({
    queue: audit.device.queue,
    commandBufferCount: TOTAL_COMMAND_BUFFERS,
    createCommandBuffer: (index) => index < B_COMPUTE_COMMAND_BUFFERS
      ? encodeCandidateBatch(audit.device, dispatch, index, id)
      : encodeCandidateReadback(audit.device, resources, id),
    signal: controller.signal,
  });
  const mutableTrace = audit.endExecution();
  if (
    scheduling.commandBuffersSubmitted !== TOTAL_COMMAND_BUFFERS ||
    scheduling.queueDrains !== TOTAL_COMMAND_BUFFERS ||
    scheduling.cooperativeIdleMs !== COMPLETED_IDLE_TURNS
  ) throw new Error(`${id} scheduler accounting changed`);
  validateExecutionTrace(mutableTrace, "candidate", id);
  const output = await mapOutput(resources.outputReadback);
  await validateGuards(resources.guardReadback, resources.guarded.length);
  const scan = await scanOutput(output);
  validateCompleteOutput(scan, id);
  return Object.freeze({
    output,
    scan,
    trace: Object.freeze({
      ...compactTrace(mutableTrace, "candidate"),
      scheduling,
      batchSize: ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
      computeCommandBufferCount: B_COMPUTE_COMMAND_BUFFERS,
      readbackCommandBufferCount: 1,
      completedRealIdleTurns: COMPLETED_IDLE_TURNS,
    }),
  });
}

function encodeCandidateBatch(
  device: GPUDevice,
  dispatch: AceOpt0011Fp16VaeDecoderDispatch,
  batchIndex: number,
  id: string,
): GPUCommandBuffer {
  const first = batchIndex * ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER;
  const end = Math.min(
    first + ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
    dispatch.quanta.length,
  );
  const encoder = device.createCommandEncoder({
    label: `${id}-batch-${batchIndex}`,
  });
  const pass = encoder.beginComputePass({
    label: `${id}-batch-${batchIndex}-pass`,
  });
  for (let index = first; index < end; index += 1) {
    dispatch.quanta[index]!.encode(pass);
  }
  pass.end();
  return encoder.finish();
}

function encodeCandidateReadback(
  device: GPUDevice,
  resources: BResources,
  id: string,
): GPUCommandBuffer {
  const encoder = device.createCommandEncoder({ label: `${id}-readback` });
  encoder.copyBufferToBuffer(
    resources.output.buffer,
    GUARD_BYTES,
    resources.outputReadback,
    0,
    OUTPUT_BYTES,
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

async function runCandidateCancellation(
  audit: DeviceAudit,
  scheduler: AceCooperativeGpuScheduler,
  dispatch: AceOpt0011Fp16VaeDecoderDispatch,
): Promise<Readonly<Record<string, unknown>>> {
  const controller = new AbortController();
  const reason = new DOMException(
    "cancel-after-first-drained-b256-batch-and-real-idle",
    "AbortError",
  );
  let rejection: unknown;
  audit.beginExecution("B-cancellation");
  try {
    await scheduler.runLazy({
      queue: audit.device.queue,
      commandBufferCount: TOTAL_COMMAND_BUFFERS,
      createCommandBuffer: (index) => {
        if (index >= B_COMPUTE_COMMAND_BUFFERS) {
          throw new Error("OPT-0011 cancellation encoded forbidden readback");
        }
        return encodeCandidateBatch(
          audit.device,
          dispatch,
          index,
          "B-cancellation",
        );
      },
      signal: controller.signal,
      onProgress: (progress) => {
        if (progress.completedCommandBuffers === 1) controller.abort(reason);
      },
    });
  } catch (error) {
    rejection = error;
  }
  const trace = audit.endExecution();
  if (
    rejection !== reason || trace.commands.length !== 1 ||
    trace.submissionCount !== 1 || trace.drainCount !== 1 ||
    trace.writeBufferCount !== 0 ||
    trace.commands[0]!.label !== "B-cancellation-batch-0" ||
    trace.commands[0]!.computePassCount !== 1 ||
    trace.commands[0]!.dispatchCount !== 8 ||
    trace.commands[0]!.copyCount !== 0 ||
    trace.commands[0]!.clearCount !== 0 ||
    !trace.commands[0]!.finished || !trace.commands[0]!.submitted ||
    !trace.commands[0]!.drained
  ) throw new Error("OPT-0011 B cancellation accounting changed");
  return Object.freeze({
    rejectionName: "AbortError",
    rejectionMessage: reason.message,
    cancellationPoint: "after-first-drained-batch-and-real-idle",
    encodedCommandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    completedDecoderQuanta: 8,
    laterEncodingPrevented: true,
    laterSubmissionPrevented: true,
    readbackPrevented: true,
    metricsPublicationPrevented: true,
    outputFinalizationPrevented: true,
    realQueueEmptyIdleDeliveredBeforeRejection: true,
  });
}

function validateCandidateDispatch(
  dispatch: AceOpt0011Fp16VaeDecoderDispatch,
): void {
  const kinds = countKinds(dispatch.quanta);
  if (
    dispatch.operationCount !== B_OPERATION_COUNT ||
    dispatch.graphQuantumCount !== B_GRAPH_QUANTA ||
    dispatch.primitiveCount !== B_SEQUENCE_QUANTA ||
    dispatch.quanta.length !== B_SEQUENCE_QUANTA ||
    dispatch.graphQuanta.length !== B_GRAPH_QUANTA ||
    kinds["ingress-cast"] !== 1 || kinds.conv1d !== 2_459 ||
    kinds["conv-transpose1d"] !== 322 || kinds.snake !== 813 ||
    kinds.add !== 348
  ) throw new Error("OPT-0011 B complete dispatch topology changed");
  if (
    ACE_OPT_0011_VAE_FP16_DECODER_OPERATION_COUNT !== B_OPERATION_COUNT ||
    ACE_OPT_0011_VAE_FP16_DECODER_GRAPH_QUANTUM_COUNT !== B_GRAPH_QUANTA ||
    ACE_OPT_0011_VAE_FP16_DECODER_SEQUENCE_QUANTUM_COUNT !== B_SEQUENCE_QUANTA
  ) throw new Error("OPT-0011 B exported topology constants changed");
}

async function candidateGraphIdentity(
  dispatch: AceOpt0011Fp16VaeDecoderDispatch,
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
  const operations = packageBindings.operations.map((operation) => {
    const tensor = (value: { logicalTensor: string; record: {
      shard: string; byteOffset: number; byteLength: number; dtype: string;
    } }) => ({
      name: value.logicalTensor,
      shard: value.record.shard,
      byteOffset: value.record.byteOffset,
      byteLength: value.record.byteLength,
      dtype: value.record.dtype,
    });
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

function countKinds(
  quanta: readonly { readonly operationKind: string }[],
): Record<string, number> {
  const counts: Record<string, number> = Object.create(null) as
    Record<string, number>;
  for (const quantum of quanta) {
    counts[quantum.operationKind] = (counts[quantum.operationKind] ?? 0) + 1;
  }
  return counts;
}

class DeviceAudit {
  readonly device: GPUDevice;
  private scope: BufferRecord["scope"] = "setup";
  private readonly records = new Map<GPUBuffer, BufferRecord>();
  private readonly shaderSources = new Map<BufferRecord["scope"], string[]>();
  private readonly commandBuffers = new WeakMap<GPUCommandBuffer, MutableCommandRecord>();
  private active: MutableExecutionTrace | undefined;
  private maximumLive = 0;

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
    QUEUE_DEVICES.set(rawDevice.queue, this.device);
    QUEUE_DEVICES.set(queue, this.device);
  }

  setScope(scope: BufferRecord["scope"]): void {
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
    return trace;
  }

  requireBuffer(scope: BufferRecord["scope"], label: string): GPUBuffer {
    const matches = [...this.records].filter(([, record]) =>
      record.scope === scope && record.label === label && !record.destroyed
    );
    if (matches.length !== 1) {
      throw new Error(`OPT-0011 expected one live ${scope}/${label} buffer`);
    }
    return matches[0]![0];
  }

  liveCount(scope: BufferRecord["scope"]): number {
    return [...this.records.values()].filter(
      (record) => record.scope === scope && !record.destroyed,
    ).length;
  }

  async shaderIdentity(
    scope: BufferRecord["scope"],
  ): Promise<Readonly<Record<string, unknown>>> {
    const sources = this.shaderSources.get(scope) ?? [];
    const records = await Promise.all([...new Set(sources)].map(async (source) => {
      const bytes = new TextEncoder().encode(source);
      return Object.freeze({
        byteLength: bytes.byteLength,
        sha256: await sha256Hex(bytes),
      });
    }));
    records.sort((left, right) => left.sha256.localeCompare(right.sha256));
    const aggregate = await sha256Hex(
      new TextEncoder().encode(JSON.stringify(records)),
    );
    return Object.freeze({
      shaderModuleCreateCount: sources.length,
      uniqueShaderCount: records.length,
      uniqueShaders: Object.freeze(records),
      aggregateSha256: aggregate,
      everyExecutedShaderCaptured: true,
    });
  }

  destroyAll(): void {
    for (const [buffer, record] of this.records) {
      if (!record.destroyed) buffer.destroy();
    }
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
      usage: descriptor.usage,
      destroyCalls: 0,
      destroyed: false,
    };
    this.records.set(buffer, record);
    this.maximumLive = Math.max(
      this.maximumLive,
      [...this.records.values()].filter((candidate) => !candidate.destroyed).length,
    );
    const destroy = buffer.destroy.bind(buffer);
    Object.defineProperty(buffer, "destroy", {
      configurable: true,
      value: () => {
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
                const value = Reflect.get(passTarget, passProperty, passTarget) as unknown;
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

function validateExecutionTrace(
  trace: MutableExecutionTrace,
  arm: Opt0011ExecutionTraceArm,
  label: string,
): void {
  validateOpt0011ExecutionTraceTopology(traceTopology(trace), arm, label);
}

function compactTrace(
  trace: MutableExecutionTrace,
  arm: Opt0011ExecutionTraceArm,
): Readonly<Record<string, unknown>> {
  return compactTraceTopology(traceTopology(trace), arm);
}

function traceTopology(
  trace: MutableExecutionTrace,
): Opt0011ExecutionTraceTopology {
  return Object.freeze({
    id: trace.id,
    commandLabels: Object.freeze(trace.commands.map((command) => command.label)),
    computePassCounts: Object.freeze(
      trace.commands.map((command) => command.computePassCount),
    ),
    dispatchCounts: Object.freeze(
      trace.commands.map((command) => command.dispatchCount),
    ),
    copyCounts: Object.freeze(trace.commands.map((command) => command.copyCount)),
    clearCounts: Object.freeze(trace.commands.map((command) => command.clearCount)),
    submissionCount: trace.submissionCount,
    drainCount: trace.drainCount,
    writeBufferCount: trace.writeBufferCount,
    incompleteCommandCount: trace.commands.filter((command) =>
      !command.finished || !command.submitted || !command.drained
    ).length,
  });
}

export function validateOpt0011ExecutionTraceTopology(
  topology: Opt0011ExecutionTraceTopology,
  arm: Opt0011ExecutionTraceArm,
  label: string,
): void {
  const fail = (reason: string): never => {
    throw new Error(
      `${label} ${reason}: ${JSON.stringify(compactTraceTopology(topology, arm))}`,
    );
  };
  const arrays = [
    topology.commandLabels,
    topology.computePassCounts,
    topology.dispatchCounts,
    topology.copyCounts,
    topology.clearCounts,
  ];
  if (arrays.some((values) => values.length !== TOTAL_COMMAND_BUFFERS)) {
    fail("command-record extent changed");
  }
  const computePassCounts = topology.computePassCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const computeDispatchCounts = topology.dispatchCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const expectedPassCount = (index: number): number => arm === "oracle"
    ? (index === B_COMPUTE_COMMAND_BUFFERS - 1 ? 6 : 8)
    : 1;
  const expectedDispatchCount = (index: number): number => {
    if (arm === "candidate") {
      return index === B_COMPUTE_COMMAND_BUFFERS - 1 ? 7 : 8;
    }
    if (index === 0) return 14;
    if (index <= 5) return 16;
    return index === B_COMPUTE_COMMAND_BUFFERS - 1 ? 6 : 8;
  };
  const expectedDispatches = arm === "oracle"
    ? A_PRIMITIVE_DISPATCHES
    : B_SEQUENCE_QUANTA;
  if (
    topology.submissionCount !== TOTAL_COMMAND_BUFFERS ||
    topology.drainCount !== TOTAL_COMMAND_BUFFERS ||
    topology.incompleteCommandCount !== 0 ||
    computeDispatchCounts.reduce((sum, count) => sum + count, 0) !==
      expectedDispatches ||
    topology.writeBufferCount !== (arm === "oracle" ? 1 : 0)
  ) {
    fail("physical execution accounting changed");
  }
  if (
    computePassCounts.some((count, index) => count !== expectedPassCount(index))
  ) {
    fail("compute-pass topology changed");
  }
  if (computeDispatchCounts.some(
    (count, index) => count !== expectedDispatchCount(index),
  )) {
    fail("compute-dispatch batch topology changed");
  }
  const expectedComputeLabel = (index: number): string => arm === "oracle"
    ? `ace-vae-window-0-batch-${index}`
    : `${topology.id}-batch-${index}`;
  if (
    topology.commandLabels.slice(0, B_COMPUTE_COMMAND_BUFFERS).some(
      (commandLabel, index) => commandLabel !== expectedComputeLabel(index),
    )
  ) {
    fail("compute command labels changed");
  }
  const readbackIndex = TOTAL_COMMAND_BUFFERS - 1;
  const expectedReadbackLabel = arm === "oracle"
    ? "ace-vae-window-0-readback"
    : `${topology.id}-readback`;
  const expectedReadbackCopies = arm === "oracle"
    ? A_READBACK_COPIES
    : B_READBACK_COPIES;
  if (
    topology.copyCounts.slice(0, B_COMPUTE_COMMAND_BUFFERS).some(
      (count) => count !== 0,
    ) ||
    topology.clearCounts.some((count) => count !== 0) ||
    topology.commandLabels[readbackIndex] !== expectedReadbackLabel ||
    topology.computePassCounts[readbackIndex] !== 0 ||
    topology.dispatchCounts[readbackIndex] !== 0 ||
    topology.copyCounts[readbackIndex] !== expectedReadbackCopies
  ) {
    fail("compute/readback command topology changed");
  }
}

function compactTraceTopology(
  topology: Opt0011ExecutionTraceTopology,
  arm: Opt0011ExecutionTraceArm,
): Readonly<Record<string, unknown>> {
  const computePassCounts = topology.computePassCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const computeDispatchCounts = topology.dispatchCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const computeCopyCounts = topology.copyCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const computeClearCounts = topology.clearCounts.slice(
    0,
    B_COMPUTE_COMMAND_BUFFERS,
  );
  const readbackIndex = TOTAL_COMMAND_BUFFERS - 1;
  return Object.freeze({
    arm,
    executionId: topology.id,
    commandBufferCount: topology.commandLabels.length,
    computeCommandBufferCount: computePassCounts.length,
    submissionCount: topology.submissionCount,
    queueDrainCount: topology.drainCount,
    writeBufferCount: topology.writeBufferCount,
    incompleteCommandCount: topology.incompleteCommandCount,
    computePassCount: computePassCounts.reduce((sum, count) => sum + count, 0),
    dispatchCount: computeDispatchCounts.reduce((sum, count) => sum + count, 0),
    copyCount: topology.copyCounts.reduce((sum, count) => sum + count, 0),
    clearCount: topology.clearCounts.reduce((sum, count) => sum + count, 0),
    computeBatchPassHistogram: histogram(computePassCounts),
    computeBatchDispatchHistogram: histogram(computeDispatchCounts),
    computeBatchCopyHistogram: histogram(computeCopyCounts),
    computeBatchClearHistogram: histogram(computeClearCounts),
    firstCommandLabel: topology.commandLabels[0] ?? null,
    finalComputeCommandLabel:
      topology.commandLabels[B_COMPUTE_COMMAND_BUFFERS - 1] ?? null,
    readback: Object.freeze({
      label: topology.commandLabels[readbackIndex] ?? null,
      computePassCount: topology.computePassCounts[readbackIndex] ?? null,
      dispatchCount: topology.dispatchCounts[readbackIndex] ?? null,
      copyCount: topology.copyCounts[readbackIndex] ?? null,
      clearCount: topology.clearCounts[readbackIndex] ?? null,
    }),
    everyCommandFinishedSubmittedAndDrained:
      topology.incompleteCommandCount === 0,
  });
}

function histogram(values: readonly number[]): Readonly<Record<string, number>> {
  const result: Record<string, number> = {};
  for (const value of values) result[String(value)] = (result[String(value)] ?? 0) + 1;
  return Object.freeze(result);
}

async function prefillFp32Output(
  queue: GPUQueue,
  buffer: GPUBuffer,
  offset: number,
): Promise<void> {
  const words = new Uint32Array(OUTPUT_ELEMENTS);
  words.fill(OUTPUT_QNAN_WORD);
  queue.writeBuffer(buffer, offset, words);
  await queue.onSubmittedWorkDone();
}

async function mapOutput(buffer: GPUBuffer): Promise<Float32Array<ArrayBuffer>> {
  await buffer.mapAsync(GPUMapMode.READ, 0, OUTPUT_BYTES);
  try {
    const copy = buffer.getMappedRange(0, OUTPUT_BYTES).slice(0);
    return new Float32Array(copy);
  } finally {
    buffer.unmap();
  }
}

async function validateGuards(buffer: GPUBuffer, bindingCount: number): Promise<void> {
  const bytes = bindingCount * 2 * GUARD_BYTES;
  await buffer.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    const words = new Uint32Array(buffer.getMappedRange(0, bytes));
    for (let index = 0; index < words.length; index += 1) {
      if (words[index] !== GUARD_WORD) {
        throw new Error(`OPT-0011 B guard changed at word ${index}`);
      }
    }
  } finally {
    buffer.unmap();
  }
}

function detachFloat32(values: Float32Array): Float32Array<ArrayBuffer> {
  const copy = new Float32Array(values.length);
  copy.set(values);
  return copy;
}

async function scanOutput(output: Float32Array): Promise<OutputScan> {
  if (output.length !== OUTPUT_ELEMENTS || output.byteLength !== OUTPUT_BYTES) {
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

function validateCompleteOutput(scan: OutputScan, label: string): void {
  if (
    scan.elementCount !== OUTPUT_ELEMENTS || scan.byteLength !== OUTPUT_BYTES ||
    scan.finiteCount !== OUTPUT_ELEMENTS || scan.nonzeroCount === 0 ||
    scan.qNaNSentinelCount !== 0 || scan.stereoDifferenceFrameCount === 0 ||
    scan.clampBoundaryCount !== 0 || !Number.isFinite(scan.minimum) ||
    !Number.isFinite(scan.maximum) || scan.peak === 0
  ) throw new Error(`${label} output completeness/finiteness changed`);
}

export function compareOpt0011Fp16WindowU32(
  left: Float32Array,
  right: Float32Array,
): Readonly<{
  readonly comparedWordCount: number;
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
  readonly bitExact: boolean;
}> {
  return compareU32(left, right);
}

function compareU32(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length) throw new Error("U32 extents differ");
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] === b[index]) continue;
    mismatchCount += 1;
    firstMismatchIndex ??= index;
  }
  return Object.freeze({
    comparedWordCount: a.length,
    mismatchCount,
    firstMismatchIndex,
    bitExact: mismatchCount === 0,
  });
}

export function compareOpt0011Fp16WindowWaveforms(
  oracle: Float32Array,
  candidate: Float32Array,
): Readonly<Record<string, unknown>> {
  return compareWaveforms(oracle, candidate);
}

function compareWaveforms(
  oracle: Float32Array,
  candidate: Float32Array,
): Readonly<Record<string, unknown>> {
  if (
    oracle.length !== OUTPUT_ELEMENTS || candidate.length !== OUTPUT_ELEMENTS
  ) throw new Error("OPT-0011 waveform comparison extent changed");
  const joint = metricAccumulator();
  const channels = [metricAccumulator(), metricAccumulator()];
  let firstDifferenceIndex: number | null = null;
  let worstIndex = 0;
  let maximumAbsoluteError = -1;
  let maximumRelativeError = 0;
  for (let index = 0; index < oracle.length; index += 1) {
    const a = oracle[index]!;
    const b = candidate[index]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new Error("OPT-0011 waveform comparison encountered non-finite data");
    }
    const error = b - a;
    const absolute = Math.abs(error);
    if (absolute !== 0 && firstDifferenceIndex === null) {
      firstDifferenceIndex = index;
    }
    if (absolute > maximumAbsoluteError) {
      maximumAbsoluteError = absolute;
      worstIndex = index;
    }
    maximumRelativeError = Math.max(
      maximumRelativeError,
      absolute / Math.max(Math.abs(a), 1e-6),
    );
    accumulate(joint, a, b);
    accumulate(channels[index & 1]!, a, b);
  }
  const jointMetrics = finishMetrics(joint);
  const channelMetrics = channels.map(finishMetrics);
  const normalizedMaximumAbsoluteError = maximumAbsoluteError /
    Math.max(jointMetrics.oraclePeak, 1e-6);
  return Object.freeze({
    comparedElementCount: oracle.length,
    comparedFrameCount: OUTPUT_FRAMES,
    maximumAbsoluteError,
    maximumRelativeError,
    normalizedMaximumAbsoluteError,
    firstDifferenceIndex,
    worstIndex,
    firstDifferenceNeighborhood: neighborhood(
      oracle,
      candidate,
      firstDifferenceIndex ?? 0,
    ),
    worstErrorNeighborhood: neighborhood(oracle, candidate, worstIndex),
    joint: jointMetrics,
    channels: Object.freeze(channelMetrics),
    bounds: OPT_0011_FP16_WINDOW_BOUNDS,
  });
}

interface MetricAccumulator {
  count: number;
  sumA: number;
  sumB: number;
  sumA2: number;
  sumB2: number;
  sumAB: number;
  sumError2: number;
  peakA: number;
  peakB: number;
}

function metricAccumulator(): MetricAccumulator {
  return {
    count: 0,
    sumA: 0,
    sumB: 0,
    sumA2: 0,
    sumB2: 0,
    sumAB: 0,
    sumError2: 0,
    peakA: 0,
    peakB: 0,
  };
}

function accumulate(metrics: MetricAccumulator, a: number, b: number): void {
  metrics.count += 1;
  metrics.sumA += a;
  metrics.sumB += b;
  metrics.sumA2 += a * a;
  metrics.sumB2 += b * b;
  metrics.sumAB += a * b;
  const error = b - a;
  metrics.sumError2 += error * error;
  metrics.peakA = Math.max(metrics.peakA, Math.abs(a));
  metrics.peakB = Math.max(metrics.peakB, Math.abs(b));
}

function finishMetrics(metrics: MetricAccumulator) {
  const meanA = metrics.sumA / metrics.count;
  const meanB = metrics.sumB / metrics.count;
  const energyA = metrics.sumA2 / metrics.count;
  const energyB = metrics.sumB2 / metrics.count;
  const rmsA = Math.sqrt(energyA);
  const rmsB = Math.sqrt(energyB);
  const rmsError = Math.sqrt(metrics.sumError2 / metrics.count);
  const covariance = metrics.sumAB -
    metrics.sumA * metrics.sumB / metrics.count;
  const varianceA = metrics.sumA2 -
    metrics.sumA * metrics.sumA / metrics.count;
  const varianceB = metrics.sumB2 -
    metrics.sumB * metrics.sumB / metrics.count;
  const correlation = covariance / Math.sqrt(varianceA * varianceB);
  return Object.freeze({
    elementCount: metrics.count,
    oracleRms: rmsA,
    candidateRms: rmsB,
    rmsError,
    normalizedRmsError: rmsError / Math.max(rmsA, 1e-6),
    snrDecibels: rmsError === 0
      ? Number.POSITIVE_INFINITY
      : 20 * Math.log10(rmsA / rmsError),
    correlation,
    oraclePeak: metrics.peakA,
    candidatePeak: metrics.peakB,
    relativeRmsDrift: Math.abs(rmsB - rmsA) / Math.max(rmsA, 1e-6),
    relativeEnergyDrift:
      Math.abs(energyB - energyA) / Math.max(energyA, 1e-12),
    relativePeakDrift:
      Math.abs(metrics.peakB - metrics.peakA) /
      Math.max(metrics.peakA, 1e-6),
    oracleDcOffset: meanA,
    candidateDcOffset: meanB,
    absoluteDcOffsetDrift: Math.abs(meanB - meanA),
    dcOffsetDriftLimit:
      OPT_0011_FP16_WINDOW_BOUNDS.maximumDcDriftScale *
      Math.max(rmsA, 1e-6),
  });
}

function requireFrozenBounds(comparison: Readonly<Record<string, unknown>>): void {
  const joint = comparison["joint"] as ReturnType<typeof finishMetrics>;
  const channels = comparison["channels"] as readonly ReturnType<typeof finishMetrics>[];
  const normalizedMaximumAbsoluteError =
    comparison["normalizedMaximumAbsoluteError"] as number;
  const bounds = OPT_0011_FP16_WINDOW_BOUNDS;
  if (
    !(joint.normalizedRmsError <= bounds.maximumNormalizedRmsError) ||
    !(joint.snrDecibels >= bounds.minimumSnrDecibels) ||
    !(joint.correlation >= bounds.minimumCorrelation) ||
    channels.some((channel) =>
      !(channel.correlation >= bounds.minimumCorrelation)
    ) ||
    !(joint.relativeRmsDrift <= bounds.maximumRelativeRmsDrift) ||
    !(joint.relativeEnergyDrift <= bounds.maximumRelativeEnergyDrift) ||
    !(joint.relativePeakDrift <= bounds.maximumRelativePeakDrift) ||
    !(joint.absoluteDcOffsetDrift <= joint.dcOffsetDriftLimit) ||
    channels.some((channel) =>
      !(channel.absoluteDcOffsetDrift <= channel.dcOffsetDriftLimit)
    ) ||
    !(normalizedMaximumAbsoluteError <= bounds.maximumNormalizedAbsoluteError)
  ) throw new Error("OPT-0011 complete A/B waveform bounds failed");
}

function neighborhood(
  oracle: Float32Array,
  candidate: Float32Array,
  center: number,
): readonly unknown[] {
  const start = Math.max(0, center - 8);
  const end = Math.min(oracle.length, center + 9);
  return Object.freeze(Array.from({ length: end - start }, (_, offset) => {
    const index = start + offset;
    return Object.freeze({
      index,
      frame: Math.floor(index / 2),
      channel: index & 1,
      oracle: oracle[index],
      candidate: candidate[index],
      error: candidate[index]! - oracle[index]!,
    });
  }));
}

function requireSingleWindowPlan(plan: ReturnType<typeof planAceVaeChunkedDecode>): void {
  const window = plan.windows[0];
  if (
    plan.latentFrames !== FRAMES || plan.windows.length !== 1 ||
    window === undefined || window.latentWindowFrames !== FRAMES ||
    window.decodedAudioFrames !== OUTPUT_FRAMES ||
    plan.outputInterleavedElements !== OUTPUT_ELEMENTS ||
    plan.outputFloat32Bytes !== OUTPUT_BYTES
  ) throw new Error("OPT-0011 A single-window geometry changed");
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

function validateWorkerIdentity(identity: Opt0011Fp16WindowRunIdentity): void {
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
  if (JSON.stringify(parseOpt0011Fp16WindowRunIdentity(roundTrip)) !==
    JSON.stringify(identity)) {
    throw new Error("OPT-0011 worker identity changed during clone");
  }
}

function startWorkerHeartbeat(): HeartbeatController {
  let stopped = false;
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
      stopped = true;
      clearInterval(timer);
      receipt = Object.freeze({
        stopped,
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
    throw new Error(`OPT-0011 FP16 window requires one ${name}`);
  }
  return values[0]!;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
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
    schema: "ace-opt-0011-fp16-vae-b256-window-correctness-v1",
    status: "failed",
    experimentId: "OPT-0011",
    error: errorReceipt(error),
  });
}

export function parseOpt0011Fp16WindowRawResultChunkOffset(value: string): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error("OPT-0011 FP16 window raw-result offset is not canonical");
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error("OPT-0011 FP16 window raw-result offset is unsafe");
  }
  return offset;
}

export function sliceOpt0011Fp16WindowRawResultChunk(
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
    throw new Error("OPT-0011 FP16 window raw-result offset is invalid");
  }
  let end = Math.min(
    offset + OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS,
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
      const slice = sliceOpt0011Fp16WindowRawResultChunk(
        raw,
        parseOpt0011Fp16WindowRawResultChunkOffset(input.value),
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
  })) throw new Error("OPT-0011 FP16 window could not publish receipt");
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    schema: result["schema"] ?? null,
    status,
    experimentId: "OPT-0011",
    classification: result["classification"] ?? null,
    rawResultJsonCodeUnitLength: raw.length,
    rawResultRetrieval: "bounded-restartable-dom-chunks-from-page-start",
    rawResultChunkCodeUnitLimit:
      OPT_0011_FP16_WINDOW_RAW_RESULT_CHUNK_CODE_UNITS,
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

// Keep the public WGSL generators live in the harness bundle and static gate.
// Actual executed module source is captured from createShaderModule above.
export function opt0011Fp16WindowGeneratedShaderSourceCount(): number {
  const graph = planAceVaeDecoder(FRAMES);
  const sources = new Set<string>([aceFp16VaeIngressWgsl()]);
  for (const operation of graph.operations) {
    switch (operation.kind) {
      case "conv1d":
        sources.add(aceFp16VaeConv1dWgsl(
          operation.shape,
          operation.bias !== undefined,
          operation.output === "output" ? "float32" : "float16",
        ));
        break;
      case "conv-transpose1d":
        sources.add(aceFp16VaeConvTranspose1dWgsl(operation.shape));
        break;
      case "snake":
        sources.add(aceFp16VaeSnakeWgsl(operation.shape));
        break;
      case "add":
        sources.add(aceFp16VaeAddWgsl());
        break;
    }
  }
  // Touch the A generators too; manifest-specific part sources are captured
  // from the real accepted backend rather than guessed by this static count.
  void [
    aceCorrectnessVaeConv1dWgsl,
    aceCorrectnessVaeConvTranspose1dWgsl,
    aceCorrectnessVaeConvTranspose1dPartWgsl,
    aceCorrectnessVaeSnakeWgsl,
    aceCorrectnessVaeAddWgsl,
    resolveAceLogicalTensor,
  ];
  return sources.size;
}
