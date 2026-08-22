import { ACE_DIRECT_DCW_CONFIGURATION } from "../../src/api.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_TILE_LAYOUT,
  ACE_PACKAGE_CONVERTER_REVISION,
  type AcePackageFileRecord,
  type AcePackageManifest,
} from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  type AceExecutionProfile,
} from "../../src/webgpu/capabilities.js";
import { requestAceWebGpuDevice } from "../../src/webgpu/device.js";
import {
  AceDitGpuBackend,
  planAceDitGpuBackendMemory,
  planAceDitPhysicalCommandBufferCount,
  type AceDitGpuBackendProgress,
  type AceDitGpuBackendMemoryPlan,
  type AceDitGemmBackend,
} from "../../src/webgpu/dit-backend.js";
import { ACE_DIT_GRAPH_QUANTUM_COUNT } from "../../src/webgpu/dit-graph.js";

export const OPT_0003_REFERENCE_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
export const OPT_0003_REFERENCE_MANIFEST_PATH =
  "/model/files-reference/manifest.json";
export const OPT_0003_PACKAGE_NATIVE_SHAPE = Object.freeze({
  batch: 1,
  latentFrames: 129,
  conditionTokens: 1,
});
export const OPT_0003_DIT_LOGICAL_TENSOR_BYTES = 3_150_917_760;
export const OPT_0003_DIT_RESIDENT_FILE_BYTES = 3_150_917_888;
export const OPT_0003_DIT_FILE_COUNT = 50;
export const OPT_0003_DIT_TENSOR_COUNT = 476;
export const OPT_0003_DIT_TILE_MAJOR_GEMM_COUNT = 271;
export const OPT_0003_FINAL_LATENT_ELEMENTS = 8_256;
export const OPT_0003_PORTABLE_COMMAND_BUFFER_COUNT = 634;
export const OPT_0003_SUBGROUP_COMMAND_BUFFER_COUNT = 826;

const EXPECTED_ARENA_BYTES = 19_191_040;
const EXPECTED_READBACK_BYTES = 33_024;
const EXPECTED_ACCOUNTED_GPU_BYTES = 3_170_141_952;
const EXPECTED_BOUNDED_CPU_BYTES = 173_876;
const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;

interface HeartbeatSnapshot {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface PhaseProgressSnapshot {
  readonly eventCount: number;
  readonly phaseFileIndex: number;
  readonly phaseFileCount: number;
  readonly loadedPhaseBytes: number;
  readonly totalPhaseBytes: number;
}

interface BackendRunSummary {
  readonly profile: string;
  readonly gemmBackend: AceDitGemmBackend;
  readonly memory: AceDitGpuBackendMemoryPlan;
  readonly phaseUploadWallMilliseconds: number;
  readonly compileWallMilliseconds: number;
  readonly runWallMilliseconds: number;
  readonly phaseProgress: PhaseProgressSnapshot;
  readonly compileHeartbeat: HeartbeatSnapshot;
  readonly runHeartbeat: HeartbeatSnapshot;
  readonly compileProgressEvents: number;
  readonly denoiseProgressEvents: number;
  readonly readbackProgressEvents: number;
  readonly layerPhysicalProgressEvents: number;
  readonly evaluationLayerPairCount: number;
  readonly compileProgressIntervalsMilliseconds: readonly number[];
  readonly runProgressIntervalsMilliseconds: readonly number[];
  readonly maximumCompileProgressIntervalMilliseconds: number;
  readonly maximumRunProgressIntervalMilliseconds: number;
  readonly progressDigest: string;
  readonly result: Readonly<{
    commandBuffersSubmitted: number;
    queueDrains: number;
    cooperativeIdleMs: number;
    completedEvaluations: number;
  }>;
  readonly latent: Readonly<{
    elementCount: number;
    byteLength: number;
    sha256: string;
    finite: boolean;
    nonzeroElementCount: number;
    unchangedInitialBitCount: number;
  }>;
  readonly finalLatent: Float32Array<ArrayBuffer>;
}

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly ditFiles: readonly AcePackageFileRecord[];
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly summary: Readonly<{
    manifestSha256: string;
    manifestByteLength: number;
    converterRevision: number;
    ditTensorCount: number;
    ditTileMajorGemmCount: number;
    ditLogicalTensorBytes: number;
    ditFileCount: number;
    ditResidentFileBytes: number;
    acquiredFileCount: number;
    cachedFileCount: number;
    downloadedFileCount: number;
  }>;
}

interface DeterministicInputs {
  readonly condition: Float32Array<ArrayBuffer>;
  readonly context: Float32Array<ArrayBuffer>;
  readonly initialLatent: Float32Array<ArrayBuffer>;
}

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const startButton = document.querySelector<HTMLButtonElement>("#start");
  if (startButton === null) throw new Error("Missing start button");
  startButton.addEventListener("click", () => {
    startButton.disabled = true;
    document.body.dataset.status = "running";
    void run().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", errorResult(error)),
    );
  }, { once: true });
}

async function run(): Promise<unknown> {
  const recordedAt = new Date().toISOString();
  updateProgress("authenticating the reference rev4 manifest");
  const preparedPackage = await preparePackage();
  updateProgress("requesting the production fixed-32 WebGPU device");
  const runtimeEvents: unknown[] = [];
  const context = await requestAceWebGpuDevice({
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  try {
    validateDevice(context.capabilities);
    const inputs = deterministicInputs();
    const portable = await runBackend(
      context.device,
      context.capabilities.adapterInfo.subgroupMinSize,
      context.capabilities.adapterInfo.subgroupMaxSize,
      preparedPackage,
      ACE_REFERENCE_PORTABLE_PROFILE,
      inputs,
    );
    const subgroup = await runBackend(
      context.device,
      context.capabilities.adapterInfo.subgroupMinSize,
      context.capabilities.adapterInfo.subgroupMaxSize,
      preparedPackage,
      ACE_REFERENCE_SUBGROUP_PROFILE,
      inputs,
    );
    const comparison = await compareLatents(portable, subgroup);
    if (runtimeEvents.length !== 0) {
      throw new Error("WebGPU runtime emitted a device-loss or uncaptured-error event");
    }
    return Object.freeze({
      schema: "ace-opt-0003-package-native-dit-integration-v1",
      recordedAt,
      completedAt: new Date().toISOString(),
      scope: Object.freeze({
        completeDitGraph: true,
        layerCount: 24,
        denoisingEvaluations: 8,
        logicalGraphQuanta: ACE_DIT_GRAPH_QUANTUM_COUNT,
        shape: OPT_0003_PACKAGE_NATIVE_SHAPE,
        sequentialHeavyweightPhases: true,
        simultaneousHeavyweightPhaseCount: 1,
        performanceClaim: false,
        thermalGateRequired: false,
        directBrowserCancellationReloadPerformed: false,
        cancellationEvidence:
          "existing actual-Chrome OPT-0003 range cancellation plus DiT backend lifecycle contracts",
      }),
      package: preparedPackage.summary,
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: context.capabilities.executionProfile,
        schedulingProfile: context.capabilities.schedulingProfile,
        adapterInfo: context.capabilities.adapterInfo,
        adapterFeatures: context.capabilities.adapterFeatures,
        deviceFeatures: context.capabilities.deviceFeatures,
        adapterLimits: context.capabilities.adapterLimits,
        deviceLimits: context.capabilities.deviceLimits,
        runtimeEvents,
      }),
      portable: publicBackendSummary(portable),
      subgroup: publicBackendSummary(subgroup),
      comparison,
    });
  } finally {
    context.destroy();
  }
}

async function preparePackage(): Promise<PreparedPackage> {
  const manifestUrl = new URL(OPT_0003_REFERENCE_MANIFEST_PATH, location.href).href;
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: OPT_0003_REFERENCE_MANIFEST_SHA256,
    expectedProfile: "reference",
  });
  if (loaded.manifestSha256 !== OPT_0003_REFERENCE_MANIFEST_SHA256) {
    throw new Error("Authenticated manifest digest diverged from its trust root");
  }
  const inventory = validateDitInventory(loaded.manifest);
  const acquisitionManifest = Object.freeze({
    ...loaded.manifest,
    files: inventory.ditFiles,
  });
  const cache = await AceOpfsModelCache.open();
  let lastStatusUpdate = 0;
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      const now = performance.now();
      if (
        now - lastStatusUpdate >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
        progress.fileIndex + 1 === progress.fileCount &&
          progress.fileReceivedBytes === progress.fileBytes
      ) {
        lastStatusUpdate = now;
        updateProgress(
          `acquiring DiT shard ${progress.fileIndex + 1}/${progress.fileCount} ` +
          `(${formatBytes(progress.completedBytes)}/${formatBytes(progress.totalBytes)}, ` +
          `${progress.source})`,
        );
      }
    },
  });
  if (
    acquired.files.size !== OPT_0003_DIT_FILE_COUNT ||
    acquired.plan.files.length !== OPT_0003_DIT_FILE_COUNT ||
    acquired.plan.runtimeBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES
  ) throw new Error("Bounded DiT acquisition accounting diverged");
  return Object.freeze({
    manifest: loaded.manifest,
    manifestUrl: loaded.manifestUrl,
    ditFiles: inventory.ditFiles,
    acquiredFiles: acquired.files,
    summary: Object.freeze({
      manifestSha256: loaded.manifestSha256,
      manifestByteLength: loaded.manifestByteLength,
      converterRevision: loaded.manifest.provenance.converterRevision,
      ditTensorCount: inventory.ditTensorCount,
      ditTileMajorGemmCount: inventory.ditTileMajorGemmCount,
      ditLogicalTensorBytes: inventory.ditLogicalTensorBytes,
      ditFileCount: inventory.ditFiles.length,
      ditResidentFileBytes: inventory.ditResidentFileBytes,
      acquiredFileCount: acquired.files.size,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
    }),
  });
}

function validateDitInventory(manifest: AcePackageManifest): Readonly<{
  ditFiles: readonly AcePackageFileRecord[];
  ditTensorCount: number;
  ditTileMajorGemmCount: number;
  ditLogicalTensorBytes: number;
  ditResidentFileBytes: number;
}> {
  if (manifest.provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION) {
    throw new Error("Package-native gate requires converter revision 4");
  }
  const ditTensors = Object.values(manifest.tensors).filter(
    (tensor) => tensor.phase === "dit",
  );
  const tileMajor = ditTensors.filter(
    (tensor) => tensor.layout === ACE_DIT_GEMM_TILE_LAYOUT,
  );
  if (ditTensors.length !== OPT_0003_DIT_TENSOR_COUNT) {
    throw new Error(`Expected ${OPT_0003_DIT_TENSOR_COUNT} DiT tensor records`);
  }
  if (tileMajor.length !== OPT_0003_DIT_TILE_MAJOR_GEMM_COUNT) {
    throw new Error(`Expected ${OPT_0003_DIT_TILE_MAJOR_GEMM_COUNT} tiled DiT GEMMs`);
  }
  for (const tensor of tileMajor) {
    const [columns, inner] = tensor.logicalShape;
    if (
      tensor.dtype !== "uint32-bf16-pairs" ||
      tensor.transformation !== ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION ||
      tensor.logicalShape.length !== 2 ||
      columns === undefined || inner === undefined ||
      columns % 128 !== 0 || inner % 32 !== 0 ||
      tensor.partStart !== 0 || tensor.partEnd !== columns
    ) throw new Error("Tile-major DiT GEMM record violated the rev4 native contract");
  }
  const shardNames = new Set(ditTensors.map((tensor) => tensor.shard));
  const ditFiles = manifest.files.filter((file) => shardNames.has(file.name));
  if (
    shardNames.size !== OPT_0003_DIT_FILE_COUNT ||
    ditFiles.length !== OPT_0003_DIT_FILE_COUNT ||
    ditFiles.some((file) => file.kind !== "weights")
  ) throw new Error("DiT phase did not resolve to exactly 50 weight shards");
  const ditLogicalTensorBytes = sumSafe(
    ditTensors.map((tensor) => tensor.byteLength),
    "DiT logical tensor bytes",
  );
  const ditResidentFileBytes = sumSafe(
    ditFiles.map((file) => file.byteLength),
    "DiT resident file bytes",
  );
  if (
    ditLogicalTensorBytes !== OPT_0003_DIT_LOGICAL_TENSOR_BYTES ||
    ditResidentFileBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES
  ) throw new Error("DiT phase byte accounting diverged from the canonical package");
  return Object.freeze({
    ditFiles: Object.freeze(ditFiles),
    ditTensorCount: ditTensors.length,
    ditTileMajorGemmCount: tileMajor.length,
    ditLogicalTensorBytes,
    ditResidentFileBytes,
  });
}

async function runBackend(
  device: GPUDevice,
  subgroupMinSize: number | undefined,
  subgroupMaxSize: number | undefined,
  preparedPackage: PreparedPackage,
  executionProfile: AceExecutionProfile,
  inputs: DeterministicInputs,
): Promise<BackendRunSummary> {
  const expectedBackend: AceDitGemmBackend =
    executionProfile.id === ACE_REFERENCE_SUBGROUP_PROFILE.id
      ? "fixed32-subgroups"
      : "portable";
  const expectedCommandBuffers = expectedBackend === "portable"
    ? OPT_0003_PORTABLE_COMMAND_BUFFER_COUNT
    : OPT_0003_SUBGROUP_COMMAND_BUFFER_COUNT;
  updateProgress(`${executionProfile.id}: uploading 50 authenticated DiT shards`);
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceDitGpuBackend | undefined;
  const progress: AceDitGpuBackendProgress[] = [];
  const compileProgressIntervalsMilliseconds: number[] = [];
  const runProgressIntervalsMilliseconds: number[] = [];
  let phaseProgressEvents = 0;
  let phaseProgress: PhaseProgressSnapshot | undefined;
  let lastStatusUpdate = 0;
  const phaseStarted = performance.now();
  try {
    phase = await AceGpuTensorPhase.load(
      device,
      preparedPackage.manifest,
      preparedPackage.acquiredFiles,
      ["dit"],
      {
        onProgress: (event) => {
          phaseProgressEvents += 1;
          phaseProgress = Object.freeze({
            eventCount: phaseProgressEvents,
            phaseFileIndex: event.phaseFileIndex,
            phaseFileCount: event.phaseFileCount,
            loadedPhaseBytes: event.loadedPhaseBytes,
            totalPhaseBytes: event.totalPhaseBytes,
          });
          const now = performance.now();
          if (
            now - lastStatusUpdate >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
            event.loadedPhaseBytes === event.totalPhaseBytes
          ) {
            lastStatusUpdate = now;
            updateProgress(
              `${executionProfile.id}: GPU upload ${event.phaseFileIndex + 1}/` +
              `${event.phaseFileCount} (${formatBytes(event.loadedPhaseBytes)}/` +
              `${formatBytes(event.totalPhaseBytes)})`,
            );
          }
        },
      },
    );
    const phaseUploadWallMilliseconds = performance.now() - phaseStarted;
    if (
      phase.residentBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES ||
      phase.packageManifest !== preparedPackage.manifest ||
      phase.phases.length !== 1 || phase.phases[0] !== "dit" ||
      phaseProgress === undefined ||
      phaseProgress.phaseFileCount !== OPT_0003_DIT_FILE_COUNT ||
      phaseProgress.loadedPhaseBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES ||
      phaseProgress.totalPhaseBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES
    ) throw new Error(`${executionProfile.id} phase upload accounting diverged`);

    updateProgress(`${executionProfile.id}: compiling the 249-stage graph`);
    const compileHeartbeat = startHeartbeat();
    const compileStarted = performance.now();
    let lastCompileProgressAt = compileStarted;
    let lastRunProgressAt = 0;
    let compileHeartbeatResult: HeartbeatSnapshot;
    try {
      const backendInputs = cloneInputs(inputs);
      const transferredPhase = phase;
      phase = undefined;
      backend = await AceDitGpuBackend.create({
        device,
        executionProfile,
        ...(subgroupMinSize === undefined ? {} : { subgroupMinSize }),
        ...(subgroupMaxSize === undefined ? {} : { subgroupMaxSize }),
        shape: OPT_0003_PACKAGE_NATIVE_SHAPE,
        inputs: backendInputs,
        dcwConfiguration: ACE_DIRECT_DCW_CONFIGURATION,
        ownedDitWeights: transferredPhase,
        onProgress: (event) => {
          const now = performance.now();
          progress.push(event);
          if (event.stage === "compile") {
            compileProgressIntervalsMilliseconds.push(now - lastCompileProgressAt);
            lastCompileProgressAt = now;
            if (
              event.compiledQuanta % 10 === 0 ||
              event.compiledQuanta === event.totalQuanta
            ) updateProgress(
              `${executionProfile.id}: compiled ${event.compiledQuanta}/` +
              `${event.totalQuanta} logical stages`,
            );
          } else {
            if (lastRunProgressAt === 0) {
              throw new Error("DiT run progress arrived before its timing origin");
            }
            runProgressIntervalsMilliseconds.push(now - lastRunProgressAt);
            lastRunProgressAt = now;
          }
        },
      });
    } finally {
      compileHeartbeatResult = compileHeartbeat.stop();
    }
    const compileWallMilliseconds = performance.now() - compileStarted;
    validateHeartbeat(compileHeartbeatResult, `${executionProfile.id} compilation`);
    validateMemory(backend.memory, expectedBackend, expectedCommandBuffers);
    validateCompilationProgress(progress);

    updateProgress(
      `${executionProfile.id}: running ${expectedCommandBuffers - 1} graph commands`,
    );
    const runHeartbeat = startHeartbeat();
    const runStarted = performance.now();
    lastRunProgressAt = runStarted;
    let runHeartbeatResult: HeartbeatSnapshot;
    let result;
    try {
      result = await backend.run();
    } finally {
      runHeartbeatResult = runHeartbeat.stop();
    }
    const runWallMilliseconds = performance.now() - runStarted;
    validateHeartbeat(runHeartbeatResult, `${executionProfile.id} execution`);
    const runProgressSummary = validateRunProgress(progress, backend.memory, result);
    const finalLatent = result.finalLatent;
    if (
      finalLatent.length !== OPT_0003_FINAL_LATENT_ELEMENTS ||
      finalLatent.byteLength !== EXPECTED_READBACK_BYTES
    ) throw new Error(`${executionProfile.id} returned the wrong latent size`);
    const latentSummary = await summarizeLatent(finalLatent, inputs.initialLatent);
    if (
      !latentSummary.finite ||
      latentSummary.nonzeroElementCount === 0 ||
      latentSummary.unchangedInitialBitCount !== 0
    ) throw new Error(`${executionProfile.id} final latent failed validity checks`);
    const summary: BackendRunSummary = Object.freeze({
      profile: executionProfile.id,
      gemmBackend: expectedBackend,
      memory: backend.memory,
      phaseUploadWallMilliseconds,
      compileWallMilliseconds,
      runWallMilliseconds,
      phaseProgress,
      compileHeartbeat: compileHeartbeatResult,
      runHeartbeat: runHeartbeatResult,
      compileProgressEvents: progress.filter((event) => event.stage === "compile").length,
      denoiseProgressEvents: progress.filter((event) => event.stage === "denoise").length,
      readbackProgressEvents: progress.filter((event) => event.stage === "readback").length,
      layerPhysicalProgressEvents: runProgressSummary.layerPhysicalProgressEvents,
      evaluationLayerPairCount: runProgressSummary.evaluationLayerPairCount,
      compileProgressIntervalsMilliseconds: Object.freeze(
        compileProgressIntervalsMilliseconds,
      ),
      runProgressIntervalsMilliseconds: Object.freeze(
        runProgressIntervalsMilliseconds,
      ),
      maximumCompileProgressIntervalMilliseconds: Math.max(
        0,
        ...compileProgressIntervalsMilliseconds,
      ),
      maximumRunProgressIntervalMilliseconds: Math.max(
        0,
        ...runProgressIntervalsMilliseconds,
      ),
      progressDigest: progressFingerprint(progress),
      result: Object.freeze({
        commandBuffersSubmitted: result.commandBuffersSubmitted,
        queueDrains: result.queueDrains,
        cooperativeIdleMs: result.cooperativeIdleMs,
        completedEvaluations: result.completedEvaluations,
      }),
      latent: latentSummary,
      finalLatent,
    });
    backend = undefined;
    return summary;
  } finally {
    await backend?.destroy();
    phase?.destroy();
    await device.queue.onSubmittedWorkDone();
  }
}

function validateMemory(
  memory: AceDitGpuBackendMemoryPlan,
  expectedBackend: AceDitGemmBackend,
  expectedCommandBuffers: number,
): void {
  const plannedGraphCommands = planAceDitPhysicalCommandBufferCount(
    OPT_0003_PACKAGE_NATIVE_SHAPE,
    expectedBackend,
  );
  const recomputed = planAceDitGpuBackendMemory(
    "reference-bf16",
    OPT_0003_PACKAGE_NATIVE_SHAPE,
    OPT_0003_DIT_RESIDENT_FILE_BYTES,
    expectedBackend,
  );
  if (
    plannedGraphCommands + 1 !== expectedCommandBuffers ||
    memory.modelProfile !== "reference-bf16" ||
    memory.gemmBackend !== expectedBackend ||
    memory.residentWeightBytes !== OPT_0003_DIT_RESIDENT_FILE_BYTES ||
    memory.arena.allocatedArenaBytes !== EXPECTED_ARENA_BYTES ||
    memory.readbackBufferBytes !== EXPECTED_READBACK_BYTES ||
    memory.accountedGpuBytes !== EXPECTED_ACCOUNTED_GPU_BYTES ||
    memory.boundedCpuBytes !== EXPECTED_BOUNDED_CPU_BYTES ||
    memory.logicalGraphQuantumCount !== ACE_DIT_GRAPH_QUANTUM_COUNT ||
    memory.commandBufferCount !== expectedCommandBuffers ||
    JSON.stringify(memory) !== JSON.stringify(recomputed)
  ) throw new Error(`${expectedBackend} memory/command planning diverged`);
}

function validateCompilationProgress(progress: readonly AceDitGpuBackendProgress[]): void {
  const compilation = progress.filter((event) => event.stage === "compile");
  if (compilation.length !== ACE_DIT_GRAPH_QUANTUM_COUNT) {
    throw new Error("DiT compilation did not report every logical stage");
  }
  for (let index = 0; index < compilation.length; index += 1) {
    const event = compilation[index]!;
    if (
      event.stage !== "compile" ||
      event.compiledQuanta !== index + 1 ||
      event.totalQuanta !== ACE_DIT_GRAPH_QUANTUM_COUNT
    ) throw new Error("DiT compilation progress was not exact and monotonic");
  }
}

function validateRunProgress(
  progress: readonly AceDitGpuBackendProgress[],
  memory: AceDitGpuBackendMemoryPlan,
  result: Readonly<{
    commandBuffersSubmitted: number;
    queueDrains: number;
    cooperativeIdleMs: number;
    completedEvaluations: number;
  }>,
): Readonly<{
  layerPhysicalProgressEvents: number;
  evaluationLayerPairCount: number;
}> {
  const graphCommands = memory.commandBufferCount - 1;
  const denoise = progress.filter((event) => event.stage === "denoise");
  const readback = progress.filter((event) => event.stage === "readback");
  if (denoise.length !== graphCommands || readback.length !== 1) {
    throw new Error("DiT run progress count diverged from physical planning");
  }
  let priorEvaluations = 0;
  const commandIds = new Set<string>();
  const evaluationLayerCounts = new Map<string, number>();
  let layerPhysicalProgressEvents = 0;
  for (let index = 0; index < denoise.length; index += 1) {
    const event = denoise[index]!;
    if (event.stage !== "denoise") throw new Error("Unexpected progress stage");
    const completed = index + 1;
    const expectedIdle = Math.min(completed, graphCommands - 1);
    if (
      event.completedCommandBuffers !== completed ||
      event.totalCommandBuffers !== memory.commandBufferCount ||
      event.queueDrains !== completed ||
      event.cooperativeIdleMs !== expectedIdle ||
      event.graph.completedQuanta !== completed ||
      event.graph.totalQuanta !== graphCommands ||
      event.graph.queueDrains !== completed ||
      event.graph.cooperativeIdleMs !== expectedIdle ||
      event.graph.completedEvaluations !== event.completedEvaluations ||
      event.completedEvaluations < priorEvaluations ||
      event.completedEvaluations > 8 ||
      event.graph.subquantumIndex < 0 ||
      event.graph.subquantumIndex >= event.graph.subquantumCount ||
      commandIds.has(event.graph.commandId)
    ) throw new Error("DiT physical progress accounting diverged");
    priorEvaluations = event.completedEvaluations;
    commandIds.add(event.graph.commandId);
    if (event.graph.quantum.kind === "layer") {
      layerPhysicalProgressEvents += 1;
      const pair = `${event.graph.quantum.evaluation}:${event.graph.quantum.layer}`;
      evaluationLayerCounts.set(pair, (evaluationLayerCounts.get(pair) ?? 0) + 1);
    }
  }
  if (priorEvaluations !== 8 || commandIds.size !== graphCommands) {
    throw new Error("DiT graph did not complete all eight evaluations exactly once");
  }
  const final = readback[0]!;
  if (
    final.stage !== "readback" ||
    final.completedCommandBuffers !== memory.commandBufferCount ||
    final.totalCommandBuffers !== memory.commandBufferCount ||
    final.queueDrains !== memory.commandBufferCount ||
    final.cooperativeIdleMs !== memory.commandBufferCount - 1 ||
    final.completedEvaluations !== 8 ||
    result.commandBuffersSubmitted !== memory.commandBufferCount ||
    result.queueDrains !== memory.commandBufferCount ||
    result.cooperativeIdleMs !== memory.commandBufferCount - 1 ||
    result.completedEvaluations !== 8
  ) throw new Error("DiT readback/result accounting diverged");
  const expectedLayerFragments = memory.gemmBackend === "portable" ? 3 : 4;
  if (
    evaluationLayerCounts.size !== 24 * 8 ||
    [...evaluationLayerCounts.values()].some(
      (count) => count !== expectedLayerFragments,
    ) ||
    layerPhysicalProgressEvents !== 24 * 8 * expectedLayerFragments
  ) throw new Error("DiT run did not visit every evaluation/layer pair exactly");
  return Object.freeze({
    layerPhysicalProgressEvents,
    evaluationLayerPairCount: evaluationLayerCounts.size,
  });
}

async function summarizeLatent(
  latent: Float32Array<ArrayBuffer>,
  initial: Float32Array<ArrayBuffer>,
): Promise<BackendRunSummary["latent"]> {
  const latentBits = new Uint32Array(latent.buffer, latent.byteOffset, latent.length);
  const initialBits = new Uint32Array(initial.buffer, initial.byteOffset, initial.length);
  let finite = true;
  let nonzeroElementCount = 0;
  let unchangedInitialBitCount = 0;
  for (let index = 0; index < latent.length; index += 1) {
    const value = latent[index]!;
    if (!Number.isFinite(value)) finite = false;
    if (value !== 0) nonzeroElementCount += 1;
    if (latentBits[index] === initialBits[index]) unchangedInitialBitCount += 1;
  }
  return Object.freeze({
    elementCount: latent.length,
    byteLength: latent.byteLength,
    sha256: await sha256Hex(new Uint8Array(
      latent.buffer,
      latent.byteOffset,
      latent.byteLength,
    )),
    finite,
    nonzeroElementCount,
    unchangedInitialBitCount,
  });
}

async function compareLatents(
  portable: BackendRunSummary,
  subgroup: BackendRunSummary,
): Promise<Readonly<{
  bitMismatchCount: number;
  firstMismatchIndex: number | null;
  identicalSha256: boolean;
  outputElements: number;
}>> {
  const portableBits = new Uint32Array(portable.finalLatent.buffer);
  const subgroupBits = new Uint32Array(subgroup.finalLatent.buffer);
  if (portableBits.length !== subgroupBits.length) {
    throw new Error("Portable and subgroup latent lengths diverged");
  }
  let bitMismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < portableBits.length; index += 1) {
    if (portableBits[index] !== subgroupBits[index]) {
      bitMismatchCount += 1;
      firstMismatchIndex ??= index;
    }
  }
  const identicalSha256 = portable.latent.sha256 === subgroup.latent.sha256;
  if (bitMismatchCount !== 0 || !identicalSha256) {
    throw new Error(
      `Package-native DiT A/B diverged at ${bitMismatchCount} final-latent words`,
    );
  }
  return Object.freeze({
    bitMismatchCount,
    firstMismatchIndex,
    identicalSha256,
    outputElements: portableBits.length,
  });
}

function deterministicInputs(): DeterministicInputs {
  const condition = deterministicValues(2_048, 17, 11, 257, 256);
  const context = deterministicValues(129 * 128, 29, 7, 251, 512);
  const initialLatent = deterministicValues(OPT_0003_FINAL_LATENT_ELEMENTS, 43, 19, 241, 384);
  return Object.freeze({ condition, context, initialLatent });
}

function deterministicValues(
  length: number,
  multiplier: number,
  offset: number,
  modulus: number,
  divisor: number,
): Float32Array<ArrayBuffer> {
  const values = new Float32Array(length);
  const midpoint = Math.floor(modulus / 2);
  for (let index = 0; index < length; index += 1) {
    values[index] = Math.fround((((index * multiplier + offset) % modulus) - midpoint) / divisor);
  }
  return values;
}

function cloneInputs(inputs: DeterministicInputs): DeterministicInputs {
  return Object.freeze({
    condition: inputs.condition.slice(),
    context: inputs.context.slice(),
    initialLatent: inputs.initialLatent.slice(),
  });
}

function validateDevice(capabilities: Readonly<{
  executionProfile: AceExecutionProfile;
  schedulingProfile: string;
  adapterInfo: Readonly<{
    subgroupMinSize?: number;
    subgroupMaxSize?: number;
    isFallbackAdapter?: boolean;
  }>;
  deviceFeatures: readonly string[];
}>): void {
  if (
    capabilities.executionProfile.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id ||
    capabilities.schedulingProfile !== "cooperative" ||
    capabilities.adapterInfo.subgroupMinSize !== 32 ||
    capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    capabilities.adapterInfo.isFallbackAdapter === true ||
    !capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error("Package-native gate requires the M3 fixed-32 subgroup profile");
}

function publicBackendSummary(summary: BackendRunSummary): unknown {
  return Object.freeze({
    profile: summary.profile,
    gemmBackend: summary.gemmBackend,
    memory: summary.memory,
    phaseUploadWallMilliseconds: summary.phaseUploadWallMilliseconds,
    compileWallMilliseconds: summary.compileWallMilliseconds,
    runWallMilliseconds: summary.runWallMilliseconds,
    phaseProgress: summary.phaseProgress,
    compileHeartbeat: summary.compileHeartbeat,
    runHeartbeat: summary.runHeartbeat,
    compileProgressEvents: summary.compileProgressEvents,
    denoiseProgressEvents: summary.denoiseProgressEvents,
    readbackProgressEvents: summary.readbackProgressEvents,
    layerPhysicalProgressEvents: summary.layerPhysicalProgressEvents,
    evaluationLayerPairCount: summary.evaluationLayerPairCount,
    compileProgressIntervalsMilliseconds: summary.compileProgressIntervalsMilliseconds,
    runProgressIntervalsMilliseconds: summary.runProgressIntervalsMilliseconds,
    maximumCompileProgressIntervalMilliseconds:
      summary.maximumCompileProgressIntervalMilliseconds,
    maximumRunProgressIntervalMilliseconds:
      summary.maximumRunProgressIntervalMilliseconds,
    progressDigest: summary.progressDigest,
    result: summary.result,
    latent: summary.latent,
  });
}

function progressFingerprint(progress: readonly AceDitGpuBackendProgress[]): string {
  let hash = 0x811c9dc5;
  const update = (value: number): void => {
    hash ^= value >>> 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  };
  for (const event of progress) {
    update(event.stage === "compile" ? 1 : event.stage === "denoise" ? 2 : 3);
    if (event.stage === "compile") {
      update(event.compiledQuanta);
      update(event.totalQuanta);
    } else {
      update(event.completedCommandBuffers);
      update(event.totalCommandBuffers);
      update(event.queueDrains);
      update(event.cooperativeIdleMs);
      update(event.completedEvaluations);
    }
  }
  return hash.toString(16).padStart(8, "0");
}

function startHeartbeat(): { stop(): HeartbeatSnapshot } {
  const animationGaps: number[] = [];
  const timerGaps: number[] = [];
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let lastAnimation = performance.now();
  let lastTimer = lastAnimation;
  let frameHandle = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    animationGaps.push(now - lastAnimation);
    lastAnimation = now;
    animationFrameCount += 1;
    frameHandle = requestAnimationFrame(frame);
  };
  frameHandle = requestAnimationFrame(frame);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    timerGaps.push(now - lastTimer);
    lastTimer = now;
    timerTickCount += 1;
  }, 10);
  return {
    stop(): HeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearInterval(timerHandle);
      }
      return Object.freeze({
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds: Math.max(0, ...animationGaps),
        maximumTimerGapMilliseconds: Math.max(0, ...timerGaps),
      });
    },
  };
}

function validateHeartbeat(snapshot: HeartbeatSnapshot, label: string): void {
  if (
    snapshot.animationFrameCount + snapshot.timerTickCount === 0 ||
    !Number.isFinite(snapshot.maximumAnimationFrameGapMilliseconds) ||
    !Number.isFinite(snapshot.maximumTimerGapMilliseconds) ||
    snapshot.maximumAnimationFrameGapMilliseconds < 0 ||
    snapshot.maximumTimerGapMilliseconds < 0
  ) throw new Error(`${label} did not retain valid page-heartbeat telemetry`);
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0")).join("");
}

function sumSafe(values: readonly number[], label: string): number {
  let sum = 0;
  for (const value of values) {
    sum += value;
    if (!Number.isSafeInteger(sum)) throw new RangeError(`${label} overflowed`);
  }
  return sum;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function updateProgress(message: string): void {
  const node = document.querySelector<HTMLElement>("#progress");
  if (node === null) throw new Error("Missing progress node");
  node.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  updateProgress(status);
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result node");
  node.textContent = JSON.stringify(result);
}

function errorResult(error: unknown): unknown {
  return error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
    : Object.freeze({ error: String(error) });
}
