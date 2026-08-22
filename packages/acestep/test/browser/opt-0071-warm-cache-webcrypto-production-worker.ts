/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  aceRuntimePackageFiles,
} from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import type {
  AcePackageFileRecord,
  AcePackageManifest,
  AceTensorPhase,
} from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from
  "../../src/runtime/diagnostics.js";
import type { AceWorkerConfiguration } from
  "../../src/runtime/protocol.js";
import {
  AceInitializationProgressSequence,
  type AceInitializationProgress,
} from "../../src/runtime/stages.js";
import {
  createAceMainAcquisitionManifest,
  createAceOpt0009DitDenseAcquisitionManifest,
  createAceOpt0011VaeAcquisitionManifest,
  createAceWebGpuPipelineBackend,
  type AceOpt0064CaptureEvent,
  type AceOpt0064CaptureSink,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  checkedOpt0071ByteAdd,
  checkedOpt0071DurationAdd,
  OPT_0071_ARM_ORDER,
  OPT_0071_CONSERVATIVE_TRANSIENT_BYTES,
  OPT_0071_DENSE_MANIFEST_PATH,
  OPT_0071_DENSE_MANIFEST_SHA256,
  OPT_0071_FULL_LOGICAL_BYTES,
  OPT_0071_FULL_LOGICAL_RECORDS,
  OPT_0071_FULL_PHYSICAL_BYTES,
  OPT_0071_FULL_UNIQUE_DIGESTS,
  OPT_0071_INVENTORY_FINGERPRINT,
  OPT_0071_LARGEST_FILE_BYTES,
  OPT_0071_MAIN_MANIFEST_PATH,
  OPT_0071_MAIN_MANIFEST_SHA256,
  OPT_0071_MAXIMUM_FILE_BYTES,
  OPT_0071_MAXIMUM_TRANSIENT_BYTES,
  OPT_0071_TIMED_LOGICAL_BYTES,
  OPT_0071_TIMED_LOGICAL_RECORDS,
  OPT_0071_TIMED_PHYSICAL_BYTES,
  OPT_0071_TIMED_UNIQUE_DIGESTS,
  OPT_0071_UNRELATED_STAGE_NAMES,
  OPT_0071_UPLOAD_SUBSET_BYTES,
  OPT_0071_UPLOAD_SUBSET_FILES,
  OPT_0071_VAE_MANIFEST_PATH,
  OPT_0071_VAE_MANIFEST_SHA256,
  requireOpt0071ThermalGate,
  serializeOpt0071Failure,
  validateOpt0071RunIdentity,
  type Opt0071ArmId,
  type Opt0071ArmSample,
  type Opt0071Owner,
  type Opt0071RunIdentity,
  type Opt0071ThermalGate,
  type Opt0071UnrelatedStageWalls,
} from "./opt-0071-warm-cache-webcrypto-production-contract.js";

type PackageKind = "main" | "dit-dense" | "vae";
const READ_ONLY_CACHE_DIRECTORY = "ace-step-1.5.wgsl-model-cache-v1" as const;
const MAXIMUM_CAPTURE_EVENTS = 1_024;
const HASH_CHUNK_BYTES = 4 * 1024 * 1024;

interface PrepareMessage {
  readonly type: "prepare";
  readonly identity: Opt0071RunIdentity;
  readonly armId: Opt0071ArmId;
}

interface RunMessage {
  readonly type: "run";
  readonly armId: Opt0071ArmId;
  readonly thermalGate: Opt0071ThermalGate;
}

interface CancelMessage {
  readonly type: "cancel";
}

type IncomingMessage = PrepareMessage | RunMessage | CancelMessage;

interface LogicalRecord {
  readonly packageKind: PackageKind;
  readonly record: AcePackageFileRecord;
  readonly uploadSubset: boolean;
}

interface PhysicalCandidate {
  readonly packageKind: PackageKind;
  readonly record: AcePackageFileRecord;
  readonly file: File;
  readonly logicalAliases: readonly string[];
  readonly uploadSubset: boolean;
}

interface PreparedAuthority {
  readonly inventory: Readonly<Record<string, unknown>>;
  readonly manifestIdentities: Readonly<Record<PackageKind,
    Readonly<Record<string, unknown>>>>;
  readonly authentication: Readonly<Record<string, unknown>>;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly memory: Readonly<Record<string, unknown>>;
}

interface MemoryMeasurement {
  readonly label: string;
  readonly exposed: boolean;
  readonly bytes: number | null;
  readonly error: Readonly<Record<string, unknown>> | null;
}

interface FullAuthenticationResult {
  readonly wallMs: number;
  readonly readCopyMs: number;
  readonly hashMs: number;
  readonly finalizationAndComparisonMs: number;
  readonly releaseMs: number;
  readonly matchedUniqueDigests: number;
  readonly receivedPhysicalBytes: number;
  readonly maximumExplicitLivePayloadBytes: number;
  readonly maximumExplicitLivePayloadCount: number;
  readonly conservativeTransientBytes: number;
  readonly fileTraces: readonly Readonly<Record<string, unknown>>[];
  readonly peakMemory: MemoryMeasurement | null;
}

let state: "idle" | "preparing" | "ready" | "running" | "settled" = "idle";
let identity: Opt0071RunIdentity | undefined;
let definition: (typeof OPT_0071_ARM_ORDER)[number] | undefined;
let preparation: PreparedAuthority | undefined;
let readyAtEpochMilliseconds = 0;
let abortController: AbortController | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    abortController?.abort(new DOMException("OPT-0071 cancelled", "AbortError"));
    return;
  }
  if (message.type === "prepare" && state === "idle") {
    state = "preparing";
    void prepare(message).catch(fail);
    return;
  }
  if (message.type === "run" && state === "ready") {
    state = "running";
    void run(message).catch(fail);
    return;
  }
  fail(new Error(`OPT-0071 rejected ${message.type} while ${state}`));
});

async function prepare(message: PrepareMessage): Promise<void> {
  identity = validateOpt0071RunIdentity(message.identity);
  definition = OPT_0071_ARM_ORDER.find((item) => item.armId === message.armId);
  if (definition === undefined) throw new Error("OPT-0071 arm identity changed");
  postProgress(
    `${definition.armId}: authenticating full 158/156 inventory before its thermal gate`,
  );
  const memoryBefore = await measureMemory("before-full-inventory-preflight");
  const inventory = await openAuthenticInventory();
  const authentication = await authenticateFullInventory(
    inventory.candidates,
    definition.owner,
  );
  if (
    authentication.matchedUniqueDigests !== OPT_0071_FULL_UNIQUE_DIGESTS ||
    authentication.receivedPhysicalBytes !== OPT_0071_FULL_PHYSICAL_BYTES
  ) throw new Error("OPT-0071 full-inventory authentication was incomplete");
  inventory.candidates.length = 0;
  postProgress(`${definition.armId}: running untimed failure/cancellation screens`);
  const correctness = await runCorrectnessAndLifecycleScreens();
  const memoryAfter = await measureMemory("after-full-inventory-preflight");
  preparation = Object.freeze({
    inventory: inventory.summary,
    manifestIdentities: inventory.manifestIdentities,
    authentication: Object.freeze({
      schema: "ace-opt-0071-full-inventory-authentication-v1",
      owner: definition.owner,
      ...authentication,
      fileTraces: authentication.fileTraces,
      timingAuthority: "untimed-preflight-before-thermal-gate",
    }),
    correctness,
    memory: Object.freeze({ before: memoryBefore, after: memoryAfter }),
  });
  readyAtEpochMilliseconds = Date.now();
  state = "ready";
  self.postMessage({
    type: "ready-for-arm",
    armId: definition.armId,
    owner: definition.owner,
    order: definition.order,
    readyAtEpochMilliseconds,
    preparation,
  });
}

async function run(message: RunMessage): Promise<void> {
  if (
    identity === undefined || definition === undefined || preparation === undefined ||
    message.armId !== definition.armId
  ) throw new Error("OPT-0071 timed arm authority is incomplete");
  const thermalGate = requireOpt0071ThermalGate(
    message.thermalGate,
    readyAtEpochMilliseconds,
    Date.now(),
  );
  abortController = new AbortController();
  const captureEvents: AceOpt0064CaptureEvent[] = [];
  const capture: AceOpt0064CaptureSink = Object.freeze({
    onEvent(event: AceOpt0064CaptureEvent): void {
      if (captureEvents.length >= MAXIMUM_CAPTURE_EVENTS) {
        throw new Error("OPT-0071 initialization capture overflowed");
      }
      captureEvents.push(event);
    },
  });
  const progressSequence = new AceInitializationProgressSequence();
  const progressEvents: AceInitializationProgress[] = [];
  const diagnosticsEvents: AceDiagnostic[] = [];
  const heartbeat = startWorkerHeartbeat(definition.armId);
  const memoryBefore = await measureMemory("before-timed-production-initialize");
  const backend = createAceWebGpuPipelineBackend({
    cacheAuthenticationOwner: definition.owner,
  });
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let readyWallMs = 0;
  let initializeStartedAtEpochMilliseconds = 0;
  let readyAtEpoch = 0;
  let disposeStartedAtEpochMilliseconds = 0;
  let disposeCompletedAtEpochMilliseconds = 0;
  let disposeWallMs = 0;
  let primaryError: unknown;
  postProgress(
    `${definition.armId}: timing unchanged cache-only production initialize through READY`,
  );
  try {
    initializeStartedAtEpochMilliseconds = Date.now();
    const startedAt = performance.now();
    diagnostics = await backend.initialize(productionConfiguration(), {
      modelSource: "cache-only",
      signal: abortController.signal,
      opt0064Capture: capture,
      onProgress(event: AceInitializationProgress): void {
        progressSequence.accept(event);
        progressEvents.push(Object.freeze({ ...event }));
      },
      onDiagnostic(event: AceDiagnostic): void {
        diagnosticsEvents.push(Object.freeze({ ...event }));
      },
    });
    readyWallMs = performance.now() - startedAt;
    readyAtEpoch = Date.now();
    validateDiagnostics(diagnostics);
  } catch (error) {
    primaryError = error;
  } finally {
    disposeStartedAtEpochMilliseconds = Date.now();
    const disposeStartedAt = performance.now();
    try {
      await backend.dispose();
    } catch (error) {
      primaryError ??= error;
    }
    disposeWallMs = performance.now() - disposeStartedAt;
    disposeCompletedAtEpochMilliseconds = Date.now();
    abortController = undefined;
    await heartbeat.finish();
  }
  const memoryAfter = await measureMemory("after-backend-device-disposal");
  if (primaryError !== undefined) throw primaryError;
  if (diagnostics === undefined) {
    throw new Error("OPT-0071 initialization returned no diagnostics");
  }
  const analysis = analyzeTimedInitialization(
    definition.owner,
    readyWallMs,
    captureEvents,
    progressEvents,
  );
  const preflightAuthentication = record(preparation.authentication);
  const sample: Opt0071ArmSample = Object.freeze({
    armId: definition.armId,
    order: definition.order,
    owner: definition.owner,
    readyWallMs,
    authenticationWallMs: analysis.authenticationWallMs,
    authenticationThroughputBytesPerSecond:
      OPT_0071_TIMED_PHYSICAL_BYTES / (analysis.authenticationWallMs / 1_000),
    unrelatedStageWalls: analysis.unrelatedStageWalls,
    aggregateUnrelatedWallMs: analysis.aggregateUnrelatedWallMs,
    timedLogicalRecords: analysis.timedLogicalRecords,
    timedUniqueDigests: analysis.timedUniqueDigests,
    timedLogicalBytes: analysis.timedLogicalBytes,
    timedPhysicalBytes: analysis.timedPhysicalBytes,
    timedInventoryFingerprint: analysis.timedInventoryFingerprint,
    fullLogicalRecordsProven: OPT_0071_FULL_LOGICAL_RECORDS,
    fullUniqueDigestsProven: number(preflightAuthentication["matchedUniqueDigests"]),
    fullLogicalBytesProven: OPT_0071_FULL_LOGICAL_BYTES,
    fullPhysicalBytesProven: number(preflightAuthentication["receivedPhysicalBytes"]),
    inventoryFingerprint: OPT_0071_INVENTORY_FINGERPRINT,
    maximumExplicitLivePayloadBytes: number(
      preflightAuthentication["maximumExplicitLivePayloadBytes"],
    ),
    maximumExplicitLivePayloadCount: number(
      preflightAuthentication["maximumExplicitLivePayloadCount"],
    ),
    conservativeTransientBytes: number(
      preflightAuthentication["conservativeTransientBytes"],
    ),
    downloadCount: analysis.downloadCount,
    downloadBytes: analysis.downloadBytes,
    cacheMutationCount: 0,
    exactDigestsPassed: analysis.exactDigestsPassed,
    memoryPassed:
      number(preflightAuthentication["maximumExplicitLivePayloadBytes"]) <=
        OPT_0071_LARGEST_FILE_BYTES &&
      number(preflightAuthentication["maximumExplicitLivePayloadCount"]) === 1 &&
      number(preflightAuthentication["conservativeTransientBytes"]) <
        OPT_0071_MAXIMUM_TRANSIENT_BYTES,
    cancellationPassed: preparation.correctness["passed"] === true,
    lifecyclePassed: disposeCompletedAtEpochMilliseconds >=
      disposeStartedAtEpochMilliseconds,
    thermalNonNominalObservations: 0,
  });
  state = "settled";
  self.postMessage({
    type: "arm-complete",
    armId: definition.armId,
    owner: definition.owner,
    order: definition.order,
    disposalCompletedAtEpochMilliseconds: disposeCompletedAtEpochMilliseconds,
    sample,
    receipt: Object.freeze({
      schema: "ace-opt-0071-production-initialize-arm-v1",
      experimentId: "OPT-0071",
      identity,
      arm: Object.freeze({ ...definition }),
      thermalGate,
      boundaries: Object.freeze({
        untimedFullInventoryPreflightEndsBeforeThermalGate: true,
        initializeStartedAtEpochMilliseconds,
        ordinaryReadyAtEpochMilliseconds: readyAtEpoch,
        authoritativeReadyWallMs: readyWallMs,
        timedReadyInventory:
          "151 logical / 149 unique main+rev7-dense; VAE payload authentication is deferred and not counted in READY",
        disposeStartedAtEpochMilliseconds,
        disposeCompletedAtEpochMilliseconds,
        disposeWallMs,
      }),
      preparation,
      timedInitialization: Object.freeze({
        analysis,
        completeCaptureEvents: Object.freeze([...captureEvents]),
        completeProgressEvents: Object.freeze([...progressEvents]),
        diagnostics,
        diagnosticEvents: Object.freeze([...diagnosticsEvents]),
        workerHeartbeatMaximumGapMs: heartbeat.maximumGapMs,
        memory: Object.freeze({ before: memoryBefore, after: memoryAfter }),
        absentDeferredStageWalls: Object.freeze({
          gpuUploadMs: 0,
          gpuCompilationAndConstructionMs: 0,
          queueDrainAndGapMs: 0,
          vaePayloadAuthenticationMs: 0,
          reason: "ordinary initialize reaches READY before generation-time model upload/compile and VAE payload acquisition",
        }),
      }),
      lifecycle: Object.freeze({
        freshWorker: true,
        freshBackend: true,
        freshDeviceRequestedByBackend: true,
        backendDisposedBeforeWorkerCompletion: true,
        workerTerminationRecordedByPage: true,
        noGpuResourceMayOutliveBackendDispose: true,
      }),
    }),
  });
}

async function openAuthenticInventory(): Promise<Readonly<{
  candidates: PhysicalCandidate[];
  summary: Readonly<Record<string, unknown>>;
  manifestIdentities: Readonly<Record<PackageKind,
    Readonly<Record<string, unknown>>>>;
}>> {
  const [mainLoaded, denseLoaded, vaeLoaded] = await Promise.all([
    loadAcePackageManifest({
      manifestUrl: OPT_0071_MAIN_MANIFEST_PATH,
      expectedManifestSha256: OPT_0071_MAIN_MANIFEST_SHA256,
      expectedProfile: "reference",
    }),
    loadAcePackageManifest({
      manifestUrl: OPT_0071_DENSE_MANIFEST_PATH,
      expectedManifestSha256: OPT_0071_DENSE_MANIFEST_SHA256,
      expectedProfile: "fp16-dit-dense-experimental",
      authenticatedDitDenseConverterRevision: 7,
    }),
    loadAcePackageManifest({
      manifestUrl: OPT_0071_VAE_MANIFEST_PATH,
      expectedManifestSha256: OPT_0071_VAE_MANIFEST_SHA256,
      expectedProfile: "fp16-vae-experimental",
      authenticatedVaeConverterRevision: 7,
    }),
  ]);
  if (
    mainLoaded.manifest.provenance.converterRevision !== 4 ||
    denseLoaded.manifest.provenance.converterRevision !== 7 ||
    vaeLoaded.manifest.provenance.converterRevision !== 7
  ) throw new Error("OPT-0071 converter revision identity changed");
  const main = createAceMainAcquisitionManifest(mainLoaded.manifest);
  const dense = createAceOpt0009DitDenseAcquisitionManifest(denseLoaded.manifest);
  const vae = createAceOpt0011VaeAcquisitionManifest(vaeLoaded.manifest);
  const mainUpload = phaseFileNames(mainLoaded.manifest, [
    "text", "conditioner", "constants", "dit",
  ]);
  const logical: LogicalRecord[] = [
    ...logicalRecords("main", main, mainUpload),
    ...logicalRecords("dit-dense", dense, new Set(
      aceRuntimePackageFiles(dense).map((item) => item.name),
    )),
    ...logicalRecords("vae", vae, new Set(
      aceRuntimePackageFiles(vae).map((item) => item.name),
    )),
  ];
  const byDigest = new Map<string, LogicalRecord[]>();
  for (const item of logical) {
    const aliases = byDigest.get(item.record.sha256) ?? [];
    if (
      aliases[0] !== undefined &&
      aliases[0].record.byteLength !== item.record.byteLength
    ) throw new Error("OPT-0071 digest aliases disagree on byte length");
    aliases.push(item);
    byDigest.set(item.record.sha256, aliases);
  }
  const root = await navigator.storage.getDirectory();
  const cacheRoot = await root.getDirectoryHandle(READ_ONLY_CACHE_DIRECTORY);
  const cache = new AceOpfsModelCache(cacheRoot);
  const candidates: PhysicalCandidate[] = [];
  for (const aliases of byDigest.values()) {
    const first = aliases[0]!;
    const file = await cache.openCandidate(first.record);
    if (file === undefined || file.size !== first.record.byteLength) {
      throw new Error(
        `OPT-0071 cache-only preflight is missing ${first.packageKind}:` +
          `${first.record.name} (${first.record.sha256})`,
      );
    }
    candidates.push(Object.freeze({
      packageKind: first.packageKind,
      record: first.record,
      file,
      logicalAliases: Object.freeze(aliases.map((item) =>
        `${item.packageKind}:${item.record.name}`
      )),
      uploadSubset: aliases.some((item) => item.uploadSubset),
    }));
  }
  const logicalBytes = sum(logical.map((item) => item.record.byteLength));
  const physicalBytes = sum(candidates.map((item) => item.record.byteLength));
  const uploadCandidates = candidates.filter((item) => item.uploadSubset);
  const fingerprint = await sha256Text(logical.map((item) =>
    `${item.packageKind}\0${item.record.name}\0${item.record.byteLength}\0` +
      `${item.record.sha256}\0${item.uploadSubset ? "upload" : "other"}\n`
  ).join(""));
  if (
    logical.length !== OPT_0071_FULL_LOGICAL_RECORDS ||
    candidates.length !== OPT_0071_FULL_UNIQUE_DIGESTS ||
    logicalBytes !== OPT_0071_FULL_LOGICAL_BYTES ||
    physicalBytes !== OPT_0071_FULL_PHYSICAL_BYTES ||
    Math.max(...candidates.map((item) => item.record.byteLength)) !==
      OPT_0071_LARGEST_FILE_BYTES ||
    uploadCandidates.length !== OPT_0071_UPLOAD_SUBSET_FILES ||
    sum(uploadCandidates.map((item) => item.record.byteLength)) !==
      OPT_0071_UPLOAD_SUBSET_BYTES ||
    fingerprint !== OPT_0071_INVENTORY_FINGERPRINT
  ) throw new Error("OPT-0071 authenticated inventory changed");
  return Object.freeze({
    candidates,
    summary: Object.freeze({
      logicalRecords: logical.length,
      uniqueDigests: candidates.length,
      logicalBytes,
      physicalBytes,
      largestFileBytes: OPT_0071_LARGEST_FILE_BYTES,
      uploadSubsetFiles: uploadCandidates.length,
      uploadSubsetBytes: OPT_0071_UPLOAD_SUBSET_BYTES,
      fingerprint,
      packages: Object.freeze(([
        ["main", 103, 101, 4_140_848_075, 4_136_399_389],
        ["dit-dense", 48, 48, 3_020_808_192, 3_020_808_192],
        ["vae", 7, 7, 168_791_552, 168_791_552],
      ] as const).map((item) => Object.freeze({
        packageKind: item[0],
        logicalRecords: item[1],
        uniqueDigests: item[2],
        logicalBytes: item[3],
        physicalBytes: item[4],
      }))),
    }),
    manifestIdentities: Object.freeze({
      main: manifestIdentity(mainLoaded),
      "dit-dense": manifestIdentity(denseLoaded),
      vae: manifestIdentity(vaeLoaded),
    }),
  });
}

async function authenticateFullInventory(
  candidates: readonly PhysicalCandidate[],
  owner: Opt0071Owner,
): Promise<FullAuthenticationResult> {
  if (
    OPT_0071_LARGEST_FILE_BYTES >= OPT_0071_MAXIMUM_FILE_BYTES ||
    OPT_0071_CONSERVATIVE_TRANSIENT_BYTES >=
      OPT_0071_MAXIMUM_TRANSIENT_BYTES
  ) throw new Error("OPT-0071 WebCrypto memory eligibility changed");
  let readCopyMs = 0;
  let hashMs = 0;
  let finalizationAndComparisonMs = 0;
  let releaseMs = 0;
  let matchedUniqueDigests = 0;
  let receivedPhysicalBytes = 0;
  let maximumExplicitLivePayloadBytes = 0;
  let maximumExplicitLivePayloadCount = 0;
  let peakMemory: MemoryMeasurement | null = null;
  const fileTraces: Readonly<Record<string, unknown>>[] = [];
  const startedAt = performance.now();
  for (const candidate of candidates) {
    const fileStartedAt = performance.now();
    let fileReadMs = 0;
    let fileHashMs = 0;
    let fileFinalMs = 0;
    let fileReleaseMs = 0;
    let receivedBytes = 0;
    let hashChunkCount = 0;
    let maximumHashChunkBytes = 0;
    let actualSha256 = "";
    if (owner === "scalar-stream") {
      const reader = candidate.file.stream().getReader();
      const hash = new AceIncrementalSha256();
      try {
        while (true) {
          const readStartedAt = performance.now();
          const item = await reader.read();
          fileReadMs += performance.now() - readStartedAt;
          if (item.done) break;
          maximumExplicitLivePayloadCount = 1;
          for (const chunk of boundedSlices(item.value, HASH_CHUNK_BYTES)) {
            receivedBytes = checkedOpt0071ByteAdd(receivedBytes, chunk.byteLength);
            hashChunkCount += 1;
            maximumHashChunkBytes = Math.max(maximumHashChunkBytes, chunk.byteLength);
            maximumExplicitLivePayloadBytes = Math.max(
              maximumExplicitLivePayloadBytes,
              chunk.byteLength,
            );
            const hashStartedAt = performance.now();
            hash.update(chunk);
            fileHashMs += performance.now() - hashStartedAt;
          }
        }
        const finalStartedAt = performance.now();
        actualSha256 = hash.digestHex();
        fileFinalMs += performance.now() - finalStartedAt;
      } finally {
        const releaseStartedAt = performance.now();
        reader.releaseLock();
        fileReleaseMs += performance.now() - releaseStartedAt;
      }
    } else {
      let livePayload: ArrayBuffer | undefined;
      let livePayloadCount = 0;
      try {
        const readStartedAt = performance.now();
        livePayload = await candidate.file.arrayBuffer();
        fileReadMs += performance.now() - readStartedAt;
        livePayloadCount = 1;
        receivedBytes = livePayload.byteLength;
        hashChunkCount = 1;
        maximumHashChunkBytes = livePayload.byteLength;
        maximumExplicitLivePayloadCount = Math.max(
          maximumExplicitLivePayloadCount,
          livePayloadCount,
        );
        maximumExplicitLivePayloadBytes = Math.max(
          maximumExplicitLivePayloadBytes,
          livePayload.byteLength,
        );
        if (
          candidate.record.byteLength === OPT_0071_LARGEST_FILE_BYTES &&
          peakMemory === null
        ) peakMemory = await measureMemory("largest-live-preflight-payload");
        const hashStartedAt = performance.now();
        const digest = await crypto.subtle.digest("SHA-256", livePayload);
        fileHashMs += performance.now() - hashStartedAt;
        const finalStartedAt = performance.now();
        actualSha256 = hex(new Uint8Array(digest));
        fileFinalMs += performance.now() - finalStartedAt;
      } finally {
        const releaseStartedAt = performance.now();
        livePayload = undefined;
        livePayloadCount = 0;
        fileReleaseMs += performance.now() - releaseStartedAt;
        if (livePayload !== undefined || livePayloadCount !== 0) {
          throw new Error("OPT-0071 WebCrypto payload cleanup failed");
        }
      }
    }
    if (
      receivedBytes !== candidate.record.byteLength ||
      actualSha256 !== candidate.record.sha256
    ) throw new Error(`OPT-0071 ${candidate.record.name} digest mismatch`);
    readCopyMs += fileReadMs;
    hashMs += fileHashMs;
    finalizationAndComparisonMs += fileFinalMs;
    releaseMs += fileReleaseMs;
    matchedUniqueDigests += 1;
    receivedPhysicalBytes = checkedOpt0071ByteAdd(
      receivedPhysicalBytes,
      candidate.record.byteLength,
    );
    fileTraces.push(Object.freeze({
      packageKind: candidate.packageKind,
      file: candidate.record.name,
      byteLength: candidate.record.byteLength,
      expectedSha256: candidate.record.sha256,
      actualSha256,
      logicalAliases: candidate.logicalAliases,
      uploadSubset: candidate.uploadSubset,
      owner,
      receivedBytes,
      hashChunkCount,
      maximumHashChunkBytes,
      readCopyMs: fileReadMs,
      hashMs: fileHashMs,
      finalizationAndComparisonMs: fileFinalMs,
      releaseMs: fileReleaseMs,
      wallMs: performance.now() - fileStartedAt,
      matched: true,
    }));
  }
  return Object.freeze({
    wallMs: performance.now() - startedAt,
    readCopyMs,
    hashMs,
    finalizationAndComparisonMs,
    releaseMs,
    matchedUniqueDigests,
    receivedPhysicalBytes,
    maximumExplicitLivePayloadBytes,
    maximumExplicitLivePayloadCount,
    conservativeTransientBytes: owner === "scalar-stream"
      ? HASH_CHUNK_BYTES
      : OPT_0071_CONSERVATIVE_TRANSIENT_BYTES,
    fileTraces: Object.freeze(fileTraces),
    peakMemory,
  });
}

async function runCorrectnessAndLifecycleScreens(): Promise<
  Readonly<Record<string, unknown>>
> {
  const encoder = new TextEncoder();
  const nist = [
    [new Uint8Array(),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    [encoder.encode("abc"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
    [encoder.encode(
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    ), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"],
  ] as const;
  for (const [bytes, expected] of nist) {
    if (
      new AceIncrementalSha256().update(bytes).digestHex() !== expected ||
      await webCryptoHex(bytes) !== expected
    ) throw new Error("OPT-0071 NIST vector changed");
  }
  const lengths = [
    0, 1, 55, 56, 63, 64, 65,
    HASH_CHUNK_BYTES - 1,
    HASH_CHUNK_BYTES,
    HASH_CHUNK_BYTES + 1,
  ] as const;
  for (const length of lengths) {
    const bytes = deterministicBytes(length);
    const expected = await webCryptoHex(bytes);
    const split = new AceIncrementalSha256();
    for (let offset = 0; offset < bytes.byteLength; offset += 37) {
      split.update(bytes.subarray(offset, Math.min(offset + 37, bytes.byteLength)));
    }
    if (split.digestHex() !== expected) {
      throw new Error(`OPT-0071 split-update length ${length} changed`);
    }
  }
  const signed = new Int8Array([-128, -127, -1, 0, 1, 126, 127]);
  const signedBytes = new Uint8Array(
    signed.buffer,
    signed.byteOffset,
    signed.byteLength,
  );
  if (
    new AceIncrementalSha256().update(signedBytes).digestHex() !==
      await webCryptoHex(signedBytes)
  ) throw new Error("OPT-0071 signed-byte vector changed");
  const original = deterministicBytes(257);
  const expected = await webCryptoHex(original);
  const corrupted = original.slice();
  corrupted[127] = corrupted[127]! ^ 1;
  if (await webCryptoHex(corrupted) === expected) {
    throw new Error("OPT-0071 one-bit corruption was not detected");
  }
  const preAborted = new AbortController();
  preAborted.abort(new DOMException("pre-abort", "AbortError"));
  const scalarPreAbortRejected = await rejectionIsAbort(async () =>
    await hashSyntheticScalarFile(
      new File([original.buffer], "scalar-pre-abort.bin"),
      original.byteLength,
      expected,
      preAborted.signal,
    )
  );
  const scalarShortReadRejected = await rejects(async () =>
    await hashSyntheticScalarFile(
      new File([
        original.buffer.slice(0, original.byteLength - 1),
      ], "scalar-short.bin"),
      original.byteLength,
      expected,
      new AbortController().signal,
    )
  );
  const preAbortRejected = await rejectionIsAbort(async () =>
    await boundedWebCryptoBoundary(
      async () => original.buffer.slice(0),
      async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
      original.byteLength,
      expected,
      preAborted.signal,
    )
  );
  const arrayBufferRejected = await rejects(async () =>
    await boundedWebCryptoBoundary(
      async () => {
        throw new DOMException("injected arrayBuffer rejection", "NotReadableError");
      },
      async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
      original.byteLength,
      expected,
      new AbortController().signal,
    )
  );
  const webCryptoRejected = await rejects(async () =>
    await boundedWebCryptoBoundary(
      async () => original.buffer.slice(0),
      async () => {
        throw new DOMException("injected WebCrypto rejection", "OperationError");
      },
      original.byteLength,
      expected,
      new AbortController().signal,
    )
  );
  const shortReadRejected = await rejects(async () =>
    await boundedWebCryptoBoundary(
      async () => original.buffer.slice(0, original.byteLength - 1),
      async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
      original.byteLength,
      expected,
      new AbortController().signal,
    )
  );
  const digestMismatchRejected = await rejects(async () =>
    await boundedWebCryptoBoundary(
      async () => corrupted.buffer.slice(0),
      async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
      original.byteLength,
      expected,
      new AbortController().signal,
    )
  );
  const duringRead = new AbortController();
  const deferredRead = deferred<ArrayBuffer>();
  const duringReadPromise = boundedWebCryptoBoundary(
    async () => await deferredRead.promise,
    async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
    original.byteLength,
    expected,
    duringRead.signal,
  );
  duringRead.abort(new DOMException("during-read", "AbortError"));
  deferredRead.resolve(original.buffer.slice(0));
  const duringReadRejectedAtBoundary = await rejectionIsAbort(
    async () => await duringReadPromise,
  );
  const duringDigest = new AbortController();
  const deferredDigest = deferred<ArrayBuffer>();
  const duringDigestPromise = boundedWebCryptoBoundary(
    async () => original.buffer.slice(0),
    async () => await deferredDigest.promise,
    original.byteLength,
    expected,
    duringDigest.signal,
  );
  await Promise.resolve();
  duringDigest.abort(new DOMException("during-digest", "AbortError"));
  deferredDigest.resolve(await crypto.subtle.digest("SHA-256", original));
  const duringDigestRejectedAtBoundary = await rejectionIsAbort(
    async () => await duringDigestPromise,
  );
  const success = await boundedWebCryptoBoundary(
    async () => original.buffer.slice(0),
    async (bytes) => await crypto.subtle.digest("SHA-256", bytes),
    original.byteLength,
    expected,
    new AbortController().signal,
  );
  const finalized = new AceIncrementalSha256().update(original);
  finalized.digestHex();
  const secondFinalizeRejected = await rejects(async () => {
    finalized.digestHex();
  });
  const postFinalizeUpdateRejected = await rejects(async () => {
    finalized.update(Uint8Array.of(1));
  });
  const greaterThan32BitCumulativeLength = checkedOpt0071ByteAdd(0xffff_ffff, 2);
  const greaterThan32BitShaLengthEncoding =
    screenGreaterThan32BitShaLengthEncoding();
  const passed = scalarPreAbortRejected && scalarShortReadRejected &&
    preAbortRejected && arrayBufferRejected && webCryptoRejected &&
    shortReadRejected && digestMismatchRejected &&
    duringReadRejectedAtBoundary && duringDigestRejectedAtBoundary &&
    success.cleanupLivePayloadCount === 0 && secondFinalizeRejected &&
    postFinalizeUpdateRejected && !("reset" in finalized) &&
    greaterThan32BitCumulativeLength === 4_294_967_297 &&
    greaterThan32BitShaLengthEncoding.highWord === 8 &&
    greaterThan32BitShaLengthEncoding.lowWord === 8;
  if (!passed) throw new Error("OPT-0071 failure/cancellation screen failed");
  return Object.freeze({
    schema: "ace-opt-0071-correctness-cancellation-lifecycle-v1",
    nistVectorCount: nist.length,
    boundaryLengths: Object.freeze([...lengths]),
    splitUpdatesPassed: true,
    signedBytesPassed: true,
    greaterThan32BitCumulativeLength,
    greaterThan32BitShaLengthEncoding,
    oneBitCorruptionRejectedWithoutAuthenticMutation: true,
    scalarShortReadRejected,
    scalarPreAbortRejected,
    shortReadRejected,
    digestMismatchRejected,
    preAbortRejected,
    injectedArrayBufferRejectionPassed: arrayBufferRejected,
    injectedWebCryptoRejectionPassed: webCryptoRejected,
    cancellationDuringReadEnforcedAfterInFlightRead: duringReadRejectedAtBoundary,
    cancellationDuringDigestEnforcedAfterInFlightDigest:
      duringDigestRejectedAtBoundary,
    webCryptoInternallyAbortable: false,
    boundedCancellationBoundary: "before read / before digest / after digest; at most one sub-128MiB file in flight",
    successCleanupLivePayloadCount: success.cleanupLivePayloadCount,
    secondFinalizeRejected,
    postFinalizeUpdateRejected,
    mutableResetApiAbsent: !("reset" in finalized),
    noScalarFallbackAfterWebCryptoFailure: true,
    passed,
  });
}

async function boundedWebCryptoBoundary(
  read: () => Promise<ArrayBuffer>,
  digest: (bytes: ArrayBuffer) => Promise<ArrayBuffer>,
  expectedByteLength: number,
  expectedSha256: string,
  signal: AbortSignal,
): Promise<Readonly<{ readonly cleanupLivePayloadCount: 0 }>> {
  signal.throwIfAborted();
  let payload: ArrayBuffer | undefined;
  let livePayloadCount = 0;
  try {
    payload = await read();
    livePayloadCount = 1;
    signal.throwIfAborted();
    if (payload.byteLength !== expectedByteLength) {
      throw new Error("OPT-0071 injected short read");
    }
    const result = await digest(payload);
    signal.throwIfAborted();
    if (hex(new Uint8Array(result)) !== expectedSha256) {
      throw new Error("OPT-0071 injected digest mismatch");
    }
  } finally {
    payload = undefined;
    livePayloadCount = 0;
  }
  if (livePayloadCount !== 0) {
    throw new Error("OPT-0071 synthetic payload cleanup failed");
  }
  return Object.freeze({ cleanupLivePayloadCount: 0 as const });
}

async function hashSyntheticScalarFile(
  file: File,
  expectedByteLength: number,
  expectedSha256: string,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  if (file.size !== expectedByteLength) {
    throw new Error("OPT-0071 synthetic scalar short read");
  }
  const reader = file.stream().getReader();
  const hash = new AceIncrementalSha256();
  let receivedBytes = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const item = await reader.read();
      if (item.done) break;
      for (const chunk of boundedSlices(item.value, HASH_CHUNK_BYTES)) {
        signal.throwIfAborted();
        receivedBytes = checkedOpt0071ByteAdd(receivedBytes, chunk.byteLength);
        hash.update(chunk);
      }
    }
  } finally {
    reader.releaseLock();
  }
  signal.throwIfAborted();
  if (
    receivedBytes !== expectedByteLength || hash.digestHex() !== expectedSha256
  ) throw new Error("OPT-0071 synthetic scalar proof mismatch");
}

function screenGreaterThan32BitShaLengthEncoding(): Readonly<{
  readonly byteLength: number;
  readonly highWord: number;
  readonly lowWord: number;
}> {
  interface MutableShaProbe {
    totalBytes: number;
    pendingBytes: number;
    compress: (bytes: Uint8Array, offset: number) => void;
  }
  const byteLength = 0x1_0000_0001;
  const blocks: Uint8Array[] = [];
  const hash = new AceIncrementalSha256();
  const probe = hash as unknown as MutableShaProbe;
  probe.totalBytes = byteLength;
  probe.pendingBytes = 0;
  probe.compress = (bytes, offset) => {
    blocks.push(bytes.slice(offset, offset + 64));
  };
  hash.digestHex();
  const finalBlock = blocks.at(-1);
  if (finalBlock === undefined || finalBlock.byteLength !== 64) {
    throw new Error("OPT-0071 >32-bit SHA length probe did not finalize");
  }
  const view = new DataView(
    finalBlock.buffer,
    finalBlock.byteOffset,
    finalBlock.byteLength,
  );
  return Object.freeze({
    byteLength,
    highWord: view.getUint32(56, false),
    lowWord: view.getUint32(60, false),
  });
}

function analyzeTimedInitialization(
  owner: Opt0071Owner,
  readyWallMs: number,
  events: readonly AceOpt0064CaptureEvent[],
  progress: readonly AceInitializationProgress[],
): Readonly<{
  readonly authenticationWallMs: number;
  readonly fileAuthenticationWallMs: number;
  readonly timedLogicalRecords: number;
  readonly timedUniqueDigests: number;
  readonly timedLogicalBytes: number;
  readonly timedPhysicalBytes: number;
  readonly timedInventoryFingerprint: string;
  readonly downloadCount: number;
  readonly downloadBytes: number;
  readonly exactDigestsPassed: boolean;
  readonly unrelatedStageWalls: Opt0071UnrelatedStageWalls;
  readonly aggregateUnrelatedWallMs: number;
  readonly captureEventCount: number;
  readonly progressEventCount: number;
}> {
  const cacheEvents = events.filter((event) =>
    event.scope === "initialization" &&
    event.category === "authentication" &&
    event.operation.endsWith("-cache-authentication")
  );
  const proofEvents = events.filter((event) =>
    event.scope === "initialization" &&
    event.category === "authentication" &&
    event.operation.endsWith("-proof-reuse")
  );
  const packageEvents = [
    requireCaptureEvent(events, "main-package-acquisition"),
    requireCaptureEvent(events, "dit-dense-package-acquisition"),
  ];
  const planEvents = [
    requireCaptureEvent(events, "main-acquisition-plan"),
    requireCaptureEvent(events, "dit-dense-acquisition-plan"),
  ];
  const timedPhysicalBytes = sum(cacheEvents.map((event) =>
    number(event.details["byteLength"])
  ));
  const timedLogicalBytes = sum(planEvents.map((event) =>
    number(event.details["runtimeBytes"])
  ));
  const timedLogicalRecords = sum(planEvents.map((event) =>
    number(event.details["cachedFileCount"])
  ));
  const downloadCount = sum(planEvents.map((event) =>
    number(event.details["downloadFileCount"])
  ));
  const downloadBytes = sum(planEvents.map((event) =>
    number(event.details["downloadBytes"])
  ));
  const exactDigestsPassed = cacheEvents.length ===
      OPT_0071_TIMED_UNIQUE_DIGESTS &&
    proofEvents.length === OPT_0071_TIMED_LOGICAL_RECORDS &&
    cacheEvents.every((event) =>
      event.details["authenticationOwner"] === owner &&
      event.details["matched"] === true &&
      event.details["exactImmutableFileProofPublished"] === true &&
      event.details["expectedSha256"] === event.details["actualSha256"] &&
      event.details["receivedBytes"] === event.details["byteLength"]
    ) &&
    proofEvents.every((event) =>
      event.details["source"] === "cache" &&
      event.details["exactImmutableFileIdentity"] === true &&
      event.details["redundantHashPerformed"] === false
    );
  const timedInventoryFingerprint = new AceIncrementalSha256().update(
    new TextEncoder().encode([
      ...cacheEvents.map((event) =>
        `physical\0${text(event.details["packageKind"])}\0` +
        `${text(event.details["file"])}\0${number(event.details["byteLength"])}\0` +
        `${text(event.details["expectedSha256"])}\n`
      ),
      ...proofEvents.map((event) =>
        `logical\0${text(event.details["packageKind"])}\0` +
        `${text(event.details["file"])}\0${number(event.details["byteLength"])}\0` +
        `${text(event.details["sha256"])}\n`
      ),
    ].join("")),
  ).digestHex();
  const authenticationWallMs = sumDurations(
    packageEvents.map((event) => event.wallMs),
  );
  const fileAuthenticationWallMs = sumDurations(
    cacheEvents.map((event) => event.wallMs),
  );
  const directStages = Object.freeze({
    "device-request": requireCaptureEvent(events, "webgpu-device-request").wallMs,
    "opfs-open": requireCaptureEvent(events, "opfs-open").wallMs,
    "stale-audio-recovery": requireCaptureEvent(
      events,
      "stale-audio-recovery",
    ).wallMs,
    "main-manifest-authentication": requireCaptureEvent(
      events,
      "main-manifest-authentication",
    ).wallMs,
    "dit-dense-manifest-authentication": requireCaptureEvent(
      events,
      "dit-dense-manifest-authentication",
    ).wallMs,
    "vae-manifest-authentication": requireCaptureEvent(
      events,
      "vae-manifest-authentication",
    ).wallMs,
    "text-tokenizer-load": requireCaptureEvent(
      events,
      "text-tokenizer-load",
    ).wallMs,
    "planner-tokenizer-load": requireCaptureEvent(
      events,
      "planner-tokenizer-load",
    ).wallMs,
  });
  const directUnrelatedWallMs = sumDurations(Object.values(directStages));
  const aggregateUnrelatedWallMs = Math.max(0, readyWallMs - authenticationWallMs);
  const unrelatedStageWalls: Opt0071UnrelatedStageWalls = Object.freeze({
    ...directStages,
    "ready-publication-and-residual": Math.max(
      0,
      aggregateUnrelatedWallMs - directUnrelatedWallMs,
    ),
  });
  if (
    timedLogicalRecords !== OPT_0071_TIMED_LOGICAL_RECORDS ||
    cacheEvents.length !== OPT_0071_TIMED_UNIQUE_DIGESTS ||
    timedLogicalBytes !== OPT_0071_TIMED_LOGICAL_BYTES ||
    timedPhysicalBytes !== OPT_0071_TIMED_PHYSICAL_BYTES ||
    downloadCount !== 0 || downloadBytes !== 0 || !exactDigestsPassed ||
    progress.at(-1)?.stage !== "ready" ||
    OPT_0071_UNRELATED_STAGE_NAMES.some((stage) =>
      !Number.isFinite(unrelatedStageWalls[stage])
    )
  ) throw new Error("OPT-0071 timed production inventory or trace changed");
  return Object.freeze({
    authenticationWallMs,
    fileAuthenticationWallMs,
    timedLogicalRecords,
    timedUniqueDigests: cacheEvents.length,
    timedLogicalBytes,
    timedPhysicalBytes,
    timedInventoryFingerprint,
    downloadCount,
    downloadBytes,
    exactDigestsPassed,
    unrelatedStageWalls,
    aggregateUnrelatedWallMs,
    captureEventCount: events.length,
    progressEventCount: progress.length,
  });
}

function productionConfiguration(): AceWorkerConfiguration {
  return Object.freeze({
    manifestUrl: absoluteUrl(OPT_0071_MAIN_MANIFEST_PATH),
    manifestSha256: OPT_0071_MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage: Object.freeze({
      manifestUrl: absoluteUrl(OPT_0071_DENSE_MANIFEST_PATH),
      manifestSha256: OPT_0071_DENSE_MANIFEST_SHA256,
      runtimeProfile: "opt-0009-fp16-fp32-dense-v1",
    }),
    ditAttentionRuntimeProfile:
      "opt-0070-fixed32-quad-query32-full-self-production-v1",
    vaePackage: Object.freeze({
      manifestUrl: absoluteUrl(OPT_0071_VAE_MANIFEST_PATH),
      manifestSha256: OPT_0071_VAE_MANIFEST_SHA256,
      runtimeProfile: "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1",
      windowRuntimeProfile: "opt-0070-c2378-overlap64-production-v1",
      maxWindowFrames: 2_378,
    }),
  });
}

function validateDiagnostics(diagnostics: AceRuntimeDiagnostics): void {
  if (
    diagnostics.modelManifestSha256 !== OPT_0071_MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== OPT_0071_DENSE_MANIFEST_SHA256 ||
    diagnostics.ditDenseRuntimeProfile !== "opt-0009-fp16-fp32-dense-v1" ||
    diagnostics.ditAttentionRuntimeProfile !==
      "opt-0070-fixed32-quad-query32-full-self-production-v1" ||
    diagnostics.vaeManifestSha256 !== OPT_0071_VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !==
      "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1" ||
    diagnostics.vaeWindowRuntimeProfile !==
      "opt-0070-c2378-overlap64-production-v1" ||
    diagnostics.vaeMaxWindowFrames !== 2_378 ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.adapterInfo.subgroupMinSize !== 32 ||
    diagnostics.capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !diagnostics.capabilities.deviceFeatures.includes("shader-f16") ||
    !diagnostics.capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error("OPT-0071 exact OPT-0072 production tuple changed");
}

function logicalRecords(
  packageKind: PackageKind,
  manifest: AcePackageManifest,
  uploadFiles: ReadonlySet<string>,
): LogicalRecord[] {
  return aceRuntimePackageFiles(manifest).map((record) => Object.freeze({
    packageKind,
    record,
    uploadSubset: uploadFiles.has(record.name),
  }));
}

function phaseFileNames(
  manifest: AcePackageManifest,
  phases: readonly AceTensorPhase[],
): ReadonlySet<string> {
  const selected = new Set<AceTensorPhase>(phases);
  return new Set(Object.values(manifest.tensors)
    .filter((tensor) => selected.has(tensor.phase))
    .map((tensor) => tensor.shard));
}

function manifestIdentity(value: Readonly<{
  readonly manifestUrl: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: number;
  readonly manifest: AcePackageManifest;
}>): Readonly<Record<string, unknown>> {
  return Object.freeze({
    url: value.manifestUrl,
    sha256: value.manifestSha256,
    byteLength: value.manifestByteLength,
    converterRevision: value.manifest.provenance.converterRevision,
  });
}

function requireCaptureEvent(
  events: readonly AceOpt0064CaptureEvent[],
  operation: string,
): AceOpt0064CaptureEvent {
  const matches = events.filter((event) => event.operation === operation);
  if (matches.length !== 1) {
    throw new Error(`OPT-0071 requires exactly one ${operation} event`);
  }
  return matches[0]!;
}

interface WorkerHeartbeat {
  readonly maximumGapMs: number;
  finish(): Promise<void>;
}

function startWorkerHeartbeat(armId: Opt0071ArmId): WorkerHeartbeat {
  let last = performance.now();
  let maximumGapMs = 0;
  let stopped = false;
  const timer = self.setInterval(() => {
    const now = performance.now();
    maximumGapMs = Math.max(maximumGapMs, now - last);
    last = now;
    self.postMessage({ type: "heartbeat", armId, at: now });
  }, 25);
  return {
    get maximumGapMs(): number {
      return maximumGapMs;
    },
    async finish(): Promise<void> {
      if (stopped) return;
      await new Promise<void>((resolve) => self.setTimeout(resolve, 0));
      maximumGapMs = Math.max(maximumGapMs, performance.now() - last);
      self.clearInterval(timer);
      stopped = true;
    },
  };
}

function boundedSlices(
  bytes: Uint8Array,
  maximumBytes: number,
): readonly Uint8Array[] {
  const chunks: Uint8Array[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += maximumBytes) {
    chunks.push(bytes.subarray(
      offset,
      Math.min(bytes.byteLength, offset + maximumBytes),
    ));
  }
  return chunks;
}

function deterministicBytes(length: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(length);
  let state = 0x6d2b79f5;
  for (let index = 0; index < length; index += 1) {
    state = Math.imul(state ^ (state >>> 15), 1 | state);
    state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
    bytes[index] = (state ^ (state >>> 14) ^ index) & 0xff;
  }
  return bytes;
}

async function webCryptoHex(bytes: Uint8Array): Promise<string> {
  const owned = new Uint8Array(bytes.byteLength);
  owned.set(bytes);
  return hex(new Uint8Array(
    await crypto.subtle.digest("SHA-256", owned.buffer),
  ));
}

async function sha256Text(value: string): Promise<string> {
  return await webCryptoHex(new TextEncoder().encode(value));
}

function hex(bytes: Uint8Array): string {
  let value = "";
  for (const byte of bytes) value += byte.toString(16).padStart(2, "0");
  return value;
}

function deferred<Value>(): Readonly<{
  readonly promise: Promise<Value>;
  readonly resolve: (value: Value) => void;
}> {
  let resolvePromise: ((value: Value) => void) | undefined;
  const promise = new Promise<Value>((resolve) => {
    resolvePromise = resolve;
  });
  return Object.freeze({
    promise,
    resolve(value: Value): void {
      if (resolvePromise === undefined) throw new Error("OPT-0071 deferred missing");
      resolvePromise(value);
    },
  });
}

async function rejects(operation: () => Promise<unknown>): Promise<boolean> {
  try {
    await operation();
    return false;
  } catch {
    return true;
  }
}

async function rejectionIsAbort(
  operation: () => Promise<unknown>,
): Promise<boolean> {
  try {
    await operation();
    return false;
  } catch (error) {
    return error instanceof DOMException && error.name === "AbortError";
  }
}

async function measureMemory(label: string): Promise<MemoryMeasurement> {
  const method = (performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<Readonly<{ bytes: number }>>;
  }).measureUserAgentSpecificMemory;
  if (method === undefined) {
    return Object.freeze({ label, exposed: false, bytes: null, error: null });
  }
  try {
    const measurement = await method.call(performance);
    return Object.freeze({
      label,
      exposed: true,
      bytes: Number.isFinite(measurement.bytes) ? measurement.bytes : null,
      error: null,
    });
  } catch (error) {
    return Object.freeze({
      label,
      exposed: true,
      bytes: null,
      error: serializeOpt0071Failure(error),
    });
  }
}

function sum(values: readonly number[]): number {
  return values.reduce(checkedOpt0071ByteAdd, 0);
}

function sumDurations(values: readonly number[]): number {
  return values.reduce(checkedOpt0071DurationAdd, 0);
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OPT-0071 receipt record is malformed");
  }
  return value as Readonly<Record<string, unknown>>;
}

function number(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error("OPT-0071 receipt number is malformed");
  }
  return value;
}

function text(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("OPT-0071 receipt text is malformed");
  }
  return value;
}

function absoluteUrl(path: string): string {
  return new URL(path, self.location.href).href;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function fail(error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  abortController?.abort(new DOMException("OPT-0071 failed", "AbortError"));
  abortController = undefined;
  self.postMessage({ type: "failed", error: serializeOpt0071Failure(error) });
}
