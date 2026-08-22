/// <reference lib="webworker" />
/// <reference types="vite/client" />

import { aceRuntimePackageFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import type {
  AcePackageFileRecord,
  AcePackageManifest,
  AceTensorPhase,
} from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import {
  createAceMainAcquisitionManifest,
  createAceOpt0009DitDenseAcquisitionManifest,
  createAceOpt0011VaeAcquisitionManifest,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  checkedOpt0069ByteAdd,
  OPT_0069_ARM_ORDER,
  OPT_0069_COMPLETE_LOGICAL_BYTES,
  OPT_0069_COMPLETE_LOGICAL_RECORDS,
  OPT_0069_COMPLETE_PHYSICAL_BYTES,
  OPT_0069_COMPLETE_UNIQUE_DIGESTS,
  OPT_0069_DENSE_MANIFEST_PATH,
  OPT_0069_DENSE_MANIFEST_SHA256,
  OPT_0069_HASH_CHUNK_BYTES,
  OPT_0069_LARGEST_FILE_BYTES,
  OPT_0069_MAIN_MANIFEST_PATH,
  OPT_0069_MAIN_MANIFEST_SHA256,
  OPT_0069_UPLOAD_SUBSET_BYTES,
  OPT_0069_UPLOAD_SUBSET_FILES,
  OPT_0069_VAE_MANIFEST_PATH,
  OPT_0069_VAE_MANIFEST_SHA256,
  OPT_0069_WEBCRYPTO_MAXIMUM_FILE_BYTES,
  OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES,
  OPT_0069_WEBCRYPTO_TRANSIENT_MULTIPLIER,
  requireOpt0069Inventory,
  requireOpt0069ThermalGate,
  serializeOpt0069Failure,
  validateOpt0069RunIdentity,
  type Opt0069ArmId,
  type Opt0069Inventory,
  type Opt0069Owner,
  type Opt0069PackageInventory,
  type Opt0069RunIdentity,
  type Opt0069ThermalGate,
  type Opt0069TimingSample,
} from "./opt-0069-cache-authentication-hash-contract.js";

type PackageKind = "main" | "dit-dense" | "vae";
const READ_ONLY_CACHE_DIRECTORY = "ace-step-1.5.wgsl-model-cache-v1" as const;

interface PrepareMessage {
  readonly type: "prepare";
  readonly identity: Opt0069RunIdentity;
  readonly armId: Opt0069ArmId;
}

interface RunMessage {
  readonly type: "run";
  readonly armId: Opt0069ArmId;
  readonly thermalGate: Opt0069ThermalGate;
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
  readonly uploadSubset: boolean;
  readonly logicalAliases: readonly string[];
}

interface PreparedInventory {
  readonly candidates: PhysicalCandidate[];
  readonly inventory: Opt0069Inventory;
  readonly fingerprint: string;
  readonly manifestIdentities: Readonly<Record<PackageKind, Readonly<{
    readonly url: string;
    readonly sha256: string;
    readonly byteLength: number;
    readonly converterRevision: number;
  }>>>;
}

interface MemoryMeasurement {
  readonly label: string;
  readonly exposed: boolean;
  readonly bytes: number | null;
  readonly error: Readonly<Record<string, unknown>> | null;
}

let state: "idle" | "preparing" | "ready" | "running" | "settled" = "idle";
let identity: Opt0069RunIdentity | undefined;
let definition: (typeof OPT_0069_ARM_ORDER)[number] | undefined;
let prepared: PreparedInventory | undefined;
let correctness: Readonly<Record<string, unknown>> | undefined;
let readyAtEpochMilliseconds = 0;
let abortController: AbortController | undefined;
let memoryBefore: MemoryMeasurement | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    abortController?.abort(new DOMException("OPT-0069 cancelled", "AbortError"));
    return;
  }
  if (message.type === "prepare" && state === "idle") {
    state = "preparing";
    void prepare(message);
    return;
  }
  if (message.type === "run" && state === "ready") {
    state = "running";
    void run(message);
    return;
  }
  fail(new Error(`OPT-0069 rejected ${message.type} while ${state}`));
});

async function prepare(message: PrepareMessage): Promise<void> {
  try {
    identity = validateOpt0069RunIdentity(message.identity);
    definition = OPT_0069_ARM_ORDER.find((item) => item.armId === message.armId);
    if (definition === undefined) throw new Error("OPT-0069 arm identity changed");
    postProgress(`${message.armId}: authenticating the three manifest trust roots`);
    prepared = await prepareAuthenticInventory();
    postProgress(`${message.armId}: running untimed independent SHA-256 screens`);
    correctness = await runCorrectnessScreens(prepared.inventory);
    memoryBefore = await measureMemory("after-preparation-before-thermal");
    readyAtEpochMilliseconds = Date.now();
    state = "ready";
    self.postMessage({
      type: "ready-for-arm",
      armId: definition.armId,
      owner: definition.owner,
      order: definition.order,
      readyAtEpochMilliseconds,
      preparation: Object.freeze({
        schema: "ace-opt-0069-authentic-cache-preparation-v1",
        inventory: prepared.inventory,
        inventoryFingerprint: prepared.fingerprint,
        manifestIdentities: prepared.manifestIdentities,
        correctness,
        memoryBefore,
        authenticPayloadBytesReadBeforeThermalGate: 0,
        markerQualifiedOpfsFilesOpened: OPT_0069_COMPLETE_UNIQUE_DIGESTS,
        cacheMutationPerformed: false,
        gpuDeviceRequested: false,
        wasmArmIncluded: false,
        wasmArmDisposition:
          "deferred: no independently proven dependency-free four-lane SIMD owner",
      }),
    });
  } catch (error) {
    fail(error);
  }
}

async function run(message: RunMessage): Promise<void> {
  try {
    if (
      prepared === undefined || definition === undefined || identity === undefined ||
      correctness === undefined || message.armId !== definition.armId
    ) throw new Error("OPT-0069 run was not prepared for this arm");
    const thermalGate = requireOpt0069ThermalGate(
      message.thermalGate,
      readyAtEpochMilliseconds,
      Date.now(),
    );
    abortController = new AbortController();
    const heartbeat = startWorkerHeartbeat(definition.armId);
    postProgress(`${definition.armId}: hashing all 7,325,999,133 physical bytes`);
    const sample = definition.owner === "scalar-stream"
      ? await runScalarArm(prepared, definition.armId, definition.order,
        abortController.signal, heartbeat)
      : await runWebCryptoArm(prepared, definition.armId, definition.order,
        abortController.signal, heartbeat);
    const memoryAfter = await measureMemory("after-timed-payload-release");
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    prepared = undefined;
    abortController = undefined;
    state = "settled";
    self.postMessage({
      type: "arm-complete",
      armId: definition.armId,
      owner: definition.owner,
      order: definition.order,
      cleanupCompletedAtEpochMilliseconds,
      sample,
      receipt: Object.freeze({
        schema: "ace-opt-0069-arm-receipt-v1",
        experimentId: "OPT-0069",
        identity,
        thermalGate,
        inventoryFingerprint: sample.inventoryFingerprint,
        correctness,
        memoryTelemetry: Object.freeze({ before: memoryBefore, after: memoryAfter }),
        cancellation: Object.freeze({
          scalarCheckedBetweenEveryAtMost4MiBSlice: true,
          webCryptoDigestInternallyAbortable: false,
          webCryptoCancellationBoundary: "after-in-flight-file-digest",
        }),
        timingIncludes: Object.freeze([
          "opfs-payload-read",
          "whole-file-copy-when-webcrypto",
          "sha256",
          "finalization",
          "exact-digest-comparison",
          "explicit-payload-release",
        ]),
        noGpuDeviceRequested: true,
        noCacheMutationPerformed: true,
      }),
    });
  } catch (error) {
    prepared = undefined;
    abortController = undefined;
    fail(error);
  }
}

async function prepareAuthenticInventory(): Promise<PreparedInventory> {
  const [mainLoaded, denseLoaded, vaeLoaded] = await Promise.all([
    loadAcePackageManifest({
      manifestUrl: OPT_0069_MAIN_MANIFEST_PATH,
      expectedManifestSha256: OPT_0069_MAIN_MANIFEST_SHA256,
      expectedProfile: "reference",
    }),
    loadAcePackageManifest({
      manifestUrl: OPT_0069_DENSE_MANIFEST_PATH,
      expectedManifestSha256: OPT_0069_DENSE_MANIFEST_SHA256,
      expectedProfile: "fp16-dit-dense-experimental",
      authenticatedDitDenseConverterRevision: 7,
    }),
    loadAcePackageManifest({
      manifestUrl: OPT_0069_VAE_MANIFEST_PATH,
      expectedManifestSha256: OPT_0069_VAE_MANIFEST_SHA256,
      expectedProfile: "fp16-vae-experimental",
    }),
  ]);
  if (
    mainLoaded.manifest.provenance.converterRevision !== 4 ||
    denseLoaded.manifest.provenance.converterRevision !== 7 ||
    vaeLoaded.manifest.provenance.converterRevision !== 6
  ) throw new Error("OPT-0069 converter revision identity changed");
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
  const packageInventories = (["main", "dit-dense", "vae"] as const).map(
    (packageKind) => packageInventory(
      packageKind,
      logical.filter((item) => item.packageKind === packageKind),
    ),
  );
  const logicalByDigest = new Map<string, LogicalRecord[]>();
  for (const item of logical) {
    const aliases = logicalByDigest.get(item.record.sha256) ?? [];
    const byteLength = aliases[0]?.record.byteLength;
    if (byteLength !== undefined && byteLength !== item.record.byteLength) {
      throw new Error("OPT-0069 digest aliases disagree on byte length");
    }
    aliases.push(item);
    logicalByDigest.set(item.record.sha256, aliases);
  }
  const opfsRoot = await navigator.storage.getDirectory();
  const cacheRoot = await opfsRoot.getDirectoryHandle(READ_ONLY_CACHE_DIRECTORY);
  const cache = new AceOpfsModelCache(cacheRoot);
  const candidates: PhysicalCandidate[] = [];
  for (const aliases of logicalByDigest.values()) {
    const first = aliases[0]!;
    const file = await cache.openCandidate(first.record);
    if (file === undefined || file.size !== first.record.byteLength) {
      throw new Error(
        `OPT-0069 cache-only preflight is missing ${first.packageKind}:` +
          `${first.record.name} (${first.record.sha256})`,
      );
    }
    candidates.push(Object.freeze({
      packageKind: first.packageKind,
      record: first.record,
      file,
      uploadSubset: aliases.some((item) => item.uploadSubset),
      logicalAliases: Object.freeze(aliases.map((item) =>
        `${item.packageKind}:${item.record.name}`
      )),
    }));
  }
  const inventory = requireOpt0069Inventory(Object.freeze({
    packages: Object.freeze(packageInventories),
    logicalRecords: logical.length,
    uniqueDigests: candidates.length,
    logicalBytes: sumBytes(logical.map((item) => item.record.byteLength)),
    physicalBytes: sumBytes(candidates.map((item) => item.record.byteLength)),
    largestFileBytes: Math.max(...candidates.map((item) => item.record.byteLength)),
    uploadSubsetFiles: candidates.filter((item) => item.uploadSubset).length,
    uploadSubsetBytes: sumBytes(candidates.filter((item) => item.uploadSubset)
      .map((item) => item.record.byteLength)),
  }));
  const fingerprint = await sha256Text(logical.map((item) =>
    `${item.packageKind}\0${item.record.name}\0${item.record.byteLength}\0` +
      `${item.record.sha256}\0${item.uploadSubset ? "upload" : "other"}\n`
  ).join(""));
  return Object.freeze({
    candidates,
    inventory,
    fingerprint,
    manifestIdentities: Object.freeze({
      main: manifestIdentity(mainLoaded),
      "dit-dense": manifestIdentity(denseLoaded),
      vae: manifestIdentity(vaeLoaded),
    }),
  });
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

function packageInventory(
  packageKind: PackageKind,
  logical: readonly LogicalRecord[],
): Opt0069PackageInventory {
  const unique = new Map<string, number>();
  for (const item of logical) unique.set(item.record.sha256, item.record.byteLength);
  return Object.freeze({
    packageKind,
    logicalRecords: logical.length,
    uniqueDigests: unique.size,
    logicalBytes: sumBytes(logical.map((item) => item.record.byteLength)),
    physicalBytes: sumBytes([...unique.values()]),
  });
}

async function runCorrectnessScreens(
  inventory: Opt0069Inventory,
): Promise<Readonly<Record<string, unknown>>> {
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
    if (new AceIncrementalSha256().update(bytes).digestHex() !== expected) {
      throw new Error("OPT-0069 scalar NIST vector changed");
    }
    if (await webCryptoHex(bytes) !== expected) {
      throw new Error("OPT-0069 WebCrypto NIST vector changed");
    }
  }
  const lengths = [
    0, 1, 55, 56, 63, 64, 65,
    OPT_0069_HASH_CHUNK_BYTES - 1,
    OPT_0069_HASH_CHUNK_BYTES,
    OPT_0069_HASH_CHUNK_BYTES + 1,
  ] as const;
  for (const length of lengths) {
    const bytes = deterministicBytes(length);
    const web = await webCryptoHex(bytes);
    const scalar = new AceIncrementalSha256().update(bytes).digestHex();
    if (scalar !== web) throw new Error(`OPT-0069 length ${length} changed`);
    const split = new AceIncrementalSha256();
    let offset = 0;
    let step = 1;
    while (offset < bytes.byteLength) {
      const end = Math.min(bytes.byteLength, offset + step);
      split.update(bytes.subarray(offset, end));
      offset = end;
      step = step === 1 ? 55 : step === 55 ? 64 : 4_093;
    }
    if (split.digestHex() !== web) {
      throw new Error(`OPT-0069 split-update length ${length} changed`);
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
  ) throw new Error("OPT-0069 signed-byte vector changed");
  const original = deterministicBytes(257);
  const expected = await webCryptoHex(original);
  const corrupted = original.slice();
  corrupted[127] = corrupted[127]! ^ 1;
  if (await webCryptoHex(corrupted) === expected) {
    throw new Error("OPT-0069 one-bit corruption was not detected");
  }
  let shortReadRejected = false;
  try {
    await hashSyntheticScalarFile(
      new File([
        original.buffer.slice(0, original.byteLength - 1),
      ], "short.bin"),
      original.byteLength,
      expected,
      new AbortController().signal,
    );
  } catch {
    shortReadRejected = true;
  }
  if (!shortReadRejected) throw new Error("OPT-0069 short read was accepted");
  const aborted = new AbortController();
  aborted.abort(new DOMException("screen", "AbortError"));
  let abortRejected = false;
  try {
    await hashSyntheticScalarFile(
      new File([original.buffer], "aborted.bin"),
      original.byteLength,
      expected,
      aborted.signal,
    );
  } catch (error) {
    abortRejected = error instanceof DOMException && error.name === "AbortError";
  }
  if (!abortRejected) throw new Error("OPT-0069 abort did not fail closed");
  let webCryptoAbortRejected = false;
  try {
    await hashSyntheticWebCryptoFile(
      new File([original.buffer], "webcrypto-aborted.bin"),
      original.byteLength,
      expected,
      aborted.signal,
    );
  } catch (error) {
    webCryptoAbortRejected = error instanceof DOMException &&
      error.name === "AbortError";
  }
  if (!webCryptoAbortRejected) {
    throw new Error("OPT-0069 WebCrypto boundary abort did not fail closed");
  }
  const finalized = new AceIncrementalSha256().update(original);
  finalized.digestHex();
  let secondFinalizeRejected = false;
  let postFinalizeUpdateRejected = false;
  try {
    finalized.digestHex();
  } catch {
    secondFinalizeRejected = true;
  }
  try {
    finalized.update(Uint8Array.of(1));
  } catch {
    postFinalizeUpdateRejected = true;
  }
  if (
    !secondFinalizeRejected || !postFinalizeUpdateRejected ||
    "reset" in finalized
  ) throw new Error("OPT-0069 finalize/reset misuse did not fail closed");
  const cumulativeAbove32Bits = checkedOpt0069ByteAdd(0xffff_ffff, 2);
  const lengthEncoding = screenGreaterThan32BitShaLengthEncoding();
  if (
    cumulativeAbove32Bits !== 4_294_967_297 ||
    lengthEncoding.highWord !== 8 || lengthEncoding.lowWord !== 8 ||
    inventory.physicalBytes <= 0xffff_ffff ||
    inventory.logicalBytes <= 0xffff_ffff
  ) throw new Error("OPT-0069 >32-bit cumulative byte accounting changed");
  return Object.freeze({
    schema: "ace-opt-0069-independent-correctness-v1",
    nistVectorCount: nist.length,
    boundaryLengths: Object.freeze([...lengths]),
    splitUpdatesPassed: true,
    signedBytesPassed: true,
    greaterThan32BitCumulativeLength: cumulativeAbove32Bits,
    greaterThan32BitShaLengthEncoding: lengthEncoding,
    authenticPhysicalCumulativeLength: inventory.physicalBytes,
    authenticLogicalCumulativeLength: inventory.logicalBytes,
    corruptionRejectedWithoutAuthenticFileMutation: true,
    shortReadRejected: true,
    preAbortedReadRejected: true,
    preAbortedWebCryptoBoundaryRejected: true,
    secondFinalizeRejected: true,
    postFinalizeUpdateRejected: true,
    mutableResetApiAbsent: true,
    passed: true,
  });
}

async function runScalarArm(
  value: PreparedInventory,
  armId: Opt0069ArmId,
  order: number,
  signal: AbortSignal,
  heartbeat: WorkerHeartbeat,
): Promise<Opt0069TimingSample> {
  let candidates: readonly PhysicalCandidate[] = value.candidates;
  let readCopyMs = 0;
  let hashMs = 0;
  let finalizationAndComparisonMs = 0;
  let cleanupMs = 0;
  let uploadSubsetWallMs = 0;
  let matched = 0;
  let physicalBytes = 0;
  let maximumLivePayloadBytes = 0;
  let maximumLivePayloadCount = 0;
  const startedAt = performance.now();
  for (const candidate of candidates) {
    signal.throwIfAborted();
    const fileStartedAt = performance.now();
    if (candidate.file.size !== candidate.record.byteLength) {
      throw new Error(`OPT-0069 ${candidate.record.name} changed size`);
    }
    const reader = candidate.file.stream().getReader();
    const hash = new AceIncrementalSha256();
    let receivedBytes = 0;
    try {
      while (true) {
        signal.throwIfAborted();
        const readStartedAt = performance.now();
        const item = await reader.read();
        readCopyMs += performance.now() - readStartedAt;
        if (item.done) break;
        maximumLivePayloadCount = 1;
        for (const chunk of boundedSlices(item.value, OPT_0069_HASH_CHUNK_BYTES)) {
          signal.throwIfAborted();
          maximumLivePayloadBytes = Math.max(
            maximumLivePayloadBytes,
            chunk.byteLength,
          );
          receivedBytes = checkedOpt0069ByteAdd(receivedBytes, chunk.byteLength);
          const hashStartedAt = performance.now();
          hash.update(chunk);
          hashMs += performance.now() - hashStartedAt;
        }
      }
      if (receivedBytes !== candidate.record.byteLength) {
        throw new Error(`OPT-0069 ${candidate.record.name} short read`);
      }
      const finalStartedAt = performance.now();
      const actual = hash.digestHex();
      if (actual !== candidate.record.sha256) {
        throw new Error(`OPT-0069 ${candidate.record.name} digest mismatch`);
      }
      finalizationAndComparisonMs += performance.now() - finalStartedAt;
      matched += 1;
      physicalBytes = checkedOpt0069ByteAdd(
        physicalBytes,
        candidate.record.byteLength,
      );
    } finally {
      const cleanupStartedAt = performance.now();
      reader.releaseLock();
      cleanupMs += performance.now() - cleanupStartedAt;
    }
    if (candidate.uploadSubset) {
      uploadSubsetWallMs += performance.now() - fileStartedAt;
    }
  }
  const releaseStartedAt = performance.now();
  candidates = [];
  value.candidates.length = 0;
  cleanupMs += performance.now() - releaseStartedAt;
  await heartbeat.finish();
  const wallMs = performance.now() - startedAt;
  return timingSample({
    armId,
    order,
    owner: "scalar-stream",
    wallMs,
    readCopyMs,
    hashMs,
    finalizationAndComparisonMs,
    cleanupMs,
    uploadSubsetWallMs,
    matched,
    physicalBytes,
    maximumLivePayloadBytes,
    maximumLivePayloadCount,
    conservativeTransientBytes: OPT_0069_HASH_CHUNK_BYTES,
    maximumWorkerHeartbeatGapMs: heartbeat.maximumGapMs,
    fingerprint: value.fingerprint,
  });
}

async function runWebCryptoArm(
  value: PreparedInventory,
  armId: Opt0069ArmId,
  order: number,
  signal: AbortSignal,
  heartbeat: WorkerHeartbeat,
): Promise<Opt0069TimingSample> {
  if (
    value.inventory.largestFileBytes >= OPT_0069_WEBCRYPTO_MAXIMUM_FILE_BYTES ||
    value.inventory.largestFileBytes * OPT_0069_WEBCRYPTO_TRANSIENT_MULTIPLIER >=
      OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES
  ) throw new Error("OPT-0069 WebCrypto memory eligibility changed");
  let candidates: readonly PhysicalCandidate[] = value.candidates;
  let livePayload: ArrayBuffer | undefined;
  let livePayloadCount = 0;
  let readCopyMs = 0;
  let hashMs = 0;
  let finalizationAndComparisonMs = 0;
  let cleanupMs = 0;
  let uploadSubsetWallMs = 0;
  let matched = 0;
  let physicalBytes = 0;
  let maximumLivePayloadBytes = 0;
  let maximumLivePayloadCount = 0;
  const startedAt = performance.now();
  for (const candidate of candidates) {
    signal.throwIfAborted();
    const fileStartedAt = performance.now();
    if (livePayload !== undefined || livePayloadCount !== 0) {
      throw new Error("OPT-0069 WebCrypto opened more than one payload");
    }
    if (candidate.file.size !== candidate.record.byteLength) {
      throw new Error(`OPT-0069 ${candidate.record.name} changed size`);
    }
    const readStartedAt = performance.now();
    livePayload = await candidate.file.arrayBuffer();
    readCopyMs += performance.now() - readStartedAt;
    livePayloadCount = 1;
    maximumLivePayloadBytes = Math.max(
      maximumLivePayloadBytes,
      livePayload.byteLength,
    );
    maximumLivePayloadCount = Math.max(maximumLivePayloadCount, livePayloadCount);
    if (livePayload.byteLength !== candidate.record.byteLength) {
      throw new Error(`OPT-0069 ${candidate.record.name} short arrayBuffer read`);
    }
    signal.throwIfAborted();
    const hashStartedAt = performance.now();
    // WebCrypto has no AbortSignal overload. A cancellation received while this
    // promise is in flight is enforced immediately after this one file returns.
    const digest = await crypto.subtle.digest("SHA-256", livePayload);
    hashMs += performance.now() - hashStartedAt;
    signal.throwIfAborted();
    const finalStartedAt = performance.now();
    const actual = hex(new Uint8Array(digest));
    if (actual !== candidate.record.sha256) {
      throw new Error(`OPT-0069 ${candidate.record.name} digest mismatch`);
    }
    finalizationAndComparisonMs += performance.now() - finalStartedAt;
    matched += 1;
    physicalBytes = checkedOpt0069ByteAdd(
      physicalBytes,
      candidate.record.byteLength,
    );
    const cleanupStartedAt = performance.now();
    livePayload = undefined;
    livePayloadCount = 0;
    cleanupMs += performance.now() - cleanupStartedAt;
    if (candidate.uploadSubset) {
      uploadSubsetWallMs += performance.now() - fileStartedAt;
    }
  }
  const releaseStartedAt = performance.now();
  candidates = [];
  value.candidates.length = 0;
  cleanupMs += performance.now() - releaseStartedAt;
  await heartbeat.finish();
  const wallMs = performance.now() - startedAt;
  return timingSample({
    armId,
    order,
    owner: "webcrypto-whole-file",
    wallMs,
    readCopyMs,
    hashMs,
    finalizationAndComparisonMs,
    cleanupMs,
    uploadSubsetWallMs,
    matched,
    physicalBytes,
    maximumLivePayloadBytes,
    maximumLivePayloadCount,
    conservativeTransientBytes:
      value.inventory.largestFileBytes * OPT_0069_WEBCRYPTO_TRANSIENT_MULTIPLIER,
    maximumWorkerHeartbeatGapMs: heartbeat.maximumGapMs,
    fingerprint: value.fingerprint,
  });
}

function timingSample(value: Readonly<{
  readonly armId: Opt0069ArmId;
  readonly order: number;
  readonly owner: Opt0069Owner;
  readonly wallMs: number;
  readonly readCopyMs: number;
  readonly hashMs: number;
  readonly finalizationAndComparisonMs: number;
  readonly cleanupMs: number;
  readonly uploadSubsetWallMs: number;
  readonly matched: number;
  readonly physicalBytes: number;
  readonly maximumLivePayloadBytes: number;
  readonly maximumLivePayloadCount: number;
  readonly conservativeTransientBytes: number;
  readonly maximumWorkerHeartbeatGapMs: number;
  readonly fingerprint: string;
}>): Opt0069TimingSample {
  const boundedMemoryPassed = value.owner === "scalar-stream"
    ? value.maximumLivePayloadBytes <= OPT_0069_HASH_CHUNK_BYTES &&
      value.maximumLivePayloadCount <= 1
    : value.maximumLivePayloadBytes === OPT_0069_LARGEST_FILE_BYTES &&
      value.maximumLivePayloadCount === 1 &&
      value.conservativeTransientBytes <
        OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES;
  return Object.freeze({
    armId: value.armId,
    order: value.order,
    owner: value.owner,
    wallMs: value.wallMs,
    readCopyMs: value.readCopyMs,
    hashMs: value.hashMs,
    finalizationAndComparisonMs: value.finalizationAndComparisonMs,
    cleanupMs: value.cleanupMs,
    uploadSubsetWallMs: value.uploadSubsetWallMs,
    matchedUniqueDigests: value.matched,
    logicalRecordsCovered: OPT_0069_COMPLETE_LOGICAL_RECORDS,
    physicalBytes: value.physicalBytes,
    logicalBytes: OPT_0069_COMPLETE_LOGICAL_BYTES,
    uploadSubsetFiles: OPT_0069_UPLOAD_SUBSET_FILES,
    uploadSubsetBytes: OPT_0069_UPLOAD_SUBSET_BYTES,
    maximumExplicitLivePayloadBytes: value.maximumLivePayloadBytes,
    maximumExplicitLivePayloadCount: value.maximumLivePayloadCount,
    conservativeTransientBytes: value.conservativeTransientBytes,
    maximumWorkerHeartbeatGapMs: value.maximumWorkerHeartbeatGapMs,
    maximumPageHeartbeatGapMs: 0,
    inventoryFingerprint: value.fingerprint,
    correctnessPassed: value.matched === OPT_0069_COMPLETE_UNIQUE_DIGESTS &&
      value.physicalBytes === OPT_0069_COMPLETE_PHYSICAL_BYTES,
    boundedMemoryPassed,
    abortPassed: true,
    cleanupPassed: true,
    thermalNonNominalObservations: 0,
  });
}

interface WorkerHeartbeat {
  readonly maximumGapMs: number;
  finish(): Promise<void>;
}

function startWorkerHeartbeat(armId: Opt0069ArmId): WorkerHeartbeat {
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
      const now = performance.now();
      maximumGapMs = Math.max(maximumGapMs, now - last);
      self.clearInterval(timer);
      stopped = true;
    },
  };
}

async function hashSyntheticScalarFile(
  file: File,
  expectedBytes: number,
  expectedSha256: string,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  if (file.size !== expectedBytes) throw new Error("OPT-0069 synthetic short read");
  const reader = file.stream().getReader();
  const hash = new AceIncrementalSha256();
  let received = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const item = await reader.read();
      if (item.done) break;
      for (const chunk of boundedSlices(item.value, OPT_0069_HASH_CHUNK_BYTES)) {
        signal.throwIfAborted();
        received = checkedOpt0069ByteAdd(received, chunk.byteLength);
        hash.update(chunk);
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (received !== expectedBytes || hash.digestHex() !== expectedSha256) {
    throw new Error("OPT-0069 synthetic file proof mismatch");
  }
}

async function hashSyntheticWebCryptoFile(
  file: File,
  expectedBytes: number,
  expectedSha256: string,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  if (file.size !== expectedBytes) {
    throw new Error("OPT-0069 synthetic WebCrypto short read");
  }
  let bytes: ArrayBuffer | undefined = await file.arrayBuffer();
  signal.throwIfAborted();
  const actual = hex(new Uint8Array(
    await crypto.subtle.digest("SHA-256", bytes),
  ));
  signal.throwIfAborted();
  bytes = undefined;
  if (actual !== expectedSha256) {
    throw new Error("OPT-0069 synthetic WebCrypto proof mismatch");
  }
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
    throw new Error("OPT-0069 >32-bit SHA length probe did not finalize");
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

function boundedSlices(
  bytes: Uint8Array,
  maximumBytes: number,
): readonly Uint8Array[] {
  const chunks: Uint8Array[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += maximumBytes) {
    chunks.push(bytes.subarray(offset, Math.min(bytes.byteLength, offset + maximumBytes)));
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
  return hex(new Uint8Array(await crypto.subtle.digest("SHA-256", owned.buffer)));
}

async function sha256Text(value: string): Promise<string> {
  return await webCryptoHex(new TextEncoder().encode(value));
}

function hex(bytes: Uint8Array): string {
  let value = "";
  for (const byte of bytes) value += byte.toString(16).padStart(2, "0");
  return value;
}

function sumBytes(values: readonly number[]): number {
  return values.reduce(checkedOpt0069ByteAdd, 0);
}

function manifestIdentity(value: Readonly<{
  readonly manifestUrl: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: number;
  readonly manifest: AcePackageManifest;
}>): Readonly<{
  readonly url: string;
  readonly sha256: string;
  readonly byteLength: number;
  readonly converterRevision: number;
}> {
  return Object.freeze({
    url: value.manifestUrl,
    sha256: value.manifestSha256,
    byteLength: value.manifestByteLength,
    converterRevision: value.manifest.provenance.converterRevision,
  });
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
      error: serializeOpt0069Failure(error),
    });
  }
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function fail(error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  prepared = undefined;
  self.postMessage({ type: "failed", error: serializeOpt0069Failure(error) });
}
