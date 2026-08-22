import { describe, expect, it } from "vitest";
import { aceRuntimePackageFiles } from "../src/model/acquire.js";
import type {
  AcePackageManifest,
  AceTensorPhase,
} from "../src/model/manifest.js";
import {
  createAceMainAcquisitionManifest,
  createAceOpt0009DitDenseAcquisitionManifest,
  createAceOpt0011VaeAcquisitionManifest,
} from "../src/runtime/webgpu-pipeline.js";
import mainManifestSource from "../model/files-reference/manifest.json?raw";
import denseManifestSource from
  "../model/files-fp16-dit-rev7-oracle/manifest.json?raw";
import vaeManifestSource from
  "../model/files-fp16-vae-experimental/manifest.json?raw";
import experimentSource from
  "../optimization/experiments/OPT-0069-warm-cache-authentication-hash.md?raw";
import acquireSource from "../src/model/acquire.ts?raw";
import shaSource from "../src/model/sha256.ts?raw";
import workerSource from
  "./browser/opt-0069-cache-authentication-hash-worker.ts?raw";
import pageSource from
  "./browser/opt-0069-cache-authentication-hash.ts?raw";
import htmlSource from
  "./browser/opt-0069-cache-authentication-hash.html?raw";
import {
  checkedOpt0069ByteAdd,
  OPT_0069_ARM_ORDER,
  OPT_0069_COMPLETE_LOGICAL_BYTES,
  OPT_0069_COMPLETE_LOGICAL_RECORDS,
  OPT_0069_COMPLETE_PHYSICAL_BYTES,
  OPT_0069_COMPLETE_UNIQUE_DIGESTS,
  OPT_0069_HASH_CHUNK_BYTES,
  OPT_0069_LARGEST_FILE_BYTES,
  OPT_0069_UPLOAD_SUBSET_BYTES,
  OPT_0069_UPLOAD_SUBSET_FILES,
  OPT_0069_UPLOAD_SUBSET_REPORTED_GB,
  OPT_0069_WEBCRYPTO_MAXIMUM_FILE_BYTES,
  OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES,
  requireOpt0069Inventory,
  requireOpt0069ThermalGate,
  requireOpt0069ThermalTrace,
  summarizeOpt0069Performance,
  type Opt0069ArmId,
  type Opt0069Owner,
  type Opt0069ThermalGate,
  type Opt0069ThermalTrace,
  type Opt0069TimingSample,
} from "./browser/opt-0069-cache-authentication-hash-contract.js";

describe("OPT-0069 bounded warm-cache hash screen", () => {
  it("derives the exact 158/156/7.326GB inventory from production helpers", () => {
    const mainSource = JSON.parse(mainManifestSource) as AcePackageManifest;
    const denseSource = JSON.parse(denseManifestSource) as AcePackageManifest;
    const vaeSource = JSON.parse(vaeManifestSource) as AcePackageManifest;
    const main = createAceMainAcquisitionManifest(mainSource);
    const dense = createAceOpt0009DitDenseAcquisitionManifest(denseSource);
    const vae = createAceOpt0011VaeAcquisitionManifest(vaeSource);
    const packages = [
      packageInventory("main", main),
      packageInventory("dit-dense", dense),
      packageInventory("vae", vae),
    ] as const;
    const logical = [
      ...aceRuntimePackageFiles(main),
      ...aceRuntimePackageFiles(dense),
      ...aceRuntimePackageFiles(vae),
    ];
    const unique = new Map(logical.map((file) => [file.sha256, file.byteLength]));
    const mainUploadNames = phaseFileNames(mainSource, [
      "text", "conditioner", "constants", "dit",
    ]);
    const uploadRecords = [
      ...aceRuntimePackageFiles(main).filter((file) =>
        mainUploadNames.has(file.name)
      ),
      ...aceRuntimePackageFiles(dense),
      ...aceRuntimePackageFiles(vae),
    ];
    const uploadUnique = new Map(uploadRecords.map((file) =>
      [file.sha256, file.byteLength]
    ));
    const inventory = requireOpt0069Inventory({
      packages,
      logicalRecords: logical.length,
      uniqueDigests: unique.size,
      logicalBytes: sum([...logical.map((file) => file.byteLength)]),
      physicalBytes: sum([...unique.values()]),
      largestFileBytes: Math.max(...unique.values()),
      uploadSubsetFiles: uploadUnique.size,
      uploadSubsetBytes: sum([...uploadUnique.values()]),
    });
    expect(inventory).toMatchObject({
      logicalRecords: OPT_0069_COMPLETE_LOGICAL_RECORDS,
      uniqueDigests: OPT_0069_COMPLETE_UNIQUE_DIGESTS,
      logicalBytes: OPT_0069_COMPLETE_LOGICAL_BYTES,
      physicalBytes: OPT_0069_COMPLETE_PHYSICAL_BYTES,
      largestFileBytes: OPT_0069_LARGEST_FILE_BYTES,
      uploadSubsetFiles: OPT_0069_UPLOAD_SUBSET_FILES,
      uploadSubsetBytes: OPT_0069_UPLOAD_SUBSET_BYTES,
    });
    expect(packages).toEqual([
      {
        packageKind: "main",
        logicalRecords: 103,
        uniqueDigests: 101,
        logicalBytes: 4_140_848_075,
        physicalBytes: 4_136_399_389,
      },
      {
        packageKind: "dit-dense",
        logicalRecords: 48,
        uniqueDigests: 48,
        logicalBytes: 3_020_808_192,
        physicalBytes: 3_020_808_192,
      },
      {
        packageKind: "vae",
        logicalRecords: 7,
        uniqueDigests: 7,
        logicalBytes: 168_791_552,
        physicalBytes: 168_791_552,
      },
    ]);
    expect(uploadUnique.size).toBe(31 + 14 + 2 + 48 + 7);
    expect(OPT_0069_UPLOAD_SUBSET_REPORTED_GB).toBe("5.7318 GB");
  });

  it("freezes the balanced four-pair order and both directional gates", () => {
    expect(OPT_0069_ARM_ORDER.map((item) =>
      `${item.armId}-${item.owner}`
    )).toEqual([
      "A1-scalar-stream",
      "B1-webcrypto-whole-file",
      "B2-webcrypto-whole-file",
      "A2-scalar-stream",
      "B3-webcrypto-whole-file",
      "A3-scalar-stream",
      "A4-scalar-stream",
      "B4-webcrypto-whole-file",
    ]);
    const passing = OPT_0069_ARM_ORDER.map((item) => sample(
      item.armId,
      item.order,
      item.owner,
      item.owner === "scalar-stream" ? 24_000 : 7_000,
    ));
    expect(summarizeOpt0069Performance(passing)).toMatchObject({
      candidateMedianMs: 7_000,
      forwardMedianSavingMs: 17_000,
      reverseMedianSavingMs: 17_000,
      maximumReadCopyRegression: 0,
      maximumResponsivenessRegression: 0,
      allSamplesExact: true,
      allSamplesNominal: true,
      passed: true,
    });
    expect(
      summarizeOpt0069Performance(passing).candidateThroughputBytesPerSecond,
    ).toBeGreaterThan(915_749_892);
    expect(summarizeOpt0069Performance(passing.map((item) =>
      item.armId === "B4" ? { ...item, wallMs: 10_000 } : item
    )).passed).toBe(true);
    expect(summarizeOpt0069Performance(passing.map((item) =>
      item.owner === "webcrypto-whole-file"
        ? { ...item, wallMs: 9_000 }
        : item
    )).passed).toBe(false);
    expect(summarizeOpt0069Performance(passing.map((item) =>
      item.armId === "B2" ? { ...item, readCopyMs: 1_300 } : item
    )).passed).toBe(false);
    expect(summarizeOpt0069Performance(passing.map((item) =>
      item.armId === "B3"
        ? { ...item, thermalNonNominalObservations: 1 }
        : item
    )).passed).toBe(false);
    expect(() => summarizeOpt0069Performance([
      passing[1]!,
      passing[0]!,
      ...passing.slice(2),
    ])).toThrow(/balanced sample inventory/);
  });

  it("accepts only a fresh nominal gate and its continuous cleanup trace", () => {
    const observations = Object.freeze(Array.from({ length: 32 }, (_, index) =>
      Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level: 0,
        rawValue: "0",
      })
    ));
    const gate: Opt0069ThermalGate = Object.freeze({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      startedAtEpochMilliseconds: 2_000,
      completedAtEpochMilliseconds: 33_000,
      observationCount: observations.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations,
    });
    const accepted = requireOpt0069ThermalGate(gate, 1_000, 33_500);
    const traceObservations = Object.freeze([
      ...observations,
      Object.freeze({ atEpochMilliseconds: 34_000, level: 0, rawValue: "0" }),
      Object.freeze({ atEpochMilliseconds: 35_000, level: 0, rawValue: "0" }),
    ]);
    const trace: Opt0069ThermalTrace = Object.freeze({
      source: gate.source,
      command: gate.command,
      rawTraceSha256: "a".repeat(64),
      completedAtEpochMilliseconds: 35_000,
      observationCount: traceObservations.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations: traceObservations,
      transitions: Object.freeze([
        Object.freeze({ atEpochMilliseconds: 33_000, level: 0 }),
      ]),
    });
    expect(requireOpt0069ThermalTrace(trace, accepted, 34_500, 35_200))
      .toEqual(trace);
    expect(() => requireOpt0069ThermalGate(
      { ...gate, startedAtEpochMilliseconds: 0 },
      1_000,
      33_500,
    )).toThrow(/thermal/);
    expect(() => requireOpt0069ThermalTrace(
      { ...trace, completedAtEpochMilliseconds: 34_000 },
      accepted,
      34_500,
      35_200,
    )).toThrow(/thermal|cleanup/);
  });

  it("bounds WebCrypto to one sub-128MiB file and a sub-384MiB transient", () => {
    expect(OPT_0069_LARGEST_FILE_BYTES).toBeLessThan(
      OPT_0069_WEBCRYPTO_MAXIMUM_FILE_BYTES,
    );
    expect(OPT_0069_LARGEST_FILE_BYTES * 3).toBeLessThan(
      OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES,
    );
    expect(workerSource).toContain("let livePayload: ArrayBuffer | undefined");
    expect(workerSource).toContain("livePayloadCount = 1");
    expect(workerSource).toContain("livePayload = undefined");
    expect(workerSource).toContain("livePayloadCount = 0");
    expect(workerSource).toContain(
      'crypto.subtle.digest("SHA-256", livePayload)',
    );
    expect(workerSource).toContain("webCryptoDigestInternallyAbortable: false");
    expect(workerSource).toContain("measureUserAgentSpecificMemory");
    expect(pageSource).toContain("usedJSHeapSize");
  });

  it("keeps both frozen owners and selects the promoted OPT-0071 owner", () => {
    expect(acquireSource).toContain(
      "ACE_MODEL_CACHE_WEBCRYPTO_MAX_FILE_BYTES = 128 * 1024 * 1024",
    );
    expect(acquireSource).toContain('subtle.digest("SHA-256", payload)');
    expect(acquireSource).toContain("candidate.stream().getReader()");
    expect(acquireSource).toContain("new AceIncrementalSha256()");
    expect(acquireSource).toMatch(
      /ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER:[\s\S]*= "webcrypto-whole-file"/u,
    );
    expect(workerSource).toContain("candidate.file.stream().getReader()");
    expect(workerSource).toContain(
      "boundedSlices(item.value, OPT_0069_HASH_CHUNK_BYTES)",
    );
    expect(workerSource).toContain("new AceIncrementalSha256()");
    expect(workerSource).toContain("const lengths = [");
    expect(workerSource).toContain("OPT_0069_HASH_CHUNK_BYTES - 1");
    expect(workerSource).toContain("signed-byte vector");
    expect(workerSource).toContain("one-bit corruption");
    expect(workerSource).toContain("short read");
    expect(workerSource).toContain("secondFinalizeRejected");
    expect(workerSource).toContain("greaterThan32BitCumulativeLength");
    expect(checkedOpt0069ByteAdd(0xffff_ffff, 2)).toBe(4_294_967_297);
    expect(shaSource).toContain("private totalBytes = 0");
    expect(workerSource).not.toMatch(/requestAdapter|requestDevice|GPUDevice/u);
    expect(workerSource).not.toMatch(/cache\.(?:begin|remove)\(/u);
    expect(workerSource).toContain(
      'getDirectoryHandle(READ_ONLY_CACHE_DIRECTORY)',
    );
    expect(workerSource).not.toContain("AceOpfsModelCache.open()");
    expect(workerSource).not.toContain("createWritable");
  });

  it("uses fresh workers/traces, preserves heartbeats, and states the narrow authority", () => {
    expect(pageSource).toContain("function startNextArm()");
    expect(pageSource).toContain("new Worker(");
    expect(pageSource).toContain("worker?.terminate()");
    expect(pageSource).toContain("startPageHeartbeat()");
    expect(workerSource).toContain("startWorkerHeartbeat(definition.armId)");
    expect(htmlSource).toContain("thermal-poller.py poll /tmp/opt69-ARM.jsonl");
    expect(htmlSource).toContain("thermal-poller.py gate");
    expect(htmlSource).toContain("thermal-poller.py trace");
    expect(htmlSource).toContain("never reuse a gate or trace");
    expect(pageSource).toContain(
      "isolatedPassAuthorizesOnlyProductionSeamIntegrationUnderOpt0069: true",
    );
    expect(pageSource).toContain("productionIntegrationPerformed: false");
    expect(pageSource).toContain("endToEndInitializationSavingClaimed: false");
    expect(experimentSource).toContain("median complete authentication wall");
    expect(experimentSource).toContain("`915,749,892 B/s`");
  });

  it("defers rather than mislabels an unproven WASM SIMD arm", () => {
    expect(workerSource).toContain("wasmArmIncluded: false");
    expect(pageSource).toContain("optionalWasmArmIncluded: false");
    expect(pageSource).toContain("true four-file simd128 owner");
    expect(workerSource).not.toMatch(/WebAssembly\.(?:instantiate|compile)/u);
    expect(htmlSource).toContain("optional WASM arm is deliberately absent");
  });
});

function packageInventory(
  packageKind: "main" | "dit-dense" | "vae",
  manifest: AcePackageManifest,
): Readonly<{
  readonly packageKind: "main" | "dit-dense" | "vae";
  readonly logicalRecords: number;
  readonly uniqueDigests: number;
  readonly logicalBytes: number;
  readonly physicalBytes: number;
}> {
  const logical = aceRuntimePackageFiles(manifest);
  const unique = new Map(logical.map((file) => [file.sha256, file.byteLength]));
  return Object.freeze({
    packageKind,
    logicalRecords: logical.length,
    uniqueDigests: unique.size,
    logicalBytes: sum(logical.map((file) => file.byteLength)),
    physicalBytes: sum([...unique.values()]),
  });
}

function phaseFileNames(
  manifest: AcePackageManifest,
  phases: readonly AceTensorPhase[],
): ReadonlySet<string> {
  const selected = new Set(phases);
  return new Set(Object.values(manifest.tensors)
    .filter((tensor) => selected.has(tensor.phase))
    .map((tensor) => tensor.shard));
}

function sum(values: readonly number[]): number {
  return values.reduce(checkedOpt0069ByteAdd, 0);
}

function sample(
  armId: Opt0069ArmId,
  order: number,
  owner: Opt0069Owner,
  wallMs: number,
): Opt0069TimingSample {
  return Object.freeze({
    armId,
    order,
    owner,
    wallMs,
    readCopyMs: 1_000,
    hashMs: wallMs - 1_010,
    finalizationAndComparisonMs: 5,
    cleanupMs: 5,
    uploadSubsetWallMs: wallMs *
      (OPT_0069_UPLOAD_SUBSET_BYTES / OPT_0069_COMPLETE_PHYSICAL_BYTES),
    matchedUniqueDigests: OPT_0069_COMPLETE_UNIQUE_DIGESTS,
    logicalRecordsCovered: OPT_0069_COMPLETE_LOGICAL_RECORDS,
    physicalBytes: OPT_0069_COMPLETE_PHYSICAL_BYTES,
    logicalBytes: OPT_0069_COMPLETE_LOGICAL_BYTES,
    uploadSubsetFiles: OPT_0069_UPLOAD_SUBSET_FILES,
    uploadSubsetBytes: OPT_0069_UPLOAD_SUBSET_BYTES,
    maximumExplicitLivePayloadBytes: owner === "scalar-stream"
      ? OPT_0069_HASH_CHUNK_BYTES
      : OPT_0069_LARGEST_FILE_BYTES,
    maximumExplicitLivePayloadCount: 1,
    conservativeTransientBytes: owner === "scalar-stream"
      ? OPT_0069_HASH_CHUNK_BYTES
      : OPT_0069_LARGEST_FILE_BYTES * 3,
    maximumWorkerHeartbeatGapMs: 25,
    maximumPageHeartbeatGapMs: 25,
    inventoryFingerprint: "a".repeat(64),
    correctnessPassed: true,
    boundedMemoryPassed: true,
    abortPassed: true,
    cleanupPassed: true,
    thermalNonNominalObservations: 0,
  });
}
