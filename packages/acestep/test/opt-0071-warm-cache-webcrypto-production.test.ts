import { createHash } from "node:crypto";
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
  "../model/files-fp16-vae-revision7-experimental/manifest.json?raw";
import acquireSource from "../src/model/acquire.ts?raw";
import pipelineSource from "../src/runtime/webgpu-pipeline.ts?raw";
import demoSource from "../demo/main.ts?raw";
import workerSource from
  "./browser/opt-0071-warm-cache-webcrypto-production-worker.ts?raw";
import pageSource from
  "./browser/opt-0071-warm-cache-webcrypto-production.ts?raw";
import htmlSource from
  "./browser/opt-0071-warm-cache-webcrypto-production.html?raw";
import {
  checkedOpt0071ByteAdd,
  checkedOpt0071DurationAdd,
  OPT_0071_ARM_ORDER,
  OPT_0071_CONSERVATIVE_TRANSIENT_BYTES,
  OPT_0071_FULL_LOGICAL_BYTES,
  OPT_0071_FULL_LOGICAL_RECORDS,
  OPT_0071_FULL_PHYSICAL_BYTES,
  OPT_0071_FULL_UNIQUE_DIGESTS,
  OPT_0071_INVENTORY_FINGERPRINT,
  OPT_0071_LARGEST_FILE_BYTES,
  OPT_0071_MAXIMUM_FILE_BYTES,
  OPT_0071_MAXIMUM_TRANSIENT_BYTES,
  OPT_0071_TIMED_LOGICAL_BYTES,
  OPT_0071_TIMED_LOGICAL_RECORDS,
  OPT_0071_TIMED_PHYSICAL_BYTES,
  OPT_0071_TIMED_UNIQUE_DIGESTS,
  OPT_0071_UNRELATED_STAGE_NAMES,
  OPT_0071_UPLOAD_SUBSET_BYTES,
  OPT_0071_UPLOAD_SUBSET_FILES,
  requireOpt0071ThermalGate,
  requireOpt0071ThermalTrace,
  summarizeOpt0071Performance,
  type Opt0071ArmId,
  type Opt0071ArmSample,
  type Opt0071Owner,
  type Opt0071ThermalGate,
  type Opt0071ThermalTrace,
  type Opt0071UnrelatedStageWalls,
} from "./browser/opt-0071-warm-cache-webcrypto-production-contract.js";

describe("OPT-0071 warm-cache WebCrypto production gate", () => {
  it("derives the exact full 158/156 authority and actual timed 151/149 seam", () => {
    const mainSource = JSON.parse(mainManifestSource) as AcePackageManifest;
    const denseSource = JSON.parse(denseManifestSource) as AcePackageManifest;
    const vaeSource = JSON.parse(vaeManifestSource) as AcePackageManifest;
    const main = createAceMainAcquisitionManifest(mainSource);
    const dense = createAceOpt0009DitDenseAcquisitionManifest(denseSource);
    const vae = createAceOpt0011VaeAcquisitionManifest(vaeSource);
    const logical = [
      ...records("main", main, phaseFileNames(mainSource, [
        "text", "conditioner", "constants", "dit",
      ])),
      ...records("dit-dense", dense, new Set(
        aceRuntimePackageFiles(dense).map((item) => item.name),
      )),
      ...records("vae", vae, new Set(
        aceRuntimePackageFiles(vae).map((item) => item.name),
      )),
    ];
    const unique = new Map(logical.map((item) => [
      item.file.sha256,
      item.file.byteLength,
    ]));
    const timed = logical.filter((item) => item.packageKind !== "vae");
    const timedUnique = new Map(timed.map((item) => [
      item.file.sha256,
      item.file.byteLength,
    ]));
    const uploadUnique = new Map(logical.filter((item) => item.uploadSubset)
      .map((item) => [item.file.sha256, item.file.byteLength]));
    const fingerprint = createHash("sha256").update(logical.map((item) =>
      `${item.packageKind}\0${item.file.name}\0${item.file.byteLength}\0` +
        `${item.file.sha256}\0${item.uploadSubset ? "upload" : "other"}\n`
    ).join("")).digest("hex");
    expect({
      logicalRecords: logical.length,
      uniqueDigests: unique.size,
      logicalBytes: sum(logical.map((item) => item.file.byteLength)),
      physicalBytes: sum([...unique.values()]),
      largestFileBytes: Math.max(...unique.values()),
      uploadSubsetFiles: uploadUnique.size,
      uploadSubsetBytes: sum([...uploadUnique.values()]),
      fingerprint,
    }).toEqual({
      logicalRecords: OPT_0071_FULL_LOGICAL_RECORDS,
      uniqueDigests: OPT_0071_FULL_UNIQUE_DIGESTS,
      logicalBytes: OPT_0071_FULL_LOGICAL_BYTES,
      physicalBytes: OPT_0071_FULL_PHYSICAL_BYTES,
      largestFileBytes: OPT_0071_LARGEST_FILE_BYTES,
      uploadSubsetFiles: OPT_0071_UPLOAD_SUBSET_FILES,
      uploadSubsetBytes: OPT_0071_UPLOAD_SUBSET_BYTES,
      fingerprint: OPT_0071_INVENTORY_FINGERPRINT,
    });
    expect({
      logicalRecords: timed.length,
      uniqueDigests: timedUnique.size,
      logicalBytes: sum(timed.map((item) => item.file.byteLength)),
      physicalBytes: sum([...timedUnique.values()]),
    }).toEqual({
      logicalRecords: OPT_0071_TIMED_LOGICAL_RECORDS,
      uniqueDigests: OPT_0071_TIMED_UNIQUE_DIGESTS,
      logicalBytes: OPT_0071_TIMED_LOGICAL_BYTES,
      physicalBytes: OPT_0071_TIMED_PHYSICAL_BYTES,
    });
  });

  it("freezes ABBA and requires both READY/auth savings plus every unrelated stage", () => {
    expect(OPT_0071_ARM_ORDER.map((item) =>
      `${item.armId}-${item.owner}`
    )).toEqual([
      "A1-scalar-stream",
      "B1-webcrypto-whole-file",
      "B2-webcrypto-whole-file",
      "A2-scalar-stream",
    ]);
    const passing = OPT_0071_ARM_ORDER.map((item) => sample(
      item.armId,
      item.order,
      item.owner,
      item.owner === "scalar-stream" ? 30_000 : 9_000,
      item.owner === "scalar-stream" ? 28_000 : 7_000,
    ));
    expect(summarizeOpt0071Performance(passing)).toMatchObject({
      candidateAuthenticationMedianMs: 7_000,
      a1MinusB1ReadySavingMs: 21_000,
      a2MinusB2ReadySavingMs: 21_000,
      a1MinusB1AuthenticationSavingMs: 21_000,
      a2MinusB2AuthenticationSavingMs: 21_000,
      maximumUnrelatedRegression: 0,
      allSamplesExactAndSafe: true,
      allSamplesNominal: true,
      passed: true,
    });
    expect(
      summarizeOpt0071Performance(passing)
        .candidateAuthenticationThroughputBytesPerSecond,
    ).toBeGreaterThan(915_749_892);
    expect(summarizeOpt0071Performance(passing.map((item) =>
      item.armId === "B1" ? { ...item, readyWallMs: 16_000 } : item
    )).passed).toBe(false);
    expect(summarizeOpt0071Performance(passing.map((item) =>
      item.armId === "B2"
        ? { ...item, authenticationWallMs: 14_000 }
        : item
    )).passed).toBe(false);
    expect(summarizeOpt0071Performance(passing.map((item) =>
      item.armId === "B1"
        ? {
            ...item,
            unrelatedStageWalls: {
              ...item.unrelatedStageWalls,
              "opfs-open": item.unrelatedStageWalls["opfs-open"] * 1.021,
            },
          }
        : item
    )).passed).toBe(false);
    expect(summarizeOpt0071Performance(passing.map((item) =>
      item.armId === "B2"
        ? { ...item, thermalNonNominalObservations: 1 }
        : item
    )).passed).toBe(false);
    expect(summarizeOpt0071Performance(passing.map((item) =>
      item.armId === "B1"
        ? { ...item, timedInventoryFingerprint: "d".repeat(64) }
        : item
    )).passed).toBe(false);
    expect(() => summarizeOpt0071Performance([
      passing[1]!,
      passing[0]!,
      ...passing.slice(2),
    ])).toThrow(/balanced sample inventory/);
  });

  it("accepts only a fresh nominal gate and continuous through-termination trace", () => {
    const observations = Object.freeze(Array.from({ length: 32 }, (_, index) =>
      Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level: 0,
        rawValue: "0",
      })
    ));
    const gate: Opt0071ThermalGate = Object.freeze({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      startedAtEpochMilliseconds: 2_000,
      completedAtEpochMilliseconds: 33_000,
      observationCount: observations.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations,
    });
    const accepted = requireOpt0071ThermalGate(gate, 1_000, 33_500);
    const traceObservations = Object.freeze([
      ...observations,
      Object.freeze({ atEpochMilliseconds: 34_000, level: 0, rawValue: "0" }),
      Object.freeze({ atEpochMilliseconds: 35_000, level: 0, rawValue: "0" }),
    ]);
    const trace: Opt0071ThermalTrace = Object.freeze({
      source: gate.source,
      command: gate.command,
      rawTraceSha256: "a".repeat(64),
      completedAtEpochMilliseconds: 35_000,
      observationCount: traceObservations.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations: traceObservations,
      transitions: Object.freeze([
        Object.freeze({ atEpochMilliseconds: 2_000, level: 0 }),
      ]),
    });
    expect(requireOpt0071ThermalTrace(trace, accepted, 34_500, 35_200))
      .toEqual(trace);
    expect(() => requireOpt0071ThermalGate(
      { ...gate, startedAtEpochMilliseconds: 0 },
      1_000,
      33_500,
    )).toThrow(/thermal/);
    expect(() => requireOpt0071ThermalTrace(
      { ...trace, completedAtEpochMilliseconds: 34_000 },
      accepted,
      34_500,
      35_200,
    )).toThrow(/thermal|termination/);
  });

  it("uses the exact OPT-0072 cache-only product tuple and owner seam", () => {
    expect(workerSource).toContain(
      "cacheAuthenticationOwner: definition.owner",
    );
    expect(workerSource).toContain('modelSource: "cache-only"');
    expect(workerSource).toContain(
      '"opt-0070-fixed32-quad-query32-full-self-production-v1"',
    );
    expect(workerSource).toContain(
      '"opt-0070-c2378-overlap64-production-v1"',
    );
    expect(workerSource).toContain(
      '"opt-0072-mixed-fp16-fixed32-dual-k4-production-v1"',
    );
    expect(workerSource).toContain(
      'authenticatedVaeConverterRevision: 7',
    );
    expect(workerSource).toContain("maxWindowFrames: 2_378");
    expect(demoSource).toContain(
      '"opt-0070-fixed32-quad-query32-full-self-production-v1"',
    );
    expect(demoSource).toContain(
      '"opt-0070-c2378-overlap64-production-v1"',
    );
    expect(demoSource).toContain(
      '"/model/files-fp16-vae-revision7-experimental/manifest.json"',
    );
    expect(demoSource).toContain(
      '"opt-0072-mixed-fp16-fixed32-dual-k4-production-v1"',
    );
    expect(demoSource).toContain(
      '"36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7"',
    );
    expect(pipelineSource).toContain(
      "readonly cacheAuthenticationOwner?: AceCacheAuthenticationOwner",
    );
    expect(pipelineSource).toContain(
      "{ cacheAuthenticationOwner: options.cacheAuthenticationOwner }",
    );
    expect(acquireSource).toMatch(
      /ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER:[\s\S]*= "webcrypto-whole-file"/u,
    );
  });

  it("keeps full trust preflight read-only, sequential, bounded, and fail-closed", () => {
    expect(OPT_0071_LARGEST_FILE_BYTES).toBeLessThan(
      OPT_0071_MAXIMUM_FILE_BYTES,
    );
    expect(OPT_0071_CONSERVATIVE_TRANSIENT_BYTES).toBeLessThan(
      OPT_0071_MAXIMUM_TRANSIENT_BYTES,
    );
    expect(workerSource).toContain("new AceOpfsModelCache(cacheRoot)");
    expect(workerSource).toContain("await cache.openCandidate(first.record)");
    expect(workerSource).not.toMatch(/cache\.(?:begin|remove)\(/u);
    expect(workerSource).not.toContain("createWritable");
    expect(workerSource).toContain("let livePayload: ArrayBuffer | undefined");
    expect(workerSource).toContain("livePayload = undefined");
    expect(workerSource).toContain(
      'crypto.subtle.digest("SHA-256", livePayload)',
    );
    expect(workerSource).toContain("injected arrayBuffer rejection");
    expect(workerSource).toContain("injected WebCrypto rejection");
    expect(workerSource).toContain("HASH_CHUNK_BYTES - 1");
    expect(workerSource).toContain("scalarShortReadRejected");
    expect(workerSource).toContain("scalarPreAbortRejected");
    expect(workerSource).toContain("secondFinalizeRejected");
    expect(workerSource).toContain("postFinalizeUpdateRejected");
    expect(workerSource).toContain("screenGreaterThan32BitShaLengthEncoding");
    expect(workerSource).toContain("duringReadRejectedAtBoundary");
    expect(workerSource).toContain("duringDigestRejectedAtBoundary");
    expect(workerSource).toContain("noScalarFallbackAfterWebCryptoFailure: true");
    const arrayBufferIndex = acquireSource.indexOf(
      "payload = await candidate.arrayBuffer()",
    );
    const beforeDigestAbortIndex = acquireSource.indexOf(
      "signal?.throwIfAborted()",
      arrayBufferIndex,
    );
    const digestIndex = acquireSource.indexOf(
      'subtle.digest("SHA-256", payload)',
      arrayBufferIndex,
    );
    const afterDigestAbortIndex = acquireSource.indexOf(
      "signal?.throwIfAborted()",
      digestIndex,
    );
    expect(arrayBufferIndex).toBeGreaterThan(0);
    expect(beforeDigestAbortIndex).toBeGreaterThan(arrayBufferIndex);
    expect(digestIndex).toBeGreaterThan(beforeDigestAbortIndex);
    expect(afterDigestAbortIndex).toBeGreaterThan(digestIndex);
    expect(checkedOpt0071ByteAdd(0xffff_ffff, 2)).toBe(4_294_967_297);
    expect(checkedOpt0071DurationAdd(0.3, 0.4)).toBeCloseTo(0.7, 12);
    expect(() => checkedOpt0071DurationAdd(-0.1, 1)).toThrow(
      /duration accounting/,
    );
    expect(() => checkedOpt0071DurationAdd(1, Number.NaN)).toThrow(
      /duration accounting/,
    );
    expect(workerSource).toMatch(
      /sumDurations\(\s*packageEvents\.map\(\(event\) => event\.wallMs\)/u,
    );
    expect(workerSource).not.toContain(
      "sum(packageEvents.map((event) => event.wallMs))",
    );
  });

  it("captures complete arm evidence through disposal and fresh worker termination", () => {
    expect(workerSource).toContain("completeCaptureEvents");
    expect(workerSource).toContain("completeProgressEvents");
    expect(workerSource).toContain("await backend.dispose()");
    expect(workerSource).toContain("backendDisposedBeforeWorkerCompletion: true");
    expect(workerSource).toContain("VAE payload authentication is deferred");
    expect(pageSource).toContain("function startNextArm()");
    expect(pageSource).toContain("worker?.terminate()");
    expect(pageSource).toContain("workerTerminatedAtEpochMilliseconds");
    expect(pageSource).toContain("deferredVaePayloadAuthenticationCountedInsideReady: false");
    expect(pageSource).toContain("window.__ACE_OPT0071_RESULT__ = receipt");
    expect(htmlSource).toContain("PRE-FLIGHT READY");
    expect(htmlSource).toContain("exact current OPT-0072 demo tuple");
    expect(htmlSource).toContain("thermal-poller.py poll /tmp/opt71-ARM.jsonl");
    expect(htmlSource).toContain("thermal-poller.py gate");
    expect(htmlSource).toContain("thermal-poller.py trace");
    expect(htmlSource).toContain("never reuse a gate or trace");
    expect(htmlSource).toContain(
      "/test/browser/opt-0071-warm-cache-webcrypto-production.html?coreCommit=CORE40",
    );
  });
});

function records(
  packageKind: "main" | "dit-dense" | "vae",
  manifest: AcePackageManifest,
  uploadFiles: ReadonlySet<string>,
): readonly Readonly<{
  readonly packageKind: "main" | "dit-dense" | "vae";
  readonly file: ReturnType<typeof aceRuntimePackageFiles>[number];
  readonly uploadSubset: boolean;
}>[] {
  return aceRuntimePackageFiles(manifest).map((file) => Object.freeze({
    packageKind,
    file,
    uploadSubset: uploadFiles.has(file.name),
  }));
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

function sample(
  armId: Opt0071ArmId,
  order: number,
  owner: Opt0071Owner,
  readyWallMs: number,
  authenticationWallMs: number,
): Opt0071ArmSample {
  const unrelatedStageWalls = Object.freeze(Object.fromEntries(
    OPT_0071_UNRELATED_STAGE_NAMES.map((stage) => [
      stage,
      stage === "ready-publication-and-residual" ? 1_200 : 100,
    ]),
  )) as Opt0071UnrelatedStageWalls;
  return Object.freeze({
    armId,
    order,
    owner,
    readyWallMs,
    authenticationWallMs,
    authenticationThroughputBytesPerSecond:
      OPT_0071_TIMED_PHYSICAL_BYTES / (authenticationWallMs / 1_000),
    unrelatedStageWalls,
    aggregateUnrelatedWallMs: 2_000,
    timedLogicalRecords: OPT_0071_TIMED_LOGICAL_RECORDS,
    timedUniqueDigests: OPT_0071_TIMED_UNIQUE_DIGESTS,
    timedLogicalBytes: OPT_0071_TIMED_LOGICAL_BYTES,
    timedPhysicalBytes: OPT_0071_TIMED_PHYSICAL_BYTES,
    timedInventoryFingerprint: "c".repeat(64),
    fullLogicalRecordsProven: OPT_0071_FULL_LOGICAL_RECORDS,
    fullUniqueDigestsProven: OPT_0071_FULL_UNIQUE_DIGESTS,
    fullLogicalBytesProven: OPT_0071_FULL_LOGICAL_BYTES,
    fullPhysicalBytesProven: OPT_0071_FULL_PHYSICAL_BYTES,
    inventoryFingerprint: OPT_0071_INVENTORY_FINGERPRINT,
    maximumExplicitLivePayloadBytes: owner === "scalar-stream"
      ? 4 * 1024 * 1024
      : OPT_0071_LARGEST_FILE_BYTES,
    maximumExplicitLivePayloadCount: 1,
    conservativeTransientBytes: owner === "scalar-stream"
      ? 4 * 1024 * 1024
      : OPT_0071_CONSERVATIVE_TRANSIENT_BYTES,
    downloadCount: 0,
    downloadBytes: 0,
    cacheMutationCount: 0,
    exactDigestsPassed: true,
    memoryPassed: true,
    cancellationPassed: true,
    lifecyclePassed: true,
    thermalNonNominalObservations: 0,
  });
}

function sum(values: readonly number[]): number {
  return values.reduce(checkedOpt0071ByteAdd, 0);
}
