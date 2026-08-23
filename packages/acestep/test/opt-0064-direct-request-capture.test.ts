import { describe, expect, it } from "vitest";
import acquireSource from "../src/model/acquire.ts?raw";
import tensorSource from "../src/model/gpu-tensors.ts?raw";
import uploadSource from "../src/model/gpu-upload.ts?raw";
import audioSource from "../src/runtime/audio-output.ts?raw";
import pipelineSource from "../src/runtime/webgpu-pipeline.ts?raw";
import rawSource from "../src/webgpu/vae-chunks.ts?raw";
import wavSource from "../src/webgpu/vae-wav.ts?raw";
import experimentSource from
  "../optimization/experiments/OPT-0064-direct-request-warm-start-load-overlap.md?raw";
import workerSource from
  "./browser/opt-0064-direct-request-capture-worker.ts?raw";
import pageSource from
  "./browser/opt-0064-direct-request-capture.ts?raw";
import htmlSource from
  "./browser/opt-0064-direct-request-capture.html?raw";
import {
  OPT_0064_ACCEPTED_WAV_SHA256,
  OPT_0064_ACCEPTED_WAV_AUTHORITY,
  OPT_0064_OUTPUT_WAV_BYTES,
  OPT_0064_REQUEST_BYTE_LENGTH,
  OPT_0064_REQUEST_CANONICAL_JSON,
  OPT_0064_REQUEST_SHA256,
  OPT_0064_TOTAL_UPLOAD_BYTES,
  OPT_0064_TOTAL_UPLOAD_DRAINS,
  OPT_0064_TOTAL_UPLOAD_FILES,
  OPT_0064_TOTAL_UPLOAD_GAPS,
  OPT_0064_UPLOAD_PHASES,
  requireOpt0064ThermalGate,
  requireOpt0064ThermalTrace,
  serializeOpt0064Failure,
  type Opt0064ThermalGate,
  type Opt0064ThermalTrace,
} from "./browser/opt-0064-direct-request-capture-contract.js";

describe("OPT-0064 capture-only direct warm request", () => {
  it("freezes the accepted request/output and audited 5.73 GB upload inventory", async () => {
    const bytes = new TextEncoder().encode(OPT_0064_REQUEST_CANONICAL_JSON);
    expect(bytes.byteLength).toBe(OPT_0064_REQUEST_BYTE_LENGTH);
    expect(
      [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
        .map((value) => value.toString(16).padStart(2, "0")).join(""),
    ).toBe(OPT_0064_REQUEST_SHA256);
    expect(OPT_0064_ACCEPTED_WAV_SHA256).toBe(
      "d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477",
    );
    expect(OPT_0064_ACCEPTED_WAV_AUTHORITY).toBe(
      "owner-approved-stage1-packed-bf16-fp32-candidate-a",
    );
    expect(OPT_0064_OUTPUT_WAV_BYTES).toBe(4_608_044);
    expect(OPT_0064_UPLOAD_PHASES).toEqual([
      { id: "text", fileCount: 31, bytes: 1_191_553_024,
        queueDrains: 34, queueEmptyGaps: 3 },
      { id: "conditioner-constants", fileCount: 14, bytes: 1_220_575_232,
        queueDrains: 26, queueEmptyGaps: 12 },
      { id: "dit", fileCount: 50, bytes: 3_150_917_888,
        queueDrains: 75, queueEmptyGaps: 25 },
      { id: "vae", fileCount: 7, bytes: 168_791_552,
        queueDrains: 8, queueEmptyGaps: 1 },
    ]);
    expect(OPT_0064_UPLOAD_PHASES.reduce((sum, phase) =>
      sum + phase.fileCount, 0)).toBe(OPT_0064_TOTAL_UPLOAD_FILES);
    expect(OPT_0064_UPLOAD_PHASES.reduce((sum, phase) =>
      sum + phase.bytes, 0)).toBe(OPT_0064_TOTAL_UPLOAD_BYTES);
    expect(OPT_0064_UPLOAD_PHASES.reduce((sum, phase) =>
      sum + phase.queueDrains, 0)).toBe(OPT_0064_TOTAL_UPLOAD_DRAINS);
    expect(OPT_0064_UPLOAD_PHASES.reduce((sum, phase) =>
      sum + phase.queueEmptyGaps, 0)).toBe(OPT_0064_TOTAL_UPLOAD_GAPS);
    expect(experimentSource).toContain("`5,731,837,696` bytes across `102` files");
    expect(experimentSource).toContain("`143` queue drains and `41` explicit");
  });

  it("accepts only a fresh nominal gate and the same trace through cleanup", () => {
    const observations = Object.freeze(Array.from({ length: 32 }, (_, index) =>
      Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level: 0,
        rawValue: "0",
      })
    ));
    const gate: Opt0064ThermalGate = Object.freeze({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      startedAtEpochMilliseconds: 2_000,
      completedAtEpochMilliseconds: 33_000,
      observationCount: observations.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations,
    });
    const accepted = requireOpt0064ThermalGate(gate, 1_000, 33_500);
    const continued = Object.freeze([
      ...observations,
      ...Array.from({ length: 7 }, (_, index) => Object.freeze({
        atEpochMilliseconds: 34_000 + index * 1_000,
        level: index < 2 ? 0 : 1,
        rawValue: index < 2 ? "0" : "1",
      })),
    ]);
    const trace: Opt0064ThermalTrace = Object.freeze({
      source: gate.source,
      command: gate.command,
      rawTraceSha256: "a".repeat(64),
      completedAtEpochMilliseconds: 40_000,
      observationCount: continued.length,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 5,
      observations: continued,
      transitions: Object.freeze([
        Object.freeze({ atEpochMilliseconds: 2_000, level: 0 }),
        Object.freeze({ atEpochMilliseconds: 36_000, level: 1 }),
      ]),
    });
    expect(requireOpt0064ThermalTrace(trace, accepted, 39_500, 40_100))
      .toEqual(trace);
    expect(() => requireOpt0064ThermalGate(
      { ...gate, nonNominalObservationCount: 0,
        observations: [{ ...observations[0]!, level: 1, rawValue: "1" },
          ...observations.slice(1)] },
      1_000,
      33_500,
    )).toThrow(/thermal gate/);
    expect(() => requireOpt0064ThermalTrace(
      { ...trace, observations: trace.observations.slice(1),
        observationCount: trace.observationCount - 1 },
      accepted,
      39_500,
      40_100,
    )).toThrow(/through-cleanup/);
  });

  it("preserves bounded AggregateError children for a setup-only failure", () => {
    const missing = Object.assign(
      new Error("missing cached shard"),
      { name: "NotFoundError", code: "CACHE_MISS" },
    );
    const cleanup = new Error("device cleanup failed");
    const serialized = serializeOpt0064Failure(new AggregateError(
      [missing, new AggregateError([cleanup], "nested cleanup")],
      "initialization and cleanup failed",
    ));
    expect(serialized).toMatchObject({
      name: "AggregateError",
      message: "initialization and cleanup failed",
      errorCount: 2,
      errors: [
        {
          name: "NotFoundError",
          message: "missing cached shard",
          code: "CACHE_MISS",
        },
        {
          name: "AggregateError",
          message: "nested cleanup",
          errorCount: 1,
          errors: [{ message: "device cleanup failed" }],
        },
      ],
    });
    expect(workerSource).toContain(
      'schema: "ace-opt-0064-worker-failure-evidence-v1"',
    );
    expect(workerSource).toContain("lastInitializationProgress");
    expect(workerSource).toContain("failedCaptureOperations");
    expect(workerSource).toContain("fatalDiagnosticCodes");
    expect(workerSource).toContain("configuredVaeRuntimeProfile");
    expect(workerSource).toContain("sha256Matches");
    expect(workerSource).toContain("byteLengthMatches");
    expect(workerSource).toContain(
      "captureSummary: input.captureSummary ?? null",
    );
    expect(workerSource.indexOf(
      "const validatedCaptureSummary = validateCapture(events)",
    )).toBeLessThan(workerSource.indexOf(
      'throw new Error("OPT-0064 direct WAV identity changed")',
    ));
  });

  it("keeps authentication/upload capture observational and exact-File-only", () => {
    expect(acquireSource).toContain(
      "markAceAuthenticatedGpuSource(candidate, file)",
    );
    expect(acquireSource).toContain("exactImmutableFileIdentity: true");
    expect(acquireSource).toContain("redundantHashPerformed: false");
    expect(uploadSource).toContain(
      "const authenticatedGpuSources = new WeakMap<File",
    );
    expect(uploadSource).toContain(
      "const hash = authenticatedSnapshot ? undefined : new AceIncrementalSha256()",
    );
    expect(uploadSource).toContain(
      'authentication: authenticatedSnapshot\n          ? "exact-immutable-file-proof-reused"',
    );
    expect(uploadSource).toContain(
      "Capture is observational and cannot alter package publication",
    );
    expect(tensorSource).toContain("onUploadTrace?.(Object.freeze");
    expect(workerSource).toContain('modelSource: "cache-only"');
    expect(workerSource).toContain("mappedAtCreationAttempted: false");
    expect(workerSource).toContain(
      "pipelineOrAuthenticationOverlapAttempted: false",
    );
    expect(workerSource).not.toContain("mappedAtCreation: true");
    expect(workerSource).not.toContain("captureTrace: true");
    expect(workerSource).not.toContain("opt-0018-dit-m2250");
    expect(pageSource).not.toContain("opt-0018-dit-m2250");
  });

  it("attributes construction, execution, finalization, and cleanup end to end", () => {
    expect(pipelineSource).toContain(
      "operation: `${owner}-gpu-api-construction-summary`",
    );
    expect(workerSource).toContain(
      '(["conditioning", "dit", "vae"] as const).map((owner)',
    );
    expect(workerSource).toContain(
      "`${owner}-gpu-api-construction-summary`",
    );
    for (const operation of [
      "conditioning-execution",
      "dit-eight-evaluation-execution-and-readback",
      "vae-decode-and-raw-stream",
      "normalize-encode-and-durable-wav-commit",
      "conditioning-destroy",
      "dit-drain-and-destroy",
      "vae-drain-and-destroy",
    ]) expect(pipelineSource).toContain(operation);
    expect(rawSource).toContain('schema: "ace-vae-raw-stream-capture-v1"');
    expect(wavSource).toContain('schema: "ace-vae-wav-write-capture-v1"');
    expect(audioSource).toContain(
      'schema: "ace-audio-output-transaction-capture-v1"',
    );
    expect(workerSource).toContain("await backend.releaseResult(result)");
    expect(workerSource).toContain("await backend.dispose()");
    expect(workerSource).toContain("cleanupCompletedAtEpochMilliseconds");
  });

  it("permits all package/device/GPU actions only after explicit UI authorization", () => {
    const prepareListener = pageSource.indexOf(
      'prepare.addEventListener("click"',
    );
    const workerCreation = pageSource.indexOf("const active = new Worker");
    const runListener = pageSource.indexOf('run.addEventListener("click"');
    const runMessage = pageSource.indexOf(
      'worker.postMessage({ type: "run", thermalGate })',
    );
    expect(prepareListener).toBeGreaterThan(0);
    expect(workerCreation).toBeGreaterThan(prepareListener);
    expect(runMessage).toBeGreaterThan(runListener);
    expect(workerSource.indexOf("backend.initialize(")).toBeGreaterThan(
      workerSource.indexOf("async function executeCapture"),
    );
    expect(htmlSource).toContain("It performs no package acquisition");
    expect(htmlSource).toContain("Run one capture-only request");
    expect(htmlSource).toContain("neither tries mapped-at-creation upload nor");
    expect(htmlSource).toContain("repeats the exact gate observations as its prefix");
  });
});
