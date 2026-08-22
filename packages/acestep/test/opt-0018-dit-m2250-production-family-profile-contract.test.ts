import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  OPT_0018_FIXTURE_CONTRACT_SHA256,
  OPT_0018_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0018_THERMAL_POLL_MILLISECONDS,
  OPT_0018_THERMAL_SOURCE,
  createOpt0018Request,
  parseOpt0018RunIdentity,
  parseOpt0018ThermalCompletion,
  parseOpt0018ThermalGate,
  serializeOpt0018Failure,
} from "./browser/opt-0018-dit-m2250-production-family-profile.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0018-dit-m2250-production-family-profile-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0018-dit-m2250-production-family-profile.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0018-dit-m2250-production-family-profile.html",
  import.meta.url,
), "utf8");
const PIPELINE_SOURCE = readFileSync(new URL(
  "../src/runtime/webgpu-pipeline.ts",
  import.meta.url,
), "utf8");

describe("OPT-0018 current-production M2250 DiT browser profiler", () => {
  it("freezes the exact 180-second direct-instrumental request bytes", () => {
    const request = createOpt0018Request();
    expect(request).toEqual({
      generationProfile: "ace-turbo-v1-correctness",
      prompt:
        "Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.",
      lyrics: "",
      instrumental: true,
      durationSeconds: 180,
      seed: "0000000000c0ffee",
      planner: { mode: "disabled" },
      metadata: { bpm: 104, keyScale: "D minor", timeSignature: "4" },
    });
    expect(JSON.stringify(request)).toBe(OPT_0018_CANONICAL_REQUEST_JSON);
    expect(Buffer.byteLength(OPT_0018_CANONICAL_REQUEST_JSON)).toBe(
      OPT_0018_CANONICAL_REQUEST_BYTES,
    );
    expect(createHash("sha256").update(OPT_0018_CANONICAL_REQUEST_JSON).digest("hex"))
      .toBe(OPT_0018_CANONICAL_REQUEST_SHA256);
    expect(OPT_0018_FIXTURE_CONTRACT_SHA256).toBe(
      "63269e63a38b8baeb535d9d7847b7d549a6472d13997459105f722ecbc609e3e",
    );
    expect(WORKER_SOURCE).toContain(
      'effectiveInstrumentalLyrics: "[Instrumental]"',
    );
    expect(WORKER_SOURCE).toContain("dcw.lowBandScale !== 0.05");
    expect(WORKER_SOURCE).toContain("dcw.highBandScale !== 0.02");
    expect(WORKER_SOURCE).toContain("const TEXT_TOKEN_COUNT = 82");
    expect(WORKER_SOURCE).toContain("const LYRIC_TOKEN_COUNT = 15");
    expect(WORKER_SOURCE).toContain("const CONDITION_TOKEN_COUNT = 98");
    expect(WORKER_SOURCE).toContain(
      '"8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2"',
    );
    expect(WORKER_SOURCE).toContain(
      '"b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764"',
    );
    expect(WORKER_SOURCE).toContain(
      '"82 text tokens + 15 lyric tokens + 1 packed no-reference timbre row = 98"',
    );
    expect(HTML_SOURCE).toMatch(/one packed no-reference timbre\s+row: C98/u);
    expect(WORKER_SOURCE).toContain(
      '"cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb"',
    );
  });

  it("fails closed on the complete source and machine identity", () => {
    const valid = new URLSearchParams({
      coreCommit: "0123456789abcdef0123456789abcdef01234567",
      harnessCommit: "89abcdef0123456789abcdef0123456789abcdef",
      machineModel: "Mac15,12",
      osVersion: "macOS 26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    expect(parseOpt0018RunIdentity(valid)).toEqual({
      coreCommit: "0123456789abcdef0123456789abcdef01234567",
      harnessCommit: "89abcdef0123456789abcdef0123456789abcdef",
      machineModel: "Mac15,12",
      osVersion: "macOS 26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    for (const [name, value] of [
      ["coreCommit", "bad"],
      ["harnessCommit", "bad"],
      ["machineModel", " "],
      ["gpuCoreCount", "0"],
      ["memoryBytes", "9007199254740992"],
    ] as const) {
      const invalid = new URLSearchParams(valid);
      invalid.set(name, value);
      expect(() => parseOpt0018RunIdentity(invalid)).toThrow(/OPT-0018/);
    }
  });

  it("retains bounded DOM/WebGPU object failures and cleanup context", () => {
    const thrown = Object.create({
      name: "GPUValidationError",
      message: "Bind group entry is too small for the shader declaration",
    }) as Record<string, unknown>;
    Object.assign(thrown, {
      code: "GPU_VALIDATION",
      reason: { stage: "dit-denoise", kind: "validation" },
      commandId: "ace-dit-eval-3-layer-17-cross-attention",
      oversized: "x".repeat(8_000),
    });
    Object.defineProperty(thrown, "unreadable", {
      enumerable: true,
      get: () => {
        throw new Error("getter must not run");
      },
    });
    for (let index = 0; index < 20; index += 1) {
      thrown[`field${index}`] = index;
    }
    const serialized = serializeOpt0018Failure(thrown, {
      name: "OperationError",
      message: "device disposal failed",
      reason: "device-lost",
      liveBufferCount: 2,
    });
    expect(serialized).toMatchObject({
      name: "GPUValidationError",
      message: "Bind group entry is too small for the shader declaration",
      code: "GPU_VALIDATION",
      reason: { stage: "dit-denoise", kind: "validation" },
      cleanupError: {
        name: "OperationError",
        message: "device disposal failed",
        reason: "device-lost",
        ownFields: { liveBufferCount: 2 },
      },
    });
    const ownFields = serialized.ownFields as Record<string, unknown>;
    expect(ownFields.commandId).toBe(
      "ace-dit-eval-3-layer-17-cross-attention",
    );
    expect((ownFields.oversized as string).length).toBeLessThan(1_100);
    expect(ownFields.unreadable).toBe("[accessor]");
    expect(Object.keys(ownFields).length).toBeLessThanOrEqual(12);
    expect(JSON.stringify(serialized)).not.toContain("[object Object]");
    expect(WORKER_SOURCE).toContain(
      "serializeOpt0018Failure(error, cleanupError)",
    );
    expect(PAGE_SOURCE).toContain("finishFailure(message.error, true)");
  });

  it("runs after one nominal 30-second gate and accepts the later trace", () => {
    const readyAt = 1_999;
    const gateParameters = new URLSearchParams({
      thermalSource: OPT_0018_THERMAL_SOURCE,
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32000",
      thermalObservations: "31",
      thermalPollMilliseconds: String(OPT_0018_THERMAL_POLL_MILLISECONDS),
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    const gate = parseOpt0018ThermalGate(gateParameters, readyAt, 32_001);
    expect(gate.durationMilliseconds).toBe(
      OPT_0018_MINIMUM_NOMINAL_MILLISECONDS,
    );

    const completion = parseOpt0018ThermalCompletion(
      new URLSearchParams({
        thermalTraceSha256: "a".repeat(64),
        thermalTraceCompletedAtEpochMilliseconds: "53000",
        thermalTraceObservations: "52",
        thermalTraceMaximumPollGapMilliseconds: "1012",
        thermalTraceNonNominalObservations: "10",
        thermalTraceTransitionsJson:
          '[{"atEpochMilliseconds":40000,"level":1}]',
      }),
      gate,
      52_000,
      53_001,
    );
    expect(completion).toMatchObject({
      coversGateRunAndCleanup: true,
      nonNominalObservationCount: 10,
      unchangedThermalRetryPerformed: false,
    });

    const pressuredGate = new URLSearchParams(gateParameters);
    pressuredGate.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0018ThermalGate(pressuredGate, readyAt, 32_001))
      .toThrow(/not nominal/);
    const shortGate = new URLSearchParams(gateParameters);
    shortGate.set("thermalCompletedAtEpochMilliseconds", "31999");
    expect(() => parseOpt0018ThermalGate(shortGate, readyAt, 32_001))
      .toThrow(/30-second/);
  });

  it("uses one dedicated worker and the normal production initialization", () => {
    expect(PAGE_SOURCE).toContain("const worker = new Worker(");
    expect(PAGE_SOURCE).toContain('{ type: "module" }');
    expect(WORKER_SOURCE).toContain("createAceWebGpuPipelineBackend()");
    expect(WORKER_SOURCE).toContain("await backend.initialize(");
    expect(WORKER_SOURCE).toContain('modelSource: "cache-or-network"');
    expect(WORKER_SOURCE).toContain("/model/files-reference/manifest.json");
    expect(WORKER_SOURCE).toContain(
      "/model/files-fp16-dit-layer-mixed-experimental/manifest.json",
    );
    expect(WORKER_SOURCE).toContain(
      "/model/files-fp16-vae-experimental/manifest.json",
    );
    expect(WORKER_SOURCE).toContain(
      'modelProfile: "reference-bf16" as const',
    );
    expect(WORKER_SOURCE).toContain(
      'schedulingProfile: "cooperative" as const',
    );
    expect(WORKER_SOURCE).toContain(
      'runtimeProfile: "opt-0009-fp16-fp32-dense-v1" as const',
    );
    expect(WORKER_SOURCE).toContain("maxWindowFrames: 512 as const");
  });

  it("aborts only by private identity after detached latent and DiT cleanup", () => {
    expect(WORKER_SOURCE).toContain(
      "const PRIVATE_DIT_STOP = Object.freeze(",
    );
    expect(WORKER_SOURCE).toContain("validateCheckpoint(value);");
    expect(WORKER_SOURCE).toContain("controller.abort(PRIVATE_DIT_STOP);");
    expect(WORKER_SOURCE).toContain("error === PRIVATE_DIT_STOP");
    expect(WORKER_SOURCE).toContain("await session.backend.dispose();");
    const generate = WORKER_SOURCE.indexOf("await session.backend.generate(");
    const rejected = WORKER_SOURCE.indexOf(
      "generationRejectedAtEpochMilliseconds = Date.now()",
    );
    const dispose = WORKER_SOURCE.indexOf("await session.backend.dispose();");
    const publish = WORKER_SOURCE.indexOf(
      'schema: "ace-opt-0018-dit-m2250-production-family-profile-v1"',
    );
    expect(generate).toBeGreaterThan(0);
    expect(rejected).toBeGreaterThan(generate);
    expect(dispose).toBeGreaterThan(rejected);
    expect(publish).toBeGreaterThan(dispose);

    const hook = PIPELINE_SOURCE.indexOf("context.onDitCheckpoint(checkpoint)");
    const ditDestroy = PIPELINE_SOURCE.lastIndexOf("await dit!.destroy()", hook);
    const vaeAcquire = PIPELINE_SOURCE.indexOf(
      'kind: "vae"',
      hook,
    );
    const audioBegin = PIPELINE_SOURCE.indexOf(
      "await this.dependencies.beginAudio(",
      hook,
    );
    expect(ditDestroy).toBeGreaterThan(0);
    expect(hook).toBeGreaterThan(ditDestroy);
    expect(vaeAcquire).toBeGreaterThan(hook);
    expect(audioBegin).toBeGreaterThan(vaeAcquire);
  });

  it("retains exact 2553 graph plus one readback attribution", () => {
    expect(WORKER_SOURCE).toContain("const GRAPH_COMMAND_COUNT = 2_553");
    expect(WORKER_SOURCE).toContain("const TOTAL_COMMAND_COUNT = 2_554");
    expect(WORKER_SOURCE).toContain(
      'profile.schema !== "ace-dit-m2250-command-profile-v1"',
    );
    expect(WORKER_SOURCE).toContain(
      "descriptor.physicalIndex !== index",
    );
    expect(WORKER_SOURCE).toContain("profile.graphSubmitThroughDrainMs");
    expect(WORKER_SOURCE).toContain("profile.familyByBucket");
    expect(WORKER_SOURCE).toContain("profile.slowest");
    expect(WORKER_SOURCE).toContain(
      '"aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76"',
    );
    expect(WORKER_SOURCE).toContain(
      "const DESCRIPTOR_TABLE_SERIALIZED_BYTES = 1_869_566",
    );
    expect(WORKER_SOURCE).toContain("const DESCRIPTOR_MEMBER_COUNT = 6_833");
    expect(WORKER_SOURCE).toContain("const MIXED_COMMAND_COUNT = 1_344");
    expect(WORKER_SOURCE).toContain(
      "const TOTAL_SCHEDULED_MULTIPLY_ADDS = 26_840_955_355_136",
    );
    expect(WORKER_SOURCE).toContain(
      "details.conditionTokens !== CONDITION_TOKEN_COUNT",
    );
    expect(WORKER_SOURCE).toContain("descriptorRows: Object.freeze(");
    expect(WORKER_SOURCE).toContain(
      'descriptorRowsJson !== JSON.stringify(descriptorRows)',
    );
    expect(WORKER_SOURCE).toContain("crossFamilyCommandsAssignedToMixed: true");
    expect(WORKER_SOURCE).toContain("mixedCommandWallSplitOrEstimated: false");
    expect(WORKER_SOURCE).toContain(
      'details.schema !== "ace-dit-m2250-family-profile-receipt-v1"',
    );
    expect(WORKER_SOURCE).toContain(
      'diagnostic.stage !== "release-dit"',
    );
    expect(WORKER_SOURCE).toContain("commandTimingTuples:");
    expect(WORKER_SOURCE).toContain("descriptorRowsSha256:");
  });

  it("keeps per-drain capture bounded and authenticates controlling sources", () => {
    for (const source of [
      "golden/MANIFEST.json",
      "golden/fixtures/direct-instrumental-short.json",
      "src/api.ts",
      "src/model/graph-contract.ts",
      "src/model/manifest.ts",
      "src/model/sha256.ts",
      "src/runtime/generation-inputs.ts",
      "src/runtime/planner-coordinator.ts",
      "src/runtime/webgpu-pipeline.ts",
      "src/runtime/scheduler.ts",
      "src/tokenizer/conditioning-text.ts",
      "src/tokenizer/loader.ts",
      "src/tokenizer/qwen-bpe.ts",
      "src/webgpu/ace-dit-package.ts",
      "src/webgpu/capabilities.ts",
      "src/webgpu/conditioning-executor.ts",
      "src/webgpu/dit-backend.ts",
      "src/webgpu/dit-fp16-package.ts",
      "src/webgpu/dit-graph.ts",
      "src/webgpu/ace-dit.ts",
      "src/webgpu/qwen3.ts",
      "src/webgpu/semantic-conditioner.ts",
      "src/webgpu/kernels/attention.ts",
      "src/webgpu/kernels/correctness-utils.ts",
      "src/webgpu/kernels/dcw.ts",
      "src/webgpu/kernels/dit-dense-fp16.ts",
      "src/webgpu/kernels/dit-plumbing.ts",
      "src/webgpu/kernels/gemm.ts",
      "src/webgpu/kernels/rmsnorm.ts",
      "src/webgpu/kernels/rope.ts",
      "src/webgpu/kernels/subgroup-gemm.ts",
      "src/webgpu/kernels/transformer-plumbing.ts",
      "test/browser/opt-0018-dit-m2250-production-family-profile-worker.ts",
      "test/browser/opt-0018-dit-m2250-production-family-profile.ts",
      "test/browser/opt-0018-dit-m2250-production-family-profile.html",
      "test/opt-0018-dit-m2250-production-family-profile-contract.test.ts",
    ]) expect(WORKER_SOURCE).toContain(`"${source}"`);

    const progressCallback = WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("onProgress: (progress: AceGenerationProgress)"),
      WORKER_SOURCE.indexOf("onDiagnostic: (diagnostic: AceDiagnostic)"),
    );
    expect(progressCallback).not.toContain("postMessage");
    expect(progressCallback).not.toContain("JSON.stringify");
    expect(progressCallback).not.toContain("await");
    expect(WORKER_SOURCE).toContain(
      'perDrainLoggingSerializationPostMessageOrGpuAllocation: false',
    );
  });

  it("does not use stable Cancel or authorize product/integration claims", () => {
    expect(HTML_SOURCE).toMatch(/run\s+once and accept the disclosed result/u);
    expect(HTML_SOURCE).toMatch(
      /Keep the\s+same external logger polling through cleanup/u,
    );
    expect(HTML_SOURCE).toContain("private identity sentinel");
    expect(HTML_SOURCE).toContain("does not acquire VAE weights");
    expect(WORKER_SOURCE).toContain(
      "vaeSizedDeviceLimitsRequestedByNormalInitialization: true",
    );
    expect(WORKER_SOURCE).toContain(
      "vaeManifestAuthenticatedDuringInitialization: true",
    );
    expect(WORKER_SOURCE).toContain(
      "vaeSizedDeviceLimitsRequestedDuringInitialization: true",
    );
    expect(PAGE_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(WORKER_SOURCE).toContain("stableCancelMessageUsed: false");
    expect(WORKER_SOURCE).toContain("publicWorkerProtocolChanged: false");
    expect(WORKER_SOURCE).toContain("optimizationIntegrated: false");
    expect(WORKER_SOURCE).toContain("under60SecondClaim: false");
    expect(WORKER_SOURCE).not.toContain('type: "cancel"');
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0018_RESULT__ = receipt");
    expect(HTML_SOURCE).toContain("Download full authenticated raw receipt");
  });
});
