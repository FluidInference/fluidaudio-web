import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { OPT_0066_NUMERICAL_ENVELOPE } from
  "./browser/opt-0066-vae-dual-k4-quality-c512-contract.js";
import {
  OPT_0054_0066_ARM_ORDER,
  OPT_0054_0066_CANDIDATE_CHUNK_FRAMES,
  OPT_0054_0066_CONTROL_CHUNK_FRAMES,
  OPT_0054_0066_CONTROL_RAW_SHA256,
  OPT_0054_0066_EXPERIMENT_ASSOCIATIONS,
  OPT_0054_0066_LATENT_BYTES,
  OPT_0054_0066_LATENT_ELEMENTS,
  OPT_0054_0066_LATENT_FRAMES,
  OPT_0054_0066_LATENT_SHA256,
  OPT_0054_0066_LONG_PROTOCOL_ID,
  OPT_0054_0066_LONG_SCHEMA,
  OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
  OPT_0054_0066_OUTPUT_BYTES,
  OPT_0054_0066_OUTPUT_ELEMENTS,
  OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
  OPT_0054_0066_RAW_BLOCK_BYTES,
  compareOpt00540066Waveforms,
  planOpt00540066LongGate,
  reconcileOpt00540066CapturedWindowTopology,
} from "./browser/opt-0054-0066-vae-revision7-c2378-long-contract.js";

const WORKER_SOURCE = readFileSync(new URL(
  "./browser/opt-0054-0066-vae-revision7-c2378-long-worker.ts",
  import.meta.url,
), "utf8");
const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0054-0066-vae-revision7-c2378-long.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0054-0066-vae-revision7-c2378-long.html",
  import.meta.url,
), "utf8");

describe("OPT-0054/0066 revision-7 C2378 long-waveform browser gate", () => {
  it("pins the authenticated C4500 identity and isolated protocol", () => {
    expect(OPT_0054_0066_LONG_SCHEMA).toBe(
      "ace-opt-0054-0066-vae-revision7-c2378-c4500-long-waveform-v1",
    );
    expect(OPT_0054_0066_LONG_PROTOCOL_ID).toBe(
      "opt-0054-0066-rev6-c512-rev7-c2378-candidate-repeat-v1",
    );
    expect(OPT_0054_0066_EXPERIMENT_ASSOCIATIONS).toEqual([
      "OPT-0054",
      "OPT-0066",
    ]);
    expect(OPT_0054_0066_ARM_ORDER).toEqual([
      "rev7-cancellation-probe",
      "rev6-opt0028-c512-control",
      "rev7-opt0066-c2378-candidate",
      "rev7-opt0066-c2378-repeat",
    ]);
    expect(OPT_0054_0066_LATENT_FRAMES).toBe(4_500);
    expect(OPT_0054_0066_LATENT_ELEMENTS).toBe(288_000);
    expect(OPT_0054_0066_LATENT_BYTES).toBe(1_152_000);
    expect(OPT_0054_0066_LATENT_SHA256).toBe(
      "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971",
    );
    expect(OPT_0054_0066_CONTROL_RAW_SHA256).toBe(
      "fb8aae85e21a8a93b39baf738d0f2577e18134c627a05562b710341d0d590f7c",
    );
    expect(OPT_0054_0066_OUTPUT_ELEMENTS).toBe(17_280_000);
    expect(OPT_0054_0066_OUTPUT_BYTES).toBe(69_120_000);
    expect(OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER).toBe(64);
    expect(OPT_0054_0066_RAW_BLOCK_BYTES).toBe(1_048_576);
  });

  it("pins exact C512 and C2378 coverage, topology, and bounded memory", () => {
    const gate = planOpt00540066LongGate();

    expect(OPT_0054_0066_CONTROL_CHUNK_FRAMES).toBe(512);
    expect(OPT_0054_0066_CANDIDATE_CHUNK_FRAMES).toBe(2_378);
    expect(gate.control.windows).toHaveLength(12);
    expect(gate.controlSeams).toEqual([
      384, 768, 1_152, 1_536, 1_920, 2_304, 2_688, 3_072, 3_456,
      3_840, 4_224,
    ]);
    expect(gate.candidate.windows.map((window) => ({
      latent: [
        window.windowStartLatentFrame,
        window.windowEndLatentFrame,
      ],
      core: [
        window.coreStartLatentFrame,
        window.coreEndLatentFrame,
      ],
      prefix: window.discardPrefixLatentFrames,
      suffix: window.discardSuffixLatentFrames,
    }))).toEqual([
      { latent: [0, 2_314], core: [0, 2_250], prefix: 0, suffix: 64 },
      { latent: [2_186, 4_500], core: [2_250, 4_500], prefix: 64, suffix: 0 },
    ]);
    expect(gate.candidateSeams).toEqual([2_250]);
    expect(gate.controlTopology).toEqual({
      maximumWindowFrames: 512,
      uniqueWindowFrames: [340, 448, 512],
      windowCount: 12,
      decodedLatentFrames: 5_908,
      aggregateGraphQuantumCount: 90_675,
      aggregateSequenceQuantumCount: 90_687,
      decoderCommandBufferCount: 1_420,
      readbackCommandBufferCount: 12,
      totalCommandBufferCount: 1_432,
      decoderRequestedCooperativeIdleMs: 1_420,
      betweenWindowRequestedCooperativeIdleMs: 11,
      totalRequestedCooperativeIdleMs: 1_431,
    });
    expect(gate.candidateTopology).toEqual({
      maximumWindowFrames: 2_378,
      uniqueWindowFrames: [2_314],
      windowCount: 2,
      decodedLatentFrames: 4_628,
      aggregateGraphQuantumCount: 70_994,
      aggregateSequenceQuantumCount: 70_996,
      decoderCommandBufferCount: 1_110,
      readbackCommandBufferCount: 2,
      totalCommandBufferCount: 1_112,
      decoderRequestedCooperativeIdleMs: 1_110,
      betweenWindowRequestedCooperativeIdleMs: 1,
      totalRequestedCooperativeIdleMs: 1_111,
    });
    expect(gate.controlMemory).toMatchObject({
      residentWeightBytes: 168_791_552,
      workspaceBufferBytes: 251_658_240,
      workspaceBufferCount: 3,
      controlBufferBytes: 5_117_232,
      accountedGpuBytes: 944_808_752,
      boundedCpuBytes: 9_016_320,
      maximumWindowFrames: 512,
      quantaPerCommandBuffer: 64,
    });
    expect(gate.candidateMemory).toMatchObject({
      residentWeightBytes: 168_791_552,
      workspaceBufferBytes: 1_168_834_560,
      workspaceBufferCount: 3,
      controlBufferBytes: 9_087_248,
      accountedGpuBytes: 3_758_347_792,
      boundedCpuBytes: 37_678_080,
      maximumWindowFrames: 2_378,
      quantaPerCommandBuffer: 64,
    });
    expect(gate.candidateMemory.accountedGpuBytes).toBeLessThan(
      OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES,
    );
  });

  it("streams complete stereo numerical gates and rejects hidden failures", () => {
    const control = new Float32Array([
      0.25, -0.5, 0.75, -1, 0.5, -0.25, -0.75, 1,
    ]);
    const exact = compareOpt00540066Waveforms(
      control,
      new Float32Array(control),
    );
    expect(exact.joint).toMatchObject({
      count: 8,
      finiteCount: 8,
      nonFiniteCount: 0,
      nrmse: 0,
      snrDb: Number.POSITIVE_INFINITY,
      pearson: 1,
      maximumAbsoluteError: 0,
      relativeEnergyDrift: 0,
      finite: true,
      passed: true,
    });
    expect(exact.left.count).toBe(4);
    expect(exact.right.count).toBe(4);

    const inverted = compareOpt00540066Waveforms(
      control,
      Float32Array.from(control, (value) => -value),
    );
    expect(inverted.joint.pearson).toBeCloseTo(-1, 12);
    expect(inverted.joint.passed).toBe(false);

    const nonFinite = new Float32Array(control);
    nonFinite[3] = Number.NaN;
    expect(compareOpt00540066Waveforms(control, nonFinite).joint).toMatchObject({
      count: 8,
      finiteCount: 7,
      nonFiniteCount: 1,
      finite: false,
      passed: false,
    });
    expect(() => compareOpt00540066Waveforms(
      new Float32Array(2),
      new Float32Array(4),
    )).toThrow(/equal stereo blocks/);
    expect(OPT_0066_NUMERICAL_ENVELOPE).toMatchObject({
      nrmseMaximum: 0.003,
      snrMinimumDb: 50,
      pearsonMinimum: 0.9999,
    });
  });

  it("reconciles the captured ingress quantum with the complete kernel sequence", () => {
    const authoritative = reconcileOpt00540066CapturedWindowTopology({
      inputFrames: 512,
      operationCount: 88,
      graphQuantumCount: 7_854,
      sequenceQuantumCount: 7_855,
      kernelQuantumCounts: {
        ingress: 1,
        decoder: 7_854,
      },
      operationQuantumCounts: [
        { quantumCount: 4_000 },
        { quantumCount: 3_854 },
      ],
    });
    expect(authoritative).toEqual({
      inputFrames: 512,
      operationCount: 88,
      graphQuantumCount: 7_854,
      sequenceQuantumCount: 7_855,
      ingressQuantumCount: 1,
      kernelQuantumTotal: 7_855,
      operationQuantumTotal: 7_854,
      passed: true,
    });
    expect(reconcileOpt00540066CapturedWindowTopology({
      ...authoritative,
      kernelQuantumCounts: { decoder: 7_854 },
      operationQuantumCounts: [{ quantumCount: 7_854 }],
    }).passed).toBe(false);
  });

  it("statically freezes authentication, ownership, cancellation, and cleanup", () => {
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-experimental/manifest.json"',
    );
    expect(WORKER_SOURCE).toContain(
      '"/model/files-fp16-vae-revision7-experimental/manifest.json"',
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id",
    );
    expect(WORKER_SOURCE).toContain(
      "ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id",
    );
    expect(WORKER_SOURCE).toContain("class DeviceResourceAudit");
    expect(WORKER_SOURCE).toContain(
      "reconcileOpt00540066CapturedWindowTopology",
    );
    expect(WORKER_SOURCE).toContain("backendValidation: execution.backendValidation");
    expect(WORKER_SOURCE).not.toContain(
      "window.kernelQuantumTotal !== window.graphQuantumCount",
    );
    expect(WORKER_SOURCE).toContain(
      'throw new Error("OPT-0054/0066 refused co-resident VAE package owners")',
    );
    expect(WORKER_SOURCE).toContain("activeMaximumLiveBytes ===");
    expect(WORKER_SOURCE).toContain("everyBufferDestroyedExactlyOnce");
    expect(WORKER_SOURCE).toContain("const first = backend.destroy(reason)");
    expect(WORKER_SOURCE).toContain("const second = backend.destroy(reason)");
    expect(WORKER_SOURCE).toContain("await input.context.device.queue.onSubmittedWorkDone()");
    expect(WORKER_SOURCE).toContain("runCancellationProbe");
    expect(WORKER_SOURCE).toContain("abortRequestedAtProgressEvent !== 1");
    expect(WORKER_SOURCE).toContain("callbackCountAfterAbort !== 0");
    expect(WORKER_SOURCE).toContain("disposeAfterActive");
    expect(WORKER_SOURCE).toContain("createSyncAccessHandle()");
    expect(WORKER_SOURCE).toContain("new AceVaeRawF32FileSink");
    expect(WORKER_SOURCE).toContain("new Opt00540066StereoMetricAccumulator()");
    expect(WORKER_SOURCE).toContain("new Uint32Array(controlBytes.buffer)");
    expect(WORKER_SOURCE).toContain("await artifact.remove()");
    expect(WORKER_SOURCE).toContain("context?.destroy()");
    expect(WORKER_SOURCE).toContain("performanceTimingPerformed: false");
    expect(WORKER_SOURCE).toContain("seamNeighborhoodMetricsDescriptiveOnly: true");
    expect(WORKER_SOURCE).toContain("productSelectionAuthorized: false");
    expect(WORKER_SOURCE).toContain("productionDefaultChanged: false");
    expect(WORKER_SOURCE).not.toContain("writeNormalizedAceVaeFloat32Wav");
    expect(WORKER_SOURCE).not.toContain("performance.now(");
  });

  it("keeps the page explicit, cancellable, and benchmark-only", () => {
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain('type: "run"');
    expect(PAGE_SOURCE).toContain('type: "cancel"');
    expect(PAGE_SOURCE).toContain('type: "dispose"');
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT00540066_RESULT__ = receipt");
    expect(HTML_SOURCE).toContain('id="run" type="button"');
    expect(HTML_SOURCE).toContain('id="cancel" type="button" disabled');
    expect(HTML_SOURCE).toContain("Nothing requests a GPU until");
    expect(HTML_SOURCE).toContain("exactly one VAE arm at a time");
    expect(HTML_SOURCE).toContain("Full pre-normalization stereo waveforms");
    expect(HTML_SOURCE).toContain("performs no timing");
    expect(HTML_SOURCE).toContain("production runtime and its C512 default untouched");
  });
});
