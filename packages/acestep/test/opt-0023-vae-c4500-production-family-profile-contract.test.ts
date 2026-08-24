import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { createAceOpt0011LatentFixture } from
  "../benchmark/opt-0011-vae-fp16-storage-window.js";
import {
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
} from "../src/webgpu/vae-fp16-backend.js";
import {
  planAceVaeChunkedDecode,
} from "../src/webgpu/vae-chunks.js";
import {
  planAceOpt0011Fp16VaeChunkDispatches,
  type AceOpt0011Fp16VaeWindowTopology,
} from "../src/webgpu/vae-fp16-decoder.js";
import { ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256 } from
  "../src/webgpu/vae-fp16-profile.js";
import {
  OPT_0023_MAXIMUM_RECEIPT_BYTES,
  OPT_0023_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0023_THERMAL_COMMAND,
  OPT_0023_THERMAL_POLL_MILLISECONDS,
  OPT_0023_THERMAL_SOURCE,
  parseOpt0023RunIdentity,
  parseOpt0023ThermalCompletion,
  parseOpt0023ThermalGate,
  serializeOpt0023Failure,
} from "./browser/opt-0023-vae-c4500-production-family-profile.js";
import {
  Opt0023RawFile,
  settleOpt0023InitializationOwners,
  settleOpt0023CleanupOwners,
} from "./browser/opt-0023-vae-c4500-production-family-profile-worker.js";

const WORKER_PATH = new URL(
  "./browser/opt-0023-vae-c4500-production-family-profile-worker.ts",
  import.meta.url,
);
const PAGE_PATH = new URL(
  "./browser/opt-0023-vae-c4500-production-family-profile.ts",
  import.meta.url,
);
const HTML_PATH = new URL(
  "./browser/opt-0023-vae-c4500-production-family-profile.html",
  import.meta.url,
);
const RECORD_PATH = new URL(
  "../optimization/experiments/OPT-0023-vae-c4500-production-family-profile.md",
  import.meta.url,
);
const WORKER_SOURCE = readFileSync(WORKER_PATH, "utf8");
const PAGE_SOURCE = readFileSync(PAGE_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const RECORD_SOURCE = readFileSync(RECORD_PATH, "utf8");
const BACKEND_SOURCE = readFileSync(new URL(
  "../src/webgpu/vae-fp16-backend.ts",
  import.meta.url,
), "utf8");
const CHUNKS_SOURCE = readFileSync(new URL(
  "../src/webgpu/vae-chunks.ts",
  import.meta.url,
), "utf8");
const WAV_SOURCE = readFileSync(new URL(
  "../src/webgpu/vae-wav.ts",
  import.meta.url,
), "utf8");

describe("OPT-0023 current-production C4500 VAE browser profile", () => {
  it("freezes the exported xorshift C4500 latent fixture", () => {
    const fixture = createAceOpt0011LatentFixture(4_500);
    expect(fixture.byteLength).toBe(1_152_000);
    expect(fixture.byteOffset).toBe(0);
    expect(fixture.buffer.byteLength).toBe(fixture.byteLength);
    expect(sha256(fixture)).toBe(
      "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971",
    );
    const values = new Float32Array(fixture.buffer as ArrayBuffer);
    expect(values.length).toBe(288_000);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values.some((value) => value !== 0)).toBe(true);
    expect(WORKER_SOURCE).toContain(
      "createAceOpt0011LatentFixture(OPT_0023_LATENT_FRAMES)",
    );
    expect(WORKER_SOURCE).not.toContain("OPT_0008_DETERMINISTIC_LATENT_SEED");
  });

  it("independently freezes all twelve production chunk windows", () => {
    const plan = planAceVaeChunkedDecode(4_500, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    expect(plan).toMatchObject({
      batch: 1,
      latentFrames: 4_500,
      chunkFrames: 512,
      overlapFrames: 64,
      strideFrames: 384,
      hopLength: 1_920,
      sampleRateHz: 48_000,
      audioChannels: 2,
      outputAudioFrames: 8_640_000,
      outputInterleavedElements: 17_280_000,
      outputFloat32Bytes: 69_120_000,
      maximumWindowFrames: 512,
      maximumDecodedInterleavedElements: 1_966_080,
      maximumDecodedFloat32Bytes: 7_864_320,
      direct: false,
    });
    expect(plan.windows.map((window) => [
      window.coreStartLatentFrame,
      window.coreEndLatentFrame,
      window.windowStartLatentFrame,
      window.windowEndLatentFrame,
      window.latentWindowFrames,
      window.discardPrefixLatentFrames,
      window.discardSuffixLatentFrames,
      window.outputStartAudioFrame,
      window.outputAudioFrames,
      window.decodedAudioFrames,
    ])).toEqual([
      [0, 384, 0, 448, 448, 0, 64, 0, 737_280, 860_160],
      [384, 768, 320, 832, 512, 64, 64, 737_280, 737_280, 983_040],
      [768, 1_152, 704, 1_216, 512, 64, 64, 1_474_560, 737_280, 983_040],
      [1_152, 1_536, 1_088, 1_600, 512, 64, 64, 2_211_840, 737_280, 983_040],
      [1_536, 1_920, 1_472, 1_984, 512, 64, 64, 2_949_120, 737_280, 983_040],
      [1_920, 2_304, 1_856, 2_368, 512, 64, 64, 3_686_400, 737_280, 983_040],
      [2_304, 2_688, 2_240, 2_752, 512, 64, 64, 4_423_680, 737_280, 983_040],
      [2_688, 3_072, 2_624, 3_136, 512, 64, 64, 5_160_960, 737_280, 983_040],
      [3_072, 3_456, 3_008, 3_520, 512, 64, 64, 5_898_240, 737_280, 983_040],
      [3_456, 3_840, 3_392, 3_904, 512, 64, 64, 6_635_520, 737_280, 983_040],
      [3_840, 4_224, 3_776, 4_288, 512, 64, 64, 7_372_800, 737_280, 983_040],
      [4_224, 4_500, 4_160, 4_500, 340, 64, 0, 8_110_080, 529_920, 652_800],
    ]);
    expect(plan.windows.reduce(
      (sum, window) => sum + window.latentWindowFrames,
      0,
    )).toBe(5_908);
    expect(plan.windows.reduce(
      (sum, window) => sum +
        window.decodedAudioFrames * 2 * Float32Array.BYTES_PER_ELEMENT,
      0,
    )).toBe(90_746_880);
  });

  it("independently derives exact dispatch and homogeneous-family inventory", () => {
    const topology = planAceOpt0011Fp16VaeChunkDispatches(4_500, 512, 256);
    expect(topology.uniqueWindowFrames).toEqual([340, 448, 512]);
    expect(topology.topologies.map((entry) => ({
      inputFrames: entry.inputFrames,
      operations: entry.operationCount,
      graphQuanta: entry.graphQuantumCount,
      sequenceQuanta: entry.sequenceQuantumCount,
      decoderCommandBuffers: entry.decoderCommandBufferCountAtBatch8,
      totalCommandBuffers: entry.commandBufferCountAtBatch8,
      controlBytes: entry.dynamicControls.byteLength,
    }))).toEqual([
      { inputFrames: 340, operations: 88, graphQuanta: 5_241,
        sequenceQuanta: 5_242, decoderCommandBuffers: 656,
        totalCommandBuffers: 657, controlBytes: 1_341_712 },
      { inputFrames: 448, operations: 88, graphQuanta: 6_894,
        sequenceQuanta: 6_895, decoderCommandBuffers: 862,
        totalCommandBuffers: 863, controlBytes: 1_764_880 },
      { inputFrames: 512, operations: 88, graphQuanta: 7_854,
        sequenceQuanta: 7_855, decoderCommandBuffers: 982,
        totalCommandBuffers: 983, controlBytes: 2_010_640 },
    ]);
    expect(topology).toMatchObject({
      maximumFp16WorkspaceBytes: 251_658_240,
      uniqueDynamicControlBytes: 5_117_232,
      aggregateGraphQuantumCount: 90_675,
      aggregateSequenceQuantumCount: 90_687,
      aggregateCommandBufferCountAtBatch8: 11_350,
    });

    const byShape = topology.topologies.map(independentFamilyBatches);
    expect(byShape).toEqual([
      {
        inputFrames: 340,
        graph: [2_725, 546, 429, 1_079, 462],
        pure: [[327, 2_610], [54, 432], [50, 400], [106, 848], [45, 360]],
        mixed: [74, 592],
      },
      {
        inputFrames: 448,
        graph: [3_579, 714, 560, 1_429, 612],
        pure: [[436, 3_487], [75, 600], [66, 528], [153, 1_224], [63, 504]],
        mixed: [69, 552],
      },
      {
        inputFrames: 512,
        graph: [4_090, 819, 644, 1_611, 690],
        pure: [[500, 3_999], [89, 712], [78, 624], [170, 1_360], [74, 592]],
        mixed: [71, 568],
      },
    ]);
    expect(weightFamilyRows(byShape, [1, 1, 10])).toEqual({
      graph: [47_204, 9_450, 7_429, 18_618, 7_974],
      pure: [[5_763, 46_087], [1_019, 8_152], [896, 7_168],
        [1_959, 15_672], [848, 6_784]],
      mixed: [853, 6_824],
    });
  });

  it("freezes the exact steady memory and map/resource evidence contract", () => {
    const plan = planAceVaeChunkedDecode(4_500, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    expect(planAceOpt0011Fp16VaeChunkGpuBackendMemory(plan, 256)).toEqual({
      residentWeightBytes: 168_791_552,
      stagingInputBufferBytes: 131_072,
      decoderInputBufferBytes: 65_536,
      workspaceBufferBytes: 251_658_240,
      workspaceBufferCount: 3,
      outputBufferBytes: 7_864_320,
      readbackBufferBytes: 7_864_320,
      controlBufferBytes: 5_117_232,
      accountedGpuBytes: 944_808_752,
      latentSnapshotBytes: 1_152_000,
      maximumReturnedWindowBytes: 7_864_320,
      boundedCpuBytes: 9_016_320,
      maximumWindowFrames: 512,
      quantaPerCommandBuffer: 8,
    });
    expect(WORKER_SOURCE).toContain("OPT_0023_TRACKED_BUFFER_COUNT = 17");
    expect(WORKER_SOURCE).toContain("OPT_0023_TIMED_MAP_COUNT = 12");
    expect(WORKER_SOURCE).toContain("OPT_0023_TOTAL_MAP_COUNT = 13");
    expect(WORKER_SOURCE).toContain("newBufferCount: 0");
    expect(WORKER_SOURCE).toContain("mappedBufferCount");
    expect(WORKER_SOURCE).toContain("everyBufferDestroyedExactlyOnce");
    expect(WORKER_SOURCE).toContain("resourceCleanupIsBalanced(afterCleanup)");
    expect(WORKER_SOURCE).not.toContain("createObservedCommandEncoder");
    expect(WORKER_SOURCE).not.toContain("beginTimedTrace");
  });

  it("uses one C512 warmup then one unchanged full production stream", () => {
    const warmup = WORKER_SOURCE.indexOf("running the one untimed exact C512 warmup");
    const rawSetup = WORKER_SOURCE.indexOf(
      "prepared.raw = await Opt0023RawFile.create(prepared.plan)",
    );
    const preDispatchLaunchGate = WORKER_SOURCE.indexOf(
      "const preDispatchLaunchDelayMilliseconds",
      rawSetup,
    );
    const timed = WORKER_SOURCE.indexOf("const timedStarted = performance.now()");
    const stream = WORKER_SOURCE.indexOf("await streamAceVaeRawChunks(");
    const finish = WORKER_SOURCE.indexOf("prepared.raw.sink.finish()", stream);
    const ended = WORKER_SOURCE.indexOf("const timedCompleted = performance.now()", finish);
    const rawHash = WORKER_SOURCE.indexOf("await prepared.raw.hashBounded()", ended);
    expect(warmup).toBeGreaterThan(0);
    expect(rawSetup).toBeGreaterThan(warmup);
    expect(preDispatchLaunchGate).toBeGreaterThan(rawSetup);
    expect(timed).toBeGreaterThan(preDispatchLaunchGate);
    expect(timed).toBeGreaterThan(warmup);
    expect(stream).toBeGreaterThan(timed);
    expect(finish).toBeGreaterThan(stream);
    expect(ended).toBeGreaterThan(finish);
    expect(rawHash).toBeGreaterThan(ended);
    expect(WORKER_SOURCE).toContain("const warmupWindow = plan.windows[1]");
    expect(WORKER_SOURCE).toContain("summarizeWarmupOutput");
    expect(WORKER_SOURCE).toContain("prefillDispatchOrQueueWriteAdded: false");
    expect(WORKER_SOURCE).not.toContain("prefillCompleteOutput");
    expect(WORKER_SOURCE).not.toContain("queue.writeBuffer");
    expect(WORKER_SOURCE).toContain("windowWalls.push(Object.freeze(");
    expect(WORKER_SOURCE.slice(timed, ended)).not.toContain("postMessage");
    expect(WORKER_SOURCE.slice(timed, ended)).not.toContain("postProgress");
    expect(WORKER_SOURCE.slice(timed, ended)).not.toContain("JSON.stringify");
  });

  it("keeps the family callback append-only and validates after decode", () => {
    const callback = sourceFunction(WORKER_SOURCE, "accept(profile:", "validateSingleAfter(");
    expect(callback).toContain("this.values.push(profile)");
    expect(callback).not.toContain("validateFamilyProfile");
    expect(callback).not.toContain("postMessage");
    expect(callback).not.toContain("JSON.stringify");
    expect(WORKER_SOURCE).toContain("profiles.validateSingleAfter(profileBeforeWarmup");
    expect(WORKER_SOURCE).toContain("this.values.length = 0");
    const timedCompleted = WORKER_SOURCE.indexOf(
      "const timedCompleted = performance.now()",
    );
    const timedProfileValidation = WORKER_SOURCE.indexOf(
      "prepared.profiles.finishTimed(prepared.plan.windows)",
    );
    expect(timedProfileValidation).toBeGreaterThan(timedCompleted);
    expect(WORKER_SOURCE.slice(
      WORKER_SOURCE.indexOf("const timedStarted = performance.now()"),
      timedCompleted,
    )).not.toContain("validateSingleAfter");
    expect(BACKEND_SOURCE).toMatch(
      /try \{\s*this\.resources\.onFamilyProfile\?\.\(familyProfiler\.finish\(\)\);\s*\} catch \{/u,
    );
    expect(BACKEND_SOURCE).toContain(
      "Profiling is observational and cannot invalidate decoded audio.",
    );
    const progressClass = WORKER_SOURCE.indexOf("class ProgressCollector");
    const progressAccept = WORKER_SOURCE.indexOf("  accept(event:", progressClass);
    const progressBegin = WORKER_SOURCE.indexOf("  begin(expected:", progressAccept);
    const progressCallback = WORKER_SOURCE.slice(progressAccept, progressBegin);
    expect(WORKER_SOURCE.indexOf(
      "const OPT_0023_PROGRESS_COMMAND_BUFFER_TOTALS",
    )).toBeLessThan(progressClass);
    expect(WORKER_SOURCE.indexOf(
      "const OPT_0023_PROGRESS_DECODER_QUANTA",
    )).toBeLessThan(progressClass);
    expect(progressCallback).not.toContain("const timedTotals = [");
    expect(progressCallback).not.toContain("const timedQuanta = [");
    expect(progressCallback).not.toContain("Object.freeze(");
  });

  it("uses the production sink with a harness-owned temporary OPFS file", () => {
    expect(WORKER_SOURCE).toContain("new AceVaeRawF32FileSink(access, plan)");
    expect(WORKER_SOURCE).toContain("createSyncAccessHandle()");
    expect(WORKER_SOURCE).toContain("raw.f32.partial");
    expect(WORKER_SOURCE).toContain("this.access.getSize()");
    expect(WORKER_SOURCE).toContain("this.access.close()");
    expect(WORKER_SOURCE).toContain(
      "await this.root.removeEntry(this.directoryName, { recursive: true })",
    );
    expect(WORKER_SOURCE).toContain("new AceIncrementalSha256()");
    expect(WORKER_SOURCE).toContain("RAW_HASH_MAXIMUM_CHUNK_BYTES");
    expect(WORKER_SOURCE).toContain("file.slice(");
    expect(WORKER_SOURCE).toContain("await slice.arrayBuffer()");
    expect(WORKER_SOURCE).not.toContain("AceAudioOutputTransaction");
    expect(WORKER_SOURCE).not.toContain("file.arrayBuffer()");
    expect(WORKER_SOURCE).not.toContain("crypto.subtle.digest");
    expect(WORKER_SOURCE).not.toContain("writeNormalizedAceVae");
    expect(WAV_SOURCE).toContain("export class AceVaeRawF32FileSink");
    expect(WAV_SOURCE).toContain("this.file.flush()");
    expect(CHUNKS_SOURCE).toContain("await sink.writeCore(window, core)");
  });

  it("requests and authenticates the fixed32 OPT-0015 device/backend", () => {
    expect(WORKER_SOURCE).toContain('modelProfile: "raw-fp16"');
    expect(WORKER_SOURCE).toContain('requiredFeatures: ["subgroups"]');
    expect(WORKER_SOURCE).toContain("maxBufferSize: REQUIRED_WORKSPACE_BYTES");
    expect(WORKER_SOURCE).toContain(
      "maxStorageBufferBindingSize: REQUIRED_WORKSPACE_BYTES",
    );
    expect(WORKER_SOURCE).toContain("subgroupMinSize: 32");
    expect(WORKER_SOURCE).toContain("subgroupMaxSize: 32");
    expect(WORKER_SOURCE).toContain("context.device.features.has(\"shader-f16\")");
    expect(WORKER_SOURCE).toContain("context.device.features.has(\"subgroups\")");
    expect(WORKER_SOURCE).toContain(
      '"opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1"',
    );
    expect(WORKER_SOURCE).toContain(
      '"opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1"',
    );
    expect(WORKER_SOURCE).toContain(
      '"4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85"',
    );
    expect(WORKER_SOURCE).not.toContain("OPT_0022");
    const contextCreated = WORKER_SOURCE.indexOf(
      "const context = await requestAceWebGpuDevice",
    );
    const deviceTry = WORKER_SOURCE.indexOf("try {", contextCreated);
    const deviceGate = WORKER_SOURCE.indexOf("requireDevice(context)", contextCreated);
    expect(deviceTry).toBeGreaterThan(contextCreated);
    expect(deviceGate).toBeGreaterThan(deviceTry);
  });

  it("separates the full trace from a fresh post-warmup nominal gate", () => {
    const preparation = {
      warmupStartedAtEpochMilliseconds: 10_000,
      warmupCompletedAtEpochMilliseconds: 20_000,
    };
    const gateParameters = new URLSearchParams({
      thermalSource: OPT_0023_THERMAL_SOURCE,
      thermalCommand: OPT_0023_THERMAL_COMMAND,
      thermalTraceStartedAtEpochMilliseconds: "0",
      thermalTraceStartObservationIndex: "0",
      thermalGateStartedAtEpochMilliseconds: "21000",
      thermalGateStartObservationIndex: "21",
      thermalGateCompletedAtEpochMilliseconds: "51000",
      thermalGateCompletedObservationIndex: "51",
      thermalGateObservations: "31",
      thermalPollMilliseconds: String(OPT_0023_THERMAL_POLL_MILLISECONDS),
      thermalGateMaximumPollGapMilliseconds: "1007",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    });
    const gate = parseOpt0023ThermalGate(gateParameters, preparation, 51_001);
    expect(gate.durationMilliseconds).toBe(
      OPT_0023_MINIMUM_NOMINAL_MILLISECONDS,
    );
    expect(gate.traceStartObservationIndex).toBe(0);
    expect(gate.gateStartObservationIndex).toBe(21);
    expect(gate.command).toBe(OPT_0023_THERMAL_COMMAND);
    expect(gate.missingObservationCount).toBe(0);

    const completionParameters = new URLSearchParams({
      thermalTraceSha256: "a".repeat(64),
      thermalTraceByteLength: "8192",
      thermalTraceCompletedAtEpochMilliseconds: "82000",
      thermalTraceCompletedObservationIndex: "82",
      thermalTraceObservations: "83",
      thermalTraceMaximumPollGapMilliseconds: "1011",
      thermalTraceNonNominalObservations: "8",
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0",
      thermalTraceTransitionsJson:
        '[{"atEpochMilliseconds":70000,"level":1}]',
    });
    const completion = parseOpt0023ThermalCompletion(
      completionParameters,
      gate,
      81_000,
      82_001,
    );
    expect(completion).toMatchObject({
      traceStartedAtEpochMilliseconds: 0,
      gateStartObservationIndex: 21,
      gateObservationCount: 31,
      rawTraceByteLength: 8192,
      rawTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      command: OPT_0023_THERMAL_COMMAND,
      missingObservationCount: 0,
      initialLevel: 0,
      coversWarmupGateRunValidationAndCleanup: true,
      unchangedThermalRetryPerformed: false,
    });

    const reusedTraceStart = new URLSearchParams(gateParameters);
    reusedTraceStart.set("thermalGateStartObservationIndex", "0");
    reusedTraceStart.set("thermalGateCompletedObservationIndex", "30");
    expect(() => parseOpt0023ThermalGate(
      reusedTraceStart,
      preparation,
      51_001,
    )).toThrow(/indexes/);
    const beforeWarmup = new URLSearchParams(gateParameters);
    beforeWarmup.set("thermalGateStartedAtEpochMilliseconds", "19000");
    expect(() => parseOpt0023ThermalGate(
      beforeWarmup,
      preparation,
      51_001,
    )).toThrow(/fresh gate/);
    const pressured = new URLSearchParams(gateParameters);
    pressured.set("thermalGateNonNominalObservations", "1");
    expect(() => parseOpt0023ThermalGate(
      pressured,
      preparation,
      51_001,
    )).toThrow(/not nominal/);
    const missingGate = new URLSearchParams(gateParameters);
    missingGate.set("thermalGateMissingObservations", "1");
    expect(() => parseOpt0023ThermalGate(
      missingGate,
      preparation,
      51_001,
    )).toThrow(/not nominal/);
    const blankGateCompletion = new URLSearchParams(gateParameters);
    blankGateCompletion.set("thermalGateCompletedAtEpochMilliseconds", "");
    expect(() => parseOpt0023ThermalGate(
      blankGateCompletion,
      preparation,
      51_001,
    )).toThrow(/missing/);
    const wrongCommand = new URLSearchParams(gateParameters);
    wrongCommand.set("thermalCommand", "notifyutil -g wrong.key");
    expect(() => parseOpt0023ThermalGate(
      wrongCommand,
      preparation,
      51_001,
    )).toThrow(/source/);

    const blankTraceCompletion = new URLSearchParams(completionParameters);
    blankTraceCompletion.set("thermalTraceCompletedAtEpochMilliseconds", "");
    expect(() => parseOpt0023ThermalCompletion(
      blankTraceCompletion, gate, 81_000, 82_001,
    )).toThrow(/missing/);
    const missingTrace = new URLSearchParams(completionParameters);
    missingTrace.set("thermalTraceMissingObservations", "1");
    expect(() => parseOpt0023ThermalCompletion(
      missingTrace, gate, 81_000, 82_001,
    )).toThrow(/cadence/);
    const impossibleNonNominalCount = new URLSearchParams(completionParameters);
    impossibleNonNominalCount.set("thermalTraceNonNominalObservations", "53");
    expect(() => parseOpt0023ThermalCompletion(
      impossibleNonNominalCount, gate, 81_000, 82_001,
    )).toThrow(/cadence/);
    const pressureInsideNominalGate = new URLSearchParams(completionParameters);
    pressureInsideNominalGate.set(
      "thermalTraceTransitionsJson",
      '[{"atEpochMilliseconds":30000,"level":1}]',
    );
    expect(() => parseOpt0023ThermalCompletion(
      pressureInsideNominalGate, gate, 81_000, 82_001,
    )).toThrow(/transition/);
    const tooManyTransitions = new URLSearchParams(completionParameters);
    tooManyTransitions.set(
      "thermalTraceTransitionsJson",
      JSON.stringify(Array.from({ length: 83 }, (_, index) => ({
        atEpochMilliseconds: 52_001 + index,
        level: index % 2 === 0 ? 1 : 0,
      }))),
    );
    expect(() => parseOpt0023ThermalCompletion(
      tooManyTransitions, gate, 81_000, 82_001,
    )).toThrow(/cadence/);
    const initiallyPressured = new URLSearchParams(completionParameters);
    initiallyPressured.set("thermalTraceInitialLevel", "1");
    initiallyPressured.set(
      "thermalTraceTransitionsJson",
      '[{"atEpochMilliseconds":20000,"level":0}]',
    );
    expect(parseOpt0023ThermalCompletion(
      initiallyPressured, gate, 81_000, 82_001,
    ).initialLevel).toBe(1);
  });

  it("requires complete machine identity and bounded failure/result surfaces", () => {
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
    expect(parseOpt0023RunIdentity(valid)).toMatchObject({
      machineModel: "Mac15,12",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    const invalid = new URLSearchParams(valid);
    invalid.set("harnessCommit", "bad");
    expect(() => parseOpt0023RunIdentity(invalid)).toThrow(/40-hex/);
    const serialized = serializeOpt0023Failure(Object.assign(
      new Error("GPU failure"),
      { reason: "device-lost", oversized: "x".repeat(8_000) },
    ));
    expect(serialized).toMatchObject({ name: "Error", message: "GPU failure" });
    expect(JSON.stringify(serialized).length).toBeLessThan(16_384);
    expect(OPT_0023_MAXIMUM_RECEIPT_BYTES).toBe(65_536);
    expect(PAGE_SOURCE).toContain("rawJsonBytes > OPT_0023_MAXIMUM_RECEIPT_BYTES");
    expect(PAGE_SOURCE).toContain("window.__ACE_OPT0023_RESULT__ = receipt");
  });

  it("settles every fake owner, proves idempotence, and preserves cleanup errors", async () => {
    let backendDestroyCalls = 0;
    let rawCloseCalls = 0;
    let rawRemoveCalls = 0;
    let contextDestroyCalls = 0;
    const cleaned = await settleOpt0023CleanupOwners({
      backend: {
        destroy: async () => { backendDestroyCalls += 1; },
      },
      raw: {
        close: () => { rawCloseCalls += 1; },
        remove: async () => { rawRemoveCalls += 1; },
      },
      context: {
        destroy: () => { contextDestroyCalls += 1; },
        lost: Promise.resolve({ reason: "destroyed" }),
      },
    });
    expect({
      backendDestroyCalls,
      rawCloseCalls,
      rawRemoveCalls,
      contextDestroyCalls,
    }).toEqual({
      backendDestroyCalls: 2,
      rawCloseCalls: 1,
      rawRemoveCalls: 1,
      contextDestroyCalls: 1,
    });
    expect(cleaned).toMatchObject({
      backendDestroyCalledTwice: true,
      rawTemporaryEntryRemoved: true,
      deviceDestroyed: true,
      deviceLostReason: "destroyed",
    });
    await expect(settleOpt0023CleanupOwners({
      backend: { destroy: async () => { throw new Error("backend"); } },
      raw: {
        close: () => { throw new Error("close"); },
        remove: async () => { throw new Error("remove"); },
      },
      context: {
        destroy: () => { throw new Error("device"); },
        lost: Promise.reject(new Error("lost")),
      },
    })).rejects.toBeInstanceOf(AggregateError);
  });

  it("rolls back raw setup and awaits initialization-owner cleanup", async () => {
    const plan = planAceVaeChunkedDecode(4_500, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    let accessCloseCalls = 0;
    let removeCalls = 0;
    const access = {
      truncate: () => { throw new Error("sink truncate failed"); },
      close: () => { accessCloseCalls += 1; },
    } as unknown as FileSystemSyncAccessHandle;
    const fileHandle = {
      createSyncAccessHandle: async () => access,
    } as unknown as FileSystemFileHandle;
    const directory = {
      getFileHandle: async () => fileHandle,
    } as unknown as FileSystemDirectoryHandle;
    const root = {
      getDirectoryHandle: async () => directory,
      removeEntry: async (name: string, options?: FileSystemRemoveOptions) => {
        expect(name).toBe("ace-opt-0023-fixed-id");
        expect(options).toEqual({ recursive: true });
        removeCalls += 1;
      },
    } as unknown as FileSystemDirectoryHandle;
    await expect(Opt0023RawFile.create(plan, {
      getRoot: async () => root,
      randomUuid: () => "fixed-id",
    })).rejects.toThrow("sink truncate failed");
    expect(accessCloseCalls).toBe(1);
    expect(removeCalls).toBe(1);

    let phaseDestroyCalls = 0;
    let backendDestroyCalls = 0;
    let deviceDestroyCalls = 0;
    const initialized = await settleOpt0023InitializationOwners({
      phase: { destroy: () => { phaseDestroyCalls += 1; } },
      backend: { destroy: async () => { backendDestroyCalls += 1; } },
      context: {
        destroy: () => { deviceDestroyCalls += 1; },
        lost: Promise.resolve({ reason: "destroyed" }),
      },
    });
    expect({ phaseDestroyCalls, backendDestroyCalls, deviceDestroyCalls })
      .toEqual({ phaseDestroyCalls: 1, backendDestroyCalls: 2,
        deviceDestroyCalls: 1 });
    expect(initialized).toMatchObject({
      phaseDestroyed: true,
      deviceLostReason: "destroyed",
    });
  });

  it("authenticates the exact package and all controlling source surfaces", () => {
    expect(ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256).toBe(
      "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3",
    );
    expect(WORKER_SOURCE).toContain("ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256");
    expect(WORKER_SOURCE).toContain("ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT");
    expect(WORKER_SOURCE).toContain("ACE_OPT_0011_VAE_FP16_WEIGHT_FILES");
    expect(WORKER_SOURCE).toContain("files.length !== 7");
    expect(WORKER_SOURCE).toContain("residentBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES");
    expect(WORKER_SOURCE).toContain(
      '"9a3e37d48c75139f98bfb9958f35061247b56da6"',
    );
    expect(WORKER_SOURCE).toContain(
      '"c87b472ed544ba3a0177c41ba7e66bb33cb4c9ececb88be64da0e4d2845a5ee1"',
    );
    // The worker/receipt above retain the immutable registration-time record
    // identity. The current record becomes mutable post-execution governance;
    // its closeout bytes are authenticated by opt-0023-result-contract.test.ts.
    expect(RECORD_SOURCE).toContain("## Result — positive measurement, benchmark-only");
    const registered = {
      "src/webgpu/vae-fp16-backend.ts":
        "47b2c24355f1e3d6e110a8d07d18e5b2686849685bad39b75c2a5689b5ea0447",
      "src/webgpu/vae-fp16-decoder.ts":
        "15e3a98ceec3b6d169d7ca78b921d2abde34c43ee0d3abf81b4314be0b483964",
      "src/webgpu/vae-fp16-profile.ts":
        "7ba8c89844b2f60217eb1b61a30a143628efe95d546b8f4c584288e381b5a362",
      "src/webgpu/vae-chunks.ts":
        "0059b79fcce3ad5679d3c05f5f3da1954db80194775540521d68002253f5baaf",
      "src/runtime/webgpu-pipeline.ts":
        "41acd296fee4a7e5c6613b1e34b798c8743aa717a5164761f60bd91ce1eaf2e0",
    };
    for (const [path, expected] of Object.entries(registered)) {
      expect(sha256(readFileSync(new URL(`../${path}`, import.meta.url)))).toBe(expected);
      expect(WORKER_SOURCE).toContain(`"${path}"`);
      expect(WORKER_SOURCE).toContain(`"${expected}"`);
    }
    for (const path of [
      "benchmark/opt-0011-vae-fp16-storage-window.ts",
      "src/model/acquire.ts",
      "src/model/cache.ts",
      "src/model/gpu-tensors.ts",
      "src/model/gpu-upload.ts",
      "src/model/manifest.ts",
      "src/model/package.ts",
      "src/model/sha256.ts",
      "src/model/strict-json.ts",
      "src/runtime/scheduler.ts",
      "src/webgpu/capabilities.ts",
      "src/webgpu/device.ts",
      "src/webgpu/scoped-buffer-allocation.ts",
      "src/webgpu/vae-wav.ts",
      "src/webgpu/kernels/vae-primitives.ts",
      "src/webgpu/kernels/vae-conv1d-fp16.ts",
      "src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts",
      "src/webgpu/kernels/vae-conv-transpose1d-fp16.ts",
      "src/webgpu/kernels/vae-pointwise-fp16.ts",
      "src/webgpu/kernels/vae-snake-fp16.ts",
      "test/browser/opt-0023-vae-c4500-production-family-profile-worker.ts",
      "test/browser/opt-0023-vae-c4500-production-family-profile.ts",
      "test/browser/opt-0023-vae-c4500-production-family-profile.html",
      "test/opt-0023-vae-c4500-production-family-profile-contract.test.ts",
    ]) expect(WORKER_SOURCE).toContain(`"${path}"`);
  });

  it("keeps scope exact, receipt semantics honest, and browser entry dedicated", () => {
    expect(WORKER_SOURCE).toContain(
      "combinedBackendCreateWallMilliseconds",
    );
    expect(WORKER_SOURCE).not.toContain(
      "backendConstructionCompilationAndAllocationWallMilliseconds",
    );
    expect(WORKER_SOURCE).toContain(
      'schema: "ace-opt-0023-vae-c4500-production-family-profile-v2"',
    );
    expect(WORKER_SOURCE).not.toContain("backendCompileWallMilliseconds");
    expect(WORKER_SOURCE).toContain("withinDecodeNonfamilyResidualMilliseconds");
    expect(WORKER_SOURCE).toContain("outsideDecodeStreamResidualMilliseconds");
    expect(WORKER_SOURCE).toContain("noReadbackOrResidualProration: true");
    expect(WORKER_SOURCE).toContain("finalPageJoinAndSerializationExcluded: true");
    expect(WORKER_SOURCE).toContain("finalPageSerializationWallMilliseconds: null");
    expect(WORKER_SOURCE).toContain('thermalClassification: "pending-external-artifact-join"');
    expect(WORKER_SOURCE).toContain(
      "sameTraceMustCoverWarmupGateRunValidationAndCleanup: true",
    );
    expect(WORKER_SOURCE).not.toContain("sameTraceMustCoverPreparation");
    expect(WORKER_SOURCE).toContain("extraCancellationRunPerformed: false");
    expect(WORKER_SOURCE).not.toMatch(
      /createAceWebGpuPipelineBackend|createAcePlanner|loadPlanner|createAceDit|normalize_audio/u,
    );
    expect(HTML_SOURCE).toContain("C448 + 10×C512 + C340");
    expect(HTML_SOURCE).toContain("There is no thermal retry");
    expect(HTML_SOURCE).toContain("thermalTraceStartObservationIndex");
    expect(HTML_SOURCE).toContain("thermalGateStartObservationIndex");
    expect(HTML_SOURCE).toContain("thermalTraceCompletedObservationIndex");
    expect(HTML_SOURCE).toContain("thermalTraceByteLength");
    expect(HTML_SOURCE).toContain("thermalGateMissingObservations");
    expect(HTML_SOURCE).toContain("thermalTraceMissingObservations");
    expect(HTML_SOURCE).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(HTML_SOURCE).toContain("absolute target epoch");
    expect(HTML_SOURCE).toContain("actual observed epoch");
    expect(HTML_SOURCE).toContain(
      './opt-0023-vae-c4500-production-family-profile.ts',
    );
    expect(PAGE_SOURCE).toContain("const worker = new Worker(");
    expect(PAGE_SOURCE).toContain('{ type: "module" }');
    expect(PAGE_SOURCE).toContain("run.disabled = false");
    expect(PAGE_SOURCE).toContain('worker.postMessage({ type: "dispose" })');
    expect(PAGE_SOURCE).toContain('thermalClassification: thermal.nonNominalObservationCount === 0');
    expect(PAGE_SOURCE).toContain("externalThermalArtifactJoined: true");
    expect(PAGE_SOURCE).toContain("rawTraceByteLength");
    expect(PAGE_SOURCE).toContain("coversWarmupGateRunValidationAndCleanup");
    expect(PAGE_SOURCE).not.toContain("coversPreparationRunValidationAndCleanup");
    expect(PAGE_SOURCE).not.toContain("setBlankCompletionToNow");
    expect(PAGE_SOURCE).toContain(
      "finalPageJoinAndSerializationExcludedFromAuthoritativeTiming: true",
    );
    const finalJoinStart = PAGE_SOURCE.indexOf(
      'finalize.addEventListener("click"',
    );
    const finalJoinEnd = PAGE_SOURCE.indexOf(
      'worker.addEventListener("message"',
      finalJoinStart,
    );
    const finalJoinSource = PAGE_SOURCE.slice(finalJoinStart, finalJoinEnd);
    expect(finalJoinSource).toContain("settled = true");
    expect(finalJoinSource).toContain("worker.terminate()");
    expect(finalJoinSource).toContain("}, { once: true });");
    expect(PAGE_SOURCE).not.toContain("location.reload");
  });
});

type FamilyName = "k7" | "k1" | "transpose" | "snake" | "add";
const FAMILY_ORDER: readonly FamilyName[] = [
  "k7", "k1", "transpose", "snake", "add",
];

function independentFamilyBatches(
  topology: AceOpt0011Fp16VaeWindowTopology,
): Readonly<{
  inputFrames: number;
  graph: readonly number[];
  pure: readonly (readonly [number, number])[];
  mixed: readonly [number, number];
}> {
  const graphFamilies = topology.cooperativePlan.quanta.map((quantum) => {
    const operation = topology.plan.operations[quantum.operationIndex]!;
    switch (operation.kind) {
      case "conv1d":
        if (operation.shape.kernelSize === 7) return "k7" as const;
        if (operation.shape.kernelSize === 1) return "k1" as const;
        throw new Error("unexpected Conv1D kernel");
      case "conv-transpose1d": return "transpose" as const;
      case "snake": return "snake" as const;
      case "add": return "add" as const;
    }
  });
  const graph = FAMILY_ORDER.map((family) =>
    graphFamilies.filter((value) => value === family).length
  );
  const sequence: readonly (FamilyName | null)[] = [null, ...graphFamilies];
  const pure = FAMILY_ORDER.map((family): readonly [number, number] => {
    let batches = 0;
    let quanta = 0;
    for (let offset = 0; offset < sequence.length; offset += 8) {
      const batch = sequence.slice(offset, offset + 8);
      if (batch.every((value) => value === family)) {
        batches += 1;
        quanta += batch.length;
      }
    }
    return [batches, quanta];
  });
  let mixedBatches = 0;
  let mixedQuanta = 0;
  for (let offset = 0; offset < sequence.length; offset += 8) {
    const batch = sequence.slice(offset, offset + 8);
    if (!FAMILY_ORDER.some((family) => batch.every((value) => value === family))) {
      mixedBatches += 1;
      mixedQuanta += batch.length;
    }
  }
  return Object.freeze({
    inputFrames: topology.inputFrames,
    graph: Object.freeze(graph),
    pure: Object.freeze(pure),
    mixed: Object.freeze([mixedBatches, mixedQuanta] as const),
  });
}

function weightFamilyRows(
  rows: readonly ReturnType<typeof independentFamilyBatches>[],
  multiplicities: readonly number[],
): Readonly<{
  graph: readonly number[];
  pure: readonly (readonly [number, number])[];
  mixed: readonly [number, number];
}> {
  return Object.freeze({
    graph: Object.freeze(FAMILY_ORDER.map((_, family) => rows.reduce(
      (sum, row, index) => sum + row.graph[family]! * multiplicities[index]!,
      0,
    ))),
    pure: Object.freeze(FAMILY_ORDER.map((_, family) => Object.freeze([
      rows.reduce((sum, row, index) =>
        sum + row.pure[family]![0] * multiplicities[index]!, 0),
      rows.reduce((sum, row, index) =>
        sum + row.pure[family]![1] * multiplicities[index]!, 0),
    ] as const))),
    mixed: Object.freeze([
      rows.reduce((sum, row, index) =>
        sum + row.mixed[0] * multiplicities[index]!, 0),
      rows.reduce((sum, row, index) =>
        sum + row.mixed[1] * multiplicities[index]!, 0),
    ] as const),
  });
}

function sourceFunction(source: string, start: string, end: string): string {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  if (first < 0 || last < 0) throw new Error(`source slice ${start} is absent`);
  return source.slice(first, last);
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
