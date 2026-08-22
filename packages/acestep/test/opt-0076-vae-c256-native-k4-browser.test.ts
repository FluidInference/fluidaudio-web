import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

type SummarizeOpt0076Timing = typeof import(
  "./browser/opt-0076-vae-c256-native-k4.js"
)["summarizeOpt0076Timing"];
type Opt0076TimingInput = Parameters<SummarizeOpt0076Timing>[0];
type Opt0076TimingSample = Opt0076TimingInput["current"][number];

let summarizeOpt0076Timing: SummarizeOpt0076Timing;

const HARNESS_SOURCE = source(
  "./browser/opt-0076-vae-c256-native-k4.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0076-vae-c256-native-k4.html",
);
const EXPERIMENT_SOURCE = source(
  "../optimization/experiments/OPT-0076-vae-c256-native-k4-promotion.md",
);
const LEDGER_SOURCE = source("../optimization/LEDGER.md");

beforeAll(async () => {
  const elements = new Map<string, BrowserElementStub>();
  const element = (selector: string): BrowserElementStub => {
    const existing = elements.get(selector);
    if (existing !== undefined) return existing;
    const created: BrowserElementStub = {
      disabled: true,
      textContent: "",
      addEventListener: () => undefined,
    };
    elements.set(selector, created);
    return created;
  };
  vi.stubGlobal("document", {
    body: { dataset: {} },
    querySelector: (selector: string) => element(selector),
  });
  vi.stubGlobal("window", {
    addEventListener: () => undefined,
  });
  vi.stubGlobal("navigator", { gpu: undefined });
  ({ summarizeOpt0076Timing } = await import(
    "./browser/opt-0076-vae-c256-native-k4.js"
  ));
});

afterAll(() => {
  vi.unstubAllGlobals();
});

describe("OPT-0076 C256 native-K4 browser source contract", () => {
  it("covers six fixtures over every d1/d3/d9 first/interior/tail range", () => {
    expect(HARNESS_SOURCE).toContain("const INPUT_FRAMES = 2_400");
    expect(HARNESS_SOURCE).toContain("const CHANNELS = 256");
    expect(HARNESS_SOURCE).toContain("const OUTPUT_RANGE_ROWS = 64");
    for (const fixture of [
      'caseSpec(1, "production-local")',
      'caseSpec(3, "production-local")',
      'caseSpec(9, "production-local")',
      'caseSpec(1, "bounded-adversarial")',
      'caseSpec(3, "bounded-adversarial")',
      'caseSpec(9, "bounded-adversarial")',
    ]) expect(HARNESS_SOURCE).toContain(fixture);
    expect(HARNESS_SOURCE.match(
      /caseSpec\((?:1|3|9), "(?:production-local|bounded-adversarial)"\)/g,
    )).toHaveLength(6);
    expect(HARNESS_SOURCE).toContain(
      'probe("first", 0, OUTPUT_RANGE_ROWS)',
    );
    expect(HARNESS_SOURCE).toContain(
      'probe("interior", 1_152, OUTPUT_RANGE_ROWS)',
    );
    expect(HARNESS_SOURCE).toContain(
      'probe("tail", INPUT_FRAMES - 32, 32)',
    );
    expect(HARNESS_SOURCE).toContain(
      "base: firstOutputRow * CHANNELS",
    );
    expect(HARNESS_SOURCE).toContain(
      "count: outputRowCount * CHANNELS",
    );
    expect(HARNESS_SOURCE).toContain(
      "const expectedComparedU16Count = 6 * PROBES.reduce",
    );
    expect(6 * (64 + 64 + 32) * 256).toBe(245_760);
    expect(HARNESS_SOURCE).toContain("caseCount: allCorrectness.length");
    expect(HARNESS_SOURCE).toContain(
      "productionCaseCount: PRODUCTION_CASES.length",
    );
    expect(HARNESS_SOURCE).toContain(
      "adversarialCaseCount: ADVERSARIAL_CASES.length",
    );
    expect(HARNESS_SOURCE).toContain("probesPerCase: PROBES.length");
    expect(EXPERIMENT_SOURCE).toContain(
      "On d1/d3/d9 first/interior/tail C256 ranges",
    );
  });

  it("pins current, promoted-selector, and direct OPT-0024 identities", () => {
    expect(HARNESS_SOURCE).toContain(
      "ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      'current.owner !== "native-scalar-fp32"',
    );
    expect(HARNESS_SOURCE).toContain(
      "current.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      'candidate.owner !== "native-k4"',
    );
    expect(HARNESS_SOURCE).toContain(
      "candidate.literalSelectorKernelId !==\n      ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "candidate.kernelId !==\n      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "oracle.kernelId !==\n      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "selectorAndKernelIds: Object.freeze({",
    );
    expect(HARNESS_SOURCE).toContain(
      "currentKernel: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "candidateLiteralSelector:\n          ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "candidateAndOracleKernel:\n          ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "currentScalarKernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID",
    );
    expect(HARNESS_SOURCE).toContain(
      "candidateLiteralSelectorKernelId:\n      ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID",
    );
  });

  it("derives all selected ranges from the actual primitive planners", () => {
    expect(HARNESS_SOURCE).toContain(
      "planAceFp16VaeConv1dSubgroupRange",
    );
    expect(HARNESS_SOURCE).toContain(
      "planAceOpt0024VaeConv1dDirectDot4SubgroupRange",
    );
    expect(HARNESS_SOURCE).toContain(
      "const currentRange = planAceFp16VaeConv1dSubgroupRange(\n      currentDispatch.plan",
    );
    expect(HARNESS_SOURCE).toContain(
      "planAceOpt0024VaeConv1dDirectDot4SubgroupRange(\n        candidateDispatch.plan",
    );
    expect(HARNESS_SOURCE).toContain(
      "planAceOpt0024VaeConv1dDirectDot4SubgroupRange(\n        oracleDispatch.plan",
    );
    expect(HARNESS_SOURCE).toContain(
      "oracleDispatch.outputRange.base !== oracleRange.base",
    );
    expect(HARNESS_SOURCE).toContain(
      "const current = withRangeMetadata(currentDispatch, currentRange)",
    );
    expect(HARNESS_SOURCE).toContain(
      "const candidate = withRangeMetadata(candidateDispatch, candidateRange)",
    );
    expect(HARNESS_SOURCE).toContain(
      "const oracle = withRangeMetadata(oracleDispatch, oracleRange)",
    );
    for (const arm of ["current", "candidate", "oracle"]) {
      expect(HARNESS_SOURCE).toContain(
        `${arm}.outputRange?.base !== probeSpec.base`,
      );
      expect(HARNESS_SOURCE).toContain(
        `${arm}.outputRange?.count !== probeSpec.count`,
      );
    }
    expect(HARNESS_SOURCE).toContain(
      "dispatchRangeMetadata: Object.freeze({",
    );
    expect(HARNESS_SOURCE).toContain(
      "dispatch.outputRange?.base === probeSpec.base &&\n            dispatch.outputRange.count === probeSpec.count",
    );
  });

  it("requires raw-U16 oracle identity, deterministic reruns, and bounded writes", () => {
    expect(HARNESS_SOURCE).toContain(
      '"current", "oracle", "candidate", "candidate", "oracle"',
    );
    expect(HARNESS_SOURCE).toContain('"candidate-vs-oracle"');
    expect(HARNESS_SOURCE).toContain('"candidate-rerun"');
    expect(HARNESS_SOURCE).toContain('"oracle-rerun"');
    expect(HARNESS_SOURCE).toContain(
      "candidateOracle.differingU16Count === 0",
    );
    expect(HARNESS_SOURCE).toContain(
      "candidateRepeat.differingU16Count === 0",
    );
    expect(HARNESS_SOURCE).toContain(
      "oracleRepeat.differingU16Count === 0",
    );
    expect(HARNESS_SOURCE).toContain(
      "exactOracleU16Count === expectedComparedU16Count",
    );
    expect(HARNESS_SOURCE).toContain(
      "deterministicCandidateU16Count === expectedComparedU16Count",
    );
    expect(HARNESS_SOURCE).toContain(
      "deterministicOracleU16Count === expectedComparedU16Count",
    );

    expect(HARNESS_SOURCE).toContain(
      "const totalBytes = logicalBytes + 2 * STORAGE_GUARD_BYTES",
    );
    expect(HARNESS_SOURCE).toContain("words.fill(STORAGE_GUARD_U32)");
    expect(HARNESS_SOURCE).toContain(".fill(OUTPUT_PREFILL_QNAN_F16)");
    expect(HARNESS_SOURCE).toContain(
      "binding: Object.freeze({ buffer, offset: STORAGE_GUARD_BYTES",
    );
    expect(HARNESS_SOURCE).toContain("snapshot.nonFiniteCount === 0");
    expect(HARNESS_SOURCE).toContain("snapshot.qNaNPrefillCount === 0");
    expect(HARNESS_SOURCE).toContain("snapshot.prefixGuardIntact");
    expect(HARNESS_SOURCE).toContain("snapshot.suffixGuardIntact");
    expect(HARNESS_SOURCE).toContain("snapshot.adjacentBeforeIntact");
    expect(HARNESS_SOURCE).toContain("snapshot.adjacentAfterIntact");
    expect(HARNESS_SOURCE).toContain(
      "snapshot.outOfRangeWriteCount === 0",
    );
    expect(HARNESS_SOURCE).toContain("snapshot.tailWritten");
    expect(HARNESS_SOURCE).toContain(
      "if (!selected && logical[index] !== OUTPUT_PREFILL_QNAN_F16)",
    );
    expect(HARNESS_SOURCE).toContain("outOfRangeWriteCount += 1");
    expect(HARNESS_SOURCE).toContain("allOutputsFiniteAndComplete: complete");
    expect(HARNESS_SOURCE).toContain(
      "const guardsAndTailsIntact = snapshots.every((snapshot) =>",
    );
    expect(HARNESS_SOURCE).toContain(
      "const guardsAndTailsIntact = probeReceipts.every((entry) =>\n    entry[\"guardsAndTailsIntact\"] === true)",
    );
    expect(HARNESS_SOURCE).toContain(
      "guardsAndTailsIntact: allCorrectness.every((entry) =>\n        entry[\"guardsAndTailsIntact\"] === true)",
    );
    expect(HARNESS_SOURCE).not.toContain("guardsAndTailsIntact: true");
  });

  it("applies the unchanged numerical envelope per probe and in aggregate", () => {
    expect(HARNESS_SOURCE).toContain("const NRMSE_MAXIMUM = 0.001");
    expect(HARNESS_SOURCE).toContain("const SNR_MINIMUM_DB = 60");
    expect(HARNESS_SOURCE).toContain("const PEARSON_MINIMUM = 0.99999");
    expect(HARNESS_SOURCE).toContain(
      "const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM = 0.01",
    );
    expect(HARNESS_SOURCE).toContain(
      "const numerical = compareNumerics(\n      current.words,\n      oracle.words",
    );
    expect(HARNESS_SOURCE).toContain(
      "const numericalEnvelopePassed = numericalPassed(numericalMetrics)",
    );
    expect(HARNESS_SOURCE).toContain(
      "oracleRepeat.differingU16Count === 0 && numericalEnvelopePassed",
    );
    expect(HARNESS_SOURCE).toContain(
      "const numericalEnvelopePassed = numericalPassed(aggregateNumerics)",
    );
    expect(HARNESS_SOURCE).toContain("aggregate: aggregateNumerics");
    expect(HARNESS_SOURCE).toContain(
      "productionLocal: productionNumericalMetrics",
    );
    expect(HARNESS_SOURCE).toContain(
      "boundedAdversarial: adversarialNumericalMetrics",
    );
    expect(HARNESS_SOURCE).toContain(
      "allCorrectness.every((entry) => entry[\"passed\"] === true) &&\n      numericalEnvelopePassed",
    );
    for (const threshold of [
      "NRMSE at most `0.001`",
      "SNR at least `60 dB`",
      "Pearson at least `0.99999`",
      "relative maximum absolute error at most `0.01`",
    ]) expect(EXPERIMENT_SOURCE).toContain(threshold);
  });

  it("uses six balanced fenced timestamp rounds with nine dispatches each", () => {
    const rounds = section("const TIMING_ROUNDS", "const REQUIRED_SPEEDUP");
    expect(rounds.match(/\["current", "candidate"\]/g)).toHaveLength(3);
    expect(rounds.match(/\["candidate", "current"\]/g)).toHaveLength(3);

    expect(HARNESS_SOURCE).toContain('type: "timestamp"');
    expect(HARNESS_SOURCE).toContain("timestampWrites: { querySet:");
    expect(HARNESS_SOURCE).toContain("compositeDispatchesPerSample: 9");
    expect(HARNESS_SOURCE).toContain("aggregateDispatchesPerSample: 9");
    expect(HARNESS_SOURCE).toContain("dispatchCount: 9 as const");
    expect(HARNESS_SOURCE).toContain("commandBufferCount: 1 as const");
    expect(HARNESS_SOURCE).toContain("queueDrainCount: 1 as const");
    expect(HARNESS_SOURCE).toContain(
      "oneTimestampPairOnePassOneCommandBufferOneSubmitOneDrainPerSample: true",
    );
    expect(HARNESS_SOURCE).toContain(
      'composite: "d1/d3/d9 x first/interior/tail"',
    );

    const sample = section(
      "async function timeComposite",
      "export function summarizeOpt0076Timing",
    );
    expectOrdered(sample, [
      "beginComputePass({",
      "timestampWrites:",
      "encodeComposite(pass, prepared.productionCases, arm)",
      "pass.end()",
      "resolveQuerySet",
      "copyBufferToBuffer",
      "const command = encoder.finish()",
      "const submitAtPerformanceMilliseconds = performance.now()",
      "prepared.device.queue.submit([command])",
      "await prepared.device.queue.onSubmittedWorkDone()",
      "const fenceAtPerformanceMilliseconds = performance.now()",
      "await prepared.queryReadback.mapAsync(GPUMapMode.READ)",
    ]);
    expect(HARNESS_SOURCE).toContain("outputReadbackInsideTiming: false");
  });

  it("requires every paired GPU/wall win and twofold mean and median speedup", () => {
    expect(HARNESS_SOURCE).toContain("const REQUIRED_SPEEDUP = 2");
    expect(HARNESS_SOURCE).toContain(
      "everyPairedAggregateWallWin: pairedRounds.every((round) => round.wallWin)",
    );
    expect(HARNESS_SOURCE).toContain(
      "everyPairedAggregateGpuWin: pairedRounds.every((round) => round.gpuWin)",
    );
    for (const gate of [
      "meanWallSpeedupPassed: meanWallSpeedup >= REQUIRED_SPEEDUP",
      "medianWallSpeedupPassed: medianWallSpeedup >= REQUIRED_SPEEDUP",
      "meanGpuSpeedupPassed: meanGpuSpeedup >= REQUIRED_SPEEDUP",
      "medianGpuSpeedupPassed: medianGpuSpeedup >= REQUIRED_SPEEDUP",
    ]) expect(HARNESS_SOURCE).toContain(gate);
    expect(HARNESS_SOURCE).toContain(
      "passed: gates.everyPairedAggregateWallWin &&\n      gates.everyPairedAggregateGpuWin && gates.meanWallSpeedupPassed &&\n      gates.medianWallSpeedupPassed && gates.meanGpuSpeedupPassed &&\n      gates.medianGpuSpeedupPassed",
    );
    expect(EXPERIMENT_SOURCE).toContain(
      "Require every paired GPU and fenced-wall score to win and at least `2.0x`",
    );
  });

  it("passes the real timing evaluator only when all scores clear twofold", () => {
    const result = summarizeOpt0076Timing(timingInput(
      timingArm(30, 36),
      timingArm(10, 12),
    ));
    expect(result).toMatchObject({
      sampleCountPerArm: 6,
      aggregateDispatchesPerSample: 9,
      speedup: { meanWall: 3, medianWall: 3, meanGpu: 3, medianGpu: 3 },
      gates: {
        everyPairedAggregateWallWin: true,
        everyPairedAggregateGpuWin: true,
        meanWallSpeedupPassed: true,
        medianWallSpeedupPassed: true,
        meanGpuSpeedupPassed: true,
        medianGpuSpeedupPassed: true,
      },
      passed: true,
    });
  });

  it("fails the real timing evaluator on any paired wall or GPU loss", () => {
    const wallLossCandidate = timingArm(10, 10);
    wallLossCandidate[2] = timingSample(10, 41, 2);
    const wallLoss = summarizeOpt0076Timing(timingInput(
      timingArm(30, 40),
      wallLossCandidate,
    ));
    expect(wallLoss).toMatchObject({
      gates: {
        everyPairedAggregateWallWin: false,
        everyPairedAggregateGpuWin: true,
        meanWallSpeedupPassed: true,
        medianWallSpeedupPassed: true,
        meanGpuSpeedupPassed: true,
        medianGpuSpeedupPassed: true,
      },
      passed: false,
    });

    const gpuLossCandidate = timingArm(8, 10);
    gpuLossCandidate[4] = timingSample(31, 10, 4);
    const gpuLoss = summarizeOpt0076Timing(timingInput(
      timingArm(30, 40),
      gpuLossCandidate,
    ));
    expect(gpuLoss).toMatchObject({
      gates: {
        everyPairedAggregateWallWin: true,
        everyPairedAggregateGpuWin: false,
        meanWallSpeedupPassed: true,
        medianWallSpeedupPassed: true,
        meanGpuSpeedupPassed: true,
        medianGpuSpeedupPassed: true,
      },
      passed: false,
    });
  });

  it("fails the real timing evaluator below twofold mean and median", () => {
    const result = summarizeOpt0076Timing(timingInput(
      timingArm(30, 40),
      timingArm(16, 22),
    ));
    expect(result).toMatchObject({
      gates: {
        everyPairedAggregateWallWin: true,
        everyPairedAggregateGpuWin: true,
        meanWallSpeedupPassed: false,
        medianWallSpeedupPassed: false,
        meanGpuSpeedupPassed: false,
        medianGpuSpeedupPassed: false,
      },
      passed: false,
    });
  });

  it("rejects malformed timing sample counts", () => {
    const malformed = timingInput(
      timingArm(30, 40).slice(0, 5),
      timingArm(10, 12),
    );
    expect(() => summarizeOpt0076Timing(malformed)).toThrow(
      /current requires 6 valid samples/,
    );
  });

  it("self-authenticates the harness and leaves thermal and production pending", () => {
    for (const rawImport of [
      "vae-conv1d-fp16-k4-row-reuse-shape-selector.ts?raw",
      "vae-conv1d-fp16-c256-k4-selector.ts?raw",
      "vae-conv1d-fp16-direct-dot4-subgroup.ts?raw",
      "OPT-0076-vae-c256-native-k4-promotion.md?raw",
      "opt-0076-vae-c256-native-k4.html?raw",
      "opt-0076-vae-c256-native-k4.ts?raw",
    ]) expect(HARNESS_SOURCE).toContain(rawImport);
    expect(HARNESS_SOURCE).toContain("sourceSha256: Object.freeze({");
    for (const hash of [
      "currentSelector: await sha256Text(currentSelectorSource)",
      "candidateSelector: await sha256Text(promotedSelectorSource)",
      "directK4: await sha256Text(directK4Source)",
      "harness: await sha256Text(harnessSource)",
      "html: await sha256Text(htmlSource)",
      "experimentRecord: await sha256Text(experimentSource)",
      "currentScalarWgslSha256:",
      "candidateAndOracleK4WgslSha256:",
      "inputHashes",
      "sha256: await sha256U16(words)",
    ]) expect(HARNESS_SOURCE).toContain(hash);
    expect(HARNESS_SOURCE).toContain(
      'await crypto.subtle.digest("SHA-256", copy.buffer)',
    );
    expect(HARNESS_SOURCE).toContain(
      "externalContinuousThermalTraceCapturedByPage: false",
    );
    expect(HARNESS_SOURCE).toContain(
      '"positive-in-page-selector-primitive-pending-external-thermal-audit"',
    );
    expect(HARNESS_SOURCE).toContain(
      "externalThermalGateAuditedByPage: false",
    );
    expect(HARNESS_SOURCE).toContain(
      "diagnosticProfileEscalationAuthorized: false",
    );
    expect(HARNESS_SOURCE).toContain(
      "productionIntegrationAuthorized: false",
    );
    expect(HARNESS_SOURCE).toContain("qualityOrListeningClaim: false");
    expect(HTML_SOURCE).toContain("external thermal audit pending");
    expect(HTML_SOURCE).toContain("keep polling through cleanup");
  });

  it("keeps READY disabled until correctness and cleans up idempotently", () => {
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    const prepared = section(
      "function onPrepared",
      "async function prepareHarness",
    );
    expectOrdered(prepared, [
      'prepared.correctness["passed"] !== true',
      "return",
      "readyAtEpochMilliseconds = Date.now()",
      'document.body.dataset.status = "ready"',
      '"READY — selector/oracle identity and numerical gates passed; timing has not run"',
      "runButton.disabled = false",
    ]);
    expect(HARNESS_SOURCE).toContain(
      "let readyAtEpochMilliseconds: number | null = null",
    );
    expect(HARNESS_SOURCE).toContain(
      "readyAtEpochMilliseconds: failureEvidence.readyAtEpochMilliseconds",
    );
    expect(HARNESS_SOURCE).toContain(
      "readyEpochRecordedBeforeButtonEnabled: true",
    );
    expect(HARNESS_SOURCE).toContain(
      "if (started || active === undefined) return",
    );
    expect(HARNESS_SOURCE).toContain("runButton.disabled = true");
    expect(HARNESS_SOURCE).toContain("if (destroyed) {");
    for (const acquisition of [
      "currentKernel: AceOpt0057VaeK7ShapeSelectorKernel | undefined",
      "candidateKernel: AceOpt0076VaeC256K4SelectorKernel | undefined",
      "oracleKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel | undefined",
      "querySet: GPUQuerySet | undefined",
      "queryResolve: GPUBuffer | undefined",
      "queryReadback: GPUBuffer | undefined",
    ]) expect(HARNESS_SOURCE).toContain(`let ${acquisition}`);
    expect(HARNESS_SOURCE).toContain("currentKernel?.destroy()");
    expect(HARNESS_SOURCE).toContain("candidateKernel?.destroy()");
    expect(HARNESS_SOURCE).toContain("oracleKernel?.destroy()");
    expect(HARNESS_SOURCE).toContain("querySet?.destroy()");
    expect(HARNESS_SOURCE).toContain("tracker.destroyAll()");
    expect(HARNESS_SOURCE).toContain("const cleanupFirst = prepared.destroy()");
    expect(HARNESS_SOURCE).toContain("const cleanupSecond = prepared.destroy()");
    expect(HARNESS_SOURCE).toContain("idempotent: true");
    expect(HARNESS_SOURCE).toContain(
      'zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&\n        cleanupSecond["liveBufferCount"] === 0',
    );
    expect(HARNESS_SOURCE).toContain(
      "liveBufferCount: this.live.size,\n      liveBytes: this.liveBytes",
    );
    expect(HARNESS_SOURCE).toContain(
      "const overallPassed = timing[\"passed\"] === true &&\n    prepared.correctness[\"passed\"] === true && cleanupPassed",
    );
    expect(HARNESS_SOURCE).toContain("passed: overallPassed");
    expect(HARNESS_SOURCE).toContain("passed: cleanupPassed");
    expect(HARNESS_SOURCE).toContain("active?.destroy()");
    expect(HARNESS_SOURCE).toContain("running?.destroy()");
  });

  it("persists an authenticated nominal result stopped by one wall pair", () => {
    const resultBytes = evidence("result.json");
    const browserBytes = evidence("browser-receipt.json");
    const gateBytes = evidence("thermal-gate.json");
    const traceBytes = evidence("thermal-trace.json");
    const result = JSON.parse(resultBytes.toString("utf8"));
    const browser = JSON.parse(browserBytes.toString("utf8"));
    const gate = JSON.parse(gateBytes.toString("utf8"));
    const trace = JSON.parse(traceBytes.toString("utf8"));

    expect(resultBytes.byteLength).toBe(6_368);
    expect(browserBytes.byteLength).toBe(112_255);
    expect(gateBytes.byteLength).toBe(5_591);
    expect(traceBytes.byteLength).toBe(8_930);
    expect(result).toMatchObject({
      schema: "ace-opt-0076-vae-c256-native-k4-result-v1",
      experiment: "OPT-0076",
      status: "inconclusive",
      passed: false,
      identity: {
        allocationBaselineCommit: "e44788c2945f008f0a16db8bb1b04afbeac3d890",
        registrationCommit: "fd7df99",
        candidateAndHarnessCommit:
          "6968ed5f4fc70d43d6701a1d5117842c65c076a0",
        browserReceiptBytes: browserBytes.byteLength,
        browserReceiptSha256: sha256(browserBytes),
        thermalGateSha256: sha256(gateBytes),
        thermalTraceSha256: sha256(traceBytes),
        rawThermalTraceSha256: trace.rawTraceSha256,
      },
    });
    expect(browser).toMatchObject({
      schema: "ace-opt-0076-vae-c256-native-k4-v1",
      experiment: "OPT-0076",
      status: "completed",
      passed: false,
      disposition: "negative-stop-selector-primitive-gate",
    });
    expect(browser.identity.sourceSha256).toMatchObject({
      currentSelector: sha256(source(
        "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.ts",
      )),
      candidateSelector: sha256(source(
        "../src/webgpu/kernels/vae-conv1d-fp16-c256-k4-selector.ts",
      )),
      directK4: sha256(source(
        "../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts",
      )),
      harness: sha256(HARNESS_SOURCE),
      html: sha256(HTML_SOURCE),
      experimentRecord: result.identity.experimentRecordSha256,
    });
    expect(result.identity).toMatchObject({
      candidateSelectorSourceSha256:
        browser.identity.sourceSha256.candidateSelector,
      harnessSourceSha256: browser.identity.sourceSha256.harness,
      harnessHtmlSha256: browser.identity.sourceSha256.html,
      experimentRecordSha256: browser.identity.sourceSha256.experimentRecord,
    });

    expect(browser.correctness).toMatchObject({
      caseCount: 6,
      productionCaseCount: 3,
      adversarialCaseCount: 3,
      probesPerCase: 3,
      expectedComparedU16Count: 245_760,
      candidateOracleComparedU16Count: 245_760,
      candidateRerunComparedU16Count: 245_760,
      oracleRerunComparedU16Count: 245_760,
      exactSelectorOracleRawU16: true,
      deterministicCandidateAndOracleReruns: true,
      allOutputsFiniteAndComplete: true,
      guardsAndTailsIntact: true,
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
      completedBeforeReady: true,
      passed: true,
    });
    const cases = browser.correctness.cases as readonly PersistedCase[];
    expect(cases).toHaveLength(6);
    expect(cases.map((entry) => entry.id)).toEqual([
      "d1-production-local",
      "d3-production-local",
      "d9-production-local",
      "d1-bounded-adversarial",
      "d3-bounded-adversarial",
      "d9-bounded-adversarial",
    ]);
    expect(cases.every((entry) => entry.passed &&
      entry.candidateOracleComparedU16Count === 40_960 &&
      entry.candidateRerunComparedU16Count === 40_960 &&
      entry.oracleRerunComparedU16Count === 40_960
    )).toBe(true);
    const probes = cases.flatMap((entry) => entry.probes);
    expect(probes).toHaveLength(18);
    expect(probes.reduce((sum, probe) => sum + probe.count, 0)).toBe(245_760);
    expect(probes.every((probe) => probe.passed &&
      probe.numericalEnvelopePassed && probe.guardsAndTailsIntact &&
      probe.dispatchRangeMetadata.matchesProbe &&
      probe.candidateOracle.differingU16Count === 0 &&
      probe.candidateRepeat.differingU16Count === 0 &&
      probe.oracleRepeat.differingU16Count === 0 &&
      probe.snapshotChecks.length === 5 &&
      probe.snapshotChecks.every((snapshot) => snapshot.passed &&
        snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
        snapshot.outOfRangeWriteCount === 0)
    )).toBe(true);

    const numerical = browser.correctness.numericalEnvelope.aggregate;
    expect(numerical).toMatchObject({
      comparedValueCount: 245_760,
      rawU16MismatchCount: 60_794,
      signedZeroDifferenceCount: 0,
      nrmse: 0.00022548613920154832,
      snrDb: 72.93760298732087,
      pearson: 0.9999999715712965,
      maximumAbsoluteError: 0.00048828125,
      relativeMaximumAbsoluteError: 0.0006329113924050633,
      fp16UlpDistributionCount: 245_760,
    });
    expect(Object.values(numerical.fp16UlpDistribution as
      Readonly<Record<string, number>>).reduce(
        (sum, count) => sum + count,
        0,
      )).toBe(245_760);
    expect(browser.correctness.numericalEnvelope.passed).toBe(true);
    expect(result.correctness).toMatchObject({
      passed: true,
      caseCount: 6,
      comparedU16CountPerCandidate: 245_760,
      candidateOracleRawU16DifferenceCount: 0,
      candidateRerunDifferenceCount: 0,
      oracleRerunDifferenceCount: 0,
      allOutputsFiniteAndComplete: true,
      guardsTailsAndRangeWritesIntact: true,
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
      currentVersusCandidate: {
        comparedValueCount: numerical.comparedValueCount,
        rawU16MismatchCount: numerical.rawU16MismatchCount,
        nrmse: numerical.nrmse,
        snrDb: numerical.snrDb,
        pearson: numerical.pearson,
        maximumAbsoluteError: numerical.maximumAbsoluteError,
        relativeMaximumAbsoluteError: numerical.relativeMaximumAbsoluteError,
        passed: true,
      },
    });

    const recomputedTiming = summarizeOpt0076Timing({
      current: browser.timing.arms.current.samples,
      candidate: browser.timing.arms.candidate.samples,
    });
    expect(recomputedTiming).toMatchObject({
      sampleCountPerArm: 6,
      aggregateDispatchesPerSample: 9,
      pairedRounds: browser.timing.pairedRounds,
      speedup: browser.timing.speedup,
      gates: browser.timing.gates,
      passed: false,
    });
    const paired = browser.timing.pairedRounds as readonly PersistedPair[];
    expect(paired).toHaveLength(6);
    expect(paired.filter((round) => round.gpuWin)).toHaveLength(6);
    expect(paired.filter((round) => round.wallWin)).toHaveLength(5);
    expect(paired.at(-1)).toEqual({
      roundIndex: 5,
      currentWallMilliseconds: 10.600000023841858,
      candidateWallMilliseconds: 11.699999928474426,
      wallSpeedup: 0.9059829135592141,
      wallWin: false,
      currentGpuMilliseconds: 7.20896,
      candidateGpuMilliseconds: 4.063232,
      gpuSpeedup: 1.7741935483870968,
      gpuWin: true,
    });
    expect(browser.timing.gates).toEqual({
      requiredMeanAndMedianSpeedup: 2,
      everyPairedAggregateWallWin: false,
      everyPairedAggregateGpuWin: true,
      observedMeanWallSpeedup: 2.9419355041898876,
      meanWallSpeedupPassed: true,
      observedMedianWallSpeedup: 3.6708860870279425,
      medianWallSpeedupPassed: true,
      observedMeanGpuSpeedup: 3.6038961038961044,
      meanGpuSpeedupPassed: true,
      observedMedianGpuSpeedup: 3.8019801980198027,
      medianGpuSpeedupPassed: true,
    });
    expect(result.timing).toMatchObject({
      roundCount: 6,
      rawSampleCount: 12,
      dispatchesPerSample: 9,
      speedup: {
        meanGpu: browser.timing.speedup.meanGpu,
        medianGpu: browser.timing.speedup.medianGpu,
        meanWall: browser.timing.speedup.meanWall,
        medianWall: browser.timing.speedup.medianWall,
      },
      pairedWins: { gpu: 6, wall: 5, rounds: 6 },
      failedPair: {
        roundIndex: 5,
        candidateRanFirst: true,
        currentGpuMilliseconds: 7.20896,
        candidateGpuMilliseconds: 4.063232,
        gpuWin: true,
        currentWallMilliseconds: 10.600000023841858,
        candidateWallMilliseconds: 11.699999928474426,
        wallWin: false,
      },
      predeclaredGates: {
        everyPairedAggregateGpuWin: true,
        everyPairedAggregateWallWin: false,
        meanGpuSpeedupAtLeast2: true,
        medianGpuSpeedupAtLeast2: true,
        meanWallSpeedupAtLeast2: true,
        medianWallSpeedupAtLeast2: true,
        passed: false,
      },
    });

    const gateObservations = gate.observations as readonly ThermalObservation[];
    const traceObservations = trace.observations as readonly ThermalObservation[];
    expect(gateObservations).toHaveLength(54);
    expect(traceObservations).toHaveLength(87);
    expect(gateObservations.every((entry) => entry.level === 0 &&
      entry.rawValue === "0")).toBe(true);
    expect(traceObservations.every((entry) => entry.level === 0 &&
      entry.rawValue === "0")).toBe(true);
    expect(traceObservations.slice(0, gateObservations.length)).toEqual(
      gateObservations,
    );
    expect(maximumObservationGap(gateObservations)).toBe(936);
    expect(maximumObservationGap(traceObservations)).toBe(936);
    expect(gate.startedAtEpochMilliseconds).toBe(
      gateObservations[0]!.atEpochMilliseconds,
    );
    expect(gate.completedAtEpochMilliseconds).toBe(
      gateObservations.at(-1)!.atEpochMilliseconds,
    );
    expect(gate.completedAtEpochMilliseconds - gate.startedAtEpochMilliseconds)
      .toBeGreaterThanOrEqual(30_000);
    expect(browser.readyAtEpochMilliseconds).toBe(
      result.thermal.readyAtEpochMilliseconds,
    );
    expect(gate.startedAtEpochMilliseconds).toBeGreaterThan(
      browser.readyAtEpochMilliseconds,
    );
    expect(gate.completedAtEpochMilliseconds).toBeLessThan(
      browser.timing.runStartedAtEpochMilliseconds,
    );
    expect(browser.timing.runStartedAtEpochMilliseconds -
      gate.completedAtEpochMilliseconds).toBe(2_901);
    const lastBeforeRun = traceObservations.filter((entry) =>
      entry.atEpochMilliseconds <= browser.timing.runStartedAtEpochMilliseconds
    ).at(-1)!;
    expect(browser.timing.runStartedAtEpochMilliseconds -
      lastBeforeRun.atEpochMilliseconds).toBe(156);
    expect(trace.completedAtEpochMilliseconds).toBe(
      traceObservations.at(-1)!.atEpochMilliseconds,
    );
    expect(trace.completedAtEpochMilliseconds).toBeGreaterThan(
      browser.cleanup.cleanupCompletedAtEpochMilliseconds,
    );
    expect(trace.completedAtEpochMilliseconds -
      browser.cleanup.cleanupCompletedAtEpochMilliseconds).toBe(26_997);
    expect(result.thermal).toMatchObject({
      passed: true,
      source: gate.source,
      readyAtEpochMilliseconds: browser.readyAtEpochMilliseconds,
      gateStartedAtEpochMilliseconds: gate.startedAtEpochMilliseconds,
      gateCompletedAtEpochMilliseconds: gate.completedAtEpochMilliseconds,
      runStartedAtEpochMilliseconds: browser.timing.runStartedAtEpochMilliseconds,
      measurementStartedAtEpochMilliseconds:
        browser.timing.measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds:
        browser.timing.measurementCompletedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds:
        browser.cleanup.cleanupCompletedAtEpochMilliseconds,
      traceCompletedAtEpochMilliseconds: trace.completedAtEpochMilliseconds,
      gateObservationCount: gateObservations.length,
      traceObservationCount: traceObservations.length,
      maximumPollGapMilliseconds: 936,
      nonNominalObservationCount: 0,
      gateToRunLaunchMilliseconds: 2_901,
      lastObservationBeforeRunMilliseconds: 156,
      postCleanupCoverageMilliseconds: 26_997,
    });

    expect(browser.cleanup).toMatchObject({
      firstCall: {
        createdBufferCount: 140,
        destroyedBufferCount: 140,
        liveBufferCount: 0,
        liveBytes: 0,
        repeatedCall: false,
        deviceDestroyed: true,
      },
      secondCall: {
        createdBufferCount: 140,
        destroyedBufferCount: 140,
        liveBufferCount: 0,
        liveBytes: 0,
        repeatedCall: true,
      },
      idempotent: true,
      zeroLiveBuffers: true,
      zeroLiveBytes: true,
      createdEqualsDestroyed: true,
      passed: true,
    });
    expect(result.memoryAndLifecycle).toMatchObject({
      createdBufferCount: 140,
      destroyedBufferCount: 140,
      liveBufferCountAfterCleanup: 0,
      liveBytesAfterCleanup: 0,
      repeatedDestroyIdempotent: true,
      deviceDestroyed: true,
      passed: true,
    });
    expect(browser.decision).toMatchObject({
      diagnosticProfileEscalationAuthorized: false,
      productionIntegrationAuthorized: false,
      qualityOrListeningClaim: false,
    });
    expect(result.decision).toMatchObject({
      disposition: "inconclusive-stop-selector-primitive-gate",
      diagnosticVaeProfileAuthorized: false,
      c512OrC2314EscalationAuthorized: false,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      trajectoryOrListeningRunAuthorized: false,
    });
    expect(EXPERIMENT_SOURCE).toContain(
      "Disposition is therefore inconclusive and stopped at the selector/primitive",
    );
    expect(EXPERIMENT_SOURCE).toContain(
      "No diagnostic runtime profile, C512/C2314 decoder run, waveform,",
    );
    const ledgerRow = LEDGER_SOURCE.split("\n").find((line) =>
      line.startsWith("| OPT-0076 |")
    );
    expect(ledgerRow).toContain("| inconclusive | benchmark-only |");
    expect(ledgerRow).toContain("all six GPU pairs won");
    expect(ledgerRow).toContain("one candidate-first fenced-wall pair lost");
    expect(ledgerRow).toContain("production scalar C256 retained");
  });
});

interface ThermalObservation {
  readonly atEpochMilliseconds: number;
  readonly level: number;
  readonly rawValue: string;
}

interface PersistedSnapshot {
  readonly passed: boolean;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly outOfRangeWriteCount: number;
}

interface PersistedProbe {
  readonly count: number;
  readonly passed: boolean;
  readonly numericalEnvelopePassed: boolean;
  readonly guardsAndTailsIntact: boolean;
  readonly dispatchRangeMetadata: { readonly matchesProbe: boolean };
  readonly candidateOracle: { readonly differingU16Count: number };
  readonly candidateRepeat: { readonly differingU16Count: number };
  readonly oracleRepeat: { readonly differingU16Count: number };
  readonly snapshotChecks: readonly PersistedSnapshot[];
}

interface PersistedCase {
  readonly id: string;
  readonly passed: boolean;
  readonly candidateOracleComparedU16Count: number;
  readonly candidateRerunComparedU16Count: number;
  readonly oracleRerunComparedU16Count: number;
  readonly probes: readonly PersistedProbe[];
}

interface PersistedPair {
  readonly roundIndex: number;
  readonly currentWallMilliseconds: number;
  readonly candidateWallMilliseconds: number;
  readonly wallSpeedup: number;
  readonly wallWin: boolean;
  readonly currentGpuMilliseconds: number;
  readonly candidateGpuMilliseconds: number;
  readonly gpuSpeedup: number;
  readonly gpuWin: boolean;
}

interface BrowserElementStub {
  disabled: boolean;
  textContent: string;
  addEventListener(...args: unknown[]): void;
}

function timingInput(
  current: readonly Opt0076TimingSample[],
  candidate: readonly Opt0076TimingSample[],
): Opt0076TimingInput {
  return Object.freeze({
    current: Object.freeze([...current]),
    candidate: Object.freeze([...candidate]),
  });
}

function timingArm(
  gpuMilliseconds: number,
  wallMilliseconds: number,
): Opt0076TimingSample[] {
  return Array.from({ length: 6 }, (_, index) =>
    timingSample(gpuMilliseconds, wallMilliseconds, index));
}

function timingSample(
  gpuMilliseconds: number,
  wallMilliseconds: number,
  index: number,
): Opt0076TimingSample {
  const submitAtPerformanceMilliseconds = 100 + index * 100;
  const fenceAtPerformanceMilliseconds =
    submitAtPerformanceMilliseconds + wallMilliseconds;
  const gpuElapsedNanoseconds = Math.round(gpuMilliseconds * 1_000_000);
  return Object.freeze({
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds: 1_000 + submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds: 1_000 + fenceAtPerformanceMilliseconds,
    wallMilliseconds,
    gpuMilliseconds,
    timestampBeginNanoseconds: String(index * 100_000_000),
    timestampEndNanoseconds: String(index * 100_000_000 + gpuElapsedNanoseconds),
    gpuElapsedNanoseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    dispatchCount: 9,
    commandBufferCount: 1,
    queueDrainCount: 1,
  });
}

function source(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

function evidence(name: string): Buffer {
  return readFileSync(new URL(
    `../optimization/results/OPT-0076/${name}`,
    import.meta.url,
  ));
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function maximumObservationGap(
  observations: readonly ThermalObservation[],
): number {
  return Math.max(...observations.slice(1).map((entry, index) =>
    entry.atEpochMilliseconds - observations[index]!.atEpochMilliseconds
  ));
}

function section(start: string, end: string): string {
  const begin = HARNESS_SOURCE.indexOf(start);
  const finish = HARNESS_SOURCE.indexOf(end, begin + start.length);
  expect(begin).toBeGreaterThanOrEqual(0);
  expect(finish).toBeGreaterThan(begin);
  return HARNESS_SOURCE.slice(begin, finish);
}

function expectOrdered(value: string, tokens: readonly string[]): void {
  let cursor = -1;
  for (const token of tokens) {
    const index = value.indexOf(token, cursor + 1);
    expect(index, `missing or misordered source token: ${token}`).toBeGreaterThan(
      cursor,
    );
    cursor = index;
  }
}
