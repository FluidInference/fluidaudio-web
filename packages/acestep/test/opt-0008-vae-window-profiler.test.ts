import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import { ACE_OPT_0001_VAE_TRANSPOSE_PARTS } from
  "../benchmark/opt-0001-vae-workload.js";
import {
  createAceOpt0008VaeWindowAttribution,
  stringifyAceOpt0008VaeWindowSummary,
  summarizeAceOpt0008VaeWindowTrace,
  validateAceOpt0008VaeWindowTrace,
  type AceOpt0008VaeDecoderBatchTrace,
  type AceOpt0008VaeReadbackTrace,
  type AceOpt0008VaeWindowAttribution,
  type AceOpt0008VaeWindowTrace,
} from "../benchmark/opt-0008-vae-window-profiler.js";
import {
  ACE_OOBLECK_DECODER_CONFIG,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderCooperativePlan,
} from "../src/webgpu/vae-decoder.js";

const M3_DEVICE_LIMITS = Object.freeze({
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupStorageSize: 16_384,
  maxComputeWorkgroupsPerDimension: 65_535,
  maxStorageBufferBindingSize: 268_435_456,
});

function canonicalAttribution(): AceOpt0008VaeWindowAttribution {
  const graph = planAceVaeDecoder(256, ACE_OOBLECK_DECODER_CONFIG, 1);
  const cooperativePlan = planAceVaeDecoderQuanta(
    graph,
    ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
  );
  return createAceOpt0008VaeWindowAttribution({
    graph,
    cooperativePlan,
    limits: M3_DEVICE_LIMITS,
    quantaPerCommandBuffer: 8,
  });
}

function deterministicTrace(
  attribution: AceOpt0008VaeWindowAttribution,
): AceOpt0008VaeWindowTrace {
  let cursor = 1_000;
  const decoderBatches: AceOpt0008VaeDecoderBatchTrace[] = [];
  for (const batch of attribution.batches) {
    const encodeStartedAt = cursor;
    const encodeEndedAt = encodeStartedAt + 0.2 + (batch.batchIndex % 3) * 0.01;
    const submitStartedAt = encodeEndedAt + 0.05;
    const submitReturnedAt = submitStartedAt + 0.02;
    const drainStartedAt = submitReturnedAt + 0.01;
    const drainEndedAt = drainStartedAt + 0.5 + (batch.batchIndex % 5) * 0.1;
    const progressReportedAt = drainEndedAt + 0.03;
    const nextCommandEncodeStartedAt =
      drainEndedAt + 1.2 + (batch.batchIndex % 2) * 0.05;
    decoderBatches.push(Object.freeze({
      batchIndex: batch.batchIndex,
      encodeStartedAt,
      encodeEndedAt,
      submitStartedAt,
      submitReturnedAt,
      drainStartedAt,
      drainEndedAt,
      progressReportedAt,
      nextCommandEncodeStartedAt,
      commandBufferCount: 1,
      submissionCount: 1,
      queueDrainCount: 1,
      requestedIdleMilliseconds: 1,
      completedIdleCount: 1,
    }));
    cursor = nextCommandEncodeStartedAt;
  }
  const readback: AceOpt0008VaeReadbackTrace = Object.freeze({
    encodeStartedAt: cursor,
    encodeEndedAt: cursor + 0.3,
    submitStartedAt: cursor + 0.35,
    submitReturnedAt: cursor + 0.37,
    drainStartedAt: cursor + 0.38,
    drainEndedAt: cursor + 1.08,
    progressReportedAt: cursor + 1.1,
    decodeResolvedAt: cursor + 1.6,
    commandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    requestedIdleMilliseconds: 0,
    completedIdleCount: 0,
  });
  return Object.freeze({ decoderBatches: Object.freeze(decoderBatches), readback });
}

function membershipDigest(attribution: AceOpt0008VaeWindowAttribution): string {
  return createHash("sha256").update(JSON.stringify(attribution.batches.map(
    (batch) => ({
      batchIndex: batch.batchIndex,
      firstQuantumIndex: batch.firstQuantumIndex,
      lastQuantumIndex: batch.lastQuantumIndex,
      classification: batch.classification,
      memberships: batch.memberships.map((membership) => ({
        operationIndex: membership.operationIndex,
        operationLabel: membership.operationLabel,
        operationKind: membership.operationKind,
        family: membership.family,
        selectedKernel: membership.selectedKernel,
        firstQuantumIndex: membership.firstQuantumIndex,
        lastQuantumIndex: membership.lastQuantumIndex,
        quantumCount: membership.quantumCount,
        dispatchCount: membership.dispatchCount,
        outputElementCount: membership.outputElementCount,
        validMultiplyAccumulateCount:
          membership.validMultiplyAccumulateCount,
      })),
    }),
  ))).digest("hex");
}

describe("OPT-0008 package-native VAE window attribution", () => {
  it("derives the exact shipped batch-eight topology and ordered memberships", () => {
    const attribution = canonicalAttribution();

    expect(attribution).toMatchObject({
      schemaVersion: 1,
      kind: "ace-opt-0008-vae-window-attribution",
      inputFrames: 256,
      outputFrames: 491_520,
      quantaPerCommandBuffer: 8,
      requestedIdleMillisecondsPerDecoderBatch: 1,
      totals: {
        operationCount: 88,
        quantumCount: 3_942,
        primitiveCount: 3_988,
        passCount: 3_942,
        dispatchCount: 3_988,
        outputElementCount: 2_051_997_696,
        validMultiplyAccumulateCount: 623_639_753_728,
        pureBatchCount: 420,
        mixedBatchCount: 73,
        decoderCommandBufferCount: 493,
        decoderSubmissionCount: 493,
        decoderQueueDrainCount: 493,
        decoderRequestedIdleMilliseconds: 493,
        decoderCompletedIdleCount: 493,
        readbackCommandBufferCount: 1,
        readbackSubmissionCount: 1,
        readbackQueueDrainCount: 1,
        totalCommandBufferCount: 494,
        totalSubmissionCount: 494,
        totalQueueDrainCount: 494,
      },
    });
    expect(attribution.batches[0]).toMatchObject({
      batchIndex: 0,
      firstQuantumIndex: 0,
      lastQuantumIndex: 7,
      quantumCount: 8,
      classification: "mixed",
    });
    expect(attribution.batches.at(-1)).toMatchObject({
      batchIndex: 492,
      firstQuantumIndex: 3_936,
      lastQuantumIndex: 3_941,
      quantumCount: 6,
    });
    expect(attribution.batches.every((batch, index) =>
      batch.firstQuantumIndex === index * 8 &&
      batch.lastQuantumIndex ===
        batch.firstQuantumIndex + batch.quantumCount - 1
    )).toBe(true);

    const pureKernelBatchCounts = Object.fromEntries(
      attribution.batches
        .filter((batch) => batch.classification === "pure")
        .reduce((counts, batch) => {
          const key = batch.memberships[0]!.selectedKernel;
          counts.set(key, (counts.get(key) ?? 0) + 1);
          return counts;
        }, new Map<string, number>()),
    );
    expect(pureKernelBatchCounts).toEqual({
      "portable-add": 30,
      "portable-conv-transpose1d": 35,
      "channel-chunked-conv1d": 199,
      "portable-conv1d": 42,
      "tiled-conv1d": 43,
      "portable-snake": 71,
    });

    // The digest pins every consecutive batch boundary and ordered membership,
    // including operation, family, selected production kernel, work, and tails.
    expect(membershipDigest(attribution)).toBe(
      "b802b404053754a11973ff1be9bd9a7c7598e4a123948dac1464019f5f4741c4",
    );
  });

  it("validates and deterministically aggregates one complete physical trace", () => {
    const attribution = canonicalAttribution();
    const trace = deterministicTrace(attribution);
    const validated = validateAceOpt0008VaeWindowTrace(attribution, trace);
    const summary = summarizeAceOpt0008VaeWindowTrace(attribution, trace);

    expect(validated.totals).toMatchObject({
      decoderCommandBufferCount: 493,
      decoderSubmissionCount: 493,
      decoderQueueDrainCount: 493,
      decoderRequestedIdleMilliseconds: 493,
      decoderCompletedIdleCount: 493,
      readbackCommandBufferCount: 1,
      readbackSubmissionCount: 1,
      readbackQueueDrainCount: 1,
      totalCommandBufferCount: 494,
      totalSubmissionCount: 494,
      totalQueueDrainCount: 494,
    });
    expect(validated.totals.wallReconciliationDeltaMilliseconds).toBeCloseTo(
      0,
      10,
    );
    expect(summary.pure.batchCount).toBe(420);
    expect(summary.mixed.batchCount).toBe(73);
    expect(summary.mixed.batches.map(({ batch }) => batch.batchIndex)).toEqual(
      attribution.batches
        .filter((batch) => batch.classification === "mixed")
        .map((batch) => batch.batchIndex),
    );
    for (const aggregates of [
      summary.pure.byOperation,
      summary.pure.byFamily,
      summary.pure.byKernel,
    ]) {
      expect(aggregates.reduce((total, entry) =>
        total + entry.batchCount, 0)).toBe(420);
      expect(aggregates.reduce((total, entry) =>
        total + entry.wallMilliseconds, 0)).toBeCloseTo(
          summary.pure.byKernel.reduce((total, entry) =>
            total + entry.wallMilliseconds, 0),
          10,
        );
    }
    const mixedWall = summary.mixed.batches.reduce((total, entry) =>
      total + entry.timing.wallMilliseconds, 0);
    const pureWall = summary.pure.byKernel.reduce((total, entry) =>
      total + entry.wallMilliseconds, 0);
    expect(pureWall + mixedWall).toBeCloseTo(
      validated.totals.decoderWallMilliseconds,
      10,
    );
    const mixedQuantumCount = summary.mixed.batches.reduce((total, entry) =>
      total + entry.batch.quantumCount, 0);
    const mixedOutputElements = summary.mixed.batches.reduce((total, entry) =>
      total + entry.batch.outputElementCount, 0);
    const mixedValidMacs = summary.mixed.batches.reduce((total, entry) =>
      total + entry.batch.validMultiplyAccumulateCount, 0);
    expect(summary.pure.byKernel.reduce((total, entry) =>
      total + entry.quantumCount, 0) + mixedQuantumCount).toBe(3_942);
    expect(summary.pure.byKernel.reduce((total, entry) =>
      total + entry.outputElementCount, 0) + mixedOutputElements).toBe(
        2_051_997_696,
      );
    expect(summary.pure.byKernel.reduce((total, entry) =>
      total + entry.validMultiplyAccumulateCount, 0) + mixedValidMacs).toBe(
        623_639_753_728,
      );

    const canonical = stringifyAceOpt0008VaeWindowSummary(summary);
    expect(canonical.endsWith("\n")).toBe(true);
    expect(canonical).toBe(stringifyAceOpt0008VaeWindowSummary(summary));
    expect(JSON.parse(canonical)).toEqual(summary);
    expect(createHash("sha256").update(canonical).digest("hex")).toBe(
      "e97c96c928fba679c495243d6ebde2282d60e29a9c71d7ab63d7bad7b8e287c0",
    );
  });

  it("rejects topology and cooperative-plan drift", () => {
    const graph = planAceVaeDecoder(256, ACE_OOBLECK_DECODER_CONFIG, 1);
    const cooperativePlan = planAceVaeDecoderQuanta(
      graph,
      ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
    );
    expect(() => createAceOpt0008VaeWindowAttribution({
      graph,
      cooperativePlan,
      limits: M3_DEVICE_LIMITS,
      quantaPerCommandBuffer: 4,
    })).toThrow(/batch-8 topology/);
    expect(() => createAceOpt0008VaeWindowAttribution({
      graph,
      cooperativePlan,
      limits: {
        ...M3_DEVICE_LIMITS,
        maxComputeWorkgroupStorageSize: 0,
      },
      quantaPerCommandBuffer: 8,
    })).toThrow(/selector limit/);

    const first = cooperativePlan.quanta[0]!;
    const changed = Object.freeze({
      ...cooperativePlan,
      quanta: Object.freeze([
        Object.freeze({ ...first, logicalOutputBase: 1 }),
        ...cooperativePlan.quanta.slice(1),
      ]),
    }) satisfies AceVaeDecoderCooperativePlan;
    expect(() => createAceOpt0008VaeWindowAttribution({
      graph,
      cooperativePlan: changed,
      limits: M3_DEVICE_LIMITS,
      quantaPerCommandBuffer: 8,
    })).toThrow(/output range changed/);
  });

  it("rejects malformed, missing, reordered, or unreconciled raw traces", () => {
    const attribution = canonicalAttribution();
    const trace = deterministicTrace(attribution);
    const replaceBatch = (
      index: number,
      replacement: AceOpt0008VaeDecoderBatchTrace,
    ): AceOpt0008VaeWindowTrace => Object.freeze({
      ...trace,
      decoderBatches: Object.freeze(trace.decoderBatches.map((batch, offset) =>
        offset === index ? replacement : batch
      )),
    });
    const first = trace.decoderBatches[0]!;

    expect(() => validateAceOpt0008VaeWindowTrace(attribution, {
      ...trace,
      decoderBatches: trace.decoderBatches.slice(1),
    })).toThrow(/batch count changed/);
    expect(() => validateAceOpt0008VaeWindowTrace(
      attribution,
      replaceBatch(0, { ...first, batchIndex: 1 }),
    )).toThrow(/index changed/);
    expect(() => validateAceOpt0008VaeWindowTrace(
      attribution,
      replaceBatch(0, { ...first, commandBufferCount: 2 }),
    )).toThrow(/batch 0 CB/);
    expect(() => validateAceOpt0008VaeWindowTrace(
      attribution,
      replaceBatch(0, {
        ...first,
        drainEndedAt: first.drainStartedAt - 1,
      }),
    )).toThrow(/timeline is invalid/);
    expect(() => validateAceOpt0008VaeWindowTrace(
      attribution,
      replaceBatch(0, {
        ...first,
        nextCommandEncodeStartedAt: first.nextCommandEncodeStartedAt + 1,
      }),
    )).toThrow(/next encode/);
    expect(() => validateAceOpt0008VaeWindowTrace(attribution, {
      ...trace,
      readback: { ...trace.readback, requestedIdleMilliseconds: 1 },
    })).toThrow(/readback idle/);
    expect(() => validateAceOpt0008VaeWindowTrace(attribution, {
      ...trace,
      readback: {
        ...trace.readback,
        decodeResolvedAt: trace.readback.progressReportedAt - 1,
      },
    })).toThrow(/timeline is invalid/);
  });
});
