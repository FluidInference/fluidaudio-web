import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES,
  createAceOpt0010PlannerTokenAttribution,
  summarizeAceOpt0010PlannerTokenTrace,
  validateAceOpt0010PlannerTokenTrace,
  type AceOpt0010PlannerTokenAttribution,
  type AceOpt0010PlannerTokenTrace,
} from "../benchmark/opt-0010-planner-token-profiler.js";

const CUMULATIVE_DRAIN_BASE = 11;
const CUMULATIVE_IDLE_BASE = 11;
const PEAK_ACCOUNTED_GPU_BYTES = 1_500_000_000;

function syntheticTrace(
  attribution: AceOpt0010PlannerTokenAttribution,
): AceOpt0010PlannerTokenTrace {
  let cursor = 100;
  const quanta = attribution.quanta.map((expected, index) => {
    const ordinal = index + 1;
    const encodeStartedAt = cursor;
    const encodeEndedAt = cursor + 0.1;
    const submitStartedAt = cursor + 0.12;
    const submitReturnedAt = cursor + 0.13;
    const drainStartedAt = cursor + 0.14;
    const drainEndedAt = cursor + 0.7;
    const idleStartedAt = cursor + 0.71;
    const progressReportedAt = cursor + 0.72;
    const idleEndedAt = cursor + 1.75;
    const nextEncodeStartedAt = cursor + 1.8;
    cursor = nextEncodeStartedAt;
    return Object.freeze({
      index,
      productionQuantum: expected.productionQuantum,
      physicalDispatches: expected.physicalDispatches.map((dispatch) => ({
        ...dispatch,
        workgroups: [...dispatch.workgroups] as [number, number, number],
      })),
      progress: Object.freeze({
        phaseKind: "decode" as const,
        completedCommandBuffers: ordinal,
        totalCommandBuffers: attribution.totals.commandBufferCount,
        queueDrains: ordinal,
        cooperativeIdleMs: ordinal,
        stage: "model" as const,
        quantum: expected.productionQuantum,
        peakAccountedGpuBytes: PEAK_ACCOUNTED_GPU_BYTES,
        cumulativeQueueDrains: CUMULATIVE_DRAIN_BASE + ordinal,
        cumulativeCooperativeIdleMs: CUMULATIVE_IDLE_BASE + ordinal,
      }),
      encodeStartedAt,
      encodeEndedAt,
      submitStartedAt,
      submitReturnedAt,
      drainStartedAt,
      drainEndedAt,
      idleStartedAt,
      progressReportedAt,
      idleEndedAt,
      nextEncodeStartedAt,
      commandBufferCount: 1,
      submissionCount: 1,
      queueDrainCount: 1,
      completedIdleCount: 1,
      requestedIdleMilliseconds: 1,
    });
  });
  const ordinal = attribution.totals.commandBufferCount;
  const readback = Object.freeze({
    progress: Object.freeze({
      phaseKind: "decode" as const,
      completedCommandBuffers: ordinal,
      totalCommandBuffers: ordinal,
      queueDrains: ordinal,
      cooperativeIdleMs: ordinal,
      stage: "readback" as const,
      quantum: null,
      peakAccountedGpuBytes: PEAK_ACCOUNTED_GPU_BYTES,
      cumulativeQueueDrains: CUMULATIVE_DRAIN_BASE + ordinal,
      cumulativeCooperativeIdleMs: CUMULATIVE_IDLE_BASE + ordinal,
    }),
    encodeStartedAt: cursor,
    encodeEndedAt: cursor + 0.1,
    submitStartedAt: cursor + 0.12,
    submitReturnedAt: cursor + 0.13,
    drainStartedAt: cursor + 0.14,
    drainEndedAt: cursor + 0.6,
    idleStartedAt: cursor + 0.61,
    progressReportedAt: cursor + 0.62,
    mapStartedAt: cursor + 0.63,
    mapEndedAt: cursor + 0.9,
    reconstructStartedAt: cursor + 0.91,
    reconstructEndedAt: cursor + 1.2,
    idleEndedAt: cursor + 1.7,
    invocationResolvedAt: cursor + 1.71,
    commandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    completedIdleCount: 1,
    requestedIdleMilliseconds: 1,
    copyCommands: attribution.readback.copyCommands.map((command) => ({
      ...command,
    })),
  });
  return Object.freeze({
    quanta: Object.freeze(quanta),
    readback,
    constraintStartedAt: cursor + 1.72,
    constraintEndedAt: cursor + 1.82,
    samplingStartedAt: cursor + 1.83,
    samplingEndedAt: cursor + 2.03,
  });
}

describe("OPT-0010 pure planner token attribution", () => {
  it("pins M1/M2 production topology, FP16 GEMMs, and cache-shaped attention", () => {
    const m1 = createAceOpt0010PlannerTokenAttribution("cot-m1", 17, 512);
    const m2 = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      1_024,
    );

    for (const attribution of [m1, m2]) {
      expect(attribution.quanta.map(({ kind }) => kind)).toEqual([
        "embedding",
        ...Array.from({ length: 28 }, () => "layer"),
        "final-norm",
        "last-row-gather",
        "tied-lm-head",
        "tied-lm-head",
      ]);
      expect(attribution.quanta.slice(1, 29).map(({ layer }) => layer)).toEqual(
        Array.from({ length: 28 }, (_, layer) => layer),
      );
      expect(attribution.quanta.slice(-2).map(({ tiedHeadShards }) =>
        tiedHeadShards)).toEqual([[0, 1], [2, 3, 4]]);
      expect(attribution.totals).toMatchObject({
        modelQuantumCount: 33,
        modelDispatchPrimitiveCount: 624,
        modelPhysicalPrimitiveDispatchCount: 628,
        modelCommandBufferCount: 33,
        readbackCommandBufferCount: 1,
        commandBufferCount: 34,
        queueDrainCount: 34,
        completedIdleCount: 34,
        requestedIdleMilliseconds: 34,
        residentWeightBytes: 1_325_768_704,
      });
      expect(attribution.quanta.reduce(
        (total, quantum) => total + quantum.productionQuantum.primitiveCount,
        0,
      )).toBe(628);
    }

    expect(m1).toMatchObject({
      rows: 1,
      cachedTokensBeforeAppend: 17,
      validAttentionKeyValueTokens: 18,
      scheduledAttentionKeyValueCapacity: 512,
      semanticHeadOpportunity: null,
      readback: {
        rawLogitBytes: 434_408,
        writeStatusByteOffset: 434_432,
        writeStatusByteLength: 4,
        bufferBytes: 434_688,
      },
      totals: {
        logicalWeightBytes: 1_325_770_752,
        logicalGemmActivationBytes: 1_706_216,
        logicalMultiplyAdds: 662_818_816,
        scheduledMultiplyAdds: 10_605_297_664,
      },
      attention: {
        queryRows: 1,
        queryHeads: 16,
        keyValueHeads: 8,
        headDimension: 128,
        validKeyValueTokens: 18,
        scheduledKeyValueCapacity: 512,
        perLayer: {
          queryElements: 2_048,
          logicalKeyValuePairElements: 36_864,
          scheduledKeyValuePairElements: 1_048_576,
          logicalKeyValuePairBytes: 73_728,
          scheduledKeyValuePairBytes: 2_097_152,
          logicalKeyValidityBytes: 72,
          scheduledKeyValidityBytes: 2_048,
          logicalMultiplyAdds: 73_728,
          scheduledMultiplyAdds: 2_097_152,
          logicalFloatingPointOperations: 147_456,
          scheduledFloatingPointOperations: 4_194_304,
        },
        allLayers: {
          logicalKeyValuePairBytes: 2_064_384,
          scheduledKeyValuePairBytes: 58_720_256,
          logicalKeyValidityBytes: 2_016,
          scheduledKeyValidityBytes: 57_344,
          logicalMultiplyAdds: 2_064_384,
          scheduledMultiplyAdds: 58_720_256,
          logicalFloatingPointOperations: 4_128_768,
          scheduledFloatingPointOperations: 117_440_512,
        },
      },
    });
    expect(m2).toMatchObject({
      rows: 2,
      cachedTokensBeforeAppend: 43,
      validAttentionKeyValueTokens: 44,
      scheduledAttentionKeyValueCapacity: 1_024,
      readback: {
        rawLogitBytes: 868_816,
        writeStatusByteOffset: 868_864,
        writeStatusByteLength: 8,
        bufferBytes: 869_120,
      },
      totals: {
        logicalWeightBytes: 1_325_772_800,
        logicalGemmActivationBytes: 3_412_432,
        logicalMultiplyAdds: 1_325_637_632,
        scheduledMultiplyAdds: 10_605_297_664,
      },
      attention: {
        queryRows: 2,
        validKeyValueTokens: 44,
        scheduledKeyValueCapacity: 1_024,
        perLayer: {
          queryElements: 4_096,
          logicalKeyValuePairElements: 180_224,
          scheduledKeyValuePairElements: 4_194_304,
          logicalKeyValuePairBytes: 360_448,
          scheduledKeyValuePairBytes: 8_388_608,
          logicalKeyValidityBytes: 352,
          scheduledKeyValidityBytes: 8_192,
          logicalMultiplyAdds: 360_448,
          scheduledMultiplyAdds: 8_388_608,
          logicalFloatingPointOperations: 720_896,
          scheduledFloatingPointOperations: 16_777_216,
        },
        allLayers: {
          logicalKeyValuePairBytes: 10_092_544,
          scheduledKeyValuePairBytes: 234_881_024,
          logicalKeyValidityBytes: 9_856,
          scheduledKeyValidityBytes: 229_376,
          logicalMultiplyAdds: 10_092_544,
          scheduledMultiplyAdds: 234_881_024,
          logicalFloatingPointOperations: 20_185_088,
          scheduledFloatingPointOperations: 469_762_048,
        },
      },
    });
  });

  it("pins raw production tags, readback shard order, and M2-only head scope", () => {
    const m1 = createAceOpt0010PlannerTokenAttribution("cot-m1", 17, 512);
    const m2 = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      1_024,
    );

    expect(m1.quanta[0]!.productionQuantum).toEqual({
      id: "ace-planner-decode-1x1-capacity-512-embedding",
      logicalId: "ace-planner-decode-1x1-capacity-512-embedding",
      kind: "embedding",
      layer: null,
      primitiveCount: 5,
    });
    expect(m2.quanta.slice(-2).map(({ productionQuantum }) =>
      productionQuantum)).toEqual([
      {
        id: "ace-planner-decode-2x1-capacity-1024-tied-lm-head-part-0",
        logicalId: "ace-planner-decode-2x1-capacity-1024-tied-lm-head",
        kind: "tied-lm-head",
        layer: null,
        primitiveCount: 2,
      },
      {
        id: "ace-planner-decode-2x1-capacity-1024-tied-lm-head-part-1",
        logicalId: "ace-planner-decode-2x1-capacity-1024-tied-lm-head",
        kind: "tied-lm-head",
        layer: null,
        primitiveCount: 3,
      },
    ]);
    expect(m2.quanta.map((quantum) => quantum.physicalDispatches.length)).toEqual([
      5,
      ...Array.from({ length: 28 }, () => 22),
      1,
      1,
      2,
      3,
    ]);
    expect(m2.quanta[1]!.physicalDispatches.map((dispatch) => [
      dispatch.dispatchIdentity.replace(
        "ace-planner-decode-2x1-capacity-1024-layer-0-",
        "",
      ),
      dispatch.workgroups,
    ])).toEqual([
      ["input-norm", [2, 1, 1]],
      ["query-projection-range-0", [16, 1, 1]],
      ["key-projection-range-0", [8, 1, 1]],
      ["value-projection-range-0", [8, 1, 1]],
      ["split-query-heads", [16, 1, 1]],
      ["split-key-heads", [8, 1, 1]],
      ["split-value-heads", [8, 1, 1]],
      ["query-norm", [32, 1, 1]],
      ["key-norm", [16, 1, 1]],
      ["query-rope", [16, 1, 1]],
      ["key-rope", [8, 1, 1]],
      ["cache-append", [32, 1, 1]],
      ["causal-attention", [32, 1, 1]],
      ["merge-attention-heads", [16, 1, 1]],
      ["attention-output-projection-range-0", [8, 1, 1]],
      ["attention-residual", [8, 1, 1]],
      ["post-attention-norm", [2, 1, 1]],
      ["gate-projection-range-0", [24, 1, 1]],
      ["up-projection-range-0", [24, 1, 1]],
      ["swiglu", [24, 1, 1]],
      ["down-projection-range-0", [8, 1, 1]],
      ["mlp-residual", [8, 1, 1]],
    ]);
    expect(m2.quanta.slice(-2).map((quantum) =>
      quantum.physicalDispatches.map((dispatch) => dispatch.workgroups[0])
    )).toEqual([[384, 384], [384, 384, 161]]);
    expect(m2.readback.shards).toEqual([
      {
        shardIndex: 0,
        globalFirstRow: 0,
        localFirstRow: 0,
        rowCount: 49_152,
        byteOffset: 0,
        byteLength: 196_608,
      },
      {
        shardIndex: 1,
        globalFirstRow: 49_152,
        localFirstRow: 0,
        rowCount: 49_152,
        byteOffset: 196_608,
        byteLength: 196_608,
      },
      {
        shardIndex: 2,
        globalFirstRow: 98_304,
        localFirstRow: 0,
        rowCount: 49_152,
        byteOffset: 393_216,
        byteLength: 196_608,
      },
      {
        shardIndex: 3,
        globalFirstRow: 147_456,
        localFirstRow: 0,
        rowCount: 49_152,
        byteOffset: 589_824,
        byteLength: 196_608,
      },
      {
        shardIndex: 4,
        globalFirstRow: 196_608,
        localFirstRow: 0,
        rowCount: 20_596,
        byteOffset: 786_432,
        byteLength: 82_384,
      },
    ]);
    expect(m1.semanticHeadOpportunity).toBeNull();
    expect(m2.readback.copyCommands).toEqual([
      ...m2.readback.shards.map((shard) => ({
        index: shard.shardIndex,
        sourceBufferLabel: `logits-${shard.shardIndex}`,
        shardIndex: shard.shardIndex,
        sourceOffset: 0,
        destinationBufferLabel: "ace-planner-logit-readback",
        destinationOffset: shard.byteOffset,
        copiedBytes: shard.byteLength,
      })),
      {
        index: 5,
        sourceBufferLabel: "write-status",
        shardIndex: null,
        sourceOffset: 0,
        destinationBufferLabel: "ace-planner-logit-readback",
        destinationOffset: 868_864,
        copiedBytes: 8,
      },
    ]);
    expect(m2.semanticHeadOpportunity).toMatchObject({
      firstTokenId: 151_669,
      tokenCount: 64_000,
      fullHead: {
        logicalWeightTrafficBytes: 444_833_792,
        gemmActivationBytes: 889_296,
        rawLogitBytes: 868_816,
        logicalMultiplyAdds: 444_833_792,
        logicalFloatingPointOperations: 889_667_584,
        scheduledMultiplyAdds: 3_558_866_944,
        scheduledFloatingPointOperations: 7_117_733_888,
      },
      restrictedCodeHead: {
        logicalWeightTrafficBytes: 131_072_000,
        gemmActivationBytes: 264_192,
        rawLogitBytes: 256_000,
        logicalMultiplyAdds: 131_072_000,
        logicalFloatingPointOperations: 262_144_000,
        scheduledMultiplyAdds: 1_050_673_152,
        scheduledFloatingPointOperations: 2_101_346_304,
      },
      avoidablePerToken: {
        logicalWeightTrafficBytes: 313_761_792,
        gemmActivationBytes: 625_104,
        rawLogitBytes: 612_816,
        logicalMultiplyAdds: 313_761_792,
        logicalFloatingPointOperations: 627_523_584,
        scheduledMultiplyAdds: 2_508_193_792,
        scheduledFloatingPointOperations: 5_016_387_584,
      },
      terminalEos: {
        tokenId: 151_645,
        shardIndex: 3,
        globalFirstRow: 151_645,
        localFirstRow: 4_189,
        rowCount: 1,
        terminalOnly: true,
        includedInRestrictedCodeHead: false,
        tiedEmbeddingRemainsFullyResident: true,
        optionalHeadRow: {
          logicalWeightTrafficBytes: 2_048,
          gemmActivationBytes: 4_100,
          rawLogitBytes: 4,
          logicalMultiplyAdds: 2_048,
          logicalFloatingPointOperations: 4_096,
          scheduledMultiplyAdds: 2_097_152,
          scheduledFloatingPointOperations: 4_194_304,
        },
      },
    });
    expect(m2.semanticHeadOpportunity!.restrictedCodeHead.intersections).toEqual([
      expect.objectContaining({
        shardIndex: 3,
        globalFirstRow: 151_669,
        localFirstRow: 4_213,
        rowCount: 44_939,
        logicalWeightTrafficBytes: 92_035_072,
        gemmActivationBytes: 183_852,
        rawLogitBytes: 179_756,
        logicalMultiplyAdds: 92_035_072,
        logicalFloatingPointOperations: 184_070_144,
        scheduledMultiplyAdds: 738_197_504,
        scheduledFloatingPointOperations: 1_476_395_008,
        outputRanges: [{
          index: 0,
          firstOutput: 0,
          outputCount: 89_878,
          firstWorkgroup: 0,
          workgroupCount: 352,
          multiplyAdds: 738_197_504,
        }],
      }),
      expect.objectContaining({
        shardIndex: 4,
        globalFirstRow: 196_608,
        localFirstRow: 0,
        rowCount: 19_061,
        logicalWeightTrafficBytes: 39_036_928,
        gemmActivationBytes: 80_340,
        rawLogitBytes: 76_244,
        logicalMultiplyAdds: 39_036_928,
        logicalFloatingPointOperations: 78_073_856,
        scheduledMultiplyAdds: 312_475_648,
        scheduledFloatingPointOperations: 624_951_296,
        outputRanges: [{
          index: 0,
          firstOutput: 0,
          outputCount: 38_122,
          firstWorkgroup: 0,
          workgroupCount: 149,
          multiplyAdds: 312_475_648,
        }],
      }),
    ]);
  });

  it("independently authenticates every production-derived source", () => {
    for (const [file, expectedSha256] of Object.entries(
      ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES,
    )) {
      const actual = createHash("sha256")
        .update(readFileSync(resolve(process.cwd(), file)))
        .digest("hex");
      expect(actual, file).toBe(expectedSha256);
    }
    const attribution = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      1_024,
    );
    expect(attribution.productionSourceIdentities).toBe(
      ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES,
    );
  });

  it("requires exact bounded cache capacity and keeps validity separate", () => {
    const short = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      512,
    );
    const long = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      1_024,
    );
    expect(short.validAttentionKeyValueTokens).toBe(44);
    expect(long.validAttentionKeyValueTokens).toBe(44);
    expect(long.attention.perLayer.scheduledKeyValuePairBytes).toBe(
      short.attention.perLayer.scheduledKeyValuePairBytes * 2,
    );
    expect(createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      40_959,
      40_960,
    ).validAttentionKeyValueTokens).toBe(40_960);

    expect(() => createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      -1,
      512,
    )).toThrow(/cachedTokensBeforeAppend/);
    expect(() => createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      Number.MAX_SAFE_INTEGER,
      Number.MAX_SAFE_INTEGER,
    )).toThrow(/cacheCapacity/);
    expect(() => createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      512,
      512,
    )).toThrow(/cacheCapacity/);
    expect(() => createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      512,
      40_961,
    )).toThrow(/cacheCapacity/);
  });

  it("validates every raw quantum and overlapping production progress payload", () => {
    const attribution = createAceOpt0010PlannerTokenAttribution(
      "semantic-m2",
      43,
      1_024,
    );
    const trace = syntheticTrace(attribution);
    const validated = validateAceOpt0010PlannerTokenTrace(attribution, trace);
    const summary = summarizeAceOpt0010PlannerTokenTrace(attribution, trace);

    expect(validated).toMatchObject({
      commandBufferCount: 34,
      queueDrainCount: 34,
      completedIdleCount: 34,
    });
    expect(validated.constraintMilliseconds).toBeCloseTo(0.1, 12);
    expect(validated.samplingMilliseconds).toBeCloseTo(0.2, 12);
    expect(validated.quantumTimings).toHaveLength(33);
    expect(summary.byKind.map((entry) => ({
      kind: entry.kind,
      quantumCount: entry.quantumCount,
      primitiveCount: entry.primitiveCount,
    }))).toEqual([
      { kind: "embedding", quantumCount: 1, primitiveCount: 5 },
      { kind: "layer", quantumCount: 28, primitiveCount: 616 },
      { kind: "final-norm", quantumCount: 1, primitiveCount: 1 },
      { kind: "last-row-gather", quantumCount: 1, primitiveCount: 1 },
      { kind: "tied-lm-head", quantumCount: 2, primitiveCount: 5 },
    ]);
  });

  it("rejects changed raw tags, dispatches, copies, and zero-duration idles", () => {
    const attribution = createAceOpt0010PlannerTokenAttribution(
      "cot-m1",
      17,
      512,
    );
    const trace = syntheticTrace(attribution);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      quanta: trace.quanta.map((entry, index) => index === 1
        ? {
            ...entry,
            productionQuantum: {
              ...entry.productionQuantum,
              logicalId: "reordered-production-quantum",
            },
          }
        : entry),
    })).toThrow(/quantum 1 production tag changed/);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      quanta: trace.quanta.map((entry, index) => index === 1
        ? {
            ...entry,
            progress: {
              ...entry.progress,
              completedCommandBuffers: 99,
            },
          }
        : entry),
    })).toThrow(/completed command buffers/);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      quanta: trace.quanta.map((entry, index) => index === 1
        ? {
            ...entry,
            physicalDispatches: entry.physicalDispatches.map(
              (dispatch, dispatchIndex) => dispatchIndex === 3
                ? { ...dispatch, workgroups: [999, 1, 1] as const }
                : dispatch,
            ),
          }
        : entry),
    })).toThrow(/quantum 1 physical dispatch sequence changed/);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      readback: {
        ...trace.readback,
        copyCommands: [
          trace.readback.copyCommands[1]!,
          trace.readback.copyCommands[0]!,
          ...trace.readback.copyCommands.slice(2),
        ],
      },
    })).toThrow(/readback copy command sequence changed/);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      quanta: trace.quanta.map((entry, index) => index === 4
        ? {
            ...entry,
            progressReportedAt: entry.idleStartedAt,
            idleEndedAt: entry.idleStartedAt,
          }
        : entry),
    })).toThrow(/quantum 4 idle duration must be positive/);
    expect(() => validateAceOpt0010PlannerTokenTrace(attribution, {
      ...trace,
      readback: {
        ...trace.readback,
        progressReportedAt: trace.readback.idleStartedAt,
        idleEndedAt: trace.readback.idleStartedAt,
      },
    })).toThrow(/readback idle duration must be positive/);
  });
});
