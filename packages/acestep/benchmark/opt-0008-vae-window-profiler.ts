import {
  type AceVaeDecoderCooperativePlan,
  type AceVaeDecoderGraphPlan,
  type AceVaeDecoderOperation,
  type AceVaeDecoderQuantumPlan,
} from "../src/webgpu/vae-decoder.js";
import {
  selectAceChannelChunkedVaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-channel-chunks.js";
import {
  selectAceTiledVaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d.js";
import {
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
  type AceVaeConv1dShape,
  type AceVaeConvTranspose1dShape,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
} from "../src/runtime/scheduler.js";
import {
  ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
} from "../src/webgpu/vae-backend.js";

export const ACE_OPT_0008_VAE_QUANTA_PER_COMMAND_BUFFER =
  ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER;

export type AceOpt0008VaeOperationFamily =
  | "k7-conv1d"
  | "k1-conv1d"
  | "conv-transpose1d"
  | "snake"
  | "add";

export type AceOpt0008VaeSelectedKernel =
  | "tiled-conv1d"
  | "channel-chunked-conv1d"
  | "portable-conv1d"
  | "portable-conv-transpose1d"
  | "portable-snake"
  | "portable-add";

export type AceOpt0008VaeSelectorLimits = Pick<
  GPUSupportedLimits,
  | "maxComputeInvocationsPerWorkgroup"
  | "maxComputeWorkgroupSizeX"
  | "maxComputeWorkgroupSizeY"
  | "maxComputeWorkgroupStorageSize"
  | "maxComputeWorkgroupsPerDimension"
  | "maxStorageBufferBindingSize"
>;

export interface AceOpt0008VaeWindowAttributionInput {
  readonly graph: AceVaeDecoderGraphPlan;
  readonly cooperativePlan: AceVaeDecoderCooperativePlan;
  readonly limits: AceOpt0008VaeSelectorLimits;
  readonly quantaPerCommandBuffer: number;
}

export interface AceOpt0008VaeQuantumAttribution {
  readonly quantumIndex: number;
  readonly quantumId: string;
  readonly operationIndex: number;
  readonly operationLabel: string;
  readonly operationKind: AceVaeDecoderOperation["kind"];
  readonly family: AceOpt0008VaeOperationFamily;
  readonly selectedKernel: AceOpt0008VaeSelectedKernel;
  readonly primitiveCount: number;
  readonly passCount: 1;
  readonly dispatchCount: number;
  readonly outputElementCount: number;
  readonly validMultiplyAccumulateCount: number;
  readonly estimatedMaximumMultiplyAccumulateCount: number;
}

export interface AceOpt0008VaeBatchMembership {
  readonly operationIndex: number;
  readonly operationLabel: string;
  readonly operationKind: AceVaeDecoderOperation["kind"];
  readonly family: AceOpt0008VaeOperationFamily;
  readonly selectedKernel: AceOpt0008VaeSelectedKernel;
  readonly firstQuantumIndex: number;
  readonly lastQuantumIndex: number;
  readonly quantumCount: number;
  readonly primitiveCount: number;
  readonly passCount: number;
  readonly dispatchCount: number;
  readonly outputElementCount: number;
  readonly validMultiplyAccumulateCount: number;
  readonly estimatedMaximumMultiplyAccumulateCount: number;
}

export interface AceOpt0008VaeBatchAttribution {
  readonly batchIndex: number;
  readonly firstQuantumIndex: number;
  readonly lastQuantumIndex: number;
  readonly quantumCount: number;
  readonly classification: "pure" | "mixed";
  readonly memberships: readonly AceOpt0008VaeBatchMembership[];
  readonly primitiveCount: number;
  readonly passCount: number;
  readonly dispatchCount: number;
  readonly outputElementCount: number;
  readonly validMultiplyAccumulateCount: number;
  readonly estimatedMaximumMultiplyAccumulateCount: number;
  readonly expectedCommandBufferCount: 1;
  readonly expectedSubmissionCount: 1;
  readonly expectedQueueDrainCount: 1;
  readonly expectedRequestedIdleMilliseconds: number;
  readonly expectedCompletedIdleCount: 1;
}

export interface AceOpt0008VaeWindowAttributionTotals {
  readonly operationCount: number;
  readonly quantumCount: number;
  readonly primitiveCount: number;
  readonly passCount: number;
  readonly dispatchCount: number;
  readonly outputElementCount: number;
  readonly validMultiplyAccumulateCount: number;
  readonly estimatedMaximumMultiplyAccumulateCount: number;
  readonly pureBatchCount: number;
  readonly mixedBatchCount: number;
  readonly decoderCommandBufferCount: number;
  readonly decoderSubmissionCount: number;
  readonly decoderQueueDrainCount: number;
  readonly decoderRequestedIdleMilliseconds: number;
  readonly decoderCompletedIdleCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly readbackSubmissionCount: 1;
  readonly readbackQueueDrainCount: 1;
  readonly totalCommandBufferCount: number;
  readonly totalSubmissionCount: number;
  readonly totalQueueDrainCount: number;
}

export interface AceOpt0008VaeWindowAttribution {
  readonly schemaVersion: 1;
  readonly kind: "ace-opt-0008-vae-window-attribution";
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly quantaPerCommandBuffer: number;
  readonly requestedIdleMillisecondsPerDecoderBatch: number;
  readonly quanta: readonly AceOpt0008VaeQuantumAttribution[];
  readonly batches: readonly AceOpt0008VaeBatchAttribution[];
  readonly totals: AceOpt0008VaeWindowAttributionTotals;
}

export interface AceOpt0008VaeDecoderBatchTrace {
  readonly batchIndex: number;
  readonly encodeStartedAt: number;
  readonly encodeEndedAt: number;
  readonly submitStartedAt: number;
  readonly submitReturnedAt: number;
  readonly drainStartedAt: number;
  readonly drainEndedAt: number;
  readonly progressReportedAt: number;
  readonly nextCommandEncodeStartedAt: number;
  readonly commandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly requestedIdleMilliseconds: number;
  readonly completedIdleCount: number;
}

export interface AceOpt0008VaeReadbackTrace {
  readonly encodeStartedAt: number;
  readonly encodeEndedAt: number;
  readonly submitStartedAt: number;
  readonly submitReturnedAt: number;
  readonly drainStartedAt: number;
  readonly drainEndedAt: number;
  readonly progressReportedAt: number;
  readonly decodeResolvedAt: number;
  readonly commandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly requestedIdleMilliseconds: number;
  readonly completedIdleCount: number;
}

export interface AceOpt0008VaeWindowTrace {
  readonly decoderBatches: readonly AceOpt0008VaeDecoderBatchTrace[];
  readonly readback: AceOpt0008VaeReadbackTrace;
}

export interface AceOpt0008VaeDecoderBatchTiming {
  readonly batchIndex: number;
  readonly encodeMilliseconds: number;
  readonly submitCallMilliseconds: number;
  readonly submitThroughDrainMilliseconds: number;
  readonly drainWaitMilliseconds: number;
  readonly actualIdleMilliseconds: number;
  readonly residualMilliseconds: number;
  readonly wallMilliseconds: number;
}

export interface AceOpt0008VaeReadbackTiming {
  readonly encodeMilliseconds: number;
  readonly submitCallMilliseconds: number;
  readonly submitThroughDrainMilliseconds: number;
  readonly drainWaitMilliseconds: number;
  readonly postDrainDetachMilliseconds: number;
  readonly residualMilliseconds: number;
  readonly wallMilliseconds: number;
}

export interface AceOpt0008VaeValidatedWindowTraceTotals {
  readonly decoderCommandBufferCount: number;
  readonly decoderSubmissionCount: number;
  readonly decoderQueueDrainCount: number;
  readonly decoderRequestedIdleMilliseconds: number;
  readonly decoderCompletedIdleCount: number;
  readonly readbackCommandBufferCount: number;
  readonly readbackSubmissionCount: number;
  readonly readbackQueueDrainCount: number;
  readonly totalCommandBufferCount: number;
  readonly totalSubmissionCount: number;
  readonly totalQueueDrainCount: number;
  readonly decoderEncodeMilliseconds: number;
  readonly decoderSubmitThroughDrainMilliseconds: number;
  readonly decoderActualIdleMilliseconds: number;
  readonly decoderResidualMilliseconds: number;
  readonly decoderWallMilliseconds: number;
  readonly readbackWallMilliseconds: number;
  readonly timedWindowMilliseconds: number;
  readonly wallReconciliationDeltaMilliseconds: number;
}

export interface AceOpt0008VaeValidatedWindowTrace {
  readonly decoderBatches: readonly AceOpt0008VaeDecoderBatchTiming[];
  readonly readback: AceOpt0008VaeReadbackTiming;
  readonly totals: AceOpt0008VaeValidatedWindowTraceTotals;
}

interface AceOpt0008VaeAggregateTotals {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly primitiveCount: number;
  readonly passCount: number;
  readonly dispatchCount: number;
  readonly outputElementCount: number;
  readonly validMultiplyAccumulateCount: number;
  readonly estimatedMaximumMultiplyAccumulateCount: number;
  readonly commandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly requestedIdleMilliseconds: number;
  readonly completedIdleCount: number;
  readonly encodeMilliseconds: number;
  readonly submitThroughDrainMilliseconds: number;
  readonly actualIdleMilliseconds: number;
  readonly residualMilliseconds: number;
  readonly wallMilliseconds: number;
}

export interface AceOpt0008VaeOperationAggregate
  extends AceOpt0008VaeAggregateTotals {
  readonly operationIndex: number;
  readonly operationLabel: string;
  readonly operationKind: AceVaeDecoderOperation["kind"];
  readonly family: AceOpt0008VaeOperationFamily;
  readonly selectedKernels: readonly AceOpt0008VaeSelectedKernel[];
}

export interface AceOpt0008VaeFamilyAggregate
  extends AceOpt0008VaeAggregateTotals {
  readonly family: AceOpt0008VaeOperationFamily;
}

export interface AceOpt0008VaeKernelAggregate
  extends AceOpt0008VaeAggregateTotals {
  readonly selectedKernel: AceOpt0008VaeSelectedKernel;
}

export interface AceOpt0008VaeMixedBatchSummary {
  readonly batch: AceOpt0008VaeBatchAttribution;
  readonly timing: AceOpt0008VaeDecoderBatchTiming;
}

export interface AceOpt0008VaeWindowSummary {
  readonly schemaVersion: 1;
  readonly kind: "ace-opt-0008-vae-window-summary";
  readonly attributionTotals: AceOpt0008VaeWindowAttributionTotals;
  readonly traceTotals: AceOpt0008VaeValidatedWindowTraceTotals;
  readonly pure: {
    readonly batchCount: number;
    readonly byOperation: readonly AceOpt0008VaeOperationAggregate[];
    readonly byFamily: readonly AceOpt0008VaeFamilyAggregate[];
    readonly byKernel: readonly AceOpt0008VaeKernelAggregate[];
  };
  readonly mixed: {
    readonly batchCount: number;
    readonly batches: readonly AceOpt0008VaeMixedBatchSummary[];
  };
  readonly fixedBoundaries: {
    readonly readback: AceOpt0008VaeReadbackTiming;
  };
}

const FAMILY_ORDER = Object.freeze([
  "k7-conv1d",
  "k1-conv1d",
  "conv-transpose1d",
  "snake",
  "add",
] satisfies readonly AceOpt0008VaeOperationFamily[]);

const KERNEL_ORDER = Object.freeze([
  "tiled-conv1d",
  "channel-chunked-conv1d",
  "portable-conv1d",
  "portable-conv-transpose1d",
  "portable-snake",
  "portable-add",
] satisfies readonly AceOpt0008VaeSelectedKernel[]);

const SELECTOR_LIMIT_KEYS = Object.freeze([
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupStorageSize",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBufferBindingSize",
] satisfies readonly (keyof AceOpt0008VaeSelectorLimits)[]);

export function createAceOpt0008VaeWindowAttribution(
  input: AceOpt0008VaeWindowAttributionInput,
): AceOpt0008VaeWindowAttribution {
  if (
    input.quantaPerCommandBuffer !==
      ACE_OPT_0008_VAE_QUANTA_PER_COMMAND_BUFFER
  ) {
    throw new RangeError(
      `ACE OPT-0008 requires the shipped batch-${ACE_OPT_0008_VAE_QUANTA_PER_COMMAND_BUFFER} topology`,
    );
  }
  validateSelectorLimits(input.limits);
  const { graph, cooperativePlan } = input;
  if (
    cooperativePlan.quantumCount !== cooperativePlan.quanta.length ||
    graph.primitiveCount !== graph.operations.length
  ) {
    throw new Error("ACE OPT-0008 graph/cooperative counts are inconsistent");
  }

  const operationCursors = graph.operations.map(() => 0);
  let expectedControlRecordIndex = 0;
  let previousOperationIndex = -1;
  const quanta = Object.freeze(cooperativePlan.quanta.map((quantum, index) => {
    const operation = graph.operations[quantum.operationIndex];
    if (operation === undefined) {
      throw new RangeError(`ACE OPT-0008 quantum ${index} has no operation`);
    }
    if (
      quantum.index !== index ||
      quantum.operationIndex < previousOperationIndex ||
      quantum.operationLabel !== operation.label ||
      quantum.operationKind !== operation.kind
    ) {
      throw new Error(`ACE OPT-0008 quantum ${index} metadata changed`);
    }
    previousOperationIndex = quantum.operationIndex;
    const cursor = operationCursors[quantum.operationIndex]!;
    if (
      quantum.logicalOutputBase !== cursor ||
      !Number.isSafeInteger(quantum.logicalOutputCount) ||
      quantum.logicalOutputCount <= 0 ||
      quantum.primitives.length < 1
    ) {
      throw new Error(`ACE OPT-0008 quantum ${index} output range changed`);
    }
    operationCursors[quantum.operationIndex] = checkedAdd(
      cursor,
      quantum.logicalOutputCount,
      `quantum ${index} output end`,
    );
    let physicalOutputCount = 0;
    for (const primitive of quantum.primitives) {
      if (
        primitive.controlRecordIndex !== expectedControlRecordIndex ||
        !Number.isSafeInteger(primitive.outputCount) ||
        primitive.outputCount <= 0
      ) {
        throw new Error(`ACE OPT-0008 quantum ${index} primitive metadata changed`);
      }
      expectedControlRecordIndex += 1;
      physicalOutputCount = checkedAdd(
        physicalOutputCount,
        primitive.outputCount,
        `quantum ${index} physical output count`,
      );
    }
    if (physicalOutputCount !== quantum.logicalOutputCount) {
      throw new Error(
        `ACE OPT-0008 quantum ${index} physical/logical outputs differ`,
      );
    }
    const classification = classifyQuantum(operation, quantum, input.limits);
    const validMultiplyAccumulateCount = countQuantumValidMacs(
      operation,
      quantum,
    );
    if (
      !Number.isSafeInteger(quantum.estimatedMaximumMultiplyAccumulates) ||
      quantum.estimatedMaximumMultiplyAccumulates <
        validMultiplyAccumulateCount
    ) {
      throw new Error(`ACE OPT-0008 quantum ${index} MAC bound changed`);
    }
    return Object.freeze({
      quantumIndex: index,
      quantumId: quantum.id,
      operationIndex: quantum.operationIndex,
      operationLabel: operation.label,
      operationKind: operation.kind,
      family: classification.family,
      selectedKernel: classification.selectedKernel,
      primitiveCount: quantum.primitives.length,
      passCount: 1 as const,
      dispatchCount: quantum.primitives.length,
      outputElementCount: quantum.logicalOutputCount,
      validMultiplyAccumulateCount,
      estimatedMaximumMultiplyAccumulateCount:
        quantum.estimatedMaximumMultiplyAccumulates,
    });
  }));

  if (expectedControlRecordIndex !== cooperativePlan.primitiveDispatchCount) {
    throw new Error("ACE OPT-0008 primitive dispatch count changed");
  }
  for (const [operationIndex, operation] of graph.operations.entries()) {
    if (operationCursors[operationIndex] !== operationOutputElements(operation)) {
      throw new Error(
        `ACE OPT-0008 operation ${operation.label} output coverage changed`,
      );
    }
  }

  const batches: AceOpt0008VaeBatchAttribution[] = [];
  for (
    let firstQuantumIndex = 0;
    firstQuantumIndex < quanta.length;
    firstQuantumIndex += input.quantaPerCommandBuffer
  ) {
    const slice = quanta.slice(
      firstQuantumIndex,
      firstQuantumIndex + input.quantaPerCommandBuffer,
    );
    batches.push(createBatchAttribution(batches.length, slice));
  }
  const stableBatches = Object.freeze(batches);
  const decoderCommandBufferCount = stableBatches.length;
  const totals = Object.freeze({
    operationCount: graph.operations.length,
    quantumCount: quanta.length,
    primitiveCount: sum(quanta.map((quantum) => quantum.primitiveCount)),
    passCount: sum(quanta.map((quantum) => quantum.passCount)),
    dispatchCount: sum(quanta.map((quantum) => quantum.dispatchCount)),
    outputElementCount: sum(quanta.map((quantum) => quantum.outputElementCount)),
    validMultiplyAccumulateCount: sum(
      quanta.map((quantum) => quantum.validMultiplyAccumulateCount),
    ),
    estimatedMaximumMultiplyAccumulateCount: sum(
      quanta.map((quantum) =>
        quantum.estimatedMaximumMultiplyAccumulateCount
      ),
    ),
    pureBatchCount: stableBatches.filter((batch) =>
      batch.classification === "pure"
    ).length,
    mixedBatchCount: stableBatches.filter((batch) =>
      batch.classification === "mixed"
    ).length,
    decoderCommandBufferCount,
    decoderSubmissionCount: decoderCommandBufferCount,
    decoderQueueDrainCount: decoderCommandBufferCount,
    decoderRequestedIdleMilliseconds:
      decoderCommandBufferCount * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    decoderCompletedIdleCount: decoderCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    readbackSubmissionCount: 1 as const,
    readbackQueueDrainCount: 1 as const,
    totalCommandBufferCount: decoderCommandBufferCount + 1,
    totalSubmissionCount: decoderCommandBufferCount + 1,
    totalQueueDrainCount: decoderCommandBufferCount + 1,
  }) satisfies AceOpt0008VaeWindowAttributionTotals;
  return Object.freeze({
    schemaVersion: 1,
    kind: "ace-opt-0008-vae-window-attribution",
    inputFrames: graph.inputFrames,
    outputFrames: graph.outputFrames,
    quantaPerCommandBuffer: input.quantaPerCommandBuffer,
    requestedIdleMillisecondsPerDecoderBatch:
      ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    quanta,
    batches: stableBatches,
    totals,
  });
}

export function validateAceOpt0008VaeWindowTrace(
  attribution: AceOpt0008VaeWindowAttribution,
  trace: AceOpt0008VaeWindowTrace,
): AceOpt0008VaeValidatedWindowTrace {
  if (trace.decoderBatches.length !== attribution.batches.length) {
    throw new Error("ACE OPT-0008 decoder trace batch count changed");
  }
  const decoderBatches = Object.freeze(trace.decoderBatches.map(
    (record, index) => {
      const expected = attribution.batches[index]!;
      if (record.batchIndex !== index || expected.batchIndex !== index) {
        throw new Error(`ACE OPT-0008 decoder batch ${index} index changed`);
      }
      requireExactCount(record.commandBufferCount, 1, `batch ${index} CB`);
      requireExactCount(record.submissionCount, 1, `batch ${index} submission`);
      requireExactCount(record.queueDrainCount, 1, `batch ${index} drain`);
      requireExactCount(
        record.requestedIdleMilliseconds,
        expected.expectedRequestedIdleMilliseconds,
        `batch ${index} requested idle`,
      );
      requireExactCount(
        record.completedIdleCount,
        expected.expectedCompletedIdleCount,
        `batch ${index} completed idle`,
      );
      requireTimeline([
        record.encodeStartedAt,
        record.encodeEndedAt,
        record.submitStartedAt,
        record.submitReturnedAt,
        record.drainStartedAt,
        record.drainEndedAt,
        record.progressReportedAt,
        record.nextCommandEncodeStartedAt,
      ], `decoder batch ${index}`);
      const expectedNext = index + 1 < trace.decoderBatches.length
        ? trace.decoderBatches[index + 1]!.encodeStartedAt
        : trace.readback.encodeStartedAt;
      requireSameTimestamp(
        record.nextCommandEncodeStartedAt,
        expectedNext,
        `decoder batch ${index} next encode`,
      );
      const timing = Object.freeze({
        batchIndex: index,
        encodeMilliseconds: record.encodeEndedAt - record.encodeStartedAt,
        submitCallMilliseconds:
          record.submitReturnedAt - record.submitStartedAt,
        submitThroughDrainMilliseconds:
          record.drainEndedAt - record.submitStartedAt,
        drainWaitMilliseconds: record.drainEndedAt - record.drainStartedAt,
        actualIdleMilliseconds:
          record.nextCommandEncodeStartedAt - record.drainEndedAt,
        residualMilliseconds: record.submitStartedAt - record.encodeEndedAt,
        wallMilliseconds:
          record.nextCommandEncodeStartedAt - record.encodeStartedAt,
      }) satisfies AceOpt0008VaeDecoderBatchTiming;
      requireTimingReconciliation(
        timing.wallMilliseconds,
        timing.encodeMilliseconds + timing.submitThroughDrainMilliseconds +
          timing.actualIdleMilliseconds + timing.residualMilliseconds,
        `decoder batch ${index}`,
      );
      return timing;
    },
  ));

  const readback = trace.readback;
  requireExactCount(readback.commandBufferCount, 1, "readback CB");
  requireExactCount(readback.submissionCount, 1, "readback submission");
  requireExactCount(readback.queueDrainCount, 1, "readback drain");
  requireExactCount(readback.requestedIdleMilliseconds, 0, "readback idle");
  requireExactCount(readback.completedIdleCount, 0, "readback completed idle");
  requireTimeline([
    readback.encodeStartedAt,
    readback.encodeEndedAt,
    readback.submitStartedAt,
    readback.submitReturnedAt,
    readback.drainStartedAt,
    readback.drainEndedAt,
    readback.progressReportedAt,
    readback.decodeResolvedAt,
  ], "readback");
  const readbackTiming = Object.freeze({
    encodeMilliseconds: readback.encodeEndedAt - readback.encodeStartedAt,
    submitCallMilliseconds:
      readback.submitReturnedAt - readback.submitStartedAt,
    submitThroughDrainMilliseconds:
      readback.drainEndedAt - readback.submitStartedAt,
    drainWaitMilliseconds: readback.drainEndedAt - readback.drainStartedAt,
    postDrainDetachMilliseconds:
      readback.decodeResolvedAt - readback.drainEndedAt,
    residualMilliseconds: readback.submitStartedAt - readback.encodeEndedAt,
    wallMilliseconds: readback.decodeResolvedAt - readback.encodeStartedAt,
  }) satisfies AceOpt0008VaeReadbackTiming;
  requireTimingReconciliation(
    readbackTiming.wallMilliseconds,
    readbackTiming.encodeMilliseconds +
      readbackTiming.submitThroughDrainMilliseconds +
      readbackTiming.postDrainDetachMilliseconds +
      readbackTiming.residualMilliseconds,
    "readback",
  );

  const decoderCommandBufferCount = sum(
    trace.decoderBatches.map((record) => record.commandBufferCount),
  );
  const decoderSubmissionCount = sum(
    trace.decoderBatches.map((record) => record.submissionCount),
  );
  const decoderQueueDrainCount = sum(
    trace.decoderBatches.map((record) => record.queueDrainCount),
  );
  const decoderRequestedIdleMilliseconds = sum(
    trace.decoderBatches.map((record) => record.requestedIdleMilliseconds),
  );
  const decoderCompletedIdleCount = sum(
    trace.decoderBatches.map((record) => record.completedIdleCount),
  );
  requireExactCount(
    decoderCommandBufferCount,
    attribution.totals.decoderCommandBufferCount,
    "decoder CB total",
  );
  requireExactCount(
    decoderSubmissionCount,
    attribution.totals.decoderSubmissionCount,
    "decoder submission total",
  );
  requireExactCount(
    decoderQueueDrainCount,
    attribution.totals.decoderQueueDrainCount,
    "decoder drain total",
  );
  requireExactCount(
    decoderRequestedIdleMilliseconds,
    attribution.totals.decoderRequestedIdleMilliseconds,
    "decoder requested idle total",
  );
  requireExactCount(
    decoderCompletedIdleCount,
    attribution.totals.decoderCompletedIdleCount,
    "decoder completed idle total",
  );

  const decoderWallMilliseconds = sum(
    decoderBatches.map((batch) => batch.wallMilliseconds),
  );
  const timedWindowMilliseconds = readback.decodeResolvedAt -
    trace.decoderBatches[0]!.encodeStartedAt;
  const accountedWindowMilliseconds = decoderWallMilliseconds +
    readbackTiming.wallMilliseconds;
  requireTimingReconciliation(
    timedWindowMilliseconds,
    accountedWindowMilliseconds,
    "full timed window",
  );
  const totals = Object.freeze({
    decoderCommandBufferCount,
    decoderSubmissionCount,
    decoderQueueDrainCount,
    decoderRequestedIdleMilliseconds,
    decoderCompletedIdleCount,
    readbackCommandBufferCount: readback.commandBufferCount,
    readbackSubmissionCount: readback.submissionCount,
    readbackQueueDrainCount: readback.queueDrainCount,
    totalCommandBufferCount:
      decoderCommandBufferCount + readback.commandBufferCount,
    totalSubmissionCount:
      decoderSubmissionCount + readback.submissionCount,
    totalQueueDrainCount:
      decoderQueueDrainCount + readback.queueDrainCount,
    decoderEncodeMilliseconds: sum(
      decoderBatches.map((batch) => batch.encodeMilliseconds),
    ),
    decoderSubmitThroughDrainMilliseconds: sum(
      decoderBatches.map((batch) => batch.submitThroughDrainMilliseconds),
    ),
    decoderActualIdleMilliseconds: sum(
      decoderBatches.map((batch) => batch.actualIdleMilliseconds),
    ),
    decoderResidualMilliseconds: sum(
      decoderBatches.map((batch) => batch.residualMilliseconds),
    ),
    decoderWallMilliseconds,
    readbackWallMilliseconds: readbackTiming.wallMilliseconds,
    timedWindowMilliseconds,
    wallReconciliationDeltaMilliseconds:
      timedWindowMilliseconds - accountedWindowMilliseconds,
  }) satisfies AceOpt0008VaeValidatedWindowTraceTotals;
  requireExactCount(
    totals.totalCommandBufferCount,
    attribution.totals.totalCommandBufferCount,
    "total CB count",
  );
  requireExactCount(
    totals.totalSubmissionCount,
    attribution.totals.totalSubmissionCount,
    "total submission count",
  );
  requireExactCount(
    totals.totalQueueDrainCount,
    attribution.totals.totalQueueDrainCount,
    "total drain count",
  );
  return Object.freeze({
    decoderBatches,
    readback: readbackTiming,
    totals,
  });
}

export function summarizeAceOpt0008VaeWindowTrace(
  attribution: AceOpt0008VaeWindowAttribution,
  trace: AceOpt0008VaeWindowTrace,
): AceOpt0008VaeWindowSummary {
  const validated = validateAceOpt0008VaeWindowTrace(attribution, trace);
  const pureEntries = attribution.batches.flatMap((batch, index) =>
    batch.classification === "pure"
      ? [{ batch, timing: validated.decoderBatches[index]! }]
      : []
  );
  const mixedBatches = Object.freeze(attribution.batches.flatMap((batch, index) =>
    batch.classification === "mixed"
      ? [Object.freeze({
          batch,
          timing: validated.decoderBatches[index]!,
        })]
      : []
  ));

  const byOperation = new Map<number, MutableOperationAggregate>();
  const byFamily = new Map<
    AceOpt0008VaeOperationFamily,
    MutableAggregateTotals
  >();
  const byKernel = new Map<
    AceOpt0008VaeSelectedKernel,
    MutableAggregateTotals
  >();
  for (const entry of pureEntries) {
    const membership = entry.batch.memberships[0]!;
    let operation = byOperation.get(membership.operationIndex);
    if (operation === undefined) {
      operation = {
        operationIndex: membership.operationIndex,
        operationLabel: membership.operationLabel,
        operationKind: membership.operationKind,
        family: membership.family,
        selectedKernels: new Set(),
        ...emptyAggregateTotals(),
      };
      byOperation.set(membership.operationIndex, operation);
    }
    operation.selectedKernels.add(membership.selectedKernel);
    addBatchToAggregate(operation, entry.batch, entry.timing);
    addBatchToAggregate(
      getOrCreateAggregate(byFamily, membership.family),
      entry.batch,
      entry.timing,
    );
    addBatchToAggregate(
      getOrCreateAggregate(byKernel, membership.selectedKernel),
      entry.batch,
      entry.timing,
    );
  }

  const operationAggregates = Object.freeze([...byOperation.values()]
    .sort((left, right) => left.operationIndex - right.operationIndex)
    .map((aggregate) => Object.freeze({
      operationIndex: aggregate.operationIndex,
      operationLabel: aggregate.operationLabel,
      operationKind: aggregate.operationKind,
      family: aggregate.family,
      selectedKernels: Object.freeze([...aggregate.selectedKernels]
        .sort((left, right) =>
          KERNEL_ORDER.indexOf(left) - KERNEL_ORDER.indexOf(right)
        )),
      ...freezeAggregateTotals(aggregate),
    })));
  const familyAggregates = Object.freeze(FAMILY_ORDER.flatMap((family) => {
    const aggregate = byFamily.get(family);
    return aggregate === undefined
      ? []
      : [Object.freeze({ family, ...freezeAggregateTotals(aggregate) })];
  }));
  const kernelAggregates = Object.freeze(KERNEL_ORDER.flatMap((selectedKernel) => {
    const aggregate = byKernel.get(selectedKernel);
    return aggregate === undefined
      ? []
      : [Object.freeze({
          selectedKernel,
          ...freezeAggregateTotals(aggregate),
        })];
  }));
  return Object.freeze({
    schemaVersion: 1,
    kind: "ace-opt-0008-vae-window-summary",
    attributionTotals: attribution.totals,
    traceTotals: validated.totals,
    pure: Object.freeze({
      batchCount: pureEntries.length,
      byOperation: operationAggregates,
      byFamily: familyAggregates,
      byKernel: kernelAggregates,
    }),
    mixed: Object.freeze({
      batchCount: mixedBatches.length,
      batches: mixedBatches,
    }),
    fixedBoundaries: Object.freeze({ readback: validated.readback }),
  });
}

export function stringifyAceOpt0008VaeWindowSummary(
  summary: AceOpt0008VaeWindowSummary,
): string {
  return `${JSON.stringify(canonicalizeJson(summary))}\n`;
}

function createBatchAttribution(
  batchIndex: number,
  quanta: readonly AceOpt0008VaeQuantumAttribution[],
): AceOpt0008VaeBatchAttribution {
  if (quanta.length < 1) {
    throw new Error(`ACE OPT-0008 batch ${batchIndex} is empty`);
  }
  const memberships: MutableMembership[] = [];
  for (const quantum of quanta) {
    const previous = memberships.at(-1);
    if (
      previous !== undefined &&
      previous.operationIndex === quantum.operationIndex &&
      previous.operationKind === quantum.operationKind &&
      previous.family === quantum.family &&
      previous.selectedKernel === quantum.selectedKernel
    ) {
      previous.lastQuantumIndex = quantum.quantumIndex;
      addQuantumToMembership(previous, quantum);
    } else {
      memberships.push({
        operationIndex: quantum.operationIndex,
        operationLabel: quantum.operationLabel,
        operationKind: quantum.operationKind,
        family: quantum.family,
        selectedKernel: quantum.selectedKernel,
        firstQuantumIndex: quantum.quantumIndex,
        lastQuantumIndex: quantum.quantumIndex,
        quantumCount: 0,
        primitiveCount: 0,
        passCount: 0,
        dispatchCount: 0,
        outputElementCount: 0,
        validMultiplyAccumulateCount: 0,
        estimatedMaximumMultiplyAccumulateCount: 0,
      });
      addQuantumToMembership(memberships.at(-1)!, quantum);
    }
  }
  const stableMemberships = Object.freeze(memberships.map((membership) =>
    Object.freeze({ ...membership })
  ));
  return Object.freeze({
    batchIndex,
    firstQuantumIndex: quanta[0]!.quantumIndex,
    lastQuantumIndex: quanta.at(-1)!.quantumIndex,
    quantumCount: quanta.length,
    classification: stableMemberships.length === 1 ? "pure" : "mixed",
    memberships: stableMemberships,
    primitiveCount: sum(quanta.map((quantum) => quantum.primitiveCount)),
    passCount: sum(quanta.map((quantum) => quantum.passCount)),
    dispatchCount: sum(quanta.map((quantum) => quantum.dispatchCount)),
    outputElementCount: sum(quanta.map((quantum) => quantum.outputElementCount)),
    validMultiplyAccumulateCount: sum(
      quanta.map((quantum) => quantum.validMultiplyAccumulateCount),
    ),
    estimatedMaximumMultiplyAccumulateCount: sum(
      quanta.map((quantum) =>
        quantum.estimatedMaximumMultiplyAccumulateCount
      ),
    ),
    expectedCommandBufferCount: 1,
    expectedSubmissionCount: 1,
    expectedQueueDrainCount: 1,
    expectedRequestedIdleMilliseconds:
      ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    expectedCompletedIdleCount: 1,
  });
}

interface MutableMembership {
  operationIndex: number;
  operationLabel: string;
  operationKind: AceVaeDecoderOperation["kind"];
  family: AceOpt0008VaeOperationFamily;
  selectedKernel: AceOpt0008VaeSelectedKernel;
  firstQuantumIndex: number;
  lastQuantumIndex: number;
  quantumCount: number;
  primitiveCount: number;
  passCount: number;
  dispatchCount: number;
  outputElementCount: number;
  validMultiplyAccumulateCount: number;
  estimatedMaximumMultiplyAccumulateCount: number;
}

function addQuantumToMembership(
  membership: MutableMembership,
  quantum: AceOpt0008VaeQuantumAttribution,
): void {
  membership.quantumCount += 1;
  membership.primitiveCount = checkedAdd(
    membership.primitiveCount,
    quantum.primitiveCount,
    "membership primitive count",
  );
  membership.passCount = checkedAdd(
    membership.passCount,
    quantum.passCount,
    "membership pass count",
  );
  membership.dispatchCount = checkedAdd(
    membership.dispatchCount,
    quantum.dispatchCount,
    "membership dispatch count",
  );
  membership.outputElementCount = checkedAdd(
    membership.outputElementCount,
    quantum.outputElementCount,
    "membership output count",
  );
  membership.validMultiplyAccumulateCount = checkedAdd(
    membership.validMultiplyAccumulateCount,
    quantum.validMultiplyAccumulateCount,
    "membership valid MAC count",
  );
  membership.estimatedMaximumMultiplyAccumulateCount = checkedAdd(
    membership.estimatedMaximumMultiplyAccumulateCount,
    quantum.estimatedMaximumMultiplyAccumulateCount,
    "membership maximum MAC count",
  );
}

function classifyQuantum(
  operation: AceVaeDecoderOperation,
  quantum: AceVaeDecoderQuantumPlan,
  limits: AceOpt0008VaeSelectorLimits,
): Readonly<{
  family: AceOpt0008VaeOperationFamily;
  selectedKernel: AceOpt0008VaeSelectedKernel;
}> {
  switch (operation.kind) {
    case "conv1d": {
      if (operation.shape.kernelSize !== 1 && operation.shape.kernelSize !== 7) {
        throw new Error(
          `ACE OPT-0008 cannot classify Conv1D K${operation.shape.kernelSize}`,
        );
      }
      if (quantum.primitives.length !== 1) {
        throw new Error("ACE OPT-0008 Conv1D quantum has multiple primitives");
      }
      const primitive = quantum.primitives[0]!;
      const range = { base: primitive.outputBase, count: primitive.outputCount };
      const selectedKernel = selectAceTiledVaeConv1d(
        limits,
        operation.shape,
        range,
      ).eligible
        ? "tiled-conv1d" as const
        : selectAceChannelChunkedVaeConv1d(
            limits,
            operation.shape,
            range,
          ).eligible
        ? "channel-chunked-conv1d" as const
        : "portable-conv1d" as const;
      return Object.freeze({
        family: operation.shape.kernelSize === 7
          ? "k7-conv1d" as const
          : "k1-conv1d" as const,
        selectedKernel,
      });
    }
    case "conv-transpose1d":
      return Object.freeze({
        family: "conv-transpose1d",
        selectedKernel: "portable-conv-transpose1d",
      });
    case "snake":
      return Object.freeze({
        family: "snake",
        selectedKernel: "portable-snake",
      });
    case "add":
      return Object.freeze({ family: "add", selectedKernel: "portable-add" });
  }
}

function countQuantumValidMacs(
  operation: AceVaeDecoderOperation,
  quantum: AceVaeDecoderQuantumPlan,
): number {
  switch (operation.kind) {
    case "conv1d":
      return countValidMacsInFlatRange(
        operation.shape,
        false,
        quantum.logicalOutputBase,
        quantum.logicalOutputCount,
      );
    case "conv-transpose1d":
      return countValidMacsInFlatRange(
        operation.shape,
        true,
        quantum.logicalOutputBase,
        quantum.logicalOutputCount,
      );
    case "snake":
    case "add":
      return 0;
  }
}

function countValidMacsInFlatRange(
  shape: AceVaeConv1dShape | AceVaeConvTranspose1dShape,
  transpose: boolean,
  outputBase: number,
  outputCount: number,
): number {
  const outputFrames = transpose
    ? planAceVaeConvTranspose1d(shape as AceVaeConvTranspose1dShape).outputFrames
    : planAceVaeConv1d(shape as AceVaeConv1dShape).outputFrames;
  const outputChannels = shape.outputChannels;
  const outputElements = checkedProduct(
    [shape.batch, outputFrames, outputChannels],
    "operation output elements",
  );
  const outputEnd = checkedAdd(outputBase, outputCount, "quantum output end");
  if (
    !Number.isSafeInteger(outputBase) ||
    outputBase < 0 ||
    !Number.isSafeInteger(outputCount) ||
    outputCount <= 0 ||
    outputEnd > outputElements
  ) {
    throw new RangeError("ACE OPT-0008 convolution range is invalid");
  }
  const firstRow = Math.floor(outputBase / outputChannels);
  const lastRow = Math.floor((outputEnd - 1) / outputChannels);
  const firstChannel = outputBase % outputChannels;
  const lastChannelEnd = ((outputEnd - 1) % outputChannels) + 1;
  let channelTaps: number;
  if (firstRow === lastRow) {
    channelTaps = checkedProduct([
      validTapCountForRow(shape, transpose, outputFrames, firstRow),
      outputCount,
    ], "single-row channel taps", true);
  } else {
    channelTaps = checkedProduct([
      validTapCountForRow(shape, transpose, outputFrames, firstRow),
      outputChannels - firstChannel,
    ], "first-row channel taps", true);
    channelTaps = checkedAdd(
      channelTaps,
      checkedProduct([
        validTapCountForRow(shape, transpose, outputFrames, lastRow),
        lastChannelEnd,
      ], "last-row channel taps", true),
      "range edge channel taps",
    );
    if (lastRow > firstRow + 1) {
      channelTaps = checkedAdd(
        channelTaps,
        checkedProduct([
          sumValidTapsForRows(
            shape,
            transpose,
            outputFrames,
            firstRow + 1,
            lastRow,
          ),
          outputChannels,
        ], "complete-row channel taps", true),
        "range channel taps",
      );
    }
  }
  return checkedProduct(
    [channelTaps, shape.inputChannels],
    "quantum valid MACs",
    true,
  );
}

function validTapCountForRow(
  shape: AceVaeConv1dShape | AceVaeConvTranspose1dShape,
  transpose: boolean,
  outputFrames: number,
  row: number,
): number {
  const outputTime = row % outputFrames;
  return sumValidTapsForTimes(
    shape,
    transpose,
    outputTime,
    outputTime + 1,
  );
}

function sumValidTapsForRows(
  shape: AceVaeConv1dShape | AceVaeConvTranspose1dShape,
  transpose: boolean,
  outputFrames: number,
  rowStart: number,
  rowEnd: number,
): number {
  let cursor = rowStart;
  let total = 0;
  while (cursor < rowEnd) {
    const batchIndex = Math.floor(cursor / outputFrames);
    if (batchIndex >= shape.batch) {
      throw new RangeError("ACE OPT-0008 convolution row exceeds batch");
    }
    const batchEnd = Math.min(rowEnd, (batchIndex + 1) * outputFrames);
    total = checkedAdd(
      total,
      sumValidTapsForTimes(
        shape,
        transpose,
        cursor % outputFrames,
        (batchEnd - 1) % outputFrames + 1,
      ),
      "valid row taps",
    );
    cursor = batchEnd;
  }
  return total;
}

function sumValidTapsForTimes(
  shape: AceVaeConv1dShape | AceVaeConvTranspose1dShape,
  transpose: boolean,
  timeStart: number,
  timeEnd: number,
): number {
  let total = 0;
  for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
    const kernelOffset = kernel * shape.dilation;
    if (transpose) {
      const firstInput = Math.max(
        0,
        Math.ceil((timeStart + shape.padding - kernelOffset) / shape.stride),
      );
      const lastInput = Math.min(
        shape.inputFrames - 1,
        Math.floor(
          (timeEnd - 1 + shape.padding - kernelOffset) / shape.stride,
        ),
      );
      if (lastInput >= firstInput) total += lastInput - firstInput + 1;
    } else {
      const firstTime = Math.max(
        timeStart,
        Math.ceil((shape.padding - kernelOffset) / shape.stride),
      );
      const lastTime = Math.min(
        timeEnd - 1,
        Math.floor(
          (shape.inputFrames - 1 + shape.padding - kernelOffset) /
            shape.stride,
        ),
      );
      if (lastTime >= firstTime) total += lastTime - firstTime + 1;
    }
  }
  return total;
}

function operationOutputElements(operation: AceVaeDecoderOperation): number {
  switch (operation.kind) {
    case "conv1d":
      return planAceVaeConv1d(operation.shape).outputElements;
    case "conv-transpose1d":
      return planAceVaeConvTranspose1d(operation.shape).outputElements;
    case "snake":
    case "add":
      return checkedProduct([
        operation.shape.batch,
        operation.shape.frames,
        operation.shape.channels,
      ], `${operation.label} output elements`);
  }
}

interface MutableAggregateTotals {
  batchCount: number;
  quantumCount: number;
  primitiveCount: number;
  passCount: number;
  dispatchCount: number;
  outputElementCount: number;
  validMultiplyAccumulateCount: number;
  estimatedMaximumMultiplyAccumulateCount: number;
  commandBufferCount: number;
  submissionCount: number;
  queueDrainCount: number;
  requestedIdleMilliseconds: number;
  completedIdleCount: number;
  encodeMilliseconds: number;
  submitThroughDrainMilliseconds: number;
  actualIdleMilliseconds: number;
  residualMilliseconds: number;
  wallMilliseconds: number;
}

interface MutableOperationAggregate extends MutableAggregateTotals {
  operationIndex: number;
  operationLabel: string;
  operationKind: AceVaeDecoderOperation["kind"];
  family: AceOpt0008VaeOperationFamily;
  selectedKernels: Set<AceOpt0008VaeSelectedKernel>;
}

function emptyAggregateTotals(): MutableAggregateTotals {
  return {
    batchCount: 0,
    quantumCount: 0,
    primitiveCount: 0,
    passCount: 0,
    dispatchCount: 0,
    outputElementCount: 0,
    validMultiplyAccumulateCount: 0,
    estimatedMaximumMultiplyAccumulateCount: 0,
    commandBufferCount: 0,
    submissionCount: 0,
    queueDrainCount: 0,
    requestedIdleMilliseconds: 0,
    completedIdleCount: 0,
    encodeMilliseconds: 0,
    submitThroughDrainMilliseconds: 0,
    actualIdleMilliseconds: 0,
    residualMilliseconds: 0,
    wallMilliseconds: 0,
  };
}

function getOrCreateAggregate<Key>(
  map: Map<Key, MutableAggregateTotals>,
  key: Key,
): MutableAggregateTotals {
  let aggregate = map.get(key);
  if (aggregate === undefined) {
    aggregate = emptyAggregateTotals();
    map.set(key, aggregate);
  }
  return aggregate;
}

function addBatchToAggregate(
  aggregate: MutableAggregateTotals,
  batch: AceOpt0008VaeBatchAttribution,
  timing: AceOpt0008VaeDecoderBatchTiming,
): void {
  aggregate.batchCount += 1;
  aggregate.quantumCount += batch.quantumCount;
  aggregate.primitiveCount += batch.primitiveCount;
  aggregate.passCount += batch.passCount;
  aggregate.dispatchCount += batch.dispatchCount;
  aggregate.outputElementCount += batch.outputElementCount;
  aggregate.validMultiplyAccumulateCount +=
    batch.validMultiplyAccumulateCount;
  aggregate.estimatedMaximumMultiplyAccumulateCount +=
    batch.estimatedMaximumMultiplyAccumulateCount;
  aggregate.commandBufferCount += batch.expectedCommandBufferCount;
  aggregate.submissionCount += batch.expectedSubmissionCount;
  aggregate.queueDrainCount += batch.expectedQueueDrainCount;
  aggregate.requestedIdleMilliseconds +=
    batch.expectedRequestedIdleMilliseconds;
  aggregate.completedIdleCount += batch.expectedCompletedIdleCount;
  aggregate.encodeMilliseconds += timing.encodeMilliseconds;
  aggregate.submitThroughDrainMilliseconds +=
    timing.submitThroughDrainMilliseconds;
  aggregate.actualIdleMilliseconds += timing.actualIdleMilliseconds;
  aggregate.residualMilliseconds += timing.residualMilliseconds;
  aggregate.wallMilliseconds += timing.wallMilliseconds;
}

function freezeAggregateTotals(
  aggregate: MutableAggregateTotals,
): AceOpt0008VaeAggregateTotals {
  for (const [key, value] of Object.entries(aggregate)) {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new RangeError(`ACE OPT-0008 aggregate ${key} is not finite`);
    }
  }
  return Object.freeze({
    batchCount: aggregate.batchCount,
    quantumCount: aggregate.quantumCount,
    primitiveCount: aggregate.primitiveCount,
    passCount: aggregate.passCount,
    dispatchCount: aggregate.dispatchCount,
    outputElementCount: aggregate.outputElementCount,
    validMultiplyAccumulateCount: aggregate.validMultiplyAccumulateCount,
    estimatedMaximumMultiplyAccumulateCount:
      aggregate.estimatedMaximumMultiplyAccumulateCount,
    commandBufferCount: aggregate.commandBufferCount,
    submissionCount: aggregate.submissionCount,
    queueDrainCount: aggregate.queueDrainCount,
    requestedIdleMilliseconds: aggregate.requestedIdleMilliseconds,
    completedIdleCount: aggregate.completedIdleCount,
    encodeMilliseconds: aggregate.encodeMilliseconds,
    submitThroughDrainMilliseconds:
      aggregate.submitThroughDrainMilliseconds,
    actualIdleMilliseconds: aggregate.actualIdleMilliseconds,
    residualMilliseconds: aggregate.residualMilliseconds,
    wallMilliseconds: aggregate.wallMilliseconds,
  });
}

function validateSelectorLimits(limits: AceOpt0008VaeSelectorLimits): void {
  for (const key of SELECTOR_LIMIT_KEYS) {
    const value = limits[key];
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`ACE OPT-0008 selector limit ${key} is invalid`);
    }
  }
}

function requireTimeline(values: readonly number[], label: string): void {
  let previous = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || value < previous) {
      throw new RangeError(`ACE OPT-0008 ${label} timeline is invalid`);
    }
    previous = value;
  }
}

function requireExactCount(actual: number, expected: number, label: string): void {
  if (!Number.isSafeInteger(actual) || actual !== expected) {
    throw new Error(
      `ACE OPT-0008 ${label} is ${actual}; expected ${expected}`,
    );
  }
}

function requireSameTimestamp(
  actual: number,
  expected: number,
  label: string,
): void {
  if (!nearlyEqual(actual, expected)) {
    throw new Error(
      `ACE OPT-0008 ${label} is ${actual}; expected ${expected}`,
    );
  }
}

function requireTimingReconciliation(
  actual: number,
  expected: number,
  label: string,
): void {
  if (!nearlyEqual(actual, expected)) {
    throw new Error(`ACE OPT-0008 ${label} timing does not reconcile`);
  }
}

function nearlyEqual(left: number, right: number): boolean {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) <= scale * 1e-12;
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new RangeError(`ACE OPT-0008 ${label} exceeds safe integers`);
  }
  return result;
}

function checkedProduct(
  values: readonly number[],
  label: string,
  allowZero = false,
): number {
  let result = 1;
  for (const value of values) {
    if (
      !Number.isSafeInteger(value) ||
      value < 0 ||
      (!allowZero && value === 0)
    ) {
      throw new RangeError(`ACE OPT-0008 ${label} contains an invalid value`);
    }
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`ACE OPT-0008 ${label} exceeds safe integers`);
    }
  }
  return result;
}

function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError("ACE OPT-0008 aggregate contains an invalid value");
    }
    total += value;
    if (!Number.isFinite(total)) {
      throw new RangeError("ACE OPT-0008 aggregate is not finite");
    }
  }
  return total;
}

type CanonicalJsonValue =
  | null
  | boolean
  | number
  | string
  | readonly CanonicalJsonValue[]
  | { readonly [key: string]: CanonicalJsonValue };

function canonicalizeJson(value: unknown): CanonicalJsonValue {
  if (
    value === null ||
    typeof value === "boolean" ||
    typeof value === "string"
  ) {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new RangeError("ACE OPT-0008 summary contains a non-finite number");
    }
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeJson(entry));
  }
  if (typeof value !== "object") {
    throw new TypeError("ACE OPT-0008 summary contains a non-JSON value");
  }
  return Object.fromEntries(Object.entries(value)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, entry]) => [key, canonicalizeJson(entry)]));
}
