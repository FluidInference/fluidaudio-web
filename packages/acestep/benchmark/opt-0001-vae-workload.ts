import {
  ACE_OOBLECK_DECODER_CONFIG,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderQuantumWorkPolicy,
  type AceVaeDecoderOperation,
  type AceVaeTransposePartGeometry,
} from "../src/webgpu/vae-decoder.js";
import {
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
  type AceVaeConv1dShape,
  type AceVaeConvTranspose1dShape,
} from "../src/webgpu/kernels/vae-primitives.js";
import { ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS } from
  "../src/runtime/scheduler.js";
import {
  summarizeBenchmarkSamples,
  type BenchmarkSampleSummary,
} from "./result.js";

/** Benchmark-only reconstruction of the accepted Stage-1 uniform cap. */
export const ACE_OPT_0001_VAE_LEGACY_UNIFORM_QUANTUM_WORK_POLICY =
  Object.freeze({
    maximumConvolutionMultiplyAccumulates: 234_881_024,
    maximumOutputElements: 32_768,
  }) satisfies AceVaeDecoderQuantumWorkPolicy;

export const ACE_OPT_0001_VAE_TRANSPOSE_PARTS = Object.freeze({
  "vae.decoder.block.0.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 614 }),
    Object.freeze({ partStart: 614, partEnd: 1_024 }),
  ]),
  "vae.decoder.block.1.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 512 }),
  ]),
  "vae.decoder.block.2.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 256 }),
  ]),
  "vae.decoder.block.3.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 128 }),
  ]),
  "vae.decoder.block.4.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 128 }),
  ]),
}) satisfies Readonly<Record<string, readonly AceVaeTransposePartGeometry[]>>;

export type AceOpt0001VaeOperationFamily =
  | "conv1d"
  | "conv-transpose1d"
  | "snake"
  | "add";

const OPERATION_FAMILY_ORDER = Object.freeze([
  "conv1d",
  "conv-transpose1d",
  "snake",
  "add",
] satisfies readonly AceOpt0001VaeOperationFamily[]);

export interface AceOpt0001VaeFixtureIdentity {
  readonly fixtureId: "ace-opt-0001-vae-window-256-v1";
  readonly experimentId: "OPT-0001";
  readonly aceSourceRevision: string;
  readonly aceMainModelSnapshot: string;
  readonly modelManifestSha256: string;
  readonly executionProfile: "reference-bf16-subgroups";
  readonly vaeStorageDtype: "float32";
  readonly decoderConfigId: string;
  readonly latentWindowFrames: 256;
  readonly quantumOutputElementCap: number;
  readonly transposeParts: Readonly<
    Record<string, readonly AceVaeTransposePartGeometry[]>
  >;
}

export interface AceOpt0001VaeConvShape {
  readonly type: "convolution";
  readonly batch: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly dilation: number;
  readonly padding: number;
  readonly outputPadding: number;
  readonly hasBias: boolean;
}

export interface AceOpt0001VaePointwiseShape {
  readonly type: "pointwise";
  readonly batch: number;
  readonly frames: number;
  readonly channels: number;
}

export type AceOpt0001VaeOperationShape =
  | AceOpt0001VaeConvShape
  | AceOpt0001VaePointwiseShape;

export interface AceOpt0001VaeOperationWork {
  readonly operationIndex: number;
  readonly label: string;
  readonly family: AceOpt0001VaeOperationFamily;
  readonly shape: AceOpt0001VaeOperationShape;
  readonly outputElements: number;
  /** MACs that the scalar production shader actually executes after padding tests. */
  readonly validMultiplyAccumulates: number;
  /** Full output × kernel × input-channel domain, including skipped padding taps. */
  readonly denseKernelMultiplyAccumulates: number;
  /** Two FLOPs per valid multiply-accumulate; bias and nonlinear work are excluded. */
  readonly convolutionFlops: number;
  readonly biasElements: number;
  readonly quantumCount: number;
  readonly primitiveDispatchCount: number;
  readonly configuredCooperativeIdleMilliseconds: number;
}

export interface AceOpt0001VaeFamilyWork {
  readonly family: AceOpt0001VaeOperationFamily;
  readonly operationCount: number;
  readonly outputElements: number;
  readonly validMultiplyAccumulates: number;
  readonly denseKernelMultiplyAccumulates: number;
  readonly convolutionFlops: number;
  readonly biasElements: number;
  readonly quantumCount: number;
  readonly primitiveDispatchCount: number;
  readonly configuredCooperativeIdleMilliseconds: number;
}

export interface AceOpt0001VaeQuantumClass {
  readonly id: string;
  /** Stable JSON suitable for matching a browser timing case to this class. */
  readonly signature: string;
  readonly family: AceOpt0001VaeOperationFamily;
  readonly operationLabels: readonly string[];
  readonly operationCount: number;
  readonly logicalOutputElementsPerQuantum: number;
  readonly primitiveDispatchesPerQuantum: number;
  readonly physicalOutputChannels: readonly number[];
  readonly logicalOutputBaseMinimum: number;
  readonly logicalOutputBaseMaximum: number;
  /** True when this representative intentionally spans distinct output offsets. */
  readonly collapsesLogicalOutputOffsets: boolean;
  readonly quantumCount: number;
  readonly logicalOutputElements: number;
  readonly primitiveDispatchCount: number;
}

export interface AceOpt0001VaeWindowTotals {
  readonly operationCount: number;
  readonly outputElements: number;
  readonly validMultiplyAccumulates: number;
  readonly denseKernelMultiplyAccumulates: number;
  readonly convolutionFlops: number;
  readonly biasElements: number;
  readonly decoderQuantumCount: number;
  readonly decoderPrimitiveDispatchCount: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  readonly queueDrainCount: number;
  readonly maximumOutstandingCommandBuffers: 1;
  readonly rangeControlRecordCount: number;
  /**
   * Requested timer duration. Every decoder quantum is non-final because the
   * readback command remains; actual timer wall time can be greater.
   */
  readonly configuredCooperativeIdleMilliseconds: number;
  readonly decodedAudioFrames: number;
  readonly decodedInterleavedElements: number;
  readonly decodedFloat32Bytes: number;
}

export interface AceOpt0001VaeWorkloadReport {
  readonly schemaVersion: 1;
  readonly reportKind: "ace-opt-0001-vae-static-workload";
  readonly identity: AceOpt0001VaeFixtureIdentity;
  readonly graph: {
    readonly batch: number;
    readonly inputFrames: number;
    readonly outputFrames: number;
    readonly hopLength: number;
    readonly parameterElements: number;
    readonly parameterBytes: number;
    readonly maximumActivationElements: number;
    readonly workspaceBytes: number;
    readonly allWorkspaceBytes: number;
  };
  readonly operationFamilies: readonly AceOpt0001VaeFamilyWork[];
  readonly operations: readonly AceOpt0001VaeOperationWork[];
  /**
   * Production quanta collapsed by shader shape, logical count, and physical
   * parts. Output offsets intentionally remain a measured source of variance.
   */
  readonly quantumClasses: readonly AceOpt0001VaeQuantumClass[];
  readonly totals: AceOpt0001VaeWindowTotals;
}

export interface AceOpt0001VaeQuantumTimingInput {
  readonly quantumClassId: string;
  /** CPU encode wall time for one representative quantum. */
  readonly encodeMilliseconds: readonly number[];
  /** Wall time from queue submission through its completion fence. */
  readonly submitThroughDrainMilliseconds: readonly number[];
}

export interface AceOpt0001VaeWeightedClassTiming {
  readonly quantumClassId: string;
  readonly quantumCount: number;
  readonly encode: BenchmarkSampleSummary;
  readonly submitThroughDrain: BenchmarkSampleSummary;
  readonly weightedMedianEncodeMilliseconds: number;
  readonly weightedMedianSubmitThroughDrainMilliseconds: number;
}

export interface AceOpt0001VaeRepresentativeTimingSummary {
  readonly schemaVersion: 1;
  readonly timingKind: "representative-quantum-weighted-component-model";
  readonly fixtureId: AceOpt0001VaeFixtureIdentity["fixtureId"];
  readonly measuredClassCount: number;
  readonly requiredClassCount: number;
  readonly measuredQuantumCount: number;
  readonly requiredQuantumCount: number;
  readonly quantumCoverageRatio: number;
  readonly missingQuantumClassIds: readonly string[];
  readonly classes: readonly AceOpt0001VaeWeightedClassTiming[];
  readonly coveredWeightedMedianEncodeMilliseconds: number;
  readonly coveredWeightedMedianSubmitThroughDrainMilliseconds: number;
  readonly completeWeightedMedianEncodeMilliseconds: number | null;
  readonly completeWeightedMedianSubmitThroughDrainMilliseconds: number | null;
  readonly configuredCooperativeIdleMilliseconds: number;
  /**
   * Not a wall-time prediction. This sums representative class medians and
   * configured timer durations while deliberately leaving the omissions below
   * explicit.
   */
  readonly completeWeightedComponentSumMilliseconds: number | null;
  readonly componentSumExcludes: readonly [
    "scheduler-and-owner-orchestration",
    "idle-timer-overshoot",
    "readback-and-map",
    "window-postprocessing",
  ];
}

/**
 * Deterministic, allocation-only description of one authenticated production
 * VAE window. It executes no shader and therefore replaces an otherwise
 * multi-minute scalar-window probe during optimization planning.
 */
export function createAceOpt0001VaeWorkloadReport(): AceOpt0001VaeWorkloadReport {
  const graph = planAceVaeDecoder(256, ACE_OOBLECK_DECODER_CONFIG, 1);
  const cooperative = planAceVaeDecoderQuanta(
    graph,
    ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
    ACE_OPT_0001_VAE_LEGACY_UNIFORM_QUANTUM_WORK_POLICY,
  );
  const quantumByOperation = graph.operations.map(() => ({
    quantumCount: 0,
    primitiveDispatchCount: 0,
  }));
  for (const quantum of cooperative.quanta) {
    const counters = quantumByOperation[quantum.operationIndex];
    if (counters === undefined) {
      throw new Error("ACE OPT-0001 VAE quantum references an absent operation");
    }
    counters.quantumCount += 1;
    counters.primitiveDispatchCount = checkedAdd(
      counters.primitiveDispatchCount,
      quantum.primitives.length,
      "operation primitive dispatch count",
    );
  }

  const operations = Object.freeze(graph.operations.map((operation, index) => {
    const counters = quantumByOperation[index]!;
    return profileOperation(index, operation, counters);
  }));
  const operationFamilies = Object.freeze(OPERATION_FAMILY_ORDER.map((family) =>
    aggregateFamily(family, operations)
  ));
  const quantumClasses = collapseQuantumClasses(graph.operations, cooperative.quanta);
  const decoderQuantumCount = cooperative.quantumCount;
  const totalCommandBufferCount = checkedAdd(
    decoderQuantumCount,
    1,
    "VAE total command-buffer count",
  );
  const totals = Object.freeze({
    operationCount: graph.primitiveCount,
    outputElements: sum(operations.map((operation) => operation.outputElements)),
    validMultiplyAccumulates: sum(
      operations.map((operation) => operation.validMultiplyAccumulates),
    ),
    denseKernelMultiplyAccumulates: sum(
      operations.map((operation) => operation.denseKernelMultiplyAccumulates),
    ),
    convolutionFlops: sum(
      operations.map((operation) => operation.convolutionFlops),
    ),
    biasElements: sum(operations.map((operation) => operation.biasElements)),
    decoderQuantumCount,
    decoderPrimitiveDispatchCount: cooperative.primitiveDispatchCount,
    decoderCommandBufferCount: decoderQuantumCount,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount,
    queueDrainCount: totalCommandBufferCount,
    maximumOutstandingCommandBuffers: 1,
    rangeControlRecordCount: cooperative.primitiveDispatchCount,
    configuredCooperativeIdleMilliseconds:
      decoderQuantumCount * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    decodedAudioFrames: graph.outputFrames,
    decodedInterleavedElements: graph.outputElements,
    decodedFloat32Bytes: checkedMultiply(
      graph.outputElements,
      Float32Array.BYTES_PER_ELEMENT,
      "decoded FP32 bytes",
    ),
  }) satisfies AceOpt0001VaeWindowTotals;

  return Object.freeze({
    schemaVersion: 1,
    reportKind: "ace-opt-0001-vae-static-workload",
    identity: Object.freeze({
      fixtureId: "ace-opt-0001-vae-window-256-v1",
      experimentId: "OPT-0001",
      aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      aceMainModelSnapshot: "19671f406d603126926c1b7e2adc169acbcade22",
      modelManifestSha256:
        "d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55",
      executionProfile: "reference-bf16-subgroups",
      vaeStorageDtype: "float32",
      decoderConfigId: graph.config.id,
      latentWindowFrames: 256,
      quantumOutputElementCap: 32_768,
      transposeParts: ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
    }),
    graph: Object.freeze({
      batch: graph.batch,
      inputFrames: graph.inputFrames,
      outputFrames: graph.outputFrames,
      hopLength: graph.hopLength,
      parameterElements: graph.parameterElements,
      parameterBytes: graph.parameterBytes,
      maximumActivationElements: graph.maximumActivationElements,
      workspaceBytes: graph.workspaceBytes,
      allWorkspaceBytes: graph.allWorkspaceBytes,
    }),
    operationFamilies,
    operations,
    quantumClasses,
    totals,
  });
}

export function serializeAceOpt0001VaeWorkloadReport(
  report = createAceOpt0001VaeWorkloadReport(),
): string {
  return `${JSON.stringify(canonicalizeJson(report), undefined, 2)}\n`;
}

/**
 * Weight short samples from each representative shape group by its production
 * repetition count. Callers should sample multiple member offsets/phases when
 * a class says it collapses them. Missing classes remain explicit, and the
 * result is only a component model—not a complete decoder wall-time claim.
 */
export function summarizeAceOpt0001VaeRepresentativeTimings(
  report: AceOpt0001VaeWorkloadReport,
  inputs: readonly AceOpt0001VaeQuantumTimingInput[],
): AceOpt0001VaeRepresentativeTimingSummary {
  const classesById = new Map(
    report.quantumClasses.map((quantumClass) => [quantumClass.id, quantumClass]),
  );
  const seen = new Set<string>();
  const weighted: AceOpt0001VaeWeightedClassTiming[] = [];
  for (const input of inputs) {
    const quantumClass = classesById.get(input.quantumClassId);
    if (quantumClass === undefined) {
      throw new Error(
        `ACE OPT-0001 VAE timing references unknown class ${input.quantumClassId}`,
      );
    }
    if (seen.has(input.quantumClassId)) {
      throw new Error(
        `ACE OPT-0001 VAE timing repeats class ${input.quantumClassId}`,
      );
    }
    seen.add(input.quantumClassId);
    const encode = freezeSampleSummary(summarizeBenchmarkSamples(
      input.encodeMilliseconds,
    ));
    const submitThroughDrain = freezeSampleSummary(summarizeBenchmarkSamples(
      input.submitThroughDrainMilliseconds,
    ));
    weighted.push(Object.freeze({
      quantumClassId: input.quantumClassId,
      quantumCount: quantumClass.quantumCount,
      encode,
      submitThroughDrain,
      weightedMedianEncodeMilliseconds:
        encode.median * quantumClass.quantumCount,
      weightedMedianSubmitThroughDrainMilliseconds:
        submitThroughDrain.median * quantumClass.quantumCount,
    }));
  }
  weighted.sort((left, right) =>
    compareStrings(left.quantumClassId, right.quantumClassId)
  );
  const missingQuantumClassIds = Object.freeze(report.quantumClasses
    .map((quantumClass) => quantumClass.id)
    .filter((id) => !seen.has(id)));
  const measuredQuantumCount = sum(weighted.map((entry) => entry.quantumCount));
  const coveredWeightedMedianEncodeMilliseconds = sum(
    weighted.map((entry) => entry.weightedMedianEncodeMilliseconds),
  );
  const coveredWeightedMedianSubmitThroughDrainMilliseconds = sum(
    weighted.map((entry) =>
      entry.weightedMedianSubmitThroughDrainMilliseconds
    ),
  );
  const complete = missingQuantumClassIds.length === 0;
  const completeWeightedMedianEncodeMilliseconds = complete
    ? coveredWeightedMedianEncodeMilliseconds
    : null;
  const completeWeightedMedianSubmitThroughDrainMilliseconds = complete
    ? coveredWeightedMedianSubmitThroughDrainMilliseconds
    : null;
  return Object.freeze({
    schemaVersion: 1,
    timingKind: "representative-quantum-weighted-component-model",
    fixtureId: report.identity.fixtureId,
    measuredClassCount: weighted.length,
    requiredClassCount: report.quantumClasses.length,
    measuredQuantumCount,
    requiredQuantumCount: report.totals.decoderQuantumCount,
    quantumCoverageRatio:
      measuredQuantumCount / report.totals.decoderQuantumCount,
    missingQuantumClassIds,
    classes: Object.freeze(weighted),
    coveredWeightedMedianEncodeMilliseconds,
    coveredWeightedMedianSubmitThroughDrainMilliseconds,
    completeWeightedMedianEncodeMilliseconds,
    completeWeightedMedianSubmitThroughDrainMilliseconds,
    configuredCooperativeIdleMilliseconds:
      report.totals.configuredCooperativeIdleMilliseconds,
    completeWeightedComponentSumMilliseconds: complete
      ? coveredWeightedMedianEncodeMilliseconds +
        coveredWeightedMedianSubmitThroughDrainMilliseconds +
        report.totals.configuredCooperativeIdleMilliseconds
      : null,
    componentSumExcludes: Object.freeze([
      "scheduler-and-owner-orchestration",
      "idle-timer-overshoot",
      "readback-and-map",
      "window-postprocessing",
    ] as const),
  });
}

function profileOperation(
  operationIndex: number,
  operation: AceVaeDecoderOperation,
  counters: { readonly quantumCount: number; readonly primitiveDispatchCount: number },
): AceOpt0001VaeOperationWork {
  let shape: AceOpt0001VaeOperationShape;
  let outputElements: number;
  let validMultiplyAccumulates = 0;
  let denseKernelMultiplyAccumulates = 0;
  let biasElements = 0;
  switch (operation.kind) {
    case "conv1d": {
      const plan = planAceVaeConv1d(operation.shape);
      shape = profileConvShape(operation.shape, plan.outputFrames, 0,
        operation.bias !== undefined);
      outputElements = plan.outputElements;
      validMultiplyAccumulates = countValidConv1dMacs(operation.shape);
      denseKernelMultiplyAccumulates = checkedMultiply(
        outputElements,
        checkedMultiply(
          operation.shape.kernelSize,
          operation.shape.inputChannels,
          `${operation.label} dense reduction width`,
        ),
        `${operation.label} dense MACs`,
      );
      biasElements = operation.bias === undefined ? 0 : outputElements;
      break;
    }
    case "conv-transpose1d": {
      const plan = planAceVaeConvTranspose1d(operation.shape);
      shape = profileConvShape(
        operation.shape,
        plan.outputFrames,
        operation.shape.outputPadding,
        true,
      );
      outputElements = plan.outputElements;
      validMultiplyAccumulates = countValidConvTranspose1dMacs(operation.shape);
      denseKernelMultiplyAccumulates = checkedMultiply(
        outputElements,
        checkedMultiply(
          operation.shape.kernelSize,
          operation.shape.inputChannels,
          `${operation.label} dense reduction width`,
        ),
        `${operation.label} dense MACs`,
      );
      biasElements = outputElements;
      break;
    }
    case "snake":
    case "add":
      shape = Object.freeze({ type: "pointwise", ...operation.shape });
      outputElements = checkedMultiply(
        checkedMultiply(
          operation.shape.batch,
          operation.shape.frames,
          `${operation.label} rows`,
        ),
        operation.shape.channels,
        `${operation.label} elements`,
      );
      break;
  }
  return Object.freeze({
    operationIndex,
    label: operation.label,
    family: operation.kind,
    shape,
    outputElements,
    validMultiplyAccumulates,
    denseKernelMultiplyAccumulates,
    convolutionFlops: checkedMultiply(
      validMultiplyAccumulates,
      2,
      `${operation.label} convolution FLOPs`,
    ),
    biasElements,
    quantumCount: counters.quantumCount,
    primitiveDispatchCount: counters.primitiveDispatchCount,
    configuredCooperativeIdleMilliseconds:
      counters.quantumCount * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  });
}

function profileConvShape(
  shape: AceVaeConv1dShape,
  outputFrames: number,
  outputPadding: number,
  hasBias: boolean,
): AceOpt0001VaeConvShape {
  return Object.freeze({
    type: "convolution",
    batch: shape.batch,
    inputFrames: shape.inputFrames,
    outputFrames,
    inputChannels: shape.inputChannels,
    outputChannels: shape.outputChannels,
    kernelSize: shape.kernelSize,
    stride: shape.stride,
    dilation: shape.dilation,
    padding: shape.padding,
    outputPadding,
    hasBias,
  });
}

function aggregateFamily(
  family: AceOpt0001VaeOperationFamily,
  operations: readonly AceOpt0001VaeOperationWork[],
): AceOpt0001VaeFamilyWork {
  const selected = operations.filter((operation) => operation.family === family);
  return Object.freeze({
    family,
    operationCount: selected.length,
    outputElements: sum(selected.map((operation) => operation.outputElements)),
    validMultiplyAccumulates: sum(
      selected.map((operation) => operation.validMultiplyAccumulates),
    ),
    denseKernelMultiplyAccumulates: sum(
      selected.map((operation) => operation.denseKernelMultiplyAccumulates),
    ),
    convolutionFlops: sum(
      selected.map((operation) => operation.convolutionFlops),
    ),
    biasElements: sum(selected.map((operation) => operation.biasElements)),
    quantumCount: sum(selected.map((operation) => operation.quantumCount)),
    primitiveDispatchCount: sum(
      selected.map((operation) => operation.primitiveDispatchCount),
    ),
    configuredCooperativeIdleMilliseconds: sum(
      selected.map((operation) =>
        operation.configuredCooperativeIdleMilliseconds
      ),
    ),
  });
}

function collapseQuantumClasses(
  operations: readonly AceVaeDecoderOperation[],
  quanta: ReturnType<typeof planAceVaeDecoderQuanta>["quanta"],
): readonly AceOpt0001VaeQuantumClass[] {
  interface MutableQuantumClass {
    readonly id: string;
    readonly signature: string;
    readonly family: AceOpt0001VaeOperationFamily;
    readonly operationLabels: Set<string>;
    readonly logicalOutputElementsPerQuantum: number;
    readonly primitiveDispatchesPerQuantum: number;
    readonly physicalOutputChannels: readonly number[];
    logicalOutputBaseMinimum: number;
    logicalOutputBaseMaximum: number;
    quantumCount: number;
    logicalOutputElements: number;
    primitiveDispatchCount: number;
  }
  const bySignature = new Map<string, MutableQuantumClass>();
  for (const quantum of quanta) {
    const operation = operations[quantum.operationIndex];
    if (operation === undefined) {
      throw new Error("ACE OPT-0001 VAE quantum class has no operation");
    }
    const physicalOutputChannels = Object.freeze(
      quantum.primitives.map((primitive) => primitive.outputChannels),
    );
    const signature = JSON.stringify({
      family: operation.kind,
      shape: timingShape(operation),
      logicalOutputElements: quantum.logicalOutputCount,
      physicalOutputChannels,
    });
    let aggregate = bySignature.get(signature);
    if (aggregate === undefined) {
      aggregate = {
        id: `vae-quantum-class-${bySignature.size.toString().padStart(3, "0")}`,
        signature,
        family: operation.kind,
        operationLabels: new Set<string>(),
        logicalOutputElementsPerQuantum: quantum.logicalOutputCount,
        primitiveDispatchesPerQuantum: quantum.primitives.length,
        physicalOutputChannels,
        logicalOutputBaseMinimum: quantum.logicalOutputBase,
        logicalOutputBaseMaximum: quantum.logicalOutputBase,
        quantumCount: 0,
        logicalOutputElements: 0,
        primitiveDispatchCount: 0,
      };
      bySignature.set(signature, aggregate);
    }
    aggregate.operationLabels.add(operation.label);
    aggregate.logicalOutputBaseMinimum = Math.min(
      aggregate.logicalOutputBaseMinimum,
      quantum.logicalOutputBase,
    );
    aggregate.logicalOutputBaseMaximum = Math.max(
      aggregate.logicalOutputBaseMaximum,
      quantum.logicalOutputBase,
    );
    aggregate.quantumCount += 1;
    aggregate.logicalOutputElements = checkedAdd(
      aggregate.logicalOutputElements,
      quantum.logicalOutputCount,
      "quantum-class logical elements",
    );
    aggregate.primitiveDispatchCount = checkedAdd(
      aggregate.primitiveDispatchCount,
      quantum.primitives.length,
      "quantum-class primitive dispatch count",
    );
  }
  return Object.freeze([...bySignature.values()].map((aggregate) =>
    Object.freeze({
      id: aggregate.id,
      signature: aggregate.signature,
      family: aggregate.family,
      operationLabels: Object.freeze([...aggregate.operationLabels]),
      operationCount: aggregate.operationLabels.size,
      logicalOutputElementsPerQuantum:
        aggregate.logicalOutputElementsPerQuantum,
      primitiveDispatchesPerQuantum: aggregate.primitiveDispatchesPerQuantum,
      physicalOutputChannels: aggregate.physicalOutputChannels,
      logicalOutputBaseMinimum: aggregate.logicalOutputBaseMinimum,
      logicalOutputBaseMaximum: aggregate.logicalOutputBaseMaximum,
      collapsesLogicalOutputOffsets:
        aggregate.logicalOutputBaseMinimum !== aggregate.logicalOutputBaseMaximum,
      quantumCount: aggregate.quantumCount,
      logicalOutputElements: aggregate.logicalOutputElements,
      primitiveDispatchCount: aggregate.primitiveDispatchCount,
    })
  ));
}

function timingShape(operation: AceVaeDecoderOperation): object {
  switch (operation.kind) {
    case "conv1d":
      return {
        ...operation.shape,
        outputFrames: planAceVaeConv1d(operation.shape).outputFrames,
        hasBias: operation.bias !== undefined,
      };
    case "conv-transpose1d":
      return {
        ...operation.shape,
        outputFrames: planAceVaeConvTranspose1d(operation.shape).outputFrames,
        hasBias: true,
      };
    case "snake":
    case "add":
      return operation.shape;
  }
}

function countValidConv1dMacs(shape: AceVaeConv1dShape): number {
  const outputFrames = planAceVaeConv1d(shape).outputFrames;
  let validTapsPerOutputChannel = 0;
  for (let outputTime = 0; outputTime < outputFrames; outputTime += 1) {
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      const paddedTime = outputTime * shape.stride + kernel * shape.dilation;
      const inputTime = paddedTime - shape.padding;
      if (inputTime >= 0 && inputTime < shape.inputFrames) {
        validTapsPerOutputChannel += 1;
      }
    }
  }
  return checkedMultiply(
    checkedMultiply(
      checkedMultiply(
        validTapsPerOutputChannel,
        shape.inputChannels,
        "Conv1d valid input-channel taps",
      ),
      shape.outputChannels,
      "Conv1d valid output-channel taps",
    ),
    shape.batch,
    "Conv1d valid MACs",
  );
}

function countValidConvTranspose1dMacs(
  shape: AceVaeConvTranspose1dShape,
): number {
  const outputFrames = planAceVaeConvTranspose1d(shape).outputFrames;
  let validTapsPerOutputChannel = 0;
  for (let outputTime = 0; outputTime < outputFrames; outputTime += 1) {
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      const inputNumerator = outputTime + shape.padding -
        kernel * shape.dilation;
      if (inputNumerator < 0 || inputNumerator % shape.stride !== 0) continue;
      const inputTime = inputNumerator / shape.stride;
      if (inputTime < shape.inputFrames) validTapsPerOutputChannel += 1;
    }
  }
  return checkedMultiply(
    checkedMultiply(
      checkedMultiply(
        validTapsPerOutputChannel,
        shape.inputChannels,
        "ConvTranspose1d valid input-channel taps",
      ),
      shape.outputChannels,
      "ConvTranspose1d valid output-channel taps",
    ),
    shape.batch,
    "ConvTranspose1d valid MACs",
  );
}

function freezeSampleSummary(
  summary: BenchmarkSampleSummary,
): BenchmarkSampleSummary {
  return Object.freeze({
    ...summary,
    samples: Object.freeze([...summary.samples]),
  });
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
      throw new RangeError("ACE OPT-0001 VAE JSON numbers must be finite");
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeJson(entry));
  }
  if (typeof value !== "object") {
    throw new TypeError("ACE OPT-0001 VAE report must contain only JSON values");
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => compareStrings(left, right))
      .map(([key, entry]) => [key, canonicalizeJson(entry)]),
  );
}

function compareStrings(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function sum(values: readonly number[]): number {
  return values.reduce(
    (total, value) => checkedAdd(total, value, "VAE workload aggregate"),
    0,
  );
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) && Number.isInteger(left) && Number.isInteger(right)) {
    throw new RangeError(`ACE OPT-0001 ${label} exceeds safe integer arithmetic`);
  }
  if (!Number.isFinite(result)) {
    throw new RangeError(`ACE OPT-0001 ${label} must remain finite`);
  }
  return result;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`ACE OPT-0001 ${label} exceeds safe integer arithmetic`);
  }
  return result;
}
