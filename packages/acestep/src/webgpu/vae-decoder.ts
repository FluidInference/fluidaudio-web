import {
  checkedAceProduct,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceU32,
  requirePositiveSafeInteger,
} from "./kernels/correctness-utils.js";
import {
  AceCorrectnessVaePrimitiveKernel,
  type AceVaeConv1dShape,
  type AceVaeConvTranspose1dShape,
  type AceVaePointwiseShape,
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
} from "./kernels/vae-primitives.js";
import {
  AceTiledVaeConv1dKernel,
  selectAceTiledVaeConv1d,
} from "./kernels/vae-conv1d.js";
import {
  AceChannelChunkedVaeConv1dKernel,
  selectAceChannelChunkedVaeConv1d,
} from "./kernels/vae-conv1d-channel-chunks.js";
import type { AcePackageManifest } from "../model/manifest.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const WORKSPACE_COUNT = 3;
const RANGE_CONTROL_BYTES = 16;

export interface AceVaeDecoderQuantumWorkPolicy {
  /** Conservative upper bound across every convolution output in one quantum. */
  readonly maximumConvolutionMultiplyAccumulates: number;
  /** Upper bound across convolution and pointwise output scalars. */
  readonly maximumOutputElements: number;
}

/**
 * OPT-0002 candidate/default cooperative work policy. The MAC budget equals
 * the largest accepted Stage-1 scalar quantum (`32,768 * 7 * 1,024`), while
 * inexpensive output domains may now fill a larger bounded command buffer.
 */
export const ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY = Object.freeze({
  maximumConvolutionMultiplyAccumulates: 234_881_024,
  maximumOutputElements: 1_048_576,
}) satisfies AceVaeDecoderQuantumWorkPolicy;

export interface AceVaeDecoderConfig {
  readonly id: string;
  readonly decoderInputChannels: number;
  readonly decoderChannels: number;
  readonly audioChannels: number;
  readonly channelMultiples: readonly number[];
  /** Encoder order. Decoder traversal uses the reverse. */
  readonly downsamplingRatios: readonly number[];
  readonly sampleRateHz: number;
}

/** Exact `vae/config.json` contract at ACE snapshot `19671f4…`. */
export const ACE_OOBLECK_DECODER_CONFIG: AceVaeDecoderConfig = Object.freeze({
  id: "ace-step-1.5-oobleck-decoder-v1",
  decoderInputChannels: 64,
  decoderChannels: 128,
  audioChannels: 2,
  channelMultiples: Object.freeze([1, 2, 4, 8, 16]),
  downsamplingRatios: Object.freeze([2, 4, 4, 6, 10]),
  sampleRateHz: 48_000,
});

export type AceVaeDecoderSlot =
  | "input"
  | "workspace-0"
  | "workspace-1"
  | "workspace-2"
  | "output";

interface AceVaeDecoderOperationBase {
  readonly label: string;
  readonly input: AceVaeDecoderSlot;
  readonly output: AceVaeDecoderSlot;
}

export interface AceVaeDecoderConvOperation
  extends AceVaeDecoderOperationBase {
  readonly kind: "conv1d";
  readonly shape: AceVaeConv1dShape;
  readonly weight: string;
  readonly bias?: string;
}

export interface AceVaeDecoderConvTransposeOperation
  extends AceVaeDecoderOperationBase {
  readonly kind: "conv-transpose1d";
  readonly shape: AceVaeConvTranspose1dShape;
  readonly weight: string;
  readonly bias: string;
}

export interface AceVaeDecoderSnakeOperation
  extends AceVaeDecoderOperationBase {
  readonly kind: "snake";
  readonly shape: AceVaePointwiseShape;
  readonly alpha: string;
  readonly beta: string;
}

export interface AceVaeDecoderAddOperation
  extends AceVaeDecoderOperationBase {
  readonly kind: "add";
  readonly right: AceVaeDecoderSlot;
  readonly shape: AceVaePointwiseShape;
}

export type AceVaeDecoderOperation =
  | AceVaeDecoderConvOperation
  | AceVaeDecoderConvTransposeOperation
  | AceVaeDecoderSnakeOperation
  | AceVaeDecoderAddOperation;

export interface AceVaeDecoderGraphPlan {
  readonly config: AceVaeDecoderConfig;
  readonly batch: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly hopLength: number;
  readonly inputElements: number;
  readonly outputElements: number;
  readonly maximumActivationElements: number;
  readonly workspaceBytes: number;
  readonly allWorkspaceBytes: number;
  readonly parameterElements: number;
  readonly parameterBytes: number;
  readonly primitiveCount: number;
  readonly operations: readonly AceVaeDecoderOperation[];
  readonly requiredTensorNames: readonly string[];
}

export interface AceVaeDecoderBindings {
  /** FP32 NLC `[batch,inputFrames,decoderInputChannels]`. */
  readonly input: GPUBufferBinding;
  /** FP32 NLC/interleaved `[batch,outputFrames,audioChannels]`. */
  readonly output: GPUBufferBinding;
  /** Three disjoint FP32 workspaces, each at least `plan.workspaceBytes`. */
  readonly workspaces: readonly [
    GPUBufferBinding,
    GPUBufferBinding,
    GPUBufferBinding,
  ];
  /**
   * Logical converter outputs. Ordinary tensors have one part; row-sharded
   * ConvTranspose weights expose every authenticated output-axis part.
   */
  readonly tensors: Readonly<Record<string, AceVaeLogicalTensorBinding>>;
}

export interface AceVaeTensorPartBinding {
  readonly binding: GPUBufferBinding;
  readonly partStart: number;
  readonly partEnd: number;
}

export type AceVaeTransposePartGeometry = Pick<
  AceVaeTensorPartBinding,
  "partStart" | "partEnd"
>;

export type AceVaeLogicalTensorBinding =
  | GPUBufferBinding
  | readonly AceVaeTensorPartBinding[];

export interface AceVaePhysicalTensorBinding {
  readonly physicalName: string;
  readonly binding: GPUBufferBinding;
}

export interface AceVaeDecoderDispatchOptions {
  /** Immutable work bounds used to partition the cooperative dispatch. */
  readonly quantumWorkPolicy?: AceVaeDecoderQuantumWorkPolicy;
  /** @internal Validation seam. Production selects proven optimized shapes. */
  readonly conv1dProfile?: "portable" | "optimized-when-eligible";
}

/** @internal Production-selection diagnostics used by exact integration gates. */
export interface AceVaeDecoderConv1dSelectionSummary {
  readonly profile: "portable" | "optimized-when-eligible";
  readonly tiledQuantumCount: number;
  readonly channelChunkedQuantumCount: number;
  readonly portableQuantumCount: number;
  readonly tiledOperationLabels: readonly string[];
  readonly channelChunkedOperationLabels: readonly string[];
  readonly portableOperationLabels: readonly string[];
  readonly fallbackReasons: Readonly<Record<string, number>>;
  readonly operations: readonly AceVaeDecoderConv1dOperationSelection[];
}

/** @internal */
export interface AceVaeDecoderConv1dOperationSelection {
  readonly label: string;
  readonly selection: "tiled" | "channel-chunked" | "portable" | "mixed";
  readonly reason: string;
  readonly quantumCount: number;
}

interface MutableAceVaeDecoderConv1dOperationSelection {
  label: string;
  selection: AceVaeDecoderConv1dOperationSelection["selection"];
  reason: string;
  quantumCount: number;
}

/** @internal Static view of the exact same fail-closed production selector. */
export function summarizeAceVaeDecoderConv1dOperationSelection(
  plan: AceVaeDecoderGraphPlan,
  limits: Parameters<typeof selectAceTiledVaeConv1d>[0],
  profile: "portable" | "optimized-when-eligible" =
    "optimized-when-eligible",
): readonly AceVaeDecoderConv1dOperationSelection[] {
  const cooperative = planAceVaeDecoderQuanta(plan);
  const byOperation = new Map<string, AceVaeDecoderConv1dOperationSelection>();
  for (const quantum of cooperative.quanta) {
    const operation = plan.operations[quantum.operationIndex]!;
    if (operation.kind !== "conv1d") continue;
    const primitive = quantum.primitives[0]!;
    const selected = selectDecoderConv1dQuantum(
      limits,
      profile,
      operation.shape,
      { base: primitive.outputBase, count: primitive.outputCount },
    );
    const previous = byOperation.get(operation.label);
    const selection = selected.selection;
    if (previous === undefined) {
      byOperation.set(operation.label, Object.freeze({
        label: operation.label,
        selection,
        reason: selected.reason,
        quantumCount: 1,
      }));
    } else {
      const changed = previous.selection !== selection ||
        previous.reason !== selected.reason;
      byOperation.set(operation.label, Object.freeze({
        ...previous,
        ...(changed
          ? { selection: "mixed" as const, reason: "mixed-per-quantum" }
          : {}),
        quantumCount: previous.quantumCount + 1,
      }));
    }
  }
  return Object.freeze([...byOperation.values()]);
}

export interface AceVaeDecoderQuantumPrimitivePlan {
  /** Sequential record in the immutable range-control uniform buffer. */
  readonly controlRecordIndex: number;
  /** Present only for a physical ConvTranspose output-channel weight part. */
  readonly physicalPartIndex?: number;
  readonly firstOutputChannel: number;
  readonly outputChannels: number;
  /** Range in this primitive's local flattened output domain. */
  readonly outputBase: number;
  readonly outputCount: number;
}

export interface AceVaeDecoderQuantumPlan {
  readonly index: number;
  readonly id: string;
  readonly operationIndex: number;
  readonly operationLabel: string;
  readonly operationKind: AceVaeDecoderOperation["kind"];
  /** Disjoint range in the operation's complete logical NLC output. */
  readonly logicalOutputBase: number;
  readonly logicalOutputCount: number;
  /** Conservative convolution work for this quantum; zero for pointwise work. */
  readonly estimatedMaximumMultiplyAccumulates: number;
  /** One entry normally; all physical transpose parts for a row-band. */
  readonly primitives: readonly AceVaeDecoderQuantumPrimitivePlan[];
}

export interface AceVaeDecoderCooperativePlan {
  readonly quantumWorkPolicy: AceVaeDecoderQuantumWorkPolicy;
  readonly quantumCount: number;
  readonly primitiveDispatchCount: number;
  readonly quanta: readonly AceVaeDecoderQuantumPlan[];
}

export interface AceVaeDecoderQuantum extends AceVaeDecoderQuantumPlan {
  readonly primitiveCount: number;
  /** Encode this quantum into its own compute pass and command buffer. */
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceVaeDecoderDispatch {
  readonly label: string;
  readonly plan: AceVaeDecoderGraphPlan;
  readonly cooperativePlan: AceVaeDecoderCooperativePlan;
  /**
   * FIFO production surface. Submit and fully drain exactly one entry before
   * the graph owner observes its required queue-empty interval.
   */
  readonly quanta: readonly AceVaeDecoderQuantum[];
  readonly primitiveCount: number;
  /** @internal Present on real runtime dispatches; omitted by coordinator stubs. */
  readonly conv1dSelection?: AceVaeDecoderConv1dSelectionSummary;
}

/**
 * Unfused, correctness-preserving decoder-only AutoencoderOobleck graph.
 *
 * This intentionally follows every upstream Snake, convolution, residual add,
 * and transposed convolution as a separate ordered operation. Proven tiled
 * Conv1D shapes may reuse storage reads, but each output retains the same FP32
 * source reduction order. Large output domains are subdivided only for
 * cooperative submission. Symmetric padding and activation arithmetic stay
 * FP32.
 */
export class AceCorrectnessVaeDecoderRuntime {
  private readonly primitives: AceCorrectnessVaePrimitiveKernel;
  private readonly tiledConv1d: AceTiledVaeConv1dKernel;
  private readonly channelChunkedConv1d: AceChannelChunkedVaeConv1dKernel;
  private readonly rangeControlBuffers = new Set<GPUBuffer>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {
    this.primitives = AceCorrectnessVaePrimitiveKernel.create(device);
    this.tiledConv1d = AceTiledVaeConv1dKernel.create(device);
    this.channelChunkedConv1d = AceChannelChunkedVaeConv1dKernel.create(device);
  }

  static create(device: GPUDevice): AceCorrectnessVaeDecoderRuntime {
    return new AceCorrectnessVaeDecoderRuntime(device);
  }

  async createDecoderDispatch(
    label: string,
    inputFrames: number,
    bindings: AceVaeDecoderBindings,
    config: AceVaeDecoderConfig = ACE_OOBLECK_DECODER_CONFIG,
    batch = 1,
    options: AceVaeDecoderDispatchOptions = {},
  ): Promise<AceVaeDecoderDispatch> {
    this.requireLive();
    const plan = planAceVaeDecoder(inputFrames, config, batch);
    const tensors = requireDecoderBindings(label, plan, bindings);
    const transposeParts = Object.fromEntries(
      plan.operations
        .filter((operation): operation is AceVaeDecoderConvTransposeOperation =>
          operation.kind === "conv-transpose1d")
        .map((operation) => [
          operation.weight,
          tensors[operation.weight]!.map(({ partStart, partEnd }) => ({
            partStart,
            partEnd,
          })),
        ]),
    );
    const cooperativePlan = planAceVaeDecoderQuanta(
      plan,
      transposeParts,
      options.quantumWorkPolicy,
    );
    const conv1dProfile = options.conv1dProfile ?? "optimized-when-eligible";
    if (
      conv1dProfile !== "portable" &&
      conv1dProfile !== "optimized-when-eligible"
    ) {
      throw new TypeError(
        `Unknown ACE VAE Conv1D profile ${String(conv1dProfile)}`,
      );
    }
    const rangeControls = await createRangeControlBuffer(
      this.device,
      cooperativePlan,
      `${label}-range-controls`,
    );
    let retainedRangeControls = false;
    try {
      this.requireLive();
      const quanta: AceVaeDecoderQuantum[] = [];
      let tiledQuantumCount = 0;
      let channelChunkedQuantumCount = 0;
      let portableQuantumCount = 0;
      const tiledOperationLabels = new Set<string>();
      const channelChunkedOperationLabels = new Set<string>();
      const portableOperationLabels = new Set<string>();
      const fallbackReasons: Record<string, number> = {};
      const operationSelections = new Map<
        string,
        MutableAceVaeDecoderConv1dOperationSelection
      >();
      const alignment = this.device.limits.minUniformBufferOffsetAlignment;
      for (const quantumPlan of cooperativePlan.quanta) {
        const operation = plan.operations[quantumPlan.operationIndex]!;
        const input = resolveSlot(bindings, operation.input);
        const output = resolveSlot(bindings, operation.output);
        const primitiveDispatches: Array<{
          encode(pass: GPUComputePassEncoder): void;
        }> = [];
        for (const primitivePlan of quantumPlan.primitives) {
          const range = {
            base: primitivePlan.outputBase,
            count: primitivePlan.outputCount,
            control: {
              buffer: rangeControls,
              offset: primitivePlan.controlRecordIndex * alignment,
              size: RANGE_CONTROL_BYTES,
            },
          } as const;
          switch (operation.kind) {
            case "conv1d":
              {
                const convBindings = {
                  input,
                  weight: requireSingleTensor(tensors, operation.weight),
                  ...(operation.bias === undefined
                    ? {}
                    : { bias: requireSingleTensor(tensors, operation.bias) }),
                  output,
                };
                const selection = selectDecoderConv1dQuantum(
                  this.device.limits,
                  conv1dProfile,
                  operation.shape,
                  range,
                );
                if (selection.selection === "tiled") {
                  primitiveDispatches.push(
                    await this.tiledConv1d.createDispatch(
                      `${label}-${quantumPlan.id}`,
                      operation.shape,
                      convBindings,
                      range,
                    ),
                  );
                  tiledQuantumCount += 1;
                  tiledOperationLabels.add(operation.label);
                  recordConv1dOperationSelection(
                    operationSelections,
                    operation.label,
                    "tiled",
                    selection.reason,
                  );
                } else if (selection.selection === "channel-chunked") {
                  primitiveDispatches.push(
                    await this.channelChunkedConv1d.createDispatch(
                      `${label}-${quantumPlan.id}`,
                      operation.shape,
                      convBindings,
                      range,
                    ),
                  );
                  channelChunkedQuantumCount += 1;
                  channelChunkedOperationLabels.add(operation.label);
                  recordConv1dOperationSelection(
                    operationSelections,
                    operation.label,
                    "channel-chunked",
                    selection.reason,
                  );
                } else {
                  primitiveDispatches.push(
                    await this.primitives.createConv1dDispatch(
                      `${label}-${quantumPlan.id}`,
                      operation.shape,
                      convBindings,
                      range,
                    ),
                  );
                  portableQuantumCount += 1;
                  portableOperationLabels.add(operation.label);
                  fallbackReasons[selection.reason] =
                    (fallbackReasons[selection.reason] ?? 0) + 1;
                  recordConv1dOperationSelection(
                    operationSelections,
                    operation.label,
                    "portable",
                    selection.reason,
                  );
                }
              }
              break;
            case "conv-transpose1d": {
              const physicalPartIndex = primitivePlan.physicalPartIndex;
              if (physicalPartIndex === undefined) {
                throw new Error(
                  `${quantumPlan.id} transpose quantum has no physical part`,
                );
              }
              const part = tensors[operation.weight]![physicalPartIndex];
              if (part === undefined) {
                throw new Error(
                  `${quantumPlan.id} references missing transpose part ${physicalPartIndex}`,
                );
              }
              primitiveDispatches.push(
                await this.primitives.createConvTranspose1dPartDispatch(
                  `${label}-${quantumPlan.id}-part-${physicalPartIndex}`,
                  {
                    ...operation.shape,
                    firstOutputChannel: part.partStart,
                    partOutputChannels: part.partEnd - part.partStart,
                  },
                  {
                    input,
                    weight: part.binding,
                    bias: requireSingleTensor(tensors, operation.bias),
                    output,
                  },
                  range,
                ),
              );
              break;
            }
            case "snake":
              primitiveDispatches.push(
                await this.primitives.createSnakeDispatch(
                  `${label}-${quantumPlan.id}`,
                  operation.shape,
                  {
                    input,
                    alpha: requireSingleTensor(tensors, operation.alpha),
                    beta: requireSingleTensor(tensors, operation.beta),
                    output,
                  },
                  range,
                ),
              );
              break;
            case "add":
              primitiveDispatches.push(
                await this.primitives.createAddDispatch(
                  `${label}-${quantumPlan.id}`,
                  operation.shape,
                  {
                    left: input,
                    right: resolveSlot(bindings, operation.right),
                    output,
                  },
                  range,
                ),
              );
              break;
          }
        }
        quanta.push(Object.freeze({
          ...quantumPlan,
          primitiveCount: primitiveDispatches.length,
          encode(pass: GPUComputePassEncoder): void {
            for (const dispatch of primitiveDispatches) dispatch.encode(pass);
          },
        }));
      }
      this.requireLive();
      this.rangeControlBuffers.add(rangeControls);
      retainedRangeControls = true;
      const frozenQuanta = Object.freeze(quanta);
      const conv1dSelection = Object.freeze({
        profile: conv1dProfile,
        tiledQuantumCount,
        channelChunkedQuantumCount,
        portableQuantumCount,
        tiledOperationLabels: Object.freeze([...tiledOperationLabels]),
        channelChunkedOperationLabels: Object.freeze([
          ...channelChunkedOperationLabels,
        ]),
        portableOperationLabels: Object.freeze([...portableOperationLabels]),
        fallbackReasons: Object.freeze({ ...fallbackReasons }),
        operations: Object.freeze([...operationSelections.values()].map(
          (operation) => Object.freeze({ ...operation }),
        )),
      });
      return Object.freeze({
        label,
        plan,
        cooperativePlan,
        quanta: frozenQuanta,
        primitiveCount: cooperativePlan.primitiveDispatchCount,
        conv1dSelection,
      });
    } finally {
      if (!retainedRangeControls) rangeControls.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.rangeControlBuffers) buffer.destroy();
    this.rangeControlBuffers.clear();
    this.primitives.destroy();
    this.tiledConv1d.destroy();
    this.channelChunkedConv1d.destroy();
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("ACE VAE correctness decoder runtime was destroyed");
    }
  }
}

/**
 * Partition the decoder into FIFO command-buffer quanta without changing any
 * primitive's reduction order. ConvTranspose channel shards stay together in
 * one logical row-band quantum, so the next operation cannot observe a
 * partially completed output row.
 */
export function planAceVaeDecoderQuanta(
  plan: AceVaeDecoderGraphPlan,
  transposeParts: Readonly<
    Record<string, readonly AceVaeTransposePartGeometry[]>
  > = {},
  requestedPolicy: AceVaeDecoderQuantumWorkPolicy =
    ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
): AceVaeDecoderCooperativePlan {
  const quantumWorkPolicy = snapshotAceVaeDecoderQuantumWorkPolicy(
    requestedPolicy,
  );
  requirePositiveSafeInteger(
    quantumWorkPolicy.maximumOutputElements,
    "ACE VAE quantum maximum output elements",
  );
  if (quantumWorkPolicy.maximumOutputElements > 0xffff_ffff) {
    throw new RangeError(
      "ACE VAE quantum maximum output elements exceed WGSL's u32 domain",
    );
  }
  const transposeNames = new Set(
    plan.operations
      .filter((operation): operation is AceVaeDecoderConvTransposeOperation =>
        operation.kind === "conv-transpose1d")
      .map((operation) => operation.weight),
  );
  const unexpectedParts = Object.keys(transposeParts)
    .filter((name) => !transposeNames.has(name));
  if (unexpectedParts.length > 0) {
    throw new Error(
      `ACE VAE quantum plan has parts for non-transpose tensors: ${unexpectedParts.sort().join(", ")}`,
    );
  }

  const quanta: AceVaeDecoderQuantumPlan[] = [];
  let controlRecordIndex = 0;
  for (const [operationIndex, operation] of plan.operations.entries()) {
    const outputElements = decoderOperationOutputElements(operation);
    let operationQuantumIndex = 0;
    if (operation.kind === "conv-transpose1d") {
      const outputChannels = operation.shape.outputChannels;
      const maximumTapsPerOutput = maximumCongruentTransposeTaps(
        operation.shape,
      );
      const maximumMacsPerOutput = checkedAceProduct(
        [maximumTapsPerOutput, operation.shape.inputChannels],
        `${operation.label} maximum MACs per output`,
      );
      const maximumMacsPerRow = checkedAceProduct(
        [outputChannels, maximumMacsPerOutput],
        `${operation.label} maximum MACs per output row`,
      );
      const rowsPerQuantum = Math.min(
        Math.floor(
          quantumWorkPolicy.maximumOutputElements / outputChannels,
        ),
        Math.floor(
          quantumWorkPolicy.maximumConvolutionMultiplyAccumulates /
            maximumMacsPerRow,
        ),
      );
      if (rowsPerQuantum < 1) {
        throw new RangeError(
          `ACE VAE quantum work policy cannot hold one complete ` +
            `${operation.label} output row (${outputChannels} elements, ` +
            `${maximumMacsPerRow} maximum multiply-accumulates)`,
        );
      }
      const parts = validateQuantumPartGeometry(
        operation.weight,
        transposeParts[operation.weight] ?? [{
          partStart: 0,
          partEnd: outputChannels,
        }],
        outputChannels,
      );
      const outputRows = checkedAceProduct(
        [operation.shape.batch, planAceVaeConvTranspose1d(operation.shape).outputFrames],
        `${operation.label} output rows`,
      );
      for (let rowBase = 0; rowBase < outputRows; rowBase += rowsPerQuantum) {
        const rowCount = Math.min(rowsPerQuantum, outputRows - rowBase);
        const primitives = parts.map((part, physicalPartIndex) => {
          const partChannels = part.partEnd - part.partStart;
          return Object.freeze({
            controlRecordIndex: controlRecordIndex++,
            physicalPartIndex,
            firstOutputChannel: part.partStart,
            outputChannels: partChannels,
            outputBase: rowBase * partChannels,
            outputCount: rowCount * partChannels,
          });
        });
        const logicalOutputBase = rowBase * outputChannels;
        const logicalOutputCount = rowCount * outputChannels;
        const estimatedMaximumMultiplyAccumulates = checkedAceProduct(
          [logicalOutputCount, maximumMacsPerOutput],
          `${operation.label} quantum maximum MACs`,
        );
        quanta.push(Object.freeze({
          index: quanta.length,
          id: vaeQuantumId(operationIndex, operation.label, operationQuantumIndex++),
          operationIndex,
          operationLabel: operation.label,
          operationKind: operation.kind,
          logicalOutputBase,
          logicalOutputCount,
          estimatedMaximumMultiplyAccumulates,
          primitives: Object.freeze(primitives),
        }));
      }
      continue;
    }

    const outputChannels = operation.kind === "conv1d"
      ? operation.shape.outputChannels
      : operation.shape.channels;
    const maximumMacsPerOutput = operation.kind === "conv1d"
      ? checkedAceProduct(
        [operation.shape.kernelSize, operation.shape.inputChannels],
        `${operation.label} maximum MACs per output`,
      )
      : 0;
    const outputElementsPerQuantum = operation.kind === "conv1d"
      ? Math.min(
        quantumWorkPolicy.maximumOutputElements,
        Math.floor(
          quantumWorkPolicy.maximumConvolutionMultiplyAccumulates /
            maximumMacsPerOutput,
        ),
      )
      : quantumWorkPolicy.maximumOutputElements;
    if (outputElementsPerQuantum < 1) {
      throw new RangeError(
        `ACE VAE quantum work policy cannot hold one ${operation.label} ` +
          `output requiring ${maximumMacsPerOutput} maximum multiply-accumulates`,
      );
    }
    for (
      let outputBase = 0;
      outputBase < outputElements;
      outputBase += outputElementsPerQuantum
    ) {
      const outputCount = Math.min(
        outputElementsPerQuantum,
        outputElements - outputBase,
      );
      const primitive = Object.freeze({
        controlRecordIndex: controlRecordIndex++,
        firstOutputChannel: 0,
        outputChannels,
        outputBase,
        outputCount,
      });
      quanta.push(Object.freeze({
        index: quanta.length,
        id: vaeQuantumId(operationIndex, operation.label, operationQuantumIndex++),
        operationIndex,
        operationLabel: operation.label,
        operationKind: operation.kind,
        logicalOutputBase: outputBase,
        logicalOutputCount: outputCount,
        estimatedMaximumMultiplyAccumulates: operation.kind === "conv1d"
          ? checkedAceProduct(
            [outputCount, maximumMacsPerOutput],
            `${operation.label} quantum maximum MACs`,
          )
          : 0,
        primitives: Object.freeze([primitive]),
      }));
    }
  }
  validatePlannedQuantumWork(quanta, quantumWorkPolicy);
  return Object.freeze({
    quantumWorkPolicy,
    quantumCount: quanta.length,
    primitiveDispatchCount: controlRecordIndex,
    quanta: Object.freeze(quanta),
  });
}

/** Copy and validate a caller-owned policy before any asynchronous work. */
export function snapshotAceVaeDecoderQuantumWorkPolicy(
  policy: AceVaeDecoderQuantumWorkPolicy =
    ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
): AceVaeDecoderQuantumWorkPolicy {
  const maximumConvolutionMultiplyAccumulates =
    policy.maximumConvolutionMultiplyAccumulates;
  const maximumOutputElements = policy.maximumOutputElements;
  requirePositiveSafeInteger(
    maximumConvolutionMultiplyAccumulates,
    "ACE VAE quantum maximum convolution multiply-accumulates",
  );
  requirePositiveSafeInteger(
    maximumOutputElements,
    "ACE VAE quantum maximum output elements",
  );
  return Object.freeze({
    maximumConvolutionMultiplyAccumulates,
    maximumOutputElements,
  });
}

function validatePlannedQuantumWork(
  quanta: readonly AceVaeDecoderQuantumPlan[],
  policy: AceVaeDecoderQuantumWorkPolicy,
): void {
  for (const quantum of quanta) {
    requirePositiveSafeInteger(
      quantum.logicalOutputCount,
      `${quantum.id} logical output count`,
    );
    requireAceU32(quantum.logicalOutputBase, `${quantum.id} logical output base`);
    requireAceU32(
      quantum.logicalOutputBase + quantum.logicalOutputCount,
      `${quantum.id} logical output end`,
    );
    if (quantum.logicalOutputCount > policy.maximumOutputElements) {
      throw new Error(`${quantum.id} exceeds its output-element work budget`);
    }
    if (
      !Number.isSafeInteger(quantum.estimatedMaximumMultiplyAccumulates) ||
      quantum.estimatedMaximumMultiplyAccumulates < 0
    ) {
      throw new Error(`${quantum.id} has an invalid convolution work estimate`);
    }
    if (
      quantum.estimatedMaximumMultiplyAccumulates >
        policy.maximumConvolutionMultiplyAccumulates
    ) {
      throw new Error(`${quantum.id} exceeds its convolution-MAC work budget`);
    }
    for (const primitive of quantum.primitives) {
      requireAceU32(primitive.outputBase, `${quantum.id} primitive output base`);
      requirePositiveSafeInteger(
        primitive.outputCount,
        `${quantum.id} primitive output count`,
      );
      requireAceU32(
        primitive.outputBase + primitive.outputCount,
        `${quantum.id} primitive output end`,
      );
    }
  }
}

function maximumCongruentTransposeTaps(
  shape: AceVaeConvTranspose1dShape,
): number {
  const outputFrames = planAceVaeConvTranspose1d(shape).outputFrames;
  const maximumKernelTime = (shape.kernelSize - 1) * shape.dilation;
  const maximumPaddedOutputTime = (outputFrames - 1) + shape.padding;
  if (
    !isU32(shape.kernelSize) ||
    !isU32(shape.stride) ||
    !isU32(shape.dilation) ||
    !isU32(shape.padding) ||
    !isU32(shape.inputFrames) ||
    !isU32(outputFrames) ||
    !isU32(maximumKernelTime) ||
    !isU32(maximumPaddedOutputTime)
  ) {
    // The scalar WGSL performs these products and sums in u32. If they can
    // wrap, mathematical modular-congruence counting is no longer a safe
    // upper bound, so conservatively charge every kernel tap.
    return shape.kernelSize;
  }
  const congruencePeriod = shape.stride / greatestCommonDivisor(
    shape.stride,
    shape.dilation,
  );
  return Math.ceil(shape.kernelSize / congruencePeriod);
}

function isU32(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0 && value <= 0xffff_ffff;
}

function greatestCommonDivisor(left: number, right: number): number {
  let remainderLeft = left;
  let remainderRight = right;
  while (remainderRight !== 0) {
    const next = remainderLeft % remainderRight;
    remainderLeft = remainderRight;
    remainderRight = next;
  }
  return remainderLeft;
}

function decoderOperationOutputElements(
  operation: AceVaeDecoderOperation,
): number {
  switch (operation.kind) {
    case "conv1d":
      return planAceVaeConv1d(operation.shape).outputElements;
    case "conv-transpose1d":
      return planAceVaeConvTranspose1d(operation.shape).outputElements;
    case "snake":
    case "add":
      return checkedAceProduct(
        [operation.shape.batch, operation.shape.frames, operation.shape.channels],
        `${operation.label} output`,
      );
  }
}

function validateQuantumPartGeometry(
  name: string,
  parts: readonly AceVaeTransposePartGeometry[],
  logicalRows: number,
): readonly AceVaeTransposePartGeometry[] {
  if (parts.length === 0) {
    throw new Error(`ACE VAE quantum tensor ${name} has no physical parts`);
  }
  let cursor = 0;
  const stable: AceVaeTransposePartGeometry[] = [];
  for (const part of parts) {
    if (
      !Number.isSafeInteger(part.partStart) ||
      !Number.isSafeInteger(part.partEnd) ||
      part.partStart !== cursor ||
      part.partEnd <= part.partStart ||
      part.partEnd > logicalRows
    ) {
      throw new RangeError(
        `ACE VAE quantum tensor ${name} parts must cover output channels contiguously in source order`,
      );
    }
    stable.push(Object.freeze({ ...part }));
    cursor = part.partEnd;
  }
  if (cursor !== logicalRows) {
    throw new RangeError(
      `ACE VAE quantum tensor ${name} parts end at ${cursor}, expected ${logicalRows}`,
    );
  }
  return Object.freeze(stable);
}

function vaeQuantumId(
  operationIndex: number,
  operationLabel: string,
  operationQuantumIndex: number,
): string {
  return `operation-${operationIndex}-${operationLabel}-quantum-${operationQuantumIndex}`;
}

type AceVaeDecoderConcreteConv1dSelection = Exclude<
  AceVaeDecoderConv1dOperationSelection["selection"],
  "mixed"
>;

interface AceVaeDecoderConv1dQuantumSelection {
  readonly selection: AceVaeDecoderConcreteConv1dSelection;
  readonly reason: string;
}

/** OPT-0004 retains priority; OPT-0005 fills only its fail-closed remainder. */
function selectDecoderConv1dQuantum(
  limits: Parameters<typeof selectAceTiledVaeConv1d>[0],
  profile: "portable" | "optimized-when-eligible",
  shape: AceVaeConv1dShape,
  range: Readonly<{ base: number; count: number }>,
): AceVaeDecoderConv1dQuantumSelection {
  if (profile === "portable") {
    return Object.freeze({
      selection: "portable",
      reason: "profile-portable",
    });
  }
  const tiled = selectAceTiledVaeConv1d(limits, shape, range);
  if (tiled.eligible) {
    return Object.freeze({ selection: "tiled", reason: "eligible" });
  }
  const channelChunked = selectAceChannelChunkedVaeConv1d(
    limits,
    shape,
    range,
  );
  if (channelChunked.eligible) {
    return Object.freeze({
      selection: "channel-chunked",
      reason: `tiled:${tiled.reason};channel-chunked:eligible`,
    });
  }
  return Object.freeze({
    selection: "portable",
    reason:
      `tiled:${tiled.reason};channel-chunked:${channelChunked.reason}`,
  });
}

function recordConv1dOperationSelection(
  selections: Map<
    string,
    MutableAceVaeDecoderConv1dOperationSelection
  >,
  label: string,
  selection: AceVaeDecoderConcreteConv1dSelection,
  reason: string,
): void {
  const previous = selections.get(label);
  if (previous === undefined) {
    selections.set(label, { label, selection, reason, quantumCount: 1 });
    return;
  }
  if (
    previous.selection !== "mixed" &&
    (previous.selection !== selection || previous.reason !== reason)
  ) {
    previous.selection = "mixed";
    previous.reason = "mixed-per-quantum";
  }
  previous.quantumCount += 1;
}

async function createRangeControlBuffer(
  device: GPUDevice,
  plan: AceVaeDecoderCooperativePlan,
  label: string,
): Promise<GPUBuffer> {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (
    !Number.isSafeInteger(alignment) ||
    alignment < RANGE_CONTROL_BYTES ||
    alignment % 4 !== 0
  ) {
    throw new RangeError(
      "ACE VAE range controls require a valid uniform-buffer alignment",
    );
  }
  if (device.limits.maxUniformBufferBindingSize < RANGE_CONTROL_BYTES) {
    throw new RangeError(
      "ACE VAE range controls exceed maxUniformBufferBindingSize",
    );
  }
  requirePositiveSafeInteger(
    plan.primitiveDispatchCount,
    "ACE VAE range-control record count",
  );
  const byteLength =
    (plan.primitiveDispatchCount - 1) * alignment + RANGE_CONTROL_BYTES;
  const lastDynamicOffset = (plan.primitiveDispatchCount - 1) * alignment;
  if (
    !Number.isSafeInteger(byteLength) ||
    lastDynamicOffset > 0xffff_ffff ||
    byteLength > device.limits.maxBufferSize
  ) {
    throw new RangeError("ACE VAE range-control buffer exceeds device limits");
  }
  const [buffer] = await createAceScopedBuffers(
    device,
    [{
      label,
      size: byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }],
    "ACE VAE range-control allocation",
  );
  const payload = new Uint32Array(byteLength / 4);
  const seen = new Uint8Array(plan.primitiveDispatchCount);
  for (const quantum of plan.quanta) {
    for (const primitive of quantum.primitives) {
      const record = primitive.controlRecordIndex;
      if (record >= seen.length || seen[record] !== 0) {
        buffer!.destroy();
        throw new Error("ACE VAE range-control records are not unique and dense");
      }
      seen[record] = 1;
      const wordOffset = (record * alignment) / 4;
      payload[wordOffset] = primitive.outputBase;
      payload[wordOffset + 1] = primitive.outputCount;
    }
  }
  if (seen.some((value) => value !== 1)) {
    buffer!.destroy();
    throw new Error("ACE VAE range-control records are not complete");
  }
  try {
    device.queue.writeBuffer(buffer!, 0, payload);
  } catch (error) {
    buffer!.destroy();
    throw error;
  }
  return buffer!;
}

export function planAceVaeDecoder(
  inputFrames: number,
  config: AceVaeDecoderConfig = ACE_OOBLECK_DECODER_CONFIG,
  batch = 1,
): AceVaeDecoderGraphPlan {
  validateDecoderConfig(config);
  requirePositiveSafeInteger(inputFrames, "ACE VAE decoder input frames");
  requirePositiveSafeInteger(batch, "ACE VAE decoder batch");

  const operations: AceVaeDecoderOperation[] = [];
  const requiredTensors = new Set<string>();
  const workspaceSlots = [
    "workspace-0",
    "workspace-1",
    "workspace-2",
  ] as const;
  let active: AceVaeDecoderSlot = "input";
  let frames = inputFrames;
  let channels = config.decoderInputChannels;
  let maximumActivationElements = 0;
  let parameterElements = 0;

  const chooseWorkspace = (
    excluded: ReadonlySet<AceVaeDecoderSlot>,
  ): AceVaeDecoderSlot => {
    const selected = workspaceSlots.find((slot) => !excluded.has(slot));
    if (selected === undefined) {
      throw new Error("ACE VAE decoder liveness requires more than three workspaces");
    }
    return selected;
  };
  const recordOutput = (slot: AceVaeDecoderSlot, elements: number): void => {
    if (slot.startsWith("workspace-")) {
      maximumActivationElements = Math.max(maximumActivationElements, elements);
    }
  };
  const recordTensor = (name: string, elements: number): string => {
    if (requiredTensors.has(name)) {
      throw new Error(`ACE VAE decoder tensor ${name} is consumed more than once`);
    }
    requiredTensors.add(name);
    parameterElements += elements;
    return name;
  };
  const appendConv = (
    operationLabel: string,
    prefix: string,
    inputSlot: AceVaeDecoderSlot,
    outputSlot: AceVaeDecoderSlot,
    shape: AceVaeConv1dShape,
    hasBias: boolean,
  ): void => {
    const convPlan = planAceVaeConv1d(shape);
    const weight = recordTensor(
      `vae.${prefix}.weight`,
      convPlan.weightElements,
    );
    const bias = hasBias
      ? recordTensor(`vae.${prefix}.bias`, shape.outputChannels)
      : undefined;
    operations.push(Object.freeze({
      kind: "conv1d",
      label: operationLabel,
      input: inputSlot,
      output: outputSlot,
      shape: Object.freeze({ ...shape }),
      weight,
      ...(bias === undefined ? {} : { bias }),
    }));
    recordOutput(outputSlot, convPlan.outputElements);
  };
  const appendSnake = (
    operationLabel: string,
    prefix: string,
    inputSlot: AceVaeDecoderSlot,
    outputSlot: AceVaeDecoderSlot,
    pointwiseShape: AceVaePointwiseShape,
  ): void => {
    const alpha = recordTensor(
      `vae.${prefix}.alpha`,
      pointwiseShape.channels,
    );
    const beta = recordTensor(
      `vae.${prefix}.beta`,
      pointwiseShape.channels,
    );
    operations.push(Object.freeze({
      kind: "snake",
      label: operationLabel,
      input: inputSlot,
      output: outputSlot,
      shape: Object.freeze({ ...pointwiseShape }),
      alpha,
      beta,
    }));
    recordOutput(
      outputSlot,
      checkedAceProduct(
        [pointwiseShape.batch, pointwiseShape.frames, pointwiseShape.channels],
        `${operationLabel} output`,
      ),
    );
  };

  const firstChannels =
    config.decoderChannels * config.channelMultiples.at(-1)!;
  let next = chooseWorkspace(new Set([active]));
  appendConv(
    "conv1",
    "decoder.conv1",
    active,
    next,
    {
      batch,
      inputFrames: frames,
      inputChannels: channels,
      outputChannels: firstChannels,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    },
    true,
  );
  active = next;
  channels = firstChannels;

  const upsamplingRatios = [...config.downsamplingRatios].reverse();
  const channelMultiples = [1, ...config.channelMultiples];
  for (let block = 0; block < upsamplingRatios.length; block += 1) {
    const stride = upsamplingRatios[block]!;
    const outputChannels =
      config.decoderChannels * channelMultiples[upsamplingRatios.length - block - 1]!;
    const blockPrefix = `decoder.block.${block}`;

    next = chooseWorkspace(new Set([active]));
    appendSnake(
      `block-${block}-snake1`,
      `${blockPrefix}.snake1`,
      active,
      next,
      { batch, frames, channels },
    );
    active = next;

    next = chooseWorkspace(new Set([active]));
    const transposeShape: AceVaeConvTranspose1dShape = {
      batch,
      inputFrames: frames,
      inputChannels: channels,
      outputChannels,
      kernelSize: 2 * stride,
      stride,
      dilation: 1,
      padding: Math.ceil(stride / 2),
      outputPadding: 0,
    };
    const transposePlan = planAceVaeConvTranspose1d(transposeShape);
    const transposeWeight = recordTensor(
      `vae.${blockPrefix}.conv_t1.weight`,
      transposePlan.weightElements,
    );
    const transposeBias = recordTensor(
      `vae.${blockPrefix}.conv_t1.bias`,
      outputChannels,
    );
    operations.push(Object.freeze({
      kind: "conv-transpose1d",
      label: `block-${block}-conv-t1`,
      input: active,
      output: next,
      shape: Object.freeze(transposeShape),
      weight: transposeWeight,
      bias: transposeBias,
    }));
    recordOutput(next, transposePlan.outputElements);
    active = next;
    frames = transposePlan.outputFrames;
    channels = outputChannels;

    for (const [residualIndex, dilation] of [1, 3, 9].entries()) {
      const residual = residualIndex + 1;
      const residualPrefix = `${blockPrefix}.res_unit${residual}`;
      const skip = active;
      const snake1Output = chooseWorkspace(new Set([skip]));
      appendSnake(
        `block-${block}-res-${residual}-snake1`,
        `${residualPrefix}.snake1`,
        skip,
        snake1Output,
        { batch, frames, channels },
      );

      const conv1Output = chooseWorkspace(new Set([skip, snake1Output]));
      appendConv(
        `block-${block}-res-${residual}-conv1`,
        `${residualPrefix}.conv1`,
        snake1Output,
        conv1Output,
        {
          batch,
          inputFrames: frames,
          inputChannels: channels,
          outputChannels: channels,
          kernelSize: 7,
          stride: 1,
          dilation,
          padding: 3 * dilation,
        },
        true,
      );

      appendSnake(
        `block-${block}-res-${residual}-snake2`,
        `${residualPrefix}.snake2`,
        conv1Output,
        snake1Output,
        { batch, frames, channels },
      );
      appendConv(
        `block-${block}-res-${residual}-conv2`,
        `${residualPrefix}.conv2`,
        snake1Output,
        conv1Output,
        {
          batch,
          inputFrames: frames,
          inputChannels: channels,
          outputChannels: channels,
          kernelSize: 1,
          stride: 1,
          dilation: 1,
          padding: 0,
        },
        true,
      );
      operations.push(Object.freeze({
        kind: "add",
        label: `block-${block}-res-${residual}-add`,
        input: skip,
        right: conv1Output,
        output: snake1Output,
        shape: Object.freeze({ batch, frames, channels }),
      }));
      recordOutput(
        snake1Output,
        checkedAceProduct(
          [batch, frames, channels],
          `block ${block} residual ${residual} add`,
        ),
      );
      active = snake1Output;
    }
  }

  next = chooseWorkspace(new Set([active]));
  appendSnake(
    "snake1",
    "decoder.snake1",
    active,
    next,
    { batch, frames, channels },
  );
  active = next;
  appendConv(
    "conv2",
    "decoder.conv2",
    active,
    "output",
    {
      batch,
      inputFrames: frames,
      inputChannels: channels,
      outputChannels: config.audioChannels,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    },
    false,
  );

  const hopLength = config.downsamplingRatios.reduce(
    (product, ratio) => product * ratio,
    1,
  );
  if (frames !== inputFrames * hopLength) {
    throw new Error(
      "ACE VAE decoder transposed-convolution geometry does not equal its hop length",
    );
  }
  const inputElements = checkedAceProduct(
    [batch, inputFrames, config.decoderInputChannels],
    "ACE VAE decoder input",
  );
  const outputElements = checkedAceProduct(
    [batch, frames, config.audioChannels],
    "ACE VAE decoder output",
  );
  const workspaceBytes = maximumActivationElements * FLOAT32_BYTES;
  return Object.freeze({
    config: freezeConfig(config),
    batch,
    inputFrames,
    outputFrames: frames,
    hopLength,
    inputElements,
    outputElements,
    maximumActivationElements,
    workspaceBytes,
    allWorkspaceBytes: workspaceBytes * WORKSPACE_COUNT,
    parameterElements,
    parameterBytes: parameterElements * FLOAT32_BYTES,
    primitiveCount: operations.length,
    operations: Object.freeze(operations),
    requiredTensorNames: Object.freeze([...requiredTensors]),
  });
}

function validateDecoderConfig(config: AceVaeDecoderConfig): void {
  if (typeof config.id !== "string" || config.id.length === 0) {
    throw new TypeError("ACE VAE decoder config id must be non-empty");
  }
  for (const [name, value] of Object.entries({
    decoderInputChannels: config.decoderInputChannels,
    decoderChannels: config.decoderChannels,
    audioChannels: config.audioChannels,
    sampleRateHz: config.sampleRateHz,
  })) {
    requirePositiveSafeInteger(value, `ACE VAE decoder ${name}`);
  }
  if (
    config.channelMultiples.length === 0 ||
    config.channelMultiples.length !== config.downsamplingRatios.length
  ) {
    throw new RangeError(
      "ACE VAE decoder requires one channel multiple per temporal ratio",
    );
  }
  for (const [index, multiple] of config.channelMultiples.entries()) {
    requirePositiveSafeInteger(multiple, `ACE VAE channel multiple ${index}`);
  }
  for (const [index, ratio] of config.downsamplingRatios.entries()) {
    requirePositiveSafeInteger(ratio, `ACE VAE temporal ratio ${index}`);
    if (ratio % 2 !== 0) {
      throw new RangeError(
        "ACE VAE Stage 1 decoder requires even ratios for exact length multiplication",
      );
    }
  }
}

function requireDecoderBindings(
  label: string,
  plan: AceVaeDecoderGraphPlan,
  bindings: AceVaeDecoderBindings,
): Readonly<Record<string, readonly AceVaeTensorPartBinding[]>> {
  requireAceBindingBytes(
    bindings.input,
    plan.inputElements * FLOAT32_BYTES,
    `${label} input`,
  );
  requireAceBindingBytes(
    bindings.output,
    plan.outputElements * FLOAT32_BYTES,
    `${label} output`,
  );
  for (const [index, workspace] of bindings.workspaces.entries()) {
    requireAceBindingBytes(
      workspace,
      plan.workspaceBytes,
      `${label} workspace ${index}`,
    );
  }
  const tensors = resolveAceVaeDecoderTensorBindings(
    plan,
    bindings.tensors,
    label,
  );
  const readOnly = [
    bindings.input,
    ...Object.values(tensors).flatMap((parts) =>
      parts.map((part) => part.binding)),
  ];
  requireAceDisjointOutput(
    bindings.output,
    [...readOnly, ...bindings.workspaces],
    `${label} final output`,
  );
  for (const [index, workspace] of bindings.workspaces.entries()) {
    requireAceDisjointOutput(
      workspace,
      [
        ...readOnly,
        bindings.output,
        ...bindings.workspaces.filter((_, other) => other !== index),
      ],
      `${label} workspace ${index}`,
    );
  }
  return tensors;
}

/** Resolve only canonical converter outputs and fail before GPU compilation. */
export function resolveAceVaeDecoderTensorBindings(
  plan: AceVaeDecoderGraphPlan,
  tensors: Readonly<Record<string, AceVaeLogicalTensorBinding>>,
  label = "ACE VAE decoder",
): Readonly<Record<string, readonly AceVaeTensorPartBinding[]>> {
  const transposeWeights = new Map(
    plan.operations
      .filter((operation): operation is AceVaeDecoderConvTransposeOperation =>
        operation.kind === "conv-transpose1d")
      .map((operation) => [operation.weight, operation.shape.outputChannels]),
  );
  const resolved: Record<string, readonly AceVaeTensorPartBinding[]> = {};
  for (const name of plan.requiredTensorNames) {
    const tensor = tensors[name];
    if (tensor === undefined) {
      throw new Error(`${label} is missing VAE tensor ${name}`);
    }
    const logicalRows = transposeWeights.get(name);
    const parts = Array.isArray(tensor)
      ? tensor
      : [{
          binding: tensor as GPUBufferBinding,
          partStart: 0,
          partEnd: logicalRows ?? 1,
        }];
    if (logicalRows === undefined) {
      if (
        parts.length !== 1 ||
        parts[0]!.partStart !== 0 ||
        !Number.isSafeInteger(parts[0]!.partEnd) ||
        parts[0]!.partEnd <= 0
      ) {
        throw new RangeError(
          `${label} tensor ${name} must be one complete physical part`,
        );
      }
      resolved[name] = Object.freeze([
        Object.freeze({ ...parts[0]! }),
      ]);
    } else {
      resolved[name] = validateTensorParts(name, parts, logicalRows, label);
    }
  }
  return Object.freeze(resolved);
}

/**
 * Group physical manifest records under the runtime's logical tensor names.
 * The callback is responsible for returning the authenticated uploaded slice
 * corresponding to each exact physical record name.
 */
export function createAceVaeLogicalTensorBindingsFromManifest(
  plan: AceVaeDecoderGraphPlan,
  manifest: AcePackageManifest,
  physicalBindings: readonly AceVaePhysicalTensorBinding[],
): Readonly<Record<string, readonly AceVaeTensorPartBinding[]>> {
  const byPhysicalName = new Map(
    physicalBindings.map(({ physicalName, binding }) => [physicalName, binding]),
  );
  if (byPhysicalName.size !== physicalBindings.length) {
    throw new Error("ACE VAE physical tensor bindings contain duplicate names");
  }
  const grouped: Record<string, AceVaeTensorPartBinding[]> = {};
  const required = new Set(plan.requiredTensorNames);
  for (const [physicalName, record] of Object.entries(manifest.tensors)) {
    if (!required.has(record.logicalTensor)) continue;
    const uploaded = byPhysicalName.get(physicalName);
    if (uploaded === undefined) {
      throw new Error(`ACE VAE physical tensor ${physicalName} was not uploaded`);
    }
    (grouped[record.logicalTensor] ??= []).push({
      binding: uploaded,
      partStart: record.partStart,
      partEnd: record.partEnd,
    });
    byPhysicalName.delete(physicalName);
  }
  if (byPhysicalName.size !== 0) {
    throw new Error(
      `ACE VAE physical tensor bindings include unexpected names: ${[
        ...byPhysicalName.keys(),
      ].sort().join(", ")}`,
    );
  }
  return resolveAceVaeDecoderTensorBindings(plan, grouped);
}

function validateTensorParts(
  name: string,
  parts: readonly AceVaeTensorPartBinding[],
  logicalRows: number,
  label: string,
): readonly AceVaeTensorPartBinding[] {
  if (parts.length === 0) {
    throw new Error(`${label} tensor ${name} has no parts`);
  }
  const sorted = [...parts].sort((left, right) => left.partStart - right.partStart);
  let cursor = 0;
  for (const part of sorted) {
    if (
      !Number.isSafeInteger(part.partStart) ||
      !Number.isSafeInteger(part.partEnd) ||
      part.partStart !== cursor ||
      part.partEnd <= part.partStart ||
      part.partEnd > logicalRows
    ) {
      throw new RangeError(
        `${label} tensor ${name} parts do not cover logical axis 0 contiguously`,
      );
    }
    cursor = part.partEnd;
  }
  if (cursor !== logicalRows) {
    throw new RangeError(
      `${label} tensor ${name} parts end at ${cursor}, expected ${logicalRows}`,
    );
  }
  return Object.freeze(sorted.map((part) => Object.freeze({ ...part })));
}

function requireSingleTensor(
  tensors: Readonly<Record<string, readonly AceVaeTensorPartBinding[]>>,
  name: string,
): GPUBufferBinding {
  const parts = tensors[name];
  if (parts === undefined || parts.length !== 1) {
    throw new Error(`ACE VAE tensor ${name} must have one physical part`);
  }
  return parts[0]!.binding;
}

function resolveSlot(
  bindings: AceVaeDecoderBindings,
  slot: AceVaeDecoderSlot,
): GPUBufferBinding {
  switch (slot) {
    case "input":
      return bindings.input;
    case "output":
      return bindings.output;
    case "workspace-0":
      return bindings.workspaces[0];
    case "workspace-1":
      return bindings.workspaces[1];
    case "workspace-2":
      return bindings.workspaces[2];
  }
}

function freezeConfig(config: AceVaeDecoderConfig): AceVaeDecoderConfig {
  return Object.freeze({
    ...config,
    channelMultiples: Object.freeze([...config.channelMultiples]),
    downsamplingRatios: Object.freeze([...config.downsamplingRatios]),
  });
}
