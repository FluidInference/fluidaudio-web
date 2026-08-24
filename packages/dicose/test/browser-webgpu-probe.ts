import type { GpuWeightTensor } from "../src/model/package.js";
import { packFloat16, unpackFloat16 } from "../src/runtime/audio.js";
import { GpuOps } from "../src/webgpu/ops.js";
import {
  createF16Tensor,
  destroyTensors,
  readF16Tensor,
  writeF16Tensor,
  type GpuTensor,
} from "../src/webgpu/tensor.js";

interface ProbeReport {
  readonly ok: boolean;
  readonly input?: readonly number[];
  readonly weight?: readonly number[];
  readonly copy?: readonly number[];
  readonly linear?: readonly number[];
  readonly packedLinearWords?: number;
  readonly packedLinearMismatches?: number;
  readonly packedResidualWords?: number;
  readonly packedResidualMismatches?: number;
  readonly packedN256Owner128K4Mismatches?: number;
  readonly packedDensePartials?: DensePartialProbeReport;
  readonly rmsNormWords?: number;
  readonly rmsNormMismatches?: number;
  readonly rmsNormMatrix?: readonly RmsNormMatrixReport[];
  readonly conv1x1Words?: number;
  readonly conv1x1Mismatches?: number;
  readonly conv3x3Words?: number;
  readonly conv3x3Mismatches?: number;
  readonly maskStyle?: readonly number[];
  readonly attentionWords?: number;
  readonly attentionMismatches?: number;
  readonly flashAttentionWords?: number;
  readonly flashAttentionMismatches?: number;
  readonly flashAttentionNrmse?: number;
  readonly flashAttentionMaxAbs?: number;
  readonly flashAttentionCosine?: number;
  readonly stridedAttentionWords?: number;
  readonly stridedAttentionMismatches?: number;
  readonly fusedKeyRotaryWords?: number;
  readonly fusedKeyRotaryMismatches?: number;
  readonly fusedKeyRotaryStridedMismatches?: number;
  readonly attentionSample?: readonly number[];
  readonly validationError?: string;
  readonly uncapturedErrors?: readonly string[];
  readonly adapter?: Readonly<{
    readonly features: readonly string[];
    readonly maxStorageBufferBindingSize: number;
  }>;
  readonly error?: string;
}

interface DifferenceMetrics {
  readonly nrmse: number;
  readonly maxAbs: number;
  readonly cosine: number;
  readonly finite: boolean;
}

interface DensePartialProbeReport {
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly words: number;
  readonly k2ChangedWords: number;
  readonly k2: DifferenceMetrics;
  readonly k4ChangedWords: number;
  readonly k4: DifferenceMetrics;
}

interface RmsNormMatrixReport {
  readonly columns: number;
  readonly mapped: boolean;
  readonly rows: readonly number[];
  readonly forcedWorkgroupWidthRows: readonly number[];
  readonly words: number;
  readonly mismatches: number;
  readonly failedRows: readonly number[];
}

interface RmsNormProbeCase {
  readonly rows: number;
  readonly columns: number;
  readonly mapped: boolean;
  readonly workgroupWidthLimit?: number;
  readonly input: GpuTensor;
  readonly gamma: GpuWeightTensor;
  readonly mapping?: GpuTensor;
  readonly reference: GpuTensor;
  readonly candidate: GpuTensor;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const DENSE_PARTIAL_MAGNITUDES = Array.from(unpackFloat16(Uint16Array.of(
  0x2411,
  0x28b5,
  0x2d53,
  0x31e7,
  0x356b,
  0x39ad,
)));

void run();

async function run(): Promise<void> {
  let device: GPUDevice | undefined;
  let ops: GpuOps | undefined;
  const tensors: GpuTensor[] = [];
  const uncapturedErrors: string[] = [];
  let validationScopeOpen = false;
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === undefined || adapter === null) throw new Error("WebGPU adapter is unavailable");
    for (const feature of ["shader-f16", "subgroups"] as const) {
      if (!adapter.features.has(feature)) throw new Error(`WebGPU adapter lacks ${feature}`);
    }
    device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "subgroups"],
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    ops = new GpuOps(device);

    const linearInput = keep(tensors, f16Tensor(device, "probe-linear-input", [1, 2, 3, 4]));
    const linearWeight = keepWeight(tensors, weight(device, "probe-linear-weight", [2, 3], [2, -1, 0.5, 1, 3, -2]));
    const linearBias = keepWeight(tensors, weight(device, "probe-linear-bias", [3], [0.25, -0.5, 1]));
    const linearOutput = keep(tensors, createF16Tensor(device, 6, "probe-linear-output"));
    const copyOutput = keep(tensors, createF16Tensor(device, 4, "probe-copy-output"));

    const packedRows = 7;
    const packedInner = 32;
    const packedColumns = 128;
    const packedInputValues = Array.from(
      { length: packedRows * packedInner },
      (_, index) => (((index * 13) % 31) - 15) / 16,
    );
    const packedWeightValues = Array.from(
      { length: packedInner * packedColumns },
      (_, index) => (((index * 7) % 29) - 14) / 64,
    );
    const packedBiasValues = Array.from(
      { length: packedColumns },
      (_, index) => ((index % 9) - 4) / 64,
    );
    const packedInput = keep(tensors, f16Tensor(device, "probe-packed-input", packedInputValues));
    const packedReferenceWeight = keepWeight(tensors, weight(
      device,
      "probe-packed-reference-weight",
      [packedInner, packedColumns],
      packedWeightValues,
    ));
    const packedWeight = keepWeight(tensors, weight(
      device,
      "probe-packed-weight",
      [packedInner, packedColumns],
      packTileLinear(packedWeightValues, packedInner, packedColumns, 128),
      "linear-tile-n128-k32",
    ));
    const packedBias = keepWeight(tensors, weight(
      device,
      "probe-packed-bias",
      [packedColumns],
      packedBiasValues,
      "row-major",
    ));
    const packedReferenceOutput = keep(tensors, createF16Tensor(
      device,
      packedRows * packedColumns,
      "probe-packed-reference-output",
    ));
    const packedOutput = keep(tensors, createF16Tensor(
      device,
      packedRows * packedColumns,
      "probe-packed-output",
    ));
    const packedResidual = keep(tensors, f16Tensor(
      device,
      "probe-packed-residual",
      Array.from(
        { length: packedRows * packedColumns },
        (_, index) => (((index * 19) % 41) - 20) / 32,
      ),
    ));
    const packedResidualReferenceOutput = keep(tensors, createF16Tensor(
      device,
      packedRows * packedColumns,
      "probe-packed-residual-reference-output",
    ));
    const packedResidualOutput = keep(tensors, createF16Tensor(
      device,
      packedRows * packedColumns,
      "probe-packed-residual-output",
    ));

    // Long-K owner128 probe with a partial row tile. Exact, K2, and K4 share
    // identical packed weights so only the bounded reduction arithmetic moves.
    const partialRows = 33;
    const partialInner = 384;
    const partialColumns = 128;
    const partialInputValues = Array.from(
      { length: partialRows * partialInner },
      (_, index) => densePartialValue(index, 0x3141_5926),
    );
    const partialWeightValues = Array.from(
      { length: partialInner * partialColumns },
      (_, index) => densePartialValue(index, 0x6a09_e667),
    );
    const partialBiasValues = Array.from(
      { length: partialColumns },
      (_, index) => ((index * 5) % 17 - 8) / 128,
    );
    const partialInput = keep(tensors, f16Tensor(
      device,
      "probe-packed-partial-input",
      partialInputValues,
    ));
    const partialWeight = keepWeight(tensors, weight(
      device,
      "probe-packed-partial-weight",
      [partialInner, partialColumns],
      packTileLinear(partialWeightValues, partialInner, partialColumns, 128),
      "linear-tile-n128-k32",
    ));
    const partialBias = keepWeight(tensors, weight(
      device,
      "probe-packed-partial-bias",
      [partialColumns],
      partialBiasValues,
      "row-major",
    ));
    const partialExactOutput = keep(tensors, createF16Tensor(
      device,
      partialRows * partialColumns,
      "probe-packed-partial-exact",
    ));
    const partialK2Output = keep(tensors, createF16Tensor(
      device,
      partialRows * partialColumns,
      "probe-packed-partial-k2",
    ));
    const partialK4Output = keep(tensors, createF16Tensor(
      device,
      partialRows * partialColumns,
      "probe-packed-partial-k4",
    ));

    const rmsNormCases: RmsNormProbeCase[] = [];
    const rmsNormRows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17] as const;
    const rmsNormColumns = [8, 16, 48, 96, 192, 384, 512, 516] as const;
    for (const columns of rmsNormColumns) {
      for (const rows of rmsNormRows) {
        rmsNormCases.push(createRmsNormProbeCase(device, tensors, rows, columns, false));
      }
    }
    // FiLM is only used by the 384-wide transformer norms. Keep its scale and
    // shift indexing under the same row-tail and 2D-dispatch coverage.
    for (const rows of rmsNormRows) {
      rmsNormCases.push(createRmsNormProbeCase(device, tensors, rows, 384, true));
    }

    const convRows = 37;
    const convInput = keep(tensors, f16Tensor(
      device,
      "probe-conv1x1-input",
      Array.from({ length: convRows * 128 }, (_, index) => (((index * 13) % 31) - 15) / 32),
    ));
    const convWeight = keepWeight(tensors, weight(
      device,
      "probe-conv1x1-weight",
      [128, 128, 1, 1],
      Array.from({ length: 128 * 128 }, (_, index) => (((index * 7) % 29) - 14) / 256),
      "conv-oihw",
    ));
    const convBias = keepWeight(tensors, weight(
      device,
      "probe-conv1x1-bias",
      [128],
      Array.from({ length: 128 }, (_, index) => ((index % 9) - 4) / 128),
      "row-major",
    ));
    const convGeneric = keep(tensors, createF16Tensor(
      device,
      convRows * 128,
      "probe-conv1x1-generic",
    ));
    const convSubgroup = keep(tensors, createF16Tensor(
      device,
      convRows * 128,
      "probe-conv1x1-subgroup",
    ));

    const conv3x3Height = 5;
    const conv3x3Width = 7;
    const conv3x3Rows = conv3x3Height * conv3x3Width;
    const conv3x3Input = keep(tensors, f16Tensor(
      device,
      "probe-conv3x3-input",
      Array.from({ length: conv3x3Rows * 4 }, (_, index) => (((index * 17) % 37) - 18) / 32),
    ));
    const conv3x3Weight = keepWeight(tensors, weight(
      device,
      "probe-conv3x3-weight",
      [128, 4, 3, 3],
      Array.from({ length: 128 * 4 * 3 * 3 }, (_, index) => (((index * 11) % 43) - 21) / 256),
      "conv-oihw",
    ));
    const conv3x3Bias = keepWeight(tensors, weight(
      device,
      "probe-conv3x3-bias",
      [128],
      Array.from({ length: 128 }, (_, index) => ((index % 13) - 6) / 128),
      "row-major",
    ));
    const conv3x3Generic = keep(tensors, createF16Tensor(
      device,
      conv3x3Rows * 128,
      "probe-conv3x3-generic",
    ));
    const conv3x3Subgroup = keep(tensors, createF16Tensor(
      device,
      conv3x3Rows * 128,
      "probe-conv3x3-subgroup",
    ));

    const maskInput = keep(tensors, f16Tensor(device, "probe-mask-input", [1, -2]));
    const maskWeight = keepWeight(tensors, weight(device, "probe-mask-weight", [2, 4], [1, 2, -1, 0.5, -0.5, 1, 2, -1]));
    const maskBias = keepWeight(tensors, weight(device, "probe-mask-bias", [4], [0, 0, 0, 0]));
    const maskWide = keep(tensors, createF16Tensor(device, 4, "probe-mask-wide"));
    const maskOutput = keep(tensors, createF16Tensor(device, 2, "probe-mask-output"));

    // A short, multi-sequence, non-multiple-of-32 attention shape exercises
    // every grouped stream and its tail. Run the old query8 schedule alongside
    // the exact Q64 schedule, then compare raw f16 output words exactly.
    const attentionSequences = 3;
    const attentionTokens = 37;
    const attentionQkvValues = Array.from(
      { length: attentionSequences * attentionTokens * 1_536 },
      (_, index) => (((index * 17) % 101) - 50) / 32,
    );
    const attentionGateValues = Array.from(
      { length: attentionSequences * attentionTokens * 8 },
      (_, index) => (((index * 11) % 23) - 11) / 8,
    );
    const attentionQkv = keep(tensors, f16Tensor(
      device,
      "probe-attention-qkv",
      attentionQkvValues,
    ));
    const attentionQuery8 = keep(tensors, createF16Tensor(
      device,
      attentionSequences * attentionTokens * 512,
      "probe-attention-query8",
    ));
    const attentionQ64 = keep(tensors, createF16Tensor(
      device,
      attentionSequences * attentionTokens * 512,
      "probe-attention-q64",
    ));
    const attentionFlash = keep(tensors, createF16Tensor(
      device,
      attentionSequences * attentionTokens * 512,
      "probe-attention-flash",
    ));
    const attentionStrided = keep(tensors, createF16Tensor(
      device,
      attentionSequences * attentionTokens * 512,
      "probe-attention-strided",
    ));
    const attentionGates = keep(tensors, f16Tensor(
      device,
      "probe-attention-gates",
      attentionGateValues,
    ));
    const attentionQkvStrided = keep(tensors, f16Tensor(
      device,
      "probe-attention-qkv-strided",
      sequenceMajorToTokenMajor(attentionQkvValues, attentionSequences, attentionTokens, 1_536),
    ));
    const attentionGatesStrided = keep(tensors, f16Tensor(
      device,
      "probe-attention-gates-strided",
      sequenceMajorToTokenMajor(attentionGateValues, attentionSequences, attentionTokens, 8),
    ));

    // Exercise the exact production boundary: packed QKV first rounds to f16,
    // then either attention or the fused producer rotates K from that rounded
    // pair. The strided arm also validates physical [token, sequence] rows.
    const rotaryInner = 32;
    const rotaryRows = attentionSequences * attentionTokens;
    const rotaryInputValues = Array.from(
      { length: rotaryRows * rotaryInner },
      (_, index) => (((index * 13) % 31) - 15) / 32,
    );
    const rotaryWeightValues = Array.from(
      { length: rotaryInner * 1_536 },
      (_, index) => (((index * 7) % 29) - 14) / 256,
    );
    const rotaryInput = keep(tensors, f16Tensor(
      device,
      "probe-rotary-input",
      rotaryInputValues,
    ));
    const rotaryInputStrided = keep(tensors, f16Tensor(
      device,
      "probe-rotary-input-strided",
      sequenceMajorToTokenMajor(rotaryInputValues, attentionSequences, attentionTokens, rotaryInner),
    ));
    const rotaryWeight = keepWeight(tensors, weight(
      device,
      "probe-rotary-weight",
      [rotaryInner, 1_536],
      packTileLinear(rotaryWeightValues, rotaryInner, 1_536, 256),
      "linear-tile-n256-k32",
    ));
    const rotaryQkvReference = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 1_536,
      "probe-rotary-qkv-reference",
    ));
    const rotaryQkvOwner128 = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 1_536,
      "probe-rotary-qkv-owner128",
    ));
    const rotaryQkvFused = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 1_536,
      "probe-rotary-qkv-fused",
    ));
    const rotaryQkvFusedStrided = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 1_536,
      "probe-rotary-qkv-fused-strided",
    ));
    const rotaryContextReference = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 512,
      "probe-rotary-context-reference",
    ));
    const rotaryContextFused = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 512,
      "probe-rotary-context-fused",
    ));
    const rotaryContextFusedStrided = keep(tensors, createF16Tensor(
      device,
      rotaryRows * 512,
      "probe-rotary-context-fused-strided",
    ));

    device.pushErrorScope("validation");
    validationScopeOpen = true;
    ops.beginGraph();
    const encoder = device.createCommandEncoder({ label: "dicose-raw-wgsl-probe" });
    const pass = encoder.beginComputePass({ label: "dicose-raw-wgsl-probe" });
    ops.copy(pass, linearInput, copyOutput, 4);
    ops.linear(pass, linearInput, linearWeight, linearBias, linearOutput, {
      rows: 2,
      inner: 2,
      columns: 3,
    });
    ops.linear(pass, maskInput, maskWeight, maskBias, maskWide, {
      rows: 1,
      inner: 2,
      columns: 4,
      activation: "tanh",
    });
    ops.linear(pass, packedInput, packedReferenceWeight, packedBias, packedReferenceOutput, {
      rows: packedRows,
      inner: packedInner,
      columns: packedColumns,
      activation: "gelu",
    });
    ops.linear(pass, packedInput, packedWeight, packedBias, packedOutput, {
      rows: packedRows,
      inner: packedInner,
      columns: packedColumns,
      activation: "gelu",
    });
    ops.linear(pass, packedInput, packedReferenceWeight, packedBias, packedResidualReferenceOutput, {
      rows: packedRows,
      inner: packedInner,
      columns: packedColumns,
    });
    ops.add(pass, packedResidual, packedResidualReferenceOutput, packedRows * packedColumns);
    ops.linear(pass, packedInput, packedWeight, packedBias, packedResidualOutput, {
      rows: packedRows,
      inner: packedInner,
      columns: packedColumns,
      residual: packedResidual,
    });
    for (const [accumulation, output] of [
      ["exact", partialExactOutput],
      ["k2", partialK2Output],
      ["k4", partialK4Output],
    ] as const) {
      ops.linear(pass, partialInput, partialWeight, partialBias, output, {
        rows: partialRows,
        inner: partialInner,
        columns: partialColumns,
        outputTileColumns: 128,
        vectorizeK: true,
        accumulation,
      });
    }
    for (const normCase of rmsNormCases) {
      ops.rmsNorm(
        pass,
        normCase.input,
        normCase.gamma,
        normCase.reference,
        normCase.rows,
        normCase.columns,
        normCase.mapping,
        "row1",
        normCase.workgroupWidthLimit,
      );
      ops.rmsNorm(
        pass,
        normCase.input,
        normCase.gamma,
        normCase.candidate,
        normCase.rows,
        normCase.columns,
        normCase.mapping,
        "auto",
        normCase.workgroupWidthLimit,
      );
    }
    ops.conv2d(
      pass,
      convInput,
      convWeight,
      convBias,
      convGeneric,
      1,
      convRows,
      128,
      128,
      1,
      "generic",
    );
    ops.conv2d(
      pass,
      convInput,
      convWeight,
      convBias,
      convSubgroup,
      1,
      convRows,
      128,
      128,
      1,
    );
    ops.conv2d(
      pass,
      conv3x3Input,
      conv3x3Weight,
      conv3x3Bias,
      conv3x3Generic,
      conv3x3Height,
      conv3x3Width,
      4,
      128,
      3,
      "generic",
    );
    ops.conv2d(
      pass,
      conv3x3Input,
      conv3x3Weight,
      conv3x3Bias,
      conv3x3Subgroup,
      conv3x3Height,
      conv3x3Width,
      4,
      128,
      3,
    );
    ops.gluInPlace(pass, maskWide, maskOutput, 1, 2);
    ops.attention(pass, attentionQkv, attentionQuery8, {
      sequences: attentionSequences,
      tokens: attentionTokens,
      kernel: "query8",
    });
    ops.attention(pass, attentionQkv, attentionQ64, {
      sequences: attentionSequences,
      tokens: attentionTokens,
      gates: attentionGates,
      kernel: "q64",
    });
    ops.attention(pass, attentionQkv, attentionFlash, {
      sequences: attentionSequences,
      tokens: attentionTokens,
      gates: attentionGates,
      kernel: "flash",
    });
    ops.attention(
      pass,
      attentionQkvStrided,
      attentionStrided,
      {
        sequences: attentionSequences,
        tokens: attentionTokens,
        gates: attentionGatesStrided,
        strided: true,
        kernel: "q64",
      },
    );
    ops.applyGates(pass, attentionQuery8, attentionGates, attentionSequences * attentionTokens);
    ops.linear(pass, rotaryInput, rotaryWeight, undefined, rotaryQkvReference, {
      rows: rotaryRows,
      inner: rotaryInner,
      columns: 1_536,
    });
    ops.linear(pass, rotaryInput, rotaryWeight, undefined, rotaryQkvOwner128, {
      rows: rotaryRows,
      inner: rotaryInner,
      columns: 1_536,
      outputTileColumns: 128,
      vectorizeK: true,
    });
    ops.linear(pass, rotaryInput, rotaryWeight, undefined, rotaryQkvFused, {
      rows: rotaryRows,
      inner: rotaryInner,
      columns: 1_536,
      outputTileColumns: 128,
      vectorizeK: true,
      rotaryKeys: { sequences: attentionSequences, tokens: attentionTokens },
    });
    ops.linear(pass, rotaryInputStrided, rotaryWeight, undefined, rotaryQkvFusedStrided, {
      rows: rotaryRows,
      inner: rotaryInner,
      columns: 1_536,
      outputTileColumns: 128,
      vectorizeK: true,
      rotaryKeys: { sequences: attentionSequences, tokens: attentionTokens, strided: true },
    });
    ops.attention(
      pass,
      rotaryQkvReference,
      rotaryContextReference,
      { sequences: attentionSequences, tokens: attentionTokens, kernel: "q64" },
    );
    ops.attention(
      pass,
      rotaryQkvFused,
      rotaryContextFused,
      { sequences: attentionSequences, tokens: attentionTokens, rotatedKeys: true, kernel: "q64" },
    );
    ops.attention(
      pass,
      rotaryQkvFusedStrided,
      rotaryContextFusedStrided,
      {
        sequences: attentionSequences,
        tokens: attentionTokens,
        strided: true,
        rotatedKeys: true,
        kernel: "q64",
      },
    );
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const validationError = await device.popErrorScope();
    validationScopeOpen = false;
    const input = Array.from(unpackFloat16(await readF16Tensor(device, linearInput)));
    const linearWeightReadback = Array.from(unpackFloat16(await readWeight(device, linearWeight)));
    const copy = Array.from(unpackFloat16(await readF16Tensor(device, copyOutput)));
    const linear = Array.from(unpackFloat16(await readF16Tensor(device, linearOutput)));
    const maskStyle = Array.from(unpackFloat16(await readF16Tensor(device, maskOutput)));
    const packedReferenceWords = await readF16Tensor(device, packedReferenceOutput);
    const packedWords = await readF16Tensor(device, packedOutput);
    const packedLinearMismatches = countWordMismatches(packedReferenceWords, packedWords);
    const packedResidualReferenceWords = await readF16Tensor(device, packedResidualReferenceOutput);
    const packedResidualWords = await readF16Tensor(device, packedResidualOutput);
    const packedResidualMismatches = countWordMismatches(
      packedResidualReferenceWords,
      packedResidualWords,
    );
    const partialExactWords = await readF16Tensor(device, partialExactOutput);
    const partialK2Words = await readF16Tensor(device, partialK2Output);
    const partialK4Words = await readF16Tensor(device, partialK4Output);
    const packedDensePartials: DensePartialProbeReport = {
      rows: partialRows,
      inner: partialInner,
      columns: partialColumns,
      words: partialExactWords.length,
      k2ChangedWords: countWordMismatches(partialExactWords, partialK2Words),
      k2: differenceMetrics(partialExactWords, partialK2Words),
      k4ChangedWords: countWordMismatches(partialExactWords, partialK4Words),
      k4: differenceMetrics(partialExactWords, partialK4Words),
    };
    const rmsNormMatrix = await readRmsNormMatrix(device, rmsNormCases);
    const rmsNormWords = rmsNormMatrix.reduce((sum, result) => sum + result.words, 0);
    const rmsNormMismatches = rmsNormMatrix.reduce((sum, result) => sum + result.mismatches, 0);
    const convGenericWords = await readF16Tensor(device, convGeneric);
    const convSubgroupWords = await readF16Tensor(device, convSubgroup);
    const conv1x1Mismatches = countWordMismatches(convGenericWords, convSubgroupWords);
    const conv3x3GenericWords = await readF16Tensor(device, conv3x3Generic);
    const conv3x3SubgroupWords = await readF16Tensor(device, conv3x3Subgroup);
    const conv3x3Mismatches = countWordMismatches(conv3x3GenericWords, conv3x3SubgroupWords);
    const attentionQuery8Words = await readF16Tensor(device, attentionQuery8);
    const attentionQ64Words = await readF16Tensor(device, attentionQ64);
    const attentionMismatches = countWordMismatches(attentionQuery8Words, attentionQ64Words);
    const attentionFlashWords = await readF16Tensor(device, attentionFlash);
    const flashAttentionMismatches = countWordMismatches(attentionQ64Words, attentionFlashWords);
    const flashAttentionDifference = differenceMetrics(attentionQ64Words, attentionFlashWords);
    const attentionStridedWords = await readF16Tensor(device, attentionStrided);
    const attentionStridedComparableWords = tokenMajorToSequenceMajor(
      attentionStridedWords,
      attentionSequences,
      attentionTokens,
      512,
    );
    const stridedAttentionMismatches = countWordMismatches(
      attentionQuery8Words,
      attentionStridedComparableWords,
    );
    const rotaryContextReferenceWords = await readF16Tensor(device, rotaryContextReference);
    const rotaryQkvReferenceWords = await readF16Tensor(device, rotaryQkvReference);
    const packedN256Owner128K4Mismatches = countWordMismatches(
      rotaryQkvReferenceWords,
      await readF16Tensor(device, rotaryQkvOwner128),
    );
    const rotaryContextFusedWords = await readF16Tensor(device, rotaryContextFused);
    const fusedKeyRotaryMismatches = countWordMismatches(
      rotaryContextReferenceWords,
      rotaryContextFusedWords,
    );
    const rotaryContextFusedStridedWords = await readF16Tensor(device, rotaryContextFusedStrided);
    const fusedKeyRotaryStridedMismatches = countWordMismatches(
      rotaryContextReferenceWords,
      tokenMajorToSequenceMajor(
        rotaryContextFusedStridedWords,
        attentionSequences,
        attentionTokens,
        512,
      ),
    );
    const attentionSample = Array.from(
      unpackFloat16(attentionQ64Words.slice(0, 12)),
    );
    const expectedMask = Math.tanh(2) / (1 + Math.exp(-Math.tanh(-5)));
    const checkFailures = [
      closeArrayFailure(input, [1, 2, 3, 4], 0.001, "input readback"),
      closeArrayFailure(linearWeightReadback, [2, -1, 0.5, 1, 3, -2], 0.001, "weight readback"),
      closeArrayFailure(copy, [1, 2, 3, 4], 0.001, "copy"),
      closeArrayFailure(linear, [4.25, 4.5, -2.5, 10.25, 8.5, -5.5], 0.001, "linear"),
      closeArrayFailure(maskStyle, [expectedMask, 0], 0.002, "linear+tanh+GLU"),
      packedLinearMismatches === 0 ? undefined : `packed linear raw f16 mismatch count ${packedLinearMismatches}`,
      packedResidualMismatches === 0 ? undefined : `packed residual raw f16 mismatch count ${packedResidualMismatches}`,
      packedN256Owner128K4Mismatches === 0 ? undefined : `packed N256/owner128/K4 raw f16 mismatch count ${packedN256Owner128K4Mismatches}`,
      densePartialFailure(packedDensePartials.k2, "packed K2"),
      densePartialFailure(packedDensePartials.k4, "packed K4"),
      rmsNormMismatches === 0
        ? undefined
        : `RMSNorm row1/rows8 raw f16 mismatch count ${rmsNormMismatches}`,
      conv1x1Mismatches === 0 ? undefined : `conv1x1 generic/subgroup raw f16 mismatch count ${conv1x1Mismatches}`,
      conv3x3Mismatches === 0 ? undefined : `conv3x3 generic/subgroup raw f16 mismatch count ${conv3x3Mismatches}`,
      attentionMismatches === 0 ? undefined : `attention query8/q64 raw f16 mismatch count ${attentionMismatches}`,
      flashAttentionDifference.nrmse <= 0.02
        ? undefined
        : `flash attention NRMSE ${flashAttentionDifference.nrmse} exceeds 0.02`,
      stridedAttentionMismatches === 0
        ? undefined
        : `attention query8/strided raw f16 mismatch count ${stridedAttentionMismatches}`,
      fusedKeyRotaryMismatches === 0
        ? undefined
        : `fused K rotary raw f16 mismatch count ${fusedKeyRotaryMismatches}`,
      fusedKeyRotaryStridedMismatches === 0
        ? undefined
        : `fused strided K rotary raw f16 mismatch count ${fusedKeyRotaryStridedMismatches}`,
    ].filter((failure): failure is string => failure !== undefined);
    if (validationError !== null) checkFailures.push(`validation: ${validationError.message}`);
    for (const error of uncapturedErrors) checkFailures.push(`uncaptured: ${error}`);

    publish({
      ok: checkFailures.length === 0,
      input,
      weight: linearWeightReadback,
      copy,
      linear,
      maskStyle,
      packedLinearWords: packedWords.length,
      packedLinearMismatches,
      packedResidualWords: packedResidualWords.length,
      packedResidualMismatches,
      packedN256Owner128K4Mismatches,
      packedDensePartials,
      rmsNormWords,
      rmsNormMismatches,
      rmsNormMatrix,
      conv1x1Words: convSubgroupWords.length,
      conv1x1Mismatches,
      conv3x3Words: conv3x3SubgroupWords.length,
      conv3x3Mismatches,
      attentionWords: attentionQ64Words.length,
      attentionMismatches,
      flashAttentionWords: attentionFlashWords.length,
      flashAttentionMismatches,
      flashAttentionNrmse: flashAttentionDifference.nrmse,
      flashAttentionMaxAbs: flashAttentionDifference.maxAbs,
      flashAttentionCosine: flashAttentionDifference.cosine,
      stridedAttentionWords: attentionStridedWords.length,
      stridedAttentionMismatches,
      fusedKeyRotaryWords: rotaryContextFusedWords.length,
      fusedKeyRotaryMismatches,
      fusedKeyRotaryStridedMismatches,
      attentionSample,
      ...(validationError === null ? {} : { validationError: validationError.message }),
      ...(uncapturedErrors.length === 0 ? {} : { uncapturedErrors }),
      ...(checkFailures.length === 0 ? {} : { error: checkFailures.join("; ") }),
      adapter: {
        features: [...adapter.features].sort(),
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  } catch (error) {
    publish({ ok: false, error: error instanceof Error ? error.message : String(error) });
  } finally {
    if (device !== undefined && validationScopeOpen) {
      await device.popErrorScope().catch(() => null);
    }
    ops?.destroy();
    destroyTensors(tensors);
    if (device !== undefined) {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      device.destroy();
    }
  }
}

function createRmsNormProbeCase(
  device: GPUDevice,
  tensors: GpuTensor[],
  rows: number,
  columns: number,
  mapped: boolean,
): RmsNormProbeCase {
  const tag = `${columns}x${rows}-${mapped ? "film" : "plain"}`;
  const input = keep(tensors, f16Tensor(
    device,
    `probe-rmsnorm-${tag}-input`,
    Array.from({ length: rows * columns }, (_, index) => rmsNormInputValue(index, rows, columns)),
  ));
  const gamma = keepWeight(tensors, weight(
    device,
    `probe-rmsnorm-${tag}-gamma`,
    [columns],
    Array.from({ length: columns }, (_, index) => 0.625 + ((index * 5 + rows) % 29) / 32),
    "row-major",
  ));
  const mapping = mapped
    ? keep(tensors, f16Tensor(
      device,
      `probe-rmsnorm-${tag}-mapping`,
      Array.from(
        { length: columns * 2 },
        (_, index) => (((index * 7 + rows * 3) % 31) - 15) / 256,
      ),
    ))
    : undefined;
  return {
    rows,
    columns,
    mapped,
    // Force both the row1 reference and rows8 candidate through a 2D grid:
    // row1 dispatches 2x9 and rows8 dispatches 2x2 for this case.
    ...(rows === 17 ? { workgroupWidthLimit: 2 } : {}),
    input,
    gamma,
    ...(mapping === undefined ? {} : { mapping }),
    reference: keep(tensors, createF16Tensor(
      device,
      rows * columns,
      `probe-rmsnorm-${tag}-row1`,
    )),
    candidate: keep(tensors, createF16Tensor(
      device,
      rows * columns,
      `probe-rmsnorm-${tag}-rows8`,
    )),
  };
}

function rmsNormInputValue(index: number, rows: number, columns: number): number {
  const mixed = index * 37 + rows * 19 + columns * 11;
  const exponent = [-8, -5, -2, 0, 2, 4][mixed % 6]!;
  const magnitude = (1 + ((mixed >>> 3) % 8) / 8) * 2 ** exponent;
  return (mixed & 1) === 0 ? magnitude : -magnitude;
}

async function readRmsNormMatrix(
  device: GPUDevice,
  cases: readonly RmsNormProbeCase[],
): Promise<readonly RmsNormMatrixReport[]> {
  const wordResults = await Promise.all(cases.map(async (normCase) => {
    const [reference, candidate] = await Promise.all([
      readF16Tensor(device, normCase.reference),
      readF16Tensor(device, normCase.candidate),
    ]);
    return {
      normCase,
      words: candidate.length,
      mismatches: countWordMismatches(reference, candidate),
    };
  }));
  const grouped = new Map<string, {
    columns: number;
    mapped: boolean;
    rows: number[];
    forcedWorkgroupWidthRows: number[];
    words: number;
    mismatches: number;
    failedRows: number[];
  }>();
  for (const result of wordResults) {
    const key = `${result.normCase.columns}:${result.normCase.mapped ? 1 : 0}`;
    let report = grouped.get(key);
    if (report === undefined) {
      report = {
        columns: result.normCase.columns,
        mapped: result.normCase.mapped,
        rows: [],
        forcedWorkgroupWidthRows: [],
        words: 0,
        mismatches: 0,
        failedRows: [],
      };
      grouped.set(key, report);
    }
    report.rows.push(result.normCase.rows);
    if (result.normCase.workgroupWidthLimit !== undefined) {
      report.forcedWorkgroupWidthRows.push(result.normCase.rows);
    }
    report.words += result.words;
    report.mismatches += result.mismatches;
    if (result.mismatches !== 0) report.failedRows.push(result.normCase.rows);
  }
  return [...grouped.values()];
}

function densePartialValue(index: number, salt: number): number {
  const mixed = mix32(salt ^ Math.imul(index + 1, 0x9e37_79b1));
  const magnitude = DENSE_PARTIAL_MAGNITUDES[mixed % DENSE_PARTIAL_MAGNITUDES.length]!;
  return (mixed & 0x8000_0000) === 0 ? magnitude : -magnitude;
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function f16Tensor(device: GPUDevice, label: string, values: readonly number[]): GpuTensor {
  const tensor = createF16Tensor(device, values.length, label);
  writeF16Tensor(device, tensor, packFloat16(Float32Array.from(values)));
  return tensor;
}

function weight(
  device: GPUDevice,
  name: string,
  shape: readonly number[],
  values: readonly number[],
  layout: GpuWeightTensor["layout"] = "linear-in-out",
): GpuWeightTensor {
  const tensor = f16Tensor(device, name, values);
  return {
    ...tensor,
    name,
    shape,
    offset: 0,
    byteLength: tensor.byteLength,
    dtype: "f16",
    layout,
  };
}

function packTileLinear(
  values: readonly number[],
  inner: number,
  columns: number,
  tileColumns: 128 | 256,
): number[] {
  if (values.length !== inner * columns || inner % 32 !== 0 || columns % tileColumns !== 0) {
    throw new RangeError("Invalid packed probe linear shape");
  }
  const packed: number[] = [];
  for (let columnTile = 0; columnTile < columns; columnTile += tileColumns) {
    for (let innerTile = 0; innerTile < inner; innerTile += 32) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const sourceBase = (innerTile + innerInTile) * columns + columnTile;
        for (let column = 0; column < tileColumns; column += 1) {
          packed.push(values[sourceBase + column]!);
        }
      }
    }
  }
  return packed;
}

function sequenceMajorToTokenMajor(
  values: readonly number[],
  sequences: number,
  tokens: number,
  width: number,
): number[] {
  if (values.length !== sequences * tokens * width) throw new RangeError("Invalid sequence-major tensor");
  const transposed = new Array<number>(values.length);
  for (let sequence = 0; sequence < sequences; sequence += 1) {
    for (let token = 0; token < tokens; token += 1) {
      const source = (sequence * tokens + token) * width;
      const destination = (token * sequences + sequence) * width;
      for (let column = 0; column < width; column += 1) {
        transposed[destination + column] = values[source + column]!;
      }
    }
  }
  return transposed;
}

function tokenMajorToSequenceMajor(
  values: Uint16Array,
  sequences: number,
  tokens: number,
  width: number,
): Uint16Array {
  if (values.length !== sequences * tokens * width) throw new RangeError("Invalid token-major tensor");
  const transposed = new Uint16Array(values.length);
  for (let token = 0; token < tokens; token += 1) {
    for (let sequence = 0; sequence < sequences; sequence += 1) {
      const source = (token * sequences + sequence) * width;
      const destination = (sequence * tokens + token) * width;
      transposed.set(values.subarray(source, source + width), destination);
    }
  }
  return transposed;
}

function keep<T extends GpuTensor>(tensors: GpuTensor[], tensor: T): T {
  tensors.push(tensor);
  return tensor;
}

function keepWeight(tensors: GpuTensor[], tensor: GpuWeightTensor): GpuWeightTensor {
  tensors.push({ buffer: tensor.buffer, byteLength: tensor.byteLength, label: tensor.name });
  return tensor;
}

async function readWeight(device: GPUDevice, weightTensor: GpuWeightTensor): Promise<Uint16Array> {
  return await readF16Tensor(device, {
    buffer: weightTensor.buffer,
    byteLength: weightTensor.byteLength,
    label: weightTensor.name,
  });
}

function closeArrayFailure(
  actual: readonly number[],
  expected: readonly number[],
  tolerance: number,
  label: string,
): string | undefined {
  if (actual.length !== expected.length) return `${label} length mismatch`;
  for (let index = 0; index < actual.length; index += 1) {
    const delta = Math.abs(actual[index]! - expected[index]!);
    if (!Number.isFinite(actual[index]!) || delta > tolerance) {
      return `${label}[${index}] expected ${expected[index]}, got ${actual[index]}`;
    }
  }
  return undefined;
}

function countWordMismatches(left: Uint16Array, right: Uint16Array): number {
  if (left.length !== right.length) return Math.max(left.length, right.length);
  let mismatches = 0;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) mismatches += 1;
  }
  return mismatches;
}

function differenceMetrics(
  referenceWords: Uint16Array,
  candidateWords: Uint16Array,
): DifferenceMetrics {
  if (referenceWords.length !== candidateWords.length) {
    return {
      nrmse: Number.POSITIVE_INFINITY,
      maxAbs: Number.POSITIVE_INFINITY,
      cosine: 0,
      finite: false,
    };
  }
  const reference = unpackFloat16(referenceWords);
  const candidate = unpackFloat16(candidateWords);
  let squaredError = 0;
  let squaredReference = 0;
  let squaredCandidate = 0;
  let dot = 0;
  let maxAbs = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index]!;
    const actual = candidate[index]!;
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
      return {
        nrmse: Number.POSITIVE_INFINITY,
        maxAbs: Number.POSITIVE_INFINITY,
        cosine: 0,
        finite: false,
      };
    }
    const difference = actual - expected;
    squaredError += difference * difference;
    squaredReference += expected * expected;
    squaredCandidate += actual * actual;
    dot += expected * actual;
    maxAbs = Math.max(maxAbs, Math.abs(difference));
  }
  return {
    nrmse: Math.sqrt(squaredError / Math.max(squaredReference, Number.MIN_VALUE)),
    maxAbs,
    cosine: dot / Math.sqrt(Math.max(squaredReference * squaredCandidate, Number.MIN_VALUE)),
    finite: true,
  };
}

function densePartialFailure(metrics: DifferenceMetrics, label: string): string | undefined {
  if (!metrics.finite) return `${label} produced a non-finite value`;
  if (metrics.nrmse > 0.001) return `${label} NRMSE ${metrics.nrmse} exceeds 0.001`;
  if (metrics.maxAbs > 0.05) return `${label} maxAbs ${metrics.maxAbs} exceeds 0.05`;
  if (metrics.cosine < 0.99999) return `${label} cosine ${metrics.cosine} is below 0.99999`;
  return undefined;
}

function publish(report: ProbeReport): void {
  if (statusNode !== null) statusNode.textContent = report.ok ? "WebGPU probe passed." : "WebGPU probe failed.";
  if (resultNode !== null) resultNode.textContent = JSON.stringify(report, null, 2);
}
