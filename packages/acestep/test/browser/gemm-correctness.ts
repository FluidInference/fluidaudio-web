import { AceCorrectnessGemmKernel } from "../../src/webgpu/kernels/gemm.js";
import { AceCorrectnessRmsNormKernel } from "../../src/webgpu/kernels/rmsnorm.js";
import { AceCorrectnessRopeKernel } from "../../src/webgpu/kernels/rope.js";
import { AceCorrectnessAttentionKernel } from "../../src/webgpu/kernels/attention.js";
import { ACE_DIRECT_DCW_CONFIGURATION } from "../../src/api.js";
import { AceCorrectnessDcwKernel } from "../../src/webgpu/kernels/dcw.js";

interface CaseResult {
  readonly operation: "gemm" | "rmsnorm" | "rope" | "attention" | "dcw";
  readonly profile: "reference-bf16" | "raw-fp16";
  readonly actual: readonly number[];
  readonly expected: readonly number[];
  readonly backend?: "fixed32-subgroup-query8";
  readonly caseId?: string;
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) =>
    finish("failed", error instanceof Error ? error.stack ?? error.message : String(error)),
);

async function run(): Promise<readonly CaseResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = [
    ...(adapter.features.has("shader-f16") ? ["shader-f16" as const] : []),
    ...(adapter.features.has("subgroups") ? ["subgroups" as const] : []),
  ];
  const device = await adapter.requestDevice({ requiredFeatures });
  const results: CaseResult[] = [];
  try {
    results.push(await runReferenceCase(device));
    if (device.features.has("shader-f16")) {
      results.push(await runFp16Case(device));
      results.push(await runFp16NoBiasRoundingCase(device));
    }
    results.push(await runReferenceRmsNormCase(device));
    if (device.features.has("shader-f16")) {
      results.push(await runFp16RmsNormCase(device));
    }
    results.push(await runReferenceRopeCase(device));
    if (device.features.has("shader-f16")) {
      results.push(await runFp16RopeCase(device));
    }
    results.push(await runReferenceAttentionCase(device));
    if (
      device.features.has("subgroups") &&
      adapter.info.subgroupMinSize === 32 &&
      adapter.info.subgroupMaxSize === 32
    ) {
      results.push(
        await runFixed32TiledAttentionCase(device, "square-full"),
        await runFixed32TiledAttentionCase(device, "cross-full"),
        await runFixed32TiledAttentionCase(device, "sliding"),
      );
    }
    if (device.features.has("shader-f16")) {
      results.push(await runFp16AttentionCase(device));
    }
    results.push(await runReferenceDcwCase(device));
    if (device.features.has("shader-f16")) {
      results.push(await runFp16DcwCase(device));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runReferenceCase(device: GPUDevice): Promise<CaseResult> {
  const activation = new Float32Array([
    1, 2, -1, 0.5, 3,
    -2, 1, 0, 4, -0.5,
    0.25, -1.5, 2, 1, 0,
  ]);
  const weight = [
    1, 0.5, -2, 1, 0,
    -1, 2, 0.25, 0, 3,
    0, -0.5, 1, 2, -1,
    4, 0, -1, 0.5, 0.25,
  ];
  const bias = [0.5, -1, 2, -0.25];
  const expected = cpuGemm(activation, weight, bias, 3, 5, 4);
  const activationBuffer = storageBuffer(device, activation);
  const weightBuffer = storageBuffer(device, packBf16(weight));
  const biasBuffer = storageBuffer(device, packBf16(bias));
  const outputBuffer = device.createBuffer({
    label: "ace-browser-bf16-gemm-output",
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-bf16-gemm",
      { rows: 3, inner: 5, columns: 4 },
      {
        activation: binding(activationBuffer),
        weight: binding(weightBuffer),
        output: binding(outputBuffer),
        bias: binding(biasBuffer),
      },
    );
    const actual = await executeAndReadF32(device, dispatch, outputBuffer, expected.length);
    assertClose(actual, expected, 1e-5);
    return {
      operation: "gemm",
      profile: "reference-bf16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    activationBuffer.destroy();
    weightBuffer.destroy();
    biasBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16Case(device: GPUDevice): Promise<CaseResult> {
  const activationValues = [1, 2, -1, 0.5, -2, 1, 0, 4];
  const weightValues = [1, 0.5, -2, 1, -1, 2, 0.25, 0, 0, -0.5, 1, 2];
  const biasValues = [0.5, -1, 2];
  const expected = cpuGemm(
    new Float32Array(activationValues),
    weightValues,
    biasValues,
    2,
    4,
    3,
  );
  const activationBuffer = storageBuffer(device, packFp16Exact(activationValues));
  const weightBuffer = storageBuffer(device, packFp16Exact(weightValues));
  const biasBuffer = storageBuffer(device, packFp16Exact(biasValues));
  const outputBuffer = device.createBuffer({
    label: "ace-browser-fp16-gemm-output",
    size: Math.max(4, expected.length * Uint16Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessGemmKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-gemm",
      { rows: 2, inner: 4, columns: 3 },
      {
        activation: binding(activationBuffer),
        weight: binding(weightBuffer),
        output: binding(outputBuffer),
        bias: binding(biasBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, expected.length);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 1e-3);
    return {
      operation: "gemm",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    activationBuffer.destroy();
    weightBuffer.destroy();
    biasBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16NoBiasRoundingCase(device: GPUDevice): Promise<CaseResult> {
  // This distinguishes a true source-order f16 accumulator from an FP32
  // oracle: 2048 + 1 is the halfway case and rounds back to even 2048 in f16,
  // after which adding -2048 produces zero. FP32 accumulation would produce 1.
  const activationValues = [2048, 1, -2048];
  const weightValues = [1, 1, 1];
  const expected = new Float32Array([0]);
  const activationBuffer = storageBuffer(device, packFp16Exact(activationValues));
  const weightBuffer = storageBuffer(device, packFp16Exact(weightValues));
  const outputBuffer = device.createBuffer({
    label: "ace-browser-fp16-gemm-no-bias-output",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessGemmKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-gemm-no-bias-rounding",
      { rows: 1, inner: 3, columns: 1 },
      {
        activation: binding(activationBuffer),
        weight: binding(weightBuffer),
        output: binding(outputBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, 1);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 0);
    return {
      operation: "gemm",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    activationBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runReferenceRmsNormCase(device: GPUDevice): Promise<CaseResult> {
  const input = new Float32Array([1, 2, -1, 0.5, 3, -2, 1, 0, 4, -0.5]);
  const weight = [1, 0.5, -2, 1, 0.25];
  const expected = cpuRmsNorm(input, weight, 2, 5, 1e-6);
  const inputBuffer = storageBuffer(device, input);
  const weightBuffer = storageBuffer(device, packBf16(weight));
  const outputBuffer = device.createBuffer({
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessRmsNormKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-bf16-rmsnorm",
      { rows: 2, width: 5, epsilon: 1e-6 },
      {
        input: binding(inputBuffer),
        weight: binding(weightBuffer),
        output: binding(outputBuffer),
      },
    );
    const actual = await executeAndReadF32(device, dispatch, outputBuffer, expected.length);
    assertClose(actual, expected, 2e-5);
    return {
      operation: "rmsnorm",
      profile: "reference-bf16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    inputBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16RmsNormCase(device: GPUDevice): Promise<CaseResult> {
  const inputValues = [1, 2, -1, 0.5, -2, 1, 0, 4];
  const weight = [1, 0.5, -2, 0.25];
  const expected = cpuRmsNorm(new Float32Array(inputValues), weight, 2, 4, 1e-6);
  const inputBuffer = storageBuffer(device, packFp16Exact(inputValues));
  const weightBuffer = storageBuffer(device, packFp16Exact(weight));
  const outputBuffer = device.createBuffer({
    size: expected.length * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessRmsNormKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-rmsnorm",
      { rows: 2, width: 4, epsilon: 1e-6 },
      {
        input: binding(inputBuffer),
        weight: binding(weightBuffer),
        output: binding(outputBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, expected.length);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 2e-3);
    return {
      operation: "rmsnorm",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    inputBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
  }
}

function cpuRmsNorm(
  input: Float32Array,
  weight: readonly number[],
  rows: number,
  width: number,
  epsilon: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    let squareSum = 0;
    for (let column = 0; column < width; column += 1) {
      const value = input[row * width + column]!;
      squareSum = Math.fround(squareSum + Math.fround(value * value));
    }
    const inverseRms = 1 / Math.sqrt(squareSum / width + epsilon);
    for (let column = 0; column < width; column += 1) {
      output[row * width + column] =
        input[row * width + column]! * inverseRms * weight[column]!;
    }
  }
  return output;
}

async function runReferenceRopeCase(device: GPUDevice): Promise<CaseResult> {
  const inputValues = repeatedRopeInput();
  const input = new Float32Array(inputValues);
  const { cosine, sine } = ropeTables(3, 4);
  const expected = cpuRope(input, cosine, sine, 2, 3, 4);
  const inputBuffer = storageBuffer(device, input);
  const cosineBuffer = storageBuffer(device, cosine);
  const sineBuffer = storageBuffer(device, sine);
  const outputBuffer = device.createBuffer({
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessRopeKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-reference-rope",
      { batch: 1, heads: 2, tokens: 3, headDimension: 4 },
      {
        input: binding(inputBuffer),
        cosine: binding(cosineBuffer),
        sine: binding(sineBuffer),
        output: binding(outputBuffer),
      },
    );
    const actual = await executeAndReadF32(device, dispatch, outputBuffer, expected.length);
    assertClose(actual, expected, 2e-6);
    return {
      operation: "rope",
      profile: "reference-bf16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    inputBuffer.destroy();
    cosineBuffer.destroy();
    sineBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16RopeCase(device: GPUDevice): Promise<CaseResult> {
  const inputValues = repeatedRopeInput();
  const input = new Float32Array(inputValues);
  const { cosine, sine } = ropeTables(3, 4);
  const expected = cpuRope(input, cosine, sine, 2, 3, 4);
  const inputBuffer = storageBuffer(device, packFp16Exact(inputValues));
  const cosineBuffer = storageBuffer(device, cosine);
  const sineBuffer = storageBuffer(device, sine);
  const outputBuffer = device.createBuffer({
    size: expected.length * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessRopeKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-rope",
      { batch: 1, heads: 2, tokens: 3, headDimension: 4 },
      {
        input: binding(inputBuffer),
        cosine: binding(cosineBuffer),
        sine: binding(sineBuffer),
        output: binding(outputBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, expected.length);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 2e-3);
    return {
      operation: "rope",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    inputBuffer.destroy();
    cosineBuffer.destroy();
    sineBuffer.destroy();
    outputBuffer.destroy();
  }
}

function repeatedRopeInput(): number[] {
  const pattern = [1, 2, -1, 0.5, -2, 1, 0, 4];
  return Array.from({ length: 24 }, (_, index) => pattern[index % pattern.length]!);
}

function ropeTables(
  tokens: number,
  headDimension: number,
): {
  cosine: Float32Array<ArrayBuffer>;
  sine: Float32Array<ArrayBuffer>;
} {
  const cosine = new Float32Array(tokens * headDimension);
  const sine = new Float32Array(tokens * headDimension);
  const half = headDimension / 2;
  for (let token = 0; token < tokens; token += 1) {
    for (let dimension = 0; dimension < headDimension; dimension += 1) {
      const frequencyIndex = dimension % half;
      const angle = token * (frequencyIndex + 1) * 0.25;
      cosine[token * headDimension + dimension] = Math.cos(angle);
      sine[token * headDimension + dimension] = Math.sin(angle);
    }
  }
  return { cosine, sine };
}

function cpuRope(
  input: Float32Array,
  cosine: Float32Array,
  sine: Float32Array,
  heads: number,
  tokens: number,
  headDimension: number,
): Float32Array {
  const output = new Float32Array(input.length);
  const half = headDimension / 2;
  for (let head = 0; head < heads; head += 1) {
    for (let token = 0; token < tokens; token += 1) {
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        const index = (head * tokens + token) * headDimension + dimension;
        const rotatedDimension = dimension < half ? dimension + half : dimension - half;
        const rotatedIndex = index - dimension + rotatedDimension;
        const sign = dimension < half ? -1 : 1;
        const tableIndex = token * headDimension + dimension;
        output[index] =
          input[index]! * cosine[tableIndex]! +
          sign * input[rotatedIndex]! * sine[tableIndex]!;
      }
    }
  }
  return output;
}

async function runReferenceAttentionCase(device: GPUDevice): Promise<CaseResult> {
  const input = attentionInputs();
  const expected = cpuAttention(input.query, input.key, input.value, {
    queryHeads: 2,
    keyValueHeads: 1,
    queryTokens: 4,
    keyValueTokens: 4,
    headDimension: 4,
    validQueryTokens: 3,
    validKeyTokens: 3,
    mode: "sliding",
    slidingRadius: 1,
  });
  const queryBuffer = storageBuffer(device, input.query);
  const keyBuffer = storageBuffer(device, input.key);
  const valueBuffer = storageBuffer(device, input.value);
  const lengthsBuffer = storageBuffer(device, new Uint32Array([3, 3]));
  const outputBuffer = device.createBuffer({
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessAttentionKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-reference-attention",
      {
        batch: 1,
        queryHeads: 2,
        keyValueHeads: 1,
        queryTokens: 4,
        keyValueTokens: 4,
        headDimension: 4,
        mode: "sliding",
        slidingRadius: 1,
      },
      {
        query: binding(queryBuffer),
        key: binding(keyBuffer),
        value: binding(valueBuffer),
        validLengths: binding(lengthsBuffer),
        output: binding(outputBuffer),
      },
    );
    const actual = await executeAndReadF32(device, dispatch, outputBuffer, expected.length);
    assertClose(actual, expected, 2e-5);
    return {
      operation: "attention",
      profile: "reference-bf16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    queryBuffer.destroy();
    keyBuffer.destroy();
    valueBuffer.destroy();
    lengthsBuffer.destroy();
    outputBuffer.destroy();
  }
}

type Fixed32AttentionCaseId = "square-full" | "cross-full" | "sliding";

async function runFixed32TiledAttentionCase(
  device: GPUDevice,
  caseId: Fixed32AttentionCaseId,
): Promise<CaseResult> {
  const queryHeads = 2;
  const keyValueHeads = 1;
  const queryTokens = caseId === "square-full" ? 9 : 2_250;
  const keyValueTokens = caseId === "cross-full" ? 97 : queryTokens;
  const headDimension = 128;
  const validQueryTokens = 7;
  const validKeyTokens = caseId === "cross-full" ? 4 : 8;
  const mode = caseId === "sliding" ? "sliding" as const : "full" as const;
  const slidingRadius = caseId === "sliding" ? 2 : 0;
  const query = Float32Array.from(
    { length: queryHeads * queryTokens * headDimension },
    (_, index) => Math.fround(Math.sin(index * 0.037) * 0.35),
  );
  const key = Float32Array.from(
    { length: keyValueHeads * keyValueTokens * headDimension },
    (_, index) => Math.fround(Math.cos(index * 0.029) * 0.4),
  );
  const value = Float32Array.from(
    { length: keyValueHeads * keyValueTokens * headDimension },
    (_, index) => Math.fround(Math.sin(index * 0.019 + 0.3) * 0.5),
  );
  const expected = cpuAttention(query, key, value, {
    queryHeads,
    keyValueHeads,
    queryTokens,
    keyValueTokens,
    headDimension,
    validQueryTokens,
    validKeyTokens,
    mode,
    slidingRadius,
  });
  const queryBuffer = storageBuffer(device, query);
  const keyBuffer = storageBuffer(device, key);
  const valueBuffer = storageBuffer(device, value);
  const lengthsBuffer = storageBuffer(
    device,
    new Uint32Array([validQueryTokens, validKeyTokens]),
  );
  const outputBuffer = device.createBuffer({
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessAttentionKernel.create(
    device,
    "reference-bf16",
    {
      backend: "fixed32-subgroup-query8",
      capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
    },
  );
  try {
    const dispatch = await kernel.createDispatch(
      `ace-browser-fixed32-query8-attention-${caseId}`,
      {
        batch: 1,
        queryHeads,
        keyValueHeads,
        queryTokens,
        keyValueTokens,
        headDimension,
        mode,
        ...(mode === "sliding" ? { slidingRadius } : {}),
      },
      {
        query: binding(queryBuffer),
        key: binding(keyBuffer),
        value: binding(valueBuffer),
        validLengths: binding(lengthsBuffer),
        output: binding(outputBuffer),
      },
    );
    if (dispatch.backend !== "fixed32-subgroup-query8") {
      throw new Error("fixed32 attention browser case selected the portable oracle");
    }
    const actual = await executeAndReadF32(
      device,
      dispatch,
      outputBuffer,
      expected.length,
    );
    assertClose(actual, expected, 2e-4);
    return {
      operation: "attention",
      profile: "reference-bf16",
      backend: "fixed32-subgroup-query8",
      caseId,
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    queryBuffer.destroy();
    keyBuffer.destroy();
    valueBuffer.destroy();
    lengthsBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16AttentionCase(device: GPUDevice): Promise<CaseResult> {
  const input = attentionInputs();
  const incrementalQuery = input.query.slice(0, 8);
  const expected = cpuAttention(incrementalQuery, input.key, input.value, {
    queryHeads: 2,
    keyValueHeads: 1,
    queryTokens: 1,
    keyValueTokens: 4,
    headDimension: 4,
    validQueryTokens: 1,
    validKeyTokens: 4,
    mode: "causal",
    slidingRadius: 0,
    queryPositionOffset: 3,
  });
  const queryBuffer = storageBuffer(device, packFp16Exact([...incrementalQuery]));
  const keyBuffer = storageBuffer(device, packFp16Exact([...input.key]));
  const valueBuffer = storageBuffer(device, packFp16Exact([...input.value]));
  const lengthsBuffer = storageBuffer(device, new Uint32Array([1, 4]));
  const outputBuffer = device.createBuffer({
    size: expected.length * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessAttentionKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-attention",
      {
        batch: 1,
        queryHeads: 2,
        keyValueHeads: 1,
        queryTokens: 1,
        keyValueTokens: 4,
        headDimension: 4,
        mode: "causal",
        queryPositionOffset: 3,
      },
      {
        query: binding(queryBuffer),
        key: binding(keyBuffer),
        value: binding(valueBuffer),
        validLengths: binding(lengthsBuffer),
        output: binding(outputBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, expected.length);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 2e-3);
    return {
      operation: "attention",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    queryBuffer.destroy();
    keyBuffer.destroy();
    valueBuffer.destroy();
    lengthsBuffer.destroy();
    outputBuffer.destroy();
  }
}

function attentionInputs(): {
  query: Float32Array<ArrayBuffer>;
  key: Float32Array<ArrayBuffer>;
  value: Float32Array<ArrayBuffer>;
} {
  const queryPattern = [1, 0.5, -1, 2, 0, 1, 0.5, -0.5];
  const keyPattern = [1, 0, -1, 0.5, 0.5, 1, 0, -1];
  const valuePattern = [1, 2, 0.5, -1, -0.5, 1, 2, 0];
  return {
    query: new Float32Array(
      Array.from({ length: 32 }, (_, index) => queryPattern[index % queryPattern.length]!),
    ),
    key: new Float32Array(
      Array.from({ length: 16 }, (_, index) => keyPattern[index % keyPattern.length]!),
    ),
    value: new Float32Array(
      Array.from({ length: 16 }, (_, index) => valuePattern[index % valuePattern.length]!),
    ),
  };
}

function cpuAttention(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  options: {
    queryHeads: number;
    keyValueHeads: number;
    queryTokens: number;
    keyValueTokens: number;
    headDimension: number;
    validQueryTokens: number;
    validKeyTokens: number;
    mode: "full" | "sliding" | "causal";
    slidingRadius: number;
    queryPositionOffset?: number;
  },
): Float32Array {
  const output = new Float32Array(query.length);
  const headsPerKeyValue = options.queryHeads / options.keyValueHeads;
  for (let head = 0; head < options.queryHeads; head += 1) {
    const keyValueHead = Math.floor(head / headsPerKeyValue);
    for (let queryToken = 0; queryToken < options.validQueryTokens; queryToken += 1) {
      const start = options.mode === "sliding"
        ? Math.max(0, queryToken - options.slidingRadius)
        : 0;
      const end = options.mode === "sliding"
        ? Math.min(options.validKeyTokens, queryToken + options.slidingRadius + 1)
        : options.mode === "causal"
          ? Math.min(
              options.validKeyTokens,
              (options.queryPositionOffset ?? 0) + queryToken + 1,
            )
          : options.validKeyTokens;
      const scores: number[] = [];
      for (let keyToken = start; keyToken < end; keyToken += 1) {
        let score = 0;
        for (let dimension = 0; dimension < options.headDimension; dimension += 1) {
          const queryIndex =
            (head * options.queryTokens + queryToken) * options.headDimension + dimension;
          const keyIndex =
            (keyValueHead * options.keyValueTokens + keyToken) *
              options.headDimension +
            dimension;
          score += query[queryIndex]! * key[keyIndex]!;
        }
        scores.push(score / Math.sqrt(options.headDimension));
      }
      const maximum = Math.max(...scores);
      const probabilities = scores.map((score) => Math.exp(score - maximum));
      const denominator = probabilities.reduce((sum, probability) => sum + probability, 0);
      for (let dimension = 0; dimension < options.headDimension; dimension += 1) {
        let weighted = 0;
        for (let offset = 0; offset < probabilities.length; offset += 1) {
          const keyToken = start + offset;
          const valueIndex =
            (keyValueHead * options.keyValueTokens + keyToken) *
              options.headDimension +
            dimension;
          weighted += probabilities[offset]! * value[valueIndex]!;
        }
        const outputIndex =
          (head * options.queryTokens + queryToken) * options.headDimension + dimension;
        output[outputIndex] = weighted / denominator;
      }
    }
  }
  return output;
}

async function runReferenceDcwCase(device: GPUDevice): Promise<CaseResult> {
  const stepped = new Float32Array([1, 2, -1, 0.5, 3, -2, 1, 0, 4, -0.5]);
  const clean = new Float32Array([0.5, 1, -2, 1, 2, -1, 0, 0.5, 1, -2]);
  const scales = { low: Math.fround(0.3 * 0.05), high: Math.fround(0.7 * 0.02) };
  const expected = cpuDcw(stepped, clean, 5, 2, scales.low, scales.high);
  const steppedBuffer = storageBuffer(device, stepped);
  const cleanBuffer = storageBuffer(device, clean);
  const outputBuffer = device.createBuffer({
    size: expected.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessDcwKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-reference-dcw",
      { batch: 1, time: 5, channels: 2 },
      ACE_DIRECT_DCW_CONFIGURATION,
      0.3,
      {
        steppedLatent: binding(steppedBuffer),
        predictedCleanLatent: binding(cleanBuffer),
        output: binding(outputBuffer),
      },
    );
    const actual = await executeAndReadF32(device, dispatch, outputBuffer, expected.length);
    assertClose(actual, expected, 2e-6);
    return {
      operation: "dcw",
      profile: "reference-bf16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    steppedBuffer.destroy();
    cleanBuffer.destroy();
    outputBuffer.destroy();
  }
}

async function runFp16DcwCase(device: GPUDevice): Promise<CaseResult> {
  const steppedValues = [1, 2, -1, 0.5, 4, -2, 1, 0, 4, -0.5];
  const cleanValues = [0.5, 1, -2, 1, 2, -1, 0, 0.5, 1, -2];
  const stepped = new Float32Array(steppedValues);
  const clean = new Float32Array(cleanValues);
  const scales = { low: Math.fround(0.3 * 0.05), high: Math.fround(0.7 * 0.02) };
  const expected = cpuDcw(stepped, clean, 5, 2, scales.low, scales.high);
  const steppedBuffer = storageBuffer(device, packFp16Exact(steppedValues));
  const cleanBuffer = storageBuffer(device, packFp16Exact(cleanValues));
  const outputBuffer = device.createBuffer({
    size: expected.length * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const kernel = AceCorrectnessDcwKernel.create(device, "raw-fp16");
  try {
    const dispatch = await kernel.createDispatch(
      "ace-browser-fp16-dcw",
      { batch: 1, time: 5, channels: 2 },
      ACE_DIRECT_DCW_CONFIGURATION,
      0.3,
      {
        steppedLatent: binding(steppedBuffer),
        predictedCleanLatent: binding(cleanBuffer),
        output: binding(outputBuffer),
      },
    );
    const words = await executeAndReadU16(device, dispatch, outputBuffer, expected.length);
    const actual = Float32Array.from(words, fp16BitsToNumber);
    assertClose(actual, expected, 2e-3);
    return {
      operation: "dcw",
      profile: "raw-fp16",
      actual: [...actual],
      expected: [...expected],
    };
  } finally {
    kernel.destroy();
    steppedBuffer.destroy();
    cleanBuffer.destroy();
    outputBuffer.destroy();
  }
}

function cpuDcw(
  stepped: Float32Array,
  clean: Float32Array,
  time: number,
  channels: number,
  lowScale: number,
  highScale: number,
): Float32Array {
  const output = new Float32Array(stepped.length);
  const inverseSqrtTwo = Math.fround(0.7071067690849304);
  for (let evenTime = 0; evenTime < time; evenTime += 2) {
    const oddTime = evenTime + 1;
    for (let channel = 0; channel < channels; channel += 1) {
      const evenIndex = evenTime * channels + channel;
      const oddIndex = oddTime * channels + channel;
      const xEven = stepped[evenIndex]!;
      const yEven = clean[evenIndex]!;
      const xOdd = oddTime < time ? stepped[oddIndex]! : 0;
      const yOdd = oddTime < time ? clean[oddIndex]! : 0;
      let xLow = Math.fround(Math.fround(xEven + xOdd) * inverseSqrtTwo);
      let xHigh = Math.fround(Math.fround(xEven - xOdd) * inverseSqrtTwo);
      const yLow = Math.fround(Math.fround(yEven + yOdd) * inverseSqrtTwo);
      const yHigh = Math.fround(Math.fround(yEven - yOdd) * inverseSqrtTwo);
      xLow = Math.fround(xLow + Math.fround(lowScale * Math.fround(xLow - yLow)));
      xHigh = Math.fround(
        xHigh + Math.fround(highScale * Math.fround(xHigh - yHigh)),
      );
      output[evenIndex] = Math.fround(Math.fround(xLow + xHigh) * inverseSqrtTwo);
      if (oddTime < time) {
        output[oddIndex] = Math.fround(Math.fround(xLow - xHigh) * inverseSqrtTwo);
      }
    }
  }
  return output;
}

function cpuGemm(
  activation: Float32Array,
  weight: readonly number[],
  bias: readonly number[],
  rows: number,
  inner: number,
  columns: number,
): Float32Array {
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let sum = 0;
      for (let index = 0; index < inner; index += 1) {
        sum = Math.fround(
          sum +
            Math.fround(activation[row * inner + index]!) *
              Math.fround(weight[column * inner + index]!),
        );
      }
      output[row * columns + column] = Math.fround(sum + bias[column]!);
    }
  }
  return output;
}

function packBf16(values: readonly number[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let index = 0; index < values.length; index += 1) {
    f32[0] = values[index]!;
    const bits = u32[0]!;
    const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
    const word = index >> 1;
    words[word] = words[word]! | (rounded << ((index & 1) * 16));
  }
  return words;
}

const EXACT_FP16_BITS = new Map<number, number>([
  [0, 0x0000],
  [0.25, 0x3400],
  [0.5, 0x3800],
  [1, 0x3c00],
  [2, 0x4000],
  [4, 0x4400],
  [2048, 0x6800],
  [-0.5, 0xb800],
  [-1, 0xbc00],
  [-2, 0xc000],
  [-2048, 0xe800],
]);

function packFp16Exact(values: readonly number[]): Uint16Array<ArrayBuffer> {
  return Uint16Array.from(values, (value) => {
    const bits = EXACT_FP16_BITS.get(value);
    if (bits === undefined) throw new Error(`No exact FP16 test encoding for ${value}`);
    return bits;
  });
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBuffer | ArrayBufferView<ArrayBufferLike>,
): GPUBuffer {
  const byteLength = "byteLength" in data ? data.byteLength : 0;
  const padded = new Uint8Array(Math.max(4, Math.ceil(byteLength / 4) * 4));
  const sourceBytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  padded.set(sourceBytes);
  const buffer = device.createBuffer({
    size: padded.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, padded);
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

async function executeAndReadF32(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
  output: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const bytes = count * Float32Array.BYTES_PER_ELEMENT;
  const raw = await executeAndRead(device, dispatch, output, bytes);
  return new Float32Array(raw);
}

async function executeAndReadU16(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
  output: GPUBuffer,
  count: number,
): Promise<Uint16Array> {
  const bytes = count * Uint16Array.BYTES_PER_ELEMENT;
  const raw = await executeAndRead(device, dispatch, output, bytes);
  return new Uint16Array(raw, 0, count);
}

async function executeAndRead(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
  output: GPUBuffer,
  bytes: number,
): Promise<ArrayBuffer> {
  const readback = device.createBuffer({
    size: Math.max(4, bytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, Math.max(4, bytes));
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    return readback.getMappedRange(0, Math.max(4, bytes)).slice(0, bytes);
  } finally {
    readback.destroy();
  }
}

function assertClose(
  actual: Float32Array,
  expected: Float32Array,
  tolerance: number,
): void {
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error(`Invalid vector tolerance ${tolerance}`);
  }
  if (actual.length !== expected.length) throw new Error("GPU output length mismatch");
  for (let index = 0; index < actual.length; index += 1) {
    const received = actual[index]!;
    const wanted = expected[index]!;
    if (!Number.isFinite(received) || !Number.isFinite(wanted)) {
      throw new Error(
        `Non-finite vector value at ${index}: received ${received}, expected ${wanted}`,
      );
    }
    if (Math.abs(received - wanted) > tolerance) {
      throw new Error(
        `GPU mismatch at ${index}: received ${received}, expected ${wanted}`,
      );
    }
  }
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = message;
  document.title = `ACE GEMM ${status}`;
}

function requireResultNode(): HTMLPreElement {
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result element");
  return node;
}
