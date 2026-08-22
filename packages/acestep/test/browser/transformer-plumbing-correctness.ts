import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import { AceCorrectnessEmbeddingKernel } from "../../src/webgpu/kernels/embedding.js";
import { AceCorrectnessTensorCopyKernel } from "../../src/webgpu/kernels/tensor-copy.js";
import { AceCorrectnessTransformerPlumbingKernel } from "../../src/webgpu/kernels/transformer-plumbing.js";

interface CaseResult {
  readonly operation: string;
  readonly profile: AceModelProfileId;
  readonly actual: readonly number[];
  readonly expected: readonly number[];
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) =>
    finish(
      "failed",
      error instanceof Error ? error.stack ?? error.message : String(error),
    ),
);

async function run(): Promise<readonly CaseResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  try {
    const results = await runProfile(device, "reference-bf16");
    if (device.features.has("shader-f16")) {
      results.push(...await runProfile(device, "raw-fp16"));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<CaseResult[]> {
  const embedding = AceCorrectnessEmbeddingKernel.create(device, profile);
  const transformer = AceCorrectnessTransformerPlumbingKernel.create(device, profile);
  const copy = AceCorrectnessTensorCopyKernel.create(device, profile);
  try {
    return [
      await runEmbedding(device, profile, embedding),
      ...await runHeadTransforms(device, profile, transformer),
      ...await runElementwise(device, profile, transformer),
      await runGather(device, profile, copy),
      ...await runAxisCopies(device, profile, copy),
      await runStablePack(device, profile, copy),
    ];
  } finally {
    embedding.destroy();
    transformer.destroy();
    copy.destroy();
  }
}

async function runEmbedding(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessEmbeddingKernel,
): Promise<CaseResult> {
  const tokenIds = new Uint32Array([0, 2, 5, 1]);
  const rows = [
    [1, 0.5, -1],
    [2, 0.25, 4],
    [-2, 1, 0.5],
    [0.25, -0.5, 2],
    [4, 2, 1],
    [-1, -2, 0.25],
  ] as const;
  const expected = Float32Array.from(
    [...tokenIds].flatMap((token) => rows[token]!),
    (value) => profileValue(profile, value),
  );
  const idsBuffer = storageBuffer(device, tokenIds);
  const firstWeight = weightBuffer(device, profile, rows.slice(0, 2).flat());
  const secondWeight = weightBuffer(device, profile, rows.slice(2).flat());
  const output = activationOutput(device, profile, expected.length);
  try {
    const dispatch = await kernel.createDispatch(
      `browser-${profile}-embedding`,
      { tokenCount: tokenIds.length, width: 3, vocabularySize: rows.length },
      {
        tokenIds: binding(idsBuffer),
        shards: [
          { firstRow: 0, rowCount: 2, weight: binding(firstWeight) },
          { firstRow: 2, rowCount: 4, weight: binding(secondWeight) },
        ],
        output: binding(output),
      },
    );
    const actual = await executeAndReadActivation(
      device,
      dispatch,
      output,
      expected.length,
      profile,
    );
    assertClose(actual, expected, 0);
    return result("embedding-shards", profile, actual, expected);
  } finally {
    idsBuffer.destroy();
    firstWeight.destroy();
    secondWeight.destroy();
    output.destroy();
  }
}

async function runHeadTransforms(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessTransformerPlumbingKernel,
): Promise<CaseResult[]> {
  const shape = { batch: 2, tokens: 3, heads: 2, headDimension: 2 };
  const inputValues = Array.from({ length: 24 }, (_, index) => index - 12);
  const input = activationBuffer(device, profile, inputValues);
  const splitOutput = activationOutput(device, profile, inputValues.length);
  const mergeOutput = activationOutput(device, profile, inputValues.length);
  try {
    const split = await kernel.createHeadTransformDispatch(
      `browser-${profile}-split-heads`,
      "split-heads",
      shape,
      { input: binding(input), output: binding(splitOutput) },
    );
    const merge = await kernel.createHeadTransformDispatch(
      `browser-${profile}-merge-heads`,
      "merge-heads",
      shape,
      { input: binding(splitOutput), output: binding(mergeOutput) },
    );
    const [actualSplit, actualMerge] = await executeAndReadActivations(
      device,
      [split, merge],
      [splitOutput, mergeOutput],
      [inputValues.length, inputValues.length],
      profile,
    );
    const expectedInput = Float32Array.from(
      inputValues,
      (value) => profileValue(profile, value),
    );
    const expectedSplit = cpuSplitHeads(expectedInput, shape);
    assertClose(actualSplit!, expectedSplit, 0);
    assertClose(actualMerge!, expectedInput, 0);
    return [
      result("split-heads", profile, actualSplit!, expectedSplit),
      result("merge-heads", profile, actualMerge!, expectedInput),
    ];
  } finally {
    input.destroy();
    splitOutput.destroy();
    mergeOutput.destroy();
  }
}

async function runElementwise(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessTransformerPlumbingKernel,
): Promise<CaseResult[]> {
  const shape = { batch: 2, tokens: 2, width: 2 };
  const left = [1, -2, 0.5, 2, -1, 0.25, 4, -0.5];
  const right = [0.5, 1, -1, 0.25, 2, -0.5, 0.5, 1];
  const broadcast = [0.5, 2, -1, 0.25];
  const roundedLeft = left.map((value) => profileValue(profile, value));
  const roundedRight = right.map((value) => profileValue(profile, value));
  const roundedBroadcast = broadcast.map((value) => profileValue(profile, value));
  const cases: CaseResult[] = [];

  cases.push(await runActivationCase(
    device,
    profile,
    "residual-add",
    [left, right],
    roundedLeft.map((value, index) => add(profile, value, roundedRight[index]!)),
    (buffers, output) => kernel.createResidualAddDispatch(
      `browser-${profile}-residual-add`,
      shape,
      { left: binding(buffers[0]!), right: binding(buffers[1]!), output: binding(output) },
    ),
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "broadcast-add",
    [left, broadcast],
    roundedLeft.map((value, index) =>
      add(profile, value, roundedBroadcast[broadcastIndex(index, shape)]!),
    ),
    (buffers, output) => kernel.createBroadcastDispatch(
      `browser-${profile}-broadcast-add`,
      "broadcast-add",
      shape,
      { input: binding(buffers[0]!), broadcast: binding(buffers[1]!), output: binding(output) },
    ),
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "broadcast-multiply",
    [left, broadcast],
    roundedLeft.map((value, index) =>
      multiply(profile, value, roundedBroadcast[broadcastIndex(index, shape)]!),
    ),
    (buffers, output) => kernel.createBroadcastDispatch(
      `browser-${profile}-broadcast-multiply`,
      "broadcast-multiply",
      shape,
      { input: binding(buffers[0]!), broadcast: binding(buffers[1]!), output: binding(output) },
    ),
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "silu",
    [left],
    roundedLeft.map((value) => silu(profile, value)),
    (buffers, output) => kernel.createSiluDispatch(
      `browser-${profile}-silu`,
      shape,
      { input: binding(buffers[0]!), output: binding(output) },
    ),
    profile === "raw-fp16" ? 0 : 2e-6,
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "swiglu",
    [left, right],
    roundedLeft.map((value, index) =>
      multiply(profile, silu(profile, value), roundedRight[index]!),
    ),
    (buffers, output) => kernel.createSwiGluDispatch(
      `browser-${profile}-swiglu`,
      shape,
      { gate: binding(buffers[0]!), up: binding(buffers[1]!), output: binding(output) },
    ),
    profile === "raw-fp16" ? 0 : 4e-6,
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "adaln",
    [left, broadcast, right.slice(0, 4)],
    roundedLeft.map((value, index) => {
      const position = broadcastIndex(index, shape);
      const scaled = multiply(
        profile,
        value,
        add(profile, 1, roundedBroadcast[position]!),
      );
      return add(profile, scaled, profileValue(profile, right[position]!));
    }),
    (buffers, output) => kernel.createAdaLnDispatch(
      `browser-${profile}-adaln`,
      shape,
      {
        normalized: binding(buffers[0]!),
        scale: binding(buffers[1]!),
        shift: binding(buffers[2]!),
        output: binding(output),
      },
    ),
  ));
  cases.push(await runActivationCase(
    device,
    profile,
    "gated-residual",
    [left, right, broadcast],
    roundedLeft.map((value, index) => {
      const gated = multiply(
        profile,
        roundedRight[index]!,
        roundedBroadcast[broadcastIndex(index, shape)]!,
      );
      return add(profile, value, gated);
    }),
    (buffers, output) => kernel.createGatedResidualDispatch(
      `browser-${profile}-gated-residual`,
      shape,
      {
        residual: binding(buffers[0]!),
        branch: binding(buffers[1]!),
        gate: binding(buffers[2]!),
        output: binding(output),
      },
    ),
  ));
  return cases;
}

async function runGather(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessTensorCopyKernel,
): Promise<CaseResult> {
  const source = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6];
  const indices = new Uint32Array([2, 0, 1, 1]);
  const expected = [5, 6, 1, 2, -3, -4, -3, -4].map((value) =>
    profileValue(profile, value)
  );
  const input = activationBuffer(device, profile, source);
  const indexBuffer = storageBuffer(device, indices);
  const output = activationOutput(device, profile, expected.length);
  try {
    const dispatch = await kernel.createGatherRowsDispatch(
      `browser-${profile}-gather`,
      { outer: 2, sourceRows: 3, outputRows: 2, width: 2 },
      { input: binding(input), indices: binding(indexBuffer), output: binding(output) },
    );
    const actual = await executeAndReadActivation(
      device,
      dispatch,
      output,
      expected.length,
      profile,
    );
    const expectedArray = Float32Array.from(expected);
    assertClose(actual, expectedArray, 0);
    return result("gather-rows", profile, actual, expectedArray);
  } finally {
    input.destroy();
    indexBuffer.destroy();
    output.destroy();
  }
}

async function runAxisCopies(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessTensorCopyKernel,
): Promise<CaseResult[]> {
  const cases: CaseResult[] = [];
  cases.push(await runCopyCase(
    device,
    profile,
    "right-pad",
    [[1, 2, 3, 4]],
    [1, 2, 3, 4, 0, 0],
    (buffers, output) => kernel.createRightPadDispatch(
      `browser-${profile}-right-pad`,
      { outer: 1, inputLength: 2, outputLength: 3, inner: 2 },
      { input: binding(buffers[0]!), output: binding(output) },
    ),
  ));
  cases.push(await runCopyCase(
    device,
    profile,
    "crop",
    [[1, 2, 3, 4, 5, 6, 7, 8]],
    [3, 4, 5, 6],
    (buffers, output) => kernel.createCropDispatch(
      `browser-${profile}-crop`,
      { outer: 1, inputLength: 4, offset: 1, outputLength: 2, inner: 2 },
      { input: binding(buffers[0]!), output: binding(output) },
    ),
  ));
  cases.push(await runCopyCase(
    device,
    profile,
    "repeat",
    [[1, 2, 3, 4]],
    [1, 2, 3, 4, 1, 2, 3, 4],
    (buffers, output) => kernel.createRepeatDispatch(
      `browser-${profile}-repeat`,
      { outer: 1, inputLength: 2, repeats: 2, inner: 2 },
      { input: binding(buffers[0]!), output: binding(output) },
    ),
  ));
  cases.push(await runCopyCase(
    device,
    profile,
    "concat",
    [[1, 2, 3, 4], [5, 6]],
    [1, 2, 3, 4, 5, 6],
    (buffers, output) => kernel.createConcatDispatch(
      `browser-${profile}-concat`,
      { outer: 1, leftLength: 2, rightLength: 1, inner: 2 },
      { left: binding(buffers[0]!), right: binding(buffers[1]!), output: binding(output) },
    ),
  ));
  return cases;
}

async function runStablePack(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessTensorCopyKernel,
): Promise<CaseResult> {
  const leftValues = [10, 11, 20, 21, 30, 31, 40, 41];
  const rightValues = [50, 51, 60, 61, 70, 71, 80, 81];
  const leftMask = new Uint32Array([0, 1, 1, 0]);
  const rightMask = new Uint32Array([1, 0, 0, 1]);
  const expected = [
    20, 21, 50, 51, 10, 11, 60, 61,
    30, 31, 80, 81, 40, 41, 70, 71,
  ].map((value) => profileValue(profile, value));
  const expectedMask = new Uint32Array([1, 1, 0, 0, 1, 1, 0, 0]);
  const left = activationBuffer(device, profile, leftValues);
  const right = activationBuffer(device, profile, rightValues);
  const leftMaskBuffer = storageBuffer(device, leftMask);
  const rightMaskBuffer = storageBuffer(device, rightMask);
  const scratch = emptyStorageBuffer(device, expectedMask.byteLength);
  const output = activationOutput(device, profile, expected.length);
  const outputMask = emptyStorageBuffer(
    device,
    expectedMask.byteLength,
    GPUBufferUsage.COPY_SRC,
  );
  try {
    const dispatch = await kernel.createStablePackDispatch(
      `browser-${profile}-stable-pack`,
      { batch: 2, leftLength: 2, rightLength: 2, width: 2 },
      {
        left: binding(left),
        right: binding(right),
        leftMask: binding(leftMaskBuffer),
        rightMask: binding(rightMaskBuffer),
        indicesScratch: binding(scratch),
        output: binding(output),
        outputMask: binding(outputMask),
      },
    );
    const [actualBytes, maskBytes] = await executeAndReadMany(
      device,
      [dispatch],
      [
        { buffer: output, bytes: activationByteLength(profile, expected.length) },
        { buffer: outputMask, bytes: expectedMask.byteLength },
      ],
    );
    const actual = decodeActivation(actualBytes!, profile, expected.length);
    const actualMask = new Uint32Array(maskBytes!);
    const expectedArray = Float32Array.from(expected);
    assertClose(actual, expectedArray, 0);
    assertU32Equal(actualMask, expectedMask);
    return result("stable-pack", profile, actual, expectedArray);
  } finally {
    left.destroy();
    right.destroy();
    leftMaskBuffer.destroy();
    rightMaskBuffer.destroy();
    scratch.destroy();
    output.destroy();
    outputMask.destroy();
  }
}

async function runActivationCase(
  device: GPUDevice,
  profile: AceModelProfileId,
  operation: string,
  inputValues: readonly (readonly number[])[],
  expectedValues: readonly number[],
  createDispatch: (
    buffers: readonly GPUBuffer[],
    output: GPUBuffer,
  ) => Promise<{ encode(pass: GPUComputePassEncoder): void }>,
  tolerance = 0,
): Promise<CaseResult> {
  const buffers = inputValues.map((values) => activationBuffer(device, profile, values));
  const output = activationOutput(device, profile, expectedValues.length);
  try {
    const dispatch = await createDispatch(buffers, output);
    const actual = await executeAndReadActivation(
      device,
      dispatch,
      output,
      expectedValues.length,
      profile,
    );
    const expected = Float32Array.from(expectedValues);
    assertClose(actual, expected, tolerance);
    return result(operation, profile, actual, expected);
  } finally {
    for (const buffer of buffers) buffer.destroy();
    output.destroy();
  }
}

async function runCopyCase(
  device: GPUDevice,
  profile: AceModelProfileId,
  operation: string,
  inputs: readonly (readonly number[])[],
  expectedValues: readonly number[],
  createDispatch: (
    buffers: readonly GPUBuffer[],
    output: GPUBuffer,
  ) => Promise<{ encode(pass: GPUComputePassEncoder): void }>,
): Promise<CaseResult> {
  return runActivationCase(
    device,
    profile,
    operation,
    inputs,
    expectedValues.map((value) => profileValue(profile, value)),
    createDispatch,
  );
}

function cpuSplitHeads(
  input: Float32Array,
  shape: { readonly batch: number; readonly tokens: number; readonly heads: number; readonly headDimension: number },
): Float32Array {
  const output = new Float32Array(input.length);
  for (let batch = 0; batch < shape.batch; batch += 1) {
    for (let head = 0; head < shape.heads; head += 1) {
      for (let token = 0; token < shape.tokens; token += 1) {
        for (let dimension = 0; dimension < shape.headDimension; dimension += 1) {
          const source = ((batch * shape.tokens + token) * shape.heads + head) *
            shape.headDimension + dimension;
          const destination = ((batch * shape.heads + head) * shape.tokens + token) *
            shape.headDimension + dimension;
          output[destination] = input[source]!;
        }
      }
    }
  }
  return output;
}

function broadcastIndex(
  index: number,
  shape: { readonly tokens: number; readonly width: number },
): number {
  const feature = index % shape.width;
  const batch = Math.floor(index / (shape.tokens * shape.width));
  return batch * shape.width + feature;
}

function profileValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16" ? roundFp16(value) : Math.fround(value);
}

function add(profile: AceModelProfileId, left: number, right: number): number {
  return profileValue(profile, left + right);
}

function multiply(profile: AceModelProfileId, left: number, right: number): number {
  return profileValue(profile, left * right);
}

function silu(profile: AceModelProfileId, value: number): number {
  return profileValue(profile, value / (1 + Math.exp(-value)));
}

function activationBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "raw-fp16"
      ? Uint16Array.from(values, numberToFp16Bits)
      : Float32Array.from(values),
  );
}

function weightBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "raw-fp16"
      ? Uint16Array.from(values, numberToFp16Bits)
      : packBf16(values),
  );
}

function activationOutput(
  device: GPUDevice,
  profile: AceModelProfileId,
  count: number,
): GPUBuffer {
  return emptyStorageBuffer(
    device,
    activationByteLength(profile, count),
    GPUBufferUsage.COPY_SRC,
  );
}

function emptyStorageBuffer(
  device: GPUDevice,
  byteLength: number,
  extraUsage = 0,
): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, Math.ceil(byteLength / 4) * 4),
    usage: GPUBufferUsage.STORAGE | extraUsage,
  });
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBufferView<ArrayBufferLike>,
): GPUBuffer {
  const padded = new Uint8Array(Math.max(4, Math.ceil(data.byteLength / 4) * 4));
  padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
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

async function executeAndReadActivation(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
  output: GPUBuffer,
  count: number,
  profile: AceModelProfileId,
): Promise<Float32Array> {
  const [bytes] = await executeAndReadMany(
    device,
    [dispatch],
    [{ buffer: output, bytes: activationByteLength(profile, count) }],
  );
  return decodeActivation(bytes!, profile, count);
}

async function executeAndReadActivations(
  device: GPUDevice,
  dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[],
  outputs: readonly GPUBuffer[],
  counts: readonly number[],
  profile: AceModelProfileId,
): Promise<Float32Array[]> {
  const raw = await executeAndReadMany(
    device,
    dispatches,
    outputs.map((buffer, index) => ({
      buffer,
      bytes: activationByteLength(profile, counts[index]!),
    })),
  );
  return raw.map((bytes, index) => decodeActivation(bytes, profile, counts[index]!));
}

async function executeAndReadMany(
  device: GPUDevice,
  dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[],
  outputs: readonly { readonly buffer: GPUBuffer; readonly bytes: number }[],
): Promise<ArrayBuffer[]> {
  const readbacks = outputs.map(({ bytes }) => device.createBuffer({
    size: Math.max(4, Math.ceil(bytes / 4) * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  }));
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches) dispatch.encode(pass);
    pass.end();
    for (let index = 0; index < outputs.length; index += 1) {
      const output = outputs[index]!;
      encoder.copyBufferToBuffer(
        output.buffer,
        0,
        readbacks[index]!,
        0,
        Math.max(4, Math.ceil(output.bytes / 4) * 4),
      );
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const result: ArrayBuffer[] = [];
    for (let index = 0; index < readbacks.length; index += 1) {
      const readback = readbacks[index]!;
      const bytes = outputs[index]!.bytes;
      await readback.mapAsync(GPUMapMode.READ);
      result.push(readback.getMappedRange(0, readback.size).slice(0, bytes));
    }
    return result;
  } finally {
    for (const readback of readbacks) readback.destroy();
  }
}

function activationByteLength(profile: AceModelProfileId, count: number): number {
  return count * (profile === "raw-fp16" ? 2 : 4);
}

function decodeActivation(
  bytes: ArrayBuffer,
  profile: AceModelProfileId,
  count: number,
): Float32Array {
  if (profile === "reference-bf16") return new Float32Array(bytes, 0, count);
  return Float32Array.from(new Uint16Array(bytes, 0, count), fp16BitsToNumber);
}

function packBf16(values: readonly number[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let index = 0; index < values.length; index += 1) {
    f32[0] = values[index]!;
    const bits = u32[0]!;
    const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
    words[index >> 1] = words[index >> 1]! | (rounded << ((index & 1) * 16));
  }
  return words;
}

function roundFp16(value: number): number {
  return fp16BitsToNumber(numberToFp16Bits(value));
}

function numberToFp16Bits(value: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x800000;
    const shift = 14 - halfExponent;
    const truncated = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    const rounded = truncated + (remainder > halfway ||
      (remainder === halfway && (truncated & 1) !== 0) ? 1 : 0);
    return sign | rounded;
  }
  let roundedMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (roundedMantissa & 1) !== 0)) {
    roundedMantissa += 1;
  }
  if (roundedMantissa === 0x400) {
    const nextExponent = halfExponent + 1;
    return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
  }
  return sign | (halfExponent << 10) | roundedMantissa;
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
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
      throw new Error(`Non-finite output at ${index}: ${received}, expected ${wanted}`);
    }
    if (Math.abs(received - wanted) > tolerance) {
      throw new Error(
        `GPU output mismatch at ${index}: ${received}, expected ${wanted}, tolerance ${tolerance}`,
      );
    }
  }
}

function assertU32Equal(actual: Uint32Array, expected: Uint32Array): void {
  if (actual.length !== expected.length) throw new Error("GPU u32 output length mismatch");
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `GPU u32 output mismatch at ${index}: ${actual[index]}, expected ${expected[index]}`,
      );
    }
  }
}

function result(
  operation: string,
  profile: AceModelProfileId,
  actual: Float32Array,
  expected: Float32Array,
): CaseResult {
  return { operation, profile, actual: [...actual], expected: [...expected] };
}

function requireResultNode(): HTMLElement {
  const node = document.querySelector<HTMLElement>("#result");
  if (node === null) throw new Error("Missing result node");
  return node;
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = message;
}
