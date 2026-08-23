import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  AceCorrectnessGemmKernel,
  aceScalarGemmOracleWgsl,
  planAceGemm,
  type AceGemmBufferBindings,
  type AceGemmDispatch,
  type AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

type Profile = Extract<AceModelProfileId, "reference-bf16" | "raw-fp16">;

interface DifferentialResult {
  readonly test: "scalar-gpu-differential";
  readonly profile: Profile;
  readonly shape: AceGemmShape;
  readonly bias: boolean;
  readonly comparedBits: number;
}

interface MultiRangeResult {
  readonly test: "multi-range-routing";
  readonly shape: AceGemmShape;
  readonly rangeCount: number;
  readonly comparedValues: number;
  readonly queueEmptyIdleMilliseconds: readonly number[];
}

interface HeartbeatResult {
  readonly animationFrames: number;
  readonly timerTicks: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface ExactShapeResult {
  readonly test: "exact-shape-cooperative-timing";
  readonly classification: "stage-1-enabling-diagnostic-not-a-benchmark";
  readonly profile: Profile;
  readonly shape: AceGemmShape;
  readonly workgroupCount: number;
  readonly logicalOutputCount: number;
  readonly logicalMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly rangeCount: number;
  readonly totalGpuPhaseMilliseconds: number;
  readonly medianRangeMilliseconds: number;
  readonly maximumRangeMilliseconds: number;
  readonly minimumQueueEmptyIdleMilliseconds: number;
  readonly heartbeat: HeartbeatResult;
  readonly checkedSamples: number;
  readonly maximumAbsoluteSampleError: number;
  readonly outputChecksum: OutputChecksum;
  readonly boundarySentinels: readonly OutputSentinel[];
}

interface OutputChecksum {
  readonly nonzeroCount: number;
  readonly finiteCount: number;
  readonly sum: number;
  readonly absoluteSum: number;
  readonly weightedSum: number;
}

interface OutputSentinel {
  readonly index: number;
  readonly actual: number;
  readonly expected: number;
}

interface PageResult {
  readonly userAgent: string;
  readonly adapter: GPUAdapterInfo;
  readonly differential: readonly DifferentialResult[];
  readonly multiRange: MultiRangeResult;
  readonly exactShape: readonly ExactShapeResult[];
}

interface OwnedBuffers {
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly bias?: GPUBuffer;
  readonly output: GPUBuffer;
}

const DIFFERENTIAL_SHAPE = Object.freeze({ rows: 33, inner: 37, columns: 131 });
const MULTI_RANGE_SHAPE = Object.freeze({ rows: 4_097, inner: 1, columns: 2_049 });
const EXACT_EXPANSION_SHAPE = Object.freeze({
  rows: 2_250,
  inner: 2_048,
  columns: 6_144,
});
const resultNode = requireResultNode();

void run().then(
  (result) => finish("passed", JSON.stringify(result, null, 2)),
  (error: unknown) => finish(
    "failed",
    error instanceof Error ? error.stack ?? error.message : String(error),
  ),
);

async function run(): Promise<PageResult> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  if (!adapter.features.has("shader-f16")) {
    throw new Error("The two-profile GEMM check requires shader-f16");
  }
  assertExactShapeLimits(adapter, EXACT_EXPANSION_SHAPE);
  const device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16"],
  });
  try {
    const differential = [
      await runDifferential(device, "reference-bf16"),
      await runDifferential(device, "raw-fp16"),
    ];
    const multiRange = await runMultiRange(device);
    const exactShape = [
      await runExactShape(device, "reference-bf16"),
      await runExactShape(device, "raw-fp16"),
    ];
    return {
      userAgent: navigator.userAgent,
      adapter: adapter.info,
      differential,
      multiRange,
      exactShape,
    };
  } finally {
    device.destroy();
  }
}

async function runDifferential(
  device: GPUDevice,
  profile: Profile,
): Promise<DifferentialResult> {
  const shape = DIFFERENTIAL_SHAPE;
  const activationValues = Float32Array.from(
    { length: shape.rows * shape.inner },
    (_, index) => Math.fround((((index * 17 + 3) % 29) - 14) / 32),
  );
  const weightValues = Float32Array.from(
    { length: shape.columns * shape.inner },
    (_, index) => Math.fround((((index * 13 + 7) % 31) - 15) / 64),
  );
  const biasValues = Float32Array.from(
    { length: shape.columns },
    (_, index) => Math.fround((((index * 5 + 2) % 17) - 8) / 128),
  );
  const activationData = profile === "reference-bf16"
    ? activationValues
    : Uint16Array.from(activationValues, numberToFp16Bits);
  const weightData = profile === "reference-bf16"
    ? packBf16(weightValues)
    : Uint16Array.from(weightValues, numberToFp16Bits);
  const biasData = profile === "reference-bf16"
    ? packBf16(biasValues)
    : Uint16Array.from(biasValues, numberToFp16Bits);
  const bytesPerOutput = profile === "reference-bf16" ? 4 : 2;
  const outputBytes = shape.rows * shape.columns * bytesPerOutput;
  const activation = storageBuffer(device, activationData);
  const weight = storageBuffer(device, weightData);
  const bias = storageBuffer(device, biasData);
  const scalarOutput = outputBuffer(device, outputBytes);
  const tiledOutput = outputBuffer(device, outputBytes);
  const kernel = AceCorrectnessGemmKernel.create(device, profile);
  try {
    await executeScalarOracle(
      device,
      profile,
      shape,
      {
        activation: binding(activation),
        weight: binding(weight),
        output: binding(scalarOutput),
        bias: binding(bias),
      },
    );
    const dispatch = await kernel.createDispatch(
      `browser-${profile}-differential-tiled`,
      shape,
      {
        activation: binding(activation),
        weight: binding(weight),
        output: binding(tiledOutput),
        bias: binding(bias),
      },
    );
    await executeDispatch(device, dispatch);
    const [scalarRaw, tiledRaw] = await Promise.all([
      readBuffer(device, scalarOutput, outputBytes),
      readBuffer(device, tiledOutput, outputBytes),
    ]);
    if (profile === "reference-bf16") {
      assertEqualWords(
        new Uint32Array(scalarRaw),
        new Uint32Array(tiledRaw),
        `${profile} scalar/tiled`,
      );
    } else {
      assertEqualWords(
        new Uint16Array(scalarRaw, 0, shape.rows * shape.columns),
        new Uint16Array(tiledRaw, 0, shape.rows * shape.columns),
        `${profile} scalar/tiled`,
      );
    }
    return {
      test: "scalar-gpu-differential",
      profile,
      shape,
      bias: true,
      comparedBits: shape.rows * shape.columns,
    };
  } finally {
    kernel.destroy();
    activation.destroy();
    weight.destroy();
    bias.destroy();
    scalarOutput.destroy();
    tiledOutput.destroy();
  }
}

async function runMultiRange(device: GPUDevice): Promise<MultiRangeResult> {
  const shape = MULTI_RANGE_SHAPE;
  const activationValues = Float32Array.from(
    { length: shape.rows },
    (_, index) => Math.fround(((index % 7) + 1) / 8),
  );
  const weightValues = Float32Array.from(
    { length: shape.columns },
    (_, index) => Math.fround(((index % 11) + 1) / 16),
  );
  const activation = storageBuffer(device, activationValues);
  const weight = storageBuffer(device, packBf16(weightValues));
  const output = outputBuffer(
    device,
    shape.rows * shape.columns * Float32Array.BYTES_PER_ELEMENT,
  );
  const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
  try {
    const dispatch = await kernel.createDispatch(
      "browser-reference-bf16-multi-range",
      shape,
      {
        activation: binding(activation),
        weight: binding(weight),
        output: binding(output),
      },
    );
    if (dispatch.rangeCount < 2) {
      throw new Error("multi-range fixture unexpectedly fit in one range");
    }
    const idleMilliseconds = await executeCooperatively(device, dispatch);
    const raw = await readBuffer(
      device,
      output,
      shape.rows * shape.columns * Float32Array.BYTES_PER_ELEMENT,
    );
    const actual = new Float32Array(raw);
    for (let row = 0; row < shape.rows; row += 1) {
      for (let column = 0; column < shape.columns; column += 1) {
        const index = row * shape.columns + column;
        const expected = Math.fround(
          activationValues[row]! * weightValues[column]!,
        );
        if (!Object.is(actual[index], expected)) {
          throw new Error(
            `multi-range routing mismatch at row ${row}, column ${column}: ` +
              `${actual[index]} != ${expected}`,
          );
        }
      }
    }
    return {
      test: "multi-range-routing",
      shape,
      rangeCount: dispatch.rangeCount,
      comparedValues: actual.length,
      queueEmptyIdleMilliseconds: idleMilliseconds,
    };
  } finally {
    kernel.destroy();
    activation.destroy();
    weight.destroy();
    output.destroy();
  }
}

async function runExactShape(
  device: GPUDevice,
  profile: Profile,
): Promise<ExactShapeResult> {
  const shape = EXACT_EXPANSION_SHAPE;
  const activationPattern = [-0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25] as const;
  const weightPattern = [-0.125, -0.0625, 0.03125, 0.0625, 0.125] as const;
  const biasPattern = [-0.0625, -0.03125, 0.03125, 0.0625] as const;
  const activationValues = Float32Array.from(
    { length: shape.rows * shape.inner },
    (_, index) => activationPattern[(index * 13 + 3) % activationPattern.length]!,
  );
  const weightValues = Float32Array.from(
    { length: shape.columns * shape.inner },
    (_, index) => weightPattern[(index * 7 + 1) % weightPattern.length]!,
  );
  const biasValues = Float32Array.from(
    { length: shape.columns },
    (_, index) => biasPattern[(index * 3 + 2) % biasPattern.length]!,
  );
  const activationData = profile === "reference-bf16"
    ? activationValues
    : Uint16Array.from(activationValues, numberToFp16Bits);
  const weightData = profile === "reference-bf16"
    ? packBf16(weightValues)
    : Uint16Array.from(weightValues, numberToFp16Bits);
  const biasData = profile === "reference-bf16"
    ? packBf16(biasValues)
    : Uint16Array.from(biasValues, numberToFp16Bits);
  const outputBytesPerElement = profile === "reference-bf16" ? 4 : 2;
  const activation = storageBuffer(device, activationData);
  const weight = storageBuffer(device, weightData);
  const bias = storageBuffer(device, biasData);
  const output = outputBuffer(
    device,
    shape.rows * shape.columns * outputBytesPerElement,
  );
  const owned: OwnedBuffers = { activation, weight, bias, output };
  const kernel = AceCorrectnessGemmKernel.create(device, profile);
  try {
    const dispatch = await kernel.createDispatch(
      `browser-${profile}-exact-expansion`,
      shape,
      {
        activation: binding(activation),
        weight: binding(weight),
        output: binding(output),
        bias: binding(bias),
      },
    );
    const heartbeat = startHeartbeat();
    const started = performance.now();
    const rangeMilliseconds: number[] = [];
    const idleMilliseconds: number[] = [];
    for (let range = 0; range < dispatch.rangeCount; range += 1) {
      const rangeStarted = performance.now();
      const encoder = device.createCommandEncoder({
        label: `${profile}-exact-range-${range}`,
      });
      const pass = encoder.beginComputePass();
      dispatch.encodeRange(pass, range);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      rangeMilliseconds.push(performance.now() - rangeStarted);
      if (range + 1 < dispatch.rangeCount) {
        idleMilliseconds.push(await queueEmptyIdle());
      }
    }
    const totalGpuPhaseMilliseconds = performance.now() - started;
    const heartbeatResult = heartbeat.stop();
    const checksum = await checksumOutput(
      device,
      output,
      profile,
      shape.rows * shape.columns,
    );
    const sampleIndices = exactSampleIndices(shape, dispatch);
    const actualSamples = await readSamples(
      device,
      output,
      profile,
      sampleIndices,
    );
    let maximumAbsoluteSampleError = 0;
    const boundarySentinels: OutputSentinel[] = [];
    for (let sample = 0; sample < sampleIndices.length; sample += 1) {
      const index = sampleIndices[sample]!;
      const row = Math.floor(index / shape.columns);
      const column = index % shape.columns;
      const expected = cpuExactSample(
        profile,
        shape,
        activationValues,
        weightValues,
        biasValues,
        row,
        column,
      );
      const error = Math.abs(actualSamples[sample]! - expected);
      maximumAbsoluteSampleError = Math.max(maximumAbsoluteSampleError, error);
      const tolerance = profile === "reference-bf16" ? 1e-5 : 0;
      if (!Number.isFinite(actualSamples[sample]!) || error > tolerance) {
        throw new Error(
          `${profile} exact-shape sample ${index} mismatch: ` +
            `${actualSamples[sample]} != ${expected} (error ${error})`,
        );
      }
      boundarySentinels.push({
        index,
        actual: actualSamples[sample]!,
        expected,
      });
    }
    const workgroupCount = dispatch.plan.outputRanges.reduce(
      (total, range) => total + range.workgroupCount,
      0,
    );
    const scheduledMultiplyAdds = dispatch.plan.outputRanges.reduce(
      (total, range) => total + range.multiplyAdds,
      0,
    );
    const orderedRanges = [...rangeMilliseconds].sort((left, right) => left - right);
    const maximumRangeMilliseconds = Math.max(...rangeMilliseconds);
    if (maximumRangeMilliseconds > 100) {
      throw new Error(
        `${profile} provisional safety range took ${maximumRangeMilliseconds} ms`,
      );
    }
    if (heartbeatResult.animationFrames === 0 || heartbeatResult.timerTicks === 0) {
      throw new Error(`${profile} exact-shape run starved the page heartbeat`);
    }
    return {
      test: "exact-shape-cooperative-timing",
      classification: "stage-1-enabling-diagnostic-not-a-benchmark",
      profile,
      shape,
      workgroupCount,
      logicalOutputCount: shape.rows * shape.columns,
      logicalMultiplyAdds: shape.rows * shape.columns * shape.inner,
      scheduledMultiplyAdds,
      rangeCount: dispatch.rangeCount,
      totalGpuPhaseMilliseconds,
      medianRangeMilliseconds: orderedRanges[Math.floor(orderedRanges.length / 2)]!,
      maximumRangeMilliseconds,
      minimumQueueEmptyIdleMilliseconds: Math.min(...idleMilliseconds),
      heartbeat: heartbeatResult,
      checkedSamples: sampleIndices.length,
      maximumAbsoluteSampleError,
      outputChecksum: checksum,
      boundarySentinels,
    };
  } finally {
    kernel.destroy();
    destroyOwnedBuffers(owned);
  }
}

async function executeScalarOracle(
  device: GPUDevice,
  profile: Profile,
  shape: AceGemmShape,
  bindings: AceGemmBufferBindings,
): Promise<void> {
  const module = device.createShaderModule({
    label: `${profile}-scalar-gemm-oracle-module`,
    code: aceScalarGemmOracleWgsl(profile, shape, bindings.bias !== undefined),
  });
  await assertShaderCompiles(module, `${profile} scalar GEMM oracle`);
  const pipeline = await device.createComputePipelineAsync({
    label: `${profile}-scalar-gemm-oracle-pipeline`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    label: `${profile}-scalar-gemm-oracle-bindings`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: bindings.activation },
      { binding: 1, resource: bindings.weight },
      { binding: 2, resource: bindings.output },
      ...(bindings.bias === undefined
        ? []
        : [{ binding: 3, resource: bindings.bias }]),
    ],
  });
  const plan = planAceGemm(shape);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeDispatch(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeCooperatively(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<readonly number[]> {
  const idleMilliseconds: number[] = [];
  for (let range = 0; range < dispatch.rangeCount; range += 1) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encodeRange(pass, range);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (range + 1 < dispatch.rangeCount) {
      idleMilliseconds.push(await queueEmptyIdle());
    }
  }
  return idleMilliseconds;
}

async function queueEmptyIdle(): Promise<number> {
  const started = performance.now();
  await new Promise<void>((resolve) => window.setTimeout(resolve, 1));
  const elapsed = performance.now() - started;
  if (elapsed < 0.9) {
    throw new Error(`queue-empty idle interval was only ${elapsed} ms`);
  }
  return elapsed;
}

function startHeartbeat(): { stop(): HeartbeatResult } {
  let active = true;
  let animationFrames = 0;
  let timerTicks = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = lastAnimationFrame;
  let animationFrameId = 0;
  const onAnimationFrame = (now: number): void => {
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrames += 1;
    if (active) animationFrameId = requestAnimationFrame(onAnimationFrame);
  };
  animationFrameId = requestAnimationFrame(onAnimationFrame);
  const timerId = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    timerTicks += 1;
  }, 16);
  return {
    stop(): HeartbeatResult {
      active = false;
      cancelAnimationFrame(animationFrameId);
      clearInterval(timerId);
      const now = performance.now();
      maximumAnimationFrameGapMilliseconds = Math.max(
        maximumAnimationFrameGapMilliseconds,
        now - lastAnimationFrame,
      );
      maximumTimerGapMilliseconds = Math.max(
        maximumTimerGapMilliseconds,
        now - lastTimer,
      );
      return {
        animationFrames,
        timerTicks,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
      };
    },
  };
}

function cpuExactSample(
  profile: Profile,
  shape: AceGemmShape,
  activation: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  row: number,
  column: number,
): number {
  let sum = 0;
  for (let inner = 0; inner < shape.inner; inner += 1) {
    const product = activation[row * shape.inner + inner]! *
      weight[column * shape.inner + inner]!;
    sum = profile === "reference-bf16"
      ? Math.fround(sum + Math.fround(product))
      : roundFp16(roundFp16(sum) + roundFp16(product));
  }
  return profile === "reference-bf16"
    ? Math.fround(sum + bias[column]!)
    : roundFp16(roundFp16(sum) + roundFp16(bias[column]!));
}

function exactSampleIndices(
  shape: AceGemmShape,
  dispatch: AceGemmDispatch,
): readonly number[] {
  const candidates: number[] = [
    0,
    126,
    128,
    shape.columns - 2,
    Math.floor(shape.rows / 2) * shape.columns + 2,
    Math.floor(shape.rows / 2) * shape.columns + shape.columns - 2,
    (shape.rows - 1) * shape.columns,
    shape.rows * shape.columns - 2,
  ];
  const columnTiles = Math.ceil(shape.columns / dispatch.plan.tileColumns);
  for (const range of dispatch.plan.outputRanges) {
    for (const workgroup of [
      range.firstWorkgroup,
      range.firstWorkgroup + range.workgroupCount - 1,
    ]) {
      const rowTile = Math.floor(workgroup / columnTiles);
      const columnTile = workgroup % columnTiles;
      const row = Math.min(shape.rows - 1, rowTile * dispatch.plan.tileRows);
      const column = Math.min(
        shape.columns - 2,
        columnTile * dispatch.plan.tileColumns,
      );
      candidates.push(row * shape.columns + column);
    }
  }
  return Object.freeze([...new Set(candidates.map((index) => index & ~1))]);
}

async function checksumOutput(
  device: GPUDevice,
  output: GPUBuffer,
  profile: Profile,
  elementCount: number,
): Promise<OutputChecksum> {
  const byteLength = elementCount * (profile === "reference-bf16" ? 4 : 2);
  const raw = await readBuffer(device, output, byteLength);
  const values = profile === "reference-bf16"
    ? new Float32Array(raw)
    : Float32Array.from(
        new Uint16Array(raw, 0, elementCount),
        fp16BitsToNumber,
      );
  let nonzeroCount = 0;
  let finiteCount = 0;
  let sum = 0;
  let absoluteSum = 0;
  let weightedSum = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    sum += value;
    absoluteSum += Math.abs(value);
    weightedSum += value * ((index % 251) + 1);
  }
  if (
    finiteCount !== elementCount ||
    nonzeroCount < Math.floor(elementCount * 0.95) ||
    !Number.isFinite(sum) ||
    !Number.isFinite(absoluteSum) ||
    !Number.isFinite(weightedSum) ||
    absoluteSum === 0
  ) {
    throw new Error(
      `${profile} exact-shape checksum is not a fully populated finite output`,
    );
  }
  return { nonzeroCount, finiteCount, sum, absoluteSum, weightedSum };
}

async function readSamples(
  device: GPUDevice,
  output: GPUBuffer,
  profile: Profile,
  sampleIndices: readonly number[],
): Promise<Float32Array> {
  const readback = device.createBuffer({
    label: `${profile}-exact-sample-readback`,
    size: sampleIndices.length * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    for (let sample = 0; sample < sampleIndices.length; sample += 1) {
      const index = sampleIndices[sample]!;
      const sourceOffset = index * (profile === "reference-bf16" ? 4 : 2);
      encoder.copyBufferToBuffer(output, sourceOffset, readback, sample * 4, 4);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    if (profile === "reference-bf16") {
      return new Float32Array(mapped.slice(0));
    }
    const raw = new DataView(mapped);
    return Float32Array.from(
      sampleIndices,
      (_, sample) => fp16BitsToNumber(raw.getUint16(sample * 4, true)),
    );
  } finally {
    readback.destroy();
  }
}

async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  const paddedBytes = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const readback = device.createBuffer({
    size: paddedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, paddedBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    return readback.getMappedRange().slice(0, byteLength);
  } finally {
    readback.destroy();
  }
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBuffer | ArrayBufferView<ArrayBufferLike>,
): GPUBuffer {
  const byteLength = data.byteLength;
  const paddedBytes = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const buffer = device.createBuffer({
    size: paddedBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const source = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  new Uint8Array(buffer.getMappedRange()).set(source);
  buffer.unmap();
  return buffer;
}

function outputBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, Math.ceil(byteLength / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function destroyOwnedBuffers(buffers: OwnedBuffers): void {
  buffers.activation.destroy();
  buffers.weight.destroy();
  buffers.bias?.destroy();
  buffers.output.destroy();
}

function packBf16(values: Float32Array): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  const scratch = new Float32Array(1);
  const bits = new Uint32Array(scratch.buffer);
  for (let index = 0; index < values.length; index += 1) {
    scratch[0] = values[index]!;
    const source = bits[0]!;
    const rounded = ((source + 0x7fff + ((source >>> 16) & 1)) >>> 16) & 0xffff;
    words[index >> 1] = words[index >> 1]! |
      (rounded << ((index & 1) * 16));
  }
  return words;
}

function roundFp16(value: number): number {
  return fp16BitsToNumber(numberToFp16Bits(value));
}

function numberToFp16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x800000;
    const shift = 14 - halfExponent;
    const truncated = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (remainder > halfway ||
      (remainder === halfway && (truncated & 1) !== 0) ? 1 : 0));
  }
  let roundedMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 ||
    (remainder === 0x1000 && (roundedMantissa & 1) !== 0)) {
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

function assertEqualWords(
  actual: Uint16Array | Uint32Array,
  expected: Uint16Array | Uint32Array,
  label: string,
): void {
  if (actual.length !== expected.length) {
    throw new Error(`${label} output length mismatch`);
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `${label} bit mismatch at ${index}: 0x${actual[index]!.toString(16)} != ` +
          `0x${expected[index]!.toString(16)}`,
      );
    }
  }
}

async function assertShaderCompiles(
  module: GPUShaderModule,
  label: string,
): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(errors.map((message) =>
      `${label} ${message.lineNum}:${message.linePos}: ${message.message}`
    ).join("\n"));
  }
}

function assertExactShapeLimits(adapter: GPUAdapter, shape: AceGemmShape): void {
  const largestBinding = Math.max(
    shape.rows * shape.inner * Float32Array.BYTES_PER_ELEMENT,
    shape.columns * shape.inner * Float32Array.BYTES_PER_ELEMENT / 2,
    shape.rows * shape.columns * Float32Array.BYTES_PER_ELEMENT,
  );
  if (
    adapter.limits.maxBufferSize < largestBinding ||
    adapter.limits.maxStorageBufferBindingSize < largestBinding
  ) {
    throw new Error(
      `adapter cannot bind exact-shape GEMM buffer of ${largestBinding} bytes`,
    );
  }
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = message;
  document.title = `ACE tiled GEMM ${status}`;
}

function requireResultNode(): HTMLPreElement {
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result element");
  return node;
}
