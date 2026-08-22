import {
  AceCorrectnessAttentionKernel,
} from "../../src/webgpu/kernels/attention.js";
import {
  AceCorrectnessKvCacheWriteKernel,
} from "../../src/webgpu/kernels/kv-cache.js";
import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";

const BATCH = 2;
const KV_HEADS = 1;
const HEAD_DIMENSION = 2;
const CACHE_CAPACITY = 6;

interface PlannerCaseResult {
  readonly profile: AceModelProfileId;
  readonly prefillStatus: readonly number[];
  readonly appendStatus: readonly number[];
  readonly attentionActual: readonly number[];
  readonly attentionExpected: readonly number[];
  readonly failClosedAttentionActual: readonly number[];
  readonly finalValidity: readonly number[];
  readonly rangeFailureStatus: readonly number[];
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) =>
    finish("failed", error instanceof Error ? error.stack ?? error.message : String(error)),
);

async function run(): Promise<readonly PlannerCaseResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  try {
    const results: PlannerCaseResult[] = [
      await runPlannerCase(device, "reference-bf16"),
    ];
    if (device.features.has("shader-f16")) {
      results.push(await runPlannerCase(device, "raw-fp16"));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runPlannerCase(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<PlannerCaseResult> {
  const buffers: GPUBuffer[] = [];
  const cacheKernel = AceCorrectnessKvCacheWriteKernel.create(device, profile);
  const attentionKernel = AceCorrectnessAttentionKernel.create(device, profile);
  const cacheElements = BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIMENSION;
  const cacheKeyExpected = new Float32Array(cacheElements).fill(4);
  const cacheValueExpected = new Float32Array(cacheElements).fill(4);
  const cacheValidityExpected = new Uint32Array(BATCH * CACHE_CAPACITY);
  const cacheKey = track(
    buffers,
    storageBuffer(device, profileData(profile, cacheKeyExpected), true),
  );
  const cacheValue = track(
    buffers,
    storageBuffer(device, profileData(profile, cacheValueExpected), true),
  );
  const cacheValidity = track(
    buffers,
    storageBuffer(device, cacheValidityExpected, true),
  );
  const writeStatus = track(
    buffers,
    storageBuffer(device, new Uint32Array([9, 9]), true),
  );

  try {
    const prefillKey = new Float32Array([
      1, 0, 0, 1, 1, 1,
      0.5, 0, 0, 0.5, 1, -1,
    ]);
    const prefillValue = new Float32Array([
      1, 0.5, 2, 1, 3, 1.5,
      -1, 0.25, 0.5, -0.5, 4, 2,
    ]);
    const prefillValidity = new Uint32Array([1, 1, 2, 1, 0, 1]);
    const prefillStarts = new Uint32Array([2, 1]);
    const prefillBuffers = makeWriteSources(
      device,
      profile,
      buffers,
      prefillKey,
      prefillValue,
      prefillValidity,
      prefillStarts,
    );
    const prefill = await cacheKernel.createDispatch(
      `ace-browser-${profile}-planner-prefill`,
      {
        batch: BATCH,
        keyValueHeads: KV_HEADS,
        appendTokens: 3,
        cacheCapacity: CACHE_CAPACITY,
        headDimension: HEAD_DIMENSION,
      },
      {
        ...prefillBuffers,
        cacheKey: binding(cacheKey),
        cacheValue: binding(cacheValue),
        cacheValidity: binding(cacheValidity),
        writeStatus: binding(writeStatus),
      },
    );
    await execute(device, prefill);
    const prefillStatus = await readU32(device, writeStatus, BATCH);
    assertExact(prefillStatus, new Uint32Array([1, 1]), "prefill status");
    cpuCacheWrite(
      cacheKeyExpected,
      cacheValueExpected,
      cacheValidityExpected,
      prefillKey,
      prefillValue,
      prefillValidity,
      prefillStarts,
      3,
    );

    const appendKey = new Float32Array([0.25, -0.5, -0.5, 0.25]);
    const appendValue = new Float32Array([2, -2, -2, 1]);
    const appendValidity = new Uint32Array([1, 1]);
    const appendStarts = new Uint32Array([5, 4]);
    const appendBuffers = makeWriteSources(
      device,
      profile,
      buffers,
      appendKey,
      appendValue,
      appendValidity,
      appendStarts,
    );
    const append = await cacheKernel.createDispatch(
      `ace-browser-${profile}-planner-append`,
      {
        batch: BATCH,
        keyValueHeads: KV_HEADS,
        appendTokens: 1,
        cacheCapacity: CACHE_CAPACITY,
        headDimension: HEAD_DIMENSION,
      },
      {
        ...appendBuffers,
        cacheKey: binding(cacheKey),
        cacheValue: binding(cacheValue),
        cacheValidity: binding(cacheValidity),
        writeStatus: binding(writeStatus),
      },
    );
    await execute(device, append);
    const appendStatus = await readU32(device, writeStatus, BATCH);
    assertExact(appendStatus, new Uint32Array([1, 1]), "append status");
    cpuCacheWrite(
      cacheKeyExpected,
      cacheValueExpected,
      cacheValidityExpected,
      appendKey,
      appendValue,
      appendValidity,
      appendStarts,
      1,
    );

    const cacheKeyActual = await readProfile(device, profile, cacheKey, cacheElements);
    const cacheValueActual = await readProfile(device, profile, cacheValue, cacheElements);
    const cacheValidityActual = await readU32(
      device,
      cacheValidity,
      cacheValidityExpected.length,
    );
    assertClose(cacheKeyActual, cacheKeyExpected, 0, "cache key");
    assertClose(cacheValueActual, cacheValueExpected, 0, "cache value");
    assertExact(cacheValidityActual, cacheValidityExpected, "cache validity");

    // Rows have different left padding and different physical next-token
    // positions. The noncanonical prefill mask value `2` must remain invalid.
    const query = new Float32Array([1, 0.5, 0.5, -1]);
    const queryPositions = new Uint32Array([5, 4]);
    const validLengths = new Uint32Array([1, 6, 1, 5]);
    const expected = cpuPlannerAttention(
      query,
      cacheKeyExpected,
      cacheValueExpected,
      cacheValidityExpected,
      queryPositions,
      validLengths,
    );
    const queryBuffer = track(buffers, storageBuffer(device, profileData(profile, query)));
    const queryPositionsBuffer = track(buffers, storageBuffer(device, queryPositions));
    const validLengthsBuffer = track(buffers, storageBuffer(device, validLengths));
    const attentionOutput = track(
      buffers,
      device.createBuffer({
        size: Math.max(4, expected.length * profileElementBytes(profile)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    );
    const attention = await attentionKernel.createDispatch(
      `ace-browser-${profile}-masked-planner-attention`,
      {
        batch: BATCH,
        queryHeads: 1,
        keyValueHeads: KV_HEADS,
        queryTokens: 1,
        keyValueTokens: CACHE_CAPACITY,
        headDimension: HEAD_DIMENSION,
        mode: "causal",
        keyValidity: "causal-per-key",
      },
      {
        query: binding(queryBuffer),
        key: binding(cacheKey),
        value: binding(cacheValue),
        validLengths: binding(validLengthsBuffer),
        output: binding(attentionOutput),
        keyValidity: binding(cacheValidity),
        queryPositions: binding(queryPositionsBuffer),
      },
    );
    await execute(device, attention);
    const actual = await readProfile(device, profile, attentionOutput, expected.length);
    assertClose(actual, expected, profile === "raw-fp16" ? 3e-3 : 2e-5, "attention");

    // Row 0 has visible keys but a query position outside its declared cache
    // extent; row 1 has a valid position but no admitted keys. Both fail shut.
    const emptyPositions = new Uint32Array([5, 0]);
    const emptyLengths = new Uint32Array([1, 4, 1, 1]);
    const emptyPositionsBuffer = track(buffers, storageBuffer(device, emptyPositions));
    const emptyLengthsBuffer = track(buffers, storageBuffer(device, emptyLengths));
    const emptyOutput = track(
      buffers,
      device.createBuffer({
        size: Math.max(4, query.length * profileElementBytes(profile)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    );
    const emptyAttention = await attentionKernel.createDispatch(
      `ace-browser-${profile}-empty-planner-mask`,
      {
        batch: BATCH,
        queryHeads: 1,
        keyValueHeads: KV_HEADS,
        queryTokens: 1,
        keyValueTokens: CACHE_CAPACITY,
        headDimension: HEAD_DIMENSION,
        mode: "causal",
        keyValidity: "causal-per-key",
      },
      {
        query: binding(queryBuffer),
        key: binding(cacheKey),
        value: binding(cacheValue),
        validLengths: binding(emptyLengthsBuffer),
        output: binding(emptyOutput),
        keyValidity: binding(cacheValidity),
        queryPositions: binding(emptyPositionsBuffer),
      },
    );
    await execute(device, emptyAttention);
    const failClosedAttentionActual = await readProfile(
      device,
      profile,
      emptyOutput,
      query.length,
    );
    assertClose(
      failClosedAttentionActual,
      new Float32Array(query.length),
      0,
      "fail-closed attention",
    );

    // A two-token write at row 0 position 5 cannot fit and must leave that
    // whole row untouched; row 1 position 4 fits exactly and is committed.
    const rangeKey = new Float32Array([
      -1, -1, -2, -2,
      1, 1, 2, 2,
    ]);
    const rangeValue = new Float32Array([
      -2, -1, -1, -2,
      0.5, 1, 1.5, 2,
    ]);
    const rangeValidity = new Uint32Array([1, 1, 1, 1]);
    const rangeStarts = new Uint32Array([5, 4]);
    const rangeBuffers = makeWriteSources(
      device,
      profile,
      buffers,
      rangeKey,
      rangeValue,
      rangeValidity,
      rangeStarts,
    );
    const rangeWrite = await cacheKernel.createDispatch(
      `ace-browser-${profile}-planner-range-failure`,
      {
        batch: BATCH,
        keyValueHeads: KV_HEADS,
        appendTokens: 2,
        cacheCapacity: CACHE_CAPACITY,
        headDimension: HEAD_DIMENSION,
      },
      {
        ...rangeBuffers,
        cacheKey: binding(cacheKey),
        cacheValue: binding(cacheValue),
        cacheValidity: binding(cacheValidity),
        writeStatus: binding(writeStatus),
      },
    );
    await execute(device, rangeWrite);
    const rangeFailureStatus = await readU32(device, writeStatus, BATCH);
    assertExact(rangeFailureStatus, new Uint32Array([0, 1]), "range status");
    cpuCacheWrite(
      cacheKeyExpected,
      cacheValueExpected,
      cacheValidityExpected,
      rangeKey,
      rangeValue,
      rangeValidity,
      rangeStarts,
      2,
    );
    assertClose(
      await readProfile(device, profile, cacheKey, cacheElements),
      cacheKeyExpected,
      0,
      "post-range cache key",
    );
    assertClose(
      await readProfile(device, profile, cacheValue, cacheElements),
      cacheValueExpected,
      0,
      "post-range cache value",
    );
    const finalValidity = await readU32(device, cacheValidity, cacheValidityExpected.length);
    assertExact(finalValidity, cacheValidityExpected, "post-range cache validity");

    return {
      profile,
      prefillStatus: [...prefillStatus],
      appendStatus: [...appendStatus],
      attentionActual: [...actual],
      attentionExpected: [...expected],
      failClosedAttentionActual: [...failClosedAttentionActual],
      finalValidity: [...finalValidity],
      rangeFailureStatus: [...rangeFailureStatus],
    };
  } finally {
    attentionKernel.destroy();
    cacheKernel.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function makeWriteSources(
  device: GPUDevice,
  profile: AceModelProfileId,
  buffers: GPUBuffer[],
  key: Float32Array,
  value: Float32Array,
  validity: Uint32Array,
  starts: Uint32Array,
) {
  return {
    sourceKey: binding(track(buffers, storageBuffer(device, profileData(profile, key)))),
    sourceValue: binding(track(buffers, storageBuffer(device, profileData(profile, value)))),
    sourceValidity: binding(track(buffers, storageBuffer(device, validity))),
    rowStartPositions: binding(track(buffers, storageBuffer(device, starts))),
  };
}

function cpuCacheWrite(
  cacheKey: Float32Array,
  cacheValue: Float32Array,
  cacheValidity: Uint32Array,
  sourceKey: Float32Array,
  sourceValue: Float32Array,
  sourceValidity: Uint32Array,
  starts: Uint32Array,
  appendTokens: number,
): Uint32Array {
  const status = new Uint32Array(BATCH);
  for (let batch = 0; batch < BATCH; batch += 1) {
    const start = starts[batch]!;
    if (start + appendTokens > CACHE_CAPACITY) continue;
    status[batch] = 1;
    for (let head = 0; head < KV_HEADS; head += 1) {
      for (let token = 0; token < appendTokens; token += 1) {
        for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
          const sourceIndex =
            ((batch * KV_HEADS + head) * appendTokens + token) * HEAD_DIMENSION + dimension;
          const destinationIndex =
            ((batch * KV_HEADS + head) * CACHE_CAPACITY + start + token) *
              HEAD_DIMENSION + dimension;
          cacheKey[destinationIndex] = sourceKey[sourceIndex]!;
          cacheValue[destinationIndex] = sourceValue[sourceIndex]!;
        }
      }
    }
    for (let token = 0; token < appendTokens; token += 1) {
      cacheValidity[batch * CACHE_CAPACITY + start + token] =
        sourceValidity[batch * appendTokens + token] === 1 ? 1 : 0;
    }
  }
  return status;
}

function cpuPlannerAttention(
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  keyValidity: Uint32Array,
  queryPositions: Uint32Array,
  validLengths: Uint32Array,
): Float32Array {
  const output = new Float32Array(query.length);
  for (let batch = 0; batch < BATCH; batch += 1) {
    const physicalPosition = queryPositions[batch]!;
    if (physicalPosition >= CACHE_CAPACITY || validLengths[batch * 2]! === 0) continue;
    const keyEnd = Math.min(validLengths[batch * 2 + 1]!, physicalPosition + 1);
    const admitted: number[] = [];
    const scores: number[] = [];
    for (let token = 0; token < keyEnd; token += 1) {
      if (keyValidity[batch * CACHE_CAPACITY + token] !== 1) continue;
      admitted.push(token);
      let score = 0;
      for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
        const queryIndex = batch * HEAD_DIMENSION + dimension;
        const keyIndex = (batch * CACHE_CAPACITY + token) * HEAD_DIMENSION + dimension;
        score += query[queryIndex]! * key[keyIndex]!;
      }
      scores.push(score / Math.sqrt(HEAD_DIMENSION));
    }
    if (scores.length === 0) continue;
    const maximum = Math.max(...scores);
    const weights = scores.map((score) => Math.exp(score - maximum));
    const denominator = weights.reduce((sum, weight) => sum + weight, 0);
    for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
      let weighted = 0;
      for (let index = 0; index < admitted.length; index += 1) {
        const valueIndex =
          (batch * CACHE_CAPACITY + admitted[index]!) * HEAD_DIMENSION + dimension;
        weighted += weights[index]! * value[valueIndex]!;
      }
      output[batch * HEAD_DIMENSION + dimension] = weighted / denominator;
    }
  }
  return output;
}

function profileData(
  profile: AceModelProfileId,
  values: Float32Array,
): Float32Array | Uint16Array<ArrayBuffer> {
  return profile === "raw-fp16" ? packFp16Exact(values) : values;
}

const EXACT_FP16_BITS = new Map<number, number>([
  [-2, 0xc000],
  [-1, 0xbc00],
  [-0.5, 0xb800],
  [0, 0x0000],
  [0.25, 0x3400],
  [0.5, 0x3800],
  [1, 0x3c00],
  [1.5, 0x3e00],
  [2, 0x4000],
  [3, 0x4200],
  [4, 0x4400],
]);

function packFp16Exact(values: ArrayLike<number>): Uint16Array<ArrayBuffer> {
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
  copySource = false,
): GPUBuffer {
  const byteLength = "byteLength" in data ? data.byteLength : 0;
  const padded = new Uint8Array(Math.max(4, Math.ceil(byteLength / 4) * 4));
  const sourceBytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  padded.set(sourceBytes);
  const buffer = device.createBuffer({
    size: padded.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      (copySource ? GPUBufferUsage.COPY_SRC : 0),
  });
  device.queue.writeBuffer(buffer, 0, padded);
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function track(buffers: GPUBuffer[], buffer: GPUBuffer): GPUBuffer {
  buffers.push(buffer);
  return buffer;
}

function profileElementBytes(profile: AceModelProfileId): number {
  return profile === "raw-fp16" ? 2 : 4;
}

async function execute(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
  source: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  if (profile === "reference-bf16") {
    return new Float32Array(await readBytes(device, source, count * 4));
  }
  const words = new Uint16Array(await readBytes(device, source, count * 2), 0, count);
  return Float32Array.from(words, fp16BitsToNumber);
}

async function readU32(
  device: GPUDevice,
  source: GPUBuffer,
  count: number,
): Promise<Uint32Array> {
  return new Uint32Array(await readBytes(device, source, count * 4));
}

async function readBytes(
  device: GPUDevice,
  source: GPUBuffer,
  bytes: number,
): Promise<ArrayBuffer> {
  const paddedBytes = Math.max(4, Math.ceil(bytes / 4) * 4);
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
    return readback.getMappedRange(0, paddedBytes).slice(0, bytes);
  } finally {
    readback.destroy();
  }
}

function assertClose(
  actual: Float32Array,
  expected: Float32Array,
  tolerance: number,
  label: string,
): void {
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error(`Invalid ${label} tolerance ${tolerance}`);
  }
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  for (let index = 0; index < actual.length; index += 1) {
    const received = actual[index]!;
    const wanted = expected[index]!;
    if (
      !Number.isFinite(received) ||
      !Number.isFinite(wanted) ||
      Math.abs(received - wanted) > tolerance
    ) {
      throw new Error(
        `${label} mismatch at ${index}: received ${received}, expected ${wanted}`,
      );
    }
  }
}

function assertExact(
  actual: Uint32Array,
  expected: Uint32Array,
  label: string,
): void {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `${label} mismatch at ${index}: received ${actual[index]}, expected ${expected[index]}`,
      );
    }
  }
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = message;
  document.title = `ACE planner attention/cache ${status}`;
}

function requireResultNode(): HTMLPreElement {
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result element");
  return node;
}
