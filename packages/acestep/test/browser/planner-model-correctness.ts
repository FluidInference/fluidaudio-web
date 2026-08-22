import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  AceCorrectnessPlannerModelRuntime,
  type AcePlannerModelBindings,
} from "../../src/webgpu/planner-model.js";
import {
  createAceQwen3CausalControlData,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  type AceQwen3BlockScratch,
  type AceQwen3BlockWeights,
  type AceQwen3Config,
} from "../../src/webgpu/qwen3.js";

const BATCH = 2 as const;
const TOKENS = 3;
const CACHE_CAPACITY = 4;
const HIDDEN = 4;
const INTERMEDIATE = 6;
const QUERY_HEADS = 2;
const KV_HEADS = 1;
const HEAD_DIMENSION = 2;
const VOCABULARY = 7;

const CONFIG: AceQwen3Config = Object.freeze({
  id: "ace-browser-tiny-planner-composer",
  hiddenSize: HIDDEN,
  intermediateSize: INTERMEDIATE,
  layerCount: 1,
  queryHeads: QUERY_HEADS,
  keyValueHeads: KV_HEADS,
  headDimension: HEAD_DIMENSION,
  vocabularySize: VOCABULARY,
  maximumPositionEmbeddings: 8,
  ropeTheta: 100,
  rmsNormEpsilon: 1e-5,
  attentionBias: false,
  hiddenActivation: "silu",
  tieWordEmbeddings: true,
});

// Exactly representable in BF16 and FP16. Zero projections make the decoder
// block an identity while still executing every cached-Qwen primitive. The
// independently computed oracle therefore isolates embedding, residual,
// final-norm, physical-last-row gather, and tied sharded output wiring.
const EMBEDDING = new Float32Array([
  0.25, -0.5, 0.75, 0.125,
  -0.75, 0.5, 0.25, -0.125,
  0.5, 0.25, -0.25, 0.75,
  -0.5, 0.125, 0.625, -0.25,
  0.375, -0.625, 0.5, 0.25,
  -0.25, 0.75, -0.5, 0.375,
  0.625, 0.25, 0.125, -0.75,
]);
const TOKEN_IDS = new Uint32Array([0, 1, 2, 3, 4, 5]);

interface ProfileResult {
  readonly profile: AceModelProfileId;
  readonly quantumKinds: readonly string[];
  readonly primitiveCount: number;
  readonly maximumLogitError: number;
  readonly writeStatus: readonly number[];
  readonly cacheValidity: readonly number[];
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) =>
    finish("failed", error instanceof Error ? error.stack ?? error.message : String(error)),
);

async function run(): Promise<readonly ProfileResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  try {
    const results: ProfileResult[] = [await runProfile(device, "reference-bf16")];
    if (device.features.has("shader-f16")) {
      results.push(await runProfile(device, "raw-fp16"));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<ProfileResult> {
  const runtime = AceCorrectnessPlannerModelRuntime.create(device, profile);
  const buffers: GPUBuffer[] = [];
  try {
    const block = planAceQwen3Block(CONFIG, {
      batch: BATCH,
      tokens: TOKENS,
      attention: { kind: "cached", cacheCapacity: CACHE_CAPACITY },
    });
    const controls = createAceQwen3CausalControlData({
      batch: BATCH,
      tokens: TOKENS,
      cacheCapacity: CACHE_CAPACITY,
      rowStartPositions: [0, 0],
      validKeyLengths: [3, 3],
      sourceValidity: [0, 1, 1, 1, 1, 1],
    });
    const rope = createAceQwen3RopeTables([0, 1, 2, 0, 1, 2], {
      batch: BATCH,
      tokens: TOKENS,
      headDimension: HEAD_DIMENSION,
      ropeTheta: CONFIG.ropeTheta,
      maximumPositionEmbeddings: CONFIG.maximumPositionEmbeddings,
    });
    const embedding0 = EMBEDDING.slice(0, 4 * HIDDEN);
    const embedding1 = EMBEDDING.slice(4 * HIDDEN);
    const embeddingWeights = [
      {
        firstRow: 0,
        rowCount: 4,
        weight: weightBinding(device, buffers, profile, embedding0),
      },
      {
        firstRow: 4,
        rowCount: 3,
        weight: weightBinding(device, buffers, profile, embedding1),
      },
    ] as const;
    const zero = (elements: number): GPUBufferBinding =>
      weightBinding(device, buffers, profile, new Float32Array(elements));
    const one = (elements: number): GPUBufferBinding =>
      weightBinding(device, buffers, profile, new Float32Array(elements).fill(1));
    const layerWeights: AceQwen3BlockWeights = {
      inputLayerNorm: one(HIDDEN),
      queryProjection: zero(QUERY_HEADS * HEAD_DIMENSION * HIDDEN),
      keyProjection: zero(KV_HEADS * HEAD_DIMENSION * HIDDEN),
      valueProjection: zero(KV_HEADS * HEAD_DIMENSION * HIDDEN),
      queryNorm: one(HEAD_DIMENSION),
      keyNorm: one(HEAD_DIMENSION),
      outputProjection: zero(HIDDEN * QUERY_HEADS * HEAD_DIMENSION),
      postAttentionLayerNorm: one(HIDDEN),
      gateProjection: zero(INTERMEDIATE * HIDDEN),
      upProjection: zero(INTERMEDIATE * HIDDEN),
      downProjection: zero(HIDDEN * INTERMEDIATE),
    };
    const createActivation = (elements: number, copySource = true): GPUBufferBinding =>
      binding(track(buffers, activationBuffer(device, profile, elements, copySource)));
    const cacheValidity = track(buffers, u32Buffer(
      device,
      new Uint32Array(BATCH * CACHE_CAPACITY),
      true,
    ));
    const writeStatus = track(buffers, u32Buffer(device, new Uint32Array(BATCH), true));
    const logitBuffers = [
      track(buffers, activationBuffer(device, profile, BATCH * 4, true)),
      track(buffers, activationBuffer(device, profile, BATCH * 3, true)),
    ] as const;
    const bindings: AcePlannerModelBindings = {
      tokenIds: binding(track(buffers, u32Buffer(device, TOKEN_IDS))),
      weights: {
        embedding: embeddingWeights,
        layers: [layerWeights],
        finalNorm: one(HIDDEN),
      },
      controls: {
        validLengths: binding(track(buffers, u32Buffer(device, controls.validLengths))),
        queryPositions: binding(track(buffers, u32Buffer(device, controls.queryPositions))),
        sourceValidity: binding(track(buffers, u32Buffer(device, controls.sourceValidity))),
        rowStartPositions: binding(
          track(buffers, u32Buffer(device, controls.rowStartPositions)),
        ),
        cosine: binding(track(buffers, f32Buffer(device, rope.cosine))),
        sine: binding(track(buffers, f32Buffer(device, rope.sine))),
        lastPhysicalRowIndices: binding(
          track(buffers, u32Buffer(device, new Uint32Array([2, 2]))),
        ),
        cacheValidity: binding(cacheValidity),
        writeStatus: binding(writeStatus),
      },
      cache: {
        layers: [{
          key: createActivation(
            BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIMENSION,
          ),
          value: createActivation(
            BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIMENSION,
          ),
        }],
      },
      scratch: {
        embedded: createActivation(block.hiddenElements),
        block: blockScratch(createActivation, block),
        layerOutputs: [
          createActivation(block.hiddenElements),
          createActivation(block.hiddenElements),
        ],
        normalizedSequence: createActivation(block.hiddenElements),
        lastHiddenRows: createActivation(BATCH * HIDDEN),
      },
      logits: logitBuffers.map(binding),
    };
    const dispatch = await runtime.createQwen3PlannerDispatch(
      `ace-browser-${profile}-planner-model`,
      CONFIG,
      {
        kind: "prefill",
        batch: BATCH,
        tokens: TOKENS,
        cacheCapacity: CACHE_CAPACITY,
      },
      bindings,
    );
    const expectedKinds = [
      "embedding",
      "layer",
      "final-norm",
      "last-row-gather",
      "tied-lm-head",
    ];
    const actualKinds = dispatch.quanta.filter((quantum, index) =>
      index === 0 || quantum.logicalId !== dispatch.quanta[index - 1]?.logicalId
    ).map(({ kind }) => kind);
    if (actualKinds.join(",") !== expectedKinds.join(",")) {
      throw new Error("Planner cooperative quantum order changed");
    }
    for (let index = 0; index < dispatch.quanta.length; index += 1) {
      await execute(device, dispatch.quanta[index]!);
      if (index + 1 < dispatch.quanta.length) await delay(1);
    }
    const shard0 = await readProfile(device, profile, logitBuffers[0], BATCH * 4);
    const shard1 = await readProfile(device, profile, logitBuffers[1], BATCH * 3);
    const actual = new Float32Array(BATCH * VOCABULARY);
    for (let row = 0; row < BATCH; row += 1) {
      actual.set(shard0.slice(row * 4, row * 4 + 4), row * VOCABULARY);
      actual.set(shard1.slice(row * 3, row * 3 + 3), row * VOCABULARY + 4);
    }
    const expected = expectedLogits(profile);
    const actualLastHidden = await readProfile(
      device,
      profile,
      bindings.scratch.lastHiddenRows.buffer,
      BATCH * HIDDEN,
    );
    const maximumLogitError = assertClose(
      actual,
      expected,
      profile === "reference-bf16" ? 2e-5 : 3e-3,
      `${profile} planner composer logits`,
      actualLastHidden,
    );
    const actualStatus = await readU32(device, writeStatus, BATCH);
    const actualValidity = await readU32(
      device,
      cacheValidity,
      BATCH * CACHE_CAPACITY,
    );
    assertExact(actualStatus, new Uint32Array([1, 1]), `${profile} write status`);
    assertExact(
      actualValidity,
      new Uint32Array([0, 1, 1, 0, 1, 1, 1, 0]),
      `${profile} cache validity`,
    );
    return {
      profile,
      quantumKinds: expectedKinds,
      primitiveCount: dispatch.primitiveCount,
      maximumLogitError,
      writeStatus: [...actualStatus],
      cacheValidity: [...actualValidity],
    };
  } finally {
    runtime.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function blockScratch(
  create: (elements: number) => GPUBufferBinding,
  plan: ReturnType<typeof planAceQwen3Block>,
): AceQwen3BlockScratch {
  return {
    normalizedInput: create(plan.hiddenElements),
    queryFlat: create(plan.queryElements),
    keyFlat: create(plan.keyValueElements),
    valueFlat: create(plan.keyValueElements),
    queryHeads: create(plan.queryElements),
    keyHeads: create(plan.keyValueElements),
    valueHeads: create(plan.keyValueElements),
    normalizedQueryHeads: create(plan.queryElements),
    normalizedKeyHeads: create(plan.keyValueElements),
    rotatedQueryHeads: create(plan.queryElements),
    rotatedKeyHeads: create(plan.keyValueElements),
    attentionHeads: create(plan.queryElements),
    mergedAttention: create(plan.queryElements),
    projectedAttention: create(plan.hiddenElements),
    afterAttention: create(plan.hiddenElements),
    normalizedAfterAttention: create(plan.hiddenElements),
    gate: create(plan.intermediateElements),
    up: create(plan.intermediateElements),
    gatedActivation: create(plan.intermediateElements),
    projectedMlp: create(plan.hiddenElements),
  };
}

function expectedLogits(profile: AceModelProfileId): Float32Array {
  const weights = Float32Array.from(EMBEDDING, (value) => weightValue(profile, value));
  const output = new Float32Array(BATCH * VOCABULARY);
  for (let row = 0; row < BATCH; row += 1) {
    const tokenId = row === 0 ? 2 : 5;
    const hidden = weights.slice(tokenId * HIDDEN, (tokenId + 1) * HIDDEN);
    let squareSum = Math.fround(0);
    for (const value of hidden) {
      squareSum = Math.fround(squareSum + Math.fround(value * value));
    }
    const inverse = Math.fround(
      1 / Math.sqrt(Math.fround(squareSum / HIDDEN + CONFIG.rmsNormEpsilon)),
    );
    const normalized = Float32Array.from(hidden, (value) =>
      store(profile, Math.fround(value * inverse))
    );
    for (let vocabulary = 0; vocabulary < VOCABULARY; vocabulary += 1) {
      let sum = store(profile, 0);
      for (let dimension = 0; dimension < HIDDEN; dimension += 1) {
        const product = store(
          profile,
          normalized[dimension]! * weights[vocabulary * HIDDEN + dimension]!,
        );
        sum = store(profile, sum + product);
      }
      output[row * VOCABULARY + vocabulary] = sum;
    }
  }
  return output;
}

function weightBinding(
  device: GPUDevice,
  buffers: GPUBuffer[],
  profile: AceModelProfileId,
  values: Float32Array,
): GPUBufferBinding {
  const data = profile === "reference-bf16" ? packBf16(values) : packFp16(values);
  return binding(track(buffers, bytesBuffer(device, data)));
}

function activationBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  elements: number,
  copySource = false,
): GPUBuffer {
  return bytesBuffer(
    device,
    new Uint8Array(elements * (profile === "reference-bf16" ? 4 : 2)),
    copySource,
  );
}

function u32Buffer(
  device: GPUDevice,
  values: Uint32Array,
  copySource = false,
): GPUBuffer {
  return bytesBuffer(device, values, copySource);
}

function f32Buffer(device: GPUDevice, values: Float32Array): GPUBuffer {
  return bytesBuffer(device, values);
}

function bytesBuffer(
  device: GPUDevice,
  source: ArrayBufferView<ArrayBufferLike>,
  copySource = false,
): GPUBuffer {
  const padded = new Uint8Array(Math.max(4, Math.ceil(source.byteLength / 4) * 4));
  padded.set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength));
  const buffer = device.createBuffer({
    size: padded.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
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
  return Float32Array.from(
    new Uint16Array(await readBytes(device, source, count * 2)),
    fp16BitsToNumber,
  );
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
  const padded = Math.max(4, Math.ceil(bytes / 4) * 4);
  const readback = device.createBuffer({
    size: padded,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, padded);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    return readback.getMappedRange(0, padded).slice(0, bytes);
  } finally {
    readback.destroy();
  }
}

function assertClose(
  actual: Float32Array,
  expected: Float32Array,
  tolerance: number,
  label: string,
  diagnostic?: Float32Array,
): number {
  let maximum = 0;
  if (actual.length !== expected.length) throw new Error(`${label} length changed`);
  for (let index = 0; index < actual.length; index += 1) {
    const error = Math.abs(actual[index]! - expected[index]!);
    maximum = Math.max(maximum, error);
    if (!Number.isFinite(error) || error > tolerance) {
      throw new Error(
        `${label} mismatch at ${index}: ${actual[index]} vs ${expected[index]}; ` +
          `lastHidden=[${diagnostic === undefined ? "" : [...diagnostic].join(",")}]`,
      );
    }
  }
  return maximum;
}

function assertExact(actual: Uint32Array, expected: Uint32Array, label: string): void {
  if (actual.length !== expected.length) throw new Error(`${label} length changed`);
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${label} mismatch at ${index}`);
    }
  }
}

function store(profile: AceModelProfileId, value: number): number {
  return profile === "reference-bf16"
    ? Math.fround(value)
    : fp16BitsToNumber(numberToFp16Bits(value));
}

function weightValue(profile: AceModelProfileId, value: number): number {
  return profile === "reference-bf16"
    ? bf16BitsToNumber(numberToBf16Bits(value))
    : fp16BitsToNumber(numberToFp16Bits(value));
}

function packBf16(values: ArrayLike<number>): Uint32Array<ArrayBuffer> {
  const packed = new Uint32Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index += 1) {
    const bits = numberToBf16Bits(values[index]!);
    const pair = index >> 1;
    packed[pair] = index % 2 === 0
      ? (packed[pair]! & 0xffff_0000) | bits
      : (packed[pair]! & 0x0000_ffff) | (bits << 16);
  }
  return packed;
}

function packFp16(values: ArrayLike<number>): Uint16Array<ArrayBuffer> {
  return Uint16Array.from(values, numberToFp16Bits);
}

const FLOAT_BUFFER = new ArrayBuffer(4);
const FLOAT_F32 = new Float32Array(FLOAT_BUFFER);
const FLOAT_U32 = new Uint32Array(FLOAT_BUFFER);

function numberToBf16Bits(value: number): number {
  FLOAT_F32[0] = value;
  const bits = FLOAT_U32[0]!;
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
}

function bf16BitsToNumber(bits: number): number {
  FLOAT_U32[0] = bits << 16;
  return FLOAT_F32[0]!;
}

function numberToFp16Bits(value: number): number {
  FLOAT_F32[0] = value;
  const bits = FLOAT_U32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated +
      (remainder > halfway || (remainder === halfway && (truncated & 1)) ? 1 : 0));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1))) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) {
      FLOAT_U32[0] = sign;
      return FLOAT_F32[0]!;
    }
    exponent = 1;
    while ((mantissa & 0x400) === 0) {
      mantissa <<= 1;
      exponent -= 1;
    }
    mantissa &= 0x3ff;
  } else if (exponent === 0x1f) {
    FLOAT_U32[0] = sign | 0x7f80_0000 | (mantissa << 13);
    return FLOAT_F32[0]!;
  }
  FLOAT_U32[0] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return FLOAT_F32[0]!;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function requireResultNode(): HTMLElement {
  const node = document.querySelector<HTMLElement>("#result");
  if (node === null) throw new Error("Missing result node");
  return node;
}

function finish(status: "passed" | "failed", detail: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = `${status}: ${detail}`;
}
