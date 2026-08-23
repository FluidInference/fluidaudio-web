import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  ACE_CONTEXT_CHANNELS,
  ACE_SILENCE_SOURCE_CHANNELS,
  ACE_SILENCE_SOURCE_FRAMES,
  AceCorrectnessConditionLayoutKernel,
} from "../../src/webgpu/kernels/condition-layout.js";
import {
  AceCorrectnessDetokenizerExpandKernel,
  aceCorrectnessDetokenizerExpandWgsl,
} from "../../src/webgpu/kernels/detokenizer-expand.js";

interface CaseResult {
  readonly profile: AceModelProfileId;
  readonly operation: string;
  readonly valuesChecked: number;
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) => finish(
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
  const expand = AceCorrectnessDetokenizerExpandKernel.create(device, profile);
  const layout = AceCorrectnessConditionLayoutKernel.create(device, profile);
  try {
    return [
      await runDetokenizerExpand(device, profile, expand),
      await runConditionLayout(device, profile, layout),
    ];
  } finally {
    expand.destroy();
    layout.destroy();
  }
}

async function runDetokenizerExpand(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessDetokenizerExpandKernel,
): Promise<CaseResult> {
  const width = 4;
  await requireShaderValid(
    device,
    `browser-${profile}-detokenizer-expand-preflight`,
    aceCorrectnessDetokenizerExpandWgsl(profile, { codeCount: 2, width }),
  );
  const embedded = [1, -2, 0.5, 4, -1, 3, 2, -0.5];
  const specials = [
    0, 0.25, -0.5, 1,
    1, -1, 2, 0.5,
    -2, 0.5, 1, -0.25,
    0.5, 2, -1, 0,
    -0.25, -0.5, 0.25, 2,
  ];
  const roundedEmbedded = embedded.map((value) => profileValue(profile, value));
  const roundedSpecials = specials.map((value) => weightValue(profile, value));
  const expected = new Float32Array(2 * 5 * width);
  for (let code = 0; code < 2; code += 1) {
    for (let patch = 0; patch < 5; patch += 1) {
      for (let column = 0; column < width; column += 1) {
        expected[(code * 5 + patch) * width + column] = profileAdd(
          profile,
          roundedEmbedded[code * width + column]!,
          roundedSpecials[patch * width + column]!,
        );
      }
    }
  }
  const input = activationBuffer(device, profile, embedded);
  const special = weightBuffer(device, profile, specials);
  const output = activationOutput(device, profile, expected.length);
  try {
    const dispatch = await kernel.createDispatch(
      `browser-${profile}-detokenizer-expand`,
      { codeCount: 2, width },
      {
        embeddedCodes: binding(input),
        specialTokens: binding(special),
        output: binding(output),
      },
    );
    const actual = await executeAndReadActivation(
      device,
      [dispatch],
      output,
      expected.length,
      profile,
    );
    assertEqual(actual, expected);
    return { profile, operation: "detokenizer-expand", valuesChecked: actual.length };
  } finally {
    input.destroy();
    special.destroy();
    output.destroy();
  }
}

async function requireShaderValid(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<void> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`
    ).join("\n"));
  }
}

async function runConditionLayout(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessConditionLayoutKernel,
): Promise<CaseResult> {
  const frames = 2;
  const batch = 2;
  const source = new Float32Array(
    ACE_SILENCE_SOURCE_CHANNELS * ACE_SILENCE_SOURCE_FRAMES,
  );
  for (let channel = 0; channel < ACE_SILENCE_SOURCE_CHANNELS; channel += 1) {
    source[channel * ACE_SILENCE_SOURCE_FRAMES] = channel / 4;
    source[channel * ACE_SILENCE_SOURCE_FRAMES + 1] = -channel / 8;
  }
  const expectedLatents = new Float32Array(batch * frames * 64);
  for (let row = 0; row < batch * frames; row += 1) {
    const frame = row % frames;
    for (let channel = 0; channel < 64; channel += 1) {
      expectedLatents[row * 64 + channel] = profileValue(
        profile,
        source[channel * ACE_SILENCE_SOURCE_FRAMES + frame]!,
      );
    }
  }
  const maskValues = new Uint32Array([1, 0, 0, 1]);
  const expectedContext = new Float32Array(batch * frames * ACE_CONTEXT_CHANNELS);
  for (let row = 0; row < batch * frames; row += 1) {
    expectedContext.set(expectedLatents.subarray(row * 64, row * 64 + 64), row * 128);
    expectedContext.fill(maskValues[row]!, row * 128 + 64, row * 128 + 128);
  }

  const sourceBuffer = storageBuffer(device, source);
  const latentBuffer = activationOutput(device, profile, expectedLatents.length);
  const maskBuffer = storageBuffer(device, maskValues);
  const contextBuffer = activationOutput(device, profile, expectedContext.length);
  try {
    const silence = await kernel.createSilenceExpandDispatch(
      `browser-${profile}-silence-expand`,
      { batch, frames },
      { source: binding(sourceBuffer), output: binding(latentBuffer) },
    );
    const context = await kernel.createDirectContextDispatch(
      `browser-${profile}-direct-context`,
      { batch, frames },
      {
        sourceLatents: binding(latentBuffer),
        chunkMask: binding(maskBuffer),
        output: binding(contextBuffer),
      },
    );
    const actual = await executeAndReadActivation(
      device,
      [silence, context],
      contextBuffer,
      expectedContext.length,
      profile,
    );
    assertEqual(actual, expectedContext);
    return { profile, operation: "silence-and-context-layout", valuesChecked: actual.length };
  } finally {
    sourceBuffer.destroy();
    latentBuffer.destroy();
    maskBuffer.destroy();
    contextBuffer.destroy();
  }
}

function activationBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "reference-bf16"
      ? Float32Array.from(values)
      : Uint16Array.from(values, numberToFp16Bits),
  );
}

function weightBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "reference-bf16"
      ? packBf16(values)
      : Uint16Array.from(values, numberToFp16Bits),
  );
}

function activationOutput(
  device: GPUDevice,
  profile: AceModelProfileId,
  elements: number,
): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * (profile === "raw-fp16" ? 2 : 4)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBufferView<ArrayBufferLike>,
): GPUBuffer {
  const bytes = new Uint8Array(alignedSize(data.byteLength));
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function executeAndReadActivation(
  device: GPUDevice,
  dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[],
  output: GPUBuffer,
  elements: number,
  profile: AceModelProfileId,
): Promise<Float32Array> {
  const byteLength = elements * (profile === "raw-fp16" ? 2 : 4);
  const readback = device.createBuffer({
    size: alignedSize(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches) dispatch.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, alignedSize(byteLength));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0, byteLength);
    return profile === "reference-bf16"
      ? Float32Array.from(new Float32Array(bytes, 0, elements))
      : Float32Array.from(new Uint16Array(bytes, 0, elements), fp16BitsToNumber);
  } finally {
    readback.destroy();
  }
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function profileValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16" ? roundFp16(value) : Math.fround(value);
}

function weightValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16" ? roundFp16(value) : roundBf16(value);
}

function profileAdd(
  profile: AceModelProfileId,
  left: number,
  right: number,
): number {
  return profile === "raw-fp16"
    ? roundFp16(left + right)
    : Math.fround(left + right);
}

function packBf16(values: readonly number[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index += 1) {
    const bits = numberToBf16Bits(values[index]!);
    words[index >> 1] = words[index >> 1]! | (bits << ((index & 1) * 16));
  }
  return words;
}

function roundBf16(value: number): number {
  const bits = numberToBf16Bits(value);
  const u32 = new Uint32Array([bits << 16]);
  return new Float32Array(u32.buffer)[0]!;
}

function numberToBf16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
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

function assertEqual(actual: Float32Array, expected: Float32Array): void {
  if (actual.length !== expected.length) throw new Error("GPU output length mismatch");
  for (let index = 0; index < actual.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(
        `GPU mismatch at ${index}: ${actual[index]} != ${expected[index]}`,
      );
    }
  }
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
