import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  AceCorrectnessFsqDecodeKernel,
} from "../../src/webgpu/kernels/fsq-decode.js";
import oracle from "../fsq-codebook-vectors.json";

interface CaseResult {
  readonly profile: AceModelProfileId;
  readonly operation: "valid" | "invalid";
  readonly status: number;
  readonly actual: readonly number[];
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
  const kernel = AceCorrectnessFsqDecodeKernel.create(device, profile);
  try {
    const validCodes = new Uint32Array(
      oracle.vectors.map((vector) => vector.code),
    );
    const valid = await execute(device, kernel, profile, validCodes);
    const expected = profile === "reference-bf16"
      ? oracle.vectors.flatMap((vector) => vector.bfloat16Values)
      : oracle.vectors.flatMap((vector) =>
        vector.fp16BitsFromBfloat16.map((bits) =>
          fp16BitsToNumber(Number.parseInt(bits, 16))
        )
      );
    for (let index = 0; index < expected.length; index += 1) {
      const expectedValue = expected[index]!;
      if (!Number.isFinite(valid.actual[index])) {
        throw new Error(`${profile} FSQ produced a non-finite value at ${index}`);
      }
      if (valid.actual[index] !== expectedValue) {
        throw new Error(
          `${profile} FSQ mismatch at ${index}: ${valid.actual[index]} != ${expectedValue}`,
        );
      }
    }
    if (valid.status !== 0) throw new Error(`${profile} valid status was nonzero`);

    const invalid = await execute(
      device,
      kernel,
      profile,
      new Uint32Array([64_000]),
    );
    if (invalid.status !== 1 || invalid.actual.some((value) => value !== 0)) {
      throw new Error(`${profile} invalid FSQ code did not fail closed`);
    }
    return [
      { profile, operation: "valid", status: valid.status, actual: [...valid.actual] },
      { profile, operation: "invalid", status: invalid.status, actual: [...invalid.actual] },
    ];
  } finally {
    kernel.destroy();
  }
}

async function execute(
  device: GPUDevice,
  kernel: AceCorrectnessFsqDecodeKernel,
  profile: AceModelProfileId,
  codes: Uint32Array<ArrayBuffer>,
): Promise<{ readonly actual: Float32Array; readonly status: number }> {
  const outputElements = codes.length * 6;
  const outputBytes = outputElements * (profile === "raw-fp16" ? 2 : 4);
  const codeBuffer = uploadBuffer(device, codes);
  const output = device.createBuffer({
    size: alignedSize(outputBytes),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const status = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(status, 0, new Uint32Array([0]));
  const outputRead = device.createBuffer({
    size: alignedSize(outputBytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const statusRead = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const dispatch = await kernel.createDispatch(
      `browser-${profile}-fsq`,
      { codeCount: codes.length },
      {
        codeIds: binding(codeBuffer),
        output: binding(output),
        validationStatus: binding(status),
      },
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, outputRead, 0, alignedSize(outputBytes));
    encoder.copyBufferToBuffer(status, 0, statusRead, 0, 4);
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      outputRead.mapAsync(GPUMapMode.READ),
      statusRead.mapAsync(GPUMapMode.READ),
    ]);
    const outputCopy = outputRead.getMappedRange().slice(0);
    const statusCopy = statusRead.getMappedRange().slice(0);
    const actual = profile === "raw-fp16"
      ? Float32Array.from(
        new Uint16Array(outputCopy, 0, outputElements),
        fp16BitsToNumber,
      )
      : new Float32Array(outputCopy, 0, outputElements);
    return { actual: Float32Array.from(actual), status: new Uint32Array(statusCopy)[0]! };
  } finally {
    codeBuffer.destroy();
    output.destroy();
    status.destroy();
    outputRead.destroy();
    statusRead.destroy();
  }
}

function uploadBuffer(
  device: GPUDevice,
  values: Uint32Array<ArrayBuffer>,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignedSize(values.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (mantissa / 1024);
  if (exponent === 31) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

function finish(status: "passed" | "failed", text: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = text;
}

function requireResultNode(): HTMLElement {
  const node = document.querySelector<HTMLElement>("#result");
  if (node === null) throw new Error("Missing result node");
  return node;
}
