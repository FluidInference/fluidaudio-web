import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  type AceVaeDecoderConfig,
  type AceVaeDecoderOperation,
  type AceVaeDecoderSlot,
} from "../../src/webgpu/vae-decoder.js";
import { planAceVaeChunkedDecode } from "../../src/webgpu/vae-chunks.js";
import {
  AceVaeChunkGpuBackend,
  planAceVaeChunkGpuBackendMemory,
  type AceVaeChunkGpuBackendProgress,
} from "../../src/webgpu/vae-backend.js";
import {
  AceCorrectnessVaePrimitiveKernel,
  aceCorrectnessVaeAddWgsl,
  aceCorrectnessVaeConv1dWgsl,
  aceCorrectnessVaeConvTranspose1dWgsl,
  aceCorrectnessVaeSnakeWgsl,
  type AceVaeConv1dShape,
  type AceVaeConvTranspose1dShape,
} from "../../src/webgpu/kernels/vae-primitives.js";

interface CaseResult {
  readonly operation: string;
  readonly valuesChecked: number;
  readonly maximumAbsoluteError: number;
  readonly bitMismatchCount?: number;
  readonly tiledOperationLabels?: readonly string[];
  readonly channelChunkedOperationLabels?: readonly string[];
  readonly portableOperationLabels?: readonly string[];
  readonly tiledQuantumCount?: number;
  readonly channelChunkedQuantumCount?: number;
  readonly portableQuantumCount?: number;
  readonly fallbackReasons?: Readonly<Record<string, number>>;
  readonly productionBatchBitMismatchCount?: number;
  readonly logicalQuantumCount?: number;
  readonly physicalDispatchCount?: number;
  readonly optimizedBatch1CommandBufferCount?: number;
  readonly optimizedBatch8CommandBufferCount?: number;
  readonly optimizedBatch1QueueDrainCount?: number;
  readonly optimizedBatch8QueueDrainCount?: number;
  readonly optimizedBatch1IdleCount?: number;
  readonly optimizedBatch8IdleCount?: number;
  readonly optimizedBatch1ProgressEventCount?: number;
  readonly optimizedBatch8ProgressEventCount?: number;
}

interface IntegratedBackendRun {
  readonly output: Float32Array;
  readonly tiledOperationLabels: readonly string[];
  readonly channelChunkedOperationLabels: readonly string[];
  readonly portableOperationLabels: readonly string[];
  readonly tiledQuantumCount: number;
  readonly channelChunkedQuantumCount: number;
  readonly portableQuantumCount: number;
  readonly fallbackReasons: Readonly<Record<string, number>>;
  readonly logicalQuantumCount: number;
  readonly physicalDispatchCount: number;
  readonly commandBufferCount: number;
  readonly queueDrainCount: number;
  readonly idleCount: number;
  readonly progress: readonly AceVaeChunkGpuBackendProgress[];
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
  const productionBindingBytes = 256 * 1024 * 1024;
  if (
    adapter.limits.maxBufferSize < productionBindingBytes ||
    adapter.limits.maxStorageBufferBindingSize < productionBindingBytes
  ) {
    throw new Error(
      "adapter cannot satisfy the canonical VAE 256 MiB device contract; a smaller chunk needs a separate oracle",
    );
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: productionBindingBytes,
      maxStorageBufferBindingSize: productionBindingBytes,
      maxComputeWorkgroupStorageSize: 16 * 1024,
    },
  });
  try {
    const production = planAceVaeChunkedDecode(750);
    if (
      production.windows.length !== 6 ||
      production.strideFrames !== 128 ||
      production.outputAudioFrames !== 1_440_000 ||
      production.decoderWorkspacePlan.workspaceBytes !== 251_658_240
    ) {
      throw new Error("production 256/64 VAE chunk contract changed");
    }
    if (
      adapter.limits.maxBufferSize < production.decoderWorkspacePlan.workspaceBytes ||
      adapter.limits.maxStorageBufferBindingSize <
        production.decoderWorkspacePlan.workspaceBytes
    ) {
      throw new Error(
        "adapter cannot bind the canonical 240 MiB VAE workspace; a smaller chunk needs a separate oracle",
      );
    }
    if (
      device.limits.maxBufferSize < productionBindingBytes ||
      device.limits.maxStorageBufferBindingSize < productionBindingBytes
    ) {
      throw new Error("requested 256 MiB VAE limits were not enabled on the device");
    }
    await preflightShaders(device);
    const primitive = AceCorrectnessVaePrimitiveKernel.create(device);
    try {
      return [
        await runNontrivialConv(device, primitive),
        await runNontrivialTranspose(device, primitive),
        await runNontrivialSnakeAndAdd(device, primitive),
        await runCompleteToyDecoder(device),
        await runIntegratedOptimizedDecoderAb(device),
      ];
    } finally {
      primitive.destroy();
    }
  } finally {
    device.destroy();
  }
}

async function runIntegratedOptimizedDecoderAb(
  device: GPUDevice,
): Promise<CaseResult> {
  const config: AceVaeDecoderConfig = Object.freeze({
    id: "browser-opt-0005-oobleck",
    decoderInputChannels: 136,
    decoderChannels: 128,
    audioChannels: 2,
    channelMultiples: Object.freeze([1]),
    downsamplingRatios: Object.freeze([2]),
    sampleRateHz: 48_000,
  });
  const inputFrames = 3;
  const plan = planAceVaeDecoder(inputFrames, config);
  const tensorValues = createToyTensorValues(plan.operations);
  const inputValues = Float32Array.from(
    { length: inputFrames * config.decoderInputChannels },
    (_, index) => Math.fround(((index * 13) % 31 - 15) * 0.025),
  );
  const expected = cpuDecoder(plan.operations, tensorValues, inputValues);
  const chunkPlan = planAceVaeChunkedDecode(inputFrames, {
    chunkFrames: inputFrames,
    overlapFrames: 0,
    config,
  });
  const quantumWorkPolicy = Object.freeze({
    ...ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
    maximumOutputElements: 128,
  });
  const transposeParts = Object.freeze({
    "vae.decoder.block.0.conv_t1.weight": Object.freeze([
      Object.freeze({ partStart: 0, partEnd: 64 }),
      Object.freeze({ partStart: 64, partEnd: 128 }),
    ]),
  });
  const residentWeightBytes = [...tensorValues.values()].reduce(
    (bytes, values) => bytes + values.byteLength,
    0,
  );
  const runProfile = async (
    profile: "portable" | "optimized-when-eligible",
    batchSize: 1 | 8,
  ): Promise<IntegratedBackendRun> => {
    const owned: GPUBuffer[] = [];
    const own = (buffer: GPUBuffer): GPUBuffer => {
      owned.push(buffer);
      return buffer;
    };
    let runtime: AceCorrectnessVaeDecoderRuntime | undefined;
    let backend: AceVaeChunkGpuBackend | undefined;
    try {
      const input = own(storageBuffer(device, inputValues));
      const output = own(outputBuffer(device, plan.outputElements));
      const poisonedOutput = new Uint32Array(plan.outputElements);
      poisonedOutput.fill(0x7fc0_0000);
      device.queue.writeBuffer(output, 0, poisonedOutput);
      const workspaces = [
        own(outputBuffer(device, plan.maximumActivationElements)),
        own(outputBuffer(device, plan.maximumActivationElements)),
        own(outputBuffer(device, plan.maximumActivationElements)),
      ] as const;
      const tensors = Object.fromEntries(
        [...tensorValues].map(([name, values]) => {
          const operation = plan.operations.find((candidate) =>
            candidate.kind === "conv-transpose1d" && candidate.weight === name);
          if (operation?.kind !== "conv-transpose1d") {
            return [name, binding(own(storageBuffer(device, values)))];
          }
          const rowElements = operation.shape.kernelSize *
            operation.shape.inputChannels;
          const splitOutputChannel = Math.floor(
            operation.shape.outputChannels / 2,
          );
          const splitElement = splitOutputChannel * rowElements;
          return [name, [
            {
              binding: binding(own(storageBuffer(
                device,
                values.subarray(0, splitElement),
              ))),
              partStart: 0,
              partEnd: splitOutputChannel,
            },
            {
              binding: binding(own(storageBuffer(
                device,
                values.subarray(splitElement),
              ))),
              partStart: splitOutputChannel,
              partEnd: operation.shape.outputChannels,
            },
          ]];
        }),
      );
      runtime = AceCorrectnessVaeDecoderRuntime.create(device);
      const dispatch = await runtime.createDecoderDispatch(
        `browser-opt-0006-${profile}-batch-${batchSize}`,
        inputFrames,
        {
          input: binding(input),
          output: binding(output),
          workspaces: [
            binding(workspaces[0]),
            binding(workspaces[1]),
            binding(workspaces[2]),
          ],
          tensors,
        },
        config,
        1,
        {
          quantumWorkPolicy,
          conv1dProfile: profile,
        },
      );
      if (dispatch.quanta.length !== 109 || dispatch.primitiveCount !== 115) {
        throw new Error(
          `integrated backend dispatch changed: ${dispatch.quanta.length} logical quanta / ${dispatch.primitiveCount} physical dispatches`,
        );
      }
      const physicalDispatchCount = dispatch.quanta.reduce(
        (total, quantum) => total + quantum.primitiveCount,
        0,
      );
      if (physicalDispatchCount !== 115) {
        throw new Error(
          `integrated backend quantum accounting changed: ${physicalDispatchCount} physical dispatches`,
        );
      }
      const readback = own(device.createBuffer({
        size: alignedSize(chunkPlan.maximumDecodedFloat32Bytes),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }));
      const memory = planAceVaeChunkGpuBackendMemory(
        chunkPlan,
        residentWeightBytes,
        device.limits.minUniformBufferOffsetAlignment,
        transposeParts,
        quantumWorkPolicy,
        batchSize,
      );
      const progress: AceVaeChunkGpuBackendProgress[] = [];
      let resourcesDestroyed = false;
      backend = AceVaeChunkGpuBackend.fromPreparedResources({
        device,
        plan: chunkPlan,
        finalLatents: inputValues,
        input: binding(input),
        output: binding(output),
        readback,
        decoderDispatches: new Map([[inputFrames, dispatch]]),
        memory,
        decoderQuantaPerCommandBuffer: batchSize,
        onProgress: (event) => progress.push(Object.freeze({ ...event })),
        destroy(): void {
          if (resourcesDestroyed) return;
          resourcesDestroyed = true;
          runtime?.destroy();
          readback.destroy();
        },
      });
      const actual = await backend.decodeWindow(chunkPlan.windows[0]!);
      const actualBits = new Uint32Array(
        actual.buffer,
        actual.byteOffset,
        actual.length,
      );
      if (actualBits.some((bits) => bits === 0x7fc0_0000)) {
        throw new Error(
          `integrated backend batch ${batchSize} left qNaN-prefilled output unwritten`,
        );
      }
      await backend.destroy();
      backend = undefined;
      assertIntegratedBackendProgress(progress, batchSize);
      const expectedDecoderCommandBuffers = Math.ceil(109 / batchSize);
      const expectedCommandBuffers = expectedDecoderCommandBuffers + 1;
      const idleCount = progress.at(-1)!.cooperativeIdleMs;
      if (
        idleCount !== expectedDecoderCommandBuffers ||
        memory.maximumDecoderQuantaPerWindow !== 109 ||
        memory.maximumCommandBuffersPerWindow !== expectedCommandBuffers
      ) {
        throw new Error(
          `integrated backend batch ${batchSize} accounting changed: ${JSON.stringify({
            idleCount,
            maximumDecoderQuantaPerWindow:
              memory.maximumDecoderQuantaPerWindow,
            maximumCommandBuffersPerWindow: memory.maximumCommandBuffersPerWindow,
          })}`,
        );
      }
      const selection = dispatch.conv1dSelection;
      if (selection === undefined) {
        throw new Error("integrated decoder omitted Conv1D selection telemetry");
      }
      return Object.freeze({
        output: actual,
        tiledOperationLabels: selection.tiledOperationLabels,
        channelChunkedOperationLabels:
          selection.channelChunkedOperationLabels,
        portableOperationLabels: selection.portableOperationLabels,
        tiledQuantumCount: selection.tiledQuantumCount,
        channelChunkedQuantumCount: selection.channelChunkedQuantumCount,
        portableQuantumCount: selection.portableQuantumCount,
        fallbackReasons: selection.fallbackReasons,
        logicalQuantumCount: dispatch.quanta.length,
        physicalDispatchCount,
        commandBufferCount: expectedCommandBuffers,
        queueDrainCount: expectedCommandBuffers,
        idleCount,
        progress: Object.freeze(progress),
      });
    } finally {
      await backend?.destroy();
      runtime?.destroy();
      for (const buffer of owned) buffer.destroy();
    }
  };
  const portable = await runProfile("portable", 1);
  const optimized = await runProfile("optimized-when-eligible", 1);
  const optimizedProduction = await runProfile("optimized-when-eligible", 8);
  const portableBits = new Uint32Array(
    portable.output.buffer,
    portable.output.byteOffset,
    portable.output.length,
  );
  const optimizedBits = new Uint32Array(
    optimized.output.buffer,
    optimized.output.byteOffset,
    optimized.output.length,
  );
  let bitMismatchCount = 0;
  for (let index = 0; index < portableBits.length; index += 1) {
    if (portableBits[index] !== optimizedBits[index]) bitMismatchCount += 1;
  }
  if (bitMismatchCount !== 0) {
    throw new Error(
      `portable batch-1 and optimized batch-1 differ in ${bitMismatchCount} FP32 values`,
    );
  }
  const optimizedProductionBits = new Uint32Array(
    optimizedProduction.output.buffer,
    optimizedProduction.output.byteOffset,
    optimizedProduction.output.length,
  );
  let productionBatchBitMismatchCount = 0;
  for (let index = 0; index < optimizedBits.length; index += 1) {
    if (optimizedBits[index] !== optimizedProductionBits[index]) {
      productionBatchBitMismatchCount += 1;
    }
  }
  if (productionBatchBitMismatchCount !== 0) {
    throw new Error(
      `optimized batch-1 and production batch-8 differ in ${productionBatchBitMismatchCount} FP32 values`,
    );
  }
  if (
    optimized.tiledOperationLabels.join("|") !== [
      "block-0-res-1-conv1",
      "conv2",
    ].join("|") ||
    optimized.channelChunkedOperationLabels.join("|") !== [
      "conv1",
      "block-0-res-2-conv1",
      "block-0-res-3-conv1",
    ].join("|") ||
    optimized.portableOperationLabels.join("|") !== [
      "block-0-res-1-conv2",
      "block-0-res-2-conv2",
      "block-0-res-3-conv2",
    ].join("|") ||
    optimized.tiledQuantumCount !== 7 ||
    optimized.channelChunkedQuantumCount !== 15 ||
    optimized.portableQuantumCount !== 18 ||
    optimizedProduction.tiledOperationLabels.join("|") !==
      optimized.tiledOperationLabels.join("|") ||
    optimizedProduction.channelChunkedOperationLabels.join("|") !==
      optimized.channelChunkedOperationLabels.join("|") ||
    optimizedProduction.portableOperationLabels.join("|") !==
      optimized.portableOperationLabels.join("|") ||
    optimizedProduction.tiledQuantumCount !== 7 ||
    optimizedProduction.channelChunkedQuantumCount !== 15 ||
    optimizedProduction.portableQuantumCount !== 18 ||
    portable.tiledQuantumCount !== 0 ||
    portable.channelChunkedQuantumCount !== 0 ||
    portable.portableQuantumCount !== 40
  ) {
    throw new Error(
      `OPT-0005 integrated selector did not exercise the exact tiled, channel-chunked, and portable families: ${JSON.stringify({
        optimized: {
          tiledOperationLabels: optimized.tiledOperationLabels,
          channelChunkedOperationLabels: optimized.channelChunkedOperationLabels,
          portableOperationLabels: optimized.portableOperationLabels,
          tiledQuantumCount: optimized.tiledQuantumCount,
          channelChunkedQuantumCount: optimized.channelChunkedQuantumCount,
          portableQuantumCount: optimized.portableQuantumCount,
        },
        portable: {
          tiledQuantumCount: portable.tiledQuantumCount,
          channelChunkedQuantumCount: portable.channelChunkedQuantumCount,
          portableQuantumCount: portable.portableQuantumCount,
        },
      })}`,
    );
  }
  if (
    optimized.fallbackReasons[
      "tiled:unsupported-math;channel-chunked:unsupported-math"
    ] !== 18 ||
    Object.keys(optimized.fallbackReasons).length !== 1 ||
    optimizedProduction.fallbackReasons[
      "tiled:unsupported-math;channel-chunked:unsupported-math"
    ] !== 18 ||
    Object.keys(optimizedProduction.fallbackReasons).length !== 1 ||
    portable.fallbackReasons["profile-portable"] !== 40 ||
    Object.keys(portable.fallbackReasons).length !== 1
  ) {
    throw new Error("OPT-0005 integrated fallback telemetry changed");
  }
  if (
    portable.logicalQuantumCount !== 109 ||
    portable.physicalDispatchCount !== 115 ||
    optimized.logicalQuantumCount !== 109 ||
    optimized.physicalDispatchCount !== 115 ||
    optimizedProduction.logicalQuantumCount !== 109 ||
    optimizedProduction.physicalDispatchCount !== 115 ||
    optimized.commandBufferCount !== 110 ||
    optimized.queueDrainCount !== 110 ||
    optimized.idleCount !== 109 ||
    optimized.progress.length !== 110 ||
    optimizedProduction.commandBufferCount !== 15 ||
    optimizedProduction.queueDrainCount !== 15 ||
    optimizedProduction.idleCount !== 14 ||
    optimizedProduction.progress.length !== 110
  ) {
    throw new Error("integrated OPT-0006 backend accounting changed");
  }
  const compared = compare(
    "complete-opt-0005-portable-vs-optimized-decoder",
    optimized.output,
    expected,
    2e-4,
  );
  return {
    ...compared,
    bitMismatchCount,
    productionBatchBitMismatchCount,
    tiledOperationLabels: optimized.tiledOperationLabels,
    channelChunkedOperationLabels: optimized.channelChunkedOperationLabels,
    portableOperationLabels: optimized.portableOperationLabels,
    tiledQuantumCount: optimized.tiledQuantumCount,
    channelChunkedQuantumCount: optimized.channelChunkedQuantumCount,
    portableQuantumCount: optimized.portableQuantumCount,
    fallbackReasons: optimized.fallbackReasons,
    logicalQuantumCount: optimizedProduction.logicalQuantumCount,
    physicalDispatchCount: optimizedProduction.physicalDispatchCount,
    optimizedBatch1CommandBufferCount: optimized.commandBufferCount,
    optimizedBatch8CommandBufferCount: optimizedProduction.commandBufferCount,
    optimizedBatch1QueueDrainCount: optimized.queueDrainCount,
    optimizedBatch8QueueDrainCount: optimizedProduction.queueDrainCount,
    optimizedBatch1IdleCount: optimized.idleCount,
    optimizedBatch8IdleCount: optimizedProduction.idleCount,
    optimizedBatch1ProgressEventCount: optimized.progress.length,
    optimizedBatch8ProgressEventCount: optimizedProduction.progress.length,
  };
}

function assertIntegratedBackendProgress(
  progress: readonly AceVaeChunkGpuBackendProgress[],
  batchSize: 1 | 8,
): void {
  const decoderQuanta = 109;
  const decoderCommandBuffers = Math.ceil(decoderQuanta / batchSize);
  const totalCommandBuffers = decoderCommandBuffers + 1;
  if (progress.length !== decoderQuanta + 1) {
    throw new Error(
      `integrated backend batch ${batchSize} emitted ${progress.length} progress events; expected 110`,
    );
  }
  for (let index = 0; index < decoderQuanta; index += 1) {
    const event = progress[index]!;
    const completedBatch = Math.floor(index / batchSize) + 1;
    if (
      event.windowIndex !== 0 ||
      event.stage !== "decoder" ||
      event.completedDecoderQuanta !== index + 1 ||
      event.totalDecoderQuanta !== decoderQuanta ||
      event.completedCommandBuffers !== completedBatch ||
      event.totalCommandBuffers !== totalCommandBuffers ||
      event.queueDrains !== completedBatch ||
      event.cooperativeIdleMs !== completedBatch
    ) {
      throw new Error(
        `integrated backend batch ${batchSize} decoder progress ${index} changed: ${JSON.stringify(event)}`,
      );
    }
  }
  const readback = progress[decoderQuanta]!;
  if (
    readback.windowIndex !== 0 ||
    readback.stage !== "readback" ||
    readback.completedDecoderQuanta !== decoderQuanta ||
    readback.totalDecoderQuanta !== decoderQuanta ||
    readback.completedCommandBuffers !== totalCommandBuffers ||
    readback.totalCommandBuffers !== totalCommandBuffers ||
    readback.queueDrains !== totalCommandBuffers ||
    readback.cooperativeIdleMs !== decoderCommandBuffers
  ) {
    throw new Error(
      `integrated backend batch ${batchSize} readback progress changed: ${JSON.stringify(readback)}`,
    );
  }
}

async function preflightShaders(device: GPUDevice): Promise<void> {
  const conv: AceVaeConv1dShape = {
    batch: 1,
    inputFrames: 4,
    inputChannels: 2,
    outputChannels: 2,
    kernelSize: 3,
    stride: 1,
    dilation: 2,
    padding: 2,
  };
  const transpose: AceVaeConvTranspose1dShape = {
    batch: 1,
    inputFrames: 3,
    inputChannels: 2,
    outputChannels: 2,
    kernelSize: 4,
    stride: 2,
    dilation: 1,
    padding: 1,
    outputPadding: 0,
  };
  const sources = [
    ["conv", aceCorrectnessVaeConv1dWgsl(conv, true)],
    ["transpose", aceCorrectnessVaeConvTranspose1dWgsl(transpose, true)],
    ["snake", aceCorrectnessVaeSnakeWgsl({ batch: 1, frames: 4, channels: 2 })],
    ["add", aceCorrectnessVaeAddWgsl({ batch: 1, frames: 4, channels: 2 })],
  ] as const;
  for (const [label, code] of sources) {
    const module = device.createShaderModule({
      label: `browser-vae-${label}-preflight`,
      code,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(errors.map((message) =>
        `${label} ${message.lineNum}:${message.linePos} ${message.message}`
      ).join("\n"));
    }
  }
}

async function runNontrivialConv(
  device: GPUDevice,
  kernel: AceCorrectnessVaePrimitiveKernel,
): Promise<CaseResult> {
  const shape: AceVaeConv1dShape = {
    batch: 1,
    inputFrames: 4,
    inputChannels: 2,
    outputChannels: 2,
    kernelSize: 3,
    stride: 1,
    dilation: 2,
    padding: 2,
  };
  const inputValues = Float32Array.from([
    0.25, -0.75, 1.5, 0.2, -0.4, 0.9, 0.6, -1.1,
  ]);
  const weights = Float32Array.from([
    0.2, -0.1, 0.4, 0.3, -0.25, 0.5,
    -0.3, 0.15, 0.2, -0.45, 0.1, 0.35,
  ]);
  const bias = Float32Array.from([0.07, -0.03]);
  const expected = cpuConv1d(inputValues, weights, bias, shape);
  const input = storageBuffer(device, inputValues);
  const weight = storageBuffer(device, weights);
  const biasBuffer = storageBuffer(device, bias);
  const output = outputBuffer(device, expected.length);
  const rangeControl = device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  try {
    await expectRejected(
      () => kernel.createConv1dDispatch(
        "browser-vae-invalid-range-domain",
        shape,
        {
          input: binding(input),
          weight: binding(weight),
          bias: binding(biasBuffer),
          output: binding(output),
        },
        {
          base: expected.length - 1,
          count: 2,
          control: { buffer: rangeControl, offset: 0, size: 16 },
        },
      ),
      /exceeds its complete output domain/,
    );
    await expectRejected(
      () => kernel.createConv1dDispatch(
        "browser-vae-invalid-range-alignment",
        shape,
        {
          input: binding(input),
          weight: binding(weight),
          bias: binding(biasBuffer),
          output: binding(output),
        },
        {
          base: 0,
          count: 1,
          control: { buffer: rangeControl, offset: 4, size: 16 },
        },
      ),
      /uniform-buffer alignment/,
    );
    await expectRejected(
      () => kernel.createConv1dDispatch(
        "browser-vae-invalid-range-control-size",
        shape,
        {
          input: binding(input),
          weight: binding(weight),
          bias: binding(biasBuffer),
          output: binding(output),
        },
        {
          base: 0,
          count: 1,
          control: { buffer: rangeControl, offset: 0, size: 8 },
        },
      ),
      /does not expose 16 bytes/,
    );
    await expectRejected(
      () => kernel.createConv1dDispatch(
        "browser-vae-overlapping-range-control",
        shape,
        {
          input: binding(input),
          weight: binding(weight),
          bias: binding(biasBuffer),
          output: binding(output),
        },
        {
          base: 0,
          count: 1,
          control: { buffer: output, offset: 0, size: 16 },
        },
      ),
      /range control output must not overlap/,
    );
    const dispatch = await kernel.createConv1dDispatch(
      "browser-vae-nontrivial-conv",
      shape,
      {
        input: binding(input),
        weight: binding(weight),
        bias: binding(biasBuffer),
        output: binding(output),
      },
    );
    const actual = await executeAndRead(device, [dispatch], output, expected.length);
    return compare("dilated-conv1d", actual, expected, 2e-6);
  } finally {
    input.destroy();
    weight.destroy();
    biasBuffer.destroy();
    output.destroy();
    rangeControl.destroy();
  }
}

async function expectRejected(
  action: () => Promise<unknown>,
  expected: RegExp,
): Promise<void> {
  try {
    await action();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (expected.test(message)) return;
    throw new Error(`unexpected rejection: ${message}`);
  }
  throw new Error(`expected rejection matching ${expected}`);
}

async function runNontrivialTranspose(
  device: GPUDevice,
  kernel: AceCorrectnessVaePrimitiveKernel,
): Promise<CaseResult> {
  const shape: AceVaeConvTranspose1dShape = {
    batch: 1,
    inputFrames: 3,
    inputChannels: 2,
    outputChannels: 2,
    kernelSize: 4,
    stride: 2,
    dilation: 1,
    padding: 1,
    outputPadding: 0,
  };
  const inputValues = Float32Array.from([0.5, -1, 1.25, 0.3, -0.2, 0.8]);
  const weights = Float32Array.from([
    0.1, 0.2, -0.3, 0.4, 0.25, -0.15, 0.5, 0.05,
    -0.2, 0.35, 0.1, -0.4, 0.3, 0.2, -0.1, 0.45,
  ]);
  const bias = Float32Array.from([0.01, -0.02]);
  const expected = cpuConvTranspose1d(inputValues, weights, bias, shape);
  const input = storageBuffer(device, inputValues);
  const weight = storageBuffer(device, weights);
  const biasBuffer = storageBuffer(device, bias);
  const output = outputBuffer(device, expected.length);
  try {
    const dispatch = await kernel.createConvTranspose1dDispatch(
      "browser-vae-nontrivial-transpose",
      shape,
      {
        input: binding(input),
        weight: binding(weight),
        bias: binding(biasBuffer),
        output: binding(output),
      },
    );
    const actual = await executeAndRead(device, [dispatch], output, expected.length);
    return compare("conv-transpose1d", actual, expected, 2e-6);
  } finally {
    input.destroy();
    weight.destroy();
    biasBuffer.destroy();
    output.destroy();
  }
}

async function runNontrivialSnakeAndAdd(
  device: GPUDevice,
  kernel: AceCorrectnessVaePrimitiveKernel,
): Promise<CaseResult> {
  const shape = { batch: 1, frames: 4, channels: 2 } as const;
  const inputValues = Float32Array.from([
    -1.2, -0.5, 0.1, 0.35, 0.8, 1.4, 2.1, -2.4,
  ]);
  const alpha = Float32Array.from([-0.7, 0.2]);
  const beta = Float32Array.from([0.3, -0.4]);
  const right = Float32Array.from([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
  const snakeExpected = cpuSnake(inputValues, alpha, beta, shape.channels);
  const expected = Float32Array.from(
    snakeExpected,
    (value, index) => Math.fround(value + right[index]!),
  );
  const input = storageBuffer(device, inputValues);
  const alphaBuffer = storageBuffer(device, alpha);
  const betaBuffer = storageBuffer(device, beta);
  const snakeOutput = outputBuffer(device, inputValues.length);
  const rightBuffer = storageBuffer(device, right);
  const output = outputBuffer(device, inputValues.length);
  try {
    const snake = await kernel.createSnakeDispatch(
      "browser-vae-nontrivial-snake",
      shape,
      {
        input: binding(input),
        alpha: binding(alphaBuffer),
        beta: binding(betaBuffer),
        output: binding(snakeOutput),
      },
    );
    const add = await kernel.createAddDispatch(
      "browser-vae-nontrivial-add",
      shape,
      {
        left: binding(snakeOutput),
        right: binding(rightBuffer),
        output: binding(output),
      },
    );
    const actual = await executeAndRead(device, [snake, add], output, expected.length);
    return compare("snake-and-residual-add", actual, expected, 4e-6);
  } finally {
    input.destroy();
    alphaBuffer.destroy();
    betaBuffer.destroy();
    snakeOutput.destroy();
    rightBuffer.destroy();
    output.destroy();
  }
}

async function runCompleteToyDecoder(device: GPUDevice): Promise<CaseResult> {
  const config: AceVaeDecoderConfig = Object.freeze({
    id: "browser-nontrivial-oobleck",
    decoderInputChannels: 2,
    decoderChannels: 2,
    audioChannels: 2,
    channelMultiples: Object.freeze([1]),
    downsamplingRatios: Object.freeze([2]),
    sampleRateHz: 48_000,
  });
  const plan = planAceVaeDecoder(3, config);
  const tensorValues = createToyTensorValues(plan.operations);
  const inputValues = Float32Array.from([
    0.25, -0.5, 0.9, 0.3, -0.7, 1.1,
  ]);
  const expected = cpuDecoder(plan.operations, tensorValues, inputValues);
  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  const input = own(storageBuffer(device, inputValues));
  const output = own(outputBuffer(device, plan.outputElements));
  const workspaces = [
    own(outputBuffer(device, plan.maximumActivationElements)),
    own(outputBuffer(device, plan.maximumActivationElements)),
    own(outputBuffer(device, plan.maximumActivationElements)),
  ] as const;
  const tensors = Object.fromEntries(
    [...tensorValues].map(([name, values]) => {
      const operation = plan.operations.find((candidate) =>
        candidate.kind === "conv-transpose1d" && candidate.weight === name);
      if (operation?.kind !== "conv-transpose1d") {
        return [name, binding(own(storageBuffer(device, values)))];
      }
      // Exercise the package's output-axis logical-part path. The production
      // block-0 split is 0:614 / 614:1024; this toy graph uses 0:1 / 1:2.
      const rowElements =
        operation.shape.kernelSize * operation.shape.inputChannels;
      return [name, [
        {
          binding: binding(own(storageBuffer(
            device,
            values.subarray(0, rowElements),
          ))),
          partStart: 0,
          partEnd: 1,
        },
        {
          binding: binding(own(storageBuffer(
            device,
            values.subarray(rowElements),
          ))),
          partStart: 1,
          partEnd: 2,
        },
      ]];
    }),
  );
  const runtime = AceCorrectnessVaeDecoderRuntime.create(device);
  try {
    const dispatch = await runtime.createDecoderDispatch(
      "browser-nontrivial-oobleck",
      3,
      {
        input: binding(input),
        output: binding(output),
        workspaces: [binding(workspaces[0]), binding(workspaces[1]), binding(workspaces[2])],
        tensors,
      },
      config,
      1,
      {
        quantumWorkPolicy: {
          ...ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
          maximumOutputElements: 2,
        },
      },
    );
    if (dispatch.quanta.length <= plan.primitiveCount) {
      throw new Error("toy decoder did not exercise ranged cooperative quanta");
    }
    if (
      dispatch.quanta.some((quantum) => quantum.logicalOutputCount > 2) ||
      dispatch.primitiveCount !== dispatch.quanta.reduce(
        (total, quantum) => total + quantum.primitiveCount,
        0,
      )
    ) {
      throw new Error("toy decoder cooperative accounting changed");
    }
    const transposeQuanta = dispatch.quanta.filter((quantum) =>
      quantum.operationKind === "conv-transpose1d"
    );
    if (
      transposeQuanta.length === 0 ||
      transposeQuanta.some((quantum) => quantum.primitiveCount !== 2)
    ) {
      throw new Error("toy decoder did not group both transpose parts per row quantum");
    }
    const chunkPlan = planAceVaeChunkedDecode(3, {
      chunkFrames: 3,
      overlapFrames: 0,
      config,
    });
    const readback = device.createBuffer({
      size: alignedSize(chunkPlan.maximumDecodedFloat32Bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const memory = planAceVaeChunkGpuBackendMemory(
      chunkPlan,
      [...tensorValues.values()].reduce(
        (bytes, values) => bytes + values.byteLength,
        0,
      ),
      device.limits.minUniformBufferOffsetAlignment,
      {
        "vae.decoder.block.0.conv_t1.weight": [
          { partStart: 0, partEnd: 1 },
          { partStart: 1, partEnd: 2 },
        ],
      },
      {
        ...ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
        maximumOutputElements: 2,
      },
    );
    let resourcesDestroyed = false;
    const backend = AceVaeChunkGpuBackend.fromPreparedResources({
      device,
      plan: chunkPlan,
      finalLatents: inputValues,
      input: binding(input),
      output: binding(output),
      readback,
      decoderDispatches: new Map([[3, dispatch]]),
      memory,
      destroy(): void {
        if (resourcesDestroyed) return;
        resourcesDestroyed = true;
        runtime.destroy();
        readback.destroy();
      },
    });
    let actual: Float32Array;
    try {
      actual = await backend.decodeWindow(chunkPlan.windows[0]!);
    } finally {
      await backend.destroy();
    }
    return compare(
      "complete-nontrivial-oobleck-two-part-transpose",
      actual,
      expected,
      2e-4,
    );
  } finally {
    runtime.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

function createToyTensorValues(
  operations: readonly AceVaeDecoderOperation[],
): ReadonlyMap<string, Float32Array> {
  const tensors = new Map<string, Float32Array>();
  const add = (name: string, elements: number, scale: number, offset = 0): void => {
    if (tensors.has(name)) return;
    const seed = stableHash(name);
    tensors.set(name, Float32Array.from(
      { length: elements },
      (_, index) => Math.fround((((seed + index * 17) % 19) - 9) * scale + offset),
    ));
  };
  for (const operation of operations) {
    switch (operation.kind) {
      case "conv1d":
      case "conv-transpose1d":
        add(
          operation.weight,
          operation.shape.outputChannels * operation.shape.kernelSize *
            operation.shape.inputChannels,
          0.0125,
        );
        if (operation.bias !== undefined) {
          add(operation.bias, operation.shape.outputChannels, 0.005);
        }
        break;
      case "snake":
        add(operation.alpha, operation.shape.channels, 0.015, -0.8);
        add(operation.beta, operation.shape.channels, 0.0125, 0.1);
        break;
      case "add":
        break;
    }
  }
  return tensors;
}

function cpuDecoder(
  operations: readonly AceVaeDecoderOperation[],
  tensors: ReadonlyMap<string, Float32Array>,
  input: Float32Array,
): Float32Array {
  const slots = new Map<AceVaeDecoderSlot, Float32Array>([["input", input]]);
  for (const operation of operations) {
    const source = requiredSlot(slots, operation.input);
    let result: Float32Array;
    switch (operation.kind) {
      case "conv1d":
        result = cpuConv1d(
          source,
          requiredTensor(tensors, operation.weight),
          operation.bias === undefined
            ? undefined
            : requiredTensor(tensors, operation.bias),
          operation.shape,
        );
        break;
      case "conv-transpose1d":
        result = cpuConvTranspose1d(
          source,
          requiredTensor(tensors, operation.weight),
          requiredTensor(tensors, operation.bias),
          operation.shape,
        );
        break;
      case "snake":
        result = cpuSnake(
          source,
          requiredTensor(tensors, operation.alpha),
          requiredTensor(tensors, operation.beta),
          operation.shape.channels,
        );
        break;
      case "add": {
        const right = requiredSlot(slots, operation.right);
        result = Float32Array.from(source, (value, index) =>
          Math.fround(value + right[index]!));
        break;
      }
    }
    slots.set(operation.output, result);
  }
  return requiredSlot(slots, "output");
}

function cpuConv1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | undefined,
  shape: AceVaeConv1dShape,
): Float32Array {
  const effectiveKernel = shape.dilation * (shape.kernelSize - 1) + 1;
  const outputFrames = Math.floor(
    (shape.inputFrames + 2 * shape.padding - effectiveKernel) / shape.stride,
  ) + 1;
  const output = new Float32Array(
    shape.batch * outputFrames * shape.outputChannels,
  );
  for (let batch = 0; batch < shape.batch; batch += 1) {
    for (let time = 0; time < outputFrames; time += 1) {
      for (let out = 0; out < shape.outputChannels; out += 1) {
        let sum = bias?.[out] ?? 0;
        for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
          const inputTime =
            time * shape.stride + kernel * shape.dilation - shape.padding;
          if (inputTime < 0 || inputTime >= shape.inputFrames) continue;
          for (let channel = 0; channel < shape.inputChannels; channel += 1) {
            const inputIndex =
              (batch * shape.inputFrames + inputTime) * shape.inputChannels + channel;
            const weightIndex =
              (out * shape.kernelSize + kernel) * shape.inputChannels + channel;
            sum = Math.fround(sum + Math.fround(input[inputIndex]! * weight[weightIndex]!));
          }
        }
        output[(batch * outputFrames + time) * shape.outputChannels + out] = sum;
      }
    }
  }
  return output;
}

function cpuConvTranspose1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | undefined,
  shape: AceVaeConvTranspose1dShape,
): Float32Array {
  const outputFrames =
    (shape.inputFrames - 1) * shape.stride - 2 * shape.padding +
    shape.dilation * (shape.kernelSize - 1) + shape.outputPadding + 1;
  const output = new Float32Array(
    shape.batch * outputFrames * shape.outputChannels,
  );
  for (let batch = 0; batch < shape.batch; batch += 1) {
    for (let time = 0; time < outputFrames; time += 1) {
      for (let out = 0; out < shape.outputChannels; out += 1) {
        let sum = bias?.[out] ?? 0;
        for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
          const numerator = time + shape.padding - kernel * shape.dilation;
          if (numerator < 0 || numerator % shape.stride !== 0) continue;
          const inputTime = numerator / shape.stride;
          if (inputTime >= shape.inputFrames) continue;
          for (let channel = 0; channel < shape.inputChannels; channel += 1) {
            const inputIndex =
              (batch * shape.inputFrames + inputTime) * shape.inputChannels + channel;
            const weightIndex =
              (out * shape.kernelSize + kernel) * shape.inputChannels + channel;
            sum = Math.fround(sum + Math.fround(input[inputIndex]! * weight[weightIndex]!));
          }
        }
        output[(batch * outputFrames + time) * shape.outputChannels + out] = sum;
      }
    }
  }
  return output;
}

function cpuSnake(
  input: Float32Array,
  alpha: Float32Array,
  beta: Float32Array,
  channels: number,
): Float32Array {
  return Float32Array.from(input, (value, index) => {
    const channel = index % channels;
    const alphaValue = Math.exp(alpha[channel]!);
    const betaValue = Math.exp(beta[channel]!);
    const periodic = Math.sin(alphaValue * value);
    const reciprocalBeta = 1 / (betaValue + 1e-9);
    return Math.fround(value + reciprocalBeta * periodic * periodic);
  });
}

function storageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const owned = new Float32Array(data.length);
  owned.set(data);
  const buffer = device.createBuffer({
    size: alignedSize(owned.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(buffer, 0, owned);
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}

function outputBuffer(device: GPUDevice, elements: number): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * Float32Array.BYTES_PER_ELEMENT),
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

async function executeAndRead(
  device: GPUDevice,
  dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[],
  output: GPUBuffer,
  elements: number,
): Promise<Float32Array> {
  const bytes = elements * Float32Array.BYTES_PER_ELEMENT;
  const readback = device.createBuffer({
    size: alignedSize(bytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches) dispatch.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, alignedSize(bytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return Float32Array.from(
      new Float32Array(readback.getMappedRange(), 0, elements),
    );
  } finally {
    readback.destroy();
  }
}

function compare(
  operation: string,
  actual: Float32Array,
  expected: Float32Array,
  tolerance: number,
): CaseResult {
  if (actual.length !== expected.length) {
    throw new Error(`${operation} output length mismatch`);
  }
  let maximumAbsoluteError = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const value = actual[index]!;
    if (!Number.isFinite(value)) {
      throw new Error(`${operation} produced non-finite value at ${index}`);
    }
    const error = Math.abs(value - expected[index]!);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    if (error > tolerance * Math.max(1, Math.abs(expected[index]!))) {
      throw new Error(
        `${operation} mismatch at ${index}: ${value} != ${expected[index]} (error ${error})`,
      );
    }
  }
  return { operation, valuesChecked: actual.length, maximumAbsoluteError };
}

function requiredTensor(
  tensors: ReadonlyMap<string, Float32Array>,
  name: string,
): Float32Array {
  const tensor = tensors.get(name);
  if (tensor === undefined) throw new Error(`missing CPU tensor ${name}`);
  return tensor;
}

function requiredSlot(
  slots: ReadonlyMap<AceVaeDecoderSlot, Float32Array>,
  slot: AceVaeDecoderSlot,
): Float32Array {
  const value = slots.get(slot);
  if (value === undefined) throw new Error(`missing CPU decoder slot ${slot}`);
  return value;
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619) >>> 0;
  }
  return hash;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
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
