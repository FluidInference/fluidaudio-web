import {
  ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES,
  runAceOpt0006QuantumBatches,
  type AceOpt0006BatchProgress,
} from "../../benchmark/opt-0006-vae-command-buffer-coalescing.js";
import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  type AceVaeDecoderConfig,
  type AceVaeDecoderDispatch,
  type AceVaeDecoderOperation,
} from "../../src/webgpu/vae-decoder.js";

const QNAN_BITS = 0x7fc0_0000;
const resultNode = requireNode("#result");
const progressNode = requireNode("#progress");
const startButton = requireButton("#start");

interface PreparedMixedDecoder {
  readonly planOutputElements: number;
  readonly maximumActivationElements: number;
  readonly dispatch: AceVaeDecoderDispatch;
  readonly output: GPUBuffer;
  readonly workspaces: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  readonly owned: readonly GPUBuffer[];
  readonly runtime: AceCorrectnessVaeDecoderRuntime;
}

interface RunMeasurement {
  readonly batchSize: number;
  readonly outputBits: readonly number[];
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly progressEventCount: number;
  readonly finalCompletedQuanta: number;
  readonly logicalQuantumCount: number;
  readonly physicalDispatchCount: number;
  readonly commandBufferCount: number;
  readonly queueDrainCount: number;
  readonly cooperativeIdleMs: number;
  readonly wallMilliseconds: number;
  readonly drainMilliseconds: readonly number[];
  readonly maximumDrainMilliseconds: number;
}

interface HeartbeatResult {
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

startButton.addEventListener("click", () => {
  startButton.disabled = true;
  document.body.dataset.status = "running";
  progressNode.textContent = "requesting WebGPU device";
  void run().then(
    (result) => finish("passed", JSON.stringify(result)),
    (error: unknown) => finish(
      "failed",
      error instanceof Error ? error.stack ?? error.message : String(error),
    ),
  );
});

async function run(): Promise<Readonly<Record<string, unknown>>> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const required = {
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16 * 1024,
    maxStorageBufferBindingSize: 256 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
  } as const;
  for (const [name, value] of Object.entries(required)) {
    if ((adapter.limits as unknown as Record<string, number>)[name]! < value) {
      throw new Error(`Adapter limit ${name} is below ${value}`);
    }
  }
  const device = await adapter.requestDevice({ requiredLimits: required });
  const heartbeat = startHeartbeat();
  let prepared: PreparedMixedDecoder | undefined;
  try {
    prepared = await prepareMixedDecoder(device);
    const measurements: RunMeasurement[] = [];
    let baseline: readonly number[] | undefined;
    for (const batchSize of ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES) {
      progressNode.textContent = `correctness batch size ${batchSize}`;
      const measurement = await executePrepared(device, prepared, batchSize);
      if (baseline === undefined) baseline = measurement.outputBits;
      requireBitIdentity(baseline, measurement.outputBits, batchSize);
      requireMeasurementAccounting(measurement);
      measurements.push(measurement);
    }
    progressNode.textContent = "cancellation proof";
    const cancellation = await runCancellationProof(device, prepared, 4);
    const heartbeatResult = heartbeat.stop();
    return Object.freeze({
      schemaVersion: 1,
      kind: "ace-opt-0006-vae-command-buffer-coalescing-correctness",
      adapter: Object.freeze({
        info: adapter.info,
        limits: Object.freeze({
          maxComputeInvocationsPerWorkgroup:
            device.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupStorageSize:
            device.limits.maxComputeWorkgroupStorageSize,
          maxStorageBufferBindingSize:
            device.limits.maxStorageBufferBindingSize,
        }),
      }),
      fixture: Object.freeze({
        id: "opt-0006-c136-c128-one-block-mixed-decoder-v1",
        decoderInputChannels: 136,
        decoderChannels: 128,
        inputFrames: 3,
        logicalQuantumCount: prepared.dispatch.quanta.length,
        physicalDispatchCount: prepared.dispatch.primitiveCount,
        outputElements: prepared.planOutputElements,
        tiledOperationLabels:
          prepared.dispatch.conv1dSelection?.tiledOperationLabels ?? [],
        channelChunkedOperationLabels:
          prepared.dispatch.conv1dSelection?.channelChunkedOperationLabels ?? [],
        portableOperationLabels:
          prepared.dispatch.conv1dSelection?.portableOperationLabels ?? [],
      }),
      batchSizes: ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES,
      measurements: Object.freeze(measurements),
      cancellation,
      heartbeat: heartbeatResult,
      allBitIdentical: true,
      listeningRequired: false,
      productionWindowExecuted: false,
      songExecuted: false,
    });
  } finally {
    heartbeat.stop();
    prepared?.runtime.destroy();
    for (const buffer of prepared?.owned ?? []) buffer.destroy();
    device.destroy();
  }
}

async function prepareMixedDecoder(device: GPUDevice): Promise<PreparedMixedDecoder> {
  const config: AceVaeDecoderConfig = Object.freeze({
    id: "browser-opt-0006-oobleck",
    decoderInputChannels: 136,
    decoderChannels: 128,
    audioChannels: 2,
    channelMultiples: Object.freeze([1]),
    downsamplingRatios: Object.freeze([2]),
    sampleRateHz: 48_000,
  });
  const inputFrames = 3;
  const plan = planAceVaeDecoder(inputFrames, config);
  const tensorValues = createTensorValues(plan.operations);
  const inputValues = Float32Array.from(
    { length: inputFrames * config.decoderInputChannels },
    (_, index) => Math.fround(((index * 13) % 31 - 15) * 0.025),
  );
  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  let runtime: AceCorrectnessVaeDecoderRuntime | undefined;
  try {
    const input = own(storageBuffer(device, inputValues));
    const output = own(outputBuffer(device, plan.outputElements));
    const workspaces = [
      own(outputBuffer(device, plan.maximumActivationElements)),
      own(outputBuffer(device, plan.maximumActivationElements)),
      own(outputBuffer(device, plan.maximumActivationElements)),
    ] as const;
    const tensors = Object.fromEntries([...tensorValues].map(([name, values]) => {
      const operation = plan.operations.find((candidate) =>
        candidate.kind === "conv-transpose1d" && candidate.weight === name
      );
      if (operation?.kind !== "conv-transpose1d") {
        return [name, binding(own(storageBuffer(device, values)))];
      }
      const rowElements = operation.shape.kernelSize * operation.shape.inputChannels;
      const splitOutputChannel = Math.floor(operation.shape.outputChannels / 2);
      const splitElement = splitOutputChannel * rowElements;
      return [name, [
        {
          binding: binding(own(storageBuffer(device, values.subarray(0, splitElement)))),
          partStart: 0,
          partEnd: splitOutputChannel,
        },
        {
          binding: binding(own(storageBuffer(device, values.subarray(splitElement)))),
          partStart: splitOutputChannel,
          partEnd: operation.shape.outputChannels,
        },
      ]];
    }));
    runtime = AceCorrectnessVaeDecoderRuntime.create(device);
    const dispatch = await runtime.createDecoderDispatch(
      "browser-opt-0006",
      inputFrames,
      {
        input: binding(input),
        output: binding(output),
        workspaces: workspaces.map(binding) as unknown as readonly [
          GPUBufferBinding,
          GPUBufferBinding,
          GPUBufferBinding,
        ],
        tensors,
      },
      config,
      1,
      {
        quantumWorkPolicy: {
          ...ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
          maximumOutputElements: 128,
        },
      },
    );
    if (
      dispatch.quanta.length !== 109 ||
      dispatch.primitiveCount !== 115 ||
      dispatch.conv1dSelection?.tiledQuantumCount !== 7 ||
      dispatch.conv1dSelection.channelChunkedQuantumCount !== 15 ||
      dispatch.conv1dSelection.portableQuantumCount !== 18
    ) {
      throw new Error(
        `OPT-0006 mixed decoder fixture selection changed: ${JSON.stringify({
          quanta: dispatch.quanta.length,
          primitiveCount: dispatch.primitiveCount,
          selection: dispatch.conv1dSelection,
        })}`,
      );
    }
    return Object.freeze({
      planOutputElements: plan.outputElements,
      maximumActivationElements: plan.maximumActivationElements,
      dispatch,
      output,
      workspaces,
      owned: Object.freeze(owned),
      runtime,
    });
  } catch (error) {
    runtime?.destroy();
    for (const buffer of owned) buffer.destroy();
    throw error;
  }
}

async function executePrepared(
  device: GPUDevice,
  prepared: PreparedMixedDecoder,
  batchSize: number,
): Promise<RunMeasurement> {
  prefill(device, prepared);
  await device.queue.onSubmittedWorkDone();
  const progress: AceOpt0006BatchProgress[] = [];
  const drainMilliseconds: number[] = [];
  const queue = measuredQueue(device.queue, drainMilliseconds);
  const started = performance.now();
  const scheduled = await runAceOpt0006QuantumBatches({
    device,
    queue,
    quanta: prepared.dispatch.quanta,
    maximumQuantaPerCommandBuffer: batchSize,
    signal: new AbortController().signal,
    finalCommandBufferRemains: true,
    onProgress: (value) => progress.push(value),
  });
  const wallMilliseconds = performance.now() - started;
  const output = await readBuffer(
    device,
    prepared.output,
    prepared.planOutputElements,
  );
  const outputBits = Array.from(new Uint32Array(
    output.buffer,
    output.byteOffset,
    output.length,
  ));
  const finiteCount = output.reduce(
    (count, value) => count + (Number.isFinite(value) ? 1 : 0),
    0,
  );
  const nonzeroCount = output.reduce(
    (count, value) => count + (value !== 0 ? 1 : 0),
    0,
  );
  return Object.freeze({
    batchSize,
    outputBits: Object.freeze(outputBits),
    finiteCount,
    nonzeroCount,
    progressEventCount: progress.length,
    finalCompletedQuanta: progress.at(-1)?.completedQuanta ?? 0,
    logicalQuantumCount: prepared.dispatch.quanta.length,
    physicalDispatchCount: prepared.dispatch.primitiveCount,
    commandBufferCount: scheduled.commandBuffersSubmitted,
    queueDrainCount: scheduled.queueDrains,
    cooperativeIdleMs: scheduled.cooperativeIdleMs,
    wallMilliseconds,
    drainMilliseconds: Object.freeze(drainMilliseconds),
    maximumDrainMilliseconds: Math.max(...drainMilliseconds),
  });
}

async function runCancellationProof(
  device: GPUDevice,
  prepared: PreparedMixedDecoder,
  batchSize: number,
): Promise<Readonly<Record<string, unknown>>> {
  prefill(device, prepared);
  await device.queue.onSubmittedWorkDone();
  const controller = new AbortController();
  const reason = new DOMException("OPT-0006 cancellation proof", "AbortError");
  let submits = 0;
  let drains = 0;
  const queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone"> = {
    submit(commandBuffers): undefined {
      submits += 1;
      device.queue.submit(commandBuffers);
      return undefined;
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      await device.queue.onSubmittedWorkDone();
      drains += 1;
      return undefined;
    },
  };
  const started = performance.now();
  let progressEvents = 0;
  let caught: unknown;
  try {
    await runAceOpt0006QuantumBatches({
      device,
      queue,
      quanta: prepared.dispatch.quanta,
      maximumQuantaPerCommandBuffer: batchSize,
      signal: controller.signal,
      finalCommandBufferRemains: true,
      onProgress: () => {
        progressEvents += 1;
        if (!controller.signal.aborted) controller.abort(reason);
      },
    });
  } catch (error) {
    caught = error;
  }
  if (caught !== reason || submits !== 1 || drains !== 1) {
    throw new Error("OPT-0006 cancellation did not stop after one drained batch");
  }
  return Object.freeze({
    batchSize,
    activeBatchDrained: true,
    laterSubmissionPrevented: true,
    submits,
    drains,
    progressEvents,
    elapsedMilliseconds: performance.now() - started,
  });
}

function requireMeasurementAccounting(measurement: RunMeasurement): void {
  const expectedCommandBuffers = Math.ceil(
    measurement.logicalQuantumCount / measurement.batchSize,
  );
  if (
    measurement.finiteCount !== measurement.outputBits.length ||
    measurement.nonzeroCount === 0 ||
    measurement.progressEventCount !== measurement.logicalQuantumCount ||
    measurement.finalCompletedQuanta !== measurement.logicalQuantumCount ||
    measurement.commandBufferCount !== expectedCommandBuffers ||
    measurement.queueDrainCount !== expectedCommandBuffers ||
    measurement.cooperativeIdleMs !== expectedCommandBuffers
  ) {
    throw new Error(
      `OPT-0006 batch-size ${measurement.batchSize} accounting failed`,
    );
  }
}

function requireBitIdentity(
  baseline: readonly number[],
  candidate: readonly number[],
  batchSize: number,
): void {
  if (baseline.length !== candidate.length) throw new Error("output length changed");
  for (let index = 0; index < baseline.length; index += 1) {
    if (baseline[index] !== candidate[index]) {
      throw new Error(
        `OPT-0006 batch-size ${batchSize} differs at ${index}: ${candidate[index]} != ${baseline[index]}`,
      );
    }
  }
}

function measuredQueue(
  realQueue: GPUQueue,
  drainMilliseconds: number[],
): Pick<GPUQueue, "submit" | "onSubmittedWorkDone"> {
  let submittedAt = 0;
  return {
    submit(commandBuffers): undefined {
      submittedAt = performance.now();
      realQueue.submit(commandBuffers);
      return undefined;
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      await realQueue.onSubmittedWorkDone();
      drainMilliseconds.push(performance.now() - submittedAt);
      return undefined;
    },
  };
}

function prefill(device: GPUDevice, prepared: PreparedMixedDecoder): void {
  const write = (buffer: GPUBuffer, elements: number): void => {
    device.queue.writeBuffer(buffer, 0, new Uint32Array(elements).fill(QNAN_BITS));
  };
  write(prepared.output, prepared.planOutputElements);
  for (const workspace of prepared.workspaces) {
    write(workspace, prepared.maximumActivationElements);
  }
}

async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  elements: number,
): Promise<Float32Array> {
  const bytes = alignedSize(elements * Float32Array.BYTES_PER_ELEMENT);
  const readback = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    return Float32Array.from(
      new Float32Array(readback.getMappedRange(), 0, elements),
    );
  } finally {
    readback.destroy();
  }
}

function createTensorValues(
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

function storageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const owned = new Float32Array(data.length);
  owned.set(data);
  const buffer = device.createBuffer({
    size: alignedSize(owned.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, owned);
  return buffer;
}

function outputBuffer(device: GPUDevice, elements: number): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * Float32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619) >>> 0;
  }
  return hash;
}

function startHeartbeat(): { stop(): HeartbeatResult } {
  let stopped = false;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let frameHandle = 0;
  let timerHandle = 0;
  const frame = (time: number): void => {
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      time - lastAnimationFrame,
    );
    lastAnimationFrame = time;
    if (!stopped) frameHandle = requestAnimationFrame(frame);
  };
  const timer = (): void => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    if (!stopped) timerHandle = window.setTimeout(timer, 10);
  };
  frameHandle = requestAnimationFrame(frame);
  timerHandle = window.setTimeout(timer, 10);
  return {
    stop(): HeartbeatResult {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearTimeout(timerHandle);
      }
      return Object.freeze({
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

function requireNode(selector: string): HTMLElement {
  const node = document.querySelector<HTMLElement>(selector);
  if (node === null) throw new Error(`Missing ${selector}`);
  return node;
}

function requireButton(selector: string): HTMLButtonElement {
  const node = document.querySelector<HTMLButtonElement>(selector);
  if (node === null) throw new Error(`Missing ${selector}`);
  return node;
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  progressNode.textContent = status;
  resultNode.textContent = message;
  startButton.disabled = false;
}
