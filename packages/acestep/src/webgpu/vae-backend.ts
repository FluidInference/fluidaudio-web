import { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import {
  AceCooperativeGpuScheduler,
  AceFifoGraphOwner,
} from "../runtime/scheduler.js";
import { AceGpuArena } from "./arena.js";
import {
  type AceVaeChunkBackend,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
} from "./vae-chunks.js";
import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  resolveAceVaeDecoderTensorBindings,
  snapshotAceVaeDecoderQuantumWorkPolicy,
  type AceVaeDecoderDispatch,
  type AceVaeDecoderQuantum,
  type AceVaeDecoderQuantumWorkPolicy,
  type AceVaeLogicalTensorBinding,
  type AceVaeTransposePartGeometry,
} from "./vae-decoder.js";
import {
  requireAceBindingBytes,
  requirePositiveSafeInteger,
} from "./kernels/correctness-utils.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const STORAGE_ALIGNMENT = 256;
const RANGE_CONTROL_BYTES = 16;

/** Measured OPT-0006 production topology. Readback remains a separate buffer. */
export const ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER = 8;

export interface AceVaeChunkGpuBackendProgress {
  readonly windowIndex: number;
  readonly completedDecoderQuanta: number;
  readonly totalDecoderQuanta: number;
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly stage: "decoder" | "readback";
}

/** Exact bounded allocations; driver-owned pipeline/command metadata is opaque. */
export interface AceVaeChunkGpuBackendMemoryPlan {
  readonly residentWeightBytes: number;
  readonly inputBufferBytes: number;
  readonly outputBufferBytes: number;
  readonly workspaceBufferBytes: number;
  readonly workspaceBufferCount: 3;
  readonly arenaBytes: number;
  readonly rangeControlBytes: number;
  readonly readbackBufferBytes: number;
  readonly accountedGpuBytes: number;
  readonly latentSnapshotBytes: number;
  readonly maximumReturnedWindowBytes: number;
  readonly boundedCpuBytes: number;
  readonly uniqueDecoderInputFrames: readonly number[];
  readonly decoderQuantaPerCommandBuffer: number;
  readonly maximumDecoderQuantaPerWindow: number;
  readonly maximumDecoderCommandBuffersPerWindow: number;
  /** Bounded decoder batches plus one bounded output-copy command buffer. */
  readonly maximumCommandBuffersPerWindow: number;
}

export interface AceVaeChunkGpuBackendOptions {
  readonly device: GPUDevice;
  readonly plan: AceVaeChunkedDecodePlan;
  /** FP32 NLC `[latentFrames,decoderInputChannels]`; snapshotted on creation. */
  readonly finalLatents: Float32Array;
  /** Ownership transfers at call entry, including on factory failure. */
  readonly ownedVaeWeights: AceGpuTensorPhase;
  readonly signal?: AbortSignal;
  readonly quantumWorkPolicy?: AceVaeDecoderQuantumWorkPolicy;
  readonly onProgress?: (progress: AceVaeChunkGpuBackendProgress) => void;
}

export interface AceVaePreparedChunkGpuResources {
  readonly device: GPUDevice;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly finalLatents: Float32Array;
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly readback: GPUBuffer;
  readonly decoderDispatches: ReadonlyMap<number, AceVaeDecoderDispatch>;
  readonly memory: AceVaeChunkGpuBackendMemoryPlan;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceVaeChunkGpuBackendProgress) => void;
  /** @internal Test seam; production resources use a real one-millisecond timer. */
  readonly yieldQueueIdle?: () => Promise<void>;
  /** @internal Test seam; production resources use the measured constant. */
  readonly decoderQuantaPerCommandBuffer?: number;
  /** Must drain before invocation. Called exactly once. */
  destroy(): void;
}

export class AceVaeBackendDeviceLostError extends Error {
  override readonly name = "AceVaeBackendDeviceLostError";

  constructor(info: Pick<GPUDeviceLostInfo, "reason" | "message">) {
    super(
      `ACE VAE WebGPU device lost (${info.reason}): ${info.message || "no device message"}`,
    );
  }
}

/**
 * Concrete bounded GPU implementation of `AceVaeChunkBackend`.
 *
 * One outer FIFO lease covers latent upload, every decoder batch, bounded
 * readback, and CPU detachment. The cooperative scheduler lazily creates and
 * submits one command buffer at a time, with a real queue-empty 1 ms interval
 * between them. Readback is always a separate final command buffer.
 */
export class AceVaeChunkGpuBackend implements AceVaeChunkBackend {
  readonly memory: AceVaeChunkGpuBackendMemoryPlan;

  private readonly scheduler = new AceCooperativeGpuScheduler();
  private readonly graphOwner = new AceFifoGraphOwner();
  private readonly lifetime = new AbortController();
  private readonly finalLatents: Float32Array<ArrayBuffer>;
  private readonly decoderQuantaPerCommandBuffer: number;
  private readonly resources: Omit<
    AceVaePreparedChunkGpuResources,
    "finalLatents"
  >;
  private readonly resourceAbortSignal: AbortSignal | undefined;
  private readonly resourceAbortListener: (() => void) | undefined;
  private destroyPromise: Promise<void> | undefined;
  private state: "live" | "destroying" | "destroyed" = "live";

  private constructor(
    resources: AceVaePreparedChunkGpuResources,
    takeLatentSnapshotOwnership = false,
  ) {
    const decoderQuantaPerCommandBuffer =
      resources.decoderQuantaPerCommandBuffer ??
        ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER;
    requirePositiveSafeInteger(
      decoderQuantaPerCommandBuffer,
      "ACE VAE decoder quanta per command buffer",
    );
    validatePreparedResources(resources, decoderQuantaPerCommandBuffer);
    resources.signal?.throwIfAborted();
    this.memory = resources.memory;
    this.decoderQuantaPerCommandBuffer = decoderQuantaPerCommandBuffer;
    this.finalLatents = takeLatentSnapshotOwnership
      ? requireOwnedLatentSnapshot(resources.plan, resources.finalLatents)
      : snapshotLatents(resources.plan, resources.finalLatents);
    const { finalLatents: _callerLatents, ...runtimeResources } = resources;
    this.resources = runtimeResources;
    this.resourceAbortSignal = resources.signal;
    this.resourceAbortListener = resources.signal === undefined
      ? undefined
      : () => {
          const reason = resources.signal!.reason;
          if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
          void this.beginDestroy(reason).catch(() => undefined);
        };
    this.resourceAbortSignal?.addEventListener(
      "abort",
      this.resourceAbortListener!,
      { once: true },
    );
    const weakBackend = new WeakRef(this);
    void resources.device.lost.then((info) => {
      const backend = weakBackend.deref();
      if (backend === undefined || backend.state !== "live") return;
      const error = new AceVaeBackendDeviceLostError(info);
      backend.lifetime.abort(error);
      void backend.beginDestroy(error).catch(() => undefined);
    });
  }

  static async create(
    options: AceVaeChunkGpuBackendOptions,
  ): Promise<AceVaeChunkGpuBackend> {
    const resources = await createConcreteResources(options);
    try {
      return new AceVaeChunkGpuBackend(resources, true);
    } catch (error) {
      resources.destroy();
      throw error;
    }
  }

  /** @internal Deterministic coordinator seam; not exported from the package. */
  static fromPreparedResources(
    resources: AceVaePreparedChunkGpuResources,
  ): AceVaeChunkGpuBackend {
    try {
      return new AceVaeChunkGpuBackend(resources);
    } catch (error) {
      resources.destroy();
      throw error;
    }
  }

  async decodeWindow(
    window: AceVaeDecodeWindow,
    signal?: AbortSignal,
  ): Promise<Float32Array> {
    this.requireLive();
    requirePlanWindow(this.resources.plan, window);
    const activeSignal = combineSignals([
      this.lifetime.signal,
      this.resources.signal,
      signal,
    ]);
    activeSignal.throwIfAborted();
    const lease = await this.graphOwner.acquire(activeSignal);
    try {
      this.requireLive();
      activeSignal.throwIfAborted();
      const dispatch = this.resources.decoderDispatches.get(
        window.latentWindowFrames,
      );
      if (dispatch === undefined) {
        throw new Error(
          `ACE VAE backend did not compile ${window.latentWindowFrames} latent frames`,
        );
      }
      this.uploadLatentWindow(window);
      activeSignal.throwIfAborted();

      const batches = planDecoderQuantumBatches(
        dispatch.quanta.length,
        this.decoderQuantaPerCommandBuffer,
      );
      const totalCommandBuffers = batches.length + 1;
      const outputBytes = checkedProduct(
        [window.decodedAudioFrames, this.resources.plan.audioChannels, FLOAT32_BYTES],
        `ACE VAE window ${window.index} readback bytes`,
      );
      await this.scheduler.runLazy({
        queue: this.resources.device.queue,
        commandBufferCount: totalCommandBuffers,
        createCommandBuffer: (index) => {
          activeSignal.throwIfAborted();
          const batch = batches[index];
          return batch === undefined
            ? encodeReadbackCopy(
              this.resources.device,
              this.resources.output,
              this.resources.readback,
              outputBytes,
              window.index,
            )
            : encodeDecoderQuantumBatch(
              this.resources.device,
              window,
              dispatch.quanta,
              batch,
              activeSignal,
            );
        },
        signal: activeSignal,
        ownerSignal: activeSignal,
        ...(this.resources.yieldQueueIdle === undefined
          ? {}
          : { yieldQueueIdle: this.resources.yieldQueueIdle }),
        onProgress: (progress) => {
          const batch = batches[progress.completedCommandBuffers - 1];
          if (batch === undefined) {
            this.resources.onProgress?.({
              windowIndex: window.index,
              completedDecoderQuanta: dispatch.quanta.length,
              totalDecoderQuanta: dispatch.quanta.length,
              ...progress,
              stage: "readback",
            });
            return;
          }
          for (let offset = 1; offset <= batch.quantumCount; offset += 1) {
            this.resources.onProgress?.({
              windowIndex: window.index,
              completedDecoderQuanta: batch.firstQuantumIndex + offset,
              totalDecoderQuanta: dispatch.quanta.length,
              ...progress,
              stage: "decoder",
            });
          }
        },
      });
      return await mapDetachedWindow(
        this.resources.readback,
        outputBytes,
        activeSignal,
      );
    } finally {
      lease.release();
    }
  }

  destroy(reason: unknown = destroyedError()): Promise<void> {
    return this.beginDestroy(reason);
  }

  private uploadLatentWindow(window: AceVaeDecodeWindow): void {
    const channels = this.resources.plan.decoderWorkspacePlan.config
      .decoderInputChannels;
    const elementStart = window.windowStartLatentFrame * channels;
    const elementEnd = window.windowEndLatentFrame * channels;
    const source = this.finalLatents.subarray(elementStart, elementEnd);
    const expectedElements = window.latentWindowFrames * channels;
    if (source.length !== expectedElements) {
      throw new Error(
        `ACE VAE window ${window.index} latent slice has ${source.length} ` +
          `elements; expected ${expectedElements}`,
      );
    }
    this.resources.device.queue.writeBuffer(
      this.resources.input.buffer,
      this.resources.input.offset ?? 0,
      source,
    );
  }

  private beginDestroy(reason: unknown): Promise<void> {
    if (this.destroyPromise !== undefined) return this.destroyPromise;
    this.state = "destroying";
    if (
      this.resourceAbortSignal !== undefined &&
      this.resourceAbortListener !== undefined
    ) {
      this.resourceAbortSignal.removeEventListener(
        "abort",
        this.resourceAbortListener,
      );
    }
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    this.destroyPromise = (async () => {
      try {
        await this.graphOwner.dispose();
        await this.scheduler.dispose();
      } finally {
        try {
          this.resources.destroy();
        } finally {
          this.state = "destroyed";
        }
      }
    })();
    return this.destroyPromise;
  }

  private requireLive(): void {
    if (this.state !== "live") {
      throw new DOMException(
        `ACE VAE chunk GPU backend is ${this.state}`,
        "InvalidStateError",
      );
    }
  }
}

export function planAceVaeChunkGpuBackendMemory(
  plan: AceVaeChunkedDecodePlan,
  residentWeightBytes: number,
  minUniformBufferOffsetAlignment: number,
  transposeParts: Readonly<
    Record<string, readonly AceVaeTransposePartGeometry[]>
  > = {},
  quantumWorkPolicy: AceVaeDecoderQuantumWorkPolicy =
    ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  decoderQuantaPerCommandBuffer = ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
): AceVaeChunkGpuBackendMemoryPlan {
  const stableQuantumWorkPolicy = snapshotAceVaeDecoderQuantumWorkPolicy(
    quantumWorkPolicy,
  );
  requirePositiveSafeInteger(
    decoderQuantaPerCommandBuffer,
    "ACE VAE decoder quanta per command buffer",
  );
  requirePositiveSafeInteger(residentWeightBytes, "ACE VAE resident weight bytes");
  requirePositiveSafeInteger(
    minUniformBufferOffsetAlignment,
    "ACE VAE uniform offset alignment",
  );
  if (
    minUniformBufferOffsetAlignment < RANGE_CONTROL_BYTES ||
    minUniformBufferOffsetAlignment % 4 !== 0
  ) {
    throw new RangeError("ACE VAE uniform alignment cannot hold a range record");
  }
  const uniqueDecoderInputFrames = Object.freeze(
    [...new Set(plan.windows.map((window) => window.latentWindowFrames))]
      .sort((left, right) => left - right),
  );
  let rangeControlBytes = 0;
  let maximumDecoderQuantaPerWindow = 0;
  for (const inputFrames of uniqueDecoderInputFrames) {
    const graph = planAceVaeDecoder(
      inputFrames,
      plan.decoderWorkspacePlan.config,
      1,
    );
    const cooperative = planAceVaeDecoderQuanta(
      graph,
      transposeParts,
      stableQuantumWorkPolicy,
    );
    const bytes = checkedSum(
      checkedProduct(
        [cooperative.primitiveDispatchCount - 1, minUniformBufferOffsetAlignment],
        "ACE VAE range-control offsets",
        true,
      ),
      RANGE_CONTROL_BYTES,
      "ACE VAE range-control bytes",
    );
    rangeControlBytes = checkedSum(
      rangeControlBytes,
      bytes,
      "ACE VAE range-control total",
    );
    maximumDecoderQuantaPerWindow = Math.max(
      maximumDecoderQuantaPerWindow,
      cooperative.quantumCount,
    );
  }
  const inputBufferBytes = align(
    plan.decoderWorkspacePlan.inputElements * FLOAT32_BYTES,
    STORAGE_ALIGNMENT,
  );
  const outputBufferBytes = align(
    plan.maximumDecodedFloat32Bytes,
    STORAGE_ALIGNMENT,
  );
  const workspaceBufferBytes = align(
    plan.decoderWorkspacePlan.workspaceBytes,
    STORAGE_ALIGNMENT,
  );
  const arenaBytes = checkedSum(
    checkedSum(inputBufferBytes, outputBufferBytes, "ACE VAE arena IO bytes"),
    checkedProduct(
      [workspaceBufferBytes, 3],
      "ACE VAE arena workspace bytes",
    ),
    "ACE VAE arena bytes",
  );
  const readbackBufferBytes = align(plan.maximumDecodedFloat32Bytes, 4);
  const accountedGpuBytes = [
    residentWeightBytes,
    arenaBytes,
    rangeControlBytes,
    readbackBufferBytes,
  ].reduce(
    (total, value) => checkedSum(total, value, "ACE VAE accounted GPU bytes"),
    0,
  );
  const latentSnapshotBytes = checkedProduct(
    [
      plan.latentFrames,
      plan.decoderWorkspacePlan.config.decoderInputChannels,
      FLOAT32_BYTES,
    ],
    "ACE VAE latent snapshot bytes",
  );
  const maximumReturnedWindowBytes = plan.maximumDecodedFloat32Bytes;
  const maximumDecoderCommandBuffersPerWindow = Math.ceil(
    maximumDecoderQuantaPerWindow / decoderQuantaPerCommandBuffer,
  );
  return Object.freeze({
    residentWeightBytes,
    inputBufferBytes,
    outputBufferBytes,
    workspaceBufferBytes,
    workspaceBufferCount: 3,
    arenaBytes,
    rangeControlBytes,
    readbackBufferBytes,
    accountedGpuBytes,
    latentSnapshotBytes,
    maximumReturnedWindowBytes,
    boundedCpuBytes: checkedSum(
      latentSnapshotBytes,
      maximumReturnedWindowBytes,
      "ACE VAE bounded CPU bytes",
    ),
    uniqueDecoderInputFrames,
    decoderQuantaPerCommandBuffer,
    maximumDecoderQuantaPerWindow,
    maximumDecoderCommandBuffersPerWindow,
    maximumCommandBuffersPerWindow:
      maximumDecoderCommandBuffersPerWindow + 1,
  });
}

async function createConcreteResources(
  options: AceVaeChunkGpuBackendOptions,
): Promise<AceVaePreparedChunkGpuResources> {
  const weights = options.ownedVaeWeights;
  const requestedQuantumWorkPolicy =
    options.quantumWorkPolicy ?? ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY;
  const quantumWorkPolicy = snapshotAceVaeDecoderQuantumWorkPolicy(
    requestedQuantumWorkPolicy,
  );
  let arena: AceGpuArena | undefined;
  let readback: GPUBuffer | undefined;
  let decoder: AceCorrectnessVaeDecoderRuntime | undefined;
  let resourcesPublished = false;
  try {
    options.signal?.throwIfAborted();
    const latentSnapshot = snapshotLatents(options.plan, options.finalLatents);
    if (weights.phases.length !== 1 || weights.phases[0] !== "vae") {
      throw new Error(
        `ACE VAE backend requires an exclusively resident vae phase; got ${weights.phases.join("+")}`,
      );
    }
    const graph = options.plan.decoderWorkspacePlan;
    const residentTensors: Record<string, AceVaeLogicalTensorBinding> = {};
    for (const name of graph.requiredTensorNames) {
      const tensor = weights.logicalTensor(name);
      for (const part of tensor.parts) {
        if (part.tensor.phase !== "vae") {
          throw new Error(`ACE VAE tensor ${name} is resident in ${part.tensor.phase}`);
        }
      }
      residentTensors[name] = Object.freeze(tensor.parts.map((part) =>
        Object.freeze({
          binding: part.binding,
          partStart: part.tensor.partStart,
          partEnd: part.tensor.partEnd,
        })
      ));
    }
    const tensors = resolveAceVaeDecoderTensorBindings(
      graph,
      residentTensors,
      "ACE authenticated VAE phase",
    );
    const transposeParts = transposePartGeometry(graph, tensors);
    const memory = planAceVaeChunkGpuBackendMemory(
      options.plan,
      weights.residentBytes,
      options.device.limits.minUniformBufferOffsetAlignment,
      transposeParts,
      quantumWorkPolicy,
    );
    arena = await AceGpuArena.create(options.device, [
      { label: "ace-vae-chunk-input", byteLength: memory.inputBufferBytes },
      { label: "ace-vae-chunk-output", byteLength: memory.outputBufferBytes },
      ...Array.from({ length: 3 }, (_, index) => ({
        label: `ace-vae-chunk-workspace-${index}`,
        byteLength: memory.workspaceBufferBytes,
      })),
    ]);
    options.signal?.throwIfAborted();
    [readback] = await createAceScopedBuffers(
      options.device,
      [{
        label: "ace-vae-chunk-readback",
        size: memory.readbackBufferBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }],
      "ACE VAE chunk readback allocation",
    );
    const input = arena.binding(arena.slice(
      "ace-vae-chunk-input",
      0,
      0,
      memory.inputBufferBytes,
    ));
    const output = arena.binding(arena.slice(
      "ace-vae-chunk-output",
      1,
      0,
      memory.outputBufferBytes,
    ));
    const workspaces = [0, 1, 2].map((index) =>
      arena!.binding(arena!.slice(
        `ace-vae-chunk-workspace-${index}`,
        index + 2,
        0,
        memory.workspaceBufferBytes,
      ))
    ) as [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding];
    decoder = AceCorrectnessVaeDecoderRuntime.create(options.device);
    const decoderDispatches = new Map<number, AceVaeDecoderDispatch>();
    for (const inputFrames of memory.uniqueDecoderInputFrames) {
      options.signal?.throwIfAborted();
      const dispatch = await decoder.createDecoderDispatch(
        `ace-vae-window-${inputFrames}`,
        inputFrames,
        { input, output, workspaces, tensors },
        graph.config,
        1,
        { quantumWorkPolicy },
      );
      decoderDispatches.set(inputFrames, dispatch);
    }
    options.signal?.throwIfAborted();
    let destroyed = false;
    const stableArena = arena!;
    const stableReadback = readback!;
    const stableDecoder = decoder!;
    const resources: AceVaePreparedChunkGpuResources = Object.freeze({
      device: options.device,
      plan: options.plan,
      finalLatents: latentSnapshot,
      input,
      output,
      readback: stableReadback,
      decoderDispatches: Object.freeze(decoderDispatches),
      memory,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.onProgress === undefined
        ? {}
        : { onProgress: options.onProgress }),
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        stableDecoder.destroy();
        stableReadback.destroy();
        stableArena.destroy();
        weights.destroy();
      },
    });
    resourcesPublished = true;
    return resources;
  } finally {
    if (!resourcesPublished) {
      decoder?.destroy();
      readback?.destroy();
      arena?.destroy();
      weights.destroy();
    }
  }
}

function transposePartGeometry(
  graph: ReturnType<typeof planAceVaeDecoder>,
  tensors: Readonly<Record<string, readonly {
    readonly partStart: number;
    readonly partEnd: number;
  }[]>>,
): Readonly<Record<string, readonly AceVaeTransposePartGeometry[]>> {
  return Object.freeze(Object.fromEntries(
    graph.operations
      .filter((operation) => operation.kind === "conv-transpose1d")
      .map((operation) => [
        operation.weight,
        Object.freeze(tensors[operation.weight]!.map(({ partStart, partEnd }) =>
          Object.freeze({ partStart, partEnd })
        )),
      ]),
  ));
}

function validatePreparedResources(
  resources: AceVaePreparedChunkGpuResources,
  decoderQuantaPerCommandBuffer: number,
): void {
  if (
    resources.memory.decoderQuantaPerCommandBuffer !==
      decoderQuantaPerCommandBuffer
  ) {
    throw new Error(
      "ACE VAE backend memory plan does not match decoder command batching",
    );
  }
  const expectedMaximumDecoderCommandBuffers = Math.ceil(
    resources.memory.maximumDecoderQuantaPerWindow /
      decoderQuantaPerCommandBuffer,
  );
  if (
    resources.memory.maximumDecoderCommandBuffersPerWindow !==
      expectedMaximumDecoderCommandBuffers ||
    resources.memory.maximumCommandBuffersPerWindow !==
      expectedMaximumDecoderCommandBuffers + 1
  ) {
    throw new Error(
      "ACE VAE backend memory plan has inconsistent command-buffer bounds",
    );
  }
  const inputBytes = resources.plan.decoderWorkspacePlan.inputElements *
    FLOAT32_BYTES;
  requireAceBindingBytes(resources.input, inputBytes, "ACE VAE backend input");
  requireAceBindingBytes(
    resources.output,
    resources.plan.maximumDecodedFloat32Bytes,
    "ACE VAE backend output",
  );
  if (resources.readback.size < resources.plan.maximumDecodedFloat32Bytes) {
    throw new RangeError("ACE VAE backend readback is smaller than one window");
  }
  const expectedFrames = new Set(
    resources.plan.windows.map((window) => window.latentWindowFrames),
  );
  if (
    resources.decoderDispatches.size !== expectedFrames.size ||
    [...expectedFrames].some((frames) =>
      !resources.decoderDispatches.has(frames))
  ) {
    throw new Error("ACE VAE backend decoder-shape cache is incomplete");
  }
  for (const [frames, dispatch] of resources.decoderDispatches) {
    if (
      !expectedFrames.has(frames) ||
      dispatch.plan.inputFrames !== frames ||
      dispatch.quanta.length === 0
    ) {
      throw new Error("ACE VAE backend decoder-shape cache is inconsistent");
    }
  }
  if (typeof resources.destroy !== "function") {
    throw new TypeError("ACE VAE backend resources require explicit cleanup");
  }
}

function snapshotLatents(
  plan: AceVaeChunkedDecodePlan,
  latents: Float32Array,
): Float32Array<ArrayBuffer> {
  validateLatents(plan, latents);
  const snapshot = new Float32Array(latents.length);
  snapshot.set(latents);
  return snapshot;
}

function requireOwnedLatentSnapshot(
  plan: AceVaeChunkedDecodePlan,
  latents: Float32Array,
): Float32Array<ArrayBuffer> {
  validateLatents(plan, latents);
  if (!(latents.buffer instanceof ArrayBuffer)) {
    throw new TypeError("ACE VAE owned latent snapshot must use an ArrayBuffer");
  }
  return latents as Float32Array<ArrayBuffer>;
}

function validateLatents(
  plan: AceVaeChunkedDecodePlan,
  latents: Float32Array,
): void {
  const expected = plan.latentFrames *
    plan.decoderWorkspacePlan.config.decoderInputChannels;
  if (!(latents instanceof Float32Array) || latents.length !== expected) {
    throw new RangeError(
      `ACE VAE final latent has ${latents.length} elements; expected ${expected}`,
    );
  }
  for (let index = 0; index < latents.length; index += 1) {
    const value = latents[index]!;
    if (!Number.isFinite(value)) {
      throw new Error(`ACE VAE final latent is non-finite at ${index}`);
    }
  }
}

function requirePlanWindow(
  plan: AceVaeChunkedDecodePlan,
  window: AceVaeDecodeWindow,
): void {
  const expected = plan.windows[window.index];
  if (expected === undefined) {
    throw new RangeError(`ACE VAE window index ${window.index} is outside the plan`);
  }
  for (const key of Object.keys(expected) as Array<keyof AceVaeDecodeWindow>) {
    if (window[key] !== expected[key]) {
      throw new Error(
        `ACE VAE window ${window.index} field ${key} does not match the fixed plan`,
      );
    }
  }
}

interface DecoderQuantumBatch {
  readonly index: number;
  readonly firstQuantumIndex: number;
  readonly quantumCount: number;
}

function planDecoderQuantumBatches(
  quantumCount: number,
  maximumQuantaPerCommandBuffer: number,
): readonly DecoderQuantumBatch[] {
  requirePositiveSafeInteger(quantumCount, "ACE VAE decoder quantum count");
  requirePositiveSafeInteger(
    maximumQuantaPerCommandBuffer,
    "ACE VAE decoder quanta per command buffer",
  );
  const batchCount = Math.ceil(quantumCount / maximumQuantaPerCommandBuffer);
  return Object.freeze(Array.from({ length: batchCount }, (_, index) => {
    const firstQuantumIndex = index * maximumQuantaPerCommandBuffer;
    return Object.freeze({
      index,
      firstQuantumIndex,
      quantumCount: Math.min(
        maximumQuantaPerCommandBuffer,
        quantumCount - firstQuantumIndex,
      ),
    });
  }));
}

function encodeDecoderQuantumBatch(
  device: GPUDevice,
  window: AceVaeDecodeWindow,
  quanta: readonly AceVaeDecoderQuantum[],
  batch: DecoderQuantumBatch,
  signal: AbortSignal,
): GPUCommandBuffer {
  if (
    batch.firstQuantumIndex < 0 ||
    batch.quantumCount < 1 ||
    batch.firstQuantumIndex + batch.quantumCount > quanta.length
  ) {
    throw new RangeError("ACE VAE decoder batch is outside the quantum list");
  }
  const encoder = device.createCommandEncoder({
    label: `ace-vae-window-${window.index}-batch-${batch.index}`,
  });
  const end = batch.firstQuantumIndex + batch.quantumCount;
  for (
    let quantumIndex = batch.firstQuantumIndex;
    quantumIndex < end;
    quantumIndex += 1
  ) {
    signal.throwIfAborted();
    const quantum = quanta[quantumIndex]!;
    const pass = encoder.beginComputePass({
      label: `ace-vae-window-${window.index}-batch-${batch.index}-${quantum.id}`,
    });
    quantum.encode(pass);
    pass.end();
  }
  signal.throwIfAborted();
  return encoder.finish();
}

function encodeReadbackCopy(
  device: GPUDevice,
  output: GPUBufferBinding,
  readback: GPUBuffer,
  bytes: number,
  windowIndex: number,
): GPUCommandBuffer {
  requireAceBindingBytes(output, bytes, `ACE VAE window ${windowIndex} output`);
  if (readback.size < bytes) {
    throw new RangeError(`ACE VAE window ${windowIndex} exceeds readback capacity`);
  }
  const encoder = device.createCommandEncoder({
    label: `ace-vae-window-${windowIndex}-readback`,
  });
  encoder.copyBufferToBuffer(
    output.buffer,
    output.offset ?? 0,
    readback,
    0,
    bytes,
  );
  return encoder.finish();
}

async function mapDetachedWindow(
  readback: GPUBuffer,
  bytes: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  signal.throwIfAborted();
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    signal.throwIfAborted();
    return Float32Array.from(
      new Float32Array(readback.getMappedRange(0, bytes)),
    );
  } finally {
    readback.unmap();
  }
}

function combineSignals(
  signals: readonly (AbortSignal | undefined)[],
): AbortSignal {
  const present = signals.filter((signal): signal is AbortSignal =>
    signal !== undefined
  );
  return present.length === 1 ? present[0]! : AbortSignal.any(present);
}

function align(value: number, alignment: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError("ACE VAE aligned byte count must be positive");
  }
  const aligned = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(aligned)) {
    throw new RangeError("ACE VAE aligned byte count exceeds safe integers");
  }
  return aligned;
}

function checkedProduct(
  values: readonly number[],
  label: string,
  allowZero = false,
): number {
  let result = 1;
  for (const value of values) {
    if (
      !Number.isSafeInteger(value) ||
      value < 0 ||
      (!allowZero && value === 0)
    ) {
      throw new RangeError(`${label} contains an invalid value`);
    }
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}

function checkedSum(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) ||
    !Number.isSafeInteger(right) ||
    left < 0 ||
    right < 0 ||
    !Number.isSafeInteger(left + right)
  ) {
    throw new RangeError(`${label} exceeds safe integer arithmetic`);
  }
  return left + right;
}

function destroyedError(): DOMException {
  return new DOMException("ACE VAE chunk GPU backend was destroyed", "AbortError");
}
