import { subtractStereoPcm, type StereoPcm } from "./runtime/audio.js";
import { DiCoSeSeparator } from "./runtime/separator.js";
import {
  DICOSE_STEM_NAMES,
  type DiCoSePcmTransfer,
  type DiCoSeProgress,
  type DiCoSeStemName,
  type DiCoSeWorkerEvent,
  type DiCoSeWorkerInitOptions,
  type DiCoSeWorkerRequest,
  type DiCoSeWorkerResult,
} from "./worker-protocol.js";

const worker = self as DedicatedWorkerGlobalScope;

let separator: DiCoSeSeparator | undefined;
let running = Promise.resolve();
let disposed = false;
let progressRequestId: number | undefined;

worker.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (!isWorkerRequest(event.data)) return;
  const request = event.data;
  // BS-RoFormer workspaces are intentionally reused by the runtime. A serial
  // queue ensures a second caller cannot overwrite a first inference graph.
  running = running.then(
    () => handle(request),
    () => handle(request),
  );
});

async function handle(request: DiCoSeWorkerRequest): Promise<void> {
  try {
    switch (request.type) {
      case "initialize":
        await initialize(request.options, request.id);
        post({ type: "initialized", id: request.id });
        return;
      case "separate": {
        progressRequestId = request.id;
        const runtime = await initialize(undefined, request.id);
        postProgress(request.id, {
          phase: "separating",
          detail: request.options.outputMode === "deterministic"
            ? "Running deterministic DiCoSe BS-RoFormer"
            : "Running DiCoSe BS-RoFormer + CD",
        });
        const input = hydratePcm(request.pcm);
        const result = await runtime.separatePcm(input, request.options);
        const instrumental = subtractStereoPcm(input, result.stems.vocals);
        const serialized = serializeResult(result, instrumental);
        post({ type: "result", id: request.id, result: serialized.result }, serialized.transfer);
        return;
      }
      case "dispose":
        await separator?.dispose();
        separator = undefined;
        disposed = true;
        post({ type: "disposed", id: request.id });
        return;
      default:
        assertNever(request);
    }
  } catch (error) {
    post({ type: "error", id: request.id, error: serializeError(error) });
  }
}

async function initialize(
  options: DiCoSeWorkerInitOptions | undefined,
  requestId: number,
): Promise<DiCoSeSeparator> {
  if (disposed) throw new Error("DiCoSe worker has been disposed");
  if (separator !== undefined) return separator;
  progressRequestId = requestId;
  postProgress(requestId, { phase: "initializing", detail: "Creating WebGPU device" });
  separator = await DiCoSeSeparator.create({
    ...(options?.manifestUrl === undefined ? {} : { manifestUrl: options.manifestUrl }),
    ...(options?.attentionKernel === undefined ? {} : { attentionKernel: options.attentionKernel }),
    // The same runtime is kept for later `separate` messages. Route its
    // callback through the request currently using that runtime rather than
    // pinning it to the one-time initialization request.
    onProgress: (progress: DiCoSeProgress) => {
      if (progressRequestId !== undefined) postProgress(progressRequestId, progress);
    },
  });
  return separator;
}

function hydratePcm(transfer: DiCoSePcmTransfer): StereoPcm {
  const left = new Float32Array(transfer.left);
  const right = new Float32Array(transfer.right);
  if (
    !Number.isInteger(transfer.sampleRate) ||
    transfer.sampleRate <= 0 ||
    !Number.isSafeInteger(transfer.length) ||
    transfer.length < 0 ||
    left.length !== transfer.length ||
    right.length !== transfer.length
  ) {
    throw new RangeError("Received malformed PCM transfer");
  }
  return Object.freeze({
    sampleRate: transfer.sampleRate,
    length: transfer.length,
    left,
    right,
    channels: [left, right] as const,
  });
}

function serializeResult(
  result: Awaited<ReturnType<DiCoSeSeparator["separatePcm"]>>,
  instrumental: StereoPcm,
): {
  readonly result: DiCoSeWorkerResult;
  readonly transfer: Transferable[];
} {
  const stems = {} as Record<DiCoSeStemName, {
    sampleRate: number;
    length: number;
    left: ArrayBuffer;
    right: ArrayBuffer;
  }>;
  const transfer: Transferable[] = [];
  for (const name of DICOSE_STEM_NAMES) {
    const pcm = result.stems[name];
    if (pcm === undefined) throw new Error(`DiCoSe runtime omitted the ${name} stem`);
    if (pcm.left.length !== pcm.length || pcm.right.length !== pcm.length) {
      throw new Error(`DiCoSe runtime returned malformed ${name} PCM`);
    }
    // slice() gives the result message independently owned, transferable data
    // even if a runtime retains a larger backing allocation for scratch reuse.
    const left = pcm.left.slice();
    const right = pcm.right.slice();
    stems[name] = {
      sampleRate: pcm.sampleRate,
      length: pcm.length,
      left: left.buffer,
      right: right.buffer,
    };
    transfer.push(left.buffer, right.buffer);
  }
  if (instrumental.left.length !== instrumental.length || instrumental.right.length !== instrumental.length) {
    throw new Error("DiCoSe runtime returned malformed instrumental PCM");
  }
  // `subtractStereoPcm` allocated these arrays specifically for this result,
  // so their buffers can transfer directly without another whole-track copy.
  const instrumentalLeft = requireOwnedBuffer(instrumental.left, "instrumental left");
  const instrumentalRight = requireOwnedBuffer(instrumental.right, "instrumental right");
  transfer.push(instrumentalLeft, instrumentalRight);
  return {
    result: {
      outputMode: result.outputMode,
      stems,
      instrumental: {
        sampleRate: instrumental.sampleRate,
        length: instrumental.length,
        left: instrumentalLeft,
        right: instrumentalRight,
      },
      timing: sanitizeTiming(result.timing),
      diagnostics: sanitizeDiagnostics(result.diagnostics),
    },
    transfer,
  };
}

function requireOwnedBuffer(array: Float32Array, label: string): ArrayBuffer {
  const buffer = array.buffer;
  if (
    !(buffer instanceof ArrayBuffer) ||
    array.byteOffset !== 0 ||
    array.byteLength !== buffer.byteLength
  ) {
    throw new Error(`${label} PCM is not independently transferable`);
  }
  return buffer;
}

function sanitizeTiming(value: object): Readonly<Record<string, number>> {
  const timing: Record<string, number> = {};
  for (const [name, elapsed] of Object.entries(value)) {
    if (typeof elapsed === "number" && Number.isFinite(elapsed) && elapsed >= 0) timing[name] = elapsed;
  }
  return timing;
}

function sanitizeDiagnostics(
  value: Awaited<ReturnType<DiCoSeSeparator["separatePcm"]>>["diagnostics"],
): DiCoSeWorkerResult["diagnostics"] {
  const output: Record<string, Record<DiCoSeStemName, { peak: number; rms: number }>> = {};
  for (const [stage, stems] of Object.entries(value)) {
    const stageOutput = {} as Record<DiCoSeStemName, { peak: number; rms: number }>;
    for (const name of DICOSE_STEM_NAMES) {
      const stats = stems[name];
      if (stats === undefined || !Number.isFinite(stats.peak) || !Number.isFinite(stats.rms)) {
        throw new Error(`DiCoSe runtime produced invalid ${stage}/${name} diagnostics`);
      }
      stageOutput[name] = { peak: stats.peak, rms: stats.rms };
    }
    output[stage] = stageOutput;
  }
  return output;
}

function postProgress(id: number, progress: DiCoSeProgress): void {
  post({ type: "progress", id, progress });
}

function post(message: DiCoSeWorkerEvent, transfer: Transferable[] = []): void {
  worker.postMessage(message, transfer);
}

function isWorkerRequest(value: unknown): value is DiCoSeWorkerRequest {
  if (typeof value !== "object" || value === null || !("type" in value) || !("id" in value)) return false;
  const request = value as { readonly type?: unknown; readonly id?: unknown };
  return typeof request.id === "number" && ["initialize", "separate", "dispose"].includes(String(request.type));
}

function serializeError(error: unknown): { readonly name: string; readonly message: string; readonly stack?: string } {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      ...(error.stack === undefined ? {} : { stack: error.stack }),
    };
  }
  return { name: "Error", message: String(error) };
}

function assertNever(value: never): never {
  throw new Error(`Unhandled worker request: ${JSON.stringify(value)}`);
}
