import {
  parseModelManifest,
  type DiCoSeModelManifest,
  type DiCoSeTensorManifest,
} from "./manifest.js";

export interface GpuWeightPackage {
  readonly manifest: DiCoSeModelManifest;
  readonly buffer: GPUBuffer;
  tensor(name: string): GpuWeightTensor;
  destroy(): void;
}

export interface GpuWeightTensor extends DiCoSeTensorManifest {
  readonly buffer: GPUBuffer;
}

export interface LoadModelProgress {
  readonly phase: "manifest" | "weights";
  readonly loadedBytes: number;
  readonly totalBytes: number;
}

/**
 * Stream package bytes straight to GPU memory. This deliberately never holds
 * the full ~625 MB inference package in JavaScript memory.
 */
export async function loadGpuWeightPackage(
  device: GPUDevice,
  manifestUrl: string | URL,
  onProgress?: (event: LoadModelProgress) => void,
): Promise<GpuWeightPackage> {
  const url = new URL(manifestUrl, globalThis.location?.href);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load model manifest: HTTP ${response.status}`);
  const manifest = parseModelManifest(await response.json());
  onProgress?.({ phase: "manifest", loadedBytes: 1, totalBytes: 1 });
  // The converter intentionally keeps a stable filename. Bind the browser
  // cache key to the manifest digest so a freshly fetched manifest can never
  // be paired with an older same-name package from HTTP cache.
  const weightsUrl = new URL(manifest.weights.file, url);
  weightsUrl.searchParams.set("sha256", manifest.weights.sha256);
  const weightResponse = await fetch(weightsUrl, { cache: "force-cache" });
  if (!weightResponse.ok || weightResponse.body === null) {
    throw new Error(`Could not load DiCoSe weights: HTTP ${weightResponse.status}`);
  }
  const reportedLengthHeader = weightResponse.headers.get("content-length");
  if (reportedLengthHeader !== null) {
    const reportedLength = Number(reportedLengthHeader);
    if (!Number.isFinite(reportedLength) || reportedLength !== manifest.weights.byteLength) {
      throw new Error("DiCoSe weight file length differs from its manifest");
    }
  }
  const buffer = device.createBuffer({
    label: "dicose-f16-weights",
    size: align4(manifest.weights.byteLength),
    // CD mapping copies four stem-embedding rows into transient tensors before
    // adding the time embedding. Without COPY_SRC Chrome invalidates the whole
    // mapping encoder and the refiner continues with unusable FiLM vectors.
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  let loaded = 0;
  let written = 0;
  let tail = new Uint8Array(0);
  const reader = weightResponse.body.getReader();
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      if (chunk.value === undefined) continue;
      const bytes = chunk.value;
      loaded += bytes.byteLength;
      const joined = join(tail, bytes);
      const writableLength = joined.byteLength - (joined.byteLength % 4);
      if (writableLength > 0) {
        const upload = new Uint8Array(writableLength);
        upload.set(joined.subarray(0, writableLength));
        device.queue.writeBuffer(buffer, written, upload.buffer);
        written += writableLength;
      }
      tail = joined.slice(writableLength);
      onProgress?.({
        phase: "weights",
        loadedBytes: loaded,
        totalBytes: manifest.weights.byteLength,
      });
    }
  } catch (error) {
    buffer.destroy();
    throw error;
  }
  if (loaded !== manifest.weights.byteLength) {
    buffer.destroy();
    throw new Error(`DiCoSe weight stream ended at ${loaded}, expected ${manifest.weights.byteLength}`);
  }
  if (tail.byteLength > 0) {
    const padded = padToFour(tail);
    device.queue.writeBuffer(buffer, written, padded.buffer);
  }
  const tensors = new Map(manifest.tensors.map((tensor) => [tensor.name, {
    ...tensor,
    buffer,
  }] as const));
  let destroyed = false;
  return Object.freeze({
    manifest,
    buffer,
    tensor(name: string): GpuWeightTensor {
      const tensor = tensors.get(name);
      if (tensor === undefined) throw new Error(`Model package does not contain ${name}`);
      return tensor;
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      buffer.destroy();
    },
  });
}

function padToFour(bytes: Uint8Array): Uint8Array {
  const padded = new Uint8Array(align4(bytes.byteLength));
  padded.set(bytes);
  return padded;
}

function join(left: Uint8Array, right: Uint8Array): Uint8Array {
  if (left.byteLength === 0) return right;
  const joined = new Uint8Array(left.byteLength + right.byteLength);
  joined.set(left);
  joined.set(right, left.byteLength);
  return joined;
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
