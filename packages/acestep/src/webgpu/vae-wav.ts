import type {
  AceVaeChunkedDecodePlan,
  AceVaeDecodeWindow,
  AceVaePostprocessPlan,
  AceVaeRawChunkSink,
} from "./vae-chunks.js";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const FLOAT_WAV_HEADER_BYTES = 44;
const RIFF_MAX_U32 = 0xffff_ffff;
const HOST_IS_LITTLE_ENDIAN = new Uint8Array(
  new Uint16Array([0x0102]).buffer,
)[0] === 0x02;

/** Structural subset implemented by worker-only OPFS sync access handles. */
export interface AceSeekableFile {
  write(buffer: ArrayBufferView, options: { readonly at: number }): number;
  read(buffer: ArrayBufferView, options: { readonly at: number }): number;
  truncate(newSize: number): void;
  flush(): void;
}

/**
 * First-pass sink for raw interleaved FP32 decoder cores.
 *
 * Use a temporary OPFS sync access handle in the inference worker. The sink
 * never retains a core after `writeCore` returns and writes exact f32-le bytes.
 */
export class AceVaeRawF32FileSink implements AceVaeRawChunkSink {
  private nextElement = 0;
  private finished = false;

  constructor(
    private readonly file: AceSeekableFile,
    private readonly plan: AceVaeChunkedDecodePlan,
  ) {
    file.truncate(0);
  }

  async writeCore(
    window: AceVaeDecodeWindow,
    interleavedCore: Float32Array,
  ): Promise<void> {
    if (this.finished) {
      throw new Error("ACE VAE raw sink was already finalized");
    }
    const expectedElement =
      window.outputStartAudioFrame * this.plan.audioChannels;
    if (expectedElement !== this.nextElement) {
      throw new Error(
        `ACE VAE raw sink expected element ${this.nextElement}, got ${expectedElement}`,
      );
    }
    const expectedElements =
      window.outputAudioFrames * this.plan.audioChannels;
    if (interleavedCore.length !== expectedElements) {
      throw new RangeError(
        `ACE VAE raw core has ${interleavedCore.length} elements; expected ${expectedElements}`,
      );
    }
    const bytes = float32LittleEndianBytes(interleavedCore);
    writeExact(this.file, bytes, this.nextElement * FLOAT32_BYTES);
    this.nextElement += interleavedCore.length;
  }

  finish(): void {
    if (this.finished) return;
    if (this.nextElement !== this.plan.outputInterleavedElements) {
      throw new Error(
        `ACE VAE raw sink contains ${this.nextElement} elements; expected ${this.plan.outputInterleavedElements}`,
      );
    }
    this.file.truncate(this.plan.outputFloat32Bytes);
    this.file.flush();
    this.finished = true;
  }
}

export interface AceVaeWavWriteResult {
  readonly headerBytes: 44;
  readonly dataBytes: number;
  readonly wavBytes: number;
  readonly outputPeak: number;
}

export interface AceVaeWavCooperativeWriteOptions {
  /** Number of audio frames converted by one bounded synchronous block. */
  readonly blockAudioFrames?: number;
  /** Check cancellation before every block and after every event-loop yield. */
  readonly signal?: AbortSignal;
  /**
   * Yield after this many non-final blocks. Keeping this greater than one
   * avoids thousands of timer tasks for a long song while bounding how long
   * the worker can delay delivery of a cancellation message.
   */
  readonly yieldEveryBlocks?: number;
  /** @internal Deterministic test seam; production uses a real timer task. */
  readonly yieldToEventLoop?: () => Promise<void>;
  /** @internal Capture-only finalization attribution. */
  readonly onTrace?: (trace: AceVaeWavWriteTrace) => void;
  /** @internal Deterministic capture clock. */
  readonly now?: () => number;
}

export interface AceVaeWavWriteTrace {
  readonly schema: "ace-vae-wav-write-capture-v1";
  readonly startedAtMs: number;
  readonly completedAtMs: number;
  readonly wallMs: number;
  readonly blockAudioFrames: number;
  readonly blockCount: number;
  readonly yieldCount: number;
  readonly rawReadBytes: number;
  readonly wavDataBytes: number;
  readonly maximumReadBufferBytes: number;
  readonly maximumScaledBufferBytes: number;
  readonly timing: Readonly<{
    readonly headerWriteMs: number;
    readonly rawReadMs: number;
    readonly normalizeAndEncodeMs: number;
    readonly wavWriteMs: number;
    readonly cooperativeYieldMs: number;
    readonly truncateAndFlushMs: number;
  }>;
}

/**
 * Second bounded pass: read raw OPFS f32-le blocks, apply upstream's safety
 * division and normalization multiplication as two FP32 operations, and stream the canonical
 * stereo WAVE_FORMAT_IEEE_FLOAT container to another seekable file. No
 * full-song ArrayBuffer or duplicate waveform is created.
 */
export function writeNormalizedAceVaeFloat32Wav(
  rawFile: AceSeekableFile,
  wavFile: AceSeekableFile,
  plan: AceVaeChunkedDecodePlan,
  postprocess: AceVaePostprocessPlan,
  blockAudioFrames = 16_384,
): AceVaeWavWriteResult {
  const writer = new NormalizedAceVaeWavWriter(
    rawFile,
    wavFile,
    plan,
    postprocess,
    blockAudioFrames,
  );
  while (writer.hasRemainingBlocks) writer.writeNextBlock();
  return writer.finish();
}

/**
 * Cooperative form of the bounded second pass used by the worker pipeline.
 *
 * Each conversion block performs the exact same scalar FP32 operations as the
 * synchronous reference function. Periodic timer tasks let the worker receive
 * an abort message instead of remaining monopolized for the entire song.
 */
export async function writeNormalizedAceVaeFloat32WavCooperatively(
  rawFile: AceSeekableFile,
  wavFile: AceSeekableFile,
  plan: AceVaeChunkedDecodePlan,
  postprocess: AceVaePostprocessPlan,
  options: AceVaeWavCooperativeWriteOptions = {},
): Promise<AceVaeWavWriteResult> {
  const {
    blockAudioFrames = 16_384,
    signal,
    yieldEveryBlocks = 8,
    yieldToEventLoop = yieldTimerTask,
    onTrace,
    now = defaultNow,
  } = options;
  if (!Number.isSafeInteger(yieldEveryBlocks) || yieldEveryBlocks <= 0) {
    throw new RangeError(
      "ACE VAE WAV yield interval must be a positive safe integer",
    );
  }
  signal?.throwIfAborted();
  const capture = onTrace !== undefined;
  const startedAtMs = capture ? now() : 0;
  const writer = new NormalizedAceVaeWavWriter(
    rawFile,
    wavFile,
    plan,
    postprocess,
    blockAudioFrames,
    capture ? now : undefined,
  );
  let blocksSinceYield = 0;
  let cooperativeYieldMs = 0;
  let yieldCount = 0;
  while (writer.hasRemainingBlocks) {
    signal?.throwIfAborted();
    writer.writeNextBlock();
    blocksSinceYield += 1;
    if (
      writer.hasRemainingBlocks &&
      blocksSinceYield === yieldEveryBlocks
    ) {
      blocksSinceYield = 0;
      const yieldStartedAt = capture ? now() : 0;
      await yieldToEventLoop();
      if (capture) {
        cooperativeYieldMs += nonnegativeElapsed(now(), yieldStartedAt);
        yieldCount += 1;
      }
      signal?.throwIfAborted();
    }
  }
  signal?.throwIfAborted();
  const result = writer.finish();
  if (capture) {
    const completedAtMs = now();
    const timing = writer.captureTiming;
    try {
      onTrace(Object.freeze({
        schema: "ace-vae-wav-write-capture-v1",
        startedAtMs,
        completedAtMs,
        wallMs: nonnegativeElapsed(completedAtMs, startedAtMs),
        blockAudioFrames,
        blockCount: writer.completedBlockCount,
        yieldCount,
        rawReadBytes: result.dataBytes,
        wavDataBytes: result.dataBytes,
        maximumReadBufferBytes: writer.maximumBlockBytes,
        maximumScaledBufferBytes: writer.maximumBlockBytes,
        timing: Object.freeze({
          ...timing,
          cooperativeYieldMs,
        }),
      }));
    } catch {
      // Capture is observational and cannot alter a committed WAV.
    }
  }
  return result;
}

class NormalizedAceVaeWavWriter {
  private readonly dataBytes: number;
  private readonly wavBytes: number;
  private readonly blockElements: number;
  private readonly blockBytes: Uint8Array;
  private readonly scaled: Float32Array;
  private elementOffset = 0;
  private outputPeak = 0;
  private finished = false;
  private blockCount = 0;
  private headerWriteMs = 0;
  private rawReadMs = 0;
  private normalizeAndEncodeMs = 0;
  private wavWriteMs = 0;
  private truncateAndFlushMs = 0;

  constructor(
    private readonly rawFile: AceSeekableFile,
    private readonly wavFile: AceSeekableFile,
    private readonly plan: AceVaeChunkedDecodePlan,
    private readonly postprocess: AceVaePostprocessPlan,
    blockAudioFrames: number,
    private readonly now?: () => number,
  ) {
    if (!Number.isSafeInteger(blockAudioFrames) || blockAudioFrames <= 0) {
      throw new RangeError(
        "ACE VAE WAV block size must be a positive safe integer",
      );
    }
    this.dataBytes = plan.outputFloat32Bytes;
    const riffSize = 36 + this.dataBytes;
    this.wavBytes = FLOAT_WAV_HEADER_BYTES + this.dataBytes;
    if (riffSize > RIFF_MAX_U32 || this.dataBytes > RIFF_MAX_U32) {
      throw new RangeError("ACE VAE float WAV exceeds the RIFF-32 size domain");
    }
    this.blockElements = blockAudioFrames * plan.audioChannels;
    if (!Number.isSafeInteger(this.blockElements)) {
      throw new RangeError("ACE VAE WAV block geometry exceeds safe integers");
    }
    this.blockBytes = new Uint8Array(this.blockElements * FLOAT32_BYTES);
    this.scaled = new Float32Array(this.blockElements);
    const headerStartedAt = this.captureNow();
    wavFile.truncate(0);
    writeExact(
      wavFile,
      createAceFloat32WavHeader(
        plan.outputAudioFrames,
        plan.audioChannels,
        plan.sampleRateHz,
      ),
      0,
    );
    this.headerWriteMs += this.captureElapsed(headerStartedAt);
  }

  get hasRemainingBlocks(): boolean {
    return this.elementOffset < this.plan.outputInterleavedElements;
  }

  get completedBlockCount(): number {
    return this.blockCount;
  }

  get maximumBlockBytes(): number {
    return this.blockBytes.byteLength;
  }

  get captureTiming(): Readonly<{
    readonly headerWriteMs: number;
    readonly rawReadMs: number;
    readonly normalizeAndEncodeMs: number;
    readonly wavWriteMs: number;
    readonly truncateAndFlushMs: number;
  }> {
    return Object.freeze({
      headerWriteMs: this.headerWriteMs,
      rawReadMs: this.rawReadMs,
      normalizeAndEncodeMs: this.normalizeAndEncodeMs,
      wavWriteMs: this.wavWriteMs,
      truncateAndFlushMs: this.truncateAndFlushMs,
    });
  }

  writeNextBlock(): void {
    if (this.finished || !this.hasRemainingBlocks) {
      throw new Error("ACE VAE WAV writer has no remaining block");
    }
    const elements = Math.min(
      this.blockElements,
      this.plan.outputInterleavedElements - this.elementOffset,
    );
    const byteCount = elements * FLOAT32_BYTES;
    const inputBytes = this.blockBytes.subarray(0, byteCount);
    const readStartedAt = this.captureNow();
    readExact(
      this.rawFile,
      inputBytes,
      this.elementOffset * FLOAT32_BYTES,
    );
    this.rawReadMs += this.captureElapsed(readStartedAt);
    const view = new DataView(
      inputBytes.buffer,
      inputBytes.byteOffset,
      inputBytes.byteLength,
    );
    const normalizeStartedAt = this.captureNow();
    for (let index = 0; index < elements; index += 1) {
      const raw = view.getFloat32(index * FLOAT32_BYTES, true);
      if (!Number.isFinite(raw)) {
        throw new Error("ACE VAE raw store contains a non-finite sample");
      }
      // Preserve the upstream FP32 operations independently: elementwise
      // generation safety division first, then inference.py multiplication.
      const safe = Math.fround(raw / this.postprocess.safetyDivisor);
      const value = Math.fround(
        safe * this.postprocess.normalizationScale,
      );
      if (!Number.isFinite(value)) {
        throw new Error("ACE VAE output scaling produced a non-finite sample");
      }
      this.scaled[index] = value;
      this.outputPeak = Math.max(this.outputPeak, Math.abs(value));
    }
    this.normalizeAndEncodeMs += this.captureElapsed(normalizeStartedAt);
    const writeStartedAt = this.captureNow();
    writeExact(
      this.wavFile,
      float32LittleEndianBytes(this.scaled.subarray(0, elements)),
      FLOAT_WAV_HEADER_BYTES + this.elementOffset * FLOAT32_BYTES,
    );
    this.wavWriteMs += this.captureElapsed(writeStartedAt);
    this.elementOffset += elements;
    this.blockCount += 1;
  }

  finish(): AceVaeWavWriteResult {
    if (this.finished) throw new Error("ACE VAE WAV writer was already finalized");
    if (this.hasRemainingBlocks) {
      throw new Error("ACE VAE WAV writer cannot finish before all blocks");
    }
    const flushStartedAt = this.captureNow();
    this.wavFile.truncate(this.wavBytes);
    this.wavFile.flush();
    this.truncateAndFlushMs += this.captureElapsed(flushStartedAt);
    this.finished = true;
    return Object.freeze({
      headerBytes: FLOAT_WAV_HEADER_BYTES,
      dataBytes: this.dataBytes,
      wavBytes: this.wavBytes,
      outputPeak: this.outputPeak,
    });
  }

  private captureNow(): number {
    return this.now?.() ?? 0;
  }

  private captureElapsed(startedAtMs: number): number {
    return this.now === undefined
      ? 0
      : nonnegativeElapsed(this.now(), startedAtMs);
  }
}

async function yieldTimerTask(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function defaultNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function nonnegativeElapsed(completedAtMs: number, startedAtMs: number): number {
  return Math.max(0, completedAtMs - startedAtMs);
}

/** Header byte-for-byte compatible with the committed Python reference tool. */
export function createAceFloat32WavHeader(
  audioFrames: number,
  channels: number,
  sampleRateHz: number,
): Uint8Array {
  for (const [label, value] of Object.entries({
    audioFrames,
    channels,
    sampleRateHz,
  })) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`ACE float WAV ${label} must be a positive integer`);
    }
  }
  if (channels > 0xffff) {
    throw new RangeError("ACE float WAV channel count exceeds u16");
  }
  const blockAlign = channels * FLOAT32_BYTES;
  const byteRate = sampleRateHz * blockAlign;
  const dataBytes = audioFrames * blockAlign;
  const riffSize = 36 + dataBytes;
  if (
    !Number.isSafeInteger(dataBytes) ||
    dataBytes > RIFF_MAX_U32 ||
    riffSize > RIFF_MAX_U32 ||
    blockAlign > 0xffff ||
    byteRate > RIFF_MAX_U32
  ) {
    throw new RangeError("ACE float WAV geometry exceeds RIFF-32 fields");
  }
  const header = new Uint8Array(FLOAT_WAV_HEADER_BYTES);
  const view = new DataView(header.buffer);
  ascii(header, 0, "RIFF");
  view.setUint32(4, riffSize, true);
  ascii(header, 8, "WAVE");
  ascii(header, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true); // WAVE_FORMAT_IEEE_FLOAT
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRateHz, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 32, true);
  ascii(header, 36, "data");
  view.setUint32(40, dataBytes, true);
  return header;
}

function float32LittleEndianBytes(values: Float32Array): Uint8Array {
  if (HOST_IS_LITTLE_ENDIAN) {
    return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  }
  const bytes = new Uint8Array(values.byteLength);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setFloat32(index * FLOAT32_BYTES, values[index]!, true);
  }
  return bytes;
}

function writeExact(
  file: AceSeekableFile,
  bytes: Uint8Array,
  fileOffset: number,
): void {
  let consumed = 0;
  while (consumed < bytes.byteLength) {
    const written = file.write(bytes.subarray(consumed), {
      at: fileOffset + consumed,
    });
    if (
      !Number.isSafeInteger(written) ||
      written <= 0 ||
      written > bytes.byteLength - consumed
    ) {
      throw new Error("ACE seekable file made no forward write progress");
    }
    consumed += written;
  }
}

function readExact(
  file: AceSeekableFile,
  bytes: Uint8Array,
  fileOffset: number,
): void {
  let consumed = 0;
  while (consumed < bytes.byteLength) {
    const read = file.read(bytes.subarray(consumed), {
      at: fileOffset + consumed,
    });
    if (
      !Number.isSafeInteger(read) ||
      read <= 0 ||
      read > bytes.byteLength - consumed
    ) {
      throw new Error("ACE seekable file ended before the expected raw audio");
    }
    consumed += read;
  }
}

function ascii(target: Uint8Array, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    target[offset + index] = value.charCodeAt(index);
  }
}
