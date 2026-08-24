// Common contracts every engine implements. The demo UI and any downstream
// consumer only ever touch these interfaces, never a specific runtime.

/** 16 kHz mono float PCM in [-1, 1] unless an engine documents otherwise. */
export interface AudioData {
  samples: Float32Array;
  sampleRate: number;
}

export interface LoadProgress {
  /** File or component currently loading. */
  file: string;
  /** Bytes fetched so far / total (total may be 0 if unknown). */
  loaded: number;
  total: number;
  /** 0..1 overall, best-effort. */
  fraction: number;
}

export type ProgressCb = (p: LoadProgress) => void;

/** Preferred execution backend; engines may downgrade if unsupported. */
export type Backend = "webgpu" | "wasm";

export interface Engine {
  readonly id: string;
  readonly label: string;
  /** Fetch + compile models. Idempotent. */
  load(onProgress?: ProgressCb): Promise<void>;
  dispose(): Promise<void>;
}

// ---- ASR ----

export interface AsrSegment {
  text: string;
  start: number;
  end: number;
}

export interface AsrStageMetrics {
  melMs: number;
  encodeMs: number; // NOTE: pipelined engines (Parakeet) report the UNHIDDEN GPU wait, not total encode time — see engines/asr-parakeet/pipeline.js
  decodeMs: number;
  totalMs: number;
}

export interface AsrResult {
  text: string;
  segments?: AsrSegment[];
  /** Per-stage timings, when the engine exposes them. */
  metrics?: AsrStageMetrics;
}

/** Batch-transcription progress, emitted at engine window/slice boundaries. */
export interface TranscribeProgress {
  /** Audio seconds whose transcription work has completed. */
  processedSeconds: number;
  /** Duration of the input clip in seconds. */
  totalSeconds: number;
  /** 0..1, monotonic; exactly 1 only on the final emit. */
  fraction: number;
}

export interface TranscribeOpts {
  /** Progress at the engine's natural window boundary (~15s–4min of audio per
   * emit depending on the engine). Best-effort: engines that cannot estimate
   * progress simply never call it. */
  onProgress?: (p: TranscribeProgress) => void;
}

export interface AsrEngine extends Engine {
  /** Optional: custom-vocabulary fuzzy correction (Parakeet). */
  setVocabulary?(terms: Array<string | { text: string; aliases?: string[]; minSimilarity?: number }>): void;
  /** Optional: opt-in inverse text normalization on transcripts (Parakeet). */
  setItn?(enabled: boolean): void;
  transcribe(audio: AudioData, opts?: TranscribeOpts): Promise<AsrResult>;
}

/** Streaming ASR (Nemotron, EOU): push chunks, get incremental text. */
export interface StreamingAsrEngine extends Engine {
  /** Feed one chunk (engine-defined frame size). Returns text emitted so far. */
  push(chunk: Float32Array): Promise<string>;
  /** Flush the tail (right-padded final frames) and return the final text.
   * After finish(), push() throws until reset(). */
  finish(): Promise<string>;
  /** Clears decoder + encoder caches for a new utterance. */
  reset(): void;
}

// ---- Source separation ----

/** One separated stem; `samples` is the left/mono channel, `right` when stereo. */
export interface StemAudio {
  name: string;
  samples: Float32Array;
  right?: Float32Array;
  sampleRate: number;
}

/** Full-band separation input: `samples` is the left/mono channel at the clip's
 * native rate (NOT the 16 kHz ASR contract), `right` when stereo. */
export interface SeparationInput extends AudioData {
  right?: Float32Array;
}

export interface SeparateOpts {
  /** Model chunk-boundary progress, same semantics as TranscribeOpts.onProgress. */
  onProgress?: (p: TranscribeProgress) => void;
}

/** Audio → multi-audio engines (stem splitters). */
export interface SeparationEngine extends Engine {
  /** Decode an encoded audio file to full-band PCM (stereo and native rate
   * preserved where the engine can) — separation must not run through the
   * shared 16 kHz mono decode path. */
  decodeFile(input: ArrayBuffer): Promise<SeparationInput>;
  separate(audio: SeparationInput, opts?: SeparateOpts): Promise<StemAudio[]>;
}

// ---- Diarization ----

export interface DiarSegment {
  speaker: number;
  start: number;
  end: number;
}

export interface DiarizationEngine extends Engine {
  diarize(audio: AudioData, opts?: { numSpeakers?: number }): Promise<DiarSegment[]>;
}

// ---- TTS ----

export interface TtsEngine extends Engine {
  synthesize(text: string, opts?: { voice?: string; speed?: number }): Promise<AudioData>;
  voices(): Promise<string[]>;
}

// ---- VAD ----

export interface SpeechRange {
  start: number;
  end: number;
}

export interface VadEngine extends Engine {
  detect(audio: AudioData): Promise<SpeechRange[]>;
}
