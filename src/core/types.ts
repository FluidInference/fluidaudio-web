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

export interface AsrEngine extends Engine {
  /** Optional: custom-vocabulary fuzzy correction (Parakeet). */
  setVocabulary?(terms: Array<string | { text: string; aliases?: string[]; minSimilarity?: number }>): void;
  /** Optional: opt-in inverse text normalization on transcripts (Parakeet). */
  setItn?(enabled: boolean): void;
  transcribe(audio: AudioData): Promise<AsrResult>;
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
