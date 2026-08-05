export interface KokoroBackend {
  backend: string;
  synthFromPhonemes(phonemes: string, voice?: string, speed?: number): Promise<Float32Array>;
}

export function loadKokoroBackend(
  fetchCached: (url: string, onProgress?: unknown, name?: string) => Promise<Uint8Array>,
  hfUrl: (repo: string, path: string) => string,
  vocab: Record<string, number>,
  opts?: { modelDir?: string; voiceRepo?: string; onProgress?: unknown },
): Promise<KokoroBackend>;
