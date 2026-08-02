// End-to-end smoke test: Kokoro TTS in Node (proves the model + pipeline work
// outside the browser). The browser engine is the same model via kokoro-js.
//   node scripts/smoke-kokoro.mjs
import { KokoroTTS } from "kokoro-js";
import { writeFileSync } from "node:fs";

const text = process.argv[2] ?? "Hello from FluidAudio, running a real model end to end.";
console.log("loading Kokoro 82M (onnx-community/Kokoro-82M-v1.0-ONNX)…");
const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
  dtype: "q8",
  device: "cpu",
});
console.log("generating…");
const audio = await tts.generate(text, { voice: "af_heart" });
const wav = audio.toWav(); // ArrayBuffer, 16-bit PCM WAV
writeFileSync("/tmp/kokoro_smoke.wav", Buffer.from(wav));
console.log(`ok → /tmp/kokoro_smoke.wav  (${audio.audio.length} samples @ ${audio.sampling_rate}Hz)`);
