// Speed run for Kokoro TTS (kokoro-js) on Node CPU.
//   node scripts/bench-kokoro.mjs
import { KokoroTTS } from "kokoro-js";

const texts = [
  "Hello world.",
  "The quick brown fox jumps over the lazy dog while the sun sets slowly.",
  "FluidAudio runs neural text to speech entirely in your browser using WebGPU and WebAssembly, with no server and no data leaving the device. " +
    "This sentence is deliberately long to measure sustained synthesis throughput in characters per second.",
];

const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype: "q8", device: "cpu" });
await tts.generate("warm up", { voice: "af_heart" }); // warm

console.log("chars   gen_ms   audio_s   RTFx    chars/s");
for (const text of texts) {
  const t0 = performance.now();
  const audio = await tts.generate(text, { voice: "af_heart" });
  const ms = performance.now() - t0;
  const dur = audio.audio.length / audio.sampling_rate;
  console.log(
    `${String(text.length).padStart(5)}   ${ms.toFixed(0).padStart(6)}   ${dur.toFixed(2).padStart(7)}   ${(dur / (ms / 1000)).toFixed(2).padStart(5)}   ${(text.length / (ms / 1000)).toFixed(0).padStart(6)}`,
  );
}
