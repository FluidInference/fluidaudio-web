// Speech to Text page (index.html): the five ASR engines, with live mic,
// caption downloads, and the Parakeet custom-vocabulary/ITN row.
import { initPlayground } from "./pages/playground.js";

initPlayground({ category: "stt", defaultEngineId: "asr-parakeet", mic: true, captions: true, vocab: true });
