// Nemotron inference in a Web Worker. The int4 encoder is correct on WASM but slow
// single-threaded, and its streaming RNNT decode is thousands of tiny sequential
// calls — on the main thread that freezes the page. Running the whole load +
// transcribe here keeps the UI responsive (one message in, one result out).

import { configureOrt, ort } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import { JsPreprocessor } from "./nemotron-mel.js";
import { nemotronTranscribe, makeNemotronTokenizer, makeNemotronLangMap } from "./nemotron-decode.js";

const REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";

async function createWithData(name: string) {
  const model = await fetchCached(hfUrl(REPO, `${name}.onnx`), undefined, `${name}.onnx`);
  const data = await fetchCached(hfUrl(REPO, `${name}.onnx.data`), undefined, `${name}.onnx.data`);
  configureOrt();
  return ort.InferenceSession.create(model, {
    executionProviders: ["wasm"], // int4 is healthy on WASM; WebGPU EP mishandles it
    graphOptimizationLevel: "all",
    externalData: [{ path: `${name}.onnx.data`, data }] as any,
  });
}

let state: any = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, id, audio, language } = e.data;
  const post = (msg: any) => (self as any).postMessage({ id, ...msg });
  try {
    if (type === "load") {
      const encoder = await createWithData("encoder");
      const decoder = await createWithData("decoder");
      const joint = await createWithData("joint");
      const vocab = new TextDecoder().decode(await fetchCached(hfUrl(REPO, "vocab.txt"), undefined, "vocab.txt"));
      state = {
        encoder, decoder, joint,
        preprocessor: new JsPreprocessor({ nMels: 128 }),
        tokenizer: makeNemotronTokenizer(vocab),
        langMap: makeNemotronLangMap(vocab),
      };
      post({ ok: true });
    } else if (type === "transcribe") {
      const langId = state.langMap[language ?? "en-US"] ?? state.langMap["en"] ?? 24;
      const { text } = await nemotronTranscribe({
        ort,
        encoder: state.encoder, decoder: state.decoder, joint: state.joint,
        preprocessor: state.preprocessor, tokenizer: state.tokenizer,
        audio: new Float32Array(audio), langId,
      });
      post({ ok: true, text });
    }
  } catch (err) {
    post({ ok: false, error: String(err) });
  }
};
