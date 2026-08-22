import { releaseAceAudioOutput } from "../../src/runtime/audio-output.js";

interface WorkerSuccess {
  readonly type: "passed";
  readonly audio: Blob;
  readonly wavBytes: number;
  readonly transactionId: string;
}

interface WorkerFailure {
  readonly type: "failed";
  readonly message: string;
}

const resultNode = document.querySelector<HTMLPreElement>("#result");
if (resultNode === null) throw new Error("missing result node");

const worker = new Worker(
  new URL("./audio-output-worker.ts", import.meta.url),
  { type: "module" },
);
worker.addEventListener("message", (event: MessageEvent<WorkerSuccess | WorkerFailure>) => {
  if (event.data.type === "failed") {
    finish("failed", event.data.message);
    worker.terminate();
    return;
  }
  const { audio, wavBytes, transactionId } = event.data;
  worker.terminate();
  void verifyAfterWorkerTermination(audio, wavBytes, transactionId);
});
worker.addEventListener("error", (event) => {
  finish("failed", event.message);
  worker.terminate();
});
worker.postMessage({ type: "run" });

async function verifyAfterWorkerTermination(
  audio: Blob,
  wavBytes: number,
  transactionId: string,
): Promise<void> {
  try {
    const playable = document.createElement("audio").canPlayType("audio/wav");
    const bytes = new Uint8Array(await audio.arrayBuffer());
    const header = new TextDecoder().decode(bytes.subarray(0, 4));
    if (
      !(audio instanceof Blob) ||
      audio.type !== "audio/wav" ||
      audio.size !== wavBytes ||
      playable === "" ||
      header !== "RIFF"
    ) {
      throw new Error(JSON.stringify({
        isBlob: audio instanceof Blob,
        type: audio.type,
        size: audio.size,
        wavBytes,
        playable,
        header,
      }));
    }
    await releaseAceAudioOutput(transactionId);
    finish("passed", JSON.stringify({
      type: audio.type,
      size: audio.size,
      playable,
      header,
      readableAfterWorkerTermination: true,
      released: true,
    }));
  } catch (error) {
    finish("failed", error instanceof Error ? error.stack ?? error.message : String(error));
  }
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode!.textContent = message;
}
