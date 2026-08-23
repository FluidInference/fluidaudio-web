/// <reference lib="webworker" />

import { AceAudioOutputTransaction } from "../../src/runtime/audio-output.js";
import {
  deriveAceVaePostprocessPlan,
  planAceVaeChunkedDecode,
} from "../../src/webgpu/vae-chunks.js";
import type { AceVaeDecoderConfig } from "../../src/webgpu/vae-decoder.js";

const TOY_CONFIG: AceVaeDecoderConfig = {
  id: "audio-output-browser-test",
  decoderInputChannels: 2,
  decoderChannels: 4,
  audioChannels: 2,
  channelMultiples: [1],
  downsamplingRatios: [2],
  sampleRateHz: 48_000,
};

self.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (
    typeof event.data !== "object" ||
    event.data === null ||
    (event.data as { readonly type?: unknown }).type !== "run"
  ) return;
  void run().then(
    (result) => self.postMessage({ type: "passed", ...result }),
    (error: unknown) => self.postMessage({
      type: "failed",
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    }),
  );
});

async function run(): Promise<Readonly<{
  audio: Blob;
  wavBytes: number;
  transactionId: string;
}>> {
  const plan = planAceVaeChunkedDecode(1, {
    config: TOY_CONFIG,
    chunkFrames: 2,
    overlapFrames: 0,
  });
  const transaction = await AceAudioOutputTransaction.begin(
    `browser-contract-${crypto.randomUUID()}`,
    plan,
  );
  try {
    await transaction.rawSink.writeCore(
      plan.windows[0]!,
      new Float32Array([2, -2, 0.5, -0.5]),
    );
    const committed = await transaction.commit(
      deriveAceVaePostprocessPlan(2),
    );
    return Object.freeze({
      audio: committed.audio,
      wavBytes: committed.wav.wavBytes,
      transactionId: committed.transactionId,
    });
  } catch (error) {
    try {
      await transaction.rollback();
    } catch {
      // Preserve the generation failure in this focused browser contract.
    }
    throw error;
  }
}
