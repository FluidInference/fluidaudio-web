// Verify the tarball outside the monorepo so workspace dependencies cannot hide missing files.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "vite";

const consumer = mkdtempSync(join(tmpdir(), "fluidaudio-sdk-test-"));
try {
  const packed = JSON.parse(
    execFileSync("npm", ["pack", "--json", "--pack-destination", consumer], {
      cwd: resolve("dist-sdk"),
      encoding: "utf8",
    }),
  );
  writeFileSync(join(consumer, "package.json"), JSON.stringify({ private: true, type: "module" }));
  execFileSync("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund", join(consumer, packed[0].filename)], {
    cwd: consumer,
    stdio: "inherit",
  });
  const manifest = JSON.parse(readFileSync(join(consumer, "node_modules/@fluidinference/fluidaudio-web/package.json"), "utf8"));
  // Both entry points must resolve only complete engines. Keep every exported subpath reachable.
  const specifiers = Object.keys(manifest.exports).filter((key) => key !== "./package.json");
  const source = specifiers
    .map((key, i) => `import * as m${i} from ${JSON.stringify("@fluidinference/fluidaudio-web" + (key === "." ? "" : key.slice(1)))};\nexport { m${i} };`)
    .join("\n");
  const entry = join(consumer, "consumer.js");
  writeFileSync(entry, source);
  await build({
    root: consumer,
    configFile: false,
    logLevel: "warn",
    build: { outDir: "browser", target: "es2022", lib: { entry, formats: ["es"] } },
    worker: { format: "es" },
  });
  // Bundle JSON imports for Node, then instantiate the real engines without downloading weights.
  await build({
    root: consumer,
    configFile: false,
    logLevel: "warn",
    ssr: { noExternal: ["@fluidinference/fluidaudio-web"] },
    build: { ssr: entry, outDir: "server" },
  });
  const modules = await import(pathToFileURL(join(consumer, "server/consumer.js")).href);
  const expected = ["vad-silero", "asr-parakeet", "asr-whisper", "diarization-sortformer", "tts-kokoro-en", "tts-kokoro-zh", "asr-nemotron", "eou-parakeet"];
  const rootRegistry = modules.m0.ENGINES;
  assert.deepEqual(
    rootRegistry.map((entry) => entry.id),
    expected,
  );
  const registryModule = modules[`m${specifiers.indexOf("./registry")}`];
  assert.deepEqual(
    registryModule.ENGINES.map((entry) => entry.id),
    expected,
  );
  for (const entry of rootRegistry) {
    const engine = await entry.make();
    await engine.dispose();
  }
  console.log(`SDK tarball: browser build and ${expected.length} real engine factories passed.`);
} finally {
  rmSync(consumer, { recursive: true, force: true });
}
