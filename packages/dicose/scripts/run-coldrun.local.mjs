#!/usr/bin/env node
// Local assessment driver: run the DiCoSe demo page unattended on a caller-
// supplied WAV (served from the repository root) and print the raw report.
import { runBrowserHarness } from "./browser-harness.mjs";

const outputMode = process.env.MODE ?? "deterministic";
const sourcePath = process.env.SOURCE ?? "/cold-run.wav";

const result = await runBrowserHarness({
  label: `cold-run ${outputMode}`,
  mode: "e2e",
  warmupRuns: 0,
  measuredRuns: 1,
  timeoutMs: 20 * 60 * 1000,
  sourcePath,
  pagePath: `/?outputMode=${encodeURIComponent(outputMode)}`,
});
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
