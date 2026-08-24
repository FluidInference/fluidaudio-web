import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const CONTRACT_PATH = resolve(ROOT, "test/fixtures/deterministic-reference.json");

/**
 * Fail an unattended browser run if its deterministic WebGPU checkpoint leaves
 * the separately-recorded upstream f32/f16 acceptance envelope. Runtime
 * completion alone is not a quality signal: a finite but silent mask would
 * otherwise look successful to the CDP harness.
 */
export function assertFixtureAcceptance(harnessResult) {
  const contract = JSON.parse(readFileSync(CONTRACT_PATH, "utf8"));
  const report = harnessResult?.report;
  if (report?.ok !== true || report.output === undefined || report.metrics === undefined) {
    throw new Error("Browser report has no successful fixture output to validate");
  }
  const diagnostics = report.output.diagnostics?.deterministic;
  if (diagnostics === undefined) {
    throw new Error("Browser report omitted deterministic diagnostic statistics");
  }
  const expected = contract.reference?.stems;
  const envelope = contract.webgpuF16Acceptance;
  if (expected === undefined || envelope === undefined) {
    throw new Error("Deterministic fixture contract is malformed");
  }

  for (const name of ["drums", "bass", "other"]) {
    const actual = diagnostics[name];
    const target = expected[name];
    assertStat(
      actual?.rms,
      target?.rms,
      envelope.audibleStems.rmsRelativeTolerance,
      envelope.audibleStems.rmsAbsoluteTolerance,
      `${name} deterministic RMS`,
    );
    assertStat(
      actual?.peak,
      target?.peak,
      envelope.audibleStems.peakRelativeTolerance,
      envelope.audibleStems.peakAbsoluteTolerance,
      `${name} deterministic peak`,
    );
  }
  const vocals = diagnostics.vocals;
  if (!Number.isFinite(vocals?.rms) || vocals.rms > envelope.vocals.maxRms) {
    throw new Error(`vocals deterministic RMS outside f16 acceptance envelope: ${vocals?.rms}`);
  }
  if (!Number.isFinite(vocals?.peak) || vocals.peak > envelope.vocals.maxPeak) {
    throw new Error(`vocals deterministic peak outside f16 acceptance envelope: ${vocals?.peak}`);
  }

  const outputSampleRate = contract.fixture?.input?.sampleRate;
  const outputFrames = contract.fixture?.input?.frames;
  if (!Number.isSafeInteger(outputSampleRate) || !Number.isSafeInteger(outputFrames)) {
    throw new Error("Deterministic fixture contract omits native output geometry");
  }
  for (const name of ["drums", "bass", "other", "vocals"]) {
    const stem = report.output.stems?.[name];
    if (
      stem?.sampleRate !== outputSampleRate ||
      stem.samples !== outputFrames ||
      stem.finiteSamples !== outputFrames * 2 ||
      !Number.isFinite(stem.rms) ||
      !Number.isFinite(stem.peak) ||
      stem.rms <= 0 ||
      stem.peak <= 0
    ) {
      throw new Error(`final ${name} output does not match the finite native fixture timeline`);
    }
  }
  const instrumental = report.output.instrumental;
  if (
    instrumental?.sampleRate !== outputSampleRate ||
    instrumental.samples !== outputFrames ||
    instrumental.finiteSamples !== outputFrames * 2 ||
    !Number.isFinite(instrumental.rms) ||
    !Number.isFinite(instrumental.peak) ||
    instrumental.rms <= 0 ||
    instrumental.peak <= 0
  ) {
    throw new Error("derived instrumental does not match the finite native fixture timeline");
  }
}

function assertStat(actual, expected, relativeTolerance, absoluteTolerance, label) {
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    throw new Error(`${label} is non-finite`);
  }
  const allowed = Math.max(absoluteTolerance, Math.abs(expected) * relativeTolerance);
  if (Math.abs(actual - expected) > allowed) {
    throw new Error(`${label} outside f16 acceptance envelope: expected ${expected}, got ${actual}, allowed ±${allowed}`);
  }
}
