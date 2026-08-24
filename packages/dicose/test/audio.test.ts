import { describe, expect, it } from "vitest";

import {
  audioBufferToStereoPcm,
  centeredHannIstft,
  centeredHannStft,
  createPeriodicHannWindow,
  encodeFloat32Wav,
  encodePcm16Wav,
  encodeStereoWav,
  fillSeededGaussian,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  packFloat16,
  resampleLinear,
  resampleSinc,
  resampleStereoSincToLength,
  seededGaussianNoise,
  subtractStereoPcm,
  unpackFloat16,
} from "../src/runtime/audio.js";
import type { AudioBufferLike, StereoPcm } from "../src/runtime/audio.js";

function stereoPcm(
  sampleRate: number,
  leftValues: readonly number[],
  rightValues: readonly number[],
): StereoPcm {
  const left = Float32Array.from(leftValues);
  const right = Float32Array.from(rightValues);
  return {
    sampleRate,
    length: left.length,
    left,
    right,
    channels: [left, right],
  };
}

describe("browser audio normalization", () => {
  it("duplicates mono and applies the torchaudio-compatible sinc resampler", () => {
    const mono = new Float32Array(16);
    mono[0] = 1;
    const audioBuffer: AudioBufferLike = {
      sampleRate: 22_050,
      length: mono.length,
      numberOfChannels: 1,
      getChannelData(channel: number): Float32Array {
        if (channel !== 0) throw new RangeError("unexpected channel");
        return mono;
      },
    };

    const stereo = audioBufferToStereoPcm(audioBuffer);
    expect(stereo.sampleRate).toBe(44_100);
    expect(stereo.length).toBe(32);
    expect(Array.from(stereo.right)).toEqual(Array.from(stereo.left));
    expect(stereo.left).not.toBe(stereo.right);

    const direct = resampleSinc(mono, 22_050, 44_100);
    expect(Array.from(direct)).toEqual(Array.from(stereo.left));
  });

  it("can retain the decoded source timeline for file-level restoration", () => {
    const mono = Float32Array.from([0.25, -0.5, 0.75]);
    const source: AudioBufferLike = {
      sampleRate: 16_000,
      length: mono.length,
      numberOfChannels: 1,
      getChannelData: () => mono,
    };
    const stereo = audioBufferToStereoPcm(source, "source");
    expect(stereo.sampleRate).toBe(16_000);
    expect(stereo.length).toBe(3);
    expect(Array.from(stereo.left)).toEqual(Array.from(mono));
    expect(stereo.right).not.toBe(stereo.left);
  });
});

describe("torchaudio 2.0.2 Hann-sinc resampling", () => {
  it("enforces the original frame count after a reverse rate conversion", () => {
    const left = Float32Array.from({ length: 45 }, (_, index) => Math.sin(index * 0.2));
    const right = Float32Array.from(left, (value) => -value);
    const source = {
      sampleRate: 44_100,
      length: 45,
      left,
      right,
      channels: [left, right] as const,
    };
    expect(resampleSinc(left, 44_100, 16_000)).toHaveLength(17);
    const restored = resampleStereoSincToLength(source, 16_000, 16);
    expect(restored.sampleRate).toBe(16_000);
    expect(restored.length).toBe(16);
    expect(restored.left).toHaveLength(16);
    expect(restored.right).toHaveLength(16);
  });
  it("matches the upstream 16-to-44.1 kHz edge oracle and ceil length", () => {
    const impulse = new Float32Array(16);
    impulse[0] = 1;

    const output = resampleSinc(impulse, 16_000, 44_100);
    // Generated with torch 2.0.1 + torchaudio 2.0.2's default
    // transforms.Resample(orig_freq=16000, new_freq=44100).
    const expected = [
      0.9900000095367432,
      0.7858914732933044,
      0.3275667130947113,
      -0.06508029997348785,
      -0.1860049068927765,
      -0.08340922743082047,
      0.04888651520013809,
      0.07835452258586884,
      0.022632667794823647,
      -0.028583010658621788,
      -0.02922963537275791,
      -0.003192827571183443,
      0.011086936108767986,
      0.0067762997932732105,
      -0.0003555649018380791,
      -0.001398177002556622,
      -0.00017156278772745281,
    ];

    // ceil(16 * 441 / 160) is 45; a round-based path incorrectly returns 44.
    expect(output).toHaveLength(45);
    for (let index = 0; index < expected.length; index += 1) {
      expect(output[index]).toBeCloseTo(expected[index]!, 6);
    }
  });

  it("matches the 22.05-to-44.1 kHz edge-impulse oracle", () => {
    const impulse = new Float32Array(32);
    impulse[0] = 1;

    const output = resampleSinc(impulse, 22_050, 44_100);
    const expected = [
      0.9900000095367432,
      0.6259110569953918,
      0.009341620840132236,
      -0.18151485919952393,
      -0.007540243677794933,
      0.08069289475679398,
      0.005071021616458893,
      -0.034302063286304474,
      -0.0025844171177595854,
      0.010845829732716084,
      0.0007337729330174625,
      -0.0011951893102377653,
    ];

    expect(output).toHaveLength(64);
    for (let index = 0; index < expected.length; index += 1) {
      expect(output[index]).toBeCloseTo(expected[index]!, 6);
    }
  });

  it("matches the 48-to-44.1 kHz downsampling edge and ceil length", () => {
    const impulse = new Float32Array(7);
    impulse[0] = 1;

    const output = resampleSinc(impulse, 48_000, 44_100);
    const expected = [
      0.9095625281333923,
      0.008582614362239838,
      -0.006927598733454943,
      0.004659000784158707,
      -0.0023744332138448954,
      0.0006741539109498262,
      -0.0000022533390620083082,
    ];

    // torchaudio uses ceil(7 * 147 / 160); a round-based path returns 6.
    expect(output).toHaveLength(7);
    for (let index = 0; index < expected.length; index += 1) {
      expect(output[index]).toBeCloseTo(expected[index]!, 6);
    }
  });

  it("returns an independent copy when no rate conversion is needed", () => {
    const source = Float32Array.from([0.25, -0.5, 1]);
    const output = resampleSinc(source, 44_100, 44_100);

    expect(Array.from(output)).toEqual(Array.from(source));
    expect(output).not.toBe(source);
  });

  it("retains linear interpolation only as an explicit legacy path", () => {
    const output = resampleLinear(
      Float32Array.from([0, 1]),
      22_050,
      44_100,
    );
    expect(Array.from(output)).toEqual([0, 0.5, 1, 1]);
  });
});

describe("derived instrumental PCM", () => {
  it("subtracts vocals sample-wise into independent native-timeline arrays", () => {
    const mixture = stereoPcm(22_050, [0.75, -0.5, 0.125], [-0.25, 0.5, -0.75]);
    const vocals = stereoPcm(22_050, [0.25, -0.125, -0.375], [0.5, 0.25, -0.25]);

    const instrumental = subtractStereoPcm(mixture, vocals);

    expect(instrumental.sampleRate).toBe(22_050);
    expect(instrumental.length).toBe(3);
    expect(Array.from(instrumental.left)).toEqual([0.5, -0.375, 0.5]);
    expect(Array.from(instrumental.right)).toEqual([-0.75, 0.25, -0.5]);
    expect(instrumental.left).not.toBe(mixture.left);
    expect(instrumental.left).not.toBe(vocals.left);
    expect(instrumental.right).not.toBe(mixture.right);
    expect(instrumental.right).not.toBe(vocals.right);
    instrumental.left[0] = 0;
    expect(mixture.left[0]).toBe(0.75);
    expect(vocals.left[0]).toBe(0.25);
  });

  it("does not clamp a complementary waveform that exceeds unit peak", async () => {
    const mixture = stereoPcm(48_000, [0.75, -0.75], [-0.75, 0.75]);
    const vocals = stereoPcm(48_000, [-0.5, 0.5], [0.5, -0.5]);

    const instrumental = subtractStereoPcm(mixture, vocals);
    expect(Array.from(instrumental.left)).toEqual([1.25, -1.25]);
    expect(Array.from(instrumental.right)).toEqual([-1.25, 1.25]);

    const view = new DataView(await encodeStereoWav(instrumental).arrayBuffer());
    expect(view.getUint16(20, true)).toBe(3);
    expect(view.getFloat32(44, true)).toBe(1.25);
    expect(view.getFloat32(48, true)).toBe(-1.25);
  });

  it("rejects mismatched sample rates or frame counts", () => {
    const mixture = stereoPcm(44_100, [0, 1], [0, -1]);
    expect(() => subtractStereoPcm(
      mixture,
      stereoPcm(48_000, [0, 1], [0, -1]),
    )).toThrow(/matching timelines/);
    expect(() => subtractStereoPcm(
      mixture,
      stereoPcm(44_100, [0], [0]),
    )).toThrow(/matching timelines/);
  });
});

describe("centered periodic-Hann STFT", () => {
  it("uses torch-style reflection padding and a one-sided frame-major layout", () => {
    const samples = Float32Array.from([0, 1, 2, 3, 4, 5]);
    const spectrum = centeredHannStft(samples, { nFft: 8, hopLength: 2 });
    const window = createPeriodicHannWindow(8);
    const reflectedFirstFrame = Float32Array.from([4, 3, 2, 1, 0, 1, 2, 3]);

    expect(spectrum.frameCount).toBe(4);
    expect(spectrum.binCount).toBe(5);
    for (let bin = 0; bin < spectrum.binCount; bin += 1) {
      let expectedReal = 0;
      let expectedImag = 0;
      for (let sample = 0; sample < 8; sample += 1) {
        const value = reflectedFirstFrame[sample]! * window[sample]!;
        const phase = (2 * Math.PI * bin * sample) / 8;
        expectedReal += value * Math.cos(phase);
        expectedImag -= value * Math.sin(phase);
      }
      expect(spectrum.real[bin]).toBeCloseTo(expectedReal, 5);
      expect(spectrum.imag[bin]).toBeCloseTo(expectedImag, 5);
    }
  });

  it("round-trips the DiCoSe 2048/441 analysis geometry with overlap-add", () => {
    const samples = new Float32Array(6_000);
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = Math.fround(
        0.65 * Math.sin((2 * Math.PI * index) / 97) +
          0.2 * Math.cos((2 * Math.PI * index) / 311),
      );
    }

    const spectrum = centeredHannStft(samples);
    const reconstructed = centeredHannIstft(spectrum);
    expect(spectrum.binCount).toBe(1_025);
    expect(spectrum.frameCount).toBe(Math.floor(samples.length / 441) + 1);
    expect(reconstructed).toHaveLength(samples.length);

    let maximumError = 0;
    for (let index = 0; index < samples.length; index += 1) {
      maximumError = Math.max(
        maximumError,
        Math.abs(samples[index]! - reconstructed[index]!),
      );
    }
    expect(maximumError).toBeLessThan(3e-5);
  });
});

describe("float16 transport", () => {
  it("uses IEEE binary16 storage with round-to-nearest-even", () => {
    expect(float32ToFloat16Bits(0)).toBe(0x0000);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
    expect(float32ToFloat16Bits(1)).toBe(0x3c00);
    expect(float32ToFloat16Bits(-2)).toBe(0xc000);
    expect(float32ToFloat16Bits(65_504)).toBe(0x7bff);
    expect(float32ToFloat16Bits(2 ** -24)).toBe(0x0001);
    expect(float32ToFloat16Bits(2 ** -25)).toBe(0x0000);
    expect(float16BitsToFloat32(0x3c00)).toBe(1);
    expect(Object.is(float16BitsToFloat32(0x8000), -0)).toBe(true);

    const packed = packFloat16(Float32Array.from([-1, -0, 0.25, 1]));
    expect(Array.from(packed)).toEqual([0xbc00, 0x8000, 0x3400, 0x3c00]);
    expect(Array.from(unpackFloat16(packed))).toEqual([-1, -0, 0.25, 1]);
  });
});

describe("seeded Gaussian noise", () => {
  it("is repeatable across allocation and in-place fill paths", () => {
    const first = seededGaussianNoise(8, 0x1234_5678);
    const repeated = seededGaussianNoise(8, 0x1234_5678);
    const otherSeed = seededGaussianNoise(8, 0x1234_5679);
    const inPlace = new Float32Array(8);
    fillSeededGaussian(inPlace, 0x1234_5678);

    expect(Array.from(repeated)).toEqual(Array.from(first));
    expect(Array.from(inPlace)).toEqual(Array.from(first));
    expect(Array.from(otherSeed)).not.toEqual(Array.from(first));
    expect(Array.from(first)).toEqual([
      0.9760341644287109,
      0.5642386674880981,
      -1.5902931690216064,
      -0.06646478921175003,
      0.669416606426239,
      1.0959903001785278,
      0.13883674144744873,
      -1.1043612957000732,
    ]);
  });
});

describe("PCM16 WAV encoding", () => {
  it("writes a canonical interleaved RIFF WAV without a browser save action", () => {
    const wav = encodePcm16Wav(
      [Float32Array.from([1, -1]), Float32Array.from([-1, 0.5])],
      44_100,
    );
    const view = new DataView(wav);
    const ascii = new TextDecoder().decode(new Uint8Array(wav, 0, 4));
    expect(ascii).toBe("RIFF");
    expect(view.getUint32(24, true)).toBe(44_100);
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getInt16(44, true)).toBe(32_767);
    expect(view.getInt16(46, true)).toBe(-32_768);
    expect(view.getInt16(48, true)).toBe(-32_768);
    expect(view.getInt16(50, true)).toBe(16_384);
  });

  it("selects IEEE float WAV only when a stereo stem exceeds unit peak", async () => {
    const atLimitLeft = Float32Array.from([1, -1]);
    const atLimitRight = Float32Array.from([0.25, -0.5]);
    const atLimit = {
      sampleRate: 44_100,
      length: 2,
      left: atLimitLeft,
      right: atLimitRight,
      channels: [atLimitLeft, atLimitRight] as const,
    };
    const pcmBlob = encodeStereoWav(atLimit);
    expect(new DataView(await pcmBlob.arrayBuffer()).getUint16(20, true)).toBe(1);

    const overRange = Float32Array.from([1.5, -1.25]);
    const overRangeRight = Float32Array.from([0.125, -0.25]);
    const floatBlob = encodeStereoWav({
      sampleRate: 48_000,
      length: 2,
      left: overRange,
      right: overRangeRight,
      channels: [overRange, overRangeRight],
    });
    const view = new DataView(await floatBlob.arrayBuffer());
    expect(view.getUint16(20, true)).toBe(3);
    expect(view.getUint32(24, true)).toBe(48_000);
    expect(view.getUint16(34, true)).toBe(32);
    expect(view.getFloat32(44, true)).toBe(1.5);
    expect(view.getFloat32(48, true)).toBeCloseTo(0.125);
    expect(view.getFloat32(52, true)).toBe(-1.25);
    expect(view.getFloat32(56, true)).toBeCloseTo(-0.25);
  });

  it("writes exact planar interleave in an IEEE float WAV", () => {
    const wav = encodeFloat32Wav(
      [Float32Array.from([1.25, -1.5]), Float32Array.from([0.5, -0.25])],
      32_000,
    );
    const view = new DataView(wav);
    expect(view.getUint32(40, true)).toBe(16);
    expect(view.getFloat32(44, true)).toBe(1.25);
    expect(view.getFloat32(48, true)).toBe(0.5);
    expect(view.getFloat32(52, true)).toBe(-1.5);
    expect(view.getFloat32(56, true)).toBe(-0.25);
  });
});
