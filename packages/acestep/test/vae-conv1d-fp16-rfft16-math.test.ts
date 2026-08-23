import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0077_RFFT16_COORDINATE_ORDER,
  ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
  ACE_OPT_0077_RFFT16_KERNEL_SIZE,
  ACE_OPT_0077_RFFT16_LENGTH,
  ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
  ACE_OPT_0077_RFFT16_OVERLAP,
  ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
  ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32,
  ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32,
  ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT,
  aceOpt0077DirectK7Correlation,
  aceOpt0077Float16BitsToNumber,
  aceOpt0077NumberToFloat16Bits,
  aceOpt0077Rfft16CorrelateF32,
  aceOpt0077Rfft16DirectDftReference,
  aceOpt0077Rfft16ForwardF32,
  aceOpt0077Rfft16InverseF32,
  aceOpt0077Rfft16NativeWeightIndex,
  aceOpt0077Rfft16PackedWeightCoordinate,
  aceOpt0077Rfft16PackedWeightIndex,
  aceOpt0077Rfft16StreamLength,
  aceOpt0077TransformK7WeightF32,
  packAceOpt0077Rfft16WeightU16,
  planAceOpt0077Rfft16Range,
} from "../src/webgpu/kernels/vae-conv1d-fp16-rfft16-math.js";

describe("OPT-0077 deterministic RFFT16 math and layout", () => {
  it("freezes the finite orthonormal coordinates, scales, and twiddles", () => {
    expect({
      length: ACE_OPT_0077_RFFT16_LENGTH,
      kernel: ACE_OPT_0077_RFFT16_KERNEL_SIZE,
      outputs: ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
      overlap: ACE_OPT_0077_RFFT16_OVERLAP,
      endpointScale: ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
      pairScale: ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
      layout: ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT,
    }).toEqual({
      length: 16,
      kernel: 7,
      outputs: 10,
      overlap: 6,
      endpointScale: Math.fround(1 / 4),
      pairScale: Math.fround(1 / Math.sqrt(8)),
      layout: "coord16-cin4-cout-band128-subgroup4-lane32-cin-element4",
    });
    expect(ACE_OPT_0077_RFFT16_COORDINATE_ORDER).toEqual([
      "dc",
      "nyquist",
      "cos1",
      "sin1",
      "cos2",
      "sin2",
      "cos3",
      "sin3",
      "cos4",
      "sin4",
      "cos5",
      "sin5",
      "cos6",
      "sin6",
      "cos7",
      "sin7",
    ]);
    expect(ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32).toHaveLength(8);
    expect(ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32).toHaveLength(8);
    expect(Object.is(ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32[0], -0)).toBe(true);
    for (const coefficient of [
      ...ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32,
      ...ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32,
      ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
      ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
    ]) {
      expect(Number.isFinite(coefficient)).toBe(true);
      expect(Math.fround(coefficient)).toBe(coefficient);
    }
    for (const bits of [
      0x0000,
      0x8000,
      0x0001,
      0x03ff,
      0x0400,
      0x3c00,
      0x7bff,
    ]) {
      expect(aceOpt0077NumberToFloat16Bits(
        aceOpt0077Float16BitsToNumber(bits),
      )).toBe(bits);
    }
    expect(Object.is(aceOpt0077Float16BitsToNumber(0x8000), -0)).toBe(true);
  });

  it("proves the fixed radix-2 schedule against an independent direct DFT", () => {
    let maximumBasisError = 0;
    const transformRows = Array.from(
      { length: ACE_OPT_0077_RFFT16_LENGTH },
      () => new Float64Array(ACE_OPT_0077_RFFT16_LENGTH),
    );
    for (let basis = 0; basis < ACE_OPT_0077_RFFT16_LENGTH; basis += 1) {
      const input = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
      input[basis] = 1;
      const radix = aceOpt0077Rfft16ForwardF32(input);
      const direct = aceOpt0077Rfft16DirectDftReference(input);
      for (let coordinate = 0; coordinate < 16; coordinate += 1) {
        maximumBasisError = Math.max(
          maximumBasisError,
          Math.abs(radix[coordinate]! - direct[coordinate]!),
        );
        transformRows[coordinate]![basis] = direct[coordinate]!;
      }
    }
    expect(maximumBasisError).toBeLessThan(1.3e-7);

    let maximumOrthonormalError = 0;
    for (let left = 0; left < 16; left += 1) {
      for (let right = 0; right < 16; right += 1) {
        let dot = 0;
        for (let index = 0; index < 16; index += 1) {
          dot += transformRows[left]![index]! *
            transformRows[right]![index]!;
        }
        maximumOrthonormalError = Math.max(
          maximumOrthonormalError,
          Math.abs(dot - (left === right ? 1 : 0)),
        );
      }
    }
    expect(maximumOrthonormalError).toBeLessThan(2e-15);

    for (let kernelBasis = 0; kernelBasis < 7; kernelBasis += 1) {
      const kernel = new Float32Array(7);
      kernel[kernelBasis] = 1;
      const padded = new Float32Array(16);
      padded[kernelBasis] = 1;
      expectMaximumDifference(
        aceOpt0077TransformK7WeightF32(kernel),
        aceOpt0077Rfft16DirectDftReference(padded),
        1.3e-7,
      );
    }
  });

  it("isolates DC, Nyquist, and every cosine/sine bin without a hidden scale", () => {
    const dc = aceOpt0077Rfft16ForwardF32(new Float32Array(16).fill(1));
    expect(dc[0]).toBe(4);
    expectMaximumOffCoordinate(dc, 0, 1e-6);

    const nyquistInput = Float32Array.from(
      { length: 16 },
      (_, index) => (index & 1) === 0 ? 1 : -1,
    );
    const nyquist = aceOpt0077Rfft16ForwardF32(nyquistInput);
    expect(nyquist[1]).toBe(4);
    expectMaximumOffCoordinate(nyquist, 1, 1e-6);

    for (let frequency = 1; frequency < 8; frequency += 1) {
      const cosineInput = Float32Array.from(
        { length: 16 },
        (_, index) => Math.fround(Math.cos(2 * Math.PI * frequency * index / 16)),
      );
      const sineInput = Float32Array.from(
        { length: 16 },
        (_, index) => Math.fround(Math.sin(2 * Math.PI * frequency * index / 16)),
      );
      const cosine = aceOpt0077Rfft16ForwardF32(cosineInput);
      const sine = aceOpt0077Rfft16ForwardF32(sineInput);
      expect(cosine[frequency * 2]).toBeCloseTo(Math.sqrt(8), 5);
      expect(sine[frequency * 2 + 1]).toBeCloseTo(Math.sqrt(8), 5);
      expectMaximumOffCoordinate(cosine, frequency * 2, 1.1e-6);
      expectMaximumOffCoordinate(sine, frequency * 2 + 1, 1.1e-6);
    }
  });

  it("proves the inverse radix-2 schedule and correlation signs by coordinate", () => {
    let maximumError = 0;
    for (let coordinate = 0; coordinate < 16; coordinate += 1) {
      const product = new Float32Array(16);
      product[coordinate] = 1;
      const actual = aceOpt0077Rfft16InverseF32(product);
      for (let output = 0; output < 10; output += 1) {
        let expected: number;
        if (coordinate === 0) expected = 1;
        else if (coordinate === 1) expected = (output & 1) === 0 ? 1 : -1;
        else {
          const frequency = Math.floor(coordinate / 2);
          const angle = 2 * Math.PI * frequency * output / 16;
          expected = (coordinate & 1) === 0
            ? Math.cos(angle)
            : -Math.sin(angle);
        }
        maximumError = Math.max(
          maximumError,
          Math.abs(actual[output]! - expected),
        );
      }
    }
    expect(maximumError).toBeLessThan(1.3e-7);
  });

  it("reconstructs all 16 x 7 basis correlations and a signed dense tile", () => {
    let maximumBasisError = 0;
    for (let inputBasis = 0; inputBasis < 16; inputBasis += 1) {
      const input = new Float32Array(16);
      input[inputBasis] = 1;
      const transformedInput = aceOpt0077Rfft16ForwardF32(input);
      for (let kernelBasis = 0; kernelBasis < 7; kernelBasis += 1) {
        const kernel = new Float32Array(7);
        kernel[kernelBasis] = 1;
        const actual = aceOpt0077Rfft16CorrelateF32(
          transformedInput,
          aceOpt0077TransformK7WeightF32(kernel),
        );
        const expected = aceOpt0077DirectK7Correlation(input, kernel);
        for (let output = 0; output < 10; output += 1) {
          maximumBasisError = Math.max(
            maximumBasisError,
            Math.abs(actual[output]! - expected[output]!),
          );
        }
      }
    }
    expect(maximumBasisError).toBeLessThan(7e-7);

    const input = Float32Array.from(
      { length: 16 },
      (_, index) => Math.fround(((index * 29 + 7) % 31 - 15) / 8),
    );
    const kernel = Float32Array.from(
      { length: 7 },
      (_, index) => Math.fround(((index * 11 + 3) % 17 - 8) / 16),
    );
    const actual = aceOpt0077Rfft16CorrelateF32(
      aceOpt0077Rfft16ForwardF32(input),
      aceOpt0077TransformK7WeightF32(kernel),
      -0.125,
    );
    const expected = aceOpt0077DirectK7Correlation(input, kernel, -0.125);
    expectMaximumDifference(actual, expected, 2.5e-6);
  });

  it("exhaustively inverts the declared transformed-weight scalar layout", () => {
    const inputChannels = 8;
    const outputChannels = 256;
    const packedElementCount = 16 * inputChannels * outputChannels;
    const seenCoordinates = new Set<string>();
    for (let packedIndex = 0; packedIndex < packedElementCount; packedIndex++) {
      const coordinate = aceOpt0077Rfft16PackedWeightCoordinate(
        inputChannels,
        outputChannels,
        packedIndex,
      );
      expect(aceOpt0077Rfft16PackedWeightIndex(
        inputChannels,
        outputChannels,
        coordinate,
      )).toBe(packedIndex);
      seenCoordinates.add(
        `${coordinate.coordinate}/${coordinate.outputChannel}/${coordinate.inputChannel}`,
      );
    }
    expect(seenCoordinates.size).toBe(packedElementCount);

    const nativeElementCount = 7 * inputChannels * outputChannels;
    const seenNative = new Uint8Array(nativeElementCount);
    for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
      for (let kernel = 0; kernel < 7; kernel += 1) {
        for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
          const index = aceOpt0077Rfft16NativeWeightIndex(
            inputChannels,
            outputChannels,
            { outputChannel, kernel, inputChannel },
          );
          expect(seenNative[index]).toBe(0);
          seenNative[index] = 1;
        }
      }
    }
    expect(seenNative.every((value) => value === 1)).toBe(true);
  });

  it("transforms native O-K-I FP16 words into finite coordinate-major words", () => {
    const inputChannels = 4;
    const outputChannels = 128;
    const native = new Uint16Array(7 * inputChannels * outputChannels);
    for (let index = 0; index < native.length; index += 1) {
      native[index] = aceOpt0077NumberToFloat16Bits(
        (((index * 37 + 11) % 33) - 16) / 16,
      );
    }
    const packed = packAceOpt0077Rfft16WeightU16(
      native,
      inputChannels,
      outputChannels,
    );
    expect(packed).toHaveLength(16 * inputChannels * outputChannels);
    expect(packed.every((bits) =>
      Number.isFinite(aceOpt0077Float16BitsToNumber(bits))
    )).toBe(true);

    for (const [outputChannel, inputChannel] of [
      [0, 0],
      [31, 1],
      [32, 2],
      [127, 3],
    ] as const) {
      const kernel = Float32Array.from(
        { length: 7 },
        (_, kernelIndex) => aceOpt0077Float16BitsToNumber(native[
          (outputChannel * 7 + kernelIndex) * inputChannels + inputChannel
        ]!),
      );
      const transformed = aceOpt0077TransformK7WeightF32(kernel);
      for (let coordinate = 0; coordinate < 16; coordinate += 1) {
        const packedIndex = aceOpt0077Rfft16PackedWeightIndex(
          inputChannels,
          outputChannels,
          { coordinate, outputChannel, inputChannel },
        );
        expect(packed[packedIndex]).toBe(
          aceOpt0077NumberToFloat16Bits(transformed[coordinate]!),
        );
      }
    }
  });

  it("exhaustively covers bounded globally anchored range intersections", () => {
    let rangeCount = 0;
    let tileCount = 0;
    let planCountMismatch = 0;
    let anchorMismatch = 0;
    let streamLengthMismatch = 0;
    let firstOutputMismatch = 0;
    let inputMaskMismatch = 0;
    let logicalMaskMismatch = 0;
    let requestedMaskMismatch = 0;
    let maskSubsetMismatch = 0;
    let coverageMismatch = 0;
    for (const dilation of [1, 3, 9] as const) {
      for (let frames = 1; frames <= 40; frames += 1) {
        for (let firstOutputTime = 0; firstOutputTime < frames; firstOutputTime++) {
          for (
            let outputTimeCount = 1;
            outputTimeCount <= frames - firstOutputTime;
            outputTimeCount++
          ) {
            const plan = planAceOpt0077Rfft16Range(frames, dilation, {
              firstOutputTime,
              outputTimeCount,
            });
            const coverage = new Uint8Array(frames);
            if (plan.plannedOutputCount !== outputTimeCount) {
              planCountMismatch += 1;
            }
            for (const tile of plan.tiles) {
              tileCount += 1;
              if (tile.streamOutputBase % 10 !== 0) anchorMismatch += 1;
              if (tile.streamLength !== aceOpt0077Rfft16StreamLength(
                frames,
                dilation,
                tile.residue,
              )) streamLengthMismatch += 1;
              if (tile.firstOutputTime !==
                tile.residue + dilation * tile.streamOutputBase) {
                firstOutputMismatch += 1;
              }
              let expectedInputMask = 0;
              let expectedLogicalMask = 0;
              let expectedRequestedMask = 0;
              for (let input = 0; input < 16; input += 1) {
                const stream = tile.streamOutputBase - 3 + input;
                if (stream >= 0 && stream < tile.streamLength) {
                  expectedInputMask |= 1 << input;
                }
              }
              for (let output = 0; output < 10; output += 1) {
                const stream = tile.streamOutputBase + output;
                if (stream >= tile.streamLength) continue;
                expectedLogicalMask |= 1 << output;
                const time = tile.residue + dilation * stream;
                if (
                  time >= firstOutputTime &&
                  time < firstOutputTime + outputTimeCount
                ) {
                  expectedRequestedMask |= 1 << output;
                  coverage[time] = coverage[time]! + 1;
                }
              }
              if (tile.inputMask !== expectedInputMask) inputMaskMismatch += 1;
              if (tile.logicalOutputMask !== expectedLogicalMask) {
                logicalMaskMismatch += 1;
              }
              if (tile.requestedOutputMask !== expectedRequestedMask) {
                requestedMaskMismatch += 1;
              }
              if ((tile.requestedOutputMask & ~tile.logicalOutputMask) !== 0) {
                maskSubsetMismatch += 1;
              }
            }
            for (let time = 0; time < frames; time += 1) {
              const expectedCoverage = time >= firstOutputTime &&
                  time < firstOutputTime + outputTimeCount
                ? 1
                : 0;
              if (coverage[time] !== expectedCoverage) coverageMismatch += 1;
            }
            rangeCount += 1;
          }
        }
      }
    }
    expect(rangeCount).toBe(34_440);
    expect(tileCount).toBeGreaterThan(rangeCount);
    expect({
      planCountMismatch,
      anchorMismatch,
      streamLengthMismatch,
      firstOutputMismatch,
      inputMaskMismatch,
      logicalMaskMismatch,
      requestedMaskMismatch,
      maskSubsetMismatch,
      coverageMismatch,
    }).toEqual({
      planCountMismatch: 0,
      anchorMismatch: 0,
      streamLengthMismatch: 0,
      firstOutputMismatch: 0,
      inputMaskMismatch: 0,
      logicalMaskMismatch: 0,
      requestedMaskMismatch: 0,
      maskSubsetMismatch: 0,
      coverageMismatch: 0,
    });
  }, 30_000);

  it("matches padded direct K7 through d1/d3/d9 tails and unaligned masks", () => {
    const kernel = Float32Array.from(
      { length: 7 },
      (_, index) => Math.fround(((index * 13 + 5) % 19 - 9) / 16),
    );
    const transformedWeight = aceOpt0077TransformK7WeightF32(kernel);
    let maximumError = 0;
    for (const dilation of [1, 3, 9] as const) {
      for (const frames of [1, 2, 9, 10, 11, 31, 64, 65]) {
        const input = Float32Array.from(
          { length: frames },
          (_, index) => Math.fround(((index * 29 + dilation * 7) % 37 - 18) / 16),
        );
        const ranges = [{ firstOutputTime: 0, outputTimeCount: frames }];
        if (frames >= 5) {
          ranges.push({
            firstOutputTime: 2,
            outputTimeCount: frames - 4,
          });
        }
        for (const range of ranges) {
          const plan = planAceOpt0077Rfft16Range(frames, dilation, range);
          const actual = new Float64Array(frames).fill(1_234_567);
          const writes = new Uint8Array(frames);
          for (const tile of plan.tiles) {
            const inputTile = new Float32Array(16);
            for (let tileInput = 0; tileInput < 16; tileInput += 1) {
              if ((tile.inputMask & 1 << tileInput) === 0) continue;
              const stream = tile.firstInputStreamPosition + tileInput;
              const time = tile.residue + dilation * stream;
              inputTile[tileInput] = input[time]!;
            }
            const tileOutput = aceOpt0077Rfft16CorrelateF32(
              aceOpt0077Rfft16ForwardF32(inputTile),
              transformedWeight,
              0.03125,
            );
            for (let output = 0; output < 10; output += 1) {
              if ((tile.requestedOutputMask & 1 << output) === 0) continue;
              const time = tile.firstOutputTime + dilation * output;
              actual[time] = tileOutput[output]!;
              writes[time] = writes[time]! + 1;
            }
          }
          for (let time = 0; time < frames; time += 1) {
            const requested = time >= range.firstOutputTime &&
              time < range.firstOutputTime + range.outputTimeCount;
            if (!requested) {
              expect(actual[time]).toBe(1_234_567);
              expect(writes[time]).toBe(0);
              continue;
            }
            let expected = 0.03125;
            for (let tap = 0; tap < 7; tap += 1) {
              const inputTime = time + dilation * (tap - 3);
              if (inputTime >= 0 && inputTime < frames) {
                expected += input[inputTime]! * kernel[tap]!;
              }
            }
            maximumError = Math.max(
              maximumError,
              Math.abs(actual[time]! - expected),
            );
            expect(writes[time]).toBe(1);
          }
        }
      }
    }
    expect(maximumError).toBeLessThan(3e-6);

    const everyResidue = planAceOpt0077Rfft16Range(65, 9, {
      firstOutputTime: 7,
      outputTimeCount: 53,
    });
    expect(new Set(everyResidue.tiles.map(({ residue }) => residue))).toEqual(
      new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    );
    expect(everyResidue.tiles.some(({ requestedOutputMask, logicalOutputMask }) =>
      requestedOutputMask !== logicalOutputMask
    )).toBe(true);
  });

  it("fails closed on invalid vectors, dimensions, coordinates, and ranges", () => {
    expect(() => aceOpt0077Rfft16ForwardF32(new Float32Array(15)))
      .toThrow(/exactly 16/);
    const nonFinite = new Float32Array(16);
    nonFinite[7] = Infinity;
    expect(() => aceOpt0077Rfft16ForwardF32(nonFinite)).toThrow(/finite/);
    expect(() => aceOpt0077TransformK7WeightF32(new Float32Array(6)))
      .toThrow(/exactly 7/);
    expect(() => aceOpt0077Rfft16PackedWeightIndex(6, 128, {
      coordinate: 0,
      outputChannel: 0,
      inputChannel: 0,
    })).toThrow(/Cin divisible by 4/);
    expect(() => aceOpt0077Rfft16PackedWeightIndex(8, 64, {
      coordinate: 0,
      outputChannel: 0,
      inputChannel: 0,
    })).toThrow(/Cout divisible by 128/);
    expect(() => aceOpt0077Rfft16PackedWeightIndex(8, 128, {
      coordinate: 16,
      outputChannel: 0,
      inputChannel: 0,
    })).toThrow(/spectral coordinate/);
    expect(() => planAceOpt0077Rfft16Range(32, 2 as 1, {
      firstOutputTime: 0,
      outputTimeCount: 1,
    })).toThrow(/dilation/);
    expect(() => planAceOpt0077Rfft16Range(32, 3, {
      firstOutputTime: 31,
      outputTimeCount: 2,
    })).toThrow(/exceeds/);
    const native = new Uint16Array(7 * 4 * 128);
    native[0] = 0x7e00;
    expect(() => packAceOpt0077Rfft16WeightU16(native, 4, 128))
      .toThrow(/finite/);
  });
});

function expectMaximumDifference(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  maximum: number,
): void {
  expect(actual.length).toBe(expected.length);
  let difference = 0;
  for (let index = 0; index < actual.length; index += 1) {
    difference = Math.max(
      difference,
      Math.abs(actual[index]! - expected[index]!),
    );
  }
  expect(difference).toBeLessThan(maximum);
}

function expectMaximumOffCoordinate(
  values: ArrayLike<number>,
  selected: number,
  maximum: number,
): void {
  let offCoordinate = 0;
  for (let index = 0; index < values.length; index += 1) {
    if (index === selected) continue;
    offCoordinate = Math.max(offCoordinate, Math.abs(values[index]!));
  }
  expect(offCoordinate).toBeLessThan(maximum);
}
