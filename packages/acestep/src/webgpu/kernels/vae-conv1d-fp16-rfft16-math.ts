export const ACE_OPT_0077_RFFT16_LENGTH = 16 as const;
export const ACE_OPT_0077_RFFT16_KERNEL_SIZE = 7 as const;
export const ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE = 10 as const;
export const ACE_OPT_0077_RFFT16_OVERLAP = 6 as const;
export const ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32 = 0.25;
export const ACE_OPT_0077_RFFT16_PAIR_SCALE_F32 =
  0.3535533845424652;
export const ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT =
  "coord16-cin4-cout-band128-subgroup4-lane32-cin-element4" as const;

export const ACE_OPT_0077_RFFT16_COORDINATE_ORDER = Object.freeze([
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
] as const);

/** Float32 `cos(2*pi*k/16)` for the forward radix-2 twiddles, k=0..7. */
export const ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32 = Object.freeze([
  1,
  0.9238795042037964,
  0.7071067690849304,
  0.3826834261417389,
  0,
  -0.3826834261417389,
  -0.7071067690849304,
  -0.9238795042037964,
] as const);

/** Float32 `-sin(2*pi*k/16)` for the forward radix-2 twiddles, k=0..7. */
export const ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32 = Object.freeze([
  -0,
  -0.3826834261417389,
  -0.7071067690849304,
  -0.9238795042037964,
  -1,
  -0.9238795042037964,
  -0.7071067690849304,
  -0.3826834261417389,
] as const);

const BIT_REVERSE_4 = Object.freeze([
  0,
  8,
  4,
  12,
  2,
  10,
  6,
  14,
  1,
  9,
  5,
  13,
  3,
  11,
  7,
  15,
] as const);

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

export type AceOpt0077Rfft16Dilation = 1 | 3 | 9;

export interface AceOpt0077Rfft16NativeWeightCoordinate {
  readonly outputChannel: number;
  readonly kernel: number;
  readonly inputChannel: number;
}

export interface AceOpt0077Rfft16PackedWeightCoordinate {
  readonly coordinate: number;
  readonly outputChannel: number;
  readonly inputChannel: number;
}

export interface AceOpt0077Rfft16OutputTimeRange {
  readonly firstOutputTime: number;
  readonly outputTimeCount: number;
}

export interface AceOpt0077Rfft16TilePlan {
  readonly tileIndex: number;
  readonly dilation: AceOpt0077Rfft16Dilation;
  readonly residue: number;
  readonly streamLength: number;
  readonly streamOutputBase: number;
  readonly firstInputStreamPosition: number;
  readonly firstInputTime: number;
  readonly firstOutputTime: number;
  /** Sixteen-bit mask of in-bounds padded input positions. */
  readonly inputMask: number;
  /** Ten-bit mask of outputs inside the logical stream extent. */
  readonly logicalOutputMask: number;
  /** Ten-bit subset requested by this range. */
  readonly requestedOutputMask: number;
}

export interface AceOpt0077Rfft16RangePlan {
  readonly outputFrames: number;
  readonly dilation: AceOpt0077Rfft16Dilation;
  readonly firstOutputTime: number;
  readonly outputTimeCount: number;
  readonly endOutputTime: number;
  readonly plannedOutputCount: number;
  readonly tiles: readonly AceOpt0077Rfft16TilePlan[];
}

/**
 * Fixed bit-reversed DIT radix-2 RFFT16 with an explicit Float32 step after
 * every multiply, add, subtract, and final normalization used by the GPU
 * candidate. Coordinates are `[DC, NYQUIST, COS1, SIN1, ..., COS7, SIN7]`.
 */
export function aceOpt0077Rfft16ForwardF32(
  values: ArrayLike<number>,
): Float32Array {
  requireFiniteVector(values, ACE_OPT_0077_RFFT16_LENGTH, "RFFT16 input");
  const naturalReal = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  const naturalImaginary = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  for (let index = 0; index < ACE_OPT_0077_RFFT16_LENGTH; index += 1) {
    naturalReal[index] = Math.fround(values[index]!);
  }
  const { real, imaginary } = radix2DitF32(
    naturalReal,
    naturalImaginary,
    false,
  );

  const coordinates = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  coordinates[0] = f32Multiply(
    real[0]!,
    ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
  );
  coordinates[1] = f32Multiply(
    real[8]!,
    ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
  );
  for (let frequency = 1; frequency < 8; frequency += 1) {
    const coordinate = frequency * 2;
    coordinates[coordinate] = f32Multiply(
      real[frequency]!,
      ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
    );
    coordinates[coordinate + 1] = f32Multiply(
      Math.fround(-imaginary[frequency]!),
      ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
    );
  }
  requireFiniteVector(
    coordinates,
    ACE_OPT_0077_RFFT16_LENGTH,
    "RFFT16 coordinates",
  );
  return coordinates;
}

/** Independent direct-DFT definition used to prove the radix-2 schedule. */
export function aceOpt0077Rfft16DirectDftReference(
  values: ArrayLike<number>,
): Float64Array {
  requireFiniteVector(values, ACE_OPT_0077_RFFT16_LENGTH, "DFT input");
  const coordinates = new Float64Array(ACE_OPT_0077_RFFT16_LENGTH);
  let dc = 0;
  let nyquist = 0;
  for (let index = 0; index < ACE_OPT_0077_RFFT16_LENGTH; index += 1) {
    const value = values[index]!;
    dc += value;
    nyquist += (index & 1) === 0 ? value : -value;
  }
  coordinates[0] = dc / 4;
  coordinates[1] = nyquist / 4;
  const scale = 1 / Math.sqrt(8);
  for (let frequency = 1; frequency < 8; frequency += 1) {
    let cosine = 0;
    let sine = 0;
    for (let index = 0; index < ACE_OPT_0077_RFFT16_LENGTH; index += 1) {
      const angle = 2 * Math.PI * frequency * index /
        ACE_OPT_0077_RFFT16_LENGTH;
      cosine += values[index]! * Math.cos(angle);
      sine += values[index]! * Math.sin(angle);
    }
    coordinates[frequency * 2] = cosine * scale;
    coordinates[frequency * 2 + 1] = sine * scale;
  }
  return coordinates;
}

export function aceOpt0077TransformK7WeightF32(
  kernel: ArrayLike<number>,
): Float32Array {
  requireFiniteVector(kernel, ACE_OPT_0077_RFFT16_KERNEL_SIZE, "K7 weight");
  const padded = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  for (let index = 0; index < ACE_OPT_0077_RFFT16_KERNEL_SIZE; index += 1) {
    padded[index] = Math.fround(kernel[index]!);
  }
  return aceOpt0077Rfft16ForwardF32(padded);
}

/** Standard four-product complex-domain multiply for one input channel. */
export function aceOpt0077Rfft16MultiplyF32(
  inputCoordinates: ArrayLike<number>,
  weightCoordinates: ArrayLike<number>,
): Float32Array {
  requireFiniteVector(
    inputCoordinates,
    ACE_OPT_0077_RFFT16_LENGTH,
    "input coordinates",
  );
  requireFiniteVector(
    weightCoordinates,
    ACE_OPT_0077_RFFT16_LENGTH,
    "weight coordinates",
  );
  const product = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  product[0] = f32Multiply(inputCoordinates[0]!, weightCoordinates[0]!);
  product[1] = f32Multiply(inputCoordinates[1]!, weightCoordinates[1]!);
  for (let frequency = 1; frequency < 8; frequency += 1) {
    const coordinate = frequency * 2;
    const inputCosine = inputCoordinates[coordinate]!;
    const inputSine = inputCoordinates[coordinate + 1]!;
    const weightCosine = weightCoordinates[coordinate]!;
    const weightSine = weightCoordinates[coordinate + 1]!;
    product[coordinate] = f32Add(
      f32Multiply(inputCosine, weightCosine),
      f32Multiply(inputSine, weightSine),
    );
    product[coordinate + 1] = f32Subtract(
      f32Multiply(inputCosine, weightSine),
      f32Multiply(inputSine, weightCosine),
    );
  }
  return product;
}

/**
 * Fixed unitary inverse radix-2 transform of
 * `[Z0,Z8,R1,I1,...,R7,I7]`, followed by FP32 bias addition.
 */
export function aceOpt0077Rfft16InverseF32(
  productCoordinates: ArrayLike<number>,
  bias = 0,
): Float32Array {
  requireFiniteVector(
    productCoordinates,
    ACE_OPT_0077_RFFT16_LENGTH,
    "product coordinates",
  );
  requireFiniteNumber(bias, "RFFT16 inverse bias");
  const spectrumReal = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  const spectrumImaginary = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  spectrumReal[0] = f32Multiply(productCoordinates[0]!, 4);
  spectrumReal[8] = f32Multiply(productCoordinates[1]!, 4);
  for (let frequency = 1; frequency < 8; frequency += 1) {
    const coordinate = frequency * 2;
    const real = f32Multiply(productCoordinates[coordinate]!, 2);
    const imaginary = f32Multiply(productCoordinates[coordinate + 1]!, 2);
    spectrumReal[frequency] = real;
    spectrumImaginary[frequency] = imaginary;
    spectrumReal[16 - frequency] = real;
    spectrumImaginary[16 - frequency] = Math.fround(-imaginary);
  }
  const inverse = radix2DitF32(spectrumReal, spectrumImaginary, true);
  const output = new Float32Array(ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE);
  const roundedBias = Math.fround(bias);
  for (
    let outputIndex = 0;
    outputIndex < ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE;
    outputIndex += 1
  ) {
    output[outputIndex] = f32Add(
      f32Multiply(
        inverse.real[outputIndex]!,
        ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
      ),
      roundedBias,
    );
  }
  requireFiniteVector(
    output,
    ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
    "RFFT16 inverse output",
  );
  return output;
}

export function aceOpt0077Rfft16CorrelateF32(
  inputCoordinates: ArrayLike<number>,
  weightCoordinates: ArrayLike<number>,
  bias = 0,
): Float32Array {
  return aceOpt0077Rfft16InverseF32(
    aceOpt0077Rfft16MultiplyF32(inputCoordinates, weightCoordinates),
    bias,
  );
}

/** Independent scalar-F64 ten-output K7 cross-correlation. */
export function aceOpt0077DirectK7Correlation(
  inputTile: ArrayLike<number>,
  kernel: ArrayLike<number>,
  bias = 0,
): Float64Array {
  requireFiniteVector(
    inputTile,
    ACE_OPT_0077_RFFT16_LENGTH,
    "direct K7 input",
  );
  requireFiniteVector(kernel, ACE_OPT_0077_RFFT16_KERNEL_SIZE, "direct K7");
  requireFiniteNumber(bias, "direct K7 bias");
  const output = new Float64Array(ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE);
  for (
    let outputIndex = 0;
    outputIndex < ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE;
    outputIndex += 1
  ) {
    let sum = bias;
    for (let kernelIndex = 0; kernelIndex < 7; kernelIndex += 1) {
      sum += kernel[kernelIndex]! * inputTile[outputIndex + kernelIndex]!;
    }
    output[outputIndex] = sum;
  }
  return output;
}

export function aceOpt0077Rfft16NativeWeightIndex(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0077Rfft16NativeWeightCoordinate,
): number {
  requireWeightDimensions(inputChannels, outputChannels);
  requireIntegerInRange(
    coordinate.outputChannel,
    outputChannels,
    "native output channel",
  );
  requireIntegerInRange(
    coordinate.kernel,
    ACE_OPT_0077_RFFT16_KERNEL_SIZE,
    "native kernel coordinate",
  );
  requireIntegerInRange(
    coordinate.inputChannel,
    inputChannels,
    "native input channel",
  );
  return (coordinate.outputChannel * ACE_OPT_0077_RFFT16_KERNEL_SIZE +
    coordinate.kernel) * inputChannels + coordinate.inputChannel;
}

export function aceOpt0077Rfft16PackedWeightIndex(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0077Rfft16PackedWeightCoordinate,
): number {
  requireWeightDimensions(inputChannels, outputChannels);
  requirePackedWeightCoordinate(inputChannels, outputChannels, coordinate);
  const inputChannel4 = Math.floor(coordinate.inputChannel / 4);
  const inputElement4 = coordinate.inputChannel & 3;
  const outputBand128 = Math.floor(coordinate.outputChannel / 128);
  const outputInBand = coordinate.outputChannel % 128;
  const subgroup4 = Math.floor(outputInBand / 32);
  const lane32 = outputInBand & 31;
  return (((((coordinate.coordinate * (inputChannels / 4) + inputChannel4) *
    (outputChannels / 128) + outputBand128) * 4 + subgroup4) * 32 +
    lane32) * 4 + inputElement4);
}

export function aceOpt0077Rfft16PackedWeightCoordinate(
  inputChannels: number,
  outputChannels: number,
  packedScalarIndex: number,
): AceOpt0077Rfft16PackedWeightCoordinate {
  const elementCount = requireWeightDimensions(inputChannels, outputChannels);
  const packedElementCount = elementCount /
    ACE_OPT_0077_RFFT16_KERNEL_SIZE * ACE_OPT_0077_RFFT16_LENGTH;
  requireIntegerInRange(
    packedScalarIndex,
    packedElementCount,
    "packed scalar index",
  );
  let remainder = packedScalarIndex;
  const inputElement4 = remainder % 4;
  remainder = Math.floor(remainder / 4);
  const lane32 = remainder % 32;
  remainder = Math.floor(remainder / 32);
  const subgroup4 = remainder % 4;
  remainder = Math.floor(remainder / 4);
  const outputBand128 = remainder % (outputChannels / 128);
  remainder = Math.floor(remainder / (outputChannels / 128));
  const inputChannel4 = remainder % (inputChannels / 4);
  const coordinate = Math.floor(remainder / (inputChannels / 4));
  const unpacked = Object.freeze({
    coordinate,
    outputChannel: outputBand128 * 128 + subgroup4 * 32 + lane32,
    inputChannel: inputChannel4 * 4 + inputElement4,
  });
  requirePackedWeightCoordinate(inputChannels, outputChannels, unpacked);
  return unpacked;
}

/**
 * Transform native `[Cout,K7,Cin]` FP16 words and pack them as
 * `[coord16,Cin4,CoutBand128,subgroup4,lane32,CinElement4]` FP16 words.
 */
export function packAceOpt0077Rfft16WeightU16(
  nativeWeight: Uint16Array,
  inputChannels: number,
  outputChannels: number,
): Uint16Array {
  const nativeElementCount = requireWeightDimensions(
    inputChannels,
    outputChannels,
  );
  if (nativeWeight.length !== nativeElementCount) {
    throw new RangeError("OPT-0077 native K7 weight length changed");
  }
  const packedElementCount = nativeElementCount /
    ACE_OPT_0077_RFFT16_KERNEL_SIZE * ACE_OPT_0077_RFFT16_LENGTH;
  const packed = new Uint16Array(packedElementCount);
  const kernel = new Float32Array(ACE_OPT_0077_RFFT16_KERNEL_SIZE);
  for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
    for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
      for (let kernelIndex = 0; kernelIndex < 7; kernelIndex += 1) {
        const bits = nativeWeight[
          (outputChannel * 7 + kernelIndex) * inputChannels + inputChannel
        ]!;
        const value = aceOpt0077Float16BitsToNumber(bits);
        if (!Number.isFinite(value)) {
          throw new RangeError("OPT-0077 native K7 weight must be finite");
        }
        kernel[kernelIndex] = value;
      }
      const transformed = aceOpt0077TransformK7WeightF32(kernel);
      for (let coordinate = 0; coordinate < 16; coordinate += 1) {
        const packedIndex = aceOpt0077Rfft16PackedWeightIndex(
          inputChannels,
          outputChannels,
          { coordinate, outputChannel, inputChannel },
        );
        packed[packedIndex] = aceOpt0077NumberToFloat16Bits(
          transformed[coordinate]!,
        );
      }
    }
  }
  return packed;
}

export function aceOpt0077Rfft16StreamLength(
  outputFrames: number,
  dilation: AceOpt0077Rfft16Dilation,
  residue: number,
): number {
  requirePositiveSafeInteger(outputFrames, "RFFT16 output frames");
  requireDilation(dilation);
  requireIntegerInRange(residue, dilation, "RFFT16 residue");
  return residue >= outputFrames
    ? 0
    : Math.floor((outputFrames - 1 - residue) / dilation) + 1;
}

/** Globally anchors every residue-stream tile at a multiple of ten outputs. */
export function planAceOpt0077Rfft16Range(
  outputFrames: number,
  dilation: AceOpt0077Rfft16Dilation,
  range: AceOpt0077Rfft16OutputTimeRange,
): AceOpt0077Rfft16RangePlan {
  requirePositiveSafeInteger(outputFrames, "RFFT16 output frames");
  requireDilation(dilation);
  requirePositiveSafeInteger(range.outputTimeCount, "RFFT16 output count");
  if (!Number.isSafeInteger(range.firstOutputTime) ||
    range.firstOutputTime < 0) {
    throw new RangeError("RFFT16 first output time must be non-negative");
  }
  const endOutputTime = range.firstOutputTime + range.outputTimeCount;
  if (!Number.isSafeInteger(endOutputTime) || endOutputTime > outputFrames) {
    throw new RangeError("RFFT16 output range exceeds the logical extent");
  }

  const tiles: AceOpt0077Rfft16TilePlan[] = [];
  let plannedOutputCount = 0;
  for (let residue = 0; residue < dilation; residue += 1) {
    const streamLength = aceOpt0077Rfft16StreamLength(
      outputFrames,
      dilation,
      residue,
    );
    for (
      let streamOutputBase = 0;
      streamOutputBase < streamLength;
      streamOutputBase += ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE
    ) {
      let inputMask = 0;
      let logicalOutputMask = 0;
      let requestedOutputMask = 0;
      for (let input = 0; input < 16; input += 1) {
        const streamPosition = streamOutputBase - 3 + input;
        if (streamPosition >= 0 && streamPosition < streamLength) {
          inputMask |= 1 << input;
        }
      }
      for (let output = 0; output < 10; output += 1) {
        const streamPosition = streamOutputBase + output;
        if (streamPosition >= streamLength) continue;
        logicalOutputMask |= 1 << output;
        const outputTime = residue + dilation * streamPosition;
        if (
          outputTime >= range.firstOutputTime &&
          outputTime < endOutputTime
        ) {
          requestedOutputMask |= 1 << output;
        }
      }
      if (requestedOutputMask === 0) continue;
      plannedOutputCount += popcount16(requestedOutputMask);
      tiles.push(Object.freeze({
        tileIndex: tiles.length,
        dilation,
        residue,
        streamLength,
        streamOutputBase,
        firstInputStreamPosition: streamOutputBase - 3,
        firstInputTime: residue + dilation * (streamOutputBase - 3),
        firstOutputTime: residue + dilation * streamOutputBase,
        inputMask,
        logicalOutputMask,
        requestedOutputMask,
      }));
    }
  }
  if (plannedOutputCount !== range.outputTimeCount) {
    throw new Error("RFFT16 planner did not cover the requested output range");
  }
  return Object.freeze({
    outputFrames,
    dilation,
    firstOutputTime: range.firstOutputTime,
    outputTimeCount: range.outputTimeCount,
    endOutputTime,
    plannedOutputCount,
    tiles: Object.freeze(tiles),
  });
}

export function aceOpt0077Float16BitsToNumber(bits: number): number {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) {
    throw new RangeError("OPT-0077 FP16 bits must be an unsigned 16-bit word");
  }
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = bits >>> 10 & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15);
}

export function aceOpt0077NumberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
  const sign = bits >>> 16 & 0x8000;
  const exponent = bits >>> 23 & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << shift - 1;
    return sign | (truncated + (
      remainder > halfway || (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)
  ) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return nextExponent >= 0x1f
        ? sign | 0x7c00
        : sign | nextExponent << 10;
    }
  }
  return sign | halfExponent << 10 | halfMantissa;
}

function requireWeightDimensions(
  inputChannels: number,
  outputChannels: number,
): number {
  requirePositiveSafeInteger(inputChannels, "OPT-0077 input channels");
  requirePositiveSafeInteger(outputChannels, "OPT-0077 output channels");
  if (inputChannels % 4 !== 0) {
    throw new RangeError("OPT-0077 RFFT16 weights require Cin divisible by 4");
  }
  if (outputChannels % 128 !== 0) {
    throw new RangeError("OPT-0077 RFFT16 weights require Cout divisible by 128");
  }
  const elementCount = inputChannels * outputChannels * 7;
  const packedElementCount = inputChannels * outputChannels * 16;
  if (!Number.isSafeInteger(elementCount) ||
    !Number.isSafeInteger(packedElementCount)) {
    throw new RangeError("OPT-0077 RFFT16 weight dimensions overflow");
  }
  return elementCount;
}

function requirePackedWeightCoordinate(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0077Rfft16PackedWeightCoordinate,
): void {
  requireIntegerInRange(
    coordinate.coordinate,
    ACE_OPT_0077_RFFT16_LENGTH,
    "spectral coordinate",
  );
  requireIntegerInRange(
    coordinate.outputChannel,
    outputChannels,
    "packed output channel",
  );
  requireIntegerInRange(
    coordinate.inputChannel,
    inputChannels,
    "packed input channel",
  );
}

function requireFiniteVector(
  values: ArrayLike<number>,
  expectedLength: number,
  label: string,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(`${label} must contain exactly ${expectedLength} values`);
  }
  for (let index = 0; index < expectedLength; index += 1) {
    requireFiniteNumber(values[index]!, `${label}[${index}]`);
  }
}

function requireFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function requireIntegerInRange(
  value: number,
  exclusiveMaximum: number,
  label: string,
): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= exclusiveMaximum) {
    throw new RangeError(`${label} is out of range`);
  }
}

function requireDilation(
  dilation: AceOpt0077Rfft16Dilation,
): void {
  if (dilation !== 1 && dilation !== 3 && dilation !== 9) {
    throw new RangeError("OPT-0077 RFFT16 dilation must be 1, 3, or 9");
  }
}

function radix2DitF32(
  naturalReal: Float32Array,
  naturalImaginary: Float32Array,
  inverse: boolean,
): { readonly real: Float32Array; readonly imaginary: Float32Array } {
  const real = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  const imaginary = new Float32Array(ACE_OPT_0077_RFFT16_LENGTH);
  for (let index = 0; index < ACE_OPT_0077_RFFT16_LENGTH; index += 1) {
    const source = BIT_REVERSE_4[index]!;
    real[index] = Math.fround(naturalReal[source]!);
    imaginary[index] = Math.fround(naturalImaginary[source]!);
  }

  for (const size of [2, 4, 8, 16] as const) {
    const half = size / 2;
    const twiddleStride = ACE_OPT_0077_RFFT16_LENGTH / size;
    for (let block = 0; block < ACE_OPT_0077_RFFT16_LENGTH; block += size) {
      for (let offset = 0; offset < half; offset += 1) {
        const even = block + offset;
        const odd = even + half;
        const twiddle = offset * twiddleStride;
        const oddReal = real[odd]!;
        const oddImaginary = imaginary[odd]!;
        const twiddleReal = ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32[twiddle]!;
        const forwardTwiddleImaginary =
          ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32[twiddle]!;
        const twiddleImaginary = inverse
          ? Math.fround(-forwardTwiddleImaginary)
          : forwardTwiddleImaginary;
        const rotatedReal = f32Subtract(
          f32Multiply(oddReal, twiddleReal),
          f32Multiply(oddImaginary, twiddleImaginary),
        );
        const rotatedImaginary = f32Add(
          f32Multiply(oddReal, twiddleImaginary),
          f32Multiply(oddImaginary, twiddleReal),
        );
        const evenReal = real[even]!;
        const evenImaginary = imaginary[even]!;
        real[even] = f32Add(evenReal, rotatedReal);
        imaginary[even] = f32Add(evenImaginary, rotatedImaginary);
        real[odd] = f32Subtract(evenReal, rotatedReal);
        imaginary[odd] = f32Subtract(evenImaginary, rotatedImaginary);
      }
    }
  }
  return Object.freeze({ real, imaginary });
}

function f32Multiply(left: number, right: number): number {
  return Math.fround(Math.fround(left) * Math.fround(right));
}

function f32Add(left: number, right: number): number {
  return Math.fround(Math.fround(left) + Math.fround(right));
}

function f32Subtract(left: number, right: number): number {
  return Math.fround(Math.fround(left) - Math.fround(right));
}

function popcount16(value: number): number {
  let remaining = value & 0xffff;
  let count = 0;
  while (remaining !== 0) {
    remaining &= remaining - 1;
    count += 1;
  }
  return count;
}
