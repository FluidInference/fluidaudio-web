/// <reference types="@webgpu/types" />

import type { AceOpt0011Fp16VaeDecoderRuntime } from
  "../../src/webgpu/vae-fp16-decoder.js";
import {
  AceOpt0040VaeConvTranspose1dShapeSelectorKernel,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-shape-selector.js";
import {
  AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import type { AceVaeOutputRangeBinding } from
  "../../src/webgpu/kernels/vae-primitives.js";
import type { Opt0066DerivedTransposeWeight } from
  "./opt-0066-vae-dual-k4-package-proof.js";

type NativeOwner = AceOpt0040VaeConvTranspose1dShapeSelectorKernel;
type NativeCreateDispatch = NativeOwner["createDispatch"];
type NativeDispatch = Awaited<ReturnType<NativeCreateDispatch>>;

export interface Opt0066NativeTransposeOwner {
  createDispatch(
    ...args: Parameters<NativeCreateDispatch>
  ): Promise<NativeDispatch>;
  destroy(): void;
}

interface K4Owner {
  createDispatch(
    label: string,
    operationLabel: string,
    shape: Parameters<NativeCreateDispatch>[2],
    bindings: Readonly<{
      readonly input: GPUBufferBinding;
      readonly weight: GPUBufferBinding;
      readonly bias: GPUBufferBinding;
      readonly output: GPUBufferBinding;
    }>,
    range: AceVaeOutputRangeBinding,
  ): Promise<Readonly<{
    readonly kernelId: string;
    encode(pass: GPUComputePassEncoder): void;
  }>>;
  destroy(): void;
}

export interface Opt0066DerivedTransposeBinding {
  readonly operationLabel: string;
  readonly binding: GPUBufferBinding;
}

export interface Opt0066CompleteTransposeOracleOwner {
  createDispatch(
    ...args: Parameters<NativeCreateDispatch>
  ): Promise<NativeDispatch>;
  destroy(): void;
}

/**
 * Pure composition seam used by focused tests and the browser oracle. Block 0
 * remains literal OPT40. Blocks 1-4 replace only the physical weight binding
 * while executing the real OPT52/OPT48 K4 arithmetic owner.
 */
export function composeOpt0066CompleteTransposeOracleOwner(
  native: Opt0066NativeTransposeOwner,
  k4: K4Owner,
  derived: readonly Opt0066DerivedTransposeBinding[],
  destroyDerived: () => void,
): Opt0066CompleteTransposeOracleOwner {
  const byLabel = new Map(derived.map((entry) => [
    entry.operationLabel,
    entry.binding,
  ] as const));
  if (
    byLabel.size !== 4 || byLabel.has("block-0-conv-t1") ||
    ![1, 2, 3, 4].every((block) =>
      byLabel.has(`block-${block}-conv-t1`)
    )
  ) {
    throw new Error("OPT-0066 derived ConvTranspose binding inventory changed");
  }
  let destroyed = false;
  return Object.freeze({
    async createDispatch(
      ...args: Parameters<NativeCreateDispatch>
    ): Promise<NativeDispatch> {
      if (destroyed) {
        throw new Error("OPT-0066 complete transpose oracle was destroyed");
      }
      const [label, operationLabel, shape, bindings, range] = args;
      if (operationLabel === "block-0-conv-t1") {
        return await native.createDispatch(...args);
      }
      const weight = byLabel.get(operationLabel);
      if (weight === undefined) {
        throw new Error(
          `OPT-0066 has no derived ConvTranspose binding for ${operationLabel}`,
        );
      }
      const dispatch = await k4.createDispatch(
        label,
        operationLabel,
        shape,
        Object.freeze({
          input: bindings.input,
          weight,
          bias: bindings.bias,
          output: bindings.output,
        }),
        range,
      );
      return dispatch as unknown as NativeDispatch;
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      try {
        native.destroy();
      } finally {
        try {
          k4.destroy();
        } finally {
          destroyDerived();
        }
      }
    },
  });
}

/** Install the complete transpose oracle into an OPT40 diagnostic runtime. */
export function installOpt0066CompleteTransposeOracle(
  device: GPUDevice,
  runtime: AceOpt0011Fp16VaeDecoderRuntime,
  weights: readonly Opt0066DerivedTransposeWeight[],
): Readonly<{
  readonly derivedBufferCount: 4;
  readonly derivedWeightBytes: 15_335_424;
}> {
  type Replaceable = {
    shapeSelectedPackedConvTranspose1d: NativeOwner;
  };
  const replaceable = runtime as unknown as Replaceable;
  const native = replaceable.shapeSelectedPackedConvTranspose1d;
  if (native === undefined) {
    throw new Error("OPT-0066 requires the OPT40 transpose selector seam");
  }
  const buffers: GPUBuffer[] = [];
  let k4: AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel | undefined;
  try {
    const bindings = weights.map((entry): Opt0066DerivedTransposeBinding => {
      const buffer = device.createBuffer({
        label: `opt-0066-${entry.operationLabel}-revision6-derived-k4-weight`,
        size: entry.words.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      buffers.push(buffer);
      new Uint16Array(buffer.getMappedRange()).set(entry.words);
      buffer.unmap();
      return Object.freeze({
        operationLabel: entry.operationLabel,
        binding: Object.freeze({
          buffer,
          offset: 0,
          size: entry.words.byteLength,
        }),
      });
    });
    if (
      buffers.length !== 4 ||
      weights.reduce((sum, entry) => sum + entry.words.byteLength, 0) !==
        15_335_424
    ) {
      throw new Error("OPT-0066 derived ConvTranspose GPU inventory changed");
    }
    k4 = AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const owner = composeOpt0066CompleteTransposeOracleOwner(
      native,
      k4,
      bindings,
      () => {
        for (const buffer of buffers) buffer.destroy();
      },
    );
    replaceable.shapeSelectedPackedConvTranspose1d =
      owner as unknown as NativeOwner;
    return Object.freeze({
      derivedBufferCount: 4 as const,
      derivedWeightBytes: 15_335_424 as const,
    });
  } catch (error) {
    k4?.destroy();
    for (const buffer of buffers) buffer.destroy();
    throw error;
  }
}
