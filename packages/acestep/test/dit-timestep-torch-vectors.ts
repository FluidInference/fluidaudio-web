/**
 * Independent ACE Turbo timestep fixtures generated from the pinned upstream
 * expression at source commit 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
 * with PyTorch 2.10.0 on macOS arm64. These literals must not be regenerated
 * from the WebGPU implementation under test.
 */

export const ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS = Object.freeze([
  0, 1, 7, 31, 63, 64, 95, 127,
  128, 129, 135, 159, 191, 192, 223, 255,
]);

export interface AceTorchTimestepVector {
  readonly timestep: number;
  /** Result of upstream BF16 `t * 1000` before `.float()`. */
  readonly scaledBfloat16: number;
  readonly selectedFloat32Bits: readonly number[];
  readonly selectedFloat16Bits: readonly number[];
}

export const ACE_TORCH_210_TIMESTEP_VECTORS: readonly AceTorchTimestepVector[] =
  Object.freeze([
    Object.freeze({
      timestep: 1,
      scaledBfloat16: 1_000,
      selectedFloat32Bits: Object.freeze([
        0x3f0ff813, 0x3f4a24f9, 0x3ee3648d, 0x3f4c51ed,
        0xbe7cd7e0, 0xbf56cd64, 0x3ef3c069, 0x3f7e85f7,
        0x3f53ae61, 0x3f1d1414, 0x3f655de7, 0x3f1a3ce6,
        0xbf7812c8, 0xbf0b44f8, 0x3f612095, 0x3ddba804,
      ]),
      selectedFloat16Bits: Object.freeze([
        0x3880, 0x3a51, 0x371b, 0x3a63,
        0xb3e7, 0xbab6, 0x379e, 0x3bf4,
        0x3a9d, 0x38e9, 0x3b2b, 0x38d2,
        0xbbc1, 0xb85a, 0x3b09, 0x2edd,
      ]),
    }),
    Object.freeze({
      timestep: 0.953125,
      scaledBfloat16: 952,
      selectedFloat32Bits: Object.freeze([
        0xbf7ec938, 0x3f7fec4a, 0xbf6ddc62, 0xbe4c5468,
        0xbf3158be, 0xbf7ed71d, 0x3f055225, 0x3f7ea95b,
        0xbdc7367f, 0xbcc8e624, 0xbebd4bae, 0x3f7ada0a,
        0xbf389e76, 0xbdc2b81a, 0x3f5a8b4d, 0x3dd1268a,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xbbf6, 0x3bff, 0xbb6f, 0xb263,
        0xb98b, 0xbbf7, 0x382b, 0x3bf5,
        0xae3a, 0xa647, 0xb5ea, 0x3bd7,
        0xb9c5, 0xae16, 0x3ad4, 0x2e89,
      ]),
    }),
    Object.freeze({
      timestep: 0.8984375,
      scaledBfloat16: 900,
      selectedFloat32Bits: Object.freeze([
        0x3d87ac5a, 0xbe8daf88, 0xbf6e8fc8, 0xbf47e978,
        0xbf783ff4, 0xbf693fd5, 0x3f115190, 0x3f7ecdbc,
        0x3f7f7009, 0x3f7600ac, 0xbeb9bc13, 0x3f1fe8f4,
        0xbe7a0e96, 0x3ed30132, 0x3f52c1c9, 0x3dc5c356,
      ]),
      selectedFloat16Bits: Object.freeze([
        0x2c3d, 0xb46d, 0xbb74, 0xba3f,
        0xbbc2, 0xbb4a, 0x388b, 0x3bf6,
        0x3bfc, 0x3bb0, 0xb5ce, 0x38ff,
        0xb3d0, 0x3698, 0x3a96, 0x2e2e,
      ]),
    }),
    Object.freeze({
      timestep: 0.83203125,
      scaledBfloat16: 832,
      selectedFloat32Bits: Object.freeze([
        0xbf5de66b, 0x3e29d679, 0x3f7e2a7d, 0x3e02bfd1,
        0xbf62974c, 0xbee60eea, 0x3f20515b, 0x3f7efa3d,
        0x3eff507b, 0x3f7c7450, 0x3df4b54e, 0x3f7de793,
        0x3eee43c4, 0x3f64b38b, 0x3f4795c8, 0x3db6dcc6,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xbaef, 0x314f, 0x3bf1, 0x3016,
        0xbb15, 0xb730, 0x3903, 0x3bf8,
        0x37fb, 0x3be4, 0x2fa6, 0x3bef,
        0x3772, 0x3b26, 0x3a3d, 0x2db7,
      ]),
    }),
    Object.freeze({
      timestep: 0.75,
      scaledBfloat16: 752,
      selectedFloat32Bits: Object.freeze([
        0xbeccc3a2, 0xbf35175e, 0xbee83715, 0x3f24e57a,
        0xbe668664, 0x3ea7d65b, 0x3f30dcd7, 0x3f7f2a21,
        0xbf6aa2bd, 0x3f34f286, 0x3f6427dd, 0xbf43d1bb,
        0x3f796dbb, 0x3f71dac9, 0x3f39152b, 0x3da551df,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xb666, 0xb9a9, 0xb742, 0x3927,
        0xb334, 0x353f, 0x3987, 0x3bf9,
        0xbb55, 0x39a8, 0x3b21, 0xba1f,
        0x3bcb, 0x3b8f, 0x39c9, 0x2d2b,
      ]),
    }),
    Object.freeze({
      timestep: 0.64453125,
      scaledBfloat16: 644,
      selectedFloat32Bits: Object.freeze([
        0xbf7fe900, 0xbf3a4bcd, 0x3f6cb768, 0x3f7ef88b,
        0x3f4dc016, 0x3f7cdbdb, 0x3f451ade, 0x3f7f6321,
        0x3cd9036a, 0x3f2f9585, 0xbec2f314, 0x3db773ba,
        0x3f18531d, 0x3e1febc6, 0x3f235b7e, 0x3d8d9e41,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xbbff, 0xb9d2, 0x3b66, 0x3bf8,
        0x3a6e, 0x3be7, 0x3a29, 0x3bfb,
        0x26c8, 0x397d, 0xb618, 0x2dbc,
        0x38c3, 0x30ff, 0x391b, 0x2c6d,
      ]),
    }),
    Object.freeze({
      timestep: 0.5,
      scaledBfloat16: 500,
      selectedFloat32Bits: Object.freeze([
        0xbf6243f2, 0x3f722980, 0x3f5988c3, 0xbf72bc85,
        0x3f1d16d6, 0x3e913c2c, 0x3f5bed70, 0x3f7fa16c,
        0xbeef7fc9, 0x3ea60deb, 0x3f06f663, 0xbea2aa6e,
        0xbf4a22d4, 0xbf757c10, 0x3f0306bd, 0x3d5bf948,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xbb12, 0x3b91, 0x3acc, 0xbb96,
        0x38e9, 0x348a, 0x3adf, 0x3bfd,
        0xb77c, 0x3530, 0x3838, 0xb515,
        0xba51, 0xbbac, 0x3818, 0x2ae0,
      ]),
    }),
    Object.freeze({
      timestep: 0.30078125,
      scaledBfloat16: 300,
      selectedFloat32Bits: Object.freeze([
        0xbcb503f8, 0xbf68adff, 0x3f1a622d, 0x3f2e377a,
        0xbf7f228d, 0xbf7d7026, 0x3f72cfce, 0x3f7fddf2,
        0xbf7ff000, 0x3ed58177, 0xbf4c35c4, 0x3f3b934f,
        0xbda837da, 0x3e1081c3, 0x3ea23726, 0x3d040660,
      ]),
      selectedFloat16Bits: Object.freeze([
        0xa5a8, 0xbb45, 0x38d3, 0x3972,
        0xbbf9, 0xbbec, 0x3b96, 0x3bff,
        0xbc00, 0x36ac, 0xba62, 0x39dd,
        0xad42, 0x3084, 0x3512, 0x2820,
      ]),
    }),
  ]);
