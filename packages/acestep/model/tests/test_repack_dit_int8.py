import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repack_dit_int8 import PACKED_WORDS_PER_TILE, pack_dense_tensor


class RepackDitInt8Test(unittest.TestCase):
    def test_packs_one_n256_k32_tile_with_fp16_scales(self) -> None:
        source = np.arange(32 * 256, dtype=np.float32).reshape(32, 256)
        source = ((source % 255) - 127).astype("<f2")

        payload, shape = pack_dense_tensor(source.tobytes(), [256, 32])

        self.assertEqual(shape, [1, 1, PACKED_WORDS_PER_TILE])
        self.assertEqual(len(payload), 8_704)
        quantized = np.frombuffer(payload[:8_192], dtype=np.int8).reshape(32, 256)
        scales = np.frombuffer(payload[8_192:], dtype="<f2")
        expected_scales = (
            np.max(np.abs(source.astype(np.float32)), axis=0) / np.float32(127)
        ).astype("<f2")
        np.testing.assert_array_equal(scales, expected_scales)
        expected_quantized = np.clip(
            np.rint(source.astype(np.float32) / expected_scales.astype(np.float32)),
            -127,
            127,
        ).astype(np.int8)
        np.testing.assert_array_equal(quantized, expected_quantized)

    def test_rejects_non_native_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "N256/K32"):
            pack_dense_tensor(bytes(255 * 32 * 2), [255, 32])


if __name__ == "__main__":
    unittest.main()
