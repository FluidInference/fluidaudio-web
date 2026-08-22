from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert  # noqa: E402
from conversion_plan import (  # noqa: E402
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    DIT_DENSE_FP16_TILE_LAYOUT,
    DIT_DENSE_FP16_TILE_TRANSFORMATION,
    DIT_GEMM_TILE_LAYOUT,
    DIT_GEMM_TILE_TRANSFORMATION,
    EXPERIMENTAL_DIT_DENSE_PROFILE,
    EXPERIMENTAL_VAE_PROFILE,
    SOURCE_ROW_MAJOR_LAYOUT,
    VAE_BIAS_FP16_TRANSFORMATION,
    VAE_CHANNEL_VECTOR_FP16_LAYOUT,
    VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
    VAE_CHANNEL_VECTOR_LAYOUT,
    VAE_CHANNEL_VECTOR_TRANSFORMATION,
    VAE_CONV1D_FP16_LAYOUT,
    VAE_CONV1D_FP16_TRANSFORMATION,
    VAE_CONV1D_LAYOUT,
    VAE_CONV1D_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_LAYOUT,
    VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
    VAE_K1_FP16_TILE_LAYOUT,
    VAE_K1_FP16_TILE_TRANSFORMATION,
    VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    OutputTensorPlan,
)
from safetensors_mmap import SafetensorsFile, TensorInfo  # noqa: E402


def bf16_payload(values: np.ndarray) -> bytes:
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return np.right_shift(bits, np.uint32(16)).astype("<u2").tobytes()


def write_bf16_safetensors(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]],
) -> None:
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    cursor = 0
    for name, shape, values in tensors:
        payload = bf16_payload(values)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        payloads.append(payload)
        cursor += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def write_raw_bf16_safetensor(
    path: Path,
    name: str,
    shape: tuple[int, ...],
    bits: np.ndarray,
) -> None:
    payload = np.asarray(bits, dtype="<u2").reshape(shape).tobytes()
    encoded = json.dumps(
        {
            name: {
                "dtype": "BF16",
                "shape": list(shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def vae_plan(
    source_tensor: str,
    transformation: str,
    runtime_layout: str,
    *,
    output_dtype: str = "float32",
) -> OutputTensorPlan:
    return OutputTensorPlan(
        source="vae-weights",
        sourceTensor=source_tensor,
        output=f"vae.{source_tensor}",
        phase="vae",
        lifetime="vae",
        group="vae/shared",
        transformation=transformation,
        outputDtype=output_dtype,
        runtimeLayout=runtime_layout,
    )


def dit_gemm_plan(source_tensor: str) -> OutputTensorPlan:
    return OutputTensorPlan(
        source="ace-turbo-weights",
        sourceTensor=source_tensor,
        output=f"ace.{source_tensor}",
        phase="dit",
        lifetime="dit",
        group="dit/layer-00",
        transformation=DIT_GEMM_TILE_TRANSFORMATION,
        outputDtype="profile-float",
        runtimeLayout=DIT_GEMM_TILE_LAYOUT,
    )


def dit_dense_k4_plan(source_tensor: str) -> OutputTensorPlan:
    return OutputTensorPlan(
        source="ace-turbo-weights",
        sourceTensor=source_tensor,
        output=f"ace.{source_tensor}",
        phase="dit",
        lifetime="dit",
        group="dit/layer-00",
        transformation=DIT_DENSE_FP16_TILE_TRANSFORMATION,
        outputDtype="float16",
        runtimeLayout=DIT_DENSE_FP16_TILE_LAYOUT,
    )


def dit_dense_revision7_plan(source_tensor: str) -> OutputTensorPlan:
    return OutputTensorPlan(
        source="ace-turbo-weights",
        sourceTensor=source_tensor,
        output=f"ace.{source_tensor}",
        phase="dit",
        lifetime="dit",
        group="dit/layer-00",
        transformation=DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        outputDtype="float16",
        runtimeLayout=DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    )


class ConverterTransformTests(unittest.TestCase):
    def test_revision_7_k7_row_reuse_layout_is_exhaustively_bit_bijective(
        self,
    ) -> None:
        # This proves only physical-word identity to the native-layout K4
        # arithmetic oracle. K4 output is intentionally not claimed raw-equal
        # to the scalar-FP32 revision-6 production owner.
        for name, source_shape in VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES.items():
            output_channels, input_channels, kernel = source_shape
            with self.subTest(name=name):
                native = (
                    np.arange(
                        output_channels * kernel * input_channels,
                        dtype="<u2",
                    )
                    .reshape(output_channels, kernel, input_channels)
                )
                packed = convert.pack_vae_k7_row_reuse_u16(native)
                repeated = convert.pack_vae_k7_row_reuse_u16(native)
                self.assertEqual(
                    hashlib.sha256(packed.tobytes()).digest(),
                    hashlib.sha256(repeated.tobytes()).digest(),
                )
                self.assertEqual(
                    packed.shape,
                    (
                        7,
                        input_channels // 4,
                        output_channels // 64,
                        32,
                        2,
                        4,
                    ),
                )
                np.testing.assert_array_equal(
                    convert.unpack_vae_k7_row_reuse_u16(packed),
                    native,
                )

    def test_revision_7_transpose_k4_layout_is_exhaustively_rev6_bit_exact(
        self,
    ) -> None:
        for name, (source_shape, reuse_axis) in (
            VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS.items()
        ):
            if name == "decoder.block.0.conv_t1.weight_v":
                continue
            input_channels, output_channels, kernel = source_shape
            stride = kernel // 2
            with self.subTest(name=name):
                polyphase = np.arange(
                    stride * 2 * input_channels * output_channels,
                    dtype="<u2",
                ).reshape(stride, 2, input_channels, output_channels)
                packed = convert.pack_vae_conv_transpose_k4_u16(
                    polyphase,
                    reuse_axis,
                )
                repeated = convert.pack_vae_conv_transpose_k4_u16(
                    polyphase,
                    reuse_axis,
                )
                self.assertEqual(
                    hashlib.sha256(packed.tobytes()).digest(),
                    hashlib.sha256(repeated.tobytes()).digest(),
                )
                outputs_per_lane = 8 if reuse_axis == "channel" else 4
                self.assertEqual(
                    packed.shape,
                    (
                        stride,
                        2,
                        input_channels // 4,
                        output_channels // (32 * outputs_per_lane),
                        32,
                        outputs_per_lane,
                        4,
                    ),
                )
                np.testing.assert_array_equal(
                    convert.unpack_vae_conv_transpose_k4_u16(
                        packed,
                        reuse_axis,
                    ),
                    polyphase,
                )

    def test_revision_8_mixed_dit_target_pins_authenticated_canonical_replacement(
        self,
    ) -> None:
        target = convert.DEFAULT_OUTPUT[EXPERIMENTAL_DIT_DENSE_PROFILE]
        self.assertEqual(
            target.name,
            "files-fp16-dit-layer-mixed-experimental",
        )
        self.assertEqual(
            convert.canonical_replacement_manifest_sha256(
                EXPERIMENTAL_DIT_DENSE_PROFILE,
                target,
            ),
            "a2f70c123fb7c4dbc3b51be68b4b494107c13b575ad2bed68c639791c93574d1",
        )
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(convert.canonical_replacement_manifest_sha256(
                EXPERIMENTAL_DIT_DENSE_PROFILE,
                Path(temporary) / "noncanonical-target",
            ))

    def test_revision_7_vae_target_authenticates_only_the_exact_rev6_predecessor(
        self,
    ) -> None:
        target = convert.DEFAULT_OUTPUT[EXPERIMENTAL_VAE_PROFILE]
        self.assertEqual(
            convert.canonical_replacement_manifest_sha256(
                EXPERIMENTAL_VAE_PROFILE,
                target,
            ),
            "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949",
        )
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(convert.canonical_replacement_manifest_sha256(
                EXPERIMENTAL_VAE_PROFILE,
                Path(temporary) / "noncanonical-target",
            ))

    def test_production_selector_pins_exact_browser_tuple(self) -> None:
        preparations = convert.preparations_for_selector("production")
        self.assertEqual(
            [preparation.target.name for preparation in preparations],
            [
                "files-reference",
                "files-fp16-dit-rev7-oracle",
                "files-fp16-vae-revision7-experimental",
            ],
        )
        self.assertEqual(
            [preparation.converter_revision for preparation in preparations],
            [4, 7, 7],
        )
        self.assertEqual(
            [
                preparation.dit_dense_converter_revision
                for preparation in preparations
            ],
            [8, 7, 8],
        )
        self.assertEqual(
            [preparation.expected_manifest_sha256 for preparation in preparations],
            [
                "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
                "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f",
                "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7",
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dit"
            (dit,) = convert.preparations_for_selector(
                convert.PRODUCTION_DIT_PROFILE,
                output_dir=output,
            )
            self.assertEqual(dit.target, output)
            self.assertEqual(dit.converter_revision, 7)
            self.assertEqual(dit.dit_dense_converter_revision, 7)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            convert.preparations_for_selector(
                "production",
                output_dir=Path("not-allowed"),
            )

    def test_production_manifest_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_bytes(b"canonical manifest bytes")
            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(
                convert.require_manifest_identity(manifest, expected),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "production identity"):
                convert.require_manifest_identity(manifest, "0" * 64)

    def test_revision_7_production_dit_layout_is_exact_fp16_bijection(self) -> None:
        source_name = "decoder.layers.0.self_attn.q_proj.weight"
        columns = 256
        inner = 32
        source_values = (
            np.arange(columns * inner, dtype="<f4").reshape(columns, inner)
            / np.float32(257.0)
            - np.float32(4.0)
        )
        source_bf16 = convert.bf16_bytes_to_float32(
            bf16_payload(source_values)
        ).reshape(columns, inner)
        expected_logical_fp16 = source_bf16.astype("<f2")
        expected_packed = expected_logical_fp16.reshape(
            columns // 256,
            256,
            inner // 32,
            32,
        ).transpose(0, 2, 3, 1).copy(order="C")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dit-revision7.safetensors"
            write_bf16_safetensors(
                path,
                [(source_name, source_values.shape, source_values)],
            )
            with SafetensorsFile(path) as checkpoint:
                plan = dit_dense_revision7_plan(source_name)
                pieces = convert.split_output_plan(
                    plan,
                    checkpoint.tensor(source_name),
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                )
                observed_bytes = b"".join(convert._piece_bytes(
                    checkpoint,
                    pieces[0],
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                ))
        observed = np.frombuffer(observed_bytes, dtype="<f2").reshape(
            columns // 256,
            inner // 32,
            32,
            256,
        )
        np.testing.assert_array_equal(observed, expected_packed)

        flattened = observed.reshape(-1)
        visited: set[int] = set()
        for column in range(columns):
            for inner_index in range(inner):
                physical = (
                    (
                        column // 256 * (inner // 32)
                        + inner_index // 32
                    )
                    * 32
                    + inner_index % 32
                ) * 256 + column % 256
                visited.add(physical)
                self.assertEqual(
                    flattened[physical].view("<u2"),
                    expected_logical_fp16[column, inner_index].view("<u2"),
                )
        self.assertEqual(visited, set(range(columns * inner)))

    def test_revision_8_dit_k4_layout_is_an_exact_fp16_bijection(self) -> None:
        source_name = "decoder.layers.0.mlp.down_proj.weight"
        columns = 256
        inner = 12
        source_values = (
            np.arange(columns * inner, dtype="<f4").reshape(columns, inner)
            / np.float32(257.0)
            - np.float32(4.0)
        )
        source_bf16 = convert.bf16_bytes_to_float32(
            bf16_payload(source_values)
        ).reshape(columns, inner)
        expected_logical_fp16 = source_bf16.astype("<f2")
        expected_packed = expected_logical_fp16.reshape(
            columns // 128,
            32,
            4,
            inner // 4,
            4,
        ).transpose(0, 3, 2, 1, 4).copy(order="C")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dit-k4.safetensors"
            write_bf16_safetensors(
                path,
                [(source_name, source_values.shape, source_values)],
            )
            with SafetensorsFile(path) as checkpoint:
                plan = dit_dense_k4_plan(source_name)
                pieces = convert.split_output_plan(
                    plan,
                    checkpoint.tensor(source_name),
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                )
                self.assertEqual(len(pieces), 1)
                self.assertEqual(pieces[0].logical_shape, (columns, inner))
                observed_bytes = b"".join(convert._piece_bytes(
                    checkpoint,
                    pieces[0],
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                ))
        observed = np.frombuffer(observed_bytes, dtype="<f2").reshape(
            columns // 128,
            inner // 4,
            4,
            32,
            4,
        )
        np.testing.assert_array_equal(observed, expected_packed)

        flattened = observed.reshape(-1)
        visited: set[int] = set()
        for column in range(columns):
            for inner_index in range(inner):
                physical = (((((
                    column // 128 * (inner // 4) + inner_index // 4
                ) * 4 + column % 4) * 32 + (column % 128) // 4) * 4) +
                    inner_index % 4)
                visited.add(physical)
                self.assertEqual(
                    flattened[physical].view("<u2"),
                    expected_logical_fp16[column, inner_index].view("<u2"),
                )
        self.assertEqual(visited, set(range(columns * inner)))

    def test_revision_8_dit_k4_layout_fails_closed_on_shape_and_sharding(
        self,
    ) -> None:
        plan = dit_dense_k4_plan("decoder.layers.0.self_attn.q_proj.weight")
        for shape in ((127, 4), (128, 6), (128, 4, 1)):
            source = TensorInfo(
                plan.sourceTensor,
                "BF16",
                shape,
                0,
                int(np.prod(shape)) * 2,
            )
            with self.assertRaisesRegex(ValueError, "rank two.*divisible"):
                convert.split_output_plan(
                    plan,
                    source,
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                )
        source = TensorInfo(plan.sourceTensor, "BF16", (128, 4), 0, 1_024)
        with mock.patch.object(convert, "PIECE_TARGET_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "exceeds one package piece"):
                convert.split_output_plan(
                    plan,
                    source,
                    EXPERIMENTAL_DIT_DENSE_PROFILE,
                )

    def test_revision_6_vae_packed_weight_permutations_are_bit_exact(self) -> None:
        k1_source = (
            (np.arange(128 * 128, dtype="<f4") % 17) - 8
        ).reshape(128, 128, 1)
        k1_g = np.linspace(0.125, 1.125, 128, dtype="<f4").reshape(128, 1, 1)
        transpose_source = (
            (np.arange(3 * 4 * 4, dtype="<f4") % 13) - 6
        ).reshape(3, 4, 4)
        transpose_g = np.array([0.5, 1.0, 1.5], dtype="<f4").reshape(3, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vae-revision-6.safetensors"
            write_bf16_safetensors(
                path,
                [
                    ("decoder.block.0.res_unit1.conv2.weight_g", k1_g.shape, k1_g),
                    (
                        "decoder.block.0.res_unit1.conv2.weight_v",
                        k1_source.shape,
                        k1_source,
                    ),
                    ("decoder.block.0.conv_t1.weight_g", transpose_g.shape, transpose_g),
                    (
                        "decoder.block.0.conv_t1.weight_v",
                        transpose_source.shape,
                        transpose_source,
                    ),
                ],
            )
            with SafetensorsFile(path) as checkpoint:
                k1_plan = vae_plan(
                    "decoder.block.0.res_unit1.conv2.weight_v",
                    VAE_K1_FP16_TILE_TRANSFORMATION,
                    VAE_K1_FP16_TILE_LAYOUT,
                    output_dtype="float16",
                )
                transpose_plan = vae_plan(
                    "decoder.block.0.conv_t1.weight_v",
                    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
                    output_dtype="float16",
                )
                k1_piece = convert.split_output_plan(
                    k1_plan,
                    checkpoint.tensor(k1_plan.sourceTensor),
                    EXPERIMENTAL_VAE_PROFILE,
                )[0]
                transpose_piece = convert.split_output_plan(
                    transpose_plan,
                    checkpoint.tensor(transpose_plan.sourceTensor),
                    EXPERIMENTAL_VAE_PROFILE,
                )[0]
                observed_k1 = np.frombuffer(
                    b"".join(convert._piece_bytes(
                        checkpoint,
                        k1_piece,
                        EXPERIMENTAL_VAE_PROFILE,
                    )),
                    dtype="<f2",
                ).reshape(1, 4, 32, 128)
                observed_transpose = np.frombuffer(
                    b"".join(convert._piece_bytes(
                        checkpoint,
                        transpose_piece,
                        EXPERIMENTAL_VAE_PROFILE,
                    )),
                    dtype="<f2",
                ).reshape(2, 2, 3, 4)

        rounded_k1_source = convert.bf16_bytes_to_float32(
            bf16_payload(k1_source)
        ).reshape(k1_source.shape)
        rounded_k1_g = convert.bf16_bytes_to_float32(
            bf16_payload(k1_g)
        ).reshape(k1_g.shape)
        k1_matrix = np.stack([
            convert.fuse_weightnorm_row_fp32(
                rounded_k1_g[output_channel, 0, 0],
                rounded_k1_source[output_channel],
            )
            for output_channel in range(128)
        ]).reshape(128, 128)
        expected_k1 = (
            k1_matrix.reshape(1, 128, 4, 32)
            .transpose(0, 2, 3, 1)
            .astype("<f2")
        )

        rounded_transpose_source = convert.bf16_bytes_to_float32(
            bf16_payload(transpose_source)
        ).reshape(transpose_source.shape)
        rounded_transpose_g = convert.bf16_bytes_to_float32(
            bf16_payload(transpose_g)
        ).reshape(transpose_g.shape)
        native_transpose = np.stack([
            convert.fuse_weightnorm_row_fp32(
                rounded_transpose_g[input_channel, 0, 0],
                rounded_transpose_source[input_channel],
            ).reshape(4, 4)
            for input_channel in range(3)
        ]).transpose(1, 2, 0)
        expected_transpose = np.empty((2, 2, 3, 4), dtype="<f2")
        for phase in range(2):
            for tap in range(2):
                expected_transpose[phase, tap] = native_transpose[
                    :, phase + tap * 2, :
                ].T.astype("<f2")

        np.testing.assert_array_equal(observed_k1, expected_k1)
        np.testing.assert_array_equal(observed_transpose, expected_transpose)

    def test_revision_7_emitters_preserve_post_fusion_fp16_words(self) -> None:
        k7_name = "decoder.block.4.res_unit1.conv1.weight_v"
        k7_g_name = k7_name.removesuffix(".weight_v") + ".weight_g"
        k7_source = (
            (np.arange(128 * 128 * 7, dtype="<f4") % 29) - 14
        ).reshape(128, 128, 7)
        k7_g = np.linspace(0.25, 1.25, 128, dtype="<f4").reshape(128, 1, 1)
        transpose_name = "decoder.block.4.conv_t1.weight_v"
        transpose_g_name = transpose_name.removesuffix(".weight_v") + ".weight_g"
        transpose_source = (
            (np.arange(128 * 128 * 4, dtype="<f4") % 31) - 15
        ).reshape(128, 128, 4)
        transpose_g = np.linspace(0.5, 1.5, 128, dtype="<f4").reshape(128, 1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vae-revision-7.safetensors"
            write_bf16_safetensors(
                path,
                [
                    (k7_g_name, k7_g.shape, k7_g),
                    (k7_name, k7_source.shape, k7_source),
                    (transpose_g_name, transpose_g.shape, transpose_g),
                    (transpose_name, transpose_source.shape, transpose_source),
                ],
            )
            with SafetensorsFile(path) as checkpoint:
                k7_plan = vae_plan(
                    k7_name,
                    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                    VAE_K7_ROW_REUSE_FP16_LAYOUT,
                    output_dtype="float16",
                )
                transpose_plan = vae_plan(
                    transpose_name,
                    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
                    output_dtype="float16",
                )
                observed_k7 = np.frombuffer(
                    b"".join(convert._piece_bytes(
                        checkpoint,
                        convert.split_output_plan(
                            k7_plan,
                            checkpoint.tensor(k7_name),
                            EXPERIMENTAL_VAE_PROFILE,
                        )[0],
                        EXPERIMENTAL_VAE_PROFILE,
                    )),
                    dtype="<u2",
                ).reshape(7, 32, 2, 32, 2, 4)
                observed_transpose = np.frombuffer(
                    b"".join(convert._piece_bytes(
                        checkpoint,
                        convert.split_output_plan(
                            transpose_plan,
                            checkpoint.tensor(transpose_name),
                            EXPERIMENTAL_VAE_PROFILE,
                        )[0],
                        EXPERIMENTAL_VAE_PROFILE,
                    )),
                    dtype="<u2",
                ).reshape(2, 2, 32, 1, 32, 4, 4)

        rounded_k7_source = convert.bf16_bytes_to_float32(
            bf16_payload(k7_source)
        ).reshape(k7_source.shape)
        rounded_k7_g = convert.bf16_bytes_to_float32(
            bf16_payload(k7_g)
        ).reshape(k7_g.shape)
        native_k7 = np.stack([
            convert.fuse_weightnorm_row_fp32(
                rounded_k7_g[o, 0, 0],
                rounded_k7_source[o],
            )
            .reshape(128, 7)
            .T
            for o in range(128)
        ]).astype("<f2").view("<u2")
        expected_k7 = convert.pack_vae_k7_row_reuse_u16(native_k7)

        rounded_transpose_source = convert.bf16_bytes_to_float32(
            bf16_payload(transpose_source)
        ).reshape(transpose_source.shape)
        rounded_transpose_g = convert.bf16_bytes_to_float32(
            bf16_payload(transpose_g)
        ).reshape(transpose_g.shape)
        native_transpose = np.stack([
            convert.fuse_weightnorm_row_fp32(
                rounded_transpose_g[i, 0, 0],
                rounded_transpose_source[i],
            ).reshape(128, 4)
            for i in range(128)
        ])
        polyphase = np.empty((2, 2, 128, 128), dtype="<f2")
        for phase in range(2):
            polyphase[phase, 0] = native_transpose[:, :, phase]
            polyphase[phase, 1] = native_transpose[:, :, phase + 2]
        expected_transpose = convert.pack_vae_conv_transpose_k4_u16(
            polyphase.view("<u2"),
            "row",
        )
        np.testing.assert_array_equal(observed_k7, expected_k7)
        np.testing.assert_array_equal(observed_transpose, expected_transpose)

    def test_experimental_vae_rounds_only_after_deterministic_fp32_transforms(self) -> None:
        conv_source = np.arange(1, 13, dtype="<f4").reshape(2, 3, 2)
        conv_g = np.array([2.0, 3.0], dtype="<f4").reshape(2, 1, 1)
        transpose_source = np.arange(1, 13, dtype="<f4").reshape(3, 2, 2)
        transpose_g = np.array([1.0, 2.0, 3.0], dtype="<f4").reshape(3, 1, 1)
        snake = np.array([0.333, -2.001, 4.125, 7.777], dtype="<f4").reshape(1, 4, 1)
        bias = np.array([0.333, -2.001], dtype="<f4")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vae-fp16.safetensors"
            write_bf16_safetensors(
                path,
                [
                    ("decoder.conv1.weight_g", conv_g.shape, conv_g),
                    ("decoder.conv1.weight_v", conv_source.shape, conv_source),
                    (
                        "decoder.block.0.conv_t1.weight_g",
                        transpose_g.shape,
                        transpose_g,
                    ),
                    (
                        "decoder.block.0.conv_t1.weight_v",
                        transpose_source.shape,
                        transpose_source,
                    ),
                    ("decoder.snake1.alpha", snake.shape, snake),
                    ("decoder.conv1.bias", bias.shape, bias),
                ],
            )
            with SafetensorsFile(path) as checkpoint:
                plans = (
                    vae_plan(
                        "decoder.conv1.weight_v",
                        VAE_CONV1D_FP16_TRANSFORMATION,
                        VAE_CONV1D_FP16_LAYOUT,
                        output_dtype="float16",
                    ),
                    vae_plan(
                        "decoder.block.0.conv_t1.weight_v",
                        VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
                        VAE_CONV_TRANSPOSE1D_FP16_LAYOUT,
                        output_dtype="float16",
                    ),
                    vae_plan(
                        "decoder.snake1.alpha",
                        VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
                        VAE_CHANNEL_VECTOR_FP16_LAYOUT,
                        output_dtype="float16",
                    ),
                    vae_plan(
                        "decoder.conv1.bias",
                        VAE_BIAS_FP16_TRANSFORMATION,
                        SOURCE_ROW_MAJOR_LAYOUT,
                        output_dtype="float16",
                    ),
                )
                actual: list[np.ndarray] = []
                for plan in plans:
                    piece = convert.split_output_plan(
                        plan,
                        checkpoint.tensor(plan.sourceTensor),
                        EXPERIMENTAL_VAE_PROFILE,
                    )[0]
                    payload = b"".join(
                        convert._piece_bytes(
                            checkpoint,
                            piece,
                            EXPERIMENTAL_VAE_PROFILE,
                        )
                    )
                    self.assertEqual(len(payload), int(np.prod(piece.logical_shape)) * 2)
                    actual.append(
                        np.frombuffer(payload, dtype="<f2").reshape(piece.logical_shape)
                    )

        expected_conv = np.stack(
            [
                convert.fuse_weightnorm_row_fp32(conv_g[o, 0, 0], conv_source[o])
                for o in range(conv_source.shape[0])
            ]
        ).reshape(conv_source.shape).transpose(0, 2, 1)
        expected_transpose = np.stack(
            [
                convert.fuse_weightnorm_row_fp32(
                    transpose_g[i, 0, 0],
                    transpose_source[i],
                )
                for i in range(transpose_source.shape[0])
            ]
        ).reshape(transpose_source.shape).transpose(1, 2, 0)
        expected_snake = convert.bf16_bytes_to_float32(
            bf16_payload(snake)
        ).reshape(4)
        expected_bias = convert.bf16_bytes_to_float32(bf16_payload(bias))
        for observed, expected in zip(
            actual,
            (expected_conv, expected_transpose, expected_snake, expected_bias),
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected.astype("<f2"))

    def test_experimental_largest_conv_transpose_is_one_bounded_piece(self) -> None:
        experimental = vae_plan(
            "decoder.block.0.conv_t1.weight_v",
            VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
            VAE_CONV_TRANSPOSE1D_FP16_LAYOUT,
            output_dtype="float16",
        )
        source = TensorInfo(
            experimental.sourceTensor,
            "BF16",
            (2_048, 1_024, 20),
            0,
            2_048 * 1_024 * 20 * 2,
        )
        pieces = convert.split_output_plan(
            experimental,
            source,
            EXPERIMENTAL_VAE_PROFILE,
        )
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].logical_shape, (1_024, 20, 2_048))
        self.assertEqual(pieces[0].output_bytes, 83_886_080)
        self.assertLessEqual(pieces[0].output_bytes, convert.PIECE_TARGET_BYTES)

    def test_dit_gemm_tile_major_reference_and_fp16_index_mapping(self) -> None:
        source_name = "decoder.layers.0.self_attn.q_proj.weight"
        source_bits = (
            np.arange(256 * 64, dtype="<u2") * np.uint16(73)
            + np.uint16(0x1234)
        ).reshape(256, 64)
        expected_tiled = (
            source_bits.reshape(2, 128, 2, 32)
            .transpose(0, 2, 3, 1)
            .copy(order="C")
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dit.safetensors"
            write_raw_bf16_safetensor(
                path,
                source_name,
                source_bits.shape,
                source_bits,
            )
            with SafetensorsFile(path) as checkpoint:
                plan = dit_gemm_plan(source_name)
                pieces = convert.split_output_plan(
                    plan,
                    checkpoint.tensor(source_name),
                    "reference",
                )
                self.assertEqual(len(pieces), 1)
                reference = b"".join(
                    convert._piece_bytes(checkpoint, pieces[0], "reference")
                )
                fp16 = b"".join(
                    convert._piece_bytes(checkpoint, pieces[0], "fp16")
                )
                mixed_support = b"".join(
                    convert._piece_bytes(
                        checkpoint,
                        pieces[0],
                        EXPERIMENTAL_DIT_DENSE_PROFILE,
                    )
                )
        self.assertEqual(reference, expected_tiled.tobytes())
        self.assertEqual(mixed_support, reference)
        self.assertEqual(
            fp16,
            convert.bf16_bytes_to_fp16(expected_tiled.tobytes()),
        )
        # Pin the physical scalar address contract used by both GPU kernels.
        for column, inner in ((0, 0), (127, 31), (128, 32), (201, 47)):
            physical = (
                (((column // 128) * (64 // 32) + inner // 32) * 32 + inner % 32)
                * 128
                + column % 128
            )
            self.assertEqual(
                np.frombuffer(reference, dtype="<u2")[physical],
                source_bits[column, inner],
            )

    def test_dit_gemm_tile_major_fails_closed_on_shape_and_sharding(self) -> None:
        plan = dit_gemm_plan("decoder.layers.0.self_attn.q_proj.weight")
        for shape in ((127, 32), (128, 31), (128, 32, 1)):
            byte_length = int(np.prod(shape)) * 2
            source = TensorInfo(plan.sourceTensor, "BF16", shape, 0, byte_length)
            with self.assertRaisesRegex(ValueError, "rank two.*divisible"):
                convert.split_output_plan(plan, source, "reference")
        source = TensorInfo(plan.sourceTensor, "BF16", (128, 32), 0, 8192)
        with mock.patch.object(convert, "PIECE_TARGET_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "exceeds one package piece"):
                convert.split_output_plan(plan, source, "reference")

    def test_bf16_to_float32_and_fp16(self) -> None:
        source = np.array([0x3F80, 0xC020, 0x0000, 0x7F80], dtype="<u2").tobytes()
        values = convert.bf16_bytes_to_float32(source)
        np.testing.assert_array_equal(values[:3], np.array([1.0, -2.5, 0.0], dtype="<f4"))
        self.assertTrue(np.isinf(values[3]))
        fp16 = np.frombuffer(convert.bf16_bytes_to_fp16(source), dtype="<f2")
        np.testing.assert_array_equal(fp16[:3], np.array([1.0, -2.5, 0.0], dtype="<f2"))
        self.assertTrue(np.isinf(fp16[3]))

    def test_fp32_to_fp16_storage_uses_ieee_ties_to_even(self) -> None:
        values = np.array(
            [1.00048828125, 1.00146484375, 65_504.0],
            dtype="<f4",
        )
        bits = np.frombuffer(
            convert.fp32_values_to_fp16_bytes(values),
            dtype="<u2",
        )
        self.assertEqual(bits.tolist(), [0x3C00, 0x3C02, 0x7BFF])

    def test_fp32_to_fp16_storage_rejects_nonfinite_input_and_overflow(self) -> None:
        cases = (
            (np.array([65_520.0], dtype="<f4"), "overflowed"),
            (np.array([np.nan], dtype="<f4"), "input must be finite"),
            (np.array([np.inf], dtype="<f4"), "input must be finite"),
        )
        for values, message in cases:
            with self.subTest(values=values, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    convert.fp32_values_to_fp16_bytes(values)

    def test_vae_bias_transform_requires_rank_one_source(self) -> None:
        plan = vae_plan(
            "decoder.conv1.bias",
            VAE_BIAS_FP16_TRANSFORMATION,
            "source-row-major",
            output_dtype="float16",
        )
        source = TensorInfo(plan.sourceTensor, "BF16", (2, 1), 0, 4)
        with self.assertRaisesRegex(ValueError, "bias source must have rank 1"):
            convert.split_output_plan(plan, source, EXPERIMENTAL_VAE_PROFILE)

    def test_large_first_axis_tensor_is_row_sharded_under_target(self) -> None:
        plan = OutputTensorPlan(
            source="qwen-weights",
            sourceTensor="embed_tokens.weight",
            output="text.embed_tokens.weight",
            phase="text",
            lifetime="text",
            group="text/shared",
            transformation="profile-float-storage",
            outputDtype="profile-float",
            runtimeLayout=SOURCE_ROW_MAJOR_LAYOUT,
        )
        info = TensorInfo("embed_tokens.weight", "BF16", (60_000, 1024), 0, 60_000 * 1024 * 2)
        pieces = convert.split_output_plan(plan, info, "reference")
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(piece.output_bytes <= convert.PIECE_TARGET_BYTES for piece in pieces))
        self.assertEqual(sum(piece.row_end - piece.row_start for piece in pieces), 60_000)

    def test_conversion_rejects_unexpected_source_dtype(self) -> None:
        plan = OutputTensorPlan(
            source="qwen-weights",
            sourceTensor="x",
            output="text.x",
            phase="text",
            lifetime="text",
            group="text/shared",
            transformation="profile-float-storage",
            outputDtype="profile-float",
            runtimeLayout=SOURCE_ROW_MAJOR_LAYOUT,
        )
        info = TensorInfo("x", "F32", (2,), 0, 8)
        with self.assertRaisesRegex(ValueError, "require audited BF16"):
            convert.split_output_plan(plan, info, "reference")

    def test_weightnorm_requires_dim_zero_broadcast_shape(self) -> None:
        v = TensorInfo("v", "BF16", (4, 3, 2), 0, 48)
        valid_g = TensorInfo("g", "BF16", (4, 1, 1), 0, 8)
        convert.validate_weightnorm_pair("v", v, "g", valid_g)
        for wrong in (
            TensorInfo("g", "BF16", (1, 4, 1), 0, 8),
            TensorInfo("g", "F32", (4, 1, 1), 0, 16),
        ):
            with self.assertRaisesRegex(ValueError, "Incompatible weight"):
                convert.validate_weightnorm_pair("v", v, "g", wrong)

    def test_weightnorm_fp32_pairwise_oracle_bits(self) -> None:
        vectors = (
            (
                np.float32(2.0),
                np.array([3.0, 4.0], dtype="<f4"),
                0x40A00000,
                [0x3F99999A, 0x3FCCCCCD],
            ),
            (
                np.float32(1.5),
                np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype="<f4"),
                0x40ED517F,
                [0x3E4F1D3C, 0xBECF1D3C, 0x3F1B55ED, 0xBF4F1D3C, 0x3F817246],
            ),
            (
                np.float32(-0.75),
                np.array([4096.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="<f4"),
                0x45800001,
                [0xBF3FFFFF, *([0xB93FFFFF] * 6)],
            ),
        )
        for coefficient, values, norm_bits, output_bits in vectors:
            norm = convert.deterministic_fp32_norm(values)
            self.assertEqual(int(np.asarray(norm).view(np.uint32)), norm_bits)
            fused = convert.fuse_weightnorm_row_fp32(coefficient, values)
            self.assertEqual(fused.view(np.uint32).tolist(), output_bits)

        zeros = np.zeros(3, dtype="<f4")
        self.assertEqual(
            convert.fuse_weightnorm_row_fp32(np.float32(2.0), zeros)
            .view(np.uint32)
            .tolist(),
            [0, 0, 0],
        )

    def test_vae_operation_native_layouts_have_exact_index_mapping(self) -> None:
        conv1_source = np.arange(1, 13, dtype="<f4").reshape(2, 3, 2)
        conv1_g = np.array([2.0, 3.0], dtype="<f4").reshape(2, 1, 1)
        conv_t_source = np.arange(1, 13, dtype="<f4").reshape(3, 2, 2)
        conv_t_g = np.array([1.0, 2.0, 3.0], dtype="<f4").reshape(3, 1, 1)
        snake = np.array([4.0, 5.0, 6.0, 7.0], dtype="<f4").reshape(1, 4, 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vae.safetensors"
            write_bf16_safetensors(
                path,
                [
                    ("decoder.conv1.weight_g", conv1_g.shape, conv1_g),
                    ("decoder.conv1.weight_v", conv1_source.shape, conv1_source),
                    ("decoder.block.0.conv_t1.weight_g", conv_t_g.shape, conv_t_g),
                    ("decoder.block.0.conv_t1.weight_v", conv_t_source.shape, conv_t_source),
                    ("decoder.snake1.alpha", snake.shape, snake),
                ],
            )
            with SafetensorsFile(path) as checkpoint:
                conv1 = vae_plan(
                    "decoder.conv1.weight_v",
                    VAE_CONV1D_TRANSFORMATION,
                    VAE_CONV1D_LAYOUT,
                )
                conv1_piece = convert.split_output_plan(
                    conv1,
                    checkpoint.tensor(conv1.sourceTensor),
                    "reference",
                )[0]
                conv1_actual = np.frombuffer(
                    b"".join(convert._piece_bytes(checkpoint, conv1_piece, "reference")),
                    dtype="<f4",
                ).reshape(conv1_piece.logical_shape)
                conv1_fused = np.stack(
                    [
                        convert.fuse_weightnorm_row_fp32(conv1_g[o, 0, 0], conv1_source[o])
                        for o in range(conv1_source.shape[0])
                    ]
                ).reshape(conv1_source.shape)
                for o in range(2):
                    for k in range(2):
                        for i in range(3):
                            self.assertEqual(conv1_actual[o, k, i], conv1_fused[o, i, k])

                conv_t = vae_plan(
                    "decoder.block.0.conv_t1.weight_v",
                    VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_LAYOUT,
                )
                conv_t_piece = convert.split_output_plan(
                    conv_t,
                    checkpoint.tensor(conv_t.sourceTensor),
                    "reference",
                )[0]
                conv_t_actual = np.frombuffer(
                    b"".join(convert._piece_bytes(checkpoint, conv_t_piece, "reference")),
                    dtype="<f4",
                ).reshape(conv_t_piece.logical_shape)
                conv_t_fused = np.stack(
                    [
                        convert.fuse_weightnorm_row_fp32(conv_t_g[i, 0, 0], conv_t_source[i])
                        for i in range(conv_t_source.shape[0])
                    ]
                ).reshape(conv_t_source.shape)
                for o in range(2):
                    for k in range(2):
                        for i in range(3):
                            self.assertEqual(conv_t_actual[o, k, i], conv_t_fused[i, o, k])

                snake_plan = vae_plan(
                    "decoder.snake1.alpha",
                    VAE_CHANNEL_VECTOR_TRANSFORMATION,
                    VAE_CHANNEL_VECTOR_LAYOUT,
                )
                snake_piece = convert.split_output_plan(
                    snake_plan,
                    checkpoint.tensor(snake_plan.sourceTensor),
                    "reference",
                )[0]
                snake_actual = np.frombuffer(
                    b"".join(convert._piece_bytes(checkpoint, snake_piece, "reference")),
                    dtype="<f4",
                )
                np.testing.assert_array_equal(snake_actual, snake.reshape(4))

    def test_conv_transpose_shards_post_transform_runtime_axis_zero(self) -> None:
        plan = vae_plan(
            "decoder.block.0.conv_t1.weight_v",
            VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
            VAE_CONV_TRANSPOSE1D_LAYOUT,
        )
        source = TensorInfo(plan.sourceTensor, "BF16", (3, 5, 2), 0, 60)
        with mock.patch.object(convert, "PIECE_TARGET_BYTES", 3 * 2 * 4 * 2):
            pieces = convert.split_output_plan(plan, source, "reference")
        self.assertEqual(pieces[0].logical_shape, (5, 2, 3))
        self.assertEqual(
            [(piece.row_start, piece.row_end) for piece in pieces],
            [(0, 2), (2, 4), (4, 5)],
        )


if __name__ == "__main__":
    unittest.main()
