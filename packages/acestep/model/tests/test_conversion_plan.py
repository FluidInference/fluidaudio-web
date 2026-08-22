from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion_plan import (  # noqa: E402
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    ACE_ENCODE_ONLY_PARAMETER_COUNT,
    ACE_ENCODE_ONLY_PREFIX_CONTRACT,
    ACE_ENCODE_ONLY_SOURCE_BYTES,
    ACE_ENCODE_ONLY_TENSOR_COUNT,
    DIT_GEMM_TILE_LAYOUT,
    DIT_GEMM_TILE_TRANSFORMATION,
    DIT_GEMM_WEIGHT_NAMES,
    DIT_DENSE_FP16_TILE_LAYOUT,
    DIT_DENSE_FP16_TILE_TRANSFORMATION,
    DIT_REPEATED_DENSE_WEIGHT_NAMES,
    EXPERIMENTAL_DIT_DENSE_PROFILE,
    EXPERIMENTAL_VAE_PROFILE,
    VAE_BIAS_FP16_TRANSFORMATION,
    VAE_CHANNEL_VECTOR_FP16_LAYOUT,
    VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
    VAE_CONV1D_FP16_LAYOUT,
    VAE_CONV1D_FP16_TRANSFORMATION,
    VAE_CONV1D_LAYOUT,
    VAE_CONV1D_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
    VAE_K1_FP16_TILE_LAYOUT,
    VAE_K1_FP16_TILE_TRANSFORMATION,
    VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_REVISION7_NATIVE_K7_SOURCE_SHAPES,
    VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES,
    VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    VAE_REVISION7_WEIGHT_SOURCE_SHAPES,
    assert_canonical_encode_only_exclusions,
    build_conversion_plan,
    revision7_vae_weight_layout,
)
from safetensors_mmap import TensorInfo  # noqa: E402
from source_contract import (  # noqa: E402
    ACE_REVISION,
    EXPECTED_SAFETENSOR_COUNT,
    PLANNER_REVISION,
    SAFETENSOR_ARTIFACTS,
    SOURCE_ARTIFACTS,
)


def info(name: str, shape: tuple[int, ...] = (2,)) -> TensorInfo:
    return TensorInfo(name, "BF16", shape, 0, 2 * max(1, shape[0]))


class SourceContractTests(unittest.TestCase):
    def test_revisions_and_audited_tensor_total_are_fixed(self) -> None:
        self.assertEqual(ACE_REVISION, "19671f406d603126926c1b7e2adc169acbcade22")
        self.assertEqual(PLANNER_REVISION, "148d8ea0225bdab342ee1ae3a354275ccd60ca80")
        self.assertEqual(sum(item.safetensors.tensor_count for item in SAFETENSOR_ARTIFACTS), EXPECTED_SAFETENSOR_COUNT)
        self.assertEqual(EXPECTED_SAFETENSOR_COUNT, 1662)

    def test_complete_tokenizer_and_chat_contract(self) -> None:
        keys = {artifact.key for artifact in SOURCE_ARTIFACTS}
        for prefix in ("qwen", "planner"):
            for suffix in (
                "tokenizer",
                "tokenizer-config",
                "merges",
                "vocab",
                "added-tokens",
                "special-tokens",
                "chat-template",
            ):
                self.assertIn(f"{prefix}-{suffix}", keys)


class ConversionPlanTests(unittest.TestCase):
    def test_revision_7_vae_selector_closes_over_exact_production_inventory(
        self,
    ) -> None:
        self.assertEqual(len(VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES), 12)
        self.assertEqual(len(VAE_REVISION7_NATIVE_K7_SOURCE_SHAPES), 5)
        self.assertEqual(len(VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS), 5)
        self.assertEqual(len(VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE), 145)
        self.assertEqual(
            sum(
                math.prod(shape)
                for shape in VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.values()
            ),
            84_395_776,
        )
        for name, shape in VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES.items():
            self.assertEqual(
                revision7_vae_weight_layout(name, shape),
                (
                    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                    VAE_K7_ROW_REUSE_FP16_LAYOUT,
                ),
            )
        for name, (shape, _) in VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS.items():
            expected = (
                (
                    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
                )
                if name == "decoder.block.0.conv_t1.weight_v"
                else (
                    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
                )
            )
            self.assertEqual(revision7_vae_weight_layout(name, shape), expected)
        with self.assertRaisesRegex(ValueError, "Unsupported revision-7"):
            revision7_vae_weight_layout(
                "decoder.block.2.res_unit1.conv1.weight_v",
                (512, 512, 7),
            )

    def test_revision_7_vae_plan_accounts_every_replaced_tensor_once(self) -> None:
        vae_tensors: dict[str, TensorInfo] = {}
        for name, logical_shape in VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.items():
            source_shape = VAE_REVISION7_WEIGHT_SOURCE_SHAPES.get(name)
            if source_shape is not None:
                vae_tensors[name] = info(name, source_shape)
                g_name = name.removesuffix(".weight_v") + ".weight_g"
                vae_tensors[g_name] = info(
                    g_name,
                    (source_shape[0], 1, 1),
                )
            elif name.endswith(".bias"):
                vae_tensors[name] = info(name, logical_shape)
            else:
                vae_tensors[name] = info(name, (1, logical_shape[0], 1))
        plan = build_conversion_plan(
            {
                "ace-turbo-weights": SimpleNamespace(tensors={}),
                "qwen-weights": SimpleNamespace(tensors={}),
                "planner-weights": SimpleNamespace(tensors={}),
                "vae-weights": SimpleNamespace(tensors=vae_tensors),
            },
            profile=EXPERIMENTAL_VAE_PROFILE,
        )
        self.assertEqual(len(plan.outputs), 145)
        self.assertEqual(len(plan.decisions), 182)
        self.assertEqual(
            sum(
                decision.disposition == "consumed-by-transform"
                for decision in plan.decisions
            ),
            37,
        )
        transformations = [output.transformation for output in plan.outputs]
        self.assertEqual(
            transformations.count(VAE_K7_ROW_REUSE_FP16_TRANSFORMATION),
            12,
        )
        self.assertEqual(
            transformations.count(VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION),
            4,
        )
        self.assertEqual(len({output.output for output in plan.outputs}), 145)

    def test_experimental_dit_package_mixes_only_dense_layer_weights(self) -> None:
        inventories = {
            "ace-turbo-weights": SimpleNamespace(
                tensors={
                    "decoder.layers.0.self_attn.q_proj.weight": info(
                        "dense", (256, 32)
                    ),
                    "decoder.layers.0.cross_attn.k_proj.weight": info(
                        "cross-cache", (128, 32)
                    ),
                    "decoder.layers.0.self_attn_norm.weight": info("norm"),
                    "decoder.time_embed.linear_1.weight": info(
                        "excluded-shared", (128, 32)
                    ),
                }
            ),
            "qwen-weights": SimpleNamespace(tensors={}),
            "planner-weights": SimpleNamespace(tensors={}),
            "vae-weights": SimpleNamespace(tensors={}),
        }
        plan = build_conversion_plan(
            inventories,
            profile=EXPERIMENTAL_DIT_DENSE_PROFILE,
        )
        outputs = {output.sourceTensor: output for output in plan.outputs}
        self.assertEqual(
            set(outputs),
            {
                "decoder.layers.0.self_attn.q_proj.weight",
                "decoder.layers.0.cross_attn.k_proj.weight",
                "decoder.layers.0.self_attn_norm.weight",
            },
        )
        dense = outputs["decoder.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(dense.outputDtype, "float16")
        self.assertEqual(dense.transformation, DIT_DENSE_FP16_TILE_TRANSFORMATION)
        self.assertEqual(dense.runtimeLayout, DIT_DENSE_FP16_TILE_LAYOUT)
        self.assertEqual(
            DIT_DENSE_FP16_TILE_TRANSFORMATION,
            "bf16-to-ieee-fp16-dit-gemm-n128-k4-output4-lane32-k4-v1",
        )
        self.assertEqual(
            DIT_DENSE_FP16_TILE_LAYOUT,
            "dit-gemm-n128-k4-output4-lane32-k4-v1",
        )
        self.assertEqual(len(DIT_REPEATED_DENSE_WEIGHT_NAMES), 24 * 9)
        cross_cache = outputs["decoder.layers.0.cross_attn.k_proj.weight"]
        self.assertEqual(cross_cache.outputDtype, "profile-float")
        self.assertEqual(cross_cache.transformation, DIT_GEMM_TILE_TRANSFORMATION)
        self.assertEqual(cross_cache.runtimeLayout, DIT_GEMM_TILE_LAYOUT)
        norm = outputs["decoder.layers.0.self_attn_norm.weight"]
        self.assertEqual(norm.outputDtype, "profile-float")
        self.assertEqual(norm.transformation, "profile-float-storage")

    def test_production_revision_7_dit_plan_is_separately_selectable(self) -> None:
        inventories = {
            "ace-turbo-weights": SimpleNamespace(
                tensors={
                    "decoder.layers.0.self_attn.q_proj.weight": info(
                        "dense", (256, 32)
                    ),
                    "decoder.layers.0.cross_attn.k_proj.weight": info(
                        "cross-cache", (128, 32)
                    ),
                }
            ),
            "qwen-weights": SimpleNamespace(tensors={}),
            "planner-weights": SimpleNamespace(tensors={}),
            "vae-weights": SimpleNamespace(tensors={}),
        }
        plan = build_conversion_plan(
            inventories,
            profile=EXPERIMENTAL_DIT_DENSE_PROFILE,
            dit_dense_converter_revision=7,
        )
        outputs = {output.sourceTensor: output for output in plan.outputs}
        dense = outputs["decoder.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(
            dense.transformation,
            DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        )
        self.assertEqual(dense.runtimeLayout, DIT_DENSE_FP16_REV7_TILE_LAYOUT)
        cross_cache = outputs["decoder.layers.0.cross_attn.k_proj.weight"]
        self.assertEqual(cross_cache.transformation, DIT_GEMM_TILE_TRANSFORMATION)

    def test_experimental_profile_is_vae_only_and_authenticates_weightnorm_pairs(self) -> None:
        inventories = {
            "ace-turbo-weights": SimpleNamespace(
                tensors={"decoder.norm.weight": info("ace")}
            ),
            "qwen-weights": SimpleNamespace(
                tensors={"layers.0.norm.weight": info("qwen")}
            ),
            "planner-weights": SimpleNamespace(
                tensors={"model.layers.0.norm.weight": info("planner")}
            ),
            "vae-weights": SimpleNamespace(
                tensors={
                    "decoder.conv1.weight_g": info("conv-g", (2_048, 1, 1)),
                    "decoder.conv1.weight_v": info("conv-v", (2_048, 64, 7)),
                    "decoder.conv1.bias": info("conv-bias", (2_048,)),
                    "decoder.block.1.res_unit1.conv1.weight_g": info(
                        "row-reuse-g", (512, 1, 1)
                    ),
                    "decoder.block.1.res_unit1.conv1.weight_v": info(
                        "row-reuse-v", (512, 512, 7)
                    ),
                    "decoder.block.0.res_unit1.conv2.weight_g": info(
                        "k1-g", (1_024, 1, 1)
                    ),
                    "decoder.block.0.res_unit1.conv2.weight_v": info(
                        "k1-v", (1_024, 1_024, 1)
                    ),
                    "decoder.block.0.conv_t1.weight_g": info(
                        "transpose-g", (2_048, 1, 1)
                    ),
                    "decoder.block.0.conv_t1.weight_v": info(
                        "transpose-v", (2_048, 1_024, 20)
                    ),
                    "decoder.snake1.alpha": info("snake", (1, 128, 1)),
                }
            ),
        }
        reference = build_conversion_plan(inventories, profile="reference")
        self.assertEqual(reference, build_conversion_plan(inventories))
        self.assertEqual(reference, build_conversion_plan(inventories, profile="fp16"))

        experimental = build_conversion_plan(
            inventories,
            profile=EXPERIMENTAL_VAE_PROFILE,
        )
        reference_outputs = {output.output: output for output in reference.outputs}
        outputs = {output.output: output for output in experimental.outputs}
        for name in ("ace.decoder.norm.weight", "text.layers.0.norm.weight", "planner.model.layers.0.norm.weight"):
            self.assertEqual(outputs[name], reference_outputs[name])
        expected = {
            "vae.decoder.conv1.weight": (
                VAE_CONV1D_FP16_TRANSFORMATION,
                VAE_CONV1D_FP16_LAYOUT,
            ),
            "vae.decoder.block.1.res_unit1.conv1.weight": (
                VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                VAE_K7_ROW_REUSE_FP16_LAYOUT,
            ),
            "vae.decoder.block.0.conv_t1.weight": (
                VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
                VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
            ),
            "vae.decoder.block.0.res_unit1.conv2.weight": (
                VAE_K1_FP16_TILE_TRANSFORMATION,
                VAE_K1_FP16_TILE_LAYOUT,
            ),
            "vae.decoder.snake1.alpha": (
                VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
                VAE_CHANNEL_VECTOR_FP16_LAYOUT,
            ),
            "vae.decoder.conv1.bias": (
                VAE_BIAS_FP16_TRANSFORMATION,
                "source-row-major",
            ),
        }
        for name, (transformation, layout) in expected.items():
            self.assertEqual(outputs[name].outputDtype, "float16")
            self.assertEqual(outputs[name].transformation, transformation)
            self.assertEqual(outputs[name].runtimeLayout, layout)

        consumed = {
            (decision.tensor, decision.output)
            for decision in experimental.decisions
            if decision.disposition == "consumed-by-transform"
        }
        self.assertEqual(
            consumed,
            {
                ("decoder.conv1.weight_g", "vae.decoder.conv1.weight"),
                (
                    "decoder.block.0.conv_t1.weight_g",
                    "vae.decoder.block.0.conv_t1.weight",
                ),
                (
                    "decoder.block.0.res_unit1.conv2.weight_g",
                    "vae.decoder.block.0.res_unit1.conv2.weight",
                ),
                (
                    "decoder.block.1.res_unit1.conv1.weight_g",
                    "vae.decoder.block.1.res_unit1.conv1.weight",
                ),
            },
        )

        missing_g = dict(inventories)
        missing_g["vae-weights"] = SimpleNamespace(
            tensors={
                name: tensor
                for name, tensor in inventories["vae-weights"].tensors.items()
                if name != "decoder.conv1.weight_g"
            }
        )
        with self.assertRaisesRegex(ValueError, "weight_g companion"):
            build_conversion_plan(missing_g, profile=EXPERIMENTAL_VAE_PROFILE)

        unexpected_bias = dict(inventories)
        unexpected_bias["vae-weights"] = SimpleNamespace(
            tensors={
                **inventories["vae-weights"].tensors,
                "decoder.fixture.bias": info("unexpected-bias"),
            }
        )
        with self.assertRaisesRegex(ValueError, "Unclassified VAE decoder tensor"):
            build_conversion_plan(
                unexpected_bias,
                profile=EXPERIMENTAL_VAE_PROFILE,
            )

    def test_exact_canonical_dit_gemm_allowlist_uses_tile_major_layout(self) -> None:
        false_friends = {
            "decoder.layers.24.self_attn.q_proj.weight": info("outside-layer"),
            "decoder.layers.0.self_attn.q_proj.bias": info("bias"),
            "decoder.layers.0.norm1.weight": info("norm"),
            "decoder.time_embed.linear_1.bias": info("time-bias"),
        }
        inventories = {
            "ace-turbo-weights": SimpleNamespace(
                tensors={
                    **{
                        name: info(name, (128, 32))
                        for name in DIT_GEMM_WEIGHT_NAMES
                    },
                    **false_friends,
                }
            ),
            "qwen-weights": SimpleNamespace(tensors={}),
            "planner-weights": SimpleNamespace(tensors={}),
            "vae-weights": SimpleNamespace(tensors={}),
        }
        plan = build_conversion_plan(inventories)
        outputs = {output.sourceTensor: output for output in plan.outputs}
        tiled = {
            name
            for name, output in outputs.items()
            if output.transformation == DIT_GEMM_TILE_TRANSFORMATION
        }
        self.assertEqual(len(DIT_GEMM_WEIGHT_NAMES), 271)
        self.assertEqual(tiled, DIT_GEMM_WEIGHT_NAMES)
        for name in DIT_GEMM_WEIGHT_NAMES:
            self.assertEqual(outputs[name].runtimeLayout, DIT_GEMM_TILE_LAYOUT)
            self.assertEqual(outputs[name].outputDtype, "profile-float")
            self.assertEqual(outputs[name].phase, "dit")
        for name in false_friends:
            self.assertEqual(outputs[name].transformation, "profile-float-storage")
            self.assertEqual(outputs[name].runtimeLayout, "source-row-major")

    def test_accounts_for_included_transformed_and_excluded_tensors(self) -> None:
        inventories = {
            "ace-turbo-weights": SimpleNamespace(
                tensors={
                    "decoder.layers.0.mlp.weight": info("a"),
                    "encoder.lyric.weight": info("b"),
                    "detokenizer.norm.weight": info("c"),
                    "tokenizer.quantizer.project_out.weight": info("d"),
                    "tokenizer.quantizer.project_in.weight": info("e"),
                    "tokenizer.audio_acoustic_proj.weight": info("e2"),
                    "tokenizer.attention_pooler.norm.weight": info("e3"),
                    "null_condition_emb": info("f"),
                }
            ),
            "qwen-weights": SimpleNamespace(tensors={"layers.0.norm.weight": info("g")}),
            "planner-weights": SimpleNamespace(tensors={"model.layers.0.norm.weight": info("h")}),
            "vae-weights": SimpleNamespace(
                tensors={
                    "decoder.conv1.weight_g": info("i"),
                    "decoder.conv1.weight_v": info("j", (2, 1)),
                    "decoder.conv1.bias": info("k"),
                    "encoder.conv1.weight_v": info("l"),
                }
            ),
        }
        plan = build_conversion_plan(inventories)
        summary = plan.as_json()["summary"]
        self.assertEqual(summary["sourceTensors"], 14)
        self.assertEqual(summary["consumedByTransform"], 1)
        self.assertEqual(summary["excluded"], 5)
        output_names = {item.output for item in plan.outputs}
        self.assertNotIn("ace.tokenizer.quantizer.project_in.weight", output_names)
        self.assertIn("ace.tokenizer.quantizer.project_out.weight", output_names)
        self.assertIn("ace.detokenizer.norm.weight", output_names)
        outputs = {output.output: output for output in plan.outputs}
        fused = outputs["vae.decoder.conv1.weight"]
        self.assertEqual(fused.transformation, VAE_CONV1D_TRANSFORMATION)
        self.assertEqual(fused.runtimeLayout, VAE_CONV1D_LAYOUT)
        self.assertEqual(fused.group, "vae/shared")

    def test_encode_only_exclusion_contract_fixes_exact_counts_and_bytes(self) -> None:
        ace_tensors: dict[str, TensorInfo] = {}
        for prefix, (count, byte_length) in ACE_ENCODE_ONLY_PREFIX_CONTRACT.items():
            remaining = byte_length
            for index in range(count):
                item_bytes = remaining if index == count - 1 else 2
                name = f"{prefix}fixture_{index:02d}"
                ace_tensors[name] = TensorInfo(
                    name,
                    "BF16",
                    (item_bytes // 2,),
                    0,
                    item_bytes,
                )
                remaining -= item_bytes
        inventories = {
            "ace-turbo-weights": SimpleNamespace(tensors=ace_tensors),
            "qwen-weights": SimpleNamespace(tensors={}),
            "planner-weights": SimpleNamespace(tensors={}),
            "vae-weights": SimpleNamespace(tensors={}),
        }
        plan = build_conversion_plan(inventories)
        assert_canonical_encode_only_exclusions(
            inventories["ace-turbo-weights"],
            plan,
        )
        self.assertEqual(ACE_ENCODE_ONLY_TENSOR_COUNT, 30)
        self.assertEqual(ACE_ENCODE_ONLY_SOURCE_BYTES, 210_035_724)
        self.assertEqual(ACE_ENCODE_ONLY_PARAMETER_COUNT, 105_017_862)
        self.assertEqual(
            sum(item.disposition == "excluded" for item in plan.decisions),
            ACE_ENCODE_ONLY_TENSOR_COUNT,
        )

        first_name = next(iter(ace_tensors))
        original = ace_tensors[first_name]
        ace_tensors[first_name] = TensorInfo(
            original.name,
            original.dtype,
            (original.parameter_count + 1,),
            original.data_start,
            original.data_end + 2,
        )
        with self.assertRaisesRegex(ValueError, "prefix.*changed"):
            assert_canonical_encode_only_exclusions(
                inventories["ace-turbo-weights"],
                plan,
            )


if __name__ == "__main__":
    unittest.main()
