"""Account for every source tensor and assign runtime-native package groups."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from safetensors_mmap import TensorInfo


class TensorInventory(Protocol):
    tensors: dict[str, TensorInfo]


@dataclass(frozen=True, slots=True)
class SourceDecision:
    source: str
    tensor: str
    disposition: str
    reason: str
    output: str | None


@dataclass(frozen=True, slots=True)
class OutputTensorPlan:
    source: str
    sourceTensor: str
    output: str
    phase: str
    lifetime: str
    group: str
    transformation: str
    outputDtype: str
    runtimeLayout: str


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    decisions: tuple[SourceDecision, ...]
    outputs: tuple[OutputTensorPlan, ...]

    def as_json(self) -> dict[str, object]:
        included = sum(decision.disposition == "included" for decision in self.decisions)
        transformed = sum(
            decision.disposition == "consumed-by-transform" for decision in self.decisions
        )
        excluded = sum(decision.disposition == "excluded" for decision in self.decisions)
        return {
            "schema": "ace-step-conversion-plan-v2",
            "summary": {
                "sourceTensors": len(self.decisions),
                "directlyIncluded": included,
                "consumedByTransform": transformed,
                "excluded": excluded,
                "outputTensorsBeforeRowSharding": len(self.outputs),
            },
            "decisions": [asdict(decision) for decision in self.decisions],
            "outputs": [asdict(output) for output in self.outputs],
        }


_LAYER = re.compile(r"(?:^|\.)(?:layers?|block)\.(\d+)(?:\.|$)")

SOURCE_ROW_MAJOR_LAYOUT = "source-row-major"
DIT_GEMM_TILE_LAYOUT = "dit-gemm-n128-k32-tile-major-v1"
DIT_GEMM_TILE_TRANSFORMATION = (
    "profile-float-dit-gemm-n128-k32-tile-major-v1"
)
DIT_DENSE_FP16_REV7_TILE_LAYOUT = "dit-gemm-n256-k32-tile-major-v1"
DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION = (
    "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1"
)
DIT_DENSE_FP16_REV8_TILE_LAYOUT = "dit-gemm-n128-k4-output4-lane32-k4-v1"
DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION = (
    "bf16-to-ieee-fp16-dit-gemm-n128-k4-output4-lane32-k4-v1"
)
# Existing experimental callers continue to mean the current revision-8 K4
# recipe. Production revision 7 is selected explicitly by the converter.
DIT_DENSE_FP16_TILE_LAYOUT = DIT_DENSE_FP16_REV8_TILE_LAYOUT
DIT_DENSE_FP16_TILE_TRANSFORMATION = DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION
VAE_CONV1D_LAYOUT = "conv1d-output-kernel-input-f32-v1"
VAE_CONV_TRANSPOSE1D_LAYOUT = "conv-transpose1d-output-kernel-input-f32-v1"
VAE_CHANNEL_VECTOR_LAYOUT = "channel-vector-f32-v1"
VAE_CONV1D_FP16_LAYOUT = "conv1d-output-kernel-input-f16-v1"
VAE_CONV_TRANSPOSE1D_FP16_LAYOUT = (
    "conv-transpose1d-output-kernel-input-f16-v1"
)
VAE_K1_FP16_TILE_LAYOUT = "conv1d-k1-cout128-cin32-tile-major-f16-v1"
VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT = (
    "conv-transpose1d-phase-tap-input-output-f16-v1"
)
VAE_K7_ROW_REUSE_FP16_LAYOUT = (
    "k7-cin4-cout-band64-lane32-output2-cin-element4"
)
VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT = (
    "ace-opt-0048-phase-tap-cin4-cout-tile-lane-output-k4-f16-v1"
)
VAE_CHANNEL_VECTOR_FP16_LAYOUT = "channel-vector-f16-v1"
VAE_CONV1D_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-oik-to-oki-v1"
)
VAE_CONV_TRANSPOSE1D_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-iok-to-oki-v1"
)
VAE_CHANNEL_VECTOR_TRANSFORMATION = "bf16-to-fp32-flatten-1-c-1-to-c-v1"
VAE_BIAS_FP16_TRANSFORMATION = "bf16-to-fp32-to-ieee-fp16-v1"
VAE_CONV1D_FP16_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1"
)
VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1"
)
VAE_K1_FP16_TILE_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-"
    "tile-major-ieee-fp16-v1"
)
VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-"
    "ieee-fp16-v1"
)
VAE_K7_ROW_REUSE_FP16_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-oik-to-k7-cin4-cout-band64-"
    "lane32-output2-cin-element4-ieee-fp16-v1"
)
VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION = (
    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-cin4-cout-tile-"
    "lane-output-k4-ieee-fp16-v1"
)
VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION = (
    "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1"
)
EXPERIMENTAL_VAE_PROFILE = "fp16-vae-experimental"
EXPERIMENTAL_DIT_DENSE_PROFILE = "fp16-dit-dense-experimental"

ACE_ENCODE_ONLY_PREFIX_CONTRACT: dict[str, tuple[int, int]] = {
    "tokenizer.audio_acoustic_proj.": (2, 266_240),
    "tokenizer.attention_pooler.": (26, 209_744_896),
    "tokenizer.quantizer.project_in.": (2, 24_588),
}
ACE_ENCODE_ONLY_TENSOR_COUNT = 30
ACE_ENCODE_ONLY_SOURCE_BYTES = 210_035_724
ACE_ENCODE_ONLY_PARAMETER_COUNT = 105_017_862

_VAE_CONV_TRANSPOSE_WEIGHT = re.compile(
    r"^decoder\.block\.\d+\.conv_t1\.weight_v$"
)
_VAE_CONV1D_WEIGHT = re.compile(
    r"^decoder\.(?:conv[12]|block\.\d+\.res_unit[123]\.conv[12])\.weight_v$"
)
_VAE_SNAKE_PARAMETER = re.compile(
    r"^decoder\.(?:snake1|block\.\d+\.(?:snake1|res_unit[123]\.snake[12]))\."
    r"(?:alpha|beta)$"
)
_VAE_BIAS = re.compile(
    r"^decoder\.(?:conv1|block\.[0-4]\."
    r"(?:conv_t1|res_unit[123]\.conv[12]))\.bias$"
)


def _revision7_vae_contracts() -> tuple[
    Mapping[str, tuple[int, ...]],
    Mapping[str, tuple[int, ...]],
    Mapping[str, tuple[int, ...]],
    Mapping[str, tuple[tuple[int, ...], str]],
    Mapping[str, tuple[int, ...]],
]:
    """Build the exact pinned decoder inventory for VAE package revision 7."""

    block_channels = (1_024, 512, 256, 128, 128)
    block_inputs = (2_048, 1_024, 512, 256, 128)
    transpose_kernels = (20, 12, 8, 8, 4)

    row_reuse_k7 = {
        f"decoder.block.{block}.res_unit{unit}.conv1.weight_v":
            (channels, channels, 7)
        for block, channels in enumerate(block_channels)
        if block != 2
        for unit in range(1, 4)
    }
    native_k7 = {
        "decoder.conv1.weight_v": (2_048, 64, 7),
        "decoder.conv2.weight_v": (2, 128, 7),
        **{
            f"decoder.block.{block}.res_unit{unit}.conv1.weight_v":
                (channels, channels, 7)
            for block, channels in enumerate(block_channels)
            if block == 2
            for unit in range(1, 4)
        },
    }
    k1 = {
        f"decoder.block.{block}.res_unit{unit}.conv2.weight_v":
            (channels, channels, 1)
        for block, channels in enumerate(block_channels)
        for unit in range(1, 4)
    }
    transpose = {
        f"decoder.block.{block}.conv_t1.weight_v": (
            (input_channels, output_channels, kernel),
            "channel" if block <= 2 else "row",
        )
        for block, (input_channels, output_channels, kernel) in enumerate(
            zip(block_inputs, block_channels, transpose_kernels, strict=True)
        )
    }

    runtime_shapes: dict[str, tuple[int, ...]] = {}
    for name, source_shape in {**native_k7, **row_reuse_k7, **k1}.items():
        output_channels, input_channels, kernel = source_shape
        runtime_shapes[name] = (output_channels, kernel, input_channels)
    for name, (source_shape, _) in transpose.items():
        input_channels, output_channels, kernel = source_shape
        runtime_shapes[name] = (output_channels, kernel, input_channels)

    runtime_shapes["decoder.conv1.bias"] = (2_048,)
    for block, channels in enumerate(block_channels):
        runtime_shapes[f"decoder.block.{block}.conv_t1.bias"] = (channels,)
        for unit in range(1, 4):
            for convolution in (1, 2):
                runtime_shapes[
                    f"decoder.block.{block}.res_unit{unit}.conv{convolution}.bias"
                ] = (channels,)
            for snake in (1, 2):
                for parameter in ("alpha", "beta"):
                    runtime_shapes[
                        f"decoder.block.{block}.res_unit{unit}."
                        f"snake{snake}.{parameter}"
                    ] = (channels,)
        for parameter in ("alpha", "beta"):
            runtime_shapes[f"decoder.block.{block}.snake1.{parameter}"] = (
                block_inputs[block],
            )
    for parameter in ("alpha", "beta"):
        runtime_shapes[f"decoder.snake1.{parameter}"] = (128,)

    if len(runtime_shapes) != 145:
        raise RuntimeError(
            f"Revision-7 VAE contract has {len(runtime_shapes)} tensors"
        )
    return (
        MappingProxyType(row_reuse_k7),
        MappingProxyType(native_k7),
        MappingProxyType(k1),
        MappingProxyType(transpose),
        MappingProxyType(runtime_shapes),
    )


(
    VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES,
    VAE_REVISION7_NATIVE_K7_SOURCE_SHAPES,
    VAE_REVISION7_K1_SOURCE_SHAPES,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE,
) = _revision7_vae_contracts()

VAE_REVISION7_WEIGHT_SOURCE_SHAPES: Mapping[str, tuple[int, ...]] = (
    MappingProxyType({
        **VAE_REVISION7_NATIVE_K7_SOURCE_SHAPES,
        **VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES,
        **VAE_REVISION7_K1_SOURCE_SHAPES,
        **{
            name: contract[0]
            for name, contract in VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS.items()
        },
    })
)


def revision7_vae_weight_layout(
    source_tensor: str,
    source_shape: tuple[int, ...],
) -> tuple[str, str]:
    """Select one revision-7 physical layout by exact source label and shape."""

    expected_shape = VAE_REVISION7_WEIGHT_SOURCE_SHAPES.get(source_tensor)
    if expected_shape is None or tuple(source_shape) != expected_shape:
        raise ValueError(
            f"Unsupported revision-7 VAE weight {source_tensor!r}: "
            f"{tuple(source_shape)}"
        )
    if source_tensor in VAE_REVISION7_ROW_REUSE_K7_SOURCE_SHAPES:
        return (
            VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
            VAE_K7_ROW_REUSE_FP16_LAYOUT,
        )
    if source_tensor in VAE_REVISION7_NATIVE_K7_SOURCE_SHAPES:
        return VAE_CONV1D_FP16_TRANSFORMATION, VAE_CONV1D_FP16_LAYOUT
    if source_tensor in VAE_REVISION7_K1_SOURCE_SHAPES:
        return VAE_K1_FP16_TILE_TRANSFORMATION, VAE_K1_FP16_TILE_LAYOUT
    if source_tensor == "decoder.block.0.conv_t1.weight_v":
        return (
            VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
            VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
        )
    return (
        VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    )


def revision7_vae_runtime_weight_layout(
    source_tensor: str,
    logical_shape: tuple[int, ...],
) -> tuple[str, str]:
    """Validate a manifest logical shape and return its exact layout contract."""

    if source_tensor in VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS:
        if len(logical_shape) != 3:
            source_shape = logical_shape
        else:
            source_shape = (
                logical_shape[2],
                logical_shape[0],
                logical_shape[1],
            )
    elif len(logical_shape) != 3:
        source_shape = logical_shape
    else:
        source_shape = (
            logical_shape[0],
            logical_shape[2],
            logical_shape[1],
        )
    return revision7_vae_weight_layout(source_tensor, tuple(source_shape))


def canonical_dit_gemm_weight_names() -> frozenset[str]:
    """Exact rank-two DiT weights consumed by the production GEMM kernel."""

    names = {
        "decoder.condition_embedder.weight",
        "decoder.time_embed.linear_1.weight",
        "decoder.time_embed.linear_2.weight",
        "decoder.time_embed.time_proj.weight",
        "decoder.time_embed_r.linear_1.weight",
        "decoder.time_embed_r.linear_2.weight",
        "decoder.time_embed_r.time_proj.weight",
    }
    layer_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight",
        "cross_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for layer in range(24):
        names.update(
            f"decoder.layers.{layer}.{suffix}" for suffix in layer_suffixes
        )
    if len(names) != 271:
        raise RuntimeError(f"Canonical DiT GEMM allowlist has {len(names)} tensors")
    return frozenset(names)


DIT_GEMM_WEIGHT_NAMES = canonical_dit_gemm_weight_names()


def canonical_repeated_dit_dense_weight_names() -> frozenset[str]:
    """The nine denoise-time dense matrices repeated in every DiT layer."""

    suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    names = frozenset(
        f"decoder.layers.{layer}.{suffix}"
        for layer in range(24)
        for suffix in suffixes
    )
    if len(names) != 216:
        raise RuntimeError(f"Repeated DiT dense allowlist has {len(names)} tensors")
    return names


DIT_REPEATED_DENSE_WEIGHT_NAMES = canonical_repeated_dit_dense_weight_names()


def _group(phase: str, name: str) -> str:
    match = _LAYER.search(name)
    if match:
        return f"{phase}/layer-{int(match.group(1)):02d}"
    if phase == "dit" and name.startswith("decoder.time_embed"):
        return "dit/time-embedding"
    return f"{phase}/shared"


def _ordinary_output(
    source: str,
    name: str,
    *,
    phase: str,
    prefix: str,
) -> OutputTensorPlan:
    output_name = f"{prefix}.{name}"
    return OutputTensorPlan(
        source=source,
        sourceTensor=name,
        output=output_name,
        phase=phase,
        lifetime=phase,
        group=_group(phase, name),
        transformation="profile-float-storage",
        outputDtype="profile-float",
        runtimeLayout=SOURCE_ROW_MAJOR_LAYOUT,
    )


def _excluded_encode_only_prefix(name: str) -> str | None:
    return next(
        (prefix for prefix in ACE_ENCODE_ONLY_PREFIX_CONTRACT if name.startswith(prefix)),
        None,
    )


def assert_canonical_encode_only_exclusions(
    inventory: TensorInventory,
    plan: ConversionPlan,
) -> None:
    """Fail unless the pinned encode-only branches are excluded exactly.

    The source header itself is authenticated separately. These byte/count
    checks make the scoped removal visible and prevent a future classifier
    change from silently retaining or dropping a different branch.
    """

    decisions = {
        decision.tensor: decision
        for decision in plan.decisions
        if decision.source == "ace-turbo-weights"
    }
    total_count = 0
    total_bytes = 0
    for prefix, (expected_count, expected_bytes) in (
        ACE_ENCODE_ONLY_PREFIX_CONTRACT.items()
    ):
        tensors = [
            tensor
            for name, tensor in inventory.tensors.items()
            if name.startswith(prefix)
        ]
        actual_bytes = sum(tensor.byte_length for tensor in tensors)
        if len(tensors) != expected_count or actual_bytes != expected_bytes:
            raise ValueError(
                f"Pinned encode-only prefix {prefix!r} changed: "
                f"got {len(tensors)} tensors/{actual_bytes} bytes, expected "
                f"{expected_count}/{expected_bytes}"
            )
        for tensor in tensors:
            decision = decisions.get(tensor.name)
            if decision is None or decision.disposition != "excluded":
                raise ValueError(
                    f"Encode-only tensor {tensor.name!r} is not exactly excluded"
                )
        total_count += len(tensors)
        total_bytes += actual_bytes
    if (
        total_count != ACE_ENCODE_ONLY_TENSOR_COUNT
        or total_bytes != ACE_ENCODE_ONLY_SOURCE_BYTES
        or total_bytes // 2 != ACE_ENCODE_ONLY_PARAMETER_COUNT
    ):
        raise ValueError("Pinned encode-only exclusion totals changed")


def build_conversion_plan(
    inventories: Mapping[str, TensorInventory],
    *,
    profile: str = "reference",
    dit_dense_converter_revision: int = 8,
) -> ConversionPlan:
    if profile not in {
        "reference",
        "fp16",
        EXPERIMENTAL_VAE_PROFILE,
        EXPERIMENTAL_DIT_DENSE_PROFILE,
    }:
        raise ValueError(f"Unsupported package profile {profile!r}")
    experimental_vae = profile == EXPERIMENTAL_VAE_PROFILE
    experimental_dit_dense = profile == EXPERIMENTAL_DIT_DENSE_PROFILE
    if dit_dense_converter_revision not in {7, 8}:
        raise ValueError(
            "Experimental mixed DiT converter revision must be 7 or 8"
        )
    dit_dense_transformation = (
        DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION
        if dit_dense_converter_revision == 7
        else DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION
    )
    dit_dense_layout = (
        DIT_DENSE_FP16_REV7_TILE_LAYOUT
        if dit_dense_converter_revision == 7
        else DIT_DENSE_FP16_REV8_TILE_LAYOUT
    )
    required = {"ace-turbo-weights", "qwen-weights", "planner-weights", "vae-weights"}
    if set(inventories) != required:
        raise ValueError(
            f"Expected inventories {sorted(required)}, got {sorted(inventories)}"
        )
    decisions: list[SourceDecision] = []
    outputs: list[OutputTensorPlan] = []

    def include(source: str, name: str, output: OutputTensorPlan) -> None:
        decisions.append(
            SourceDecision(
                source=source,
                tensor=name,
                disposition="included",
                reason="required by scoped text-to-music runtime",
                output=output.output,
            )
        )
        outputs.append(output)

    for source in sorted(inventories):
        inventory = inventories[source]
        for name in sorted(inventory.tensors):
            if experimental_dit_dense:
                if source == "ace-turbo-weights" and name.startswith(
                    "decoder.layers."
                ):
                    output = _ordinary_output(
                        source,
                        name,
                        phase="dit",
                        prefix="ace",
                    )
                    if name in DIT_REPEATED_DENSE_WEIGHT_NAMES:
                        output = OutputTensorPlan(
                            source=source,
                            sourceTensor=name,
                            output=f"ace.{name}",
                            phase="dit",
                            lifetime="dit",
                            group=_group("dit", name),
                            transformation=dit_dense_transformation,
                            outputDtype="float16",
                            runtimeLayout=dit_dense_layout,
                        )
                    elif name in DIT_GEMM_WEIGHT_NAMES:
                        output = OutputTensorPlan(
                            source=output.source,
                            sourceTensor=output.sourceTensor,
                            output=output.output,
                            phase=output.phase,
                            lifetime=output.lifetime,
                            group=output.group,
                            transformation=DIT_GEMM_TILE_TRANSFORMATION,
                            outputDtype=output.outputDtype,
                            runtimeLayout=DIT_GEMM_TILE_LAYOUT,
                        )
                    include(source, name, output)
                else:
                    decisions.append(
                        SourceDecision(
                            source=source,
                            tensor=name,
                            disposition="excluded",
                            reason=(
                                "outside the OPT-0009 mixed repeated-layer package"
                            ),
                            output=None,
                        )
                    )
                continue
            if source == "qwen-weights":
                include(
                    source,
                    name,
                    _ordinary_output(source, name, phase="text", prefix="text"),
                )
                continue
            if source == "planner-weights":
                include(
                    source,
                    name,
                    _ordinary_output(source, name, phase="planner", prefix="planner"),
                )
                continue
            if source == "ace-turbo-weights":
                if name.startswith("decoder."):
                    output = _ordinary_output(
                        source,
                        name,
                        phase="dit",
                        prefix="ace",
                    )
                    if name in DIT_GEMM_WEIGHT_NAMES:
                        output = OutputTensorPlan(
                            source=output.source,
                            sourceTensor=output.sourceTensor,
                            output=output.output,
                            phase=output.phase,
                            lifetime=output.lifetime,
                            group=output.group,
                            transformation=DIT_GEMM_TILE_TRANSFORMATION,
                            outputDtype=output.outputDtype,
                            runtimeLayout=DIT_GEMM_TILE_LAYOUT,
                        )
                    include(
                        source,
                        name,
                        output,
                    )
                elif name.startswith("encoder."):
                    include(
                        source,
                        name,
                        _ordinary_output(
                            source,
                            name,
                            phase="conditioner",
                            prefix="ace",
                        ),
                    )
                elif name.startswith("detokenizer."):
                    include(
                        source,
                        name,
                        _ordinary_output(
                            source,
                            name,
                            phase="semantic",
                            prefix="ace",
                        ),
                    )
                elif _excluded_encode_only_prefix(name) is not None:
                    decisions.append(
                        SourceDecision(
                            source=source,
                            tensor=name,
                            disposition="excluded",
                            reason=(
                                "encode-only audio conditioning is outside the v1 "
                                "text-to-music generation graph"
                            ),
                            output=None,
                        )
                    )
                elif name.startswith("tokenizer."):
                    include(
                        source,
                        name,
                        _ordinary_output(
                            source,
                            name,
                            phase="semantic",
                            prefix="ace",
                        ),
                    )
                elif name == "null_condition_emb":
                    decisions.append(
                        SourceDecision(
                            source=source,
                            tensor=name,
                            disposition="excluded",
                            reason=(
                                "reference-audio/null conditioning is outside the v1 "
                                "text-to-music scope"
                            ),
                            output=None,
                        )
                    )
                else:
                    raise ValueError(f"Unclassified ACE tensor {name!r}")
                continue
            if source == "vae-weights":
                if name.startswith("encoder."):
                    decisions.append(
                        SourceDecision(
                            source=source,
                            tensor=name,
                            disposition="excluded",
                            reason="VAE encoder is outside decoder-only v1 generation",
                            output=None,
                        )
                    )
                elif name.startswith("decoder.") and name.endswith(".weight_g"):
                    companion = name.removesuffix(".weight_g") + ".weight_v"
                    if (
                        companion not in inventory.tensors
                        or (
                            _VAE_CONV1D_WEIGHT.fullmatch(companion) is None
                            and _VAE_CONV_TRANSPOSE_WEIGHT.fullmatch(companion) is None
                        )
                    ):
                        raise ValueError(
                            f"Weight-normalization source {name!r} has no exact "
                            "decoder weight_v companion"
                        )
                    output_name = f"vae.{name.removesuffix('.weight_g')}.weight"
                    decisions.append(
                        SourceDecision(
                            source=source,
                            tensor=name,
                            disposition="consumed-by-transform",
                            reason="fused with weight_v by offline weight normalization",
                            output=output_name,
                        )
                    )
                elif name.startswith("decoder."):
                    output_name = (
                        f"vae.{name.removesuffix('.weight_v')}.weight"
                        if name.endswith(".weight_v")
                        else f"vae.{name}"
                    )
                    if name.endswith(".weight_v"):
                        companion = name.removesuffix(".weight_v") + ".weight_g"
                        if companion not in inventory.tensors:
                            raise ValueError(
                                f"Weight-normalization source {name!r} has no exact "
                                "decoder weight_g companion"
                            )
                    if (
                        _VAE_CONV_TRANSPOSE_WEIGHT.fullmatch(name)
                        or _VAE_CONV1D_WEIGHT.fullmatch(name)
                    ) and experimental_vae:
                        source_shape = inventory.tensors[name].shape
                        transform, runtime_layout = revision7_vae_weight_layout(
                            name,
                            tuple(source_shape),
                        )
                    elif _VAE_CONV_TRANSPOSE_WEIGHT.fullmatch(name):
                        transform = VAE_CONV_TRANSPOSE1D_TRANSFORMATION
                        runtime_layout = VAE_CONV_TRANSPOSE1D_LAYOUT
                    elif _VAE_CONV1D_WEIGHT.fullmatch(name):
                        transform = VAE_CONV1D_TRANSFORMATION
                        runtime_layout = VAE_CONV1D_LAYOUT
                    elif _VAE_SNAKE_PARAMETER.fullmatch(name):
                        if experimental_vae:
                            expected_runtime_shape = (
                                VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.get(name)
                            )
                            source_shape = tuple(inventory.tensors[name].shape)
                            if (
                                expected_runtime_shape is None
                                or source_shape
                                != (1, expected_runtime_shape[0], 1)
                            ):
                                raise ValueError(
                                    f"Unsupported revision-7 VAE Snake "
                                    f"{name!r}: {source_shape}"
                                )
                        transform = (
                            VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION
                            if experimental_vae
                            else VAE_CHANNEL_VECTOR_TRANSFORMATION
                        )
                        runtime_layout = (
                            VAE_CHANNEL_VECTOR_FP16_LAYOUT
                            if experimental_vae
                            else VAE_CHANNEL_VECTOR_LAYOUT
                        )
                    elif _VAE_BIAS.fullmatch(name):
                        if experimental_vae:
                            expected_runtime_shape = (
                                VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.get(name)
                            )
                            source_shape = tuple(inventory.tensors[name].shape)
                            if (
                                expected_runtime_shape is None
                                or source_shape != expected_runtime_shape
                            ):
                                raise ValueError(
                                    f"Unsupported revision-7 VAE bias "
                                    f"{name!r}: {source_shape}"
                                )
                        transform = (
                            VAE_BIAS_FP16_TRANSFORMATION
                            if experimental_vae
                            else "bf16-to-fp32"
                        )
                        runtime_layout = SOURCE_ROW_MAJOR_LAYOUT
                    else:
                        raise ValueError(
                            f"Unclassified VAE decoder tensor layout {name!r}"
                        )
                    output = OutputTensorPlan(
                        source=source,
                        sourceTensor=name,
                        output=output_name,
                        phase="vae",
                        lifetime="vae",
                        group=_group("vae", name),
                        transformation=transform,
                        outputDtype="float16" if experimental_vae else "float32",
                        runtimeLayout=runtime_layout,
                    )
                    include(source, name, output)
                else:
                    raise ValueError(f"Unclassified VAE tensor {name!r}")

    if len(decisions) != sum(len(item.tensors) for item in inventories.values()):
        raise RuntimeError("Conversion plan did not account for every source tensor")
    decision_keys = {(item.source, item.tensor) for item in decisions}
    if len(decision_keys) != len(decisions):
        raise RuntimeError("Conversion plan accounts for a source tensor more than once")
    output_names = {item.output for item in outputs}
    if len(output_names) != len(outputs):
        raise RuntimeError("Conversion plan emits duplicate output tensor names")
    return ConversionPlan(tuple(decisions), tuple(outputs))
