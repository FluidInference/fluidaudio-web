"""PyTorch-side deterministic injection and tensor serialization primitives.

This module is imported inside the pinned upstream Python 3.12 environment.
It deliberately does not import PyTorch at module import time, allowing the
repository-owned contract tests to remain on Python 3.13 without installing the
multi-gigabyte upstream dependency set.
"""

from __future__ import annotations

import math
import os
import platform
import json
import importlib.metadata
import subprocess
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifacts import CaptureWriter
from .contracts import TapContract
from .inputs import InputBundle, load_prng_oracle


def stable_descending_token_order(values: Iterable[float]) -> tuple[int, ...]:
    """Sort by descending logit and then ascending original token ID."""

    materialized = tuple(float(value) for value in values)
    if any(math.isnan(value) for value in materialized):
        raise ValueError("planner logits contain NaN")
    return tuple(sorted(range(len(materialized)), key=lambda token: (-materialized[token], token)))


def top_p_keep_mask(values: Iterable[float], top_p: float | None) -> tuple[bool, ...]:
    """Weight-free oracle for pinned upstream top-p cutoff semantics.

    Sorting is stable by token ID. Like upstream, the first item whose
    cumulative probability crosses ``top_p`` is retained by shifting the
    removal mask right one position.
    """

    logits = tuple(float(value) for value in values)
    if not logits:
        raise ValueError("planner top-p filter requires non-empty logits")
    if top_p is None or not (0.0 < top_p < 1.0):
        return (True,) * len(logits)
    order = stable_descending_token_order(logits)
    finite_values = [logits[token] for token in order if math.isfinite(logits[token])]
    if not finite_values:
        raise ValueError("planner top-p filter has no finite candidate")
    maximum = finite_values[0]
    weights = [
        math.exp(logits[token] - maximum) if math.isfinite(logits[token]) else 0.0
        for token in order
    ]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("planner top-p probability sum is invalid")
    remove_sorted: list[bool] = []
    cumulative = 0.0
    for weight in weights:
        cumulative += weight / total
        remove_sorted.append(cumulative > top_p)
    remove_sorted = [False] + remove_sorted[:-1]
    keep = [False] * len(logits)
    for token, remove in zip(order, remove_sorted, strict=True):
        keep[token] = not remove
    return tuple(keep)


REFERENCE_ENVIRONMENT = {
    "ACESTEP_LM_BACKEND": "pt",
    "ACESTEP_DISABLE_TQDM": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_BACKEND": "eager",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def install_reference_environment() -> None:
    """Set process controls that must be present before importing upstream ACE."""

    for key, value in REFERENCE_ENVIRONMENT.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"{key}={existing!r} conflicts with reference value {value!r}")
        os.environ[key] = value


def assert_vae_float32(vae: object) -> None:
    """Require the declared VAE dtype and every floating tensor to be FP32."""

    import torch

    if getattr(vae, "dtype", None) != torch.float32:
        raise ValueError("reference VAE dtype must be torch.float32")
    parameters = getattr(vae, "parameters", None)
    buffers = getattr(vae, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise ValueError("reference VAE does not expose parameters and buffers")
    if any(parameter.dtype != torch.float32 for parameter in parameters()):
        raise ValueError("reference VAE contains a non-float32 parameter")
    if any(
        buffer.dtype.is_floating_point and buffer.dtype != torch.float32
        for buffer in buffers()
    ):
        raise ValueError("reference VAE contains a non-float32 floating buffer")


def _command_output(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot query accelerator driver with {arguments[0]}: {error}") from error
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"accelerator driver query {arguments[0]} returned no value")
    return value


def _accelerator_receipt(torch: Any, device: str) -> dict[str, object]:
    if device == "cuda":
        index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        runtime = getattr(torch.version, "cuda", None)
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("Torch did not expose the CUDA runtime version")
        driver = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                f"--id={index}",
            ]
        ).splitlines()[0].strip()
        major, minor = torch.cuda.get_device_capability(index)
        return {
            "backend": "cuda",
            "deviceIndex": index,
            "name": str(properties.name),
            "capability": f"{int(major)}.{int(minor)}",
            "runtimeVersion": runtime,
            "driverVersion": driver,
            "totalMemoryBytes": int(properties.total_memory),
        }
    if device == "xpu":
        index = int(torch.xpu.current_device())
        properties = torch.xpu.get_device_properties(index)
        runtime = getattr(torch.version, "xpu", None) or getattr(
            torch.version, "oneapi", None
        )
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("Torch did not expose the XPU runtime version")
        capability_getter = getattr(torch.xpu, "get_device_capability", None)
        capability_value = (
            capability_getter(index) if callable(capability_getter) else properties
        )
        capability = (
            json.dumps(capability_value, sort_keys=True, separators=(",", ":"))
            if isinstance(capability_value, dict)
            else str(capability_value)
        )
        driver = getattr(properties, "driver_version", None)
        if not isinstance(driver, str) or not driver:
            discovery = json.loads(_command_output(["xpu-smi", "discovery", "-j"]))

            def find_driver(value: object) -> str | None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if "driver" in str(key).lower() and isinstance(item, (str, int)):
                            return str(item)
                        found = find_driver(item)
                        if found is not None:
                            return found
                elif isinstance(value, list):
                    for item in value:
                        found = find_driver(item)
                        if found is not None:
                            return found
                return None

            driver = find_driver(discovery)
        if not isinstance(driver, str) or not driver:
            raise ValueError("XPU driver version could not be authenticated")
        name = getattr(properties, "name", None) or torch.xpu.get_device_name(index)
        total_memory = getattr(properties, "total_memory", None)
        if not isinstance(total_memory, int) or total_memory <= 0:
            raise ValueError("XPU total memory could not be authenticated")
        return {
            "backend": "xpu",
            "deviceIndex": index,
            "name": str(name),
            "capability": capability,
            "runtimeVersion": runtime,
            "driverVersion": driver,
            "totalMemoryBytes": total_memory,
        }
    raise ValueError(f"unsupported reference accelerator {device!r}")


def configure_eager_pytorch(handler: object, planner_handler: object | None) -> dict[str, object]:
    """Force deterministic eager attention and reject MLX/compiled model paths."""

    import torch

    if getattr(handler, "compiled", False):
        raise ValueError("compiled ACE model is not a reference-safe path")
    if getattr(handler, "use_mlx_dit", False) or getattr(handler, "use_mlx_vae", False):
        raise ValueError("MLX DiT/VAE is not the pinned PyTorch reference path")
    if planner_handler is not None and getattr(planner_handler, "llm_backend", None) != "pt":
        raise ValueError("planner capture requires the PyTorch LM backend")

    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    roots = [
        getattr(handler, "model", None),
        getattr(handler, "text_encoder", None),
        getattr(planner_handler, "llm", None) if planner_handler is not None else None,
    ]
    configured = 0
    for root in roots:
        if root is None:
            continue
        modules = root.modules() if hasattr(root, "modules") else (root,)
        for module in modules:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
                configured += 1
    for root in roots:
        if root is None:
            continue
        modules = root.modules() if hasattr(root, "modules") else (root,)
        for module in modules:
            config = getattr(module, "config", None)
            implementation = getattr(config, "_attn_implementation", "eager")
            if implementation != "eager":
                raise ValueError(f"failed to force eager attention on {type(module).__name__}")
    if configured == 0:
        raise ValueError("no upstream attention configurations were found")

    try:
        from acestep.models.common.dcw_correction import DCWCorrector

        corrector = DCWCorrector(
            enabled=True,
            mode="double",
            scaler=0.05,
            high_scaler=0.02,
            wavelet="haar",
        )
        if not corrector.is_active:
            raise ValueError("DCWCorrector reported an inactive reference path")
        import pywt
        import pytorch_wavelets
        probe_x = torch.arange(32, device=handler.device, dtype=torch.float32).reshape(
            1, 8, 4
        ).to(handler.dtype)
        probe_y = torch.zeros_like(probe_x)
        probe_output = corrector.apply(probe_x, probe_y, 0.5)
        if torch.equal(probe_output, probe_x):
            raise ValueError("Haar DCW probe was a no-op; wavelet backend is not active")
    except Exception as error:
        raise ValueError(f"active pytorch-wavelets Haar DCW is required: {error}") from error

    device = str(getattr(handler, "device", "unknown"))
    vae = getattr(handler, "vae", None)
    if vae is None:
        raise ValueError("reference handler has no VAE")
    assert_vae_float32(vae)
    return {
        "backend": "pytorch-eager",
        "attentionImplementation": "eager",
        "dcwBackend": "pytorch-wavelets-haar",
        "vaeComputeDtype": "float32",
        "vaeChunkSize": 256,
        "vaeOverlap": 64,
        "vaeOffloadWavToCpu": True,
        "device": device,
        "accelerator": _accelerator_receipt(torch, device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": str(torch.__version__),
            "pytorchWavelets": importlib.metadata.version("pytorch-wavelets"),
            "pyWavelets": importlib.metadata.version("PyWavelets"),
        },
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False,
    }


def _torch_dtype_name(torch: Any, tensor: Any) -> str:
    mapping = {
        torch.bool: "bool",
        torch.uint32: "uint32",
        torch.int64: "int64",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported capture tensor dtype {tensor.dtype}") from error


def _tensor_bytes(torch: Any, tensor: Any) -> bytes:
    if sys.byteorder != "little":
        raise ValueError("native tensor capture currently requires a little-endian host")
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy().tobytes(order="C")
    if cpu.dtype == torch.bool:
        return cpu.view(torch.uint8).numpy().tobytes(order="C")
    return cpu.numpy().tobytes(order="C")


def _stats_bounded(torch: Any, tensor: Any) -> dict[str, int | float]:
    flattened = tensor.detach().reshape(-1)
    count = int(flattened.numel())
    finite_count = 0
    nan_count = 0
    inf_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    total_square = 0.0
    total_abs = 0.0
    chunk_elements = 1_048_576
    for start in range(0, count, chunk_elements):
        chunk = flattened[start : start + chunk_elements].to(
            device="cpu", dtype=torch.float64
        )
        finite = torch.isfinite(chunk)
        finite_values = chunk[finite]
        finite_here = int(finite_values.numel())
        finite_count += finite_here
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
        if finite_here:
            minimum = min(minimum, float(finite_values.min().item()))
            maximum = max(maximum, float(finite_values.max().item()))
            total += float(finite_values.sum().item())
            total_square += float((finite_values * finite_values).sum().item())
            total_abs += float(finite_values.abs().sum().item())
    if finite_count != count:
        raise ValueError(
            f"reference tensor is not finite: nan={nan_count}, inf={inf_count}, count={count}"
        )
    denominator = max(finite_count, 1)
    return {
        "count": count,
        "finiteCount": finite_count,
        "nanCount": nan_count,
        "infCount": inf_count,
        "min": minimum,
        "max": maximum,
        "meanF64": total / denominator,
        "rmsF64": math.sqrt(total_square / denominator),
        "l1F64": total_abs,
    }


def _sequence_axis(shape_contract: tuple[object, ...]) -> int:
    sequence_symbols = {"T", "P", "LT", "LL", "E", "C5", "S", "variable"}
    for index, value in enumerate(shape_contract):
        if value in sequence_symbols:
            return index
    raise ValueError(f"cannot resolve a sequence axis from tap shape {shape_contract}")


def _slice_tensor(torch: Any, tensor: Any, shape_contract: tuple[object, ...]) -> tuple[Any, dict[str, object]]:
    axis = _sequence_axis(shape_contract)
    length = int(tensor.shape[axis])
    indices = set(range(min(4, length)))
    indices.update(range(max(0, length - 4), length))
    if length:
        indices.update((length // 4, length // 2, (3 * length) // 4))
    selected = sorted(index for index in indices if 0 <= index < length)
    index_tensor = torch.tensor(selected, device=tensor.device, dtype=torch.int64)
    sliced = torch.index_select(tensor, axis, index_tensor)
    channel_axis: int | None = None
    channel_count: int | None = None
    if tensor.ndim >= 2:
        candidate = tensor.ndim - 1
        if candidate != axis and int(tensor.shape[candidate]) > 64:
            channel_axis = candidate
            channel_count = 64
            sliced = sliced.narrow(candidate, 0, channel_count)
    selection: dict[str, object] = {
        "sequenceAxis": axis,
        "sequenceIndices": selected,
        "channelAxis": channel_axis,
        "channelStart": 0 if channel_axis is not None else None,
        "channelEnd": channel_count,
    }
    return sliced.contiguous(), selection


class TorchTapCapture:
    """Convert upstream Torch tensors into the exact tap artifact contract."""

    def __init__(self, writer: CaptureWriter, taps: TapContract) -> None:
        self.writer = writer
        self.taps = taps

    def capture(self, tap_id: str, tensor: object) -> None:
        import torch

        if tap_id not in self.taps.specs:
            raise ValueError(f"unknown native tap {tap_id!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{tap_id}: upstream value is not a torch.Tensor")
        spec = self.taps.specs[tap_id]
        dtype = _torch_dtype_name(torch, tensor)
        if dtype != spec.dtype:
            raise ValueError(
                f"{tap_id}: upstream dtype {dtype} differs from tap contract {spec.dtype}"
            )
        logical_shape = list(tensor.shape)
        if spec.capture == "full":
            payload = _tensor_bytes(torch, tensor)
            self.writer.add_tensor(
                tap_id,
                payload,
                dtype=dtype,
                logical_shape=logical_shape,
                capture="full",
            )
            return
        stats = _stats_bounded(torch, tensor)
        sliced, selection = _slice_tensor(torch, tensor, spec.shape)
        self.writer.add_tensor(
            tap_id,
            _tensor_bytes(torch, sliced),
            dtype=dtype,
            logical_shape=logical_shape,
            stored_shape=list(sliced.shape),
            capture="slice+stats",
            selection=selection,
            stats=stats,
        )


class InitialNoiseInjector:
    """Replace ``prepare_noise`` with one externally supplied float32 tensor."""

    def __init__(self, inputs: InputBundle, capture: TorchTapCapture) -> None:
        self.inputs = inputs
        self.capture = capture
        self.calls = 0
        self._model: object | None = None
        self._original: object | None = None

    def install(self, model: object) -> None:
        if self._model is not None:
            raise RuntimeError("initial-noise injector is already installed")
        original = getattr(model, "prepare_noise", None)
        if not callable(original):
            raise ValueError("upstream model has no prepare_noise method")
        self._model = model
        self._original = original
        injector = self

        def injected_prepare_noise(instance: object, reference: object, seed: object) -> object:
            del instance, seed
            import torch

            injector.calls += 1
            if injector.calls != 1:
                raise ValueError("unexpected second upstream noise request")
            expected_shape = injector.inputs.initial_noise.shape
            if tuple(reference.shape) != expected_shape:
                raise ValueError(
                    f"upstream requested noise shape {tuple(reference.shape)}, "
                    f"fixture supplies {expected_shape}"
                )
            payload = bytearray(injector.inputs.initial_noise.path.read_bytes())
            noise_f32 = torch.frombuffer(payload, dtype=torch.float32).clone().reshape(expected_shape)
            injector.capture.capture("diffusion.initial-noise", noise_f32)
            return noise_f32.to(device=reference.device, dtype=reference.dtype)

        setattr(model, "prepare_noise", types.MethodType(injected_prepare_noise, model))

    def uninstall(self) -> None:
        if self._model is not None and self._original is not None:
            setattr(self._model, "prepare_noise", self._original)
        self._model = None
        self._original = None


@dataclass(slots=True)
class _WordCursor:
    values: tuple[int, ...]
    consumed: int = 0

    def take(self) -> int:
        if self.consumed >= len(self.values):
            raise ValueError("externally supplied planner word stream is exhausted")
        value = self.values[self.consumed]
        self.consumed += 1
        return value


class PlannerWordInjector:
    """Replace PyTorch multinomial with the browser's raw-word categorical map.

    One continuous word stream spans CoT and semantic generation. A word is
    consumed for every emitted token, including a forced one-candidate token
    and the final stop token. This deliberately does not assume that a Torch
    seed or ``torch.multinomial`` has browser-compatible semantics.
    """

    def __init__(
        self,
        inputs: InputBundle,
        capture: TorchTapCapture,
        *,
        semantic_codes: int,
    ) -> None:
        if inputs.planner is None:
            raise ValueError("planner injector requires planner input artifacts")
        self.inputs = inputs
        self.capture = capture
        self.semantic_codes = semantic_codes
        self.cursor = _WordCursor(inputs.planner.words.read_values())
        self._oracle = load_prng_oracle()
        self.phase: str | None = None
        self.phase_draws = {"cot": 0, "codes": 0}
        self._handler: object | None = None
        self._original_sample: object | None = None
        self._original_top_p: object | None = None
        self._original_generate: object | None = None
        self._semantic_logits: dict[int, object] = {}
        self._audio_code_token_ids: tuple[int, ...] = ()
        self._token_id_to_code: dict[int, int] = {}
        self._audio_code_index: object | None = None
        self._selected_code_values: list[int] = []
        self._parsed_code_ids_captured = False

    def install(self, planner_handler: object) -> None:
        if getattr(planner_handler, "llm_backend", None) != "pt":
            raise ValueError("planner word injection supports only the eager PyTorch backend")
        original_sample = getattr(planner_handler, "_sample_tokens", None)
        original_top_p = getattr(planner_handler, "_apply_top_p_filter", None)
        original_generate = getattr(planner_handler, "generate_from_formatted_prompt", None)
        if (
            not callable(original_sample)
            or not callable(original_top_p)
            or not callable(original_generate)
        ):
            raise ValueError("upstream planner sampling methods are unavailable")
        tokenizer = getattr(planner_handler, "llm_tokenizer", None)
        if tokenizer is None or not callable(
            getattr(tokenizer, "convert_tokens_to_ids", None)
        ):
            raise ValueError("upstream planner tokenizer cannot resolve audio-code IDs")
        code_tokens = [
            f"<|audio_code_{code}|>" for code in range(self.semantic_vocabulary_size)
        ]
        resolved = tokenizer.convert_tokens_to_ids(code_tokens)
        if (
            not isinstance(resolved, list)
            or len(resolved) != self.semantic_vocabulary_size
            or any(not isinstance(token_id, int) for token_id in resolved)
            or len(set(resolved)) != self.semantic_vocabulary_size
        ):
            raise ValueError("planner tokenizer does not provide 64,000 unique audio-code IDs")
        roundtrip = tokenizer.convert_ids_to_tokens(resolved)
        if roundtrip != code_tokens:
            raise ValueError("planner audio-code token/value mapping failed round-trip")
        self._audio_code_token_ids = tuple(resolved)
        self._token_id_to_code = {
            token_id: code for code, token_id in enumerate(self._audio_code_token_ids)
        }

        import torch

        self._audio_code_index = torch.tensor(
            self._audio_code_token_ids,
            dtype=torch.int64,
            device=getattr(planner_handler, "device", "cpu"),
        )
        self._handler = planner_handler
        self._original_sample = original_sample
        self._original_top_p = original_top_p
        self._original_generate = original_generate
        injector = self

        def phase_generate(instance: object, formatted_prompt: str, cfg: object = None, *args: object, **kwargs: object) -> object:
            del instance
            config = cfg if isinstance(cfg, dict) else {}
            phase = str(config.get("generation_phase", "cot"))
            if phase not in injector.phase_draws:
                raise ValueError(f"unknown planner sampling phase {phase!r}")
            if phase == "codes" and not injector.capture.writer.has_tensor("planner.prompt.ids"):
                import torch

                tokenized = planner_handler.llm_tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                )
                injector.capture.capture(
                    "planner.prompt.ids", tokenized["input_ids"].to(torch.int64)
                )
            previous = injector.phase
            injector.phase = phase
            try:
                return original_generate(formatted_prompt, cfg, *args, **kwargs)
            finally:
                injector.phase = previous

        def sample_tokens(instance: object, logits: object, temperature: float) -> object:
            del instance
            return injector._sample(logits, temperature)

        def apply_top_p(instance: object, logits: object, top_p: float | None) -> object:
            del instance
            return injector._apply_stable_top_p(logits, top_p)

        setattr(
            planner_handler,
            "generate_from_formatted_prompt",
            types.MethodType(phase_generate, planner_handler),
        )
        setattr(planner_handler, "_sample_tokens", types.MethodType(sample_tokens, planner_handler))
        setattr(
            planner_handler,
            "_apply_top_p_filter",
            types.MethodType(apply_top_p, planner_handler),
        )

    def _apply_stable_top_p(self, logits: object, top_p: float | None) -> object:
        import torch

        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("planner top-p filter expected rank-2 Torch logits")
        if top_p is None or not (0.0 < top_p < 1.0):
            return logits
        if torch.isnan(logits).any():
            raise ValueError("planner logits contain NaN")
        # Stable descending sort preserves the original ascending token-ID
        # order for ties. This is the committed browser ordering contract.
        sorted_logits, sorted_indices = torch.sort(
            logits,
            dim=-1,
            descending=True,
            stable=True,
        )
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits.float(), dim=-1),
            dim=-1,
        )
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        logits[remove] = float("-inf")
        return logits

    def _sample(self, logits: object, temperature: float) -> object:
        import torch

        if self.phase is None:
            raise ValueError("planner sampled outside an identified CoT/codes phase")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("planner sampler expected rank-2 Torch logits")
        if not isinstance(temperature, (float, int)) or temperature <= 0:
            raise ValueError("pinned planner capture requires positive sampling temperature")
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        results: list[object] = []
        for row in range(int(probabilities.shape[0])):
            row_probs = probabilities[row]
            phase_index = self.phase_draws[self.phase]
            word = self.cursor.take()
            weights = row_probs.detach().to(device="cpu", dtype=torch.float32).tolist()
            selected_id = self._oracle.categorical_token_from_word(weights, word)
            selected = torch.tensor(selected_id, device=logits.device, dtype=torch.int64)
            results.append(selected)
            self.phase_draws[self.phase] += 1
            semantic_index = phase_index if self.phase == "codes" else None
            if semantic_index is not None and semantic_index in {
                0,
                self.semantic_codes // 2,
                self.semantic_codes - 1,
            }:
                if self._audio_code_index is None:
                    raise RuntimeError("planner audio-code index was not installed")
                index = self._audio_code_index
                if getattr(index, "device", None) != logits.device:
                    index = index.to(logits.device)
                    self._audio_code_index = index
                self._semantic_logits[semantic_index] = (
                    torch.index_select(logits[row].float(), 0, index)
                    .detach()
                    .cpu()
                )
            if semantic_index is not None:
                if semantic_index < self.semantic_codes:
                    code_value = self._token_id_to_code.get(selected_id)
                    if code_value is None:
                        raise ValueError(
                            "planner emitted a non-audio token before the expected "
                            "semantic-code count"
                        )
                    self._selected_code_values.append(code_value)
                elif semantic_index == self.semantic_codes:
                    eos = getattr(self._handler.llm_tokenizer, "eos_token_id", None)
                    if selected_id != eos:
                        raise ValueError("planner did not emit EOS after the expected codes")
                else:
                    raise ValueError("planner emitted tokens after its semantic stop token")
        return torch.stack(results).to(device=logits.device, dtype=torch.int64)

    @property
    def semantic_vocabulary_size(self) -> int:
        return 64_000

    def capture_code_ids(self, code_ids: object) -> None:
        import torch

        if not isinstance(code_ids, torch.Tensor):
            raise ValueError("planner code ids must be a torch tensor")
        if code_ids.ndim == 2:
            code_ids = code_ids.unsqueeze(-1)
        code_ids_i64 = code_ids.to(torch.int64)
        parsed_values = [int(value) for value in code_ids_i64.reshape(-1).tolist()]
        if parsed_values != self._selected_code_values:
            raise ValueError(
                "parsed planner semantic codes differ from sampled audio-code tokens"
            )
        self.capture.capture("planner.code.ids", code_ids_i64)
        self._parsed_code_ids_captured = True

    def finalize(self) -> dict[str, object]:
        import torch

        if not self._parsed_code_ids_captured:
            raise ValueError("planner parsed code IDs were not captured")
        if self.phase_draws["codes"] != self.semantic_codes + 1:
            raise ValueError(
                f"planner consumed {self.phase_draws['codes']} codes-phase words, "
                f"expected {self.semantic_codes + 1} including EOS"
            )
        expected_probes = {0, self.semantic_codes // 2, self.semantic_codes - 1}
        if set(self._semantic_logits) != expected_probes:
            raise ValueError("planner did not produce all declared semantic logits probes")
        consumed_words = self.cursor.values[: self.cursor.consumed]
        word_payload = struct.pack(
            f"<{len(consumed_words)}I",
            *consumed_words,
        )
        self.capture.writer.add_tensor(
            "planner.sample.words",
            word_payload,
            dtype="uint32",
            logical_shape=[len(consumed_words)],
            capture="full",
        )
        probe_tensor = torch.stack(
            [self._semantic_logits[index] for index in sorted(expected_probes)]
        ).to(torch.float32)
        self.capture.capture("planner.logits.probes", probe_tensor)
        return {
            "plannerWordsInjected": True,
            "plannerMapping": self.inputs.planner.mapping,
            "plannerWordsInputSha256": self.inputs.planner.words.sha256,
            "plannerWordsConsumed": self.cursor.consumed,
            "plannerWordCapacity": len(self.cursor.values),
            "phaseDraws": dict(self.phase_draws),
        }

    def uninstall(self) -> None:
        if self._handler is not None:
            if self._original_sample is not None:
                setattr(self._handler, "_sample_tokens", self._original_sample)
            if self._original_top_p is not None:
                setattr(self._handler, "_apply_top_p_filter", self._original_top_p)
            if self._original_generate is not None:
                setattr(
                    self._handler,
                    "generate_from_formatted_prompt",
                    self._original_generate,
                )
        self._handler = None
        self._original_sample = None
        self._original_top_p = None
        self._original_generate = None
        self._audio_code_index = None
