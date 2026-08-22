"""In-memory instrumentation of the pinned upstream ACE implementation."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable

from .native import TorchTapCapture, assert_vae_float32


def _first_tensor(value: object) -> object:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value
    except ImportError:
        pass
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if hasattr(value, "last_hidden_state"):
        return getattr(value, "last_hidden_state")
    raise ValueError(f"cannot extract a tensor from upstream output {type(value).__name__}")


def requires_nonchunked_vae_capture(capture: object) -> bool:
    """Return whether this fixture's expanded tap inventory requires a full decode."""

    writer = getattr(capture, "writer", None)
    fixture = getattr(writer, "fixture", None)
    taps = getattr(writer, "taps", None)
    if fixture is None or taps is None:
        raise ValueError("tap capture does not expose its fixture/tap contracts")
    return "vae.nonchunked.output" in taps.required_ids(fixture)


def pinned_tiled_decode_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Reject adaptive geometry and return the immutable reference controls."""

    forbidden = {"chunk_size", "overlap", "offload_wav_to_cpu"} & set(kwargs)
    if forbidden:
        raise ValueError(
            f"upstream supplied adaptive tiled-decode controls: {sorted(forbidden)}"
        )
    return {
        **kwargs,
        "chunk_size": 256,
        "overlap": 64,
        "offload_wav_to_cpu": True,
    }


@dataclass(slots=True)
class _MethodPatch:
    target: object
    name: str
    original: object


class UpstreamInstrumentation:
    """Install the complete ``golden/taps.json`` hook map without source edits."""

    def __init__(
        self,
        handler: object,
        capture: TorchTapCapture,
        *,
        planner_enabled: bool,
        expected_semantic_codes: int,
        expected_dit_steps: int,
    ) -> None:
        self.handler = handler
        self.capture = capture
        self.planner_enabled = planner_enabled
        self.expected_semantic_codes = expected_semantic_codes
        self.expected_dit_steps = expected_dit_steps
        self.scope: str | None = None
        self.decoder_step: int | None = None
        self.decoder_calls = 0
        self.dcw_calls = 0
        self._patches: list[_MethodPatch] = []
        self._hooks: list[object] = []
        self._dcw_class: object | None = None
        self._dcw_original: object | None = None
        self._installed = False

    def _patch_bound(self, target: object, name: str, function: Callable[..., object]) -> None:
        original = getattr(target, name, None)
        if not callable(original):
            raise ValueError(f"upstream target has no callable {name}: {target!r}")
        self._patches.append(_MethodPatch(target, name, original))
        setattr(target, name, types.MethodType(function, target))

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("upstream instrumentation is already installed")
        model = getattr(self.handler, "model", None)
        text_encoder = getattr(self.handler, "text_encoder", None)
        vae = getattr(self.handler, "vae", None)
        if model is None or text_encoder is None or vae is None:
            raise ValueError("ACE handler models must be initialized before instrumentation")

        instrumentation = self

        original_prepare_text = getattr(self.handler, "_prepare_text_conditioning_inputs")

        def prepare_text(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            result = original_prepare_text(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 7:
                raise ValueError("upstream text-conditioning return contract changed")
            instrumentation.capture.capture("token.caption.ids", result[1].to(dtype=_torch().int64))
            instrumentation.capture.capture("token.caption.mask", result[2].bool())
            instrumentation.capture.capture("token.lyric.ids", result[3].to(dtype=_torch().int64))
            instrumentation.capture.capture("token.lyric.mask", result[4].bool())
            return result

        self._patch_bound(self.handler, "_prepare_text_conditioning_inputs", prepare_text)

        original_text = getattr(self.handler, "infer_text_embeddings")

        def infer_text(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            previous = instrumentation.scope
            instrumentation.scope = "caption"
            try:
                result = original_text(*args, **kwargs)
                instrumentation.capture.capture("text.final.output", result)
                return result
            finally:
                instrumentation.scope = previous

        self._patch_bound(self.handler, "infer_text_embeddings", infer_text)

        original_lyric = getattr(self.handler, "infer_lyric_embeddings")

        def infer_lyric(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            previous = instrumentation.scope
            instrumentation.scope = "lyric-embedding"
            try:
                result = original_lyric(*args, **kwargs)
                instrumentation.capture.capture("lyric.embedding.input", result)
                return result
            finally:
                instrumentation.scope = previous

        self._patch_bound(self.handler, "infer_lyric_embeddings", infer_lyric)

        self._hooks.append(
            text_encoder.embed_tokens.register_forward_hook(self._caption_embedding_hook)
        )
        text_layers = getattr(text_encoder, "layers", None)
        if text_layers is None or len(text_layers) <= 13:
            raise ValueError("text encoder layer 13 is unavailable")
        self._hooks.append(text_layers[13].register_forward_hook(self._text_layer_hook))

        lyric_encoder = model.encoder.lyric_encoder
        if len(lyric_encoder.layers) < 1:
            raise ValueError("lyric encoder layer 0 is unavailable")
        self._hooks.append(
            lyric_encoder.layers[0].register_forward_hook(self._lyric_layer_hook)
        )
        self._hooks.append(lyric_encoder.norm.register_forward_hook(self._lyric_final_hook))
        self._hooks.append(
            model.encoder.timbre_encoder.register_forward_hook(self._timbre_hook)
        )
        self._hooks.append(model.encoder.register_forward_hook(self._condition_encoder_hook))

        original_prepare_condition = getattr(model, "prepare_condition")

        def prepare_condition(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            result = original_prepare_condition(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError("upstream prepare_condition return contract changed")
            instrumentation.capture.capture("condition.context", result[2])
            return result

        self._patch_bound(model, "prepare_condition", prepare_condition)

        if self.planner_enabled:
            original_decode_codes = getattr(self.handler, "_decode_audio_codes_to_latents")

            def decode_codes(instance: object, *args: object, **kwargs: object) -> object:
                del instance
                previous = instrumentation.scope
                instrumentation.scope = "semantic"
                try:
                    result = original_decode_codes(*args, **kwargs)
                    if result is None:
                        raise ValueError("planner code decode unexpectedly returned None")
                    instrumentation.capture.capture("semantic.hints.25hz", result)
                    return result
                finally:
                    instrumentation.scope = previous

            self._patch_bound(self.handler, "_decode_audio_codes_to_latents", decode_codes)
            quantizer = model.tokenizer.quantizer
            original_quantizer = getattr(quantizer, "get_output_from_indices")

            def quantizer_output(instance: object, *args: object, **kwargs: object) -> object:
                del instance
                result = original_quantizer(*args, **kwargs)
                if instrumentation.scope == "semantic":
                    instrumentation.capture.capture("semantic.fsq.output", result)
                return result

            self._patch_bound(quantizer, "get_output_from_indices", quantizer_output)
            self._hooks.append(
                model.detokenizer.layers[0].register_forward_hook(
                    self._semantic_detokenizer_layer_hook
                )
            )

        decoder = model.decoder
        if len(decoder.layers) <= 23:
            raise ValueError("expected 24 DiT layers at the pinned source")
        self._hooks.append(
            decoder.register_forward_pre_hook(self._decoder_pre_hook, with_kwargs=True)
        )
        self._hooks.append(decoder.register_forward_hook(self._decoder_output_hook))
        for layer_index in (0, 1, 22, 23):
            self._hooks.append(
                decoder.layers[layer_index].register_forward_hook(
                    self._dit_layer_hook(layer_index)
                )
            )

        original_generate_audio = getattr(model, "generate_audio")

        def generate_audio(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            instrumentation.decoder_calls = 0
            instrumentation.dcw_calls = 0
            instrumentation.decoder_step = None
            result = original_generate_audio(*args, **kwargs)
            if instrumentation.decoder_calls != instrumentation.expected_dit_steps:
                raise ValueError(
                    f"upstream performed {instrumentation.decoder_calls} DiT evaluations, "
                    f"expected {instrumentation.expected_dit_steps}"
                )
            if instrumentation.dcw_calls != instrumentation.expected_dit_steps:
                raise ValueError(
                    f"upstream performed {instrumentation.dcw_calls} DCW corrections, "
                    f"expected {instrumentation.expected_dit_steps}"
                )
            target = result.get("target_latents") if isinstance(result, dict) else None
            if target is None:
                raise ValueError("upstream generate_audio omitted target_latents")
            instrumentation.capture.capture("diffusion.final-latent", target)
            return result

        self._patch_bound(model, "generate_audio", generate_audio)

        from acestep.models.common.dcw_correction import DCWCorrector

        self._dcw_class = DCWCorrector
        self._dcw_original = DCWCorrector.apply
        original_dcw = self._dcw_original

        def dcw_apply(instance: object, x_next: object, denoised: object, t_curr: float) -> object:
            step = instrumentation.dcw_calls
            if step in (0, instrumentation.expected_dit_steps - 1):
                instrumentation.capture.capture(
                    f"dcw.step.{step}.denoised", denoised.float()
                )
            result = original_dcw(instance, x_next, denoised, t_curr)
            if step in (0, instrumentation.expected_dit_steps - 1):
                instrumentation.capture.capture(f"dcw.step.{step}.output", result)
            instrumentation.dcw_calls += 1
            return result

        DCWCorrector.apply = dcw_apply

        original_decode = getattr(self.handler, "_decode_generate_music_pred_latents")
        capture_nonchunked_vae = requires_nonchunked_vae_capture(self.capture)

        def decode_pred_latents(instance: object, pred_latents: object, *args: object, **kwargs: object) -> object:
            del instance
            assert_vae_float32(instrumentation.handler.vae)
            if capture_nonchunked_vae:
                import torch

                with torch.inference_mode():
                    with instrumentation.handler._load_model_context("vae"):
                        vae_latents = pred_latents.transpose(1, 2).contiguous().to(
                            instrumentation.handler.vae.dtype
                        )
                        nonchunked = instrumentation.handler.vae.decode(vae_latents).sample.float()
                        instrumentation.capture.capture("vae.nonchunked.output", nonchunked)
                        del vae_latents, nonchunked
            return original_decode(pred_latents, *args, **kwargs)

        self._patch_bound(
            self.handler, "_decode_generate_music_pred_latents", decode_pred_latents
        )
        original_tiled_decode = getattr(self.handler, "tiled_decode")

        def tiled_decode(instance: object, *args: object, **kwargs: object) -> object:
            del instance
            result = original_tiled_decode(*args, **pinned_tiled_decode_kwargs(kwargs))
            instrumentation.capture.capture("vae.chunked.raw-output", result.float())
            return result

        self._patch_bound(self.handler, "tiled_decode", tiled_decode)
        self._installed = True

    def _caption_embedding_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        if self.scope == "caption":
            self.capture.capture("text.embedding.input", _first_tensor(output))

    def _text_layer_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        if self.scope == "caption":
            self.capture.capture("text.layer.13.output", _first_tensor(output))

    def _lyric_layer_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        self.capture.capture("condition.lyric.layer.0", _first_tensor(output))

    def _lyric_final_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        self.capture.capture("condition.lyric.final", _first_tensor(output))

    def _timbre_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        self.capture.capture("condition.timbre.final", _first_tensor(output))

    def _condition_encoder_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("condition encoder return contract changed")
        self.capture.capture("condition.encoder.output", output[0])
        self.capture.capture("condition.encoder.mask", output[1].bool())

    def _semantic_detokenizer_layer_hook(
        self, module: object, inputs: object, output: object
    ) -> None:
        del module, inputs
        if self.scope != "semantic":
            return
        tensor = _first_tensor(output)
        if tensor.ndim != 3 or tensor.shape[0] != self.expected_semantic_codes:
            raise ValueError(
                "semantic detokenizer layer layout changed; expected "
                "[B*C5,poolWindow,2048] with B=1"
            )
        logical = tensor.reshape(1, tensor.shape[0] * tensor.shape[1], tensor.shape[2])
        self.capture.capture("semantic.detokenizer.layer.0", logical)

    def _decoder_pre_hook(
        self, module: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        del module
        step = self.decoder_calls
        self.decoder_step = step
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            raise ValueError("DiT decoder call omitted hidden_states")
        if step in (0, self.expected_dit_steps - 1):
            self.capture.capture(f"dit.step.{step}.input", hidden)

    def _decoder_output_hook(self, module: object, inputs: object, output: object) -> None:
        del module, inputs
        step = self.decoder_calls
        if step in (0, self.expected_dit_steps - 1):
            self.capture.capture(f"dit.step.{step}.velocity", _first_tensor(output))
        self.decoder_calls += 1
        self.decoder_step = None

    def _dit_layer_hook(self, layer_index: int) -> Callable[..., None]:
        def hook(module: object, inputs: object, output: object) -> None:
            del module, inputs
            step = self.decoder_step
            if step in (0, self.expected_dit_steps - 1):
                self.capture.capture(
                    f"dit.step.{step}.layer.{layer_index}.output", _first_tensor(output)
                )

        return hook

    def uninstall(self) -> None:
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        for patch in reversed(self._patches):
            setattr(patch.target, patch.name, patch.original)
        self._patches.clear()
        if self._dcw_class is not None and self._dcw_original is not None:
            self._dcw_class.apply = self._dcw_original
        self._dcw_class = None
        self._dcw_original = None
        self._installed = False


def _torch() -> Any:
    import torch

    return torch
