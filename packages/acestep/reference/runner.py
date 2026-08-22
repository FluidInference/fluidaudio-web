"""End-to-end execution inside the pinned upstream Python 3.12 environment."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .artifacts import (
    CaptureReplayContract,
    CaptureWriter,
    write_environment_receipt,
)
from .instrumentation import UpstreamInstrumentation
from .native import (
    InitialNoiseInjector,
    PlannerWordInjector,
    TorchTapCapture,
    assert_vae_float32,
    configure_eager_pytorch,
    install_reference_environment,
)
from .preflight import PreflightResult, authenticated_source_artifacts


def _link(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"authenticated checkpoint payload is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=False)


def force_reference_vae_float32(handler: object) -> None:
    """Cast the initialized Diffusers VAE and authenticate its actual tensors."""

    import torch

    vae = getattr(handler, "vae", None)
    if vae is None or not callable(getattr(vae, "to", None)):
        raise ValueError("upstream handler has no castable VAE")
    vae.to(dtype=torch.float32)
    assert_vae_float32(vae)


@contextmanager
def authenticated_checkpoint_view(
    *, model_cache: Path, scratch_root: Path
) -> Iterator[Path]:
    """Expose authenticated cache files in ACE's expected directory structure."""

    scratch_root.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix=".reference-runtime.", dir=scratch_root))
    checkpoints = runtime / "checkpoints"
    checkpoints.mkdir()
    from model.source_contract import ARTIFACT_BY_KEY

    resolved = dict(authenticated_source_artifacts(model_cache))
    runtime_roots = {
        "ACE-Step/Ace-Step1.5": checkpoints,
        "ACE-Step/acestep-5Hz-lm-0.6B": checkpoints / "acestep-5Hz-lm-0.6B",
    }
    consumed_keys = {
        key
        for key, artifact in ARTIFACT_BY_KEY.items()
        if artifact.component not in {"licenses"}
    }
    for key in sorted(consumed_keys):
        artifact = ARTIFACT_BY_KEY[key]
        destination_root = runtime_roots.get(artifact.repository)
        if destination_root is None:
            raise ValueError(f"no authenticated runtime mapping for {artifact.repository}")
        _link(resolved[key], destination_root / artifact.path)
    try:
        yield runtime
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def _install_source(source_root: Path) -> None:
    source_text = str(source_root.resolve())
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    import acestep

    actual = Path(acestep.__file__).resolve()
    if source_root.resolve() not in actual.parents:
        raise ValueError(f"imported ACE package from {actual}, not {source_root}")


def _initialize_upstream(
    runtime_root: Path,
    *,
    source_root: Path,
    planner_enabled: bool,
    device: str,
) -> tuple[object, object | None, dict[str, object]]:
    if device not in {"cuda", "xpu"}:
        raise ValueError(
            "accepted BF16 reference captures require --device cuda or xpu; "
            "CPU/MPS execute the pinned upstream model in float32 and do not match taps.json"
        )
    install_reference_environment()
    _install_source(source_root)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(runtime_root / "checkpoints")

    import torch
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()

    def authenticated_models_present(instance: object, **kwargs: object) -> None:
        del instance, kwargs
        # The ordinary helper insists that the unrelated default 1.7B LM is
        # installed. Preflight already authenticated every component this run uses.
        return None

    handler._ensure_models_present = types.MethodType(authenticated_models_present, handler)
    status, success = handler.initialize_service(
        project_root=str(runtime_root),
        config_path="acestep-v15-turbo",
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        prefer_source=None,
        use_mlx_dit=False,
        vae_checkpoint="official",
    )
    if not success:
        raise ValueError(f"upstream DiT initialization failed: {status}")
    if str(handler.device) != device:
        raise ValueError(f"upstream fell back from {device} to {handler.device}")
    if handler.dtype != torch.bfloat16:
        raise ValueError(f"upstream DiT dtype is {handler.dtype}, expected torch.bfloat16")
    force_reference_vae_float32(handler)

    planner_handler: object | None = None
    if planner_enabled:
        from acestep.llm_inference import LLMHandler

        planner_handler = LLMHandler()
        status, success = planner_handler.initialize(
            checkpoint_dir=str(runtime_root / "checkpoints"),
            lm_model_path="acestep-5Hz-lm-0.6B",
            backend="pt",
            device=device,
            offload_to_cpu=False,
            dtype=torch.bfloat16,
        )
        if not success:
            raise ValueError(f"upstream planner initialization failed: {status}")

    environment = configure_eager_pytorch(handler, planner_handler)
    for distribution in ("transformers", "diffusers", "numpy", "einops"):
        try:
            environment["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f"upstream dependency is missing: {distribution}")
    return handler, planner_handler, environment


def _generation_request(preflight: PreflightResult) -> tuple[object, object]:
    from acestep.inference import GenerationConfig, GenerationParams

    contract = preflight.fixture.contract
    request = contract["request"]
    planner = contract["planner"]
    diffusion = contract["diffusion"]
    postprocess = contract["postprocess"]
    dcw = diffusion["dcw"]
    seed = int(contract["random"]["userSeed"], 16)
    params = GenerationParams(
        task_type=request["taskType"],
        instruction=request["instruction"],
        caption=request["caption"],
        lyrics=request["lyrics"],
        instrumental=request["instrumental"],
        vocal_language=request["vocalLanguage"],
        bpm=request["bpm"],
        keyscale=request["keyScale"],
        timesignature=request["timeSignature"],
        duration=request["durationSeconds"],
        enable_normalization=postprocess["normalize"],
        normalization_db=postprocess["normalizationDb"],
        fade_in_duration=postprocess["fadeInSeconds"],
        fade_out_duration=postprocess["fadeOutSeconds"],
        latent_shift=diffusion["latentShift"],
        latent_rescale=diffusion["latentRescale"],
        inference_steps=diffusion["inferenceSteps"],
        seed=seed,
        guidance_scale=diffusion["guidanceScale"],
        use_adg=diffusion["useAdg"],
        cfg_interval_start=diffusion["cfgIntervalStart"],
        cfg_interval_end=diffusion["cfgIntervalEnd"],
        shift=diffusion["shift"],
        infer_method=diffusion["inferMethod"],
        sampler_mode=diffusion["samplerMode"],
        velocity_norm_threshold=diffusion["velocityNormThreshold"],
        velocity_ema_factor=diffusion["velocityEmaFactor"],
        dcw_enabled=dcw["enabled"],
        dcw_mode=dcw["mode"],
        dcw_scaler=dcw["lowScaler"],
        dcw_high_scaler=dcw["highScaler"],
        dcw_wavelet=dcw["wavelet"],
        timesteps=diffusion["customTimesteps"],
        reference_audio=request["referenceAudio"],
        src_audio=request["sourceAudio"],
        audio_codes=request["audioCodes"] or "",
        audio_cover_strength=request["audioCoverStrength"],
        cover_noise_strength=request["coverNoiseStrength"],
        chunk_mask_mode=request["chunkMaskMode"],
        repainting_start=request["repaintingStart"],
        repainting_end=(
            request["repaintingEnd"] if request["repaintingEnd"] is not None else -1
        ),
        retake_variance=request["retakeVariance"],
        retake_seed=request["retakeSeed"],
        thinking=planner["thinking"],
        lm_temperature=planner["temperature"],
        lm_cfg_scale=planner["cfgScale"],
        lm_top_k=planner["topK"],
        lm_top_p=planner["topP"],
        lm_negative_prompt=planner["negativePrompt"],
        use_cot_metas=planner["useCotMetas"],
        use_cot_caption=planner["useCotCaption"],
        use_cot_lyrics=planner["useCotLyrics"],
        use_cot_language=planner["useCotLanguage"],
        use_constrained_decoding=planner["useConstrainedDecoding"],
    )
    config = GenerationConfig(
        batch_size=request["batchSize"],
        allow_lm_batch=False,
        use_random_seed=False,
        seeds=[seed],
        lm_batch_chunk_size=1,
        constrained_decoding_debug=False,
        audio_format="wav32",
    )
    return params, config


def execute_capture(
    preflight: PreflightResult,
    *,
    source_root: Path,
    model_cache: Path,
    output_root: Path,
    capture_id: str,
    device: str,
) -> Path:
    """Execute one complete reference capture, atomically, or leave no capture."""

    if not preflight.capture_authorized or preflight.inputs is None:
        raise ValueError(f"capture is not authorized: {list(preflight.blockers)}")
    writer = CaptureWriter(
        output_root=output_root,
        capture_id=capture_id,
        fixture=preflight.fixture,
        taps=preflight.taps,
    )
    hooks: UpstreamInstrumentation | None = None
    noise: InitialNoiseInjector | None = None
    planner_random: PlannerWordInjector | None = None
    try:
        with authenticated_checkpoint_view(
            model_cache=model_cache, scratch_root=output_root
        ) as runtime_root:
            handler, planner_handler, environment = _initialize_upstream(
                runtime_root,
                source_root=source_root,
                planner_enabled=preflight.fixture.planner_enabled,
                device=device,
            )
            tap_capture = TorchTapCapture(writer, preflight.taps)
            hooks = UpstreamInstrumentation(
                handler,
                tap_capture,
                planner_enabled=preflight.fixture.planner_enabled,
                expected_semantic_codes=int(preflight.fixture.expected["semanticCodes"]),
                expected_dit_steps=int(
                    preflight.fixture.contract["diffusion"]["inferenceSteps"]
                ),
            )
            hooks.install()
            noise = InitialNoiseInjector(preflight.inputs, tap_capture)
            noise.install(handler.model)
            if planner_handler is not None:
                planner_random = PlannerWordInjector(
                    preflight.inputs,
                    tap_capture,
                    semantic_codes=int(preflight.fixture.expected["semanticCodes"]),
                )
                planner_random.install(planner_handler)

            params, config = _generation_request(preflight)
            from acestep.inference import generate_music

            result = generate_music(
                handler,
                planner_handler,
                params=params,
                config=config,
                save_dir=None,
            )
            if not result.success or len(result.audios) != 1:
                raise ValueError(
                    f"upstream generation failed or returned the wrong batch: {result.error}"
                )
            if noise.calls != 1:
                raise ValueError(f"upstream consumed initial noise {noise.calls} times")

            random_receipt: dict[str, object] = {
                "algorithmId": preflight.inputs.algorithm_id,
                "gaussianMapping": preflight.inputs.gaussian_mapping,
                "initialNoiseInjected": True,
                "initialNoiseSha256": preflight.inputs.initial_noise.sha256,
                "plannerWordsInjected": False,
            }
            if planner_random is not None:
                import torch

                code_string = result.audios[0]["params"].get("audio_codes", "")
                code_ids = handler._parse_audio_code_string(code_string)
                if len(code_ids) != preflight.fixture.expected["semanticCodes"]:
                    raise ValueError("upstream planner code count differs from fixture contract")
                planner_random.capture_code_ids(
                    torch.tensor(code_ids, dtype=torch.int64).reshape(1, -1, 1)
                )
                random_receipt.update(planner_random.finalize())

            import torch

            audio = result.audios[0]["tensor"].detach().cpu().float()
            if audio.ndim != 2 or audio.shape[0] != 2:
                raise ValueError(f"upstream final audio shape changed: {tuple(audio.shape)}")
            tap_capture.capture("audio.final-output", audio.unsqueeze(0))
            planar = audio.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
            writer.add_float32_stereo_output(
                planar,
                samples_per_channel=int(audio.shape[1]),
                sample_rate=int(result.audios[0]["sample_rate"]),
            )
            capture_root = writer.finalize(
                replay=CaptureReplayContract(
                    fixture=preflight.fixture,
                    taps=preflight.taps,
                    golden_manifest_id=preflight.golden_manifest_id,
                    reference_tool=preflight.reference_tool,
                    source=preflight.source,
                    browser_package=preflight.browser_package,
                    input_bundle=preflight.inputs.identity(),
                    environment=environment,
                ),
                random_injection=random_receipt,
                environment=environment,
            )
            write_environment_receipt(capture_root)
            return capture_root
    except BaseException:
        writer.abort()
        raise
    finally:
        if planner_random is not None:
            planner_random.uninstall()
        if noise is not None:
            noise.uninstall()
        if hooks is not None:
            hooks.uninstall()
