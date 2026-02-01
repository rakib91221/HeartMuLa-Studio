"""
HeartMula Pipeline - adapted from Wan2GP for standalone HeartMula-Studio-GP use.

This module uses mmgp for efficient memory management and model offloading,
allowing HeartMula to run on GPUs with limited VRAM.

Key changes from Wan2GP version:
- Removed dependency on shared.utils.files_locator
- Added standalone file path resolution for models
- Model paths are configured via environment variables or explicit paths

Optimizations added:
- Enhanced coTenantsMap with codec for better memory scheduling
- CPU offload decorator pattern (from ACE-Step) for manual offloading
- Support for pre-quantized INT8/INT4 models
- Async model loading for reduced overhead
"""
from __future__ import annotations

import os
import json
import random
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, TypeVar
import contextlib

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm


# CPU Offload utilities (inspired by ACE-Step)
class CpuOffloader:
    """Context manager for automatic CPU offloading of models."""

    def __init__(self, model, device="cuda", dtype=None):
        self.model = model
        self.target_device = device
        self.dtype = dtype or getattr(model, "dtype", torch.bfloat16)

    def __enter__(self):
        # Move model to target device
        if hasattr(self.model, "_mm_manager"):
            # mmgp managed - let it handle device
            pass
        else:
            self.model.to(self.target_device, dtype=self.dtype)
        return self.model

    def __exit__(self, *args):
        # Move back to CPU if not mmgp managed
        if not hasattr(self.model, "_mm_manager"):
            self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


T = TypeVar("T")


def cpu_offload(model_attr: str):
    """
    Decorator to automatically move model to GPU, execute, move back to CPU.

    This is useful when NOT using mmgp, or as a fallback.
    Inspired by ACE-Step's cpu_offload pattern.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if CPU offload is enabled
            if not getattr(self, "use_cpu_offload", False):
                return func(self, *args, **kwargs)

            device = getattr(self, "device", torch.device("cuda"))
            model = getattr(self, model_attr)

            with CpuOffloader(model, device):
                return func(self, *args, **kwargs)

        return wrapper
    return decorator

from .heartcodec.configuration_heartcodec import HeartCodecConfig
from .heartcodec.modeling_heartcodec import HeartCodec
from .heartmula.configuration_heartmula import HeartMuLaConfig
from .heartmula.modeling_heartmula import HeartMuLa


# Default model directory - can be overridden via HEARTMULA_MODEL_DIR env var
_default_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_DIR = os.environ.get("HEARTMULA_MODEL_DIR", _default_model_dir)


def _resolve_paths(
    pretrained_path: Path, version: str, heartmula_weights_path: Optional[str] = None
):
    """
    Resolve model file paths for standalone HeartMula-Studio-GP operation.

    Supports two directory structures:

    1. HeartMula-Studio-GP structure (legacy):
        HeartMula/
            tokenizer.json
            gen_config.json
            HeartMula_codec.safetensors
            codec_config.json
            model-*.safetensors (main model weights)

    2. HuggingFace download structure (HeartMuLa-Studio):
        models/
            tokenizer.json
            gen_config.json
            HeartCodec-oss/
                model.safetensors
                config.json
            HeartMuLa-oss-{version}/
                model-*.safetensors
    """
    config_path = Path(__file__).resolve().parent / "config" / f"heartmula_{version}.json"

    # Model directory - use pretrained_path or default
    model_dir = Path(pretrained_path) if pretrained_path else Path(DEFAULT_MODEL_DIR)

    # Try the legacy HeartMula subdirectory structure first
    heartmula_dir = model_dir / "HeartMula"

    # If HeartMula subdirectory doesn't exist, use HuggingFace download structure
    if not heartmula_dir.exists():
        # For HuggingFace structure, tokenizer.json is at root
        if (model_dir / "tokenizer.json").exists():
            heartmula_dir = model_dir  # Use model_dir as the base
        else:
            # Check if we're given the wrong path
            raise FileNotFoundError(f"Could not find tokenizer.json in {model_dir} or {heartmula_dir}")

    tokenizer_path = heartmula_dir / "tokenizer.json"
    gen_config_path = heartmula_dir / "gen_config.json"

    # For HuggingFace structure, codec is in HeartCodec-oss subdirectory
    codec_dir = model_dir / "HeartCodec-oss"
    if codec_dir.exists():
        # HeartMuLa-Studio HuggingFace download structure
        codec_path = codec_dir
    else:
        # Legacy structure: codec is in heartmula_dir
        codec_path = heartmula_dir

    weights_path = heartmula_weights_path

    return (
        weights_path,
        Path(config_path),
        codec_path,  # codec directory (HeartCodec-oss for HuggingFace, HeartMula for legacy)
        Path(tokenizer_path),
        Path(gen_config_path),
    )


def _resolve_codec_names(codec_version: Optional[str]) -> tuple[str, str]:
    if codec_version:
        suffix = f"_{codec_version}"
    else:
        suffix = ""
    return f"HeartMula_codec{suffix}.safetensors", f"codec_config{suffix}.json"


def _strip_heartmula_rope_cache(state_dict):
    remove_keys = (
        "backbone.layers.0.attn.pos_embeddings.theta",
        "backbone.layers.0.attn.pos_embeddings.cache",
        "decoder.layers.0.attn.pos_embeddings.theta",
        "decoder.layers.0.attn.pos_embeddings.cache",
    )
    for key in remove_keys:
        state_dict.pop(key, None)
    return state_dict


def _dequantize_int8_state_dict(state_dict, target_dtype=None):
    """
    Dequantize INT8 quantized weights back to floating point.

    The quantized format stores:
    - param_name: int8 tensor
    - param_name_scale: float16 scale

    Dequantization: weight_fp = weight_int8.float() * scale.float()
    """
    import torch

    if target_dtype is None:
        target_dtype = torch.bfloat16

    # Find all scale keys to identify quantized weights
    scale_keys = [k for k in state_dict.keys() if k.endswith("_scale")]

    if not scale_keys:
        # No quantized weights found, just strip rope cache
        return _strip_heartmula_rope_cache(state_dict)

    print(f"[mmgp] Dequantizing {len(scale_keys)} INT8 weights to {target_dtype}...", flush=True)

    dequantized_dict = {}
    for key, tensor in state_dict.items():
        if key.endswith("_scale"):
            # Skip scale tensors - they're used during dequantization
            continue

        scale_key = f"{key}_scale"
        if scale_key in state_dict:
            # This is a quantized weight - dequantize it
            scale = state_dict[scale_key]
            dequantized = tensor.float() * scale.float()
            dequantized_dict[key] = dequantized.to(target_dtype)
        else:
            # Regular tensor, keep as-is
            dequantized_dict[key] = tensor

    # Strip rope cache from dequantized dict
    return _strip_heartmula_rope_cache(dequantized_dict)


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = fp.read()
        import json

        return cls(**json.loads(data))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HeartMuLaPipeline:
    def __init__(
        self,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
        version: str = "3B",
        heartmula_dtype: Optional[torch.dtype] = None,
        heartcodec_dtype: Optional[torch.dtype] = None,
        heartmula_weights_path: Optional[str] = None,
        cfg_scale: float = 1.5,
        topk: int = 50,
        max_audio_length_ms: int = 120000,
        codec_steps: int = 10,
        codec_guidance_scale: float = 1.25,
        codec_version: str = "",
        VAE_dtype = torch.float32,
    ):
        self.device = torch.device("cpu")
        self.mula_device = self.device
        self.codec_device = self.device
        self.mula_dtype = None
        self.codec_dtype = None

        self.cfg_scale = cfg_scale
        self.topk = topk
        self.max_audio_length_ms = max_audio_length_ms
        self.codec_steps = codec_steps
        self.codec_guidance_scale = codec_guidance_scale
        self.codec_version = codec_version
        self.heartmula_weights_path = heartmula_weights_path
        self.VAE_dtype = VAE_dtype

        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(DEFAULT_MODEL_DIR)
        self.version = version
        self._interrupt = False
        self._early_stop = False

        self._parallel_number = 8 + 1
        self._muq_dim = 512

        self._load_models()

    def _load_models(self):
        (
            mula_weights_path,
            mula_config_path,
            codec_path,
            tokenizer_path,
            gen_config_path,
        ) = _resolve_paths(
            self.ckpt_root,
            self.version,
            heartmula_weights_path=self.heartmula_weights_path,
        )
        from accelerate import init_empty_weights
        from mmgp import offload

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.gen_config = HeartMuLaGenConfig.from_file(str(gen_config_path))
        with open(mula_config_path, encoding="utf-8") as fp:
            mula_config = HeartMuLaConfig(**json.load(fp))

        with init_empty_weights():
            self.mula = HeartMuLa(mula_config)

        # Check if loading INT8 quantized weights (need dequantization)
        is_quantized = "_int8" in str(mula_weights_path)
        if is_quantized:
            print(f"[mmgp] Loading INT8 quantized weights (will dequantize to bfloat16)...", flush=True)
            preprocess_fn = _dequantize_int8_state_dict
        else:
            preprocess_fn = _strip_heartmula_rope_cache

        offload.load_model_data(
            self.mula,
            str(mula_weights_path),
            default_dtype=None,
            writable_tensors=False,
            preprocess_sd=preprocess_fn,
        )

        decoder = self.mula.decoder
        delattr(self.mula, "decoder")
        self.mula.decoder = [decoder]

        if hasattr(self.mula, "_interrupt_check"):
            self.mula._interrupt_check = self._abort_requested
        self.model = self.mula
        self.mula.eval()

        # Disable compilation for these layers (not supported with mmgp)
        self.mula.decoder[0].layers._compile_me = False
        self.mula.backbone.layers._compile_me = False

        first_param = next(self.mula.parameters(), None)
        if first_param is not None:
            self.mula_dtype = first_param.dtype
        codec_weights_name, codec_config_name = _resolve_codec_names(self.codec_version)

        # Check if using pre-quantized transformer - if so, use quantized codec too
        quantized_codec_path = None
        if self.heartmula_weights_path and "quantized" in str(self.heartmula_weights_path):
            # Look for quantized codec in same directory
            quantized_dir = Path(self.heartmula_weights_path).parent
            quantized_codec = quantized_dir / "HeartMula_codec_int8.safetensors"
            if quantized_codec.is_file():
                quantized_codec_path = quantized_codec
                print(f"[mmgp] Using pre-quantized codec: {quantized_codec_path}", flush=True)

        # Look for codec weights in the codec_path directory
        # Try quantized first, then legacy naming, then HuggingFace naming
        if quantized_codec_path:
            codec_weights_path = quantized_codec_path
        else:
            codec_weights_path = Path(codec_path) / codec_weights_name
            if not codec_weights_path.is_file():
                # Try HuggingFace naming: model.safetensors
                codec_weights_path = Path(codec_path) / "model.safetensors"
        if not codec_weights_path.is_file():
            raise FileNotFoundError(
                f"Expected HeartCodec weights at {codec_path}/{codec_weights_name} or model.safetensors but not found."
            )

        # Try legacy config naming first, then HuggingFace naming
        codec_config_path = Path(codec_path) / codec_config_name
        if not codec_config_path.is_file():
            # Try HuggingFace naming: config.json
            codec_config_path = Path(codec_path) / "config.json"
        if not codec_config_path.is_file():
            raise FileNotFoundError(
                f"Expected HeartCodec config at {codec_path}/{codec_config_name} or config.json but not found."
            )

        with open(codec_config_path, encoding="utf-8") as fp:
            codec_config = HeartCodecConfig(**json.load(fp))
        with init_empty_weights():
            self.codec = HeartCodec(codec_config)
        self.codec._offload_hooks = ["detokenize"]

        self.codec._model_dtype = self.VAE_dtype

        # Check if loading INT8 quantized codec (need dequantization)
        is_codec_quantized = "_int8" in str(codec_weights_path)
        if is_codec_quantized:
            print(f"[mmgp] Loading INT8 quantized codec (will dequantize to {self.VAE_dtype})...", flush=True)
            codec_preprocess_fn = lambda sd: _dequantize_int8_state_dict(sd, target_dtype=self.VAE_dtype)
        else:
            codec_preprocess_fn = None

        offload.load_model_data(
            self.codec,
            str(codec_weights_path),
            default_dtype=self.VAE_dtype,
            writable_tensors=False,
            preprocess_sd=codec_preprocess_fn,
        )
        self.codec.eval()

        first_param = next(self.codec.parameters(), None)
        if first_param is not None:
            self.codec_dtype = first_param.dtype

        self.sample_rate = getattr(self.codec, "sample_rate", 48000)
        self._offload_obj = None

    def get_mmgp_pipe_config(self, optimization_level: str = "balanced") -> dict:
        """
        Get the mmgp pipe configuration for memory management.

        This should be called by the service layer to set up mmgp profiling
        AFTER the pipeline is created, similar to how Wan2GP does it.

        Args:
            optimization_level: One of:
                - "conservative": Lower VRAM, more swapping (safe for 8GB)
                - "balanced": Default for 12GB VRAM (RTX 3060)
                - "aggressive": Higher VRAM usage, faster (24GB+)

        Returns a dict with:
        - pipe: dict of model components
        - coTenantsMap: defines which models share VRAM (can't be loaded simultaneously)
        - budgets: memory budgets for certain profiles
        """
        pipe = {
            "transformer": self.mula,
            "transformer2": self.mula.decoder[0],
            "codec": self.codec
        }

        # coTenantsMap: defines which models CAN coexist in VRAM
        # Models NOT in each other's list will trigger unload before loading
        # For 12GB GPUs: transformer (~7GB) and codec (~6GB) cannot coexist
        # transformer2 is a submodule of transformer so they must coexist
        pipe_config = {
            "pipe": pipe,
            "coTenantsMap": {
                # transformer and transformer2 can coexist (they're related)
                # codec is NOT listed - loading codec will unload transformer
                "transformer": ["transformer2"],
                "transformer2": ["transformer"],
                # codec has no co-tenants - loading it unloads everything else
                "codec": [],
            }
        }

        # Add optimization-level specific settings
        if optimization_level == "conservative":
            # For 8GB VRAM - aggressive swapping
            pipe_config["budgets"] = {
                "transformer": 2500,
                "transformer2": 1500,
                "codec": 2500,
                "*": 1500
            }
        elif optimization_level == "aggressive":
            # For 24GB+ VRAM - minimal swapping
            pipe_config["budgets"] = {
                "transformer": 8000,
                "codec": 6000,
                "*": 4000
            }
            pipe_config["pinnedMemory"] = ["transformer", "transformer2"]
        # "balanced" uses default profile budgets

        return pipe_config

    def set_offload_obj(self, offload_obj):
        """Set the mmgp offload object after profiling is set up."""
        self._offload_obj = offload_obj

    def _get_mula_device(self) -> torch.device:
        if self.mula is None:
            return self.device
        text_embed = getattr(self.mula, "text_embeddings", None)
        if text_embed is not None and hasattr(text_embed, "weight"):
            return text_embed.weight.device
        first_param = next(self.mula.parameters(), None)
        return first_param.device if first_param is not None else self.device

    def _ensure_mula_loaded(self) -> None:
        if self.mula is None:
            return
        mm_manager = getattr(self.mula, "_mm_manager", None)
        mm_id = getattr(self.mula, "_mm_id", None)
        if mm_manager is None or mm_id is None:
            return
        mm_manager.ensure_model_loaded(mm_id)

    def _ensure_codec_loaded(self) -> None:
        """Ensure codec is loaded into VRAM (mmgp managed)."""
        if self.codec is None:
            return
        mm_manager = getattr(self.codec, "_mm_manager", None)
        mm_id = getattr(self.codec, "_mm_id", None)
        if mm_manager is None or mm_id is None:
            return
        mm_manager.ensure_model_loaded(mm_id)

    def _start_async_codec_preload(self):
        """
        Start preloading codec in background while transformer generates.

        This overlaps the codec loading with transformer inference,
        potentially saving 1-3 seconds of overhead.

        Returns a Future that completes when codec is ready.
        """
        if not hasattr(self, "_codec_executor"):
            self._codec_executor = ThreadPoolExecutor(max_workers=1)

        def preload_codec():
            # Small delay to let transformer start first
            import time
            time.sleep(0.5)
            self._ensure_codec_loaded()
            # Touch codec to ensure it's on device
            first_param = next(self.codec.parameters(), None)
            if first_param is not None:
                self.codec_device = first_param.device
                self.codec_dtype = first_param.dtype

        return self._codec_executor.submit(preload_codec)

    def _move_model_inputs(
        self, model_inputs: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        non_blocking = device.type == "cuda"
        for key, value in model_inputs.items():
            if torch.is_tensor(value) and value.device != device:
                model_inputs[key] = value.to(device, non_blocking=non_blocking)
        return model_inputs

    def _move_embeddings_to_device(self, device: torch.device) -> None:
        """Move small embedding layers to CUDA manually.

        mmgp handles lazy loading of large blocks (backbone, decoder), but
        small embedding layers like text_embeddings, audio_embeddings, and
        unconditional_text_embedding need to be on the same device as inputs.
        These are small enough (~50-100MB total) to keep in VRAM.
        """
        if self.mula is None:
            return

        non_blocking = device.type == "cuda"

        # List of embedding module names to move
        embedding_names = [
            "text_embeddings",
            "audio_embeddings",
            "unconditional_text_embedding",
            "muq_linear",
            "projection",
            "codebook0_head",
        ]

        for name in embedding_names:
            module = getattr(self.mula, name, None)
            if module is not None and hasattr(module, "to"):
                # Check if it's already on the target device
                first_param = next(module.parameters(), None) if hasattr(module, "parameters") else None
                if first_param is not None and first_param.device != device:
                    module.to(device, non_blocking=non_blocking)

        # Handle audio_head separately - it's an nn.Parameter
        audio_head = getattr(self.mula, "audio_head", None)
        if audio_head is not None and isinstance(audio_head, torch.nn.Parameter):
            if audio_head.data.device != device:
                # For nn.Parameter, we need to move the data in place
                audio_head.data = audio_head.data.to(device, non_blocking=non_blocking)

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def request_interrupt(self) -> None:
        """Request cancellation of the current generation."""
        self._interrupt = True

    def _read_text_or_file(self, value: str, label: str) -> str:
        if os.path.isfile(value):
            with open(value, encoding="utf-8") as fp:
                return fp.read()
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string, got {type(value)}")
        return value

    def _build_model_inputs(
        self, lyrics: str, tags: str, cfg_scale: float
    ) -> Dict[str, Any]:
        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.tokenizer.encode(tags).ids
        if tags_ids[0] != self.gen_config.text_bos_id:
            tags_ids = [self.gen_config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.gen_config.text_eos_id:
            tags_ids = tags_ids + [self.gen_config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.mula_dtype)
        muq_idx = len(tags_ids)

        lyrics = lyrics.lower()
        lyrics_ids = self.tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.gen_config.text_bos_id:
            lyrics_ids = [self.gen_config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.gen_config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.gen_config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1
        muq_idx_tensor = torch.full((bs_size,), muq_idx, dtype=torch.long)

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": muq_idx_tensor,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        callback=None,
    ):
        prompt_tokens = model_inputs["tokens"]
        mula_device = prompt_tokens.device
        prompt_tokens_mask = model_inputs["tokens_mask"]
        continuous_segment = model_inputs["muq_embed"]
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"]
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1

        self.mula.setup_caches(bs_size)
        self.mula.move_causal_masks(mula_device)
        flash_dtype = self.mula_dtype
        if flash_dtype is None:
            first_param = next(self.mula.parameters(), None)
            if first_param is not None:
                flash_dtype = first_param.dtype
        if flash_dtype is not None:
            self.mula.prepare_flash(mula_device, flash_dtype)
        if self._abort_requested():
            return None
        try:
            curr_token = self.mula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
            if curr_token is None:
                return None
            frames.append(curr_token[0:1,])
            early_stop_now = self._early_stop_requested()

            def _pad_audio_token(token: torch.Tensor):
                padded_token = (
                    torch.ones(
                        (token.shape[0], self._parallel_number),
                        device=token.device,
                        dtype=torch.long,
                    )
                    * self.gen_config.empty_id
                )
                padded_token[:, :-1] = token
                padded_token = padded_token.unsqueeze(1)
                padded_token_mask = torch.ones_like(
                    padded_token, device=token.device, dtype=torch.bool
                )
                padded_token_mask[..., -1] = False
                return padded_token, padded_token_mask

            frame_duration_ms = 80  # 80 ms per audio token frame.
            max_audio_frames = max_audio_length_ms // frame_duration_ms
            progress_total_seconds = max(1, max_audio_length_ms // 1000)
            if callback is not None:
                callback(
                    step_idx=-1,
                    override_num_inference_steps=progress_total_seconds,
                    denoising_extra=f"0s/{progress_total_seconds}s",
                    progress_unit="seconds",
                )

            if not early_stop_now:
                for i in tqdm(range(max_audio_frames)):
                    if self._abort_requested():
                        return None
                    curr_token, curr_token_mask = _pad_audio_token(curr_token)
                    curr_token = self.mula.generate_frame(
                        tokens=curr_token,
                        tokens_mask=curr_token_mask,
                        input_pos=prompt_pos[..., -1:] + i + 1,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )
                    if curr_token is None:
                        return None
                    if torch.any(curr_token[0:1, :] >= self.gen_config.audio_eos_id):
                        break
                    frames.append(curr_token[0:1,])
                    if self._early_stop_requested():
                        break
                    if i % 10 == 0 and callback is not None:
                        generated_ms = len(frames) * frame_duration_ms
                        generated_seconds_int = min(
                            progress_total_seconds,
                            generated_ms // 1000,
                        )
                        callback(
                            step_idx=generated_seconds_int - 1,
                            override_num_inference_steps=progress_total_seconds,
                            denoising_extra=(
                                f"{generated_seconds_int}s/{progress_total_seconds}s"
                            ),
                            progress_unit="seconds",
                        )
            frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
            return {"frames": frames}
        finally:
            # Drop KV cache tensors as soon as we're done with generation.
            try:
                self.mula.move_causal_masks(torch.device("cpu"))
                self.mula.release_caches()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _decode(self, frames: torch.Tensor, callback=None):
        if self._abort_requested():
            return None

        # Clean up transformer caches before codec loads
        # (mmgp will automatically unload transformer when codec loads due to coTenantsMap)
        try:
            if hasattr(self.mula, "release_caches"):
                self.mula.release_caches()
            if hasattr(self.mula, "move_causal_masks"):
                self.mula.move_causal_masks(torch.device("cpu"))
        except Exception:
            pass

        # Clamp frame indices to valid codec vocabulary range [0, 8191]
        # The model may generate special tokens (EOS=8193, etc.) that are out of bounds
        codec_vocab_size = 8192  # From codec_config.json codebook_size
        frames_clamped = frames.clamp(0, codec_vocab_size - 1)

        # mmgp will automatically:
        # 1. Check if codec can coexist with active models (transformer)
        # 2. Since codec has no co-tenants, it will unload_all() first
        # 3. Then load codec

        # Update codec device (will be set when mmgp loads it)
        first_param = next(self.codec.parameters(), None)
        if first_param is not None:
            self.codec_device = first_param.device
            self.codec_dtype = first_param.dtype

        wav = self.codec.detokenize(
            frames_clamped.to(self.codec_device),
            num_steps=self.codec_steps,
            guidance_scale=self.codec_guidance_scale,
            disable_progress=False,
            abort_signal=self._abort_requested,
        )
        if wav is None:
            return None
        if callback is not None:
            callback(step_idx=-1, force_refresh=True)
        return wav

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        alt_prompt: Optional[str] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        seed = kwargs.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is not None and seed >= 0:
            _seed_everything(seed)
        if self.mula is not None:
            self.mula_device = torch.device("cpu")
            text_embed = getattr(self.mula, "text_embeddings", None)
            if text_embed is not None and hasattr(text_embed, "weight"):
                self.mula_dtype = text_embed.weight.dtype
            else:
                first_param = next(self.mula.parameters(), None)
                if first_param is not None:
                    self.mula_dtype = first_param.dtype
        if self.codec is not None:
            first_param = next(self.codec.parameters(), None)
            if first_param is not None:
                self.codec_device = first_param.device
                self.codec_dtype = first_param.dtype
        if not input_prompt or not input_prompt.strip():
            raise ValueError("Lyrics prompt cannot be empty for HeartMuLa generation.")
        if alt_prompt is None or not str(alt_prompt).strip():
            raise ValueError("Keywords prompt cannot be empty for HeartMuLa generation.")
        if audio_guide or kwargs.get("audio_guide2"):
            raise ValueError("HeartMuLa does not support reference audio yet.")

        lyrics = self._read_text_or_file(input_prompt, "Lyrics prompt")
        tags = self._read_text_or_file(str(alt_prompt), "Keywords prompt")
        if not lyrics.strip():
            raise ValueError("Lyrics prompt cannot be empty for HeartMuLa generation.")
        if not tags.strip():
            raise ValueError("Keywords prompt cannot be empty for HeartMuLa generation.")

        cfg_scale = float(kwargs.get("cfg_scale", self.cfg_scale))
        topk_value = kwargs.get("topk", None)
        if topk_value is None:
            topk_value = kwargs.get("top_k", self.topk)
        try:
            topk = int(topk_value)
        except (TypeError, ValueError):
            topk = int(self.topk)
        duration_seconds = kwargs.get("duration_seconds", None)
        if duration_seconds is not None:
            try:
                duration_seconds = float(duration_seconds)
            except (TypeError, ValueError):
                duration_seconds = None
        if duration_seconds is not None and duration_seconds > 0:
            max_audio_length_ms = int(round(duration_seconds * 1000.0))
        else:
            max_audio_length_ms = int(
                kwargs.get("max_audio_length_ms", self.max_audio_length_ms)
            )
        callback = kwargs.get("callback")

        model_inputs = self._build_model_inputs(lyrics, tags, cfg_scale=cfg_scale)

        # mmgp will automatically load mula when needed and unload codec if active
        # (due to coTenantsMap configuration)
        self._ensure_mula_loaded()

        # Determine target device - mmgp handles lazy loading of model weights
        target_device = self._get_mula_device()
        if target_device.type != "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")

        model_inputs = self._move_model_inputs(model_inputs, target_device)

        # Async codec preloading - DISABLED by default for 12GB GPUs
        # On 12GB VRAM, this causes OOM because transformer (~7GB) and codec (~6GB)
        # cannot coexist. Set _enable_async_codec_load=True for 24GB+ GPUs.
        codec_preload_future = None
        if getattr(self, "_enable_async_codec_load", False):
            codec_preload_future = self._start_async_codec_preload()

        outputs = self._forward(
            model_inputs,
            max_audio_length_ms=max_audio_length_ms,
            temperature=float(temperature),
            topk=topk,
            cfg_scale=cfg_scale,
            callback=callback,
        )
        if outputs is None:
            return None

        # Wait for codec preload if started
        if codec_preload_future is not None:
            try:
                codec_preload_future.result(timeout=5.0)
            except Exception:
                pass  # Continue even if preload failed

        wav = self._decode(outputs["frames"], callback=callback)
        if wav is None:
            return None
        return {"x": wav, "audio_sampling_rate": self.sample_rate}

    def release(self) -> None:
        if hasattr(self, "mula") and self.mula is not None:
            self.mula = None
        if hasattr(self, "model"):
            self.model = None
        if hasattr(self, "codec") and self.codec is not None:
            self.codec = None
