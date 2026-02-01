#!/usr/bin/env python3
"""
HeartMula Model Quantization Script

Quantizes HeartMula models to INT8/INT4 for reduced VRAM usage.
This allows both transformer and codec to fit in 12GB VRAM simultaneously,
eliminating the ~7.2s model shuttling overhead.

Usage:
    python quantize_heartmula.py --method int8  # INT8 quantization (~50% size reduction)
    python quantize_heartmula.py --method int4  # INT4 quantization (~75% size reduction)
"""
import os
import sys
import argparse
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm


def quantize_int8_quanto(model_path: Path, output_path: Path):
    """
    Quantize model weights to INT8 using optimum-quanto.

    This creates a quantized safetensors file that can be loaded with
    mmgp's quantization-aware loading.
    """
    try:
        from optimum.quanto import quantize, freeze, qint8
    except ImportError:
        print("ERROR: optimum-quanto not installed. Run: pip install optimum-quanto")
        sys.exit(1)

    print(f"Loading model from {model_path}")

    # Load the state dict
    state_dict = {}
    safetensor_files = sorted(model_path.glob("model-*.safetensors"))

    if not safetensor_files:
        print(f"ERROR: No model-*.safetensors files found in {model_path}")
        sys.exit(1)

    for sf_file in tqdm(safetensor_files, desc="Loading shards"):
        shard = load_file(str(sf_file))
        state_dict.update(shard)

    print(f"Loaded {len(state_dict)} tensors")

    # Quantize linear layer weights
    quantized_dict = {}
    linear_count = 0

    for name, tensor in tqdm(state_dict.items(), desc="Quantizing"):
        # Only quantize linear layer weights (2D tensors with sufficient size)
        if tensor.ndim == 2 and tensor.numel() > 1024:
            # Symmetric INT8 quantization
            tensor_fp32 = tensor.to(torch.float32)
            amax = tensor_fp32.abs().max()
            scale = amax / 127.0
            if scale == 0:
                scale = torch.tensor(1.0)

            quantized = torch.round(tensor_fp32 / scale).clamp(-127, 127).to(torch.int8)

            # Store quantized weight and scale
            quantized_dict[name] = quantized
            quantized_dict[f"{name}_scale"] = scale.to(torch.float16)
            linear_count += 1
        else:
            # Keep other tensors as-is (embeddings, norms, etc.)
            quantized_dict[name] = tensor

    print(f"Quantized {linear_count} linear layers")

    # Calculate size reduction
    original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    quantized_size = sum(t.numel() * t.element_size() for t in quantized_dict.values())

    print(f"Original size: {original_size / 1e9:.2f} GB")
    print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
    print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    # Save quantized weights
    output_file = output_path / "heartmula_int8.safetensors"
    print(f"Saving to {output_file}")
    save_file(quantized_dict, str(output_file))

    # Save metadata
    meta = {
        "quantization": "int8",
        "original_files": [f.name for f in safetensor_files],
        "linear_layers_quantized": linear_count,
        "original_size_gb": round(original_size / 1e9, 2),
        "quantized_size_gb": round(quantized_size / 1e9, 2),
    }
    with open(output_path / "heartmula_int8_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return output_file


def quantize_int4_torchao(model_path: Path, output_path: Path):
    """
    Quantize model weights to INT4 using torchao (like ACE-Step).

    This provides ~75% size reduction but requires torchao for inference.
    """
    try:
        from torch.ao.quantization import quantize_, Int4WeightOnlyConfig
    except ImportError:
        print("ERROR: torchao not available. Using INT8 instead.")
        return quantize_int8_quanto(model_path, output_path)

    print(f"Loading model from {model_path}")

    # For INT4, we need to load the full model first
    # This requires the model architecture
    sys.path.insert(0, str(Path(__file__).parent))

    from heartmula.heartmula.configuration_heartmula import HeartMuLaConfig
    from heartmula.heartmula.modeling_heartmula import HeartMuLa

    config_path = Path(__file__).parent / "heartmula" / "config" / "heartmula_3B.json"
    with open(config_path) as f:
        config = HeartMuLaConfig(**json.load(f))

    # Load model
    model = HeartMuLa(config)

    # Load weights
    safetensor_files = sorted(model_path.glob("model-*.safetensors"))
    state_dict = {}
    for sf_file in tqdm(safetensor_files, desc="Loading shards"):
        shard = load_file(str(sf_file))
        state_dict.update(shard)

    # Remove rope cache if present
    for key in ["backbone.layers.0.attn.pos_embeddings.theta",
                "backbone.layers.0.attn.pos_embeddings.cache",
                "decoder.layers.0.attn.pos_embeddings.theta",
                "decoder.layers.0.attn.pos_embeddings.cache"]:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Quantize using torchao INT4
    print("Applying INT4 quantization...")
    quantize_(model, Int4WeightOnlyConfig(group_size=128, use_hqq=True))

    # Save quantized state dict
    output_file = output_path / "heartmula_int4.bin"
    print(f"Saving to {output_file}")
    torch.save(model.state_dict(), str(output_file))

    # Calculate approximate size
    quantized_size = os.path.getsize(output_file)
    print(f"Quantized size: {quantized_size / 1e9:.2f} GB")

    return output_file


def quantize_codec_int8(model_path: Path, output_path: Path):
    """
    Quantize HeartCodec model to INT8.

    The codec is 6.3GB - quantizing to INT8 brings it to ~3.2GB.
    """
    # Try HuggingFace structure first (HeartMuLa-Studio)
    codec_file = model_path / "HeartCodec-oss" / "model.safetensors"
    if not codec_file.exists():
        # Try legacy structure (HeartMula-Studio-GP)
        codec_file = model_path / "HeartMula_codec.safetensors"
    if not codec_file.exists():
        codec_file = model_path / "HeartMula" / "HeartMula_codec.safetensors"
    if not codec_file.exists():
        print(f"ERROR: Codec not found. Tried:")
        print(f"  - {model_path / 'HeartCodec-oss' / 'model.safetensors'}")
        print(f"  - {model_path / 'HeartMula_codec.safetensors'}")
        return None

    print(f"Loading codec from {codec_file}")
    state_dict = load_file(str(codec_file))

    print(f"Loaded {len(state_dict)} tensors")

    # Quantize linear layer weights
    quantized_dict = {}
    linear_count = 0

    for name, tensor in tqdm(state_dict.items(), desc="Quantizing codec"):
        if tensor.ndim == 2 and tensor.numel() > 1024:
            tensor_fp32 = tensor.to(torch.float32)
            amax = tensor_fp32.abs().max()
            scale = amax / 127.0
            if scale == 0:
                scale = torch.tensor(1.0)

            quantized = torch.round(tensor_fp32 / scale).clamp(-127, 127).to(torch.int8)
            quantized_dict[name] = quantized
            quantized_dict[f"{name}_scale"] = scale.to(torch.float16)
            linear_count += 1
        else:
            quantized_dict[name] = tensor

    print(f"Quantized {linear_count} linear layers in codec")

    original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    quantized_size = sum(t.numel() * t.element_size() for t in quantized_dict.values())

    print(f"Codec original: {original_size / 1e9:.2f} GB")
    print(f"Codec quantized: {quantized_size / 1e9:.2f} GB")

    output_file = output_path / "HeartMula_codec_int8.safetensors"
    save_file(quantized_dict, str(output_file))

    return output_file


def find_heartmula_model_dir(base_path: Path) -> Path:
    """Find the HeartMuLa model directory (handles different structures)."""
    # HuggingFace structure: models/HeartMuLa-oss-*/
    candidates = list(base_path.glob("HeartMuLa-oss-*"))
    if candidates:
        return candidates[0]
    # Legacy structure: models/HeartMula/
    if (base_path / "HeartMula").exists():
        return base_path / "HeartMula"
    # Direct path
    if (base_path / "model-00001-of-00004.safetensors").exists():
        return base_path
    return base_path


def main():
    parser = argparse.ArgumentParser(description="Quantize HeartMula models")
    parser.add_argument("--method", choices=["int8", "int4"], default="int8",
                       help="Quantization method (default: int8)")
    parser.add_argument("--model-dir", type=str,
                       default=None,
                       help="Path to models directory (default: backend/models)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: model-dir/quantized)")
    parser.add_argument("--skip-codec", action="store_true",
                       help="Skip codec quantization")
    args = parser.parse_args()

    # Default to backend/models
    if args.model_dir is None:
        args.model_dir = str(Path(__file__).parent / "models")

    base_model_path = Path(args.model_dir)
    if not base_model_path.exists():
        print(f"ERROR: Model directory not found: {base_model_path}")
        sys.exit(1)

    # Find the HeartMuLa model subdirectory
    model_path = find_heartmula_model_dir(base_model_path)
    print(f"Found HeartMuLa model at: {model_path}")

    output_path = Path(args.output_dir) if args.output_dir else base_model_path / "quantized"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"HeartMula Quantization - {args.method.upper()}")
    print("=" * 70)
    print(f"Model dir: {model_path}")
    print(f"Output dir: {output_path}")
    print()

    # Quantize transformer
    print("=== Quantizing Transformer ===")
    if args.method == "int8":
        transformer_file = quantize_int8_quanto(model_path, output_path)
    else:
        transformer_file = quantize_int4_torchao(model_path, output_path)

    # Quantize codec (codec is at base_model_path level, not inside HeartMuLa dir)
    if not args.skip_codec:
        print("\n=== Quantizing Codec ===")
        codec_file = quantize_codec_int8(base_model_path, output_path)

    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"Transformer: {transformer_file}")
    if not args.skip_codec:
        print(f"Codec: {codec_file}")
    print()
    print("To use quantized models, update pipeline.py to load from:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()
