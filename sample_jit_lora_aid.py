#!/usr/bin/env python3
"""Generate unconditional 256x256 aerial images from a JiT + AID LoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from train_jit_lora_aid import (
    UnconditionalDenoiser,
    UnconditionalEmbeddingDelta,
    autocast_context,
    inject_lora,
    load_adapter_state,
    load_checkpoint_file,
    load_pretrained,
    make_model_args,
    resolve_adapter_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unconditional aerial images from a JiT AID LoRA"
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="The same frozen JiT .pth file or checkpoint directory used for training",
    )
    parser.add_argument("--adapter", required=True, help="lora-last.pth or its directory")
    parser.add_argument("--output_dir", default="./generated-aid")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pretrained_key",
        choices=("saved", "auto", "model", "model_ema1", "model_ema2"),
        default="saved",
        help="By default, reuse the checkpoint branch recorded during LoRA training",
    )
    parser.add_argument(
        "--adapter_weights",
        choices=("ema", "raw"),
        default="ema",
        help="EMA is usually the better choice for generation",
    )
    parser.add_argument("--sampling_method", choices=("heun", "euler"), default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=None)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--no_grid", action="store_true", help="Do not save grid.png")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing same-named PNGs in a non-empty output directory",
    )
    return parser.parse_args()


def saved_value(saved_args: dict, name: str, default):
    value = saved_args.get(name, default)
    return default if value is None else value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("JiT generation requires a CUDA GPU")
    if args.num_images < 1 or args.batch_size < 1:
        raise ValueError("num_images and batch_size must be positive")
    if not args.no_grid and args.num_images > 256:
        raise ValueError("Use --no_grid when generating more than 256 images")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, for example cuda:0")
    torch.cuda.set_device(device)

    adapter_path = resolve_adapter_checkpoint(args.adapter)
    adapter_checkpoint = load_checkpoint_file(adapter_path)
    if adapter_checkpoint.get("format") != "jit-unconditional-lora-v1":
        raise ValueError("This is not a checkpoint produced by train_jit_lora_aid.py")
    saved_args = adapter_checkpoint.get("args", {})

    sampling_method = args.sampling_method or saved_value(
        saved_args, "sampling_method", "heun"
    )
    sampling_steps = args.num_sampling_steps or saved_value(
        saved_args, "num_sampling_steps", 50
    )
    amp_dtype = args.amp_dtype or saved_value(saved_args, "amp_dtype", "bf16")

    model_config = SimpleNamespace(
        model=saved_value(saved_args, "model", "JiT-B/16"),
        proj_dropout=saved_value(saved_args, "proj_dropout", 0.0),
        P_mean=saved_value(saved_args, "P_mean", -0.8),
        P_std=saved_value(saved_args, "P_std", 0.8),
        t_eps=saved_value(saved_args, "t_eps", 0.05),
        ema_decay=saved_value(saved_args, "ema_decay", 0.999),
        sampling_method=sampling_method,
        num_sampling_steps=sampling_steps,
    )

    model = UnconditionalDenoiser(make_model_args(model_config))
    if args.pretrained_key == "saved":
        pretrained_key = adapter_checkpoint.get("pretrained_key", "auto")
        if pretrained_key not in ("model", "model_ema1", "model_ema2"):
            pretrained_key = "auto"
    else:
        pretrained_key = args.pretrained_key
    loaded_pretrained_key = load_pretrained(model, args.pretrained, pretrained_key)

    model.requires_grad_(False)
    lora_targets = saved_value(
        saved_args, "lora_targets", ["attn.qkv", "attn.proj", "mlp.w12", "mlp.w3"]
    )
    matched_targets = inject_lora(
        model,
        tuple(lora_targets),
        int(saved_value(saved_args, "lora_rank", 16)),
        float(saved_value(saved_args, "lora_alpha", 16.0)),
        float(saved_value(saved_args, "lora_dropout", 0.0)),
    )
    if bool(saved_value(saved_args, "train_uncond_embedding", False)):
        model.net.y_embedder = UnconditionalEmbeddingDelta(
            model.net.y_embedder, model.net.hidden_size
        )
    model.to(device)

    adapter_state_key = "adapter_ema" if args.adapter_weights == "ema" else "adapter"
    if adapter_state_key not in adapter_checkpoint:
        raise KeyError(f"Adapter checkpoint has no {adapter_state_key!r} weights")
    load_adapter_state(model, adapter_checkpoint[adapter_state_key])
    model.eval()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_pngs = list(output_dir.glob("*.png"))
    if existing_pngs and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains PNG files; choose another directory or pass --overwrite"
        )

    # Reset the RNG after model construction so LoRA initialization does not
    # affect the requested sample seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    generated = 0
    grid_images: list[torch.Tensor] = []
    with torch.inference_mode():
        while generated < args.num_images:
            current_batch = min(args.batch_size, args.num_images - generated)
            with autocast_context(amp_dtype):
                samples = model.generate_unconditional(current_batch, device)
            samples = samples.float().add(1.0).div(2.0).clamp(0.0, 1.0).cpu()

            for batch_index, image in enumerate(samples):
                image_index = generated + batch_index
                save_image(image, output_dir / f"{image_index:06d}.png")
                if not args.no_grid:
                    grid_images.append(image)
            generated += current_batch
            print(f"Generated {generated}/{args.num_images}")

    if grid_images:
        grid_columns = max(1, math.ceil(math.sqrt(len(grid_images))))
        save_image(torch.stack(grid_images), output_dir / "grid.png", nrow=grid_columns)

    metadata = {
        "pretrained": str(Path(args.pretrained).expanduser()),
        "pretrained_key": loaded_pretrained_key,
        "adapter": str(adapter_path),
        "adapter_weights": args.adapter_weights,
        "model": model_config.model,
        "lora_modules": len(matched_targets),
        "unconditional_label_id": model.num_classes,
        "num_images": args.num_images,
        "seed": args.seed,
        "sampling_method": sampling_method,
        "num_sampling_steps": sampling_steps,
        "amp_dtype": amp_dtype,
    }
    with (output_dir / "generation.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"Images saved to {output_dir}")


if __name__ == "__main__":
    main()
