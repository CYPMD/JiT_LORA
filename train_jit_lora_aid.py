#!/usr/bin/env python3
"""LoRA fine-tuning of a pretrained 256x256 JiT model on AID, unconditionally.

Place this file in the root of the LTH14/JiT repository. The base ImageNet
model keeps num_classes=1000 so its checkpoint loads exactly; every AID image
is passed the learned null label (ID 1000), and AID folder/class labels are
never used.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

from denoiser import Denoiser
from util.crop import center_crop_arr


DEFAULT_LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.w12", "mlp.w3")


class ImagesOnly(Dataset):
    """Use ImageFolder for discovery, but deliberately discard every label."""

    def __init__(self, root: str | Path, transform):
        self.dataset = datasets.ImageFolder(str(root), transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _unused_class = self.dataset[index]
        return image


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + lora_output * self.scaling


class UnconditionalEmbeddingDelta(nn.Module):
    """Optional trainable global offset on top of JiT's frozen null embedding."""

    def __init__(self, base: nn.Module, hidden_size: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.delta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.base(labels) + self.delta.unsqueeze(0)


class UnconditionalDenoiser(Denoiser):
    """JiT x-prediction/v-loss with the null label used for every image."""

    def _null_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size,), self.num_classes, dtype=torch.long, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # label_drop_prob is set to zero, and the input is already the null token.
        return super().forward(x, self._null_labels(x.shape[0], x.device))

    @torch.no_grad()
    def _forward_sample(
        self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        # One unconditional network pass. CFG is neither needed nor meaningful.
        null_labels = self._null_labels(z.shape[0], z.device)
        x_pred = self.net(z, t.flatten(), null_labels)
        return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def generate_unconditional(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        z = self.noise_scale * torch.randn(
            batch_size, 3, self.img_size, self.img_size, device=device
        )
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, 1, 1, 1, 1).expand(
            -1, batch_size, -1, -1, -1
        )

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

        labels = self._null_labels(batch_size, device)
        for index in range(self.steps - 1):
            z = stepper(z, timesteps[index], timesteps[index + 1], labels)
        return self._euler_step(z, timesteps[-2], timesteps[-1], labels)


def get_submodule_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), child_name


def inject_lora(
    model: nn.Module,
    targets: tuple[str, ...] | list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            matches.append((name, module))

    if not matches:
        raise ValueError(f"No Linear modules matched --lora_targets={targets}")

    for name, module in matches:
        parent, child_name = get_submodule_parent(model, name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    return [name for name, _ in matches]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def clean_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module.") :]: value for name, value in state.items()}
    return state


def resolve_checkpoint(path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint-last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_adapter_checkpoint(path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "lora-last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint_file(path: str | Path) -> dict:
    # Checkpoints can contain argparse metadata and optimizer state. Only load
    # files you trust; weights_only=False is explicit for PyTorch >= 2.6.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pretrained(
    model: nn.Module, checkpoint_path: str | Path, requested_key: str
) -> str:
    checkpoint = load_checkpoint_file(resolve_checkpoint(checkpoint_path))
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected the pretrained checkpoint to contain a dictionary")

    if requested_key == "auto":
        key = next(
            (candidate for candidate in ("model_ema1", "model", "model_ema2") if candidate in checkpoint),
            None,
        )
        if key is None:
            state = checkpoint
            key = "<root>"
        else:
            state = checkpoint[key]
    else:
        if requested_key not in checkpoint:
            raise KeyError(
                f"Checkpoint has no key {requested_key!r}; available keys: {list(checkpoint)}"
            )
        key = requested_key
        state = checkpoint[key]

    state = clean_state_dict_keys(state)
    model_keys = model.state_dict().keys()
    if state and not any(name.startswith("net.") for name in state) and any(
        name.startswith("net.") for name in model_keys
    ):
        state = {f"net.{name}": value for name, value in state.items()}

    model.load_state_dict(state, strict=True)
    return key


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(trainable_state_dict(model))
    received = set(state)
    if expected != received:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise RuntimeError(
            f"Adapter key mismatch. Missing={missing}; unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")


def init_distributed() -> tuple[bool, int, int, int]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(local_rank)
    return distributed, rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def learning_rate_at(
    update: int, total_updates: int, warmup_updates: int, peak_lr: float, min_lr: float
) -> float:
    if warmup_updates > 0 and update < warmup_updates:
        return peak_lr * (update + 1) / warmup_updates
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(
    ema: dict[str, torch.Tensor], model: nn.Module, decay: float
) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            ema[name].mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)


@contextlib.contextmanager
def use_ema_weights(model: nn.Module, ema: dict[str, torch.Tensor]):
    parameters = dict(model.named_parameters())
    backup = {name: parameters[name].detach().clone() for name in ema}
    try:
        with torch.no_grad():
            for name, value in ema.items():
                parameters[name].copy_(value.to(dtype=parameters[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, value in backup.items():
                parameters[name].copy_(value)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    ema: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    matched_targets: list[str],
    pretrained_key: str,
) -> None:
    payload = {
        "format": "jit-unconditional-lora-v1",
        "adapter": trainable_state_dict(model),
        "adapter_ema": {name: value.detach().cpu() for name, value in ema.items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "matched_targets": matched_targets,
        "pretrained_key": pretrained_key,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    # Keep class_num=1000: 1000 is the pretrained checkpoint's null label ID.
    return SimpleNamespace(
        model=args.model,
        img_size=256,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=args.proj_dropout,
        label_drop_prob=0.0,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=1.0,
        ema_decay1=args.ema_decay,
        ema_decay2=args.ema_decay,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
    )


def build_dataset(data_path: str) -> tuple[ImagesOnly, Path]:
    root = Path(data_path).expanduser()
    if (root / "train").is_dir():
        root = root / "train"
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )
    return ImagesOnly(root, transform), root


def autocast_context(dtype_name: str):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unconditional LoRA fine-tuning of JiT-256 on AID"
    )
    parser.add_argument("--data_path", required=True, help="AID root (class folders are ignored)")
    parser.add_argument("--pretrained", required=True, help="Base .pth file or checkpoint directory")
    parser.add_argument(
        "--pretrained_key",
        choices=("auto", "model", "model_ema1", "model_ema2"),
        default="auto",
    )
    parser.add_argument("--output_dir", default="./output/aid-jit-lora")
    parser.add_argument("--resume", default="", help="LoRA checkpoint produced by this script")

    parser.add_argument(
        "--model",
        default="JiT-B/16",
        choices=("JiT-B/16", "JiT-L/16", "JiT-H/16"),
    )
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--t_eps", type=float, default=0.05)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets", nargs="+", default=list(DEFAULT_LORA_TARGETS)
    )
    parser.add_argument(
        "--train_uncond_embedding",
        action="store_true",
        help="Also train one global null-conditioning offset (still unconditional)",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16, help="Per GPU")
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "fp32"), default="bf16")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5, help="Epoch frequency")
    parser.add_argument("--sample_every", type=int, default=5, help="0 disables samples")
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--sampling_method", choices=("heun", "euler"), default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The JiT repository constructs CUDA RoPE tensors; a CUDA GPU is required")
    if args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError("batch_size and accum_steps must be positive")
    if args.save_every < 1:
        raise ValueError("save_every must be positive")
    if args.sample_every > 0 and args.sample_count < 1:
        raise ValueError("sample_count must be positive when sampling is enabled")

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    dataset, dataset_root = build_dataset(args.data_path)
    sampler = (
        DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.seed)
        if distributed
        else RandomSampler(dataset)
    )
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(data_loader) == 0:
        raise ValueError("No complete batch is available; lower --batch_size")

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    model = UnconditionalDenoiser(make_model_args(args))
    pretrained_key = load_pretrained(model, args.pretrained, args.pretrained_key)
    model.requires_grad_(False)
    matched_targets = inject_lora(
        model, tuple(args.lora_targets), args.lora_rank, args.lora_alpha, args.lora_dropout
    )
    if args.train_uncond_embedding:
        model.net.y_embedder = UnconditionalEmbeddingDelta(
            model.net.y_embedder, model.net.hidden_size
        )
    model.to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    ema = {
        name: parameter.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_checkpoint = load_checkpoint_file(resolve_adapter_checkpoint(args.resume))
        saved_args = resume_checkpoint.get("args", {})
        for field in ("model", "lora_rank", "lora_alpha", "lora_targets", "train_uncond_embedding"):
            if field in saved_args and saved_args[field] != getattr(args, field):
                raise ValueError(
                    f"Resume mismatch for {field}: saved={saved_args[field]!r}, current={getattr(args, field)!r}"
                )
        load_adapter_state(model, resume_checkpoint["adapter"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if "adapter_ema" in resume_checkpoint:
            ema = {
                name: value.to(device=device, dtype=torch.float32)
                for name, value in resume_checkpoint["adapter_ema"].items()
            }
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_step = int(resume_checkpoint.get("global_step", 0))

    wrapped_model: nn.Module = model
    if distributed:
        wrapped_model = DDP(
            model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False
        )

    updates_per_epoch = math.ceil(len(data_loader) / args.accum_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = int(args.warmup_epochs * updates_per_epoch)
    writer = SummaryWriter(output_dir / "tensorboard") if is_main(rank) else None

    if is_main(rank):
        print(f"AID image root: {dataset_root}")
        print(f"Images: {len(dataset):,}")
        print(f"Loaded pretrained weights from key: {pretrained_key}")
        print(f"LoRA modules: {len(matched_targets)}")
        print(f"Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.3f}%)")
        print(f"Effective batch: {args.batch_size * args.accum_steps * world_size}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            assert isinstance(sampler, DistributedSampler)
            sampler.set_epoch(epoch)
        wrapped_model.train()
        epoch_loss = 0.0
        epoch_items = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, images in enumerate(data_loader):
            group_start = (batch_index // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, len(data_loader) - group_start)
            group_end = batch_index + 1 == group_start + group_size

            images = images.to(device, non_blocking=True).float().div_(255.0)
            images = images.mul_(2.0).sub_(1.0)
            sync_context = (
                contextlib.nullcontext()
                if not distributed or group_end
                else wrapped_model.no_sync()  # type: ignore[attr-defined]
            )
            with sync_context:
                with autocast_context(args.amp_dtype):
                    loss = wrapped_model(images)
                scaler.scale(loss / group_size).backward()

            reduced_loss = reduce_mean(loss, world_size)
            epoch_loss += reduced_loss.item() * images.shape[0]
            epoch_items += images.shape[0]

            if group_end:
                lr = learning_rate_at(
                    global_step, total_updates, warmup_updates, args.lr, args.min_lr
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, args.ema_decay)
                global_step += 1

                if writer is not None:
                    writer.add_scalar("train/loss", reduced_loss.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                if is_main(rank) and global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} update={global_step}/{total_updates} "
                        f"loss={reduced_loss.item():.6f} lr={lr:.3e}"
                    )

        mean_epoch_loss = epoch_loss / max(1, epoch_items)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", mean_epoch_loss, epoch + 1)

        should_save = (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs
        if is_main(rank):
            print(f"epoch={epoch + 1} mean_loss={mean_epoch_loss:.6f}")
            save_checkpoint(
                output_dir / "lora-last.pth",
                model,
                ema,
                optimizer,
                scaler,
                epoch,
                global_step,
                args,
                matched_targets,
                pretrained_key,
            )
            if should_save:
                save_checkpoint(
                    output_dir / f"lora-epoch-{epoch + 1:04d}.pth",
                    model,
                    ema,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    args,
                    matched_targets,
                    pretrained_key,
                )

        should_sample = args.sample_every > 0 and (
            (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs
        )
        if distributed:
            dist.barrier()
        if is_main(rank) and should_sample:
            model.eval()
            with use_ema_weights(model, ema):
                with autocast_context(args.amp_dtype):
                    samples = model.generate_unconditional(args.sample_count, device)
            grid_rows = max(1, int(math.sqrt(args.sample_count)))
            save_image(
                samples.float().add(1.0).div(2.0).clamp(0.0, 1.0),
                output_dir / f"samples-epoch-{epoch + 1:04d}.png",
                nrow=grid_rows,
            )
        if distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    if is_main(rank):
        elapsed = time.time() - start_time
        print(f"Finished in {elapsed / 3600:.2f} hours. Adapter: {output_dir / 'lora-last.pth'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
