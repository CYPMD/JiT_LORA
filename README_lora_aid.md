# Unconditional JiT-256 LoRA fine-tuning on AID

This trainer is written for the provided `LTH14/JiT` repository and its
256x256 ImageNet checkpoints. Copy `train_jit_lora_aid.py` into the JiT
repository root so it can import `denoiser.py` and `util/`.

## Why the model remains `class_num=1000`

The pretrained checkpoint contains an embedding table with 1,001 rows:
ImageNet IDs `0..999` plus JiT's learned null/CFG ID `1000`. The trainer keeps
that table unchanged and sends ID `1000` for every AID image. AID's directory
labels are discarded. This is unconditional training, while retaining exact
checkpoint compatibility.

Do **not** set `class_num=30`, and do not pass AID labels into the model.

## Dataset layout

Point `--data_path` at the directory containing all AID images directly:

```text
/datasets/AID/
  image_00001.jpg
  image_00002.jpg
  image_00003.png
  ...
```

The loader auto-detects this flat layout and reads `.jpg`, `.jpeg`, `.png`, and
other image extensions supported by Torchvision. It also accepts the original
class-folder layout, or a parent containing `train/`, but any numeric folder
labels are ignored in every case.

## One GPU

From the JiT repository root:

```bash
torchrun --standalone --nproc_per_node=1 train_jit_lora_aid.py \
  --data_path /datasets/AID \
  --pretrained /checkpoints/jit-b16-256/checkpoint-last.pth \
  --model JiT-B/16 \
  --output_dir ./output/aid-jit-b16-lora \
  --batch_size 16 \
  --accum_steps 4 \
  --epochs 50 \
  --warmup_epochs 2 \
  --lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --sample_every 5
```

That command has effective batch size `16 x 4 = 64`. Lower `--batch_size` and
increase `--accum_steps` if memory is tight. `bf16` is the default; use
`--amp_dtype fp16` on GPUs without BF16 support.

## Four GPUs

```bash
torchrun --standalone --nproc_per_node=4 train_jit_lora_aid.py \
  --data_path /datasets/AID \
  --pretrained /checkpoints/jit-b16-256/checkpoint-last.pth \
  --model JiT-B/16 \
  --output_dir ./output/aid-jit-b16-lora \
  --batch_size 8 \
  --accum_steps 2 \
  --epochs 50 \
  --lr 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16
```

The effective batch is `8 x 2 x 4 = 64`.

## Useful options

- `--pretrained_key auto` prefers `model_ema1`, then `model`, then
  `model_ema2`. Set it explicitly if you want another checkpoint branch.
- The default LoRA targets are attention QKV/output and both SwiGLU linear
  projections in every JiT block.
- Add `--train_uncond_embedding` to learn one small global offset on the null
  embedding. This remains unconditional, but is no longer strictly LoRA-only.
- Resume with `--resume ./output/aid-jit-b16-lora/lora-last.pth` while still
  passing the same `--pretrained` base checkpoint.
- Set `--sample_every 0` to disable sample grids. Sampling uses one null-label
  pass per denoising evaluation and no classifier-free guidance.

## Outputs

- `lora-last.pth`: trainable adapter weights, adapter EMA, optimizer, scaler,
  and resume metadata. It intentionally does not duplicate the frozen base.
- `lora-epoch-XXXX.pth`: periodic adapter checkpoints.
- `samples-epoch-XXXX.png`: unconditional EMA sample grids.
- `tensorboard/`: loss and learning-rate logs.

The loss and noise path remain the original JiT recipe: images in `[-1, 1]`,
pixel-space interpolation `z = t*x + (1-t)*noise`, direct clean-image
prediction, and velocity-space MSE with denominator clipping at `0.05`.
