"""Train the class-conditional DDPM on the (rare-reduced) REAL TRAIN split only.

The diffusion model sees exactly the images the classifier would see - never val/test - so the ablation
'real vs real+synthetic' is fair and leak-free.

    python -m cropiq.training.train_ddpm                 # ~hours on a 4 GB GPU; checkpoints every epoch
    python -m cropiq.training.train_ddpm --resume
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import RawImageDataset, load_frames
from cropiq.models.ddpm import EMA, GaussianDiffusion, build_unet
from cropiq.utils import Tracker, count_params, ensure_dir, get_device, get_logger, load_checkpoint, save_checkpoint, seed_everything

log = get_logger("train_ddpm")


@torch.no_grad()
def save_sample_grid(ema_model, diffusion, class_names, cfg, path, per_class: int = 6, max_classes: int = 12, device="cpu"):
    d = cfg.diffusion
    classes = list(range(min(len(class_names), max_classes)))
    labels = torch.tensor([c for c in classes for _ in range(per_class)], device=device)
    imgs = diffusion.ddim_sample(ema_model, labels, d.image_size, steps=min(30, d.generation.ddim_steps), guidance_scale=d.generation.guidance_scale)
    save_image((imgs + 1) / 2, str(path), nrow=per_class, padding=1)


def main() -> None:
    parser = base_parser("Train the class-conditional DDPM")
    parser.add_argument("--run-name", default="ddpm")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    d = cfg.diffusion
    run_dir = ensure_dir(Path(cfg.paths.runs_dir) / args.run_name)

    frames = load_frames(cfg, with_synthetic=False)
    names = frames.class_names
    ds = RawImageDataset(frames.train, frames.class_to_idx, d.image_size, augment=True, value_range="-11")
    accum = int(d.get("grad_accum", 1))
    loader = DataLoader(ds, d.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    log.info("DDPM training images: %d across %d classes", len(ds), len(names))

    model = build_unet(cfg, len(names)).to(device)
    ema = EMA(model, d.ema_decay)
    diffusion = GaussianDiffusion(d.timesteps, d.schedule).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=d.lr, weight_decay=0.0)
    use_amp = bool(d.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    log.info("UNet params: %.2fM | device=%s | amp=%s", count_params(model) / 1e6, device, use_amp)

    start_epoch = 1
    ckpt_path = run_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ck["model_state"])
        ema.model.load_state_dict(ck["ema_state"])
        ema.step = ck["ema_step"]
        opt.load_state_dict(ck["opt_state"])
        start_epoch = ck["epoch"] + 1
        log.info("Resumed from epoch %d", ck["epoch"])

    tracker = Tracker(cfg, args.run_name, {"n_images": len(ds)})
    for epoch in range(start_epoch, d.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = diffusion.loss(model, x, y, d.p_uncond) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            total += loss.item() * accum
            n += 1
        log.info("epoch %03d/%d loss %.4f (%.0fs)", epoch, d.epochs, total / max(n, 1), time.time() - t0)
        tracker.log({"loss": total / max(n, 1)}, step=epoch)
        payload = dict(
            model_state=model.state_dict(),
            ema_state=ema.model.state_dict(),
            ema_step=ema.step,
            opt_state=opt.state_dict(),
            epoch=epoch,
            class_names=names,
            timesteps=d.timesteps,
            schedule=d.schedule,
            unet=dict(
                image_size=d.image_size,
                base=d.base_channels,
                mults=list(d.channel_mults),
                num_res_blocks=d.num_res_blocks,
                attn_resolutions=list(d.attn_resolutions),
            ),
        )
        save_checkpoint(ckpt_path, **payload)
        if epoch % d.sample_every == 0 or epoch == d.epochs:
            save_sample_grid(ema.model, diffusion, names, cfg, run_dir / f"samples_epoch{epoch:03d}.png", device=device)
            save_checkpoint(run_dir / "best.pt", **payload)  # 'best' = latest EMA snapshot (loss is not a quality metric)
    tracker.finish()
    log.info("Done. EMA checkpoint: %s", run_dir / "best.pt")


if __name__ == "__main__":
    main()
