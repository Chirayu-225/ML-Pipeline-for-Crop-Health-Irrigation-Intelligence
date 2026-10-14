"""Train the anomaly-screening VAE on HEALTHY leaves only, then calibrate its alert threshold.

    python -m cropiq.training.train_vae
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import RawImageDataset, load_frames
from cropiq.models.vae import ConvVAE
from cropiq.utils import Tracker, count_params, ensure_dir, get_device, get_logger, save_checkpoint, seed_everything

log = get_logger("train_vae")


@torch.no_grad()
def score_dataset(model, ds, cfg, device, batch_size=64) -> np.ndarray:
    v = cfg.vae
    loader = DataLoader(ds, batch_size, shuffle=False)
    out = []
    for x, _ in loader:
        out.append(model.anomaly_score(x.to(device), v.score, v.topk_frac).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def main() -> None:
    parser = base_parser("Train the healthy-leaf VAE")
    parser.add_argument("--run-name", default="vae")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    v = cfg.vae
    run_dir = ensure_dir(Path(cfg.paths.runs_dir) / args.run_name)

    frames = load_frames(cfg, with_synthetic=False)
    healthy_train = frames.train[frames.train.is_healthy]
    healthy_val = frames.val[frames.val.is_healthy]
    if healthy_train.empty:
        raise SystemExit("No healthy training images found (class name must end with 'healthy').")
    size = cfg.data.image_size
    train_ds = RawImageDataset(healthy_train, frames.class_to_idx, size, augment=True)
    val_ds = RawImageDataset(healthy_val, frames.class_to_idx, size, augment=False)
    loader = DataLoader(train_ds, v.batch_size, shuffle=True, drop_last=len(train_ds) > v.batch_size)
    log.info("VAE: %d healthy train / %d healthy val images", len(train_ds), len(val_ds))

    model = ConvVAE(size, v.latent_dim, v.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=v.lr, weight_decay=1e-5)
    use_amp = bool(v.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    log.info("VAE params: %.2fM | device=%s", count_params(model) / 1e6, device)
    tracker = Tracker(cfg, args.run_name, {"n_train": len(train_ds)})

    for epoch in range(1, v.epochs + 1):
        model.train()
        beta = min(1.0, epoch / max(v.kl_warmup_epochs, 1))
        t0, tot, mse_tot, kl_tot, n = time.time(), 0.0, 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                recon, mu, logvar = model(x)
            loss, mse, kl = ConvVAE.loss(x, recon.float(), mu.float(), logvar.float(), v.recon_sigma, beta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot, mse_tot, kl_tot, n = tot + loss.item(), mse_tot + mse.item(), kl_tot + kl.item(), n + 1
        if epoch % 5 == 0 or epoch == v.epochs or epoch == 1:
            log.info("epoch %03d/%d loss %.1f | pixel-MSE %.5f | KL %.1f | beta %.2f (%.0fs)", epoch, v.epochs, tot / n, mse_tot / n, kl_tot / n, beta, time.time() - t0)
        tracker.log({"loss": tot / n, "pixel_mse": mse_tot / n, "kl": kl_tot / n}, step=epoch)

    model.eval()
    val_scores = score_dataset(model, val_ds, cfg, device)
    threshold = float(np.percentile(val_scores, v.threshold_percentile)) if len(val_scores) else float("inf")
    log.info("Anomaly threshold (p%d of healthy val scores): %.5f", v.threshold_percentile, threshold)

    with torch.no_grad():
        x = torch.stack([val_ds[i][0] for i in range(min(8, len(val_ds)))]).to(device)
        recon = model(x)[0]
        save_image(torch.cat([x, recon]), str(run_dir / "reconstructions_healthy.png"), nrow=len(x), padding=1)

    save_checkpoint(
        run_dir / "best.pt",
        model_state=model.state_dict(),
        image_size=size,
        latent_dim=v.latent_dim,
        base=v.base_channels,
        threshold=threshold,
        score_mode=v.score,
        topk_frac=v.topk_frac,
        val_score_percentiles={p: float(np.percentile(val_scores, p)) for p in (50, 90, 95, 99)} if len(val_scores) else {},
    )
    tracker.finish()

    from cropiq.evaluation.eval_vae import evaluate  # local import avoids a cycle

    evaluate(cfg, args.run_name)


if __name__ == "__main__":
    main()
