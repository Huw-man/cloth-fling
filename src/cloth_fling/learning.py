"""Small RGB-only immediate-value learner. Torch is imported only by model operations."""

import io
import json
from pathlib import Path

import numpy as np

from cloth_fling.data.storage import atomic_write, file_sha256, json_bytes
from cloth_fling.frozen_laydown import CONFIG_SHA256
from cloth_fling.spatial_actions import learning_samples

MODEL = "rgb_unet_32_64_128_256_v1"
PREPROCESSING = "reference_spatial_v1_nearest_uint8_div255"


def transformed_rgb(rgb, matrix):
    """Replay the *saved* transform, without depth, masks or a new foreground crop."""
    rgb, matrix = np.asarray(rgb), np.asarray(matrix, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("learning requires uint8 RGB")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid saved RGB transform")
    yy, xx = np.mgrid[:64, :64]
    original = (np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ matrix.T)[..., :2]
    h, w = rgb.shape[:2]
    good = (
        (original[..., 0] >= 0)
        & (original[..., 0] <= w - 1)
        & (original[..., 1] >= 0)
        & (original[..., 1] <= h - 1)
    )
    ij = np.rint(original).astype(int)
    result = np.zeros((64, 64, 3), dtype=np.uint8)
    result[good] = rgb[ij[..., 1][good], ij[..., 0][good]]
    return result


def training_arrays(datasets):
    """Strict train-only projection; invalid/unsettled rows stay in source datasets."""
    images, pixels, rewards, sources = [], [], [], []
    for directory in datasets:
        root = Path(directory)
        sources.append({"relative_file": str(root), "sha256": file_sha256(root / "manifest.json")})
        for sample in learning_samples(root):
            if sample["split"] != "train":
                raise ValueError("non-training family cannot enter fitting")
            encoding = sample["action"]
            if encoding.get("codec") != "reference_spatial_v1":
                raise ValueError("unsupported learning action codec")
            t, v, u = encoding["map_index_tvu"]
            if not (0 <= t < 96 and 0 <= v < 64 and 0 <= u < 64):
                raise ValueError("invalid executed map index")
            if not sample["valid_action_mask"][t, v, u]:
                raise ValueError("executed action absent from saved feasible mask")
            images.append(transformed_rgb(sample["rgb"], encoding["to_original"]))
            pixels.append((v, u))
            rewards.append(sample["reward"])
    if not images:
        raise ValueError("no eligible settled training samples")
    return np.stack(images), np.asarray(pixels), np.asarray(rewards, np.float32), sources


def build_model():
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(8, b),
            nn.ReLU(),
        )

    class ValueUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([block(3, 32), block(32, 64), block(64, 128)])
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = block(128, 256)
            self.decoders = nn.ModuleList([block(384, 128), block(192, 64), block(96, 32)])
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            skips = []
            for encoder in self.encoders:
                x = encoder(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for decoder, skip in zip(self.decoders, reversed(skips)):
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = decoder(torch.cat((x, skip), dim=1))
            return self.head(x)[:, 0]

    return ValueUNet()


def seed_training(seed=1001):
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def executed_loss(predictions, pixels, rewards):
    import torch

    values = predictions[
        torch.arange(len(predictions), device=predictions.device), pixels[:, 0], pixels[:, 1]
    ]
    return torch.nn.functional.huber_loss(values, rewards, delta=0.1)


def load_model(checkpoint, *, device="cpu"):
    import torch

    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    if (
        metadata["model"] != MODEL
        or metadata["preprocessing"] != PREPROCESSING
        or metadata["frozen_config_sha256"] != CONFIG_SHA256
        or file_sha256(checkpoint) != metadata["weights_sha256"]
    ):
        raise ValueError("incompatible or corrupted learning checkpoint")
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite checkpoint weights")
    model.eval()
    return model, metadata


def predictor(checkpoint, *, device="cuda"):
    import torch

    model, _ = load_model(checkpoint, device=device)

    def predict(rgb):
        if rgb.shape != (96, 64, 64, 3) or rgb.dtype != np.uint8:
            raise ValueError("predictor expects transformed uint8 RGB only")
        output = []
        with torch.inference_mode():
            for chunk in np.array_split(rgb, 6):
                x = torch.as_tensor(chunk.copy(), device=device).permute(0, 3, 1, 2).float() / 255
                output.append(model(x).cpu().numpy())
        result = np.concatenate(output)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite predicted action values")
        return result

    return predict


def fit(datasets, output, *, previous=None, device="cuda", steps=2000, check_deadline=lambda: None):
    return fit_arrays(
        training_arrays(datasets), output, previous=previous, device=device,
        steps=steps, check_deadline=check_deadline,
    )


def fit_arrays(arrays, output, *, previous=None, device="cpu", steps=2000,
               check_deadline=lambda: None):
    """Fit the same learner from an independently verified saved RGB projection."""
    import torch

    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("training checkpoint is immutable")
    if steps < 1:
        raise ValueError("training steps must be positive")
    images, pixels, rewards, sources = arrays
    if len(images) < 32:
        raise ValueError("insufficient eligible training data: need 32 settled samples")
    seed_training()
    if previous is None:
        model = build_model().to(device)
    else:
        model, _ = load_model(previous, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x = torch.as_tensor(images, device=device).permute(0, 3, 1, 2).float() / 255
    uv = torch.as_tensor(pixels, device=device)
    y = torch.as_tensor(rewards, device=device)
    rng = np.random.default_rng(1001)
    losses = []
    for step in range(steps):
        check_deadline()
        indices = torch.as_tensor(rng.integers(len(x), size=16), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = executed_loss(model(x[indices]), uv[indices], y[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        reference = model(x[:1]).cpu()
    stream = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
    atomic_write(output.parent, output.name, stream.getvalue())
    metadata = {
        "model": MODEL,
        "preprocessing": PREPROCESSING,
        "seed": 1001,
        "steps": steps,
        "batch_size": 16,
        "huber_delta": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "frozen_config_sha256": CONFIG_SHA256,
        "weights_sha256": file_sha256(output),
        "sources": sources,
        "eligible_samples": len(images),
        "losses": losses,
        "previous_weights_sha256": file_sha256(previous) if previous else None,
    }
    atomic_write(output.parent, output.with_suffix(".json").name, json_bytes(metadata))
    reloaded, _ = load_model(output, device=device)
    with torch.inference_mode():
        if not torch.equal(reference, reloaded(x[:1]).cpu()):
            raise RuntimeError("checkpoint inference changed after reload")
    return metadata
