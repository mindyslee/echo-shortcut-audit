"""Extract one embedding per echo video from a public echo foundation model.

    python scripts/extract_embeddings.py --model echoclip  --videos EchoNet-Dynamic/Videos --out data/echoclip.npz
    python scripts/extract_embeddings.py --model echoprime --videos ... --weights model_data/weights/echo_prime_encoder.pt --out data/echoprime.npz
    python scripts/extract_embeddings.py --model panecho   --videos ... --out data/panecho.npz
    python scripts/extract_embeddings.py --model echofm    --videos ... --repo /path/to/EchoFM --out data/echofm.npz
    python scripts/extract_embeddings.py --model echojepa  --videos ... --repo /path/to/EchoJEPA --weights vitl-vmix22m-pt220-c55.pt --out data/echojepa.npz

Output is ``.npz`` with arrays ``ids`` (video id without extension) and ``X`` (n, d) float32,
which ``echoaudit run --embeddings data/<model>.npz`` reads directly.

Needs a GPU (a laptop CPU works for EchoCLIP at a few videos/s; the video models are slower),
``torch``, ``torchvision``, ``opencv-python``, and the model's own dependencies (see
``docs/extraction.md``). Each adapter follows the model's official preprocessing as closely as
the released code allows and makes the clip choice deterministic: where the official code picks
a random window, several evenly spaced windows are averaged instead.

The EchoJEPA adapter is assembled from the repository's pieces because no official embedding
script exists; treat its output as experimental until checked against the repo's configs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------- video io


def read_avi(path: Path) -> np.ndarray:
    """(T, H, W, 3) uint8. EchoNet videos are grayscale stored as 3 channels."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames)


def resize_frames(frames: np.ndarray, size: int, interpolation: str = "cubic") -> np.ndarray:
    import cv2

    interp = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}[
        interpolation
    ]
    return np.stack([cv2.resize(f, (size, size), interpolation=interp) for f in frames])


def windows(n_frames: int, length: int, stride: int, n_windows: int) -> list[np.ndarray]:
    """Evenly spaced start indices for ``n_windows`` clips of ``length`` frames at ``stride``."""
    span = (length - 1) * stride + 1
    if n_frames <= span:
        idx = np.arange(0, span, stride)
        idx = np.minimum(idx, n_frames - 1)  # last-frame carry-forward
        return [idx]
    starts = np.linspace(0, n_frames - span, n_windows).round().astype(int)
    return [s + np.arange(length) * stride for s in np.unique(starts)]


def to_tensor_clip(frames: np.ndarray, idx: np.ndarray, mean, std, scale_to_unit: bool):
    """(3, T, H, W) float tensor, normalized."""
    import torch

    clip = frames[idx].astype(np.float32)
    if scale_to_unit:
        clip = clip / 255.0
    clip = (clip - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(clip).permute(3, 0, 1, 2)


# ----------------------------------------------------------------- adapters
# Each adapter is a class with ``dim`` and ``embed(frames: (T,H,W,3) uint8) -> np.ndarray (d,)``.


class EchoCLIP:
    """Per-frame ConvNeXt-B image tower; video = mean of L2-normalised frame embeddings."""

    dim = 512

    def __init__(self, device: str, n_frames: int = 20, **_):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip", device=device
        )
        self.model.eval()
        self.n_frames = n_frames

    def embed(self, frames: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F
        from PIL import Image

        idx = np.linspace(0, len(frames) - 1, min(self.n_frames, len(frames))).round().astype(int)
        x = self.torch.stack([self.pre(Image.fromarray(frames[i])) for i in idx]).to(self.device)
        with self.torch.no_grad():
            fe = F.normalize(self.model.encode_image(x), dim=-1)
        return fe.mean(0).float().cpu().numpy()


class EchoPrime:
    """MViTv2-S video encoder, 512-d. Official: 16 frames at stride 2 from the start, 224px."""

    dim = 512
    MEAN = (29.110628, 28.076836, 29.096405)  # applied to 0-255 pixels
    STD = (47.989223, 46.456997, 47.20083)

    def __init__(self, device: str, weights: str, n_windows: int = 4, **_):
        import torch
        import torchvision

        self.torch, self.device, self.n_windows = torch, device, n_windows
        enc = torchvision.models.video.mvit_v2_s()
        enc.head[-1] = torch.nn.Linear(enc.head[-1].in_features, 512)
        enc.load_state_dict(torch.load(weights, map_location=device))
        self.enc = enc.eval().to(device)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        fr = resize_frames(frames, 224, "cubic")
        outs = []
        with self.torch.no_grad():
            for idx in windows(len(fr), 16, 2, self.n_windows):
                x = to_tensor_clip(fr, idx, self.MEAN, self.STD, scale_to_unit=False)
                outs.append(self.enc(x[None].to(self.device))[0])
        return self.torch.stack(outs).mean(0).float().cpu().numpy()


class PanEcho:
    """ConvNeXt-T per frame + temporal transformer backbone, 768-d. Official: 16 contiguous frames."""

    dim = 768

    def __init__(self, device: str, n_windows: int = 4, **_):
        import torch

        self.torch, self.device, self.n_windows = torch, device, n_windows
        self.backbone = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", backbone_only=True)
        self.backbone.eval().to(device)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        fr = resize_frames(frames, 256, "area")[
            :, 16:240, 16:240
        ]  # resize 256 then centre-crop 224
        outs = []
        with self.torch.no_grad():
            for idx in windows(len(fr), 16, 1, self.n_windows):
                x = to_tensor_clip(fr, idx, IMAGENET_MEAN, IMAGENET_STD, scale_to_unit=True)
                outs.append(self.backbone(x[None].to(self.device))[0])
        return self.torch.stack(outs).mean(0).float().cpu().numpy()


class EchoFM:
    """ViT-L video MAE, 1024-d mean-pooled tokens. Official: 32 frames, 224px, [0,1] input."""

    dim = 1024

    def __init__(self, device: str, repo: str, n_windows: int = 2, **_):
        import torch
        from huggingface_hub import hf_hub_download

        sys.path.insert(0, repo)
        from EchoFM import models_mae  # type: ignore

        self.torch, self.device, self.n_windows = torch, device, n_windows
        ck = torch.load(hf_hub_download("sekeun/EchoFM", "echofm_vitl.pth"), map_location="cpu")
        keys = [
            "num_frames",
            "t_patch_size",
            "pred_t_dim",
            "sep_pos_embed",
            "cls_embed",
            "norm_pix_loss",
        ]
        self.model = models_mae.mae_vit_large_patch16(**{k: ck["model_args"][k] for k in keys})
        missing = self.model.load_state_dict(ck["model"], strict=False)
        if missing.missing_keys:
            print(
                f"EchoFM: {len(missing.missing_keys)} missing keys (expected for decoder-only parts)",
                file=sys.stderr,
            )
        self.model.eval().to(device)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        fr = resize_frames(frames, 224, "cubic")
        outs = []
        with self.torch.no_grad():
            for idx in windows(len(fr), 32, 1, self.n_windows):
                x = to_tensor_clip(fr, idx, (0, 0, 0), (1, 1, 1), scale_to_unit=True)
                latent, _, _ = self.model.forward_encoder(x[None].to(self.device), mask_ratio=0.0)
                outs.append(latent.mean(1)[0])
        return self.torch.stack(outs).mean(0).float().cpu().numpy()


class EchoJEPA:
    """EXPERIMENTAL. ViT-L V-JEPA2-style encoder, 1024-d mean-pooled tokens.

    No official embedding script exists; this mirrors the frozen-eval transforms
    (16 frames, step 2, 224px, ImageNet-normalised) and the hub kwargs for vit_large.
    Check the kwargs against the repo's configs/ for the checkpoint you use.
    """

    dim = 1024

    def __init__(self, device: str, repo: str, weights: str, n_windows: int = 4, **_):
        import torch

        sys.path.insert(0, repo)
        from src.models.vision_transformer import vit_large  # type: ignore

        self.torch, self.device, self.n_windows = torch, device, n_windows
        enc = vit_large(
            img_size=(224, 224),
            num_frames=16,
            tubelet_size=2,
            patch_size=16,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
        )
        ck = torch.load(weights, map_location="cpu")
        sd = ck.get("target_encoder", ck.get("encoder", ck))
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        res = enc.load_state_dict(sd, strict=False)
        print(
            f"EchoJEPA: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
            file=sys.stderr,
        )
        self.enc = enc.eval().to(device)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        fr = resize_frames(frames, 224, "linear")
        outs = []
        with self.torch.no_grad():
            for idx in windows(len(fr), 16, 2, self.n_windows):
                x = to_tensor_clip(fr, idx, IMAGENET_MEAN, IMAGENET_STD, scale_to_unit=True)
                outs.append(self.enc(x[None].to(self.device)).mean(1)[0])
        return self.torch.stack(outs).mean(0).float().cpu().numpy()


ADAPTERS = {
    "echoclip": EchoCLIP,
    "echoprime": EchoPrime,
    "panecho": PanEcho,
    "echofm": EchoFM,
    "echojepa": EchoJEPA,
}


# ----------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--videos", required=True, help="directory of .avi files")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--weights", default=None, help="checkpoint path (echoprime, echojepa)")
    ap.add_argument("--repo", default=None, help="cloned model repo path (echofm, echojepa)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N videos, for a smoke test")
    ap.add_argument(
        "--n-windows", type=int, default=4, help="clips averaged per video (video models)"
    )
    a = ap.parse_args(argv)

    import torch

    device = a.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    paths = sorted(Path(a.videos).glob("*.avi"))[: a.limit]
    if not paths:
        sys.exit(f"no .avi files in {a.videos}")
    adapter = ADAPTERS[a.model](
        device=device, weights=a.weights, repo=a.repo, n_windows=a.n_windows
    )
    print(f"{a.model} on {device}: {len(paths)} videos -> {adapter.dim}-d", file=sys.stderr)

    ids, X, failed = [], [], []
    for i, p in enumerate(paths):
        try:
            v = adapter.embed(read_avi(p))
            ids.append(p.stem)
            X.append(v.astype(np.float32))
        except Exception as e:  # keep going; report at the end
            failed.append((p.name, str(e)[:120]))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}", file=sys.stderr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.asarray(ids), X=np.stack(X))
    print(f"wrote {out}: {len(ids)} x {X[0].shape[0]}; {len(failed)} failed", file=sys.stderr)
    for name, err in failed[:10]:
        print(f"  failed {name}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
