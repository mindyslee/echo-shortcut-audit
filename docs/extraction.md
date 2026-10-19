# Extracting embeddings from other echo foundation models

`scripts/extract_embeddings.py` turns a folder of EchoNet `.avi` files into the `.npz` that
`echoaudit run --embeddings` reads. One adapter per model; what each needs and where it comes
from (checked 19 Oct 2026 against the official repos — details may drift):

| model | weights | license | input the adapter builds | d | notes |
|---|---|---|---|---|---|
| EchoCLIP | `hf-hub:mkaichristensen/echo-clip` (open_clip; HF login needed) | research-only (Cedars academic license) | up to 20 evenly spaced frames, 224px, CLIP norm; mean of L2-normalised frame embeddings | 512 | per-frame model, no temporal modelling |
| EchoPrime | `model_data/weights/echo_prime_encoder.pt` from the GitHub release | MIT | 16 frames at stride 2, 224px cubic, 0–255 pixels with the model's own mean/std; 4 windows averaged | 512 | `pip install torchvision`; `--weights` required |
| PanEcho | `torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", backbone_only=True)` | CC BY-NC-SA 4.0 (weights), AGPL-3.0 (code) | 16 contiguous frames, resize 256 → centre-crop 224, ImageNet norm; 4 windows averaged | 768 | needs `timm` |
| EchoFM | `sekeun/EchoFM` on HF (`echofm_vitl.pth`, 1.4 GB) + cloned repo for `models_mae` | CC BY-NC(-ND) 4.0 | 32 frames, 224px, [0,1], no normalisation; mean-pooled encoder tokens | 1024 | `--repo` required |
| EchoJEPA | Google Drive `vitl-vmix22m-pt220-c55.pt` (4.8 GB) + cloned repo | Apache-2.0 | 16 frames at stride 2, 224px, ImageNet norm; mean-pooled tokens | 1024 | **experimental**: no official embedding script; verify vit_large kwargs against the repo's `configs/` |

Videos are DUA-covered: run this on the machine that holds them. A CPU handles EchoCLIP at
a few videos per second; the video models want a GPU (an M-series Mac with `--device mps`
works, slowly). `--limit 20` is a smoke test.

```bash
python scripts/extract_embeddings.py --model echoprime --videos EchoNet-Dynamic/Videos \
    --weights model_data/weights/echo_prime_encoder.pt --out data/echoprime.npz
echoaudit run --embeddings data/echoprime.npz --filelist data/FileList.csv --out results_echoprime/
python scripts/compare.py results_r3d/audit.json results_echoprime/audit.json
```

Two choices that differ from the official inference code, on purpose: clips are taken at
evenly spaced windows and averaged rather than at a random start, so the embedding is
deterministic; and no sector mask is applied, because EchoNet videos are already masked.

## EchoNet-Pediatric

Its `FileList.csv` carries `FileName, EF, Sex, Age, Weight, Height, Split` (verify against your
copy; `Split` is a 0–9 fold index) and **no** acquisition columns. `scripts/add_video_metadata.py`
reads each video and appends `FrameHeight, FrameWidth, FPS, NumberOfFrames`, and maps folds
0–7/8/9 onto TRAIN/VAL/TEST. Then:

```bash
echoaudit run --embeddings data/r3d_pediatric.npz --filelist data/pediatric_a4c_filelist.csv \
    --out results_peds/ --clinical EF --nuisance FPS NumberOfFrames Age Sex Weight Height
```

The R3D model that produced the adult embeddings is in the r3d-v0-embeddings repo
(`embedder/`, Git LFS); run it over the pediatric A4C videos to get the matching embedding.
