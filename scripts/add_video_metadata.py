"""Add FrameHeight, FrameWidth, FPS, NumberOfFrames to a FileList.csv by reading the videos.

EchoNet-Pediatric's FileList lacks these columns (EchoNet-Dynamic has them). Also maps a
10-fold integer ``Split`` (0-9) onto TRAIN/VAL/TEST the way EchoNet-Synthetic does (0-7 / 8 / 9).

    python scripts/add_video_metadata.py --filelist EchoNet-Pediatric/A4C/FileList.csv \
        --videos EchoNet-Pediatric/A4C/Videos --out data/pediatric_a4c_filelist.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def video_meta(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    m = {
        "FrameHeight": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "FrameWidth": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "FPS": round(float(cap.get(cv2.CAP_PROP_FPS)), 2),
        "NumberOfFrames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.filelist)
    stems = df["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    rows = []
    for s in stems:
        p = Path(a.videos) / f"{s}.avi"
        rows.append(video_meta(p) if p.exists() else {})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    if "Split" in df.columns and pd.api.types.is_integer_dtype(df["Split"]):
        df["Fold"] = df["Split"]
        df["Split"] = df["Fold"].map(lambda f: "TRAIN" if f <= 7 else "VAL" if f == 8 else "TEST")
    df.to_csv(a.out, index=False)
    missing = int(df["NumberOfFrames"].isna().sum()) if "NumberOfFrames" in df else len(df)
    print(f"wrote {a.out}: {len(df)} rows, {missing} without a video")


if __name__ == "__main__":
    main()
