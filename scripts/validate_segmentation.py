"""Stage 2 segmentation smoke-validation on the real DIC images we have
(179997/TRX2, 165478/YAP1, 182391/HAC1 — the three single-GFP entries, the
only ones in data/raw with a bright-field/DIC channel).

No manual ground-truth annotations exist yet, so this is not accuracy
validation -- it runs segment.segment_cells(method="otsu") on real data and
writes an overlay PNG (DIC + detected cell boundaries) per image plus basic
per-image stats (cell count, area distribution), for a human (Vicky) to
eyeball before deciding whether Otsu+watershed is good enough or cellpose is
needed.

Usage: python scripts/validate_segmentation.py
"""
from pathlib import Path

import numpy as np
from skimage.segmentation import mark_boundaries
from PIL import Image

from ichnos_image import segment
from ichnos_image.synthesize import load_png16

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "segmentation_test"

DIC_IMAGES = {
    "TRX2_179997": RAW_DIR / "179997_TRX2_dic.png",
    "YAP1_165478": RAW_DIR / "165478_YAP1_dic.png",
    "HAC1_182391": RAW_DIR / "182391_HAC1_dic.png",
}


def to_uint8_rgb(gray16: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray16, (1, 99))
    scaled = np.clip((gray16 - lo) / max(hi - lo, 1e-6), 0, 1)
    gray8 = (scaled * 255).astype(np.uint8)
    return np.stack([gray8] * 3, axis=-1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'image':15s} {'n_cells':>8s} {'median_area_px':>16s} {'area_p5-p95':>18s}")
    for name, dic_path in DIC_IMAGES.items():
        dic = load_png16(dic_path)
        labels = segment.segment_cells(dic, method="otsu", min_size=30)

        n_cells = int(labels.max())
        if n_cells == 0:
            print(f"{name:15s} {'0':>8s} {'-':>16s} {'-':>18s}")
            continue

        areas = np.array([(labels == i).sum() for i in range(1, n_cells + 1)])
        p5, p95 = np.percentile(areas, (5, 95))
        print(f"{name:15s} {n_cells:8d} {int(np.median(areas)):16d} {f'{p5:.0f}-{p95:.0f}':>18s}")

        overlay = mark_boundaries(to_uint8_rgb(dic), labels, color=(1, 1, 0))
        Image.fromarray((overlay * 255).astype(np.uint8)).save(OUT_DIR / f"{name}_overlay.png")

    print(f"\noverlays written to {OUT_DIR}")


if __name__ == "__main__":
    main()
