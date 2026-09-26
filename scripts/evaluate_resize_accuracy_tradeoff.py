"""How much segmentation accuracy costs a resize_factor speedup -- reuses
validate_segmentation_quantitative.py's synthetic ground truth (method="otsu",
since it doesn't need cellpose installed to test the *mechanism* segment_cells
shares between both methods: downsample -> segment -> upsample labels).
Cellpose is where resize_factor actually saves meaningful wall-clock time
(otsu is already fast); this only answers "what does downsampling cost in
accuracy", not "how much time does it save on cellpose" -- see
estimate_cellpose_throughput.py for the timing side.

Usage: python scripts/evaluate_resize_accuracy_tradeoff.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_segmentation_quantitative import _make_scene, _match_and_score  # noqa: E402

from ichnos_image import segment

RESIZE_FACTORS = [1.0, 0.75, 0.5, 0.35, 0.25]


def main():
    print(f"{'resize_factor':>13s} {'F1':>6s} {'mean_IoU':>9s} {'n_pred':>7s} {'time_s':>8s}")
    for resize_factor in RESIZE_FACTORS:
        scores, times = [], []
        for seed in range(5):
            bf, gt = _make_scene(seed=seed, n_cells=25, size=300)
            t0 = time.time()
            pred = segment.segment_cells(bf, method="otsu", min_size=30, resize_factor=resize_factor)
            times.append(time.time() - t0)
            scores.append(_match_and_score(pred, gt))

        f1 = sum(s["f1"] for s in scores) / len(scores)
        iou = sum(s["mean_iou"] for s in scores) / len(scores)
        n_pred = sum(s["n_pred"] for s in scores) / len(scores)
        mean_time = sum(times) / len(times)
        print(f"{resize_factor:13.2f} {f1:6.3f} {iou:9.3f} {n_pred:7.1f} {mean_time:8.4f}")


if __name__ == "__main__":
    main()
