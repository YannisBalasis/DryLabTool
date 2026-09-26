"""Stage 2 quantitative segmentation validation: precision/recall/F1 and mean
IoU against KNOWN ground truth, comparing baseline (regular, circular) vs.
"stressed" (irregular boundary + internal vacuole-like hole) synthetic cell
morphology -- the original open question from the team's protocol was
whether segmentation accuracy holds up under stress-induced morphology
change, and this is untestable on the real reference images we have (none
of the downloaded dual-channel entries include a bright-field/DIC channel,
and the one real stress-treated entry, 200646/HSP104, only shows a
GFP-intensity change -- foci -- not a segmentation-relevant shape change).

No real expert-annotated ground truth exists for this project yet; this
uses synthetic ground truth (known cell positions/shapes) instead, which is
honest about being a synthetic proxy, not a replacement for annotating real
images once they exist.

Usage: python scripts/validate_segmentation_quantitative.py
"""
import numpy as np
from scipy import ndimage as ndi

from ichnos_image import segment


def _make_scene(n_cells=25, size=300, seed=0, stressed=False):
    """Synthetic bright-field-like scene with known ground-truth labels.

    baseline: regular circles, slight phase-contrast-like dark rim.
    stressed: irregular (multi-harmonic radius perturbation) boundary, plus
    an internal low-contrast "vacuole" hole -- a stand-in for the
    swollen-vacuole / deformed morphology the team's protocol flagged as a
    stress-response concern.
    """
    rng = np.random.default_rng(seed)
    bf = np.full((size, size), 0.5)
    ground_truth = np.zeros((size, size), dtype=np.int32)

    centers = rng.integers(20, size - 20, size=(n_cells, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    for idx, (cy, cx) in enumerate(centers, start=1):
        base_r = rng.uniform(7, 10)
        theta = np.arctan2(yy - cy, xx - cx)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        if stressed:
            n_harmonics = 3
            perturb = sum(
                rng.uniform(0.08, 0.18) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
                for k in range(2, 2 + n_harmonics)
            )
            boundary = base_r * (1 + perturb)
        else:
            boundary = base_r

        cell = r < boundary
        if not cell.any():
            continue

        ground_truth[cell & (ground_truth == 0)] = idx
        bf[cell] -= 0.28  # phase-contrast-like darker interior
        bf[cell & (r < boundary * 0.35)] = 0.5  # bright interior halo (real phase-contrast optics, not an artifact)

        if stressed:
            vacuole = r < (base_r * 0.4)
            bf[cell & vacuole] += 0.15  # vacuole: less contrast than cytoplasm

        bf[cell] += rng.normal(0, 0.01, size=cell.sum())  # texture, avoids flat interior

    return bf, ground_truth


def _match_and_score(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5):
    gt_ids = [i for i in np.unique(gt_labels) if i != 0]
    pred_ids = [i for i in np.unique(pred_labels) if i != 0]

    ious = []
    matched_gt, matched_pred = set(), set()
    pairs = []
    for gt_id in gt_ids:
        gt_mask = gt_labels == gt_id
        overlapping_pred_ids = np.unique(pred_labels[gt_mask])
        for pred_id in overlapping_pred_ids:
            if pred_id == 0:
                continue
            pred_mask = pred_labels == pred_id
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            iou = inter / union if union else 0.0
            pairs.append((iou, gt_id, pred_id))

    for iou, gt_id, pred_id in sorted(pairs, key=lambda p: -p[0]):
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        if iou >= iou_threshold:
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            ious.append(iou)

    tp = len(matched_gt)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return dict(n_gt=len(gt_ids), n_pred=len(pred_ids), tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, mean_iou=mean_iou)


def main():
    print(f"{'condition':10s} {'n_gt':>5s} {'n_pred':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'precision':>10s} {'recall':>8s} {'F1':>6s} {'mean_IoU':>9s}")
    rows = []
    for stressed, label in [(False, "baseline"), (True, "stressed")]:
        for seed in range(10):
            bf, gt = _make_scene(seed=seed, stressed=stressed)
            pred = segment.segment_cells(bf, method="otsu", min_size=30)
            score = _match_and_score(pred, gt)
            score["condition"] = label
            score["seed"] = seed
            rows.append(score)

        all_scores = [r for r in rows if r["condition"] == label]
        agg = {k: np.mean([s[k] for s in all_scores]) for k in all_scores[0] if k not in ("condition", "seed")}
        print(f"{label:10s} {agg['n_gt']:5.1f} {agg['n_pred']:6.1f} {agg['tp']:4.1f} "
              f"{agg['fp']:4.1f} {agg['fn']:4.1f} {agg['precision']:10.3f} {agg['recall']:8.3f} "
              f"{agg['f1']:6.3f} {agg['mean_iou']:9.3f}")

    import pandas as pd
    from pathlib import Path

    out_path = Path(__file__).resolve().parent.parent / "data" / "segmentation_test" / "quantitative_validation.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nper-seed results written to {out_path}")


if __name__ == "__main__":
    main()
