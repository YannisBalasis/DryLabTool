"""End-to-end demo: run the full ichnos_image.pipeline (segment -> correct
-> extract -> export) over the real downloaded reference images in
data/raw/, producing one combined experiment CSV -- proof the orchestration
layer works on real pixel data, not just synthetic.

Caveat, printed at the end too: bleed_green_to_red used here is illustrative
(0.05), NOT a real per-session calibration -- these 4 real dual-channel
entries have no matched same-session GFP-only control in the public dataset
(see correct.calibrate_crosstalk_from_control()'s docstring and
project memory for why that matters), so there is nothing to calibrate
against. Each entry is treated as its own "session" since they're unrelated
experiments, not real repeats. None of these entries have a bright-field/DIC
channel (confirmed during Stage 4 investigation), so segmentation runs on
the green channel directly.

Usage: python scripts/run_real_data_demo.py
"""
from pathlib import Path

from ichnos_image import ImageSet, process_experiment
from ichnos_image.synthesize import load_png16

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "segmentation_test" / "real_data_demo_experiment.csv"

ENTRIES = [
    ("210071", "IRE1_NUP49", "210071_IRE1_green.png", "210071_NUP49_red.png", 0.80, 2.00),
    ("201943", "KAR2_NUP49", "201943_KAR2_green.png", "201943_NUP49_red.png", 0.80, 2.00),
    ("200646", "HSP104_NUP49", "200646_HSP104_green.png", "200646_NUP49_red.png", 0.80, 2.00),
    ("210637", "TRX2_NUP49", "210637_TRX2_green.png", "210637_NUP49_red.png", 0.80, 2.00),
]

ILLUSTRATIVE_BLEED = 0.05


def main():
    image_sets = []
    for experiment_id, pair_name, green_file, red_file, exp_g, exp_r in ENTRIES:
        green = load_png16(RAW_DIR / green_file)
        red = load_png16(RAW_DIR / red_file)
        image_sets.append(
            ImageSet(
                green=green,
                red=red,
                bright_field=None,  # no DIC channel available for these entries
                session_id=f"{experiment_id}_{pair_name}",
                timepoint=0,
                acquisition_order=1,
                exposure_ms_green=exp_g,
                exposure_ms_red=exp_r,
                nd_filter_green=0.0,
                nd_filter_red=0.0,
                objective="60x",
                burner_hours=0.0,
                lamp_warmup_minutes=30.0,
            )
        )

    out_path = process_experiment(image_sets, OUT_CSV, bleed_green_to_red=ILLUSTRATIVE_BLEED)

    import pandas as pd

    df = pd.read_csv(out_path)
    print(f"processed {len(image_sets)} real image sets -> {len(df)} cell records")
    print(df.groupby("session_id").agg(
        n_cells=("cell_id", "count"),
        mean_ratio=("ratio_red_green", "mean"),
        qc_pass_frac=("qc_pass", "mean"),
    ))
    print(f"\nwritten to {out_path}")
    print(
        "\nNOTE: bleed_green_to_red=0.05 above is illustrative, not a real "
        "per-session calibration -- these 4 entries have no matched "
        "same-session GFP-only control (different experiments, not repeats "
        "of one session). This demo shows the orchestration mechanics work "
        "end-to-end on real pixel data; it is not a scientifically valid "
        "crosstalk correction for these specific images."
    )


if __name__ == "__main__":
    main()
