"""Stage 4 crosstalk-correction validation: run correct.estimate_crosstalk_coefficient
on every sample in data/synthetic/labels.csv (built by ichnos_image.synthesize)
and compare the estimate against the known ground-truth bleed_green_to_red,
broken down by true_ratio_red_green -- since the estimator's "low quantile =
crosstalk floor" assumption should degrade as more pixels carry real red
signal at every green level.

Usage: python scripts/validate_crosstalk.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ichnos_image import correct
from ichnos_image.synthesize import load_png16

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def main():
    labels = pd.read_csv(DATA_DIR / "labels.csv")

    results = []
    for row in labels.itertuples():
        green = load_png16(DATA_DIR / row.green_file)
        red = load_png16(DATA_DIR / row.red_file)
        try:
            estimated_bleed, _intercept = correct.estimate_crosstalk_coefficient(green, red)
        except ValueError as exc:
            estimated_bleed = np.nan
            print(f"  ! {row.sample_id}: {exc}")

        results.append(
            dict(
                sample_id=row.sample_id,
                base_image=row.base_image,
                true_bleed=row.bleed_green_to_red,
                true_ratio=row.true_ratio_red_green,
                estimated_bleed=estimated_bleed,
                abs_error=abs(estimated_bleed - row.bleed_green_to_red),
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "crosstalk_validation.csv", index=False)

    print(f"\n{len(df)} samples, {df['estimated_bleed'].isna().sum()} failed to fit\n")
    print("Mean absolute error in estimated bleed, by true_ratio_red_green (higher = harder):")
    print(df.groupby("true_ratio")["abs_error"].agg(["mean", "max", "count"]).round(4))

    print("\nMean absolute error, by base image:")
    print(df.groupby("base_image")["abs_error"].agg(["mean", "max"]).round(4))

    overall_mae = df["abs_error"].mean()
    print(f"\noverall MAE: {overall_mae:.4f}")
    print(f"results written to {DATA_DIR / 'crosstalk_validation.csv'}")


if __name__ == "__main__":
    main()
