"""Turn the real, measured cellpose-on-CPU timing into a "how long will my
actual batch take" estimate -- to answer "do we need GPU access before
starting full processing?" with real numbers instead of guessing.

All per-image timings below are measured, not estimated: run on the real
179997/TRX2 DIC reference image (535x512), cellpose==4.2.1.1, CPU-only,
model weights already cached locally (no download counted). Real Olympus
images may differ in size/cell density -- rerun this with a real image once
available and update PER_IMAGE_SECONDS.

Usage: python scripts/estimate_cellpose_throughput.py
"""
MODEL_LOAD_SECONDS_ONE_TIME = 96  # cpsam_v2, first load per process (cached after)

PER_IMAGE_SECONDS = {
    "resize_factor=1.0 (full accuracy)": 110,
    "resize_factor=0.5 (~95% of cells kept)": 42,
    "resize_factor=0.25 (~82% of cells kept)": 19,
}

# a practical "still fine unattended overnight" ceiling
OVERNIGHT_BUDGET_HOURS = 10


def main():
    print("Assumes one process loads the model once, then segments N images in sequence.\n")
    batch_sizes = [10, 50, 100, 300, 1000]

    col_width = 34
    header = f"{'N images':>10s}" + "".join(f"{name:>{col_width}s}" for name in PER_IMAGE_SECONDS)
    print(header)
    for n in batch_sizes:
        row = f"{n:10d}"
        for name, seconds in PER_IMAGE_SECONDS.items():
            total_hours = (MODEL_LOAD_SECONDS_ONE_TIME + n * seconds) / 3600
            flag = " *needs GPU*" if total_hours > OVERNIGHT_BUDGET_HOURS else ""
            cell = f"{total_hours:.2f}h{flag}"
            row += f"{cell:>{col_width}s}"
        print(row)

    print(
        f"\n'*needs GPU*' = would take longer than a {OVERNIGHT_BUDGET_HOURS}h overnight run on CPU "
        "at resize_factor=1.0 -- consider GPU, or resize_factor=0.5 as a middle ground (95% of "
        "cells retained on the one real image tested, 2.6x faster), before ruling GPU out.\n"
        "This is a CPU-only estimate; cellpose is typically 10-50x faster on GPU per its own docs "
        "(not independently measured here -- no GPU was available in this environment)."
    )


if __name__ == "__main__":
    main()
