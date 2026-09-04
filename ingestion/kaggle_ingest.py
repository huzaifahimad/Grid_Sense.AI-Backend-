"""
GridSense AI — Kaggle Ingestion (fallback / bootstrap source)

Use this to get a pipeline running TODAY while ENTSO-E API access is pending.
Downloads a Kaggle dataset, validates it, and normalizes it into the same
schema the ENTSO-E client produces — so switching sources later requires
no changes to features/models/api.

Requires a free Kaggle API token: kaggle.com -> Account -> Create New API Token
Place the downloaded kaggle.json at ~/.kaggle/kaggle.json

Usage:
    python -m ingestion.kaggle_ingest --dataset ziya07/smart-grid-real-time-load-monitoring-dataset
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_RAW_DIR

try:
    import kaggle
except ImportError:
    raise ImportError("Missing dependency. Run: pip install kaggle --break-system-packages")


TARGET_SCHEMA = ["timestamp", "load_mw", "zone", "source"]


def download_dataset(dataset_slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {dataset_slug} -> {dest}")
    kaggle.api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
    return dest


def normalize_to_schema(df: pd.DataFrame, zone_label: str,
                         timestamp_col: str, load_col: str) -> pd.DataFrame:
    """
    Map whatever columns the raw Kaggle CSV has onto GridSense's standard schema.
    You WILL need to inspect the raw file first and adjust timestamp_col/load_col —
    Kaggle datasets are not standardized, unlike ENTSO-E/NASA POWER.
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    out["load_mw"] = pd.to_numeric(df[load_col], errors="coerce")
    out["zone"] = zone_label
    out["source"] = "kaggle"

    n_before = len(out)
    out = out.dropna(subset=["timestamp", "load_mw"])
    out = out.drop_duplicates(subset=["timestamp"])
    n_after = len(out)
    if n_after < n_before:
        print(f"[kaggle]   WARNING: dropped {n_before - n_after} rows during normalization "
              f"(bad timestamps/loads/dupes) — inspect raw file if this is a large fraction")

    return out[TARGET_SCHEMA]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. owner/dataset-name")
    parser.add_argument("--csv-file", required=True, help="Filename inside the dataset to load")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--load-col", default="load")
    parser.add_argument("--zone-label", default="kaggle_zone_1")
    args = parser.parse_args()

    raw_dir = DATA_RAW_DIR / "kaggle_raw"
    download_dataset(args.dataset, raw_dir)

    raw_csv_path = raw_dir / args.csv_file
    raw_df = pd.read_csv(raw_csv_path)
    print(f"[kaggle] raw columns found: {list(raw_df.columns)}")

    clean_df = normalize_to_schema(raw_df, args.zone_label, args.timestamp_col, args.load_col)

    out_path = DATA_RAW_DIR / "kaggle_load.parquet"
    clean_df.to_parquet(out_path, index=False)
    print(f"[kaggle] wrote {len(clean_df)} rows -> {out_path}")
