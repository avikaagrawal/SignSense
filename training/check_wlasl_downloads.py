import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

samples = pd.read_csv(
    ROOT / "data/wlasl/selected_videos.csv"
)

downloads = pd.read_csv(
    ROOT / "data/wlasl/raw_videos/source_manifest.csv"
)

# Map each source URL to its download result
download_map = downloads.set_index("url").to_dict("index")

def check_sample(row):
    result = download_map.get(row["url"])

    if not result or result["status"] != "downloaded":
        return False

    path = Path(result["local_path"])

    return path.exists() and path.stat().st_size > 0


samples["available"] = samples.apply(check_sample, axis=1)

print("\nCOVERAGE BY WORD")
print("=" * 55)

summary = samples.groupby("gloss").agg(
    total=("video_id", "count"),
    available=("available", "sum")
)

summary["missing"] = summary["total"] - summary["available"]
summary["coverage"] = (
    summary["available"] / summary["total"] * 100
).round(1)

print(summary.to_string())

print("\nCOVERAGE BY DATASET SPLIT")
print("=" * 55)

print(
    samples.groupby("split")["available"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nTOTAL AVAILABLE SAMPLES:", samples["available"].sum())
print("TOTAL MISSING SAMPLES:", (~samples["available"]).sum())

output = ROOT / "data/wlasl/downloaded_samples.csv"
samples.to_csv(output, index=False)

print(f"\nDetailed report saved to: {output}")