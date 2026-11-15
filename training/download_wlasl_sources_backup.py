import csv
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "wlasl" / "selected_videos.csv"
OUTPUT_DIR = ROOT / "data" / "wlasl" / "raw_videos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_source_key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_source_type(url):
    lower = url.lower()

    if "youtube.com" in lower or "youtu.be" in lower:
        return "youtube"

    if lower.endswith(".swf"):
        return "swf"

    return "direct"


def download_direct(url, key):
    parsed = urlparse(url)
    extension = Path(parsed.path).suffix.lower()

    if not extension or len(extension) > 6:
        extension = ".mp4"

    destination = OUTPUT_DIR / f"{key}{extension}"

    if destination.exists() and destination.stat().st_size > 0:
        return str(destination)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()

    if not data:
        raise RuntimeError("Downloaded file is empty")

    destination.write_bytes(data)
    return str(destination)


def download_youtube(url, key):
    output_template = str(OUTPUT_DIR / f"{key}.%(ext)s")

    command = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "--output", output_template,
        url,
    ]

    subprocess.run(command, check=True)

    files = list(OUTPUT_DIR.glob(f"{key}.*"))
    files = [
        file for file in files
        if file.suffix not in [".part", ".ytdl"]
    ]

    if not files:
        raise RuntimeError("No downloaded video found")

    return str(files[0])


def main():
    df = pd.read_csv(CSV_PATH)

    # One download per unique URL
    sources = df.drop_duplicates(subset=["url"])

    results = []

    print(f"Total samples: {len(df)}")
    print(f"Unique sources: {len(sources)}")
    print("-" * 50)

    for index, row in sources.iterrows():
        url = row["url"]
        key = get_source_key(url)
        source_type = get_source_type(url)

        sample_count = int((df["url"] == url).sum())

        print(
            f"\n[{len(results) + 1}/{len(sources)}] "
            f"{source_type}: {url}"
        )

        status = "skipped"
        local_path = ""

        try:
            if source_type == "swf":
                print("Skipping SWF: requires separate handling.")

            elif source_type == "youtube":
                local_path = download_youtube(url, key)
                status = "downloaded"

            else:
                local_path = download_direct(url, key)
                status = "downloaded"

            print(f"Status: {status}")

        except Exception as error:
            status = "failed"
            print(f"Download failed: {error}")

        results.append({
            "source_key": key,
            "url": url,
            "source_type": source_type,
            "local_path": local_path,
            "sample_count": sample_count,
            "status": status,
        })

    manifest_path = OUTPUT_DIR / "source_manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_key",
                "url",
                "source_type",
                "local_path",
                "sample_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)

    for status in ["downloaded", "failed", "skipped"]:
        count = sum(r["status"] == status for r in results)
        print(f"{status.capitalize()}: {count}")

    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()