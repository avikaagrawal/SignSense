import csv
import hashlib
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "wlasl" / "selected_videos.csv"
OUTPUT_DIR = ROOT / "data" / "wlasl" / "raw_videos"
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.csv"

MAX_RETRIES = 3

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


def is_valid_file(path):
    if not path:
        return False

    file_path = Path(path)

    return (
        file_path.exists()
        and file_path.is_file()
        and file_path.stat().st_size > 0
    )


def is_transient_error(error):
    message = str(error).lower()

    temporary_indicators = [
        "timed out",
        "timeout",
        "winerror 10060",
        "errno 11001",
        "getaddrinfo failed",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "temporary failure",
        "remote end closed",
        "name resolution",
        "429",
        "500",
        "502",
        "503",
        "504",
        "522",
        "524",
    ]

    return any(item in message for item in temporary_indicators)


def download_direct(url, key):
    parsed = urlparse(url)
    extension = Path(parsed.path).suffix.lower()

    if not extension or len(extension) > 6:
        extension = ".mp4"

    destination = OUTPUT_DIR / f"{key}{extension}"
    temporary_file = OUTPUT_DIR / f"{key}.download"

    if is_valid_file(destination):
        return str(destination)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with open(temporary_file, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

        if not is_valid_file(temporary_file):
            raise RuntimeError("Downloaded file is empty")

        temporary_file.replace(destination)

        return str(destination)

    finally:
        if temporary_file.exists():
            temporary_file.unlink()


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

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-2000:] or result.stdout[-2000:]
        )

    files = list(OUTPUT_DIR.glob(f"{key}.*"))

    files = [
        file for file in files
        if file.suffix.lower() not in [".part", ".ytdl", ".download"]
        and is_valid_file(file)
    ]

    if not files:
        raise RuntimeError("No downloaded video found")

    return str(files[0])


def download_with_retry(url, key, source_type):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            if source_type == "youtube":
                path = download_youtube(url, key)
            else:
                path = download_direct(url, key)

            return path, ""

        except Exception as error:
            last_error = error

            print(f"Attempt {attempt}/{MAX_RETRIES} failed:")
            print(str(error)[:500])

            if not is_transient_error(error):
                print("Permanent-looking error. Not retrying.")
                break

            if attempt < MAX_RETRIES:
                wait_time = attempt * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    return "", str(last_error)


def main():
    df = pd.read_csv(CSV_PATH)

    sources = df.drop_duplicates(subset=["url"])

    # Load previous manifest without discarding successful downloads.
    previous = {}

    if MANIFEST_PATH.exists():
        old_df = pd.read_csv(MANIFEST_PATH).fillna("")

        for _, row in old_df.iterrows():
            previous[row["url"]] = row.to_dict()

    results = []

    print(f"Total samples: {len(df)}")
    print(f"Unique sources: {len(sources)}")
    print("-" * 60)

    for index, row in enumerate(
        sources.itertuples(index=False),
        start=1
    ):
        url = str(row.url)
        key = get_source_key(url)
        source_type = get_source_type(url)

        sample_count = int((df["url"] == url).sum())

        old = previous.get(url, {})

        old_path = str(old.get("local_path", ""))
        old_status = str(old.get("status", ""))
        old_error = str(old.get("error", ""))

        print(
            f"\n[{index}/{len(sources)}] "
            f"{source_type}: {url}"
        )

        # Preserve successful existing downloads.
        if old_status == "downloaded" and is_valid_file(old_path):
            print("Already downloaded. Preserving existing file.")

            results.append({
                "source_key": key,
                "url": url,
                "source_type": source_type,
                "local_path": old_path,
                "sample_count": sample_count,
                "status": "downloaded",
                "error": "",
            })

            continue

        # Check whether a valid file exists even if the
        # previous manifest did not record it correctly.
        existing_files = [
            file for file in OUTPUT_DIR.glob(f"{key}.*")
            if file.suffix.lower() not in [".part", ".ytdl", ".download"]
            and is_valid_file(file)
        ]

        if existing_files:
            path = str(existing_files[0])

            print("Found existing video file. Preserving it.")

            results.append({
                "source_key": key,
                "url": url,
                "source_type": source_type,
                "local_path": path,
                "sample_count": sample_count,
                "status": "downloaded",
                "error": "",
            })

            continue

        # NEW: Do not retry previously identified permanent failures.
        if (
            old_status == "failed"
            and old_error
            and not is_transient_error(old_error)
        ):
            print("Permanent failure previously recorded. Skipping.")

            results.append({
                "source_key": key,
                "url": url,
                "source_type": source_type,
                "local_path": "",
                "sample_count": sample_count,
                "status": "failed",
                "error": old_error,
            })

            continue

        # Skip SWF files.
        if source_type == "swf":
            print("Skipping SWF file.")

            results.append({
                "source_key": key,
                "url": url,
                "source_type": source_type,
                "local_path": "",
                "sample_count": sample_count,
                "status": "skipped",
                "error": "SWF format not supported by this downloader",
            })

            continue

        print("Starting download...")

        local_path, error = download_with_retry(
            url,
            key,
            source_type
        )

        status = "downloaded" if local_path else "failed"

        print(f"Status: {status}")

        results.append({
            "source_key": key,
            "url": url,
            "source_type": source_type,
            "local_path": local_path,
            "sample_count": sample_count,
            "status": status,
            "error": error,
        })

        # Save progress after every source.
        save_manifest(results)

    save_manifest(results)

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)

    for status in ["downloaded", "failed", "skipped"]:
        count = sum(
            item["status"] == status
            for item in results
        )
        print(f"{status.capitalize()}: {count}")

    print(f"\nManifest saved to: {MANIFEST_PATH}")


def save_manifest(results):
    temporary_manifest = MANIFEST_PATH.with_suffix(".tmp")

    fieldnames = [
        "source_key",
        "url",
        "source_type",
        "local_path",
        "sample_count",
        "status",
        "error",
    ]

    with open(
        temporary_manifest,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    temporary_manifest.replace(MANIFEST_PATH)


if __name__ == "__main__":
    main()