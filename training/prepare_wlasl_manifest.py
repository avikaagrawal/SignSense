import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JSON_PATH = ROOT / "data" / "wlasl" / "WLASL_v0.3.json"
OUTPUT_PATH = ROOT / "data" / "wlasl" / "selected_videos.csv"

TARGET_WORDS = [
    "hello",
    "goodbye",
    "help",
    "thank you",
    "please",
    "yes",
    "no",
    "stop",
    "friend",
    "name",
    "what",
    "again",
    "slow",
    "fast",
]

with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

rows = []
seen_ids = set()

for item in data:

    gloss = item["gloss"].lower()

    if gloss not in TARGET_WORDS:
        continue

    for instance in item["instances"]:

        video_id = str(instance["video_id"])

        if video_id in seen_ids:
            continue

        seen_ids.add(video_id)

        rows.append({
            "video_id": video_id,
            "gloss": gloss,
            "url": instance.get("url", ""),
            "split": instance.get("split", ""),
            "signer_id": instance.get("signer_id", ""),
            "frame_start": instance.get("frame_start", ""),
            "frame_end": instance.get("frame_end", ""),
            "fps": instance.get("fps", 25),
        })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nWLASL MANIFEST CREATED")
print("=" * 40)

print("Target words:", len(TARGET_WORDS))
print("Unique videos:", len(rows))

print("\nVideos per word:")

for word in TARGET_WORDS:

    count = sum(
        1 for row in rows
        if row["gloss"] == word
    )

    print(f"{word:12} : {count}")

print("\nVideos per split:")

for split in ["train", "val", "test"]:

    count = sum(
        1 for row in rows
        if row["split"] == split
    )

    print(f"{split:12} : {count}")

print("\nSaved to:", OUTPUT_PATH)