import json
from pathlib import Path

path = Path("data/wlasl/WLASL_v0.3.json")

with open(path, encoding="utf-8") as f:
    data = json.load(f)

targets = [
    "hello", "goodbye", "help", "thank you",
    "please", "yes", "no", "stop",
    "friend", "name", "what", "again",
    "slow", "fast"
]

for item in data:

    if item["gloss"].lower() in targets:

        print("\nWORD:", item["gloss"])

        for video in item["instances"][:2]:

            print({
                "video_id": video.get("video_id"),
                "url": video.get("url"),
                "split": video.get("split"),
                "frame_start": video.get("frame_start"),
                "frame_end": video.get("frame_end")
            })