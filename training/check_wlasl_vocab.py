import json

from pathlib import Path

DATA_PATH = Path("data/wlasl/WLASL_v0.3.json")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

targets = [
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
    "fast"
]

available = {item["gloss"].lower(): item for item in data}

print("\nTOTAL WORD CLASSES:", len(data))
print("\nVOCABULARY CHECK\n")

for word in targets:

    if word in available:
        item = available[word]

        print(
            f"[FOUND] {word.upper()} "
            f"- {len(item['instances'])} videos"
        )

    else:
        print(f"[NOT FOUND] {word.upper()}")