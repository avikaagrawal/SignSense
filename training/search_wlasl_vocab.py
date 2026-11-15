import json
from pathlib import Path

path = Path("data/wlasl/WLASL_v0.3.json")

with open(path, encoding="utf-8") as f:
    data = json.load(f)

keywords = [
    "hello", "goodbye", "help", "thank",
    "please", "yes", "no", "stop",
    "friend", "name", "what", "again",
    "slow", "fast", "welcome"
]

print("WLASL VOCABULARY SEARCH")
print("=" * 45)

for keyword in keywords:
    matches = [
        item for item in data
        if keyword in item["gloss"].lower()
    ]

    print(f"\n{keyword.upper()}:")

    if matches:
        for item in matches:
            print(
                f"  {item['gloss']} "
                f"({len(item['instances'])} videos)"
            )
    else:
        print("  No matching gloss found")