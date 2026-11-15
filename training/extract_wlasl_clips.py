import re
import cv2
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SAMPLES_CSV = ROOT / "data/wlasl/selected_videos.csv"
SOURCE_MANIFEST = ROOT / "data/wlasl/raw_videos/source_manifest.csv"

OUTPUT_DIR = ROOT / "data/wlasl/clips"
REPORT_PATH = ROOT / "data/wlasl/extracted_clips.csv"


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))


def extract_clip(row, video_path):
    video_id = str(row.video_id)
    gloss = safe_name(row.gloss)
    split = str(row.split)

    start_frame = int(row.frame_start)
    end_frame = int(row.frame_end)

    if start_frame < 1:
        return "invalid_start_frame", ""

    if end_frame != -1 and end_frame < start_frame:
        return "invalid_frame_range", ""

    output_folder = OUTPUT_DIR / split / gloss
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{video_id}.mp4"

    if output_path.exists() and output_path.stat().st_size > 0:
        return "already_extracted", str(output_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return "unreadable_video", ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        cap.release()
        return "invalid_video_metadata", ""

    # WLASL frame_start is 1-indexed.
    first_frame = start_frame - 1

    # frame_end is inclusive. -1 means the final frame.
    last_frame = total_frames - 1 if end_frame == -1 else end_frame - 1

    if first_frame >= total_frames:
        cap.release()
        return "start_frame_out_of_range", ""

    last_frame = min(last_frame, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        return "invalid_video_dimensions", ""

    temporary_path = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temporary_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        return "writer_error", ""

    extracted_frames = 0

    for frame_number in range(first_frame, last_frame + 1):
        success, frame = cap.read()

        if not success:
            break

        writer.write(frame)
        extracted_frames += 1

    cap.release()
    writer.release()

    if temporary_path.exists() and extracted_frames > 0:
        temporary_path.replace(output_path)
        return "extracted", str(output_path)

    if temporary_path.exists():
        temporary_path.unlink()

    return "extraction_failed", ""


def main():
    samples = pd.read_csv(SAMPLES_CSV)
    sources = pd.read_csv(SOURCE_MANIFEST).fillna("")

    # Match each sample URL with its downloaded source.
    source_map = {}

    for _, source in sources.iterrows():
        if source["status"] == "downloaded":
            source_map[source["url"]] = source["local_path"]

    results = []

    print("Total WLASL samples:", len(samples))
    print("Downloaded source files:", len(source_map))
    print("-" * 55)

    for index, row in enumerate(
        samples.itertuples(index=False),
        start=1
    ):
        video_path = source_map.get(row.url, "")

        print(
            f"[{index}/{len(samples)}] "
            f"{row.gloss} | {row.video_id}"
        )

        if not video_path or not Path(video_path).exists():
            status = "source_missing"
            output_path = ""

        else:
            status, output_path = extract_clip(row, video_path)

        print("Status:", status)

        results.append({
            "video_id": row.video_id,
            "gloss": row.gloss,
            "split": row.split,
            "signer_id": row.signer_id,
            "source_path": video_path,
            "clip_path": output_path,
            "status": status,
        })

        # Save progress after each sample.
        pd.DataFrame(results).to_csv(REPORT_PATH, index=False)

    report = pd.DataFrame(results)

    print("\n" + "=" * 55)
    print("EXTRACTION SUMMARY")
    print("=" * 55)

    print(report["status"].value_counts().to_string())

    print("\nResults by word:")
    print(
        report[report["status"].isin(
            ["extracted", "already_extracted"]
        )]
        .groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()