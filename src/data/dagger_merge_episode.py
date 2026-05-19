import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


def json_dump(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for src_path in sorted(src_dir.rglob("*")):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src_dir)
        hardlink_or_copy(src_path, dst_dir / rel)


def build_summary(dataset: str, metrics_all: List[Dict], metrics_kept: List[Dict], source_shards: List[str]) -> Dict:
    return {
        "dataset": dataset,
        "source_shards": source_shards,
        "num_episodes": len(metrics_all),
        "num_kept": len(metrics_kept),
        "keep_rate": safe_mean([float(row["kept"]) for row in metrics_all]),
        "success_rate": safe_mean([float(row["success"]) for row in metrics_all]),
        "avg_distance_to_goal": safe_mean([float(row["distance_to_goal"]) for row in metrics_all]),
        "avg_pl": safe_mean([float(row["pl"]) for row in metrics_all]),
        "avg_num_rescue_events": safe_mean([float(row["num_rescue_events"]) for row in metrics_all]),
        "avg_kept_pl": safe_mean([float(row["pl"]) for row in metrics_kept]),
        "avg_kept_distance_to_goal": safe_mean([float(row["distance_to_goal"]) for row in metrics_kept]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    shard_dirs = sorted(Path().glob(args.input_glob))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    debug_videos_dir = output_dir / "debug_videos"
    prepared_dir = output_dir / "prepared"
    images_dir.mkdir(parents=True, exist_ok=True)
    debug_videos_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: List[Dict] = []
    metrics_kept: List[Dict] = []
    kept_annotations: List[Dict] = []
    configs: List[Dict] = []
    source_shards: List[str] = []

    for shard_dir in shard_dirs:
        source_shards.append(shard_dir.name)
        with (shard_dir / "config.json").open("r", encoding="utf-8") as f:
            configs.append(json.load(f))
        metrics_all.extend(read_jsonl(shard_dir / "metrics_all.jsonl"))
        metrics_kept.extend(read_jsonl(shard_dir / "metrics_kept.jsonl"))
        with (shard_dir / "kept_annotations.json").open("r", encoding="utf-8") as f:
            kept_annotations.extend(json.load(f))
        mirror_tree(shard_dir / "images", images_dir)
        mirror_tree(shard_dir / "debug_videos", debug_videos_dir)

    metrics_all.sort(key=lambda row: (int(row["episode_id"]), int(row.get("source_episode_index", -1))))
    metrics_kept.sort(key=lambda row: (int(row["episode_id"]), int(row.get("source_episode_index", -1))))
    kept_annotations.sort(key=lambda row: int(row["episode_id"]))

    base_config = dict(configs[0])
    base_config["source_shards"] = source_shards
    base_config["merged_from"] = args.input_glob
    json_dump(base_config, output_dir / "config.json")
    write_jsonl(output_dir / "metrics_all.jsonl", metrics_all)
    write_jsonl(output_dir / "metrics_kept.jsonl", metrics_kept)
    json_dump(kept_annotations, output_dir / "kept_annotations.json")
    json_dump(build_summary(base_config["dataset"], metrics_all, metrics_kept, source_shards), output_dir / "result_summary.json")


if __name__ == "__main__":
    main()
