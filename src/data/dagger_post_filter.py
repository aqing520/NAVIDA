import argparse
import json
import os
from typing import Dict, List


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_summary(metrics_all: List[Dict], metrics_kept: List[Dict], base_summary: Dict) -> Dict:
    summary = dict(base_summary)
    summary["num_episodes"] = len(metrics_all)
    summary["num_kept"] = len(metrics_kept)
    summary["keep_rate"] = len(metrics_kept) / len(metrics_all) if metrics_all else 0.0
    summary["success_rate"] = (
        sum(float(item.get("success", 0.0)) for item in metrics_all) / len(metrics_all) if metrics_all else 0.0
    )
    summary["avg_distance_to_goal"] = (
        sum(float(item.get("distance_to_goal", 0.0)) for item in metrics_all) / len(metrics_all) if metrics_all else 0.0
    )
    summary["avg_pl"] = sum(float(item.get("pl", 0.0)) for item in metrics_all) / len(metrics_all) if metrics_all else 0.0
    summary["avg_num_rescue_events"] = (
        sum(int(item.get("num_rescue_events", 0)) for item in metrics_all) / len(metrics_all) if metrics_all else 0.0
    )
    summary["avg_kept_pl"] = (
        sum(float(item.get("pl", 0.0)) for item in metrics_kept) / len(metrics_kept) if metrics_kept else 0.0
    )
    summary["avg_kept_distance_to_goal"] = (
        sum(float(item.get("distance_to_goal", 0.0)) for item in metrics_kept) / len(metrics_kept)
        if metrics_kept
        else 0.0
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    annotations_path = os.path.join(input_dir, "kept_annotations.json")
    metrics_all_path = os.path.join(input_dir, "metrics_all.jsonl")
    metrics_kept_path = os.path.join(input_dir, "metrics_kept.jsonl")
    summary_path = os.path.join(input_dir, "result_summary.json")

    annotations = json.load(open(annotations_path, "r", encoding="utf-8"))
    metrics_all = read_jsonl(metrics_all_path)
    metrics_kept = read_jsonl(metrics_kept_path)
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    terminal_episode_ids = {
        int(item["episode_id"])
        for item in annotations
        if item.get("actions") and item["actions"][-1] == 0
    }

    filtered_annotations = [item for item in annotations if int(item["episode_id"]) in terminal_episode_ids]
    filtered_metrics_kept = [item for item in metrics_kept if int(item["episode_id"]) in terminal_episode_ids]
    filtered_metrics_all = []
    for item in metrics_all:
        episode_id = int(item["episode_id"])
        is_terminal = episode_id in terminal_episode_ids
        if item.get("kept") and not is_terminal:
            item = dict(item)
            item["kept"] = False
            item["kept_reason"] = ""
            item["terminal_stop"] = False
        elif is_terminal and item.get("kept"):
            item = dict(item)
            item["terminal_stop"] = True
        filtered_metrics_all.append(item)

    write_json(annotations_path, filtered_annotations)
    write_jsonl(metrics_kept_path, filtered_metrics_kept)
    write_jsonl(metrics_all_path, filtered_metrics_all)
    write_json(summary_path, build_summary(filtered_metrics_all, filtered_metrics_kept, summary))

    print(
        json.dumps(
            {
                "input_dir": input_dir,
                "original_kept": len(annotations),
                "filtered_kept": len(filtered_annotations),
                "removed_non_terminal": len(annotations) - len(filtered_annotations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
