"""
Build stability / anchor data from rollout traces.

- Success episodes: keep decision-step chunks produced by the baseline model.
- Failed episodes: keep only pre-deviation decision steps where model and oracle
  agree, so the normal prefix can act as a policy anchor.
"""

import os
import json
import gzip
import argparse
import re
import random
from collections import Counter

import numpy as np
from tqdm import tqdm


FORWARD_DISTANCE = 25
TURN_ANGLE = 15

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


def action_id_to_str(action_id):
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return f"forward {FORWARD_DISTANCE} cm"
    if action_id == 2:
        return f"turn left {TURN_ANGLE} degree"
    if action_id == 3:
        return f"turn right {TURN_ANGLE} degree"
    raise ValueError(f"Invalid action ID: {action_id}")


def combine(action1, action2):
    idx = action1.rfind(", ")
    subaction0 = action1[: idx + 1] + " " if action1[: idx + 1] != "" else action1[: idx + 1]
    subaction1 = action1[idx + 1 :]
    match1 = int(re.search(r"-?\d+", subaction1).group())
    match2 = int(re.search(r"-?\d+", action2).group())
    if "forward" in subaction1:
        if match1 + match2 <= 3 * FORWARD_DISTANCE:
            return f"{subaction0}forward {match1 + match2} cm"
        return None
    if "turn left" in subaction1:
        if match1 + match2 <= 3 * TURN_ANGLE:
            return f"{subaction0}turn left {match1 + match2} degree"
        return None
    if "turn right" in subaction1:
        if match1 + match2 <= 3 * TURN_ANGLE:
            return f"{subaction0}turn right {match1 + match2} degree"
        return None
    raise ValueError(f"Invalid action: {action1}")


def action_ids_to_chunk_exact(action_ids):
    if not action_ids:
        return None
    first_action = action_ids[0]
    if first_action == 0:
        return "stop"

    chunk = action_id_to_str(first_action)
    for next_action in action_ids[1:]:
        if next_action == 0:
            break
        next_action_str = action_id_to_str(next_action)
        merged = combine(chunk, next_action_str)
        if merged is not None:
            chunk = merged
        elif chunk.count(",") < 2:
            chunk += ", " + next_action_str
        else:
            break
    return chunk


def interpolate_path(waypoints, interval=0.25):
    if len(waypoints) < 2:
        return np.array(waypoints)
    waypoints = [np.array(p[:3]) for p in waypoints]
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        seg = waypoints[i + 1] - waypoints[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        n_points = max(1, int(seg_len / interval))
        for j in range(1, n_points + 1):
            t = j / n_points
            dense.append(waypoints[i] + t * seg)
    return np.array(dense)


def point_to_path_distance(point, dense_path):
    point = np.array(point[:3])
    diffs = dense_path - point
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.min(dists))


def load_reference_paths(annotations_path):
    with gzip.open(annotations_path, "rt") as f:
        data = json.load(f)
    return {
        str(ep["episode_id"]): ep.get("reference_path", [])
        for ep in data["episodes"]
    }


def load_traces(rollout_dir):
    import glob

    traces = {}
    for trace_file in glob.glob(os.path.join(rollout_dir, "traces", "*.jsonl")):
        episode_id = os.path.splitext(os.path.basename(trace_file))[0]
        records = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if records:
            traces[episode_id] = records
    return traces


def load_summaries(rollout_dir):
    import glob

    summaries = {}
    for path in glob.glob(os.path.join(rollout_dir, "summaries", "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        summaries[str(item["episode_id"])] = item
    return summaries


def get_context_frames(records, idx, max_history):
    start = max(0, idx - max_history + 1)
    frames = []
    for i in range(start, idx + 1):
        frame_path = os.path.abspath(records[i].get("frame_path", ""))
        if os.path.exists(frame_path):
            frames.append(frame_path)
    return frames


def find_deviation_idx(records, dense_path, threshold):
    for idx, record in enumerate(records):
        if not record.get("is_decision_step", False):
            continue
        agent_pos = record.get("agent_position")
        if agent_pos is None:
            continue
        if point_to_path_distance(agent_pos, dense_path) > threshold:
            return idx
    return None


def build_sample(record, context_frames, label, source):
    instruction = record.get("instruction", "")
    if not instruction or not context_frames or not label:
        return None
    return {
        "system": "You are a helpful assistant.",
        "conversations": [
            {
                "from": "user",
                "value": VLN_PROMPT_TEMPLATE.format(instruction),
                "image": context_frames,
            },
            {
                "from": "assistant",
                "value": label,
            },
        ],
        "action_history": [],
        "task type": "vln",
        "episode_id": str(record.get("episode_id")),
        "source": source,
    }


def extract_label(record):
    action_ids = record.get("parsed_action_ids")
    if action_ids:
        valid = [int(x) for x in action_ids if x in (0, 1, 2, 3)]
        if valid:
            return action_ids_to_chunk_exact(valid)
    model_action = record.get("model_action")
    if model_action in (0, 1, 2, 3):
        return action_id_to_str(int(model_action))
    return None


def build_anchor_dataset(
    rollout_dir,
    annotations_path,
    output_path,
    threshold=1.5,
    max_history=8,
    max_success_samples=None,
    max_prefix_samples=None,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    traces = load_traces(rollout_dir)
    summaries = load_summaries(rollout_dir)
    ref_paths = load_reference_paths(annotations_path)
    dense_paths = {
        episode_id: interpolate_path(ref_path, interval=0.25)
        for episode_id, ref_path in ref_paths.items()
        if ref_path and len(ref_path) >= 2
    }

    success_samples = []
    prefix_samples = []
    reason_counts = Counter()

    success_ids = [
        episode_id for episode_id, summary in summaries.items()
        if summary.get("final_success", 0)
    ]
    failed_ids = [
        episode_id for episode_id, summary in summaries.items()
        if not summary.get("final_success", 0)
    ]

    for episode_id in tqdm(success_ids, desc="Success anchors"):
        records = traces.get(episode_id)
        if not records:
            reason_counts["success_missing_trace"] += 1
            continue
        for idx, record in enumerate(records):
            if not record.get("is_decision_step", False):
                continue
            label = extract_label(record)
            frames = get_context_frames(records, idx, max_history)
            sample = build_sample(record, frames, label, "anchor_success_model")
            if sample is not None:
                success_samples.append(sample)

    for episode_id in tqdm(failed_ids, desc="Failed prefix anchors"):
        records = traces.get(episode_id)
        dense_path = dense_paths.get(episode_id)
        if not records:
            reason_counts["failed_missing_trace"] += 1
            continue
        if dense_path is None:
            reason_counts["failed_missing_reference_path"] += 1
            continue
        deviation_idx = find_deviation_idx(records, dense_path, threshold)
        if deviation_idx is None:
            reason_counts["failed_no_deviation"] += 1
            continue
        for idx, record in enumerate(records[:deviation_idx]):
            if not record.get("is_decision_step", False):
                continue
            if record.get("model_action") != record.get("oracle_action"):
                continue
            label = extract_label(record)
            frames = get_context_frames(records, idx, max_history)
            sample = build_sample(record, frames, label, "anchor_failed_prefix_agree")
            if sample is not None:
                prefix_samples.append(sample)

    random.shuffle(success_samples)
    random.shuffle(prefix_samples)
    if max_success_samples is not None:
        success_samples = success_samples[:max_success_samples]
    if max_prefix_samples is not None:
        prefix_samples = prefix_samples[:max_prefix_samples]

    samples = success_samples + prefix_samples
    random.shuffle(samples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {
        "rollout_dir": rollout_dir,
        "annotations_path": annotations_path,
        "threshold": threshold,
        "max_history": max_history,
        "success_samples": len(success_samples),
        "prefix_samples": len(prefix_samples),
        "total_samples": len(samples),
        "skip_reasons": dict(reason_counts),
        "source_distribution": {
            "anchor_success_model": len(success_samples),
            "anchor_failed_prefix_agree": len(prefix_samples),
        },
    }
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Built {len(samples)} anchor samples")
    print(f"Output: {output_path}")
    print(f"Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build stability anchor data from rollout traces.")
    parser.add_argument("--rollout-dir", type=str, required=True)
    parser.add_argument("--annotations-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-success-samples", type=int, default=None)
    parser.add_argument("--max-prefix-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_anchor_dataset(
        rollout_dir=args.rollout_dir,
        annotations_path=args.annotations_path,
        output_path=args.output,
        threshold=args.threshold,
        max_history=args.max_history,
        max_success_samples=args.max_success_samples,
        max_prefix_samples=args.max_prefix_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
