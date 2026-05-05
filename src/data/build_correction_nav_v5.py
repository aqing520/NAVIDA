"""
Build CorrectNAV-style correction training data (v2).

Geometric deviation detection:
- Load reference_path from episode annotations
- Interpolate to dense path T'_g
- Compute h_t = min distance(agent_position, T'_g) at each step
- Find first h_t > threshold S as deviation point
- Use oracle actions from deviation point as corrective label

Only uses FAILED episodes for correction data.
"""

import json
import os
import gzip
import argparse
import random
import re
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
    elif action_id == 1:
        return f"forward {FORWARD_DISTANCE} cm"
    elif action_id == 2:
        return f"turn left {TURN_ANGLE} degree"
    elif action_id == 3:
        return f"turn right {TURN_ANGLE} degree"
    else:
        raise ValueError(f"Invalid action ID: {action_id}")


def combine(action1, action2):
    """Merge two consecutive same-type actions."""
    idx = action1.rfind(', ')
    subaction0 = action1[:idx+1]+' ' if action1[:idx+1] != '' else action1[:idx+1]
    subaction1 = action1[idx+1:]
    match1 = re.search(r'-?\d+', subaction1)
    match1 = int(match1.group())
    match2 = re.search(r'-?\d+', action2)
    match2 = int(match2.group())
    if "forward" in subaction1:
        if match1+match2 <= 3*FORWARD_DISTANCE:
            return f"{subaction0}forward {match1+match2} cm"
        else:
            return None
    elif "turn left" in subaction1:
        if match1+match2 <= 3*TURN_ANGLE:
            return f"{subaction0}turn left {match1+match2} degree"
        else:
            return None
    elif "turn right" in subaction1:
        if match1+match2 <= 3*TURN_ANGLE:
            return f"{subaction0}turn right {match1+match2} degree"
        else:
            return None
    else:
        raise ValueError(f"Invalid action: {action1}")


def build_action_chunk(action_ids):
    """Build an action chunk from a sequence of action IDs."""
    if not action_ids:
        return None, 0

    first_action = action_ids[0]
    if first_action == 0:
        return "stop", 1

    chunk = action_id_to_str(first_action)
    last_action = first_action
    actions_used = 1

    for i in range(1, len(action_ids)):
        next_action = action_ids[i]
        if next_action == 0:
            break

        next_action_str = action_id_to_str(next_action)

        prob = random.random()
        if prob <= 0.7 and next_action == last_action:
            merged = combine(chunk, next_action_str)
            if merged is not None:
                chunk = merged
                last_action = next_action
                actions_used += 1
                continue

        comma_count = chunk.count(',')
        if comma_count < 2:
            chunk += ', ' + next_action_str
            last_action = next_action
            actions_used += 1
        else:
            break

    return chunk, actions_used


def interpolate_path(waypoints, interval=0.25):
    """
    Interpolate a path from waypoints with given interval (meters).
    Returns dense path as numpy array of shape (N, 3).
    """
    if len(waypoints) < 2:
        return np.array(waypoints)

    waypoints = [np.array(p[:3]) for p in waypoints]
    dense = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        seg = waypoints[i+1] - waypoints[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        n_points = max(1, int(seg_len / interval))
        for j in range(1, n_points + 1):
            t = j / n_points
            point = waypoints[i] + t * seg
            dense.append(point)

    return np.array(dense)


def point_to_path_distance(point, dense_path):
    """Compute minimum distance from a point to a dense path."""
    point = np.array(point[:3])
    diffs = dense_path - point
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.min(dists))


def load_reference_paths(annotations_path):
    """Load reference_path for all episodes from annotations."""
    with gzip.open(annotations_path, 'rt') as f:
        data = json.load(f)
    ref_paths = {}
    for ep in data['episodes']:
        ref_paths[str(ep['episode_id'])] = ep.get('reference_path', [])
    return ref_paths


def load_traces(rollout_dir):
    """Load all trace files."""
    import glob
    traces = {}
    trace_dir = os.path.join(rollout_dir, "traces")
    bad_files = 0
    for trace_file in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        episode_id = os.path.splitext(os.path.basename(trace_file))[0]
        records = []
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if records:
                traces[episode_id] = records
        except Exception:
            bad_files += 1
            continue
    if bad_files > 0:
        print(f"Warning: skipped {bad_files} files with read errors")
    return traces


def load_summaries(rollout_dir):
    """Load episode summaries to determine success/fail."""
    import glob
    summaries = {}
    summary_dir = os.path.join(rollout_dir, "summaries")
    for f in glob.glob(os.path.join(summary_dir, "*.json")):
        with open(f) as fh:
            d = json.load(fh)
            summaries[str(d['episode_id'])] = d
    return summaries


def build_correction_samples(rollout_dir, annotations_path, threshold=1.5,
                             max_history=8, max_correction_steps=10):
    """
    Build correction samples using geometric deviation detection.

    Only from failed episodes. Deviation = first step where
    distance(agent_position, reference_path) > threshold.
    """
    # Load data
    ref_paths = load_reference_paths(annotations_path)
    print(f"Loaded reference paths for {len(ref_paths)} episodes")

    traces = load_traces(rollout_dir)
    print(f"Loaded {len(traces)} episode traces")

    summaries = load_summaries(rollout_dir)
    print(f"Loaded {len(summaries)} episode summaries")

    # Filter to failed episodes only
    failed_episodes = {eid for eid, s in summaries.items()
                       if not s.get('final_success', 0)}
    print(f"Failed episodes: {len(failed_episodes)}")

    # Interpolate reference paths
    dense_paths = {}
    for eid, rp in ref_paths.items():
        if rp and len(rp) >= 2:
            dense_paths[eid] = interpolate_path(rp, interval=0.25)
    print(f"Interpolated {len(dense_paths)} reference paths")

    samples = []
    skipped = 0
    no_ref_path = 0

    for episode_id in tqdm(failed_episodes, desc="Processing failed episodes"):
        records = traces.get(episode_id, [])
        if not records:
            skipped += 1
            continue

        dense_path = dense_paths.get(episode_id)
        if dense_path is None:
            no_ref_path += 1
            continue

        instruction = records[0].get("instruction", "")
        if not instruction:
            continue

        # Compute deviation distance at each decision step
        deviation_step = None
        deviation_idx = None
        for i, record in enumerate(records):
            if not record.get("is_decision_step", False):
                continue

            agent_pos = record.get("agent_position")
            if agent_pos is None:
                continue

            h_t = point_to_path_distance(agent_pos, dense_path)
            if h_t > threshold:
                deviation_step = record["step"]
                deviation_idx = i
                break

        if deviation_idx is None:
            skipped += 1
            continue

        # Get context frames
        start = max(0, deviation_idx - max_history + 1)
        context_frames = []
        for i in range(start, deviation_idx + 1):
            fp = records[i].get("frame_path", "")
            abs_path = os.path.abspath(fp)
            if os.path.exists(abs_path):
                context_frames.append(abs_path)
        if not context_frames:
            skipped += 1
            continue

        # Build corrective action chunk from oracle actions
        oracle_actions = []
        for i in range(deviation_idx, min(deviation_idx + max_correction_steps, len(records))):
            oa = records[i].get("oracle_action")
            if oa is None:
                break
            oracle_actions.append(oa)
            if oa == 0:
                break

        if not oracle_actions:
            skipped += 1
            continue

        chunk_str, _ = build_action_chunk(oracle_actions)
        if chunk_str is None:
            skipped += 1
            continue

        prompt = VLN_PROMPT_TEMPLATE.format(instruction)

        sample = {
            "system": "You are a helpful assistant.",
            "conversations": [
                {
                    "from": "user",
                    "value": prompt,
                    "image": context_frames
                },
                {
                    "from": "assistant",
                    "value": chunk_str
                }
            ],
            "action_history": [],
            "task type": "vln",
            "episode_id": str(episode_id),
            "source": "correctnav_geometric_deviation",
            "deviation_step": deviation_step,
            "deviation_distance": float(point_to_path_distance(
                records[deviation_idx].get("agent_position", [0,0,0]), dense_path)),
            "distance_to_goal": records[deviation_idx].get("distance_to_goal", 0),
        }
        samples.append(sample)

    print(f"\nResults:")
    print(f"  Total failed episodes: {len(failed_episodes)}")
    print(f"  No reference path: {no_ref_path}")
    print(f"  Skipped (no frames, etc): {skipped}")
    print(f"  Built samples: {len(samples)}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build CorrectNAV-style correction data (geometric deviation)")
    parser.add_argument("--rollout-dir", type=str, required=True)
    parser.add_argument("--annotations-path", type=str, required=True,
                        help="Path to train.json.gz with reference_path")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Deviation threshold in meters")
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-correction-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    samples = build_correction_samples(
        args.rollout_dir, args.annotations_path,
        args.threshold, args.max_history, args.max_correction_steps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "rollout_dir": args.rollout_dir,
        "annotations_path": args.annotations_path,
        "total_samples": len(samples),
        "threshold": args.threshold,
        "seed": args.seed,
    }

    action_dist = {}
    for s in samples:
        label = s["conversations"][1]["value"]
        if "stop" in label:
            action_dist["stop"] = action_dist.get("stop", 0) + 1
        elif "forward" in label:
            action_dist["forward"] = action_dist.get("forward", 0) + 1
        elif "turn left" in label:
            action_dist["turn_left"] = action_dist.get("turn_left", 0) + 1
        elif "turn right" in label:
            action_dist["turn_right"] = action_dist.get("turn_right", 0) + 1
    stats["action_distribution"] = action_dist

    first_actions = {}
    for s in samples:
        label = s["conversations"][1]["value"]
        first = label.split(',')[0].strip().split()[0]
        first_actions[first] = first_actions.get(first, 0) + 1
    stats["first_action_distribution"] = first_actions

    distances = [s["deviation_distance"] for s in samples]
    if distances:
        stats["deviation_distance"] = {
            "mean": sum(distances) / len(distances),
            "min": min(distances),
            "max": max(distances),
        }

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOutput: {args.output}")
    print(f"Stats: {stats_path}")
    print(f"Action distribution: {action_dist}")
    print(f"First action distribution: {first_actions}")
    if distances:
        print(f"Deviation distance: mean={stats['deviation_distance']['mean']:.2f}m")


if __name__ == "__main__":
    main()
