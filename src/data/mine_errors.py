"""
Stage 2: Error Mining for Self-correction Flywheel

Read rollout traces and episode summaries, classify each decision step
into one of four correction types (or "correct").

Error types:
- Type 1: near-goal no-stop (d_goal <= 3.0, model_action != stop)
- Type 2: false-stop (d_goal > 3.0, model_action == stop)
- Type 3: off-track (point_to_path_distance > threshold)
- Type 4: early unstable (step <= 10, model_action != oracle_action)

Exp A: only Type 1
"""

import json
import os
import glob
import argparse
import numpy as np
from collections import defaultdict


def point_to_path_distance(point, path):
    """Minimum perpendicular distance from a 3D point to a polyline path."""
    if not path or len(path) < 2:
        return float('inf')
    min_dist = float('inf')
    point = np.array(point, dtype=np.float64)
    for i in range(len(path) - 1):
        seg_start = np.array(path[i], dtype=np.float64)
        seg_end = np.array(path[i + 1], dtype=np.float64)
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            dist = np.linalg.norm(point - seg_start)
        else:
            t = max(0, min(1, np.dot(point - seg_start, seg_vec) / (seg_len ** 2)))
            proj = seg_start + t * seg_vec
            dist = np.linalg.norm(point - proj)
        min_dist = min(min_dist, dist)
    return min_dist


def load_traces(rollout_dir):
    """Load all trace files and episode summaries."""
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

    summaries = {}
    summary_dir = os.path.join(rollout_dir, "summaries")
    for summary_file in glob.glob(os.path.join(summary_dir, "*.json")):
        episode_id = os.path.splitext(os.path.basename(summary_file))[0]
        with open(summary_file, "r", encoding="utf-8") as f:
            summaries[episode_id] = json.load(f)

    return traces, summaries


def get_context_frames(records, current_idx, max_history=8):
    """Get the last N frame paths leading up to current step."""
    start = max(0, current_idx - max_history + 1)
    return [records[i]["frame_path"] for i in range(start, current_idx + 1)]


def mine_errors(rollout_dir, output_path, goal_radius=3.0,
                offtrack_threshold=1.5, early_steps=10,
                error_types=None):
    """
    Mine error states from rollout traces.

    Args:
        rollout_dir: directory containing traces/ and summaries/
        output_path: path to write errors.jsonl
        goal_radius: threshold for near-goal detection (meters)
        offtrack_threshold: threshold for off-track detection (meters)
        early_steps: step threshold for early unstable detection
        error_types: list of error types to mine, e.g. ["type1"]
                     if None, mine all types
    """
    if error_types is None:
        error_types = ["type1", "type2", "type3", "type4"]

    traces, summaries = load_traces(rollout_dir)
    print(f"Loaded {len(traces)} episodes, {len(summaries)} summaries")

    # Load reference paths from summaries (they store goal_position but not reference_path)
    # We need to load reference_path from the original dataset or from the trace records
    # For now, we use goal_position + agent_position for Type 1/2, and skip Type 3 if no ref_path

    errors = []
    stats = defaultdict(int)

    for episode_id, records in traces.items():
        if not records:
            continue

        summary = summaries.get(episode_id, {})
        goal_position = summary.get("goal_position")
        ref_path = None  # Will be loaded from dataset if needed for Type 3

        for idx, record in enumerate(records):
            stats["total_decision_steps"] += 1

            dtg = record.get("distance_to_goal", float('inf'))
            model_action = record.get("model_action")
            oracle_action = record.get("oracle_action")
            is_decision = record.get("is_decision_step", False)
            step = record.get("step", 0)

            # Only classify decision steps (model output, not pending actions)
            if not is_decision:
                stats["pending_steps"] += 1
                continue

            error_type = None
            corrected_action = None

            # Type 1: near-goal no-stop
            if "type1" in error_types:
                if dtg <= goal_radius and model_action != 0:
                    error_type = "type1_near_goal_no_stop"
                    corrected_action = 0  # stop
                    stats["type1"] += 1

            # Type 2: false-stop
            if error_type is None and "type2" in error_types:
                if dtg > goal_radius and model_action == 0:
                    error_type = "type2_false_stop"
                    corrected_action = oracle_action
                    stats["type2"] += 1

            # Type 3: off-track (requires reference_path)
            if error_type is None and "type3" in error_types:
                agent_pos = record.get("agent_position")
                if agent_pos is not None and ref_path is not None:
                    dist = point_to_path_distance(agent_pos, ref_path)
                    if dist > offtrack_threshold:
                        error_type = "type3_off_track"
                        corrected_action = oracle_action
                        stats["type3"] += 1

            # Type 4: early unstable
            if error_type is None and "type4" in error_types:
                if step <= early_steps and oracle_action is not None and model_action != oracle_action:
                    error_type = "type4_early_unstable"
                    corrected_action = oracle_action
                    stats["type4"] += 1

            if error_type is not None and corrected_action is not None:
                context_frames = get_context_frames(records, idx)
                error_record = {
                    "episode_id": int(episode_id),
                    "step": step,
                    "error_type": error_type,
                    "distance_to_goal": float(dtg),
                    "model_action": int(model_action),
                    "model_raw_output": record.get("model_raw_output"),
                    "oracle_action": int(oracle_action) if oracle_action is not None else None,
                    "corrected_action": int(corrected_action),
                    "agent_position": record.get("agent_position"),
                    "goal_position": goal_position,
                    "instruction": record.get("instruction"),
                    "frame_path": record.get("frame_path"),
                    "context_frames": context_frames,
                }
                errors.append(error_record)

    # Write errors
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    # Print statistics
    print("\n=== Error Mining Statistics ===")
    print(f"Total decision steps: {stats['total_decision_steps']}")
    print(f"Pending steps (skipped): {stats['pending_steps']}")
    print(f"Type 1 (near-goal no-stop): {stats['type1']}")
    print(f"Type 2 (false-stop): {stats['type2']}")
    print(f"Type 3 (off-track): {stats['type3']}")
    print(f"Type 4 (early unstable): {stats['type4']}")
    print(f"Total errors: {len(errors)}")
    print(f"Output: {output_path}")

    # Save stats
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, indent=2)

    return errors, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Mine error states from rollout traces")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing traces/ and summaries/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for errors.jsonl")
    parser.add_argument("--goal-radius", type=float, default=3.0,
                        help="Threshold for near-goal detection (meters)")
    parser.add_argument("--offtrack-threshold", type=float, default=1.5,
                        help="Threshold for off-track detection (meters)")
    parser.add_argument("--early-steps", type=int, default=10,
                        help="Step threshold for early unstable detection")
    parser.add_argument("--error-types", nargs="+", default=["type1"],
                        choices=["type1", "type2", "type3", "type4"],
                        help="Error types to mine")
    args = parser.parse_args()

    mine_errors(
        args.rollout_dir,
        args.output,
        args.goal_radius,
        args.offtrack_threshold,
        args.early_steps,
        args.error_types,
    )


if __name__ == "__main__":
    main()
