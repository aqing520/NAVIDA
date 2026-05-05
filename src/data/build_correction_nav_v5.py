"""
Build CorrectNAV-style correction training data.

Detects deviation points where model action diverges from oracle action,
then generates corrective action chunks from the oracle trajectory.

Key difference from build_correction_data_v2.py:
- Uses deviation detection (model vs oracle action mismatch at decision steps)
- Generates complete corrective action chunks from deviation point
- No "stop" label bias - corrective trajectories naturally contain forward/turn actions
"""

import json
import os
import argparse
import random
import re
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
    """
    Build an action chunk from a sequence of action IDs.
    Same chunking logic as prepare_training_data.py.
    """
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

        # 70% probability to try merging if same type
        prob = random.random()
        if prob <= 0.7 and next_action == last_action:
            merged = combine(chunk, next_action_str)
            if merged is not None:
                chunk = merged
                last_action = next_action
                actions_used += 1
                continue

        # Different type or merge failed: try to append with comma
        comma_count = chunk.count(',')
        if comma_count < 2:
            chunk += ', ' + next_action_str
            last_action = next_action
            actions_used += 1
        else:
            break  # max 3 actions per chunk

    return chunk, actions_used


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


def detect_deviations(records):
    """
    Detect deviation points in a rollout trace.

    A deviation is the first decision step where model_action != oracle_action,
    AND the model was previously following the oracle (or this is the first step).

    Returns list of deviation dicts with deviation_step and context info.
    """
    deviations = []
    was_following = True  # Start assuming model follows oracle

    for i, record in enumerate(records):
        if not record.get("is_decision_step", False):
            continue

        model_action = record.get("model_action")
        oracle_action = record.get("oracle_action")

        if model_action is None or oracle_action is None:
            continue

        if model_action != oracle_action:
            if was_following:
                # This is the first deviation point
                deviations.append({
                    "deviation_step": record["step"],
                    "deviation_idx": i,
                    "model_action": model_action,
                    "oracle_action": oracle_action,
                    "distance_to_goal": record.get("distance_to_goal", 0),
                    "agent_position": record.get("agent_position"),
                })
                was_following = False
        else:
            was_following = True

    return deviations


def build_correction_samples(rollout_dir, max_history=8, max_correction_steps=10):
    """
    Build CorrectNAV-style correction training samples.

    For each deviation point:
    - Get context frames (last max_history frames up to deviation step)
    - Build corrective action chunk from oracle actions starting at deviation
    - Output in expert data format
    """
    traces = load_traces(rollout_dir)
    print(f"Loaded {len(traces)} episode traces")

    samples = []
    skipped = 0
    episodes_with_deviations = 0

    for episode_id, records in tqdm(traces.items(), desc="Building correction samples"):
        instruction = records[0].get("instruction", "") if records else ""
        if not instruction:
            continue

        deviations = detect_deviations(records)
        if not deviations:
            continue

        episodes_with_deviations += 1

        for dev in deviations:
            dev_step = dev["deviation_step"]
            dev_idx = dev["deviation_idx"]

            # Get context frames (last max_history frames up to deviation step)
            start = max(0, dev_idx - max_history + 1)
            context_frames = []
            for i in range(start, dev_idx + 1):
                fp = records[i].get("frame_path", "")
                abs_path = os.path.abspath(fp)
                if os.path.exists(abs_path):
                    context_frames.append(abs_path)
            if not context_frames:
                skipped += 1
                continue

            # Build corrective action chunk from oracle actions starting at deviation
            oracle_actions = []
            for i in range(dev_idx, min(dev_idx + max_correction_steps, len(records))):
                oa = records[i].get("oracle_action")
                if oa is None:
                    break
                oracle_actions.append(oa)
                if oa == 0:  # stop ends the sequence
                    break

            if not oracle_actions:
                skipped += 1
                continue

            # If first oracle action is stop, skip (not a useful correction)
            if oracle_actions[0] == 0:
                skipped += 1
                continue

            chunk_str, _ = build_action_chunk(oracle_actions)
            if chunk_str is None or chunk_str == "stop":
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
                "source": "correctnav_deviation",
                "deviation_step": dev_step,
                "model_action": dev["model_action"],
                "oracle_action": dev["oracle_action"],
                "distance_to_goal": dev["distance_to_goal"],
            }
            samples.append(sample)

    print(f"Episodes with deviations: {episodes_with_deviations}")
    print(f"Built {len(samples)} correction samples, skipped {skipped}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build CorrectNAV-style correction data")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing traces/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for correction training data jsonl")
    parser.add_argument("--max-history", type=int, default=8,
                        help="Max history frames")
    parser.add_argument("--max-correction-steps", type=int, default=10,
                        help="Max steps to look ahead for corrective actions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunking")
    args = parser.parse_args()

    random.seed(args.seed)

    samples = build_correction_samples(args.rollout_dir, args.max_history, args.max_correction_steps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "rollout_dir": args.rollout_dir,
        "total_samples": len(samples),
        "seed": args.seed,
        "max_correction_steps": args.max_correction_steps,
    }

    # Analyze action distribution
    action_dist = {}
    for s in samples:
        label = s["conversations"][1]["value"]
        # Extract first action type
        if "stop" in label:
            action_dist["stop"] = action_dist.get("stop", 0) + 1
        elif "forward" in label:
            action_dist["forward"] = action_dist.get("forward", 0) + 1
        elif "turn left" in label:
            action_dist["turn_left"] = action_dist.get("turn_left", 0) + 1
        elif "turn right" in label:
            action_dist["turn_right"] = action_dist.get("turn_right", 0) + 1
    stats["action_distribution"] = action_dist

    # Analyze distance_to_goal distribution
    distances = [s["distance_to_goal"] for s in samples]
    if distances:
        stats["distance_to_goal"] = {
            "mean": sum(distances) / len(distances),
            "min": min(distances),
            "max": max(distances),
            "lt_3m": sum(1 for d in distances if d < 3.0),
            "lt_5m": sum(1 for d in distances if d < 5.0),
            "gte_5m": sum(1 for d in distances if d >= 5.0),
        }

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOutput: {args.output}")
    print(f"Stats: {stats_path}")
    print(f"Action distribution: {action_dist}")
    if distances:
        print(f"Distance to goal: mean={stats['distance_to_goal']['mean']:.1f}m, "
              f"<3m: {stats['distance_to_goal']['lt_3m']}, "
              f"<5m: {stats['distance_to_goal']['lt_5m']}, "
              f">=5m: {stats['distance_to_goal']['gte_5m']}")


if __name__ == "__main__":
    main()
