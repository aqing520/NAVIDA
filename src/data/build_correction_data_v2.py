"""
Build correction training data from rollout error steps.

Aligns with expert data format: action chunking (1-3 actions, same merge logic).
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
    """Merge two consecutive same-type actions (same logic as prepare_training_data.py)."""
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


def build_action_chunk(oracle_actions_from_step):
    """
    Build an action chunk from a sequence of oracle actions (same logic as prepare_training_data.py).

    Args:
        oracle_actions_from_step: list of oracle action IDs starting from the error step

    Returns:
        chunk_str: the action chunk string
        actions_used: number of actions consumed
    """
    if not oracle_actions_from_step:
        return None, 0

    first_action = oracle_actions_from_step[0]
    if first_action == 0:
        return "stop", 1

    chunk = action_id_to_str(first_action)
    last_action = first_action
    actions_used = 1

    for i in range(1, len(oracle_actions_from_step)):
        next_action = oracle_actions_from_step[i]
        if next_action == 0:
            break  # stop ends the chunk

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


def build_correction_samples(errors_path, rollout_dir, max_history=8):
    """
    Build correction training samples from error records.

    For each error step:
    - Get context frames (last max_history frames up to current step)
    - Build action chunk from consecutive oracle actions starting at error step
    - Output in expert data format
    """
    # Load errors
    errors = []
    with open(errors_path, "r") as f:
        for line in f:
            if line.strip():
                errors.append(json.loads(line))
    print(f"Loaded {len(errors)} error records")

    # Load traces
    traces = load_traces(rollout_dir)
    print(f"Loaded {len(traces)} episode traces")

    # Build lookup: (episode_id, step) -> record index in trace
    trace_lookup = {}
    for episode_id, records in traces.items():
        for idx, record in enumerate(records):
            step = record.get("step", idx)
            trace_lookup[(episode_id, step)] = idx

    samples = []
    skipped = 0

    for error in tqdm(errors, desc="Building correction samples"):
        episode_id = error["episode_id"]
        step = error["step"]
        error_type = error["error_type"]
        instruction = error.get("instruction", "")

        # Get trace records for this episode
        records = traces.get(str(episode_id), [])
        if not records:
            skipped += 1
            continue

        # Find current step index in trace
        current_idx = trace_lookup.get((episode_id, step), None)
        if current_idx is None:
            # Try to find by matching step number
            for idx, r in enumerate(records):
                if r.get("step") == step:
                    current_idx = idx
                    break
        if current_idx is None:
            skipped += 1
            continue

        # Get context frames (last max_history frames up to current step)
        start = max(0, current_idx - max_history + 1)
        context_frames = []
        for i in range(start, current_idx + 1):
            fp = records[i].get("frame_path", "")
            abs_path = os.path.abspath(fp)
            if os.path.exists(abs_path):
                context_frames.append(abs_path)
        if not context_frames:
            skipped += 1
            continue

        # Build action chunk from oracle actions
        if error_type == "type1_near_goal_no_stop":
            # Type 1: label is always "stop"
            chunk_str = "stop"
        else:
            # Type 2/4: build chunk from consecutive oracle actions
            oracle_actions = []
            for i in range(current_idx, len(records)):
                oa = records[i].get("oracle_action")
                if oa is None:
                    break
                oracle_actions.append(oa)
                if len(oracle_actions) >= 10:  # look ahead max 10 steps
                    break

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
            "source": f"correction_{error_type}",
        }
        samples.append(sample)

    print(f"Built {len(samples)} correction samples, skipped {skipped}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build correction data (v2, with action chunking)")
    parser.add_argument("--errors-path", type=str, required=True,
                        help="Path to errors.jsonl")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing traces/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for correction training data jsonl")
    parser.add_argument("--max-history", type=int, default=8,
                        help="Max history frames")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunking")
    args = parser.parse_args()

    random.seed(args.seed)

    samples = build_correction_samples(args.errors_path, args.rollout_dir, args.max_history)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "errors_path": args.errors_path,
        "rollout_dir": args.rollout_dir,
        "total_samples": len(samples),
        "by_type": {},
        "seed": args.seed,
    }
    for s in samples:
        src = s.get("source", "unknown")
        stats["by_type"][src] = stats["by_type"].get(src, 0) + 1

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOutput: {args.output}")
    print(f"Stats: {stats_path}")
    for k, v in stats["by_type"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
