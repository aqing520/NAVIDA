"""
Build training data from rollout episodes.

Label strategy:
- Successful episodes: use model_action (preserve model's own working strategy)
- Failed episodes: use oracle_action/SPF (correct wrong decisions with expert guidance)
"""

import json
import os
import argparse
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


def build_samples_from_episode(trace_path, is_success):
    """
    Build training samples from a single episode's trace.

    Args:
        trace_path: path to the episode's trace jsonl
        is_success: whether this episode succeeded
            - True: use model_action as label
            - False: use oracle_action as label
    """
    records = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return []

    instruction = records[0].get("instruction", "")
    episode_id = records[0].get("episode_id", 0)

    samples = []
    for idx, record in enumerate(records):
        if not record.get("is_decision_step", False):
            continue

        # Choose label based on episode success
        if is_success:
            action = record.get("model_action")
        else:
            action = record.get("oracle_action")

        if action is None:
            continue

        # Get context frames (last 8 frames up to current step)
        start = max(0, idx - 7)
        context_frames = []
        for i in range(start, idx + 1):
            fp = records[i].get("frame_path", "")
            abs_path = os.path.abspath(fp)
            if os.path.exists(abs_path):
                context_frames.append(abs_path)

        if not context_frames:
            continue

        prompt = VLN_PROMPT_TEMPLATE.format(instruction)
        action_str = action_id_to_str(action)

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
                    "value": action_str
                }
            ],
            "task type": "vln",
            "episode_id": str(episode_id),
            "source": "success_model" if is_success else "fail_oracle",
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Build training data from rollout episodes")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Directory containing traces/ and summaries/")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for training data jsonl")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes to process")
    args = parser.parse_args()

    summaries_dir = os.path.join(args.rollout_dir, "summaries")
    traces_dir = os.path.join(args.rollout_dir, "traces")

    # Collect all episodes with their success status
    success_ids = []
    fail_ids = []
    for fname in os.listdir(summaries_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(summaries_dir, fname)) as f:
            summary = json.load(f)
        episode_id = fname.replace(".json", "")
        if summary.get("final_success") == 1:
            success_ids.append(episode_id)
        else:
            fail_ids.append(episode_id)

    print(f"Successful episodes: {len(success_ids)}")
    print(f"Failed episodes: {len(fail_ids)}")

    # Build samples
    all_samples = []
    success_samples = 0
    fail_samples = 0

    for episode_id in tqdm(success_ids, desc="Success episodes (model_action)"):
        trace_path = os.path.join(traces_dir, f"{episode_id}.jsonl")
        if not os.path.exists(trace_path):
            continue
        samples = build_samples_from_episode(trace_path, is_success=True)
        all_samples.extend(samples)
        success_samples += len(samples)

    for episode_id in tqdm(fail_ids, desc="Failed episodes (oracle_action)"):
        trace_path = os.path.join(traces_dir, f"{episode_id}.jsonl")
        if not os.path.exists(trace_path):
            continue
        samples = build_samples_from_episode(trace_path, is_success=False)
        all_samples.extend(samples)
        fail_samples += len(samples)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  From success (model_action): {success_samples}")
    print(f"  From failure (oracle_action): {fail_samples}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
