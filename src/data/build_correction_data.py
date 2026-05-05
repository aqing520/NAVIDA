"""
Stage 3: Build Correction Training Data

Convert mined error states into training samples that match
the exact format of navida_train_data.jsonl.

Each error state generates one training sample with:
- The model's actual observation (historical frames + current frame)
- The corrected action as the label
"""

import json
import os
import argparse
from tqdm import tqdm


# Action constants (matching prepare_training_data.py)
FORWARD_DISTANCE = 25
TURN_ANGLE = 15


def action_id_to_str(action_id):
    """Convert action ID to string representation."""
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


VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


def build_correction_sample(error_record, rollout_dir):
    """
    Build a single correction training sample from an error record.

    The sample format matches navida_train_data.jsonl exactly.
    """
    instruction = error_record.get("instruction", "")
    corrected_action = error_record.get("corrected_action")
    context_frames = error_record.get("context_frames", [])
    episode_id = error_record.get("episode_id")
    error_type = error_record.get("error_type", "unknown")

    if corrected_action is None:
        return None

    # Build the prompt
    prompt = VLN_PROMPT_TEMPLATE.format(instruction)

    # Build image paths (absolute paths)
    image_paths = []
    for frame_path in context_frames:
        # frame_path is already relative to project root (e.g. data/rollout_train_baseline/frames/...)
        abs_path = os.path.abspath(frame_path)
        if os.path.exists(abs_path):
            image_paths.append(abs_path)

    if not image_paths:
        return None

    # Build the corrected action string
    corrected_action_str = action_id_to_str(corrected_action)

    # Build the training sample
    sample = {
        "system": "You are a helpful assistant.",
        "conversations": [
            {
                "from": "user",
                "value": prompt,
                "image": image_paths
            },
            {
                "from": "assistant",
                "value": corrected_action_str
            }
        ],
        "task type": "vln",
        "episode_id": str(episode_id),
        "correction_type": error_type,
    }

    return sample


def build_correction_data(errors_path, rollout_dir, output_path,
                          error_types=None, max_samples=None):
    """
    Build correction training data from error records.

    Args:
        errors_path: path to errors.jsonl from mine_errors.py
        rollout_dir: base directory of rollout results
        output_path: path to write correction_train.jsonl
        error_types: list of error types to include (if None, include all)
        max_samples: maximum number of samples to generate

    Returns:
        dict with statistics
    """
    # Load error records
    errors = []
    with open(errors_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                errors.append(json.loads(line))

    print(f"Loaded {len(errors)} error records")

    # Filter by error type if specified
    if error_types:
        errors = [e for e in errors if e.get("error_type") in error_types]
        print(f"Filtered to {len(errors)} errors of types {error_types}")

    # Limit samples if specified
    if max_samples and len(errors) > max_samples:
        errors = errors[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Build correction samples
    samples = []
    skipped = 0
    for error in tqdm(errors, desc="Building correction data"):
        sample = build_correction_sample(error, rollout_dir)
        if sample is not None:
            samples.append(sample)
        else:
            skipped += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {
        "total_errors": len(errors),
        "samples_generated": len(samples),
        "skipped": skipped,
        "output_path": output_path,
    }

    # Print statistics
    print("\n=== Correction Data Statistics ===")
    print(f"Total errors: {stats['total_errors']}")
    print(f"Samples generated: {stats['samples_generated']}")
    print(f"Skipped (no valid frames): {stats['skipped']}")
    print(f"Output: {output_path}")

    # Save stats
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build correction training data from error records")
    parser.add_argument("--errors", type=str, required=True,
                        help="Path to errors.jsonl from mine_errors.py")
    parser.add_argument("--rollout-dir", type=str, required=True,
                        help="Base directory of rollout results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for correction_train.jsonl")
    parser.add_argument("--error-types", nargs="+", default=None,
                        help="Error types to include (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to generate")
    args = parser.parse_args()

    build_correction_data(
        args.errors,
        args.rollout_dir,
        args.output,
        args.error_types,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
