"""
Stage 4: Mix Original and Correction Datasets

Merge original VLN training data and correction data with configurable ratio.
"""

import json
import os
import random
import argparse
from collections import defaultdict


def count_lines(filepath):
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_jsonl(filepath, max_lines=None):
    """Load records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                records.append(json.loads(line))
                if max_lines and len(records) >= max_lines:
                    break
    return records


def mix_datasets(original_path, correction_path, output_path,
                 original_ratio=0.9, correction_ratio=0.1,
                 seed=42, max_correction_samples=None):
    """
    Mix original and correction datasets with specified ratio.

    Args:
        original_path: path to original training data JSONL
        correction_path: path to correction training data JSONL
        output_path: path to write mixed dataset JSONL
        original_ratio: weight of original data
        correction_ratio: weight of correction data
        seed: random seed for shuffling
        max_correction_samples: cap on correction samples

    Returns:
        dict with statistics
    """
    random.seed(seed)

    # Load original data
    print(f"Loading original data from {original_path}...")
    original_records = load_jsonl(original_path)
    n_original = len(original_records)
    print(f"  Original samples: {n_original}")

    # Load correction data
    print(f"Loading correction data from {correction_path}...")
    correction_records = load_jsonl(correction_path, max_correction_samples)
    n_correction = len(correction_records)
    print(f"  Correction samples: {n_correction}")

    # Compute target correction count based on ratio
    target_correction = int(n_original * correction_ratio / original_ratio)
    if max_correction_samples:
        target_correction = min(target_correction, max_correction_samples)

    # Use all correction data if we have fewer than target
    actual_correction = min(n_correction, target_correction)
    if actual_correction < n_correction:
        correction_records = random.sample(correction_records, actual_correction)

    # Mix: original + correction
    mixed_records = original_records + correction_records
    random.shuffle(mixed_records)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in mixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "original_path": original_path,
        "correction_path": correction_path,
        "output_path": output_path,
        "n_original": n_original,
        "n_correction_available": n_correction,
        "n_correction_used": actual_correction,
        "n_total": len(mixed_records),
        "actual_ratio": f"{n_original}:{actual_correction}",
        "seed": seed,
    }

    print("\n=== Mixed Dataset Statistics ===")
    print(f"Original samples: {n_original}")
    print(f"Correction samples used: {actual_correction}")
    print(f"Total mixed samples: {len(mixed_records)}")
    print(f"Ratio (original:correction): {n_original}:{actual_correction}")
    print(f"Output: {output_path}")

    # Save stats
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Mix original and correction training datasets")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original training data JSONL")
    parser.add_argument("--correction", type=str, required=True,
                        help="Path to correction training data JSONL")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for mixed dataset JSONL")
    parser.add_argument("--original-ratio", type=float, default=0.9,
                        help="Weight of original data (default: 0.9)")
    parser.add_argument("--correction-ratio", type=float, default=0.1,
                        help="Weight of correction data (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--max-correction-samples", type=int, default=None,
                        help="Maximum number of correction samples to use")
    args = parser.parse_args()

    mix_datasets(
        args.original,
        args.correction,
        args.output,
        args.original_ratio,
        args.correction_ratio,
        args.seed,
        args.max_correction_samples,
    )


if __name__ == "__main__":
    main()
