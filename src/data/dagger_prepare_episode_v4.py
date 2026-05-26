import argparse
import copy
import json
import os
import random
import re
from collections import Counter
from typing import Dict, Iterable, List

from tqdm import tqdm


FORWARD_DISTANCE = 25
TURN_ANGLE = 15
DEFAULT_PREPARED_SUBDIR = "prepared_daggerv4"


def action_id_to_str(action_id: int) -> str:
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return f"forward {FORWARD_DISTANCE} cm"
    if action_id == 2:
        return f"turn left {TURN_ANGLE} degree"
    if action_id == 3:
        return f"turn right {TURN_ANGLE} degree"
    raise ValueError(f"Invalid action ID: {action_id}")


def combine(action1: str, action2: str):
    idx = action1.rfind(", ")
    prefix = action1[: idx + 1] + " " if action1[: idx + 1] else action1[: idx + 1]
    last_action = action1[idx + 1 :]
    match1 = re.search(r"-?\d+", last_action)
    match2 = re.search(r"-?\d+", action2)
    if match1 is None or match2 is None:
        return None
    value1 = int(match1.group())
    value2 = int(match2.group())
    if "forward" in last_action:
        if value1 + value2 <= 3 * FORWARD_DISTANCE:
            return f"{prefix}forward {value1 + value2} cm"
        return None
    if "turn left" in last_action:
        if value1 + value2 <= 3 * TURN_ANGLE:
            return f"{prefix}turn left {value1 + value2} degree"
        return None
    if "turn right" in last_action:
        if value1 + value2 <= 3 * TURN_ANGLE:
            return f"{prefix}turn right {value1 + value2} degree"
        return None
    raise ValueError(f"Invalid action: {action1}")


def image_sort_key(name: str) -> int:
    matches = re.findall(r"\d+", name)
    if not matches:
        raise ValueError(f"Cannot parse frame index from image name: {name}")
    return int(matches[-1])


def load_kept_annotations(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_single_type(
    annotations: List[Dict],
    input_dir: str,
    system_prompt: str,
    prompt_template: str,
    task_type: str,
) -> List[Dict]:
    rows: List[Dict] = []
    for episode_item in tqdm(annotations):
        actions = [int(action) for action in episode_item["actions"]]
        if not actions:
            continue

        instructions = episode_item["instructions"]
        instruction_list = instructions if isinstance(instructions, list) else [instructions]
        if task_type == "idm":
            instruction_list = instruction_list[:1]

        episode_image_dir = os.path.join(input_dir, episode_item["image_dir"])
        episode_image_list = sorted(os.listdir(episode_image_dir), key=image_sort_key)
        episode_image_list = [os.path.join(episode_image_dir, image) for image in episode_image_list]
        if len(episode_image_list) != len(actions) + 1:
            episode_image_list.append(episode_image_list[-1])
        assert len(episode_image_list) == len(actions) + 1

        for instruction in instruction_list:
            sample = {
                "system": system_prompt,
                "conversations": [],
                "action_history": [],
                "episode_id": str(episode_item["episode_id"]),
                "task type": task_type,
            }
            if task_type == "vln":
                formatted_instruction = prompt_template.format(instruction)
                sample["conversations"].append(
                    {"from": "user", "value": formatted_instruction, "image": [episode_image_list[0]]}
                )
                sample["conversations"].append(
                    {"from": "assistant", "value": action_id_to_str(actions[0])}
                )
                last_action = actions[0]
                pending_rgb_list: List[str] = []
                for idx in range(1, len(actions)):
                    prob = random.random()
                    action_text = action_id_to_str(actions[idx])
                    merged_value = combine(sample["conversations"][-1]["value"], action_text)
                    if prob <= 0.7 and actions[idx] == last_action and merged_value is not None:
                        sample["conversations"][-1]["value"] = merged_value
                        pending_rgb_list.append(episode_image_list[idx])
                    else:
                        count = sample["conversations"][-1]["value"].count(",")
                        if count < 2:
                            sample["conversations"][-1]["value"] += ", " + action_text
                            pending_rgb_list.append(episode_image_list[idx])
                        else:
                            rows.append(copy.deepcopy(sample))
                            sample["action_history"].append(sample["conversations"][1]["value"])
                            while pending_rgb_list:
                                sample["conversations"][0]["image"].append(pending_rgb_list.pop(0))
                            sample["conversations"][0]["image"].append(episode_image_list[idx])
                            sample["conversations"][1]["value"] = action_text
                    last_action = actions[idx]
                rows.append(copy.deepcopy(sample))
            elif task_type == "idm":
                if len(episode_image_list) < 2:
                    continue
                sample["conversations"].append(
                    {"from": "user", "value": prompt_template, "image": [episode_image_list[0], episode_image_list[1]]}
                )
                sample["conversations"].append(
                    {"from": "assistant", "value": action_id_to_str(actions[0])}
                )
                last_action = actions[0]
                for idx in range(1, len(actions)):
                    prob = random.random()
                    action_text = action_id_to_str(actions[idx])
                    merged_value = combine(sample["conversations"][-1]["value"], action_text)
                    if prob <= 0.7 and actions[idx] == last_action and merged_value is not None:
                        sample["conversations"][-1]["value"] = merged_value
                        sample["conversations"][-2]["image"][-1] = episode_image_list[idx + 1]
                    else:
                        count = sample["conversations"][-1]["value"].count(",")
                        if count < 2:
                            sample["conversations"][-1]["value"] += ", " + action_text
                            sample["conversations"][-2]["image"][-1] = episode_image_list[idx + 1]
                        else:
                            rows.append(copy.deepcopy(sample))
                            sample["action_history"].append(sample["conversations"][1]["value"])
                            sample["conversations"][0]["image"] = [episode_image_list[idx], episode_image_list[idx + 1]]
                            sample["conversations"][1]["value"] = action_text
                    last_action = actions[idx]
                if sample["conversations"][1]["value"] != "stop":
                    rows.append(copy.deepcopy(sample))
            else:
                raise NotImplementedError(task_type)
    return rows


def build_training_files(input_dir: str, prepared_dir: str, dataset: str) -> None:
    annotations = load_kept_annotations(os.path.join(input_dir, "kept_annotations.json"))

    system_prompt = "You are a helpful assistant."
    vln_prompt_template = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video of historical observations and an image of the current observation. "
        "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
        "which could involve turning left or right by a specific degree or moving forward a certain distance."
    )
    idm_prompt_template = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given an image of current view and an image of the goal view. "
        "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "
        "which could involve turning left or right by a specific degree or moving forward a certain distance."
    )

    vln_rows = process_single_type(annotations, input_dir, system_prompt, vln_prompt_template, "vln")
    idm_rows = process_single_type(annotations, input_dir, system_prompt, idm_prompt_template, "idm")

    write_jsonl(os.path.join(prepared_dir, "train_vln.jsonl"), vln_rows)
    write_jsonl(os.path.join(prepared_dir, "train_idm.jsonl"), idm_rows)
    write_jsonl(os.path.join(prepared_dir, "train_full.jsonl"), vln_rows + idm_rows)

    tier_counts = Counter(item.get("rescue_tier", "unknown") for item in annotations)
    action_lengths = [len(item["actions"]) for item in annotations if item.get("actions")]
    summary = {
        "dataset": dataset,
        "input_dir": input_dir,
        "num_input_annotations": len(annotations),
        "rescue_tier_counts": dict(tier_counts),
        "avg_episode_actions": (sum(action_lengths) / len(action_lengths)) if action_lengths else 0.0,
        "num_vln_rows": len(vln_rows),
        "num_idm_rows": len(idm_rows),
        "num_full_rows": len(vln_rows) + len(idm_rows),
    }
    with open(os.path.join(prepared_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["r2r", "rxr"], required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prepared-subdir", default=DEFAULT_PREPARED_SUBDIR)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    input_dir = os.path.abspath(args.input_dir)
    prepared_dir = os.path.join(input_dir, args.prepared_subdir)
    os.makedirs(prepared_dir, exist_ok=True)
    build_training_files(input_dir=input_dir, prepared_dir=prepared_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
