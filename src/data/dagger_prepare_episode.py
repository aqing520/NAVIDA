import argparse
import copy
import json
import os
import random
import re
from typing import Dict, List

from tqdm import tqdm


DATASET_DEFAULT_DIR = {
    "r2r": "data/dagger/r2r_dagger_v1",
    "rxr": "data/dagger/rxr_dagger_v1",
}

FORWARD_DISTANCE = 25
TURN_ANGLE = 15


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
    subaction0 = action1[: idx + 1] + " " if action1[: idx + 1] != "" else action1[: idx + 1]
    subaction1 = action1[idx + 1 :]
    match1 = re.search(r"-?\d+", subaction1)
    match2 = re.search(r"-?\d+", action2)
    if match1 is None or match2 is None:
        return None
    match1 = int(match1.group())
    match2 = int(match2.group())
    if "forward" in subaction1:
        return f"{subaction0}forward {match1 + match2} cm" if match1 + match2 <= 3 * FORWARD_DISTANCE else None
    if "turn left" in subaction1:
        return f"{subaction0}turn left {match1 + match2} degree" if match1 + match2 <= 3 * TURN_ANGLE else None
    if "turn right" in subaction1:
        return f"{subaction0}turn right {match1 + match2} degree" if match1 + match2 <= 3 * TURN_ANGLE else None
    raise ValueError(f"Invalid action: {action1}")


def split_method(name: str) -> str:
    return name.split("_")[1].split(".")[0]


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_kept_annotations(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sub_dataset(dataset: str, annotations: List[Dict]) -> List[Dict]:
    rows = []
    for item in annotations:
        instruction = item["instructions"]
        if dataset == "r2r" and isinstance(instruction, list) and instruction:
            instruction = instruction[0]
        rows.append(
            {
                "episode_id": int(item["episode_id"]),
                "video_id": item["video_id"],
                "instruction": instruction,
                "actions": item["actions"],
            }
        )
    return rows


def process_single_type(
    annotation: List[Dict],
    image_path: str,
    system_prompt: str,
    prompt_template: str,
    task_type: str,
) -> List[Dict]:
    data2save: List[Dict] = []
    for episode_item in tqdm(annotation):
        episode_id = episode_item["episode_id"]
        video_id = episode_item["video_id"]
        actions = list(episode_item["actions"])
        # Some retained DAgger episodes end at the rollout step cap instead of
        # emitting a final stop. Append a terminal stop so the sample matches
        # the original training data contract.
        if actions and actions[-1] != 0:
            actions.append(0)

        instructions = episode_item["instruction"]
        instruction_list = instructions if isinstance(instructions, list) else [instructions]
        if task_type == "idm":
            instruction_list = instruction_list[:1]

        for instruction in instruction_list:
            episode_image_path = os.path.join(image_path, str(video_id))
            episode_image_list = os.listdir(episode_image_path)
            episode_image_list = sorted(episode_image_list, key=lambda x: int(split_method(x)))
            episode_image_list = [os.path.join(episode_image_path, image) for image in episode_image_list]
            if len(episode_image_list) != len(actions) + 1:
                episode_image_list.append(episode_image_list[-1])
            assert len(episode_image_list) == len(actions) + 1

            tmp_data = {
                "system": system_prompt,
                "conversations": [],
                "action_history": [],
                "episode_id": str(episode_id),
                "task type": task_type,
            }
            if task_type == "vln":
                formatted_instruction = prompt_template.format(instruction)
                tmp_data["conversations"].append(
                    {"from": "user", "value": formatted_instruction, "image": [episode_image_list[0]]}
                )
                tmp_data["conversations"].append(
                    {"from": "assistant", "value": action_id_to_str(actions[0])}
                )
                last_action = actions[0]
                pending_rgb_list: List[str] = []
                for i in range(1, len(actions)):
                    prob = random.random()
                    action_text = action_id_to_str(actions[i])
                    if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]["value"], action_text) is not None:
                        tmp_data["conversations"][-1]["value"] = combine(tmp_data["conversations"][-1]["value"], action_text)
                        pending_rgb_list.append(episode_image_list[i])
                    else:
                        count = tmp_data["conversations"][-1]["value"].count(",")
                        if count < 2:
                            tmp_data["conversations"][-1]["value"] += ", " + action_text
                            pending_rgb_list.append(episode_image_list[i])
                        else:
                            data2save.append(copy.deepcopy(tmp_data))
                            tmp_data["action_history"].append(tmp_data["conversations"][1]["value"])
                            while pending_rgb_list:
                                tmp_data["conversations"][0]["image"].append(pending_rgb_list.pop(0))
                            tmp_data["conversations"][0]["image"].append(episode_image_list[i])
                            tmp_data["conversations"][1]["value"] = action_text
                    last_action = actions[i]
                data2save.append(copy.deepcopy(tmp_data))
            elif task_type == "idm":
                tmp_data["conversations"].append(
                    {"from": "user", "value": prompt_template, "image": [episode_image_list[0], episode_image_list[1]]}
                )
                tmp_data["conversations"].append(
                    {"from": "assistant", "value": action_id_to_str(actions[0])}
                )
                last_action = actions[0]
                for i in range(1, len(actions)):
                    prob = random.random()
                    action_text = action_id_to_str(actions[i])
                    if prob <= 0.7 and actions[i] == last_action and combine(tmp_data["conversations"][-1]["value"], action_text) is not None:
                        tmp_data["conversations"][-1]["value"] = combine(tmp_data["conversations"][-1]["value"], action_text)
                        tmp_data["conversations"][-2]["image"][-1] = episode_image_list[i + 1]
                    else:
                        count = tmp_data["conversations"][-1]["value"].count(",")
                        if count < 2:
                            tmp_data["conversations"][-1]["value"] += ", " + action_text
                            tmp_data["conversations"][-2]["image"][-1] = episode_image_list[i + 1]
                        else:
                            data2save.append(copy.deepcopy(tmp_data))
                            tmp_data["action_history"].append(tmp_data["conversations"][1]["value"])
                            tmp_data["conversations"][0]["image"] = [episode_image_list[i], episode_image_list[i + 1]]
                            tmp_data["conversations"][1]["value"] = action_text
                    last_action = actions[i]
                if tmp_data["conversations"][1]["value"] != "stop":
                    data2save.append(copy.deepcopy(tmp_data))
            else:
                raise NotImplementedError(task_type)
    return data2save


def build_training_files(dataset: str, input_dir: str, prepared_dir: str) -> None:
    kept_annotations_path = os.path.join(input_dir, "kept_annotations.json")
    annotations = load_kept_annotations(kept_annotations_path)
    sub_dataset = build_sub_dataset(dataset, annotations)
    sub_dataset_path = os.path.join(prepared_dir, "sub_dataset.jsonl")
    write_jsonl(sub_dataset_path, sub_dataset)

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

    image_path = os.path.join(input_dir, "images")
    vln_rows = process_single_type(sub_dataset, image_path, system_prompt, vln_prompt_template, "vln")
    idm_rows = process_single_type(sub_dataset, image_path, system_prompt, idm_prompt_template, "idm")

    train_vln_path = os.path.join(prepared_dir, "train_vln.jsonl")
    train_idm_path = os.path.join(prepared_dir, "train_idm.jsonl")
    train_full_path = os.path.join(prepared_dir, "train_full.jsonl")

    write_jsonl(train_vln_path, vln_rows)
    write_jsonl(train_idm_path, idm_rows)
    write_jsonl(train_full_path, vln_rows + idm_rows)

    summary = {
        "dataset": dataset,
        "num_sub_dataset_rows": len(sub_dataset),
        "num_vln_rows": len(vln_rows),
        "num_idm_rows": len(idm_rows),
        "num_full_rows": len(vln_rows) + len(idm_rows),
    }
    with open(os.path.join(prepared_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["r2r", "rxr"], required=True)
    parser.add_argument("--input-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir or DATASET_DEFAULT_DIR[args.dataset])
    prepared_dir = os.path.join(input_dir, "prepared")
    os.makedirs(prepared_dir, exist_ok=True)
    random.seed(41)
    build_training_files(args.dataset, input_dir, prepared_dir)


if __name__ == "__main__":
    main()
