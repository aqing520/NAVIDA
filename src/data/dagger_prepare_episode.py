import argparse
import copy
import json
import os
import random
import re
from typing import Dict, List, Tuple

from tqdm import tqdm


DATASET_DEFAULT_DIR = {
    "r2r": "data/dagger/r2r_dagger_v1",
    "rxr": "data/dagger/rxr_dagger_v1",
}

DATASET_ANNOTATION_PATH = {
    "r2r": "data/sub_dataset/r2r.jsonl",
    "rxr": "data/sub_dataset/streamvln_rxr.jsonl",
}

DEFAULT_PREPARED_SUBDIR = "prepared_daggerv2"
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


def load_reference_actions(path: str) -> Dict[int, List[int]]:
    actions_by_episode: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            actions_by_episode[int(item["episode_id"])] = [int(action) for action in item["actions"]]
    return actions_by_episode


def is_small_step_difference(
    rollout_actions: List[int],
    reference_actions: List[int],
    max_abs_step_diff: int,
    max_step_diff_ratio: float,
) -> bool:
    step_diff = abs(len(rollout_actions) - len(reference_actions))
    if step_diff > max_abs_step_diff:
        return False
    ref_len = max(1, len(reference_actions))
    return (step_diff / ref_len) <= max_step_diff_ratio


def find_expert_spans(action_source: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start_idx = None
    for idx, source in enumerate(action_source):
        if source == "expert":
            if start_idx is None:
                start_idx = idx
        elif start_idx is not None:
            spans.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        spans.append((start_idx, len(action_source)))
    return spans


def build_alt_path_rows(
    dataset: str,
    annotations: List[Dict],
    reference_actions_by_episode: Dict[int, List[int]],
    max_abs_step_diff: int,
    max_step_diff_ratio: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for item in annotations:
        episode_id = int(item["episode_id"])
        reference_actions = reference_actions_by_episode.get(episode_id)
        if reference_actions is None:
            continue
        rollout_actions = [int(action) for action in item["actions"]]
        if item.get("num_rescue_events", 0) != 0:
            continue
        if rollout_actions == reference_actions:
            continue
        if not is_small_step_difference(
            rollout_actions,
            reference_actions,
            max_abs_step_diff=max_abs_step_diff,
            max_step_diff_ratio=max_step_diff_ratio,
        ):
            continue

        instruction = item["instructions"]
        if dataset == "r2r" and isinstance(instruction, list) and instruction:
            instruction = instruction[0]
        rows.append(
            {
                "episode_id": episode_id,
                "video_id": item["video_id"],
                "instruction": instruction,
                "actions": rollout_actions,
                "segment_type": "alt_path",
                "segment_id": f"{episode_id}_alt_path",
                "frame_start_idx": 0,
                "frame_end_idx": len(rollout_actions),
                "num_rescue_events": 0,
                "segment_len": len(rollout_actions),
                "step_diff_vs_ref": abs(len(rollout_actions) - len(reference_actions)),
            }
        )
    return rows


def build_correction_rows(dataset: str, annotations: List[Dict], min_correction_len: int) -> List[Dict]:
    rows: List[Dict] = []
    for item in annotations:
        action_source = list(item.get("action_source", []))
        if not action_source:
            continue
        actions = [int(action) for action in item["actions"]]
        instruction = item["instructions"]
        if dataset == "r2r" and isinstance(instruction, list) and instruction:
            instruction = instruction[0]
        for span_idx, (start_idx, end_idx) in enumerate(find_expert_spans(action_source)):
            if end_idx - start_idx < min_correction_len:
                continue
            segment_actions = actions[start_idx:end_idx]
            if not segment_actions:
                continue
            rows.append(
                {
                    "episode_id": int(item["episode_id"]),
                    "video_id": item["video_id"],
                    "instruction": instruction,
                    "actions": segment_actions,
                    "segment_type": "correction",
                    "segment_id": f"{item['episode_id']}_correction_{span_idx}",
                    "frame_start_idx": start_idx,
                    "frame_end_idx": end_idx,
                    "num_rescue_events": int(item.get("num_rescue_events", 0)),
                    "segment_len": len(segment_actions),
                    "step_diff_vs_ref": None,
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
        if not actions:
            continue

        instructions = episode_item["instruction"]
        instruction_list = instructions if isinstance(instructions, list) else [instructions]
        if task_type == "idm":
            instruction_list = instruction_list[:1]

        episode_image_path = os.path.join(image_path, str(video_id))
        episode_image_list = os.listdir(episode_image_path)
        episode_image_list = sorted(episode_image_list, key=lambda x: int(split_method(x)))
        episode_image_list = [os.path.join(episode_image_path, image) for image in episode_image_list]

        frame_start_idx = int(episode_item.get("frame_start_idx", 0))
        frame_end_idx = int(episode_item.get("frame_end_idx", len(actions)))
        episode_image_list = episode_image_list[frame_start_idx : frame_end_idx + 1]
        if len(episode_image_list) != len(actions) + 1:
            episode_image_list.append(episode_image_list[-1])
        assert len(episode_image_list) == len(actions) + 1

        for instruction in instruction_list:
            tmp_data = {
                "system": system_prompt,
                "conversations": [],
                "action_history": [],
                "episode_id": str(episode_id),
                "task type": task_type,
                "segment_type": episode_item["segment_type"],
                "segment_id": episode_item["segment_id"],
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
                if len(episode_image_list) < 2:
                    continue
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


def build_training_files(
    dataset: str,
    input_dir: str,
    prepared_dir: str,
    max_abs_step_diff: int,
    max_step_diff_ratio: float,
    min_correction_len: int,
) -> None:
    kept_annotations_path = os.path.join(input_dir, "kept_annotations.json")
    annotations = load_kept_annotations(kept_annotations_path)
    reference_actions_by_episode = load_reference_actions(DATASET_ANNOTATION_PATH[dataset])

    alt_path_rows = build_alt_path_rows(
        dataset,
        annotations,
        reference_actions_by_episode,
        max_abs_step_diff=max_abs_step_diff,
        max_step_diff_ratio=max_step_diff_ratio,
    )
    correction_rows = build_correction_rows(
        dataset,
        annotations,
        min_correction_len=min_correction_len,
    )
    sub_dataset = alt_path_rows + correction_rows
    random.shuffle(sub_dataset)

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

    segment_type_counts: Dict[str, int] = {}
    for row in sub_dataset:
        segment_type = row["segment_type"]
        segment_type_counts[segment_type] = segment_type_counts.get(segment_type, 0) + 1

    summary = {
        "dataset": dataset,
        "input_dir": input_dir,
        "num_input_annotations": len(annotations),
        "num_alt_path_rows": len(alt_path_rows),
        "num_correction_rows": len(correction_rows),
        "segment_type_counts": segment_type_counts,
        "max_abs_step_diff": max_abs_step_diff,
        "max_step_diff_ratio": max_step_diff_ratio,
        "min_correction_len": min_correction_len,
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
    parser.add_argument("--prepared-subdir", type=str, default=DEFAULT_PREPARED_SUBDIR)
    parser.add_argument("--max-abs-step-diff", type=int, default=6)
    parser.add_argument("--max-step-diff-ratio", type=float, default=0.2)
    parser.add_argument("--min-correction-len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.max_abs_step_diff < 0:
        parser.error("--max-abs-step-diff must be >= 0")
    if args.max_step_diff_ratio < 0.0:
        parser.error("--max-step-diff-ratio must be >= 0")
    if args.min_correction_len < 1:
        parser.error("--min-correction-len must be >= 1")
    return args


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir or DATASET_DEFAULT_DIR[args.dataset])
    prepared_dir = os.path.join(input_dir, args.prepared_subdir)
    os.makedirs(prepared_dir, exist_ok=True)
    random.seed(args.seed)
    build_training_files(
        args.dataset,
        input_dir,
        prepared_dir,
        max_abs_step_diff=args.max_abs_step_diff,
        max_step_diff_ratio=args.max_step_diff_ratio,
        min_correction_len=args.min_correction_len,
    )


if __name__ == "__main__":
    main()
