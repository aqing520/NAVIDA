import argparse
import json
import os
import random
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

import cv2
import habitat
import imageio
import numpy as np
import torch
from PIL import Image
from habitat import Env
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import trange
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from habitat_extensions import measures, task  # noqa: F401
from src.eval.eval import (  # noqa: E402
    BASE_PROMPT_TEMPLATE,
    SR_STOP_PROMPT_TEMPLATE,
    STOP_HINT_PROMPT_TEMPLATE,
    encode_image_base64,
)


SYSTEM_PROMPT = "You are a helpful assistant."
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85
RESCUE_ACTION_LEN = 4
SHORT_RESCUE_MAX_EVENTS = 1
RESCUE_WINDOW_BEFORE = 2
RESCUE_WINDOW_AFTER = 2
STOP_WINDOW_ACTIONS = 4
DEFAULT_EPISODE_LENGTH = 60


DATASET_CONFIG = {
    "r2r": {
        "config_path": "config/vln_r2r_train.yaml",
        "annotation_path": "data/sub_dataset/r2r.jsonl",
        "default_output_dir": "data/dagger/r2r_daggerv2",
    },
    "rxr": {
        "config_path": "config/vln_rxr_train.yaml",
        "annotation_path": "data/sub_dataset/streamvln_rxr.jsonl",
        "default_output_dir": "data/dagger/rxr_daggerv2",
    },
}


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def json_dump(data: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_prompt_template(prompt_style: str) -> str:
    if prompt_style == "baseline":
        return BASE_PROMPT_TEMPLATE
    if prompt_style == "stop_hint":
        return STOP_HINT_PROMPT_TEMPLATE
    if prompt_style == "sr_stop":
        return SR_STOP_PROMPT_TEMPLATE
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def extract_instruction_text(episode) -> str:
    instruction = episode.instruction.instruction_text
    if isinstance(instruction, list):
        return instruction[0]
    return instruction


def normalize_instruction(text: str) -> str:
    return text.strip()


def short_scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def load_reference_action_lengths(path: str) -> Dict[int, int]:
    action_lengths: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            action_lengths[int(item["episode_id"])] = len(item["actions"])
    return action_lengths


def load_streamvln_rxr_subset(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            video_id = item["video_id"]
            match = re.search(r"images/([^/_]+)_rxr_", video_id)
            if match is None:
                raise ValueError(f"Failed to parse scene name from video_id: {video_id}")
            instructions = item["instruction"]
            if not isinstance(instructions, list):
                instructions = [instructions]
            rows.append(
                {
                    "scene_short": match.group(1),
                    "instructions": instructions,
                    "instruction_set": frozenset(normalize_instruction(text) for text in instructions),
                    "action_len": len(item["actions"]),
                }
            )
    return rows


def build_instruction_groups(episodes) -> Dict[Tuple[str, str], List[str]]:
    grouped: Dict[Tuple[str, str], OrderedDict[str, None]] = {}
    for episode in episodes:
        trajectory_id = getattr(episode, "trajectory_id", None)
        key = (
            episode.scene_id,
            str(trajectory_id if trajectory_id is not None else episode.episode_id),
        )
        if key not in grouped:
            grouped[key] = OrderedDict()
        grouped[key][extract_instruction_text(episode)] = None
    return {key: list(values.keys()) for key, values in grouped.items()}


def make_debug_frame(
    rgb: np.ndarray,
    top_down_map,
    instruction: str,
    action_text: str,
    action_source: str,
    rescue_remaining: int,
    distance_to_goal: float,
) -> np.ndarray:
    if top_down_map is not None:
        color_map = maps.colorize_draw_agent_and_fit_to_height(top_down_map, rgb.shape[0])
        canvas = np.concatenate((rgb, color_map), axis=1)
    else:
        canvas = rgb

    h, w = canvas.shape[:2]
    panel_height = 140
    out = np.full((h + panel_height, w, 3), 255, dtype=np.uint8)
    out[:h, :w] = canvas

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        instruction,
        f"action={action_text}",
        f"source={action_source} rescue_remaining={rescue_remaining}",
        f"distance_to_goal={distance_to_goal:.3f}",
    ]
    y = h + 24
    for line in lines:
        for wrapped in wrap_text(line, max_chars=110):
            cv2.putText(out, wrapped, (10, y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 20
    return out


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split(" ")
    if not words:
        return [""]
    lines: List[str] = []
    line = words[0]
    for word in words[1:]:
        if len(line) + len(word) + 1 > max_chars:
            lines.append(line)
            line = word
        else:
            line += " " + word
    lines.append(line)
    return lines


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def classify_rescue_tier(num_rescue_events: int) -> str:
    if num_rescue_events <= 0:
        return "clean"
    if num_rescue_events <= SHORT_RESCUE_MAX_EVENTS:
        return "short_rescue"
    return "long_rescue"


def build_training_segments(
    num_actions: int,
    rescue_spans: List[Tuple[int, int]],
    terminal_stop: bool,
    rescue_tier: str,
) -> List[Dict]:
    segments: List[Dict] = []
    seen = set()

    def add_segment(segment_type: str, start_action_idx: int, end_action_idx: int) -> None:
        start_action_idx = max(0, int(start_action_idx))
        end_action_idx = min(num_actions, int(end_action_idx))
        if end_action_idx <= start_action_idx:
            return
        key = (segment_type, start_action_idx, end_action_idx)
        if key in seen:
            return
        seen.add(key)
        segments.append(
            {
                "segment_type": segment_type,
                "start_action_idx": start_action_idx,
                "end_action_idx": end_action_idx,
            }
        )

    if rescue_tier == "clean":
        add_segment("full", 0, num_actions)
    else:
        for idx, (start_action_idx, end_action_idx) in enumerate(rescue_spans):
            add_segment(
                f"rescue_window_{idx}",
                start_action_idx - RESCUE_WINDOW_BEFORE,
                end_action_idx + RESCUE_WINDOW_AFTER,
            )

    if terminal_stop:
        add_segment("stop_window", num_actions - STOP_WINDOW_ACTIONS, num_actions)
    return segments


@dataclass
class EpisodeOutcome:
    episode_id: int
    trajectory_id: Optional[str]
    dataset: str
    video_id: str
    image_dir: str
    instructions: List[str]
    actions: List[int]
    action_source: List[str]
    num_rescue_events: int
    model_success: bool
    rescue_tier: str
    kept_reason: str
    training_segments: List[Dict]


class DAggerRolloutAgent(Agent):
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str],
        forward_distance: int,
        turn_angle: int,
        max_action_history: int,
        resolution_ratio: float,
        num_generations: int,
        prompt_style: str,
        temperature: float,
    ):
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.max_action_history = max_action_history
        self.resolution_ratio = resolution_ratio
        self.num_generations = num_generations
        self.prompt_template = get_prompt_template(prompt_style)

        model_init_kwargs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        }
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_init_kwargs)
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
        self.device = "cuda"
        self.model.to(self.device)
        self.model = self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.processor.image_processor.max_pixels = 501760

        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
            repetition_penalty=1.05,
            num_return_sequences=self.num_generations,
        )

        self.reset()

    def reset(self):
        self.rgb_list: List[Image.Image] = []
        self.conversations = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]

    def act(self, observations):
        raise NotImplementedError("DAggerRolloutAgent uses plan_actions() in the collector loop.")

    def uniform_sample_with_ends(self, data: List[Image.Image], n: int) -> List[Image.Image]:
        if len(data) <= n:
            return data
        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def observe(self, observations) -> Image.Image:
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
        self.rgb_list.append(rgb_img.resize((308, 252)))
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]
        return rgb_img.resize((320, 240))

    def plan_actions(self, observations) -> Tuple[List[int], str]:
        self.conversations = self.conversations[:1]
        content = [{"type": "text", "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations"}]
        if len(self.rgb_list) > 1:
            content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encode_image_base64(item)}",
                    }
                    for item in self.uniform_sample_with_ends(self.rgb_list[:-1], 8)
                ]
            )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}",
                }
            )
        content.append({"type": "text", "text": "and an image of the current observation"})
        content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}",
            }
        )
        item = self.prompt_template.format(observations["instruction"]["text"]).split("current observation")
        content.append({"type": "text", "text": item[1]})
        self.conversations.append({"role": "user", "content": content})

        texts = [self.processor.apply_chat_template(self.conversations, tokenize=False, add_generation_prompt=True)]
        imgs, _ = process_vision_info(self.conversations)
        prompt_inputs = self.processor(
            text=texts,
            images=[imgs],
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs.to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=True,
            )
        input_token_len = prompt_inputs["input_ids"].shape[1]
        navigation = self.processor.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0].strip()
        return self.expand_action_text(navigation), navigation

    def expand_action_text(self, output: str) -> List[int]:
        sub_actions = [item for item in output.strip().split(",") if item.strip()]
        if not sub_actions:
            sub_actions = [output]
        pending: List[int] = []
        for sub_action in sub_actions[:2]:
            action_index, numeric = self.extract_result(sub_action)
            step_actions: List[int] = []
            if action_index == 0:
                step_actions = [0]
            elif action_index == 1:
                step_actions = [1] * min(3, round((numeric or self.forward_distance) / self.forward_distance))
            elif action_index == 2:
                step_actions = [2] * min(3, round((numeric or self.turn_angle) / self.turn_angle))
            elif action_index == 3:
                step_actions = [3] * min(3, round((numeric or self.turn_angle) / self.turn_angle))
            if action_index is None or not step_actions:
                step_actions = [random.randint(1, 3)]
            pending.extend(step_actions)
        return pending

    def extract_result(self, output: str) -> Tuple[Optional[int], Optional[float]]:
        output = output.lower().strip()
        if "stop" in output:
            return 0, None
        if "forward" in output:
            match = next(iter(re.findall(r"-?\d+", output)), None)
            return 1, float(match) if match is not None else float(self.forward_distance)
        if "left" in output:
            match = next(iter(re.findall(r"-?\d+", output)), None)
            return 2, float(match) if match is not None else float(self.turn_angle)
        if "right" in output:
            match = next(iter(re.findall(r"-?\d+", output)), None)
            return 3, float(match) if match is not None else float(self.turn_angle)
        return None, None

    def action_id_to_str(self, action_id: int) -> str:
        if action_id == 0:
            return "stop"
        if action_id == 1:
            return "forward"
        if action_id == 2:
            return "turn left"
        if action_id == 3:
            return "turn right"
        return "unknown"


class EpisodeCollector:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.dataset_cfg = DATASET_CONFIG[self.dataset_name]
        self.output_dir = os.path.abspath(args.output_dir or self.dataset_cfg["default_output_dir"])
        self.images_dir = os.path.join(self.output_dir, "images")
        self.debug_videos_dir = os.path.join(self.output_dir, "debug_videos")
        self.prepared_dir = os.path.join(self.output_dir, "prepared")
        self.annotation_path = self.dataset_cfg["annotation_path"]
        self.ref_action_lengths = (
            load_reference_action_lengths(self.annotation_path) if self.dataset_name == "r2r" else {}
        )
        self.rxr_subset_rows = (
            load_streamvln_rxr_subset(self.annotation_path) if self.dataset_name == "rxr" else None
        )
        self.debug_video_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.debug_videos_dir, exist_ok=True)
        os.makedirs(self.prepared_dir, exist_ok=True)

        self.agent = DAggerRolloutAgent(
            model_path=args.model_path,
            lora_path=args.lora_path,
            forward_distance=args.forward_distance,
            turn_angle=args.turn_angle,
            max_action_history=args.max_action_history,
            resolution_ratio=args.resolution_ratio,
            num_generations=args.num_generations,
            prompt_style=args.prompt_style,
            temperature=args.temperature,
        )

    def build_env(self):
        from habitat_baselines.config.default import get_config

        config = get_config(self.args.exp_config or self.dataset_cfg["config_path"])
        with habitat.config.read_write(config):
            config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
        return config, dataset

    def collect(self):
        config, dataset = self.build_env()
        env = Env(config.habitat, dataset)
        if self.dataset_name == "rxr":
            candidate_indices, instruction_groups = self.build_rxr_subset_selection(env.episodes)
        else:
            candidate_indices = list(range(len(env.episodes)))
            instruction_groups = build_instruction_groups(env.episodes)
        metrics_all: List[Dict] = []
        metrics_kept: List[Dict] = []
        kept_annotations: List[Dict] = []

        config_dump = {
            "dataset": self.dataset_name,
            "exp_config": self.args.exp_config or self.dataset_cfg["config_path"],
            "model_path": self.args.model_path,
            "lora_path": self.args.lora_path,
            "output_dir": self.output_dir,
            "forward_distance": self.args.forward_distance,
            "turn_angle": self.args.turn_angle,
            "max_action_history": self.args.max_action_history,
            "resolution_ratio": self.args.resolution_ratio,
            "prompt_style": self.args.prompt_style,
            "num_generations": self.args.num_generations,
            "temperature": self.args.temperature,
            "midgoal_radius": MIDGOAL_RADIUS,
            "goal_radius": GOAL_RADIUS,
            "relative_pl_threshold": RELATIVE_PATH_LENGTH_THRESHOLD,
            "success_relative_pl_threshold": SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD,
            "rescue_action_len": RESCUE_ACTION_LEN,
            "short_rescue_max_events": SHORT_RESCUE_MAX_EVENTS,
            "rescue_window_before": RESCUE_WINDOW_BEFORE,
            "rescue_window_after": RESCUE_WINDOW_AFTER,
            "stop_window_actions": STOP_WINDOW_ACTIONS,
            "max_episodes": self.args.max_episodes,
            "max_debug_videos": self.args.max_debug_videos,
            "split_num": self.args.split_num,
            "split_id": self.args.split_id,
            "candidate_episode_count": len(candidate_indices),
            "seed": self.args.seed,
        }
        json_dump(config_dump, os.path.join(self.output_dir, "config.json"))

        if self.args.max_episodes is not None:
            candidate_indices = candidate_indices[: self.args.max_episodes]
        selected_indices = [
            episode_idx
            for episode_idx in candidate_indices
            if episode_idx % self.args.split_num == self.args.split_id
        ]

        for episode_idx in trange(len(selected_indices)):
            source_episode_idx = selected_indices[episode_idx]
            episode = env.episodes[source_episode_idx]
            env.current_episode = episode
            env.current_episode.goals[0].radius = MIDGOAL_RADIUS
            outcome, metric_row, debug_frames, kept_rgb_frames = self.rollout_episode(env, instruction_groups)
            metric_row["source_episode_index"] = source_episode_idx
            metrics_all.append(metric_row)
            if metric_row["kept"]:
                metrics_kept.append(metric_row)
                kept_annotations.append(asdict(outcome))
                self.save_episode_images(outcome.video_id, kept_rgb_frames)
            should_save_debug = (
                (metric_row["num_rescue_events"] > 0 or metric_row["kept"])
                and self.debug_video_count < self.args.max_debug_videos
            )
            if should_save_debug:
                self.save_debug_video(outcome.video_id, metric_row["kept"], metric_row["num_rescue_events"] > 0, debug_frames)

        write_jsonl(os.path.join(self.output_dir, "metrics_all.jsonl"), metrics_all)
        write_jsonl(os.path.join(self.output_dir, "metrics_kept.jsonl"), metrics_kept)
        json_dump(kept_annotations, os.path.join(self.output_dir, "kept_annotations.json"))
        json_dump(self.build_summary(metrics_all, metrics_kept), os.path.join(self.output_dir, "result_summary.json"))
        env.close()

    def build_rxr_subset_selection(self, episodes) -> Tuple[List[int], Dict[Tuple[str, str], List[str]]]:
        if self.rxr_subset_rows is None:
            raise RuntimeError("RxR subset rows are not loaded.")

        grouped_by_traj: Dict[Tuple[str, str], Dict] = {}
        for idx, episode in enumerate(episodes):
            scene_short = short_scene_name(episode.scene_id)
            trajectory_id = str(getattr(episode, "trajectory_id", episode.episode_id))
            traj_key = (scene_short, trajectory_id)
            entry = grouped_by_traj.setdefault(
                traj_key,
                {
                    "scene_id": episode.scene_id,
                    "representative_by_instruction": OrderedDict(),
                },
            )
            instruction_text = normalize_instruction(extract_instruction_text(episode))
            if instruction_text not in entry["representative_by_instruction"]:
                entry["representative_by_instruction"][instruction_text] = idx

        grouped_by_instruction_set: Dict[Tuple[str, frozenset], List[Tuple[str, str]]] = {}
        for traj_key, info in grouped_by_traj.items():
            instruction_set = frozenset(info["representative_by_instruction"].keys())
            grouped_by_instruction_set.setdefault((traj_key[0], instruction_set), []).append(traj_key)

        selected_indices: List[int] = []
        instruction_groups: Dict[Tuple[str, str], List[str]] = {}
        matched_traj_keys = set()
        for subset_row in self.rxr_subset_rows:
            lookup_key = (subset_row["scene_short"], subset_row["instruction_set"])
            candidates = grouped_by_instruction_set.get(lookup_key, [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected exactly one Habitat RxR trajectory for subset key {lookup_key}, found {len(candidates)}."
                )
            traj_key = candidates[0]
            if traj_key in matched_traj_keys:
                raise RuntimeError(f"Duplicate subset-to-trajectory mapping for {traj_key}.")
            matched_traj_keys.add(traj_key)

            info = grouped_by_traj[traj_key]
            representative_idx = None
            for instruction_text in subset_row["instructions"]:
                representative_idx = info["representative_by_instruction"].get(normalize_instruction(instruction_text))
                if representative_idx is not None:
                    break
            if representative_idx is None:
                representative_idx = next(iter(info["representative_by_instruction"].values()))

            episode = episodes[representative_idx]
            trajectory_key = (episode.scene_id, str(getattr(episode, "trajectory_id", episode.episode_id)))
            instruction_groups[trajectory_key] = subset_row["instructions"]
            self.ref_action_lengths[int(episode.episode_id)] = subset_row["action_len"]
            selected_indices.append(representative_idx)

        if len(selected_indices) != len(self.rxr_subset_rows):
            raise RuntimeError(
                f"Matched {len(selected_indices)} RxR trajectories, expected {len(self.rxr_subset_rows)}."
            )
        selected_indices.sort()
        return selected_indices, instruction_groups

    def rollout_episode(self, env: Env, instruction_groups: Dict[Tuple[str, str], List[str]]):
        self.agent.reset()
        episode = env.current_episode
        observation = env.reset()
        metrics = env.get_metrics()
        initial_distance = float(metrics["distance_to_goal"])
        scene_id = episode.scene_id
        trajectory_id = getattr(episode, "trajectory_id", None)
        trajectory_key = (
            scene_id,
            str(trajectory_id if trajectory_id is not None else episode.episode_id),
        )
        instructions = instruction_groups.get(trajectory_key, [extract_instruction_text(episode)])
        episode_id = int(episode.episode_id)
        video_id = str(episode_id)
        image_dir = os.path.join("images", video_id)
        ref_path = episode.reference_path
        expert = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        next_waypoint_id = 1 if len(ref_path) > 1 else 0
        model_pending_actions: List[int] = []
        rescue_remaining = 0
        accumulated_error = 0
        accumulated_error_max = 0
        num_rescue_events = 0
        model_success = True
        force_episode_end = False
        debug_frames: List[np.ndarray] = []
        kept_rgb_frames: List[Image.Image] = []
        actions: List[int] = []
        action_source: List[str] = []
        num_model_actions = 0
        num_expert_actions = 0
        last_navigation = "rollout start"
        rescue_spans: List[Tuple[int, int]] = []
        rescue_event_start_idx: Optional[int] = None

        current_rgb = self.agent.observe(observation)
        kept_rgb_frames.append(current_rgb)

        while not env.episode_over:
            top_down_map = metrics.get("top_down_map")
            oracle_action = self.get_oracle_action(expert, ref_path, next_waypoint_id)
            current_action_is_rescue = False

            if rescue_remaining > 0:
                action = oracle_action
                source = "expert"
                rescue_remaining -= 1
                last_navigation = f"expert rescue: {self.agent.action_id_to_str(action)}"
                current_action_is_rescue = True
            else:
                if not model_pending_actions:
                    model_pending_actions, last_navigation = self.agent.plan_actions(observation)
                action = model_pending_actions.pop(0) if model_pending_actions else 0
                source = "model"

            if action != oracle_action:
                accumulated_error += 1
                accumulated_error_max = max(accumulated_error_max, accumulated_error)

            expert, next_waypoint_id, force_episode_end = self.advance_waypoint_if_reached(
                env, expert, ref_path, next_waypoint_id
            )
            if force_episode_end:
                action = 0
                source = "expert"

            wp_available = next_waypoint_id < len(ref_path)
            ref_actions_len = self.ref_action_lengths.get(episode_id, DEFAULT_EPISODE_LENGTH)
            denom = max(1, int(ref_actions_len / max(1, len(ref_path) - 1)))
            error_not_tolerated = (
                (source == "model" and action == 0 and metrics["distance_to_goal"] >= 3.0)
                or (accumulated_error / denom > 0.8)
                or (accumulated_error > 12)
            )
            if wp_available and error_not_tolerated:
                model_success = False
                num_rescue_events += 1
                accumulated_error = 0
                model_pending_actions = []
                rescue_event_start_idx = len(actions)
                action = self.get_oracle_action(expert, ref_path, next_waypoint_id)
                source = "expert"
                rescue_remaining = RESCUE_ACTION_LEN - 1
                last_navigation = f"expert rescue: {self.agent.action_id_to_str(action)}"
                current_action_is_rescue = True

            if action == 0 and not force_episode_end:
                action = self.get_oracle_action(expert, ref_path, next_waypoint_id)
                source = "expert"
                last_navigation = f"expert continue: {self.agent.action_id_to_str(action)}"

            debug_frames.append(
                make_debug_frame(
                    np.array(current_rgb),
                    top_down_map,
                    observation["instruction"]["text"],
                    last_navigation,
                    source,
                    rescue_remaining,
                    metrics["distance_to_goal"],
                )
            )

            observation = env.step({"action": action})
            metrics = env.get_metrics()
            current_rgb = self.agent.observe(observation)
            kept_rgb_frames.append(current_rgb)
            actions.append(int(action))
            action_source.append(source)
            if source == "model":
                num_model_actions += 1
            else:
                num_expert_actions += 1
            if current_action_is_rescue and rescue_event_start_idx is not None and rescue_remaining == 0:
                rescue_spans.append((rescue_event_start_idx, len(actions)))
                rescue_event_start_idx = None

            if force_episode_end:
                break

        if rescue_event_start_idx is not None:
            rescue_spans.append((rescue_event_start_idx, len(actions)))

        relative_pl = initial_distance / max(initial_distance, float(metrics["path_length"]))
        rescued = not model_success
        rescue_tier = classify_rescue_tier(num_rescue_events)
        kept_reason = ""
        terminal_stop = bool(actions) and actions[-1] == 0
        kept = terminal_stop and metrics["distance_to_goal"] < MIDGOAL_RADIUS and (
            ((rescued and relative_pl < RELATIVE_PATH_LENGTH_THRESHOLD))
            or (relative_pl < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD)
        )
        if kept:
            if rescue_tier == "clean":
                kept_reason = "clean_good_path"
            elif rescue_tier == "short_rescue":
                kept_reason = "short_rescue_good_path"
            else:
                kept_reason = "long_rescue_good_path"
        training_segments = build_training_segments(len(actions), rescue_spans, terminal_stop, rescue_tier)

        outcome = EpisodeOutcome(
            episode_id=episode_id,
            trajectory_id=str(trajectory_id) if trajectory_id is not None else None,
            dataset=self.dataset_name,
            video_id=video_id,
            image_dir=image_dir,
            instructions=instructions,
            actions=actions,
            action_source=action_source,
            num_rescue_events=num_rescue_events,
            model_success=model_success,
            rescue_tier=rescue_tier,
            kept_reason=kept_reason,
            training_segments=training_segments,
        )
        metric_row = {
            "episode_id": episode_id,
            "trajectory_id": str(trajectory_id) if trajectory_id is not None else None,
            "dataset": self.dataset_name,
            "success": float(metrics.get("success", 0.0)),
            "oracle_success": float(metrics.get("oracle_success", 0.0)),
            "spl": float(metrics.get("spl", 0.0)),
            "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
            "path_length": float(metrics.get("path_length", 0.0)),
            "ndtw": float(metrics.get("ndtw", 0.0)) if "ndtw" in metrics else None,
            "pl": float(relative_pl),
            "num_steps": len(actions),
            "num_model_actions": num_model_actions,
            "num_expert_actions": num_expert_actions,
            "num_rescue_events": num_rescue_events,
            "accumulated_error_max": accumulated_error_max,
            "model_success": model_success,
            "rescue_tier": rescue_tier,
            "terminal_stop": terminal_stop,
            "kept": kept,
            "kept_reason": kept_reason,
            "num_training_segments": len(training_segments),
        }
        return outcome, metric_row, debug_frames, kept_rgb_frames

    def get_oracle_action(self, expert, ref_path, next_waypoint_id: int) -> int:
        if not ref_path or next_waypoint_id >= len(ref_path):
            return 0
        action = expert.get_next_action(ref_path[next_waypoint_id])
        return int(0 if action is None else action)

    def advance_waypoint_if_reached(self, env: Env, expert, ref_path, next_waypoint_id: int):
        force_episode_end = False
        while next_waypoint_id < len(ref_path) and self.get_oracle_action(expert, ref_path, next_waypoint_id) == 0:
            next_waypoint_id += 1
            if next_waypoint_id == len(ref_path) - 1:
                expert = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
            if next_waypoint_id >= len(ref_path):
                force_episode_end = True
                break
        return expert, next_waypoint_id, force_episode_end

    def save_episode_images(self, video_id: str, frames: List[Image.Image]) -> None:
        target_dir = os.path.join(self.images_dir, video_id)
        os.makedirs(target_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(target_dir, f"frame_{idx}.jpg"))

    def save_debug_video(self, video_id: str, kept: bool, rescued: bool, frames: List[np.ndarray]) -> None:
        tags = []
        if rescued:
            tags.append("rescue")
        if kept:
            tags.append("kept")
        prefix = "_".join(tags) if tags else "debug"
        path = os.path.join(self.debug_videos_dir, f"{prefix}_{video_id}.gif")
        imageio.mimsave(path, frames, fps=4)
        self.debug_video_count += 1

    def build_summary(self, metrics_all: List[Dict], metrics_kept: List[Dict]) -> Dict:
        kept_clean = [row for row in metrics_kept if row["rescue_tier"] == "clean"]
        kept_short = [row for row in metrics_kept if row["rescue_tier"] == "short_rescue"]
        kept_long = [row for row in metrics_kept if row["rescue_tier"] == "long_rescue"]
        return {
            "dataset": self.dataset_name,
            "num_episodes": len(metrics_all),
            "num_kept": len(metrics_kept),
            "keep_rate": safe_mean([float(row["kept"]) for row in metrics_all]),
            "success_rate": safe_mean([row["success"] for row in metrics_all]),
            "avg_distance_to_goal": safe_mean([row["distance_to_goal"] for row in metrics_all]),
            "avg_pl": safe_mean([row["pl"] for row in metrics_all]),
            "avg_num_rescue_events": safe_mean([float(row["num_rescue_events"]) for row in metrics_all]),
            "avg_kept_pl": safe_mean([row["pl"] for row in metrics_kept]),
            "avg_kept_distance_to_goal": safe_mean([row["distance_to_goal"] for row in metrics_kept]),
            "num_kept_clean": len(kept_clean),
            "num_kept_short_rescue": len(kept_short),
            "num_kept_long_rescue": len(kept_long),
            "avg_kept_segments": safe_mean([float(row["num_training_segments"]) for row in metrics_kept]),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["r2r", "rxr"], required=True)
    parser.add_argument("--exp-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--max-action-history", type=int, default=10)
    parser.add_argument("--resolution-ratio", type=float, default=0.5)
    parser.add_argument("--prompt-style", choices=["baseline", "stop_hint", "sr_stop"], default="baseline")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-debug-videos", type=int, default=100)
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.split_num < 1:
        parser.error("--split-num must be >= 1")
    if args.split_id < 0 or args.split_id >= args.split_num:
        parser.error("--split-id must satisfy 0 <= split-id < split-num")
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    collector = EpisodeCollector(args)
    collector.collect()


if __name__ == "__main__":
    main()
