"""
Stage 1: Rollout Collection for Self-correction Flywheel

Run the current model on train split, recording per-step state:
- agent pose (position, rotation)
- RGB frames
- oracle action (from ShortestPathFollower)
- distance_to_goal, success, oracle_success
- model output

Output:
- traces/{episode_id}.jsonl: per-step trace
- frames/{episode_id}/frame_{step}.jpg: per-step RGB frames
- summaries/{episode_id}.json: episode-level summary
"""

import json
import os
import io
import re
import time
import math
import random
import base64
import argparse
import multiprocessing as mp
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

import habitat
from habitat import Env
from habitat.core.agent import Agent
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat_extensions import measures, task


def encode_image_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


SYSTEM_PROMPT = "You are a helpful assistant."

BASE_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


def seed_all():
    np.random.seed(41)
    random.seed(41)


def episode_scene_name(scene_id):
    return os.path.splitext(os.path.basename(scene_id))[0]


def observations_instruction_text(observations):
    if "instruction" not in observations:
        return ""
    instruction = observations["instruction"]
    if isinstance(instruction, dict):
        return instruction.get("text", "")
    return ""


def safe_json(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    return obj


class CaptureAgent(Agent):
    """Agent that records detailed per-step state during rollout."""

    def __init__(self, api_key, base_url, result_path, forward_distance,
                 turn_angle, max_action_history, resolution_ratio,
                 temperature=0.1):
        self.result_path = result_path
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history

        os.makedirs(os.path.join(self.result_path, "traces"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "summaries"), exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.client.models.list().data[0].id

        self.sampling_params = SimpleNamespace(
            n=1,
            temperature=temperature,
            max_tokens=512,
            top_p=1.0,
        )

        self.promt_template = BASE_PROMPT_TEMPLATE
        self.rgb_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })
        self.pending_action_list = []
        self.step_id = 0
        self.episode_id = None
        self.follower = None
        self.goal_position = None

        self.last_action_meta = {
            "decision_source": "reset",
            "raw_output": None,
            "selected_action": None,
            "parsed_action_ids": [],
            "pending_length_after": 0,
        }

    def reset(self, episode_id, goal_position, sim):
        self.episode_id = episode_id
        self.goal_position = goal_position
        self.pending_action_list = []
        self.rgb_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })
        self.step_id = 0

        # Initialize ShortestPathFollower for this episode
        try:
            self.follower = ShortestPathFollower(
                sim, goal_radius=3.0, return_one_hot=False
            )
        except Exception as e:
            print(f"Warning: ShortestPathFollower init failed for episode {episode_id}: {e}")
            self.follower = None

        self.last_action_meta = {
            "decision_source": "reset",
            "raw_output": None,
            "selected_action": None,
            "parsed_action_ids": [],
            "pending_length_after": 0,
        }

    def get_oracle_action(self):
        """Get oracle action from ShortestPathFollower."""
        if self.follower is None or self.goal_position is None:
            return None
        try:
            action = self.follower.get_next_action(self.goal_position)
            return int(action)
        except Exception:
            return None

    def uniform_sample_with_ends(self, data, n):
        if len(data) <= n:
            return data
        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def predict_inference(self, messages=None):
        outputs = self.client.chat.completions.create(
            messages=self.conversations if messages is None else messages,
            model=self.model,
            max_completion_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
        )
        output_text = outputs.choices[0].message.content.strip()
        return output_text

    def extract_multi_result(self, output):
        sub_actions = [item for item in re.split(r'\s*,\s*', output.strip()) if item]
        if len(sub_actions) == 0:
            sub_actions = [output]
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()
        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            return 1, float(match.group())
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            return 2, float(match.group())
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            return 3, float(match.group())
        return None, None

    def expand_action_repeats(self, action_index, numeric):
        if action_index == 1:
            return min(3, round(numeric / self.forward_distance))
        if action_index in (2, 3):
            repeats = min(3, round(numeric / self.turn_angle))
            if numeric is not None and numeric >= 45:
                repeats = min(repeats, 2)
            return repeats
        return 0

    def save_frame(self, rgb, episode_id, step):
        """Save RGB frame to disk."""
        frame_dir = os.path.join(self.result_path, "frames", str(episode_id))
        os.makedirs(frame_dir, exist_ok=True)
        rgb_pil = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_pil = rgb_pil.resize((308, 252))
        frame_path = os.path.join(frame_dir, f"frame_{step}.jpg")
        rgb_pil.save(frame_path, quality=85)
        return frame_path

    def act_and_record(self, observations, info, episode_id, iter_step,
                       continuse_rotation_count, forced_stop_reason):
        """Act and return action + per-step record."""
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308, 252))
        self.rgb_list.append(rgb_)
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        # Save frame
        frame_path = self.save_frame(rgb, episode_id, self.step_id)

        # Get oracle action
        oracle_action = self.get_oracle_action()

        # Get agent pose
        agent_state = observations.get("_agent_state", None)
        agent_position = None
        agent_rotation = None
        if agent_state is not None:
            agent_position = agent_state.position.tolist() if hasattr(agent_state.position, 'tolist') else list(agent_state.position)
            if agent_state.rotation is not None:
                rot = agent_state.rotation
                if hasattr(rot, 'components'):
                    agent_rotation = [rot.x, rot.y, rot.z, rot.w]
                else:
                    agent_rotation = list(rot)

        goal_position = None
        if self.goal_position is not None:
            goal_position = self.goal_position.tolist() if hasattr(self.goal_position, 'tolist') else list(self.goal_position)

        # If there are pending actions, execute them
        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)
            self.last_action_meta = {
                "decision_source": "pending_action",
                "raw_output": None,
                "selected_action": temp_action,
                "parsed_action_ids": [],
                "pending_length_after": len(self.pending_action_list),
            }
            record = {
                "episode_id": int(episode_id),
                "step": self.step_id,
                "iter_step": int(iter_step),
                "instruction": observations_instruction_text(observations),
                "agent_position": agent_position,
                "agent_rotation": agent_rotation,
                "distance_to_goal": float(info.get("distance_to_goal", 0)),
                "model_action": int(temp_action),
                "model_raw_output": None,
                "oracle_action": oracle_action,
                "goal_position": goal_position,
                "success": int(info.get("success", 0)),
                "oracle_success": int(info.get("oracle_success", 0)),
                "is_decision_step": False,
                "frame_path": frame_path,
                "forced_stop_reason": forced_stop_reason,
            }
            self.step_id += 1
            return {"action": temp_action}, record

        # Model inference
        self.conversations = self.conversations[:1]
        content = []
        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1], 8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
            "role": "user",
            "content": content
        })

        navigation = self.predict_inference()

        result = self.extract_multi_result(navigation)
        parsed_action_ids = []
        select_action_idx = 2
        execution_result = result[:select_action_idx]

        for action_index, numeric in result:
            parsed_action_ids.append(action_index)

        for action_index, numeric in execution_result:
            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(self.expand_action_repeats(action_index, numeric)):
                    self.pending_action_list.append(1)
            elif action_index == 2:
                for _ in range(self.expand_action_repeats(action_index, numeric)):
                    self.pending_action_list.append(2)
            elif action_index == 3:
                for _ in range(self.expand_action_repeats(action_index, numeric)):
                    self.pending_action_list.append(3)

            if action_index is None or len(self.pending_action_list) == 0:
                action_index = random.randint(1, 3)
                self.pending_action_list.append(action_index)

        if len(self.pending_action_list) == 0:
            action_index = random.randint(1, 3)
            self.pending_action_list.append(action_index)

        selected_action = self.pending_action_list.pop(0)
        self.last_action_meta = {
            "decision_source": "model_output",
            "raw_output": navigation,
            "selected_action": selected_action,
            "parsed_action_ids": parsed_action_ids,
            "pending_length_after": len(self.pending_action_list),
        }

        record = {
            "episode_id": int(episode_id),
            "step": self.step_id,
            "iter_step": int(iter_step),
            "instruction": observations_instruction_text(observations),
            "agent_position": agent_position,
            "agent_rotation": agent_rotation,
            "distance_to_goal": float(info.get("distance_to_goal", 0)),
            "model_action": int(selected_action),
            "model_raw_output": navigation,
            "oracle_action": oracle_action,
            "goal_position": goal_position,
            "success": int(info.get("success", 0)),
            "oracle_success": int(info.get("oracle_success", 0)),
            "is_decision_step": True,
            "frame_path": frame_path,
            "forced_stop_reason": forced_stop_reason,
            "parsed_action_ids": parsed_action_ids,
        }
        self.step_id += 1
        return {"action": selected_action}, record


def capture_rollout(result_queue, api_key, base_url, config, dataset,
                    result_path, forward_distance, turn_angle,
                    max_action_history, resolution_ratio,
                    temperature, max_episodes) -> None:
    if len(dataset.episodes) == 0:
        if result_queue is not None:
            result_queue.put({"t_episode": 0, "empty_split": 1})
        return

    env = Env(config.habitat, dataset)

    agent = CaptureAgent(
        api_key, base_url, result_path,
        forward_distance, turn_angle,
        max_action_history, resolution_ratio,
        temperature)

    num_episodes = len(env.episodes)
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    for _ in range(num_episodes):
        episode_start_time = time.time()

        obs = env.reset()
        iter_step = 0

        episode_id = env.current_episode.episode_id
        goal_position = env.current_episode.goals[0].position
        scene_id = env.current_episode.scene_id
        instruction = observations_instruction_text(obs)

        agent.reset(episode_id, goal_position, env._sim)

        # Skip if already processed
        if os.path.exists(os.path.join(result_path, "summaries", f"{episode_id}.json")):
            if result_queue is not None:
                result_queue.put({"t_episode": 0, "skipped": 1})
            continue

        # Open trace file for this episode
        trace_path = os.path.join(result_path, "traces", f"{episode_id}.jsonl")
        trace_file = open(trace_path, "w", encoding="utf-8")

        continuse_rotation_count = 0
        last_dtg = 999
        min_distance_to_goal = float('inf')
        min_distance_step = 0

        while not env.episode_over:
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            # Track minimum distance
            if info["distance_to_goal"] < min_distance_to_goal:
                min_distance_to_goal = info["distance_to_goal"]
                min_distance_step = agent.step_id

            # Inject agent state into observations for pose recording
            try:
                obs["_agent_state"] = env._sim.get_agent_state()
            except Exception:
                obs["_agent_state"] = None

            forced_stop_reason = None
            if continuse_rotation_count > EARLY_STOP_ROTATION:
                forced_stop_reason = "rotation_stall"
            elif iter_step > EARLY_STOP_STEPS:
                forced_stop_reason = "step_limit"

            action, record = agent.act_and_record(
                obs, info, episode_id, iter_step,
                continuse_rotation_count, forced_stop_reason)

            if forced_stop_reason is not None:
                action = {"action": 0}
                record["model_action"] = 0
                record["forced_stop_reason"] = forced_stop_reason

            # Write trace record
            trace_file.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")

            iter_step += 1
            obs = env.step(action)

        trace_file.close()

        # Write episode summary
        info = env.get_metrics()
        summary = {
            "episode_id": int(episode_id),
            "final_success": int(info.get("success", 0)),
            "final_oracle_success": int(info.get("oracle_success", 0)),
            "total_steps": agent.step_id,
            "min_distance_to_goal": float(min_distance_to_goal),
            "min_distance_step": int(min_distance_step),
            "instruction": instruction,
            "goal_position": safe_json(goal_position),
            "scene_id": scene_id,
            "path_length": float(info.get("path_length", 0)),
            "spl": float(info.get("spl", 0)),
        }
        summary_path = os.path.join(result_path, "summaries", f"{episode_id}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        t_episode = time.time() - episode_start_time
        if result_queue is not None:
            result_queue.put({"t_episode": t_episode, "episode_id": int(episode_id)})

    env.close()


def main():
    seed_all()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--split-num", type=int, required=True)
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--resolution-ratio", type=float, default=0.5)
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--max-action-history", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--scene-id", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None, "Set OPENAI_API_KEY and OPENAI_API_BASE"

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3, map_resolution=1024,
                    draw_source=True, draw_border=True,
                    draw_shortest_path=True, draw_view_points=True,
                    draw_goal_positions=True, draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    if args.scene_id is not None:
        dataset = habitat.datasets.make_dataset(
            id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
        dataset.episodes = [
            ep for ep in dataset.episodes
            if episode_scene_name(ep.scene_id) == args.scene_id
        ]
    else:
        dataset = habitat.datasets.make_dataset(
            id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)

    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)

    if args.split_id is not None:
        capture_rollout(
            None, api_key, base_url, config, dataset_splits[args.split_id],
            args.result_path, args.forward_distance, args.turn_angle,
            args.max_action_history, args.resolution_ratio,
            args.temperature, args.max_episodes)
        return

    num_episodes = len(dataset.episodes)
    if args.max_episodes is not None:
        num_episodes = sum(min(len(split.episodes), args.max_episodes) for split in dataset_splits)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (
            result_queue, api_key, base_url, config, dataset_splits[i],
            args.result_path, args.forward_distance, args.turn_angle,
            args.max_action_history, args.resolution_ratio,
            args.temperature, args.max_episodes)
        p = mp.Process(target=capture_rollout, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Capturing rollout") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
