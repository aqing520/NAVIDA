import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import io
import base64
import re
from tqdm import tqdm
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
from types import SimpleNamespace
import argparse, habitat
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image, ImageFont, ImageDraw
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
import time, math
from openai import OpenAI

from src.eval.topo_memory import (
    DEFAULT_SIGLIP_PATH,
    RemoteSiglipImageEncoder,
    SiglipImageEncoder,
    SiglipMultimodalEncoder,
    TopoMemoryGraph,
)


def encode_image_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


SYSTEM_PROMPT = "You are a helpful assistant."

BASE_PROMPT_TEMPLATE = "Imagine you are a robot programmed for navigation tasks. "\
    "You have been given a video of historical observations and an image of the current observation. "\
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
    "which could involve turning left or right by a specific degree or moving forward a certain distance."

STOP_HINT_PROMPT_TEMPLATE = BASE_PROMPT_TEMPLATE + " You may answer stop when the goal has been reached."

SR_STOP_PROMPT_TEMPLATE = "Imagine you are a robot programmed for navigation tasks. "\
    "You have been given a video of historical observations and an image of the current observation. "\
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move. "\
    "Available actions are: stop; move forward by a distance in cm; turn left by degrees; turn right by degrees. "\
    "If the current view already satisfies the destination description or reaches the final landmark, choose stop instead of moving on. "\
    "Otherwise choose the safest next one or two actions that continue following the instruction. "\
    "Respond only with comma-separated actions in these formats: stop, forward <number> cm, turn left <number> degree, turn right <number> degree."


def get_prompt_template(prompt_style):
    if prompt_style == "baseline":
        return BASE_PROMPT_TEMPLATE
    if prompt_style == "stop_hint":
        return STOP_HINT_PROMPT_TEMPLATE
    if prompt_style == "sr_stop":
        return SR_STOP_PROMPT_TEMPLATE
    raise ValueError(f"Unsupported prompt style: {prompt_style}")

def seed_all():
    np.random.seed(41)
    random.seed(41)


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio, prompt_style,
                    temperature, max_episodes, memory_style, memory_encoder_path,
                    memory_encoder_device, memory_encoder_server, memory_sim_threshold,
                    memory_max_nodes, memory_text_max_lines, history_selection, stop_verifier, stop_verifier_base_url,
                    target_stop_hint, target_hint_min_step, target_hint_top_k, target_hint_margin) -> None:
 
    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(
        api_key, 
        base_url, 
        result_path, 
        forward_distance, 
        turn_angle, 
        max_action_history, 
        resolution_ratio, 
        num_generations,
        prompt_style,
        temperature,
        memory_style=memory_style,
        memory_encoder_path=memory_encoder_path,
        memory_encoder_device=memory_encoder_device,
        memory_encoder_server=memory_encoder_server,
        memory_sim_threshold=memory_sim_threshold,
        memory_max_nodes=memory_max_nodes,
        memory_text_max_lines=memory_text_max_lines,
        history_selection=history_selection,
        stop_verifier=stop_verifier,
        stop_verifier_base_url=stop_verifier_base_url,
        target_stop_hint=target_stop_hint,
        target_hint_min_step=target_hint_min_step,
        target_hint_top_k=target_hint_top_k,
        target_hint_margin=target_hint_margin)

    num_episodes = len(env.episodes)
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)
    
    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success","ndtw"}

    count = 0
    
    for _ in range(num_episodes):
        episode_start_time = time.time()

        obs = env.reset()
        iter_step = 0
        agent.reset()

        t_dict = {
            "t_episode": 0,
        }

        continuse_rotation_count = 0
        last_dtg = 999
        if os.path.exists(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id))):
            if result_queue is not None:
                result_queue.put({"t_episode": 0, "skipped": 1})
            continue
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            
            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)
        
        t_dict["t_episode"] = time.time() - episode_start_time
        if result_queue is not None:
            result_queue.put(t_dict)

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, num_generations = 1,
                    prompt_style="baseline", temperature=0.2, require_map=True,
                    memory_style="none", memory_encoder_path=DEFAULT_SIGLIP_PATH,
                    memory_encoder_device="cpu", memory_encoder_server=None,
                    memory_sim_threshold=0.84, memory_max_nodes=80,
                    memory_text_max_lines=4, history_selection="uniform", stop_verifier="none",
                    stop_verifier_base_url=None, target_stop_hint="none",
                    target_hint_min_step=8, target_hint_top_k=3, target_hint_margin=0.01):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.memory_style = memory_style
        self.history_selection = history_selection
        self.stop_verifier = stop_verifier
        self.target_stop_hint = target_stop_hint
        self.target_hint_min_step = target_hint_min_step
        self.target_hint_top_k = target_hint_top_k
        self.target_hint_margin = target_hint_margin
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "trace"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        self.stop_client = self.client
        self.stop_model = self.model
        if stop_verifier_base_url:
            self.stop_client = OpenAI(api_key=api_key, base_url=stop_verifier_base_url)
            self.stop_model = self.stop_client.models.list().data[0].id
        
        self.sampling_params = SimpleNamespace(
            n=1,
            temperature=temperature,
            max_tokens=512, # i.e. max_completion_tokens
            top_p=1.0,
        )

        self.promt_template = get_prompt_template(prompt_style)
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.pending_transition_summary = None
        self.topo_memory = None
        self.history_encoder = None
        self.keyframe_bank = []
        self.keyframe_max_nodes = memory_max_nodes
        self.instruction_embedding = None
        self.target_score_history = []
        self.target_best_score = None
        needs_multimodal_encoder = self.history_selection in ("semantic_keyframe", "uniform_relevant") or self.target_stop_hint != "none"
        needs_encoder = self.memory_style in ("topo_text", "topo_semantic_text") or needs_multimodal_encoder
        encoder = None
        if needs_encoder:
            if memory_encoder_server:
                encoder = RemoteSiglipImageEncoder(memory_encoder_server)
            elif needs_multimodal_encoder:
                encoder = SiglipMultimodalEncoder(
                    model_path=memory_encoder_path,
                    device=memory_encoder_device,
                )
            else:
                encoder = SiglipImageEncoder(
                    model_path=memory_encoder_path,
                    device=memory_encoder_device,
                )
        if self.history_selection in ("semantic_keyframe", "uniform_relevant") or self.target_stop_hint != "none":
            self.history_encoder = encoder
        if self.memory_style in ("topo_text", "topo_semantic_text"):
            use_semantic_memory = self.memory_style == "topo_semantic_text"
            self.topo_memory = TopoMemoryGraph(
                encoder=encoder,
                sim_threshold=memory_sim_threshold,
                max_nodes=memory_max_nodes,
                text_max_lines=memory_text_max_lines,
                semantic=use_semantic_memory,
                node_captioner=self.describe_memory_node if use_semantic_memory else None,
                include_avoid_hint=not use_semantic_memory,
            )
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def _add_unique_frame(self, selected, frame_idx, image, source, score=None):
        if frame_idx in selected:
            return False
        selected[frame_idx] = {"image": image, "source": source, "score": score}
        return True

    def _ensure_instruction_embedding(self, instruction):
        if self.instruction_embedding is not None:
            return self.instruction_embedding
        if self.history_encoder is None or not hasattr(self.history_encoder, "encode_text"):
            return None
        try:
            self.instruction_embedding = self.history_encoder.encode_text(instruction)
        except Exception as exc:
            print(f"instruction embedding failed: {exc}")
            self.instruction_embedding = None
        return self.instruction_embedding

    def _add_keyframe_bank_item(self, image, frame_idx):
        if self.history_encoder is None:
            return
        try:
            embedding = self.history_encoder.encode(image)
        except Exception as exc:
            print(f"keyframe image embedding failed: {exc}")
            return
        self.keyframe_bank.append({
            "frame_idx": frame_idx,
            "image": image,
            "embedding": embedding,
        })
        if len(self.keyframe_bank) > self.keyframe_max_nodes:
            self.keyframe_bank = self.keyframe_bank[-self.keyframe_max_nodes:]

    def _rank_instruction_relevant_frames(self, instruction, history, selected_ids=None, top_k=2):
        selected_ids = selected_ids or set()
        text_embedding = self._ensure_instruction_embedding(instruction)
        if text_embedding is None:
            return []
        valid_history_ids = {idx for idx, _ in history}
        scored = []
        for item in self.keyframe_bank:
            frame_idx = item["frame_idx"]
            if frame_idx not in valid_history_ids or frame_idx in selected_ids:
                continue
            score = float(np.dot(text_embedding, item["embedding"]))
            scored.append((score, frame_idx, item["image"]))
        return sorted(scored, reverse=True)[:top_k]

    def update_target_stop_hint(self, instruction, image, frame_idx):
        if self.target_stop_hint == "none":
            return None, None
        if self.history_encoder is None or not hasattr(self.history_encoder, "encode_text"):
            return None, {"enabled": self.target_stop_hint, "error": "missing_multimodal_encoder"}

        text_embedding = self._ensure_instruction_embedding(instruction)
        if text_embedding is None:
            return None, {"enabled": self.target_stop_hint, "error": "missing_instruction_embedding"}

        try:
            image_embedding = self.history_encoder.encode(image)
            score = float(np.dot(text_embedding, image_embedding))
        except Exception as exc:
            print(f"target stop hint embedding failed: {exc}")
            return None, {"enabled": self.target_stop_hint, "error": str(exc)}

        self.target_score_history.append({"step": self.step_idx, "frame_idx": frame_idx, "score": score})
        scores = [item["score"] for item in self.target_score_history]
        best_score = max(scores)
        self.target_best_score = best_score
        rank = 1 + sum(1 for item_score in scores if item_score > score)
        enough_history = self.step_idx >= self.target_hint_min_step and len(scores) >= self.target_hint_min_step
        near_best = score >= best_score - self.target_hint_margin
        in_top_k = rank <= self.target_hint_top_k
        active = enough_history and near_best and in_top_k

        info = {
            "enabled": self.target_stop_hint,
            "score": round(score, 4),
            "best_score": round(best_score, 4),
            "rank": rank,
            "active": active,
            "min_step": self.target_hint_min_step,
            "top_k": self.target_hint_top_k,
            "margin": self.target_hint_margin,
        }
        if not active:
            return None, info

        hint_text = (
            "Potential destination cue: the current view is visually similar to the instruction. "
            "If the current observation satisfies the instruction, stop."
        )
        return hint_text, info

    def select_history_frames(self, instruction, max_frames=8):
        history = list(zip(self.rgb_frame_indices[:-1], self.rgb_list[:-1]))
        if len(history) <= max_frames or self.history_selection == "uniform":
            sampled = self.uniform_sample_with_ends(history, max_frames)
            return [image for _, image in sampled], {
                "history_selection": self.history_selection,
                "selected": [idx for idx, _ in sampled],
            }

        if self.history_selection == "uniform_relevant":
            base = self.uniform_sample_with_ends(history, max_frames)
            selected_ids = {idx for idx, _ in base}
            relevant = self._rank_instruction_relevant_frames(instruction, history, selected_ids, top_k=2)
            slots = [3, 4] if len(base) >= 6 else list(range(1, max(1, len(base) - 1)))
            selected = {idx: {"image": image, "source": "uniform", "score": None} for idx, image in base}
            for slot, (score, frame_idx, image) in zip(slots, relevant):
                if slot < len(base):
                    old_idx = base[slot][0]
                    if old_idx not in (base[0][0], base[-1][0]):
                        selected.pop(old_idx, None)
                selected[frame_idx] = {"image": image, "source": "instruction_relevant", "score": score}
            ordered = sorted(selected.items(), key=lambda item: item[0])[:max_frames]
            info = {
                "history_selection": self.history_selection,
                "selected": [idx for idx, _ in ordered],
                "sources": {str(idx): meta["source"] for idx, meta in ordered},
                "scores": {str(idx): round(meta["score"], 4) for idx, meta in ordered if meta["score"] is not None},
            }
            return [meta["image"] for _, meta in ordered], info

        selected = {}
        self._add_unique_frame(selected, history[0][0], history[0][1], "start")

        for frame_idx, image in history[-3:]:
            self._add_unique_frame(selected, frame_idx, image, "recent")

        existing = set(selected)
        remaining = [(idx, img) for idx, img in history if idx not in existing]
        for frame_idx, image in self.uniform_sample_with_ends(remaining, 2):
            self._add_unique_frame(selected, frame_idx, image, "long_uniform")

        for score, frame_idx, image in self._rank_instruction_relevant_frames(instruction, history, set(selected), top_k=2):
            self._add_unique_frame(selected, frame_idx, image, "instruction_relevant", score)

        if len(selected) < max_frames:
            for frame_idx, image in self.uniform_sample_with_ends(history, max_frames):
                self._add_unique_frame(selected, frame_idx, image, "fill_uniform")
                if len(selected) >= max_frames:
                    break

        ordered = sorted(selected.items(), key=lambda item: item[0])[:max_frames]
        info = {
            "history_selection": self.history_selection,
            "selected": [idx for idx, _ in ordered],
            "sources": {str(idx): meta["source"] for idx, meta in ordered},
            "scores": {str(idx): round(meta["score"], 4) for idx, meta in ordered if meta["score"] is not None},
        }
        return [meta["image"] for _, meta in ordered], info


    def predict_inference(self):

        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

    def describe_memory_node(self, image):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an indoor visual landmark captioner. "
                            "This is not a navigation action task; never output movement commands."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(image)}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Write exactly one short visual place label for this indoor view. "
                            "Start with 'place:'. Mention only static visual landmarks such as room, hallway, "
                            "door, stairs, furniture, windows, walls, signs, or openings. "
                            "Forbidden words: forward, turn, left degree, right degree, cm, stop, move. "
                            "Maximum 14 words after 'place:'."
                        ),
                    },
                ],
            },
        ]
        try:
            outputs = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_completion_tokens=48,
                temperature=0.0,
                top_p=1.0,
            )
            raw_caption = outputs.choices[0].message.content.strip()
        except Exception as exc:
            print(f"memory caption failed: {exc}")
            return None

        caption = self._sanitize_memory_caption(raw_caption)
        if caption is None and raw_caption:
            print(f"memory caption rejected: {raw_caption[:120]}")
        return caption

    def _sanitize_memory_caption(self, caption):
        caption = re.sub(r"<[^>]+>", "", str(caption))
        caption = caption.replace('"', "").replace("'", "")
        caption = " ".join(caption.strip().split())
        if not caption:
            return None

        lower = caption.lower()
        if lower.startswith("place:"):
            caption = caption.split(":", 1)[1].strip()
            lower = caption.lower()
        for prefix in ("the image shows ", "this image shows ", "this view shows ", "a view of "):
            if lower.startswith(prefix):
                caption = caption[len(prefix):].strip()
                lower = caption.lower()
                break

        action_pattern = r"\b(stop|forward|turn|move|left|right|degree|degrees|cm|centimeter|centimeters)\b"
        if re.search(action_pattern, lower) or re.search(r"\b\d+(?:\.\d+)?\s*(cm|degree|degrees)\b", lower):
            return None

        words = caption.split()
        if len(words) > 16:
            caption = " ".join(words[:16])
        return caption[:120] if caption else None

    def verify_should_stop(self, observations, image):
        if self.stop_verifier == "none":
            return False, None
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a strict visual navigation destination verifier. "
                            "Answer only yes or no."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(image)}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Navigation instruction: "
                            f"{observations['instruction']['text']}\n"
                            "Is the robot currently at the final destination described by the instruction? "
                            "Answer yes only if it should stop now. Otherwise answer no."
                        ),
                    },
                ],
            },
        ]
        try:
            outputs = self.stop_client.chat.completions.create(
                messages=messages,
                model=self.stop_model,
                max_completion_tokens=8,
                temperature=0.0,
                top_p=1.0,
            )
            answer = outputs.choices[0].message.content.strip()
        except Exception as exc:
            print(f"stop verifier failed: {exc}")
            return False, {"enabled": self.stop_verifier, "error": str(exc)}

        normalized = re.sub(r"<[^>]+>", "", answer).strip().lower()
        should_stop = normalized.startswith("yes") and not normalized.startswith("no")
        return should_stop, {"enabled": self.stop_verifier, "raw": answer, "should_stop": should_stop}


    def _message_text(self, content):
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                text_parts.append("<image>")
        return "\n".join(text_parts)

    def _build_step_record(
        self,
        step_idx,
        observations,
        info,
        raw_output,
        chosen_action,
        action_source,
        memory_info=None,
        memory_text=None,
        history_selection_info=None,
        stop_verifier_info=None,
        target_stop_hint_info=None,
    ):
        record = {
            "episode_id": self.episode_id,
            "step_idx": step_idx,
            "distance_to_goal": info.get("distance_to_goal"),
            "success": info.get("success"),
            "raw_output": raw_output,
            "chosen_action": chosen_action,
            "action_source": action_source,
            "history_frames": len(self.rgb_list),
            "pending_action_queue": list(self.pending_action_list),
        }
        if memory_info is not None:
            record["memory_style"] = self.memory_style
            record["memory_info"] = memory_info
            record["memory_text"] = memory_text
        if history_selection_info is not None:
            record["history_selection_info"] = history_selection_info
        if stop_verifier_info is not None:
            record["stop_verifier_info"] = stop_verifier_info
        if target_stop_hint_info is not None:
            record["target_stop_hint_info"] = target_stop_hint_info
        if step_idx == 0:
            record["instruction"] = observations["instruction"]["text"]
            record["user_prompt"] = (
                self._message_text(self.conversations[-1]["content"])
                if len(self.conversations) > 1
                else None
            )
        return record

    def _log_step(self, record):
        append_jsonl(
            os.path.join(self.result_path, "trace", f"trace_{self.episode_id}.jsonl"),
            record,
        )

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
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")

    def summarize_action_plan(self, action_plan):
        parts = []
        for action_id, numeric in action_plan:
            if action_id == 0:
                parts.append("stop")
            elif action_id == 1:
                value = numeric if numeric is not None else self.forward_distance
                parts.append(f"forward {value:g} cm")
            elif action_id in (2, 3):
                value = numeric if numeric is not None else self.turn_angle
                direction = "left" if action_id == 2 else "right"
                parts.append(f"turn {direction} {value:g} degree")
        return ", ".join(parts) if parts else None
        
    def reset(self):       
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.topdown_map_list = []

        self.pending_action_list = []
        self.rgb_list = []
        self.rgb_frame_indices = []
        self.global_frame_idx = 0
        self.keyframe_bank = []
        self.instruction_embedding = None
        self.target_score_history = []
        self.target_best_score = None
        self.pending_transition_summary = None
        if self.topo_memory is not None:
            self.topo_memory.reset()

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        self.step_idx = 0
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))
        current_frame_idx = self.global_frame_idx
        self.global_frame_idx += 1
        self.rgb_list.append(rgb_)
        self.rgb_frame_indices.append(current_frame_idx)
        # do not cut down rgb list while using uniform sampling
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]
            self.rgb_frame_indices = self.rgb_frame_indices[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            chosen_action = {"action": temp_action}
            self._log_step(
                self._build_step_record(
                    step_idx=self.step_idx,
                    observations=observations,
                    info=info,
                    raw_output=None,
                    chosen_action=chosen_action,
                    action_source="pending_queue",
                )
            )
            self.step_idx += 1
            return chosen_action

        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []
        memory_info = None
        memory_text = None
        history_selection_info = None
        target_stop_hint_text = None
        target_stop_hint_info = None
        target_stop_hint_text, target_stop_hint_info = self.update_target_stop_hint(
            observations["instruction"]["text"],
            rgb_,
            current_frame_idx,
        )
        if self.topo_memory is not None:
            memory_info = self.topo_memory.observe(
                self.rgb_list[-1],
                self.step_idx,
                self.pending_transition_summary,
            )
            self.pending_transition_summary = None
            memory_text = self.topo_memory.build_prompt_text()

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            history_images, history_selection_info = self.select_history_frames(observations["instruction"]["text"], 8)
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in history_images])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
            history_selection_info = {"history_selection": self.history_selection, "selected": [current_frame_idx], "sources": {str(current_frame_idx): "current_fallback"}}
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        if memory_text:
            content.append({"type": "text", "text": memory_text})
        if target_stop_hint_text:
            content.append({"type": "text", "text": target_stop_hint_text})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        raw_navigation = navigation
        stop_verifier_info = None
        should_stop, stop_verifier_info = self.verify_should_stop(observations, rgb_)
        if should_stop:
            navigation = "stop"
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        if self.history_selection in ("semantic_keyframe", "uniform_relevant"):
            self._add_keyframe_bank_item(rgb_, current_frame_idx)
        
        result = self.extract_multi_result(navigation)
        parsed_actions = [
            {"action_id": action_index, "value": numeric}
            for action_index, numeric in result
        ]

        select_action_idx = 2

        result = result[:select_action_idx]
        self.pending_transition_summary = self.summarize_action_plan(result)
        for action_index,numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                self.pending_action_list.append(action_index)

        chosen_action = {"action": self.pending_action_list.pop(0)}
        self._log_step(
            self._build_step_record(
                step_idx=self.step_idx,
                observations=observations,
                info=info,
                raw_output=raw_navigation,
                chosen_action=chosen_action,
                action_source="model",
                memory_info=memory_info,
                memory_text=memory_text,
                history_selection_info=history_selection_info,
                stop_verifier_info=stop_verifier_info,
                target_stop_hint_info=target_stop_hint_info,
            )
        )
        self.step_idx += 1
        return chosen_action


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,default=None,help="optional split ID; when set, run only this split in the current process")
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    parser.add_argument("--prompt-style", choices=["baseline", "stop_hint", "sr_stop"], default="baseline",
                        help="prompt template used for navigation decisions")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="sampling temperature for vLLM chat completions")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="optional maximum number of episodes to evaluate in each split")
    parser.add_argument("--memory-style", choices=["none", "topo_text", "topo_semantic_text"], default="none",
                        help="optional topological memory prompt style")
    parser.add_argument("--memory-encoder-path", type=str, default=DEFAULT_SIGLIP_PATH,
                        help="local SigLIP model path used by topo_text memory")
    parser.add_argument("--memory-encoder-device", type=str, default="cpu",
                        help="device used by the memory image encoder")
    parser.add_argument("--memory-encoder-server", type=str, default=None,
                        help="optional SigLIP embedding server URL used by topo_text memory")
    parser.add_argument("--memory-sim-threshold", type=float, default=0.84,
                        help="cosine similarity threshold for topological revisits")
    parser.add_argument("--memory-max-nodes", type=int, default=80,
                        help="maximum topological memory nodes kept per episode")
    parser.add_argument("--memory-text-max-lines", type=int, default=4,
                        help="maximum lines inserted into the prompt by topo_text memory")
    parser.add_argument("--history-selection", choices=["uniform", "semantic_keyframe", "uniform_relevant"], default="uniform",
                        help="strategy used to choose historical images before the current observation")
    parser.add_argument("--stop-verifier", choices=["none", "current"], default="none",
                        help="optional VLM verifier that can override the selected action with stop")
    parser.add_argument("--stop-verifier-base-url", type=str, default=None,
                        help="optional OpenAI-compatible base URL for a separate verifier VLM")
    parser.add_argument("--target-stop-hint", choices=["none", "siglip"], default="none",
                        help="optional SigLIP current-view destination hint; does not change historical image selection")
    parser.add_argument("--target-hint-min-step", type=int, default=8,
                        help="minimum model-decision step before enabling target stop hints")
    parser.add_argument("--target-hint-top-k", type=int, default=3,
                        help="current view must rank within this many best SigLIP scores in the episode")
    parser.add_argument("--target-hint-margin", type=float, default=0.01,
                        help="current SigLIP score must be within this margin of the episode best score")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        # self.config.habitat.task.measurements.success.success_distance=3.0
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
    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)

    if args.split_id is not None:
        evaluate_agent(None, api_key, base_url, config, dataset_splits[args.split_id], args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle,
                args.max_action_history, args.resolution_ratio, args.prompt_style,
                args.temperature, args.max_episodes, args.memory_style, args.memory_encoder_path,
                args.memory_encoder_device, args.memory_encoder_server, args.memory_sim_threshold,
                args.memory_max_nodes, args.memory_text_max_lines, args.history_selection, args.stop_verifier, args.stop_verifier_base_url,
                args.target_stop_hint, args.target_hint_min_step, args.target_hint_top_k, args.target_hint_margin)
        return

    num_episodes = len(dataset.episodes)
    if args.max_episodes is not None:
        num_episodes = sum(min(len(split.episodes), args.max_episodes) for split in dataset_splits)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (result_queue, api_key, base_url, config, dataset_splits[i], args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle, 
                args.max_action_history, args.resolution_ratio, args.prompt_style,
                args.temperature, args.max_episodes, args.memory_style, args.memory_encoder_path,
                args.memory_encoder_device, args.memory_encoder_server, args.memory_sim_threshold,
                args.memory_max_nodes, args.memory_text_max_lines, args.history_selection, args.stop_verifier, args.stop_verifier_base_url,
                args.target_stop_hint, args.target_hint_min_step, args.target_hint_top_k, args.target_hint_margin)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
