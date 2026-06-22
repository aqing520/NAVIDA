import os
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import habitat
import numpy as np
import torch
from habitat import Env
from habitat.config.default_structured_configs import CollisionsMeasurementConfig
from habitat_baselines.config.default import get_config
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig

from habitat_extensions import measures, task  # noqa: F401
from src.rl.actions import ParsedAction, parse_action_text
from src.rl.prompts import RL_PROMPT_TEMPLATE, build_vln_prompt_messages
from src.rl.rewards import RewardConfig, compute_step_reward


@dataclass
class RolloutStep:
    prompt_messages: list[dict]
    response_text: str
    action_id: int
    action_valid: bool
    reward: float
    prev_metrics: dict[str, Any]
    next_metrics: dict[str, Any]


@dataclass
class EpisodeRollout:
    episode_id: str
    steps: list[RolloutStep] = field(default_factory=list)
    total_reward: float = 0.0
    success: float = 0.0
    spl: float = 0.0
    distance_to_goal: float = 0.0
    path_length: float = 0.0


class VlnRlRolloutRunner:
    def __init__(
        self,
        exp_config: str,
        processor,
        forward_distance: int = 25,
        turn_angle: int = 15,
        resolution_ratio: float = 0.5,
        max_action_history: int = 200,
        max_history_images: int = 8,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 400,
        prompt_template: str = RL_PROMPT_TEMPLATE,
    ):
        self.processor = processor
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.max_history_images = max_history_images
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.prompt_template = prompt_template

        config = get_config(exp_config)
        with habitat.config.read_write(config):
            config.habitat.task.measurements.update({"collisions": CollisionsMeasurementConfig()})
        dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
        self.env = Env(config.habitat, dataset)

    def close(self) -> None:
        self.env.close()

    @property
    def num_episodes(self) -> int:
        return len(self.env.episodes)

    def rollout_episode(
        self,
        model,
        episode_index: int,
        generation_config: GenerationConfig,
        device: str,
    ) -> EpisodeRollout:
        self.env.current_episode = self.env.episodes[episode_index]
        observations = self.env.reset()
        episode_id = str(self.env.current_episode.episode_id)
        rgb_history: list[Image.Image] = []
        steps: list[RolloutStep] = []

        for step_idx in range(self.max_steps):
            rgb_history.append(self._observation_to_image(observations))
            if len(rgb_history) > self.max_action_history:
                rgb_history = rgb_history[-self.max_action_history :]

            prev_metrics = dict(self.env.get_metrics())
            prompt_messages = build_vln_prompt_messages(
                instruction=observations["instruction"]["text"],
                rgb_history=rgb_history,
                max_history_images=self.max_history_images,
                prompt_template=self.prompt_template,
            )
            response_text = self._generate_action_text(model, prompt_messages, generation_config, device)
            parsed = parse_action_text(response_text, self.forward_distance, self.turn_angle)

            observations = self.env.step({"action": parsed.action_id})
            next_metrics = dict(self.env.get_metrics())
            reward = compute_step_reward(
                prev_metrics=prev_metrics,
                next_metrics=next_metrics,
                action_id=parsed.action_id,
                action_valid=parsed.valid,
                cfg=self.reward_config,
            )
            steps.append(
                RolloutStep(
                    prompt_messages=prompt_messages,
                    response_text=_training_response(response_text, parsed),
                    action_id=parsed.action_id,
                    action_valid=parsed.valid,
                    reward=reward,
                    prev_metrics=prev_metrics,
                    next_metrics=next_metrics,
                )
            )
            if self.env.episode_over:
                break
            if step_idx == self.max_steps - 1:
                break

        final_metrics = dict(self.env.get_metrics())
        return EpisodeRollout(
            episode_id=episode_id,
            steps=steps,
            total_reward=float(sum(step.reward for step in steps)),
            success=float(final_metrics.get("success", 0.0)),
            spl=float(final_metrics.get("spl", 0.0)),
            distance_to_goal=float(final_metrics.get("distance_to_goal", 0.0)),
            path_length=float(final_metrics.get("path_length", 0.0)),
        )

    def _observation_to_image(self, observations) -> Image.Image:
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        return Image.fromarray(rgb.astype(np.uint8)).convert("RGB").resize((308, 252))

    def _generate_action_text(self, model, prompt_messages: list[dict], generation_config: GenerationConfig, device: str) -> str:
        text = _apply_chat_template(self.processor, prompt_messages, add_generation_prompt=True)
        imgs, _ = process_vision_info(prompt_messages)
        inputs = self.processor(text=[text], images=[imgs], return_tensors="pt", padding=True)
        inputs.to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=True,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()


def _training_response(response_text: str, parsed: ParsedAction) -> str:
    return response_text.strip() or parsed.text


def _apply_chat_template(processor, messages: list[dict], add_generation_prompt: bool) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)
