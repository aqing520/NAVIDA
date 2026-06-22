import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

from src.train.train import collect_ignore_label_token_ids, freeze_linear_attention_modules, resolve_torch_dtype
from src.rl.rewards import RewardConfig
from src.rl.rollout import EpisodeRollout, RolloutStep, VlnRlRolloutRunner, _apply_chat_template


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_processor(args):
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    model = model.to(args.device)
    model.accepts_loss_kwargs = False

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    processor.tokenizer.padding_side = "right"
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels

    if args.freeze_vision:
        visual_module = getattr(model, "visual", None)
        if visual_module is None:
            visual_module = getattr(getattr(model, "model", None), "visual", None)
        if visual_module is not None:
            for param in visual_module.parameters():
                param.requires_grad = False
            if hasattr(visual_module, "merger"):
                for param in visual_module.merger.parameters():
                    param.requires_grad = True
    if args.freeze_linear_attention:
        freeze_linear_attention_modules(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, processor


def build_generation_config(args) -> GenerationConfig:
    return GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1,
        use_cache=not args.gradient_checkpointing,
        repetition_penalty=args.repetition_penalty,
    )


def step_logprob_loss(
    model,
    processor,
    step: RolloutStep,
    advantage: float,
    ignore_token_ids: set[int],
    device: str,
) -> tuple[torch.Tensor, int, float]:
    prompt_text = _apply_chat_template(processor, step.prompt_messages, add_generation_prompt=True)
    full_messages = step.prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": step.response_text}]}
    ]
    full_text = _apply_chat_template(processor, full_messages, add_generation_prompt=False)

    imgs, _ = process_vision_info(step.prompt_messages)
    batch = processor(text=[full_text], images=[imgs], return_tensors="pt", padding=True)
    prompt_batch = processor(text=[prompt_text], images=[imgs], return_tensors="pt", padding=True)
    batch.to(device)
    prompt_batch.to(device)

    labels = batch["input_ids"].clone()
    prompt_len = int(prompt_batch["attention_mask"].sum(dim=1).item())
    labels[:, :prompt_len] = -100
    for token_id in ignore_token_ids:
        labels[labels == token_id] = -100

    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels != -100
    if valid_mask.sum().item() == 0:
        return logits.sum() * 0.0, 0, 0.0

    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mean_logprob = token_logprobs.masked_select(valid_mask).mean()
    adv = torch.tensor(float(advantage), dtype=mean_logprob.dtype, device=mean_logprob.device)
    loss = -adv * mean_logprob
    return loss, int(valid_mask.sum().item()), float(mean_logprob.detach().float().cpu())


def normalize_group_returns(rollouts: list[EpisodeRollout], eps: float = 1e-6) -> list[float]:
    returns = np.array([rollout.total_reward for rollout in rollouts], dtype=np.float32)
    if len(returns) == 1:
        return [float(returns[0])]
    std = float(returns.std())
    if std < eps:
        return [0.0 for _ in returns]
    mean = float(returns.mean())
    return [float((value - mean) / (std + eps)) for value in returns]


def train(args) -> None:
    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model, processor = load_model_and_processor(args)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ignore_token_ids = collect_ignore_label_token_ids(processor, model.config)
    reward_config = RewardConfig(
        distance_delta_scale=args.distance_delta_scale,
        success_bonus=args.success_bonus,
        wrong_stop_penalty=args.wrong_stop_penalty,
        invalid_action_penalty=args.invalid_action_penalty,
        collision_penalty=args.collision_penalty,
        step_penalty=args.step_penalty,
        goal_distance=args.goal_distance,
    )
    runner = VlnRlRolloutRunner(
        exp_config=args.exp_config,
        processor=processor,
        forward_distance=args.forward_distance,
        turn_angle=args.turn_angle,
        resolution_ratio=args.resolution_ratio,
        max_action_history=args.max_action_history,
        max_history_images=args.max_history_images,
        reward_config=reward_config,
        max_steps=args.max_steps,
    )
    generation_config = build_generation_config(args)
    log_path = os.path.join(args.output_dir, "rl_train_log.jsonl")
    episode_indices = list(range(runner.num_episodes))
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    try:
        global_step = 0
        for epoch in range(args.num_epochs):
            random.shuffle(episode_indices)
            for episode_index in episode_indices:
                model.eval()
                rollouts = [
                    runner.rollout_episode(model, episode_index, generation_config, args.device)
                    for _ in range(args.group_size)
                ]
                advantages = normalize_group_returns(rollouts)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                num_tokens = 0
                logprobs = []
                for rollout, advantage in zip(rollouts, advantages):
                    for step in rollout.steps:
                        loss, valid_tokens, mean_logprob = step_logprob_loss(
                            model=model,
                            processor=processor,
                            step=step,
                            advantage=advantage,
                            ignore_token_ids=ignore_token_ids,
                            device=args.device,
                        )
                        if valid_tokens == 0:
                            continue
                        (loss / args.gradient_accumulation_steps).backward()
                        losses.append(float(loss.detach().float().cpu()))
                        num_tokens += valid_tokens
                        logprobs.append(mean_logprob)
                        if len(losses) % args.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1

                if losses and len(losses) % args.gradient_accumulation_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                log_row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "episode_index": episode_index,
                    "episode_id": rollouts[0].episode_id if rollouts else None,
                    "group_returns": [rollout.total_reward for rollout in rollouts],
                    "group_advantages": advantages,
                    "success": [rollout.success for rollout in rollouts],
                    "spl": [rollout.spl for rollout in rollouts],
                    "distance_to_goal": [rollout.distance_to_goal for rollout in rollouts],
                    "num_steps": [len(rollout.steps) for rollout in rollouts],
                    "loss_mean": float(np.mean(losses)) if losses else 0.0,
                    "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
                    "num_tokens": num_tokens,
                }
                append_jsonl(log_path, log_row)
                print(json.dumps(log_row, ensure_ascii=False), flush=True)

                if global_step > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model, processor, args.output_dir, global_step)
                if args.max_updates is not None and global_step >= args.max_updates:
                    save_checkpoint(model, processor, args.output_dir, global_step)
                    return
        save_checkpoint(model, processor, args.output_dir, global_step)
    finally:
        runner.close()


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(model, processor, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", default="config/vln_r2r_train.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-linear-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--resolution-ratio", type=float, default=0.5)
    parser.add_argument("--max-action-history", type=int, default=200)
    parser.add_argument("--max-history-images", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--distance-delta-scale", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--wrong-stop-penalty", type=float, default=-3.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=-0.5)
    parser.add_argument("--collision-penalty", type=float, default=-0.2)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--goal-distance", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
