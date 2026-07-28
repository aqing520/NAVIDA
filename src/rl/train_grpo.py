import argparse
from datetime import datetime
from datetime import timedelta
import json
import os
import random
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model
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


def init_distributed(args) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    device_index = 0
    if distributed:
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count == 0:
            raise RuntimeError("Distributed RL training requires at least one visible CUDA device")
        if local_rank >= visible_gpu_count:
            raise RuntimeError(
                "NCCL DDP requires at most one rank per visible GPU. "
                f"Got LOCAL_RANK={local_rank} with {visible_gpu_count} visible GPUs."
            )
        device_index = local_rank
        torch.cuda.set_device(device_index)
        args.device = f"cuda:{device_index}"
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.ddp_timeout_seconds))
    return distributed, rank, local_rank, world_size, device_index


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def is_main_process(rank: int) -> bool:
    return rank == 0


def load_model_and_processor(args):
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    elif args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        )
        model = get_peft_model(model, lora_config)
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
    if hasattr(model, "print_trainable_parameters") and int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()
    return model, processor


def build_generation_config(args) -> GenerationConfig:
    return GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1,
        use_cache=True,
        repetition_penalty=args.repetition_penalty,
    )


def step_logprob_loss(
    model,
    processor,
    step: RolloutStep,
    advantage: float,
    ignore_token_ids: set[int],
    device: str,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, int, float, float]:
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
        return logits.sum() * 0.0, 0, 0.0, 0.0

    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    response_logprobs = token_logprobs.masked_select(valid_mask)
    mean_logprob = response_logprobs.mean()

    kl_loss = mean_logprob.new_tensor(0.0)
    if kl_beta > 0:
        ref_token_logprobs = reference_token_logprobs(model, batch, safe_labels, valid_mask)
        ref_response_logprobs = ref_token_logprobs.masked_select(valid_mask)
        log_ratio = ref_response_logprobs - response_logprobs
        kl_loss = (log_ratio.exp() - log_ratio - 1.0).mean()

    adv = torch.tensor(float(advantage), dtype=mean_logprob.dtype, device=mean_logprob.device)
    loss = -adv * mean_logprob + float(kl_beta) * kl_loss
    return (
        loss,
        int(valid_mask.sum().item()),
        float(mean_logprob.detach().float().cpu()),
        float(kl_loss.detach().float().cpu()),
    )


def reference_token_logprobs(model, batch: dict, safe_labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    base_model = unwrap_model(model)
    adapter_context = disable_adapter_context(base_model)
    with torch.no_grad(), adapter_context:
        ref_outputs = base_model(**batch)
        ref_logits = ref_outputs.logits[:, :-1, :]
        return F.log_softmax(ref_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def disable_adapter_context(model):
    model = unwrap_model(model)
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "disable_adapter"):
        return base_model.disable_adapter()
    raise ValueError("--kl-beta > 0 currently requires a PEFT LoRA model so adapters can be disabled for the reference policy")


def zero_trainable_loss(model) -> torch.Tensor:
    zero = None
    for param in model.parameters():
        if param.requires_grad:
            term = param.sum() * 0.0
            zero = term if zero is None else zero + term
    if zero is None:
        raise ValueError("No trainable parameters found")
    return zero


def normalize_group_returns(rollouts: list[EpisodeRollout], eps: float = 1e-6) -> list[float]:
    returns = np.array([rollout.total_reward for rollout in rollouts], dtype=np.float32)
    if len(returns) == 1:
        return [float(returns[0])]
    std = float(returns.std())
    if std < eps:
        return [0.0 for _ in returns]
    mean = float(returns.mean())
    return [float((value - mean) / (std + eps)) for value in returns]


def distributed_min_int(value: int, device: str) -> int:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def resolve_run_layout(args) -> tuple[str, str, str]:
    if args.run_name:
        run_name = args.run_name
    elif args.output_dir:
        run_name = os.path.basename(os.path.normpath(args.output_dir))
    else:
        run_name = datetime.now().strftime("rl_%Y%m%d_%H%M%S")

    output_dir = None
    if args.output_dir:
        output_dir = os.path.normpath(args.output_dir)
    result_dir = os.path.normpath(args.result_dir)

    if output_dir and (output_dir == result_dir or output_dir.startswith(result_dir + os.sep)):
        checkpoint_dir = output_dir
    else:
        checkpoint_dir = os.path.join(result_dir, run_name)

    log_path = os.path.join(args.log_dir, f"{run_name}.jsonl")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    return run_name, checkpoint_dir, log_path


def train(args) -> None:
    distributed, rank, local_rank, world_size, device_index = init_distributed(args)
    seed_all(args.seed + rank)
    run_name, checkpoint_dir, log_path = resolve_run_layout(args)
    if distributed:
        root, ext = os.path.splitext(log_path)
        log_path = f"{root}.rank{rank}{ext}"
    if is_main_process(rank):
        print(
            json.dumps(
                {
                    "event": "rl_run_layout",
                    "run_name": run_name,
                    "checkpoint_dir": checkpoint_dir,
                    "log_path": log_path,
                    "world_size": world_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "rl_rank_start",
                "rank": rank,
                "local_rank": local_rank,
                "device_index": device_index,
                "device": args.device,
                "log_path": log_path,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model, processor = load_model_and_processor(args)
    if distributed:
        model = DDP(model, device_ids=[device_index], output_device=device_index, find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ignore_token_ids = collect_ignore_label_token_ids(processor, unwrap_model(model).config)
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
    episode_indices = list(range(runner.num_episodes))
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    try:
        global_step = 0
        last_regular_save_step = 0
        for epoch in range(args.num_epochs):
            random.shuffle(episode_indices)
            if distributed:
                usable = (len(episode_indices) // world_size) * world_size
                local_episode_indices = episode_indices[:usable][rank::world_size]
            else:
                local_episode_indices = episode_indices
            for local_episode_pos, episode_index in enumerate(local_episode_indices):
                rollout_model = unwrap_model(model)
                rollout_model.eval()
                rollouts = [
                    runner.rollout_episode(rollout_model, episode_index, generation_config, args.device)
                    for _ in range(args.group_size)
                ]
                if distributed:
                    dist.barrier()
                advantages = normalize_group_returns(rollouts)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                num_tokens = 0
                logprobs = []
                kl_values = []
                rollout_steps = [
                    (step, advantage)
                    for rollout, advantage in zip(rollouts, advantages)
                    for step in rollout.steps
                ]
                max_backward_steps = len(rollout_steps)
                if distributed:
                    max_backward_steps = distributed_min_int(max_backward_steps, args.device)
                for backward_idx in range(max_backward_steps):
                    step, advantage = rollout_steps[backward_idx]
                    loss, valid_tokens, mean_logprob, kl_value = step_logprob_loss(
                        model=model,
                        processor=processor,
                        step=step,
                        advantage=advantage,
                        ignore_token_ids=ignore_token_ids,
                        device=args.device,
                        kl_beta=args.kl_beta,
                    )
                    if valid_tokens == 0:
                        loss = zero_trainable_loss(model)
                    else:
                        losses.append(float(loss.detach().float().cpu()))
                        num_tokens += valid_tokens
                        logprobs.append(mean_logprob)
                        kl_values.append(kl_value)
                    (loss / args.gradient_accumulation_steps).backward()
                    if (backward_idx + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                if max_backward_steps % args.gradient_accumulation_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                log_row = {
                    "global_step": global_step,
                    "rank": rank,
                    "world_size": world_size,
                    "epoch": epoch,
                    "local_episode_pos": local_episode_pos,
                    "episode_index": episode_index,
                    "episode_id": rollouts[0].episode_id if rollouts else None,
                    "group_returns": [rollout.total_reward for rollout in rollouts],
                    "group_advantages": advantages,
                    "success": [rollout.success for rollout in rollouts],
                    "spl": [rollout.spl for rollout in rollouts],
                    "distance_to_goal": [rollout.distance_to_goal for rollout in rollouts],
                    "num_steps": [rollout.primitive_steps for rollout in rollouts],
                    "num_blocks": [len(rollout.steps) for rollout in rollouts],
                    "loss_mean": float(np.mean(losses)) if losses else 0.0,
                    "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
                    "kl_mean": float(np.mean(kl_values)) if kl_values else 0.0,
                    "kl_beta": args.kl_beta,
                    "num_tokens": num_tokens,
                }
                append_jsonl(log_path, log_row)
                print(json.dumps(log_row, ensure_ascii=False), flush=True)

                regular_save_due = (
                    args.save_steps > 0
                    and global_step > 0
                    and global_step // args.save_steps > last_regular_save_step // args.save_steps
                )
                if regular_save_due:
                    if is_main_process(rank):
                        save_checkpoint(model, processor, checkpoint_dir, global_step)
                    last_regular_save_step = global_step
                if distributed:
                    dist.barrier()
                if args.max_updates is not None and global_step >= args.max_updates:
                    if is_main_process(rank):
                        save_checkpoint(model, processor, checkpoint_dir, global_step)
                    if distributed:
                        dist.barrier()
                    return
        if is_main_process(rank):
            save_checkpoint(model, processor, checkpoint_dir, global_step)
    finally:
        runner.close()
        cleanup_distributed(distributed)


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(model, processor, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrap_model(model).save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", default="config/vln_r2r_train.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--result-dir", default="result/rl")
    parser.add_argument("--log-dir", default="result/log")
    parser.add_argument("--run-name", default=None)
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
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=7200)
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
