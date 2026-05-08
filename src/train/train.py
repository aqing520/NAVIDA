import os
os.environ["HF_DATASETS_CACHE"] = "./.cache/huggingface_cache"
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import logging
import transformers
from transformers import (
    set_seed, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    TrainingArguments,
    TrainerCallback,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_dataset
import sys
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

processor = None


def _safe_batch_meta(example):
    first_conv = example.get("conversations", [{}])[0]
    image_field = first_conv.get("image")
    if isinstance(image_field, list):
        num_images = len(image_field)
    elif image_field:
        num_images = 1
    else:
        num_images = 0

    sample_id = (
        example.get("id")
        or example.get("episode_id")
        or example.get("video")
        or "NA"
    )
    task_type = example.get("task type", "NA")
    answer = example.get("conversations", [{}])[-1].get("value", "")
    return task_type, sample_id, num_images, repr(answer[:120])


def check_model_finite(model, step, rank, max_report=20):
    bad = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        data = p.data
        if data is not None and torch.is_floating_point(data):
            if not torch.isfinite(data).all():
                bad.append(name)
                if len(bad) >= max_report:
                    break
    if bad:
        print(
            f"[BAD_PARAM_BEFORE_FORWARD][rank={rank}][step={step}] {bad}",
            flush=True,
        )
        raise RuntimeError("Model parameters already contain NaN/Inf before forward")


def resolve_torch_dtype(dtype_name: Optional[str]):
    if dtype_name in (None, "auto"):
        return dtype_name
    return getattr(torch, dtype_name)


def log_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
    ratio = 100 * trainable_params / total_params if total_params else 0
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        ratio,
    )


def freeze_linear_attention_modules(model):
    frozen_params = 0
    frozen_modules = set()
    for name, param in model.named_parameters():
        if ".linear_attn." in name:
            param.requires_grad = False
            frozen_params += param.numel()
            module_name = name.rsplit(".", 1)[0]
            frozen_modules.add(module_name)
    logger.info(
        "Frozen linear_attn params: %s across %s modules",
        f"{frozen_params:,}",
        len(frozen_modules),
    )

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Dataset name."})

@dataclass
class ModelArguments:
    model_name_or_path: str = None
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    

class LogCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(logs)


class DebugTrainer(Trainer):
    def __init__(self, *args, debug_raw_loss: bool = False, debug_raw_loss_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_raw_loss = debug_raw_loss
        self.debug_raw_loss_steps = debug_raw_loss_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rank = os.environ.get("LOCAL_RANK", "NA")
        step = self.state.global_step
        if self.debug_raw_loss:
            check_model_finite(model, step, rank)

        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        labels = inputs.get("labels", None)
        valid = (labels != -100).sum().item() if labels is not None else -1

        if self.debug_raw_loss and step < self.debug_raw_loss_steps:
            logger.info(
                "RAW_LOSS rank=%s step=%s value=%.8f valid_labels=%s",
                rank,
                step,
                float(loss.detach().float().cpu()),
                valid,
            )

        if self.debug_raw_loss and not torch.isfinite(loss.detach()):
            print(f"[NAN_LOSS][rank={rank}][step={step}]", flush=True)
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if torch.is_floating_point(v):
                        finite = torch.isfinite(v).all().item()
                        msg = (
                            f"[INPUT][rank={rank}][step={step}] "
                            f"{k} shape={tuple(v.shape)} dtype={v.dtype} finite={finite}"
                        )
                        if finite and v.numel() > 0:
                            msg += f" min={v.min().item()} max={v.max().item()}"
                        print(msg, flush=True)
                    else:
                        print(
                            f"[INPUT][rank={rank}][step={step}] "
                            f"{k} shape={tuple(v.shape)} dtype={v.dtype}",
                            flush=True,
                        )
            raise RuntimeError("NaN raw loss detected")

        return (loss, outputs) if return_outputs else loss


def uniform_sample_with_ends(data, n):
    # n > 2
    if len(data) <= n:
        return data

    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]

def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

    for i in range(0,len(example["conversations"]),2):
        if not isinstance(example["conversations"][i]["image"],list):
            example["conversations"][i]["image"] = [example["conversations"][i]["image"]]

        content = []

        if example['task type'] == 'vln':
            content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
            if len(example["conversations"][i]["image"]) > 1:
                content.extend([{"type": "image", "image": item} for item in uniform_sample_with_ends(example["conversations"][i]["image"][:-1],8)])
            else:
                content.append({"type": "image", "image": example["conversations"][i]["image"][-1]} )
            content.append({"type": "text", "text": 'and an image of the current observation'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][-1]})
            item = example["conversations"][i]["value"].split('current observation')
            content.append({"type": "text", "text": item[1]})
        elif example['task type'] == 'trajectory summarization':
            content.append({"type": "text", "text": 'Assume you are a robot designed for navigation. You are provided with captured images sequences'})
            content.extend([{"type": "image", "image": item} for item in uniform_sample_with_ends(example["conversations"][i]["image"],8)])
            item = example["conversations"][i]["value"].split('images sequences')
            content.append({"type": "text", "text": item[1]})
        elif example['task type'] == 'idm':
            content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given an image of current view'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][0]} )
            content.append({"type": "text", "text": 'and an image of the goal view'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][1]} )
            item = example["conversations"][i]["value"].split('goal view. ')
            content.append({"type": "text", "text": item[1]})
        else:
            raise NotImplementedError
        
        messages.append({
                    "role": "user",
                    "content": content
                })

        messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["conversations"][i+1]["value"]},
                        ]
                })
    
    example["messages"] = messages
    return example



def collate_fn(examples):
    texts = []
    prompt_texts = []
    image_inputs = []
    debug_raw_loss = os.environ.get("NAVIDA_DEBUG_RAW_LOSS", "0") == "1"

    for example in examples:
        messages = convert_example(example)["messages"]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        prompt_texts.append(
            processor.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        imgs, vids = process_vision_info(example["messages"])
        imgs = [item.resize((308,252)) for item in imgs]
        image_inputs.append(imgs)

    if debug_raw_loss:
        rank = os.environ.get("LOCAL_RANK", "NA")
        batch_meta = [_safe_batch_meta(example) for example in examples]
        print(
            f"[BATCH_META][rank={rank}] "
            f"tasks={[m[0] for m in batch_meta]} "
            f"ids={[m[1] for m in batch_meta]} "
            f"num_images={[m[2] for m in batch_meta]} "
            f"answer={[m[3] for m in batch_meta]}",
            flush=True,
        )

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    prompt_batch = processor(
        text=prompt_texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100

    # Mask the full prompt prefix and train only on assistant continuation tokens.
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    for label, prompt_len in zip(labels, prompt_lengths):
        label[:prompt_len] = -100

    valid_label_counts = (labels != -100).sum(dim=1)
    if (valid_label_counts == 0).any():
        rank = os.environ.get("LOCAL_RANK", "NA")
        print(f"[ZERO_LABEL][rank={rank}] {valid_label_counts.tolist()}", flush=True)
        raise RuntimeError("zero supervised tokens")

    batch["labels"] = labels

    for k, v in batch.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            if not torch.isfinite(v).all():
                rank = os.environ.get("LOCAL_RANK", "NA")
                print(f"[BAD_INPUT_IN_COLLATE][rank={rank}] key={k}", flush=True)
                raise RuntimeError(f"Bad tensor in collate: {k}")

    return batch




def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, f"train.log"))
        file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S", )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)

    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")


    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = load_dataset("json", data_files=data_args.dataset_name)
    dataset = dataset.shuffle(seed=42)

    global processor
    processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, use_fast=False
        )
    processor.tokenizer.padding_side = "right"


    logger.info("Using AutoProcessor for VLM model.")

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
            torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
            )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
    )

    model = model.to(torch_dtype)
    # set accepts_loss_kwargs for loss scaler bug when setting gradient_accumulation_steps > 1
    model.accepts_loss_kwargs = False

    ###################
    #  (Optional) Frozen vision encoder
    ###################
    visual_module = getattr(model, "visual", None)
    if visual_module is None:
        visual_module = getattr(getattr(model, "model", None), "visual", None)
    if visual_module is not None:
        for p in visual_module.parameters():
            p.requires_grad = False
        if hasattr(visual_module, "merger"):
            for p in visual_module.merger.parameters():
                p.requires_grad = True

    freeze_linear_attention_modules(model)

    # model.enable_input_require_grads() # important when using adapter
    logger.info(f"*** Model in {torch_dtype}***")
    log_trainable_parameters(model)
    

    ############################
    # Initialize the NaVIDA Trainer
    ############################
    # training_args.dataset_kwargs = {
    #     "skip_prepare_dataset": True,
    # }
    training_args.remove_unused_columns = False
    debug_raw_loss = os.environ.get("NAVIDA_DEBUG_RAW_LOSS", "0") == "1"
    debug_raw_loss_steps = int(os.environ.get("NAVIDA_DEBUG_RAW_LOSS_STEPS", "0"))
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=collate_fn,
        debug_raw_loss=debug_raw_loss,
        debug_raw_loss_steps=debug_raw_loss_steps,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset": data_args.dataset_name,
        "tags": ["NaVIDA Training"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
