import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a partial fine-tuned Qwen3.5 checkpoint into a full base model."
    )
    parser.add_argument("--base-model", required=True, help="Path to the full base model directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to the partial checkpoint directory.")
    parser.add_argument("--output", required=True, help="Path to write the merged model directory.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype used when loading the base model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype = getattr(torch, args.torch_dtype)
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_file = checkpoint_dir / "model.safetensors"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print(f"Loading checkpoint weights from {checkpoint_file}")
    state_dict = load_file(str(checkpoint_file))
    load_result = model.load_state_dict(state_dict, strict=False)

    missing_keys = sorted(load_result.missing_keys)
    unexpected_keys = sorted(load_result.unexpected_keys)

    print(f"Loaded {len(state_dict)} tensors from checkpoint")
    print(f"Missing keys after merge: {len(missing_keys)}")
    if missing_keys:
        print("First 20 missing keys:")
        for key in missing_keys[:20]:
            print(f"  {key}")
    print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
    if unexpected_keys:
        print("First 20 unexpected keys:")
        for key in unexpected_keys[:20]:
            print(f"  {key}")

    print(f"Saving merged model to {output_dir}")
    state_dict = model.state_dict()
    exported_state = {}
    seen_ptrs = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen_ptrs:
            tensor = tensor.clone()
        else:
            seen_ptrs[ptr] = key
        exported_state[key] = tensor.contiguous()
    save_file(exported_state, str(output_dir / "model.safetensors"))

    config_dict = model.config.to_dict()
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)

    print(f"Saving processor to {output_dir}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print("Merge complete")


if __name__ == "__main__":
    main()
