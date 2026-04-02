import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.latentum.configuration_latentum import LatentUMConfig


def build_export_config(source_config_path: str, source_checkpoint_path: str) -> LatentUMConfig:
    raw = OmegaConf.load(source_config_path)
    model_cfg = OmegaConf.to_container(raw.model, resolve=True)
    quantizer_cfg = OmegaConf.to_container(raw.model.quantizer, resolve=True)
    head_cfg = OmegaConf.to_container(raw.model.head, resolve=True)
    return LatentUMConfig(
        base_model_name_or_path=model_cfg["internvl_path"],
        quantizer_ckpt_path=quantizer_cfg["ckpt_path"],
        llm_hidden_size=model_cfg["llm_hidden_size"],
        mixture_mode=model_cfg["mixture_mode"],
        embedding_dim=model_cfg["embedding_dim"],
        image_size=raw.data.img_size,
        num_image_tokens=raw.data.num_img_token,
        model=model_cfg,
        quantizer=quantizer_cfg,
        head=head_cfg,
        legacy_checkpoint_path=source_checkpoint_path,
    )


def prefix_state_dict(state_dict, prefix: str):
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        default="/root/data1/jjc/ssd/experiment/prev/0110_sft_geneval_zimage/config.yaml",
    )
    parser.add_argument(
        "--source-checkpoint",
        default="/root/data1/jjc/ssd/experiment/prev/0110_sft_geneval_zimage/checkpoint-2404/pytorch_model/mp_rank_00_model_states.pt",
    )
    parser.add_argument("--output-dir", default="ckpt/latentum-base")
    parser.add_argument("--export-weights", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_export_config(args.source_config, args.source_checkpoint)
    config.save_pretrained(output_dir)
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        f.write('{\n  "cfg_scale": 3.0,\n  "temperature": 0.9,\n  "top_k": 50,\n  "top_p": 0.95\n}\n')

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.save_pretrained(output_dir)

    if not args.export_weights:
        return

    raw = torch.load(args.source_checkpoint, map_location="cpu", mmap=True)
    model_state = raw.get("module", raw)
    quantizer_state = torch.load(config.quantizer_ckpt_path, map_location="cpu", weights_only=True)
    merged_state = {}
    merged_state.update(prefix_state_dict(model_state, "internvl."))
    merged_state.update(prefix_state_dict(quantizer_state, "quantizer."))
    save_file(merged_state, str(output_dir / "model.safetensors"))
    legacy_quantizer_path = output_dir / "quantizer.safetensors"
    if legacy_quantizer_path.exists():
        legacy_quantizer_path.unlink()


if __name__ == "__main__":
    main()
