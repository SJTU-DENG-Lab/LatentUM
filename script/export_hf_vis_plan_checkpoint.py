import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _extract_state_dict(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", mmap=True)
    if "state_dict" in raw:
        return raw["state_dict"]
    if "module" in raw:
        return raw["module"]
    return raw


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _prefix_internvl_if_needed(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("internvl.") or key.startswith("quantizer.") for key in state_dict):
        return state_dict
    return {f"internvl.{key}": value for key, value in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", default="ckpt/latentum-base")
    parser.add_argument("--lang-checkpoint", default=None)
    parser.add_argument("--vis-checkpoint", required=True)
    parser.add_argument("--output-dir", default="ckpt/latentum-vis-plan")
    args = parser.parse_args()

    base_dir = Path(args.base_model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    ]:
        _copy_if_exists(base_dir / name, output_dir / name)

    base_state = load_file(base_dir / "model.safetensors")
    merged_state = dict(base_state)

    if args.lang_checkpoint:
        merged_state.update(_prefix_internvl_if_needed(_extract_state_dict(args.lang_checkpoint)))
    merged_state.update(_prefix_internvl_if_needed(_extract_state_dict(args.vis_checkpoint)))

    save_file(merged_state, str(output_dir / "model.safetensors"))


if __name__ == "__main__":
    main()
