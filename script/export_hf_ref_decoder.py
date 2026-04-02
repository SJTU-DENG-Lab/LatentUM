import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder.configuration_decoder import LatentUMDecoderConfig


def build_export_config(source_config_path: str, source_checkpoint_path: str) -> LatentUMDecoderConfig:
    raw = OmegaConf.load(source_config_path)
    return LatentUMDecoderConfig(
        sd3_5_path=raw.model.sd3_5_path,
        context_dim=raw.model.context_dim,
        load_pretrained=raw.model.load_pretrained,
        image_size=raw.data.img_size,
        ref_mode=getattr(raw.model, "ref_mode", "concat"),
        num_ref_frames=int(getattr(raw.model, "num_ref_frames", 4)),
        legacy_checkpoint_path=source_checkpoint_path,
    )


def load_export_state_dict(source_checkpoint_path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(source_checkpoint_path, map_location="cpu", mmap=True)
    if isinstance(raw, dict):
        for key in ("state_dict", "module"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format at {source_checkpoint_path}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        required=True,
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True,
    )
    parser.add_argument("--output-dir", default="ckpt/latentum-base/ref_decoder")
    parser.add_argument("--export-weights", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_export_config(args.source_config, args.source_checkpoint)
    if config.ref_mode != "concat":
        raise ValueError(f"Only ref_mode='concat' is supported, got {config.ref_mode!r}")
    if config.num_ref_frames != 4:
        raise ValueError(f"Expected num_ref_frames=4 for LatentUM-WM, got {config.num_ref_frames}")
    config.save_pretrained(output_dir)

    if not args.export_weights:
        return

    state_dict = load_export_state_dict(args.source_checkpoint)
    save_file(state_dict, str(output_dir / "model.safetensors"))


if __name__ == "__main__":
    main()
