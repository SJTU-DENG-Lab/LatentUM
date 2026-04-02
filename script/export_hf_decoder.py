import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder.configuration_decoder import LatentUMDecoderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        default="/root/data1/jjc/ssd/experiment/prev/0121_decoder_ft_highquality/config.yaml",
    )
    parser.add_argument(
        "--source-checkpoint",
        default="/root/data1/jjc/ssd/experiment/prev/0121_decoder_ft_highquality/model-sd_decoder-190000",
    )
    parser.add_argument("--output-dir", default="ckpt/latentum-base/decoder")
    parser.add_argument("--export-weights", action="store_true")
    args = parser.parse_args()

    raw = OmegaConf.load(args.source_config)
    config = LatentUMDecoderConfig(
        sd3_5_path=raw.model.sd3_5_path,
        context_dim=raw.model.context_dim,
        load_pretrained=raw.model.load_pretrained,
        image_size=raw.data.img_size,
        legacy_checkpoint_path=args.source_checkpoint,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)

    if not args.export_weights:
        return

    state_dict = torch.load(args.source_checkpoint, map_location="cpu", mmap=True)
    save_file(state_dict, str(output_dir / "model.safetensors"))


if __name__ == "__main__":
    main()
