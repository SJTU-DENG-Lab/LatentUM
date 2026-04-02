import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder import LatentUMDecoderModel
from model.latentum import LatentUMModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ckpt/latentum-base")
    parser.add_argument("--decoder-path", default="ckpt/latentum-base/decoder")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print("===== Loading LatentUM =====")
    model = LatentUMModel.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    print("===== Loading Decoder =====")
    decoder = LatentUMDecoderModel.from_pretrained(args.decoder_path, device=args.device, dtype=dtype)
    print("===== start to generate image =====")
    images = model.generate_images(args.prompt, decoder=decoder)
    images[0].save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
