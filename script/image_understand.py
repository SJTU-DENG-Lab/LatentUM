import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.latentum import LatentUMModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ckpt/latentum-base")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model = LatentUMModel.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    print(model.answer(args.image, args.question))


if __name__ == "__main__":
    main()
