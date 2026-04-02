import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder import LatentUMRefDecoderModel
from model.latentum import LatentUMModel
from model.latentum.world_model import run_wm_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ckpt/latentum-base")
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--context-images", nargs=4, required=True)
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--output-dir", default="wm_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.actions) < 4:
        raise ValueError("--actions must contain at least 4 items: 3 observed actions and 1 future action")

    dtype = getattr(torch, args.dtype)
    print("===== Loading LatentUM =====")
    model = LatentUMModel.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    print("===== Loading Ref Decoder =====")
    decoder = LatentUMRefDecoderModel.from_pretrained(args.decoder_path, device=args.device, dtype=dtype)
    print("===== Running LatentUM-WM inference =====")
    result = run_wm_inference(
        model,
        decoder,
        context_images=args.context_images,
        actions=args.actions,
        output_dir=args.output_dir,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        show_progress=args.show_progress,
    )
    print(os.path.join(args.output_dir, "result.json"))
    print(f"generated {len(result['generated_images'])} image(s)")


if __name__ == "__main__":
    main()
