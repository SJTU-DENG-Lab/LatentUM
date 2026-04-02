import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder import LatentUMDecoderModel
from model.latentum import LatentUMModel
from model.latentum.spatial_planning import run_frozenlake_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="FrozenLake visual spatial planning inference")
    parser.add_argument("--model-path", default="ckpt/latentum-vis-plan")
    parser.add_argument("--decoder-path", default="ckpt/latentum-base/decoder")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-text-tokens-per-step", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-dir", default="asset/frozen_lake_infer_release")
    parser.add_argument("--gif-duration", type=int, default=500)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print("===== Loading LatentUM planning model =====")
    model = LatentUMModel.from_pretrained(args.model_path, device=args.device, dtype=dtype)
    print("===== Loading decoder =====")
    decoder = LatentUMDecoderModel.from_pretrained(args.decoder_path, device=args.device, dtype=dtype)
    print("===== Running FrozenLake inference =====")
    run_frozenlake_inference(
        model,
        decoder,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        levels=args.levels,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        max_text_tokens_per_step=args.max_text_tokens_per_step,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        gif_duration=args.gif_duration,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
