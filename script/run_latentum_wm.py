import torch

from model.decoder import LatentUMRefDecoderModel
from model.latentum import LatentUMModel
from model.latentum.world_model import run_wm_inference


MODEL_PATH = "ckpt/latentum-wm"
DECODER_PATH = "ckpt/latentum-wm/decoder"
OUTPUT_DIR = "wm_output"

# Need 4 frames as context
CONTEXT_IMAGES = [
    "asset/recon_0.jpg",
    "asset/recon_1.jpg",
    "asset/recon_2.jpg",
    "asset/recon_3.jpg",
]

# Need 3 actions as context
# Action(dx, dy, dyaw)
# dx ~ N(0.25, 0.04^2)m     '+' means forward
# dy ~ N(0, 0.02^2)m        '+' means left
# dyaw ~ N(0, 8^2)degree    '+' means turn left
ACTIONS = [
    "Action(0.23, 0.01, 5)",
    "Action(0.22, 0.01, 6)",
    "Action(0.23, 0.02, 7)",
    
    "Action(0.23, 0.01, 7)",
    "Action(0.24, -0.02, -2)",
    "Action(0.20, -0.02, -5)",
    "Action(0.22, 0.02, 3)",
    "Action(0.21, 0.03, 10)",
    "Action(0.21, 0.02, 12)",
    "Action(0.21, 0.02, 10)",
    "Action(0.23, 0.01, 10)",
    "Action(0.23, -0.01, 1)",
    "Action(0.22, -0.02, -4)",
    "Action(0.20, 0.0, -3)",
    "Action(0.19, -0.01, -3)",
    "Action(0.25, -0.02, -6)",
    "Action(0.26, -0.01, -5)",
    "Action(0.24, -0.01, -5)",
    "Action(0.24, 0.02, 2)",
    "Action(0.27, 0.04, 9)",
    "Action(0.24, 0.02, 7)",
    "Action(0.23, 0.01, 6)",
    "Action(0.20, 0.01, 6)",
    "Action(0.25, 0.0, 2)",
    "Action(0.31, -0.03, -5)",
    "Action(0.24, -0.01, -5)",
    "Action(0.30, -0.02, -8)",
    "Action(0.24, -0.02, -8)",
    "Action(0.23, -0.02, -9)",
    "Action(0.25, -0.03, -11)",
    "Action(0.25, -0.03, -13)",
    "Action(0.20, -0.01, -9)",
    "Action(0.24, -0.01, -8)",
    "Action(0.26, 0.03, 1)",
    "Action(0.24, 0.04, 8)",
    "Action(0.21, 0.01, 7)",
    "Action(0.20, 0.01, 7)",
    "Action(0.21, 0.0, 5)",
    "Action(0.25, 0.0, 3)",
]

SEED = 42
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
SHOW_PROGRESS = True


def main() -> None:
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LatentUMModel.from_pretrained(
        MODEL_PATH,
        device=device,
        dtype=dtype,
    )
    decoder = LatentUMRefDecoderModel.from_pretrained(
        DECODER_PATH,
        device=device,
        dtype=dtype,
    )
    result = run_wm_inference(
        model,
        decoder,
        context_images=CONTEXT_IMAGES,
        actions=ACTIONS,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        show_progress=SHOW_PROGRESS,
    )
    print(f"saved to {OUTPUT_DIR}")
    print(f"result json: {OUTPUT_DIR}/result.json")
    print(f"generated images: {len(result['generated_images'])}")


if __name__ == "__main__":
    main()
