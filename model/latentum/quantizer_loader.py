import torch

from .quantizer.multi_vq import get_multi_vq_quantizer


def build_quantizer(config):
    vq_type = getattr(config, "vq_type", "multi_vq")
    if vq_type == "multi_vq":
        return get_multi_vq_quantizer(config)
    raise ValueError(f"Unsupported VQ type for open-source build: {vq_type}")


def load_quantizer(config):
    quantizer = build_quantizer(config)
    if config.ckpt_path is None:
        raise ValueError("ckpt_path is required")
    ckpt = torch.load(config.ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer.requires_grad_(False)
    return quantizer
