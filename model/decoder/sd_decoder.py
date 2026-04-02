import os

import torch


def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    return x.permute(0, 2, 1, 3).contiguous()


def load_mmdit_half_trainable(config):
    from safetensors.torch import load_file

    from .sd_35.mmditx import MMDiTX

    device = torch.device("cpu")
    dtype = torch.bfloat16

    transformer = MMDiTX(
        input_size=None,
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=384,
        patch_size=2,
        in_channels=16,
        depth=24,
        num_patches=147456,
        adm_in_channels=2048,
        context_embedder_config={
            "target": "torch.nn.Linear",
            "params": {"in_features": config.context_dim, "out_features": 1536},
        },
        qk_norm="rms",
        x_block_self_attn_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        device=device,
        dtype=dtype,
        verbose=False,
    )

    if config.load_pretrained:
        ckpt = load_file(os.path.join(config.sd3_5_path, "sd3.5_medium.safetensors"))
        prefix = "model.diffusion_model."
        new_ckpt = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        del new_ckpt["context_embedder.weight"]
        transformer.load_state_dict(new_ckpt, strict=False)

    transformer.requires_grad_(False)
    transformer.context_embedder.requires_grad_(True)
    for name, param in transformer.named_parameters():
        if "context_block" in name:
            param.requires_grad_(True)

    return transformer


def load_mmdit_ref(config):
    from safetensors.torch import load_file

    from .sd_35.mmditx import MMDiTX
    from .sd_35.mmditx_ref import MMDiTXRef

    device = torch.device("cpu")
    dtype = torch.bfloat16

    mmdit_kwargs = {
        "input_size": None,
        "pos_embed_scaling_factor": None,
        "pos_embed_offset": None,
        "pos_embed_max_size": 384,
        "patch_size": 2,
        "in_channels": 16,
        "depth": 24,
        "num_patches": 147456,
        "adm_in_channels": 2048,
        "context_embedder_config": {
            "target": "torch.nn.Linear",
            "params": {"in_features": config.context_dim, "out_features": 1536},
        },
        "qk_norm": "rms",
        "x_block_self_attn_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "device": device,
        "dtype": dtype,
        "verbose": False,
    }

    base_transformer = MMDiTX(**mmdit_kwargs)
    if config.load_pretrained:
        ckpt = load_file(os.path.join(config.sd3_5_path, "sd3.5_medium.safetensors"))
        prefix = "model.diffusion_model."
        new_ckpt = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        del new_ckpt["context_embedder.weight"]
        base_transformer.load_state_dict(new_ckpt, strict=False)

    transformer = MMDiTXRef(
        ref_mode=config.ref_mode,
        num_ref_frames=config.num_ref_frames,
        **mmdit_kwargs,
    )
    ref_state = transformer.state_dict()
    for key, value in base_transformer.state_dict().items():
        if key.startswith("x_embedder."):
            continue
        ref_state[key] = value
    transformer.load_state_dict(ref_state, strict=True)

    with torch.no_grad():
        transformer.x_embedder.proj.weight[:, :16, :, :] = base_transformer.x_embedder.proj.weight
        transformer.x_embedder.proj.bias.copy_(base_transformer.x_embedder.proj.bias)

    transformer.requires_grad_(False)
    return transformer
