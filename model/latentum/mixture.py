import torch
import torch.nn as nn

from .generation.autoregressive_head import AutoregressiveHead
from .internvl.mot import make_internvl_mot


def modify_internvl_to_mixture(internvl, config):
    visual_projector = nn.Sequential(
        nn.Linear(config.embedding_dim, config.llm_hidden_size),
        nn.GELU(),
        nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
    )
    visual_projector.requires_grad_(True)

    if config.mixture_mode == "mot":
        internvl = make_internvl_mot(internvl)
    else:
        raise ValueError(f"Unsupported mixture mode for open-source build: {config.mixture_mode}")

    train_language_model = getattr(config, "train_language_model", False)
    if train_language_model:
        for name, param in internvl.language_model.named_parameters():
            if "_vision" not in name:
                param.requires_grad = True

    internvl.visual_projector = visual_projector

    if getattr(config.head, "type", "ar") == "linear":
        linear_head = nn.Linear(config.head.input_dim, config.head.output_dim)
        linear_head.requires_grad_(True)
        internvl.visual_embeddings = torch.nn.Embedding(2 ** 14, config.llm_hidden_size)
        internvl.linear_head = linear_head
    else:
        ar_head = AutoregressiveHead(config.head)
        ar_head.requires_grad_(True)
        internvl.ar_head = ar_head

    return internvl
