from typing import List, Optional

import torch

from .mmditx import MMDiTX


class MMDiTXRef(MMDiTX):
    def __init__(self, ref_mode: str = "concat", num_ref_frames: int = 1, **kwargs):
        if ref_mode != "concat":
            raise ValueError(f"Unsupported ref_mode: {ref_mode!r}. Only 'concat' is implemented.")

        self.ref_mode = ref_mode
        self.num_ref_frames = int(num_ref_frames)
        self.orig_in_channels = int(kwargs.get("in_channels", 16))
        kwargs["in_channels"] = self.orig_in_channels * (1 + self.num_ref_frames)
        kwargs.setdefault("out_channels", self.orig_in_channels)
        super().__init__(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        ref_vae: Optional[torch.Tensor] = None,
        multi_modal_context: bool = False,
        controlnet_hidden_states: Optional[torch.Tensor] = None,
        skip_layers: Optional[List] = [],
    ) -> torch.Tensor:
        if ref_vae is None:
            raise ValueError("ref_vae is required for MMDiTXRef.")

        expected_channels = self.orig_in_channels * self.num_ref_frames
        if ref_vae.shape[1] != expected_channels:
            raise ValueError(
                f"ref_vae channels mismatch: expected {expected_channels}, got {ref_vae.shape[1]}"
            )
        if ref_vae.shape[0] != x.shape[0]:
            raise ValueError(f"ref_vae batch mismatch: expected {x.shape[0]}, got {ref_vae.shape[0]}")
        if ref_vae.shape[2:] != x.shape[2:]:
            raise ValueError(f"ref_vae spatial mismatch: expected {x.shape[2:]}, got {ref_vae.shape[2:]}")

        x = torch.cat([x, ref_vae], dim=1)
        return super().forward(
            x=x,
            t=t,
            y=y,
            context=context,
            multi_modal_context=multi_modal_context,
            controlnet_hidden_states=controlnet_hidden_states,
            skip_layers=skip_layers,
        )
