<div align="center">

# LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model

Jiachun Jin<sup>1</sup>, Zetong Zhou<sup>1</sup>, Xiao Yang<sup>2</sup>, Hao Zhang<sup>3</sup>, Pengfei Liu<sup>1</sup>, Jun Zhu<sup>2</sup>, Zhijie Deng<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University &nbsp;&nbsp; <sup>2</sup>Tsinghua University &nbsp;&nbsp; <sup>3</sup>UCSD

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](ARXIV_URL)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)](HUGGINGFACE_URL)
<!-- [![Project Page](https://img.shields.io/badge/Project-Page-blue)](PROJECT_PAGE_URL)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE) -->

</div>

---

## Overview

**LatentUM** unifies all modalities within a shared semantic latent space, enabling interleaved cross-modal reasoning without pixel-space mediation. Unlike existing unified models that require pixel decoding as a bridge between understanding and generation, LatentUM reasons directly over its own generated visual content.

## Key Features

- **Shared Semantic Latent Space:** Text and visual tokens share the same space, enabling direct cross-modal reasoning over generated visual content.
- **MBAQ:** Visual tokenizer trained to preserve VLM understanding behavior rather than pixel reconstruction.
- **MoME:** Decoupled understanding/generation branches with shared self-attention for cross-modal interaction.
- **Decoupled Pixel Decoder:** Optional diffusion decoder for pixel rendering, trained independently to keep the latent space semantics-focused.

## Getting Started

### Installation

```bash
git clone https://github.com/SJTU-DENG-Lab/LatentUM.git
cd LatentUM
uv sync
```

### Model Weights

Pre-trained weights are available on [HuggingFace](HUGGINGFACE_URL):

| Model | Base | Description | Download |
|-------|------|-------------|----------|
| LatentUM_Base | InternVL3.5-4B | Base model for understanding + generation | [Link](HUGGINGFACE_URL) |
| LatentUM_Vis-Gen | LatentUM_Base | Post-trained via GRPO for visual generation with self-reflection | [Link](HUGGINGFACE_URL) |
| LatentUM_Vis-Plan | LatentUM_Base | Fine-tuned for visual spatial planning | [Link](HUGGINGFACE_URL) |
| LatentUM_WM | LatentUM_Base | Fine-tuned for action-conditioned world modeling | [Link](HUGGINGFACE_URL) |

### Inference

```bash
# TODO
```

### Training

```bash
# TODO
```

## Citation

If you find this work useful, please cite:

```bibtex
# TODO
```

## Acknowledgements

We thank the authors of InternVL, BLIP3o, UniTok, and Stable Diffusion 3.5 for open-sourcing their models and data.

## License

This project is released under the [Apache 2.0 License](LICENSE).
