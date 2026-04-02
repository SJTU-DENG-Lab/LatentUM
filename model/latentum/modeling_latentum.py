import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file

from model.decoder.reconstruct import sample_sd3_5, sample_sd3_5_ref
from model.decoder.sd_decoder import load_mmdit_half_trainable, load_mmdit_ref

from .image_utils import load_image
from .internvl.modeling_internvl_chat import InternVLChatModel
from .latent_generator import intern_gen
from .mixture import modify_internvl_to_mixture
from .quantizer_loader import build_quantizer
from .utils import disable_torch_init

from .configuration_latentum import LatentUMConfig, LatentUMDecoderConfig


def _to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    converted = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted[key] = _to_namespace(value)
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def _prefix_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def _extract_prefixed_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class LatentUMDecoderModel(nn.Module):
    def __init__(self, config: LatentUMDecoderConfig):
        super().__init__()
        self.config = config
        self._runtime_device = torch.device("cpu")
        self._runtime_dtype = torch.float32

        runtime_config = _to_namespace(config.to_dict())
        self.transformer = load_mmdit_half_trainable(runtime_config)
        self.vae = None
        self.noise_scheduler = None

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LatentUMDecoderModel":
        start_time = perf_counter()
        path = Path(path)
        config = LatentUMDecoderConfig.from_pretrained(path)
        model = cls(config)
        print(f"[Decoder] Initialized model skeleton in {perf_counter() - start_time:.2f}s")

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

        state_path = path / "model.safetensors"
        weights_start = perf_counter()
        if state_path.exists():
            model.transformer.load_state_dict(load_file(state_path), strict=True)
        else:
            raise FileNotFoundError(
                f"Missing decoder weights at {state_path}. "
                "Legacy checkpoint loading is disabled; please export decoder weights to model.safetensors first."
            )
        print(f"[Decoder] Loaded transformer weights in {perf_counter() - weights_start:.2f}s")

        aux_start = perf_counter()
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.sd3_5_path, subfolder="scheduler"
        )
        model.vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
        model.vae.requires_grad_(False)
        print(f"[Decoder] Loaded scheduler and VAE in {perf_counter() - aux_start:.2f}s")

        if device is not None or dtype is not None:
            move_start = perf_counter()
            model.to(device=device, dtype=dtype)
            print(f"[Decoder] Moved model to {device} ({dtype}) in {perf_counter() - move_start:.2f}s")
        print(f"[Decoder] Total load time: {perf_counter() - start_time:.2f}s")
        return model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) > 0 and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if len(args) > 1 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None:
            self._runtime_device = torch.device(device)
        if dtype is not None:
            self._runtime_dtype = dtype
        self.transformer = self.transformer.to(device=self._runtime_device, dtype=self._runtime_dtype)
        if self.vae is not None:
            self.vae = self.vae.to(device=self._runtime_device, dtype=self._runtime_dtype).eval()
        return self

    def decode(
        self,
        latents: torch.Tensor,
        *,
        seed: int | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        show_progress: bool = False,
    ) -> list[Image.Image]:
        if self.vae is None or self.noise_scheduler is None:
            raise RuntimeError("Decoder components are not initialized.")

        height = height or self.config.image_size
        width = width or self.config.image_size
        latents = latents.to(self._runtime_device, self._runtime_dtype)
        samples = sample_sd3_5(
            transformer         = self.transformer,
            vae                 = self.vae,
            noise_scheduler     = self.noise_scheduler,
            device              = self._runtime_device,
            dtype               = self._runtime_dtype,
            context             = latents,
            batch_size          = latents.shape[0],
            height              = height,
            width               = width,
            num_inference_steps = num_inference_steps,
            guidance_scale      = guidance_scale,
            seed                = seed,
            show_progress       = show_progress,
        )
        images = []
        for sample in samples:
            image = (sample.float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            images.append(Image.fromarray(image))
        return images

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        save_file(self.transformer.state_dict(), str(path / "model.safetensors"))


class LatentUMRefDecoderModel(LatentUMDecoderModel):
    def __init__(self, config: LatentUMDecoderConfig):
        nn.Module.__init__(self)
        if config.ref_mode != "concat":
            raise ValueError(
                f"LatentUMRefDecoderModel only supports ref_mode='concat', got {config.ref_mode!r}"
            )
        if config.num_ref_frames != 4:
            raise ValueError(
                f"LatentUMRefDecoderModel expects num_ref_frames=4, got {config.num_ref_frames}"
            )

        self.config = config
        self._runtime_device = torch.device("cpu")
        self._runtime_dtype = torch.float32
        runtime_config = _to_namespace(config.to_dict())
        self.transformer = load_mmdit_ref(runtime_config)
        self.vae = None
        self.noise_scheduler = None

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LatentUMRefDecoderModel":
        start_time = perf_counter()
        path = Path(path)
        config = LatentUMDecoderConfig.from_pretrained(path)
        model = cls(config)
        print(f"[RefDecoder] Initialized model skeleton in {perf_counter() - start_time:.2f}s")

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

        state_path = path / "model.safetensors"
        weights_start = perf_counter()
        if state_path.exists():
            model.transformer.load_state_dict(load_file(state_path), strict=True)
        else:
            raise FileNotFoundError(
                f"Missing decoder weights at {state_path}. "
                "Legacy checkpoint loading is disabled; please export decoder weights to model.safetensors first."
            )
        print(f"[RefDecoder] Loaded transformer weights in {perf_counter() - weights_start:.2f}s")

        aux_start = perf_counter()
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.sd3_5_path, subfolder="scheduler"
        )
        model.vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
        model.vae.requires_grad_(False)
        print(f"[RefDecoder] Loaded scheduler and VAE in {perf_counter() - aux_start:.2f}s")

        if device is not None or dtype is not None:
            move_start = perf_counter()
            model.to(device=device, dtype=dtype)
            print(f"[RefDecoder] Moved model to {device} ({dtype}) in {perf_counter() - move_start:.2f}s")
        print(f"[RefDecoder] Total load time: {perf_counter() - start_time:.2f}s")
        return model.eval()

    def _normalize_ref_images(self, ref_images: Sequence[str | Path | Image.Image] | torch.Tensor) -> torch.Tensor:
        expected_num_ref_frames = int(self.config.num_ref_frames)

        if isinstance(ref_images, torch.Tensor):
            ref_tensor = ref_images
            if ref_tensor.ndim == 4:
                if ref_tensor.shape[0] != expected_num_ref_frames:
                    raise ValueError(
                        f"ref_images must have {expected_num_ref_frames} frames, got {ref_tensor.shape[0]}"
                    )
                ref_tensor = ref_tensor.unsqueeze(0)
            elif ref_tensor.ndim != 5:
                raise ValueError(
                    "ref_images tensor must have shape (N, 3, H, W) or (B, N, 3, H, W)"
                )
            if ref_tensor.shape[1] != expected_num_ref_frames:
                raise ValueError(
                    f"ref_images tensor must contain {expected_num_ref_frames} frames, got {ref_tensor.shape[1]}"
                )
            return ref_tensor.to(device=self._runtime_device, dtype=self._runtime_dtype)

        if len(ref_images) != expected_num_ref_frames:
            raise ValueError(f"ref_images must contain exactly {expected_num_ref_frames} images.")

        import torchvision.transforms as transforms

        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
            ]
        )
        processed = []
        for image in ref_images:
            pil_image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
            processed.append(preprocess(pil_image))
        return torch.stack(processed, dim=0).unsqueeze(0).to(
            device=self._runtime_device,
            dtype=self._runtime_dtype,
        )

    def _encode_ref_images(self, ref_images: torch.Tensor, target_batch_size: int | None = None) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Decoder VAE is not initialized.")

        batch_size = ref_images.shape[0]
        if target_batch_size is not None:
            if batch_size == 1 and target_batch_size > 1:
                ref_images = ref_images.expand(target_batch_size, -1, -1, -1, -1)
                batch_size = target_batch_size
            elif batch_size != target_batch_size:
                raise ValueError(
                    f"ref_images batch size {batch_size} does not match target batch size {target_batch_size}"
                )

        num_ref_frames = ref_images.shape[1]
        if num_ref_frames != self.config.num_ref_frames:
            raise ValueError(
                f"Decoder expects {self.config.num_ref_frames} ref frames, got {num_ref_frames}"
            )

        _, _, _, height, width = ref_images.shape
        flat_ref_images = ref_images.reshape(batch_size * num_ref_frames, 3, height, width)
        ref_vae = self.vae.encode(flat_ref_images * 2 - 1).latent_dist.sample()
        ref_vae = (ref_vae - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_channels, latent_height, latent_width = ref_vae.shape[1:]
        ref_vae = ref_vae.reshape(batch_size, num_ref_frames, latent_channels, latent_height, latent_width)
        return ref_vae.reshape(batch_size, num_ref_frames * latent_channels, latent_height, latent_width)

    def decode(
        self,
        latents: torch.Tensor,
        *,
        ref_images: Sequence[str | Path | Image.Image] | torch.Tensor,
        seed: int | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        show_progress: bool = False,
    ) -> list[Image.Image]:
        del guidance_scale

        if self.vae is None or self.noise_scheduler is None:
            raise RuntimeError("Decoder components are not initialized.")

        height = height or self.config.image_size
        width = width or self.config.image_size
        latents = latents.to(self._runtime_device, self._runtime_dtype)
        ref_tensor = self._normalize_ref_images(ref_images)
        ref_vae = self._encode_ref_images(ref_tensor, target_batch_size=latents.shape[0])

        samples = sample_sd3_5_ref(
            transformer=self.transformer,
            vae=self.vae,
            noise_scheduler=self.noise_scheduler,
            device=self._runtime_device,
            dtype=self._runtime_dtype,
            context=latents,
            ref_vae=ref_vae,
            batch_size=latents.shape[0],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            show_progress=show_progress,
        )
        images = []
        for sample in samples:
            image = (sample.float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            images.append(Image.fromarray(image))
        return images


class LatentUMModel(nn.Module):
    def __init__(self, config: LatentUMConfig):
        super().__init__()
        self.config = config
        self._runtime_device = torch.device("cpu")
        self._runtime_dtype = torch.float32

        disable_torch_init()
        model_config = _to_namespace(config.model)
        quantizer_config = _to_namespace(config.quantizer)

        self.internvl = InternVLChatModel.from_pretrained(config.base_model_name_or_path)
        self.internvl = modify_internvl_to_mixture(self.internvl, model_config)
        self.quantizer = build_quantizer(quantizer_config)
        self.tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LatentUMModel":
        start_time = perf_counter()
        path = Path(path)
        config = LatentUMConfig.from_pretrained(path)
        model = cls(config)
        print(f"[LatentUM] Initialized model skeleton in {perf_counter() - start_time:.2f}s")

        model_path = path / "model.safetensors"

        weights_start = perf_counter()
        if model_path.exists():
            state_dict = load_file(model_path)
            internvl_state = _extract_prefixed_state_dict(state_dict, "internvl.")
            quantizer_state = _extract_prefixed_state_dict(state_dict, "quantizer.")
            if internvl_state:
                model.internvl.load_state_dict(internvl_state, strict=False)
            else:
                model.internvl.load_state_dict(state_dict, strict=False)
            if quantizer_state:
                model.quantizer.load_state_dict(quantizer_state, strict=True)
            elif config.quantizer_ckpt_path:
                state_dict = torch.load(config.quantizer_ckpt_path, map_location="cpu", weights_only=True)
                model.quantizer.load_state_dict(state_dict, strict=True)
            else:
                raise FileNotFoundError("Quantizer weights are missing from model.safetensors.")
        else:
            raise FileNotFoundError(
                f"Missing model weights at {model_path}. "
                "Legacy checkpoint loading is disabled; please run script/export_hf_checkpoint.py --export-weights first."
            )
        print(f"[LatentUM] Loaded weights in {perf_counter() - weights_start:.2f}s")

        from transformers import AutoTokenizer

        tokenizer_start = perf_counter()
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False,
        }
        tokenizer_source = path if (path / "tokenizer_config.json").exists() else config.base_model_name_or_path
        try:
            model.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
            print(f"[LatentUM] Loaded tokenizer from {tokenizer_source} with fix_mistral_regex=True")
        except AttributeError as exc:
            if "backend_tokenizer" not in str(exc):
                raise
            print(
                "[LatentUM] Tokenizer does not support fix_mistral_regex; "
                "falling back to default tokenizer loading."
            )
            model.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                **tokenizer_kwargs,
            )
        print(f"[LatentUM] Loaded tokenizer in {perf_counter() - tokenizer_start:.2f}s")
        if device is not None or dtype is not None:
            move_start = perf_counter()
            model.to(device=device, dtype=dtype)
            print(f"[LatentUM] Moved model to {device} ({dtype}) in {perf_counter() - move_start:.2f}s")
        print(f"[LatentUM] Total load time: {perf_counter() - start_time:.2f}s")
        return model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) > 0 and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if len(args) > 1 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None:
            self._runtime_device = torch.device(device)
        if dtype is not None:
            self._runtime_dtype = dtype
        self.internvl = self.internvl.to(device=self._runtime_device, dtype=self._runtime_dtype)
        self.quantizer = self.quantizer.to(device=self._runtime_device, dtype=self._runtime_dtype)
        return self

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        with open(path / "generation_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {"cfg_scale": 3.0, "temperature": 0.9, "top_k": 50, "top_p": 0.95},
                f,
                indent=2,
                ensure_ascii=False,
            )
        merged_state_dict = {}
        merged_state_dict.update(_prefix_state_dict(self.internvl.state_dict(), "internvl."))
        merged_state_dict.update(_prefix_state_dict(self.quantizer.state_dict(), "quantizer."))
        save_file(merged_state_dict, str(path / "model.safetensors"))
        legacy_quantizer_path = path / "quantizer.safetensors"
        if legacy_quantizer_path.exists():
            legacy_quantizer_path.unlink()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    def generate_latents(
        self,
        prompts: str | Iterable[str],
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 3.0,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        sample_logits: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        if seed is not None:
            torch.manual_seed(seed)

        sampling_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_logits": sample_logits,
        }
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        return intern_gen(
            self.internvl,
            self.quantizer,
            self.tokenizer,
            prompt_list,
            num_images_per_prompt,
            cfg_scale,
            sampling_kwargs,
            self._runtime_device,
            verbose=verbose,
        )

    def generate_images(
        self,
        prompts: str | Iterable[str],
        *,
        decoder: LatentUMDecoderModel | None = None,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 3.0,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        show_progress: bool = False,
    ) -> list[Image.Image]:
        if decoder is None:
            raise ValueError("A decoder is required for generate_images().")
        latents = self.generate_latents(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            verbose=show_progress,
        )
        z_q, _ = self.quantizer.indices_to_feature(latents.to(self._runtime_device))
        return decoder.decode(
            z_q,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.config.image_size,
            width=self.config.image_size,
            show_progress=show_progress,
        )

    def answer(
        self,
        image: str | Image.Image,
        question: str,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        pixel_values = load_image(
            image,
            input_size=self.config.image_size,
            max_num=self.config.max_num_patches,
        )
        if pixel_values is None:
            raise ValueError("Failed to load input image.")
        pixel_values = pixel_values.to(self._runtime_device, self._runtime_dtype)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        return self.internvl.chat(self.tokenizer, pixel_values, question, generation_config)
