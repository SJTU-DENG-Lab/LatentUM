import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import trange

from .modeling_latentum import LatentUMModel, LatentUMRefDecoderModel

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

WM_INITIAL_USER_PROMPT_TEMPLATE = """You are a navigation agent predicting future first-person observations. Here is the trajectory you have traveled so far:

{context_block}

Given this history, predict the next observation if you {action_text} over 0.25 seconds. Return exactly one image."""

ACTION_RE = re.compile(
    r"^Action\(\s*(?:(?:dx\s*=\s*)?([-+]?\d+(?:\.\d+)?))\s*,\s*"
    r"(?:(?:dy\s*=\s*)?([-+]?\d+(?:\.\d+)?))\s*,\s*"
    r"(?:(?:dyaw\s*=\s*)?([-+]?\d+(?:\.\d+)?))\s*\)$",
    re.IGNORECASE,
)


@dataclass
class WMRolloutStep:
    step_idx: int
    action_input: str
    action_text: str
    codes: torch.Tensor
    image: Image.Image


@dataclass
class WMRolloutResult:
    normalized_actions: list[str]
    observed_action_texts: list[str]
    future_action_texts: list[str]
    steps: list[WMRolloutStep]

    @property
    def generated_images(self) -> list[Image.Image]:
        return [step.image for step in self.steps]


def _format_action_scalar(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}"


def _format_action_text(dx: float, dy: float, dyaw: float) -> str:
    return (
        f"move {_format_action_scalar(dx)} meters along the X-axis and "
        f"{_format_action_scalar(dy)} meters along the Y-axis, rotating by "
        f"{_format_action_scalar(dyaw)} degrees"
    )


def _parse_structured_action(action: str) -> tuple[float, float, float] | None:
    match = ACTION_RE.fullmatch(action.strip())
    if match is None:
        return None
    dx, dy, dyaw = match.groups()
    return float(dx), float(dy), float(dyaw)


def normalize_action_text(action: str) -> str:
    normalized = action.strip()
    if not normalized:
        raise ValueError("Action text must not be empty.")
    parsed = _parse_structured_action(normalized)
    if parsed is None:
        return normalized
    return _format_action_text(*parsed)


def normalize_action_list(actions: Sequence[str]) -> list[str]:
    return [normalize_action_text(action) for action in actions]


def _build_image_placeholder(num_image_tokens: int) -> str:
    return IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens + IMG_END_TOKEN


def _build_context_block(
    num_context_images: int,
    observed_action_texts: Sequence[str],
    num_image_tokens: int,
) -> str:
    if num_context_images != 4:
        raise ValueError(f"WM inference expects 4 context images, got {num_context_images}")
    expected_actions = num_context_images - 1
    if len(observed_action_texts) != expected_actions:
        raise ValueError(
            f"Expected {expected_actions} observed actions for {num_context_images} context images, "
            f"got {len(observed_action_texts)}"
        )

    placeholder = _build_image_placeholder(num_image_tokens)
    parts = [f"First observation: {placeholder}"]
    for action_text in observed_action_texts:
        parts.append(f"You then {action_text} over 0.25 seconds.\nNext observation: {placeholder}")
    return "\n\n".join(parts)


def _split_actions(normalized_actions: Sequence[str]) -> tuple[list[str], list[str]]:
    if len(normalized_actions) < 4:
        raise ValueError("actions must contain at least 4 items: 3 observed actions and 1 future action")
    observed_action_texts = list(normalized_actions[:3])
    future_action_texts = list(normalized_actions[3:])
    if not future_action_texts:
        raise ValueError("At least one future action is required.")
    return observed_action_texts, future_action_texts


def _prepare_image_sequence(
    context_images: Sequence[str | Path | Image.Image],
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[Image.Image], torch.Tensor]:
    if len(context_images) != 4:
        raise ValueError(f"context_images must contain exactly 4 images, got {len(context_images)}")

    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    images: list[Image.Image] = []
    tensors = []
    for image in context_images:
        pil_image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        processed_tensor = preprocess(pil_image)
        images.append(transforms.ToPILImage()(processed_tensor))
        tensors.append(processed_tensor)

    pixel_values = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=dtype).view(1, 3, 1, 1)
    return images, (pixel_values - mean) / std


class LatentUMWorldModelInferencer:
    def __init__(self, model: LatentUMModel, decoder: LatentUMRefDecoderModel):
        self.model = model
        self.decoder = decoder
        self.internvl = model.internvl
        self.quantizer = model.quantizer
        self.tokenizer = model.tokenizer
        self.device = model._runtime_device
        self.dtype = model._runtime_dtype
        self.num_image_tokens = model.config.num_image_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _encode_context_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        visual_emb = self.internvl.extract_feature(pixel_values)
        return visual_emb.view(1, pixel_values.shape[0], self.num_image_tokens, -1)

    def _build_prompt_text(self, observed_action_texts: Sequence[str], future_action_text: str) -> str:
        from .internvl.conversation import get_conv_template

        context_block = _build_context_block(
            num_context_images=4,
            observed_action_texts=observed_action_texts,
            num_image_tokens=self.num_image_tokens,
        )
        template = get_conv_template("internvl2_5")
        template.append_message(
            template.roles[0],
            WM_INITIAL_USER_PROMPT_TEMPLATE.format(
                context_block=context_block,
                action_text=future_action_text,
            ),
        )
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def _build_prompt_inputs(
        self,
        prompt_text: str,
        visual_emb_src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_output = self.tokenizer(
            [prompt_text],
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        )
        input_ids = tokenizer_output["input_ids"].to(self.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.device)
        input_embeds = self.internvl.language_model.get_input_embeddings()(input_ids).clone()

        img_positions = (input_ids[0] == self.img_context_token_id).nonzero(as_tuple=True)[0]
        required_ctx_tokens = visual_emb_src.shape[1] * self.num_image_tokens
        if img_positions.numel() < required_ctx_tokens:
            raise RuntimeError(
                f"Prompt IMG_CONTEXT count mismatch: expected at least {required_ctx_tokens}, got {img_positions.numel()}"
            )
        for src_idx in range(visual_emb_src.shape[1]):
            start = src_idx * self.num_image_tokens
            end = start + self.num_image_tokens
            input_embeds[0, img_positions[start:end]] = visual_emb_src[0, src_idx]
        return input_embeds, attention_mask

    def _forward_prompt(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        vision_token_mask = torch.zeros(1, input_embeds.shape[1], device=self.device, dtype=self.dtype)
        outputs = self.internvl.language_model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            vision_token_mask=vision_token_mask,
            use_cache=True,
        )
        return outputs.past_key_values, outputs.last_hidden_state[:, -1:, :]

    @torch.inference_mode()
    def _generate_next_frame_codes(self, past_key_values, sampling_kwargs: dict) -> torch.Tensor:
        img_embed = self.internvl.language_model.get_input_embeddings()(
            torch.tensor([[self.img_start_token_id]], device=self.device)
        )
        outputs = self.internvl.language_model.model(
            inputs_embeds=img_embed,
            past_key_values=past_key_values,
            vision_token_mask=torch.ones(1, 1, device=self.device, dtype=self.dtype),
            use_cache=True,
        )
        past_key_values_phase2 = outputs.past_key_values
        base_hidden = outputs.last_hidden_state

        generated_codes = []
        code = self.internvl.ar_head.generate_from_base_token(
            base_hidden,
            cfg_scale=1.0,
            sampling_kwargs=sampling_kwargs,
        )
        generated_codes.append(code)

        for _ in trange(self.num_image_tokens - 1):
            z_q, _ = self.quantizer.indices_to_feature(code.unsqueeze(1))
            current_input = self.internvl.visual_projector(z_q)
            outputs = self.internvl.language_model.model(
                inputs_embeds=current_input,
                past_key_values=past_key_values_phase2,
                vision_token_mask=torch.ones(1, 1, device=self.device, dtype=self.dtype),
                use_cache=True,
            )
            past_key_values_phase2 = outputs.past_key_values
            code = self.internvl.ar_head.generate_from_base_token(
                outputs.last_hidden_state,
                cfg_scale=1.0,
                sampling_kwargs=sampling_kwargs,
            )
            generated_codes.append(code)

        return torch.stack(generated_codes, dim=1)

    def _decode_codes(
        self,
        generated_codes: torch.Tensor,
        ref_images: Sequence[Image.Image],
        *,
        seed: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        show_progress: bool,
    ) -> Image.Image:
        z_q, _ = self.quantizer.indices_to_feature(generated_codes.to(self.device))
        return self.decoder.decode(
            z_q,
            ref_images=ref_images,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.model.config.image_size,
            width=self.model.config.image_size,
            show_progress=show_progress,
        )[0]

    def _update_sliding_window(
        self,
        context_images: list[Image.Image],
        observed_action_texts: list[str],
        next_image: Image.Image,
        action_text: str,
    ) -> tuple[list[Image.Image], list[str]]:
        updated_images = (context_images + [next_image])[-4:]
        updated_actions = (observed_action_texts + [action_text])[-3:]
        return updated_images, updated_actions

    @torch.inference_mode()
    def rollout(
        self,
        context_images: Sequence[str | Path | Image.Image],
        actions: Sequence[str],
        *,
        seed: int | None = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        show_progress: bool = False,
    ) -> WMRolloutResult:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        if seed is not None:
            torch.manual_seed(seed)

        normalized_actions = normalize_action_list(actions)
        observed_action_texts, future_action_texts = _split_actions(normalized_actions)
        current_context_images, pixel_values = _prepare_image_sequence(
            context_images=context_images,
            image_size=self.model.config.image_size,
            device=self.device,
            dtype=self.dtype,
        )
        ref_context_images = list(current_context_images)
        current_observed_action_texts = list(observed_action_texts)

        sampling_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_logits": True,
        }

        steps: list[WMRolloutStep] = []
        for step_idx, action_text in enumerate(future_action_texts):
            visual_emb_src = self._encode_context_images(pixel_values)
            prompt_text = self._build_prompt_text(current_observed_action_texts, action_text)
            input_embeds, attention_mask = self._build_prompt_inputs(prompt_text, visual_emb_src)
            past_key_values, _ = self._forward_prompt(input_embeds, attention_mask)
            generated_codes = self._generate_next_frame_codes(past_key_values, sampling_kwargs)
            decoded_image = self._decode_codes(
                generated_codes,
                ref_context_images,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                show_progress=show_progress,
            )
            steps.append(
                WMRolloutStep(
                    step_idx=step_idx,
                    action_input=actions[step_idx + 3],
                    action_text=action_text,
                    codes=generated_codes.detach().cpu(),
                    image=decoded_image,
                )
            )
            current_context_images, current_observed_action_texts = self._update_sliding_window(
                current_context_images,
                current_observed_action_texts,
                decoded_image,
                action_text,
            )
            _, pixel_values = _prepare_image_sequence(
                context_images=current_context_images,
                image_size=self.model.config.image_size,
                device=self.device,
                dtype=self.dtype,
            )

        return WMRolloutResult(
            normalized_actions=normalized_actions,
            observed_action_texts=observed_action_texts,
            future_action_texts=future_action_texts,
            steps=steps,
        )


@torch.inference_mode()
def run_wm_inference(
    model: LatentUMModel,
    decoder: LatentUMRefDecoderModel,
    *,
    context_images: Sequence[str | Path | Image.Image],
    actions: Sequence[str],
    output_dir: str,
    seed: int | None = None,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    num_inference_steps: int = 25,
    guidance_scale: float = 1.0,
    show_progress: bool = False,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    inferencer = LatentUMWorldModelInferencer(model, decoder)
    result = inferencer.rollout(
        context_images=context_images,
        actions=actions,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        show_progress=show_progress,
    )

    generated_paths = []
    total_images = []
    for step in result.steps:
        output_path = images_dir / f"step_{step.step_idx:03d}.png"
        step.image.save(output_path)
        total_images.append(step.image)
        generated_paths.append(str(output_path))
    total_images[0].save(
        images_dir / 'generated.gif',
        save_all=True,
        append_images=total_images[1:],
        duration=400,
        loop=0
    )

    payload = {
        "context_images": [
            str(image) if isinstance(image, (str, Path)) else f"<PIL:{idx}>"
            for idx, image in enumerate(context_images)
        ],
        "raw_actions": list(actions),
        "normalized_actions": result.normalized_actions,
        "observed_action_texts": result.observed_action_texts,
        "future_action_texts": result.future_action_texts,
        "generated_images": generated_paths,
    }
    with open(output_root / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
