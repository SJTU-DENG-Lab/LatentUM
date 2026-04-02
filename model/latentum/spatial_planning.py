import glob
import json
from copy import deepcopy
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from .modeling_latentum import LatentUMDecoderModel, LatentUMModel

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

USER_PROMPT_PREFIX = """You are navigating a grid-based frozen lake maze. Your goal is to move step by step from the starting position to the goal while avoiding holes.

Rules:
- You can move one square at a time: U (up), D (down), L (left), R (right).
- Falling into a hole means failure. Reaching the goal means success.

Task:
Look at the initial maze state below. Solve the maze by outputting one move at a time. After each move, visualize the resulting maze state as an image. When you reach the goal, output "Done!" to finish.

Here is the initial maze state: """


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        probs = probs.scatter(
            -1,
            sorted_indices,
            sorted_probs.masked_fill(sorted_mask, 0.0),
        )
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _load_initial_image(image_path: str, image_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Image.Image, torch.Tensor]:
    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    pixel_values = preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=dtype).view(1, 3, 1, 1)
    return image, (pixel_values - mean) / std


def _scan_maze_dirs(data_dir: str, levels: list[int] | None, num_samples: int) -> list[str]:
    if levels:
        maze_dirs = []
        for level in levels:
            maze_dirs.extend(glob.glob(str(Path(data_dir) / f"level{level}" / f"level{level}_*")))
    else:
        maze_dirs = glob.glob(str(Path(data_dir) / "level*" / "level*_*"))
    maze_dirs = sorted(maze_dirs)
    if num_samples > 0:
        maze_dirs = maze_dirs[:num_samples]
    return maze_dirs


class FrozenLakePlanner:
    def __init__(self, model: LatentUMModel):
        self.model = model
        self.internvl = model.internvl
        self.quantizer = model.quantizer
        self.tokenizer = model.tokenizer
        self.device = model._runtime_device
        self.dtype = model._runtime_dtype
        self.num_image_tokens = model.config.num_image_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.done_token_ids = self.tokenizer.encode("Done!", add_special_tokens=False)

    def _build_prompt(self, visual_emb_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from .internvl.conversation import get_conv_template

        user_prompt = (
            USER_PROMPT_PREFIX
            + IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * self.num_image_tokens
            + IMG_END_TOKEN
        )
        template = get_conv_template("internvl2_5")
        template.append_message(template.roles[0], user_prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()

        tokenizer_output = self.tokenizer(
            [prompt],
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        )
        input_ids = tokenizer_output["input_ids"].to(self.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.device)

        input_embeds = self.internvl.language_model.get_input_embeddings()(input_ids).clone()
        img_positions = (input_ids[0] == self.img_context_token_id).nonzero(as_tuple=True)[0]
        input_embeds[0, img_positions[: self.num_image_tokens]] = visual_emb_src[0]
        return input_embeds, attention_mask

    def _check_done(self, step_tokens: list[int]) -> bool:
        if len(step_tokens) < len(self.done_token_ids):
            return False
        return step_tokens[-len(self.done_token_ids) :] == self.done_token_ids

    @torch.inference_mode()
    def generate(
        self,
        pixel_values_src: torch.Tensor,
        *,
        sample_name: str | None = None,
        max_steps: int,
        max_text_tokens_per_step: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[list[dict], str]:
        visual_emb_src = self.internvl.extract_feature(pixel_values_src)
        input_embeds, attention_mask = self._build_prompt(visual_emb_src)

        vision_token_mask = torch.zeros(
            1, input_embeds.shape[1], device=self.device, dtype=self.dtype
        )
        outputs = self.internvl.language_model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            vision_token_mask=vision_token_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1:, :]

        all_tokens: list[int] = []
        steps: list[dict] = []

        sampling_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_logits": True,
        }

        for step_idx in range(max_steps):
            step_tokens: list[int] = []
            found_img = False

            for _ in range(max_text_tokens_per_step):
                logits = self.internvl.language_model.lm_head(last_hidden)[:, -1, :]
                next_token_id = _sample(logits, temperature, top_k, top_p).item()
                step_tokens.append(next_token_id)
                all_tokens.append(next_token_id)

                if next_token_id == self.img_start_token_id:
                    kv_before_img = deepcopy(past_key_values)
                    img_embed = self.internvl.language_model.get_input_embeddings()(
                        torch.tensor([[next_token_id]], device=self.device)
                    )
                    outputs = self.internvl.language_model.model(
                        inputs_embeds=img_embed,
                        past_key_values=past_key_values,
                        vision_token_mask=torch.ones(1, 1, device=self.device, dtype=self.dtype),
                        use_cache=True,
                    )
                    past_key_values_phase2 = deepcopy(outputs.past_key_values)
                    base_hidden = outputs.last_hidden_state
                    found_img = True
                    break

                if next_token_id == self.eos_token_id:
                    action_text = self.tokenizer.decode(step_tokens, skip_special_tokens=False)
                    steps.append({"action": action_text, "codes": None, "step": step_idx})
                    return steps, self.tokenizer.decode(all_tokens, skip_special_tokens=False)

                next_embed = self.internvl.language_model.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=self.device)
                )
                outputs = self.internvl.language_model.model(
                    inputs_embeds=next_embed,
                    past_key_values=past_key_values,
                    vision_token_mask=torch.zeros(1, 1, device=self.device, dtype=self.dtype),
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state

                if self._check_done(step_tokens):
                    continue

            if not found_img:
                action_text = self.tokenizer.decode(step_tokens, skip_special_tokens=False)
                steps.append({"action": action_text, "codes": None, "step": step_idx})
                break

            generated_codes = []
            code = self.internvl.ar_head.generate_from_base_token(
                base_hidden,
                cfg_scale=1.0,
                sampling_kwargs=sampling_kwargs,
            )
            generated_codes.append(code)

            action_text = self.tokenizer.decode(step_tokens, skip_special_tokens=False)
            action_text = action_text.strip().replace("\n", " ")
            print(f"[FrozenLake] action={action_text or '<empty>'}")

            progress_desc = f"[FrozenLake] visual tokens"
            if sample_name:
                progress_desc = f"[FrozenLake] {sample_name} visual tokens"
            progress_bar = tqdm(
                total=self.num_image_tokens,
                desc=progress_desc,
                leave=False,
            )
            progress_bar.update(1)

            for _ in range(self.num_image_tokens - 1):
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
                progress_bar.update(1)

            progress_bar.close()

            generated_codes = torch.stack(generated_codes, dim=1)

            img_embed = self.internvl.language_model.get_input_embeddings()(
                torch.tensor([[self.img_start_token_id]], device=self.device)
            )
            outputs = self.internvl.language_model.model(
                inputs_embeds=img_embed,
                past_key_values=kv_before_img,
                vision_token_mask=torch.zeros(1, 1, device=self.device, dtype=self.dtype),
                use_cache=True,
            )

            z_q, _ = self.quantizer.indices_to_feature(generated_codes)
            x_proj = self.internvl.visual_projector(z_q)
            outputs = self.internvl.language_model.model(
                inputs_embeds=x_proj,
                past_key_values=outputs.past_key_values,
                vision_token_mask=torch.zeros(
                    1,
                    self.num_image_tokens,
                    device=self.device,
                    dtype=self.dtype,
                ),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            img_end_embed = self.internvl.language_model.get_input_embeddings()(
                torch.tensor([[self.img_end_token_id]], device=self.device)
            )
            outputs = self.internvl.language_model.model(
                inputs_embeds=img_end_embed,
                past_key_values=past_key_values,
                vision_token_mask=torch.zeros(1, 1, device=self.device, dtype=self.dtype),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state

            steps.append({"action": action_text, "codes": generated_codes, "step": step_idx})
            all_tokens.append(self.img_end_token_id)

        return steps, self.tokenizer.decode(all_tokens, skip_special_tokens=False)


def save_gif(images: list[Image.Image], output_path: str, duration: int = 500) -> None:
    if len(images) < 2:
        if images:
            images[0].save(output_path)
        return
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


@torch.inference_mode()
def _run_frozenlake_sample(
    model: LatentUMModel,
    decoder: LatentUMDecoderModel,
    *,
    image: str | Path,
    sample_name: str,
    output_dir: str,
    max_steps: int,
    max_text_tokens_per_step: int,
    temperature: float,
    top_k: int,
    top_p: float,
    gif_duration: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> dict:
    output_root = Path(output_dir)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image)
    planner = FrozenLakePlanner(model)
    initial_img, pixel_values = _load_initial_image(
        str(image_path),
        model.config.image_size,
        model._runtime_device,
        model._runtime_dtype,
    )
    steps, full_text = planner.generate(
        pixel_values,
        sample_name=sample_name,
        max_steps=max_steps,
        max_text_tokens_per_step=max_text_tokens_per_step,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    resized_initial = initial_img.resize((model.config.image_size, model.config.image_size))
    all_images = [resized_initial]
    resized_initial.save(images_dir / f"{sample_name}_initial.png")

    for step_info in steps:
        if step_info["codes"] is None:
            continue
        z_q, _ = model.quantizer.indices_to_feature(step_info["codes"].to(model._runtime_device))
        decoded_images = decoder.decode(
            z_q,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=model.config.image_size,
            width=model.config.image_size,
        )
        decoded_image = decoded_images[0]
        decoded_image.save(images_dir / f"{sample_name}_step{step_info['step']}.png")
        all_images.append(decoded_image)

    gif_path = output_root / f"{sample_name}.gif"
    if len(all_images) >= 2:
        save_gif(all_images, str(gif_path), duration=gif_duration)

    result = {
        "maze": sample_name,
        "image": str(image_path),
        "full_text": full_text,
        "num_steps": len(steps),
        "actions": [step["action"] for step in steps],
        "has_images": [step["codes"] is not None for step in steps],
        "gif_path": str(gif_path) if len(all_images) >= 2 else None,
    }
    return result


@torch.inference_mode()
def run_frozenlake_demo(
    model: LatentUMModel,
    decoder: LatentUMDecoderModel,
    *,
    image: str | Path,
    output_dir: str,
    max_steps: int,
    max_text_tokens_per_step: int,
    temperature: float,
    top_k: int,
    top_p: float,
    gif_duration: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> dict:
    image_path = Path(image)
    output_root = Path(output_dir)
    result = _run_frozenlake_sample(
        model,
        decoder,
        image=image_path,
        sample_name=image_path.stem,
        output_dir=output_dir,
        max_steps=max_steps,
        max_text_tokens_per_step=max_text_tokens_per_step,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        gif_duration=gif_duration,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    with open(output_root / "inference_log.json", "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    return result


@torch.inference_mode()
def run_frozenlake_inference(
    model: LatentUMModel,
    decoder: LatentUMDecoderModel,
    *,
    data_dir: str,
    output_dir: str,
    levels: list[int] | None,
    num_samples: int,
    max_steps: int,
    max_text_tokens_per_step: int,
    temperature: float,
    top_k: int,
    top_p: float,
    gif_duration: int,
    num_inference_steps: int = 20,
    guidance_scale: float = 1.0,
) -> None:
    maze_dirs = _scan_maze_dirs(data_dir, levels, num_samples)
    print(f"Found {len(maze_dirs)} maze samples")

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    gifs_dir = output_root / "gifs"
    images_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    results_log = []

    for idx, maze_dir in enumerate(maze_dirs, start=1):
        maze_name = Path(maze_dir).name
        print(f"[{idx}/{len(maze_dirs)}] Running {maze_name}")
        result = _run_frozenlake_sample(
            model,
            decoder,
            image=Path(maze_dir) / "0.png",
            sample_name=maze_name,
            output_dir=output_root,
            max_steps=max_steps,
            max_text_tokens_per_step=max_text_tokens_per_step,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            gif_duration=gif_duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if result["gif_path"] is not None:
            gif_src = Path(result["gif_path"])
            gif_dst = gifs_dir / f"{maze_name}.gif"
            if gif_src != gif_dst:
                gif_dst.write_bytes(gif_src.read_bytes())
                gif_src.unlink()
                result["gif_path"] = str(gif_dst)
        result["maze_dir"] = maze_dir
        results_log.append(result)

    with open(output_root / "inference_log.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
