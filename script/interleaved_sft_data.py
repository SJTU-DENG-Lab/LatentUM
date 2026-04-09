import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.latentum.internvl.conversation import get_conv_template


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

DEFAULT_FROZENLAKE_USER_PROMPT = """You are navigating a grid-based frozen lake maze. Your goal is to move step by step from the starting position to the goal while avoiding holes.

Rules:
- You can move one square at a time: U (up), D (down), L (left), R (right).
- Falling into a hole means failure. Reaching the goal means success.

Task:
Look at the initial maze state below. Solve the maze by outputting one move at a time. After each move, visualize the resulting maze state as an image. When you reach the goal, output "Done!" to finish.

Here is the initial maze state: """


def build_image_placeholder(num_image_tokens: int) -> str:
    return IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens + IMG_END_TOKEN


def add_noise_to_indices(indices: torch.Tensor, noise_ratio: float, num_embeddings: int) -> torch.Tensor:
    noisy = indices.clone()
    mask = torch.rand_like(indices.float()) < noise_ratio
    noisy[mask] = torch.randint(0, num_embeddings, indices.shape, device=indices.device)[mask]
    return noisy


def build_dual_seq_attention_mask(
    L_U: int,
    num_tgt_images: torch.Tensor,
    delim_positions: torch.Tensor,
    max_K: int,
    *,
    L_img_gen: int = 257,
    padding_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B = num_tgt_images.shape[0]
    L_G = max_K * L_img_gen
    L_total = L_U + L_G

    mask = torch.zeros(B, L_total, L_total, device=device, dtype=torch.bool)
    causal_u = torch.tril(torch.ones(L_U, L_U, device=device, dtype=torch.bool))
    mask[:, :L_U, :L_U] = causal_u

    if padding_mask is not None:
        pad_col_mask = padding_mask.bool().unsqueeze(1)
        mask[:, :L_U, :L_U] = mask[:, :L_U, :L_U] & pad_col_mask

    block_causal = torch.tril(torch.ones(L_img_gen, L_img_gen, device=device, dtype=torch.bool))
    for n in range(max_K):
        gs = L_U + n * L_img_gen
        ge = gs + L_img_gen
        mask[:, gs:ge, gs:ge] = block_causal

    col_idx = torch.arange(L_U, device=device).unsqueeze(0)
    for n in range(max_K):
        gs = L_U + n * L_img_gen
        ge = gs + L_img_gen
        cutoff = delim_positions[:, n]
        img_valid = n < num_tgt_images
        valid_mask = (col_idx <= cutoff.unsqueeze(1)) & img_valid.unsqueeze(1)
        if padding_mask is not None:
            valid_mask = valid_mask & padding_mask.bool()
        mask[:, gs:ge, :L_U] = valid_mask.unsqueeze(1).expand(B, L_img_gen, L_U)

    for b in range(B):
        num_valid = num_tgt_images[b].item()
        if num_valid < max_K:
            pad_start = L_U + num_valid * L_img_gen
            mask[b, pad_start:, :] = False
            mask[b, :, pad_start:] = False

    float_mask = torch.zeros(B, 1, L_total, L_total, device=device, dtype=dtype)
    float_mask.masked_fill_(~mask.unsqueeze(1), float("-inf"))
    return float_mask


def build_dual_seq_position_ids(
    L_U: int,
    num_tgt_images: torch.Tensor,
    delim_positions: torch.Tensor,
    ctx_start_positions: torch.Tensor,
    max_K: int,
    *,
    L_img_gen: int = 257,
    device: torch.device | None = None,
) -> torch.Tensor:
    B = num_tgt_images.shape[0]
    L_G = max_K * L_img_gen
    L_total = L_U + L_G

    position_ids = torch.zeros(B, L_total, dtype=torch.long, device=device)
    position_ids[:, :L_U] = torch.arange(L_U, device=device).unsqueeze(0)

    num_gen_tokens = L_img_gen - 1
    gen_offsets = torch.arange(num_gen_tokens, device=device)
    for n in range(max_K):
        gs = L_U + n * L_img_gen
        for b in range(B):
            if n < num_tgt_images[b].item():
                position_ids[b, gs] = delim_positions[b, n]
                position_ids[b, gs + 1:gs + L_img_gen] = ctx_start_positions[b, n] + gen_offsets

    return position_ids


@dataclass
class ParsedInterleavedSample:
    sample_id: str
    source_image: Path
    target_images: list[Path]
    prompt_text: str
    response_text: str


def _resolve_image_path(root: Path, image_value: str) -> Path:
    candidate = Path(image_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Missing image file: {candidate}")
    return candidate


def _parse_content_items(
    *,
    items: list[dict[str, Any]],
    root: Path,
    num_image_tokens: int,
) -> tuple[str, list[Path]]:
    placeholder = build_image_placeholder(num_image_tokens)
    text_parts: list[str] = []
    images: list[Path] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(item["text"])
        elif item_type == "image":
            image_path = _resolve_image_path(root, item["image"])
            images.append(image_path)
            text_parts.append(placeholder)
        else:
            raise ValueError(f"Unsupported content item type: {item_type!r}")
    return "".join(text_parts), images


def parse_interleaved_sample(record: dict[str, Any], *, root: Path, num_image_tokens: int) -> ParsedInterleavedSample:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Each sample must contain exactly two messages: [user, assistant].")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("v1 expects messages ordered as [user, assistant].")

    sample_id = str(record.get("id", "sample"))
    user_prompt, user_images = _parse_content_items(
        items=messages[0].get("content", []),
        root=root,
        num_image_tokens=num_image_tokens,
    )
    assistant_response, assistant_images = _parse_content_items(
        items=messages[1].get("content", []),
        root=root,
        num_image_tokens=num_image_tokens,
    )

    if len(user_images) != 1:
        raise ValueError(f"Sample {sample_id} must contain exactly one user image, got {len(user_images)}.")
    if not assistant_response:
        raise ValueError(f"Sample {sample_id} has empty assistant content.")
    if assistant_response.endswith(build_image_placeholder(num_image_tokens)):
        raise ValueError(f"Sample {sample_id} assistant content must end with text, not an image.")

    template = get_conv_template("internvl2_5")
    template.append_message(template.roles[0], user_prompt)
    template.append_message(template.roles[1], None)
    prompt_text = template.get_prompt()

    return ParsedInterleavedSample(
        sample_id=sample_id,
        source_image=user_images[0],
        target_images=assistant_images,
        prompt_text=prompt_text,
        response_text=assistant_response,
    )


class InterleavedSFTDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, *, num_image_tokens: int):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.num_image_tokens = num_image_tokens
        self.samples: list[ParsedInterleavedSample] = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_idx} of {self.jsonl_path}") from exc
                self.samples.append(
                    parse_interleaved_sample(
                        record,
                        root=self.root,
                        num_image_tokens=self.num_image_tokens,
                    )
                )

        if not self.samples:
            raise ValueError(f"No samples found in {self.jsonl_path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        source_image = Image.open(sample.source_image).convert("RGB")
        target_images = [Image.open(path).convert("RGB") for path in sample.target_images]
        return {
            "id": sample.sample_id,
            "source_image": source_image,
            "target_images": target_images,
            "prompt_text": sample.prompt_text,
            "response_text": sample.response_text,
        }


class InterleavedSFTCollator:
    def __init__(
        self,
        tokenizer,
        *,
        image_size: int,
        num_image_tokens: int,
        max_seq_length: int,
        max_target_images: int,
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.max_seq_length = max_seq_length
        self.max_target_images = max_target_images
        self.img_placeholder = build_image_placeholder(num_image_tokens)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values_src = []
        pixel_values_tgt = []
        num_tgt_images = []
        text_list = []
        prompt_text_list = []
        response_text_list = []
        sample_ids = []

        max_K = max(1, min(self.max_target_images, max(len(sample["target_images"]) for sample in batch)))

        for sample in batch:
            source_image = self.preprocess(sample["source_image"])
            target_images = [self.preprocess(img) for img in sample["target_images"][:max_K]]
            actual_K = len(target_images)

            while len(target_images) < max_K:
                target_images.append(torch.zeros(3, self.image_size, self.image_size))

            prompt_text = sample["prompt_text"]
            response_text = sample["response_text"] + self.tokenizer.eos_token
            full_text = prompt_text + response_text

            pixel_values_src.append(source_image)
            pixel_values_tgt.append(torch.stack(target_images))
            num_tgt_images.append(actual_K)
            prompt_text_list.append(prompt_text)
            response_text_list.append(response_text)
            text_list.append(full_text)
            sample_ids.append(sample["id"])

        pixel_values_src = torch.stack(pixel_values_src)
        pixel_values_tgt = torch.stack(pixel_values_tgt)
        num_tgt_images_tensor = torch.tensor(num_tgt_images, dtype=torch.long)

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokenizer_output = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )
        self.tokenizer.padding_side = original_padding_side

        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]
        labels = input_ids.clone()
        B = input_ids.shape[0]

        delim_positions = torch.full((B, max_K), -1, dtype=torch.long)
        ctx_start_positions = torch.full((B, max_K), -1, dtype=torch.long)

        for i in range(B):
            prompt_ids = self.tokenizer(
                prompt_text_list[i],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            response_ids = self.tokenizer(
                response_text_list[i],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]

            prompt_len = prompt_ids.shape[1]
            pad_len = (attention_mask[i] == 0).sum().item()
            labels[i, :pad_len + prompt_len] = -100

            img_positions = (response_ids[0] == self.img_start_token_id).nonzero(as_tuple=True)[0]
            for j, rel_pos in enumerate(img_positions[:max_K]):
                abs_pos = pad_len + prompt_len + rel_pos.item()
                if abs_pos >= input_ids.shape[1]:
                    break
                delim_positions[i, j] = abs_pos
                ctx_start_positions[i, j] = abs_pos + 1
                labels[i, abs_pos + 1:abs_pos + self.num_image_tokens + 2] = -100

        return {
            "sample_ids": sample_ids,
            "pixel_values_src": pixel_values_src,
            "pixel_values_tgt": pixel_values_tgt,
            "num_tgt_images": num_tgt_images_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "delim_positions": delim_positions,
            "ctx_start_positions": ctx_start_positions,
        }


def create_interleaved_dataloader(
    *,
    jsonl_path: str | Path,
    tokenizer,
    image_size: int,
    num_image_tokens: int,
    max_seq_length: int,
    max_target_images: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = InterleavedSFTDataset(jsonl_path, num_image_tokens=num_image_tokens)
    collator = InterleavedSFTCollator(
        tokenizer,
        image_size=image_size,
        num_image_tokens=num_image_tokens,
        max_seq_length=max_seq_length,
        max_target_images=max_target_images,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )


def _decode_token_span(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def extract_text_loss_spans(tokenizer, input_ids: torch.Tensor, labels: torch.Tensor) -> list[str]:
    spans: list[str] = []
    current: list[int] = []
    for token_id, label_id in zip(input_ids.tolist(), labels.tolist()):
        if label_id == -100:
            if current:
                spans.append(_decode_token_span(tokenizer, current))
                current = []
            continue
        current.append(token_id)
    if current:
        spans.append(_decode_token_span(tokenizer, current))
    return [span for span in spans if span]


def format_lang_loss_preview(tokenizer, batch: dict[str, Any]) -> str:
    lines = ["[Loss Preview] Language supervision"]
    sample_ids = batch["sample_ids"]
    for row_idx, sample_id in enumerate(sample_ids):
        text_spans = extract_text_loss_spans(
            tokenizer,
            batch["input_ids"][row_idx].cpu(),
            batch["labels"][row_idx].cpu(),
        )
        lines.append(f"- sample: {sample_id}")
        if not text_spans:
            lines.append("  supervised_text: <none>")
            continue
        lines.append("  supervised_text:")
        for span_idx, span in enumerate(text_spans, start=1):
            cleaned = span.replace("\n", "\\n")
            lines.append(f"    {span_idx}. {cleaned}")
    return "\n".join(lines)


def format_vision_loss_preview(tokenizer, batch: dict[str, Any]) -> str:
    lines = ["[Loss Preview] Vision supervision"]
    sample_ids = batch["sample_ids"]
    for row_idx, sample_id in enumerate(sample_ids):
        text_spans = extract_text_loss_spans(
            tokenizer,
            batch["input_ids"][row_idx].cpu(),
            batch["labels"][row_idx].cpu(),
        )
        num_images = int(batch["num_tgt_images"][row_idx].item())
        delim_positions = batch["delim_positions"][row_idx, :num_images].tolist()
        lines.append(f"- sample: {sample_id}")
        lines.append(f"  target_image_blocks: {num_images}")
        if delim_positions:
            lines.append(f"  image_delimiters: {delim_positions}")
        if text_spans:
            lines.append("  preceding_text_segments:")
            for span_idx, span in enumerate(text_spans, start=1):
                cleaned = span.replace("\n", "\\n")
                lines.append(f"    {span_idx}. {cleaned}")
        else:
            lines.append("  preceding_text_segments: <none>")
    return "\n".join(lines)
