import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.latentum import LatentUMModel
from script.interleaved_sft_data import (
    IMG_CONTEXT_TOKEN,
    add_noise_to_indices,
    create_interleaved_dataloader,
    format_lang_loss_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interleaved text-image SFT example for LatentUM language path.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-target-images", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    return parser.parse_args()


def get_runtime_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16


def get_mixed_precision() -> str:
    if not torch.cuda.is_available():
        return "no"
    return "bf16"


def save_checkpoint(accelerator: Accelerator, model: LatentUMModel, output_dir: Path, step: int) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        checkpoint_dir = output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(checkpoint_dir)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=get_mixed_precision(),
    )
    device = accelerator.device
    runtime_dtype = get_runtime_dtype(device)

    model = LatentUMModel.from_pretrained(args.model_path)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer failed to load from the pretrained LatentUM checkpoint.")

    image_size = int(model.config.image_size)
    num_image_tokens = int(model.config.num_image_tokens)

    # Stage 1 only adapts the understanding / language branch.
    # Everything is frozen first, then we selectively unfreeze the text path.
    model.internvl.requires_grad_(False)
    model.quantizer.requires_grad_(False)
    for name, param in model.internvl.language_model.model.layers.named_parameters():
        if "_vision" not in name:
            param.requires_grad_(True)
    model.internvl.language_model.lm_head.requires_grad_(True)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    accelerator.print(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")

    # The dataloader turns JSONL interleaved samples into:
    # - one source image for understanding
    # - N target images for future visual steps
    # - one flattened text sequence with image placeholders already expanded
    dataloader = create_interleaved_dataloader(
        jsonl_path=args.data,
        tokenizer=tokenizer,
        image_size=image_size,
        num_image_tokens=num_image_tokens,
        max_seq_length=args.max_seq_length,
        max_target_images=args.max_target_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    params_to_learn = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    model = model.to(device=device, dtype=runtime_dtype)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=runtime_dtype).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=runtime_dtype).view(1, 3, 1, 1)
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    data_iter = iter(dataloader)
    # Print a readable preview before training so users can verify which text
    # spans are actually supervised by the language loss.
    preview_batch = next(iter(dataloader))
    if accelerator.is_main_process:
        print(format_lang_loss_preview(tokenizer, preview_batch))

    while step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            model.train()
            model.internvl.vision_model.eval()

            # The collator has already aligned text tokens, source image, and
            # target images into one training example.
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values_src = batch["pixel_values_src"].to(device=device, dtype=runtime_dtype)
            pixel_values_tgt = batch["pixel_values_tgt"].to(device=device, dtype=runtime_dtype)
            num_tgt_images = batch["num_tgt_images"].to(device)

            B = input_ids.shape[0]
            max_K = pixel_values_tgt.shape[1]
            H = W = image_size

            # Encode assistant-side target images into latent tokens.
            # These latents are inserted into the later image placeholders so
            # the language model can reason over the generated visual context.
            with torch.no_grad():
                tgt_flat = pixel_values_tgt.reshape(B * max_K, 3, H, W)
                tgt_norm = (tgt_flat - img_mean) / img_std
                vit_feat_tgt = model.internvl.get_vit_feature(tgt_norm)
                z_q, code_tgt = model.quantizer.get_zq_indices(vit_feat_tgt)
                if args.noise_ratio > 0:
                    code_noisy = add_noise_to_indices(
                        code_tgt,
                        args.noise_ratio,
                        int(model.config.quantizer["num_embeddings"]),
                    )
                    z_q, _ = model.quantizer.indices_to_feature(code_noisy)
                vq_dim = z_q.shape[-1]
                z_q = z_q.view(B, max_K, num_image_tokens, vq_dim)

            # Encode the user-side source image into understanding features.
            with torch.no_grad():
                src_norm = (pixel_values_src - img_mean) / img_std
                vit_embeds_src = model.internvl.extract_feature(src_norm)
                vis_emb = model.internvl.visual_projector(
                    z_q.reshape(-1, vq_dim)
                ).view(B, max_K, num_image_tokens, -1)

            # Start from standard token embeddings, then replace IMG_CONTEXT
            # slots with visual embeddings:
            # - first image block: the observed source image
            # - later image blocks: teacher-forced target-image latents
            text_emb = model.internvl.language_model.get_input_embeddings()(input_ids)
            input_embeds = text_emb.clone()
            for b in range(B):
                ctx_pos = (input_ids[b] == img_ctx_id).nonzero(as_tuple=True)[0]
                required_ctx = num_image_tokens * (1 + num_tgt_images[b].item())
                if len(ctx_pos) < required_ctx:
                    raise RuntimeError(
                        f"Sample {batch['sample_ids'][b]} does not contain enough IMG_CONTEXT tokens."
                    )
                input_embeds[b, ctx_pos[:num_image_tokens]] = vit_embeds_src[b]
                for k in range(num_tgt_images[b].item()):
                    s = num_image_tokens + k * num_image_tokens
                    e = s + num_image_tokens
                    input_embeds[b, ctx_pos[s:e]] = vis_emb[b, k]

            # Standard causal LM loss on assistant text only.
            # Prompt tokens and image placeholder internals are masked out.
            outputs = model.internvl.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_learn, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                print(f"[lang-only] step={step} loss={loss.item():.4f}")
            if args.save_every_steps and step % args.save_every_steps == 0:
                save_checkpoint(accelerator, model, output_dir, step)

    save_checkpoint(accelerator, model, output_dir, step)


if __name__ == "__main__":
    main()
