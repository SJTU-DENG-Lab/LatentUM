"""
Stage-2 interleaved SFT example for LatentUM.

Important:
This script is intended to continue training from a checkpoint produced by
`script/train_interleaved_lang_only.py`, not directly from the original base
checkpoint. The vision-generation branch depends on the already-adapted
understanding branch, so readers should first run the language-only stage and
then pass that saved checkpoint to `--model-path` here.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.latentum import LatentUMModel
from script.interleaved_sft_data import (
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    add_noise_to_indices,
    build_dual_seq_attention_mask,
    build_dual_seq_position_ids,
    create_interleaved_dataloader,
    format_vision_loss_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interleaved text-image SFT example for LatentUM vision path.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Checkpoint path to continue from. This should be a checkpoint saved by "
            "script/train_interleaved_lang_only.py."
        ),
    )
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
    parser.add_argument("--noise-ratio", type=float, default=0.15)
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
    print(
        "[vision-only] Expected checkpoint source: a checkpoint saved by "
        "script/train_interleaved_lang_only.py. This stage is designed for continued "
        "training after the language-only stage."
    )
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
    num_embeddings = int(model.config.quantizer["num_embeddings"])
    vocab_size = int(model.config.head["num_embeddings"])
    num_codebooks = int(model.config.head["num_codebooks"])

    # Stage 2 adapts the generation branch after the understanding branch has
    # already been aligned. Only the vision-specific path and AR head learn.
    model.internvl.requires_grad_(False)
    model.quantizer.requires_grad_(False)
    for name, param in model.internvl.language_model.model.layers.named_parameters():
        if "_vision" in name:
            param.requires_grad_(True)
    model.internvl.ar_head.requires_grad_(True)
    model.internvl.language_model.config._attn_implementation = "sdpa"

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    accelerator.print(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")


    # The same JSONL interface is reused here, but the loss is now computed on
    # visual token prediction instead of assistant text tokens.
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
    img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    L_img_gen = num_image_tokens + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    data_iter = iter(dataloader)
    # Print a readable preview before training so users can inspect which image
    # blocks will contribute to the visual loss.
    preview_batch = next(iter(dataloader))
    if accelerator.is_main_process:
        print(format_vision_loss_preview(tokenizer, preview_batch))

    while step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            model.train()
            model.internvl.vision_model.eval()

            # The batch contains one understanding image and N target images,
            # along with delimiter positions that tell us where each future
            # image is referenced in the text sequence.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values_src = batch["pixel_values_src"].to(device=device, dtype=runtime_dtype)
            pixel_values_tgt = batch["pixel_values_tgt"].to(device=device, dtype=runtime_dtype)
            num_tgt_images = batch["num_tgt_images"].to(device)
            delim_positions = batch["delim_positions"].to(device)
            ctx_start_positions = batch["ctx_start_positions"].to(device)

            B = input_ids.shape[0]
            max_K = pixel_values_tgt.shape[1]
            L_U = input_ids.shape[1]
            H = W = image_size

            # Encode target images into visual latents.
            # We keep both:
            # - clean latents for generation supervision
            # - optional noisy latents for the understanding sequence
            with torch.no_grad():
                tgt_flat = pixel_values_tgt.reshape(B * max_K, 3, H, W)
                tgt_norm = (tgt_flat - img_mean) / img_std
                vit_feat_tgt = model.internvl.get_vit_feature(tgt_norm)
                z_q, code_tgt = model.quantizer.get_zq_indices(vit_feat_tgt)

                if args.noise_ratio > 0:
                    code_noisy = add_noise_to_indices(code_tgt, args.noise_ratio, num_embeddings)
                    z_q_noisy, _ = model.quantizer.indices_to_feature(code_noisy)
                else:
                    z_q_noisy = z_q

                vq_dim = z_q.shape[-1]
                z_q_clean = z_q.view(B, max_K, num_image_tokens, vq_dim)
                z_q_noisy = z_q_noisy.view(B, max_K, num_image_tokens, vq_dim)
                code_tgt = code_tgt.view(B, max_K, num_image_tokens, num_codebooks)

                src_norm = (pixel_values_src - img_mean) / img_std
                vit_embeds_src = model.internvl.extract_feature(src_norm)
                vis_emb_noisy = model.internvl.visual_projector(
                    z_q_noisy.reshape(-1, vq_dim)
                ).view(B, max_K, num_image_tokens, -1)
                vis_emb_clean = model.internvl.visual_projector(
                    z_q_clean.reshape(-1, vq_dim)
                ).view(B, max_K, num_image_tokens, -1)

            # Build the understanding sequence U by replacing IMG_CONTEXT slots
            # with:
            # - source-image features in the first image block
            # - target-image latent features in later blocks
            text_emb = model.internvl.language_model.get_input_embeddings()(input_ids)
            hidden_size = text_emb.shape[-1]

            u_emb = text_emb.clone()
            for b in range(B):
                ctx_pos = (input_ids[b] == img_ctx_id).nonzero(as_tuple=True)[0]
                required_ctx = num_image_tokens * (1 + num_tgt_images[b].item())
                if len(ctx_pos) < required_ctx:
                    raise RuntimeError(
                        f"Sample {batch['sample_ids'][b]} does not contain enough IMG_CONTEXT tokens."
                    )
                u_emb[b, ctx_pos[:num_image_tokens]] = vit_embeds_src[b]
                for k in range(num_tgt_images[b].item()):
                    s = num_image_tokens + k * num_image_tokens
                    e = s + num_image_tokens
                    u_emb[b, ctx_pos[s:e]] = vis_emb_noisy[b, k]

            # Build the generation sequence G explicitly.
            # Each block is: <img> + 256 latent-image tokens.
            gen_emb = torch.zeros(B, max_K * L_img_gen, hidden_size, device=device, dtype=runtime_dtype)
            img_start_emb = model.internvl.language_model.get_input_embeddings()(
                torch.tensor([img_start_id], device=device)
            ).squeeze(0)

            for n in range(max_K):
                gs = n * L_img_gen
                gen_emb[:, gs, :] = img_start_emb
                gen_emb[:, gs + 1:gs + L_img_gen, :] = vis_emb_clean[:, n]

            for b in range(B):
                num_valid = num_tgt_images[b].item()
                if num_valid < max_K:
                    gen_emb[b, num_valid * L_img_gen:] = 0

            # Concatenate understanding sequence U and generation sequence G.
            # U provides the condition; G is where visual-token prediction happens.
            full_emb = torch.cat([u_emb, gen_emb], dim=1)

            # vision_token_mask marks which tokens belong to the generation path.
            vision_token_mask = torch.cat(
                [
                    torch.zeros(B, L_U, device=device, dtype=runtime_dtype),
                    torch.ones(B, max_K * L_img_gen, device=device, dtype=runtime_dtype),
                ],
                dim=1,
            )
            for b in range(B):
                num_valid = num_tgt_images[b].item()
                if num_valid < max_K:
                    vision_token_mask[b, L_U + num_valid * L_img_gen:] = 0

            # The custom attention mask enforces the intended dependency:
            # generation blocks can look back into the appropriate prefix of the
            # understanding sequence, while remaining causal inside each block.
            attn_mask_4d = build_dual_seq_attention_mask(
                L_U=L_U,
                num_tgt_images=num_tgt_images,
                delim_positions=delim_positions,
                max_K=max_K,
                L_img_gen=L_img_gen,
                padding_mask=attention_mask,
                device=device,
                dtype=runtime_dtype,
            )
            position_ids = build_dual_seq_position_ids(
                L_U=L_U,
                num_tgt_images=num_tgt_images,
                delim_positions=delim_positions,
                ctx_start_positions=ctx_start_positions,
                max_K=max_K,
                L_img_gen=L_img_gen,
                device=device,
            )

            # Forward the combined sequence through the language model, then
            # slice out the generation hidden states for AR visual-token loss.
            outputs = model.internvl.language_model(
                inputs_embeds=full_emb,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                vision_token_mask=vision_token_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            all_hidden = outputs.hidden_states[-1]
            gen_hidden = all_hidden[:, L_U:, :]

            # Each generated image contributes an autoregressive loss over its
            # codebook indices, predicted from the corresponding generation block.
            total_loss_visual = torch.tensor(0.0, device=device, dtype=runtime_dtype)
            valid_image_count = 0
            for n in range(max_K):
                valid_batch = n < num_tgt_images
                if not valid_batch.any():
                    continue

                gs = n * L_img_gen
                hs_n = gen_hidden[:, gs:gs + num_image_tokens, :]
                code_n = code_tgt[:, n]

                hs_valid = hs_n[valid_batch]
                code_valid = code_n[valid_batch]
                B_valid = hs_valid.shape[0]

                prefix = rearrange(hs_valid, "B L D -> (B L) 1 D")
                head_visual_emb = model.internvl.ar_head._code_to_embeddings(code_valid)
                h = torch.cat((prefix, head_visual_emb), dim=1)
                logits_n = model.internvl.ar_head(h[:, :-1, :])
                logits_n = rearrange(logits_n, "(B L) K V -> B L K V", B=B_valid, L=num_image_tokens)

                loss_n = F.cross_entropy(logits_n.reshape(-1, vocab_size), code_valid.reshape(-1))
                total_loss_visual = total_loss_visual + loss_n
                valid_image_count += 1

            loss = total_loss_visual / max(valid_image_count, 1)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_learn, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                print(f"[vision-only] step={step} loss={loss.item():.4f}")
            if args.save_every_steps and step % args.save_every_steps == 0:
                save_checkpoint(accelerator, model, output_dir, step)

    save_checkpoint(accelerator, model, output_dir, step)


if __name__ == "__main__":
    main()
