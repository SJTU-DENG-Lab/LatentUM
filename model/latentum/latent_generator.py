import torch


@torch.inference_mode()
def intern_gen(internvl, quantizer, tokenizer, prompt, batch_size, cfg_scale, sampling_kwargs, device, verbose=True):
    from tqdm import trange

    dtype = torch.bfloat16
    img_start_token = "<img>"

    if isinstance(prompt, str):
        prompt = prompt + img_start_token
        batch_prompts = [prompt] * batch_size
    else:
        batch_prompts = [p + img_start_token for p in prompt for _ in range(batch_size)]

    tokenizer_output = tokenizer(
        batch_prompts,
        padding=True,
        padding_side="left",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
    attention_mask = tokenizer_output["attention_mask"].to(device)
    text_embedding = internvl.language_model.get_input_embeddings()(input_ids)

    if cfg_scale > 1:
        uncond_input_ids = input_ids.clone()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        img_token_id = tokenizer.convert_tokens_to_ids(img_start_token)
        for b in range(uncond_input_ids.shape[0]):
            for t in range(uncond_input_ids.shape[1]):
                if uncond_input_ids[b, t] == img_token_id:
                    break
                uncond_input_ids[b, t] = pad_token_id

        uncond_text_embedding = internvl.language_model.get_input_embeddings()(uncond_input_ids)
        text_embedding_cfg = torch.cat([text_embedding, uncond_text_embedding], dim=0)
        attention_mask_cfg = torch.cat([attention_mask, attention_mask.clone()], dim=0)
    else:
        text_embedding_cfg = text_embedding
        attention_mask_cfg = attention_mask

    past_key_values = None
    generated_codes = []
    accumulated_attention_mask = attention_mask_cfg.clone()
    iterator = trange(256) if verbose else range(256)

    for i in iterator:
        if i == 0:
            current_input = text_embedding_cfg
            current_attention_mask = accumulated_attention_mask
            vision_token_mask = torch.zeros(
                current_input.shape[0], current_input.shape[1], device=device, dtype=dtype
            )
            vision_token_mask[:, -1] = 1.0
        else:
            if cfg_scale > 1:
                current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
            else:
                current_input = img_embeds_current

            accumulated_attention_mask = torch.cat(
                [
                    accumulated_attention_mask,
                    torch.ones(
                        accumulated_attention_mask.shape[0],
                        1,
                        device=device,
                        dtype=accumulated_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            current_attention_mask = accumulated_attention_mask
            vision_token_mask = torch.ones(
                current_input.shape[0], current_input.shape[1], device=device, dtype=dtype
            )

        outputs = internvl.language_model.model(
            inputs_embeds=current_input,
            attention_mask=current_attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            vision_token_mask=vision_token_mask,
        )
        base_token = outputs.last_hidden_state[:, -1:, :]
        past_key_values = outputs.past_key_values

        generated_code = internvl.ar_head.generate_from_base_token(base_token, cfg_scale, sampling_kwargs)
        z_q_current, _ = quantizer.indices_to_feature(generated_code.unsqueeze(1))
        img_embeds_current = internvl.visual_projector(z_q_current)
        generated_codes.append(generated_code)

    return torch.stack(generated_codes, dim=1)
