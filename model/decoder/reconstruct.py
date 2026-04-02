import torch


@torch.no_grad()
def sample_sd3_5(
    transformer,
    vae,
    noise_scheduler,
    device,
    dtype,
    context,
    batch_size=1,
    height=192,
    width=192,
    num_inference_steps=20,
    guidance_scale=1.0,
    seed=None,
    multi_modal_context=False,
    show_progress=False,
):
    if show_progress:
        from tqdm.auto import tqdm

    if seed is not None:
        torch.manual_seed(seed)

    transformer.eval()

    latent_height = height // 8
    latent_width = width // 8
    latents = torch.randn((batch_size, 16, latent_height, latent_width), device=device, dtype=dtype)

    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device=device, dtype=dtype)

    step_iterator = tqdm(timesteps, desc="Decoding image", leave=False) if show_progress else timesteps

    for t in step_iterator:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.repeat(batch_size)

        latent_model_input = latents
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            t = torch.cat([t, t], dim=0)
            context_ = torch.cat(
                [context, torch.zeros_like(context, device=context.device, dtype=context.dtype)], dim=0
            )
        else:
            context_ = context

        noise_pred = transformer(
            x=latent_model_input,
            t=t,
            context=context_,
            y=None,
            multi_modal_context=multi_modal_context,
        )

        if guidance_scale > 1.0:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t[0] if t.ndim > 0 else t,
            sample=latents,
            return_dict=False,
        )[0]

    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    image = vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)
