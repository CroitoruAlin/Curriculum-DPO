import torch
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(device=emb.device, dtype=dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb
def process_single_batch(batch, teacher_unet, unet, unet_ref, target_unet, accelerator, vae,
                         compute_embeddings_fn, noise_scheduler, solver,
                         scalings_for_boundary_conditions, predicted_origin, weight_dtype, args,
                         append_dims, alpha_schedule, sigma_schedule, uncond_prompt_embeds):
    image_w, image_l, text = batch['pixel_values_w'], batch['pixel_values_l'], batch['input_ids']
    bsz = image_w.shape[0]
    
    
    def process_image(image, noise=None, index=None, w=None):
        # for name, param in unet.named_parameters():
        #     if param.requires_grad:
        #         print("Unet grad:",name)
        image = image.to(accelerator.device, non_blocking=True)
        encoded_text = compute_embeddings_fn(text)

        pixel_values = image.to(dtype=weight_dtype)
        if vae.dtype != weight_dtype:
            vae.to(dtype=weight_dtype)

        # encode pixel values with batch size of at most 32
        latents = []
        for i in range(0, pixel_values.shape[0], 32):
            latents.append(vae.encode(pixel_values[i : i + 32]).latent_dist.sample())
        latents = torch.cat(latents, dim=0)

        latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)

        # Sample noise that we'll add to the latents
        if noise is None:  
            noise = torch.randn_like(latents)
        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = noise_scheduler.config.num_train_timesteps // args.sample.num_ddim_timesteps
        if index is None:
            index = torch.randint(0, args.sample.num_ddim_timesteps, (bsz,), device=accelerator.device).long()
    
        start_timesteps = solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
        if w is None:
            w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
        w = w.reshape(bsz)
        w_embedding = guidance_scale_embedding(w, embedding_dim=args.unet_time_cond_proj_dim)
        w = w.reshape(bsz, 1, 1, 1)
        w = w.to(device=latents.device, dtype=latents.dtype)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
        # 20.4.8. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text.pop("prompt_embeds")

        # 20.4.9. Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
        with torch.enable_grad():
            noise_pred = unet(
                noisy_model_input,
                start_timesteps,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds.float(),
                added_cond_kwargs={"text_embeds":encoded_text},
            ).sample
        # accelerator.backward(noise_pred)
        # print("Model pred", noise_pred.requires_grad)
        # for name, param in unet.named_parameters():
        #     if param.requires_grad:
        #         print("Unet grad:",name)
        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
        with torch.no_grad():
            with torch.autocast("cuda"):
                noise_pred_ref = unet_ref(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states = prompt_embeds.float(),
                    added_cond_kwargs={"text_embeds":encoded_text}
                ).sample

                pred_x_0_ref = predicted_origin(
                            noise_pred_ref,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                model_ref_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0_ref

        # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
        # noisy_latents with both the conditioning embedding c and unconditional embedding 0
        # Get teacher model prediction on noisy_latents and conditional embedding
        with torch.no_grad():
            with torch.autocast("cuda"):
                cond_teacher_output = teacher_unet(
                    noisy_model_input.to(weight_dtype),
                    start_timesteps,
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                ).sample
                cond_pred_x0 = predicted_origin(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                # Get teacher model prediction on noisy_latents and unconditional embedding
                uncond_teacher_output = teacher_unet(
                    noisy_model_input.to(weight_dtype),
                    start_timesteps,
                    encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                ).sample
                uncond_pred_x0 = predicted_origin(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                x_prev = solver.ddim_step(pred_x0, pred_noise, index)

        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                target_noise_pred = unet_ref(
                    x_prev.float(),
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                ).sample
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            target = c_skip * x_prev + c_out * pred_x_0

     # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                target_noise_ref = unet_ref(
                    x_prev.float(),
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                ).sample
            pred_x_0 = predicted_origin(
                target_noise_ref,
                timesteps,
                x_prev,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            target_ref = c_skip * x_prev + c_out * pred_x_0
        
        return model_pred, target, model_ref_pred, target_ref, noise, index, w
    
    model_pred_w, target_pred_w, model_ref_pred_w, target_ref_w, noise, index, w = process_image(image_w)
    model_pred_l, target_pred_l, model_ref_pred_l, target_ref_l, _, _, _ = process_image(image_l, noise, index, w)
    return model_pred_w, target_pred_w, target_ref_w, model_ref_pred_w, model_ref_pred_l, model_pred_l, target_pred_l, target_ref_l, index

def process_single_batch_lcm(batch, teacher_unet, unet, accelerator, vae,
                         compute_embeddings_fn, noise_scheduler, solver,
                         scalings_for_boundary_conditions, predicted_origin, weight_dtype, args,
                         append_dims, alpha_schedule, sigma_schedule, uncond_prompt_embeds):
    image, text = batch['pixel_values'], batch['input_ids']
    bsz = image.shape[0]
    
    
    def process_image(image, noise=None, index=None, w=None):
        # for name, param in unet.named_parameters():
        #     if param.requires_grad:
        #         print("Unet grad:",name)
        image = image.to(accelerator.device, non_blocking=True)
        encoded_text = compute_embeddings_fn(text)

        pixel_values = image.to(dtype=weight_dtype)
        if vae.dtype != weight_dtype:
            vae.to(dtype=weight_dtype)

        # encode pixel values with batch size of at most 32
        latents = []
        for i in range(0, pixel_values.shape[0], 32):
            latents.append(vae.encode(pixel_values[i : i + 32]).latent_dist.sample())
        latents = torch.cat(latents, dim=0)

        latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)

        # Sample noise that we'll add to the latents
        if noise is None:  
            noise = torch.randn_like(latents)
        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = noise_scheduler.config.num_train_timesteps // args.sample.num_ddim_timesteps
        if index is None:
            index = torch.randint(0, args.sample.num_ddim_timesteps, (bsz,), device=accelerator.device).long()
    
        start_timesteps = solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
        noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

        # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
        if w is None:
            w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
        w = w.reshape(bsz)
        w_embedding = guidance_scale_embedding(w, embedding_dim=args.unet_time_cond_proj_dim)
        w = w.reshape(bsz, 1, 1, 1)
        w = w.to(device=latents.device, dtype=latents.dtype)
        w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
        # 20.4.8. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text.pop("prompt_embeds")

        # 20.4.9. Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
        with torch.enable_grad():
            noise_pred = unet(
                noisy_model_input,
                start_timesteps,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds.float(),
                added_cond_kwargs={"text_embeds":encoded_text},
            ).sample
        
        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            noise_scheduler.config.prediction_type,
            alpha_schedule,
            sigma_schedule,
        )

        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
        
        # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
        # noisy_latents with both the conditioning embedding c and unconditional embedding 0
        # Get teacher model prediction on noisy_latents and conditional embedding
        with torch.no_grad():
            with torch.autocast("cuda"):
                cond_teacher_output = teacher_unet(
                    noisy_model_input.to(weight_dtype),
                    start_timesteps,
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                ).sample
                cond_pred_x0 = predicted_origin(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                # Get teacher model prediction on noisy_latents and unconditional embedding
                uncond_teacher_output = teacher_unet(
                    noisy_model_input.to(weight_dtype),
                    start_timesteps,
                    encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                ).sample
                uncond_pred_x0 = predicted_origin(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                x_prev = solver.ddim_step(pred_x0, pred_noise, index)

        # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                target_noise_pred = unet(
                    x_prev.float(),
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                ).sample
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        return model_pred, target, noise, index, w
    
    model_pred, target_pred, noise, index, w = process_image(image)
    return  model_pred, target_pred
