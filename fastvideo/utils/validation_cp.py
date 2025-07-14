import gc
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

from fastvideo.distill.solver import PCMFMScheduler
from fastvideo.models.mochi_hf.pipeline_mochi import (
    linear_quadratic_schedule, retrieve_timesteps)
from fastvideo.utils.communications import all_gather
from fastvideo.utils.load import load_vae
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)
import imageio
from einops import rearrange
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    num_frames,
    dtype,
    device,
    generator,
    vae_spatial_scale_factor,
    vae_temporal_scale_factor,
):
    height = height // vae_spatial_scale_factor
    width = width // vae_spatial_scale_factor
    num_frames = (num_frames - 1) // vae_temporal_scale_factor + 1

    shape = (batch_size, num_channels_latents, num_frames, height, width)

    latents = randn_tensor(shape,
                           generator=generator,
                           device=device,
                           dtype=dtype)
    return latents


# def sample_validation_video(
#     model_type,
#     transformer,
#     vae,
#     scheduler,
#     scheduler_type="euler",
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     num_frames: int = 16,
#     num_inference_steps: int = 28,
#     timesteps: List[int] = None,
#     guidance_scale: float = 4.5,
#     num_videos_per_prompt: Optional[int] = 1,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     prompt_embeds: Optional[torch.Tensor] = None,
#     prompt_attention_mask: Optional[torch.Tensor] = None,
#     negative_prompt_embeds: Optional[torch.Tensor] = None,
#     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#     output_type: Optional[str] = "pil",
#     vae_spatial_scale_factor=8,
#     vae_temporal_scale_factor=6,
#     num_channels_latents=12,
# ):
#     device = vae.device

#     batch_size = prompt_embeds.shape[0]
#     do_classifier_free_guidance = guidance_scale > 1.0
#     if do_classifier_free_guidance:
#         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
#                                   dim=0)
#         prompt_attention_mask = torch.cat(
#             [negative_prompt_attention_mask, prompt_attention_mask], dim=0)

#     # 4. Prepare latent variables
#     # TODO: Remove hardcore
#     latents = prepare_latents(
#         batch_size * num_videos_per_prompt,
#         num_channels_latents,
#         height,
#         width,
#         num_frames,
#         prompt_embeds.dtype,
#         device,
#         generator,
#         vae_spatial_scale_factor,
#         vae_temporal_scale_factor,
#     )
#     world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
#     if get_sequence_parallel_state():
#         latents = rearrange(latents,
#                             "b t (n s) h w -> b t n s h w",
#                             n=world_size).contiguous()
#         latents = latents[:, :, rank, :, :, :]

#     # 5. Prepare timestep
#     # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
#     threshold_noise = 0.025
#     sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
#     sigmas = np.array(sigmas)
#     if scheduler_type == "euler" and model_type == "mochi":  #todo
#         timesteps, num_inference_steps = retrieve_timesteps(
#             scheduler,
#             num_inference_steps,
#             device,
#             timesteps,
#             sigmas,
#         )
#     else:
#         timesteps, num_inference_steps = retrieve_timesteps(
#             scheduler,
#             num_inference_steps,
#             device,
#         )
#     num_warmup_steps = max(
#         len(timesteps) - num_inference_steps * scheduler.order, 0)

#     # 6. Denoising loop
#     # with self.progress_bar(total=num_inference_steps) as progress_bar:
#     # write with tqdm instead
#     # only enable if nccl_info.global_rank == 0
#     #print("jjjjjjjjjjjjjjjjjj")
#     with tqdm(
#             total=num_inference_steps,
#             disable=nccl_info.rank_within_group != 0,
#             desc="Validation sampling...",
#     ) as progress_bar:
#         for i, t in enumerate(timesteps):
#             latent_model_input = (torch.cat([latents] * 2)
#                                   if do_classifier_free_guidance else latents)
#             # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#             timestep = t.expand(latent_model_input.shape[0])
#             guidance=torch.tensor(
#                     [1000.0],
#                     device=latent_model_input.device,
#                     dtype=torch.bfloat16)
#             with torch.autocast("cuda", dtype=torch.bfloat16):
#                 noise_pred = transformer(
#                     hidden_states=latent_model_input,
#                     encoder_hidden_states=prompt_embeds,
#                     timestep=timestep,
#                     encoder_attention_mask=prompt_attention_mask,
#                     return_dict=False,
#                     guidance=guidance,
#                 )[0]
#             #print(noise_pred.shape,"ssssssssssssssss")

#             # Mochi CFG + Sampling runs in FP32
#             noise_pred = noise_pred.to(torch.float32)
#             if do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (
#                     noise_pred_text - noise_pred_uncond)

#             # compute the previous noisy sample x_t -> x_t-1
#             latents_dtype = latents.dtype
#             latents = scheduler.step(noise_pred,
#                                      t,
#                                      latents.to(torch.float32),
#                                      return_dict=False)[0]
#             latents = latents.to(latents_dtype)

#             if latents.dtype != latents_dtype:
#                 if torch.backends.mps.is_available():
#                     # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
#                     latents = latents.to(latents_dtype)

#             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
#                                            (i + 1) % scheduler.order == 0):
#                 progress_bar.update()

#     if get_sequence_parallel_state():
#         latents = all_gather(latents, dim=2)

#     if output_type == "latent":
#         video = latents
#     else:
#         # unscale/denormalize the latents
#         # denormalize with the mean and std if available and not None
#         has_latents_mean = (hasattr(vae.config, "latents_mean")
#                             and vae.config.latents_mean is not None)
#         has_latents_std = (hasattr(vae.config, "latents_std")
#                            and vae.config.latents_std is not None)
#         if has_latents_mean and has_latents_std:
#             latents_mean = (torch.tensor(vae.config.latents_mean).view(
#                 1, 12, 1, 1, 1).to(latents.device, latents.dtype))
#             latents_std = (torch.tensor(vae.config.latents_std).view(
#                 1, 12, 1, 1, 1).to(latents.device, latents.dtype))
#             latents = latents * latents_std / vae.config.scaling_factor + latents_mean
#         else:
#             latents = latents / vae.config.scaling_factor
#         with torch.autocast("cuda", dtype=vae.dtype):
#             video = vae.decode(latents, return_dict=False)[0]
#         print(video.shape,"sssssssssssssssssssssssssssssssssssssssssssssssss")
#         video_processor = VideoProcessor(
#             vae_scale_factor=vae_spatial_scale_factor)
#         video = video_processor.postprocess_video(video,
#                                                   output_type=output_type)

#     return (video, )
import gc
def sample_validation_video(
    model_type,
    transformer,
    vae,
    scheduler,
    scheduler_type="euler",
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 16,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 4.5,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    vae_spatial_scale_factor=8,
    vae_temporal_scale_factor=6,
    num_channels_latents=12,
    shift = None,
):
    device = vae.device

    batch_size = prompt_embeds.shape[0]
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
                                  dim=0)
        prompt_attention_mask = torch.cat(
            [negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # 5. Prepare timestep
    # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
    threshold_noise = 0.025
    sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
    sigmas = np.array(sigmas)
    if scheduler_type == "euler" and model_type == "mochi":  #todo
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
    else:
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
        )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * scheduler.order, 0)

    # 6. Denoising loop
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # write with tqdm instead
    # only enable if nccl_info.global_rank == 0
    #print("jjjjjjjjjjjjjjjjjj")
    # latents_pre_start = []
    # latents_pre_end = []
    for step in range(2):
        scheduler._step_index = 0
        device = vae.device
        batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
                                      dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        # 4. Prepare latent variables
        # TODO: Remove hardcore
        if step==0:
            latents = prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                prompt_embeds.dtype,
                device,
                generator,
                vae_spatial_scale_factor,
                vae_temporal_scale_factor,
            )
        else:
            latents = prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames*2,
                prompt_embeds.dtype,
                device,
                generator,
                vae_spatial_scale_factor,
                vae_temporal_scale_factor,
            )
        world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
        if get_sequence_parallel_state():
            latents = rearrange(latents,
                                "b t (n s) h w -> b t n s h w",
                                n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]
        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        sigmas = np.array(sigmas)
        if scheduler_type == "euler" and model_type == "mochi":  #todo
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps,
                device,
                timesteps,
                sigmas,
            )
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps,
                device,
            )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0)
        
        latents_pre_start.append(latents)
        with tqdm(
                total=num_inference_steps,
                disable=nccl_info.rank_within_group != 0,
                desc="Validation sampling...",
        ) as progress_bar:
#             if step != 0:
#                 for i, t in enumerate(timesteps):
#                     sigmas = 970 / scheduler.config.num_train_timesteps
#                     if i==5:
#                         sigmas = 970 / scheduler.config.num_train_timesteps
#                         latents = sigmas * torch.randn_like(latents_pre_start[step-1]) + (1.0 - sigmas) * latents_pre_end[-1]
#                         scheduler._step_index = 0
#                         break
#                     if step != 0:
#                         if i == 0:
#                             print(sigmas)
#                             latents[:,:,:6,:,:] = 0.5*latents[:,:,:6,:,:] + 0.5*( sigmas * latents_pre_start[step-1] + (1.0 - sigmas) * latents_pre_end[-1] )[:,:,-6:,:,:]
#                             latents = torch.cat([sigmas * latents_pre_start[step-1] + (1.0 - sigmas) * latents_pre_end[-1], latents],dim=2)
#                         else:
#                             latents[:,:,:6,:,:] = 0.5*latents[:,:,:6,:,:] + 0.5*latents_pre_end[i][:,:,-6:,:,:]
#                             latents = torch.cat([latents_pre_end[i], latents],dim=2)
#                         latents_pre_end[i] = latents[:,:,latents_pre_start[step-1].shape[2]:,:,:]
#                     else:
#                         latents_pre_end.append(latents)
#                     latent_model_input = (torch.cat([latents] * 2)
#                                           if do_classifier_free_guidance else latents)
#                     # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#                     timestep = t.expand(latent_model_input.shape[0])
#                     print(latent_model_input.shape, prompt_embeds.shape,timestep.shape, prompt_attention_mask.shape)
#                     with torch.autocast("cuda", dtype=torch.bfloat16):
#                         noise_pred = transformer(
#                             hidden_states=latent_model_input,
#                             encoder_hidden_states=prompt_embeds,
#                             timestep=timestep,
#                             encoder_attention_mask=prompt_attention_mask,
#                             return_dict=False,
#                         )[0]
#                     #print(noise_pred.shape,"ssssssssssssssss")

#                     # Mochi CFG + Sampling runs in FP32
#                     noise_pred = noise_pred.to(torch.float32)
#                     if do_classifier_free_guidance:
#                         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                         noise_pred = noise_pred_uncond + guidance_scale * (
#                             noise_pred_text - noise_pred_uncond)

#                     # compute the previous noisy sample x_t -> x_t-1
#                     latents_dtype = latents.dtype
#                     latents = scheduler.step(noise_pred,
#                                              t,
#                                              latents.to(torch.float32),
#                                              return_dict=False)[0]
#                     latents = latents.to(latents_dtype)
#                     if step != 0:
#                         latents = latents[:,:,latents_pre_start[step-1].shape[2]:,:,:]

            for i, t in enumerate(timesteps):
                #index_i = min(i+4,num_inference_steps-1)
                #sigmas = timesteps[index_i] / scheduler.config.num_train_timesteps
                if step != 0:
                    noisy_model_input_mask = torch.zeros_like(latents)
                    noisy_model_input_mask[:,:,:num_frames,:,:] = 1
                    model_input_ori_padd = torch.cat([latents_pre_end[:,:,-num_frames:,:,:], torch.zeros_like(latents_pre_end)], dim=2)
                    latents = torch.cat([latents, model_input_ori_padd, noisy_model_input_mask], dim=1)
#                     if i == 0:
#                         #print(sigmas)
#                         #latents[:,:,:6,:,:] = 0.5*latents[:,:,:6,:,:] + 0.5*( sigmas * latents_pre_start[step-1] + (1.0 - sigmas) * latents_pre_end[-1] )[:,:,-6:,:,:]
#                         latents = torch.cat([sigmas * latents_pre_start[step-1] + (1.0 - sigmas) * latents_pre_end[-1], latents],dim=2)
#                     else:
#                         # if i <= 10:
#                         #     latents[:,:,:6,:,:] = 0.5*latents[:,:,:6,:,:] + 0.5*latents_pre_end[i][:,:,-6:,:,:]
#                         #latents = torch.cat([sigmas * latents_pre_start[step-1] + (1.0 - sigmas) * latents_pre_end[-1], latents],dim=2)
#                         latents = torch.cat([latents_pre_end[index_i], latents],dim=2)
                        
#                     latents_pre_end[i] = latents[:,:,latents_pre_start[step-1].shape[2]:,:,:]
                # else:
                #     latents_pre_end.append(latents)
                latent_model_input = (torch.cat([latents] * 2)
                                      if do_classifier_free_guidance else latents)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                # if i != 0:
                #     timestep = t.expand(latent_model_input.shape[0])
                # else:
                #     timestep = torch.cat([timesteps[index_i].reshape(1).repeat(latents_pre_start[step-1].shape[2]),t.reshape(1).repeat(latents_pre_start[step-1].shape[2])],dim=0)
                # print(latent_model_input.shape, prompt_embeds.shape,timestep.shape, prompt_attention_mask.shape)
                timestep = t.expand(latent_model_input.shape[0])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if step != 0:
                        noise_pred = transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            encoder_attention_mask=prompt_attention_mask,
                            return_dict=False,
                            i2v = True,
                        )[0]
                    else:
                        noise_pred = transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            encoder_attention_mask=prompt_attention_mask,
                            return_dict=False,
                        )[0]
                #print(noise_pred.shape,"ssssssssssssssss")

                # Mochi CFG + Sampling runs in FP32
                noise_pred = noise_pred.to(torch.float32)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred,
                                         t,
                                         latents.to(torch.float32),
                                         return_dict=False)[0]
                latents = latents.to(latents_dtype)
                # if step != 0:
                #     latents = latents[:,:,latents_pre_start[step-1].shape[2]:,:,:]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

        if get_sequence_parallel_state():
            latents = all_gather(latents, dim=2)
        latents_pre_end = latents
        # if step != 0:
        #     latents_pre_end[-1] = latents
        # else:
        #     latents_pre_end.append(latents)
        if output_type == "latent":
            video = latents
        else:
            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (hasattr(vae.config, "latents_mean")
                                and vae.config.latents_mean is not None)
            has_latents_std = (hasattr(vae.config, "latents_std")
                               and vae.config.latents_std is not None)
            if has_latents_mean and has_latents_std:
                latents_mean = (torch.tensor(vae.config.latents_mean).view(
                    1, 12, 1, 1, 1).to(latents.device, latents.dtype))
                latents_std = (torch.tensor(vae.config.latents_std).view(
                    1, 12, 1, 1, 1).to(latents.device, latents.dtype))
                latents = latents * latents_std / vae.config.scaling_factor + latents_mean
            else:
                latents = latents / vae.config.scaling_factor
            with torch.autocast("cuda", dtype=vae.dtype):
                video = vae.decode(latents, return_dict=False)[0]
            if step != 0:
                video_ori = torch.cat([video_ori, video],dim=2)
            else:
                video_ori = video
            #print(video.shape,"sssssssssssssssssssssssssssssssssssssssssssssssss")

    video_processor = VideoProcessor(
                vae_scale_factor=vae_spatial_scale_factor)
    video = video_processor.postprocess_video(video_ori, output_type=output_type)

    return (video, )

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def log_validation(
    args,
    transformer,
    device,
    weight_dtype,  # TODO
    global_step,
    scheduler_type="euler",
    shift=1.0,
    num_euler_timesteps=100,
    linear_quadratic_threshold=0.025,
    linear_range=0.5,
    ema=False,
    loader_val=None,
    vae=None,
    text_encoder=None,
    fps=None,
):
    # TODO
    print("Running validation....\n")
    if args.model_type == "mochi":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 6
        num_channels_latents = 12
    elif args.model_type == "hunyuan" or "hunyuan_hf":
        vae_spatial_scale_factor = 8
        vae_temporal_scale_factor = 4
        num_channels_latents = 16
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    if scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    else:
        linear_quadraic = True if scheduler_type == "pcm_linear_quadratic" else False
        scheduler = PCMFMScheduler(
            1000,
            shift,
            num_euler_timesteps,
            linear_quadraic,
            linear_quadratic_threshold,
            linear_range,
        )
    # args.validation_prompt_dir
    validation_sampling_steps = []
    for validation_sampling_step in args.validation_sampling_steps.split(","):
        validation_sampling_step = int(validation_sampling_step)
        validation_sampling_steps.append(validation_sampling_step)
    #print("sssssssssssszzzzz")
    caption = [
    # "Time-lapse of ice grass growing in a pot over 24 days, showing the transformation from a seedling to a robust plant with distinctive crystalline structures on its leaves, evident in the increasing leaf size, density, and pot space filled.",
    "A cat walks on the grass, realistic style.",
  #  "Time-lapse of a peanut plant's development over a period of 157 days. Beginning with the emergence of a shoot from the soil, the plant exhibits gradual stages of growth, including increased height, foliage, and overall robustness, illustrating its journey to full maturity.",
 #   "Time-lapse of cat grass (wheat) seeds germination over a 5-day period: starting with soaked seeds and progressing through visible sprouting stages to fuller growth, capturing the development from beginning to end.",
    #"Time-lapse of a corn plant's growth over 9 days, starting from a sprout with a single leaf and progressing to a taller seedling with two fully developed leaves, thus indicating a healthy vegetative state.",
   # "Time-lapse of cat grass germination and growth over a series of stages: starting as freshly sprouted short grass, gradually increasing in height and density, leading to the grass at its tallest and most robust form at the final stage.",
   # "Time-lapse of a pink rose transitioning from a closed bud to full bloom. The sequence progresses forward, capturing each stage as the petals curl outward, eventually displaying the intricate layers of the fully opened flower against a contrasting black background."
]
    negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    for _ in range(1):
        if nccl_info.global_rank == 0:
            validation_guidance_scale = 1.0
            # (
            #     pixel_values_vid,
            #     pixel_values_ref_img,
            #     caption,
            # ) = next(loader_val)
            #print("zzzzzzzzzz111111")
            prompt_embeds, prompt_attention_mask = text_encoder.encode_prompt(prompt=caption[_])
            prompt_embeds, prompt_attention_mask = prompt_embeds.to(device),prompt_attention_mask.to(device)
            #print("zzzzzzzzzz")
            videos = []

            #prompt_embeds = prompt_embeds[nccl_info.group_id]
            #prompt_attention_mask = prompt_attention_mask[nccl_info.group_id]
            negative_prompt_embeds, negative_prompt_attention_mask = text_encoder.encode_prompt(prompt=negative_prompt)
            negative_prompt_embeds, negative_prompt_attention_mask = negative_prompt_embeds.to(device),negative_prompt_attention_mask.to(device)
            
            for validation_sampling_step in validation_sampling_steps:
                generator = torch.Generator(device="cpu").manual_seed(12345)
                video = sample_validation_video(
                            args.model_type,
                            transformer,
                            vae,
                            scheduler,
                            scheduler_type=scheduler_type,
                            num_frames=args.num_frames,
                            height=args.num_height,
                            width=args.num_width,
                            num_inference_steps=validation_sampling_step,
                            guidance_scale=validation_guidance_scale,
                            generator=generator,
                            prompt_embeds=prompt_embeds,
                            prompt_attention_mask=prompt_attention_mask,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_prompt_attention_mask=negative_prompt_attention_mask,
                            vae_spatial_scale_factor=vae_spatial_scale_factor,
                            vae_temporal_scale_factor=vae_temporal_scale_factor,
                            num_channels_latents=num_channels_latents,
                            shift = shift,
                        )[0]
                filename = os.path.join(
                            args.output_dir,
                            f"step_{global_step}_sample_{validation_sampling_step}_guidance_{validation_guidance_scale}_video.mp4",
                        )
                print(filename)
                export_to_video(video[0], filename, fps=fps)
#                 print("sssssssssssssssss",nccl_info.rank_within_group)
#                 if nccl_info.rank_within_group == 0:
#                     videos.append(video[0])
#                 # collect videos from all process to process zero

#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 # log if main process
#                 #torch.distributed.barrier()
#                 all_videos = [
#                         None for i in range(int(os.getenv("WORLD_SIZE", "1")))
#                     ]  # remove padded videos
#                 torch.distributed.all_gather_object(all_videos, videos)
#                 print("sssssssssssssssss",nccl_info.global_rank)
#                 if nccl_info.global_rank == 0:
#                     # remove padding
#                     videos = [video for videos in all_videos for video in videos]
#                     # linearize all videos
#                     video_filenames = []
#                     for i, video in enumerate(videos):
#                         filename = os.path.join(
#                             args.output_dir,
#                             f"step_{global_step}_sample_{validation_sampling_step}_guidance_{validation_guidance_scale}_video_{i}.mp4",
#                         )
#                         print(filename, video)
#                         save_videos_grid(video, filename, fps=24)
#                         #export_to_video(video, filename, fps=fps)
#                         #video_filenames.append(filename)
       # torch.distributed.barrier()
