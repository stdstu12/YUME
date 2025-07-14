# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import datetime
import time
from collections import deque
import torch
import gc
import random
import wan
from wan.configs import WAN_CONFIGS
import torch.nn as nn
from packaging import version as pver
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from copy import deepcopy
import torch.nn.functional as F
import torch.distributed as dist
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import bitsandbytes as bnb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import numpy as np
from hyvideo.diffusion import load_denoiser
from fastvideo.dataset.t2v_datasets import (StableVideoAnimationDataset)
from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint,
                                        save_lora_checkpoint, resume_checkpoint, resume_training)
from fastvideo.utils.communications import (broadcast,
                                            sp_parallel_dataloader_wrapper)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing,
                                       get_dit_fsdp_kwargs,
                                      get_discriminator_fsdp_kwargs,
                                      get_DINO_fsdp_kwargs)
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group,
                                             get_sequence_parallel_state,
                                             initialize_sequence_parallel_state
                                             )

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)

def get_norm(model_pred, norms, gradient_accumulation_steps):
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) /
                              gradient_accumulation_steps)
    absolute_mean = torch.mean(
        torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(
        torch.abs(model_pred)) / gradient_accumulation_steps
    dist.all_reduce(fro_norm, op=dist.ReduceOp.AVG)
    dist.all_reduce(largest_singular_value, op=dist.ReduceOp.AVG)
    dist.all_reduce(absolute_mean, op=dist.ReduceOp.AVG)
    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(
        largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()

def latent_collate_function(latents,prompt_embeds,prompt_attention_masks):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    #prompt_embeds = torch.stack(prompt_embeds, dim=0)
    #prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latents, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks

def scale(vae,latents):
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
        
    with torch.no_grad():
        video = vae.decode([latents.to(torch.float32)])[0]
    video_processor = VideoProcessor(
        vae_scale_factor=vae_spatial_scale_factor)
    #print(video.shape,video)
    video = video_processor.postprocess_video(video.unsqueeze(0), output_type="pil")
    return video

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


def tensor_to_pil(tensor):
    array = ((tensor+1)/2.0).detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))  
    array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def distill_one_step(
    transformer,
    ema_transformer,
    optimizer,
    discriminator,
    discriminator_optimizer,
    lr_scheduler,
    loader,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    ema_decay,
    pred_decay_weight,
    pred_decay_type,
    device,
    vae=None,
    text_encoder=None,
    step=None,
    wan_i2v=None,
    denoiser=None,
    validation_steps=None,
    MVDT=False,
    Distil=False,
):
    total_loss = 0.0
    optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0, 
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16


    for _ in range(gradient_accumulation_steps):
        (
            pixel_values_vid,
            pixel_values_ref_img,
            caption,
            keys,
            mouse,
            videoid,
        ) = next(loader)

        frame_pixel = pixel_values_vid.shape[1]
        print(frame_pixel,"frame_pixelframe_pixelframe_pixelframe_pixelframe_pixelframe_pixelframe_pixelframe_pixelframe_pixelframe_pixel")
        frame_pixel = ( frame_pixel // 4 ) * 4 + 1
        if frame_pixel > pixel_values_vid.shape[1]:
            frame_pixel = frame_pixel - 4

        pixel_values_vid = pixel_values_vid[:,:frame_pixel]
            
        rand_num_img = random.random()  # i2v or v2v
        if pixel_values_vid.shape[1] <= 33:
            rand_num_img = 0.3

        multiphase = 1
        with torch.no_grad():
            pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
            pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
            latents = pixel_values_vid 
    
        model_input = latents 
        img = tensor_to_pil(pixel_values_ref_img)

        rand_num_img_flag = rand_num_img
        if rand_num_img < 0.4:
            model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:33]],dim=1)
            rand_num_img = 0.6
            rand_num_img_flag = 0.3

        
        latent_model_input, timestep, arg_c, noise, model_input, _ , arg_null= wan_i2v.generate(
            model_input,
            device,
            caption,
            img,
            max_area=544*960,
            frame_num=model_input.shape[1],
            sample_solver="unipc",
            sampling_steps=50,
            guide_scale=5.0,
            seed=None,
            rand_num_img=rand_num_img,
            offload_model=False)

        if MVDT:
            print("MVDTMVDTMVDTMVDTMVDTMVDTMVDT")
            # Incorporate masks during training
            _, _, _, loss_dict_mask = denoiser.training_losses(
                        transformer,
                        model_input,
                        arg_c,
                        n_tokens=None,
                        i2v_mode=None,
                        cond_latents=None,
                        args=args,
                        rand_num_img=rand_num_img,
                        enable_mask = True
            )
            loss = loss_dict_mask["loss"].mean()
            loss.backward()


        xt, t, model_output, loss_dict = denoiser.training_losses(
                        transformer,
                        model_input,
                        arg_c,
                        n_tokens=None,
                        i2v_mode=None,
                        cond_latents=None,
                        args=args,
                        rand_num_img=rand_num_img,
                        training_cache=True,
                        enable_mask = False
        )
        loss = loss_dict["loss"].mean()

        if Distil:
            print("DistilDistilDistilDistilDistil")
            model_denoing = xt - t*model_output
            model_denoing = model_denoing[:,-9:]
            model_input_gan = model_input[:,-9:]
            # # GAN Loss
            c1,f1,h1,w1 = model_denoing.shape
            b1 = 1
            bsz = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_real, pred_real_f, _ = discriminator(model_input_gan.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None) 
                pred_fake, pred_fake_f, _ = discriminator(model_denoing.permute(1,0,2,3).detach().reshape(b1*f1,c1,h1,w1), None) 

            pred_fake = torch.cat(pred_fake, dim=1)
            r1_lamda = 0
            pred_real = torch.cat(pred_real, dim=1)

            pred_fake_f = torch.cat(pred_fake_f, dim=1)
            pred_real_f = torch.cat(pred_real_f, dim=1)

            loss_real = torch.mean(torch.relu(torch.ones((bsz*f1,1)).to(pred_real.device) -pred_real))+torch.mean(torch.relu(torch.ones((bsz*196,1)).to(pred_real_f.device) - pred_real_f))
            loss_fake = torch.mean(torch.relu(torch.ones((bsz*f1,1)).to(pred_real.device) + pred_fake))+torch.mean(torch.relu(torch.ones((bsz*196,1)).to(pred_real_f.device) + pred_fake_f))

            loss_d = (loss_real + loss_fake)/2.0
            loss_d.backward()
            d_grad_norm = discriminator.clip_grad_norm_(max_grad_norm).item()
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

            gan_loss = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_fake, pred_fake_f, _ = discriminator(model_denoing.permute(1,0,2,3).reshape(b1*f1,c1,h1,w1), None)
            pred_fake = torch.cat(pred_fake, dim=1)
            pred_fake_f = torch.cat(pred_fake_f, dim=1)
            gan_loss = -torch.mean(pred_fake)-torch.mean(pred_fake_f)
            loss = loss + 0.01*gan_loss 

        loss.backward()

 
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if step % validation_steps == 0:
        sampling_sigmas = get_sampling_sigmas(50, 3.0)
        latent = noise.detach()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(50):
                    latent_model_input = [latent]

                    timestep = [sampling_sigmas[i]*1000]
                    timestep = torch.tensor(timestep).to(device)

                    noise_pred_cond, _ = transformer(\
                        latent_model_input, t=timestep, rand_num_img=rand_num_img, **arg_c)
                    noise_pred_uncond, _ = transformer(\
                        latent_model_input, t=timestep, rand_num_img=rand_num_img, **arg_null)

                    noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)
                    if i+1 == 50:
                        temp_x0 = latent[:,-9:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-9:,:,:]
                    else:
                        temp_x0 = latent[:,-9:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-9:,:,:]

                    latent = torch.cat([noise[:,:-9,:,:]*sampling_sigmas[max(i-1,0)]+(1-sampling_sigmas[max(i-1,0)])*model_input[:,:-9,:,:], temp_x0], dim=1)
        global_step = 1
        latent = latent[:,-9:,:,:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = scale(vae, latent)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_ori = scale(vae, model_input[:,-9:,:,:])

        # Ensure videoid is a string
        if isinstance(videoid, list):
            videoid_str = "_".join(map(str, videoid))
        else:
            videoid_str = str(videoid)

        if rand_num_img_flag < 0.4:
            filename = os.path.join(
                "./generated_video/",
                videoid_str+"_"+"_img_"+str(device)+".mp4",
            )
            export_to_video(video[0] , filename, fps=16)
            filename = os.path.join(
                "./generated_video/",
                videoid_str+"_"+"_img_"+str(device)+".mp4",
            )
            export_to_video(video_ori[0] , filename, fps=16)
        else:
            filename = os.path.join(
                "./generated_video/",
                videoid_str+"_"+str(device)+".mp4",
            )
            export_to_video(video[0] , filename, fps=16)
            filename = os.path.join(
                "./generated_video/",
                videoid_str+"_"+str(device)+".mp4",
            )
            export_to_video(video_ori[0] , filename, fps=16)


    # update ema                              
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(),
                                       transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(
                    torch.lerp(p_averaged.detach(), p_model.detach(),
                               1 - ema_decay))


    del model_output
    del pixel_values_vid
    del pixel_values_ref_img
    del caption
    del keys
    del mouse
    del videoid

    return total_loss, grad_norm.item(), model_pred_norm

def upsample_conv3d_weights(conv_small,size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                    
        size=size,                 
        mode='trilinear',         
        align_corners=False      
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create model:
    cfg = WAN_CONFIGS["i2v-14B"]
    ckpt_dir = "./Yume-I2V-540P"
    wan_i2v = wan.Yume(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device="cpu",
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )    
    from wan.modules.model import WanAttentionBlock,WanI2VCrossAttention,WanLayerNorm
    transformer = wan_i2v.model

    # transformer.patch_embedding_2x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,4,4))
    # transformer.patch_embedding_4x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,8,8))
    # transformer.patch_embedding_8x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,16,16))
    # transformer.patch_embedding_16x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,32,32))
    # transformer.patch_embedding_2x_f = torch.nn.Conv3d(36, 36, kernel_size=(1,4,4), stride=(1,4,4))

    if args.MVDT:
        transformer.sideblock = WanAttentionBlock("i2v_cross_attn", 5120, 13824, 40, (-1, -1), True, True, 1e-06)
        hidden_size = 5120 
        transformer.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, hidden_size, device=transformer.device)
        )
        torch.nn.init.normal_(transformer.mask_token, std=.02)
        transformer.mask_token.requires_grad = True

    # from safetensors.torch import load_file
    # state_dict = load_file("/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/outputs/checkpoint-1275/diffusion_pytorch_model.safetensors")
    # missing_keys, unexpected_keys = transformer.load_state_dict(state_dict)
    # print("missing_keys_unexpected_keys",missing_keys,unexpected_keys)
    # del state_dict

    if args.resume_from_checkpoint:
        (
            transformer,
            init_steps,
        ) = resume_checkpoint(
            transformer,
            args.resume_from_checkpoint,
        )

    # Referencing https://github.com/NJU-PCALab/AddSR, we have adapted the implementation to the OSV format.
    if args.Distil:
        from ADD.models.discriminator import ProjectedDiscriminator
        discriminator = ProjectedDiscriminator(c_dim=384).train()

    if args.use_ema:
        ema_transformer = deepcopy(transformer)
    else:
        ema_transformer = None
        
    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
        rank=device,
    )
    if args.Distil:
        discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(
            args.master_weight_type)

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0"
        ]
        transformer._no_split_modules = no_split_modules
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
        use_orig_params=True,
    )
    if args.Distil:
        discriminator = FSDP(
            discriminator,
            **discriminator_fsdp_kwargs,
            use_orig_params=True,
        )

    main_print("--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules,
                                 args.selective_checkpointing)
        if args.use_ema:
            apply_fsdp_checkpointing(ema_transformer, no_split_modules,
                                     args.selective_checkpointing)

    # Set model as trainable.
    transformer.train()
    if args.use_ema:
        ema_transformer.requires_grad_(False)
 
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = bnb.optim.Adam8bit(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
     )
    # optimizer = torch.optim.AdamW(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(0.9, 0.999),
    #     weight_decay=args.weight_decay,
    #     eps=1e-8,
    #  )

    if args.Distil:
        params_to_optimize_dis = discriminator.parameters()
        params_to_optimize_dis = list(
            filter(lambda p: p.requires_grad, params_to_optimize_dis))
        discriminator_optimizer = bnb.optim.Adam8bit(
            params_to_optimize_dis,
            lr=args.discriminator_learning_rate,
            betas=(0, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")
    init_steps_opt = 0

    # todo add lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps_opt - 1,
    )

    train_dataset = StableVideoAnimationDataset(height=544, width=960, n_sample_frames=33, sample_rate=1)

    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        #collate_fn=latent_collate_function,
        #prefetch_factor=1,
        pin_memory=False,
        batch_size=args.train_batch_size,
        num_workers=0,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps *
        args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)


    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps /
                        args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(
        f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError(
            "resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    #loader = train_dataloader
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # todo future
    for i in range(init_steps):
        print(i,init_steps)
        _ = next(loader)
        del _
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
    
    wan_i2v.init_model(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    wan_i2v.device = device
    vae = wan_i2v.vae
    wan_i2v.text_encoder.model.to(torch.bfloat16)
    fsdp_kwargs = get_DINO_fsdp_kwargs()    
    wan_i2v.text_encoder.model = FSDP(
        wan_i2v.text_encoder.model,
        **fsdp_kwargs,
        use_orig_params=True,
    )

    dist.barrier()

    denoiser = load_denoiser()
    
    if not args.Distil:
        discriminator = None
        discriminator_optimizer = None

    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        loss, grad_norm, pred_norm = distill_one_step(
            transformer,
            ema_transformer,
            optimizer,
            discriminator,
            discriminator_optimizer,
            lr_scheduler,
            loader,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.max_grad_norm,
            args.ema_decay,
            args.pred_decay_weight,
            args.pred_decay_type,
            device,
            vae = vae,
            text_encoder = None,
            step = step,
            wan_i2v = wan_i2v,
            denoiser = denoiser,
            validation_steps = args.validation_steps,
            MVDT = args.MVDT,
            Distil = args.Distil
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
        })
        progress_bar.update(1)
        
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank,
                                     args.output_dir, step)
            else:
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                # Your existing checkpoint saving code
                if args.use_ema:
                    save_checkpoint(ema_transformer, rank, args.output_dir,
                                    step)
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                             args.max_train_steps)
    else:
        save_checkpoint(transformer, rank, args.output_dir,
                        args.max_train_steps)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--MVDT", action="store_true") 
    parser.add_argument("--Distil", action="store_true") 


    # dataset & dataloader
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--validation_sampling_steps", type=str, default="64")

    parser.add_argument("--validation_steps", type=int, default=64)
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
         " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--discriminator_learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial discriminator learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help=
        "Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size",
                        type=int,
                        default=1,
                        help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=256,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=128,
                        help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type",
                        type=str,
                        default="pcm",
                        help="The scheduler type to use.")
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="Threshold for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.001,
                        help="Weight decay to apply.")
    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA.")
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)

