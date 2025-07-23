# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps,EulerSolver)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .checkpoint import (resume_lora_optimizer, save_checkpoint,
                                        save_lora_checkpoint, resume_checkpoint, resume_training)

from diffusers.video_processor import VideoProcessor

import torch
import numpy as np
from diffusers.utils import export_to_video

import random
rand_num = random.random()  # 生成 [0.0, 1.0) 之间的随机数

def scale(vae,latents):
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
        
    # has_latents_mean = (hasattr(vae.model.config, "latents_mean")
    #                         and vae.model.config.latents_mean is not None)
    # has_latents_std = (hasattr(vae.model.config, "latents_std")
    #                        and vae.model.config.latents_std is not None)
    # if has_latents_mean and has_latents_std:
    #     latents_mean = (torch.tensor(vae.model.config.latents_mean).view(
    #             1, 12, 1, 1, 1).to(latents.device, latents.dtype))
    #     latents_std = (torch.tensor(vae.model.config.latents_std).view(
    #             1, 12, 1, 1, 1).to(latents.device, latents.dtype))
    #     latents = latents * latents_std / vae.model.config.scaling_factor + latents_mean
    # else:
    #     latents = latents / vae.model.config.scaling_factor
    # with torch.autocast("cuda", dtype=vae.dtype):
    with torch.no_grad():
        video = vae.decode(latents.to(torch.float32))[0]
    video_processor = VideoProcessor(
        vae_scale_factor=vae_spatial_scale_factor)
    video = video_processor.postprocess_video(video.unsqueeze(0), output_type="pil")
    return video


from wan.modules.model import WanModel  
from fastvideo.utils.checkpoint import resume_checkpoint_yume
import torch.nn as nn
import torch.nn.functional as F
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

from copy import deepcopy

class Yume:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device="cpu",
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        
        self.config = config


        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        #shard_fn = partial(shard_model, device_id=device_id)

    
        self.patch_size = config.patch_size

        
        #logging.info(f"Creating WanModel from {checkpoint_dir}")
        #self.model = WanModel.from_pretrained(checkpoint_dir)
        config_wan = {
            "model_type": "i2v",
            "text_len": 512,
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
            "_class_name": "WanModel",
            "_diffusers_version": "0.30.0",
        }
        self.model = WanModel.from_config(config_wan)  
        self.model.patch_embedding_2x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,4,4))
        self.model.patch_embedding_2x_f = torch.nn.Conv3d(36, 36, kernel_size=(1,4,4), stride=(1,4,4))
        self.model.patch_embedding_4x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,8,8))
        self.model.patch_embedding_8x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,16,16))
        self.model.patch_embedding_16x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,32,32))
        self.model = resume_checkpoint_yume(
            self.model,
            checkpoint_dir+"/Yume-Dit",
        )

        self.sp_size = 1
        self.sample_neg_prompt = config.sample_neg_prompt
        
    def init_model(self,
            config,
            checkpoint_dir,
            device_id=None,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=True,
        ):
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.device = torch.device(f"cuda:{device_id}")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device="cpu",
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )
        self.text_encoder.model.to(torch.bfloat16)
        self.text_encoder.model.eval().requires_grad_(False)
        if not t5_cpu:
            self.text_encoder.model.to(self.device)
            self.t5_device = self.device


        self.vae_stride = config.vae_stride
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        self.vae.model.eval().requires_grad_(False)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device="cpu",
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))
        self.clip.model.to(torch.bfloat16).to(self.device)
        self.clip.model.eval().requires_grad_(False)
    
    def generate(self,
                 model_input,
                 device,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 rand_num_img=None,
                 offload_model=False,
                 clip_context=None,
                 flag_sample=False,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        frame_num = model_input.shape[1]

        
        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w

        frame_zero = 33
        if flag_sample:
            frame_zero = 32


        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]


        if rand_num_img < 0.4:
            img = model_input[:,0,:,:]
        else:
            img = model_input[:,0:-frame_zero,:,:]


        if rand_num_img >= 0.4:
            model_input = torch.cat([ self.vae.encode([model_input[:,0:-frame_zero]])[0], self.vae.encode([model_input[:,-frame_zero:]])[0] ],dim=1)
        else:
            model_input = self.vae.encode([model_input])[0]

        
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])

        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
        noise = torch.randn_like(model_input)

        if rand_num_img < 0.4:    
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
        else:
            
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, -frame_zero:] = 0
            msk1 = msk
            msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                ],
                                dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]     

        
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            context = self.text_encoder(input_prompt, self.device)
        else:
            context = self.text_encoder(input_prompt, torch.device('cpu'))
            context = [context[0].to(self.device)]

        cache_path_null = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        if not self.t5_cpu:
            context_null = self.text_encoder([cache_path_null], self.device)
        else:
            context_null = self.text_encoder([cache_path_null], torch.device('cpu'))
            context_null = [context_null[0].to(self.device)]

        self.clip.model.to(self.device)
        if rand_num_img < 0.4:
            if clip_context == None:
                clip_context = self.clip.visual([img[:, None, :, :]])
        else:
            if clip_context == None:
                clip_context = self.clip.visual([img[:, -1, :, :].unsqueeze(1)])


        if rand_num_img < 0.4:
            y = self.vae.encode(
                [
                torch.concat([
                    img[None].cpu().transpose(0, 1),
                    torch.zeros(3, frame_num-1, h, w)
                ],
                            dim=1).to(self.device)
            ]
            )[0]
            y1 = y
        else:
            y = self.vae.encode(
                    [
                    torch.concat([
                        img.cpu(),
                        torch.zeros(3, frame_zero, h, w)
                    ],
                                dim=1).to(self.device)
                ]
                )[0]
            y1 = y
        y = torch.concat([msk, y])
          


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)


        # sample videos
        sigmas = torch.sigmoid(torch.randn((1,), device=self.device))
        timesteps = (sigmas * 1000).view(-1)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
        #print(y.shape)
        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }
        arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
        }
        latent_model_input = [noisy_model_input.to(self.device).squeeze()]
        timestep = [timesteps[0]]

        timestep = torch.stack(timestep).to(self.device)

        return latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null

    def generate_next(self,
                 model_input,
                 model_input_1,
                 device,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 rand_num_img=None,
                 offload_model=False,
                 clip_context=None,
                 flag_sample=False,):
        frame_zero = 33
        if flag_sample:
            frame_zero = 32
            
        frame_num = model_input.shape[1]
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
   
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        img = model_input
        
        model_input = model_input_1

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])

        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
       
        noise = torch.randn(
            16,
            ((F - 1) // self.vae_stride[0] + 1),
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=self.device)


        msk = torch.ones(1, frame_zero+img.shape[1], lat_h, lat_w, device=self.device)
        msk[:, -frame_zero:] = 0
        msk1 = msk
        msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                ],
                                dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            context = self.text_encoder(input_prompt, self.device)
        else:
            context = self.text_encoder(input_prompt, torch.device('cpu'))
            context = [context[0].to(self.device)]

        cache_path_null = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        if not self.t5_cpu:
            context_null = self.text_encoder([cache_path_null], self.device)
        else:
            context_null = self.text_encoder([cache_path_null], torch.device('cpu'))
            context_null = [context_null[0].to(self.device)]

        self.clip.model.to(self.device)
        if rand_num_img < 0.4:
            if clip_context == None:
                clip_context = self.clip.visual([img[:, None, :, :]])
        else:
            if clip_context == None:
                clip_context = self.clip.visual([img[:, -1, :, :].unsqueeze(1)])


        shape_y = msk.shape[1]
        shape_y = ( shape_y - 1 )*4 + 1
        shape_y = shape_y - img.shape[1]

        y = self.vae.encode(
                [
                    torch.concat([
                        img.cpu(),
                        torch.zeros(3, frame_zero, h, w)
                    ],
                                dim=1).to(self.device)
                ]
            )[0]

        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # sample videos
        sigmas = torch.sigmoid(torch.randn((1,), device=self.device))
        timesteps = (sigmas * 1000).view(-1)

        #print(y.shape)
        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }
        arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
        }

        timestep = [timesteps[0]]

        timestep = torch.stack(timestep).to(self.device)

        return None, timestep, arg_c, noise, model_input, clip_context, arg_null
