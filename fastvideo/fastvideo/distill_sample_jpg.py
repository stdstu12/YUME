# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import time
from collections import deque
from copy import deepcopy
import torch.nn.functional as F
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
import gc
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import bitsandbytes as bnb
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from hyvideo.diffusion import load_denoiser
from fastvideo.dataset.latent_datasets import (LatentDataset)
from fastvideo.dataset.t2v_datasets import (StableVideoAnimationDataset, SAmpleStableVideoAnimationDataset)

from fastvideo.distill.solver import EulerSolver, extract_into_tensor
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule
from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint,
                                        save_lora_checkpoint, resume_checkpoint, resume_training)
from fastvideo.utils.communications import (broadcast,
                                            sp_parallel_dataloader_wrapper)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing,
                                       get_dit_fsdp_kwargs,
                                      get_discriminator_fsdp_kwargs)
from fastvideo.utils.load import load_transformer,load_transformer_small
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group,
                                             get_sequence_parallel_state,
                                             initialize_sequence_parallel_state
                                             )
from fastvideo.utils.validation import log_validation
from fastvideo.utils.load import load_text_encoder, load_vae
import time
import torch.distributed as dist

from fastvideo.models.hunyuan.modules.t5 import T5EncoderModel
from fastvideo.models.hunyuan.modules.clip import CLIPModel
from fastvideo.models.hunyuan.modules.model import WanModel
from fastvideo.models.hunyuan.modules.vae import WanVAE

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


import torchvision
def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


import random


def project_null_space(x, A_operator):
    """投影到 A 的零空间：I - A†A"""
    # 将视频张量转换为 [B*C*F, H, W]
    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-2], original_shape[-1])
    
    # 计算 A†(A(x))
    Ax = A_operator.A(x_flat)
    A_pinv_Ax = A_operator.A_inv(Ax)
    
    # 恢复形状并计算 I_A_A_INV(x) = x - A†A(x)
    I_A_A_inv = x_flat - A_pinv_Ax
    return I_A_A_inv.view(original_shape)

class LinearOperator2D:
    def __init__(self, kernel_H, kernel_W, H, W, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.H, self.W = H, W
        
        # ------------------ 高度方向模糊算子 A_H (H x H) ------------------
        A_H = torch.zeros(H, H, device=self.device)
        kernel_size_H = kernel_H.shape[0]
        for i in range(H):
            for j in range(i - kernel_size_H//2, i + kernel_size_H//2):
                if 0 <= j < H:
                    A_H[i, j] = kernel_H[j - i + kernel_size_H//2]
        U_H, S_H, Vt_H = torch.linalg.svd(A_H, full_matrices=False)
        self.U_H, self.S_H, self.Vt_H = U_H.to(self.device), S_H.to(self.device), Vt_H.to(self.device)
        self.S_pinv_H = torch.where(S_H > 1e-6, 1/S_H, torch.zeros_like(S_H)).to(self.device)
        
        # ------------------ 宽度方向模糊算子 A_W (W x W) ------------------
        A_W = torch.zeros(W, W, device=self.device)
        kernel_size_W = kernel_W.shape[0]
        for i in range(W):
            for j in range(i - kernel_size_W//2, i + kernel_size_W//2):
                if 0 <= j < W:
                    A_W[i, j] = kernel_W[j - i + kernel_size_W//2]
        U_W, S_W, Vt_W = torch.linalg.svd(A_W, full_matrices=False)
        self.U_W, self.S_W, self.Vt_W = U_W.to(self.device), S_W.to(self.device), Vt_W.to(self.device)
        self.S_pinv_W = torch.where(S_W > 1e-6, 1/S_W, torch.zeros_like(S_W)).to(self.device)

    def A(self, x):
        """模糊操作: 先高度方向 (H)，再宽度方向 (W)"""
        # x 形状: [..., H, W]
        # 高度方向处理 ------------------------------------------------
        # 将 H 维度移动到最后，以便批量处理 [..., W, H]
        x_permuted = x.movedim(-2, -1)  # [..., W, H]
        # 应用 Vt_H @ x_permuted [..., W, H]
        x_h = torch.matmul(x_permuted, self.Vt_H.T)  # 注意维度对齐
        # 应用 S_H [..., W, H]
        x_h = torch.matmul(x_h, torch.diag(self.S_H))
        # 应用 U_H [..., W, H]
        x_h = torch.matmul(x_h, self.U_H.T)
        # 恢复原始维度 [..., H, W]
        x_h = x_h.movedim(-1, -2)  # [..., H, W]
        
        # 宽度方向处理 ------------------------------------------------
        # 应用 Vt_W @ x_h [..., H, W]
        x_hw = torch.matmul(x_h, self.Vt_W.T)
        # 应用 S_W [..., H, W]
        x_hw = torch.matmul(x_hw, torch.diag(self.S_W))
        # 应用 U_W [..., H, W]
        x_hw = torch.matmul(x_hw, self.U_W.T)
        return x_hw

    def A_inv(self, y):
        """伪逆操作: 先宽度方向 (W†)，再高度方向 (H†)"""
        # y 形状: [..., H, W]
        # 宽度方向伪逆 ------------------------------------------------
        # 应用 U_W @ y [..., H, W]
        y_w = torch.matmul(y, self.U_W)
        # 应用 S_pinv_W [..., H, W]
        y_w = torch.matmul(y_w, torch.diag(self.S_pinv_W))
        # 应用 Vt_W [..., H, W]
        y_w = torch.matmul(y_w, self.Vt_W)
        
        # 高度方向伪逆 ------------------------------------------------
        # 将 H 维度移动到最后，以便批量处理 [..., W, H]
        y_permuted = y_w.movedim(-2, -1)  # [..., W, H]
        # 应用 U_H @ y_permuted [..., W, H]
        y_hw = torch.matmul(y_permuted, self.U_H)
        # 应用 S_pinv_H [..., W, H]
        y_hw = torch.matmul(y_hw, torch.diag(self.S_pinv_H))
        # 应用 Vt_H [..., W, H]
        y_hw = torch.matmul(y_hw, self.Vt_H)
        # 恢复原始维度 [..., H, W]
        y_hw = y_hw.movedim(-1, -2)  # [..., H, W]
        return y_hw


def distill_loss(distill_double, distill_single, intermediate_double_s, intermediate_double_t, intermediate_single_s, intermediate_single_t):

    intermediate_double_t_index = distill_double
    intermediate_double_t = [x for i, x in enumerate(intermediate_double_t) if i in intermediate_double_t_index]
        
    intermediate_single_t_index = distill_single
    intermediate_single_t = [x for i, x in enumerate(intermediate_single_t) if i in intermediate_single_t_index]
    
    fn = F.mse_loss     
    loss_kd_double = {}
    for i, (feat_t, feat_s) in enumerate(zip(intermediate_double_t, intermediate_double_s)):
        img_t, txt_t = feat_t
        img_s, txt_s = feat_s

        loss_img = fn(img_s, img_t)
        loss_txt = fn(txt_s, txt_t)

        loss_kd_double[f'double_{str(i)}_img'] = loss_img
        loss_kd_double[f'double_{str(i)}_txt'] = loss_txt

    loss_kd_single = {}
    for i, (feat_t, feat_s) in enumerate(zip(intermediate_single_t, intermediate_single_s)):
        img_t, txt_t = feat_t
        img_s, txt_s = feat_s

        loss_img = fn(img_s, img_t)
        loss_txt = fn(txt_s, txt_t)

        loss_kd_single[f'single_{str(i)}_img'] = loss_img
        loss_kd_single[f'single_{str(i)}_txt'] = loss_txt

    return loss_kd_double, loss_kd_single
from diffusers.video_processor import VideoProcessor

import torch
import numpy as np
from diffusers.utils import export_to_video

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
        video = vae.decode([latents.to(torch.float32)])[0]

    return video

def save_video(video):
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
    video_processor = VideoProcessor(
        vae_scale_factor=vae_spatial_scale_factor)
    video = video_processor.postprocess_video(video.unsqueeze(0), output_type="pil")
    return video

from diffusers.utils import load_image

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

# 转换为 PIL.Image
def tensor_to_pil(tensor):
    # 1. 转换为 NumPy 数组
    array = ((tensor+1)/2.0).detach().cpu().numpy()  # 如果 tensor 在 GPU 上，先移到 CPU
    
    # 2. 调整形状为 (H, W, C)
    array = np.transpose(array, (1, 2, 0))  # 从 (C, H, W) 变为 (H, W, C)
    
    # 3. 转换为 [0, 255] 范围并转为 uint8
    array = (array * 255).astype(np.uint8)
    
    # 4. 创建 PIL 图像
    return Image.fromarray(array)


from packaging import version as pver
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

import random


def distill_one_step(
    transformer,
    model_type,
    teacher_transformer,
    ema_transformer,
    optimizer,
    discriminator,
    discriminator_optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    solver,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    ema_decay,
    pred_decay_weight,
    pred_decay_type,
    hunyuan_teacher_disable_cfg,
    device,
    vae=None,
    text_encoder=None,
    clip = None,
    source_idx_double=None,
    source_idx_single=None,
    step=None,
    wan_i2v=None,
    denoiser=None,
    video_tensors=None,
    rank = None,
):
    total_loss = 0.0
    model_pred_norm = {
        "fro": 0.0,  # codespell:ignore
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
    negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    
    if True:
        model_input = video_tensors[(rank+(step-1)*8)%len(video_tensors)]

        model_input = model_input[:,0].unsqueeze(1).repeat(1,49,1,1).to(device)
        model_input = (model_input-0.5)*2
        rand_num_img = 0.6
        img = tensor_to_pil(model_input[:,0])
        
        videoid = str((rank+(step-1)*8))+"_"+str((rank+(step-1)*8)%len(video_tensors))
        caption = ["jwjsnb"]
        caption[0] = "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179."
        plucker = None
        model_input_de = model_input.squeeze()
        #print(model_input_de.shape,"q0odhx0a9hwd")
        latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null = wan_i2v.generate(
            model_input,
            device,
            caption,
            img,
            max_area=544*960,
            frame_num=model_input.shape[1],
            shift=17,
            sample_solver="unipc",
            sampling_steps=50,
            guide_scale=5.0,
            seed=None,
            rand_num_img=rand_num_img,
            offload_model=False,
            flag_sample=True, )
        #print(model_input.shape,"model_inputmodel_inputmodel_inputmodel_inputmodel_inputmodel_inputmodel_inputmodel_inputmodel_input--")
        frame_zero = 32
        latent_frame_zero = (frame_zero-1)//4 + 1

        # prompts = ["This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward and left (W+A).Camera turns right (→).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward and left (W+A).Camera turns right (→).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward and left (W+A).Camera turns right (→).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person stands still (·).Camera tilts up (↑).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person stands still (·).Camera tilts up (↑).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
        #           "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179."]
        prompts = ["This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179.",
                  "This video depicts a city walk scene with a first-person view (FPV).Person moves forward (W).Camera remains still (·).Actual distance moved:4.3697374288015297at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179."]

        print(step,"step---------------------------------------------------------------")
        if step % 1 == 0:
            sampling_sigmas = get_sampling_sigmas(50, 3.0)
            video_all = []
            for step_sample in range(13):
                c1,f1,h1,w1 = model_input.shape
                
                if step_sample > 0:
                    c1,f1,h1,w1 = model_input_1.shape
                kernel_H = torch.tensor([0.1, 0.8, 0.1], device='cpu')  # 高度方向模糊核
                kernel_W = torch.tensor([0.2, 0.6, 0.2], device='cpu')  # 宽度方向模糊核
                A_op = LinearOperator2D(kernel_H, kernel_W, h1, w1, device=device)
                A_op_1 = LinearOperator2D(kernel_H, kernel_W, h1*8, w1*8, device=device)


                if step_sample > 0:
                    noise = torch.randn_like(model_input_1)#torch.cat([noise,torch.randn(c1,8,h1,w1,device=device)],dim=1)
                    latent = noise.clone()

                    # latent = torch.randn(
                    #                 c1,
                    #                 f1+8,
                    #                 h1,
                    #                 w1,
                    #                 device=device)
                    print(latent.shape,model_input.shape,"latentlatentlatentlatent---")
                    #torch.Size([16, 43, 68, 120]) latentlatentlatentlatent---
                else:
                    latent = noise
                    # latent = torch.randn(
                    #                 c1,
                    #                 f1,
                    #                 h1,
                    #                 w1,
                    #                 device=device)

                import time
                start_time = time.time()
                sample_step = 50
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # 时间旅行参数配置
                        time_travel_step = 5  # 时间旅行步长l
                        time_travel_interval = 5  # 时间旅行间隔s (每10步执行一次)
                        time_travel_repeat = 1  # 重复次数r

                        for i in range(50):
                            print("iii-----------",i)
                            # === 1. 保存当前状态用于时间旅行 ===
                            #if i % time_travel_interval == 0 and i < 49:
                            #    latent_original = latent.clone()
                            
                            # === 2. 执行原始采样操作 ===
                            latent_model_input = [latent]
                            timestep = [sampling_sigmas[i] * 1000]
                            timestep = torch.tensor(timestep).to(device)
                            
                            # 条件预测
                            noise_pred_cond = transformer(
                                latent_model_input, 
                                t=timestep, 
                                noise=noise, 
                                rand_num_img=rand_num_img, 
                                plucker=plucker, 
                                plucker_train=True, 
                                latent_frame_zero=latent_frame_zero, 
                                **arg_c
                            )[0]
                            
                            # 无条件预测
                            noise_pred_uncond = transformer(
                                latent_model_input, 
                                t=timestep, 
                                noise=noise, 
                                rand_num_img=rand_num_img, 
                                plucker=plucker, 
                                plucker_train=False, 
                                latent_frame_zero=latent_frame_zero, 
                                **arg_null
                            )[0]
                            
                            # 组合条件与无条件预测
                            noise_pred_cond = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
                            
                            # 计算x0估计
                            if i + 1 == 50:
                                temp_x0 = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]
                            else:
                                temp_x0 = latent[:, -latent_frame_zero:, :, :] + (sampling_sigmas[i+1] - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]


                            prev_sample_mean = temp_x0
                            pred_original_sample = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]
                            eta = 0.3
                            if i + 1 == 50:
                                delta_t = 0 #(sampling_sigmas[i] - 0) #sigma - sigmas[index + 1]
                            else:
                                delta_t = (sampling_sigmas[i] - sampling_sigmas[i+1]) #sigma - sigmas[index + 1]
                            if delta_t < 0:
                                delta_t = 0
                            if i + 1 == 50:
                                dsigma = 0 - sampling_sigmas[i]
                            else:
                                dsigma = sampling_sigmas[i+1] - sampling_sigmas[i]
                            std_dev_t = eta * math.sqrt(delta_t)
                            score_estimate = -(latent[:, -latent_frame_zero:, :, :]-pred_original_sample*(1 - sampling_sigmas[i]))/sampling_sigmas[i]**2
                            log_term = -0.5 * eta**2 * score_estimate
                            prev_sample_mean = prev_sample_mean + log_term * dsigma
                            temp_x0 = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
                                            

                            # === 3. 添加时间旅行操作 ===
                            if i % time_travel_interval == 0:
                                #current_latent = latent.clone()     
                                latent_original = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]   
                                noise_ori_pred_cond = noise_pred_cond[:, -latent_frame_zero:, :, :] + latent_original

                                noise_ped_v = noise_pred_cond

                                if True: #for _ in range(1):
                                    
                                    # # 3.1 正向扩散到t+l (增加噪声)
                                    # travel_step = min(49, i + time_travel_step)

                                    # r = 0.8  # 70%来自a，30%来自b
                                    # # 计算归一化权重
                                    # w_a = math.sqrt(r)
                                    # w_b = math.sqrt(1 - r)

                                    # # 合成新噪声
                                    noise_travel = noise_ori_pred_cond #w_a * noise_ori_pred_cond + w_b * torch.randn_like(latent_original)


                                    # sigma_diff = sampling_sigmas[travel_step] - sampling_sigmas[i] 
                                    # #latent_travel = latent_original + sigma_diff * (noise_travel-latent_original)
                                    # latent_travel = (1-sampling_sigmas[travel_step])*latent_original + sampling_sigmas[travel_step] * noise[:, -latent_frame_zero:, :, :]
                                    
                                    # # 更新潜在状态
                                    # index1 = travel_step #min(49, i + 1)
                                    # if step_sample > 0:
                                    #     latent_travel = torch.cat([
                                    #         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                    #         (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                                    #         latent_travel
                                    #     ], dim=1)
                                    # else:
                                    #     latent_travel = torch.cat([
                                    #         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                    #         (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                                    #         latent_travel
                                    #     ], dim=1)

                                    travel_step = min(49, i + time_travel_step)
                                    index1 = travel_step
                                    if step_sample > 0:
                                        latent_travel = torch.cat([
                                            noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                            (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                                            temp_x0
                                        ], dim=1)
                                    else:
                                        latent_travel = torch.cat([
                                            noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                            (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                                            temp_x0
                                        ], dim=1)

                                    # 3.2 从t+l反向采样回t-1
                                    for j in range(i+1, travel_step):
                                        print("ji------------",j,i)
                                        # 准备输入
                                        latent_model_input_travel = [latent_travel]
                                        travel_timestep = [sampling_sigmas[j] * 1000]
                                        travel_timestep = torch.tensor(travel_timestep).to(device)
                                        
                                        # 条件预测（时间旅行）
                                        noise_pred_cond_travel = transformer(
                                            latent_model_input_travel, 
                                            t=travel_timestep, 
                                            noise=noise_travel, 
                                            rand_num_img=rand_num_img, 
                                            plucker=plucker, 
                                            plucker_train=True, 
                                            latent_frame_zero=latent_frame_zero, 
                                            **arg_c
                                        )[0]
                                        
                                        # 无条件预测（时间旅行）
                                        noise_pred_uncond_travel = transformer(
                                            latent_model_input_travel, 
                                            t=travel_timestep, 
                                            noise=noise_travel, 
                                            rand_num_img=rand_num_img, 
                                            plucker=plucker, 
                                            plucker_train=False, 
                                            latent_frame_zero=latent_frame_zero, 
                                            **arg_null
                                        )[0]
                                        
                                        # 组合预测（时间旅行）
                                        noise_pred_cond_travel = noise_pred_uncond_travel + 5.0 * (noise_pred_cond_travel - noise_pred_uncond_travel)
                                        
                                        #noise_pred_cond_travel = (noise_ped_v+noise_pred_cond_travel)/2.0
                                        # 计算下一步的x0估计（时间旅行）
                                        # if j - 1 == i:
                                        #     temp_x0_travel = latent_travel[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                                        # else:
                                        temp_x0_travel = latent_travel[:, -latent_frame_zero:, :, :] + (sampling_sigmas[j+1] - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                                        
                                        prev_sample_mean = temp_x0_travel
                                        pred_original_sample = latent_travel[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                                        eta = 0.3
                                        delta_t = (sampling_sigmas[j] - sampling_sigmas[j+1]) #sigma - sigmas[index + 1]
                                        if delta_t < 0:
                                            delta_t = 0
                                        dsigma = sampling_sigmas[j+1] - sampling_sigmas[j]
                                        print(sampling_sigmas[j],sampling_sigmas[j+1],sampling_sigmas[j] - sampling_sigmas[j+1],"cj0ajc0a")
                                        #0.09259259259259267 1.0 -0.9074074074074073 cj0ajc0a
                                        std_dev_t = eta * math.sqrt(delta_t)
                                        score_estimate = -(latent_travel[:, -latent_frame_zero:, :, :]-pred_original_sample*(1 - sampling_sigmas[j]))/sampling_sigmas[j]**2
                                        log_term = -0.5 * eta**2 * score_estimate
                                        prev_sample_mean = prev_sample_mean + log_term * dsigma
                                        temp_x0_travel = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
                                            


                                        # 更新潜在状态（时间旅行）
                                        index_travel = min(49, j + 1)
                                        print(noise_travel.shape,model_input[:, :-latent_frame_zero, :, :].shape,"model_input_1[:, :-latent_frame_zero, :, :]")
                                        if step_sample > 0:
                                            latent_travel = torch.cat([
                                                noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index_travel] + 
                                                (1 - sampling_sigmas[index_travel]) * model_input_1[:, :-latent_frame_zero, :, :], 
                                                temp_x0_travel
                                            ], dim=1)
                                        else:
                                            latent_travel = torch.cat([
                                                noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index_travel] + 
                                                (1 - sampling_sigmas[index_travel]) * model_input[:, :-latent_frame_zero, :, :], 
                                                temp_x0_travel
                                            ], dim=1)
                                        current_pred = noise_pred_cond_travel
                                    
                                    # 保存时间旅行结果
                                    current_latent = latent_travel.clone()
                                
                                # 3.3 用时间旅行结果替换原始采样结果
                                #latent = current_latent

                                # 计算x0估计
                                if i + 1 == 50:
                                    temp_x0 = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * current_pred[:, -latent_frame_zero:, :, :]
                                else:
                                    temp_x0 = latent[:, -latent_frame_zero:, :, :] + (sampling_sigmas[i+1] - sampling_sigmas[i]) * current_pred[:, -latent_frame_zero:, :, :]
                                # 更新潜在状态
                                index1 = min(49, i + 1)
                                if step_sample > 0:
                                    latent = torch.cat([
                                        noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                        (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                                        temp_x0
                                    ], dim=1)
                                else:
                                    latent = torch.cat([
                                        noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                        (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                                        temp_x0
                                    ], dim=1)
                            else:
                                # 更新潜在状态
                                index1 = min(49, i + 1)
                                if step_sample > 0:
                                    latent = torch.cat([
                                        noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                        (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                                        temp_x0
                                    ], dim=1)
                                else:
                                    latent = torch.cat([
                                        noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                                        (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                                        temp_x0
                                    ], dim=1)


                    
                # with torch.no_grad():
                #     with torch.autocast("cuda", dtype=torch.bfloat16):
                #         cache = None
                #         cache_null = None
                #         for i in range(50):
                #             #print("range(10)",step_sample)
                #             latent_model_input = [latent]
                #             #print(latent,"latent......")
                #             timestep = [sampling_sigmas[i]*1000]
                #             timestep = torch.tensor(timestep).to(device)
                #             print(timestep)
                #             #plucker = noise.permute(0,2,3,4,1)*sampling_sigmas[i] + plucker*(1-sampling_sigmas[i])
                #             print("range(10)",step_sample,latent_model_input[0].shape,arg_c['y'][0].shape,model_input.shape)
                #             #range(10) 1 torch.Size([16, 22, 68, 120]) torch.Size([20, 30, 68, 120]) torch.Size([16, 22, 68, 120])
                #             if i%4==0:
                #                 return_cache=True
                #                 cache_list=[20,21,22,23,24,25,26,27,28,29,30,32,34,36,38]
                #                 del cache
                #                 del cache_null
                #                 cache = None
                #                 cache_null = None
                #             else:
                #                 return_cache=True #False
                #                 cache_list=[20,21,22,23,24,25,26,27,28,29,30,32,34,36,38]
                #             noise_pred_cond = transformer(\
                #                 latent_model_input, t=timestep, noise=noise,cache rand_num_img=rand_num_img,cache=cache,return_cache=return_cache,cache_list=cache_list, \
                #                 plucker=plucker, cache_sample=False, plucker_train=True, **arg_c)
                #             noise_pred_uncond = transformer(\
                #                     latent_model_input, t=timestep, noise=noise, rand_num_img=rand_num_img,cache=cache_null,return_cache=return_cache,cache_list=cache_list,  \
                #                     plucker=plucker, cache_sample=False, plucker_train=False, **arg_null)
                #             if return_cache:
                #                 noise_pred_cond,cache=noise_pred_cond
                #                 noise_pred_uncond,cache_null=noise_pred_uncond
                         

                #             noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)
                #             if i+1 == 50:
                #                 temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                #             else:
                #                 #torch.Size([16, 35, 68, 120]) torch.Size([16, 13, 68, 120]) noise_pred_condnoise_pred_cond
                #                 #print(latent.shape,noise_pred_cond.shape,"noise_pred_condnoise_pred_cond")
                #                 temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                #             #print(temp_x0.shape,noise.shape,model_input.shape,(noise[:,:-9,:,:]*sampling_sigmas[i]+(1-sampling_sigmas[i])*model_input[:,:-9,:,:]).shape,"64964346494")
                #             index1 = min(49,i+1)
                #             if step_sample > 0:
                #                 latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input_1[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                #             else:
                #                 latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                
                        # latent_ori = latent
                        # C,F,H,W = noise.shape
                        # latent = noise

                        # for i in range(50):
                        #     timestep = [sampling_sigmas[i]*1000]
                        #     timestep = torch.tensor(timestep).to(device)
                        #     if i <= 4:
                        #         latent_ori_noise = sampling_sigmas[i]*noise+(1-sampling_sigmas[i])*latent_ori
                        #         A_pinv_Ax = A_op.A_inv(A_op.A(latent_ori_noise.view(-1, H, W))).view_as(latent_ori_noise)
                        #         I_A_A_inv = project_null_space(latent, A_op)
                        #         latent = A_pinv_Ax + I_A_A_inv

                        #     latent_model_input = [latent]
                            
                            
                        #     noise_pred_cond = transformer(
                        #             latent_model_input, t=timestep, noise=noise, rand_num_img=rand_num_img, plucker=plucker, plucker_train=True,latent_frame_zero=latent_frame_zero, **arg_c)[0]
                        #     noise_pred_uncond = transformer(
                        #             latent_model_input, t=timestep, noise=noise, rand_num_img=rand_num_img, plucker=plucker, plucker_train=False,latent_frame_zero=latent_frame_zero, **arg_null)[0]
                        #     noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)
                        #     if i+1 == 50:
                        #         temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                        #     else:
                        #         #torch.Size([16, 35, 68, 120]) torch.Size([16, 13, 68, 120]) noise_pred_condnoise_pred_cond
                        #         #print(latent.shape,noise_pred_cond.shape,"noise_pred_condnoise_pred_cond")
                        #         temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                        #     #print(temp_x0.shape,noise.shape,model_input.shape,(noise[:,:-9,:,:]*sampling_sigmas[i]+(1-sampling_sigmas[i])*model_input[:,:-9,:,:]).shape,"64964346494")
                        #     index1 = min(49,i+1)
                        #     if step_sample > 0:
                        #         latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input_1[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                        #     else:
                        #         latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"函数运行时间: {elapsed:.4f} 秒")
                # 函数运行时间: 553.6614 秒 
                
                # 函数运行时间: 566.1119 秒
                # 函数运行时间: 720.5872 秒
                #zzzz
                
                global_step = 1
                #latent = latent[:,-9:,:,:]
                print(model_input.shape,"model_input1")
                if step_sample > 0:
                    model_input_2 = model_input_1[:,-3:]
                    model_input = torch.cat([model_input, latent[:,-latent_frame_zero:,:,:]],dim=1)
                else:
                    model_input_2 = model_input[:,:-latent_frame_zero]
                    model_input_2 = model_input_2[:,-3:]
                    model_input = torch.cat([model_input[:,:-latent_frame_zero,:,:], latent[:,-latent_frame_zero:,:,:]],dim=1)
                    print(model_input.shape,"model_input2")
            
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    #latent1 = torch.cat([model_input_2,latent[:,-latent_frame_zero:,:,:]],dim=1)
                    #latent1 = torch.cat([latent[:,-(latent_frame_zero+3):-latent_frame_zero,:,:],latent[:,-latent_frame_zero:,:,:]],dim=1)
                    
                    video1 = scale(vae, model_input) #latent[:,-latent_frame_zero:,:,:])
                    #video = scale(vae, latent1)
                    video = video1[:,-frame_zero:]
                    #print(video.shape,"video.shape")#torch.Size([3, 29, 544, 960]) video.shape
                    video_all.append(video)
                    # f_m = model_input_de.shape[1]
                    # f_v = video.shape[1] - 1 + f_m
                    # f_v1 = (f_v//4)*4
                    # f_v1 = f_v1 - f_m
                    # model_input_de = torch.cat([model_input_de,video[:,1:f_v1+1]],dim=1)
                    print(model_input_de.shape,"model_input_de1")
                    if step_sample > 0:
                        model_input_de = torch.cat([model_input_de, video[:,-frame_zero:,:,:]],dim=1)
                    else:
                        model_input_de = torch.cat([model_input_de[:,:-frame_zero,:,:], video[:,-frame_zero:,:,:]],dim=1)
                    print(model_input_de.shape,video.shape,frame_zero,"model_input_de2")
                    #print(model_input_de_1.shape, model_input_de.shape,torch.cat([model_input_de,torch.zeros_like(model_input_de)[:,:33]],dim=1).shape,"89hahd0a")
                    video = save_video(torch.cat(video_all,dim=1))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video_ori = scale(vae, model_input[:,-latent_frame_zero:,:,:])
                    video_ori = save_video(video_ori)
                
                # Ensure videoid is a string
                if isinstance(videoid, list):
                    videoid_str = "_".join(map(str, videoid))
                else:
                    videoid_str = str(videoid)
                #print("/mnt/hwfile/gveval/maoxiaofeng/FastVideo_i2v_pack/test/")

                print(rand_num_img,"/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/test_jpg/")
                filename = os.path.join(
                                        "/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/test_jpg/",
                                        videoid_str+"_"+"_pluckernormnew"+str(device)+"_"+str(step_sample)+"_"+".mp4",
                                    )
                export_to_video(video[0] , filename, fps=16)
                filename = os.path.join(
                                        "/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/test_jpg/",
                                        videoid_str+"_"+"_pluckernormnew_ori"+str(device)+"_"+str(step_sample)+"_"+".mp4",
                                    )#_plucker是不加mask但是有plucker,_pluckermask是加mask有plucker,_pluckernorm是普通3.0版本但是有plucker
                export_to_video(video_ori[0] , filename, fps=16)


                # _, _, arg_c, _, model_input_1, clip_context = wan_i2v.generate(
                #     torch.cat([model_input_de,torch.zeros_like(model_input_de)[:,:33]],dim=1),
                #     device,
                #     caption,
                #     img,
                #     max_area=544*960,
                #     frame_num=model_input.squeeze().shape[1],
                #     shift=17,
                #     sample_solver="unipc",
                #     sampling_steps=50,
                #     guide_scale=5.0,
                #     seed=None,
                #     rand_num_img=rand_num_img,
                #     offload_model=False,
                #     clip_context=clip_context,
                # )
                if step_sample > 0:
                    ins = 3
                else:
                    ins = 0
                _, _, arg_c, _, model_input_1, clip_context, arg_null = wan_i2v.generate_next(
                    model_input_de,
                    model_input,
                    device,
                    prompts[step_sample],
                    img,
                    max_area=544*960,
                    frame_num=model_input.squeeze().shape[1],
                    shift=17,
                    sample_solver="unipc",
                    sampling_steps=50,
                    guide_scale=5.0,
                    seed=None,
                    rand_num_img=rand_num_img,
                    offload_model=False,
                    clip_context=clip_context,
                    ins=ins,
                    flag_sample=True,
                )
                model_input_1 = torch.cat([model_input_1,torch.zeros(16,latent_frame_zero,model_input_1.shape[2],model_input_1.shape[3]).to(device)],dim=1)
        import torchvision.io

        # 输入示例：形状为 (C, F, H, W) 的Tensor，值范围[0,1]
        #tensor = torch.rand(3, 60, 256, 256)  # 3通道，60帧，256x256分辨率
        # with torch.no_grad():
        #     print(pixel_values_vid.shape,"pixel_values_vidpixel_values_vidpixel_values_vidpixel_values_vidpixel_values_vidpixel_values_vidpixel_values_vidpixel_values_vid")
        #     video_tensor = ((pixel_values_vid+1)/2.0* 255).byte() 
        #     video_tensor = video_tensor.permute(1, 2, 3, 0)
        #     filename = os.path.join(
        #                             "/mnt/hwfile/gveval/maoxiaofeng/FastVideo_i2v/test/",
        #                             "_"+str(0)+"_"+str(device)+".mp4",
        #                         )
        #     torchvision.io.write_video(filename, video_tensor, fps=16)
        #     zzzzzz 

        # with torch.autocast("cuda", dtype=torch.bfloat16):
        #    video = scale(vae, model_input)
        # filename = os.path.join(
        #                         "/mnt/hwfile/gveval/maoxiaofeng/FastVideo_i2v/test/",
        #                         "_"+str(0)+"_"+str(device)+".mp4",
        #                     )
        # #print(video[0].shape,pixel_values_vid.shape,"/mnt/hwfile/gveval/maoxiaofen")
        # export_to_video(video[0], filename, fps=16)
        # zzzzz


        # calculate model_pred norm and mean
        #get_norm(model_pred.detach().float(), model_pred_norm,
        #         gradient_accumulation_steps)
        
        total_loss = 0.0

    grad_norm = 0.0

    return 0.0, 0.0, 0.0



import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample_conv3d_weights(conv_small,size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                      # 输入张量
        size=size,                  # 目标尺寸（时间维度不变）
        mode='trilinear',                # 3D插值
        align_corners=False              # 避免边缘对齐伪影
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    # 如果有偏置项，直接复制（无需修改）
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 为每个rank设置独立的缓存目录
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{rank}"
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    torch.cuda.set_device(local_rank)
    #torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)
    print(world_size,rank,local_rank,device,"world_sizeworld_sizeworld_sizeworld_sizeworld_sizeworld_sizeworld_size")

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:

    main_print(f"--> loading model from {args.dit_model_name_or_path}")

    cfg = WAN_CONFIGS["i2v-14B"]
    ckpt_dir = "/mnt/petrelfs/maoxiaofeng/FastVideo_i2v/Wan2.1-I2V-14B-480P"
    #print(wan.WanI2V)
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device="cpu",
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )    
    from wan.modules.model import ConvNext3D,DoubleFCWithNorm,WanAttentionBlock,WanI2VCrossAttention,WanLayerNorm,DoubleFCWithNorm_nozero
    transformer = wan_i2v.model
    # 添加 ConvNext3D 作为 pose_encoder
    print(transformer.device,"transformertransformertransformertransformertransformertransformer")
    #transformer.pose_encoder = torch.nn.Sequential(ConvNext3D())  # 替换 ... 为适当的参数
    transformer.DoubleFCWithNorm = DoubleFCWithNorm(5120,5120)
    transformer.DoubleFCWithNorm1 = torch.nn.Conv3d(256, 5120, kernel_size=(1,2,2), stride=(1,2,2)) #DoubleFCWithNorm(256,5120)
    #transformer.DoubleFCWithNorm2 = torch.nn.Conv3d(256, 5120, kernel_size=(1,2,2), stride=(1,1,1)) #DoubleFCWithNorm(256,5120)

    transformer.patch_embedding_2x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,4,4))
    transformer.patch_embedding_4x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,8,8))
    transformer.patch_embedding_8x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,16,16))
    transformer.patch_embedding_16x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,32,32))
    #transformer.patch_embedding_32x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,32,32))
    transformer.patch_embedding_2x_f = torch.nn.Conv3d(36, 36, kernel_size=(1,4,4), stride=(1,4,4))




    transformer.sideblock = WanAttentionBlock("i2v_cross_attn", 5120, 13824, 40, (-1, -1), True, True, 1e-06)
    # 3. 统一mask_token处理（修正self/transformer引用问题）
    hidden_size = 5120  # 假设的hidden_size，需根据实际情况修改
    transformer.mask_token = torch.nn.Parameter(
        torch.zeros(1, 1, hidden_size, device=transformer.device)
    )
    torch.nn.init.normal_(transformer.mask_token, std=.02)
    transformer.mask_token.requires_grad = True
    # 遍历所有子模块，找到 WanAttentionBlock 并添加 DoubleFCWithNorm 作为 mlp_e
    #cnt = 0
    # for module in transformer.modules():
    #     if isinstance(module, WanAttentionBlock):
    #         cnt+=1
    #         #print(module)
    #         module.mlp_e = torch.nn.Sequential(DoubleFCWithNorm(96, 5120))  # 替换 ... 为适当的参数
    #         module.mlp_e_x = torch.nn.Sequential(DoubleFCWithNorm_nozero(5120, 96))
    #         module.mlp_e_x1 = torch.nn.Sequential(DoubleFCWithNorm_nozero(96, 5120))
    #         # 创建Conv3d并初始化为0
    #         # conv3d = torch.nn.Conv3d(
    #         #     in_channels=96,
    #         #     out_channels=96,
    #         #     kernel_size=(3, 3, 3),
    #         #     padding=(1, 1, 1),
    #         #     stride=1
    #         # )
    #         # # 将权重和偏置初始化为0
    #         # with torch.no_grad():
    #         #     conv3d.weight.data.zero_()  # 权重初始化为0
    #         #     if conv3d.bias is not None:
    #         #         conv3d.bias.data.zero_()  # 偏置初始化为0
    #         # module.mlp_e_ori_conv = torch.nn.Sequential(conv3d)  # 替换 ... 为适当的参数
    #         module.norm_plucker =  WanLayerNorm(96, 1e-6,elementwise_affine=True)
    #         module.plucker_cross = WanI2VCrossAttention(96,4,(-1, -1),True,1e-6)
    # print(cnt,"cntcntcntcntcnt")
    


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

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
        use_orig_params=True,
    )
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules,
                                 args.selective_checkpointing)
        if args.use_ema:
            apply_fsdp_checkpointing(ema_transformer, no_split_modules,
                                     args.selective_checkpointing)
    
    # Set model as eval.
    transformer.eval()

    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    if args.scheduler_type == "pcm_linear_quadratic":
        linear_steps = int(noise_scheduler.config.num_train_timesteps *
                           args.linear_range)
        sigmas = linear_quadratic_schedule(
            noise_scheduler.config.num_train_timesteps,
            args.linear_quadratic_threshold,
            linear_steps,
        )
        sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
    else:
        sigmas = noise_scheduler.sigmas
    solver = EulerSolver(
        sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
    )
    solver.to(device)

    init_steps = 0

    if args.resume_from_checkpoint:
        (
            transformer,
            init_steps,
        ) = resume_checkpoint(
            transformer,
            args.resume_from_checkpoint,
        )

    init_steps_opt = 0
    # todo add lr scheduler

    train_dataset = SAmpleStableVideoAnimationDataset(height=544, width=960, n_sample_frames=81, sample_rate=1, training=False)
    train_dataset_val = SAmpleStableVideoAnimationDataset(height=544, width=960, n_sample_frames=81, sample_rate=1, training=False)

    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    sampler_val = DistributedSampler(
        train_dataset_val, rank=rank, num_replicas=world_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        #collate_fn=latent_collate_function,
        pin_memory=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        train_dataset_val,
        sampler=sampler_val,
        #collate_fn=latent_collate_function,
        pin_memory=False,
        batch_size=world_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
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


    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )
    loader_val = sp_parallel_dataloader_wrapper(
        val_dataloader,
        device,
        world_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    init_steps = 0
    # todo future
    for i in range(init_steps):
        next(loader)

    wan_i2v.init_model(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    vae = wan_i2v.vae
    

    

    discriminator = None
    dist.barrier()
    transformer.guidance_embed = False
    teacher_transformer = None
    
    wan_i2v.device = device
    denoiser = load_denoiser()
    import cv2

    def create_scaled_videos(folder_path, total_frames=30, H1=256, W1=256):
        """创建缩放到指定尺寸的视频张量列表"""
        # 获取所有图像文件路径
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        
        video_list = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 转换为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 转换为PyTorch张量并归一化
                img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                
                # 创建全零视频张量 (C, F, H, W)
                C, H, W = img_tensor.shape
                video_tensor = torch.zeros(C, total_frames, H1, W1)
                
                # 缩放并放入第0帧
                resized_frame = F.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(H1, W1),
                    mode='bilinear',
                    align_corners=False
                )[0]
                
                video_tensor[:, 0] = resized_frame
                video_list.append(video_tensor)
        
        return video_list

    # 使用示例
    video_tensors = create_scaled_videos("/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/jpg/", 
                                    total_frames=30, 
                                    H1=544, 
                                    W1=960)

    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        #print("h89h9aw8ghd9a8h")
        loss, grad_norm, pred_norm = distill_one_step(
            transformer,
            args.model_type,
            teacher_transformer,
            ema_transformer,
            None,
            discriminator,
            None,
            None,
            loader,
            noise_scheduler,
            solver,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.max_grad_norm,
            args.num_euler_timesteps,
            None,
            args.not_apply_cfg_solver,
            args.distill_cfg,
            args.ema_decay,
            args.pred_decay_weight,
            args.pred_decay_type,
            args.hunyuan_teacher_disable_cfg,
            device,
            vae = vae,
            text_encoder = None,
            clip = None,
            source_idx_double = args.source_idx_double,
            source_idx_single = args.source_idx_single,
            step = step,
            wan_i2v = wan_i2v,
            denoiser = denoiser,
            video_tensors = video_tensors,
            rank = rank,
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
            "phases": 0,
        })
        progress_bar.update(1)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type",
                        type=str,
                        default="mochi",
                        help="The type of model to train.")
    # dataset & dataloader
    # parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
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
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    # parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str)
    parser.add_argument("--model_vae_path", type=str)
    parser.add_argument("--model_text_emb", type=str)
    # parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--validation_sampling_steps", type=str, default="64")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")

    parser.add_argument("--validation_steps", type=float, default=64)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
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
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
    parser.add_argument("--shift", type=float, default=1.0)
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
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
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
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument("--distill_cfg",
                        type=float,
                        default=3.0,
                        help="Distillation coefficient.")
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
    parser.add_argument("--multi_phased_distill_schedule",
                        type=str,
                        default=None)
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument("--source_idx_double", nargs='+', type=int)
    parser.add_argument("--source_idx_single", nargs='+', type=int)
    parser.add_argument("--hunyuan_teacher_disable_cfg", action="store_true")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)

