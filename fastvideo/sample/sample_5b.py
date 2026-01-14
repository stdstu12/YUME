# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import sys
import torchvision
import time
import glob
import gc
import random
import torch
import wan23
from wan23.configs import WAN_CONFIGS
from decord import VideoReader, cpu
from packaging import version as pver
from scipy.spatial.transform import Rotation
from os.path import join as opj
from einops import rearrange
from torchvision import transforms
from pathlib import Path
from collections import deque
from copy import deepcopy
import torch.nn.functional as F
import torch.distributed as dist
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import bitsandbytes as bnb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import torch.nn as nn
from hyvideo.diffusion import load_denoiser
from diffusers.video_processor import VideoProcessor
import numpy as np
from diffusers.utils import export_to_video
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
import torch.distributed as dist

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from wan23.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

# cam_c2w, [N * 4 * 4]
# stride, frame stride
def get_traj_position_change(cam_c2w, stride=1):
    positions = cam_c2w[:, :3, 3]
    
    traj_coord = []
    tarj_angle = []
    for i in range(0, len(positions) - 2 * stride):
        v1 = positions[i + stride] - positions[i]
        v2 = positions[i + 2 * stride] - positions[i + stride]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        traj_coord.append(v1)
        tarj_angle.append(angle)
    
    # traj_coord: list of coordinate changes, each element is a [dx, dy, dz]
    # tarj_angle: list of position angle changes, each element is an angle in range (0, 180)
    return traj_coord, tarj_angle

def get_traj_rotation_change(cam_c2w, stride=1):
    rotations = cam_c2w[:, :3, :3]
    
    traj_rot_angle = []
    for i in range(0, len(rotations) - stride):
        z1 = rotations[i][:, 2]
        z2 = rotations[i + stride][:, 2]

        norm1 = np.linalg.norm(z1)
        norm2 = np.linalg.norm(z2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.dot(z1, z2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        traj_rot_angle.append(angle)

    # traj_rot_angle: list of rotation angle changes, each element is an angle in range (0, 180)
    return traj_rot_angle

def calculate_speed(traj_pos_coord, fps=30, stride=1):
    """
    Calculate Actual Speed (m/s) from Position Changes

    Parameters:
    traj_pos_coord: List of position change vectors
    fps: Video frame rate (default 30fps)
    stride: Frame step length

    Returns:
    speeds: List of speed values corresponding to each displacement (m/s)
    """
    speeds = []
    time_interval = stride / fps  
    
    for displacement in traj_pos_coord:
        distance = np.linalg.norm(displacement)
        
        speed = distance / time_interval
        speeds.append(speed)
    
    return speeds

def normalize_c2w_matrices(T_list):
    # Step 1: Align to the first frame
    T0_inv = np.linalg.inv(T_list[0])
    T_aligned = [T0_inv @ T for T in T_list]

    # Step 2: OpenGL -> Open3d
    T_convert = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Invert Y-axis
            [0, 0, -1, 0],  # Invert Z-axis (converts to right-handed coordinate system)
            [0, 0, 0, 1],
        ]
    )
    T_aligned = [T_convert @ T for T in T_aligned]

    return np.array(T_aligned)

def calculate_metrics_in_range(data, start_frame, end_frame, stride=1, fps=30):
    """
    Calculate average trajectory metrics within specified frame range (without modifying original function)
    
    Parameters:
    data: Camera pose data array (N, 4, 4)
    start_frame: Start frame index
    end_frame: End frame index (exclusive)
    stride: Frame step size (default=1)
    fps: Frame rate (default=30fps)
    
    Returns:
    (average speed, average direction change angle, average rotation angle)
    """
    # Use original functions to compute full trajectory data
    traj_pos_coord_full, tarj_pos_angle_full = get_traj_position_change(data, stride)
    traj_rot_angle_full = get_traj_rotation_change(data, stride)
    
    # Calculate starting frame index for each displacement vector
    pos_coord_frames = [i for i in range(len(traj_pos_coord_full))]
    pos_angle_frames = [i for i in range(len(traj_pos_coord_full))]  # Note: displacement angles correspond to position changes
    rot_angle_frames = [i for i in range(len(traj_rot_angle_full))]
    
    # Filter metrics within specified range
    traj_pos_coord_filtered = [
        vec for i, vec in enumerate(traj_pos_coord_full)
        if start_frame <= i < end_frame - 2 * stride   # Note: position change requires 3 consecutive points
    ]
    
    tarj_pos_angle_filtered = [
        angle for i, angle in enumerate(tarj_pos_angle_full)
        if start_frame <= i < end_frame - 2 * stride
    ]
    
    traj_rot_angle_filtered = [
        angle for i, angle in enumerate(traj_rot_angle_full)
        if start_frame <= i < end_frame - stride
    ]
    
    # Calculate averages
    if traj_pos_coord_filtered:
        # Calculate average speed
        time_interval = stride / fps
        distances = [np.linalg.norm(vec) for vec in traj_pos_coord_filtered]
        avg_speed = np.mean(distances) / time_interval if distances else 0
        
        # Calculate average direction change angle
        avg_traj_angle = np.mean(tarj_pos_angle_filtered) if tarj_pos_angle_filtered else 0
        
        # Calculate average rotation angle
        avg_rot_angle = np.mean(traj_rot_angle_filtered) if traj_rot_angle_filtered else 0
        
        return avg_speed, avg_traj_angle, avg_rot_angle
    else:
        # If insufficient data points in range
        return 0, 0, 0

def parse_txt_file(txt_path):
    """Parse TXT file to extract Keys and Mouse information"""
    keys = None
    mouse = None
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('Keys:'):
                    keys = line.split(':', 1)[1].strip()
                elif line.startswith('Mouse:'):
                    mouse = line.split(':', 1)[1].strip()
                if keys is not None and mouse is not None:
                    break
    except Exception as e:
        print(f"Error parsing {txt_path}: {str(e)}")
    return keys, mouse

def parse_txt_frame(txt_path):
    """Extract Keys and Mouse data from TXT file"""
    Start_Frame = None
    End_Frame = None
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('Start Frame:'):
                    Start_Frame = line.split(':', 1)[1].strip()
                elif line.startswith('End Frame:'):
                    End_Frame = line.split(':', 1)[1].strip()
                if Start_Frame is not None and End_Frame is not None:
                    break
    except Exception as e:
        print(f"Error parsing {txt_path}: {str(e)}")
    return Start_Frame, End_Frame

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

def project_null_space(x, A_operator):
    """Project onto the null space of A: I - A†A"""
    # Reshape video tensor to [B*C*F, H, W]
    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-2], original_shape[-1])
    
    # Compute A†(A(x))
    Ax = A_operator.A(x_flat)
    A_pinv_Ax = A_operator.A_inv(Ax)
    
    # Restore shape and compute I_A_A_INV(x) = x - A†A(x)
    I_A_A_inv = x_flat - A_pinv_Ax
    return I_A_A_inv.view(original_shape)

class LinearOperator2D:
    def __init__(self, kernel_H, kernel_W, H, W, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.H, self.W = H, W
        
        # ------------------ Height-direction blur operator A_H (H x H) ------------------
        A_H = torch.zeros(H, H, device=self.device)
        kernel_size_H = kernel_H.shape[0]
        for i in range(H):
            for j in range(i - kernel_size_H//2, i + kernel_size_H//2):
                if 0 <= j < H:
                    A_H[i, j] = kernel_H[j - i + kernel_size_H//2]
        U_H, S_H, Vt_H = torch.linalg.svd(A_H, full_matrices=False)
        self.U_H, self.S_H, self.Vt_H = U_H.to(self.device), S_H.to(self.device), Vt_H.to(self.device)
        self.S_pinv_H = torch.where(S_H > 1e-6, 1/S_H, torch.zeros_like(S_H)).to(self.device)
        
        # ------------------ Width-direction blur operator A_W (W x W) ------------------
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
        """Blur operation: First height direction (H), then width direction (W)"""
        # x shape: [..., H, W]
        # Height-direction processing ------------------------------------------------
        # Move H dimension to last for batch processing [..., W, H]
        x_permuted = x.movedim(-2, -1)  # [..., W, H]
        # Apply Vt_H @ x_permuted [..., W, H]
        x_h = torch.matmul(x_permuted, self.Vt_H.T)  # Note dimension alignment
        # Apply S_H [..., W, H]
        x_h = torch.matmul(x_h, torch.diag(self.S_H))
        # Apply U_H [..., W, H]
        x_h = torch.matmul(x_h, self.U_H.T)
        # Restore original dimensions [..., H, W]
        x_h = x_h.movedim(-1, -2)  # [..., H, W]
        
        # Width-direction processing ------------------------------------------------
        # Apply Vt_W @ x_h [..., H, W]
        x_hw = torch.matmul(x_h, self.Vt_W.T)
        # Apply S_W [..., H, W]
        x_hw = torch.matmul(x_hw, torch.diag(self.S_W))
        # Apply U_W [..., H, W]
        x_hw = torch.matmul(x_hw, self.U_W.T)
        return x_hw

    def A_inv(self, y):
        """Pseudo-inverse operation: First width direction (W†), then height direction (H†)"""
        # y shape: [..., H, W]
        # Width-direction pseudo-inverse ------------------------------------------------
        # Apply U_W @ y [..., H, W]
        y_w = torch.matmul(y, self.U_W)
        # Apply S_pinv_W [..., H, W]
        y_w = torch.matmul(y_w, torch.diag(self.S_pinv_W))
        # Apply Vt_W [..., H, W]
        y_w = torch.matmul(y_w, self.Vt_W)
        
        # Height-direction pseudo-inverse ------------------------------------------------
        # Move H dimension to last for batch processing [..., W, H]
        y_permuted = y_w.movedim(-2, -1)  # [..., W, H]
        # Apply U_H @ y_permuted [..., W, H]
        y_hw = torch.matmul(y_permuted, self.U_H)
        # Apply S_pinv_H [..., W, H]
        y_hw = torch.matmul(y_hw, torch.diag(self.S_pinv_H))
        # Apply Vt_H [..., W, H]
        y_hw = torch.matmul(y_hw, self.Vt_H)
        # Restore original dimensions [..., H, W]
        y_hw = y_hw.movedim(-1, -2)  # [..., H, W]
        return y_hw

def scale(vae,latents):
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
        
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

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

# Convert to PIL.Image
def tensor_to_pil(tensor):
    # 1. Convert to NumPy array
    array = ((tensor + 1) / 2.0).detach().cpu().numpy()  # Move to CPU if tensor is on GPU
    
    # 2. Reshape to (H, W, C)
    array = np.transpose(array, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    
    # 3. Convert to [0, 255] range and uint8 type
    array = (array * 255).astype(np.uint8)
    
    # 4. Create PIL Image
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


def extract_first_frame_from_latents(latents):
    """
    直接从(C,F,H,W)的latents张量提取首帧并转为PIL图像
    :param latents: 输入潜变量张量，形状为(C,F,H,W)
    :return: 首帧PIL图像（自动转换为[0,255]范围）
    """
    # 确保输入是4D张量
    assert len(latents.shape) == 4, "Input must be (C,F,H,W) tensor"
    
    # 提取首帧并反归一化 [-1,1]->[0,255]
    first_frame = latents[:, 0, :, :]  # 取第0帧 -> (C,H,W)
    first_frame = (first_frame + 1) * 127.5  # 数值映射到[0,255]
    
    # 转换为PIL.Image
    first_frame = first_frame.clamp(0, 255).byte()  # 确保数值范围有效
    first_frame = first_frame.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    return Image.fromarray(first_frame)

def mp4_data(root_dir=None):
    vid_meta = []
    dataset_ddp = []
    vid_meta = []
    img_size = (704,1280)
    height = 704
    width = 1280
    for subdir in glob.glob(os.path.join(root_dir, '*/')):
        for mp4_path in glob.glob(os.path.join(subdir, '*.mp4')):
            base_name = os.path.splitext(os.path.basename(mp4_path))[0]
            txt_path = os.path.join(subdir, f"{base_name}.txt")

            if os.path.exists(txt_path):
                keys, mouse = parse_txt_file(txt_path)
                start_frame, end_frame = parse_txt_frame(txt_path)
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                if keys is not None and mouse is not None:
                    video_id = base_name.split('_frames_')[0]

                    vid_meta.append((mp4_path, video_id, [keys], [mouse]))
                else:
                    print(f"Warning: Missing Keys/Mouse in {txt_path}")
            else:
                print(f"Warning: Missing TXT file for {mp4_path}")


    simple_transform = transforms.ToTensor()
    resize_norm_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    cnt = 0
    
    for (video_path, videoid, keys, mouse) in vid_meta:
        try:
            caption = "This video depicts a city walk scene with a first-person view (FPV)."
            vocab_k = {
                "W": "Person moves forward (W).",  
                "A": "Person moves left (A).",  
                "S": "Person moves backward (S).",  
                "D": "Person moves right (D).",  
                "W+A": "Person moves forward and left (W+A).",  
                "W+D": "Person moves forward and right (W+D).",  
                "S+D": "Person moves backward and right (S+D).",  
                "S+A": "Person moves backward and left (S+A).",  
                "None": "Person stands still (·).",
                "·": "Person stands still (·)."  
            }
            caption = caption + vocab_k[keys[0]]

            vocab_c = {
                "→": "Camera turns right (→).",  
                "←": "Camera turns left (←).",  
                "↑": "Camera tilts up (↑).",  
                "↓": "Camera tilts down (↓).",  
                "↑→": "Camera tilts up and turns right (↑→).",  
                "↑←": "Camera tilts up and turns left (↑←).",  
                "↓→": "Camera tilts down and turns right (↓→).",  
                "↓←": "Camera tilts down and turns left (↓←).",  
                "·": "Camera remains still (·)."  
            }
            caption = caption + vocab_c[mouse[0]]

            video_reader = VideoReader(video_path)
            video_length = len(video_reader)
            total_frames_target = 33 
            

            if video_length > 0:
                target_times = np.arange(total_frames_target) / 30 
                original_indices = np.round(target_times * 30).astype(int)   
                start_idx = 0
                batch_index = [idx+start_idx for idx in original_indices]
            else:
                batch_index = []
            if len(batch_index) < total_frames_target:
                batch_index = batch_index[:total_frames_target]

            vid_pil_image_list = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index]
            ref_img_pil = Image.fromarray(video_reader[batch_index[0]].asnumpy())
            pixel_values = torch.stack([simple_transform(img) for img in vid_pil_image_list], dim=0)
            pixel_values_ref_img = simple_transform(ref_img_pil)

            pixel_values = (torch.nn.functional.interpolate(pixel_values.sub_(0.5).div_(0.5), \
                size=(height, width), mode='bicubic'  )   ).clamp_(-1, 1) 

            pixel_values_ref_img = pixel_values[0,:,:,:]

            dataset_ddp.append((pixel_values,pixel_values_ref_img,[caption],videoid))
        except Exception as e:
            print("__getitem__[Error]", str(e))


    main_print(
        f" length: {len(dataset_ddp)}")

    return dataset_ddp, len(dataset_ddp)

def create_scaled_videos(folder_path, total_frames=33, H1=256, W1=256):
    """Create video tensors scaled to specified dimensions, supporting multiple image formats"""
    # Get all supported image file paths
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_formats)
    ], key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])) or 0))
    
    video_list = []
    cnt = 0 
     
    #random.shuffle(image_files)
    for img_file in image_files:
        cnt+=1
        print(cnt)
        if cnt>=150:
            break
        img_path = os.path.join(folder_path, img_file)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            img = Image.open(img_path)
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])
                img = background
            
            img = np.array(img)
            
            if len(img.shape) == 2:  
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:  
                img = img[:, :, :3]
            
        except Exception as e:
            print(f"Error Load {img_path}: {e}")
            continue
        
        try:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            C, H, W = img_tensor.shape
            video_tensor = torch.zeros(C, total_frames, H1, W1)
            
            resized_frame = F.interpolate(
                img_tensor.unsqueeze(0), 
                size=(H1, W1),
                mode='bilinear',
                align_corners=False
            )[0]
            
            video_tensor[:, 0] = (resized_frame - 0.5)*2
            video_list.append((video_tensor.permute(1,0,2,3),base_name,img_path))
            
        except Exception as e:
            print(f"Error {img_path}: {e}")
    #random.shuffle(video_list)
    return video_list, len(video_list)

def sample_one(
    transformer,
    not_apply_cfg_solver,
    device,
    rand_num_img=0.6,
    num_euler_timesteps=50,
    dataset_ddp=None,
    video_output_dir=None,
    vae=None,
    step=None,
    wan_i2v=None,
    denoiser=None,
    rank=None,
    world_size=None,
    image_sample=None,
    caption_path=None,
    camption_model=None,
    tokenizer=None,
    t2v=False,
    prompt1=None,
):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
    
    if dataset_ddp!=None:
        index = min((step-1)*world_size+rank, len(dataset_ddp)-1)
    else:
        index = (step-1)*world_size+rank

    main_print(
        f"--> GPU Index: {index}"
    )
    if image_sample and not t2v:
        # Read image files and caption.txt (for text conditioning).
        pixel_values_vid, videoid, img_path = dataset_ddp[index]
        
        # set the max number of tiles in `max_num`
        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).to(device)
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        # single-image single-round conversation (单图单轮对话)
        if prompt1 == None:
            question = '<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        else: 
            question = '<image>\nWe want to generate a video using this prompt: \"'+prompt1+'\". Please modify and refine this prompt for the video of this image (<image>). Note that  \"'+prompt1+'\" must appear and revolve around the extension. Don\'t split it into points; just write a paragraph directly'
        #'<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        
        response = camption_model.chat(tokenizer, pixel_values, question, generation_config)
        pixel_values_ref_img = pixel_values_vid[0,:,:,:]
        caption = []
        #rot = "First-person perspective. The camera's movement direction remains stationary (·). The camera pans to the right (→). Actual distance moved:4 at 100 meters per second. Angular change rate (turn speed):0. View rotation speed:0."

        #for i in range(3):
        #    caption.append( rot + "A sophisticated robot is walking with purposeful strides, its sensors actively scanning the environment for any signs of activity.") #response )
        
        with open(caption_path, 'r', encoding='utf-8') as file:
            for line in file:
                #if prompt1!=None:
                #    caption.append(line.rstrip('\n') + prompt1 + response )
                #else:
                caption.append(line.rstrip('\n') + response )
        main_print(
            f"  Caption: {str(caption)}")
        sample_num = len(caption)
        caption_ori = caption[0][:caption[0].find("Actual distance moved:")].strip()
    elif t2v:
        videoid = prompt1[:60]#str(step)+"T2I"
        caption = []

        with open(caption_path, 'r', encoding='utf-8') as file:
            for line in file:
                #question = '<image>\nWe want to generate a video using this prompt: \"'+line.rstrip('\n')+'\". Please modify and refine this prompt for the video. Don\'t split it into points; just write a paragraph directly'
                #response = camption_model.chat(tokenizer, pixel_values, question, generation_config)
                response = prompt1
                caption.append(line.rstrip('\n') + response )
        main_print(
            f"  Caption: {str(caption)}")
        sample_num = len(caption)
        caption_ori = caption[0][:caption[0].find("Actual distance moved:")].strip()
    else:
        # Read test videos
        pixel_values_vid, pixel_values_ref_img,caption,videoid = dataset_ddp[index]
        caption = list(caption)
        caption_ori = caption[0]
        
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        # single-image single-round conversation (单图单轮对话)
        #question = '<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        if prompt1 == None:
            question = '<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        else: 
            question = '<image>\nWe want to generate a video using this prompt: \"'+prompt1+'\". Please modify and refine this prompt for the video of this image (<image>). Note that  \"'+prompt1+'\" must appear and revolve around the extension. Don\'t split it into points; just write a paragraph directly'
        #'<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        
        #'<image>\nWe want to generate a video using this prompt: \"'+"The sun comes out."+'\". Please modify and refine this prompt for the video of this image (<image>). Note that "The sun comes out." must appear and revolve around the extension. Don\'t split it into points; just write a paragraph directly'
        #'<image>\nWe want to generate a video using this image. Please generate a prompt word for the video of this image. Don\'t split it into points; just write a paragraph directly' 
        pixel_values_ref_img = torch.nn.functional.interpolate(pixel_values_ref_img.unsqueeze(0).to(torch.bfloat16).to(device), size=(448, 448), mode='bilinear', align_corners=False)
        response = camption_model.chat(tokenizer, pixel_values_ref_img, question, generation_config)


        caption[0] = caption[0] \
        + "Actual distance moved:4 at 100 meters per second. Angular change rate (turn speed):4. View rotation speed:4" + response
        

        caption.append(caption[0])
        for i in range(3):
            caption.append(caption[0])
        main_print(
            f"  Caption: {str(caption)}")
        sample_num = len(caption)
        

    # Generate diverse output videos from identical input conditions
    max_area = 704 * 1280
    # pixel_values_vid = torch.nn.functional.interpolate(pixel_values_vid, size=(544, 960), mode='bilinear', align_corners=False)
    
    repeat_nums = 1
    for repeat_num in range(repeat_nums):
        latent_frame_zero = 8
        frame_zero = 32

        if not t2v:
            with torch.no_grad():
                pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
                pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
                latents = pixel_values_vid

            model_input = latents

            # When the input is an image, extend it to 16 frames
            model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:33]],dim=1)
            model_input_de = model_input.squeeze()

            frame = model_input.shape[1]

            model_input = torch.cat([wan_i2v.vae.encode([model_input.to(device)[:,:-32].to(device)])[0], \
                                     wan_i2v.vae.encode([model_input.to(device)[:,-32:].to(device)])[0]],dim=1) 

            latents = model_input

            img =  model_input[:,:-latent_frame_zero]


            with torch.no_grad():
                arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
                            caption[0],
                            frame_num=frame,
                            max_area=max_area,
                            latent_frame_zero=latent_frame_zero,
                            img=img)
        else:
            frame = 32
            with torch.no_grad():
                arg_c, arg_null, noise = wan_i2v.generate(
                            caption[0],
                            frame_num=32,
                            max_area=max_area,
                            latent_frame_zero=latent_frame_zero,)
        

       
        if step % 1 == 0:
            video_all = []
            for step_sample in range(sample_num):
                
                # UniPC
                # sample_scheduler = FlowUniPCMultistepScheduler(
                #                 num_train_timesteps=1000,
                #                 shift=1,
                #                 use_dynamic_shifting=False)
                # sample_scheduler.set_timesteps(
                #                 num_euler_timesteps, device=device, shift=5.0)
                # timesteps = sample_scheduler.timesteps
                # seed = random.randint(0, sys.maxsize)
                # seed_g = torch.Generator(device=device)
                # seed_g.manual_seed(seed)

                #c1,f1,h1,w1 = model_input.shape
                # if t2v and step_sample == 0:
                #     num_euler_timesteps_next = num_euler_timesteps
                #     num_euler_timesteps = 14
                # else:
                #     num_euler_timesteps = num_euler_timesteps_next

                sample_step = num_euler_timesteps
                sampling_sigmas = get_sampling_sigmas(sample_step, 7.0)

                if step_sample > 0:
                    noise = torch.randn_like(model_input_1)
                    latent = noise.clone()
                else:
                    latent = noise

                import time
                start_time = time.time()
                
                if not t2v or step_sample > 0:
                    latent = torch.cat([img[0][:, :-latent_frame_zero, :, :], latent[:, -latent_frame_zero:, :, :]], dim=1)
                #(1. - mask2[0]) * img[0]  + mask2[0] * latent
                print(latent.shape, "nbxkasbcna090-")
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):

                        for i in range(sample_step):
                            latent_model_input = [latent.squeeze(0)]

                            if not t2v or step_sample>0:
                               
                                timestep = [sampling_sigmas[i]*1000]
                                timestep = torch.tensor(timestep).to(device)
                                temp_ts = (mask2[0][0][:-latent_frame_zero, ::2, ::2] ).flatten()
                                temp_ts = torch.cat([
                                    temp_ts,
                                    temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                                ])
                                timestep = temp_ts.unsqueeze(0)

                                # # UniPC
                                # timestep = [t]
                                # timestep = torch.stack(timestep)
                                # temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                                # temp_ts = torch.cat([
                                #     temp_ts,
                                #     temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                                # ])
                                # timestep = temp_ts.unsqueeze(0)

                                print(latent_model_input[0].shape,"0-2=ffje0r=----------a")
                                noise_pred_cond = transformer(latent_model_input, t=timestep, **arg_c)[0]

                                if i+1 == sample_step:
                                    temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                                else:
                                    temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]

                                # # UniPC                                
                                # print("w0a9w0j90weah", noise_pred_cond.unsqueeze(0).shape,t.shape,latent.unsqueeze(0).shape,"q11eq3rw31")
                                # #w0a9w0j90weah torch.Size([1, 48, 8, 44, 80]) torch.Size([]) torch.Size([1, 48, 16, 44, 80]) q11eq3rw31
                                # temp_x0 = sample_scheduler.step(
                                #             noise_pred_cond[:,-latent_frame_zero:,:,:].unsqueeze(0),
                                #             t,
                                #             latent[:,-latent_frame_zero:,:,:].unsqueeze(0),
                                #             return_dict=False,
                                #             generator=seed_g
                                #         )[0]
                                # temp_x0 = temp_x0.squeeze()[:,-latent_frame_zero:,:,:]

                            else:
                                timestep = [sampling_sigmas[i]*1000]
                                timestep = torch.tensor(timestep).to(device)

                                # # UniPC
                                # timestep = [t]
                                # timestep = torch.stack(timestep)
                                # temp_ts = timestep.flatten()
                                # timestep = temp_ts#.unsqueeze(0)
                                print(latent_model_input[0].shape,"0-2=ffje0r=----------a")
                                noise_pred_cond = transformer(latent_model_input, t=timestep, flag=False, **arg_c)[0]

                                # # UniPC
                                # temp_x0 = sample_scheduler.step(
                                #             noise_pred_cond[:,-latent_frame_zero:,:,:].unsqueeze(0),
                                #             t,
                                #             latent[:,-latent_frame_zero:,:,:].unsqueeze(0),
                                #             return_dict=False,
                                #             generator=seed_g
                                #         )[0].squeeze()
                                # latent = temp_x0.squeeze()[:,-latent_frame_zero:,:,:]

                                if i+1 == sample_step:
                                    latent = latent + (0-sampling_sigmas[i])*noise_pred_cond
                                else:
                                    latent = latent + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond

                            if step_sample > 0:
                                latent = torch.cat([model_input_1[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                            elif not t2v:
                                latent = torch.cat([model_input[:,:-latent_frame_zero,:,:], temp_x0], dim=1)

                end_time = time.time()
                elapsed = end_time - start_time
                main_print(
                    f"--> Function running time: {elapsed:.4f} s"
                )

                global_step = 1
                if step_sample > 0:
                    model_input = torch.cat([model_input, latent[:,-latent_frame_zero:,:,:]],dim=1)
                else:
                    if not t2v:
                        model_input = torch.cat([model_input[:,:-latent_frame_zero,:,:], latent[:,-latent_frame_zero:,:,:]],dim=1)
                    else:
                        model_input = latent

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video_cat = scale(vae, model_input[:,-latent_frame_zero:,:,:]) 
                    video = video_cat[:,-frame_zero:]
                    video_all.append(video)

                    if step_sample > 0:
                        #if video.shape[1] < frame_zero:
                        #    video = torch.cat([video[:,0].unsqueeze(1).repeat(1,frame_zero-video.shape[1],1,1),video],dim=1)                            
                        model_input_de = torch.cat([model_input_de, video[:,-frame_zero:,:,:]],dim=1)
                    else:
                        #if video.shape[1] < frame_zero:
                        #    video = torch.cat([video[:,0].unsqueeze(1).repeat(1,frame_zero-video.shape[1],1,1),video],dim=1) 
                        if not t2v: 
                            model_input_de = torch.cat([model_input_de[:,:-frame_zero,:,:], video[:,-frame_zero:,:,:]],dim=1)
                        else:
                            model_input_de = video[:,-frame_zero:,:,:]

                    video = save_video(torch.cat(video_all,dim=1))

                # Ensure videoid is a string
                if isinstance(videoid, list):
                    videoid_str = "_".join(map(str, videoid))
                else:
                    videoid_str = str(videoid)

                filename = os.path.join(
                                        video_output_dir,
                                        videoid_str+"_"+str(caption_ori)+"_"+str(repeat_num)+"_"+str(rank)+"_"+str(step_sample)+".mp4",
                                    )
                export_to_video(video[0] , filename, fps=16)


                if step_sample + 1 < sample_num:
                    with torch.no_grad():
                        # Jump to ./wan/image2video.py
                        img =  model_input#[:,:-latent_frame_zero]
                        with torch.no_grad():
                            arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
                                        caption[step_sample],
                                        frame_num=(model_input.shape[1]-1)*4+1+32,
                                        max_area=max_area,
                                        latent_frame_zero=latent_frame_zero,
                                        img=img
                            )
                            model_input_1 = model_input #wan_i2v.vae.encode([model_input_de.to(device)])[0]
                            
                        model_input_1=torch.cat([model_input_1,torch.zeros(48, latent_frame_zero, model_input_1.shape[2], model_input_1.shape[3]).to(device)],dim=1)

    return

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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set independent cache directories for each rank
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{rank}"
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    #if args.seed is not None:
    #    # TODO: t within the same seq parallel group should be the same. Noise should be different.
    #    set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Create model:
    cfg = WAN_CONFIGS["ti2v-5B"]
    ckpt_dir = "./Yume-5B-720P"

    # Referenced from https://github.com/Wan-Video/Wan2.2
    wan_i2v = wan23.Yume(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
    )  
    transformer = wan_i2v.model  
    transformer = transformer.eval().requires_grad_(False)

    main_print(
        f"  Total Sample parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
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

    if args.resume_from_checkpoint:
        (
            transformer,
            init_steps,
        ) = resume_checkpoint(
            transformer,
            args.resume_from_checkpoint,
        )
        
      
    from safetensors import safe_open
    from safetensors.torch import save_file

    # # 配置路径
    # MODEL1 = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/outputs/checkpoint-23100/diffusion_pytorch_model.safetensors"
    # MODEL2 = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/test_long_2/checkpoint-270/diffusion_pytorch_model.safetensors"
    # MODEL3 = "/mnt/petrelfs/maoxiaofeng/merge/diffusion_pytorch_model.safetensors"
    # OUTPUT = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/merged_model/diffusion_pytorch_model.safetensors"

    # # 1. 合并模型权重
    # def merge_models():
    #     """合并两个模型的权重并保存"""
    #     # 确保输出目录存在
    #     os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    #    # 加载模型权重 - 正确使用 safe_open
    #     weights1 = {}
    #     with safe_open(MODEL1, framework="pt") as f:
    #         for key in f.keys():
    #             weights1[key] = f.get_tensor(key)

    #     weights2 = {}
    #     with safe_open(MODEL2, framework="pt") as f:
    #         for key in f.keys():
    #             weights2[key] = f.get_tensor(key)

    #     weights3 = {}
    #     with safe_open(MODEL3, framework="pt") as f:
    #         for key in f.keys():
    #             weights3[key] = f.get_tensor(key)


    #     # 合并权重
    #     merged = {}
    #     for key in set(weights1.keys()) & set(weights2.keys()):
    #         #print(key)
    #         if True: #"cross" in key:
    #             #print("436888")
    #             #merged[key] = (1.0*weights1[key] + 1.0*weights2[key] ) / 2
    #             merged[key] = (1.0*weights1[key] + 2.0*weights2[key] + 0.*weights3[key]) / 3

    #         else:
    #             merged[key] = (weights1[key] + weights2[key]) / 2

    #     # 保存合并后的模型
    #     save_file(merged, OUTPUT)
    #     print(f"✅ 模型已合并保存至: {OUTPUT}")
    #     return merged
    
    # merged_weights = merge_models()
       # 加载模型权重 - 正确使用 safe_open
       
    # merged_weights = {}
    # with safe_open("/mnt/petrelfs/maoxiaofeng/Yume_v2_release/merged_model/diffusion_pytorch_model.safetensors", framework="pt") as f:
    #     for key in f.keys():
    #         merged_weights[key] = f.get_tensor(key)

    # transformer.load_state_dict(merged_weights, strict=False)
    
    transformer = transformer.to(torch.bfloat16)
    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
        use_orig_params=True,
    )



    # Set model as eval.
    transformer.eval().requires_grad_(False)

    # print dtype
    main_print(
        f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    
    main_print(
        f"  T5 CPU: {args.t5_cpu}")
    
    #init t5, clip and vae
    vae = wan_i2v.vae

    dist.barrier()
    
    wan_i2v.device = device
    denoiser = load_denoiser()
    
    print("jpg_dir", args.jpg_dir)
    image_sample = False
    dataset_ddp = None
    dataset_length = None
    if args.jpg_dir != None and not args.T2V:
        dataset_ddp, dataset_length = create_scaled_videos(args.jpg_dir, 
                                        total_frames=33, 
                                        H1=704, 
                                        W1=1280)
        image_sample = True
    elif not args.T2V:
        dataset_ddp, dataset_length = mp4_data(args.video_root_dir)
        image_sample = False

    print(dataset_ddp,"dataset_ddpdataset_ddpdataset_ddp")

    step_times = deque(maxlen=100)
    #image_sample = True
    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    path = '/mnt/petrelfs/maoxiaofeng/Yume_v2_release/InternVL3-2B-Instruct'
    camption_model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    if args.prompt!=None:
        prompt1 = args.prompt
    else:
        prompt1 = None #""

    if args.prompt:
        date_len = 1 + 1
    else:
        date_len = int(dataset_length)//world_size + 1
   
    for step in range(1, date_len):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        sample_one(
            transformer,
            args.not_apply_cfg_solver,
            device,
            dataset_ddp = dataset_ddp,
            video_output_dir = args.video_output_dir,
            rand_num_img = args.rand_num_img,
            num_euler_timesteps=args.num_euler_timesteps,
            vae = vae,
            step = step,
            wan_i2v = wan_i2v,
            denoiser = denoiser,
            rank = rank,
            world_size = world_size,
            image_sample = image_sample,
            caption_path = args.caption_path,
            camption_model = camption_model,
            tokenizer = tokenizer,
            t2v = args.T2V, #True,
            prompt1 = prompt1,
        )


        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)


        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()


    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    parser.add_argument("--t5_cpu", action="store_true") 

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--jpg_dir",
        type=str,
        default=None,
        help="Images used for model input."
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *video_output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_sample_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--T2V",
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
        "--rand_num_img",
        type=float,
        default=0.6,
        help="Determine whether it is in i2v mode or v2v mode.",
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
    parser.add_argument("--source_idx_double", nargs='+', type=int)
    parser.add_argument("--source_idx_single", nargs='+', type=int)
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
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
        "--video_root_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)

