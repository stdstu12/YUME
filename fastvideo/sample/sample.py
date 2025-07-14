# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import torchvision
import time
import glob
import gc
import torch
import wan
from wan.configs import WAN_CONFIGS
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
    time_interval = stride / fps  # 每次位移的时间间隔(秒)
    
    for displacement in traj_pos_coord:
        # 计算位移向量模长(总移动距离)
        distance = np.linalg.norm(displacement)
        
        # 速度 = 距离 / 时间
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

def mp4_data(root_dir=None):
    vid_meta = []
    dataset_ddp = []
    vid_meta = []
    img_size = (544,960)
    height = 544
    width = 960
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
    
    for img_file in image_files:
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
            video_list.append((video_tensor.permute(1,0,2,3),base_name))
            
        except Exception as e:
            print(f"Error {img_path}: {e}")
    
    return video_list

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
):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16

    index = min((step-1)*world_size+rank, len(dataset_ddp)-1)
    main_print(
        f"--> GPU Index: {index}"
    )
    if image_sample:
        # Read image files and caption.txt (for text conditioning).
        pixel_values_vid, videoid = dataset_ddp[(rank+(step-1)*8)%len(dataset_ddp)]
        pixel_values_ref_img = pixel_values_vid[0,:,:,:]
        caption = []
        with open(caption_path, 'r', encoding='utf-8') as file:
            for line in file:
                caption.append(line.rstrip('\n'))
        main_print(
            f"  Caption: {str(caption)}")
        sample_num = len(caption)
        caption_ori = caption[0][:caption[0].find("Actual distance moved:")].strip()

    else:
        # Read test videos
        pixel_values_vid, pixel_values_ref_img,caption,videoid = dataset_ddp[index]
        caption = list(caption)
        caption_ori = caption[0]
        caption[0] = caption[0] + "Actual distance moved:4.3697374288015297 at 100 meters per second.Angular change rate (turn speed):4.520279996588001.View rotation speed:4.14601429683874179."
        caption.append(caption[0])
        caption.append(caption[0])
        main_print(
            f"  Caption: {str(caption)}")
        sample_num = 3

    # Generate diverse output videos from identical input conditions
    repeat_nums = 1
    for repeat_num in range(repeat_nums):
        # i2v or v2v
        rand_num_img = 0.6        

        with torch.no_grad():
            pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
            pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
            latents = pixel_values_vid

        model_input = latents
        if rand_num_img < 0.4:
            model_input = model_input[:,:33]
        # When the input is an image, extend it to 16 frames
        model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:33]],dim=1)

        model_input_de = model_input.squeeze()

        img = tensor_to_pil(pixel_values_ref_img)

        with torch.no_grad():
            # Jump to ./wan/image2video.py
            latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null = wan_i2v.generate(
                model_input,
                device,
                caption[0],
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

        frame_zero = 32
        latent_frame_zero = (frame_zero-1)//4 + 1

        if step % 1 == 0:
            
            video_all = []
            for step_sample in range(sample_num):
                
                c1,f1,h1,w1 = model_input.shape
                sample_step = num_euler_timesteps
                sampling_sigmas = get_sampling_sigmas(sample_step, 3.0)

                if step_sample > 0:
                    c1,f1,h1,w1 = model_input_1.shape

                # kernel_H = torch.tensor([0.1, 0.8, 0.1], device='cpu')  
                # kernel_W = torch.tensor([0.2, 0.6, 0.2], device='cpu')  
                # A_op = LinearOperator2D(kernel_H, kernel_W, h1, w1, device=device)
                # A_op_1 = LinearOperator2D(kernel_H, kernel_W, h1*8, w1*8, device=device)


                if step_sample > 0:
                    noise = torch.randn_like(model_input_1)
                    latent = noise.clone()
                else:
                    latent = noise


                import time
                start_time = time.time()

                # with torch.no_grad():
                #     with torch.autocast("cuda", dtype=torch.bfloat16):
                #         # 时间旅行参数配置
                #         time_travel_step = 2 #5  # 时间旅行步长l
                #         time_travel_interval = 2 #2 #5  # 时间旅行间隔s (每10步执行一次)
                #         time_travel_repeat = 1  # 重复次数r

                #         for i in range(sample_step):
                #             print("iii-----------",i)
                #             # === 1. 保存当前状态用于时间旅行 ===
                #             #if i % time_travel_interval == 0 and i < 49:
                #             #    latent_original = latent.clone()
                            
                #             # === 2. 执行原始采样操作 ===
                #             latent_model_input = [latent]
                #             timestep = [sampling_sigmas[i] * 1000]
                #             timestep = torch.tensor(timestep).to(device)
                            
                #             # 条件预测
                #             noise_pred_cond = transformer(
                #                 latent_model_input, 
                #                 t=timestep, 
                #                 noise=noise, 
                #                 rand_num_img=rand_num_img, 
                #                 plucker=plucker, 
                #                 plucker_train=True, 
                #                 latent_frame_zero=latent_frame_zero, 
                #                 **arg_c
                #             )[0]
                            
                #             # # 无条件预测
                #             # noise_pred_uncond = transformer(
                #             #     latent_model_input, 
                #             #     t=timestep, 
                #             #     noise=noise, 
                #             #     rand_num_img=rand_num_img, 
                #             #     plucker=plucker, 
                #             #     plucker_train=False, 
                #             #     latent_frame_zero=latent_frame_zero, 
                #             #     **arg_null
                #             # )[0]
                            
                #             # # 组合条件与无条件预测
                #             # noise_pred_cond = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
                            
                #             # 计算x0估计
                #             if i + 1 == sample_step:
                #                 temp_x0 = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]
                #             else:
                #                 temp_x0 = latent[:, -latent_frame_zero:, :, :] + (sampling_sigmas[i+1] - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]


                #             # prev_sample_mean = temp_x0
                #             # pred_original_sample = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]
                #             # eta = 0.3
                #             # if i + 1 == 50:
                #             #     delta_t = 0 #(sampling_sigmas[i] - 0) #sigma - sigmas[index + 1]
                #             # else:
                #             #     delta_t = (sampling_sigmas[i] - sampling_sigmas[i+1]) #sigma - sigmas[index + 1]
                #             # if delta_t < 0:
                #             #     delta_t = 0
                #             # if i + 1 == 50:
                #             #     dsigma = 0 - sampling_sigmas[i]
                #             # else:
                #             #     dsigma = sampling_sigmas[i+1] - sampling_sigmas[i]
                #             # std_dev_t = eta * math.sqrt(delta_t)
                #             # score_estimate = -(latent[:, -latent_frame_zero:, :, :]-pred_original_sample*(1 - sampling_sigmas[i]))/sampling_sigmas[i]**2
                #             # log_term = -0.5 * eta**2 * score_estimate
                #             # prev_sample_mean = prev_sample_mean + log_term * dsigma
                #             # temp_x0 = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
                                            

                #             # === 3. 添加时间旅行操作 ===
                #             if time_travel_interval > 0 and i % time_travel_interval == 0:
                #                 #current_latent = latent.clone()     
                #                 latent_original = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -latent_frame_zero:, :, :]   
                #                 noise_ori_pred_cond = noise_pred_cond[:, -latent_frame_zero:, :, :] + latent_original

                #                 noise_ped_v = noise_pred_cond

                #                 if True: #for _ in range(1):
                                    
                #                     # # 3.1 正向扩散到t+l (增加噪声)
                #                     # travel_step = min(49, i + time_travel_step)

                #                     # r = 0.8  # 70%来自a，30%来自b
                #                     # # 计算归一化权重
                #                     # w_a = math.sqrt(r)
                #                     # w_b = math.sqrt(1 - r)

                #                     # # 合成新噪声
                #                     noise_travel = noise_ori_pred_cond #w_a * noise_ori_pred_cond + w_b * torch.randn_like(latent_original)


                #                     # sigma_diff = sampling_sigmas[travel_step] - sampling_sigmas[i] 
                #                     # #latent_travel = latent_original + sigma_diff * (noise_travel-latent_original)
                #                     # latent_travel = (1-sampling_sigmas[travel_step])*latent_original + sampling_sigmas[travel_step] * noise[:, -latent_frame_zero:, :, :]
                                    
                #                     # # 更新潜在状态
                #                     # index1 = travel_step #min(49, i + 1)
                #                     # if step_sample > 0:
                #                     #     latent_travel = torch.cat([
                #                     #         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                     #         (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                #                     #         latent_travel
                #                     #     ], dim=1)
                #                     # else:
                #                     #     latent_travel = torch.cat([
                #                     #         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                     #         (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                #                     #         latent_travel
                #                     #     ], dim=1)

                #                     travel_step = min(sample_step - 1, i + time_travel_step)
                #                     index1 = travel_step
                #                     if step_sample > 0:
                #                         latent_travel = torch.cat([
                #                             noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                             (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                #                             temp_x0
                #                         ], dim=1)
                #                     else:
                #                         latent_travel = torch.cat([
                #                             noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                             (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                #                             temp_x0
                #                         ], dim=1)

                #                     # 3.2 从t+l反向采样回t-1
                #                     for j in range(i+1, travel_step):
                #                         print("ji------------",j,i)
                #                         # 准备输入
                #                         latent_model_input_travel = [latent_travel]
                #                         travel_timestep = [sampling_sigmas[j] * 1000]
                #                         travel_timestep = torch.tensor(travel_timestep).to(device)
                                        
                #                         # 条件预测（时间旅行）
                #                         noise_pred_cond_travel = transformer(
                #                             latent_model_input_travel, 
                #                             t=travel_timestep, 
                #                             noise=noise_travel, 
                #                             rand_num_img=rand_num_img, 
                #                             plucker=plucker, 
                #                             plucker_train=True, 
                #                             latent_frame_zero=latent_frame_zero, 
                #                             **arg_c
                #                         )[0]
                                        
                #                         # # 无条件预测（时间旅行）
                #                         # noise_pred_uncond_travel = transformer(
                #                         #     latent_model_input_travel, 
                #                         #     t=travel_timestep, 
                #                         #     noise=noise_travel, 
                #                         #     rand_num_img=rand_num_img, 
                #                         #     plucker=plucker, 
                #                         #     plucker_train=False, 
                #                         #     latent_frame_zero=latent_frame_zero, 
                #                         #     **arg_null
                #                         # )[0]
                                        
                #                         # # 组合预测（时间旅行）
                #                         # noise_pred_cond_travel = noise_pred_uncond_travel + 5.0 * (noise_pred_cond_travel - noise_pred_uncond_travel)
                                        
                #                         #noise_pred_cond_travel = (noise_ped_v+noise_pred_cond_travel)/2.0
                #                         # 计算下一步的x0估计（时间旅行）
                #                         # if j - 1 == i:
                #                         #     temp_x0_travel = latent_travel[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                #                         # else:
                #                         temp_x0_travel = latent_travel[:, -latent_frame_zero:, :, :] + (sampling_sigmas[j+1] - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                                        
                #                         # prev_sample_mean = temp_x0_travel
                #                         # pred_original_sample = latent_travel[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[j]) * noise_pred_cond_travel[:, -latent_frame_zero:, :, :]
                #                         # eta = 0.3
                #                         # delta_t = (sampling_sigmas[j] - sampling_sigmas[j+1]) #sigma - sigmas[index + 1]
                #                         # if delta_t < 0:
                #                         #     delta_t = 0
                #                         # dsigma = sampling_sigmas[j+1] - sampling_sigmas[j]
                #                         # print(sampling_sigmas[j],sampling_sigmas[j+1],sampling_sigmas[j] - sampling_sigmas[j+1],"cj0ajc0a")
                #                         # #0.09259259259259267 1.0 -0.9074074074074073 cj0ajc0a
                #                         # std_dev_t = eta * math.sqrt(delta_t)
                #                         # score_estimate = -(latent_travel[:, -latent_frame_zero:, :, :]-pred_original_sample*(1 - sampling_sigmas[j]))/sampling_sigmas[j]**2
                #                         # log_term = -0.5 * eta**2 * score_estimate
                #                         # prev_sample_mean = prev_sample_mean + log_term * dsigma
                #                         # temp_x0_travel = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
                                            


                #                         # 更新潜在状态（时间旅行）
                #                         index_travel = min(sample_step-1, j + 1)
                #                         print(noise_travel.shape,model_input[:, :-latent_frame_zero, :, :].shape,"model_input_1[:, :-latent_frame_zero, :, :]")
                #                         if step_sample > 0:
                #                             latent_travel = torch.cat([
                #                                 noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index_travel] + 
                #                                 (1 - sampling_sigmas[index_travel]) * model_input_1[:, :-latent_frame_zero, :, :], 
                #                                 temp_x0_travel
                #                             ], dim=1)
                #                         else:
                #                             latent_travel = torch.cat([
                #                                 noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index_travel] + 
                #                                 (1 - sampling_sigmas[index_travel]) * model_input[:, :-latent_frame_zero, :, :], 
                #                                 temp_x0_travel
                #                             ], dim=1)
                #                         current_pred = noise_pred_cond_travel
                                    
                #                     # 保存时间旅行结果
                #                     current_latent = latent_travel.clone()
                                
                #                 # 3.3 用时间旅行结果替换原始采样结果
                #                 #latent = current_latent

                #                 # 计算x0估计
                #                 if i + 1 == sample_step:
                #                     temp_x0 = latent[:, -latent_frame_zero:, :, :] + (0 - sampling_sigmas[i]) * current_pred[:, -latent_frame_zero:, :, :]
                #                 else:
                #                     temp_x0 = latent[:, -latent_frame_zero:, :, :] + (sampling_sigmas[i+1] - sampling_sigmas[i]) * current_pred[:, -latent_frame_zero:, :, :]
                #                 # 更新潜在状态
                #                 index1 = min(sample_step-1, i + 1)
                #                 if step_sample > 0:
                #                     latent = torch.cat([
                #                         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                         (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                #                         temp_x0
                #                     ], dim=1)
                #                 else:
                #                     latent = torch.cat([
                #                         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                         (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                #                         temp_x0
                #                     ], dim=1)
                #             else:
                #                 # 更新潜在状态
                #                 index1 = min(sample_step-1, i + 1)
                #                 if step_sample > 0:
                #                     latent = torch.cat([
                #                         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                         (1 - sampling_sigmas[index1]) * model_input_1[:, :-latent_frame_zero, :, :], 
                #                         temp_x0
                #                     ], dim=1)
                #                 else:
                #                     latent = torch.cat([
                #                         noise[:, :-latent_frame_zero, :, :] * sampling_sigmas[index1] + 
                #                         (1 - sampling_sigmas[index1]) * model_input[:, :-latent_frame_zero, :, :], 
                #                         temp_x0
                #                     ], dim=1)


                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):

                        for i in range(sample_step):
                            latent_model_input = [latent]
                            timestep = [sampling_sigmas[i]*1000]
                            timestep = torch.tensor(timestep).to(device)

                            noise_pred_cond, _ = transformer(\
                                latent_model_input, t=timestep, rand_num_img=rand_num_img, **arg_c)
                            noise_pred_uncond, _ = transformer(\
                                latent_model_input, t=timestep, rand_num_img=rand_num_img, **arg_null)

                            noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)

                            if i+1 == sample_step:
                                temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                            else:
                                temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]

                            index1 = min(sample_step-1,i+1)
                            if step_sample > 0:
                                latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input_1[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                            else:
                                latent = torch.cat([noise[:,:-latent_frame_zero,:,:]*sampling_sigmas[index1]+(1-sampling_sigmas[index1])*model_input[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                

                # latent_ori = latent
                # C,F,H,W = noise.shape
                # latent = noise
                # with torch.no_grad():
                #     with torch.autocast("cuda", dtype=torch.bfloat16):
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
                main_print(
                    f"--> Function running time: {elapsed:.4f} s"
                )
                
                global_step = 1
                if step_sample > 0:
                    model_input = torch.cat([model_input, latent[:,-latent_frame_zero:,:,:]],dim=1)
                else:
                    model_input = torch.cat([model_input[:,:-latent_frame_zero,:,:], latent[:,-latent_frame_zero:,:,:]],dim=1)
            
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video_cat = scale(vae, model_input) 
                    video = video_cat[:,-frame_zero:]
                    video_all.append(video)

                    if step_sample > 0:
                        model_input_de = torch.cat([model_input_de, video[:,-frame_zero:,:,:]],dim=1)
                    else:
                        model_input_de = torch.cat([model_input_de[:,:-frame_zero,:,:], video[:,-frame_zero:,:,:]],dim=1)

                    video = save_video(torch.cat(video_all,dim=1))
                
                # Ensure videoid is a string
                if isinstance(videoid, list):
                    videoid_str = "_".join(map(str, videoid))
                else:
                    videoid_str = str(videoid)

                filename = os.path.join(
                                        video_output_dir,
                                        videoid_str+"_"+str(caption_ori)+"_"+str(step_sample)+"_"+str(repeat_num)+".mp4",
                                    )
                export_to_video(video[0] , filename, fps=16)

                if step_sample + 1 < sample_num:
                    with torch.no_grad():
                        # Jump to ./wan/image2video.py
                        _, _, arg_c, _, model_input_1, clip_context, arg_null = wan_i2v.generate_next(
                            model_input_de,
                            model_input,
                            device,
                            caption[step_sample+1],
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
                            flag_sample=True,
                        )
                    model_input_1 = torch.cat([model_input_1,torch.zeros(16,latent_frame_zero,model_input_1.shape[2],model_input_1.shape[3]).to(device)],dim=1)
            
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
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Create model:
    cfg = WAN_CONFIGS["i2v-14B"]
    ckpt_dir = "./Yume-I2V-540P"

    # Referenced from https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py
    wan_i2v = wan.Yume(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device="cpu",
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )    
    from wan.modules.model import ConvNext3D,WanAttentionBlock,WanI2VCrossAttention,WanLayerNorm
    transformer = wan_i2v.model

    # transformer.patch_embedding_2x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,4,4))
    # transformer.patch_embedding_4x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,8,8))
    # transformer.patch_embedding_8x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,16,16))
    # transformer.patch_embedding_16x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,32,32))
    # transformer.patch_embedding_2x_f = torch.nn.Conv3d(36, 36, kernel_size=(1,4,4), stride=(1,4,4))

    transformer = transformer.to(torch.bfloat16)
        
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

    #transformer = transformer.to(device)

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

    #init t5, clip and vae
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

    dist.barrier()
    
    wan_i2v.device = device
    denoiser = load_denoiser()

    if args.jpg_dir != None:
        dataset_ddp = create_scaled_videos("/mnt/petrelfs/maoxiaofeng/FastVideo_i2v_pack/jpg/", 
                                        total_frames=33, 
                                        H1=544, 
                                        W1=960)
        image_sample = True
    else:
        dataset_ddp, dataset_length = mp4_data(args.video_root_dir)
        image_sample = False

    step_times = deque(maxlen=100)

    for step in range(1, (int(dataset_length)//world_size) + 2):
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
            caption_path = args.caption_path
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


    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
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

