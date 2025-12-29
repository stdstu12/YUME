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
import copy

from hyvideo.diffusion import load_denoiser
from fastvideo.dataset.latent_datasets import (LatentDataset)
from fastvideo.dataset.t2v_datasets import (StableVideoAnimationDataset)

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
                                      get_discriminator_fsdp_kwargs,
                                      get_DINO_fsdp_kwargs)
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

from transformers import AutoModel, AutoTokenizer


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
    video_processor = VideoProcessor(
        vae_scale_factor=vae_spatial_scale_factor)
    #print(video.shape,video)
    video = video_processor.postprocess_video(video.unsqueeze(0), output_type="pil")
    return video

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

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

import random


import pandas as pd

def get_caption(csv_path, video_file):
    """根据videoFile获取caption, 包含完整异常处理"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查必需列是否存在
        if 'videoFile' not in df.columns or 'caption' not in df.columns:
            raise ValueError("CSV文件中缺少'videoFile'或'caption'列")

        # 查询匹配项
        matches = df.loc[df['videoFile'] == video_file, 'caption']

        if len(matches) == 0:
            raise ValueError(f"未找到videoFile为'{video_file}'的记录")

        return matches.values[0]
    
    except FileNotFoundError:
        print(f"错误：文件'{csv_path}'不存在")
    except pd.errors.EmptyDataError:
        print("错误：CSV文件为空")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
    return None

from wan23.utils.utils import best_output_size

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

def distill_one_step_t2i(
    transformer,
    result_list,
    prompt_all,
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
    step1=None,
    step2=None,
    wan_i2v=None,
    denoiser=None,
    pipe=None,
    camption_model = None,
    tokenizer = None,
    rank = None,
    world_size = None,
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
        
        rand_num = random.random()  # i2v or v2v
      
        rank = dist.get_rank()
        rand_num = torch.ones(1)

        if rank == 0:
            rand_num = random.random()  # i2v or v2v
            s1 = torch.tensor([rand_num])
        else:
            s1 = torch.ones(1, dtype=torch.float)

        s1 = s1.to(device)
        dist.broadcast(s1, src=0)
        rand_num = float(s1)
        print(rand_num,rank)
        if rand_num < 0.8:
            rand_numca = random.random()  # i2v or v2v
            (
                pixel_values_vid,
                pixel_values_ref_img,
                caption,
                K_ctrl,
                c2w_ctrl,
                videoid,
            ) = next(loader)
            with torch.no_grad():
                pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
                pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
                latents = pixel_values_vid 
                latents = latents[:,-32:]
                latents = F.interpolate(latents, size=(704, 1280), mode='bilinear', align_corners=False)
            frame = ( latents.shape[1] - 1 )*4 + 1
            img1,img2,img3,img4 = extract_first_frame_from_latents(latents[:,0:1]),extract_first_frame_from_latents(latents[:,7:8]),\
            extract_first_frame_from_latents(latents[:,15:16]),extract_first_frame_from_latents(latents[:,31:32])

            latents = wan_i2v.vae.encode([latents.to(device)])[0]

            path1 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"1.jpg"
            img1.save(path1)
            path2 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"2.jpg"
            img2.save(path2)
            path3 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"3.jpg"
            img3.save(path3)
            path4 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"4.jpg"
            img4.save(path4)

            generation_config = dict(max_new_tokens=1024, do_sample=True)

            # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
            pixel_values1 = load_image(path1, max_num=12).to(torch.bfloat16).to(device)
            pixel_values2 = load_image(path2, max_num=12).to(torch.bfloat16).to(device)
            pixel_values3 = load_image(path3, max_num=12).to(torch.bfloat16).to(device)
            pixel_values4 = load_image(path4, max_num=12).to(torch.bfloat16).to(device)
            pixel_values = torch.cat((pixel_values1, pixel_values2, pixel_values3, pixel_values4), dim=0)

            question = '<image>\nWatch the given egocentric (first-person) walking video (multi-image) and write a detailed, content-rich caption of approximately 70 words for video generation. Describe the scene, focusing solely on visible people, objects, scenery, weather, lighting, atmosphere, and activities, while avoiding any mention of camera movement, lens changes, or filming techniques.'

            response, history = camption_model.chat(tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)
 
            caption = "realistic style. " + caption[0] + response
            video_id = "city_walk_"

        else: # rand_num >= 0.3 and rand_num < 0.6:
            import json
            video_id = "hecheng"+str(rank)+"_"+str(step)
            step1 += 1
            json_path = prompt_all[ ( (step1-1)*world_size + rank) % len(prompt_all) ]

            max_retries = 300 # 最大重试次数
            retry_count = 0
            success = False

            while not success and retry_count < max_retries:
                #if True:
                try:
                    # 读取JSON文件
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)

                    # 提取caption和视频ID
                    caption = metadata.get("prompt", "")
                    video_id = metadata.get("id", "")
                    #caption = "AI video style. " + caption

#                     # 构建视频路径
#                     video_path = os.path.join(os.path.dirname(json_path), f"{video_id}.mp4")
#                     from torchvision.io import read_video
#                     from torchvision.transforms import Resize
#                     # 读取视频并转换为张量
#                     video, audio, info = read_video(
#                         video_path, 
#                         pts_unit='sec', 
#                         output_format="TCHW"
#                     )

#                     # 检查视频是否为空
#                     if len(video) == 0:
#                         raise RuntimeError("视频文件为空")

#                     # 调整视频尺寸
#                     target_height = 704
#                     target_width = 1280
#                     resize_transform = Resize((target_height, target_width))
#                     resized_video = torch.stack([resize_transform(frame) for frame in video])
#                     video_tensor = resized_video.permute(1, 0, 2, 3)

#                     latents = (video_tensor/255.0)*2-1
#                     frame = latents.shape[1]
#                     latents = wan_i2v.vae.encode([latents.to(device).to(torch.float32)])[0]
                    
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        latents = wan_i2v.t2v_dmd(
                                    caption,
                                    teacher_transformer=teacher_transformer,
                        )[0]
                    frame = ( latents.shape[1] - 1 ) * 4 + 1

                    caption = "AI video style. " + caption                        
                    rand_num_ca = 0.3
                    video_id = "AI_hecheng_"

                    success = True  # 标记成功

                except Exception as e:
                    print(f"处理 {json_path} 时出错: {str(e)}")
                    retry_count += 1

                    # 随机选择新的JSON路径
                    json_path = random.choice(prompt_all)
                    print(f"重试 {retry_count}/{max_retries}: 随机选择新路径 {json_path}")

                    # 如果是最后一次尝试仍然失败
                    if retry_count >= max_retries:
                        print(f"所有尝试失败，跳过此样本")
                        # 可以在这里设置默认值或跳过处理
                        # 例如: latents = torch.zeros(...) 或 continue
                        break
        print("T2I" ,caption)
        
        with torch.no_grad():
            # Jump to ./wan/image2video.py
            arg_c, arg_null, noise = wan_i2v.generate(
                       caption,
                       frame_num=frame )
        model_input = latents

#         if rand_num_ca >= 0.6:  
#             context1 = wan_i2v.text_encoder([caption1], device)  
#             context2 = wan_i2v.text_encoder([caption], device)  

#             context = torch.cat([context1[0],context2[0]],dim=0)
#             arg_c['context'] = [context]


        #wan_i2v.text_encoder.model.to("cpu")
            
        if True:
            # Incorporate masks during training
            xt, t, model_output, loss_dict_mask, x0, t  = denoiser.training_losses(
                        transformer,
                        latents,
                        arg_c,
                        n_tokens=None,
                        i2v_mode=None,
                        cond_latents=None,
                        args=args,
                        training_cache=True,
                        enable_mask = True,
            )
            
            loss = loss_dict_mask.mean()
            loss.backward()


        xt, t, model_output, loss_dict, x0, t  = denoiser.training_losses(
                    transformer,
                    latents,
                    arg_c,
                    n_tokens=None,
                    i2v_mode=None,
                    cond_latents=None,
                    args=args,
                    training_cache=True,
                    enable_mask = False,
        )
        loss = loss_dict.mean()

        if False:
            model_denoing = xt - t*model_output
            model_denoing = model_denoing
            model_input_gan = model_input
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

    if step % 15 == 0:
        sampling_sigmas = get_sampling_sigmas(25, 7.0)
        latent = noise[0].detach()
        latent = [torch.randn_like(noise[0])]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(25):
                    latent_model_input = latent

                    timestep = [sampling_sigmas[i]*1000]
                    timestep = torch.tensor(timestep).to(device)

                    noise_pred_cond = transformer(\
                    latent_model_input, t=timestep, **arg_c, flag=False)[0]
                    
                    noise_pred_uncond = transformer(\
                            latent_model_input, t=timestep, **arg_null, flag=False)[0]

                    noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)

                    if i+1 == 25:
                        latent[0] = latent[0] + (0-sampling_sigmas[i])*noise_pred_cond
                    else:
                        latent[0] = latent[0] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond
        latent = latent[0]                       
        global_step = 1
        latent = latent[:,:,:,:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = scale(vae, latent)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_ori = scale(vae, model_input)

        filename = os.path.join(
                "./outputs_i2v_pack_distil/",
                video_id+str(step)+"_"+"_imgt2i_"+str(device)+".mp4",
            )
        export_to_video(video[0] , filename, fps=16)
        
        filename = os.path.join(
                "./outputs_i2v_pack_distil/",
                video_id+str(step)+"_"+"_imgorit2i_"+str(device)+".mp4",
            )
        export_to_video(video_ori[0] , filename, fps=16)

        
        filename = os.path.join(
                "./outputs_i2v_pack_distil/",
                video_id+str(step)+"_"+"_imgorit2i_"+str(device)+".txt",
            )
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(caption)  
                
    # update ema                              
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(),
                                       transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(
                    torch.lerp(p_averaged.detach(), p_model.detach(),
                               1 - ema_decay))


    #wan_i2v.text_encoder.model.to(device)

    return total_loss, grad_norm.item(), model_pred_norm, step1, step2

def distill_one_step(
    transformer,
    result_list,
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
    step2=None,
    wan_i2v=None,
    denoiser=None,
    camption_model = None,
    tokenizer = None,
    rank = None,
    world_size = None,
):
    total_loss = 0.0
    optimizer.zero_grad()
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
  
    for _ in range(gradient_accumulation_steps):
       
        #rand_num = random.random()  # i2v or v2v

        rand_num = random.random()  # i2v or v2v
        rank = dist.get_rank()
        rand_num = torch.ones(1)
        if rank == 0:
            rand_num = random.random()  # i2v or v2v

            s1 = torch.tensor([rand_num], dtype=torch.float32, device=device)
        else:
            s1 = torch.ones(1, dtype=torch.float32)
        s1 = s1.to(device)
        dist.broadcast(s1, src=0)
        rand_num = float(s1)
        print("rand_numrand_num29129h9h89h",rand_num)
        
        if rand_num >0.1:
            (
                pixel_values_vid,
                pixel_values_ref_img,
                caption,
                K_ctrl,
                c2w_ctrl,
                videoid,
            ) = next(loader)
            latent_frame_zero = 8


            rand_num = 0.3

            frame_pixel = pixel_values_vid.shape[1]
            frame_pixel = ( frame_pixel // 4 ) * 4 + 1
            if frame_pixel > pixel_values_vid.shape[1]:
                frame_pixel = frame_pixel - 4

            pixel_values_vid = pixel_values_vid[:,:frame_pixel]



            rand_num_img = random.random()  # i2v or v2v
            if pixel_values_vid.shape[1] <= 33:
                rand_num_img = 0.3



            with torch.no_grad():
                pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
                pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
                latents = pixel_values_vid 

            max_area=704 * 1280
            iw, ih  = latents.shape[2:] 
            dh, dw = wan_i2v.patch_size[1] * wan_i2v.vae_stride[1], wan_i2v.patch_size[
                2] * wan_i2v.vae_stride[2]
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)
            scale1 = max(ow / iw, oh / ih)
            latents = F.interpolate(latents, size=(round(iw * scale1), round(ih * scale1)), mode='bilinear', align_corners=False)

            model_input = latents
            h1,w1 = latents.shape[2:]

            rand_num_img1 = rand_num_img
            if rand_num_img < 0.4:
                model_input = model_input[:,-33:]
                model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:33]],dim=1)
                rand_num_img = 0.6
                rand_num_img1 = 0.3

            model_input_caption = model_input[:,-32:]

            frame = model_input.shape[1]

            #rand_num_caption = random.random()  
  
            rand_num_caption = random.random()  # i2v or v2v
            rank = dist.get_rank()
            rand_num_caption = torch.ones(1, dtype=torch.float32, device=device)
            if rank == 0:
                rand_num_caption = random.random()  # i2v or v2v
                s1 = torch.tensor([rand_num_caption], dtype=torch.float32, device=device)
            else:
                s1 = torch.ones(1, dtype=torch.float32)
            s1 = s1.to(device)
            dist.broadcast(s1, src=0)
            rand_num_caption = float(s1)
            
            print("rand_num_captionrand_num_captionrand_num_caption1111111",rand_num_caption)
            print(rand_num,rank,"-------------------------------------------")
            if rand_num_caption > 0.4:
                img1,img2,img3,img4 = extract_first_frame_from_latents(model_input_caption[:,0:1]),extract_first_frame_from_latents(model_input_caption[:,7:8]),\
                extract_first_frame_from_latents(model_input_caption[:,15:16]),extract_first_frame_from_latents(model_input_caption[:,31:32])

                path1 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"1.jpg"
                img1.save(path1)
                path2 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"2.jpg"
                img2.save(path2)
                path3 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"3.jpg"
                img3.save(path3)
                path4 = './img1/'+"_"+str(step)+"_"+str(rank)+"_"+"4.jpg"
                img4.save(path4)

                generation_config = dict(max_new_tokens=1024, do_sample=True)

                # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
                pixel_values1 = load_image(path1, max_num=12).to(torch.bfloat16).to(device)
                pixel_values2 = load_image(path2, max_num=12).to(torch.bfloat16).to(device)
                pixel_values3 = load_image(path3, max_num=12).to(torch.bfloat16).to(device)
                pixel_values4 = load_image(path4, max_num=12).to(torch.bfloat16).to(device)
                pixel_values = torch.cat((pixel_values1, pixel_values2, pixel_values3, pixel_values4), dim=0)


                question = '<image>\nWatch the given egocentric (first-person) walking image and write a detailed, content-rich caption of around 70 words for video generation, focusing only on visible people, objects, scenery, weather, lighting, atmosphere, and activities, and avoiding any mention of camera movement, lens changes, or filming techniques.'

                response, history = camption_model.chat(tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)
                caption_ori = caption[0]
                caption = list(caption)
                caption[0] = caption[0] + " " + response
                print(f"获取到的caption: {caption}")
            else:
                print(f"未能获取caption: {caption}")


            print(model_input.to(device)[:,:-32].to(device).shape,model_input.to(device)[:,-32:].to(device).shape,"he89qged98qgw9wg")
            print(model_input.to(device)[:,:-32].to(device).shape,model_input.to(device)[:,-32:].to(device).shape,"he89qged98qgw9wg")\
                                
            #wan_i2v.text_encoder.model.to("cpu")
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model_input = torch.cat([wan_i2v.vae.encode([model_input.to(device)[:,:-32].to(device)])[0], \
                                             wan_i2v.vae.encode([model_input.to(device)[:,-32:].to(device)])[0]],dim=1) 
                    model_input_sample = model_input
            #wan_i2v.text_encoder.model.to(device)


            latents = model_input
            img =  model_input[:,:-latent_frame_zero]

            input_encoder = True
            if rand_num_img < 0.4:
                input_encoder = False

                
        else:
            videoid = "hailuo"+str(rank)+"_"+str(step)
            rand_num_caption = 0.3
            
            # 初始化重试机制
            max_retries = 300
            retry_count = 0
            success = False
            step2+=1

            while not success and retry_count < max_retries:
                try:
                    from torchvision.io import read_video
                    from torchvision.transforms import Resize
                    # 获取视频路径（首次或重试时使用）
                    if retry_count == 0:
                        # 首次尝试使用原始索引
                        index = ((step2-1)*world_size + rank) % len(result_list)
                    else:
                        # 重试时随机选择新索引
                        index = random.randint(0, len(result_list) - 1)

                    mp4_path, caption, caption1 = result_list[index]

                    print(f"尝试读取视频: {mp4_path} (尝试 {retry_count+1}/{max_retries})")
                    max_area = 704 * 1280

                    # 读取视频
                    video, audio, info = read_video(
                        mp4_path, 
                        pts_unit='sec', 
                        output_format="TCHW"
                    )

                    # 如果视频帧数为0，视为读取失败
                    if video.shape[0] == 0:
                        raise RuntimeError("读取到空视频帧")

                    # 调整视频尺寸
                    target_height = 704
                    target_width = 1280
                    resize_transform = Resize((target_height, target_width))
                    resized_video = torch.stack([resize_transform(frame) for frame in video])
                    video_tensor = resized_video.permute(1, 0, 2, 3)

                    # 后续处理...
                    latents = ((video_tensor / 255.0) * 2 - 1).to(device)
                    model_input = latents

                    rand_num_img = 0.3
                    if rand_num_img < 0.4:
                        model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:]], dim=1)
                        rand_num_img = 0.6
                        rand_num_img1 = 0.3

                    frame = model_input.shape[1]
                    
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            model_input = torch.cat([
                                wan_i2v.vae.encode([model_input.to(device)[:,:16].to(device)])[0], 
                                wan_i2v.vae.encode([model_input.to(device)[:,16:].to(device)])[0]
                            ], dim=1)

                    model_input_sample = model_input
                    latents = model_input
                    latent_frame_zero = model_input.shape[1] - 5
                    img = model_input[:,:5]
                    caption = [caption + caption1]
                    print("hailuo", caption)

                    # 标记成功
                    success = True

                except Exception as e:
                    retry_count += 1
                    print(f"视频读取失败: {e}")
                    if retry_count < max_retries:
                        print("随机选择另一个视频路径重试...")
                    else:
                        print(f"达到最大重试次数({max_retries})，跳过此视频")
                        # 可以在这里添加日志记录或错误处理
                        continue
            
            
#             mp4_path, caption, caption1 = result_list[ ( (step2-1)*world_size + rank ) % len(result_list) ]
#             step2+=1
#             print(mp4_path)
#             max_area=704 * 1280

#             # 构建视频路径
#             video_path = mp4_path
#             from torchvision.io import read_video
#             from torchvision.transforms import Resize

#             target_height = 704
#             target_width = 1280
#             # 读取视频并转换为张量
#             video, audio, info = read_video(
#                 video_path, 
#                 pts_unit='sec', 
#                 output_format="TCHW"
#             )

#             # 调整视频尺寸
#             resize_transform = Resize((target_height, target_width))
#             resized_video = torch.stack([resize_transform(frame) for frame in video])
#             video_tensor = resized_video.permute(1, 0, 2, 3)

#             latents = ((video_tensor/ 255.0)*2-1).to(device)
#             model_input =latents
            
#             rand_num_img = 0.3
#             if rand_num_img < 0.4:
#                 model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:]],dim=1)
#                 rand_num_img = 0.6
#                 rand_num_img1 = 0.3
                
#             frame = model_input.shape[1]

#             model_input = torch.cat([wan_i2v.vae.encode([model_input.to(device)[:,:16].to(device)])[0], \
#                                      wan_i2v.vae.encode([model_input.to(device)[:,16:].to(device)])[0]],dim=1)
#             model_input_sample = model_input
#             latents = model_input
#             latent_frame_zero=model_input.shape[1]-5
#             img =   model_input[:,:5]
#             caption = [caption + caption1]
#             print("hailuo",caption)
            
        with torch.no_grad():
            arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
                        caption[0],
                        frame_num=frame,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        img=img)
        mask2[0] = mask2[0].to(latents.dtype)


        
        if rand_num_caption > 0.4:  
            context1 = wan_i2v.text_encoder([caption_ori], device)  
            context2 = wan_i2v.text_encoder([response], device)  
            
            context = torch.cat([context1[0],context2[0]],dim=0)
            arg_c['context'] = [context]
        
        #wan_i2v.text_encoder.model.to("cpu")
        if True:
            xt, t, model_output, loss_dict_mask, x0, t  = denoiser.training_losses_i2v_pack(
                        transformer,
                        latents,
                        arg_c,
                        n_tokens=None,
                        i2v_mode=None,
                        cond_latents=None,
                        args=args,
                        latent_frame_zero=latent_frame_zero,
                        training_cache=True,
                        enable_mask = True,
                        mask2 =  mask2,
                        img = img[0],
                        step = step,
            )
            loss = loss_dict_mask.mean()
            loss.backward()

        xt, t, model_output, loss_dict, x0, t  = denoiser.training_losses_i2v_pack(
                    transformer,
                    latents,
                    arg_c,
                    n_tokens=None,
                    i2v_mode=None,
                    cond_latents=None,
                    args=args,
                    latent_frame_zero=latent_frame_zero,
                    training_cache=True,
                    enable_mask = False,
                    mask2 =  mask2,
                    img = img[0],
                    step = step,
        )
        loss = loss_dict.mean()


        if False:
            model_denoing = xt[:,-latent_frame_zero:] - t*model_output
            model_input_gan = model_input[:,-latent_frame_zero:] 
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
        latent_frame_zero = 8

        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(step,"step---------------------------------------------------------------")
        if step % 16 == 0:

            latent = noise.detach().squeeze()
            sample_step = 25
            sampling_sigmas = get_sampling_sigmas(sample_step, 7.0)
            latent = (1. - mask2[0]) * img[0]  + mask2[0] * latent


            latent_frame_zero = 8

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(sample_step):
                        latent_model_input = [latent]

                        timestep = [sampling_sigmas[i]*1000]
                        timestep = torch.tensor(timestep).to(device)
                        temp_ts = (mask2[0][0][:-latent_frame_zero, ::2, ::2] ).flatten()
                        temp_ts = torch.cat([
                            temp_ts,
                            temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                        ])
                        timestep = temp_ts.unsqueeze(0)

                        noise_pred_cond = transformer(\
                                latent_model_input, t=timestep, latent_frame_zero=latent_frame_zero, **arg_c)[0]
                        noise_pred_uncond = transformer(\
                                latent_model_input, t=timestep, latent_frame_zero=latent_frame_zero, **arg_null)[0]

                        noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)
                        if i+1 == sample_step:
                            temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                        else:
                            temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]

                        latent = torch.cat([model_input_sample[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                        print(latent.shape, img[0].shape,temp_x0.shape,"dxh98d1")
                        latent = (1. - mask2[0]) * img[0]  + mask2[0] * latent


            global_step = 1
            latent = latent[:,-latent_frame_zero:,:,:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = scale(vae, latent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video_ori = scale(vae, model_input[:,-latent_frame_zero:,:,:]) #[:,-latent_frame_zero:,:,:])
                
            # Ensure videoid is a string
            if isinstance(videoid, list):
                videoid_str = "_".join(map(str, videoid))
            else:
                videoid_str = str(videoid)

            if rand_num_img1 < 0.4:
                filename = os.path.join(
                                    "./outputs_i2v_pack_distil/",
                                    videoid_str+"_"+"_i2vnormnew_img_2_"+str(device)+".mp4",
                                )
                export_to_video(video[0] , filename, fps=16)
                filename = os.path.join(
                                    "./outputs_i2v_pack_distil/",
                                    videoid_str+"_"+"_i2vnormnewori_img_2_"+str(device)+".mp4",
                                )
                export_to_video(video_ori[0] , filename, fps=16)
            else:
                filename = os.path.join(
                                    "./outputs_i2v_pack_distil/",
                                    videoid_str+"_"+"_normnew_2_"+str(device)+".mp4",
                                )
                export_to_video(video[0] , filename, fps=16)
                filename = os.path.join(
                                    "./outputs_i2v_pack_distil/",
                                    videoid_str+"_"+"_normnew_ori_2_"+str(device)+".mp4",
                                )
                export_to_video(video_ori[0] , filename, fps=16)
            filename = os.path.join(
                                    "./outputs_i2v_pack_distil/",
                                    videoid_str+"_"+"_i2vnormnew_img_2_"+str(device)+".txt",
                                )
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(caption[0])  
        import torchvision.io

        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        print(dist.get_rank(),dist.get_world_size(),avg_loss.item(),"avg_loss")
        total_loss += avg_loss.item()

    # update ema
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(),
                                       transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(
                    torch.lerp(p_averaged.detach(), p_model.detach(),
                               1 - ema_decay))


    #wan_i2v.text_encoder.model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss, grad_norm.item(), model_pred_norm, step2


import wan23
from wan23.configs import WAN_CONFIGS

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

import datetime

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
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    # Create model:
    cfg = WAN_CONFIGS["ti2v-5B"]
    ckpt_dir = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/Wan2.2-TI2V-5B"

    # Referenced from https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py
    wan_i2v = wan23.WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=rank,
    )
    
    from wan23.modules.model import WanAttentionBlock,WanLayerNorm
    transformer = wan_i2v.model
    transformer = transformer.train().requires_grad_(True)
    
    transformer = wan_i2v.model
    
    transformer_tea = copy.deepcopy(transformer).to(torch.bfloat16)
  
    from safetensors import safe_open
    # 创建全面的映射字典
    key_mapping = {
        # Self-Attention 部分
        "attn1.norm_k.weight": "self_attn.norm_k.weight",
        "attn1.norm_q.weight": "self_attn.norm_q.weight",
        "attn1.to_k.bias": "self_attn.k.bias",
        "attn1.to_k.weight": "self_attn.k.weight",
        "attn1.to_out.0.bias": "self_attn.o.bias",
        "attn1.to_out.0.weight": "self_attn.o.weight",
        "attn1.to_q.bias": "self_attn.q.bias",
        "attn1.to_q.weight": "self_attn.q.weight",
        "attn1.to_v.bias": "self_attn.v.bias",
        "attn1.to_v.weight": "self_attn.v.weight",

        # Cross-Attention 部分
        "attn2.norm_k.weight": "cross_attn.norm_k.weight",
        "attn2.norm_q.weight": "cross_attn.norm_q.weight",
        "attn2.to_k.bias": "cross_attn.k.bias",
        "attn2.to_k.weight": "cross_attn.k.weight",
        "attn2.to_out.0.bias": "cross_attn.o.bias",
        "attn2.to_out.0.weight": "cross_attn.o.weight",
        "attn2.to_q.bias": "cross_attn.q.bias",
        "attn2.to_q.weight": "cross_attn.q.weight",
        "attn2.to_v.bias": "cross_attn.v.bias",
        "attn2.to_v.weight": "cross_attn.v.weight",

        # FFN 部分
        "ffn.net.0.proj.bias": "ffn.0.bias",
        "ffn.net.0.proj.weight": "ffn.0.weight",
        "ffn.net.2.bias": "ffn.2.bias",
        "ffn.net.2.weight": "ffn.2.weight",

        # Norm 部分
        "norm2.bias": "norm3.bias",
        "norm2.weight": "norm3.weight",

        # 文本和时间嵌入
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",

        # 时间投影
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",

        # 输出层
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",

        # Modulation参数
        "scale_shift_table": "head.modulation",
        #"scale_shift_table": "modulation",  # 用于块级映射
    }

    def convert_key(old_key):
        """将 safetensors 的键转换为模型需要的键"""
        # 处理非 blocks 开头的键名
        if not old_key.startswith("blocks."):
            # 特殊处理全局 modulation
            if old_key == "modulation":
                return "head.modulation"
            
            # 尝试直接匹配映射
            if old_key in key_mapping:
                return key_mapping[old_key]
            # 或者保持原样
            return old_key

        # 处理 blocks 开头的键名
        parts = old_key.split(".")
        if len(parts) < 3:
            return old_key

        block_idx = parts[1]  # 例如 "0"
        suffix = ".".join(parts[2:])  # 例如 "attn1.to_q.weight"

        # 特殊处理scale_shift_table -> modulation
        if suffix == "scale_shift_table":
            return f"blocks.{block_idx}.modulation"

        if suffix in key_mapping:
            return f"blocks.{block_idx}.{key_mapping[suffix]}"

        # 尝试部分匹配
        for pattern, replacement in key_mapping.items():
            if suffix.startswith(pattern):
                return f"blocks.{block_idx}.{suffix.replace(pattern, replacement, 1)}"

        # 如果都不匹配，保持原样但记录警告
        print(f"警告：未映射的键名: {old_key}")
        return old_key

    # 加载并转换权重
    fusionx_weights = {}
    with safe_open("/mnt/petrelfs/maoxiaofeng/Yume_v2_release/zip_data/diffusion_pytorch_model.safetensors", framework="pt") as f:
        for k in f.keys():
            new_key = convert_key(k)
            print(f"转换: {k} -> {new_key}")
            fusionx_weights[new_key] = f.get_tensor(k).to(torch.bfloat16)

    # 加载到模型
    missing_keys, unexpected_keys = transformer_tea.load_state_dict(fusionx_weights, strict=False)
    print("\n缺失的键:", missing_keys)
    print("意外的键:", unexpected_keys)
    del fusionx_weights
    
    # 检查是否有未映射的键
    if missing_keys or unexpected_keys:
        print("\n警告：仍有未匹配的键！")
        print("建议检查模型结构和权重文件的兼容性")

    # 添加 ConvNext3D 作为 pose_encoder
    print(transformer.device,"transformertransformertransformertransformertransformertransformer")

    transformer.patch_embedding_2x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,4,4))
    transformer.patch_embedding_4x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,8,8))
    transformer.patch_embedding_8x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,16,16))
    transformer.patch_embedding_16x = upsample_conv3d_weights(deepcopy(transformer.patch_embedding),(1,32,32))
    transformer.patch_embedding_2x_f = torch.nn.Conv3d(48, 48, kernel_size=(1,4,4), stride=(1,4,4))


    transformer.sideblock = WanAttentionBlock(transformer.dim, transformer.ffn_dim, transformer.num_heads, transformer.window_size, \
                                                  transformer.qk_norm, transformer.cross_attn_norm, transformer.eps)
    transformer.mask_token = torch.nn.Parameter(
        torch.zeros(1, 1, transformer.dim, device=transformer.device)
    )
    torch.nn.init.normal_(transformer.mask_token, std=.02)
    transformer.mask_token.requires_grad = True
    
    transformer = transformer.train().requires_grad_(True)
    
    
    if args.resume_from_checkpoint:
        print("args.resume_from_checkpoint", args.resume_from_checkpoint)
        (
            transformer,
            init_steps,
        ) = resume_checkpoint(
            transformer,
            args.resume_from_checkpoint,
        )

    
#     from ADD.models.discriminator import ProjectedDiscriminator
#     discriminator = ProjectedDiscriminator(c_dim=384).train()

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
    # discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(
    #      args.master_weight_type)

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

    transformer_tea = FSDP(
        transformer_tea,
        **fsdp_kwargs,
        use_orig_params=True,
    )

    # discriminator = FSDP(
    #      discriminator,
    #      **discriminator_fsdp_kwargs,
    #      use_orig_params=True,
    # )

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

    # params_to_optimize_dis = discriminator.parameters()
    # params_to_optimize_dis = list(
    #     filter(lambda p: p.requires_grad, params_to_optimize_dis))

    # discriminator_optimizer = bnb.optim.Adam8bit(
    #     params_to_optimize_dis,
    #     lr=args.discriminator_learning_rate,
    #     betas=(0, 0.999),
    #     weight_decay=args.weight_decay,
    #     eps=1e-8,
    #  )

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
    train_dataset = StableVideoAnimationDataset(height=704, width=1280, n_sample_frames=33, sample_rate=1)

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
    val_dataloader = train_dataloader

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

    init_steps = 15500
    if init_steps > 0:
        train_dataloader.dataset.skip = True
        # todo future
        for i in range(init_steps):
            print(i,init_steps)
            _ = next(loader)
        train_dataloader.dataset.skip = False

    vae = wan_i2v.vae


    

    dist.barrier()
    transformer.guidance_embed = False
    teacher_transformer = None
    
    wan_i2v.device = device
    denoiser = load_denoiser(7.0)

    #wan_i2v.text_encoder.model.to(torch.bfloat16)
    fsdp_kwargs = get_DINO_fsdp_kwargs()
    wan_i2v.text_encoder.model = FSDP(
        wan_i2v.text_encoder.model,
        **fsdp_kwargs,
        use_orig_params=True,
    )


#     # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
#     path = '/inspire/hdd/global_user/zhangkaipeng-24043/mxf/InternVL2-1B'
#     camption_model = AutoModel.from_pretrained(
#         path,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         use_flash_attn=True,
#         trust_remote_code=True).eval().to(device)
#     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    # from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
    # from lmdeploy.vl import load_image
    # from lmdeploy.vl.constants import IMAGE_TOKEN


    path = '/mnt/petrelfs/maoxiaofeng/Yume_v2_release/InternVL3-2B-Instruct'
    camption_model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    
    video_dir = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/prompt"
    # 2. 获取所有JSON文件路径
    prompt_all = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".json"):
                prompt_all.append(os.path.join(root, file))
    random.shuffle(prompt_all)
    random.shuffle(prompt_all)
    random.shuffle(prompt_all)
    random.shuffle(prompt_all)
    random.shuffle(prompt_all)
    
    
    data_root = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/dataset/data_hailuo"
    result_list = []
    # 获取所有子文件夹
    import glob
    subfolders = glob.glob(os.path.join(data_root, "*/"))
    
    for folder in subfolders:
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(folder, "*.txt"))
        
        for txt_path in txt_files:
            # 获取不带扩展名的文件名
            filename = os.path.splitext(os.path.basename(txt_path))[0].lower()
            
            # 确定caption1
            caption1 = ""
            if "right" in filename and "pan_" not in filename:
                caption1 = "The camera moves to the right (D). The rotation direction of the camera remains stationary."
            elif "left" in filename and "pan_" not in filename:
                caption1 = "The camera moves to the left (A). The rotation direction of the camera remains stationary."
            elif "forward" in filename:
                caption1 = "The camera pushes forward (W). The rotation direction of the camera remains stationary."
            elif "backward" in filename:
                caption1 = "The camera pulls back (S). The rotation direction of the camera remains stationary."
            elif "pan_down" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera tilts down (↓)."
            elif "pan_up" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera tilts up (↑)."
            elif "pan_right" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera pans to the right (→)."
            elif "pan_left" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera pans to the left (←)."
            elif "no_control" in filename:
                caption1 = ""  # 明确的空值
            
            # 读取txt内容作为caption
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            
            # 获取对应的mp4路径
            mp4_path = txt_path.replace(".txt", ".mp4")
            if not os.path.exists(mp4_path):
                continue
                
            # 添加到结果列表
            result_list.append((
                mp4_path,
                caption,
                caption1
            ))

    data_root = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/dataset/data_hailuo_911"
    subfolders = glob.glob(os.path.join(data_root, "*/"))
    
    for folder in subfolders:
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(folder, "*.txt"))
        
        for txt_path in txt_files:
            # 获取不带扩展名的文件名
            filename = os.path.splitext(os.path.basename(txt_path))[0].lower()
            
            # 确定caption1
            caption1 = ""
            if "right" in filename and "pan_" not in filename:
                caption1 = "The camera moves to the right (D). The rotation direction of the camera remains stationary."
            elif "left" in filename and "pan_" not in filename:
                caption1 = "The camera moves to the left (A). The rotation direction of the camera remains stationary."
            elif "forward" in filename:
                caption1 = "The camera pushes forward (W). The rotation direction of the camera remains stationary."
            elif "backward" in filename:
                caption1 = "The camera pulls back (S). The rotation direction of the camera remains stationary."
            elif "pan_down" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera tilts down (↓)."
            elif "pan_up" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera tilts up (↑)."
            elif "pan_right" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera pans to the right (→)."
            elif "pan_left" in filename:
                caption1 = "The camera's movement direction remains stationary (·). The camera pans to the left (←)."
            elif "no_control" in filename:
                caption1 = ""  # 明确的空值
            
            # 读取txt内容作为caption
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            
            # 获取对应的mp4路径
            mp4_path = txt_path.replace(".txt", ".mp4")
            if not os.path.exists(mp4_path):
                continue
                
            # 添加到结果列表
            result_list.append((
                mp4_path,
                caption,
                caption1
            ))  

    directory = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/dataset/mp4_api"
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否是MP4文件
        if filename.lower().endswith('.mp4'):
            mp4_path = os.path.join(directory, filename)
            
            # 构建对应的TXT文件路径
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(directory, f"{base_name}.txt")
            
            # 检查对应的TXT文件是否存在
            if os.path.exists(txt_path):
                # 读取TXT文件内容
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        txt_content = f.read()
                    
                    # 添加到结果列表
                    result_list.append((mp4_path, txt_content, ""))
                except Exception as e:
                    print(f"读取文件 {txt_path} 出错: {str(e)}")
            else:
                print(f"警告: 找不到 {mp4_path} 对应的TXT文件")
    
    directory = "/mnt/petrelfs/maoxiaofeng/Yume_v2_release/dataset/mp4_api_1000"
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否是MP4文件
        if filename.lower().endswith('.mp4'):
            mp4_path = os.path.join(directory, filename)
            
            # 构建对应的TXT文件路径
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(directory, f"{base_name}.txt")
            
            # 检查对应的TXT文件是否存在
            if os.path.exists(txt_path):
                # 读取TXT文件内容
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        txt_content = f.read()
                    
                    # 添加到结果列表
                    result_list.append((mp4_path, txt_content, ""))
                except Exception as e:
                    print(f"读取文件 {txt_path} 出错: {str(e)}")
            else:
                print(f"警告: 找不到 {mp4_path} 对应的TXT文件")

    random.shuffle(result_list)
    random.shuffle(result_list)
    random.shuffle(result_list)
    random.shuffle(result_list)
    
    step1 = 0
    step2 = 0
    step3 = 0

#     fsdp_kwargs = get_DINO_fsdp_kwargs()    
#     wan_i2v.text_encoder.model = FSDP(
#         wan_i2v.text_encoder.model,
#         **fsdp_kwargs,
#     )
    discriminator = None
    discriminator_optimizer = None
    
    #init_steps = 63
    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        if step%2 == 0:
            loss, grad_norm, pred_norm, step2 = distill_one_step(
                transformer,
                result_list,
                args.model_type,
                transformer_tea,
                ema_transformer,
                optimizer,
                discriminator,
                discriminator_optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                solver,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.max_grad_norm,
                args.num_euler_timesteps,
                1,
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
                step2 = step2,
                wan_i2v = wan_i2v,
                denoiser = denoiser,
                camption_model = camption_model,
                tokenizer = tokenizer,
                rank = rank,
                world_size = world_size,
            )
        else:
            loss, grad_norm, pred_norm, step1, step2 = distill_one_step_t2i(
                transformer,
                result_list,
                prompt_all,
                args.model_type,
                transformer_tea,
                ema_transformer,
                optimizer,
                discriminator,
                discriminator_optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                solver,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.max_grad_norm,
                args.num_euler_timesteps,
                1,
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
                step1 = step1,
                step2=step2,
                wan_i2v = wan_i2v,
                denoiser = denoiser,
                camption_model = camption_model,
                tokenizer = tokenizer,
                rank = rank,
                world_size=world_size,
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
        if args.log_validation and step % args.validation_steps == 0:
            optimizer.zero_grad()
            log_validation(
                args,
                transformer,
                device,
                torch.bfloat16,
                step,
                scheduler_type=args.scheduler_type,
                shift=args.shift,
                num_euler_timesteps=args.num_euler_timesteps,
                linear_quadratic_threshold=args.linear_quadratic_threshold,
                linear_range=args.linear_range,
                ema=False,
                loader_val=loader_val,
                vae = vae,
                text_encoder = text_encoder,
                fps=fps,
            )
            if args.use_ema:
                log_validation(
                    args,
                    ema_transformer,
                    device,
                    torch.bfloat16,
                    step,
                    scheduler_type=args.scheduler_type,
                    shift=args.shift,
                    num_euler_timesteps=args.num_euler_timesteps,
                    linear_quadratic_threshold=args.linear_quadratic_threshold,
                    linear_range=args.linear_range,
                    ema=True,
                    loader_val=loader_val,
                    vae = vae,
                    text_encoder = text_encoder,
                    fps=fps,
                )
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
