# webapp_single_gpu.py
# Flask 版：长视频生成（单图 i2v 首段 + 续帧），单卡、全程 BF16，
# 4 模型：transformer & vae (GPU 常驻)，text_encoder & caption_model (CPU 常驻，临时上 GPU)
# 采样逻辑与 sample_one 对齐：仅更新尾部 latent_frame_zero 帧，逐段拼接输出

from import_shim import ensure_packages, WAN_CONFIGS
ensure_packages()

import os
import sys
import time
import platform
import socket
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request, send_from_directory, Response
try:
    from flask_cors import CORS  # 可选
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

# ----------------------------- Logging setup -----------------------------
def setup_logging(app_name: str = "webapp", level=logging.INFO):
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{app_name}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(app_name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    def _excepthook(exc_type, exc, tb):
        logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass
    sys.excepthook = _excepthook

    try:
        import transformers, diffusers
        cuda_ok = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        logger.info("==== Runtime Env ====")
        logger.info("Python: %s", sys.version.replace("\n", " "))
        logger.info("OS: %s %s", platform.system(), platform.version())
        logger.info("torch: %s (cuda=%s) | transformers: %s | diffusers: %s",
                    torch.__version__, cuda_ok,
                    getattr(transformers, "__version__", "?"),
                    getattr(diffusers, "__version__", "?"))
        logger.info("Device: %s", dev)
    except Exception as e:
        logger.warning("Env probe failed: %s", e)

    return logger, log_file

LOGGER, LOG_PATH = setup_logging("webapp")
LOGGER.info("Log file: %s", LOG_PATH)

# ------------------------- Paths & runtime options -----------------------
CKPT_DIR = "./Wan2.2-TI2V-5B"              # Wan checkpoint dir
INTERNVL_PATH = "./InternVL3-2B-Instruct"  # InternVL dir
DEVICE_ID = 0                               # single GPU index
DTYPE = torch.bfloat16                      # 全程 BF16
OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- Small utils ------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # 让 SDPA/MatMul 在需要时走 TF32

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_sampling_sigmas(steps: int, shift: float):
    sigma = np.linspace(1, 0, steps + 1)[:steps]
    return (shift * sigma / (1 + (shift - 1) * sigma))

@torch.inference_mode()
def _postprocess_video(video: torch.Tensor, fps: int, out_path: str):
    # video: (C,F,H,W) in [-1,1]
    v = (video.clamp(-1,1).add(1).div(2))
    v = (v * 255).byte().cpu().numpy()       # (C,F,H,W)
    v = np.transpose(v, (1,2,3,0))           # (F,H,W,C)
    frames = [Image.fromarray(f) for f in v]
    export_to_video(frames, out_path, fps=fps)

def create_video_from_image(image_path: str, total_frames: int = 33, H1: int = 704, W1: int = 1280):
    """
    从单张图片创建 (F=total_frames, C, H1, W1) 的视频张量：
    - 第 0 帧放置该图（resize 到 H1xW1，并做 [-1,1] 归一化）
    - 其他帧为 0（后续采样会在尾段注入/更新）
    返回: (video(F,C,H,W), base_name, image_path)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    ten = torch.from_numpy(arr).permute(2,0,1).float() / 255.0  # (C,H,W)
    C,H,W = ten.shape
    vid = torch.zeros(C, total_frames, H1, W1)
    resized = F.interpolate(ten.unsqueeze(0), size=(H1,W1), mode='bilinear', align_corners=False)[0]
    vid[:,0] = (resized - 0.5) * 2
    base = os.path.splitext(os.path.basename(image_path))[0]
    return vid.permute(1,0,2,3), base, image_path  # (F,C,H,W)

# ----------------------------- Global state ------------------------------
@dataclass
class Models:
    device: Optional[torch.device] = None
    # Wan stack
    wan_i2v: Optional[object] = None
    transformer: Optional[nn.Module] = None
    vae: Optional[object] = None
    text_encoder: Optional[object] = None  # T5 (inside wan_i2v, kept CPU by default)
    # Caption
    caption_model: Optional[object] = None
    tokenizer: Optional[object] = None

MODELS = Models()
WAN_READY = False
CAP_READY = False

# 长视频上下文缓存（可续帧）
LAST: Dict[str, Any] = {
    "last_model_input_latent": None,  # (C,F,H,W) latent
    "last_model_input_de": None,      # (C,F,H,W) pixel-space [-1,1]
    "frame_total": 0,
    "last_video_path": None,
    "last_prompt": "",
}

def _ensure_device():
    LOGGER.info("[device] checking CUDA…")
    if not torch.cuda.is_available():
        LOGGER.error("[device] CUDA not available.")
        raise RuntimeError("CUDA 不可用，WanTI2V 需要 GPU。")
    torch.cuda.set_device(DEVICE_ID)
    dev_name = torch.cuda.get_device_name(DEVICE_ID)
    LOGGER.info("[device] using cuda:%d - %s", DEVICE_ID, dev_name)
    MODELS.device = torch.device(f"cuda:{DEVICE_ID}")
    torch.backends.cuda.matmul.allow_tf32 = True

def _trace_text(e: Exception) -> str:
    et = type(e).__name__
    return f"{et}: {e}\n\n" + traceback.format_exc()

# ---------- (保留) 可能用到的 patch-embedding 放大 ----------
def upsample_conv3d_weights_auto(conv_small: nn.Conv3d, size: Tuple[int,int,int], device, dtype):
    OC, IC, _, _, _ = conv_small.weight.shape
    with torch.no_grad():
        w = F.interpolate(conv_small.weight.data.to(dtype=dtype, device=device),
                          size=size, mode='trilinear', align_corners=False)
        big = nn.Conv3d(in_channels=IC, out_channels=OC,
                        kernel_size=size, stride=size, padding=0,
                        dtype=dtype, device=device)
        big.weight.copy_(w)
        if conv_small.bias is not None:
            big.bias = nn.Parameter(conv_small.bias.data.to(dtype=dtype, device=device).clone())
        else:
            big.bias = None
    return big

# ------------------------ On-demand loaders (BF16) -----------------------
@torch.inference_mode()
def load_wan() -> str:
    global WAN_READY
    LOGGER.info("[load_wan] start (BF16, DEVICE_ID=%s)", DEVICE_ID)
    t0 = time.perf_counter()
    if WAN_READY:
        LOGGER.info("[load_wan] already loaded.")
        return "✅ Wan 已加载（BF16）"

    _ensure_device()
    import importlib
    _wan23 = importlib.import_module("wan23")

    cfg = WAN_CONFIGS["ti2v-5B"]
    wan_i2v = _wan23.WanTI2V(config=cfg, checkpoint_dir=CKPT_DIR, device_id=DEVICE_ID)
    transformer = wan_i2v.model
    vae = wan_i2v.vae
    text_encoder = wan_i2v.text_encoder  # T5 wrapper（后续始终常驻 CPU）

    # transformer & vae 常驻 GPU + BF16
    transformer = transformer.to(device=MODELS.device, dtype=DTYPE).eval()
    try:
        for p in vae.model.parameters():
            p.data = p.data.to(DTYPE)
        vae.model.to(device=MODELS.device)
    except Exception:
        vae.model.to(device=MODELS.device)

    # sideblock + mask_token（与样例一致）
    from wan23.modules.model import WanAttentionBlock
    transformer.sideblock = WanAttentionBlock(
        transformer.dim, transformer.ffn_dim, transformer.num_heads,
        transformer.window_size, transformer.qk_norm, transformer.cross_attn_norm,
        transformer.eps
    ).to(device=MODELS.device, dtype=DTYPE)
    transformer.mask_token = nn.Parameter(torch.zeros(1,1,transformer.dim, device=MODELS.device, dtype=DTYPE))
    nn.init.normal_(transformer.mask_token, std=.02)
    transformer.eval()

    # T5 常驻 CPU
    try:
        text_encoder.model.cpu()
    except Exception:
        pass

    MODELS.wan_i2v = wan_i2v
    MODELS.transformer = transformer
    MODELS.vae = vae
    MODELS.text_encoder = text_encoder
    WAN_READY = True

    dt = time.perf_counter() - t0
    LOGGER.info("[load_wan] OK in %.2fs", dt)
    return f"✅ Wan 已加载（BF16）  用时 {dt:.1f}s"

@torch.inference_mode()
def load_caption_model() -> str:
    global CAP_READY
    LOGGER.info("[load_caption_model] start (BF16)")
    t0 = time.perf_counter()
    if CAP_READY:
        LOGGER.info("[load_caption_model] already loaded.")
        return "✅ InternVL 已加载（BF16）"

    from transformers import AutoModel, AutoTokenizer
    caption_model = AutoModel.from_pretrained(
        INTERNVL_PATH,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval()  # 先放 CPU
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)

    MODELS.caption_model = caption_model.cpu()
    MODELS.tokenizer = tokenizer
    CAP_READY = True

    dt = time.perf_counter() - t0
    LOGGER.info("[load_caption_model] OK in %.2fs", dt)
    return f"✅ InternVL 已加载（BF16）  用时 {dt:.1f}s"

# -------------------- Prompt 精炼（临时上 GPU，用完回 CPU） --------------------
@torch.inference_mode()
def refine_prompt_from_image(image_path: str, user_prompt: str) -> str:
    if not CAP_READY or MODELS.caption_model is None:
        return user_prompt
    try:
        def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height
            target = set( (i, j)
                          for n in range(min_num, max_num + 1)
                          for i in range(1, n + 1)
                          for j in range(1, n + 1)
                          if i * j <= max_num and i * j >= min_num )
            target = sorted(target, key=lambda x: x[0]*x[1])

            best = (1,1); best_diff = 1e9
            for r in target:
                ar = r[0]/r[1]
                d = abs(aspect_ratio - ar)
                if d < best_diff: best_diff, best = d, r
            tw, th = best[0]*image_size, best[1]*image_size
            blocks = best[0]*best[1]

            resized = image.resize((tw, th))
            imgs = []
            for i in range(blocks):
                box = ((i % (tw//image_size))*image_size,
                       (i // (tw//image_size))*image_size,
                       ((i % (tw//image_size))+1)*image_size,
                       ((i // (tw//image_size))+1)*image_size)
                imgs.append(resized.crop(box))
            if use_thumbnail and len(imgs)!=1:
                imgs.append(image.resize((image_size,image_size)))
            return imgs

        tr = build_transform(448)
        img = Image.open(image_path).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
        px = torch.stack([tr(im) for im in tiles])

        caption_model = MODELS.caption_model.to(MODELS.device, dtype=DTYPE)
        px = px.to(MODELS.device, dtype=DTYPE)
        question = (f"<image>\nWe want to generate a video using this prompt: \"{user_prompt}\". "
                    "Please refine it for this image (<image>). Keep it one paragraph.")
        gen_cfg = dict(max_new_tokens=512, do_sample=True)
        out = caption_model.chat(MODELS.tokenizer, px, question, gen_cfg)
        MODELS.caption_model.cpu()
        return out or user_prompt
    except Exception as e:
        LOGGER.exception("[caption] refine failed: %s", e)
        try:
            MODELS.caption_model.cpu()
        except Exception:
            pass
        return user_prompt

# -------------------------- 长视频生成 ---------------------
@dataclass
class LongGenArgs:
    prompt: str
    jpg_path: Optional[str]
    output_dir: str
    fps: int
    sample_steps: int
    sample_num: int
    frame_zero: int
    latent_frame_zero: int
    shift: float
    seed: int
    continue_from_last: bool
    refine_from_image: bool
    caption_path: Optional[str]
    mode: str  # Added mode for I2V or T2V

def _to_bf16(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=MODELS.device, dtype=DTYPE)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_bf16(t) for t in x)
    return x

def tiled_decode_overlap(vae, latents: torch.Tensor, n_tiles: int = 5, 
                         image_overlap_size: int = 32, latent_frame_zero=None) -> torch.Tensor:
    """
    精确匹配输出宽度的分块解码函数
    
    参数:
        vae: VAE 模型
        latents: 输入 latent 张量，形状为 (B, C, H, W)
        n_tiles: 分块数量
        image_overlap_size: 图像空间的重叠大小（像素）
        latent_frame_zero: 选择时间维度的参数
    
    返回:
        解码后的图像，宽度与输入 latent 精确匹配
    """
    # 获取 latent 尺寸
    b, c, latents_h, latents_w = latents.shape
    
    # VAE 上采样因子（根据您的VAE设置为16）
    scale_factor = 16
    
    # 计算期望的输出宽度
    expected_width = latents_w * scale_factor
    
    print(f"Latent宽度: {latents_w}, 期望输出宽度: {expected_width}")
    
    # 计算 latent 空间的重叠大小
    latent_overlap = max(1, image_overlap_size // scale_factor)
    print(f"Latent空间重叠大小: {latent_overlap}")
    
    # 计算每个分块的基本宽度（latent 空间）
    base_w = latents_w // n_tiles
    remainder = latents_w % n_tiles
    
    # 分配宽度，考虑余数
    tile_widths = [base_w + 1 if i < remainder else base_w for i in range(n_tiles)]
    print(f"各分块宽度: {tile_widths}")
    
    # 计算每个分块的起始和结束位置（考虑重叠）
    starts = []
    ends = []
    current = 0
    for i in range(n_tiles):
        # 起始位置
        start = current
        # 结束位置（考虑重叠）
        end = current + tile_widths[i]
        
        # 为除第一个外的所有分块添加向前重叠
        if i > 0:
            start -= latent_overlap
            
        # 为除最后一个外的所有分块添加向后重叠
        if i < n_tiles - 1:
            end += latent_overlap
            
        start = max(start, 0)
        end = min(end, latents_w)
        
        starts.append(start)
        ends.append(end)
        current += tile_widths[i]
    
    print(f"分块起始位置: {starts}")
    print(f"分块结束位置: {ends}")
    
    # 解码每个分块
    images = []
    for i in range(n_tiles):
        start = starts[i]
        end = ends[i]
        
        # 提取 latent 分块
        if latent_frame_zero is not None:
            latent_chunk = latents[:, -latent_frame_zero:, :, start:end]
        else:
            latent_chunk = latents[:, :, :, start:end]
            
        print(f"分块 {i}: latent尺寸 {latent_chunk.shape}")
        
        # 解码
        with torch.no_grad():
            image_chunk = vae.decode([latent_chunk])[0]
        print(f"分块 {i}: 解码后图像尺寸 {image_chunk.shape}")
        images.append(image_chunk)
        
        # 立即释放显存
        del latent_chunk
        torch.cuda.empty_cache()
    
    # 创建一个全零的结果张量
    result_height = images[0].shape[2]
    result = torch.zeros(images[0].shape[0], images[0].shape[1], result_height, expected_width, 
                        device=images[0].device, dtype=images[0].dtype)
    
    # 创建混合权重掩码
    blend_mask = torch.zeros(result_height, expected_width, device=result.device)
    
    # 计算每个分块在结果中的位置
    positions = []
    for i in range(n_tiles):
        # 计算这个分块在结果中的起始位置
        start_pos = starts[i] * scale_factor
        
        # 计算这个分块在结果中的结束位置
        end_pos = ends[i] * scale_factor
        end_pos = min(end_pos, expected_width)  # 确保不超出边界
        
        positions.append((start_pos, end_pos))
    
    print(f"各分块在结果中的位置: {positions}")
    
    # 对每个分块进行加权混合
    for i, (start_pos, end_pos) in enumerate(positions):
        image_chunk = images[i]
        chunk_width = image_chunk.shape[3]
        result_width_this_chunk = end_pos - start_pos
        
        print(f"分块 {i}: 结果位置 {start_pos}-{end_pos}, 分块宽度 {chunk_width}, 需要宽度 {result_width_this_chunk}")
        
        # 创建这个分块的权重掩码（只针对分块对应的区域）
        chunk_mask = torch.zeros(result_height, result_width_this_chunk, device=result.device)
        
        # 对于第一个和最后一个分块，使用全权重
        if i == 0 or i == n_tiles - 1:
            chunk_mask[:, :] = 1.0
        else:
            # 对于中间分块，创建渐变权重
            for j in range(result_width_this_chunk):
                if j < image_overlap_size:
                    # 左侧渐变：从0到1
                    weight = j / image_overlap_size
                elif j > result_width_this_chunk - image_overlap_size:
                    # 右侧渐变：从1到0
                    weight = (result_width_this_chunk - j) / image_overlap_size
                else:
                    # 中间部分：全权重
                    weight = 1.0
                
                chunk_mask[:, j] = weight
        
        # 确保分块宽度与需要宽度匹配
        if chunk_width != result_width_this_chunk:
            # 使用插值调整分块尺寸
            image_chunk = torch.nn.functional.interpolate(
                image_chunk, 
                size=(result_height, result_width_this_chunk), 
                mode='bilinear', 
                align_corners=False
            )
            print(f"分块 {i}: 使用插值调整尺寸从 {chunk_width} 到 {result_width_this_chunk}")
        
        # 应用权重到分块
        weighted_chunk = image_chunk * chunk_mask.unsqueeze(0).unsqueeze(0)
        
        # 累加到结果
        result[:, :, :, start_pos:end_pos] += weighted_chunk
        
        # 更新总权重掩码的对应部分
        blend_mask[:, start_pos:end_pos] += chunk_mask
    
    # 避免除以零
    blend_mask = torch.clamp(blend_mask, min=1e-8)
    
    # 归一化结果
    result = result / blend_mask.unsqueeze(0).unsqueeze(0)
    
    # 最终尺寸调整（应该不需要，但保留作为保险）
    if result.shape[3] != expected_width:
        result = torch.nn.functional.interpolate(
            result, 
            size=(result_height, expected_width), 
            mode='bilinear', 
            align_corners=False
        )
        print("使用插值进行最终宽度调整")
    
    # 清理内存
    del images, blend_mask
    torch.cuda.empty_cache()
    
    return result

# 检查并移动所有模型参数和缓冲区
def move_model_to_cpu(model):
    model = model.to('cpu')
    
    # 确保所有参数都在 CPU
    for param in model.parameters():
        param.data = param.data.cpu()
        if param.grad is not None:
            param.grad = param.grad.cpu()
    return model

import gc
import random
from wan23.utils.utils import best_output_size, masks_like

@torch.inference_mode()
def long_generate(g: LongGenArgs) -> Tuple[str, str]:
    if not WAN_READY or MODELS.wan_i2v is None or MODELS.vae is None or MODELS.transformer is None:
        raise RuntimeError("Wan 未加载，请先点击“加载所选模型”。")

    os.makedirs(g.output_dir, exist_ok=True)
    device = MODELS.device
    transformer = MODELS.transformer
    vae = MODELS.vae
    wan = MODELS.wan_i2v

    print("long_generate", g.mode)
    is_i2v_mode = g.mode == "I2V"  # Check if in I2V mode
    is_t2v_mode = g.mode == "T2V"  # Check if in T2V mode

    # 3) 采样循环（尾部 latent_frame_zero 帧）
    latent_frame_zero = int(g.latent_frame_zero)
    frame_zero = int(g.frame_zero)
    steps = int(g.sample_steps)
    sample_num = int(g.sample_num)
    shift = float(g.shift)
    frame_total = 0


    max_area=704*1280
    base_name = str(random.random())

    wan.text_encoder.model = wan.text_encoder.model.to("cpu")
    transformer = transformer.to("cpu")
    move_model_to_cpu(wan.text_encoder.model)
    move_model_to_cpu(transformer)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()


    # 1) 初始化
    if g.continue_from_last and LAST["last_model_input_de"] is not None:
        model_input_de: torch.Tensor = LAST["last_model_input_de"].to(device)               # (C,F,H,W)
        model_input_latent: torch.Tensor = LAST["last_model_input_latent"].to(device)       # (C,Fz,Hz,Wz)
        frame_total = int(LAST["frame_total"])
        first_img_path = None
    elif is_i2v_mode:
        if not g.jpg_path and is_i2v_mode:
            raise ValueError("首轮生成必须提供 jpg_path（单张图片路径）。")
        
        pixel_values_vid, base_name, img_path = create_video_from_image(
                g.jpg_path, total_frames=frame_zero, H1=704, W1=1280
            )  # (F,C,H,W) 704 1280
            

        first_img_path = img_path
        pixel_values_vid = pixel_values_vid.permute(1,0,2,3).contiguous().to(device)  # (C,F,H,W)

        # 头部复制 16 帧
        pixel_values_vid = torch.cat([pixel_values_vid[:,0:1].repeat(1,16,1,1),
                                      pixel_values_vid], dim=1)  # (C, 16+33, H, W)
        model_input_de = pixel_values_vid.clone()

        with torch.amp.autocast("cuda", dtype=DTYPE):
            lat_a = wan.vae.encode([model_input_de[:,:-frame_zero]])[0]
            lat_b = wan.vae.encode([model_input_de[:,-frame_zero:]])[0]
        model_input_latent = torch.cat([lat_a, lat_b], dim=1)  # (C,Fz,Hz,Wz)

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # vae.model = vae.model.to("cpu").to(DTYPE)
        # vae.scale[0] = vae.scale[0].to("cpu").to(DTYPE)
        # vae.scale[1] = vae.scale[1].to("cpu").to(DTYPE)
        # for p in vae.model.parameters():
        #     p.data = p.data.to("cpu").to(DTYPE)
        #print(latent_frame_zero,"latent_frame_zero")
        #with torch.amp.autocast("cuda", dtype=DTYPE):
            #print(vae.model.device,"W08RHF9W8HF9W8GHF9W48")
            #tiled_decode_overlap(wan.vae, model_input_latent[:,-latent_frame_zero:], latent_frame_zero=latent_frame_zero)
            #wan.vae.decode([model_input_latent[:,-latent_frame_zero:]])[0]
        
        print("vae_end")
        #zzzz

        frame_total = model_input_de.shape[1] - 16  # 可视帧（扣除头部 16）

    # 2) Prompt（可选图片精炼）
    final_prompt = g.prompt
    if g.refine_from_image and first_img_path:
        final_prompt = refine_prompt_from_image(first_img_path, final_prompt)

    if g.caption_path:
        try:
            os.makedirs(os.path.dirname(g.caption_path), exist_ok=True)
            with open(g.caption_path, "w", encoding="utf-8") as f:
                f.write(final_prompt)
        except Exception as e:
            LOGGER.warning("write caption failed: %s", e)


    arg_c = {}; arg_null = {}; seq_len = None
    try:
        try:
            if hasattr(wan, "text_encoder") and hasattr(wan.text_encoder, "model"):
                wan.text_encoder.model = wan.text_encoder.model.to("cuda")
        except Exception:
            pass

        with torch.amp.autocast("cuda", dtype=DTYPE):
            if is_t2v_mode and not g.continue_from_last:
                gen_ret = wan.generate(
                    final_prompt,
                    frame_num=frame_zero,
                    max_area=max_area,
                    latent_frame_zero=latent_frame_zero,
                    sampling_steps=steps,
                    shift=shift,
                )
            else:
                if g.continue_from_last:
                    gen_ret = wan.generate(
                        final_prompt,
                        img=model_input_latent,
                        frame_num=model_input_de.shape[1]+frame_zero,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        sampling_steps=steps,
                        shift=shift,
                    )
                else:
                    gen_ret = wan.generate(
                        final_prompt,
                        img=model_input_latent[:, :-latent_frame_zero],
                        frame_num=model_input_de.shape[1],
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        sampling_steps=steps,
                        shift=shift,
                    )
        try:
            if hasattr(wan, "text_encoder") and hasattr(wan.text_encoder, "model"):
                wan.text_encoder.model = wan.text_encoder.model.to("cpu")
            transformer = transformer.to("cuda")
        except Exception:
            pass


        if is_i2v_mode or g.continue_from_last:
            arg_c, arg_null, noise, mask2, img_lat = gen_ret
        else:
            arg_c, arg_null, noise = gen_ret
            
        if is_i2v_mode or g.continue_from_last:
            model_input_latent = _to_bf16(model_input_latent)

        noise = _to_bf16(noise)
        if is_i2v_mode:
            mask2     = _to_bf16(mask2)
            img_lat   = _to_bf16(img_lat)

        seq_len = int(arg_c.get("seq_len", 0))
        sampling_sigmas = get_sampling_sigmas(steps, shift)

        videos_to_concat = []
        if g.continue_from_last:
            videos_to_concat.append(model_input_de)

        for seg in range(sample_num):
            if seg == 0 and is_i2v_mode and not g.continue_from_last:
                latent = noise.clone()
                latent = _to_bf16(torch.cat([model_input_latent[:, :-latent_frame_zero, :, :], latent[:, -latent_frame_zero:, :, :]], dim=1))
            elif seg == 0 and is_t2v_mode and not g.continue_from_last:
                latent = noise.clone()
            else:
                latent = torch.randn(
                    wan.vae.model.z_dim, model_input_latent.shape[1] + latent_frame_zero,
                    model_input_latent.shape[2],
                    model_input_latent.shape[3],
                    dtype=DTYPE,
                    device=device
                )
                latent = _to_bf16(torch.cat([model_input_latent, latent[:, -latent_frame_zero:, :, :]], dim=1))
                mask1, mask2 = masks_like([latent], zero=True, latent_frame_zero=latent_frame_zero)

            #torch.randn_like(model_input_latent, dtype=DTYPE, device=device)

            #(1. - mask2[0]) * img_lat[0] + mask2[0] * latent)

            for i in range(steps):
                #ts_scalar = float(sampling_sigmas[i] * 1000.0)
                #tvec = torch.full((1, seq_len), ts_scalar, device=device, dtype=DTYPE)

                if is_i2v_mode or seg > 0 or g.continue_from_last:
                    ts_scalar = [sampling_sigmas[i]*1000]
                    timestep = torch.tensor(ts_scalar).to(device)
                    temp_ts = (mask2[0][0][:-latent_frame_zero, ::2, ::2] ).flatten()
                    temp_ts = torch.cat([
                                    temp_ts,
                                    temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                                ])
                    tvec = temp_ts.unsqueeze(0)
                else:
                    ts_scalar = [sampling_sigmas[i]*1000]
                    timestep = torch.tensor(ts_scalar).to(device)
                    tvec = timestep

                latent_model_input = [_to_bf16(latent)]

                with torch.autocast("cuda", dtype=DTYPE):
                    if is_i2v_mode or seg > 0 or g.continue_from_last:
                        noise_pred = transformer(latent_model_input, t=tvec,latent_frame_zero = latent_frame_zero, **arg_c)[0]
                    else:
                        noise_pred = transformer(latent_model_input, t=tvec,latent_frame_zero = latent_frame_zero, **arg_c, flag=False)[0]

                tail = latent[:,-latent_frame_zero:,:,:]
                pred_tail = noise_pred[:,-latent_frame_zero:,:,:]
                if i+1 == steps:
                    new_tail = tail + (0.0 - sampling_sigmas[i]) * pred_tail
                else:
                    new_tail = tail + (sampling_sigmas[i+1] - sampling_sigmas[i]) * pred_tail
                new_tail = _to_bf16(new_tail)
                latent = _to_bf16(torch.cat([latent[:,:-latent_frame_zero,:,:], new_tail], dim=1))

            transformer = transformer.to("cpu")
            move_model_to_cpu(transformer)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()


            with torch.amp.autocast("cuda", dtype=DTYPE):
                #video_tail = vae.decode([latent[:,-latent_frame_zero:]])[0].cuda() 
                video_tail = tiled_decode_overlap(wan.vae, latent, latent_frame_zero=latent_frame_zero) #vae.decode([latent[:,-latent_frame_zero:]])[0]

            videos_to_concat.append(video_tail)
            
            if is_i2v_mode or seg > 0:
                model_input_latent = torch.cat([model_input_latent[:,:-latent_frame_zero,:,:],
                                                latent[:,-latent_frame_zero:,:,:]], dim=1)
            else:
                model_input_latent = latent[:,-latent_frame_zero:,:,:]

            with torch.amp.autocast("cuda", dtype=DTYPE):
                #video_tail_px = vae.decode([latent[:,-latent_frame_zero:]])[0].cuda() 
                video_tail_px = video_tail #tiled_decode_overlap(wan.vae, latent, latent_frame_zero=latent_frame_zero) # vae.decode([latent[:,-latent_frame_zero:]])[0]
           
            transformer = transformer.to("cuda")


            # if video_tail_px.shape[1] < frame_zero:
            #     pad = video_tail_px[:,0:1,:,:].repeat(1, frame_zero - video_tail_px.shape[1], 1, 1)
            #     video_tail_px = torch.cat([pad, video_tail_px], dim=1)
            if is_i2v_mode or seg > 0:
                model_input_de = torch.cat([model_input_de[:,:-frame_zero,:,:],
                                            video_tail_px[:,-frame_zero:,:,:]], dim=1)
            else:
                model_input_de = video_tail_px[:,-frame_zero:,:,:]

            frame_total +=  video_tail_px[:,-frame_zero:,:,:].shape[1] #frame_zero

    

        video_cat = torch.cat(videos_to_concat, dim=1)
        ts = int(time.time())
        out_path = os.path.join(g.output_dir, f"{ts}_long.mp4")
        _postprocess_video(video_cat, g.fps, out_path)

        LAST["last_model_input_latent"] = model_input_latent.detach()#.to("cpu")
        LAST["last_model_input_de"] = model_input_de.detach()#.to("cpu")
        LAST["frame_total"] = frame_total
        LAST["last_video_path"] = out_path
        LAST["last_prompt"] = final_prompt

        return out_path, final_prompt
    finally:
        None


# ============================= Flask App ================================
app = Flask(__name__, static_url_path="/outputs", static_folder=OUTPUT_DIR)
if _HAS_CORS:
    CORS(app)

# ---- Home page (simple UI) ----
_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Long Video Generation (Flask, BF16, Single-GPU)</title>
<style>
:root {
  --bg:#0b1021; --fg:#c8d3f5; --muted:#8a98c9; --ok:#2ecc71; --err:#ff6b6b; --panel:#12183a; --accent:#7aa2f7;
}
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, Segoe UI, Arial; margin:0; color:var(--fg); background:linear-gradient(120deg,#0b1021,#10173a); }
header { padding:18px 28px; background:rgba(0,0,0,.25); position:sticky; top:0; backdrop-filter: blur(8px); border-bottom:1px solid #1e2754; }
h1 { margin:0; font-size:20px; letter-spacing:.4px; }
main { padding:24px; max-width:1080px; margin:0 auto; }
.card { background:var(--panel); border:1px solid #1b2450; border-radius:16px; padding:16px 18px; margin-bottom:16px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
.row { display:flex; gap:16px; flex-wrap:wrap; }
.col { flex:1 1 340px; min-width:320px; }
label { display:block; margin:8px 0 6px; color:#aab6ee; font-size:13px; }
input[type=text], input[type=number], textarea {
  width:100%; padding:10px 12px; border-radius:12px; border:1px solid #27306a; background:#0d1433; color:var(--fg);
}
textarea { min-height:120px; }
button { padding:10px 16px; border-radius:12px; border:1px solid #2b336d; background:linear-gradient(180deg,#172154,#101a46);
  color:#e9edff; cursor:pointer; transition: transform .05s ease, box-shadow .2s;
}
button:hover { box-shadow:0 8px 18px rgba(0,0,0,.35); }
button:active { transform: translateY(1px) scale(.99); }
.badge { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; border:1px solid #2a3168; background:#0e1540; margin-right:8px; font-size:12px; color:#b7c3ff; }
.badge.ok { background: rgba(46,204,113,.1); border-color:#284b36; color:#80ffb3; }
.badge.err { background: rgba(255,107,107,.1); border-color:#5a2a2a; color:#ff9b9b; }
video { width:100%; max-height:420px; outline: 1px solid #1e2754; border-radius:12px; background:#000; }
pre { margin:0; white-space:pre-wrap; word-break:break-word; }
.panel-title { font-weight:600; color:#bcd1ff; margin-bottom:6px; }
#overlay {
  position: fixed; inset: 0; background: rgba(10, 14, 35, .66);
  display:none; align-items: center; justify-content: center; backdrop-filter: blur(2px); z-index:999;
}
.spinner {
  width:56px; height:56px; border-radius:50%; border:4px solid rgba(255,255,255,.18);
  border-top-color: var(--accent); animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg) } }
.small { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>📹 长视频生成 — Flask / BF16 / 单卡</h1>
</header>
<div id="overlay"><div class="spinner"></div></div>
<main>

<div class="card">
  <div>
    <span id="wan_state" class="badge">Wan: 未加载</span>
    <span id="cap_state" class="badge">InternVL: 未加载</span>
  </div>
  <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
    <label><input id="chk_wan" type="checkbox"/> 加载 Wan (DiT + VAE + T5)</label>
    <label><input id="chk_cap" type="checkbox"/> 加载 InternVL (Caption)</label>
    <button onclick="doLoad()">📦 加载所选</button>
  </div>
</div>

<div class="card row">
  <div class="col">
    <div class="panel-title">1) 条件与参数</div>

    <!-- Mode Selection -->
    <div style="margin-top:8px;">
      <label><input id="mode_i2v" type="radio" name="mode" checked /> I2V 模式</label>
      <label><input id="mode_t2v" type="radio" name="mode" /> T2V 模式</label>
      <br/>
    </div>

    <label>首帧图片路径 (jpg_path)</label>
    <input id="jpg_path" type="text" placeholder="例如 D:\\imgs\\001.jpg"/>

    <label>Prompt</label>
    <textarea id="prompt" rows="5">A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage...</textarea>

    <div class="row">
      <div class="col">
        <label>FPS</label>
        <input id="fps" type="number" value="15"/>
      </div>
      <div class="col">
        <label>采样步数 (steps)</label>
        <input id="steps" type="number" value="50"/>
      </div>
      <div class="col">
        <label>每段可视帧 (frame_zero)</label>
        <input id="frame_zero" type="number" value="32"/>
      </div>
    </div>

    <div class="row">
      <div class="col">
        <label>尾部潜帧 (latent_frame_zero)</label>
        <input id="latent_zero" type="number" value="8"/>
      </div>
      <div class="col">
        <label>段数 (sample_num)</label>
        <input id="sample_num" type="number" value="1"/>
      </div>
      <div class="col">
        <label>shift</label>
        <input id="shift" type="number" step="0.1" value="7.0"/>
      </div>
    </div>

    <div class="row">
      <div class="col">
        <label>Seed（-1 随机）</label>
        <input id="seed" type="number" value="-1"/>
      </div>
      <div class="col">
        <label>输出目录</label>
        <input id="out_dir" type="text" value="outputs"/>
      </div>
      <div class="col">
        <label>(可选) 保存精炼文案到文件</label>
        <input id="cap_path" type="text"/>
      </div>
    </div>

    <div style="margin-top:8px;">
      <label><input id="cont" type="checkbox"/> 继续续帧（使用上次生成的干净 latent 作为条件）</label>
      <br/>
      <label><input id="refine" type="checkbox"/> 从图片精炼 Prompt（需加载 InternVL）</label>
    </div>

    <div style="margin-top:12px;">
      <button onclick="doGen()">🚀 开始/继续 生成</button>
    </div>
  </div>

  <div class="col">
    <div class="panel-title">2) 预览</div>
    <video id="video" controls></video>
    <div class="small" style="margin-top:6px;">生成成功后会自动更新到最新。</div>
  </div>
</div>

<div class="card">
  <div class="panel-title">日志</div>
  <pre id="log">（点击“拉取日志”查看）</pre>
  <div style="margin-top:8px;">
    <button onclick="pullLog()">拉取日志</button>
  </div>
</div>

<div class="card">
  <div class="panel-title">错误详情（完整 traceback ）</div>
  <pre id="trace">（若发生错误，这里会显示完整堆栈）</pre>
</div>

</main>

<script>
let LOG_PATH = "";

function el(id) { return document.getElementById(id); }
function showOverlay(v) {
  el('overlay').style.display = v ? 'flex' : 'none';
}
async function refreshStatus() {
  const r = await fetch('/api/status'); const j = await r.json();
  const wan = el('wan_state'); const cap = el('cap_state');
  wan.textContent = 'Wan: ' + (j.wan_ready ? '已加载' : '未加载');
  cap.textContent = 'InternVL: ' + (j.cap_ready ? '已加载' : '未加载');
  wan.className = 'badge ' + (j.wan_ready ? 'ok' : '');
  cap.className = 'badge ' + (j.cap_ready ? 'ok' : '');
  LOG_PATH = j.log_path || '';
}
async function doLoad() {
  el('trace').textContent = '';
  showOverlay(true);
  try {
    const sel = { wan: el('chk_wan').checked, cap: el('chk_cap').checked };
    const r = await fetch('/api/load', {
      method:'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sel)
    });
    const j = await r.json();
    let log = '';
    if (j.wan_msg) log += j.wan_msg + '\\n';
    if (j.cap_msg) log += j.cap_msg + '\\n';
    if (LOG_PATH) log += '日志文件: ' + LOG_PATH;
    el('log').textContent = log;
    if (!j.success) el('trace').textContent = j.trace || '';
  } catch (e) {
    el('trace').textContent = String(e);
  } finally {
    showOverlay(false);
    await refreshStatus();
  }
}
async function doGen() {
  el('trace').textContent = '';
  showOverlay(true);
  try {
    const payload = {
      prompt: el('prompt').value,
      jpg_path: el('jpg_path').value,
      output_dir: el('out_dir').value,
      fps: parseInt(el('fps').value||'15'),
      sample_steps: parseInt(el('steps').value||'50'),
      sample_num: parseInt(el('sample_num').value||'1'),
      frame_zero: parseInt(el('frame_zero').value||'32'),
      latent_frame_zero: parseInt(el('latent_zero').value||'8'),
      shift: parseFloat(el('shift').value||'7'),
      seed: parseInt(el('seed').value||'-1'),
      continue_from_last: el('cont').checked,
      refine_from_image: el('refine').checked,
      caption_path: el('cap_path').value,
      mode: el('mode_t2v').checked ? "T2V" : "I2V",  // This line ensures correct mode
    };
    const r = await fetch('/api/generate_long', {
      method:'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (j.success) {
      if (j.video_rel) el('video').src = j.video_rel;
      let msg = (j.info || '');
      if (LOG_PATH) msg += '\\n日志文件: ' + LOG_PATH;
      el('log').textContent = msg;
    } else {
      let msg = (j.error || 'ERROR');
      if (LOG_PATH) msg += '\\n日志文件: ' + LOG_PATH;
      el('log').textContent = msg;
      el('trace').textContent = j.trace || '';
    }
  } catch (e) {
    el('trace').textContent = String(e);
  } finally {
    showOverlay(false);
  }
}
async function pullLog() {
  const r = await fetch('/api/log/tail?n=500');
  const t = await r.text();
  el('log').textContent = t;
}
refreshStatus();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return Response(_HTML, mimetype="text/html; charset=utf-8")

@app.get("/api/status")
def api_status():
    return jsonify({
        "wan_ready": bool(WAN_READY),
        "cap_ready": bool(CAP_READY),
        "log_path": LOG_PATH,
        "last_video": LAST.get("last_video_path"),
        "frame_total": LAST.get("frame_total", 0),
    })

@app.post("/api/load")
def api_load():
    data = request.get_json(force=True, silent=True) or {}
    to_load_wan = bool(data.get("wan"))
    to_load_cap = bool(data.get("cap"))
    rst = {"success": True, "wan_msg": None, "cap_msg": None, "trace": None}

    try:
        if to_load_wan:
            rst["wan_msg"] = load_wan()
    except Exception as e:
        LOGGER.exception("[api_load] load_wan failed: %s", e)
        rst["success"] = False
        rst["wan_msg"] = f"[ERROR@Wan] {type(e).__name__}: {e}"
        rst["trace"] = _trace_text(e)

    try:
        if to_load_cap:
            rst["cap_msg"] = load_caption_model()
    except Exception as e:
        LOGGER.exception("[api_load] load_caption failed: %s", e)
        rst["success"] = False
        rst["cap_msg"] = f"[ERROR@InternVL] {type(e).__name__}: {e}"
        rst["trace"] = (rst["trace"] or "") + "\n\n" + _trace_text(e)

    return jsonify(rst)

@app.post("/api/generate_long")
def api_generate_long():
    data = request.get_json(force=True, silent=True) or {}
    try:
        g = LongGenArgs(
            prompt=str(data.get("prompt") or ""),
            jpg_path=(str(data.get("jpg_path") or "") or None),
            output_dir=str(data.get("output_dir") or OUTPUT_DIR),
            fps=int(data.get("fps") or 15),
            sample_steps=int(data.get("sample_steps") or 50),
            sample_num=int(data.get("sample_num") or 1),
            frame_zero=int(data.get("frame_zero") or 32),
            latent_frame_zero=int(data.get("latent_frame_zero") or 8),
            shift=float(data.get("shift") or 7.0),
            seed=int(data.get("seed") or -1),
            continue_from_last=bool(data.get("continue_from_last")),
            refine_from_image=bool(data.get("refine_from_image")),
            caption_path=(str(data.get("caption_path") or "") or None),
            mode=str(data.get("mode") or "I2V"),  # Added mode parameter
        )
        print(g.mode, "g_mode")
        # Check only for I2V mode if jpg_path is provided when not continuing from the last frame
        if g.mode == "I2V" and (not g.continue_from_last and not g.jpg_path):
            raise ValueError("首轮生成必须提供 jpg_path（单张图片路径）。若要续帧，请勾选“继续续帧”。")

        out_path, final_prompt = long_generate(g)
        out_abs = os.path.abspath(out_path)
        rel = os.path.relpath(out_abs, OUTPUT_DIR).replace("\\", "/")
        video_rel = f"/outputs/{rel}"

        return jsonify({
            "success": True,
            "video_abs": out_abs,
            "video_rel": video_rel,
            "info": f"Saved to {out_abs} | Device cuda:{DEVICE_ID} | DType BF16",
            "prompt": final_prompt
        })
    except Exception as e:
        LOGGER.exception("[api_generate_long] failed: %s", e)
        return jsonify({
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": _trace_text(e),
        })

@app.get("/api/log/tail")
def api_log_tail():
    n = int(request.args.get("n", 200))
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = "".join(lines[-n:])
        return Response(tail, mimetype="text/plain; charset=utf-8")
    except Exception as e:
        return Response(f"[log read error] {e}", mimetype="text/plain; charset=utf-8")

@app.get("/outputs/<path:filename>")
def static_outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

# ------------------------------- Launcher --------------------------------
def _port_is_free(h, p) -> bool:
    s = socket.socket(); s.settimeout(0.2)
    ok = s.connect_ex((h, p)) != 0
    s.close()
    return ok

def main():
    os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))

    host = "127.0.0.1"
    port = int(os.environ.get("WEB_PORT", "7666"))

    if not _port_is_free(host, port):
        LOGGER.error("[PORT] 端口已被占用：%s:%s", host, port)
        LOGGER.error("       解决：关闭占用端口的程序，或设置环境变量 WEB_PORT 改端口")
        print("Press Enter to exit …"); 
        try: input()
        except Exception: pass
        return

    url = f"http://{host}:{port}"
    LOGGER.info("[LAUNCH] 即将启动：%s", url)
    LOGGER.info("[LAUNCH] Log file: %s", LOG_PATH)

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except Exception as e:
        LOGGER.critical("[LAUNCH] 启动失败：%s", e, exc_info=True)
        print("Press Enter to exit …"); 
        try: input()
        except Exception: pass

if __name__ == "__main__":
    main() ,需要做3个修改，第一个修改点加入按钮控制prompt，        vocab1 = { 
            "W": "The camera pushes forward (W).",
            "A": "The camera moves to the left (A).",
            "S": "The camera pulls back (S).",
            "D": "The camera moves to the right (D).",
            "W+A": "The camera pushes forward and moves to the left (W+A).",
            """: "The camera pushes forward and moves to the right (W+D).", 
            "S+D": "The camera pulls back and moves to the right (S+D).",
            "S+A": "The camera pulls back and moves to the left (S+A).",
            "None": "The camera's movement direction remains stationary (·).",
        }，这有5个按钮包括了W,S,A,D，.(代表None), 分别以上下左右中间排列， 
        vocab2 = { 
            "→": "The camera pans to the right (→).",
            "←": "The camera pans to the left (←).",
            "↑": "The camera tilts up (↑).",
            "↓": "The camera tilts down (↓).",
            "↑→": "The camera tilts up and pans to the right (↑→).",
            "↑←": "The camera tilts up and pans to the left (↑←).",
            "↓→": "The camera tilts down and pans to the right (↓→).",
            "↓←": "The camera tilts down and pans to the left (↓←).",
            "·": "The rotation direction of the camera remains stationary (·)."
        }，这也有5个按钮包括了↑,↓,←,→，.(代表None), 分别以上下左右中间排列。这些位置放在1) 条件与参数中的Prompt上面，排列这些键盘一定要看起来美观舒适。注意组合只有"W+A","W+D"， "↓←"等等vocab1 和vocab2出现的"None"不能和wasd组合， "·"也一样。将这些按钮控制得到后的prompt和原始用户输入的g.prompt连接一起作为新的g.prompt。对于第2点修改，加入中英文按钮转换，将前端界面转换成英文或者中文，按钮在最上面。对于第3点修改，加入分辨率修改，可以选择544x960或者704x1280。移除尾部潜帧 (latent_frame_zero)，因为latent_frame_zero = (frame_zero-1)//4+1,用这个作为latent_frame_zero。加入显存占用优化按钮，如果使用，则"    wan.text_encoder.model = wan.text_encoder.model.to("cpu") 
    transformer = transformer.to("cpu")
    move_model_to_cpu(wan.text_encoder.model)
    move_model_to_cpu(transformer)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
"这段在long_generate函数的代码是运行的，否则不运行，添加提示(使用这个降低显存占用，但是减少生成速度)，加入VAE显存占用优化按钮，如果使用，则video_tail = tiled_decode_overlap(wan.vae, latent, latent_frame_zero=latent_frame_zero)，否则video_tail = vae.decode([latent[:,-latent_frame_zero:]])[0]，添加提示(使用这个降低显存占用，但是可能会影响生成质量)。注意只修改我提到的部分，其他部分不管写的怎样都不要修改，一字一句都不要修改，把完整的代码发给我，千万注意不要修改其他部分: