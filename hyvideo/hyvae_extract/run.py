from typing import Tuple, List, Dict
import sys
from pathlib import Path
import argparse
import time
import os
import traceback
import random
import numpy as np
from einops import rearrange
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VideoDataset, MultiBucketDataset, split_video_urls
import json
import glob
from omegaconf import OmegaConf
from hyvideo.vae import load_vae

DEVICE = "cuda"
DTYPE = torch.float16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def extract(
    vae: torch.nn.Module,
    meta_files: List[str],
    output_base_dir: str,
    sample_n_frames: int,
    target_size: Tuple[int, int],
    enable_multi_aspect_ratio: bool = False,
    use_stride: bool = False,
    batch_size=None,
):
    dataset = VideoDataset(
        meta_files=meta_files,
        latent_cache_dir=output_base_dir,
        sample_size=target_size,
        sample_n_frames=sample_n_frames,
        is_center_crop=True,
        enable_multi_aspect_ratio=enable_multi_aspect_ratio,
        vae_time_compression_ratio=vae.time_compression_ratio,
        use_stride=use_stride
    )
    if batch_size is not None:
        dataset = MultiBucketDataset(dataset, batch_size=batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=dataset.collate_fn if batch_size is not None else None,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=False,
    )
    normalize_fn = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    save_json_path = Path(output_base_dir) / "json_path"
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path, exist_ok=True)

    for i, item in enumerate(dataloader):
        print(f"processing video latent extraction {i}")
        if batch_size is None:
            if item.get("valid", True) is False:
                continue
            item["videoid"] = [item["videoid"]]
            item["valid"] = [item["valid"]]
            item["prompt"] = [item["prompt"]]
        try:
            pixel_values = item["pixel_values"]
            pixel_values = pixel_values.to(device=vae.device, dtype=vae.dtype)
            pixel_values = pixel_values / 255.
            pixel_values = normalize_fn(pixel_values)
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
            z = vae.encode(pixel_values).latent_dist.mode()
            z = z.detach().to(DTYPE).cpu().numpy()

            assert z.shape[0] == len(item["videoid"])
            for k in range(z.shape[0]):
                save_path = Path(output_base_dir) / f"{item['videoid'][k]}.npy"
                np.save(save_path, z[k][None, ...])
                data = {"video_id": item["videoid"][k],
                    "latent_shape": z[k][None,...].shape,
                    "video_path": item["video_path"][k], 
                    "prompt": item["prompt"][k],
                    "npy_save_path": str(save_path)}
                with open(save_json_path / f"{item['videoid'][k]}.json", "w", encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()

def main(
    local_rank: int,
    vae_path: str,
    meta_files: str,
    output_base_dir: str,
    sample_n_frames: int,
    target_size: Tuple[int, int],
    enable_multi_aspect_ratio: bool = False,
    use_stride: bool = False,
    seed: int = 42,
):
    seed_everything(seed)

    global_rank = local_rank
    world_size = int(os.environ["HOST_GPU_NUM"])

    print(f"split video urls")
    start, end, meta_files = split_video_urls(meta_files, global_rank, world_size)

    print(f"Load VAE")
    vae, vae_path, spatial_compression_ratio, time_compression_ratio = load_vae(
        vae_type="884-16c-hy",
        vae_precision='fp16',
        vae_path=vae_path,
        device=DEVICE,
    )

    # vae.enable_temporal_tiling()
    vae.enable_spatial_tiling()
    vae.eval()

    print(f"processing video latent extraction")
    extract(vae, meta_files, output_base_dir, sample_n_frames, target_size, enable_multi_aspect_ratio, use_stride)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, required=True)
    parser.add_argument("--config", default='./vae.yaml', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    vae_path = config.vae_path
    sample_n_frames = config.sample_n_frames
    target_size = [config.target_size[0], config.target_size[1]]
    enable_multi_aspect_ratio = config.enable_multi_aspect_ratio
    output_base_dir = config.output_base_dir
    use_stride = config.use_stride
    meta_files = config.video_url_files
    print(meta_files,"nnnnnnnnnn")
    main(args.local_rank, vae_path, meta_files, output_base_dir, sample_n_frames, target_size, enable_multi_aspect_ratio, use_stride)