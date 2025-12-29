from typing import Tuple, List
from decord import VideoReader
import urllib
import io
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from pathlib import Path
import sys
import json


def split_video_urls(meta_files: str, global_rank: int, world_size: int):
    meta_paths = []
    meta_paths.extend([line.strip() for line in open(meta_files, 'r').readlines()])
    num_videos = len(meta_paths)
    num_videos_per_rank = num_videos // world_size
    remainder = num_videos % world_size

    # Calculate start and end indices
    start = num_videos_per_rank * global_rank + min(global_rank, remainder)
    end = start + num_videos_per_rank + (1 if global_rank < remainder else 0)

    return start, end, meta_paths[start:end]

class MultiBucketDataset(IterableDataset):
    def __init__(self, source: Dataset, batch_size: int, max_buf = 64):
        super().__init__()
        self.source = source
        self.batch_size = batch_size
        self.buffer = {}   
        self.max_buf = max_buf
        self.size = 0

    @staticmethod
    def collate_fn(samples):
        pixel_values = torch.stack([sample["pixel_values"] for sample in samples]).contiguous()
        videoid = [sample["videoid"] for sample in samples]
        valid = [sample["valid"] for sample in samples]
        batch = {"pixel_values": pixel_values, "videoid": videoid, "valid": valid}
        return batch

    def __iter__(self):
        # split dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.source)
        else:
            worker_id = int(worker_info.id)
            per_worker = len(self.source) // int(worker_info.num_workers)
            per_worker += int(worker_id < len(self.source) % int(worker_info.num_workers))
            if worker_id >= len(self.source) % int(worker_info.num_workers):
                iter_start = worker_id * per_worker + len(self.source) % int(worker_info.num_workers)  
            else:          
                iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
       
       # bucketing
        for i in range(iter_start, iter_end):
            sample = self.source[i]
            if sample["valid"] is False:
                continue
            T, C, H, W = sample["pixel_values"].shape
            if (T, H, W) not in self.buffer:
                self.buffer[(T, H, W)] = []
            self.buffer[(T, H, W)].append(sample)
            self.size += 1
            if len(self.buffer[(T, H, W)]) == self.batch_size:
                yield self.buffer[(T, H, W)]
                self.size -= self.batch_size
                self.buffer[(T, H, W)] = []
            if self.size > self.max_buf and (len(self.buffer[(T, H, W)]) > 0):
                self.size -= len(self.buffer[(T, H, W)])
                yield self.buffer[(T, H, W)]
                self.buffer[(T, H, W)] = []
        # yield the remaining batch
        for bucket, samples in self.buffer.items():
            if len(samples) > 0:
                yield samples

class VideoDataset(Dataset):
    def __init__(
        self,
        meta_files: List[str],
        latent_cache_dir: str,
        sample_size: Tuple[int, int],
        sample_n_frames: int,
        is_center_crop: bool = True,
        enable_multi_aspect_ratio: bool = False,
        vae_time_compression_ratio: int = 4,
        use_stride: bool = False,
    ):
        if not Path(latent_cache_dir).exists():
            Path(latent_cache_dir).mkdir(parents=True, exist_ok=True)
        self.latent_cache_dir = latent_cache_dir

        self.sample_n_frames = sample_n_frames
        self.sample_size = tuple(sample_size)
        self.is_center_crop = is_center_crop
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.enable_multi_aspect_ratio = enable_multi_aspect_ratio
        self.dataset = meta_files
        self.length = len(self.dataset)
        self.use_stride = use_stride

        # multi-aspect-ratio buckets
        if enable_multi_aspect_ratio:
            assert self.sample_size[0] == self.sample_size[1]
            if self.sample_size[0] < 540:
                self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0])
            else:
                self.buckets = self.generate_crop_size_list(base_size=self.sample_size[0], patch_size=32)
            self.aspect_ratios = np.array([float(w) / float(h) for w, h in self.buckets])
            print(f"Multi-aspect-ratio bucket num: {len(self.buckets)}")
        # image preprocess
        if not enable_multi_aspect_ratio:
            self.train_crop = transforms.CenterCrop(self.sample_size) if self.is_center_crop else transforms.RandomCrop(self.sample_size)

    def request_ceph_data(self, path):
        try:
            video_reader = VideoReader(path)
        except Exception as e:
            print(f"Error: {e}")
            raise
        return video_reader

    def preprocess_url(self, data_json_path):

        with open(data_json_path, "r") as f:
            data_dict = json.load(f)

        video_path = data_dict['video_path']
        video_id = video_path.split('/')[-1].split('.')[0]
        prompt = data_dict['raw_caption']["long caption"]

        item = {"video_path": video_path, "videoid": video_id, "prompt": prompt}
        return item

    def get_item(self, idx):
        # Create Video Reader
        data_json_path = self.dataset[idx]
        video_item = self.preprocess_url(data_json_path)

        # Skip if exists
        latent_save_path = Path(self.latent_cache_dir) / f"{video_item['videoid']}.npy"
        if latent_save_path.exists():
            return None, None, False

        video_reader = self.request_ceph_data(video_item["video_path"])

        fps = video_reader.get_avg_fps()

        stride = 1
        if self.use_stride:
            if int(fps) >= 50:
                stride = 2
            else:
                stride = 1
        else:
            stride = 1
            
        video_length = len(video_reader)
        #stride = video_length // self.sample_n_frames
        print(video_length,"...................",stride)
        if video_length < self.sample_n_frames*stride:
            sample_n_frames = video_length - (video_length - 1) % (self.vae_time_compression_ratio*stride)  # 4n+1/8n+1
        else:
            sample_n_frames = self.sample_n_frames*stride  

        start_idx = 0
        batch_index = list(range(start_idx, start_idx + sample_n_frames, stride))

        if len(batch_index) == 0:
            print("get video len=0, skip")
            return None, None, None, False
        # Read frames
        try:
            video_images = video_reader.get_batch(batch_index).asnumpy()
        except Exception as e:
            print(f'Error: {e}, video_path: {video_item["video_path"]}')
            raise
        pixel_values = torch.from_numpy(video_images).permute(0, 3, 1, 2).contiguous()
        del video_reader

        return pixel_values, video_item["videoid"], video_item["video_path"], video_item["prompt"], True

    def preprocess_train(self, frames):
        height, width = frames.shape[-2:]
        # Resize & Crop
        if self.enable_multi_aspect_ratio:
            bw, bh = self.get_closest_ratio(width=width, height=height, ratios=self.aspect_ratios, buckets=self.buckets)
            sample_size = bh, bw
            target_size = self.get_target_size(frames, sample_size)
            train_crop = transforms.CenterCrop(sample_size) if self.is_center_crop else transforms.RandomCrop(sample_size)
        else:
            sample_size = self.sample_size
            target_size = self.get_target_size(frames, sample_size)
            train_crop = self.train_crop

        frames = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(frames)
        if self.is_center_crop:
            y1 = max(0, int(round((height - sample_size[0]) / 2.0)))
            x1 = max(0, int(round((width - sample_size[1]) / 2.0)))
            frames = train_crop(frames)
        else:
            y1, x1, h, w = train_crop.get_params(frames, sample_size)
            frames = crop(frames, y1, x1, h, w)
        return frames

    @staticmethod
    def get_closest_ratio(width: float, height: float, ratios: list, buckets: list):
        aspect_ratio = float(width) / float(height)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        return buckets[closest_ratio_id]

    @staticmethod
    def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def get_target_size(self, frames, target_size):
        T, C, H, W = frames.shape
        th, tw = target_size
        r = max(th / H, tw / W)
        target_size = int(H * r), int(W * r)
        return target_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            pixel, videoid, video_path, prompt, valid = self.get_item(idx)
            if pixel is not None and valid:
                pixel = self.preprocess_train(pixel)
            sample = dict(pixel_values=pixel, videoid=videoid, video_path=video_path, prompt=prompt,valid=valid)
            return sample
        except Exception as e:
            print(e)
            return dict(pixel_values=None, videoid=None, video_path=None, prompt=None, valid=False)