import json
import math
import os
import random
from collections import Counter
from os.path import join as opj

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from fastvideo.utils.dataset_utils import DecordInit
from fastvideo.utils.logging_ import main_print

from fastvideo.dataset.transform import (CenterCropResizeVideo, Normalize255,
                                         TemporalRandomCrop)
import pandas as pd
import pickle
from decord import VideoReader, cpu
from torchvision import transforms
import glob
from scipy.spatial.transform import Rotation

import cv2
import numpy as np
from PIL import Image

import subprocess
import tempfile
import time
from pathlib import Path

def save_pil_images_to_mp4(vid_pil_image_list, output_path, fps=30):

    if not vid_pil_image_list:
        print("Error")
        return
    

    width, height = vid_pil_image_list[0].size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for pil_img in vid_pil_image_list:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(cv_img)
    
    out.release()
    print(f"Saved: {output_path}")

class T2V_dataset(Dataset):
    def __init__(
            self,
            sample_rate,
            n_sample_frames,
            width,
            height,
            training=True,
            cache_file="vid_meta_cache.pkl",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames


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
    pos_angle_frames = [i for i in range(len(traj_pos_coord_full))] # Note: displacement angles correspond to position changes
    rot_angle_frames = [i for i in range(len(traj_rot_angle_full))]
    
    # Filter metrics within specified range
    traj_pos_coord_filtered = [
        vec for i, vec in enumerate(traj_pos_coord_full)
        if start_frame <= i < end_frame - 2 * stride  # Note: position change requires 3 consecutive points
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



class StableVideoAnimationDataset(Dataset):
    def __init__(
            self,
            sample_rate,
            n_sample_frames,
            width,
            height,
            root_dir="./mp4_frame",
            full_mp4="./sekai/",
            training=True,
            cache_file="vid_meta_cache.pkl",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.img_size = (height, width)
        self.height = height
        self.width = width
        self.root_dir = root_dir
        self.full_mp4 = full_mp4
        self.cache_file = cache_file
        resize = [
            CenterCropResizeVideo((height, width)),
        ]

        self.transform = transforms.Compose([
            *resize,
        ])

        self.vid_meta = []


        MAX_FILES_PER_CATEGORY = 4000 

        cnt = 0
        for subdir in glob.glob(os.path.join(root_dir, '*/')):
            mp4_files = glob.glob(os.path.join(subdir, '*.mp4'))
            if len(mp4_files) > MAX_FILES_PER_CATEGORY:
                mp4_files = random.sample(mp4_files, MAX_FILES_PER_CATEGORY)
                print(f"Sample {MAX_FILES_PER_CATEGORY}/{len(mp4_files)} in {subdir}")
            else:
                print(f"Use {len(mp4_files)} in {subdir}")
            
            for mp4_path in mp4_files:
                base_name = os.path.splitext(os.path.basename(mp4_path))[0]
                txt_path = os.path.join(subdir, f"{base_name}.txt")
                npz_path = os.path.join(subdir, f"{base_name}.npy")

                prefix = base_name.split('_frames_')[0]
                parts = prefix.split('_')
                npz_dir_name = '_'.join(parts[:-2])
                full_mp4_1 = os.path.join(full_mp4+npz_dir_name+"/", f"{base_name.split('_frames_')[0]}.mp4")

                if os.path.exists(txt_path):
                    keys, mouse = parse_txt_file(txt_path)
                    if keys is not None and mouse is not None:
                        video_id = base_name.split('_frames_')[0]
                        start_frame, end_frame = parse_txt_frame(txt_path)
                        start_frame = int(start_frame)
                        end_frame = int(end_frame)
                        self.vid_meta.append((mp4_path, video_id, [keys], [mouse], npz_path, start_frame, end_frame, False, full_mp4_1))
                    else:
                        print(f"Warning: Missing Keys/Mouse in {txt_path}")
                else:
                    print(f"Warning: Missing TXT file for {mp4_path}")

        random.shuffle(self.vid_meta)
        self.length = len(self.vid_meta)

        print(f"We have {self.length} videos")

        self.simple_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size),
        ])


    def get_sample(self, index):
        video_path, videoid, keys, mouse, npz_path, start_frame, end_frame, flag, full_mp4_1 = self.vid_meta[index]

        caption = "This video depicts a city walk scene with a first-person view (FPV)."

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        total_frames_target = min(self.n_sample_frames, video_length) 

        
        target_times = np.arange(total_frames_target)  
        original_indices = target_times
        start_idx = random.randint(0, video_length - original_indices[-1] - 1 )
        batch_index = [idx + start_idx for idx in original_indices]

        vid_pil_image_list = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index]
        pixel_values_vid = torch.stack([self.simple_transform(img) for img in vid_pil_image_list], dim=0)

        
        if full_mp4_1 != None and start_frame + batch_index[0] > 0:
            rand_num_img = random.random()
            if rand_num_img<0.4:
                len_cat = 400
            else:
                len_cat = 1000
            if start_frame + batch_index[0] > 10:
                pack_frame = random.randint(10, min(len_cat, start_frame + batch_index[0]))
            else:
                pack_frame = random.randint(0, start_frame + batch_index[0])

            if pack_frame > batch_index[0]:
                pack_frame_start = start_frame - (pack_frame - batch_index[0])
                target_times = np.arange(pack_frame_start, start_frame + batch_index[0])
                try:
                    video_reader = VideoReader(full_mp4_1)
                    batch_index_i = [idx for idx in target_times]
                    vid_pil_image_list_pre = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index_i]
                    vid_pil_image_list = vid_pil_image_list_pre + vid_pil_image_list
                except Exception as e:
                    print(f"Error: {str(e)}, Skip: {full_mp4_1}")    
            else:
                pack_frame_start = start_frame + (pack_frame - batch_index[0])
                target_times = np.arange(batch_index[0] - pack_frame , start_frame + batch_index[0])
                try:
                    video_reader = VideoReader(full_mp4_1)
                    batch_index = [idx for idx in target_times]
                    vid_pil_image_list_pre = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index]
                    vid_pil_image_list = vid_pil_image_list_pre + vid_pil_image_list                 
                except Exception as e:
                    print(f"Error: {str(e)}, Skip: {full_mp4_1}")    
            pixel_values_vid = torch.stack([self.simple_transform(img) for img in vid_pil_image_list], dim=0)

        if len(vid_pil_image_list) < 33:
            error_msg = f"{len(vid_pil_image_list)}, Skip: {full_mp4_1}"
            print(error_msg)
            raise ValueError(error_msg) 

        ref_img_pil = Image.fromarray(video_reader[batch_index[0]].asnumpy())
        pixel_values_ref_img = self.simple_transform(ref_img_pil)



        vocab = {
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
        caption = caption + vocab[keys[0]]

        vocab = {
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
        caption = caption + vocab[mouse[0]]

        rand_num_img = random.random()  # i2v or v2v
        if npz_path!=None and rand_num_img > 0.35:
            if flag:
                data = np.load(npz_path)["extrinsic"]
                avg_speed, avg_traj_angle, avg_rot_angle = calculate_metrics_in_range(data, batch_index[0]+start_frame, batch_index[-1]+start_frame, stride=1, fps=30)
            else:
                data = np.load(npz_path)
                avg_speed, avg_traj_angle, avg_rot_angle = calculate_metrics_in_range(data, batch_index[0], batch_index[-1], stride=1, fps=30)

            avg_speed, avg_traj_angle, avg_rot_angle = calculate_metrics_in_range(data, batch_index[0], batch_index[-1], stride=1, fps=30)
            caption = caption + "Actual distance moved:"+str(avg_speed*100)+ " at 100 meters per second."+\
                "Angular change rate (turn speed):"+str(avg_traj_angle)+"."+\
                "View rotation speed:"+str(avg_rot_angle)+"."

        sample = dict(
            pixel_values=pixel_values_vid,
            pixel_values_ref_img=pixel_values_ref_img,
            caption=caption,  
            keys=keys, 
            mouse=mouse,
            videoid=videoid,
        )

        return sample

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.get_sample(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)
                print("__getitem__[Error]", str(e), " reroll:", idx)



        pixel_values = sample['pixel_values']
        pixel_values_ref_img = sample['pixel_values_ref_img']

        pixel_values = pixel_values.sub_(0.5).div_(0.5) 



        pixel_values_ref_img = pixel_values[0,:,:,:]


        sample['pixel_values'] = pixel_values
        sample['pixel_values_ref_img'] = pixel_values_ref_img

        return sample['pixel_values'],sample['pixel_values_ref_img'], sample['caption'] ,sample['keys'],sample['mouse'],sample['videoid']

    def __len__(self):
        return self.length
