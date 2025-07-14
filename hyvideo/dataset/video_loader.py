import random
import os
import io
import torch
import numpy as np
import json
import traceback
import time
import pyarrow as pa

from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self,
                 data_jsons_path: str,
                 sample_n_frames: int = 129,
                 sample_stride: int = 1,
                 text_encoder=None,
                 text_encoder_2=None,
                 uncond_p=0.0,
                 args=None,
                 logger=None,
                 ) -> None:
        """_summary_

        Args:
            data_jsons_path (str): input data json path
            sample_n_frames (int, optional): training video length. Defaults to 129.
            sample_stride (int, optional): video frame sample stride. Defaults to 1 (No strid).
            text_encoder (_type_, optional): text encoder to tokenize. Defaults to None.
            text_encoder_2 (_type_, optional): second text encoder to tokenize. Defaults to None.
            uncond_p (float, optional): text uncondition prod. Defaults to 0.0.
            args (_type_, optional): args. Defaults to None.
            logger (_type_, optional): logger. Defaults to None.
        """
        self.args = args
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.uncond_p = uncond_p

        if logger is None:
            from loguru import logger
        self.logger = logger

        json_files = os.listdir(data_jsons_path)

        video_id_list = []
        latent_shape_list = []
        prompt_list = []
        npy_save_path_list = []
        height_list = []
        width_list = []
        for json_file in json_files:
            with open(f"{data_jsons_path}/{json_file}", 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
            video_id = data.get('video_id')
            latent_shape = data.get('latent_shape')
            prompt = data.get('prompt')
            npy_save_path = data.get('npy_save_path')

            video_id_list.append(video_id)
            latent_shape_list.append(latent_shape)
            prompt_list.append(prompt)
            npy_save_path_list.append(npy_save_path)
            height_list.append(latent_shape[3])
            width_list.append(latent_shape[4])

        schema = pa.schema([
            ('video_id', pa.string()),
            ('latent_shape', pa.list_(pa.int64())),
            ('prompt', pa.string()),
            ('npy_save_path', pa.string()),
            ('height', pa.int64()),
            ('width', pa.int64()),
        ])

        video_id_array = pa.array(video_id_list, type=pa.string())
        latent_shape_array = pa.array(latent_shape_list, type=pa.list_(pa.int64()))
        prompt_array = pa.array(prompt_list, type=pa.string())
        npy_save_path_array = pa.array(npy_save_path_list, type=pa.string())
        height_array = pa.array(height_list, type=pa.int64())
        width_array = pa.array(width_list, type=pa.int64())

        record_batch = pa.RecordBatch.from_arrays([video_id_array, latent_shape_array, prompt_array,
                                                   npy_save_path_array, height_array, width_array], schema=schema)
        self.table = pa.Table.from_batches([record_batch])

        s_time = time.time()
        logger.info(f"load {data_jsons_path} \t cost {time.time() - s_time} s \t total length {len(self.table)}")

    def __len__(self):
        return len(self.table)

    def get_data_info(self, index):

        latent_shape = self.table['latent_shape'][index].as_py()
        assert isinstance(latent_shape, list), "latent_shape must be list"
        num_frames = latent_shape[-3]
        height = latent_shape[-2]
        width = latent_shape[-1]
        num_frames = (num_frames - 1) * 4 + 1

        return {'height': height,
                'width': width,
                'num_frames': num_frames}

    @staticmethod
    def get_text_tokens(text_encoder, description):
        text_inputs = text_encoder.text2tokens(description, data_type='video')
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask

    def get_batch(self, idx):
        videoid = self.table['video_id'][idx].as_py()
        prompt = self.table['prompt'][idx].as_py()
        pixel_values = torch.tensor(0)

        if random.random() < self.uncond_p:
            prompt = ''

        text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt)
        sample_n_frames = self.sample_n_frames

        cache_path = self.table['npy_save_path'][idx].as_py()
        latents = torch.from_numpy(np.load(cache_path)).squeeze(0)
        sample_n_latent = (sample_n_frames - 1) // 4 + 1
        start_idx = 0
        latents = latents[:, start_idx:start_idx + sample_n_latent, ...]

        if latents.shape[1] < sample_n_latent:
            raise Exception(
                f' videoid: {videoid} has wrong cache data for temporal buckets of shape {latents.shape}, expected length: {sample_n_latent}')

        data_info = self.get_data_info(idx)
        num_frames, height, width = data_info['num_frames'], data_info['height'], data_info['width']
        kwargs = {
            "text": prompt,
            "index": idx,
            "type": 'video',
            'bucket': [num_frames, height, width],
            "videoid": videoid
        }
        if self.text_encoder_2 is None:
            return (
                pixel_values,
                latents,
                text_ids.clone(),
                text_mask.clone(),
                {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
            )
        else:
            text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt)
            return (
                pixel_values,
                latents,
                text_ids.clone(),
                text_mask.clone(),
                text_ids_2.clone(),
                text_mask_2.clone(),
                {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
            )

    def __getitem__(self, idx):
        try_times = 100
        for i in range(try_times):
            try:
                return self.get_batch(idx)
            except Exception as e:
                self.logger.warning(
                    f"Error details: {str(e)}-{self.table['video_id'][idx]}-{traceback.format_exc()}\n")
                idx = np.random.randint(len(self))

        raise RuntimeError('Too many bad data.')

if __name__ == "__main__":

    data_jsons_path = "test_path"
    dataset = VideoDataset(args=None,
                                      data_jsons_path=data_jsons_path)

    print(dataset.__getitem__(0))
