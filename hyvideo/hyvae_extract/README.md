
[中文阅读](./README_zh.md)

# HunyuanVideo Latent Feature Extraction Tool

This project provides an efficient tool for extracting latent features from videos, preparing them for subsequent video generation and processing tasks.

## Features

- Support for various video formats and resolutions
- Multi-GPU parallel processing for improved efficiency
- Support for multiple aspect ratios
- High-performance VAE model for feature extraction
- Automatic skipping of already processed videos, supporting resume functionality

## Usage

### 1. Configuration File

## Input dataset Format

The input video metadata file (meta_file.list) should be a list of JSON file paths, with each JSON file containing the following fields:

The format of meta_file.list (e.g., ./assets/demo/i2v_lora/train_dataset/meta_file.list) is as follows
```
/path/to/0.json
/path/to/1.json
/path/to/2.json
...
```

The format of /path/to/0.json (e.g., ./assets/demo/i2v_lora/train_dataset/meta_data.json) is as follows
```json
{
  "video_path": "/path/to/video.mp4",
  "raw_caption": {
    "long caption": "Detailed description text of the video"
  }
}
```

Configure parameters in `hyvideo/hyvae_extract/vae.yaml`:

```yaml
vae_path: "./ckpts/hunyuan-video-i2v-720p/vae" # VAE model path
video_url_files: "/path/to/meta_file.list"     # Video metadata file list
output_base_dir: "/path/to/output/directory"   # Output directory
sample_n_frames: 129                           # Number of frames to sample
target_size:                                   # Target size
  - bucket_size
  - bucket_size
enable_multi_aspect_ratio: True                # Enable multiple aspect ratios
use_stride: True                               # Use stride sampling
```

#### Bucket Size Reference

The `target_size` parameter defines the resolution bucket size. Here are the recommended values for different quality levels:

| Quality | Bucket Size | Typical Resolution |
|---------|-------------|-------------------|
| 720p    | 960         | 1280×720 or similar |
| 540p    | 720         | 960×540 or similar |
| 360p    | 480         | 640×360 or similar |

When `enable_multi_aspect_ratio` is set to `True`, the system will use these bucket sizes as a base to generate multiple aspect ratio buckets. For optimal performance, choose a bucket size that balances quality and memory usage based on your hardware capabilities.

### 2. Run Extraction

```bash
# Set environment variables
export HOST_GPU_NUM=8  # Set the number of GPUs to use

# Run extraction script
cd HunyuanVideo-I2V
bash hyvideo/hyvae_extract/start.sh
```

### 3. Single GPU Run

```bash
cd HunyuanVideo-I2V
export PYTHONPATH=${PYTHONPATH}:`pwd`
export HOST_GPU_NUM=1
CUDA_VISIBLE_DEVICES=0 python3 -u hyvideo/hyvae_extract/run.py --local_rank 0 --config 'hyvideo/hyvae_extract/vae.yaml'
```

## Output Files

The program generates the following files in the specified output directory:

1. `{video_id}.npy` - Latent feature array of the video
2. `json_path/{video_id}.json` - JSON file containing video metadata, including:
   - video_id: Video ID
   - latent_shape: Shape of the latent features
   - video_path: Original video path
   - prompt: Video description/prompt
   - npy_save_path: Path where the latent features are saved

```
output_base_dir/
│
├── {video_id_1}.npy # Latent feature array for video 1
├── {video_id_2}.npy # Latent feature array for video 2
├── {video_id_3}.npy # Latent feature array for video 3
│ ...
├── {video_id_n}.npy # Latent feature array for video n
│
└── json_path/ # Directory containing metadata JSON files
│     ├── {video_id_1}.json # Metadata for video 1
│     ├── {video_id_2}.json # Metadata for video 2
│     ├── {video_id_3}.json # Metadata for video 3
│     │ ...
│     └── {video_id_n}.json # Metadata for video n
```

## Advanced Configuration

### Multiple Aspect Ratio Processing

When `enable_multi_aspect_ratio` is set to `True`, the system selects the target size closest to the original aspect ratio of the video, rather than forcing it to be cropped to a fixed size. This is useful for maintaining the integrity of the video content.

### Stride Sampling

When `use_stride` is set to `True`, the system automatically adjusts the sampling stride based on the video's frame rate:
- When frame rate >= 50fps, stride is 2
- When frame rate < 50fps, stride is 1