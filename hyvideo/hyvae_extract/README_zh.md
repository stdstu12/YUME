[English](./README.md)

# 混元视频特征提取工具

本项目提供了一个高效的工具，用于从视频中提取潜在特征，为后续的视频生成和处理任务做准备。

## 功能特点

- 支持各种视频格式和分辨率
- 多GPU并行处理，提高效率
- 支持多种宽高比
- 高性能VAE模型用于特征提取
- 自动跳过已处理的视频，支持断点续传功能

## 使用方法

### 1. 配置文件

## 输入数据集格式

输入的视频元数据文件(meta_file.list)应为JSON文件路径的列表，每个JSON文件包含以下字段：

meta_file.list的格式（例如，./assets/demo/i2v_lora/train_dataset/meta_file.list）如下：
```
/path/to/0.json
/path/to/1.json
/path/to/2.json
...
```

/path/to/0.json的格式（例如，./assets/demo/i2v_lora/train_dataset/meta_data.json）如下：
```json
{
  "video_path": "/path/to/video.mp4",
  "raw_caption": {
    "long caption": "视频的详细描述文本"
  }
}
```

在`hyvideo/hyvae_extract/vae.yaml`中配置参数：

```yaml
vae_path: "./ckpts/hunyuan-video-i2v-720p/vae" # VAE模型路径
video_url_files: "/path/to/meta_file.list"     # 视频元数据文件列表
output_base_dir: "/path/to/output/directory"   # 输出目录
sample_n_frames: 129                           # 采样帧数
target_size:                                   # 目标尺寸
  - bucket_size
  - bucket_size
enable_multi_aspect_ratio: True                # 启用多种宽高比
use_stride: True                               # 使用步长采样
```

#### 分辨率桶大小参考

`target_size`参数定义了分辨率桶大小。以下是不同质量级别的推荐值：

| 质量 | 桶大小 | 典型分辨率 |
|---------|-------------|-------------------|
| 720p    | 960         | 1280×720或类似 |
| 540p    | 720         | 960×540或类似 |
| 360p    | 480         | 640×360或类似 |

当`enable_multi_aspect_ratio`设置为`True`时，系统将使用这些桶大小作为基础来生成多种宽高比的桶。为了获得最佳性能，请根据您的硬件能力选择平衡质量和内存使用的桶大小。

### 2. 运行提取

```bash
# 设置环境变量
export HOST_GPU_NUM=8  # 设置要使用的GPU数量

# 运行提取脚本
cd HunyuanVideo-I2V
bash hyvideo/hyvae_extract/start.sh
```

### 3. 单GPU运行

```bash
cd HunyuanVideo-I2V
export PYTHONPATH=${PYTHONPATH}:`pwd`
export HOST_GPU_NUM=1
CUDA_VISIBLE_DEVICES=0 python3 -u hyvideo/hyvae_extract/run.py --local_rank 0 --config 'hyvideo/hyvae_extract/vae.yaml'
```

## 输出文件

程序在指定的输出目录中生成以下文件：

1. `{video_id}.npy` - 视频的潜在特征数组
2. `json_path/{video_id}.json` - 包含视频元数据的JSON文件，包括：
   - video_id: 视频ID
   - latent_shape: 潜在特征的形状
   - video_path: 原始视频路径
   - prompt: 视频描述/提示
   - npy_save_path: 保存潜在特征的路径

```
output_base_dir/
│
├── {video_id_1}.npy # 视频1的潜在特征数组
├── {video_id_2}.npy # 视频2的潜在特征数组
├── {video_id_3}.npy # 视频3的潜在特征数组
│ ...
├── {video_id_n}.npy # 视频n的潜在特征数组
│
└── json_path/ # 包含元数据JSON文件的目录
      ├── {video_id_1}.json # 视频1的元数据
      ├── {video_id_2}.json # 视频2的元数据
      ├── {video_id_3}.json # 视频3的元数据
      │ ...
      └── {video_id_n}.json # 视频n的元数据
```

## 高级配置

### 多宽高比处理

当`enable_multi_aspect_ratio`设置为`True`时，系统会选择最接近视频原始宽高比的目标尺寸，而不是强制将其裁剪为固定尺寸。这有助于保持视频内容的完整性。

### 步长采样

当`use_stride`设置为`True`时，系统会根据视频的帧率自动调整采样步长：
- 当帧率 >= 50fps时，步长为2
- 当帧率 < 50fps时，步长为1 