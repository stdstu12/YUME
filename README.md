<div align="center">
<img src=assets/yume.png width="20%"/>
</div>

Yume is a long-term project that aims to create an interactive, realistic, and dynamic world through the input of text, images, or videos.



<p align="center">
    🤗 <a href="https://huggingface.co/stdstu123/Yume-I2V-540P"  target="_blank">YUME</a>  | 📜 <a href="https://github.com/stdstu12/YUME-World" target="_blank">Paper</a> 
</p> 

[![project page](https://img.shields.io/badge/Project-Page-2ea44f)](https://stdstu12.github.io/YUME-Project/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.15675-b31b1b.svg)](https://arxiv.org/abs/2506.15675)&nbsp;
[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/stdstu123/Yume-I2V-540P)&nbsp;
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=X6fFzsLp_3Q)&nbsp;

- A distillation recipes for video DiT.
- [FramePack-based](https://github.com/lllyasviel/FramePack) training code.
- Long video generation method with DDP sampling support

## 🎥 Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/X6fFzsLp_3Q" frameborder="0" allowfullscreen></iframe>


## 🔧 Installation
The code is tested on Python 3.10.0, CUDA 12.1 and A100.
```
./env_setup.sh fastvideo
pip install -r requirements.txt
```
You need to run `pip install .` after each code modification, or alternatively, you can copy the modified files directly into your virtual environment. For example, if I modified `wan/image2video.py` and my virtual environment is `yume`, I can copy the file to:
`envs/yume/lib/python3.10/site-packages/wan/image2video.py`.

## 🚀 Inference

### ODE
For image-to-video generation, we use `--jpg_dir="./jpg"` to specify the input image directory and `--caption_path="./caption.txt"` to provide text conditioning inputs, where each line corresponds to a generation instance controlling 2-second video output.
```bash
# Download the model weights and place them in Path_To_Yume.
bash scripts/inference/sample_jpg.sh 
```
We also consider generating videos using the data from `./val`, where `--test_data_dir="./val"` specifies the location of the example data.
```bash
# Download the model weights and place them in Path_To_Yume.
bash scripts/inference/sample.sh 
```
We perform TTS sampling, where `args.sde` controls whether to use SDE-based sampling.
```bash
# Download the model weights and place them in Path_To_Yume.
bash scripts/inference/sample_tts.sh 
```

For optimal results, we recommend keeping Actual distance, Angular change rate (turn speed), and View rotation speed within the range of 0.1 to 10. 

Key adjustment guidelines:
1. When executing Camera remains still (·), reduce the Actual distance value
2. When executing Person stands still, decrease both Angular change rate and View rotation speed values

Note that these parameters (Actual distance, Angular change rate, and View rotation speed) do impact generation results. As an alternative approach, you may consider removing these parameters entirely for simplified operation.



## 🎯 Training & Distill 
For model training, we use `args.MVDT` to launch the MVDT framework, which requires at least 16 A100 GPUs. Loading T5 onto the CPU may help conserve GPU memory. We employ `args.Distil` to enable adversarial distillation.
```bash
# Download the model weights and place them in Path_To_Yume.
bash scripts/finetune/finetune.sh
```

## 🧱 Dataset Preparation
Please refer to https://github.com/Lixsp11/sekai-codebase to download the dataset. For the processed data format, refer to `./test_video`.
```
path_to_processed_dataset_folder/
├── Keys_None_Mouse_Down/ 
│   ├── video_id.mp4
│   ├── video_id.txt
├── Keys_None_Mouse_Up
│──  ...
└── Keys_S_Mouse_·
```
The provided TXT file content record either camera motion control parameters or animation keyframe data, with the following field definitions:
```
Start Frame: 2 #Starting frame number (begins at frame 2 at origin video)

End Frame: 50 #Ending frame number

Duration: 49 frames #Total duration

Keys: W #Keyboard input

Mouse: ↓ #Mouse action
```
In `scripts/finetune/finetune.sh`, `args.root_dir` represents the `path_to_processed_dataset_folder`, and `args.root_dir` represents the full path to the Sekai dataset.


## 📑 Development Plan
- Dataset processing
  - [ ] Providing processed datasets
- Code update
  - [ ] fp8 support
  - [ ] Better distillation methods
- ​​Model Update
  - [ ] Quantized and Distilled Models
  - [ ] Models for 720p Resolution Generation​

## 🤝 Contributing
We welcome all contributions.


## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [diffusers](https://github.com/huggingface/diffusers)
- [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Skywork-Reward-V2](https://github.com/SkyworkAI/Skywork-Reward-V2)
- [MDT](https://github.com/sail-sg/MDT)
- [AddSR](https://github.com/NJU-PCALab/AddSR)
