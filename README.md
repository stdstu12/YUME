
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=stdstu12&project=YUME&lang=as">অসমীয়া</
      </div>
    </div>
  </details>
</div>

<div align="center">
<img src=assets/yume.png width="20%"/>
</div>

# Yume: An Interactive World Generation Model

Yume is a long-term project that aims to create an interactive, realistic, and dynamic world through the input of text, images, or videos.


<div align="center">




[![project page](https://img.shields.io/badge/Project-Page-2ea44f)](https://stdstu12.github.io/YUME-Project/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2507.17744-b31b1b.svg)](https://arxiv.org/abs/2507.17744)&nbsp;
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/stdstu123/Yume-I2V-540P)&nbsp;
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=51VII_iJ1EM)&nbsp;

</div>

- A distillation recipes for video DiT.
- [FramePack-Like](https://github.com/lllyasviel/FramePack) training code.
- Long video generation method with DDP/FSDP sampling support



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
### SDE
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

## Citation
If you use Yume for your research, please cite our paper:

```bibtex
@article{mao2025yume,
  title={Yume: An Interactive World Generation Model},
  author={Mao, Xiaofeng and Lin, Shaoheng and Li, Zhen and Li, Chuanhao and Peng, Wenshuo and He, Tong and Pang, Jiangmiao and Chi, Mingmin and Qiao, Yu and Zhang, Kaipeng},
  journal={arXiv preprint arXiv:2507.17744},
  year={2025}
}

```
