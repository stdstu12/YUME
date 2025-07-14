<div align="center">
<img src=assets/yume.png width="20%"/>
</div>

Yume is a long-term project that aims to create an interactive, realistic, and dynamic world through the input of text, images, or videos.



<p align="center">
    🤗 <a href="https://github.com/stdstu12/YUME-World"  target="_blank">YUME</a>  | 📜 <a href="https://github.com/stdstu12/YUME-World" target="_blank">Paper</a> 
</p> 


Yume currently offers: (with more to come)


- A distillation recipes for video DiT.
- [FramePack-based](https://github.com/lllyasviel/FramePack) training code.
- Long video generation method with DDP sampling support



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

## 🎯 Training & Distill 
For model training, we use args.MVDT to launch the MVDT framework, which requires at least 16 A100 GPUs. Loading T5 onto the CPU may help conserve GPU memory. We employ args.Distil to enable adversarial distillation.
```bash
# Download the model weights and place them in Path_To_Yume.
bash scripts/finetune/finetune.sh
```

#### Dataset Preparation
Please refer to https://github.com/Lixsp11/sekai-codebase to download the dataset. For the processed data format, refer to `./test_video`.

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
