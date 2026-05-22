@echo off
setlocal

set TOKENIZERS_PARALLELISM=false
set TF_ENABLE_ONEDNN_OPTS=0
set LOCAL_RANK=0
set RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500

cd /d "%~dp0..\.."

python fastvideo/sample/sample.py ^
    --seed 42 ^
    --gradient_checkpointing ^
    --train_batch_size=1 ^
    --max_sample_steps=600000 ^
    --mixed_precision="bf16" ^
    --allow_tf32 ^
    --video_output_dir="./outputs" ^
    --jpg_dir="./jpg/" ^
    --caption_path="./caption.txt" ^
    --test_data_dir="./val" ^
    --num_euler_timesteps 50 ^
    --rand_num_img 0.6 ^
    --t5_cpu

exit /b %ERRORLEVEL%
