#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node 1 --master_port 29709 \
    fastvideo/sample/sample.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs" \
    --test_data_dir="./val" \
    --num_euler_timesteps 50 \
    --rand_num_img 0.6 \
    --video_root_dir "./test_video" \
    --t5_cpu