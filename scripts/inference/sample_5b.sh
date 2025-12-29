#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node 2 --master_port 29717 \
    fastvideo/sample/sample_5b.py \
    --seed 43 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_sample_steps=600000 \
    --mixed_precision="bf16" \
    --allow_tf32 \
    --video_output_dir="./outputs" \
    --caption_path="./caption_re.txt" \
    --test_data_dir="./val" \
    --num_euler_timesteps 4 \
    --rand_num_img 0.6 \
    --jpg_dir="./jpg/" \
    --prompt "A fire-breathing dragon appeared."
    #--T2V \
    #--prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage." \


    # --jpg_dir="./jpg/" \
    #--jpg_dir="./jpg/" \
    #--video_root_dir "./test_video"
