#!/usr/bin/bash	

# DATA_DIR=./data
# IP=[MASTER NODE IP]
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node 8 --master_port 29607 \
    fastvideo/distill_model.py \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=600000 \
    --learning_rate=1e-5 \
    --discriminator_learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=25 \
    --validation_steps 24 \
    --allow_tf32 \
    --MVDT \
    --Distil \
    --output_dir="./outputs"
    #--resume_from_checkpoint "./outputs/checkpoint-350/"
