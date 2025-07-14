#!/usr/bin/bash
srun -J job --partition eaigc1 --gres gpu:8 --nodes 1 --ntasks-per-node 1 --cpus-per-task 128 --quotatype reserved -o slurm_output/slurm-%j.out -e slurm_output/slurm-%j.err scripts/distill/distill_hunyuan_model.sh
