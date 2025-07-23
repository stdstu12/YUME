#!/usr/bin/bash
srun -J job --partition eaigc1_t --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 24 --quotatype reserved -o slurm_output/slurm-%j.out -e slurm_output/slurm-%j.err scripts/inference/sample_tts.sh
