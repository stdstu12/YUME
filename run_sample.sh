#!/usr/bin/bash
srun -J job --partition Gveval-T --gres gpu:4 --nodes 1 --ntasks-per-node 1 --cpus-per-task 26 --quotatype reserved -o slurm_output/slurm-%j.out -e slurm_output/slurm-%j.err scripts/inference/sample_image.sh
