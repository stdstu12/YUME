import argparse
from pathlib import Path


def get_tensorboard_config(output_dir: str, job_name: str):
    tensorboard_config = {
        "enabled": True,
        "output_path": output_dir,
        "job_name": job_name
    }
    return tensorboard_config


def get_deepspeed_config(args: argparse.Namespace,
                         micro_batch_size: int,
                         global_batch_size: int,
                         output_dir: str = None,
                         job_name: str = None,
                         ):
    config = {
        "zero_allow_untested_optimizer": True,
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.log_every,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [
                    args.adam_beta1,
                    args.adam_beta2
                ],
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": True,
        "fp16": {
            "enabled": args.precision == 'fp16',
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15
        },
        "bf16": {
            "enabled": args.precision == 'bf16'
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.zero_stage,
            "reduce_scatter": False,
            "reduce_bucket_size": 1e9,
        },
    }

    if args.tensorboard:
        config["tensorboard"] = get_tensorboard_config(output_dir, job_name)

    return config
