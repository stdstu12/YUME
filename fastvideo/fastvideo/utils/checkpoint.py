# import
import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.default_planner import (DefaultLoadPlanner,
                                                          DefaultSavePlanner)
from torch.distributed.checkpoint.optimizer import \
    load_sharded_optimizer_state_dict
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fastvideo.utils.logging_ import main_print


def save_checkpoint_optimizer(model,
                              optimizer,
                              rank,
                              output_dir,
                              step,
                              discriminator=False):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(
            model,
            optimizer,
        )

    # todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0 and not discriminator:
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(model.config)
        config_dict.pop('dtype')
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state, optimizer_path)
    else:
        weight_path = os.path.join(save_dir,
                                   "discriminator_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        optimizer_path = os.path.join(save_dir, "discriminator_optimizer.pt")
        torch.save(optim_state, optimizer_path)
    main_print(f"--> checkpoint saved at step {step}")


def save_checkpoint(transformer, rank, output_dir, step):
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
    # todo move to get_state_dict
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(transformer.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        del cpu_state
    main_print(f"--> checkpoint saved at step {step}")


def save_checkpoint_generator_discriminator(
    model,
    optimizer,
    discriminator,
    discriminator_optimizer,
    rank,
    output_dir,
    step,
):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.state_dict()

    # todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    hf_weight_dir = os.path.join(save_dir, "hf_weights")
    os.makedirs(hf_weight_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0:
        config_dict = dict(model.config)
        config_path = os.path.join(hf_weight_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        weight_path = os.path.join(hf_weight_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

    main_print(f"--> saved HF weight checkpoint at path {hf_weight_dir}")
    model_weight_dir = os.path.join(save_dir, "model_weights_state")
    os.makedirs(model_weight_dir, exist_ok=True)
    model_optimizer_dir = os.path.join(save_dir, "model_optimizer_state")
    os.makedirs(model_optimizer_dir, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        model_state = model.state_dict()
        weight_state_dict = {"model": model_state}
        dist_cp.save_state_dict(
            state_dict=weight_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_weight_dir),
            planner=DefaultSavePlanner(),
        )
        optimizer_state_dict = {"optimizer": optim_state}
        dist_cp.save_state_dict(
            state_dict=optimizer_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_optimizer_dir),
            planner=DefaultSavePlanner(),
        )

    discriminator_fsdp_state_dir = os.path.join(save_dir,
                                                "discriminator_fsdp_state")
    os.makedirs(discriminator_fsdp_state_dir, exist_ok=True)
    with FSDP.state_dict_type(
            discriminator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        optim_state = FSDP.optim_state_dict(discriminator,
                                            discriminator_optimizer)
        model_state = discriminator.state_dict()
        state_dict = {"optimizer": optim_state, "model": model_state}
        if rank <= 0:
            discriminator_fsdp_state_fil = os.path.join(
                discriminator_fsdp_state_dir, "discriminator_state.pt")
            torch.save(state_dict, discriminator_fsdp_state_fil)

    main_print("--> saved FSDP state checkpoint")


def load_sharded_model(model, optimizer, model_dir, optimizer_dir):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        weight_state_dict = {"model": model.state_dict()}

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=weight_state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=dist_cp.FileSystemReader(optimizer_dir),
        )
        optim_state = optim_state["optimizer"]
        flattened_osd = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(flattened_osd)
        dist_cp.load_state_dict(
            state_dict=weight_state_dict,
            storage_reader=dist_cp.FileSystemReader(model_dir),
            planner=DefaultLoadPlanner(),
        )
        model_state = weight_state_dict["model"]
        model.load_state_dict(model_state)
    main_print(f"--> loaded model and optimizer from path {model_dir}")
    return model, optimizer


def load_full_state_model(model, optimizer, checkpoint_file, rank):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        discriminator_state = torch.load(checkpoint_file)
        model_state = discriminator_state["model"]
        if rank <= 0:
            optim_state = discriminator_state["optimizer"]
        else:
            optim_state = None
        model.load_state_dict(model_state)
        discriminator_optim_state = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(discriminator_optim_state)
    main_print(
        f"--> loaded discriminator and discriminator optimizer from path {checkpoint_file}"
    )
    return model, optimizer


def resume_training_generator_discriminator(model, optimizer, discriminator,
                                            discriminator_optimizer,
                                            checkpoint_dir, rank):
    step = int(checkpoint_dir.split("-")[-1])
    model_weight_dir = os.path.join(checkpoint_dir, "model_weights_state")
    model_optimizer_dir = os.path.join(checkpoint_dir, "model_optimizer_state")
    model, optimizer = load_sharded_model(model, optimizer, model_weight_dir,
                                          model_optimizer_dir)
    discriminator_ckpt_file = os.path.join(checkpoint_dir,
                                           "discriminator_fsdp_state",
                                           "discriminator_state.pt")
    discriminator, discriminator_optimizer = load_full_state_model(
        discriminator, discriminator_optimizer, discriminator_ckpt_file, rank)
    return model, optimizer, discriminator, discriminator_optimizer, step


def resume_training(model, optimizer, checkpoint_dir, discriminator=False):
    weight_path = os.path.join(checkpoint_dir,
                               "diffusion_pytorch_model.safetensors")
    if discriminator:
        weight_path = os.path.join(checkpoint_dir,
                                   "discriminator_pytorch_model.safetensors")
    model_weights = load_file(weight_path)

    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        current_state = model.state_dict()
        current_state.update(model_weights)
        model.load_state_dict(current_state, strict=False)
        del current_state
    if discriminator:
        optim_path = os.path.join(checkpoint_dir, "discriminator_optimizer.pt")
    else:
        optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
    optimizer_state_dict = torch.load(optim_path, weights_only=False)
    optim_state = FSDP.optim_state_dict_to_load(
        model=model, optim=optimizer, optim_state_dict=optimizer_state_dict)
    optimizer.load_state_dict(optim_state)
    step = int(checkpoint_dir.split("-")[-1])
    return model, optimizer, step

import torch
import gc 

def resume_checkpoint(model, checkpoint_dir):
    weight_path = os.path.join(checkpoint_dir,
                               "diffusion_pytorch_model.safetensors")

    model_weights = load_file(weight_path)


    current_state = model.state_dict()
    current_state.update(model_weights)
    model.load_state_dict(current_state, strict=False)
    del current_state
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    step = int(checkpoint_dir.split("-")[-1])
    return model, step

import os
import json
import gc
import torch
from safetensors import safe_open
from safetensors.torch import load_file

def resume_checkpoint_yume(model, checkpoint_dir, merge=True):
    device = next(model.parameters()).device

    index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        try:

            with open(index_path, "r") as f:
                index_data = json.load(f)
            
            unique_shards = set(index_data["weight_map"].values())
            model_weights = {}

            for shard_name in unique_shards:
                shard_path = os.path.join(checkpoint_dir, shard_name)

                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Missing shard file: {shard_path}")

                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        model_weights[key] = f.get_tensor(key).to(device)
                        torch.cuda.empty_cache()
                        gc.collect()
        except Exception as e:
            raise RuntimeError(f"Failed to load sharded model: {str(e)}")
    else:
        full_model_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"No model files found in: {checkpoint_dir}")
        try:
            model_weights = load_file(full_model_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load full model: {str(e)}")


    # # 打印权重key和形状
    # print("\n=== Model Weights Summary ===")
    # print(f"Total parameters: {len(model_weights)}")
    # for i, (key, weight) in enumerate(model_weights.items()):
    #     print(f"{i+1}. {key}: {tuple(weight.shape)}")


    # # 加载Yume和Fusionx模型的权重
    # yume_weights = model_weights

    # fusionx_weights = {}
    # with safe_open("/mnt/petrelfs/maoxiaofeng/Yume_v1_release/wan2114BFusionx_fusionxImage2video.safetensors", framework="pt") as f:
    #     for k in f.keys():
    #         # Remove 'model.diffusion_model.' prefix
    #         simplified_key = k.replace('model.diffusion_model.', '')
    #         fusionx_weights[simplified_key] = f.get_tensor(k).float().to(torch.bfloat16)  # FP8→BF16 conversion

    # # 打印权重key和形状
    # print("\n=== Model Weights Summary ===")
    # print(f"Total parameters: {len(fusionx_weights)}")
    # for i, (key, weight) in enumerate(fusionx_weights.items()):
    #     print(f"{i+1}. {key}: {tuple(weight.shape)}","-----------------------------------")


    # merged_weights = {}
    # for key in yume_weights:
    #     # 修改后（自动跳过缺失的键）：
    #     if key in yume_weights and key in fusionx_weights:  # 检查键是否存在
    #         print(key,"keykey")
    #         merged_weights[key] = (yume_weights[key] * 0.5 + fusionx_weights[key] * 0.5).to(torch.bfloat16)
    #     else:
    #         merged_weights[key] = yume_weights[key].to(torch.bfloat16)

    # model_weights = merged_weights
    
    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
        print("Successfully loaded")
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys[:5]}... (total {len(missing_keys)})")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}... (total {len(unexpected_keys)})")
    
    except Exception as e:
        raise RuntimeError(f"Failed to update model weights: {str(e)}")
    
    finally:
        del model_weights
        torch.cuda.empty_cache()
        gc.collect()
    
    return model


def save_lora_checkpoint(transformer, optimizer, rank, output_dir, step,
                         pipeline):
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()
        lora_optim_state = FSDP.optim_state_dict(
            transformer,
            optimizer,
        )

    if rank <= 0:
        save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # save optimizer
        optim_path = os.path.join(save_dir, "lora_optimizer.pt")
        torch.save(lora_optim_state, optim_path)
        # save lora weight
        main_print(f"--> saving LoRA checkpoint at step {step}")
        transformer_lora_layers = get_peft_model_state_dict(
            model=transformer, state_dict=full_state_dict)
        pipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=transformer_lora_layers,
            is_main_process=True,
        )
        # save config
        lora_config = {
            "step": step,
            "lora_params": {
                "lora_rank": transformer.config.lora_rank,
                "lora_alpha": transformer.config.lora_alpha,
                "target_modules": transformer.config.lora_target_modules,
            },
        }
        config_path = os.path.join(save_dir, "lora_config.json")
        with open(config_path, "w") as f:
            json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at step {step}")


def resume_lora_optimizer(transformer, checkpoint_dir, optimizer):
    config_path = os.path.join(checkpoint_dir, "lora_config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    optim_path = os.path.join(checkpoint_dir, "lora_optimizer.pt")
    optimizer_state_dict = torch.load(optim_path, weights_only=False)
    optim_state = FSDP.optim_state_dict_to_load(
        model=transformer,
        optim=optimizer,
        optim_state_dict=optimizer_state_dict)
    optimizer.load_state_dict(optim_state)
    step = config_dict["step"]
    main_print(f"-->  Successfully resuming LoRA optimizer from step {step}")
    return transformer, optimizer, step
