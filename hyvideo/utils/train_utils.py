import random
import torchvision.transforms as transforms

import numpy as np
import torch

import imageio
import os
import PIL.Image
from typing import Union, Optional, List
from peft import get_peft_model_state_dict

from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.vae import AutoencoderKLCausal3D

from pathlib import Path
from einops import rearrange
from PIL import Image

from hyvideo.constants import PRECISION_TO_TYPE
from safetensors.torch import load_file


def convert_kohya_to_peft_keys(
    kohya_dict: dict,
    kohya_prefix="",
    peft_prefix: str = "base_model.model",
    device="cpu",
) -> dict:
    peft_dict = {}
    for k, v in kohya_dict.items():
        if ".alpha" in k:
            continue
        new_key = k.replace(f"{kohya_prefix}_lora_", f"{peft_prefix}.")
        new_key = new_key.replace("single_blocks_", "single_blocks.")
        new_key = new_key.replace("double_blocks_", "double_blocks.")
        new_key = new_key.replace("_img_attn_proj", ".img_attn_proj")
        new_key = new_key.replace("_img_attn_qkv", ".img_attn_qkv")
        new_key = new_key.replace("_img_mlp_fc", ".img_mlp.fc")
        new_key = new_key.replace("_txt_mlp_fc", ".txt_mlp.fc")
        new_key = new_key.replace("_img_mod", ".img_mod")
        new_key = new_key.replace("_txt", ".txt")
        new_key = new_key.replace("_modulation", ".modulation")
        new_key = new_key.replace("_linear", ".linear")
        new_key = new_key.replace("lora_down", "lora_A.default")
        new_key = new_key.replace("lora_up", "lora_B.default")
        new_key = new_key.replace(
            "_individual_token_refiner_blocks_", ".individual_token_refiner.blocks."
        )
        new_key = new_key.replace("_mlp_fc", ".mlp.fc")

        peft_dict[new_key] = v.to(device)
    return peft_dict


def load_lora(model, lora_path, device):
    kohya_weights = load_file(lora_path)
    peft_weights = convert_kohya_to_peft_keys(
        kohya_weights, kohya_prefix="Hunyuan_video_I2V", device=device
    )
    model.load_state_dict(peft_weights, strict=False)
    return model


def black_image(width, height):
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image


def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def get_cond_latents(args, latents, vae):
    """get conditioned latent by decode and encode the first frame latents"""
    first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
    first_images = vae.decode(
        first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False
    )[0]
    first_images = first_images.squeeze(2)
    first_images = (first_images / 2 + 0.5).clamp(0, 1)
    first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
    first_images = numpy_to_pil(first_images)

    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    first_images_pixel_values = [image_transform(image) for image in first_images]
    first_images_pixel_values = (
        torch.cat(first_images_pixel_values).unsqueeze(0).unsqueeze(2).to(vae.device)
    )

    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
    with torch.autocast(
        device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32
    ):
        cond_latents = vae.encode(
            first_images_pixel_values
        ).latent_dist.sample()  # B, C, F, H, W
        cond_latents.mul_(vae.config.scaling_factor)

    return cond_latents


def get_cond_images(args, latents, vae, is_uncond=False):
    """get conditioned images by decode the first frame latents"""
    sematic_image_latents = (
        latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    )
    sematic_image_latents = 1 / vae.config.scaling_factor * sematic_image_latents
    semantic_images = vae.decode(
        sematic_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False
    )[0]
    semantic_images = semantic_images.squeeze(2)
    semantic_images = (semantic_images / 2 + 0.5).clamp(0, 1)
    semantic_images = semantic_images.cpu().permute(0, 2, 3, 1).float().numpy()
    semantic_images = numpy_to_pil(semantic_images)
    if is_uncond:
        semantic_images = [
            black_image(img.size[0], img.size[1]) for img in semantic_images
        ]

    return semantic_images


def load_state_dict(args, model, logger):
    pretrained_model_path = Path(args.model_base)
    if not pretrained_model_path.exists():
        raise ValueError(f"`models_root` not exists: {pretrained_model_path}")

    load_key = args.load_key
    if args.i2v_mode:
        dit_weight = Path(args.i2v_dit_weight)
    else:
        dit_weight = Path(args.dit_weight)

    if dit_weight is None:
        model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
        files = list(model_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No model weights found in {model_dir}")
        if str(files[0]).startswith("pytorch_model_"):
            model_path = dit_weight / f"pytorch_model_{load_key}.pt"
            bare_model = True
        elif any(str(f).endswith("_model_states.pt") for f in files):
            files = [f for f in files if str(f).endswith("_model_states.pt")]
            model_path = files[0]
            if len(files) > 1:
                logger.warning(
                    f"Multiple model weights found in {dit_weight}, using {model_path}"
                )
            bare_model = False
        else:
            raise ValueError(
                f"Invalid model path: {dit_weight} with unrecognized weight format: "
                f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                f"`pytorch_model_*.pt`(provided by HunyuanVideo official) and "
                f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                f"specific weight file, please provide the full path to the file."
            )
    else:
        if dit_weight.is_dir():
            files = list(dit_weight.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {dit_weight}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {dit_weight}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanVideo official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        elif dit_weight.is_file():
            model_path = dit_weight
            bare_model = "unknown"
        else:
            raise ValueError(f"Invalid model path: {dit_weight}")

    if not model_path.exists():
        raise ValueError(f"model_path not exists: {model_path}")
    logger.info(f"Loading torch model {model_path}...")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )
    model.load_state_dict(state_dict, strict=True)
    return model


class set_worker_seed_builder:
    def __init__(self, global_rank):
        self.global_rank = global_rank

    def __call__(self, worker_id):
        set_manual_seed(torch.initial_seed() % (2 ** 32 - 1))


def set_reproducibility(enable, global_seed=None):
    if enable:
        # Configure the seed for reproducibility
        set_manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = not enable
    # Use deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(enable)

    # LSTM and RNN networks are not deterministic


def prepare_model_inputs(
    args,
    batch: tuple,
    device: Union[int, str],
    model,
    vae,
    text_encoder,
    text_encoder_2=None,
    rope_theta_rescale_factor: Union[float, List[float]] = 1.0,
    rope_interpolation_factor: Union[float, List[float]] = 1.0,
):
    media, latents, *batch_args = batch
    if len(batch_args) == 3:
        text_ids, text_mask, kwargs = batch_args
        text_ids_2, text_mask_2 = None, None
    elif len(batch_args) == 5:
        text_ids, text_mask, text_ids_2, text_mask_2, kwargs = batch_args
    else:
        raise ValueError(f"Unexpected batch_args.")
    data_type = kwargs["type"][0]

    # Move batch to device
    media = media.to(device)
    latents = latents.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    # ======================================== Encode media ======================================
    # Used for 3D VAE with 2D inputs(image).
    # Prepare media shape for 2D/3D VAE
    if len(latents.shape) == 1:
        if len(media.shape) == 4:
            # media is a batch of image with shape [b, c, h, w]
            if isinstance(vae, AutoencoderKLCausal3D):
                media = media.unsqueeze(2)  # [b, c, 1, h, w]
        elif len(media.shape) == 5:
            # media is a batch of video with shape [b, c, f, h, w]
            if not isinstance(vae, AutoencoderKLCausal3D):
                media = rearrange(media, "b c f h w -> (b f) c h w")
        else:
            raise ValueError(
                f"Only support media with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape}."
            )

        vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32
        ):
            latents = vae.encode(media).latent_dist.sample()
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
                latents.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
            else:
                latents.mul_(vae.config.scaling_factor)
    elif len(latents.shape) == 5 or len(latents.shape) == 4:  # Using video/image cache
        latents = (
            latents * vae.config.scaling_factor
        )  # vae cache is not multiplied by scaling_factor
    else:
        raise ValueError(
            f"Only support media/latent with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape} {latents.shape}."
        )

    cond_latents = get_cond_latents(args, latents, vae)
    is_uncond = (
        torch.tensor(1).to(torch.int64)
        if random.random() < args.sematic_cond_drop_p
        else torch.tensor(0).to(torch.int64)
    )
    semantic_images = get_cond_images(args, latents, vae, is_uncond=is_uncond)

    # ======================================== Encode text ======================================
    # Autocast is handled by text_encoder itself.
    # Whether to apply text_mask is determined by args.use_attention_mask.
    text_outputs = text_encoder.encode(
        {"input_ids": text_ids, "attention_mask": text_mask},
        data_type=batch_args[-1]["type"][0],
        semantic_images=semantic_images,
    )
    text_states = text_outputs.hidden_state
    text_mask = text_outputs.attention_mask
    text_states_2 = (
        text_encoder_2.encode(
            {"input_ids": text_ids_2, "attention_mask": text_mask_2},
            data_type=data_type,
        ).hidden_state
        if text_encoder_2 is not None
        else None
    )

    # ======================================== Build RoPE ======================================
    target_ndim = 3  # n-d RoPE
    ndim = len(latents.shape) - 2
    latents_size = list(latents.shape[-ndim:])
    freqs_cos, freqs_sin = get_rope_freq_from_size(
        args,
        model,
        latents_size,
        ndim,
        target_ndim,
        rope_theta_rescale_factor=rope_theta_rescale_factor,
        rope_interpolation_factor=rope_interpolation_factor,
    )

    # ===================================== Pack model kwargs ==================================
    model_kwargs = dict(
        text_states=text_states,  # [b, 256, 4096]
        text_mask=text_mask,  # [b, 256]
        text_states_2=text_states_2,  # [b, 768]
        freqs_cos=freqs_cos,  # [seqlen, head_dim]
        freqs_sin=freqs_sin,  # [seqlen, head_dim]
        return_dict=True,
    )

    return latents, model_kwargs, freqs_cos.shape[0], cond_latents


def format_params(params):
    if params < 1e6:
        return f"{params} (less than 1M)"
    elif params < 1e9:
        return f"{params / 1e6:.2f}M"
    else:
        return f"{params / 1e9:.2f}B"


def set_manual_seed(global_seed):
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)


def get_rope_freq_from_size(
    args,
    model,
    latents_size,
    ndim,
    target_ndim,
    rope_theta_rescale_factor=1.0,
    rope_interpolation_factor=1.0,
):

    if isinstance(model.patch_size, int):
        assert all(s % model.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size for s in latents_size]

    elif isinstance(model.patch_size, list):
        assert all(
            s % model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size[idx] for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=args.rope_theta,
        use_real=True,
        theta_rescale_factor=rope_theta_rescale_factor,
        interpolation_factor=rope_interpolation_factor,
    )

    return freqs_cos, freqs_sin


# copy from https://github.com/huggingface/diffusers/blob/ec9bfa9e148b7764137dd92247ce859d915abcb0/examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py#L258
# get kohya lora state dict
def get_module_kohya_state_dict(module, prefix, dtype, adapter_name="default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(
                module.peft_config[adapter_name].lora_alpha
            ).to(dtype)

    return kohya_ss_state_dict


# get diffusers lora state dict
def get_module_diffusers_state_dict(module, dtype, adapter_name="default"):
    diffusers_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        diffusers_key = peft_key.replace("base_model.model", "diffusion_model")
        diffusers_ss_state_dict[diffusers_key] = weight.to(dtype)

    return diffusers_ss_state_dict
