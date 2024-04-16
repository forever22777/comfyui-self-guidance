import torch


from comfy import model_management


import comfy.utils
from comfy import model_detection

import comfy.model_patcher
import comfy.lora
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import comfy.taesd.taesd
from comfy.sd import VAE, CLIP
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


class NoMatchingModelError(Exception):
    err_code = "03000"


class NoCorrespondModel(Exception):
    err_code = "03001"


supported_models_dict = {"SD15":
                         StableDiffusionPipeline, "SDXL": StableDiffusionXLPipeline}


def load_checkpoint_mix_with_diffusers(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd = comfy.utils.load_torch_file(ckpt_path)
    clip = None
    vae = None
    model = None
    clip_target = None

    load_device = model_management.get_torch_device()
    parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
    model_config = model_detection.model_config_from_unet(
        sd, "model.diffusion_model.")
    if not model_config:
        raise NoMatchingModelError(
            "No matching model config in ComfyUI and diffusers")

    unet_dtype = model_management.unet_dtype(
        model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    if model_config is None:
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(ckpt_path))

    latent_format = model_config.latent_format
    diffusion_model_name = model_config.latent_format.__init__.__qualname__.split(".")[
        0]
    correspond_pipeline = supported_models_dict.get(
        diffusion_model_name, None)
    if not correspond_pipeline:
        raise NoCorrespondModel(
            f"{diffusion_model_name} not in the supported correspond dict")
    model = correspond_pipeline.from_single_file(
        ckpt_path, torch_dtype=unet_dtype, variant="fp16", use_safetensors=True)

    unet = model.unet.to(device=load_device)
    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                with torch.no_grad():
                    clip = CLIP(
                        clip_target, embedding_directory=embedding_directory)
                    m, u = clip.load_sd(clip_sd, full_model=True)
                    if len(m) > 0:
                        print("clip missing:", m)

                    if len(u) > 0:
                        print("clip unexpected:", u)
            else:
                print(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    del sd, model
    return (unet, clip, vae, latent_format)
