import os
from collections import OrderedDict
from fuzzywuzzy import fuzz, process
from comfy import latent_formats

import torch
import folder_paths
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

from .models.pipeline import SelfGuidanceStableDiffusionPipeline
from .models.load import load_checkpoint_mix_with_diffusers

ROOT_PATH = os.path.join(folder_paths.get_folder_paths(
    "custom_nodes")[0], "comfyui_self_guidance")

DEVICE = 'cuda'
WEIGHT_DETYPE = torch.float16


class MaxGuidanceIterError(Exception):
    err_code = "02000"


class ObjectError(Exception):
    err_code = "02001"


SCHEDULER_DICT = OrderedDict([
    ("Euler Discrete", EulerDiscreteScheduler),
    ("Euler Ancestral Discrete", EulerAncestralDiscreteScheduler),
    ("DDIM", DDIMScheduler),
    ("DPM++ 2M Karras", DPMSolverMultistepScheduler),
    ("LCM", LCMScheduler),
    ("LMS", LMSDiscreteScheduler),
    ("PNDM", PNDMScheduler),
    ("UniPC", UniPCMultistepScheduler),

])


class SelfGuidanceSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet": ("UNET2D",),
                "latent_image": ("LATENT",),
                "height": ("INT", {"default": 1024,}),
                "width": ("INT", {"default": 1024,}),
                "prompt": ("CONDITIONING", ),
                "seed": ("INT", {"default": 999999999, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 128, "min": 0, "max": 150}),
                "max_guidance_iter": ("INT", {"default": 84, "min": 0, "max": 150}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "loss_scale": ("FLOAT", {"default": 1, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "sampler_scheduler_pairs": (list(SCHEDULER_DICT.keys()),),
                "obj_scale": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 2.5, "step": 0.01}),
                "target_object": ("CONDITIONING",),
                "fix_object": ("CONDITIONING",),
                "beta_start": ("FLOAT", {"default": 0.00085, "min": 0.0, "step": 0.00001}),
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.0, "step": 0.00001}),
                "beta_schedule": (["scaled_linear", "linear", "squaredcos_cap_v2"],),
                "prediction_type": (["epsilon", "v_prediction", "sample"],),
                "timestep_spacing": (["leading", "trailing", "linspace"],),
                "steps_offset": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "size_omega": ("FLOAT", {"default": 5, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "shape_omega": ("FLOAT", {"default": 8, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "app_omega": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "fix_size_omega": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "fix_shape_omega": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
                "fix_app_omega": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.05, "round": 0.01}),
            }
        }

    RETURN_TYPES = (
        "LATENT",
    )
    RETURN_NAMES = (
        "latent",
    )
    FUNCTION = "self_guidance"

    CATEGORY = "sampling"

    def self_guidance(
        self,
        unet,
        latent_image,
        width,
        height,
        prompt,
        target_object,
        fix_object,
        obj_scale,
        seed,
        steps,
        guidance_scale,
        loss_scale,
        sampler_scheduler_pairs,
        beta_start,
        beta_end,
        beta_schedule,
        prediction_type,
        timestep_spacing,
        steps_offset,
        max_guidance_iter,
        size_omega,
        shape_omega,
        app_omega,
        fix_size_omega,
        fix_shape_omega,
        fix_app_omega
    ):
        if max_guidance_iter > steps:
            raise MaxGuidanceIterError(
                "max_guidance_iter should be less than steps")
        key_count = len(unet.attn_processors)
        if key_count == 32:
            latent_format = latent_formats.SD15()
        elif key_count == 140:
            latent_format = latent_formats.SDXL()
        else:
            raise NotImplementedError("Other model are not supported now")
        latent_image = latent_format.process_in(
            latent_image["samples"]).to(DEVICE, dtype=WEIGHT_DETYPE)
        scheduler_class = SCHEDULER_DICT[sampler_scheduler_pairs]

        sched_kwargs = {
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "steps_offset": steps_offset,
            "prediction_type": prediction_type,
            "timestep_spacing": timestep_spacing,
        }

        scheduler = scheduler_class(**sched_kwargs)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        pipeline = SelfGuidanceStableDiffusionPipeline(unet, scheduler)
        samples = pipeline(token_indices=target_object,
                           fix_token_indices=fix_object,
                           latent_image=latent_image,
                           height=height,
                           width=width,
                           num_inference_steps=steps,
                           max_guidance_iter=max_guidance_iter,
                           guidance_scale=guidance_scale,
                           loss_scale=loss_scale,
                           generator=generator,
                           obj_scale=obj_scale,
                           positive=prompt,
                           size_omega=size_omega,
                           shape_omega=shape_omega,
                           app_omega=app_omega,
                           fix_size_omega=fix_size_omega,
                           fix_shape_omega=fix_shape_omega,
                           fix_app_omega=fix_app_omega
                           )
        samples = latent_format.process_out(samples)
        return ({"samples": samples}, )


class CLIPConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True}),
                             "target_object": ("STRING", {"multiline": True}),
                             "fix_object": ("STRING", {"multiline": True}),
                             "clip": ("CLIP", )}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING")

    RETURN_NAMES = ("prompt", "target_object", "fix_object")

    FUNCTION = "clip_conditioning"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        if 'l' in tokens:
            ids = clip.tokenizer.untokenize(tokens['l'][0])
        else:
            ids = clip.tokenizer.untokenize(tokens['h'][0])
        return ([[cond, {"pooled_output": pooled}]], ids)

    def clip_conditioning(self, prompt, target_object, fix_object, clip):
        cond, ids = self.encode(clip, prompt)
        cleaned_ids = []
        tmp_info = ["", []]
        for i in range(len(ids)):
            if ids[i][1] == "<|startoftext|>":
                continue
            if ids[i][1] == "<|endoftext|>":
                break
            tmp_info[0] += ids[i][1].split("</w>")[0]
            tmp_info[1].append(i)
            if ids[i][1].endswith('</w>'):
                cleaned_ids.append(tmp_info)
                tmp_info = ["", []]
        cleaned_token = [cleaned_ids[i][0] for i in range(len(cleaned_ids))]
        cleaned_token_set = list(set(cleaned_token))
        target_object = target_object.split(",")
        fix_object = fix_object.split(",")
        target_object_ids = []
        fix_object_ids = []
        for object in target_object:
            matches = process.extract(
                object, cleaned_token_set, scorer=fuzz.ratio, limit=1)
            if matches[0][1] >= 90:
                target_object_ids.extend([cleaned_ids[i][1] for i in range(
                    len(cleaned_token)) if object == cleaned_token[i]])
        for object in fix_object:
            matches = process.extract(
                object, cleaned_token_set, scorer=fuzz.ratio, limit=1)
            if matches[0][1] >= 90:
                fix_object_ids.extend([cleaned_ids[i][1] for i in range(
                    len(cleaned_token)) if object == cleaned_token[i]])
        fix_object_ids = [id for item in fix_object_ids for id in item]
        target_object_ids = [id for item in target_object_ids for id in item]
        if not target_object_ids:
            raise ObjectError("target_object or fix_object not in prompt")
        return cond, target_object_ids, fix_object_ids


class CheckpointLoaderMixWithDiffusers:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}

    RETURN_TYPES = (
        "UNET2D", "CLIP", "VAE"
    )
    RETURN_NAMES = (
        "unet", "clip", "vae"
    )
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_mix_with_diffusers(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out


NODE_CLASS_MAPPINGS = {
    "SelfGuidanceSampler": SelfGuidanceSampler,
    "CLIPConditioning": CLIPConditioning,
    "CheckpointLoaderMixWithDiffusers": CheckpointLoaderMixWithDiffusers
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfGuidanceSampler": "SelfGuidanceSampler",
    "CLIPConditioning": "CLIPConditioning",
    "CheckpointLoaderMixWithDiffusers": "Load Checkpoint Mix With Diffusers"
}
