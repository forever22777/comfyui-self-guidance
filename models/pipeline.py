import copy
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import comfy.utils
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + \
        (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            if place_in_unet in ["mid", "up"]:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self,):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0


class SelfGuidanceAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        self.attnstore(attention_probs, is_cross, self.place_in_unet)
        return hidden_states


class SelfGuidanceStableDiffusionPipeline(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin):
    def __init__(
        self,
        unet,
        scheduler,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

    def _encode_prompt(
        self,
        positive
    ):
        prompt_embeds = positive[0][0].to(self.unet.dtype)
        pooled_prompt_embeds = positive[0][1]['pooled_output'].to(
            self.unet.dtype)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def normalization(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def calculate_appearence(self, norm_data, feature):
        norm_data_detach = norm_data.detach().clone().to(torch.float32)
        norm_data_detach = TF.resize(norm_data_detach.unsqueeze(
            0), feature.shape[1], antialias=True)
        masked_feature = torch.sum(
            torch.sum(norm_data_detach * feature.to(torch.float32), dim=1), dim=1)
        app = masked_feature / torch.sum(norm_data_detach)
        return app

    def calculate_size(self, data, atten_binary_sharpness_scale=30):
        data = self.normalization(data)-0.5
        data = torch.sigmoid(atten_binary_sharpness_scale*data)
        norm_data = self.normalization(data)
        size_rate = torch.sum(norm_data)/(data.shape[1]*data.shape[0])
        return size_rate, norm_data

    def calculate_shape(self, norm_data):
        return norm_data

    def _calculate_property(self, tmp_sample, token_indices, fix_token_indices) -> dict:
        property_dict = {}
        token_indices = token_indices + fix_token_indices
        object_number = len(token_indices)
        for location in ["mid", "up"]:
            location_count = 0
            for attn_map_integrated in self.attention_store.maps(location):
                location_count += 1
                attn_map = attn_map_integrated.chunk(3)[0]
                b, i, j = attn_map.shape
                H = W = int(math.sqrt(i))
                for obj_idx in range(object_number):
                    obj_position = token_indices[obj_idx]
                    ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                    ca_map_obj = torch.sum(ca_map_obj, dim=0)
                    size_rate, norm_data = self.calculate_size(ca_map_obj)
                    obj_shape = self.calculate_shape(norm_data)
                    obj_app = self.calculate_appearence(norm_data, tmp_sample)
                    property_name = f"{location}_l{location_count}_o{obj_position}"
                    property_dict[property_name] = {}
                    property_dict[property_name]["size_rate"] = size_rate.detach(
                    ).clone()
                    property_dict[property_name]["shape"] = obj_shape.detach(
                    ).clone()
                    property_dict[property_name]["app"] = obj_app.detach(
                    ).clone()

        return property_dict

    def tau(self, shape, shape_rate):
        shape = shape.view(1, 1, shape.shape[0], shape.shape[1])
        scaled_shape = F.interpolate(
            shape, mode='bilinear', align_corners=False, scale_factor=shape_rate).squeeze()
        scale_shape_0 = scaled_shape.shape[0]
        if not scaled_shape.shape:
            scaled_shape = scaled_shape.reshape(1, 1)
        if scale_shape_0 > shape.shape[2]:
            center_scaled_x_y = math.floor(scaled_shape.shape[0]/2)
            scaled_t = int(center_scaled_x_y-shape.shape[2]/2+1)
            scaled_b = int(center_scaled_x_y+shape.shape[2]/2+1)
            scaled_shape = scaled_shape[scaled_t:scaled_b, scaled_t:scaled_b]
        if scale_shape_0 < shape.shape[2]:
            center_x_y = shape.shape[2]/2
            scaled_t = int(center_x_y-scaled_shape.shape[0]/2+1)
            scaled_b = int(center_x_y+scaled_shape.shape[0]/2+1)
            scaled_shape_eps = torch.ones(
                shape.shape[2], shape.shape[2]) * 1e-6 * torch.rand(1).item()
            scaled_shape_eps[scaled_t:scaled_b,
                             scaled_t:scaled_b] = scaled_shape
            scaled_shape = scaled_shape_eps
        return scaled_shape

    def get_property(self, property_name, property_dict, obj_position, fix_token_indices):
        if obj_position in fix_token_indices:
            obj_size = property_dict[property_name]["size_rate"]
            obj_shape = property_dict[property_name]["shape"]
            obj_app = property_dict[property_name]["app"]
        else:
            obj_size = property_dict[property_name]["size_rate"]
            if self.obj_scale:
                obj_size = self.obj_scale*obj_size
            if self.obj_scale:
                shape_rate = self.obj_scale
            else:
                shape_rate = 1
            if not self.centers:
                obj_shape = self.tau(property_dict[property_name]["shape"], shape_rate).to(
                    property_dict[property_name]["shape"].device)
            else:
                obj_shape = None
            obj_app = property_dict[property_name]["app"]
        return obj_size, obj_shape, obj_app

    def _compute_loss(self, token_indices, properties, activation, fix_token_indices=[]) -> torch.Tensor:
        loss = 0
        token_indices = token_indices + fix_token_indices
        object_number = len(token_indices)
        total_maps = 0
        for location in ["mid", "up"]:
            location_count = 0
            for attn_map in self.attention_store.maps(location):
                location_count += 1
                b, i, _ = attn_map.shape
                H = W = int(math.sqrt(i))
                total_maps += 1
                obj_loss = 0
                for obj_idx in range(object_number):
                    obj_position = token_indices[obj_idx]
                    ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                    ca_map_obj = torch.sum(ca_map_obj, dim=0)
                    property_name = f"{location}_l{location_count}_o{obj_position}"
                    obj_scale, obj_shape, obj_app = self.get_property(
                        property_name, properties, obj_position, fix_token_indices)
                    size_rate, norm_data = self.calculate_size(ca_map_obj)
                    target_app = self.calculate_appearence(
                        norm_data, activation)
                    if obj_position in fix_token_indices:
                        size_loss = 0
                    else:
                        size_loss = torch.abs(obj_scale-size_rate)
                    if obj_shape is not None:
                        target_shape = self.calculate_shape(norm_data)
                        shape_loss = torch.sum(
                            torch.abs(obj_shape-target_shape))
                    else:
                        shape_loss = 0
                    app_loss = torch.sum(torch.abs(obj_app-target_app))
                    if obj_position not in fix_token_indices:
                        app_omega_dynamic, size_omega_dynamic, shape_omega_dynamic = self.app_omega, self.size_omega, self.shape_omega
                    else:
                        app_omega_dynamic, size_omega_dynamic, shape_omega_dynamic = self.fix_app_omega, self.fix_size_omega, self.fix_shape_omega

                    if location not in ["mid"]:
                        app_omega_dynamic = 0
                    else:
                        pass
                    obj_loss += size_omega_dynamic*size_loss + \
                        app_omega_dynamic*app_loss+shape_omega_dynamic*shape_loss
                loss += obj_loss
        loss /= object_number * total_maps
        return loss

    def register_attention_control(self, model):
        attn_procs = {}
        cross_att_count = 0
        key_count = len(model.attn_processors)
        for name in model.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                if (key_count == 32 and not name.startswith("up_blocks.3")) or (key_count == 140 and not name.startswith("up_blocks.1")):
                    place_in_unet = "up"
                else:
                    place_in_unet = "up_discard"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            if place_in_unet in ['mid', 'up']:
                cross_att_count += 1
                attn_procs[name] = SelfGuidanceAttnProcessor(
                    attnstore=self.attention_store, place_in_unet=place_in_unet)
            else:
                attn_procs[name] = model.attn_processors[name]
        model.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def prepare_latents(self, init_latents, batch_size, timestep, dtype, device, generator, latents=None):
        shape = init_latents.shape
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)
            init_latents = torch.cat([init_latents], dim=0)
            init_latents = self.scheduler.add_noise(
                init_latents, latents, timestep).to(dtype=torch.float16)
        else:
            init_latents = init_latents.to(device, dtype=torch.float16)
        return init_latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(
            original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def __call__(
        self,
        token_indices: Union[List[List[List[int]]], List[List[int]]] = None,
        fix_token_indices: Union[List[List[List[int]]],
                                 List[List[int]]] = None,
        latent_image: Union[List[torch.FloatTensor],] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 128,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        positive: Union[List[torch.FloatTensor],] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        max_guidance_iter: int = 128,
        loss_scale: int = 1,
        obj_scale: Optional[float] = None,
        centers: Union[List[List[List[int]]], List[List[int]]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        size_omega: float = 3.0,
        shape_omega: float = 4.,
        app_omega: float = 0.2,
        fix_size_omega: float = 0.9,
        fix_shape_omega: float = 0.9,
        fix_app_omega: float = 0.3

    ):

        self.centers = centers
        self.obj_scale = obj_scale
        original_size = (height, width)
        target_size = (height, width)

        batch_size = 1
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        self.size_omega = size_omega
        self.shape_omega = shape_omega
        self.app_omega = app_omega
        self.fix_size_omega = fix_size_omega
        self.fix_shape_omega = fix_shape_omega
        self.fix_app_omega = fix_app_omega

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            positive
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        last_timestep = torch.tensor([timesteps[0]])
        latents_gene = self.prepare_latents(
            latent_image,
            batch_size * num_images_per_prompt,
            last_timestep,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if denoising_end is not None:
            num_inference_steps = int(
                round(denoising_end * num_inference_steps))
            timesteps = timesteps[: num_warmup_steps +
                                  self.scheduler.order * num_inference_steps]

        self.attention_store = AttentionStore()
        self.register_attention_control(self.unet)
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order

        atten_dict = {}
        add_text_embeds = torch.stack(
            [add_text_embeds[1], add_text_embeds[0], add_text_embeds[1]])
        add_time_ids = torch.stack(
            [add_time_ids[1], add_time_ids[0], add_time_ids[1]])
        prompt_embeds = torch.stack(
            [prompt_embeds[1], prompt_embeds[0], prompt_embeds[1]])

        with torch.inference_mode(False):
            with torch.enable_grad():
                self.unet_enable_grad = copy.deepcopy(self.unet)
                self.register_attention_control(self.unet_enable_grad)
                encoder_hidden_states = copy.deepcopy(
                    prompt_embeds[2].unsqueeze(0)).requires_grad_(True)

        comfy_pbar = comfy.utils.ProgressBar(num_inference_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                last_timestep = torch.tensor([t])
                latents_real = self.prepare_latents(
                    latent_image,
                    batch_size * num_images_per_prompt,
                    last_timestep,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents=None,
                )

                latent_model_input = (
                    torch.cat([latents_real, latents_gene, latents_gene])
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    added_cond_kwargs=added_cond_kwargs,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                real_tmp_sample = noise_pred[0].detach().squeeze()

                grad_cond = torch.tensor(0)
                if i < max_guidance_iter:
                    atten_dict = self._calculate_property(
                        real_tmp_sample, token_indices, fix_token_indices)
                    with torch.inference_mode(False):
                        with torch.enable_grad():
                            latents_gene = latents_gene.detach().clone().requires_grad_(True)
                            added_cond_kwargs = {"text_embeds": add_text_embeds[2].unsqueeze(
                                0), "time_ids": add_time_ids[2].unsqueeze(0)}
                            latent_model_input = self.scheduler.scale_model_input(
                                latents_gene, t
                            )
                            gene_tmp_sample = self.unet_enable_grad(
                                latent_model_input,
                                t,
                                encoder_hidden_states=encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False
                            )[0].squeeze()
                            self.unet_enable_grad.zero_grad()
                            loss = (
                                self._compute_loss(
                                    token_indices, atten_dict, gene_tmp_sample, fix_token_indices)
                                * loss_scale
                            )

                            grad_cond = torch.autograd.grad(
                                loss.requires_grad_(True),
                                [latents_gene],
                                retain_graph=False
                            )
                            grad_cond = grad_cond[0] * \
                                self.scheduler.sigmas[i] * 0.5

                if do_classifier_free_guidance:
                    _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    ) + grad_cond

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                latents_gene = self.scheduler.step(
                    noise_pred, t, latents_gene, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    comfy_pbar.update_absolute(i + 1)

        return latents_gene
