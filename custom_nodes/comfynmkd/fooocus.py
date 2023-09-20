"""
from Fooocus
https://github.com/lllyasviel/Fooocus
"""


import torch
import comfy.model_management
import comfy.utils
import math
import numpy as np
import latent_preview
from comfy.sample import *
from comfy.samplers import *
import comfy.model_base


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = kwargs["pooled_output"]
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    crop_w = kwargs.get("crop_w", 0)
    crop_h = kwargs.get("crop_h", 0)
    target_width = kwargs.get("target_width", width)
    target_height = kwargs.get("target_height", height)

    if kwargs.get("prompt_type", "") == "negative":
        width *= 0.8
        height *= 0.8
    elif kwargs.get("prompt_type", "") == "positive":
        width *= 1.5
        height *= 1.5

    out = []
    out.append(self.embedder(torch.Tensor([height])))
    out.append(self.embedder(torch.Tensor([width])))
    out.append(self.embedder(torch.Tensor([crop_h])))
    out.append(self.embedder(torch.Tensor([crop_w])))
    out.append(self.embedder(torch.Tensor([target_height])))
    out.append(self.embedder(torch.Tensor([target_width])))
    flat = torch.flatten(torch.cat(out))[None, ]
    return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def patch_negative_adm():
    comfy.model_base.SDXL.encode_adm = sdxl_encode_adm_patched

patch_negative_adm()



@torch.no_grad()
def ksampler_with_refiner(model, positive, negative, refiner, refiner_positive, refiner_negative, latent,
                          seed=None, steps=30, refiner_switch_step=20, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
                          scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(1, 2 ** 64)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    comfy.model_management.load_model_gpu(model)

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    refiner_positive_copy = broadcast_cond(refiner_positive, noise.shape[0], device)
    refiner_negative_copy = broadcast_cond(refiner_negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSamplerWithRefiner(model=model, refiner_model=refiner, steps=steps, device=device,
                                  sampler=sampler_name, scheduler=scheduler,
                                  denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, refiner_positive=refiner_positive_copy,
                             refiner_negative=refiner_negative_copy, refiner_switch_step=refiner_switch_step,
                             cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback_function=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return (out, )


class KSamplerWithRefiner:
    SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
    SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    def __init__(self, model, refiner_model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model_patcher = model
        self.refiner_model_patcher = refiner_model

        self.model = model.model
        self.refiner_model = refiner_model.model

        self.model_denoise = CFGNoisePredictor(self.model)
        self.refiner_model_denoise = CFGNoisePredictor(self.refiner_model)

        if self.model.model_type == model_base.ModelType.V_PREDICTION:
            self.model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        else:
            self.model_wrap = k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)

        if self.refiner_model.model_type == model_base.ModelType.V_PREDICTION:
            self.refiner_model_wrap = CompVisVDenoiser(self.refiner_model_denoise, quantize=True)
        else:
            self.refiner_model_wrap = k_diffusion_external.CompVisDenoiser(self.refiner_model_denoise, quantize=True)

        self.model_k = KSamplerX0Inpaint(self.model_wrap)
        self.refiner_model_k = KSamplerX0Inpaint(self.refiner_model_wrap)

        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.sigma_min = float(self.model_wrap.sigma_min)
        self.sigma_max = float(self.model_wrap.sigma_max)
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        if self.scheduler == "karras":
            sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        elif self.scheduler == "exponential":
            sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=self.sigma_min,
                                                                 sigma_max=self.sigma_max)
        elif self.scheduler == "normal":
            sigmas = self.model_wrap.get_sigmas(steps)
        elif self.scheduler == "simple":
            sigmas = simple_scheduler(self.model_wrap, steps)
        elif self.scheduler == "ddim_uniform":
            sigmas = ddim_scheduler(self.model_wrap, steps)
        else:
            print("error invalid scheduler", self.scheduler)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, refiner_positive, refiner_negative, cfg, latent_image=None,
               start_step=None, last_step=None, refiner_switch_step=None,
               force_full_denoise=False, denoise_mask=None, sigmas=None, callback_function=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas
        sigma_min = self.sigma_min

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigma_min = sigmas[last_step]
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        positive = positive[:]
        negative = negative[:]

        resolve_cond_masks(positive, noise.shape[2], noise.shape[3], self.device)
        resolve_cond_masks(negative, noise.shape[2], noise.shape[3], self.device)

        calculate_start_end_timesteps(self.model_wrap, negative)
        calculate_start_end_timesteps(self.model_wrap, positive)

        # make sure each cond area has an opposite one with the same area
        for c in positive:
            create_cond_with_same_area_if_none(negative, c)
        for c in negative:
            create_cond_with_same_area_if_none(positive, c)

        pre_run_control(self.model_wrap, negative + positive)

        apply_empty_x_to_equal_area(
            list(filter(lambda c: c[1].get('control_apply_to_uncond', False) == True, positive)), negative, 'control',
            lambda cond_cnets, x: cond_cnets[x])
        apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        if self.model.is_adm():
            positive = encode_adm(self.model, positive, noise.shape[0], noise.shape[3], noise.shape[2], self.device,
                                  "positive")
            negative = encode_adm(self.model, negative, noise.shape[0], noise.shape[3], noise.shape[2], self.device,
                                  "negative")

        refiner_positive = refiner_positive[:]
        refiner_negative = refiner_negative[:]

        resolve_cond_masks(refiner_positive, noise.shape[2], noise.shape[3], self.device)
        resolve_cond_masks(refiner_negative, noise.shape[2], noise.shape[3], self.device)

        calculate_start_end_timesteps(self.refiner_model_wrap, refiner_positive)
        calculate_start_end_timesteps(self.refiner_model_wrap, refiner_negative)

        # make sure each cond area has an opposite one with the same area
        for c in refiner_positive:
            create_cond_with_same_area_if_none(refiner_negative, c)
        for c in refiner_negative:
            create_cond_with_same_area_if_none(refiner_positive, c)

        if self.model.is_adm():
            refiner_positive = encode_adm(self.refiner_model, refiner_positive, noise.shape[0],
                                          noise.shape[3], noise.shape[2], self.device, "positive")
            refiner_negative = encode_adm(self.refiner_model, refiner_negative, noise.shape[0],
                                          noise.shape[3], noise.shape[2], self.device, "negative")

        def refiner_switch():
            comfy.model_management.load_model_gpu(self.refiner_model_patcher)
            self.model_denoise.inner_model = self.refiner_model_denoise.inner_model
            for i in range(len(positive)):
                positive[i] = refiner_positive[i]
            for i in range(len(negative)):
                negative[i] = refiner_negative[i]
            print('Refiner swapped.')
            return

        def callback(step, x0, x, total_steps):
            if step == refiner_switch_step:
                refiner_switch()
            if callback_function is not None:
                callback_function(step, x0, x, total_steps)

        if latent_image is not None:
            latent_image = self.model.process_latent_in(latent_image)

        extra_args = {"cond": positive, "uncond": negative, "cond_scale": cfg, "model_options": self.model_options,
                      "seed": seed}

        cond_concat = None
        if hasattr(self.model, 'concat_keys'):  # inpaint
            cond_concat = []
            for ck in self.model.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(
                            latent_image)  # NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            extra_args["cond_concat"] = cond_concat

        if sigmas[0] != self.sigmas[0] or (self.denoise is not None and self.denoise < 1.0):
            max_denoise = False
        else:
            max_denoise = True

        if self.sampler == "uni_pc":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas,
                                          sampling_function=sampling_function, max_denoise=max_denoise,
                                          extra_args=extra_args, noise_mask=denoise_mask, callback=callback,
                                          disable=disable_pbar)
        elif self.sampler == "uni_pc_bh2":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas,
                                          sampling_function=sampling_function, max_denoise=max_denoise,
                                          extra_args=extra_args, noise_mask=denoise_mask, callback=callback,
                                          variant='bh2', disable=disable_pbar)
        elif self.sampler == "ddim":
            raise NotImplementedError('Swapped Refiner Does not support DDIM.')
        else:
            extra_args["denoise_mask"] = denoise_mask
            self.model_k.latent_image = latent_image
            self.model_k.noise = noise

            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
            else:
                noise = noise * sigmas[0]

            k_callback = None
            total_steps = len(sigmas) - 1
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

            if latent_image is not None:
                noise += latent_image
            if self.sampler == "dpm_fast":
                samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], total_steps,
                                                               extra_args=extra_args, callback=k_callback,
                                                               disable=disable_pbar)
            elif self.sampler == "dpm_adaptive":
                samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0],
                                                                   extra_args=extra_args, callback=k_callback,
                                                                   disable=disable_pbar)
            else:
                samples = getattr(k_diffusion_sampling, "sample_{}".format(self.sampler))(self.model_k, noise, sigmas,
                                                                                          extra_args=extra_args,
                                                                                          callback=k_callback,
                                                                                          disable=disable_pbar)

        return self.model.process_latent_out(samples.to(torch.float32))
