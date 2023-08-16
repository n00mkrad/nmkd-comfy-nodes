# nmkd-comfy-nodes
ComfyUI custom nodes, primarily optimized for usage with API, not GUI

**NOTE:** THIS README MIGHT BE OUT OF DATE. LAST UPDATED: 2023-08-16



**Node Collections**

- comfynmkd: Specific nodes, partially custom-made and partially kitbashed from existing stuff. Dependency-free so far.
- comfy_controlnet_preprocessors: Slimmed-down version of [Fannovel16/comfy_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors)



**Extra Dependencies**

- comfynmkd: None
- comfy_controlnet_preprocessors: `timm==0.6.7 mediapipe==0.10.3` (For all currently enabled nodes)



**Goals**

- Flexibility: Loaders are designed to accept absolute paths. Original ComfyUI only takes filenames and searches certain dirs.
- Better handling of variables: When empty values are used, a node is skipped or uses defaults instead of erroring (e.g. Upscaler with empty path).
- Optimization: Import times and requirements are kept to a minimum.
- Logging: Some nodes print info messages for better headless use.



**My Nodes**

- `NmkdIntegerConstant`, `NmkdFloatConstant`, `NmkdStringConstant` - Self explanatory. They just output a value.
- `NmkdCheckpointLoader` - Loads a checkpoint from path, VAE loading is optional, external VAE can be specified, CLIP Skip can be set.
- `NmkdKSampler`: Mostly the same as the original KSamplerAdvanced, but with an info log message, and `denoise` option (experimental).
- `NmkdImageLoader`: Same as original, but accepts an absolute path.
- `NmkdImageUpscale`: Same as original, but accepts absolute path to upscale model. Is skipped if no path provided, or if an exception occurs.
- `NmkdMultiLoraLoader`: Same as original, but accepts a list of LoRA models and weights (abs paths, comma-separated values).
- `NmkdVaeEncode`: 2-in-1 version of the original VAE Encoders - For regular generation if no mask specified, or for inpainting if mask specified.
- `NmkdImageMaskComposite`: Composites two images based on a mask. Returns first image if the image sizes mismatch, or if an exception occurs.
- `NmkdControlNet`: Combines ControlNet Load+Apply nodes, and accepts abs. paths. Can load Diffusers format if an input model is also specified.
- `NmkdHypernetworkLoader`: Same as original, but accepts absolute paths.
- `NmkdColorPreprocessor`: Scales down an image 64x, then back up again, intended for use with `t2iadapter_color_sd14v1`



**Credits**

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [comfy_controlnet_preprocessors)](https://github.com/Fannovel16/comfy_controlnet_preprocessors)
