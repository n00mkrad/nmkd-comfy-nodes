import cv2
import numpy as np
import torch
import os

from einops import rearrange
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

from custom_nodes.comfy_controlnet_preprocessors.util import annotator_ckpts_path
import comfy.model_management as model_management


remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth"
#Change to a new URL but the file is still the same

class MLSDdetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            from custom_nodes.comfy_controlnet_preprocessors.util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.to(model_management.get_torch_device()).eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass
        return img_output[:, :, 0]