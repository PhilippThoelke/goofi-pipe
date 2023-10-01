from typing import Any, Dict, Tuple

import cv2
import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class ImageGeneration(Node):
    def config_params():
        return {
            "image_generation": {
                "inference_steps": IntParam(50, 5, 100),
                "guidance_scale": FloatParam(7.5, 0.1, 20),
                "width": IntParam(512, 100, 1024),
                "height": IntParam(512, 100, 1024),
                "scheduler": StringParam(
                    "DDPMScheduler",
                    options=[
                        "DDPMScheduler",
                        "DDIMScheduler",
                        "PNDMScheduler",
                        "LMSDiscreteScheduler",
                        "EulerDiscreteScheduler",
                        "EulerAncestralDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                    ],
                ),
                "device": "cuda",
            },
            "img2img": {
                "enabled": False,
                "strength": FloatParam(0.8, 0, 1),
                "resize_input": False,
            },
        }

    def config_input_slots():
        return {"prompt": DataType.STRING, "negative_prompt": DataType.STRING, "base_image": DataType.ARRAY}

    def config_output_slots():
        return {"img": DataType.ARRAY}

    def setup(self):
        self.torch, self.diffusers = import_libs()

        model_id = "stabilityai/stable-diffusion-2-1"

        # load StableDiffusion model
        if self.params.img2img.enabled.value:
            self.sd_pipe = self.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=self.torch.float16
            )
        else:
            self.sd_pipe = self.diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.torch.float16)

        # initialize scheduler
        self.sd_pipe.scheduler = getattr(self.diffusers, self.params.image_generation.scheduler.value).from_config(
            self.sd_pipe.scheduler.config
        )

        # set device
        self.sd_pipe.to(self.params.image_generation.device.value)

        # initialize last image
        if not hasattr(self, "last_img"):
            self.last_img = np.zeros((self.params.image_generation.height.value, self.params.image_generation.width.value, 3))

    def process(self, prompt: Data, negative_prompt: Data, base_image: Data) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        if prompt is None:
            return None

        with self.torch.inference_mode():
            if self.params.img2img.enabled.value:
                if base_image is None:
                    base_image = self.last_img

                if self.params.img2img.resize_input.value:
                    # resize input image to match the last image
                    base_image = cv2.resize(
                        base_image, (self.params.image_generation.width.value, self.params.image_generation.height.value)
                    )

                # run the img2img stable diffusion pipeline
                img, _ = self.sd_pipe(
                    image=self.last_img,
                    strength=self.params.img2img.strength.value,
                    prompt=prompt.data,
                    negative_prompt=negative_prompt.data if negative_prompt is not None else None,
                    num_inference_steps=self.params.image_generation.inference_steps.value,
                    guidance_scale=self.params.image_generation.guidance_scale.value,
                    return_dict=False,
                    output_type="np",
                )
            else:
                if base_image is not None:
                    raise ValueError("base_image is not supported for text2img.")

                # run the text2img stable diffusion pipeline
                img, _ = self.sd_pipe(
                    prompt=prompt.data,
                    negative_prompt=negative_prompt.data if negative_prompt is not None else None,
                    width=self.params.image_generation.width.value,
                    height=self.params.image_generation.height.value,
                    num_inference_steps=self.params.image_generation.inference_steps.value,
                    guidance_scale=self.params.image_generation.guidance_scale.value,
                    return_dict=False,
                    output_type="np",
                )

        # remove the batch dimension
        img = img[0]
        # save last image
        self.last_img = img

        return {
            "img": (
                img,
                {"prompt": prompt.data, "negative_prompt": negative_prompt.data if negative_prompt is not None else None},
            )
        }

    def image_generation_scheduler_changed(self, value):
        """Change the scheduler of the Stable Diffusion pipeline."""
        self.sd_pipe.scheduler = getattr(self.diffusers, value).from_config(self.sd_pipe.scheduler.config)

    def image_generation_width_changed(self, value):
        """Resize the last image to match the new width (for img2img)."""
        self.last_img = cv2.resize(self.last_img, (value, self.params.image_generation.height.value))

    def image_generation_height_changed(self, value):
        """Resize the last image to match the new height (fog img2img)."""
        self.last_img = cv2.resize(self.last_img, (self.params.image_generation.width.value, value))

    def img2img_enabled_changed(self, value):
        """Load the correct Stable Diffusion pipeline."""
        self.setup()


def import_libs():
    try:
        import torch
    except ImportError:
        raise ImportError(
            "You need to install torch to use the ImageGeneration node with Stable Diffusion: "
            "https://pytorch.org/get-started/locally/"
        )

    try:
        import diffusers
    except ImportError:
        raise ImportError(
            "You need to install diffusers to use the ImageGeneration node with Stable Diffusion: pip install diffusers"
        )

    return torch, diffusers
