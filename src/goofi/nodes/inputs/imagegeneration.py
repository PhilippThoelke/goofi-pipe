from typing import Any, Dict, Tuple

import cv2
import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class ImageGeneration(Node):
    def config_params():
        return {
            "image_generation": {
                "inference_steps": IntParam(50, 5, 100),
                "guidance_scale": FloatParam(7.5, 0.1, 20),
                "seed": IntParam(-1, -1, 1000000, doc="-1 for random seed"),
                "width": IntParam(512, 100, 1024),
                "height": IntParam(512, 100, 1024),
                "scheduler": StringParam(
                    list(SCHEDULER_MAPPING.keys())[0],
                    options=list(SCHEDULER_MAPPING.keys()),
                ),
                "device": "cuda",
            },
            "img2img": {
                "enabled": False,
                "strength": FloatParam(0.8, 0, 1),
                "resize_input": True,
                "reset_image": BoolParam(False, trigger=True),
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
            # TODO: make sure this works without internet access
            self.sd_pipe = self.diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=self.torch.float16
            )
        else:
            # TODO: make sure this works without internet access
            self.sd_pipe = self.diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.torch.float16)

        # initialize scheduler
        self.image_generation_scheduler_changed(self.params.image_generation.scheduler.value)

        # set device
        self.sd_pipe.to(self.params.image_generation.device.value)

        # initialize last image
        if not hasattr(self, "last_img"):
            self.last_img = None
            self.reset_last_img()

    def process(self, prompt: Data, negative_prompt: Data, base_image: Data) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        if prompt is None:
            return None

        if self.params.img2img.reset_image.value:
            # reset the last image to zeros
            self.reset_last_img()

        # set seed
        if self.params.image_generation.seed.value != -1:
            self.torch.manual_seed(self.params.image_generation.seed.value)

        with self.torch.inference_mode():
            if self.params.img2img.enabled.value:
                if base_image is None:
                    base_image = self.last_img
                else:
                    base_image = base_image.data

                if self.params.img2img.resize_input.value:
                    # resize input image to match the last image
                    base_image = cv2.resize(
                        base_image, (self.params.image_generation.width.value, self.params.image_generation.height.value)
                    )

                if base_image.ndim == 3:
                    # add batch dimension
                    base_image = np.expand_dims(base_image, 0)

                # run the img2img stable diffusion pipeline
                img, _ = self.sd_pipe(
                    image=base_image,
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
                    raise ValueError("base_image is not supported in text2img mode. Enable img2img or disconnect base_image.")

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
        img = np.array(img[0])
        # save last image
        self.last_img = img

        return {
            "img": (
                img,
                {"prompt": prompt.data, "negative_prompt": negative_prompt.data if negative_prompt is not None else None},
            )
        }

    def reset_last_img(self):
        """Reset the last image."""
        self.last_img = np.zeros((self.params.image_generation.height.value, self.params.image_generation.width.value, 3))

    def image_generation_scheduler_changed(self, value):
        """Change the scheduler of the Stable Diffusion pipeline."""
        scheduler_settings = dict(SCHEDULER_MAPPING[value])
        scheduler_type = scheduler_settings.pop("_sched")
        self.sd_pipe.scheduler = getattr(self.diffusers, scheduler_type).from_config(
            self.sd_pipe.scheduler.config, **scheduler_settings
        )

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


SCHEDULER_MAPPING = {
    "DPM++ 2M": dict(_sched="DPMSolverMultistepScheduler"),
    "DPM++ 2M Karras": dict(_sched="DPMSolverMultistepScheduler", use_karras_sigmas=True),
    "DPM++ 2M SDE": dict(_sched="DPMSolverMultistepScheduler", algorithm_type="sde-dpmsolver++"),
    "DPM++ 2M SDE Karras": dict(_sched="DPMSolverMultistepScheduler", use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ SDE": dict(_sched="DPMSolverSinglestepScheduler"),
    "DPM++ SDE Karras": dict(_sched="DPMSolverSinglestepScheduler", use_karras_sigmas=True),
    "DPM2": dict(_sched="KDPM2DiscreteScheduler"),
    "DPM2 Karras": dict(_sched="KDPM2DiscreteScheduler", use_karras_sigmas=True),
    "DPM2 a": dict(_sched="KDPM2AncestralDiscreteScheduler"),
    "DPM2 a Karras": dict(_sched="KDPM2AncestralDiscreteScheduler", use_karras_sigmas=True),
    "Euler": dict(_sched="EulerDiscreteScheduler"),
    "Euler a": dict(_sched="EulerAncestralDiscreteScheduler"),
    "Heun": dict(_sched="HeunDiscreteScheduler"),
    "LMS": dict(_sched="LMSDiscreteScheduler"),
    "LMS Karras": dict(_sched="LMSDiscreteScheduler", use_karras_sigmas=True),
}
