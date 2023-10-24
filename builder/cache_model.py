import os
import torch
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
    }

    fetch_pretrained_model(
        StableDiffusionXLPipeline, "stabilityai/stable-diffusion-xl-base-1.0", **common_args
    )

    fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    )

    fetch_pretrained_model(
        UNet2DConditionModel, "stabilityai/stable-diffusion-xl-base-1.0", **{"subfolder": "unet"}
    )

    fetch_pretrained_model(
        DDPMScheduler, "stabilityai/stable-diffusion-xl-base-1.0", **{"subfolder": "scheduler"}
    )

    fetch_pretrained_model(
        PretrainedConfig, "stabilityai/stable-diffusion-xl-base-1.0", **{"subfolder": "text_encoder"}
    )

    fetch_pretrained_model(
        AutoTokenizer, "stabilityai/stable-diffusion-xl-base-1.0", **{
            "subfolder": "tokenizer",
            "revision": None,
            "use_fast": False
        }
    )

    fetch_pretrained_model(
        PretrainedConfig, "stabilityai/stable-diffusion-xl-base-1.0", **{
            "subfolder": "text_encoder_2",
            "revision": None,
            "use_fast": False
        }
    )




if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        raise ValueError("HF_HOME must be set to /cache/huggingface")

    get_diffusion_pipelines()
