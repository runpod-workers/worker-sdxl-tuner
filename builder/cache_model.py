import os
import torch
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

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

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-xl-base-1.0", None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        "stabilityai/stable-diffusion-xl-base-1.0", None, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", revision=None
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
         "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", revision=None
    )
    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)



if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        print(f"HF_HOME is set to {os.environ.get('HF_HOME')}")
        raise ValueError("HF_HOME must be set to /cache/huggingface")

    get_diffusion_pipelines()
