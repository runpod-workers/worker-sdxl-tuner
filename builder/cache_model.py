# import os
# from huggingface_hub import snapshot_download

# # Get the hugging face token
# HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_WRITE_TOKEN', None)
# MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_REVISION = os.environ.get('MODEL_REVISION', "main")
# MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/workspace/')

# # Download the model from hugging face
# download_kwargs = {}

# if HUGGING_FACE_HUB_TOKEN:
#     download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

# snapshot_download(
#     MODEL_NAME,
#     revision="main",
#     # allow_patterns="*.safetensors",
#     local_dir=f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}",
#     **download_kwargs
# )


# builder/model_fetcher.py

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL


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
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()