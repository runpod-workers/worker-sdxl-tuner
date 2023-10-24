import runpod
from runpod.serverless.utils import rp_download
import subprocess
import os
from huggingface_hub import snapshot_download
import shutil

def huggingface_login(token):
    try:
        if token:
            # Run the huggingface-cli login command with the TOKEN environment variable
            subprocess.run(["huggingface-cli", "login", "--token", token], check=True)
            # If the command was successful, you can print a success message or perform other actions
            print("Hugging Face login successful!, Token from input")
        elif os.environ.get("HUGGING_FACE_HUB_WRITE_TOKEN"):
            token = os.environ.get("HUGGING_FACE_HUB_WRITE_TOKEN")
            subprocess.run(["huggingface-cli", "login", "--token", token], check=True)
            print("Hugging Face login successful!, Token from env")
        else:
            # Handle the case where the TOKEN environment variable is not set
            print("TOKEN environment variable is not set. Please set it before running the command.")

    except subprocess.CalledProcessError as e:
        # If the command failed, you can print an error message or handle the error as needed
        error_message = f"Error running huggingface-cli login: {e}"
        print(error_message)


def run_accelerate_config():
    try:
        subprocess.run(["accelerate", "config", "default"], check=True)
        print("Accelerate config successful!")
    except subprocess.CalledProcessError as e:
        error_message = f"Error running accelerate config: {e}"
        print(error_message)

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/workspace/')
def handler(job):
    '''
    This is the handler function that will be called by the serverless.
    '''
    job_input = job["input"]
    job_id = job["id"]

    # Get the parameters from the job input
    dataset_directory_path = job_input["dataset_url"]
    instance_prompt = job_input["instance_prompt"]
    batch_size = job_input["batch_size"]
    hf_token = job_input["hf_token"]
    training_steps = job_input["training_steps"]

    # example:  local_dataset_directory = "./dog"



     # -------------------------- Download Training Data -------------------------- #
    downloaded_input = rp_download.file(dataset_directory_path)
    print(f"Downloaded input: {downloaded_input}")
    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"job_files/{job_id}/clean_data"
    os.makedirs(flat_directory, exist_ok=True)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    os.makedirs(f"job_files/{job_id}", exist_ok=True)
    os.makedirs(f"job_files/{job_id}/fine_tuned_model", exist_ok=True)

    # -------------------------- Run Training -------------------------- #
    job_output = {}

    # most of the parameteres will be path (Network storage)

    training_command = (
        "accelerate launch train_dreambooth_lora_sdxl.py "
        "--pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' "
        "--pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' "
        f"--instance_data_dir=job_files/{job_id}/clean_data "
        f"--output_dir=outputjob_files/{job_id}/fine_tuned_model "
        "--mixed_precision=fp16 "
        f"--instance_prompt='{instance_prompt}' "
        "--resolution=1024 "
        f"--train_batch_size={batch_size} "
        "--gradient_accumulation_steps=2 "
        "--gradient_checkpointing "
        "--learning_rate=1e-4 "
        "--lr_scheduler='constant' "
        "--lr_warmup_steps=0 "
        "--enable_xformers_memory_efficient_attention "
        "--mixed_precision=fp16 "
        "--use_8bit_adam "
        f"--max_train_steps={training_steps} "
        "--checkpointing_steps=717 "
        "--seed=0 "
        "--push_to_hub"
    )
    try:
        # Execute the command and capture the output
        huggingface_login(hf_token)
        run_accelerate_config()
        output = subprocess.run(training_command, stderr=subprocess.STDOUT, text=True, shell=True, check=True)

        # Return the output directory or a message indicating success
        job_output = {"output_directory": f"job_files/{job_id}/fine_tuned_model"}
        return job_output

    except subprocess.CalledProcessError as e:
        error_message = f"Error running command: {e}\nOutput: {e.output}"
        print(error_message)
        # Return an error message or status code
        return error_message

# Call the handler function

runpod.serverless.start({"handler": handler})
