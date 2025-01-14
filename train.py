import argparse
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from collections import OrderedDict
from huggingface_hub import HfApi
from toolkit.job import run_job
import requests
from urllib.parse import urlparse
import zipfile
import shutil

def get_username_from_token(hf_token):
    """Fetch the username associated with the Hugging Face token."""
    url = "https://huggingface.co/api/whoami-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("name", "Username not found")
    else:
        raise ValueError(f"Error: Unable to fetch username (Status Code: {response.status_code})")

def download_file(url, save_dir=f"{cwd}/dataset"):
    """
    Downloads a file from the given URL and saves it to the specified directory.

    Parameters:
        url (str): The URL of the file to download.
        save_dir (str): The directory where the file will be saved (default: "downloads").

    Returns:
        str: The full path of the downloaded file.
    """
    # Parse the file name from the URL
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    
    if not file_name:
        raise ValueError("The URL does not contain a valid file name.")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path to save the file
    file_path = os.path.join(save_dir, file_name)
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Write the file to the specified path
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    # Extract the contents of the ZIP file
    extract_dir = os.path.join(save_dir, os.path.splitext(file_name)[0])
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    
    os.remove(file_path)
    
    return extract_dir

def main(args):
    """Main function to execute the training and upload process."""
    os.environ["HF_TOKEN"] = args.hf_token

    
    layer_numbers = []
    layers = []
    if args.layer_numbers and args.layer_numbers.strip() and args.layer_numbers.lower() != 'none':
        layer_numbers = [num.strip() for num in args.layer_numbers.split(",")] if args.layer_numbers else []
        base_layer_path = "transformer.single_transformer_blocks"
        layers = [f"{base_layer_path}.{num}.proj_out" for num in layer_numbers]

    dataset = download_file(args.dataset_url)

    # Training parameters
    params = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "training_type": args.training_type,
        "lora_rank": args.lora_rank,
        "save_every": args.save_every,
        "caption_dropout_rate": args.caption_dropout_rate,
        "sample_every": args.sample_every,
        "sampling_prompts": args.sampling_prompts or [],
        "sampling_seed": args.sampling_seed,
        "sampling_steps": args.sampling_steps,
        "max_step_saves_to_keep": args.max_step_saves_to_keep,
        "resolution": args.resolution,
        "cache_latents_to_disk": args.cache_latents_to_disk,
        "batch_size": args.batch_size,
        "trigger_word":args.trigger_word,
    }
    
    if layers:
        params["layers"] = layers

    # Generate job configuration
    lora_name = f"{args.hf_repo_id.split('/')[-1]}"
    output_path = f"{cwd}/output/{lora_name}"
    
    network_dict = OrderedDict([
        ('type', params["training_type"]),
        ('linear', params["lora_rank"]),
        ('linear_alpha', 16),
    ])
    
    # Only add network_kwargs if layers exist
    if layers:
        network_dict['network_kwargs'] = {'only_if_contains': layers}
    
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', lora_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', f'{cwd}/output'),
                    ('device', 'cuda:0'),
                    ('trigger_word', params["trigger_word"]),
                    ('network', network_dict),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', params["save_every"]),
                        ('max_step_saves_to_keep', params["max_step_saves_to_keep"])
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', dataset),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', params["caption_dropout_rate"]),
                            ('cache_latents_to_disk', params["cache_latents_to_disk"]),
                            ('resolution', params["resolution"])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', params["batch_size"]),
                        ('steps', params["steps"]),
                        ('lr', params["learning_rate"]),
                        ('gradient_checkpointing', True),
                        ('optimizer', 'adamw8bit'),
                        ('disable_sampling', True),
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                        ('is_flux', True),
                        ('quantize', params["batch_size"] >= 4),
                    ])),
                    ('sample', OrderedDict([
                        ('sample_every', params["sample_every"]),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', params["sampling_prompts"]),
                        ('seed', params["sampling_seed"]),
                        ('sample_steps', params["sampling_steps"])
                    ]))
                ])
            ])
        ]))
    ])

    # Run the job
    run_job(job_to_run)

    # Upload to Hugging Face Hub
    username = get_username_from_token(args.hf_token)
    repo = f"{username}/{lora_name}"
    api = HfApi()
    
    try:
        api.create_repo(
            repo_id=repo,
            token=args.hf_token
        )
        api.upload_file(
            path_or_fileobj=f"{output_path}/{lora_name}.safetensors",
            path_in_repo="lora.safetensors",
            repo_id=repo,
            repo_type="model",
            token=args.hf_token
        )
        api.upload_file(
            path_or_fileobj=f"{output_path}/config.yaml",
            path_in_repo="config.yaml",
            repo_id=repo,
            repo_type="model",
            token=args.hf_token
        )
        if os.path.exists(f"{output_path}/samples"):
            api.upload_folder(
                folder_path=f"{output_path}/samples",
                path_in_repo="./samples",
                repo_id=repo,
                repo_type="model",
                token=args.hf_token
            )
        print("Upload successful!")
    except Exception as e:
        print(f"Error during upload: {e}")

    # Cleanup
    print("Cleaning up temporary files...")
    try:
        if os.path.exists(dataset):
            shutil.rmtree(dataset)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA training and upload results.")
    parser.add_argument("--hf_token",type=str, default="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", help="Hugging Face token")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--dataset_url", required=True, help="Dataset url")
    parser.add_argument("--training_type", default="lora", help="Training type (e.g., lora, lokr)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--save_every", type=int, default=10000, help="Save frequency")
    parser.add_argument("--caption_dropout_rate", type=float, default=0.05, help="Caption dropout rate")
    parser.add_argument("--sample_every", type=int, default=10000, help="Sampling frequency")
    parser.add_argument("--sampling_prompts", nargs="*", help="Sampling prompts")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--sampling_steps", type=int, default=28, help="Sampling steps")
    parser.add_argument("--layer_numbers", default="'7,12,16,20'", help="Comma-separated list of layer numbers (e.g., '7,12,16,20')")
    parser.add_argument("--max_step_saves_to_keep", type=int, default=4, help="Max saved steps to keep")
    parser.add_argument("--resolution", nargs=3, type=int, default=[512, 768, 1024], help="Resolution for training")
    parser.add_argument("--cache_latents_to_disk", type=bool, default="True", help="Cache latents to disk")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--trigger_word", type=str, default="UNST", help="Trigger word for lora training")
    parser.add_argument("--hf_repo_id", type=str, default="bb1070/trained_lora", help="huggingface repo id")
    args = parser.parse_args()
    main(args)
