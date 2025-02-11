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


def main(args):
    """Main function to execute the training and upload process."""
    os.environ["HF_TOKEN"] = args.hf_token

    # Generate full layer paths from provided layer numbers
    layer_numbers = [int(num.strip()) for num in args.layer_numbers.split(",")] if args.layer_numbers else []
    base_layer_path = "transformer.single_transformer_blocks"
    layers = [f"{base_layer_path}.{num}.proj_out" for num in layer_numbers]

    # Training parameters
    params = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "dataset": args.dataset,
        "training_type": args.training_type,
        "lora_rank": args.lora_rank,
        "save_every": args.save_every,
        "caption_dropout_rate": args.caption_dropout_rate,
        "sample_every": args.sample_every,
        "sampling_prompts": args.sampling_prompts or [],
        "sampling_seed": args.sampling_seed,
        "sampling_steps": args.sampling_steps,
        "layers": layers,
        "max_step_saves_to_keep": args.max_step_saves_to_keep,
        "resolution": args.resolution,
        "cache_latents_to_disk": args.cache_latents_to_disk,
        "batch_size": args.batch_size,
    }

    # Generate job configuration
    lora_name = f"{params['dataset']}-steps{params['steps']}-lr{params['learning_rate']}-lrank{params['lora_rank']}-b{params['batch_size']}"
    if params["layers"]:
        lora_name += "-layers" + "-".join([layer.split('.')[-2] for layer in params["layers"]])
    else:
        lora_name += "-full-layer"

    output_path = f"{cwd}/output/{lora_name}"
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', lora_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', f'{cwd}/output'),
                    ('device', 'cuda:0'),
                    ('network', OrderedDict([
                        ('type', params["training_type"]),
                        ('linear', params["lora_rank"]),
                        ('linear_alpha', 16),
                        ('network_kwargs', OrderedDict([
                            ('only_if_contains', params["layers"])
                        ])) if params["layers"] else None,
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', params["save_every"]),
                        ('max_step_saves_to_keep', params["max_step_saves_to_keep"])
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', f'{cwd}/dataset/{params["dataset"]}'),
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
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                        ('is_flux', True),
                        ('quantize', False),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA training and upload results.")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--training_type", default="lora", help="Training type (e.g., lora, lokr)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--save_every", type=int, default=100, help="Save frequency")
    parser.add_argument("--caption_dropout_rate", type=float, default=0.05, help="Caption dropout rate")
    parser.add_argument("--sample_every", type=int, default=10000, help="Sampling frequency")
    parser.add_argument("--sampling_prompts", nargs="*", help="Sampling prompts")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--sampling_steps", type=int, default=28, help="Sampling steps")
    parser.add_argument("--layer_numbers", default="'7,12,16,20'", help="Comma-separated list of layer numbers (e.g., '7,12,16,20')")
    parser.add_argument("--max_step_saves_to_keep", type=int, default=4, help="Max saved steps to keep")
    parser.add_argument("--resolution", nargs=3, type=int, default=[512, 768, 1024], help="Resolution for training")
    parser.add_argument("--cache_latents_to_disk", action="store_true", help="Cache latents to disk")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()
    main(args)
