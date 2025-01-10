import getpass
import os
import torch
from toolkit.job import run_job
from collections import OrderedDict
from PIL import Image
import sys
from collections import OrderedDict
from itertools import product

# Prompt for the token
hf_token = "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"

# Set the environment variable
os.environ['HF_TOKEN'] = hf_token
print("HF_TOKEN environment variable has been set.")
cwd = os.getcwd()
sys.path.append(cwd)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

params = {
        "STEPS": 5,
        "LEARNING_RATE": 4e-4,
        "DATASET": "og-chair-bg",
        "TRAINING_TYPE": "balanced",
        "LORA_RANK": 16,
        "SAVE_EVERY": 2,
        "CAPTION_DROPOUT_RATE": 0.00,
        "SAMPLE_EVERY": 100,
        "SAMPLING_PROMPT": [],
        "SEED": 42,
        "SAMPLING_STEPS": 1,
        "LAYERS": [
            "transformer.single_transformer_blocks.7.proj_out",
            "transformer.single_transformer_blocks.12.proj_out",
            "transformer.single_transformer_blocks.16.proj_out",
            "transformer.single_transformer_blocks.20.proj_out"
        ],
}

if params["LAYERS"]:
    lora_name = f"{params['DATASET']}-steps{params['STEPS']}-lr{params['LEARNING_RATE']}-lrank{params['LORA_RANK']}-b1_layers"
    output_path = f"{cwd}/{lora_name}"
    for i in params["LAYERS"]:
        lora_name += f"-{i.split('.')[-2]}"
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', lora_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', f'{cwd}/output'),
                    ('performance_log_every', 1000),
                    ('device', 'cuda:0'),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', params["LORA_RANK"]),
                        ('linear_alpha', 16),
                        ('network_kwargs', OrderedDict([
                            ('only_if_contains', params["LAYERS"])
                        ]))
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', params["SAVE_EVERY"]),
                        ('max_step_saves_to_keep', 4000)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', f'{cwd}/dataset/{params["DATASET"]}'),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', params["CAPTION_DROPOUT_RATE"]),
                            ('shuffle_tokens', False),
                            ('cache_latents_to_disk', True),
                            ('resolution', [512, 768, 1024])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 1),
                        ('steps', params["STEPS"]),
                        ('gradient_accumulation_steps', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('content_or_style', params["TRAINING_TYPE"]),
                        ('gradient_checkpointing', True),
                        ('noise_scheduler', 'flowmatch'),
                        ('optimizer', 'adamw8bit'),
                        ('lr', params["LEARNING_RATE"]),
                        ('skip_first_sample', True),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                        ('is_flux', True),
                        ('quantize', False),
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', params["SAMPLE_EVERY"]),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', params["SAMPLING_PROMPT"]),
                        ('neg', ''),
                        ('seed', params["SEED"]),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', params["SAMPLING_STEPS"])
                    ]))
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', lora_name),
            ('version', '1.0')
        ]))
    ])
else:
    lora_name = f"{params['DATASET']}-steps{params['STEPS']}-lr{params['LEARNING_RATE']}-lrank{params['LORA_RANK']}-b1_full-layer"
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', lora_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', f'{cwd}/output'),
                    ('performance_log_every', 1000),
                    ('device', 'cuda:0'),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', params["LORA_RANK"]),
                        ('linear_alpha', 16)
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', params["SAVE_EVERY"]),
                        ('max_step_saves_to_keep', 4000)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', f'{cwd}/dataset/{params["DATASET"]}'),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', params["CAPTION_DROPOUT_RATE"]),
                            ('shuffle_tokens', False),
                            ('cache_latents_to_disk', True),
                            ('resolution', [512, 768, 1024])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 1),
                        ('steps', params["STEPS"]),
                        ('gradient_accumulation_steps', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('content_or_style', params["TRAINING_TYPE"]),
                        ('gradient_checkpointing', True),
                        ('noise_scheduler', 'flowmatch'),
                        ('optimizer', 'adamw8bit'),
                        ('lr', params["LEARNING_RATE"]),
                        ('skip_first_sample', True),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                        ('is_flux', True),
                        ('quantize', False),
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', params["SAMPLE_EVERY"]),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', params["SAMPLING_PROMPT"]),
                        ('neg', ''),
                        ('seed', params["SEED"]),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', params["SAMPLING_STEPS"])
                    ]))
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', lora_name),
            ('version', '1.0')
        ]))
    ])
    
output_path = f"{cwd}/output/{lora_name}/"
run_job(job_to_run)

import requests

def get_username_from_token(hf_token):
    url = "https://huggingface.co/api/whoami-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("name", "Username not found")
    else:
        return f"Error: Unable to fetch username (Status Code: {response.status_code})"

username = get_username_from_token(hf_token)
repo = f"{username}/{lora_name}"

from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj=f"{output_path}/{lora_name}.safetensors",
    path_in_repo="lora.safetensors",
    repo_id=repo,
    repo_type="model",
    token=hf_token,    
)

api.upload_file(
    path_or_fileobj=f"{output_path}/config.yaml",
    path_in_repo="config.yaml",
    repo_id=repo,
    repo_type="model",
    token=hf_token
)

if os.path.exists(f"{output_path}/samples"):
    api.upload_folder(
        folder_path=f"{output_path}/samples",
        path_in_repo="./samples",  # Path in the repository (root in this case)
        repo_id=repo,
        repo_type="model",  # Change to 'dataset' if uploading to a dataset repo
        # commit_message=commit_message,
        token=hf_token
    )