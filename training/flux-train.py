import argparse
import os
import sys
sys.path.append('../../ai-toolkit')
from toolkit.job import run_job
from collections import OrderedDict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Flux LoRA training script.")
parser.add_argument("--hf_token", type=str, required=True, help="Your Hugging Face access token.")
parser.add_argument("--project_name", type=str, required=True, help="Name of the project.")
parser.add_argument("--lora_rank", type=int, required=True, help="Rank for LoRA configuration.")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
parser.add_argument("--steps", type=int, required=True, help="Number of training steps.")
args = parser.parse_args()

# Set the environment variable for HF token
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_TOKEN"] = args.hf_token

print("HF_TOKEN environment variable has been set.")

job_to_run = OrderedDict([
    ('job', 'extension'),
    ('config', OrderedDict([
        ('name', args.project_name),
        ('process', [
            OrderedDict([
                ('type', 'sd_trainer'),
                ('training_folder', '../output'),
                ('device', 'cuda:0'),
                ('network', OrderedDict([
                    ('type', 'lora'),
                    ('linear', args.lora_rank),
                    ('linear_alpha', 16),
                    ('network_kwargs', OrderedDict([
                        ('only_if_contains', [
                            "transformer.single_transformer_blocks.7.proj_out",
                            "transformer.single_transformer_blocks.20.proj_out"
                        ])
                    ]))
                ])),
                ('save', OrderedDict([
                    ('dtype', 'float16'),
                    ('save_every', 25000),
                    ('max_step_saves_to_keep', 4)
                ])),
                ('datasets', [
                    OrderedDict([
                        ('folder_path', f'../dataset/{args.dataset_name}'),
                        ('caption_ext', 'txt'),
                        ('caption_dropout_rate', 0.05),
                        ('shuffle_tokens', False),
                        ('cache_latents_to_disk', True),
                        ('resolution', [512, 768, 1024])
                    ])
                ]),
                ('train', OrderedDict([
                    ('batch_size', args.batch_size),
                    ('steps', args.steps),
                    ('gradient_accumulation_steps', 1),
                    ('train_unet', True),
                    ('train_text_encoder', False),
                    ('content_or_style', 'balanced'),
                    ('gradient_checkpointing', True),
                    ('noise_scheduler', 'flowmatch'),
                    ('optimizer', 'adamw8bit'),
                    ('lr', 4e-4),
                    ('ema_config', OrderedDict([
                        ('use_ema', True),
                        ('ema_decay', 0.99)
                    ])),
                    ('dtype', 'bf16')
                ])),
                ('model', OrderedDict([
                    ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                    ('is_flux', True),
                    ('quantize', False)
                ])),
                ('sample', OrderedDict([
                    ('sampler', 'flowmatch'),
                    ('sample_every', 25000),
                    ('width', 1024),
                    ('height', 1024),
                    ('prompts', []),
                    ('neg', ''),
                    ('seed', 42),
                    ('walk_seed', True),
                    ('guidance_scale', 4),
                    ('sample_steps', 20)
                ]))
            ])
        ])
    ])),
    ('meta', OrderedDict([
        ('name', '[name]'),
        ('version', '1.0')
    ]))
])

run_job(job_to_run)
