import os
import sys
import torch
import getpass
from PIL import Image
from toolkit.job import run_job
from collections import OrderedDict

# Prompt for the token
hf_token = "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"
print("HF_TOKEN environment variable has been set.")

# Set the environment variable
os.environ['HF_TOKEN'] = hf_token
sys.path.append('/home/bhara/ai-toolkit')
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from collections import OrderedDict
from itertools import product

# Define your parameters
parameters_range = {
    "STEPS": [5000, 2500],
    "LEARNING_RATE": [1e-4, 4e-4, 1e-3],
    "DATASET": "og-chair-bg",
    "TRAINING_TYPE": "balanced",
    "LORA_RANK": 16,
    "SAVE_EVERY": 100,
    "CAPTION_DROPOUT_RATE": [0.00, 0.05],
    "SAMPLE_EVERY": 100,
    "SAMPLING_PROMPT": [
        "The UNST chair is positioned in a cozy, modern living room with natural wooden flooring and a large floor-to-ceiling window letting in soft daylight. The chair's sleek black wooden frame and soft beige textured upholstery complement the room's neutral-toned decor. A minimalist coffee table with a few stacked books and a potted plant is placed nearby, enhancing the inviting atmosphere. The background features a subtle off-white wall with a large abstract art piece, adding sophistication and balance to the space. The UNST chair, as the centerpiece, radiates contemporary charm and warmth.",
        "The UNST chair is elegantly placed on a spacious outdoor patio surrounded by lush greenery and vibrant flowering plants. The sleek black wooden frame contrasts beautifully with the natural tones of the beige textured upholstery and the earthy stone tile flooring. In the background, tall trees and a clear blue sky create a serene ambiance. A small side table holding a cup of coffee and a book sits next to the chair, emphasizing relaxation. The setting highlights the versatility of the UNST chair, seamlessly blending with the tranquility of outdoor spaces.",
        "The UNST chair is featured in a trendy urban loft studio with exposed brick walls and industrial-style windows that allow streams of natural light to fill the space. The chair's modern black wooden frame and beige textured upholstery stand out against the rustic yet chic backdrop. A modern floor lamp with a black metal frame is placed beside the chair, casting a warm glow over the setting. The loft is accented with minimalistic furniture, such as a sleek black metal shelf and a concrete planter, creating a perfect environment for the UNST chair to showcase its bold, contemporary design.",
        "The UNST chair is situated in a peaceful bedroom corner, creating an inviting reading nook. The chair's black wooden frame and soft beige upholstery contrast gently with the light gray walls and a plush cream-colored rug beneath. A tall bookshelf filled with neatly arranged books and decor items is placed behind the chair, while a small wooden side table holds a steaming cup of tea and a vase with fresh flowers. Warm ambient lighting from a nearby floor lamp enhances the calming atmosphere. The UNST chair completes the space with its ergonomic elegance and timeless style.",
        "The UNST chair is placed in a sophisticated office setting with dark hardwood flooring and a large glass desk nearby. The chair's sleek black wooden frame and beige textured upholstery contrast beautifully with the room’s modern, high-end decor. Behind the chair, a tall bookshelf with neatly organized files and decorative items lines the wall, while a large window offers a panoramic cityscape view. The soft lighting from a designer table lamp casts a warm glow over the space, highlighting the UNST chair as a stylish yet functional seating option for an executive workspace.",
        "The UNST chair is prominently displayed in the center of a modern art gallery with pristine white walls and polished concrete flooring. The minimalist black wooden frame and soft beige upholstery of the chair perfectly align with the clean and elegant aesthetic of the gallery. Surrounding the chair are contemporary sculptures and abstract paintings mounted on the walls, providing a creative and artistic backdrop. Subtle track lighting from above illuminates the UNST chair, making it a standout piece within this refined, artistic environment.",
        "The UNST chair is positioned on a serene coastal balcony overlooking the ocean. The sleek black wooden frame and beige textured upholstery contrast against the light wooden decking and the soft blue hues of the sky and water. A small side table with a glass of lemonade and a folded magazine sits beside the chair, while a cozy throw blanket drapes over the armrest. Gentle sunlight streams through the slats of a wooden pergola above, creating a tranquil and breezy atmosphere that highlights the UNST chair’s adaptability to outdoor coastal settings.",
        "The UNST chair is situated in a stylish industrial-themed cafe with exposed metal beams, raw brick walls, and a polished concrete floor. The chair’s modern black wooden frame and beige textured upholstery offer a striking yet harmonious contrast to the rugged industrial decor. Nearby, a wooden communal table with coffee cups and pastries adds to the inviting atmosphere. Edison-style hanging lights cast a warm glow, emphasizing the UNST chair as both a functional and aesthetic choice for a trendy, urban dining environment."
    ],
    "SEED": 42,
    "SAMPLING_STEPS": 27,
    "LAYERS": [
        "transformer.single_transformer_blocks.7.proj_out",
        "transformer.single_transformer_blocks.12.proj_out",
        "transformer.single_transformer_blocks.16.proj_out",
        "transformer.single_transformer_blocks.20.proj_out"
    ]
}

# Generate all combinations of parameters
param_combinations = list(product(
    parameters_range["STEPS"],
    parameters_range["LEARNING_RATE"],
    [parameters_range["DATASET"]],      # Only a single value
    [parameters_range["TRAINING_TYPE"]], # Only a single value
    [parameters_range["LORA_RANK"]],     # Only a single value
    [parameters_range["SAVE_EVERY"]],    # Only a single value
    parameters_range["CAPTION_DROPOUT_RATE"],  # Only a single value
    [parameters_range["SAMPLE_EVERY"]],  # Only a single value
    [parameters_range["SAMPLING_PROMPT"]], # Only a single value
    [parameters_range["SEED"]],          # Only a single value
    [parameters_range["SAMPLING_STEPS"]], # Only a single value
    [parameters_range["LAYERS"]] # Only a single value
))

# Iterate over parameter combinations and generate job configurations
for idx, combination in enumerate(param_combinations):
    # Map combination to parameter names
    params = {
        "STEPS": combination[0],
        "LEARNING_RATE": combination[1],
        "DATASET": combination[2],
        "TRAINING_TYPE": combination[3],
        "LORA_RANK": combination[4],
        "SAVE_EVERY": combination[5],
        "CAPTION_DROPOUT_RATE": combination[6],
        "SAMPLE_EVERY": combination[7],
        "SAMPLING_PROMPT": combination[8],
        "SEED": combination[9],
        "SAMPLING_STEPS": combination[10],
        "LAYERS": combination[11],
    }

    # Generate the layer name
    try:
        layers = ""
        for i in params["LAYERS"]:
            layers += f"-{i.split('.')[-2]}"
        lora_name = f"{params['DATASET']}-steps{params['STEPS']}-lr{params['LEARNING_RATE']}-lrank{params['LORA_RANK']}-b1_layers{layers}"
    except:
        lora_name = f"{params['DATASET']}-steps{params['STEPS']}-lr{params['LEARNING_RATE']}-lrank{params['LORA_RANK']}-b1_full-layer"

    # Generate job configuration
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', lora_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', '~/ai-toolkit/output'),
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
                        ('max_step_saves_to_keep', 4)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', f'/home/bhara/ai-toolkit/dataset/{params["DATASET"]}'),
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

    run_job(job_to_run)
    torch.cuda.empty_cache()

    print(idx, lora_name)