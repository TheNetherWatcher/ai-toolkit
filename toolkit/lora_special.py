import os
import sys
from typing import List

import torch
from .paths import SD_SCRIPTS_ROOT

sys.path.append(SD_SCRIPTS_ROOT)

from networks.lora import LoRANetwork, LoRAModule, get_block_index


class LoRASpecialNetwork(LoRANetwork):
    _multiplier: float = 1.0
    is_active: bool = False

    def __init__(
            self,
            text_encoder,
            unet,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            conv_lora_dim=None,
            conv_alpha=None,
            block_dims=None,
            block_alphas=None,
            conv_block_dims=None,
            conv_block_alphas=None,
            modules_dim=None,
            modules_alpha=None,
            module_class=LoRAModule,
            varbose=False,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        # call the parent of the parent we are replacing (LoRANetwork) init
        super(LoRANetwork, self).__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            if self.conv_lora_dim is not None:
                print(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
            prefix = LoRANetwork.LORA_PREFIX_UNET if is_unet else LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                block_idx = get_block_index(lora_name)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (
                                        self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                            )
                            loras.append(lora)
            return loras, skipped

        self.text_encoder_loras, skipped_te = create_modules(False, text_encoder,
                                                             LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(True, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            print(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            # doesnt work on new diffusers. TODO make sure we are not missing something
            # assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        self._multiplier = value
        self._update_lora_multiplier()

    def _update_lora_multiplier(self):

        if self.is_active:
            if hasattr(self, 'unet_loras'):
                for lora in self.unet_loras:
                    lora.multiplier = self._multiplier
            if hasattr(self, 'text_encoder_loras'):
                for lora in self.text_encoder_loras:
                    lora.multiplier = self._multiplier
        else:
            if hasattr(self, 'unet_loras'):
                for lora in self.unet_loras:
                    lora.multiplier = 0
            if hasattr(self, 'text_encoder_loras'):
                for lora in self.text_encoder_loras:
                    lora.multiplier = 0

    def __enter__(self):
        self.is_active = True
        self._update_lora_multiplier()

    def __exit__(self, exc_type, exc_value, tb):
        self.is_active = False
        self._update_lora_multiplier()

    def force_to(self, device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)