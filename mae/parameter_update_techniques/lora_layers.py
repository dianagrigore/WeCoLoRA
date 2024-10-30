# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

from models_vit import VisionTransformer


class _LoRA_qkv_timm(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class _LoRA_fc(nn.Module):
    def __init__(
            self,
            old_fc: nn.Module,
            linear_a_fc: nn.Module,
            linear_b_fc: nn.Module,
    ):
        super().__init__()
        self.old_fc = old_fc
        self.linear_a_fc = linear_a_fc
        self.linear_b_fc = linear_b_fc

    def forward(self, x):
        fc = self.old_fc(x)
        new_fc = self.linear_b_fc(self.linear_a_fc(x))

        return fc + new_fc

class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module
    ):
        super().__init__()
        self.qkv = qkv

        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q

        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k

        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        new_k = self.linear_b_k(self.linear_a_k(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, self.dim:-self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: VisionTransformer, r: int, reduction_factor=2):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.lora_layer = list(range(0, len(vit_model.blocks), reduction_factor))

        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            # TODO: why not removing the unused layers in the beginning?
            if t_layer_i not in self.lora_layer:
                continue

            ### qkv
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                w_a_linear_k,
                w_b_linear_k,
            )

            ### proj
            proj_linear = blk.attn.proj
            w_a_linear_p = nn.Linear(self.dim, r, bias=False)
            w_b_linear_p = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_p)
            self.w_Bs.append(w_b_linear_p)
            blk.attn.proj = _LoRA_fc(
                proj_linear,
                w_a_linear_p, w_b_linear_p)

            ### fc1
            fc1_linear = blk.mlp.fc1
            w_a_linear_fc1 = nn.Linear(fc1_linear.in_features, r, bias=False)
            w_b_linear_fc1 = nn.Linear(r, fc1_linear.out_features, bias=False)
            self.w_As.append(w_a_linear_fc1)
            self.w_Bs.append(w_b_linear_fc1)
            blk.mlp.fc1 = _LoRA_fc(
                fc1_linear,
                w_a_linear_fc1, w_b_linear_fc1)

            ### fc2
            fc2_linear = blk.mlp.fc2
            w_a_linear_fc2 = nn.Linear(fc2_linear.in_features, r, bias=False)
            w_b_linear_fc2 = nn.Linear(r, fc2_linear.out_features, bias=False)
            self.w_As.append(w_a_linear_fc2)
            self.w_Bs.append(w_b_linear_fc2)
            blk.mlp.fc2 = _LoRA_fc(
                fc2_linear,
                w_a_linear_fc2, w_b_linear_fc2)


        self.reset_parameters()

        self.lora_vit = vit_model
        self.lora_vit.blocks = nn.ModuleList([blk for i, blk in enumerate(self.lora_vit.blocks) if i % reduction_factor == 0])


    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        #assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit.forward(x)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.lora_vit.forward_features(x)

    def forward_features_lora(self, x: Tensor, mask_already_applied=True) -> Tensor:
        return self.lora_vit.forward_features_lora(x, mask_already_applied=mask_already_applied)

    def forward_features_kd(self, x: Tensor) -> Tensor:
        return self.lora_vit.forward_features_kd(x)