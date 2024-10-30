# Sheng Wang at Feb 22 2023
import copy
import math

import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from models_vit import VisionTransformer

class _DoRA_fc(nn.Module):
    def __init__(
            self,
            old_fc: nn.Module,
            linear_a_fc: nn.Module,
            linear_b_fc: nn.Module,
            alpha = 1,
    ):
        super().__init__()
        self.old_fc = old_fc
        self.linear_a_fc = linear_a_fc
        self.linear_b_fc = linear_b_fc
        self.m = nn.Parameter(self.old_fc.weight.norm(p=2, dim=0, keepdim=True))
        self.alpha = alpha

    def forward(self, x):
        lora = self.linear_b_fc.weight @ self.linear_a_fc.weight
        combined_weights = self.old_fc.weight + self.alpha * lora
        column_norm = combined_weights.norm(p=2, dim=0, keepdim=True)
        V = combined_weights / column_norm
        new_weight = self.m * V
        return F.linear(x, new_weight, self.old_fc.bias)


class _DoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module,
            alpha = 1
    ):
        super().__init__()
        self.qkv = qkv
        self.m = nn.Parameter(self.qkv.weight.norm(p=2, dim=1, keepdim=True))
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q

        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k

        self.alpha = alpha
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
         # B,N,3*org_C
        lora_q = self.linear_b_q.weight @ self.linear_a_q.weight
        lora_v = self.linear_b_v.weight @ self.linear_a_v.weight
        lora_k = self.linear_b_k.weight @ self.linear_a_k.weight

        combined_weights_q = self.qkv.weight[:self.dim, :] + self.alpha * lora_q
        combined_weights_k = self.qkv.weight[self.dim:-self.dim, :] + self.alpha * lora_k
        combined_weights_v = self.qkv.weight[-self.dim:, :] + self.alpha * lora_v

        column_norm_q = combined_weights_q.norm(p=2, dim=0, keepdim=True)
        column_norm_v = combined_weights_v.norm(p=2, dim=0, keepdim=True)
        column_norm_k = combined_weights_k.norm(p=2, dim=0, keepdim=True)

        V_q = combined_weights_q / column_norm_q
        V_v = combined_weights_v / column_norm_v
        V_k = combined_weights_k / column_norm_k

        new_weight_q = self.m[:self.dim, :] * V_q
        new_weight_v = self.m[-self.dim:, :] * V_v
        new_weight_k = self.m[self.dim:-self.dim, :] * V_k

        copied_qkv = copy.deepcopy(self.qkv)
        copied_qkv.weight[:self.dim, :] = new_weight_q
        copied_qkv.weight[-self.dim:, :] = new_weight_v
        copied_qkv.weight[self.dim:-self.dim, :] = new_weight_k

        return copied_qkv(x)

class DoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: VisionTransformer, r: int, reduction_factor=2):
        super(DoRA_ViT_timm, self).__init__()

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

            blk.attn.qkv = _DoRA_qkv(
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

            blk.attn.proj = _DoRA_fc(
                proj_linear,
                w_a_linear_p, w_b_linear_p)

            ### fc1
            fc1_linear = blk.mlp.fc1
            w_a_linear_fc1 = nn.Linear(fc1_linear.in_features, r, bias=False)
            w_b_linear_fc1 = nn.Linear(r, fc1_linear.out_features, bias=False)
            self.w_As.append(w_a_linear_fc1)
            self.w_Bs.append(w_b_linear_fc1)
            blk.mlp.fc1 = _DoRA_fc(
                fc1_linear,
                w_a_linear_fc1, w_b_linear_fc1)

            ### fc2
            fc2_linear = blk.mlp.fc2
            w_a_linear_fc2 = nn.Linear(fc2_linear.in_features, r, bias=False)
            w_b_linear_fc2 = nn.Linear(r, fc2_linear.out_features, bias=False)
            self.w_As.append(w_a_linear_fc2)
            self.w_Bs.append(w_b_linear_fc2)
            blk.mlp.fc2 = _DoRA_fc(
                fc2_linear,
                w_a_linear_fc2, w_b_linear_fc2)


        self.reset_parameters()

        self.lora_vit = vit_model
        self.lora_vit.blocks = nn.ModuleList([blk for i, blk in enumerate(self.lora_vit.blocks) if i % reduction_factor == 0])

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