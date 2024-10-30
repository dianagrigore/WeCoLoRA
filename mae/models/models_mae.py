# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
from multiprocessing.pool import ThreadPool

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.transforms as transforms

from util.pos_embed import get_2d_sincos_pos_embed
from hog import HOGLayerC


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_mae=False, use_pos_pred=False, use_contrastive=False, use_all=False, use_contr_and_pos_pred=False,
                 use_mae_and_pos_pred=False,
                 use_positional_embedding=False,
                 use_mae_pos_emb_pos_pred=False,
                 use_cutmix_mae=False,
                 use_blended=False,
                 use_cutout_tokens_mae=False,
                 use_mae_cutout_tokens_mae=False,
                 use_hog=False,
                 use_bottleneck=False,
                 use_mae_rgb_and_hog=False,
                 use_flip_rotate=False,
                 use_hog_to_grayscale=False,
                 use_blended_hog=False,
                 use_mae_and_blended=False,
                 use_clip=False):
        super().__init__()

        # assert use_positional_embedding is True
        self.use_positional_embedding = use_positional_embedding  # For position prediction we do not use pos embedding.
        self.use_mae = use_mae
        self.use_pos_pred = use_pos_pred
        self.use_contrastive = use_contrastive
        self.use_all = use_all  # TODO: when implemented
        self.use_contr_and_pos_pred = use_contr_and_pos_pred
        self.use_mae_and_pos_pred = use_mae_and_pos_pred
        self.use_mae_pos_emb_pos_pred = use_mae_pos_emb_pos_pred
        self.use_blended = use_blended
        self.use_cutmix_mae = use_cutmix_mae
        self.use_bottleneck = use_bottleneck
        self.use_cutout_tokens_mae = use_cutout_tokens_mae
        self.use_mae_cutout_tokens_mae = use_mae_cutout_tokens_mae
        self.use_mae_rgb_and_hog = use_mae_rgb_and_hog
        self.use_flip_rotate = use_flip_rotate
        self.use_hog = use_hog
        self.use_hog_to_grayscale = use_hog_to_grayscale
        self.use_clip = use_clip
        self.use_blended_hog = use_blended_hog
        self.use_mae_and_blended = use_mae_and_blended
        self.norm_pix_loss = norm_pix_loss

        assert any([self.use_mae, self.use_pos_pred, self.use_contrastive, self.use_all,
                    self.use_contr_and_pos_pred, self.use_mae_and_pos_pred, self.use_mae_pos_emb_pos_pred,
                    self.use_blended, self.use_cutmix_mae, self.use_cutout_tokens_mae,
                    self.use_mae_cutout_tokens_mae, self.use_bottleneck, self.use_hog,
                    self.use_mae_rgb_and_hog, self.use_flip_rotate, self.use_hog_to_grayscale,
                    self.use_blended_hog,
                    self.use_clip, self.use_mae_and_blended])

        # --------------------------------------------------------------------------

        # MAE encoder specifics
        self.num_lines = img_size // patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        num_proxy_tasks = 1
        self.cls_token = nn.Parameter(torch.zeros(1, num_proxy_tasks, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        if self.use_bottleneck:
            start_feat_ = 16
            end_feat_ = embed_dim / num_heads
            factor_ = np.int32(np.ceil(np.linspace(start_feat_, end_feat_, depth)))
            hidden_dims = factor_[::-1] * num_heads
            assert hidden_dims[0] == embed_dim
            self.projection_layers = []
            for index_hid_dim in range(len(hidden_dims) - 1):
                self.projection_layers.append(nn.Linear(hidden_dims[index_hid_dim], hidden_dims[index_hid_dim + 1], bias=True))
            self.projection_layers =  nn.ModuleList(self.projection_layers)
            # TODO: should I add norm too?
        else:
            hidden_dims = [embed_dim] * depth

        self.blocks = nn.ModuleList([
            Block(hidden_dims[i], num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(hidden_dims[-1])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(hidden_dims[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # MAE-HOG decoder specifics
        decoder_embed_dim_hog = 256
        decoder_depth_hog = 4
        self.mask_token_hog = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_hog))

        self.decoder_embed_hog = nn.Linear(hidden_dims[-1], decoder_embed_dim_hog, bias=True)
        self.decoder_pos_embed_hog = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim_hog),
                                                  requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks_hog = nn.ModuleList([
            Block(decoder_embed_dim_hog, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth_hog)])

        self.decoder_norm_hog = norm_layer(decoder_embed_dim_hog)
        self.decoder_pred_hog = nn.Linear(decoder_embed_dim_hog, 27, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------

        # Position prediction specifics
        self.pos_pred_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        pos_pred_depth = 2  # Empirically set by Lili
        self.pos_pred_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(pos_pred_depth)])

        self.pos_pred_norm = norm_layer(decoder_embed_dim)
        self.pos_pred_pred = nn.Linear(decoder_embed_dim, 2, bias=True)  # two items for 2d only one item to determine the position of the patch
        self.criterion_pos_pred = nn.CrossEntropyLoss(reduction='mean')
        # --------------------------------------------------------------------------
        """
        # Contrastive branch specifics
        self.contrastive_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        contrastive_depth = 2  # Empirically set by Lili
        self.contrastive_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(contrastive_depth)])

        self.contrastive_norm = norm_layer(decoder_embed_dim)
        self.contrastive_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.criterion_contrastive = nn.BCELoss()
        # --------------------------------------------------------------------------
         

        # --------------------------------------------------------------------------
        """
        # Flip rotate
        self.flip_rotate_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        flip_rotate_depth = 1  # Empirically set by Lili
        self.flip_rotate_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(flip_rotate_depth)])

        self.flip_rotate_norm = norm_layer(decoder_embed_dim)
        self.flip_rotate_pred = nn.Linear(decoder_embed_dim, 2, bias=True)
        self.criterion_flip_rotate = nn.CrossEntropyLoss(reduction='mean')
        # --------------------------------------------------------------------------
        """

        # ------------------------------Clip features------------------------------
        
        if self.use_clip:
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
            self.tr_clip = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        self.clip_embed = nn.Linear(hidden_dims[-1], decoder_embed_dim, bias=True)
        self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding
        decoder_depth_clip = 2
        self.clip_blocks= nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth_clip)])

        self.clip_norm = norm_layer(decoder_embed_dim)
        self.clip_pred = nn.Linear(decoder_embed_dim, 512, bias=True)  # decoder to clip dimension
        """
        # --------------------------------------------------------------------------


        self.hog = HOGLayerC(nbins=9, pool=16)
        self.beta_cutmix = 1.0
        self.prob_cutmix = 0.5
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_hog = get_2d_sincos_pos_embed(self.decoder_pos_embed_hog.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed_hog.data.copy_(torch.from_numpy(decoder_pos_embed_hog).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token_hog, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, use_pos_emb=False):
        use_pos_emb_decision = use_pos_emb or self.use_positional_embedding
        # embed patches
        x = self.patch_embed(x)

        if use_pos_emb_decision:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_to_keep = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token

        if use_pos_emb_decision:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for index_bottleneck, blk in enumerate(self.blocks):
            x = blk(x)
            if self.use_bottleneck and index_bottleneck != len(self.blocks) - 1:
                x = self.projection_layers[index_bottleneck](x)

        x = self.norm(x)

        return x, mask, ids_restore, ids_to_keep

    def forward_encoder_two_images(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        if self.use_positional_embedding:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_to_keep = self.random_masking(x, mask_ratio)

        # split into two at batch level, then concatenate on the token level
        x = torch.cat(torch.split(x, x.shape[0] // 2, dim=0), dim=1)

        # shuffle the tokens.
        N = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        x = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))

        # append cls token
        cls_token = self.cls_token
        if self.use_positional_embedding:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_shuffle

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_hog(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed_hog(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token_hog.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed_hog

        # apply Transformer blocks
        for blk in self.decoder_blocks_hog:
            x = blk(x)
        x = self.decoder_norm_hog(x)

        # predictor projection
        x = self.decoder_pred_hog(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_pos_pred_head(self, x):
        # embed tokens
        x = self.pos_pred_embed(x)

        # apply Transformer blocks
        for blk in self.pos_pred_blocks:
            x = blk(x)

        x = self.pos_pred_norm(x)

        # predictor projection
        x = self.pos_pred_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_clip_head(self, x):
        # embed tokens
        x = self.clip_embed(x)

        # apply Transformer blocks
        for blk in self.clip_blocks:
            x = blk(x)

        x = self.clip_norm(x)

        # predictor projection
        x = self.clip_pred(x)

        # remove cls token
        x = x[:, 0, :]

        return x

    def forward_loss_mae(self, imgs, pred, mask, type_reconstruction="rgb"):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        if type_reconstruction == "rgb":
            target = self.patchify(imgs)
        else:
            target = imgs
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_clip(self, target, pred):
        """

        """

        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss

    def forward_loss_hog(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_cutout(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss

    def forward_mae(self, imgs, mask_ratio=0.75, freeze_encoder: bool = False):

        if freeze_encoder:
            with torch.no_grad():
                latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio)
        else:
            latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss_mae(imgs, pred, mask)
        loss_dict = {'total_loss': loss, 'mae_loss': loss}
        return loss_dict, pred, mask

    def forward_mae_and_blended(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio)
        latent = torch.cat((latent[:, :1], latent[:, 2:]), dim=1)

        pred_mae = self.forward_decoder(latent, ids_restore)
        loss_mae = self.forward_loss_mae(imgs, pred_mae, mask)

        latent_blended, mask_blended, ids_restore_blended, _ = self.forward_encoder_blended(imgs, mask_ratio)
        latent_blended = latent_blended[:, 1:]

        new_blended_latent = torch.cat((latent_blended, latent_blended), dim=0)
        pred_blended = self.forward_decoder(new_blended_latent, ids_restore)
        loss_blended = self.forward_loss_mae(imgs, pred_blended, mask)

        total_loss = 0.5 * loss_mae + 0.5 * loss_blended
        loss_dict = {'total_loss': total_loss, 'mae_loss': loss_mae, 'blended_loss': loss_blended}
        return loss_dict, (pred_mae, pred_blended), mask

    def forward_mae_rgb_and_hog(self, imgs, mask_ratio=0.75):
        all_latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio)

        latent_rgb = torch.cat((all_latent[:, :1], all_latent[:, 2:]), dim=1)
        latent_hog = all_latent[:, 1:]

        pred_rgb = self.forward_decoder(latent_rgb, ids_restore)
        loss_rgb = self.forward_loss_mae(imgs, pred_rgb, mask)

        result_hog = self.hog(imgs)
        pred_hog = self.forward_decoder_hog(latent_hog, ids_restore)
        loss_hog = self.forward_loss_hog(result_hog, pred_hog, mask)
        loss = 0.5 * loss_rgb + 0.5 * loss_hog
        loss_dict = {'total_loss': loss, 'loss_rgb': loss_rgb, 'loss_hog': loss_hog}

        return loss_dict, (pred_rgb, pred_hog), mask


    def forward_mae_rgb_and_hog_and_flip(self, imgs_, mask_ratio=0.75):
        imgs, labels = imgs_
        all_latent, mask, ids_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio)

        pred_rgb = self.forward_decoder(all_latent, ids_restore)
        loss_rgb = self.forward_loss_mae(imgs, pred_rgb, mask)

        result_hog = self.hog(imgs)
        pred_hog = self.forward_decoder_hog(all_latent, ids_restore)
        loss_hog = self.forward_loss_hog(result_hog, pred_hog, mask)

        pred_flip_rotate = self.forward_rotate_flip_head(all_latent)
        loss_flip, ground_truth = self.forward_loss_flip_rotate(labels, pred_flip_rotate, ids_to_keep)

        loss = 0.33 * loss_rgb + 0.33 * loss_hog  + 0.33 * loss_flip
        loss_dict = {'total_loss': loss,
                     'loss_rgb': loss_rgb,
                     'loss_hog': loss_hog,
                     'loss_flip': loss_flip
                     }

        return loss_dict, (pred_rgb, pred_hog, ground_truth), mask

    def foward_mae_rgb_hog_pos_pred(self, imgs, mask_ratio=0.75):
        #imgs, labels = imgs_
        all_latent, mask, ids_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio, True)

        pred_rgb = self.forward_decoder(all_latent, ids_restore)
        loss_rgb = self.forward_loss_mae(imgs, pred_rgb, mask)

        result_hog = self.hog(imgs)
        pred_hog = self.forward_decoder_hog(all_latent, ids_restore)
        loss_hog = self.forward_loss_hog(result_hog, pred_hog, mask)

        latent_pos_pred, mask_pos_pred, ids_to_restore_pos_pred, ids_to_keep_pos_pred = \
            self.forward_encoder(imgs, mask_ratio, False)
        pred_pos_pred = self.forward_pos_pred_head(latent_pos_pred)
        loss_pos_pred, ground_truth_pos = self.forward_pos_pred_loss(pred_pos_pred, ids_to_keep_pos_pred)

        loss = 0.33 * loss_rgb + 0.33 * loss_hog + 0.33 * loss_pos_pred
        loss_dict = {'total_loss': loss,
                     'loss_rgb': loss_rgb,
                     'loss_hog': loss_hog,
                     'loss_pospred': loss_pos_pred
                     }

        return loss_dict, (pred_rgb, pred_hog, ground_truth_pos), mask


    def forward_hog(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio)
        result_hog = self.hog(imgs)
        pred_hog = self.forward_decoder_hog(latent, ids_restore)
        loss_hog = self.forward_loss_hog(result_hog, pred_hog, mask)
        loss_dict = {"total_loss": loss_hog, "hog_loss": loss_hog}
        return loss_dict, pred_hog, mask

    def forward_encoder_blended(self, x, mask_ratio, use_pos_emb=False):
        use_pos_emb_decision = use_pos_emb or self.use_positional_embedding
        # embed patches
        x = self.patch_embed(x)

        if use_pos_emb_decision:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_to_keep = self.random_masking(x, mask_ratio)

        # Blend
        B = x.shape[0]
        min_value = 0.3
        max_value = 0.7
        noise = (max_value - min_value) * torch.rand(B // 2, 1, 1,
                                                     device=x.device) + min_value  # noise in [0, 1] => [min_val, max_val]
        mix_x = noise * x[:B // 2] + (1 - noise) * x[B // 2:]

        x = mix_x

        # append cls token
        cls_token = self.cls_token
        if use_pos_emb_decision:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_to_keep

    def forward_encoder_cutmix(self, x, mask_ratio, use_pos_emb=False):
        use_pos_emb_decision = use_pos_emb or self.use_positional_embedding
        # embed patches
        x = self.patch_embed(x)
        x_target = x.clone()
        if use_pos_emb_decision:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_to_keep = self.random_masking(x, mask_ratio)

        if np.random.rand() > self.prob_cutmix:
            # apply cutmix
            lam = np.random.beta(self.beta_cutmix, self.beta_cutmix)
            rand_index = torch.randperm(x.size()[0]).cuda()

            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, bbx1:bbx2, bby1:bby2] = x[rand_index, bbx1:bbx2, bby1:bby2]

        # append cls token
        cls_token = self.cls_token
        if use_pos_emb_decision:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_to_keep, x_target

    def forward_blended_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, _ = self.forward_encoder_blended(imgs, mask_ratio)
        new_latent = torch.cat((latent, latent), dim=0)

        pred = self.forward_decoder(new_latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss_mae(imgs, pred, mask)
        loss_dict = {'total_loss': loss, 'mae_loss': loss}
        return loss_dict, pred, mask

    def forward_blended_hog(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, _ = self.forward_encoder_blended(imgs, mask_ratio)
        new_latent = torch.cat((latent, latent), dim=0)
        result_hog = self.hog(imgs)
        pred = self.forward_decoder_hog(new_latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss_hog(result_hog, pred, mask)

        loss_dict = {'total_loss': loss, 'hog_blended_loss': loss}
        return loss_dict, pred, mask

    def forward_cutmix_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, _, target_x = self.forward_encoder_cutmix(imgs, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss_cutout(target_x, pred)
        loss_dict = {'total_loss': loss, 'mae_loss': loss}
        return loss_dict, pred, mask

    def forward_cutout_tokens(self, x, mask_ratio, use_pos_emb=False):
        use_pos_emb_decision = use_pos_emb or self.use_positional_embedding
        x = self.patch_embed(x)

        if use_pos_emb_decision:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

        initial_features = x
        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_to_keep = self.random_masking(x, mask_ratio)

        h = w = int(np.sqrt(x.shape[2] // 3))
        x = x.reshape(x.shape[0], x.shape[1], h, 3 * w)
        if np.random.rand() > self.prob_cutmix:
            # apply cutmix
            lam = np.random.beta(self.beta_cutmix, self.beta_cutmix)

            rand_index_patch = torch.randperm(x.shape[0]).cuda()
            rand_index_token = torch.randperm(x.shape[1]).cuda()

            bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape[1:], lam)  # batch, token, 3, h, w
            x = x[rand_index_patch, :, :, :]
            # noise = torch.tensor(np.random.normal(0, 1, x[:, :, bbx1:bbx2, bby1:bby2].size()), dtype=torch.float).cuda()
            x[:, :, bbx1:bbx2, bby1:bby2] = x[:, rand_index_token, bbx1:bbx2, bby1:bby2]
        x = x.reshape(x.shape[0], x.shape[1], h * 3 * w)
        # append cls token
        cls_token = self.cls_token
        if use_pos_emb_decision:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_to_keep, initial_features

    def forward_mae_cutout_tokens_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, ids_to_keep, initial_features = self.forward_cutout_tokens(imgs, mask_ratio)
        pred_mae = self.forward_decoder(latent, ids_restore)

        loss_mae = self.forward_loss_mae(imgs, pred_mae, mask)
        loss_cutout = self.forward_loss_mae(initial_features, pred_mae, mask, type_reconstruction="features")

        loss = loss_mae + loss_cutout
        loss_dict = {'total_loss': loss, 'mae_loss': loss_mae, 'cutout_loss': loss_cutout}
        return loss_dict, pred_mae, mask

    def forward_cutout_tokens_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, ids_to_keep, initial_features = self.forward_cutout_tokens(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss_mae(initial_features, pred, mask, type_reconstruction="features")
        loss_dict = {'total_loss': loss, 'mae_loss': loss}
        return loss_dict, pred, mask

    def forward_pos_pred_loss(self, pred, ids_to_keep):
        # Compute the ground-truth for 2D prediction
        lines = ids_to_keep // self.num_lines
        cols = ids_to_keep % self.num_lines
        ground_truth_pos = torch.stack((lines, cols), dim=2)
        loss = (pred.contiguous().view(-1) - ground_truth_pos.view(-1)) ** 2
        loss = loss.mean(dim=-1)
        return loss, ground_truth_pos

    def forward_pos_pred(self, imgs, mask_ratio=0.75):

        latent, _, _, ids_to_keep = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_pos_pred_head(latent)
        # img_tokens = self.patchify(imgs)
        # stds = torch.std(img_tokens, dim=2)
        loss, ground_truth_pos = self.forward_pos_pred_loss(pred, ids_to_keep)
        loss_dict = {'total_loss': loss, 'pos_loss': loss}
        return loss_dict, pred, ground_truth_pos


    def forward_clip(self, imgs, mask_ratio=0.75):

        latent, _, _, ids_to_keep = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_clip_head(latent)
        with torch.no_grad():
            new_img = imgs - torch.amin(imgs, dim=(1, 2, 3), keepdim=True)
            new_img = new_img / torch.amax(new_img, dim=(1, 2, 3), keepdim=True)
            target_new = self.clip_model.encode_image(self.tr_clip(new_img))
            target_old = self.clip_model.encode_image(self.tr_clip(imgs))
        import pdb; pdb.set_trace()
        loss = self.forward_loss_clip(pred, target)
        loss_dict = {'total_loss': loss, 'cli_loss': loss}
        return loss_dict, pred, target


    def forward_contrastive_head(self, x):
        # embed tokens
        x = self.contrastive_embed(x)

        # apply Transformer blocks
        for blk in self.contrastive_blocks:
            x = blk(x)

        x = self.contrastive_norm(x)

        # predictor projection
        x = self.contrastive_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_rotate_flip_head(self, x):
        # embed tokens
        x = self.flip_rotate_embed(x)

        # apply Transformer blocks
        for blk in self.flip_rotate_blocks:
            x = blk(x)

        x = self.flip_rotate_norm(x)

        # predictor projection
        x = self.flip_rotate_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def create_gt_row(self, ids_shuffle_row, num_elem, num_tokens, ground_truth):

        for index_row in range(num_elem):
            for index_col in range(index_row, num_elem):
                if ids_shuffle_row[index_row] < num_tokens and ids_shuffle_row[index_col] < num_tokens:
                    ground_truth[index_row, index_col] = 1
                    ground_truth[index_col, index_row] = 1
                elif ids_shuffle_row[index_row] >= num_tokens and ids_shuffle_row[index_col] >= num_tokens:
                    ground_truth[index_row, index_col] = 1
                    ground_truth[index_col, index_row] = 1
                else:
                    ground_truth[index_row, index_col] = 0
                    ground_truth[index_col, index_row] = 0

    def forward_contrastive(self, imgs, mask_ratio, return_predictions=False):
        latent, mask, ids_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio)
        predictions_unnorm = self.forward_contrastive_head(latent)

        half_num_tokens = predictions_unnorm.shape[1] // 2
        first_half = predictions_unnorm[:, :half_num_tokens].mean(dim=1)
        second_half = predictions_unnorm[:, half_num_tokens:].mean(dim=1)

        first_half = torch.nn.functional.normalize(first_half, dim=-1)
        second_half = torch.nn.functional.normalize(second_half, dim=-1)

        total = torch.mm(first_half, torch.transpose(second_half, 0, 1)) / 0.05

        nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        nce = 0.01 * nce
        c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0),  torch.arange(0, total.shape[0], device=first_half.device))) / total.shape[0]

        loss_dict = {'total_loss': nce, 'const_loss': nce, 'cons acc': c_acc}
        return loss_dict, None, None

        # # Compute the loss.
        # num_tokens = predictions_unnorm.shape[1] // 2
        # batch_size = predictions_unnorm.shape[0]
        #
        # predictions = predictions_unnorm / predictions_unnorm.norm(dim=2, keepdim=True)
        # attention_prediction = []
        # ground_truth = []
        # for index in range(len(predictions)):
        #     result = predictions[index] @ predictions[index].T
        #     attention_prediction.append(result)
        #     ids_shuffle_row = ids_shuffle[index].unsqueeze(1)
        #     gt = (ids_shuffle_row < num_tokens) * (ids_shuffle_row < num_tokens).T * 1 + (ids_shuffle_row >= num_tokens) * (ids_shuffle_row >= num_tokens).T * 1
        #     ground_truth.append(gt)
        #
        # attention_prediction = torch.stack(attention_prediction).to(predictions.device)
        # ground_truth = torch.stack(ground_truth).to(predictions.device)
        #
        # loss = (attention_prediction.contiguous().view(-1) - ground_truth.view(-1)) ** 2
        #
        # loss = loss.mean(dim=-1)
        # loss_dict = {'total_loss': loss, 'contr_loss': loss}
        # if return_predictions:
        #     return loss_dict, attention_prediction, ground_truth, predictions_unnorm
        # else:
        #     return loss_dict, attention_prediction, ground_truth

    def forward_contrastive_and_pos_pred(self, imgs, mask_ratio):
        # TODO: redo it to do only a FWD pass
        loss_contrastive, attention_prediction, ground_truth = self.forward_contrastive(imgs, mask_ratio)
        loss_pos_pred, pred, ground_truth_pos = self.forward_pos_pred(imgs, mask_ratio)

        total_loss = loss_contrastive + loss_pos_pred
        loss_dict = {'total_loss': total_loss,
                     'contr_loss': loss_contrastive,
                     'pos_loss': loss_pos_pred}
        return loss_dict, (attention_prediction, pred), (ground_truth, ground_truth_pos)

    def forward_mae_and_pos_pred(self, imgs, mask_ratio):
        latent, mask, ids_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio)

        pred_mae = self.forward_decoder(latent, ids_restore)
        loss_mae = self.forward_loss_mae(imgs, pred_mae, mask)

        pred_pos_pred = self.forward_pos_pred_head(latent)
        loss_pos_pred, ground_truth_pos = self.forward_pos_pred_loss(pred_pos_pred, ids_to_keep)

        total_loss = loss_mae + loss_pos_pred
        loss_dict = {'total_loss': total_loss,
                     'mae_loss': loss_mae,
                     'pos_loss': loss_pos_pred}

        return loss_dict, (pred_mae, pred_pos_pred), ground_truth_pos

    def forward_loss_flip_rotate(self, ground_truth, pred, ids_to_keep):

        ground_truth_ = torch.gather(ground_truth, dim=1, index=ids_to_keep)
        loss = self.criterion_flip_rotate(pred.flatten(start_dim=0, end_dim=1), ground_truth_.view(-1))
        return loss, ground_truth_

    def forward_mae_flip_rotate(self, imgs_, mask_ratio):
        imgs, labels = imgs_
        latent, mask, ids_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio)

        pred_flip_rotate = self.forward_rotate_flip_head(latent)
        loss, ground_truth = self.forward_loss_flip_rotate(labels, pred_flip_rotate, ids_to_keep)

        loss_dict = {'total_loss': loss,
                     'loss_flip_rotate': loss}

        return loss_dict, pred_flip_rotate, ground_truth


    def forward_mae_pos_emb_pos_pred(self, imgs, mask_ratio):
        latent_mae, mask_mae, ids_restore_mae, ids_to_keep_mae = self.forward_encoder(imgs, mask_ratio, True)
        latent, mask, ids_to_restore, ids_to_keep = self.forward_encoder(imgs, mask_ratio, False)

        pred_mae = self.forward_decoder(latent_mae, ids_restore_mae)
        loss_mae = self.forward_loss_mae(imgs, pred_mae, mask_mae)

        pred_pos_pred = self.forward_pos_pred_head(latent)
        loss_pos_pred, ground_truth_pos = self.forward_pos_pred_loss(pred_pos_pred, ids_to_keep)

        total_loss = loss_mae + loss_pos_pred
        loss_dict = {'total_loss': total_loss,
                     'mae_loss': loss_mae,
                     'pos_loss': loss_pos_pred}

        return loss_dict, (pred_mae, pred_pos_pred), ground_truth_pos

    def forward(self, imgs, mask_ratio=0.75, freeze_encoder: bool = False):
        if self.use_mae and self.use_hog and self.use_flip_rotate:
            return self.forward_mae_rgb_and_hog_and_flip(imgs, mask_ratio)
        elif self.use_mae and self.use_hog and self.use_pos_pred:
            return self.foward_mae_rgb_hog_pos_pred(imgs, mask_ratio)
        elif self.use_mae:
            return self.forward_mae(imgs, mask_ratio, freeze_encoder)
        elif self.use_pos_pred:
            return self.forward_pos_pred(imgs, mask_ratio)
        elif self.use_contrastive:
            return self.forward_contrastive(imgs, mask_ratio)
        elif self.use_blended:
            return self.forward_blended_mae(imgs, mask_ratio)
        elif self.use_cutmix_mae:
            return self.forward_cutmix_mae(imgs, mask_ratio)
        elif self.use_cutout_tokens_mae:
            return self.forward_cutout_tokens_mae(imgs, mask_ratio)
        elif self.use_contr_and_pos_pred:
            return self.forward_contrastive_and_pos_pred(imgs, mask_ratio)
        elif self.use_mae_and_pos_pred:
            return self.forward_mae_and_pos_pred(imgs, mask_ratio)
        elif self.use_mae_pos_emb_pos_pred:
            return self.forward_mae_pos_emb_pos_pred(imgs, mask_ratio)
        elif self.use_mae_cutout_tokens_mae:
            return self.forward_mae_cutout_tokens_mae(imgs, mask_ratio)
        elif self.use_hog:
            return self.forward_hog(imgs, mask_ratio)
        elif self.use_blended_hog:
            return self.forward_blended_hog(imgs, mask_ratio)
        elif self.use_mae_rgb_and_hog:
            return self.forward_mae_rgb_and_hog(imgs, mask_ratio)
        elif self.use_flip_rotate:
            return self.forward_mae_flip_rotate(imgs, mask_ratio)
        elif self.use_clip:
            return self.forward_clip(imgs, mask_ratio)
        elif self.use_mae_and_blended:
            return self.forward_mae_and_blended(imgs, mask_ratio)
        else:
            raise ValueError('Not implemented yet!')


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2