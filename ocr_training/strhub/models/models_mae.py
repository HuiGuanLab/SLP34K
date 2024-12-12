# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

import torch


class MaskedAutoencoderViT_Encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches #196    224/16 ,224/16 = 14*14

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
    
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)  

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] 
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs): 
        latent = self.forward_encoder(imgs) 
        return latent


def mae_vit_base_patch4_enc384_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_Encoder(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def mae_vit_base_patch4_enc768_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_Encoder(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def mae_vit_base_patch16_enc768_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_Encoder(
        img_size=(224, 224),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

# set recommended archs
mae_vit_base_patch4_384_32x128 = mae_vit_base_patch4_enc384_dec512d8b 
mae_vit_base_patch4_768_32x128 = mae_vit_base_patch4_enc768_dec512d8b
mae_vit_base_patch16_224x224 = mae_vit_base_patch16_enc768_dec512d8b



if __name__ == '__main__':
    net =  MaskedAutoencoderViT_Encoder(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    x = torch.randn(22,3,32,128)
    y = net(x)
    print(y.shape)