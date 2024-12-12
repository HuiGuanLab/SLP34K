# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from strhub.models.maevit_infonce_plm.modules import DecoderLayer, Decoder, TokenEmbedding


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import numpy as np
import random

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from torch.optim.lr_scheduler import OneCycleLR

    
from einops import rearrange

from strhub.models.maevit_infonce_plm import clip


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class Model(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], embed_dim: int, mae_pretrained_path: str,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.coef_lr = kwargs["coef_lr"] if "coef_lr" in kwargs.keys() else 1.0
        self.coef_wd = kwargs["coef_wd"] if "coef_wd" in kwargs.keys() else 1.0
        self.image_freeze_nlayer = kwargs["image_freeze_nlayer"] if "image_freeze_nlayer" in kwargs.keys() else -1
        self.text_freeze_nlayer = kwargs["text_freeze_nlayer"] if "text_freeze_nlayer" in kwargs.keys() else -1

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        import strhub.models.models_mae as models_mae
        if img_size[0] == 32 and img_size[1] == 128:
            if embed_dim == 384:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_384_32x128')()
            elif embed_dim == 768:
                mae_model = getattr(models_mae, 'mae_vit_base_patch4_768_32x128')()
        elif img_size[0] == img_size[1] == 224:
            mae_model = getattr(models_mae, 'mae_vit_base_patch16_224x224')()

        chkpt_dir = mae_pretrained_path
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        
        mae_model.load_state_dict(checkpoint['model'], strict=False)
        self.encoder = mae_model
        
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
            
        del self.clip_model.visual   
        if self.clip_model is not None:
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False
            self.clip_model.token_embedding.eval()  

            self.clip_model.positional_embedding.requires_grad = False 

            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False 
            self.clip_model.transformer.eval() 

            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = False 
            self.clip_model.ln_final.eval() 
            
        scale = embed_dim ** -0.5
        self.proj =  nn.Parameter(scale * torch.randn(embed_dim, 512))

        self.InfoNCELoss = InfoNCELoss()


        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)
  

    def encode(self, img,labels = None):

        mae_memory = self.encoder(img)

        if labels is not None:
            labels = clip.tokenize(labels).to(mae_memory.device)
            text_features = self.clip_model.encode_text(labels) 
           
            loss = self.InfoNCELoss(mae_memory[:,0,:]@self.proj, text_features)

            return mae_memory,loss
        else:
            return mae_memory


    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape #B,tgt_len
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1)) #B*T*768
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1) #B*T*768
        tgt_query = self.dropout(tgt_query)#B*T*768
        return self.decoder(query=tgt_query, content=tgt_emb, memory=memory, query_mask=tgt_query_mask, content_mask=tgt_mask, content_key_padding_mask=tgt_padding_mask)
   
        
    def forward(self, images, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length) #25
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1 
        
        memory = self.encode(images) 

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.tokenizer.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.tokenizer.eos_id).any(dim=-1).all():
                        
                        break
            t = len(logits)                
            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self._device)
            num_steps = t
            tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == self.tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.head(tgt_out)

        return logits
        

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device) # B * (T+1)

        memory,cross_modal_loss = self.encode(images,labels) # B*197*768
        
        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt) # k*(T+1)
        tgt_in = tgt[:, :-1] #B*32
        tgt_out = tgt[:, 1:] #B*32
        # # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)#B*32

        ocr_loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm) 
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask) # B * T * 768
            logits = self.head(out).flatten(end_dim=1)  #(B * len(label+1),NUM_CLASS)
            ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        
        logits = self.head(out).flatten(end_dim=1)
        ocr_loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel += n        
        ocr_loss /= loss_numel
        loss =ocr_loss + 0.1*cross_modal_loss

        self.log('loss', loss)
        self.log('ocr_loss', ocr_loss)
        self.log('cross_modal_loss', cross_modal_loss)

        return loss
    
    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * (self.trainer.num_devices * self.batch_size) / 512  
        lr = lr_scale * self.lr

        # https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/training/main.py#L171
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n \
                            or "pos_queries" in n or "text_embed" in n
        include = lambda n, p: not exclude(n, p)

        encoder_params = list(self.encoder.named_parameters())
        enc_gain_or_bias_params = [p for n, p in encoder_params if exclude(n, p) and p.requires_grad]
        enc_rest_params = [p for n, p in encoder_params if include(n, p) and p.requires_grad]


        # decoder parameters
        decoder_params = [(n, p) for n, p in list(self.named_parameters()) if "encoder" not in n and "clip_model" not in n] 
        dec_gain_or_bias_params = [p for n, p in decoder_params if exclude(n, p) and p.requires_grad]
        dec_rest_params = [p for n, p in decoder_params if include(n, p) and p.requires_grad]


        optimizer = torch.optim.AdamW(
            [
                {"params": enc_gain_or_bias_params, "weight_decay": 0., 'lr': lr},
                {"params": enc_rest_params, "weight_decay": self.weight_decay, 'lr': lr},
                {"params": dec_gain_or_bias_params, "weight_decay": 0., 'lr': lr * self.coef_lr},
                {"params": dec_rest_params, "weight_decay": self.weight_decay * self.coef_wd, 'lr': lr * self.coef_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1.0e-6,
            )
        sched = OneCycleLR(optimizer, [lr,lr,lr * self.coef_lr, lr * self.coef_lr],
                            self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                            cycle_momentum=False)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}