# Adapted from https://github.com/facebookresearch/esm/blob/4e0ebb7a7b875ef40178cbb11e830eb5859b4180/esm/model/esm2.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified to add parameter initialization and Flash Attention
# You can extend ESM2 with forward() calling esm_forward()
# For self-attention where qkv are all from x, packing linear layer
#  is done for FlashAttention.To load weights from ESM2,
#  use upgrade_state_dict_qkv_to_packed()

from typing import Union, List, Optional
import torch
import torch.nn as nn
import esm
import re
import warnings

from transformer_modules import TransformerLayer
from modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    # use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def upgrade_state_dict(self, state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    def upgrade_state_dict_qkv_to_packed(self, state_dict):
        '''Load weights from ESM2 by packing QKV parameters'''
        for layer in range(self.num_layers):
            for wb in ['weight', 'bias']:
                params, param_names = [], []
                for qkv in ['q_proj', 'k_proj', 'v_proj']:
                    param_name = 'layers.' + str(layer) + (
                        '.self_attn.' + qkv + '.' + wb)
                    params.append(state_dict[param_name])
                    param_names.append(param_name)
                packed_name = 'layers.' + str(layer) + '.self_attn.Wqkv.' + wb
                state_dict[packed_name] = torch.cat(params, dim=0)
                for name in param_names:
                    del state_dict[name]
        return state_dict

    def downgrade_state_dict_qkv_to_unpacked(self, state_dict):
        for layer in range(self.num_layers):
            for wb in ['weight', 'bias']:
                packed_name = 'layers.' + str(layer) + '.self_attn.Wqkv.' + wb
                qkv = ['q_proj', 'k_proj', 'v_proj']
                for i in range(3):
                    param_name = 'layers.' + str(layer) + (
                        '.self_attn.' + qkv[i] + '.' + wb)
                    state_dict[param_name] = state_dict[packed_name][
                        self.embed_dim*i: self.embed_dim*(1+i), :]
                del state_dict[packed_name]
        return state_dict

    def rename_rot_to_rotary(self, state_dict, reverse=False):
        raise RuntimeWarning('rename_rot_to_rotary no longer necessary')
        # name1 = 'self_attn.rot_emb'
        # name2 = 'self_attn.rotary_emb'
        # if reverse:
        #     state_dict = self.rename_state_dict(state_dict, name2, name1)
        # else:
        #     state_dict = self.rename_state_dict(state_dict, name1, name2)
        return state_dict

    def rename_state_dict(self, state_dict, old_str, new_str):
        '''rename any matching parameters, careful'''
        pattern = re.compile(old_str)
        state_dict = {pattern.sub(new_str, name): param for name, param in state_dict.items()}
        return state_dict