"""Cream
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license"""
import copy
import math
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import timm
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import resize
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBartPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import _expand_mask
from transformers.models.mbart.modeling_mbart import MBartDecoder, MBartEncoderLayer


def get_pos_embed(d_model: int, length: int, padding_idx: int = 0):
    """def get_pos_embed(d_model: int, length: int, padding_idx: int = 0):

    This method is designed to return a nn.Embedding module for posional encoding."""
    if padding_idx > 100:
        print(
            "[Warning] The function `get_pos_embed` is built upon an assumption that `padding_idx` has a small index number."
        )
    embedding = nn.Embedding(length + padding_idx + 1, d_model, padding_idx=padding_idx)
    if not is_deepspeed_zero3_enabled():
        with torch.no_grad():
            position = torch.arange(0, length).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
            embedding.weight[padding_idx + 1 :, 0::2] = torch.sin(position.float() * div_term)
            embedding.weight[padding_idx + 1 :, 1::2] = torch.cos(position.float() * div_term)
    embedding.weight.requires_grad = True
    return embedding


class PrefixLMAttention:
    """class PrefixLMAttention:
    def __init__(self, prefix_length=224):

    This class is designed to make decoder modules to have a bi-directional attention in a prefix prompt."""

    def __init__(self, prefix_length=224):
        super().__init__()
        self.prefix_length = prefix_length

    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int,
        prefix_length: int,
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        if prefix_length > 0:
            mask.masked_fill_(mask_cond < prefix_length, 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                prefix_length=self.prefix_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class CreamConfig(PretrainedConfig):
    """class CreamConfig(PretrainedConfig):
    def __init__(self, encoder_ffn_dim: int = 4096, encoder_attention_heads: int = 16, decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16, encoder_layerdrop: float = 0.0, decoder_layerdrop: float = 0.0,
        use_cache: bool = True, is_encoder_decoder: bool = True, activation_function: str = "gelu", dropout: float = 0.1,
        attention_dropout: float = 0.0, activation_dropout: float = 0.0, init_std: float = 0.02,
        classifier_dropout: float = 0.0, scale_embedding: bool = False, pad_token_id: int = 1, bos_token_id: int = 0,
        eos_token_id: int = 2, forced_eos_token_id: int = 2, patch_size: int = 14,
        vision_input_norm_mean: List[float] = [0.5, 0.5, 0.5], vision_input_norm_std: List[float] = [0.5, 0.5, 0.5],
        num_heads: int = 16, vision_d_model: int = 1024, aux_d_model: int = 1024, d_model: int = 1024,
        max_patches: int = 3072, max_enc_position_embeddings: int = 1024, max_position_embeddings: int = 128,
        vision_layers: int = 18, encoder_layers: int = 12, decoder_layers: int = 12, cl_enabled: bool = False,
        llm_integration_enabled: bool = False, llm_vision_query_length: int = 224,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        vocab_size: int = 57522, tie_word_embeddings: bool = True, num_aux_types: int = 2,
        vison_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "vit_large_patch14_clip_224.laion2b",
        aux_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "hyunwoongko/asian-bart-ecjk",
        decoder_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "hyunwoongko/asian-bart-ecjk",
        tokenizer_name_or_path: Union[str, bytes, os.PathLike] = "gwkrsrch/creamdonut-tokenizer", **kwargs
    ):

    This class is designed to contain and control the configurations of CreamModel."""

    model_type = "cream"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        encoder_ffn_dim: int = 4096,
        encoder_attention_heads: int = 16,
        decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,
        activation_function: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        classifier_dropout: float = 0.0,
        scale_embedding: bool = False,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        forced_eos_token_id: int = 2,
        patch_size: int = 14,
        vision_input_norm_mean: List[float] = [0.5, 0.5, 0.5],
        vision_input_norm_std: List[float] = [0.5, 0.5, 0.5],
        num_heads: int = 16,
        vision_d_model: int = 1024,
        aux_d_model: int = 1024,
        d_model: int = 1024,
        max_patches: int = 3072,
        max_enc_position_embeddings: int = 1024,
        max_position_embeddings: int = 128,
        vision_layers: int = 18,
        encoder_layers: int = 12,
        decoder_layers: int = 12,
        cl_enabled: bool = False,
        llm_integration_enabled: bool = False,
        llm_vision_query_length: int = 224,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        vocab_size: int = 57522,
        tie_word_embeddings: bool = True,
        num_aux_types: int = 2,
        vison_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "vit_large_patch14_clip_224.laion2b",
        aux_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "hyunwoongko/asian-bart-ecjk",
        decoder_pretrained_name_or_path: Union[str, bytes, os.PathLike] = "hyunwoongko/asian-bart-ecjk",
        tokenizer_name_or_path: Union[str, bytes, os.PathLike] = "gwkrsrch/creamdonut-tokenizer",
        **kwargs,
    ):
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding
        self.patch_size = patch_size
        self.vision_input_norm_mean = vision_input_norm_mean
        self.vision_input_norm_std = vision_input_norm_std
        self.num_heads = num_heads
        self.d_model = d_model
        self.vision_d_model = vision_d_model
        self.aux_d_model = aux_d_model
        self.max_patches = max_patches
        self.vision_layers = vision_layers
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.vison_pretrained_name_or_path = vison_pretrained_name_or_path
        self.max_enc_position_embeddings = max_enc_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.cl_enabled = cl_enabled
        self.num_aux_types = num_aux_types
        self.llm_integration_enabled = llm_integration_enabled
        self.llm_vision_query_length = llm_vision_query_length
        self.aux_pretrained_name_or_path = aux_pretrained_name_or_path
        self.decoder_pretrained_name_or_path = decoder_pretrained_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        if self.llm_integration_enabled and self.llm_vision_query_length >= self.max_position_embeddings:
            raise ValueError("[Warning] Set a larger visual soft prompt length than max_position_embeddings.")
        if self.max_position_embeddings < 32:
            raise ValueError("[Warning] Please consider to set a larger max length to the decoder.")


class PatchEmbed(nn.Module):
    """class PatchEmbed(nn.Module):
    def __init__(self,
        img_size: int = None, patch_size=14, in_chans=3,
        embed_dim=1024, bias=False, dynamic_img_pad: bool = False
    ):

    This class is designed to encode the input image patches with a linear mapping (nn.Linear)."""

    def __init__(
        self,
        img_size: int = None,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        bias=False,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = 0
        self.proj = nn.Linear(in_chans * patch_size[0] * patch_size[1], embed_dim, bias=bias)

    def forward(self, x):
        return self.proj(x)


class CreamVisionEncoder(nn.Module):
    """class CreamVisionEncoder(nn.Module):
    def __init__(self, patch_size, num_heads, max_patches, embed_dim, vision_layers, output_dim,
        input_norm_mean=[0.5, 0.5, 0.5], input_norm_std=[0.5, 0.5, 0.5],
        model: Optional[VisionTransformer] = None
    ):

    This class is designed to implement a vision encoder of Cream."""

    def __init__(
        self,
        patch_size,
        num_heads,
        max_patches,
        embed_dim,
        vision_layers,
        output_dim,
        input_norm_mean=[0.5, 0.5, 0.5],
        input_norm_std=[0.5, 0.5, 0.5],
        model: Optional[VisionTransformer] = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_patches = max_patches
        self.embed_dim = embed_dim
        self.vision_layers = vision_layers
        self.output_dim = output_dim

        self.to_tensor_and_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(input_norm_mean, input_norm_std),
            ]
        )
        self.patchify = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.model = (
            VisionTransformer(
                num_classes=0,
                embed_dim=self.embed_dim,
                patch_size=self.patch_size,
                num_heads=self.num_heads,
                depth=self.vision_layers,
                class_token=False,
                no_embed_class=True,
                embed_layer=PatchEmbed,
                weight_init="skip",
                global_pool="",
            )
            if model is None
            else model
        )
        del self.model.pos_embed
        self.model.x_pos_embed = get_pos_embed(self.embed_dim, self.max_patches, padding_idx=0)
        self.model.y_pos_embed = get_pos_embed(self.embed_dim, self.max_patches, padding_idx=0)

        if self.embed_dim != self.output_dim:
            self.connector = nn.Linear(self.embed_dim, self.output_dim)
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.connector_norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pos, x_pos = x[:, :, 0].long(), x[:, :, 1].long()
        x = self.model.patch_embed(x[:, :, 2:])
        x = x + self.model.y_pos_embed(y_pos) + self.model.x_pos_embed(x_pos)
        x = self.model.pos_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        if self.embed_dim != self.output_dim:
            x = self.connector(x)
            x = self.connector_norm(x)
        return x

    def prepare_input(
        self,
        img: PIL.Image.Image,
        max_patches: int = 0,
        verbose: bool = False,
    ):
        """def prepare_input(
            self, img: PIL.Image.Image, max_patches: int = 0, verbose: bool = False
        ):

        This method is designed to convert an input PIL image into a set of patchyfied tensors."""
        if max_patches < 1:
            max_patches = self.max_patches

        img = img.convert("RGB")

        image_width, image_height = img.size
        patch_height, patch_width = (
            (self.patch_size, self.patch_size) if isinstance(self.patch_size, int) else self.patch_size
        )
        if verbose:
            print(f"image_height, image_width : {image_height, image_width}")
            print(f"patch_height, patch_width : {patch_height, patch_width}")
        scale = np.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = int(
            np.max(
                (
                    np.min(
                        (
                            np.floor(scale * image_height / patch_height),
                            max_patches,
                        )
                    ),
                    1,
                )
            )
        )
        num_feasible_cols = int(
            np.max(
                (
                    np.min(
                        (
                            np.floor(scale * image_width / patch_width),
                            max_patches,
                        )
                    ),
                    1,
                )
            )
        )
        resized_height = int(np.max((num_feasible_rows * patch_height, 1)))
        resized_width = int(np.max((num_feasible_cols * patch_width, 1)))

        img = resize(img=img, size=(resized_height, resized_width), antialias=True)
        img = self.to_tensor_and_normalize(img)
        patches = self.patchify(img.unsqueeze(0))
        patches = patches.transpose(1, 2).squeeze()

        if verbose:
            print(patches.size())

        row_ids = torch.tile(torch.arange(num_feasible_rows).unsqueeze(1), (1, num_feasible_cols)).reshape(-1, 1)
        col_ids = torch.tile(torch.arange(num_feasible_cols).unsqueeze(0), (num_feasible_rows, 1)).reshape(-1, 1)

        row_ids[row_ids >= (self.max_patches - 1)] = self.max_patches - 1
        col_ids[col_ids >= (self.max_patches - 1)] = self.max_patches - 1

        row_ids += 1
        col_ids += 1

        input_tensors = torch.concat((row_ids, col_ids, patches), -1)

        attention_mask = torch.ones(max_patches, dtype=torch.int)
        if (num_feasible_rows * num_feasible_cols) >= max_patches:
            input_tensors = input_tensors[:max_patches]
        else:
            input_tensors = F.pad(
                input_tensors,
                (0, 0, 0, max_patches - (num_feasible_rows * num_feasible_cols)),
            )
            attention_mask[num_feasible_rows * num_feasible_cols :] = 0

        return (
            input_tensors,
            attention_mask,
            {
                "resized_height": resized_height,
                "resized_width": resized_width,
                "image_height": image_height,
                "image_width": image_width,
                "num_feasible_rows": num_feasible_rows,
                "num_feasible_cols": num_feasible_cols,
                "patch_height": patch_height,
                "patch_width": patch_width,
            },
        )


class MBartEncoderBackbone(MBartPreTrainedModel):
    """class MBartEncoderBackbone(MBartPreTrainedModel):
     def __init__(self, config: PretrainedConfig):

    This class is designed to implement a MBartEncoder-based backbone network that can be used in the Auxiliary Encoder.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.layerdrop = config.encoder_layerdrop
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        last_hidden_state = hidden_states

        if output_hidden_states:
            encoder_states = encoder_states + (last_hidden_state,)

        if not return_dict:
            return tuple(v for v in [last_hidden_state, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CreamAuxilaryEncoder(nn.Module):
    """class CreamAuxilaryEncoder(nn.Module):
    def __init__(
        self, config: CreamConfig, embed_tokens: Optional[nn.Embedding] = None, encoder: Optional[nn.Module] = None
    ):

    This class is designed to implement an auxiliary encoder of Cream."""

    def __init__(
        self,
        config: CreamConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        encoder: Optional[MBartEncoderBackbone] = None,
    ):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.num_aux_types = config.num_aux_types
        embed_dim = config.aux_d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_enc_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, embed_dim, self.padding_idx) if embed_tokens is None else embed_tokens
        )
        if self.num_aux_types > 1:
            self.embed_types = nn.Embedding(
                self.num_aux_types + 1,
                embed_dim,
                0,
            )
        self.x_pos_embed = get_pos_embed(embed_dim, config.max_patches, padding_idx=config.pad_token_id)
        self.y_pos_embed = get_pos_embed(embed_dim, config.max_patches, padding_idx=config.pad_token_id)
        self.encoder = encoder
        if encoder is None:
            mbart_encoder_config = copy.deepcopy(config)
            mbart_encoder_config.d_model = embed_dim
            self.encoder = MBartEncoderBackbone(mbart_encoder_config)
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_x_pos_ids: torch.LongTensor = None,
        input_y_pos_ids: torch.LongTensor = None,
        input_type_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_textread=None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        x_embeds = self.x_pos_embed(input_x_pos_ids)
        y_embeds = self.y_pos_embed(input_y_pos_ids)
        hidden_states = inputs_embeds + x_embeds + y_embeds

        if self.num_aux_types > 1:
            hidden_states += self.embed_types(input_type_ids)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        if head_mask is not None:
            raise ValueError("[Warning] CreamAuxilaryEncoder.encoder do not use head_mask.")

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        encoder_states = encoder_outputs[1] if output_hidden_states else None
        all_attentions = encoder_outputs[2] if output_attentions else None

        last_hidden_state = self.layer_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [last_hidden_state, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class MLP(nn.Module):
    """class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):

    This class is designed to implement a MLP module. Cream used this in the contrastive learning framework."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@dataclass
class CreamSeq2SeqLMOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class CreamModel(PreTrainedModel):
    """class CreamModel(PreTrainedModel):
    def __init__(self, config: CreamConfig, image_encoder: Optional[CreamVisionEncoder] = None,
        tokenizer: PreTrainedTokenizer = None, aux_encoder: Optional[CreamAuxilaryEncoder] = None,
        text_decoder: Optional[MBartDecoder] = None, embed_tokens: Optional[nn.Embedding] = None,
        skip_custom_init: bool = False
    ):

    This class is designed to implement the Cream (Contrastive Reading Model). To enable customization,
    this class could receive modules: `image_encoder`, `tokenizer`, `aux_encoder`, `text_decoder`, and `embed_tokens`.
    """

    config_class = CreamConfig
    base_model_prefix = "cream"

    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
    ]

    def __init__(
        self,
        config: CreamConfig,
        image_encoder: Optional[CreamVisionEncoder] = None,
        tokenizer: PreTrainedTokenizer = None,
        aux_encoder: Optional[CreamAuxilaryEncoder] = None,
        text_decoder: Optional[MBartDecoder] = None,
        embed_tokens: Optional[nn.Embedding] = None,
        skip_custom_init: bool = False,
    ):
        super().__init__(config)

        if tokenizer is not None:
            print(
                "[Info] a custom `tokenizer` is provided."
                + " Please check the tokenizer setting if it is okay with the `aux_encoder` and `text_decoder`."
            )

        if not skip_custom_init and not (
            image_encoder is None and aux_encoder is None and text_decoder is None and embed_tokens is None
        ):
            print(
                "[Info] There is a custom module and `skip_custom_init` is not set to True."
                + " Please check the method `custom_init` is okay with the target modules."
            )

        self.config = config
        self.ignore_id = -100
        self.mode = "vl"

        if config.vision_layers == 0 and config.encoder_layers == 0:
            raise ValueError("[Warning] Either image_encoder or aux_encoder should have at least one layer.")
        elif config.vision_layers == 0:
            self.mode = "l"
            print("[Info] This model is a text-to-text-encoder-decoder model.")
        elif config.encoder_layers == 0:
            self.mode = "v"
            print("[Info] This model is a simple image-to-text model without the use of any auxiliary information.")

        self.image_encoder = (
            CreamVisionEncoder(
                patch_size=config.patch_size,
                num_heads=config.num_heads,
                max_patches=config.max_patches,
                embed_dim=config.vision_d_model,
                vision_layers=config.vision_layers,
                output_dim=config.d_model,
                input_norm_mean=config.vision_input_norm_mean,
                input_norm_std=config.vision_input_norm_std,
            )
            if image_encoder is None
            else image_encoder
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(self.config.tokenizer_name_or_path, use_fast=False)
            if tokenizer is None
            else tokenizer
        )
        assert self.config.vocab_size >= len(self.tokenizer)
        if self.config.vocab_size != len(self.tokenizer):
            print(
                f"[Warning] `self.config.vocab_size` is different with `len(self.tokenizer)`. "
                + "Please check the vocabulary setting in tokenizer to avoid any mistake."
            )
        if self.config.aux_pretrained_name_or_path != self.config.tokenizer_name_or_path and (
            "gwkrsrch/creamdonut-tokenizer" != self.config.tokenizer_name_or_path
            and "hyunwoongko/asian-bart-ecjk" != self.config.aux_pretrained_name_or_path
        ):
            print(
                "[Info] The current `aux_pretrained_name_or_path` and `tokenizer_name_or_path` are not the same. "
                + "Please check the vocabulary settings to avoid any mistake."
            )

        if self.config.encoder_layers == 0:
            self.aux_encoder = None
        else:
            self.aux_encoder = (
                CreamAuxilaryEncoder(self.config, embed_tokens=embed_tokens) if aux_encoder is None else aux_encoder
            )

        self.text_decoder = (
            MBartDecoder(self.config, embed_tokens=embed_tokens) if text_decoder is None else text_decoder
        )

        if self.config.llm_integration_enabled:
            print(
                f"[Info] `llm_integration_enabled` is set to True. This will set a bi-directional attention flow in the auxilary encoder."
            )
            self.text_decoder._prepare_decoder_attention_mask = PrefixLMAttention(
                prefix_length=self.config.max_position_embeddings
            )._prepare_decoder_attention_mask
            self.query_tokens = nn.Parameter(torch.zeros(1, self.config.llm_vision_query_length, self.config.d_model))
        else:
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.text_decoder.embed_tokens.weight

        self.ce_loss = torch.nn.CrossEntropyLoss()

        if self.config.cl_enabled:
            self.projector = MLP(config.d_model, config.d_model, int(config.d_model / 8))

        if image_encoder is None and aux_encoder is None and text_decoder is None:
            super().post_init()

        if not self.config.name_or_path and not skip_custom_init:
            self.custom_init()

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def custom_init(self):
        print(
            "[Info] The original `custom_init` method is called."
            + f" This will initilize the Cream model with the following weights:"
            + f" {self.config.vison_pretrained_name_or_path} and {self.config.aux_pretrained_name_or_path}"
        )

        print(f"[Info] initialize the VisionEncoder with {self.config.vison_pretrained_name_or_path}")
        pretrained = timm.create_model(self.config.vison_pretrained_name_or_path, pretrained=True)
        state_dict = pretrained.state_dict()
        new_state_dict = self.image_encoder.model.state_dict()
        for x in set(state_dict.keys()).union(set(new_state_dict.keys())):
            if x not in new_state_dict or x not in state_dict or x in {"pos_embed", "patch_embed.proj.weight"}:
                print("Unmatched parameter name: ", x)
            else:
                new_state_dict[x] = state_dict[x]
        self.image_encoder.model.load_state_dict(new_state_dict)
        del state_dict
        del new_state_dict
        del pretrained

        assert self.config.decoder_pretrained_name_or_path == self.config.aux_pretrained_name_or_path, print(
            "[Warning] Please modify the `custom_init` method if the auxiliary encoder and text decoder are initialized from different sources."
        )

        pretrained = AutoModelForSeq2SeqLM.from_pretrained(self.config.aux_pretrained_name_or_path).model
        for module_name, pretrained_module, new_module in [
            ("AuxEncoder", pretrained.encoder, self.aux_encoder),
            ("CreamDecoder", pretrained.decoder, self.text_decoder),
        ]:
            if new_module is not None:
                print(f"[Info] initialize the {module_name} with {self.config.aux_pretrained_name_or_path}.")
                state_dict = pretrained_module.state_dict()
                new_state_dict = new_module.state_dict()
                if module_name == "AuxEncoder":
                    state_dict = dict(
                        ("encoder." + k, v) if k[:6] == "layers" else (k, v) for k, v in state_dict.items()
                    )
                for x in set(state_dict.keys()).union(set(new_state_dict.keys())):
                    if x not in new_state_dict:
                        print("[Info] Unmatched parameter name (the current instance does not have this): ", x)
                    elif x not in state_dict:
                        print("[Info] Unmatched parameter name (the loaded pretrained weight does not have this): ", x)
                    else:
                        if (
                            x.endswith("embed_positions.weight")
                            and module_name == "AuxEncoder"
                            and self.config.max_enc_position_embeddings != 1024
                        ):
                            pass
                        elif (
                            x.endswith("embed_positions.weight")
                            and module_name == "CreamDecoder"
                            and self.config.max_position_embeddings != 1024
                        ):
                            new_state_dict[x] = torch.nn.Parameter(
                                self.resize_bart_abs_pos_emb(
                                    state_dict[x],
                                    self.config.max_position_embeddings + 2,
                                )
                            )
                        elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                            if state_dict[x].size(0) >= new_state_dict[x].size(0):
                                new_state_dict[x] = state_dict[x][: new_state_dict[x].size(0), :]
                            else:
                                new_state_dict[x] = torch.concat(
                                    [state_dict[x]]
                                    + [state_dict[x][-1].unsqueeze(0)]
                                    * (new_state_dict[x].size(0) - state_dict[x].size(0))
                                )
                        else:
                            new_state_dict[x] = state_dict[x]

                new_module.load_state_dict(new_state_dict)

                if self.config.tie_word_embeddings and not self.config.llm_integration_enabled:
                    self.lm_head.weight = self.text_decoder.embed_tokens.weight
                del state_dict
                del new_state_dict

        del pretrained

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_x_pos_ids: Optional[torch.LongTensor] = None,
        input_y_pos_ids: Optional[torch.LongTensor] = None,
        input_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_input_tensors: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cl_word_candidates: Optional[torch.LongTensor] = None,
        is_textread=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cl_loss = 0.0
        if encoder_outputs is None:
            if self.mode == "v" or self.aux_encoder is None:
                image_embeddings = self.image_encoder(image_input_tensors)

                encoder_outputs = BaseModelOutput(
                    last_hidden_state=image_embeddings,
                    hidden_states=None,
                    attentions=None,
                )
                if encoder_attention_mask is None:
                    encoder_attention_mask = image_attention_mask

            elif self.mode == "l":
                aux_embeddings = self.aux_encoder(
                    input_ids=input_ids,
                    input_x_pos_ids=input_x_pos_ids,
                    input_y_pos_ids=input_y_pos_ids,
                    input_type_ids=input_type_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    is_textread=is_textread,
                ).last_hidden_state

                encoder_outputs = BaseModelOutput(
                    last_hidden_state=aux_embeddings,
                    hidden_states=None,
                    attentions=None,
                )

                if encoder_attention_mask is None:
                    encoder_attention_mask = attention_mask

            else:
                image_embeddings = self.image_encoder(image_input_tensors)

                aux_embeddings = self.aux_encoder(
                    input_ids=input_ids,
                    input_x_pos_ids=input_x_pos_ids,
                    input_y_pos_ids=input_y_pos_ids,
                    input_type_ids=input_type_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    is_textread=is_textread,
                ).last_hidden_state

                encoder_outputs = BaseModelOutput(
                    last_hidden_state=torch.cat([image_embeddings, aux_embeddings], dim=1),
                    hidden_states=None,
                    attentions=None,
                )
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

                if self.config.cl_enabled and cl_word_candidates is not None:
                    flags = cl_word_candidates != -1
                    text_index = cl_word_candidates[:, :, 0].long()
                    img_index = cl_word_candidates[:, :, 1].long()

                    img_cl = torch.cat(
                        [image_embeddings[i][img_index[i]][flags[i, :, 1]] for i in range(image_embeddings.size(0))]
                    )
                    text_cl = torch.cat(
                        [aux_embeddings[i][text_index[i]][flags[i, :, 0]] for i in range(aux_embeddings.size(0))]
                    )

                    anchor_feature = self.projector(torch.cat([img_cl, text_cl]))
                    anchor_feature = nn.functional.normalize(anchor_feature, dim=1).view(-1, 128)

                    cl_label = torch.arange(img_cl.size(0), device=img_cl.device)
                    cl_labels = torch.cat([cl_label, cl_label], dim=0)

                    mask = torch.eq(cl_labels.view(-1, 1), cl_labels.view(1, -1))

                    inner_product = torch.einsum("ik,jk->ij", img_cl, img_cl)
                    square_norm = torch.einsum("ij->i", img_cl * img_cl).view(-1, 1)
                    img_cl_mask = torch.eq(inner_product, square_norm)

                    mask[: img_cl_mask.size(0), : img_cl_mask.size(0)] = img_cl_mask
                    mask[img_cl_mask.size(0) :, : img_cl_mask.size(0)] = img_cl_mask

                    logits = torch.einsum("nc,ck->nk", [anchor_feature, anchor_feature.T])
                    logits /= 0.07

                    cl_loss = (
                        (-torch.log_softmax(logits, dim=-1) * mask)
                        .sum(dim=-1, keepdim=True)
                        .div(mask.sum(dim=-1, keepdim=True) + 1e-5)
                    ).mean() * 0.5

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if self.config.llm_integration_enabled:
            inputs_embeds2 = self.text_decoder.embed_tokens(decoder_input_ids) * self.text_decoder.embed_scale
            inputs_embeds1 = self.query_tokens.expand(inputs_embeds2.shape[0], -1, -1)
            inputs_embeds = torch.cat([inputs_embeds1, inputs_embeds2], dim=1)
            outputs = self.text_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=encoder_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = outputs[0]
        else:
            outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=encoder_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0])

        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_id)
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CreamSeq2SeqLMOutput(
            lm_loss=lm_loss,
            cl_loss=cl_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_x_pos_ids: Optional[torch.LongTensor] = None,
        input_y_pos_ids: Optional[torch.LongTensor] = None,
        input_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_input_tensors: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        infer_mode: Optional[str] = "vl",
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if (infer_mode == "v" or self.mode == "v") or self.aux_encoder is None:
                image_embeddings = self.image_encoder(image_input_tensors)
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=image_embeddings,
                    hidden_states=None,
                    attentions=None,
                )
                if encoder_attention_mask is None:
                    encoder_attention_mask = image_attention_mask
            elif infer_mode == "l" or self.mode == "l":
                aux_embeddings = self.aux_encoder(
                    input_ids=input_ids,
                    input_x_pos_ids=input_x_pos_ids,
                    input_y_pos_ids=input_y_pos_ids,
                    input_type_ids=input_type_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ).last_hidden_state
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=aux_embeddings,
                    hidden_states=None,
                    attentions=None,
                )
                if encoder_attention_mask is None:
                    encoder_attention_mask = attention_mask
            else:
                image_embeddings = self.image_encoder(image_input_tensors)
                aux_embeddings = self.aux_encoder(
                    input_ids=input_ids,
                    input_x_pos_ids=input_x_pos_ids,
                    input_y_pos_ids=input_y_pos_ids,
                    input_type_ids=input_type_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ).last_hidden_state
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=torch.cat([image_embeddings, aux_embeddings], dim=1),
                    hidden_states=None,
                    attentions=None,
                )
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        decoder_output = self.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            num_beams=1,
            max_length=self.config.max_position_embeddings,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=return_dict,
            output_attentions=output_attentions,
        )

        output = {
            "queries": list(),
            "predictions": list(),
            "prediction_ids": decoder_output.sequences,
            "output_tokens": list(),
        }
        for seq in decoder_output.sequences:
            query, prediction = None, None
            eos_idxes = (seq == self.tokenizer.eos_token_id).nonzero().flatten()
            begin_idx = 0
            for i, end_idx in enumerate(eos_idxes):
                if i == 0:
                    query = self.tokenizer.decode(seq[begin_idx : end_idx + 1], skip_special_tokens=True).strip()
                else:
                    prediction = self.tokenizer.decode(seq[begin_idx : end_idx + 1], skip_special_tokens=True).strip()
                begin_idx = end_idx + 1

            output["queries"].append(query)
            output["predictions"].append(prediction)
            output["output_tokens"].append(self.tokenizer.decode(seq))

        if output_attentions:
            cross_attn_scores = list()
            for i in range(0, len(decoder_output.cross_attentions)):
                cross_attn_scores.append(torch.cat(decoder_output.cross_attentions[i], dim=0))
            cross_attn_scores = torch.cat(cross_attn_scores, dim=2)

            output["decoder_attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": cross_attn_scores,
            }

        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        model = super(CreamModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.config.name_or_path = pretrained_model_name_or_path
        return model
