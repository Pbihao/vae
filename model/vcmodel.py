# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from model.vae import Decoder, DecoderOutput, Encoder, VectorQuantizer
from diffusers.models.modeling_utils import ModelMixin
import torch.nn.functional as F


@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The encoded output sample from the last layer of the model.
    """

    latents: torch.Tensor


class VQModel(ModelMixin, ConfigMixin):
    r"""
    A VQ-VAE model for decoding latent representations.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `1`): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        norm_num_groups (`int`, *optional*, defaults to `32`): Number of groups for normalization layers.
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        norm_type (`str`, *optional*, defaults to `"group"`):
            Type of normalization layer to use. Can be one of `"group"` or `"spatial"`.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        lookup_from_codebook=False,
        force_upcast=False,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention,
        )

        # vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        # self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        # self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        # self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
            mid_block_add_attention=mid_block_add_attention,
        )

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, return_features=False) -> VQEncoderOutput:
        if not return_features:
            h = self.encoder(x)
        else:
            h, features = self.encoder(x, return_features=return_features)

        if not return_features:
            if not return_dict:
                return (h,) 
            return VQEncoderOutput(latents=h)
        else:
            if not return_dict:
                return (h, features) 
            return VQEncoderOutput(latents=h), features

    @apply_forward_hook
    def decode(
        self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None, return_features=False
    ) -> Union[DecoderOutput, torch.Tensor]:
        # also go through quantization layer
        # if not force_not_quantize:
        #     quant, commit_loss, _ = self.quantize(h)
        # elif self.config.lookup_from_codebook:
        #     quant = self.quantize.get_codebook_entry(h, shape)
        #     commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
        # else:
        #     quant = h
        #     commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
        # quant2 = self.post_quant_conv(quant)
        # dec = self.decoder(quant2, quant if self.config.norm_type == "spatial" else None)

        # tensor = F.softmax(h, dim=1) + 1e-10
        # commit_loss = -torch.mean(tensor * torch.log(tensor))
        commit_loss = entropy_loss(h)

        if not return_features:
            dec = self.decoder(h)
        else:
            dec, features = self.decoder(h, return_features=return_features)

        if not return_features:
            if not return_dict:
                return dec, commit_loss
            return DecoderOutput(sample=dec, commit_loss=commit_loss)
        else:
            if not return_dict:
                return dec, commit_loss, features
            return DecoderOutput(sample=dec, commit_loss=commit_loss), features

    def forward(
        self, sample: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        The [`VQModel`] forward method.

        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vq_model.VQEncoderOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """

        h, enc_features = self.encode(sample, return_features=True)
        h = h.latents
        dec, dec_features = self.decode(h, return_features=True)

        # enc_features = enc_features[1:]
        # dec_features = list(reversed(dec_features))[:-1]
        # loss = 0
        # for enc0, dec0 in zip(enc_features, dec_features):
        #     loss += F.mse_loss(enc0, F.adaptive_avg_pool2d(dec0, enc0.shape[-2:]))

        # dec.commit_loss += loss

        if not return_dict:
            return dec.sample, dec.commit_loss
        return dec

def entropy_loss(tensor):
    tensor = F.softmax(tensor, dim=1)
    loss = torch.sum(tensor * torch.log(tensor), dim=1)
    loss = -torch.mean(loss)
    return loss
















# from einops import rearrange
# from model.resnet import ResnetBlock2D

# # from model.vcmodel import VQModel
# # import torch
# # model=VQModel(1,2)
# # model(torch.rand(4,3,64,64))

# class Down4xBLock(nn.Module):
#     def __init__(self, inchannel, outchannel, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.proj = ResnetBlock2D(inchannel * 16, outchannel)
#         self.conv = nn.Sequential(*[ResnetBlock2D(outchannel, outchannel) for _ in range(6)])

#     def forward(self, tensor,*args, **kwargs):
#         tensor = rearrange(tensor, "b c (nh h) (nw w) -> b (h w c) nh nw", h=4, w=4)
#         tensor = self.proj(tensor)
#         tensor = self.conv(tensor)
#         return tensor

# class up4xBLock(nn.Module):
#     def __init__(self, inchannel, outchannel, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.proj = ResnetBlock2D(inchannel, outchannel * 16)
#         self.conv = nn.Sequential(*[ResnetBlock2D(outchannel, outchannel) for _ in range(6)])

#     def forward(self, tensor,*args, **kwargs):
#         tensor = self.proj(tensor)
#         tensor = rearrange(tensor, "b (h w c) nh nw -> b c (nh h) (nw w)", h=4, w=4)
#         tensor = self.conv(tensor)
#         return tensor

# class VQModel(ModelMixin, ConfigMixin):

#     @register_to_config
#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
#         up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
#         block_out_channels: Tuple[int, ...] = (64,),
#         layers_per_block: int = 1,
#         act_fn: str = "silu",
#         latent_channels: int = 3,
#         sample_size: int = 32,
#         num_vq_embeddings: int = 256,
#         norm_num_groups: int = 32,
#         vq_embed_dim: Optional[int] = None,
#         scaling_factor: float = 0.18215,
#         norm_type: str = "group",  # group, spatial
#         mid_block_add_attention=True,
#         lookup_from_codebook=False,
#         force_upcast=False,
#     ):
#         super().__init__()

#         # pass init params to Encoder
#         self.encoder = nn.Sequential(
#             Down4xBLock(3, 64),
#             Down4xBLock(64, 256),
#             Down4xBLock(256, 512),
#             ResnetBlock2D(512, 128)
#         )


#         # pass init params to Decoder
#         self.decoder = nn.Sequential(
#             ResnetBlock2D(128, 512),
#             up4xBLock(512, 256),
#             up4xBLock(256, 64),
#             up4xBLock(64, 3),
#         )

#     @apply_forward_hook
#     def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
#         h = self.encoder(x)
#         # h = self.quant_conv(h)

#         if not return_dict:
#             return (h,)

#         return VQEncoderOutput(latents=h)

#     @apply_forward_hook
#     def decode(
#         self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None
#     ) -> Union[DecoderOutput, torch.Tensor]:
#         commit_loss = entropy_loss(h)

#         dec = self.decoder(h)

#         if not return_dict:
#             return dec, commit_loss

#         return DecoderOutput(sample=dec, commit_loss=commit_loss)

#     def forward(
#         self, sample: torch.Tensor, return_dict: bool = True
#     ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
#         r"""
#         The [`VQModel`] forward method.

#         Args:
#             sample (`torch.Tensor`): Input sample.
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`models.vq_model.VQEncoderOutput`] instead of a plain tuple.

#         Returns:
#             [`~models.vq_model.VQEncoderOutput`] or `tuple`:
#                 If return_dict is True, a [`~models.vq_model.VQEncoderOutput`] is returned, otherwise a plain `tuple`
#                 is returned.
#         """
#         h = self.encode(sample).latents
#         dec = self.decode(h)

#         if not return_dict:
#             return dec.sample, dec.commit_loss
#         return dec

# def entropy_loss(tensor):
#     tensor = F.softmax(tensor, dim=1)
#     loss = torch.sum(tensor * torch.log(tensor), dim=1)
#     loss = -torch.mean(loss)
#     return loss