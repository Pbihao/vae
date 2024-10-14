# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import accelerate
import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm
from model.autoencoders import AutoencoderKL

from model.vcmodel import VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
import cv2
from einops import rearrange
import torch.nn as nn
from utils.dct import compute_dct_loss

os.environ["HF_HOME"] = "/your/custom/cache/directory"



if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _map_layer_to_idx(backbone, layers, offset=0):
    """Maps set of layer names to indices of model. Ported from anomalib

    Returns:
        Feature map extracted from the CNN
    """
    idx = []
    features = timm.create_model(
        backbone,
        pretrained=False,
        features_only=False,
        exportable=True,
    )
    for i in layers:
        try:
            idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
        except ValueError:
            raise ValueError(
                f"Layer {i} not found in model {backbone}. Select layer from {list(dict(features.named_children()).keys())}. The network architecture is {features}"
            )
    return idx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kl_divergence(reconstructed_tensor, groundtruth_tensor, path_size = 16, overlap=True, weight=1):
    B, C, H, W = reconstructed_tensor.shape
    assert H % path_size == 0 and W % path_size == 0

    def kld_loss(reconstructed, groundtruth, patch=True):
        dtype = reconstructed.dtype
        reconstructed = reconstructed.to(dtype=torch.float32)
        groundtruth = groundtruth.to(dtype=torch.float32)
        if patch:
            reconstructed = rearrange(reconstructed, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)
            groundtruth = rearrange(groundtruth, "b c (h ph) (w pw) -> (b c h w) (ph pw)", ph=path_size, pw=path_size)

            p = F.softmax(groundtruth, dim=-1) 
            q = F.softmax(reconstructed, dim=-1) 
            log_p = F.log_softmax(groundtruth, dim=-1)
            log_q = F.log_softmax(reconstructed, dim=-1)
            kl_div = torch.sum(p * (log_p - log_q), dim=-1) * weight
        else:
            p = F.softmax(groundtruth, dim=1) 
            q = F.softmax(reconstructed, dim=1)
            log_p = F.log_softmax(groundtruth, dim=1)
            log_q = F.log_softmax(reconstructed, dim=1)
            kl_div = torch.sum(p * (log_p - log_q), dim=1) * weight
        return kl_div.mean().to(dtype=dtype)
    

    loss = kld_loss(reconstructed_tensor, groundtruth_tensor)
    if overlap:
        shift_size = path_size // 2
        shifted_H_reconstructed = torch.zeros_like(reconstructed_tensor)
        shifted_H_groundtruth = torch.zeros_like(groundtruth_tensor)
        shifted_H_reconstructed[:, :, shift_size:, :] = reconstructed_tensor[:, :, :-shift_size, :]
        shifted_H_reconstructed[:, :, :shift_size, :] = reconstructed_tensor[:, :, -shift_size:, :]
        shifted_H_groundtruth[:, :, shift_size:, :] = groundtruth_tensor[:, :, :-shift_size, :]
        shifted_H_groundtruth[:, :, :shift_size, :] = groundtruth_tensor[:, :, -shift_size:, :]

        shifted_HW_reconstructed = torch.zeros_like(shifted_H_reconstructed)
        shifted_HW_groundtruth = torch.zeros_like(shifted_H_groundtruth)
        shifted_HW_reconstructed[:, :, :, shift_size:] = shifted_H_reconstructed[:, :, :, :-shift_size]
        shifted_HW_reconstructed[:, :, :, :shift_size] = shifted_H_reconstructed[:, :, :, -shift_size:]
        shifted_HW_groundtruth[:, :, :, shift_size:] = shifted_H_groundtruth[:, :, :, :-shift_size]
        shifted_HW_groundtruth[:, :, :, :shift_size] = shifted_H_groundtruth[:, :, :, -shift_size:]

        loss += kld_loss(shifted_HW_reconstructed, shifted_HW_groundtruth)
        loss /= 2

    loss += kld_loss(reconstructed_tensor, groundtruth_tensor, patch=False)

    return loss


def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bsz = gradients.shape[0]
    gradients = torch.reshape(gradients, (bsz, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def log_validation(model, args, validation_transform, accelerator, global_step, vae):
    logger.info("Generating images...")
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    original_images = []
    for image_path in args.validation_images:
        image = PIL.Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = validation_transform(image).to(accelerator.device, dtype=dtype)
        original_images.append(image[None])
    # Generate images
    model.eval()
    images = []
    with torch.no_grad():
        for original_image in original_images:
            latents = accelerator.unwrap_model(vae).encode(original_image).latent_dist.mean
            image = accelerator.unwrap_model(model)(latents).sample
            image = accelerator.unwrap_model(vae).decode(image).sample
            # image = (image / 2 + 0.5).clamp(0, 1)
            images.append(image)
    model.train()


    saved_images = []
    for ori_img, img in zip(original_images, images):
        img = torch.clamp(img / 2 + 0.5, 0.0, 1.0)
        ori_img = torch.clamp(ori_img / 2 + 0.5, 0.0, 1.0)
        img *= 255.0
        ori_img *= 255.0
        img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        ori_img = ori_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        img = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
        image = np.concatenate([ori_img, img], axis=1)
        saved_images.append(Image.fromarray(image))

    os.makedirs(args.val_image_dir, exist_ok=True)
    for idx, img in enumerate(saved_images):
        path = os.path.join(args.val_image_dir, "{}_{}.jpg".format(global_step, idx))
        img.save(path)

    torch.cuda.empty_cache()
    return images


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--log_grad_norm_steps",
        type=int,
        default=500,
        help=("Print logs of gradient norms every X steps."),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help=("Print logs every X steps."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running reconstruction on images in"
            " `args.validation_images` and logging the reconstructed images."
        ),
    )
    parser.add_argument(
        "--vae_loss",
        type=str,
        default="l2",
        help="The loss function for vae reconstruction loss.",
    )
    parser.add_argument(
        "--timm_model_offset",
        type=int,
        default=0,
        help="Offset of timm layers to indices.",
    )
    parser.add_argument(
        "--timm_model_layers",
        type=str,
        default="head",
        help="The layers to get output from in the timm model.",
    )
    parser.add_argument(
        "--timm_model_backend",
        type=str,
        default="vgg19",
        help="Timm model used to get the lpips loss",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the Vq model to train, leave as None to use standard Vq model configuration.",
    )
    parser.add_argument(
        "--discriminator_config_name_or_path",
        type=str,
        default=None,
        help="The config of the discriminator model to train, leave as None to use standard Vq model configuration.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=("A set of validation images evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vqgan-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--discr_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--discr_lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--val_image_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vqgan-training",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    args = parse_args()

    # Enable TF32 on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    if args.model_config_name_or_path is None and args.pretrained_model_name_or_path is None:
        # model= AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        # model.encoder.conv_in = nn.Conv2d(4, 128, kernel_size=3, padding=1, stride=1)
        # model.decoder.conv_out = nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1)
        # model.config['in_channels'] = 4
        # model.config['out_channels'] = 4

        # num_latent_channels = 64
        # model.config['latent_channels'] = num_latent_channels
        # model.encoder.conv_out = nn.Conv2d(512, num_latent_channels * 2, kernel_size=3, padding=1, stride=1)
        # model.quant_conv = nn.Conv2d(num_latent_channels * 2, num_latent_channels * 2, kernel_size=1, stride=1)
        # model.post_quant_conv = nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=1, stride=1)
        # model.decoder.conv_in = nn.Conv2d(num_latent_channels, 512, kernel_size=3, stride=1, padding=1)
        model = VQModel(
            act_fn="silu",
            block_out_channels=[
                256,
                512,
                512,
            ],
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            in_channels=4,
            latent_channels=32,
            layers_per_block=2,
            norm_num_groups=32,
            norm_type="group",
            num_vq_embeddings=16384,
            out_channels=4,
            sample_size=32,
            scaling_factor=0.18215,
            up_block_types=[
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", ],
            vq_embed_dim=4,
        )
    elif args.pretrained_model_name_or_path is not None:
        model = VQModel.from_pretrained(args.pretrained_model_name_or_path)
    else:
        config = VQModel.load_config(args.model_config_name_or_path)
        model = VQModel.from_config(config)
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=VQModel, model_config=model.config)


    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "vqmodel_ema"))
                vqmodel = models[0]
                vqmodel.save_pretrained(os.path.join(output_dir, "vqmodel"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vqmodel_ema"), VQModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            vqmodel = models.pop()
            load_model = VQModel.from_pretrained(input_dir, subfolder="vqmodel")
            vqmodel.register_to_config(**load_model.config)
            vqmodel.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = (
            learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    args.train_batch_size * accelerator.num_processes
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # DataLoaders creation:
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    assert args.image_column is not None
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}")
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae")
    vae.eval()

    vae, model, optimizer, lr_scheduler = accelerator.prepare(
       vae, model, optimizer, lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            accelerator.wait_for_everyone()
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    avg_gen_loss = None
    avg_mse_loss = None
    avg_entropy_loss = None
    avg_kld_loss = None
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"]
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)
            optimizer.zero_grad(set_to_none=True)
            # encode images to the latent space and get the commit loss from vq tokenization
            # Return commit loss

            with torch.no_grad():
                latents = accelerator.unwrap_model(vae).encode(pixel_values).latent_dist.mean
            fmap, commit_loss = model(latents, return_dict=False)
            # with torch.no_grad():
            #     reconstructed_image = accelerator.unwrap_model(vae).decode(fmap).sample

            with accelerator.accumulate(model):
                # reconstruction loss. Pixel level differences between input vs output
                loss = 0
                mse_loss = F.mse_loss(latents, fmap) 
                kld_loss = kl_divergence(latents, fmap)
                # kld_loss = compute_dct_loss(pixel_values, fmap)
                loss += kld_loss
                loss += mse_loss
                loss += commit_loss * 0
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_gen_loss = accelerator.gather(loss.repeat(args.train_batch_size)).float().mean()
                avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).float().mean()
                avg_entropy_loss = accelerator.gather(commit_loss.repeat(args.train_batch_size)).float().mean()
                avg_kld_loss = accelerator.gather(kld_loss.repeat(args.train_batch_size)).float().mean()
                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                # log gradient norm before zeroing it
                if (
                    accelerator.sync_gradients
                    and global_step % args.log_grad_norm_steps == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step)
            batch_time_m.update(time.time() - end)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if args.use_ema:
                    ema_model.step(model.parameters())
            if accelerator.sync_gradients and accelerator.is_main_process:
                # wait for both generator and discriminator to settle
                # Log metrics
                if global_step % args.log_steps == 0:
                    samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps * args.train_batch_size / batch_time_m.val
                    )
                    logs = {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if avg_gen_loss is not None:
                        logs["step_gen_loss"] = avg_gen_loss.item()
                        logs["step_mse_loss"] = avg_mse_loss.item()
                        logs["step_entropy_loss"] = avg_entropy_loss.item()
                        logs["step_kld_loss"] = avg_kld_loss.item()
                    accelerator.log(logs, step=global_step)
                    print(global_step, logs)
                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Generate images
                if global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the VQGAN parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    log_validation(model, args, validation_transform, accelerator, global_step, vae=vae)
                    if args.use_ema:
                        # Switch back to the original VQGAN parameters.
                        ema_model.restore(model.parameters())
            end = time.time()
            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(os.path.join(args.output_dir, "vqmodel"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
