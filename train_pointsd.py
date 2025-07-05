#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import json
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from easydict import EasyDict
import yaml
import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.utils import ContextManagers

from resampler import Resampler
from test import validate_model

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from pointbert.point_encoder import PointTransformer
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    # DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from vpd import UNetWrapper
from attention_processor import (
    IPAttnProcessor2_0 as IPAttnProcessor,
    AttnProcessor2_0 as AttnProcessor,
    PCProjModel,
)

if is_wandb_available():
    import wandb
import itertools

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(
        self,
        unet,
        pc_model,
        global_proj_model,
        local_proj_model,
        adapter_modules,
        pc_model_frz=None,
        global_proj_model_frz=None,
    ):
        super().__init__()
        self.unet = unet
        self.pc_model = pc_model
        self.global_proj_model = global_proj_model
        self.local_proj_model = local_proj_model
        self.adapter_modules = adapter_modules
        self.pc_model_frz = pc_model_frz
        self.global_proj_model_frz = global_proj_model_frz

    def get_pc_embed(self, pc, mix=False):
        all_embeds, embeds = self.pc_model(pc, mix)

        if self.pc_model_frz is not None:
            with torch.no_grad():
                all_embeds_frz, embeds_frz = self.pc_model_frz(pc, mix)
                global_tokens_frz = self.global_proj_model_frz(all_embeds_frz)
        else:
            global_tokens_frz = torch.zeros(pc.shape[0], 4, 768).to(pc.device)

        global_tokens = self.global_proj_model(all_embeds)
        if self.local_proj_model is not None:
            local_tokens = self.local_proj_model(all_embeds.unsqueeze(1))
        else:
            local_tokens = None
        return global_tokens, local_tokens, global_tokens_frz

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, global_tokens):
        encoder_hidden_states = torch.cat([encoder_hidden_states, global_tokens], dim=1)
        out_list = self.unet(noisy_latents, timesteps, encoder_hidden_states)
        noise_pred = out_list[0]
        feat = out_list[-1:][::-1]
        return noise_pred, feat


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--validation_before_training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="textual_inversion",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--sample_points_num",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the image data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
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
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--random_view", action="store_true", help="Whether to use random view."
    )
    parser.add_argument("--view_idx", type=int, default=0, help="select view idx.")
    parser.add_argument("--depth", type=int, default=4, help="resampler depth.")
    parser.add_argument(
        "--heads", type=int, default=12, help="resampler multi-head num."
    )
    parser.add_argument(
        "--max_seg_len", type=int, default=64, help="resampler input tokens num."
    )
    parser.add_argument(
        "--run_stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2"],
        help="decide to run which stage.",
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, default=None, help="dir of stage1 ckpt."
    )
    parser.add_argument(
        "--num_tokens", type=int, default=4, help="number of project tokens"
    )
    parser.add_argument(
        "--perform_mix", action="store_true", help="whether to perform mix strategy."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--num_warm_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_timesteps_max",
        type=int,
        default=None,
        help="max training step",
    )
    parser.add_argument(
        "--train_timesteps_min",
        type=int,
        default=None,
        help="min training step",
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
        "--scale",
        type=float,
        default=0,
        help="Scale the hidden states from text encoder.",
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
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
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
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_root,
        img_root,
        size=512,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
        sample_points_num=1024,
        random_view=False,
        view_idx=0,
    ):
        self.data_root = data_root
        self.img_root = img_root
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.ids = []
        self.sample_points_num = sample_points_num
        with open(os.path.join(self.data_root, "train.txt"), "r") as f:
            for line in f:
                line = line.strip()
                self.ids.append(line.split(".")[0])
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.random_view = random_view
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.augment = True
        self.permutation = np.arange(sample_points_num)
        self.view_idx = view_idx
        with open(os.path.join(self.data_root, "shapenet-55_taxonomy.json")) as f:
            self.id_map = json.load(f)
        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

    def __len__(self):
        return len(self.ids)

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def img_aug(self, image):
        if not (image.size[0] == self.size and image.size[1] == self.size):
            new_image = Image.new("RGB", image.size, "WHITE")
            new_image.paste(image, (0, 0), image)
            image = new_image
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = {}
        if self.random_view:
            img_idx = random.randint(0, 29)
        else:
            img_idx = self.view_idx

        img_idx = img_idx * 12
        if img_idx < 10:
            img_idx = "00" + str(img_idx)
        elif img_idx < 100:
            img_idx = "0" + str(img_idx)
        else:
            img_idx = str(img_idx)
        image = Image.open(
            os.path.join(
                self.img_root, f"{self.ids[i]}", f"{self.ids[i]}_r_{img_idx}.png"
            )
        )
        image = self.img_aug(image)

        pc = np.load(os.path.join(self.data_root, "shapenet_pc", self.ids[i] + ".npy"))

        if self.sample_points_num < pc.shape[0]:
            # pc = farthest_point_sample(pc, self.sample_points_num)
            pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)[:, :3]

        if self.augment:
            pc = random_point_dropout(pc[None, ...])
            pc = random_scale_point_cloud(pc)
            pc = shift_point_cloud(pc)
            pc = rotate_perturbation_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = pc.squeeze()
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["pc"] = torch.from_numpy(pc)
        example["id"] = self.ids[i]
        return example


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def rotate_point_cloud_z(batch_data, rotation_angle):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        print("DEEPSPEED", "train_micro_batch_size_per_gpu", args.train_batch_size)
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = args.train_batch_size
        print(
            "DEEPSPEED",
            "train_micro_batch_size_per_gpu",
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ],
        )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    unet_wrapper = UNetWrapper(unet, base_size=args.resolution, use_attn=False)
    # turn to 1024 64 32
    config_addr = "./pointbert/PointTransformer.yaml"
    pc_feat_dims = 768
    config = cfg_from_yaml_file(config_addr)
    pc_encoder = PointTransformer(config.model)

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder

    global_proj_model = PCProjModel(
        cross_attention_dim=768,
        clip_embeddings_dim=pc_feat_dims,
        clip_extra_context_tokens=4,
    )

    if args.run_stage == "stage2":
        local_proj_model = Resampler(
            dim=768,
            depth=args.depth,
            dim_head=64,
            heads=args.heads,
            num_queries=1,
            embedding_dim=pc_feat_dims,
            output_dim=1280,
            ff_mult=4,
            max_seq_len=1,
        )
    else:
        local_proj_model = None
    attn_procs = {}
    unet_sd = unet.state_dict()
    for i, name in enumerate(unet.attn_processors.keys()):
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()

        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=args.scale,
                num_tokens=args.num_tokens,
            )
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    if args.run_stage == "stage2":
        pc_encoder_frz = PointTransformer(config.model)
        pc_encoder_frz.load_state_dict(
            torch.load(
                os.path.join(args.stage1_ckpt, "ckpt-stage1.pt"), map_location="cuda"
            )
        )
        global_proj_model_frz = PCProjModel(
            cross_attention_dim=768,
            clip_embeddings_dim=768,
            clip_extra_context_tokens=4,
        )
        global_proj_model_frz.load_state_dict(
            torch.load(
                os.path.join(args.stage1_ckpt, "global_proj_model.pt"),
                map_location="cuda",
            )
        )
        adapter_modules.load_state_dict(
            torch.load(
                os.path.join(args.stage1_ckpt, "adapter_modules.pt"),
                map_location="cuda",
            )
        )
        pc_encoder_frz.requires_grad_(False)
        global_proj_model_frz.requires_grad_(False)
        adapter_modules.requires_grad_(False)
        pc_encoder_frz.eval()
        global_proj_model_frz.eval()
        ip_adapter = IPAdapter(
            unet_wrapper,
            pc_encoder,
            global_proj_model,
            local_proj_model,
            adapter_modules,
            pc_encoder_frz,
            global_proj_model_frz,
        )
    else:
        ip_adapter = IPAdapter(
            unet_wrapper,
            pc_encoder,
            global_proj_model,
            local_proj_model,
            adapter_modules,
        )
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    params_to_opt = itertools.chain(
        ip_adapter.pc_model.parameters(),
        ip_adapter.global_proj_model.parameters(),
        ip_adapter.adapter_modules.parameters(),
    )
    if args.run_stage == "stage2":
        params_to_opt = itertools.chain(
            params_to_opt, ip_adapter.local_proj_model.parameters()
        )
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = PretrainDataset(
        data_root=args.train_data_dir,
        img_root=args.img_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        sample_points_num=args.sample_points_num,
        random_view=args.random_view,
        view_idx=args.view_idx,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    args.lr_warmup_steps = args.num_warm_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        # num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        # num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader, lr_scheduler
    )
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("textual_inversion", config=vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            dict(vars(args)),
            # init_kwargs={"wandb": {"entity": ""}}, #wandb usrname
        )
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    global_acc = 0
    best_acc = 0
    ip_adapter.pc_model.train()
    ip_adapter.global_proj_model.train()
    if args.run_stage == "stage2":
        ip_adapter.local_proj_model.train()
    ip_adapter.adapter_modules.train()
    if args.validation_before_training:
        ip_adapter.pc_model.eval()
        acc = validate_model(ip_adapter.pc_model, weight_dtype)
        if accelerator.is_main_process:
            global_acc = acc
            if best_acc < acc:
                best_acc = acc
        ip_adapter.pc_model.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        # ip_adapter.pc_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                if args.perform_mix:
                    to_mix = random.random()
                else:
                    to_mix = 0
                if to_mix > 0.5:
                    b, c, h, w = batch["pixel_values"].shape
                    batch["pixel_values"] = batch["pixel_values"].reshape(
                        -1, 2, c, h, w
                    )
                    pixel_values_cat_1 = torch.cat(
                        [batch["pixel_values"][:, 0], batch["pixel_values"][:, 1]],
                        dim=-1,
                    )
                    pixel_values_cat_2 = torch.cat(
                        [batch["pixel_values"][:, 1], batch["pixel_values"][:, 0]],
                        dim=-1,
                    )
                    pixel_values_pad = torch.ones_like(pixel_values_cat_1).to(
                        pixel_values_cat_1.device
                    )
                    pixel_values_cat_1 = torch.cat(
                        [
                            pixel_values_pad[:, :, : h // 2, :],
                            pixel_values_cat_1,
                            pixel_values_pad[:, :, h // 2 :, :],
                        ],
                        dim=2,
                    )
                    pixel_values_cat_2 = torch.cat(
                        [
                            pixel_values_pad[:, :, : h // 2, :],
                            pixel_values_cat_2,
                            pixel_values_pad[:, :, h // 2 :, :],
                        ],
                        dim=2,
                    )
                    batch["pixel_values"] = torch.cat(
                        [
                            pixel_values_cat_1.unsqueeze(1),
                            pixel_values_cat_2.unsqueeze(1),
                        ],
                        dim=1,
                    ).reshape(b, c, 2 * h, 2 * w)
                    batch["pixel_values"] = F.interpolate(
                        batch["pixel_values"], size=(h, w)
                    )

                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                    .latent_dist.sample()
                    .detach()
                )
                latents = latents * vae.config.scaling_factor

                global_tokens, recon_feat, global_tokens_frz = ip_adapter.get_pc_embed(
                    batch["pc"], to_mix > 0.5
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                if (
                    args.train_timesteps_min is not None
                    and args.train_timesteps_max is not None
                ):
                    timesteps = torch.randint(
                        args.train_timesteps_min,
                        args.train_timesteps_max,
                        (bsz,),
                        device=latents.device,
                    )
                else:
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.run_stage == "stage2":
                    timesteps = (
                        torch.zeros_like(timesteps).to(device=latents.device).long()
                    )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Predict the noise residual
                encoder_hidden_states = torch.zeros([batch["pc"].shape[0], 77, 768]).to(
                    device=latents.device, dtype=weight_dtype
                )

                if args.run_stage == "stage1":
                    model_pred, feat = ip_adapter(
                        noisy_latents, timesteps, encoder_hidden_states, global_tokens
                    )
                    loss_denoising = F.mse_loss(
                        model_pred.float(), noise.float(), reduction="mean"
                    )
                    loss = loss_denoising
                elif args.run_stage == "stage2":
                    model_pred, feat = ip_adapter(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        global_tokens_frz,
                    )
                    target = feat[0].squeeze()
                    loss_alignment = F.mse_loss(
                        recon_feat.squeeze(), target, reduction="mean"
                    )
                    loss = loss_alignment
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if (
                    AcceleratorState().deepspeed_plugin is not None
                ) or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        # accelerator.save_state(save_path)
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(
                            ip_adapter.pc_model.state_dict(),
                            save_path + f"/ckpt-{args.run_stage}.pt",
                        )
                        torch.save(
                            ip_adapter.global_proj_model.state_dict(),
                            save_path + "/global_proj_model.pt",
                        )
                        torch.save(
                            ip_adapter.adapter_modules.state_dict(),
                            save_path + "/adapter_modules.pt",
                        )
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        ip_adapter.pc_model.eval()
                        acc = validate_model(ip_adapter.pc_model, weight_dtype)
                        if accelerator.is_main_process:
                            global_acc = acc
                            if best_acc < acc:
                                best_acc = acc
                                save_path = os.path.join(
                                    args.output_dir, f"checkpoint-best"
                                )
                                os.makedirs(save_path, exist_ok=True)
                                torch.save(
                                    ip_adapter.pc_model.state_dict(),
                                    save_path + f"/ckpt-{args.run_stage}.pt",
                                )
                                torch.save(
                                    ip_adapter.global_proj_model.state_dict(),
                                    save_path + "/global_proj_model.pt",
                                )
                                torch.save(
                                    ip_adapter.adapter_modules.state_dict(),
                                    save_path + "/adapter_modules.pt",
                                )
                        ip_adapter.pc_model.train()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "acc": global_acc,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    accelerator.end_training()


if __name__ == "__main__":
    main()
