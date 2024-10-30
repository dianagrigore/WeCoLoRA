# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import copy
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from mae.models import models_vit

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

from mae.parameter_update_techniques.lora_layers import LoRA_ViT_timm
from mae.parameter_update_techniques.dora_layers import DoRA_ViT_timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from mae.engines.engine_kd import train_one_epoch
from mae.datasets.dataset_folders.image_folder_with_ratio import ImageFolderWithRatio


def get_args_parser():
    parser = argparse.ArgumentParser('Standard KD', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--teacher_model', default='vit_base_patch16', type=str,
                        help='Name of teacher model')
    parser.add_argument('--teacher_checkpoint', default='',
                        help='teacher checkpoint to resume from')
    # Model parameters
    parser.add_argument('--student_model', default=None, type=str,
                        help='Name of student model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/media/tonio/p2/datasets/Imagenet', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # lora distillation parameters
    parser.add_argument('--copy_weights_from_the_teacher', action='store_true',
                        help='copy the students weights from the teacher.')

    parser.add_argument('--lora_distillation', action='store_true',
                        help='Distillate using Lora weights')
    parser.add_argument('--dora_distillation', action='store_true',
                        help='Distillate using Dora weights')
    parser.add_argument('--loss_multiple_blocks', action='store_true',
                        help='Use he lora loss after every block ')
    parser.add_argument('--loss_mae_kld', action='store_true',
                        help='Use MAE + KLD loss')
    parser.add_argument('--linear_lr_sched', action='store_true',
                        help='Use linear rate scheduling')
    parser.add_argument('--lora_matrix_rank', default=4, type=int)
    parser.add_argument('--reduction_factor', default=2, type=int)
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_ratio',default=0.1, type=float)
    return parser


def main(args):
    misc.init_distributed_mode(args)
    args.log_dir = args.output_dir
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Simple augmentation for KD too.
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # transform_train = transforms.Compose([
    #     # RandomResizedCrop(224, interpolation=3),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.Resize(256, interpolation=3),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.few_shot:
        dataset_train = ImageFolderWithRatio(os.path.join(args.data_path, 'train'),
                                             transform=transform_train,
                                             ratio=args.few_shot_ratio)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # define the model
    teacher_model = models_vit.__dict__[args.teacher_model](is_lora=args.lora_distillation)
    if args.teacher_model == "vit_base_patch16_224":
        assert len(args.teacher_checkpoint) == 0

    if len(args.teacher_checkpoint) != 0:
        # Restore the teacher
        checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
        print("Load teacher checkpoint from: %s" % args.teacher_checkpoint)
        checkpoint_model = checkpoint['model']
        msg = teacher_model.load_state_dict(checkpoint_model, strict=False)
        print('The teacher model was restored with the following message:', msg)
    teacher_model.eval()
    teacher_model.to(device)

    if args.lora_distillation is True:
        assert args.student_model is None, args.student_model

    if args.lora_distillation:
        student_model = LoRA_ViT_timm(copy.deepcopy(teacher_model),
                                      args.lora_matrix_rank,
                                      args.reduction_factor)
    elif args.dora_distillation:
        student_model = DoRA_ViT_timm(copy.deepcopy(teacher_model),
                                      args.lora_matrix_rank,
                                      args.reduction_factor)
    else:
        student_model = models_vit.__dict__[args.student_model]()

    if args.copy_weights_from_the_teacher:
        student_model = copy.deepcopy(teacher_model)
        student_model.blocks = student_model.blocks[0::args.reduction_factor]

    student_model.to(device)
    student_model_without_ddp = student_model
    print("Model = %s" % str(student_model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(student_model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=student_model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # Count the number of trainable weights
    pytorch_total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print("The number of trainable parameters are:", pytorch_total_params / 1e6)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student_model, teacher_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=student_model, model_without_ddp=student_model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
