# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from mae.datasets.satellite_dataset import create_NWPU_RESISC
from util.datasets import build_dataset
import torchvision.datasets as datasets
# from flowers102 import Flowers102
from mae.datasets.dataset_folders.dataset_folder_with_path import DatasetFolderWithPath
import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS

from mae.models import models_vit
from mae.datasets import cxr_dataset
from mae.engines.engine_finetune import train_one_epoch, evaluate
from mae.parameter_update_techniques.lora_layers import LoRA_ViT_timm
import copy
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset',
                        help='dataset used for pretraining medical/satellite')
    parser.add_argument('--data_path', default='/media/lili/SSD2/datasets/ssl/medical/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

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
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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

    # lora parameters
    parser.add_argument('--lora_model', action='store_true',
                        help='This model comes from a LoRA model.')

    parser.add_argument('--dora_model', action='store_true',
                        help='Distillate using Dora weights')

    parser.add_argument('--lora_matrix_rank', default=4, type=int)
    parser.add_argument('--reduction_factor', default=2, type=int)


    parser.add_argument('--extract_features', action='store_true',
                        help='extract_features.')

    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--top_pruning', action='store_true')
    parser.add_argument('--use_teacher_head', action='store_true')

    parser.add_argument('--teacher_model', default='vit_base_patch16', type=str,
                        help='Name of teacher model')
    parser.add_argument('--teacher_checkpoint', default='',
                        help='teacher checkpoint to resume from')
    return parser



def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation

    if args.dataset == "cxr":
        transform_train = transforms.Compose([
                                 transforms.Resize(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5057, 0.5057, 0.5057], std=[0.2507, 0.2507, 0.2507]),
                                 ])
        transform_val = transforms.Compose([
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5057, 0.5057, 0.5057], std=[0.2508, 0.2508, 0.2508])
                                            ])
        dataset_train = cxr_dataset.CXRDataset(is_train=True, transform=transform_train)
        dataset_val = cxr_dataset.CXRDataset(is_train=False, transform=transform_val)

    elif args.dataset == "NWPU-RESISC":
        dataset_train, dataset_val = create_NWPU_RESISC(args.data_path, args)

    elif args.dataset == "imagenet" or args.dataset == 'inat19':
        transform_train = transforms.Compose([
            # RandomResizedCrop(224, interpolation=3),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ratio = 1
        if args.few_shot:
            ratio = 0.1

        dataset_train = DatasetFolderWithPath(os.path.join(args.data_path, 'train'),
                                              transform=transform_train, ratio=ratio)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        print(dataset_train)
        print(dataset_val)
    elif args.dataset == "cifar-100":
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        dataset_train = CIFAR100(download=True, root=args.data_path, transform=transform_train)
        dataset_val = CIFAR100(root=args.data_path, train=False, transform=transform_val)
    elif args.dataset == "flowers":
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])

        dataset_train = Flowers102(root=args.data_path, split='train', transform=transform_train)
        dataset_val = Flowers102(root=args.data_path, split='test', transform=transform_val)
        print(dataset_train)
        print(dataset_val)
    else: # RSNA
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if 'mini' not in args.model and 'deit' not in args.model :
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
        )
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes
        )

    if args.lora_model:
        model = LoRA_ViT_timm(copy.deepcopy(model), args.lora_matrix_rank, args.reduction_factor)

    elif args.dora_model:
        from dora import DoraModel, DoraConfig
        model.blocks = model.blocks[0::args.reduction_factor]
        dora_config = DoraConfig(r=args.lora_matrix_rank,  target_modules=["q", "v"],
                                 lora_dropout=0.0, lora_alpha=1)
        model = DoraModel(dora_config, copy.deepcopy(model))

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        keys_list = ['lora_vit.head.weight', 'lora_vit.head.bias'] if args.lora_model or args.dora_model else ['head.weight', 'head.bias']

        keys_list = ['model.head.weight', 'model.head.bias'] if args.dora_model else keys_list
        for k in keys_list:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        if args.lora_model:
            interpolate_pos_embed(model.lora_vit, checkpoint_model, pos_embed_key='lora_vit.pos_embed')
        else:
            interpolate_pos_embed(model, checkpoint_model, pos_embed_key='pos_embed')

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if args.lora_model:
            model = model.lora_vit
        if args.dora_model:
            model = model.model

        if args.global_pool:
            # TODO: make it also work
            pass
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'} or len(msg.missing_keys) == 0
        else:
            assert set(msg.missing_keys) == set(keys_list) or len(msg.missing_keys) == 0,  msg.missing_keys

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    if args.top_pruning:
        num_of_kept_blocks = len(model.blocks) // args.reduction_factor
        model.blocks = model.blocks[:num_of_kept_blocks]

    if args.dataset == "imagenet" or args.dataset == 'inat19':
        # Extract Features
        # TODO: delete the previous features
        if args.extract_features:
            model.to(device)
            model.eval()
            for (samples, _, paths) in tqdm(data_loader_train):
                with torch.no_grad():
                    samples = samples.to(device, non_blocking=True)
                    embeddings = model.forward_features_disk(samples)

                for path_, embedding in zip(paths, embeddings):
                    ext = path_.split('.')[-1]
                    new_path = path_.replace("/train/", '/train_features/').replace(ext, 'npy')
                    base_folder = '/'.join(new_path.split(os.path.sep)[:-1])
                    os.makedirs(base_folder, exist_ok=True)
                    np.save(new_path, np.float16(embedding.cpu().numpy()))
            model.train()
            exit()

        def is_valid_file_fn(file_path):
            return 'npy' in file_path
        def loader_fn(file_path):
            return np.load(file_path)

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train_features'),
                                              is_valid_file=is_valid_file_fn,
                                              loader=loader_fn,
                                              transform=None)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # for linear prob only
    # hack: revise model's head with BN
    if args.use_teacher_head:
        # Load the teacher checkpoint
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')

        # Initialize weights of the current model's head layer
        trunc_normal_(model.head.weight, std=0.01)

        # Copy weights and biases from the teacher model
        model.head.weight.data.copy_(teacher_checkpoint['model']['head.1.weight'])
        model.head.bias.data.copy_(teacher_checkpoint['model']['head.1.bias'])
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)


    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.dataset == 'rsna' or args.dataset == 'cxr':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, dataset=args.dataset)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 5 == 0 or epoch == (args.epochs - 1)):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, dataset=args.dataset, epoch=epoch, output_dir=args.output_dir)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

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