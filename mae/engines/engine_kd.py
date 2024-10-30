# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(student_model: torch.nn.Module,
                    teacher_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    student_model.train(True)
    teacher_model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            sched_type = "linear" if args.linear_lr_sched else "cosine"
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args, sched_type)

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_features = teacher_model.forward_features_kd(samples)
        student_features = student_model.forward_features_kd(samples)

        if args.loss_multiple_blocks:
            required_teacher_features = teacher_features[1::args.reduction_factor]
            assert len(required_teacher_features) == len(student_features)
            loss_list = [torch.abs(t_f - s_f).mean() for t_f, s_f in zip(required_teacher_features, student_features)]
            loss = torch.stack(loss_list).mean()
        elif args.loss_mae_kld:
            loss = (torch.abs(teacher_features[-1] - student_features[-1])).mean()
            kl_input = F.log_softmax(student_features[-1], dim=2)
            kl_target = F.log_softmax(teacher_features[-1], dim=2)
            loss += torch.nn.functional.kl_div(kl_input, kl_target)
            loss /= 2
        else:
            loss = (torch.abs(teacher_features[-1] - student_features[-1])).mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=student_model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}