import os
import json
import yaml
import logging
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lib.dataset.build import build_data_loader
from lib.solver.build import build_optimizer_scheduler
from lib.utils.metric_logger import MetricLogger
from lib.utils.comm import synchronize, is_main_process
from lib.utils.miscellaneous import AverageMeter, calculate_eta_iter 
from lib.utils.logger import setup_logger, get_timestamp
from lib.modeling import *
from .model import BaseModel


def train(gpu, cfg, args, basedata):
    rank = args.local_rank * args.ngpus_per_node + gpu
    logger = setup_logger(name='train', save_dir='logs', distributed_rank=rank, filename=f'{get_timestamp()}_train.txt')
    logger = logging.getLogger('train')
    logger.info(f'args: {args}')
    logger.info(f'config: {cfg}')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    # synchronize()

    phase = 'train'
    display_freq = cfg.ETC.DISPLAY_FREQ
    save_freq = cfg.ETC.SAVE_FREQ
    model_dump_file = cfg.ETC.MODEL_DUMP_FILE
    model_name = cfg.MODEL.NAME

    model = BaseModel(cfg)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    model.train()

    data_loader = build_data_loader(
        cfg,
        basedata,
        phase=phase,
        is_distributed=True,
        start_iter=0
    )

    optimizer, scheduler = build_optimizer_scheduler(
        cfg,
        model
    )

    max_iter = len(data_loader)
    meters = MetricLogger(delimiter="  ")
    end = time.time()

    try:
        for iteration, (pair_list, target_list, index) in enumerate(data_loader):
            data_time = time.time() - end
            pair_list = [plist.to(gpu) for plist in pair_list]
            target_list = [tlist.to(gpu) for tlist in target_list]

            optimizer.zero_grad()
            loss_dict = model(pair_list, target_list) # 64x132
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            scheduler.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            
            eta_seconds = calculate_eta_iter(meters.time.global_avg, iteration, max_iter)
            eta_string = str(timedelta(seconds=int(eta_seconds)))

            if iteration % display_freq == 0 and is_main_process():
                logger.info(
                    '  '.join(
                        [
                        '[{iter}/{max_iter}]',
                        '{meters}',
                        'lr: {lr:.6f}',
                        'eta: {eta}',
                        'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=iteration+1,
                        max_iter=max_iter,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]['lr'],
                        eta=eta_string,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            if (iteration+1) % save_freq == 0 and is_main_process():
                cfg.ETC.MODEL_DUMP_FILE = '{}_weights_iter_{}.pt'.format(model_name, iteration+1)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': meters.loss.global_avg,
                            'iter': iteration+1},
                            os.path.join(get_model_path(), cfg.ETC.MODEL_DUMP_FILE))

    except KeyboardInterrupt:
        logger.info('Early Stop.')
    else:
        if not is_main_process():
            return

        # save model
        cfg.ETC.MODEL_DUMP_FILE = '{}_weights_iter_{}.pt'.format(model_name, max_iter)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': meters.loss.global_avg,
                    'iter': max_iter},
                    os.path.join(get_model_path(), cfg.ETC.MODEL_DUMP_FILE))

        # save settings
        with open('configs/{}_config.yaml'.format(model_name), 'w') as fout:
            yaml.dump(cfg, fout)

        logger.info('Training Finished Successfully.')