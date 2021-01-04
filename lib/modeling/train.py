import os
import json
import logging
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.dataset.vrdataset import VRDataset
from lib.utils.comm import synchronize, is_main_process
from lib.utils.miscellaneous import AverageMeter, calculate_eta 
from lib.utils.logger import setup_logger, get_timestamp
from lib.modeling import *
from lib.modeling.relpn import relpn
from .model import RelationPredictor

def train(gpu, cfg, args, dataset):
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

    cfg.MODEL.PHASE = 'train'
    # cfg.PREDICT.OBJECT_NUM = dataset.get_object_num()
    # cfg.PREDICT.PREDICATE_NUM = dataset.get_predicate_num()
    max_epoch = cfg.SOLVER.MAX_EPOCH
    batch_size = cfg.MODEL.TRAIN_BATCH_SIZE
    num_workers = cfg.DATALOADER.TRAIN_NUM_WORKERS
    lr = cfg.SOLVER.LEARNING_RATE
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY 
    display_freq = cfg.ETC.DISPLAY_FREQ
    save_freq = cfg.ETC.SAVE_FREQ
    model_dump_file = cfg.ETC.MODEL_DUMP_FILE
    model_name = cfg.MODEL.NAME

    train_data = VRDataset(cfg, dataset, logger)
    data_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=data_sampler
    )

    model = RelationPredictor(cfg)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    end = time.time()

    for epoch in range(max_epoch):
        try:
            # feats: 64x11070 (batch_Size x feature_dim), pred_id: 64 (batch_size)
            for iteration, (feats, pred_id) in enumerate(data_loader):
                feats = feats.to(gpu)
                target = pred_id.to(gpu)

                optimizer.zero_grad()
                output = model(feats) # 64x132

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                loss_meter.update(float(loss))
                # loss_meter.update(float(loss), output.shape[0])

                batch_time = time.time() - end
                end = time.time()
                time_meter.update(batch_time)
                # time_meter.update(batch_time, output.shape[0])
                eta_seconds = calculate_eta(time_meter.avg, epoch, max_epoch, iteration, len(data_loader)) 
                eta_string = str(timedelta(seconds=int(eta_seconds)))

                if iteration % display_freq == 0 and is_main_process():
                    logger.info(
                        '  '.join(
                            [
                            '[{epoch}/{max_epoch}][{iteration}/{max_iter}]',
                            'loss: {loss:.4f} ({avg_loss:.4f})',
                            'eta: {eta}',
                            'max mem: {memory:.0f}',
                            ]
                        ).format(
                            epoch=epoch+1,
                            max_epoch=max_epoch,
                            iteration=iteration+1,
                            max_iter=len(data_loader),
                            loss=loss_meter.val,
                            avg_loss=loss_meter.avg,
                            eta=eta_string,
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

            if (epoch+1) % save_freq == 0 and is_main_process():
                model_dump_file = '{}_weights_epoch_{}.pt'.format(model_name, epoch+1)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss_meter.avg,
                            'epoch': epoch+1},
                            os.path.join(get_model_path(), model_dump_file))

        except KeyboardInterrupt:
            logger.info('Early Stop.')
            break
    else:
        if not is_main_process():
            return

        # save model
        model_dump_file = '{}_weights_epoch_{}.pt'.format(model_name, max_epoch)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_meter.avg,
                    'epoch': epoch+1},
                    os.path.join(get_model_path(), model_dump_file))

    # save settings
    with open(os.path.join(get_model_path(), '{}_config.yaml'.format(model_name)), 'w') as fout:
        fout.write(cfg.dump())