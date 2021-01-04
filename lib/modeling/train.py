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
from .model import RelationPredictor


def train(gpu, args, dataset, param):
    rank = args.local_rank * args.ngpus_per_node + gpu
    logger = setup_logger(name='train', save_dir='logs', distributed_rank=rank, filename=f'{get_timestamp()}_train.txt')
    logger = logging.getLogger('train')
    logger.info(f'args: {args}')
    logger.info(f'param: {param}')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    # synchronize()

    param['phase'] = 'train'
    param['object_num'] = dataset.get_object_num()
    param['predicate_num'] = dataset.get_predicate_num()

    train_data = VRDataset(dataset, param, logger)
    data_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    data_loader = DataLoader(dataset=train_data, batch_size=param['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True, sampler=data_sampler)

    # logger.info('Feature dimension is {}'.format(param['feature_dim']))
    # logger.info('Number of observed training triplets is {}'.format(param['triplet_num']))

    model = RelationPredictor(param)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    # optimizer = torch.optim.SGD(params=model.parameters(), momentum=param['momentum'], lr=param['learning_rate'], weight_decay=param['weight_decay'])
    criterion = nn.BCELoss()

    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    end = time.time()

    for epoch in range(param['max_epoch']):
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
                eta_seconds = calculate_eta(time_meter.avg, epoch, param['max_epoch'], iteration, len(data_loader)) 
                eta_string = str(timedelta(seconds=int(eta_seconds)))

                if iteration % param['display_freq'] == 0 and is_main_process():
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
                            max_epoch=param['max_epoch'],
                            iteration=iteration+1,
                            max_iter=len(data_loader),
                            loss=loss_meter.val,
                            avg_loss=loss_meter.avg,
                            eta=eta_string,
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

            if (epoch+1) % param['save_freq'] == 0 and is_main_process():
                param['model_dump_file'] = '{}_weights_epoch_{}.pt'.format(param['model_name'], epoch+1)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss_meter.avg,
                            'epoch': epoch+1},
                            os.path.join(get_model_path(), param['model_dump_file']))

        except KeyboardInterrupt:
            logger.info('Early Stop.')
            break
    else:
        if not is_main_process():
            return

        # save model
        param['model_dump_file'] = '{}_weights_epoch_{}.pt'.format(param['model_name'], param['max_epoch'])
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_meter.avg,
                    'epoch': epoch+1},
                    os.path.join(get_model_path(), param['model_dump_file']))

    # save settings
    with open(os.path.join(get_model_path(), '{}_setting.json'.format(param['model_name'])), 'w') as fout:
        json.dump(param, fout, indent=4)