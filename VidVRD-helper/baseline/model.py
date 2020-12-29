import os
import json
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .comm import synchronize
# from .feature import FeatureExtractor
from .utils import AverageMeter, setup_logger, get_timestamp, calculate_eta, normalize, to_onehot
# from .dataset import VRDDataset
from .preprocessed_dataset import VRDDataset
from baseline import *

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.linear = nn.Linear(param['feature_dim'], param['predicate_num'])

    def forward(self, feats):
        output = self.linear(feats) # 64x11070 -> 64x132
        return output

def train(gpu, args, dataset, param):
    rank = args.local_rank * args.ngpus_per_node + gpu
    logger = setup_logger(name='vidvrd', save_dir='logs', distributed_rank=rank, filename=f'{get_timestamp()}_vidvrd.txt')
    logger = logging.getLogger('vidvrd')
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

    vrd_dataset = VRDDataset(param, logger)
    data_sampler = DistributedSampler(vrd_dataset, num_replicas=args.world_size, rank=rank)
    data_loader = DataLoader(dataset=vrd_dataset, batch_size=param['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True, sampler=data_sampler)

    # logger.info('Feature dimension is {}'.format(param['feature_dim']))
    # logger.info('Number of observed training triplets is {}'.format(param['triplet_num']))

    model = Model(param)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    # optimizer = torch.optim.SGD(params=model.parameters(), momentum=param['momentum'], lr=param['learning_rate'], weight_decay=param['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    end = time.time()

    for epoch in range(param['max_epoch']):
        try:
            # feats: 64x11070 (batch_Size x feature_dim), pred_id: 64 (batch_size)
            for iteration, (feats, triplet_idx, pred_id) in enumerate(data_loader):
                feats = feats.to(gpu)
                target = pred_id.long().to(gpu)

                optimizer.zero_grad()
                output = model(feats)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                loss_meter.update(float(loss))

                batch_time = time.time() - end
                end = time.time()
                time_meter.update(batch_time)
                eta_seconds = calculate_eta(time_meter.avg, epoch, param['max_epoch'], iteration, len(data_loader)) 
                eta_string = str(timedelta(seconds=int(eta_seconds)))

                if iteration % param['display_freq'] == 0 and gpu == 0:
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

            if epoch % param['save_freq'] == 0 and epoch > 0:
                param['model_dump_file'] = '{}_weights_epoch_{}.pt'.format(param['model_name'], epoch)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss_meter.avg,
                            'epoch': epoch},
                            os.path.join(get_model_path(), param['model_dump_file']))

        except KeyboardInterrupt:
            logger.info('Early Stop.')
            break
    else:
        # save model
        param['model_dump_file'] = '{}_weights_epoch_{}.pt'.format(param['model_name'], param['max_epoch'])
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_meter.avg,
                    'epoch': epoch},
                    os.path.join(get_model_path(), param['model_dump_file']))
    # save settings
    with open(os.path.join(get_model_path(), '{}_setting.json'.format(param['model_name'])), 'w') as fout:
        json.dump(param, fout, indent=4)


def predict(dataset, param, logger):
    param['phase'] = 'test'
    data_generator = DataGenerator(dataset, param, logger)
    # load model
    model = Model(param)
    model.load_state_dict(torch.load(os.path.join(get_model_path(), param['model_dump_file'])))
    model.eval()

    logger.info('predicting short-term visual relation...')
    pbar = tqdm(total=len(data_generator.index))
    short_term_relations = dict()
    # do not support prefetching mode in test phase
    data = data_generator.get_data()
    while data:
        # get all possible pairs and the respective features and annos
        index, pairs, feats, iou, trackid = data
        # make prediction
        prob_p = model(feats)
        prob_s = feats[:, :35]
        prob_o = feats[:, 35: 70]
        predictions = []
        for i in range(len(pairs)):
            top_s_ind = np.argsort(prob_s[i])[-param['pair_topk']:]
            top_p_ind = np.argsort(prob_p[i])[-param['pair_topk']:]
            top_o_ind = np.argsort(prob_o[i])[-param['pair_topk']:]
            score = prob_s[i][top_s_ind, None, None]*prob_p[i][None, top_p_ind, None]*prob_o[i][None, None, top_o_ind]
            top_flat_ind = np.argsort(score, axis = None)[-param['pair_topk']:]
            top_score = score.ravel()[top_flat_ind]
            top_s, top_p, top_o = np.unravel_index(top_flat_ind, score.shape)
            predictions.extend((
                    top_score[j], 
                    (top_s_ind[top_s[j]], top_p_ind[top_p[j]], top_o_ind[top_o[j]]), 
                    tuple(pairs[i])) 
                    for j in range(top_score.size))
        predictions = sorted(predictions, key=lambda x: x[0], reverse=True)[:param['seg_topk']]
        short_term_relations[index] = (predictions, iou, trackid)

        data = data_generator.get_data()
        pbar.update(1)

    pbar.close()
    return short_term_relations
