import os
import json
import logging
import time
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .vrdataset import VRDataset
# from .feature import FeatureExtractor
from .comm import synchronize, is_main_process
from .utils import AverageMeter, calculate_eta 
from .logger import setup_logger, get_timestamp
from .serialize import load_checkpoint
# from .duration_proposal import duration_proposal_loss
from baseline import *


class RelationPredictor(nn.Module):
    def __init__(self, param):
        super(RelationPredictor, self).__init__()
        self.classifier = nn.Linear(param['feature_dim'], param['predicate_num'])

    def forward(self, feats):
        relation = self.classifier(feats) # 64x11070 -> 64x132
        relation = torch.sigmoid(relation)
        return relation

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
    bce_loss = nn.BCELoss()

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

                loss = bce_loss(output, target)
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


def predict(dataset, param, logger):
    param['phase'] = 'test'

    # load model
    model = RelationPredictor(param)
    checkpoint = torch.load(os.path.join(get_model_path(), param['model_dump_file']))
    load_checkpoint(model, checkpoint['model'])
    logger.info(f"=> checkpoint succesfully loaded")
    logger.info(f"=> epoch: {checkpoint['epoch']}")
    logger.info(f"=> average loss:{checkpoint['loss']:.4f}")
    model.eval()

    test_data = VRDataset(dataset, param, logger)
    data_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    path = os.path.join('vidvrd-baseline-output', 'preprocessed_data')
    logger.info('predicting short-term visual relation...')
    pbar = tqdm(total=len(data_loader))

    short_term_relations = dict()
    with torch.no_grad():
        for index, pairs_, feats_, iou_, trackid_ in data_loader:
            for pairs, feats, iou, trackid in zip(pairs_, feats_, iou_, trackid_):
                '''
                P: num of all possible relations per segment
                GT: num of ground-truth relations per segment

                index [vid, fstart, fend]
                [('ILSVRC2015_train_00219001',), tensor([0]), tensor([30])]
                
                pairs ( shape: P(P-1)x2 )
                tensor([[[ 0,  1],
                         [ 0,  2],
                         [ 0,  3],
                         ...
                         [17, 14],
                         [17, 15],
                         [17, 16]]])

                feats ( shape: P(P-1)x11070 )
                tensor([[[8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         ...,
                         [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00]]])

                iou ( shape: (P+GT)x(P+GT) )
                tensor([[[1.0000, 0.0000, 0.1238, ..., 0.6484, 0.1181, 0.0000],
                         [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.7938],
                         ...
                         [0.1181, 0.0000, 0.7218, ..., 0.0382, 1.0000, 0.0000],
                         [0.0000, 0.7938, 0.0000, ..., 0.0000, 0.0000, 1.0000]]])

                trackid ( shape: (P+GT) / proposal: -1, GT: 0,1,2)
                tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2]])
                '''
                vid, fstart, fend = index
                vid = vid[0]
                fstart = fstart.int()
                fend = fend.int()
                index = (vid, fstart, fend)

                prob_p = model(feats).numpy()
                prob_s = feats[:, :35].numpy()
                prob_o = feats[:, 35: 70].numpy()
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

                pbar.update(1)
        pbar.close()

    return short_term_relations


    # short_term_relations = dict()
    # predictions = []
    # idx = 0
    # cur = 0
    # track_per_seg = len(iou[idx])
    # track_proposal_per_seg = sum(trackid[cur:cur+track_per_seg] < 0)
    # pair_per_seg = track_proposal_per_seg * (track_proposal_per_seg-1)

    # with torch.no_grad():
    #     for iteration, (pairs, feats) in enumerate(data_loader):
    #         prob_s = feats[:, :35].numpy()
    #         prob_p = model(feats).numpy()
    #         prob_o = feats[:, 35: 70].numpy()

    #         sub_idx = np.argsort(-prob_s)[:, :1]
    #         obj_idx = np.argsort(-prob_o)[:, :1]
    #         topk_pred_ind = np.argsort(-prob_p)[:, :param['pair_topk']]
    #         topk_prob_p = np.sort(-prob_p)[:, :param['pair_topk']]

    #         for j in range(param['pair_topk']):
    #             predictions.append(
    #                 [
    #                     topk_prob_p[:, j],
    #                     (sub_idx, topk_pred_ind[:, j], obj_idx),
    #                     (np.array(pairs[0]), np.array(pairs[1]))
    #                 ]
    #             )

    #         if iteration+1 == pair_per_seg:
    #             predictions
    #             predictions = sorted(predictions, key=lambda x: x[0], reverse=True)[:param['seg_topk']]
    #             short_term_relations[index[idx]] = (predictions, iou[idx], trackid[cur:cur+track_per_seg])
    #             predictions = []
    #             idx += 1
    #             cur = cur+track_per_seg
    #             track_per_seg = len(iou[idx])
    #             track_proposal_per_seg = sum(trackid[cur:cur+track_per_seg] < 0)
    #             pair_per_seg = track_proposal_per_seg * (track_proposal_per_seg-1)

    #         pbar.update(1)
    # pbar.close()
    # return short_term_relations

    # with torch.no_grad():
    #     for iteration, (index, pairs, feats, iou, trackid) in enumerate(data_loader):
    #         # vid, fstart, fend = index
    #         vid = vid[0]
    #         fstart = fstart.int()
    #         fend = fend.int()
    #         # index = (vid, fstart, fend)

    #         prob_p = model(feats).numpy()
    #         prob_s = feats[:, :35].numpy()
    #         prob_o = feats[:, 35: 70].numpy()
    #         predictions = []
    #         for i in range(len(pairs)):
    #             top_s_ind = np.argsort(prob_s[i])[-param['pair_topk']:]
    #             top_p_ind = np.argsort(prob_p[i])[-param['pair_topk']:]
    #             top_o_ind = np.argsort(prob_o[i])[-param['pair_topk']:]
    #             score = prob_s[i][top_s_ind, None, None]*prob_p[i][None, top_p_ind, None]*prob_o[i][None, None, top_o_ind]
    #             top_flat_ind = np.argsort(score, axis = None)[-param['pair_topk']:]
    #             top_score = score.ravel()[top_flat_ind]
    #             top_s, top_p, top_o = np.unravel_index(top_flat_ind, score.shape)
    #             predictions.extend((
    #                     top_score[j], 
    #                     (top_s_ind[top_s[j]], top_p_ind[top_p[j]], top_o_ind[top_o[j]]), 
    #                     tuple(pairs[i])) 
    #                     for j in range(top_score.size))
    #         predictions = sorted(predictions, key=lambda x: x[0], reverse=True)[:param['seg_topk']]
    #         short_term_relations[index] = (predictions, iou, trackid)

    #         pbar.update(1)
    #     pbar.close()

    # return short_term_relations