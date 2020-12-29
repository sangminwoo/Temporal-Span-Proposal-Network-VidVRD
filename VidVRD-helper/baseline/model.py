import os
import json
import time
from datetime import datetime, timedelta
from itertools import product, cycle
from collections import defaultdict, OrderedDict

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
from .feature import FeatureExtractor
from .utils import AverageMeter, normalize, to_onehot
from .dataset import VRDDataset
from baseline import *

_train_triplet_id = OrderedDict()

def feature_preprocess(feat):
    """ 
    Input feature is extracted according to Section 4.2 in the paper
    """
    # subject classeme + object classeme (70)
    # feat[:, 0: 70]

    # subject TrajectoryShape + HoG + HoF + MBH motion feature (8000)
    # (since this feature is Bag-of-Word type, we l1-normalize it so that
    # each element represents the fraction instead of count)

    feat[:, 70: 1070] = normalize(feat[:, 70: 1070], axis=-1, order=1)
    feat[:, 1070: 2070] = normalize(feat[:, 1070: 2070], axis=-1, order=1)
    feat[:, 2070: 3070] = normalize(feat[:, 2070: 3070], axis=-1, order=1)
    feat[:, 3070: 4070] = normalize(feat[:, 3070: 4070], axis=-1, order=1)

    # object TrajectoryShape + HoG + HoF + MBH motion feature
    feat[:, 4070: 5070] = normalize(feat[:, 4070: 5070], axis=-1, order=1)
    feat[:, 5070: 6070] = normalize(feat[:, 5070: 6070], axis=-1, order=1)
    feat[:, 6070: 7070] = normalize(feat[:, 6070: 7070], axis=-1, order=1)
    feat[:, 7070: 8070] = normalize(feat[:, 7070: 8070], axis=-1, order=1)

    # relative posititon + size + motion feature (3000)
    # feat[:, 8070: 9070]
    # feat[:, 9070: 10070]
    # feat[:, 10070: 11070]
    return feat

class DataGenerator(FeatureExtractor):
    """
    Generate (incl. sample) relation features of (subject, object)s
    in the video segments that have multiple objects detected.
    """
    def __init__(self, dataset, param, logger, prefetch_count=2):
        super(DataGenerator, self).__init__(dataset, logger, prefetch_count)
        self.rng = np.random.RandomState(param['rng_seed'])
        self.batch_size = param['batch_size']
        self.max_sampling_in_batch = param['max_sampling_in_batch']
        assert self.max_sampling_in_batch <= self.batch_size
        self.phase = param['phase']

        logger.info('preparing video segments for {}...'.format(self.phase))
        if self.phase == 'train':
            # initialize training triplet id
            _train_triplet_id.clear()
            triplets = dataset.get_triplets(split='train')
            # triplets = {('car', 'faster', 'bus'), ('car', 'taller', 'antelope'), ...} len:2961
            for i, triplet in enumerate(triplets):
                sub_name, pred_name, obj_name = triplet
                sub_id = dataset.get_object_id(sub_name)
                pred_id = dataset.get_predicate_id(pred_name)
                obj_id = dataset.get_object_id(obj_name)
                _train_triplet_id[(sub_id, pred_id, obj_id)] = i
            # _train_triplet_id = OrderedDict([((2, 68, 4), 0), ((34, 131, 13), 1), ...]) len:2961

            self.short_rel_insts = defaultdict(list)
            video_indices = dataset.get_index(split='train')
            for vid in video_indices:
                for rel_inst in dataset.get_relation_insts(vid, no_traj=True):
                    segs = segment_video(*rel_inst['duration'])
                    for fstart, fend in segs:
                        # if multiple objects detected and the relation features extracted
                        # cache the corresponding groudtruth labels
                        if self.extract_feature(vid, fstart, fend, dry_run=True):
                            sub_name, pred_name, obj_name = rel_inst['triplet']
                            self.short_rel_insts[(vid, fstart, fend)].append((
                                rel_inst['subject_tid'], # 0th
                                rel_inst['object_tid'], # 1st
                                dataset.get_object_id(sub_name), # 2: car
                                dataset.get_predicate_id(pred_name), # 68: faster
                                dataset.get_object_id(obj_name) # 4: bus
                            ))
            self.index = list(self.short_rel_insts.keys())
            self.ind_iter = cycle(range(len(self.index)))
        elif self.phase == 'test':
            self.index = []
            video_indices = dataset.get_index(split='test')
            for vid in video_indices:
                anno = dataset.get_anno(vid)
                segs = segment_video(0, anno['frame_count'])
                # enumerate all the possible segments
                for fstart, fend in segs:
                    # if multiple objects detected and the relation features extracted
                    if self.extract_feature(vid, fstart, fend, dry_run=True):
                        self.index.append((vid, fstart, fend))
            self.ind_iter = iter(range(len(self.index)))
        else:
            raise ValueError('Unknown phase: {}'.format(self.phase))

    def get_data(self):
        if self.phase == 'train':
            feats = []
            triplet_idx = []
            pred_id = [] ###
            remaining_size = self.batch_size
            while remaining_size > 0:
                i = self.ind_iter.__next__()
                vid, fstart, fend = self.index[i]
                sample_num = np.minimum(remaining_size, self.max_sampling_in_batch)
                _feats, _triplet_idx, _pred_id = self._data_sampling(vid, fstart, fend, sample_num)
                remaining_size -= _feats.shape[0]
                _feats = feature_preprocess(_feats)
                feats.append(_feats.astype(np.float32))
                triplet_idx.append(_triplet_idx.astype(np.float32))
                pred_id.append(_pred_id.astype(np.float32)) ###
            feats = np.concatenate(feats)
            triplet_idx = np.concatenate(triplet_idx)
            pred_id = np.concatenate(pred_id) ###
            return feats, triplet_idx, pred_id
        else:
            try:
                i = self.ind_iter.__next__()
            except StopIteration: #, e
                return None
            index = self.index[i]
            pairs, feats, iou, trackid = self.extract_feature( *index)
            test_inds = [ind for ind, (traj1, traj2) in enumerate(pairs)
                    if trackid[traj1] < 0 and trackid[traj2] < 0]
            pairs = pairs[test_inds]
            feats = feature_preprocess(feats[test_inds])
            feats = feats.astype(np.float32)
            return index, pairs, feats, iou, trackid

    def _data_sampling(self, vid, fstart, fend, sample_num, iou_thres=0.5):
        pairs, feats, iou, trackid = self.extract_feature(vid, fstart, fend)
        feats = feats.astype(np.float32)
        pair_to_find = dict([((traj1, traj2), find)
                for find, (traj1, traj2) in enumerate(pairs)])
        tid_to_ind = dict([(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0])

        pos = np.empty((0, 3), dtype = np.int32)
        for tid1, tid2, s, p, o in self.short_rel_insts[(vid, fstart, fend)]:
            if tid1 in tid_to_ind and tid2 in tid_to_ind:
                iou1 = iou[:, tid_to_ind[tid1]]
                iou2 = iou[:, tid_to_ind[tid2]]
                pos_inds1 = np.where(iou1 >= iou_thres)[0]
                pos_inds2 = np.where(iou2 >= iou_thres)[0]
                tmp = [(pair_to_find[(traj1, traj2)], _train_triplet_id[(s, p, o)], p)
                        for traj1, traj2 in product(pos_inds1, pos_inds2) if traj1 != traj2]
                if len(tmp) > 0:
                    pos = np.concatenate((pos, tmp))

        num_pos_in_this = np.minimum(pos.shape[0], sample_num)
        if pos.shape[0] > 0:
            pos = pos[np.random.choice(pos.shape[0], num_pos_in_this, replace=False)]

        return feats[pos[:, 0]], pos[:, 1], pos[:, 2]

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.linear = nn.Linear(param['feature_dim'], param['predicate_num'])

    def forward(self, feats):
        output = self.linear(feats) # 64x11070 -> 64x132
        return output

def train(gpu, args, dataset, param, logger):
    rank = args.local_rank * args.ngpus_per_node + gpu
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

    vrd_dataset = VRDDataset(dataset, param, logger)
    data_sampler = DistributedSampler(vrd_dataset, num_replicas=args.world_size, rank=rank)
    data_loader = DataLoader(dataset=vrd_dataset, batch_size=param['batch_size'], shuffle=False,
        num_workers=param['num_workers'], pin_memory=True, sampler=data_sampler)
    # data_generator = DataGenerator(dataset, param, logger)
    # data_generator = DataGenerator(dataset, param, prefetch_count = 0)
    # param['feature_dim'] = data_generator.get_data_shapes()[0][1]
    logger.info('Feature dimension is {}'.format(param['feature_dim']))
    param['triplet_num'] = len(vrd_dataset._train_triplet_id)
    logger.info('Number of observed training triplets is {}'.format(param['triplet_num']))

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

    for iteration in range(param['max_iter']):
        try:
            # feats, triplet_idx, pred_id = data_generator.get_prefected_data() # feats: 64x11070 (batch_Size x feature_dim), target: 64 (batch_size)
            for idx, (feats, _, pred_id) in enumerate(data_loader):
                print('feats', feats.shape)
                feats = torch.tensor(feats).cuda(non_blocking=True)
                target = torch.tensor(pred_id, dtype=torch.long).cuda(non_blocking=True)

                optimizer.zero_grad()
                output = model(feats)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                loss_meter.update(float(loss))

            batch_time = time.time() - end
            end = time.time()
            time_meter.update(batch_time)
            eta_seconds = time_meter.avg * (param['max_iter'] - iteration)
            eta_string = str(timedelta(seconds=int(eta_seconds)))

            if iteration % param['display_freq'] == 0 and gpu == 0:
                logger.info(
                    '  '.join(
                        [
                        'iter: [{iter}/{max_iter}]',
                        'loss: {loss:.4f} ({avg_loss:.4f})',
                        'eta: {eta}',
                        'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=iteration,
                        max_iter=param['max_iter'],
                        loss=loss_meter.val,
                        avg_loss=loss_meter.avg,
                        eta=eta_string,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            if iteration % param['save_freq'] == 0 and iteration > 0:
                param['model_dump_file'] = '{}_weights_iter_{}.pt'.format(param['model_name'], iteration)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss_meter.avg,
                            'iter': iteration},
                            os.path.join(get_model_path(), param['model_dump_file']))

        except KeyboardInterrupt:
            logger.info('Early Stop.')
            break
    else:
        # save model
        param['model_dump_file'] = '{}_weights_iter_{}.pt'.format(param['model_name'], param['max_iter'])
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_meter.avg,
                    'iter': it},
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
