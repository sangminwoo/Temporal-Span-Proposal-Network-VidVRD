import logging
import os
import json
import argparse
import _pickle as pkl
from collections import defaultdict
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import h5py

from lib.config import cfg
from lib.dataset import vrdataset, BaseVidVRD, BaseVidOR
from lib.modeling import *
from lib.modeling import association
from lib.modeling.train import train
from lib.modeling.predict import predict
from lib.utils.logger import setup_logger, get_timestamp


def preprocessing(cfg, args, data_dir):
    dataset = BaseVidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])

    logger = setup_logger(name='preprocess', save_dir='logs', distributed_rank=0, filename=f'{get_timestamp()}_preprocess.txt')
    logger = logging.getLogger('preprocess')
    logger.info(f'args: {args}')
    logger.info(f'cfg: {cfg}')

    feats, pairs, pred_label = vrdataset.preprocess_data(cfg, dataset, logger)

    path = os.path.join('./vidvrd-baseline-output', 'preprocessed_data')
    if not os.path.exists(path):
        os.makedirs(path)

    logger.info('saving preprocessed data...')
    with h5py.File(os.path.join(path,'preprocessed_train_dataset.hdf5'), 'a') as f:
        f['feats'] = feats
        f['pairs'] = pairs
        f['pred_label'] = pred_label

    logger.info('successfully saved preprocessed data...')


def training(cfg, args, data_dir):
    if args.dataset == 'vidvrd':
        basedata = BaseVidVRD(
            data_dir,
            os.path.join(data_dir, 'videos'),
            ['train', 'test']
        )
    elif args.dataset == 'vidor':
        basedata = BaseVidOR(
            os.path.join(data_dir, 'annotation'),
            os.path.join(data_dir, 'videos'),
            ['train', 'test']
        )
    else:
        raise ValueError(f"No dataset named {args.dataset}")

    # distributed
    args.world_size = args.ngpus_per_node * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(train, nprocs=args.ngpus_per_node, args=(cfg, args, basedata))


def detect(cfg, args, data_dir):
    if args.dataset == 'vidvrd':
        basedata = BaseVidVRD(
            data_dir,
            os.path.join(data_dir, 'videos'),
            ['train', 'test']
        )
    elif args.dataset == 'vidor':
        basedata = BaseVidOR(
            os.path.join(data_dir, 'annotation'),
            os.path.join(data_dir, 'videos'),
            ['train', 'test']
        )
    else:
        raise ValueError(f"No dataset named {args.dataset}")

    logger = setup_logger(name='detect', save_dir='logs', distributed_rank=0, filename=f'{get_timestamp()}_detect.txt')
    logger = logging.getLogger('detect')
    logger.info(f'args: {args}')
    logger.info(f'cfg: {cfg}')

    logger.info('predict short term relations')
    short_term_relations = predict(cfg, basedata, logger)

    logger.info('group short term relations by video')
    video_st_relations = defaultdict(list)
    for index, st_rel in short_term_relations.items():
        vid = index[0]
        video_st_relations[vid].append((index, st_rel))

    logger.info('video-level visual relation detection by greedy relational association')
    video_relations = dict()
    for vid in tqdm(video_st_relations.keys()):
        video_relations[vid] = association.greedy_relational_association(
            basedata,
            video_st_relations[vid],
            max_traj_num_in_clip=100
        )

    logger.info('saving detection result')
    with open(os.path.join(get_model_path(), 'baseline_relation_prediction.json'), 'w') as fout:
        output = {
            'version': 'VERSION 1.0',
            'results': video_relations
        }
        json.dump(output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VidVRD baseline')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='the dataset name for training')
    parser.add_argument('--preprocess', action='store_true', default=False, help='Preprocess dataset')
    parser.add_argument('--train', action='store_true', default=False, help='Train model')
    parser.add_argument('--detect', action='store_true', default=False, help='Detect video visual relation')
    parser.add_argument('--nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--ngpus_per_node', type=int, default=1, help='Number of gpus per node')
    parser.add_argument('--local_rank', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    if args.train or args.detect or args.preprocess:
        if args.preprocess:
            preprocessing(cfg, args, os.path.join(args.data_dir, args.dataset))
        if args.train:
            training(cfg, args, os.path.join(args.data_dir, args.dataset))
        if args.detect:
            detect(cfg, args, os.path.join(args.data_dir, args.dataset))
    else:
        parser.print_help()
