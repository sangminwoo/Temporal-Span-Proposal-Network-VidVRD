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

from dataset import VidVRD
from baseline import segment_video, get_model_path
from baseline import trajectory, feature, model, association, vrdataset
from baseline.logger import setup_logger, get_timestamp

def load_object_trajectory_proposal(data_dir):
    """
    Test loading precomputed object trajectory proposals
    """
    dataset = VidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)


def load_relation_feature(data_dir):
    """
    Test loading precomputed relation features
    """
    dataset = VidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])
    extractor = feature.FeatureExtractor(dataset, prefetch_count=0)

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                extractor.extract_feature(vid, fstart, fend, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            extractor.extract_feature(vid, fstart, fend, verbose=True)


def preprocessing(args, data_dir):
    dataset = VidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])

    with open('default.json', 'r') as fin:
        param = json.load(fin)

    logger = setup_logger(name='preprocess', save_dir='logs', distributed_rank=0, filename=f'{get_timestamp()}_preprocess.txt')
    logger = logging.getLogger('preprocess')
    logger.info(f'args: {args}')
    logger.info(f'param: {param}')

    feats, pred_id = vrdataset.preprocess_data(dataset, param, logger)

    path = os.path.join('./vidvrd-baseline-output', 'preprocessed_data')
    if not os.path.exists(path):
        os.makedirs(path)

    logger.info('saving preprocessed data...')
    with h5py.File(os.path.join(path,'preprocessed_train_dataset.hdf5'), 'a') as f:
        f['feats'] = feats
        f['pred_id'] = pred_id

    logger.info('successfully saved preprocessed data...')


def train(args, data_dir):
    dataset = VidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])

    with open('default.json', 'r') as fin:
        param = json.load(fin)

    # param['batch_size'] = int(param['batch_size'] / args.ngpus_per_node)
    # param['max_sampling_in_batch'] = int(param['max_sampling_in_batch'] / args.ngpus_per_node)
    # param['num_workers'] = int(param['num_workers'] / args.ngpus_per_node)

    # distributed
    args.world_size = args.ngpus_per_node * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(model.train, nprocs=args.ngpus_per_node, args=(args, dataset, param))


def detect(data_dir):
    dataset = VidVRD(data_dir, os.path.join(data_dir, 'videos'), ['train', 'test'])

    with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
        param = json.load(fin)

    logger = setup_logger(name='detect', save_dir='logs', distributed_rank=0, filename=f'{get_timestamp()}_detect.txt')
    logger = logging.getLogger('detect')
    logger.info(f'args: {args}')
    logger.info(f'param: {param}')

    logger.info('predict short term relations')
    short_term_relations = model.predict(dataset, param, logger)

    logger.info('group short term relations by video')
    video_st_relations = defaultdict(list)
    for index, st_rel in short_term_relations.items():
        vid = index[0]
        video_st_relations[vid].append((index, st_rel))

    logger.info('video-level visual relation detection by greedy relational association')
    video_relations = dict()
    for vid in tqdm(video_st_relations.keys()):
        video_relations[vid] = association.greedy_relational_association(
                dataset, video_st_relations[vid], max_traj_num_in_clip=100)

    logger.info('saving detection result')
    with open(os.path.join(get_model_path(), 'baseline_relation_prediction.json'), 'w') as fout:
        output = {
            'version': 'VERSION 1.0',
            'results': video_relations
        }
        json.dump(output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VidVRD baseline')
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='the dataset name for training')
    parser.add_argument('--load_feature', action='store_true', default=False, help='Test loading precomputed features')
    parser.add_argument('--preprocess', action='store_true', default=False, help='Preprocess dataset')
    parser.add_argument('--train', action='store_true', default=False, help='Train model')
    parser.add_argument('--detect', action='store_true', default=False, help='Detect video visual relation')
    parser.add_argument('--nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--ngpus_per_node', type=int, default=1, help='Number of gpus per node')
    parser.add_argument('--local_rank', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    if args.load_feature or args.train or args.detect or args.preprocess:
        if args.load_feature:
            load_object_trajectory_proposal(os.path.join(args.data_dir, args.dataset))
            load_relation_feature(os.path.join(args.data_dir, args.dataset))
        if args.preprocess:
            preprocessing(args, os.path.join(args.data_dir, args.dataset))
        if args.train:
            train(args, os.path.join(args.data_dir, args.dataset))
        if args.detect:
            detect(os.path.join(args.data_dir, args.dataset))
    else:
        parser.print_help()
