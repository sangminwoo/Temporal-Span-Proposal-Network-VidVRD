import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .model import BaseModel
from lib.modeling import *
from lib.dataset.vrdataset import VRDataset
from lib.dataset.build import build_data_loader
from lib.utils.serialize import load_checkpoint


def predict(cfg, basedata, logger):
    phase = 'test'
    batch_size = cfg.DATASET.TEST_BATCH_SIZE
    num_workers = cfg.DATASET.TEST_NUM_WORKERS
    topk_per_pair = cfg.PREDICT.TOPK_PER_PAIR
    topk_per_seg = cfg.PREDICT.TOPK_PER_SEG

    data_loader = build_data_loader(
        cfg,
        basedata,
        phase=phase,
        is_distributed=False,
        start_iter=0
    )

    # load model
    model = BaseModel(cfg)
    checkpoint = torch.load(os.path.join(get_model_path(), cfg.ETC.MODEL_DUMP_FILE))
    load_checkpoint(model, checkpoint['model'])
    logger.info(f"=> checkpoint succesfully loaded")
    logger.info(f"=> iter: {checkpoint['iter']}")
    logger.info(f"=> average loss:{checkpoint['loss']:.4f}")
    model.eval()

    logger.info('predicting short-term visual relation...')
    pbar = tqdm(total=len(data_loader))
    short_term_relations = dict()

    with torch.no_grad():
        for iteration, (pair_list, _, indexs) in enumerate(data_loader):
            features = [plist.features for plist in pair_list]
            tracklet_pairs = [plist.get_field('tracklet_pairs') for plist in pair_list]
            track_cls_logits = [plist.get_field('track_cls_logits') for plist in pair_list]
            num_tracklets = [plist.get_field('num_tracklets') for plist in pair_list]
            ious = [plist.get_field('ious') for plist in pair_list]
            trackids = [plist.get_field('track_ids') for plist in pair_list]
            # assert len(pairs) > 0, \
            #     f"There is no object tracklet proposals in {index}"
            pair_proposals, duration_proposals, rel_logits = model(pair_list, _)
            
            for index, rel_logit, tracklet_pair, track_cls_logit, num_tracklet, iou, trackid in \
                zip(indexs, rel_logits, tracklet_pairs, track_cls_logits, num_tracklets, ious, trackids):
                if num_tracklet <= 1:
                    logger.info(f'No relation exists in video segment {index}')
                    pbar.update(1)
                    continue

                # pick top-20 predictions per pair
                topk_pred_per_pair = torch.sort(rel_logit, descending=True, dim=-1) 
                topk_score_per_pair = topk_pred_per_pair[0][:, :topk_per_pair] # N(N-1) x top-k
                topk_idx_per_pair = topk_pred_per_pair[1][:, :topk_per_pair] # N(N-1) x top-k
                r, c = topk_score_per_pair.shape # N(N-1), top-k

                # pick top-200 predictions per segment
                topk_pred_per_seg = torch.sort(topk_score_per_pair.flatten(), descending=True, dim=-1)
                # topk_score_per_seg = topk_pred_per_seg[0][:topk_per_seg] # top-K
                topk_idx_per_seg = topk_pred_per_seg[1][:topk_per_seg] # top-K
                topk_idx = torch.tensor(
                    [(idx // c, idx % c) for idx in topk_idx_per_seg]
                ) # top-K x 2
                
                # get tracklet pairs' ids
                top_pair_idx = topk_idx[:, 0] # top-K in N(N-1)
                top_pair_tid = tracklet_pair[top_pair_idx] # top-K x 2

                # get object labels
                top_sub_logit = track_cls_logit[top_pair_tid[:, 0]] # top-K x 35 
                top_obj_logit = track_cls_logit[top_pair_tid[:, 1]] # top-K x 35

                top_sub_label = torch.argmax(top_sub_logit, dim=1) # top-K
                top_obj_label = torch.argmax(top_obj_logit, dim=1) # top-K
                
                # get relation labels
                top_rel_label = torch.tensor(
                    [topk_idx_per_pair[idx[0], idx[1]] for idx in topk_idx]
                ) # top-K in 132

                top_triplet_label = torch.stack(
                    [top_sub_label, top_rel_label, top_obj_label]
                ).t() # top-K x 3

                # get relation scores
                top_rel_score = torch.tensor(
                    [topk_score_per_pair[idx[0], idx[1]] for idx in topk_idx]
                ) # top-K in R
                
                predictions = [
                    (score, triplet, pair_tid) for score, triplet, pair_tid \
                        in zip(top_rel_score, top_triplet_label, top_pair_tid)
                ]

                short_term_relations[index] = (
                    predictions,
                    iou,
                    trackid
                )
                # from pprint import pprint; pprint(short_term_relations)
                # raise ValueError
                pbar.update(1)
    pbar.close()

    return short_term_relations

'''
N: number of object proposals per segment
GT: number of ground-truth objects per segment
N(N-1): number of all possible relations per segment

index [vid, fstart, fend]
[('ILSVRC2015_train_00219001',), tensor([0]), tensor([30])]            

tracklet_pairs ( shape: N(N-1)x2 )
tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        ...
        [17, 14],
        [17, 15],
        [17, 16]])

feats ( shape: N(N-1)x11070 )
tensor([[8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [8.2992e-08, 1.2913e-05, 4.4002e-08,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        ...,
        [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [3.9329e-05, 1.3184e-04, 4.6081e-05,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00]])

iou ( shape: (N+GT)x(N+GT) )
tensor([[1.0000, 0.0000, 0.1238, ..., 0.6484, 0.1181, 0.0000],
        [0.0000, 1.0000, 0.0000, ..., 0.0000, 0.0000, 0.7938],
        ...
        [0.1181, 0.0000, 0.7218, ..., 0.0382, 1.0000, 0.0000],
        [0.0000, 0.7938, 0.0000, ..., 0.0000, 0.0000, 1.0000]])

trackid ( shape: (N+GT) / proposal: -1, GT: 0,1,2)
tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2])
'''