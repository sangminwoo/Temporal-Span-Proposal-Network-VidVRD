import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .model import BaseModel
from lib.modeling import *
from lib.dataset.vrdataset import VRDataset
from lib.utils.serialize import load_checkpoint


def predict(cfg, basedata, logger):
    cfg.MODEL.PHASE = 'test'
    batch_size = cfg.DATASET.TEST_BATCH_SIZE
    num_workers = cfg.DATASET.TEST_NUM_WORKERS
    topk_per_pair = cfg.PREDICT.TOPK_PER_PAIR
    topk_per_seg = cfg.PREDICT.TOPK_PER_SEG

    test_data = VRDataset(cfg, basedata, logger)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    # load model
    model = BaseModel(cfg)
    checkpoint = torch.load(os.path.join(get_model_path(), cfg.ETC.MODEL_DUMP_FILE))
    load_checkpoint(model, checkpoint['model'])
    logger.info(f"=> checkpoint succesfully loaded")
    logger.info(f"=> epoch: {checkpoint['epoch']}")
    logger.info(f"=> average loss:{checkpoint['loss']:.4f}")
    model.eval()

    logger.info('predicting short-term visual relation...')
    pbar = tqdm(total=len(data_loader))
    predictions = dict()

    with torch.no_grad():
        for batch_i, (index, pairs_per_seg, feats_per_seg, iou_per_seg, trackid_per_seg) \
            in enumerate(data_loader):
            '''
            index [vid, fstart, fend]
            [('ILSVRC2015_train_00219001',), tensor([0]), tensor([30])]
            '''
            vid, fstart, fend = index
            vid = vid[0]
            fstart = int(fstart)
            fend = int(fend)
            index = (vid, fstart, fend)

            for segment_i, (pairs, feats, iou, trackid) \
                in enumerate(zip(pairs_per_seg, feats_per_seg, iou_per_seg, trackid_per_seg)):
                '''
                N: number of object proposals per segment
                GT: number of ground-truth objects per segment
                N(N-1): number of all possible relations per segment

                pairs ( shape: N(N-1)x2 )
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
                assert len(pairs) > 0, \
                    f"There is no object tracklet proposals in {index}"
                prob_p = model(feats)
                prob_s = feats[:, :35]
                prob_o = feats[:, 35: 70]

                sub_label = torch.argmax(prob_s, dim=1) # N(N-1)
                obj_label = torch.argmax(prob_o, dim=1) # N(N-1)

                # pick top-20 predictions per pair
                topk_pred_per_pair = torch.sort(prob_p, dim=-1) 
                topk_score_per_pair = topk_pred_per_pair[0][:topk_per_pair] # N(N-1) x topk
                topk_idx_per_pair = topk_pred_per_pair[1][:topk_per_pair] # N(N-1) x topk
                r, c = topk_score_per_pair.shape

                # pick top-200 predictions per segment
                topk_pred_per_seg = torch.sort(topk_score_per_pair.flatten(), descending=True, dim=-1)
                topk_score_per_seg = topk_pred_per_seg[0][:topk_per_seg] # K
                topk_idx_per_seg = topk_pred_per_seg[1][:topk_per_seg] # K
                topk_idx = torch.tensor(
                    [(idx // c, idx % c) for idx in topk_idx_per_seg]
                ) # topK x 2
                top_pair_idx = topk_idx[:, 0] # topK in N(N-1)

                # get tracklet pairs' ids
                num_objs = len(torch.nonzero(trackid==-1).view(-1)) # N
                top_pair_tid = torch.tensor([(idx//(num_objs-1), idx%(num_objs-1)) for idx in top_pair_idx]) # topK x 2
                
                # get relation triplet labels
                top_sub_label = sub_label[top_pair_idx] # topK in N(N-1)
                top_obj_label = obj_label[top_pair_idx] # topK in N(N-1)
                top_rel_label = torch.tensor(
                    [topk_idx_per_pair[idx[0], idx[1]] for idx in topk_idx]
                ) # topK in 132
                top_triplet_label = torch.stack(
                    [top_sub_label, top_rel_label, top_obj_label]
                ).t() # topK x 3

                # get relation scores
                top_rel_score = torch.tensor(
                    [topk_score_per_pair[idx[0], idx[1]] for idx in topk_idx]
                ) # topK in R
                
                predictions[index] = (
                    top_pair_tid,
                    top_triplet_label,
                    top_rel_score
                )
                # from pprint import pprint; pprint(predictions)
                # raise ValueError
            pbar.update(1)
        pbar.close()

    return predictions