import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import BalancedPositiveNegativePairSampler

class PPN(nn.Module):
    '''
    Pair Proposal Network: "What object pairs are related? Are object A and B related?"
    '''
    def __init__(self, cfg):
        super(PPN, self).__init__()
        self.num_pair_proposals = cfg.RELPN.PPN.NUM_PAIR_PROPOSALS
        self.ppn_head = PPNHead(
            in_channels=cfg.RELPN.PPN.IN_CHANNELS,
            hidden_channels=cfg.RELPN.PPN.HIDDEN_CHANNELS,
            out_channels=cfg.RELPN.PPN.OUT_CHANNELS,
        )

        self.fg_bg_sampler = BalancedPositiveNegativePairSampler(
          batch_size_per_image=cfg.RELPN.PPN.BATCH_SIZE_PER_SEGMENT,
          positive_fraction=cfg.RELPN.PPN.POSITIVE_FRACTION
        )

    def _predict_pair_matrices(self, pair_list):
        cls_logits = [plist.get_field('track_cls_logits') for plist in pair_list]
        
        pair_matrices = []
        for cls_logit in cls_logits:
            pair_matrices.append(
                self.ppn_head(cls_logit, cls_logit)
            )

        return pair_matrices

    def _generate_nxn_gt_matrices(self, pair_list, target_list):
        track_pairs = [plist.get_field('tracklet_pairs') for plist in pair_list] # batch x num_pairs_per_seg x 2
        num_tracks = [plist.get_field('num_tracklets') for plist in pair_list] # batch x 1
        pred_labels = [tlist.target for tlist in target_list] # batch x num_pairs_per_seg x 132

        gt_matrices = []
        for track_pair, num_track, pred_label in zip(track_pairs, num_tracks, pred_labels):
            gt_matrix = torch.zeros(num_track, num_track)
            for p, lbl in zip(track_pair, pred_label):
                if sum(lbl) > 0:
                    gt_matrix[p[0], p[1]] = 1
            gt_matrices.append(gt_matrix)

        return gt_matrices

    def forward(self, pair_list, target_list=None):
        if self.training:
            return self._forward_train(pair_list, target_list)
        else:
            return self._forward_test(pair_list)

    def _forward_train(self, pair_list, target_list):
        pair_matrices = self._predict_pair_matrices(pair_list)
        gt_matrices = self._generate_nxn_gt_matrices(pair_list, target_list) # NxN

        pair_proposals = []
        loss_pair_proposal = 0
        for seg_idx, (pair_matrix, gt_matrix) in enumerate(zip(pair_matrices, gt_matrices)):
            loss_pair_proposal += F.binary_cross_entropy(pair_matrix, gt_matrix.to(pair_matrix.device))

            pair_matrix_sorted, order = torch.sort(pair_matrix.view(-1), descending=True)
            sampled_pair_ind = order[:self.num_pair_proposals]
            pair_proposals.append(sampled_pair_ind)

        loss_ppn = {
            "loss_pair": loss_pair_proposal,
        }

        # feats = torch.stack([feats[i][pair_proposals[i]] for i in range(len(feats))])
        # targets = torch.stack([targets[i][pair_proposals[i]] for i in range(len(targets))])
        
        return pair_proposals, loss_ppn

    def _forward_test(self, pair_list):
        pair_matrices = self._predict_pair_matrices(pair_list)

        pair_proposals = []
        for seg_idx, pair_matrix in enumerate(pair_matrices):
            pair_matrix_sorted, order = torch.sort(pair_matrix.view(-1), descending=True)
            sampled_pair_ind = order[:self.num_pair_proposals]
            pair_proposals.append(sampled_pair_ind)

        # feats = torch.stack([feats[i][pair_proposals[i]] for i in range(len(feats))])

        return pair_proposals, {}

class PPNHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PPNHead, self).__init__()
        self.sub_emb = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )

        self.obj_emb = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, sub_logits, obj_logits):
        sub_emb = self.sub_emb(sub_logits)
        obj_emb = self.obj_emb(obj_logits)
        pair_matrix = torch.mm(sub_emb, obj_emb.t())
        pair_matrix = torch.sigmoid(pair_matrix)
        return pair_matrix


def make_ppn(cfg):
    return PPN(
        cfg
    )