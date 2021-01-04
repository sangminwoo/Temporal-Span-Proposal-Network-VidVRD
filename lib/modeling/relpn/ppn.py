import torch
import torch.nn as nn
import torch.nn.functional as F

from sampler import BalancedPositiveNegativePairSampler

class PPN(nn.Module):
	'''
	Pair Proposal Network
	'''
	def __init__(self, cfg):
		super(PPN, self).__init__()
		self.num_pairs = cfg.RELPN.PPN.NUM_PAIR_PROPOSALS
        self.ppn_head = PPNHead(
            in_channels=cfg.RELPN.PPN.IN_CHANNELS,
            out_channels=cfg.RELPN.PPN.OUT_CHANNELS
        )

		fg_bg_sampler = BalancedPositiveNegativePairSampler(
	        batch_size_per_image=cfg.RELPN.PPN.BATCH_SIZE_PER_IMAGE,
	        positive_fraction=cfg.RELPN.PPN.POSITIVE_FRACTION
        )

    def _get_ground_truth(self, gt_rels):
        return gt_pairs

    def forward(self, obj_feats, gt_rels):
        pair_score = self.ppn_head(obj_feats)

        if self.training:
            return self._forward_train(pair_score, gt_rels)
        else:
            return self._forward_test(pair_score)

    def _forward_train(self, pair_score, gt_rels):
        gt_pairs = _get_ground_truth(gt_rels) # NxN

        pair_score_sorted, order = torch.sort(pair_score.view(-1), descending=True)
        # img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
        # proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
        # proposal_pairs[img_idx] = proposal_pairs_per_image
        pair_proposals = pair_score_sorted[:self.num_pairs]

        loss_pair_proposal = F.binary_cross_entropy(pair_score, gt_pairs)
        losses = {
            "loss_pair_proposal": loss_pair_proposal,
        }
        return pair_proposals, losses

    def _forward_test(self, pair_score):
        pair_score_sorted, order = torch.sort(pair_score.view(-1), descending=True)
        # img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
        # proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
        # proposal_pairs[img_idx] = proposal_pairs_per_image
        pair_proposals = pair_score_sorted[:self.num_pairs]

        return pair_proposals, {}


class PPNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PPNHead, self).__init__()
        self.sub_emb = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

        self.obj_emb = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

        for l in [self.sub_emb, self.obj_emb]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, obj_feats):
        sub_emb = self.sub_emb(obj_feats)
        obj_emb = self.obj_emb(obj_feats)
        pair_score = torch.mm(sub_emb, obj_emb)
        pair_score = torch.sigmoid(pair_score)
        return pair_score


def make_ppn(cfg):
    return PPN(
        cfg,
        in_channels = cfg.RELPN.DPN.IN_CHANNELS,
        out_channels = cfg.RELPN.DPN.NUM_ANCHORS_PER_LOCATION
    )