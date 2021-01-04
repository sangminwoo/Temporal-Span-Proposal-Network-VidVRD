import torch
import torch.nn as nn
import torch.nn.functional as F

from rel_nms import RelNMS
from sampler import BalancedPositiveNegativePairSampler
from anchor_generator import make_anchor_generator

class DPN(nn.Module):
    '''
    Duration Proposal Network
    '''
    def __init__(self, param):
        super(DPN, self).__init__()
        head = DPNHead(
            in_channels=param['dpn_in_channels'],
            num_windows=param['num_windows']
        )

        self.dpn_head = head
        self.rel_nms = RelNMS(param)

    def _get_ground_truth(self, gt_rels):
        return gt_relness, gt_duration

    def forward(self, rel_feats, gt_rels):
        '''
        rel_feats: NxCxT
        '''
        relationness, duration_proposals = self.dpn_head(rel_feats)

        if self.training:
            return self._forward_train(relationness, duration_proposals, gt_rels)
        else:
            return self._forward_test(relationness, duration_proposals)

    def _forward_train(self, relationness, duration_proposals, gt_rels):
        gt_relness, gt_duration = _get_ground_truth(gt_rels)

        relationness, duration_proposals = self.rel_nms(relationness, duration_proposals)

        loss_relationness += F.cross_entropy(relationness, gt_relness)
        loss_duration_proposal += F.smooth_l1_loss(duration_proposals, gt_duration)

        loss_relationness, loss_duration_proposal = self.loss_evaluator(
            relationness, duration_proposals, gt_rels
        )
        losses = {
            "loss_relationness": loss_relationness,
            "loss_duration_proposal": loss_duration_proposal,
        }
        return relationness, duration_proposals, losses

    def _forward_test(self, relationness, duration_proposals):
        relationness, duration_proposals = self.rel_nms(relationness, duration_proposals)
        return relationness, duration_proposals, {}


class DPNHead(nn.Module):
    def __init__(self, in_channels, num_windows):
        super(DPNHead, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.relness_pred = nn.Conv1d(
            in_channels, num_windows, kernel_size=1, stride=1
        )
        self.duration_pred = nn.Conv1d(
            in_channels, num_windows * 2, kernel_size=1, stride=1
        )

        for l in [self.conv, self.relness_pred, self.duration_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, rel_feats):
        relness = []
        duration_reg = []

        t = F.relu(self.conv(rel_feats))
        relness.append(self.relness_pred(t))
        duration_reg.append(self.duration_pred(t))

        return relness, duration_reg