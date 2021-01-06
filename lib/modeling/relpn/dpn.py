import torch
import torch.nn as nn
import torch.nn.functional as F

from .rel_nms import RelNMS
from .sampler import BalancedPositiveNegativePairSampler
from .anchor_generator import make_anchor_generator

class DPN(nn.Module):
    '''
    Duration Proposal Network: "How long the relation lasts? From t=0 to t=30?"
    '''
    def __init__(self, cfg, in_channels, num_windows):
        super(DPN, self).__init__()

        head = DPNHead(
            in_channels=in_channels,
            num_windows=num_windows
        )

        self.dpn_head = head
        self.rel_nms = RelNMS(cfg)

    def _preidct_durations(self, pair_list):
        pair_feats = pair_lists.features
        duration_proposals = self.dpn_head(feats)

        return duration_proposals

    def _get_ground_truth(self, target_list):
        gt_duration = [tlist.get_field('duration') for tlist in target_list]
        return gt_duration

    def forward(self, pair_list, target_list=None):
        if self.training:
            return self._forward_train(pair_list, target_list)
        else:
            return self._forward_test(pair_list)

    def _forward_train(self, pair_list, target_list):
        duration_proposals = self._preidct_durations(pair_list)
        gt_duration = self._get_ground_truth(target_list)

        loss_duration_proposal = F.binary_cross_entropy_with_logits(duration_proposals, gt_duration)
        loss_dpn = {
            "loss_duration": loss_duration_proposal,
        }
        return duration_proposals, loss_dpn

    def _forward_test(self, pair_list):
        duration_proposals = self._preidct_durations(pair_list)
        return duration_proposals, {}


class DPNHead(nn.Module):
    def __init__(self, in_channels, num_windows):
        super(DPNHead, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.duration_pred = nn.Conv1d(
            in_channels, num_windows * 2, kernel_size=1, stride=1
        )

        for l in [self.conv, self.duration_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, feats):
        t = F.relu(self.conv(feats))
        duration_reg = self.duration_pred(t)

        return duration_reg


def make_dpn(cfg):
    return DPN(
        cfg,
        in_channels = cfg.RELPN.DPN.IN_CHANNELS,
        num_windows = cfg.RELPN.DPN.NUM_ANCHORS_PER_LOCATION
    )