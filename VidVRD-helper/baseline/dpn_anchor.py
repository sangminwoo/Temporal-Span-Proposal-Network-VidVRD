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
        anchor_generator = make_anchor_generator(param)
        head = DPNHead(
            in_channels=param['dpn_in_channels'],
            num_anchors=anchor_generator.num_anchors_per_location()[0]
        )

        window_selector_train = make_dpn_postprocessor(param, is_train=True)
        window_selector_test = make_dpn_postprocessor(param, is_train=False)
        loss_evaluator = make_dpn_loss_evaluator(param)
        
        self.dpn_only = param['dpn_only']
        self.anchor_generator = anchor_generator
        self.dpn_head = head
        self.window_selector_train = window_selector_train
        self.window_selector_test = window_selector_test
        self.loss_evaluator = loss_evaluator
        self.rel_nms = RelNMS(param)

    def _get_ground_truth(self, gt_rels):
        return gt_relness, gt_duration

    def forward(self, rel_feats, gt_rels):
        '''
        rel_feats: NxCxT
        '''
        relationness, duration_regression = self.dpn_head(rel_feats)
        anchors = self.anchor_generator(rel_feats)

        if self.training:
            return self._forward_train(anchors, relationness, duration_regression, gt_rels)
        else:
            return self._forward_test(anchors, relationness, duration_regression)

        duration_proposals, relationness = self.rel_nms(duration_proposals, relationness)

    def _forward_train(self, anchors, relationness, duration_regression, gt_rels):
        if self.dpn_only:
            windows = anchors
        else:
            with torch.no_grad():
                windows = self.window_selector_train(
                    anchors, relationness, duration_regression, gt_rels
                )
        loss_relationness, loss_duration_proposal = self.loss_evaluator(
            anchors, relationness, duration_regression, gt_rels
        )
        losses = {
            "loss_relationness": loss_relationness,
            "loss_duration_proposal": loss_duration_proposal,
        }
        return windows, losses

        gt_relness, gt_duration = _get_ground_truth(gt_rels)

        loss_relationness += F.cross_entropy(relationness, gt_relness)
        loss_duration_proposal += F.smooth_l1_loss(duration_proposals, gt_duration)

    def _forward_test(self, anchors, relationness, duration_regression):
        windows = self.window_selector_test(anchors, relationness, duration_regression)
        if self.dpn_only:
            inds = [
                window.get_field("relationness").sort(descending=True)[1] for window in windows
            ]
            windows = [window[ind] for window, ind in zip(windows, inds)]
        return windows, {}


class DPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(DPNHead, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.relness_pred = nn.Conv1d(
            in_channels, num_anchors, kernel_size=1, stride=1
        )
        self.duration_pred = nn.Conv1d(
            in_channels, num_anchors * 2, kernel_size=1, stride=1
        )

        for l in [self.conv, self.relness_pred, self.duration_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, rel_feats):
        relness = []
        duration_reg = []

        for feature in rel_feats:
            t = F.relu(self.conv(feature))
            relness.append(self.relness_pred(t))
            duration_reg.append(self.duration_pred(t))

        return relness, duration_reg