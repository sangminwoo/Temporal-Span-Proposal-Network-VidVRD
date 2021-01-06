import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modeling.relpn import make_relpn


class BaseModel(nn.Module):
	def __init__(self, cfg):
		super(BaseModel, self).__init__()
		self.relpn = make_relpn(cfg)
		self.rel_of_interest_pool = RelOIPool()
		self.classifier = RelationPredictor(
			in_channels=cfg.PREDICT.FEATURE_DIM,
			out_channels=cfg.PREDICT.PREDICATE_NUM
		)

		# pair_feats = pair_lists.features
		# track_pairs = pair_lists.get_field('tracklet_pairs')
		# track_cls_logits = pair_lists.get_field('track_cls_logits')
		# num_tracks = pair_lists.get_field('num_tracklets')
		# targets = target_list.target

	def forward(self, pair_list, target_list):
		if self.training:
			return self._forward_train(pair_list, target_list)
		else:
			return self._forward_test(pair_list)

	def _forward_train(self, pair_list, target_list):
		loss_dict = {}
		pair_proposals, duration_proposals, relpn_losses = self.relpn(pair_list, target_list)
		loss_dict.update(relpn_losses)

		feats = [plist.features for plist in pair_list]
		targets = [tlist.target for tlist in target_list]
		# feats = torch.stack([feats[i][pair_proposals[i]] for i in range(len(feats))])
		# targets = torch.stack([targets[i][pair_proposals[i]] for i in range(len(targets))])

		# relation of interest pooling
		reloi_feats = self.rel_of_interest_pool(feats, duration_proposals)

		# for multi-label classification
		loss_relation = 0
		for reloi_feat, target in zip(reloi_feats, targets):
			relation = self.classifier(reloi_feat) # batch x pair_per_seg x 11070
			loss_relation += F.binary_cross_entropy_with_logits(relation, target)

		loss_rel = {
			"loss_rel": loss_relation
		}
		loss_dict.update(loss_rel)
		return loss_dict

	def _forward_test(self, pair_list):
		pair_proposals, duration_proposals, _ = self.relpn(pair_list, target_list)
		reloi_feats = self.rel_of_interest_pool(feats, duration_proposals)
		relations = self.classifier(reloi_feats)
		return pair_proposals, duration_proposals, relations


class RelOIPool:
	def __call__(self, feats, duration_proposals):
		if duration_proposals is None:
			return feats

		return feats[duration_proposals]


class RelationPredictor(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(RelationPredictor, self).__init__()
		self.rel_predictor = nn.Linear(in_channels, out_channels)

		for l in [self.rel_predictor]:
			torch.nn.init.normal_(l.weight, std=0.01)
			torch.nn.init.constant_(l.bias, 0)

	def forward(self, reloi_feats):
		relations = self.rel_predictor(reloi_feats)
		return relations