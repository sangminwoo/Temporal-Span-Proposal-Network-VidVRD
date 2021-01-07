import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modeling.relpn import make_relpn


class BaseModel(nn.Module):
	def __init__(self, cfg):
		super(BaseModel, self).__init__()
		self.use_ppn = cfg.RELPN.USE_PPN
		self.use_dpn = cfg.RELPN.USE_DPN

		self.relpn = make_relpn(cfg)
		self.rel_of_interest_pool = RelOIPool()
		self.classifier = RelationPredictor(
			in_channels=cfg.PREDICT.FEATURE_DIM,
			out_channels=cfg.PREDICT.PREDICATE_NUM
		)

	def forward(self, pair_list, target_list=None):
		if self.training:
			return self._forward_train(pair_list, target_list)
		else:
			return self._forward_test(pair_list)

	def _forward_train(self, pair_list, target_list):
		loss_dict = {}
		pair_proposals = None
		duration_proposals = None
		
		feats = [plist.features for plist in pair_list]
		targets = [tlist.target for tlist in target_list]

		if self.use_ppn or self.use_dpn:
			pair_proposals, duration_proposals, relpn_losses = self.relpn(pair_list, target_list)
			loss_dict.update(relpn_losses)

		# relation of interest pooling
		reloi_feats = self.rel_of_interest_pool(feats, duration_proposals)

		# for multi-label classification
		loss_relation = 0
		for reloi_feat, target in zip(reloi_feats, targets):
			rel_logit = self.classifier(reloi_feat) # batch x pair_per_seg x 11070
			loss_relation += F.binary_cross_entropy(rel_logit, target)

		loss_rel = {
			"loss_rel": loss_relation
		}
		loss_dict.update(loss_rel)
		return loss_dict

	def _forward_test(self, pair_list):
		pair_proposals = None
		duration_proposals = None

		feats = [plist.features for plist in pair_list]

		pair_proposals, duration_proposals, _ = self.relpn(pair_list)
		reloi_feats = self.rel_of_interest_pool(feats, duration_proposals)
		
		rel_logits = []
		for reloi_feat in reloi_feats:
			rel_logits.append(self.classifier(reloi_feat)) # batch x pair_per_seg x 11070
		return pair_proposals, duration_proposals, rel_logits


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
		rel_logit = self.rel_predictor(reloi_feats)
		rel_logit = torch.sigmoid(rel_logit)
		return rel_logit