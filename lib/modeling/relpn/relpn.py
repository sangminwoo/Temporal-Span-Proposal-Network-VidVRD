import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppn import make_ppn
from .dpn import make_dpn


class RelPN(nn.Module):
	'''
	Relation Proposal Network (RelPN) =
		Pair Proposal Network (PPN) + Relationship Duration Proposal Network (DPN)
	'''
	def __init__(self, cfg):
		super(RelPN, self).__init__()
		self.use_ppn = cfg.RELPN.USE_PPN
		self.use_dpn = cfg.RELPN.USE_DPN
		self.pair_proposal_network = make_ppn(cfg)
		self.duration_proposal_network = make_dpn(cfg)
		
	def forward(self, pair_list, target_list):
		if self.training:
			return self._forward_train(pair_list, target_list)
		else:
			return self._forward_test(pair_list)

	def _forward_train(self, pair_list, target_list):
		relpn_losses = {}
		pair_proposals = None
		duration_proposals = None
		
		# relationship pair proposal: "What object pairs are related? Are object A and B related?"
		if self.use_ppn:
			pair_proposals, loss_ppn = self.pair_proposal_network(pair_list, target_list)
			relpn_losses.update(loss_ppn)

		# relationship duration proposal: "How long the relation lasts? From t=0 to t=30?"
		if self.use_dpn:
			duration_proposals, loss_dpn = self.duration_proposal_network(pair_list, target_list, pair_proposals)
			relpn_losses.update(loss_dpn)

		return pair_proposals, duration_proposals, relpn_losses

	def _forward_test(self, pair_list, target_list):
		pair_proposals, _ = self.pair_proposal_network(pair_list, target_list)

		duration_proposals, _ = self.duration_proposal_network(pair_list, target_list, pair_proposals)

		return pair_proposals, duration_proposals, {}


def make_relpn(cfg):
	return RelPN(
		cfg
	)