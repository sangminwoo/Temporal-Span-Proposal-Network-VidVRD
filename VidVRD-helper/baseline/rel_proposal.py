import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationProposal(nn.Module):
    def __init__(self, param):
        super(RelationProposal, self).__init__()
        self.relavance = nn.Linear(param['object_num']*2, 1)

    def forward(self, sub, obj):
        x = torch.cat([sub, obj], dim=1)
        rel_score = self.relavance(x) # 64x70 -> 64x1
        rel_score = F.sigmoid(rel_score)
        return rel_score

def gt_relavance_to_target(gt_relevance):
	pass

def rel_proposal_loss(param, sub, obj, gt_relevance):
	rel_proposal = RelationProposal(param)
	rel_score = rel_proposal(sub, obj) # 64x1

	sorted_ind = rel_score.sort(dim=0)[1].flatten()
	topk_ind = sorted_ind[:param['num_rel_proposal']] # relation proposals

	target = gt_relevance_to_target(gt_relevance)
	loss = F.binary_cross_entropy(rel_score, target)
	return topk_ind, loss