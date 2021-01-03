import torch
import torch.nn as nn
import torch.nn.functional as F

from rel_nms import RelNMS
from sampler import BalancedPositiveNegativePairSampler

class RelOIPool:
	def __call__(self, rel_feats, duration_proposals):
		return rel_feats[duration_proposals]
		
class RelPN(nn.Module):
	'''
	Relation Proposal Network = Pair Proposal Network + Duration Proposal Network
	'''
	def __init__(self, param):
		super(PPN, self).__init__()
		self.pair_proposal_network = PPN(param)
		self.duration_proposal_network = DPN(param)
		self.rel_feature_extractor = nn.Sequential(
			nn.Linear(param['obj_dim']*2, param['obj_dim']),
			nn.ReLU(True),
			nn.Linear(param['obj_dim'], param['obj_dim'])
		)
        self.rel_of_interest_pool = RelOIPool(param)
		self.rel_predictor = nn.Linear()

    def _extract_relation_feature(self, obj_feats, pair_proposals):
		head_feats = obj_feats[pair_proposals[:, 0]] # 10x1024
		tail_feats = obj_feats[pair_proposals[:, 1]] # 10x1024
		pair_feats = torch.cat([head_feats, tail_feats], dim=1) # 10x2048
		rel_feats = self.rel_feature_extractor(pair_feats) # 10x2048 -> 10x1024
		return rel_feats

	def _predict_relations(self, duration, relness_score, rel_feats):
		relations = self.rel_predictor(rel_feats, duration)
		return relations

	def _get_ground_truth(self, gt_rels):
		return gt_relations

	def forward(self, obj_feats, gt_rels):
		gt_relations = self._get_ground_truth(gt_rels)

		pair_proposals, pair_proposal_loss = self.pair_proposal_network(obj_feats, gt_rels)
		rel_feats = self._extract_relation_feature(obj_feats, pair_proposals)
        duration_proposals, relness_score, relness_loss, duration_proposal_loss = self.duration_proposal_network(rel_feats, gt_rels)
        
        rel_feats = self.rel_of_interest_pool(rel_feats, duration_proposals)

        relations = self._predict_relations(rel_feats)
        relation_prediction_loss = F.cross_entropy(relations, gt_relations)

        loss = {'pair_loss':pair_proposal_loss,
        		'relness_loss':relness_loss,
        		'duration_loss':duration_proposal_loss,
        		'rel_loss':relation_prediction_loss,}

        return pair_proposals, relations, loss

class PPN(nn.Module):
	'''
	Pair Proposal Network
	'''
	def __init__(self, param):
		super(PPN, self).__init__()
		self.num_pairs = param['num_pairs']

		fg_bg_sampler = BalancedPositiveNegativePairSampler(
	        batch_size_per_image=param['batch_size_per_image'],
	        positive_fraction=param['positive_fraction']
        )

		self.sub_emb = nn.Sequential(
			nn.Linear(param['object_num']*2, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim)
		)

		self.sub_emb = nn.Sequential(
			nn.Linear(param['object_num']*2, dim),
			nn.ReLU(True),
			nn.Linear(dim, dim)
		)

	def _pair_scoring(self, obj_feats):
        sub_emb = self.sub_emb(obj_feats)
        obj_emb = self.obj_emb(obj_feats)
        pair_score = torch.mm(sub_emb, obj_emb)
        pair_score = torch.sigmoid(pair_score)
        return pair_score

    def _get_ground_truth(self, gt_rels):
        return gt_pairs

    def forward(self, obj_feats, gt_rels):
    	gt_pairs = _get_ground_truth(gt_rels) # NxN

    	pair_score = self._pair_scoring(obj_feats)
    	pair_score_sorted, order = torch.sort(pair_score.view(-1), descending=True)
    	img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
        
        proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
        proposal_pairs[img_idx] = proposal_pairs_per_image

        pair_proposal_loss += F.binary_cross_entropy(pair_score, gt_pairs)
    	
    	pair_proposals = pair_score[:self.num_pairs]
    	return pair_proposals, pair_proposal_loss

class DPN(nn.Module):
	'''
	Duration Proposal Network
	'''
    def __init__(self, param):
        super(DPN, self).__init__()
        self.relationness_predictor = nn.Linear(param['object_num']*2, 2)
        self.duration_predictor = nn.Linear(param['object_num']*2, 4)
		self.rel_nms = RelNMS(param)

    def _get_ground_truth(self, gt_rels):
    	return gt_relness, gt_duration

    def forward(self, rel_feats, gt_rels):
    	'''
		obj_feats: NxCxTxHxW
    	'''
    	gt_relness, gt_duration = _get_ground_truth(gt_rels)

    	relness_score = self.relationness_predictor(rel_feats) # 64x70 -> 64x2
    	duration_proposals = self.duration_predictor(rel_feats)

    	relness_loss += F.cross_entropy(relness_score, gt_relness)
    	duration_proposal_loss += F.smooth_l1_loss(duration_proposals, gt_duration)

    	duration_proposals, relness_score = self.rel_nms(duration_proposals, relness_score)

        return duration_proposals, relness_score, relness_loss, duration_proposal_loss