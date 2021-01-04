import torch
import torch.nn as nn
import torch.nn.functional as F

class RelOIPool:
	def __call__(self, rel_feats, duration_proposals):
		return rel_feats[duration_proposals]

class RelPN(nn.Module):
	'''
	Relation Proposal Network (RelPN) =
		Pair Proposal Network (PPN) + Relationship Duration Proposal Network (DPN)
	'''
	def __init__(self, param):
		super(PPN, self).__init__()
		self.pair_proposal_network = PPN(param)
		self.duration_proposal_network = DPN(param)
        self.rel_of_interest_pool = RelOIPool(param)
		self.rel_predictor = nn.Linear()

		self.rel_feature_extractor = RelFeatureExtractor(
			in_channels=param['obj_dim']*2,
			out_channels=param['obj_dim']
		)
		self.relpn_head = RelPNHead(
			in_channels=param['obj_dim'],
			out_channels=param['predicate_num']
		)

	def _get_ground_truth(self, gt_rels):
		return gt_relations

	def forward(self, obj_feats, gt_rels):
		if self.training:
			return self._forward_train(obj_feats, gt_rels)
		else:
			return self._forward_test(obj_feats)

    def _forward_train(self, obj_feats, gt_rels):
    	'''
		obj_feats: NxCxTxHxW
		rel_feats: NxCxT
    	'''
    	gt_relations = self._get_ground_truth(gt_rels)
    	
    	pair_proposals, loss_ppn = self.pair_proposal_network(obj_feats, gt_rels)
		
		rel_feats = self.rel_feature_extractor(obj_feats, pair_proposals)
        relationness, duration_proposals, loss_dpn = self.duration_proposal_network(rel_feats, gt_rels)
        
        rel_feats = self.rel_of_interest_pool(rel_feats, duration_proposals)
        relations = self.relpn_head(rel_feats)
        loss_relation = F.cross_entropy(relations, gt_relations)

    	losses = {
        	'loss_pair':loss_ppn['loss_pair'],
    		'loss_relationness':loss_dpn['loss_relationness'],
    		'loss_duration_reg':loss_dpn['loss_duration_reg'],
    		'loss_rel':loss_relation,
        }
    	return pair_proposals, relations, duration_proposals, losses

    def _forward_test(self, obj_feats):
    	pair_proposals, _ = self.pair_proposal_network(obj_feats, gt_rels)

    	rel_feats = self.rel_feature_extractor(obj_feats, pair_proposals)
    	relationness, duration_proposals, _ = self.duration_proposal_network(rel_feats, gt_rels)
        
        rel_feats = self.rel_of_interest_pool(rel_feats, duration_proposals)
        relations = self.relpn_head(rel_feats)
    	return pair_proposals, relations, duration_proposals, {}


class RelFeatureExtractor(nn.Module):
	def __init__(self, in_channels, out_channels):
        super(RelFeatureExtractor, self).__init__()
        self.rel_feature_extractor = nn.Sequential(
			nn.Linear(in_channels, out_channels),
			nn.ReLU(True),
			nn.Linear(out_channels, out_channels)
		)

        for l in [self.rel_feature_extractor]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, obj_feats, pair_proposals):
		head_feats = obj_feats[pair_proposals[:, 0]] # 10x1024
		tail_feats = obj_feats[pair_proposals[:, 1]] # 10x1024
		pair_feats = torch.cat([head_feats, tail_feats], dim=1) # 10x2048
		rel_feats = self.rel_feature_extractor(pair_feats) # 10x2048 -> 10x1024
		return rel_feats

class RelPNHead(nn.Module):
	def __init__(self, in_channels, out_channels):
        super(RelPNHead, self).__init__()
        self.rel_predictor = nn.Linear(in_channels, out_channels)

        for l in [self.rel_predictor]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def _predict_relations(self, rel_feats):
		relations = self.rel_predictor(rel_feats)
		return relations