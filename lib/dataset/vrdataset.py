import os
import torch
import numpy as np
import h5py
import json
from itertools import product, cycle
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset

from lib.dataset.list_pair import PairList
from lib.dataset.list_target import TargetList
from lib.modeling.trajectory import Trajectory
from lib.utils.miscellaneous import normalize, to_multi_onehot
from lib.modeling import *

class VRDataset(Dataset):
	def __init__(self, cfg, dataset, phase):
		self.num_predicates = cfg.PREDICT.PREDICATE_NUM
		self.phase = phase
		self.logit_only = cfg.DATASET.LOGIT_ONLY
		self.use_gt_obj_trajs = cfg.DATASET.USE_GT_OBJ_TRAJS
		
		video_indices = dataset.get_index(split=self.phase)
		if self.phase == 'train':
			self.trajs = dict()
			self.gt_rel_insts = defaultdict(list)

			for vid in video_indices:
				for rel_inst in dataset.get_relation_insts(vid, no_traj=True):
					'''
					rel_inst = {
						'triplet': ('dog', 'bite', 'frisbee'),
						'subject_tid': 0,
						'object_tid': 1,
						'duration': (60, 90)
					}
					'''
					sub_name, pred_name, obj_name = rel_inst['triplet']
					sub_tid = rel_inst['subject_tid'], # 1st in current segment
					obj_tid = rel_inst['object_tid'], # 3rd in current segment
					sub_idx = dataset.get_object_id(sub_name), # 2: car
					obj_idx = dataset.get_object_id(obj_name) # 4: bus
					pred_idx = dataset.get_predicate_id(pred_name) # 68: faster
					
					segs = segment_video(*rel_inst['duration'])
					for fstart, fend in segs:
						if self._get_rel_feature(vid, fstart, fend, dont_return=True, verbose=False):
							self.gt_rel_insts[(vid, fstart, fend)].append(
								(sub_tid, obj_tid, sub_idx, obj_idx, pred_idx)
							)
			self.index = list(self.gt_rel_insts.keys())

		elif self.phase == 'test':
			self.index = []
			for vid in video_indices:
				anno = dataset.get_anno(vid)

				segs = segment_video(0, anno['frame_count'])
				for fstart, fend in segs:
					if self._get_rel_feature(vid, fstart, fend, dont_return=True, verbose=False):
						self.index.append(
							(vid, fstart, fend)
						)

		else:
			raise ValueError('Unknown phase: {}'.format(self.phase))


	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		if self.phase == 'train':
			index = self.index[idx]

			pairs, feats, iou, trackid = self._get_rel_feature(*index)
			feats, pairs, pred_labels = self._get_proposals_rel_feature(*index, pairs, feats, iou, trackid)
			proposal_idx = self._get_proposal_idx(pairs, trackid)
			feats, pairs, pred_labels = feats[proposal_idx], pairs[proposal_idx], pred_labels[proposal_idx]

			cls_logits = self._get_class_logit(*index)

			num_tracks = self._get_num_tracklet_proposals(trackid)
			
			feats = self._feature_preprocess(feats)

			pair_list = PairList(feats)
			pair_list.add_field('tracklet_pairs', pairs)
			pair_list.add_field('track_cls_logits', cls_logits)
			pair_list.add_field('num_tracklets', num_tracks)

			target_list = TargetList(pred_labels)

			return pair_list, target_list, index
		else:
			index = self.index[idx]

			pairs, feats, iou, trackid = self._get_rel_feature(*index)
			proposal_idx = self._get_proposal_idx(pairs, trackid)
			pairs = pairs[proposal_idx]

			feats = self._feature_preprocess(feats[proposal_idx])

			return index, pairs, feats, iou, trackid

	def _get_proposals_rel_feature(self, vid, fstart, fend, pairs, feats, iou, trackid, iou_thres=0.5):
		feats = torch.tensor(feats, dtype=torch.float32)
		pair_to_find = dict([
			((traj_idx1, traj_idx2), find) for find, (traj_idx1, traj_idx2) in enumerate(pairs) # (N+GT)(N+GT-1)
		])
		gt_tid_to_idx = dict([
			(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0 # GT
		])
		'''
		pair_to_find = {( 0,  1): 0,
				        ( 0,  2): 1,
				        ( 0,  3): 2,
				        ...
				        (17, 14): 304,
				        (17, 15): 305,
				        (17, 16): 306}

		gt_tid_to_idx = {0: 16,
					     1: 17,
					     2: 18}
		'''
		pos_idx = []
		pred_labels = {}
		for sub_tid, obj_tid, sub_idx, obj_idx, pred_idx in self.gt_rel_insts[(vid, fstart, fend)]:
			# 0, 1, 21(dog), 61(play), 30(frisbee) 
			if sub_tid in gt_tid_to_idx and obj_tid in gt_tid_to_idx: # if ground-truth tid & different
				iou1 = iou[:, gt_tid_to_idx[sub_tid]] # iou[:,15] : iou of proposals with ground-truth
				iou2 = iou[:, gt_tid_to_idx[obj_tid]] # iou[:,16]=[0.1, 0.2, 0.6, 0.3, 0.4, 0.8, ...]
				overlap_tids1 = torch.where(iou1 >= iou_thres)[0] # pick proposal idx that largely overlaps with ground-truth
				overlap_tids2 = torch.where(iou2 >= iou_thres)[0] # [2, 5, 13, ...], [4, 7, 10, ...]

				for traj_idx1, traj_idx2 in product(overlap_tids1, overlap_tids2): # over all possible combination of idxs
					if traj_idx1 != traj_idx2 and \
						trackid[traj_idx1] < 0 and trackid[traj_idx2] < 0 : # if two idxs are different & not ground-truth
						pos_idx.append(
							pair_to_find[(traj_idx1, traj_idx2)]
						) # [3, 15, 50, 100]
						pred_labels[
							pair_to_find[(traj_idx1, traj_idx2)]
						] = to_multi_onehot(pred_idx, self.num_predicates) # [0 0 ... 1 ... 0 0]

		# If traj-traj has no relation -> fill pred_labels with zeros
		neg_idx = [
			idx for idx in range(len(pairs)) if idx not in pos_idx
		]
		pred_labels.update(
			dict([
				(i, torch.zeros(self.num_predicates)) for i in neg_idx
			])
		)
		pred_labels = list(pred_labels.values())
		pred_labels = torch.stack(pred_labels)

		return feats, pairs, pred_labels

	def _get_proposal_idx(self, pairs, trackid):
		proposal_idx = [
			ind for ind, (traj_idx1, traj_idx2) in enumerate(pairs) \
			if trackid[traj_idx1] < 0 and trackid[traj_idx2] < 0
		]
		return proposal_idx

	def _get_num_tracklet_proposals(self, trackid):
		num_tracks = sum(trackid < 0)
		return num_tracks

	def _get_class_logit(self, vid, fstart, fend):
		trajs = self._get_object_trajectory_proposal(
			vid, fstart, fend,
			self.logit_only,
			gt=self.use_gt_obj_trajs,
			verbose=False
		)

		if self.logit_only:
			class_logits = torch.tensor(trajs, dtype=torch.float32)
		else:
			class_logits = [torch.tensor(traj.classeme, dtype=torch.float32) for traj in trajs]
			class_logits = torch.stack(class_logits)
		return class_logits

	def _get_object_trajectory_proposal(self, vid, fstart, fend, logit_only=False, gt=False, verbose=False):
	    """
	    Set gt=True for providing groundtruth bounding box trajectories and
	    predicting classme feature only
	    """
	    vsig = get_segment_signature(vid, fstart, fend)
	    name = 'traj_cls_gt' if gt else 'traj_cls'
	    path = get_feature_path(name, vid)
	    path = os.path.join(path, '{}-{}.json'.format(vsig, name))
	    if os.path.exists(path):
	        if verbose:
	            print('loading object {} proposal for video segment {}'.format(name, vsig))
	        with open(path, 'r') as fin:
	            trajs = json.load(fin)
	        if logit_only:
	        	trajs = [traj['classeme'] for traj in trajs]
	        else:
		        trajs = [Trajectory(**traj) for traj in trajs]
	    else:
	        if verbose:
	            print('no object {} proposal for video segment {}'.format(name, vsig))
	        trajs = []
	    return trajs

	def _get_rel_feature(self, vid, fstart, fend, dont_return=False, verbose=False):
		vsig = get_segment_signature(vid, fstart, fend)
		# vid=ILSVRC2015_train_00005003, fstart=0000, fend=0030
		path = get_feature_path('relation', vid)
		# ./vidvrd-baseline-output/features/relation/ILSVRC2015_train_00005003
		path = os.path.join(path, '{}-{}.h5'.format(vsig, 'relation'))
		# ./vidvrd-baseline-output/features/relation/ILSVRC2015_train_00005003/ILSVRC2015_train_00005003-0000-0030-relation.h5
		if os.path.exists(path):
			if dont_return:
				return None, None, None, None
			else:
				if verbose:
					print('loading relation feature for video segment {}...'.format(vsig))
				with h5py.File(path, 'r') as fin:
					# N object trajectory proposals, whose trackids are all -1
					# and M groundtruth object trajectories, whose trackids are provided by dataset
					trackid = fin['trackid'][:]
					# all possible pairs among N+M object trajectories
					pairs = fin['pairs'][:]
					# relation feature for each pair (in same order)
					feats = fin['feats'][:]
					# vIoU (traj_iou) for each pair (in same order)
					iou = fin['iou'][:]
				return pairs, feats, iou, trackid
		else:
			if verbose:
				print('no relation feature for video segment  {}'.format(vsig))
		return None

	def _feature_preprocess(self, feats):
		# subject classeme + object classeme (70)
		# feats[:, 0: 70]

		# subject TrajectoryShape + HoG + HoF + MBH motion feature (8000)
		# (since this feature is Bag-of-Word type, we l1-normalize it so that
		# each element represents the fraction instead of count)

		feats[:, 70: 1070] = normalize(feats[:, 70: 1070], axis=-1, order=1)
		feats[:, 1070: 2070] = normalize(feats[:, 1070: 2070], axis=-1, order=1)
		feats[:, 2070: 3070] = normalize(feats[:, 2070: 3070], axis=-1, order=1)
		feats[:, 3070: 4070] = normalize(feats[:, 3070: 4070], axis=-1, order=1)

		# object TrajectoryShape + HoG + HoF + MBH motion feature
		feats[:, 4070: 5070] = normalize(feats[:, 4070: 5070], axis=-1, order=1)
		feats[:, 5070: 6070] = normalize(feats[:, 5070: 6070], axis=-1, order=1)
		feats[:, 6070: 7070] = normalize(feats[:, 6070: 7070], axis=-1, order=1)
		feats[:, 7070: 8070] = normalize(feats[:, 7070: 8070], axis=-1, order=1)

		# relative posititon + size + motion feature (3000)
		# feats[:, 8070: 9070]
		# feats[:, 9070: 10070]
		# feats[:, 10070: 11070]

		return feats