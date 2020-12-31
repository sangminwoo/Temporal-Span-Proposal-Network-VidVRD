import os
import numpy as np
import h5py
import json
from itertools import product, cycle
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset

from .utils import normalize, to_multi_onehot
from baseline import *

class VRDataset(Dataset):
	def __init__(self, dataset, param, logger):
		self.num_predicates = param['predicate_num']
		self.rng = np.random.RandomState(param['rng_seed'])
		self.phase = param['phase']
		self.logger = logger
		self.logger.info('preparing video segments for {}...'.format(self.phase))
		
		if self.phase == 'train':
			self.short_rel_insts = defaultdict(list)
			video_indices = dataset.get_index(split='train')
			for vid in video_indices:
				rel_dict = defaultdict(list)
				for rel_inst in dataset.get_relation_insts(vid, no_traj=True):
					sub_name, pred_name, obj_name = rel_inst['triplet']
					segs = segment_video(*rel_inst['duration'])

					for fstart, fend in segs:
						rel_dict[fstart, # 0
								 fend, # 30
								 rel_inst['subject_tid'], # 1st in current segment
								 rel_inst['object_tid'], # 3rd in current segment
								 dataset.get_object_id(sub_name), # 2: car
								 dataset.get_object_id(obj_name) # 4: bus
								].append(dataset.get_predicate_id(pred_name)) # 68: faster

				for (fstart, fend, sub_tid, obj_tid, sub, obj), pred in rel_dict.items():
					if self._extract_feature(vid, fstart, fend, dry_run=True):
						self.short_rel_insts[(vid, fstart, fend)].append((sub_tid, obj_tid, sub, pred, obj))

			self.index = list(self.short_rel_insts.keys())
		elif self.phase == 'test':
			self.index = []
			video_indices = dataset.get_index(split='test')
			for vid in video_indices:
				anno = dataset.get_anno(vid)
				segs = segment_video(0, anno['frame_count'])
				# enumerate all the possible segments
				for fstart, fend in segs:
					# if multiple objects detected and the relation features extracted
					if self._extract_feature(vid, fstart, fend, dry_run=True):
						self.index.append((vid, fstart, fend))
		else:
			raise ValueError('Unknown phase: {}'.format(self.phase))

		if self.phase == 'train':
			data_path = os.path.join('vidvrd-baseline-output', 'preprocessed_data', 'preprocessed_train_dataset.hdf5')
			if os.path.exists(data_path):
				dataset = h5py.File(data_path, 'r')
				self.feats, self.pred_id = dataset['feats'], dataset['pred_id']
				self.logger.info('Preprocessed data loaded...'.format(len(self.feats)))
			else:
				self.feats, self.pred_id = self.preprocess()

			assert len(self.feats) == len(self.pred_id)
			self.logger.info('Total {} relation instance proposals for train'.format(len(self.feats)))
		else:
			self.logger.info('Total {} videos segments for test'.format(len(self.index)))

	def __len__(self):
		if self.phase == 'train':
			return len(self.feats)
		else:
			return len(self.index)

	def __getitem__(self, idx):
		if self.phase == 'train':
			return self.feats[idx], self.pred_id[idx]
		else:
			index = self.index[idx]
			pairs, feats, iou, trackid = self._extract_feature(*index)
			test_inds = [ind for ind, (traj1, traj2) in enumerate(pairs) if trackid[traj1] < 0 and trackid[traj2] < 0]
			pairs = pairs[test_inds]
			feats = self._feature_preprocess(feats[test_inds])
			feats = feats.astype(np.float32)
			return index, pairs, feats, iou, trackid
			
	def preprocess(self):
		self.logger.info(f'Total {len(self.index)} video segments')
		feats = []
		pred_id = []
		for i in range(len(self.index)):
			self.logger.info(f'processing {i+1}th segment of train data')
			vid, fstart, fend = self.index[i]
			_feats, _pred_id = self._data_sampling(vid, fstart, fend)

			_feats = self._feature_preprocess(_feats)
			feats.append(_feats.astype(np.float32))
			pred_id.append(_pred_id.astype(np.float32))

		feats = np.concatenate(feats)
		pred_id = np.concatenate(pred_id)
		return feats, pred_id

	def _data_sampling(self, vid, fstart, fend, iou_thres=0.5):
		pairs, feats, iou, trackid = self._extract_feature(vid, fstart, fend)
		feats = feats.astype(np.float32)
		pair_to_find = dict([((traj1, traj2), find)
				for find, (traj1, traj2) in enumerate(pairs)])
		tid_to_ind = dict([(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0])

		feat_idx = []
		pred_id = []
		for sub_tid, obj_tid, sub, pred, obj in self.short_rel_insts[(vid, fstart, fend)]:
			if sub_tid in tid_to_ind and obj_tid in tid_to_ind:
				iou1 = iou[:, tid_to_ind[sub_tid]]
				iou2 = iou[:, tid_to_ind[obj_tid]]
				pos_inds1 = np.where(iou1 >= iou_thres)[0]
				pos_inds2 = np.where(iou2 >= iou_thres)[0]

				for traj1, traj2 in product(pos_inds1, pos_inds2):
					if traj1 != traj2:
						feat_idx.append(pair_to_find[(traj1, traj2)])
						pred_id.append(to_multi_onehot(pred, self.num_predicates))

		if len(pred_id) > 0:
			pred_id = np.stack(pred_id)
		else:
			pred_id = np.array([]).reshape(0, self.num_predicates)
		return feats[feat_idx], pred_id  # _feats, _pred_id

	def _extract_feature(self, vid, fstart, fend, dry_run=False, verbose=False):
		vsig = get_segment_signature(vid, fstart, fend)
		# vid=ILSVRC2015_train_00005003, fstart=0000, fend=0030
		path = get_feature_path('relation', vid)
		# ./vidvrd-baseline-output/features/relation/ILSVRC2015_train_00005003
		path = os.path.join(path, '{}-{}.h5'.format(vsig, 'relation'))
		# ./vidvrd-baseline-output/features/relation/ILSVRC2015_train_00005003/ILSVRC2015_train_00005003-0000-0030-relation.h5
		if os.path.exists(path):
			if dry_run:
				return None, None, None, None
			else:
				if verbose:
					self.logger.info('loading relation feature for video segment {}...'.format(vsig))
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
				self.logger.info('no relation feature for video segment  {}'.format(vsig))
		return None

	def _feature_preprocess(self, feat):
		# subject classeme + object classeme (70)
		# feat[:, 0: 70]

		# subject TrajectoryShape + HoG + HoF + MBH motion feature (8000)
		# (since this feature is Bag-of-Word type, we l1-normalize it so that
		# each element represents the fraction instead of count)

		feat[:, 70: 1070] = normalize(feat[:, 70: 1070], axis=-1, order=1)
		feat[:, 1070: 2070] = normalize(feat[:, 1070: 2070], axis=-1, order=1)
		feat[:, 2070: 3070] = normalize(feat[:, 2070: 3070], axis=-1, order=1)
		feat[:, 3070: 4070] = normalize(feat[:, 3070: 4070], axis=-1, order=1)

		# object TrajectoryShape + HoG + HoF + MBH motion feature
		feat[:, 4070: 5070] = normalize(feat[:, 4070: 5070], axis=-1, order=1)
		feat[:, 5070: 6070] = normalize(feat[:, 5070: 6070], axis=-1, order=1)
		feat[:, 6070: 7070] = normalize(feat[:, 6070: 7070], axis=-1, order=1)
		feat[:, 7070: 8070] = normalize(feat[:, 7070: 8070], axis=-1, order=1)

		# relative posititon + size + motion feature (3000)
		# feat[:, 8070: 9070]
		# feat[:, 9070: 10070]
		# feat[:, 10070: 11070]
		return feat

def preprocess_data(dataset, param, logger):
	processor = VRDataset(dataset, param, logger)
	return processor.preprocess()