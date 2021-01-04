import numpy as np
import h5py
from itertools import product
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset

from lib.utils.miscellaneous import normalize
from lib.modeling import *

class VRDDataset(Dataset):
	def __init__(self, dataset, param, logger):
		self.rng = np.random.RandomState(param['rng_seed'])
		self.batch_size = param['batch_size']
		self.max_sampling_in_batch = param['max_sampling_in_batch']
		assert self.max_sampling_in_batch <= self.batch_size
		self.phase = param['phase']
		self._train_triplet_id = OrderedDict()
		self.logger = logger
		self.logger.info('preparing video segments for {}...'.format(self.phase))
		if self.phase == 'train':
			# initialize training triplet id
			self._train_triplet_id.clear()
			triplets = dataset.get_triplets(split='train')
			# triplets = {('car', 'faster', 'bus'), ('car', 'taller', 'antelope'), ...} len:2961
			for i, triplet in enumerate(triplets):
				sub_name, pred_name, obj_name = triplet
				sub_id = dataset.get_object_id(sub_name)
				pred_id = dataset.get_predicate_id(pred_name)
				obj_id = dataset.get_object_id(obj_name)
				self._train_triplet_id[(sub_id, pred_id, obj_id)] = i
			# _train_triplet_id = OrderedDict([((2, 68, 4), 0), ((34, 131, 13), 1), ...]) len:2961

			self.short_rel_insts = defaultdict(list)
			video_indices = dataset.get_index(split='train')
			for vid in video_indices:
				for rel_inst in dataset.get_relation_insts(vid, no_traj=True):
					segs = segment_video(*rel_inst['duration'])
					for fstart, fend in segs:
						# if multiple objects detected and the relation features extracted
						# cache the corresponding groudtruth labels
						if self._extract_feature(vid, fstart, fend, dry_run=True):
							sub_name, pred_name, obj_name = rel_inst['triplet']
							self.short_rel_insts[(vid, fstart, fend)].append((
								rel_inst['subject_tid'], # 0th
								rel_inst['object_tid'], # 1st
								dataset.get_object_id(sub_name), # 2: car
								dataset.get_predicate_id(pred_name), # 68: faster
								dataset.get_object_id(obj_name) # 4: bus
							))
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

	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		if self.phase == 'train':
			feats = []
			triplet_idx = []
			pred_id = []
			remaining_size = self.batch_size
			while remaining_size > 0:
				vid, fstart, fend = self.index[idx] ### TODO FIX
				sample_num = np.minimum(remaining_size, self.max_sampling_in_batch)
				_feats, _triplet_idx, _pred_id = self._data_sampling(vid, fstart, fend, sample_num)
				remaining_size -= _feats.shape[0]
				_feats = self._feature_preprocess(_feats)
				feats.append(_feats.astype(np.float32))
				triplet_idx.append(_triplet_idx.astype(np.float32))
				pred_id.append(_pred_id.astype(np.float32))
			feats = np.concatenate(feats)
			triplet_idx = np.concatenate(triplet_idx)
			pred_id = np.concatenate(pred_id)
			return feats, triplet_idx, pred_id
		else:
			index = self.index[idx] ### TODO FIX
			pairs, feats, iou, trackid = self._extract_feature(*index)
			test_inds = [ind for ind, (traj1, traj2) in enumerate(pairs)
					if trackid[traj1] < 0 and trackid[traj2] < 0]
			pairs = pairs[test_inds]
			feats = self._feature_preprocess(feats[test_inds])
			feats = feats.astype(np.float32)
			return index, pairs, feats, iou, trackid

	def _data_sampling(self, vid, fstart, fend, sample_num, iou_thres=0.5):
		pairs, feats, iou, trackid = self._extract_feature(vid, fstart, fend)
		feats = feats.astype(np.float32)
		pair_to_find = dict([((traj1, traj2), find)
				for find, (traj1, traj2) in enumerate(pairs)])
		tid_to_ind = dict([(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0])

		pos = np.empty((0, 3), dtype = np.int32)
		for tid1, tid2, s, p, o in self.short_rel_insts[(vid, fstart, fend)]:
			if tid1 in tid_to_ind and tid2 in tid_to_ind:
				iou1 = iou[:, tid_to_ind[tid1]]
				iou2 = iou[:, tid_to_ind[tid2]]
				pos_inds1 = np.where(iou1 >= iou_thres)[0]
				pos_inds2 = np.where(iou2 >= iou_thres)[0]
				tmp = [(pair_to_find[(traj1, traj2)], self._train_triplet_id[(s, p, o)], p)
						for traj1, traj2 in product(pos_inds1, pos_inds2) if traj1 != traj2]
				if len(tmp) > 0:
					pos = np.concatenate((pos, tmp))

		num_pos_in_this = np.minimum(pos.shape[0], sample_num)
		if pos.shape[0] > 0:
			pos = pos[np.random.choice(pos.shape[0], num_pos_in_this, replace=False)]

		return feats[pos[:, 0]], pos[:, 1], pos[:, 2]

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