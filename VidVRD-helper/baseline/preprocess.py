import numpy as np
import h5py
import json
from itertools import product, cycle
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset

from .utils import normalize, to_multi_onehot
from baseline import *

class Preprocess:
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
					'''
					rel_inst = {'triplet': ('zebra', 'walk_front', 'zebra'),
								'subject_tid': 0,
								'object_tid': 1, 
								'duration': (150, 180)}
					rel_inst = {'triplet': ('person', 'sit_next_to', 'dog'),
								'subject_tid': 9,
								'object_tid': 0,
								'duration': (0, 30)}
					'''
					sub_name, pred_name, obj_name = rel_inst['triplet']
					segs = segment_video(*rel_inst['duration'])
					'''
					segment_video(0, 30)
					-> segs = [(0, 30)]

					segment_video(0, 45)
					-> segs = [(0, 30), (15, 45)]
					'''
					for fstart, fend in segs:
						rel_dict[fstart, # 0
								 fend, # 30
								 rel_inst['subject_tid'], # 1st in current segment
								 rel_inst['object_tid'], # 3rd in current segment
								 dataset.get_object_id(sub_name), # 2: car
								 dataset.get_object_id(obj_name) # 4: bus
								].append(dataset.get_predicate_id(pred_name)) # 68: faster
				'''
				defaultdict(list,
				            {(0, 30, 0, 1, 19, 10): [92, 86, 70, 4, 131, 118],
				             (0, 30, 1, 0, 10, 19): [51, 53, 46, 131],
				             (0, 30, 0, 2, 19, 27): [92, 88],
				             (15, 45, 0, 2, 19, 27): [92, 88], # 15-45
				             (0, 30, 2, 0, 27, 19): [2, 47, 46],
				             (0, 30, 0, 3, 19, 27): [85, 118],
				             (0, 30, 3, 0, 27, 19): [3, 46],
				             (0, 30, 1, 2, 10, 27): [55],
				             (15, 45, 1, 2, 10, 27): [55]}} # 15-45
				            )
				'''
				for (fstart, fend, sub_tid, obj_tid, sub, obj), pred in rel_dict.items():
					if self._extract_feature(vid, fstart, fend, dry_run=True):
						self.short_rel_insts[(vid, fstart, fend)].append((sub_tid, obj_tid, sub, pred, obj))
				'''
				self.short_rel_insts[(vid, fstart=0, fend=30)] = 
					[(0, 1, 19, [92, 86, 70, 4, 131, 118], 10),
					 (1, 0, 10, [51, 53, 46, 131], 19),
					 (0, 2, 19, [92, 88], 27),
					 (2, 0, 27, [2, 47, 46], 19),
					 (0, 3, 19, [85, 118], 27),
					 (3, 0, 27, [3, 46], 19),
					 (1, 2, 10, [55], 27)]
				self.short_rel_insts[(vid, fstart=15, fend=45)] = 
					[(0, 2, 19, [92, 88], 27),
					 (1, 2, 10, [55], 27)]
				'''
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

	def preprocess(self):
		self.logger.info(f'Total {len(self.index)} video segments')

		if self.phase == 'train':
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
		else:
			index = []
			pairs = []
			feats = []
			iou = []
			trackid = []
			for i in range(len(self.index)):
				self.logger.info(f'processing {i+1}th segment of test data')
				_index = self.index[i] # vid, fstart, fend
				_pairs, _feats, _iou, _trackid = self._extract_feature(*_index)
				test_inds = [ind for ind, (traj1, traj2) in enumerate(_pairs)
						if _trackid[traj1] < 0 and _trackid[traj2] < 0]
				_pairs = _pairs[test_inds]
				_feats = self._feature_preprocess(_feats[test_inds])
				
				index.append(_index)
				pairs.append(_pairs)
				feats.append(_feats.astype(np.float32))
				iou.append(_iou.tolist())
				trackid.append(_trackid)

			# index = np.concatenate(index)
			pairs = np.concatenate(pairs)
			feats = np.concatenate(feats)
			# iou = np.concatenate(iou)
			trackid = np.concatenate(trackid)
			return index, pairs, feats, iou, trackid

	def _data_sampling(self, vid, fstart, fend, iou_thres=0.5):
		pairs, feats, iou, trackid = self._extract_feature(vid, fstart, fend)
		'''
		pairs = [[ 0  1]
				 [ 0  2]
				 [ 0  3]
				 ...
				 [69 66]
				 [69 67]
				 [69 68]] (4830, 2)

		feats = [[6.02378805e-07 3.02198659e-05 9.47627029e-07 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]
				 [6.02378805e-07 3.02198659e-05 9.47627029e-07 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]
				 [6.02378805e-07 3.02198659e-05 9.47627029e-07 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]
				 ...
				 [3.13814926e-05 2.34911786e-05 2.29112455e-03 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]
				 [3.13814926e-05 2.34911786e-05 2.29112455e-03 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]
				 [3.13814926e-05 2.34911786e-05 2.29112455e-03 ... 0.00000000e+00
				  0.00000000e+00 0.00000000e+00]] (4830, 11070)

		iou =  [[0.99999976 0.10673315 0.         ... 0.         0.6488573  0.09914057]
				[0.10673315 0.9999998  0.         ... 0.         0.18019408 0.01014457]
				[0.         0.         1.0000001  ... 0.         0.         0.        ]
				...
				[0.         0.         0.         ... 0.9999999  0.         0.        ]
				[0.6488573  0.18019408 0.         ... 0.         1.         0.06941444]
				[0.09914057 0.01014457 0.         ... 0.         0.06941444 1.        ]] (70, 70)

		trackid =  [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
					-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
					-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  1] (70,)
		'''
		feats = feats.astype(np.float32)
		pair_to_find = dict([((traj1, traj2), find)
				for find, (traj1, traj2) in enumerate(pairs)])
		tid_to_ind = dict([(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0])

		# pos = np.empty((0, 2), dtype = np.int32)
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

		if len(pred_id) > 1:
			pred_id = np.stack(pred_id)
		else:
			pred_id = np.array([])
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
	processor = Preprocess(dataset, param, logger)
	if param['phase'] == 'train':
		feats, pred_id = processor.preprocess()
		return feats, pred_id
	elif param['phase'] == 'test':
		index, pairs, feats, iou, trackid = processor.preprocess()
		return index, pairs, feats, iou, trackid