import os
import numpy as np
import h5py
import json
from torch.utils.data import Dataset

class VRDDataset(Dataset):
	def __init__(self, cfg, logger):
		self.rng = np.random.RandomState(cfg.ETC.RANDOM_SEED)
		self.phase = cfg.MODEL.PHASE
		self.logger = logger
		self.logger.info('loading preprocessed video segments for {}...'.format(self.phase))
		
		path = os.path.join('vidvrd-baseline-output', 'preprocessed_data')
		
		if self.phase == 'train':
			self.dataset = h5py.File(os.path.join(path, 'preprocessed_'+self.phase+'_dataset.hdf5'), 'r')
			self.feats = self.dataset['feats']
			self.pred_id = self.dataset['pred_id']

			assert len(self.feats) == len(self.pred_id)
			self.logger.info('total {} preprocessed relation instance proposals'.format(len(self.feats)))
		elif self.phase == 'test':
			self.dataset = h5py.File(os.path.join(path, 'preprocessed_'+self.phase+'_dataset.hdf5'), 'r')
			self.pairs = self.dataset['pairs']
			self.feats = self.dataset['feats']
			# self.trackid = self.dataset['trackid']

			# with open(os.path.join(path, 'preprocessed_'+self.phase+'_dataset.json'), 'r') as f:
			# 	self.dataset2 = json.load(f)
			# self.index = self.dataset2['index']
			# self.iou = self.dataset2['iou']

			assert len(self.pairs) == len(self.feats)
			self.logger.info('total {} preprocessed relation instance proposals'.format(len(self.feats)))
		else:
			raise ValueError('Unknown phase: {}'.format(self.phase))

	def __len__(self):
		return len(self.feats)

	def __getitem__(self, idx):
		if self.phase == 'train':
			return self.feats[idx], self.pred_id[idx]
		else:
			return self.pairs[idx], self.feats[idx]
			# return self.index[idx], self.pairs[idx], self.feats[idx], self.iou[idx], self.trackid[idx]