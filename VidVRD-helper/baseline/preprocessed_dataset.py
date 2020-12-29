import os
import numpy as np
import h5py
from torch.utils.data import Dataset

class VRDDataset(Dataset):
	def __init__(self, param, logger):
		self.rng = np.random.RandomState(param['rng_seed'])
		self.phase = param['phase']
		self.logger = logger
		self.logger.info('loading preprocessed video segments for {}...'.format(self.phase))
		
		path = os.path.join('vidvrd-baseline-output', 'preprocessed_data', 'preprocessed_'+self.phase+'_dataset.hdf5')
		if self.phase == 'train':
			self.dataset = h5py.File(path, 'r')
			self.feats = self.dataset['feats']
			self.triplet_idx = self.dataset['triplet_idx']
			self.pred_id = self.dataset['pred_id']

			assert len(self.feats) == len(self.triplet_idx) == len(self.pred_id)
			self.logger.info('total {} preprocessed relation instance proposals'.format(len(self.feats)))
		else:
			pass

	def __len__(self):
		return len(self.feats)

	def __getitem__(self, idx):
		if self.phase == 'train':
			return self.feats[idx], self.triplet_idx[idx], self.pred_id[idx]
		else:
			pass
			# return index, pairs, feats, iou, trackid