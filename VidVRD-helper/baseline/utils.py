import logging
import os
import sys
from datetime import datetime
import numpy as np

class AverageMeter:
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
		self.avg = self.sum / self.count

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING
	# don't log results for the non-master process
	if distributed_rank > 0:
		return logger

	stream_handler = logging.StreamHandler(stream=sys.stdout)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	file_handler = logging.FileHandler(os.path.join(save_dir, filename))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger

def get_timestamp():
	now = datetime.now()
	timestamp = datetime.timestamp(now)
	st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
	return st

def calculate_eta(one_batch_time, cur_epoch, max_epoch, cur_iter, max_iter):
	eta_in_iter = one_batch_time * (max_iter-cur_iter-1)
	eta_in_epoch = (one_batch_time * max_iter) * (max_epoch-cur_epoch-1) 
	eta_seconds = eta_in_iter + eta_in_epoch
	return eta_seconds

def normalize(x, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

def to_onehot(x, num_classes):
    """ one-hot encodes a tensor """
    return np.eye(num_classes, dtype='float32')[x]