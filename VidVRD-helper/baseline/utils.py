import logging
import os
import sys
from datetime import datetime

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
		self.sum += val #* n
		self.count += n
		self.avg = self.sum / self.count

def setup_logger(name, save_dir, filename="log.txt"):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	stream_handler = logging.StreamHandler(stream=sys.stdout)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	file_handler = logging.FileHandler(os.path.join(save_dir, filename))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger

def get_timestamp():
	now = datetime.now()
	timestamp = datetime.timestamp(now)
	st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
	return st