import torch
import torch.nn as nn
import torch.nn.functional as F

class RelNMS(nn.Module):
	def __init__(self, param):
		super(RelNMS, self).__init__()
		self.fg_iou_threshold = 0.7
		self.bg_iou_threshold = 0.3
		self.nms_threshold = 0.5
		self.top_k_proposals = param['num_duration_proposals']
		self.anchor = 

	def forward(self, relationness, duration_proposals):
		relationness
