import torch
import torch.nn as nn


class RelationPredictor(nn.Module):
    def __init__(self, cfg):
        super(RelationPredictor, self).__init__()
        self.linear = nn.Linear(cfg.PREDICT.FEATURE_DIM, cfg.PREDICT.PREDICATE_NUM)

    def forward(self, feats):
        relation = self.linear(feats) # 64x11070 -> 64x132
        relation = torch.sigmoid(relation) # for multi-label classification
        return relation