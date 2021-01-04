import torch
import torch.nn as nn


class RelationPredictor(nn.Module):
    def __init__(self, param):
        super(RelationPredictor, self).__init__()
        self.classifier = nn.Linear(param['feature_dim'], param['predicate_num'])

    def forward(self, feats):
        relation = self.classifier(feats) # 64x11070 -> 64x132
        relation = torch.sigmoid(relation) # for multi-label classification
        return relation