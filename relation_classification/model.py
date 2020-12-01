import torch
from torch import nn
import os
from IPython import embed

class RelationPredictor(nn.Module):
    def __init__(self):
        super(RelationPredictor, self).__init__()
        num_classes = 133
        self.fc1 = nn.Linear(2048, 512)
        self.fc2= nn.Linear(2048, 512)
        self.fc1_i3d = nn.Linear(832, 512)
        self.fc2_i3d = nn.Linear(832, 512)
        self.fc_motion = nn.Linear(118, 512)
        self.fc = nn.Linear(2560, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.8)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x1, x2, x1_i3d, x2_i3d, motion):
        ori_shape = x1.shape
        i3d_shape = x1_i3d.shape
        motion_shape = motion.shape
        x1 = x1.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x2 = x2.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x1_i3d = x1_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x2_i3d = x2_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x_m = motion.view(motion_shape[0]*motion_shape[1], -1)

        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x1_i3d = self.drop(x1_i3d)
        x2_i3d = self.drop(x2_i3d)
        x_m = self.drop(x_m)

        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)
        x1_i3d = self.fc1_i3d(x1_i3d)
        x1_i3d = self.relu(x1_i3d)
        x2_i3d = self.fc2_i3d(x2_i3d)
        x2_i3d = self.relu(x2_i3d)
        x_m = self.fc_motion(x_m)
        x_m = self.relu(x_m)
        x = torch.cat((x1, x1_i3d, x_m, x2, x2_i3d), 1)
        x = self.fc(x)
        x = x.view(ori_shape[0], ori_shape[1], -1)
        return x
