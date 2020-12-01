import os
import torch
import random
import numpy as nnp
import json

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from utils import map_exec
from IPython import embed
import numpy as np


class RELATION(Dataset):
    def __init__(self, mode, file_path):
        super(RELATION, self).__init__()
        self.mode = mode
        self.is_train = (mode == "train")
        self.vid_names, self.proposals = get_proposals(file_path, self.is_train)
        self.pos_num = 64
        self.pos_neg_rate = 0.5
        self.features_dir = "../preprocess_data/tracking/videovrd_detect_tracking"
        self.i3d_dir = "../preprocess_data/tracking/videovrd_i3d"
        self.label_ids = get_label_ids()
        self.num_classes = 133 
        self.weight = get_weight(self.label_ids)

    def __getitem__(self, idx):
        if self.is_train:
            vid_name = self.vid_names[idx]
            proposals = self.proposals[vid_name]
            pos_sampled, neg_sampled = fg_bg_sampler(proposals, self.pos_num,
                    self.pos_neg_rate)
            sampled = pos_sampled + neg_sampled
            sampled_proposals = random.shuffle(sampled)
            sub_features, obj_features, motions, labels = get_features_dict(sampled, self.features_dir,
                    vid_name, self.is_train)
            motion_c, motion_s, motion_m = motions
            motion_c, motion_s, motion_m = np.array(motion_c), np.array(motion_s), np.array(motion_m)
            motion_c = (motion_c - motion_c.mean()) / motion_c.std()
            motion_s = (motion_s - motion_s.mean()) / motion_s.std()
            motion_m = (motion_m - motion_m.mean()) / motion_m.std()

            i3d_sub_features, i3d_obj_features, _, _ = get_features_dict(sampled,
                    self.i3d_dir, vid_name, self.is_train)
            target = labels_matrix(self.label_ids, labels, self.num_classes)
            weight = np.array(self.weight)
            weight = target * weight
            sub_features = torch.from_numpy(np.array(sub_features)).float()
            obj_features = torch.from_numpy(np.array(obj_features)).float()
            i3d_sub_features = torch.from_numpy(np.array(i3d_sub_features)).float()
            i3d_obj_features = torch.from_numpy(np.array(i3d_obj_features)).float()
            motion_c = torch.from_numpy(motion_c).float()
            motion_s = torch.from_numpy(motion_s).float()
            motion_m = torch.from_numpy(motion_m).float()

            sub_features = (sub_features - sub_features.mean()) / sub_features.std()
            obj_features = (obj_features - obj_features.mean()) / obj_features.std()
            i3d_sub_features = (i3d_sub_features - i3d_sub_features.mean()) / i3d_sub_features.std()
            i3d_obj_features = (i3d_obj_features - i3d_obj_features.mean()) / i3d_obj_features.std()
            motions = torch.cat((motion_c, motion_s, motion_m), 1)
            target = torch.from_numpy(target).float()
            weight = torch.from_numpy(np.array(weight)).float()

            return sub_features, obj_features, i3d_sub_features, i3d_obj_features, motions, target, weight
        else:
            vid_name = self.vid_names[idx]
            proposals = self.proposals[vid_name]
            proposals = proposals[:40]
            sub_features, obj_features, motions, labels, meta_info = get_features_dict(proposals,
                    self.features_dir, vid_name, self.is_train)
            i3d_sub_features, i3d_obj_features, _, _, _ = get_features_dict(proposals,
                    self.i3d_dir, vid_name, self.is_train)
            motion_c, motion_s, motion_m = motions
            motion_c, motion_s, motion_m = np.array(motion_c), np.array(motion_s), np.array(motion_m)
            motion_c = (motion_c - motion_c.mean()) / motion_c.std()
            motion_s = (motion_s - motion_s.mean()) / motion_s.std()
            motion_m = (motion_m - motion_m.mean()) / motion_m.std()
            target = labels_matrix(self.label_ids, labels, self.num_classes)
            sub_features = torch.from_numpy(np.array(sub_features)).float()
            obj_features = torch.from_numpy(np.array(obj_features)).float()
            i3d_sub_features = torch.from_numpy(np.array(i3d_sub_features)).float()
            i3d_obj_features = torch.from_numpy(np.array(i3d_obj_features)).float()

            motion_c = torch.from_numpy(motion_c).float()
            motion_s = torch.from_numpy(motion_s).float()
            motion_m = torch.from_numpy(motion_m).float()
            motions = torch.cat((motion_c, motion_s, motion_m), 1)
            target = torch.from_numpy(target).float()

            meta_info.append(vid_name)

            return sub_features, obj_features, i3d_sub_features, i3d_obj_features, motions, target, meta_info

    
    def __len__(self):
        return len(self.vid_names)

def get_training_set():
    return RELATION(mode="train", file_path="data/train_set_proposals.json")

def get_testing_set():
    return RELATION(mode="test",
            file_path="data/test_set_proposals.json")

def get_weight(label_ids):
    tmp_dict = dict()
    for line in open("data/train_set_state.txt"):
        line = line.strip().split()
        tmp_dict[line[0]] = int(line[1])
    weight = []
    for i in range(133):
        weight.append(10)
    for key in tmp_dict.keys():
        weight[label_ids[key]] = max(min(30, int(25000 / tmp_dict[key])), 10)
    weight[0] = 1
    return weight

    

def labels_matrix(label_ids, labels, num_classes):
    num_proposals = len(labels)
    ids = []
    for triplets in labels:
        tmp = []
        for ele in triplets:
            tmp.append(label_ids[ele]) 
        ids.append(tmp)
    res = np.zeros((num_proposals, num_classes))
    for i in range(num_proposals):
        res[i][ids[i]] = 1
    return res

def get_features_dict(sampled, root_dir, vid_name, training):
    feat_dict = dict()
    bbox_dict = dict()
    ans = np.load(os.path.join(root_dir, vid_name+".npy"))
    for ele in ans:
        frame_id, track_id = ele[:2]
        frame_id, track_id = int(frame_id), int(track_id)
        feat = ele[12:].tolist()
        feat_dict.setdefault(str(track_id), dict())
        feat_dict[str(track_id)][str(frame_id)] = feat
        bbox = ele[2:6].tolist()
        bbox_dict.setdefault(str(track_id), dict())
        bbox_dict[str(track_id)][str(frame_id)] = bbox

    sub_traj_fea = []
    obj_traj_fea = []
    sub_traj_ids = []
    obj_traj_ids = []
    sub_cls = []
    obj_cls = []
    labels = []
    scores = []
    durations = []
    motion_c = []
    motion_s = []
    motion_m = []

    for index, ele in enumerate(sampled):
        sub_traj = ele["sub_traj"]
        obj_traj = ele["obj_traj"]
        duration = ele["duration"]
        triplet = ele["triplet"]
        begin, end = duration
        tmp_sub_fea = []
        tmp_obj_fea = []
        tmp_sub_bbox = []
        tmp_obj_bbox = []
        for i in range(begin, end):
            if sub_traj in feat_dict.keys():
                if str(i) in feat_dict[sub_traj].keys():
                    tmp_sub_fea.append(feat_dict[sub_traj][str(i)])
            if obj_traj in feat_dict.keys():
                if str(i) in feat_dict[obj_traj].keys():
                    tmp_obj_fea.append(feat_dict[obj_traj][str(i)])
            if (sub_traj in bbox_dict.keys()) and (obj_traj in bbox_dict.keys()):
                if (str(i) in bbox_dict[sub_traj].keys()) and (str(i) in
                        bbox_dict[obj_traj].keys()):
                    tmp_sub_bbox.append(bbox_dict[sub_traj][str(i)])
                    tmp_obj_bbox.append(bbox_dict[obj_traj][str(i)])
        if tmp_sub_fea == [] or tmp_obj_fea == []:
            print(vid_name, begin, end, sub_traj, obj_traj, index)
            continue
        else:
            tmp_delta_c = []
            tmp_delta_s = []
            tmp_delta_m = []
            len_sub_bbox = len(tmp_sub_bbox)
            for i in range(len_sub_bbox):
                sub_x, sub_y, sub_w, sub_h = tmp_sub_bbox[i]
                obj_x, obj_y, obj_w, obj_h = tmp_obj_bbox[i]
                sub_center_x, sub_center_y = sub_x+sub_w / 2, sub_y+sub_h/2
                obj_center_x, obj_center_y = obj_x+obj_w/2, obj_y+obj_h/2
                tmp_delta_c.append([sub_center_x-obj_center_x,
                    sub_center_y-obj_center_y])
                tmp_delta_s.append([sub_w-obj_w, sub_h-obj_h])
            for i in range(len_sub_bbox-1):
                c2 = tmp_delta_c[i+1]
                c1 = tmp_delta_c[i]
                tmp_delta_m.append([c2[0]-c1[0], c2[1]-c1[1]])
            stride = len_sub_bbox // 20
            if stride < 1:
                print("error")
            delta_c, delta_s, delta_m = [], [], []
            for i in range(20):
                delta_c.append(tmp_delta_c[i*stride])
                delta_s.append(tmp_delta_s[i*stride])
                if i == 19:
                    continue
                delta_m.append(tmp_delta_m[i*stride])
            motion_c.append(delta_c)
            motion_s.append(delta_s)
            motion_m.append(delta_m)

            sub_traj_fea.append(np.mean(tmp_sub_fea, axis=0))
            obj_traj_fea.append(np.mean(tmp_obj_fea, axis=0))
            scores.append(ele["score"])
            sub_cls.append(triplet[0])
            obj_cls.append(triplet[2])
            durations.append(duration)
            if triplet[1] == []:
                labels.append(["bg"])
            else:
                labels.append(triplet[1])
            sub_traj_ids.append(sub_traj)
            obj_traj_ids.append(obj_traj)
    if training:
        return sub_traj_fea, obj_traj_fea, [motion_c, motion_s, motion_m], labels
    else:
        return sub_traj_fea, obj_traj_fea, [motion_c, motion_s, motion_m], labels, [sub_traj_ids, \
                obj_traj_ids, sub_cls, obj_cls,
                        scores, durations]

def fg_bg_sampler(proposals, pos_nums, pos_neg_rate):
    pos_count = 0
    pos_proposals = []
    neg_proposals = []
    for proposal in proposals:
        if proposal["triplet"][1] != ["bg"] and proposal["triplet"][1] != []:
            pos_count += 1
            pos_proposals.append(proposal)
        else:
            neg_proposals.append(proposal)
    pos_nums = min(pos_count, pos_nums)
    neg_nums = int(pos_nums * pos_neg_rate)
    neg_nums = min(len(neg_proposals), neg_nums)
    pos_sampled = random.choices(pos_proposals, k=pos_nums)
    neg_sampled = random.choices(neg_proposals, k=neg_nums)
    return pos_sampled, neg_sampled


def get_label_ids():
    label_ids = dict()
    for i, line in enumerate(open("data/relations.txt")):
        if i == 0:
            continue
        label, id = line.strip().split()
        label_ids[label] = i
    label_ids["bg"] = 0
    return label_ids

def get_proposals(file_path, training):
    ans = json.load(open(file_path))
    ans = ans["results"]
    vid_names = []
    for vid_name, content in ans.items():
        if not training:
            if len(content) == 0:
                continue
            vid_names.append(vid_name)
            continue
        tmp = 0
        for pair_proposals in content:
            if pair_proposals["triplet"][1] != ["bg"]:
                tmp += 1
        if tmp != 0:
            vid_names.append(vid_name)
    return vid_names, ans
