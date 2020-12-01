import os
import numpy as np
import json
from IPython import embed
from tqdm import tqdm

pred_res = json.load(open("logs/result.json"))
det_res = json.load(open("../preprocess_data/tracking/videovrd_tracklets.json"))
vid_names = []
for line in open("data/vid_names_test.txt"):
    vid_names.append(line.strip())

cate_names = dict()
for line in open("../preprocess_data/tracking/cateid_name.txt"):
    line = line.strip().split()
    cate_names[line[0]] = line[1]

rel_names = dict()
for i, line in enumerate(open("data/relations.txt")):
    if i == 0:
        continue
    line = line.strip().split()
    rel_names[i] = line[0]

result = dict()
result["results"] = dict()

for vid_name in tqdm(vid_names):
    if vid_name in pred_res.keys():
        preds = pred_res[vid_name]
        det = det_res[vid_name]
        result["results"][vid_name] = []
        for pred in preds:
            sub_traj_id = pred["sub_traj"][0]
            obj_traj_id = pred["obj_traj"][0]
            duration = pred["duration"]
            rel_scores = pred["rel_scores"]
            scores = pred["score"]
            sub = pred["triplet"][0][0]
            obj = pred["triplet"][2][0]
            relations = pred["triplet"][1]
            begin, end = duration
            sub_trajs = []
            obj_trajs = []
            for traj_indx in range(begin, end):
                sub_bbox = det[sub_traj_id][str(traj_indx)]["track_bbox"]
                x, y, w, h = sub_bbox
                tmp_sub_bbox = [x, y, x+w, y+h]
                sub_trajs.append(tmp_sub_bbox)
                obj_bbox = det[obj_traj_id][str(traj_indx)]["track_bbox"]
                x, y, w, h = obj_bbox
                tmp_obj_bbox = [x, y, x+w, y+h]
                obj_trajs.append(tmp_obj_bbox)

            for indx, rel in enumerate(relations):
                if rel == 0 or rel == 133:
                    continue
                if obj == '0' or sub == '0':
                    print(indx)
                    continue
                tmp = dict()
                rel_score = rel_scores[indx]
                tmp["score"] = scores * rel_score
                tmp["duration"] = duration
                tmp["sub_traj"] = sub_trajs
                tmp["obj_traj"] = obj_trajs
                new_triplet = [cate_names[sub], rel_names[rel],
                        cate_names[obj]]
                tmp["triplet"] = new_triplet
                result["results"][vid_name].append(tmp)
    else:
        result["results"][vid_name] = []

for key, value in result["results"].items():
    scores = []
    for ele in value:
        scores.append(ele["score"]) 
    index = np.argsort(-np.array(scores))
    tmp = np.array(value)
    result["results"][key] = list(tmp[index])[:200]
json.dump(result,
        open("results/result_final.json", "w"))
