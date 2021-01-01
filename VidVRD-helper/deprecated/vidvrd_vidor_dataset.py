import os
import json
import argparse
from pprint import pprint
from collections import defaultdict

class VidvrdVidorDataset:
	"""
    Dataset base class with Json annotations without the "version" field.
    It helps maintaining the mapping between category id and category name,
    and parsing the annotations to get instances of object, action and visual relation.
    """
	def __init__(self, data_dir, dataset, split, anno):
		# vidvrd anno_path: '/home/t2_u1/data/vidvrd/train/ILSVRC2015_train_00005003.json'
		# vidor anno_path: '/home/t2_u1/data/vidor/annotation/training/0000/2401075277.json'
		anno_path = os.path.join(data_dir, dataset, split, anno)
		if dataset == 'vidvrd':
			# VidVRD (obj:35, rel:132)
			self.idx_to_obj = {0: 'airplane', 1: 'antelope', 2: 'ball', 3: 'bear', 4: 'bicycle', 5: 'bird', 6: 'bus', 7: 'car', 8: 'cattle', 9: 'dog', 10: 'domestic_cat', 11: 'elephant', 12: 'fox', 13: 'frisbee', 14: 'giant_panda', 15: 'hamster', 16: 'horse', 17: 'lion', 18: 'lizard', 19: 'monkey', 20: 'motorcycle', 21: 'person', 22: 'rabbit', 23: 'red_panda', 24: 'sheep', 25: 'skateboard', 26: 'snake', 27: 'sofa', 28: 'squirrel', 29: 'tiger', 30: 'train', 31: 'turtle', 32: 'watercraft', 33: 'whale', 34: 'zebra'}
			self.obj_to_idx = {'airplane': 0, 'antelope': 1, 'ball': 2, 'bear': 3, 'bicycle': 4, 'bird': 5, 'bus': 6, 'car': 7, 'cattle': 8, 'dog': 9, 'domestic_cat': 10, 'elephant': 11, 'fox': 12, 'frisbee': 13, 'giant_panda': 14, 'hamster': 15, 'horse': 16, 'lion': 17, 'lizard': 18, 'monkey': 19, 'motorcycle': 20, 'person': 21, 'rabbit': 22, 'red_panda': 23, 'sheep': 24, 'skateboard': 25, 'snake': 26, 'sofa': 27, 'squirrel': 28, 'tiger': 29, 'train': 30, 'turtle': 31, 'watercraft': 32, 'whale': 33, 'zebra': 34}
			self.idx_to_rel = {0: 'above', 1: 'away', 2: 'behind', 3: 'beneath', 4: 'bite', 5: 'chase', 6: 'creep_above', 7: 'creep_away', 8: 'creep_behind', 9: 'creep_beneath', 10: 'creep_front', 11: 'creep_left', 12: 'creep_next_to', 13: 'creep_past', 14: 'creep_right', 15: 'creep_toward', 16: 'drive', 17: 'fall_off', 18: 'faster', 19: 'feed', 20: 'fight', 21: 'fly_above', 22: 'fly_away', 23: 'fly_behind', 24: 'fly_front', 25: 'fly_left', 26: 'fly_next_to', 27: 'fly_past', 28: 'fly_right', 29: 'fly_toward', 30: 'fly_with', 31: 'follow', 32: 'front', 33: 'hold', 34: 'jump_above', 35: 'jump_away', 36: 'jump_behind', 37: 'jump_beneath', 38: 'jump_front', 39: 'jump_left', 40: 'jump_next_to', 41: 'jump_past', 42: 'jump_right', 43: 'jump_toward', 44: 'jump_with', 45: 'kick', 46: 'larger', 47: 'left', 48: 'lie_above', 49: 'lie_behind', 50: 'lie_beneath', 51: 'lie_front', 52: 'lie_inside', 53: 'lie_left', 54: 'lie_next_to', 55: 'lie_right', 56: 'lie_with', 57: 'move_above', 58: 'move_away', 59: 'move_behind', 60: 'move_beneath', 61: 'move_front', 62: 'move_left', 63: 'move_next_to', 64: 'move_past', 65: 'move_right', 66: 'move_toward', 67: 'move_with', 68: 'next_to', 69: 'past', 70: 'play', 71: 'pull', 72: 'ride', 73: 'right', 74: 'run_above', 75: 'run_away', 76: 'run_behind', 77: 'run_beneath', 78: 'run_front', 79: 'run_left', 80: 'run_next_to', 81: 'run_past', 82: 'run_right', 83: 'run_toward', 84: 'run_with', 85: 'sit_above', 86: 'sit_behind', 87: 'sit_beneath', 88: 'sit_front', 89: 'sit_inside', 90: 'sit_left', 91: 'sit_next_to', 92: 'sit_right', 93: 'stand_above', 94: 'stand_behind', 95: 'stand_beneath', 96: 'stand_front', 97: 'stand_inside', 98: 'stand_left', 99: 'stand_next_to', 100: 'stand_right', 101: 'stand_with', 102: 'stop_above', 103: 'stop_behind', 104: 'stop_beneath', 105: 'stop_front', 106: 'stop_left', 107: 'stop_next_to', 108: 'stop_right', 109: 'stop_with', 110: 'swim_behind', 111: 'swim_beneath', 112: 'swim_front', 113: 'swim_left', 114: 'swim_next_to', 115: 'swim_right', 116: 'swim_with', 117: 'taller', 118: 'touch', 119: 'toward', 120: 'walk_above', 121: 'walk_away', 122: 'walk_behind', 123: 'walk_beneath', 124: 'walk_front', 125: 'walk_left', 126: 'walk_next_to', 127: 'walk_past', 128: 'walk_right', 129: 'walk_toward', 130: 'walk_with', 131: 'watch'}
			self.rel_to_idx = {'above': 0, 'away': 1, 'behind': 2, 'beneath': 3, 'bite': 4, 'chase': 5, 'creep_above': 6, 'creep_away': 7, 'creep_behind': 8, 'creep_beneath': 9, 'creep_front': 10, 'creep_left': 11, 'creep_next_to': 12, 'creep_past': 13, 'creep_right': 14, 'creep_toward': 15, 'drive': 16, 'fall_off': 17, 'faster': 18, 'feed': 19, 'fight': 20, 'fly_above': 21, 'fly_away': 22, 'fly_behind': 23, 'fly_front': 24, 'fly_left': 25, 'fly_next_to': 26, 'fly_past': 27, 'fly_right': 28, 'fly_toward': 29, 'fly_with': 30, 'follow': 31, 'front': 32, 'hold': 33, 'jump_above': 34, 'jump_away': 35, 'jump_behind': 36, 'jump_beneath': 37, 'jump_front': 38, 'jump_left': 39, 'jump_next_to': 40, 'jump_past': 41, 'jump_right': 42, 'jump_toward': 43, 'jump_with': 44, 'kick': 45, 'larger': 46, 'left': 47, 'lie_above': 48, 'lie_behind': 49, 'lie_beneath': 50, 'lie_front': 51, 'lie_inside': 52, 'lie_left': 53, 'lie_next_to': 54, 'lie_right': 55, 'lie_with': 56, 'move_above': 57, 'move_away': 58, 'move_behind': 59, 'move_beneath': 60, 'move_front': 61, 'move_left': 62, 'move_next_to': 63, 'move_past': 64, 'move_right': 65, 'move_toward': 66, 'move_with': 67, 'next_to': 68, 'past': 69, 'play': 70, 'pull': 71, 'ride': 72, 'right': 73, 'run_above': 74, 'run_away': 75, 'run_behind': 76, 'run_beneath': 77, 'run_front': 78, 'run_left': 79, 'run_next_to': 80, 'run_past': 81, 'run_right': 82, 'run_toward': 83, 'run_with': 84, 'sit_above': 85, 'sit_behind': 86, 'sit_beneath': 87, 'sit_front': 88, 'sit_inside': 89, 'sit_left': 90, 'sit_next_to': 91, 'sit_right': 92, 'stand_above': 93, 'stand_behind': 94, 'stand_beneath': 95, 'stand_front': 96, 'stand_inside': 97, 'stand_left': 98, 'stand_next_to': 99, 'stand_right': 100, 'stand_with': 101, 'stop_above': 102, 'stop_behind': 103, 'stop_beneath': 104, 'stop_front': 105, 'stop_left': 106, 'stop_next_to': 107, 'stop_right': 108, 'stop_with': 109, 'swim_behind': 110, 'swim_beneath': 111, 'swim_front': 112, 'swim_left': 113, 'swim_next_to': 114, 'swim_right': 115, 'swim_with': 116, 'taller': 117, 'touch': 118, 'toward': 119, 'walk_above': 120, 'walk_away': 121, 'walk_behind': 122, 'walk_beneath': 123, 'walk_front': 124, 'walk_left': 125, 'walk_next_to': 126, 'walk_past': 127, 'walk_right': 128, 'walk_toward': 129, 'walk_with': 130, 'watch': 131}
		elif dataset == 'vidor':
			# VidOR (obj:80, rel:50)
			self.idx_to_obj = {0: 'adult', 1: 'aircraft', 2: 'antelope', 3: 'baby', 4: 'baby_seat', 5: 'baby_walker', 6: 'backpack', 7: 'ball/sports_ball', 8: 'bat', 9: 'bear', 10: 'bench', 11: 'bicycle', 12: 'bird', 13: 'bottle', 14: 'bread', 15: 'bus/truck', 16: 'cake', 17: 'camel', 18: 'camera', 19: 'car', 20: 'cat', 21: 'cattle/cow', 22: 'cellphone', 23: 'chair', 24: 'chicken', 25: 'child', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dish', 30: 'dog', 31: 'duck', 32: 'electric_fan', 33: 'elephant', 34: 'faucet', 35: 'fish', 36: 'frisbee', 37: 'fruits', 38: 'guitar', 39: 'hamster/rat', 40: 'handbag', 41: 'horse', 42: 'kangaroo', 43: 'laptop', 44: 'leopard', 45: 'lion', 46: 'microwave', 47: 'motorcycle', 48: 'oven', 49: 'panda', 50: 'penguin', 51: 'piano', 52: 'pig', 53: 'rabbit', 54: 'racket', 55: 'refrigerator', 56: 'scooter', 57: 'screen/monitor', 58: 'sheep/goat', 59: 'sink', 60: 'skateboard', 61: 'ski', 62: 'snake', 63: 'snowboard', 64: 'sofa', 65: 'squirrel', 66: 'stingray', 67: 'stool', 68: 'stop_sign', 69: 'suitcase', 70: 'surfboard', 71: 'table', 72: 'tiger', 73: 'toilet', 74: 'toy', 75: 'traffic_light', 76: 'train', 77: 'turtle', 78: 'vegetables', 79: 'watercraft'}
			self.obj_to_idx = {'adult': 0, 'aircraft': 1, 'antelope': 2, 'baby': 3, 'baby_seat': 4, 'baby_walker': 5, 'backpack': 6, 'ball/sports_ball': 7, 'bat': 8, 'bear': 9, 'bench': 10, 'bicycle': 11, 'bird': 12, 'bottle': 13, 'bread': 14, 'bus/truck': 15, 'cake': 16, 'camel': 17, 'camera': 18, 'car': 19, 'cat': 20, 'cattle/cow': 21, 'cellphone': 22, 'chair': 23, 'chicken': 24, 'child': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dish': 29, 'dog': 30, 'duck': 31, 'electric_fan': 32, 'elephant': 33, 'faucet': 34, 'fish': 35, 'frisbee': 36, 'fruits': 37, 'guitar': 38, 'hamster/rat': 39, 'handbag': 40, 'horse': 41, 'kangaroo': 42, 'laptop': 43, 'leopard': 44, 'lion': 45, 'microwave': 46, 'motorcycle': 47, 'oven': 48, 'panda': 49, 'penguin': 50, 'piano': 51, 'pig': 52, 'rabbit': 53, 'racket': 54, 'refrigerator': 55, 'scooter': 56, 'screen/monitor': 57, 'sheep/goat': 58, 'sink': 59, 'skateboard': 60, 'ski': 61, 'snake': 62, 'snowboard': 63, 'sofa': 64, 'squirrel': 65, 'stingray': 66, 'stool': 67, 'stop_sign': 68, 'suitcase': 69, 'surfboard': 70, 'table': 71, 'tiger': 72, 'toilet': 73, 'toy': 74, 'traffic_light': 75, 'train': 76, 'turtle': 77, 'vegetables': 78, 'watercraft': 79}
			self.idx_to_rel = {0: 'above', 1: 'away', 2: 'behind', 3: 'beneath', 4: 'bite', 5: 'caress', 6: 'carry', 7: 'chase', 8: 'clean', 9: 'close', 10: 'cut', 11: 'drive', 12: 'feed', 13: 'get_off', 14: 'get_on', 15: 'grab', 16: 'hit', 17: 'hold', 18: 'hold_hand_of', 19: 'hug', 20: 'in_front_of', 21: 'inside', 22: 'kick', 23: 'kiss', 24: 'knock', 25: 'lean_on', 26: 'lick', 27: 'lift', 28: 'next_to', 29: 'open', 30: 'pat', 31: 'play(instrument)', 32: 'point_to', 33: 'press', 34: 'pull', 35: 'push', 36: 'release', 37: 'ride', 38: 'shake_hand_with', 39: 'shout_at', 40: 'smell', 41: 'speak_to', 42: 'squeeze', 43: 'throw', 44: 'touch', 45: 'towards', 46: 'use', 47: 'watch', 48: 'wave', 49: 'wave_hand_to'}
			self.rel_to_idx = {'above': 0, 'away': 1, 'behind': 2, 'beneath': 3, 'bite': 4, 'caress': 5, 'carry': 6, 'chase': 7, 'clean': 8, 'close': 9, 'cut': 10, 'drive': 11, 'feed': 12, 'get_off': 13, 'get_on': 14, 'grab': 15, 'hit': 16, 'hold': 17, 'hold_hand_of': 18, 'hug': 19, 'in_front_of': 20, 'inside': 21, 'kick': 22, 'kiss': 23, 'knock': 24, 'lean_on': 25, 'lick': 26, 'lift': 27, 'next_to': 28, 'open': 29, 'pat': 30, 'play(instrument)': 31, 'point_to': 32, 'press': 33, 'pull': 34, 'push': 35, 'release': 36, 'ride': 37, 'shake_hand_with': 38, 'shout_at': 39, 'smell': 40, 'speak_to': 41, 'squeeze': 42, 'throw': 43, 'touch': 44, 'towards': 45, 'use': 46, 'watch': 47, 'wave': 48, 'wave_hand_to': 49}
		else:
			raise ValueError('Unknown dataset: {}'.format(dataset))

		self.vid, self.width, self.height, self.traj, self.obj, self.rel, self.traj_len = self._load_anno(anno_path)
		self._merge_rel()
		self._print_anno()

	def _load_anno(self, anno_path):
		with open(anno_path, 'r') as f:
			anno = json.load(f)

		vid = anno['video_id']
		width = anno['width']
		height = anno['height']
		traj_per_video = anno['trajectories'] # 60
		obj_categories = anno['subject/objects']
		rel_per_seg = anno['relation_instances']
		traj_len = len(traj_per_video) # in vidvrd, 'traj_len' is lesser than '# of frames' (x15)

		#####################################################################################
		traj = defaultdict(list)
		for traj_per_frame in traj_per_video:
			for traj_per_inst in traj_per_frame:
				traj[traj_per_inst['tid']].append([traj_per_inst['bbox']['xmin'],
												   traj_per_inst['bbox']['ymin'],
												   traj_per_inst['bbox']['xmax'],
												   traj_per_inst['bbox']['ymax']])
		''' (example)
		traj = defaultdict(list,
            {0: [[14, 8, 912, 574],
	             [13, 7, 913, 573],
	             [12, 7, 915, 573],
	   			 ...
	             [1, 131, 862, 572],
	             [1, 124, 855, 570],
	             [1, 117, 849, 568]],
	        1: [[758, 121, 926, 409],
				[758, 121, 926, 409],
				[758, 121, 926, 409],
				...
				[703, 252, 957, 477],
				[695, 249, 951, 481],
				[686, 247, 944, 484]]})
		'''
		#####################################################################################
		obj = {} # {0: 'dog', 1: 'frisbee'}
		for obj_category in obj_categories:
			obj_idx = self.obj_to_idx[obj_category['category']]
			obj[obj_category['tid']] = obj_idx
		''' (example)
		obj = {0: 9, 1: 13}
		'''
		#####################################################################################
		rel = defaultdict(list)
		for rel_inst in rel_per_seg:
		    rel_idx = self.rel_to_idx[rel_inst['predicate']]
		    rel[rel_inst['begin_fid'], rel_inst['end_fid']].append([rel_inst['subject_tid'],
		                                                            rel_idx,
		                                                            rel_inst['object_tid']])
		''' (example)
		defaultdict(list,
            {(0, 30): [[0, 37, 1], [1, 85, 0], [1, 72, 0]],
             (15, 45): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (30, 60): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (45, 75): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (60, 90): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (75, 105): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (90, 120): [[0, 77, 1], [1, 85, 0], [1, 72, 0]],
             (105, 135): [[0, 3, 1], [1, 85, 0], [1, 72, 0]],
             (120, 150): [[0, 37, 1], [1, 85, 0], [1, 72, 0]],
             (135, 165): [[0, 37, 1], [1, 85, 0], [1, 72, 0]],
             (150, 180): [[0, 37, 1], [1, 85, 0], [1, 72, 0]],
             (165, 195): [[0, 3, 1], [1, 85, 0], [1, 72, 0]]})
		'''

		for key, value in rel.items():
		    rel_per_obj_pair = defaultdict(list)
		    for s, r, o in value:
		        rel_per_obj_pair[s, o].append(r)
		    rel[key] = dict(rel_per_obj_pair)
		''' (example) 
		rel = defaultdict(list,
            {(0, 30): defaultdict(list, {(0, 1): [37], (1, 0): [85, 72]}),
             (15, 45): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (30, 60): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (45, 75): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (60, 90): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (75, 105): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (90, 120): defaultdict(list, {(0, 1): [77], (1, 0): [85, 72]}),
             (105, 135): defaultdict(list, {(0, 1): [3], (1, 0): [85, 72]}),
             (120, 150): defaultdict(list, {(0, 1): [37], (1, 0): [85, 72]}),
             (135, 165): defaultdict(list, {(0, 1): [37], (1, 0): [85, 72]}),
             (150, 180): defaultdict(list, {(0, 1): [37], (1, 0): [85, 72]}),
             (165, 195): defaultdict(list, {(0, 1): [3], (1, 0): [85, 72]})})

        rel[begin, end][sub, obj] = list(pred)
		'''
		#####################################################################################

		return vid, width, height, dict(traj), obj, dict(rel), traj_len

	def _merge_rel(self):
		rel_duration_dict = defaultdict(list)
		for duration, rel_per_seg in self.rel.items():
			for (sub_id, obj_id), rel_idx_list in rel_per_seg.items():
				for rel_idx in rel_idx_list:
					rel_duration_dict[(sub_id, rel_idx, obj_id)].append(duration)

		new_rel_duration_dict = defaultdict(list)
		for rel_triplet, duration_list in rel_duration_dict.items():
			prev_start = 0 
			prev_end = 0
			for start, end in duration_list:
				if prev_start==0 and prev_end==0:
					prev_start = start
					prev_end = end
				elif prev_start <= start <= prev_end:
					prev_end = max(prev_end, end)
				else:
					new_rel_duration_dict[rel_triplet].append((prev_start, prev_end))
					prev_start = start
					prev_end = end
			new_rel_duration_dict[rel_triplet].append((prev_start, prev_end))

		self.rel = dict(new_rel_duration_dict)

	def _print_anno(self):
		print(f'==============================='*2)
		print(f'- video id: {self.vid}')
		print(f'- video size (width,height): {self.width, self.height}')
		print(f'- trajectory length: {self.traj_len}')
		obj = {tid:self.idx_to_obj[obj_idx] for tid, obj_idx in self.obj.items()}
		print(f'- trajectory id to object idx: {obj}')
		print(f'- object trajectories (xmin, ymin, xmax, ymax)')
		pprint(self.traj)
		# rel = {}
		# for duration, rel_per_seg in self.rel.items():
		# 	new_rel_per_seg = {}
		# 	for (sub_id, obj_id), rel_idx_list in rel_per_seg.items():
		# 		rels = []
		# 		for rel_idx in rel_idx_list:
		# 			rels.append(self.idx_to_rel[rel_idx])
		# 		new_rel_per_seg[(obj[sub_id], obj[obj_id])] = rels
		# 	rel[duration] = new_rel_per_seg
		# print(f'- relation instances (start, end):(sub, obj):[relations]')
		# pprint(rel)
		rel = {}
		for (sub_id, rel_idx, obj_id), duration_list in self.rel.items():
			rel[obj[sub_id], self.idx_to_rel[rel_idx], obj[obj_id]] = duration_list
		print(f'- relation instances (sub, rel, obj):[durations]')
		pprint(rel)
		print(f'==============================='*2)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='VidVRD/VidOR annotation')
	parser.add_argument('--data_dir', type=str, default='/home/t2_u1/data/', help='dataset directory')
	parser.add_argument('--dataset', type=str, default='vidvrd', help='the dataset name for training (vidvrd OR vidor/annotation)')
	parser.add_argument('--split', type=str, default='train', help='vidvrd: train/test OR vidor: training/validation')
	parser.add_argument('--anno', type=str, help='ex) ILSVRC2015_train_00005003.json')

	args = parser.parse_args()

	assert (args.dataset=='vidvrd' and args.split in ['train', 'test']) or \
	 args.dataset=='vidor/annotation' and args.split in ['training', 'validation']

	dataset = VidvrdVidorDataset(data_dir=args.data_dir, dataset=args.dataset, split=args.split, anno=args.anno)