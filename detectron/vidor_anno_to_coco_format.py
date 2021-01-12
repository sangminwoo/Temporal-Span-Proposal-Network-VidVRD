import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

setup_logger()

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# vidor
vidor_obj_to_idx = {
	'adult': 0, 'aircraft': 1, 'antelope': 2, 'baby': 3, 'baby_seat': 4,
	'baby_walker': 5, 'backpack': 6, 'ball/sports_ball': 7, 'bat': 8, 'bear': 9,
	'bench': 10, 'bicycle': 11, 'bird': 12, 'bottle': 13, 'bread': 14,
	'bus/truck': 15, 'cake': 16, 'camel': 17, 'camera': 18, 'car': 19,
	'cat': 20, 'cattle/cow': 21, 'cellphone': 22, 'chair': 23, 'chicken': 24,
	'child': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dish': 29,
	'dog': 30, 'duck': 31, 'electric_fan': 32, 'elephant': 33, 'faucet': 34,
	'fish': 35, 'frisbee': 36, 'fruits': 37, 'guitar': 38, 'hamster/rat': 39,
	'handbag': 40, 'horse': 41, 'kangaroo': 42, 'laptop': 43, 'leopard': 44,
	'lion': 45, 'microwave': 46, 'motorcycle': 47, 'oven': 48, 'panda': 49,
	'penguin': 50, 'piano': 51, 'pig': 52, 'rabbit': 53, 'racket': 54,
	'refrigerator': 55, 'scooter': 56, 'screen/monitor': 57, 'sheep/goat': 58, 'sink': 59,
	'skateboard': 60, 'ski': 61, 'snake': 62, 'snowboard': 63, 'sofa': 64,
	'squirrel': 65, 'stingray': 66, 'stool': 67, 'stop_sign': 68, 'suitcase': 69,
	'surfboard': 70, 'table': 71, 'tiger': 72, 'toilet': 73, 'toy': 74,
	'traffic_light': 75, 'train': 76, 'turtle': 77, 'vegetables': 78, 'watercraft': 79
}

def vidor_to_coco_format(anno_dir, split):
	dataset_dicts = []
	for dirs in os.listdir(os.path.join(anno_dir, 'annotation', split)):
		for files in os.listdir(os.path.join(anno_dir, 'annotation', split, dirs)):
			with open(os.path.join(anno_dir, 'annotation', split, dirs, files)) as f:
				anno = json.load(f)

			tid_to_obj = {
				obj_tid['tid']:obj_tid['category'] for obj_tid in anno['subject/objects']
			}
			  
			record = {}
			record['height'] = anno['height']
			record['width'] = anno['width']

			for bbox_idx, bboxes in enumerate(anno['trajectories']):
				record['file_name'] = os.path.join(
					anno_dir, 'image', anno['video_id'], '{0:0=5d}.jpg'.format(bbox_idx)
				)
				record['image_id'] = '{0:0=5d}'.format(bbox_idx)
				objs = []
				for bbox in bboxes:
					obj = {
						'bbox': [bbox['bbox']['xmin'],
								 bbox['bbox']['ymin'],
								 bbox['bbox']['xmax'],
								 bbox['bbox']['ymax']],
						'bbox_mode': BoxMode.XYXY_ABS,
						'category_id': vidor_obj_to_idx[tid_to_obj[bbox['tid']]]
					}
					objs.append(obj)

				record['annotations'] = objs
				dataset_dicts.append(record)

	return dataset_dicts

if __name__=='__main__':
	# vidor_to_coco_format
	anno_dir = "/home/t2_u1/data/vidor/" 
	for d in ["training", "validation"]:
		DatasetCatalog.register("vidor_" + d, lambda d=d:vidor_to_coco_format(anno_dir, d))
		MetadataCatalog.get("vidor_" + d).set(
			thing_classes=[
				'adult', 'aircraft', 'antelope', 'baby', 'baby_seat',
				'baby_walker', 'backpack', 'ball/sports_ball', 'bat', 'bear',
				'bench', 'bicycle', 'bird', 'bottle', 'bread',
				'bus/truck', 'cake', 'camel', 'camera', 'car',
				'cat', 'cattle/cow', 'cellphone', 'chair', 'chicken',
				'child', 'crab', 'crocodile', 'cup', 'dish',
				'dog', 'duck', 'electric_fan', 'elephant', 'faucet',
				'fish', 'frisbee', 'fruits', 'guitar', 'hamster/rat',
				'handbag', 'horse', 'kangaroo', 'laptop', 'leopard',
				'lion', 'microwave', 'motorcycle', 'oven', 'panda',
				'penguin', 'piano', 'pig', 'rabbit', 'racket',
				'refrigerator', 'scooter', 'screen/monitor', 'sheep/goat', 'sink',
				'skateboard', 'ski', 'snake', 'snowboard', 'sofa',
				'squirrel', 'stingray', 'stool', 'stop_sign', 'suitcase',
				'surfboard', 'table', 'tiger', 'toilet', 'toy',
				'traffic_light', 'train', 'turtle', 'vegetables', 'watercraft'
			]
		)
	vidor_metadata = MetadataCatalog.get("vidor_training")
	dataset_dicts = vidor_to_coco_format(anno_dir, "training")
	with open("./vidor_coco_format.json", "w") as f:
		j = json.dump(dataset_dicts, f)

	# with open("./vidor_coco_format.json", "r") as f:
	# 	dataset_dicts = json.load(f)

	num_images_to_show = 3
	for d in random.sample(dataset_dicts, num_images_to_show):
		img = cv2.imread(d["file_name"])
		visualizer = Visualizer(img[:, :, ::-1], metadata=vidor_metadata, scale=0.5)
		out = visualizer.draw_dataset_dict(d)
		cv2.imshow('sample', out.get_image()[:, :, ::-1])
		cv2.waitKey(0)
		cv2.destroyAllWindows()