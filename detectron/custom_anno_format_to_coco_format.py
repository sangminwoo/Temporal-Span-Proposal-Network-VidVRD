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

obj_to_idx = {
	'airplane': 0, 'antelope': 1, 'ball': 2, 'bear': 3, 'bicycle': 4,
	'bird': 5, 'bus': 6, 'car': 7, 'cattle': 8, 'dog': 9,
	'domestic_cat': 10, 'elephant': 11, 'fox': 12, 'frisbee': 13, 'giant_panda': 14,
	'hamster': 15, 'horse': 16, 'lion': 17, 'lizard': 18, 'monkey': 19,
	'motorcycle': 20, 'person': 21, 'rabbit': 22, 'red_panda': 23, 'sheep': 24,
	'skateboard': 25, 'snake': 26, 'sofa': 27, 'squirrel': 28, 'tiger': 29,
	'train': 30, 'turtle': 31, 'watercraft': 32, 'whale': 33, 'zebra': 34
}

def to_coco_format(anno_dir, split):
	dataset_dicts = []
	for root, dirs, files in os.walk(os.path.join(anno_dir, split)):
		assert len(files) > 0, "annotation files must be exist!"
		for video_idx, file_name in enumerate(files):
			with open(os.path.join(root, file_name)) as f:
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
						'category_id': obj_to_idx[tid_to_obj[bbox['tid']]]
					}
					objs.append(obj)

				record['annotations'] = objs
				dataset_dicts.append(record)

	return dataset_dicts

if __name__=='__main__':
	anno_dir = "/home/t2_u1/data/vidvrd/"
	for d in ["train", "test"]:
	    DatasetCatalog.register("vidvrd_" + d, lambda d=d: to_coco_format(anno_dir, d))
	    MetadataCatalog.get("vidvrd_" + d).set(
	    	thing_classes=[
	    	'airplane', 'antelope', 'ball', 'bear', 'bicycle',
	    	'bird', 'bus', 'car', 'cattle', 'dog',
	    	'domestic_cat', 'elephant', 'fox', 'frisbee', 'giant_panda',
	    	'hamster', 'horse', 'lion', 'lizard', 'monkey',
	    	'motorcycle', 'person', 'rabbit', 'red_panda', 'sheep',
	    	'skateboard', 'snake', 'sofa', 'squirrel', 'tiger',
	    	'train', 'turtle', 'watercraft', 'whale', 'zebra']
	    )
	# vidvrd_metadata = MetadataCatalog.get("vidvrd_train")
	vidvrd_metadata = MetadataCatalog.get("vidvrd_test")

	# dataset_dicts = to_coco_format(anno_dir, "train")
	dataset_dicts = to_coco_format(anno_dir, "test")
	for d in random.sample(dataset_dicts, 3):
		print(d)
		print(d["file_name"])
		img = cv2.imread(d["file_name"])
		visualizer = Visualizer(img[:, :, ::-1], metadata=vidvrd_metadata, scale=0.5)
		out = visualizer.draw_dataset_dict(d)
		cv2.imshow('sample', out.get_image()[:, :, ::-1])
		cv2.waitKey(0)
		cv2.destroyAllWindows()