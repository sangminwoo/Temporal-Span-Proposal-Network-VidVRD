import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from dataset_dict import to_coco_format
from detectron2.data import MetadataCatalog, DatasetCatalog

anno_dir = "/home/t2_u1/data/vidvrd/"
for d in ["train", "test"]:
    DatasetCatalog.register("vidvrd_" + d, lambda d=d:to_coco_format(anno_dir, d))
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
vidvrd_metadata = MetadataCatalog.get("vidvrd_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = ("vidvrd_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 35  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
model = trainer.model