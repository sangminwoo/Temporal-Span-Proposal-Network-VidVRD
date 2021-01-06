import copy
import bisect
import torch
from torch.utils.data import DataLoader
from . import samplers
from .transforms import build_transforms
from .collate_batch import BatchCollator
from lib.utils.comm import get_world_size, get_rank
from .vrdataset import VRDataset

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def build_data_loader(cfg, basedata, phase, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    dataset = VRDataset(cfg, dataset=basedata)
    sampler = make_data_sampler(
        dataset,
        shuffle=True if phase in ["train", "val"] else False,
        distributed=is_distributed
    )
    
    images_per_batch = cfg.DATASET.TRAIN_BATCH_SIZE if phase == "train" else cfg.DATASET.TEST_BATCH_SIZE
    if get_rank() == 0:
        print("segments_per_batch: {}, num_gpus: {}".format(images_per_batch, num_gpus))
    
    images_per_gpu = images_per_batch // num_gpus if phase == "train" else images_per_batch
    start_iter = start_iter if phase == "train" else 0
    num_iters = cfg.SOLVER.MAX_ITER if phase == "train" or (phase == "val" and cfg.val) else None
    
    aspect_grouping = False
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
    )
    
    collator = BatchCollator()
    
    dataloader = DataLoader(dataset,
            num_workers=images_per_batch,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
    return dataloader