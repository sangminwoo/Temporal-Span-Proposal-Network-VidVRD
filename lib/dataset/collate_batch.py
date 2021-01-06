# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from lib.dataset.list_data import to_data_list

class BatchCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # inputs = to_data_list(transposed_batch[0])
        inputs = transposed_batch[0]
        targets = transposed_batch[1]
        index = transposed_batch[2]
        return inputs, targets, index


class BBoxAugCollator:
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
