# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER.TYPE == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER.TYPE == "adam":
        optimizer = torch.optim.Adam(params, lr)
    else:
        raise ValueError("{} is not defined".format(cfg.SOLVER.OPTIMIZER.TYPE))
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER.TYPE == "multi":
        return MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.SCHEDULER.MILESTONES,
            gamma=cfg.SOLVER.SCHEDULER.GAMMA
        )
    elif cfg.SOLVER.SCHEDULER.TYPE == "warmup_multi":
        return WarmupMultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.SCHEDULER.MILESTONES,
            gamma=cfg.SOLVER.SCHEDULER.GAMMA,
            warmup_factor=cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
        )
    elif cfg.SOLVER.SCHEDULER.TYPE == "plateau":
        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.9,
            patience=100,
            verbose=False
        )
    else:
        raise ValueError("{} is not defined".format(cfg.SOLVER.SCHEDULER.TYPE))

def build_optimizer_scheduler(cfg, model):
    optimizer = make_optimizer(cfg, model) 
    scheduler = make_lr_scheduler(cfg, optimizer)
    return optimizer, scheduler
