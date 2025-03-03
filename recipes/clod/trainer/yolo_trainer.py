"""
YOLO training.

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train_yolov8.py cfg/<cfg_file>.py

Authors:
    - Matteo Beltrami, 2024
    - Francesco Paissan, 2024
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loss.yolo_loss import Loss
import math
from copy import deepcopy

import micromind as mm
from .mytrainer import BaseCLODYOLO
from ultralytics import YOLO

from micromind.networks.yolo import Darknet, Yolov8Neck, DetectionHead, SPPF
from micromind.utils.yolo import get_variant_multiples
import os
from validation.validator import DetectionValidator, Pseudolabel, PseudolabelReplay
from copy import deepcopy
from micromind.utils.helpers import get_logger
from data.OCDM import OCDM


logger = get_logger()

# This is used ONLY if you are not using argparse to get the hparams
default_cfg = {
    "output_folder": "results",
    "experiment_name": "micromind_exp",
    "opt": "adam",  # this is ignored if you are overriding the configure_optimizers
    "lr": 0.001,  # this is ignored if you are overriding the configure_optimizers
    "debug": False,
}

class YOLOv8Backbone(torch.nn.Module):

    def __init__(self, version="n"):
        super().__init__()
        model = "yolov8"+version+"-cls.pt"
        classifier = YOLO(model)

        self.sequential = classifier.model.model

        self.ps_indices = [4, 6, 8]
        self.num_blocks = 9

    def forward(self, x):

        ps = []
        for i in range(self.num_blocks):
            if i in self.ps_indices:
                ps.append(self.sequential[i](x))
                x = ps[-1]
            else:
                x = self.sequential[i](x)

        return ps   


"""Base CLOD trainer for any ER based method and pseudo-label method. No CL approach is actually used. Use it as parent class."""
class YOLOOurs(BaseCLODYOLO):
    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, logger=None, oldlabels=False, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg
        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = Darknet(w, r, d)
        self.modules["backbone"] = YOLOv8Backbone(version=hparams.model_size)
        sppf_size = 768 if hparams.model_size=="m" else int(512 * w * r)
        self.modules["sppf"] = SPPF(sppf_size, int(512 * w * r))
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        print(hparams.num_classes)
        self.modules["head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        """
        if oldlabels and self.m_cfg.classes[0]>0:
            old_classes = [i for i in range(self.m_cfg.classes[0])]
            classes = old_classes + self.m_cfg.classes
        else:
            classes = self.m_cfg.classes
        """
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def add_pseudo_labels(self, data, classes):

        dataloader = deepcopy(data)

        # disable temp augmentation
        dataloader.dataset.augment = False
        dataloader.dataset.transforms = dataloader.dataset.build_transforms(hyp=dataloader.dataset.hyp)

        pseudolabel = Pseudolabel(classes=classes, dataloader=dataloader, ths=self.hparams.inference_ths)
        # add labels to dataset
        pseudolabel(data, model=self)
    
    def add_pseudo_lables_replay_memory(self, replay_memory):


        is_ocdm = isinstance(replay_memory, OCDM)

        # create loader
        loader = DataLoader(
            replay_memory,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=getattr(replay_memory, "collate_fnv2", OCDM.collate_fn),
        )

        pseudolabel = PseudolabelReplay(dataloader=loader, ths=self.hparams.inference_ths, ocdm=is_ocdm)
        pseudolabel(model=self, classes=self.m_cfg.classes)

        return replay_memory