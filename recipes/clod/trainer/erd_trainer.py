import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from loss.LwFloss import LwFLoss, LwFLossV2, ERS
from loss.erd_loss import ERDLoss
import math
from copy import deepcopy

import micromind as mm
from .mytrainer import BaseCLODYOLO
from ultralytics import YOLO
from micromind.networks.yolo import Darknet, Yolov8Neck, DetectionHead, SPPF

from micromind.utils.yolo import get_variant_multiples
import os
from validation.validator import DetectionValidator
from copy import deepcopy
from micromind.utils.helpers import get_logger

from collections import OrderedDict
from micromind.utils.checkpointer import Checkpointer
from micromind.core import Metric, Stage
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from validation.validator import PseudolabelOCDMLwF
from data.OCDM import OCDM, OCDMLwF

from loss.lwf_replay_loss import LwFLossReplay2, NaiveLwFLossReplay, LwFLossReplayERD
from loss.yolo_loss import Loss


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



class YOLOERD(BaseCLODYOLO):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, teacher_dict, logger=None, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg

        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = Darknet(w, r, d)
        self.modules["backbone"] = YOLOv8Backbone()
        self.modules["sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        # load student (previous model)
        self.modules.load_state_dict(teacher_dict)

        # teacher
        self.modules["teacher_backbone"] = Darknet(w, r, d)
        self.modules["teacher_backbone"] = YOLOv8Backbone()
        self.modules["teacher_sppf"] = SPPF(int(512 * w * r), int(512 * w * r))
        self.modules["teacher_neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["teacher_head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        # modify state dict to match teacher keys
        teacher_state_dict = OrderedDict([("teacher_"+k, v) for k,v in teacher_dict.items()])
        # load just teacher modules
        self.modules.load_state_dict(teacher_state_dict, strict=False)

        self.output_teacher = None # used to save output teacher

        self.lwf_params = self.hparams.lwf

        if len(self.lwf_params) > 2:
            # temp fix
            old_classes = [i for i in range(self.m_cfg.classes[0])]


            # yolo loss + ERD loss
            self.criterion = ERDLoss(self.m_cfg, self.modules["head"], self.device,
                                       c1 = self.lwf_params[0], c2 = self.lwf_params[1],
                                       old_classes=old_classes, c3=self.lwf_params[2], classes=self.m_cfg.classes)

        else:
            # yolov8 loss + l2 for lwf
            self.criterion = LwFLoss(self.m_cfg, self.modules["head"], self.device,
                                      lwf=self.lwf_params[0], new_classes=self.m_cfg.classes)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        if self.modules.training:   

            preprocessed_batch = self.preprocess_batch(batch)
            backbone = self.modules["backbone"](
                preprocessed_batch["img"].to(self.device)
            )
            with torch.no_grad():
                backbone_teacher = self.modules["teacher_backbone"](
                    preprocessed_batch["img"].to(self.device)
                )
        else:

            if torch.is_tensor(batch):
                backbone = self.modules["backbone"](batch)
                if "sppf" in self.modules.keys():
                    neck_input = list(backbone)[0:2]
                    neck_input.append(self.modules["sppf"](backbone[2]))
                else:
                    neck_input = backbone
                neck = self.modules["neck"](*neck_input)
                head = self.modules["head"](neck)
                return head

            backbone = self.modules["backbone"](batch["img"] / 255)
            with torch.no_grad():
                backbone_teacher = self.modules["teacher_backbone"](batch["img"] / 255)

        if "sppf" in self.modules.keys():
            neck_input = list(backbone)[0:2]
            neck_input.append(self.modules["sppf"](backbone[2]))
            neck_input_teacher = list(backbone_teacher)[0:2]
            neck_input_teacher.append(self.modules["teacher_sppf"](backbone_teacher[2]))
        else:
            neck_input = backbone
            neck_input_teacher = backbone_teacher

        neck = self.modules["neck"](*neck_input)

        with torch.no_grad():
            neck_teacher = self.modules["teacher_neck"](*neck_input_teacher)
            self.output_teacher = self.modules["teacher_head"](neck_teacher)

        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
            self.output_teacher
        )

        return lossi_sum

    def save_last_model(self, path, task):
        """Save just student model"""

        state_dict = self.modules.state_dict()

        list_params_student = []
        # remove teacher and save just student: remove (key, value) if key has "teacher"
        for k,v in state_dict.items():
            if "teacher" in k:
                continue
            list_params_student.append((k,v))

        student_state_dict = OrderedDict(list_params_student)

        torch.save(student_state_dict, path+f"/model_task_{task}.pt")


    def load_model_prev_task(self, state_dict):
        """load student net from previous task"""
        self.modules.load_state_dict(state_dict, strict=False)



class YOLOERDReplay(BaseCLODYOLO):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, teacher_dict, logger=None, classes_per_task=None,*args, **kwargs):
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

        # load student (previous model)
        self.modules.load_state_dict(teacher_dict)

        # teacher
        self.modules["teacher_backbone"] = Darknet(w, r, d)
        self.modules["teacher_backbone"] = YOLOv8Backbone(version=hparams.model_size)
        self.modules["teacher_sppf"] = SPPF(sppf_size, int(512 * w * r))
        self.modules["teacher_neck"] = Yolov8Neck(
            filters=[int(256 * w), int(512 * w), int(512 * w * r)],
            heads=hparams.heads,
            d=d,
        )
        self.modules["teacher_head"] = DetectionHead(
            hparams.num_classes,
            filters=(int(256 * w), int(512 * w), int(512 * w * r)),
            heads=hparams.heads,
        )

        # modify state dict to match teacher keys
        teacher_state_dict = OrderedDict([("teacher_"+k, v) for k,v in teacher_dict.items()])
        # load just teacher modules
        self.modules.load_state_dict(teacher_state_dict, strict=False)

        self.output_teacher = None # used to save output teacher

        self.lwf_params = self.hparams.lwf
        old_classes = [i for i in range(self.m_cfg.classes[0])]

        if len(self.lwf_params) > 2:
            # temp fix
            
            # yolov8 loss + ERD loss
            self.criterion = LwFLossReplayERD(self.m_cfg, self.modules["head"], self.device,
                                       c1 = self.lwf_params[0], c2 = self.lwf_params[1],
                                       old_classes=old_classes, c3=self.lwf_params[2], classes=self.m_cfg.classes, mask_labels=classes_per_task)
            self.val_criterion = Loss(self.m_cfg, self.modules["head"],
                                   self.device, classes=old_classes+self.m_cfg.classes)
        else:

            self.criterion = NaiveLwFLossReplay(self.m_cfg, self.modules["head"], self.device,
                                                self.lwf_params, new_classes=self.m_cfg.classes)
            self.val_criterion = Loss(self.m_cfg, self.modules["head"],
                                   self.device, classes=old_classes+self.m_cfg.classes)

        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}

        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)
            if k == "num_labels":
                preprocessed_batch["num_labels"] = batch["num_labels"]

        return preprocessed_batch
    def forward(self, batch):
        """Runs the forward method by calling every module."""
        if self.modules.training:

            preprocessed_batch = self.preprocess_batch(batch)
            backbone = self.modules["backbone"](
                preprocessed_batch["img"].to(self.device)
            )
            with torch.no_grad():
                backbone_teacher = self.modules["teacher_backbone"](
                    preprocessed_batch["img"].to(self.device)
                )
        else:

            if torch.is_tensor(batch):
                backbone = self.modules["backbone"](batch)
                if "sppf" in self.modules.keys():
                    neck_input = list(backbone)[0:2]
                    neck_input.append(self.modules["sppf"](backbone[2]))
                else:
                    neck_input = backbone
                neck = self.modules["neck"](*neck_input)
                head = self.modules["head"](neck)
                return head

            backbone = self.modules["backbone"](batch["img"] / 255)
            with torch.no_grad():
                backbone_teacher = self.modules["teacher_backbone"](batch["img"] / 255)

        if "sppf" in self.modules.keys():
            neck_input = list(backbone)[0:2]
            neck_input.append(self.modules["sppf"](backbone[2]))
            neck_input_teacher = list(backbone_teacher)[0:2]
            neck_input_teacher.append(self.modules["teacher_sppf"](backbone_teacher[2]))
        else:
            neck_input = backbone
            neck_input_teacher = backbone_teacher

        neck = self.modules["neck"](*neck_input)

        with torch.no_grad():
            neck_teacher = self.modules["teacher_neck"](*neck_input_teacher)
            self.output_teacher = self.modules["teacher_head"](neck_teacher)

        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        if self.modules.training:

            lossi_sum, lossi = self.criterion(
                pred,
                preprocessed_batch,
                self.output_teacher

            )
        
        else:
            lossi_sum, lossi = self.val_criterion(
                pred,
                preprocessed_batch,
            )

        return lossi_sum

    def save_last_model(self, path, task):
        """Save just student model"""

        state_dict = self.modules.state_dict()

        list_params_student = []
        # remove teacher and save just student: remove (key, value) if key has "teacher"
        for k,v in state_dict.items():
            if "teacher" in k:
                continue
            list_params_student.append((k,v))

        student_state_dict = OrderedDict(list_params_student)

        torch.save(student_state_dict, path+f"/model_task_{task}.pt")


    def load_model_prev_task(self, state_dict):
        """load student net from previous task"""
        self.modules.load_state_dict(state_dict, strict=False)


    def train(
        self,
        epochs: int = 1,
        warmup: bool = False,
        datasets: Dict = {},
        metrics: List[Metric] = [],
        checkpointer: Optional[Checkpointer] = None,
        max_norm=10.0,
        debug: Optional[bool] = False,
        skip=False
    ):
        self.epochs = epochs
        if not warmup:
            logger.info("No warmup!")
            super().train(epochs, datasets, metrics, checkpointer, max_norm, debug)
        else:
            warmup_finished = False
            self.datasets = datasets
            self.metrics = metrics
            self.checkpointer = checkpointer
            assert "train" in self.datasets, "Training dataloader was not specified."
            assert epochs > 0, "You must specify at least one epoch."
            self.epochs -= self.hparams.warmup_epochs
            self.debug = debug

            self.on_train_start()
            
            if skip:
                return None

            if self.accelerator.is_local_main_process:
                logger.info(
                    f"Starting from epoch {self.start_epoch + 1}."
                    + f" Training is scheduled for {epochs} epochs."
                )
            
            warmup_epochs = self.hparams.warmup_epochs
            warmup_bias_lr = self.hparams.warmup_bias_lr
            warmup_lrf = self.hparams.lr0
            warmup_momentum = self.hparams.warmup_momentum
            warmup_f_momentum = self.hparams.momentum

            nb = len(self.datasets["train"])
            nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1  # warmup iterations

            
            for e in range(self.start_epoch + 1, epochs + 1):
                self.current_epoch = e
                pbar = tqdm(
                    self.datasets["train"],
                    unit="batches",
                    ascii=True,
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                loss_epoch = 0
                pbar.set_description(f"Running epoch {self.current_epoch}/{epochs}")
                self.modules.train()
                for idx, batch in enumerate(pbar):
                    
                    # warmup
                    ni = idx + nb * (e - 1)
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        for j, x in enumerate(self.opt.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [warmup_bias_lr if j == 0 else 0.0, warmup_lrf]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [warmup_momentum, warmup_f_momentum])
                    else:
                        warmup_finished = True



                    if isinstance(batch, list):
                        batch = [b.to(self.device) for b in batch]

                    self.opt.zero_grad()

                    with self.accelerator.autocast():
                        model_out = self(batch)
                        loss = self.compute_loss(model_out, batch)
                        loss_epoch += loss.item()

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        self.modules.parameters(), max_norm=max_norm
                    )
                    self.opt.step()


                    # loss_epoch += loss.item()

                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            m(model_out, batch, Stage.train, self.device)

                    running_train = {}
                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            running_train["train_" + m.name] = m.reduce(Stage.train)

                    running_train.update({"train_loss": loss_epoch / (idx + 1)})

                    pbar.set_postfix(**running_train)

                    if self.debug and idx > 10:
                        break

                pbar.close()

                train_metrics = {}
                for m in self.metrics:
                    if (self.current_epoch + 1) % m.eval_period == 0 and not m.eval_only:
                        train_metrics["train_" + m.name] = m.reduce(Stage.train, True)

                train_metrics.update({"train_loss": loss_epoch / (idx + 1)})

                if "val" in datasets:
                    val_metrics = self.validate()
                else:
                    train_metrics.update({"val_loss": loss_epoch / (idx + 1)})
                    val_metrics = train_metrics

                self.on_train_epoch_end()

                if self.accelerator.is_local_main_process and self.checkpointer is not None:
                    self.checkpointer(
                        self,
                        train_metrics,
                        val_metrics,
                    )

                if e >= 1 and self.debug:
                    break

                if hasattr(self, "lr_sched") and warmup_finished:
                    # ok for cos_lr
                    # self.lr_sched.step(val_metrics["val_loss"])
                    print(f"sched step - old LR={self.lr_sched.get_lr()}")
                    self.lr_sched.step()
                    print(f"sched step - new LR={self.lr_sched.get_lr()}")

            self.on_train_end()
        return None
    

    def add_labels_for_ocdm(self, dataset):

        if not isinstance(dataset, OCDM):
            dataset.augment = False
            dataset.transforms = dataset.build_transforms(hyp=dataset.hyp)


        # create loader
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fnv2", OCDM.collate_fn),
        )

        pseudolabel = PseudolabelOCDMLwF(dataloader=loader, ths=self.hparams.inference_ths, ocdm=True)

        pseudolabel(model=self, classes=list(range(self.m_cfg.classes[-1]+1)))

        return dataset
    
