import torch
from torch.utils.data import DataLoader
from loss.our_loss import OurLoss
from copy import deepcopy

from .mytrainer import BaseCLODYOLO
from ultralytics import YOLO
from micromind.networks.yolo import Darknet, Yolov8Neck, DetectionHead, SPPF
from micromind.utils.yolo import get_variant_multiples
from copy import deepcopy
from micromind.utils.helpers import get_logger

from collections import OrderedDict
from validation.validator import Pseudolabel, PseudolabelReplay
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

"""LwF (L2) CLOD class for YOLO training"""

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
    

"""Trainer for RCLPOD"""

class YOLORCLPOD(BaseCLODYOLO):

    def __init__(self, m_cfg, hparams, data_cfg_path_val, exp_folder, teacher_dict, logger=None, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(m_cfg, hparams, data_cfg_path_val, exp_folder, logger, *args, **kwargs)
        self.hparams = deepcopy(hparams)
        self.exp_folder = exp_folder
        self.hparams.data_cfg_val = data_cfg_path_val
        self.m_cfg = m_cfg

        w, r, d = get_variant_multiples(hparams.model_size)

        self.modules["backbone"] = Darknet(w, r, d)
        sppf_size = 768 if hparams.model_size=="m" else int(512 * w * r)
        self.modules["backbone"] = YOLOv8Backbone(version=hparams.model_size)
        self.modules["sppf"] = SPPF(sppf_size, int(512 * w * r))
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

        self.feats_distill = hparams.feats_distill
        old_classes = [i for i in range(self.m_cfg.classes[0])]
        self.criterion = OurLoss(self.m_cfg, self.modules["head"], self.device,
                                 old_classes=old_classes, consts=self.feats_distill,
                                 new_classes=self.m_cfg.classes)


        # logger for mAPs
        self.logger = logger

        print("Number of parameters for each module:")
        print(self.compute_params())

    def forward(self, batch):
        """Runs the forward method by calling every module."""

        self.teacher_feats = []
        self.student_feats = []

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
        
        if self.feats_distill[0]:
            self.teacher_feats += neck_input_teacher
            self.student_feats += neck_input

        neck = self.modules["neck"](*neck_input)

        if self.feats_distill[1]:
            with torch.no_grad():
                neck_teacher = self.modules["teacher_neck"](*neck_input_teacher)

            self.teacher_feats += neck_teacher
            self.student_feats += neck

        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
            self.student_feats,
            self.teacher_feats
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