import os
import yaml
import random
from yaml import Loader, CDumper as Dumper
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from prepare_data import TasksGenerator
import numpy as np

import micromind as mm
from trainer.yolo_trainer import YOLOOurs
from trainer.erd_trainer import YOLOERD
from micromind.utils import parse_configuration
from micromind.utils.yolo import load_config
import sys
import os
import sys
from utils import CLODLogger


def modify_yaml(path, key, path_val):
    """Modify .yaml by changing val path.
    Return path to new .yaml"""

    with open(path) as f:
        doc = yaml.load(f, Loader=Loader)

    doc[key] = path_val

    new_path = path.split(".")[0]+"v2.yaml"

    with open(new_path, 'w') as f:
        yaml.dump(doc, f, Dumper=Dumper)

    return new_path


def set_seed():
    """ set seed for reproducibility"""
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":

    set_seed()
    
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])
    if len(hparams.input_shape) != 3:
        hparams.input_shape = [
            int(x) for x in "".join(hparams.input_shape).split(",")
        ]  # temp solution
        print(f"Setting input shape to {hparams.input_shape}.")
    
    # get clod exp e.g. 15p1
    exp_type = hparams.exp
    # save statistics of classes for each task or not
    save_stats = hparams.save_stats

    m_cfg, data_cfg = load_config(hparams.data_cfg)

    # check if specified path for images is different, correct it in case
    # data_cfg = replace_datafolder(hparams, data_cfg)
    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution
    dire = "results" 

    for i, (train_loader, val_loader, other_info) in enumerate(TasksGenerator(m_cfg, data_cfg, hparams, exp_type, save_stats)):

        # define logger for CLOD (one file per task with mAPs classes seen)
        logger = CLODLogger("./"+dire, len(data_cfg["names"]), i, use_tensorboard=hparams.use_tensorboard)

        # define experiment folder for current task
        exp_folder = mm.utils.checkpointer.create_experiment_folder(
                        dire, hparams.experiment_name+f"_task_{i}"
                        )
        
        checkpointer = mm.utils.checkpointer.Checkpointer(
                        exp_folder, hparams=hparams, key="loss"
                        )
        
        # modify cfg for validator (temp fix)
        data_cfg_new_path = modify_yaml(hparams.data_cfg, "val", data_cfg["val"])

        if i == 0:

            yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path, exp_folder = exp_folder, logger=logger)
        else:
            # load teacher state dict
            teacher_dict = torch.load("./"+dire+f"/model_task_{i-1}.pt")
            yolo_mind = YOLOERD(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                                 exp_folder = exp_folder, teacher_dict=teacher_dict, logger=logger)


        yolo_mind.train(
            epochs=other_info["epochs"],  # number epochs based on current task
            datasets={"train": train_loader, "val": val_loader},
            metrics=[],
            checkpointer=checkpointer,
            debug=hparams.debug,
            warmup=True
        )

        # save model
        yolo_mind.save_last_model("./"+dire+"/", i)
