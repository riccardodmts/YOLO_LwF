import os
import yaml
import random
from yaml import Loader, CDumper as Dumper
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from prepare_data import TasksGenerator
import numpy as np

import micromind as mm
from trainer.ours_trainer import YOLORCLPOD
from trainer.yolo_trainer import YOLOOurs
from micromind.utils import parse_configuration
from micromind.utils.yolo import load_config
import sys
import os
import sys
from utils import CLODLogger
from data.replay_data import ReplayDataloader 
from data.OCDM import OCDM


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

    di = "results"

    for i, (train_loader, val_loader, other_info) in enumerate(TasksGenerator(m_cfg, data_cfg, hparams, exp_type, save_stats)):

        # define logger for CLOD (one file per task with mAPs classes seen)
        logger = CLODLogger("./"+di, len(data_cfg["names"]), i, use_tensorboard=hparams.use_tensorboard)

        # define experiment folder for current task
        exp_folder = mm.utils.checkpointer.create_experiment_folder(
                        di, hparams.experiment_name+f"_task_{i}"
                        )
        
        checkpointer = mm.utils.checkpointer.Checkpointer(
                        exp_folder, hparams=hparams, key="loss"
                        )
        
        # modify cfg for validator (temp fix)
        data_cfg_new_path = modify_yaml(hparams.data_cfg, "val", data_cfg["val"])

        yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path, exp_folder = exp_folder, logger=logger)

        if i>= 1:
            # load model previous task and add pseudo labels
            yolo_mind.load_model_prev_task("./"+di, i-1)
            yolo_mind.add_pseudo_labels(train_loader, other_info["old_classes"])

            # temp fix (reload model)
            teacher_dict = torch.load("./"+di+f"/model_task_{i-1}.pt")
            yolo_mind = YOLORCLPOD(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                              exp_folder = exp_folder, logger=logger, teacher_dict=teacher_dict)

            # if current task is not the first task, use replay memory
            task_tr_loader = ReplayDataloader(train_loader, replay_mem)
        else:
            task_tr_loader = train_loader


        yolo_mind.train(
            epochs=other_info["epochs"],  # number epochs based on current task
            datasets={"train": task_tr_loader, "val": val_loader},
            metrics=[],
            checkpointer=checkpointer,
            debug=hparams.debug,
            warmup=True
        )


        # save model
        yolo_mind.save_last_model("./"+di, i)

        if i == 0:
            # create replay memory with first dataset
            replay_mem = OCDM(hparams.replay_mem_size, train_loader.dataset,
                               nc=other_info["nc_val"], max_nc=len(data_cfg["names"]),
                                results_dir="./"+di,
                                  batch_size=hparams.batch_size_ocdm, ths=hparams.ocdm_ths,
                                count_dup=hparams.save_num_dup)
        else:

            yolo_mind = YOLOOurs(m_cfg, hparams=hparams, data_cfg_path_val = data_cfg_new_path,
                              exp_folder = exp_folder, logger=logger, oldlabels=True)
            yolo_mind.load_model_prev_task("./"+di, i)
            yolo_mind.add_pseudo_lables_replay_memory(replay_mem)
            
            # update memeory with new dataset
            replay_mem.update_memory(train_loader.dataset, nc=other_info["nc_val"])
