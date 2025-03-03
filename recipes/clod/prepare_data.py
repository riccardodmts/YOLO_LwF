"""
Data preparation script for YOLO training. Parses ultralytics yaml files
and, if needed, downloads them on disk.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

from typing import Dict, Union
import os

from torch.utils.data import DataLoader, ConcatDataset
from data.mybuild import build_yolo_dataset
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import sys

def get_class_ids(all_names : list[str], task_names : list[str]) -> list[str]:
    """
    Get list of ids (str format) for the given task names and all the possible class names.7

    :param all_names: list of all the class names e.g. all 80 COCO class names
    :param task_names: list of class names for current task

    :return: list of ids (str format) for the given task names
    """

    ids = [str(id) for  id, class_name in enumerate(all_names) if class_name in task_names]

    if len(ids) == 0:
        raise Exception("None of the task-class names appear in the original list!")
    
    return ids


def create_task_txt(imgs_paths : Union[str, list[str]], all_names : list[str], task_names : list[str], task_name : str, other_criteria = None, return_stats = False) -> str:
    """
    Filter the images not involved in the current CL task: create a .txt file (task_name.txt)
    with the names of images/labels that have at least one instance of the classes 
    involved in the current CL task. 
    
    NOTE: example file format with two images dog.jpg, cat.jpg -> dog\ncat\nEOF

    :param imgs_paths: relative paths (list[str] or str) to dirs with images
    :param all_names: list of all possible classes (e.g. all 80 COCO class names)
    :param task_names: list of class names for current task
    :param task_name: name for the task used as output file name e.g. task_name.txt
    :other_creiteria: not used
    :return_stats: compute and return also stats of classes present in the dataset

    :return: str path to file
    """
    paths = deepcopy(imgs_paths)
    if isinstance(paths, str):
        paths = [paths]
    # erase old file
    with open(task_name+".txt", "w") as f:
        pass

    # count number of images in txt
    counter = 0
    
    if return_stats:
        stats_classes = np.zeros(len(all_names), dtype=np.int32)
        has_class_oi = False

    # NOTE: temp fix for coco
    is_coco = ".txt" in imgs_paths[0]
   
    if is_coco:
        paths = open(imgs_paths[0], "r")
        path_coco = str(Path(imgs_paths[0]).parent)
        str_ids_to_include = get_class_ids(all_names, task_names)

        for img_path in paths:
            absoulte_path = path_coco + img_path[1:]
            str_label_file = str(absoulte_path).replace("images", "labels").replace(".jpg", ".txt").replace("\n", "")

            if os.path.isfile(str_label_file):

                if return_stats:
                    classes_counters = np.zeros(len(all_names), dtype=np.int32)

                with open(str_label_file) as f:

                    while bbox := f.readline():
                        class_id_obj = bbox.split(" ")[0]
                        class_id_obj_int = int(class_id_obj)

                        if return_stats:
                            classes_counters[class_id_obj_int] += 1
                        
                        if other_criteria is None:
                            if class_id_obj in str_ids_to_include:

                                counter +=1 

                                if return_stats:
                                    if has_class_oi == False:
                                        with open(task_name+".txt", "a") as txt:
                                            txt.write(img_path)   
                                    has_class_oi = True
                                else:
                                    with open(task_name+".txt", "a") as txt:
                                        txt.write(img_path)
                                    break
                        else:
                            pass

                    if return_stats:
                        if has_class_oi:
                            stats_classes += classes_counters
                            
                        has_class_oi = False

        paths.close()
        stats = None

        if return_stats:
            stats_classes_task = np.zeros(len(all_names), dtype=np.int32)
            for class_id in str_ids_to_include:
                stats_classes_task[int(class_id)] = stats_classes[int(class_id)]

            stats = (stats_classes, stats_classes_task)
        return str(Path().resolve())+"/"+task_name+".txt", stats


    for imgs_path in paths:
        imgs_path = Path(imgs_path)

        # get class ids to consider for the task (str format)
        str_ids_to_include = get_class_ids(all_names, task_names)

        # get labels dir path
        labels_path = Path(str(imgs_path).replace("images", "labels"))

        # NOTE: delete sorted
        for i, label_file_path in enumerate(sorted(labels_path.glob("*.txt"))):

            str_label_file = str(label_file_path)  # e.g. .../file_name.txt
            name_label_file = label_file_path.name.split(".")[0]  # e.g. file_name
            class_ids_seen = []

            with open(str_label_file) as f:

                if return_stats:
                    classes_counters = np.zeros(len(all_names), dtype=np.int32)

                while bbox := f.readline():
                    class_id_obj = bbox.split(" ")[0]
                    class_id_obj_int = int(class_id_obj)
                    class_ids_seen.append(class_id_obj)

                    if return_stats:
                        classes_counters[class_id_obj_int] += 1
                    
                    if other_criteria is None:
                        if class_id_obj in str_ids_to_include:

                            # print(f"{name_label_file}: {class_id_obj}, FOUND")
                            counter +=1 
                            if return_stats:
                                if has_class_oi == False:
                                    with open(task_name+".txt", "a") as txt:
                                        txt.write(name_label_file+"\n")
                                has_class_oi = True
                            else:
                                with open(task_name+".txt", "a") as txt:
                                        txt.write(name_label_file+"\n")
                                break
                    else:
                        pass

                if return_stats:
                    if has_class_oi:
                        stats_classes += classes_counters
                            
                has_class_oi = False

    stats = None
    if return_stats:
        stats_classes_task = np.zeros(len(all_names), dtype=np.int32)
        for class_id in str_ids_to_include:
            stats_classes_task[int(class_id)] = stats_classes[int(class_id)]

        stats = (stats_classes, stats_classes_task)

    return str(Path().resolve())+"/"+task_name+".txt", stats

def create_loaders(train_m_cfg: Dict, val_m_cfg : Dict, data_cfg: Dict, batch_size: int, filters=None):
    """Creates DataLoaders for dataset specified in the configuration file.
    Refer to ... for how to select the proper configuration.

    Arguments
    ---------
    m_cfg : Dict
        Contains information about the training process (e.g., data augmentation).
    data_cfg : Dict
        Contains details about the data configurations (e.g., image size, etc.).
    batch_size : int
        Batch size for the training process.
    filters: list[str]
        List with two items: path to filter for training and path to filter for validation

    """

    mode = "train"
    filter_file = None if filters is None else filters[0]

    train_set = build_yolo_dataset(
        train_m_cfg,
        data_cfg["train"],
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val",
        filter_file=filter_file
    )

    print(f"Number of images for training: {len(train_set)}, {len(train_m_cfg.classes)} classes")

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
        collate_fn=getattr(train_set, "collate_fn", None),
    )

    mode = "val"
    filter_file = None if filters[1] is None else filters[1]

    val_set = build_yolo_dataset(
        val_m_cfg,
        data_cfg["val"],
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val",
        filter_file=filter_file
    )
    
    n_val_classes = len(data_cfg["names"]) if val_m_cfg.classes is None else len(val_m_cfg.classes)
    print(f"Number of images for validation: {len(val_set)}, {n_val_classes} classes")

    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=False,
        collate_fn=getattr(val_set, "collate_fn", None),
    )

    return train_loader, val_loader


def path_to_string(path : Union[list[Path], Path])->list[str]:
    """Convert Path/list[Path] to list[str]"""

    if isinstance(path, Path):
        path = [path]
    return [str(p) for p in path]


def get_dataloaders_task(all_class_names : list[str], task_class_names : list[str], m_cfg, data_cfg, hparams, old_class_names : list[str] = None, return_stats=False):

    """Create loaders for the current task. Download data if needed."""

    data_cfg["train"] = path_to_string(data_cfg["train"])
    data_cfg["val"] = path_to_string(data_cfg["val"])

    # dowload images if needed
    if "download" in data_cfg and not os.path.exists(data_cfg["path"]):
        # download data if it's not there
        exec(data_cfg["download"])

    # create filters for current task (filter images for current task):
    # 1) create files to filter images
    # 2) create list of class-ids. Used to filter labels (done by ultralytics function)
    task_train_filter, stats = create_task_txt(data_cfg["train"], all_class_names, task_class_names, "task_train", return_stats=return_stats)

    # for validation, consider either all classes or the ones seen up to now
    task_val_filter = None
    stats_val = None
    if isinstance(old_class_names, list):
        task_val_filter, stats_val = create_task_txt(data_cfg["val"], all_class_names, old_class_names + task_class_names, "task_val", return_stats=return_stats)

    filter_files = [task_train_filter, task_val_filter]


    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution
    val_m_cfg = deepcopy(m_cfg)
    
    # filter labels during training
    m_cfg.classes = [int(class_id) for class_id in get_class_ids(all_class_names, task_class_names)]
    # for validation consider either all classes or the ones seen up to now
    if isinstance(old_class_names, list):
        classes_seen = old_class_names + task_class_names
        val_m_cfg.classes = [int(class_id) for class_id in get_class_ids(all_class_names, classes_seen)]
    else:
        val_m_cfg.classes = None
        

    # create current task loaders
    train_loader, val_loader = create_loaders(m_cfg, val_m_cfg, data_cfg, hparams.batch_size, filters=filter_files)

    return train_loader, val_loader, m_cfg, val_m_cfg, (stats, stats_val)



class TasksGenerator:

    def __init__(self, m_cfg, data_cfg, hparams, clpipeline : str, save_stats, skip=False):
        
        self.m_cfg = m_cfg
        self.data_cfg = data_cfg
        self.hparams = hparams

        # possible strings: npm, with m<n and n>1
        self.clpipeline = clpipeline
        # save csv with statistics for each task or not
        self.save_stats = save_stats

        # all names
        self.all_class_names = [self.data_cfg["names"][id] for id in sorted(list(self.data_cfg["names"].keys()))]
        # upper bound iterations (at least one)
        self.high = 1  # max number iterations = # tasks
        self.counter = 0
        self.class_increment = 0  # number of classes per tasks
        # num classes for first task
        self.init_num_classes = None

        # init high, class_increment and init_num_classes based on cl pipeline
        self._parse_clpipeline(clpipeline=clpipeline)
        self.skip = skip



    def _parse_clpipeline(self, clpipeline : str):

        list_after_split = clpipeline.split("p")

        # no p in the pipeline -> joint training
        if len(list_after_split) == 1:
            self.init_num_classes = len(self.all_class_names)
        
        else:
            self.init_num_classes = int(list_after_split[0])
            self.class_increment = int(list_after_split[1])
            self.high += (len(self.all_class_names) - self.init_num_classes)/self.class_increment

    def __iter__(self):
        return self

    def __next__(self):

        if self.skip and self.counter==0:
            self.counter += 1
            epochs = self.hparams.epochs
            return None, None, {"epochs": epochs}

        if self.counter < self.high:

            task_names = None
            # list with classes all task seen before the current one
            old_task_names = list()

            # define class names for current task and the class names seen before the current task
            if self.counter == 0:
                task_names = self.all_class_names[:self.init_num_classes]
                epochs = self.hparams.epochs

            else:
                jump = (self.counter-1) * self.class_increment
                task_names = self.all_class_names[ self.init_num_classes + jump:self.init_num_classes + jump + self.class_increment ] # e.g. 15p1 [15:16], [16:17], ...
                old_task_names += self.all_class_names[:self.init_num_classes + jump]
                epochs = self.hparams.epochs_per_task

            self.counter += 1

            # get dataloaders
            train_loader, val_loader, train_cfg, val_cfg, stats = get_dataloaders_task(self.all_class_names, task_names, self.m_cfg, self.data_cfg, self.hparams, old_task_names, return_stats=self.save_stats)
            
            if self.save_stats:
                self._save_stats(stats)

            old_classes = [int(idx) for idx in get_class_ids(self.all_class_names, old_task_names)] if len(old_task_names)>0 else None

            other_info = {"tr_cfg" : train_cfg, "val_cfg" : val_cfg,
                           "nc_val" : self._get_nc_seen(),
                            "nc_tr" : len(task_names+old_task_names),
                            "epochs" : epochs,
                            "old_classes" : old_classes
                        }

            return train_loader, val_loader, other_info
            
        else:
            raise StopIteration
        
    def _get_nc_seen(self):

        return self.init_num_classes + (self.counter - 1) * self.class_increment
    
    def _save_stats(self, stats):

        tr_stats = stats[0]
        val_stats = stats[1]

        stats_pd = pd.DataFrame(tr_stats[0].reshape(1,-1))
        stats_pd = pd.concat((stats_pd, pd.DataFrame(tr_stats[1].reshape(1,-1))), ignore_index=True)
        stats_pd = pd.concat((stats_pd, pd.DataFrame(val_stats[0].reshape(1,-1))), ignore_index=True)
        stats_pd = pd.concat((stats_pd, pd.DataFrame(val_stats[1].reshape(1,-1))), ignore_index=True)

        stats_pd.to_csv(f"./stats_{self.counter-1}.csv")


class CumulativeGenerator:

    def __init__(self, m_cfg, data_cfg, hparams, clpipeline : str, save_stats):
        
        self.m_cfg = m_cfg
        self.data_cfg = data_cfg
        self.hparams = hparams

        # possible strings: npm, with m<n and n>1
        self.clpipeline = clpipeline
        # save csv with statistics for each task or not
        self.save_stats = save_stats

        # all names
        self.all_class_names = [self.data_cfg["names"][id] for id in sorted(list(self.data_cfg["names"].keys()))]
        # upper bound iterations (at least one)
        self.high = 1  # max number iterations = # tasks
        self.counter = 0
        self.class_increment = 0  # number of classes per tasks
        # num classes for first task
        self.init_num_classes = None

        # init high, class_increment and init_num_classes based on cl pipeline
        self._parse_clpipeline(clpipeline=clpipeline)


    def _parse_clpipeline(self, clpipeline : str):

        list_after_split = clpipeline.split("p")

        # no p in the pipeline -> joint training
        if len(list_after_split) == 1:
            self.init_num_classes = len(self.all_class_names)
        
        else:
            self.init_num_classes = int(list_after_split[0])
            self.class_increment = int(list_after_split[1])
            self.high += (len(self.all_class_names) - self.init_num_classes)/self.class_increment

    def __iter__(self):
        return self

    def __next__(self):

        if self.counter < self.high:

            task_names = None
            # list with classes all task seen before the current one
            old_task_names = list()

            # define class names for current task and the class names seen before the current task
            if self.counter == 0:
                task_names = self.all_class_names[:self.init_num_classes]
                epochs = self.hparams.epochs

            else:
                jump = (self.counter-1) * self.class_increment
                task_names = self.all_class_names[0:self.init_num_classes + jump + self.class_increment ] # e.g. 15p1 [15:16], [16:17], ...
                epochs = self.hparams.epochs_per_task

            self.counter += 1

            # get dataloaders
            train_loader, val_loader, train_cfg, val_cfg, stats = get_dataloaders_task(self.all_class_names, task_names, self.m_cfg, self.data_cfg, self.hparams, old_task_names, return_stats=self.save_stats)
            
            if self.save_stats:
                self._save_stats(stats)

            old_classes = [int(idx) for idx in get_class_ids(self.all_class_names, old_task_names)] if len(old_task_names)>0 else None

            other_info = {"tr_cfg" : train_cfg, "val_cfg" : val_cfg,
                           "nc_val" : self._get_nc_seen(),
                            "nc_tr" : len(task_names+old_task_names),
                            "epochs" : epochs,
                            "old_classes" : old_classes
                        }

            return train_loader, val_loader, other_info
            
        else:
            raise StopIteration
        
    def _get_nc_seen(self):

        return self.init_num_classes + (self.counter - 1) * self.class_increment
    
    def _save_stats(self, stats):

        tr_stats = stats[0]
        val_stats = stats[1]

        stats_pd = pd.DataFrame(tr_stats[0].reshape(1,-1))
        stats_pd = pd.concat((stats_pd, pd.DataFrame(tr_stats[1].reshape(1,-1))), ignore_index=True)
        stats_pd = pd.concat((stats_pd, pd.DataFrame(val_stats[0].reshape(1,-1))), ignore_index=True)
        stats_pd = pd.concat((stats_pd, pd.DataFrame(val_stats[1].reshape(1,-1))), ignore_index=True)

        stats_pd.to_csv(f"./stats_{self.counter-1}.csv")