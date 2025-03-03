# Teach YOLO to Remember: A Self-Distillation Approach for Continual Object Detection

## Table of contents

* [Introduction](#introduction)
* [Results](#results)
* [Setup](#setup)
* [Usage](#usage)
* [Citation](#citation)

## Introduction
This work aims to let the YOLOv8 object detector to learn incrementally new classes without forgetting, by use of self-distillation. We propose YOLO LwF and a second version with a reaply memory e.g. YOLO LwF + OCDM.

## Results


## Setup
In addition to PyTorch, install the extra requirements by following these steps:
1. Install ```micromind``` with ```pip install git+https://github.com/fpaissan/micromind```.
2. You find a ```extra_requirements.txt``` file. Please run  ```pip install -r extra_requirements.txt``` to install extra requirements e.g.  ```ultralytics```.


## Usage

1. In the ```recipes/clod``` directory you find the source code.
2. Run ```lwfocdm_train.py cfg/yolov8coco_yololwf.py``` to run YOLO LwF + OCDM (the best approach).
3. In the directory ```recipes/clod/results``` you will find a set of ```.csv``` files, one per task, with the logs. e.g. ```mAPs_task_0.csv``` with the results for the first task.

To run an experiment on VOC, use instead the configuration file ```cfg/yolov8yololwf.py```.

**NOTE**: to change the CL scenario (e.g. 40p40, 40p10, 15p1, etc.) modify the configuration file (either the one for COCO ```cfg/yolov8coco_yololwf.py``` or the VOC one  ```cfg/yolov8yololwf.py```) as follows:

```Python
# CLOD
exp = "40p10"  # for COCO40p10
```

```Python
# CLOD
exp = "40p40"  # for COCO40p40
```


To test the other methods:
* ```lwf_train_yolov8.py cfg/yolov8yololwf.py``` for **YOLO LwF** on VOC. for coco use ```lwf_train_yolov8.py cfg/yolov8coco_yololwf.py```
* ```lwf_train_yolov8.py cfg/yolov8lwf.py``` for **LwF** on VOC. for coco use ```lwf_train_yolov8.py cfg/yolov8coco_lwf.py```
* ```lwfocdm_train.py cfg/yolov8lwf.py``` for **LwF+OCDM** on VOC. for coco use ```lwf_train_yolov8.py cfg/yolov8coco_lwf.py```
* ```erd_train.py cfg/yolov8erd.py``` for **ERD** on VOC. for coco use ```erd_train.py cfg/yolov8coco_erd.py```
* ```erd_ocdm_train.py cfg/yolov8erd.py``` for **ERD+OCDM** on VOC. for coco use ```erd_ocdm_train.py cfg/yolov8coco_erd.py```
* ```RCLPOD_train_yolov8.py cfg/yolov8erd.py``` for **RCLPOD** on VOC. for coco use ```RCLPOD_train_yolov8.py cfg/yolov8coco_erd.py```
* ```naive_train_yolov8.py cfg/yolov8erd.py``` for **Fine-Tuning** on VOC. for coco use ```naive_train_yolov8.py cfg/yolov8coco_erd.py```

## Citation

If you find this project useful in your research, please add a star and cite us 😊 

```BibTeX
@misc{,
  title={Teach YOLO to Remember: A Self-Distillation Approach for Continual Object Detection},
  author={},
  booktitle={},
  year={2025},
}
```

## Thanks

https://github.com/ultralytics/ultralytics

https://github.com/micromind-toolkit/micromind
