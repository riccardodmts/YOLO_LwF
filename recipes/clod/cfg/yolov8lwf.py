

# Data configuration and optimization
batch_size = 8
data_cfg = "cfg/data/VOC.yaml"
data_dir = "datasets/VOC"
epochs = 100
num_classes = 20
momentum = 0.937
warmup_epochs = 3
warmup_momentum = 0.8
warmup_bias_lr = 0.1
lr0 = 0.01
lrf = 0.01  # the actual lrf = lr0 * lrf -> 10^-4

# Model configuration
model_size = "n" # n, s, m, l, x
input_shape = [3, 640, 640]
heads = [True, True, True]

# just returning 2 intermediate layers (last is default)
return_layers = [6, 8]

# CLOD
exp = "19p1"
epochs_per_task = 100  # number epochs used for all tasks except the first one
save_stats = True
use_tensorboard = False

"""REPLAY/OCDM"""
replay_mem_size = 800

"""Psudeo-label"""
inference_ths = 0.5  # threshold for pseudolabel

"""OCDM"""
ocdm_ths = None  # ocdm threshold
save_num_dup = True  # save num of duplicates
batch_size_ocdm = 1000

"""LwF"""
lwf = (1.0, 1.0)  # lwf params. if len(lwf)>2 -> use LwFV2 (DERL2: (100000,10000000)), DER (4000, 1000, 0), 125,300, 10-2->10-3,

"""Distill loss for RCLPOD method: backbone and neck"""
feats_distill = (1.0, 1.0) 



# Placeholder for inference
ckpt_pretrained = ""
output_dir = "detection_output"
coco_names = "cfg/data/coco.names"