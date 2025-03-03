# Code from Ultralytics
# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from copy import deepcopy
from pathlib import Path
from itertools import repeat
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics.utils import LOGGER, TQDM, LOCAL_RANK, NUM_THREADS, is_dir_writeable
from ultralytics.utils.ops import resample_segments
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.data.utils import LOGGER
from ultralytics.data.utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label
from multiprocessing.pool import ThreadPool
import sys
from collections import Counter
# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

class ReplayDataloader:

    """
    Custom dataloader for Replay: given the dataloader for the current task and a ReplayMemory (torch Dataset), next() returns 
    teh concatenation of: a batch from the dataloader for the current task and a random batch from the replay memory.
    """

    def __init__(self, task_tr_loader, replay_memory, lwf=False):

        self.task_tr_loader = task_tr_loader
        #self.dataset = self.task_tr_loader.dataset
        self.replay_memory = replay_memory
        #self.batch_sampler = self.task_tr_loader.batch_sampler

        # get batch size for current task loader
        self.batch_size_per_dataset = self.task_tr_loader.batch_size

        self.lwf=lwf

    def __len__(self):
        return len(self.task_tr_loader)

    def __iter__(self):
        
        self.task_tr_iter = iter(self.task_tr_loader)
        return self
    
    def __next__(self):

        try:
            task_batch = next(self.task_tr_iter)
        except StopIteration:
            raise StopIteration
        
        # get batch from memory
        b = task_batch["img"].shape[0]  # batch_size for current task batch
        replay_batch = self.replay_memory.get_batch(b)

        new_batch = {}
        
        if self.lwf:
            new_batch["num_labels"] = task_batch["cls"].shape[0]

        # concatenate the two batches
        for key in task_batch.keys():

            # for 'batch_idx', update the second batch
            if key == "batch_idx":
                replay_batch[key] += b
                new_batch[key] = torch.cat([task_batch[key], replay_batch[key]])

            elif isinstance(task_batch[key], torch.Tensor):
                new_batch[key] = torch.cat([task_batch[key], replay_batch[key]])

            elif isinstance(task_batch[key], tuple) and isinstance(replay_batch[key], tuple):
                new_batch[key] = tuple(list(task_batch[key]) + list(replay_batch[key]))

            elif isinstance(replay_batch[key], tuple):
                new_batch[key] = task_batch[key] + list(replay_batch[key])

            else: 
                new_batch[key] = task_batch[key] + replay_batch[key]

        if "task_id" in replay_batch.keys():
            task_id = torch.zeros(b+len(replay_batch["task_id"])).int()
            for i,t in enumerate(replay_batch["task_id"]):
                task_id[i+b] = t

            new_batch["task_id"] = task_id

        return new_batch


class ReplayMemory(Dataset):
    """
    Replay memory for YOLOv8: each time a new task is encountered, a subset of samples from the new task are added to this buffer. Since
    the buffer has a fixed capacity, some samples of the older tasks are removed once a new task is encountered. The implementation aims
    to keep  equal partitions among the different tasks encountered.

    For each task a dataset is kept in memory. For each task a list (numpy array) with the indices associated to the corresponding dataset are saved.
    Once some samples should be removed for a specific task, just the indices are removed, namely the dataset remains the same, but the available
    indices, saved in this class, are reduced.
    """

    def __init__(self, capacity, augment=False):

        self.augment = augment
        self.capacity = capacity

        self.ntasks = 0
        # list of datasets: one per task
        self.task_replay_memories = []
        # list with # samples per task
        self.n_available_images_per_task = []

        # list of numpy array: each array stores the indices of the images still present in the replay buffer of a specific task.
        # each list will be changed once a new task is encountered
        self.available_images_per_task = []

        self.nimgs_per_task = None

    def __len__(self):
        return self.capacity
    
    def __getitem__(self, index):

        c = 0
        task_idx = None
        idx = None
        
        # get task index and index for that sample in the task dataset
        for i, n in enumerate(self.n_available_images_per_task):

            if index < (c + n):
                task_idx = i
                idx = index - c
                break

            c += n

        sample_idx = self.available_images_per_task[task_idx][idx]


        to_return = self.task_replay_memories[task_idx][sample_idx]
        to_return["task_idx"] = task_idx
        to_return["idx"] = sample_idx

        return to_return
    

 
    def get_item(self):
        """Get a random sample"""

        # select randomly task dataset
        task_idx = np.random.randint(0, self.ntasks)  # random number in [0, ntasks - 1]

        # select one index from the ones available
        sample_idx = self.available_images_per_task[task_idx][np.random.randint(0, self.n_available_images_per_task[task_idx])]

        return self.task_replay_memories[task_idx][sample_idx]
    
    def get_batch(self, batch_dim):
        """Get a random batch of samples
        
        :param batch_dim: number of samples in the batch

        return: dict, for each key a batch of items e.g. for key "img" a batch of tensors (shape: [batch_dim x 3 x 640 x 640]).
                see ultralytics documentation for the format
        """

        list_samples = [self.get_item() for i in range(batch_dim)]

        # concat by key
        batch = self.collate_fn(list_samples)

        return batch

    def update(self, dataset):
        """
        Update replay memory:
            1) add images for the last task seen
            2) remove some images from old tasks to get equal partitions
    
        :param dataset: YOLODataset instance for the last task
        """

        # compute number of images per task to store, given the new task
        self.nimgs_per_task = math.floor(self.capacity/(self.ntasks + 1))
        # compute number of images to remove for each old task
        nimgs_per_task_to_remove = math.floor(self.nimgs_per_task/self.ntasks) if self.ntasks > 0 else 0
        # remove images old tasks
        if self.ntasks > 0:
            self.remove_imgs(nimgs_per_task_to_remove)

        idx = self.ntasks
        self.ntasks += 1
        """ Select images to add from new task """
        # shuffle possible indices of images in the dataset of the current task
        possible_indices = np.arange(len(dataset))
        np.random.shuffle(possible_indices)

        # if number of images > dataset, add multiple times samples
        if self.nimgs_per_task > len(dataset):
            num_repeat = self.nimgs_per_task // len(dataset)
            indices = np.tile(possible_indices, num_repeat)
            indices = np.concatenate((indices, possible_indices[:(self.nimgs_per_task - num_repeat * len(dataset))]))
        else:
            indices = possible_indices[:self.nimgs_per_task]
    
        # create a ReplayMemoryTask with self.nimgs_per_task images
        self.task_replay_memories.append(ReplayMemoryTask(idx, indices, dataset, self.augment))
        # save available indices: at the beggining all of them. When a new task comes, the number of indices will be reduced
        self.available_images_per_task.append(np.arange(len(self.task_replay_memories[-1])))
        self.n_available_images_per_task.append(len(self.task_replay_memories[-1]))

    def remove_imgs(self, nimgs_per_task_to_remove):

        """
        Remove images of old tasks

        :param nimgs_per_task_to_remove: number of images to remove for each old task
        """

        imgs_removed = 0

        for i in range(self.ntasks):

            #remove imgs for one old task
            # 1) shuffle indices to array of indices (to select a subset of indices to remove)
            possible_indices = np.arange(self.n_available_images_per_task[i])
            np.random.shuffle(possible_indices)

            # 2) get a subset of indices
            to_delete = possible_indices[:nimgs_per_task_to_remove]

            # 3) delete indices
            self.available_images_per_task[i] = np.delete(self.available_images_per_task[i], to_delete)

            # update counter availbale images per task
            self.n_available_images_per_task[i] -= nimgs_per_task_to_remove
            # count tot images removed
            imgs_removed += nimgs_per_task_to_remove
        
        # if it is not possible to mantain equal partitions, remove extra images
        if imgs_removed < self.nimgs_per_task:

            # create pointers to tasks
            idx_tasks = np.arange(self.ntasks)
            # shuffle the pointers
            np.random.shuffle(idx_tasks)

            i = 0
            while imgs_removed < self.nimgs_per_task:
                # until enough images are removed, select next pointer to task
                idx = idx_tasks[i]
                # remove one image from the pointed task
                to_delete = np.random.randint(0, self.n_available_images_per_task[idx])
                self.available_images_per_task[idx] = np.delete(self.available_images_per_task[idx], to_delete)
                self.n_available_images_per_task[idx] -= 1
                imgs_removed += 1
                i += 1

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
    
    @staticmethod
    def collate_fnv2(batch):
        """Collates data samples into batches."""
        new_batch = {}

        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]

            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            if k in ["task_idx", "idx"]:
                value = torch.tensor(value)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    def save_num_duplicate(self):
        imgs = []
        task = len(self.task_replay_memories)
        for i, mem in enumerate(self.task_replay_memories):
            for j in self.available_images_per_task[i]:
                imgs.append(mem.im_files[j])
        
        dict_counts = dict(Counter(imgs))

        num_duplicates = 0
        for k in dict_counts:
            num_duplicates += 1 if dict_counts[k]>1 else 0

        np.save(f"num_dup_task{task}.npy", np.array([num_duplicates]))
        
        

class ReplayMemoryTask(Dataset):
    """
    
    """

    def __init__(self, idx, indices, dataset, augment=False, nc=80):

        self.hyp = dataset.hyp
        self.imgsz = dataset.hyp.imgsz
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.prefix = ""
        self.rect = False
        self.data = dataset.data

        self.augment = augment

        self.idx = idx
        self.nc = nc
        self.ni = len(indices)

        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni

        # list of paths: create a subset based on the current task dataset and the indices for the images to keep
        self.im_files = [deepcopy(dataset.im_files[i]) for i in indices]
        self.labels = [deepcopy(dataset.labels[i]) for i in indices]

        # get labels: create cache for labels
        #self.labels = self.get_labels()
        # update labels with classes for that task
        #self.update_labels(self.hyp.classes)

        # create transform
        self.transforms = self.build_transforms(dataset.hyp)

        # Buffer for mosaic augmentation: up to now juts 8 images to use to create mosaic
        self.buffer = []  
        self.max_buffer_length = 8 if self.augment else 0


    def __len__(self):
        return len(self.labels)


    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment else 0.0
            hyp.mixup = hyp.mixup if self.augment else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
    

    """Functions to get an item"""


    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))
    

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)


    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = None, self.im_files[i], None
        if im is None:  # not cached in RAM
            if fn:  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]   


    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")















































