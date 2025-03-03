# Code from Ultralytics
# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ultralytics.utils import LOGGER
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.data.utils import LOGGER
from ultralytics.data.utils import LOGGER
from collections import Counter


# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

def count_labels(data: list, nc=None, one_hot=False):
    """
    Compute absolute freq for each class. NOTE: for Object Detection the absolute frequency is related to the number of
    images that has a specific class, namely if N object of the same class appear in one image, they count as 1.
    :param data:    list of np array. Each array can be either a one-hot or a 1d array with the
                    class indices of the object present (Object Detection)
    :param nc:      # classes involved
    :param one_hot: if the labels of data are one-hot
    :return: np array with nc values, the absolute frequencies
    """

    # compute nc if not provided
    if nc is None:
        max_id = 0
        for sample in data:
            max_id_sample = np.max(sample) if not one_hot else sample.shape[0] - 1
            max_id = max_id_sample if max_id_sample > max_id else max_id

        nc = max_id + 1

    counters = np.zeros(nc, dtype=np.int32)

    for sample in data:

        if one_hot:
            counters[np.where(sample == 1)] += 1
        else:
            counters[np.unique(sample)] += 1

    return counters


def count_labelsv2(data: list, nc, ths=None):
    """
    Compute absolute freq for each class
    :param data:    list of np array. Each array can be either a one-hot or a 1d array with the
                    class indices of the object present (Object Detection)
    :param nc:      # classes involved
    :param ths:
    :return: np array with nc values, the absolute frequencies
    """

    counters = np.zeros(nc, dtype=np.int32)

    for sample in data:

        values, counts = np.unique(sample, return_counts=True)

        if ths is not None:
            counts[counts > ths] = ths

        counters[values] += counts

    return counters

def get_labels_distribution(data : list, nc=None, one_hot=False, rho=1.0, ths=None):

    abs_freq = count_labels(data, nc, one_hot) if ths is None else count_labelsv2(data, nc, ths)
    pow_abs_freq = abs_freq ** rho
    denom = np.sum(pow_abs_freq)
    return pow_abs_freq/denom


def cross_entropy(p, q):
    """

    :param p: target distribution
    :param q:
    :return: cross-entropy, H(p,q)
    """
    return np.sum(- p * np.log(q))

def cross_entropy_torch(p, q):
    """

    :param p: target distribution
    :param q:
    :return: cross-entropy, H(p,q)
    """
    return torch.sum(- p * torch.log(q), dim=1)


def memory_update_indices(data, nc, one_hot, dist_fn, num_iter, target_distr=None, count_fn=None, *fn_params):

    """

    :param data: list of numpy array. each array is associated to one image and it represents classes present in the image. e.g. [1, 4, 1, 5]
    :nc: number of classes
    :one_hot: boolean. If True, the arrays has just ones or zeros and dimension (nc,)
    :dist_fn: ditance function used to compare distribution. Used to select imgs to keep in memory
    :num_iter: number of images to remove from data
    :target_distr: target distribution to reach by removing images e.g. uniform
    :count_fn: optional, custom function to compute absolute frequencies
    :fn_params: args for count_fn

    :return: list of indices for the images to be removed
    """

    if target_distr is None:
        target_distr = np.ones(nc, dtype=np.float32) / nc

    idxs_to_remove = []

    for _ in range(num_iter):

        min_dist = float("inf")
        min_index = -1
        data_copy = []

        for i, item in enumerate(data):
            if i not in idxs_to_remove:
                data_copy.append(item)

        if count_fn is None:
            abs_freq = count_labels(data_copy, nc, one_hot)
        else:
            abs_freq = count_fn(data_copy, nc, *fn_params)

        for idx, sample in enumerate(data):

            if idx in idxs_to_remove:
                continue

            if one_hot:
                new_abs_freq = abs_freq - sample
            else: 
                if count_fn is None:
                    to_sub = np.zeros_like(abs_freq)
                    to_sub[np.unique(sample)] += 1
                    new_abs_freq = abs_freq - to_sub
                else:
                    new_abs_freq = abs_freq - count_fn([sample], nc, *fn_params)
                          

            distr_dist = dist_fn(target_distr, new_abs_freq/np.sum(new_abs_freq))

            if distr_dist < min_dist:
                min_dist = distr_dist
                min_index = idx

        idxs_to_remove.append(min_index)

    return idxs_to_remove


def efficient_memory_update_indices(data, nc, dist_fn, num_iter, target_distr=None):

    """

    :param data: list of numpy array. each array is associated to one image and it represents classes present in the image. e.g. [1, 4, 1, 5]
    :nc: number of classes
    :one_hot: boolean. If True, the arrays has just ones or zeros and dimension (nc,)
    :dist_fn: ditance function used to compare distribution. Used to select imgs to keep in memory
    :num_iter: number of images to remove from data
    :target_distr: target distribution to reach by removing images e.g. uniform
    :count_fn: optional, custom function to compute absolute frequencies
    :fn_params: args for count_fn

    :return: list of indices for the images to be removed
    """

    if target_distr is None:
        target_distr = torch.ones(len(data), nc).float() / nc

    idxs_to_remove = []

    data_copy = []
    for i, sample in enumerate(data):
        labels_sample = np.zeros(nc)
        labels_sample[np.unique(sample)] += 1
        data_copy.append(labels_sample)

    matrix_labels_samples = np.asarray(data_copy)
    matrix_labels_samples = torch.from_numpy(matrix_labels_samples)

    abs_freq = torch.sum(matrix_labels_samples, axis=0)

    for i in range(num_iter):

        n = matrix_labels_samples.shape[0]
        abs_freq_matrix = abs_freq.repeat(n, 1)

        diff_matrix = abs_freq_matrix - matrix_labels_samples
        normalize = torch.sum(diff_matrix, dim=1).reshape(-1,1)

        q = diff_matrix/normalize

        scores = dist_fn(target_distr, q)
        scores[idxs_to_remove] = float("inf")


        index = torch.argmin(scores).item()
        abs_freq = diff_matrix[index].clone()

        idxs_to_remove.append(index)

    return idxs_to_remove



class OCDM(Dataset):
    """Implementation of Optimizing Class Distribution in Memory (OCDM) for YOLOv8"""

    def __init__(self, capacity, first_dataset, nc, max_nc, results_dir, batch_size = 0, ths=None, count_dup=False):

        self.capacity = capacity
        self.ntasks = 0
        self.batch_size = batch_size # if batch_size = -1, update memory in one shot
        self.nimages = 0
        self.nc = nc
        self.max_nc = max_nc
        self.results_dir = results_dir
        self.ntasks = 0
        self.ths = ths

        self.augment = False
        self.hyp = first_dataset.hyp
        self.imgsz = first_dataset.hyp.imgsz
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.prefix = ""
        self.rect = False
        self.data = first_dataset.data
        self.count_dup = count_dup

        self.header = [f"class_{id}" for id in range(max_nc)] + ["imgs_added"] + ["nc"]
        self.header += ["count_dup"] if self.count_dup else []

        self.im_files = []
        self.labels = []
        self.to_tasks = []

        self.transforms = self.build_transforms(first_dataset.hyp)
        self.update_memory(first_dataset, self.nc)


    def update_memory(self, dataset, nc=None):

        self.ntasks += 1

        if nc is not None:
            self.nc = nc

        batch_size = self.batch_size if self.batch_size > 0 else len(dataset)

        # get labels current memory
        labels_mem = []
        for i in range(self.nimages):
            labels_mem.append(deepcopy(self.labels[i]["cls"]).reshape(-1,).astype(int))

        # map (dataset, idx) where dataset identifies the dataset from which the sample come from e.g. either memory or dataset last task, idx is the index in the original dataset
        map_list = [("mem", idx) for idx in range(self.nimages)]

        num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)

        for i in range(num_batches):

            # start index for batch from dataset
            start_index = batch_size * i
            # compute batch size (always = batch_size except for the last one)
            current_batch_size = batch_size if batch_size + start_index <= len(dataset) else len(dataset) - start_index
           
           # get labels for batch from dataset
            labels_batch = self.get_cls(dataset, start_index, current_batch_size)

            if (self.nimages + current_batch_size) <= self.capacity:
                # if memory not full and batch size not enough to reach max capacity, add all samples
                labels_mem += labels_batch
                # update map (mark them as samples from dataset)
                map_list += [("data", idx + start_index) for idx in range(current_batch_size)]
                self.nimages += current_batch_size
            else:
                # get indices to remove
                indices_to_remove = self.get_indices_to_remove(labels_mem + labels_batch, self.nc,
                                                                self.nimages + current_batch_size - self.capacity)
                self.nimages = self.capacity

                # --- UPDATE TEMP MEMORY ---
                tot_imgs = len(labels_mem) + len(labels_batch)
                
                new_mem = []
                new_map_list = []
                imgs_added = 0

                for j in range(tot_imgs):
                    if j not in indices_to_remove:
                        if j < len(labels_mem):
                            # if sample from memory copy from memory
                            new_mem.append(deepcopy(labels_mem[j]))
                            new_map_list.append((map_list[j][0], map_list[j][1]))
                        else:
                            imgs_added += 1
                            # copy from batch and add to memory
                            new_mem.append(deepcopy(labels_batch[j - len(labels_mem)]))
                            new_map_list.append(("data", start_index + j - len(labels_mem)))  # index in dataset: start index + idx, idx = [0, batch_size]

                labels_mem = new_mem
                map_list = new_map_list

                print(f"Batch {i+1} done, images added from the batch: {imgs_added}")

        labels = []
        im_files = []
        to_tasks = []

        imgs_added = 0

        # update real memory
        for loc, idx in map_list:

            if loc == "mem":
                labels.append(deepcopy(self.labels[idx]))
                im_files.append(deepcopy(self.im_files[idx]))
                to_tasks.append(self.to_tasks[idx])

            else:
                imgs_added += 1
                labels.append(deepcopy(dataset.labels[idx]))
                im_files.append(deepcopy(dataset.im_files[idx]))
                to_tasks.append(self.ntasks-1)

        self.labels = labels
        self.im_files = im_files
        self.to_tasks = to_tasks

        self.save_stats(imgs_added)

    def get_indices_to_remove(self, data, nc, num_iter):

        if self.ths is None:

            #return memory_update_indices(data, nc, False, cross_entropy, num_iter)
            return efficient_memory_update_indices(data, nc, cross_entropy_torch, num_iter)
            
        else:
             return memory_update_indices(data, nc, False, cross_entropy, num_iter, None, count_labelsv2, self.ths)


    def get_cls(self, dataset, start_index, batch_size):
        """Get batch (list) of np arrays with classes present in each image of the batch"""

        list_cls = []

        for i in range(batch_size):
            list_cls.append(deepcopy(dataset.labels[start_index + i]["cls"].reshape(-1,).astype(int)))

        return list_cls
    
    def count_duplicates(self):

        dict_counts = dict(Counter(self.im_files))

        num_duplicates = 0
        for k in dict_counts:
            num_duplicates += 1 if dict_counts[k]>1 else 0

        return num_duplicates


    def get_stats(self):

        labels_mem = []
        for i in range(self.nimages):
            labels_mem.append(deepcopy(self.labels[i]["cls"]).reshape(-1,).astype(int))

        return get_labels_distribution(labels_mem, self.nc, ths=self.ths)
    
    def save_stats(self, imgs_added):
        
        add = 1 if self.count_dup else 0
        to_save = np.zeros(self.max_nc+2+add, dtype=np.float32)
        to_save[:self.nc] = self.get_stats()
        to_save[-(2+add)] = imgs_added
        to_save[-(1+add)] = self.nc
        if add:
            to_save[-1] = self.count_duplicates()

        if self.ntasks == 1:
            pd.DataFrame(to_save.reshape(1,-1), columns=self.header).to_csv(self.results_dir+f"/ocdm.csv", sep="\t", header=None, index=False)
        else:
            pd.DataFrame(to_save.reshape(1,-1), columns=self.header).to_csv(self.results_dir+f"/ocdm.csv", sep="\t", header=None, index=False, mode="a")

        
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
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))
    
    def get_batch(self, batch_dim):
        """Get a random batch of samples
        
        :param batch_dim: number of samples in the batch

        return: dict, for each key a batch of items e.g. for key "img" a batch of tensors (shape: [batch_dim x 3 x 640 x 640]).
                see ultralytics documentation for the format
        """

        indices = np.arange(self.nimages)
        np.random.shuffle(indices)

        list_samples = [self[i] for i in indices[:batch_dim]]

        # concat by key
        batch = self.collate_fn(list_samples)

        return batch
    
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["task_id"] = self.to_tasks[index]#torch.Tensor([self.to_tasks[index]]).int()
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
    







class OCDMLwF(Dataset):
    """Implementation of Optimizing Class Distribution in Memory (OCDM) for YOLOv8. Version for LwF"""

    def __init__(self, capacity, first_dataset, nc, max_nc, results_dir, batch_size = 0, ths=None, count_dup=False, trainer=None):

        self.capacity = capacity
        self.ntasks = 0
        self.batch_size = batch_size # if batch_size = -1, update memory in one shot
        self.nimages = 0
        self.nc = nc
        self.max_nc = max_nc
        self.results_dir = results_dir
        self.ntasks = 0
        self.ths = ths

        self.augment = False
        self.hyp = first_dataset.hyp
        self.imgsz = first_dataset.hyp.imgsz
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.prefix = ""
        self.rect = False
        self.data = first_dataset.data
        self.count_dup = count_dup

        self.header = [f"class_{id}" for id in range(max_nc)] + ["imgs_added"] + ["nc"]
        self.header += ["count_dup"] if self.count_dup else []

        self.im_files = []
        self.labels = []
        self.to_tasks = []

        self.transforms = self.build_transforms(first_dataset.hyp)
        self.update_memory(first_dataset, self.nc)

        self.trainer = trainer


    def update_memory(self, dataset, nc=None, trainer=None):
        self.trainer = trainer

        self.ntasks += 1

        if nc is not None:
            self.nc = nc

        batch_size = self.batch_size if self.batch_size > 0 else len(dataset)

        # update labels with current model (create just a copy. we don't add actually labels since labels are used just for ocdm)
        if self.nimages > 0:
            data_mem = deepcopy(self)
            data_mem = self.trainer.add_labels_for_ocdm(data_mem)

            # add labels also to current dataset
            dataset = self.trainer.add_labels_for_ocdm(dataset)
        else:
            # if end first task, don't add any labels
            data_mem = self


        # get labels current memory
        labels_mem = []
        for i in range(self.nimages):
            labels_mem.append(deepcopy(data_mem.labels[i]["cls"]).reshape(-1,).astype(int))

        

        # map (dataset, idx) where dataset identifies the dataset from which the sample come from e.g. either memory or dataset last task, idx is the index in the original dataset
        map_list = [("mem", idx) for idx in range(self.nimages)]

        num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)

        for i in range(num_batches):

            # start index for batch from dataset
            start_index = batch_size * i
            # compute batch size (always = batch_size except for the last one)
            current_batch_size = batch_size if batch_size + start_index <= len(dataset) else len(dataset) - start_index
           
           # get labels for batch from dataset
            labels_batch = self.get_cls(dataset, start_index, current_batch_size)

            if (self.nimages + current_batch_size) <= self.capacity:
                # if memory not full and batch size not enough to reach max capacity, add all samples
                labels_mem += labels_batch
                # update map (mark them as samples from dataset)
                map_list += [("data", idx + start_index) for idx in range(current_batch_size)]
                self.nimages += current_batch_size
            else:
                # get indices to remove
                indices_to_remove = self.get_indices_to_remove(labels_mem + labels_batch, self.nc,
                                                                self.nimages + current_batch_size - self.capacity)
                self.nimages = self.capacity

                # --- UPDATE TEMP MEMORY ---
                tot_imgs = len(labels_mem) + len(labels_batch)
                
                new_mem = []
                new_map_list = []
                imgs_added = 0

                for j in range(tot_imgs):
                    if j not in indices_to_remove:
                        if j < len(labels_mem):
                            # if sample from memory copy from memory
                            new_mem.append(deepcopy(labels_mem[j]))
                            new_map_list.append((map_list[j][0], map_list[j][1]))
                        else:
                            imgs_added += 1
                            # copy from batch and add to memory
                            new_mem.append(deepcopy(labels_batch[j - len(labels_mem)]))
                            new_map_list.append(("data", start_index + j - len(labels_mem)))  # index in dataset: start index + idx, idx = [0, batch_size]

                labels_mem = new_mem
                map_list = new_map_list

                print(f"Batch {i+1} done, images added from the batch: {imgs_added}")

        labels = []
        im_files = []
        to_tasks = []

        imgs_added = 0

        # update real memory
        for loc, idx in map_list:

            if loc == "mem":
                labels.append(deepcopy(self.labels[idx]))
                im_files.append(deepcopy(self.im_files[idx]))
                to_tasks.append(self.to_tasks[idx])

            else:
                imgs_added += 1
                labels.append(deepcopy(dataset.labels[idx]))
                im_files.append(deepcopy(dataset.im_files[idx]))
                to_tasks.append(self.ntasks-1)

        self.labels = labels
        self.im_files = im_files
        self.to_tasks = to_tasks

        self.save_stats(imgs_added)

    def get_indices_to_remove(self, data, nc, num_iter):

        if self.ths is None:

            #return memory_update_indices(data, nc, False, cross_entropy, num_iter)
            return efficient_memory_update_indices(data, nc, cross_entropy_torch, num_iter)
            
        else:
             return memory_update_indices(data, nc, False, cross_entropy, num_iter, None, count_labelsv2, self.ths)


    def get_cls(self, dataset, start_index, batch_size):
        """Get batch (list) of np arrays with classes present in each image of the batch"""

        list_cls = []

        for i in range(batch_size):
            list_cls.append(deepcopy(dataset.labels[start_index + i]["cls"].reshape(-1,).astype(int)))

        return list_cls
    
    def count_duplicates(self):

        dict_counts = dict(Counter(self.im_files))

        num_duplicates = 0
        for k in dict_counts:
            num_duplicates += 1 if dict_counts[k]>1 else 0

        return num_duplicates


    def get_stats(self):

        labels_mem = []
        for i in range(self.nimages):
            labels_mem.append(deepcopy(self.labels[i]["cls"]).reshape(-1,).astype(int))

        return get_labels_distribution(labels_mem, self.nc, ths=self.ths)
    
    def save_stats(self, imgs_added):
        
        add = 1 if self.count_dup else 0
        to_save = np.zeros(self.max_nc+2+add, dtype=np.float32)
        to_save[:self.nc] = self.get_stats()
        to_save[-(2+add)] = imgs_added
        to_save[-(1+add)] = self.nc
        if add:
            to_save[-1] = self.count_duplicates()

        if self.ntasks == 1:
            pd.DataFrame(to_save.reshape(1,-1), columns=self.header).to_csv(self.results_dir+f"/ocdm.csv", sep="\t", header=None, index=False)
        else:
            pd.DataFrame(to_save.reshape(1,-1), columns=self.header).to_csv(self.results_dir+f"/ocdm.csv", sep="\t", header=None, index=False, mode="a")

        
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
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))
    
    def get_batch(self, batch_dim):
        """Get a random batch of samples
        
        :param batch_dim: number of samples in the batch

        return: dict, for each key a batch of items e.g. for key "img" a batch of tensors (shape: [batch_dim x 3 x 640 x 640]).
                see ultralytics documentation for the format
        """

        indices = np.arange(self.nimages)
        np.random.shuffle(indices)

        list_samples = [self[i] for i in indices[:batch_dim]]

        # concat by key
        batch = self.collate_fn(list_samples)

        return batch
    
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["task_id"] = self.to_tasks[index]#torch.Tensor([self.to_tasks[index]]).int()
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