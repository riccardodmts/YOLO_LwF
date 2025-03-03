import math
from copy import deepcopy
from pathlib import Path
from itertools import repeat
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics.utils import LOGGER, TQDM, LOCAL_RANK, NUM_THREADS, is_dir_writeable
from ultralytics.utils.ops import resample_segments
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.data.utils import LOGGER
from ultralytics.data.utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label
from multiprocessing.pool import ThreadPool

from .OCDM import OCDM


class DERLoader:

    def __init__(self, task_tr_loader, replay_memory):

        self.task_tr_loader = task_tr_loader
        #self.dataset = self.task_tr_loader.dataset
        self.replay_memory = replay_memory
        #self.batch_sampler = self.task_tr_loader.batch_sampler

        # get batch size for current task loader
        self.batch_size_per_dataset = self.task_tr_loader.batch_size

    def __len__(self):
        return len(self.task_tr_loader)

    def __iter__(self):
        
        self.task_tr_iter = iter(self.task_tr_loader)
        self.replay_memory.ready_to_sample()
        return self
    
    def __next__(self):

        try:
            task_batch = next(self.task_tr_iter)
        except StopIteration:
            self.replay_memory.end_sample()
            raise StopIteration
        
        # get batch from memory
        b = task_batch["img"].shape[0]  # batch_size for current task batch
        replay_batch = self.replay_memory.get_batch(b)

        num_labels = task_batch["cls"].shape[0]
        fake_der_targets = torch.zeros_like(replay_batch["der_target"])
        fake_class = torch.zeros_like(replay_batch["class_seen"])

        new_batch = {}

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

        new_batch["der_target"] = torch.cat([fake_der_targets, replay_batch["der_target"]])
        new_batch["class_seen"] = torch.cat([fake_class, replay_batch["class_seen"]])
        new_batch["num_labels"] = num_labels

        return new_batch

class DERMemory(Dataset):
    """Dark Experience Replay for YOLOv8. NOTE: the implementation differs form the original since task boundaries are taken into account."""

    def __init__(self, capacity, results_dir, batch_size = 8, shape=[144, 8400]):

        self.capacity = capacity
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.ntasks = 0
        self.ntasks = 0
        self.shape = (capacity, shape[0], shape[1])  # shape memory map
        self.filename = results_dir + "memmap.mmap"

        self.augment = False
        self.hyp = None # first_dataset.hyp
        self.imgsz = None #first_dataset.hyp.imgsz
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.prefix = ""
        self.rect = False
        self.data = None # first_dataset.data

        # list with # samples per task
        self.n_available_images_per_task = []
        self.map_per_task = [list(range(self.capacity))]
        self.nimgs_per_task = None

        self.im_files = [None] * self.capacity
        self.labels = [None] * self.capacity

        self._ready_to_sample = True
        self.classes = []
        self.mask = self.capacity * [None]

    def update_memory(self, dataset, trainer, hyp=None, classes_seen = []):

        self.classes += classes_seen

        # compute number of images per task to store (given the new task)
        self.nimgs_per_task = math.floor(self.capacity/(self.ntasks + 1))
        # compute number of images to remove for each old task
        nimgs_per_task_to_remove = math.floor(self.nimgs_per_task/self.ntasks) if self.ntasks > 0 else 0

        indices_removed = None

        if nimgs_per_task_to_remove > 0:
            # remove images (for each task, save in list indeces w.r.t. the overall memory batch)
            indices_removed = self.get_indices_to_remove(nimgs_per_task_to_remove)

        else:
            # if first task, no samples are removed -> indices_removed = all memory indices 
            indices_removed = np.arange(self.capacity)
            np.random.shuffle(indices_removed)
            self.imgsz = dataset.imgsz
            self.transforms = self.build_transforms(hyp)
            
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

        # disable augmentation
        dataset.augment = False
        dataset.transforms = dataset.build_transforms(hyp=dataset.hyp)

        dataset_interface = DatasetInterface(dataset, indices)

        # create loader
        loader = DataLoader(
            dataset_interface,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=getattr(dataset, "collate_fn"),
        )

        # create memory map or open it
        if self.ntasks == 1:
            # create mem
            f = np.memmap(self.filename, dtype=np.float32, mode='w+', shape=self.shape)

        else:
            # open mem
            f = np.memmap(self.filename, dtype=np.float32, mode="r+", shape=self.shape)

        idx = 0

        for i, batch in enumerate(loader):

            # replace img path and label (label just needed to avoid bugs in ultralytics code)
            for j in range(batch["img"].shape[0]):
                dataset_idx = indices[i * self.batch_size + j]
                self.im_files[indices_removed[idx]] = deepcopy(dataset.im_files[dataset_idx])
                self.labels[indices_removed[idx]] = deepcopy(dataset.labels[dataset_idx])
                idx+=1

            # compute output current model
            output = trainer.der_forward(batch)

            # get indices_removed for current batch (get a list of indices)
            indices_batch = indices_removed[idx-batch["img"].shape[0]:idx]  # e.g. indices_removed [5, 10, 35, 3], batch_size = 2, -> [5,10], [35, 3]
            #update memmap
            f[indices_batch, :, :] = output

        for i in range(self.capacity):
            if i in indices_removed:
                self.mask[i] = classes_seen[-1]

        f.flush()
        del f
        print(set(self.mask))
        print(f"Removed images: {idx}, last 10 indeces removed: {indices_removed[-10:]}")


    def get_indices_to_remove(self, num_imgs):

        indeces_to_remove = []

        # if num_imgs to remove per task is lower than self.nimgs_per_task, add extra samples to remove per task
        extra_samples_to_remove = np.zeros(self.ntasks, dtype=np.int32)

        if num_imgs * self.ntasks < self.nimgs_per_task:
            task_idxs = np.random.randint(0, self.ntasks, self.nimgs_per_task - num_imgs * self.ntasks)
            values, counts = np.unique(task_idxs, return_counts=True)
            extra_samples_to_remove[values] += counts

        
        for i, task_map in enumerate(self.map_per_task):

            indeces = np.arange(len(task_map), dtype=np.int32)
            np.random.shuffle(indeces)
            indices_to_remove_task = indeces[:num_imgs + extra_samples_to_remove[i]]

            map_copy = []
            for j, idx in enumerate(task_map):
                if j not in indices_to_remove_task:
                    map_copy.append(idx)
                else:
                    indeces_to_remove.append(idx)

            self.map_per_task[i] = map_copy
            # create map for new task
        self.map_per_task.append(deepcopy(indeces_to_remove))

        return indeces_to_remove
            

        
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
        return len(self.im_files)
    
    def ready_to_sample(self):
        self._ready_to_sample = True
        self.f = np.memmap(self.filename, dtype=np.float32, mode="r+", shape=self.shape)
    
    def end_sample(self):
        self._ready_to_sample = False
        del self.f
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        if self._ready_to_sample:
            sample = self.transforms(self.get_image_and_label(index))
            sample["der_target"] = torch.tensor(self.f[index]).float()
            sample["class_seen"] = torch.tensor(self.mask[index]).int()
            return sample
        raise Exception("NOT READY TO SAMPLE!!!")
    
    def get_batch(self, batch_dim):
        """Get a random batch of samples
        
        :param batch_dim: number of samples in the batch

        return: dict, for each key a batch of items e.g. for key "img" a batch of tensors (shape: [batch_dim x 3 x 640 x 640]).
                see ultralytics documentation for the format
        """

        indices = np.arange(self.capacity)
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
            if k =="der_target":
                value = torch.stack(value, 0)
            if k =="class_seen":
                value = torch.stack(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
    


class DatasetInterface(Dataset):

    def __init__(self, dataset, indices):
        super().__init__()

        self.dataset = dataset
        self.indices = indices
        self.len = len(indices)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        idx = self.indices[index]
        return self.dataset[idx]



class DEROCDMemory(OCDM):


    def __init__(self, capacity, first_dataset, first_trainer, nc, max_nc, results_dir,  batch_size = 8, ocdm_batch_size=0, shape=[144, 8400], ths=None, count_dup=False):
        
        self.batch_size_der = batch_size
        self.dir = results_dir

        self.shape = (capacity, shape[0], shape[1])  # shape memory map
        self.filename = results_dir + "memmap.mmap"

        self.capacity = capacity
        self.ntasks = 0
        self.batch_size = ocdm_batch_size # if batch_size = -1, update memory in one shot
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

        self.transforms = self.build_transforms(first_dataset.hyp)
        self.update_memory(first_dataset, first_trainer, self.nc)

        self._ready_to_sample = False

    
    def update_memory(self, dataset, trainer, nc=None):

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

        data_indices = []
        data_map = []
        mem_old_indices = []
        mem_new_indeces = []

        imgs_added = 0
        imgs_from_mem = 0
        abs_idx=0

        # update real memory
        for loc, idx in map_list:

            if loc == "mem":
                imgs_from_mem+=1
                labels.append(deepcopy(self.labels[idx]))
                im_files.append(deepcopy(self.im_files[idx]))
                mem_old_indices.append(idx)
                mem_new_indeces.append(abs_idx)
            else:
                imgs_added += 1
                labels.append(deepcopy(dataset.labels[idx]))
                im_files.append(deepcopy(dataset.im_files[idx]))
                data_indices.append(idx)
                data_map.append((idx, abs_idx))

            abs_idx+=1

        if self.ntasks == 1:
            f = np.memmap(self.filename, dtype=np.float32, mode='w+', shape=self.shape)

        else:
            f = np.memmap(self.filename, dtype=np.float32, mode="r+", shape=self.shape)
            # preserve old samples
            f[mem_new_indeces, :, :] = f[mem_old_indices, :, :]


        dataset.augment = False
        dataset.transforms = dataset.build_transforms(hyp=dataset.hyp)
        dataset_interface = DatasetInterface(dataset, data_indices)

        # create loader
        loader = DataLoader(
            dataset_interface,
            batch_size=self.batch_size_der,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,
            collate_fn=getattr(dataset, "collate_fn"),
        )

        for i, batch in enumerate(loader):

            temp = [t[1] for t in data_map[i*self.batch_size_der : i*self.batch_size_der+batch["img"].shape[0]]]

            # compute output current model
            output = trainer.der_forward(batch)

            f[temp, :, :] = output

        f.flush()
        del f

        print(f"Images from mem: {imgs_from_mem}, new images: {imgs_added}")

        self.labels = labels
        self.im_files = im_files

        self.save_stats(imgs_added)
        

    def __len__(self):
        return len(self.im_files)
    
    def ready_to_sample(self):
        self._ready_to_sample = True
        self.f = np.memmap(self.filename, dtype=np.float32, mode="r+", shape=self.shape)
    
    def end_sample(self):
        self._ready_to_sample = False
        del self.f
    
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        if self._ready_to_sample:
            sample = self.transforms(self.get_image_and_label(index))
            sample["der_target"] = torch.tensor(self.f[index]).float()
            return sample
        raise Exception("NOT READY TO SAMPLE!!!")
    
    def get_batch(self, batch_dim):
        """Get a random batch of samples
        
        :param batch_dim: number of samples in the batch

        return: dict, for each key a batch of items e.g. for key "img" a batch of tensors (shape: [batch_dim x 3 x 640 x 640]).
                see ultralytics documentation for the format
        """

        indices = np.arange(self.capacity)
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
            if k =="der_target":
                value = torch.stack(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch