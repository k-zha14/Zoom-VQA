import os
import sys
import os.path as ops
import json
import math
import random

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from einops import rearrange

from itertools import cycle


DATAHUB = {
    'mgtv': 'configs/mgtv_dataset_05fps.json',
    'mgtv_10frames': 'configs/mgtv_dataset_10frames.json', 
    'mgtv_all': 'configs/mgtv_dataset_all.json',
    # open-source datasets
    'konvid1k': 'configs/opensrc/konvid1k_dataset.json',    # 1200 vids
    'livevqa': 'configs/opensrc/live_vqc_dataset.json',    # 585 vids
    'youtubeugc': 'configs/opensrc/youtubeUGC_dataset.json',    # 1056 vids
    'koniq10k': 'configs/opensrc/koniq10k_dataset.json',    # 10073 imgs 
}

    
def get_transforms(is_val, args):
    """Return FR or NR data augmentations.

    Arguments:
        mode: str, in ['FR', 'NR'] group;
    Returns:
        data_trans: dict, including 'train' and 'val' subsets;
    """
    rsize, csize = args.rsize, args.csize
    means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # print(f'Train/Val: {not is_val}, Resize: {rsize}, Crop size: {csize}')
   
    if not is_val:    # for train
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rsize),
            transforms.RandomCrop(csize),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(means, stds),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rsize),
            transforms.CenterCrop(csize),    # CenterCrop
            transforms.Normalize(means, stds), 
        ])
        

# Setup for video/image training dataset
class UnifiedMixDataset(Dataset):
    """Implementation of torch dataset for unified video QA dataset and image QA dataset.
    
    Attrubutes:
        items: list of tuple element, such as (img_path, mos, cls_label).
        is_val: whether apply train/val augmentations.
        args: other arguments.
    """ 
    def __init__(self, items, is_val, args):
        self.items = items
        self.transform = get_transforms(is_val=is_val, args=args)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # image dataset 
        if len(item) == 2:
            img_path, mos = item[0], item[1]

        # video dataset
        else:
            img_folder, n_frame, mos = item[0], item[1], item[2]
            img_path = ops.join(img_folder, str(random.choice(range(1, n_frame + 1))).zfill(6) + '.png')
        
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img, np.array([mos], dtype=np.float32)


# Setup for video validation dataset
class UnifiedVideoDataset(Dataset):
    """Implementation of torch dataset for unified video QA dataset and train image QA dataset.
    
    Attrubutes:
        items: list of tuple element, such as (img_path, mos, cls_label).
        transform: torchvision transform functions.
        n_frames: list of no. of video frame.
    """ 
    def __init__(self, items, is_val, args):
        self.items = items
        self.transform = get_transforms(is_val=is_val, args=args)
        self.img_paths, self.moses = self.form_samples()

    def form_samples(self):
        img_paths, moses = list(), list()
        for item in self.items:
            path, n_frame, mos = item[0], item[1], item[2]
            img_paths.extend([ops.join(path, str(i).zfill(6) + '.png') for i in range(1, n_frame + 1)])    
            moses.extend([mos] * n_frame)

        return img_paths, moses

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]))
        return img, np.array([self.moses[idx]], dtype=np.float32)


def get_loader(args):
    # set main process lock and freeze other processes
    if args.local_rank not in (-1, 0):
        torch.distributed.barrier()
    
    train_items, val_items = list(), list()
    dataset_scores = list()

    for dataset in args.dataset.split('+'):
        with open(DATAHUB[dataset], 'r') as f: 
            label_file = json.load(f)

        if dataset == 'mgtv':
            val_items = label_file['val']
        train_items.extend(label_file['train'])
        scores = [item[2] if len(item) > 2 else item[1] for item in label_file['train']]
        dataset_scores.append(scores)
        # print('scores range:', min(scores), max(scores), len(scores))
            
    args.vid_frames = [item[1] for item in val_items]    # initialize val vid frames
    train_dataset = UnifiedMixDataset(items=train_items, is_val=False, args=args)
    val_dataset = UnifiedVideoDataset(items=val_items, is_val=True, args=args) if args.local_rank in (-1, 0) else None
    
    if args.local_rank in (-1, 0):
        print('No. of train&val vids/images: {} & {}, boundaries: {}'.format(
            len(train_items), len(val_items), [len(scores) for scores in dataset_scores], 
        ))

    # release main process lock
    if args.local_rank == 0:
        torch.distributed.barrier()

    # 'shuffle' confict with 'sampler' argument, and RandomSampler equals set 'shffle=True'
    train_sampler = myBatchSampler2(dataset_scores=dataset_scores, batch_size=args.batch_size, n_iters_epoch=175)
    
    # SequentialSampler euqals set 'shuffle=False'
    val_sampler = SequentialSampler(val_dataset)

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=12)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=12, pin_memory=False) if val_dataset is not None else None

    return train_loader, val_loader


class RandRotation(object):
    """Reimplementation of Rotation on 90, 180, 270 degrees."""
    def __call__(self, imgs):
        """Imgs must be tensors."""
        prob = random.random()

        if prob < 0.25:
            pass
        elif prob < 0.5:
            imgs = transforms.functional.rotate(imgs, 90)
        elif prob < 0.75:
            imgs = transforms.functional.rotate(imgs, 180)
        else:
            imgs = transforms.functional.rotate(imgs, 270)

        return imgs


class myBatchSampler(BatchSampler):
    def __init__(self, dataset_scores, batch_size, n_iters_epoch=None): 
        assert len(dataset_scores) >= 2, f'Must be multi-dataset, n_dataset > 1!'
        self.dataset_scores = dataset_scores
        dataset_lens = [len(group) for group in dataset_scores]
        # 1. split every dataset batch size, mgtv occupy 50%
        self.n_batch_mgtv = batch_size // 2
        self.n_batch_aux = batch_size // 2 // (len(dataset_lens) - 1)
        print('Batch Split, mgtv: {}, others: {}'.format(self.n_batch_mgtv, self.n_batch_aux))

        self.dataset_idxs = self.build_lut(dataset_lens)
        self.shuffle_lut()

        # 2. calculate n_iters_epoch, larger than the largetest
        if n_iters_epoch:
            self.n_iters_epoch = n_iters_epoch
        else:
            self.n_iters_epoch = 0
            for i in range(1, len(dataset_lens)):
                if i == 1:
                    n_iters = math.ceil(dataset_lens[i] / self.n_batch_mgtv)
                else:
                    n_iters = math.ceil((dataset_lens[i] - dataset_lens[i-1]) / self.n_batch_aux)
                self.n_iters_epoch = max(self.n_iters_epoch, n_iters)
            # print('n_iters:', dataset_lens[i], dataset_lens[i-1], n_iters)

    def __iter__(self):
        for _ in range(self.n_iters_epoch):
            batch_idxs = list()
            
            for i, group in enumerate(self.dataset_iters):
                if i == 0:
                    for j in range(self.n_batch_mgtv):
                        batch_idxs.append(next(group))
                else:
                    for j in range(self.n_batch_aux):
                        batch_idxs.append(next(group))
            # print('Dataset1 idxs:', batch_idxs[:self.n_batch_mgtv])
            # print('Dataset2 idxs:', batch_idxs[self.n_batch_mgtv:])
            yield batch_idxs
    
    def __len__(self):
        return self.n_iters_epoch

    def build_lut(self, dataset_lens):
        start_idx = 0 
        dataset_idxs = list()
        for length in dataset_lens:
            dataset_idxs.append(list(range(start_idx, start_idx + length)))
            start_idx += length

        return dataset_idxs

    def shuffle_lut(self):
        self.dataset_iters = list()
        for group in self.dataset_idxs:
            random.shuffle(group)
            self.dataset_iters.append(cycle(group))


# batch sampler v2.0, Opensource dataset is groupped by scores 
class myBatchSampler2(BatchSampler):
    def __init__(self, dataset_scores, batch_size, n_iters_epoch=None):
        assert len(dataset_scores) >= 2, f'Must be multi-dataset, n_dataset > 1!'
        self.dataset_scores = dataset_scores
        dataset_lens = [len(group) for group in dataset_scores]
        # 1. split every dataset batch size, mgtv occupy 50%
        self.n_batch_mgtv = batch_size // 2
        self.n_batch_aux = batch_size // 2 // (len(dataset_lens) - 1)
        # print('Batch Split, mgtv: {}, others: {}'.format(self.n_batch_mgtv, self.n_batch_aux))

        self.dataset_idxs = self.build_lut(dataset_lens)
        self.shuffle_lut()
        
        # 2. calculate n_iters_epoch, larger than the largetest
        if n_iters_epoch:
            self.n_iters_epoch = n_iters_epoch
        else:
            self.n_iters_epoch = 0
            for i in range(1, len(dataset_lens)):
                if i == 1:
                    n_iters = math.ceil(dataset_lens[i] / self.n_batch_mgtv)
                else:
                    n_iters = math.ceil((dataset_lens[i] - dataset_lens[i-1]) / self.n_batch_aux)
                self.n_iters_epoch = max(self.n_iters_epoch, n_iters)
            # print('n_iters:', dataset_lens[i], dataset_lens[i-1], n_iters)

    def __iter__(self):
        for _ in range(self.n_iters_epoch):
            batch_idxs = list()
            
            for i, group in enumerate(self.dataset_iters):
                batch_idxs.extend(next(group))

            yield batch_idxs
    
    def __len__(self):
        return self.n_iters_epoch

    def build_lut(self, dataset_lens):
        # Initialize the indexes and group by batch
        dataset_idxs = list()
        start_idx = 0
        for i, length in enumerate(dataset_lens):
            idxs = list(range(start_idx, start_idx + length))
            if i == 0:
                # mgtv dataset, random sample
                random.shuffle(idxs)
                n_batchs = math.ceil(len(idxs) / self.n_batch_mgtv)
                idxs = cycle(idxs)
                g_idxs = [[next(idxs) for _ in range(self.n_batch_mgtv)] for _ in range(n_batchs)]

            else:
                # opensource dataset, grouop by scores, 1~2, 2~3, 3~4, 4~5
                scores = self.dataset_scores[i]
                g_idxs_byScores = [list() for i in range(4)]
                assert len(idxs) == len(scores), 'Unexpected dataset indexs length and scores: {} & {}'.foramt(len(idxs), len(scores))
                for idx, score in zip(idxs, scores):
                    score = np.clip(score, 1.0001, 4.9999)
                    g_idxs_byScores[int(score - 1.)].append(idx)

                max_group = max([len(group) for group in g_idxs_byScores])
                n_iters = math.ceil(max_group * 4 / self.n_batch_aux)
                
                # shuffle and generate batch indexes
                for j, group in enumerate(g_idxs_byScores):
                    random.shuffle(group)
                    # print('bucket: {}, idxs: {}'.format(j, group))

                g_idxs_byScores = [cycle(group) for group in g_idxs_byScores]
                g_idxs = list()
                for _ in range(n_iters):
                    batch_idxs = list()
                    for group in g_idxs_byScores:
                        for _ in range(self.n_batch_aux // 4):
                            batch_idxs.append(next(group))

                    g_idxs.append(batch_idxs)

            start_idx += length
            dataset_idxs.append(g_idxs)

        return dataset_idxs

    def shuffle_lut(self):
        # Actually, shuffle on batch-level order, not sample-level
        self.dataset_iters = list()
        for g_idxs in self.dataset_idxs:
            random.shuffle(g_idxs)
            self.dataset_iters.append(cycle(g_idxs))


if __name__ == '__main__':
    pass
