import os
import sys
import os.path as ops
import json
import random
from itertools import cycle

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data.sampler import Sampler, BatchSampler
from einops import rearrange


DATAHUB = {
    # open-source dataset
    'spaq': 'cd_configs/spaq_720p_dataset.json',
    'koniq10k': 'cd_configs/koniq10k_mosZscore_dataset.json',
    'clive': 'cd_confgs/clive_dataset.json',
    'lsvq': 'configs/lsvd_key_frames.json', 

    # cvpr2023 VDPVE
    'vdpve_2fps': 'configs/vdpve_2fps.json', 
}

    
def get_transforms(is_val=False, is_test=False, args=None):
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
        augs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rsize),

            transforms.RandomCrop(csize),
            # GetFragments(fragments=7, fsize=args.csize//7, random=True),

            transforms.RandomHorizontalFlip(),
            transforms.Normalize(means, stds),
        ])
    else:
        augs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rsize),

            transforms.CenterCrop(csize),    # CenterCrop
            # MultiCrops(n_crops=args.n_crops, csize=csize),
            # GetFragments(fragments=7, fsize=args.csize//7, random=False),

            transforms.Normalize(means, stds), 
        ])

    if is_test:
        augs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rsize),

            # transforms.CenterCrop(csize),    # CenterCrop
            MultiCrops(n_crops=args.n_crops, csize=csize),
            # GetFragments(fragments=7, fsize=args.csize//7, random=False),

            transforms.Normalize(means, stds), 
        ])

    return augs
        

class UnifiedVideoDataset(Dataset):
    """Implementation of torch dataset for unified video QA dataset and train image QA dataset.
    
    Attrubutes:
        items: list of tuple element, such as (img_path, mos, cls_label).
        transform: torchvision transform functions.
        n_frames: list of no. of video frame.
    """ 
    def __init__(self, items, is_val, args, is_test=False):
        self.items = items
        self.transform = get_transforms(is_val=is_val, is_test=is_test, args=args)
        self.img_paths, self.moses = self.form_samples()

    def form_samples(self):
        img_paths, moses = list(), list()
        for item in self.items:
            if len(item) == 2:
                img_paths.append(item[0])
                moses.append(item[1])
            else:
                path, n_frame, mos = item[0], item[1], item[2]
                img_paths.extend([ops.join(path, str(i).zfill(6) + '.png') for i in range(1, n_frame + 1)])    
                moses.extend([mos] * n_frame)

        return img_paths, moses

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]))
        return img, np.array([self.moses[idx]], dtype=np.float32)


# VideoDataset With Resolution
class UnifiedVideoResolDataset(Dataset):
    """Implementation of torch dataset for unified video QA dataset and train image QA dataset.
    
    Attrubutes:
        items: list of tuple element, such as (img_path, mos, cls_label).
        transform: torchvision transform functions.
        n_frames: list of no. of video frame.
    """ 
    def __init__(self, items, is_val, args):
        self.items = items
        self.transform = get_transforms(is_val=is_val, args=args)
        self.resol_dict = {'360': 0, '480': 1, '720': 2, '1080': 3}
        self.img_paths, self.moses, self.resols = self.form_samples()
        

    def form_samples(self):
        img_paths, moses, resols = list(), list(), list()
        for item in self.items:
            if len(item) == 2:
                img_paths.append(item[0])
                moses.append(item[1])
            else:
                path, n_frame, mos = item[0], item[1], item[2]
                resol = path.split('_')[-1]
                img_paths.extend([ops.join(path, str(i).zfill(6) + '.png') for i in range(1, n_frame + 1)])    
                moses.extend([mos] * n_frame)
                resols.extend([self.resol_dict[resol]] * n_frame)

        return img_paths, moses, resols

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]))
        return img, np.array([self.moses[idx]], dtype=np.float32), np.array([self.resols[idx]], dtype=np.float32) 


def get_loader(args):
    # set main process lock and freeze other processes
    if args.local_rank not in (-1, 0):
        torch.distributed.barrier()
    
    with open(DATAHUB[args.dataset], 'r') as f: 
        label_file = json.load(f)

    # IMPORTANT, change train items to all labelled data
    if args.dataset == 'lsvq':
        train_items = label_file['train'] + label_file['val']
        val_items = label_file['val_1080p']
    else:
        train_items = label_file['train'] + label_file['val'] if args.all_data else label_file['train']
        val_items = label_file['val']
    
    # (Option) Add offline validation dataset to train dataset
    # with open('configs/mgtv_dataset.json', 'r') as f:
    #     label_file = json.load(f)
    # train_items = train_items + label_file['val']

    args.vid_frames = [item[1] for item in val_items]    # initialize val vid frames

    if args.arch == 'cpnet_resol':
        train_dataset = UnifiedVideoResolDataset(items=train_items, is_val=False, args=args)
        val_dataset = UnifiedVideoResolDataset(items=val_items, is_val=True, args=args) if args.local_rank in (-1, 0) else None
    else:
        train_dataset = UnifiedVideoDataset(items=train_items, is_val=False, args=args)
        val_dataset = UnifiedVideoDataset(items=val_items, is_val=True, args=args) if args.local_rank in (-1, 0) else None
    
    if args.local_rank in (-1, 0):
        print('No. of train&val vids: {} & {}, inflated imgs: {} & {}'.format(
            len(train_items), len(val_items), len(train_dataset), len(val_dataset)
        ))

    # release main process lock
    if args.local_rank == 0:
        torch.distributed.barrier()

    # 1. original version: train loader
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=12, pin_memory=False)

    # 2. sampler version: train loader
    # scores = list()
    # for item in train_items:
    #     n_frame, mos = item[1], item[2]
    #     for _ in range(n_frame):
    #         scores.append(mos)
    # assert len(scores) == len(train_dataset), f'Unexpected amount of moses and items: {len(socres)} {len(train_dataset)}'
    # train_sampler = myBatchSampler(scores, batch_size=args.batch_size, weights=[0.125, 0.21875, 0.4375, 0.21875], n_iters_epoch=175)    # original 175 iterations
    # n_bucket_samples = [len(bucket) for bucket in train_sampler.bucket_idxs]
    # print('Amount of buckets:', n_bucket_samples)
    # print('Selected samples of buckets:', train_sampler.n_sample_buckets)
    # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=12)
    
    # define validation loader, SequentialSampler euqals set 'shuffle=False'
    val_sampler = SequentialSampler(val_dataset)
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


class MultiCrops():
    def __init__(self, n_crops=1, csize=224):
        self.n_crops = n_crops
        self.crop_size = csize

    def __call__(self, img):
        # img: 3D tensor, in (c, h, w) shape, 0~1 range
        h, w = img.shape[-2:]
        assert min(h, w) >= self.crop_size, 'Invalid input image size: {} less than target: {}'.format(min(h, w), self.crop_size)
        
        # center crop
        if self.n_crops == 1:
            h_s = (h - self.crop_size) // 2
            w_s = (w - self.crop_size) // 2
            h_e = h_s + self.crop_size
            w_e = w_s + self.crop_size
            return img[None, :, h_s:h_e, w_s:w_e]

        else:
            # FiveCrop
            if self.n_crops == 5:
                h_points = [0, (h - self.crop_size)//2, 0, h - self.crop_size, h - self.crop_size]
                w_points = [0, (w - self.crop_size)//2, w - self.crop_size, 0, w - self.crop_size]
            # RandomCrop
            else:
                h_points = np.random.choice(h-self.crop_size+1, self.n_crops)
                w_points = np.random.choice(w-self.crop_size+1, self.n_crops)
                # print('h_points:', h_points)
                # print('w_points:', w_points)
            
            patches = list()
            for h_s, w_s in zip(h_points, w_points):
                h_e, w_e = h_s + self.crop_size, w_s + self.crop_size
                patches.append(img[:, h_s:h_e, w_s:w_e])

            return torch.stack(patches)    # 4D tensor, [n_patch, c, h, w]


class GetFragments():
    def __init__(self, fragments=7, fsize=32, random=False):
        self.fragments = fragments
        self.fsize = fsize
        self.random = random
        print(f'Apply fragments, random: {random}')

    def __call__(self, img):
        '''Random crop patches from images.

        Args:
            img: 3D tensor, in (c, h, w) shape;
        '''
        short_edge = min(img.shape[-2:])
        target_size = self.fragments * self.fsize

        assert short_edge >= target_size, 'Original Video shorter edge: {} < patch size: {}'.format(short_edge, target_size)
        res_h, res_w = img.shape[-2:]
        h_frag_length, w_frag_length = res_h // self.fragments, res_w // self.fragments
        hgrids = torch.LongTensor([h_frag_length * i for i in range(self.fragments)])
        wgrids = torch.LongTensor([w_frag_length * i for i in range(self.fragments)])
        
        # 1. if random=True, indexes ramdomly selected for every fragments
        if self.random:
            rnd_h = torch.randint(h_frag_length - self.fsize + 1, (len(hgrids), len(wgrids)))    # [low, high), high is exclusive
            rnd_w = torch.randint(w_frag_length - self.fsize + 1, (len(hgrids), len(wgrids)))

        # 2. Select the center index, for inference
        else:
            rnd_h = torch.ones(len(hgrids), len(wgrids)).int() * ((h_frag_length - self.fsize) // 2)
            rnd_w = torch.ones(len(hgrids), len(wgrids)).int() * ((w_frag_length - self.fsize) // 2)

        target_img = torch.zeros(img.shape[0], self.fsize*self.fragments, self.fsize*self.fragments).to(img.device)

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                # Indexes for orginal images
                h_shift, w_shift = rnd_h[i][j], rnd_w[i][j]
                h_so, h_eo = hs + h_shift, hs + h_shift + self.fsize
                w_so, w_eo = ws + w_shift, ws + w_shift + self.fsize

                # Indexes for new images
                h_s, h_e = i*self.fsize, (i+1)*self.fsize
                w_s, w_e = j*self.fsize, (j+1)*self.fsize
                
                # print(target_img.shape)
                # print(h_so, h_eo, w_so, w_eo)
                target_img[:, h_s: h_e, w_s: w_e] = img[:, h_so: h_eo, w_so: w_eo]

        return target_img


class myBatchSampler(BatchSampler):
    """ Define the different buckets and samples.

    Args:
        scores: list, store every sample score.
        weights: list, sample weight of every bucket.
        batch_size: int.    
    """
    def __init__(self, scores, batch_size, weights=[0.25, 0.25, 0.25, 0.25], n_iters_epoch=None):
        # 1. assign every sample to different bucket and give the result
        self.n_sample_buckets = np.ceil(np.array(weights) * batch_size).astype(np.int32)
        self.bucket_idxs = self.build_lut(scores)
        self.shuffle_lut()
        
        # 2. calculate the maximum iterations on the whole dataset
        if n_iters_epoch:
            self.n_iters_epoch = n_iters_epoch
        else:
            n_buckets = np.array([len(bucket) for bucket in self.bucket_idxs])
            self.n_iters_epoch = np.ceil(np.max(n_buckets / self.n_sample_buckets)).astype(np.int32).item()

    def __iter__(self):
        for _ in range(self.n_iters_epoch):
            batch_idxs = list()
            
            for n_sample, group in zip(self.n_sample_buckets, self.bucket_iters):
                for _ in range(n_sample):
                    batch_idxs.append(next(group))
    
            yield batch_idxs
    
    def __len__(self):
        return self.n_iters_epoch

    def build_lut(self, scores):
        # assign samples by scores to: 1~2, 2~3, 3~4, 4~5 buckets
        bucket_idxs = [list() for i in range(4)]
        for idx, score in enumerate(scores):
            score = np.clip(score, 1.0001, 4.9999)
            bucket_idxs[int(score - 1.)].append(idx)
        
        return bucket_idxs

    def shuffle_lut(self):
        # Actually, shuffle on batch-level order, not sample-level
        self.bucket_iters = list()
        for bucket in self.bucket_idxs:
            random.shuffle(bucket)
            self.bucket_iters.append(cycle(bucket))
        
        
if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms

    path = '../MGTV_VQA_DATA/MGTV_OGC_V1_dataset/validation_1fps/4_EVNSUT4T_34_H264_848_480/000010.png'
    img = Image.open(path)
    array = transforms.ToTensor()(img)
    array = transforms.Resize(256)(array)
    get_fragments = GetFragments(random=False)

    crop_patches = get_fragments(array)
    print(crop_patches.shape, crop_patches.dtype, torch.max(crop_patches), torch.min(crop_patches))
    
