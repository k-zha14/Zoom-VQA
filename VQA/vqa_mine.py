import os
import os.path as ops
import time
import sys
import yaml

import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
import fastvqa.models as models
import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader


def sigmoid_rescale(score, model="FasterVQA"):
    # rescale by self-mean & self-std
    assert len(score) > 1, 'Expect input is a list not a single number.'
    score = (score - np.mean(score)) / np.std(score)
    score = 1 / (1 + np.exp(-score)) * 100
    
    return score


class MyDataset(Dataset):
    def __init__(self, opt='./options/cvpr2023/finetune_vdpve.yml', cfg_path='./examplar_data_labels/VDPVE/val_offline_labels.txt', 
                 data_dir='../data/cvpr2023_vdpve/train/', args=None):
        # load evaluate settings
        with open(opt, 'r') as f:
            opt = yaml.safe_load(f) 

        self.t_data_opt = opt["data"]["val-vqpve"]["args"]    # temporal sample policy
        self.s_data_opt = opt["data"]["val-vqpve"]["args"]["sample_types"]    # spatial sample policy
        
        # update patch size
        self.s_data_opt['fragments']['fsize_h'] = args.patch_size * 8 
        self.s_data_opt['fragments']['fsize_w'] = args.patch_size * 8 
        
        # load data from json file 
        with open(cfg_path, 'r') as f:
            lines = f.read().splitlines()

        self.paths = [ops.join(data_dir, line.split(',')[0]) for line in lines]
        self.moses = [float(line.split(',')[-1]) for line in lines]
        self.filenames = [ops.basename(path) for path in self.paths]
        self.args = args
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # extract batch from video
        video_reader = decord.VideoReader(self.paths[idx])
        vsamples = {}
        
        for sample_type, sample_args in self.s_data_opt.items():
            ## Sample Temporally
            sample_args["frame_interval"] = args.frame_interval
            sample_args["num_clips"] = args.num_clips
        
            sampler = FragmentSampleFrames(sample_args["clip_len"], sample_args["num_clips"], sample_args["frame_interval"])
            num_clips = sample_args.get("num_clips", 1)
            frames = sampler(len(video_reader), train=False)
            
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            # print('before resize:', video.shape)

            # video resize input
            if self.args.rsize != -1:
                h, w = video.shape[-2:]

                if h > w:
                    t_h, t_w = int(h / (w / self.args.rsize)), self.args.rsize
                else:
                    t_h, t_w = self.args.rsize, int(w / (h / self.args.rsize))

                r_video = torch.nn.functional.interpolate(
                    video / 255.0, size=(t_h, t_w), mode="bilinear"
                )
                r_video = (r_video * 255.0).type_as(video)
            else:
                r_video = video
            # print('after resize:', r_video.shape)
                
            ## Sample Spatially
            # print('sampled args:', sample_args)
            sampled_video = get_spatial_fragments(r_video, **sample_args, is_train=False)
            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            # print('after pre-process:', sampled_video.shape)
            # sys.exit(1)
            
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
            vsamples[sample_type] = sampled_video
            vsamples['label'] = self.moses[idx]
            vsamples['filename'] = self.filenames[idx]

        return vsamples


def main(args):    
    with open('./options/cvpr2023/finetune_vdpve.yml', "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    opt['model']['args']['backbone_size'] = args.backbone
    opt['model']['args']['patch_size'] = (2, args.patch_size, args.patch_size)
    evaluator = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(args.device)
    print('Load pretrained weights from:', args.ckpt)
    evaluator.load_state_dict(torch.load(args.ckpt, map_location=args.device)["state_dict"])

    ### Data Definition
    if args.dataset == 'off_val':
        # 1. VDPVE-offline val, 111 videos
        cfg_path = './examplar_data_labels/VDPVE/val_offline_labels.txt'
        data_dir = '../data/train/'
    elif args.dataset == 'ol_val':
        # 2. VDPVE-online val, 119 videos
        cfg_path = './examplar_data_labels/VDPVE/val_ol_labels.txt'
        data_dir = '../data/validation/'
    elif args.dataset == 'test':
        # 3. Final-test, 253 videos
        cfg_path = './examplar_data_labels/VDPVE/test_labels.txt'
        data_dir = '../data/test/'

    
    mydataset = MyDataset(cfg_path=cfg_path, data_dir=data_dir, args=args)
    data_loader = DataLoader(mydataset, batch_size=1, shuffle=False, num_workers=12)
    
    filenames, scores, moses = list(), list(), list()
    elapsed_time = 0. 
    
    for batch in tqdm(data_loader):
        vsamples = dict()
        vsamples['fragments'] = batch['fragments'].squeeze().to(args.device)   

        torch.cuda.synchronize()
        tik = time.time()

        result = evaluator(vsamples)

        torch.cuda.synchronize()
        elapsed_time += time.time() - tik
        
        scores.append(result.mean().item())

        filenames.append(batch['filename'][0])
        moses.append(batch['label'].item())

    # statistics the SRCC&PLCC
    scores = np.array(scores) if args.no_rescale else sigmoid_rescale(np.array(scores))
    # moses = np.array(moses)
    # srcc, plcc = spearmanr(scores, moses)[0], pearsonr(scores, moses)[0]

    # print('srcc: {:.4f}, plcc: {:.4f}, score: {:.4f}'.format(srcc, plcc, (srcc+plcc)/2) )
    print('Elapsed time {:.4f}s in total, {:.4f}s per video.'.format(elapsed_time, elapsed_time / len(scores)))

    # save as a txt file
    with open(f'../results/vqa_preds.txt', 'w') as f:
        for name, score in zip(filenames, scores):
            f.write('{},{}\n'.format(name, score))


if __name__ == "__main__":
    # update: 1) support original torch.Dataset; 2) resacle depends on the training mode 'original' or 'regression' 
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset', type=str, default='test', choices=['test', 'ol_val', 'off_val'], help='select the inferred dataset')
    parser.add_argument("-v", "--video_path", type=str, default="./demos/10053703034.mp4", help="the input video path")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the running device")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/vqa_best_29e_val-vqpve_s.pth", help="the trained checkpoints")
    parser.add_argument("--backbone", type=str, default="swin_tiny_grpb", help="the trained checkpoints")
    parser.add_argument("--rsize", type=int, default=480, help="the minimum edge of resized video")
    parser.add_argument("--no_rescale", action='store_true', help="whethre rescale the predicted score to 0~100 range")
    parser.add_argument('--patch_size', type=int, default=4, help='the minimum image edge')
    parser.add_argument('--frame_interval', type=int, default=2, help='the interval of temporal sampled frames')
    parser.add_argument('--num_clips', type=int, default=4, help='the number of sampled video clips ')

    args = parser.parse_args()

    # start inferring
    main(args)