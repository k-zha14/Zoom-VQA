import os
import sys
import time 
import json
import os.path as ops

import torch
from tqdm import tqdm
from torchvision import transforms

import utils_single as utils
from datasets_single import UnifiedVideoDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from einops import rearrange


def run(args):
    # Print infer settings 
    print('*' * 30)
    print(args)
    print('*' * 30)
    
    # 1. build model & load trained weighted
    net = utils.get_model(args)
    ckpt = torch.load(args.ckpt, map_location='cpu')['state_dict']
    
    trim_ckpt = {key.replace('module.', ''): val for key, val in ckpt.items()}    # delete 'module.' prefix
    net.load_state_dict(trim_ckpt)
    net.cuda()
    print('No. of learned parameters(M):', utils.count_parameters(net))

    # 2. build dataloader, support 'vdpve_val_2fps.json' & 'vdpve_test_2fps.json'
    with open('configs/vdpve_{}_2fps.json'.format(args.dataset), 'r') as f: 
        config = json.load(f)
    val_items = config['val']

    vid_names = [ops.basename(item[0]) for item in val_items]

    if len(val_items[0]) == 3:
        print('Infer on videos.')
        args.vid_frames = [item[1] for item in val_items]    # for video
    else:
        print('Infer on images.')
        args.vid_frames = None    # for images
    
    dataset = UnifiedVideoDataset(items=val_items, is_val=False, is_test=True, args=args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    print('No. of test vids: {}, inflated imgs: {}'.format(
        len(val_items), len(dataset)
    ))

    # 3. run & infer 
    # TODO(@Kai) Record data loading and model forward time seperately
    preds, labels = list(), list()
    elapsed_time = 0.

    with torch.no_grad():
        net.eval()

        for batch_x, batch_y in tqdm(loader, total=len(loader)):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_x = rearrange(batch_x, 'b n c h w -> (b n) c h w')
            
            torch.cuda.synchronize()
            tik = time.time()

            output = net(batch_x)

            torch.cuda.synchronize()
            elapsed_time += time.time() - tik
            
            output = rearrange(output, '(b n) o -> b n o', b=len(batch_x) // args.n_crops, n=args.n_crops).mean(dim=1)

            preds.append(output)
            labels.append(batch_y)

        # apply metrics, compute img-level and vid-level seperately
        preds, labels = torch.cat(preds).cpu().squeeze(), torch.cat(labels).cpu().squeeze()
        # srocc, plcc, rmse = utils.eval_preds(preds, labels, vid_frames=args.vid_frames)
        # main_score = 0.5 * srocc + 0.5 * plcc   # delete plcc, srcc matters most

    # output log
    # print(
    #     'RMSE: {rmse:.4f}, SRCC: {srocc:.4f}, PLCC:{plcc:.4f}, Score:{main_score:.4f}'.format(
    #         srocc=srocc, plcc=plcc, main_score=main_score, rmse=rmse,
    #     )
    # )
    print('Elapsed time {:.4f}s in total, {:.4f}s per video.'.format(elapsed_time, elapsed_time / len(preds)))

    # convert image preds to vid scores and dump as txt file 
    f_name = args.ckpt.replace('/', '_')
    vid_preds, vid_moses = list(), list()
    idx = 0 
    if args.vid_frames:
        for frame in args.vid_frames:
            vid_preds.append(preds[idx: idx+frame].mean().item())
            vid_moses.append(labels[idx: idx+frame].mean().item())
            idx += frame
        assert idx == sum(args.vid_frames)
    else:
        vid_preds = preds
    
    ## seriralize as .txt file 
    with open('../results/iqa_preds.txt'.format(f_name), 'w') as f:
        for name, pred in zip(vid_names, vid_preds):
            print('{},{}'.format(name+'.mp4', pred), file=f)
    

if __name__ == '__main__':
    parser = ArgumentParser(description='2022 NTIRE Challenge&Infer')

    # Model Hyper-parameters
    parser.add_argument('--arch', type=str, default='cpnet_multi', help='the name of model structure')
    parser.add_argument('--drop_path', type=float, default=0.0, help='stochastic depth ratio, default 0.0')
    parser.add_argument('--dropout', type=float, default=0.0, help='stochastic drop out ratio, default 0.0')
    parser.add_argument('--pretrain', action='store_true', help='whether load pretrained weights')

    # Dataset 
    parser.add_argument('--dataset', type=str, default='test', choices=['test', 'val'], help='select the inferred dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='the no. of batch samples')
    parser.add_argument('--n_crops', type=int, default=1, help='the no. of cropped patches')
    parser.add_argument('--rsize', type=int, default=512, help='height of a patch to crop')
    parser.add_argument('--csize', type=int, default=320, help='width of a patch to crop')

    # Trained Checkpoints
    parser.add_argument("--ckpt", type=str, default='./checkpoints/iqa_best_29epoch_checkpoint.pth.tar', help="Name of checkpoint weights.")
    
    # Environment
    parser.add_argument('--gpu', type=str, default='0', help='Set the ith gpu device.')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(args)
