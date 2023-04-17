import os
import sys
import json
import copy
import time
import os.path as ops
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from IQALoss import IQALoss
from scheduler import get_scheduler
from lottery_config import args


def run(args):
    # Only excecute in main process
    if args.level == 'img':
        import utils_single as utils
        from datasets_single import get_loader
    else:
        import utils_vid as utils
        from datasets_vid import get_loader

    # Frozen all seeds, inlcuding: numpy, torch, random
    utils.set_random_seed(args.seed)
    
    if args.local_rank in (-1, 0):
        # print all hypyer-parameters
        print('*' * 30)
        print(args)
        print('*' * 30)
        print('Log dir:', args.log_dir)
        print('Alchemy Tricks, EMA: {}, decay ratio: {:.4f}, CutMix(0.- False): {}, MixUp(1.-False): {}'.format(
            args.EMA, args.ema_decay, args.cutmix, args.mixup
        ))
        # Initialize summary writer
        args.writer = SummaryWriter(args.log_dir)
    
    # Setup CUDA environment, CPU / GPU / DDP
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    
    # Run DDP, where local_rank >= 0
    else:  
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    # Output hyper-settings
    print(f'Process rank: {args.local_rank}, device: {args.device}, local n_gpu: {n_gpu}, '
        f'distributed training: {args.local_rank != -1}, pretrain: {args.pretrain}')
    
    # 1. build model, must warp DDP model before define define optimizer
    net = utils.get_model(args)
    net = net.cuda()

    # whether apply EMA, initialize the shadow weights
    if args.local_rank in (-1, 0):
        args.model_ema = utils.ModelEMA(net, decay=args.ema_decay, device=args.device) if args.EMA else None
        print('Total learnable parameters: {:.2f}M'.format(utils.count_parameters(net)))      

    # DDP warper
    if args.local_rank != -1:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank], output_device=args.local_rank,
        )

    # 2. build dataloader
    train_loader, val_loader = get_loader(args)
    optimizer = utils.make_optimizer(args, net)
    args.num_iter_epoch = len(train_loader)
    
    scheduler = get_scheduler(optimizer, args)
    criterion = IQALoss(
        loss_type=args.loss_type, monotonicity_regularization=args.rank_regular, 
        plcc_regularization=args.plcc_regular,
        rank_weight=args.rank_weight, plcc_weight=args.plcc_weight,
    )

    if args.local_rank in (-1, 0):
        print('***** Running training *****')
        print('  Total optimization steps = {}'.format(args.epochs))
        print('  Instantaneous batch size per GPU = {}'.format(args.batch_size))
        print('  Total train batch size (w. parallel, distributed & accumulation) = {}'.format(
            args.batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
        ))

    # best_score = -1e6
    
    # Train & Validation
    for n_epoch in range(args.epochs):
        # Train
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(n_epoch)    # shuffle before every epoch
        
        utils.train_epoch(net, train_loader, criterion, optimizer, n_epoch, args, scheduler)    # scheduler update every iter

        # Main process saves model checkpoint
        if args.local_rank in (-1, 0):
            score = utils.validate_epoch(
                args.model_ema.module if args.EMA else net,
                val_loader, criterion, n_epoch, args
            )
    
            # save checkpoints
            # is_best = score > best_score
            # best_score = max(score, best_score)
            
            state = {
                'epoch': n_epoch,
                'arch': args.arch,
                'state_dict': args.model_ema.module.state_dict() if args.EMA else net.state_dict(),
                'score': score,
            }
            
            utils.save_checkpoint_epoch(ops.join('checkpoints', args.exp_name), state, is_best=False, n_epoch=n_epoch)

    return score


if __name__ == '__main__':
    # explicit for-loop to solve CUDA OOM problem
    args_bak = copy.deepcopy(args)    # the original args
    
    # for _ in range(150): 
    with open('resume_seeds.json', 'r') as f:
        seeds = json.load(f)
    
    for seed in seeds:
        args = copy.deepcopy(args_bak)    # create the tmp args file 

        # select the random seed 
        # with open('visited_seeds.json', 'r') as f:
        #     visited_seeds = json.load(f)
        
        # seed = args.seed
        # ## decouple the seed 
        # while seed in visited_seeds:
        #     seed = random.randint(0, 2**24-1)
        # visited_seeds.append(seed)
        # ## update the visited seeds
        # with open('visited_seeds.json', 'w') as f:
        #     json.dump(visited_seeds, f)
        args.seed = seed    # assign the mission seed 

        # update exp_name 
        date_str = time.strftime('%Y%m%d', time.localtime())
        args.exp_name = '_'.join((
            date_str, args.arch, args.dataset, 'r'+str(args.rsize)+'c'+str(args.csize), 
            args.optimizer, args.loss_type, str(args.epochs)+'e', 'seed'+str(args.seed), args.suffix, 
        ))

        if args.all_data:
            args.exp_name += '_alldata'
        args.log_dir = ops.join('logs', args.exp_name)

        # run the main process
        final_score = run(args)

        # date_str = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())
        # # record the results & start the next run
        # if final_score < 0.7513:
        #     with open('lottery_log.txt', 'a') as f:
        #         print(date_str + 'Seed {} failed and final score: {:.4f} and archived.'.format(args.seed, final_score), file=f)

        #     # archive this run & start new run
        #     mv_log_cmd = 'mv logs/{} archive/logs/{}'.format(args.exp_name, args.exp_name)
        #     mv_ckpt_cmd = 'mv checkpoints/{} archive/checkpoints/{}'.format(args.exp_name, args.exp_name)
        #     os.system(mv_log_cmd)
        #     os.system(mv_ckpt_cmd)

        # else:
        #     with open('lottery_log.txt', 'a') as f:
        #         print(date_str + 'Seed {} succeeds and final score: {:.4f}, please check!'.format(args.seed, final_score), file=f)
