import os
import sys
import json
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
from config import args


def run(args):
    # Only excecute in main process
    if args.level == 'img':
        import utils_single as utils
        from datasets_single import get_loader
    else:
        import utils_vid as utils
        from datasets_vid import get_loader
    
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

    best_score = -1e6
    
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
            is_best = score > best_score
            best_score = max(score, best_score)
            
            state = {
                'epoch': n_epoch,
                'arch': args.arch,
                'state_dict': args.model_ema.module.state_dict() if args.EMA else net.state_dict(),
                'score': score,
            }
            
            utils.save_checkpoint_epoch(ops.join('checkpoints', args.exp_name), state, is_best, n_epoch)

        # scheduler.step()


if __name__ == '__main__':
    run(args)
