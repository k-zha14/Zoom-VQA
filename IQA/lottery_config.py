"""
    All of the parameters are defined here.
"""

import os
import json
import time
import random
import os.path as ops
from argparse import ArgumentParser


parser = ArgumentParser(description='2022 Mongo Challenge')

# Model Hyper-parameters
parser.add_argument('--arch', type=str, default='triq', help='the name of model structure')
parser.add_argument('--drop_path', type=float, default=0.0, help='stochastic depth ratio, default 0.0')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate of patch regression head, default 0.1')
parser.add_argument('--level', type=str, default='img', choices=['img', 'vid'] , help='decide the training level')
parser.add_argument('--seed', type=int, default='42', help='global random seed')

# Dataset 
parser.add_argument('--dataset', type=str, default='mgtv', help='path to dataset config')
parser.add_argument('--rsize', type=int, default=256, help='height of a patch to crop')
parser.add_argument('--csize', type=int, default=224, help='width of a patch to crop')

# Training Settings
parser.add_argument('--loss_type', type=str, default='mse', help='the loss function')
parser.add_argument('--plcc_regular', action='store_true', help='whether apply plcc regularization')
parser.add_argument('--plcc_weight', type=float, default=0.02, help='plcc loss weight for mgtv')
parser.add_argument('--rank_regular', action='store_true', help='whether apply rank regularization')
parser.add_argument('--rank_weight', type=float, default=0.1, help='rank loss weight for mgtv')
parser.add_argument('--os_rank_weight', type=float, default=1.0, help='rank loss weight for opensource dataset')

parser.add_argument('--pretrain', action='store_true', help='whether load pretrained weights')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--aux_batch_size', type=int, default=8, help='batch size for multi-dataset training on auxillary dataset')


# Optimizer
parser.add_argument('--scheduler', type=str, default='cosine', help='choose scheduler')
parser.add_argument('--warm_up', type=int, default=0, help='No. of warm up epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop','ADAMW'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

# Logs&Tricks
parser.add_argument("--suffix", type=str, default='base', help="suffix appendix to add flag")
parser.add_argument('--EMA', action='store_true', help='whether apply EMA weights trick')
parser.add_argument('--ema_decay', type=float, default=0.999, help='the EMA decay ratio')
parser.add_argument('--backbone_lr_decay', type=float, default=1., help='decay the learning rate of backbone network.')
parser.add_argument('--gpu', type=str, default='0,1', help='visiable gpu cards')
parser.add_argument("--mixup", type=float, default=1.0, help="mixup ratio")
parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix ratio")
parser.add_argument('--all_data', action='store_true', help='whether apply all labelled data')
parser.add_argument("--aux_weight", type=float, default=1.0, help="weight for auxillary dataset rank loss")


args = parser.parse_args()
# args = parser.parse_args(['--data_dir', '../data/train', '--data_list', 'data_train.list'])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# After torch 1.10.0, torchrun set '--use_env' as default
args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1

# Loose the random seed
if args.seed == -1:
    args.seed = random.randint(0, 2**24-1)