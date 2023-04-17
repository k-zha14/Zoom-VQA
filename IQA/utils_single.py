import os
import sys
import os.path as ops
import time
import random
import shutil
import json
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import timm
from tqdm import tqdm
from scipy import stats
from copy import deepcopy
from PIL import Image

            
def set_random_seed(random_seed: int=0):
    """Set random seed to reproduce the training.     
    
    After Pytorch updates in 3/19/21, DDP has already made initial states the same across multi-gpus.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(random_seed)    # set gpu seed deterministic
        # fix convolution calculate methods
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(args):
    """Return desired model, including NR and FR model."""
    if args.arch == 'cpnet_multi':
        '''Multi-patch network, based on last feat of ConvNext & ViT.'''
        from models.CPNetMulti import CPNet
        net = CPNet(args)

    else:
        net = timm.create_model(args.arch, num_classes=1, pretrained=args.pretrain, drop_path_rate=args.drop_path)
    
    return net


class Timer(object):
    """Record multiple times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start and record the tik."""
        self.tik = time.time()

    def stop(self):
        """Store the elapsed time in the list and return the last time."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return self.sum() / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accumulator(object):
    """For accumulating sums over N variables, eg. top1-acc, loss.
    
    Attributes:
        data: A list of AverageMeter to record different indexes.
    """
    def __init__(self, n: int):
        self.n = n
        self.reset()

    def update(self, batch_size: int, *args):
        for i, vals in enumerate(zip(self.data, args)):
            a, b = vals
            if i < 2:
                a.update(b)    # for data_time, batch_time, n=1
            else:
                a.update(b, batch_size)

    def reset(self):
        self.data = [AverageMeter() for _ in range(self.n)] 

    def __getitem__(self, idx: int):
        return self.data[idx]


def save_checkpoint_epoch(prefix: str, state: dict, is_best: bool, n_epoch: int, n_iter=None, ext: str='checkpoint.pth.tar'):
    # make the diretory, store every epoch weights to lookup
    if not ops.exists(prefix):
        os.makedirs(prefix, exist_ok=True)
    
    # train & val after every epoch
    if n_iter is None:
        file_name = ops.join(prefix, '_'.join((str(n_epoch)+'epoch', ext)))
    # train & val after every freq iters
    else:
        file_name = ops.join(prefix, '_'.join((str(n_epoch)+'epoch', str(n_iter)+'iters', ext)))
    
    torch.save(state, file_name) 

    # still save the best ckp
    if is_best:
        best_name = '_'.join((prefix, 'best', ext))
        shutil.copyfile(file_name, best_name)


def train_epoch(net: nn.Module, train_loader: data.DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, n_epoch: int, args, scheduler=None):
    """Train the network on training dataset for one epoch.
    
    Args:
        net: The built network.
        train_loader: The training dataset, warpped by data.Dataloader.
        criterion: The loss function, eg. CrossEntropyLoss, l1_loss.
        optimizer: Parameters optimizer, eg. SGD, Adam.
        n_epoch: The number of epochs.
    """
    if hasattr(train_loader.batch_sampler, 'shuffle_lut'):
        # print('Shuffle train loader before every epoch!')
        train_loader.batch_sampler.shuffle_lut()

    records = Accumulator(3)    # including: data_time, elap_time, loss 
    net.train()    # switch to train mode, influence on Dropout and BN
    timer = Timer()    
    preds, labels = list(), list()    # record train srocc, plcc, main score and rmse

    epoch_iterator = tqdm(train_loader, bar_format="{l_bar}|[{elapsed}<{remaining}]", dynamic_ncols=True, disable=args.local_rank not in (-1, 0))

    for n_iter, batch in enumerate(epoch_iterator):
        if args.arch == 'cpnet_resol':
            batch_x, batch_y, batch_resols = tuple(t.to(args.device) for t in batch)
            # print('batch_resols:', batch_resols)
        else:
            batch_x, batch_y = tuple(t.to(args.device) for t in batch)
        
        # mixup
        if args.mixup < 1.0:
            batch_x, target_a, target_b, lam = mixup_data(batch_x, batch_y, args.mixup)

        # cutmix
        if args.cutmix > 0.0:
            batch_x, target_a, target_b, lam = cutmix_data(batch_x, batch_y, args.cutmix)

        # data loading time 
        data_time = timer.stop()
        timer.start()
        optimizer.zero_grad()
        
        # forward propagate
        if args.arch == 'cpnet_resol':
            output = net(batch_x, batch_resols)
        else:
            output = net(batch_x)

        # modify loss function to adapt to mixup&cutmix
        if args.mixup < 1.0 or args.cutmix > 0.0:
            loss = mix_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, batch_y)
        
        preds.append(output)
        labels.append(batch_y)

        # backward
        loss.backward()
        optimizer.step()

        # update the shadow weights
        if args.EMA:
            args.model_ema.update(net)    # update shadow weights
        
        # update scheduelr rate
        if scheduler is not None:
            scheduler.step()

        # compute indexes and record
        torch.cuda.synchronize()
        elapsed_time = timer.stop()
        timer.start()
        records.update(batch_y.size(0), data_time, elapsed_time, loss.detach().item())

        epoch_iterator.set_description(
            'Training epoch {}: {} / {}, lr: {:.5f}, loading time ratio: {:.3f} loss: {:.4f}'.format(
                n_epoch, n_iter + 1, len(train_loader), optimizer.param_groups[-1]['lr'], records[0].sum / records[1].sum, 
                records[2].avg
            )
        )

    preds, labels = torch.cat(preds).cpu().squeeze(), torch.cat(labels).cpu().squeeze()
    srocc, plcc, rmse = eval_preds(preds.detach(), labels.detach())
    lr, loss = [round(group['lr'], 8) for group in optimizer.param_groups], records[2].avg

    # save log files, actually these only statistic 1/n datset, not precise
    if args.local_rank in (-1, 0):
        args.writer.add_scalar('train/lr', scalar_value=lr[-1], global_step=n_epoch)
        args.writer.add_scalar('train/reg_loss', scalar_value=loss, global_step=n_epoch)
        args.writer.add_scalar('train/rmse', scalar_value=rmse, global_step=n_epoch)
        args.writer.add_scalar('train/srocc', scalar_value=srocc, global_step=n_epoch)
        args.writer.add_scalar('train/plcc', scalar_value=plcc, global_step=n_epoch)
    
        print('Epoch {}:, lr: {lr}, loading time ratio: {ratio:.4f}, loss: {loss:.5f}'.format(
            n_epoch, lr=lr, ratio=records[0].sum / records[1].sum, loss=loss
        ))
    

@torch.no_grad() 
def validate_epoch(net: nn.Module, val_loader: data.DataLoader, criterion: nn.Module, n_epoch: int, args, main_infer=True):
    """The definition of arguments are equal with the above train_epoch function.
    
    """
    records = Accumulator(3)    # including: data_time, elap_time, loss  
    preds, labels = list(), list()
    net.eval()

    timer = Timer()  
    for batch in tqdm(val_loader, total=len(val_loader)):
        if args.arch == 'cpnet_resol':
            batch_x, batch_y, batch_resols = tuple(t.to(args.device) for t in batch)
            print('batch_resols:', batch_resols)
        else:
            batch_x, batch_y = tuple(t.to(args.device) for t in batch)

        # data loading time 
        data_time = timer.stop()
        timer.start()
        
        if args.arch == 'cpnet_resol':
            output = net(batch_x, batch_resols)
        else:
            output = net(batch_x)

        loss = criterion(output, batch_y)

        preds.append(output)
        labels.append(batch_y)

        # compute indexes and record
        torch.cuda.synchronize()
        elapsed_time = timer.stop()
        timer.start()
        records.update(batch_y.size(0), data_time, elapsed_time, loss.item())

    # apply metrics, compute img-level and vid-level seperately
    preds, labels = torch.cat(preds).cpu().squeeze(), torch.cat(labels).cpu().squeeze()
    srocc, plcc, rmse = eval_preds(preds, labels, vid_frames=args.vid_frames)
    main_score = 0.5 * srocc + 0.5 * plcc   # delete plcc, srcc matters most

    # output log
    print(
        'Validate at {}, loss: {loss:.4f}, rmse: {rmse:.4f}, srocc: {srocc:.4f}, plcc:{plcc:.4f}, main_score:{main_score:.4f}'.format(
            n_epoch, loss=records[2].avg, srocc=srocc, plcc=plcc, main_score=main_score, rmse=rmse,
        )
    )

    # save log
    args.writer.add_scalar('val/frame_loss' if main_infer else 'val_n/frame_loss', scalar_value=records[2].avg, global_step=n_epoch)
    args.writer.add_scalar('val/vid_srocc' if main_infer else 'val_n/vid_srocc', scalar_value=srocc, global_step=n_epoch)
    args.writer.add_scalar('val/vid_plcc' if main_infer else 'val_n/vid_plcc', scalar_value=plcc, global_step=n_epoch)
    args.writer.add_scalar('val/vid_rmse' if main_infer else 'val_n/vid_rmse', scalar_value=rmse, global_step=n_epoch)
    args.writer.add_scalar('val/vid_main_score' if main_infer else 'val_n/vid_main_score', scalar_value=main_score, global_step=n_epoch)

    return main_score


def cal_metrics(sq, q):
    # calculate the metrics, eg. srcc, plcc, diff5, dif3, rmse 
    try:
        srocc = stats.spearmanr(sq, q)[0].item()
        plcc = stats.pearsonr(sq, q)[0].item()
        rmse = np.sqrt(np.mean((sq - q) ** 2)).item()
            
    except AttributeError:
        # save as json file, to reproduce the fault
        import json

        print('Oops, unexpected predictions and GT-labels, write to tmp/errs.json file!')
        with open('tmp/errs.json', 'w') as f:
            json.dump({'preds': sq.tolist(), 'labels': q.tolist()}, f)
        srocc, plcc, diff5, diff3, rmse = 0, 0, 0, 0, 0

    finally:
        return srocc, plcc, rmse


def eval_preds(preds: torch.Tensor, labels: torch.Tensor, vid_frames=None):
    """Metric video/imgs by SRCC, PLCC, and mse."""
    # 1. deal with images 
    if vid_frames == None:
        sq, q = preds.numpy(), labels.numpy()
        logs = cal_metrics(sq, q)

    # 2. deal with videos
    else:
        idx = 0
        vid_preds, vid_labels = list(), list()
        vid_frame_preds = list()

        for frame in vid_frames:
            vid_preds.append(preds[idx: idx + frame].mean().item())    # the whole
            vid_labels.append(labels[idx: idx + frame].mean().item())
            vid_frame_preds.append(preds[idx: idx + frame].tolist())
            idx += frame
        
        assert idx == sum(vid_frames), 'Unexpected EOF idx: {} {}'.format(idx, sum(vid_frames))
        sq, q = np.array(vid_preds), np.array(vid_labels)
        logs = cal_metrics(sq, q)

    return logs


def count_parameters(net: nn.Module):
    """Statistics the trainable parameters of model."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6


class ModelEMA(nn.Module):
    """Define model exponential moving average warpper."""
    def __init__(self, model, decay=0.999, device=None):
        super(ModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device   
        if self.device is not None:
            self.module.to(device=device)
        self.num_updates = 0

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        # Only average when stable
        decay = min(self.decay, (1 + self.num_updates)  / (2 + self.num_updates))
        # print('Update real EMA ratio:', decay)
        self._update(model, update_fn=lambda e, m: decay * e + (1. - decay) * m)
        self.num_updates += 1

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def make_optimizer(args, net):
    """Return selected optimizer instance."""
    # trainable = filter(lambda x: x.requires_grad, net.parameters())
    
    backbone_params = [value for name, value in net.named_parameters() if 'head' not in name and value.requires_grad]
    head_params = [value for name, value in net.named_parameters() if 'head' in name and value.requires_grad]
    print('Number of total params(by layers): {}, backbone params: {}, head params: {}'.format(
        len(list(net.parameters())), len(backbone_params), len(head_params)
    ))

    params = [
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_decay},
        {'params': head_params, 'lr': args.lr}
    ]

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSProp
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(params, **kwargs_optimizer)
    # optimizer = optimizer_class(trainable, **kwargs_optimizer)    # older verison, same LR for layers

    return optimizer


class cvtCV2Image:
    def __call__(self, img):
        return Image.fromarray(img[:,:, ::-1])    # BGR -> RGB


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=0.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


if __name__ == '__main__':
    tmp = [AverageMeter() for _ in range(5)]

    for i in range(5):
        print(id(tmp[i]))
