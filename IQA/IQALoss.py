import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsort

eps = 1e-8

class IQALoss(nn.Module):
    def __init__(self, loss_type='mse', multihead_used=False, **kwargs):
        super().__init__()
        self.multihead_used = multihead_used
        self.loss_func = LossFunc(loss_type, **kwargs)
        
    def forward(self, y_pred, y):
        # single-head mode, normal case
        if not self.multihead_used:
            return self.loss_func(y_pred, y)

        # multi-head mode
        else:
            loss = 0.
            for head in y_pred:
                loss += self.loss_func(head, y)

            return loss
                

class LossFunc(torch.nn.Module):
    def __init__(self, loss_type='mse', alpha=[1, 0], p=2, q=2, 
                 monotonicity_regularization=False, rank_weight=0.1, plcc_regularization=False, plcc_weight=0.02):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.p = p
        self.q = q
        self.monotonicity_regularization = monotonicity_regularization
        self.rank_weight = rank_weight
        self.detach = False
        self.plcc_regularization = plcc_regularization
        self.plcc_weight = plcc_weight
        
        # output log information
        if monotonicity_regularization:
            print(f'Rank loss is turned on, and weight: {self.rank_weight:.4f}')
        
        if plcc_regularization:
            print(f'PLCC loss is turned on, and weight: {self.plcc_weight:.4f}')

    def forward(self, y_pred, y):
        return self.loss_func(y_pred, y)

    def loss_func(self, y_pred, y):
        # l1 loss
        if self.loss_type == 'mae':
            loss = F.l1_loss(y_pred, y)
        elif self.loss_type == 'rank':
            loss = monotonicity_regularization(y_pred, y)   # default turn on margin as 0.5
        
        # smooth l1 loss, diff at 0.5
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(y_pred, y)

        elif self.loss_type == 'weighted_mse':
            loss = weighted_mse(y_pred, y)
        # l2 loss
        elif self.loss_type == 'mse':
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'norm-in-norm':
            loss = norm_loss_with_normalization(y_pred, y, alpha=self.alpha, p=self.p, q=self.q, detach=self.detach)
        elif self.loss_type == 'min-max-norm':
            loss = norm_loss_with_min_max_normalization(y_pred, y, alpha=self.alpha, detach=self.detach)
        elif self.loss_type == 'mean-norm':
            loss = norm_loss_with_mean_normalization(y_pred, y, alpha=self.alpha, detach=self.detach)
        elif self.loss_type == 'scaling':
            loss = norm_loss_with_scaling(y_pred, y, alpha=self.alpha, p=self.p, detach=self.detach)

        # same loss with Fast-VQA
        elif self.loss_type == 'fastvqa':
            loss =  plcc_loss(y_pred, y) + 0.5 * monotonicity_regularization(y_pred, y)
        else:
            loss = linearity_induced_loss(y_pred, y, self.alpha, detach=self.detach)
        
        # add monotonicity_regularization
        if self.monotonicity_regularization:
            # loss += self.rank_weight * monotonicity_regularization(y_pred, y)
            loss += self.rank_weight * soft_srcc_loss(y_pred, y)

        # add plcc loss
        if self.plcc_regularization:
            loss += self.plcc_weight * plcc_loss(y_pred, y)
    
        return loss  
 

def weighted_mse(y_pred, y, weights=[10, 7, 5, 3, 1, 1, 1, 1, 5, 10]):
    """Weighted loss on score range."""
    indices = [int( (t - 1e-7) *10 ) for t in y]
    w = torch.tensor([weights[i] for i in indices], dtype=y.dtype, device=y.device).unsqueeze(1)
    loss = ((y_pred - y)**2 * w).mean()

    return loss


def plcc_loss(y_pred, y):
    """Calculate plcc loss to maximum metrics"""
    if y_pred.size(0) > 1:
        return (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2
    else:
        return F.l1_loss(y_pred, y) 


def monotonicity_regularization(y_pred, y):
    """monotonicity regularization, or called rank loss"""
    if y_pred.size(0) > 1:  #
        ranking_loss = F.relu((y_pred-y_pred.t()) * torch.sign((y.t()-y)))
        # scale = 1 + torch.max(ranking_loss.detach())    # rescale in 0~1 scale
        scale = 1 + torch.max(ranking_loss)
        return torch.sum(ranking_loss) / y_pred.size(0) / (y_pred.size(0)-1) / scale
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def soft_srcc_loss(y_pred, y):
    """return soft srocc loss."""
    pred = torchsort.soft_rank(y_pred.permute(1, 0))
    target = torchsort.soft_rank(y.permute(1, 0))
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()

    return 1 - (pred * target).sum()    # Very Important: minus!


def linearity_induced_loss(y_pred, y, alpha=[1, 1], detach=False):
    """linearity-induced loss, actually MSE loss with z-score normalization"""
    if y_pred.size(0) > 1:  # z-score normalization: (x-m(x))/sigma(x).
        sigma_hat, m_hat = torch.std_mean(y_pred.detach(), unbiased=False) if detach else torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + eps)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + eps)
        scale = 4
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / scale  # ~ 1 - rho, rho is PLCC
        if alpha[1] > 0:
            rho = torch.mean(y_pred * y)
            loss1 = F.mse_loss(rho * y_pred, y) / scale  # 1 - rho ** 2 = 1 - R^2, R^2 is Coefficient of determination
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm"""
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  # very important!!
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
        # print('bhat = {}'.format(normalization.item()))
        y_pred = y_pred / (eps + normalization)  # very important!
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
            loss0 = torch.pow(loss0, p) if exponent else loss0 #
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #  
            err = rho * y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss1 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1 #  
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


def norm_loss_with_min_max_normalization(y_pred, y, alpha=[1, 1], detach=False):
    if y_pred.size(0) > 1:  
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - m_hat) / (eps + M_hat - m_hat)  # min-max normalization
        y = (y - torch.min(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y)
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) 
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

def norm_loss_with_mean_normalization(y_pred, y, alpha=[1, 1], detach=False):
    if y_pred.size(0) > 1:  
        mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - mean_hat) / (eps + M_hat - m_hat)  # mean normalization
        y = (y - torch.mean(y)) / (eps + torch.max(y) - torch.min(y))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

def norm_loss_with_scaling(y_pred, y, alpha=[1, 1], p=2, detach=False):
    if y_pred.size(0) > 1:  
        normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p) 
        y_pred = y_pred / (eps + normalization)  # mean normalization
        y = y / (eps + torch.norm(y, p=p))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


class UncertaintyLoss(torch.nn.Module):
    """The loss value varys according to the value of predicted MOS.

    Based on our observation, the variance of MOS is smaller when MOS approachs to the best(5. score) or 
    the worst(1. score) end, vice versa. So a vanilla idea is that model confidence or tolerance should comply
    with this phenomenon. Inspired by ICCV2021-Rotation Uncertainty loss, we propose the uncertainy loss.

    Params:
        theta: torch.tensor in [n, 1], 
        gamma: scalar, 
    """
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        pass


    def forward(y, y_pred):
        pass


if __name__ == '__main__':
    preds, labels = torch.rand((10, 1)), torch.rand((10, 1))
    mae_loss_torch = torch.nn.L1Loss()
    mse_loss_torch = torch.nn.MSELoss()
    mae_loss = IQALoss(loss_type='mae', monotonicity_regularization=False)
    mae_loss_res = IQALoss(loss_type='mae', monotonicity_regularization=True) 
    mse_loss = IQALoss(loss_type='mse', monotonicity_regularization=False)
    mse_loss_res = IQALoss(loss_type='mse', monotonicity_regularization=True)
    smooth_mae = IQALoss(loss_type='mse', monotonicity_regularization=True) 
    
    val1 = mae_loss(preds, labels)
    val2 = mae_loss_res(preds, labels)
    val3 = mse_loss(preds, labels)
    val4 = mse_loss_res(preds, labels)
    
    print('mae:', val1, type(val1), val2, mae_loss_torch(preds, labels))
    print('mse', val3, val4, mse_loss_torch(preds, labels))