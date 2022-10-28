import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def my_adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    if epoch < 50:
        lr = 1e-4
    else:
        cur_epoch = epoch-50
        num_modify = num_epochs-50
        lr = base_lr * (1 - cur_epoch / num_modify) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def adjust_lr_4(optimizer, init_lr, epoch):
    if epoch < 30:
        new_lr = init_lr
    elif epoch < 60:
        new_lr = 5e-5
    elif epoch < 90:
        new_lr = 1e-5
    else:
        new_lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        lr = param_group['lr']
    return lr


def adjust_lr_base(optimizer, init_lr, epoch):
    if epoch < 25:
        new_lr = init_lr
    else:
        new_lr = 5e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        lr = param_group['lr']
    return lr


def adjust_lr_refine(optimizer, init_lr, epoch):
    if epoch < 20:
        new_lr = init_lr
    else:
        new_lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        lr = param_group['lr']
    return lr
# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))