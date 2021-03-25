'''Some helper functions for PyTorch, including:
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import time


def cal_run_time(start_time):
    ss = int(time.time() - start_time)
    mm = 0
    hh = 0
    dd = 0
    if ss >= 60:
        mm = ss / 60
        ss %= 60
    if mm >= 60:
        hh = mm / 60
        mm %= 60
    if hh >= 24:
        dd = hh / 24
        hh %= 24

    str = ''
    if dd != 0:
        str += '%dD ' % dd
    if hh != 0:
        str += '%dh ' % hh
    if mm != 0:
        str += '%dm ' % mm
    if ss != 0:
        str += '%ds' % ss

    return str


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def set_lr_fusion(optimizer, lr_base, lr_logits):
    i_group = 0
    for group in optimizer.param_groups:
        i_group += 1
        if i_group == 1:
            group['lr'] = lr_base
        elif i_group == 2 or i_group == 3 or i_group == 4 or i_group == 5 or i_group == 6 or i_group == 7:
            group['lr'] = lr_logits
        else:
            print('Error!')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
