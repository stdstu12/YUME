#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 20:29:36

import math
import torch
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
from copy import deepcopy

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt):
    if list(model.state_dict().keys())[0].startswith('module.'):
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'module.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    model.load_state_dict(ckpt, True)

def compute_hinge_loss(real_output, fake_output, x_start_, r1_lambda):
    if r1_lambda == 0:
        real_loss_total = torch.relu(torch.ones_like(real_output) - real_output).mean()
        fake_loss_total = torch.relu(torch.ones_like(fake_output) + fake_output).mean()

    else:
        real_loss_ = torch.relu(torch.ones_like(real_output) - real_output).mean()

        # 计算真实样本的梯度
        grad_real = torch.autograd.grad(outputs=real_output.sum(), inputs=x_start_, create_graph=True)[0]

        # 计算梯度惩罚
        grad_penalty = (grad_real.contiguous().view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean() * r1_lambda

        real_loss_total = real_loss_ + grad_penalty
        fake_loss_total = torch.relu(torch.ones_like(fake_output) + fake_output).mean()

    real_loss = real_loss_total
    fake_loss = fake_loss_total

    loss_d = real_loss + fake_loss

    return loss_d



def reload_model_(model, ckpt):
    if list(model.state_dict().keys())[0].startswith('model.'):
        if list(ckpt.keys())[0].startswith('model.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'model.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('model.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    model.load_state_dict(ckpt, True)



def reload_model_IDE(model, ckpt):
    extracted_dict = OrderedDict()
    for key, value in ckpt.items():
        if key.startswith('E_st'):
            new_key = key.replace('E_st.', '')
            extracted_dict[new_key] = value

    model.load_state_dict(extracted_dict, True)



class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
