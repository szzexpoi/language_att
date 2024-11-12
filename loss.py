import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import pi
import math

epsilon = 1e-7 

def NSS(input, fixation, reduce=True):    
    if len(input.shape)==3:
        input = input.view(input.size(0), -1)
        fixation = fixation.view(fixation.size(0), -1)
    else:
        input = input.view(input.size(0)*input.size(1), -1)
        fixation = fixation.view(fixation.size(0)*fixation.size(1), -1)
    input = torch.div(input, input.max(-1,keepdim=True)[0].expand_as(input)+epsilon)
    input = torch.div(input-input.mean(-1, keepdim=True).expand_as(input),input.std(-1, keepdim=True).expand_as(input) + epsilon)
    loss = torch.div(torch.mul(input, fixation).sum(-1), fixation.sum(-1) + epsilon)

    if reduce:
        return torch.mean(loss)
    else:
        return loss

def CC(input, fixmap, reduce=True): 
    if len(input.shape)==3:
        input = input.view(input.size(0), -1)
        fixmap = fixmap.view(fixmap.size(0), -1)
    else:
        input = input.view(input.size(0)*input.size(1), -1)
        fixmap = fixmap.view(fixmap.size(0)*fixmap.size(1), -1)

    input = torch.div(input,input.max(-1, keepdim=True)[0].expand_as(input)+epsilon)
    fixmap = torch.div(fixmap,fixmap.sum(-1, keepdim=True).expand_as(fixmap)+epsilon)
    input = torch.div(input,input.sum(-1, keepdim=True).expand_as(input)+epsilon)

    sum_prod = torch.mul(input, fixmap).sum(-1,keepdim=True)
    sum_x = input.sum(-1, keepdim=True)
    sum_y = fixmap.sum(-1, keepdim=True)
    sum_x_square = (input**2).sum(-1, keepdim=True)
    sum_y_square = (fixmap**2).sum(-1, keepdim=True)
    num = sum_prod - torch.mul(sum_x, sum_y)/input.size(-1)
    den = torch.sqrt((sum_x_square-sum_x**2/input.size(-1))*(sum_y_square-sum_y**2/input.size(-1)))
    loss = torch.div(num, den+epsilon)

    if reduce:
        return torch.mean(loss)
    else:
        return loss

def KLD(input, fixmap):
    if len(input.shape)==3:
        input = input.view(input.size(0), -1)
        fixmap = fixmap.view(fixmap.size(0), -1)
    else:
        input = input.view(input.size(0)*input.size(1), -1)
        fixmap = fixmap.view(fixmap.size(0)*fixmap.size(1), -1)
    input = torch.div(input,input.max(-1, keepdim=True)[0].expand_as(input)+epsilon)
    fixmap = torch.div(fixmap,fixmap.sum(-1, keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1, keepdim=True).expand_as(input)+epsilon)
    loss = torch.mul(fixmap,torch.log(torch.div(fixmap, input+epsilon) + epsilon)).sum(-1)

    return torch.mean(loss)


def LL(input, fixmap):
    input = input.view(input.size(0), -1)
    input = F.softmax(input, dim=-1)
    fixmap = fixmap.view(fixmap.size(0),-1)
    loss =  torch.mul(torch.log(input+epsilon),fixmap).sum(-1)

    return torch.sum(loss)


def cross_entropy(input, target):
    input = input.view(input.size(0), -1)
    input = F.softmax(input, dim=-1)
    target = target.view(target.size(0), -1)
    loss = (-target*torch.log(torch.clamp(input, min=epsilon, max=1))).sum(-1)
    loss = torch.mean(loss)
    return loss.mean()

def attention_entropy(att, valid_len):
    threshold = 1.5 # originally 1.5
    entropy = -att*torch.log(torch.clamp(att, min=1e-7))
    b, seq = entropy.shape
    binary_mask = torch.ones(b, seq).bool().cuda()
    for i in range(b):
        binary_mask[i, valid_len[i]:] = 0
    loss = torch.relu((entropy*binary_mask).sum(-1)-threshold).mean()
    return loss