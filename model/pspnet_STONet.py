"""
TGRS 2024 paper "STONet-S*: A Knowledge-Distilled Approach for Semantic Segmentation in Remote Sensing Images"
https://github.com/MAXHAN22/STONet
https://github.com/MAXHAN22/STONet/blob/main/train_net_kd_yph.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice




class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None, \
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher, gt, n_iter):
        if self.warmup_config:
            self.warmup(n_iter)
        if self.earlydecay_config:
            self.earlydecay(n_iter)

        if self.resize_config:
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        if self.transform_config:
            x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)

        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        loss = self.alpha * loss
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class feature_kd_loss(nn.Module):
    def __init__(self):
        super(feature_kd_loss, self).__init__()

 


    def forward(self, student_feature1, teacher_feature1):

        loss_feature = hcl(student_feature1, teacher_feature1)
        return loss_feature



import itertools
import numpy as np
import torch
import torch.nn as nn
import cv2


from torch.nn import functional as F
import torch.nn as nn
import torch

up_kwargs = {'mode': 'nearest'}






class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.75, stride=1, padding=1):
        super(OctaveConv2, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)

    def forward(self, x):
        X_h, X_l = x



        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.75, stride=1, padding=1):
        super(OctaveConv2, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stride = stride
        self.l2l = nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.l2h = nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2l = nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2h = nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)

    def forward(self, x):
        X_h, X_l = x[:,:64,:,:,], x[:,64:,:,:,]
        X_l = self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        """
        RuntimeError: The size of tensor a (6) must match the size of tensor b (7) at non-singleton dimension 3
        """
        if X_h2h.size()[2:] != X_l2h.size()[2:]:
            X_l2h = F.interpolate(X_l2h, size = X_h2h.size()[2:], mode='bilinear', align_corners=True)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l






class frequency_transfer(nn.Module):

    def __init__(self):
        super(frequency_transfer, self).__init__()

        
        self.fre_highandlow = OctaveConv2(kernel_size=(3, 3), in_channels=256, out_channels=256,  stride=1, alpha=0.75)

    def forward(self, x):
        
        X_h, X_l = self.fre_highandlow(x)

        return X_h, X_l




class frequency_kd2(nn.Module):


    def __init__(self):
        super(frequency_kd2, self).__init__()
        self.frequency_transfer = frequency_transfer()

        self.conv4_T = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv4_S = nn.Conv2d(256, 256, 3, 2, 1)


        self.conv3_T = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv3_S = nn.Conv2d(256, 256, 1, 1, 0)


        self.conv2_T = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv2_S = nn.Conv2d(256, 256, 1, 1, 0)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_T = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv1_S = nn.Conv2d(256, 256, 1, 1, 0)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


        # self.softmax = nn.Softmax()
        # self.linear4 = nn.Linear(256*4, 4)
        # self.linear3 = nn.Linear(256*3, 3)
        # self.linear2 = nn.Linear(256*2, 2)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        # self.creterion = KLDLoss()

    def forward(self, F4, F3, F2, F1, f4, f3, f2, f1):
        F4= self.conv4_T(F4)
        f4 = self.conv4_S(f4)

        F3 = self.conv3_T(F3)
        f3 = self.conv3_S(f3)

        F2 = self.up2(self.conv2_T(F2))
        f2 = self.up2(self.conv2_S(f2))

        F1 = self.up1(self.conv1_T(F1))
        f1 = self.up1(self.conv1_S(f1))



        F4_h, F4_l = self.frequency_transfer(F4)
        F3_h, F3_l = self.frequency_transfer(F3)
        F2_h, F2_l = self.frequency_transfer(F2)
        F1_h, F1_l = self.frequency_transfer(F1)

        f4_h, f4_l = self.frequency_transfer(f4)
        f3_h, f3_l = self.frequency_transfer(f3)
        f2_h, f2_l = self.frequency_transfer(f2)
        f1_h, f1_l = self.frequency_transfer(f1)

        # F4_h = F4_h.mean(1)
        # f4_h = f4_h.mean(1)


        # F4_h = F4_h.reshape(F4_h.shape[0], -1)
        # f4_h = f4_h.reshape(f4_h.shape[0], -1)

        # loss = torch.sqrt(torch.sum(torch.pow(F4_h - f4_h, 2), dim=1))

        loss1 = hcl(f1_h, F1_h) + hcl(f1_l, F1_l)
        loss2 = hcl(f2_h, F2_h) + hcl(f2_l, F2_l)
        loss3 = hcl(f3_h, F3_h) + hcl(f3_l, F3_l)
        loss4 = hcl(f4_h, F4_h) + hcl(f4_l, F4_l)

        # t = 2
        # F4_h = F.softmax(F4_h / t, dim=1)
        # F3_h = F.softmax(F3_h / t, dim=1)
        # F2_h = F.softmax(F2_h / t, dim=1)
        # F1_h = F.softmax(F1_h / t, dim=1)
        #
        # F4_l = F.softmax(F4_l / t, dim=1)
        # F3_l = F.softmax(F3_l / t, dim=1)
        # F2_l = F.softmax(F2_l / t, dim=1)
        # F1_l = F.softmax(F1_l / t, dim=1)
        #
        # f4_h = F.softmax(f4_h / t, dim=1)
        # f3_h = F.softmax(f3_h / t, dim=1)
        # f2_h = F.softmax(f2_h / t, dim=1)
        # f1_h = F.softmax(f1_h / t, dim=1)
        #
        # f4_l = F.softmax(f4_l / t, dim=1)
        # f3_l = F.softmax(f3_l / t, dim=1)
        # f2_l = F.softmax(f2_l / t, dim=1)
        # f1_l = F.softmax(f1_l / t, dim=1)
        # #
        # # # torch.Size([1, 64, 32, 32])
        # # # torch.Size([1, 192, 16, 16])
        # #
        # #
        # loss1 = (self.creterion(f1_h, F1_h, F1_h, 6) + self.creterion(f1_l, F1_l, F1_l, 6))* t * t
        # loss2 = (self.creterion(f2_h, F2_h, F1_h, 6) + self.creterion(f2_l, F2_l, F1_l, 6))* t * t
        # loss3 = (self.creterion(f3_h, F3_h, F1_h, 6) + self.creterion(f3_l, F3_l, F1_l, 6))* t * t
        # loss4 = (self.creterion(f4_h, F4_h, F1_h, 6) + self.creterion(f4_l, F4_l, F1_l, 6))* t * t



        loss = (loss1 + loss2 + loss3 + loss4)/4

        return loss

class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)

#!/usr/bin/python3
#coding=utf-8

import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .pspnet_teacher import PSPNet_teacher
from .pspnet import PSPNet

"""
  smaps : BCE + wIOU
  edges: BCE
"""
def structure_loss(pred, mask):
    #mask = mask.detach()
    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#


class PSPNet_STONet(nn.Module):
    def __init__(self):
        super(PSPNet_STONet, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.student = PSPNet()
        """
        define STONet distillation modules
        """
        self.frequency_kd2 = frequency_kd2().cuda()
        self.feature_kd_loss = feature_kd_loss().cuda()
        self.KLDLoss = KLDLoss().cuda()
        self.criterion_without =  MscCrossEntropyLoss().cuda()
    def forward(self, x_lr, x_hr=None, labels_lr=None, labels_hr=None):
       

        if self.training:
            student_out1, student_out2, student_out3, student_out4, student_out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5 = self.student(x_lr)
            with torch.no_grad():    
                teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5 = self.teacher(x_hr)
            
            loss1_1 = structure_loss(student_smap1, labels_lr)
            loss1_2 = structure_loss(student_smap2, labels_lr)
            loss1_3 = structure_loss(student_smap3, labels_lr)
            loss1_4 = structure_loss(student_smap4, labels_lr)
            loss1_5 = structure_loss(student_smap5, labels_lr)
            loss_sod = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)   

            student_out1 = F.interpolate(student_out1, size = teacher_out1.size()[2:], mode='bilinear', align_corners=True)
            student_out2 = F.interpolate(student_out2, size = teacher_out2.size()[2:], mode='bilinear', align_corners=True)
            student_out3 = F.interpolate(student_out3, size = teacher_out3.size()[2:], mode='bilinear', align_corners=True)
            student_out4 = F.interpolate(student_out4, size = teacher_out4.size()[2:], mode='bilinear', align_corners=True)
            student_out5 = F.interpolate(student_out5, size = teacher_out5.size()[2:], mode='bilinear', align_corners=True)
            student_smap1 = F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
            student_smap2 = F.interpolate(student_smap2, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
            student_smap3 = F.interpolate(student_smap3, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
            student_smap4 = F.interpolate(student_smap4, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)

            loss_FAKD = self.frequency_kd2(teacher_out5, teacher_out4, teacher_out3, teacher_out2, student_out5, student_out4, student_out3, student_out2)
            loss_DDKD = self.feature_kd_loss(teacher_out1, student_out1)

            loss_SRKD = (self.KLDLoss(student_smap2, teacher_smap1, labels_hr, 4) + self.KLDLoss(student_smap3, teacher_smap1, labels_hr, 4) + self.KLDLoss(student_smap4, teacher_smap1, labels_hr, 4)) / 3
            loss_DICE = (dice_loss(student_smap2, teacher_smap1) + dice_loss(student_smap3, teacher_smap1) + dice_loss(student_smap4, teacher_smap1)) / 3
            #loss0 = self.criterion_without(student_smap1, teacher_smap1.long())  #/pytorch/aten/src/THCUNN/SpatialClassNLLCriterion.cu:106: cunn_SpatialClassNLLCriterion_updateOutput_kernel: block: [1,0,0], thread: [985,0,0] Assertion `t >= 0 && t < n_classes` failed.
            loss_distillation = loss_FAKD + loss_DDKD + loss_SRKD + loss_DICE #+ loss0

            return loss_sod, loss_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

        





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 112, 112)
    y = torch.Tensor(2, 3, 224, 224)
    z = torch.Tensor(2, 1, 112, 112)
    model = PSPNet_STONet()

    print(model(x,y,z).eval().size())