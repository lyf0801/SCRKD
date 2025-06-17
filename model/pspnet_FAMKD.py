"""
WACV 2024 paper: "Frequency Attention for Knowledge Distillation"
https://github.com/cuong-pv/FAM-KD
https://github.com/cuong-pv/FAM-KD/blob/main/distillers/FAM_KD.py
Note: need torch1.13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
import torch.nn.init as init
import torch.fft
import math


def feat_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        loss_all = loss_all + loss
    return loss_all



class CROSSATF(nn.Module):
    def __init__(self, in_channel, in_channel_x, mid_channel, out_channel, fuse,  out_shape):
        super(CROSSATF, self).__init__()
        
        if in_channel != mid_channel:
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )
            nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        else:
            self.conv1 = None

        if in_channel_x != mid_channel:
            self.conv1_x = nn.Sequential(
                    nn.Conv2d(in_channel_x, mid_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU()
            )
            nn.init.kaiming_uniform_(self.conv1_x[0].weight, a=1)  # pyre-ignore
        else:
            self.conv1_x = None

        if fuse:
            self.att_conv = AttentionConv(mid_channel, mid_channel)
        else:
            self.att_conv = None
        self.conv2 = nn.Sequential(
                FAM_Module(mid_channel, out_channel, out_shape),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, x_channel, h, w = x.shape
        # transform student features
        x_residual = x
        if self.conv1_x is not None:
            x = self.conv1_x(x)
            x_residual = x
        
        if self.att_conv is not None:
            # reduce channel dimension of residual features
            if self.conv1 is not None:
                y = self.conv1(y)
            # upsample residual features
            y = F.interpolate(y, x.size()[2:], mode="nearest")
            # fusion
            x = self.att_conv(x, y)
        
        y = self.conv2(x)
        return y, x




# source https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
       # print(out_channels)
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x, y):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(y)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class FAM_Module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(FAM_Module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes
      #  print(self.shapes)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
       # self.out_channels = feat_t_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        if isinstance(x, tuple):
            x, cuton = x
        else:
            cuton = 0.1
        batchsize = x.shape[0]
        x_ft = torch.fft.fftn(x, norm="ortho")
      #  print(x_ft.shape)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        batch_fftshift = batch_fftshift2d(out_ft)

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.
        batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
        # test with batch shift
        out_ft = batch_ifftshift2d(batch_fftshift)
        out_ft = torch.view_as_complex(out_ft)
        #Return to physical space
        """       
        #旧版          新版
        torch.fftn   torch.fft.fft2
        torch.ifftn  torch.fft.ifft2
        """
        out = torch.fft.ifftn(out_ft, s=(x.size(-2), x.size(-1)),norm="ortho").real
        out2 = self.w0(x)
        return self.rate1 * out + self.rate2*out2
    
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

def batch_fftshift2d(x):
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None)
            if i != axis else slice(0, n, None)
            for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
            if i != axis else slice(n, None, None)
            for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


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


class PSPNet_FAMKD(nn.Module):
    def __init__(self):
        super(PSPNet_FAMKD, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.student = PSPNet()

        #self.out_shapes = [4, 7, 14, 28, 56] #112 as student input
        self.out_shapes = [7, 14, 28, 56, 112] #224 as student input
        in_channels = [256, 256, 256, 256, 256]
        in_channels_x = [256, 256, 256, 256, 256]

        out_channels = [256, 256, 256, 256, 256]
        self.ce_loss_weight = 1.0
        self.famkd_loss_weight = 1.0


        self.temperature = 4
 

        self.warmup_epochs = 20
        self.stu_preact = False
        self.max_mid_channel = 256


        atfs = nn.ModuleList()
        # mid_channel = min(512, in_channels[-1])
        mid_channel = [256, 256, 256, 256, 256]
        # in_channels = [32, 64, 64,1]

        ## KD loss
        
        for idx, in_channel in enumerate(in_channels):
          #  print(idx)
            atfs.append(
                CROSSATF(
                    in_channel,
                    in_channels_x[idx],
                    mid_channel[idx],
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                    self.out_shapes[::-1][idx] 
                )
            )
        self.atfs = atfs[::-1]

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_out1, student_out2, student_out3, student_out4, student_out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5 = self.student(x_lr)
            with torch.no_grad():    
                teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5 = self.teacher(x_hr)
            
            # print(student_out1.size())
            # print(student_out2.size())
            # print(student_out3.size())
            # print(student_out4.size())
            # print(student_out5.size())
            # torch.Size([2, 256, 56, 56])
            # torch.Size([2, 256, 28, 28])
            # torch.Size([2, 256, 14, 14])
            # torch.Size([2, 256, 7, 7])
            # torch.Size([2, 256, 4, 4])
            """
            FAM forward
            """
            results = []
            out_features, res_features = self.atfs[0](student_out5)
            results.append(out_features)
            for features, abf in zip([student_out4, student_out3, student_out2, student_out1], self.atfs[1:]):
                out_features, res_features = abf(features, res_features)
                results.insert(0, out_features)
            # print(results[0].size())
            # print(results[1].size())
            # print(results[2].size())
            # print(results[3].size())
            # print(results[4].size())
            # torch.Size([2, 256, 56, 56])
            # torch.Size([2, 256, 28, 28])
            # torch.Size([2, 256, 14, 14])
            # torch.Size([2, 256, 7, 7])
            # torch.Size([2, 256, 4, 4])
            loss_distillation = self.distillation_loss(results, [teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5])
            loss1_1 = structure_loss(student_smap1, labels_lr)
            loss1_2 = structure_loss(student_smap2, labels_lr)
            loss1_3 = structure_loss(student_smap3, labels_lr)
            loss1_4 = structure_loss(student_smap4, labels_lr)
            loss1_5 = structure_loss(student_smap5, labels_lr)
            loss_sod = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)   
            return loss_sod, loss_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, results, teacher_outs):
        student_out1, student_out2, student_out3, student_out4, student_out5 = results
        teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5 = teacher_outs
        
        """
        align student and teacher feature maps
        """
        student_out1 = F.interpolate(student_out1, size = teacher_out1.size()[2:], mode='bilinear', align_corners=True)
        student_out2 = F.interpolate(student_out2, size = teacher_out2.size()[2:], mode='bilinear', align_corners=True)
        student_out3 = F.interpolate(student_out3, size = teacher_out3.size()[2:], mode='bilinear', align_corners=True)
        student_out4 = F.interpolate(student_out4, size = teacher_out4.size()[2:], mode='bilinear', align_corners=True)
        student_out5 = F.interpolate(student_out5, size = teacher_out5.size()[2:], mode='bilinear', align_corners=True)

        """
        five-stage level 
        """

        distillation_loss = feat_loss([student_out1, student_out2, student_out3, student_out4, student_out5], [teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5])

        return distillation_loss * self.famkd_loss_weight
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 112, 112)
    y = torch.Tensor(2, 3, 224, 224)
    z = torch.Tensor(2, 1, 112, 112)
    model = PSPNet_FAMKD()

    print(model(x,y,z).eval().size())