"""
CVPR 2021 paper "Distilling Knowledge via Knowledge Review"
https://github.com/dvlab-research/ReviewKD/blob/master/Detection/model/kd_trans.py
"""
import torch
from torch import nn
import torch.nn.functional as F

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y, x

class ReviewKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel
    ):
        super(ReviewKD, self).__init__()

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, student_features):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            results.insert(0, out_features)

        return results


def build_kd_trans():
    in_channels = [256,256,256,256,256]
    out_channels = [256,256,256,256,256]
    mid_channel = 256
    model = ReviewKD(in_channels, out_channels, mid_channel)
    return model

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .TransXNet_teacher import TransXNet_teacher
from .TransXNet import TransXNet

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


class TransXNet_ReviewKD(nn.Module):
    def __init__(self, img_size):
        super(TransXNet_ReviewKD, self).__init__()
        self.teacher = TransXNet_teacher(img_size*2)
       
        self.teacher.eval()

        self.student = TransXNet(img_size)

        self.reviewKD = build_kd_trans()

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            student_out1, student_out2, student_out3, student_out4 = self.reviewKD(student_features)
            teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features
            loss_distillation = self.distillation_loss(student_out1, student_out2, student_out3, student_out4, teacher_out1, teacher_out2, teacher_out3, teacher_out4)
            loss_sod = structure_loss(student_smap1, labels_lr)
 
            return loss_sod, loss_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_out1, student_out2, student_out3, student_out4, teacher_out1, teacher_out2, teacher_out3, teacher_out4):

        """
        align student and teacher feature maps
        """
        student_out1 = F.interpolate(student_out1, size = teacher_out1.size()[2:], mode='bilinear', align_corners=True)
        student_out2 = F.interpolate(student_out2, size = teacher_out2.size()[2:], mode='bilinear', align_corners=True)
        student_out3 = F.interpolate(student_out3, size = teacher_out3.size()[2:], mode='bilinear', align_corners=True)
        student_out4 = F.interpolate(student_out4, size = teacher_out4.size()[2:], mode='bilinear', align_corners=True)
        

        """
        five-stage level 
        """

        distillation_loss = hcl([student_out1, student_out2, student_out3, student_out4], [teacher_out1, teacher_out2, teacher_out3, teacher_out4]) 

        return distillation_loss 
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = TransXNet_ReviewKD()
    print(model(x)[4].size())