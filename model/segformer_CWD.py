"""
ICCV 2021 paper "Channel-Wise Knowledge Distillation for Dense Prediction"
https://github.com/irfanICMLL/TorchDistiller/blob/main/SemSeg-distill/utils/criterion.py
"""
import torch
import torch.nn as nn
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap



class CriterionCWD(nn.Module):

    def __init__(self,norm_type='channel',divergence='kl',temperature=1.0):
    
        super(CriterionCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self,preds_S, preds_T):
        
        n,c,h,w = preds_S.shape
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)
#!/usr/bin/python3
#coding=utf-8

import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .segformer_teacher import get_segformer_teacher
from .segformer import get_segformer

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


class SegFormer_CWD(nn.Module):
    def __init__(self, img_size):
        super(SegFormer_CWD, self).__init__()
        self.teacher = get_segformer_teacher(img_size=img_size)
       
        self.teacher.eval()

        self.student = get_segformer(img_size=img_size)


    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            
            loss_distillation = self.distillation_loss(student_features, teacher_features)
            loss_sod = structure_loss(student_smap1, labels_lr)
           
       
            return loss_sod, loss_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_features, teacher_features):
        student_out1, student_out2, student_out3, student_out4 = student_features
        teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features
        """
        define criterion_cwd
        """
        criterion_cwd = CriterionCWD().cuda()
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

        distillation_loss = criterion_cwd(student_out1, teacher_out1) + criterion_cwd(student_out2, teacher_out2) + criterion_cwd(student_out3, teacher_out3) + criterion_cwd(student_out4, teacher_out4)

        return distillation_loss 
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = SegFormer_CWD()
    print(model(x)[4].size())