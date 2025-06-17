
#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
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


class PSPNet_FitNet(nn.Module):
    def __init__(self):
        super(PSPNet_FitNet, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.student = PSPNet()

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_out1, student_out2, student_out3, student_out4, student_out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5 = self.student(x_lr)
            with torch.no_grad():    
                teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5 = self.teacher(x_hr)

            loss_distillation = self.distillation_loss(student_out1, student_out2, student_out3, student_out4, student_out5, teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5)
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

    def distillation_loss(self, student_out1, student_out2, student_out3, student_out4, student_out5, teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5):
        """
        Define FitNet KD loss
        MSE loss
        """

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
        MSEloss = nn.MSELoss()

        distillation_loss = MSEloss(student_out1, teacher_out1) + MSEloss(student_out2, teacher_out2) + MSEloss(student_out3, teacher_out3) + MSEloss(student_out4, teacher_out4) + MSEloss(student_out5, teacher_out5)

        return distillation_loss
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = PSPNet_FitNet()
    print(model(x)[4].size())