
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


class TransXNet_FitNet(nn.Module):
    def __init__(self, img_size):
        super(TransXNet_FitNet, self).__init__()
        self.teacher = TransXNet_teacher(2*img_size)
       
        self.teacher.eval()

        self.student = TransXNet(img_size)

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
        """
        Define FitNet KD loss
        MSE loss
        """
        student_feature1, student_feature2, student_feature3, student_feature4 = student_features
        teacher_feature1, teacher_feature2, teacher_feature3, teacher_feature4 = teacher_features
        """
        align student and teacher feature maps
        """
        student_feature1 = F.interpolate(student_feature1, size = teacher_feature1.size()[2:], mode='bilinear', align_corners=True)
        student_feature2 = F.interpolate(student_feature2, size = teacher_feature2.size()[2:], mode='bilinear', align_corners=True)
        student_feature3 = F.interpolate(student_feature3, size = teacher_feature3.size()[2:], mode='bilinear', align_corners=True)
        student_feature4 = F.interpolate(student_feature4, size = teacher_feature4.size()[2:], mode='bilinear', align_corners=True)
        

        """
        five-stage level 
        """
        MSEloss = nn.MSELoss()

        distillation_loss = MSEloss(student_feature1, teacher_feature1) + MSEloss(student_feature2, teacher_feature2) + MSEloss(student_feature3, teacher_feature3) + MSEloss(student_feature4, teacher_feature4) 

        return distillation_loss / 4
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = TransXNet_FitNet()
    print(model(x)[4].size())