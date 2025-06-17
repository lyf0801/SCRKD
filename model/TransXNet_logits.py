
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


class TransXNet_logits(nn.Module):
    def __init__(self, img_size):
        super(TransXNet_logits, self).__init__()
        self.teacher = TransXNet(img_size*2) #224x224
       
        self.teacher.eval()

        self.student = TransXNet(img_size)#112x112

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)

            loss_distillation = self.distillation_loss(student_smap1, teacher_smap1)
            loss_sod = structure_loss(student_smap1, labels_lr)

            return loss_sod, loss_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_smap1, teacher_smap1):
        """
        Define logits KD loss
        KL loss
        """

        """
        align student and teacher smaps
        """
        student_smap1 = F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)


        """
        five-stage level  [B, class, W, H]
        """
        Temp = 2
        KLloss = nn.KLDivLoss(reduction='batchmean')

        distillation_loss = KLloss(F.log_softmax(student_smap1.view(student_smap1.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap1.view(teacher_smap1.size(0), -1).detach() / Temp, dim=1)) 
                            
        return distillation_loss * 0.1
    





if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = TransXNet_logits()
    print(model(x)[4].size())