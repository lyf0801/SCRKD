
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


class TransXNet_SRD(nn.Module):
    def __init__(self, img_size):
        super(TransXNet_SRD, self).__init__()
        self.teacher = TransXNet_teacher(img_size*2)
       
        self.teacher.eval()

        self.student = TransXNet(img_size)

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            student_out1, student_out2, student_out3, student_out4 = student_features
            teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features
            loss_feature_distillation = self.distillation_loss(student_out1, student_out2, student_out3, student_out4, teacher_out1, teacher_out2, teacher_out3, teacher_out4)
            loss_logits_distillation = self.logits_distillation_loss(student_smap1, teacher_smap1)
            loss_sod = structure_loss(student_smap1, labels_lr)

            return loss_sod, loss_feature_distillation, loss_logits_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_out1, student_out2, student_out3, student_out4, teacher_out1, teacher_out2, teacher_out3, teacher_out4):
        """
        https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/losses/mid_level.py line 1619

        Understanding the Role of the Projector in Knowledge Distillation, AAAI 2024
        """

        def feat_distill_loss(a, b):
            exponent=1.0
            return  torch.log(torch.abs(a - b).pow(exponent).sum())
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
        

        distillation_loss = feat_distill_loss(student_out1, teacher_out1) + feat_distill_loss(student_out2, teacher_out2) + feat_distill_loss(student_out3, teacher_out3) + feat_distill_loss(student_out4, teacher_out4)

        return distillation_loss * 0.005  #balanced weights
    
    def logits_distillation_loss(self, student_smap1, teacher_smap1):
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
    model = TransXNet_SRD()
    print(model(x)[4].size())
