
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


class PSPNet_SRD(nn.Module):
    def __init__(self):
        super(PSPNet_SRD, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.student = PSPNet()

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_out1, student_out2, student_out3, student_out4, student_out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5 = self.student(x_lr)
            with torch.no_grad():    
                teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5 = self.teacher(x_hr)

            loss_feature_distillation = self.distillation_loss(student_out1, student_out2, student_out3, student_out4, student_out5, teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5)
            loss_logits_distillation = self.logits_distillation_loss(student_smap1, student_smap2, student_smap3, student_smap4, student_smap5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5)
            loss1_1 = structure_loss(student_smap1, labels_lr)
            loss1_2 = structure_loss(student_smap2, labels_lr)
            loss1_3 = structure_loss(student_smap3, labels_lr)
            loss1_4 = structure_loss(student_smap4, labels_lr)
            loss1_5 = structure_loss(student_smap5, labels_lr)
            loss_sod = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)   
            return loss_sod, loss_feature_distillation, loss_logits_distillation
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_out1, student_out2, student_out3, student_out4, student_out5, teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5):
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
        student_out5 = F.interpolate(student_out5, size = teacher_out5.size()[2:], mode='bilinear', align_corners=True)

        """
        five-stage level 
        """
        

        distillation_loss = feat_distill_loss(student_out1, teacher_out1) + feat_distill_loss(student_out2, teacher_out2) + feat_distill_loss(student_out3, teacher_out3) + feat_distill_loss(student_out4, teacher_out4) + feat_distill_loss(student_out5, teacher_out5)

        return distillation_loss * 0.005  #balanced weights
    
    def logits_distillation_loss(self, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5):
        """
        Define logits KD loss
        KL loss
        """

        """
        align student and teacher smaps
        """
        student_smap1 = F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
        student_smap2 = F.interpolate(student_smap2, size = teacher_smap2.size()[2:], mode='bilinear', align_corners=True)
        student_smap3 = F.interpolate(student_smap3, size = teacher_smap3.size()[2:], mode='bilinear', align_corners=True)
        student_smap4 = F.interpolate(student_smap4, size = teacher_smap4.size()[2:], mode='bilinear', align_corners=True)
        student_smap5 = F.interpolate(student_smap5, size = teacher_smap5.size()[2:], mode='bilinear', align_corners=True)

        """
        five-stage level  [B, class, W, H]
        """
        Temp = 2
        KLloss = nn.KLDivLoss(reduction='batchmean')

        distillation_loss = KLloss(F.log_softmax(student_smap1.view(student_smap1.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap1.view(teacher_smap1.size(0), -1).detach() / Temp, dim=1)) +\
                            KLloss(F.log_softmax(student_smap2.view(student_smap2.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap2.view(teacher_smap2.size(0), -1).detach() / Temp, dim=1)) /2 +\
                            KLloss(F.log_softmax(student_smap3.view(student_smap3.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap3.view(teacher_smap3.size(0), -1).detach() / Temp, dim=1)) /4 +\
                            KLloss(F.log_softmax(student_smap4.view(student_smap4.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap4.view(teacher_smap4.size(0), -1).detach() / Temp, dim=1)) /8 +\
                            KLloss(F.log_softmax(student_smap5.view(student_smap5.size(0), -1) / Temp, dim=1), F.softmax(teacher_smap5.view(teacher_smap5.size(0), -1).detach() / Temp, dim=1)) /16

        return distillation_loss * 0.1




if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = PSPNet_SRD()
    print(model(x)[4].size())
