"""
TPAMI 2024 paper "Pixel Distillation: Cost-Flexible Distillation Across Image Sizes and Heterogeneous Networks"
https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
https://github.com/gyguo/PixelDistillation/blob/main/configs/ts/aircraft/aircraft_resnet50_resnet_isrd.yaml
According to Eq. (3) in paper, L_PD = L_PKD + \gamma * L_ISRD
"""
import torch
import torch.nn as nn




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


# class DistillKL(torch.nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, temp):
#         super(DistillKL, self).__init__()
#         self.temp = temp

#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s/self.temp, dim=1)
#         p_t = F.softmax(y_t/self.temp, dim=1)
#         loss = F.kl_div(p_s, p_t, size_average=False) * (self.temp**2) / y_s.shape[0]
#         return loss


# class KDLoss(nn.Module):
#     def __init__(self, cfg):
#         super(KDLoss, self).__init__()
#         self.alpha = cfg.KD.ALPHA
#         self.DistillKL = DistillKL(cfg.KD.TEMP)
#         self.cls_criterion = torch.nn.CrossEntropyLoss()

#     def forward(self, output_s, output_t, target):
#         cls_loss = self.cls_criterion(output_s, target)
#         kd_loss = self.DistillKL(output_s, output_t)
#         loss = (1 - self.alpha) * cls_loss + self.alpha * kd_loss
#         return loss

# https://github.com/gyguo/PixelDistillation/blob/main/lib/models/hrir.py
class SR1x1(nn.Module):
    """
    448 teacher, 224 student, scale_factor = 4
    448 teacher, 112 student, scale_factor = 8
    448 teacher, 56  student, scale_factor = 16
    """
    def __init__(self,  scale_factor = 4, inplanes=256):
        super(SR1x1, self).__init__()

        
        self.scale_factor = scale_factor

        self.outplanes = self.scale_factor ** 2 * 3
        self.inplanes = inplanes

        self.conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.prelu = nn.PReLU(3)

    def forward(self, feat_s):

        feat = self.conv(feat_s)

        image_sr = self.pixel_shuffle(feat)

        image_sr = self.prelu(image_sr)

        #if self.image_size % self.in_size == 0:
        return image_sr
        #else:
        #    return image_sr[:, :, 0:self.image_size, 0:self.image_size]
        
class SegFormer_PD(nn.Module):
    def __init__(self, img_size, scale_factor):
        """
        scale_factor = teacher_size / student_size * 2
        """
        super(SegFormer_PD, self).__init__()
        self.teacher = get_segformer_teacher(img_size=img_size)
       
        self.teacher.eval()

        self.student = get_segformer(img_size=img_size)


        # line 67 in https://github.com/gyguo/PixelDistillation/blob/main/tools_1_ts/train_isrd_5runs.py
        # self.pkd_criterion = KDLoss(cfg).to('cuda')#KDLoss等价于传统的KDloss
        self.isr_criterion = torch.nn.L1Loss(reduction='mean').to('cuda') 

        #line 49 in https://github.com/gyguo/PixelDistillation/blob/main/tools_1_ts/train_isrd_5runs.py
        # model_sr = SR1x1(cfg, list(feats_s[cfg.FSR.POSITION].shape))
        # from lib.models.hrir import SR1x1
        # cfg.FSR.POSITION = 0

        self.model_sr = SR1x1(scale_factor=scale_factor, inplanes=256) #

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            
            student_out1, student_out2, student_out3, student_out4 = student_features
            teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features

            loss_sod = structure_loss(student_smap1, labels_lr)
           
            
            # line 103 in https://github.com/gyguo/PixelDistillation/blob/main/lib/models/model_builder.py
            # image_sr = self.model_sr(feats_s[self.position])
            image_sr = self.model_sr(student_out1)
            
            # line 157 in https://github.com/gyguo/PixelDistillation/blob/main/tools_1_ts/train_isrd_5runs.py
            # isr_loss = isr_criterion(image_sr, input_large)
            # loss = kd_loss + isr_loss*cfg.FSR.ETA
            # cfg.FSR.ETA = 20 for image classification
            # lyf: as for RSI-SOD, cfg.FSR.ETA = 1 could be more match magnitude
            loss_ISR = self.isr_criterion(image_sr, x_hr) * 0.1 #balanced weights for SegFormer_PD
            # Response KD
            loss_KD = self.logits_distillation_loss(student_smap1,  teacher_smap1)
            
            return loss_sod, loss_ISR, loss_KD
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1
    

    
    #Response KD
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
    model = SegFormer_PD()
    print(model(x)[4].size())