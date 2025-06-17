"""
TGRS 2024 paper "Learning Cross-Modality High-Resolution Representation for Thermal Small-Object Detection"
https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
"""
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


    
class PSPNet_CMHRD(nn.Module):
    def __init__(self, s_gap):
        """
        s_gap = log2(teacher_size / student_size)
        """
        super(PSPNet_CMHRD, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.student = PSPNet()

        # line 60 in https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
        self.adp_layer_high = nn.Conv2d(256, 256, 1, 1)

        # line 74 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
        self.s_gap = s_gap

        # line 52 in https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
        # self.sr_generation_layer = FeatureSRModule(neck['out_channels'], 2 ** self.s_gap)
        # self.sr_generation_layer = FeatureSRModule(in_channels=256, upscale_factor=2 ** self.s_gap) 
        # if self.method == 'bicubic':
        #    x = F.interpolate(x, scale_factor=self.upscale_factor)
        # lyf: if teacher size == 7, student size == 3, upscale_factor would not be a int, and thus we simply use F.interpilate(x, size=teacher.size()[2:]) to replace gen_gt_feat = self.sr_generation_layer(gt_feat)

        # line 64 in https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
        self.criterion = nn.L1Loss(reduction='none')
        

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_out1, student_out2, student_out3, student_out4, student_out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5 = self.student(x_lr)

            with torch.no_grad():    
                teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5 = self.teacher(x_hr)
            
            student_features = [student_out1, student_out2, student_out3, student_out4, student_out5]
            teacher_features = [teacher_out1, teacher_out2, teacher_out3, teacher_out4, teacher_out5]
            teacher_features.append(F.max_pool2d(teacher_features[-1], 1, stride=2))
            teacher_features.append(F.max_pool2d(teacher_features[-1], 1, stride=2))
            teacher_features.append(F.max_pool2d(teacher_features[-1], 1, stride=2))
            
            loss1_1 = structure_loss(student_smap1, labels_lr)
            loss1_2 = structure_loss(student_smap2, labels_lr)
            loss1_3 = structure_loss(student_smap3, labels_lr)
            loss1_4 = structure_loss(student_smap4, labels_lr)
            loss1_5 = structure_loss(student_smap5, labels_lr)
            loss_sod = loss1_1 + loss1_2 + (loss1_3 / 2) + (loss1_4 / 4) + (loss1_5 / 8)   
            
            #SRG Loss
            loss_SRG = self.feat_sr_gen_kd_loss(student_features, teacher_features)
            #CMA Loss
            """
            self.s_gap = self.feature_distill['scale_gap']
            scale_gap=1, 
            referring to line 74 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
            由于学生尺度任意, scale_gap可以是1,2,3，因此复现代码采用上采样弥补教师-学生尺度差异
            """
            idx=[2,3,4]
            loss_CMA = 0
            for i in idx:
                loss_CMA += self.feat_kd_high_loss(self.adp_layer_high(student_features[i]), teacher_features[i+self.s_gap]) 
            # Response KD
            loss_KD = self.logits_distillation_loss(student_smap1, student_smap2, student_smap3, student_smap4, student_smap5, teacher_smap1, teacher_smap2, teacher_smap3, teacher_smap4, teacher_smap5)
            
            return loss_sod, loss_SRG, loss_CMA, loss_KD
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1
    
    # SRG Loss
    """
    SRG self.sr_cfg['weight']=0.5
    referring to line 80 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
    """
    def feat_sr_gen_kd_loss(self, student_features, teacher_features):

        # line 145 in https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
        # if self.method == 'bicubic':
        loss = 0
        for i in range(len(student_features)):
            x = F.interpolate(student_features[i], size = teacher_features[i].size()[2:], mode='bicubic', align_corners=True)
            loss += torch.mean(self.criterion(x, teacher_features[i]))

        return loss * 0.5
    
    # CMA Loss
    """
    idx=[2,3,4]
    CMA weights = 0.01
    referring to line 77-78 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
    """
    def feat_kd_high_loss(self, student_features, teacher_features):

        bs, c, h, w = student_features.shape
        flatten_feat = student_features.reshape(bs, c, -1)
        flatten_teacher_feat = teacher_features.reshape(bs, c, -1)

        stu_aff = torch.bmm(flatten_feat.permute(0, 2, 1), flatten_feat)
        # line 76 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
        # type='cross',
        #if self.high_cfg['type'] == 'cross':
        tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_feat).detach()
        #else:
        #    tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_teacher_feat)

        loss = self.criterion(stu_aff, tea_aff).mean()
        
        return loss * 0.01
    
    #Response KD
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
    model = PSPNet_CMHRD()
    print(model(x)[4].size())