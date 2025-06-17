"""
TGRS 2024 paper "Learning Cross-Modality High-Resolution Representation for Thermal Small-Object Detection"
https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
"""
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


    
class TransXNet_CMHRD(nn.Module):
    def __init__(self, img_size, s_gap):
        """
        s_gap = log2(teacher_size / student_size)
        """
        super(TransXNet_CMHRD, self).__init__()
        self.teacher = TransXNet_teacher(img_size*2)
       
        self.teacher.eval()

        self.student = TransXNet(img_size)

        # line 60 in https://github.com/NNNNerd/CMHRD/blob/main/mmdet/models/detectors/single_stage_cmhrd.py
        # lyf: change input channels for TransXNet [256, 256, 256, 256]
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
        self.criterion = nn.L1Loss(reduction='none').cuda()
        

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)

            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            
            student_out1, student_out2, student_out3, student_out4 = student_features
            teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features


            
            loss_sod = structure_loss(student_smap1, labels_lr)
            """
            print(teacher_features[0].size())
            print(teacher_features[1].size())
            print(teacher_features[2].size())
            print(teacher_features[3].size())
            print(teacher_features[4].size())
            torch.Size([8, 256, 112, 112])
            torch.Size([8, 256, 112, 112])
            torch.Size([8, 256, 112, 112])
            torch.Size([8, 256, 112, 112])
            """
            #SRG Loss
            loss_SRG = self.feat_sr_gen_kd_loss(student_features, teacher_features)
            #CMA Loss
            """
            self.s_gap = self.feature_distill['scale_gap']
            scale_gap=1, 
            referring to line 74 in https://github.com/NNNNerd/CMHRD/blob/main/cmhrd_configs/rfla_fcos_r50_fpn_p2_cmhrd_noaa.py
            由于学生尺度任意, scale_gap可以是1,2,3，因此复现代码采用上采样弥补教师-学生尺度差异
            """
            idx=[0,1,2]
            loss_CMA = 0
            for i in idx:
                loss_CMA += self.feat_kd_high_loss(self.adp_layer_high(student_features[i]), F.max_pool2d(teacher_features[i], 1, stride=2)) 
            # Response KD
            loss_KD = self.logits_distillation_loss(student_smap1, teacher_smap1)
            
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

        return loss * 0.1 #lyf: change for SOD task
    
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
        
        return loss * 0.02 #lyf: change for SOD task
    
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
    model = TransXNet_CMHRD()
    print(model(x)[4].size())