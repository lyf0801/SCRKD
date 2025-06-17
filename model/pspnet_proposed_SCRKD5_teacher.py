
#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weight_init(module):
    for n, m in module.named_children():
        try:
            #print('initialize: '+n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, (nn.ReLU,nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.AvgPool2d, nn.Softmax,nn.Dropout2d)):
                pass
            else:
                m.initialize()
        except:
            pass

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out2)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)

class SOD_Head(nn.Module):
    def __init__(self,):
        super(SOD_Head, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
        )
        self.initialize()

    def forward(self, x):
        smaps = self.process(x)        
        return smaps

    def initialize(self):
        weight_init(self)

class FuseBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2):
        super(FuseBlock, self).__init__()
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2
        self.fuse = nn.Conv2d(self.in_channel1 + self.in_channel2, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
       
        self.initialize()
    def forward(self, x, y):
        out = F.relu(self.bn1(self.fuse(torch.cat((x,y), dim = 1))))
        return out

    def initialize(self):
        weight_init(self)


###CVPR2017 Pyramid Scene Parsing Network
class PPM(nn.Module): # pspnet
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(2048,down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.initialize()
    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))
    
    def initialize(self):
        weight_init(self)


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
"""
def structure_loss(pred, mask):
    #mask = mask.detach()
    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#

"""
IoU loss
"""
def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

import torch
import torch.nn.functional as F

def convert_to_probability_map(predictions):
    """
    将 Bx1xHxW 的显著预测图转换为 Bx2xHxW 的概率图
    :param predictions: Bx1xHxW 的显著预测图
    :return: Bx2xHxW 的概率图
    """
    # 复制一份预测图，用于表示非显著区域的概率
    non_salient = 1 - predictions
    # 拼接显著区域和非显著区域的概率图
    probability_map = torch.cat([predictions, non_salient], dim=1)
    # 使用 softmax 函数将其转换为概率分布
    probability_map = torch.softmax(probability_map, dim=1)
    return probability_map

def CriterionKD(pred, soft):
    '''
    knowledge distillation loss
    '''
    soft.detach()
    pred = torch.sigmoid(pred)
    soft = torch.sigmoid(soft)

    pred = convert_to_probability_map(pred)
    soft = convert_to_probability_map(soft)

    """
    KL loss
    """
    temperature = 1
    loss = nn.KLDivLoss(reduction='mean')(torch.log(pred / temperature), soft / temperature)
    return loss * temperature * temperature

def compute_multi_view_self_similarity(tensor):
    """
    计算 BxCxHxW 张量在 C、H、W 维度上的自相似性矩阵
    :param tensor: 输入的 BxCxHxW 张量
    :return: 通道、高度、宽度维度的自相似性矩阵
    """
    B, C, H, W = tensor.shape

    # 计算通道维度的自相似性矩阵
    channel_tensor = tensor.view(B, C, -1)  # 合并 H 和 W 维度
    channel_similarity = torch.zeros(B, C, C, device=device) #lyf: define GPU tensor
    for b in range(B):
        channel_similarity[b] = F.cosine_similarity(channel_tensor[b].unsqueeze(1), channel_tensor[b].unsqueeze(0), dim=-1)

    # 计算高度维度的自相似性矩阵
    height_tensor = tensor.permute(0, 2, 1, 3).contiguous().view(B, H, -1)  # 调整维度并合并 C 和 W 维度
    height_similarity = torch.zeros(B, H, H, device=device) #lyf: define GPU tensor
    for b in range(B):
        height_similarity[b] = F.cosine_similarity(height_tensor[b].unsqueeze(1), height_tensor[b].unsqueeze(0), dim=-1)

    # 计算宽度维度的自相似性矩阵
    width_tensor = tensor.permute(0, 3, 1, 2).contiguous().view(B, W, -1)  # 调整维度并合并 C 和 H 维度
    width_similarity = torch.zeros(B, W, W, device=device) #lyf: define GPU tensor
    for b in range(B):
        width_similarity[b] = F.cosine_similarity(width_tensor[b].unsqueeze(1), width_tensor[b].unsqueeze(0), dim=-1)

    return channel_similarity, height_similarity, width_similarity


import torch
import torch.nn as nn

# 定义一个解耦卷积模块
class DecoupledConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(DecoupledConvolution, self).__init__()
        # 定义 3 个卷积层
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(in_channels), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(in_channels), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(in_channels), nn.PReLU())

    def forward(self, x):
        # 分别通过 3 个卷积层
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        return out1, out2, out3

def resize_and_combine_features(features, target_size):
    """
    将多个多尺度特征调整到同一固定尺度，并组合成一个五维特征图
    :param features: 包含五个 BxCxHxW 特征的列表
    :return: 形状为 BxCxHxWx5 的特征图
    """


    resized_features = []
    for feature in features:
        # 使用双线性插值将特征调整到固定尺度
        resized_feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=True)
        resized_features.append(resized_feature)

    # 在新的维度上堆叠调整后的特征
    combined_features = torch.stack(resized_features, dim=-1)

    return combined_features



# 新颖相似性计算方法：计算每个聚合特征与教师特征的余弦相似度
def compute_similarity_weighted_fusion(student_aggregated_features, teacher_feature):
    """
    计算学生聚合特征与教师特征的相似性
    :param student_aggregated_features: 学生模型的聚合特征，形状为 BxCxHxWx5
    :param teacher_feature: 教师模型的特征，形状为 BxCxHxW
    :return: 相似性矩阵，形状为 Bx5
    """
    B, C, H, W, _ = student_aggregated_features.shape
    similarity_scores = []
    for i in range(5):
        # 提取第 i 个聚合特征
        student_feature = student_aggregated_features[..., i]

        # downsampling
        # if H >= 16:
        #     student_feature =  F.adaptive_avg_pool2d(student_feature, output_size=(16,16))
        #     teacher_feature =  F.adaptive_avg_pool2d(teacher_feature, output_size=(16,16))

        # 计算余弦相似度
        similarity = F.cosine_similarity(student_feature.view(B, -1), teacher_feature.view(B, -1), dim=1)
        similarity_scores.append(similarity)
    # 堆叠相似性得分
    similarity_matrix = torch.stack(similarity_scores, dim=1)

    """
    使用相似性矩阵对学生聚合特征进行加权融合
    :param student_aggregated_features: 学生模型的聚合特征，形状为 BxCxHxWx5
    :param similarity_matrix: 相似性矩阵，形状为 Bx5
    :return: 融合后的特征，形状为 BxCxHxW
    """
    # 对相似性矩阵进行 softmax 操作
    weights = F.softmax(similarity_matrix, dim=1)
    # 扩展权重维度以匹配聚合特征
    weights = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    # 加权融合
    fused_features = torch.sum(student_aggregated_features * weights, dim=-1)
    
    return fused_features

def hcl(fstudent, fteacher):
    
    
    n,c,h,w = fstudent.shape
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [16, 8, 4, 2, 1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
        
    return loss


class PSPNet_SCRKD(nn.Module):
    def __init__(self):
        super(PSPNet_SCRKD, self).__init__()
        self.teacher = PSPNet_teacher()
       
        self.teacher.eval()

        self.bkbone  = ResNet()
        self.ppm = PPM(down_dim=256)

        self.fuse5 = FuseBlock(in_channel1 = 2048,  in_channel2 = 256)
        self.fuse4 = FuseBlock(in_channel1 = 1024,  in_channel2 = 256)
        self.fuse3 = FuseBlock(in_channel1 = 512,  in_channel2 = 256)
        self.fuse2 = FuseBlock(in_channel1 = 256,  in_channel2 = 256)
        self.fuse1 = FuseBlock(in_channel1 = 64,  in_channel2 = 256)


        self.SOD_head1 = SOD_Head()
        self.SOD_head2 = SOD_Head()
        self.SOD_head3 = SOD_Head()
        self.SOD_head4 = SOD_Head()
        self.SOD_head5 = SOD_Head()

        self.decouple1 = DecoupledConvolution(in_channels=256)
        self.decouple2 = DecoupledConvolution(in_channels=256)
        self.decouple3 = DecoupledConvolution(in_channels=256)
        self.decouple4 = DecoupledConvolution(in_channels=256)
        self.decouple5 = DecoupledConvolution(in_channels=256)

        self.adaptive1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())


        self.att_conv1 = nn.Sequential(
                    nn.Conv2d(256*3, 3, kernel_size=1),
                    nn.Sigmoid(),
                )
        self.att_conv2 = nn.Sequential(
                    nn.Conv2d(256*3, 3, kernel_size=1),
                    nn.Sigmoid(),
                )
        self.att_conv3 = nn.Sequential(
                    nn.Conv2d(256*3, 3, kernel_size=1),
                    nn.Sigmoid(),
                )
        self.att_conv4 = nn.Sequential(
                    nn.Conv2d(256*3, 3, kernel_size=1),
                    nn.Sigmoid(),
                )
        self.att_conv5 = nn.Sequential(
                    nn.Conv2d(256*3, 3, kernel_size=1),
                    nn.Sigmoid(),
                )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.fusion_conv4 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.fusion_conv5 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
        )
        self.similarity_criterion = nn.L1Loss(reduction='mean')
        
        self.initialize()


    def forward(self, x_lr, x_hr=None, labels_lr=None, labels_hr=None):
        """
        baseline operations
        """
        s1,s2,s3,s4,s5 = self.bkbone(x_lr)
        s6 = self.ppm(s5)

        out5 =  self.fuse5(s5, s6)

        out4 =  self.fuse4(s4, F.interpolate(out5, size = s4.size()[2:], mode='bilinear',align_corners=True))

        out3  = self.fuse3(s3, F.interpolate(out4, size = s3.size()[2:], mode='bilinear',align_corners=True))

        out2  = self.fuse2(s2, F.interpolate(out3, size = s2.size()[2:], mode='bilinear',align_corners=True))

        out1  = self.fuse1(s1, F.interpolate(out2, size = s1.size()[2:], mode='bilinear',align_corners=True))
        """
        decoupeled three part
        """
        out51, out52, out53 = self.decouple5(out5)
        out41, out42, out43 = self.decouple4(out4)
        out31, out32, out33 = self.decouple3(out3)
        out21, out22, out23 = self.decouple2(out2)
        out11, out12, out13 = self.decouple1(out1)

        


        


    
        """
        SOD   attention fusion
        """
        att1 = self.att_conv1(torch.cat((out11, out12, out13), dim=1))
        n,_,h,w = out11.shape
        new_out1 = self.fusion_conv1((out11 * att1[:,0].view(n,1,h,w) + out12 * att1[:,1].view(n,1,h,w) + out13 * att1[:,2].view(n,1,h,w)))
        student_smap1 = self.SOD_head1(new_out1)
        att2 = self.att_conv2(torch.cat((out21, out22, out23), dim=1))
        n,_,h,w = out21.shape
        new_out2 = self.fusion_conv2((out21 * att2[:,0].view(n,1,h,w) + out22 * att2[:,1].view(n,1,h,w) + out23 * att2[:,2].view(n,1,h,w)))
        student_smap2 = self.SOD_head2(new_out2)
        att3 = self.att_conv3(torch.cat((out31, out32, out33), dim=1))
        n,_,h,w = out31.shape
        new_out3 = self.fusion_conv3((out31 * att3[:,0].view(n,1,h,w) + out32 * att3[:,1].view(n,1,h,w) + out33 * att3[:,2].view(n,1,h,w)))
        student_smap3 = self.SOD_head3(new_out3)
        att4 = self.att_conv4(torch.cat((out41, out42, out43), dim=1))
        n,_,h,w = out41.shape
        new_out4 = self.fusion_conv4((out41 * att4[:,0].view(n,1,h,w) + out42 * att4[:,1].view(n,1,h,w) + out43 * att4[:,2].view(n,1,h,w)))
        student_smap4 = self.SOD_head4(new_out4)
        att5 = self.att_conv5(torch.cat((out51, out52, out53), dim=1))
        n,_,h,w = out51.shape
        new_out5 = self.fusion_conv5((out51 * att5[:,0].view(n,1,h,w) + out52 * att5[:,1].view(n,1,h,w) + out53 * att5[:,2].view(n,1,h,w)))
        student_smap5 = self.SOD_head5(new_out5)
        ### interpolate
        student_smap1 = F.interpolate(student_smap1, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        student_smap2 = F.interpolate(student_smap2, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        student_smap3 = F.interpolate(student_smap3, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        student_smap4 = F.interpolate(student_smap4, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        student_smap5 = F.interpolate(student_smap5, size = x_lr.size()[2:], mode='bilinear', align_corners=True)


        return out1, out2, out3, out4, out5, student_smap1, student_smap2, student_smap3, student_smap4, student_smap5


       

    def initialize(self):
        weight_init(self)



if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = PSPNet_SCRKD()
    print(model(x)[4].size())