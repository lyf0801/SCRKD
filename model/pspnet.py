
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







class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
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

        self.initialize()

    def forward(self, x):
        """
        baseline operations
        """
        s1,s2,s3,s4,s5 = self.bkbone(x)
        s6 = self.ppm(s5)

        out5 =  self.fuse5(s5, s6)

        out4 =  self.fuse4(s4, F.interpolate(out5, size = s4.size()[2:], mode='bilinear',align_corners=True))

        out3  = self.fuse3(s3, F.interpolate(out4, size = s3.size()[2:], mode='bilinear',align_corners=True))

        out2  = self.fuse2(s2, F.interpolate(out3, size = s2.size()[2:], mode='bilinear',align_corners=True))

        out1  = self.fuse1(s1, F.interpolate(out2, size = s1.size()[2:], mode='bilinear',align_corners=True))
        
        """
        SOD 
        """
        smap1 = self.SOD_head1(out1)
        smap2 = self.SOD_head2(out2)
        smap3 = self.SOD_head3(out3)
        smap4 = self.SOD_head4(out4)
        smap5 = self.SOD_head5(out5)
        ### interpolate
        smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear', align_corners=True)
        smap2 = F.interpolate(smap2, size = x.size()[2:], mode='bilinear', align_corners=True)
        smap3 = F.interpolate(smap3, size = x.size()[2:], mode='bilinear', align_corners=True)
        smap4 = F.interpolate(smap4, size = x.size()[2:], mode='bilinear', align_corners=True)
        smap5 = F.interpolate(smap5, size = x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return out1, out2, out3, out4, out5, smap1, smap2, smap3, smap4, smap5
        else: #inference 
            return torch.sigmoid(smap1)


    def initialize(self):
        weight_init(self)



if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = PSPNet()
    print(model(x)[4].size())