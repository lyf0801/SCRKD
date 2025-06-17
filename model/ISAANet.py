"""
TGRS 2024 paper "Iterative Saliency Aggregation and Assignment Network for Efficient Salient Object Detection in Optical Remote Sensing Images"
https://github.com/YiuCK/ISAANet/blob/main/network.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2

class Fusion(nn.Module):
    def __init__(self, channel, ratio):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channel // 2, channel // 2 // ratio),
            nn.ReLU(),
            nn.Linear(channel // 2 // ratio, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))

    def forward(self, x, y):
        B, C, W, H = x.size()
        x1, x2 = torch.split(x, C // 2, 1)
        y1, y2 = torch.split(y, C // 2, 1)

        xy1 = self.conv1(torch.cat((x1, y1), 1))
        xy2 = self.conv2(torch.cat((x2, y2), 1))

        result = torch.where(xy1 > xy2, xy1, xy2)

        xx = xy1.view(xy1.shape[0], xy1.shape[1], -1)
        yy = xy2.view(xy2.shape[0], xy2.shape[1], -1)
        result = result.view(result.shape[0], result.shape[1], -1)
        #
        #
        sim1 = F.cosine_similarity(xx, result, dim=2)
        sim2 = F.cosine_similarity(yy, result, dim=2)

        a = self.mlp(sim1)
        b = self.mlp(sim2)

        wei = torch.cat((a, b), 1)
        w = F.softmax(wei, dim=1)
        w1, w2 = torch.split(w, 1, 1)
        z = self.conv3(torch.cat(
            (xy1 * w1.unsqueeze(2).unsqueeze(3).expand_as(xy1), xy2 * w2.unsqueeze(2).unsqueeze(3).expand_as(xy2)), 1))

        out = self.conv(z)
        return out


def get_open_map(sal, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations),
                        sal.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


def dilate(input, ksize=3):
    src_size = input.size()
    out = F.max_pool2d(input, kernel_size=ksize, stride=1, padding=0)
    out = F.interpolate(out, size=src_size[2:], mode="bilinear")
    return out


class SGR(nn.Module): #sal guidance refinement
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, x, y):
        y= F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)

        out = self.conv2(self.conv1(x*y+x))

        return out

class ISAANet(nn.Module):
    def __init__(self, cfg):
        super(ISAANet, self).__init__()
        self.backbone = timm.models.mobilenetv3_large_100(features_only=True, in_chans=3, pretrained=True,
                                                          pretrained_cfg=cfg)

        self.n1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.n2 = nn.Sequential(nn.Conv2d(24, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.n3 = nn.Sequential(nn.Conv2d(40, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.n4 = nn.Sequential(nn.Conv2d(112, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.n5 = nn.Sequential(nn.Conv2d(960, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))

        self.th = nn.Parameter(torch.ones(1, dtype=torch.float32).cuda(), requires_grad=True) * 0.2


        self.k1 = Fusion(32, 4)
        self.k2 = Fusion(32, 4)
        self.k3 = Fusion(32, 4)
        self.k4 = Fusion(32, 4)

        self.kk1 = Fusion(32, 4)
        self.kk2 = Fusion(32, 4)
        self.kk3 = Fusion(32, 4)
        self.kk4 = Fusion(32, 4)

        self.kkk1 = Fusion(32, 4)
        self.kkk2 = Fusion(32, 4)
        self.kkk3 = Fusion(32, 4)
        self.kkk4 = Fusion(32, 4)

        self.sal_sgr1 = SGR()
        self.sal_sgr2 = SGR()
        self.sal_sgr3 = SGR()
        self.edge_sgr1 = SGR()
        self.edge_sgr2 = SGR()

        self.sal_sgr1a = SGR()
        self.sal_sgr2a = SGR()
        self.sal_sgr3a = SGR()
        self.edge_sgr1a = SGR()
        self.edge_sgr2a = SGR()

        self.sal1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, kernel_size=1))
        self.sal2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, kernel_size=1))
        self.sal3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, kernel_size=1))


    def forward(self, img):
        feats = self.backbone(img)

        f1, f2, f3, f4, f5 = feats[0], feats[1], feats[2], feats[3], feats[4]
        ff1 = self.n1(f1)
        ff2 = self.n2(f2)
        ff3 = self.n3(f3)
        ff4 = self.n4(f4)
        ff5 = self.n5(f5)

        k4 = self.k4(ff4, F.interpolate(ff5, size=ff4.size()[2:], mode='bilinear', align_corners=True))
        k3 = self.k3(ff3, F.interpolate(k4, size=ff3.size()[2:], mode='bilinear', align_corners=True))

        k2 = self.k2(ff2, F.interpolate(k3, size=ff2.size()[2:], mode='bilinear', align_corners=True))
        k1 = self.k1(ff1, F.interpolate(k2, size=ff1.size()[2:], mode='bilinear', align_corners=True))
        sal2 = self.sal1(k1)

        keep = torch.ones_like(sal2)
        igore = torch.zeros_like(sal2)
        th = self.th

        guidance = torch.where(torch.sigmoid(sal2) >= th, keep, igore)

        dmap = dilate(guidance)
        edge = dmap - guidance
        edger = dilate(edge)

        r_ff1 = self.edge_sgr1(ff1, edger)
        r_ff2 = self.edge_sgr2(ff2, edger)
        r_ff3 = self.sal_sgr1(ff3, dmap)
        r_ff4 = self.sal_sgr2(ff4, dmap)
        r_ff5 = self.sal_sgr3(ff5, dmap)

        kk4 = self.kk4(r_ff4, F.interpolate(r_ff5, size=r_ff4.size()[2:], mode='bilinear', align_corners=True))
        kk3 = self.kk3(r_ff3, F.interpolate(kk4, size=r_ff3.size()[2:], mode='bilinear', align_corners=True))
        kk2 = self.kk2(r_ff2, F.interpolate(kk3, size=r_ff2.size()[2:], mode='bilinear', align_corners=True))
        kk1 = self.kk1(r_ff1, F.interpolate(kk2, size=r_ff1.size()[2:], mode='bilinear', align_corners=True))

        sal4 = self.sal2(kk1)

        keep1 = torch.ones_like(sal4)
        igore1 = torch.zeros_like(sal4)

        guidance1 = torch.where(torch.sigmoid(sal4) >= th, keep1, igore1)

        dmap1 = dilate(guidance1)
        edge1 = dmap1 - guidance1
        edge1r = dilate(edge1)

        r_ff1a = self.edge_sgr1a(r_ff1, edge1r)
        r_ff2a = self.edge_sgr2a(r_ff2, edge1r)
        r_ff3a = self.sal_sgr1a(r_ff3, dmap1)
        r_ff4a = self.sal_sgr2a(r_ff4, dmap1)
        r_ff5a = self.sal_sgr3a(r_ff5, dmap1)

        kkk4 = self.kkk4(r_ff4a, F.interpolate(r_ff5a, size=r_ff4a.size()[2:], mode='bilinear', align_corners=True))
        kkk3 = self.kkk3(r_ff3a, F.interpolate(kkk4, size=r_ff3a.size()[2:], mode='bilinear', align_corners=True))
        kkk2 = self.kkk2(r_ff2a, F.interpolate(kkk3, size=r_ff2a.size()[2:], mode='bilinear', align_corners=True))
        kkk1 = self.kkk1(r_ff1a, F.interpolate(kkk2, size=r_ff1a.size()[2:], mode='bilinear', align_corners=True))

        sal6 = self.sal3(kkk1)

        sal2 = F.interpolate(sal2, size=img.size()[2:], mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=img.size()[2:], mode='bilinear', align_corners=True)
        sal6 = F.interpolate(sal6, size=img.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return sal2, sal4, sal6
        else:
            return torch.sigmoid(sal6)