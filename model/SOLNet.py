"""
TGRS 2025 paper "Speed-Oriented Lightweight Salient Object Detection in Optical Remote Sensing Images"
https://github.com/SpiritAshes/SOLNet/blob/main/model/SOLNet.py
https://github.com/SpiritAshes/SOLNet/blob/main/model/Conv_component.py
https://github.com/SpiritAshes/SOLNet/blob/main/model/attention_component.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append('.')
from torch.nn import Parameter


from thop import profile
# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# --------------------------------------------------------

import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv2d(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 5
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[2]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[3]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[4]), num_blocks[3], stride=2)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        layers_result = []
        out = self.stage0(x)
        # out1 = self.stage1(out0)
        # out2 = self.stage2(out1)
        # out3 = self.stage3(out2)
        # out4 = self.stage4(out3)
        # layers_result.append(out0, out1, out2, out3, out4)
        layers_result.append(out)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
                    layers_result.append(out)
        # # out = self.gap(out)
        # # out = out.view(out.size(0), -1)
        # # out = self.linear(out)
        return out, layers_result


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_A1(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_A2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, kernel_num=4):
        super(Attention, self).__init__()
        self.in_channel = in_planes
        self.out_channel = out_planes
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1d_1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False) 
        self.relu = nn.ReLU(inplace=True)
        self.out_channel_conv2d = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)
        self.kernel_conv2d = nn.Conv2d(self.in_channel, kernel_num, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)

        x_inchannel = x.squeeze(-1).permute(0, 2, 1)
        x_inchannel = self.conv1d_1(x_inchannel)
        x_inchannel = x_inchannel.permute(0, 2, 1).unsqueeze(-1)
        x_inchannel = torch.sigmoid(x_inchannel)

        x_outchannel = x.squeeze(-1).permute(0, 2, 1)
        x_outchannel = self.conv1d_1(x_outchannel)
        x_outchannel = x_outchannel.permute(0, 2, 1).unsqueeze(-1)
        x_outchannel = self.out_channel_conv2d(x_outchannel)
        x_outchannel = torch.sigmoid(x_outchannel)

        x_kernel = x.squeeze(-1).permute(0, 2, 1)
        x_kernel = self.conv1d_1(x_kernel)
        x_kernel = x_kernel.permute(0, 2, 1).unsqueeze(-1)
        x_kernel = self.kernel_conv2d(x_kernel)
        x_kernel = x_kernel.view(x_kernel.size(0), -1, 1, 1, 1, 1)
        x_kernel = F.softmax(x_kernel, dim=1)

        return x_inchannel, x_outchannel, x_kernel

class EDE(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_num=1, kernel_size=3, stride=1, padding=0, groups=1, dilation=1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        in_channel_attention, out_channel_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.size()
        x = x * in_channel_attention
        x = x.reshape(1, -1, h, w)
        aggregate_weight = kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * b)
        output = output.view(b, self.out_planes, output.size(-2), output.size(-1))
        output = output * out_channel_attention
        return output

class LGA(nn.Module):
    def __init__(self, channels, k_size, reduction, groups):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2) 
        self.sigmoid = nn.Sigmoid()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels // 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // 2, 1, bias=False)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, inputs):
        input_1, input_2 = inputs.chunk(2, dim=1)

        x = self.avg_pool_1(input_1)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view(-1, self.channels // 2, 1, 1)
        x = input_1 * x.expand_as(input_1)

        max_pool = self.mlp(self.max_pool(input_2))
        avg_pool = self.mlp(self.avg_pool_2(input_2))
        channel_out = self.sigmoid(max_pool + avg_pool)
        y = channel_out * input_2
        # y = y.view(-1, self.input_channels // 2, 1, 1)

        x = torch.cat([x, y], dim=1)
        x = self.channel_shuffle(x, self.groups)
        return x


class PS(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.hidden_channels = in_channels // (scale * scale)
        self.out_channels = out_channels
        self.ps = nn.PixelShuffle(scale)

        # self.conv = Conv(self.hidden_channels, out_channels, ksize, 1)

    def forward(self, x):
        x = self.ps(x)
        # x = self.conv(x)
        _, _, h, w = x.data.size()
        x = x.view(-1, self.out_channels, h, w)
        return x

class DEAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_num, k_size, stride, padding, reduction=16, groups=4):
        super().__init__()

        self.groups = groups
        self.K = LGA(channels=in_channels, k_size=3, reduction=reduction, groups=groups)
        self.enhance_conv = EDE(in_channels, in_channels, kernel_num, k_size, 
                                               stride, padding, groups=1, dilation=1)

        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(-1, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, num_channels, height, width)
        return x

    def forward(self, x):
        K = self.K(x)
        Q = self.enhance_conv(x)
        x = Q + K
        x = self.bn(x)
        x = self.relu(x)
        return x
    
# RepVGG-A1-my-self
class SOLNet(nn.Module):
    def __init__(self, deploy=False, use_checkpoint=False):
        super(SOLNet, self).__init__()
        
        width_multiplier = [0.5, 0.75, 0.75, 0.75, 1]
        self.backbone = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000, width_multiplier=width_multiplier, override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)
        # self.K1 = LGA(channels=int(64 * width_multiplier[0]), k_size=3, reduction=16, groups=4)
        # self.K2 = LGA(channels=int(64 * width_multiplier[1]), k_size=3, reduction=16, groups=4)
        self.QK = DEAM(int(512 * width_multiplier[4]), int(512 * width_multiplier[4]), 1, 3, 1, 1)
        self.PS = nn.PixelShuffle(2)

        # self.Conv1 = nn.Conv2d(int(512 * width_multiplier[4]), int(512 * width_multiplier[4]) // 4, 3, 1, 1)
        self.Conv1 = nn.Conv2d(int(512 * width_multiplier[4]) // 4, int(512 * width_multiplier[4]) // 4, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(int(512 * width_multiplier[4]) // 4)

        self.Conv2 = nn.Conv2d(int(256 * width_multiplier[3])+ int(512 * width_multiplier[4]) // 4, int(128 * width_multiplier[2]), 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(int(128 * width_multiplier[2]))
        self.feature_1 = nn.Conv2d(int(128 * width_multiplier[2]), 1, 1, 1, 0)

        self.Conv3 = nn.Conv2d(int(128 * width_multiplier[2]), int(64 * width_multiplier[1]), 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(int(64 * width_multiplier[1]))
        self.feature_2 = nn.Conv2d(int(64 * width_multiplier[1]), 1, 1, 1, 0)

        self.Conv4 = nn.Conv2d(int(64 * width_multiplier[1]), int(64 * width_multiplier[0]), 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(int(64 * width_multiplier[0]))
        self.feature_3 = nn.Conv2d(int(64 * width_multiplier[0]), 1, 1, 1, 0)

        self.Conv5 = nn.Conv2d(int(64 * width_multiplier[0]), int(64 * width_multiplier[0]),3, 1, 1)
        self.bn5 = nn.BatchNorm2d(int(64 * width_multiplier[0]))
        self.feature_4 = nn.Conv2d(int(64 * width_multiplier[0]), 1, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        self.Up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.Up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        input = x
        layers_result = []
        x, layers_result = self.backbone(x)

        x = self.QK(x)
        # x = self.Conv1(x)
        x = self.PS(x)
        x = self.relu(self.bn1(self.Conv1(x)))
        """
        lyf add: RuntimeError: Sizes of tensors must match except in dimension 2. Got 7 and 8 (The offending index is 0)
        """
        if layers_result[20].size()[2:] != x.size()[2:]:
            layers_result[20] = F.interpolate(layers_result[20], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, layers_result[20]], dim=1)
        x = self.relu(self.bn2(self.Conv2(x)))


        feature_1 = self.feature_1(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, layers_result[6]], dim=1)
        """
        lyf add: RuntimeError: The size of tensor a (16) must match the size of tensor b (14) at non-singleton dimension 3
        """
        if layers_result[6].size()[2:] != x.size()[2:]:
            layers_result[6] = F.interpolate(layers_result[6], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x = x + layers_result[6]
        x = self.relu(self.bn3(self.Conv3(x)))

        feature_2 = self.feature_2(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, self.K2(layers_result[2])], dim=1)
        """
        lyf add: RuntimeError: The size of tensor a (32) must match the size of tensor b (28) at non-singleton dimension 3
        """
        if layers_result[2].size()[2:] != x.size()[2:]:
            layers_result[2] = F.interpolate(layers_result[2], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x = x + layers_result[2]
        x = self.relu(self.bn4(self.Conv4(x)))

        feature_3 = self.feature_3(self.drop(x))

        x = self.Up_2(x)
        # x = torch.cat([x, self.K1(layers_result[0])], dim=1)
        """
        lyf add: RuntimeError: The size of tensor a (64) must match the size of tensor b (56) at non-singleton dimension 3
        """
        if layers_result[0].size()[2:] != x.size()[2:]:
            layers_result[0] = F.interpolate(layers_result[0], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x = x + layers_result[0]
        x = self.relu(self.bn5(self.Conv5(x)))

        x = self.feature_4(self.drop(x))
        """
        lyf revise: RuntimeError: The size of tensor a (56) must match the size of tensor b (64) at non-singleton dimension 2
        """
        #feature_1 = self.Up_8(feature_1)
        #feature_2 = self.Up_4(feature_2)
        #feature_3 = self.Up_2(feature_3)
        feature_1 = F.interpolate(feature_1, size=input.size()[2:], mode='bilinear', align_corners=False)
        feature_2 = F.interpolate(feature_2, size=input.size()[2:], mode='bilinear', align_corners=False)
        feature_3 = F.interpolate(feature_3, size=input.size()[2:], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=False)

        if self.training:
            return feature_1, feature_2, feature_3, x, self.sigmoid(feature_1), self.sigmoid(feature_2), self.sigmoid(feature_3), self.sigmoid(x)
        else:
            return torch.sigmoid(x)

if __name__=='__main__':
    model = SOLNet(deploy=True)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"Ours---->>>>>\nFLOPs: {flops / (10 ** 9)}G\nParams: {params / (10 ** 6)}M")