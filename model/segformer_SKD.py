"""
TPAMI 2020 paper "Structured Knowledge Distillation for Dense Prediction"
https://github.com/irfanICMLL/structure_knowledge_distillation/blob/ce208e1e5ba9177ecfc42519a2c64148d396fb71/networks/kd_model.py
https://github.com/irfanICMLL/structure_knowledge_distillation/blob/ce208e1e5ba9177ecfc42519a2c64148d396fb71/utils/train_options.py
"""
import torch
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, preprocess_GAN_mode, input_channel, conv_dim=64):
        super(Discriminator, self).__init__()
       
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        #layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(SpectralNorm(nn.Conv2d(input_channel, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

        if preprocess_GAN_mode == 1: #'bn':
            self.preprocess_additional = nn.BatchNorm2d(input_channel)
        elif preprocess_GAN_mode == 2: #'tanh':
            self.preprocess_additional = nn.Tanh()
        elif preprocess_GAN_mode == 3:
            self.preprocess_additional = lambda x: 2*(x/255 - 0.5)
        else:
            raise ValueError('preprocess_GAN_mode should be 1:bn or 2:tanh or 3:-1 - 1')

    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.preprocess_additional(x)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        #return [out.squeeze(), p1, p2]
        return [out, p1, p2]

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S.shape == d_in_T.shape,'the output dim of D with teacher and student as input differ'

        real_images = d_in_T
        fake_images = d_in_S
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out[0].size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss

class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S):
        g_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake

class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S.shape == d_out_T.shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake

class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        # N,C,W,H = preds_S.shape
        # softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        # logsoftmax = nn.LogSoftmax(dim=1)
        # loss = (torch.sum(- softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        """
        according to the original paper, PixelWise loss is KL loss
        """
        Temp = 2.0
        KLloss = nn.KLDivLoss(reduction='batchmean')
        loss = KLloss(F.log_softmax(preds_S.view(preds_S.size(0), -1) / Temp, dim=1), F.softmax(preds_T.view(preds_T.size(0), -1).detach() / Temp, dim=1))
        return loss

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss
        
#!/usr/bin/python3
#coding=utf-8

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


class SegFormer_SKD(nn.Module):
    def __init__(self, img_size):
        super(SegFormer_SKD, self).__init__()
        self.teacher = get_segformer_teacher(img_size=img_size)
       
        self.teacher.eval()

        self.student = get_segformer(img_size=img_size)

        """
        add Discriminator model
        """
        self.D_model = Discriminator(preprocess_GAN_mode=1, input_channel=1)
        """
        define distillation loss
        """
        self.criterion_pixel_wise = CriterionPixelWise()
        self.criterion_pair_wise_for_interfeat = CriterionPairWiseforWholeFeatAfterPool(scale=0.5, feat_ind=-4)
        self.criterion_adv = CriterionAdv(adv_type="wgan-gp")
        self.criterion_AdditionalGP = CriterionAdditionalGP(self.D_model, lambda_gp=10.0)
        self.criterion_adv_for_G = CriterionAdvForG(adv_type="wgan-gp")

    def forward(self, x_lr, x_hr=None, labels_lr=None):
       

        if self.training:
            student_smap1, student_features = self.student(x_lr)
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
            
            loss_distillation = self.distillation_loss(student_features, teacher_features, student_smap1, teacher_smap1)
            loss_sod = structure_loss(student_smap1, labels_lr)

            return loss_sod, loss_distillation, student_smap1, teacher_smap1
        else: #inference
            student_smap1 = self.student(x_lr) 
            return student_smap1

    def distillation_loss(self, student_outs, teacher_outs, student_smap1, teacher_smap1):
        student_out1, student_out2, student_out3, student_out4= student_outs
        teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_outs
        """
        define 
        """
        distillation_loss = 0.0
        
        """
        align student and teacher feature maps
        """
        student_out1 = F.interpolate(student_out1, size = teacher_out1.size()[2:], mode='bilinear', align_corners=True)
        student_out2 = F.interpolate(student_out2, size = teacher_out2.size()[2:], mode='bilinear', align_corners=True)
        student_out3 = F.interpolate(student_out3, size = teacher_out3.size()[2:], mode='bilinear', align_corners=True)
        student_out4 = F.interpolate(student_out4, size = teacher_out4.size()[2:], mode='bilinear', align_corners=True)

        student_smap1 = F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
        """
        five-stage level 
        """
        a = self.criterion_pixel_wise(student_smap1, teacher_smap1)
        b = self.criterion_pair_wise_for_interfeat([student_out1, student_out2, student_out3, student_out4], [teacher_out1, teacher_out2, teacher_out3, teacher_out4])
        c = self.criterion_adv_for_G(self.D_model(student_smap1)[0])
        print("a:")
        print(a)
        print("b:")
        print(b)
        print("c:")
        print(c)

        distillation_loss += 0.1 * a  #scale factor for segformer
        distillation_loss += 1.0 * b
        distillation_loss += 0.0001 * c
        return distillation_loss 
    
    def Discriminator_backward(self, student_smap1, teacher_smap1):
        d_loss = 0.0

        student_smap1 = F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True)
        d_out_teacher = self.D_model(teacher_smap1)
        d_out_student = self.D_model(student_smap1)
        d_loss += 0.1 * self.criterion_adv(d_out_student[0].detach(), d_out_teacher[0].detach())
        d_loss += 0.1 * self.criterion_AdditionalGP(student_smap1, teacher_smap1)
        return d_loss * 0.1 # scale factor for segformer
  
if __name__ == '__main__':
    x = torch.Tensor(8, 1, 224, 224)
    model = Discriminator(preprocess_GAN_mode=1, input_channel=1)
    print(model)
    print(model(x)[0].size())
    print(model(x)[1].size())
    print(model(x)[2].size())