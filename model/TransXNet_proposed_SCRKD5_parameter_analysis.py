"""
TNNLS 2025 Paper "TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition"
https://github.com/LMMMEng/TransXNet/tree/main/semantic_segmentation

pip install mmcv==1.7.1

#lyf add pretrained weights: https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar
pretrained="./transx-s.pth.tar",  
"""
import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from timm.models.layers import DropPath, to_2tuple
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from .TransXNet import *
from .TransXNet_teacher import TransXNet_teacher

class TransXNet_SCRKD5(nn.Module):
    """
    liuyanfeng99@whu.edu.cn

    def transxnet_s(pretrained=False, init_cfg=None, **kwargs):
        if pretrained:
            init_cfg=dict(type='Pretrained', 
                        checkpoint='https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar',)
        model = TransXNet(arch='s', fork_feat=True, init_cfg=init_cfg, **kwargs)
        return model
    """
    """
    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``arch_settings``. And if dict, it
            should include the following two keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - mlp_ratios (list[int]): Expansion ratio of MLPs.
            - layer_scale_init_value (float): Init value for Layer Scale.

            Defaults to 'tiny'.

        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        in_patch_size (int): The patch size of input image patch embedding.
            Defaults to 7.
        in_stride (int): The stride of input image patch embedding.
            Defaults to 4.
        in_pad (int): The padding of input image patch embedding.
            Defaults to 2.
        down_patch_size (int): The patch size of downsampling patch embedding.
            Defaults to 3.
        down_stride (int): The stride of downsampling patch embedding.
            Defaults to 2.
        down_pad (int): The padding of downsampling patch embedding.
            Defaults to 1.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        grad_checkpoint (bool): Using grad checkpointing for saving memory.
        checkpoint_stage (Sequence | bool): Decide which layer uses grad checkpointing. 
                                            For example, checkpoint_stage=[0,0,1,1] means that stage3 and stage4 use gd
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims, --mlp_ratios:
    #     embedding dims and mlp ratios for the four stages
    # --downsamples: flags to apply downsampling or not in four blocks
    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [3, 3, 9, 3],
                         'embed_dims': [48, 96, 224, 448],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 4, 8],
                         'mlp_ratios': [4, 4, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [4, 4, 12, 4],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 3, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratios': [6, 6, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [4, 4, 21, 4],
                         'embed_dims': [76, 152, 336, 672],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 4, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mlp_ratios': [8, 8, 4, 4],
                         'layer_scale_init_value': 1e-5,}),}

    def __init__(self,
                 img_size:int=None,# lyf: Need define for patch embedding
                 a:float=0.1, #lyf: add for loss coefficients analysis
                 b:float=0.5, #lyf: add for loss coefficients analysis
                 c:float=20, #lyf: add for loss coefficients analysis
                 arch='t',
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=2,# lyf: change in_stride=4 to 2 for SOD tasks
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 num_classes=1000,
                 fork_feat=True,
                 start_level=0,
                 init_cfg=None,
                 pretrained="./transx-t.pth.tar",  #lyf add pretrained weights: https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar
                 **kwargs):

        super().__init__()
        '''
        The above img_size does not need to be adjusted, 
        even if the image input size is not 224x224,
        unless you want to change the size of the relative positional encoding
        '''
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4, 4]
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.relative_pos_enc = []
        self.pos_enc_record = []
        self.img_size = img_size #lyf: original init value
        img_size = to_2tuple(img_size)
        img_size = [math.ceil(img_size[0]/in_stride),
                      math.ceil(img_size[1]/in_stride)]
        for i in range(4):
            num_patches = img_size[0]*img_size[1]
            sr_patches = math.ceil(
                img_size[0]/sr_ratio[i])*math.ceil(img_size[1]/sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))
            self.pos_enc_record.append([img_size[0], img_size[1], 
                                        math.ceil(img_size[0]/sr_ratio[i]), 
                                        math.ceil(img_size[1]/sr_ratio[i]),])
            img_size = [math.ceil(img_size[0]/2),
                          math.ceil(img_size[1]/2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)
        # self.relative_pos_enc = [None] * 4

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                kernel_size=kernel_size[i],
                num_groups=num_groups[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i],)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1]))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb < start_level:
                    layer = nn.Identity()
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

        #lyf: add FPNneck and UPerHead
        self.fpn = FPN(in_channels_list=[48, 96, 224, 448], out_channels=256)
        self.sod_head = UPerHead(in_channels=[256, 256, 256, 256], num_classes=1)

        #lyf: add for SCRKD modules
        self.teacher = TransXNet_teacher(2*self.img_size)
       
        self.teacher.eval()

        #lyf: add for distillation loss coefficients
        #lyf: add for loss coefficients analysis
        self.a = a
        self.b = b
        self.c = c


        self.decouple1 = DecoupledConvolution(in_channels=256)
        self.decouple2 = DecoupledConvolution(in_channels=256)
        self.decouple3 = DecoupledConvolution(in_channels=256)
        self.decouple4 = DecoupledConvolution(in_channels=256)
       

        self.adaptive1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.adaptive4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        


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

        self.similarity_criterion = nn.L1Loss(reduction='mean')
        
        

        self.apply(self._init_model_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights(pretrained)
            if torch.distributed.is_initialized():
                self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    # init for image classification
    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    '''
    init for mmdetection or mmsegmentation 
    by loading imagenet pre-trained weights
    '''
    def init_weights(self, pretrained=None):


            ckpt_path = pretrained

            ckpt = torch.load(
                ckpt_path, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt


            state_dict = _state_dict
            """
            lyf: 
            RuntimeError: Error(s) in loading state_dict for TransXNet:
            size mismatch for relative_pos_enc.0: copying a param with shape torch.Size([1, 1, 3136, 49]) from checkpoint, the shape in current model is torch.Size([1, 1, 784, 16]).
            size mismatch for relative_pos_enc.1: copying a param with shape torch.Size([1, 2, 784, 49]) from checkpoint, the shape in current model is torch.Size([1, 2, 196, 16]).
            size mismatch for relative_pos_enc.2: copying a param with shape torch.Size([1, 5, 196, 49]) from checkpoint, the shape in current model is torch.Size([1, 5, 49, 16]).
            size mismatch for relative_pos_enc.3: copying a param with shape torch.Size([1, 8, 49, 49]) from checkpoint, the shape in current model is torch.Size([1, 8, 16, 16]).
            """
            # 1. 过滤掉形状不匹配的键
            filtered_dict = {
                k: v for k, v in state_dict.items() 
                if k in self.state_dict() and v.shape == self.state_dict()[k].shape
            }
            self.load_state_dict(filtered_dict, False)

            # # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

 

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        pos_idx = 0
        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:
                for blk in self.network[idx]:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                pos_idx += 1       
            else:
                x = self.network[idx](x)
            if self.fork_feat and (idx in self.out_indices):
                x_out = getattr(self, f'norm{idx}')(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self,  x_lr, x_hr=None, labels_lr=None, labels_hr=None):
        
        # input embedding
        x = self.forward_embeddings(x_lr)
        # through backbone
        outs = self.forward_tokens(x)
        
        outs = self.fpn(outs)
        
        out1, out2, out3, out4 = outs
        """
        decoupeled three part
        """
        
        out41, out42, out43 = self.decouple4(out4)
        out31, out32, out33 = self.decouple3(out3)
        out21, out22, out23 = self.decouple2(out2)
        out11, out12, out13 = self.decouple1(out1)

        
        """
        compute multi-scale similarity distillation loss
        """
        merge_student_feature1 = resize_and_combine_features([out13, out23, out33, out43], out13.size()[2:])    
        merge_student_feature2 = resize_and_combine_features([out13, out23, out33, out43], out23.size()[2:])   
        merge_student_feature3 = resize_and_combine_features([out13, out23, out33, out43], out33.size()[2:])   
        merge_student_feature4 = resize_and_combine_features([out13, out23, out33, out43], out43.size()[2:])   
 
        B, C, H, W, _ = merge_student_feature1.size()
        merge_student_feature1 = self.adaptive1(merge_student_feature1.permute(0, 4, 1, 2, 3).contiguous().view(B * 4, C, H, W)).view(B, 4, C, H, W).permute(0, 2, 3, 4, 1)
        B, C, H, W, _ = merge_student_feature2.size()
        merge_student_feature2 = self.adaptive2(merge_student_feature2.permute(0, 4, 1, 2, 3).contiguous().view(B * 4, C, H, W)).view(B, 4, C, H, W).permute(0, 2, 3, 4, 1)
        B, C, H, W, _ = merge_student_feature3.size()
        merge_student_feature3 = self.adaptive3(merge_student_feature3.permute(0, 4, 1, 2, 3).contiguous().view(B * 4, C, H, W)).view(B, 4, C, H, W).permute(0, 2, 3, 4, 1)
        B, C, H, W, _ = merge_student_feature4.size()
        merge_student_feature4 = self.adaptive4(merge_student_feature4.permute(0, 4, 1, 2, 3).contiguous().view(B * 4, C, H, W)).view(B, 4, C, H, W).permute(0, 2, 3, 4, 1)

    
        """
        SOD   attention fusion
        """
        att1 = self.att_conv1(torch.cat((out11, out12, out13), dim=1))
        n,_,h,w = out11.shape
        out1 = self.fusion_conv1((out11 * att1[:,0].view(n,1,h,w) + out12 * att1[:,1].view(n,1,h,w) + out13 * att1[:,2].view(n,1,h,w)))
        #student_smap1 = self.linear_pred1(out1)
        att2 = self.att_conv2(torch.cat((out21, out22, out23), dim=1))
        n,_,h,w = out21.shape
        out2 = self.fusion_conv2((out21 * att2[:,0].view(n,1,h,w) + out22 * att2[:,1].view(n,1,h,w) + out23 * att2[:,2].view(n,1,h,w)))
        #student_smap2 = self.linear_pred2(out2)
        att3 = self.att_conv3(torch.cat((out31, out32, out33), dim=1))
        n,_,h,w = out31.shape
        out3 = self.fusion_conv3((out31 * att3[:,0].view(n,1,h,w) + out32 * att3[:,1].view(n,1,h,w) + out33 * att3[:,2].view(n,1,h,w)))
        #student_smap3 = self.linear_pred3(out3)
        att4 = self.att_conv4(torch.cat((out41, out42, out43), dim=1))
        n,_,h,w = out41.shape
        out4 = self.fusion_conv4((out41 * att4[:,0].view(n,1,h,w) + out42 * att4[:,1].view(n,1,h,w) + out43 * att4[:,2].view(n,1,h,w)))
        #student_smap4 = self.linear_pred4(out4)

  
        student_smap1 = self.sod_head([out1, out2, out3, out4])
        
        ### interpolate
        student_smap1 = F.interpolate(student_smap1, size = x_lr.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            """
            compute multi-view self-similarity distillation loss
            """
            with torch.no_grad():    
                teacher_smap1, teacher_features = self.teacher(x_hr)
                teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features
        
            student_channel_similarity1, student_height_similarity1, student_width_similarity1 = compute_multi_view_self_similarity(out12)
            student_channel_similarity2, student_height_similarity2, student_width_similarity2 = compute_multi_view_self_similarity(out22)
            student_channel_similarity3, student_height_similarity3, student_width_similarity3 = compute_multi_view_self_similarity(out32)
            student_channel_similarity4, student_height_similarity4, student_width_similarity4 = compute_multi_view_self_similarity(out42)
            
                
            teacher_channel_similarity1, teacher_height_similarity1, teacher_width_similarity1 = compute_multi_view_self_similarity(F.adaptive_avg_pool2d(teacher_out1, output_size=out1.size()[2:]))
            teacher_channel_similarity2, teacher_height_similarity2, teacher_width_similarity2 = compute_multi_view_self_similarity(F.adaptive_avg_pool2d(teacher_out2, output_size=out2.size()[2:]))
            teacher_channel_similarity3, teacher_height_similarity3, teacher_width_similarity3 = compute_multi_view_self_similarity(F.adaptive_avg_pool2d(teacher_out3, output_size=out3.size()[2:]))
            teacher_channel_similarity4, teacher_height_similarity4, teacher_width_similarity4 = compute_multi_view_self_similarity(F.adaptive_avg_pool2d(teacher_out4, output_size=out4.size()[2:]))


            loss_multi_view_similarity = self.similarity_criterion(student_channel_similarity1, teacher_channel_similarity1) + self.similarity_criterion(student_channel_similarity2, teacher_channel_similarity2) + self.similarity_criterion(student_channel_similarity3, teacher_channel_similarity3) + self.similarity_criterion(student_channel_similarity4, teacher_channel_similarity4)  +\
                                         self.similarity_criterion(student_height_similarity1, teacher_height_similarity1) + self.similarity_criterion(student_height_similarity2, teacher_height_similarity2) + self.similarity_criterion(student_height_similarity3, teacher_height_similarity3) + self.similarity_criterion(student_height_similarity4, teacher_height_similarity4)  +\
                                         self.similarity_criterion(student_width_similarity1, teacher_width_similarity1) + self.similarity_criterion(student_width_similarity2, teacher_width_similarity2) + self.similarity_criterion(student_width_similarity3, teacher_width_similarity3) + self.similarity_criterion(student_width_similarity4, teacher_width_similarity4) 
            """
            compute multiscale feature distillation loss
            """

            loss_multi_scale_feature_loss = hcl(compute_similarity_weighted_fusion(merge_student_feature1, F.adaptive_avg_pool2d(teacher_out1, output_size=out1.size()[2:])), F.adaptive_avg_pool2d(teacher_out1, output_size=out1.size()[2:])) +\
                                            hcl(compute_similarity_weighted_fusion(merge_student_feature2, F.adaptive_avg_pool2d(teacher_out2, output_size=out2.size()[2:])), F.adaptive_avg_pool2d(teacher_out2, output_size=out2.size()[2:])) +\
                                            hcl(compute_similarity_weighted_fusion(merge_student_feature3, F.adaptive_avg_pool2d(teacher_out3, output_size=out3.size()[2:])), F.adaptive_avg_pool2d(teacher_out3, output_size=out3.size()[2:])) +\
                                            hcl(compute_similarity_weighted_fusion(merge_student_feature4, F.adaptive_avg_pool2d(teacher_out4, output_size=out4.size()[2:])), F.adaptive_avg_pool2d(teacher_out4, output_size=out4.size()[2:]))
                                            
            """
            compute prediction guidance distillation loss
            """
            loss_prediction_distillation = CriterionKD(F.interpolate(student_smap1, size = teacher_smap1.size()[2:], mode='bilinear', align_corners=True), teacher_smap1)
                                           
            """
            compute sod loss
            """

            loss_sod = structure_loss(student_smap1, labels_lr)

            return loss_sod, loss_multi_view_similarity * self.a, loss_multi_scale_feature_loss * self.b, loss_prediction_distillation * self.c #scale weights

        return torch.sigmoid(student_smap1)
 

       
        

#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

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
    temperature = 2
    loss = nn.KLDivLoss(reduction='mean')(torch.log(pred / temperature), soft.detach() / temperature)
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
    for i in range(4):  #seformer only four stage
        # 提取第 i 个聚合特征
        student_feature = student_aggregated_features[..., i]

        # downsampling
        # if H >= 16:
        #     student_feature =  F.adaptive_avg_pool2d(student_feature, output_size=(16,16))
        #     teacher_feature =  F.adaptive_avg_pool2d(teacher_feature, output_size=(16,16))

        # 计算余弦相似度

        similarity = F.cosine_similarity(student_feature.reshape(B, -1).contiguous(), teacher_feature.reshape(B, -1).contiguous(), dim=1)
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



"""
lyf add: for UpNet Segmentation Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    """金字塔池化模块(Pyramid Pooling Module)"""
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for bin in bins:
            self.blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv = nn.Conv2d(in_channels + len(bins)*out_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        h, w = x.shape[2:]
        features = [x]
        for block in self.blocks:
            y = block(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
            features.append(y)
        x = torch.cat(features, dim=1)
        x = self.conv(x)
        return x

class FPN(nn.Module):
    """特征金字塔网络(Feature Pyramid Network)"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels = in_channels_list
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        

        # PPM模块
        self.ppm = PPM(in_channels_list[-1], in_channels_list[-1])

    def forward(self, inputs):
        # inputs应该是来自backbone的多级特征图列表
        assert len(inputs) == len(self.in_channels)

        # 应用PPM到最高级特征
        ppm_out = self.ppm(inputs[-1])
        
        # 替换最高级特征为PPM输出（确保fpn_inputs是列表）
        inputs = inputs[:-1] + [ppm_out]

        # 自顶向下路径
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # 特征融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=True)
        
        # 构建输出
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
         
        return outs
   
    
class UPerHead(nn.Module):
    """UPerHead解码器"""
    """
    lyf add: num_classes = 1 for SOD task
    """
    def __init__(self, in_channels=None, out_channels=256, num_classes=1):
        super().__init__()
        
        self.channels = out_channels
        self.num_classes = num_classes

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels)*out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分类头
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, num_classes, 1)
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
         # 上采样并拼接
        outs = [
            F.interpolate(inputs[i], size=inputs[0].shape[2:], mode='bilinear', align_corners=True)
            for i in range(1, len(inputs))
        ]
        outs = [outs[0]] + outs
        out = torch.cat(outs, dim=1)
        out = self.fpn_bottleneck(out)
        # 分类
        out = self.cls_seg(out)
        
        return out
    


if __name__ == '__main__':
    x = torch.Tensor(2, 3, 112, 112).cuda()
    model = TransXNet(img_size=112).cuda()
    print(model(x)[1][0].size())
    print(model(x)[1][1].size())
    print(model(x)[1][2].size())
    print(model(x)[1][3].size())
    print(model(x)[0].size())
    """
    torch.Size([2, 128, 112, 112])
    torch.Size([2, 256, 56, 56])
    torch.Size([2, 384, 28, 28])
    torch.Size([2, 512, 14, 14])
    """