"""
https://github.com/winycg/CIRKD/blob/main/models/segformer.py
"""
import math
from statistics import mode
from tkinter.tix import MAIN
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .segformer_teacher import get_segformer_teacher
import os

__all__ = ['MiT_B0', 'MiT_B1', 'MiT_B2', 'get_segformer']

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        #img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
    
    
class LinearMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
    
class Segformer_SCRKD(nn.Module):
    def __init__(
        self, 
        pretrained = "./mit_b1.pth",
        img_size=None, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1, 
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4], 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0.,
        attn_drop_rate=0., 
        drop_path_rate=0.1, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        batchnorm_layer=nn.BatchNorm2d,
        depths=[2, 2, 2, 2], 
        sr_ratios=[8, 4, 2, 1],
        decoder_dim = 256
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, 
            patch_size=7, 
            stride=2, #original stride=4, 
            in_chans=in_chans,
            embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=(img_size[0] // 4, img_size[1] // 4), 
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=(img_size[0] // 8, img_size[1] // 8), 
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=(img_size[0] // 16, img_size[1] // 16), 
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3]
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        
        self.teacher = get_segformer_teacher(img_size=img_size)
       
        self.teacher.eval()

        

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
        
        
        # segmentation head
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Sequential(*[nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
                                           batchnorm_layer(decoder_dim),
                                           nn.ReLU()])
        self.dropout = nn.Dropout2d(drop_rate)

        self.linear_pred1 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        # self.linear_pred2 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        # self.linear_pred3 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        # self.linear_pred4 = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         this_dir = os.getcwd()
    #         pretrained =  os.path.join(this_dir, pretrained)
    #         old_dict = torch.load(pretrained)
    #         model_dict = self.state_dict()
    #         old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    #         #model_dict.update(old_dict)
    #         # print(model_dict.keys())
    #         # print("____________________________________")
    #         # print(old_dict.keys())
    #         for k in list(model_dict.keys()):
    #             if k in old_dict:
    #                 if model_dict[k].shape != old_dict[k].shape:
    #                     print("delete:{};shape pretrain:{};shape model:{}".format(k,old_dict[k].shape,model_dict[k].shape))
    #                     del model_dict[k]

    #         self.load_state_dict(model_dict)
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            this_dir = os.getcwd()
            pretrained =  os.path.join(this_dir, pretrained)
            old_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            self.load_state_dict(model_dict)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.size())

        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage1")  torch.Size([1, 64, H/2, W/2])
        # print(x.size())  
        # stage 2
        x, H, W = self.patch_embed2(x)
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.size())
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage2")  torch.Size([1, 64, H/4, W/4])
        # print(x.size())
        # stage 3
        x, H, W = self.patch_embed3(x)
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.size())
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage3")  torch.Size([1, 64, H/8, W/8])
        # print(x.size())  torch.Size([1, 320, 14, 14])
        # stage 4
        x, H, W = self.patch_embed4(x)
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.size())
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage4")  torch.Size([1, 64, H/16, W/16])
        # print(x.size())  torch.Size([1, 512, 7, 7])
        return outs 

    def forward(self, x_lr, x_hr=None, labels_lr=None, labels_hr=None):
        x = self.forward_features(x_lr)
        
        c1, c2, c3, c4 = x 
        
         ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        h_out, w_out = c1.size()[2], c1.size()[3]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        

        # x = self.dropout(_c)
        # x = self.linear_pred(x)

        # """
        # lyf: resize the x to the input shape, according to MMSeg
        # https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/utils/wrappers.py
        # https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/decode_head.py#L310
        # """
        # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        out1 = _c1
        out2 = _c2
        out3 = _c3
        out4 = _c4

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

        _c = self.linear_fuse(torch.cat([out4, out3, out2, out1], dim = 1))
        x = self.dropout(_c)
        student_smap1 = self.linear_pred1(x)

        ### interpolate
        student_smap1 = F.interpolate(student_smap1, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        # student_smap2 = F.interpolate(student_smap2, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        # student_smap3 = F.interpolate(student_smap3, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        # student_smap4 = F.interpolate(student_smap4, size = x_lr.size()[2:], mode='bilinear', align_corners=True)
        





        if self.training:
            with torch.no_grad():    
             teacher_smap1, teacher_features = self.teacher(x_hr)
             teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher_features
            """
            compute multi-view self-similarity distillation loss
            """
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

            return loss_sod, loss_multi_view_similarity * 0.1, loss_multi_scale_feature_loss, loss_prediction_distillation * 100 #scale weights

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




if __name__ == '__main__':
    x = torch.Tensor(2, 3, 224, 224)
    model = Segformer_SCRKD()
    print(model(x)[4].size())