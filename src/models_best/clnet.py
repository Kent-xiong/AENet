# TODO: make norm3d module types changeable in temporal branch.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._blocks import Conv1x1, Conv3x3
from ._myblock import StaticEncoder, DynamicEncoder, Fusion
import torch

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class DecBlock(nn.Module):   #  将两路数据堆叠起来（将x2的高宽尺寸上采样成x1），然后进行卷积。
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1+in_ch2, out_ch)

    def forward(self, x1, x2):   # x1.shape[2:]，选择上采样到x1的高宽。
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')  # 插值法，默认是最邻近，这种速度最快，但是像素值不连续。F.interpolate(x2, size=x1.shape[2:]) 调用 F.interpolate 函数，将 x2 的空间尺寸调整为与 x1 相同。size 参数指定了目标尺寸。这意味着 x2 将被上采样或下采样到与 x1 相同的 height 和 width。默认使用最邻近插值。参数align_corners 默认为False，仅当使用的算法为'linear', 'bilinear', 'bilinear'or 'trilinear'时可以使用，四个角的像素值对齐，在中间扩充元素。参数size与scale_factor两个参数只能定义一个，即两种采样模式只能用一个。要么让数组放大成特定大小、要么给定特定系数，来等比放大数组。
        x = torch.cat([x1, x2], dim=1)  # dim：指定连接的维度，默认为0。
        return self.conv_fuse(x)

class SimpleDecoder(nn.Module):
    def __init__(self, channel = [256, 128, 64 ,32]):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, in_ch2)
            for in_ch1, in_ch2 in zip(channel, channel)
        ])
        self.foot1 = DecBlock(256, 128, 128)
        self.foot2 = DecBlock(128, 64, 64)
        self.foot3 = DecBlock(64, 32, 32)   # 这边可以直接上采样 + 下面的32个通道一起输出
        self.conv_out = Conv1x1(32, 1)
    
    def forward(self, feats):
        feats = feats[::-1]
        list = []
        out1 = self.foot1(feats[1], feats[0])
        out2 = self.foot2(feats[2], out1)
        out3 = self.foot3(feats[3], out2)
        y = self.conv_out(out3)
        list.append(y)
        list.append(out3)
        list.append(out2)
        list.append(out1)

        return y, list
    

class CLNet(nn.Module):   
    def __init__(self): 
        super().__init__()
        self.change_unit_encoder = StaticEncoder(in_ch= [3, 32, 64, 128], out_ch = [32, 64, 128, 256])
        self.conv_out_auxiliary1 = Conv1x1(256, 1)        # 深度监督    
        self.change_rate_encoder = DynamicEncoder(in_ch= [6, 32, 64, 128], out_ch = [32, 64, 128, 256])
        self.conv_out_auxiliary2= Conv1x1(256, 1)        # 深度监督
        self.change_len_get = Fusion(in_ch = [32,64,128,256], out_ch = [32,64,128,256])  
        self.decoder = SimpleDecoder(channel = [256, 128, 64 ,32]) 
    def forward(self, t1, t2, return_aux=True):          
        feats_changerate_list = self.change_rate_encoder(t1, t2)  
        feats_change_unit_list1, feats_change_unit_list2 = self.change_unit_encoder(t1, t2)  # 返回输入，feats_p = [输入，第一个编码输出，第二个编码输出，第三个编码输出]，  函数参数(feats_p[-1], feats_p)
        change_len_list, ss_list = self.change_len_get(feats_change_unit_list1, feats_change_unit_list2, feats_changerate_list)
        pred , decoder_list= self.decoder(change_len_list)  # 返回一张与图像大小相同的也即batchx1x256x256的 数据
        if return_aux:  # 下面两行可以返回过来试试
            last_change_unit_feats = torch.abs(feats_change_unit_list1[-1] - feats_change_unit_list2[-1])
            pred_v1 = self.conv_out_auxiliary1(last_change_unit_feats)
            pred_v1 = F.interpolate(pred_v1, size=pred.shape[2:])
            last_changerate_feats = feats_changerate_list[-1]
            pred_v2 = self.conv_out_auxiliary2(last_changerate_feats)
            pred_v2 = F.interpolate(pred_v2, size=pred.shape[2:])
            return pred, pred_v1, pred_v2
        else:
            return pred





        
        
