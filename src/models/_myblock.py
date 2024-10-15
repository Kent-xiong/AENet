import torch
import torch.nn as nn
import torch.nn.functional as F      
import math
import numpy as np



# 一种新的卷积方式

class MSDConv_SSFC(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        # native feature maps
        self.native = nn.Sequential(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        self.aux = nn.Sequential(
            CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
                   bias=False),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(inplace=True),
        )

        self.att = SSFC(aux_ch)

    def forward(self, x):
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]   # 只要保证输入输出通道都是偶数的就可以了。

class CMConv(nn.Module):   # 这个其实写的不是很规范
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,   #       out_ch =   aux_ch = native_ch * (ratio - 1),groups=int(native_ch / 4)
                 bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,  # 设置padding = dilation的话可以保持特征尺寸不变
                              groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)  # 在下面把某些卷积核的权重设为0，来屏蔽掉部分卷积核对某些输入通道的卷积运算。

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        # self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        self.mask = torch.zeros(self.conv.weight.shape, device=self.conv.weight.device).byte()
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) *
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):   # 好有意思啊这个代码，写的乱糟糟的，但是就是有用是吧
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift

class SSFC(torch.nn.Module):
    def __init__(self, in_ch):
        super(SSFC, self).__init__()

        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv

    def forward(self, x):  # 自注意力机制，就是用一种方法，对所有像素，在我需要提取的特征中，计算出来一个个权重，这个权重乘以原先的像素值，就得到了一个带注意力的数据
        _, _, h, w = x.size()

        q = x.mean(dim=[2, 3], keepdim=True)
        # k = self.proj(x)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        # print(sigma)
   
        return x * att_weight
  
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
        )
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.Conv(input)
        shortcut = self.shortcut(input)
        return self.relu(out+shortcut)
    

class ChangeLength(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChangeLength,self).__init__()
    #             setattr(self, f'ScaleConv_{i+1}', DoubleScaleandRes_01(self.in_ch[i], self.out_ch[i]))
    # else:
    #     setattr(self, f'DoubleScaleConv_{i+1}', ScaleandPsyconvRes(self.in_ch[i], self.out_ch[i]))
        self.conv = DoubleScaleandRes_01(in_ch, out_ch)
        # self.conv = ScaleandPsyconvRes(in_ch, out_ch)
        # self.sa1 = SpatialAttention()
        # self.sa2 = SpatialAttention()

    def forward(self, x1, x2, x3):        
        if x1.shape != x2.shape:       
           print("孪生网络中,同一尺度的输出尺寸不一致")      
           raise ValueError      
        # locationchangerate = torch.cat([x1, x2], dim =0)
        # locationchangerate = self.sa1(locationchangerate)
        # x1, x2 = torch.split(locationchangerate, x1.size(0), dim=0)
        locationchangerate = torch.abs(x1 - x2)     
        # gradualchangeunit = self.sa2(x3)
        gradualchangeunit = x3       
        if gradualchangeunit.shape  != locationchangerate.shape:      
            print("两个编码器的输出尺寸不相同")     
            raise ValueError       
        changelen = torch.mul(gradualchangeunit, locationchangerate)       
        changelen = self.conv(changelen)       
        # return self.conv_len(changelen)
        return changelen, locationchangerate       

class Fusion(nn.Module):      
    def __init__(self,in_ch = [16,32,64,128], out_ch = [128,128,128,128]):    
        super().__init__()    
        self.layers = len(in_ch)   
        for i in range(1, self.layers+1):   
            # setattr(self,f'len{i}',ChangeLength())   
            setattr(self,f'len{i}',ChangeLength(in_ch[i-1], out_ch[i-1]))  
            # setattr(self,f'towpsconv{i}', ScaleandPsyconvRes(in_ch[i-1],out_ch[i-1]))   
    
    def forward(self,t1_feature_list,t2_feature_list,changerate_list):
        f = []
        ss = []
        if len(t1_feature_list) == len(t2_feature_list) == len(changerate_list):
            for i in range(1,self.layers+1):
                change_len_obj = getattr(self, f'len{i}')
                change_len, ss_value = change_len_obj(t1_feature_list[i-1], t2_feature_list[i-1], changerate_list[i-1])
                f.append(change_len)
                ss.append(ss_value)
        else:
            print("三个列表的长度不一致")
            raise ValueError      
        return f, ss



# 新的提取特征主干
class ScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm = True, activate = True):  # 现在是平均来分输出channel，其实也可以 1x1占剩下一半的1/2，3x3占1/4，5x5占1/8，7x7占1/8，这样其实也是减小参数了！！！！！好思路，如果不行，可以尝试一下空洞卷积，
        super().__init__()  
        self.bn = batchnorm
        self.act = activate
        self.conv1 = nn.Conv2d(in_ch, out_ch//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_ch, out_ch//2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, out_ch//8, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_ch, out_ch//8, kernel_size=7, padding=3)
        self.btach = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out1, out3, out5, out7], dim = 1)
        if self.bn:
            out = self.btach(out)
        if self.act:
            out = self.relu(out)
        return out

class DoubleScaleandRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        self.conv1 = ScaleConv(in_ch= in_ch, out_ch= out_ch)
        self.conv2 = ScaleConv(in_ch= out_ch, out_ch= out_ch, activate = False)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        return self.relu(shortcut + out)
    
class DoubleScaleandRes_01(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.shortcut = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        self.conv1 = ScaleConv(in_ch= in_ch, out_ch= out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        shortcut = self.shortcut(out1)
        return self.relu(shortcut + out)
            
class ThreeScaleandRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ScaleConv(in_ch= in_ch, out_ch= out_ch)
        self.shortcut = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch)
            )
        self.conv2 = ScaleConv(in_ch= out_ch, out_ch= out_ch)
        self.conv3 = ScaleConv(in_ch= out_ch, out_ch= out_ch, activate = False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        out = self.conv3(out)
        shortcut = self.shortcut(out1)
        return self.relu(out + shortcut)

class ScaleandPsyconvRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sconv = ScaleConv(in_ch, out_ch)
        self.psconv = nn.Sequential(
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch)
        )
        self.con1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.sconv(x)
        out = self.psconv(out)
        shortcut = self.con1x1(x)
        return self.relu(out + shortcut)
        
        
class StaticEncoder(nn.Module):
    def __init__(self, in_ch= [3, 16, 32, 64], out_ch = [16, 32, 64, 128]):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layers = len(self.out_ch)
        for i in range(self.layers):
            if i == 0:
                setattr(self, f'ScaleConv_{i+1}', DoubleScaleandRes_01(self.in_ch[i], self.out_ch[i]))
            else:
                setattr(self, f'DoubleScaleConv_{i+1}', ScaleandPsyconvRes(self.in_ch[i], self.out_ch[i]))
            # setattr(self, f'Channel_Attention_{i+1}', Channel_Attention_Module_Conv(self.out_ch[i])) 
            # setattr(self, f'Self_Attention_{i+1}', SpatialAttention())    
        
    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim = 0)
        out1_list = []
        out2_list = []
        for i in range(self.layers):
            if i == 0:
                scaleconvobj = getattr(self, f'ScaleConv_{i+1}')
            else:
                scaleconvobj = getattr(self, f'DoubleScaleConv_{i+1}')
            out = scaleconvobj(out)
            # channel_attention_obj = getattr(self, f'Channel_Attention_{i+1}')
            # out = channel_attention_obj(out)
            if i != 0:
                out = self.maxpool(out)
            # sa_obj = getattr(self, f'Self_Attention_{i+1}')  # 在这里进行空间自注意力操作，是为了防止后面在getLen时进行两次操作，而延长计算时间，这里算完再沿着批次维度拆开
            # out =  sa_obj(out)
            out1, out2 = torch.split(out, x1.size(0), dim=0)
            out1_list.append(out1)
            out2_list.append(out2)
        return out1_list, out2_list
    
    
class DynamicEncoder(nn.Module):
    def __init__(self, in_ch = [6, 16, 32, 64],  out_ch = [16, 32, 64, 128]): 
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layers = len(self.out_ch)
        for i in range(self.layers):
            if i == 0:
                setattr(self, f'ScaleConv_{i+1}', DoubleScaleandRes_01(self.in_ch[i], self.out_ch[i]))
            else:
                setattr(self, f'DoubleScaleConv_{i+1}', ScaleandPsyconvRes(self.in_ch[i], self.out_ch[i]))
            # setattr(self, f'Channel_Attention_{i+1}', Channel_Attention_Module_Conv(self.out_ch[i]))          
        
    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim = 1) 
        out_list = []
        for i in range(self.layers):
            if i == 0:
                scaleconvobj = getattr(self, f'ScaleConv_{i+1}')
            else:
                scaleconvobj = getattr(self, f'DoubleScaleConv_{i+1}')
            out = scaleconvobj(out)
            # channel_attention_obj = getattr(self, f'Channel_Attention_{i+1}')
            # out = channel_attention_obj(out)
            if i != 0:
                out = self.maxpool(out)
            out_list.append(out)
        return out_list