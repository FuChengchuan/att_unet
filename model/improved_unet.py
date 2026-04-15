import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

# ===== Simplified Fusion Module =====
class SimpleFuse(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Upsample, concatenate, then apply a single convolution
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                               padding=padding, stride=stride,
                               dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

class Grouped_multi_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #---------- xy ----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #---------- zx ----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #---------- zy ----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #---------- dw ----------#
        x4 = self.dw(x4)
        #---------- concat ----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #---------- ldw ----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x



class improved_unet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, c_list=[8,16,24,32,48,64]):
        super().__init__()
        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, padding=1)
        self.encoder4 = Grouped_multi_Attention(c_list[2], c_list[3])
        self.encoder5 = Grouped_multi_Attention(c_list[3], c_list[4])
        self.encoder6 = Grouped_multi_Attention(c_list[4], c_list[5])

        # Simplified feature fusion modules
        # (each module takes the decoder output plus the skip connection feature as input)
        self.fuse5 = SimpleFuse(c_list[4] * 2, c_list[4])  # d5_out_ch + x5_ch
        self.fuse4 = SimpleFuse(c_list[3] * 2, c_list[3])  # d4_out_ch + x4_ch
        self.fuse3 = SimpleFuse(c_list[2] * 2, c_list[2])  # d3_out_ch + x3_ch
        self.fuse2 = SimpleFuse(c_list[1] * 2, c_list[1])  # d2_out_ch + x2_ch
        self.fuse1 = SimpleFuse(c_list[0] * 2, c_list[0])  # d1_out_ch + x1_ch

        # Decoder
        self.decoder5 = Grouped_multi_Attention(c_list[5], c_list[4])
        self.decoder4 = Grouped_multi_Attention(c_list[4], c_list[3])
        self.decoder3 = Grouped_multi_Attention(c_list[3], c_list[2])
        self.decoder2 = nn.Conv2d(c_list[2], c_list[1], 3, padding=1)
        self.decoder1 = nn.Conv2d(c_list[1], c_list[0], 3, padding=1)

        # Final output layer
        self.final = nn.Conv2d(c_list[0], num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoding
        x1 = F.gelu(self.encoder1(x))
        x2 = F.gelu(self.encoder2(F.max_pool2d(x1, 2)))
        x3 = F.gelu(self.encoder3(F.max_pool2d(x2, 2)))
        x4 = F.gelu(self.encoder4(F.max_pool2d(x3, 2)))
        x5 = F.gelu(self.encoder5(F.max_pool2d(x4, 2)))
        x6 = F.gelu(self.encoder6(F.max_pool2d(x5, 2)))

        # Decoding and fusion
        d5 = self.decoder5(x6)
        u5 = self.fuse5(d5, x5)
        d4 = self.decoder4(u5)
        u4 = self.fuse4(d4, x4)
        d3 = self.decoder3(u4)
        u3 = self.fuse3(d3, x3)
        d2 = self.decoder2(F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False))
        u2 = self.fuse2(d2, x2)
        d1 = self.decoder1(F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False))
        u1 = self.fuse1(d1, x1)

        out = F.interpolate(self.final(u1), scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)