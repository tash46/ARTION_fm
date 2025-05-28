import torch
import torch.nn as nn
import torch.nn.functional as F

# ASPP implementation
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1×1 conv branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # parallel 3×3 atrous conv branches
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # image‐level pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        self.branches = nn.ModuleList(modules)

        # projection after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(len(modules)*out_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for branch in self.branches[:-1]:
            res.append(branch(x))
        # image‐level branch: upsample to x’s spatial size
        img_pool = self.branches[-1](x)
        img_pool = F.interpolate(img_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(img_pool)
        x = torch.cat(res, dim=1)
        return self.project(x)
    
# Initial Block (kept similar to X1)
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        # Split: one branch is a 3x3 conv, the other is a 2x2 max pool
        self.main_conv = nn.Conv2d(in_channels, out_channels - in_channels,
                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv = self.main_conv(x)
        pool = self.pool(x)
        out = torch.cat([conv, pool], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


# Separable Bottleneck Block: Using Depthwise Separable Conv
class SeparableBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, downsample=False, dropout_prob=0.1):
        """
        A bottleneck block that replaces the standard 3x3 convolution with
        a depthwise separable convolution (depthwise + pointwise).
        """
        super(SeparableBottleneck, self).__init__()
        internal_channels = out_channels // 4
        self.downsample = downsample
        stride = 2 if downsample else 1

        # 1x1 expansion (channel reduction)
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Depthwise convolution: spatial filtering with groups=channels
        self.depthwise = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=stride, padding=dilation, dilation=dilation, 
                                   groups=internal_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(internal_channels)
        
        # Pointwise convolution: combine channels
        self.pointwise = nn.Conv2d(internal_channels, internal_channels, kernel_size=1, bias=False)
        self.bn_pointwise = nn.BatchNorm2d(internal_channels)
        
        # 1x1 projection (channel expansion)
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # For matching dimensions in residual connection if needed.
        if downsample or in_channels != out_channels:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample_layer = None

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn_dw(self.depthwise(out)))
        out = self.relu(self.bn_pointwise(self.pointwise(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        
        if self.downsample_layer is not None:
            identity = self.downsample_layer(identity)
        
        out += identity
        out = self.relu(out)
        return out

class X2(nn.Module):
    def __init__(self, num_classes):
        super(X2, self).__init__()
        
        # Initial Block: from 1 channel (grayscale) to 16 channels
        self.initial = InitialBlock(in_channels=1, out_channels=16)
        
        # Stage 1: Here we use two separable bottlenecks
        # First block downsamples from 16 -> 64 channels
        self.stage1_0 = SeparableBottleneck(16, 64, downsample=True, dropout_prob=0.01)
        self.stage1_1 = SeparableBottleneck(64, 64, dropout_prob=0.01)
        
        # Stage 2: Increase channels to 128 with one downsampling block followed by one block with dilation
        self.stage2_0 = SeparableBottleneck(64, 128, downsample=True, dropout_prob=0.1)
        self.stage2_1 = SeparableBottleneck(128, 128, dilation=2, dropout_prob=0.1)
        
        # Stage 3: One block to further process the features (using mild dilation)
        self.stage3_0 = SeparableBottleneck(128, 128, dilation=2, dropout_prob=0.1)

        # ASPP module
        self.aspp = ASPP(in_channels=128, out_channels=128, atrous_rates=[6,12,18])
        
        # Simplified Decoder:
        # First, use a transposed convolution to upsample from 1/8 resolution (H/8) to 1/4 resolution (H/4)
        self.decoder_conv = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                                 padding=1, output_padding=1, bias=False)
        self.decoder_bn = nn.BatchNorm2d(64)
        self.decoder_relu = nn.ReLU(inplace=True)
        
        # Final classifier using a 1x1 convolution.
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
    
    def forward(self, x):
        # x is expected to have shape [B, 1, H, W] (grayscale input)
        # Encoder
        x = self.initial(x)           # [B, 16, H/2, W/2]
        x = self.stage1_0(x)          # [B, 64, H/4, W/4]
        x = self.stage1_1(x)          # [B, 64, H/4, W/4]
        x = self.stage2_0(x)          # [B, 128, H/8, W/8]
        x = self.stage2_1(x)          # [B, 128, H/8, W/8]
        x = self.stage3_0(x)          # [B, 128, H/8, W/8]

        # apply ASPP for richer multi-scale context
        x = self.aspp(x)              # [B,128,H/8,W/8]
        
        # Decoder
        x = self.decoder_conv(x)      # [B, 64, H/4, W/4]
        x = self.decoder_bn(x)
        x = self.decoder_relu(x)
        # Upsample from H/4 to full resolution using fast bilinear interpolation
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.classifier(x)        # [B, num_classes, H, W]
        return x
