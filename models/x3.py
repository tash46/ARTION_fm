import torch
import torch.nn as nn
import torch.nn.functional as F

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

class X3(nn.Module):
    def __init__(self, num_classes):
        super(X3, self).__init__()
        # encoder
        self.initial   = InitialBlock(1, 16)
        self.stage1_0  = SeparableBottleneck(16,  64, downsample=True,  dropout_prob=0.01)
        self.stage1_1  = SeparableBottleneck(64,  64, downsample=False, dropout_prob=0.01)
        self.stage2_0  = SeparableBottleneck(64, 128, downsample=True,  dropout_prob=0.1)
        self.stage2_1  = SeparableBottleneck(128,128, downsample=False, dilation=2, dropout_prob=0.1)
        self.stage3_0  = SeparableBottleneck(128,128, downsample=False, dilation=2, dropout_prob=0.1)

        # multi-scale context
        self.aspp      = ASPP(in_channels=128, out_channels=128, atrous_rates=[6,12,18])

        # decoder—level 1 
        self.up1       = nn.ConvTranspose2d(128,  64, kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=False)
        self.bn_up1    = nn.BatchNorm2d(64)
        self.conv_up1  = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # decoder—level 2 
        self.up2       = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=False)
        self.bn_up2    = nn.BatchNorm2d(32)
        self.conv_up2  = nn.Sequential(
            nn.Conv2d(32 + 16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # aux_head1 taps into level-2 feature
        self.aux1 = nn.Conv2d(32, num_classes, kernel_size=1)
        # aux_head2 taps into level-1 feature
        self.aux2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # final classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x0 = self.initial(x)                     # [B,16,H/2,W/2]
        x1 = self.stage1_1(self.stage1_0(x0))    # [B,64,H/4,W/4]
        x2 = self.stage2_1(self.stage2_0(x1))    # [B,128,H/8,W/8]
        x3 = self.stage3_0(x2)                   # [B,128,H/8,W/8]
        x3 = self.aspp(x3)                       # [B,128,H/8,W/8]

        # decoder level-1
        y1 = self.up1(x3)                        # [B,64,H/4,W/4]
        y1 = F.relu(self.bn_up1(y1), inplace=True)
        y1 = torch.cat([y1, x1], dim=1)          # fuse skip @ H/4
        y1 = self.conv_up1(y1)                   # [B,64,H/4,W/4]

        # decoder level-2
        y2 = self.up2(y1)                        # [B,32,H/2,W/2]
        y2 = F.relu(self.bn_up2(y2), inplace=True)
        y2 = torch.cat([y2, x0], dim=1)          # fuse skip @ H/2
        y2 = self.conv_up2(y2)                   # [B,32,H/2,W/2]

        # upsample aux2
        aux2_out = F.interpolate(
            self.aux2(y1), size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=True
        )
        # upsample aux1
        aux1_out = F.interpolate(
            self.aux1(y2), size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=True
        )

        y = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=True)  # [B,32,H,W]
        main_out = self.classifier(y)                                               # [B,C,H,W]

        return main_out, aux1_out, aux2_out
