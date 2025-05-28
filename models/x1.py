import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_conv = nn.Conv2d(in_channels, out_channels - in_channels,
                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.main_conv(x)
        pool = self.pool(x)
        out = torch.cat([conv, pool], dim=1)
        out = self.bn(out)
        return self.relu(out)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, downsample=False, dropout_prob=0.1):
        super().__init__()
        internal = out_channels // 4

        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, internal, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal)

        self.conv2 = nn.Conv2d(
            internal, internal, kernel_size=3,
            stride=2 if downsample else 1,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(internal)

        self.conv3 = nn.Conv2d(internal, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.downsample_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False) \
            if downsample else nn.Identity()
        self.downsample_bn = nn.BatchNorm2d(out_channels) if downsample else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.bn3(self.conv3(out)))

        identity = self.downsample_bn(self.downsample_layer(identity))
        return self.relu(out + identity)

class X1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial = InitialBlock(1, 16)

        self.bottleneck1_0 = Bottleneck(16, 64, downsample=True, dropout_prob=0.01)
        self.bottleneck1_1 = Bottleneck(64, 64, dropout_prob=0.01)
        self.bottleneck1_2 = Bottleneck(64, 64, dropout_prob=0.01)
        self.bottleneck1_3 = Bottleneck(64, 64, dropout_prob=0.01)

        self.bottleneck2_0 = Bottleneck(64, 128, downsample=True, dropout_prob=0.1)
        self.bottleneck2_1 = Bottleneck(128, 128, dropout_prob=0.1)
        self.bottleneck2_2 = Bottleneck(128, 128, dilation=2, dropout_prob=0.1)
        self.bottleneck2_3 = Bottleneck(128, 128, dropout_prob=0.1)
        self.bottleneck2_4 = Bottleneck(128, 128, dilation=4, dropout_prob=0.1)
        self.bottleneck2_5 = Bottleneck(128, 128, dropout_prob=0.1)
        self.bottleneck2_6 = Bottleneck(128, 128, dilation=8, dropout_prob=0.1)
        self.bottleneck2_7 = Bottleneck(128, 128, dropout_prob=0.1)

        self.bottleneck3_1 = Bottleneck(128, 128, dilation=2, dropout_prob=0.1)
        self.bottleneck3_2 = Bottleneck(128, 128, dilation=4, dropout_prob=0.1)
        self.bottleneck3_3 = Bottleneck(128, 128, dilation=8, dropout_prob=0.1)
        self.bottleneck3_4 = Bottleneck(128, 128, dilation=16, dropout_prob=0.1)

        self.decoder_bottleneck4_0 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1, bias=False)
        self.decoder_bn4_0 = nn.BatchNorm2d(64)
        self.decoder_relu4_0 = nn.ReLU(inplace=True)
        self.decoder_bottleneck4_1 = Bottleneck(64, 64, dropout_prob=0.1)

        self.decoder_bottleneck5_0 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1, bias=False)
        self.decoder_bn5_0 = nn.BatchNorm2d(16)
        self.decoder_relu5_0 = nn.ReLU(inplace=True)
        self.decoder_bottleneck5_1 = Bottleneck(16, 16, dropout_prob=0.1)

        self.decoder_bottleneck6_0 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2,
                                                        padding=1, output_padding=1, bias=False)
        self.decoder_bn6_0 = nn.BatchNorm2d(16)
        self.decoder_relu6_0 = nn.ReLU(inplace=True)

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)

        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)

        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)

        x = self.decoder_bottleneck4_0(x)
        x = self.decoder_bn4_0(x)
        x = self.decoder_relu4_0(x)
        x = self.decoder_bottleneck4_1(x)

        x = self.decoder_bottleneck5_0(x)
        x = self.decoder_bn5_0(x)
        x = self.decoder_relu5_0(x)
        x = self.decoder_bottleneck5_1(x)

        x = self.decoder_bottleneck6_0(x)
        x = self.decoder_bn6_0(x)
        x = self.decoder_relu6_0(x)

        return self.classifier(x)
