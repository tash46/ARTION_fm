import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(F_g + F_l, F_int, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()  
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_g, F_g // 8, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_g // 8, F_g, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        combined_features = torch.cat([g, x], dim=1)
        spatial_attention_map = self.spatial_attention(combined_features)
        spatially_attended = x * spatial_attention_map
        channel_attention_map = self.channel_attention(g)
        channel_attended = g * channel_attention_map
        combined_attention = spatially_attended + channel_attended
        return combined_attention

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.features = nn.ModuleList()

        for pool_size in pool_sizes:
            self.features.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=(pool_size, pool_size), stride=(pool_size, pool_size), padding=0),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pyramids = [x]

        # Apply the static pooling sizes
        for feature in self.features:
            pooled = feature(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(upsampled)

        return torch.cat(pyramids, dim=1)

class X4(nn.Module):
    def __init__(self):
        super(X4, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        #self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512) #(512, 1024)
        
        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(in_channels=512) #1024
        
        # Decoder
        #self.upconv4 = nn.ConvTranspose2d(1024 * 2, 512, kernel_size=2, stride=2)  # Adjusted input for PPM
        #self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512 * 2, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Attention Blocks
        #self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        #e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e3))
        
        # Pyramid Pooling Module applied after bottleneck
        b = self.ppm(b)
        
        # Decoder path with attention
        #d4 = self.upconv4(b)
        #e4 = self.att4(g=d4, x=e4)  # Apply attention block
        #d4 = torch.cat((d4, e4), dim=1)
        #d4 = self.dec4(d4)
        
        d3 = self.upconv3(b)
        e3 = self.att3(g=d3, x=e3)  # Apply attention block
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        e2 = self.att2(g=d2, x=e2)  # Apply attention block
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        e1 = self.att1(g=d1, x=e1)  # Apply attention block
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)
