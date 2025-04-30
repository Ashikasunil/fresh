import torch
import torch.nn as nn
import timm

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=False, features_only=True)
        enc_channels = self.encoder.feature_info.channels()
        self.center = DecoderBlock(enc_channels[-1], enc_channels[-2], 512)
        self.dec4 = DecoderBlock(512, enc_channels[-3], 256)
        self.dec3 = DecoderBlock(256, enc_channels[-4], 128)
        self.dec2 = DecoderBlock(128, enc_channels[-5], 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = self.center(enc_feats[-1], enc_feats[-2])
        x = self.dec4(x, enc_feats[-3])
        x = self.dec3(x, enc_feats[-4])
        x = self.dec2(x, enc_feats[-5])
        x = self.dec1(x)
        return self.final(x)