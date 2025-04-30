
import torch
import torch.nn as nn
import timm

class AdaptiveEdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveEdgeAttention, self).__init__()
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        edge_map = self.sigmoid(self.edge_conv(x))
        edge_weighted = x * edge_map
        return self.refine(edge_weighted)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MobileViT_QRC_U_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MobileViT_QRC_U_Net, self).__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True, in_chans=in_channels)
        enc_channels = self.encoder.feature_info.channels()

        self.bridge = AdaptiveEdgeAttention(enc_channels[-1])
        self.up4 = nn.ConvTranspose2d(enc_channels[-1], enc_channels[-2], kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(enc_channels[-2]*2, enc_channels[-2])
        self.up3 = nn.ConvTranspose2d(enc_channels[-2], enc_channels[-3], kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(enc_channels[-3]*2, enc_channels[-3])
        self.up2 = nn.ConvTranspose2d(enc_channels[-3], enc_channels[-4], kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(enc_channels[-4]*2, enc_channels[-4])
        self.up1 = nn.ConvTranspose2d(enc_channels[-4], enc_channels[-4]//2, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(enc_channels[-4]//2, enc_channels[-4]//2)
        self.out_conv = nn.Conv2d(enc_channels[-4]//2, out_channels, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = self.bridge(enc_feats[-1])
        x = self.up4(x); x = torch.cat([x, enc_feats[-2]], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, enc_feats[-3]], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, enc_feats[-4]], dim=1); x = self.dec2(x)
        x = self.up1(x); x = self.dec1(x)
        x = torch.sigmoid(self.out_conv(x))
        return self.final_upsample(x)
