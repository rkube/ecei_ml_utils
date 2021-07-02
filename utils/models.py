# -*- Encoding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Create a U-Net architecture for semantic segmentation
class Block(nn.Module):
    """Block at each level of the U"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="zeros")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="zeros")

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(nn.Module):
    """Left part of the U, a number of Blocks"""
    def __init__(self, ch_list=(1, 4, 8, 16)):
        super().__init__()
        num_ch = len(ch_list)
        self.encoder_blocks = nn.ModuleList([Block(ch_list[i], ch_list[i+1]) for i in range(num_ch-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.encoder_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, ch_list=(16, 8, 4)):
        super().__init__()
        self.ch_list = ch_list
        num_ch = len(ch_list)
        # Up-convolutions
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(self.ch_list[i], self.ch_list[i+1], 2, 2)
                                      for i in range(num_ch-1)])
        self.dec_blocks = nn.ModuleList([Block(ch_list[i], ch_list[i+1]) for i in range(num_ch-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.ch_list) - 1):
            # Up-convolution
            x = self.upconvs[i](x)
            # Concatenate with corresponding encoder-feature map
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            # Perform convolutions
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        _, _, H, W = x.shape
        enc_features = torchvision.transforms.CenterCrop([H, W])(enc_features)
        return enc_features

class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 4, 8, 16), dec_chs=(16, 8, 4), num_class=5, retain_dim=False, out_size=(24,8)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_size)
        return out


# End of file models.py
