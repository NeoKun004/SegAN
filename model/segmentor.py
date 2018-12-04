import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from .base import GlobalConvBlock, ResidualBlock, ResidualBlock_D, CHANNEL_DIM, NDF


class SegmentorNet(nn.Module):
    def __init__(self):
        super(SegmentorNet, self).__init__()
        self.convblock1 = nn.Sequential(
            # input is (CHANNEL_DIM) x 128 x 128
            nn.Conv2d(CHANNEL_DIM, NDF, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x 64 x 64
        )
        self.convblock1_1 = ResidualBlock(NDF)
        self.convblock2 = nn.Sequential(
            # state size. (NDF) x 64 x 64
            nn.Conv2d(NDF, NDF * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x 32 x 32
        )
        self.convblock2_1 = ResidualBlock(NDF * 2)
        self.convblock3 = nn.Sequential(
            # state size. (NDF*2) x 32 x 32
            nn.Conv2d(NDF * 2, NDF * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x 16 x 16
        )
        self.convblock3_1 = ResidualBlock(NDF * 4)
        self.convblock4 = nn.Sequential(
            # state size. (NDF*4) x 16 x 16
            nn.Conv2d(NDF * 4, NDF * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 8 x 8
        )
        self.convblock4_1 = ResidualBlock(NDF * 8)
        self.convblock5 = nn.Sequential(
            # state size. (NDF*8) x 8 x 8
            nn.Conv2d(NDF * 8, NDF * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 4 x 4
        )
        self.convblock5_1 = ResidualBlock(NDF * 8)
        self.convblock6 = nn.Sequential(
            # state size. (NDF*8) x 4 x 4
            nn.Conv2d(NDF * 8, NDF * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*16) x 2 x 2
        )
        self.convblock6_1 = nn.Sequential(
            # state size. (NDF*16) x 2 x 2
            nn.Conv2d(NDF * 16, NDF * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*16) x 2 x 2
        )
        self.convblock7 = nn.Sequential(
            # state size. (NDF*16) x 2 x 2
            nn.Conv2d(NDF * 16, NDF * 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*32) x 1 x 1
        )
        # self.convblock7_1 = ResidualBlock(NDF*32)
        self.convblock8 = nn.Sequential(
            # state size. (NDF*32) x 1 x 1
            nn.Conv2d(NDF * 32, NDF * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 1 x 1
        )

        self.deconvblock1 = nn.Sequential(
            # state size. (ngf*8) x 1 x 1
            nn.ConvTranspose2d(NDF * 8, NDF * 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(NDF * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 1 x 1
        )
        self.deconvblock2 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1
            nn.Conv2d(NDF * 64, NDF * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
        )
        self.deconvblock2_1 = nn.Sequential(
            # state size. (NDF*16) x 2 x 2
            nn.Conv2d(NDF * 16, NDF * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.ReLU(inplace=True),
            # state size. (NDF*16) x 2 x 2
        )
        self.deconvblock3 = nn.Sequential(
            # state size. (cat: ngf*16) x 2 x 2
            nn.Conv2d(NDF * 16 * 2, NDF * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
        )
        self.deconvblock3_1 = ResidualBlock_D(NDF * 8)
        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            GlobalConvBlock(NDF * 8 * 2, NDF * 8, (7, 7)),
            # nn.ConvTranspose2d(NDF * 8 * 2, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
        )
        self.deconvblock4_1 = ResidualBlock_D(NDF * 8)
        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            GlobalConvBlock(NDF * 8 * 2, NDF * 4, (7, 7)),
            # nn.ConvTranspose2d(NDF * 8 * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
        )
        self.deconvblock5_1 = ResidualBlock_D(NDF * 4)
        self.deconvblock6 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16
            GlobalConvBlock(NDF * 4 * 2, NDF * 2, (9, 9)),
            # nn.ConvTranspose2d(NDF * 4 * 2, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
        )
        self.deconvblock6_1 = ResidualBlock_D(NDF * 2)
        self.deconvblock7 = nn.Sequential(
            # state size. (ngf*2) x 32 x 32
            GlobalConvBlock(NDF * 2 * 2, NDF, (9, 9)),
            # nn.ConvTranspose2d(NDF * 2 * 2,     NDF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
        )
        self.deconvblock7_1 = ResidualBlock_D(NDF)
        self.deconvblock8 = nn.Sequential(
            # state size. (ngf) x 64 x 64
            GlobalConvBlock(NDF * 2, NDF, (11, 11)),
            # nn.ConvTranspose2d( NDF * 2, NDF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
        )
        self.deconvblock8_1 = ResidualBlock_D(NDF)
        self.deconvblock9 = nn.Sequential(
            # state size. (ngf) x 128 x 128
            nn.Conv2d(NDF, 1, 5, 1, 2, bias=False),
            # state size. (CHANNEL_DIM) x 128 x 128
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder1 = self.convblock1_1(encoder1)
        encoder2 = self.convblock2(encoder1)
        encoder2 = self.convblock2_1(encoder2)
        encoder3 = self.convblock3(encoder2)
        encoder3 = self.convblock3_1(encoder3)
        encoder4 = self.convblock4(encoder3)
        encoder4 = self.convblock4_1(encoder4)
        encoder5 = self.convblock5(encoder4)
        encoder5 = self.convblock5_1(encoder5)
        encoder6 = self.convblock6(encoder5)
        encoder6 = self.convblock6_1(encoder6) + encoder6
        encoder7 = self.convblock7(encoder6)
        encoder8 = self.convblock8(encoder7)

        decoder1 = self.deconvblock1(encoder8)
        decoder1 = torch.cat([encoder7, decoder1], 1)
        decoder1 = F.upsample(decoder1, size=encoder6.size()[2:], mode='bilinear')
        decoder2 = self.deconvblock2(decoder1)
        decoder2 = self.deconvblock2_1(decoder2) + decoder2
        # concatenate along depth dimension
        decoder2 = torch.cat([encoder6, decoder2], 1)
        decoder2 = F.upsample(decoder2, size=encoder5.size()[2:], mode='bilinear')
        decoder3 = self.deconvblock3(decoder2)
        decoder3 = self.deconvblock3_1(decoder3)
        decoder3 = torch.cat([encoder5, decoder3], 1)
        decoder3 = F.upsample(decoder3, size=encoder4.size()[2:], mode='bilinear')
        decoder4 = self.deconvblock4(decoder3)
        decoder4 = self.deconvblock4_1(decoder4)
        decoder4 = torch.cat([encoder4, decoder4], 1)
        decoder4 = F.upsample(decoder4, size=encoder3.size()[2:], mode='bilinear')
        decoder5 = self.deconvblock5(decoder4)
        decoder5 = self.deconvblock5_1(decoder5)
        decoder5 = torch.cat([encoder3, decoder5], 1)
        decoder5 = F.upsample(decoder5, size=encoder2.size()[2:], mode='bilinear')
        decoder6 = self.deconvblock6(decoder5)
        decoder6 = self.deconvblock6_1(decoder6)
        decoder6 = torch.cat([encoder2, decoder6], 1)
        decoder6 = F.upsample(decoder6, size=encoder1.size()[2:], mode='bilinear')
        decoder7 = self.deconvblock7(decoder6)
        decoder7 = self.deconvblock7_1(decoder7)
        decoder7 = torch.cat([encoder1, decoder7], 1)
        decoder7 = F.upsample(decoder7, size=input.size()[2:], mode='bilinear')
        decoder8 = self.deconvblock8(decoder7)
        decoder8 = self.deconvblock8_1(decoder8)
        decoder9 = self.deconvblock9(decoder8)

        return decoder9
