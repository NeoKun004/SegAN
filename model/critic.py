import torch
import torch.nn as nn

from math import sqrt

from .base import GlobalConvBlock, ResidualBlock, ResidualBlock_D, CHANNEL_DIM, NDF


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.convblock1 = nn.Sequential(
            # input is (CHANNEL_DIM) x 128 x 128
            nn.Conv2d(CHANNEL_DIM, NDF, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (NDF) x 64 x 64
            GlobalConvBlock(NDF, NDF * 2, (13, 13)),
            # nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (NDF * 2) x 64 x 64
            nn.Conv2d(NDF * 1, NDF * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (NDF*2) x 32 x 32
            GlobalConvBlock(NDF * 2, NDF * 4, (11, 11)),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (NDF * 4) x 32 x 32
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (NDF*4) x 16 x 16
            GlobalConvBlock(NDF * 4, NDF * 8, (9, 9)),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (NDF*4) x 16 x 16
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (NDF*8) x 8 x 8
            GlobalConvBlock(NDF * 8, NDF * 16, (7, 7)),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (NDF*8) x 8 x 8
            nn.Conv2d(NDF * 8, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (NDF*16) x 4 x 4
            nn.Conv2d(NDF * 16, NDF * 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(NDF * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (NDF*32) x 4 x 4
            nn.Conv2d(NDF * 8, NDF * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (NDF*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        batchsize = input.size()[0]
        out1 = self.convblock1(input)
        # out1 = self.convblock1_1(out1)
        out2 = self.convblock2(out1)
        # out2 = self.convblock2_1(out2)
        out3 = self.convblock3(out2)
        # out3 = self.convblock3_1(out3)
        out4 = self.convblock4(out3)
        # out4 = self.convblock4_1(out4)
        out5 = self.convblock5(out4)
        # out5 = self.convblock5_1(out5)
        out6 = self.convblock6(out5)
        # out6 = self.convblock6_1(out6) + out6
        output = torch.cat((input.view(batchsize, -1), 1*out1.view(batchsize, -1),
                            2*out2.view(batchsize, -1), 2*out3.view(batchsize, -1),
                            2*out4.view(batchsize, -1), 2*out5.view(batchsize, -1),
                            4*out6.view(batchsize, -1)), 1)

        return output
