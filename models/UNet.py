import torch
from torch import nn
from torch.nn import functional


def two_conv(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def copy_and_crop(cont_map, expa_map):
    gap = (cont_map.shape[2] - expa_map.shape[2]) // 2
    cont_map = functional.pad(cont_map, [-gap, -gap, -gap, -gap])
    return torch.cat((cont_map, expa_map), dim=1)


class UNet(nn.Module):
    """
    Architecture described by Figure 1 of
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # contracting path
        self.cont1 = two_conv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2)
        self.cont2 = two_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)
        self.cont3 = two_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)
        self.cont4 = two_conv(256, 512)
        self.maxpool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = two_conv(512, 1024)

        # expansive path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.expa1 = two_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.expa2 = two_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.expa3 = two_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.expa4 = two_conv(128, 64)

        # output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # contracting path
        x1 = self.cont1(x)
        x = self.maxpool1(x1)
        x2 = self.cont2(x)
        x = self.maxpool2(x2)
        x3 = self.cont3(x)
        x = self.maxpool3(x3)
        x4 = self.cont4(x)
        x = self.maxpool4(x4)

        # bottleneck
        x = self.bottleneck(x)

        # expansive path
        x = self.up1(x)
        x = self.expa1(copy_and_crop(x4, x))
        x = self.up2(x)
        x = self.expa2(copy_and_crop(x3, x))
        x = self.up3(x)
        x = self.expa3(copy_and_crop(x2, x))
        x = self.up4(x)
        x = self.expa4(copy_and_crop(x1, x))

        return self.output(x)


class UNetM(nn.Module):
    """
    Simplified architecture described by Figure 1 of
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, in_channels, out_channels):
        super(UNetM, self).__init__()

        # contracting path
        self.cont1 = two_conv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2)
        self.cont2 = two_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)
        self.cont3 = two_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)
        # self.cont4 = two_conv(256, 512)
        # self.maxpool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = two_conv(256, 512)

        # expansive path
        # self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.expa1 = two_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.expa2 = two_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.expa3 = two_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.expa4 = two_conv(128, 64)

        # output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # contracting path
        x1 = self.cont1(x)
        x = self.maxpool1(x1)
        x2 = self.cont2(x)
        x = self.maxpool2(x2)
        x3 = self.cont3(x)
        x = self.maxpool3(x3)
        # x4 = self.cont4(x)
        # x = self.maxpool4(x4)

        # bottleneck
        x = self.bottleneck(x)

        # expansive path
        # x = self.up1(x)
        # x = self.expa1(copy_and_crop(x4, x))
        x = self.up2(x)
        x = self.expa2(copy_and_crop(x3, x))
        x = self.up3(x)
        x = self.expa3(copy_and_crop(x2, x))
        x = self.up4(x)
        x = self.expa4(copy_and_crop(x1, x))

        return self.output(x)
