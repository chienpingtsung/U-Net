from torch import nn

from models.UNet import double_conv, copy_and_crop


class WNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WNet, self).__init__()
        self.use_edge_net = True
        self.use_seg_net = True

        # edge net
        self.edge_cont1 = double_conv(in_channels, 64)
        self.edge_maxpool1 = nn.MaxPool2d(2)
        self.edge_cont2 = double_conv(64, 128)
        self.edge_maxpool2 = nn.MaxPool2d(2)
        self.edge_cont3 = double_conv(128, 256)
        self.edge_maxpool3 = nn.MaxPool2d(2)

        self.edge_bottleneck = double_conv(256, 512)

        self.edge_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.edge_expa1 = double_conv(512, 256)
        self.edge_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.edge_expa2 = double_conv(256, 128)
        self.edge_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.edge_expa3 = double_conv(128, 64)

        self.edge_output = nn.Conv2d(64, out_channels, kernel_size=1)

        # seg net
        self.seg_cont1 = double_conv(in_channels + 64, 64)
        self.seg_maxpool1 = nn.MaxPool2d(2)
        self.seg_cont2 = double_conv(64 + 128, 128)
        self.seg_maxpool2 = nn.MaxPool2d(2)
        self.seg_cont3 = double_conv(128 + 256, 256)
        self.seg_maxpool3 = nn.MaxPool2d(2)

        self.seg_bottleneck = double_conv(256 + 512, 512)

        self.seg_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_expa1 = double_conv(512, 256)
        self.seg_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_expa2 = double_conv(256, 128)
        self.seg_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_expa3 = double_conv(128, 64)

        self.seg_output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward_edge(self, x):
        x1 = self.edge_cont1(x)
        x = self.edge_maxpool1(x1)
        x2 = self.edge_cont2(x)
        x = self.edge_maxpool2(x2)
        x3 = self.edge_cont3(x)
        x = self.edge_maxpool3(x3)

        x = self.edge_bottleneck(x)

        x = self.edge_up1(x)
        x = self.edge_expa1(copy_and_crop(x3, x))
        x = self.edge_up2(x)
        x = self.edge_expa2(copy_and_crop(x2, x))
        x = self.edge_up3(x)
        x = self.edge_expa3(copy_and_crop(x1, x))

        return self.edge_output(x)

    def forward_seg(self, x):
        """UNAVAILABLE.

        :param x:
        :return:
        """
        x1 = self.seg_cont1(x)
        x = self.seg_maxpool1(x1)
        x2 = self.seg_cont2(x)
        x = self.seg_maxpool2(x2)
        x3 = self.seg_cont3(x)
        x = self.seg_maxpool3(x3)

        x = self.seg_bottleneck(x)

        x = self.seg_up1(x)
        x = self.seg_expa1(copy_and_crop(x3, x))
        x = self.seg_up2(x)
        x = self.seg_expa2(copy_and_crop(x2, x))
        x = self.seg_up3(x)
        x = self.seg_expa3(copy_and_crop(x1, x))

        return self.seg_output(x)

    def forward_fusion(self, x):
        # edge net
        y1 = self.edge_cont1(x)
        y = self.edge_maxpool1(y1)
        y2 = self.edge_cont2(y)
        y = self.edge_maxpool2(y2)
        y3 = self.edge_cont3(y)
        y = self.edge_maxpool3(y3)

        y4 = self.edge_bottleneck(y)

        # seg net
        x1 = self.seg_cont1(copy_and_crop(y1, x))
        x = self.seg_maxpool1(x1)
        x2 = self.seg_cont2(copy_and_crop(y2, x))
        x = self.seg_maxpool2(x2)
        x3 = self.seg_cont3(copy_and_crop(y3, x))
        x = self.seg_maxpool3(x3)

        x = self.seg_bottleneck(copy_and_crop(y4, x))

        x = self.seg_up1(x)
        x = self.seg_expa1(copy_and_crop(x3, x))
        x = self.seg_up2(x)
        x = self.seg_expa2(copy_and_crop(x2, x))
        x = self.seg_up3(x)
        x = self.seg_expa3(copy_and_crop(x1, x))

        return self.seg_output(x)

    def forward(self, x):
        if self.use_edge_net and not self.use_seg_net:
            return self.forward_edge(x)

        if not self.use_edge_net and self.use_seg_net:
            raise RuntimeError('forward_seg is unavailable now.')

        if self.use_edge_net and self.use_seg_net:
            return self.forward_fusion(x)

        return x

    def requires_grad_for_layers(self, rg: bool, prefix: str = ''):
        for _, param in self.named_parameters(prefix):
            param.requires_grad = rg
