import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


class UNet_Atten(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet_Atten, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_dw = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_dw = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_dw = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_dw = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_dw = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_dw = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_dn = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_dn = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_dn = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_dn = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_dn = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_dn = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_wm = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_wm = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_wm = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_wm = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_wm = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_wm = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

        self.avg_dn = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_dn = nn.Sequential(
            nn.Conv2d(144, 6, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 96, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.avg_wm = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_wm = nn.Sequential(
            nn.Conv2d(144, 6, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 144, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # 第一阶段，先去噪声
        # Encoder
        pool1_dn = self._block1_dn(x)  # 128x128
        pool2_dn = self._block2_dn(pool1_dn)  # 64x64
        pool3_dn = self._block2_dn(pool2_dn)  # 32x32
        pool4_dn = self._block2_dn(pool3_dn)  # 16x16
        pool5_dn = self._block2_dn(pool4_dn)  # 8x8

        # Decoder
        upsample5_dn = self._block3_dn(pool5_dn)  # 16x16
        concat5_dn = torch.cat((upsample5_dn, pool4_dn), dim=1)  # 在这里引入注意力机制

        upsample4_dn = self._block4_dn(concat5_dn)
        concat4_dn = torch.cat((upsample4_dn, pool3_dn), dim=1)
        upsample3_dn = self._block5_dn(concat4_dn)
        concat3_dn = torch.cat((upsample3_dn, pool2_dn), dim=1)
        upsample2_dn = self._block5_dn(concat3_dn)
        concat2_dn = torch.cat((upsample2_dn, pool1_dn), dim=1)
        upsample1_dn = self._block5_dn(concat2_dn)
        concat1_dn = torch.cat((upsample1_dn, x), dim=1)
        wm_x = self._block6_dn(concat1_dn)

        # 第二阶段，去水印
        # Encoder
        pool1_wm = self._block1_wm(wm_x)
        pool2_wm = self._block2_wm(pool1_wm)
        pool3_wm = self._block2_wm(pool2_wm)
        pool4_wm = self._block2_wm(pool3_wm)
        pool5_wm = self._block2_wm(pool4_wm)

        # Decoder
        upsample5_wm = self._block3_wm(pool5_wm)
        concat5_wm = torch.cat((upsample5_wm, pool4_wm), dim=1)
        upsample4_wm = self._block4_wm(concat5_wm)
        concat4_wm = torch.cat((upsample4_wm, pool3_wm), dim=1)
        upsample3_wm = self._block5_wm(concat4_wm)
        concat3_wm = torch.cat((upsample3_wm, pool2_wm), dim=1)
        upsample2_wm = self._block5_wm(concat3_wm)
        concat2_wm = torch.cat((upsample2_wm, pool1_wm), dim=1)
        upsample1_wm = self._block5_wm(concat2_wm)
        concat1_wm = torch.cat((upsample1_wm, x), dim=1)
        out_cascade = self._block6_wm(concat1_wm)

        # Encoder
        pool1_dw = self._block1_dw(x)
        pool2_dw = self._block2_dw(pool1_dw)
        pool3_dw = self._block2_dw(pool2_dw)
        pool4_dw = self._block2_dw(pool3_dw)
        pool5_dw = self._block2_dw(pool4_dw)

        # Decoder
        upsample5_dw = self._block3_dw(pool5_dw)
        concat5_dw = torch.cat((upsample5_dw, pool4_dw), dim=1)
        # Attention 去噪引入
        mid_dn = self.avg_dn(concat2_dn)
        Scale_dn = self.attn_dn(mid_dn)
        concat5_dw = concat5_dw * Scale_dn

        upsample4_dw = self._block4_dw(concat5_dw)
        concat4_dw = torch.cat((upsample4_dw, pool3_dw), dim=1)
        upsample3_dw = self._block5_dw(concat4_dw)
        concat3_dw = torch.cat((upsample3_dw, pool2_dw), dim=1)
        upsample2_dw = self._block5_dw(concat3_dw)
        concat2_dw = torch.cat((upsample2_dw, pool1_dw), dim=1)
        # Attention 去水印引入
        mid_wm = self.avg_wm(concat2_wm)
        Scale_wm = self.attn_wm(mid_wm)
        concat2_dw = concat2_dw * Scale_wm

        upsample1_dw = self._block5_dw(concat2_dw)
        concat1_dw = torch.cat((upsample1_dw, x), dim=1)
        out = self._block6_dw(concat1_dw)
        concat_out = torch.cat((out_cascade, out), dim=1)

        # Final activation
        return self.out(concat_out)


class UNet_Atten_3(nn.Module):
    """Custom U-Net architecture for Noise2Noise"""

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_Atten_3, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_dw = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_dw = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_dw = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_dw = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_dw = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_dw = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_dw2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_dw2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_dw2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_dw2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_dw2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_dw2 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_dn = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_dn = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_dn = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_dn = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_dn = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_dn = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1_wm = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2_wm = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3_wm = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_wm = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5_wm = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6_wm = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

        self.avg_dn = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_dn = nn.Sequential(
            nn.Conv2d(144, 6, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 144, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.avg_wm = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_wm = nn.Sequential(
            nn.Conv2d(144, 6, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 144, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        # Encoder
        pool1_dn = self._block1_dn(x)  # 128x128
        pool2_dn = self._block2_dn(pool1_dn)  # 64x64
        pool3_dn = self._block2_dn(pool2_dn)  # 32x32
        pool4_dn = self._block2_dn(pool3_dn)  # 16x16
        pool5_dn = self._block2_dn(pool4_dn)  # 8x8

        # Decoder
        upsample5_dn = self._block3_dn(pool5_dn)  # 16x16
        concat5_dn = torch.cat((upsample5_dn, pool4_dn), dim=1)

        upsample4_dn = self._block4_dn(concat5_dn)
        concat4_dn = torch.cat((upsample4_dn, pool3_dn), dim=1)
        upsample3_dn = self._block5_dn(concat4_dn)
        concat3_dn = torch.cat((upsample3_dn, pool2_dn), dim=1)
        upsample2_dn = self._block5_dn(concat3_dn)
        concat2_dn = torch.cat((upsample2_dn, pool1_dn), dim=1)
        upsample1_dn = self._block5_dn(concat2_dn)
        concat1_dn = torch.cat((upsample1_dn, x), dim=1)
        out_denoise = self._block6_dn(concat1_dn)

        # Encoder
        pool1_wm = self._block1_wm(out_denoise)
        pool2_wm = self._block2_wm(pool1_wm)
        pool3_wm = self._block2_wm(pool2_wm)
        pool4_wm = self._block2_wm(pool3_wm)
        pool5_wm = self._block2_wm(pool4_wm)

        # Decoder
        upsample5_wm = self._block3_wm(pool5_wm)
        concat5_wm = torch.cat((upsample5_wm, pool4_wm), dim=1)
        upsample4_wm = self._block4_wm(concat5_wm)
        concat4_wm = torch.cat((upsample4_wm, pool3_wm), dim=1)
        upsample3_wm = self._block5_wm(concat4_wm)
        concat3_wm = torch.cat((upsample3_wm, pool2_wm), dim=1)
        upsample2_wm = self._block5_wm(concat3_wm)
        concat2_wm = torch.cat((upsample2_wm, pool1_wm), dim=1)
        upsample1_wm = self._block5_wm(concat2_wm)
        concat1_wm = torch.cat((upsample1_wm, x), dim=1)
        out_wm = self._block6_wm(concat1_wm)

        # Encoder
        pool1_dw = self._block1_dw(x)
        pool2_dw = self._block2_dw(pool1_dw)
        pool3_dw = self._block2_dw(pool2_dw)
        pool4_dw = self._block2_dw(pool3_dw)
        pool5_dw = self._block2_dw(pool4_dw)

        # Decoder
        upsample5_dw = self._block3_dw(pool5_dw)
        concat5_dw = torch.cat((upsample5_dw, pool4_dw), dim=1)


        upsample4_dw = self._block4_dw(concat5_dw)
        concat4_dw = torch.cat((upsample4_dw, pool3_dw), dim=1)
        upsample3_dw = self._block5_dw(concat4_dw)
        concat3_dw = torch.cat((upsample3_dw, pool2_dw), dim=1)
        upsample2_dw = self._block5_dw(concat3_dw)
        concat2_dw = torch.cat((upsample2_dw, pool1_dw), dim=1)


        mid_dn = self.avg_dn(concat2_dn)
        Scale_dn = self.attn_dn(mid_dn)
        concat2_dw = concat2_dw * Scale_dn



        upsample1_dw = self._block5_dw(concat2_dw)
        concat1_dw = torch.cat((upsample1_dw, x), dim=1)
        main_out_mid = self._block6_dw(concat1_dw)
        # concat_out = torch.cat((out_wm, out), dim=1)

        # Encoder
        pool1_dw2 = self._block1_dw2(main_out_mid)
        pool2_dw2 = self._block2_dw2(pool1_dw2)
        pool3_dw2 = self._block2_dw2(pool2_dw2)
        pool4_dw2 = self._block2_dw(pool3_dw2)
        pool5_dw2 = self._block2_dw(pool4_dw2)

        # Decoder
        upsample5_dw2 = self._block3_dw(pool5_dw2)
        concat5_dw2 = torch.cat((upsample5_dw2, pool4_dw2), dim=1)


        upsample4_dw2 = self._block4_dw(concat5_dw2)
        concat4_dw2 = torch.cat((upsample4_dw2, pool3_dw2), dim=1)
        upsample3_dw2 = self._block5_dw(concat4_dw2)
        concat3_dw2 = torch.cat((upsample3_dw2, pool2_dw2), dim=1)
        upsample2_dw2 = self._block5_dw(concat3_dw2)
        concat2_dw2 = torch.cat((upsample2_dw2, pool1_dw2), dim=1)

        mid_wm = self.avg_wm(concat2_wm)
        Scale_wm = self.attn_wm(mid_wm)
        concat2_dw2 = concat2_dw2 * Scale_wm

        upsample1_dw2 = self._block5_dw2(concat2_dw2)
        concat1_dw2 = torch.cat((upsample1_dw2, x), dim=1)
        main_out_final = self._block6_dw2(concat1_dw2)
        # concat_out = torch.cat((out_wm, out), dim=1)

        # Final activation
        return main_out_final, out_denoise, out_wm
