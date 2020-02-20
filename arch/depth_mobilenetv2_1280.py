import os
import collections
import math

import torch
import torch.nn as nn
import torchvision.models as models
#Some modules inherit MobileNetV2
class BLConv(nn.Module):

    def __init__(self, kernel_size, depthwise_chose, out_channels=1280):
        super(BLConv, self).__init__(kernel_size, depthwise_chose, out_channels)
        if depthwise_chose:
            self.conv1 = nn.Sequential(
                depthwise(1280, kernel_size),
                pointwise(1280, out_channels//2))
            self.conv2 = nn.Sequential(
                depthwise(out_channels//2, kernel_size),
                pointwise(out_channels//2, out_channels//4))
            self.conv3 = nn.Sequential(
                depthwise(out_channels//4, kernel_size),
                pointwise(out_channels//4, out_channels//8))
            self.conv4 = nn.Sequential(
                depthwise(out_channels//8, kernel_size),
                pointwise(out_channels//8, out_channels//16))
            self.conv5 = nn.Sequential(
                depthwise(out_channels//16, kernel_size),
                pointwise(out_channels//16, out_channels//32))
            self.conv6 = pointwise(out_channels//32, 1)
        else:
            self.conv1 = conv(out_channels, out_channels//2, kernel_size)
            self.conv2 = conv(out_channels//2, out_channels//4, kernel_size)
            self.conv3 = conv(out_channels//4, out_channels//8, kernel_size)
            self.conv4 = conv(out_channels//8, out_channels//16, kernel_size)
            self.conv5 = conv(out_channels//16, out_channels//32, kernel_size)
            self.conv6 = pointwise(out_channels//32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv5(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv6(x)
        return x

def choose_decoder(decoder, channels):
    depthwise = ('depthwise_chose' in decoder)
    if decoder[:6] == 'blconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'depthwise_chose' in decoder)
        kernel_size = int(decoder[6])
        model = BLConv(kernel_size, depthwise_chose, channels)
    else:
        assert False, "Please use the decoder blconv7dw, and the option of decoder is invalid:{}".format(decoder)
    model.apply(weights_init)
    return model


class DepthMobileV2(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, out_channels=1280, pretrained=True):
        super(DepthMobileV2, self).__init__()

        model = models.mobilenet_v2(pretrained=pretrained)
        model.load_state_dict(torch.load('/home/vision/mobilenet_v2-b0353104.pth'))
        self.features = model.features
        del model

        self.output_size = output_size

        num_channels = 1280
        self.decoder = choose_decoder(decoder, out_channels)
        self.decoder.apply(weights_init)

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.decoder(x)
        return x


