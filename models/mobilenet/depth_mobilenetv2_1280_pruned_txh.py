import os
import collections
import math

import torch
import torch.nn as nn
import torchvision.models as models

#Some modules inherit MobileNetV2
class BLConv(nn.Module):

    def __init__(self, kernel_size, depthw, out_channels=1280):
        super(BLConv, self).__init__(kernel_size, depthw, out_channels)
        if depthw:
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
    depthwise = ('depthw' in decoder)
    if decoder[:6] == 'blconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'depthw' in decoder)
        kernel_size = int(decoder[6])
        model = BLConv(kernel_size, depthwise, channels)
    else:
        assert False, "Please use the decoder blconv7dw, and the option of decoder is invalid: {}".format(decoder)
    model.apply(weights_init)
    return model


class DepthMobileV2(nn.Module):
    def __init__(self, pruning_index, pruned_channels, decoder="blconv7dw", output_size=224, in_channels=3, out_channels=1280, pretrained=True):
        super(DepthMobileV2, self).__init__()

        from .mobilenet_v2_pruned_txh import get_DepthMobileNetV2_Pruned
        model = get_DepthMobileNetV2_Pruned(pruning_index[:36], pruned_channels[:36])
        self.features = model.features
        del model
        m1 = torch.load("results/model_best.pth_v2_0.3imagenet_prunede.tar")
        model_params = m1['state_dict']
        for i, m in list(model_params.items()):
            if "classifier" in i:
                model_params.pop(i)
        self.features.load_state_dict({k.replace('module.features.', ''): v for k, v in model_params.items()})


        self.output_size = output_size

        num_channels = 1280

        self.decoder = choose_decoder(decoder, out_channels)
        # pruning_index[35:], pruned_channels[35:]   =  1+5
        self.decoder.conv1[0][0] = nn.Conv2d(pruned_channels[35], pruned_channels[35], kernel_size=(7, 7), stride=(1, 1),
                                            padding=(3, 3), groups=pruned_channels[35], bias=False)
        self.decoder.conv1[0][1] = nn.BatchNorm2d(pruned_channels[35], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)
        self.decoder.conv1[1][0] = nn.Conv2d(pruned_channels[35], pruned_channels[36], kernel_size=(1, 1), stride=(1, 1),
                                            bias=False)
        self.decoder.conv1[1][1] = nn.BatchNorm2d(pruned_channels[36], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)

        self.decoder.conv2[0][0] = nn.Conv2d(pruned_channels[36], pruned_channels[36], kernel_size=(7, 7), stride=(1, 1),
                                            padding=(3, 3), groups=pruned_channels[36], bias=False)
        self.decoder.conv2[0][1] = nn.BatchNorm2d(pruned_channels[36], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)
        self.decoder.conv2[1][0] = nn.Conv2d(pruned_channels[36], pruned_channels[37], kernel_size=(1, 1), stride=(1, 1),
                                            bias=False)
        self.decoder.conv2[1][1] = nn.BatchNorm2d(pruned_channels[37], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)

        self.decoder.conv3[0][0] = nn.Conv2d(pruned_channels[37], pruned_channels[37], kernel_size=(7, 7), stride=(1, 1),
                                            padding=(3, 3), groups=pruned_channels[37], bias=False)
        self.decoder.conv3[0][1] = nn.BatchNorm2d(pruned_channels[37], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)
        self.decoder.conv3[1][0] = nn.Conv2d(pruned_channels[37], pruned_channels[38], kernel_size=(1, 1), stride=(1, 1),
                                            bias=False)
        self.decoder.conv3[1][1] = nn.BatchNorm2d(pruned_channels[38], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)

        self.decoder.conv4[0][0] = nn.Conv2d(pruned_channels[38], pruned_channels[38], kernel_size=(7, 7), stride=(1, 1),
                                            padding=(3, 3), groups=pruned_channels[38], bias=False)
        self.decoder.conv4[0][1] = nn.BatchNorm2d(pruned_channels[38], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)
        self.decoder.conv4[1][0] = nn.Conv2d(pruned_channels[38], pruned_channels[39], kernel_size=(1, 1), stride=(1, 1),
                                            bias=False)
        self.decoder.conv4[1][1] = nn.BatchNorm2d(pruned_channels[39], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)

        self.decoder.conv5[0][0] = nn.Conv2d(pruned_channels[39], pruned_channels[39], kernel_size=(7, 7), stride=(1, 1),
                                            padding=(3, 3), groups=pruned_channels[39], bias=False)
        self.decoder.conv5[0][1] = nn.BatchNorm2d(pruned_channels[39], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)
        self.decoder.conv5[1][0] = nn.Conv2d(pruned_channels[39], pruned_channels[40], kernel_size=(1, 1), stride=(1, 1),
                                            bias=False)
        self.decoder.conv5[1][1] = nn.BatchNorm2d(pruned_channels[40], eps=1e-05, momentum=0.1, affine=True,
                                                  track_running_stats=True)

        self.decoder.conv6[0] = nn.Conv2d(pruned_channels[40], 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.decoder.apply(weights_init)

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.decoder(x)
        return x


