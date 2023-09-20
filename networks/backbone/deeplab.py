import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import build_decoder
from .resnet import *


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        # if backbone == "drn":
        #     inplanes = 512
        # elif backbone == "mobilenet":
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            #  BatchNorm(256),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    # @torch.cuda.amp.autocast()
    def forward(self, x, gt_mode=False):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if gt_mode is False:
            return self.dropout(x)
        else:
            return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(inplanes, output_stride, BatchNorm):
    return ASPP(inplanes, output_stride, BatchNorm)


class classifier(nn.Module):
    def __init__(self, num_classes=21, out_dim=256):
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, num_classes, kernel_size=1, stride=1),
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x, expand=None):
        cls = self.classifier(x)
        return cls  # b x num_class


class kl(nn.Module):
    def __init__(self, num_classes,kc):
        super(kl, self).__init__()

        self.kl_exact = nn.Sequential(
            nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, kc, 1)
        )
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.kl_exact(x)
        return x

class boundary(nn.Module):
    def __init__(self, low_level_channels,num_cls):
        super(boundary, self).__init__()

        self.bound = nn.Sequential(
            nn.Conv2d(low_level_channels, 256, 3, padding=1, bias=False),
            SynchronizedBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_cls, 1)
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.bound(x)
        return x

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, resnet_name=101):
        super(DeepLab, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d

        if resnet_name == 'resnet101':
            # self.backbone = ResNet101(output_stride, BatchNorm) ####
            self.backbone = resnet101(pretrained=False, progress=True) ####
        elif resnet_name == 'resnet50':
            # self.backbone = ResNet50(output_stride, BatchNorm)
            self.backbone = resnet50(pretrained=False, progress=True)

    # @torch.cuda.amp.autocast()
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        return x, low_level_feat

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class confidence(nn.Module):
    def __init__(self,  num_classes, num_cls):
        super(confidence, self).__init__()
        self.confidence_1 = nn.Sequential(
            nn.Conv2d(num_classes+num_cls, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.confidence_2 = nn.Sequential(
            nn.Conv2d(num_classes*2, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.ap2 = nn.AdaptiveAvgPool2d(1)
        self.confidence_3 = nn.Sequential(
            nn.Conv2d(num_classes*2, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.confidence_4 = nn.Sequential(
            nn.Conv2d(num_classes*2, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            SynchronizedBatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, 1, 1),
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x,cls, expand=None):
        con1 = self.confidence_1(torch.cat([x,cls],dim=1))
        con1 = con1+con1*self.ap1(con1)
        con2 = (con1+self.confidence_2(torch.cat([con1,x],dim=1)))
        con2 = con2+con2*self.ap2(con2)
        con3 = (con2+self.confidence_3(torch.cat([con2,con1],dim=1)))
        con4 = self.confidence_4(torch.cat([con3,con2],dim=1))
        return con4  # b x num_class

class Decoder(nn.Module):
    def __init__(
        self, in_channels, low_level_channels=256,output_stride=16, mid_channel=256, num_classes=21, kl_fea_size=(128,32,32)
    ):
        BatchNorm = SynchronizedBatchNorm2d

        super(Decoder, self).__init__()
        self.aspp = build_aspp(in_channels, output_stride, BatchNorm)
        self.decoder = build_decoder(mid_channel, low_level_channels, BatchNorm)
        self.cls = classifier(num_classes)
        self.kl = kl(mid_channel,kl_fea_size[0])
        self.confide = confidence(mid_channel, num_classes)
        self.boundary = boundary(low_level_channels,num_classes)

    # @torch.cuda.amp.autocast()
    def forward(self, feature, gt_mode=False):
        x = feature['out']
        low_level_feat = feature['low_level']
        x = self.aspp(x, gt_mode)
        x_ = self.decoder(x, low_level_feat, gt_mode)
        bound = self.boundary(low_level_feat)
        cls_s = self.cls(x_)
        kl = self.kl(x_)
        con = self.confide(x_,cls_s)
        return {'mask':cls_s,'boundary':bound \
            ,'kl':kl,'confidence':con}

    def get_other_params(self):
        modules = [self.aspp, self.decoder, self.cls, self.proj]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p