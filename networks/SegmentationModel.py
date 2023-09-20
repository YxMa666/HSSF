import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch import nn
from torch.nn import functional as F
# from .batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
BatchNorm2d = nn.BatchNorm2d


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        self.last_conv1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )
        
        self.confidence_1 = nn.Sequential(
            nn.Conv2d(258, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.confidence_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.ap2 = nn.AdaptiveAvgPool2d(1)
        self.confidence_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.confidence_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
       
        self._init_weight()

    def forward(self, feature,need_fp=False):
        if need_fp:
            output_feature = self.aspp(torch.cat((feature['out'],nn.Dropout2d(0.5)(feature['out']))))
            low_level_feature = self.project(torch.cat((feature['low_level'],nn.Dropout2d(0.5)(feature['low_level']))))
        else:
            output_feature = self.aspp(feature['out'])
            low_level_feature = self.project( feature['low_level'] )
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat([ low_level_feature, output_feature], dim=1)
        output_feature = self.last_conv1(output_feature)
        con1 = self.confidence_1(torch.cat([output_feature,self.classifier(output_feature)],dim=1))
        con1 = con1+con1*self.ap1(con1)
        # con1 = con1+con1*torch.sigmoid(self.ap1(con1))
        con2 = (con1+self.confidence_2(torch.cat([con1,output_feature],dim=1)))
        con2 = con2+con2*self.ap2(con2)
        # con2 = con2+con2*torch.sigmoid(self.ap2(con2))
        con3 = (con2+self.confidence_3(torch.cat([con1,con2],dim=1)))
        con4 = self.confidence_4(torch.cat([con3,con2],dim=1))
        
        return {'mask':self.classifier(output_feature),'confidence':con4}
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return 

class SegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x,need_fp=False):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features,need_fp)
        mask = F.interpolate(x['mask'], size=input_shape, mode='bilinear', align_corners=False)
        confidence = F.interpolate(x['confidence'], size=input_shape, mode='bilinear', align_corners=False)
        if need_fp:
            return {'mask':mask.chunk(2),'confidence':confidence,'proto':x['mask'].chunk(2)}
        else:
            return {'mask':mask,'confidence':confidence,'proto':x['mask']}

if __name__ =='__main__':
    from backbone.resnet import resnet34 #networks.
    bkbone = resnet34().cuda()
    head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,kl_fea_size=(64,72,72),num_classes=1)
    model =  SegmentationModel(bkbone,head,kl_fea_size=(64,72,72)).cuda()
    
    from backbone.xception import xception #networks.
    bkbone = xception(pretrained=False).cuda()
    head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=64,kl_fea_size=(64,72,72),num_classes=1)
    model =  SegmentationModel(bkbone,head,kl_fea_size=(64,72,72)).cuda()
    
    from backbone.mobilenetv2 import mobilenet_v2 #networks.
    bkbone = mobilenet_v2(pretrained=False).cuda()
    head = DeepLabHeadV3Plus(in_channels=1280,low_level_channels=24,kl_fea_size=(64,72,72),num_classes=1)
    model =  SegmentationModel(bkbone,head,kl_fea_size=(64,72,72)).cuda()
    
    from backbone.vgg import VGG16 #networks.
    bkbone = VGG16(pretrained=False).cuda()
    head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,kl_fea_size=(64,72,72),num_classes=1)
    model =  SegmentationModel(bkbone,head,kl_fea_size=(64,72,72)).cuda()
    
    from backbone.pvt import pvt_v2_b2 #networks.
    bkbone = pvt_v2_b2().cuda()
    head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,kl_fea_size=(64,72,72),num_classes=1)
    model =  SegmentationModel(bkbone,head,kl_fea_size=(64,72,72)).cuda()
    
    img = torch.randn((2,3,384,384)).cuda()
    mask_pre,boundary_pre,kl_fea = model(img)
    
    print(mask_pre,boundary_pre,kl_fea)
    print(mask_pre.shape,boundary_pre.shape,kl_fea)