import torch
import torch.nn as nn
from modules.ASPP import ASPP
import backbone.resnet.resnet as resnet
import torch.nn.functional as F
from modules.SCBlock import SCBlock

# P66_20
# 将err_map扩张
# 矫正模块：split and concat as guidence
# feature_embedding
# 矫正模块:空间att


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, flag=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)
        self.flag = flag

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.flag == 1:
            x = self.relu(x)
        return x


class ConvTwoD(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvTwoD, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, x):
        x = self.conv(x)
        return x


class MyBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(MyBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    # ratio = 16
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.ratio = ratio

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
        out = max_out  # 32 1 1
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out  # 1 44 44
        x = self.conv1(x)
        return self.sigmoid(x)


# like Mobile-Former 双向桥接模块
class BiTrans(nn.Module):
    def __init__(self, inplanes):
        super(BiTrans, self).__init__()

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.B1 = MyBottleNeck(inplanes, inplanes)
        self.f1_spatial = SpatialAttention()

        self.B2 = ASPP(inplanes, inplanes)
        self.f2_channel = ChannelAttention(inplanes)

        self.conv_cat1 = BasicConv2d(inplanes * 2, inplanes, 3, 1, padding=1, flag=1)
        self.conv_cat2 = BasicConv2d(inplanes * 2, inplanes, 3, 1, padding=1, flag=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, f1, f2):

        f2 = self.upsample_2(f2)

        # T2 channel
        temp_2 = f2.mul(self.f2_channel(f2))
        f1 = self.conv_cat1(torch.cat((f1, temp_2), dim=1))
        f_B1 = self.B1(f1)
        f1_out = f_B1

        f_B2 = self.B2(f2)
        temp_1 = f_B1.mul(self.f1_spatial(f_B1))
        f2_out = self.conv_cat2(torch.cat((temp_1, f_B2), dim=1))

        return f1_out, f2_out


class GenerateMessage(nn.Module):
    def __init__(self, inplanes=32, theta=3):
        super(GenerateMessage, self).__init__()

        self.theta = theta

        self.output_err = nn.Conv2d(inplanes, 1, 1)
        self.output_coarse = nn.Conv2d(inplanes, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, fea):
        coarse_pre = self.output_coarse(fea)  # 1 x 88 x 88
        coarse_pre_att = torch.sigmoid(coarse_pre)

        err_map = self.output_err(fea)  # 1 x 88 x 88
        err_map_att = torch.sigmoid(err_map)
        # extended err map
        err_map_ext = F.max_pool2d(
            err_map_att, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        return coarse_pre, err_map, coarse_pre_att, err_map_att, err_map_ext


class G_Fuse(nn.Module):
    def __init__(self, inplanes, sub_channel):
        super(G_Fuse, self).__init__()

        self.sc_1 = SCBlock(inplanes, sub_channel)
        self.sc_2 = SCBlock(inplanes, sub_channel)

        self.cbr_1 = BasicConv2d(inplanes, inplanes, 3, 1, padding=1, flag=1)
        self.cbr_2 = BasicConv2d(inplanes, inplanes, 3, 1, padding=1, flag=1)

        self.weight = SpatialAttention()

    def forward(self, fea, coarse, err):
        sc_coarse = self.sc_1(fea, coarse)
        sc_err = self.sc_2(fea, err)

        sc_coarse = self.cbr_1(sc_coarse)
        sc_err = self.cbr_2(sc_err)

        fuse = sc_coarse + sc_err
        fuse_att = self.weight(fuse)
        res = fuse_att * fea + fea
        return res


class EmbingBlock(nn.Module):
    def __init__(self, inplanes):
        super(EmbingBlock, self).__init__()
        self.conva = ConvTwoD(inplanes, inplanes, 3, 1, padding=1)
        self.convb = ConvTwoD(inplanes, inplanes, 3, 1, padding=1)

    def forward(self, fea):
        fea_a = self.conva(fea)
        fea_b = self.convb(fea)
        return fea_a, fea_b


class SAM_ResNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, backbone_path='./ckpt/resnet50-19c8e357.pth'):
        super(SAM_ResNet, self).__init__()

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # channel reduction
        # Conv+BN+Relu
        self.cbr1 = BasicConv2d(256, 128, 3, 1, padding=1, flag=1)
        self.cbr2 = BasicConv2d(512, 128, 3, 1, padding=1, flag=1)
        self.cbr3 = BasicConv2d(1024, 256, 3, 1, padding=1, flag=1)
        self.cbr4 = BasicConv2d(2048, 512, 3, 1, padding=1, flag=1)

        # channel reduction2
        self.cbr0_1 = BasicConv2d(64, 64, 3, padding=1, flag=1)
        self.cbr1_1 = BasicConv2d(128, 64, 3, padding=1, flag=1)
        self.cbr2_1 = BasicConv2d(128, 64, 3, padding=1, flag=1)
        self.cbr3_1 = BasicConv2d(256, 64, 3, padding=1, flag=1)
        self.cbr4_1 = BasicConv2d(512, 64, 3, padding=1, flag=1)

        self.BiStage0 = BiTrans(64)
        self.BiStage1 = BiTrans(64)
        self.BiStage2 = BiTrans(64)
        self.BiStage3 = BiTrans(64)

        # channel reduction3
        self.cbr1_2 = BasicConv2d(128, 32, 3, padding=1, flag=1)
        self.cbr2_2 = BasicConv2d(128, 32, 3, padding=1, flag=1)
        self.cbr3_2 = BasicConv2d(128, 32, 3, padding=1, flag=1)
        self.cbr4_2 = BasicConv2d(128, 32, 3, padding=1, flag=1)

        # upsample
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # CBR
        self.cbr4_3 = BasicConv2d(32, 32, 3, padding=1, flag=1)
        self.cbr3_3 = BasicConv2d(64, 32, 3, padding=1, flag=1)
        self.cbr2_3 = BasicConv2d(64, 32, 3, padding=1, flag=1)
        self.cbr1_3 = BasicConv2d(64, 32, 3, padding=1, flag=1)

        self.emb_1 = EmbingBlock(32)
        self.emb_2 = EmbingBlock(32)
        self.emb_3 = EmbingBlock(32)
        self.emb_4 = EmbingBlock(32)

        self.generate_message4 = GenerateMessage(inplanes=32)
        self.generate_message3 = GenerateMessage(inplanes=32)
        self.generate_message2 = GenerateMessage(inplanes=32)
        self.generate_message1 = GenerateMessage(inplanes=32)

        #-------------refine--------
        self.Calibrate1 = G_Fuse(32, 8)
        self.Calibrate2 = G_Fuse(32, 8)
        self.Calibrate3 = G_Fuse(32, 8)
        self.Calibrate4 = G_Fuse(32, 8)


        self.cbr3_4 = BasicConv2d(64, 32, 3, padding=1, flag=1)
        self.cbr2_4 = BasicConv2d(64, 32, 3, padding=1, flag=1)
        self.cbr1_4 = BasicConv2d(64, 32, 3, padding=1, flag=1)

        self.out_2 = BasicConv2d(32, 32, 3, padding=1, flag=1)
        self.out_3 = nn.Conv2d(32, 1, 1)


    def forward(self, x):
        # ----------------------------------  backbone  --------------------------------------
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        x0_1 = layer0  # 64 x 176 x 176
        x1_1 = layer1  # 256 x 88 x 88
        x2_1 = layer2  # 512 x 44 x 44
        x3_1 = layer3  # 1024 x 22 x 22
        x4_1 = layer4  # 2048 x 11 x 11

        # ------------------------------stage1: base network-----------------------------------
        # backbone 特征降维
        x1_1 = self.cbr1(x1_1)  # 128 x 88 x 88
        x2_1 = self.cbr2(x2_1)  # 128 x 44 x 44
        x3_1 = self.cbr3(x3_1)  # 256 x 22 x 22
        x4_1 = self.cbr4(x4_1)  # 512 x 11 x 11

        # backbone 降维
        x0_1 = self.cbr0_1(x0_1)  # 64 x 176 x 176
        x1_1 = self.cbr1_1(x1_1)  # 64 x 88 x 88
        x2_1 = self.cbr2_1(x2_1)  # 64 x 44 x 44
        x3_1 = self.cbr3_1(x3_1)  # 64 x 22 x 22
        x4_1 = self.cbr4_1(x4_1)  # 64 x 11 x 11

        # Bitrans
        f0_after, f1_after1 = self.BiStage0(x0_1, x1_1)
        f1_after2, f2_after1 = self.BiStage1(x1_1, x2_1)
        f2_after2, f3_after1 = self.BiStage2(x2_1, x3_1)
        f3_after2, f4_after = self.BiStage3(x3_1, x4_1)

        # Bitrans: outputs
        f_out01 = torch.cat((f0_after, f1_after1), 1)  # 128 x 176 x 176
        f_out12 = torch.cat((f1_after2, f2_after1), 1)  # 128 x 88 x 88
        f_out23 = torch.cat((f2_after2, f3_after1), 1)  # 128 x 44 x 44
        f_out34 = torch.cat((f3_after2, f4_after), 1)  # 128 x 22 x 22

        # 降维
        f_out01 = self.cbr1_2(f_out01)  # 32 x 176 x 176
        f_out12 = self.cbr2_2(f_out12)  # 32 x 88 x 88
        f_out23 = self.cbr3_2(f_out23)  # 32 x 44 x 44
        f_out34 = self.cbr4_2(f_out34)  # 32 x 22 x 22

        f_out01_a, f_out01_b = self.emb_1(f_out01)
        f_out12_a, f_out12_b = self.emb_2(f_out12)
        f_out23_a, f_out23_b = self.emb_3(f_out23)
        f_out34_a, f_out34_b = self.emb_4(f_out34)

        # -----------------------------------Stage2: Dual Decoder-------------------
        # -----------------------------------Policy Decoder and 矫正decoder-------------------

        # ------feature level 4-----
        f_out34_cbr = self.cbr4_3(f_out34_a)
        coarse_pre_out4, err_map4, coarse_pre_out4_att, err_map4_att, err_map4_ext = self.generate_message4(f_out34_cbr)
        updated_fea_04 = self.Calibrate4(f_out34_b, coarse_pre_out4_att, err_map4_ext)  # 32 22 22 updated_fea

        # ------feature level 3-----
        f_out23_cbr = self.cbr3_3(torch.cat((f_out23_a, F.interpolate(f_out34_cbr, scale_factor=2, mode='bilinear')), 1))
        coarse_pre_out3, err_map3, coarse_pre_out3_att, err_map3_att, err_map3_ext = self.generate_message3(f_out23_cbr)
        f_out23_cal = self.cbr3_4(torch.cat((f_out23_b, F.interpolate(updated_fea_04, scale_factor=2, mode='bilinear')), dim=1)) # 32 x 44 x 44
        updated_fea_03 = self.Calibrate3(f_out23_cal, coarse_pre_out3_att, err_map3_ext)  # 32 44 44 updated_fea

        # ------feature level 2-----
        f_out12_cbr = self.cbr2_3(torch.cat((f_out12_a, F.interpolate(f_out23_cbr, scale_factor=2, mode='bilinear')), 1))
        coarse_pre_out2, err_map2, coarse_pre_out2_att, err_map2_att, err_map2_ext = self.generate_message2(f_out12_cbr)
        f_out12_cal = self.cbr2_4(torch.cat((f_out12_b, F.interpolate(updated_fea_03, scale_factor=2, mode='bilinear')), dim=1)) # 32 x 44 x 44
        updated_fea_02 = self.Calibrate2(f_out12_cal, coarse_pre_out2_att, err_map2_ext)  # 32 88 88 updated_fea

        # ------feature level 1-----
        f_out01_cbr = self.cbr1_3(torch.cat((f_out01_a, F.interpolate(f_out12_cbr, scale_factor=2, mode='bilinear')), 1))
        coarse_pre_out1, err_map1, coarse_pre_out1_att, err_map1_att, err_map1_ext = self.generate_message1(f_out01_cbr)
        f_out01_cal = self.cbr1_4(torch.cat((f_out01_b, F.interpolate(updated_fea_02, scale_factor=2, mode='bilinear')), dim=1)) # 32 x 44 x 44
        updated_fea_01 = self.Calibrate1(f_out01_cal, coarse_pre_out1_att, err_map1_ext)  # 32 176 176 updated_fea

        # refine output
        out = updated_fea_01  # 32 176 176
        fine_pre = self.out_3(self.out_2(out))  # 1 x 176 x 176

        # training return
        return self.upsample_2(fine_pre), \
               self.upsample_2(coarse_pre_out1), self.upsample_4(coarse_pre_out2), self.upsample_8(coarse_pre_out3), self.upsample_16(coarse_pre_out4), \
               self.upsample_2(err_map1), self.upsample_4(err_map2), self.upsample_8(err_map3), self.upsample_16(err_map4)

        # testing return
        # return self.upsample_2(fine_pre), \
        #        self.upsample_2(coarse_pre_out1), self.upsample_4(coarse_pre_out2), self.upsample_8(coarse_pre_out3), self.upsample_16(coarse_pre_out4), \
        #        self.upsample_2(err_map1), self.upsample_4(err_map2), self.upsample_8(err_map3), self.upsample_16(err_map4), self.upsample_2(err_map1_ext)



