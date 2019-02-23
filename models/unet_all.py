import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights
import models as models


class ConvBReLU(nn.Module):
    def __init__(self, ins, outs, kernel_size, padding, stride, dropout=False):
        super(ConvBReLU, self).__init__()

        self.conv = nn.Conv2d(ins, outs, kernel_size = kernel_size, padding = padding, stride = stride)
        self.batch = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)

        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


#class EncoderBlock(nn.Module):
#    def __init__(self, ins, outs):
#        super(EncoderBlock, self).__init__()
#        
#        self.block = nn.Sequential(
#            ConvBReLU(ins, outs, kernel_size = (3, 3), stride = 1, padding = 1),
#            ConvBReLU(outs, outs, kernel_size = (3, 3), stride = 1, padding = 1),
#            nn.MaxPool2d(kernel_size(2, 2), stride = 2),
#        )
#
#    def forward(self, x):
#        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, ins, mids, outs):
        super(DecoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            #nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            Interpolate(scale_factor = 2, mode = 'bilinear'),
            ConvBReLU(ins, mids, kernel_size = (3, 3), stride = 1, padding = 1),
            ConvBReLU(mids, outs, kernel_size = (3, 3), stride = 1, padding = 1),
            ###nn.MaxPool2d(2,2)
        )

    def forward(self, x):
        return self.block(x)



class UNet_resnet50(nn.Module):
    def __init__(self, num_classes, num_filters = 32, encoder = False):
        super(UNet_resnet50, self).__init__()
        
        #enc_ch = [512, 256, 128, 64] #resnet18
        enc_ch = [2048, 1024, 512, 256] #renset50
        channels = [num_filters * 8, num_filters * 4, num_filters * 2, num_filters]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace = True)
        #self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.upsample = Interpolate(scale_factor = 2, mode = 'bilinear')

        self.encoder = models.__dict__["resnet50"](pretrained = True, num_classes = num_classes)
        
        self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        self.enc1 = self.encoder.layer1 
        self.enc2 = self.encoder.layer2 
        self.enc3 = self.encoder.layer3 
        self.enc4 = self.encoder.layer4 

        self.center = DecoderBlock(enc_ch[0], channels[0] * 2, channels[0])

        self.dec4 = DecoderBlock(enc_ch[0] + channels[0], channels[0] * 2, channels[1])
        self.dec3 = DecoderBlock(enc_ch[1] + channels[1], channels[1] * 2, channels[2])
        self.dec2 = DecoderBlock(enc_ch[2] + channels[2], channels[2] * 2, channels[3])
        self.dec1 = DecoderBlock(enc_ch[3] + channels[3], channels[3] * 2, channels[3])
        self.dec0 = nn.Sequential(
		nn.Conv2d(channels[3], channels[3], kernel_size = (3,3), padding = 1),
		nn.ReLU(inplace=True)
	)
        
        self.final = nn.Conv2d(num_filters, 1, kernel_size = 1)


    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        ####Upsampling added to match 256x256 resolution of input image
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, self.upsample(enc4)], 1))
        dec3 = self.dec3(torch.cat([dec4, self.upsample(enc3)], 1))
        dec2 = self.dec2(torch.cat([dec3, self.upsample(enc2)], 1))
        dec1 = self.dec1(torch.cat([dec2, self.upsample(enc1)], 1))
        dec0 = self.dec0(dec1)

        final = self.final(dec0)
        return final




def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet_vgg11(nn.Module):
    #def __init__(self, num_classes):
    def __init__(self, num_classes, num_filters = 32, encoder = False):
        super(UNet_vgg11, self).__init__()


        self.pool = nn.MaxPool2d(2,2)
        self.encoder = models.__dict__["vgg11"](pretrained = True).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = _DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = _DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = _DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = _DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = _DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)


    def forward(self, x):

        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(dec1)





class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Encoder, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)
    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    #def __init__(self, num_classes):
    def __init__(self, num_classes, encoder = False):
        super(UNet, self).__init__()


        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024, dropout = True)
        self.center = Decoder(512, 1024, 512)
        self.dec4 = Decoder(2048, 1024, 512)
        self.dec3 = Decoder(1024, 512, 256)
        self.dec2 = Decoder(512, 256, 128)
        self.dec1 = Decoder(256, 128, 64)



        ######################
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')
