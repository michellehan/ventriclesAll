import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights
import models as models

model_meta = {
        "resnet18":[8,6], "resnet34":[8,6], "resnet50":[8,6], "resnet101":[8,6], "resnet152":[8,6],
        #vgg16:[0,22], vgg19:[0,22],
        #resnext50:[8,6], resnext101:[8,6], resnext101_64:[8,6],
        #wrn:[8,6], inceptionresnet_2:[-2,9], inception_4:[-1,9],
        #dn121:[0,7], dn161:[0,7], dn169:[0,7], dn201:[0,7],
}


class ConvBReLU(nn.Module):
    def __init__(self, ins, outs, kernel_size, padding, stride, dropout=False):
        super(ConvBReLU, self).__init__()

        self.conv = nn.Conv2d(ins, outs, kernel_size = kernel_size, padding = padding, stride = stride)
        self.batch = nn.BatchNorm2d(outs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
#        print(x)
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, ins, outs):
        super(EncoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            ConvBReLU(ins, outs, kernel_size = (3, 3), stride = 1, padding = 1),
            ConvBReLU(outs, outs, kernel_size = (3, 3), stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size(2, 2), stride = 2),
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, ins, mids, outs):
        super(DecoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            ConvBReLU(ins, mids, kernel_size = (3, 3), stride = 1, padding = 1),
            ConvBReLU(mids, outs, kernel_size = (3, 3), stride = 1, padding = 1),
            ##nn.MaxPool2d(2,2)
        )

    def forward(self, x):
        return self.block(x)



class UNet(nn.Module):
    #def __init__(self, num_classes):
    def __init__(self, num_classes, num_filters = 32, encoder = False):
        super(UNet, self).__init__()
        
        #enc_ch = [512, 256, 128, 64] #resnet18
        enc_ch = [2048, 1024, 512, 256] #renset50
        channels = [num_filters * 8, num_filters * 4, num_filters * 2, num_filters]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace = True)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

        #self.encoder = models.__dict__[encoder](pretrained = True, num_classes = num_classes)
        self.encoder = models.__dict__["resnet50"](pretrained = True, num_classes = num_classes)
        
        self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        #self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.enc1 = self.encoder.layer1 
        self.enc2 = self.encoder.layer2 
        self.enc3 = self.encoder.layer3 
        self.enc4 = self.encoder.layer4 

        #self.center = DecoderBlock(enc_ch[0], channels[0] * 2, num_filters * 8)
        self.center = DecoderBlock(enc_ch[0], channels[0] * 2, channels[0])

        self.dec4 = DecoderBlock(enc_ch[0] + channels[0], channels[0] * 2, channels[1])
        self.dec3 = DecoderBlock(enc_ch[1] + channels[1], channels[1] * 2, channels[2])
        self.dec2 = DecoderBlock(enc_ch[2] + channels[2], channels[2] * 2, channels[3])
        self.dec1 = DecoderBlock(enc_ch[3] + channels[3], channels[3] * 2, channels[3])
        self.dec0 = nn.Sequential(
		nn.Conv2d(channels[3], channels[3], kernel_size = (3,3), padding = 1),
		nn.ReLU(inplace=True)
	)
        
        #self.dec5 = DecoderBlock(enc_ch[0] + channels[0], channels[0] * 2, channels[1])
        #self.dec4 = DecoderBlock(enc_ch[1] + channels[1], channels[1] * 2, channels[2])
        #self.dec3 = DecoderBlock(enc_ch[2] + channels[2], channels[2] * 2, channels[3])
        #self.dec2 = DecoderBlock(enc_ch[3] + channels[3], channels[3] * 2, channels[3] * 2)
        #self.dec1 = DecoderBlock(channels[3] * 2, channels[3] * 2, channels[3])
        #self.dec0 = nn.Sequential(
	#	nn.Conv2d(channels[3], channels[3], kernel_size = (3,3)),
	#	nn.ReLU(inplace=True)
	#)
        
        self.final = nn.Conv2d(num_filters, 1, kernel_size = 1)


    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        #enc4 = self.upsample(enc4)
        #enc4 = self.relu(self.pool(enc4))
        #center = self.center(self.pool(enc4))
        #center = self.center(self.relu(self.pool(enc4)))
        #center = self.center(self.relu(enc4))
        #print('enc0', enc0.size())
        #print('enc1', enc1.size())
        #print('enc2', enc2.size())
        #print('enc3', enc3.size())
        #print('enc4', enc4.size())

        #center = self.pool(self.center(enc4))

        ####Upsampling added to match 256x256 resolution of input image
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, self.upsample(enc4)], 1))
        dec3 = self.dec3(torch.cat([dec4, self.upsample(enc3)], 1))
        dec2 = self.dec2(torch.cat([dec3, self.upsample(enc2)], 1))
        dec1 = self.dec1(torch.cat([dec2, self.upsample(enc1)], 1))
        dec0 = self.dec0(dec1)

        #center = self.pool(self.center(enc4))
        #print('\ncenter', center.size())
        #print('enc4', enc4.size())
        #dec4 = self.dec4(torch.cat([center, enc4], 1))
        #print('\ndec4', dec4.size())
        #print('enc3', enc3.size())
        #dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        #print('\ndec3', dec3.size())
        #print('enc2', enc2.size())
        #dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        #print('\ndec2', dec2.size())
        #print('enc1', enc1.size())
        #dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        #print('\ndec1', dec1.size())
        #print('enc0', enc0.size())
        #dec0 = self.dec0(torch.cat([self.pool(dec1), enc0], 1))
        #dec0 = self.dec0(dec1)

        #print('\ndec0', dec0.size())
        final = self.final(dec0)

        #print('final', final.size())
        return final

