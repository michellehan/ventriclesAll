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
 
        ######
        if encoder:
            model = models.__dict__[encoder](pretrained = True, num_classes = num_classes)
        
            #cut, cut_lr = model_meta[encoder]
            #self.encoder = (list(model.children())[:cut] if cut else [model])
            #print('self.encoder', self.encoder)
            #self.enc0 = nn.Sequential(*self.encoder)

            print(model.conv1)
            self.encoder = model
            self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
 
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024, dropout = True)
        self.center = Decoder(512, 1024, 512)
        self.dec4 = Decoder(2048, 1024, 512)
        self.dec3 = Decoder(1024, 512, 256)
        self.dec2 = Decoder(512, 256, 128)
        self.dec1 = Decoder(256, 128, 64)


        if encoder:
            self.dec0 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        ######################
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        ####
        #enc1 = self.enc1(x)
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)

        #####
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        #####
        dec0 = self.dec0(torch.cat([dec1, F.upsample(enc0, dec1.size()[2:], mode='bilinear')], 1))
        #final = self.final(dec1)
        final = self.final(dec0)
        #####

        return F.upsample(final, x.size()[2:], mode='bilinear')
