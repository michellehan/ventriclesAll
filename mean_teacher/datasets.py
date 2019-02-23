from __future__ import division

import os
import warnings
import pandas as pd
import numpy as np
import random
import nibabel

import torch
import torchvision.transforms as transforms
#import torch.utils.transforms as extended_transforms
from torch.utils.data import Dataset, DataLoader

from . import data
from .utils import export


from skimage import io
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.transform import resize

######################################################
######################################################
######################################################

@export
def RotateFlip(angle, flip): 
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=flip),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    target_transformation = transforms.Compose([
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=flip),
            transforms.Resize(256),
            transforms.ToTensor()
    ])

    return train_transformation, target_transformation


def RotateFlipFlip(angle, hflip, vflip): 
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
 
    train_transformation = transforms.Compose([
#            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    target_transformation = transforms.Compose([
#            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(angle,angle)),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
    ])

    return train_transformation, target_transformation



@export
def ventricleNormal():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

    chance = random.random()
    angles = range(-5,6) #rotate angles -5 to 5
    num_transforms = len(angles)

    #for i in range(num_transforms * 2):
    #    if i/(num_transforms * 2) <= chance < (1 + i)/(num_transforms * 2):
    #        train_transformation, target_transformation = RotateFlip( angles[i % num_transforms], i // num_transforms)
    for i in range(num_transforms * 4):
        if i/(num_transforms * 4) <= chance < (1 + i)/(num_transforms * 4):
            train_transformation, target_transformation = RotateFlipFlip( angles[i % num_transforms], i // num_transforms, (i // num_transforms) % 2)

    eval_transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_target_transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    return {
        'train_transformation': train_transformation,
        'target_transformation': target_transformation,
        'eval_transformation': eval_transformation,
        'eval_target_transformation': eval_target_transformation
    }


def loadImages(image, basedir):
    img_name = os.path.join(basedir, image)
 
    #img_name = nibabel.load(img_name).get_data()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = io.imread(img_name)
    
    if (len(image.shape)==3):
        image = image[:,:,0]
    h, w  = image.shape
    c = 3
    images = np.zeros((h, w, c), dtype = np.uint8)
    for i in range(c):
        images[:,:,i] = image
    images = Image.fromarray(images)         
    #trans = transforms.Compose([transforms.Resize(256)])
    #images = trans(images)
    return images




class Ventricles(Dataset):
    def __init__(self, csv_file, path_raw, path_segs, input_transform=None, target_transform=None, train=False):
        self.path_raw = path_raw
        self.path_segs = path_segs
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.train = train

        df = pd.read_csv(csv_file, header=None)
        #print("Dataset size: ", df.shape[0])

        samples = []
        #lower = round( len(df) / 5 )
        #upper = round( len(df) / 5 * 4 )
        for i in range(len(df)):
            name = df.iloc[i,0]
            target = df.iloc[i,1]

#            image_name = os.path.join(path_raw, name)
#            target_name = os.path.join(path_segs, target)
#            image_ni = nibabel.load(image_name).get_data()
#            target_ni = nibabel.load(target_name).get_data()


#            slices = image_ni.shape[2]
#            lower = slices / 4
#            upper = slices / 4 * 3
#            for i in range(slices):
#                name = image_ni[:,:,i]
#                target = target_ni[:,:,i]
#                item = (name, target)
#                samples.append(item)
#                
#                if train and lower < i < upper:
#                    for _ in range(3): samples.append(item)
            item = (name, target)
            samples.append(item)

            slices = df[0].str.count(name.split('slice')[0]).sum()
            lower = slices / 5
            upper = slices / 5 * 4
            index = int(name.split('slice')[1].split('.jpg')[0])
            if train and lower < index < upper:
                for _ in range(3): samples.append(item)
        self.samples = samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, target = self.samples[index]
        images = loadImages(image, self.path_raw)
        targets = loadImages(target, self.path_segs)

        tobinary = targets.convert('L')
        targets_mask = tobinary.point(lambda x: 0 if x < 100 else 1, '1')

        if self.train:
            channel_stats = dict(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

            chance = random.random()
            angle = range(-10,11) #rotate angles -5 to 5
            n_angles = len(angle)


            #for i in range(num_transforms * 2):
            #    if i/(num_transforms * 2) <= chance < (1 + i)/(num_transforms * 2):
            #        train_transformation, target_transformation = RotateFlip( angles[i % num_transforms], i // num_transforms)
            #        print('angles/flip', angles[i % num_transforms], i // num_transforms)    
            for i in range(n_angles * 4):
                if i/(n_angles * 4) <= chance < (1 + i)/(n_angles * 4):
                    input_transform, target_transform = RotateFlipFlip( angle[i % n_angles], i // n_angles, (i // n_angles) % 2)

            images = input_transform(images)
            targets_mask = target_transform(targets_mask)

        else:
            if self.input_transform:
                images = self.input_transform(images)
            if self.target_transform:
                targets_mask = self.target_transform(targets_mask)

        return (images, targets_mask)


