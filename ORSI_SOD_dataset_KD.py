from torchvision import models, transforms
from torch.utils import data
from PIL import Image
import os
import numpy as np
import json
import csv
import torch

def dataset_info(dt):   
    assert dt in ['EORSSD']
    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std


def random_aug_transform(): 
    flip_h = transforms.RandomHorizontalFlip(p=1)
    flip_v = transforms.RandomVerticalFlip(p=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = transforms.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = transforms.Compose([flip_h, flip_v, rotate])
    elif r <= 0.5:
        flip_rot = transforms.Compose([flip_h, rotate])
    elif r <= 0.75:
        flip_rot = transforms.Compose([flip_v, flip_h, rotate])  
    else:
        flip_rot = transforms.Compose([flip_v, rotate])
    return flip_rot


class ORSI_SOD_Dataset(data.Dataset):
    def __init__(self, root,  mode='train', img_size = 448, aug=False):
        self.mode = mode 
        self.aug = aug 
        self.dt_mean, self.dt_std = dataset_info('EORSSD')
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        self.image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'labels', prefix + '.png') for prefix in self.prefixes]
        self.edge_paths = [os.path.join(root, 'edges', prefix + '.png') for prefix in self.prefixes]

        self.image_transformation_448x448 = transforms.Compose([transforms.Resize((448, 448),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.image_transformation_224x224 = transforms.Compose([transforms.Resize((224, 224),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.image_transformation_112x112 = transforms.Compose([transforms.Resize((112, 112),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])
        self.image_transformation_56x56 = transforms.Compose([transforms.Resize((56, 56),Image.BILINEAR),transforms.ToTensor(), transforms.Normalize(self.dt_mean, self.dt_std)])

        self.label_transformation_448x448 = transforms.Compose([transforms.Resize((448, 448),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation_224x224 = transforms.Compose([transforms.Resize((224, 224),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation_112x112 = transforms.Compose([transforms.Resize((112, 112),Image.BILINEAR),transforms.ToTensor()])
        self.label_transformation_56x56 = transforms.Compose([transforms.Resize((56, 56),Image.BILINEAR),transforms.ToTensor()])
            
    def __getitem__(self, index):
        if self.mode == "train": 
            if self.aug:
                flip_rot = random_aug_transform()
                image_448x448 = self.image_transformation_448x448(flip_rot(Image.open(self.image_paths[index]).convert('RGB'))) 
                image_224x224 = self.image_transformation_224x224(flip_rot(Image.open(self.image_paths[index]).convert('RGB'))) 
                image_112x112 = self.image_transformation_112x112(flip_rot(Image.open(self.image_paths[index]).convert('RGB')))
                image_56x56 = self.image_transformation_56x56(flip_rot(Image.open(self.image_paths[index]).convert('RGB'))) 

                label_448x448 = self.label_transformation_448x448(flip_rot(Image.open(self.label_paths[index]).convert('L')))
                label_224x224 = self.label_transformation_224x224(flip_rot(Image.open(self.label_paths[index]).convert('L')))
                label_112x112 = self.label_transformation_112x112(flip_rot(Image.open(self.label_paths[index]).convert('L')))
                label_56x56 = self.label_transformation_56x56(flip_rot(Image.open(self.label_paths[index]).convert('L')))

                edge_448x448 = self.label_transformation_448x448(flip_rot(Image.open(self.edge_paths[index])))
                edge_224x224 = self.label_transformation_224x224(flip_rot(Image.open(self.edge_paths[index])))
                edge_112x112 = self.label_transformation_112x112(flip_rot(Image.open(self.edge_paths[index])))
                edge_56x56 = self.label_transformation_56x56(flip_rot(Image.open(self.edge_paths[index])))

            else:
                image_448x448 = self.image_transformation_448x448(Image.open(self.image_paths[index]).convert('RGB'))
                image_224x224 = self.image_transformation_224x224(Image.open(self.image_paths[index]).convert('RGB'))
                image_112x112 = self.image_transformation_112x112(Image.open(self.image_paths[index]).convert('RGB'))
                image_56x56 = self.image_transformation_56x56(Image.open(self.image_paths[index]).convert('RGB'))

                label_448x448 = self.label_transformation_448x448(Image.open(self.label_paths[index]).convert('L'))
                label_224x224 = self.label_transformation_224x224(Image.open(self.label_paths[index]).convert('L'))
                label_112x112 = self.label_transformation_112x112(Image.open(self.label_paths[index]).convert('L'))
                label_56x56 = self.label_transformation_56x56(Image.open(self.label_paths[index]).convert('L'))

                edge_448x448 = self.label_transformation_448x448(Image.open(self.edge_paths[index]))
                edge_224x224 = self.label_transformation_224x224(Image.open(self.edge_paths[index]))
                edge_112x112 = self.label_transformation_112x112(Image.open(self.edge_paths[index]))
                edge_56x56 = self.label_transformation_56x56(Image.open(self.edge_paths[index]))
        
        elif self.mode == "test": 
            image_448x448 = self.image_transformation_448x448(Image.open(self.image_paths[index]).convert('RGB')) 
            image_224x224 = self.image_transformation_224x224(Image.open(self.image_paths[index]).convert('RGB')) 
            image_112x112 = self.image_transformation_112x112(Image.open(self.image_paths[index]).convert('RGB')) 
            image_56x56 = self.image_transformation_56x56(Image.open(self.image_paths[index]).convert('RGB')) 

            label_448x448 = self.label_transformation_448x448(Image.open(self.label_paths[index]).convert('L'))
            label_224x224 = self.label_transformation_224x224(Image.open(self.label_paths[index]).convert('L'))
            label_112x112 = self.label_transformation_112x112(Image.open(self.label_paths[index]).convert('L'))
            label_56x56 = self.label_transformation_56x56(Image.open(self.label_paths[index]).convert('L'))

            edge_448x448 = self.label_transformation_448x448(Image.open(self.edge_paths[index]))
            edge_224x224 = self.label_transformation_224x224(Image.open(self.edge_paths[index]))
            edge_112x112 = self.label_transformation_112x112(Image.open(self.edge_paths[index]))
            edge_56x56 = self.label_transformation_56x56(Image.open(self.edge_paths[index]))

        name = self.prefixes[index]  
        return {'img_448x448':image_448x448, 'img_224x224':image_224x224, 'img_112x112':image_112x112, 'img_56x56':image_56x56, 'label_448x448':label_448x448, 'label_224x224':label_224x224, 'label_112x112':label_112x112, 'label_56x56':label_56x56, 'edge_448x448':edge_448x448, 'edge_224x224':edge_224x224, 'edge_112x112':edge_112x112, 'edge_56x56':edge_56x56, 'name':name}
        

    def __len__(self):
        return len(self.prefixes)


