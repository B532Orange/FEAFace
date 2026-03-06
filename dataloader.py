import sys
sys.path.append("..")


from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np

class MagTrainDataset(data.Dataset):
    def __init__(self, ann_file, transform = None):
        self.ann_file = ann_file
        self.transform = transform
        
        self.init()

    def init(self):
        
        self.im_names = []
        self.targets = []
        self.ground_face = []
        with open(self.ann_file) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                if len(data) > 1:
                    self.targets.append(int(data[1]))
                    if len(data) > 2 :
                        self.ground_face.append(data[2])
                    else:
                        self.ground_face.append(0)
                else:
                    self.targets.append(-1)
                    print("No targets!")


    def __getitem__(self, index):
        if index < 0 or index >= len(self.im_names):
            print("Index out of range")
        im_name = self.im_names[index]
        target = self.targets[index]
        ground = self.ground_face[index]
        img1 = Image.open(im_name)
        img1 = self.transform(img1)

        if ground != 0 :
           ground = Image.open(ground)
           ground = self.transform(ground)
        else :
            ground = "None"
        return img1, ground, target, im_name
    def __len__(self):
        return len(self.im_names)


def img_loader(args):
    img_trans = transforms.Compose([
    transforms.ColorJitter(brightness=(0.9, 1.1)),
    transforms.CenterCrop(100),
    transforms.ToTensor()
])
    
    img_dataset = MagTrainDataset(
        args.img_list,
        transform=img_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        img_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader,len(img_dataset)

def val_loader(args):
    img_trans = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.ToTensor()
    ])
   
    img_dataset = MagTrainDataset(
        args.val_list,
        transform=img_trans

    )
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        img_dataset,
        shuffle=(val_sampler is None),
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=(val_sampler is None))

    return val_loader,len(img_dataset)