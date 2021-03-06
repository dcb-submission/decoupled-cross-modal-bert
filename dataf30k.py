"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """
    def __init__(self, data_path, data_split):
        loc = data_path + '/'
        # Captions
        self.captions = []
        self.max_seq_length = 44
        self.max_vision = 36
        self.imgfold = "data/f30k_precomp/flickr30k_%s_frcnnnew/"%data_split

        with open('data/f30k_precomp/flickr30k_%s_caps.txt.bt'%data_split, 'r') as f:
            for line in f:
                arr = line.strip().split()
                arr = [int(astr) for astr in arr]
                self.captions.append(arr)
        #print(a)
        # Image features
        self.images = open("data/f30k_precomp/flickr30k_%s_names.txt"%data_split,"r").readlines()#np.load(loc+'%s_64featc32.npy' % data_split)
        self.length = len(self.captions)
        self.im_div = self.length//len(self.images)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev' or data_split == 'test':
            self.length = 5000
        #print(a)
    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        imgname = self.images[img_id]
        imgname = imgname.split('.')[0]+'.npy'
        imgname = self.imgfold + imgname
        image = torch.Tensor(np.load(imgname)).type(torch.float16)
        caption = self.captions[index].copy()
        input_mask = [1] * len(caption)
        vision_mask = [1] * image.size(0)  
        while len(caption) < self.max_seq_length:
             caption.append(0)
             input_mask.append(0)
        if image.size(0) < self.max_vision:
             addlen = self.max_vision-image.size(0)
             vision_mask = vision_mask + [0]*addlen
             imageadd = torch.zeros(addlen,2048).type(torch.float16)
             image = torch.cat([image,imageadd],0)#.type(torch.float16)

        if len(caption) > self.max_seq_length:
             caption = caption[:self.max_seq_length]
             input_mask = input_mask[:self.max_seq_length]


        if image.size(0) > self.max_vision:
             image = image[:self.max_vision]
             vision_mask = vision_mask[:self.max_vision]


        target = torch.Tensor(caption)
        target_mask = torch.Tensor(input_mask)
        vision_mask = torch.Tensor(vision_mask)

        return image, target, target_mask, vision_mask, index#image, target, #index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    images, captions, cap_mask, vision_mask, ids = zip(*data)

    images = torch.stack(images, 0)
    targets = torch.stack(captions, 0).long()
    cap_mask = torch.stack(cap_mask,0).long()
    vision_mask = torch.stack(vision_mask,0).long()

    return images, targets, cap_mask, vision_mask, ids


def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,num_workers=num_workers)
    return data_loader

def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train',  opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test',  opt,
                                    64, False, workers)
    return train_loader, val_loader


def get_test_loader(data_name,  batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, 'test',  opt,
                                    64, False, workers)
    return test_loader
