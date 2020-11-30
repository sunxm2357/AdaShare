import os
from PIL import Image
import torch
from torchvision import transforms
import pdb


class DomainNet(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, h, w, overlap=None, copy=None):
        print(self.name())
        self.dataroot = dataroot
        if mode in ['train', 'train1', 'train2']:
            name = 'train.txt'
            if overlap is not None and copy is not None:
                name = 'train_overlap%.2f_copy%d.txt' % (overlap, copy)
            with open(os.path.join(dataroot, name), 'r') as f:
                self.img_list = f.readlines()
            self.transform = self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((h, w), scale=(0.6, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif mode in ['val', 'test']:
            name = 'test.txt'
            if overlap is not None and copy is not None:
                name = 'test_overlap%.2f_copy%d.txt' % (overlap, copy)
            with open(os.path.join(dataroot, name), 'r') as f:
                self.img_list = f.readlines()
            self.transform = self.transform = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError('Mode %s is not supported' % mode)
        self.mode = mode

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        line = self.img_list[item]
        tokens = line.strip().split()
        img_path = tokens[0]
        img_idx = int(tokens[1])
        img = Image.open(os.path.join(self.dataroot, img_path))
        img = img.convert('RGB')
        img_t = self.transform(img)

        return {'img': img_t, 'img_path': img_path, "img_idx": img_idx}

    def name(self):
        return 'DomainNet DATALOADER'
