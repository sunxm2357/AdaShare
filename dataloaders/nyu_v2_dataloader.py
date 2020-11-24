import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from torchvision import transforms


class NYU_v2(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None, opt=None):
        print(self.name())
        json_file = os.path.join(dataroot, 'nyu_v2_3task.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.triples = info[mode]
        self.opt = opt
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 480
            self.crop_w = 640
        self.mode = mode
        self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))

    def __len__(self):
        return len(self.triples)
        # return 6

    @staticmethod
    def __scale__(img, label1, label2, label3):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        label1 = np.expand_dims(cv2.resize(label1, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label2 = cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        label3 = np.expand_dims(cv2.resize(label3, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)

        return img_new, label1, label2, label3

    @staticmethod
    def __mirror__(img, label1, label2, label3):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            label1 = label1[:, ::-1]
            label2 = label2[:, ::-1]
            label3 = label3[:, ::-1]
        return img, label1, label2, label3

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img, label1, label2, label3, crop_h, crop_w, ignore_label=255):
        # combining
        label = np.concatenate((label1, label2), axis=2).astype('float32')
        label -= ignore_label
        combined = np.concatenate((img, label, label3), axis=2)
        image_shape = img.shape
        label3_shape = label3.shape
        # padding to the crop size
        pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # separating
        img_cdim = image_shape[-1]
        label1_cdim = label1.shape[-1]
        label3_cdim = label3_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        label3_crop = deepcopy(combined_crop[:, :, -label3_cdim:])
        label_crop = combined_crop[:, :, img_cdim: -label3_cdim]
        label_crop = (label_crop + ignore_label).astype('uint8')
        label1_crop = label_crop[:, :, :label1_cdim]
        label2_crop = label_crop[:, :, label1_cdim:]
        return img_crop, label1_crop, label2_crop, label3_crop

    def __getitem__(self, item):
        img_path, seg_path, normal_path, depth_path = self.triples[item]
        img = cv2.imread(os.path.join(self.dataroot, img_path))
        seg = np.expand_dims(cv2.imread(os.path.join(self.dataroot, seg_path), cv2.IMREAD_GRAYSCALE), axis=-1)
        normal = cv2.imread(os.path.join(self.dataroot, normal_path))
        depth = np.expand_dims(np.load(os.path.join(self.dataroot, depth_path)), axis=-1)
        if self.mode in ['train', 'train1', 'train2']:
            img, seg, normal, depth = self.__scale__(img, seg, normal, depth)
            img, seg, normal, depth = self.__mirror__(img, seg, normal, depth)
            img, seg, normal, depth = self.__random_crop_and_pad_image_and_labels__(img, seg, normal, depth, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN
        name = img_path.split('/')[-1]
        img_id = name.split('.')[0]
        # loading the instance-level policy
        batch = {'img': self.transform(img).float(), 'seg': torch.from_numpy(seg).permute(2, 0, 1),
                 'normal': torch.from_numpy(normal).permute(2, 0, 1), 'depth': torch.from_numpy(depth).permute(2, 0, 1),
                 'name': name}

        if self.opt is not None:
            policy_dir = os.path.join(self.opt['paths']['result_dir'], self.opt['exp_name'], 'policy')
            for t_id, task in enumerate(self.opt['tasks']):
                task_policy_dir = os.path.join(policy_dir, task)
                policy_path = os.path.join(task_policy_dir, img_id + '.npy')
                policy = np.load(policy_path)
                batch['%s_policy' % task] = torch.from_numpy(policy)

        return batch

    def name(self):
        return 'NYU_v2'

class NYU_v2_Customize(NYU_v2):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None, opt=None):
        if not os.path.exists(os.path.join(dataroot, 'nyu_v2_3task.json')):
            json_file = {'test': []}
            # img_dir = os.path.join(dataroot, 'img')
            # seg_dir = os.path.join(dataroot, 'seg')
            # depth_dir = os.path.join(dataroot, 'depth')
            # sn_dir = os.path.join(dataroot, 'normal_mask')
            img_dir = 'img'
            seg_dir = 'seg'
            depth_dir = 'depth'
            sn_dir = 'normal_mask'
            for f_name in os.listdir(os.path.join(dataroot, img_dir)):
                if f_name.endswith('png'):
                    f_id = f_name.split('.')[0]
                    img_path = os.path.join(img_dir, f_name)
                    seg_path = os.path.join(seg_dir, f_name)
                    sn_path = os.path.join(sn_dir, f_name)
                    depth_path = os.path.join(depth_dir, "%s.npy" % f_id)
                    if os.path.exists(os.path.join(dataroot, img_path)) \
                            and os.path.exists(os.path.join(dataroot, seg_path)) \
                            and os.path.exists(os.path.join(dataroot, sn_path)) and \
                            os.path.exists(os.path.join(dataroot, depth_path)):
                        json_file['test'].append((img_path, seg_path, sn_path, depth_path))
            with open(os.path.join(dataroot, 'nyu_v2_3task.json'), 'w+') as f:
                json.dump(json_file, f)
        super(NYU_v2_Customize, self).__init__(dataroot, mode, crop_h, crop_w, opt)

    def name(self):
         print('NYU_v2_Customize')



class NYU_v2_Single_IMG(NYU_v2):
    def __init__(self, dataroot, mode, img_id, crop_h=None, crop_w=None, policy_dir=None):
        super(NYU_v2_Single_IMG, self).__init__(dataroot, mode, crop_h, crop_w, policy_dir)
        assert (img_id < len(self.triples))
        self.img_id = img_id

    def __len__(self):
        return 1

    def __del__(self):
        print("deleted")

    def __getitem__(self, item):
        img_path, seg_path, normal_path, depth_path = self.triples[self.img_id]
        img = cv2.imread(os.path.join(self.dataroot, img_path))
        seg = np.expand_dims(cv2.imread(os.path.join(self.dataroot, seg_path), cv2.IMREAD_GRAYSCALE), axis=-1)
        normal = cv2.imread(os.path.join(self.dataroot, normal_path))
        depth = np.expand_dims(np.load(os.path.join(self.dataroot, depth_path)), axis=-1)
        if self.mode in ['train', 'train1', 'train2']:
            img, seg, normal, depth = self.__scale__(img, seg, normal, depth)
            img, seg, normal, depth = self.__mirror__(img, seg, normal, depth)
            img, seg, normal, depth = self.__random_crop_and_pad_image_and_labels__(img, seg, normal, depth, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN
        name = img_path.split('/')[-1]
        img_id = name.split('.')[0]
        if self.policy_dir is not None:
            name1 = os.path.join(self.policy_dir, 'task1', '%s.npy' % img_id)
            name2 = os.path.join(self.policy_dir, 'task2', '%s.npy' % img_id)
            policy1 = np.load(name1)
            policy2 = np.load(name2)
            return {'img': self.transform(img).float(), 'seg': torch.from_numpy(seg).permute(2, 0, 1),
                    'normal': torch.from_numpy(normal).permute(2, 0, 1), 'depth': torch.from_numpy(depth).permute(2, 0, 1),
                    'name': name, 'policy1': torch.from_numpy(policy1), 'policy2': torch.from_numpy(policy2)}

        return {'img': self.transform(img).float(), 'seg': torch.from_numpy(seg).permute(2, 0, 1),
                'depth': torch.from_numpy(depth).permute(2, 0, 1), 'normal': torch.from_numpy(normal).permute(2, 0, 1),
                'name': name}

    def name(self):
        return 'NYU_v2_Single_IMG'