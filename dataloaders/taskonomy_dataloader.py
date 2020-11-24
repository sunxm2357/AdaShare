import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from PIL import Image
from torchvision import transforms
import pdb


class Taskonomy(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None):
        print(self.name())
        json_file = os.path.join(dataroot, 'taskonomy.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.groups = info[mode]
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = 256
            self.crop_w = 256
        self.mode = mode
        # self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.prior_factor = np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))

    def __len__(self):
        return len(self.groups)

    @staticmethod
    def __scale__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img_p.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img_p, (w_new, h_new))
        seg_p = np.expand_dims(cv2.resize(seg_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        seg_mask = np.expand_dims(cv2.resize(seg_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        sn_p = cv2.resize(sn_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        depth_p = np.expand_dims(cv2.resize(depth_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        depth_mask = np.expand_dims(cv2.resize(depth_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        keypoint_p = np.expand_dims(cv2.resize(keypoint_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        edge_p = np.expand_dims(cv2.resize(edge_p, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p

    @staticmethod
    def __mirror__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p):
        flag = random.random()
        if flag > 0.5:
            img_p = img_p[:, ::-1]
            seg_p = seg_p[:, ::-1]
            seg_mask = seg_mask[:, ::-1]
            sn_p = sn_p[:, ::-1]
            sn_p[:, :, 0] *= -1
            depth_p = depth_p[:, ::-1]
            depth_mask = depth_mask[:, ::-1]
            keypoint_p = keypoint_p[:, ::-1]
            edge_p = edge_p[:, ::-1]
        return img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p, crop_h, crop_w, ignore_label=255):
        # combining
        # TODO: check the ignoring labels
        label = np.concatenate((seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p), axis=2).astype('float32')
        label -= ignore_label
        # label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
        # label -= ignore_label
        combined = np.concatenate((img_p, label), axis=2)
        image_shape = img_p.shape
        c_dims = [3, 1, 1, 3, 1, 1, 1, 1]
        assert (sum(c_dims) == combined.shape[2])
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
        img_crop = deepcopy(combined_crop[:, :, 0: sum(c_dims[:1])])
        combined_crop[:, :, sum(c_dims[:1]):] += ignore_label
        seg_crop = deepcopy(combined_crop[:, :, sum(c_dims[:1]): sum(c_dims[:2])])
        seg_mask_crop = deepcopy(combined_crop[:, :, sum(c_dims[:2]): sum(c_dims[:3])])
        sn_crop = deepcopy(combined_crop[:, :, sum(c_dims[:3]): sum(c_dims[:4])])
        depth_crop = deepcopy(combined_crop[:, :, sum(c_dims[:4]): sum(c_dims[:5])])
        depth_mask_crop = deepcopy(combined_crop[:, :, sum(c_dims[:5]): sum(c_dims[:6])])
        keypoint_crop = deepcopy(combined_crop[:, :, sum(c_dims[:6]): sum(c_dims[:7])])
        edge_crop = deepcopy(combined_crop[:, :, sum(c_dims[:7]): sum(c_dims)])

        return img_crop, seg_crop, seg_mask_crop, sn_crop, depth_crop, depth_mask_crop, keypoint_crop, edge_crop

    def semantic_segment_rebalanced(self, img, new_dims=(256, 256)):
        '''
        Segmentation
        Returns:
        --------
            pixels: size num_pixels x 3 numpy array
        '''
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        mask = img > 0.1
        mask = mask.astype(float)
        img[img == 0] = 1
        img = img - 1
        rebalance = self.prior_factor[img]
        mask = mask * rebalance
        return img, mask

    @staticmethod
    def rescale_image(img, new_scale=(-1., 1.), current_scale=None, no_clip=False):
        """
        Rescales an image pixel values to target_scale

        Args:
            img: A np.float_32 array, assumed between [0,1]
            new_scale: [min,max]
            current_scale: If not supplied, it is assumed to be in:
                [0, 1]: if dtype=float
                [0, 2^16]: if dtype=uint
                [0, 255]: if dtype=ubyte
        Returns:
            rescaled_image
        """
        img = img.astype('float32')
        # min_val, max_val = img.min(), img.max()
        # img = (img - min_val)/(max_val-min_val)
        if current_scale is not None:
            min_val, max_val = current_scale
            if not no_clip:
                img = np.clip(img, min_val, max_val)
            img = img - min_val
            img /= (max_val - min_val)
        min_val, max_val = new_scale
        img *= (max_val - min_val)
        img += min_val

        return img

    def resize_rescale_image(self, img, new_scale=(-1, 1), new_dims=(256, 256), no_clip=False, current_scale=None):
        """
        Resize an image array with interpolation, and rescale to be
          between
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
        img = img.astype('float32')
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        img = self.rescale_image(img, new_scale, current_scale=current_scale, no_clip=no_clip)
        return img

    def resize_and_rescale_image_log(self, img, new_dims=(256, 256), offset=1., normalizer=np.log(2. ** 16)):
        """
            Resizes and rescales an img to log-linear

            Args:
                img: A np array
                offset: Shifts values by offset before taking log. Prevents
                    taking the log of a negative number
                normalizer: divide by the normalizing factor after taking log
            Returns:
                rescaled_image
        """
        img = np.log(float(offset) + img) / normalizer
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def mask_if_channel_ge(img, threshold, channel_idx, broadcast_to_shape=None, broadcast_to_dim=None):
        '''
            Returns a mask that masks an entire pixel iff the channel
                specified has values ge a specified value
        '''
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape
        mask = (img[:, :, channel_idx] < threshold)  # keep if lt
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis].astype(np.float32)
        if broadcast_to_shape is not None:
            return np.broadcast_to(mask, broadcast_to_shape)
        elif broadcast_to_dim is not None:
            return np.broadcast_to(mask, [h, w, broadcast_to_dim])
        else:
            return np.broadcast_to(mask, img.shape)

    def make_depth_mask(self, img, new_dims=(256, 256), broadcast_to_dim=1):
        target_mask = self.mask_if_channel_ge(img, threshold=64500, channel_idx=0, broadcast_to_dim=broadcast_to_dim)
        target_mask = cv2.resize(target_mask, new_dims, interpolation=cv2.INTER_NEAREST)
        target_mask[target_mask < 0.99] = 0.
        return target_mask

    def __getitem__(self, item):
        # TODO RGB -> BGR
        while True:
            img_path, seg_path, sn_path, depth_path, keypoint_path, edge_path = self.groups[item]
            try:
                img = np.array(Image.open(os.path.join(self.dataroot, img_path))).astype('float32')[:, :, ::-1]
                img_p = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
                seg = np.array(Image.open(os.path.join(self.dataroot, seg_path)))
                seg_p, seg_mask = self.semantic_segment_rebalanced(seg)
                sn = np.array(Image.open(os.path.join(self.dataroot, sn_path))).astype('float32') / 255
                sn_p = self.resize_rescale_image(sn)
                depth = np.array(Image.open(os.path.join(self.dataroot, depth_path))).astype('float32')
                depth_p = self.resize_and_rescale_image_log(depth)
                depth_mask = self.make_depth_mask(depth)
                keypoint = np.array(Image.open(os.path.join(self.dataroot, keypoint_path))).astype('float32') / (2 ** 16)
                keypoint_p = self.resize_rescale_image(keypoint, current_scale=(0, 0.005))
                edge = np.array(Image.open(os.path.join(self.dataroot, edge_path))).astype('float32') / (2 ** 16)
                edge_p = self.resize_rescale_image(edge, current_scale=(0, 0.08))
            except:
                print('Error in loading %s' % img_path)
                item = 0
            else:
                break

        seg_p = seg_p.astype('float32')
        seg_mask = seg_mask.astype('float32')
        sn_p = sn_p.astype('float32')
        depth_mask = depth_mask.astype('float32')

        if self.mode in ['train', 'train1', 'train2']:
            img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p = \
                self.__scale__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p)
            img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p =\
                self.__mirror__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p)
            img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p = \
                self.__random_crop_and_pad_image_and_labels__(img_p, seg_p, seg_mask, sn_p, depth_p, depth_mask, keypoint_p, edge_p, self.crop_h, self.crop_w)

        img_p = img_p.astype('float32')
        img_p = img_p - self.IMG_MEAN
        sn_mask = np.tile(depth_mask, [1, 1, 3])
        name = '-'.join(img_path.strip().split('/'))

        if seg_p.ndim == 2:
            seg_p = seg_p[:, :, np.newaxis]
        if seg_mask.ndim == 2:
            seg_mask = seg_mask[:, :, np.newaxis]
        if sn_p.ndim == 2:
            sn_p = sn_p[:, :, np.newaxis]
        if sn_mask.ndim == 2:
            sn_mask = sn_mask[:, :, np.newaxis]
        if depth_p.ndim == 2:
            depth_p = depth_p[:, :, np.newaxis]
        if depth_mask.ndim == 2:
            depth_mask = depth_mask[:, :, np.newaxis]
        if keypoint_p.ndim == 2:
            keypoint_p = keypoint_p[:, :, np.newaxis]
        if edge_p.ndim == 2:
            edge_p = edge_p[:, :, np.newaxis]

        return {'img': torch.from_numpy(img_p).permute(2, 0, 1).float(),
                'seg': torch.from_numpy(seg_p).permute(2, 0, 1).int(),
                'seg_mask': torch.from_numpy(seg_mask).permute(2, 0, 1).float(),
                'normal': torch.from_numpy(sn_p).permute(2, 0, 1).float(),
                'normal_mask': torch.from_numpy(sn_mask).permute(2, 0, 1).float(),
                'depth': torch.from_numpy(depth_p).permute(2, 0, 1).float(),
                'depth_mask': torch.from_numpy(depth_mask).permute(2, 0, 1).float(),
                'keypoint': torch.from_numpy(keypoint_p).permute(2, 0, 1).float(),
                'edge': torch.from_numpy(edge_p).permute(2, 0, 1).float(),
                'name': name}

    def name(self):
        return 'Taskonomy'
