import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.util import print_current_errors
# from data_utils.image_decoder import inv_preprocess, decode_labels2


class BaseEnv():
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, device=0, is_train=True, opt=None):
        """
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        print(self.name())
        self.checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
        self.log_dir = os.path.join(log_dir, exp_name)
        self.is_train = is_train
        self.tasks_num_class = tasks_num_class
        self.device_id = device
        self.opt = opt
        self.dataset = self.opt['dataload']['dataset']
        self.tasks = self.opt['tasks']
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device)

        self.networks = {}
        self.define_networks(tasks_num_class)

        self.define_loss()
        self.losses = {}

        self.optimizers = {}
        self.schedulers = {}
        if is_train:
            # define optimizer
            self.define_optimizer()
            self.define_scheduler()
            # define summary writer
            self.writer = SummaryWriter(log_dir=self.log_dir)

    # ##################### define networks / optimizers / losses ####################################

    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        self.l1_loss2 = nn.L1Loss(reduction='none')
        if self.dataset == 'NYU_v2':
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=255)

            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        elif self.dataset == 'Taskonomy':
            dataroot = self.opt['dataload']['dataroot']
            weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).to(self.device).float()
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            self.cross_entropy_sparisty = nn.CrossEntropyLoss(ignore_index=255)
        elif self.dataset == 'CityScapes':
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
            self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            self.cross_entropy_sparsity = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise NotImplementedError('Dataset %s is not implemented' % self.dataset)

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    # ##################### train / test ####################################
    def set_inputs(self, batch):
        """
        :param batch: {'images': a tensor [batch_size, c, video_len, h, w], 'categories': np.ndarray [batch_size,]}
        """
        self.img = batch['img']
        if torch.cuda.is_available():
            self.img = self.img.to(self.device)

        if 'depth' in self.tasks:
            self.depth = batch['depth']
            if torch.cuda.is_available():
                self.depth = self.depth.to(self.device)
            if 'depth_mask' in batch.keys():
                self.depth_mask = batch['depth_mask']
                if torch.cuda.is_available():
                    self.depth_mask = self.depth_mask.to(self.device)
            if 'depth_policy' in batch.keys():
                self.depth_policy = batch['depth_policy']
                if torch.cuda.is_available():
                    self.depth_policy = self.depth_policy.to(self.device)

        if 'sn' in self.tasks:
            self.normal = batch['normal']
            if torch.cuda.is_available():
                self.normal = self.normal.to(self.device)
            if 'normal_mask' in batch.keys():
                self.sn_mask = batch['normal_mask']
                if torch.cuda.is_available():
                    self.sn_mask = self.sn_mask.to(self.device)
            if 'sn_policy' in batch.keys():
                self.sn_policy = batch['sn_policy']
                if torch.cuda.is_available():
                    self.sn_policy = self.sn_policy.to(self.device)

        if 'seg' in self.tasks:
            self.seg = batch['seg']
            if torch.cuda.is_available():
                self.seg = self.seg.to(self.device)
            if 'seg_mask' in batch.keys():
                self.seg_mask = batch['seg_mask']
                if torch.cuda.is_available():
                    self.seg_mask = self.seg_mask.to(self.device)
            if 'seg_policy' in batch.keys():
                self.seg_policy = batch['seg_policy']
                if torch.cuda.is_available():
                    self.seg_policy = self.seg_policy.to(self.device)

        if 'keypoint' in self.tasks:
            self.keypoint = batch['keypoint']
            if torch.cuda.is_available():
                self.keypoint = self.keypoint.to(self.device)
            if 'keypoint_policy' in batch.keys():
                self.keypoint_policy = batch['keypoint_policy']
                if torch.cuda.is_available():
                    self.keypoint_policy = self.keypoint_policy.to(self.device)

        if 'edge' in self.tasks:
            self.edge = batch['edge']
            if torch.cuda.is_available():
                self.edge = self.edge.to(self.device)
            if 'edge_policy' in batch.keys():
                self.edge_policy = batch['edge_policy']
                if torch.cuda.is_available():
                    self.edge_policy = self.edge_policy.to(self.device)

    def extract_features(self):
        pass

    def resize_results(self):
        new_shape = self.img.shape[-2:]
        if 'seg' in self.tasks:
            self.seg_output = F.interpolate(self.seg_pred, size=new_shape)
        if 'sn' in self.tasks:
            self.sn_output = F.interpolate(self.sn_pred, size=new_shape)
        if 'depth' in self.tasks:
            self.depth_output = F.interpolate(self.depth_pred, size=new_shape)
        if 'keypoint' in self.tasks:
            self.keypoint_output = F.interpolate(self.keypoint_pred, size=new_shape)
        if 'edge' in self.tasks:
            self.edge_output = F.interpolate(self.edge_pred, size=new_shape)

    def get_seg_loss(self, seg_num_class, instance=False):
        self.losses['seg'] = {}
        prediction = self.seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        batch_size = self.seg_pred.shape[0]
        new_shape = self.seg_pred.shape[-2:]
        seg_resize = F.interpolate(self.seg.float(), size=new_shape)
        gt = seg_resize.permute(0, 2, 3, 1).contiguous().view(-1)
        # max_label_num = prediction.argmax(dim=-1).max()
        # print(max_label_num)
        loss = self.cross_entropy(prediction, gt.long())
        self.losses['seg']['total'] = loss
        if instance:
            instance_loss = self.cross_entropy2(prediction, gt.long()).view(batch_size, -1).mean(dim=-1)
            return instance_loss
        else:
            return None

    def get_sn_loss(self, instance=False):
        self.losses['sn'] = {}
        prediction = self.sn_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        new_shape = self.sn_pred.shape[-2:]
        sn_resize = F.interpolate(self.normal.float(), size=new_shape)
        gt = sn_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (gt.max(dim=1)[0] < 255)
        if hasattr(self, 'normal_mask'):
            normal_mask_resize = F.interpolate(self.normal_mask.float(), size=new_shape)
            gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        prediction = prediction[labels]
        gt = gt[labels]

        prediction = F.normalize(prediction)
        gt = F.normalize(gt)

        self.losses['sn']['total'] = 1 - self.cosine_similiarity(prediction, gt).mean()
        if instance:
            batch_size = self.sn_pred.shape[0]
            instance_stats = labels.view(batch_size, -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = 1 - self.cosine_similiarity(prediction, gt)
            cuda_device = self.sn_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None

    def get_depth_loss(self, instance=False):
        self.losses['depth'] = {}
        new_shape = self.depth_pred.shape[-2:]
        depth_resize = F.interpolate(self.depth.float(), size=new_shape)
        if hasattr(self, 'depth_mask'):
            depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)

        if self.dataset in ['NYU_v2', 'CityScapes']:
            binary_mask = (torch.sum(depth_resize, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
        elif self.dataset == 'Taskonomy' and hasattr(self, 'depth_mask'):
            binary_mask = (depth_resize != 255) * (depth_mask_resize.int() == 1)
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        depth_output = self.depth_pred.masked_select(binary_mask)
        depth_gt = depth_resize.masked_select(binary_mask)

        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['depth']['total'] = self.l1_loss(depth_output, depth_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(depth_output, depth_gt)
            batch_size = self.depth_pred.shape[0]
            cuda_device = self.depth_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx-1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None

    def get_keypoint_loss(self, instance=False):
        self.losses['keypoint'] = {}
        new_shape = self.keypoint_pred.shape[-2:]
        keypoint_resize = F.interpolate(self.keypoint.float(), size=new_shape)
        if self.dataset == 'Taskonomy':
            binary_mask = keypoint_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        keypoint_output = self.keypoint_pred.masked_select(binary_mask)
        keypoint_gt = keypoint_resize.masked_select(binary_mask)
        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['keypoint']['total'] = self.l1_loss(keypoint_output, keypoint_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(keypoint_output, keypoint_gt)
            batch_size = self.keypoint_pred.shape[0]
            cuda_device = self.keypoint_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None

    def get_edge_loss(self, instance=False):
        self.losses['edge'] = {}
        new_shape = self.edge_pred.shape[-2:]
        edge_resize = F.interpolate(self.edge.float(), size=new_shape)
        if self.dataset == 'Taskonomy':
            binary_mask = edge_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        edge_output = self.edge_pred.masked_select(binary_mask)
        edge_gt = edge_resize.masked_select(binary_mask)
        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        self.losses['edge']['total'] = self.l1_loss(edge_output, edge_gt)
        if instance:
            instance_stats = binary_mask.view(binary_mask.shape[0], -1).long().sum(dim=-1)
            cum_stats = torch.cumsum(instance_stats, dim=0)
            loss2 = self.l1_loss2(edge_output, edge_gt)
            batch_size = self.edge_pred.shape[0]
            cuda_device = self.edge_pred.get_device()
            if cuda_device != -1:
                instance_loss = torch.zeros(batch_size).to(cuda_device)
            else:
                instance_loss = torch.zeros(batch_size).cpu()

            for b_idx in range(batch_size):
                if b_idx == 0:
                    left = 0
                    right = cum_stats[b_idx]
                else:
                    left = cum_stats[b_idx - 1]
                    right = cum_stats[b_idx]
                instance_loss[b_idx] += loss2[left: right].mean()
            return instance_loss
        else:
            return None

    def seg_error(self, seg_num_class):
        gt = self.seg.view(-1)
        labels = gt < seg_num_class
        gt = gt[labels].int()

        logits = self.seg_output.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        logits = logits[labels]
        err = self.cross_entropy(logits, gt.long())

        prediction = torch.argmax(self.seg_output, dim=1)
        prediction = prediction.unsqueeze(1)

        # pixel acc
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()

        return prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc, err.cpu().numpy()

    def normal_error(self):
        # normalized, ignored gt and prediction
        prediction = self.sn_output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = self.normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        if hasattr(self, 'normal_mask'):
            gt_mask = self.normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        gt = gt[labels]
        prediction = prediction[labels]

        gt = F.normalize(gt.float(), dim=1)
        prediction = F.normalize(prediction, dim=1)

        cos_similarity = self.cosine_similiarity(gt, prediction)

        return cos_similarity.cpu().numpy()

    def depth_error(self):
        if self.dataset in ['NYU_v2', 'CityScapes']:
            binary_mask = (torch.sum(self.depth, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
        elif self.dataset == 'Taskonomy' and hasattr(self, 'depth_mask'):
            binary_mask = (self.depth != 255) * (self.depth_mask.int() == 1)
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        depth_output_true = self.depth_output.masked_select(binary_mask)
        depth_gt_true = self.depth.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

        return abs_err.cpu().numpy(), rel_err.cpu().numpy(), sq_rel_err.cpu().numpy(), ratio[0].cpu().numpy(), \
               rms.cpu().numpy(), rms_log.cpu().numpy()

    def keypoint_error(self):
        binary_mask = (self.keypoint != 255)
        keypoint_output_true = self.keypoint_output.masked_select(binary_mask)
        keypoint_gt_true = self.keypoint.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        return abs_err.cpu().numpy()

    def edge_error(self):
        binary_mask = (self.edge != 255)
        edge_output_true = self.edge_output.masked_select(binary_mask)
        edge_gt_true = self.edge.masked_select(binary_mask)
        abs_err = torch.abs(edge_output_true - edge_gt_true).mean()
        return abs_err.cpu().numpy()

    # ##################### print loss ####################################
    def get_loss_dict(self):
        loss = {}
        for key in self.losses.keys():
            loss[key] = {}
            for subkey, v in self.losses[key].items():
                loss[key][subkey] = v.data
        return loss

    def print_loss(self, current_iter, start_time, metrics=None):
        if metrics is None:
            loss = self.get_loss_dict()
        else:
            # loss = {'metrics': metrics}
            loss = metrics

        print('-------------------------------------------------------------')
        for key in loss.keys():
            print(key + ':')
            for subkey in loss[key].keys():
                self.writer.add_scalar('%s/%s'%(key, subkey), loss[key][subkey], current_iter)
            print_current_errors(os.path.join(self.log_dir, 'loss.txt'), current_iter, loss[key],
                                 time.time() - start_time)

    # ##################### change the state of each module ####################################
    def get_current_state(self, current_iter):
        current_state = {}
        for k, v in self.networks.items():
            if isinstance(v, nn.DataParallel):
                current_state[k] = v.module.state_dict()
            else:
                current_state[k] = v.state_dict()
        for k, v in self.optimizers.items():
            current_state[k] = v.state_dict()
        current_state['iter'] = current_iter
        return current_state

    def save(self, label, current_iter):
        """
        Save the current checkpoint
        :param label: str, the label for the loading checkpoint
        :param current_iter: int, the current iteration
        """
        current_state = self.get_current_state(current_iter)
        save_filename = '%s_model.pth.tar' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(current_state, save_path)

    def load_snapshot(self, snapshot):
        for k, v in self.networks.items():
            if k in snapshot.keys():
                # loading values for the existed keys
                model_dict = v.state_dict()
                pretrained_dict = {}
                for kk, vv in snapshot[k].items():
                    if kk in model_dict.keys() and model_dict[kk].shape == vv.shape:
                        pretrained_dict[kk] = vv
                    else:
                        print('skipping %s' % kk)
                model_dict.update(pretrained_dict)
                self.networks[k].load_state_dict(model_dict)
                # self.networks[k].load_state_dict(snapshot[k])
        if self.is_train:
            for k, v in self.optimizers.items():
                if k in snapshot.keys():
                    self.optimizers[k].load_state_dict(snapshot[k])
        return snapshot['iter']

    def load(self, label, path=None):
        """
        load the checkpoint
        :param label: str, the label for the loading checkpoint
        :param path: str, specify if knowing the checkpoint path
        """
        if path is None:
            save_filename = '%s_model.pth.tar' % label
            save_path = os.path.join(self.checkpoint_dir, save_filename)
        else:
            save_path = path
        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path, map_location='cuda:%d' % self.device_id)
            return self.load_snapshot(snapshot)
        else:
            raise ValueError('snapshot %s does not exist' % save_path)

    # ##################### visualize #######################
    # def visualize(self):
    #     # TODO: implement the visualization of depth
    #     save_results = {}
    #     if 'seg' in self.tasks:
    #         num_seg_class = self.tasks_num_class[self.tasks.index('seg')]
    #         self.save_seg = decode_labels2(torch.argmax(self.seg_output, dim=1).unsqueeze(dim=1), num_seg_class, 'seg', self.seg)
    #         self.save_gt_seg = decode_labels2(self.seg, num_seg_class, 'seg', self.seg)
    #         save_results['save_seg'] = self.save_seg
    #         save_results['save_gt_seg'] = self.save_gt_seg
    #     if 'sn' in self.tasks:
    #         self.save_normal = decode_labels2(F.normalize(self.sn_output) * 255, None, 'normal', F.normalize(self.normal.float()) * 255)
    #         self.save_gt_normal = decode_labels2(F.normalize(self.normal.float()) * 255, None, 'normal', F.normalize(self.normal.float()) * 255,)
    #         save_results['save_sn'] = self.save_normal
    #         save_results['save_gt_sn'] = self.save_gt_normal
    #     if 'depth' in self.tasks:
    #         self.save_depth = decode_labels2(self.depth_output, None, 'depth', self.depth.float())
    #         self.save_gt_depth = decode_labels2(self.depth.float(), None, 'depth', self.depth.float())
    #         save_results['save_depth'] = self.save_depth
    #         save_results['save_gt_depth'] = self.save_gt_depth
    #     self.save_img = inv_preprocess(self.img)
    #     save_results['save_img'] = self.save_img
    #     return save_results

    # ##################### change the state of each module ####################################

    def train(self):
        """
        Change to the training mode
        """
        for k, v in self.networks.items():
            v.train()

    def eval(self):
        """
        Change to the eval mode
        """
        for k, v in self.networks.items():
            v.eval()

    def cuda(self, gpu_ids):
        if len(gpu_ids) == 1:
            for k, v in self.networks.items():
                v.to(self.device)
        else:
            for k, v in self.networks.items():
                self.networks[k] = nn.DataParallel(v, device_ids=gpu_ids)
                self.networks[k].to(self.device)

    def cpu(self):
        for k, v in self.networks.items():
            v.cpu()

    def name(self):
        return 'BaseEnv'