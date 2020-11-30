import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.util import print_current_errors


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
        if self.opt['dataload']['dataset'] == 'DomainNet':
            self.cross_entropy = nn.CrossEntropyLoss()
            self.cross_entropy_instance = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Dataset %s is not supported' % self.dataset)

    def define_networks(self, tasks_num_class):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    # ##################### train / test ####################################
    def set_inputs(self, batch):
        if self.opt['dataload']['dataset'] == 'DomainNet':
            for domain in self.opt['dataload']['domains']:
                if torch.cuda.is_available():
                    setattr(self, '%s_img' % domain, batch['%s_img' % domain].to(self.device))
                    setattr(self, '%s_cls' % domain, batch['%s_img_idx' % domain].to(self.device))
                else:
                    setattr(self, '%s_img' % domain, batch['%s_img' % domain])
                    setattr(self, '%s_cls' % domain, batch['%s_img_idx' % domain])
        else:
            raise ValueError('Dataset %s is not supported' % self.opt['dataload']['dataset'])

    def extract_features(self):
        pass

    def get_loss(self, instance=False, domain=None):
        if self.opt['dataload']['dataset'] == 'DomainNet':
            if instance:
                output = getattr(self, '%s_output' % domain)
                label = getattr(self, '%s_cls' % domain)
                instance_loss = self.cross_entropy_instance(output, label)
                setattr(self, '%s_instance_loss' % domain, instance_loss)
            else:
                for domain in self.opt['dataload']['domains']:
                    output = getattr(self, '%s_output' % domain)
                    label = getattr(self, '%s_cls' % domain)
                    self.losses[domain] = {}
                    self.losses[domain]['total'] = self.cross_entropy(output, label)

        else:
            raise ValueError('Dataset %s is not supported' % self.dataset)

    def get_classification_err(self, output, label):
        gt = label.view(-1)
        err = self.cross_entropy(output, gt.long())

        prediction = torch.argmax(output, dim=1)
        # pixel acc
        prediction = prediction.view(-1)
        acc = (gt == prediction).float().mean()

        return prediction.cpu().numpy(), gt.cpu().numpy(), acc.cpu().numpy(), err.cpu().numpy()

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
        for k, v in self.schedulers.items():
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
            for k, v in self.schedulers.items():
                if k in snapshot.keys():
                    self.schedulers[k].load_state_dict(snapshot[k])
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