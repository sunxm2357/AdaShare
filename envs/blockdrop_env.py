import os
import torch
from torch import nn
import torch.optim as optim
from models.deeplab_resnet import MTL2, MTL_Instance
from models.base import Bottleneck, BasicBlock
from envs.base_env import BaseEnv
from scipy.special import softmax
import pickle
import torch.optim.lr_scheduler as scheduler


class BlockDropEnv(BaseEnv):
    """
    The environment to train a simple classification model
    """

    def __init__(self, log_dir, checkpoint_dir, exp_name, tasks_num_class, init_neg_logits=-10, device=0, init_temperature=5.0, temperature_decay=0.965,
                 is_train=True, opt=None):
        """
        :param num_class: int, the number of classes in the dataset
        :param log_dir: str, the path to save logs
        :param checkpoint_dir: str, the path to save checkpoints
        :param lr: float, the learning rate
        :param is_train: bool, specify during the training
        """
        self.init_neg_logits = init_neg_logits
        self.temp = init_temperature
        self._tem_decay = temperature_decay
        self.num_tasks = len(tasks_num_class)
        super(BlockDropEnv, self).__init__(log_dir, checkpoint_dir, exp_name, tasks_num_class, device,
                                           is_train, opt)

    # ##################### define networks / optimizers / losses ####################################

    def define_networks(self, tasks_num_class):
        # construct a deeplab resnet 101
        if self.opt['backbone'].startswith('ResNet'):
            init_method = self.opt['train']['init_method']
            if self.opt['backbone'] == 'ResNet101':
                block = Bottleneck
                layers = [3, 4, 23, 3]
            elif self.opt['backbone'] == 'ResNet18':
                block = BasicBlock
                layers = [2, 2, 2, 2]
            elif self.opt['backbone'] == 'ResNet34':
                block = BasicBlock
                layers = [3, 4, 6, 3]
            else:
                raise NotImplementedError('backbone %s is not implemented' % self.opt['backbone'])

            if self.opt['policy_model'] == 'task-specific':
                self.networks['mtl-net'] = MTL2(block, layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])
            elif self.opt['policy_model'] == 'instance-specific':
                self.networks['mtl-net'] = MTL_Instance(block, layers, tasks_num_class, init_method, self.init_neg_logits, self.opt['skip_layer'])

            else:
                raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])

        else:
            raise NotImplementedError('backbone %s is not implemented' % self.opt['backbone'])

    def get_task_specific_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            task_specific_params = self.networks['mtl-net'].module.task_specific_parameters()
        else:
            task_specific_params = self.networks['mtl-net'].task_specific_parameters()

        return task_specific_params

    def get_arch_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            arch_parameters = self.networks['mtl-net'].module.arch_parameters()
        else:
            arch_parameters = self.networks['mtl-net'].arch_parameters()

        return arch_parameters

    def get_network_parameters(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            network_parameters = self.networks['mtl-net'].module.network_parameters()
        else:
            network_parameters = self.networks['mtl-net'].network_parameters()
        return network_parameters

    def get_backbone_parameters(self):
        if self.opt['backbone'] == 'WRN':
            network_parameters = []
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            else:
                for name, param in self.networks['mtl-net'].named_parameters():
                    if name.startswith('task') and ('logits' in name or 'fc' in name):
                        continue
                    else:
                        network_parameters.append(param)
            return network_parameters
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                backbone_parameters = self.networks['mtl-net'].module.backbone_parameters()
            else:
                backbone_parameters = self.networks['mtl-net'].backbone_parameters()
            return backbone_parameters

    def define_optimizer(self, policy_learning=False):
        task_specific_params = self.get_task_specific_parameters()
        arch_parameters = self.get_arch_parameters()
        backbone_parameters = self.get_backbone_parameters()
        # TODO: add policy learning to yaml

        if policy_learning:
            self.optimizers['weights'] = optim.SGD([{'params': task_specific_params, 'lr': self.opt['train']['lr']},
                                                    {'params': backbone_parameters, 'lr': self.opt['train']['backbone_lr']}],
                                                   momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizers['weights'] = optim.Adam([{'params': task_specific_params, 'lr': self.opt['train']['lr']},
                                                     {'params': backbone_parameters,'lr': self.opt['train']['backbone_lr']}],
                                                    betas=(0.5, 0.999), weight_decay=0.0001)

        if self.opt['train']['init_method'] == 'all_chosen':
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)
        else:
            self.optimizers['alphas'] = optim.Adam(arch_parameters, lr=0.01, weight_decay=5*1e-4)

    def define_scheduler(self, policy_learning=False):
        if policy_learning:
            if 'policy_decay_lr_freq' in self.opt['train'].keys() and 'policy_decay_lr_rate' in self.opt['train'].keys():
                print('define the scheduler (policy learning)')
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['policy_decay_lr_freq'],
                                                              gamma=self.opt['train']['policy_decay_lr_rate'])

        else:
            if 'decay_lr_freq' in self.opt['train'].keys() and 'decay_lr_rate' in self.opt['train'].keys():
                print('define the scheduler (not policy learning)')
                self.schedulers['weights'] = scheduler.StepLR(self.optimizers['weights'],
                                                              step_size=self.opt['train']['decay_lr_freq'],
                                                              gamma=self.opt['train']['decay_lr_rate'])

    # ##################### train / test ####################################
    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def sample_policy(self, hard_sampling):
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        for t_id, p in enumerate(policys):
            setattr(self, 'policy%d' % (t_id+1), p)

    def optimize(self, lambdas, is_policy=False, flag='update_w', num_train_layers=None, hard_sampling=False):
        # print('num_train_layers in optimize = ', num_train_layers)
        self.forward(is_policy, num_train_layers, hard_sampling)
        if 'seg' in self.tasks:
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.get_seg_loss(seg_num_class)
        if 'sn' in self.tasks:
            self.get_sn_loss()
        if 'depth' in self.tasks:
            self.get_depth_loss()
        if 'keypoint' in self.tasks:
            self.get_keypoint_loss()
        if 'edge' in self.tasks:
            self.get_edge_loss()
        if flag == 'update_w':
            self.backward_network(lambdas)
        elif flag == 'update_alpha':
            self.backward_policy(lambdas, num_train_layers)
        else:
            raise NotImplementedError('Training flag %s is not implemented' % flag)

    def optimize_fix_policy(self, lambdas, num_train_layer=None):
        self.forward_fix_policy(num_train_layer)
        if 'seg' in self.tasks:
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.get_seg_loss(seg_num_class)
        if 'sn' in self.tasks:
            self.get_sn_loss()
        if 'depth' in self.tasks:
            self.get_depth_loss()
        if 'keypoint' in self.tasks:
            self.get_keypoint_loss()
        if 'edge' in self.tasks:
            self.get_edge_loss()
        self.backward_network(lambdas)

    def val(self, policy, num_train_layers=None, hard_sampling=False):
        metrics = {}
        self.forward(is_policy=policy, num_train_layers=num_train_layers, hard_sampling=hard_sampling)
        self.resize_results()
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
        if 'depth' in self.tasks:
            metrics['depth'] = {}
            abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
            metrics['depth']['abs_err'] = abs_err
            metrics['depth']['rel_err'] = rel_err
            metrics['depth']['sq_rel_err'] = sq_rel_err
            metrics['depth']['ratio'] = ratio
            metrics['depth']['rms'] = rms
            metrics['depth']['rms_log'] = rms_log
        if 'keypoint' in self.tasks:
            metrics['keypoint'] = {}
            err = self.keypoint_error()
            metrics['keypoint']['err'] = err
        if 'edge' in self.tasks:
            metrics['edge'] = {}
            err = self.edge_error()
            metrics['edge']['err'] = err
        return metrics

    def val2(self, policy, num_train_layers=None, hard_sampling=False):
        metrics = {}
        if policy:
            self.forward_eval(num_train_layers, hard_sampling)
        else:
            self.forward(policy, num_train_layers, hard_sampling)
        self.resize_results()
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
        if 'depth' in self.tasks:
            metrics['depth'] = {}
            abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
            metrics['depth']['abs_err'] = abs_err
            metrics['depth']['rel_err'] = rel_err
            metrics['depth']['sq_rel_err'] = sq_rel_err
            metrics['depth']['ratio'] = ratio
            metrics['depth']['rms'] = rms
            metrics['depth']['rms_log'] = rms_log
        if 'keypoint' in self.tasks:
            metrics['keypoint'] = {}
            err = self.keypoint_error()
            metrics['keypoint']['err'] = err
        if 'edge' in self.tasks:
            metrics['edge'] = {}
            err = self.edge_error()
            metrics['edge']['err'] = err
        return metrics

    def val_fix_policy(self, num_train_layers=None):
        metrics = {}
        self.forward_fix_policy(num_train_layers)
        self.resize_results()
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            pred, gt, pixelAcc, err = self.seg_error(seg_num_class)
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
        if 'depth' in self.tasks:
            metrics['depth'] = {}
            abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
            metrics['depth']['abs_err'] = abs_err
            metrics['depth']['rel_err'] = rel_err
            metrics['depth']['sq_rel_err'] = sq_rel_err
            metrics['depth']['ratio'] = ratio
            metrics['depth']['rms'] = rms
            metrics['depth']['rms_log'] = rms_log
        if 'keypoint' in self.tasks:
            metrics['keypoint'] = {}
            err = self.keypoint_error()
            metrics['keypoint']['err'] = err
        if 'edge' in self.tasks:
            metrics['edge'] = {}
            err = self.edge_error()
            metrics['edge']['err'] = err
        return metrics

    def forward(self, is_policy, num_train_layers, hard_sampling):
        # print('in forward, is_policy = ', is_policy)
        # print('num_train_layers in forward = ', num_train_layers)

        outputs, policys, logits = self.networks['mtl-net'](self.img, self.temp, is_policy, num_train_layers=num_train_layers,
                                                    hard_sampling=hard_sampling, mode='train')
        # import pdb
        # pdb.set_trace()
        for t_id,  task in enumerate(self.tasks):
            setattr(self, '%s_pred' % task, outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1), policys[t_id])
            setattr(self, 'logit%d' % (t_id+1), logits[t_id])

    def forward_eval(self, num_train_layers, hard_sampling):
        outputs,policys, logits = self.networks['mtl-net'](self.img, self.temp, True, num_train_layers=num_train_layers,
                                              hard_sampling=hard_sampling,  mode='eval')

        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_pred' % task, outputs[t_id])
            setattr(self, 'policy%d' % (t_id + 1), policys[t_id])
            setattr(self, 'logit%d' % (t_id + 1), logits[t_id])

    def forward_fix_policy(self, num_train_layers):
        if self.opt['policy_model'] == 'instance-specific':
            policys = []
            for task in self.opt['tasks']:
                policys.append(getattr(self, '%s_policy' % task))

            outputs, _, _ = self.networks['mtl-net'](self.img, self.temp, True, policys=policys, num_train_layers=num_train_layers,
                                                  mode='fix_policy')
        elif self.opt['policy_model'] == 'task-specific':
            outputs, _, _ = self.networks['mtl-net'](self.img, self.temp, True,  num_train_layers=num_train_layers,
                                                     mode='fix_policy')
        else:
            raise ValueError('policy model = %s is not supported' % self.opt['policy_model'])

        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_pred' % task, outputs[t_id])

    def get_sparsity_loss2(self, num_train_layers):
        self.losses['sparsity'] = {}
        self.losses['sparsity']['total'] = 0
        num_policy_layers = None
        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1))
                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1))
                if num_policy_layers is None:
                    num_policy_layers = logits.shape[0]
                else:
                    assert (num_policy_layers == logits.shape[0])

                if num_train_layers is None:
                    num_train_layers = num_policy_layers

                num_blocks = min(num_train_layers, logits.shape[0])
                gt = torch.ones((num_blocks)).long().to(self.device)

                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    self.losses['sparsity']['task%d' % (t_id + 1)] = 2 * (
                                loss_weights[-num_blocks:] * self.cross_entropy2(logits[-num_blocks:], gt)).mean()
                else:
                    self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy_sparsity(logits[-num_blocks:], gt)

                self.losses['sparsity']['total'] += self.losses['sparsity']['task%d' % (t_id + 1)]

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                logit = getattr(self, 'policy%d' % (t_id+1))
                if num_policy_layers is None:
                    num_policy_layers = logit.shape[1]
                else:
                    assert (num_policy_layers == logit.shape[1])

                if num_train_layers is None:
                    num_train_layers = num_policy_layers

                num_blocks = min(num_train_layers, logit.shape[1])
                batch_size = logit.shape[0]
                gt = torch.ones((batch_size * num_blocks)).long().to(self.device)
                logit = logit[:, -num_blocks:].contiguous().view(-1, 2)
                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(
                        self.device)
                    loss_weights = loss_weights.view(1, -1).repeat(batch_size, 1).view(-1)

                    self.losses['sparsity']['task%d' % (t_id + 1)] = 2 * (
                            loss_weights[-batch_size * num_blocks:] * self.cross_entropy2(logit, gt)).mean()
                else:
                    self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy_sparsity(logit, gt)

                self.losses['sparsity']['total'] += self.losses['sparsity']['task%d' % (t_id + 1)]
        else:
            raise ValueError('Policy Model = %s is not supported' % self.opt['policy_model'])

    def get_hamming_loss(self):
        self.losses['hamming'] = {}
        self.losses['hamming']['total'] = 0
        num_policy_layers = None
        for t_i in range(self.num_tasks):
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                logits_i = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_i + 1))
            else:
                logits_i = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_i + 1))
            if num_policy_layers is None:
                num_policy_layers = logits_i.shape[0]
                if self.opt['diff_sparsity_weights']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(
                        self.device)
                else:
                    loss_weights = (torch.ones((num_policy_layers)).float()).to(self.device)
            else:
                assert (num_policy_layers == logits_i.shape[0])
            for t_j in range(t_i, self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits_j = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_j + 1))
                else:
                    logits_j = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_j + 1))

                if num_policy_layers is None:
                    num_policy_layers = logits_j.shape[0]
                else:
                    assert (num_policy_layers == logits_j.shape[0])
                # print('loss weights = ', loss_weights)
                # print('hamming = ', torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0])))
                self.losses['hamming']['total'] += torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))

    def backward_policy(self, lambdas, num_train_layers):
        self.optimizers['alphas'].zero_grad()

        loss = 0
        for t_id, task in enumerate(self.tasks):
            loss += lambdas[t_id] * self.losses[task]['total']

        if self.opt['is_sharing']:
            self.get_hamming_loss()
            loss += self.opt['train']['reg_w_hamming'] * self.losses['hamming']['total']
        if self.opt['is_sparse']:
            # self.get_sparsity_loss()
            self.get_sparsity_loss2(num_train_layers)
            loss += self.opt['train']['reg_w'] * self.losses['sparsity']['total']

        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        self.optimizers['alphas'].step()

    def backward_network(self, lambdas):
        self.optimizers['weights'].zero_grad()
        loss = 0
        for t_id, task in enumerate(self.tasks):
            loss += lambdas[t_id] * self.losses[task]['total']
        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        self.optimizers['weights'].step()
        if 'weights' in self.schedulers.keys():
            self.schedulers['weights'].step()

    def get_policy_prob(self):
        distributions = []
        if self.opt['policy_model'] == 'task-specific':
            for t_id in range(self.num_tasks):

                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()

                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()
            distributions.append(softmax(logits, axis=-1))

        elif self.opt['policy_model'] == 'instance-specific':
            for t_id in range(self.num_tasks):
                logit = getattr(self, 'logit%d' % (t_id+1)).detach().cpu().numpy()
                distributions.append(logit.mean(axis=0))
        else:
            raise ValueError('policy mode = %s is not supported' % self.opt['policy_model']  )
        return distributions

    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys

    # ##################### change the state of each module ####################################
    def get_current_state(self, current_iter):
        current_state = super(BlockDropEnv, self).get_current_state(current_iter)
        current_state['temp'] = self.temp
        return current_state

    def save_policy(self, label):
        policy = {}
        for t_id in range(self.num_tasks):
            tmp = getattr(self, 'policy%d' % (t_id + 1))
            policy['task%d_policy' % (t_id + 1)] = tmp.cpu().data
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)
        for t_id in range(self.num_tasks):
            setattr(self, 'policy%d' % (t_id + 1), policy['task%d_policy' % (t_id+1)])
            print(getattr(self, 'policy%d' % (t_id + 1)))

    def check_exist_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        return os.path.exists(save_path)

    def load_snapshot(self, snapshot):
        super(BlockDropEnv, self).load_snapshot(snapshot)
        self.temp = snapshot['temp']
        return snapshot['iter']


    def fix_w(self):
        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = False
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for param in self.networks['mtl-net'].module.backbone.parameters():
                    param.requires_grad = False

            else:
                for param in self.networks['mtl-net'].backbone.parameters():
                    param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()
            for param in task_specific_parameters:
                param.requires_grad = False

    def free_w(self, fix_BN):
        if self.opt['backbone'] == 'WRN':
            network_params = self.get_network_parameters()
            for param in network_params:
                param.requires_grad = True
        else:
            if isinstance(self.networks['mtl-net'], nn.DataParallel):
                for name, param in self.networks['mtl-net'].module.backbone.named_parameters():
                    param.requires_grad = True

                    if fix_BN and 'bn' in name:
                        param.requires_grad = False
            else:
                for name, param in self.networks['mtl-net'].backbone.named_parameters():
                    param.requires_grad = True
                    if fix_BN and 'bn' in name:
                        param.requires_grad = False

            task_specific_parameters = self.get_task_specific_parameters()
            for param in task_specific_parameters:
                param.requires_grad = True

    def fix_alpha(self):
        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = False

    def free_alpha(self):
        arch_parameters = self.get_arch_parameters()
        for param in arch_parameters:
            param.requires_grad = True

    # ##################### change the state of each module ####################################
    def cuda(self, gpu_ids):
        super(BlockDropEnv, self).cuda(gpu_ids)
        policys = []

        for t_id in range(self.num_tasks):
            if not hasattr(self, 'policy%d' % (t_id+1)):
                return
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.to(self.device)
            policys.append(policy)

        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            setattr(self.networks['mtl-net'].module, 'policys', policys)

        else:
            setattr(self.networks['mtl-net'], 'policys', policys)

    def name(self):
        return 'BlockDropEnv'
