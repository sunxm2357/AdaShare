import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from models.adashare import MTL2
from models.cw_resnet import AdaShare, Group, Group2
from models.cw_resnet_instance import Instance_Group
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
        if self.opt['is_sparse']:
            self.reg_w = self.opt['train'].get('reg_w_start', self.opt['train']['reg_w'])

        if self.opt['is_deterministic']:
            self.reg_w_d = self.opt['train'].get('reg_w_d_start', self.opt['train']['reg_w_d'])

        # import pdb
        # pdb.set_trace()

    # ##################### define networks / optimizers / losses ####################################
    def define_loss(self):
        super(BlockDropEnv, self).define_loss()
        self.cross_entropy2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.cross_entropy_sparisty = nn.CrossEntropyLoss(ignore_index=255)
        self.softmax = nn.Softmax(dim=1)

    def define_networks(self, tasks_num_class):
        # construct a deeplab resnet 101
        init_method = self.opt['train']['init_method']
        if self.opt['policy_model'] == 'blockwise':
            from models.base import Bottleneck, BasicBlock
        else:
            raise ValueError('policy model %s is not implemented' % self.opt['policy_model'])

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

        if self.opt['policy_model'] == 'blockwise':
            self.networks['mtl-net'] = MTL2(block, layers, tasks_num_class, init_method, self.init_neg_logits,
                                            self.opt['skip_layer'], task_bn=self.opt['task_bn'])
        else:
            raise ValueError('policy model %s is not implemented' % self.opt['policy_model'])

    def get_fc_params(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            self.fc_params = self.networks['mtl-net'].module.fc_params()
        else:
            self.fc_params = self.networks['mtl-net'].fc_params()

    def get_arch_params(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            self.arch_params = self.networks['mtl-net'].module.arch_parameters()
        else:
            self.arch_params = self.networks['mtl-net'].arch_parameters()

    def get_backbone_params(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            self.backbone_params = self.networks['mtl-net'].module.backbone_params()
        else:
            self.backbone_params = self.networks['mtl-net'].backbone_params()

    def get_network_params(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            self.network_params = self.networks['mtl-net'].module.network_params()
        else:
            self.network_params = self.networks['mtl-net'].network_params()

    def define_optimizer(self, policy_learning=False):
        if self.opt['train']['imagenet_pretrain']:
            self.get_backbone_params()
            self.get_fc_params()
            self.optimizers['fc_weights_op'] = optim.Adam(self.fc_params, lr=self.opt['train']['fc_lr'],
                                                          betas=(0.5, 0.999),
                                                          weight_decay=0.0001)
            self.optimizers['backbone_weights_op'] = optim.Adam(self.backbone_params,
                                                                lr=self.opt['train']['backbone_lr'], betas=(0.5, 0.999),
                                                                weight_decay=0.0001)
        else:
            self.get_network_params()
            self.optimizers['weights_op'] = optim.Adam(self.network_params, lr=self.opt['train']['lr'], betas=(0.5, 0.999),
                                                                weight_decay=0.0001)
        self.get_arch_params()
        self.optimizers['arch_weights_op'] = optim.Adam(self.arch_params, lr=self.opt['train']['policy_lr'], weight_decay=5*1e-4)

    def define_scheduler(self, policy_learning=False):
        if 'decay_fc_lr_freq' in self.opt['train'].keys() and 'decay_fc_lr_rate' in self.opt['train'].keys():
            print('define the scheduler for fc layers')
            self.schedulers['fc_weights_sc'] = scheduler.StepLR(self.optimizers['fc_weights_op'],
                                                                step_size=self.opt['train']['decay_fc_lr_freq'],
                                                                gamma=self.opt['train']['decay_fc_lr_rate'])

        if 'decay_backbone_lr_freq' in self.opt['train'].keys() and 'decay_backbone_lr_rate' in self.opt['train'].keys():
            print('define the scheduler for backbone layers')
            self.schedulers['backbone_weights_sc'] = scheduler.StepLR(self.optimizers['backbone_weights_op'],
                                                                      step_size=self.opt['train'][
                                                                          'decay_backbone_lr_freq'],
                                                                      gamma=self.opt['train']['decay_backbone_lr_rate'])

        if 'decay_arch_lr_freq' in self.opt['train'].keys() and 'decay_arch_lr_rate' in self.opt['train'].keys():
            self.schedulers['arch_weights_sc'] = scheduler.StepLR(self.optimizers['arch_weights_op'],
                                                                  step_size=self.opt['train']['decay_arch_lr_freq'],
                                                                  gamma=self.opt['train']['decay_arch_lr_rate'])

    # ##################### train / test ####################################
    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def temperature_scheduler(self, current_iter):
        if self.opt['activation'] == 'sigmoid':
            if current_iter < self.opt['train']['temp_start']:
                self.temp = self.opt['train']['temp_start']
            elif current_iter > self.opt['train']['temp_end']:
                self.temp = self.opt['train']['temp_end']
            else:
                self.temp = (self.opt['train']['temp_end'] - self.opt['train']['temp_start']) / \
                               (self.opt['train']['temp_end_iter'] - self.opt['train']['temp_start_iter']) * \
                               (current_iter - self.opt['train']['temp_start_iter']) + \
                               self.opt['train']['temp_start']


    def sample_policy(self, hard_sampling):
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        for t_id, p in enumerate(policys):
            setattr(self, 'policy%d' % (t_id+1), p)

    def optimize(self, is_policy=False, flag='update_w', num_train_layers=None, hard_sampling=False):
        # print('num_train_layers in optimize = ', num_train_layers)
        self.forward(is_policy, num_train_layers, hard_sampling)

        self.get_loss()

        if flag == 'update_w':
            self.backward_network()
        elif flag == 'update_alpha':
            self.backward_policy(num_train_layers)
        elif flag == 'update_all':
            self.backward_all(num_train_layers)
        else:
            raise NotImplementedError('Training flag %s is not implemented' % flag)

    def optimize_fix_policy(self, num_train_layer=None):
        self.forward_fix_policy(num_train_layer)
        self.get_loss()
        self.backward_network()

    def val(self, policy=None, fix_policy=False, num_train_layers=None, hard_sampling=False):
        metrics = {}
        if fix_policy:
            self.forward_fix_policy(num_train_layers)
        elif policy:
            self.forward_eval(num_train_layers, hard_sampling)
        else:
            self.forward(policy, num_train_layers, hard_sampling)

        if self.opt['dataload']['dataset'] == 'DomainNet':
            for domain in self.opt['dataload']['domains']:
                metrics[domain] = {}
                output = getattr(self, '%s_output' % domain)
                label = getattr(self, '%s_cls' % domain)
                metrics[domain]['pred'], metrics[domain]['gt'], metrics[domain]['acc'], \
                metrics[domain]['err'] = self.get_classification_err(output, label)
        else:
            raise ValueError('Dataset %s is not supported' % self.dataset)
        return metrics

    def forward(self, is_policy, num_train_layers, hard_sampling):
        # print('in forward, is_policy = ', is_policy)
        # print('num_train_layers in forward = ', num_train_layers)
        inputs = []
        if self.opt['dataload']['dataset'] == 'DomainNet':
            for domain in self.opt['dataload']['domains']:
                if torch.cuda.is_available():
                    img = getattr(self, '%s_img' % domain)
                    inputs.append(img)
        else:
            raise ValueError('Dataset %s is not supported' % self.opt['dataload']['dataset'])

        outputs, policys = self.networks['mtl-net'](inputs, self.temp, is_policy, num_train_layers=num_train_layers,
                                                    hard_sampling=hard_sampling, mode='train')

        for t_id,  task in enumerate(self.tasks):
            setattr(self, '%s_output' % task, outputs[t_id])
            setattr(self, 'policy%d' % (t_id+1), policys[t_id])

    def forward_eval(self, num_train_layers, hard_sampling):
        inputs = []
        if self.opt['dataload']['dataset'] == 'DomainNet':
            for domain in self.opt['dataload']['domains']:
                if torch.cuda.is_available():
                    img = getattr(self, '%s_img' % domain)
                    inputs.append(img)
        else:
            raise ValueError('Dataset %s is not supported' % self.opt['dataload']['dataset'])

        outputs, _ = self.networks['mtl-net'](inputs, self.temp, True, num_train_layers=num_train_layers,
                                              hard_sampling=hard_sampling,  mode='eval')

        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_output' % task, outputs[t_id])

    def forward_fix_policy(self, num_train_layers):
        inputs = []
        if self.opt['dataload']['dataset'] == 'DomainNet':
            for domain in self.opt['dataload']['domains']:
                if torch.cuda.is_available():
                    img = getattr(self, '%s_img' % domain)
                    inputs.append(img)
        else:
            raise ValueError('Dataset %s is not supported' % self.opt['dataload']['dataset'])

        outputs, _ = self.networks['mtl-net'](inputs, self.temp, True, num_train_layers=num_train_layers,
                                              mode='fix_policy')
        for t_id, task in enumerate(self.tasks):
            setattr(self, '%s_output' % task, outputs[t_id])

    def get_sparsity_loss2(self, num_train_layers):
        self.losses['sparsity'] = {}
        self.losses['sparsity']['total'] = 0
        num_policy_layers = None
        for t_id in range(self.num_tasks):
            if self.opt['policy_model'] in ['blockwise', 'channelwise']:
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1))
                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1))
            elif self.opt['policy_model'] in ['group', 'group2', 'instance_group']:
                logits = getattr(self, 'policy%d' % (t_id + 1))
            else:
                raise ValueError('policy model (%s) is not supported' % self.opt['policy_model'])

            if num_policy_layers is None:
                num_policy_layers = logits.shape[0]
            else:
                assert (num_policy_layers == logits.shape[0])

            if num_train_layers is None:
                num_train_layers = num_policy_layers

            num_blocks = min(num_train_layers, logits.shape[0])
            if self.opt['policy_model'] == 'blockwise':
                gt = torch.ones((num_blocks)).long().to(self.device)
                if self.opt['diff_sparsity_weights'] and not self.opt['is_sharing']:
                    loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).to(self.device)
                    self.losses['sparsity']['task%d' % (t_id + 1)] = 2 * (
                                loss_weights[-num_blocks] * self.cross_entropy2(logits[-num_blocks:], gt)).mean()
                else:
                    self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy_sparisty(logits[-num_blocks:], gt)

            elif self.opt['policy_model'] == 'channelwise':
                channels = self.networks['mtl-net'].channels
                num_train_channels = sum(channels[-num_blocks:])
                gt = torch.ones(num_train_channels).long().to(self.device)

                self.losses['sparsity']['task%d' % (t_id + 1)] = self.cross_entropy(logits[-num_train_channels:], gt)
            elif self.opt['policy_model'] in ['group', 'group2']:
                self.losses['sparsity']['task%d' % (t_id + 1)] = torch.abs(logits[-num_train_layers:]).sum() / num_train_layers
            elif self.opt['policy_model'] in ['instance_group', 'instance_group2']:
                self.losses['sparsity']['task%d' % (t_id + 1)] = torch.abs(logits).mean()
            else:
                raise ValueError('Policy Mode %s is not supported' % self.opt['policy_model'])

            self.losses['sparsity']['total'] += self.losses['sparsity']['task%d' % (t_id + 1)]

    def get_hamming_loss(self):
        self.losses['hamming'] = {}
        self.losses['hamming']['total'] = 0
        num_policy_layers = None
        for t_i in range(self.num_tasks):
            if self.opt['policy_model'] in ['blockwise', 'channelwise']:
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits_i = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_i + 1))
                else:
                    logits_i = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_i + 1))
            elif self.opt['policy_model'] == 'group':
                logits_i = getattr(self, 'policy%d' % (t_i + 1))
            else:
                raise ValueError('policy model (%s) is not supported' % self.opt['policy_model'])

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
                if self.opt['policy_model'] in ['blockwise', 'channelwise']:
                    if isinstance(self.networks['mtl-net'], nn.DataParallel):
                        logits_j = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_j + 1))
                    else:
                        logits_j = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_j + 1))
                elif self.opt['policy_model'] == 'group':
                    logits_j = getattr(self, 'policy%d' % (t_j + 1))
                else:
                    raise ValueError('policy model (%s) is not supported' % self.opt['policy_model'])

                if num_policy_layers is None:
                    num_policy_layers = logits_j.shape[0]
                else:
                    assert (num_policy_layers == logits_j.shape[0])
                # print('loss weights = ', loss_weights)
                # print('hamming = ', torch.sum(torch.abs(logits_i[:, 0] - logits_j[:, 0])))
                if self.opt['policy_model'] in ['blockwise', 'channelwise']:
                    self.losses['hamming']['total'] += torch.sum(loss_weights * torch.abs(logits_i[:, 0] - logits_j[:, 0]))
                elif self.opt['policy_model'] == 'group':
                    self.losses['hamming']['total'] += torch.sum(torch.abs(logits_i - logits_j))
                else:
                    raise ValueError('policy model (%s) is not supported' % self.opt['policy_model'])

    def get_deterministic_loss(self, num_train_layers):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            channels = self.networks['mtl-net'].module.channels
        else:
            channels = self.networks['mtl-net'].channels

        num_policy_layers = len(channels)
        if num_train_layers is None:
            num_train_layers = num_policy_layers

        num_blocks = min(num_train_layers, num_policy_layers)


        self.losses['deterministic'] = {}
        self.losses['deterministic']['total'] = 0

        if self.opt['policy_model'] == 'channelwise':
            num_train_channels = sum(channels[-num_blocks:])

            for t_id in range(self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1))
                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1))
                policy = self.softmax(logits[-num_train_channels:])
                self.losses['deterministic']['task%d' % (t_id + 1)] = -torch.mean(policy * torch.log(1e-3 + policy))
                self.losses['deterministic']['total'] += self.losses['deterministic']['task%d' % (t_id + 1)]
        elif self.opt['policy_model'] in ['group', 'group2']:
            num_train_channels = num_blocks * 8
            mean_policy = None
            for t_id in range(self.num_tasks):
                policy = getattr(self, 'policy%d' % (t_id + 1)).clone().detach()[-num_train_channels:]
                if mean_policy is None:
                    mean_policy = policy
                else:
                    mean_policy += policy
            mean_policy /= self.num_tasks

            cuda_device = getattr(self, 'policy1').get_device()
            compared_policy = torch.stack([0.5 * torch.ones((num_train_channels), device=cuda_device),
                                           mean_policy], dim=1).min(dim=-1)[0]

            for t_id in range(self.num_tasks):
                policy = getattr(self, 'policy%d' % (t_id + 1))[-num_train_channels:]

                self.losses['deterministic']['task%d' % (t_id + 1)] = -torch.abs(policy - compared_policy).mean()
                self.losses['deterministic']['total'] += self.losses['deterministic']['task%d' % (t_id + 1)]

        elif self.opt['policy_model'] in ['instance_group', 'instance_group2']:

            mean_policy = None
            for t_id in range(self.num_tasks):
                policy = getattr(self, 'policy%d' % (t_id + 1)).clone().detach()
                if mean_policy is None:
                    mean_policy = policy
                else:
                    mean_policy += policy
            mean_policy /= self.num_tasks
            mean_policy = mean_policy.mean(dim=0)

            cuda_device = getattr(self, 'policy1').get_device()
            num_train_layers = getattr(self, 'policy1').shape[1]
            compared_policy = torch.stack([0.5 * torch.ones((num_train_layers), device=cuda_device),
                                           mean_policy], dim=1).min(dim=-1)[0]


            for t_id in range(self.num_tasks):
                policy = getattr(self, 'policy%d' % (t_id + 1))
                batch_size = policy.shape[0]
                compare = compared_policy.view(1, -1).repeat(batch_size, 1)
                self.losses['deterministic']['task%d' % (t_id + 1)] = -torch.abs(policy - compare).mean()
                self.losses['deterministic']['total'] += self.losses['deterministic']['task%d' % (t_id + 1)]

    def backward_policy(self, num_train_layers):
        self.optimizers['arch_weights_op'].zero_grad()

        loss = 0
        for t_id, task in enumerate(self.tasks):
            loss += self.opt['lambdas'][t_id] * self.losses[task]['total']

        if self.opt['is_sharing']:
            self.get_hamming_loss()
            loss += self.opt['train']['reg_w_hamming'] * self.losses['hamming']['total']
        if self.opt['is_sparse']:
            self.get_sparsity_loss2(num_train_layers)
            loss += self.reg_w * self.losses['sparsity']['total']
        if self.opt['is_deterministic'] and self.reg_w_d != 0:
            self.get_deterministic_loss(num_train_layers)
            loss += self.reg_w_d * self.losses['deterministic']['total']

        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        self.optimizers['arch_weights_op'].step()
        if 'arch_weights_sc' in self.schedulers.keys():
            self.schedulers['arch_weights_sc'].step()

    def backward_network(self):
        if self.opt['train']['imagenet_pretrain']:
            self.optimizers['fc_weights_op'].zero_grad()
            self.optimizers['backbone_weights_op'].zero_grad()
        else:
            self.optimizers['weights_op'].zero_grad()
        loss = 0
        for t_id, task in enumerate(self.tasks):
            loss += self.opt['lambdas'][t_id] * self.losses[task]['total']
        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        if self.opt['train']['imagenet_pretrain']:
            self.optimizers['fc_weights_op'].step()
            self.optimizers['backbone_weights_op'].step()

            if 'fc_weights_sc' in self.schedulers.keys():
                self.schedulers['fc_weights_sc'].step()
            if 'backbone_weights_sc' in self.schedulers.keys():
                self.schedulers['backbone_weights_sc'].step()
        else:
            self.optimizers['weights_op'].step()

            if 'weights_sc' in self.schedulers.keys():
                self.schedulers['weights_sc'].step()


    def backward_all(self, num_train_layers):
        if self.opt['train']['imagenet_pretrain']:
            self.optimizers['fc_weights_op'].zero_grad()
            self.optimizers['backbone_weights_op'].zero_grad()
        else:
            self.optimizers['weights_op'].zero_grad()
        self.optimizers['arch_weights_op'].zero_grad()

        loss = 0
        for t_id, task in enumerate(self.tasks):
            loss += self.opt['lambdas'][t_id] * self.losses[task]['total']

        if self.opt['is_sharing']:
            self.get_hamming_loss()
            loss += self.opt['train']['reg_w_hamming'] * self.losses['hamming']['total']
        if self.opt['is_sparse']:
            self.get_sparsity_loss2(num_train_layers)
            loss += self.reg_w * self.losses['sparsity']['total']
        if self.opt['is_deterministic'] and self.reg_w_d != 0:
            self.get_deterministic_loss(num_train_layers)
            loss += self.reg_w_d * self.losses['deterministic']['total']

        self.losses['total'] = {}
        self.losses['total']['total'] = loss
        self.losses['total']['total'].backward()
        self.optimizers['arch_weights_op'].step()
        if 'arch_weights_sc' in self.schedulers.keys():
            self.schedulers['arch_weights_sc'].step()

        if self.opt['train']['imagenet_pretrain']:
            self.optimizers['fc_weights_op'].step()
            self.optimizers['backbone_weights_op'].step()

            if 'fc_weights_sc' in self.schedulers.keys():
                self.schedulers['fc_weights_sc'].step()
            if 'backbone_weights_sc' in self.schedulers.keys():
                self.schedulers['backbone_weights_sc'].step()
        else:
            self.optimizers['weights_op'].step()

            if 'weights_sc' in self.schedulers.keys():
                self.schedulers['weights_sc'].step()

    def get_policy_prob(self):
        distributions = []
        if self.opt['policy_model'] in ['channelwise', 'blockwise']:
            for t_id in range(self.num_tasks):
                if isinstance(self.networks['mtl-net'], nn.DataParallel):
                    logits = getattr(self.networks['mtl-net'].module, 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()

                else:
                    logits = getattr(self.networks['mtl-net'], 'task%d_logits' % (t_id + 1)).detach().cpu().numpy()
                distributions.append(softmax(logits, axis=-1))
        elif self.opt['policy_model'] in ['group', 'group2']:
            # logits = getattr(self, 'policy%d' % (t_id+1)).detach().view(-1, 1).cpu().numpy()
            _, activations = self.networks['mtl-net'].generate_policys()
            for a in activations:
                distributions.append(a.detach().view(-1, 1).cpu().numpy())
        elif self.opt['policy_model'] in ['instance_group', 'instance_group2']:
            for t_id in range(self.num_tasks):
                logits = getattr(self, 'policy%d' % (t_id+1)).detach().mean(dim=0).view(-1, 1).cpu().numpy()
                distributions.append(logits)
        else:
            raise ValueError('Policy Model (%s) is not supported' % self.opt['policy_model'] )

        return distributions

    def get_current_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            policy = getattr(self, 'policy%d' % (t_id + 1))
            policy = policy.detach().cpu().numpy()
            policys.append(policy)

        return policys

    def weight_step(self, loss_name, method, current_iter):
        if loss_name == 'deterministic':
            if method == 'slope':
                tmp = (current_iter - self.opt['train']['reg_w_d_start_iter'])/ \
                               (self.opt['train']['reg_w_d_end_iter'] - self.opt['train']['reg_w_d_start_iter'])
                tmp = min(1, max(0, tmp))
                self.reg_w_d = (self.opt['train']['reg_w_d_end'] - self.opt['train']['reg_w_d_start']) * tmp + \
                               self.opt['train']['reg_w_d_start']
            elif method == 'step':
                self.reg_w_d = self.opt['train']['reg_w_d_start'] if current_iter < self.opt['train']['reg_w_d_step'] \
                    else self.opt['train']['reg_w_d_end']
            else:
                raise ValueError('In loss %s, method = %s is not supported' % (loss_name, method))
        elif loss_name == 'sparsity':
            if method == 'slope':
                tmp = (current_iter - self.opt['train']['reg_w_start_iter']) / \
                               (self.opt['train']['reg_w_end_iter'] - self.opt['train']['reg_w_start_iter'])
                tmp = min(1, max(0, tmp))
                self.reg_w = (self.opt['train']['reg_w_end'] - self.opt['train']['reg_w_start']) * tmp + \
                               self.opt['train']['reg_w_start']
            elif method == 'step':
                self.reg_w_d = self.opt['train']['reg_w_start'] if current_iter < self.opt['train']['reg_w_step'] \
                    else self.opt['train']['reg_w_end']
            else:
                raise ValueError('In loss %s, method = %s is not supported' % (loss_name, method))
        else:
            raise ValueError('loss %s is not supported' % loss_name)

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
        self.get_network_params()
        for param in self.network_params:
            param.requires_grad = False
        # if isinstance(self.networks['mtl-net'], nn.DataParallel):
        #     for param in self.networks['mtl-net'].module.backbone.parameters():
        #         param.requires_grad = False
        #
        # else:
        #     for param in self.networks['mtl-net'].backbone.parameters():
        #         param.requires_grad = False
        #
        # self.get_fc_params()
        # for param in self.fc_params:
        #     param.requires_grad = False

    def free_w(self, fix_BN):
        self.get_network_params()
        for param in self.network_params:
            param.requires_grad = True
        # if isinstance(self.networks['mtl-net'], nn.DataParallel):
        #     for name, param in self.networks['mtl-net'].module.backbone.named_parameters():
        #         param.requires_grad = True
        #
        #         if fix_BN and 'bn' in name:
        #             param.requires_grad = False
        # else:
        #     for name, param in self.networks['mtl-net'].backbone.named_parameters():
        #         param.requires_grad = True
        #         if fix_BN and 'bn' in name:
        #             param.requires_grad = False
        #
        # self.get_fc_params()
        # for param in self.fc_params:
        #     param.requires_grad = True

    def fix_alpha(self):
        self.get_arch_params()
        for param in self.arch_params:
            param.requires_grad = False

    def free_alpha(self):
        self.get_arch_params()
        for param in self.arch_params:
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

    def load_imagenet_pretrain(self):
        if isinstance(self.networks['mtl-net'], nn.DataParallel):
            self.networks['mtl-net'].module.load_imagenet_pretrain()
        else:
            self.networks['mtl-net'].load_imagenet_pretrain()

    def name(self):
        return 'BlockDropEnv'