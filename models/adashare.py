import sys
sys.path.insert(0, '..')
from models.base import *
import torch.nn.functional as F
from scipy.special import softmax
from models.util import count_params
import torch
import tqdm
import time
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers,  task_bn=False, num_tasks=-1):
        self.inplanes = 64
        super(Deeplab_ResNet_Backbone, self).__init__()
        self.block = block
        self.layers = layers
        self.task_bn = task_bn
        self.num_tasks = num_tasks
        if task_bn:
            assert (num_tasks != -1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if not task_bn:
            self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        else:
            for t_idx in range(self.num_tasks):
                bn = nn.BatchNorm2d(64, affine=affine_par)
                setattr(self, 'task%d_bn1' % t_idx, bn)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # change

        strides = [1, 2, 2, 2]
        dilations = [1, 1, 1, 1]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []
        if not task_bn:
            self.ds = []
        else:
            for t_idx in range(self.num_tasks):
                setattr(self, 'task%d_ds' % t_idx, [])

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            if task_bn:
                for t_idx in range(self.num_tasks):
                    task_ds = getattr(self, 'task%d_ds' % t_idx)
                    task_ds.append(ds[t_idx])
                    setattr(self, 'task%d_ds' % t_idx, task_ds)
            else:
                self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        if not task_bn:
            self.ds = nn.ModuleList(self.ds)
        else:
            for t_idx in range(self.num_tasks):
                task_ds = getattr(self, 'task%d_ds' % t_idx)
                setattr(self, 'task%d_ds' % t_idx, nn.ModuleList(task_ds))

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x, t_id=-1):
        x = self.conv1(x)
        if t_id != -1:
            bn1 = getattr(self, 'task%d_bn1' % t_id)
            x = bn1(x)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        if not self.task_bn:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
                conv = nn.Conv2d(self.inplanes, planes * block.expansion,
                                 kernel_size=1, stride=stride, bias=False)
                bn = nn.BatchNorm2d(planes * block.expansion)
                downsample = nn.Sequential(conv, bn)
        else:
            downsample = [None] * self.num_tasks
            if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
                conv = nn.Conv2d(self.inplanes, planes * block.expansion,
                                 kernel_size=1, stride=stride, bias=False)
                downsample = []
                for t_id in range(self.num_tasks):
                    bn = nn.BatchNorm2d(planes * block.expansion)
                    downsample.append(nn.Sequential(conv, bn))

        layers = []
        if self.task_bn:
            layers.append(block(self.inplanes, planes, stride, dilation=dilation, num_tasks=self.num_tasks))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, num_tasks=self.num_tasks))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=dilation))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    # ########################## Parameter Loading ###################################
    @staticmethod
    def key_mapping(kk):
        tokens = kk.split('.')
        new_tokens = []
        for token in tokens:
            if 'task' in token and 'fc' not in token:
                sub_tokens = token.split('_')
                new_tokens.append(sub_tokens[1])
            else:
                new_tokens.append(token)
        kk = '.'.join(new_tokens)

        tokens = kk.split('.')
        if len(tokens) == 2:
            # e.g. task0_conv1.weight => conv1.weight
            new_kk = '.'.join([tokens[0], tokens[1]])
        elif len(tokens) == 4:
            # e.g. task0_ds.1.0.weight => layer2.0.downsample.0.weight
            new_kk = 'layer%d.0.downsample.%d.%s' % (int(tokens[1]) + 1, int(tokens[2]), tokens[3])
        elif len(tokens) == 5:
            # e.g. task0_blocks.0.0.conv1.weight => layer1.0.conv1.weight
            new_kk = 'layer%d.%d.%s.%s' % (int(tokens[1]) + 1, int(tokens[2]), tokens[3], tokens[4])
        else:
            new_kk = kk

        return new_kk

    def load_imagenet_pretrain(self):
        if self.block == BasicBlock and self.layers == [2, 2, 2, 2]:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
        elif self.block == BasicBlock and self.layers == [3, 4, 6, 3]:
            state_dict = model_zoo.load_url(model_urls['resnet34'])
        elif self.block == Bottleneck and self.layers == [3, 4, 6, 3]:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        elif self.block == Bottleneck and self.layers == [3, 4, 23, 3]:
            state_dict = model_zoo.load_url(model_urls['resnet101'])
        elif self.block == Bottleneck and self.layers == [3, 8, 36, 3]:
            state_dict = model_zoo.load_url(model_urls['resnet152'])
        else:
            raise ValueError('The ResNet Architecture is not recognized')

        model_dict = self.state_dict()
        pretrained_dict = {}
        for kk, vv in model_dict.items():
            imagenet_kk = self.key_mapping(kk)
            if imagenet_kk in state_dict.keys() and state_dict[imagenet_kk].shape == vv.shape:
                pretrained_dict[kk] = state_dict[imagenet_kk]
            else:
                print('skipping %s' % kk)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('=> loaded imagenet pretrain weights')

    def forward(self, x, policy=None, t_id=-1):
        if policy is None:
            # forward through the all blocks without dropping
            x = self.seed(x, t_id)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_
                    if t_id == -1:
                        residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    else:
                        ds = getattr(self, 'task%d_ds' % t_id)
                        residual = ds[segment](x) if b == 0 and ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x, t_id=t_id))

        else:
            # do the block dropping
            x = self.seed(x, t_id)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    if t_id == -1:
                        residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    else:
                        ds = getattr(self, 'task%d_ds' % t_id)
                        residual = ds[segment](x) if b == 0 and ds[segment] is not None else x

                    fx = F.relu(residual + self.blocks[segment][b](x, t_id=t_id))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].view(-1, 1, 1, 1) + residual * policy[:, t, 1].view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1-policy[t])
                    t += 1
        return x


class MTL2(nn.Module):
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0,
                 task_bn=False):
        super(MTL2, self).__init__()
        self.num_tasks = len(num_classes_tasks)
        self.task_bn = task_bn

        self.backbone = Deeplab_ResNet_Backbone(block, layers, task_bn, self.num_tasks)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc' % t_id, nn.Linear(512 * block.expansion, num_class))

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'logits' in name:
                params.append(param)
        return params

    def network_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'logits' not in name:
                params.append(param)
        return params

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'logits' not in name and 'fc' not in name:
                params.append(param)
        return params

    def fc_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'logits' not in name and 'fc' in name:
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(getattr(self, 'task%d_logits' % (t_id + 1)), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if not hard_sampling:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, 'task%d_logits' % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                # setattr(self, 'policy%d' % t_id, policy)
                self.policys.append(policy)
        else:
            raise ValueError
        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == 'all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers-self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers-self.skip_layer, 2)
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)

            self._arch_parameters = []
            self.register_parameter('task%d_logits' % (t_id + 1), nn.Parameter(task_logits, requires_grad=True))
            self._arch_parameters.append(getattr(self, 'task%d_logits' % (t_id + 1)))

    def load_imagenet_pretrain(self):
        self.backbone.load_imagenet_pretrain()

    def forward(self, imgs, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('num_train_layers in mtl forward = ', num_train_layers)

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        if isinstance(imgs, list):
            cuda_device = imgs[0].get_device()
        else:
            cuda_device = imgs.get_device()
        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == 'fix_policy':
                for p in self.policys:
                    assert(p is not None)
            else:
                raise NotImplementedError('mode %s is not implemented' % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)
                padding_policys.append(padding_policy)
                t_id_tmp = t_id if self.task_bn else -1
                if isinstance(imgs, list):
                    feats.append(self.backbone(imgs[t_id], padding_policy, t_id_tmp))
                else:
                    feats.append(self.backbone(imgs, padding_policy, t_id_tmp))

        else:
            if isinstance(imgs, list):
                feats = []
                for t_id in range(self.num_tasks):
                    t_id_tmp = t_id if self.task_bn else -1
                    feats.append(self.backbone(imgs[t_id], t_id=t_id_tmp))
            else:
                if self.task_bn:
                    feats = []
                    for t_id in range(self.num_tasks):
                        feats.append(self.backbone(imgs, t_id=t_id))
                else:
                    feats = [self.backbone(imgs, t_id=-1)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = self.avgpool(feats[t_id])
            output = getattr(self, 'task%d_fc' % t_id)(output.view(output.size(0), -1))
            outputs.append(output)

        return outputs, self.policys