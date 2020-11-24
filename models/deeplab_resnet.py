import sys
sys.path.insert(0, '..')
from models.base import *
import torch.nn.functional as F
from scipy.special import softmax
from models.util import count_params, compute_flops
import torch
import tqdm
import time
import math

def get_shape(shape1, shape2):
    out_shape = []
    for d1, d2 in zip(shape1, shape2):
        out_shape.append(min(d1, d2))
    return out_shape

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet, self).__init__()

        factor = 1
        self.in_planes = int(32 * factor)
        self.conv1 = conv3x3(3, int(32 * factor))
        self.bn1 = nn.BatchNorm2d(int(32 * factor))
        self.relu = nn.ReLU(inplace=True)

        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]
        self.blocks, self.ds = [], []
        self.parallel_blocks, self.parallel_ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.parallel_blocks.append(nn.ModuleList(blocks))
            self.parallel_ds.append(ds)
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
        self.parallel_ds = nn.ModuleList(self.parallel_ds)

        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256 * factor)), nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(256 * factor), num_class)

        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return layers, downsample

    def forward(self, x):
        t = 0
        x = self.seed(x)
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                output = self.blocks[segment][b](x)
                res_shape = residual.shape
                out_shape = output.shape
                new_shape = get_shape(res_shape, out_shape)
                x = F.relu(residual[:new_shape[0], :new_shape[1], :new_shape[2], :new_shape[3]] +
                           output[:new_shape[0], :new_shape[1], :new_shape[2], :new_shape[3]])
                t += 1

        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet(num_class=10, blocks=BasicBlock):
    return ResNet(blocks, [1, 1, 1], num_class)


class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(Deeplab_ResNet_Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        #
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
            # for i in downsample._modules['1'].parameters():
            #     i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_

                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(-1, 1, 1, 1) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1-policy[t])
                    t += 1
        return x


class Deeplab_ResNet_Backbone2(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(Deeplab_ResNet_Backbone2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        #
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
            # for i in downsample._modules['1'].parameters():
            #     i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_

                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    p = policy[t, 0]
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    if p == 1.:
                        x = F.relu(residual + self.blocks[segment][b](x))
                    elif p == 0.:
                        x = residual
                    else:
                        raise ValueError('p = %.2f is incorrect' % p)
                    t += 1
        return x


class MTL2(nn.Module):
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL2, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

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
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
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
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to('cuda:%d' % cuda_device)
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
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

    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('num_train_layers in mtl forward = ', num_train_layers)

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
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

                feats.append(self.backbone(img, padding_policy))
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)

        return outputs, self.policys, [None] * self.num_tasks


class MTL2_Backbone(nn.Module):
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL2_Backbone, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone2(block, layers)
        self.num_tasks = len(num_classes_tasks)

        # for t_id, num_class in enumerate(num_classes_tasks):
        #     setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
        #     setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
        #     setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
        #     setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def arch_parameters(self):
        return self._arch_parameters

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(getattr(self, 'task%d_logits' % (t_id + 1)), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to('cuda:%d' % cuda_device)
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to('cuda:%d' % cuda_device)
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
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

    def forward(self, img, temperature, is_policy, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('num_train_layers in mtl forward = ', num_train_layers)
        feats = []
        if is_policy:
            for t_id in range(self.num_tasks):
                padding_policy = self.policys[t_id]
                feats.append(self.backbone(img, padding_policy))
        else:
            self.backbone(img)

        outputs = []

        return outputs, self.policys, [None] * self.num_tasks


class MTL_SD(nn.Module):
    def __init__(self, block, layers, num_classes_tasks):
        super(MTL_SD, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def network_params(self):
        params = []
        for name, param in self.named_parameters():
            params.append(param)
        return params

    def train_sample_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            logit = getattr(self, 'task%d_logits' % (t_id + 1))
            policy = torch.bernoulli(logit)
            policys.append(policy)
        return policys

    def test_sample_policy(self):
        policys = []
        for t_id in range(self.num_tasks):
            logit = getattr(self, 'task%d_logits' % (t_id + 1))
            policys.append(logit)
        return policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        self.p_l = (1 - torch.arange(start=1, end=num_layers+1) / num_layers * 0.5)
        for t_id in range(self.num_tasks):
            task_logits = self.p_l * torch.ones(num_layers)
            self.register_buffer('task%d_logits' % (t_id + 1), task_logits)

    def forward(self, img, mode='train'):
        feats = []
        if mode == 'train':
            self.policys = self.train_sample_policy()
        elif mode == 'eval':
            self.policys = self.test_sample_policy()
        else:
            raise NotImplementedError('mode %s is not implemented' % mode)

        for t_id in range(self.num_tasks):
            policy = self.policys[t_id].float()

            feats.append(self.backbone(img, policy))

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)

        return outputs


class MTL_Instance(nn.Module):
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL_Instance, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1),
                    Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1),
                    Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1),
                    Classification_Module(512 * block.expansion, num_class, rate=24))

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
            if 'policynet' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'policynet' in name:
                params.append(param)
        return params

    def get_logits(self, img):
        batch_size = img.shape[0]
        self.task_logits = self.policynet(img).contiguous().view(batch_size, -1, 2)

    def train_sample_policy(self, img, temperature, hard_sampling):
        self.get_logits(img)
        task_logits_shape = self.task_logits.shape
        policy = F.gumbel_softmax(self.task_logits.contiguous().view(-1, 2), temperature, hard=hard_sampling)
        policy = policy.contiguous().view(task_logits_shape)
        policys = list(torch.split(policy,  sum(self.layers) - self.skip_layer, dim=1))
        return policys

    def test_sample_policy(self, img,  hard_sampling):
        self.policys = []
        if not hard_sampling:
            self.get_logits(img)
            task_logits_shape = self.task_logits.shape
            cuda_device = self.task_logits.get_device()
            logits = self.task_logits.contiguous().view(-1, 2)
            # logits = self.task_logits.detach().cpu().numpy().reshape(-1, 2)
            distribution = F.softmax(logits, dim=-1)
            s1 = distribution.shape[0]
            esl = torch.rand(s1).float().to(cuda_device)
            sampled = torch.where(esl < distribution[:, 0], torch.ones(s1, device=cuda_device),
                              torch.zeros(s1, device=cuda_device) )
            policy = [sampled, 1 - sampled]
            policy = torch.stack(policy, dim=-1).contiguous().view(task_logits_shape)
            policys = list(torch.split(policy, sum(self.layers) - self.skip_layer, dim=1))
        else:
            raise ValueError('hard sample is not supported')

        return policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        self.policynet = resnet(num_class=2 * self.num_tasks * (num_layers - self.skip_layer))
        self._arch_parameters = self.policynet.parameters()

    def forward(self, img, temperature, is_policy, policys=None, num_train_layers=None, hard_sampling=False, mode='train'):
        # print('num_train_layers in mtl forward = ', num_train_layers)

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        if is_policy:
            if mode == 'train':
                self.policys = self.train_sample_policy(img, temperature, hard_sampling)
            elif mode == 'eval':
                self.policys = self.test_sample_policy(img, hard_sampling)
            elif mode == 'fix_policy':
                assert(policys is not None)
                self.policys = policys
                # import pdb
                # pdb.set_trace()
                # for p in policys:
                #     print(p.shape)
            else:
                raise NotImplementedError('mode %s is not implemented' % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            batch_size = img.shape[0]
            if cuda_device != -1:
                padding = torch.ones(batch_size, skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(batch_size, skip_layer, 2)
            padding[:, :, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat((padding.float(), self.policys[t_id][:, -num_train_layers:].float()), dim=1).contiguous()
                padding_policys.append(padding_policy)
                feats.append(self.backbone(img, padding_policy))

            if mode == 'fix_policy':
                logits = [None] * self.num_tasks
            else:
                logits = list(torch.split(self.task_logits, sum(self.layers) - self.skip_layer, dim=1))
            self.policys = padding_policys
        else:
            feats = [self.backbone(img)] * self.num_tasks
            logits = [None] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)


        return outputs, self.policys, logits


class MTL_RL(nn.Module):
    def __init__(self, block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0):
        super(MTL_RL, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

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
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    def network_params(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def train_sample_policy(self):
        policys, logits = [], []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_logits' % (t_id + 1))
            logit = F.softmax(output, dim=-1)
            # policy [num_block]
            policy = torch.argmax(logit, dim=-1)
            policys.append(policy)
            logits.append(logit)
        return policys, logits

    def merge_logits(self, logits, random_logits, epsilon, cuda_device):
        merged_logits = []
        for t_id, (logit, random_logit) in enumerate(zip(logits, random_logits)):
            esl = torch.rand(1, device=cuda_device).float()
            esl = torch.where(esl > epsilon, torch.ones(1, device=cuda_device),
                              torch.zeros(1, device=cuda_device))

            merged_logits.append(esl * logit + (1. - esl) * random_logit)
        return merged_logits

    def random_policy(self, cuda_devices):
        """replace the output of policynet with Uniform [0, 1]"""
        num_layers = sum(self.layers)
        random_outputs = torch.rand(self.num_tasks * (num_layers - self.skip_layer), 2, device=cuda_devices)

        random_logits = F.softmax(random_outputs, dim=-1)
        random_policys = torch.argmax(random_logits, dim=-1)
        random_policys = torch.chunk(random_policys, self.num_tasks, dim=0)
        random_logits = torch.chunk(random_logits, self.num_tasks, dim=0)
        return random_policys, random_logits

    def sample_pred(self, logit):
        logit_shape = logit.shape
        cuda_device = logit.get_device()
        rand = torch.rand(logit_shape[0], device=cuda_device).float()
        rlo = torch.zeros(logit_shape[0], device=cuda_device).float()

        for i in range(2):
            if i == 0:
                temp = logit[:, 0]
            else:
                temp = torch.sum(logit[:, :i + 1], dim=1, keepdim=False)

            rlo += torch.where(torch.ge(temp, rand),
                               torch.zeros(logit_shape[0], device=cuda_device),
                               torch.ones(logit_shape[0], device=cuda_device))

        policy = rlo.float()
        return policy

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == 'all_chosen':
                assert (self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)

            self._arch_parameters = []
            self.register_parameter('task%d_logits' % (t_id + 1), nn.Parameter(task_logits, requires_grad=True))
            self._arch_parameters.append(getattr(self, 'task%d_logits' % (t_id + 1)))

    def forward(self, img, mode, epsilon=None, pred_logits=None, num_train_layers=None):
        # get the policy in each different mode
        cuda_device = img.get_device()

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        skip_layer = sum(self.layers) - num_train_layers
        if cuda_device != -1:
            padding = torch.ones(skip_layer).to(cuda_device)
        else:
            padding = torch.ones(skip_layer)

        self.policys = []
        self.logits = []
        if mode == 'full':
            self.policys = [None] * self.num_tasks
            self.logits = [None] * self.num_tasks
        elif mode == 'pred':
            # compute directly from the policy network
            self.policys, self.logits = self.train_sample_policy()
        elif mode == 'random':
            assert (pred_logits is not None)
            _, random_logits = self.random_policy(cuda_device)
            self.logits = self.merge_logits(pred_logits, random_logits, epsilon, cuda_device)
            self.policys = []
            for logit in self.logits:
                self.policys.append(self.sample_pred(logit))

        padding_policys = []
        for t_id in range(self.num_tasks):
            if self.policys[t_id] is not None:
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)
                padding_policys.append(padding_policy)
            else:
                padding_policys.append(None)

        feats = []
        for t_id in range(self.num_tasks):
            feats.append(self.backbone(img, padding_policys[t_id]))

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats[t_id]) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats[t_id])
            outputs.append(output)

        return outputs, padding_policys


if __name__ == '__main__':
    # block = Bottleneck
    backbone = 'ResNet34'
    # tasks_num_class = [40, 3]
    # tasks_num_class = [19, 1]
    # tasks_num_class = [40, 3, 1]
    tasks_num_class = [17, 3, 1, 1, 1]
    if backbone == 'ResNet18':
        layers = [2, 2, 2, 2]
        block = BasicBlock
    elif backbone == 'ResNet34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif backbone == 'ResNet101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
    else:
        raise ValueError('backbone %s is invalid' % backbone)

    # block, layers, num_classes_tasks, init_method, init_neg_logits=None, skip_layer=0
    net = MTL2_Backbone(block, layers, tasks_num_class, 'equal')

    img = torch.ones((1, 3, 224, 224))

    # outs, policys = net(img, 5, True)
    count_params(net.backbone)

    import numpy as np
    if len(tasks_num_class) == 3:
        policy1 = np.array([1, 1, 1, 1,
                   1, 1, 0, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 0, 0, 0,
                            1, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()

        policy3 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 0, 0,
                            1, 1, 1, 1, ])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype('float')
        policy3 = torch.from_numpy(policy3).cuda()
        policys = [policy1, policy2, policy3]

    elif len(tasks_num_class) == 2 and backbone == 'ResNet34':
        policy1 = np.array([1, 1, 1, 1,
                   1, 0, 0, 1,
                   0, 1, 1, 1,
                   0, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 0, 1,
                            1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 2 and backbone == 'ResNet18':
        policy1 = np.array([1, 0, 1, 1,
                   1, 1, 1, 1])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1 ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 5:
        policy1 = np.array([1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype('float')
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1,
                            1, 0, 1, 1,
                            1, 1, 1, 1,
                            0, 1, 1, 1, ])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype('float')
        policy2 = torch.from_numpy(policy2).cuda()
        policy3 = np.array([1, 1, 1, 1,
                   0, 1, 0, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype('float')
        policy3 = torch.from_numpy(policy3).cuda()
        policy4 = np.array([1, 1, 1, 0,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,])
        policy4 = np.stack([policy4, 1 - policy4], axis=1).astype('float')
        policy4 = torch.from_numpy(policy4).cuda()
        policy5 = np.array([1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 0, 0, 1,
                   0, 1, 1, 1,])
        policy5 = np.stack([policy5, 1 - policy5], axis=1).astype('float')
        policy5 = torch.from_numpy(policy5).cuda()
        policys = [policy1, policy2, policy3, policy4, policy5]
    else:
        raise ValueError

    setattr(net, 'policys', policys)

    times = []
    input_dict = {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'}
    net.cuda()
    for _ in tqdm.tqdm(range(1000)):
        start_time = time.time()
        img = torch.rand((1, 3, 224, 224)).cuda()
        net(img, **input_dict)
        times.append(time.time() - start_time)

    print('Average time = ', np.mean(times))

    gflops = compute_flops(net, img.cuda(), {'temperature': 5, 'is_policy': True, 'mode': 'fix_policy'})
    print('Number of FLOPs = %.2f G' % (gflops / 1e9 / 2))
    pdb.set_trace()



