import torch.nn as nn
import numpy as np
import torch
affine_par = True


# ################################################
# ############## ResNet Modules ##################
# ################################################

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, num_tasks=-1):
        super(BasicBlock, self).__init__()
        self.planes = planes

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype('int')
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if num_tasks == -1:
            self.bn1 = nn.BatchNorm2d(planes,  affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn1' % t_idx, bn)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if num_tasks == -1:
            self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn2' % t_idx, bn)

    def forward(self, x, keep=None, t_id=-1):
        # keep: [batch_size], int
        cuda_device = x.get_device()
        out = self.conv1(x)
        if t_id == -1:
            out = self.bn1(out)
        else:
            bn = getattr(self, 'task%d_bn1' % t_id)
            out = bn(out)

        # used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask

        out = self.relu(out)
        out = self.conv2(out)
        if t_id == -1:
            y = self.bn2(out)
        else:
            bn = getattr(self, 'task%d_bn2' % t_id)
            y = bn(out)
        return y


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, num_tasks=-1):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if num_tasks == -1:
            self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn1' % t_idx, bn)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if num_tasks == -1:
            self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn2' % t_idx, bn)


    def forward(self, x, t_id=-1):
        out = self.conv1(x)
        if t_id == -1:
            out = self.bn1(out)
        else:
            bn = getattr(self, 'task%d_bn1' % t_id)
            out = bn(out)
        out1 = self.relu(out)
        out = self.conv2(out1)
        if t_id == -1:
            y = self.bn2(out)
        else:
            bn = getattr(self, 'task%d_bn2' % t_id)
            y = bn(out)
        return y, out1


# No projection: identity shortcut and atrous
class Bottleneck(nn.Module):
    expansion = 4

    # |----------------------------------------------------------------|
    # 1x1 conv -> bn -> relu -> 3x3 conv -> bn -> relu -> 1x1 conv -> bn -> relu
    def __init__(self, inplanes, planes, stride=1, dilation=1, num_tasks=-1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        if num_tasks == -1:
            self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn1' % t_idx, bn)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        if num_tasks == -1:
            self.bn2 = nn.BatchNorm2d(planes,  affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn2' % t_idx, bn)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if num_tasks == -1:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion,  affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes * self.expansion, affine=affine_par)
                setattr(self, 'task%d_bn3' % t_idx, bn)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, t_id=-1):
        out = self.conv1(x)
        if t_id == -1:
            out = self.bn1(out)
        else:
            bn = getattr(self, 'task%d_bn1' % t_id)
            out = bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        if t_id == -1:
            out = self.bn2(out)
        else:
            bn = getattr(self, 'task%d_bn2' % t_id)
            out = bn(out)

        out = self.relu(out)
        out = self.conv3(out)
        if t_id == -1:
            out = self.bn3(out)
        else:
            bn = getattr(self, 'task%d_bn3' % t_id)
            out = bn(out)

        return out


class Bottleneck2(nn.Module):
    expansion = 4

    # |----------------------------------------------------------------|
    # 1x1 conv -> bn -> relu -> 3x3 conv -> bn -> relu -> 1x1 conv -> bn -> relu
    def __init__(self, inplanes, planes, stride=1, dilation=1, num_tasks=-1):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        if num_tasks == -1:
            self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn1' % t_idx, bn)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        if num_tasks == -1:
            self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes, affine=affine_par)
                setattr(self, 'task%d_bn2' % t_idx, bn)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if num_tasks == -1:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion,  affine=affine_par)
        else:
            for t_idx in range(num_tasks):
                bn = nn.BatchNorm2d(planes * self.expansion, affine=affine_par)
                setattr(self, 'task%d_bn3' % t_idx, bn)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x,  t_id=-1):
        out = self.conv1(x)
        if t_id == -1:
            out = self.bn1(out)
        else:
            bn = getattr(self, 'task%d_bn1' % t_id)
            out = bn(out)
        out1 = self.relu(out)

        out = self.conv2(out1)
        if t_id == -1:
            out = self.bn2(out)
        else:
            bn = getattr(self, 'task%d_bn2' % t_id)
            out = bn(out)

        out = self.relu(out)

        out = self.conv3(out)
        if t_id == -1:
            out = self.bn3(out)
        else:
            bn = getattr(self, 'task%d_bn3' % t_id)
            out = bn(out)

        return out, out1


class Residual_Convolution(nn.Module):
    def __init__(self, icol, ocol, num_classes, rate):
        super(Residual_Convolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        dow1 = self.dropout(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        add1 = self.dropout(add1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


# coarse and fine segmentation
class Residual_Refinement_Module(nn.Module):

    def __init__(self, inplanes, base_channels, num_classes, rate):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Convolution(inplanes, base_channels, num_classes, rate)
        self.RC2 = Residual_Convolution(inplanes, base_channels, num_classes, rate)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]


class Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(Classification_Module, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x
