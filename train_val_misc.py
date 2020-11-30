import sys
sys.path.insert(0, '..')
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.MultiDomainSampler import MultiDomainSampler
from dataloaders.domainnet_dataloader import *
from utils.util import read_yaml, create_path, print_yaml, fix_random_seed
import torch
import json


def eval(environ, dataloader, tasks, bs, domains, val_kwarg):
    batch_size = []
    records = {}
    val_metrics = {}
    for t in tasks:
        records[t] = {'Accs': [],  'gts': [], 'preds': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # if batch_idx > 10:
            #     break
            batch = organize_batch(batch, bs, domains)
            environ.set_inputs(batch)
            metrics = environ.val(**val_kwarg)
            for t in tasks:
                records[t]['gts'].append(metrics[t]['gt'])
                records[t]['preds'].append(metrics[t]['pred'])
                records[t]['Accs'].append(metrics[t]['acc'])
            batch_size.append(len(batch['%s_img' % domains[0]]))

    for t in tasks:
        val_metrics[t] = {}
        # calculate the number of pixels in each class
        records[t]['gts'] = np.concatenate(records[t]['gts'], axis=0)
        records[t]['preds'] = np.concatenate(records[t]['preds'], axis=0)
        val_metrics[t]['Acc'] = (np.array(records[t]['Accs']) * np.array(batch_size)).sum() / sum(batch_size)
    return val_metrics


def create_folder_print_args():
    opt, gpu_ids, exp_ids = read_yaml()
    fix_random_seed(opt["seed"][0])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)

    return opt, gpu_ids, exp_ids


def init_dataloaders(opt, modes):
    datasets = {}
    samplers = {}
    dataloaders = {}
    if opt['dataload']['dataset'] == 'DomainNet':
        name = 'DomainNet.json'
        overlap = opt['dataload'].get('overlap', None)
        copy = opt['dataload'].get('copy', None)
        if overlap is not None and copy is not None:
            name = 'DomainNet_overlap%.2f_copy%d.json' % (overlap, copy)
        with open(os.path.join(opt['dataload']['dataroot'], name), 'r') as f:
            split_domain_info = json.load(f)
        for mode in modes:
            batch_size = opt['train']['batch_size'] if mode in ['train', 'train1', 'train2', 'val'] else opt['test']['batch_size']
            random_shuffle = True if mode in ['train', 'train1', 'train2'] else False
            sampler = MultiDomainSampler(split_domain_info, mode, batch_size,  opt['dataload']['domains'], random_shuffle)
            dataset = DomainNet(opt['dataload']['dataroot'], mode,  opt['dataload']['crop_h'],  opt['dataload']['crop_w'],
                                overlap=overlap, copy=copy)
            print('size of %s dataset: ' % mode, len(dataset))
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
            datasets[mode] = dataset
            samplers[mode] = sampler
            dataloaders[mode] = loader
    else:
        raise ValueError('Dataset %s is not supported' % opt['dataload']['dataset'])

    return datasets, samplers, dataloaders


def set_reference(opt):
    if opt['dataload']['dataset'] == 'DomainNet':
        refer_metrics = {}
        for domain in opt['dataload']['domains']:
            refer_metrics[domain] = {'Acc': 1, 'perturb Acc': 1}
    else:
        raise NotImplementedError('Dataset %s is not implemented' % opt['dataload']['dataset'])
    return refer_metrics


def rel_performance(refer_metrics, val_metrics):
    new_value = 0
    for k in refer_metrics.keys():
        if k in val_metrics.keys():
            for kk in val_metrics[k].keys():
                if not kk in refer_metrics[k].keys():
                    continue
                # higher is better
                value = val_metrics[k][kk] / refer_metrics[k][kk]
                value = value / len(list(set(val_metrics[k].keys()) & set(refer_metrics[k].keys())))
                new_value += value

    return new_value


def organize_batch(batch, batch_size, domains):
    assert batch['img'].shape[0] == batch_size * len(domains)
    assert len(batch['img_path']) == batch_size * len(domains)
    assert len(batch['img_idx']) == batch_size * len(domains)
    new_batch = {}
    for d_idx, domain in enumerate(domains):
        new_batch['%s_img' % domain] = batch['img'][d_idx * batch_size: (d_idx + 1) * batch_size]
        new_batch['%s_img_path' % domain] = batch['img_path'][d_idx * batch_size :(d_idx + 1) * batch_size]
        new_batch['%s_img_idx' % domain] = batch['img_idx'][d_idx * batch_size: (d_idx + 1) * batch_size]
    return new_batch