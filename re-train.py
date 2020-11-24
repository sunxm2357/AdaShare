import sys
sys.path.insert(0, '..')

import os
import time
import numpy as np

from torch.utils.data import DataLoader

from dataloaders.nyu_v2_dataloader import NYU_v2
from dataloaders.cityscapes_dataloader import CityScapes
from dataloaders.taskonomy_dataloader import Taskonomy
from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import print_separator, read_yaml, create_path, print_yaml, should, fix_random_seed
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def eval_fix_policy(environ, dataloader, tasks, num_seg_cls=-1, eval_iter=10):
    batch_size = []
    records = {}
    val_metrics = {}

    if 'seg' in tasks:
        assert (num_seg_cls != -1)
        records['seg'] = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((num_seg_cls, num_seg_cls)),
                          'labels': np.arange(num_seg_cls), 'gts': [], 'preds': []}
    if 'sn' in tasks:
        records['sn'] = {'cos_similaritys': []}
    if 'depth' in tasks:
        records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
    if 'keypoint' in tasks:
        records['keypoint'] = {'errs': []}
    if 'edge' in tasks:
        records['edge'] = {'errs': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if eval_iter != -1:
                if batch_idx > eval_iter:
                    break
            environ.set_inputs(batch)
            metrics = environ.val_fix_policy()
            # environ.networks['mtl-net'].task1_logits
            # mIoUs.append(mIoU)
            if 'seg' in tasks:
                records['seg']['gts'].append(metrics['seg']['gt'])
                records['seg']['preds'].append(metrics['seg']['pred'])
                new_mat = confusion_matrix(metrics['seg']['gt'], metrics['seg']['pred'], records['seg']['labels'])
                assert (records['seg']['conf_mat'].shape == new_mat.shape)
                records['seg']['conf_mat'] += new_mat
                records['seg']['pixelAccs'].append(metrics['seg']['pixelAcc'])
                records['seg']['errs'].append(metrics['seg']['err'])
            if 'sn' in tasks:
                records['sn']['cos_similaritys'].append(metrics['sn']['cos_similarity'])
            if 'depth' in tasks:
                records['depth']['abs_errs'].append(metrics['depth']['abs_err'])
                records['depth']['rel_errs'].append(metrics['depth']['rel_err'])
                records['depth']['sq_rel_errs'].append(metrics['depth']['sq_rel_err'])
                records['depth']['ratios'].append(metrics['depth']['ratio'])
                records['depth']['rms'].append(metrics['depth']['rms'])
                records['depth']['rms_log'].append(metrics['depth']['rms_log'])
            if 'keypoint' in tasks:
                records['keypoint']['errs'].append(metrics['keypoint']['err'])
            if 'edge' in tasks:
                records['edge']['errs'].append(metrics['edge']['err'])
            batch_size.append(len(batch['img']))

    if 'seg' in tasks:
        val_metrics['seg'] = {}
        jaccard_perclass = []
        for i in range(num_seg_cls):
            if not records['seg']['conf_mat'][i, i] == 0:
                jaccard_perclass.append(records['seg']['conf_mat'][i, i] / (np.sum(records['seg']['conf_mat'][i, :]) +
                                                                            np.sum(records['seg']['conf_mat'][:, i]) -
                                                                            records['seg']['conf_mat'][i, i]))

        val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

        val_metrics['seg']['Pixel Acc'] = (np.array(records['seg']['pixelAccs']) * np.array(batch_size)).sum() / sum(
            batch_size)

        val_metrics['seg']['err'] = (np.array(records['seg']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

    if 'sn' in tasks:
        val_metrics['sn'] = {}
        overall_cos = np.clip(np.concatenate(records['sn']['cos_similaritys']), -1, 1)

        angles = np.arccos(overall_cos) / np.pi * 180.0
        val_metrics['sn']['cosine_similarity'] = overall_cos.mean()
        val_metrics['sn']['Angle Mean'] = np.mean(angles)
        val_metrics['sn']['Angle Median'] = np.median(angles)
        val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

    if 'depth' in tasks:
        val_metrics['depth'] = {}
        records['depth']['abs_errs'] = np.stack(records['depth']['abs_errs'], axis=0)
        records['depth']['rel_errs'] = np.stack(records['depth']['rel_errs'], axis=0)
        records['depth']['ratios'] = np.concatenate(records['depth']['ratios'], axis=0)
        val_metrics['depth']['abs_err'] = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['rel_err'] = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
        val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
        val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100

    if 'keypoint' in tasks:
        val_metrics['keypoint'] = {}
        val_metrics['keypoint']['err'] = (np.array(records['keypoint']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    if 'edge' in tasks:
        val_metrics['edge'] = {}
        val_metrics['edge']['err'] = (np.array(records['edge']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    return val_metrics


def _train(exp_id, opt, gpu_ids):

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')
    opt_tmp = opt if opt['policy_model'] == 'instance-specific' else None
    if opt['dataload']['dataset'] == 'NYU_v2':
        trainset = NYU_v2(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'], opt['dataload']['crop_w'], opt=opt_tmp)
        valset = NYU_v2(opt['dataload']['dataroot'], 'test', opt=opt_tmp)
    elif opt['dataload']['dataset'] == 'CityScapes':
        num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1
        trainset = CityScapes(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'],
                              opt['dataload']['crop_w'], num_class=num_seg_class, small_res=opt['dataload']['small_res'],
                              opt=opt_tmp)
        valset = CityScapes(opt['dataload']['dataroot'], 'test', num_class=num_seg_class,
                            small_res=opt['dataload']['small_res'], opt=opt_tmp)
    elif opt['dataload']['dataset'] == 'Taskonomy':
        trainset = Taskonomy(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        valset = Taskonomy(opt['dataload']['dataroot'], 'test_small')
    else:
        raise NotImplementedError('Dataset %s is not implemented' % opt['dataload']['dataset'])
    print('size of training set: ', len(trainset))
    print('size of validation set: ', len(valset))

    train_loader = DataLoader(trainset, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2,
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2, shuffle=False)


    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                           opt['tasks_num_class'], opt['init_neg_logits'],
                           gpu_ids[0], opt['train']['init_temp'], opt['train']['decay_temp'],
                           is_train=True, opt=opt)

    current_iter = 0
    policy_label = 'Iter%s_rs%04d' % (opt['train']['policy_iter'], opt['seed'][exp_id])
    if opt['train']['retrain_resume']:
        current_iter = environ.load(opt['train']['which_iter'])
        if opt['policy_model'] == 'task-specific':
            environ.load_policy(policy_label)
    else:
        if opt['policy_model'] == 'task-specific':
            init_state = deepcopy(environ.get_current_state(0))
            if environ.check_exist_policy(policy_label):
                environ.load_policy(policy_label)
            else:
                environ.load(opt['train']['policy_iter'])
                dists = environ.get_policy_prob()
                overall_dist = np.concatenate(dists, axis=-1)
                print(overall_dist)
                environ.sample_policy(opt['train']['hard_sampling'])
                environ.save_policy(policy_label)

            if opt['retrain_from_pl']:
                environ.load(opt['train']['policy_iter'])
            else:
                environ.load_snapshot(init_state)

    if opt['policy_model'] == 'task-specific':
        policys = environ.get_current_policy()
        overall_policy = np.concatenate(policys, axis=-1)
        print(overall_policy)

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    environ.fix_alpha()
    environ.free_w(fix_BN=opt['fix_BN'])
    batch_enumerator = enumerate(train_loader)
    if opt['dataload']['dataset'] == 'NYU_v2':
        if len(opt['tasks_num_class']) == 2:
            refer_metrics = {'seg': {'mIoU': 0.413, 'Pixel Acc': 0.691},
                             'sn': {'Angle Mean': 15, 'Angle Median': 11.5, 'Angle 11.25': 49.2, 'Angle 22.5': 76.7,
                                    'Angle 30': 86.8}}
        elif len(opt['tasks_num_class']) == 3:
            refer_metrics = {'seg': {'mIoU': 0.275, 'Pixel Acc': 0.589},
                             'sn': {'Angle Mean': 17.5, 'Angle Median': 14.2, 'Angle 11.25': 34.9, 'Angle 22.5': 73.3,
                                    'Angle 30': 85.7},
                             'depth': {'abs_err': 0.62, 'rel_err': 0.25, 'sigma_1.25': 57.9,
                                       'sigma_1.25^2': 85.8, 'sigma_1.25^3': 95.7}}
        else:
            raise ValueError('num_class = %d is invalid' % len(opt['tasks_num_class']))

    elif opt['dataload']['dataset'] == 'CityScapes':
        num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1

        if num_seg_class == 7 and opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.519, 'Pixel Acc': 0.722},
                         'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                   'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        elif num_seg_class == 7 and not opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.644, 'Pixel Acc': 0.778},
                         'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                   'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        
        elif num_seg_class == 19 and not opt['dataload']['small_res']:
            refer_metrics = {'seg': {'mIoU': 0.402, 'Pixel Acc': 0.747},
                            'depth': {'abs_err': 0.017, 'rel_err': 0.33, 'sigma_1.25': 70.3,
                                    'sigma_1.25^2': 86.3, 'sigma_1.25^3': 93.3}}
        else:
            raise ValueError('num_seg_class = %d and small res = %d are not supported' % (num_seg_class, opt['dataload']['small_res']))
 
    elif opt['dataload']['dataset'] == 'Taskonomy':
        refer_metrics = {'seg': {'err': 0.517},
                         'sn': {'cosine_similarity': 0.716},
                         'depth': {'abs_err': 0.021},
                         'keypoint': {'err': 0.197},
                         'edge': {'err': 0.212}}
    else:
        raise NotImplementedError('Dataset %s is not implemented' % opt['dataload']['dataset'])

    best_value, best_iter = 0, 0
    best_metrics = None
    opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])
    while current_iter < opt['train']['retrain_total_iters']:
        start_time = time.time()
        environ.train()
        current_iter += 1
        # image-level training
        batch_idx, batch = next(batch_enumerator)

        environ.set_inputs(batch)
        environ.optimize_fix_policy(opt['lambdas'])

        if should(current_iter, opt['train']['print_freq']):
            environ.print_loss(current_iter, start_time)
            environ.resize_results()

        if should(current_iter, opt['train']['val_freq']):
            environ.eval()
            num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1
            val_metrics = eval_fix_policy(environ, val_loader, opt['tasks'], num_seg_class)
            environ.print_loss(current_iter, start_time, val_metrics)
            environ.save('retrain%03d_policyIter%s_latest' % (exp_id, opt['train']['policy_iter']), current_iter)
            environ.train()
            new_value = 0

            for k in refer_metrics.keys():
                if k in val_metrics.keys():
                    for kk in val_metrics[k].keys():
                        if not kk in refer_metrics[k].keys():
                            continue
                        if (k == 'sn' and kk in ['Angle Mean', 'Angle Median']) or (
                                k == 'depth' and not kk.startswith('sigma')) or (kk == 'err'):
                            value = refer_metrics[k][kk] / val_metrics[k][kk]
                        else:
                            value = val_metrics[k][kk] / refer_metrics[k][kk]

                        value = value / len(list(set(val_metrics[k].keys()) & set(refer_metrics[k].keys())))
                        new_value += value

            if new_value > best_value:
                best_value = new_value
                best_metrics = val_metrics
                best_iter = current_iter
                environ.save('retrain%03d_policyIter%s_best' % (exp_id, opt['train']['policy_iter']), current_iter)
            print('new value: %.3f' % new_value)
            print('best iter: %d, best value: %.3f' % (best_iter, best_value), best_metrics)

        if batch_idx == len(train_loader) - 1:
            batch_enumerator = enumerate(train_loader)

    return best_metrics


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    print_separator('READ YAML')
    opt, gpu_ids, exp_ids = read_yaml()
    # fix_random_seed(opt["seed"])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)

    best_results = {}
    for exp_id in exp_ids:
        fix_random_seed(opt["seed"][exp_id])
        # fix_random_seed(48)
        _, policy = _train(exp_id, opt, gpu_ids)


if __name__ == "__main__":
    train()
