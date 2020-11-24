import sys
sys.path.insert(0, '..')

import os
import time

from torch.utils.data import DataLoader

from dataloaders.nyu_v2_dataloader import NYU_v2
from dataloaders.cityscapes_dataloader import CityScapes
from dataloaders.taskonomy_dataloader import Taskonomy
from envs.blockdrop_env import BlockDropEnv
import torch
from utils.util import makedir, print_separator, read_yaml, create_path, print_yaml, should, fix_random_seed
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def eval(environ, dataloader, tasks, policy=False, num_train_layers=None, hard_sampling=False, num_seg_cls=-1,
         eval_iter=10):
    batch_size = []
    records = {}
    val_metrics = {}
    if 'seg' in tasks:
        assert (num_seg_cls != -1)
        records['seg'] = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((num_seg_cls, num_seg_cls)),
                          'labels': np.arange(num_seg_cls)}
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
            # seg_pred, seg_gt, pixelAcc, cos_similarity = environ.val(policy, num_train_layers, hard_sampling)
            metrics = environ.val2(policy, num_train_layers, hard_sampling)

            # environ.networks['mtl-net'].task1_logits
            # mIoUs.append(mIoU)
            if 'seg' in tasks:
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

    # overall_mIoU = (np.array(mIoUs) * np.array(batch_size)).sum() / sum(batch_size)
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
        val_metrics['sn']['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
        val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

    if 'depth' in tasks:
        val_metrics['depth'] = {}
        records['depth']['abs_errs'] = np.stack(records['depth']['abs_errs'], axis=0)
        records['depth']['rel_errs'] = np.stack(records['depth']['rel_errs'], axis=0)
        records['depth']['sq_rel_errs'] = np.stack(records['depth']['sq_rel_errs'], axis=0)
        records['depth']['ratios'] = np.concatenate(records['depth']['ratios'], axis=0)
        records['depth']['rms'] = np.concatenate(records['depth']['rms'], axis=0)
        records['depth']['rms_log'] = np.concatenate(records['depth']['rms_log'], axis=0)
        records['depth']['rms_log'] = records['depth']['rms_log'][~np.isnan(records['depth']['rms_log'])]
        val_metrics['depth']['abs_err'] = (records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['rel_err'] = (records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
        val_metrics['depth']['sq_rel_err'] = (records['depth']['sq_rel_errs'] * np.array(batch_size)).sum() / sum(
            batch_size)
        val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25)) * 100
        val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 2)) * 100
        val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(records['depth']['ratios'], 1.25 ** 3)) * 100
        val_metrics['depth']['rms'] = (np.sum(records['depth']['rms']) / len(records['depth']['rms'])) ** 0.5
        # val_metrics['depth']['rms_log'] = (np.sum(records['depth']['rms_log']) / len(records['depth']['rms_log'])) ** 0.5

    if 'keypoint' in tasks:
        val_metrics['keypoint'] = {}
        val_metrics['keypoint']['err'] = (np.array(records['keypoint']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    if 'edge' in tasks:
        val_metrics['edge'] = {}
        val_metrics['edge']['err'] = (np.array(records['edge']['errs']) * np.array(batch_size)).sum() / sum(
            batch_size)

    return val_metrics


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    # read the yaml
    print_separator('READ YAML')
    opt, gpu_ids, _ = read_yaml()
    fix_random_seed(opt["seed"][0])
    create_path(opt)
    # print yaml on the screen
    lines = print_yaml(opt)
    for line in lines: print(line)
    # print to file
    with open(os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'opt.txt'), 'w+') as f:
        f.writelines(lines)

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')
    if opt['dataload']['dataset'] == 'NYU_v2':
        # To warm up
        trainset = NYU_v2(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        # To update the network parameters
        trainset1 = NYU_v2(opt['dataload']['dataroot'], 'train1', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        # To update the policy weights
        trainset2 = NYU_v2(opt['dataload']['dataroot'], 'train2', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        valset = NYU_v2(opt['dataload']['dataroot'], 'test')
    elif opt['dataload']['dataset'] == 'CityScapes':
        num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1
        # To warm up
        trainset = CityScapes(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'], opt['dataload']['crop_w'],
                              num_class=num_seg_class, small_res=opt['dataload']['small_res'])
        # To update the network parameters
        trainset1 = CityScapes(opt['dataload']['dataroot'], 'train1', opt['dataload']['crop_h'], opt['dataload']['crop_w'],
                               num_class=num_seg_class, small_res=opt['dataload']['small_res'])
        # To update the policy weights
        trainset2 = CityScapes(opt['dataload']['dataroot'], 'train2', opt['dataload']['crop_h'], opt['dataload']['crop_w'],
                               num_class=num_seg_class, small_res=opt['dataload']['small_res'])
        valset = CityScapes(opt['dataload']['dataroot'], 'test', num_class=num_seg_class, small_res=opt['dataload']['small_res'])
    elif opt['dataload']['dataset'] == 'Taskonomy':
        trainset = Taskonomy(opt['dataload']['dataroot'], 'train', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        trainset1 = Taskonomy(opt['dataload']['dataroot'], 'train1', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        trainset2 = Taskonomy(opt['dataload']['dataroot'], 'train2', opt['dataload']['crop_h'], opt['dataload']['crop_w'])
        valset = Taskonomy(opt['dataload']['dataroot'], 'test_small')
    else:
        raise NotImplementedError('Dataset %s is not implemented' % opt['dataload']['dataset'])

    print('size of training set: ', len(trainset))
    print('size of training set 1: ', len(trainset1))
    print('size of training set 2: ', len(trainset2))
    print('size of test set: ', len(valset))

    train_loader = DataLoader(trainset, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2, shuffle=True)
    train1_loader = DataLoader(trainset1, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2, shuffle=True)
    train2_loader = DataLoader(trainset2, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2, shuffle=True)
    val_loader = DataLoader(valset, batch_size=opt['train']['batch_size'], drop_last=True, num_workers=2, shuffle=False)

    opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate', len(train1_loader))
    opt['train']['alpha_iter_alternate'] = opt['train'].get('alpha_iter_alternate', len(train2_loader))

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                           opt['tasks_num_class'], opt['init_neg_logits'], gpu_ids[0],
                           opt['train']['init_temp'], opt['train']['decay_temp'], is_train=True, opt=opt)

    current_iter = 0
    current_iter_w, current_iter_a = 0, 0
    if opt['train']['resume']:
        current_iter = environ.load(opt['train']['which_iter'])
        environ.networks['mtl-net'].reset_logits()

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    batch_enumerator = enumerate(train_loader)
    batch_enumerator1 = enumerate(train1_loader)
    batch_enumerator2 = enumerate(train2_loader)
    flag = 'update_w'
    environ.fix_alpha()
    environ.free_w(opt['fix_BN'])
    best_value, best_iter = 0, 0

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

    best_metrics = None
    p_epoch = 0
    flag_warmup = True
    if opt['backbone'] == 'ResNet18':
        num_blocks = 8
    elif opt['backbone'] in ['ResNet34', 'ResNet50']:
        num_blocks = 18
    elif opt['backbone'] == 'ResNet101':
        num_blocks = 33
    elif opt['backbone'] == 'WRN':
        num_blocks = 15
    else:
        raise ValueError('Backbone %s is invalid' % opt['backbone'])

    while current_iter < opt['train']['total_iters']:
        start_time = time.time()
        environ.train()
        current_iter += 1
        # warm up
        if current_iter < opt['train']['warm_up_iters']:
            batch_idx, batch = next(batch_enumerator)
            environ.set_inputs(batch)
            environ.optimize(opt['lambdas'], is_policy=False, flag='update_w')
            if batch_idx == len(train_loader) - 1:
                batch_enumerator = enumerate(train_loader)

            if should(current_iter, opt['train']['print_freq']):
                environ.print_loss(current_iter, start_time)
                environ.resize_results()

            # validation
            if should(current_iter, opt['train']['val_freq']):
                environ.eval()
                num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1
                val_metrics = eval(environ, val_loader, opt['tasks'], policy=False, num_train_layers=None, num_seg_cls=num_seg_class)
                environ.print_loss(current_iter, start_time, val_metrics)
                environ.save('latest', current_iter)
                environ.train()

        else:
            if flag_warmup:
                environ.define_optimizer(policy_learning=True)
                environ.define_scheduler(True)

                flag_warmup = False

            if current_iter == opt['train']['warm_up_iters']:
                environ.save('warmup', current_iter)
                environ.fix_alpha()

            # Update the network weights
            if flag == 'update_w':
                current_iter_w += 1
                batch_idx_w, batch = next(batch_enumerator1)
                environ.set_inputs(batch)

                if opt['is_curriculum']:
                    num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                else:
                    num_train_layers = None

                environ.optimize(opt['lambdas'], is_policy=opt['policy'], flag=flag, num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'])

                if should(current_iter, opt['train']['print_freq']):
                    environ.print_loss(current_iter, start_time)
                    environ.resize_results()

                if should(current_iter_w, opt['train']['weight_iter_alternate']):
                    flag = 'update_alpha'
                    environ.fix_w()
                    environ.free_alpha()
                    # do the validation on the test set
                    environ.eval()
                    print('Evaluating...')
                    num_seg_class = opt['tasks_num_class'][opt['tasks'].index('seg')] if 'seg' in opt['tasks'] else -1
                    val_metrics = eval(environ, val_loader, opt['tasks'], policy=opt['policy'],
                                       num_train_layers=num_train_layers, hard_sampling=opt['train']['hard_sampling'],
                                       num_seg_cls=num_seg_class)
                    environ.print_loss(current_iter, start_time, val_metrics)
                    environ.save('latest', current_iter)

                    if current_iter - opt['train']['warm_up_iters'] >= num_blocks * opt['curriculum_speed'] * \
                            (opt['train']['weight_iter_alternate'] + opt['train']['alpha_iter_alternate']):
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
                            environ.save('best', current_iter)
                        print('new value: %.3f' % new_value)
                        print('best iter: %d, best_value: %.3f' % (best_iter, best_value), best_metrics)
                    environ.train()

                if batch_idx_w == len(train1_loader) - 1:
                    batch_enumerator1 = enumerate(train1_loader)

            # update the policy network
            elif flag == 'update_alpha':
                current_iter_a += 1
                batch_idx_a, batch = next(batch_enumerator2)
                environ.set_inputs(batch)
                if opt['is_curriculum']:
                    num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                else:
                    num_train_layers = None

                environ.optimize(opt['lambdas'], is_policy=opt['policy'], flag=flag, num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'])

                if should(current_iter, opt['train']['print_freq']):
                    environ.print_loss(current_iter, start_time)
                    environ.resize_results()
                    # environ.visual_policy(current_iter)

                if should(current_iter_a, opt['train']['alpha_iter_alternate']):
                    flag = 'update_w'
                    environ.fix_alpha()
                    environ.free_w(opt['fix_BN'])
                    environ.decay_temperature()
                    # print the distribution
                    dists = environ.get_policy_prob()
                    print(np.concatenate(dists, axis=-1))
                    p_epoch += 1

                if batch_idx_a == len(train2_loader) - 1:
                    batch_enumerator2 = enumerate(train2_loader)

            else:
                raise ValueError('flag %s is not recognized' % flag)

if __name__ == "__main__":
    train()
