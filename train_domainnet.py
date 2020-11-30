import sys
sys.path.insert(0, '..')

import time

from envs.domainnet_blockdrop_env import BlockDropEnv
from utils.util import print_separator, should
from train_val_misc import *


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    # read the yaml
    print_separator('READ YAML')
    opt, gpu_ids, _ = create_folder_print_args()

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')
    datasets, samplers, loaders = init_dataloaders(opt, ['train', 'train1', "train2", 'val'])

    opt['train']['weight_iter_alternate'] = opt['train'].get('weight_iter_alternate', len(loaders['train1']))
    opt['train']['alpha_iter_alternate'] = opt['train'].get('alpha_iter_alternate', len(loaders['train2']))

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
    elif opt['train']['imagenet_pretrain']:
        environ.load_imagenet_pretrain()

    environ.define_optimizer(False)
    environ.define_scheduler(False)
    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    batch_enumerator = enumerate(loaders['train'])
    batch_enumerator1 = enumerate(loaders['train1'])
    batch_enumerator2 = enumerate(loaders['train2'])
    refer_metrics = set_reference(opt)
    flag = 'update_w'
    environ.fix_alpha()
    environ.free_w(opt['fix_BN'])
    best_value, best_iter = 0, 0
    best_metrics = None
    p_epoch = 0
    flag_warmup = True
    if opt['policy_model'] == 'blockwise':
        if opt['backbone'] == 'ResNet18':
            num_blocks = 8
        elif opt['backbone'] in ['ResNet34', 'ResNet50']:
            num_blocks = 16
        elif opt['backbone'] == 'ResNet101':
            num_blocks = 33
        else:
            raise ValueError('Backbone %s is invalid' % opt['backbone'])
    else:
        raise ValueError('Policy Model %s is not supported' % opt['policy_model'])


    while current_iter < opt['train']['total_iters']:
        start_time = time.time()
        environ.train()
        current_iter += 1
        # warm up
        if current_iter < opt['train']['warm_up_iters']:
            batch_idx, batch = next(batch_enumerator)
            batch = organize_batch(batch, opt['train']['batch_size'], opt['dataload']['domains'])
            environ.set_inputs(batch)
            environ.optimize(is_policy=False, flag='update_w')
            if batch_idx == len(loaders['train']) - 1:
                batch_enumerator = enumerate(loaders['train'])

            if should(current_iter, opt['train']['print_freq']):
                environ.print_loss(current_iter, start_time)

            # validation
            if should(current_iter, opt['train']['val_freq']):
                environ.eval()
                val_kwarg = {"policy": False, "fix_policy": False, "num_train_layers": None, "hard_sampling": False}
                val_metrics = eval(environ, loaders['val'], opt['tasks'], opt['train']['batch_size'],
                               opt['dataload']['domains'], val_kwarg)
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
                batch = organize_batch(batch, opt['train']['batch_size'], opt['dataload']['domains'])
                environ.set_inputs(batch)

                if opt['is_curriculum']:
                    num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                else:
                    num_train_layers = None

                environ.optimize(is_policy=opt['policy'], flag=flag, num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'])

                if should(current_iter, opt['train']['print_freq']):
                    environ.print_loss(current_iter, start_time)

                if should(current_iter_w, opt['train']['weight_iter_alternate']):
                    flag = 'update_alpha'
                    environ.fix_w()
                    environ.free_alpha()
                    # do the validation on the test set
                    environ.eval()
                    print('Evaluating...')
                    val_kwarg = {"policy": opt['policy'], "fix_policy": False, "num_train_layers": num_train_layers,
                                 "hard_sampling": opt['train']['hard_sampling']}

                    val_metrics = eval(environ, loaders['val'], opt['tasks'], opt['train']['batch_size'],
                                       opt['dataload']['domains'], val_kwarg)

                    environ.print_loss(current_iter, start_time, val_metrics)
                    environ.save('latest', current_iter)

                    new_value = rel_performance(refer_metrics, val_metrics)
                    print('new value: %.3f' % new_value)
                    print(val_metrics)
                    if current_iter - opt['train']['warm_up_iters'] >= num_blocks * opt['curriculum_speed'] * \
                            (opt['train']['weight_iter_alternate'] + opt['train']['alpha_iter_alternate']):
                        if new_value > best_value:
                            best_value = new_value
                            best_metrics = val_metrics
                            best_iter = current_iter
                            environ.save('best', current_iter)
                    print('best iter: %d, best_value: %.3f' % (best_iter, best_value), best_metrics)
                    environ.train()

                if batch_idx_w == len(loaders['train1']) - 1:
                    batch_enumerator1 = enumerate(loaders['train1'])

            # update the policy network
            elif flag == 'update_alpha':
                current_iter_a += 1
                batch_idx_a, batch = next(batch_enumerator2)
                batch = organize_batch(batch, opt['train']['batch_size'], opt['dataload']['domains'])
                environ.set_inputs(batch)
                if opt['is_curriculum']:
                    num_train_layers = p_epoch // opt['curriculum_speed'] + 1
                else:
                    num_train_layers = None

                environ.optimize(is_policy=opt['policy'], flag=flag, num_train_layers=num_train_layers,
                                 hard_sampling=opt['train']['hard_sampling'])

                if should(current_iter, opt['train']['print_freq']):
                    environ.print_loss(current_iter, start_time)

                if should(current_iter_a, opt['train']['alpha_iter_alternate']):
                    flag = 'update_w'
                    environ.fix_alpha()
                    environ.free_w(opt['fix_BN'])
                    environ.decay_temperature()
                    # print the distribution
                    dists = environ.get_policy_prob()
                    print(np.concatenate(dists, axis=-1))
                    p_epoch += 1

                if batch_idx_a == len(loaders['train2']) - 1:
                    batch_enumerator2 = enumerate(loaders['train2'])

            else:
                raise ValueError('flag %s is not recognized' % flag)


if __name__ == "__main__":
    train()