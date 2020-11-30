import sys
sys.path.insert(0, '..')

import time
from envs.domainnet_blockdrop_env import BlockDropEnv
from utils.util import print_separator,  should
from copy import deepcopy
from train_val_misc import *


def _train(exp_id, opt, gpu_ids):

    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')
    datasets, samplers, loaders = init_dataloaders(opt, ['train', "train2", 'val'])

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
        environ.load_policy(policy_label)
    else:
        init_state = deepcopy(environ.get_current_state(0))
        if environ.check_exist_policy(policy_label):
            environ.load_policy(policy_label)
        else:
            environ.load(opt['train']['policy_iter'])
            dists = environ.get_policy_prob()
            overall_dist = np.concatenate(dists, axis=-1)
            print(overall_dist)
            environ.sample_policy(opt['test']['hard_sampling'])
            environ.save_policy(policy_label)

        if opt['retrain_from_pl']:
            environ.load(opt['train']['policy_iter'])
        elif opt['train']['imagenet_pretrain']:
            environ.load_imagenet_pretrain()
        else:
            environ.load_snapshot(init_state)

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
    batch_enumerator = enumerate(loaders['train'])
    refer_metrics = set_reference(opt)

    best_value, best_iter = 0, 0
    best_metrics = None
    opt['train']['retrain_total_iters'] = opt['train'].get('retrain_total_iters', opt['train']['total_iters'])
    while current_iter < opt['train']['retrain_total_iters']:
        start_time = time.time()
        environ.train()
        current_iter += 1
        # image-level training
        batch_idx, batch = next(batch_enumerator)
        batch = organize_batch(batch, opt['train']['batch_size'], opt['dataload']['domains'])
        environ.set_inputs(batch)
        environ.optimize_fix_policy()

        if should(current_iter, opt['train']['print_freq']):
            environ.print_loss(current_iter, start_time)

        if should(current_iter, opt['train']['val_freq']):
            environ.eval()
            val_kwarg = {'fix_policy': True}

            val_metrics = eval(environ, loaders['val'], opt['tasks'], opt['train']['batch_size'],
                               opt['dataload']['domains'], val_kwarg)
            print('val metrics: ', val_metrics)
            environ.print_loss(current_iter, start_time, val_metrics)
            environ.save('retrain%03d_policyIter%s_latest' % (exp_id, opt['train']['policy_iter']), current_iter)
            environ.train()
            new_value = rel_performance(refer_metrics, val_metrics)
            print('new value: %.3f' % new_value)

            if new_value > best_value:
                best_value = new_value
                best_metrics = val_metrics
                best_iter = current_iter
                environ.save('retrain%03d_policyIter%s_best' % (exp_id, opt['train']['policy_iter']), current_iter)
            print('best iter: %d, best value: %.3f' % (best_iter, best_value), best_metrics)

        if batch_idx == len(loaders['train']) - 1:
            batch_enumerator = enumerate(loaders['train'])

    if os.path.exists(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json')):
        with open(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json'), 'r') as f:
            results = json.load(f)
        results[str(opt["seed"][exp_id])] = best_metrics
    else:
        results = {str(opt["seed"][exp_id]): best_metrics}

    with open(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json'), 'w+') as f:
        json.dump(results, f)

    return best_metrics, overall_policy


def test(exp_id, opt, gpu_ids):
    # ********************************************************************
    # ******************** Prepare the dataloaders ***********************
    # ********************************************************************
    # load the dataloader
    print_separator('CREATE DATALOADERS')
    datasets, samplers, loaders = init_dataloaders(opt, ['test'])

    # ********************************************************************
    # ********************Create the environment *************************
    # ********************************************************************

    # create the model and the pretrain model
    print_separator('CREATE THE ENVIRONMENT')
    environ = BlockDropEnv(opt['paths']['log_dir'], opt['paths']['checkpoint_dir'], opt['exp_name'],
                            opt['tasks_num_class'], device=gpu_ids[0], is_train=False, opt=opt)

    current_iter = environ.load('retrain%03d_policyIter%s_best' % (exp_id, opt['train']['policy_iter']))

    print('Evaluating the snapshot saved at %d iter' % current_iter)

    policy_label = 'Iter%s_rs%04d' % (opt['train']['policy_iter'], opt['seed'][exp_id])

    if environ.check_exist_policy(policy_label):
        environ.load_policy(policy_label)

    policys = environ.get_current_policy()
    overall_policy = np.concatenate(policys, axis=-1)
    print(overall_policy)

    if torch.cuda.is_available():
        environ.cuda(gpu_ids)

    # ********************************************************************
    # ***************************  Training  *****************************
    # ********************************************************************
    refer_metrics = set_reference(opt)

    environ.eval()
    val_kwarg = {'fix_policy': True}

    val_metrics = eval(environ, loaders['test'], opt['tasks'], opt['train']['batch_size'],
                       opt['dataload']['domains'], val_kwarg)
    new_value = rel_performance(refer_metrics, val_metrics)

    print('iter: %d, value: %.3f' % (current_iter, new_value), val_metrics)
    # np.save(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'seg_per_class.npy'), val_metrics['seg']['per_class'])


def train():
    # ********************************************************************
    # ****************** create folders and print options ****************
    # ********************************************************************
    print_separator('READ YAML')
    opt, gpu_ids, exp_ids = create_folder_print_args()

    best_results = {}
    for exp_id in exp_ids:
        fix_random_seed(opt["seed"][exp_id])
        # fix_random_seed(48)
        _, policy = _train(exp_id, opt, gpu_ids)
        best_metrics = test(exp_id, opt, gpu_ids)
        print(best_metrics)
        best_results[exp_id] = {'sampled_policy': policy.tolist(), 'best_metrics': best_metrics}
    print(best_results)

    best_results_path = os.path.join(opt['paths']['log_dir'], opt['exp_name'], 'best_results.json')

    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            results = json.load(f)
            results.update(best_results)
    else:
        results = best_results

    print(results)
    with open(best_results_path, 'w+') as f:
        json.dump(results, f)

if __name__ == "__main__":
    train()
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

    best_metrics = test(exp_ids[0], opt, gpu_ids)

    if os.path.exists(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json')):
        with open(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json'), 'r') as f:
            results = json.load(f)
        results[str(opt["seed"][exp_ids[0]])] = best_metrics
    else:
        results = {str(opt["seed"][exp_ids[0]]): best_metrics}

    with open(os.path.join(opt['paths']['result_dir'], opt['exp_name'], 'result.json'), 'w+') as f:
        json.dump(results, f)