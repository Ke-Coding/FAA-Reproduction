import copy
import os
import sys
import time
from collections import OrderedDict

import torch
import numpy as np

import json
import argparse
import gorilla
from hyperopt import hp
import ray
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
# from ray.tune.suggest import HyperOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments
from tqdm import tqdm

from FastAutoAugment.archive import remove_deplicates, policy_decoder
from FastAutoAugment.augmentations import augment_list
from FastAutoAugment.common import get_logger, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.metrics import Accumulator
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.train import train_and_eval
from theconf import Config, ConfigArgumentParser
from pystopwatch2 import PyStopwatch

EXEC_ROOT, MODEL_ROOT, MODEL_PATHS, DATASET_ROOT, POLICY_PATH = None, None, None, None, None
timer = PyStopwatch()
logger = get_logger('Fast AutoAugment')


def _check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _get_model_path(dataset, model, config):  # os.path.dirname(os.path.realpath(__file__): abspath of search.py
    # return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s_%s.model' % (dataset, model, tag))
    global MODEL_ROOT
    return os.path.join(MODEL_ROOT, '%s_%s_%s.model' % (dataset, model, config))


def step_w_log(self):
    original = gorilla.get_original_attribute(TrialRunner, 'step')
    
    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


@ray.remote(num_gpus=4, max_calls=1)
def train_model(config, dataroot, augment_replace: str, testset_ratio, fold_idx, model_path=None, only_eval=False):  # TODO: 解耦这里的config相关操作
    """
    pretrain:
        config=copy.deepcopy(copied_conf),      # 最原始的.yaml文件
        dataroot=DATASET_ROOT,
        augment_replace=Config.get()['aug'],    # 不进行augment_replace，augment_replace就是最原始的.yaml文件的aug
        testset_ratio=testset_ratio,            # 默认是0.4
        fold_idx=fold_idx,                      # 0至num_fold
        model_path=MODEL_PATHS[fold_idx],
        only_eval=True                          # pretrain：only_eval
    
    retrain:
        config=copy.deepcopy(copied_conf),
        dataroot=DATASET_ROOT,
        augment_replace=Config.get()['aug']或final_policy_set,    # .yaml中的默认aug，和搜出来的aug
        testset_ratio=0.0,                      # 不需要测试集
        fold_idx=0,
        model_path=default_path[retrain_idx]或augment_path[retrain_idx],
        only_eval=True或False                   # 默认aug：only_eval，搜出aug：不only_eval
    """
    
    Config.get()
    Config.get().conf = config
    Config.get()['aug'] = augment_replace

    result = train_and_eval(
        tag=None,
        dataroot=dataroot,
        testset_ratio=testset_ratio,
        fold_idx=fold_idx,
        reporter=None,
        metric='last',
        model_path=model_path,
        only_eval=only_eval,
        horovod=False,
    )
    return Config.get()['model']['type'], fold_idx, result


def eval_tta(config, exp_config, reporter):
    Config.get()
    Config.get().conf = config
    testset_ratio, fold_idx, model_path = exp_config['testset_ratio'], exp_config['fold_idx'], exp_config['model_path']
    # testset_ratio: 默认0.4
    
    # setup - provided augmentation rules
    Config.get()['aug'] = policy_decoder(exp_config, exp_config['num_policy'], exp_config['num_op'])
    
    # eval
    model = get_model(Config.get()['model'], num_class(Config.get()['dataset']))
    ckpt = torch.load(model_path)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    loaders = []
    for _ in range(exp_config['num_policy']):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(Config.get()['dataset'], Config.get()['batch'], exp_config['dataroot'], testset_ratio=testset_ratio, split_idx=fold_idx)
        loaders.append(iter(validloader))
        del tl, tl2
    
    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()
                
                logits = model(data)
                
                loss = loss_fn(logits, label)
                losses.append(loss.detach().cpu().numpy())
                
                _, logits = logits.topk(1, 1, True, True)
                logits.t_()
                correct = logits.eq(label.view(1, -1).expand_as(logits)).detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, logits, data, label
            
            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()
            
            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': -1 * np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass
    
    del model
    metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], elapsed_time=gpu_secs, done=True)
    return metrics['correct']


def prepare() -> argparse.Namespace:
    parser = ConfigArgumentParser(conflict_handler='resolve')
    # parser.add_argument('--dataroot', type=str, default='~/datasets', help='torchvision data folder')
    parser.add_argument('--start_phase', type=int, default=1)
    parser.add_argument('--num_fold', type=int, default=5)
    parser.add_argument('--num_result_per_fold', type=int, default=10)
    parser.add_argument('--num_op', type=int, default=2)
    parser.add_argument('--num_policy', type=int, default=5)
    parser.add_argument('--num_search', type=int, default=200)
    parser.add_argument('--num_retrain', type=int, default=5)
    parser.add_argument('--testset_ratio', type=float, default=0.4)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str, default='')
    # parser.add_argument('--per_class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    args: argparse.Namespace = parser.parse_args()
    
    add_filehandler(logger, '%s_%s_cv%.1f.log' % (Config.get()['dataset'], Config.get()['model']['type'], args.testset_ratio))
    
    logger.info('args type: %s' % str(type(args)))
    
    global EXEC_ROOT, MODEL_ROOT, MODEL_PATHS, DATASET_ROOT, POLICY_PATH
    
    EXEC_ROOT = os.getcwd()  # fast-autoaugment/experiments/xxx
    logger.info('EXEC_ROOT: %s' % EXEC_ROOT)

    MODEL_ROOT = os.path.join(EXEC_ROOT, 'models')  # fast-autoaugment/experiments/xxx/models
    _check_directory(MODEL_ROOT)
    logger.info('MODEL_ROOT: %s' % MODEL_ROOT)
    
    DATASET_ROOT = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', Config.get()['dataset'].lower()))  # ~/datasets/cifar10
    _check_directory(DATASET_ROOT)
    logger.info('DATASET_ROOT: %s' % DATASET_ROOT)

    POLICY_PATH = os.path.join(MODEL_ROOT, 'final_policy_set.json')
    MODEL_PATHS = [
        _get_model_path(
            dataset=Config.get()['dataset'],
            model=Config.get()['model']['type'],
            config='ratio%.1f_fold%d' % (args.testset_ratio, i)  # without_aug
        )
        for i in range(args.num_fold)
    ]
    print('MODEL_PATHS:', MODEL_PATHS)
    logger.info('MODEL_PATHS: %s' % MODEL_PATHS)
    
    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        Config.get()['optimizer']['decay'] = args.decay
    
    logger.info('configuration...')
    logger.info(json.dumps(Config.get().conf, sort_keys=True, indent=4))
    logger.info('initialize ray...')
    # ray.init(redis_address=args.redis)
    address_info = ray.init(include_webui=True)
    logger.info('ray initialization: address information:')
    logger.info(str(address_info))
    logger.info('start searching augmentation policies, dataset=%s model=%s' % (Config.get()['dataset'], Config.get()['model']['type']))
    
    return args


def pretrain_k_folds(copied_conf, testset_ratio, num_fold) -> None:
    global MODEL_PATHS, DATASET_ROOT
    global logger, timer
    logger.info('----- [Phase 1.] Train with Default Augmentations cv=%d testset ratio=%.2f -----' % (num_fold, testset_ratio))
    timer.start(tag='train_no_aug')
    
    pretrain_reqs = [
        train_model.remote(
            config=copy.deepcopy(copied_conf),
            dataroot=DATASET_ROOT,
            augment_replace=Config.get()['aug'],    # .yaml中的aug，即default的aug
            testset_ratio=testset_ratio,
            fold_idx=fold_idx,
            model_path=MODEL_PATHS[fold_idx],
            only_eval=True
        )
        for fold_idx in range(num_fold)
    ]
    
    tqdm_epoch = tqdm(range(Config.get()['epoch']))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_fold = OrderedDict()
            for fold_idx in range(num_fold):
                try:
                    latest_ckpt = torch.load(MODEL_PATHS[fold_idx])
                    if 'epoch' not in latest_ckpt:
                        epochs_per_fold['fold%d' % (fold_idx + 1)] = Config.get()['epoch']
                        continue
                    epochs_per_fold['fold%d' % (fold_idx + 1)] = latest_ckpt['epoch']
                except Exception as e:
                    continue
            tqdm_epoch.set_postfix(epochs_per_fold)
            if len(epochs_per_fold) == num_fold and min(epochs_per_fold.values()) >= Config.get()['epoch']:
                is_done = True
            if len(epochs_per_fold) == num_fold and min(epochs_per_fold.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break
    
    logger.info('getting results...')
    pretrain_results = ray.get(pretrain_reqs)
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (r_model, r_cv + 1, r_dict['top1_train'], r_dict['top1_valid']))
    logger.info('processed in %.4f secs' % timer.pause('train_no_aug'))


def search_aug_policy(copied_conf, testset_ratio, num_fold, num_result_per_fold, num_policy, num_op, smoke_test, num_search, resume) -> list:
    global MODEL_ROOT, MODEL_PATHS, DATASET_ROOT, POLICY_PATH
    global logger, timer
    logger.info('----- [Phase 2.] Search Test-Time Augmentation Policies -----')
    timer.start(tag='search')
    
    ops = augment_list(False)
    space = {}
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)
    
    final_policy_set = []
    total_computation = 0
    reward_attr = 'top1_valid'  # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for fold_idx in range(num_fold):
            name = "search_%s_%s_fold%d_ratio%.1f" % (Config.get()['dataset'], Config.get()['model']['type'], fold_idx, testset_ratio)
            print(name)
            register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_conf), augs, rpt))  # augs: a dict, just like the 'exp_config'
            algo = HyperOptSearch(space, max_concurrent=4 * 20, reward_attr=reward_attr)
            
            exp_config = {
                name: {
                    'run': name,
                    'num_samples': 4 if smoke_test else num_search,
                    'resources_per_trial': {'gpu': 1},  # TODO: 改成4？
                    'stop': {'training_iteration': num_policy},
                    'config': {
                        'dataroot': DATASET_ROOT,
                        'model_path': MODEL_PATHS[fold_idx],
                        'testset_ratio': testset_ratio,
                        'fold_idx': fold_idx,
                        'num_op': num_op,
                        'num_policy': num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=resume, raise_on_failed_trial=False)
            print()
            results = [x for x in results if x.last_result is not None]     # erase all None items
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)
            
            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']
            
            for result in results[:num_result_per_fold]:
                final_policy = policy_decoder(result.config, num_policy, num_op)
                logger.info('loss=%.12f top1_valid=%.4f %s' % (result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))
                
                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)

    logger.info(json.dumps(final_policy_set))
    logger.info('len(final_policy_set)=%d' % len(final_policy_set))
    logger.info('processed in %.4f secs, gpu hours=%.4f' % (timer.pause('search'), total_computation / 3600.))
    with open(POLICY_PATH, 'w') as fp:
        json.dump(final_policy_set, fp, indent=4)

    return final_policy_set


def retrain(copied_conf, final_policy_set, testset_ratio, num_retrain):
    global DATASET_ROOT
    global logger, timer
    logger.info('----- [Phase 3.] Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----' % (Config.get()['model']['type'], Config.get()['dataset'], Config.get()['aug'], testset_ratio))
    timer.start(tag='train_aug')
    
    default_path = [_get_model_path(Config.get()['dataset'], Config.get()['model']['type'], 'ratio%.1f_default%d' % (testset_ratio, retrain_idx)) for retrain_idx in range(num_retrain)]
    augment_path = [_get_model_path(Config.get()['dataset'], Config.get()['model']['type'], 'ratio%.1f_augment%d' % (testset_ratio, retrain_idx)) for retrain_idx in range(num_retrain)]
    
    retrain_reqs = []
    retrain_reqs.extend([
        train_model.remote(
            config=copy.deepcopy(copied_conf),
            dataroot=DATASET_ROOT,
            augment_replace=Config.get()['aug'],    # .yaml中的aug，即default的aug
            testset_ratio=0.0,
            fold_idx=0,
            model_path=default_path[retrain_idx],
            only_eval=True
        )
        for retrain_idx in range(num_retrain)
    ])
    retrain_reqs.extend([
        train_model.remote(
            config=copy.deepcopy(copied_conf),
            dataroot=DATASET_ROOT,
            augment_replace=final_policy_set,       # 搜出的aug
            testset_ratio=0.0,
            fold_idx=0,
            model_path=augment_path[retrain_idx],
            only_eval=False
        )
        for retrain_idx in range(num_retrain)
    ])
    
    
    tqdm_epoch = tqdm(range(Config.get()['epoch']))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for retrain_idx in range(num_retrain):
                try:
                    if os.path.exists(default_path[retrain_idx]):
                        latest_ckpt = torch.load(default_path[retrain_idx])
                        epochs['default_exp%d' % (retrain_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass
                try:
                    if os.path.exists(augment_path[retrain_idx]):
                        latest_ckpt = torch.load(augment_path[retrain_idx])
                        epochs['augment_exp%d' % (retrain_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass
            
            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == num_retrain * 2 and min(epochs.values()) >= Config.get()['epoch']:
                is_done = True
            if len(epochs) == num_retrain * 2 and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break
    
    logger.info('getting results...')
    final_results = ray.get(retrain_reqs)
    
    for train_mode in ['default', 'augment']:
        avg = 0.
        for _ in range(num_retrain):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
            avg += r_dict['top1_test']
        avg /= num_retrain
        logger.info('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_retrain))
    logger.info('processed in %.4f secs' % timer.pause('train_aug'))
    
    logger.info(timer)


def main():
    patch = gorilla.Patch(TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
    gorilla.apply(patch)

    args = prepare()
    copied_conf = copy.deepcopy(Config.get().conf)

    if args.start_phase <= 1:
        pretrain_k_folds(copied_conf, args.testset_ratio, args.num_fold)

    final_policy_set = []
    if args.start_phase <= 2:
        final_policy_set.extend(search_aug_policy(copied_conf, args.testset_ratio, args.num_fold, args.num_result_per_fold, args.num_policy, args.num_op, args.smoke_test, args.num_search, args.resume))
    else:
        global POLICY_PATH
        with open(POLICY_PATH, 'r') as fp:
            final_policy_set.extend(list(json.load(fp)))
    
    if args.start_phase <= 3:
        retrain(copied_conf, final_policy_set, args.testset_ratio, args.num_retrain)


if __name__ == '__main__':
    main()
