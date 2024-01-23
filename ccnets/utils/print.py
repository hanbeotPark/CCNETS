'''-ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Modified by PARK, JunHo in April 10, 2022

[2] Writed by Jinsu, Kim in August 11, 2022

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
from collections.abc import Iterable

def print_iter(epoch, num_epoch, iters, len_dataloader, et):
    print('[%d/%d][%d/%d][Time %.2f]'
        % (epoch, num_epoch, iters, len_dataloader, et))

def print_train(losses, errors):
    print('Inf: %.4f\tGen: %.4f\tRec: %.4f\tE: %.4f\tR: %.4f\tP: %.4f'
        % (losses[0], losses[1], losses[2], errors[0], errors[1], errors[2]))

def print_metric(label_type, matric):
    if label_type == "C":
        auc = matric
        print('auc: %.4f'
            % (auc))
    elif label_type == "UC":
        precision, recall, f1 = matric
        print('precision: %.4f\trecall: %.4f\tf1: %.4f'
            % (precision, recall, f1))
    elif label_type == "R":
        mse, mae, r2_score = matric
        print('mse: %.4f\tmae: %.4f\tr2_score: %.4f'
            % (mse, mae, r2_score))


def print_loss(loss):
    print('Inf: %.4f'% (loss))

def print_lr(optimizers):
    if isinstance(optimizers, Iterable):
        cur_lr = [opt.param_groups[0]['lr'] for opt in optimizers]
        name_opt = [type(opt).__name__ for opt in optimizers]
        
        if len(set(cur_lr)) == 1 and len(set(name_opt)) == 1:
            print('Opt-{0} lr: {1}'.format(name_opt[0], cur_lr[0]))
        else:
            for i, (lr, opt_name) in enumerate(zip(cur_lr, name_opt)):
                print('Opt-{0} lr_{1}: {2}'.format(opt_name, i, lr))
    else:
        opt = optimizers
        cur_lr = opt.param_groups[0]['lr']
        name_opt = type(opt).__name__ 
        print('Opt-{0} lr: {1}'.format(name_opt, cur_lr))
