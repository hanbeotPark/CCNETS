'''-ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Modified by PARK, JunHo in April 10, 2022

[2] Writed by Jinsu, Kim in August 11, 2022

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import os
from collections.abc import Iterable

def create_log_name(log_path, log_details, tag = None):
    i = 0
    while True: 
        log_name = f"{log_path}/{log_details}/{i}" if tag is None else f"{log_path}/{log_details}/{tag}/{i}"
        if os.path.isdir(log_name):
            i+=1
        else:
            log_dir = log_name; break 
    return log_dir

def create_log_details(args):
    log_params = f"ss_{args.step_size}_bs_{args.batch_size}"
    log_nn = f"ln_{args.num_layer}_hs_{args.hidden_size}"
    log_sz = f"os_{args.obs_size}_ls_{args.label_size}_es_{args.explain_size}_sz_{args.seq_len}"
    log_joint = f"rj_{args.reasoner_joint_type}_pj_{args.producer_joint_type}"
    log_type = f"le_{args.loss_type}_{args.error_type}"
    log_rd = f"rd_{args.loss_reduction}_{args.error_reduction}"
    log_details = log_params + "_" + log_nn + "_" + log_sz + "_" + log_joint + "_" + log_type + "_" + log_rd
    return log_details

def log_metric(log, label_type, iters, matric):
    if label_type == "UC":
        precision, recall, f1 = matric
        log.add_scalar("Test/precision", precision, iters)
        log.add_scalar("Test/recall", recall, iters)
        log.add_scalar("Test/f1", f1, iters)
    elif label_type == "C":
        auc = matric
        log.add_scalar("Test/auc", auc, iters)
    elif label_type == "R":
        mse, mae, r2_score = matric
        log.add_scalar("Test/mse", mse, iters)
        log.add_scalar("Test/mae", mae, iters)
        log.add_scalar("Test/r2_score", r2_score, iters)

def log_train(log, iters, losses, errors = None):
    if isinstance(losses, Iterable):
        log.add_scalar("Train/inference_loss", losses[0], iters)
        log.add_scalar("Train/generation_loss", losses[1], iters)
        log.add_scalar("Train/reconstruction_loss", losses[2], iters)
    else:
        log.add_scalar("Train/inference_loss", losses, iters)
        
    if isinstance(errors, Iterable):
        log.add_scalar("Train/explainer_error", errors[0], iters)
        log.add_scalar("Train/reasoner_error", errors[1], iters)
        log.add_scalar("Train/producer_error", errors[2], iters)

        
def log_test(log, iters,losses, errors = None):

    if isinstance(losses, Iterable):
        log.add_scalar("Test/inference_loss", losses[0], iters)
        log.add_scalar("Test/generation_loss", losses[1], iters)
        log.add_scalar("Test/reconstruction_loss", losses[2], iters)
    else:
        log.add_scalar("Train/inference_loss", losses, iters)        
    if isinstance(errors, Iterable):
        log.add_scalar("Test/explainer_error", errors[0], iters)
        log.add_scalar("Test/reasoner_error", errors[1], iters)
        log.add_scalar("Test/producer_error", errors[2], iters)
