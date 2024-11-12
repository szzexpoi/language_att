import math
import sys
from typing import Iterable, Optional
import misc as misc
import lr_sched as lr_sched
import torch
from loss import NSS, CC, KLD, attention_entropy
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
import numpy as np
import gc

def single_epoch_training(iteration: int, model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, subj_prefix='', args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.entropy:
        metric_logger.add_meter('entropy', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    cur_alpha = max(((args.epochs-epoch)/args.epochs), 0.4)*args.alpha
    # cur_alpha = args.alpha

    for data_iter_step, (hidden_state, base_attention, patch_attention, valid_len, saliency_map, fixation_map, img, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        hidden_state = hidden_state.to(device, non_blocking=True)        
        base_attention = base_attention.to(device, non_blocking=True)
        patch_attention = patch_attention.to(device, non_blocking=True)
        saliency_map = saliency_map.to(device, non_blocking=True)
        fixation_map = fixation_map.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)

        if not args.entropy:
            if not args.is_reweight:
                pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len)
            else:
                pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len, img)

            loss = (-NSS(pred_saliency_map, fixation_map) - 2*CC(pred_saliency_map, saliency_map)
                    + 10*KLD(pred_saliency_map, saliency_map))
        else:
            if not args.is_reweight:
                pred_saliency_map, lang_att = model(hidden_state, patch_attention, base_attention, valid_len, get_lang_att=True)
            else:
                pred_saliency_map, lang_att, _, _ = model(hidden_state, patch_attention, base_attention, 
                                                valid_len, img, get_lang_att=True)

            entropy = attention_entropy(lang_att, valid_len)
            loss = (-NSS(pred_saliency_map, fixation_map) - 2*CC(pred_saliency_map, saliency_map)
                    + 10*KLD(pred_saliency_map, saliency_map) + cur_alpha*entropy)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.entropy:
            entropy_value = entropy.item()
            metric_logger.update(entropy=entropy_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % 25 == 0:
            log_writer.add_scalar(subj_prefix+'loss:', loss_value_reduce, iteration)
            log_writer.add_scalar(subj_prefix+'lr:', max_lr, iteration)

        iteration += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # flush gpu memory
    del hidden_state
    del base_attention
    del patch_attention
    del saliency_map
    del fixation_map
    del img
    gc.collect()
    torch.cuda.empty_cache()

    return iteration


@torch.no_grad()
def evaluation(cur_epoch, iteration, model, data_loader, device, log_writer, subj_prefix='', args=None):
    # switch to evaluation mode
    model.eval()
    record_nss = 0
    record_cc = 0
    record_kld = 0
    total = 0

    for data_iter_step, (hidden_state, base_attention, patch_attention, valid_len, saliency_map, fixation_map, img, _) in enumerate(data_loader):
        hidden_state = hidden_state.to(device, non_blocking=True)        
        base_attention = base_attention.to(device, non_blocking=True)
        patch_attention = patch_attention.to(device, non_blocking=True)
        saliency_map = saliency_map.to(device, non_blocking=True)
        fixation_map = fixation_map.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)

        if not args.is_reweight:
            pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len)
        else:
            pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len, img)

        record_nss += NSS(pred_saliency_map, fixation_map).data.cpu().numpy()*len(pred_saliency_map)
        record_kld += KLD(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
        record_cc += CC(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
        total += len(pred_saliency_map)
            
    record_nss /= total
    record_cc /= total
    record_kld /= total

    log_writer.add_scalar(subj_prefix+'Validation NSS', record_nss.item(), iteration)
    log_writer.add_scalar(subj_prefix+'Validation CC', record_cc.item(), iteration)
    log_writer.add_scalar(subj_prefix+'Validation KLD', record_kld.item(), iteration)

    print("Validation NSS:", record_nss.item())
    print("Validation CC:", record_cc.item())
    print("Validation KLD:", record_kld.item())

    # flush gpu memory
    del hidden_state
    del base_attention
    del patch_attention
    del saliency_map
    del fixation_map
    del img
    gc.collect()
    torch.cuda.empty_cache()

    return record_nss.item()

def single_epoch_training_image(iteration: int, model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (image, saliency_map, fixation_map, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        image = image.to(device, non_blocking=True)        
        saliency_map = saliency_map.to(device, non_blocking=True)
        fixation_map = fixation_map.to(device, non_blocking=True)
        pred_saliency_map = model(image).squeeze(1)

        loss = (-NSS(pred_saliency_map, fixation_map) - 2*CC(pred_saliency_map, saliency_map)
                 + 10*KLD(pred_saliency_map, saliency_map))
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % 25 == 0:
            log_writer.add_scalar('loss:', loss_value_reduce, iteration)
            log_writer.add_scalar('lr:', max_lr, iteration)

        iteration += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return iteration


@torch.no_grad()
def evaluation_image(cur_epoch, iteration, model, data_loader, device, log_writer, args):
    # switch to evaluation mode
    model.eval()
    record_nss = 0
    record_cc = 0
    record_kld = 0
    total = 0

    for data_iter_step, (image, saliency_map, fixation_map, _) in enumerate(data_loader):
        image = image.to(device, non_blocking=True)        
        saliency_map = saliency_map.to(device, non_blocking=True)
        fixation_map = fixation_map.to(device, non_blocking=True)
        pred_saliency_map = model(image).squeeze(1)

        record_nss += NSS(pred_saliency_map, fixation_map).data.cpu().numpy()*len(pred_saliency_map)
        record_kld += KLD(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
        record_cc += CC(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
        total += len(pred_saliency_map)
            
    record_nss /= total
    record_cc /= total
    record_kld /= total

    log_writer.add_scalar('Validation NSS', record_nss.item(), iteration)
    log_writer.add_scalar('Validation CC', record_cc.item(), iteration)
    log_writer.add_scalar('Validation KLD', record_kld.item(), iteration)

    print("Validation NSS:", record_nss.item())
    print("Validation CC:", record_cc.item())
    print("Validation KLD:", record_kld.item())

    return record_nss.item()
