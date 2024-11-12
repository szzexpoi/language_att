import sys
import argparse
from random import shuffle
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler
from dataloader import SALICON, MIT, OSIE_ASD, SALICON_Image, MIT_Image, Emotion
from model import Blind_Transformer_MS, Blind_Transformer_MS_Reweight
from trainer import single_epoch_training, evaluation, single_epoch_training_image, evaluation_image
import pickle
from glob import glob
import matplotlib
from matplotlib import pyplot as plt
import cv2
import json
import re
import Levenshtein
from loss import NSS, CC, KLD
from DINet import DINet
from transalnet import TranSalNet
from copy import deepcopy
from drawing_util import draw_bounding_box_around_important_regions, draw_optimal_bounding_box, overlay_heatmap
from evaluation import cal_sim_score, cal_auc_score
from matplotlib.gridspec import GridSpec

def get_args_parser():
    parser = argparse.ArgumentParser('Blind model for visual saliency modeling', 
                                     add_help=False)
    parser.add_argument('--mode', default='train', type=str,
                        help='running mode')
    parser.add_argument('--batch_size', default=24, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=60, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--anno_path', default=None, type=str,
                        help='path to annotations, i.e., salicon root directory')
    parser.add_argument('--language_path', default=None, type=str,
                        help='path to pre-extracted language data')

    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log and model checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
            
    # model setting
    parser.add_argument('--width', default=320, type=int,
                        help='width of saliency maps')
    parser.add_argument('--height', default=240, type=int,
                        help='height of saliency maps')
    parser.add_argument('--num_head', default=8, type=int,
                        help='number of attention heads in the language decoder')
    parser.add_argument('--depth', default=4, type=int,
                        help='depth of the language decoder')
    parser.add_argument('--input_dim', default=4096, type=int,
                        help='Feature dimension for pretrained LLM')        
    parser.add_argument('--input_head', default=32, type=int,
                        help='Number of attention heads for pretrained LLM')        
    parser.add_argument('--weights', default=None, type=str,
                        help='Pretrained weights file')        
    parser.add_argument('--save_dir', default=None, type=str,
                        help='Directory for saving the visualization')        
    parser.add_argument('--raw_img_dir', default=None, type=str,
                        help='Directory for storing the raw image data')        
    parser.add_argument('--asd_label', default=None, type=str,
                        help='Autism label, asd or ctrl')   
    parser.add_argument('--is_visual', action='store_true',
                        help='Running visual model or not')
    parser.add_argument('--percentage', default=None, type=int,
                        help='Percentage of training data')     
    parser.add_argument('--temporal_step', default=1, type=int,
                        help='Number of temporal steps for studying attention dynamics in ASD data, 1 means aggregated data')      
    parser.add_argument('--temporal_id', default=None, type=int,
                        help='Experimenting with data from a specific time period for ASD')      
    parser.add_argument('--joint_training', action='store_true',
                        help='jointly training with language and visual components or not')      
    parser.add_argument('--entropy', action='store_true',
                        help='Training with attention entropy or njot')      
    parser.add_argument('--alpha', default=5e-1, type=float,
                        help='weights for entropy loss')    
    parser.add_argument('--is_reweight', action='store_true',
                        help='training with visual model or not')   
    parser.add_argument('--reweight_module', default='dinet', type=str,
                        help='visual network for reweighting')    
    parser.add_argument('--max_len', default=100, type=int,
                        help='maximum token length for the input description')      

    return parser.parse_args()

# trial-based experiment
def main(args):
    # tensorboard initialization
    log_writer = SummaryWriter(log_dir=args.log_dir)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    if not args.is_visual:
        if args.asd_label is not None:
            train_dataset = OSIE_ASD(args.language_path, args.anno_path, args.height, args.width, 
                                     'train', asd_label=args.asd_label, temporal_step=args.temporal_step, temporal_id=args.temporal_id)
            val_dataset = OSIE_ASD(args.language_path, args.anno_path, args.height, args.width, 
                                   'val', asd_label=args.asd_label, temporal_step=args.temporal_step, temporal_id=args.temporal_id)
        else:
            train_dataset = SALICON(args.language_path, args.anno_path, args.height, args.width, 'train', percentage=args.percentage, max_len=args.max_len)
            val_dataset = SALICON(args.language_path, args.anno_path, args.height, args.width, 'val', max_len=args.max_len)
            # train_dataset = Emotion(args.language_path, args.anno_path, args.height, args.width, 'train')
            # val_dataset = Emotion(args.language_path, args.anno_path, args.height, args.width, 'val')

    else:
            train_dataset = SALICON_Image(args.anno_path, args.language_path, args.height, args.width, 'train', percentage=args.percentage)
            val_dataset = SALICON_Image(args.anno_path, args.language_path, args.height, args.width, 'val')

    # code for distributed training, borrowed from MAE repo
    if args.distributed:  # args.distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, 
        # sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        # sampler=sampler_val,
        batch_size=4,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )    

    # model configuration
    if not args.is_reweight:
        model = Blind_Transformer_MS(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len, temporal_step=args.temporal_step)
    else:
        model = Blind_Transformer_MS_Reweight(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len, 
                                            joint_training=args.joint_training, reweight_module=args.reweight_module)
    model.to(device)

    # load from SALICON pretrained model for autism data fine-tuning
    if args.asd_label is not None or args.weights is not None:
        model.load_state_dict(torch.load(args.weights)['model'], strict=False)

        if args.temporal_step!=1:
            model.set_temporal_layer()

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimization parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_without_ddp.parameters()), 
                                lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    loss_scaler = NativeScaler()

    # main loop for training
    best_score = 0
    iteration = 0
    for epoch in range(args.epochs):
        if not args.is_visual:
            iteration = single_epoch_training(iteration, model, data_loader_train, 
                                            optimizer, device, epoch, 
                                            loss_scaler, log_writer, '', args)
            
            cur_score = evaluation(epoch, iteration, model, data_loader_val, device, log_writer, '', args)
        else:
            iteration = single_epoch_training_image(iteration, model, data_loader_train, 
                                            optimizer, device, epoch, 
                                            loss_scaler, log_writer, args)
            
            cur_score = evaluation_image(epoch, iteration, model, data_loader_val, device, log_writer, args)

        if cur_score>best_score:
            best_score = cur_score
            torch.save({
                        'epoch': epoch,
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, os.path.join(args.log_dir, 'model_best.pt'))


def validation(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    if not args.is_visual:
        val_dataset = MIT(args.language_path, args.anno_path, args.height, args.width, 'test_reshaped')
        # val_dataset = SALICON(args.language_path, args.anno_path, args.height, args.width, 'val')
        # val_dataset = Emotion(args.language_path, args.anno_path, args.height, args.width, 'all')

    else:
        val_dataset = MIT_Image(args.anno_path, args.height, args.width)
        # val_dataset = SALICON_Image(args.anno_path, args.language_path, args.height, args.width, 'val')


    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        # sampler=sampler_val,
        batch_size=10,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )

    if not args.is_reweight:
        model = Blind_Transformer_MS(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len)
    else:
        model = Blind_Transformer_MS_Reweight(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len,
                                            joint_training=args.joint_training, reweight_module=args.reweight_module)

    model.load_state_dict(torch.load(args.weights)['model'])
    model.to(device)  
    model.eval()    

    record_nss = 0
    record_cc = 0
    record_kld = 0
    record_sim = 0
    record_auc = 0
    total = 0

    with torch.no_grad():
        if not args.is_visual:
            for data_iter_step, (hidden_state, base_attention, patch_attention, valid_len, saliency_map, fixation_map, img, _) in enumerate(data_loader_val):
                hidden_state = hidden_state.to(device, non_blocking=True)        
                base_attention = base_attention.to(device, non_blocking=True)
                patch_attention = patch_attention.to(device, non_blocking=True)
                saliency_map = saliency_map.to(device, non_blocking=True)
                fixation_map = fixation_map.to(device, non_blocking=True)
                img = img.to(device, non_blocking=True)

                if args.is_reweight:
                    pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len)
                else:
                    pred_saliency_map = model(hidden_state, patch_attention, base_attention, valid_len, img)

                record_nss += NSS(pred_saliency_map, fixation_map).data.cpu().numpy()*len(pred_saliency_map)
                record_kld += KLD(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
                record_cc += CC(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
                total += len(pred_saliency_map)

                pred_saliency_map = pred_saliency_map.data.cpu().numpy()
                fixation_map = fixation_map.data.cpu().numpy()
                saliency_map = saliency_map.data.cpu().numpy()
                for i in range(len(pred_saliency_map)):
                    record_sim += cal_sim_score(pred_saliency_map[i], saliency_map[i])
                    tmp_auc = cal_auc_score(pred_saliency_map[i], fixation_map[i])
                    tmp_auc = tmp_auc if not np.isnan(tmp_auc) else 0
                    record_auc += tmp_auc

        else:
            for data_iter_step, (image, saliency_map, fixation_map, _) in enumerate(data_loader_val):
                image = image.to(device, non_blocking=True)        
                saliency_map = saliency_map.to(device, non_blocking=True)
                fixation_map = fixation_map.to(device, non_blocking=True)

                pred_saliency_map = model(image)

                record_nss += NSS(pred_saliency_map, fixation_map).data.cpu().numpy()*len(pred_saliency_map)
                record_kld += KLD(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
                record_cc += CC(pred_saliency_map, saliency_map).data.cpu().numpy()*len(pred_saliency_map)
                total += len(pred_saliency_map)
    
    record_nss /= total
    record_cc /= total
    record_kld /= total
    record_sim /= total
    record_auc /= total

    print('NSS: ', record_nss)
    print('CC: ', record_cc)
    print('KLD: ', record_kld)
    print('SIM: ', record_sim)
    print('AUC: ', record_auc)

# Function to plot the sentence with varying transparency
def plot_sentence_with_weights(sentence_weight, ax, fig):
    x_pos, y_pos = 0, 1  # Start at the top-left of the axis (normalized coordinates)
    max_width = 1  # Max width for text line wrapping in normalized coordinates
    line_height = 0.3  # Adjust line height for better spacing

    renderer = fig.canvas.get_renderer()  # Renderer for text size calculations

    for idx, (word, weight) in enumerate(sentence_weight):
        # If word starts with '-' (like '-example'), connect it with the previous word.
        if word.startswith('-') and idx > 0:
            x_pos -= prev_word_width + 0.01  # Shift back to connect it seamlessly

        # Plot the word with transparency based on weight
        text = ax.text(
            x_pos, y_pos, word, fontsize=15, alpha=np.minimum(weight, 1),
            ha='left', va='bottom', transform=ax.transAxes
        )

        # Get the bounding box width of the word in normalized coordinates
        text_bbox = text.get_window_extent(renderer=renderer)
        word_width = text_bbox.width / fig.canvas.get_width_height()[0]

        # If the word ends with punctuation, avoid extra space after it
        if word[-1] in [',', '.', ':', ';', '?', '!']:
            space = 0.005  # Minimal space after punctuation
        else:
            space = 0.01  # Normal space between words

        # Check if the word fits within the max width, otherwise move to the next line
        if x_pos + word_width > max_width:
            x_pos = 0  # Reset x position to start of the line
            y_pos -= line_height  # Move down by line height

        # Set the new position and update the x_pos for the next word
        text.set_position((x_pos, y_pos))
        x_pos += word_width + space  # Add space after the word

        # Store the current word width for the next iteration (in case of '-')
        prev_word_width = word_width

    # Return the total height of the text block for layout adjustment
    return 1 - y_pos  # Distance from top to the last line

def visualization(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    if args.asd_label is not None:
        val_dataset = OSIE_ASD(args.language_path, args.anno_path, args.height, args.width, 
                'all', asd_label=args.asd_label, temporal_step=args.temporal_step, temporal_id=args.temporal_id)
    else:
        val_dataset = SALICON(args.language_path, args.anno_path, args.height, args.width, 'val')
        # val_dataset = MIT(args.language_path, args.anno_path, args.height, args.width, 'test_reshaped')


    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        # sampler=sampler_val,
        batch_size=4,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )

    if not args.is_reweight:
        model = Blind_Transformer_MS(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len, temporal_step=args.temporal_step)
    else:
        model = Blind_Transformer_MS_Reweight(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len)

    if args.temporal_step!=1:
        model.set_temporal_layer()

    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'])
    model.to(device)  
    model.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # for salicon
    lang_info = json.load(open(os.path.join(args.language_path, 'caption.json')))['val']

    # for MIT
    # lang_info = json.load(open(os.path.join(args.language_path, 'caption.json')))['test_reshaped']
    
    # for OSIE-ASD
    # lang_info = json.load(open(os.path.join(args.language_path, 'caption.json')))


    if args.is_reweight:
        os.makedirs(os.path.join(args.save_dir, 'language_bbox'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'visual_bbox'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'final_pred'), exist_ok=True)

    offset = 10
    save_attention = dict()
    with torch.no_grad():
        for data_iter_step, (hidden_state, base_attention, patch_attention, valid_len, saliency_map, fixation_map, image, img_id) in enumerate(data_loader_val):
            hidden_state = hidden_state.to(device, non_blocking=True)        
            base_attention = base_attention.to(device, non_blocking=True)
            patch_attention = patch_attention.to(device, non_blocking=True)
            saliency_map = saliency_map.to(device, non_blocking=True)
            fixation_map = fixation_map.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)

            if not args.is_reweight:
                pred_saliency_map, lang_att = model(hidden_state, patch_attention, base_attention, valid_len, get_lang_att=True)
            else:
                pred_saliency_map, lang_att, ori_map, reweight_map = model(hidden_state, patch_attention, base_attention, valid_len, image, get_lang_att=True)

            pred_saliency_map = pred_saliency_map.data.cpu().numpy()
            saliency_map = saliency_map.data.cpu().numpy()
            lang_att = lang_att.data.cpu().numpy()

            ori_map = ori_map.data.cpu().numpy()
            reweight_map = reweight_map.data.cpu().numpy()

            # for static attention visualization
            for i in range(len(pred_saliency_map)):
                cur_img_id = img_id[i]
                normalized_att = deepcopy(lang_att[i][:valid_len[i]])
                normalized_att = (normalized_att-np.min(normalized_att))/(np.max(normalized_att)-np.min(normalized_att))
                # save_attention[cur_img_id] = normalized_att
                if cur_img_id not in lang_info:
                    continue
                
                # for SALICON
                cur_img = cur_img_id+'.jpg'
                cur_img = cv2.imread(os.path.join(args.raw_img_dir, 'val', cur_img))

                # # for MIT
                # cur_img = cur_img_id+'.jpeg'
                # cur_img = cv2.imread(os.path.join(args.raw_img_dir, 'test_reshaped', cur_img))

                # # for OSIE-ASD
                # cur_img = cur_img_id+'.jpg'
                # cur_img = cv2.imread(os.path.join(args.raw_img_dir, cur_img))

                cur_img = cv2.resize(cur_img, (args.width, args.height))
                gt_vis = overlay_heatmap(cur_img, saliency_map[i])
                tmp_pred = pred_saliency_map[i]/pred_saliency_map[i].sum()
                pred_vis = overlay_heatmap(cur_img, tmp_pred/tmp_pred.max())

                # for standard analysis
                if not args.is_reweight:
                    final_vis = np.ones([args.height, 3*args.width+2*offset, 3])*255
                    final_vis[:, :args.width] = cur_img
                    final_vis[:, args.width+offset:2*args.width+offset] = gt_vis
                    final_vis[:, 2*args.width+2*offset:] = pred_vis
                else:
                    # for residual model analysis
                    ori_vis = overlay_heatmap(cur_img, ori_map[i]/ori_map[i].max())
                    reweight_vis = overlay_heatmap(cur_img, reweight_map[i]/reweight_map[i].max())
                    final_vis = np.zeros([args.height, 4*args.width+3*offset, 3])
                    final_vis[:, :args.width] = gt_vis
                    final_vis[:, args.width+offset:2*args.width+offset:] = ori_vis
                    final_vis[:, 2*args.width+2*offset:3*args.width+2*offset] = reweight_vis
                    final_vis[:, 3*args.width+3*offset:] = pred_vis

                    # saving the bounding box figures for LLM analysis
                    draw_optimal_bounding_box(os.path.join(args.raw_img_dir, 'val', cur_img_id+'.jpg'),
                                                               ori_map[i]/ori_map[i].sum(),
                                                               os.path.join(args.save_dir, 'language_bbox_final', cur_img_id+'.jpg'),
                                                               (0, 0, 255),
                                                               thickness=3,
                                                                max_area_ratio=0.20)

                    draw_optimal_bounding_box(os.path.join(args.raw_img_dir, 'val', cur_img_id+'.jpg'),
                                                               reweight_map[i]/reweight_map[i].sum(),
                                                               os.path.join(args.save_dir, 'visual_bbox_final', cur_img_id+'.jpg'),
                                                               (0, 0, 255),
                                                               thickness=3,
                                                                max_area_ratio=0.20)


                # plot the semantic weights and saliency maps together
                raw_token = lang_info[cur_img_id]['raw_token']
                merged_sentence = lang_info[cur_img_id]['merged_token']
                token_mapping, merged_tokens, word_mapping = token_matching(raw_token, merged_sentence)

                sentence_weight = [[merged_tokens[k], 0] for k in range(len(merged_tokens))]
                for k in token_mapping:
                    sentence_weight[token_mapping[k]][1] += float(lang_att[i][k])

                min_val = np.min([sentence_weight[cur][1] for cur in range(len(sentence_weight))])
                max_val = np.max([sentence_weight[cur][1] for cur in range(len(sentence_weight))])
                for j in range(len(sentence_weight)):
                    sentence_weight[j][1] = (sentence_weight[j][1]-min_val)/(max_val-min_val)
                save_attention[cur_img_id] = deepcopy(sentence_weight)

                # normalizing the weights for word based on ranking
                weight_ranking = dict()
                all_weight = [sentence_weight[j][1] for j in range(len(sentence_weight))]
                tmp_ranking = np.argsort(all_weight) # acsending
                for j in range(len(tmp_ranking)):
                    weight_ranking[tmp_ranking[j]] = j
                
                min_val, max_val = 0.2, 1
                for j in range(len(sentence_weight)):
                    if weight_ranking[j] > 10: 
                        sentence_weight[j][1] = min_val + (weight_ranking[j]-10)* (max_val - min_val) / (len(tmp_ranking)-11)
                    else:
                        sentence_weight[j][1] = min_val

                plt.close('all')
                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 4), constrained_layout=True)

                ax1.imshow(final_vis[:, :, ::-1]/255)
                ax1.axis('off')  # Hide the axes
                ax2.axis('off')  # Hide the axes
                plot_sentence_with_weights(sentence_weight, ax2, fig)
                plt.savefig(os.path.join(args.save_dir, cur_img_id+'.jpg'), bbox_inches='tight', pad_inches=0)
                

from unidecode import unidecode
def normalize_token(token):
    return unidecode(token)

def split_merged_sentence(merged_sentence):
    # Split the merged sentence into tokens while keeping special characters and punctuation
    return re.findall(r'\w+|\'\w+|[^\w\s-]|[\w-]+', merged_sentence)

def token_matching(raw_tokens, merged_sentence):
    raw_counter = 0
    merged_counter = 0
    mapping = {}
    word_mapping = {}

    # Normalize the merged sentence and split it into tokens
    normalized_sentence = unidecode(merged_sentence)
    merged_tokens = split_merged_sentence(normalized_sentence)

    # Normalize raw tokens
    raw_tokens = [normalize_token(token) for token in raw_tokens]

    # Iterate over raw tokens and merged tokens to create the mapping
    while raw_counter < len(raw_tokens) and merged_counter < len(merged_tokens):
        raw_token = raw_tokens[raw_counter]
        merged_token = merged_tokens[merged_counter]

        # Handle cases where multiple raw tokens map to a single merged token
        tmp = ''
        start_counter = raw_counter
        while raw_counter < len(raw_tokens) and tmp.replace(' ', '') != merged_token.replace(' ', ''):
            tmp += raw_tokens[raw_counter]
            tmp = tmp.strip()
            raw_counter += 1

        if tmp.replace(' ', '') == merged_token.replace(' ', ''):
            for i in range(start_counter, raw_counter):
                mapping[i] = merged_counter
                word_mapping[i] = merged_tokens[merged_counter]

            merged_counter += 1
        else:
            merged_counter += 1

    return mapping, merged_tokens, word_mapping

def check_token(token_mapping, raw_tokens, merged_sentence):
    # check token_mapping
    check = [''] * (max(token_mapping.values()) + 1)
    for k, v in token_mapping.items():
        check[v] += raw_tokens[k]

    # Join the tokens and normalize spaces around punctuation
    check_sentence = ' '.join(check)
    check_sentence = re.sub(r'\s([,.!?])', r'\1', check_sentence)  # Remove space before punctuation
    check_sentence = re.sub(r'\s+', ' ', check_sentence).strip()  # Normalize multiple spaces to single

    # Normalize spaces in the merged_sentence for comparison
    merged_sentence_normalized = re.sub(r'\s([,.!?])', r'\1', merged_sentence)
    merged_sentence_normalized = re.sub(r'\s+', ' ', merged_sentence_normalized).strip()
    check_sentence = check_sentence.replace(" 's", "'s")

    return Levenshtein.distance(check_sentence, merged_sentence_normalized)<=3, check

def language_parsing(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    if args.asd_label is not None:
        val_dataset = OSIE_ASD(args.language_path, args.anno_path, args.height, args.width, 'val', asd_label=args.asd_label)
        # load language information
        lang_info = json.load(open(os.path.join(args.language_path, 'caption.json')))
    else:
        val_dataset = SALICON(args.language_path, args.anno_path, args.height, args.width, 'val')
        # val_dataset = Emotion(args.language_path, args.anno_path, args.height, args.width, 'train')

        # load language information
        lang_info = json.load(open(os.path.join(args.language_path, 'caption.json')))['val'] # SALICON
        # lang_info = json.load(open(os.path.join(args.language_path, 'caption.json'))) # EMOd

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, 
        # sampler=sampler_val,
        batch_size=4,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )

    model = Blind_Transformer_MS(args.input_dim, args.input_head, args.num_head, args.depth, max_len=args.max_len)

    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'])
    model.to(device)  
    model.eval()

    record_result = dict()
    record_interaction = dict()
    record_coocurrence = dict()
    record_coocurrence_ratio = dict()
    counter = 0
    with torch.no_grad():
        for data_iter_step, (hidden_state, base_attention, patch_attention, valid_len, saliency_map, fixation_map, image, img_id) in enumerate(data_loader_val):
            hidden_state = hidden_state.to(device, non_blocking=True)        
            base_attention = base_attention.to(device, non_blocking=True)
            patch_attention = patch_attention.to(device, non_blocking=True)

            _, lang_att = model(hidden_state, patch_attention, base_attention, valid_len, get_lang_att=True)

            lang_att = lang_att.data.cpu().numpy()

            for i in range(len(lang_att)):
                if img_id[i] not in lang_info:
                    continue
                raw_token = lang_info[img_id[i]]['raw_token']
                merged_sentence = lang_info[img_id[i]]['merged_token']
                token_mapping, merged_tokens, word_mapping = token_matching(raw_token, merged_sentence)

                valid_mapping, check_sentence = check_token(token_mapping, raw_token, merged_sentence)
                
                if not valid_mapping:
                    counter += 1  
                    continue

                merged_attention = {}
                assert len(raw_token)==valid_len[i], 'Inconsistent shape detected'
                
                for k in word_mapping:
                    if not word_mapping[k] in merged_attention:
                        merged_attention[word_mapping[k]] = 0
                    merged_attention[word_mapping[k]] += lang_att[i][k]
                merged_attention['EOS'] = lang_att[i][valid_len[i]-1]

                # # convert the raw attention weights into attention rank (smaller rank means larger attention weights)
                # sorted_weight = {k: v for k, v in sorted(merged_attention.items(), 
                #                                          key=lambda item: item[1], 
                #                                          reverse=True)}
                # for rank_idx, k in enumerate(sorted_weight):
                #     merged_attention[k] = rank_idx

                for cur_token in merged_attention:
                    if cur_token not in record_result:
                        record_result[cur_token] = dict()
                        record_result[cur_token]['weight'] = []
                        record_result[cur_token]['frequency'] = 0

                    record_result[cur_token]['weight'].append(float(merged_attention[cur_token]))
                    record_result[cur_token]['frequency'] += 1
                
                # recording the interactions between each pair of words
                # for each source word, record the average ranking for each of its target pair
                for source_token in merged_attention:
                    if source_token in ['.', ',', ';', '?', '"', 'a', 'A', 'the', 'The', 'with', 'and']:
                        continue
                    if source_token not in record_interaction:
                        record_interaction[source_token] = dict()

                    if source_token not in record_coocurrence:
                        record_coocurrence[source_token] = dict()
                        record_coocurrence_ratio[source_token] = dict()
                    
                    if merged_attention[source_token] not in record_coocurrence[source_token]:
                        record_coocurrence[source_token][merged_attention[source_token]] = dict()
                        record_coocurrence_ratio[source_token][merged_attention[source_token]] = []

                    for target_token in merged_attention:
                        if target_token in ['.', ',', ';', '?', '"', 'a', 'A', 'the', 'The', 'with', 'and']:
                            continue
                        if target_token == source_token:
                            continue
                        if target_token not in record_interaction[source_token]:
                            record_interaction[source_token][target_token] = []
                        if target_token not in record_coocurrence[source_token][merged_attention[source_token]]:
                            record_coocurrence[source_token][merged_attention[source_token]][target_token] = 0
                        
                        record_interaction[source_token][target_token].append(float(merged_attention[target_token]))
                        record_coocurrence[source_token][merged_attention[source_token]][target_token] +=1

                    record_coocurrence_ratio[source_token][merged_attention[source_token]].append(
                                [cur for cur in merged_attention if merged_attention[cur]<=10 and cur!=source_token])

                # EOS token
                if 'EOS' not in record_result:
                    record_result['EOS'] = dict()
                    record_result['EOS']['weight'] = []
                    record_result['EOS']['frequency'] = 0    
                record_result['EOS']['weight'].append(float(merged_attention['EOS']))
                record_result['EOS']['frequency'] += 1      
                                        
    print('Invalid Counter: %d' %counter)
    with open(os.path.join(args.save_dir), 'w') as f:
        json.dump(record_result, f)


def language_statistics(args):
    data = json.load(open(args.save_dir))
    
    freq_threshold = 40 # 40 for SALICON, 1 for OSIE, 10 for emotion
    # first select words based on their frequencies
    valid_word = dict()
    for k in data:
        if data[k]['frequency']>freq_threshold and k not in ['.', ',', ';', '?', '"', 'a', 'and', 'the', 'with', 'is', 'The']:
            valid_word[k] = 1
    print('Number of valid words:', len(valid_word))
    # extract the attention weights for the sorted word
    word_att = dict()
    for idx, k in enumerate(valid_word):
        word_att[k] = np.mean(data[k]['weight'])
    
    # # normalize the attention weights
    # max_val, min_val = np.max(list(word_att.values())), np.min(list(word_att.values()))
    # for k in word_att:
    #     word_att[k] = (word_att[k]-min_val)/(max_val-min_val)

    # word_att = {k: v for k, v in sorted(word_att.items(), 
    #             key=lambda item: item[1], reverse=True)}        
    
    # for ranking-based analysis
    word_att = {k: v for k, v in sorted(word_att.items(), 
                key=lambda item: item[1])}        

    # just print out top-20 words with highest/lowest weights
    word_list = list(word_att.keys())
    print('Words with highest contribution:')
    for i in range(60):
        print(word_list[i], word_att[word_list[i]])

    print('-----------------------------')
    print('Words with lowest contribution:')
    for i in range(len(word_list)-1, len(word_list)-31, -1):
        print(word_list[i], word_att[word_list[i]])
    print('-------------------------------')


if __name__ == '__main__':
    args = get_args_parser()
    if args.mode == 'train':
        main(args)
    elif args.mode == 'validation':
        validation(args)
    elif args.mode == 'visualization':
        visualization(args)
    elif args.mode == 'language_parsing':
        language_parsing(args)
    elif args.mode == 'language_stat':
        language_statistics(args)