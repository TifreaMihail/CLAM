import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_abmil import ABMIL, GABMIL
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb', 'abmil']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type =='abmil':
        model = ABMIL(**model_dict)
    elif args.model_type =='gabmil':
        model_dict.update({'win_size': args.window_size})
        model_dict.update({'use_grid': args.use_grid})
        model_dict.update({'use_skip': args.use_skip})
        model_dict.update({'use_norm': args.use_norm})
        model_dict.update({'use_block': args.use_block})
        model_dict.update({'use_weight_norm': args.use_weight_norm})
        model = GABMIL(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        # ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        new_key = key.replace('.module', '')
        if args.model_type =='abmil':
            new_key2 = new_key.replace('attention_net.3', 'attention_net.2')
        elif args.model_type =='gabmil':
            new_key2 = new_key.replace('attention_net.3', 'attention_net.2')
        else:
            new_key2 = new_key
        # Update the clean checkpoint dictionary
        ckpt_clean[new_key2] = ckpt[key]
        
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, f1, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, f1, df

def get_f1(preds, labels):
    y_true = np.asarray(preds).reshape(-1,)
    y_pred = np.asarray(labels).reshape(-1,)
    f1 = f1_score(y_true,y_pred,average='macro')
    return f1

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
        f1 = -1
    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            f1 = get_f1(all_preds, all_labels)
            #f1 = f1_score(all_labels, all_preds, average='binary')
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            # f1 = f1_score(all_labels, all_preds, average='weighted')
            f1 = get_f1(all_preds, all_labels)
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, f1, df, acc_logger
