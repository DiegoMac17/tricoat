import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import multi_attn_enc
# from train import train
from train_refactored import train
# from test import test_model
from test_refactored import test_model

from utils  import compute_auc


def train_test_no_tune(full_data, train_index, val_index, test_index, paths, args, config = None):
    if config == None:
        config = {
            # Data indices
            'train_index' : train_index,
            'val_index' : val_index,
            'test_index' : test_index,
            'tune' : args['tune_flag'],
            'ablation' : args['ablation'],
            ## Training hyperparms
            "lr": 0.0001,
            "batch_size" : 64,
            ## Model hyperparams
            'units': 256,
            'num_layers' : 4,
            'd_model' : 256, 
            'nhead': 4,
            'dim_feedforward' : 256,
            'dropout' : 0.0,
            'layer_norm_eps' : 0.000001,
            'activation' : 'relu',
            'alpha' : 0.5,
            'beta' : 0.5,
        }
    else:
        config['train_index'] = train_index
        config['val_index'] = val_index
        config['test_index'] = test_index

    # Run train    
    train_losses, train_aucs, val_losses, val_aucs = train(config, data=full_data, checkpoint_dir = paths['checkpoint_name'])

    # Evaluate on all data paritions
    loss_train, auc_train, ids_out_train, clinicals_train, imgs_train, gens_train, targets_train, sev_scores_train, out_embeddings_train, out_img_clin_attn_l_train, out_img_gen_attn_l_train, out_self_attn_train, out_clin_gen_attn_l_train, attributions_l_train = test_model(config, full_data, args['ablation'], train_index, paths['checkpoint_name'])
    loss_val, auc_val, ids_out_val, clinicals_val, imgs_val, gens_val, targets_val, sev_scores_val, out_embeddings_val, out_img_clin_attn_l_val, out_img_gen_attn_l_val, out_self_attn_val, out_clin_gen_attn_l_val, attributions_l_val = test_model(config, full_data, args['ablation'], val_index, paths['checkpoint_name'])
    loss_test, auc_test, ids_out_test, clinicals_test, imgs_test, gens_test, targets_test, sev_scores_test, out_embeddings_test, out_img_clin_attn_l_test, out_img_gen_attn_l_test, out_self_attn_test, out_clin_gen_attn_l_test, attributions_l_test = test_model(config, full_data, args['ablation'], test_index, paths['checkpoint_name'])


    print("Best trial test set loss: {}".format(loss_test))

    # Return dict with results
    results_ret = {
        'metrics':{
            'loss_train':loss_train,
            'loss_val':loss_val,
            'loss_test':loss_test,
            'auc_train':auc_train,
            'auc_val':auc_val,
            'auc_test':auc_test,
        },
        'data':{
            'ids_out_train':ids_out_train,
            'ids_out_val':ids_out_val,
            'ids_out_test':ids_out_test,
            'sev_scores_train':sev_scores_train,
            'sev_scores_val':sev_scores_val,
            'sev_scores_test':sev_scores_test,
            'Pheno_og_train':targets_train,
            'Pheno_og_val':targets_val,
            'Pheno_og_test':targets_test,
            'out_embeddings_train':out_embeddings_train,
            'out_embeddings_val':out_embeddings_val,
            'out_embeddings_test':out_embeddings_test,
            'out_img_clin_attn_l_train': out_img_clin_attn_l_train,
            'out_img_clin_attn_l_val': out_img_clin_attn_l_val,
            'out_img_clin_attn_l_test': out_img_clin_attn_l_test, 
            'out_img_gen_attn_l_train': out_img_gen_attn_l_train, 
            'out_img_gen_attn_l_val': out_img_gen_attn_l_val,
            'out_img_gen_attn_l_test': out_img_gen_attn_l_test,
            'out_clin_gen_attn_l_train': out_clin_gen_attn_l_train, 
            'out_clin_gen_attn_l_val': out_clin_gen_attn_l_val,
            'out_clin_gen_attn_l_test': out_clin_gen_attn_l_test,
            'out_self_attn_train' : out_self_attn_train,
            'out_self_attn_val' : out_self_attn_val,
            'out_self_attn_test' : out_self_attn_test,
            'train_losses': train_losses,
            'train_aucs':train_aucs,
            'val_losses':val_losses,
            'val_aucs':val_aucs,
            'attributions_l_train':attributions_l_train,
            'attributions_l_val':attributions_l_val,
            'attributions_l_test':attributions_l_test,


        },
        'hyperparams':{
            'lr':config["lr"],
            'batch_size':config["batch_size"],
            'units':config["units"],
            'd_model':config["d_model"],
            'nhead':config["nhead"],
            'dim_feedforward':config["dim_feedforward"],
            'dropout':config["dropout"],
            'layer_norm_eps':config["layer_norm_eps"],
            'activation':config["activation"],
            'checkpoint_path':os.path.join('../checkpoints',paths['checkpoint_name'])
        }
    }
    return results_ret