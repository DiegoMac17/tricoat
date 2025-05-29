from functools import partial
from re import A
from tkinter.messagebox import NO
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from models import multi_attn_enc
from test import test_model
from train import train

from all_data_dataset import all_data_dataset
from pprint import pprint

from utils  import compute_auc


def tuner(full_data, train_index, val_index, test_index, paths, args, config = None, num_samples=10, max_num_epochs=10, device='gpu', gpus_per_trial=1):
    if config == None:
        config = {
            # Data indices
            'train_index' : train_index,
            'val_index' : val_index,
            'test_index' : test_index,
            'tune' : args['tune_flag'],
            'ablation' : args['ablation'],
            ## Training hyperparms
            "lr": tune.loguniform(1e-5, 1e-1), 
            "batch_size" : tune.grid_search([64]),
            ## Model hyperparams
            # 'units': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            'units': tune.grid_search([128,256,512]),
            'num_layers' : tune.grid_search([2,4,6]),
            'num_layers_joint' : tune.grid_search([0]),
            'd_model' :tune.grid_search([256]), 
            'nhead': tune.grid_search([2,4,8]),
            'dim_feedforward' : tune.grid_search([256]),
            'dropout' : tune.grid_search([0.1,0.3]),
            'dropout_fc' : tune.grid_search([0.1,0.3]),
            'layer_norm_eps' : tune.grid_search([0.000001]),
            'activation' : 'relu',
            'alpha' : tune.grid_search([0.25, 0.5, 0.75]),
            'beta' : tune.grid_search([0.25, 0.5, 0.75]),
            'maxpool_flag' : tune.grid_search([False]),
            'alt' : args['alt'],
        }
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "auc", "training_iteration"])
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, data = full_data),
            resources={"cpu": 32, "gpu": float(args['gpus_per_trial'])}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=config,
        run_config=RunConfig(local_dir="../ray_results")
    )


    result = tuner.fit()

    best_trial = result.get_best_result("loss", "min")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))
    print("Best trial final OVO AUC: {}".format(best_trial.metrics["auc"]))

    # Evaluate on all data paritions
    loss_train, auc_train, ids_out_train, clinicals_train, imgs_train, gens_train, targets_train, sev_scores_train, out_embeddings_train, out_img_clin_attn_l_train, out_img_gen_attn_l_train, out_self_attn_train, out_clin_gen_attn_l_train = test_model(best_trial.config, full_data, args['ablation'], train_index, best_trial.checkpoint.to_directory())
    loss_val, auc_val, ids_out_val, clinicals_val, imgs_val, gens_val, targets_val, sev_scores_val, out_embeddings_val, out_img_clin_attn_l_val, out_img_gen_attn_l_val, out_self_attn_val, out_clin_gen_attn_l_val = test_model(best_trial.config, full_data, args['ablation'], val_index, best_trial.checkpoint.to_directory())
    loss_test, auc_test, ids_out_test, clinicals_test, imgs_test, gens_test, targets_test, sev_scores_test, out_embeddings_test, out_img_clin_attn_l_test, out_img_gen_attn_l_test, out_self_attn_test, out_clin_gen_attn_l_test = test_model(best_trial.config, full_data, args['ablation'], test_index, best_trial.checkpoint.to_directory())

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

        },
        'hyperparams':{
            'lr':best_trial.config["lr"],
            'batch_size':best_trial.config["batch_size"],
            'units':best_trial.config["units"],
            'd_model':best_trial.config["d_model"],
            'nhead':best_trial.config["nhead"],
            'dim_feedforward':best_trial.config["dim_feedforward"],
            'dropout':best_trial.config["dropout"],
            'layer_norm_eps':best_trial.config["layer_norm_eps"],
            'activation':best_trial.config["activation"],
            'best_checkpoint_path':best_trial.checkpoint._local_path
        }
    }
    return results_ret