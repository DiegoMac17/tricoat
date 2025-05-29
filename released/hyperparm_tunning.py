from functools import partial
from re import A
from tkinter.messagebox import NO
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from models import multi_attn_enc
from test import test_model


def train(config, full_data,train_index,val_index,test_index,checkpoint_dir=None):

    # Define dataloaders
    # TODO: Check why data loaders are doing [dims, batch, feats] -> is this expected behavior!
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_data,train_index), batch_size=int(config["batch_size"]), shuffle=True)
    # train_loaders.append(torch.utils.data.DataLoader(train_datasets[e], batch_size=128, shuffle=True))
    
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_data,val_index), batch_size=len(val_index), shuffle=True)
    
    # Define model and training hyperparameters
    model = multi_attn_enc(config)
    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()     

    print(device)
    # Training loop
    for epoch in range(1, 100): 
        model.train()
        sum_loss = 0.      

        for batch_idx, (_, _, clincal, target, gen, img) in enumerate(train_loader):
            # FIXME: Now that target is clusters make sure that the slicing [] matches the correct thing!
            # Make sure to fix on test.py as well.
            clincal, target, gen, img = clincal.to(device).float(), target.to(device).float(), gen.to(device).float(), img.to(device).float()
            optimizer.zero_grad()
            severity_score, out_embedding  = model(clincal, img, gen)
            # out_embedding = out_embedding.detach().cpu().numpy()
            train_loss = criterion(torch.squeeze(severity_score,-1),target)  # sum up batch loss 
            # train_loss = F.mse_loss(severity_score, torch.swapaxes(target,0,1))
            sum_loss += train_loss.item()

        
            train_loss.backward()
            optimizer.step()
        
        # loss per epoch
        sum_loss /= (batch_idx+1)
        # Do a test pass every x epochs
        if epoch % 10 == 10-1:
            # bring models to evaluation mode
            model.eval()
            for batch_idx, (_, _, clincal_val, target_val, gen_val, img_val) in enumerate(val_loader):
                clincal_val, target_val, gen_val, img_val = clincal_val.to(device).float(), target_val.to(device).float(), gen_val.to(device).float(), img_val.to(device).float()
                severity_score, out_embedding  = model(clincal_val, img_val, gen_val)
                val_loss = criterion(torch.squeeze(severity_score,-1), target_val).item()  # sum up batch loss 
                # val_loss = F.mse_loss(severity_score, torch.swapaxes(target_val,0,1))

    # TODO: check if when training wihtout tuner, should we return model or just path to load weights 

    
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss)
    print("Finished Training")
    # return model


def tuner(full_data, train_index, val_index, test_index, config = None, num_samples=10, max_num_epochs=10, device='gpu', gpus_per_trial=1):
    if config == None:
        config = {
            ## Training hyperparms
            "lr": tune.loguniform(1e-5, 1e-1), 
            "batch_size" : tune.grid_search([64]),
            "penalty": tune.uniform(0.5, 1.5),
            "alpha":tune.grid_search([0.05,0.1,0.15,0.25,0.5,0.75]),
            "beta": tune.grid_search([0.1,0.15,0.25,0.5,0.75,1]),
            ## Model hyperparams
            'units': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            'num_layers' : tune.grid_search([1]),
            'd_model' :tune.grid_search([8]), 
            'nhead': tune.grid_search([1]),
            'dim_feedforward' : tune.grid_search([8]),
            'dropout' : tune.grid_search([0.1]),
            'layer_norm_eps' : tune.grid_search([0.000001]),
            'activation' : tune.grid_search(['ReLU']),
        }
        # config = {
        #     ## Training hyperparms
        #     "lr": 0.001,
        #     "batch_size" : 64,
        #     ## Model hyperparams
        #     'units': 16,
        #     'num_layers' : 1,
        #     'd_model' : 1, # need to double check this
        #     'nhead': 1,
        #     'dim_feedforward' : 8,
        #     'dropout' : 0.1,
        #     'layer_norm_eps' : 0.000001,
        #     'activation' : 'relu',
        # }
    # best_trained_model = train(config=config,full_data=full_data,train_index=train_index,val_index=val_index,test_index=test_index)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train, full_data=full_data,train_index=train_index,val_index=val_index,test_index=test_index),
        resources_per_trial={"cpu": 32, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    # Evaluation on test data of best performing model (on val data)
    best_trained_model = multi_attn_enc(config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)

    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data 
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # Evaluate on all data paritions
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_data,train_index),  batch_size=len(train_index), shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_data,val_index),  batch_size=len(val_index), shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_data,test_index),  batch_size=len(test_index), shuffle=True)

    loss_train, ids_out_train, clinicals_train, imgs_train, gens_train, targets_train, sev_scores_train, out_embeddings_train = test_model(best_trained_model, train_loader)
    loss_val, ids_out_val, clinicals_val, imgs_val, gens_val, targets_val, sev_scores_val, out_embeddings_val = test_model(best_trained_model, val_loader)
    loss_test, ids_out_test, clinicals_test, imgs_test, gens_test, targets_test, sev_scores_test, out_embeddings_test = test_model(best_trained_model, test_loader)

    del best_trained_model

    print("Best trial test set R2: {}".format(loss_test))

    # Return dict with results
    results_ret = {
        'metrics':{
            'loss_train':loss_train,
            'loss_val':loss_val,
            'loss_test':loss_test
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
        },
        'hyperparams':{
            'lr':best_trial.config["lr"],
            'pen_mult':best_trial.config["penalty"],
            'units':best_trial.config["units"],
            'alpha':best_trial.config["alpha"],
            'beta':best_trial.config["beta"]
        }
    }
    return results_ret