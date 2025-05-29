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
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from models import multi_attn_enc, img_abl, gen_abl, clin_abl, multi_attn_enc_aux, multi_attn_enc_aux_new, multi_attn_enc_aux_new_mlp, multi_attn_enc_aux_new_mlp_clinF, clin_abl_pool,multi_attn_enc_aux_new_mlp_2b2m, multi_attn_enc_concat_all, multi_attn_enc_concat_all_norescon, multi_attn_enc_concat_all_1d_reduced, multi_attn_enc_concat_all_nomaxpool, multi_attn_enc_concat_all_1d_reduced_nomaxp, multi_attn_enc_concat_all_res_maxp_no1d, multi_attn_enc_concat_all_1d_reduced_addQ,multi_attn_cls,early_fusion,late_fusion,benchmark
from test import test_model

from all_data_dataset import all_data_dataset
from pprint import pprint

from utils  import compute_auc
from torch_utils import SaveBestModel

from torch.utils.tensorboard import SummaryWriter

from sklearn.utils.class_weight import compute_class_weight



def train(config, data=None, checkpoint_dir=None):
    
    ablation = config['ablation']
    tune = config['tune']

    if tune == False:
        os.makedirs(os.path.join('../tensorboard_logs',config['fo']), exist_ok= True)
        writer = SummaryWriter(os.path.join('../tensorboard_logs',config['fo']))

    # Define dataloaders
    # TODO: Check why data loaders are doing [dims, batch, feats] -> is this expected behavior!
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['train_index']), batch_size=int(config["batch_size"]), shuffle=True)
    # train_loaders.append(torch.utils.data.DataLoader(train_datasets[e], batch_size=128, shuffle=True))
    
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['val_index']), batch_size=len(config['val_index']), shuffle=True)

    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    config['device'] = device

    # Define model
    if ablation == 'full': 
        if tune:
            if config['alt']:
                # model = multi_attn_enc_aux_new_mlp(config)
                model = multi_attn_enc_aux_new_mlp_clinF(config)
            else:
                model = multi_attn_enc_aux_new(config)
        else: 
            if config['alt']:
                # model = multi_attn_cls(config)
                model = multi_attn_enc_aux_new_mlp_clinF(config)
                # model = early_fusion(config)
            else: 
                # model = multi_attn_enc_concat_all_1d_reduced_nomaxp(config)
                # model = late_fusion(config)
                # model = benchmark(config)
                model = multi_attn_cls(config)
                
    elif ablation == 'img':
        model = img_abl(config)
    elif ablation == 'clin':
        model = clin_abl_pool(config)
    elif ablation == 'gen':
        model = gen_abl(config)
    
    # GPU settings
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # criterion = nn.CrossEntropyLoss()
    class_weights=torch.tensor(compute_class_weight('balanced', classes=np.unique(data.get_labels(config['train_index'])), y=data.get_labels(config['train_index']).numpy()),dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)     

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    save_best_model = SaveBestModel()

    # Training loop
    for epoch in range(1, 100): 
        model.train()
        sum_loss = 0.
        auc_avg = 0. 
        if ablation == 'full':
            sum_loss_aux_clin = 0.
            sum_loss_aux_gen = 0.
            auc_avg_aux_clin = 0.
            auc_avg_aux_gen = 0.
            sum_loss_aux_img = 0.
            auc_avg_aux_img = 0.


        for batch_idx, (_, _, clincal, target, gen, img, gene) in enumerate(train_loader):
            optimizer.zero_grad()
            if ablation == 'full':     
                clincal, target, gen, img, gene = clincal.to(device).float(), target.to(device).long(), gen.to(device).float(), img.to(device).float(), gene.to(device).long()
                # severity_score, img_aux, clin_aux, gen_aux, out_embedding, img_clin_attn, img_gen_attn, clin_gen_attn  = model(clincal, img, gen, gene)
                severity_score  = model(clincal, img, gen, gene)
            elif ablation == 'img':
                img, target= img.to(device).float(), target.to(device).long()
                severity_score, out_embedding, self_attn  = model(img)
            elif ablation == 'clin':
                clincal, target = clincal.to(device).float(), target.to(device).long()
                severity_score, out_embedding, self_attn  = model(clincal)
            elif ablation == 'gen':
                gen, target, gene = gen.to(device).float(), target.to(device).long(), gene.to(device).long()
                severity_score, out_embedding, self_attn  = model(gen, gene)


            if ablation == 'full':
                loss_joint = criterion(severity_score, target)
                # loss_aux_clin = criterion(clin_aux, target)
                # loss_aux_gen = criterion(gen_aux, target)
                # loss_aux_img = criterion(img_aux, target)
                auc_avg += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())
                # train_loss = loss_joint + config['alpha'] * loss_aux_clin + config['beta'] * loss_aux_gen + config['beta'] * loss_aux_img
                train_loss = loss_joint
                sum_loss += train_loss.item()
                # if tune == False:
                #     sum_loss_aux_clin += loss_aux_clin.item()
                #     sum_loss_aux_gen += loss_aux_gen.item()
                #     sum_loss_aux_img += loss_aux_img.item()
                #     auc_avg_aux_clin += compute_auc(torch.exp(F.log_softmax(clin_aux,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())
                #     auc_avg_aux_gen += compute_auc(torch.exp(F.log_softmax(gen_aux,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())
                #     auc_avg_aux_img += compute_auc(torch.exp(F.log_softmax(img_aux,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())
                train_loss.backward()
                optimizer.step()        

            else: 
                train_loss = criterion(severity_score,target) 
                sum_loss += train_loss.item()
                auc_avg += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())
                train_loss.backward()
                optimizer.step()
         

            
            # if ablation == 'full':     
            #     del clincal, target, gen, img, gene, out_embedding, img_clin_attn, img_gen_attn, clin_gen_attn
            # elif ablation == 'img':
            #     del target, img, out_embedding, self_attn
            # elif ablation == 'clin':
            #     del target, clincal, out_embedding, self_attn
            # elif ablation == 'gen':
            #     del target, gen, out_embedding, self_attn, gene


        
        # loss and auc per epoch
        sum_loss /= (batch_idx+1)
        auc_avg /= (batch_idx+1)
        # if ablation == 'full':
        #     sum_loss_aux_clin /= (batch_idx+1)
        #     sum_loss_aux_gen /= (batch_idx+1)
        #     sum_loss_aux_img /= (batch_idx+1)
        #     auc_avg_aux_clin /= (batch_idx+1)
        #     auc_avg_aux_gen /= (batch_idx+1)
        #     auc_avg_aux_img /= (batch_idx+1)


        if tune == False:
            train_losses.append(sum_loss)
            train_aucs.append(auc_avg)
            writer.add_scalar("Loss/train", sum_loss, epoch)
            writer.add_scalar("Auc/train", auc_avg, epoch)
            # if ablation == 'full':
            #     writer.add_scalar("Loss_clin/train", sum_loss_aux_clin, epoch)
            #     writer.add_scalar("Auc_clin/train", auc_avg_aux_clin, epoch)
            #     writer.add_scalar("Loss_gen/train", sum_loss_aux_gen, epoch)
            #     writer.add_scalar("Auc_gen/train", auc_avg_aux_gen, epoch)
            #     writer.add_scalar("Loss_img/train", sum_loss_aux_img, epoch)
            #     writer.add_scalar("Auc_img/train", auc_avg_aux_img, epoch)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_auc = 0.
        val_loss_aux_clin = 0.
        val_loss_aux_gen = 0.
        val_auc_aux_clin = 0.
        val_auc_aux_gen = 0.
        val_loss_aux_img = 0.
        val_auc_aux_img = 0.
        # bring models to evaluation mode
        model.eval()
        with torch.no_grad():
            for batch_idx, (_, _, clincal_val, target_val, gen_val, img_val, gene_val) in enumerate(val_loader, 0):
                
                if ablation == 'full':     
                    clincal_val, target_val, gen_val, img_val, gene_val = clincal_val.to(device).float(), target_val.to(device).long(), gen_val.to(device).float(), img_val.to(device).float(), gene_val.to(device).long()
                    # severity_score, img_aux, clin_aux, gen_aux, out_embedding, img_clin_attn, img_gen_attn, clin_gen_attn = model(clincal_val, img_val, gen_val, gene_val)
                    severity_score = model(clincal_val, img_val, gen_val, gene_val)
                elif ablation == 'img':
                    img_val, target_val= img_val.to(device).float(), target_val.to(device).long()
                    severity_score, out_embedding,self_attn  = model(img_val)
                elif ablation == 'clin':
                    clincal_val, target_val = clincal_val.to(device).float(), target_val.to(device).long()
                    severity_score, out_embedding,self_attn  = model(clincal_val)
                elif ablation == 'gen':
                    gen_val, target_val, gene_val = gen_val.to(device).float(), target_val.to(device).long(), gene_val.to(device).long()
                    severity_score, out_embedding,self_attn  = model(gen_val, gene_val)

                if ablation == 'full':
                    loss_joint = criterion(severity_score, target_val)
                    # loss_aux_clin = criterion(clin_aux, target_val)
                    # loss_aux_gen = criterion(gen_aux, target_val)
                    # loss_aux_img = criterion(img_aux, target_val)
                    # v_loss = loss_joint + config['alpha'] * loss_aux_clin + config['beta'] * loss_aux_gen + config['beta'] * loss_aux_img
                    v_loss = loss_joint
                    val_auc += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target_val.detach().cpu().numpy())          
                    val_loss += v_loss.item()
                    # if tune == False:
                    #     val_auc_aux_clin += compute_auc(torch.exp(F.log_softmax(clin_aux,dim=1)).detach().cpu().numpy(), target_val.detach().cpu().numpy())
                    #     val_auc_aux_gen += compute_auc(torch.exp(F.log_softmax(gen_aux,dim=1)).detach().cpu().numpy(), target_val.detach().cpu().numpy())
                    #     val_auc_aux_img += compute_auc(torch.exp(F.log_softmax(img_aux,dim=1)).detach().cpu().numpy(), target_val.detach().cpu().numpy())
                    #     val_loss_aux_clin += loss_aux_clin.item()
                    #     val_loss_aux_gen += loss_aux_gen.item()  
                    #     val_loss_aux_img += loss_aux_img.item()  
                    val_steps += 1
                else: 
                    loss = criterion(severity_score, target_val)
                    val_auc += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target_val.detach().cpu().numpy())
                    val_loss += loss.item()
                    val_steps += 1
                
                # if ablation == 'full':     
                #     del clincal_val, target_val, gen_val, img_val, gene_val, out_embedding, img_clin_attn, img_gen_attn, clin_gen_attn
                # elif ablation == 'img':
                #     del target_val, img_val, out_embedding, self_attn
                # elif ablation == 'clin':
                #     del target_val, clincal_val, out_embedding, self_attn
                # elif ablation == 'gen':
                #     del target_val, gen_val, out_embedding, self_attn, gene_val    
 
            
            
            if tune:
                os.makedirs("my_model", exist_ok=True)
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
                checkpoint = Checkpoint.from_directory("my_model")
                session.report({"loss": (val_loss / val_steps), 'auc':(val_auc / val_steps)}, checkpoint=checkpoint)

            else:
                os.makedirs("../checkpoints", exist_ok=True)
                # torch.save((model.state_dict(), optimizer.state_dict()), os.path.join('../checkpoints',checkpoint_dir))
                save_best_model((val_auc / val_steps),epoch,model,optimizer,os.path.join('../checkpoints',checkpoint_dir))   

        if tune == False:
            val_losses.append(val_loss/val_steps)
            val_aucs.append(val_auc/val_steps)
            writer.add_scalar("Loss/val", val_loss/val_steps, epoch)
            writer.add_scalar("Auc/val", val_auc/val_steps, epoch)
            # if ablation == 'full':
            #     writer.add_scalar("Loss_clin/val", val_loss_aux_clin/val_steps, epoch)
            #     writer.add_scalar("Auc_clin/val", val_auc_aux_clin/val_steps, epoch)
            #     writer.add_scalar("Loss_gen/val", val_loss_aux_gen/val_steps, epoch)
            #     writer.add_scalar("Auc_gen/val", val_auc_aux_gen/val_steps, epoch)
            #     writer.add_scalar("Loss_img/val", val_loss_aux_img/val_steps, epoch)
            #     writer.add_scalar("Auc_img/val", val_auc_aux_img/val_steps, epoch)

    print("Finished Training")
    if tune == False:
        writer.flush()
        writer.close()
        return train_losses, train_aucs, val_losses, val_aucs
