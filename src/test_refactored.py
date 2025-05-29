import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from utils import compute_auc
from models import multi_attn_enc, img_abl, gen_abl, clin_abl, multi_attn_enc_aux, multi_attn_enc_aux_new, multi_attn_enc_aux_new_mlp, multi_attn_enc_aux_new_mlp_clinF, clin_abl_pool,multi_attn_enc_aux_new_mlp_2b2m,multi_attn_enc_concat_all, multi_attn_enc_concat_all_norescon, multi_attn_enc_concat_all_1d_reduced, multi_attn_enc_concat_all_nomaxpool, multi_attn_enc_concat_all_1d_reduced_nomaxp, multi_attn_enc_concat_all_res_maxp_no1d, multi_attn_enc_concat_all_1d_reduced_addQ,multi_attn_cls, early_fusion,late_fusion, benchmark


def test_model(config, data=None, ablation='full', subset_index=None, best_checkpoint_name=None):    
    # Define test loader
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,subset_index), batch_size=int(config['batch_size']), shuffle=True)

    # Define model
    if ablation == 'full':     
        if config['tune']:
            if config['alt']:
                # model = multi_attn_enc_aux_new_mlp(config)
                model = multi_attn_enc_aux_new_mlp_clinF(config)
            else:
                model = multi_attn_enc_aux_new(config)
        else: 
            if config['alt']:
                # model = multi_attn_enc_aux_new_mlp_2b2m(config)
                # model = multi_attn_enc_concat_all(config)
                # model = multi_attn_enc_concat_all_norescon(config)
                # model = multi_attn_enc_concat_all_1d_reduced(config)
                # model = multi_attn_enc_concat_all_nomaxpool(config)
                # model = multi_attn_enc_concat_all_res_maxp_no1d(config)
                # model = multi_attn_enc_concat_all_1d_reduced_addQ(config)
                # model = multi_attn_cls(config)
                # model = early_fusion(config)
                model = multi_attn_enc_aux_new_mlp_clinF(config)
                
            else: 
                # model = multi_attn_enc_aux_new_mlp_clinF(config)
                # model = multi_attn_enc_aux(config)
                # model = multi_attn_enc_concat_all_1d_reduced_nomaxp(config)
                # model = multi_attn_enc_concat_all_1d_reduced(config)
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
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    model = model.to(device)

    # Load state for best model
    if config['tune']:
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_name, "checkpoint.pt"))

    else:
        model_state, optimizer_state = torch.load(os.path.join('../checkpoints',best_checkpoint_name))
    
    model.load_state_dict(model_state)

    # Run testing
    model.eval()
    test_loss = 0.
    test_auc = 0.
    ids_l = []
    clincal_l = []
    img_l = []
    gen_l = []
    severity_score_l = []
    clin_aux_l = []
    gen_aux_l = []
    img_aux_l = []
    out_embedding_l = []
    out_img_clin_attn_l = []
    out_img_gen_attn_l = []
    out_clin_gen_attn_l = []
    self_attn_l = []
    target_l = []
    criterion = nn.CrossEntropyLoss() 
    attributions_l = []

    for batch_idx, (ids, _, clincal, target, gen, img, gene) in enumerate(test_loader):

        if ablation == 'full':     
            clincal, target, gen, img, gene = clincal.to(device).float().requires_grad_(), target.to(device).long(), gen.to(device).float().requires_grad_(), img.to(device).float().requires_grad_(), gene.to(device).long()
            # severity_score, img_aux, clin_aux, gen_aux, out_embedding,img_clin_attn, img_gen_attn, clin_gen_attn  = model(clincal, img, gen, gene)
            severity_score  = model(clincal, img, gen, gene)
            ig = IntegratedGradients(model)
            # attributions, approximation_error = ig.attribute((clincal, img, gen),
                                                # baselines=(torch.zeros(clincal.size()).to(device), torch.zeros(img.size()).to(device), torch.zeros(gen.size()).to(device)),
                                                # method='gausslegendre',
                                                # target = 0,
                                                # return_convergence_delta=True) 

            attributions, approximation_error = [0],[0]                 
        elif ablation == 'img':
            img, target= img.to(device).float().requires_grad_(), target.to(device).long().requires_grad_()
            severity_score, out_embedding, self_attn  = model(img)
            ig = IntegratedGradients(model)
            attributions, approximation_error = ig.attribute((img),baselines=(torch.zeros(img.size()).to(device)),method='gausslegendre',return_convergence_delta=True)  
        elif ablation == 'clin':
            clincal, target = clincal.to(device).float().requires_grad_(), target.to(device).long().requires_grad_()
            severity_score, out_embedding, self_attn  = model(clincal)
            ig = IntegratedGradients(model)
            attributions, approximation_error = ig.attribute((clincal),baselines=(torch.zeros(clincal.size()).to(device)),method='gausslegendre',return_convergence_delta=True)  
        elif ablation == 'gen':
            gen, target, gene = gen.to(device).float().requires_grad_(), target.to(device).long().requires_grad_(), gene.to(device).long().requires_grad_()
            severity_score, out_embedding, self_attn  = model(gen, gene)
            ig = IntegratedGradients(model)
            attributions, approximation_error = ig.attribute((gen, gene),baselines=(torch.zeros(gen.size()).to(device), torch.zeros(gene.size()).to(device)),method='gausslegendre',return_convergence_delta=True)  


        if ablation == 'full':
            loss_joint = criterion(severity_score, target)
            # loss_aux_clin = criterion(clin_aux, target)
            # loss_aux_gen = criterion(gen_aux, target)
            # loss_aux_img = criterion(img_aux, target)
            # loss = loss_joint + config['alpha'] * loss_aux_clin + config['beta'] * loss_aux_gen + config['beta'] * loss_aux_img
            loss = loss_joint
            test_loss += loss.item()
            test_auc += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())     
        else: 
            loss = criterion(severity_score,target)
            # sum up batch loss
            test_loss += loss.item()
            test_auc += compute_auc(torch.exp(F.log_softmax(severity_score,dim=1)).detach().cpu().numpy(), target.detach().cpu().numpy())          


        # Inputs and Outputs per batch for returning
        ids_l.extend(ids.detach().cpu().numpy())
        target_l.extend(target.detach().cpu().numpy())
        attributions_l.extend(attributions)

        severity_score_l.extend(severity_score.detach().cpu().numpy())
        # if ablation == 'full':
        #     clin_aux_l.extend(clin_aux.detach().cpu().numpy())
        #     gen_aux_l.extend(gen_aux.detach().cpu().numpy())
        #     img_aux_l.extend(img_aux.detach().cpu().numpy())
        # out_embedding_l.extend(out_embedding.detach().cpu().numpy())

        # if ablation == 'full':     
        #     clincal_l.extend(clincal.detach().cpu().numpy())
        #     img_l.extend(img.detach().cpu().numpy())
        #     gen_l.extend(gen.detach().cpu().numpy())
        #     out_img_clin_attn_l.extend(img_clin_attn.detach().cpu().numpy())
        #     out_img_gen_attn_l.extend(img_gen_attn.detach().cpu().numpy())
        #     out_clin_gen_attn_l.extend(clin_gen_attn.detach().cpu().numpy())
        #     self_attn_l.extend([None])
        # elif ablation == 'img':
        #     clincal_l.extend([None])
        #     img_l.extend(img.detach().cpu().numpy())
        #     gen_l.extend([None])
        #     out_img_clin_attn_l.extend([None])
        #     out_img_gen_attn_l.extend([None])
        #     self_attn_l.extend(self_attn)
        # elif ablation == 'clin':
        #     clincal_l.extend(clincal.detach().cpu().numpy())
        #     img_l.extend([None])
        #     gen_l.extend([None])
        #     self_attn_l.extend(self_attn)
        # elif ablation == 'gen':
        #     clincal_l.extend([None])
        #     img_l.extend([None])
        #     gen_l.extend(gen.detach().cpu().numpy())
        #     self_attn_l.extend(self_attn)

    # Dataloader metric computation
    # test_loss /= len(test_loader.dataset)
    test_loss /= (batch_idx+1)
    test_auc /= (batch_idx+1)
    
    return test_loss, test_auc, ids_l, clincal_l, img_l, gen_l, target_l, severity_score_l, out_embedding_l, out_img_clin_attn_l, out_img_gen_attn_l, self_attn_l, out_clin_gen_attn_l, attributions_l