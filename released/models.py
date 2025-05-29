import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiheadAttention as mod_mha
from einops.layers.torch import Reduce

############################# Benchmark: intermediate fusion  #################################

class benchmark(nn.Module):
    # multi_attn_enc_concat_all_1d_reduced_addQ_cls
    def __init__(self, config):
        super(benchmark, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']
        self.device = config['device']

        self.b_dim = 7
        self.m_dim = 72 * 4
        self.n_dim = 70 * 4

        self.dims1h1 = 64
        self.dims1h2 = 32
        self.dims1h3 = 32
        self.dims1h4 = 16
        self.dims2h1 = 32
        self.dims2h2 = 16
        self.dims3h1 = 32
        self.dims3h2 = 16

        ## Single modality fcs
        self.c1 = nn.Linear(self.b_dim,self.dims1h1)
        self.i1 = nn.Linear(self.m_dim,self.dims1h1)
        self.g1 = nn.Linear(self.n_dim, self.dims1h1)

        self.c2 = nn.Linear(self.dims1h1,self.dims1h2)
        self.i2 = nn.Linear(self.dims1h1,self.dims1h2)
        self.g2 = nn.Linear(self.dims1h1,self.dims1h2)

        self.c3 = nn.Linear(self.dims1h2,self.dims1h3)
        self.i3 = nn.Linear(self.dims1h2,self.dims1h3)
        self.g3 = nn.Linear(self.dims1h2,self.dims1h3)

        self.logits_c = nn.Linear(self.dims1h1,3)
        self.logits_i = nn.Linear(self.dims1h1,3)
        self.logits_g = nn.Linear(self.dims1h1,3)

        ## Stage2 fusion
        self.c_i1 = nn.Linear(self.dims1h3+self.dims1h3,self.dims2h1)
        self.c_g1 = nn.Linear(self.dims1h3+self.dims1h3,self.dims2h1)
        self.i_g1 = nn.Linear(self.dims1h3+self.dims1h3,self.dims2h1)

        self.c_i2 = nn.Linear(self.dims2h1,self.dims2h2)
        self.c_g2 = nn.Linear(self.dims2h1,self.dims2h2)
        self.i_g2 = nn.Linear(self.dims2h1,self.dims2h2)

        self.logits_c_i = nn.Linear(self.dims2h2,3)
        self.logits_c_g = nn.Linear(self.dims2h2,3)
        self.logits_i_g = nn.Linear(self.dims2h2,3)

        ## Stage 3 fusion
        self.f = nn.Linear(self.dims2h2+self.dims2h2+self.dims2h2,self.dims3h1)
        self.logits_f = nn.Linear(self.dims3h1,3)

    def forward(self, clinical, img, gen, gene):
        
        # Flatten all tokens for MLPs

        img = torch.flatten(img, 1)
        gen = torch.flatten(gen, 1)

        # Stage 1
        clinical = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(clinical))))))
        img = F.relu(self.i3(F.relu(self.i2(F.relu(self.i1(img))))))
        gen = F.relu(self.g3(F.relu(self.g2(F.relu(self.g1(gen))))))
        # self.logits_c(clinical)
        # self.logits_i(img)
        # self.logits_g(gen)

        # Stage 2 
        c_i = F.relu(self.c_i2(F.relu(self.c_i1(torch.cat((clinical,img), dim = 1)))))
        c_g = F.relu(self.c_g2(F.relu(self.c_g1(torch.cat((clinical,gen), dim = 1)))))
        i_g = F.relu(self.i_g2(F.relu(self.i_g1(torch.cat((img,gen), dim = 1)))))

        # Stage 3
        f = F.relu(self.f(torch.cat((c_i,c_g,i_g), dim = 1)))
        logits = self.logits_f(f)

        return logits 

############################# Late fusion ####################################

class late_fusion(nn.Module):
    # multi_attn_enc_concat_all_1d_reduced_addQ_cls
    def __init__(self, config):
        super(late_fusion, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']
        self.device = config['device']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model))
        # self.gen_embd = nn.Linear(4,int(self.d_model/2))
        # self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))
        
        # CLS token 
        self.cls_emb = nn.Embedding(3,int(self.d_model)) # one cls token per modality, i.e. 3 tokens

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)
        self.conv_1 = nn.Conv1d(((self.b_dim + self.m_dim + self.n_dim)),((self.b_dim + self.m_dim + self.n_dim)), kernel_size=self.d_model, stride=1)
        self.conv_2 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/2), stride=1)
        self.conv_3 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/4), stride=1)
        self.conv_4 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=16, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Fused
        self.linear = nn.Linear(9,self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # Img
        self.linear_img = nn.Linear(self.d_model,self.units)
        self.logits_img = nn.Linear(self.units,3)
        # clin
        self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen):
        # TODO: figure out a way to put it into device in a more simple way
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        # gene = self.gen_chr_embd(gene)
        # gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))

        # cls token emb
        self.gen_tk, self.clin_tk, self.img_tk = torch.zeros(gen.shape[0],1).long().to(self.device), torch.ones(gen.shape[0],1).long().to(self.device),torch.ones(gen.shape[0],1).long().to(self.device)*2
        gen = torch.cat((self.cls_emb(self.gen_tk),gen), dim=1)
        clinical = torch.cat((self.cls_emb(self.clin_tk),clinical), dim=1)
        img = torch.cat((self.cls_emb(self.img_tk),img), dim=1)
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)
        
        # MLP classification based on [cls] token
        # Img
        logits_img = F.relu(self.linear_img(img[0]))
        logits_img = self.logits_img(logits_img)
        # clin
        logits_clinical = F.relu(self.linear_clin(clinical[0]))
        logits_clinical = self.logits_clin(logits_clinical)
        # gen
        logits_gen = F.relu(self.linear_gen(gen[0]))
        logits_gen = self.logits_gen(logits_gen)
        # Late fusion
        fused = torch.cat((logits_clinical, logits_img, logits_gen), dim=1)
        logits = F.relu(self.linear(fused))
        logits = self.logits(logits)

        return logits


############################# Early fusion ####################################
class early_fusion(nn.Module):
    # multi_attn_enc_concat_all_1d_reduced_addQ_cls
    def __init__(self, config):
        super(early_fusion, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']
        self.device = config['device']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model))
        
        # CLS token 
        self.cls_emb = nn.Embedding(1,int(self.d_model)) # one cls token per modality, i.e. 3 tokens

        # Transformer encoders
        self.all_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.transformer_encoder_all = nn.TransformerEncoder( self.all_encoder_tf_layer, num_layers=self.num_layers_single)

        # MLPs
        self.linear = nn.Linear(self.d_model,self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen):
        # TODO: figure out a way to put it into device in a more simple way
        # Embedding       
        gen = F.relu(self.gen_embd(gen))     
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))

        # cls token emb
        self.tk = torch.zeros(img.shape[0],1).long().to(self.device)
        all_concat = torch.cat((self.cls_emb(self.tk),clinical,img,gen), dim = 1)
        # self.gen_tk, self.clin_tk, self.img_tk = torch.zeros(gen.shape[0],1).long().to(self.device), torch.ones(gen.shape[0],1).long().to(self.device),torch.ones(gen.shape[0],1).long().to(self.device)*2
        # gen = torch.cat((self.cls_emb(self.gen_tk),gen), dim=1)
        # clinical = torch.cat((self.cls_emb(self.clin_tk),clinical), dim=1)
        # img = torch.cat((self.cls_emb(self.img_tk),img), dim=1)

        # Swap axis for batches in following transformer opertions
        all_concat = self.transformer_encoder_all(torch.swapaxes(all_concat,0,1))

        logits = F.relu(self.linear(torch.swapaxes(all_concat,0,1)[:,0]))
        logits = self.logits(logits)
        return logits 



################################# Post MICCAI full model: CLS token mod #################################
class multi_attn_cls(nn.Module):
    # multi_attn_enc_concat_all_1d_reduced_addQ_cls
    def __init__(self, config):
        super(multi_attn_cls, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']
        self.device = config['device']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model))
        # self.gen_embd = nn.Linear(4,int(self.d_model/2))
        # self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))
        
        # CLS token 
        self.cls_emb = nn.Embedding(3,int(self.d_model)) # one cls token per modality, i.e. 3 tokens

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)
        self.conv_1 = nn.Conv1d(((self.b_dim + self.m_dim + self.n_dim)),((self.b_dim + self.m_dim + self.n_dim)), kernel_size=self.d_model, stride=1)
        self.conv_2 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/2), stride=1)
        self.conv_3 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/4), stride=1)
        self.conv_4 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=16, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Fused
        self.linear = nn.Linear(self.d_model*3,self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,2)
        # Img
        self.linear_img = nn.Linear(self.d_model,self.units)
        self.logits_img = nn.Linear(self.units,3)
        # clin
        self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # TODO: figure out a way to put it into device in a more simple way
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        # gene = self.gen_chr_embd(gene)
        # gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))

        # cls token emb
        self.gen_tk, self.clin_tk, self.img_tk = torch.zeros(gen.shape[0],1).long().to(self.device), torch.ones(gen.shape[0],1).long().to(self.device),torch.ones(gen.shape[0],1).long().to(self.device)*2
        gen = torch.cat((self.cls_emb(self.gen_tk),gen), dim=1)
        clinical = torch.cat((self.cls_emb(self.clin_tk),clinical), dim=1)
        img = torch.cat((self.cls_emb(self.img_tk),img), dim=1)
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V
        
        # MLP classification based on [cls] token
        # Fused
        fused = torch.cat(((img_clin + gen_clin + clinical)[0:1], (gen_img + clin_img + img)[0:1], (clin_gen + img_gen + gen)[0:1]), dim=0)
        logits = F.relu(self.linear(torch.flatten(torch.swapaxes(fused,0,1), 1)))
        logits = self.logits(logits)
        # Img
        logits_img = F.relu(self.linear_img(img[0]))
        logits_img = self.logits_img(logits_img)
        # clin
        logits_clinical = F.relu(self.linear_clin(clinical[0]))
        logits_clinical = self.logits_clin(logits_clinical)
        # gen
        logits_gen = F.relu(self.linear_gen(gen[0]))
        logits_gen = self.logits_gen(logits_gen)



        # return logits, logits_img, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 
        return logits 




################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_1d_reduced_addQ(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_1d_reduced_addQ, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)
        self.conv_1 = nn.Conv1d(((self.b_dim + self.m_dim + self.n_dim)),((self.b_dim + self.m_dim + self.n_dim)), kernel_size=self.d_model, stride=1)
        self.conv_2 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/2), stride=1)
        self.conv_3 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/4), stride=1)
        self.conv_4 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=16, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear(( (self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        
        fused = torch.cat(((img_clin + gen_clin + clinical), (gen_img + clin_img + img), (clin_gen + img_gen + gen) ),dim=0)

        # 1d conv and max pool
        fused = self.conv_1(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 



################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_res_maxp_no1d(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_res_maxp_no1d, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear((3*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        fused = torch.cat((img_clin, clinical, gen_img, img, gen_clin, gen, clin_gen, clin_img, img_gen),dim=0)

        # 1d conv and max pool
        # fused = self.conv(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(torch.swapaxes(fused,0,1)))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 



################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_1d_reduced_nomaxp(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_1d_reduced_nomaxp, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        # self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((2*(self.b_dim + self.m_dim + self.n_dim)),(2*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1) # ablation with no residual connections
        self.conv_1 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=self.d_model, stride=1)
        self.conv_2 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/2), stride=1)
        self.conv_3 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/4), stride=1)
        self.conv_4 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=16, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        # self.linear = nn.Linear((3*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.linear = nn.Linear((self.d_model*2*(self.b_dim + self.m_dim + self.n_dim)),self.units) # ablation with no residual connections
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        # fused = torch.cat((img_clin, clinical, gen_img, img, gen_clin, gen, clin_gen, clin_img, img_gen),dim=0)
        fused = torch.cat((img_clin, gen_img, gen_clin, clin_gen, clin_img, img_gen),dim=0) # ablation with no residual connections

        # 1d conv and max pool
        fused = self.conv(torch.swapaxes(fused,0,1))
        # fused = torch.squeeze(fused, -1)
        fused = torch.flatten(fused, 1)
        # fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 



################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_1d_reduced(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_1d_reduced, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)
        self.conv_1 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=self.d_model, stride=1)
        self.conv_2 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/2), stride=1)
        self.conv_3 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=int(self.d_model/4), stride=1)
        self.conv_4 = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=16, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear((3*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        fused = torch.cat((img_clin, clinical, gen_img, img, gen_clin, gen, clin_gen, clin_img, img_gen),dim=0)

        # 1d conv and max pool
        fused = self.conv_2(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 



################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_nomaxpool(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_nomaxpool, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        # self.linear = nn.Linear((self.d_model*3*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.linear = nn.Linear((self.d_model*2*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        # fused = torch.cat((img_clin, clinical, gen_img, img, gen_clin, gen, clin_gen, clin_img, img_gen),dim=0)
        fused = torch.cat((img_clin, gen_img, gen_clin, clin_gen, clin_img, img_gen),dim=0)

        # 1d conv and max pool
        fused = torch.swapaxes(fused,0,1)
        # fused = self.conv(fused)
        # fused = F.relu(self.max_pooling_layer_fused(fused))
        fused = torch.flatten(fused,1)

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 






################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all_norescon(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all_norescon, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((2*(self.b_dim + self.m_dim + self.n_dim)),(2*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear((2*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        fused = torch.cat((img_clin, gen_img, gen_clin, clin_gen, clin_img, img_gen),dim=0)

        # 1d conv and max pool
        fused = self.conv(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 

################################# Post MICCAI full model #################################
class multi_attn_enc_concat_all(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_concat_all, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        # self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        self.conv = nn.Conv1d((3*(self.b_dim + self.m_dim + self.n_dim)),(3*(self.b_dim + self.m_dim + self.n_dim)), kernel_size=1, stride=1)

        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear((3*(self.b_dim + self.m_dim + self.n_dim)),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_clin, gen_clin_coattn = self.coattn_img_clin(clinical, gen, gen) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V
        clin_gen, clin_gen_coattn = self.coattn_gen_img(gen, clinical, clinical) # Q, K, V
        clin_img, clin_img_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_gen_img(gen, img, img) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        fused = torch.cat((img_clin, clinical, gen_img, img, gen_clin, gen, clin_gen, clin_img, img_gen),dim=0)

        # 1d conv and max pool
        fused = self.conv(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, clin_gen 





################################# Full model with auxiliary losses #################################
class multi_attn_enc_aux_new_mlp_2b2m(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_aux_new_mlp_2b2m, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        self.b_dim = 7
        self.m_dim = 72
        self.n_dim = 70

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)

        self.conv = nn.Conv1d((2*self.b_dim + 2*self.m_dim),(2*self.b_dim + 2*self.m_dim), kernel_size=1, stride=1)
        # MLPs
        self.max_pooling_layer_fused = Reduce('b s d -> b s', 'max')
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        # self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.linear = nn.Linear((2*self.b_dim + 2*self.m_dim),self.units)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen) # Q, K, V



        # Res connection
        # img_clin = torch.cat((img_clin, clinical),dim=1)
        # gen_img = torch.cat((gen_img, img),dim=1)
        fused = torch.cat((img_clin, clinical, gen_img, img),dim=0)

        # 1d conv and max pool
        fused = self.conv(torch.swapaxes(fused,0,1))
        fused = F.relu(self.max_pooling_layer_fused(fused))

        # img_clin = F.relu(self.max_pooling_layer(img_clin))
        # gen_img = F.relu(self.max_pooling_layer(gen_img))

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, gen_img_coattn, 





################################# Full model with auxiliary losses #################################
class multi_attn_enc_aux_new_mlp_clinF(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_aux_new_mlp_clinF, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_clin = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)


        # MLPs
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_img_gen(gen, img, img)
        clin_img, clin_img_coattn = self.coattn_clin_img(img, clinical, clinical) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen)
        clin_gen, clin_gen_coattn = self.coattn_gen_clin(gen, clinical, clinical)
        
        # img_fused = img_clin + img_gen # or concatenate?
        # add q_clin,gen + v_i and then concat
        # img_clin = img_clin + clinical
        # img_gen = img_gen + gen
        # clin_img = clin_img + clinical
        # gen_img = gen_img + gen

        img_clin = F.relu(self.max_pooling_layer(img_clin))
        img_gen = F.relu(self.max_pooling_layer(img_gen))
        clin_img = F.relu(self.max_pooling_layer(clin_img))
        gen_img = F.relu(self.max_pooling_layer(gen_img))

        fused = torch.cat((img_clin, img_gen, clin_img, gen_img),dim=1)

        # Multimodal transformer
        # img_fused = self.multi_transformer_encoder(img_fused)

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.swapaxes(gen,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        # return logits_clinical, logits, logits_gen, fused, img_clin_coattn, img_gen_coattn, clin_gen_coattn
        return logits_clinical


################################# Full model with auxiliary losses #################################
class multi_attn_enc_aux_new_mlp(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_aux_new_mlp, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_clin_img = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_gen_img = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        # self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)


        # MLPs
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        self.linear = nn.Linear(self.d_model*4,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_img_gen(gen, img, img)
        clin_img, clin_img_coattn = self.coattn_clin_img(img, clinical, clinical) # Q, K, V
        gen_img, gen_img_coattn = self.coattn_gen_img(img, gen, gen)
        
        # img_fused = img_clin + img_gen # or concatenate?
        # add q_clin,gen + v_i and then concat
        # img_clin = img_clin + clinical
        # img_gen = img_gen + gen
        # clin_img = clin_img + clinical
        # gen_img = gen_img + gen

        img_clin = F.relu(self.max_pooling_layer(img_clin))
        img_gen = F.relu(self.max_pooling_layer(img_gen))
        clin_img = F.relu(self.max_pooling_layer(clin_img))
        gen_img = F.relu(self.max_pooling_layer(gen_img))

        fused = torch.cat((img_clin, img_gen, clin_img, gen_img),dim=1)

        # Multimodal transformer
        # img_fused = self.multi_transformer_encoder(img_fused)

        # Classification mlps 
        # Img
        logits = F.relu(self.linear(fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Clin
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, fused, img_clin_coattn, img_gen_coattn, 


################################# Full model with auxiliary losses #################################
class multi_attn_enc_aux_new(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_aux_new, self).__init__()
        self.units = config['units']
        self.num_layers_single = config['num_layers']
        self.num_layers_joint = config['num_layers_joint']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.dropout_fc = config['dropout_fc']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']

        self.maxpool_flag = config['maxpool_flag']

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers_single)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers_single)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers_single)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers_joint)


        # MLPs
        self.max_pooling_layer = Reduce('b s d -> b d', 'max')
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        if self.maxpool_flag:
            self.linear = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear = nn.Linear(self.d_model*(70+11),self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits = nn.Linear(self.units,3)
        # clin
        if self.maxpool_flag:
            self.linear_clin = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_clin = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        if self.maxpool_flag:
            self.linear_gen = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear_gen = nn.Linear(self.d_model*70,self.units) # for concat
        self.dropout = nn.Dropout(self.dropout_fc)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_img_gen(gen, img, img)
        # img_clin, img_clin_coattn = self.coattn_clin_img(img, clinical, clinical) # Q, K, V
        # img_gen, img_gen_coattn = self.coattn_gen_img(img, gen, gen)
        
        # img_fused = img_clin + img_gen # or concatenate?
        # add q_clin,gen + v_i and then concat
        img_clin = img_clin + clinical
        img_gen = img_gen + gen
        img_fused = torch.cat((img_clin, img_gen),dim=0)

        # Multimodal transformer
        img_fused = self.multi_transformer_encoder(img_fused)

        # Classification mlps 
        # Img
        img_fused = torch.swapaxes(img_fused,0,1)
        if self.maxpool_flag:
            img_fused = F.relu(self.max_pooling_layer(img_fused))
        else:
            img_fused = torch.flatten(img_fused, 1)
        logits = F.relu(self.linear(img_fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Img
        clinical = torch.swapaxes(clinical,0,1)
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        # Gen
        gen = torch.swapaxes(gen,0,1)
        if self.maxpool_flag:
            gen = F.relu(self.max_pooling_layer(gen))
        else:
            gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, img_fused, img_clin_coattn, img_gen_coattn, 





################################# Full model with auxiliary losses #################################
class multi_attn_enc_aux(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc_aux, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Transformer encoders
        # self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model*2, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers)


        # MLPs
        # Img
        # self.linear = nn.Linear(self.d_model*(72),self.units)
        self.linear = nn.Linear(self.d_model*72*2,self.units) # for concat
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)
        # clin
        # self.linear_clin = nn.Linear(self.d_model*(11),self.units) # for concat
        self.linear_clin = nn.Linear(self.d_model*(7),self.units) # for reduced clin feat
        self.dropout = nn.Dropout(0.10)
        self.logits_clin = nn.Linear(self.units,3)
        # Gen
        self.linear_gen = nn.Linear(self.d_model*(70),self.units) # for concat
        self.dropout = nn.Dropout(0.10)
        self.logits_gen_linear = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # clinical = torch.swapaxes(clinical, 0, 2)
        # img = torch.swapaxes(img, 0, 2)
        # gen = torch.swapaxes(gen, 0, 2)

        # print(clinical.shape, img.shape, gen.shape)

        # Expand dims?? TODO: double check this         
        # clinical = self.clin_embd(torch.unsqueeze(clinical,-1))
        # img = self.img_embd(torch.unsqueeze(img,-1))
        # gen = self.gen_embd(torch.unsqueeze(gen,-1))

        # clinical = torch.swapaxes(torch.unsqueeze(clinical,-1),0,1)
        # img =  torch.swapaxes(torch.unsqueeze(img,-1),0,1)
        # gen =  torch.swapaxes(torch.unsqueeze(gen,-1),0,1)

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        # img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        # img_gen, img_gen_coattn = self.coattn_img_gen(gen, img, img)
        img_clin, img_clin_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_img_gen(img, gen, gen)
        
        # img_fused = img_clin + img_gen # or concatenate?
        img_fused = torch.cat((img_clin, img_gen),dim=2)

        # Multimodal transformer
        img_fused = self.multi_transformer_encoder(img_fused)

        # Classification mlps 
        # Img
        img_fused = torch.swapaxes(img_fused,0,1)
        img_fused = torch.flatten(img_fused, 1)
        logits = F.relu(self.linear(img_fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        # Img
        clinical = torch.swapaxes(clinical,0,1)
        clinical = torch.flatten(clinical, 1)
        logits_clinical = F.relu(self.linear_clin(clinical))
        logits_clinical = self.dropout(logits_clinical)
        logits_clinical = self.logits_clin(logits_clinical)
        # Gen
        gen = torch.swapaxes(gen,0,1)
        gen = torch.flatten(gen, 1)
        logits_gen = F.relu(self.linear_gen(gen))
        logits_gen = self.dropout(logits_gen)
        logits_gen = self.logits_gen_linear(logits_gen)
        
        return logits, logits_clinical, logits_gen, img_fused, img_clin_coattn, img_gen_coattn, 



################################# Full model #################################
class multi_attn_enc(nn.Module):
    def __init__(self, config):
        super(multi_attn_enc, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)
        self.img_embd = nn.Linear(4,self.d_model)
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)

        ## Multi modality branch encoding
        ## Triple Co-attention
        self.coattn_img_clin = mod_mha(embed_dim=self.d_model, num_heads=1)
        self.coattn_img_gen = mod_mha(embed_dim=self.d_model, num_heads=1)

        # Concatenate encodings 

        

        # Transformer encoders
        self.multi_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.multi_transformer_encoder = nn.TransformerEncoder( self.multi_encoder_tf_layer, num_layers=self.num_layers)



        self.linear = nn.Linear(self.d_model*(72),self.units)
        # self.linear = nn.Linear(self.d_model*(72*2),self.units)
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)

    def forward(self, clinical, img, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))
        img = F.relu(self.img_embd(img))
        
        # FIXME: Double check of this correct! Switching the order of dimensions on data to match co-attention impletmentation, also for transformer encoder layer
        # From documentation:
        """"
         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # clinical = torch.swapaxes(clinical, 0, 2)
        # img = torch.swapaxes(img, 0, 2)
        # gen = torch.swapaxes(gen, 0, 2)

        # print(clinical.shape, img.shape, gen.shape)

        # Expand dims?? TODO: double check this         
        # clinical = self.clin_embd(torch.unsqueeze(clinical,-1))
        # img = self.img_embd(torch.unsqueeze(img,-1))
        # gen = self.gen_embd(torch.unsqueeze(gen,-1))

        # clinical = torch.swapaxes(torch.unsqueeze(clinical,-1),0,1)
        # img =  torch.swapaxes(torch.unsqueeze(img,-1),0,1)
        # gen =  torch.swapaxes(torch.unsqueeze(gen,-1),0,1)

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)
        img =  torch.swapaxes(img,0,1)
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)
        img = self.img_transformer_encoder(img)
        gen = self.gen_transformer_encoder(gen)

        # Tri-attention
        # img_clin, img_clin_coattn = self.coattn_img_clin(clinical, img, img) # Q, K, V
        # img_gen, img_gen_coattn = self.coattn_img_gen(gen, img, img)
        img_clin, img_clin_coattn = self.coattn_img_clin(img, clinical, clinical) # Q, K, V
        img_gen, img_gen_coattn = self.coattn_img_gen(img, gen, gen)
        
        img_fused = img_clin + img_gen # or concatenate?
        # img_fused = torch.cat((img_clin, img_gen),dim=2)

        # Multimodal transformer
        img_fused = self.multi_transformer_encoder(img_fused)

        # Classification mlp
        img_fused = torch.swapaxes(img_fused,0,1)
        img_fused = torch.flatten(img_fused, 1)
        logits = F.relu(self.linear(img_fused))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        
        return logits, img_fused, img_clin_coattn, img_gen_coattn
    
################################# Imaging ablation model #################################
class img_abl(nn.Module): # example from fairprs
    def __init__(self, config):
        super(img_abl, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.img_embd = nn.Linear(4,self.d_model)

        # Transformer encoders
        self.img_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.img_transformer_encoder = nn.TransformerEncoder( self.img_encoder_tf_layer, num_layers=self.num_layers)

        # MLP
        self.linear = nn.Linear(self.d_model*72,self.units)
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)

    def forward(self, img):
        # Embeddings
        img = F.relu(self.img_embd(img))

        # Swap axis for batches in following transformer opertions
        img =  torch.swapaxes(img,0,1)

        # Single mod branch encoding
        img = self.img_transformer_encoder(img)

        # Classification mlp
        img = torch.swapaxes(img,0,1)
        img = torch.flatten(img, 1)
        logits = F.relu(self.linear(img))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        
        # FIXME: Figure out a way to return self attention from pytorch transformer encoder
        return logits, img, [None]
    



################################# Clinical ablation model #################################
class clin_abl(nn.Module): # example from fairprs
    def __init__(self, config):
        super(clin_abl, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers)

        # MLP
        # self.linear = nn.Linear(self.d_model*11,self.units)
        self.linear = nn.Linear(self.d_model*7,self.units)
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)

    def forward(self, clinical):
        # Embedding      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)

        # Classification mlp
        clinical = torch.swapaxes(clinical,0,1)
        clinical = torch.flatten(clinical, 1)
        logits = F.relu(self.linear(clinical))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        
        return logits, clinical, [None]



################################# Genetics ablation model #################################
class gen_abl(nn.Module): # example from fairprs
    def __init__(self, config):
        super(gen_abl, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.gen_embd = nn.Linear(4,int(self.d_model/2))
        self.gen_chr_embd = nn.Embedding(22,int(self.d_model/2))

        # Transformer encoders
        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)

        # MLP
        self.linear = nn.Linear(self.d_model*70,self.units)
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)

    def forward(self, gen, gene):
        # Embedding       
        gen = F.relu(self.gen_embd(gen))
        gene = self.gen_chr_embd(gene)
        gen = torch.cat((gen,gene),dim=2)

        # Swap axis for batches in following transformer opertions
        gen =  torch.swapaxes(gen,0,1)

        # Single mod branch encoding
        gen = self.gen_transformer_encoder(gen)

        # Classification mlp
        gen = torch.swapaxes(gen,0,1)
        gen = torch.flatten(gen, 1)
        logits = F.relu(self.linear(gen))
        logits = self.dropout(logits)
        logits = self.logits(logits)
        
        return logits, gen, [None]



################################# Clinical ablation model #################################
class clin_abl_pool(nn.Module): # example from fairprs
    def __init__(self, config):
        super(clin_abl_pool, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.maxpool_flag = config['maxpool_flag']


        self.linears = nn.ModuleList()

        ## Single modality branch encoding
        # Initial embedding
        self.clin_embd = nn.Linear(1,self.d_model)

        # Transformer encoders
        self.clinical_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.clinical_transformer_encoder = nn.TransformerEncoder( self.clinical_encoder_tf_layer, num_layers=self.num_layers)

        # MLP
        self.max_pooling_layer = Reduce('s b d -> b d', 'max')
        # self.linear = nn.Linear(self.d_model*11,self.units)
        # self.linear = nn.Linear(self.d_model*7,self.units)
        if self.maxpool_flag:
            self.linear = nn.Linear(self.d_model,self.units) # for concat
        else:
            self.linear = nn.Linear(self.d_model*7,self.units) # for concat
        self.dropout = nn.Dropout(0.10)
        self.logits = nn.Linear(self.units,3)

    def forward(self, clinical):
        # Embedding      
        clinical = F.relu(self.clin_embd(torch.unsqueeze(clinical,-1)))

        # Swap axis for batches in following transformer opertions
        clinical = torch.swapaxes(clinical,0,1)

        # Single mod branch encoding
        clinical = self.clinical_transformer_encoder(clinical)

        # Classification mlp
        if self.maxpool_flag:
            clinical = F.relu(self.max_pooling_layer(clinical))
        else:
            clinical = torch.swapaxes(clinical,0,1)
            clinical = torch.flatten(clinical, 1)
        logits = F.relu(self.linear(clinical))
        logits = self.dropout(logits)
        logits = self.logits(logits)

        return logits, clinical, [None]