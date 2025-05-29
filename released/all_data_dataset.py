import pandas as pd
import numpy as np
import os
from functools import reduce
from numpy import squeeze
from torch.utils.data import Dataset
from torch import Tensor, vstack, cat, mean, std, torch
from utils import wgs_cleaner, select_snps, gene_rep, filter_and_imput, clean_pat_wgs_id_ADNI, filter_and_impute_img
from img_tokenization import preprocess_img
from gen_tokenization import gene_tokenization


class all_data_dataset(Dataset):

    def load_and_process_gen(self):
        ### Genotype ###
        # Load GWAS -> load betas
        # NOTE: GWAS files columns are: SNPS,BETA,REGION,CHR_ID,CHR_POS,DISEASE/TRAIT,MAPPED_GENE
        self.df_gwas = pd.read_csv(os.path.join(self.path_gwas,self.fn_gwas))
        # FIXME: right now just dropping duplicate SNPs betas, need to choose the right GWAS or calculate an independet GWAS for this project
        self.df_gwas = self.df_gwas.drop_duplicates(subset=['SNV'])
        self.betas = self.df_gwas.loc[:,['SNV','OR']]
        self.betas = self.betas.set_index('SNV')
        # Only applies if we use the full GWAS for neurodeg disease, if we use the short version not needed
        # FIXME: right now just dropping SNPs without mapped gene -> probably need to assign some alternative placeholder
        self.df_gwas = self.df_gwas[self.df_gwas['Gene'].notna()]

        # Load SNPS
        self.df_snps = pd.read_csv(os.path.join(self.path_snp,self.fn_snp),delim_whitespace=True, dtype='str')
        self.df_snps = wgs_cleaner(self.df_snps)

        # Extract patient ids to match with other modalities
        self.gen_pat_ids = self.df_snps.index.values

        # Load gene data
        self.genes = self.df_gwas.loc[:,['SNV','Gene']]
        self.genes = self.genes.set_index('SNV')

        # Keep only SNPs that we have Betas for
        self.matching_snps = self.df_snps.columns[2:].intersection(self.betas.index)
        # self.df_snps = self.df_snps.loc[:,pd.concat((pd.Series(['PATNO','PHENOTYPE']),pd.Series(self.matching_snps)))] # replaced by next line. Removed the PHENTOYPE column and made PATNO the index 2/10/2023
        self.df_snps = self.df_snps.loc[:,pd.Series(self.matching_snps)]

        # Parition SNPs per gene & Weighted sum per gene
        # TODO: Find more efficient way than for loop. Probaly can do chunks for SNPS- ID in one pass.
        # gene_reps = {}
        # for gene in self.genes.Gene:
        #     snplist = select_snps(self.genes,gene)
        #     gene_reps[gene] = squeeze(gene_rep(self.df_snps,self.betas,snplist))
        # TODO: concatenate gene reps back into a single DF -> rows: patients, cols: genes
        # self.encoded_genes_pat = pd.DataFrame.from_dict(gene_reps, orient='columns')
        # TODO: Make sure that the patient / row order was preserved -> currently just assuming order is kept and adding pat ID as index of DF
        # self.encoded_genes_pat.index = self.df_snps.index

        # Drop columns if all values are nan
        # self.encoded_genes_pat = self.encoded_genes_pat.dropna(axis=1, how='all')

        # # Return as tensors instead of pandas DF
        # self.encoded_genes_pat_tensor = Tensor(self.encoded_genes_pat.values)


        self.gene_data_idx, self.encoded_snps, self.genes_or_chromosome = gene_tokenization(self.df_snps,self.df_gwas)
        assert np.array_equal(self.gen_pat_ids, self.gene_data_idx)
        self.gene_data_idx_df = pd.DataFrame(zip(self.gene_data_idx,range(len(self.gene_data_idx))), columns = ['PTID','Tensor_idx'])
        self.gene_data_idx_df = self.gene_data_idx_df.set_index('PTID')

        

    def load_and_process_img(self):
        ### Imaging ###
        # Read data
        self.df = pd.read_csv(os.path.join(self.path,self.fn), dtype='str')
        self.img_feat_nm = np.loadtxt(os.path.join(self.path_img_feat,self.fn_img_feat), dtype= 'str').tolist()
        self.roi_df = pd.read_csv(os.path.join(self.path_feat_dict,self.fn_feat_dict), dtype='str')

        #Extract IDs
        # TODO: if doing imputation/ dropping rows below this is a potential problem
        # self.ids = self.df['PTID']
        # For the moment better to change PTID to index instead - cleaning indices to match clin_datset format
        self.df['PTID']
        self.df['PTID'] = self.df.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        self.df = self.df.set_index('PTID')

        # Filter cols / vars of interest at selected visits
        # TODO: filter and impute -> drop all columns that are all nas
        # self.df_filt = filter_and_impute_img(self.df, 'bl' , self.visit_colnm,  self.img_feat_nm, True)
        self.img_idx, self.proc_img = preprocess_img(self.df, self.roi_df, 'bl' , self.visit_colnm,  self.img_feat_nm)
        self.img_idx_df = pd.DataFrame(zip(self.img_idx,range(len(self.img_idx))), columns = ['PTID','Tensor_idx'])
        self.img_idx_df = self.img_idx_df.set_index('PTID')


        # self.df_filt = self.df.loc[:,self.cols_interest]

        # self.pat_ids_tensor = Tensor(self.df_filt.index.values.astype('float'))
        # self.df_filt_tensor = Tensor(self.df_filt.values)

    def load_and_process_clinical(self):
        ### Clinical ###
        # load data 
        self.df_full = pd.read_csv(os.path.join(self.path,self.fn))
        
        # Drop rows with NAs on outcome
        for var in self.outcome_vars:
            self.df_full = self.df_full[self.df_full[var].notna()]
        # self.df_full = self.df_full[self.df_full['updrs_totscore'].notna()]
        # self.df_full = self.df_full[self.df_full['fampd_new'].notna()]  
 
        
        # Recoding demo variables in ADNI
        self.df_full.PTID = self.df_full.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        # FIXME: need to doubke check if recoding makes sense
        self.gender_mapping = {'Female':0, 'Male':1} 
        self.ethnicity_mapping = {'Not Hisp/Latino': 0 , 'Hisp/Latino': 1, 'Unknown':2}
        self.race_mapping = {'White': 0, 'Black': 1 , 'Asian': 2, 'More than one': 3, 'Am Indian/Alaskan':4, 'Unknown':4, 'Hawaiian/Other PI': 4}
        self.dx_mapping = {'CN': 0, 'AD': 1 , 'EMCI': 2, 'LMCI': 2, 'SMC':4, 'Dementia':1, 'MCI':2}

        self.df_full.PTGENDER = self.df_full.PTGENDER.map(self.gender_mapping)
        self.df_full.PTETHCAT = self.df_full.PTETHCAT.map(self.ethnicity_mapping)
        self.df_full.PTRACCAT = self.df_full.PTRACCAT.map(self.race_mapping)
        self.df_full.DX_bl = self.df_full.DX_bl.map(self.dx_mapping)
        self.df_full.DX = self.df_full.DX.map(self.dx_mapping)
        
        # Filter cols / vars of interest at selected visits
        # FIXME: Currently easy fix with different filter and imput systems for PPMI and ADNI
        # if 'BL' in self.visits:
        #     df_bl_demo, df_bl_x, df_bl_y  = filter_and_imput(self.df_full,'BL',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute)
        # if 'V04' in self.visits:
        #     df_v4_demo, df_v4_x, df_v4_y  = filter_and_imput(self.df_full,'V04',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute)
        # if 'V06' in self.visits:
        #     df_v6_demo, df_v6_x, df_v6_y  = filter_and_imput(self.df_full,'V06',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute)
        # if 'V08' in self.visits:
        #     df_v8_demo, df_v8_x, df_v8_y  = filter_and_imput(self.df_full,'V08',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute)
        # if 'V10' in self.visits:
        #     df_v10_demo, df_v10_x, df_v10_y  = filter_and_imput(self.df_full,'V10',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute)

        if 'bl' in self.visits:
            self.df_bl_demo, self.df_bl_x, self.df_bl_y  = filter_and_imput(self.df_full,'bl',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm06' in self.visits:
            self.df_v4_demo, self.df_v4_x, self.df_v4_y  = filter_and_imput(self.df_full,'m06',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm12' in self.visits:
            self.df_v6_demo, self.df_v6_x, self.df_v6_y  = filter_and_imput(self.df_full,'m12',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm24' in self.visits:
            self.df_v8_demo, self.df_v8_x, self.df_v8_y  = filter_and_imput(self.df_full,'m24',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)

        
        # Extract patient ids to match with other modalities
        # TODO: Double check that filter and impute is giving the right ids.... when doing df_bl_demo.PTID,head(10) -> ids look to large
        self.clin_pat_ids =  self.df_bl_demo.index.values


                # Convert to tensors and append
        # TODO: Select / return data for just some visits and not all -> eg. only BL and v4
        # self.demo = [Tensor(df_bl_demo.values),Tensor(df_v4_demo.values),Tensor(df_v6_demo.values),Tensor(df_v8_demo.values),Tensor(df_v10_demo.values)]
        # self.X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values),Tensor(df_v8_x.values),Tensor(df_v10_x.values)]
        # self.Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values),Tensor(df_v8_y.values),Tensor(df_v10_y.values)]
        self.demo = [Tensor(self.df_bl_demo.values.astype('float')),Tensor(self.df_v4_demo.values.astype('float')),Tensor(self.df_v6_demo.values.astype('float')), Tensor(self.df_v8_demo.values.astype('float'))]
        self.X = [Tensor(self.df_bl_x.values),Tensor(self.df_v4_x.values),Tensor(self.df_v6_x.values), Tensor(self.df_v8_x.values)]
        self.Y = [Tensor(self.df_bl_y.values),Tensor(self.df_v4_y.values),Tensor(self.df_v6_y.values), Tensor(self.df_v8_y.values)]
        
        # # TODO: Return a more optimized strucuture instead of stacking all time points
        # # This will depend / defined based on final architecture implementation
        # self.demo = vstack(self.demo)
        # self.X = vstack(self.X)
        # self.Y = vstack(self.Y)


    def load_labels(self):
        ### Labels ###
        self.df_labels = pd.read_csv(os.path.join(self.path_labels,self.fn_labels), dtype='str')
        self.df_labels['PTID'] = self.df_labels.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        self.df_labels = self.df_labels.set_index('PTID')
        # self.df_labels = self.df_labels['kmeans_cluster_timepoints'] 
        self.df_labels = self.df_labels['DX_bl'] 

        self.df_labels = self.df_labels.map({'AD':0,'LMCI':1,'EMCI':2})
        ## testing what happens if we remove extreme cases
        # self.df_labels = self.df_labels.loc[self.df_labels.isin(['1','2'])]
        # self.df_labels = self.df_labels.map({'1': 0, '2': 1 })


    def match_modalities(self):
        ### All modalities ###
        # Match ids and select only the intersecting ones
        # Note: clin_bl and img overlap = 1548 | clin_bl and gen  = ?? | gen and img = 773
        # Sort dfs to make sure index in the same order accross all datasets
        # self.matching_ids = reduce(np.intersect1d,(self.df_filt.index.values, self.clin_pat_ids, self.gen_pat_ids, self.df_labels.index.values))
        if self.use_cluster_flag:
            self.matching_ids = reduce(np.intersect1d,(self.img_idx, self.clin_pat_ids, self.gen_pat_ids, self.df_labels.index.values))
        else:
            self.matching_ids = reduce(np.intersect1d,(self.img_idx, self.clin_pat_ids, self.gen_pat_ids))

        # self.encoded_genes_pat = self.encoded_genes_pat.loc[self.matching_ids].sort_index()
        # self.df_filt = self.df_filt.loc[self.matching_ids].sort_index()
        # New after tokenization
        self.gene_data_idx_df = self.gene_data_idx_df.loc[self.matching_ids].sort_index()
        self.encoded_snps = self.encoded_snps[self.gene_data_idx_df.Tensor_idx.values]
        self.genes_or_chromosome = self.genes_or_chromosome[self.gene_data_idx_df.Tensor_idx.values]
        self.img_idx_df = self.img_idx_df.loc[self.matching_ids].sort_index()
        self.proc_img = self.proc_img[self.img_idx_df.Tensor_idx.values]
        self.df_bl_demo = self.df_bl_demo.loc[self.matching_ids].sort_index()
        self.df_bl_x = self.df_bl_x.loc[self.matching_ids].sort_index()
        self.df_bl_y = self.df_bl_y.loc[self.matching_ids].sort_index()
        

        # Return as tensors instead of pandas DF
        # self.encoded_genes_pat_tensor = Tensor(self.encoded_genes_pat.values)
        # self.pat_ids_tensor = Tensor(self.df_filt.index.values.astype('float'))
        # self.df_filt_tensor = Tensor(self.df_filt.values)

        self.pat_ids_tensor = self.matching_ids.astype('float')
        self.demo_dataset, self.clinical_dataset = Tensor(self.df_bl_demo.values.astype('float')), Tensor(self.df_bl_x.values)

        if self.use_cluster_flag:
            self.df_labels = self.df_labels.loc[self.matching_ids].sort_index()
            self.target_dataset = Tensor(self.df_labels.values.astype('int')) # Use clusters as targets instead!!
        else:
            self.target_dataset = Tensor(self.df_bl_y.values[:,-1])
        

    def set_trainset_mean_std(self,train_index):
        self.mean_trainset_img = mean(torch.flatten(self.proc_img, start_dim=1), dim = 0)
        self.std_trainset_img = std(torch.flatten(self.proc_img, start_dim=1), dim = 0)
        # self.mean_trainset_img = mean(self.df_filt_tensor[train_index], dim = 0) 
        # self.std_trainset_img = std(self.df_filt_tensor[train_index], dim = 0)
        self.mean_trainset_clinical = mean(self.clinical_dataset[train_index], dim = 0) 
        self.std_trainset_clinical = std(self.clinical_dataset[train_index], dim = 0)
        # self.mean_trainset_gen = mean(self.encoded_genes_pat_tensor[train_index], dim = 0) 
        # self.std_trainset_gen = std(self.encoded_genes_pat_tensor[train_index], dim = 0)

    def normalize_data(self, train_index, val_index, test_index):
        # Normalize each feature column by substracting mean and dividing by sd
        # Set clones to not overwritte original data
        # self.df_filt_tensor_normalized = self.df_filt_tensor.clone()
        self.proc_img_normalized = self.proc_img.clone()
        self.clinical_dataset_normalized = self.clinical_dataset.clone()
        # self.encoded_genes_pat_tensor_normalized = self.encoded_genes_pat_tensor.clone()
        # train set
        # self.df_filt_tensor_normalized[train_index] = (self.df_filt_tensor[train_index]-self.mean_trainset_img)/self.std_trainset_img #old
        self.proc_img_normalized[train_index] = ((torch.flatten(self.proc_img[train_index], start_dim=1)-self.mean_trainset_img)/self.std_trainset_img).reshape(len(train_index),72,4)
        self.clinical_dataset_normalized[train_index] = (self.clinical_dataset[train_index]-self.mean_trainset_clinical)/self.std_trainset_clinical
        # self.encoded_genes_pat_tensor_normalized[train_index] = (self.encoded_genes_pat_tensor[train_index]-self.mean_trainset_gen)/self.std_trainset_gen
        # val set
        # self.df_filt_tensor_normalized[val_index] = (self.df_filt_tensor[val_index]-self.mean_trainset_img)/self.std_trainset_img
        self.proc_img_normalized[val_index] = ((torch.flatten(self.proc_img[val_index], start_dim=1)-self.mean_trainset_img)/self.std_trainset_img).reshape(len(val_index),72,4)
        self.clinical_dataset_normalized[val_index] = (self.clinical_dataset[val_index]-self.mean_trainset_clinical)/self.std_trainset_clinical
        # self.encoded_genes_pat_tensor_normalized[val_index] = (self.encoded_genes_pat_tensor[val_index]-self.mean_trainset_gen)/self.std_trainset_gen
        # test set
        # self.df_filt_tensor_normalized[test_index] = (self.df_filt_tensor[test_index]-self.mean_trainset_img)/self.std_trainset_img
        self.proc_img_normalized[test_index] = ((torch.flatten(self.proc_img[test_index], start_dim=1)-self.mean_trainset_img)/self.std_trainset_img).reshape(len(test_index),72,4)
        self.clinical_dataset_normalized[test_index] = (self.clinical_dataset[test_index]-self.mean_trainset_clinical)/self.std_trainset_clinical
        # self.encoded_genes_pat_tensor_normalized[test_index] = (self.encoded_genes_pat_tensor[test_index]-self.mean_trainset_gen)/self.std_trainset_gen
        

    def scale(self, train_index, val_index, test_index):
        # Normalize each feature column to range of [0,1]
        self.df_filt_tensor_normalized_train = self.df_filt_tensor / self.df_filt_tensor.max(0, keepdim=True)[0]
        # Set clones to not overwritte original data
        self.df_filt_tensor_normalized = self.df_filt_tensor.clone()
        self.clinical_dataset_normalized_train = self.clinical_dataset.clone()
        self.encoded_genes_pat_tensor_normalized_train = self.encoded_genes_pat_tensor.clone()
        # train set
        self.df_filt_tensor_normalized[train_index] = self.df_filt_tensor[train_index] / self.df_filt_tensor[train_index].max(0, keepdim=True)[0]
        self.clinical_dataset_normalized[train_index] = self.df_filt_tensor[train_index] / self.df_filt_tensor[train_index].max(0, keepdim=True)[0]
        self.encoded_genes_pat_tensor_normalized[train_index] = self.df_filt_tensor[train_index] / self.df_filt_tensor[train_index].max(0, keepdim=True)[0]
        # val set
        self.df_filt_tensor_normalized[val_index] = self.df_filt_tensor[val_index] / self.df_filt_tensor[val_index].max(0, keepdim=True)[0]
        self.clinical_dataset_normalized[val_index] = self.df_filt_tensor[val_index] / self.df_filt_tensor[val_index].max(0, keepdim=True)[0]
        self.encoded_genes_pat_tensor_normalized[val_index] = self.df_filt_tensor[val_index] / self.df_filt_tensor[val_index].max(0, keepdim=True)[0]
        # test set
        self.df_filt_tensor_normalized[test_index] = self.df_filt_tensor[test_index] / self.df_filt_tensor[test_index].max(0, keepdim=True)[0]
        self.clinical_dataset_normalized[test_index] = self.df_filt_tensor[test_index] / self.df_filt_tensor[test_index].max(0, keepdim=True)[0]
        self.encoded_genes_pat_tensor_normalized[test_index] = self.df_filt_tensor[test_index] / self.df_filt_tensor[test_index].max(0, keepdim=True)[0]

    def tokenize_img_data(self):
        self.img_idx, self.proc_img = preprocess_img(self.df, self.roi_df, 'bl' , self.visit_colnm,  self.img_feat_nm)
        self.img_idx_df = pd.DataFrame(zip(self.img_idx,range(len(self.img_idx))), columns = ['PTID','Tensor_idx'])
        self.img_idx_df = self.img_idx_df.set_index('PTID')

    def get_ptidxs(self):
        return self.pat_ids_tensor

    def get_labels(self, index=[]):
        if len(index) == 0:
            return self.target_dataset
        else:
            return self.target_dataset[index]

    def __init__(self,paths, args):
        '''
        self.mean=None
        self.std=None
        '''
        # Set paths and args variables
        self.path_gwas,self.fn_gwas,self.path_snp,self.fn_snp = paths['path_gwas'], paths['fn_gwas'],paths['path_snp'],paths['fn_snp']
        self.path,self.fn = paths['path_clin'],paths['fn_clin']
        self.demo_vars,self.clin_vars,self.outcome_vars,self.visits,self.impute,self.vis_colnm = args['demo_vars'],args['clin_vars'],args['outcome_vars'],args['visits'],args['impute'],args['visit_colnm']
        self.path, self.fn, self.path_img_feat, self.fn_img_feat, self.path_feat_dict, self.fn_feat_dict, self.visit_colnm = paths['path_img'],paths['fn_img'],paths['path_img_feat'],paths['fn_img_feat'],paths['path_feat_dict'],paths['fn_feat_dict'],args['visit_colnm']
        self.path_labels, self.fn_labels = paths['path_labels'],paths['fn_labels']
        self.use_cluster_flag = args['use_cluster_flag']


        self.load_and_process_img()
        self.load_and_process_gen()
        self.load_and_process_clinical()
        self.load_labels()
        self.match_modalities()


    def __len__(self):
        # TODO: return correct len based on selected visits
        return len(self.demo_dataset)

    def __getitem__(self, index):
        return self.pat_ids_tensor[index], self.demo_dataset[index],  self.clinical_dataset_normalized[index], self.target_dataset[index], self.encoded_snps[index], self.proc_img_normalized[index], self.genes_or_chromosome[index],
    