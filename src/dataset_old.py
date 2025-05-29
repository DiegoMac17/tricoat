import pandas as pd
import numpy as np
import os
from functools import reduce
from numpy import squeeze
from torch.utils.data import Dataset
from torch import Tensor, vstack, cat
from utils import wgs_cleaner, select_snps, gene_rep, filter_and_imput, clean_pat_wgs_id_ADNI, filter_and_impute_img


class all_data_dataset(Dataset):
    # def set_norm_param(self, mean, std):
    #     self.mean = mean
    #     self.std = std
    def __init__(self,paths, args):
        '''
        self.mean=None
        self.std=None
        '''
        # Set paths and args variables
        self.path_gwas,self.fn_gwas,self.path_snp,self.fn_snp = paths['path_gwas'], paths['fn_gwas'],paths['path_snp'],paths['fn_snp']
        self.path,self.fn = paths['path_clin'],paths['fn_clin']
        self.demo_vars,self.clin_vars,self.outcome_vars,self.visits,self.impute,self.vis_colnm = args['demo_vars'],args['clin_vars'],args['outcome_vars'],args['visits'],args['impute'],args['visit_colnm']
        self.path, self.fn, self.path_img_feat, self.fn_img_feat, self.visit_colnm = paths['path_img'],paths['fn_img'],paths['path_img_feat'],paths['fn_img_feat'],args['visit_colnm']
        self.path_labels, self.fn_labels = paths['path_labels'],paths['fn_labels']

        ### Genotype ###
        # Load GWAS -> load betas
        # NOTE: GWAS files columns are: SNPS,BETA,REGION,CHR_ID,CHR_POS,DISEASE/TRAIT,MAPPED_GENE
        self.df_gwas = pd.read_csv(os.path.join(self.path_gwas,self.fn_gwas))
        # FIXME: right now just dropping duplicate SNPs betas, need to choose the right GWAS or calculate an independet GWAS for this project
        self.df_gwas = self.df_gwas.drop_duplicates(subset=['SNPS'])
        self.betas = self.df_gwas.loc[:,['SNPS','BETA']]
        self.betas = self.betas.set_index('SNPS')
        # FIXME: right now just dropping SNPs without mapped gene -> probably need to assign some alternative placeholder
        self.df_gwas = self.df_gwas[self.df_gwas['MAPPED_GENE'].notna()]

        # Load SNPS
        self.df_snps = pd.read_csv(os.path.join(self.path_snp,self.fn_snp),delim_whitespace=True, dtype='str')
        self.df_snps = wgs_cleaner(self.df_snps)


        # Extract patient ids to match with other modalities
        self.gen_pat_ids = self.df_snps.index.values

        # Load gene data
        self.genes = self.df_gwas.loc[:,['SNPS','MAPPED_GENE']]
        self.genes = self.genes.set_index('SNPS')

        # Keep only SNPs that we have Betas for
        self.matching_snps = self.df_snps.columns[2:].intersection(self.betas.index)
        # self.df_snps = self.df_snps.loc[:,pd.concat((pd.Series(['PATNO','PHENOTYPE']),pd.Series(self.matching_snps)))] # replaced by next line. Removed the PHENTOYPE column and made PATNO the index 2/10/2023
        self.df_snps = self.df_snps.loc[:,pd.Series(self.matching_snps)]

        # Parition SNPs per gene & Weighted sum per gene
        # TODO: Find more efficient way than for loop. Probaly can do chunks for SNPS- ID in one pass.
        gene_reps = {}
        for gene in self.genes.MAPPED_GENE:
            snplist = select_snps(self.genes,gene)
            gene_reps[gene] = squeeze(gene_rep(self.df_snps,self.betas,snplist))

        # TODO: concatenate gene reps back into a single DF -> rows: patients, cols: genes
        self.encoded_genes_pat = pd.DataFrame.from_dict(gene_reps, orient='columns')
        # TODO: Make sure that the patient / row order was preserved -> currently just assuming order is kept and adding pat ID as index of DF
        self.encoded_genes_pat.index = self.df_snps.index

        # # Return as tensors instead of pandas DF
        # self.encoded_genes_pat_tensor = Tensor(self.encoded_genes_pat.values)

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

        self.df_full.PTGENDER = self.df_full.PTGENDER.map(self.gender_mapping)
        self.df_full.PTETHCAT = self.df_full.PTETHCAT.map(self.ethnicity_mapping)
        self.df_full.PTRACCAT = self.df_full.PTRACCAT.map(self.race_mapping)
        
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
            df_bl_demo, df_bl_x, df_bl_y  = filter_and_imput(self.df_full,'bl',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm06' in self.visits:
            df_v4_demo, df_v4_x, df_v4_y  = filter_and_imput(self.df_full,'m06',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm12' in self.visits:
            df_v6_demo, df_v6_x, df_v6_y  = filter_and_imput(self.df_full,'m12',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)
        if 'm24' in self.visits:
            df_v8_demo, df_v8_x, df_v8_y  = filter_and_imput(self.df_full,'m24',self.demo_vars,self.clin_vars,self.outcome_vars,self.impute,self.vis_colnm)

        
        # Extract patient ids to match with other modalities
        # TODO: Double check that filter and impute is giving the right ids.... when doing df_bl_demo.PTID,head(10) -> ids look to large
        self.clin_pat_ids =  df_bl_demo.index.values


        # Convert to tensors and append
        # TODO: Select / return data for just some visits and not all -> eg. only BL and v4
        # self.demo = [Tensor(df_bl_demo.values),Tensor(df_v4_demo.values),Tensor(df_v6_demo.values),Tensor(df_v8_demo.values),Tensor(df_v10_demo.values)]
        # self.X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values),Tensor(df_v8_x.values),Tensor(df_v10_x.values)]
        # self.Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values),Tensor(df_v8_y.values),Tensor(df_v10_y.values)]
        self.demo = [Tensor(df_bl_demo.values.astype('float')),Tensor(df_v4_demo.values.astype('float')),Tensor(df_v6_demo.values.astype('float')), Tensor(df_v8_demo.values.astype('float'))]
        self.X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values), Tensor(df_v8_x.values)]
        self.Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values), Tensor(df_v8_y.values)]
        
        # # TODO: Return a more optimized strucuture instead of stacking all time points
        # # This will depend / defined based on final architecture implementation
        # self.demo = vstack(self.demo)
        # self.X = vstack(self.X)
        # self.Y = vstack(self.Y)

        ### Imaging ###
        # Read data
        self.df = pd.read_csv(os.path.join(self.path,self.fn), dtype='str')
        self.img_feat_nm = np.loadtxt(os.path.join(self.path_img_feat,self.fn_img_feat), dtype= 'str').tolist()

        #Extract IDs
        # TODO: if doing imputation/ dropping rows below this is a potential problem
        # self.ids = self.df['PTID']
        # For the moment better to change PTID to index instead - cleaning indices to match clin_datset format
        self.df['PTID']
        self.df['PTID'] = self.df.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        self.df = self.df.set_index('PTID')

        # Filter cols / vars of interest at selected visits
        # TODO: filter and impute -> drop all columns that are all nas
        self.df_filt = filter_and_impute_img(self.df, 'bl' , self.visit_colnm,  self.img_feat_nm, self.impute)
        # self.df_filt = self.df.loc[:,self.cols_interest]

        # self.pat_ids_tensor = Tensor(self.df_filt.index.values.astype('float'))
        # self.df_filt_tensor = Tensor(self.df_filt.values)


        ### Labels ###
        self.df_labels = pd.read_csv(os.path.join(self.path_labels,self.fn_labels), dtype='str')
        self.df_labels['PTID'] = self.df_labels.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        self.df_labels = self.df_labels.set_index('PTID')
        self.df_labels = self.df_labels['kmeans_cluster_timepoints'] 


        ### All modalities ###
        # Match ids and select only the intersecting ones
        # Note: clin_bl and img overlap = 1548 | clin_bl and gen  = ?? | gen and img = 773
        # Sort dfs to make sure index in the same order accross all datasets
        self.matching_ids = reduce(np.intersect1d,(self.df_filt.index.values, self.clin_pat_ids, self.gen_pat_ids, self.df_labels.index.values))

        self.encoded_genes_pat = self.encoded_genes_pat.loc[self.matching_ids].sort_index()
        self.df_filt = self.df_filt.loc[self.matching_ids].sort_index()
        df_bl_demo = df_bl_demo.loc[self.matching_ids].sort_index()
        df_bl_x = df_bl_x.loc[self.matching_ids].sort_index()
        df_bl_y = df_bl_y.loc[self.matching_ids].sort_index()
        self.df_labels = self.df_labels.loc[self.matching_ids].sort_index()

        # Return as tensors instead of pandas DF
        self.encoded_genes_pat_tensor = Tensor(self.encoded_genes_pat.values)

        self.pat_ids_tensor = Tensor(self.df_filt.index.values.astype('float'))
        self.df_filt_tensor = Tensor(self.df_filt.values)

        self.demo_dataset, self.clinical_dataset, self.target_dataset = Tensor(df_bl_demo.values.astype('float')), Tensor(df_bl_x.values), Tensor(df_bl_y.values)
        self.target_dataset = Tensor(self.df_labels.values.astype('int')) # Use clusters as targets instead!!

        # TODO: Return a more optimized strucuture instead of stacking all time points
        # This will depend / defined based on final architecture implementation
        # self.demo = vstack(self.demo)
        # self.X = vstack(self.X)
        # self.Y = vstack(self.Y)

        # Return all preprocessed datasets


    def __len__(self):
        # TODO: return correct len based on selected visits
        return len(self.demo_dataset)

    def __getitem__(self, index):
        return self.pat_ids_tensor[index], self.demo_dataset[index],  self.clinical_dataset[index], self.target_dataset[index], self.encoded_genes_pat_tensor[index], self.df_filt_tensor[index]




    # def __init__(self,genomics_dataset,clincal_dataset,img_dataset):
        
    #     self.demo_dataset = clinical_dataset[0]
    #     self.target_dataset = clinical_dataset[1]
    #     self.clinical_dataset = clinical_dataset[2]
    #     self.genomics_dataset = genomics_dataset
    #     self.img_dataset = img_dataset


    # def __len__(self):
    #     # TODO: return correct len based on selected visits
    #     return len(self.demo_dataset)

    # def __getitem__(self, index):
    #     return self.demo_dataset[index],  self.clinical_dataset[index], self.target_dataset[index], self.gen[index], self.img[index]