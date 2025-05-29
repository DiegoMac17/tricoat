import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch import Tensor, vstack, cat

from utils import filter_and_imput, clean_pat_wgs_id_ADNI

class clinical_dataset(Dataset):
    """
    Load tabular data (demographic, clinical, outcomes of interest (diagnosis, UPDRS, Hohen Yahr))
    """
    def __init__(self,path,fn,demo_vars,clin_vars,outcome_vars,visits,impute,vis_colnm):
        self.path,self.fn,self.demo_vars,self.clin_vars,self.outcome_vars,self.visits,self.impute,self.vis_colnm = path, fn, demo_vars, clin_vars, outcome_vars, visits, impute, vis_colnm
       
        # load data 
        self.df_full = pd.read_csv(os.path.join(self.path,self.fn))
        
        # Drop rows with NAs on outcome
        for var in outcome_vars:
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

        
        # Convert to tensors and append
        # TODO: Select / return data for just some visits and not all -> eg. only BL and v4
        # self.demo = [Tensor(df_bl_demo.values),Tensor(df_v4_demo.values),Tensor(df_v6_demo.values),Tensor(df_v8_demo.values),Tensor(df_v10_demo.values)]
        # self.X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values),Tensor(df_v8_x.values),Tensor(df_v10_x.values)]
        # self.Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values),Tensor(df_v8_y.values),Tensor(df_v10_y.values)]
        self.demo = [Tensor(df_bl_demo.values.astype('float')),Tensor(df_v4_demo.values.astype('float')),Tensor(df_v6_demo.values.astype('float')), Tensor(df_v8_demo.values.astype('float'))]
        self.X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values), Tensor(df_v8_x.values)]
        self.Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values), Tensor(df_v8_y.values)]
        
        # TODO: Return a more optimized strucuture instead of stacking all time points
        # This will depend / defined based on final architecture implementation
        self.demo = vstack(self.demo)
        self.X = vstack(self.X)
        self.Y = vstack(self.Y)

    def __len__(self):
        # TODO: return correct len based on selected visits
        return len(self.demo)

    def __getitem__(self, index):
        return self.demo[index],  self.X[index], self.Y[index]