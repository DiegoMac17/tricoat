import pandas as pd
import numpy as np
from torch import Tensor, stack

def gene_tokenization(snp_df,snp_info_df):
    #
    snp_info_df = snp_info_df.set_index('SNV')
    snp_shared = np.intersect1d(snp_df.columns, snp_info_df.index)
    snp_info_df = snp_info_df[snp_info_df.index.isin(snp_shared)]
    idx = snp_df.index.values
    snp_df = snp_df.transpose()

    snps = []
    genes = []
    for i in range(len(snp_df.columns)):
        pat_tokenized = snp_info_df[['OR','Intergenic','Frequency']].copy()
        pat_tokenized.insert(loc=0, column = 'allele_dosage', value = snp_df.iloc[:,i])

        snps.append(Tensor(pat_tokenized.astype('float').values))
        genes.append(Tensor(snp_info_df['Chr'].astype('int').values))


    snps = stack(snps)
    genes = stack(genes)
    

    return idx,snps, genes
































# path = '../../../../bulk/machad/ADNI/imaging/TADPOLE_D1_D2.csv'
# path_img_feat = '../../../../bulk/machad/ADNI/adni_img_feat_names_crossectional.txt'
# feat_dict_path = '../../../../bulk/machad/ADNI/TADPOLE_D1_D2_Dict.csv'
# vis_colnm = 'VISCODE'
# visit = 'bl'




# df = pd.read_csv(path, dtype='str')
# roi_df = pd.read_csv(feat_dict_path, dtype='str')
# img_feat_nm = np.loadtxt(path_img_feat, dtype= 'str').tolist()



# df = df.loc[df[vis_colnm] == visit,:]

# df = df[df[vis_colnm].notna()]

# # Filter cols / vars of interest
# df = df.loc[:,img_feat_nm]


# # Replace empty rows that have ' ' for nan
# df = df.replace(r'^\s*$', np.nan, regex=True)

# # Drop columns with all na
# # df = df.dropna(axis=1, how='all')

# # Filter for only Cross-Sectional analysis
# roi_df = roi_df[roi_df['FLDNAME'].isin(img_feat_nm)]
# roi_df = roi_df.loc[roi_df['CRFNAME'] == 'Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)',:]


# # Parse regions of interest
# roi_df['ROI'] = roi_df['TEXT'].str.split('of',expand=True).iloc[:,1]
# roi_df['Trait'] = roi_df['TEXT'].str.split('of',expand=True).iloc[:,0]
# roi_df['Trait'] =  roi_df['Trait'].replace(r"^ +| +$", r"", regex=True)
# roi_df = roi_df.set_index('FLDNAME')
# roi_shared = roi_df.loc[roi_df['Trait'] == 'Cortical Thickness Average','ROI']
# df = df.transpose()
# # roi_df = roi_df[roi_df.index.isin(df.index)]

# def per_pat_token(feat_dict_df, pat_values):

#     feat_dict_df['Values'] = pat_values
#     return pd.pivot(feat_dict_df, index = 'ROI', columns = 'Trait', values = 'Values').loc[:,['Cortical Thickness Average', 'Cortical Thickness Standard Deviation', 'Surface Area', 'Volume (Cortical Parcellation)']]

# out = []
# for i in range(len(df.columns)):
#     pat_tokenized = per_pat_token(roi_df[['ROI', 'Trait']].copy(), df.iloc[:,i])
#     pat_tokenized = pat_tokenized[pat_tokenized.index.isin(roi_shared)]
#     pat_tokenized = pat_tokenized.fillna(pat_tokenized.astype('float').mean())
#     # pat_tokenized = pat_tokenized.dropna(axis=1, how='all')
#     # out.append(pat_tokenized.isna().sum().sum())

#     out.append(Tensor(pat_tokenized.astype('float').values))


# out = stack(out)



# # roi_df = roi_df[roi_df['ROI'].isin(roi_shared)]


# # roi_df.groupby('ROI')



# def filter_and_impute_img(df_full,visit,vis_colnm,cols_interest,impute):
#     # Filter at visit 
#     df_full = df_full.loc[df_full[vis_colnm] == visit,:]
    
#     # drop column for visit ID
# #     df_full = df_full.drop('EVENT_ID', axis =1)
#     df_full = df_full[df_full[vis_colnm].notna()]
    
#     # Filter cols / vars of interest
#     df_full = df_full.loc[:,cols_interest]

#     # Replace empty rows that have ' ' for nan
#     df_full = df_full.replace(r'^\s*$', np.nan, regex=True)

#     # Drop columns if all values are nan
#     df_full = df_full.dropna(axis=1, how='all')
#     # Impute
#     if impute:
#         df_full_imp = imputer(df_full)
#     # Reset index
#     df_full_imp.index = df_full.index
#     return df_full_imp

# def imputer(df):
#     # TODO: implement alternative imputers (mean/median)
#     # Impute and fill missing values using MICE (sklearn implementation)
#     mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending', keep_empty_features = True)
#     # Temporarily filling columns with all missing values using 0s, so that number of columns is preserved.
#     imp = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
#     return imp