import pandas as pd
import numpy as np
import os
from torch import Tensor, vstack, cat


# Clinical dataset prep
def clin_data_prep(path,fn,demo_vars,clin_vars,outcome_vars,visits,impute):
    path,fn,demo_vars,clin_vars,outcome_vars,visits,impute = path, fn, demo_vars, clin_vars, outcome_vars, visits, impute
    
    # load data 
    df_full = pd.read_csv(os.path.join(path,fn))
    
    # Drop rows with NAs on outcome
    df_full = df_full[df_full['updrs_totscore'].notna()]
    df_full = df_full[df_full['fampd_new'].notna()]     
    
    # Filter cols / vars of interest at selected visits
    if 'BL' in visits:
        df_bl_demo, df_bl_x, df_bl_y  = filter_and_imput(df_full,'BL',demo_vars,clin_vars,outcome_vars,impute)
    if 'V04' in visits:
        df_v4_demo, df_v4_x, df_v4_y  = filter_and_imput(df_full,'V04',demo_vars,clin_vars,outcome_vars,impute)
    if 'V06' in visits:
        df_v6_demo, df_v6_x, df_v6_y  = filter_and_imput(df_full,'V06',demo_vars,clin_vars,outcome_vars,impute)
    if 'V08' in visits:
        df_v8_demo, df_v8_x, df_v8_y  = filter_and_imput(df_full,'V08',demo_vars,clin_vars,outcome_vars,impute)
    if 'V10' in visits:
        df_v10_demo, df_v10_x, df_v10_y  = filter_and_imput(df_full,'V10',demo_vars,clin_vars,outcome_vars,impute)
    if 'V12' in visits:
        df_v12_demo, df_v12_x, df_v12_y  = filter_and_imput(df_full,'V12',demo_vars,clin_vars,outcome_vars,impute)
    
    # Convert to tensors and append
    # TODO: Select / return data for just some visits and not all -> eg. only BL and v4
    demo = [Tensor(df_bl_demo.values),Tensor(df_v4_demo.values),Tensor(df_v6_demo.values),Tensor(df_v8_demo.values),Tensor(df_v10_demo.values),Tensor(df_v12_demo.values)]
    X = [Tensor(df_bl_x.values),Tensor(df_v4_x.values),Tensor(df_v6_x.values),Tensor(df_v8_x.values),Tensor(df_v10_x.values),Tensor(df_v12_x.values)]
    Y = [Tensor(df_bl_y.values),Tensor(df_v4_y.values),Tensor(df_v6_y.values),Tensor(df_v8_y.values),Tensor(df_v10_y.values),Tensor(df_v12_y.values)]
    
    # TODO: Return a more optimized strucuture instead of stacking all time points
    # This will depend / defined based on final architecture implementation
    demo = vstack(demo)
    X = vstack(X)
    Y = vstack(Y)

    return demo,  X, Y

# Imaging dataset prep
def img_data_prep():


    return 

# Genomics dataset prep

def gen_data_prep(path_gwas,fn_gwas,path_snp,fn_snp):
    path_gwas,fn_gwas,path_snp,fn_snp = path_gwas, fn_gwas,path_snp,fn_snp
    # Load GWAS -> load betas
    # NOTE: GWAS files columns are: SNPS,BETA,REGION,CHR_ID,CHR_POS,DISEASE/TRAIT,MAPPED_GENE
    df_gwas = pd.read_csv(os.path.join(path_gwas,fn_gwas))     
    betas = df_gwas.loc[:,['SNPS','BETA']]
    betas = betas.set_index('SNPS')
    # FIXME: right now just dropping SNPs without mapped gene -> probably need to assign some alternative placeholder
    df_gwas = df_gwas[df_gwas['MAPPED_GENE'].notna()]

        # Load SNPS
    df_snps = pd.read_csv(os.path.join(path_snp,fn_snp),delim_whitespace=True, dtype='str')
    df_snps = wgs_cleaner(df_snps)

    # Load gene data
    genes = df_gwas.loc[:,['SNPS','GENE']]
    genes = genes.set_index('SNPS')

    # Keep only SNPs that we have Betas for
    matching_snps = df_snps.columns[2:].intersection(betas.index)
    df_snps = df_snps.loc[:,pd.concat((pd.Series(['PATNO','PHENOTYPE']),pd.Series(matching_snps)))]

    # Parition SNPs per gene & Weighted sum per gene
    # TODO: Find more efficient way than for loop. Probaly can do chunks for SNPS- ID in one pass.
    gene_reps = {}
    for gene in genes.MAPPED_GENE:
        snplist = select_snps(genes,gene)
        gene_reps[gene] = gene_rep(df_snps,betas,snplist)

    # TODO: concatenate gene reps back into a single DF -> rows: patients, cols: genes
    encoded_genes_pat = pd.DataFrame.from_dict(gene_reps, orient='index')
    # TODO: Make sure that the patient / row order was preserved -> currently just assuming order is kept and adding pat ID as index of DF
    encoded_genes_pat.index = df_snps.PATNO
    
    # Skipping genesets for the momement!
        # Concatenate gene reperesentations into gene sets
                    
        # Pad geneset vectors

    # Pass thorugh dense layer -> not in dataset class but remeber to add in architecture!!

    # Return as tensors instead of pandas DF
    encoded_genes_pat_tensor = Tensor(encoded_genes_pat.values)

    return encoded_genes_pat_tensor