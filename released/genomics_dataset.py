import os
import pandas as pd
from numpy import squeeze
from torch.utils.data import Dataset
from torch import Tensor
from utils import wgs_cleaner, select_snps, gene_rep


class genomics_dataset(Dataset):
    """
    Load genomics data SNPs 
    """
    def __init__(self,path_gwas,fn_gwas,path_snp,fn_snp):
        self.path_gwas,self.fn_gwas,self.path_snp,self.fn_snp = path_gwas, fn_gwas,path_snp,fn_snp
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
        
        # Skipping genesets for the momement!
            # Concatenate gene reperesentations into gene sets
                        
            # Pad geneset vectors

        # Pass thorugh dense layer -> not in dataset class but remeber to add in architecture!!

        # Return as tensors instead of pandas DF
        self.encoded_genes_pat_tensor = Tensor(self.encoded_genes_pat.values)
    
    def __len__(self):
        # TODO: return correct len based on selected visits
        return len(self.encoded_genes_pat_tensor)

    def __getitem__(self, index):
        return self.encoded_genes_pat_tensor[index]