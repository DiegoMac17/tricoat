import os
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from utils import filter_and_impute_img, clean_pat_wgs_id_ADNI


class img_dataset(Dataset):

    """
    Load imaging features from csv file (ADNI - TADPOLE) 
    """
    def __init__(self,path,fn,cols_interest, visit_colnm, impute):
        self.path, self.fn, self.cols_interest, self.visit_colnm,  self.impute = path, fn, cols_interest, visit_colnm, impute
        
        # Read data
        self.df = pd.read_csv(os.path.join(self.path,self.fn))

        #Extract IDs
        # TODO: if doing imputation/ dropping rows below this is a potential problem
        # self.ids = self.df['PTID']
        # For the moment better to change PTID to index instead - cleaning indices to match clin_datset format
        self.df['PTID']
        self.df['PTID'] = self.df.PTID.apply(lambda x: clean_pat_wgs_id_ADNI(x))
        self.df = self.df.set_index('PTID')

        # Filter cols / vars of interest at selected visits
        # self.df_filt = filter_and_impute_img(self.df,  self.visit_colnm,  self.impute)
        self.df_filt = self.df.loc[:,cols_interest]

        self.pat_ids_tensor = Tensor(self.df_filt.index.values.astype('float'))
        self.df_filt_tensor = Tensor(self.df_filt.values)

    def __len__(self):
        # TODO: return correct len based on selected visits
        return len(self.pat_ids_tensor)

    def __getitem__(self, index):
        return self.pat_ids_tensor[index], self.df_filt_tensor[index]

# For dfmri dataset
    #     self.pat_ids = self.df.rename(columns={'Measure:volume':'ID'}).ID.str.split("-",expand=True)[0]

    #     cols_interest = ['Measure:volume', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
    #    'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
    #    'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
    #    '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus',
    #    'Left-Amygdala', 'CSF', 'Left-Accumbens-area', 'Left-VentralDC',
    #    'Left-vessel', 'Left-choroid-plexus', 'Right-Lateral-Ventricle',
    #    'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter',
    #    'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate',
    #    'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
    #    'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC',
    #    'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle',
    #    'WM-hypointensities', 'Left-WM-hypointensities',
    #    'Right-WM-hypointensities', 'non-WM-hypointensities',
    #    'Left-non-WM-hypointensities', 'Right-non-WM-hypointensities',
    #    'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central',
    #    'CC_Mid_Anterior', 'CC_Anterior', 'lhCortexVol', 'rhCortexVol',
    #    'CortexVol', 'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
    #    'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
    #    'SupraTentorialVol', 'IntraCranialVol', 'lh.aparc.volume',
    #    'lh_bankssts_volume', 'lh_caudalanteriorcingulate_volume',
    #    'lh_caudalmiddlefrontal_volume', 'lh_cuneus_volume',
    #    'lh_entorhinal_volume', 'lh_fusiform_volume',
    #    'lh_inferiorparietal_volume', 'lh_inferiortemporal_volume',
    #    'lh_isthmuscingulate_volume', 'lh_lateraloccipital_volume',
    #    'lh_lateralorbitofrontal_volume', 'lh_lingual_volume',
    #    'lh_medialorbitofrontal_volume', 'lh_middletemporal_volume',
    #    'lh_parahippocampal_volume', 'lh_paracentral_volume',
    #    'lh_parsopercularis_volume', 'lh_parsorbitalis_volume',
    #    'lh_parstriangularis_volume', 'lh_pericalcarine_volume',
    #    'lh_postcentral_volume', 'lh_posteriorcingulate_volume',
    #    'lh_precentral_volume', 'lh_precuneus_volume',
    #    'lh_rostralanteriorcingulate_volume', 'lh_rostralmiddlefrontal_volume',
    #    'lh_superiorfrontal_volume', 'lh_superiorparietal_volume',
    #    'lh_superiortemporal_volume', 'lh_supramarginal_volume',
    #    'lh_frontalpole_volume', 'lh_temporalpole_volume',
    #    'lh_transversetemporal_volume', 'lh_insula_volume', 'rh.aparc.volume',
    #    'rh_bankssts_volume', 'rh_caudalanteriorcingulate_volume',
    #    'rh_caudalmiddlefrontal_volume', 'rh_cuneus_volume',
    #    'rh_entorhinal_volume', 'rh_fusiform_volume',
    #    'rh_inferiorparietal_volume',  'rh_inferiortemporal_volume',
    #    'rh_isthmuscingulate_volume', 'rh_lateraloccipital_volume',
    #    'rh_lateralorbitofrontal_volume', 'rh_lingual_volume',
    #    'rh_medialorbitofrontal_volume', 'rh_middletemporal_volume',
    #    'rh_parahippocampal_volume', 'rh_paracentral_volume',
    #    'rh_parsopercularis_volume', 'rh_parsorbitalis_volume',
    #    'rh_parstriangularis_volume', 'rh_pericalcarine_volume',
    #    'rh_postcentral_volume', 'rh_posteriorcingulate_volume',
    #    'rh_precentral_volume', 'rh_precuneus_volume',
    #    'rh_rostralanteriorcingulate_volume', 'rh_rostralmiddlefrontal_volume',
    #    'rh_superiorfrontal_volume', 'rh_superiorparietal_volume',
    #    'rh_superiortemporal_volume', 'rh_supramarginal_volume',
    #    'rh_frontalpole_volume', 'rh_temporalpole_volume',
    #    'rh_transversetemporal_volume', 'rh_insula_volume']