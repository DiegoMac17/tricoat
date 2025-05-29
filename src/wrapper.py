import os, argparse, time
from utils import results_summary, results_display, plot_coattn
from framework import full_exp, single_run





# Parse args
def msg(name=None):
    return ''' MODEL NAME
         >> python wrapper.py -fo ADNI_large -gpu 0 -tune 0 -kfeat 50
         >> python wrapper.py -fo ADNI_img_abl__tune -gpu 5 -tune 0 -ab img -gpu_tr 0.2
         >> python wrapper.py -fo ADNI_clin_abl_tune_new_feat -gpu 5 -tune 1 -ab clin -gpu_tr 0.1
         >> python wrapper.py -fo ADNI_gen_abl_new_token_tune -gpu 5 -tune 1 -ab gen -gpu_tr 0.25
         >> python wrapper.py -fo ADNI_full_abl_new_token_tune -gpu 5 -tune 1 -ab full -gpu_tr 0.5 -kfeat 50
         >> python wrapper.py -fo ADNI_full_abl_new_auxlossandarch_no_tune_lre-5_drop03_k512_units_512_maxpool_noTFend -gpu 1 -tune 0 -ab full -k_dim 512 -u 512 -m_p_f 1 -dp_tf 0.0 -dp_fc 0.3
         >> python wrapper.py -fo ADNI_late_fusion -gpu 7 -tune 0 -ab full -k_dim 256 -u 256 -m_p_f 1 -dp_tf 0.0 -dp_fc 0.3 -alt 0 -st 1
        '''

def parse_arguments():
    parser = argparse.ArgumentParser(usage=msg())
    # Data paths
    parser.add_argument("-pg", "--path_gwas", dest='path_gwas', action='store', help="Enter path for GWAS file", metavar="PG", default='~/fast/pd_subtype/data/ad_genes_OR.csv')
    parser.add_argument("-ps", "--path_snp", dest='path_snp', action='store', help="Enter path for SNP file", metavar="PS", default='../../../../bulk/machad/ADNI/geno/final/adni_wgs_recode.raw')
    parser.add_argument("-pc", "--path_clin", dest='path_clin', action='store', help="Enter path for clinical data file", metavar="PC", default='../../../../bulk/machad/ADNI/ADNIMERGE.csv')
    parser.add_argument("-pi", "--path_img", dest='path_img', action='store', help="Enter path for imaging data file", metavar="PI", default='../../../../bulk/machad/ADNI/imaging/TADPOLE_D1_D2.csv')
    parser.add_argument("-pimft", "--path_img_feat", dest='path_img_feat', action='store', help="Enter path for imaging feature names file", metavar="PIMFT", default='../../../../bulk/machad/ADNI/adni_img_feat_names_crossectional.txt')
    parser.add_argument("-piroi", "--path_feat_dict", dest='path_feat_dict', action='store', help="Enter path for imaging ROI names file", metavar="PIMROIFT", default='../../../../bulk/machad/ADNI/TADPOLE_D1_D2_Dict.csv')
    parser.add_argument("-pl", "--path_labels", dest='path_labels', action='store', help="Enter path for subtyping labels", metavar="PL", default='~/fast/pd_subtype/results/figures/labels_ad_mci.csv')
    parser.add_argument("-pclft", "--path_clin_feat", dest='path_clin_feat', action='store', help="Enter path for clinical feature names file", metavar="PIMFT", default='../../../../bulk/machad/ADNI/adni_clin_feat_names.txt')
    parser.add_argument("-pgenft", "--path_gen_feat", dest='path_gen_feat', action='store', help="Enter path for genetics feature names file", metavar="PIMFT", default='../../../../bulk/machad/ADNI/adni_gen_feat_names.txt')
    parser.add_argument("-proinm", "--path_roi_nm", dest='path_roi_nm', action='store', help="Enter path for ROI names file", metavar="PROINM", default='../../../../bulk/machad/ADNI/adni_img_roi_names.txt')

    # Results path
    parser.add_argument("-pr", "--path_res", dest='path_res', action='store', help="Output folder", metavar="PR", default='../results') 
    parser.add_argument("-pck", "--path_ckpt", dest='path_ckpt', action='store', help="Model Checkpoint folder", metavar="PCK") 

    # Filenaming format
    parser.add_argument("-fo", "--fname_out_root", dest='fname_out_root', action='store', help='Enter prefix name for output files', metavar='FNAMEROOT', default='ADNI_gradient_explore')

    # Training parameters
    parser.add_argument("-rnd_sed", "--random_seed", dest='random_seed', action='store', help='Enter random seed to be used for data splits', metavar='RNDSEED', default='42')
    parser.add_argument("-rnfl", "--run_full_exp", dest='run_full', action='store', help='Enter 1 if want to run full experimental setting (cross-testing and val), 0 for single run.', metavar='RNFLL', default='1')
    parser.add_argument("-iters", "--iterations", dest='iters', action='store', help='Enter number of iterations to run full pipeline', metavar='ITERS', default='1')
    parser.add_argument("-tf", "--trait", dest='trait_flag', action='store', help="Enter the trait flag -- 1 for binary, 0 for continuous", metavar="TR", default='1')
    parser.add_argument("-gpu", "--gpu_nums", dest='gpu_nums', action='store', help="Enter the gpus to use", metavar="GPU", default='7')
    parser.add_argument("-tune", "--tune_flag", dest='tune_flag', action='store', help="Enter 1 if you want to tune, 0 to just run experiments", metavar="TUNE", default='0')
    parser.add_argument("-kfeat", "--top_k_feat_plot", dest='k_feat', action='store', help="Enter how many top k feat to plot on chord plot", metavar="KFEAT", default='50')
    parser.add_argument("-ab", "--ablation_flag", dest='ablation_flag', action='store', help="Enter what type of model to train, full or abaltions ('img','gen','clin')", metavar="ABLATION", default='full')
    parser.add_argument("-gpu_tr", "--gpus_per_trial", dest='gpus_per_trial', action='store', help="Enter the number of gpus per trial to use", metavar="GPUTRIAL", default='1')
    
    parser.add_argument("-lr", "--lr", dest='lr', action='store', help="Enter the k dimensions for tf models", metavar="LR", default='0.00001')
    parser.add_argument("-bs", "--batch_size", dest='batch_size', action='store', help="Enter the k dimensions for tf models", metavar="BATCHSIZE", default='64')
    parser.add_argument("-u", "--units", dest='units', action='store', help="Enter the k dimensions for tf models", metavar="FCUNITS", default='128')
    parser.add_argument("-nl_s", "--num_layers_single_mod", dest='num_layers_single_mod', action='store', help="Enter the k dimensions for tf models", metavar="NUMLAYERS", default='4')
    parser.add_argument("-nl_j", "--num_layers_joint_mod", dest='num_layers_joint_mod', action='store', help="Enter the k dimensions for tf models", metavar="NUMLAYERS", default='2')
    parser.add_argument("-k_dim", "--k_dim", dest='d_model', action='store', help="Enter the k dimensions for tf models", metavar="KDIM", default='256')
    parser.add_argument("-nh", "--nhead", dest='nhead', action='store', help="Enter the k dimensions for tf models", metavar="NUMHEAD", default='4')
    parser.add_argument("-dim_ff", "--dim_feedforward", dest='dim_feedforward', action='store', help="Enter the k dimensions for tf models", metavar="DIMFF", default='256')
    parser.add_argument("-dp_tf", "--dropout_tf", dest='dropout_tf', action='store', help="Enter the k dimensions for tf models", metavar="DPTF", default='0.0')
    parser.add_argument("-dp_fc", "--dropout_fc", dest='dropout_fc', action='store', help="Enter the k dimensions for tf models", metavar="DPFC", default='0.3')
    parser.add_argument("-ly_norm", "--layer_norm_eps", dest='layer_norm_eps', action='store', help="Enter the k dimensions for tf models", metavar="LAYERNORM", default='0.000001')
    parser.add_argument("-act", "--activation", dest='activation', action='store', help="Enter the k dimensions for tf models", metavar="ACTIVATION", default='relu')
    parser.add_argument("-alpha", "--alpha", dest='alpha', action='store', help="Enter the k dimensions for tf models", metavar="ALPHA", default='0.5')
    parser.add_argument("-beta", "--beta", dest='beta', action='store', help="Enter the k dimensions for tf models", metavar="BETA", default='0.5')
    parser.add_argument("-m_p_f", "--maxpool_flag", dest='maxpool_flag', action='store', help="Enter the k dimensions for tf models", metavar="MAXPOOLFLAG", default='1')

    parser.add_argument("-alt", "--alt", dest='alt', action='store', help="Enter 1 for alternative network testing", metavar="ALT", default='1')
    parser.add_argument("-st", "--subtype", dest='use_cluster_flag', action='store', help="Enter 1 for using subtypes for labels, 0 for diagnostic labels", metavar="SUBTYPE", default='1')
    

    args = parser.parse_args()

    return args 



# Main
if __name__ == '__main__':
    begin_time = time.time()
    args = parse_arguments()

    ### Set discoverable GPU cards
    # TODO: Double check that this is the right place to set up the number of discoverable GPUs
    # For GPU - Pytorch
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_nums #model will be trained on GPU X
    # print(os.getcwd())
    # print(args.path_snp)
    # print(os.listdir('../../../'))
    # print(os.listdir('../../../../'))
    ### Prepare environment ###
    if not os.path.isdir('../results'):
        os.system('mkdir ../results')
    
    # Parse to local variables
    iters = int(args.iters)
    path_gwas = args.path_gwas
    path_snp = args.path_snp
    path_clin = args.path_clin
    path_img = args.path_img
    path_img_feat = args.path_img_feat
    path_feat_dict = args.path_feat_dict
    path_labels = args.path_labels
    path_clin_feat = args.path_clin_feat
    path_gen_feat = args.path_gen_feat
    path_roi_nm = args.path_roi_nm

    path_res = args.path_res
    fname_root_out = args.fname_out_root
    if args.path_ckpt == None:
        path_ckpt = args.fname_out_root
    else:
        path_ckpt = args.path_ckpt


    rnd_state = int(args.random_seed)
    run_full = bool(int(args.run_full))
    tune_flag = bool(int(args.tune_flag))
    ablation_flag = args.ablation_flag
    gpus_per_trial = args.gpus_per_trial

    k_feat = int(args.k_feat)

    lr = float(args.lr)
    bs = int(args.batch_size)
    u = int(args.units)
    nl_s = int(args.num_layers_single_mod)
    nl_j = int(args.num_layers_joint_mod)
    d_model =int(args.d_model)
    nh = int(args.nhead)
    dim_feedforward = int(args.dim_feedforward)
    dp_tf = float(args.dropout_tf)
    dp_fc = float(args.dropout_fc)
    ly_norm = float(args.layer_norm_eps)
    act = args.activation
    alpha =float(args.alpha)
    beta = float(args.beta)
    maxpool_flag = bool(int(args.maxpool_flag))

    alt_flag = bool(int(args.alt))
    use_cluster_flag = bool(int(args.use_cluster_flag))

    paths = {
        'path_gwas' : os.path.dirname(path_gwas),
        'fn_gwas' : os.path.basename(path_gwas),
        'path_snp' : os.path.dirname(path_snp),
        'fn_snp' : os.path.basename(path_snp),
        'path_clin' : os.path.dirname(path_clin),
        'fn_clin' : os.path.basename(path_clin),
        'path_img' : os.path.dirname(path_img),
        'fn_img' : os.path.basename(path_img),
        'path_img_feat':os.path.dirname(path_img_feat),
        'fn_img_feat': os.path.basename(path_img_feat),
        'path_feat_dict':os.path.dirname(path_feat_dict),
        'fn_feat_dict': os.path.basename(path_feat_dict),
        'path_labels':os.path.dirname(path_labels),
        'fn_labels': os.path.basename(path_labels),
        'path_res' : path_res,
        'checkpoint_name' : path_ckpt
    }
    # args = {
    #     'demo_vars' : ['PATNO','YEAR','age','gen','educ','HISPLAT','race','fampd_new'],
    #     'clin_vars' : ['tau_ab','ptau_ab','ptau_tau','ab_asyn','tau_asyn','ptau_asyn', 'upsit', 'scopa_gi','scopa_ur','scopa_cv','scopa_therm','scopa_pm','scopa_sex','scopa', 'gds', 'quip_gamble','quip_sex','quip_buy','quip_eat','quip_hobby','quip_pund','quip_walk','quip','quip_any', 'stai_state','stai_trait', 'moca', 'lns', 'SDMTOTAL', 'rem','rem_cat','rem_q6', 'bjlot', 'ess', 'hvlt_immediaterecall','HVLTRDLY','HVLTREC','HVLTFPRL','hvlt_discrimination','hvlt_retention'],
    #     'outcome_vars' : ['updrs_totscore','NHY','primdiag'],
    #     'visits' : ['BL','V04','V06','V08','V10'],
    #     'impute': True 
    # }
    args = {
        'demo_vars' : ['PTID','AGE','PTGENDER','PTETHCAT','PTRACCAT'],
        # 'clin_vars' : ['IMAGEUID', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp'],
        # 'clin_vars' : ['ABETA_bl_UCSFFSL_02_01_16_UCSFFSL51_03_01_22', 'TAU_bl_UCSFFSL_02_01_16_UCSFFSL51_03_01_22', 'PTAU_bl_UCSFFSL_02_01_16_UCSFFSL51_03_01_22', 'FDG_bl_UCSFFSL_02_01_16_UCSFFSL51_03_01_22'],
        # 'clin_vars' : ['CDRSB', 'ADAS11' , 'ADAS13', 'ADASQ4', 'RAVLT_immediate' ,'RAVLT_learning' ,'RAVLT_forgetting' ,'RAVLT_perc_forgetting' ,'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR'],
        'clin_vars' : ['RAVLT_immediate' ,'RAVLT_learning' ,'RAVLT_forgetting' ,'RAVLT_perc_forgetting' ,'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR'],

        'outcome_vars' : ['MMSE','CDRSB', 'DX_bl'],
        'visits' : ['bl','m06','m12','m24'],
        'impute': True,
        'visit_colnm' : 'VISCODE',
        'tune_flag':tune_flag,
        'ablation':ablation_flag,
        'gpus_per_trial':gpus_per_trial,
        'alt':alt_flag,
        'use_cluster_flag':use_cluster_flag
    }

    config = {
        'tune' : tune_flag,
        'ablation' : ablation_flag,
        'fo' : fname_root_out,
        ## Training hyperparms
        "lr":lr,
        "batch_size" : bs,
        ## Model hyperparams
        'units': u,
        'num_layers' : nl_s,
        'num_layers_joint' : nl_j,
        'd_model' : d_model, 
        'nhead': nh,
        'dim_feedforward' : dim_feedforward,
        'dropout' : dp_tf,
        'dropout_fc' : dp_fc,
        'layer_norm_eps' : ly_norm,
        'activation' : act,
        'alpha' : alpha,
        'beta' : beta,
        'maxpool_flag' : maxpool_flag,
        'alt' : alt_flag,
    }

# 10x10 -> Repeated experiments (n times defined by the user -> ideally 10 times for 10x10-fold cv)
    for itr in range(iters):
        if run_full:
            # Run training, hyperparam search and testing 
            results_dict = full_exp(paths,args,config)
        else:
            results_dict = single_run(paths,args,config)
        
        # Results summary and return full results dictionary 
        results_df = results_summary(results_dict, fname_root_out, itr)
        results_display(results_df, 'auc_test')

        # Plot co-attention on test set
        if ablation_flag == 'full':
            dict_path = str('../results/'+fname_root_out+'_iter_'+str(itr)+'_results_dictionary.pkl')
            quant_res_path = str('../results/'+fname_root_out+'_iter_'+str(itr)+'_results_summary.csv')
            plot_coattn(dict_path,quant_res_path,path_clin_feat,path_roi_nm,path_gen_feat,k_feat)

        if (itr+1)%10 == 0:
            print(f'{itr} iterations completed.')
            print('='*40)
                
    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))






