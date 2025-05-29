import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import moment
import seaborn as sns
import os
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer

def load_data(path,diag_coln,visit_coln,target_coln, index_coln, diags, visits):
    df = pd.read_csv(path)
    df_targets_filt = df.loc[df[diag_coln].isin(diags)]
    df_visits_filt = df_targets_filt.loc[:,[index_coln,target_coln,visit_coln]].pivot_table(values=target_coln, index = index_coln, columns = visit_coln)
    df_visits_filt = df_visits_filt.loc[:,visits]
    df_imputed = impute(df_visits_filt)
    return df_imputed, len(df_visits_filt)

def impute(df):
    # Interpolation for middle data points
    # Nearest neighbor imputation for first or last data point
    # if row[0] == None:
    #     row = KNNImputer().fit_transform(row)
    # if row[-1] == None:
    #     row = KNNImputer().fit_transform(row)
    # if row.isnull().sum() >0:
    #     interp = interp1d(x=np.range(len(row)),y=row)
    #     row[] = interp(row)

    df = df.interpolate(method ='linear', axis=1, limit_direction ='both', limit = 2)
    # Drop rows that are all nan (i.e. they still have nan after interpolation)
    df = df[~df.isnull().any(axis=1)]

    return df

def df_adjuster(df, colnames):
    return df.sub(df.iloc[:,0].values, axis = 0)

def df2moments(df, colnames):
    m1 = moment(df.loc[:,colnames], moment=1, axis=1)
    m2 = moment(df.loc[:,colnames], moment=2, axis=1)
    m3 = moment(df.loc[:,colnames], moment=3, axis=1)
    m4 = moment(df.loc[:,colnames], moment=4, axis=1)
    return pd.DataFrame(zip(m1,m2,m3,m4), index = df.index, columns=['m1','m2','m3','m4'])


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def scatterplots(data, clust_col,results_path, fn):
    plot = sns.scatterplot(data, x='m2', y = 'm3', hue = clust_col)
    plt.savefig(os.path.join(results_path,fn+'_m2_m3.pdf'))
    plt.clf()
    plot = sns.scatterplot(data, x='m2', y = 'm4', hue = clust_col)
    plt.savefig(os.path.join(results_path,fn+'_m2_m4.pdf'))
    plt.clf()
    plot = sns.scatterplot(data, x='m3', y = 'm4', hue = clust_col)
    plt.savefig(os.path.join(results_path,fn+'_m3_m4.pdf'))
    plt.clf()


def plot_survival(data_adj, data, pat_col, vis_col, clust_col, results_path, fn):
    rcParams.update({'font.size': 22})
    # Plot line plot with one line per cluster
    data_adj = data_adj.reset_index()
    temp = data_adj.drop(columns=clust_col)
    temp = pd.melt(temp, pat_col).sort_values([pat_col, vis_col])
    temp['cluster'] = np.repeat(data_adj[clust_col].values, len(np.unique(temp[vis_col])))
    plt.figure(figsize=(12,9))
    per_clust = sns.lineplot(x=vis_col, y='value', hue='cluster', errorbar=('sd',1),
             data=temp, palette=sns.color_palette(['green', '#f76d11', 'blue']))
    per_clust.set_xlabel("Visit", fontsize=22)
    per_clust.set_ylabel('MMSE Score Difference From Baseline', fontsize=22)
    plt.legend(loc='lower left', fontsize=22)
    plt.savefig(os.path.join(results_path,fn+'_survival_plot_per_clust_new_colors.pdf'))
    plt.clf()
    # Plot line plot with one line per patient and colored with clusters
    data = data.reset_index()
    plt.figure(figsize=(12,9))
    for i, row in data.iterrows():
        c = row[clust_col]
        if c == 'Slow':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='green', alpha = 0.05, linewidth = 0.5)
        if c == 'Intermediate':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='#f76d11', alpha = 0.075, linewidth = 0.75)
        if c == 'Fast':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='blue', alpha = 0.1, linewidth = 0.5)
    per_pat.set_xlabel('Visit', fontsize=22)
    per_pat.set_ylabel('MMSE Score', fontsize=22)
    plt.savefig(os.path.join(results_path,fn+'_survival_plot_per_pat_new_colors.pdf'))
    plt.clf()
    
# old colors ['#354F60', '#BC0E4C', '#c28d11']


def plot_survival_one_plot(data_adj, data, pat_col, vis_col, clust_col, results_path, fn):
    # Plot line plot with one line per patient and colored with clusters
    data = data.reset_index()
    for i, row in data.iterrows():
        c = row[clust_col]
        if c == 'Slow':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='#354F60', alpha = 0.05, linewidth = 0.5)
        if c == 'Intermediate':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='#BC0E4C', alpha = 0.05, linewidth = 0.5)
        if c == 'Fast':
            per_pat = sns.lineplot(data.iloc[i,1:-1], color='#c28d11', alpha = 0.05, linewidth = 0.5)
    per_pat.set_xlabel('Visit')
    per_pat.set_ylabel('MMSE Score Difference')

    # Plot line plot with one line per cluster
    data_adj = data_adj.reset_index()
    temp = data_adj.drop(columns=clust_col)
    temp = pd.melt(temp, pat_col).sort_values([pat_col, vis_col])
    temp['cluster'] = np.repeat(data_adj[clust_col].values, len(np.unique(temp[vis_col])))
    per_clust = sns.lineplot(x=vis_col, y='value', hue='cluster', errorbar=('sd',1),
             data=temp, palette=sns.color_palette(['#354F60', '#BC0E4C', '#c28d11']), linewidth = 2.5)
    per_clust.set_xlabel("Visit")
    per_clust.set_ylabel("Score")

    plt.savefig(os.path.join(results_path,fn+'_survival_plot_one_plot.pdf'))
    plt.clf()
    

if __name__ == '__main__':
    data_path = 'data/'
    results_path = 'results/figures'
    # ppmi_path = str(data_path+'curated_data_cuts/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv')
    adni_path = str(data_path+'tadpole_challenge_v2_by_mansu/ADNIMERGE.csv')
    
    # Load, clean and impute data 
    # ppmi, ppmi_len_noimp = load_data(ppmi_path,'primdiag','EVENT_ID','updrs_totscore','PATNO',[1],['BL','V04','V06','V08','V10'])
    adni, adni_len_noimp = load_data(adni_path,'DX_bl','VISCODE','MMSE','PTID',['AD','EMCI','LMCI'],['bl','m06', 'm12', 'm24'])

    # Normalization (everyone starts at the same point)
    # ppmi_adj = df_adjuster(ppmi,['BL','V04','V06','V08','V10'])
    adni_adj = df_adjuster(adni,['bl','m06', 'm12', 'm24'])

    # Clustering 
    # Stats moments 
    # ppmi_moments = df2moments(ppmi_adj,['BL','V04','V06','V08','V10'])
    adni_moments = df2moments(adni_adj,['bl','m06', 'm12', 'm24'])

    
    # Hierarchical clustering
    # ppmi_clustering = AgglomerativeClustering(n_clusters=3).fit(ppmi_moments)
    # ppmi_moments['hierarchical_cluster'] = ppmi_clustering.labels_
    adni_clustering = AgglomerativeClustering(n_clusters=3).fit(adni_moments)
    adni_moments['hierarchical_cluster'] = adni_clustering.labels_

    # kmeans clustering
    # using moments
    # ppmi_clustering_moments = KMeans(n_clusters=3, random_state = 42).fit(ppmi_moments)
    # ppmi_moments['kmeans_cluster'] = ppmi_clustering_moments.labels_
    adni_clustering_moments = KMeans(n_clusters=3, random_state = 42).fit(adni_moments)
    adni_moments['kmeans_cluster'] = adni_clustering_moments.labels_

    # using raw time points
    # ppmi_clustering_timepoints = KMeans(n_clusters=3, random_state = 42).fit(ppmi_adj)
    adni_clustering_timepoints = KMeans(n_clusters=3, random_state = 42).fit(adni_adj)
    # ppmi_adj['kmeans_cluster_timepoints'] = ppmi_clustering_timepoints.labels_
    adni_adj['kmeans_cluster_timepoints'] = adni_clustering_timepoints.labels_
    # ppmi['kmeans_cluster_timepoints'] = ppmi_clustering_timepoints.labels_
    adni['kmeans_cluster_timepoints'] = adni_clustering_timepoints.labels_

    # Plot 
    # ppmi_plot_bl_v10 = sns.scatterplot(ppmi_moments, x='BL', y = 'V10', hue = 'cluster')

    # scatterplots(ppmi_moments, 'hierarchical_cluster', results_path, 'ppmi_plot_hierclust')
    # scatterplots(adni_moments, 'hierarchical_cluster', results_path, 'adni_plot_hierclust_MMSE')

    # scatterplots(ppmi_moments, 'kmeans_cluster', results_path, 'ppmi_plot_kmeansclust')
    # scatterplots(adni_moments, 'kmeans_cluster', results_path, 'adni_plot_kmeansclust_MMSE')

    # Rename labels so plot it looks nicer - just doing for plotting, comment it out for when saving labels for model
    label_mapping = {0:'Fast', 1:'Intermediate', 2:'Slow'} 
    adni_adj.kmeans_cluster_timepoints = adni_adj.kmeans_cluster_timepoints.map(label_mapping)
    adni.kmeans_cluster_timepoints = adni.kmeans_cluster_timepoints.map(label_mapping)

    # Plot survival lines
    # plot_survival(ppmi_adj, ppmi, 'PATNO', 'EVENT_ID', 'kmeans_cluster_timepoints', results_path, 'ppmi_plot_kmeansclust')
    plot_survival(adni_adj, adni, 'PTID', 'VISCODE', 'kmeans_cluster_timepoints', results_path, 'adni_plot_kmeansclust_MMSE')
    # plot_survival_one_plot(adni_adj, adni_adj, 'PTID', 'VISCODE', 'kmeans_cluster_timepoints', results_path, 'adni_plot_kmeansclust_MMSE')

    # Save moments dfs
    # ppmi_moments.to_csv(os.path.join(results_path,'ppmi_moments.csv'))
    # adni_moments.to_csv(os.path.join(results_path,'adni_moments.csv'))

    # Save clustering results
    # ppmi_labels = ppmi_moments['kmeans_cluster'].to_frame()
    # ppmi_labels['kmeans_cluster_timepoints'] = ppmi_clustering_timepoints.labels_
    adni_labels = adni_moments['kmeans_cluster'].to_frame()
    adni_labels['kmeans_cluster_timepoints'] = adni_clustering_timepoints.labels_


    # ppmi_labels.to_csv(os.path.join(results_path,'ppmi_labels.csv'))
    # adni_labels.to_csv(os.path.join(results_path,'adni_labels.csv'))

    # Save data stats to txt file
    # with open(os.path.join(results_path,'data_stats.txt'), 'w') as f:
    #     f.write('PPMI')
    #     f.write('\n')
    #     f.write('Total number of subjects before interpolation = '+str(ppmi_len_noimp))
    #     f.write('\n')
    #     f.write('Total number of subjects after interpolation = '+str(len(ppmi_moments)))
    #     f.write('\n')
    #     f.write('ADNI')
    #     f.write('\n')
    #     f.write('Total number of subjects before interpolation = '+str(adni_len_noimp))
    #     f.write('\n')
    #     f.write('Total number of subjects after interpolation = '+str(len(adni_moments)))

    # print('Results saved in: ',results_path)