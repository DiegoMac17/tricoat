import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


#### Useful definitions ####
def clean_id(x):
    return re.split('_+', x)[2]

def clean_exo_id(x):
    return re.split('_+', x)[0]

def clean_wgs_id(x):
    split_id = re.split('_+', x)
    return split_id[len(split_id)-1]

def clean_pat_wgs_id(x):
    split_id = re.split('I', x)
    return split_id[len(split_id)-1]

def clean_wgs_snpid(x):
    return re.split('_+', x)[0]

def match_ids(df1,df2, df1_index, df2_index):
    df1 = df1.loc[df1.iloc[:,df1_index].isin(df2.iloc[:,df2_index])]
    df2 = df2.loc[df2.iloc[:,df2_index].isin(df1.iloc[:,df1_index])]
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    return df1,df2
# def match_index(df1,df2):
#     df1 = df1.loc[df1.iloc[:,df1_index].isin(df2.iloc[:,df2_index])]
#     df2 = df2.loc[df2.iloc[:,df2_index].isin(df1.iloc[:,df1_index])]
#     df1 = df1.reset_index(drop=True)
#     df2 = df2.reset_index(drop=True)
#     return df1,df2
def match_index(df1,df2):
    matching_ids = df1.index.intersection(df2.index)
    df1 = df1.loc[matching_ids]
    df2 = df2.loc[matching_ids]
    return df1,df2
def loss(history,num_epochs):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(num_epochs)
    plt.figure(figsize = (16,10))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss')
    plt.legend()
    plt.show()
def ROC(y_true,probs, model_name = ""):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_true))]
    # calculate scores
    ns_auc = roc_auc_score(y_true, ns_probs)
    model_auc = roc_auc_score(y_true, probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print(f'Best performing - {model_name}: ROC AUC = {round(np.mean(model_auc) * 100, 2)}%')
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, probs,  drop_intermediate=False)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - '+model_name)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()