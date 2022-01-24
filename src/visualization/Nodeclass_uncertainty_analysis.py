import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src_bgnn.data import config as cnf
import pickle
import warnings

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

## accuracy and NLL across all test samples

fileext = "Resultsdic_amazon-comp_varpred_var12"

def get_meanacc_nll(fileext):

    with open(cnf.modelpath + fileext + ".pkl", 'rb') as b:
        resultsdf = pickle.load(b)

    y_pred = resultsdf['pred_array']
    y_true = resultsdf['true_array']
    nll =    resultsdf['nll_array']
    nll_mean = nll.mean(axis=(0, 1))
    predloss = resultsdf['loss_array']
    predloss_mean = predloss.mean(axis=(0, 1))
    sigmatot = resultsdf['sigmatot_array']
    sigmatot_mean = sigmatot.mean(axis=(0, 1))

    y_pred = np.mean(y_pred, axis=2)
    y_pred = np.argmax(y_pred, 1)
    y_true = np.argmax(y_true, 1)

    Accuracy = accuracy_score(y_true, y_pred)

    print("Accuracy, predloss, NLL, prop_variance", Accuracy, predloss_mean, nll_mean, sigmatot_mean)

get_meanacc_nll(fileext)

## longitudinal study across samples

fileext = "SemiResultsdic_pubmed_meanpred_var0"

with open(cnf.modelpath + fileext + ".pkl", 'rb') as b:
    resultsdf = pickle.load(b)

##

# filepath = cnf.modelpath + "Resultsdf_varpred_var12.xlsx"
# Results_df, Accuracy, Avg_NLL, Avg_PredLoss = get_acc_nll(filepath)
# print("Acc, predloss, NLL", Accuracy, Avg_PredLoss, Avg_NLL)

# filepath = cnf.modelpath + "Resultsdic_cora_varpred_var12.pkl"
# filepath = cnf.modelpath + "Resultsdic_cora_varpred_var0.pkl"
#
# with open(filepath, 'rb') as f:
#     mean_dic = pickle.load(f)
#
# true_array = mean_dic['true_array']
# pred_array = mean_dic['pred_array']
#
# import seaborn as sns, numpy as np
# ax = sns.distplot(pred_array[2,0,:], norm_hist=True, kde=True)

