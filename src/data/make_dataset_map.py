from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import random
from src_bgnn.data import config as cnf
import scipy.io as io
import networkx as nx
import seaborn as sns
import numpy as np
from stellargraph import datasets

"""
"""
filepath = cnf.datapath + "\\cora_MAPnew_wthresh0.1.mat"
data = io.loadmat(filepath)
data = data['Wprednew']

# generate dgl graph from weightyed adjacency matrix
g1 = nx.Graph(data)

dataset = datasets.Cora()
G, node_subjects = dataset.load(subject_as_feature=True)
print(G.info())
g2 = G.to_networkx()
g2 = nx.Graph(g2)

node_subject = node_subjects.astype("category").cat.codes
nodelabellist = list(node_subject)
nodelist2 = list(g2.nodes())

for (node1, d) in g1.nodes(data=True):
    d['label'] = nodelabellist[node1]
    d['feature'] = g2.nodes[nodelist2[node1]]['feature']

filepath = cnf.datapath + "\\cora_MAPnew_wthresh0.1_procsd.gpickle"

nx.write_gpickle(g1, filepath)

## some preprcessing

# avgDegrees = []
# avgClustCoeff = []
# degAssrt = []

# for i in range(data.shape[2]):
#     # Extracting adjacency matrix
#     data_current_iter = data[:,:,i]
#     # Graph from adjacency matrix
#     graph_current_iter = nx.Graph(data_current_iter)
#     # Evaluating average degree
#     degrees_currGraph = dict(graph_current_iter.degree())
#     countNodes = data.shape[0]
#     avgDegree_currGraph = sum(degrees_currGraph.values())/countNodes
#     avgDegrees.append(avgDegree_currGraph)
#     # Evaluating average clustering coefficient
#     avgCC_currGraph = nx.average_clustering(graph_current_iter)
#     avgClustCoeff.append(avgCC_currGraph)
#     # Evaluating degree assortativity
#     dgAsrt_currGraph = nx.degree_assortativity_coefficient(graph_current_iter)
#     degAssrt.append(dgAsrt_currGraph)


