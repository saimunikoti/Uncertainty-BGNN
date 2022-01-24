import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar
import numpy as np
from stellargraph import datasets
from IPython.display import display, HTML
from src_bgnn.data import config as cnf
import pickle
import scipy.io as io

"""
input: read raw graphs
operations:
1. Unsupervised stellargraph training model for generating node embeddings
2. get distance matrix Z

"""

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()
node_subject = node_subjects.astype("category").cat.codes

print(G.info())

nodes = list(G.nodes())
number_of_walks = 1
length = 5

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 20
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)

layer_sizes = [50, 50]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
)

# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="mul"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)

## extract node embeddings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = node_subjects.index
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1

#===== get distance matrix 1 from node embeddings ====================

zdist_1 = np.zeros(shape=(len(nodes), len(nodes)))

for count1 in range(len(nodes)):
    for count2 in range(len(nodes)):
        diffvector = node_embeddings[count1]- node_embeddings[count2]
        zdist_1[count1, count2] = np.linalg.norm(diffvector)

filepath = cnf.datapath + "\\cora_distmatrix1.pickle"

with open(filepath, 'wb') as b:
    pickle.dump(zdist_1, b)

# ===== get distance matrix 2 from node labels ====================

g = G.to_networkx()
g = nx.Graph(g)

zdist_2 = np.zeros(shape=(len(nodes), len(nodes)))

for counti in range(len(nodes)):

    for countj in range(len(nodes)):
        neigh_i = list(nx.neighbors(g, nodes[counti]))
        neigh_j = list(nx.neighbors(g, nodes[countj]))

        temp_sum = 0

        for nodes_i in neigh_i:
            for nodes_j in neigh_j:
                if node_subject[nodes_i] != node_subject[nodes_j]:
                    temp_sum+=1

        zdist_2[counti, countj] = temp_sum/(len(neigh_i)*(len(neigh_j)))

# ============= get weighing factor==========

delta = np.max(np.max(zdist_1))/np.max(np.max(zdist_2))

zdistmatrix = zdist_1 + delta*zdist_2

filepath = cnf.datapath + "\\cora_distmatrix.pickle"
with open(filepath, 'wb') as b:
    pickle.dump(zdistmatrix, b)

zdistmatrix_mat = {}
zdistmatrix_mat['zdist'] = zdistmatrix
filepath = cnf.datapath + "\\cora_distmatrix.mat"
io.savemat(filepath, zdistmatrix_mat)

##
# node_subject = node_subjects.astype("category").cat.codes
#
# X = node_embeddings
#
# if X.shape[1] > 2:
#     transform = TSNE  # PCA
#
#     trans = transform(n_components=2)
#     emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_ids)
#     emb_transformed["label"] = node_subject
# else:
#     emb_transformed = pd.DataFrame(X, index=node_ids)
#     emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
#     emb_transformed["label"] = node_subject
#
# alpha = 0.7
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.scatter(
#     emb_transformed[0],
#     emb_transformed[1],
#     c=emb_transformed["label"].astype("category"),
#     cmap="jet",
#     alpha=alpha,
# )
# ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
# plt.title(
#     "{} visualization of GraphSAGE embeddings for cora dataset".format(transform.__name__)
# )
# plt.show()
