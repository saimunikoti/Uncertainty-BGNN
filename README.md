# Uncertainty-BGNN
Propagation of Aleatoric and Epistemic Uncertainty in Graph Neural Networks

This code is provided solely for the purpose of peer review for the ICML 2022 conference.

===================== File specification =========================

1. src/data/dgl.zip: Modified source files of the package "deep graph library". Unzip for testing.

2. src/data/Linkprediction_Linkprob.py : Train GNN model for predicting links.

3. src/data/make_dataset_uncertpropag.py : Preprocess input graph and generate features and labels.

4. src/data/make_dataset_map.py: Process input graph and generate features and labels from MAP estimate (Baseline). 

5. src/models/Nodeclass_Graphsage.py: Train node classification model. 

6. src/models/Nodeclass_Graphsage_test_mean.py: Test script for mean predictions.

7. src/models/Nodeclass_Graphsage_test_var.py: Test script for variance estimation. 

8. src/models/Nodeclass_Graphsage_test_MAP.py: Test script for MAP estimation.

9. utils.py: helping functions.

10. src/features/build_features.py: helping function for feature extraction

11. src/visualization/visual.py: helping functions for visualizing features 

### Contributor 
Sai Munikoti - sai.munikoti [at] pnnl [dot] gov

Deepesh Agarwal - deepesh.agarwal [at] ksu [dot] edu

Laya Das - lay.das [at] ethz [dot] ch

Bala Natarajan - bala [at] ksu [dot] edu

==================================== end =========================
