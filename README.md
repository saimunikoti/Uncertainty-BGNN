# Uncertainty-BGNN
Propagation of Aleatoric and Epistemic Uncertainty in Graph Neural Networks

Anonymized GitHub repo for ICML 2022 Submission.

This code is provided solely for the purpose of peer review for the ICML 2022 conference.

===================== File specification =========================

1. src/data/Linkprediction_Linkprob.py : Train GNN model for predicting links.

1. src/data/make_dataset_uncertpropag.py : Preprocess input graph and generate features and labels.

1. src/data/make_dataset_map.py: Process input graph and generate features and labels from MAP estimate (Baseline). 

2. src/models/Nodeclass_Graphsage.py: Train node classification model. 

3. src/models/Nodeclass_Graphsage_test_mean.py: Test script for mean predictions.

3. src/models/Nodeclass_Graphsage_test_var.py: Test script for variance estimation. 

3. src/models/Nodeclass_Graphsage_test_MAP.py: Test script for MAP estimation.

4. utils.py: helping functions.

5. src/features/build_features.py: helping function for feature extraction

6. src/visualization/visual.py: helping functions for visualizing features 

==================================== end =========================
