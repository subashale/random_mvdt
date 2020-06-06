# random_mvdt

Decision tree algorithm plays a vital role in machine learning. The algorithm is famous for its simple nature and comprehensible decision rule makes favourable choice for supervised problems. The algorithm does recursive partition of feature space into sub-region using axis parallel split. However, for some problem partitioning in axis parallel can produce complicated decision structure to understand having overfitting. As an alternative, linear split could reduce the complexity while comprehensibility could decrease. Therefore, in this paper, we present the comparison of classification and regression tree (CART) with linear rule induction for text classification domain. We induced logistic regression(LR) and random sample split approach to create linear decision boundary or as mini classifier for each decision nodes. Document to vector (D2V) embedding is applied to create feature vectors for binary classes text dataset. A random process is carried out to select a linear combination feature. We observed our induced linear decision trees have higher accuracy than CART in all test datasets of all feature vectors. All linear induction doesn't resolve overfitting problem rather depends on classifier. We find D2V representation are suitable in linear induction than CART which creates comparably small tree with better accuracy in small vectors whereas larger are better at accuracy only.

# Approach
1. First we use non-deterministic way of to find coefficient based on RANSAC algorithm(rs_mvdt). https://en.wikipedia.org/wiki/Random_sample_consensus
2. Second, we use logistic regression to induce cart(lr_mvdt).
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
3. Scikit-learn decision tree (cart). https://scikit-learn.org/stable/modules/tree.html

# Random Feature selection
In every inner nodes we use n number of features to train our algorithms.

# Dataset
We test on binary text classifcation problem. We use Doc2vec embeddings to create feature vector.
1. 20newsgroup(two label): https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
2. IMDB review: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
4. Quora Insincere Questions: https://www.kaggle.com/c/quora-insincere-questions-classification/data

# Setting
1. datasets = "20newsgroup,imdb,quora"
2. k = 5
3. vector_sizes = [10, 25, 50, 75, 100]
4. algorithms_list = ['cart', 'lr_mvdt', 'rs_mvdt']
5. epochs_list = [100, 300, 500, 800, 1000]
6. n_features_list = [2, 5, 10, 20, all]
#depth_list = [3, 6, 9, 12, 15]
7. min_leaf_point_list = [5, 10, 15, 20, 30]

Impurity measures
1. CART (Gini)
2. LR_MVDT (solver='liblinear', entropy)
3. RS_MVDT ('entropy')

# Evaluation
Performance of above approch is compared with scikit-learn based decision tree based on following points.
1. Big0 (Time complexity): Algorithm performance
2. Tree attributes
	1. max_depth, inner_nodes, leaf_nodes, all_nodes
	2. average max depth, min depth
3. Evaluation matrix(train/test)
 	1. Accuracy, precision, recall, f1
 	2. Average of all
4. Runtime: Training time
5. Hyperparameter
	1. N combination of features
	2. Epochs
	3. Min leaf point
6. Tree depth base evaluation
	1. Intermediate: depth vs accuracy
	2. Structure of the tree
	

# Requirement
This project can be run as script and jupyter notebook
1. Numpy
2. Pandas
3. Scikit-learn
4. Gensim

# Execuite script
1. K_fold split of both datasets: split_data(datasets, k)
2. Doc2vec embedding on each K_fold datasets: buidling_d2v_kfold(datasets, k, vector_sizes)
3. Training using lr_mvdt and rs_mvdt: hell_run(datasets, k, vector_sizes, algorithms_list, epochs_list, n_features_list, depth_list)
4. Training using cart: sk_cart(datasets, k, vector_sizes, depth_list)

Use run.py to execute all process, Cheers :) 
