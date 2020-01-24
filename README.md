# random_mvdt
Random feature selection approch on multivariate decision tree.

A univariate decision tree is good it self but when data can be separate linearly its better to use multivariate concept. In multivariate we take all or combination of features to make decision boundary until all or some criteria meets. 

In this project we try two different way of achieving linear separation on normal decision tree.   

# Approach
1. First we use normal deterministic way of to find coefficient based on RANSAC algorithm(rs_mvdt). https://en.wikipedia.org/wiki/Random_sample_consensus
2. Second using logistic regression in cart decision tree(lr_mvdt). https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Random Feature selection
In every inner nodes we use n number of feature to split data. Selection of features are different in every inner node but number of feature is same. This technique is used for both rs_mvdt and lr_mvdt.

# Dataset
We test on binary text classifcation problem. We use Doc2vec embeddings to create feature.
1. IMDB review: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Quora Insincere Questions: https://www.kaggle.com/c/quora-insincere-questions-classification/data

# Setting
datasets = "2newgroup,imdb,quora"
k = 5
vector_sizes = [10, 25, 50, 75, 100]
algorithms_list = ['lr_mvdt', 'rs_mvdt'] # ['cart']
epochs_list = [100, 300, 500, 800, 1000]
n_features_list = [2, 5, 10, 20, all]
#depth_list = [3, 6, 9, 12, 15]
min_leaf_point_list = [5, 10, 15, 20, 30]

# Evaluation
Performance of above approch is compared with scikit-learn based decision tree based on following points.
1. Sklearn Matrix
  i. Accuracy
  ii. Precision
  iii. Recall
  iv. F1
2. Tree(model) information
  i. Max depth
  ii. Inner nodes
  iii. leaf nodes
  iv. all nodes
3. Complexitity BigO


# Requirement
This project can be run as script and jupyter notebook
1. Numpy
2. Pandas
3. Scikit-learn
4. Matplotlib

# Execuite script
1. K_fold split of both datasets: split_data(datasets, k)
2. Doc2vec embedding on each K_fold datasets: buidling_d2v_kfold(datasets, k, vector_sizes)
3. Training using lr_mvdt and rs_mvdt: hell_run(datasets, k, vector_sizes, algorithms_list, epochs_list, n_features_list, depth_list)
4. Training using cart: sk_cart(datasets, k, vector_sizes, depth_list)

Use run.py to execute all process
