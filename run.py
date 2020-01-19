import os
import numpy as np
import pandas as pd
# for preparing dataset and d2v features
from prepare_features.kfold_data_split import split_data
from prepare_features.make_features import buidling_d2v_kfold
# for training
from mvdts.evaluation import evaluation

def read_data(dataLocation):
    """
   Reading data and returning in proper format
   :param dataLocation: location of data
   :return: set of features, labels and combine
   """
    df = pd.read_csv(dataLocation)

    X = np.array(df[df.columns[:-1]].values.tolist(), dtype=np.float64)
    y = np.array(df[df.columns[-1]].values.tolist())
    return [X, y]

# for lr_mvdt and rs_mvdt
def hell_run(datasets, k, vector_sizes, algorithms_list, epochs_list , n_features_list, depth_list):
    total = 0
    root_folder = "data/"
    for name in datasets.split(","):
        # selecting root folder of each dataset
        data_name_folder = root_folder + name + "/"
        for n in range(k):
            # selecting kFold folder of each dataset then final dir
            count = str(n + 1)
            k_data_name_folder_final = data_name_folder + "k" + count + "/final/"

            # checking each demension datasets inside final dir
            for vector_size in vector_sizes:
                dim = str(vector_size)
                d2vDim_name_train = dim + "D_" + name + "_k" + count + "_train.csv"
                d2vDim_name_test = dim + "D_" + name + "_k" + count + "_test.csv"

                # checking if both trian and test files exists or not
                if os.path.exists(k_data_name_folder_final + d2vDim_name_train) and os.path.exists(
                        k_data_name_folder_final + d2vDim_name_test):
                    # print(k_data_name_folder_final+d2vDim_name_train)

                    # getting [X_train, y_train] and [X_test, y_test]
                    train = read_data(k_data_name_folder_final + d2vDim_name_train)
                    test = read_data(k_data_name_folder_final + d2vDim_name_test)

                    # selecting algorithm
                    for algorithm in algorithms_list:
                        # choose epochs name
                        for epochs in epochs_list:
                            # choose no of features
                            for n_features in n_features_list:
                                # choose max depth of tree
                                for depth in depth_list:
                                    # run 10 times to get average result
                                    for run in range(10):
                                        total += 1
                                        # print(name, algorithm, n, dim, df_train, df_test, epochs, n_features, depth)
                                        evaluation(name, [d2vDim_name_train, d2vDim_name_test], count, dim, algorithm,
                                                   epochs, depth, n_features, run, train, test)
                    # print(len(df_train), len(df_test))
                else:
                    pass
                    # print("datasets not found: ",k_data_name_folder_final+d2vDim_name_train)
    print(total)

# for sklearn cart
def sk_cart(datasets, k, vector_sizes, depth_list):
    total = 0
    root_folder = "data/"
    for name in datasets.split(","):
        # selecting root folder of each dataset
        data_name_folder = root_folder + name + "/"
        for n in range(k):
            # selecting kFold folder of each dataset then final dir
            count = str(n + 1)
            k_data_name_folder_final = data_name_folder + "k" + count + "/final/"

            # checking each demension datasets inside final dir
            for vector_size in vector_sizes:
                dim = str(vector_size)
                d2vDim_name_train = dim + "D_" + name + "_k" + count + "_train.csv"
                d2vDim_name_test = dim + "D_" + name + "_k" + count + "_test.csv"

                # checking if both trian and test files exists or not
                if os.path.exists(k_data_name_folder_final + d2vDim_name_train) and os.path.exists(
                        k_data_name_folder_final + d2vDim_name_test):
                    # print(k_data_name_folder_final+d2vDim_name_train)

                    # getting [X_train, y_train] and [X_test, y_test]
                    train = read_data(k_data_name_folder_final + d2vDim_name_train)
                    test = read_data(k_data_name_folder_final + d2vDim_name_test)

                    # choose max depth of tree
                    for depth in depth_list:
                        # choose epochs name
                        # can run cart algo here
                        total += 1
                        evaluation(name, [d2vDim_name_train, d2vDim_name_test], count, dim, "cart",
                                  0, depth, 0, 0, train, test)
                        # print(len(df_train), len(df_test))
                else:
                    print("datasets not found: ", k_data_name_folder_final+d2vDim_name_train)
                    #pass
    print(total)

if __name__ == '__main__':
    # settings
    # datasets = "imbd,quora"
    # k = 5 #20%test
    # vector_sizes = [150, 200, 250, 300, 350]
    # algorithms_list = ['lr_mvdt', 'rs_mvdt']
    # epochs_list = [100, 300, 700, 1000, 1200]
    # n_features_list = [10, 30, 70, 90, 140] # should below min vector_size
    # depth_list = [3, 6, 9, 12, 15]

    datasets = "20ng_CG_RM,20newsgroup"
    k = 2
    vector_sizes = [30]
    algorithms_list = ['lr_mvdt', 'rs_mvdt']
    epochs_list = [50]
    n_features_list = [10]
    depth_list = [3]

    # splitting kfold sets
    print("--------- (k_fold) raw data preparing --------- ")
    split_data(datasets, k)
    print("--------- (k_fold) raw data save saved --------- ")

    # training on d2v
    print("**************** training on doc2vec model ****************")
    buidling_d2v_kfold(datasets, k, vector_sizes)
    print("**************** datasets are prepared and saved ****************")

    # Model training
    print("**************** model training start ****************")
    print("*-*-*-*-*- Starting on Lr_mvdt and rs_mvdt -*-*-*-*-*")
    hell_run(datasets, k, vector_sizes, algorithms_list, epochs_list, n_features_list, depth_list)
    print("*-*-*-*-*- Starting on sklearn cart -*-*-*-*-*")
    sk_cart(datasets, k, vector_sizes, depth_list)
    print("**************** Finish, check results.csv ****************")