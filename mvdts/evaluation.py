import os.path
import pandas as pd
from mvdts.dt_common.prediction import print_model, predict, time_it
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from mvdts.call_algo import fit
import numpy as np
import _pickle as pickle


def evaluation(dataset_name, pkl_location, filename, k_fold, d2v_vec_size, algorithm, epochs, min_leaf_point, n_features
               , run, train_data, test_data):
    print("Fitting on, dataset: {}, filename: {}, k_fold: {}, vector_size: {},"
          " algorithm: {}, epochs: {}, depth: {}, n_feature: {}, run: {}, train_data_shape: {}".format(dataset_name,
        filename, k_fold, d2v_vec_size, algorithm, epochs, min_leaf_point, n_features, run, str(train_data[0].shape)))

    start_time = time_it()
    tree = fit(algorithm, train_data, epochs, min_leaf_point, n_features)
    end_time = time_it()
    runtime = end_time - start_time

    # getting tree information
    if algorithm == "lr_mvdt" or algorithm == "rs_mvdt":
        # storing tree in pickle file
        # take file name
        pkl_filename = filename[0].split("/")[-1].split(".")[0]+'_'+algorithm+'_'+str(epochs)+'_'+str(min_leaf_point)+\
                       '_'+str(n_features)+'_'+str(run)+'_.pkl'
        with open(pkl_location + pkl_filename, 'wb') as save_tree:
            pickle.dump(tree, save_tree)
        # getting model information
        inner_node, leaf_node, all_nodes, max_depth, left_size, right_size = get_mvdt_model_info(tree)
    else:
        pkl_filename = filename[0].split("/")[-1].split(".")[0]+'_'+algorithm+'_'+str(min_leaf_point)+'_'+str(run)+\
                       '_.pkl'
        with open(pkl_location + pkl_filename, 'wb') as save_tree:
            pickle.dump(tree, save_tree)
        # getting model information
        # counting left_size, right_size is not finish yet
        inner_node, leaf_node, all_nodes, max_depth, left_size, right_size = get_cart_model_info(tree)

    model_location_name = pkl_location + pkl_filename
    # storing all results
    store_results(dataset_name, filename, k_fold, d2v_vec_size, algorithm, epochs, min_leaf_point, n_features, run,
                  train_data, test_data, tree, max_depth, inner_node, leaf_node, all_nodes, left_size, right_size,
                  runtime, model_location_name)

    print("--------- model evaluation information saved --------- ")


def get_cart_model_info(model):
    all_nodes = model.tree_.node_count

    leaf_node = model.tree_.n_leaves
    inner_node = all_nodes - leaf_node
    max_depth = model.tree_.max_depth

    # find out left and right branch nodes
    left_size = 0
    right_size = 0

    return inner_node, leaf_node, all_nodes, max_depth, left_size, right_size


def get_mvdt_model_info(model):
    n_nodes, max_depth, leaf_nodes = print_model(model, root_node=[], leaf_node=[])
    # root decision node includes all nodes(lr) and leaf decision nodes
    inner_node = len(n_nodes) - 1  # it includes root node so -1 to count leaf node
    leaf_node = len(leaf_nodes)
    all_nodes = inner_node + leaf_node
    left_size, right_size = tree_left_right_size(n_nodes)
    return inner_node, leaf_node, all_nodes, max_depth, left_size, right_size

# left and right branch inner node count
def tree_left_right_size(inner_nodes):
    # getting all inner nodes from left and right branch
    idx = []
    print(inner_nodes)
    for i, v in enumerate(inner_nodes):
        # getting 1 1 initial depth
        # if index 1 find out which branch is it left or right
        # if first index is 1 then left else right
        if v[1] == 1:
            idx.append([i, v[2]])

    # check root node has decision node then put 0 and count inner_nodes-1 for right side
    # left skew [[1, 'l']], right skew [[1, 'r']]
    if len(idx) == 1:
        # pure left is skew
        if idx[0][1] == 'l':
            print("l: ", idx[0], idx[0][1])
            left = len(inner_nodes)-1
            right = 0
        # pure right is skew
        elif idx[0][1] == 'r':
            print("r: ", idx[0], idx[0][1])
            left = 0
            right = len(inner_nodes) - 1
        # nothing there
        else:
            left = 0
            right = 0
    elif len(idx) == 2:
        # use second position depth to separate data, with ndarray
        # eg [[1, 'l'], [9, 'r']]
        print("2", idx, len(idx))
        left = len(inner_nodes[1:idx[1][0]])
        right = len(inner_nodes[idx[1][0]:])
    elif len(idx) == 0:
        print("0", idx, len(idx))
        left = 0
        right = 0
    else:
        left = 0
        right = 0
    return left, right



def store_results(dataset, filename, k_fold, dim, algorithm, epochs, min_leaf_point, n_features, run, train_data,
                  test_data, model, max_depth, inner_node, leaf_node, all_nodes, left_size, right_size, runtime,
                  pkl_location_name):
    # result of train/test matrix
    if algorithm == "lr_mvdt" or algorithm == "rs_mvdt":
        # model evaluation on training data
        y_pred_train = predict(train_data[0], model)
        # model evaluation on testing data
        y_pred_test = predict(test_data[0], model)
    else:
        y_pred_train = model.predict(train_data[0])
        y_pred_test = model.predict(test_data[0])
    # result data in dictionary for both test and train sets

    new_result = {'dataset': dataset,  # name of dataset
                  'filename': filename,  # dataset file name
                  'k_fold': k_fold,  # which K_fold number
                  'd2v_vec_size': dim,  # dod2vec feature dimension
                  'algorithm': algorithm,  # algorithm name
                  'epochs': epochs,  # no of epochs
                  'min_leaf_point': min_leaf_point,  # given depth of tree
                  'feature_size': n_features,  # how may features are taken
                  'd2v_shape': np.array([list(train_data[0].shape), list(test_data[0].shape)]),  # d2v feature shape both
                  'run': run,  # which iteration we set 10 times
                  'accuracy': np.array([accuracy_score(train_data[1], y_pred_train), accuracy_score(test_data[1],
                                                                                                    y_pred_test)]),
                  # accuracy of both train and test
                  'precision': np.array([precision_score(train_data[1], y_pred_train), precision_score(test_data[1],
                                                                                                       y_pred_test)]),
                  # precision of both train and test
                  'recall': np.array([recall_score(train_data[1], y_pred_train), recall_score(test_data[1],
                                                                                              y_pred_test)]),
                  # recall of both train and test
                  'f1': np.array([f1_score(train_data[1], y_pred_train), f1_score(test_data[1], y_pred_test)]),
                  # f1 of both train and test
                  'confusion_matrix': np.array([list(confusion_matrix(train_data[1], y_pred_train)),
                                                list(confusion_matrix(test_data[1], y_pred_test))]),
                  # to see each class prediction
                  'max_depth': max_depth,  # max depth from model
                  'inner_node': inner_node,  # no of decision nodes (root+inner)
                  'leaf_node': leaf_node,  # no of predicted nodes (decision node)
                  'all_node': all_nodes,  # sum of inner_node and all_node
                  'train_true_predict': str([list(train_data[1]), list(y_pred_train)]),  # training true and predicted value
                  'test_true_predict': str([list(test_data[1]), list(y_pred_test)]),  # test true and predicted
                  'branch_sizes': str([left_size, right_size]),  # left and right branch inner nodes
                  'training_time': runtime,  # training time
                  'pkl_location_name': pkl_location_name  # tree model name and location
                  }

    df_new_result = pd.DataFrame().append(new_result, ignore_index=True)
    insert_result(df_new_result)


def insert_result(new_result):
    filename = "results.csv"
    file_exists = os.path.isfile(filename)


    if not file_exists:
        print("csv file is created")

        df_create_result = pd.DataFrame(
            columns=['dataset', 'filename', 'k_fold', 'd2v_vec_size', 'algorithm', 'epochs', 'min_leaf_point',
                     'feature_size', 'd2v_shape', 'run', 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix',
                     'max_depth', 'inner_node', 'leaf_node', 'all_node', 'train_true_predict', 'test_true_predict',
                     'branch_sizes', 'training_time', 'pkl_location_name'])

        df_create_result.to_csv(filename, index=None)

    # insert new result
    df_results = pd.read_csv(filename)

    # finding new row index for inserting new data
    #df_new_result = pd.DataFrame().append(new_result)
    #df_new_result = pd.DataFrame(new_result, index =[df_results['sn'].tolist()[-1]+1])
    #type(df_new_result.d2v_shape)

    # now adding new row by concatenating data frame
    df_results = pd.concat([df_results, new_result], sort=False)

    # save result
    df_results.to_csv(filename, sep=',', index=None)
