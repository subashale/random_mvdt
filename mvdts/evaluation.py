import os.path
import pandas as pd
from mvdts.dt_common.prediction import print_model, predict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from mvdts.call_algo import fit
import numpy as np

def evaluation(dataset, filename, k_fold, d2v_vec_size, algorithm, epochs, depth, n_features, run, train_data, test_data):

    print("Fitting on, dataset: {}, filename: {}, k_fold: {}, vector_size: {},"
          " algorithm: {}, epochs: {}, depth: {}, n_feature: {}, run: {}, train_data_shape: {}".format(
        dataset, filename, k_fold, d2v_vec_size, algorithm, epochs, depth, n_features, run, str(train_data[0].shape)
    ))

    tree = fit(algorithm, train_data, epochs, depth, n_features)
    # getting tree information
    if algorithm == "lr_mvdt" or algorithm == "rs_mvdt":
        inner_node, leaf_node, all_nodes, max_depth = get_mvdt_model_info(tree)
    else:
        inner_node, leaf_node, all_nodes, max_depth = get_cart_model_info(tree)
    # storing all results
    store_results(dataset, filename, k_fold, d2v_vec_size, algorithm, epochs, depth, n_features, run, train_data,
                  test_data, tree, max_depth, inner_node, leaf_node, all_nodes)

    print("--------- model evaluation information saved --------- ")

def get_cart_model_info(model):
    inner_node = model.tree_.node_count

    # getting leaf nodes https://notebooks.gesis.org/binder/jupyter/user/scikit-learn-scikit-learn-pw3uja99/lab
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    is_leaves = np.zeros(shape=inner_node, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    leaf_node = len(np.where(is_leaves == True)[0])
    all_nodes = inner_node+leaf_node
    max_depth = model.tree_.max_depth

    return inner_node, leaf_node, all_nodes, max_depth


def get_mvdt_model_info(model):
    n_nodes, max_depth, leaf_nodes = print_model(model, root_node=[], leaf_node=[])
    # root decision node includes all nodes(lr) and leaf decision nodes
    inner_node = len(n_nodes - 1)  # it includes root node so -1 to count leaf node
    leaf_node = len(leaf_nodes)
    all_nodes = inner_node + leaf_node

    return inner_node, leaf_node, all_nodes, max_depth

def store_results(dataset, filename, k_fold, dim, algorithm, epochs, depth, n_features, run, train_data, test_data, model, max_depth, inner_node, leaf_node, all_nodes):
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

    new_result = {'dataset': str(dataset),  # name of dataset
                  'filename': str(filename),  # dataset file name
                  'k_fold': str(k_fold),  # which K_fold number
                  'd2v_vec_size': str(dim),  # dod2vec feature dimension
                  'algorithm': str(algorithm),  # algorithm name
                  'epochs': str(epochs),  # no of epochs
                  'depth': str(depth),  # given dpeth of tree
                  'feature_size': str(n_features),  # how may features are taken
                  'run': str(run),  # which iteration we set 10 times
                  'd2v_shape': str([train_data[0].shape, test_data[0].shape]),  # d2v feature shape both
                  'accuracy': str([accuracy_score(train_data[1], y_pred_train), accuracy_score(test_data[1], y_pred_test)]),  # accuracy of both train and test
                  'precision': str([precision_score(train_data[1], y_pred_train), precision_score(test_data[1], y_pred_test)]),  # precision of both train and test
                  'recall': str([recall_score(train_data[1], y_pred_train), recall_score(test_data[1], y_pred_test)]),  # recall of both train and test
                  'f1': str([f1_score(train_data[1], y_pred_train), f1_score(test_data[1], y_pred_test)]),  # f1 of both train and test
                  'confusion_matrix': str([confusion_matrix(train_data[1], y_pred_train), confusion_matrix(test_data[1], y_pred_test)]), # to see each class prediction
                  'max_depth': str(max_depth),  # max depth from model
                  'inner_node': str(inner_node),  # no of decision nodes (root+inner)
                  'leaf_node': str(leaf_node),  # no of predicted nodes (decision node)
                  'all_node': str(all_nodes),  # sum of inner_node and all_node
                  }
    insert_result(new_result)


def insert_result(new_result):

    filename = "results.csv"
    file_exists = os.path.isfile(filename)

    # if file not exits create a csv file then insert new result
    if not file_exists:
        print("csv file is created")
        # create csv file
        df_create_result = pd.DataFrame(
            columns=['dataset', 'filename', 'k_fold', 'd2v_vec_size', 'algorithm', 'epochs', 'depth', 'run', 'd2v_shape',
                     'feature_size', 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'max_depth', 'inner_node', 'leaf_node',
                     'all_node'])
        df_create_result.to_csv(filename, index=None)

    # insert new result
    df_results = pd.read_csv(filename)
    df_new_result = pd.DataFrame(new_result, index=[len(df_results) + 1])
    # df_new_result = pd.DataFrame(new_result, index =[df_results['sn'].tolist()[-1]+1])

    # now adding new row by concatenating data frame
    df_results = pd.concat([df_results, df_new_result], sort=False)

    # do something else
    df_results.to_csv(filename, index=None)

    #return df_results