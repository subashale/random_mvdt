import numpy
import os
import json
from datetime import datetime
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def create_dir_model(location, new_folder_name):
    # creating directory based on argument
    model_location = location+new_folder_name+"/"
    if not os.path.exists(model_location):
        os.makedirs(model_location)
    return model_location

def call_train(algorithm, X_train, y_train):
    if algorithm == "oc1":
        #calling for oc1
        tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=20, max_perturbations=5,
                           random_state=2)
    else:
        #calling for multivariate cart or cartlc
        tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=5, random_state=2)
    print(tree)
    tree.fit(X_train, y_train)
    return tree

def evaluation(dataset_name, data_location, filename, k_fold, d2v_vec_size, algorithm, train_data, test_data):
    #print(dataset_name, data_location, filename, k_fold, d2v_vec_size, algorithm, min_leaf_point, len(train_data), len(test_data))
    print("Fitting on, dataset: {}, data_location: {}, filename: {}, k_fold: {}, vector_size: {}, algorithm: {},"
          " train_data_shape: {}, test_data_shape: {}".format(dataset_name, data_location,
            filename, k_fold, d2v_vec_size, algorithm, str(train_data[0].shape), str(test_data[0].shape)))

    store_in = create_dir_model(data_location, algorithm)
    pkl_filename = filename[0].split("/")[-1].split(".")[0] + '_' + algorithm + '_' + str(d2v_vec_size)+'.txt'

    # stop saving beacuse of memeory problem
    mode = 'a+' if os.path.exists(store_in + 'tree.txt') else 'w+'
    with open(store_in + 'tree.txt', mode) as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write("\n\nExperiment on: " + dt_string + "\n\n")
        f.write("\n: " + pkl_filename + "\n\n")
        f.write("\n*******************************************************************")

    model = call_train(algorithm, train_data[0], train_data[1])
    y_pred_train = model.predict(train_data[0])
    y_pred_test =  model.predict(test_data[0])
    store_results(dataset_name, filename, k_fold, d2v_vec_size, algorithm, [list(train_data[0].shape), list(test_data[0].shape)],
                  train_data[1], test_data[1], y_pred_train, y_pred_test)

def store_results(dataset, filename, k_fold, dim, algorithm, shapes, y_train, y_test, y_pred_train, y_pred_test):
    new_result = {'dataset': dataset,  # name of dataset
                  'filename': filename,  # dataset file name
                  'k_fold': int(k_fold),  # which K_fold number
                  'd2v_vec_size': int(dim),  # dod2vec feature dimension
                  'algorithm': algorithm,  # algorithm name
                  'd2v_shape': shapes,  # d2v feature shape both
                  #Evaluation Matrxi
                  'accuracy': [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)],
                  # accuracy of both train and test
                  'precision': [precision_score(y_train, y_pred_train), precision_score(y_test, y_pred_test)],
                  # precision of both train and test
                  'recall': [recall_score(y_train, y_pred_train), recall_score(y_test, y_pred_test)],
                  # recall of both train and test
                  'f1': [f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)],
                  # f1 of both train and test
                  'confusion_matrix': [confusion_matrix(y_train, y_pred_train).tolist(),
                                       confusion_matrix(y_test, y_pred_test).tolist()],
                  }

    # now saving and loading to json.pk file
    save_results(new_result, "store_results")

def save_results(new_data, isfrom):
    # saving information in results.json if it is from store_results
    # else other, but now it is combined
    if isfrom == "store_results":
        result_file = "results.json"
    else:
        result_file = "intermediate_result.json"

    # result file is not exits
    if not os.path.exists(result_file):
        temp = {"results": []}
        with open(result_file, 'w+') as json_result:
            json.dump(temp, json_result)

    with open(result_file) as json_file:
        data = json.load(json_file)
        temp = data["results"]
        # python object to be appended
        temp.append(new_data)

    # then update data
    with open(result_file, 'w') as f:
        json.dump(data, f, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)