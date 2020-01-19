import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os


def k_fold_data(X, y, name, k):
    kf = KFold(n_splits=k)
    c = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        c += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # storing k_fold data on their dir
        save_k_fold_split([X_train, y_train], [X_test, y_test], name, str(c))


def save_k_fold_split(train, test, name, count):
    # now save data to csv file with iteration of k fold and dataset name
    #     1. Create folder name after name
    #     2. subfolder using name_k1
    #     3. inside name_kCount store csv file as name_k1_train, name_k2_train
    # raw folder contents pure file after k fold split on test and train
    # model contains doc2vec model
    # embedding contains inference vectors for both test and train
    root_folder = "data/"
    data_name_folder = root_folder + name + "/"
    k_data_name_folder = data_name_folder + "k" + count + "/raw/"
    raw_name_train = "raw_"+name + "_k" + count + "_train.csv"
    raw_name_test = "raw_"+name + "_k" + count + "_test.csv"

    if ensure_dir(k_data_name_folder):
        # create dataframe for test and train and save inside raw folder of each k fold
        train_df = pd.DataFrame(np.column_stack([train[0], train[1]]), 
                               columns=['text', 'label'])
        test_df = pd.DataFrame(np.column_stack([test[0], test[1]]), 
                               columns=['text', 'label'])
        # saving data frame
        train_df.to_csv(k_data_name_folder + raw_name_train, index=False)
        test_df.to_csv(k_data_name_folder + raw_name_test, index=False)

        print("Fle saved:", name, count, raw_name_train, raw_name_test)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("{} folder is created".format(directory))
        return True
    else:
        print("{} folder is exists".format(directory))
        return True


def split_data(data_list, k):
    # if single data use convert to list
    if not type(data_list) == list:
        data_list = data_list.split(",")
    # looping for every dataset
    for name in data_list:
        data_location = "data/"+name+".csv"

        if os.path.exists(data_location):
            df = pd.read_csv(data_location)
            X = np.array(df[df.columns[:-1]].values.tolist())
            y = np.array(df[df.columns[-1]].values.tolist())
            k_fold_data(X, y, name, k)
        else:
            print("{} file or path doesn't exists".format(data_location))

# use , to separate datasets but with out whitespaceor put in list
# data = "circle,3rd"
# split_data(data)