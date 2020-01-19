import os
from prepare_features.d2v_train_test import get_vector


def buidling_d2v_kfold(data, k, vector_sizes):
    # looking for k_fold sets and building features sets and storing
    root_folder = "data/"
    for name in data.split(","):
        # put k as same as used in k fold
        data_name_folder = root_folder + name + "/"
        for n in range(k):
            count = str(n + 1)
            k_data_name_folder_raw = data_name_folder + "k" + count + "/raw/"
            raw_name_train = "raw_" + name + "_k" + count + "_train.csv"
            raw_name_test = "raw_" + name + "_k" + count + "_test.csv"

            # train and test file with location
            train = k_data_name_folder_raw + raw_name_train
            test = k_data_name_folder_raw + raw_name_test

            # save models and feature vectors, location set
            k_data_name_folder_model = data_name_folder + "k" + count + "/model/"
            k_data_name_folder_final = data_name_folder + "k" + count + "/final/"

            # checking dir exists else mkdir
            if not os.path.exists(k_data_name_folder_model):
                os.makedirs(k_data_name_folder_model)
            if not os.path.exists(k_data_name_folder_final):
                os.makedirs(k_data_name_folder_final)

            # for list of vector_size
            for vector_size in vector_sizes:

                # print(train, test)
                if os.path.exists(k_data_name_folder_raw + raw_name_train):

                    # training model and getting results as model, vectors in df form
                    model, train_df, test_df = get_vector(train, test, vector_size)

                    # store_results(model, train_df, test_df k_data_name_folder_model, k_data_name_folder_final)
                    dim = str(vector_size)
                    # file names
                    d2vDim_name_model = dim + "D_" + name + "_model"
                    d2vDim_name_train = dim + "D_" + name + "_k" + count + "_train.csv"
                    d2vDim_name_test = dim + "D_" + name + "_k" + count + "_test.csv"

                    # store results
                    model.save(k_data_name_folder_model + d2vDim_name_model)
                    train_df.to_csv(k_data_name_folder_final + d2vDim_name_train, index=False)
                    test_df.to_csv(k_data_name_folder_final + d2vDim_name_test, index=False)
                    print("model and feature vectors are saved", d2vDim_name_model, d2vDim_name_train, d2vDim_name_test)
                    # store_results(test_model, test_df, k_data_name_folder_model, k_data_name_folder_final, "test.csv")
                else:
                    print("{}, {} file doesn't exists".format(train, test))
# data = "20ng_CG_RM,20newsgroup"
# d2v_vector_sizes = [10, 20, 30, 40, 50]
# buidling_d2v_kfold(data, 3, d2v_vector_sizes)