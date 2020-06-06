import pandas as pd
import numpy as np
import os
# from oc_eval import evaluation

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

def occartlc(datasets, k, vector_sizes, algorithms):
    total = 0
    root_folder = "data/"
    for name in datasets.split(","):
        # selecting root folder of each dataset
        data_name_folder = root_folder + name + "/"
        for n in range(1):
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
                    for algorithm in algorithms:
                        # evaluation(name, k_data_name_folder_final, [d2vDim_name_train, d2vDim_name_test], count,
                        # dim, algorithm, train, test)
                            # print(len(df_train), len(df_test))
                        pass
                else:
                    print("datasets not found: ", k_data_name_folder_final+d2vDim_name_train)
                    #pass
    print(total)

if __name__ == '__main__':
    datasets = "20ng_CG_RM"
    k = 5
    vector_sizes = [10]  # 10, 25, 50, 75, 100
    algorithms_list = ['oc1', 'cart']

    occartlc(datasets, k, vector_sizes, algorithms_list)