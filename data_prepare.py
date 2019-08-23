# this part is responsible for preparing data and giving as needed
import numpy as np
import pandas

# this function is responsible for giving data to split

def pos_neg_giver(split_list):
    # all feature are in index 0 and 1 has class label    
    # get index of category(0, 1) from label
    idx_1 = np.where(split_list[1] == 1)
    idx_0 = np.where(split_list[1] == 0)

    # get all feature value of category
    pos = split_list[0][idx_1]
    neg = split_list[0][idx_0]

    return pos, neg

def data_giver(df):
    # select only feature and label in separate list
    X = np.array(df[df.columns[:-1]].values.tolist())
    label = np.array(df[df.columns[-1]].values.tolist())

    # get index of 1 and 0 from label
    idx_1 = np.where(df.label == 1)
    idx_0 = np.where(df.label == 0)

    # get all positive value and negative value based on index of idx_1 and idx_0
    pos = np.array(df.loc[idx_1[0], ["x1", "x2"]].values.tolist())
    neg = np.array(df.loc[idx_0[0], ["x1", "x2"]].values.tolist())

    return X, label, pos, neg

