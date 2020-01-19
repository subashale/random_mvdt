import random
import numpy as np


# for both lrdt , rmvdt
def random_features_selection(features, noOfFeature=2, typeOfRandom=1):
    """

    Ways of selecting random features
    :param features: X values
    :param noOfFeature: random choice from randint(2, len(X[0])
    :param typeOfRandom: 1: for continue same number of features to create decsion node else choose random every time
    :return: selected feature index, noOfFeature, typeOfRandom, value of selected feature
    """

    # if no of feature selection empty or out of bound then select all feature
    # feature has both X,y as list so we use features[0] to select only features value
    #print(len(features[0]), len(features[0][0]))

    if noOfFeature == '' or noOfFeature > len(features[0][0]):
        noOfFeature = len(features[0][0])

    # if no of feature is <= 1 then choose two features as default
    if noOfFeature <= 1:
        noOfFeature = 2

    # if typeOfRandom is beyond 0 or 1 then choose default 0
    if typeOfRandom < 0:
        typeOfRandom = 0

    if typeOfRandom > 1:
        typeOfRandom = 1

    # use this return value to process further
    featureIndexes = random.sample(range(0, len(features[0][0])), noOfFeature)

    # ascending ordering index
    featureIndexes.sort()

    # feature gathering
    newFeatureSet = []
    for index in featureIndexes:
        newFeatureSet.append(np.row_stack(features[0][:, index]))

    return featureIndexes, noOfFeature, typeOfRandom, np.column_stack(newFeatureSet)

def check_purity(data):
    """
    counting occurance of each label on given data
    :param data:
    :return:
    """
    # # auto count any number of label
    # return np.unique(data, return_counts=True)
    #
    # # then in called function
    # for label, count in zip(unique[0], unique[1]):
    #     print("label {}:{}".format(label, count))

    return len(np.where(data > 0)[0]), len(np.where(data <= 0)[0])

def partition(rows, pos_side, neg_side):
    # get the index of each 0 and 1 rows from predict function

    # now separate data using pos and neg side rows index to make new data for next node of operation
    # for left node find feature from pos index and also for label
    Xl = rows[0][pos_side]
    yl = rows[-1][pos_side]

    # for right node find feature from left index and also for label
    Xr = rows[0][neg_side]
    yr = rows[-1][neg_side]

    return [Xl, yl], [Xr, yr], len(pos_side), len(neg_side)
