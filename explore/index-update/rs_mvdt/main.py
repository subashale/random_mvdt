# headers import
import random
import numpy as np

# calling helper funtions
from rs_mvdt.data_split import split_data, pos_neg_giver
from dt_common.nodes import Leaf, DecisionNode
from dt_common.common import random_features_selection, check_purity, partition, entropy

import matplotlib.pyplot as plt


def viz_data_with_line_np(theta, split_list):
    # this will show decision boundary as line with scatter plot
    
    theta = theta.reshape(split_list[0].shape[1] + 1, 1)
    
    # Ploting Line, decision boundary
    theta_f = list(theta.flat)
    
    # Calculating line values x and y
    # y = np.arange(-10, 10, 0.1)
    # x = (-theta_f[2] - theta_f[1] * y) / theta_f[0]

    # ref #https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    # https://stackoverflow.com/questions/42704698/logistic-regression-plotting-decision-boundary-from-theta

    x1_line = np.arange(int(round(split_list[0].min())), int(round(split_list[0].max())), 0.1)
    x2_line = (-theta_f[2] - theta_f[0] * x1_line) / theta_f[1]
    # x2 = - (theta_f[2] + np.dot(theta_f[0], x)) / theta_f[1]
    plt.plot(x1_line, x2_line, label='Decision Boundary')

    # zooming plots
    x1 = split_list[0][:, 0]
    x2 = split_list[0][:, 1]

    # x1.min-1 left, x2.max+1: bottom, x1.max+1: right:, x2.min-7: top 
    X_min_max = [int(round(x1.min() - 4)), int(round(x1.max() + 4))]
    X2_min_max = [int(round(x2.min() - 4)), int(round(x2.max() +4))]
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(X2_min_max[0], X2_min_max[1])

    # scatter plot
    categories = split_list[1]
    colormap = np.array(['#277CB6', '#FF983E'])
    
    plt.scatter(x1, x2, c=colormap[categories])

def plot_data(theta, split_list):
    weights = theta
    inputs = split_list[0]
    targets = split_list[1] 
    # fig config
#     plt.figure(figsize=(8,8))
    plt.grid(True)

    #plot input samples(2D data points) and i have two classes. 
    #one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input,target in zip(inputs,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        theta = weights.reshape(inputs.shape[1] + 1, 1)
    
        # Ploting Line, decision boundary
        theta_f = list(weights.flat)

        x1_line = np.arange(int(round(inputs.min())), int(round(inputs[0].max())), 0.1)
        x2_line = (-theta_f[2] - theta_f[0] * x1_line) / theta_f[1]
    
        # x2 = - (theta_f[2] + np.dot(theta_f[0], x)) / theta_f[1]
        plt.plot(x1_line, x2_line, label='Decision Boundary')
        
def get_best(epochs, rand_x, label, pos, neg):
    # best_gain; highest the information gain will be good
    best_gain = 0
    # best_theta = coefficient or line
    best_theta = []
    # best_pos_side, best_neg_side = data points for pos and negative after the best_gain split
    best_pos_side = []
    best_neg_side = []
    for epoch in range(epochs):
        # find random feature points as linear combination

        # selecting points from them
        p = random.choice(pos)
        n = random.choice(neg)

        # getting parameters from split fitting line
        gain, pos_side, neg_side, theta = split_data(rand_x, label, pos, neg, p, n)

        # checking for best gain and setting rest parameters of it
        # if theta has not nan value the check  for best else do nothing
        if not np.isnan(theta[0]):
            if gain > best_gain:
                best_gain = gain
                best_pos_side = pos_side
                best_neg_side = neg_side
                best_theta = theta
        else:
#             print(type(theta[0]), type(theta))
#             print("nan found in theta: {}, pos_side: {}, neg_side: {}, gain: {}, epoch: {}"
#                   .format(theta, pos_side, neg_side, gain, epoch))

        # pred = current_prediction(best_theta, [rand_x, rows[1]])
            pass

    return best_pos_side, best_neg_side, best_theta

def build_tree(rows, epochs, min_point=2, max_depth=4, depth=0, noOfFeature=2):
    # check for max depth of tree if yes then stop building
    label_1, label_0 = check_purity(rows[1])
    # if max depth is given
    if max_depth != 0:
        if depth > max_depth:
#             print("from max depth reached, depth {} reached, 1:{}, 0:{}, min_point: {}".format(depth, label_1, label_0,
#                                                                                                min_point))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))

    if label_1 > min_point and label_0 > min_point:
        # random feature pair selection, and host idx and rand_x
        idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)

        # positive and negative feature points
        pos, neg = pos_neg_giver([rand_x, rows[1]])
#         print(len(pos), len(neg))

        pos_side, neg_side, theta = get_best(epochs, rand_x, rows[1], pos, neg)
        #print(len(pos_side), len(neg_side), len(theta), type(theta))
        #viz_data_with_line_np(theta, rows)

        # separating left and right side data
        true_rows, false_rows = partition(rows, pos_side, neg_side)

        # entropy measure
        e = entropy(len(pos_side), len(neg_side))

        # checking purity of each branch
        left_1, left_0 = check_purity(true_rows[-1])
        right_1, right_0 = check_purity(false_rows[-1])
#         print("depth {}, total rows:{} left branch labels [1={}, 0={}],right branch [1={}, 0={}]\n"
#               .format(depth, len(rows[0]),left_1, left_0, right_1, right_0))

        # while empty prediction on true and false branch, stopping criterion
        # if there is no improvement or beyond data point prediction then
        # this condition will take place
        # if e == 0:
        #     print("out of boundary decision", e)
        #     print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
        #           .format(depth, label_1, label_0, min_point))
        #     return Leaf(rows[-1].reshape(len(rows[0]), 1))
        # else:
        if len(rows[0][0]) == 2:
            viz_data_with_line_np(theta, rows)
#             plot_data(theta, rows)


        true_branch = build_tree(true_rows, epochs, min_point, depth=depth + 1,
                                 noOfFeature=noOfFeature)
        false_branch = build_tree(false_rows, epochs, min_point, depth=depth + 1,
                                  noOfFeature=noOfFeature)
        return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0],
                            depth, e)

    else:
#         print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
#               .format(depth, label_1, label_0, min_point))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))




