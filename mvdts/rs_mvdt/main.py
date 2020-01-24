# headers import
import random
import numpy as np

# calling helper funtions
from mvdts.rs_mvdt.data_split import split_data, pos_neg_giver
from mvdts.dt_common.nodes import Leaf, DecisionNode
from mvdts.dt_common.common import random_features_selection, check_purity, partition, entropy


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
            print(type(theta[0]), type(theta))
            print("nan found in theta: {}, pos_side: {}, neg_side: {}, gain: {}, epoch: {}"
                  .format(theta, pos_side, neg_side, gain, epoch))

        # pred = current_prediction(best_theta, [rand_x, rows[1]])

    return best_pos_side, best_neg_side, best_theta

def build_tree(rows, epochs, min_point, max_depth=0, depth=0, noOfFeature=2):
    # check for max depth of tree if yes then stop building
    label_1, label_0 = check_purity(rows[1])
    # if max depth is given
    if max_depth != 0:
        if depth > max_depth:
            print("from max depth reached, depth {} reached, 1:{}, 0:{}, min_point: {}".format(depth, label_1, label_0,
                                                                                               min_point))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))

    if label_1 > min_point and label_0 > min_point:
        # random feature pair selection, and host idx and rand_x
        idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)

        # positive and negative feature points
        pos, neg = pos_neg_giver([rand_x, rows[1]])
        print(len(pos), len(neg))

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
        print("depth {}, total rows:{} left branch labels [1={}, 0={}],right branch [1={}, 0={}]\n"
              .format(depth, len(rows[0]),left_1, left_0, right_1, right_0))

        # while empty prediction on true and false branch, stopping criterion
        # if there is no improvement or beyond data point prediction then
        # this condition will take place
        # if e == 0:
        #     print("out of boundary decision", e)
        #     print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
        #           .format(depth, label_1, label_0, min_point))
        #     return Leaf(rows[-1].reshape(len(rows[0]), 1))
        # else:
        true_branch = build_tree(true_rows, epochs, min_point, depth=depth + 1,
                                 noOfFeature=noOfFeature)
        false_branch = build_tree(false_rows, epochs, min_point, depth=depth + 1,
                                  noOfFeature=noOfFeature)
        return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0],
                            depth, e)

    else:
        print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
              .format(depth, label_1, label_0, min_point))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))




