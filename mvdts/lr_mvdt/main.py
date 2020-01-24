from mvdts.dt_common.nodes import Leaf, DecisionNode
from mvdts.dt_common.common import random_features_selection, check_purity, partition, entropy
import numpy as np
from sklearn.linear_model import LogisticRegression

def best_split(rows, epochs):
    # solvers ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    # https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
    clf = LogisticRegression(solver='liblinear', max_iter=epochs)

    # rows[0]: features, rows[1]:label
    clf.fit(rows[0], rows[1])

    theta = np.append(clf.coef_, clf.intercept_)
    pred = clf.predict(rows[0])

    return clf, theta, pred

def build_tree(rows, epochs, min_point, max_depth=0, depth=0, noOfFeature=2):
    # check for max depth of tree if yes then stop building
    label_1, label_0 = check_purity(rows[-1])
    # if max depth is given
    if max_depth != 0:
        if depth > max_depth:
            print("from max depth reached, depth {} reached, 1:{}, 0:{}, min_point: {}".format(depth, label_1, label_0, min_point))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))

    if label_1 > min_point and label_0 > min_point:

        # random feature pair selection, and host idx and rand_x
        idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)

        model, theta, pred = best_split([rand_x, rows[-1]], epochs)
        # viz_data_with_line_np(theta, rows)
        # getting index of pos and neg side to further split in list
        pos_side = np.where(pred > 0)
        neg_side = np.where(pred <= 0)

        # getting left and right branch subset after split
        true_rows, false_rows = partition(rows, pos_side, neg_side)

        # entropy measure
        e = entropy(len(pos_side[0]), len(neg_side[0]))

        #print(len(true_rows[0]), len(false_rows[0]))
        # checking purity of each branch
        left_1, left_0 = check_purity(true_rows[-1])
        right_1, right_0 = check_purity(false_rows[-1])
        print("depth {}, entropy: {}, total rows:{} left branch labels [1={}, 0={}],right branch labels [1={}, 0={}],"
              " min_point: {}\n".format(depth, e, len(rows[0]),left_1, left_0, right_1, right_0, min_point))
        # this condition same as entropy == 0
        # if (left_0 == 0 and left_1 == 0) or (right_0 == 0 and right_1 == 0):
        #     #print(left_0, left_1, right_0, right_1)
        #     #print(len(true_rows[1]), len(false_rows[1]), depth)
        #     print("left01 right01, depth {} reached, 1:{}, 0:{}, min_point: {}"
        #           .format(depth, label_1, label_0, min_point))
        #     return Leaf(rows[-1].reshape(len(rows[0]), 1))
        #
        # else:
        #     true_branch = build_tree(true_rows, epochs, min_point, depth=depth + 1,
        #                              noOfFeature=noOfFeature)
        #     false_branch = build_tree(false_rows, epochs, min_point, depth=depth + 1,
        #                               noOfFeature=noOfFeature)
        #
        #     return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0],
        #                         depth, e)

        # while empty prediction on true and false branch, stopping criterion
        # if there is no improvement or beyond data point prediction then
        # this condition will take place
        if e == 0:
            print("out of boundary decision", e)
            print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
                  .format(depth, label_1, label_0, min_point))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))
        else:
            true_branch = build_tree(true_rows, epochs, min_point, depth=depth + 1,
                                     noOfFeature=noOfFeature)
            false_branch = build_tree(false_rows, epochs, min_point, depth=depth + 1,
                                      noOfFeature=noOfFeature)
            return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0], depth
                                , e)
    else:
        print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
              .format(depth, label_1, label_0, min_point))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))
