from mvdts.dt_common.nodes import Leaf, DecisionNode
from mvdts.dt_common.common import random_features_selection, check_purity, partition
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

def build_tree(rows, epochs, depth=0, max_depth=4, noOfFeature=2, min_point=20):
    # check for max depth of tree if yes then stop building
    label_1, label_0 = check_purity(rows[-1])
    if depth > max_depth:
        print("depth {} reached, 1:{}, 0:{}".format(depth, label_1, label_0))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))

    if label_1 > min_point and label_0 > min_point:
        # random feature pair selection, and host idx and rand_x
        idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)

        model, theta, pred = best_split([rand_x, rows[-1]], epochs)
        # viz_data_with_line_np(theta, rows)
        # getting index of pos and neg side to further split
        pos_side = np.where(pred > 0)
        neg_side = np.where(pred <= 0)

        true_rows, false_rows, left_count, right_count = partition(rows, pos_side, neg_side)
        print(len(true_rows[0]), len(false_rows[0]))
        # checking purity of each branch
        left_1, left_0 = check_purity(true_rows[-1])
        right_1, right_0 = check_purity(false_rows[-1])
        print("depth {}, total rows:{} left branch labels [1={}, 0={}],right branch [1={}, 0={}]\n".format(depth, len(rows[0]),left_1, left_0, right_1, right_0))

        # while empty prediction
        if (left_0 == 0 and left_1 == 0) or (right_0 == 0 and right_1 == 0):
            print(left_0, left_1, right_0, right_1)
            print(len(true_rows[1]), len(false_rows[1]), depth)
            print("depth {} reached, 1:{}, 0:{}".format(depth, label_1, label_0))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))

        else:
            true_branch = build_tree(rows=true_rows, epochs=epochs, depth=depth + 1, max_depth=max_depth,
                                     noOfFeature=noOfFeature, min_point=20)
            false_branch = build_tree(rows=false_rows, epochs=epochs, depth=depth + 1, max_depth=max_depth,
                                      noOfFeature=noOfFeature, min_point=20)

            return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0], depth)

    else:
        print("depth {} reached, 1:{}, 0:{}".format(depth, label_1, label_0))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))
