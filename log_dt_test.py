import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import data_viz as dviz


# loading the data
# data ="data/weight-height.csv"
# data = "data/circle.csv"

data = "data/triangle.csv"
# data = "data/linear.csv"
# data = "data/024f2liris.csv"
# data = "data/3rd.csv"
df = pd.read_csv(data)

x = np.array(df[df.columns[:-1]].values.tolist(), dtype=np.float64)
y = np.array(df[df.columns[-1]].values.tolist())
data = np.c_[x, y]

dviz.DDScatterDFSns(0, 1, df)


class Question:
    # initialise column and value variables->
    # eg->if ques is ->is sepal_length>=1cm then
    # sepal_length==col and 1cm=value
    def __init__(self, question, value=0):
        self.question = question
        self.value = value

    def match(self, data):
        value = data[self.column]
        return value >= self.value

    # This is just a helper method to print
    # the question in a readable format.
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (
            str(self.theta), condition, str(self.value))


# this class represents all nodes in the tree
class DecisionNode:
    def __init__(self, question, true_branch, false_branch, rows):
        # question object stores col and val variables regarding the question of that node
        self.question = question  # question = theta
        # this stores the branch that is true
        self.true_branch = true_branch
        # this stores the false branch
        self.false_branch = false_branch
        # store split parts
        self.rows = rows


# Leaf class is the one whichstores leaf of trees
# these are special Leaf Nodes -> on reaching them either
# 100% purity is achieved or no features are left to split upon
class Leaf:
    def __init__(self, label):
        # stores unique labels and their values in prediction
        unique, counts = np.unique(label, return_counts=True)
        self.predictions = dict(zip(unique, counts))


class LeafT:
    def __init__(self, question, rows):
        self.question = question
        self.rows = rows


def best_split(rows):
    clf = LogisticRegression(solver='lbfgs')
    # rows[0]: features, rows[1]:label

    clf.fit(rows[0], rows[1])

    theta = np.append(clf.coef_, clf.intercept_)
    pred = clf.predict(rows[0])

    return clf, theta, pred


def partition(rows, pred):
    left_rows = np.where(pred > 0)
    right_rows = np.where(pred <= 0)    

    Xl = rows[0][left_rows]
    yl = rows[-1][left_rows]

    Xr = rows[0][right_rows]
    yr = rows[-1][right_rows]

    return [Xl, yl], [Xr, yr]

def build_tree(rows, max_depth, depth):
    # takes the whole dataset as argument
    # gets the best gain and best question    

    model, theta, pred = best_split(rows)
    dviz.viz_data_with_line_np(theta, rows)

    if len(rows[-1]) < 15:
        return Leaf(theta, rows)

    if depth >= max_depth:
        # return Leaf(rows[-1])
        return LeafT(theta, rows)

    pos_idx = np.where(y[rows[-1]] == 1)
    neg_idx = np.where(y[rows[-1]] == 0)

    if len(pos_idx[0]) < 20 and len(neg_idx[0]) < 20:
        return LeafT(theta, rows)

        # dviz.viz_data_np(rows)
    # If we reach here, we have found a useful feature / value
    # to partition on.

    true_rows, false_rows = partition([x, y], pred)

    # print(len(true_rows[0]))
    # print("true {}, false_rows {}".format(len(true_rows[-1]), len(false_rows[-1])))
    # Recursively build the true branch.
    tunique, tcounts = np.unique(true_rows[-1], return_counts=True)
    funique, fcounts = np.unique(false_rows[-1], return_counts=True)

    if len(tcounts) == 1 or len(fcounts) == 1:
        # make a leaf object and return
        return LeafT(theta, rows)

    """ Check minimun points here not in the above, 
        check on each split part for both classes it is less than 20 do not split further

    """
    if len(true_rows[-1]) > 20 and len(false_rows[-1]) > 20:
        # if tcounts[0] > 20 and tcounts[1] > 20 or fcounts[0] > 20 and fcounts[1] > 20:

        true_branch = build_tree(true_rows, max_depth, depth + 1)
        false_branch = build_tree(false_rows, max_depth, depth + 1)

        return DecisionNode(theta, true_branch, false_branch, rows)

    else:
        return LeafT(theta, rows)


def print_tree(node, indentation=""):
    '''printing function'''
    # base case means we have reached the leaf
    # if the node object is of leaf type
    if isinstance(node, LeafT):
        print(indentation + "PREDICTION", node.question)
        return
        # print the question at node
    print(indentation + str(node.question))

    # call the function on true branch 
    print(indentation + "Left Branch")
    print_tree(node.true_branch, indentation + "-->")

    # on flase branch
    print(indentation + "Right Branch")
    print_tree(node.false_branch, indentation + "-->")

tree = build_tree([x, y], 4, 1)

print_tree(tree)

def classify(x_point, node):
    """See the 'rules of recursion' above."""
    if isinstance(node, LeafT):
        pt = np.append(x_point, 1).reshape(len(x_point) + 1, 1)
        r = node.question.dot(pt)

        if r >= 0:
            r = 1
        else:
            r = 0
        return r

    pt = np.append(x_point, 1).reshape(len(x_point) + 1, 1)
    r = node.question.dot(pt)

    if r > 0:
        return classify(x_point, node.true_branch)
    else:
        return classify(x_point, node.false_branch)


print(classify(x[35], tree))

def predict(x, tree):
    yh_list = []
    if len(x.shape) == 1:
        return classify(x, tree)
    else:
        for i, value in enumerate(x):
            yh = classify(value, tree)
            yh_list.append(yh)
        return yh_list



y_pred = predict(x, tree)
print(classification_report(y, y_pred, labels=[0, 1]))
print(accuracy_score(y, y_pred))
print(confusion_matrix(y, y_pred))

