import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import data_viz as dviz



# read data as dataframe
# data = "data/linear.csv"
data = "data/weight-height.csv"
# data = "data/124f2liris.csv"
# data = "data/triangle.csv"
df = pd.read_csv(data)
#df.variety.value_counts()

#next for iris only
# for iris only taking two class droping other
# df = df[df['variety'] != 'Setosa']
# #label encoding
# df['variety'].replace(["Virginica","Versicolor"], [1,0], inplace=True)


inp_df = df.drop(df.columns[-1], axis=1)
out_df = df.drop(df.columns[:-1], axis=1)
# scaler = StandardScaler()
# inp_df = scaler.fit_transform(inp_df)
#
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.98, random_state=43)

X_tr_arr = X_train.values
X_ts_arr = X_test.values
# df to np_array
y_tr_arr = y_train.values.ravel()
y_ts_arr = y_test.values.ravel()

# x_tr = pd.DataFrame(X_train, columns=['x1','x2'])
x = pd.concat([X_train, y_train], axis=1)

dviz.DDScatterDFSns(0, 1, x)


def find_best_split(X, y):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)

    theta = np.append(clf.coef_, clf.intercept_)
    pred = clf.predict(X)

    return clf, theta, pred


def partition(y, side):
    # check node's purity
    pos_idx = np.where(y[side] == 1)
    neg_idx = np.where(y[side] == 0)

    return pos_idx, neg_idx

def getXy(X2, y2, side):
    X = X2[side]
    y = y2[side]
    return X, y


min_pts = 5
tree_depth = 4

min_pts = 5
tree_depth = 4
true_branch = []
false_branch = []

#this class represents all nodes in the tree
class DecisionNode:
    def __init__(self,theta,true_branch,false_branch):
        #question object stores col and val variables regarding the question of that node
        self.theta = theta
        #this stores the branch that is true
        self.true_branch = true_branch
        #this stores the false branch
        self.false_branch = false_branch


class Leaf:
    def __init__(self, label):
        # stores unique labels and their values in prediction
        unique, counts = np.unique(label, return_counts=True)

        self.predictions = dict(zip(unique, counts))


min_pts = 10
tree_depth = 4


def build_tree(X, y):
    # Base case:  pure node found
    # we'll return a leaf. because clf take two class to split
    # if 1 then it is pure node

    model, theta, pred = find_best_split(X, y)

    # dviz.viz_data_with_line_np(theta, [X, y])
    # get index of pos and neg from both left and right side
    # splitting data in to left and righ side, takes index only
    left_side = np.where(pred == 1)
    right_side = np.where(pred == 0)

    print(len(left_side[0]), len(right_side[0]))
    # we can check depth of tree
    # left_side

    pos_idx, neg_idx = partition(y_tr_arr, left_side)
    print(pos_idx)
    print("left:{}, pos:{},neg:{}".format(len(left_side[0]), len(pos_idx[0]), len(neg_idx[0])))

    x2, y2 = getXy(X_tr_arr, y_tr_arr, left_side)
    print(y2)
    if len(pos_idx[0]) < min_pts or len(neg_idx[0]) < min_pts:
        return Leaf(y2)
    true_branch = build_tree(x2, y2)

    # right side
    pos_idx, neg_idx = partition(y_tr_arr, right_side)
    print("Right:{}, pos:{},neg:{}".format(len(right_side[0]), len(pos_idx[0]), len(neg_idx[0])))
    x2, y2 = getXy(X_tr_arr, y_tr_arr, left_side)

    if len(pos_idx[0]) < min_pts or len(neg_idx[0]) < min_pts:
        return Leaf(y2)

    false_branch = build_tree(x2, y2)

    return DecisionNode(theta, true_branch, false_branch)


tree = build_tree(X_tr_arr, y_tr_arr)


def print_tree(node, indentation=""):
    '''printing function'''
    # base case means we have reached the leaf
    # if the node object is of leaf type
    if isinstance(node, Leaf):
        print(indentation + "PREDICTION", node.predictions)
        return

        # print the question at node
    print(indentation + str(node.theta))

    # call the function on true branch
    print(indentation + "True Branch")
    print_tree(node.true_branch, indentation + " ")

    # on flase branch
    print(indentation + "False Branch")
    print_tree(node.false_branch, indentation + " ")

print_tree(tree)