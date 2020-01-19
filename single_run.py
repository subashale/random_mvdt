# seprate test
from mvdts.call_algo import fit
import pandas as pd
import numpy as np


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


data_location = "data/20ng_CG_RM/k1/final/"
train_data = "10D_20ng_CG_RM_k1_train.csv"
test_data = "10D_20ng_CG_RM_k1_test.csv"

train = read_data(data_location+train_data)
test = read_data(data_location+test_data)

# settings
algorithm = "cart"
epochs = 100
depth = 3
n_features = 5


def get_cart_model_info(model):
    inner_node = model.tree_.node_count

    # getting leaf nodes https://notebooks.gesis.org/binder/jupyter/user/scikit-learn-scikit-learn-pw3uja99/lab
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    is_leaves = np.zeros(shape=inner_node, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    leaf_node = len(np.where(is_leaves == True)[0])
    all_nodes = inner_node + leaf_node
    max_depth = model.tree_.max_depth
    return inner_node, leaf_node, all_nodes, max_depth


for i in range(10):
    #fit(algorithm, train, epochs, depth, n_features)
    #print(print_tree(fit(algorithm, train, epochs, depth, n_features)))
    #tree = fit(algorithm, train, epochs, depth, n_features)
    #print(print_model(tree,  root_node = [], leaf_node = []))
    #for tree
    tree = fit(algorithm, train, epochs, depth, n_features)
    # y_pred_test = predict(test[0], tree)
    # y_pred_train = predict(train[0], tree)

    # print("accuracy {}, {}".format(accuracy_score(train[1], y_pred_train), accuracy_score(test[1], y_pred_test)))
    # print("\ntrain confusino matrix", len(train[1]), len(y_pred_train))
    # print(confusion_matrix(train[1], y_pred_train))
    # print("\ntest confusino matrix", len(test[1]), len(y_pred_test))
    # print(confusion_matrix(test[1], y_pred_test))

    print(get_cart_model_info(tree))

