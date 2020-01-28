import numpy as np
from mvdts.dt_common.nodes import Leaf
import time


# model print and get details, normal tree print
def print_tree(node, spacing=""):
    """
    Normal decision tree print
    :param node:
    :param spacing:
    :return:
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# print tree upto certain depth
def print_tree_depth(node, depth=1, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question), str(node.depth))

    if depth >= int(node.depth):
        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        print_tree_depth(node.true_branch, depth, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        print_tree_depth(node.false_branch, depth, spacing + "  ")


# print full model in tree form
def print_model(node, indentation="", pos='', root_node=[], leaf_node=[]):
    # if the node object is of leaf type
    if isinstance(node, Leaf):
        leaf_node.append(np.array([node.predictions]))
        print(indentation + "Predict", node.predictions)

    else:
        """    
        # if you want to get all nodes do append in each true and false part and also add root node
        # uncomment root node append from top of this function  ### point

        # root_node.append([node.true_purity, node.depth])
        # root_node.append([node.false_purity, node.depth])    
        """
        sum = np.add(np.array(node.true_purity), np.array(node.false_purity))
        # if node.depth == 1:
        #     root_node.append([sum, node.depth, pos])
        # else:
        #     root_node.append([sum, node.depth])
        root_node.append([sum, node.depth, pos])
        # showing depth, theta and entropy for each inner nodes
        print(indentation + "depth: " + str(node.depth)+" entropy: "+str(node.entropy))
        #print(indentation + "depth: " + str(node.depth)+str(node.question)+str(node.entropy))

        # call the function on true branch, l and r are indication for storing in branch
        print(indentation + "Left Branch " + str(node.true_purity))
        print_model(node.true_branch, indentation + "-->", 'l', root_node, leaf_node)

        # on false branch
        print(indentation + "Right Branch " + str(node.false_purity))
        print_model(node.false_branch, indentation + "-->", 'r', root_node, leaf_node)

        # all_nodes = [np.array(root_node), np.array(leaf_node)]
        # max_depth = find_depth(all_nodes)
    return np.array(root_node), find_depth(root_node), leaf_node

# print depth wise tree
def print_model_depth(node, depth, indentation="", pos='', root_node=[], leaf_node=[]):
    # if the node object is of leaf type
    if isinstance(node, Leaf):
        leaf_node.append(np.array([node.predictions]))
        print(indentation + "Predict", node.predictions)

    else:
        if depth >= int(node.depth):
            sum = np.add(np.array(node.true_purity), np.array(node.false_purity))

            root_node.append([sum, node.depth, pos])
            # showing depth, theta and entropy for each inner nodes
            print(indentation + "depth: " + str(node.depth)+" entropy: "+str(node.entropy))

            # call the function on true branch
            print(indentation + "Left Branch " + str(node.true_purity))
            print_model_depth(node.true_branch, depth, indentation + "-->", 'l', root_node, leaf_node)
            #print_model_depth(node.true_branch, depth, indentation + "-->", root_node, leaf_node)

            # on false branch
            print(indentation + "Right Branch " + str(node.false_purity))
            print_model_depth(node.false_branch, depth, indentation + "-->", 'r', root_node, leaf_node)
            #print_model_depth(node.false_branch, depth, indentation + "-->", root_node, leaf_node)
        return np.array(root_node), find_depth(root_node), leaf_node

# getting max depth
def find_depth(all_nodes):
    """
    finding depth of tree
    # leaf decision node count as final depth of decision node
    # we have two list 1 for root decision nodes
    # another for leaf decision node so use n_nodes[1] and check for max value
    :param n_nodes:
    :return:
    """
    depth = 0
    for i in range(len(all_nodes)):
        if np.amax(all_nodes[i][1]) > depth:
            depth = np.amax(all_nodes[i][1])
    return depth

# find probability for prediction
def predict_proba(prediction):
    total = sum({prediction[i] for i in prediction})
    result = {}
    for i, v in prediction.items():
        result[i] = v / total

    return result

# model prediction using all tree
def classify(x_point, model):
    # decision is leaf node, when it reach leaf decision theta
    # use dot product to find value based on used feature index
    # finding related feature index and adding 1 for bias

    if isinstance(model, Leaf):
        # pt = np.append(x_point[model.indexes], 1).reshape(len(x_point[model.indexes])+1, 1)
        # r = model.question.dot(pt)
        # print("feature indexes:{}, values:{}, r:{}".format(model.indexes, x_point[model.indexes], r))
        # return model.predictions

        if len(model.predictions) >= 2:
            # leaf node has more than two prediction then check high probability
            # in case of 50% choose last one
            pred_dict = predict_proba(model.predictions)
            max_prob = 0
            label_name = 0

            for i, v in pred_dict.items():
                if v >= max_prob:
                    max_prob = v # in case of of 50/50 choose last one
                    label_name = i
            return int(label_name)
        else:
            print(model.predictions)
            # int for only integer classes
            # next(iter( dfdds.items() ))[0] if you have multiple
            return next(iter(model.predictions))
    else:
        pt = np.append(x_point[model.indexes], 1)
        r = model.question.T.dot(pt)
        # pt = np.append(x_point[model.indexes], 1).reshape(len(x_point[model.indexes])+1, 1)
        # r = model.question.dot(pt)

        # recursive approach to find out leaf decision
        # look for branch only if it is not leaf node it self
        if r > 0:
            return classify(x_point, model.true_branch)
        else:
            return classify(x_point, model.false_branch)

# model prediction on depth level
def classify_depth(x_point, model, depth=1):
    if isinstance(model, Leaf):

        if len(model.predictions) >= 2:
            pred_dict = predict_proba(model.predictions)
            max_prob = 0
            label_name = 0

            for i, v in pred_dict.items():
                if v >= max_prob:
                    max_prob = v  # in case of of 50/50 choose last one
                    label_name = i
            return int(label_name)
        else:
            # print(model.predictions)
            return next(iter(model.predictions))
    else:
        pt = np.append(x_point[model.indexes], 1)
        r = model.question.T.dot(pt)
        if depth > int(model.depth):
            if r > 0:
                return classify_depth(x_point, model.true_branch, depth)
            else:
                return classify_depth(x_point, model.false_branch, depth)
        #print("from inner node: {}".format(r))

        if r > 0:
            r = 1
        else:
            r = 0

        return r

# predict call request
def predict(x, tree):
    # for storing predicted values
    yh_list = []
    # if only one value is passed then direct compute else do in loop
    if len(x.shape) == 1:
        return classify(x, tree)
    else:
        for i, value in enumerate(x):
            yh = classify(value, tree)
            yh_list.append(yh)
        return yh_list

# predict call request on depth level
def predict_depth(x, tree, depth):
    # for storing predicted values
    yh_list = []
    # if only one value is passed then direct compute else do in loop
    if len(x.shape) == 1:
        return classify_depth(x, tree, depth)
    else:
        for i, value in enumerate(x):
            yh = classify_depth(value, tree, depth)
            yh_list.append(yh)
        return yh_list

# checking runtime
def time_it():
    return time.time()

