import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report, accuracy_score

# calling helper funtions
import data_prepare as dp
import data_viz as dviz
import data_split as ds
import prediction as pred


df = pd.read_csv("data/circle.csv")
X = np.array(df[df.columns[:-1]].values.tolist(), dtype=np.float64)
label = np.array(df[df.columns[-1]].values.tolist())
index = 0
to_split = [X, label, index]
tree_depth = 4
epochs = 1000
min_pts = 5
tree = np.zeros((2**tree_depth, 5))
next_free = 0

# to_split = df

for depth in range(tree_depth):
    tmp_split = []

    # shfuffling for each node
    # to_split = shuffle(to_split)

    if len(to_split) != 0:

        X2 = to_split[0]
        label2 = to_split[1]
        # X2, label2, pos, neg = dp.data_giver(to_split)
        pos, neg = dp.pos_neg_giver(to_split)

        # best_gain; highest the information gain will be good
        best_gain = 0
        # best_theta = coefficient or line
        best_theta = []
        # best_pos_side, best_neg_side = data points for pos and negative after the best_gain split
        best_pos_side = []
        best_neg_side = []

        for epoch in range(epochs):
            # find random feature points as linear combination
            p = random.choice(pos)
            n = random.choice(neg)

            # getting parameters from split
            gain, pos_side, neg_side, theta = ds.split_data(X2, label2, pos, neg, p, n)
            # print(gain)

            # checking for best gain and setting rest parameters of it
            if gain > best_gain:
                best_gain = gain
                best_pos_side = pos_side
                best_neg_side = neg_side
                best_theta = theta

                # dviz.viz_data_with_line(best_theta, to_split)
            # dviz.viz_data_with_line(best_theta, to_split)

        # building tree
        if depth != tree_depth:
            # check left node's purity
            pos_idx = np.where(label2[[best_pos_side]] == 1)
            neg_idx = np.where(label2[[best_pos_side]] == 0)

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print("pure node found in left\n")
            else:
                if len(pos_idx[0]) > min_pts and len(neg_idx[0]) > min_pts:
                    # making sub dataframe for left
                    left_x = X2[best_pos_side]
                    left_label = label2[best_pos_side]
                    next_free += 1

                    tree[to_split[2]][3] = next_free
                    tmp_split = [left_x, left_label, next_free]
                    # tmp_split = pd_concat(left_x, left_label)

            # check right node's purity
            pos_idx = np.where(label2[[best_neg_side]] == 1)
            neg_idx = np.where(label2[[best_neg_side]] == 0)

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print("pure node found in right\n")
            else:
                if len(pos_idx[0]) > min_pts and len(neg_idx[0]) > min_pts:
                    # making sub dataframe for left
                    right_x = X2[best_neg_side]
                    right_label = label2[best_neg_side]
                    next_free += 1

                    tree[to_split[2]][4] = next_free

                    tmp_split = [right_x, right_label, next_free]
                    # tmp_split = pd_concat(right_x, right_label)

            tree[to_split[2]][:3] = best_theta.T

        # ploting graph

        dviz.viz_data_with_line_np(best_theta, to_split)
        # dviz.scatter_plot_line(best_theta, to_split)

    # dviz.viz_data_with_line(best_theta, to_split)
    to_split = tmp_split

print(pred.predict(tree, X[4]))
print(pred.predict(tree, X[5]))

y_pred = pred.predict(tree, X)
print(classification_report(label, y_pred, labels=[0, 1]))
accuracy_score(label, y_pred)