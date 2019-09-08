import numpy as np

# best split
def best_split(rows, clf):
    #clf = LogisticRegression(solver='lbfgs')

    # rows[0]: features, rows[1]:label

    clf.fit(rows[0], rows[1])

    theta = np.append(clf.coef_, clf.intercept_)
    pred = clf.predict(rows[0])

    return clf, theta, pred


def partition(rows, pred):
    left_rows = np.where(pred > 0)
    right_rows = np.where(pred <= 0)

    # check node's purity
    # pos_idx = np.where(y[true_rows] == 1)
    # neg_idx = np.where(y[side] == 0)

    Xl = rows[0][left_rows]
    yl = rows[-1][left_rows]

    Xr = rows[0][right_rows]
    yr = rows[-1][right_rows]

    return [Xl, yl], [Xr, yr]



