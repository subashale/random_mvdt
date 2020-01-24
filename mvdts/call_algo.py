from mvdts.lr_mvdt.main import build_tree as lr_dt_fit
from mvdts.rs_mvdt.main import build_tree as rs_mvdt_fit
from mvdts.sklearn_cart.main import build_tree as card_fit

def fit(algorithm, train_data, epochs, min_leaf_point, n_features):
    # fitting on each algorithm
    if algorithm == "lr_mvdt":
        tree = lr_dt_fit(rows=train_data, epochs=epochs, min_point=min_leaf_point, noOfFeature=n_features)
    elif algorithm == "rs_mvdt":
        tree = rs_mvdt_fit(rows=train_data, epochs=epochs, min_point=min_leaf_point, noOfFeature=n_features)
    elif algorithm == "cart":
        tree = card_fit(rows=train_data, min_point=min_leaf_point)
    else:
        print("No algorithm is selected, out of algorithm error")
        return

    return tree
