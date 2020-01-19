from mvdts.lr_mvdt.main import build_tree as lr_dt_fit
from mvdts.rs_mvdt.main import build_tree as rs_mvdt_fit
from mvdts.sklearn_cart.main import build_tree as card_fit

def fit(algorithm, train_data, epochs, depth, n_features):
    # fitting on each algorithm
    if algorithm == "lr_mvdt":
        tree = lr_dt_fit(train_data, epochs, max_depth=depth, noOfFeature=n_features)
    elif algorithm == "rs_mvdt":
        tree = rs_mvdt_fit(train_data, epochs, max_depth=depth, noOfFeature=n_features)
    elif algorithm == "cart":
        tree = card_fit(train_data,  max_depth=depth)
    else:
        print("No algorithm is selected, out of algorithm error")
        return

    return tree
