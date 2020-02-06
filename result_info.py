import os
import numpy as np
import pandas as pd
import ujson as json
import pymongo

# settings
datasets = "20ng_CG_RM,20newsgroup"#"imdb"
k = 2
vector_sizes = [10, 25]
algorithms_list = ['lr_mvdt', 'rs_mvdt', 'cart']
epochs_list = [50, 100]
n_features_list = [2, all]
min_leaf_point_list = [5, 10]

# mongosetting
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.thesis


def read_data(dataLocation):
    df = pd.read_csv(dataLocation)
    X = np.array(df[df.columns[:-1]].values.tolist(), dtype=np.float64)
    y = np.array(df[df.columns[-1]].values.tolist())
    return [X, y]


def avg_by_n_features_lrrs():
    for name in datasets.split(","):
        for n in range(1, k+1):
            count = str(n)
            for vector_size in vector_sizes:
                for algorithm in algorithms_list:
                    for epochs in epochs_list:
                        for min_leaf_point in min_leaf_point_list:
                            for n_features in n_features_list:
                                if n_features == all:
                                    data_loc = "data/"+name+"/k"+count+"/final/"
                                    data_train = str(vector_size) + "D_" + name + "_k" + count + "_train.csv"

                                    train = read_data(data_loc + data_train)
                                    n_features = len(train[0][0])
                                #print(name, n, vector_size, algorithm, epochs, min_leaf_point,n_features)
                                get_avg_by(name, n, vector_size, algorithm, epochs, min_leaf_point, n_features,
                                           "by_n_feature")


def avg_by_minLeafPoint():
    for name in datasets.split(","):
        for n in range(1, k+1):
            for vector_size in vector_sizes:
                for algorithm in algorithms_list:
                    for epochs in epochs_list:
                        for min_leaf_point in min_leaf_point_list:
                            get_avg_by(name, n, vector_size, algorithm, epochs, min_leaf_point, "by_min_leaf_point")


def avg_by_epochs():
    for name in datasets.split(","):
        for n in range(1, k+1):
            for vector_size in vector_sizes:
                for algorithm in algorithms_list:
                    get_avg_by(name, n, vector_size, algorithm)
                    for epochs in epochs_list:
                        get_avg_by(name, n, vector_size, algorithm, epochs, "by_epochs")


def avg_by_algorithm():
    for name in datasets.split(","):
        for n in range(1, k + 1):
            for vector_size in vector_sizes:
                for algorithm in algorithms_list:
                    get_avg_by(name, n, vector_size, algorithm, "by_algorithm")


def avg_by_vector_size():
    for name in datasets.split(","):
        for vector_size in vector_sizes:
            for algorithm in algorithms_list:
                get_avg_by(name, vector_size, algorithm, "by_vector")


def avg_by_k_fold():
    for name in datasets.split(","):
        for n in range(1, k + 1):
            for algorithm in algorithms_list:
                get_avg_by(name, n, algorithm, "by_k_fold")


def avg_by_dataset():
    for name in datasets.split(","):
        for algorithm in algorithms_list:
            print(name, algorithm)
            get_avg_by(name, algorithm, "by_dataset")


# argument order
#name, n, vector_size, algorithm, epochs, min_leaf_point,n_features
def get_avg_by(*argv):
    if argv[-1] == "by_n_feature":
        # average by n_features in 10 runs not use for cart
        query = [
                {'$match': {"$or": [
                    {"algorithm": "lr_mvdt"},
                    {"algorithm": "rs_mvdt"}],
                    'dataset': argv[0], 'k_fold': argv[1], 'd2v_vec_size':argv[2], 'algorithm':argv[3],
                    'epochs':argv[4],  'min_leaf_point':argv[5], 'feature_size':argv[6]
                    }
                }, {'$group': {'_id': '$feature_size', 'count': {'$sum': 1},
                    'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                    'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                    'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                    'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                    'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                    'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                    'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                    'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                    }
                }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'k_fold': int(argv[1]),  # which K_fold number
                          'd2v_vec_size': int(argv[2]),  # dod2vec feature dimension
                          'algorithm': argv[3],  # algorithm name
                          'epochs': argv[4],  # no of epochs
                          'min_leaf_point': argv[5],  # given depth of tree
                          'feature_size': argv[6],  # how may features are taken
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                         }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
            return
    elif argv[-1] == "by_min_leaf_point":
        # average by min_leaf_point, for cart and rest with kfold, d2vec, min_leaf_point
        query = [
            {'$match': {
                'dataset': argv[0], 'k_fold': argv[1], 'd2v_vec_size': argv[2], 'algorithm': argv[3],
                'min_leaf_point': argv[5]
            }
            }, {'$group': {'_id': '$min_leaf_point', 'count': {'$sum': 1},
                           'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                           'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                           'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                           'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                           'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                           'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                           'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                           'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                           }
                }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'k_fold': int(argv[1]),  # which K_fold number
                          'd2v_vec_size': int(argv[2]),  # dod2vec feature dimension
                          'algorithm': argv[3],  # algorithm name
                          'epochs': argv[4],  # no of epochs
                          'min_leaf_point': argv[5],  # given depth of tree
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            print(new_result)
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    elif argv[-1] == "by_epochs":
        # average by epochs with dataset, kfold, d2vec, algorithm
        print(argv[-1])
        query = [
            {'$match': {"$or": [
                {"algorithm": "lr_mvdt"},
                {"algorithm": "rs_mvdt"}],
                'dataset': argv[0], 'k_fold': argv[1], 'd2v_vec_size': argv[2], 'algorithm': argv[3],
                'epochs': argv[4]
            }
            }, {'$group': {'_id': '$epochs', 'count': {'$sum': 1},
                           'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                           'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                           'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                           'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                           'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                           'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                           'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                           'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                           }
                }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'k_fold': int(argv[1]),  # which K_fold number
                          'd2v_vec_size': int(argv[2]),  # dod2vec feature dimension
                          'algorithm': argv[3],  # algorithm name
                          'epochs': argv[4],  # no of epochs
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    elif argv[-1] == "by_algorithm":
        # average by algorithm in kfold, d2vec
        print(argv[-1])
        query = [
            {'$match': {'dataset': argv[0], 'k_fold': argv[1], 'd2v_vec_size': argv[2], 'algorithm': argv[3]}
            }, {'$group': {'_id': '$algorithm', 'count': {'$sum': 1},
                           'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                           'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                           'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                           'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                           'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                           'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                           'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                           'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                           }
                }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'k_fold': int(argv[1]),  # which K_fold number
                          'd2v_vec_size': int(argv[2]),  # dod2vec feature dimension
                          'algorithm': argv[3],  # algorithm name
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    elif argv[-1] == "by_vector":
        # average by vector size with dataset
        query = [
            {'$match': {'dataset': argv[0], 'd2v_vec_size': argv[1]}
             }, {'$group': {'_id': '$d2v_vec_size', 'count': {'$sum': 1},
                            'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                            'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                            'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                            'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                            'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                            'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                            'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                            'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                            }
                 }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'd2v_vec_size': int(argv[1]),  # dod2vec feature dimension
                          'algorithm': argv[2],  # algorithm name
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    elif argv[-1] == "by_k_fold":
        # average on kfold with dataset
        query = [
            {'$match': {'dataset': argv[0], 'k_fold': argv[1]}
             }, {'$group': {'_id': '$d2v_vec_size', 'count': {'$sum': 1},
                            'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                            'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                            'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                            'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                            'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                            'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                            'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                            'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                            }
                 }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'k_fold': int(argv[1]),  # which K_fold number
                          'algorithm': argv[2],  # algorithm name
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    elif argv[-1] == "by_dataset":
        # average on dataset by algorithm
        print("by dataset")
        query = [
            {'$match': {'dataset': argv[0], 'algorithm':argv[1]}
             }, {'$group': {'_id': '$algorithm', 'count': {'$sum': 1},
                            'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                            'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                            'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                            'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                            'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                            'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                            'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                            'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                            }
                 }
        ]
        d = db.result.aggregate(query)
        for i in d:
            new_result = {'dataset': argv[0],  # name of dataset
                          'algorithm': argv[1],  # algorithm name
                          'count': i['count'],  # total rows taken
                          'accuracy': [i['train_acc'], i['test_acc']],
                          'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                          'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                          'f1': [i['train_f1'], i['test_f1']]
                          }
            # inserting in mongodb
            collection_name = db[str(argv[-1])]
            collection_name.insert_one(new_result)
            # saving in to json
            save_results(new_result, argv[-1])
    else:
        print(argv[-1])
        print(len(argv), argv)


def save_results(new_data, isfrom):
    # result file is not exits
    print(new_data)
    result_file = isfrom+".json"
    # removing _id auto gen from mongo
    if '_id' in new_data:
        del new_data['_id']
    if not os.path.exists(result_file):
        temp = {"results": []}
        with open(result_file, 'w+') as json_result:
            json.dump(temp, json_result)
    with open(result_file) as json_file:
        data = json.load(json_file)
        temp = data["results"]

        # python object to be appended
        temp.append(new_data)

    # then update data
    with open(result_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    # avg_by_n_features_lrrs()
    # avg_by_minLeafPoint()
    # avg_by_epochs()
    # avg_by_algorithm()
    # avg_by_vector_size()
    # avg_by_k_fold()
    avg_by_dataset()

