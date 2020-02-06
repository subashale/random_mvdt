import os
import ujson as json
import pymongo

# mongosetting
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.thesis


# settings
datasets = "20ng_CG_RM,20newsgroup"#"imdb"
algorithms_list = ['lr_mvdt', 'rs_mvdt', 'cart']
k = 2
vector_sizes_list = [10, 25]
epochs_list = [50, 100]
n_features_list = [2, all]
min_leaf_point_list = [5, 10]

by_list = [['by_n_features', '$feature_size'],
           ['by_min_leaf_points', '$min_leaf_point'],
           ['by_epochs', '$epochs'],
           ['by_vectors', '$d2v_vec_size'],
           ['by_k_folds', '$k_fold'],
           ['by_datasets', '$algorithm']]


def avg_by_all():
    for name in datasets.split(","):
        for algorithm in algorithms_list:
            for by_param in by_list:
                get_avg_by(name, algorithm, by_param)
                print(name, algorithm, by_param)


def get_avg_by(*argv):
    # getting training and testing average for each by_param list
    query = [
        {'$match': {'dataset': argv[0], 'algorithm': argv[1]}
         }, {'$group': {'_id': argv[-1][1], 'count': {'$sum': 1},
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
                      argv[-1][0]: i['_id'],  # no of epochs
                      'count': i['count'],  # total rows taken
                      'accuracy': [i['train_acc'], i['test_acc']],
                      'precision': [i['train_pre'], i['test_pre']],  # precision of both train and test
                      'recall': [i['train_rec'], i['test_rec']],  # recall of both train and test
                      'f1': [i['train_f1'], i['test_f1']]
                      }

        # inserting in mongodb
        collection_name = db[str(argv[-1][0])]
        collection_name.insert_one(new_result)
        # saving in to json
        save_results(new_result, argv[-1][0])


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

# get min and max accuracy result on train and test set
#accuracy.0: train, accuracy.1: test, -1:max, +1:min


def min_max_by_dataset():
    # get min max a by_param and store that feature as well, choosing by_param gives same result since all setting are
    # document information
    # get best, and worst evaluation setting result for each dataset
    for dataset in datasets.split(","):
        for i in range(2):
            if i == 0:
                # for storing train and test information
                on_data = 'training'
            else:
                on_data = 'testing'

            for j in range(2): #-1: max, +1: min
                if j == 0:
                    print("dataset:{} on_data {}, result of max".format(dataset, on_data))
                    min_max = -1
                else:
                    print("dataset:{} on_data {}, result of min".format(dataset, on_data))
                    min_max = +1
                query = [{'$match': {'dataset': dataset}},
                        {'$sort': {'accuracy.'+str(i):min_max}},
                        {'$limit': 1}]
                d = db.result.aggregate(query)

                for result in d:
                    # removing default id and adding extra two information of on_data, min_max
                    del result['_id']
                    result['on_function'] = 'min_max_by_dataset'
                    result['on_data'] = on_data
                    result['min_max'] = min_max

                    #inserting in mongodb
                    collection_name = db['min_max_result']
                    collection_name.insert_one(result)

                    # saving in to json
                    save_results(result, 'min_max_result')


def min_max_by_algo():
    # get best and worst result on each algorithm for all dataset
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for i in range(2):
                if i == 0:
                    on_data = 'training'
                else:
                    on_data = 'testing'
                for j in range(2):
                    if j == 0:
                        print("dataset:{}, algorithm: {}, on data: {} result of max".format(dataset, algorithm, on_data))
                        min_max = -1
                    else:
                        print("dataset:{}, algorithm: {} on data: {} result of min".format(dataset, algorithm, on_data))
                        min_max = +1
                    query = [
                        {'$match': {'dataset': dataset, 'algorithm': algorithm}},
                        {'$sort': {'accuracy.'+str(i):min_max}},
                        {'$limit': 1}]
                    d = db.result.aggregate(query)
                    for result in d:
                        # removing default id and adding extra two information of on_data, min_max
                        del result['_id']
                        result['on_function'] = 'min_max_by_algo'
                        result['on_data'] = on_data
                        result['min_max'] = min_max

                        # inserting in mongodb
                        collection_name = db['min_max_result']
                        collection_name.insert_one(result)

                        # saving in to json
                        save_results(result, 'min_max_result')


def min_max_by_k_fold():
    # get best result on each algorithm for all datasets for each k_fold
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for n in range(1, k+1):
                for i in range(2):
                    if i == 0:
                        on_data = 'training'
                    else:
                        on_data = 'testing'
                    for j in range(2):
                        if j == 0:
                            print("dataset:{}, algorithm: {}, k_fold:{} on data: {} result of max".format(dataset,
                                                                                            algorithm, n, on_data))
                            min_max = -1
                        else:
                            print("dataset:{}, algorithm: {}, k_fold:{}  on data: {} result of min".format(dataset,
                                                                                            algorithm, n, on_data))
                            min_max = +1
                        query = [
                            {'$match': {'dataset': dataset, 'algorithm': algorithm, 'k_fold': n}},
                            {'$sort': {'accuracy.'+str(i):min_max}},
                            {'$limit': 1}]
                        d = db.result.aggregate(query)
                        for result in d:
                            # removing default id and adding extra two information of on_data, min_max
                            del result['_id']
                            result['on_function'] = 'min_max_by_k_fold'
                            result['on_data'] = on_data
                            result['min_max'] = min_max

                            # inserting in mongodb
                            collection_name = db['min_max_result']
                            collection_name.insert_one(result)

                            # saving in to json
                            save_results(result, 'min_max_result')


def min_max_by_dim():
    # get best result on each algorithm for all datasets for each d2v vector size
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for dim in vector_sizes_list:
                for i in range(2):
                    if i == 0:
                        on_data = 'training'
                    else:
                        on_data = 'testing'
                    for j in range(2):
                        if j == 0:
                            print("dataset:{}, algorithm: {}, dimension:{} on data: {} result of max".format(dataset,
                                                                                            algorithm, dim, on_data))
                            min_max = -1
                        else:
                            print("dataset:{}, algorithm: {}, dimension:{}  on data: {} result of min".format(dataset,
                                                                                            algorithm, dim, on_data))
                            min_max = +1
                        query = [
                            {'$match': {'dataset': dataset, 'algorithm': algorithm, 'd2v_vec_size': dim}},
                            {'$sort': {'accuracy.'+str(i):min_max}},
                            {'$limit': 1}]
                        d = db.result.aggregate(query)
                        for result in d:
                            # removing default id and adding extra two information of on_data, min_max
                            del result['_id']
                            result['on_function'] = 'min_max_by_dim'
                            result['on_data'] = on_data
                            result['min_max'] = min_max

                            # inserting in mongodb
                            collection_name = db['min_max_result']
                            collection_name.insert_one(result)

                            # saving in to json
                            save_results(result, 'min_max_result')


def min_max_by_epochs():
    # get best result on each algorithm for all datasets for each epochs, cart.epochs = 0
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for epochs in epochs_list:
                for i in range(2):
                    if i == 0:
                        on_data = 'training'
                    else:
                        on_data = 'testing'
                    for j in range(2):
                        if j == 0:
                            print("dataset:{}, algorithm: {}, epochs:{} on data: {} result of max".format(dataset,
                                                                                            algorithm, epochs, on_data))
                            min_max = -1
                        else:
                            print("dataset:{}, algorithm: {}, epochs:{}  on data: {} result of min".format(dataset,
                                                                                            algorithm, epochs, on_data))
                            min_max = +1
                        query = [
                            {'$match': {'dataset': dataset, 'algorithm': algorithm, 'epochs': epochs}},
                            {'$sort': {'accuracy.'+str(i):min_max}},
                            {'$limit': 1}]
                        d = db.result.aggregate(query)
                        for result in d:
                            del result['_id']
                            result['on_function'] = 'min_max_by_epochs'
                            result['on_data'] = on_data
                            result['min_max'] = min_max

                            # inserting in mongodb
                            collection_name = db['min_max_result']
                            collection_name.insert_one(result)

                            # saving in to json
                            save_results(result, 'min_max_result')


def min_max_by_fet_size():
    # get best result on each algorithm for all datasets for each feature size
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for n_feature in n_features_list:
                if n_feature == all:
                    # get max feature size from database but must get max feature size on each dataset so
                    # use vect_size since it is used to create features and its different
                    # for vect_size on each datasets max feature is different so apply all vect_size
                    for dim in vector_sizes_list:
                        query = [{'$match': {'dataset': dataset, 'algorithm': algorithm, 'd2v_vec_size':dim}},
                                 {'$group': {'_id': '$feature_size', 'all': {'$max': "$feature_size"}}},
                                 {'$limit': 1}]
                        d = db.result.aggregate(query)
                        for i in d:
                            n_feature = i['all']

                        for i in range(2):
                            if i == 0:
                                on_data = 'training'
                            else:
                                on_data = 'testing'
                            for j in range(2):
                                if j == 0:
                                    print("dataset:{}, algorithm: {}, n_feature: {}, on data: {} result of max ".
                                          format(dataset, algorithm, n_feature, on_data))
                                    min_max = -1
                                else:
                                    print("dataset:{}, algorithm: {}, n_feature: {},  on data: {} result of min".
                                          format(dataset, algorithm, n_feature, on_data))
                                    min_max = +1
                                query = [
                                    {'$match': {'dataset': dataset, 'algorithm': algorithm, 'feature_size': n_feature}},
                                    {'$sort': {'accuracy.'+str(i):min_max}},
                                    {'$limit': 1}]
                                d = db.result.aggregate(query)
                                for result in d:
                                    del result['_id']
                                    result['on_function'] = 'min_max_by_fet_size'
                                    result['on_data'] = on_data
                                    result['min_max'] = min_max

                                    # inserting in mongodb
                                    collection_name = db['min_max_result']
                                    collection_name.insert_one(result)

                                    # saving in to json
                                    save_results(result, 'min_max_result')
                else:
                    for i in range(2):
                        if i == 0:
                            on_data = 'training'
                        else:
                            on_data = 'testing'
                        for j in range(2):
                            if j == 0:
                                print("dataset:{}, algorithm: {}, n_feature: {}, on data: {} result of max".
                                      format(dataset, algorithm, n_feature, on_data))
                                min_max = -1
                            else:
                                print("dataset:{}, algorithm: {}, n_feature: {}, on data: {} result of min".
                                      format(dataset, algorithm, n_feature, on_data))
                                min_max = +1
                            query = [
                                {'$match': {'dataset': dataset, 'algorithm': algorithm, 'feature_size': n_feature}},
                                {'$sort': {'accuracy.'+str(i):min_max}},
                                {'$limit': 1}]
                            d = db.result.aggregate(query)
                            for result in d:
                                del result['_id']
                                result['on_function'] = 'min_max_by_fet_size'
                                result['on_data'] = on_data
                                result['min_max'] = min_max

                                # inserting in mongodb
                                collection_name = db['min_max_result']
                                collection_name.insert_one(result)

                                # saving in to json
                                save_results(result, 'min_max_result')


def min_max_by_min_leaf_point():
    # get best result on each algorithm for all datasets for each min_leaf_point
    for dataset in datasets.split(","):
        for algorithm in algorithms_list:
            for min_leaf_point in min_leaf_point_list:
                for i in range(2):
                    if i == 0:
                        on_data = 'training'
                    else:
                        on_data = 'testing'
                    for j in range(2):
                        if j == 0:
                            print("dataset:{}, algorithm: {}, min_leaf_point:{}, on data:{} result of max".
                                  format(dataset, algorithm, min_leaf_point, on_data))
                            min_max = -1
                        else:
                            print("dataset:{}, algorithm: {}, min_leaf_point:{}, on data: {} result of min".
                                  format(dataset, algorithm, min_leaf_point, on_data))
                            min_max = +1
                        query = [
                            {'$match': {'dataset': dataset, 'algorithm': algorithm, 'min_leaf_point': min_leaf_point}},
                            {'$sort': {'accuracy.'+str(i):min_max}},
                            {'$limit': 1}]
                        d = db.result.aggregate(query)
                        for result in d:
                            del result['_id']
                            result['on_function'] = 'min_max_by_min_leaf_point'
                            result['on_data'] = on_data
                            result['min_max'] = min_max

                            # inserting in mongodb
                            collection_name = db['min_max_result']
                            collection_name.insert_one(result)

                            # saving in to json
                            save_results(result, 'min_max_result')


if __name__ == '__main__':
    # output datasets *algorithm*by_pram
    #avg_by_all()
    min_max_by_dataset()
    min_max_by_algo()
    min_max_by_k_fold()
    min_max_by_dim()
    min_max_by_epochs()
    min_max_by_fet_size()
    min_max_by_min_leaf_point()

