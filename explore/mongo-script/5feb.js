db.result.aggregate([
    {'$match': {'dataset': '20ng_CG_RM', 'k_fold': 1, 'd2v_vec_size': 10, 'algorithm': 'lr_mvdt', 'epochs': 50, 'feature_size': 2,
                'min_leaf_point': 5}},
    {$group: {'_id': '$min_leaf_point', 'count': {'$sum': 1},
                'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                'test__acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}}
        
            }
        
    }])
    
db.result.aggregate([{$group:{_id:"$k_fold"}}])
db.result.aggregate([
    {$match: {"dataset": "20ng_CG_RM", "k_fold": 1, "d2v_vec_size":10,
    "algorithm":"lr_mvdt", "epochs":50, "min_leaf_point":5, "feature_size":2}},
    {$group: {_id:"$dataset", 
    train_avg_acc: { $avg: {$arrayElemAt: [ "$accuracy", 0 ]}},
    test_avg_acc: { $avg: { $arrayElemAt: ["$accuracy", 1]}},
    k_fold: {$first: "$k_fold"},
    d2v_vec_size: {$first:"$d2v_vec_size"},
    epochs: {$first: "$epochs"},
    min_leaf_point: {$first:"$min_leaf_point"},
    feature_size: {$first: "$feature_size"},
    algorithm: {$push:"$algorithm"},
    accuracy: {$push: "$accuracy"}
    }
    },
    //{$project: {"k_fold":1, "train_avg_acc":1, "test_avg_acc":1}}
    ])
db.result.find({}).limit(1)
db.result.aggregate([
    {$match:{
        "$or":[ 
        { "algorithm": "lr_mvdt" },
        { "algorithm": "rs_mvdt" }],  "dataset": "20ng_CG_RM" }
    },
    {'$group': {'_id': '$min_leaf_point', 'count': {'$sum': 1},
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
    ])

db.result.find({'algorithm':'cart'})

db.result.aggregate([
     {'$group': {'_id': '$feature_size', 'count': {'$sum': 1},
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
])


db.by_dataset.find({})


db.result.aggregate([
    {'$match': {'dataset': '20ng_CG_RM', 'algorithm':'lr_mvdt'}
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
])

db.result.aggregate([
    {'$group': {'_id': '$dataset', 'count': {'$sum': 1}}}
    ])
    
db.by_datasets.aggregate({$match:{'a':{'$arrayElemAt': ['$accuracy', 1]}}})

//-1:max, +1:min
db.result.find().sort({accuracy:-1}).limit(1)

db.result.aggregate([
    {$sort: {accuracy:-1}},
    {$limit: 1}
    ])

// accuracy.0: train, accuracy.1: test, -1:max, +1:min
db.result.aggregate([
    {$match:{
        "$or":[ 
        { "dataset": "20ng_CG_RM" },
        { "dataset": "20newsgroup" }]}},
    {$sort: {
        'accuracy.0':-1}
    },
    {$limit: 3}
    ])



db.result.aggregate(
   [ {$match:
       {'dataset': '20ng_CG_RM', 'algorithm':'lr_mvdt'}
   },
     {
       $group:
         {
           _id: '$feature_size',
           all: { $max: "$feature_size" }
         }
     },
     {'$sort': {'feature_size.0':+1}},
     {'$limit': 1}
   ]
)


// db.by_epochs.drop()
// db.by_datasets.drop()
// db.by_k_folds.drop()
// db.by_min_leaf_points.drop()
// db.by_n_features.drop()
// db.by_vectors.drop()







