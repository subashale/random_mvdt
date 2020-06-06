db.result.find({}).limit(10)

db.result.find({},{"dataset":1, "k_fold":1, "algorithm":1, "accuracy":1})
db.result.aggregate([{$group: {_id: "$run", count: {$sum: 1}}}, {$limit:10}])
db.by_n_feature.remove({})

//mongoimport --db thesis --collection result --type json --file sampleresult.json --jsonArray
//{$project:{"dataset":1, "k_fold":1, "algorithm":1, "accuracy":1}

// get average accuracy of train test on both dataset using different algorithms
// total average accuracy using $run
db.result.aggregate([
   {$group: {_id:"$dataset",  //$run
        count: {$sum: 1},
        algo: { $push: "$algorithm" },
        train_avg_acc: { $avg: {$arrayElemAt: [ "$accuracy", 0 ]}},
        test_avg_acc: { $avg: { $arrayElemAt: ["$accuracy", 1]}}
        }
   }
])

db.result.aggregate([
   {$group: 
        {_id:"$dataset", dataset:{ $push:"$algorithm"}}
   }
]);

// whole averate
db.result.aggregate(
        { $group : {
        _id : "$dataset",
        count: {$sum: 1},
        //f_key: {$push: "$algorithm"}
        }
    },
       {$group: {_id:"$algorithm",  //$run
        count: {$sum: 1},
        //algo: { $push: "$algorithm" },
        train_avg_acc: { $avg: {$arrayElemAt: [ "$accuracy", 0 ]}},
        test_avg_acc: { $avg: { $arrayElemAt: ["$accuracy", 1]}}
        }
   },

);



// select all data by dataset name
// by algo
// on kfold
// on dim
// on hpyerparameters
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
db.result.aggregate([
     {'$match': {
                'dataset': "20ng_CG_RM", 'k_fold': 1, 'd2v_vec_size': 10, 'algorithm': 'cart',
                'min_leaf_point': 5
            }}
    ])

db.result.aggregate([
    {'$match': {
                'dataset': "20ng_CG_RM", 'k_fold': 1, 'd2v_vec_size': 10, 'algorithm': 'cart',
                'min_leaf_point': 5, 'epochs': 50
            }},
    {$group: {
        _id: "$algorithm",
        total: {$sum: 1},
        //k_fold: {$push: "$k_fold"},
        
    }}
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

// db.by_min_leaf_point.drop()
// db.by_n_feature.drop()
db.by_algorithm.find({})


