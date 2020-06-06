


db.getCollection('20ng_CG_RM').aggregate([
    {'$match': {'algorithm': 'cart'}}, 
    {'$group': {_id:null, 'stdalgo':{$stdDevPop: {'$arrayElemAt': ['$accuracy', 1]}}}},
        {'$group': {'_id': null, 'count': {'$sum': 1},
                        'train_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 0]}},
                        'test_acc': {'$avg': {'$arrayElemAt': ['$accuracy', 1]}},
                        'train_pre': {'$avg': {'$arrayElemAt': ['$precision', 0]}},
                        'test_pre': {'$avg': {'$arrayElemAt': ['$precision', 1]}},
                        'train_rec': {'$avg': {'$arrayElemAt': ['$recall', 0]}},
                        'test_rec': {'$avg': {'$arrayElemAt': ['$recall', 1]}},
                        'train_f1': {'$avg': {'$arrayElemAt': ['$f1', 0]}},
                        'test_f1': {'$avg': {'$arrayElemAt': ['$f1', 1]}},
                        
                 
                        }
             },

    
    ])


db.acc_vs_depth_quora.aggregate(
    {'$match': {'dataset': 'quora', 'd2v_vec_size': 10, 'algorithm': 'lr_mvdt'}},
                     {'$group': {'_id': 'on_depth', 'count':{$sum: 1},
                                 'train_acc': {'$avg': '$train_acc'},
                                 'test__acc': {'$avg': '$test_acc'}
                      }
    )

// accuracy vs depth based on d2v_vec_size and thier algorithms for trrain test both
//Double quotes quote object names (e.g. "field"). Single quotes are for strings 'string'
mb.runSQLQuery(`
       SELECT d2v_vec_size, algorithm, on_depth, AVG("train_acc"), 
       AVG("test_acc") FROM acc_vs_depth_quora
       GROUP BY "d2v_vec_size", "algorithm", "on_depth"
       ORDER BY "d2v_vec_size","algorithm", "on_depth" ASC
`)

// accuracy vs depth  imdb
mb.runSQLQuery(`
       SELECT acc_vs_depth_imdb.d2v_vec_size, acc_vs_depth_imdb.on_depth, acc_vs_depth_imdb.algorithm, AVG("train_acc"), 
       AVG("test_acc") FROM acc_vs_depth_imdb
       GROUP BY "d2v_vec_size", "algorithm", "on_depth"
       ORDER BY "d2v_vec_size","algorithm", "on_depth" ASC
`)



//on_fuction: min_max_by_algo, min_max_by_min_leaf_point, min_max_by_k_fold, min_max_by_fet_size, min_max_by_dim
db.ng_min_max_result.aggregate(
    //{$group: { _id: "$d2v_vec_size"}},
    )
    .match({'dataset':'20ng_CG_RM', 'on_data': 'testing', 'on_function':'min_max_by_dim' })
    .project({'_id':0, 'dataset':0, 'filename':0, 'epochs': 0, 'feature_size':0, 'd2v_shape':0, 'run':0, 'k_fold':0,
            'train_true_predict':0, 'test_true_predict':0, 'branch_sizes':0, 'tree_location':0, 'intermediate_result':0,
            'confusion_matrix':0, 'min_leaf_point':0
        })
    .sort({'d2v_vec_size':-1})
    
//acc_vs_depth.feature_size, acc_vs_depth.on_depth, AVG("train_acc"), AVG("test_acc") 
    
    
// min_man by on function report 
db.ng_min_max.aggregate([
    //{$match: {'dataset': 'imdb', 'on_data': 'testing', 'on_function':'min_max_by_dim'}},
    {$project:{
        '_id':0, 
        d2v_vec_size:1,
        algorithm: 1,
        min_max:1,
        feature_size:1,
        epochs:1,
        test_acc: {$multiply: [{$round: [{ $arrayElemAt: [ "$accuracy", -1 ]}, 4]}, 100},
        test_pre: {$multiply: [{$round: [{$arrayElemAt: ["$precision",-1]},4]}, 100},
        test_rec: {$multiply: [{$round: [{$arrayElemAt: ["$recall",-1]},4]}, 100}
        test_f1: {$multiply: [{$round: [{$arrayElemAt: ["$f1",-1]},4]}, 100}
        max_depth:1,
        inner_node:1,
        branch_sizes: 1,
        training_time:{$round: ["$training_time", 2]}
        on_function:1,
      }
   },
   {$sort: {"d2v_vec_size": 1, 'algorithm': 1}}
])
//.sort({"d2v_vec_size": -1, 'algorithm': 1})
//sort({"d2v_vec_size": 1, 'feature_size':1, 'min_max': 1})


// by avg overall report by datasets
db.by_datasets.aggregate([
    //{$match: {'dataset': '20ng_CG_RM'}},
    {$project:{
        '_id':0, 
        dataset: 1,
        vector_size:1,
        algorithm: 1,
        test_acc: {$round: [{ $arrayElemAt: [ "$accuracy", -1 ]}, 2},
        test_acc_std: {$round: [{$arrayElemAt: ["$std_acc",-1]}, 1},
        test_pre: {$round: [{ $arrayElemAt: [ "$precision", -1 ]}, 2},
        test_pre_std: {$round: [{$arrayElemAt: ["$std_pre",-1]}, 1},
        test_rec: {$round: [{ $arrayElemAt: [ "$recall", -1 ]}, 2},
        test_rec_std: {$round: [{$arrayElemAt: ["$std_rec",-1]}, 1},
        test_f1: {$round: [{$arrayElemAt: ["$f1",-1]}, 2},
        test_f1_std: {$round: [{$arrayElemAt: ["$std_f1",-1]}, 2},
        depth: {$round: [{$arrayElemAt: ["$depth",0]}, 2},
        std_depth: {$round: [{$arrayElemAt: ["$std_depth",-1]}, 1},
        count: 1,
      }
   }
]).sort({"vector_size": 1, 'algorithm':1})

// by avg hyperparams: epochs, feature_size: for lr and rs
//  min_leaf_point: for all
db.by_epochs.aggregate([
    {$match: {'dataset': 'quora'}},
    {$project:{
        '_id':0, 
        vector_size:1,
        algorithm: 1,
        by_epochs: 1,
        test_acc: {$round: [{ $arrayElemAt: [ "$accuracy", -1 ]}, 2},
        test_acc_std: {$round: [{$arrayElemAt: ["$std_acc",-1]}, 1},
        test_f1: {$round: [{$arrayElemAt: ["$f1",-1]}, 2},
        test_f1_std: {$round: [{$arrayElemAt: ["$std_f1",-1]}, 2},
        depth: {$round: [{$arrayElemAt: ["$depth",0]}, 2},
        std_depth: {$round: [{$arrayElemAt: ["$std_depth",-1]}, 1},
      }
   }
]).sort({"vector_size": 1, 'by_epochs':1})



//test
db.ocng.aggregate([
    {$match: {'dataset': '20ng_CG_RM', 'algorithm': 'oc1'}},
    {$group: {'_id': 'algorithm', 'count': {$sum: 1},
            'train_acc': {$avg: {$arrayElemAt: ['$accuracy', 0]}},
            'test_acc': {$avg: {$arrayElemAt: ['$accuracy', 1]}},
            'train_pre': {$avg: {$arrayElemAt: ['$precision', 0]}},
            'test_pre': {$avg: {$arrayElemAt: ['$precision', 1]}},
            'train_rec': {$avg: {$arrayElemAt: ['$recall', 0]}},
            'test_rec': {$avg: {$arrayElemAt: ['$recall', 1]}},
            'train_f1': {$avg: {$arrayElemAt: ['$f1', 0]}},
            'test_f1': {$avg: {$arrayElemAt: ['$f1', 1]}},
            'train_std_acc': {$stdDevPop: {$arrayElemAt: ['$accuracy', 0]}},
            'test_std_acc': {$stdDevPop: {$arrayElemAt: ['$accuracy', 1]}},
            'train_std_pre': {$stdDevPop: {$arrayElemAt: ['$precision', 0]}},
            'test_std_pre': {$stdDevPop: {$arrayElemAt: ['$precision', 1]}},
            'train_std_rec': {$stdDevPop: {$arrayElemAt: ['$recall', 0]}},
            'test_std_rec': {$stdDevPop: {$arrayElemAt: ['$recall', 1]}},
            'train_std_f1': {$stdDevPop: {$arrayElemAt: ['$f1', 0]}},
            'test_std_f1': {$stdDevPop: {$arrayElemAt: ['$f1', 1]}}
            }
        
    },
])



// train accuracy branch based tree extraction 
db.ng.aggregate([
    {'$match': {'algorithm': 'lr_mvdt', 'accuracy.0':{'$gt': 0.70, '$lte': 80}}}
    ]).limit(3)
    