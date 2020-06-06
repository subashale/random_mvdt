// countin rows
db.result_server.aggregate([
    {$match:{'dataset':'20ng_CG_RM', 'algorithm':'cart'}},
    {$count: "rows"}
    ])
// separating each dataset
// for 20ng_CG_RM
db.result_server.aggregate([
    {$match: {'dataset':'20ng_CG_RM'}},
    { $out: "20ng_CG_RM" } 
    ])
    
// for imdb
db.result_server.aggregate([
    {$match: {'dataset':'imdb'}},
    { $out: "imdb" } 
    ])

// storing from both cart and uni server downloaded data

db.getCollection('20ng_CG_RM').aggregate([
    {$match:{'dataset':'imdb'}},
    {$count: "rows"}
    ])
   
// because local cart dataset is uploaded later now 25*5 cart one dataset + rest 1250*5 lr_mvdt and rs_mvdt
db.getCollection('20ng_CG_RM').aggregate([
    {$match:{'dataset':'20ng_CG_RM', 'algorithm': 'cart'}},
    {$count: "rows"}
    ])
    
// now adding rest cart result of 20ng it will not update but remove existing data and import fresh
db.result_server.aggregate([
    {$match: {'dataset':'20ng_CG_RM', 'algorithm': 'cart'}},
    { $out: "20ng_CG_RM" } 
    ])
    
    


// find max training time and other from both train and test
db.min_max_result.aggregate([
    {$match:{'dataset':'20ng_CG_RM', 'algorithm': 'lr_mvdt'}},
    {$sort:{'accuracy.1':-1}}
    ]).limit(1)


db.result_server.aggregate([
    {$match:{'dataset':'imdb', 'algorithm': 'lr_mvdt'}},
    {$sort:{'training_time':-1}}
    ]).limit(1)



// get standard deviation 

db.getCollection('imdb').aggregate([
    {$match:{'algorithm':'rs_mvdt'}},
    {$group: {_id:null, stdalgo:{$stdDevPop: {'$arrayElemAt': ['$accuracy', 1]}}}}
    ])

db.by_k_folds.aggregate([
    //{$match:{'algorithm':'lr_mvdt'}},
    //{$sample: {size:100}},
    {$group: {_id:null, 
        'train_std':{$stdDevPop: {'$arrayElemAt': ['$accuracy', 0]}},
        'test_std':{$stdDevPop: {'$arrayElemAt': ['$accuracy', 1]}}}
        }
    ])


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
    
// db.by_datasets.drop()
// db.by_epochs.drop()
// db.by_k_folds.drop()
// db.by_vectors.drop()
// db.by_n_features.drop()
// db.by_min_leaf_points.drop()


db.acc_vs_depth.aggregate(
    {'$match': {'dataset': '20ng_CG_RM', 'd2v_vec_size': 10, 'algorithm': 'lr_mvdt'}},
                     {'$group': {'_id': 'on_depth', 'count':{$sum: 1},
                                 'train_acc': {'$avg': '$train_acc'},
                                 'test__acc': {'$avg': '$test_acc'}
                      }
    )

// accuracy vs depth based on d2v_vec_size and thier algorithms for trrain test both
//Double quotes quote object names (e.g. "field"). Single quotes are for strings 'string'
mb.runSQLQuery(`
       SELECT acc_vs_depth_ng.d2v_vec_size, acc_vs_depth_ng.algorithm, acc_vs_depth_ng.on_depth, AVG("train_acc"), 
       AVG("test_acc") FROM acc_vs_depth_ng
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


db.acc_vs_depth_ng.findOne({})


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
db.ng_min_max_result.aggregate([
    {$match: {'dataset': '20ng_CG_RM', 'on_data': 'testing', 'on_function':'min_max_by_dim'}},
    {$project:{
        '_id':0, 
        d2v_vec_size:1,
        algorithm: 1,
        min_max:1,
        feature_size:1,
        test_acc: {$round: [{ $arrayElemAt: [ "$accuracy", -1 ]}, 2]},
        test_pre: {$round: [{$arrayElemAt: ["$precision",-1]},2]},
        test_rec: {$round: [{$arrayElemAt: ["$recall",-1]},2]},
        test_f1: {$round: [{$arrayElemAt: ["$f1",-1]},2]},
        max_depth:1,
        inner_node:1,
        branch_sizes: 1,
        left_branch: { $arrayElemAt: [ "$branch_sizes", 0 ]},
        right_branch: { $arrayElemAt: [ "$branch_sizes", 1 ]},
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
db.by_n_features.aggregate([
    {$match: {'dataset': '20ng_CG_RM'}},
    {$project:{
        '_id':0, 
        dataset: 1,
        vector_size:1,
        algorithm: 1,
        test_acc: {$round: [{ $arrayElemAt: [ "$accuracy", -1 ]}, 2},
        test_acc_std: {$round: [{$arrayElemAt: ["$std_acc",-1]}, 1},
        test_f1: {$round: [{$arrayElemAt: ["$f1",-1]}, 2},
        test_f1_std: {$round: [{$arrayElemAt: ["$std_f1",-1]}, 2},
        depth: {$round: [{$arrayElemAt: ["$depth",0]}, 2},
        std_depth: {$round: [{$arrayElemAt: ["$std_depth",-1]}, 1},
        count: 1,
      }
   }
]).sort({"vector_size": 1})


    
db.getCollection('20ng_CG_RM').find()
  
db.ngimdb.aggregate([
    {$match: {'dataset':'imdb', 'k_fold':3}},
    { $out: "temp_imdb" } 
    ])
    
db.imdb.find({'k_fold':3})

db.temp_imdb.aggregate({$match: {'k_fold':3}})
// export imdb if not possibile dump in json or pkl


db.imdb.find({'k_fold':3})

db.quora_lr_rs.find({'d2v_vec_size':75})

db.by_datasets.find({})


db.min_max_result.find({'algorithm':'lr_mvdt'}).limit(10)


