#import logging
import gensim
import pandas as pd
import numpy as np
import multiprocessing
cores = multiprocessing.cpu_count()
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_corpus(data, tokens_only=False):
    "tagging and tokenizing document"
    for i, line in enumerate(data):
        #takes string to tokinize data using lowercase and stopwords methods
        tokens = gensim.utils.simple_preprocess(str(line))
        if tokens_only:
            # return only token words, use for test data
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def train_vocabulary(token_corpus, v_size):
    print("--------- d2v model training ---------")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=v_size, min_count=2, epochs=40)
    model.build_vocab(token_corpus)
    print("dimesion:{}, corpus_count: {}, epochs: {}, windows_size: {}".format(v_size, model.corpus_count, model.epochs, model.window))
    model.train(token_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model

def learning_infer(model, data):
    "check if data is from train or test and infer_vector"

    X_inferVector = []
    for i in data:
        if isinstance(i, list):
            # for testing data sets, because it contents only tokenized words
            X_inferVector.append(model.infer_vector(i))
        else:
            X_inferVector.append(model.infer_vector(i.words))
    return X_inferVector

def get_vector(train_data_loc, test_data_loc, v_size):
    # for train data
    df_train = pd.read_csv(train_data_loc)

    # text features either
    X_train = np.array(df_train[df_train.columns[:-1]].values.tolist())
    y_train = df_train[[df_train.columns[-1]]]
    train_tokenized_corpus = list(read_corpus(X_train))

    print("Doc2Vec model training on: ", str(train_data_loc))
    model = train_vocabulary(train_tokenized_corpus, v_size)

    # getting train features
    X_train_df = pd.DataFrame(learning_infer(model, train_tokenized_corpus))
    result_train = pd.concat([X_train_df, y_train], axis=1, sort=False)

    # for test data
    df_test = pd.read_csv(test_data_loc)

    X_test = np.array(df_test[df_test.columns[:-1]].values.tolist())
    y_test = df_test[[df_test.columns[-1]]]
    test_tokenized_corpus = list(read_corpus(X_test, tokens_only=True))

    # getting test features
    X_test_df = pd.DataFrame(learning_infer(model, test_tokenized_corpus))
    result_test = pd.concat([X_test_df, y_test], axis=1, sort=False)

    # return this to save in theirs particular folders
    return model, result_train, result_test

