#first build vocabulary with gensim later separate and save inside document embeddings

# apply document embedding with option of types

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import multiprocessing
import nltk
from sklearn import utils
import pickle
import os
import sys
cores = multiprocessing.cpu_count()


# prepare directory to store data
def prepare_dir(location):
    datasetName = location.split("/")[-1].split(".")[0]
    storeLocation = "data/embeddings/"+datasetName

    train_vocabulary = storeLocation + "/train_vocabulary/"
    vector_vocabulary = storeLocation + "/vector_vocabulary/"

    try:
        # Create storing Directory
        os.mkdir(storeLocation)
        os.mkdir(train_vocabulary)
        os.mkdir(vector_vocabulary)

    except FileExistsError:
        print("Directory already exists")

    return datasetName, train_vocabulary, vector_vocabulary

#word tokenization
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def tagged_data(fileLocation):
    # reading as pandas dataframe
    df = pd.read_csv(fileLocation)

    # Get data frame and tag its message with label, feature:text, label:category
    df_tagged = df.apply(
        lambda r: TaggedDocument(words=tokenize_text(r[0]), tags=[r[-1]]), axis=1)
    print(df_tagged.values[30])

    # we can save train_tagged and test_tagged
    # now call for distributed bag of word and distributed memory for modeling text
    # build vocabular
    model_dbow = dbow(df_tagged)
    model_dm = dm(df_tagged)

    # getting location to store
    datasetName, train_vocabulary, vector_vocabulary = prepare_dir(fileLocation)

    # save build vocabulary
    model_dbow.save(train_vocabulary + datasetName +"_dbow")
    model_dm.save(train_vocabulary + datasetName + "_dm")

    # train vocabulary
    y_dbow, X_dbow = vec_for_learning(model_dbow, df_tagged)
    y_dm, X_dm = vec_for_learning(model_dm, df_tagged)

    # saving into csv file
    df_dbow = save_csv(y_dbow, X_dbow)
    df_dm = save_csv(y_dm, X_dm)

    df_dbow.to_csv("data/embeddings/"+datasetName+"_dbow.csv", index=None)
    df_dm.to_csv("data/embeddings/"+datasetName+"_dm.csv", index=None)

    # save trained vocabulary
    pickle.dump(y_dbow, open(vector_vocabulary + datasetName + "_y_dbow", 'wb'))
    pickle.dump(X_dbow, open(vector_vocabulary + datasetName + "_X_dbow", 'wb'))
    pickle.dump(y_dm, open(vector_vocabulary + datasetName + "_y_dm", 'wb'))
    pickle.dump(X_dm, open(vector_vocabulary + datasetName + "_X_dm", 'wb'))


def save_csv(y_train, X_train):
    # creating in to dataframe
    df_label = pd.DataFrame(list(y_train), columns=['label'], index=None)
    df_feature = pd.DataFrame(list(X_train), columns=[x for x in range(len(X_train[0]))], index=None)

    # dfs concatination
    return pd.concat([df_feature, df_label], axis=1, sort=False)

# we can change different parameters, but now its okey
# dm = ({1,0}, optional) – Defines the training algorithm.
# If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
# vector_size = (int, optional) – Dimensionality of the feature vectors.
# negative =  (int, optional) – If > 0, negative sampling will be used,
# the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
# If set to 0, no negative sampling is used.
# hs =  ({1,0}, optional) – If 1, hierarchical softmax will be used for model training.
# If set to 0, and negative is non-zero, negative sampling will be used.
# min_count = (int, optional) – Ignores all words with total frequency lower than this.
# sample = (float, optional) – The threshold for configuring
# which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
# workers =  (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
# window = (int, optional) – The maximum distance between the current and predicted word within a sentence
# alph = (float, optional) – The initial learning rate.
# min_alpha = (float, optional) – Learning rate will linearly drop to min_alpha as training progresses

def dbow(df_tagged):
    # build vocabulary for distributed bag of word model
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(df_tagged.values)])

    train_model = train_vocabulary(model_dbow, df_tagged)

    return train_model

def dm(df_tagged):
    # build vocabulary for distributed memory model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=cores, alpha=0.065,
                        min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(df_tagged.values)])

    train_model = train_vocabulary(model_dmm, df_tagged)

    train_model.docvecs

    return train_model

def train_vocabulary(model, df_tagged):
    for epoch in range(30):
        model.train(utils.shuffle([x for x in tqdm(df_tagged.values)]), total_examples=len(df_tagged.values),
                         epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model

def vec_for_learning(model, df_tagged):
    sents = df_tagged.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

if __name__ == '__main__':
    #datasetLocation = "data/20newsgroup.csv"
	dataset_location = str(sys.argv[-1])
	if os.path.exists(dataset_location):
		tagged_data(dataset_location)
	else:
		print("please provide correct file location, {} file doesn't exists".format(dataset_location))
    