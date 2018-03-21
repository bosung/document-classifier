import argparse
import glob
import numpy as np

from model import Features
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

import preprocess


def get_all_docs(data_path):
    # get only .txt files
    files = glob.glob(data_path + '/*.txt')

    docs = list()
    for f in files:
        docs += preprocess.text2dics(f)

    return docs


def get_category_id(document):
    global category_dict
    category = document.category.split("/")[1].strip()
    return category_dict[category]['cat_id']


def make_xy_params(documents, features):
    """make X, y parameters for sklearn.svm
        X: feature vector matrix
        y: class vector

    Example:
        X = np.array([[-1, -1], [1, 1], [0, -1], [1, -1]])
        y = np.array([1, 1, 2, 2])
    """

    X, y = list(), list()
    for doc in documents:
        category_id = get_category_id(doc)
        X.append(features.get_feature_vector(doc))
        y.append(category_id)

    return np.array(X), np.array(y)


def train(train_index, documents, features):
    train_data = [documents[i] for i in train_index]
    clf = LinearSVC(random_state=0)
    X, y = make_xy_params(train_data, features)
    clf.fit(X, y)
    return clf


def test(test_index, model, documents, features):
    total_size = len(test_index)
    ans_cnt = 0
    for i in test_index:
        document = documents[i]
        category = get_category_id(document)
        prediction = model.predict([features.get_feature_vector(document)])
        if prediction[0] == category:
            ans_cnt += 1

    accuracy = ans_cnt/total_size*100
    print("accuracy: %.3f" % accuracy)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', help='K fold split', default=10)
    parser.add_argument('--dir', help='for data directory')
    parser.add_argument('--file', help='for 1 data file')
    parser.add_argument('--category_file', help='for category file path', \
            default='data/hkib.categories')

    args = parser.parse_args()

    if args.dir:
        data_path = args.dir
        documents = get_all_docs(data_path)
    elif args.file:
        documents = preprocess.text2dics(args.file)
    else:
        print("Need a data/file path")
        exit()

    # get category information
    global category_dict
    category_dict = preprocess.get_category_dict(args.category_file)

    print("total document size: {}".format(len(documents)))

    features = Features(documents)
    print("total words: {} {}".format(len(features.word_index_dict.keys()), features.word_size))

    ################
    # train and test
    # ##############

    shuffle(documents)

    kf = KFold(n_splits=int(args.split))
    kf.get_n_splits(documents)

    total_accuracy = 0
    for train_index, test_index in kf.split(documents):
        model = train(train_index, documents, features)
        total_accuracy += test(test_index, model, documents, features)

    print("average accuracy: %.3f" % (total_accuracy/int(args.split)))

