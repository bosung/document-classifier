import argparse
import glob
import numpy as np

from konlpy.tag import Komoran
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

import preprocess

komoran = Komoran()
word_index_dict = {}


def get_all_docs(data_path):
    # get only .txt files
    files = glob.glob(data_path + '/*.txt')

    docs = []
    for f in files:
        docs += preprocess.text2dics(f)

    return docs


def word_counting(documents):
    global word_idx
    word_idx = 0
    # TODO: need to fix with more efficient way
    for doc in documents:
        text = doc['text']
        doc['bow'] = {}
        for sentence in text:
            nouns = komoran.nouns(sentence)
            for n in nouns:
                if n not in word_index_dict:
                    word_index_dict[n] = word_idx
                    word_idx += 1

                if n not in doc['bow']:
                    doc['bow'][n] = 1
                else:
                    doc['bow'][n] += 1


def bag_of_words(document):
    global word_idx
    vector = np.zeros(word_idx)
    for w in list(document['bow'].keys()):
        w_idx = word_index_dict[w]
        vector[w_idx] = document['bow'][w]

    return vector


def get_category_id(document):
    global category_dict
    category = document['cat03'].split("/")[-1].strip()
    return category_dict[category]['cat_id']


def make_xy_params(documents):
    """make X, y parameters for sklearn.svm
        X: feature vector matrix
        y: class vector

    Example:
        X = np.array([[-1, -1], [1, 1], [0, -1], [1, -1]])
        y = np.array([1, 1, 2, 2])
    """

    X = list()
    y = list()
    for doc in documents:
        category_id = get_category_id(doc)
        X.append(np.array(bag_of_words(doc)))
        y.append(category_id)

    return np.array(X), np.array(y)


def train(train_index, documents):
    train_data = [documents[i] for i in train_index]
    clf = LinearSVC(random_state=0)
    X, y = make_xy_params(train_data)
    clf.fit(X, y)
    return clf


def test(test_index, model, documents):
    total_size = len(test_index)
    ans_cnt = 0
    for i in test_index:
        document = documents[i]
        category = get_category_id(document)
        prediction = model.predict([bag_of_words(document)])

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

    print("total document size: {}".format(len(documents)))

    global word_idx
    word_counting(documents)
    print("total words: {} {}".format(len(word_index_dict), word_idx))

    global category_dict
    category_dict = preprocess.get_category_dict(args.cat_file)

    shuffle(documents)

    kf = KFold(n_splits=int(args.split))
    kf.get_n_splits(documents)

    total_accuracy = 0
    for train_index, test_index in kf.split(documents):
        model = train(train_index, documents)
        total_accuracy += test(test_index, model, documents)

    print("average accuracy: %.3f" % total_accuracy/args.split)
