import argparse
import glob

from konlpy.tag import Komoran
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

import preprocess

komoran = Komoran()
word_dict = {}
word_embed = {}
word_index_dict = {}


def get_all_docs(data_path):
    # get only .txt files
    files = glob.glob(data_path + '/*.txt')

    docs = []
    for f in files:
        docs += preprocess.text2dics(f)

    return docs


def bag_of_words(documents):
    global word_idx
    word_idx = 0
    # TODO: need to fix with more efficient way
    for doc in documents:
        text = doc['text']
        for sentence in text:
            nouns = komoran.nouns(sentence)
            for n in nouns:
                if n not in word_dict:
                    word_dict[n] = 1
                    word_index_dict[n] = word_idx
                    word_idx += 1
                else:
                    word_dict[n] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='for data directory')
    parser.add_argument('--file', help='for 1 data file')

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
    bag_of_words(documents)
    print("total words: {} {}".format(len(word_dict), word_idx))

