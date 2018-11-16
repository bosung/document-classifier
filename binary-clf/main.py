import argparse
from random import shuffle
from sklearn.model_selection import KFold

from config import Config
from prepro import prepro, csv_parser
from model.svm import SVM
from model.mlp import MLPClassifier


def evaluate(docs):
    shuffle(docs)

    kf = KFold(n_splits=int(config.split))
    kf.get_n_splits(docs)

    total_accuracy = 0
    total_f1 = 0
    for train_index, test_index in kf.split(docs):
        model = MLPClassifier()
        # model = SVM()
        model.train(train_index, docs)
        f1, accuracy = model.test(test_index, docs)
        total_f1 += f1
        total_accuracy += accuracy

    print("average f1: %.3f, accuracy: %.3f" % (
            total_f1/int(config.split),
            total_accuracy/int(config.split)))


def eval_batch(docs):
    model = SVM(prob=True)
    # model = MLPClassifier()
    model.train_batch(docs, save=True)


def test(docs):
    model = SVM(prob=True)
    model.test_batch(docs, pretrain=True)
    # model = MLPClassifier()
    # model.test_batch(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug')
    args = parser.parse_args()

    config = Config()

    if args.mode == 'eval':
        # k-fold cross validation
        documents = prepro(config)
        evaluate(documents)
    elif args.mode == 'eval-batch':
        # train on entire data
        documents = prepro(config)
        eval_batch(documents)
    elif args.mode == 'test':
        documents = csv_parser(config)
        test(documents)
    elif args.mode == 'prepro':
        prepro(config)
    else:
        print("Unknown mode")
        exit(0)
