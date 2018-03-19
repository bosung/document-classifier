import numpy as np


class Document():

    def __init__(self, docid, category, title, text):
        self.docid = docid
        self.category = category
        self.title = title
        self.text = text
        self.words = list()

        # features
        self.bow = {}
        self.tf_idf = {}


class Features():

    def __init__(self, word_size, word_index_dict):
        self.word_size = word_size
        self.word_index_dict = word_index_dict

    def bag_of_words(self, document):
        vector = np.zeros(self.word_size)
        for w in list(document.bow.keys()):
            w_idx = self.word_index_dict[w]
            vector[w_idx] = document.bow[w]

        return vector

    def tf_idf(self, document):
        print(1)

    def get_feature_vector(self, document):
        return np.array(self.bag_of_words(document))

