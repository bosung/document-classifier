import math
import numpy as np

from konlpy.tag import Komoran


class Document():

    def __init__(self, docid, category, title, text):
        self.docid = docid
        self.category = category
        self.title = title
        self.text = text
        self.words = list()
        self.bow = {}


class Features():

    def __init__(self, documents):
        self.komoran = Komoran()
        self.documents = documents
        self.doc_size = len(documents)
        self.word_size = 0
        self.word_index_dict = {}
        self.word_count_dict = {}
        # for tf-idf, # of docs where the term t appears
        self.term_doc_dict = {}
        self.word_counting()

    def trim_sentence(self, sentence):
        return self.komoran.nouns(sentence)

    def word_counting(self):
        # TODO: need to fix with more efficient way
        for doc in self.documents:
            text = doc.text
            for sentence in text:
                doc.words = self.trim_sentence(sentence)
                for w in doc.words:
                    if w not in self.term_doc_dict:
                        self.term_doc_dict[w] = {}
                    self.term_doc_dict[w][doc.docid] = 1

                    if w not in self.word_count_dict:
                        self.word_count_dict[w] = 1
                        self.word_index_dict[w] = self.word_size
                        self.word_size += 1
                    else:
                        self.word_count_dict[w] += 1

                    if w not in doc.bow:
                        doc.bow[w] = 1
                    else:
                        doc.bow[w] += 1

    def bag_of_words(self, document):
        vector = np.zeros(self.word_size)
        for w in list(document.bow.keys()):
            w_idx = self.word_index_dict[w]
            vector[w_idx] = document.bow[w]

        return vector

    def tf_idf(self, document):
        """make tf-idf feature vector

          tf(t, d) = log{(f, d) + 1}
          idf(t, D) = log{|D|/(1+|{d:t}|)}
          TF-IDF(t, d) = TF(t, d) * IDF(t, D)
        """
        vector = np.zeros(self.word_size)
        for w in list(document.bow.keys()):
            w_idx = self.word_index_dict[w]

            tf = math.log(document.bow[w]+1)

            ds = len(list(self.term_doc_dict[w].keys()))
            idf = math.log(self.doc_size/(1+ds))

            vector[w_idx] = tf * idf
        return vector

    def get_feature_vector(self, document):
        #return np.array(self.bag_of_words(document))
        return np.array(self.tf_idf(document))

