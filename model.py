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

        # for chi square feature
        # chi_a = # of terms appeared in this category
        # chi_b = # of terms not appeared in category
        self.chi_a = {}
        self.chi_b = 0
        self.chi_c = 0
        self.chi_d = 0


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
        # for chi square
        # category_dict['health'] = total # of terms
        self.category_dict = {}
        # term_category_dict['health']
        self.term_category_dict = {}
        self.category_term_dict = {}
        self.word_counting()

    def trim_sentence(self, sentence):
        return self.komoran.nouns(sentence)

    def word_counting(self):
        # TODO: need to fix with more efficient way
        for doc in self.documents:
            if doc.category not in self.category_dict:
                self.category_dict[doc.category] = 1
                self.category_term_dict[doc.category] = {}
            else:
                self.category_dict[doc.category] += 1

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

            # chi square
            for w in doc.bow:
                if w not in self.category_term_dict:
                    self.category_term_dict[doc.category] = {}
                    self.category_term_dict[doc.category][w] = doc.bow[w]
                elif w not in self.category_term_dict[doc.category]:
                    self.category_term_dict[doc.category][w] = 0
                    self.category_term_dict[doc.category][w] += doc.bow[w]
                else:
                    self.category_term_dict[doc.category][w] += doc.bow[w]

                if w not in self.term_category_dict:
                    self.term_category_dict[w] = {}
                    self.term_category_dict[w][doc.category] = doc.bow[w]
                elif doc.category not in self.term_category_dict[w]:
                    self.term_category_dict[w][doc.category] = 0
                    self.term_category_dict[w][doc.category] += doc.bow[w]
                else:
                    self.term_category_dict[w][doc.category] += doc.bow[w]

    def bag_of_words(self, document):
        vector = np.zeros(self.word_size)
        for w in list(document.bow.keys()):
            w_idx = self.word_index_dict[w]
            vector[w_idx] = document.bow[w]

        return vector

    def tf_idf(self, document):
        """make tf-idf feature vector
        this is logarithmically scaled definition

          tf(t, d) = log{1 + (f, d)}
          idf(t, D) = log{|D|/(1 + |{d:t}|)}
          tf-idf(t, d) = tf(t, d) * idf(t, D)
        """
        vector = np.zeros(self.word_size)
        for w in list(document.bow.keys()):
            w_idx = self.word_index_dict[w]

            tf = math.log(1 + document.bow[w])

            ds = len(list(self.term_doc_dict[w].keys()))
            idf = math.log(self.doc_size/(1 + ds))

            vector[w_idx] = tf * idf
        return vector

    #def chi_square(self, document):

    def get_feature_vector(self, document):
        #return np.array(self.bag_of_words(document))
        return np.array(self.tf_idf(document))

