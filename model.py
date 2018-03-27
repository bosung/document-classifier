import math
import numpy as np
import operator

from konlpy.tag import Komoran


class Document():

    def __init__(self, docid, category, title, text):
        self.docid = docid
        self.category = category
        self.root_category = category.split("/")[1].strip()
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
        self.word_amount = 0
        self.word_index_dict = {}
        self.word_count_dict = {}

        # for tf-idf, # of docs where the term t appears
        self.term_doc_dict = {}

        # for chi square
        # category_dict['health'] = total # of terms in 'health' category
        self.category_dict = {}
        # term_dict['water'] = total # of 'water' in all documents
        self.term_dict = {}
        # category_term_dict['health']['water'] = total # of word 'water' in 'health' category
        self.category_term_dict = {}
        self.word_counting()

    def trim_sentence(self, sentence):
        return self.komoran.nouns(sentence)

    def word_counting(self):
        # TODO: need to fix with more efficient way
        for doc in self.documents:
            if doc.root_category not in self.category_dict:
                self.category_dict[doc.root_category] = 1
                self.category_term_dict[doc.root_category] = {}
            else:
                self.category_dict[doc.root_category] += 1

            text = doc.text
            for sentence in text:
                doc.words = self.trim_sentence(sentence)
                for w in doc.words:
                    self.word_amount += 1

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
            if doc.root_category not in self.category_term_dict:
                self.category_term_dict[doc.root_category] = {}

            for w in doc.bow:
                if w not in self.category_term_dict[doc.root_category]:
                    self.category_term_dict[doc.root_category][w] = doc.bow[w]
                else:
                    self.category_term_dict[doc.root_category][w] += doc.bow[w]

                if w not in self.term_dict:
                    self.term_dict[w] = 0
                    self.term_dict[w] += doc.bow[w]
                else:
                    self.term_dict[w] += doc.bow[w]

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

    def chi_square(self, document):
        """make chi square statistic feature vector
        term-goodness measure is defined to be:

            x^2(t, c) = N*(AD-CB)^2 / {(A+C)*(B+D)*(A+B)*(C+D)}

        where,
            N: the total number of documents
            A: the number of times t and c co-occur
            B: the number of times t occurs without c
            C: the number of times c occurs without t
            D: them number of times neither c nor t occurs
        """
        chi_stat = {}
        N = self.doc_size
        for w in list(document.bow.keys()):
            A = self.category_term_dict[document.root_category][w]
            B = self.category_dict[document.root_category] - A
            C = self.term_dict[w] - A
            D = self.word_amount - (A+B+C)

            stat= N*(A*D-C*B)*(A*D-C*B) / ((A+C)*(B+D)*(A+B)*(C+D))
            chi_stat[w] = stat

        vector = np.zeros(self.word_size)

        # sort and get top 60%
        sorted_stat = sorted(chi_stat.items(), key=operator.itemgetter(1), reverse=True)
        threshold = int(round(len(sorted_stat) * 0.6))
        for word, value in sorted_stat[:threshold]:
            w_idx = self.word_index_dict[word]
            vector[w_idx] = value
        return vector

    def get_feature_vector(self, document):
        #return np.array(self.bag_of_words(document))
        #return np.array(self.tf_idf(document))
        return np.array(self.chi_square(document))

