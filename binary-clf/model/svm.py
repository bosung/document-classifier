import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
import pickle


class SVM:

    def __init__(self, prob=False):
        # self.clf = LinearSVC(random_state=0)
        self.clf = NuSVC(gamma='scale', nu=0.22, probability=prob)

    @staticmethod
    def make_xy_params(documents):
        """make X, y parameters for sklearn.svm
            X: feature vector matrix
            y: class vector
        Example:
            X = np.array([[-1, -1], [1, 1], [0, -1], [1, -1]])
            y = np.array([1, 1, 2, 2])
        """

        X, y = list(), list()
        for doc in documents:
            category_id = 1 if doc["type"] == "mal" else 0
            X.append(np.array(doc["features"]))
            y.append(category_id)

        return np.array(X), np.array(y)

    def train(self, train_index, documents):
        train_data = [documents[i] for i in train_index]
        X, y = self.make_xy_params(train_data)
        self.clf.fit(X, y)

    def test(self, test_index, documents, model=None):
        if model is None:
            model = self.clf

        total_size = len(test_index)

        # F1-score
        tp = 0  # true positive
        fp = 0  # false positive
        fn = 0  # false negative

        ans_cnt = 0
        mal_total = 0
        for i in test_index:
            doc = documents[i]
            if doc["type"] == "mal":
                category = 1
                mal_total += 1
            else:
                category = 0

            prediction = model.predict([np.array(doc["features"])])
            if prediction[0] == category:
                # correct
                ans_cnt += 1
                if category == 1:
                    tp += 1
            else:
                # incorrect
                if category == 0 and prediction[0] == 1:
                    fp += 1

        fn = mal_total - tp
        precision = tp/(tp+fp)*100
        recall = tp/(tp+fn)*100
        accuracy = ans_cnt/total_size*100
        f1 = 2*precision*recall/(precision+recall)

        print("P: %.3f, R: %.3f, f1 = %.3f, accuracy: %.3f" % (
            precision, recall, f1, accuracy))

        return f1, accuracy

    def train_batch(self, documents, save=True):
        X, y = self.make_xy_params(documents)
        self.clf.fit(X, y)
        if save is True:
            pickle.dump(self.clf, open("svm-model", "wb"))

    @staticmethod
    def load_model(filename):
        return pickle.load(open(filename, "rb"))

    def test_batch(self, documents, pretrain=False):
        if pretrain is True:
            model = self.load_model("svm-model")
        else:
            model = self.clf

        total_size = len(documents)
        ans_cnt = 0
        for doc in documents:
            category = 1 if doc["type"] == "mal" else 0
            if "features" in doc:
                temp = [float(f) for f in doc["features"]]
            else:
                temp = list()
                temp.append(float(doc["f1"]))
                temp.append(float(doc["f2"]))
                temp.append(float(doc["f3"]))
                temp.append(float(doc["f4"]))
                temp.append(float(doc["f5"]))
                temp.append(float(doc["f6"]))
                temp.append(float(doc["f7"]))
                temp.append(float(doc["f8"]))
            prediction = model.predict([np.array(temp)])
            print(category, prediction[0], model.predict_proba([np.array(temp)]))
            if prediction[0] == category:
                ans_cnt += 1

        accuracy = ans_cnt/total_size*100
        print("accuracy: %.3f" % accuracy)
        return accuracy
