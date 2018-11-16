import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set()


class MLPClassifier:

    def __init__(self, input_size=18, hidden_size=300, epoch=40, batch_size=40):
        self.model = MLP(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, train_index, docs, hitmap=False, save=False):
        train_data = []
        for i in train_index:
            doc = docs[i]
            temp = [float(f) for f in doc["features"][:self.input_size]]
            train_data.append([torch.tensor(temp, dtype=torch.float).to(device),
                               torch.tensor(1 if doc["type"] == "mal" else 0, dtype=torch.long).to(device)])

        train_loader = data.DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch):
            self.criterion = nn.NLLLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters())
            local_loss = 0

            for _iter, (batch_input, batch_target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(batch_input)
                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()

                local_loss += loss.item()

                # if (_iter + 1) % 40 == 0:
                #    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                #        epoch + 1, self.epoch, _iter + 1, len(train_data) // self.batch_size, local_loss))

        if hitmap is True:
            sns.heatmap(self.model.first.weight.data)
            plt.show()

        if save is True:
            torch.save(self.model, "mlp.model")

    def train_batch(self, docs, hitmap=False, save=False):
        train_data = []
        for doc in docs:
            temp = [float(f) for f in doc["features"][:self.input_size]]
            train_data.append([torch.tensor(temp, dtype=torch.float).to(device),
                               torch.tensor(1 if doc["type"] == "mal" else 0, dtype=torch.long).to(device)])

        train_loader = data.DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch):
            self.criterion = nn.NLLLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters())
            local_loss = 0

            for _iter, (batch_input, batch_target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(batch_input)
                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()

                local_loss += loss.item()

        if hitmap is True:
            sns.heatmap(self.model.first.weight.data)
            plt.show()

        if save is True:
            torch.save(self.model.state_dict(), "mlp.model")

    def test(self, test_index, docs):
        answer_cnt = 0
        mal_cnt = 0
        normal_cnt = 0
        total = len(test_index)

        # F1-score
        tp = 0  # true positive
        fp = 0  # false positive
        fn = 0  # false negative

        for i in test_index:
            doc = docs[i]
            temp = [float(f) for f in doc["features"][:self.input_size]]

            test_input = torch.tensor(temp, dtype=torch.float).to(device)
            if doc["type"] == "mal":
                answer = 1
                mal_cnt += 1
            else:
                answer = 0
                normal_cnt += 1

            outputs = self.model(test_input.unsqueeze(0))
            predict = int(torch.argmax(outputs))
            if predict == answer:
                if answer == 1:
                    tp += 1
                answer_cnt += 1
            else:
                if answer == 0 and predict == 1:
                    fp += 1

        fn = mal_cnt - tp
        # print(mal_cnt, answer_cnt, tp, fp)
        precision = tp/(tp+fp)*100
        recall = tp/(tp+fn)*100
        accuracy = answer_cnt/total*100
        f1 = 2*precision*recall/(precision+recall)

        print("P: %.3f, R: %.3f, f1 = %.3f, accuracy: %.3f" % (
            precision, recall, f1, accuracy))

        return f1, accuracy

    def test_batch(self, docs):
        model = MLP(self.input_size, self.hidden_size)
        model.load_state_dict(torch.load("mlp.model"))

        total = len(docs)
        answer_cnt = 0
        mal_cnt = 0
        normal_cnt = 0

        # F1-score
        tp = 0  # true positive
        fp = 0  # false positive
        fn = 0  # false negative

        for doc in docs:
            temp = [float(f) for f in doc["features"][:self.input_size]]
            test_input = torch.tensor(temp, dtype=torch.float).to(device)
            if doc["type"] == "mal":
                answer = 1
                mal_cnt += 1
            else:
                answer = 0
                normal_cnt += 1

            outputs = model(test_input.unsqueeze(0))
            predict = int(torch.argmax(outputs, dim=1))
            print(answer, predict, outputs.data)
            if predict == answer:
                answer_cnt += 1
                if answer == 1:
                    tp += 1
            else:
                if answer == 0 and predict == 1:
                    fp += 1

        fn = mal_cnt - tp
        # print(mal_cnt, answer_cnt, tp, fp)
        precision = 0 if tp+fp == 0 else tp/(tp+fp)*100
        recall = tp/(tp+fn)*100
        accuracy = answer_cnt/total*100
        f1 = 0 if precision+recall == 0 else 2*precision*recall/(precision+recall)

        print("P: %.3f, R: %.3f, f1 = %.3f, accuracy: %.3f" % (
            precision, recall, f1, accuracy))
        return accuracy


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.first = nn.Linear(input_size, hidden_size).to(device)
        self.last = nn.Linear(hidden_size, 2).to(device)
        self.temp = nn.Linear(input_size, 2).to(device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, _input):
        out1 = self.first(_input)
        out2 = F.relu(out1)
        out3 = self.last(out2)
        # print(out3.data)
        return self.softmax(out3)
        # return self.temp(_input)




